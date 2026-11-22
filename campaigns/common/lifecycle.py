"""Safe stops and verified GPU release between campaign worker processes.

Workers own every CUDA context; runners only launch them. After each worker
the runner confirms that no process of this user still holds the job's GPU
before starting the next one, so a stuck or orphaned worker cannot leak device
memory into later measurements.
"""
from __future__ import annotations

import contextlib
import csv
from datetime import datetime
import os
from pathlib import Path
import re
import signal
import subprocess
import time

# SIGUSR1 is the early warning requested with ``sbatch --signal=USR1@<seconds>``.
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGUSR1)
REQUEUE_SIGNALS = frozenset({'SIGUSR1'})


class CampaignStop(KeyboardInterrupt):
    """A requested stop; existing KeyboardInterrupt cleanup paths still run."""

    def __init__(self, signal_name):
        super().__init__(signal_name)
        self.signal_name = signal_name


def stop_signal_name(returncode):
    """Name of the stop signal that killed a worker, or None.

    Slurm signals every process in the job, so a worker can die from scancel or
    the time limit before the runner's handler runs; that is an interruption to
    resume, not a benchmark failure.
    """
    if returncode is not None and returncode < 0 and -returncode in {int(number) for number in STOP_SIGNALS}:
        return signal.Signals(-returncode).name
    return None


def log(message):
    """Print one timestamped line immediately, so Slurm logs show progress."""
    print(f'[{datetime.now().astimezone().isoformat(timespec="seconds")}] {message}', flush=True)


@contextlib.contextmanager
def stop_on_signals(signals=STOP_SIGNALS):
    """Raise CampaignStop once in the main thread; later signals cannot interrupt cleanup."""
    received = []

    def handler(number, _frame):
        name = signal.Signals(number).name
        if received:
            log(f'received {name} while stopping for {received[0]}; finishing cleanup')
            return
        received.append(name)
        log(f'received {name}; stopping the current worker and saving state')
        raise CampaignStop(name)

    previous = {number: signal.signal(number, handler) for number in signals}
    try:
        yield received
    finally:
        for number, old in previous.items():
            signal.signal(number, old)


def terminate_group(process, grace=10.):
    """Stop a ``start_new_session`` worker and every process left in its group."""
    outcome = 'exited'
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=grace)
            outcome = 'terminated'
        except subprocess.TimeoutExpired:
            outcome = 'killed'
    # Grandchildren survive their leader; sweep the group so none keep a context.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)
    if process.poll() is None:
        process.wait()
    return outcome


def _lineage():
    """This process and its ancestors, which must never be killed here."""
    pids, pid = set(), os.getpid()
    while pid > 1 and pid not in pids:
        pids.add(pid)
        try:
            pid = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return pids


def _nvidia_smi(query):
    result = subprocess.run(['nvidia-smi', query, '--format=csv,noheader,nounits'],
                            text=True, capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f'nvidia-smi {query} exited {result.returncode}: {result.stderr.strip()}')
    return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))


def _integer(value):
    """VALUE as an int, or None when nvidia-smi printed no number (for example '[N/A]')."""
    return int(value) if value.strip().isdigit() else None


def device_state():
    """Memory and compute processes on the GPUs visible to this Slurm job.

    A field nvidia-smi cannot read is printed as '[N/A]' and stored as None. A GPU
    in a bad state prints it even for process ids.
    """
    memory = [dict(index=int(index), used_mib=_integer(used), total_mib=_integer(total))
              for index, used, total in _nvidia_smi('--query-gpu=index,memory.used,memory.total')]
    processes = [dict(pid=_integer(pid), used_mib=_integer(used))
                 for pid, used in _nvidia_smi('--query-compute-apps=pid,used_memory')]
    return dict(memory=memory, processes=processes)


def _ours(pid):
    """Whether a listed GPU process belongs to this Slurm job.

    Returns 'gone' for exited processes the driver still lists, 'job' for live
    processes in this job's cgroup, and None otherwise. Outside Slurm nothing is
    ever claimed, so another job's processes are never touched.
    """
    job = os.environ.get('SLURM_JOB_ID')
    try:
        groups = Path(f'/proc/{pid}/cgroup').read_text()
    except FileNotFoundError:
        return 'gone'
    except OSError:
        return None
    return 'job' if job and re.search(rf'/job_{job}(/|$)', groups, re.MULTILINE) else None


def release_device(timeout=60.):
    """Kill this job's leftover GPU processes and wait until the driver lists none.

    Returns a JSON record. ``released`` is None when nvidia-smi is unavailable,
    so host-only runs are recorded rather than failed.
    """
    started, keep, killed = time.monotonic(), _lineage(), []
    while True:
        try:
            state = device_state()
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            return dict(released=None, error=f'{type(error).__name__}: {error}')
        unreadable = sum(row['pid'] is None for row in state['processes'])
        if unreadable:
            # Without a pid a process cannot be assigned to a job, so release can be
            # neither confirmed nor refused.
            return dict(released=None, memory=state['memory'], killed_pids=killed,
                        error=f'nvidia-smi listed {unreadable} compute process(es) without a readable pid')
        leftovers = [dict(row, owner=owner) for row in state['processes']
                     if row['pid'] not in keep and (owner := _ours(row['pid']))]
        record = dict(memory=state['memory'], killed_pids=killed, seconds=round(time.monotonic() - started, 3))
        if not leftovers:
            return dict(record, released=True)
        if time.monotonic() - started > timeout:
            return dict(record, released=False, leftover=leftovers)
        for row in leftovers:
            if row['owner'] == 'job' and row['pid'] not in killed:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(row['pid'], signal.SIGKILL)
                killed.append(row['pid'])
        time.sleep(1)


def describe_release(record):
    """One log fragment for a release_device record."""
    if record.get('released') is None:
        return f'device check unavailable ({record.get("error")})'
    used = ', '.join(f'gpu{row["index"]} {row["used_mib"]}/{row["total_mib"]} MiB' for row in record['memory'])
    killed = f'; killed leftover pids {record["killed_pids"]}' if record['killed_pids'] else ''
    state = 'released' if record['released'] else f'NOT released, still held by {record["leftover"]}'
    return f'device {state} ({used}){killed}'


def tail(path, lines=20):
    """Last lines of a worker log, for failures reported in the Slurm log."""
    try:
        return '\n'.join(Path(path).read_text(errors='replace').splitlines()[-lines:])
    except OSError:
        return ''


def tee(stream, path):
    """Copy a child's merged output line by line to our stdout and append it to PATH.

    Target of a reader thread: the live log keeps every line, and PATH keeps the child's own
    output for diagnosing how it ended.
    """
    with open(path, 'a', errors='replace') as copy:
        for line in stream:
            print(line, end='', flush=True)
            copy.write(line)
            copy.flush()


def amgx_abort(text):
    """Reason of the last fatal AMGX error printed in TEXT, or None.

    AMGX reports setups it cannot run (for example a BSR block size its smoother has no kernels
    for) by printing 'AMGX ERROR: ...' and exiting the process, so Python never sees an exception.
    Such an abort is deterministic for the configuration and block size.
    """
    messages = [m.strip() for m in re.findall(r'^AMGX ERROR: (.*)$', text, re.MULTILINE)]
    messages = [m for m in messages if not m.startswith('file ')]
    return messages[-1] if messages else None
