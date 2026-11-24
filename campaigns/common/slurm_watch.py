"""Wait for guarded submission and one Slurm job without an interactive monitor."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import getpass
import json
from pathlib import Path
import subprocess
import time


def _write(path, record):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def watch_slurm(submit, state_path, *, job_id=None, poll_seconds=120., timeout_seconds=172800.):
    """Wait for a guarded submission, then return one terminal job record.

    ``submit`` returns the job number and must apply the shared GPU guard itself.
    Only the guard's ``Shared GPU ceiling:`` refusal is retried. Scheduler errors,
    an expired timeout, or an unsuccessful job produce a terminal failure record.
    The state file retains the job number so a restarted watcher does not submit
    the job twice. One lock prevents two watchers from using the same state file.
    Nothing is printed while waiting; callers may print the returned record once.
    """
    if poll_seconds <= 0 or timeout_seconds <= 0:
        raise ValueError("Watcher interval and timeout must be positive.")
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = json.loads(path.read_text()) if path.exists() else {}
        if record.get("terminal"):
            return record
        record.setdefault("started", datetime.now(timezone.utc).isoformat())
        if job_id is not None:
            record.setdefault("job_id", str(job_id).split(";")[0])
        started = datetime.fromisoformat(record["started"]).timestamp()
        deadline = started + timeout_seconds
        missing_accounting = 0
        try:
            while time.time() < deadline:
                if not record.get("job_id"):
                    try:
                        record["job_id"] = str(submit()).strip().split(";")[0]
                        if not record["job_id"].isdigit():
                            raise ValueError(f"Submission returned an invalid job number: {record['job_id']!r}")
                        record["status"] = "submitted"
                    except RuntimeError as error:
                        if not str(error).startswith("Shared GPU ceiling:"):
                            raise
                        record.update(status="waiting_for_gpu", last_refusal=str(error))
                        _write(path, record)
                        time.sleep(min(poll_seconds, max(0., deadline - time.time())))
                        continue
                    _write(path, record)
                job = record["job_id"]
                queue = subprocess.check_output(
                    ["squeue", "--noheader", "--user", getpass.getuser(), "--format=%i|%T"],
                    text=True, timeout=30)
                queue_state = next((line.split("|", 1)[1].strip() for line in queue.splitlines()
                                    if line.split("|", 1)[0].strip() == job), "")
                if queue_state:
                    if record.get("slurm_state") != queue_state:
                        record.update(status="waiting_for_job", slurm_state=queue_state)
                        _write(path, record)
                else:
                    accounting = subprocess.check_output(
                        ["sacct", "--noheader", "--parsable2", "--allocations", "--jobs", job,
                         "--format=JobIDRaw,State,ExitCode"], text=True, timeout=30)
                    job_row = next((line.split("|") for line in accounting.splitlines()
                                    if line.split("|")[0] == job), None)
                    if job_row is not None:
                        _, state, exit_code = job_row[:3]
                        state = state.split()[0].rstrip("+")
                        if state in {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED"}:
                            record.update(slurm_state=state)
                            _write(path, record)
                        else:
                            record.update(slurm_state=state, exit_code=exit_code,
                                          status="completed" if state == "COMPLETED" and exit_code == "0:0"
                                          else "job_failed", terminal=True)
                            break
                    else:
                        missing_accounting += 1
                        if missing_accounting >= 3:
                            raise RuntimeError("Job left the queue but no accounting record appeared in three checks.")
                time.sleep(min(poll_seconds, max(0., deadline - time.time())))
            else:
                raise TimeoutError("The queue and job wait exceeded the watcher timeout.")
        except Exception as error:
            record.update(status="watch_failed", terminal=True,
                          error=f"{type(error).__name__}: {error}")
        record["finished"] = datetime.now(timezone.utc).isoformat()
        _write(path, record)
        return record
