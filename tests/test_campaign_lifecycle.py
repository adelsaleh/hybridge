"""Stops, process-group cleanup and GPU release checks without a GPU."""
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest

from campaigns.common import lifecycle
from campaigns.common.lifecycle import (CampaignStop, release_device, stop_on_signals, stop_signal_name,
                                        terminate_group)
from campaigns.h100_assembly.plan import Task
from campaigns.h100_assembly.run import run_task


def alive(pid):
    try:
        state = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state != 'Z'


def wait_gone(pid, seconds=5):
    deadline = time.monotonic() + seconds
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(.05)
    return not alive(pid)


def test_stop_signal_names_only_cover_requested_stops():
    assert stop_signal_name(-signal.SIGTERM) == 'SIGTERM'
    assert stop_signal_name(-signal.SIGUSR1) == 'SIGUSR1'
    assert stop_signal_name(-signal.SIGKILL) is None
    assert stop_signal_name(1) is None and stop_signal_name(None) is None


def test_first_signal_stops_and_repeated_signals_cannot_interrupt_cleanup():
    previous = signal.getsignal(signal.SIGTERM)
    with stop_on_signals() as received:
        with pytest.raises(CampaignStop) as stop:
            os.kill(os.getpid(), signal.SIGUSR1)
            time.sleep(1)
        assert stop.value.signal_name == 'SIGUSR1' and received == ['SIGUSR1']
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(.1)
    assert signal.getsignal(signal.SIGTERM) is previous


def test_group_sweep_kills_grandchildren_left_by_a_finished_worker(tmp_path):
    pidfile = tmp_path / 'pid'
    process = subprocess.Popen(['bash', '-c', f'sleep 300 & echo $! > {pidfile}'], start_new_session=True)
    process.wait()
    orphan = int(pidfile.read_text())
    assert alive(orphan)
    assert terminate_group(process) == 'exited'
    assert wait_gone(orphan)


def test_running_worker_is_terminated_with_its_group(tmp_path):
    pidfile = tmp_path / 'pid'
    process = subprocess.Popen(['bash', '-c', f'sleep 300 & echo $! > {pidfile}; wait'], start_new_session=True)
    while not pidfile.exists() or not pidfile.read_text().strip():
        time.sleep(.05)
    child = int(pidfile.read_text())
    assert terminate_group(process, grace=5) == 'terminated'
    assert wait_gone(child)


def test_device_check_is_recorded_not_failed_without_nvidia_smi(monkeypatch, tmp_path):
    monkeypatch.setenv('PATH', str(tmp_path))
    record = release_device(timeout=1)
    assert record['released'] is None and 'error' in record


def fake_nvidia_smi(tmp_path, pid):
    """Report one compute process until it exits, as the driver does."""
    script = tmp_path / 'nvidia-smi'
    script.write_text(f'''#!/bin/bash
case "$1" in
  --query-gpu=*) echo "0, 2048, 143771" ;;
  --query-compute-apps=*) if kill -0 {pid} 2>/dev/null && [[ $(awk '{{print $3}}' /proc/{pid}/stat) != Z ]]; then echo "{pid}, 2000"; fi ;;
esac
''')
    script.chmod(0o755)
    return script


def test_leftover_job_process_is_killed_and_release_is_confirmed(monkeypatch, tmp_path):
    leftover = subprocess.Popen(['sleep', '300'])
    try:
        fake_nvidia_smi(tmp_path, leftover.pid)
        monkeypatch.setenv('PATH', f'{tmp_path}:{os.environ["PATH"]}')
        monkeypatch.setattr(lifecycle, '_ours', lambda pid: 'job' if pid == leftover.pid else None)
        record = release_device(timeout=10)
        assert record['released'] is True and record['killed_pids'] == [leftover.pid]
        assert record['memory'] == [dict(index=0, used_mib=2048, total_mib=143771)]
    finally:
        leftover.kill()
        leftover.wait()


def test_unreadable_process_ids_are_recorded_not_raised(monkeypatch, tmp_path):
    # A GPU in a bad state makes nvidia-smi print '[N/A]' for process ids; int('[N/A]')
    # used to raise ValueError out of release_device.
    script = tmp_path / 'nvidia-smi'
    script.write_text('''#!/bin/bash
case "$1" in
  --query-gpu=*) echo "0, [N/A], [N/A]" ;;
  --query-compute-apps=*) echo "[N/A], [N/A]" ;;
esac
''')
    script.chmod(0o755)
    monkeypatch.setenv('PATH', f'{tmp_path}:{os.environ["PATH"]}')
    record = release_device(timeout=1)
    assert record['released'] is None and 'without a readable pid' in record['error']
    assert record['memory'] == [dict(index=0, used_mib=None, total_mib=None)]
    assert 'device check unavailable' in lifecycle.describe_release(record)


def test_other_jobs_processes_are_never_killed(monkeypatch, tmp_path):
    foreign = subprocess.Popen(['sleep', '300'])
    try:
        fake_nvidia_smi(tmp_path, foreign.pid)
        monkeypatch.setenv('PATH', f'{tmp_path}:{os.environ["PATH"]}')
        monkeypatch.delenv('SLURM_JOB_ID', raising=False)
        record = release_device(timeout=1)
        assert record['released'] is True and record['killed_pids'] == []
        assert foreign.poll() is None
    finally:
        foreign.kill()
        foreign.wait()


def test_worker_killed_by_slurm_stop_is_resumed_not_failed(tmp_path):
    marker = tmp_path / 'first_attempt_seen'
    script = ("from pathlib import Path; import os, signal; "
              f"m=Path({str(marker)!r}); first=not m.exists(); m.touch(); "
              "first and os.kill(os.getpid(), signal.SIGTERM); "
              "Path('{output}/data.json').write_text('{\"ok\":true}')")
    spec = Task('stop-task', 'test', ('-c', script), 'data.json', 5)
    first = run_task(spec, tmp_path, 'sources')
    assert first['status'] == 'interrupted' and first['stop_signal'] == 'SIGTERM'
    assert first['worker_exit'] == 'exited' and 'device' in first
    second = run_task(spec, tmp_path, 'sources')
    assert second['status'] == 'passed' and second['attempt'] == 2
