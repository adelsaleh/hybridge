"""Use tiny host subprocesses to verify actual resumable batch behavior."""
import json
from pathlib import Path

import pytest

from campaigns.h100_assembly.plan import Task
from campaigns.h100_assembly.run import gpu_counters_restricted, passing_smoke, pilot_tasks, run_task


def task(script, timeout=5):
    return Task('test-task', 'test', ('-c', script), 'data.json', timeout)


def test_pass_resume_and_tampered_artifact_are_distinguished(tmp_path):
    spec = task("from pathlib import Path; Path('{output}/data.json').write_text('{\"measured\": true}')")
    first = run_task(spec, tmp_path, 'sources')
    assert first['status'] == 'passed'
    assert run_task(spec, tmp_path, 'sources') == first
    assert len(list((tmp_path / 'tasks/test-task').glob('attempt_*'))) == 1
    Path(first['artifact']).write_text('{"tampered":true}')
    with pytest.raises(ValueError, match='artifact changed'):
        run_task(spec, tmp_path, 'sources')


def test_batch_budget_interruption_is_retryable_without_losing_attempts(tmp_path):
    marker = tmp_path / 'first_attempt_seen'
    script = ("from pathlib import Path; import time; "
              f"m=Path({str(marker)!r}); first=not m.exists(); m.touch(); "
              "time.sleep(10 if first else 0); Path('{output}/data.json').write_text('{\"ok\":true}')")
    spec = task(script)
    assert run_task(spec, tmp_path, 'sources', timeout=1)['status'] == 'budget_exhausted'
    result = run_task(spec, tmp_path, 'sources')
    assert result['status'] == 'passed' and result['attempt'] == 2
    assert (tmp_path / 'tasks/test-task/attempt_1/result.json').exists()


def test_known_unsupported_path_is_recorded_without_running(tmp_path):
    spec = Task('unsupported', 'test', ('-c', "raise AssertionError('must not run')"), 'data.json', 5,
                unsupported_reason='known degree constraint')
    result = run_task(spec, tmp_path, 'sources')
    assert result['status'] == 'unsupported'
    assert json.loads(Path(result['artifact']).read_text())['measured'] is False


def test_latest_smoke_attempt_must_pass_not_just_an_older_attempt(tmp_path):
    for index, status in [(1, 'passed'), (2, 'failed')]:
        root = tmp_path / f'attempt_{index}'
        root.mkdir()
        (root / 'completion.json').write_text(json.dumps({'status':status}))
    with pytest.raises(ValueError, match='latest smoke'):
        passing_smoke(tmp_path)


def test_restricted_gpu_counters_are_read_from_driver_parameters(tmp_path, monkeypatch):
    monkeypatch.setattr('os.geteuid', lambda: 1000)
    params = tmp_path / 'params'
    assert not gpu_counters_restricted(params)
    params.write_text('EnableMSI: 1\nRmProfilingAdminOnly: 1\n')
    assert gpu_counters_restricted(params)
    params.write_text('EnableMSI: 1\nRmProfilingAdminOnly: 0\n')
    assert not gpu_counters_restricted(params)


def test_pilot_covers_recovery_libraries_and_partial_path_at_supported_degrees():
    selected = pilot_tasks()
    assert len(selected) == 11
    assert {t.order for t in selected} == {1, 6, 9}
    assert all(not t.unsupported_reason for t in selected)
    assert {'partial-transport', 'transport', 'transport-layouts', 'recovery', 'postprocessing'} <= {t.group for t in selected}


def test_oversized_out_of_memory_request_is_memory_excluded():
    from campaigns.h100_assembly.run import oversized_request

    device = {'memory': [{'index': 0, 'used_mib': 5, 'total_mib': 143771}]}
    log_text = ('cupy.cuda.memory.OutOfMemoryError: Out of memory allocating 317,194,240,000 bytes '
                '(allocated so far: 22,874,265,600 bytes).').lower()
    assert oversized_request(log_text, device) == (317_194_240_000, 143771 * 2**20)
    # A request that fits the device is an ordinary failure; so is a log without a request size.
    assert oversized_request('out of memory allocating 1,000,000 bytes', device) is None
    assert oversized_request('cudaerrormemoryallocation: out of memory', device) is None
    assert oversized_request(log_text, {}) is None
