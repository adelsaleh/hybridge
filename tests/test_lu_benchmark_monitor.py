"""Exercise live progress and interruption recording without numerical work."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

from scripts.advection_diffusion_reaction.diagnostics import check_cached_adr_pardiso as runner


def settings():
    return SimpleNamespace(timeout=5., max_rss_gib=1., reserve_gib=.01,
                           poll_seconds=.01, heartbeat_seconds=30., live_output=True)


def test_live_output_arrives_before_worker_can_finish(tmp_path, monkeypatch):
    acknowledgment = tmp_path / "terminal_received"
    printed = []

    def terminal_print(*values, **kwargs):
        text = " ".join(map(str, values))
        printed.append(text)
        if "worker ready" in text:
            acknowledgment.touch()

    monkeypatch.setattr(runner, "print", terminal_print, raising=False)
    code = """
import json
from pathlib import Path
import sys
import time
output = Path(sys.argv[1])
print('worker ready', flush=True)
deadline = time.monotonic() + 3
while not (output / 'terminal_received').exists() and time.monotonic() < deadline:
    time.sleep(.01)
passed = (output / 'terminal_received').exists()
(output / 'result.json').write_text(json.dumps({'status': 'passed' if passed else 'failed'}))
raise SystemExit(0 if passed else 1)
"""
    result = runner.monitor([sys.executable, "-u", "-B", "-c", code, str(tmp_path)],
                            dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), tmp_path, settings())
    assert result["status"] == "passed"
    assert (tmp_path / "worker.log").read_text() == "worker ready\n"
    assert any("worker ready" in text for text in printed)


def test_monitor_records_caught_interrupt_instead_of_leaving_running(tmp_path, monkeypatch):
    (tmp_path / "result.json").write_text(json.dumps({"status": "running", "stage": "example"}))

    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "memory_kib", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner.monitor([sys.executable, "-B", "-c", "import time; time.sleep(3)"],
                       dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), tmp_path, settings())
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "interrupted"
    assert result["stage"] == "example"
    assert result["returncode"] != 0
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "interrupted"


def test_benchmark_summary_retains_interrupted_worker(tmp_path, monkeypatch):
    from scripts.guiding_center.poisson import benchmark_scipy_lu_gpu as benchmark

    def interrupted(command, env, output, args):
        (output / "result.json").write_text(json.dumps({"status": "interrupted"}))
        raise KeyboardInterrupt

    monkeypatch.setattr(benchmark, "monitor", interrupted)
    with pytest.raises(KeyboardInterrupt):
        benchmark.main(["--smoke", "--output", str(tmp_path), "--threads", "1"])
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["results"]["factor"]["status"] == "interrupted"
