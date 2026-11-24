"""The noninteractive watcher submits once and stops on terminal outcomes."""
import json

from campaigns.common import slurm_watch


def test_capacity_refusal_is_retried_then_success_is_recorded(tmp_path, monkeypatch):
    calls = []
    def submit():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("Shared GPU ceiling: occupied")
        return "123"
    responses = iter(["123|RUNNING\n", "", "123|COMPLETED|0:0\n"])
    monkeypatch.setattr(slurm_watch.subprocess, "check_output", lambda *a, **k: next(responses))
    monkeypatch.setattr(slurm_watch.time, "sleep", lambda _: None)
    path = tmp_path / "state.json"
    record = slurm_watch.watch_slurm(submit, path)
    assert len(calls) == 2
    assert record["status"] == "completed"
    assert record["terminal"]
    assert json.loads(path.read_text())["job_id"] == "123"


def test_restart_monitors_recorded_job_without_resubmission(tmp_path, monkeypatch):
    responses = iter(["", "123|FAILED|1:0\n"])
    monkeypatch.setattr(slurm_watch.subprocess, "check_output", lambda *a, **k: next(responses))
    record = slurm_watch.watch_slurm(lambda: (_ for _ in ()).throw(AssertionError()),
                                    tmp_path / "state.json", job_id="123")
    assert record["status"] == "job_failed"


def test_unexpected_submission_error_is_terminal(tmp_path):
    def submit():
        raise RuntimeError("scheduler unavailable")
    record = slurm_watch.watch_slurm(submit, tmp_path / "state.json")
    assert record["status"] == "watch_failed"
    assert "scheduler unavailable" in record["error"]
