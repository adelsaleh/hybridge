"""Campaign evidence collection never rewrites data or adds binary/local-agent files."""
import hashlib
import json
from pathlib import Path

import pytest

from campaigns.common import records


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    destination = repo / "campaigns" / "sample"
    destination.mkdir(parents=True)
    (destination / "README.md").write_text("# Sample\n")
    monkeypatch.setattr(records, "REPO", repo)
    data = tmp_path / "data"
    data.mkdir()
    return data, destination


def test_records_preserve_bytes_and_report_partial_attempts(campaign):
    data, destination = campaign
    raw = b'{"run_id":"first","status":"failed"}\n{"run_id":"second","status":"ok"}\n'
    (data / "runs.jsonl").write_bytes(raw)
    (data / "PHASE_DONE").write_text("done\n")
    (data / "DATA.md").write_text("original data notes\n")
    (data / "manifest.json").write_text('{"original": true}\n')
    (data / "inputs").mkdir()
    (data / "inputs" / "mesh.json").write_text('{"cells": 2}\n')
    (data / "large.json").write_bytes(b" " * 500)
    for name in ["AGENTS.md", "CODEMAP.md", "failure_system.npz", "run.log"]:
        (data / name).write_bytes(b"not a record\0")
    result = records.collect("sample", data, 200, label="phase", include=("inputs/*.json",), status="partial")
    assert result == destination / "records" / "phase"
    manifest = json.loads((result / "manifest.json").read_text())
    assert manifest["status"] == "partial"
    assert manifest["run_attempt_statuses"] == {"failed": 1, "ok": 1}
    assert manifest["completion_markers"] == ["PHASE_DONE"]
    for item in manifest["copied"]:
        assert (data / item["path"]).read_bytes() == (result / item["copy"]).read_bytes()
        assert hashlib.sha256((result / item["copy"]).read_bytes()).hexdigest() == item["sha256"]
    assert (result / "source" / "DATA.md").read_text() == "original data notes\n"
    assert (result / "inputs" / "mesh.json").exists()
    assert manifest["too_large_or_nontext"] == [{"path": "large.json", "bytes": 500}]
    for name in ["AGENTS.md", "CODEMAP.md", "failure_system.npz", "run.log", "large.json"]:
        assert not (result / name).exists()
    assert (data / "runs.jsonl").read_bytes() == raw


def test_records_reject_escaping_paths_and_skip_symlinked_evidence(campaign, tmp_path):
    data, destination = campaign
    (tmp_path / "outside.json").write_text('{"outside":true}')
    (data / "linked.json").symlink_to(tmp_path / "outside.json")
    (data / "nested").symlink_to(tmp_path, target_is_directory=True)
    result = records.collect("sample", data, include=("nested/outside.json",))
    assert not json.loads((result / "manifest.json").read_text())["copied"]
    for kwargs in [{"campaign": "../escape"}, {"label": "../escape"}, {"include": ("../*.json",)}]:
        with pytest.raises(ValueError):
            records.collect(**{"campaign": "sample", "data": data, **kwargs})
    with pytest.raises(ValueError, match="outside the data"):
        records.collect("sample", destination)
