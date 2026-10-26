"""Recovered campaign inspection modes remain read-only outside the repository cwd."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGNS = ROOT / "campaigns"
PLANS = [
    ("gc_d8_diocotron_order/launcher.py", "--plan"),
    ("gc_d8s_smoke/launcher.py", "--plan"),
    ("gc_e7_band_evolution/launcher.py", "--plan"),
    ("gc_e7_five_sites/launcher.py", "--plan"),
    ("gc_e7_lumps/launcher.py", "--plan"),
    ("gc_poisson_pmg_degrees/launcher.py", "--plan"),
    ("gc_diocotron_shell/todo_20261015/launcher.py", "--list"),
]


@pytest.mark.parametrize("script, option", PLANS)
def test_local_campaign_plan_needs_no_live_data_and_writes_nothing(tmp_path, script, option):
    output = tmp_path / "outputs"
    env = {**os.environ, "HYBRIDGE_ARTICLE_ROOT": str(tmp_path / "missing_article"),
           "HYBRIDGE_CAMPAIGN_OUT": str(output), "HYBRIDGE_PYTHON": sys.executable}
    run = subprocess.run([sys.executable, str(CAMPAIGNS / script), option], cwd=tmp_path,
                         env=env, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.strip()
    assert not output.exists()
    assert not (tmp_path / "missing_article").exists()


def test_preserved_records_match_their_manifests():
    count = 0
    for path in CAMPAIGNS.glob("*/records/**/manifest.json"):
        manifest = json.loads(path.read_text())
        if "copied" not in manifest:  # a source manifest preserved as evidence
            continue
        for row in manifest["copied"]:
            data = (path.parent / row["copy"]).read_bytes()
            assert len(data) == row["bytes"]
            assert hashlib.sha256(data).hexdigest() == row["sha256"], str(path)
            assert Path(row["copy"]).name not in {"AGENTS.md", "CODEMAP.md"}
            count += 1
    assert count > 0


def test_recovered_sources_use_current_paths_and_parse():
    source_manifest = json.loads((CAMPAIGNS / "local_sources.json").read_text())
    for row in source_manifest["files"]:
        path = CAMPAIGNS / row["copy"]
        text = path.read_text()
        assert "/home/" not in text, path
        assert not re.search(r"guiding_center[./](benchmarks|poisson)[./]", text), path
        assert not re.search(r"\bkill(?: -0)? [0-9]", text), path
        if path.suffix == ".py":
            ast.parse(text, filename=str(path))
        else:
            check = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
            assert check.returncode == 0, check.stderr


def test_magnetron_resume_requires_the_final_time(tmp_path):
    from campaigns.gc_d7_magnetron.resume_convergence import completed

    record = {"prefix": "run", "config": SimpleNamespace(dt=0.1, num_steps=20)}
    assert not completed(record, tmp_path)
    path = tmp_path / "run.jsonl"
    path.write_text('{"time": 1.9}\n')
    assert not completed(record, tmp_path)
    path.write_text('{"time": 2.0}\n')
    assert completed(record, tmp_path)
    path.write_text('{"time": 2.0}\n{"time":')
    assert not completed(record, tmp_path)
