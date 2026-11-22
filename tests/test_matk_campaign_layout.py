"""matK source aliases and byte-preserved records need no full matrix or GPU."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


import pytest


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaigns/matk"
RECORDS = ROOT / "docs/campaigns/matk"


def _relocations():
    """Return the relocation manifest of the matk records, or skip where the records are absent.

    Campaign records stay on the machine that ran the campaign; a fresh clone has none.
    """
    manifest = RECORDS / "relocations.json"
    if not manifest.is_file():
        pytest.skip("the matk campaign records are not on this machine")
    return json.loads(manifest.read_text())


def test_historical_matk_source_path_resolves_to_campaign():
    """Recorded scripts/matk entry points remain valid after the source move."""
    historical = ROOT / "scripts/matk"
    assert historical.is_symlink()
    assert os.readlink(historical) == "../campaigns/matk"
    assert historical.resolve() == CAMPAIGN
    manifest = _relocations()
    for row in manifest["source_moves"]:
        old, new = ROOT / row["historical_path"], ROOT / row["path"]
        assert old.is_file() and new.is_file()
        assert old.resolve() == new.resolve()


def test_relocated_matk_records_preserve_bytes_and_aliases():
    """Historical results keep their original hashes at canonical record paths."""
    manifest = _relocations()
    assert len(manifest["record_moves"]) == 16
    for row in manifest["record_moves"]:
        path = ROOT / row["path"]
        content = path.read_bytes()
        assert len(content) == row["bytes"]
        assert hashlib.sha256(content).hexdigest() == row["sha256"]
        assert (ROOT / row["historical_path"]).resolve() == path
        assert (ROOT / row["compatibility_path"]).resolve() == path


def test_both_matk_cli_paths_list_the_same_candidates_without_input_or_gpu(tmp_path):
    """The existing plan operation imports local helpers through either path."""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    plans = []
    for directory in ("campaigns/matk", "scripts/matk"):
        output = tmp_path / directory.replace("/", "-")
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / directory / "benchmark_amgx_strong.py"),
             "--list-configs", "--matrix", str(tmp_path / "absent.h5"), "--output", str(output)],
            cwd=tmp_path, env=environment, text=True, capture_output=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        plans.append(json.loads(result.stdout))
        assert not output.exists()
    assert plans[0] == plans[1]
    assert "fgmres_natural_ilu0_b20" in plans[0]


def test_matk_launchers_parse_and_new_checkpoints_stay_outside_source():
    """Relocation preserves shell syntax and routes generated archives to data."""
    for path in sorted(CAMPAIGN.glob("*.sbatch")) + sorted(CAMPAIGN.glob("*.sh")):
        result = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
    checkpoint = (CAMPAIGN / "amgx_checkpoints.sbatch").read_text()
    assert "$repo/run_outputs/matk/saved_iterates/reference-$SLURM_JOB_ID" in checkpoint
    assert "$repo/campaigns/matk/saved_iterates/reference-$SLURM_JOB_ID" not in checkpoint
