"""Smoke tests for the copy-runnable examples referenced by the manual."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script_name",
    ("advection_reaction_minimal.py", "diffusion_reaction_minimal.py"),
)
def test_minimal_documented_example_runs(script_name: str) -> None:
    """Each minimal example must run with the active base environment."""
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples" / script_name)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert "L2 error:" in completed.stdout
    assert "physical relative residual:" in completed.stdout
