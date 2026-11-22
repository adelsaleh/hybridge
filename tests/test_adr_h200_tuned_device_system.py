"""The H200 tuned campaign's device-resident system path, run with NumPy as the array module
(see adr_h200_tuned_device_system_helper.py)."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_device_system_matches_host_route(tmp_path):
    helper = Path(__file__).with_name('adr_h200_tuned_device_system_helper.py')
    run = subprocess.run([sys.executable, '-B', str(helper), str(ROOT), str(tmp_path)], cwd=tmp_path,
                         capture_output=True, text=True, timeout=900)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-4000:]
    assert run.stdout.count('match the host code') == 3
