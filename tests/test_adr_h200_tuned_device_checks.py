"""The H200 tuned campaign's device checks and device AMGX BSR, run with NumPy as the array module,
against the vendored host verify and SciPy BSR (see adr_h200_tuned_device_check_helper.py)."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_device_checks_match_vendored_host_verify(tmp_path):
    helper = Path(__file__).with_name('adr_h200_tuned_device_check_helper.py')
    run = subprocess.run([sys.executable, '-B', str(helper), str(ROOT), str(tmp_path)], cwd=tmp_path,
                         capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-4000:]
    assert run.stdout.count('match the vendored host code') == 3
