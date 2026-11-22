"""The alpha u = f driver solves on the reference simplex and writes its figures."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reference_simplex" / "plot_alpha_u_f.py"
POLYNOMIAL_PROBLEM = '''
import numpy as np

def alpha(x):
    """Variable positive coefficient."""
    return 1.0 + x[:, 0] ** 2

def f(x):
    """alpha times a cubic polynomial, so that u = f / alpha is that cubic."""
    u = 1.0 + x[:, 0] - 2.0 * x[:, 1] ** 2 * x[:, 0] + 0.5 * x[:, -1] ** 3
    return alpha(x) * u
'''


def _driver():
    """Import ``scripts/reference_simplex/plot_alpha_u_f.py`` as a module."""
    spec = importlib.util.spec_from_file_location("plot_alpha_u_f", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dim", (2, 3))
def test_cubic_solution_is_reproduced_from_degree_three(dim, tmp_path):
    """For u of degree 3 and a variable alpha, u_h = u to rounding once p >= 3; the figures are written."""
    problem = tmp_path / "problem.py"
    problem.write_text(POLYNOMIAL_PROBLEM)
    out = tmp_path / "out"
    assert _driver().main(["--problem", str(problem), "--dim", str(dim), "--degrees", "1", "3", "--out", str(out)]) == 0
    record = json.loads((out / "errors.json").read_text())
    assert record["degrees"]["1"]["l2_error"] > 1e-3
    assert record["degrees"]["3"]["l2_error"] < 1e-12
    assert record["degrees"]["3"]["max_error"] < 1e-11
    for name in ("alpha_u_f.pdf", "degree-01.png", "degree-03.png", "summary.png"):
        assert (out / name).stat().st_size > 1000
