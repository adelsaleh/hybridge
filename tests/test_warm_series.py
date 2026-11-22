"""Warm-started series: perturbation sizes, restart per series, residual checks and summaries."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('warm_series', ROOT/'vendor/adr_gmres/scripts/warm_series.py')
warm_series = importlib.util.module_from_spec(spec)
spec.loader.exec_module(warm_series)


def test_each_step_moves_the_rhs_by_delta_and_each_series_restarts_from_the_base():
    rhs = np.arange(1.0, 11.0)
    rows = list(warm_series.steps(rhs, (1e-4, 1e-2), 3, seed=7))
    assert [(delta, step) for delta, step, _ in rows] == [(1e-4, 1), (1e-4, 2), (1e-4, 3), (1e-2, 1), (1e-2, 2), (1e-2, 3)]
    for (delta, step, b), previous in zip(rows, [rhs, rows[0][2], rows[1][2], rhs, rows[3][2], rows[4][2]]):
        assert np.isclose(np.linalg.norm(b - previous), delta * np.linalg.norm(rhs))
    again = list(warm_series.steps(rhs, (1e-4, 1e-2), 3, seed=7))
    assert all(np.array_equal(a[2], b[2]) for a, b in zip(rows, again))


def test_relative_residual_uses_the_true_operator_and_rhs():
    matrix = np.diag([2.0, 4.0])
    rhs = np.array([2.0, 4.0])
    assert warm_series.relative_residual(lambda v: matrix @ v, np.ones(2), rhs) == 0.0
    assert np.isclose(warm_series.relative_residual(lambda v: matrix @ v, np.zeros(2), rhs), 1.0)


def test_summary_reports_medians_and_pass_counts_per_delta():
    warm = [dict(delta=1e-3, iterations=i, solve_ms=10.0 * i, passed=i < 30) for i in (10, 20, 40)]
    assert warm_series.summarize(warm) == {'0.001': dict(solves=3, passed=2, median_iterations=20.0, median_solve_ms=200.0)}
