"""PCGF warm-start regression checks using the existing CUDA runtime only."""

from __future__ import annotations

import numpy as np
import pytest

from hdgfem.linalg.multigrid.face_hp import solve_pcgf_prototype
from hdgfem.runtime.precision import REAL_DTYPE
from scripts.guiding_center.benchmarks.guiding_center_temporal_comparison import kernel_cache_only


@pytest.fixture
def cp():
    cupy = pytest.importorskip("cupy")
    try:
        available = cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        available = False
    if not available:
        pytest.skip("CuPy CUDA runtime is unavailable")
    with kernel_cache_only(True):
        yield cupy


class DenseOperator:
    def __init__(self, cp, matrix):
        self.matrix = cp.asarray(matrix, dtype=REAL_DTYPE)
        self.shape = self.matrix.shape
        self.calls = 0

    def matvec(self, vector):
        self.calls += 1
        return self.matrix @ vector


class IdentityPreconditioner:
    def __init__(self):
        self.calls = 0

    def apply(self, vector):
        self.calls += 1
        return vector.copy()


def test_pcgf_warm_start_solves_original_system_and_reports_actual_initial_residual(cp):
    matrix = np.asarray(
        [[4.0, 0.5, 0.0, 0.0], [0.5, 3.0, 0.25, 0.0],
         [0.0, 0.25, 2.0, 0.125], [0.0, 0.0, 0.125, 1.0]],
        dtype=REAL_DTYPE,
    )
    expected = np.asarray([1.0, -2.0, 0.5, 3.0], dtype=REAL_DTYPE)
    rhs = matrix @ expected
    guess = cp.asarray(expected + np.asarray([0.125, -0.25, 0.375, -0.5]))
    guess_before = cp.asnumpy(guess)
    initial_norm = np.linalg.norm(rhs - matrix @ guess_before)
    tolerance = 200 * np.finfo(REAL_DTYPE).eps
    result = solve_pcgf_prototype(
        DenseOperator(cp, matrix), rhs, IdentityPreconditioner(),
        initial_guess=guess, rtol=tolerance, maxiter=12, true_residual_every=1,
    )

    assert result.converged
    assert 0 < result.iterations <= 6
    np.testing.assert_allclose(cp.asnumpy(result.solution), expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(cp.asnumpy(guess), guess_before)
    np.testing.assert_allclose(result.history[0], initial_norm, rtol=tolerance)
    np.testing.assert_allclose(result.rhs_norm, np.linalg.norm(rhs), rtol=tolerance)
    np.testing.assert_allclose(result.target, tolerance * np.linalg.norm(rhs), rtol=tolerance)
    np.testing.assert_allclose(
        result.residual_over_initial, result.residual_norm / initial_norm, rtol=tolerance,
    )
    assert np.linalg.norm(rhs - matrix @ cp.asnumpy(result.solution)) <= 2 * result.target


def test_pcgf_exact_initial_guess_exits_before_preconditioning(cp):
    matrix = np.diag([2.0, 4.0, 8.0, 16.0])
    guess = cp.asarray([1.0, -2.0, 3.0, -4.0], dtype=REAL_DTYPE)
    rhs = np.asarray([2.0, -8.0, 24.0, -64.0], dtype=REAL_DTYPE)
    operator = DenseOperator(cp, matrix)
    preconditioner = IdentityPreconditioner()
    result = solve_pcgf_prototype(
        operator, rhs, preconditioner, initial_guess=guess, rtol=0.0, atol=0.0,
    )

    assert result.converged
    assert result.iterations == 0
    assert result.history == (0.0,)
    assert result.residual_norm == result.residual_over_initial == 0.0
    assert preconditioner.calls == 0
    assert operator.calls == 1
    np.testing.assert_array_equal(cp.asnumpy(result.solution), cp.asnumpy(guess))
    assert result.solution.data.ptr != guess.data.ptr


@pytest.mark.parametrize("shape", ((3,), (2, 2)))
def test_pcgf_rejects_incompatible_initial_guess_before_operator_application(cp, shape):
    operator = DenseOperator(cp, np.eye(4))
    preconditioner = IdentityPreconditioner()
    with pytest.raises(ValueError, match="initial_guess must have shape"):
        solve_pcgf_prototype(
            operator, np.ones(4), preconditioner, initial_guess=np.zeros(shape),
        )
    assert operator.calls == preconditioner.calls == 0


def test_pcgf_omitted_and_explicit_zero_guess_have_same_solution_and_history(cp):
    matrix = np.diag([2.0, 4.0, 8.0, 16.0])
    rhs = np.asarray([1.0, -2.0, 3.0, -4.0], dtype=REAL_DTYPE)
    options = dict(rtol=200 * np.finfo(REAL_DTYPE).eps, maxiter=12, true_residual_every=1)
    cold = solve_pcgf_prototype(DenseOperator(cp, matrix), rhs, IdentityPreconditioner(), **options)
    explicit = solve_pcgf_prototype(
        DenseOperator(cp, matrix), rhs, IdentityPreconditioner(),
        initial_guess=np.zeros_like(rhs), **options,
    )
    assert cold.converged and explicit.converged
    assert cold.iterations == explicit.iterations
    np.testing.assert_array_equal(cp.asnumpy(cold.solution), cp.asnumpy(explicit.solution))
    np.testing.assert_array_equal(cold.history, explicit.history)
