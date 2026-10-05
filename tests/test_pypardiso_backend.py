from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse

import hybridge.linalg.system as system
import hybridge.linalg.direct as linalg_direct
from hybridge.linalg import (
    LinearSolveConvergenceError,
    LinearSolveError,
    clear_pypardiso_cache,
    solve_global_system,
    solve_pypardiso_system,
)


class _FakeNativeSolver:
    def __init__(self, owner, *, mtype: int):
        self.owner = owner
        self.mtype = mtype

    def free_memory(self, *, everything: bool):
        self.owner.solver_cleanup.append((self.mtype, everything))


class _FakePardiso:
    def __init__(self, solution=None, error: Exception | None = None):
        self.solution = solution
        self.error = error
        self.calls = []
        self.cleanup = []
        self.solver_cleanup = []
        self.created_mtypes = []
        self.ps = SimpleNamespace(free_memory=self._free_memory)

    def PyPardisoSolver(self, *, mtype: int):
        self.created_mtypes.append(mtype)
        return _FakeNativeSolver(self, mtype=mtype)

    def spsolve(self, matrix, rhs, *, solver=None):
        mtype = 11 if solver is None else solver.mtype
        self.calls.append((matrix.copy(), rhs.copy(), mtype))
        if self.error is not None:
            raise self.error
        if self.solution is not None:
            return np.asarray(self.solution, dtype=np.float64)
        solve_matrix = matrix
        if mtype == 2:
            diagonal = scipy.sparse.diags(matrix.diagonal(), format="csr")
            solve_matrix = (matrix + matrix.T - diagonal).tocsr()
        return scipy.sparse.linalg.spsolve(solve_matrix, rhs)

    def _free_memory(self, *, everything: bool):
        self.cleanup.append(everything)


@pytest.fixture(autouse=True)
def _reset_pypardiso_solver_cache():
    linalg_direct._PYPARDISO_SOLVERS.clear()
    yield
    linalg_direct._PYPARDISO_SOLVERS.clear()


@pytest.mark.parametrize("solver_name", ("pypardiso", "pardiso"))
def test_pypardiso_aliases_use_normalized_direct_result(monkeypatch, solver_name: str) -> None:
    fake = _FakePardiso()
    monkeypatch.setattr(linalg_direct, "_import_pypardiso", lambda: fake)
    rows = np.array([1, 0, 1, 0], dtype=np.int64)
    cols = np.array([1, 1, 0, 0], dtype=np.int64)
    data = np.array([3.0, 1.0, 2.0, 4.0], dtype=np.float64)
    rhs = np.array([1.0, 2.0])

    result = solve_global_system(rows, cols, data, rhs, 2, solver=solver_name, rtol=1.0e-12)

    np.testing.assert_allclose(result.x, [0.1, 0.6])
    assert result.backend == "pypardiso-direct"
    assert result.converged
    assert result.physical_residual_target_met
    assert len(fake.calls) == 1
    matrix, passed_rhs, mtype = fake.calls[0]
    assert mtype == 11
    assert scipy.sparse.isspmatrix_csr(matrix)
    assert matrix.has_sorted_indices
    assert matrix.dtype == np.float64
    np.testing.assert_array_equal(passed_rhs, rhs)


@pytest.mark.parametrize(
    "solver_name",
    ("pypardiso-spd", "pardiso-spd", "pypardiso_spd", "pardiso_spd"),
)
def test_pypardiso_spd_aliases_use_upper_triangle_and_mtype_two(
    monkeypatch, solver_name: str
) -> None:
    fake = _FakePardiso()
    monkeypatch.setattr(linalg_direct, "_import_pypardiso", lambda: fake)
    matrix = scipy.sparse.csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0])

    result = solve_global_system(
        (),
        (),
        (),
        rhs,
        2,
        solver=solver_name,
        rtol=1.0e-12,
        assembled_matrix=matrix,
    )

    np.testing.assert_allclose(result.x, [1.0 / 11.0, 7.0 / 11.0])
    assert result.backend == "pypardiso-spd"
    assert result.converged
    assert fake.created_mtypes == [2]
    native_matrix, passed_rhs, mtype = fake.calls[0]
    assert mtype == 2
    assert scipy.sparse.tril(native_matrix, k=-1).nnz == 0
    np.testing.assert_array_equal(passed_rhs, rhs)


def test_pypardiso_native_success_cannot_override_true_residual_failure(monkeypatch) -> None:
    fake = _FakePardiso(solution=np.zeros(2))
    monkeypatch.setattr(linalg_direct, "_import_pypardiso", lambda: fake)
    matrix = scipy.sparse.eye(2, format="csr")
    rhs = np.ones(2)

    result = solve_pypardiso_system(matrix, rhs, rtol=1.0e-12)

    assert not result.converged
    assert result.failure_reason == "residual-target-not-met"
    assert result.backend_info == 0
    with pytest.raises(LinearSolveConvergenceError) as raised:
        solve_pypardiso_system(matrix, rhs, rtol=1.0e-12, raise_on_nonconvergence=True)
    assert raised.value.result is not None
    assert raised.value.result.failure_reason == "residual-target-not-met"


def test_pypardiso_native_failure_uses_stable_linear_solve_error(monkeypatch) -> None:
    fake = _FakePardiso(error=ValueError("native failure"))
    monkeypatch.setattr(linalg_direct, "_import_pypardiso", lambda: fake)

    with pytest.raises(LinearSolveError, match="pypardiso solve failed: native failure"):
        solve_pypardiso_system(scipy.sparse.eye(2, format="csr"), np.ones(2))
    assert fake.cleanup == [True]


@pytest.mark.parametrize("complex_rhs", (False, True))
def test_pypardiso_rejects_complex_systems_before_import(monkeypatch, complex_rhs: bool) -> None:
    def unexpected_import():
        raise AssertionError("optional runtime imported before input validation")

    monkeypatch.setattr(linalg_direct, "_import_pypardiso", unexpected_import)
    matrix_dtype = np.float64 if complex_rhs else np.complex128
    rhs_dtype = np.complex128 if complex_rhs else np.float64

    with pytest.raises(TypeError, match="real-valued systems only"):
        solve_pypardiso_system(
            scipy.sparse.eye(2, format="csr", dtype=matrix_dtype),
            np.ones(2, dtype=rhs_dtype),
        )


def test_pypardiso_spd_rejects_nonsymmetric_input_before_import(monkeypatch) -> None:
    def unexpected_import():
        raise AssertionError("optional runtime imported before symmetry validation")

    monkeypatch.setattr(linalg_direct, "_import_pypardiso", unexpected_import)
    matrix = scipy.sparse.csr_matrix([[2.0, 1.0], [0.0, 2.0]])

    with pytest.raises(ValueError, match="requires a symmetric matrix"):
        solve_pypardiso_system(matrix, np.ones(2), matrix_type="spd")


def test_pypardiso_rejects_nonfinite_inputs_before_import(monkeypatch) -> None:
    def unexpected_import():
        raise AssertionError("optional runtime imported before input validation")

    monkeypatch.setattr(linalg_direct, "_import_pypardiso", unexpected_import)
    matrix = scipy.sparse.csr_matrix(np.diag([1.0, np.nan]))

    with pytest.raises(ValueError, match="matrix data"):
        solve_pypardiso_system(matrix, np.ones(2))


def test_clear_pypardiso_cache_releases_the_singleton(monkeypatch) -> None:
    fake = _FakePardiso()
    monkeypatch.setattr(linalg_direct, "_import_pypardiso", lambda: fake)

    clear_pypardiso_cache()
    clear_pypardiso_cache(everything=False)

    assert fake.cleanup == [True, False]


def test_clear_pypardiso_cache_releases_cached_spd_solver(monkeypatch) -> None:
    fake = _FakePardiso()
    monkeypatch.setattr(linalg_direct, "_import_pypardiso", lambda: fake)
    matrix = scipy.sparse.csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.ones(2)

    solve_pypardiso_system(matrix, rhs, matrix_type="spd")
    solve_pypardiso_system(matrix, rhs, matrix_type="spd")
    clear_pypardiso_cache()

    assert fake.created_mtypes == [2]
    assert len(fake.calls) == 2
    assert fake.cleanup == [True]
    assert fake.solver_cleanup == [(2, True)]
    assert not linalg_direct._PYPARDISO_SOLVERS
