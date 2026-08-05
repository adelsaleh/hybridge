from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse

import hdgfem
import hdgfem.linalg.system as system
from hdgfem.linalg.system import (
    LinearSolveConvergenceError,
    SolveResult,
    finalize_solve_result,
    residual_history_is_stagnated,
    solve_global_system,
    solve_iterative_system,
    solve_petsc_system,
)


def _identity_problem():
    matrix = scipy.sparse.eye(2, format="csr", dtype=np.float64)
    rhs = np.array([1.0, -2.0])
    return matrix, rhs


def _diagnostic_result(*, solver_residual: float, solver_target: float, physical_residual: float, physical_target: float):
    return SolveResult(
        x=np.zeros(2),
        info=0,
        solver_residual_norm=solver_residual,
        solver_rhs_norm=1.0,
        solver_relative_residual_norm=solver_residual,
        solver_residual_target=solver_target,
        physical_residual_norm=physical_residual,
        physical_rhs_norm=1.0,
        physical_relative_residual_norm=physical_residual,
        physical_residual_target=physical_target,
        rtol=1.0e-6,
        atol=0.0,
    )


def test_direct_solve_exposes_normalized_convergence_fields() -> None:
    rows = np.array([0, 1])
    cols = np.array([0, 1])
    data = np.array([2.0, 3.0])
    rhs = np.array([4.0, 9.0])

    result = solve_global_system(rows, cols, data, rhs, 2, solver="direct", rtol=1.0e-12)

    np.testing.assert_allclose(result.x, [2.0, 3.0])
    assert result.converged
    assert result.status == "converged"
    assert result.info == 0
    assert result.backend == "scipy-direct"
    assert result.backend_info == 0
    assert result.solution_is_finite
    assert result.solver_residual_target_met
    assert result.physical_residual_target_met


@pytest.mark.parametrize(
    "data,rhs,guess,label",
    (
        (np.array([np.nan, 1.0]), np.ones(2), None, "matrix_values"),
        (np.ones(2), np.array([1.0, np.inf]), None, "rhs"),
        (np.ones(2), np.ones(2), np.array([0.0, np.nan]), "initial_guess"),
    ),
)
def test_nonfinite_inputs_are_rejected_before_backend_dispatch(data, rhs, guess, label: str) -> None:
    with pytest.raises(ValueError, match=label):
        solve_global_system(
            np.array([0, 1]),
            np.array([0, 1]),
            data,
            rhs,
            2,
            solver="bicgstab",
            initial_guess=guess,
        )


def test_backend_success_cannot_override_true_residual_failure(monkeypatch) -> None:
    matrix, rhs = _identity_problem()

    def false_success(_matrix, _rhs, **_kwargs):
        return np.zeros_like(_rhs), 0

    monkeypatch.setitem(system._ITERATIVE_SOLVERS, "BICGSTAB", false_success)
    result = solve_iterative_system(
        matrix,
        rhs,
        solver_name="bicgstab",
        preconditioner=None,
        scale_system=False,
        rtol=1.0e-12,
    )

    assert not result.converged
    assert result.status == "not-converged"
    assert result.failure_reason == "residual-target-not-met"
    assert result.info == 1
    assert result.backend_info == 0
    assert not result.physical_residual_target_met


def test_raise_on_nonconvergence_carries_the_normalized_result(monkeypatch) -> None:
    matrix, rhs = _identity_problem()

    def false_success(_matrix, _rhs, **_kwargs):
        return np.zeros_like(_rhs), 0

    monkeypatch.setitem(system._ITERATIVE_SOLVERS, "BICGSTAB", false_success)
    with pytest.raises(LinearSolveConvergenceError) as raised:
        solve_iterative_system(
            matrix,
            rhs,
            solver_name="bicgstab",
            preconditioner=None,
            scale_system=False,
            rtol=1.0e-12,
            raise_on_nonconvergence=True,
        )

    assert raised.value.result is not None
    assert raised.value.result.failure_reason == "residual-target-not-met"
    assert "physical_target=" in str(raised.value)


def test_backend_nonconvergence_is_preserved_separately_from_normalized_info(monkeypatch) -> None:
    matrix, rhs = _identity_problem()

    def native_failure(_matrix, _rhs, **_kwargs):
        return _rhs.copy(), 7

    monkeypatch.setitem(system._ITERATIVE_SOLVERS, "BICGSTAB", native_failure)
    result = solve_iterative_system(
        matrix,
        rhs,
        solver_name="bicgstab",
        preconditioner=None,
        scale_system=False,
        rtol=1.0e-12,
    )

    assert not result.converged
    assert result.failure_reason == "backend-nonconvergence"
    assert result.info == 7
    assert result.backend_info == 7
    assert result.physical_residual_target_met


def test_physical_residual_target_is_an_independent_acceptance_condition() -> None:
    result = _diagnostic_result(
        solver_residual=1.0e-9,
        solver_target=1.0e-6,
        physical_residual=1.0e-3,
        physical_target=1.0e-6,
    )

    finalized = finalize_solve_result(result, backend="test", backend_success=True)

    assert finalized.solver_residual_target_met
    assert not finalized.physical_residual_target_met
    assert not finalized.converged
    assert finalized.failure_reason == "residual-target-not-met"


def test_stagnation_history_is_bounded_and_classified() -> None:
    plateau = np.full(96, 3.0)
    improving = np.geomspace(1.0, 1.0e-8, 64)
    assert residual_history_is_stagnated(plateau)
    assert not residual_history_is_stagnated(improving)

    result = _diagnostic_result(
        solver_residual=3.0,
        solver_target=1.0e-6,
        physical_residual=3.0,
        physical_target=1.0e-6,
    )
    finalized = finalize_solve_result(
        result,
        backend="test",
        backend_success=True,
        residual_history=plateau,
    )
    assert finalized.status == "stagnated"
    assert finalized.failure_reason == "stagnation"
    assert finalized.residual_history == tuple(plateau[-64:])


def test_petsc_wrapper_destroys_registered_resources_on_failure(monkeypatch) -> None:
    destroyed = []

    class Resource:
        def __init__(self, name: str):
            self.name = name

        def destroy(self):
            destroyed.append(self.name)

    def fail_impl(*_args, _resource_owner, **_kwargs):
        _resource_owner(Resource("matrix"))
        _resource_owner(Resource("vector"))
        raise RuntimeError("native PETSc failure")

    monkeypatch.setattr(system, "_solve_petsc_system_impl", fail_impl)
    matrix, rhs = _identity_problem()
    with pytest.raises(RuntimeError, match="native PETSc failure"):
        solve_petsc_system(matrix, rhs)

    assert destroyed == ["vector", "matrix"]


def test_petsc_matrix_construction_failure_destroys_partial_matrix(monkeypatch) -> None:
    destroyed = []

    class FakePetscMatrix:
        class Type:
            AIJ = "aij"

        def create(self, **_kwargs):
            return self

        def setPreallocationCOO(self, *_args):
            pass

        def setSizes(self, _shape):
            raise RuntimeError("matrix construction failed")

        def destroy(self):
            destroyed.append("matrix")

    class FakePETSc:
        COMM_WORLD = object()
        IntType = np.int64
        Mat = FakePetscMatrix

    monkeypatch.setattr(system, "_import_petsc", lambda: FakePETSc)
    matrix, rhs = _identity_problem()
    with pytest.raises(RuntimeError, match="matrix construction failed"):
        solve_petsc_system(matrix, rhs)

    assert destroyed == ["matrix"]


def test_amgx_retry_count_is_bounded_before_optional_runtime_import() -> None:
    from hdgfem.backends.advection_cuda import solve_reduced_system_amgx_device

    retries = tuple({"label": f"retry-{index}"} for index in range(8))
    with pytest.raises(ValueError, match="at most 8 bounded attempts"):
        solve_reduced_system_amgx_device(None, retry_attempts=retries)


class _DeviceScalar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _FakeCupy:
    @staticmethod
    def isfinite(value):
        return np.isfinite(value)

    @staticmethod
    def all(value):
        return _DeviceScalar(np.all(value))


class _FakeAmgxAssembly:
    def __init__(self):
        self.data = np.ones(1)


def test_raw_amgx_config_enables_residual_history_without_mutating_input() -> None:
    from hdgfem.backends.advection_cuda import _amgx_config_for_solve

    config = {"solver": {"solver": "FGMRES", "store_res_history": 0}}
    normalized = _amgx_config_for_solve(config=config)

    assert normalized["solver"]["store_res_history"] == 1
    assert config["solver"]["store_res_history"] == 0


def test_raw_amgx_retry_exhaustion_raises_stable_convergence_error(monkeypatch) -> None:
    import hdgfem.backends.advection_cuda as raw_amgx

    result = _diagnostic_result(
        solver_residual=1.0,
        solver_target=1.0e-6,
        physical_residual=1.0,
        physical_target=1.0e-6,
    )
    finalize_solve_result(result, backend="pyamgx-device", backend_success=True)
    monkeypatch.setattr(raw_amgx, "require_cupy", lambda: _FakeCupy)
    monkeypatch.setattr(
        raw_amgx,
        "_solve_reduced_system_amgx_device_once",
        lambda *_args, **_kwargs: (result, np.ones(2)),
    )

    with pytest.raises(LinearSolveConvergenceError) as raised:
        raw_amgx.solve_reduced_system_amgx_device(
            _FakeAmgxAssembly(),
            retry_attempts=({"label": "retry"},),
        )

    assert raised.value.result is result
    assert result.amgx_attempt_count == 2


def test_raw_amgx_nonraising_retry_returns_last_nonfinite_result(monkeypatch) -> None:
    import hdgfem.backends.advection_cuda as raw_amgx

    result = _diagnostic_result(
        solver_residual=np.nan,
        solver_target=1.0e-6,
        physical_residual=np.nan,
        physical_target=1.0e-6,
    )
    finalize_solve_result(
        result,
        backend="pyamgx-device",
        backend_success=True,
        solution_is_finite=False,
    )
    solution = np.full(2, np.nan)
    monkeypatch.setattr(raw_amgx, "require_cupy", lambda: _FakeCupy)
    monkeypatch.setattr(
        raw_amgx,
        "_solve_reduced_system_amgx_device_once",
        lambda *_args, **_kwargs: (result, solution),
    )

    returned_result, returned_solution = raw_amgx.solve_reduced_system_amgx_device(
        _FakeAmgxAssembly(),
        retry_attempts=({"label": "retry"},),
        raise_on_nonconvergence=False,
    )

    assert returned_result is result
    assert returned_solution is solution
    assert result.amgx_attempt_count == 2



def test_convergence_contract_is_available_from_public_packages() -> None:
    assert hdgfem.LinearSolveConvergenceError is LinearSolveConvergenceError
    assert hdgfem.LinearSolveError is system.LinearSolveError
    assert "SolveStatus" in hdgfem.__all__
    assert issubclass(LinearSolveConvergenceError, RuntimeError)
