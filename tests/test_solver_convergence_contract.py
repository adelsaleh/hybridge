from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse

import hdgfem
import hdgfem.linalg.system as system
from hdgfem.linalg.system import (
    LinearSolveCapacityError,
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
    def asnumpy(value):
        return np.asarray(value).copy()

    @staticmethod
    def isfinite(value):
        return np.isfinite(value)

    @staticmethod
    def all(value):
        return _DeviceScalar(np.all(value))


class _FakeAmgxAssembly:
    def __init__(self):
        self.data = np.ones(1)


class _FakeAMGXNoMemoryError(RuntimeError):
    def __init__(self, message="CUDA allocation failed"):
        super().__init__(message)
        self.error_code = 7


def test_raw_amgx_retry_wrapper_avoids_full_matrix_backup(monkeypatch) -> None:
    import hdgfem.backends.advection_cuda as raw_amgx

    class FakeDeviceData:
        nbytes = 16

        def __array__(self, dtype=None):
            raise AssertionError("retry wrapper must not materialize the full matrix on the host")

        def set(self, _host_values):
            raise AssertionError("retry wrapper must not restore the matrix from the host")

    class FakeCsrAssembly:
        matrix_format = "csr"

        def __init__(self):
            self.data = FakeDeviceData()

    result = _diagnostic_result(
        solver_residual=1.0e-8,
        solver_target=1.0e-6,
        physical_residual=1.0e-8,
        physical_target=1.0e-6,
    )
    finalize_solve_result(result, backend="pyamgx-device", backend_success=True)
    solution = np.ones(2)
    assembly = FakeCsrAssembly()
    monkeypatch.setattr(raw_amgx, "require_cupy", lambda: _FakeCupy)
    monkeypatch.setattr(
        raw_amgx,
        "_solve_reduced_system_amgx_device_once",
        lambda *_args, **_kwargs: (result, solution),
    )

    returned_result, returned_solution = raw_amgx.solve_reduced_system_amgx_device(
        assembly,
        scale_system="none",
        retry_attempts=({"label": "scaled-retry", "scale_system": "left"},),
    )

    assert returned_result is result
    assert returned_solution is solution
    assert result.amgx_retry_matrix_backup_elapsed_seconds == 0.0
    assert result.amgx_retry_matrix_restore_elapsed_seconds == 0.0
    assert result.amgx_retry_matrix_restore_count == 0
    assert result.amgx_retry_matrix_backup_bytes == 0
    assert result.amgx_retry_wrapper_elapsed_seconds >= 0.0


def test_raw_amgx_retry_cache_reuses_one_preconditioner_and_replaces_coefficients(
    monkeypatch,
) -> None:
    import hdgfem.backends.advection_cuda as raw_amgx

    created = []
    calls = []

    class FakeReusableSolver:
        def __init__(self, **_kwargs):
            self.closed = False
            self.is_setup = False
            created.append(self)

    def fake_attempt(_assembly, **kwargs):
        reusable = kwargs.get("reusable_solver")
        replace_coefficients = kwargs.get("replace_reusable_coefficients", False)
        calls.append((reusable, replace_coefficients))
        if reusable is None:
            result = _diagnostic_result(
                solver_residual=1.0, solver_target=1.0e-6,
                physical_residual=1.0, physical_target=1.0e-6,
            )
            finalize_solve_result(
                result, backend="pyamgx-device", backend_success=True
            )
        else:
            result = _diagnostic_result(
                solver_residual=1.0e-8, solver_target=1.0e-6,
                physical_residual=1.0e-8, physical_target=1.0e-6,
            )
            finalize_solve_result(
                result, backend="pyamgx-device", backend_success=True
            )
            result.amgx_preconditioner_reused = replace_coefficients
            result.amgx_bsr_scalarized = True
            reusable.is_setup = True
        return result, np.ones(2)

    monkeypatch.setattr(raw_amgx, "require_cupy", lambda: _FakeCupy)
    monkeypatch.setattr(raw_amgx, "PyAMGXCsrDeviceSolver", FakeReusableSolver)
    monkeypatch.setattr(
        raw_amgx, "_solve_reduced_system_amgx_device_once", fake_attempt
    )
    cache = {}
    retry = ({
        "label": "robust-scaled",
        "config": {"solver": {"solver": "FGMRES"}},
        "scalarize_bsr": True,
        "reuse_preconditioner": True,
        "solver_cache_key": "fgmres-dilu",
    },)

    first, _ = raw_amgx.solve_reduced_system_amgx_device(
        _FakeAmgxAssembly(), retry_attempts=retry, retry_solver_cache=cache
    )
    second, _ = raw_amgx.solve_reduced_system_amgx_device(
        _FakeAmgxAssembly(), retry_attempts=retry, retry_solver_cache=cache
    )

    assert first.converged and second.converged
    assert len(created) == 1
    assert cache["fgmres-dilu"] is created[0]
    assert calls[1] == (created[0], False)
    assert calls[3] == (created[0], True)
    assert first.amgx_attempts[-1]["preconditioner_reused"] is False
    assert second.amgx_attempts[-1]["preconditioner_reused"] is True


def test_raw_amgx_capacity_failure_is_terminal_after_one_attempt(monkeypatch) -> None:
    import hdgfem.backends.advection_cuda as raw_amgx

    calls = []

    def fail_with_oom(*_args, **_kwargs):
        calls.append("attempt")
        raise _FakeAMGXNoMemoryError()

    monkeypatch.setattr(raw_amgx, "require_cupy", lambda: _FakeCupy)
    monkeypatch.setattr(
        raw_amgx,
        "_solve_reduced_system_amgx_device_once",
        fail_with_oom,
    )

    with pytest.raises(LinearSolveCapacityError) as raised:
        raw_amgx.solve_reduced_system_amgx_device(
            _FakeAmgxAssembly(),
            retry_attempts=({"label": "retry-1"}, {"label": "retry-2"}),
            scale_system=False,
        )

    assert calls == ["attempt"]
    assert raised.value.backend == "pyamgx-device"
    assert raised.value.phase == "AMGX call"
    assert raised.value.amgx_attempt_count == 1
    assert raised.value.amgx_attempts[0]["terminal_capacity_failure"]


def test_raw_amgx_generic_backend_failure_preserves_retry_policy(monkeypatch) -> None:
    import hdgfem.backends.advection_cuda as raw_amgx

    calls = []

    def fail_generically(*_args, **_kwargs):
        calls.append("attempt")
        raise RuntimeError("transient backend failure")

    monkeypatch.setattr(raw_amgx, "require_cupy", lambda: _FakeCupy)
    monkeypatch.setattr(
        raw_amgx,
        "_solve_reduced_system_amgx_device_once",
        fail_generically,
    )

    with pytest.raises(LinearSolveConvergenceError, match="exhausted 3 bounded attempts"):
        raw_amgx.solve_reduced_system_amgx_device(
            _FakeAmgxAssembly(),
            retry_attempts=({"label": "retry-1"}, {"label": "retry-2"}),
            scale_system=False,
        )

    assert calls == ["attempt", "attempt", "attempt"]


def test_raw_amgx_capacity_error_reports_structured_memory() -> None:
    import hdgfem.backends.advection_cuda as raw_amgx

    class FakeRuntime:
        @staticmethod
        def memGetInfo():
            return 2 * 1024**3, 8 * 1024**3

    class FakeCuda:
        runtime = FakeRuntime

    class FakeCupyWithMemory:
        cuda = FakeCuda

    class FakePyAMGX:
        @staticmethod
        def get_device_memory_stats():
            return {
                "live_bytes": 3 * 1024**3,
                "reserved_bytes": 4 * 1024**3,
                "peak_live_bytes": 5 * 1024**3,
                "peak_reserved_bytes": 6 * 1024**3,
            }

    error = raw_amgx._as_amgx_capacity_error(
        _FakeAMGXNoMemoryError(),
        phase="solver setup",
        cp=FakeCupyWithMemory,
        pyamgx=FakePyAMGX,
    )

    assert isinstance(error, LinearSolveCapacityError)
    assert error.phase == "solver setup"
    assert error.memory["amgx"]["live_bytes"] == 3 * 1024**3
    assert error.memory["device"]["used_bytes"] == 6 * 1024**3
    assert "AMGX memory live/reserved=3.000/4.000 GiB" in str(error)
    assert "device memory used/free/total=6.000/2.000/8.000 GiB" in str(error)


def test_raw_amgx_cleanup_continues_after_destroy_failure() -> None:
    import hdgfem.backends.advection_cuda as raw_amgx

    destroyed = []

    class FakeObject:
        def __init__(self, label, *, fail=False):
            self.label = label
            self.fail = fail

        def destroy(self):
            destroyed.append(self.label)
            if self.fail:
                raise RuntimeError(f"{self.label} destroy failed")

    solver = raw_amgx.PyAMGXCsrDeviceSolver.__new__(
        raw_amgx.PyAMGXCsrDeviceSolver
    )
    solver.solver = FakeObject("solver", fail=True)
    solver.vec_x = FakeObject("x")
    solver.vec_b = FakeObject("b")
    solver.mat = FakeObject("matrix")
    solver.cfg = FakeObject("config")
    solver.rsrc = None
    solver.closed = False
    solver._shared_resources_acquired = False

    solver.close(suppress_errors=True)

    assert destroyed == ["solver", "x", "b", "matrix", "config"]
    assert solver.closed
    assert solver.solver is None


def test_raw_amgx_config_enables_residual_history_without_mutating_input() -> None:
    from hdgfem.backends.advection_cuda import _amgx_config_for_solve

    config = {"solver": {"solver": "FGMRES", "store_res_history": 0}}
    normalized = _amgx_config_for_solve(config=config)

    assert normalized["solver"]["monitor_residual"] == 1
    assert normalized["solver"]["store_res_history"] == 1
    assert config["solver"]["store_res_history"] == 0


def test_raw_amgx_config_reserves_native_iteration_table_for_level_three() -> None:
    from hdgfem.backends.advection_cuda import _amgx_config_for_solve

    config = {"solver": {"solver": "BICGSTAB", "print_solve_stats": 0}}

    detailed = _amgx_config_for_solve(config=config, verbose=2)
    fully_verbose = _amgx_config_for_solve(config=config, verbose=3)

    assert detailed["solver"]["obtain_timings"] == 1
    assert detailed["solver"]["print_solve_stats"] == 0
    assert fully_verbose["solver"]["print_solve_stats"] == 1
    assert fully_verbose["solver"]["print_solve_stats_interval"] == 1
    assert "obtain_timings" not in fully_verbose["solver"]


def test_raw_amgx_scaled_solver_validation_uses_configured_relative_tolerance() -> None:
    from hdgfem.backends.advection_cuda import _amgx_relative_residual_check_rtol

    relative = {
        "solver": {"convergence": "RELATIVE_INI_CORE", "tolerance": 1.0e-8}
    }
    absolute = {
        "solver": {"convergence": "ABSOLUTE", "tolerance": 5.0e-9}
    }

    assert _amgx_relative_residual_check_rtol(relative, 1.0e-11) == 1.0e-8
    assert _amgx_relative_residual_check_rtol(absolute, 1.0e-11) == 1.0e-11


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
    assert hdgfem.LinearSolveCapacityError is LinearSolveCapacityError
    assert hdgfem.LinearSolveConvergenceError is LinearSolveConvergenceError
    assert hdgfem.LinearSolveError is system.LinearSolveError
    assert "SolveStatus" in hdgfem.__all__
    assert issubclass(LinearSolveConvergenceError, RuntimeError)
