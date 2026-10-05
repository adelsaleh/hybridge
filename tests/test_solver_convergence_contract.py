from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse

import hybridge
import hybridge.linalg.results as linalg_results
import hybridge.linalg.iterative as linalg_iterative
from hybridge.linalg.results import (
    LinearSolveCapacityError,
    LinearSolveConvergenceError,
    SolveResult,
    finalize_solve_result,
    residual_history_is_stagnated,
)
from hybridge.linalg.system import solve_global_system
from hybridge.linalg.iterative import solve_iterative_system, solve_petsc_system


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

    monkeypatch.setitem(linalg_iterative._ITERATIVE_SOLVERS, "BICGSTAB", false_success)
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

    monkeypatch.setitem(linalg_iterative._ITERATIVE_SOLVERS, "BICGSTAB", false_success)
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

    monkeypatch.setitem(linalg_iterative._ITERATIVE_SOLVERS, "BICGSTAB", native_failure)
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

    monkeypatch.setattr(linalg_iterative, "_solve_petsc_system_impl", fail_impl)
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

    monkeypatch.setattr(linalg_iterative, "_import_petsc", lambda: FakePETSc)
    matrix, rhs = _identity_problem()
    with pytest.raises(RuntimeError, match="matrix construction failed"):
        solve_petsc_system(matrix, rhs)

    assert destroyed == ["matrix"]


def test_amgx_retry_count_is_bounded_before_optional_runtime_import() -> None:
    from hybridge.linalg.amgx.device_solver import solve_reduced_system_amgx_device

    retries = tuple({"label": f"retry-{index}"} for index in range(8))
    with pytest.raises(ValueError, match="at most 8 bounded attempts"):
        solve_reduced_system_amgx_device(None, retry_attempts=retries)


class _DeviceScalar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class _FakeCupy:
    asarray = staticmethod(np.asarray)

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


@pytest.mark.parametrize("primary_throws", (False, True))
def test_raw_amgx_primary_retry_reuses_only_a_live_solver(monkeypatch, primary_throws):
    import hybridge.linalg.amgx.device_solver as raw_amgx
    import hybridge.linalg.amgx.device_solver as amgx_device_solver
    from types import SimpleNamespace

    primary = SimpleNamespace(closed=False, is_setup=True)
    calls = []

    def attempt(_assembly, **kwargs):
        calls.append(kwargs)
        first = len(calls) == 1
        if first and primary_throws:
            primary.closed = True
            raise RuntimeError("primary iteration breakdown")
        residual = 1.0 if first else 1.0e-8
        result = _diagnostic_result(
            solver_residual=residual, solver_target=1.0e-6,
            physical_residual=residual, physical_target=1.0e-6,
        )
        finalize_solve_result(result, backend="pyamgx-device", backend_success=True)
        return result, np.ones(2)

    monkeypatch.setattr(raw_amgx, "require_cupy", lambda: _FakeCupy)
    monkeypatch.setattr(amgx_device_solver, "_solve_reduced_system_amgx_device_once", attempt)
    result, _ = amgx_device_solver.solve_reduced_system_amgx_device(
        _FakeAmgxAssembly(), reusable_solver=primary,
        retry_attempts=({"reuse_primary_solver": True, "use_initial_guess": False},),
    )

    assert result.converged and len(calls) == 2
    assert calls[0]["reusable_solver"] is primary
    assert calls[1]["reusable_solver"] is (None if primary_throws else primary)
    assert result.amgx_attempts[-1]["primary_solver_reused"] is (not primary_throws)


@pytest.mark.parametrize("override", (
    {"config": {"solver": {"solver": "FGMRES"}}},
    {"scale_system": "left"},
))
def test_raw_amgx_primary_reuse_rejects_changed_solver_or_matrix_scaling(override):
    from hybridge.linalg.amgx.device_solver import solve_reduced_system_amgx_device

    with pytest.raises(ValueError, match="primary configuration and scaling"):
        solve_reduced_system_amgx_device(
            None, config={"solver": {"solver": "PCGF"}}, scale_system=False,
            retry_attempts=({"reuse_primary_solver": True, **override},),
        )


def test_raw_amgx_scalar_only_smoother_reaches_retry_before_native_upload(monkeypatch):
    from types import SimpleNamespace
    import hybridge.linalg.amgx.device_solver as raw_amgx
    import hybridge.linalg.amgx.device_solver as amgx_device_solver

    config = {"solver": {"solver": "PCGF", "preconditioner": {
        "solver": "AMG", "classical_bsr_hierarchy": "scalar_expand",
        "smoother": {"solver": "CHEBYSHEV_POLY"},
    }}}
    primary = SimpleNamespace(closed=False, config_dict=config, cp=_FakeCupy, pyamgx=None)
    primary.close = lambda **_kwargs: setattr(primary, "closed", True)
    # Deliberately no matrix-upload API: preflight must fail before using one.
    matrix = SimpleNamespace(shape=(14, 14), block_size=7)
    calls = []

    def attempt(_assembly, **kwargs):
        calls.append(kwargs)
        if not kwargs["scalarize_bsr"]:
            amgx_device_solver.PyAMGXCsrDeviceSolver.setup(primary, matrix)
            raise AssertionError("scalar-only smoother unexpectedly accepted face blocks")
        amgx_device_solver._validate_amgx_block_configuration(config, 1)
        result = _diagnostic_result(
            solver_residual=1.0e-8, solver_target=1.0e-6,
            physical_residual=1.0e-8, physical_target=1.0e-6,
        )
        finalize_solve_result(result, backend="pyamgx-device", backend_success=True)
        return result, np.ones(2)

    monkeypatch.setattr(raw_amgx, "require_cupy", lambda: _FakeCupy)
    monkeypatch.setattr(amgx_device_solver, "_solve_reduced_system_amgx_device_once", attempt)
    result, _ = amgx_device_solver.solve_reduced_system_amgx_device(
        _FakeAmgxAssembly(), config=config,
        retry_attempts=({"scalarize_bsr": True},),
    )

    assert primary.closed and result.converged and len(calls) == 2
    assert "scalar-only CHEBYSHEV_POLY" in result.amgx_attempts[0]["error"]
    assert "block size 7" in result.amgx_attempts[0]["error"]


@pytest.mark.parametrize("seed_norm", (np.nan, np.inf))
def test_raw_amgx_nonfinite_seed_residual_does_not_poison_best_candidate(
    monkeypatch, seed_norm,
):
    import hybridge.linalg.amgx.device_solver as raw_amgx
    import hybridge.linalg.amgx.device_solver as amgx_device_solver
    from types import SimpleNamespace

    assembly = SimpleNamespace(rhs=np.ones(2))
    seed = np.full(2, 100.0)
    first_candidate = np.full(2, 0.5)
    calls = []

    def attempt(_assembly, **kwargs):
        calls.append(kwargs)
        residual = 1.0 if len(calls) == 1 else 1.0e-8
        result = _diagnostic_result(
            solver_residual=residual, solver_target=1.0e-6,
            physical_residual=residual, physical_target=1.0e-6,
        )
        finalize_solve_result(result, backend="pyamgx-device", backend_success=True)
        return result, first_candidate.copy()

    monkeypatch.setattr(raw_amgx, "require_cupy", lambda: _FakeCupy)
    monkeypatch.setattr(raw_amgx, "require_cupyx_sparse", lambda: None)
    monkeypatch.setattr(raw_amgx, "_assembly_device_csr_matrix", lambda *_args: None)
    monkeypatch.setattr(raw_amgx, "_device_compressed_matvec", lambda *_args: np.zeros(2))
    monkeypatch.setattr(amgx_device_solver, "_residual_stats_cp", lambda *_args, **_kwargs: (
        seed_norm, 1.0, seed_norm, 1.0e-6,
    ))
    monkeypatch.setattr(amgx_device_solver, "_solve_reduced_system_amgx_device_once", attempt)
    result, _ = amgx_device_solver.solve_reduced_system_amgx_device(
        assembly, retry_seed_solution=seed,
        retry_attempts=({"use_best_solution": True},),
    )

    assert result.converged and len(calls) == 2
    np.testing.assert_array_equal(calls[1]["initial_guess"], first_candidate)
    assert not hasattr(result, "amgx_retry_seed_label")


@pytest.mark.parametrize("correction", (False, True))
def test_raw_amgx_retains_upstream_seed_until_physical_residual_passes(
    monkeypatch, correction,
):
    from dataclasses import dataclass
    import hybridge.linalg.amgx.device_solver as raw_amgx
    import hybridge.linalg.amgx.device_solver as amgx_device_solver

    @dataclass
    class Assembly:
        rhs: np.ndarray

    matrix = np.diag([2.0, 3.0])
    assembly = Assembly(rhs=np.array([2.0, 3.0]))
    seed = np.full(2, 0.9)
    expected_seed = seed.copy()
    calls = []

    def residual_stats(residual, rhs, *, rtol, atol):
        norm = float(np.linalg.norm(residual))
        rhs_norm = float(np.linalg.norm(rhs))
        return norm, rhs_norm, norm / max(rhs_norm, 1.0e-300), max(atol, rtol * rhs_norm)

    def attempt(solve_assembly, **kwargs):
        calls.append(kwargs)
        index = len(calls)
        if index == 1:
            # The upstream solver may reuse its output workspace. The retained
            # best candidate must survive that overwrite and this worse solve.
            seed.fill(-99.0)
            solution = np.zeros(2)
        elif correction:
            np.testing.assert_allclose(
                solve_assembly.rhs, assembly.rhs - matrix @ expected_seed,
            )
            assert kwargs["initial_guess"] is None
            solution = np.zeros(2) if index == 2 else np.ones(2) - expected_seed
        else:
            np.testing.assert_array_equal(kwargs["initial_guess"], expected_seed)
            assert not np.shares_memory(kwargs["initial_guess"], seed)
            solution = expected_seed.copy() if index == 2 else np.ones(2)
        physical = float(np.linalg.norm(matrix @ solution - solve_assembly.rhs))
        result = _diagnostic_result(
            solver_residual=1.0e-13, solver_target=1.0e-6,
            physical_residual=physical,
            physical_target=1.0e-6 * float(np.linalg.norm(solve_assembly.rhs)),
        )
        finalize_solve_result(result, backend="pyamgx-device", backend_success=True)
        return result, solution

    monkeypatch.setattr(raw_amgx, "require_cupy", lambda: _FakeCupy)
    monkeypatch.setattr(raw_amgx, "require_cupyx_sparse", lambda: None)
    monkeypatch.setattr(raw_amgx, "_assembly_device_csr_matrix", lambda *_args: matrix)
    monkeypatch.setattr(raw_amgx, "_device_compressed_matvec", lambda mat, x, *_args: mat @ x)
    monkeypatch.setattr(amgx_device_solver, "_residual_stats_cp", residual_stats)
    monkeypatch.setattr(amgx_device_solver, "_solve_reduced_system_amgx_device_once", attempt)
    retry = {"use_best_solution": True, "residual_correction": correction}
    result, solution = amgx_device_solver.solve_reduced_system_amgx_device(
        assembly, retry_seed_solution=seed, retry_seed_label="native-best",
        check_rtol=1.0e-6, retry_attempts=(retry, retry),
    )

    assert result.converged and len(calls) == 3
    np.testing.assert_allclose(solution, np.ones(2))
    assert [entry["success"] for entry in result.amgx_attempts] == [False, False, True]
    assert result.amgx_retry_seed_label == "native-best"
    assert result.amgx_retry_seed_physical_residual == pytest.approx(
        np.linalg.norm(matrix @ expected_seed - assembly.rhs),
    )


def test_raw_amgx_retry_wrapper_avoids_full_matrix_backup(monkeypatch) -> None:
    import hybridge.linalg.amgx.device_solver as raw_amgx
    import hybridge.linalg.amgx.device_solver as amgx_device_solver

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
        amgx_device_solver,
        "_solve_reduced_system_amgx_device_once",
        lambda *_args, **_kwargs: (result, solution),
    )

    returned_result, returned_solution = amgx_device_solver.solve_reduced_system_amgx_device(
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
    import hybridge.linalg.amgx.device_solver as raw_amgx
    import hybridge.linalg.amgx.device_solver as amgx_device_solver

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
    monkeypatch.setattr(amgx_device_solver, "PyAMGXCsrDeviceSolver", FakeReusableSolver)
    monkeypatch.setattr(
        amgx_device_solver, "_solve_reduced_system_amgx_device_once", fake_attempt
    )
    cache = {}
    retry = ({
        "label": "robust-scaled",
        "config": {"solver": {"solver": "FGMRES"}},
        "scalarize_bsr": True,
        "reuse_preconditioner": True,
        "solver_cache_key": "fgmres-dilu",
    },)

    first, _ = amgx_device_solver.solve_reduced_system_amgx_device(
        _FakeAmgxAssembly(), retry_attempts=retry, retry_solver_cache=cache
    )
    second, _ = amgx_device_solver.solve_reduced_system_amgx_device(
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
    import hybridge.linalg.amgx.device_solver as raw_amgx
    import hybridge.linalg.amgx.device_solver as amgx_device_solver

    calls = []

    def fail_with_oom(*_args, **_kwargs):
        calls.append("attempt")
        raise _FakeAMGXNoMemoryError()

    monkeypatch.setattr(raw_amgx, "require_cupy", lambda: _FakeCupy)
    monkeypatch.setattr(
        amgx_device_solver,
        "_solve_reduced_system_amgx_device_once",
        fail_with_oom,
    )

    with pytest.raises(LinearSolveCapacityError) as raised:
        amgx_device_solver.solve_reduced_system_amgx_device(
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
    import hybridge.linalg.amgx.device_solver as raw_amgx
    import hybridge.linalg.amgx.device_solver as amgx_device_solver

    calls = []

    def fail_generically(*_args, **_kwargs):
        calls.append("attempt")
        raise RuntimeError("transient backend failure")

    monkeypatch.setattr(raw_amgx, "require_cupy", lambda: _FakeCupy)
    monkeypatch.setattr(
        amgx_device_solver,
        "_solve_reduced_system_amgx_device_once",
        fail_generically,
    )

    with pytest.raises(LinearSolveConvergenceError, match="exhausted 3 bounded attempts"):
        amgx_device_solver.solve_reduced_system_amgx_device(
            _FakeAmgxAssembly(),
            retry_attempts=({"label": "retry-1"}, {"label": "retry-2"}),
            scale_system=False,
        )

    assert calls == ["attempt", "attempt", "attempt"]


def test_raw_amgx_capacity_error_reports_structured_memory() -> None:
    import hybridge.linalg.amgx.device_solver as raw_amgx

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
    import hybridge.linalg.amgx.device_solver as amgx_device_solver

    destroyed = []

    class FakeObject:
        def __init__(self, label, *, fail=False):
            self.label = label
            self.fail = fail

        def destroy(self):
            destroyed.append(self.label)
            if self.fail:
                raise RuntimeError(f"{self.label} destroy failed")

    solver = amgx_device_solver.PyAMGXCsrDeviceSolver.__new__(
        amgx_device_solver.PyAMGXCsrDeviceSolver
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


@pytest.mark.parametrize("history", [0, 1])
def test_raw_amgx_config_respects_explicit_residual_history_without_mutating_input(history) -> None:
    from hybridge.linalg.amgx.device_solver import _amgx_config_for_solve

    config = {"solver": {"solver": "FGMRES", "store_res_history": history}}
    normalized = _amgx_config_for_solve(config=config)

    assert normalized["solver"]["monitor_residual"] == 1
    assert normalized["solver"]["store_res_history"] == history
    assert config["solver"]["store_res_history"] == history


def test_raw_amgx_config_reserves_native_iteration_table_for_level_three() -> None:
    from hybridge.linalg.amgx.device_solver import _amgx_config_for_solve

    config = {"solver": {"solver": "BICGSTAB", "print_solve_stats": 0}}

    detailed = _amgx_config_for_solve(config=config, verbose=2)
    fully_verbose = _amgx_config_for_solve(config=config, verbose=3)

    assert detailed["solver"]["obtain_timings"] == 1
    assert detailed["solver"]["print_solve_stats"] == 0
    assert fully_verbose["solver"]["print_solve_stats"] == 1
    assert fully_verbose["solver"]["print_solve_stats_interval"] == 1
    assert "obtain_timings" not in fully_verbose["solver"]


def test_raw_amgx_scaled_solver_validation_uses_configured_relative_tolerance() -> None:
    from hybridge.linalg.amgx.device_solver import _amgx_relative_residual_check_rtol

    relative = {
        "solver": {"convergence": "RELATIVE_INI_CORE", "tolerance": 1.0e-8}
    }
    absolute = {
        "solver": {"convergence": "ABSOLUTE", "tolerance": 5.0e-9}
    }

    assert _amgx_relative_residual_check_rtol(relative, 1.0e-11) == 1.0e-8
    assert _amgx_relative_residual_check_rtol(absolute, 1.0e-11) == 1.0e-11


def test_raw_amgx_retry_exhaustion_raises_stable_convergence_error(monkeypatch) -> None:
    import hybridge.linalg.amgx.device_solver as raw_amgx
    import hybridge.linalg.amgx.device_solver as amgx_device_solver

    result = _diagnostic_result(
        solver_residual=1.0,
        solver_target=1.0e-6,
        physical_residual=1.0,
        physical_target=1.0e-6,
    )
    finalize_solve_result(result, backend="pyamgx-device", backend_success=True)
    monkeypatch.setattr(raw_amgx, "require_cupy", lambda: _FakeCupy)
    monkeypatch.setattr(
        amgx_device_solver,
        "_solve_reduced_system_amgx_device_once",
        lambda *_args, **_kwargs: (result, np.ones(2)),
    )

    with pytest.raises(LinearSolveConvergenceError) as raised:
        amgx_device_solver.solve_reduced_system_amgx_device(
            _FakeAmgxAssembly(),
            retry_attempts=({"label": "retry"},),
        )

    assert raised.value.result is result
    assert result.amgx_attempt_count == 2


def test_raw_amgx_nonraising_retry_returns_last_nonfinite_result(monkeypatch) -> None:
    import hybridge.linalg.amgx.device_solver as raw_amgx
    import hybridge.linalg.amgx.device_solver as amgx_device_solver

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
        amgx_device_solver,
        "_solve_reduced_system_amgx_device_once",
        lambda *_args, **_kwargs: (result, solution),
    )

    returned_result, returned_solution = amgx_device_solver.solve_reduced_system_amgx_device(
        _FakeAmgxAssembly(),
        retry_attempts=({"label": "retry"},),
        raise_on_nonconvergence=False,
    )

    assert returned_result is result
    assert returned_solution is solution
    assert result.amgx_attempt_count == 2



def test_convergence_contract_is_available_from_public_packages() -> None:
    assert hybridge.LinearSolveCapacityError is LinearSolveCapacityError
    assert hybridge.LinearSolveConvergenceError is LinearSolveConvergenceError
    assert hybridge.LinearSolveError is linalg_results.LinearSolveError
    assert "SolveStatus" in hybridge.__all__
    assert issubclass(LinearSolveConvergenceError, RuntimeError)


@pytest.mark.parametrize("accepted_attempt", range(4))
@pytest.mark.parametrize("failure", ("breakdown", "physical-residual"))
def test_guiding_center_retries_precondition_bicgstab_before_dilu(
    monkeypatch, accepted_attempt, failure
) -> None:
    import hybridge.linalg.amgx.device_solver as raw_amgx
    import hybridge.linalg.amgx.device_solver as amgx_device_solver
    from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
    from scripts.guiding_center.runtime.configuration import _make_transport_options

    options = _make_transport_options(
        preset_by_key("euler_vortex_gas_p6_50k_dt001_t50_raw_cuda_bsr"), "zero-flux"
    )
    assembly = _FakeAmgxAssembly()
    calls = []
    guess = np.ones(2)

    def fake_attempt(solve_assembly, **kwargs):
        assert solve_assembly is assembly
        index = len(calls)
        calls.append(kwargs)
        if index < accepted_attempt and failure == "breakdown":
            raise RuntimeError("BiCGSTAB breakdown")
        # AMGX's success and small solver residual must not hide physical failure.
        result = _diagnostic_result(
            solver_residual=1.0e-13, solver_target=1.0e-11,
            physical_residual=1.0e-13 if index == accepted_attempt else 1.0,
            physical_target=1.0e-11,
        )
        finalize_solve_result(result, backend="pyamgx-device", backend_success=True)
        return result, np.ones(2)

    monkeypatch.setattr(raw_amgx, "require_cupy", lambda: _FakeCupy)
    monkeypatch.setattr(amgx_device_solver, "_solve_reduced_system_amgx_device_once", fake_attempt)
    result, _ = amgx_device_solver.solve_reduced_system_amgx_device(
        assembly, config=options.amgx_config, retry_attempts=options.amgx_retry_attempts,
        initial_guess=guess, check_rtol=options.solver_rtol, atol=options.solver_atol,
    )

    assert result.converged
    assert len(calls) == result.amgx_attempt_count == accepted_attempt + 1
    expected = (
        ("BICGSTAB", None), ("PBICGSTAB", "JACOBI_L1"),
        ("PBICGSTAB", "BLOCK_JACOBI"), ("FGMRES", "MULTICOLOR_DILU"),
    )
    for index, call in enumerate(calls):
        solver = call["config"]["solver"]
        assert (solver["solver"], solver.get("preconditioner", {}).get("solver")) == expected[index]
        assert call["scalarize_bsr"] is (index == 3)
        assert call["scale_system"] == "left"
        assert call["initial_guess"] is (guess if index == 0 else None)
        assert call["check_rtol"] == options.solver_rtol
        assert call["atol"] == options.solver_atol
        if index in (1, 2):
            assert call["reusable_solver"] is None
            assert call["replace_reusable_coefficients"] is False


def test_raw_amgx_divergence_advances_retry_and_records_exit(monkeypatch, capsys) -> None:
    import hybridge.linalg.amgx.device_solver as raw_amgx
    import hybridge.linalg.amgx.device_solver as amgx_device_solver

    calls = []
    def attempt(_assembly, **kwargs):
        calls.append(kwargs)
        diverged = len(calls) == 1
        residual = 1e8 if diverged else 1e-13
        result = _diagnostic_result(
            solver_residual=residual, solver_target=1e-11,
            physical_residual=residual, physical_target=1e-11,
        )
        result.iteration_count = 15 if diverged else 2
        finalize_solve_result(
            result, backend="pyamgx-device",
            backend_info="diverged" if diverged else "success",
            backend_success=not diverged,
        )
        return result, np.ones(2)

    monkeypatch.setattr(raw_amgx, "require_cupy", lambda: _FakeCupy)
    monkeypatch.setattr(amgx_device_solver, "_solve_reduced_system_amgx_device_once", attempt)
    result, _ = amgx_device_solver.solve_reduced_system_amgx_device(
        _FakeAmgxAssembly(), retry_attempts=({"label": "recovery"},), verbose=1,
    )
    assert result.converged and len(calls) == 2
    rejected = result.amgx_attempts[0]
    assert rejected["status"] == rejected["backend_info"] == "diverged"
    assert rejected["iterations"] == 15
    assert rejected["failure_reason"] == "backend-divergence"
    assert "status=diverged iterations=15" in capsys.readouterr().out
