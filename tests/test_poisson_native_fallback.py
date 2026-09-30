"""CPU-only checks of native Poisson acceptance and AMGX handoff.

The real dispatch and residual gate run against a two-by-two cached operator.
GPU assembly, Krylov solvers and field reconstruction are mocked; importing or
running these tests does not compile kernels or perform time integration.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import hdgfem.runtime.optional as runtime_optional
from hdgfem.backends import advection_cuda, cupy, diffusion_cupy, diffusion_raw_cuda
import hdgfem.core.device as core_device
from hdgfem.linalg.face_hp_multigrid import FaceBlockHpMgPcgResult
from hdgfem.solvers import diffusion_reaction


@pytest.fixture(params=("standard", "fast"))
def poisson_handoff(monkeypatch, request):
    matrix = np.diag([2.0, 4.0])
    rhs = np.array([2.0, 8.0])
    exact_solution = np.array([1.0, 2.0])
    options = diffusion_reaction.DiffusionReactionHDGOptions(
        stabilization=1.0, solver="fb-hp-mg-pcg", solver_rtol=1e-11,
        scale_system=False, boundary_mode="eliminate",
        trace_basis="legendre-modal", raw_matrix_format="bsr",
        raw_block_size=128, verbose=False,
        fb_hp_mg_preconditioner_policy=request.param,
    )
    space = SimpleNamespace(
        order=1, el_dof=1, mesh=SimpleNamespace(num_tri=1, num_edg=1),
    )
    cspace = SimpleNamespace(device_id=0)
    raw = SimpleNamespace(
        d0_reference=None, d1_reference=None, face_element_mass=None,
    )
    assembly = SimpleNamespace(
        rhs=rhs, raw_assembly=raw, boundary_trace=None, source_rhs=None,
        timings={},
    )
    native_solver = SimpleNamespace(
        setup_seconds=0.0, workspace_bytes=32, symmetry_defect=0.0,
        positive_curvature=1.0, solve=Mock(), close=Mock(),
    )
    reaction = object()
    operator_key = (
        id(space), "legendre-modal", "bsr", 128, 1.0, id(reaction), "none",
    )
    solver = diffusion_reaction.DiffusionReactionHDGSolver.__new__(
        diffusion_reaction.DiffusionReactionHDGSolver,
    )
    solver.options = options
    solver.space = space
    solver.reaction = reaction
    solver._raw_cuda_assembly_cache = assembly
    solver._raw_cuda_operator_key = operator_key
    solver._raw_cuda_rhs_valid = True
    solver._raw_cuda_last_trace_reduced = None
    solver._raw_cuda_fb_hp_mg_failed_key = None
    solver._raw_cuda_fb_hp_mg_solver = native_solver
    solver._raw_cuda_fb_hp_mg_solver_key = (operator_key, rhs.size, request.param)
    solver._raw_cuda_amgx_solver = None
    solver._raw_cuda_amgx_retry_solver_cache = {}

    # Preserve the actual residual statistics and norm comparison, replacing
    # only the GPU array operations by NumPy operations on the tiny operator.
    cp = SimpleNamespace(
        asarray=np.asarray, ascontiguousarray=np.ascontiguousarray,
        cuda=SimpleNamespace(get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None)),
        linalg=SimpleNamespace(norm=lambda a: SimpleNamespace(get=lambda: np.linalg.norm(a))),
    )
    monkeypatch.setattr(cupy, "require_cupy", lambda: cp)
    monkeypatch.setattr(runtime_optional, "require_cupy", lambda: cp)
    monkeypatch.setattr(advection_cuda, "require_cupy", lambda: cp)
    monkeypatch.setattr(cupy, "require_cupyx_sparse", lambda: None)
    monkeypatch.setattr(runtime_optional, "require_cupyx_sparse", lambda: None)
    monkeypatch.setattr(diffusion_cupy, "as_cupy_space", lambda _: cspace)
    monkeypatch.setattr(core_device, "as_cupy_space", lambda _: cspace)
    monkeypatch.setattr(diffusion_cupy, "build_trace_reference", lambda *args: None)
    monkeypatch.setattr(diffusion_reaction, "audit_arrays", lambda *args: None)
    monkeypatch.setattr(advection_cuda, "_assembly_device_csr_matrix", lambda *args: matrix)
    physical_matvec = Mock(side_effect=lambda a, x, *args: a @ x)
    monkeypatch.setattr(advection_cuda, "_device_compressed_matvec", physical_matvec)

    amgx_result = SimpleNamespace(backend="pyamgx-device", converged=True)
    amgx = Mock(return_value=(amgx_result, exact_solution.copy()))
    monkeypatch.setattr(advection_cuda, "solve_reduced_system_amgx_device", amgx)
    monkeypatch.setattr(
        advection_cuda, "PyAMGXCsrDeviceSolver",
        Mock(return_value=SimpleNamespace(closed=False, close=Mock())),
    )
    reconstructed = Mock(side_effect=lambda trace, *args: trace.copy())
    monkeypatch.setattr(advection_cuda, "reconstruct_trace_cupy", reconstructed)
    monkeypatch.setattr(
        diffusion_raw_cuda, "reconstruct_projected_diffusion_field_raw_cuda",
        lambda **kwargs: (np.zeros((1, 1)), np.zeros((1, 3)), 0.0),
    )
    monkeypatch.setattr(core_device, "field_from_cupy_coefficients", lambda space, coefficients, **kwargs: coefficients)
    monkeypatch.setattr(diffusion_reaction, "VectorDGField", lambda components, **kwargs: components)

    def run_native(solution, *, converged):
        # A native convergence claim may differ from the original-matrix gate
        # because its Krylov action operates in orthonormal coordinates.
        native_solver.solve.return_value = FaceBlockHpMgPcgResult(
            solution=solution, converged=converged, iterations=40,
            residual_norm=1e-13 if converged else 1e-4,
            rhs_norm=float(np.linalg.norm(rhs)),
            relative_residual=1e-14 if converged else 1e-5,
            residual_over_initial=1e-14 if converged else 1e-5,
            target=options.solver_rtol * np.linalg.norm(rhs),
            history=(float(np.linalg.norm(rhs)),), elapsed_seconds=0.25,
            best_iteration=39, terminal_residual_norm=1e-3,
            true_residual_check_count=5, returned_best_iterate=True,
        )
        return solver._solve_raw_cuda_device_amgx()

    return SimpleNamespace(
        run_native=run_native, solver=solver, native_solver=native_solver,
        amgx=amgx, amgx_result=amgx_result, reconstructed=reconstructed,
        physical_matvec=physical_matvec, exact_solution=exact_solution, matrix=matrix,
    )


@pytest.mark.parametrize(
    ("converged", "reason"),
    [(False, "true-residual convergence gate"), (True, "original-matrix residual gate")],
)
def test_rejected_native_result_seeds_amgx_and_is_never_reconstructed(
    poisson_handoff, converged, reason,
):
    h = poisson_handoff
    native_best = np.array([0.9, 1.8])
    result = h.run_native(native_best, converged=converged)

    h.amgx.assert_called_once()
    attempt = h.amgx.call_args.kwargs
    assert attempt["initial_guess"] is native_best
    assert attempt["retry_seed_solution"] is native_best
    assert attempt["retry_seed_label"] == "native-fb-hp-mg-best"
    assert attempt["raise_on_nonconvergence"] is True
    assert result.global_solve_result is h.amgx_result
    np.testing.assert_array_equal(result.trace_reduced_device, h.exact_solution)
    np.testing.assert_array_equal(h.reconstructed.call_args.args[0], h.exact_solution)
    np.testing.assert_array_equal(h.solver._raw_cuda_last_trace_reduced, h.exact_solution)
    h.native_solver.close.assert_called_once()
    assert h.solver._raw_cuda_fb_hp_mg_solver is None
    assert reason in h.solver._raw_cuda_fb_hp_mg_failure_reason
    assert result.timings.details["solve.fb_hp_mg.fallback"] == 1.0
    assert result.timings.details["solve.fb_hp_mg.best_iteration"] == 39.0
    assert result.timings.solve >= 0.25
    assert h.physical_matvec.call_count == int(converged)


def test_native_solution_requires_and_reports_original_matrix_residual(poisson_handoff):
    h = poisson_handoff
    result = h.run_native(h.exact_solution.copy(), converged=True)

    h.amgx.assert_not_called()
    h.native_solver.close.assert_not_called()
    h.physical_matvec.assert_called_once()
    solved = result.global_solve_result
    assert solved.backend == "fb-hp-mg-pcg"
    assert solved.converged
    assert solved.physical_residual_norm == 0.0
    assert solved.physical_relative_residual_norm == 0.0
    assert solved.physical_residual_target_met
    # Keep both reported quantities: independent physical residual and the
    # native Krylov residual, rather than copying the native value into both.
    assert solved.solver_residual_norm == 1e-13
    assert result.timings.details["solve.fb_hp_mg.fallback"] == 0.0
    np.testing.assert_array_equal(h.reconstructed.call_args.args[0], h.exact_solution)


def test_native_syntax_error_is_not_hidden_by_numerical_fallback(poisson_handoff):
    h = poisson_handoff
    h.native_solver.solve.side_effect = IndentationError("unexpected indent")

    with pytest.raises(IndentationError, match="unexpected indent"):
        h.run_native(h.exact_solution.copy(), converged=True)

    h.amgx.assert_not_called()
    h.reconstructed.assert_not_called()
    assert h.solver._raw_cuda_fb_hp_mg_failed_key is None
    assert h.solver._raw_cuda_last_trace_reduced is None


def test_native_dispatch_supplies_original_assembly_action_and_rechecks_result(poisson_handoff):
    h = poisson_handoff
    probe = np.array([0.3, -0.7])

    def solve(rhs, **kwargs):
        original_action = kwargs['assembly_matvec']
        assert callable(original_action)
        np.testing.assert_array_equal(original_action(probe), h.matrix @ probe)
        return h.native_solver.solve.return_value

    h.native_solver.solve.side_effect = solve
    result = h.run_native(h.exact_solution.copy(), converged=True)

    h.amgx.assert_not_called()
    assert result.global_solve_result.converged
    # One action exercises the callable passed into Krylov; a separate action
    # checks its returned solution before any field reconstruction is allowed.
    assert h.physical_matvec.call_count == 2
    np.testing.assert_array_equal(h.physical_matvec.call_args_list[0].args[1], probe)
    np.testing.assert_array_equal(h.physical_matvec.call_args_list[1].args[1], h.exact_solution)
