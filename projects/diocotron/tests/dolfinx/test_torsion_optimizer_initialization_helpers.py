"""Focused tests for optimizer initialization/replay helper interfaces."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import basix
import numpy as np
import pytest
from mpi4py import MPI
import ufl


pytest.importorskip("dolfinx")
from dolfinx import fem, mesh  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parents[4] / "projects/diocotron/dolfinx"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import projects.diocotron.dolfinx.torsion.equilibrium.window_fit as window_fit  # noqa: E402
import projects.diocotron.dolfinx.torsion.search.brute_force as brute_force  # noqa: E402
import projects.diocotron.dolfinx.torsion.optimization.reduced as reduced  # noqa: E402
import projects.diocotron.dolfinx.torsion.initialization.fractional_target as direct_fractional  # noqa: E402
import projects.diocotron.dolfinx.torsion.initialization.frozen_frontier_run as frozen_frontier_runner  # noqa: E402
import projects.diocotron.dolfinx.torsion.optimization.homotopy as optimizer  # noqa: E402
from projects.diocotron.dolfinx.torsion.optimization.homotopy import (  # noqa: E402
    INEXACT_NEWTON_CSV_FIELDS,
    compare_inexact_newton_snapshot,
    initial_equilibrium_status_is_acceptable,
    model_reduction_is_sufficient,
    parse_args,
)


def test_initializer_cli_is_backward_compatible() -> None:
    defaults = parse_args([])
    assert defaults.init_search == "full"
    assert defaults.init_fallback == "none"
    assert defaults.initial_equilibrium is None
    assert defaults.run_inexact_newton_study is False
    assert defaults.inexact_newton_tolerances == "1e-2,1e-3,1e-5,1e-7,adaptive"
    assert defaults.inexact_newton_reference_tol == pytest.approx(1.0e-12)
    assert defaults.inexact_newton_max_snapshots == 4
    assert defaults.inexact_newton_output is None
    assert defaults.save_terminal_log is False

    configured = parse_args(
        [
            "--init-search", "fast",
            "--init-fallback", "window-fit",
            "--run-inexact-newton-study",
            "--inexact-newton-tolerances", "1e-3,adaptive",
            "--inexact-newton-max-snapshots", "2",
            "--save-terminal-log",
        ]
    )
    assert configured.init_search == "fast"
    assert configured.init_fallback == "window-fit"
    assert configured.run_inexact_newton_study is True
    assert configured.inexact_newton_tolerances == "1e-3,adaptive"
    assert configured.inexact_newton_max_snapshots == 2
    assert configured.save_terminal_log is True


def test_window_fit_publication_frames_can_hide_mesh_edges() -> None:
    defaults = window_fit.parse_args([])
    publication = window_fit.parse_args(["--no-plot-mesh-edges"])

    assert defaults.plot_mesh_edges is True
    assert publication.plot_mesh_edges is False


def test_active_reduced_optimizer_terminal_log_cli_is_opt_in() -> None:
    assert reduced.parse_args([]).save_terminal_log is False
    assert reduced.parse_args(["--save-terminal-log"]).save_terminal_log is True
    assert direct_fractional.parse_args(["--save-terminal-log"]).save_terminal_log is True
    assert frozen_frontier_runner.parse_args(["--save-terminal-log"]).save_terminal_log is True



def test_reduced_optimizer_supports_sharp_target_publication_frames() -> None:
    args = reduced.parse_args([
        "--alphaT1", "0.6",
        "--alphaT2", "0.7",
        "--eps-t-ratio", "0",
        "--no-plot-mesh-edges",
        "--threshold-cap-mode", "torsion",
    ])
    params = reduced.params_from_args(args)

    assert params.eps_t_ratio == 0.0
    assert args.plot_mesh_edges is False
    assert args.threshold_cap_mode == "torsion"
    with pytest.raises(ValueError, match="nonnegative"):
        reduced.params_from_args(reduced.parse_args(["--eps-t-ratio", "-0.01"]))


def test_reduced_optimizer_jaccard_oscillation_policy() -> None:
    defaults = reduced.parse_args([])
    assert defaults.jaccard_oscillation_stop is False
    assert defaults.require_inner_newton_convergence is False
    assert defaults.energy_primer is False
    assert defaults.newton_stall_forecast is False
    assert defaults.picard_spectrum is False
    assert defaults.trust_radius == pytest.approx(0.5)
    assert defaults.trust_radius_max == pytest.approx(0.95)

    configured = reduced.parse_args([
        "--jaccard-oscillation-stop",
        "--jaccard-oscillation-min-accepted", "5",
        "--jaccard-oscillation-patience", "4",
        "--inner-newton-tol", "1e-11",
        "--require-inner-newton-convergence",
    ])
    reduced.validate_args(configured)
    assert configured.jaccard_oscillation_stop is True
    assert configured.require_inner_newton_convergence is True

    assert not reduced.jaccard_oscillation_detected(
        [0.20, 0.30, 0.40, 0.50, 0.60],
        patience=4,
    )
    assert reduced.jaccard_oscillation_detected(
        [0.40, 0.50, 0.45, 0.51, 0.46],
        patience=4,
    )
    assert not reduced.jaccard_oscillation_detected(
        [0.50, 0.50, 0.50, 0.50, 0.50],
        patience=4,
    )
    with pytest.raises(ValueError, match="patience"):
        reduced.validate_args(
            reduced.parse_args(["--jaccard-oscillation-patience", "2"])
        )
    with pytest.raises(ValueError, match="trust-radius <= trust-radius-max"):
        reduced.validate_args(
            reduced.parse_args(
                ["--trust-radius", "0.9", "--trust-radius-max", "0.8"]
            )
        )
    with pytest.raises(ValueError, match="trust-radius-max < 1"):
        reduced.validate_args(reduced.parse_args(["--trust-radius-max", "1.0"]))


def test_reduced_optimizer_energy_primer_and_stall_forecast_cli() -> None:
    configured = reduced.parse_args([
        "--energy-primer",
        "--energy-primer-tol", "2e-5",
        "--energy-primer-max-it", "3",
        "--energy-primer-min-residual-reduction", "0.1",
        "--newton-stall-forecast",
        "--newton-stall-window", "5",
        "--newton-stall-patience", "2",
        "--picard-spectrum",
        "--picard-spectrum-eig-tol", "2e-6",
        "--picard-spectrum-eig-max-it", "200",
    ])
    reduced.validate_args(configured)

    assert configured.energy_primer is True
    assert configured.energy_primer_tol == pytest.approx(2.0e-5)
    assert configured.energy_primer_max_it == 3
    assert configured.energy_primer_min_residual_reduction == pytest.approx(0.1)
    assert configured.newton_stall_forecast is True
    assert configured.newton_stall_window == 5
    assert configured.newton_stall_patience == 2
    assert configured.picard_spectrum is True
    assert configured.picard_spectrum_eig_tol == pytest.approx(2.0e-6)
    assert configured.picard_spectrum_eig_max_it == 200

    with pytest.raises(ValueError, match="stall-window"):
        reduced.validate_args(reduced.parse_args(["--newton-stall-window", "1"]))
    with pytest.raises(ValueError, match="energy-primer-max-it"):
        reduced.validate_args(reduced.parse_args(["--energy-primer-max-it", "0"]))
    with pytest.raises(ValueError, match="energy-primer-max-it"):
        reduced.validate_args(reduced.parse_args(["--energy-primer-max-it", "4"]))
    with pytest.raises(ValueError, match="requires --newton-stall-forecast"):
        reduced.validate_args(reduced.parse_args(["--energy-primer"]))


def test_reduced_sensitivity_audit_cli_and_scalar_error() -> None:
    defaults = reduced.parse_args([])
    assert defaults.verify_sensitivities is False
    assert defaults.sensitivity_check_steps == "1e-3,3e-4,1e-4"
    configured = reduced.parse_args([
        "--verify-sensitivities",
        "--sensitivity-check-steps",
        "2e-3,5e-4",
        "--sensitivity-check-newton-tol",
        "1e-12",
    ])
    reduced.validate_args(configured)
    assert configured.verify_sensitivities is True
    assert reduced.relative_scalar_error(2.0, 2.2) == pytest.approx(0.2 / 2.2)
    with pytest.raises(ValueError, match="positive finite"):
        reduced.validate_args(
            reduced.parse_args(["--sensitivity-check-steps", "1e-3,-1e-4"])
        )

def test_threshold_objective_is_target_free_soft_jaccard_loss() -> None:
    args = reduced.parse_args([])
    reduced.validate_args(args)
    metrics = SimpleNamespace(
        leakage=4.0,
        missing=6.0,
        leakage_rel=0.4,
        missing_rel=0.6,
        target_area=10.0,
    )
    gradient = SimpleNamespace(
        grad_l=np.array([2.0, -1.0]),
        grad_m=np.array([-1.0, 3.0]),
    )

    assert reduced.threshold_merit_rel(metrics, args) == pytest.approx(10.0 / 14.0)
    np.testing.assert_allclose(
        reduced.threshold_objective_gradient(metrics, gradient, args),
        (4.0 * gradient.grad_l + 14.0 * gradient.grad_m) / (14.0 ** 2),
    )
    simplex = reduced.threshold_simplex_point(
        0.2,
        0.6,
        c_min=0.0,
        c_max=1.0,
        min_width=0.05,
    )
    trust_metric = reduced.threshold_trust_metric(
        simplex=simplex,
        sensitivity_metric=np.zeros((2, 2), dtype=np.float64),
        state_h1=1.0,
    )
    step = reduced.choose_parameter_step(
        c1=0.2,
        c2=0.6,
        c_min=0.0,
        c_max=1.0,
        min_width=0.05,
        metrics=metrics,
        gradient=gradient,
        trust_metric=trust_metric,
        trust_radius=0.5,
        args=args,
    )
    assert step.objective_name == "soft_jaccard_loss"
    assert step.metric_norm <= 0.5 + 1.0e-12

    for removed_option in (
        "--eta-out",
        "--tol-area",
        "--threshold-objective-mode",
        "--threshold-leakage-weight",
        "--threshold-missing-weight",
        "--jaccard-oscillation-atol",
        "--jaccard-oscillation-rtol",
    ):
        with pytest.raises(SystemExit):
            reduced.parse_args([removed_option, "0"])


def test_threshold_dikin_metric_is_relative_simplex_slack_norm() -> None:
    simplex = reduced.threshold_simplex_point(
        0.2,
        0.6,
        c_min=0.0,
        c_max=1.0,
        min_width=0.1,
    )
    metric = reduced.threshold_dikin_metric(simplex)
    expected = np.array(
        [
            [1.0 / 0.2**2 + 1.0 / 0.3**2, -1.0 / 0.3**2],
            [-1.0 / 0.3**2, 1.0 / 0.3**2 + 1.0 / 0.4**2],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(metric, expected)

    increment = np.array([0.01, -0.02], dtype=np.float64)
    relative_slack_changes = np.array(
        [0.01 / 0.2, (-0.02 - 0.01) / 0.3, 0.02 / 0.4],
        dtype=np.float64,
    )
    assert reduced.quadratic_metric_norm(increment, metric) == pytest.approx(
        np.linalg.norm(relative_slack_changes)
    )


def test_threshold_trust_metric_adds_relative_state_pullback() -> None:
    simplex = reduced.threshold_simplex_point(
        0.2,
        0.6,
        c_min=0.0,
        c_max=1.0,
        min_width=0.1,
    )
    sensitivity_metric = np.array([[4.0, 0.0], [0.0, 9.0]])
    trust_metric = reduced.threshold_trust_metric(
        simplex=simplex,
        sensitivity_metric=sensitivity_metric,
        state_h1=2.0,
    )
    np.testing.assert_allclose(
        trust_metric.pullback,
        np.array([[1.0, 0.0], [0.0, 2.25]]),
    )
    np.testing.assert_allclose(
        trust_metric.combined,
        trust_metric.dikin + trust_metric.pullback,
    )
    increment = np.array([0.01, -0.02])
    assert reduced.quadratic_metric_norm(
        increment, trust_metric.combined
    ) ** 2 == pytest.approx(
        reduced.quadratic_metric_norm(increment, trust_metric.dikin) ** 2
        + reduced.quadratic_metric_norm(increment, trust_metric.pullback) ** 2
    )


def test_metric_trust_region_qp_uses_ellipsoidal_radius() -> None:
    increment, model, status, hit_boundary = reduced.solve_trust_region_qp(
        g=np.array([1.0, 0.0]),
        rows=[],
        radius=0.5,
        hessian_scale=1.0,
        metric=np.diag([4.0, 1.0]),
    )
    np.testing.assert_allclose(increment, np.array([-0.25, 0.0]))
    assert model == pytest.approx(-0.125)
    assert status == "OK"
    assert hit_boundary is True
    assert reduced.quadratic_metric_norm(
        increment, np.diag([4.0, 1.0])
    ) == pytest.approx(0.5)

    coupled_metric = np.array([[4.0, 1.0], [1.0, 2.0]])
    coupled_gradient = np.array([0.2, -0.1])
    expected_unconstrained = -np.linalg.solve(coupled_metric, coupled_gradient)
    increment, model, status, hit_boundary = reduced.solve_trust_region_qp(
        g=coupled_gradient,
        rows=[],
        radius=0.5,
        hessian_scale=1.0,
        metric=coupled_metric,
    )
    np.testing.assert_allclose(increment, expected_unconstrained)
    assert model == pytest.approx(
        coupled_gradient.dot(expected_unconstrained)
        + 0.5 * expected_unconstrained.dot(coupled_metric).dot(
            expected_unconstrained
        )
    )
    assert status == "OK"
    assert hit_boundary is False


def test_brute_force_threshold_grid_is_center_first_and_deterministic() -> None:
    first = brute_force.threshold_grid(
        stage=0,
        center_c1=0.20,
        center_c2=0.30,
        radius_c1=0.02,
        radius_c2=0.04,
        points=5,
        c_min=0.0,
        c_max=0.5,
        min_width=1.0e-3,
    )
    second = brute_force.threshold_grid(
        stage=0,
        center_c1=0.20,
        center_c2=0.30,
        radius_c1=0.02,
        radius_c2=0.04,
        points=5,
        c_min=0.0,
        c_max=0.5,
        min_width=1.0e-3,
    )

    assert first == second
    assert len(first) == 25
    assert (first[0].c1, first[0].c2) == pytest.approx((0.20, 0.30))


def test_brute_force_objective_balances_leakage_and_area_match() -> None:
    objective, area_rel, mismatch = brute_force.objective_value(
        leakage_rel=0.1,
        activity_area=2.4,
        target_area=3.0,
        area_weight=2.0,
    )

    assert area_rel == pytest.approx(0.8)
    assert mismatch == pytest.approx(0.2)
    assert objective == pytest.approx(0.5)

    objective_with_missing, _, _ = brute_force.objective_value(
        leakage_rel=0.1,
        missing_rel=0.3,
        activity_area=2.4,
        target_area=3.0,
        area_weight=2.0,
        missing_weight=1.5,
    )
    assert objective_with_missing == pytest.approx(0.95)


def test_branch_overlap_self_normalization_scores_unchanged_diffuse_band_one() -> None:
    numerator = 0.25
    reference_self_overlap = 0.25
    assert reduced.normalized_branch_retention(
        numerator, reference_self_overlap
    ) == pytest.approx(1.0)
    assert optimizer.normalized_branch_retention(
        numerator, reference_self_overlap
    ) == pytest.approx(1.0)
    assert reduced.normalized_branch_retention(0.0, 0.0) == pytest.approx(1.0)


def test_symmetric_branch_dice_penalizes_loss_and_uncontrolled_growth() -> None:
    assert optimizer.normalized_branch_dice(0.25, 0.25, 0.25) == pytest.approx(1.0)
    assert optimizer.normalized_branch_dice(0.0, 0.25, 0.25) == pytest.approx(0.0)
    # A trial containing twice the activity cannot score one simply because
    # all of the reference activity was retained.
    assert optimizer.normalized_branch_dice(0.25, 0.25, 0.50) == pytest.approx(2.0 / 3.0)
    assert optimizer.normalized_branch_dice(0.0, 0.0, 0.0) == pytest.approx(1.0)


def test_activity_dice_compares_like_finite_element_representations() -> None:
    domain = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = fem.functionspace(domain, ("Lagrange", 2))
    reference = fem.Function(V)
    trial = fem.Function(V)
    reference.interpolate(lambda x: 0.2 + x[0] + 0.5 * x[1])
    trial.x.array[:] = reference.x.array
    trial.x.scatter_forward()
    dx = ufl.Measure("dx", domain=domain)
    assert optimizer.activity_dice_ratio(
        comm=MPI.COMM_WORLD,
        activity_ref=reference,
        activity_trial=trial,
        dx=dx,
    ) == pytest.approx(1.0, abs=2.0e-14)

    trial.x.array[:] = 2.0 * reference.x.array
    trial.x.scatter_forward()
    assert optimizer.activity_dice_ratio(
        comm=MPI.COMM_WORLD,
        activity_ref=reference,
        activity_trial=trial,
        dx=dx,
    ) == pytest.approx(0.8, abs=2.0e-14)


def test_brute_force_prefers_active_threshold_candidate_before_objective() -> None:
    inactive = {
        "converged": 1,
        "boundSatisfied": 1,
        "selectionEligible": 0,
        "objective": 0.01,
        "residual": 1.0e-10,
        "candidate": 0,
    }
    active = {
        "converged": 1,
        "boundSatisfied": 1,
        "selectionEligible": 1,
        "objective": 0.20,
        "residual": 1.0e-9,
        "candidate": 1,
    }
    assert brute_force.result_key(active) < brute_force.result_key(inactive)


def test_brute_force_torsion_cap_cli_and_full_grid() -> None:
    args = brute_force.parse_args([
        "search",
        "--output-dir", "/tmp/grid",
        "--grid-guess-c1", "0.5",
        "--grid-guess-c2", "0.500001",
        "--grid-radius-c1", "0.5",
        "--grid-radius-c2", "0.5",
        "--grid-points", "9",
        "--grid-cap-mode", "torsion",
        "--grid-seed", "both",
    ])
    brute_force.validate_grid_args(args, world_size=4)
    grid = brute_force.threshold_grid(
        stage=0,
        center_c1=0.5,
        center_c2=0.500001,
        radius_c1=0.5,
        radius_c2=0.5,
        points=9,
        c_min=0.0,
        c_max=1.0,
        min_width=1.0e-6,
    )

    assert args.grid_cap_mode == "torsion"
    assert args.grid_seed == "both"
    assert args.grid_newton_tol == pytest.approx(1.0e-6)
    assert args.grid_missing_weight == pytest.approx(1.0)
    assert args.grid_require_active_thresholds is True
    assert args.save_candidate_pngs is True
    assert "candidatePng" in brute_force.GRID_FIELDS
    assert brute_force.effective_bound_tolerance(args) == pytest.approx(1.0e-5)
    assert min(item.c1 for item in grid) == pytest.approx(0.0)
    assert max(item.c2 for item in grid) == pytest.approx(1.0)
    assert all(0.0 <= item.c1 < item.c2 <= 1.0 for item in grid)


def test_brute_force_dual_seed_prefers_physical_converged_branch() -> None:
    unphysical = {
        "converged": 1,
        "boundSatisfied": 0,
        "objective": 0.01,
        "residual": 1.0e-8,
        "candidate": 7,
        "wallTime": 2.0,
        "seed": "phi_target",
    }
    physical = {
        "converged": 1,
        "boundSatisfied": 1,
        "objective": 0.02,
        "residual": 2.0e-8,
        "candidate": 7,
        "wallTime": 3.0,
        "seed": "torsion",
    }
    physical_state = np.array([2.0])
    row, state = brute_force.choose_candidate_attempt([
        (unphysical, np.array([1.0])),
        (physical, physical_state),
    ])

    assert row["seed"] == "torsion"
    assert row["seedAttempts"] == 2
    assert row["wallTime"] == pytest.approx(5.0)
    assert np.array_equal(state, physical_state)


class _FakeCollectiveComm:
    def __init__(self, rank=0, gathered=None):
        self.rank = rank
        self._gathered = gathered

    def allgather(self, local):
        return [local] if self._gathered is None else self._gathered

    def bcast(self, value, root=0):
        assert root == 0
        return value


def test_recoverable_homotopy_solver_exception_enables_window_fit() -> None:
    error = RuntimeError("reusable linear solve failed with PETSc reason -5")
    report = optimizer.synchronize_homotopy_initialization_exception(
        _FakeCollectiveComm(),
        error,
    )
    assert report is not None
    assert report.status == "FAIL_SOLVER_EXCEPTION"
    assert report.recoverable
    assert report.ranks == (0,)
    optimizer.require_window_fit_homotopy_exception_fallback(
        report,
        init_fallback="window-fit",
        local_error=error,
    )


def test_recoverable_homotopy_exception_without_fallback_reraises() -> None:
    error = RuntimeError("linear solve 'homotopy' failed with PETSc reason -5")
    report = optimizer.synchronize_homotopy_initialization_exception(
        _FakeCollectiveComm(),
        error,
    )
    assert report is not None
    with pytest.raises(RuntimeError, match="PETSc reason -5"):
        optimizer.require_window_fit_homotopy_exception_fallback(
            report,
            init_fallback="none",
            local_error=error,
        )


@pytest.mark.parametrize(
    "error",
    [
        ValueError("bad homotopy parameter"),
        RuntimeError("homotopy trajectory callback has no workspace"),
        AssertionError("programming invariant"),
    ],
)
def test_programming_errors_never_trigger_homotopy_fallback(error: Exception) -> None:
    report = optimizer.synchronize_homotopy_initialization_exception(
        _FakeCollectiveComm(),
        error,
    )
    assert report is not None
    assert not report.recoverable
    with pytest.raises(type(error), match=str(error)):
        optimizer.require_window_fit_homotopy_exception_fallback(
            report,
            init_fallback="window-fit",
            local_error=error,
        )


def test_remote_solver_exception_gives_same_collective_fallback_decision() -> None:
    remote = {
        "rank": 0,
        "exception_type": "builtins.RuntimeError",
        "message": "reusable linear solve failed with PETSc reason -5",
        "recoverable": True,
    }
    report = optimizer.synchronize_homotopy_initialization_exception(
        _FakeCollectiveComm(rank=1, gathered=[remote, None]),
        None,
    )
    assert report is not None and report.recoverable
    optimizer.require_window_fit_homotopy_exception_fallback(
        report,
        init_fallback="window-fit",
        local_error=None,
    )


def _make_trajectory_recorder(tmp_path: Path, name: str):
    import meshio

    mesh_path = tmp_path / f"{name}.msh"
    meshio.write(
        mesh_path,
        meshio.Mesh(
            points=np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]),
            cells=[("triangle", np.array([[0, 1, 2]], dtype=np.int64))],
        ),
        file_format="gmsh22",
        binary=False,
    )
    output = tmp_path / f"{name}.npz"
    recorder = optimizer.TrajectoryRecorder(
        enabled=True,
        every=1,
        output=output,
        mesh_path=mesh_path,
        comm=_FakeCollectiveComm(),
    )
    recorder.coordinates = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    recorder.fixed = {
        "torsion": np.array([0.0, 1.0, 0.5]),
        "target_band": np.array([0.0, 1.0, 0.0]),
        "target_density": np.array([0.0, 1.0, 0.0]),
        "target_potential": np.array([0.0, 0.2, 0.0]),
    }
    recorder.set_context({"run_tag": name, "order": 2})
    return recorder, output


def test_partial_trajectory_before_first_iterate_synthesizes_design_state(tmp_path) -> None:
    recorder, output = _make_trajectory_recorder(tmp_path, "hminus1_failure")
    recorder.flush_partial({
        "terminal_stage": "hminus1_search",
        "terminal_status": "FAIL_HMINUS1",
    })

    assert recorder._finalized
    assert output.is_file()
    assert not output.with_name(f".{output.name}.tmp.npz").exists()
    with np.load(output, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        assert metadata["complete"] is False
        assert metadata["terminal_stage"] == "hminus1_search"
        assert metadata["terminal_status"] == "FAIL_HMINUS1"
        assert [state["stage"] for state in metadata["states"]] == [
            "design_only_failure",
        ]
        assert archive["states_phi"].shape == (1, 3)
        assert np.allclose(
            archive["states_mismatch"],
            np.zeros((1, 3)),
        )


def test_unhandled_exit_flush_preserves_last_gathered_state(tmp_path) -> None:
    recorder, output = _make_trajectory_recorder(tmp_path, "outer_failure")
    recorder.states_phi.append(np.array([0.0, 0.18, 0.0]))
    recorder.states_rho.append(np.array([0.0, 0.9, 0.0]))
    recorder.states.append({
        "stage": "accepted_outer",
        "c1": 0.1,
        "c2": 0.2,
        "width": 0.1,
        "eps_phi": 0.008,
        "homotopy_lambda": 1.0,
        "outer_iteration": 3,
    })

    recorder._flush_at_exit()

    with np.load(output, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        assert metadata["complete"] is False
        assert metadata["terminal_status"] == "PARTIAL_UNHANDLED_EXIT"
        assert [state["stage"] for state in metadata["states"]] == [
            "accepted_outer",
        ]
        assert np.allclose(
            archive["states_phi"][0],
            np.array([0.0, 0.18, 0.0]),
        )


def test_model_reduction_proxy_scales_fraction_by_target_area() -> None:
    assert not model_reduction_is_sufficient(
        0.02,
        target_area=10.0,
        sufficient_decrease_fraction=0.01,
    )
    assert model_reduction_is_sufficient(
        0.11,
        target_area=10.0,
        sufficient_decrease_fraction=0.01,
    )
    assert not model_reduction_is_sufficient(
        math.inf,
        target_area=10.0,
        sufficient_decrease_fraction=0.01,
    )


def test_inexact_newton_snapshot_comparison_and_schema() -> None:
    diagnostics = compare_inexact_newton_snapshot(
        state_l2_relative_error=0.1,
        state_h1_relative_error=0.2,
        sensitivity1_l2_relative_error=0.3,
        sensitivity1_h1_relative_error=0.4,
        sensitivity2_l2_relative_error=0.5,
        sensitivity2_h1_relative_error=0.6,
        sensitivity1_defect=1.0e-4,
        sensitivity2_defect=2.0e-4,
        approximate_reduced_gradient=np.array([1.0, 0.0]),
        reference_reduced_gradient=np.array([0.0, 1.0]),
        approximate_threshold_step=np.array([1.0, -2.0]),
        reference_threshold_step=np.array([1.0, -2.0]),
        approximate_predicted_reduction=0.75,
        reference_predicted_reduction=1.0,
        approximate_accepted=True,
        reference_accepted=False,
    )

    assert diagnostics.reduced_gradient_relative_error == pytest.approx(math.sqrt(2.0))
    assert diagnostics.reduced_gradient_angle_degrees == pytest.approx(90.0)
    assert diagnostics.threshold_step_relative_error == pytest.approx(0.0)
    assert diagnostics.threshold_step_angle_degrees == pytest.approx(0.0)
    assert diagnostics.predicted_reduction_relative_error == pytest.approx(0.25)
    assert diagnostics.acceptance_decision_agrees is False
    assert set(diagnostics.as_dict()).issubset(INEXACT_NEWTON_CSV_FIELDS)


@pytest.mark.parametrize(
    "status",
    ["", "OK", "converged", "  CONVERGED_CERTIFIED_SUBBAND  "],
)
def test_initial_equilibrium_success_statuses_are_accepted(status: str) -> None:
    assert initial_equilibrium_status_is_acceptable(
        status,
        math.nan,
        max_residual=1.0e-11,
    )


def test_unsuccessful_initial_equilibrium_can_be_residual_qualified() -> None:
    assert initial_equilibrium_status_is_acceptable(
        "STEP_TOO_SMALL",
        "6.65e-12",
        max_residual=1.0e-11,
    )
    assert initial_equilibrium_status_is_acceptable(
        "STEP_TOO_SMALL",
        1.0e-11,
        max_residual=1.0e-11,
    )


@pytest.mark.parametrize("residual", [None, math.nan, math.inf, "not-a-number", 1.01e-11])
def test_unsuccessful_initial_equilibrium_rejects_invalid_residual(residual: object) -> None:
    assert not initial_equilibrium_status_is_acceptable(
        "STEP_TOO_SMALL",
        residual,
        max_residual=1.0e-11,
    )


def test_unsuccessful_initial_equilibrium_requires_requested_residual_bound() -> None:
    assert not initial_equilibrium_status_is_acceptable(
        "STEP_TOO_SMALL",
        6.65e-12,
        max_residual=None,
    )


def test_fit_quadrature_samples_contain_owned_cells_only() -> None:
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD,
        3,
        2,
        cell_type=mesh.CellType.triangle,
    )
    V = fem.functionspace(domain, ("Lagrange", 2))
    phi = fem.Function(V)
    rho = fem.Function(V)
    phi.interpolate(lambda x: x[0] + 2.0 * x[1])
    rho.interpolate(lambda x: 1.0 + 0.0 * x[0])
    phi.x.scatter_forward()
    rho.x.scatter_forward()

    quadrature_degree = 4
    phi_values, rho_values, weights = window_fit.quadrature_samples_for_fit(
        phi,
        rho,
        quadrature_degree=quadrature_degree,
    )
    quadrature_points, _ = basix.make_quadrature(
        basix.CellType.triangle,
        quadrature_degree,
    )
    owned_cells = domain.topology.index_map(domain.topology.dim).size_local
    expected_local_samples = int(owned_cells) * int(quadrature_points.shape[0])
    assert phi_values.size == expected_local_samples
    assert rho_values.size == expected_local_samples
    assert weights.size == expected_local_samples

    global_samples = MPI.COMM_WORLD.allreduce(phi_values.size, op=MPI.SUM)
    global_cells = domain.topology.index_map(domain.topology.dim).size_global
    assert global_samples == int(global_cells) * int(quadrature_points.shape[0])


def test_window_fit_and_candidates_reuse_supplied_quadrature_samples(monkeypatch) -> None:
    fake_function = SimpleNamespace(
        function_space=SimpleNamespace(
            mesh=SimpleNamespace(comm=MPI.COMM_SELF),
        ),
    )
    samples = (
        np.linspace(0.05, 0.95, 19, dtype=np.float64),
        np.array([0.0] * 4 + [1.0] * 11 + [0.0] * 4, dtype=np.float64),
        np.full(19, 1.0 / 19.0, dtype=np.float64),
    )

    def fail_resampling(*_args, **_kwargs):
        raise AssertionError("quadrature fields were sampled more than once")

    monkeypatch.setattr(window_fit, "quadrature_samples_for_fit", fail_resampling)
    fit_result = window_fit.fit_phi_window_to_torsion_design(
        fake_function,
        fake_function,
        rho_design_l2=math.sqrt(float(np.dot(samples[2], samples[1] ** 2))),
        phi_design_max=0.95,
        eps_ratio=0.08,
        rho_amp=1.0,
        quadrature_degree=4,
        grid_points=8,
        refine_points=5,
        refine_passes=1,
        histogram_bins=64,
        quadrature_samples=samples,
    )
    assert 0.0 < fit_result.c1 < fit_result.c2 < 1.0
    assert fit_result.sample_count == len(samples[0])

    monkeypatch.setattr(optimizer, "quadrature_samples_for_fit", fail_resampling)
    candidates = optimizer.build_initial_window_candidates(
        phi_target=fake_function,
        rho_design=fake_function,
        fit_candidate=("density_l2", fit_result.c1, fit_result.c2),
        rho_design_l2=math.sqrt(float(np.dot(samples[2], samples[1] ** 2))),
        rho_amp=1.0,
        target_area=float(np.dot(samples[2], samples[1])),
        quadrature_degree=4,
        c_min=0.0,
        c_max=1.0,
        min_width=0.05,
        args=parse_args([]),
        quadrature_samples=samples,
    )
    assert candidates
    assert candidates[0].name == "density_l2"


def test_reduced_initializer_plotting_and_strict_projected_selection() -> None:
    defaults = reduced.parse_args([])
    assert defaults.plot_initial_candidates is False
    assert defaults.plot_design_hold_seconds == pytest.approx(2.0)
    assert reduced.design_output_enabled(defaults) is False
    assert reduced.design_output_enabled(reduced.parse_args(["--plot"])) is True
    assert reduced.design_output_enabled(reduced.parse_args(["--save-frames"])) is True
    assert reduced.parse_args(["--plot-fields", "state"]).plot_fields == "state"
    assert not hasattr(defaults, "plot_design")
    assert not hasattr(defaults, "frame_design")

    configured = reduced.parse_args(
        [
            "--plot",
            "--plot-fields", "density",
            "--plot-initial-candidates",
            "--plot-accepted-states",
            "--plot-design-hold-seconds", "3.5",
            "--inner-newton-tol", "1e-12",
            "--require-inner-newton-convergence",
        ]
    )
    reduced.validate_args(configured)
    assert configured.plot_initial_candidates is True
    assert configured.plot_accepted_states is True
    assert configured.plot_fields == "density"
    assert configured.plot_design_hold_seconds == pytest.approx(3.5)

    with pytest.raises(ValueError, match="plot-design-hold-seconds"):
        invalid_hold = reduced.parse_args(["--plot-design-hold-seconds", "-1"])
        reduced.validate_args(invalid_hold)

    failed = SimpleNamespace(
        base=SimpleNamespace(name="failed"),
        newton=SimpleNamespace(
            converged=False,
            status="FAIL_LS",
            residual=1.0e-3,
        ),
        score=0.1,
    )
    converged = SimpleNamespace(
        base=SimpleNamespace(name="converged"),
        newton=SimpleNamespace(
            converged=True,
            status="CONVERGED_RESIDUAL",
            residual=8.0e-13,
        ),
        score=2.0,
    )

    assert reduced.select_projected_initial_candidate(
        [failed, converged],
        require_converged=False,
    ) is failed
    assert reduced.select_projected_initial_candidate(
        [failed, converged],
        require_converged=True,
    ) is converged
    with pytest.raises(RuntimeError, match="no automatic initial-window candidate"):
        reduced.select_projected_initial_candidate(
            [failed],
            require_converged=True,
        )

def test_direct_fractional_runner_bypasses_fit_and_forces_resolved_states() -> None:
    args = direct_fractional.parse_args(
        [
            "--alphaT1", "0.6",
            "--alphaT2", "0.7",
            "--tol-res", "1e-12",
        ]
    )
    reduced.validate_args(args)

    assert args.initial_threshold_mode == "torsion-fraction-phi-target"
    assert args.initial_projection_mode == "homotopy"
    assert args.include_fit_init is False
    assert args.threshold_cap_mode == "torsion"
    assert args.require_direct_seed_interior_level_curves is True
    assert args.require_inner_newton_convergence is True
    assert args.inner_newton_tol == pytest.approx(1.0e-12)
    assert args.homotopy_tol_res == pytest.approx(1.0e-12)
    assert args.homotopy_initial_step == pytest.approx(0.5)
    assert args.homotopy_max_step == pytest.approx(0.5)
    assert args.homotopy_step_shrink == pytest.approx(0.5)
    assert args.final_newton_tol_res == pytest.approx(1.0e-12)
    assert args.jaccard_stagnation_stop is True
    assert args.newton_soft_cap is True
    assert args.newton_soft_cap_factor == pytest.approx(2.0)

    relaxed = direct_fractional.parse_args(
        ["--no-require-direct-seed-interior-level-curves"]
    )
    assert relaxed.require_direct_seed_interior_level_curves is False

    with pytest.raises(ValueError, match="discovers its initial thresholds"):
        direct_fractional.parse_args(["--c1-phi", "0.1", "--c2-phi", "0.2"])


def test_direct_fractional_thresholds_use_only_current_target_scale() -> None:
    c1, c2 = reduced.torsion_fraction_phi_target_thresholds(
        phi_target_max=0.25,
        alpha1=0.6,
        alpha2=0.7,
        c_min=0.0,
        c_max=1.0,
        min_width=1.0e-4,
    )
    assert c1 == pytest.approx(0.15)
    assert c2 == pytest.approx(0.175)

    with pytest.raises(ValueError, match="0 < alpha1"):
        reduced.torsion_fraction_phi_target_thresholds(
            phi_target_max=0.25,
            alpha1=0.0,
            alpha2=0.7,
            c_min=0.0,
            c_max=1.0,
            min_width=1.0e-4,
        )


def test_frozen_frontier_runner_has_no_heuristic_threshold_fallback() -> None:
    args = frozen_frontier_runner.parse_args(
        [
            "--alphaT1", "0.4",
            "--alphaT2", "0.5",
            "--frozen-frontier-bins", "2048",
            "--frozen-leakage-cap-rel", "0.03",
            "--tol-res", "1e-12",
        ]
    )
    reduced.validate_args(args)

    assert args.initial_threshold_mode == "frozen-frontier"
    assert args.initial_projection_mode == "homotopy"
    assert args.include_fit_init is False
    assert args.threshold_cap_mode == "torsion"
    assert args.frozen_frontier_bins == 2048
    assert args.frozen_leakage_cap_rel == pytest.approx(0.03)
    assert args.initial_alpha1 is None
    assert args.initial_alpha2 is None
    assert args.require_inner_newton_convergence is True
    assert args.inner_newton_tol == pytest.approx(1.0e-12)
    assert args.homotopy_tol_res == pytest.approx(1.0e-12)
    assert args.final_newton_tol_res == pytest.approx(1.0e-12)
    assert args.jaccard_stagnation_stop is True

    uncapped = frozen_frontier_runner.parse_args([])
    reduced.validate_args(uncapped)
    assert uncapped.frozen_leakage_cap_rel is None

    with pytest.raises(ValueError, match="selects its initial thresholds"):
        frozen_frontier_runner.parse_args(
            ["--c1-phi", "0.1", "--c2-phi", "0.2"]
        )
    with pytest.raises(ValueError, match="fractional threshold seeds"):
        frozen_frontier_runner.parse_args(
            ["--initial-alpha1", "0.1", "--initial-alpha2", "0.2"]
        )


def test_frozen_logistic_refinement_certifies_the_unchanged_cap() -> None:
    phi_values = np.linspace(0.0, 1.0, 501)
    target_values = (
        (phi_values > 0.35) & (phi_values < 0.65)
    ).astype(np.float64)
    weights = np.full(phi_values.size, 1.0 / (phi_values.size - 1))
    target_area = float(np.dot(weights, target_values))
    leakage_cap = 0.06
    args = SimpleNamespace(
        eps_mode="fixed",
        eps_ratio=0.1,
        eps_phi=0.03,
    )

    c1, c2, metrics, iterations, _ = reduced.refine_frozen_logistic_thresholds(
        comm=MPI.COMM_SELF,
        phi_values=phi_values,
        target_values=target_values,
        weights=weights,
        initial_c1=0.34,
        initial_c2=0.66,
        c_min=0.0,
        c_max=1.0,
        min_width=0.05,
        target_area=target_area,
        leakage_cap=leakage_cap,
        args=args,
    )

    assert 0.0 <= c1 < c2 <= 1.0
    assert c2 - c1 >= 0.05
    assert metrics.leakage <= leakage_cap
    assert metrics.overlap > 0.0
    assert iterations > 0


def test_accepted_threshold_constant_synchronization_is_explicit() -> None:
    c1_const = SimpleNamespace(value=np.array(9.0))
    c2_const = SimpleNamespace(value=np.array(8.0))
    eps_const = SimpleNamespace(value=np.array(7.0))

    mismatch = reduced.synchronize_threshold_constants(
        c1_const,
        c2_const,
        eps_const,
        c1=0.15,
        c2=0.175,
        eps_phi=0.003,
    )

    assert mismatch <= np.finfo(np.float64).eps
    assert float(np.asarray(c1_const.value)) == pytest.approx(0.15)
    assert float(np.asarray(c2_const.value)) == pytest.approx(0.175)
    assert float(np.asarray(eps_const.value)) == pytest.approx(0.003)


def test_jaccard_stagnation_requires_both_minimum_and_patience() -> None:
    assert not reduced.jaccard_stagnation_detected(
        accepted_steps=5,
        stale_steps=4,
        minimum_accepted=6,
        patience=4,
    )
    assert not reduced.jaccard_stagnation_detected(
        accepted_steps=6,
        stale_steps=3,
        minimum_accepted=6,
        patience=4,
    )
    assert reduced.jaccard_stagnation_detected(
        accepted_steps=6,
        stale_steps=4,
        minimum_accepted=6,
        patience=4,
    )
