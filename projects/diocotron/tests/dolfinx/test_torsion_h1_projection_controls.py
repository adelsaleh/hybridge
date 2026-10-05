"""Verification tests for the torsion H1 projection control-space algebra."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "projects/diocotron/dolfinx/torsion/optimization/h1_controls.py"
)
SPEC = importlib.util.spec_from_file_location("torsion_h1_projection_controls", MODULE_PATH)
controls = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = controls
SPEC.loader.exec_module(controls)


def _centered_difference(function, point, index, step=1.0e-7):
    plus = np.asarray(point, dtype=float).copy()
    minus = np.asarray(point, dtype=float).copy()
    plus[index] += step
    minus[index] -= step
    return (function(*plus) - function(*minus)) / (2.0 * step)


@pytest.mark.parametrize("value", [-0.7, -0.1, 0.17, 0.43, 1.2])
def test_window_center_width_derivatives_include_epsilon_chain(value):
    center, width, ratio = 0.21, 0.37, 0.13
    evaluation = controls.window_center_width(np.asarray([value]), center, width, ratio)

    def activity(phi, m, d):
        return float(controls.window_center_width(np.asarray([phi]), m, d, ratio).activity[0])

    state_fd = _centered_difference(lambda phi, m: activity(phi, m, width), [value, center], 0)
    center_fd = _centered_difference(lambda phi, m: activity(phi, m, width), [value, center], 1)
    width_fd = _centered_difference(lambda m, d: activity(value, m, d), [center, width], 1)
    assert evaluation.state_derivative[0] == pytest.approx(state_fd, rel=2.0e-7, abs=2.0e-9)
    assert evaluation.center_derivative[0] == pytest.approx(center_fd, rel=2.0e-7, abs=2.0e-9)
    assert evaluation.width_derivative[0] == pytest.approx(width_fd, rel=5.0e-7, abs=5.0e-9)


def test_frozen_geometric_derivatives_match_centered_differences():
    rng = np.random.default_rng(23987)
    values = rng.normal(0.2, 0.35, 300)
    target = (rng.random(values.size) > 0.45).astype(float)
    weights = rng.uniform(0.01, 0.2, values.size)
    point = np.asarray([0.17, 0.44])
    ratio = 0.09
    result = controls.frozen_window_metrics(values, target, weights, *point, ratio)

    def quantities(m, d):
        metric = controls.frozen_window_metrics(values, target, weights, m, d, ratio)
        return metric.leakage, metric.missing

    for index in range(2):
        leakage_fd = _centered_difference(lambda m, d: quantities(m, d)[0], point, index)
        missing_fd = _centered_difference(lambda m, d: quantities(m, d)[1], point, index)
        assert result.gradient_leakage[index] == pytest.approx(leakage_fd, rel=2.0e-7, abs=2.0e-9)
        assert result.gradient_missing[index] == pytest.approx(missing_fd, rel=2.0e-7, abs=2.0e-9)


@pytest.mark.parametrize("seed", range(12))
def test_sorted_scan_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    values = np.round(rng.uniform(-0.2, 1.0, 45), 2)
    target = (rng.random(values.size) > 0.5).astype(float)
    target[0] = 1.0
    weights = rng.uniform(0.05, 1.0, values.size)
    bounds = controls.CenterWidthBounds(-0.1, 0.9, 0.04, 1.1)
    budget = 0.8
    fast = controls.sorted_hard_window_scan(values, target, weights, bounds, budget)
    reference = controls.hard_window_brute_force(values, target, weights, bounds, budget)
    assert fast.target_weight == pytest.approx(reference.target_weight)
    assert fast.outside_weight == pytest.approx(reference.outside_weight)
    assert fast.width == pytest.approx(reference.width)
    assert fast.center == pytest.approx(reference.center)


def test_frozen_window_reachability_bound_is_conservative():
    values = np.asarray([-0.3, -0.05, 0.2, 0.45, 0.7, 0.95])
    target = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    weights = np.asarray([1.0, 0.7, 2.0, 1.5, 0.3, 0.5])
    bounds = controls.CenterWidthBounds(0.1, 0.4, 0.05, 0.2)
    eps_ratio = 0.08
    result = controls.frozen_window_reachability_bound(
        values, target, weights, bounds, eps_ratio
    )

    assert result.lower_edge_min == pytest.approx(0.0)
    assert result.upper_edge_max == pytest.approx(0.5)
    assert result.epsilon_max == pytest.approx(0.016)
    assert result.target_potential_min == pytest.approx(-0.3)
    assert result.target_potential_max == pytest.approx(0.95)
    assert result.hard_coverage_upper_bound == pytest.approx(3.5 / 5.0)

    sampled_coverages = []
    for center in np.linspace(bounds.center_min, bounds.center_max, 17):
        for width in np.linspace(bounds.width_min, bounds.width_max, 19):
            window = controls.window_center_width(values, center, width, eps_ratio)
            sampled_coverages.append(
                float(np.dot(weights * target, window.activity) / np.dot(weights, target))
            )
    assert result.smooth_coverage_upper_bound + 1.0e-14 >= max(sampled_coverages)


def test_frozen_window_reachability_detects_unreachable_high_target():
    values = np.asarray([0.1, 0.2, 0.8, 0.9])
    target = np.asarray([0.0, 0.0, 1.0, 1.0])
    weights = np.ones_like(values)
    bounds = controls.CenterWidthBounds(0.1, 0.4, 0.05, 0.2)
    result = controls.frozen_window_reachability_bound(
        values, target, weights, bounds, eps_ratio=0.01
    )

    assert result.hard_coverage_upper_bound == 0.0
    assert result.smooth_coverage_upper_bound < 1.0e-12


def test_convex_qp_active_set_solution():
    gradient = np.asarray([-2.0, -0.5])
    hessian = np.asarray([[2.0, 0.2], [0.2, 1.0]])
    rows = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    rhs = np.asarray([0.35, 1.0, 0.3, 1.0])
    result = controls.solve_convex_qp_2d(gradient, hessian, rows, rhs)
    assert result.success
    assert np.all(rows @ result.step <= rhs + 1.0e-11)
    grid_x = np.linspace(-1.0, 0.35, 501)
    grid_y = np.linspace(-1.0, 0.3, 501)
    xx, yy = np.meshgrid(grid_x, grid_y, indexing="ij")
    objective = (
        gradient[0] * xx
        + gradient[1] * yy
        + 0.5 * hessian[0, 0] * xx**2
        + hessian[0, 1] * xx * yy
        + 0.5 * hessian[1, 1] * yy**2
    )
    assert result.objective <= float(np.min(objective)) + 2.0e-5


def test_threshold_edge_step_constraints_scale_with_current_width():
    width = 0.2
    fraction = 0.15
    rows, rhs, names = controls.threshold_edge_step_constraints(width, fraction)
    assert len(names) == 4
    assert np.all(rhs == pytest.approx(fraction * width))

    admissible = np.asarray([0.01, 0.04])
    dc1 = admissible[0] - 0.5 * admissible[1]
    dc2 = admissible[0] + 0.5 * admissible[1]
    assert max(abs(dc1), abs(dc2)) == pytest.approx(fraction * width)
    assert np.all(rows @ admissible <= rhs + 1.0e-14)

    excessive_translation = np.asarray([0.031, 0.0])
    assert np.any(rows @ excessive_translation > rhs)


@pytest.mark.parametrize("width,fraction", [(0.0, 0.2), (0.2, 0.0), (-1.0, 0.2)])
def test_threshold_edge_step_constraints_reject_invalid_inputs(width, fraction):
    with pytest.raises(ValueError):
        controls.threshold_edge_step_constraints(width, fraction)


def test_hysteretic_activity_components_preserves_a_weak_bridge():
    # Four cells in a chain.  The middle two are below the core contour but
    # above the bridge contour, so the two core ends are one component.
    facets = np.asarray(
        [
            [0, 1, 2],
            [2, 3, 4],
            [4, 5, 6],
            [6, 7, 8],
        ]
    )
    peaks = np.asarray([0.9, 0.3, 0.3, 0.8])
    result = controls.hysteretic_activity_components(
        facets,
        peaks,
        core_level=0.5,
        bridge_level=0.25,
        min_component_fraction=0.01,
    )
    assert result.component_count == 1
    assert result.raw_component_count == 1
    assert result.largest_fraction == pytest.approx(1.0)


def test_hysteretic_activity_components_detects_a_neck_pinch():
    facets = np.asarray(
        [
            [0, 1, 2],
            [2, 3, 4],
            [4, 5, 6],
            [6, 7, 8],
        ]
    )
    peaks = np.asarray([0.9, 0.1, 0.1, 0.8])
    result = controls.hysteretic_activity_components(
        facets,
        peaks,
        core_level=0.5,
        bridge_level=0.25,
        min_component_fraction=0.01,
    )
    assert result.component_count == 2
    assert result.second_fraction == pytest.approx(0.5)


def test_hysteretic_activity_components_ignores_a_tiny_core_island():
    # Component weights are used instead of raw component count so one small
    # interpolation island cannot veto a distributed solve.
    facets = np.asarray(
        [
            [0, 1, 2],
            [2, 3, 4],
            [10, 11, 12],
        ]
    )
    peaks = np.asarray([0.9, 0.8, 0.7])
    weights = np.asarray([50.0, 49.0, 1.0])
    result = controls.hysteretic_activity_components(
        facets,
        peaks,
        core_level=0.5,
        bridge_level=0.25,
        min_component_fraction=0.02,
        cell_weights=weights,
    )
    assert result.raw_component_count == 2
    assert result.component_count == 1
    assert result.second_fraction == pytest.approx(0.01)


@pytest.mark.parametrize(
    "core,bridge,fraction",
    [(0.4, 0.5, 0.01), (1.1, 0.2, 0.01), (0.5, 0.2, 1.0)],
)
def test_hysteretic_activity_components_rejects_invalid_levels(core, bridge, fraction):
    with pytest.raises(ValueError):
        controls.hysteretic_activity_components(
            np.asarray([[0, 1, 2]]),
            np.asarray([0.8]),
            core_level=core,
            bridge_level=bridge,
            min_component_fraction=fraction,
        )


def test_outward_interval_repair_candidates_are_bounded_and_nearest_first():
    bounds = controls.CenterWidthBounds(0.0, 2.0, 0.1, 1.5)
    center, width = 1.0, 0.4
    seed_c1, seed_c2 = controls.thresholds_from_center_width(center, width)
    candidates = controls.outward_interval_repair_candidates(
        center, width, bounds, samples=8
    )
    assert np.allclose(candidates[0], [center, width])
    distances = []
    for candidate in candidates:
        assert bounds.contains(*candidate)
        c1, c2 = controls.thresholds_from_center_width(*candidate)
        assert c1 <= seed_c1 + 1.0e-14
        assert c2 >= seed_c2 - 1.0e-14
        distances.append(max(seed_c1 - c1, c2 - seed_c2) / width)
    assert np.all(np.diff(distances) >= -1.0e-14)


def test_outward_interval_repair_candidates_validate_seed_and_samples():
    bounds = controls.CenterWidthBounds(0.0, 2.0, 0.1, 1.5)
    with pytest.raises(ValueError):
        controls.outward_interval_repair_candidates(1.0, 0.4, bounds, samples=1)
    with pytest.raises(ValueError):
        controls.outward_interval_repair_candidates(3.0, 0.4, bounds, samples=8)


def test_box_interval_repair_candidates_cover_box_and_are_nearest_first():
    bounds = controls.CenterWidthBounds(0.0, 2.0, 0.1, 1.5)
    center, width = 1.0, 0.4
    seed_c1, seed_c2 = controls.thresholds_from_center_width(center, width)
    candidates = controls.box_interval_repair_candidates(
        center, width, bounds, samples=8
    )
    assert np.allclose(candidates[0], [center, width])
    assert any(np.allclose(point, bounds.lower) for point in candidates)
    assert any(np.allclose(point, bounds.upper) for point in candidates)
    distances = []
    for candidate in candidates:
        assert bounds.contains(*candidate)
        c1, c2 = controls.thresholds_from_center_width(*candidate)
        distances.append(max(abs(c1 - seed_c1), abs(c2 - seed_c2)) / width)
    assert np.all(np.diff(distances) >= -1.0e-14)


def test_box_interval_repair_candidates_include_translations_and_contractions():
    bounds = controls.CenterWidthBounds(0.0, 2.0, 0.1, 1.5)
    candidates = controls.box_interval_repair_candidates(
        1.0, 0.4, bounds, samples=7
    )
    assert any(point[0] < 1.0 and point[1] < 0.4 for point in candidates)
    assert any(point[0] > 1.0 and point[1] > 0.4 for point in candidates)


def test_box_interval_repair_candidates_validate_seed_and_samples():
    bounds = controls.CenterWidthBounds(0.0, 2.0, 0.1, 1.5)
    with pytest.raises(ValueError):
        controls.box_interval_repair_candidates(1.0, 0.4, bounds, samples=1)
    with pytest.raises(ValueError):
        controls.box_interval_repair_candidates(3.0, 0.4, bounds, samples=8)


def test_topology_repair_ranking_prioritizes_geometric_violation():
    close_but_bad = controls.FrozenWindowMetrics(
        leakage=3.4,
        missing=0.6,
        activity_area_ratio=3.8,
        gradient_leakage=np.zeros(2),
        gradient_missing=np.zeros(2),
    )
    farther_but_viable = controls.FrozenWindowMetrics(
        leakage=0.57,
        missing=0.95,
        activity_area_ratio=0.62,
        gradient_leakage=np.zeros(2),
        gradient_missing=np.zeros(2),
    )
    close_key = controls.frozen_topology_repair_rank_key(
        close_but_bad,
        leakage_max=0.30,
        missing_max=0.70,
        edge_motion=0.1,
    )
    viable_key = controls.frozen_topology_repair_rank_key(
        farther_but_viable,
        leakage_max=0.30,
        missing_max=0.70,
        edge_motion=2.0,
    )
    assert viable_key < close_key


def test_topology_repair_ranking_uses_coverage_then_edge_motion_for_ties():
    low_coverage = controls.FrozenWindowMetrics(
        leakage=0.375,
        missing=0.75,
        activity_area_ratio=0.50,
        gradient_leakage=np.zeros(2),
        gradient_missing=np.zeros(2),
    )
    high_coverage = controls.FrozenWindowMetrics(
        leakage=0.50,
        missing=0.625,
        activity_area_ratio=0.70,
        gradient_leakage=np.zeros(2),
        gradient_missing=np.zeros(2),
    )
    assert controls.frozen_topology_repair_rank_key(
        high_coverage,
        leakage_max=0.25,
        missing_max=0.50,
        edge_motion=2.0,
    ) < controls.frozen_topology_repair_rank_key(
        low_coverage,
        leakage_max=0.25,
        missing_max=0.50,
        edge_motion=0.1,
    )

    same_metrics_near = controls.frozen_topology_repair_rank_key(
        high_coverage,
        leakage_max=0.25,
        missing_max=0.50,
        edge_motion=0.5,
    )
    same_metrics_far = controls.frozen_topology_repair_rank_key(
        high_coverage,
        leakage_max=0.25,
        missing_max=0.50,
        edge_motion=1.5,
    )
    assert same_metrics_near < same_metrics_far


def test_topology_repair_ranking_validates_inputs():
    metrics = controls.FrozenWindowMetrics(
        leakage=0.1,
        missing=0.2,
        activity_area_ratio=0.9,
        gradient_leakage=np.zeros(2),
        gradient_missing=np.zeros(2),
    )
    with pytest.raises(ValueError):
        controls.frozen_topology_repair_rank_key(
            metrics,
            leakage_max=0.3,
            missing_max=0.7,
            edge_motion=-1.0,
        )


def test_continuation_anchor_ranking_prefers_clean_noncollapsed_topology():
    fragmented_corner = controls.FrozenWindowMetrics(
        leakage=0.7602462,
        missing=0.9394390,
        activity_area_ratio=0.8208072,
        gradient_leakage=np.zeros(2),
        gradient_missing=np.zeros(2),
    )
    wide_connected = controls.FrozenWindowMetrics(
        leakage=8.166446,
        missing=0.266768,
        activity_area_ratio=8.899678,
        gradient_leakage=np.zeros(2),
        gradient_missing=np.zeros(2),
    )
    fragmented_topology = controls.ActivityTopology(
        component_count=1,
        raw_component_count=90,
        bridge_cell_count=100,
        core_cell_count=100,
        core_weight=1.0,
        largest_fraction=0.80,
        second_fraction=0.01873454,
        component_fractions=(0.80, 0.01873454, 0.01, 0.005),
    )
    clean_topology = controls.ActivityTopology(
        component_count=1,
        raw_component_count=1,
        bridge_cell_count=100,
        core_cell_count=100,
        core_weight=1.0,
        largest_fraction=1.0,
        second_fraction=0.0,
        component_fractions=(1.0,),
    )

    # The geometric key intentionally prefers the low-leakage corner, while
    # the distinct continuation key recognizes the clean enclosing band as a
    # much safer branch anchor.
    assert controls.frozen_topology_repair_rank_key(
        fragmented_corner,
        leakage_max=0.30,
        missing_max=0.70,
        edge_motion=2.0,
    ) < controls.frozen_topology_repair_rank_key(
        wide_connected,
        leakage_max=0.30,
        missing_max=0.70,
        edge_motion=1.0,
    )
    assert controls.frozen_topology_continuation_rank_key(
        wide_connected,
        clean_topology,
        expected_components=1,
        leakage_max=0.30,
        missing_max=0.70,
        edge_motion=1.0,
    ) < controls.frozen_topology_continuation_rank_key(
        fragmented_corner,
        fragmented_topology,
        expected_components=1,
        leakage_max=0.30,
        missing_max=0.70,
        edge_motion=2.0,
    )


def test_continuation_anchor_ranking_validates_expected_components():
    metrics = controls.FrozenWindowMetrics(
        leakage=0.1,
        missing=0.2,
        activity_area_ratio=0.9,
        gradient_leakage=np.zeros(2),
        gradient_missing=np.zeros(2),
    )
    topology = controls.ActivityTopology(
        component_count=1,
        raw_component_count=1,
        bridge_cell_count=1,
        core_cell_count=1,
        core_weight=1.0,
        largest_fraction=1.0,
        second_fraction=0.0,
        component_fractions=(1.0,),
    )
    with pytest.raises(ValueError):
        controls.frozen_topology_continuation_rank_key(
            metrics,
            topology,
            expected_components=0,
            leakage_max=0.3,
            missing_max=0.7,
            edge_motion=1.0,
        )


def test_restoration_reduces_sum_of_geometric_violations():
    values = np.asarray([0.25, 0.1])
    gradients = np.asarray([[-1.0, 0.2], [0.1, -1.0]])
    rows = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    rhs = np.ones(4)
    result = controls.solve_restoration_step_2d(
        np.asarray([0.2, -0.1]), np.eye(2), values, gradients, rows, rhs
    )
    assert result.success
    assert result.predicted_violation < controls.normalized_violation(values)


def test_threshold_functional_progress_uses_violation_while_infeasible():
    progress = controls.threshold_functional_progress(
        old_objective=0.08,
        new_objective=0.09,
        old_leakage=0.3200,
        new_leakage=0.3199,
        old_missing=0.48,
        new_missing=0.48,
        leakage_max=0.30,
        missing_max=0.70,
        feasibility_tolerance=1.0e-9,
        absolute_tolerance=1.0e-6,
        relative_tolerance=1.0e-2,
    )
    assert progress.mode == "geometric_violation"
    assert progress.improvement == pytest.approx(1.0e-4)
    assert progress.required_improvement == pytest.approx(2.01e-4)
    assert progress.stagnated


def test_threshold_functional_progress_uses_objective_after_feasibility():
    progress = controls.threshold_functional_progress(
        old_objective=0.08,
        new_objective=0.079,
        old_leakage=0.29,
        new_leakage=0.295,
        old_missing=0.48,
        new_missing=0.49,
        leakage_max=0.30,
        missing_max=0.70,
        feasibility_tolerance=1.0e-9,
        objective_scale=2.0,
        absolute_tolerance=1.0e-6,
        relative_tolerance=1.0e-2,
    )
    assert progress.mode == "scaled_objective"
    assert progress.improvement == pytest.approx(2.0e-3)
    assert progress.required_improvement == pytest.approx(1.601e-3)
    assert not progress.stagnated


def test_threshold_functional_progress_validates_tolerances():
    with pytest.raises(ValueError):
        controls.threshold_functional_progress(
            old_objective=0.08,
            new_objective=0.079,
            old_leakage=0.29,
            new_leakage=0.28,
            old_missing=0.48,
            new_missing=0.47,
            leakage_max=0.30,
            missing_max=0.70,
            feasibility_tolerance=1.0e-9,
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
        )


def test_reduced_kkt_detects_bound_constrained_optimum():
    # min -x subject to x <= 1, with an inactive y box row.
    gradient = np.asarray([-1.0, 0.0])
    values = np.asarray([0.0, -2.0])
    gradients = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    result = controls.reduced_kkt_residual(
        gradient, values, gradients, active_tolerance=1.0e-9
    )
    assert result.residual < 1.0e-12
    assert result.multipliers[0] == pytest.approx(1.0)
    assert result.multipliers[1] == 0.0


def test_h1_objective_and_dual_residual_are_basis_invariant():
    rng = np.random.default_rng(818)
    factor = rng.normal(size=(5, 5))
    stiffness = factor.T @ factor + np.eye(5)
    error = rng.normal(size=5)
    residual = rng.normal(size=5)
    transform = rng.normal(size=(5, 5))
    while abs(np.linalg.det(transform)) < 0.1:
        transform = rng.normal(size=(5, 5))
    # Old coefficients x and new coefficients y satisfy x=T y.
    new_stiffness = transform.T @ stiffness @ transform
    new_error = np.linalg.solve(transform, error)
    new_residual = transform.T @ residual
    energy = error @ stiffness @ error
    new_energy = new_error @ new_stiffness @ new_error
    dual = residual @ np.linalg.solve(stiffness, residual)
    new_dual = new_residual @ np.linalg.solve(new_stiffness, new_residual)
    assert new_energy == pytest.approx(energy, rel=2.0e-12)
    assert new_dual == pytest.approx(dual, rel=2.0e-12)
    # The Euclidean coefficient norm is intentionally not invariant.
    assert not np.isclose(np.linalg.norm(new_residual), np.linalg.norm(residual))


def test_missing_area_rejects_collapsed_activity():
    values = np.zeros(20)
    target = np.ones(20)
    weights = np.ones(20)
    metric = controls.frozen_window_metrics(values, target, weights, 10.0, 0.2, 0.08)
    assert metric.missing > 0.999
    assert metric.missing > 0.2


def test_collapsed_strict_equilibrium_is_not_an_usable_initial_candidate():
    assert not controls.initialization_candidate_is_usable(
        strict_success=True, missing=1.0, minimum_coverage=0.01
    )
    assert controls.initialization_candidate_is_usable(
        strict_success=True, missing=0.9, minimum_coverage=0.01
    )
    assert not controls.initialization_candidate_is_usable(
        strict_success=False, missing=0.0, minimum_coverage=0.01
    )
    assert not controls.initialization_candidate_is_usable(
        strict_success=True, missing=float("nan"), minimum_coverage=0.01
    )
    with pytest.raises(ValueError, match="minimum_coverage"):
        controls.initialization_candidate_is_usable(
            strict_success=True, missing=0.0, minimum_coverage=0.0
        )


def test_leakage_gradient_predicts_small_threshold_change():
    values = np.linspace(-1.0, 1.0, 501)
    target = (np.abs(values) < 0.2).astype(float)
    weights = np.ones_like(values)
    point = np.asarray([0.05, 0.55])
    metric = controls.frozen_window_metrics(values, target, weights, *point, 0.1)
    direction = -metric.gradient_leakage
    direction /= np.linalg.norm(direction)
    step = 1.0e-5
    changed = controls.frozen_window_metrics(
        values, target, weights, *(point + step * direction), 0.1
    )
    predicted = step * float(metric.gradient_leakage @ direction)
    actual = changed.leakage - metric.leakage
    assert predicted < 0.0
    assert actual < 0.0
    assert actual == pytest.approx(predicted, rel=2.0e-3, abs=1.0e-10)


def test_shortlist_is_bounded_unique_and_small():
    bounds = controls.CenterWidthBounds(0.0, 1.0, 0.1, 0.5)
    shortlist = controls.initialization_shortlist(0.99, 0.49, bounds, count=5)
    assert 3 <= len(shortlist) <= 5
    pairs = [(center, width) for _, center, width in shortlist]
    assert len(set(pairs)) == len(pairs)
    assert all(bounds.contains(*pair) for pair in pairs)
