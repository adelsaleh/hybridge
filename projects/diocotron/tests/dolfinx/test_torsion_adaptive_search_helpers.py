from __future__ import annotations

import math

import pytest

from projects.diocotron.dolfinx.torsion.search.helpers import (
    CandidateObservation,
    RankingCriteria,
    adaptive_generation_limit,
    auto_initial_node_count,
    auto_rank_layout,
    auto_search_budget,
    candidate_rank,
    chebyshev_lobatto_nodes,
    chebyshev_triangle_pairs,
    curve_qualified,
    continuation_rank,
    deterministic_candidate_id,
    is_dominance_witness,
    make_threshold_pair,
    pareto_frontier,
    partition_candidate_ids,
    propose_frontier_midpoints,
    prune_width_dominated,
    select_continuation_parent,
    threshold_pair_key,
)


CRITERIA = RankingCriteria(
    curve1_tolerance=0.01,
    curve2_tolerance=0.01,
    minimum_containment=0.9,
    curve1_spread_limit=0.03,
    curve2_spread_limit=0.03,
)


def pair(c1: float, c2: float, generation: int = 0):
    return make_threshold_pair(
        namespace="test",
        generation=generation,
        c1_hat=c1,
        c2_hat=c2,
        source="test",
    )


def observation(
    c1: float,
    c2: float,
    *,
    r1: float,
    r2: float,
    leakage: float = 0.2,
    missing: float = 0.2,
    containment: float = 0.95,
    spread: float = 0.02,
    pde: bool = True,
    active: bool = True,
    thick: bool = False,
    group: int | None = None,
    thresholds_active: bool = False,
    nontrivial: bool = False,
) -> CandidateObservation:
    return CandidateObservation(
        pair=pair(c1, c2),
        pde_converged=pde,
        bounds_satisfied=True,
        contours_active=active,
        too_thick=thick,
        curve1_residual=r1,
        curve2_residual=r2,
        curve1_containment=containment,
        curve2_containment=containment,
        curve1_spread=spread,
        curve2_spread=spread,
        leakage_rel=leakage,
        missing_rel=missing,
        nonlinear_residual=1.0e-7,
        claimed_group=group,
        thresholds_active=thresholds_active,
        nontrivial_field=nontrivial,
    )


def test_deterministic_pair_ids_are_normalized_and_namespace_local():
    first = make_threshold_pair(
        namespace="geometry-a",
        generation=0,
        c1_hat=0.2,
        c2_hat=0.6,
    )
    later = make_threshold_pair(
        namespace="geometry-a",
        generation=4,
        c1_hat=0.2 + 1.0e-16,
        c2_hat=0.6,
    )
    assert first.candidate_id == later.candidate_id
    assert deterministic_candidate_id("geometry-b", 0.2, 0.6) != first.candidate_id
    assert first.absolute(3.0) == pytest.approx((0.6, 1.8))
    with pytest.raises(ValueError):
        threshold_pair_key(0.5, 0.5)


def test_chebyshev_grid_covers_complete_loose_triangle():
    nodes = chebyshev_lobatto_nodes(9)
    pairs = chebyshev_triangle_pairs(9, namespace="full-triangle")
    assert nodes[0] == 0.0
    assert nodes[-1] == 1.0
    assert nodes[4] == pytest.approx(0.5)
    assert len(pairs) == 9 * 8 // 2
    assert len({item.candidate_id for item in pairs}) == len(pairs)
    assert pair_keys(pairs) == {
        threshold_pair_key(c1, c2)
        for index, c1 in enumerate(nodes[:-1])
        for c2 in nodes[index + 1 :]
    }


def pair_keys(pairs):
    return {item.key for item in pairs}


def test_auto_node_and_budget_sizing_tracks_parallel_waves():
    assert auto_initial_node_count(20) == 11  # 55 pairs >= two 20-group waves
    assert auto_initial_node_count(5) == 7
    assert auto_initial_node_count(20, candidate_budget=40) == 9
    budget = auto_search_budget(20)
    assert budget.initial_nodes == 11
    assert budget.initial_candidates == 55
    assert budget.adaptive_candidates_per_generation == 40
    assert budget.total_candidate_budget == 295
    capped = auto_search_budget(20, maximum_candidates=100)
    assert capped.initial_nodes == 11
    assert capped.initial_candidates == 55
    assert capped.adaptive_candidates_per_generation == 7
    assert capped.total_candidate_budget == 100
    assert (
        capped.initial_candidates
        + capped.adaptive_generations * capped.adaptive_candidates_per_generation
        <= capped.total_candidate_budget
    )


@pytest.mark.parametrize(
    ("world", "per_candidate", "active", "groups", "idle"),
    [
        (20, 1, 20, 20, 0),
        (20, 2, 20, 10, 0),
        (20, 4, 20, 5, 0),
        (20, 8, 16, 2, 4),
        (20, 12, 12, 1, 8),
        (20, 16, 16, 1, 4),
    ],
)
def test_auto_rank_layout(world, per_candidate, active, groups, idle):
    layout = auto_rank_layout(world, per_candidate)
    assert (layout.active_ranks, layout.group_count, layout.idle_ranks) == (
        active,
        groups,
        idle,
    )
    assert layout.group_for_world_rank(0) == 0
    assert layout.group_for_world_rank(active - 1) == groups - 1
    if idle:
        assert layout.group_for_world_rank(active) is None


def test_width_dominance_prunes_only_wider_points_on_same_row_or_column():
    thick = pair(0.2, 0.6)
    candidates = [
        pair(0.2, 0.7),  # fixed c1, larger c2
        pair(0.1, 0.6),  # fixed c2, smaller c1
        pair(0.2, 0.5),  # narrower at fixed c1
        pair(0.3, 0.6),  # narrower at fixed c2
        pair(0.1, 0.7),  # neither exact row nor exact column
    ]
    result = prune_width_dominated(candidates, thick_pairs=[thick])
    assert pair_keys(result.kept) == {
        pair(0.2, 0.5).key,
        pair(0.3, 0.6).key,
        pair(0.1, 0.7).key,
    }
    assert {item.reason for item in result.rejected} == {
        "fixed_c1_larger_c2",
        "fixed_c2_smaller_c1",
    }
    assert all(item.witness_id == thick.candidate_id for item in result.rejected)


def dominance_row() -> dict[str, float | int]:
    return {
        "tooThick": 1,
        "converged": 1,
        "boundSatisfied": 1,
        "lowerCurveActive": 1,
        "upperCurveActive": 1,
        "lowerCurveResolved": 1,
        "upperCurveResolved": 1,
        "monotoneForBisection": 1,
        "bothThresholdsActive": 1,
        "activityAreaRel": 0.75,
        "targetUnderresolved": 0,
    }


def test_dominance_witness_accepts_only_fully_vetted_computed_branch():
    assert is_dominance_witness(dominance_row())


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("tooThick", 0),
        ("converged", 0),
        ("boundSatisfied", 0),
        ("lowerCurveActive", 0),
        ("upperCurveActive", 0),
        ("lowerCurveResolved", 0),
        ("upperCurveResolved", 0),
        ("monotoneForBisection", 0),
        ("bothThresholdsActive", 0),
        ("activityAreaRel", 1.0e-9),
        ("targetUnderresolved", 1),
        ("converged", math.nan),
        ("converged", "unknown"),
    ],
)
def test_dominance_witness_fails_closed_for_every_required_gate(field, invalid):
    row = dominance_row()
    row[field] = invalid
    assert not is_dominance_witness(row)


def test_dominance_witness_fails_closed_when_a_required_diagnostic_is_missing():
    for field in tuple(dominance_row()):
        row = dominance_row()
        del row[field]
        assert not is_dominance_witness(row), field


def test_candidate_ranking_qualifies_curves_then_minimizes_leakage_and_missing():
    qualified_worse_area = observation(0.2, 0.6, r1=0.005, r2=-0.005, leakage=0.4, missing=0.4)
    qualified_better_area = observation(0.3, 0.7, r1=0.009, r2=-0.009, leakage=0.1, missing=0.2)
    unqualified = observation(0.25, 0.65, r1=0.02, r2=0.0, leakage=0.0, missing=0.0)
    assert curve_qualified(qualified_worse_area, CRITERIA)
    assert candidate_rank(qualified_better_area, CRITERIA) < candidate_rank(
        qualified_worse_area, CRITERIA
    )
    assert candidate_rank(qualified_worse_area, CRITERIA) < candidate_rank(
        unqualified, CRITERIA
    )


def test_continuation_prefers_active_partial_branch_over_converged_zero():
    partial = observation(
        0.25,
        0.55,
        r1=0.03,
        r2=-0.02,
        pde=False,
        active=True,
        thresholds_active=True,
        nontrivial=True,
    )
    zero = observation(
        0.30,
        0.60,
        r1=math.inf,
        r2=math.inf,
        pde=True,
        active=False,
        thresholds_active=False,
        nontrivial=False,
    )

    assert continuation_rank(partial, CRITERIA) < continuation_rank(zero, CRITERIA)


def test_parent_selection_is_deterministic_and_uses_active_partial_state():
    partial = observation(
        0.2,
        0.6,
        r1=0.03,
        r2=-0.02,
        pde=False,
        thresholds_active=True,
        nontrivial=True,
    )
    zero = observation(
        0.4,
        0.8,
        r1=math.inf,
        r2=math.inf,
        pde=True,
        active=False,
    )
    child = pair(0.3, 0.7, generation=1)
    parent_ids = (zero.pair.candidate_id, partial.pair.candidate_id)

    forward = select_continuation_parent(
        parent_ids,
        [zero, partial],
        criteria=CRITERIA,
        child_pair=child,
    )
    reverse = select_continuation_parent(
        reversed(parent_ids),
        [partial, zero],
        criteria=CRITERIA,
        child_pair=child,
    )

    assert forward is partial
    assert reverse is partial


def test_generation_limit_reserves_future_refinement_budget():
    assert adaptive_generation_limit(
        total_cap=256,
        completed=37,
        remaining_generations=4,
        desired=40,
    ) == 40
    assert adaptive_generation_limit(
        total_cap=100,
        completed=91,
        remaining_generations=4,
        desired=40,
    ) == 3
    assert adaptive_generation_limit(
        total_cap=0,
        completed=500,
        remaining_generations=2,
        desired=40,
    ) == 40


def test_render_partition_is_deterministic_disjoint_and_complete():
    identifiers = [17, 3, 11, 2, 29, 5, 23]
    forward = partition_candidate_ids(identifiers, 4)
    reverse = partition_candidate_ids(list(reversed(identifiers)), 4)

    assert forward == reverse
    flattened = [identifier for partition in forward for identifier in partition]
    assert sorted(flattened) == sorted(identifiers)
    assert len(flattened) == len(set(flattened))
    assert max(map(len, forward)) - min(map(len, forward)) <= 1


def test_pareto_frontier_removes_strictly_dominated_observation():
    best = observation(0.2, 0.6, r1=0.002, r2=0.002, leakage=0.1, missing=0.1)
    dominated = observation(0.3, 0.7, r1=0.004, r2=0.004, leakage=0.2, missing=0.2)
    tradeoff = observation(0.4, 0.8, r1=0.001, r2=0.008, leakage=0.05, missing=0.25)
    frontier = pareto_frontier([dominated, tradeoff, best], CRITERIA)
    assert dominated.pair.candidate_id not in {item.pair.candidate_id for item in frontier}
    assert {item.pair.candidate_id for item in frontier} == {
        best.pair.candidate_id,
        tradeoff.pair.candidate_id,
    }


def test_frontier_midpoints_include_axis_bisections_and_are_deterministic():
    rows = [
        observation(0.2, 0.6, r1=-0.02, r2=-0.02, group=2),
        observation(0.4, 0.6, r1=0.02, r2=-0.01, group=2),
        observation(0.2, 0.8, r1=-0.01, r2=0.02, group=3),
        observation(0.4, 0.8, r1=0.03, r2=0.03, leakage=0.5, missing=0.5),
    ]
    forward = propose_frontier_midpoints(
        rows,
        criteria=CRITERIA,
        namespace="adaptive",
        generation=1,
        maximum_candidates=20,
    )
    reverse = propose_frontier_midpoints(
        list(reversed(rows)),
        criteria=CRITERIA,
        namespace="adaptive",
        generation=1,
        maximum_candidates=20,
    )
    assert [item.candidate_id for item in forward] == [item.candidate_id for item in reverse]
    keys = pair_keys(forward)
    assert threshold_pair_key(0.3, 0.6) in keys
    assert threshold_pair_key(0.2, 0.7) in keys
    assert all(0.0 <= item.c1_hat < item.c2_hat <= 1.0 for item in forward)
    assert len(keys) == len(forward)


def test_frontier_refinement_obeys_evaluated_and_width_dominance_filters():
    rows = [
        observation(0.2, 0.6, r1=-0.02, r2=-0.02),
        observation(0.4, 0.6, r1=0.02, r2=-0.01),
        observation(0.2, 0.8, r1=-0.01, r2=0.02),
    ]
    evaluated_midpoint = pair(0.3, 0.6)
    thick = pair(0.2, 0.65)
    proposals = propose_frontier_midpoints(
        rows,
        criteria=CRITERIA,
        namespace="test",
        generation=1,
        maximum_candidates=20,
        evaluated_ids=[evaluated_midpoint.candidate_id],
        thick_pairs=[thick],
    )
    assert evaluated_midpoint.candidate_id not in {item.candidate_id for item in proposals}
    assert threshold_pair_key(0.2, 0.7) not in pair_keys(proposals)


def test_invalid_search_inputs_fail_early():
    with pytest.raises(ValueError):
        auto_rank_layout(4, 8)
    with pytest.raises(ValueError):
        auto_initial_node_count(20, candidate_budget=5)
    with pytest.raises(ValueError):
        RankingCriteria(0.0, 0.1, 0.9, 0.1, 0.1)
    with pytest.raises(ValueError):
        make_threshold_pair(namespace="x", generation=-1, c1_hat=0.1, c2_hat=0.2)
