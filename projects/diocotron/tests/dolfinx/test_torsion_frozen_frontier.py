"""Unit tests for frozen pushforward-frontier threshold selection."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[4] / "projects/diocotron/dolfinx"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from projects.diocotron.dolfinx.torsion.initialization.frozen_frontier import (  # noqa: E402
    PushforwardHistogram,
    hard_window_pareto_frontier,
    sampled_logistic_window_metrics,
    select_frozen_frontier_point,
    weighted_pushforward_histogram,
)


def _brute_nondominated_intervals(
        histogram: PushforwardHistogram,
        *,
        min_width: float,
) -> set[tuple[int, int]]:
    """Reference Pareto enumeration for a deliberately tiny histogram."""

    edges = histogram.edges
    records: list[tuple[int, int, float, float]] = []
    for lower in range(histogram.target_mass.size):
        for upper in range(lower + 1, histogram.target_mass.size + 1):
            if edges[upper] - edges[lower] < min_width:
                continue
            overlap = float(np.sum(histogram.target_mass[lower:upper]))
            if overlap <= 0.0:
                continue
            leakage = float(np.sum(histogram.outside_mass[lower:upper]))
            missing = histogram.target_mass_total - overlap
            records.append((lower, upper, leakage, missing))

    nondominated: list[tuple[int, int, float, float]] = []
    for record in records:
        _, _, leakage, missing = record
        dominated = any(
            other_leakage <= leakage
            and other_missing <= missing
            and (other_leakage < leakage or other_missing < missing)
            for _, _, other_leakage, other_missing in records
        )
        if not dominated:
            nondominated.append(record)

    # Equal metric pairs represent the same Pareto point.  Match the
    # production rule: retain the narrowest interval, then the lowest c1.
    by_metrics: dict[tuple[float, float], tuple[int, int]] = {}
    for lower, upper, leakage, missing in nondominated:
        key = (leakage, missing)
        incumbent = by_metrics.get(key)
        if incumbent is None:
            by_metrics[key] = (lower, upper)
            continue
        old_lower, old_upper = incumbent
        old_key = (edges[old_upper] - edges[old_lower], edges[old_lower])
        new_key = (edges[upper] - edges[lower], edges[lower])
        if new_key < old_key:
            by_metrics[key] = (lower, upper)
    return set(by_metrics.values())


def test_weighted_pushforward_splits_target_and_complement_mass() -> None:
    histogram = weighted_pushforward_histogram(
        np.array([-0.1, 0.1, 0.4, 0.8, 1.2]),
        np.array([1.0, 1.0, 0.25, 0.0, 1.0]),
        np.array([2.0, 3.0, 4.0, 5.0, 7.0]),
        lower=0.0,
        upper=1.0,
        bins=4,
    )

    assert histogram.target_mass_total == pytest.approx(13.0)
    assert histogram.outside_mass_total == pytest.approx(8.0)
    assert histogram.target_mass_in_range == pytest.approx(4.0)
    assert histogram.outside_mass_in_range == pytest.approx(8.0)
    assert histogram.sample_count == 5


def test_hard_frontier_matches_complete_brute_force_enumeration() -> None:
    histogram = PushforwardHistogram(
        edges=np.linspace(0.0, 1.0, 6),
        target_mass=np.array([1.0, 4.0, 2.0, 5.0, 3.0]),
        outside_mass=np.array([5.0, 1.0, 4.0, 2.0, 3.0]),
        target_mass_total=15.0,
        outside_mass_total=15.0,
        target_mass_in_range=15.0,
        outside_mass_in_range=15.0,
        sample_count=10,
    )
    min_width = 0.3
    frontier = hard_window_pareto_frontier(
        histogram,
        min_width=min_width,
    )

    actual = {(point.lower_bin, point.upper_bin) for point in frontier.points}
    expected = _brute_nondominated_intervals(
        histogram,
        min_width=min_width,
    )
    assert actual == expected
    assert frontier.evaluated_intervals == 10
    for point in frontier.points:
        assert point.overlap == pytest.approx(frontier.target_area - point.missing)
        assert point.jaccard == pytest.approx(
            point.overlap / (frontier.target_area + point.leakage)
        )


def test_frontier_selection_never_relaxes_a_strict_cap() -> None:
    histogram = PushforwardHistogram(
        edges=np.linspace(0.0, 1.0, 5),
        target_mass=np.array([1.0, 5.0, 4.0, 2.0]),
        outside_mass=np.array([4.0, 1.0, 2.0, 5.0]),
        target_mass_total=12.0,
        outside_mass_total=12.0,
        target_mass_in_range=12.0,
        outside_mass_in_range=12.0,
        sample_count=8,
    )
    frontier = hard_window_pareto_frontier(histogram, min_width=0.2)
    leakage_values = sorted({point.leakage for point in frontier.points})
    cap = leakage_values[min(1, len(leakage_values) - 1)]
    eligible = [point for point in frontier.points if point.leakage <= cap]

    capped = select_frozen_frontier_point(frontier, leakage_cap=cap)
    assert capped == min(
        eligible,
        key=lambda point: (
            point.missing,
            point.leakage,
            point.width,
            point.c1,
            point.c2,
        ),
    )
    minimum_leakage = min(point.leakage for point in frontier.points)
    with pytest.raises(ValueError, match="no nontrivial.*satisfies leakage cap"):
        select_frozen_frontier_point(
            frontier,
            leakage_cap=minimum_leakage - 0.5,
        )

    uncapped = select_frozen_frontier_point(frontier, leakage_cap=None)
    assert uncapped == min(
        frontier.points,
        key=lambda point: (
            -point.jaccard,
            point.missing,
            point.leakage,
            point.width,
            point.c1,
            point.c2,
        ),
    )


@pytest.mark.parametrize(
    ("eps_mode", "eps_ratio", "eps_fixed"),
    [("relative", 0.17, None), ("fixed", 0.0, 0.08)],
)
def test_logistic_threshold_gradients_match_centered_differences(
        eps_mode: str,
        eps_ratio: float,
        eps_fixed: float | None,
) -> None:
    rng = np.random.default_rng(20260902)
    phi = rng.uniform(0.0, 1.0, 400)
    target = (rng.uniform(0.0, 1.0, 400) > 0.45).astype(float)
    weights = rng.uniform(0.2, 1.5, 400)
    c1 = 0.29
    c2 = 0.71
    step = 2.0e-7
    base = sampled_logistic_window_metrics(
        phi,
        target,
        weights,
        c1=c1,
        c2=c2,
        eps_mode=eps_mode,
        eps_ratio=eps_ratio,
        eps_fixed=eps_fixed,
    )

    for component in range(2):
        plus_thresholds = [c1, c2]
        minus_thresholds = [c1, c2]
        plus_thresholds[component] += step
        minus_thresholds[component] -= step
        plus = sampled_logistic_window_metrics(
            phi,
            target,
            weights,
            c1=plus_thresholds[0],
            c2=plus_thresholds[1],
            eps_mode=eps_mode,
            eps_ratio=eps_ratio,
            eps_fixed=eps_fixed,
        )
        minus = sampled_logistic_window_metrics(
            phi,
            target,
            weights,
            c1=minus_thresholds[0],
            c2=minus_thresholds[1],
            eps_mode=eps_mode,
            eps_ratio=eps_ratio,
            eps_fixed=eps_fixed,
        )
        assert base.grad_leakage[component] == pytest.approx(
            (plus.leakage - minus.leakage) / (2.0 * step),
            rel=2.0e-7,
            abs=2.0e-7,
        )
        assert base.grad_missing[component] == pytest.approx(
            (plus.missing - minus.missing) / (2.0 * step),
            rel=2.0e-7,
            abs=2.0e-7,
        )
        assert base.grad_jaccard[component] == pytest.approx(
            (plus.jaccard - minus.jaccard) / (2.0 * step),
            rel=5.0e-7,
            abs=5.0e-8,
        )
