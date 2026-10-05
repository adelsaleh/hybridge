"""Control-space utilities for the torsion H1 projection optimizer.

The functions in this module deliberately have no DOLFINx dependency.  They
centralize the center--width window derivatives, the frozen hard-window scan,
and the complete two-dimensional trust-region algebra used by
``dolfinx_torsion_h1_projection_reduced_optimization.py``.  Keeping these
small calculations separate makes it possible to verify them with ordinary
NumPy tests while the distributed finite-element assembly remains in the
existing torsion solver modules.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class CenterWidthBounds:
    """Explicit box bounds for the center ``m`` and width ``d``."""

    center_min: float
    center_max: float
    width_min: float
    width_max: float

    def __post_init__(self) -> None:
        """Reject nonfinite, reversed, or nonpositive box bounds."""
        values = (
            self.center_min,
            self.center_max,
            self.width_min,
            self.width_max,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("center-width bounds must be finite")
        if self.center_min >= self.center_max:
            raise ValueError("center_min must be smaller than center_max")
        if self.width_min <= 0.0 or self.width_min >= self.width_max:
            raise ValueError("width bounds must satisfy 0 < width_min < width_max")

    @property
    def lower(self) -> np.ndarray:
        """Return ``[m_min, d_min]``."""
        return np.asarray([self.center_min, self.width_min], dtype=np.float64)

    @property
    def upper(self) -> np.ndarray:
        """Return ``[m_max, d_max]``."""
        return np.asarray([self.center_max, self.width_max], dtype=np.float64)

    @property
    def scale(self) -> np.ndarray:
        """Return the two box spans used to scale trust-region steps."""
        return self.upper - self.lower

    def contains(self, center: float, width: float, *, tolerance: float = 0.0) -> bool:
        """Return whether ``(center,width)`` satisfies the explicit box."""
        return bool(
            self.center_min - tolerance <= center <= self.center_max + tolerance
            and self.width_min - tolerance <= width <= self.width_max + tolerance
        )

    def clip(self, center: float, width: float) -> tuple[float, float]:
        """Clip a center--width pair to the explicit box."""
        return (
            float(np.clip(center, self.center_min, self.center_max)),
            float(np.clip(width, self.width_min, self.width_max)),
        )


@dataclass(frozen=True)
class WindowEvaluation:
    """Pointwise activity and its explicit center--width derivatives."""

    activity: np.ndarray
    state_derivative: np.ndarray
    center_derivative: np.ndarray
    width_derivative: np.ndarray


@dataclass(frozen=True)
class FrozenWindowMetrics:
    """Frozen-state normalized geometry and explicit gradients."""

    leakage: float
    missing: float
    activity_area_ratio: float
    gradient_leakage: np.ndarray
    gradient_missing: np.ndarray

    @property
    def coverage(self) -> float:
        """Return frozen target coverage ``1-M``."""
        return 1.0 - self.missing


@dataclass(frozen=True)
class HardWindowSelection:
    """Best interval found by the sorted quadrature hard-window scan."""

    center: float
    width: float
    c1: float
    c2: float
    leakage: float
    missing: float
    target_coverage: float
    target_weight: float
    outside_weight: float
    left_index: int
    right_index: int


@dataclass(frozen=True)
class FrozenWindowReachabilityBound:
    """Conservative target-coverage bounds for an explicit control box."""

    lower_edge_min: float
    upper_edge_max: float
    epsilon_max: float
    target_potential_min: float
    target_potential_max: float
    hard_coverage_upper_bound: float
    smooth_coverage_upper_bound: float


@dataclass(frozen=True)
class QPResult:
    """Result of a convex two-dimensional linearly constrained QP."""

    success: bool
    step: np.ndarray
    objective: float
    active_rows: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class RestorationResult:
    """Lexicographic feasibility-restoration step."""

    success: bool
    step: np.ndarray
    predicted_violation: float
    model_objective: float
    reason: str


@dataclass(frozen=True)
class ThresholdFunctionalProgress:
    """Progress of the primary threshold merit used during homotopy.

    The homotopy threshold corrector is an initializer, not the final
    constrained optimizer.  While its current point is infeasible, its
    primary merit is the sum of positive leakage and missing-area violations.
    At a feasible point, the (possibly homotopy-scaled) projection objective
    becomes the primary merit instead.
    """

    mode: str
    old_merit: float
    new_merit: float
    improvement: float
    required_improvement: float
    stagnated: bool


@dataclass(frozen=True)
class KKTResult:
    """Reduced first-order KKT diagnostic for the two control variables."""

    residual: float
    stationarity: float
    primal_infeasibility: float
    complementarity: float
    multipliers: np.ndarray
    active_rows: tuple[int, ...]


@dataclass(frozen=True)
class ActivityTopology:
    """Hysteretic connected-component summary of a discrete activity band.

    ``component_count`` includes only components carrying at least
    ``min_component_fraction`` of the total core weight.  ``raw_component_count``
    retains every bridge-connected component containing a core cell and is
    useful for diagnosing small numerical islands.
    """

    component_count: int
    raw_component_count: int
    bridge_cell_count: int
    core_cell_count: int
    core_weight: float
    largest_fraction: float
    second_fraction: float
    component_fractions: tuple[float, ...]


def hysteretic_activity_components(
    cell_facets: np.ndarray,
    cell_peak_activity: np.ndarray,
    *,
    core_level: float,
    bridge_level: float,
    min_component_fraction: float,
    cell_weights: np.ndarray | None = None,
) -> ActivityTopology:
    """Count significant activity components through a lower-level bridge.

    Cells whose sampled peak is at least ``bridge_level`` form the graph and
    are adjacent when they share a mesh facet.  Only graph components that
    contain cells at or above ``core_level`` are reported.  This hysteresis
    makes the diagnostic substantially less sensitive to one interpolation
    point sitting infinitesimally below the nominal activity contour.

    Args:
        cell_facets: Integer array of shape ``(n_cells, facets_per_cell)``.
            Facet identifiers must be globally unique, but need not be dense.
        cell_peak_activity: Maximum sampled activity in each cell.
        core_level: Level identifying the material core.
        bridge_level: Lower level through which core regions may connect.
        min_component_fraction: Ignore a component when its core weight is
            smaller than this fraction of the total core weight.
        cell_weights: Optional positive cell weights.  Uniform weights are
            used when omitted.
    """
    facets = np.asarray(cell_facets)
    peaks = np.asarray(cell_peak_activity, dtype=np.float64).reshape(-1)
    if facets.ndim != 2 or facets.shape[0] != peaks.size:
        raise ValueError("cell_facets must have one row per activity value")
    if facets.shape[1] < 1:
        raise ValueError("each cell must have at least one facet")
    if not np.all(np.isfinite(peaks)):
        raise ValueError("cell activities must be finite")
    core_level = float(core_level)
    bridge_level = float(bridge_level)
    min_component_fraction = float(min_component_fraction)
    if not (
        math.isfinite(core_level)
        and math.isfinite(bridge_level)
        and 0.0 <= bridge_level <= core_level <= 1.0
    ):
        raise ValueError("require 0 <= bridge_level <= core_level <= 1")
    if not (
        math.isfinite(min_component_fraction)
        and 0.0 <= min_component_fraction < 1.0
    ):
        raise ValueError("min_component_fraction must lie in [0,1)")

    if cell_weights is None:
        weights = np.ones(peaks.size, dtype=np.float64)
    else:
        weights = np.asarray(cell_weights, dtype=np.float64).reshape(-1)
        if weights.size != peaks.size:
            raise ValueError("cell_weights must have one value per cell")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("cell_weights must be finite and positive")

    bridge_cells = np.flatnonzero(peaks >= bridge_level)
    core_mask = peaks >= core_level
    core_cells = np.flatnonzero(core_mask)
    total_core_weight = float(np.sum(weights[core_mask]))
    if core_cells.size == 0 or bridge_cells.size == 0:
        return ActivityTopology(
            component_count=0,
            raw_component_count=0,
            bridge_cell_count=int(bridge_cells.size),
            core_cell_count=int(core_cells.size),
            core_weight=total_core_weight,
            largest_fraction=0.0,
            second_fraction=0.0,
            component_fractions=(),
        )

    # Union only bridge-active cells.  A facet map avoids constructing a
    # global sparse adjacency matrix and keeps the operation O(n_cells).
    parent = np.arange(peaks.size, dtype=np.int64)
    rank = np.zeros(peaks.size, dtype=np.int8)

    def find(index: int) -> int:
        while int(parent[index]) != index:
            parent[index] = parent[int(parent[index])]
            index = int(parent[index])
        return index

    def union(first: int, second: int) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first == root_second:
            return
        if rank[root_first] < rank[root_second]:
            root_first, root_second = root_second, root_first
        parent[root_second] = root_first
        if rank[root_first] == rank[root_second]:
            rank[root_first] += 1

    facet_owner: dict[int, int] = {}
    for cell in bridge_cells:
        cell_index = int(cell)
        for facet in facets[cell_index]:
            facet_index = int(facet)
            owner = facet_owner.get(facet_index)
            if owner is None:
                facet_owner[facet_index] = cell_index
            else:
                union(cell_index, owner)

    component_weights: dict[int, float] = {}
    for cell in core_cells:
        cell_index = int(cell)
        root = find(cell_index)
        component_weights[root] = component_weights.get(root, 0.0) + float(
            weights[cell_index]
        )
    fractions = sorted(
        (weight / total_core_weight for weight in component_weights.values()),
        reverse=True,
    )
    significant = tuple(
        fraction
        for fraction in fractions
        if fraction + 32.0 * np.finfo(float).eps >= min_component_fraction
    )
    # The largest component is always physically meaningful when a core is
    # present, including the intentionally permissive fraction=0 case.
    if fractions and not significant:
        significant = (fractions[0],)
    return ActivityTopology(
        component_count=len(significant),
        raw_component_count=len(fractions),
        bridge_cell_count=int(bridge_cells.size),
        core_cell_count=int(core_cells.size),
        core_weight=total_core_weight,
        largest_fraction=float(fractions[0]) if fractions else 0.0,
        second_fraction=float(fractions[1]) if len(fractions) > 1 else 0.0,
        component_fractions=tuple(float(value) for value in fractions),
    )


def thresholds_from_center_width(center: float, width: float) -> tuple[float, float]:
    """Convert ``(m,d)`` to ordered thresholds ``(c1,c2)``."""
    return float(center - 0.5 * width), float(center + 0.5 * width)


def center_width_from_thresholds(c1: float, c2: float) -> tuple[float, float]:
    """Convert ordered thresholds ``(c1,c2)`` to ``(m,d)``."""
    return float(0.5 * (c1 + c2)), float(c2 - c1)


def outward_interval_repair_candidates(
    center: float,
    width: float,
    bounds: CenterWidthBounds,
    *,
    samples: int,
) -> list[np.ndarray]:
    """Generate a nearest-first scan of interval expansions within the box.

    A hard scalar window that has split into too many components can reconnect
    only after one or both threshold edges cross a separating saddle.  This
    helper expands ``(c1,c2)`` outwards on a width-scaled grid, converts every
    pair back to center--width variables, removes points outside the explicit
    bounds, and orders the result by maximum edge motion.  It performs no PDE
    work and is intended for a frozen-state topology repair.
    """
    center = float(center)
    width = float(width)
    samples = int(samples)
    if not bounds.contains(center, width):
        raise ValueError("repair seed must lie inside the center-width bounds")
    if samples < 2:
        raise ValueError("samples must be at least two")
    c1, c2 = thresholds_from_center_width(center, width)
    c1_min = float(bounds.center_min - 0.5 * bounds.width_max)
    c2_max = float(bounds.center_max + 0.5 * bounds.width_max)

    def edge_offsets(maximum: float) -> np.ndarray:
        maximum = max(float(maximum), 0.0)
        if maximum <= 32.0 * np.finfo(float).eps * max(1.0, abs(c1), abs(c2)):
            return np.asarray([0.0])
        minimum = min(0.05 * width, maximum)
        positive = np.geomspace(minimum, maximum, samples - 1)
        return np.concatenate((np.asarray([0.0]), positive))

    lower_offsets = edge_offsets(c1 - c1_min)
    upper_offsets = edge_offsets(c2_max - c2)
    candidates: list[np.ndarray] = []
    for lower_offset in lower_offsets:
        for upper_offset in upper_offsets:
            candidate_c1 = c1 - float(lower_offset)
            candidate_c2 = c2 + float(upper_offset)
            candidate_center, candidate_width = center_width_from_thresholds(
                candidate_c1, candidate_c2
            )
            if bounds.contains(candidate_center, candidate_width, tolerance=1.0e-14):
                candidates.append(
                    np.asarray([candidate_center, candidate_width], dtype=np.float64)
                )
    unique = _unique_candidates(candidates, tolerance=1.0e-14)

    def key(point: np.ndarray) -> tuple[float, float, float, float]:
        point_c1, point_c2 = thresholds_from_center_width(*point)
        lower_motion = abs(point_c1 - c1)
        upper_motion = abs(point_c2 - c2)
        return (
            max(lower_motion, upper_motion) / width,
            (lower_motion + upper_motion) / width,
            lower_motion,
            upper_motion,
        )

    return sorted(unique, key=key)


def box_interval_repair_candidates(
    center: float,
    width: float,
    bounds: CenterWidthBounds,
    *,
    samples: int,
) -> list[np.ndarray]:
    """Generate a nearest-first center--width scan over the explicit box.

    This is the second line of frozen topology repair.  Unlike
    :func:`outward_interval_repair_candidates`, it permits translations and
    contractions of the threshold interval.  That matters when the closest
    topology-compatible scalar band lies on another side of a critical level
    and cannot be reached by interval inclusion alone.

    The width grid is geometric so narrow admissible bands are not lost when
    ``width_max/width_min`` is large.  The original point and both endpoints
    of every box interval are included exactly.  Ordering uses maximum motion
    of either threshold edge, normalized by the original width.
    """
    center = float(center)
    width = float(width)
    samples = int(samples)
    if not bounds.contains(center, width):
        raise ValueError("repair seed must lie inside the center-width bounds")
    if samples < 2:
        raise ValueError("samples must be at least two")

    center_values = np.unique(
        np.concatenate(
            (
                np.asarray([center], dtype=np.float64),
                np.linspace(bounds.center_min, bounds.center_max, samples),
            )
        )
    )
    width_values = np.unique(
        np.concatenate(
            (
                np.asarray([width], dtype=np.float64),
                np.geomspace(bounds.width_min, bounds.width_max, samples),
            )
        )
    )
    candidates = [
        np.asarray([candidate_center, candidate_width], dtype=np.float64)
        for candidate_center in center_values
        for candidate_width in width_values
    ]
    unique = _unique_candidates(candidates, tolerance=1.0e-14)
    original_c1, original_c2 = thresholds_from_center_width(center, width)

    def key(point: np.ndarray) -> tuple[float, float, float, float]:
        point_c1, point_c2 = thresholds_from_center_width(*point)
        lower_motion = abs(point_c1 - original_c1)
        upper_motion = abs(point_c2 - original_c2)
        return (
            max(lower_motion, upper_motion) / width,
            (lower_motion + upper_motion) / width,
            lower_motion,
            upper_motion,
        )

    return sorted(unique, key=key)


def _logistic_and_q(argument: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a clipped logistic and ``sigma*(1-sigma)`` stably."""
    x = np.asarray(argument, dtype=np.float64)
    sigma = np.empty_like(x)
    high = x > 50.0
    low = x < -50.0
    middle = ~(high | low)
    sigma[high] = 1.0
    sigma[low] = 0.0
    sigma[middle] = 1.0 / (1.0 + np.exp(-x[middle]))
    return sigma, sigma * (1.0 - sigma)


def window_center_width(
    values: np.ndarray | Sequence[float] | float,
    center: float,
    width: float,
    eps_ratio: float,
) -> WindowEvaluation:
    """Evaluate ``W`` and all explicit derivatives for ``eps=eps_ratio*d``.

    The returned derivatives hold the state fixed.  In particular the width
    derivative contains the full chain-rule contribution through epsilon.

    Args:
        values: Potential samples.
        center: Window center ``m``.
        width: Window width ``d``.
        eps_ratio: Positive ratio ``r_eps`` in ``eps=r_eps*d``.

    Returns:
        Activity, ``W_phi``, ``partial_m W``, and ``partial_d W`` arrays.
    """
    width = float(width)
    eps_ratio = float(eps_ratio)
    if width <= 0.0:
        raise ValueError("window width must be positive")
    if eps_ratio <= 0.0:
        raise ValueError("eps_ratio must be positive")
    value_array = np.asarray(values, dtype=np.float64)
    c1, c2 = thresholds_from_center_width(float(center), width)
    eps = eps_ratio * width
    sigma1, q1 = _logistic_and_q((value_array - c1) / eps)
    sigma2, q2 = _logistic_and_q((value_array - c2) / eps)
    activity = sigma1 - sigma2
    state_derivative = (q1 - q2) / eps
    center_derivative = (-q1 + q2) / eps
    epsilon_derivative = (
        -(value_array - c1) * q1 + (value_array - c2) * q2
    ) / (eps * eps)
    width_derivative = 0.5 * (q1 + q2) / eps + eps_ratio * epsilon_derivative
    return WindowEvaluation(
        activity=np.asarray(activity),
        state_derivative=np.asarray(state_derivative),
        center_derivative=np.asarray(center_derivative),
        width_derivative=np.asarray(width_derivative),
    )


def frozen_window_metrics(
    values: np.ndarray,
    target_indicator: np.ndarray,
    weights: np.ndarray,
    center: float,
    width: float,
    eps_ratio: float,
) -> FrozenWindowMetrics:
    """Assemble frozen ``L_T``, ``M_T`` and explicit control derivatives."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    target = np.asarray(target_indicator, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size == 0 or target.shape != values.shape or weights.shape != values.shape:
        raise ValueError("values, target_indicator, and weights must be nonempty equal arrays")
    if np.any(weights < 0.0):
        raise ValueError("quadrature weights must be nonnegative")
    target_area = float(np.dot(weights, target))
    if target_area <= 0.0:
        raise ValueError("target quadrature area must be positive")
    window = window_center_width(values, center, width, eps_ratio)
    outside_weight = weights * (1.0 - target)
    target_weight = weights * target
    leakage = float(np.dot(outside_weight, window.activity) / target_area)
    missing = float(np.dot(target_weight, 1.0 - window.activity) / target_area)
    activity_ratio = float(np.dot(weights, window.activity) / target_area)
    grad_l = np.asarray(
        [
            np.dot(outside_weight, window.center_derivative),
            np.dot(outside_weight, window.width_derivative),
        ],
        dtype=np.float64,
    ) / target_area
    grad_m = -np.asarray(
        [
            np.dot(target_weight, window.center_derivative),
            np.dot(target_weight, window.width_derivative),
        ],
        dtype=np.float64,
    ) / target_area
    return FrozenWindowMetrics(leakage, missing, activity_ratio, grad_l, grad_m)


def _coalesced_sorted_samples(
    values: np.ndarray,
    target_indicator: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort quadrature samples and sum weights at identical potential values."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    target = np.asarray(target_indicator, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size == 0 or values.shape != target.shape or values.shape != weights.shape:
        raise ValueError("sample arrays must be nonempty and have equal shapes")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("sample values and weights must be finite")
    if np.any(weights < 0.0) or np.any((target < 0.0) | (target > 1.0)):
        raise ValueError("weights must be nonnegative and indicators must lie in [0,1]")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    target_weights = weights[order] * target[order]
    outside_weights = weights[order] * (1.0 - target[order])
    unique_values, first = np.unique(sorted_values, return_index=True)
    return (
        unique_values,
        np.add.reduceat(target_weights, first),
        np.add.reduceat(outside_weights, first),
    )


def hard_window_boundaries(unique_values: np.ndarray) -> np.ndarray:
    """Return deterministic gap representatives for a discrete hard scan."""
    values = np.asarray(unique_values, dtype=np.float64).reshape(-1)
    if values.size == 0 or np.any(np.diff(values) <= 0.0):
        raise ValueError("unique_values must be nonempty and strictly increasing")
    if values.size == 1:
        scale = max(1.0, abs(float(values[0])))
        return np.asarray([values[0] - scale, values[0] + scale])
    boundaries = np.empty(values.size + 1, dtype=np.float64)
    boundaries[1:-1] = values[:-1] + 0.5 * (values[1:] - values[:-1])
    left_gap = max(values[1] - values[0], np.finfo(np.float64).eps * max(1.0, abs(values[0])))
    right_gap = max(values[-1] - values[-2], np.finfo(np.float64).eps * max(1.0, abs(values[-1])))
    boundaries[0] = values[0] - 0.5 * left_gap
    boundaries[-1] = values[-1] + 0.5 * right_gap
    return boundaries


def frozen_window_reachability_bound(
    values: np.ndarray,
    target_indicator: np.ndarray,
    weights: np.ndarray,
    bounds: CenterWidthBounds,
    eps_ratio: float,
    *,
    feasibility_tolerance: float = 1.0e-13,
) -> FrozenWindowReachabilityBound:
    """Bound frozen target coverage over the complete center--width box.

    Every admissible hard interval lies in
    ``[m_min-d_max/2, m_max+d_max/2]``.  Outside that envelope, the logistic
    tails are bounded using the largest possible smoothing length
    ``eps_ratio*d_max``.  The resulting smooth bound is deliberately
    conservative: it can permit a later scan, but it cannot reject a control
    box containing a window with the requested frozen target coverage.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    target = np.asarray(target_indicator, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size == 0 or values.shape != target.shape or values.shape != weights.shape:
        raise ValueError("sample arrays must be nonempty and have equal shapes")
    if not (
        np.all(np.isfinite(values))
        and np.all(np.isfinite(target))
        and np.all(np.isfinite(weights))
    ):
        raise ValueError("sample values, indicators, and weights must be finite")
    if np.any(weights < 0.0) or np.any((target < 0.0) | (target > 1.0)):
        raise ValueError("weights must be nonnegative and indicators must lie in [0,1]")
    eps_ratio = float(eps_ratio)
    if not math.isfinite(eps_ratio) or eps_ratio <= 0.0:
        raise ValueError("eps_ratio must be finite and positive")

    target_weights = weights * target
    target_area = float(np.sum(target_weights))
    if target_area <= 0.0:
        raise ValueError("target quadrature area must be positive")
    positive_target = target_weights > feasibility_tolerance
    if not np.any(positive_target):
        raise ValueError("target quadrature weights must contain a positive sample")

    lower_edge_min = float(bounds.center_min - 0.5 * bounds.width_max)
    upper_edge_max = float(bounds.center_max + 0.5 * bounds.width_max)
    epsilon_max = float(eps_ratio * bounds.width_max)
    inside = (values >= lower_edge_min - feasibility_tolerance) & (
        values <= upper_edge_max + feasibility_tolerance
    )
    hard_coverage_upper_bound = float(np.sum(target_weights[inside]) / target_area)

    activity_upper_bound = np.ones_like(values)
    below = values < lower_edge_min
    above = values > upper_edge_max
    if np.any(below):
        activity_upper_bound[below] = _logistic_and_q(
            (values[below] - lower_edge_min) / epsilon_max
        )[0]
    if np.any(above):
        activity_upper_bound[above] = _logistic_and_q(
            (upper_edge_max - values[above]) / epsilon_max
        )[0]
    smooth_coverage_upper_bound = float(
        np.dot(target_weights, activity_upper_bound) / target_area
    )
    target_values = values[positive_target]
    return FrozenWindowReachabilityBound(
        lower_edge_min=lower_edge_min,
        upper_edge_max=upper_edge_max,
        epsilon_max=epsilon_max,
        target_potential_min=float(np.min(target_values)),
        target_potential_max=float(np.max(target_values)),
        hard_coverage_upper_bound=hard_coverage_upper_bound,
        smooth_coverage_upper_bound=smooth_coverage_upper_bound,
    )


def sorted_hard_window_scan(
    values: np.ndarray,
    target_indicator: np.ndarray,
    weights: np.ndarray,
    bounds: CenterWidthBounds,
    leakage_budget: float,
    *,
    feasibility_tolerance: float = 1.0e-13,
) -> HardWindowSelection:
    """Find the best discrete hard interval in ``O(N log N)`` time.

    Samples with identical potential are first coalesced.  Thresholds are the
    representatives of gaps between consecutive unique potential values, so
    the scan is exact for the resulting discrete quadrature classification.
    For every upper gap, cumulative outside weight identifies the earliest
    admissible lower gap.  Nonnegative target weights imply that this earliest
    box-feasible lower gap maximizes target coverage for that upper gap.

    The deterministic tie order is: most captured target weight, least outside
    weight, narrowest interval, then smallest center.
    """
    if leakage_budget < 0.0 or not math.isfinite(float(leakage_budget)):
        raise ValueError("leakage_budget must be finite and nonnegative")
    unique_values, target_weights, outside_weights = _coalesced_sorted_samples(
        values, target_indicator, weights
    )
    target_area = float(np.sum(target_weights))
    if target_area <= 0.0:
        raise ValueError("target quadrature area must be positive")
    allowed_outside = float(leakage_budget) * target_area
    boundaries = hard_window_boundaries(unique_values)
    target_prefix = np.concatenate(([0.0], np.cumsum(target_weights)))
    outside_prefix = np.concatenate(([0.0], np.cumsum(outside_weights)))
    next_positive_target = np.empty(unique_values.size + 1, dtype=np.int64)
    next_index = unique_values.size
    next_positive_target[-1] = next_index
    for index in range(unique_values.size - 1, -1, -1):
        if target_weights[index] > feasibility_tolerance:
            next_index = index
        next_positive_target[index] = next_index
    best: HardWindowSelection | None = None
    best_key: tuple[float, float, float, float] | None = None

    # r is a boundary index and [l:r] is the set of included unique values.
    for r in range(1, unique_values.size + 1):
        cap_target = outside_prefix[r] - allowed_outside - feasibility_tolerance
        leakage_l = int(np.searchsorted(outside_prefix[: r + 1], cap_target, side="left"))
        c2 = float(boundaries[r])
        numeric_lower = max(c2 - bounds.width_max, 2.0 * bounds.center_min - c2)
        l = max(
            leakage_l,
            int(np.searchsorted(boundaries[:r], numeric_lower - feasibility_tolerance, side="left")),
        )
        if l >= r:
            continue
        # Moving the lower boundary across outside-only samples preserves the
        # primary captured-target score and improves the leakage tie-break.
        numeric_upper = min(c2 - bounds.width_min, 2.0 * bounds.center_max - c2)
        upper_l = int(
            np.searchsorted(
                boundaries[:r], numeric_upper + feasibility_tolerance, side="right"
            )
            - 1
        )
        if upper_l < l:
            continue
        l = min(upper_l, int(next_positive_target[l]), r - 1)
        c1 = float(boundaries[l])
        center, width = center_width_from_thresholds(c1, c2)
        if not bounds.contains(center, width, tolerance=feasibility_tolerance):
            continue
        outside = float(outside_prefix[r] - outside_prefix[l])
        if outside > allowed_outside + feasibility_tolerance:
            continue
        captured = float(target_prefix[r] - target_prefix[l])
        leakage = outside / target_area
        missing = 1.0 - captured / target_area
        key = (-captured, outside, width, center)
        if best_key is None or key < best_key:
            best_key = key
            best = HardWindowSelection(
                center=center,
                width=width,
                c1=c1,
                c2=c2,
                leakage=leakage,
                missing=missing,
                target_coverage=1.0 - missing,
                target_weight=captured,
                outside_weight=outside,
                left_index=l,
                right_index=r,
            )
    if best is None:
        raise ValueError("no hard-window interval satisfies the leakage budget and box bounds")
    return best


def hard_window_brute_force(
    values: np.ndarray,
    target_indicator: np.ndarray,
    weights: np.ndarray,
    bounds: CenterWidthBounds,
    leakage_budget: float,
    *,
    feasibility_tolerance: float = 1.0e-13,
) -> HardWindowSelection:
    """Reference quadratic search over the same hard-window gap candidates."""
    unique_values, target_weights, outside_weights = _coalesced_sorted_samples(
        values, target_indicator, weights
    )
    target_area = float(np.sum(target_weights))
    if target_area <= 0.0:
        raise ValueError("target quadrature area must be positive")
    boundaries = hard_window_boundaries(unique_values)
    target_prefix = np.concatenate(([0.0], np.cumsum(target_weights)))
    outside_prefix = np.concatenate(([0.0], np.cumsum(outside_weights)))
    allowed_outside = float(leakage_budget) * target_area
    best: HardWindowSelection | None = None
    best_key: tuple[float, float, float, float] | None = None
    for l, r in itertools.combinations(range(boundaries.size), 2):
        center, width = center_width_from_thresholds(boundaries[l], boundaries[r])
        if not bounds.contains(center, width, tolerance=feasibility_tolerance):
            continue
        outside = float(outside_prefix[r] - outside_prefix[l])
        if outside > allowed_outside + feasibility_tolerance:
            continue
        captured = float(target_prefix[r] - target_prefix[l])
        key = (-captured, outside, width, center)
        if best_key is None or key < best_key:
            best_key = key
            best = HardWindowSelection(
                center=float(center),
                width=float(width),
                c1=float(boundaries[l]),
                c2=float(boundaries[r]),
                leakage=outside / target_area,
                missing=1.0 - captured / target_area,
                target_coverage=captured / target_area,
                target_weight=captured,
                outside_weight=outside,
                left_index=l,
                right_index=r,
            )
    if best is None:
        raise ValueError("no hard-window interval satisfies the leakage budget and box bounds")
    return best


def initialization_shortlist(
    center: float,
    width: float,
    bounds: CenterWidthBounds,
    *,
    count: int = 3,
    width_fraction: float = 0.08,
    center_fraction: float = 0.03,
) -> list[tuple[str, float, float]]:
    """Build a deterministic three-to-five member initialization shortlist."""
    if count < 1 or count > 5:
        raise ValueError("shortlist count must lie between one and five")
    center, width = bounds.clip(center, width)
    raw: list[tuple[str, float, float]] = [("refined", center, width)]
    raw.extend(
        [
            ("narrower", center, width * (1.0 - width_fraction)),
            ("wider", center, width * (1.0 + width_fraction)),
        ]
    )
    center_delta = center_fraction * (bounds.center_max - bounds.center_min)
    raw.extend(
        [
            ("center_minus", center - center_delta, width),
            ("center_plus", center + center_delta, width),
        ]
    )
    result: list[tuple[str, float, float]] = []
    seen: set[tuple[float, float]] = set()
    for name, candidate_center, candidate_width in raw:
        clipped = bounds.clip(candidate_center, candidate_width)
        key = (round(clipped[0], 15), round(clipped[1], 15))
        if key in seen:
            continue
        seen.add(key)
        result.append((name, *clipped))
        if len(result) == count:
            break
    return result


def linear_step_constraints(
    point: np.ndarray,
    bounds: CenterWidthBounds,
    trust_radius: float,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Return box and scaled-infinity trust constraints ``A p <= b``."""
    point = np.asarray(point, dtype=np.float64).reshape(2)
    radius = float(trust_radius)
    if radius <= 0.0:
        raise ValueError("trust_radius must be positive")
    lower = bounds.lower
    upper = bounds.upper
    scale = bounds.scale
    rows = np.asarray(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [1.0 / scale[0], 0.0],
            [-1.0 / scale[0], 0.0],
            [0.0, 1.0 / scale[1]],
            [0.0, -1.0 / scale[1]],
        ],
        dtype=np.float64,
    )
    rhs = np.asarray(
        [
            upper[0] - point[0],
            point[0] - lower[0],
            upper[1] - point[1],
            point[1] - lower[1],
            radius,
            radius,
            radius,
            radius,
        ],
        dtype=np.float64,
    )
    names = (
        "center_upper",
        "center_lower",
        "width_upper",
        "width_lower",
        "trust_center_upper",
        "trust_center_lower",
        "trust_width_upper",
        "trust_width_lower",
    )
    return rows, rhs, names


def threshold_edge_step_constraints(
    width: float,
    maximum_fraction: float,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Limit one update by the motion of the two threshold edges.

    A trust radius scaled by the full center/width box can be very large
    compared with a narrow current band.  Since

    ``dc1 = dm - dd/2`` and ``dc2 = dm + dd/2``,

    these four inequalities impose ``|dc1|, |dc2| <= fraction * d``.  The
    constraint is local to one SQP micro-step; repeated accepted steps can
    still move the band by an arbitrary total distance while retaining a
    meaningful density-continuation check between corrections.
    """
    width = float(width)
    maximum_fraction = float(maximum_fraction)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("width must be finite and positive")
    if not math.isfinite(maximum_fraction) or maximum_fraction <= 0.0:
        raise ValueError("maximum_fraction must be finite and positive")
    maximum_motion = maximum_fraction * width
    rows = np.asarray(
        [
            [1.0, -0.5],
            [-1.0, 0.5],
            [1.0, 0.5],
            [-1.0, -0.5],
        ],
        dtype=np.float64,
    )
    rhs = np.full(4, maximum_motion, dtype=np.float64)
    names = (
        "lower_edge_forward",
        "lower_edge_backward",
        "upper_edge_forward",
        "upper_edge_backward",
    )
    return rows, rhs, names


def _is_feasible(step: np.ndarray, rows: np.ndarray, rhs: np.ndarray, tolerance: float) -> bool:
    """Check a two-dimensional linear inequality system."""
    return bool(np.all(rows @ step <= rhs + tolerance))


def _unique_candidates(candidates: Iterable[np.ndarray], tolerance: float = 1.0e-11) -> list[np.ndarray]:
    """Remove duplicate two-dimensional candidates deterministically."""
    unique: list[np.ndarray] = []
    for candidate in candidates:
        value = np.asarray(candidate, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(value)):
            continue
        if not any(np.linalg.norm(value - existing, ord=np.inf) <= tolerance for existing in unique):
            unique.append(value)
    return unique


def polygon_candidates(
    rows: np.ndarray,
    rhs: np.ndarray,
    *,
    extra_lines: Sequence[tuple[np.ndarray, float]] = (),
    tolerance: float = 1.0e-10,
) -> list[np.ndarray]:
    """Enumerate vertices after subdividing a feasible polygon by lines."""
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, 2)
    rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
    lines: list[tuple[np.ndarray, float]] = [
        (rows[index], float(rhs[index])) for index in range(rows.shape[0])
    ]
    lines.extend((np.asarray(a, dtype=np.float64).reshape(2), float(b)) for a, b in extra_lines)
    candidates: list[np.ndarray] = []
    origin = np.zeros(2, dtype=np.float64)
    if _is_feasible(origin, rows, rhs, tolerance):
        candidates.append(origin)
    for (a1, b1), (a2, b2) in itertools.combinations(lines, 2):
        matrix = np.vstack((a1, a2))
        determinant = float(np.linalg.det(matrix))
        if abs(determinant) <= 1.0e-14 * max(1.0, np.linalg.norm(matrix) ** 2):
            continue
        point = np.linalg.solve(matrix, np.asarray([b1, b2]))
        if _is_feasible(point, rows, rhs, tolerance):
            candidates.append(point)
    return _unique_candidates(candidates)


def solve_convex_qp_2d(
    gradient: np.ndarray,
    hessian: np.ndarray,
    rows: np.ndarray,
    rhs: np.ndarray,
    *,
    feasibility_tolerance: float = 1.0e-10,
) -> QPResult:
    """Solve a positive-definite 2D QP by complete active-set enumeration."""
    gradient = np.asarray(gradient, dtype=np.float64).reshape(2)
    hessian = np.asarray(hessian, dtype=np.float64).reshape(2, 2)
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, 2)
    rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
    if rows.shape[0] != rhs.size:
        raise ValueError("constraint row and rhs counts differ")
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues = np.linalg.eigvalsh(hessian)
    if eigenvalues[0] <= 0.0:
        return QPResult(False, np.zeros(2), math.inf, (), "HESSIAN_NOT_POSITIVE_DEFINITE")
    candidates: list[np.ndarray] = []
    unconstrained = -np.linalg.solve(hessian, gradient)
    if _is_feasible(unconstrained, rows, rhs, feasibility_tolerance):
        candidates.append(unconstrained)
    zero = np.zeros(2, dtype=np.float64)
    if _is_feasible(zero, rows, rhs, feasibility_tolerance):
        candidates.append(zero)
    for index, (normal, boundary) in enumerate(zip(rows, rhs, strict=True)):
        kkt = np.block(
            [
                [hessian, normal[:, None]],
                [normal[None, :], np.zeros((1, 1))],
            ]
        )
        try:
            solution = np.linalg.solve(kkt, np.concatenate((-gradient, [boundary])))[:2]
        except np.linalg.LinAlgError:
            continue
        if _is_feasible(solution, rows, rhs, feasibility_tolerance):
            candidates.append(solution)
    candidates.extend(polygon_candidates(rows, rhs, tolerance=feasibility_tolerance))
    candidates = _unique_candidates(candidates)
    if not candidates:
        return QPResult(False, np.zeros(2), math.inf, (), "INFEASIBLE_LINEARIZED_SUBPROBLEM")

    def objective(step: np.ndarray) -> float:
        """Evaluate the local quadratic model at one candidate step."""
        return float(gradient @ step + 0.5 * step @ hessian @ step)

    best = min(candidates, key=lambda step: (objective(step), np.linalg.norm(step), step[0], step[1]))
    active = tuple(
        int(index)
        for index, value in enumerate(rows @ best - rhs)
        if abs(float(value)) <= 10.0 * feasibility_tolerance
    )
    return QPResult(True, best, objective(best), active, "OK")


def normalized_violation(signed_constraints: np.ndarray) -> float:
    """Return the sum of positive signed normalized constraint violations."""
    values = np.asarray(signed_constraints, dtype=np.float64)
    return float(np.sum(np.maximum(values, 0.0)))


def threshold_functional_progress(
    *,
    old_objective: float,
    new_objective: float,
    old_leakage: float,
    new_leakage: float,
    old_missing: float,
    new_missing: float,
    leakage_max: float,
    missing_max: float,
    feasibility_tolerance: float,
    objective_scale: float = 1.0,
    absolute_tolerance: float = 1.0e-6,
    relative_tolerance: float = 1.0e-2,
) -> ThresholdFunctionalProgress:
    """Classify meaningful progress of a homotopy threshold correction.

    Exact satisfaction of the geometric safeguards need not be attainable at
    an intermediate source-homotopy value.  Requiring it can therefore drive
    an otherwise useful branch seed through arbitrarily small threshold
    updates.  This routine instead detects stagnation of the relevant scalar
    merit: normalized geometric violation while infeasible, and the scaled
    projection objective after feasibility has been reached.
    """
    values = np.asarray(
        [
            old_objective,
            new_objective,
            old_leakage,
            new_leakage,
            old_missing,
            new_missing,
            leakage_max,
            missing_max,
            feasibility_tolerance,
            objective_scale,
            absolute_tolerance,
            relative_tolerance,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("threshold functional values and tolerances must be finite")
    if leakage_max < 0.0 or missing_max < 0.0:
        raise ValueError("geometric bounds must be nonnegative")
    if feasibility_tolerance < 0.0:
        raise ValueError("feasibility_tolerance must be nonnegative")
    if objective_scale <= 0.0:
        raise ValueError("objective_scale must be positive")
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("stagnation tolerances must be nonnegative")
    if absolute_tolerance == 0.0 and relative_tolerance == 0.0:
        raise ValueError("at least one stagnation tolerance must be positive")

    old_violation = normalized_violation(
        np.asarray(
            [old_leakage - leakage_max, old_missing - missing_max],
            dtype=np.float64,
        )
    )
    new_violation = normalized_violation(
        np.asarray(
            [new_leakage - leakage_max, new_missing - missing_max],
            dtype=np.float64,
        )
    )
    if old_violation > feasibility_tolerance:
        mode = "geometric_violation"
        old_merit = old_violation
        new_merit = new_violation
    else:
        mode = "scaled_objective"
        old_merit = objective_scale * old_objective
        new_merit = objective_scale * new_objective

    improvement = old_merit - new_merit
    required = absolute_tolerance + relative_tolerance * max(
        abs(old_merit), abs(new_merit), np.finfo(np.float64).tiny
    )
    return ThresholdFunctionalProgress(
        mode=mode,
        old_merit=float(old_merit),
        new_merit=float(new_merit),
        improvement=float(improvement),
        required_improvement=float(required),
        stagnated=bool(improvement <= required),
    )


def frozen_topology_repair_rank_key(
    metrics: FrozenWindowMetrics,
    *,
    leakage_max: float,
    missing_max: float,
    edge_motion: float,
) -> tuple[float, float, float, float, float]:
    """Rank usable topology-repair candidates by geometric viability.

    Topology and minimum coverage are hard filters owned by the caller.  Once
    those tests pass, the best continuation seed is the candidate closest to
    the final geometric feasible set, not necessarily the candidate closest
    to the topologically incompatible frozen optimum.  Higher target coverage
    and smaller threshold-edge motion provide deterministic tie breaks.
    """
    values = np.asarray(
        [metrics.leakage, metrics.missing, leakage_max, missing_max, edge_motion],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("topology-repair metrics and bounds must be finite")
    if leakage_max < 0.0 or missing_max < 0.0 or edge_motion < 0.0:
        raise ValueError("topology-repair bounds and edge motion must be nonnegative")
    violation = normalized_violation(
        np.asarray(
            [metrics.leakage - leakage_max, metrics.missing - missing_max],
            dtype=np.float64,
        )
    )
    return (
        violation,
        -float(metrics.coverage),
        float(edge_motion),
        float(metrics.leakage),
        float(metrics.missing),
    )


def frozen_topology_continuation_rank_key(
    metrics: FrozenWindowMetrics,
    topology: ActivityTopology,
    *,
    expected_components: int,
    leakage_max: float,
    missing_max: float,
    edge_motion: float,
) -> tuple[int, int, float, int, float, float, float, float]:
    """Rank frozen windows for robust source-continuation startup.

    A topology-repair point and a final geometrically admissible point have
    different jobs.  The former must provide a clean, noncollapsed branch on
    which source continuation can start; it may temporarily have substantial
    leakage.  Consequently this key prioritizes the configured significant
    component count, absence of additional raw fragments, small unexpected
    core mass, and target coverage before edge motion and geometric violation.

    This key is intentionally separate from
    :func:`frozen_topology_repair_rank_key`.  Callers can retain the latter as
    their primary geometric choice and use this one only to diversify a
    continuation fallback.
    """
    expected_components = int(expected_components)
    if expected_components < 1:
        raise ValueError("expected_components must be positive")
    values = np.asarray(
        [metrics.leakage, metrics.missing, leakage_max, missing_max, edge_motion],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("continuation-anchor metrics and bounds must be finite")
    if leakage_max < 0.0 or missing_max < 0.0 or edge_motion < 0.0:
        raise ValueError(
            "continuation-anchor bounds and edge motion must be nonnegative"
        )

    fractions = np.asarray(topology.component_fractions, dtype=np.float64)
    if fractions.ndim != 1 or not np.all(np.isfinite(fractions)):
        raise ValueError("topology component fractions must be finite")
    if np.any(fractions < 0.0):
        raise ValueError("topology component fractions must be nonnegative")
    component_mismatch = abs(int(topology.component_count) - expected_components)
    raw_excess = max(0, int(topology.raw_component_count) - expected_components)
    unexpected_core_fraction = float(np.sum(fractions[expected_components:]))
    weakest_expected_fraction = (
        float(fractions[expected_components - 1])
        if fractions.size >= expected_components
        else 0.0
    )
    violation = normalized_violation(
        np.asarray(
            [metrics.leakage - leakage_max, metrics.missing - missing_max],
            dtype=np.float64,
        )
    )
    return (
        component_mismatch,
        int(raw_excess > 0),
        unexpected_core_fraction,
        raw_excess,
        -weakest_expected_fraction,
        -float(metrics.coverage),
        float(edge_motion),
        violation,
    )


def initialization_candidate_is_usable(
    *, strict_success: bool, missing: float, minimum_coverage: float
) -> bool:
    """Reject failed or collapsed equilibria before initializer ranking.

    A zero-activity state can be an accurately Newton-corrected equilibrium,
    but it is not an admissible local branch for this projection problem.  In
    particular, Newton success must never outrank the missing-area safeguard.
    """
    if not (0.0 < minimum_coverage <= 1.0):
        raise ValueError("minimum_coverage must lie in (0, 1]")
    if not strict_success or not math.isfinite(missing):
        return False
    return 1.0 - float(missing) >= float(minimum_coverage)


def solve_restoration_step_2d(
    objective_gradient: np.ndarray,
    objective_hessian: np.ndarray,
    geometric_values: np.ndarray,
    geometric_gradients: np.ndarray,
    rows: np.ndarray,
    rhs: np.ndarray,
    *,
    feasibility_tolerance: float = 1.0e-10,
) -> RestorationResult:
    """Minimize linearized geometric violation, breaking ties by the J model.

    The two hinge lines subdivide the bounded step polygon.  A piecewise-linear
    violation reaches its minimum at a vertex of that subdivision (or on a
    flat face containing one), so enumerating polygon/hinge intersections is
    complete in two dimensions.
    """
    gradient = np.asarray(objective_gradient, dtype=np.float64).reshape(2)
    hessian = np.asarray(objective_hessian, dtype=np.float64).reshape(2, 2)
    values = np.asarray(geometric_values, dtype=np.float64).reshape(2)
    geom_grad = np.asarray(geometric_gradients, dtype=np.float64).reshape(2, 2)
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, 2)
    rhs = np.asarray(rhs, dtype=np.float64).reshape(-1)
    extra = [(geom_grad[index], -values[index]) for index in range(2)]
    candidates = polygon_candidates(
        rows, rhs, extra_lines=extra, tolerance=feasibility_tolerance
    )
    if not candidates:
        return RestorationResult(False, np.zeros(2), math.inf, math.inf, "INFEASIBLE_STEP_POLYGON")

    def violation(step: np.ndarray) -> float:
        """Evaluate the sum of positive linearized geometric violations."""
        return normalized_violation(values + geom_grad @ step)

    def objective(step: np.ndarray) -> float:
        """Evaluate the objective tie-breaking model."""
        return float(gradient @ step + 0.5 * step @ hessian @ step)

    scored = [(violation(step), objective(step), np.linalg.norm(step), step) for step in candidates]
    minimum_violation = min(item[0] for item in scored)
    near = [
        item for item in scored
        if item[0] <= minimum_violation + 10.0 * feasibility_tolerance
    ]
    _, model, _, best = min(near, key=lambda item: (item[1], item[2], item[3][0], item[3][1]))
    return RestorationResult(True, best, violation(best), model, "OK")


def filter_accepts(
    objective: float,
    violation: float,
    filter_entries: Sequence[tuple[float, float]],
    *,
    objective_margin: float,
    violation_margin: float,
) -> bool:
    """Return whether a trial is not dominated by the current filter."""
    for entry_objective, entry_violation in filter_entries:
        enough_objective = objective <= entry_objective - objective_margin * violation
        enough_violation = violation <= (1.0 - violation_margin) * entry_violation
        if not (enough_objective or enough_violation):
            return False
    return True


def update_filter(
    filter_entries: Sequence[tuple[float, float]], objective: float, violation: float
) -> list[tuple[float, float]]:
    """Insert a point and discard entries weakly dominated by it."""
    retained = [
        (entry_objective, entry_violation)
        for entry_objective, entry_violation in filter_entries
        if not (objective <= entry_objective and violation <= entry_violation)
    ]
    retained.append((float(objective), float(violation)))
    return retained


def reduced_kkt_residual(
    objective_gradient: np.ndarray,
    signed_values: np.ndarray,
    constraint_gradients: np.ndarray,
    *,
    active_tolerance: float,
) -> KKTResult:
    """Compute the best nonnegative active-set multiplier fit in two dimensions.

    Constraints use the convention ``g_i(z) <= 0``.  All violated rows and
    all rows satisfying ``g_i >= -active_tolerance`` enter the active set.
    The nonnegative least-squares problem for stationarity is solved exactly by
    enumerating zero, one, and two positive multipliers; more than two positive
    independent multipliers are unnecessary in a two-dimensional cone.
    """
    gradient = np.asarray(objective_gradient, dtype=np.float64).reshape(2)
    values = np.asarray(signed_values, dtype=np.float64).reshape(-1)
    gradients = np.asarray(constraint_gradients, dtype=np.float64).reshape(-1, 2)
    if values.size != gradients.shape[0]:
        raise ValueError("constraint values and gradients have different lengths")
    active = tuple(int(i) for i, value in enumerate(values) if value >= -active_tolerance)
    candidates: list[np.ndarray] = [np.zeros(len(active), dtype=np.float64)]
    for local_index, row_index in enumerate(active):
        row = gradients[row_index]
        denominator = float(row @ row)
        if denominator <= 0.0:
            continue
        multiplier = max(0.0, -float(row @ gradient) / denominator)
        candidate = np.zeros(len(active), dtype=np.float64)
        candidate[local_index] = multiplier
        candidates.append(candidate)
    for first, second in itertools.combinations(range(len(active)), 2):
        matrix = np.column_stack((gradients[active[first]], gradients[active[second]]))
        try:
            pair = np.linalg.solve(matrix, -gradient)
        except np.linalg.LinAlgError:
            pair, *_ = np.linalg.lstsq(matrix, -gradient, rcond=None)
        if np.all(pair >= -1.0e-13):
            candidate = np.zeros(len(active), dtype=np.float64)
            candidate[first] = max(0.0, float(pair[0]))
            candidate[second] = max(0.0, float(pair[1]))
            candidates.append(candidate)

    def stationarity_for(candidate: np.ndarray) -> float:
        """Evaluate infinity-norm stationarity for active multipliers."""
        if not active:
            return float(np.linalg.norm(gradient, ord=np.inf))
        return float(
            np.linalg.norm(
                gradient + gradients[np.asarray(active, dtype=int)].T @ candidate,
                ord=np.inf,
            )
        )

    best_local = min(candidates, key=lambda candidate: (stationarity_for(candidate), np.linalg.norm(candidate)))
    multipliers = np.zeros(values.size, dtype=np.float64)
    if active:
        multipliers[np.asarray(active, dtype=int)] = best_local
    stationarity = stationarity_for(best_local)
    primal = float(np.max(np.maximum(values, 0.0), initial=0.0))
    complementarity = float(np.max(np.abs(multipliers * values), initial=0.0))
    residual = max(stationarity, primal, complementarity)
    return KKTResult(residual, stationarity, primal, complementarity, multipliers, active)


def soft_overlap_diagnostics(leakage: float, missing: float) -> dict[str, float]:
    """Return area ratio, overlap, recall, precision, and Jaccard diagnostics.

    Values are normalized by target area.  Hence the overlap area ratio is
    ``1-M`` and total activity-area ratio is ``1-M+L``.
    """
    overlap = max(0.0, 1.0 - float(missing))
    activity = max(0.0, overlap + float(leakage))
    union = max(1.0 + float(leakage), np.finfo(np.float64).tiny)
    return {
        "activity_area_ratio": activity,
        "overlap_area_ratio": overlap,
        "recall": overlap,
        "precision": overlap / max(activity, np.finfo(np.float64).tiny),
        "jaccard": overlap / union,
    }
