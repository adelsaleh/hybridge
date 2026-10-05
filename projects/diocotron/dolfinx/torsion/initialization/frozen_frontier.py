"""Frozen hard-window Pareto initialization for torsion target potentials.

The finite-element runner supplies quadrature samples of the target potential
and of the crisp torsion band.  This module contains only NumPy algorithms:

* construction of the two pushforward measures carried by the target and its
  complement;
* complete Pareto filtering of all hard intervals on a fixed histogram; and
* exact sampled logistic-window metrics and threshold derivatives.

Keeping these operations independent of DOLFINx makes the scientific
selection rule directly testable.  The histogram is an explicitly recorded
numerical discretization.  The selected pair is subsequently refined on the
original quadrature samples by the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


LOGISTIC_CLIP = 50.0


@dataclass(frozen=True)
class PushforwardHistogram:
    """Weighted target/complement pushforwards on a common potential grid."""

    edges: np.ndarray
    target_mass: np.ndarray
    outside_mass: np.ndarray
    target_mass_total: float
    outside_mass_total: float
    target_mass_in_range: float
    outside_mass_in_range: float
    sample_count: int


@dataclass(frozen=True)
class FrozenFrontierPoint:
    """One nondominated hard interval in the binned pushforward measure."""

    lower_bin: int
    upper_bin: int
    c1: float
    c2: float
    leakage: float
    missing: float
    overlap: float
    jaccard: float

    @property
    def width(self) -> float:
        return float(self.c2 - self.c1)


@dataclass(frozen=True)
class FrozenFrontier:
    """Complete nondominated hard-window frontier for one histogram."""

    points: tuple[FrozenFrontierPoint, ...]
    evaluated_intervals: int
    numerical_tolerance: float
    target_area: float


@dataclass(frozen=True)
class LogisticWindowMetrics:
    """Sampled smooth geometry and its fixed-state threshold derivatives."""

    leakage: float
    missing: float
    overlap: float
    activity_area: float
    jaccard: float
    grad_leakage: np.ndarray
    grad_missing: np.ndarray
    grad_overlap: np.ndarray
    grad_jaccard: np.ndarray


def mass_roundoff_tolerance(*values: float) -> float:
    """Return a scale-aware roundoff allowance, not a scientific tolerance."""

    scale = max((abs(float(value)) for value in values), default=1.0)
    return 256.0 * np.finfo(np.float64).eps * max(scale, 1.0)


def weighted_pushforward_histogram(
    phi_values: np.ndarray,
    target_values: np.ndarray,
    weights: np.ndarray,
    *,
    lower: float,
    upper: float,
    bins: int,
) -> PushforwardHistogram:
    """Accumulate target and complement mass by target-potential value.

    Values outside ``[lower, upper]`` remain part of the reported total mass
    but cannot be selected by an admissible hard interval on this histogram.
    ``target_values`` may be Boolean or a quadrature representation in
    ``[0,1]``; the runner passes the crisp torsion-band indicator.
    """

    bins = int(bins)
    if bins < 2:
        raise ValueError("pushforward histogram requires at least two bins")
    lower = float(lower)
    upper = float(upper)
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise ValueError("pushforward histogram requires finite lower < upper")

    phi = np.asarray(phi_values, dtype=np.float64).ravel()
    target = np.asarray(target_values, dtype=np.float64).ravel()
    weight = np.asarray(weights, dtype=np.float64).ravel()
    if not (phi.size == target.size == weight.size):
        raise ValueError("phi, target, and weight arrays must have equal size")
    valid = (
        np.isfinite(phi)
        & np.isfinite(target)
        & np.isfinite(weight)
        & (weight >= 0.0)
    )
    phi = phi[valid]
    target = np.clip(target[valid], 0.0, 1.0)
    weight = weight[valid]
    target_weight = weight * target
    outside_weight = weight * (1.0 - target)
    edges = np.linspace(lower, upper, bins + 1, dtype=np.float64)
    target_hist, _ = np.histogram(phi, bins=edges, weights=target_weight)
    outside_hist, _ = np.histogram(phi, bins=edges, weights=outside_weight)
    return PushforwardHistogram(
        edges=edges,
        target_mass=np.asarray(target_hist, dtype=np.float64),
        outside_mass=np.asarray(outside_hist, dtype=np.float64),
        target_mass_total=float(np.sum(target_weight)),
        outside_mass_total=float(np.sum(outside_weight)),
        target_mass_in_range=float(np.sum(target_hist)),
        outside_mass_in_range=float(np.sum(outside_hist)),
        sample_count=int(phi.size),
    )


def _pareto_prune(
    lower_bin: np.ndarray,
    upper_bin: np.ndarray,
    leakage: np.ndarray,
    missing: np.ndarray,
    edges: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove dominated interval records with deterministic tie breaking."""

    if leakage.size == 0:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float64)
        return empty_i, empty_i.copy(), empty_f, empty_f.copy()
    widths = edges[upper_bin] - edges[lower_bin]
    c1 = edges[lower_bin]
    # Primary order is leakage, then missing.  Equal metric pairs retain the
    # narrowest interval and finally the lower c1.
    order = np.lexsort((c1, widths, missing, leakage))
    lower_bin = lower_bin[order]
    upper_bin = upper_bin[order]
    leakage = leakage[order]
    missing = missing[order]
    running_best = np.minimum.accumulate(missing)
    keep = np.ones(missing.size, dtype=bool)
    if missing.size > 1:
        keep[1:] = missing[1:] < running_best[:-1] - tolerance
    return lower_bin[keep], upper_bin[keep], leakage[keep], missing[keep]


def hard_window_pareto_frontier(
    histogram: PushforwardHistogram,
    *,
    target_area: float | None = None,
    min_width: float = 0.0,
) -> FrozenFrontier:
    """Enumerate every nondominated hard interval on ``histogram``.

    Enumeration is quadratic in the number of bins but vectorized by lower
    endpoint and pruned after each row.  Memory therefore scales with the
    observed frontier rather than with all ``bins*(bins+1)/2`` intervals.
    """

    edges = np.asarray(histogram.edges, dtype=np.float64)
    target_mass = np.asarray(histogram.target_mass, dtype=np.float64)
    outside_mass = np.asarray(histogram.outside_mass, dtype=np.float64)
    if edges.ndim != 1 or target_mass.ndim != 1 or outside_mass.ndim != 1:
        raise ValueError("pushforward histogram arrays must be one-dimensional")
    if edges.size != target_mass.size + 1 or target_mass.size != outside_mass.size:
        raise ValueError("inconsistent pushforward histogram dimensions")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("pushforward edges must be strictly increasing")
    min_width = float(min_width)
    if not math.isfinite(min_width) or min_width < 0.0:
        raise ValueError("minimum hard-window width must be finite and nonnegative")
    area = (
        float(histogram.target_mass_total)
        if target_area is None
        else float(target_area)
    )
    if not math.isfinite(area) or area <= 0.0:
        raise ValueError("hard-window frontier requires positive target area")
    tolerance = mass_roundoff_tolerance(
        area,
        histogram.target_mass_total,
        histogram.outside_mass_total,
    )
    target_prefix = np.concatenate(([0.0], np.cumsum(target_mass)))
    outside_prefix = np.concatenate(([0.0], np.cumsum(outside_mass)))
    bin_count = target_mass.size

    frontier_lower = np.empty(0, dtype=np.int64)
    frontier_upper = np.empty(0, dtype=np.int64)
    frontier_leakage = np.empty(0, dtype=np.float64)
    frontier_missing = np.empty(0, dtype=np.float64)
    evaluated = 0
    for lower_bin in range(bin_count):
        first_upper = int(
            np.searchsorted(
                edges,
                edges[lower_bin] + min_width,
                side="left",
            )
        )
        first_upper = max(first_upper, lower_bin + 1)
        if first_upper > bin_count:
            continue
        upper_bins = np.arange(first_upper, bin_count + 1, dtype=np.int64)
        evaluated += int(upper_bins.size)
        overlaps = target_prefix[upper_bins] - target_prefix[lower_bin]
        row_leakage = outside_prefix[upper_bins] - outside_prefix[lower_bin]
        row_missing = area - overlaps
        nontrivial = overlaps > tolerance
        if not np.any(nontrivial):
            continue
        row_lower = np.full(int(np.count_nonzero(nontrivial)), lower_bin, dtype=np.int64)
        row_upper = upper_bins[nontrivial]
        row_leakage = row_leakage[nontrivial]
        row_missing = row_missing[nontrivial]
        frontier_lower, frontier_upper, frontier_leakage, frontier_missing = _pareto_prune(
            np.concatenate((frontier_lower, row_lower)),
            np.concatenate((frontier_upper, row_upper)),
            np.concatenate((frontier_leakage, row_leakage)),
            np.concatenate((frontier_missing, row_missing)),
            edges,
            tolerance,
        )

    points: list[FrozenFrontierPoint] = []
    for lower_bin, upper_bin, leakage, missing in zip(
        frontier_lower,
        frontier_upper,
        frontier_leakage,
        frontier_missing,
        strict=True,
    ):
        overlap = area - float(missing)
        denominator = max(area + float(leakage), np.finfo(np.float64).tiny)
        points.append(
            FrozenFrontierPoint(
                lower_bin=int(lower_bin),
                upper_bin=int(upper_bin),
                c1=float(edges[lower_bin]),
                c2=float(edges[upper_bin]),
                leakage=float(leakage),
                missing=float(missing),
                overlap=float(overlap),
                jaccard=float(overlap / denominator),
            )
        )
    return FrozenFrontier(
        points=tuple(points),
        evaluated_intervals=evaluated,
        numerical_tolerance=tolerance,
        target_area=area,
    )


def select_frozen_frontier_point(
    frontier: FrozenFrontier,
    *,
    leakage_cap: float | None,
) -> FrozenFrontierPoint:
    """Select by a strict cap, or by Jaccard when no cap is supplied."""

    if not frontier.points:
        raise ValueError("frozen hard-window frontier has no nontrivial point")
    if leakage_cap is not None:
        cap = float(leakage_cap)
        if not math.isfinite(cap) or cap < 0.0:
            raise ValueError("frozen leakage cap must be finite and nonnegative")
        eligible = [point for point in frontier.points if point.leakage <= cap]
        if not eligible:
            minimum = min(point.leakage for point in frontier.points)
            raise ValueError(
                "no nontrivial frozen hard-window point satisfies leakage cap "
                f"{cap:.12e}; minimum nontrivial leakage is {minimum:.12e} "
                f"(relative {minimum / frontier.target_area:.12e})"
            )
        return min(
            eligible,
            key=lambda point: (
                point.missing,
                point.leakage,
                point.width,
                point.c1,
                point.c2,
            ),
        )
    return min(
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


def _clipped_logistic(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the exactly clipped logistic and ``sigma*(1-sigma)``."""

    z = np.asarray(z, dtype=np.float64)
    sigma = np.empty_like(z)
    sigma[z > LOGISTIC_CLIP] = 1.0
    sigma[z < -LOGISTIC_CLIP] = 0.0
    transition = (z >= -LOGISTIC_CLIP) & (z <= LOGISTIC_CLIP)
    sigma[transition] = 1.0 / (1.0 + np.exp(-z[transition]))
    return sigma, sigma * (1.0 - sigma)


def logistic_window_and_threshold_derivatives(
    values: np.ndarray,
    *,
    c1: float,
    c2: float,
    eps_mode: str,
    eps_ratio: float,
    eps_fixed: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Evaluate ``W``, total ``dW/dc1,dW/dc2``, and its smoothing width."""

    values = np.asarray(values, dtype=np.float64)
    c1 = float(c1)
    c2 = float(c2)
    if not math.isfinite(c1) or not math.isfinite(c2) or c2 <= c1:
        raise ValueError("logistic thresholds require finite c1 < c2")
    if eps_mode == "relative":
        eps = float(eps_ratio) * (c2 - c1)
        deps_dc1 = -float(eps_ratio)
        deps_dc2 = float(eps_ratio)
    elif eps_mode == "fixed":
        if eps_fixed is None:
            raise ValueError("fixed logistic smoothing requires eps_fixed")
        eps = float(eps_fixed)
        deps_dc1 = 0.0
        deps_dc2 = 0.0
    else:
        raise ValueError("eps_mode must be 'relative' or 'fixed'")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("logistic smoothing width must be positive and finite")

    z1 = (values - c1) / eps
    z2 = (values - c2) / eps
    sigma1, q1 = _clipped_logistic(z1)
    sigma2, q2 = _clipped_logistic(z2)
    activity = sigma1 - sigma2
    derivative_eps = (
        -(values - c1) * q1 + (values - c2) * q2
    ) / (eps * eps)
    derivative_c1 = -q1 / eps + deps_dc1 * derivative_eps
    derivative_c2 = q2 / eps + deps_dc2 * derivative_eps
    return activity, derivative_c1, derivative_c2, eps


def sampled_logistic_window_metrics(
    phi_values: np.ndarray,
    target_values: np.ndarray,
    weights: np.ndarray,
    *,
    c1: float,
    c2: float,
    eps_mode: str,
    eps_ratio: float,
    eps_fixed: float | None,
    target_area: float | None = None,
) -> LogisticWindowMetrics:
    """Evaluate exact-sample frozen geometry and analytic threshold gradients."""

    phi = np.asarray(phi_values, dtype=np.float64).ravel()
    target = np.asarray(target_values, dtype=np.float64).ravel()
    weight = np.asarray(weights, dtype=np.float64).ravel()
    if not (phi.size == target.size == weight.size):
        raise ValueError("phi, target, and weight arrays must have equal size")
    target = np.clip(target, 0.0, 1.0)
    activity, derivative_c1, derivative_c2, _ = (
        logistic_window_and_threshold_derivatives(
            phi,
            c1=c1,
            c2=c2,
            eps_mode=eps_mode,
            eps_ratio=eps_ratio,
            eps_fixed=eps_fixed,
        )
    )
    inside_weight = weight * target
    outside_weight = weight * (1.0 - target)
    area = float(np.sum(inside_weight)) if target_area is None else float(target_area)
    overlap = float(np.dot(inside_weight, activity))
    leakage = float(np.dot(outside_weight, activity))
    activity_area = overlap + leakage
    missing = area - overlap
    gradient_overlap = np.array(
        [
            np.dot(inside_weight, derivative_c1),
            np.dot(inside_weight, derivative_c2),
        ],
        dtype=np.float64,
    )
    gradient_leakage = np.array(
        [
            np.dot(outside_weight, derivative_c1),
            np.dot(outside_weight, derivative_c2),
        ],
        dtype=np.float64,
    )
    gradient_missing = -gradient_overlap
    denominator = max(area + leakage, np.finfo(np.float64).tiny)
    jaccard = overlap / denominator
    gradient_jaccard = (
        denominator * gradient_overlap - overlap * gradient_leakage
    ) / (denominator * denominator)
    return LogisticWindowMetrics(
        leakage=leakage,
        missing=missing,
        overlap=overlap,
        activity_area=activity_area,
        jaccard=float(jaccard),
        grad_leakage=gradient_leakage,
        grad_missing=gradient_missing,
        grad_overlap=gradient_overlap,
        grad_jaccard=gradient_jaccard,
    )


__all__ = [
    "FrozenFrontier",
    "FrozenFrontierPoint",
    "LogisticWindowMetrics",
    "PushforwardHistogram",
    "hard_window_pareto_frontier",
    "logistic_window_and_threshold_derivatives",
    "mass_roundoff_tolerance",
    "sampled_logistic_window_metrics",
    "select_frozen_frontier_point",
    "weighted_pushforward_histogram",
]
