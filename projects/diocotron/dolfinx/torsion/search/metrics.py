"""Distributed geometric metrics for threshold-window candidate searches.

The reduced optimizer defines a candidate equilibrium band through two level
sets of the nonlinear potential, while the design band is defined through two
torsion level sets.  This module compares those objects without reconstructing
global plotting contours.  It samples only owned cells, applies the coarea
formula at quadrature points, and reduces fixed-size arrays on the candidate
subcommunicator.

All threshold-position calculations are normalized by ``T_max``.  The loose
parameter box therefore remains ``0 <= c1 < c2 <= T_max`` independently of
geometry and physical scaling.  The coarea bandwidth is tied to ``h_K / p``
and to the local finite-element gradient; it is deliberately independent of
the logistic smoothing used by the nonlinear PDE.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class CurveAlignmentMetrics:
    """Arc-length diagnostics for one candidate potential level curve."""

    level: float
    target_level: float
    length: float
    active: bool
    resolved: bool
    tau_q10: float
    tau_q50: float
    tau_q90: float
    signed_position_residual: float
    tau_spread: float
    containment_crisp: float
    containment_tolerant: float
    boundary_fit_score: float
    orientation_coverage: float
    inward_fraction: float
    mean_alignment_cosine: float
    length_ratio_to_target: float
    kernel_stability_error: float


@dataclass(frozen=True)
class HardBandMetrics:
    """Crisp candidate/design set overlap and mean-thickness diagnostics."""

    area: float
    target_area: float
    intersection: float
    leakage: float
    missing: float
    precision: float
    recall: float
    dice: float
    jaccard: float
    tau_q10: float
    tau_q90: float
    tau_span: float
    mean_physical_thickness: float
    target_mean_physical_thickness: float
    physical_thickness_ratio: float


@dataclass(frozen=True)
class TargetGeometryCalibration:
    """Mesh- and geometry-dependent tolerances computed from the target."""

    alpha_t1: float
    alpha_t2: float
    delta_alpha: float
    target_area: float
    domain_area: float
    lower_curve_length: float
    upper_curve_length: float
    lower_resolution: float
    upper_resolution: float
    lower_tolerance: float
    upper_tolerance: float
    physical_resolution: float
    mean_physical_thickness: float
    histogram_bins: int
    histogram_bin_width: float
    underresolved: bool


@dataclass(frozen=True)
class CandidateGeometryMetrics:
    """Complete geometric decision record for one threshold pair."""

    c1: float
    c2: float
    normalized_c1: float
    normalized_c2: float
    lower: CurveAlignmentMetrics
    upper: CurveAlignmentMetrics
    hard_band: HardBandMetrics
    pair_fit_score: float
    pair_containment_score: float
    lower_position_tolerance: float
    upper_position_tolerance: float
    lower_target_resolution: float
    upper_target_resolution: float
    target_underresolved: bool
    robust_tau_span: float
    robust_span_limit: float
    robust_too_thick: bool
    hard_span_too_thick: bool
    robust_span_warning: bool
    too_thick: bool
    too_thin: bool
    physical_too_thick: bool
    monotone_for_bisection: bool
    geometrically_eligible: bool
    rejection_reason: str
    warning_reason: str

    def as_dict(self) -> dict[str, float | int | str]:
        """Return a stable flat representation suitable for CSV/JSON logs."""

        lower = self.lower
        upper = self.upper
        band = self.hard_band
        return {
            "geometryC1": self.c1,
            "geometryC2": self.c2,
            "geometryC1Normalized": self.normalized_c1,
            "geometryC2Normalized": self.normalized_c2,
            "r1": lower.signed_position_residual,
            "r2": upper.signed_position_residual,
            "lowerCurveLength": lower.length,
            "upperCurveLength": upper.length,
            "lowerCurveActive": int(lower.active),
            "upperCurveActive": int(upper.active),
            "lowerCurveResolved": int(lower.resolved),
            "upperCurveResolved": int(upper.resolved),
            "lowerTauQ10": lower.tau_q10,
            "lowerTauQ50": lower.tau_q50,
            "lowerTauQ90": lower.tau_q90,
            "upperTauQ10": upper.tau_q10,
            "upperTauQ50": upper.tau_q50,
            "upperTauQ90": upper.tau_q90,
            "lowerTauSpread": lower.tau_spread,
            "upperTauSpread": upper.tau_spread,
            "lowerContainmentCrisp": lower.containment_crisp,
            "upperContainmentCrisp": upper.containment_crisp,
            "lowerContainment": lower.containment_tolerant,
            "upperContainment": upper.containment_tolerant,
            "lowerBoundaryFit": lower.boundary_fit_score,
            "upperBoundaryFit": upper.boundary_fit_score,
            "lowerInwardFraction": lower.inward_fraction,
            "upperInwardFraction": upper.inward_fraction,
            "lowerOrientationCoverage": lower.orientation_coverage,
            "upperOrientationCoverage": upper.orientation_coverage,
            "lowerAlignmentCosine": lower.mean_alignment_cosine,
            "upperAlignmentCosine": upper.mean_alignment_cosine,
            "lowerKernelStabilityError": lower.kernel_stability_error,
            "upperKernelStabilityError": upper.kernel_stability_error,
            "pairFitScore": self.pair_fit_score,
            "pairContainmentScore": self.pair_containment_score,
            "lowerPositionTolerance": self.lower_position_tolerance,
            "upperPositionTolerance": self.upper_position_tolerance,
            "lowerTargetResolution": self.lower_target_resolution,
            "upperTargetResolution": self.upper_target_resolution,
            "targetUnderresolved": int(self.target_underresolved),
            "robustTauSpan": self.robust_tau_span,
            "robustSpanLimit": self.robust_span_limit,
            "robustTooThick": int(self.robust_too_thick),
            "hardSpanTooThick": int(self.hard_span_too_thick),
            "robustSpanWarning": int(self.robust_span_warning),
            "tooThick": int(self.too_thick),
            "tooThin": int(self.too_thin),
            "physicalTooThick": int(self.physical_too_thick),
            "monotoneForBisection": int(self.monotone_for_bisection),
            "geometryEligible": int(self.geometrically_eligible),
            "geometryRejectionReason": self.rejection_reason,
            "geometryWarningReason": self.warning_reason,
            "hardBandArea": band.area,
            "hardIntersection": band.intersection,
            "hardLeakage": band.leakage,
            "hardMissing": band.missing,
            "hardPrecision": band.precision,
            "hardRecall": band.recall,
            "hardDice": band.dice,
            "hardJaccard": band.jaccard,
            "hardBandTauQ10": band.tau_q10,
            "hardBandTauQ90": band.tau_q90,
            "hardBandTauSpan": band.tau_span,
            "meanPhysicalThickness": band.mean_physical_thickness,
            "targetMeanPhysicalThickness": band.target_mean_physical_thickness,
            "physicalThicknessRatio": band.physical_thickness_ratio,
        }


def cosine_delta(values: np.ndarray, bandwidth: np.ndarray | float) -> np.ndarray:
    """Evaluate a normalized compact cosine approximation of a Dirac delta.

    ``delta_eta(z) = (1 + cos(pi*z/eta))/(2*eta)`` for ``|z| <= eta`` and
    zero otherwise.  Scalar and pointwise bandwidths are both supported.
    """

    values = np.asarray(values, dtype=np.float64)
    bandwidth = np.asarray(bandwidth, dtype=np.float64)
    if np.any(~np.isfinite(bandwidth)) or np.any(bandwidth <= 0.0):
        raise ValueError("cosine-delta bandwidths must be positive and finite")
    scaled = values / bandwidth
    result = np.zeros(np.broadcast_shapes(values.shape, bandwidth.shape), dtype=np.float64)
    values_b, bandwidth_b = np.broadcast_arrays(values, bandwidth)
    scaled = values_b / bandwidth_b
    active = np.isfinite(values_b) & (np.abs(scaled) <= 1.0)
    result[active] = (
        0.5 * (1.0 + np.cos(np.pi * scaled[active])) / bandwidth_b[active]
    )
    return result


def histogram_quantiles(
        histogram: np.ndarray,
        edges: np.ndarray,
        probabilities: Sequence[float],
) -> np.ndarray:
    """Approximate weighted quantiles from a globally reduced histogram."""

    histogram = np.asarray(histogram, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    requested = np.asarray(probabilities, dtype=np.float64)
    if histogram.ndim != 1 or edges.ndim != 1 or len(edges) != len(histogram) + 1:
        raise ValueError("histogram edges must have exactly one more entry than bins")
    if np.any(histogram < 0.0) or np.any(~np.isfinite(histogram)):
        raise ValueError("histogram weights must be nonnegative and finite")
    if np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0.0):
        raise ValueError("histogram edges must be finite and strictly increasing")
    if np.any(~np.isfinite(requested)) or np.any((requested < 0.0) | (requested > 1.0)):
        raise ValueError("quantile probabilities must lie in [0,1]")
    total = float(np.sum(histogram))
    result = np.full(requested.shape, math.nan, dtype=np.float64)
    if total <= 0.0:
        return result
    cumulative = np.cumsum(histogram)
    for output_index, probability in np.ndenumerate(requested):
        target = float(probability) * total
        if probability <= 0.0:
            nonzero = int(np.flatnonzero(histogram > 0.0)[0])
            result[output_index] = edges[nonzero]
            continue
        index = int(np.searchsorted(cumulative, target, side="left"))
        index = min(max(index, 0), len(histogram) - 1)
        previous = 0.0 if index == 0 else float(cumulative[index - 1])
        bin_weight = float(histogram[index])
        fraction = 0.0 if bin_weight <= 0.0 else (target - previous) / bin_weight
        fraction = min(max(fraction, 0.0), 1.0)
        result[output_index] = edges[index] + fraction * (edges[index + 1] - edges[index])
    return result


def geometry_tolerance(
        delta_alpha: float,
        resolution_quantile: float,
        *,
        histogram_bin_width: float = 0.0,
) -> float:
    """Return the target-curve position tolerance in normalized torsion units."""

    if not math.isfinite(delta_alpha) or delta_alpha <= 0.0:
        raise ValueError("target torsion-band width must be positive and finite")
    if not math.isfinite(resolution_quantile) or resolution_quantile < 0.0:
        raise ValueError("curve resolution must be nonnegative and finite")
    if not math.isfinite(histogram_bin_width) or histogram_bin_width < 0.0:
        raise ValueError("histogram-bin width must be nonnegative and finite")
    return max(0.05 * float(delta_alpha), float(resolution_quantile), 2.0 * float(histogram_bin_width))


def hard_overlap_from_measures(
        *,
        candidate_area: float,
        target_area: float,
        intersection: float,
        tau_q10: float = math.nan,
        tau_q90: float = math.nan,
        lower_curve_length: float,
        upper_curve_length: float,
        target_mean_physical_thickness: float,
) -> HardBandMetrics:
    """Construct hard-set overlap metrics from globally reduced measures."""

    candidate_area = max(float(candidate_area), 0.0)
    target_area = max(float(target_area), 0.0)
    intersection = min(max(float(intersection), 0.0), candidate_area, target_area)
    leakage = max(candidate_area - intersection, 0.0)
    missing = max(target_area - intersection, 0.0)
    precision = intersection / candidate_area if candidate_area > 0.0 else 0.0
    recall = intersection / target_area if target_area > 0.0 else 0.0
    dice_denominator = candidate_area + target_area
    dice = 2.0 * intersection / dice_denominator if dice_denominator > 0.0 else 1.0
    union = candidate_area + target_area - intersection
    jaccard = intersection / union if union > 0.0 else 1.0
    perimeter_sum = max(float(lower_curve_length), 0.0) + max(float(upper_curve_length), 0.0)
    mean_thickness = 2.0 * candidate_area / perimeter_sum if perimeter_sum > 0.0 else math.inf
    target_thickness = float(target_mean_physical_thickness)
    thickness_ratio = (
        mean_thickness / target_thickness
        if math.isfinite(mean_thickness) and target_thickness > 0.0
        else math.inf
    )
    tau_span = (
        max(float(tau_q90) - float(tau_q10), 0.0)
        if math.isfinite(tau_q10) and math.isfinite(tau_q90)
        else math.nan
    )
    return HardBandMetrics(
        area=candidate_area,
        target_area=target_area,
        intersection=intersection,
        leakage=leakage,
        missing=missing,
        precision=precision,
        recall=recall,
        dice=dice,
        jaccard=jaccard,
        tau_q10=float(tau_q10),
        tau_q90=float(tau_q90),
        tau_span=tau_span,
        mean_physical_thickness=mean_thickness,
        target_mean_physical_thickness=target_thickness,
        physical_thickness_ratio=thickness_ratio,
    )


def robust_thickness_classification(
        *,
        lower_tau_q10: float,
        upper_tau_q90: float,
        delta_alpha: float,
        lower_tolerance: float,
        upper_tolerance: float,
) -> tuple[float, float, bool, bool]:
    """Return robust span, allowed span, too-thick and too-thin flags."""

    if not math.isfinite(lower_tau_q10) or not math.isfinite(upper_tau_q90):
        return math.nan, float(delta_alpha + lower_tolerance + upper_tolerance), False, False
    span = float(upper_tau_q90) - float(lower_tau_q10)
    limit = float(delta_alpha) + float(lower_tolerance) + float(upper_tolerance)
    lower_limit = max(float(delta_alpha) - float(lower_tolerance) - float(upper_tolerance), 0.0)
    return span, limit, span > limit, span < lower_limit


def corroborated_thickness_classification(
        *,
        robust_too_thick: bool,
        hard_band_tau_span: float,
        robust_span_limit: float,
        physical_too_thick: bool,
) -> tuple[bool, bool, bool]:
    """Require an independent hard-set or physical thickness corroboration.

    Coarea quantiles can broaden when a level curve is only marginally
    resolved.  They remain a useful warning, but do not justify monotone
    row/column pruning unless either the hard-band torsion span or mean physical
    thickness independently confirms the excessive width.
    """

    hard_span_too_thick = bool(
        math.isfinite(float(hard_band_tau_span))
        and math.isfinite(float(robust_span_limit))
        and float(hard_band_tau_span) > float(robust_span_limit)
    )
    rejected = bool(
        robust_too_thick and (hard_span_too_thick or physical_too_thick)
    )
    warning = bool(robust_too_thick and not rejected)
    return rejected, hard_span_too_thick, warning


def _import_fem_dependencies() -> tuple[Any, Any, Any, Any]:
    """Import heavy finite-element dependencies only when a workspace is built."""

    try:
        import basix
        import ufl
        from dolfinx import fem
        from mpi4py import MPI
    except ImportError as error:  # pragma: no cover - exercised only without FEniCSx
        raise RuntimeError(
            "DistributedBandMetricWorkspace requires basix, dolfinx, mpi4py, and ufl"
        ) from error
    return basix, ufl, fem, MPI


class DistributedBandMetricWorkspace:
    """Reusable owned-cell quadrature workspace for one MPI candidate group.

    The supplied ``u`` object remains mutable.  ``evaluate`` reads its current
    coefficients, so expressions and geometry arrays are compiled only once
    while Newton candidates can be scored repeatedly.
    """

    _CURVE_SCALAR_COUNT = 8

    def __init__(
            self,
            *,
            u: Any,
            torsion: Any,
            tmax: float,
            alpha_t1: float,
            alpha_t2: float,
            quadrature_degree: int,
            order: int | None = None,
            comm: Any | None = None,
            histogram_bins: int | None = None,
            kernel_scale: float = 1.5,
            kernel_floor: float | None = None,
            minimum_fit_score: float = 0.90,
            minimum_containment_score: float = 0.95,
            minimum_inward_fraction: float = 0.90,
            maximum_kernel_stability_error: float = 0.25,
    ) -> None:
        basix, ufl, fem, MPI = _import_fem_dependencies()
        self._MPI = MPI
        self._fem = fem
        self.u = u
        self.torsion = torsion
        self.domain = u.function_space.mesh
        if torsion.function_space.mesh is not self.domain:
            raise ValueError("candidate potential and torsion must use the same mesh")
        self.comm = self.domain.comm if comm is None else comm
        communicator_relation = self._MPI.Comm.Compare(self.comm, self.domain.comm)
        if communicator_relation not in (self._MPI.IDENT, self._MPI.CONGRUENT):
            raise ValueError(
                "metric communicator must be identical or congruent to the mesh communicator"
            )
        self.tmax = float(tmax)
        self.alpha_t1 = float(alpha_t1)
        self.alpha_t2 = float(alpha_t2)
        if not math.isfinite(self.tmax) or self.tmax <= 0.0:
            raise ValueError("T_max must be positive and finite")
        if not (0.0 <= self.alpha_t1 < self.alpha_t2 <= 1.0):
            raise ValueError("normalized target levels must satisfy 0 <= alpha_t1 < alpha_t2 <= 1")
        if int(quadrature_degree) < 1:
            raise ValueError("quadrature degree must be positive")
        self.order = max(
            int(order if order is not None else u.function_space.element.basix_element.degree),
            1,
        )
        self.kernel_scale = float(kernel_scale)
        if not math.isfinite(self.kernel_scale) or self.kernel_scale <= 0.0:
            raise ValueError("coarea kernel scale must be positive and finite")
        self.kernel_floor = (
            128.0 * np.finfo(np.float64).eps
            if kernel_floor is None
            else float(kernel_floor)
        )
        if not math.isfinite(self.kernel_floor) or self.kernel_floor <= 0.0:
            raise ValueError("coarea kernel floor must be positive and finite")
        self.minimum_fit_score = float(minimum_fit_score)
        self.minimum_containment_score = float(minimum_containment_score)
        self.minimum_inward_fraction = float(minimum_inward_fraction)
        self.maximum_kernel_stability_error = float(maximum_kernel_stability_error)
        for name, value in (
            ("minimum fit score", self.minimum_fit_score),
            ("minimum containment score", self.minimum_containment_score),
            ("minimum inward fraction", self.minimum_inward_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        if self.maximum_kernel_stability_error < 0.0:
            raise ValueError("maximum kernel stability error must be nonnegative")

        if self.domain.topology.dim != 2 or self.domain.basix_cell() != basix.CellType.triangle:
            raise ValueError("distributed torsion-band metrics currently require a 2D triangle mesh")
        self.q_points, reference_weights = basix.make_quadrature(
            basix.CellType.triangle, int(quadrature_degree)
        )
        self.q_count = len(reference_weights)
        owned_cell_count = int(
            self.domain.topology.index_map(self.domain.topology.dim).size_local
        )
        self.owned_cells = np.arange(owned_cell_count, dtype=np.int32)
        geometry_dofs = np.asarray(self.domain.geometry.dofmaps[0], dtype=np.int64)[
            :owned_cell_count
        ]
        if geometry_dofs.ndim != 2 or geometry_dofs.shape[1] < 3:
            raise ValueError("distributed torsion-band metrics require affine triangular geometry")
        vertices = np.asarray(
            self.domain.geometry.x[geometry_dofs[:, :3], :2], dtype=np.float64
        )
        edge01 = vertices[:, 1] - vertices[:, 0]
        edge02 = vertices[:, 2] - vertices[:, 0]
        edge12 = vertices[:, 2] - vertices[:, 1]
        determinant = np.abs(
            edge01[:, 0] * edge02[:, 1] - edge01[:, 1] * edge02[:, 0]
        )
        cell_h = np.maximum.reduce(
            (
                np.linalg.norm(edge01, axis=1),
                np.linalg.norm(edge02, axis=1),
                np.linalg.norm(edge12, axis=1),
            )
        )
        self.weights = np.ascontiguousarray(
            (determinant[:, None] * np.asarray(reference_weights)[None, :]).reshape(-1)
        )
        self.h_eff = np.ascontiguousarray(
            np.repeat(cell_h / float(self.order), self.q_count)
        )
        self._phi_expression = fem.Expression(u, self.q_points)
        self._grad_phi_expression = fem.Expression(ufl.grad(u), self.q_points)
        torsion_expression = fem.Expression(torsion, self.q_points)
        grad_torsion_expression = fem.Expression(ufl.grad(torsion), self.q_points)
        self.tau_values = self._eval_scalar(torsion_expression) / self.tmax
        self.grad_tau = self._eval_vector(grad_torsion_expression) / self.tmax
        self.grad_tau_norm = np.linalg.norm(self.grad_tau, axis=1)
        self._target_mask = (
            (self.tau_values >= self.alpha_t1)
            & (self.tau_values <= self.alpha_t2)
        )
        local_domain_area = float(np.sum(self.weights))
        local_target_area = float(np.sum(self.weights[self._target_mask]))
        domain_area, target_area = self._sum_array(
            np.asarray([local_domain_area, local_target_area], dtype=np.float64)
        )

        target_curve_weights = [
            self._curve_weights(
                self.tau_values,
                self.grad_tau_norm,
                target_level,
                scale=self.kernel_scale,
            )
            for target_level in (self.alpha_t1, self.alpha_t2)
        ]
        self._target_curve_weights = tuple(
            np.ascontiguousarray(item, dtype=np.float64)
            for item in target_curve_weights
        )
        target_lengths = self._sum_array(
            np.asarray([np.sum(item) for item in target_curve_weights], dtype=np.float64)
        )
        if np.any(target_lengths <= 0.0):
            raise RuntimeError("target torsion level curves were not resolved by quadrature")
        normalized_cell_change = self.h_eff * self.grad_tau_norm
        resolutions = [
            self._distributed_weighted_quantile(
                normalized_cell_change, curve_weights, 0.75
            )
            for curve_weights in target_curve_weights
        ]
        physical_resolution = self._distributed_weighted_quantile(
            np.concatenate((self.h_eff, self.h_eff)),
            np.concatenate(tuple(target_curve_weights)),
            0.75,
        )
        delta_alpha = self.alpha_t2 - self.alpha_t1
        preliminary_tolerances = [
            geometry_tolerance(delta_alpha, resolution)
            for resolution in resolutions
        ]
        if histogram_bins is None:
            desired_bin_width = min(
                max(min(preliminary_tolerances) / 4.0, 1.0 / 8192.0),
                delta_alpha / 64.0,
            )
            histogram_bins = int(np.clip(math.ceil(1.0 / desired_bin_width), 256, 8192))
        self.histogram_bins = int(histogram_bins)
        if self.histogram_bins < 32:
            raise ValueError("torsion-coordinate histogram needs at least 32 bins")
        self.histogram_edges = np.linspace(
            0.0, 1.0, self.histogram_bins + 1, dtype=np.float64
        )
        histogram_bin_width = 1.0 / self.histogram_bins
        tolerances = [
            geometry_tolerance(
                delta_alpha,
                resolution,
                histogram_bin_width=histogram_bin_width,
            )
            for resolution in resolutions
        ]
        target_mean_thickness = 2.0 * float(target_area) / float(np.sum(target_lengths))
        underresolved = max(resolutions) > 0.25 * delta_alpha
        self.target = TargetGeometryCalibration(
            alpha_t1=self.alpha_t1,
            alpha_t2=self.alpha_t2,
            delta_alpha=delta_alpha,
            target_area=float(target_area),
            domain_area=float(domain_area),
            lower_curve_length=float(target_lengths[0]),
            upper_curve_length=float(target_lengths[1]),
            lower_resolution=float(resolutions[0]),
            upper_resolution=float(resolutions[1]),
            lower_tolerance=float(tolerances[0]),
            upper_tolerance=float(tolerances[1]),
            physical_resolution=float(physical_resolution),
            mean_physical_thickness=target_mean_thickness,
            histogram_bins=self.histogram_bins,
            histogram_bin_width=histogram_bin_width,
            underresolved=underresolved,
        )

    def _eval_scalar(self, expression: Any) -> np.ndarray:
        if not len(self.owned_cells):
            return np.empty(0, dtype=np.float64)
        values = np.asarray(
            expression.eval(self.domain, self.owned_cells), dtype=np.float64
        )
        return np.ascontiguousarray(values.reshape(len(self.owned_cells), self.q_count, -1)[..., 0].reshape(-1))

    def _eval_vector(self, expression: Any) -> np.ndarray:
        if not len(self.owned_cells):
            return np.empty((0, 2), dtype=np.float64)
        values = np.asarray(
            expression.eval(self.domain, self.owned_cells), dtype=np.float64
        )
        reshaped = values.reshape(len(self.owned_cells), self.q_count, -1)
        return np.ascontiguousarray(reshaped[..., :2].reshape(-1, 2))

    def _sum_array(self, local: np.ndarray) -> np.ndarray:
        local = np.ascontiguousarray(local, dtype=np.float64)
        reduced = np.empty_like(local)
        self.comm.Allreduce(local, reduced, op=self._MPI.SUM)
        return reduced

    def _distributed_weighted_quantile(
            self,
            values: np.ndarray,
            weights: np.ndarray,
            probability: float,
            *,
            bins: int = 1024,
    ) -> float:
        values = np.asarray(values, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
        local_total = float(np.sum(weights[valid]))
        total = float(self.comm.allreduce(local_total, op=self._MPI.SUM))
        if total <= 0.0:
            return math.nan
        local_min = float(np.min(values[valid])) if np.any(valid) else math.inf
        local_max = float(np.max(values[valid])) if np.any(valid) else -math.inf
        lower = float(self.comm.allreduce(local_min, op=self._MPI.MIN))
        upper = float(self.comm.allreduce(local_max, op=self._MPI.MAX))
        if not math.isfinite(lower) or not math.isfinite(upper):
            return math.nan
        range_scale = max(1.0, abs(lower), abs(upper))
        if upper - lower <= 64.0 * np.finfo(np.float64).eps * range_scale:
            return 0.5 * (lower + upper)
        edges = np.linspace(lower, upper, max(int(bins), 32) + 1)
        local_histogram, _ = np.histogram(values[valid], bins=edges, weights=weights[valid])
        histogram = self._sum_array(local_histogram.astype(np.float64))
        return float(histogram_quantiles(histogram, edges, [probability])[0])

    def _curve_weights(
            self,
            normalized_values: np.ndarray,
            gradient_norm: np.ndarray,
            level: float,
            *,
            scale: float,
    ) -> np.ndarray:
        bandwidth = (
            float(scale) * self.h_eff * np.asarray(gradient_norm, dtype=np.float64)
            + self.kernel_floor
        )
        return (
            self.weights
            * cosine_delta(np.asarray(normalized_values) - float(level), bandwidth)
            * np.asarray(gradient_norm, dtype=np.float64)
        )

    def _local_histogram(self, weights: np.ndarray) -> np.ndarray:
        values = np.clip(self.tau_values, 0.0, 1.0)
        histogram, _ = np.histogram(
            values, bins=self.histogram_edges, weights=np.asarray(weights)
        )
        return histogram.astype(np.float64, copy=False)

    def predict_threshold_pair(self, field: Any | None = None) -> tuple[float, float]:
        """Predict dimensional thresholds from the current geometry and field.

        The prediction is the arc-length-weighted median of ``field/T_max``
        on each target torsion boundary.  With ``field=None``, the mutable
        candidate potential supplied at construction is used.  A reversed
        pair is rejected rather than sorted because sorting would exchange the
        two target-boundary correspondences.
        """

        if field is None or field is self.u:
            values = self._eval_scalar(self._phi_expression) / self.tmax
        else:
            if field.function_space.mesh is not self.domain:
                raise ValueError("threshold-prediction field must use the workspace mesh")
            values = self._eval_scalar(self._fem.Expression(field, self.q_points)) / self.tmax
        normalized = tuple(
            self._distributed_weighted_quantile(values, weights, 0.50)
            for weights in self._target_curve_weights
        )
        if not all(math.isfinite(value) for value in normalized):
            raise RuntimeError("target-contour threshold medians are not finite")
        normalized_c1 = min(max(float(normalized[0]), 0.0), 1.0)
        normalized_c2 = min(max(float(normalized[1]), 0.0), 1.0)
        if normalized_c2 <= normalized_c1:
            raise RuntimeError(
                "target-contour threshold medians are not strictly increasing; "
                "the prediction field is not an admissible inward-monotone seed"
            )
        return normalized_c1 * self.tmax, normalized_c2 * self.tmax

    def evaluate(self, *, c1: float, c2: float) -> CandidateGeometryMetrics:
        """Evaluate current ``u`` against the target for one threshold pair."""

        c1 = float(c1)
        c2 = float(c2)
        if not math.isfinite(c1) or not math.isfinite(c2):
            raise ValueError("candidate thresholds must be finite")
        normalized_c1 = c1 / self.tmax
        normalized_c2 = c2 / self.tmax
        if not (0.0 <= normalized_c1 < normalized_c2 <= 1.0):
            raise ValueError("candidate thresholds must satisfy 0 <= c1 < c2 <= T_max")

        phi_values = self._eval_scalar(self._phi_expression) / self.tmax
        grad_phi = self._eval_vector(self._grad_phi_expression) / self.tmax
        grad_phi_norm = np.linalg.norm(grad_phi, axis=1)
        levels = (normalized_c1, normalized_c2)
        curve_weights = [
            self._curve_weights(phi_values, grad_phi_norm, level, scale=self.kernel_scale)
            for level in levels
        ]
        wide_curve_weights = [
            self._curve_weights(phi_values, grad_phi_norm, level, scale=2.0 * self.kernel_scale)
            for level in levels
        ]
        curve_histograms = np.vstack(
            [self._local_histogram(item) for item in curve_weights]
        )
        hard_mask = (phi_values >= normalized_c1) & (phi_values <= normalized_c2)
        hard_weights = self.weights * hard_mask
        hard_histogram = self._local_histogram(hard_weights)

        gradient_product = grad_phi_norm * self.grad_tau_norm
        orientation_valid = gradient_product > 128.0 * np.finfo(np.float64).eps
        alignment_cosine = np.zeros_like(gradient_product)
        alignment_cosine[orientation_valid] = np.sum(
            grad_phi[orientation_valid] * self.grad_tau[orientation_valid], axis=1
        ) / gradient_product[orientation_valid]
        alignment_cosine = np.clip(alignment_cosine, -1.0, 1.0)

        curve_scalars = np.zeros((2, self._CURVE_SCALAR_COUNT), dtype=np.float64)
        target_levels = (self.alpha_t1, self.alpha_t2)
        tolerances = (self.target.lower_tolerance, self.target.upper_tolerance)
        tolerant_band = (
            (self.tau_values >= self.alpha_t1 - tolerances[0])
            & (self.tau_values <= self.alpha_t2 + tolerances[1])
        )
        for index, (weights, wide_weights, target_level, tolerance) in enumerate(
                zip(curve_weights, wide_curve_weights, target_levels, tolerances, strict=True)
        ):
            valid_weights = weights * orientation_valid
            curve_scalars[index] = (
                np.sum(weights),
                np.sum(weights[self._target_mask]),
                np.sum(weights[tolerant_band]),
                np.sum(weights[np.abs(self.tau_values - target_level) <= tolerance]),
                np.sum(valid_weights),
                np.sum(valid_weights[alignment_cosine > 0.0]),
                np.sum(valid_weights * alignment_cosine),
                np.sum(wide_weights),
            )

        local_payload = np.concatenate(
            (
                curve_histograms.reshape(-1),
                hard_histogram,
                curve_scalars.reshape(-1),
                np.asarray(
                    [np.sum(hard_weights[self._target_mask])], dtype=np.float64
                ),
            )
        )
        reduced = self._sum_array(local_payload)
        cursor = 0
        histogram_count = 2 * self.histogram_bins
        global_curve_histograms = reduced[cursor:cursor + histogram_count].reshape(
            2, self.histogram_bins
        )
        cursor += histogram_count
        global_hard_histogram = reduced[cursor:cursor + self.histogram_bins]
        cursor += self.histogram_bins
        global_curve_scalars = reduced[
            cursor:cursor + 2 * self._CURVE_SCALAR_COUNT
        ].reshape(2, self._CURVE_SCALAR_COUNT)
        cursor += 2 * self._CURVE_SCALAR_COUNT
        intersection = float(reduced[cursor])

        target_lengths = (
            self.target.lower_curve_length,
            self.target.upper_curve_length,
        )
        curves: list[CurveAlignmentMetrics] = []
        for index in range(2):
            q10, q50, q90 = histogram_quantiles(
                global_curve_histograms[index],
                self.histogram_edges,
                [0.10, 0.50, 0.90],
            )
            (
                length,
                crisp_measure,
                tolerant_measure,
                boundary_measure,
                orientation_measure,
                inward_measure,
                cosine_measure,
                wide_length,
            ) = map(float, global_curve_scalars[index])
            active_floor = max(1.0e-10 * math.sqrt(self.target.domain_area), 1.0e-8 * target_lengths[index])
            active = math.isfinite(length) and length > active_floor
            stability = (
                abs(length - wide_length) / max(length, wide_length, active_floor)
                if active
                else math.inf
            )
            resolved = active and stability <= self.maximum_kernel_stability_error
            orientation_coverage = orientation_measure / length if active else 0.0
            inward_fraction = inward_measure / orientation_measure if orientation_measure > 0.0 else 0.0
            mean_cosine = cosine_measure / orientation_measure if orientation_measure > 0.0 else math.nan
            target_level = target_levels[index]
            curves.append(
                CurveAlignmentMetrics(
                    level=levels[index],
                    target_level=target_level,
                    length=length,
                    active=active,
                    resolved=resolved,
                    tau_q10=float(q10),
                    tau_q50=float(q50),
                    tau_q90=float(q90),
                    signed_position_residual=float(q50 - target_level),
                    tau_spread=float(q90 - q10),
                    containment_crisp=crisp_measure / length if active else 0.0,
                    containment_tolerant=tolerant_measure / length if active else 0.0,
                    boundary_fit_score=boundary_measure / length if active else 0.0,
                    orientation_coverage=orientation_coverage,
                    inward_fraction=inward_fraction,
                    mean_alignment_cosine=mean_cosine,
                    length_ratio_to_target=length / target_lengths[index],
                    kernel_stability_error=stability,
                )
            )
        lower, upper = curves
        hard_q10, hard_q90 = histogram_quantiles(
            global_hard_histogram, self.histogram_edges, [0.10, 0.90]
        )
        hard_band = hard_overlap_from_measures(
            candidate_area=float(np.sum(global_hard_histogram)),
            target_area=self.target.target_area,
            intersection=intersection,
            tau_q10=float(hard_q10),
            tau_q90=float(hard_q90),
            lower_curve_length=lower.length,
            upper_curve_length=upper.length,
            target_mean_physical_thickness=self.target.mean_physical_thickness,
        )
        robust_span, span_limit, robust_too_thick, too_thin = robust_thickness_classification(
            lower_tau_q10=lower.tau_q10,
            upper_tau_q90=upper.tau_q90,
            delta_alpha=self.target.delta_alpha,
            lower_tolerance=self.target.lower_tolerance,
            upper_tolerance=self.target.upper_tolerance,
        )
        physical_too_thick = (
            math.isfinite(hard_band.mean_physical_thickness)
            and hard_band.mean_physical_thickness
            > self.target.mean_physical_thickness + 2.0 * self.target.physical_resolution
        )
        too_thick, hard_span_too_thick, robust_span_warning = (
            corroborated_thickness_classification(
                robust_too_thick=robust_too_thick,
                hard_band_tau_span=hard_band.tau_span,
                robust_span_limit=span_limit,
                physical_too_thick=physical_too_thick,
            )
        )
        pair_fit_score = min(lower.boundary_fit_score, upper.boundary_fit_score)
        pair_containment_score = min(
            lower.containment_tolerant, upper.containment_tolerant
        )
        monotone = all(
            curve.active
            and curve.resolved
            and curve.orientation_coverage >= self.minimum_inward_fraction
            and curve.inward_fraction >= self.minimum_inward_fraction
            for curve in curves
        )
        reasons: list[str] = []
        if not lower.active or not upper.active:
            reasons.append("INACTIVE_CURVE")
        if (lower.active and not lower.resolved) or (upper.active and not upper.resolved):
            reasons.append("UNRESOLVED_CURVE")
        if self.target.underresolved:
            reasons.append("TARGET_BAND_UNDERRESOLVED")
        if pair_containment_score < self.minimum_containment_score:
            reasons.append("OFF_TARGET_CURVE")
        if pair_fit_score < self.minimum_fit_score:
            reasons.append("BOUNDARY_FIT_INSUFFICIENT")
        if (
            abs(lower.signed_position_residual) > self.target.lower_tolerance
            or abs(upper.signed_position_residual) > self.target.upper_tolerance
        ):
            reasons.append("CURVE_POSITION_UNRESOLVED")
        if (
            lower.tau_spread > 2.0 * self.target.lower_tolerance
            or upper.tau_spread > 2.0 * self.target.upper_tolerance
        ):
            reasons.append("CURVE_SPREAD_EXCESSIVE")
        if too_thick:
            reasons.append("THICK_BAND_REJECTED")
        if not monotone:
            reasons.append("NONMONOTONE_CURVE")
        geometrically_eligible = not reasons
        return CandidateGeometryMetrics(
            c1=c1,
            c2=c2,
            normalized_c1=normalized_c1,
            normalized_c2=normalized_c2,
            lower=lower,
            upper=upper,
            hard_band=hard_band,
            pair_fit_score=pair_fit_score,
            pair_containment_score=pair_containment_score,
            lower_position_tolerance=self.target.lower_tolerance,
            upper_position_tolerance=self.target.upper_tolerance,
            lower_target_resolution=self.target.lower_resolution,
            upper_target_resolution=self.target.upper_resolution,
            target_underresolved=self.target.underresolved,
            robust_tau_span=robust_span,
            robust_span_limit=span_limit,
            robust_too_thick=robust_too_thick,
            hard_span_too_thick=hard_span_too_thick,
            robust_span_warning=robust_span_warning,
            too_thick=too_thick,
            too_thin=too_thin,
            physical_too_thick=physical_too_thick,
            monotone_for_bisection=monotone,
            geometrically_eligible=geometrically_eligible,
            rejection_reason=";".join(reasons) if reasons else "ELIGIBLE",
            warning_reason="ROBUST_SPAN_WARNING" if robust_span_warning else "",
        )


def evaluate_candidate_geometry(
        workspace: DistributedBandMetricWorkspace,
        *,
        c1: float,
        c2: float,
) -> dict[str, float | int | str]:
    """Convenience adapter returning flattened candidate geometry diagnostics."""

    return workspace.evaluate(c1=c1, c2=c2).as_dict()


__all__ = [
    "CandidateGeometryMetrics",
    "CurveAlignmentMetrics",
    "DistributedBandMetricWorkspace",
    "HardBandMetrics",
    "TargetGeometryCalibration",
    "cosine_delta",
    "corroborated_thickness_classification",
    "evaluate_candidate_geometry",
    "geometry_tolerance",
    "hard_overlap_from_measures",
    "histogram_quantiles",
    "robust_thickness_classification",
]
