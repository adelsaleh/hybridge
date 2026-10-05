"""Packed geometry and immutable accepted state records."""
from __future__ import annotations
from dataclasses import dataclass, field
import uuid
import numpy as np


@dataclass(frozen=True)
class EquilibriumState:
    values: np.ndarray
    m: float
    delta_fixed: float
    epsilon_fixed: float
    branch_id: str
    residual_norm: float
    snes_reason: int
    nonlinear_iterations: int
    state_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_id: str | None = None
    energy: float = float("nan")
    source_mass: float = float("nan")
    newton_error: float = float("nan")
    stability: str = "UNASSESSED"
    stability_eigenvalue: float = float("nan")
    # KSP is recorded separately from SNES so a positive nonlinear reason can
    # be audited together with the final linear solve.  Zero is retained for
    # checkpoints written before this field was introduced.
    ksp_reason: int = 0

    def __post_init__(self):
        values = np.array(self.values, dtype=np.float64, order="C", copy=True)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("equilibrium snapshot must contain finite owned coefficients")
        values.flags.writeable = False
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class RayAtlas:
    x_T: np.ndarray
    segments: np.ndarray       # (n_segments, 2 endpoints, 2 coordinates)
    reference_segments: np.ndarray  # same endpoints in the owning cell's reference triangle
    cells: np.ndarray          # stable global cell identifiers
    offsets: np.ndarray        # ray segment prefix offsets
    s_start: np.ndarray         # physical center-to-segment-start arclength
    segment_lengths: np.ndarray
    total_lengths: np.ndarray
    weights: np.ndarray        # global probability weights, not rank-normalized
    global_ray_ids: np.ndarray
    mesh_signature: str
    atlas_signature: str
    endpoint_error: np.ndarray
    global_ray_count: int
    resolved_torsion_fraction: float = 1.0
    flow_resolution: float = 0.0
    unresolved_neighbor_pairs: int = 0

    def __post_init__(self):
        """A fixed atlas cannot be modified while its sensitivities are reused."""
        nrays = len(self.total_lengths)
        nsegments = len(self.segments)
        if (self.segments.shape != (nsegments, 2, 2)
                or self.reference_segments.shape != (nsegments, 2, 2)
                or self.offsets.shape != (nrays+1,)
                or self.offsets[0] != 0 or self.offsets[-1] != nsegments
                or np.any(np.diff(self.offsets) <= 0)):
            raise ValueError("INVALID_PACKED_RAY_LAYOUT")
        if (len(self.cells) != nsegments or len(self.s_start) != nsegments or len(self.segment_lengths) != nsegments
                or len(self.weights) != nrays or len(self.global_ray_ids) != nrays
                or np.any(self.total_lengths <= 0) or np.any(self.segment_lengths <= 0)
                or np.any(self.weights <= 0)):
            raise ValueError("INVALID_RAY_DATA")
        if (not 0 < self.resolved_torsion_fraction <= 1
                or not np.isfinite(self.flow_resolution) or self.flow_resolution < 0
                or self.unresolved_neighbor_pairs < 0):
            raise ValueError("INVALID_FLOW_RESOLUTION_AUDIT")
        for name in ("x_T", "segments", "reference_segments", "cells", "offsets", "s_start", "segment_lengths",
                     "total_lengths", "weights", "global_ray_ids", "endpoint_error"):
            value = np.ascontiguousarray(getattr(self, name))
            if not np.all(np.isfinite(value)):
                raise ValueError(f"NONFINITE_RAY_DATA: {name}")
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    @property
    def zeta(self):
        ray = np.repeat(np.arange(len(self.total_lengths)), np.diff(self.offsets))
        return self.s_start / self.total_lengths[ray]


@dataclass
class BandMetrics:
    distance: float
    distance_error: float
    s_crossings: np.ndarray     # columns: upper, middle, lower
    zeta_middle: np.ndarray
    crossing_counts: np.ndarray
    slopes: np.ndarray          # signed dphi/ds at each crossing
    min_transversality: float
    admissible: bool
    reason: str
    mean_physical_thickness: float = float("nan")
    physical_thickness_variance: float = float("nan")
    distance_variance: float = float("nan")
    contour_distance: float = float("nan")
    # Legacy checkpoint name.  This is phi(x_T)-c_plus: it certifies that the
    # innermost threshold exists around the torsion center.  It is *not* a
    # measure of the atlas's unresolved torsion-flow core.
    core_margin: float = float("nan")
    # Difference between the atlas's resolved T/T_max cutoff and the largest
    # T/T_max reached by the inner c_plus interface. NaN means the atlas is
    # resolved all the way to its center-stop neighborhood, so no cutoff guard
    # is applicable.
    flow_core_torsion_margin: float = float("nan")

    @property
    def inner_threshold_margin(self) -> float:
        """Return ``phi(x_T)-c_plus`` under its unambiguous public name."""
        return self.core_margin


@dataclass
class BranchPoint:
    state: EquilibriumState
    metrics: BandMetrics
    segment_id: str
    arc_length: float = 0.0
    parameterization: str = "midpoint"


@dataclass
class BranchScan:
    points: list[BranchPoint] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    stop_reason: str = ""
    folds: int = 0


@dataclass
class TargetResult:
    point: BranchPoint | None
    exact_target_reached: bool
    distance_error: float
    feasibility_gap: float | None
    status: str
    explored_m_interval: tuple[float, float] | None = None
    explored_branch_ids: tuple[str, ...] = ()
