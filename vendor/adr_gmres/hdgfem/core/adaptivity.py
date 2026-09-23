"""Reusable DG mesh-adaptivity helpers.

The routines here build scalar mesh-size fields from DG indicators and generate
new Gmsh meshes with native background fields.  They are intentionally
PDE-agnostic: callers supply the indicator field and the target domain geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.spatial import cKDTree

from .mesh import DGMesh, gmsh_smooth_star_mesh_with_background_sizes
from .space import DGField, DGSpace
from ..assembly.projection import project_quadrature_values
from ..io.output import logv, timed_section


ScoreReduction = str


@dataclass(frozen=True)
class SmoothStarGeometry:
    """Smooth star geometry parameters for background-field remeshing.

    Parameters
    ----------
    boundary_points
        Number of points used to discretize the smooth star boundary.
    radius
        Base radius of the star.
    amplitude
        Sinusoidal radial perturbation amplitude.
    mode
        Angular frequency of the sinusoidal perturbation.
    """

    boundary_points: int = 260
    radius: float = 1.5
    amplitude: float = 0.32
    mode: int = 5


@dataclass(frozen=True)
class StructuredSizeOptions:
    """Controls for mapping DG indicators to a structured Gmsh size field.

    Parameters
    ----------
    low_quantile, high_quantile
        Quantiles used to normalize indicator samples before converting them to
        target sizes.
    hmin_factor, hmax_factor
        Multipliers applied to the caller-provided mesh-size bounds.
    size_sensitivity, size_power
        Parameters in ``h = hmin + (hmax-hmin)/(1+sensitivity*score**power)``.
    score_reduction
        Reduction used to turn quadrature-point scores into one score per
        element.  Supported values are ``"mean"``, ``"q75"``, ``"q90"``, and
        ``"max"``.
    verbosity, gmsh_verbosity
        Logging level for this helper and for Gmsh.
    gmsh_algorithm
        Optional Gmsh 2D meshing algorithm id.
    timing_prefix
        Prefix used for verbose timing messages.
    """

    low_quantile: float = 0.10
    high_quantile: float = 0.98
    hmin_factor: float = 1.0
    hmax_factor: float = 1.0
    size_sensitivity: float = 100.0
    size_power: float = 1.0
    score_reduction: ScoreReduction = "q75"
    verbosity: int = 0
    gmsh_verbosity: int = 0
    gmsh_algorithm: int | None = None
    timing_prefix: str = "ADAPT"


def gradient_weighted_indicator(
        field: DGField,
        length_scale: float,
        *,
        grad_weight: float = 10.0,
        name: str = "adapt_indicator",
) -> DGField:
    r"""Project ``field + grad_weight*length_scale*|grad(field)|`` into DG space.

    Parameters
    ----------
    field
        Scalar DG field whose values and elementwise gradients define the
        indicator.  The returned field lives in ``field.space``.
    length_scale
        Mesh or physical length used to make the gradient contribution have the
        same scale as the scalar field.
    grad_weight
        Dimensionless multiplier for the gradient term.
    name
        Name assigned to the returned DG field.

    Returns
    -------
    DGField
        Projection of the pointwise indicator to the same DG space.

    Notes
    -----
    This is PDE-agnostic.  It is useful for interface or band remeshing because
    it keeps the scalar field visible while adding sensitivity near steep
    gradients.
    """
    dx, dy = field.grad_values()
    values = field.values() + float(grad_weight) * float(length_scale) * np.sqrt(dx * dx + dy * dy)
    return project_quadrature_values(field.space, values, name=name)


def _reduce_scores(element_score_samples: np.ndarray, mode: ScoreReduction) -> np.ndarray:
    """Reduce pointwise indicator scores to one score per element."""
    if mode == "max":
        return np.max(element_score_samples, axis=1)
    if mode == "mean":
        return np.mean(element_score_samples, axis=1)
    if mode == "q75":
        return np.quantile(element_score_samples, 0.75, axis=1)
    if mode == "q90":
        return np.quantile(element_score_samples, 0.90, axis=1)
    raise ValueError("score_reduction must be one of 'mean', 'q75', 'q90', or 'max'")


def structured_size_field_from_indicator(
        space: DGSpace,
        indicator: DGField,
        *,
        hmin: float,
        hmax: float,
        options: StructuredSizeOptions = StructuredSizeOptions(),
) -> tuple[tuple[float, float], tuple[float, float], np.ndarray, dict[str, float]]:
    """Build a structured background size field from a scalar DG indicator.

    Parameters
    ----------
    space
        DG space containing the mesh on which the indicator is defined.
    indicator
        Scalar DG indicator field in ``space``.  Its quadrature values are
        normalized by ``options.low_quantile`` and ``options.high_quantile``.
    hmin, hmax
        Base mesh-size bounds before applying ``options.hmin_factor`` and
        ``options.hmax_factor``.
    options
        Structured-size-field construction options.

    Returns
    -------
    origin
        ``(x0, y0)`` lower-left corner of the structured field.
    spacing
        ``(dx, dy)`` grid spacing of the structured field.
    values
        Contiguous array of shape ``(nx, ny)`` containing target mesh sizes.
    info
        Dictionary with indicator quantiles, effective size bounds, target-size
        extrema, grid dimensions, and timing data added by callers when
        relevant.

    Notes
    -----
    Each structured grid point receives the target size of its nearest element
    centroid.  Gmsh consumes the result through its native ``Structured`` field,
    avoiding per-query Python mesh-size callbacks.
    """
    space.assert_same_mesh(indicator.space)
    prefix = options.timing_prefix
    with timed_section(options, 2, f"{prefix}_INDICATOR_SAMPLE", nt=space.mesh.num_tri, ndof=space.ndof):
        indicator_values = np.asarray(indicator.values(), dtype=np.float64)
        raw_values = indicator_values.reshape(-1)
        if raw_values.size == 0:
            raise ValueError("empty adaptivity indicator")
        lo = float(np.quantile(raw_values, options.low_quantile))
        hi = float(np.quantile(raw_values, options.high_quantile))
        if not np.isfinite(hi - lo) or hi <= lo:
            hi = float(np.max(raw_values))
            lo = float(np.min(raw_values))
        scale = max(hi - lo, 1.0e-30)
        scores = np.clip((raw_values - lo) / scale, 0.0, 1.0)
        logv(
            options,
            2,
            f"{prefix}_INDICATOR_STATS min={np.min(raw_values):.6e} max={np.max(raw_values):.6e} "
            f"qlo={lo:.6e} qhi={hi:.6e} samples={raw_values.size}",
        )

    with timed_section(options, 2, f"{prefix}_BACKGROUND_BUILD", nt=space.mesh.num_tri):
        hmin_eff = float(options.hmin_factor) * float(hmin)
        hmax_eff = float(options.hmax_factor) * float(hmax)
        if not np.isfinite(hmin_eff) or hmin_eff <= 0.0:
            raise ValueError(f"hmin_factor gives invalid hmin {hmin_eff}")
        if not np.isfinite(hmax_eff) or hmax_eff <= 0.0:
            raise ValueError(f"hmax_factor gives invalid hmax {hmax_eff}")
        hmax_eff = max(hmax_eff, hmin_eff)

        element_score_samples = scores.reshape(space.mesh.num_tri, -1)
        element_scores = _reduce_scores(element_score_samples, options.score_reduction)
        element_scores = np.power(element_scores, float(options.size_power))
        element_sizes = hmin_eff + (hmax_eff - hmin_eff) / (
            1.0 + float(options.size_sensitivity) * element_scores
        )
        element_sizes = np.clip(element_sizes, hmin_eff, hmax_eff)

        xy_min = np.min(space.mesh.node_coords, axis=0) - hmax_eff
        xy_max = np.max(space.mesh.node_coords, axis=0) + hmax_eff
        grid_h = max(0.5 * hmin_eff, 1.0e-12)
        nx = int(np.ceil((xy_max[0] - xy_min[0]) / grid_h)) + 1
        ny = int(np.ceil((xy_max[1] - xy_min[1]) / grid_h)) + 1
        xs = xy_min[0] + grid_h * np.arange(nx, dtype=np.float64)
        ys = xy_min[1] + grid_h * np.arange(ny, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
        grid_points = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
        centroids = np.mean(space.mesh.element_vertices, axis=1)
        _, nearest = cKDTree(centroids).query(grid_points, k=1, workers=-1)
        background_values = np.ascontiguousarray(element_sizes[nearest].reshape(nx, ny), dtype=np.float64)
        logv(
            options,
            2,
            f"{prefix}_BACKGROUND_STATS hminEff={hmin_eff:.6e} hmaxEff={hmax_eff:.6e} "
            f"targetMin={np.min(element_sizes):.6e} targetMax={np.max(element_sizes):.6e} "
            f"scoreMax={np.max(element_scores):.6e} reduction={options.score_reduction} "
            f"sensitivity={options.size_sensitivity:.6e} nx={nx} ny={ny}",
        )

    info = {
        "indicator_min": float(np.min(raw_values)),
        "indicator_max": float(np.max(raw_values)),
        "indicator_qlo": lo,
        "indicator_qhi": hi,
        "hmin_eff": hmin_eff,
        "hmax_eff": hmax_eff,
        "size_target_min": float(np.min(element_sizes)),
        "size_target_max": float(np.max(element_sizes)),
        "nx": float(nx),
        "ny": float(ny),
    }
    return (float(xy_min[0]), float(xy_min[1])), (grid_h, grid_h), background_values, info


def remesh_smooth_star_from_indicator(
        space: DGSpace,
        indicator: DGField,
        *,
        geometry: SmoothStarGeometry,
        hmin: float,
        hmax: float,
        options: StructuredSizeOptions = StructuredSizeOptions(),
) -> tuple[DGMesh, dict[str, float]]:
    """Generate a smooth-star mesh from a DG indicator and structured size field.

    Parameters
    ----------
    space
        DG space that owns the current mesh and indicator.
    indicator
        Scalar DG indicator field in ``space``.
    geometry
        Smooth-star geometry parameters used to rebuild the domain boundary.
    hmin, hmax
        Base mesh-size bounds passed to
        :func:`structured_size_field_from_indicator`.
    options
        Structured-size-field and Gmsh logging options.

    Returns
    -------
    mesh
        Newly generated DG mesh on the smooth-star domain.
    info
        Dictionary returned by :func:`structured_size_field_from_indicator`,
        augmented with total elapsed remeshing time.
    """
    total_start = time.perf_counter()
    origin, spacing, background_values, info = structured_size_field_from_indicator(
        space,
        indicator,
        hmin=hmin,
        hmax=hmax,
        options=options,
    )
    prefix = options.timing_prefix
    with timed_section(
            options,
            2,
            f"{prefix}_GMSH_GENERATE",
            hmin=f"{info['hmin_eff']:.6e}",
            hmax=f"{info['hmax_eff']:.6e}",
            field="Structured",
    ):
        mesh = gmsh_smooth_star_mesh_with_background_sizes(
            boundary_points=geometry.boundary_points,
            radius=geometry.radius,
            amplitude=geometry.amplitude,
            mode=geometry.mode,
            hmin=info["hmin_eff"],
            hmax=info["hmax_eff"],
            background_origin=origin,
            background_spacing=spacing,
            background_values=background_values,
            verbosity=options.gmsh_verbosity,
            algorithm=options.gmsh_algorithm,
            timing_prefix=f"{prefix}_GMSH" if options.verbosity >= 2 else None,
        )
    elapsed = time.perf_counter() - total_start
    logv(options, 2, f"{prefix}_MESH_RESULT nt={mesh.num_tri} nv={mesh.node_coords.shape[0]} elapsed={elapsed:.3f}")
    info["elapsed"] = elapsed
    return mesh, info


__all__ = [
    "SmoothStarGeometry",
    "StructuredSizeOptions",
    "gradient_weighted_indicator",
    "remesh_smooth_star_from_indicator",
    "structured_size_field_from_indicator",
]
