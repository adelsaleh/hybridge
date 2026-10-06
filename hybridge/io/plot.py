"""Plotting helpers for :mod:`hybridge` fields.

PyVista helpers provide interactive refined-surface visualizations for medium
and large DG meshes.  Matplotlib helpers provide lightweight discontinuous
``tricontourf`` panels for small meshes.  Both paths sample each DG element
independently and intentionally duplicate vertices on shared mesh edges so
discontinuities remain visible instead of being averaged by the rendering
backend.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import sqrt
import os

import numpy as np
from scipy.spatial import Delaunay

from hybridge.core.mesh import DGMesh
from hybridge.core.space import DGField
from hybridge.core.quadrature import reference_plot_points


_TEXT_BACKEND_SET = False


def _use_freetype_text() -> None:
    """Render VTK text with FreeType instead of Matplotlib mathtext.

    VTK's default text backend detects Matplotlib and lays out every string
    (scalar-bar labels, captions) through mathtext callbacks into Python. That
    cost about 5 s per update of six panels (2026-09-29, 30x slower than
    FreeType) and, because VTK swallows Python errors raised in those
    callbacks, a Ctrl-C arriving during a render was lost. HYBRIDGE labels use
    no LaTeX, so plain FreeType rendering is used.
    """
    global _TEXT_BACKEND_SET
    if _TEXT_BACKEND_SET:
        return
    try:
        import vtkmodules.vtkRenderingFreeType  # noqa: F401 - registers the text renderer
        from vtkmodules.vtkRenderingCore import vtkTextRenderer
        renderer = vtkTextRenderer.GetInstance()
        if renderer is not None:
            renderer.SetDefaultBackend(vtkTextRenderer.FreeType)
    except ImportError:
        pass
    _TEXT_BACKEND_SET = True


def _require_pyvista():
    """Import PyVista lazily so non-plotting code has no plotting dependency."""
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("hybridge plotting helpers require pyvista") from exc
    _use_freetype_text()
    return pv


def _normalize_sample_values(values, num_elements: int, num_points: int) -> np.ndarray:
    """Normalize scalar/sample callable output to ``(num_elements, num_points)``."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape == (num_elements, num_points):
        return array
    if array.shape == (num_points,):
        return np.broadcast_to(array[None, :], (num_elements, num_points))
    if array.ndim == 0:
        return np.full((num_elements, num_points), float(array), dtype=np.float64)
    raise ValueError(
        "sample values must be a scalar, have shape "
        f"({num_points},), or have shape ({num_elements}, {num_points}); "
        f"got {array.shape}"
    )


def _expand_clim(minimum: float, maximum: float) -> tuple[float, float]:
    """Return finite color limits, expanding constants for plotting backends."""
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        return 0.0, 1.0
    if minimum == maximum:
        return minimum, minimum + 1.0
    return minimum, maximum


def _robust_clim(
        values: np.ndarray,
        *,
        percentile: float = 95.0,
        zero_min: bool = False,
        symmetric: bool = False,
) -> tuple[float, float]:
    """Return robust finite color limits from scalar samples.

    Ordinary scalar fields use the central ``percentile`` percent of finite
    values.  Error-like fields can request ``zero_min=True`` to keep zero fixed
    and use the selected percentile as the upper limit.
    """
    percentile = float(percentile)
    if not np.isfinite(percentile) or percentile <= 0.0 or percentile > 100.0:
        raise ValueError("percentile must satisfy 0 < percentile <= 100")
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if symmetric and zero_min:
        raise ValueError("symmetric and zero_min cannot both be enabled")
    if finite.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    if symmetric:
        extent = max(float(np.percentile(np.abs(finite), percentile)), 1.e-30)
        return -extent, extent

    if zero_min:
        minimum = 0.0
        maximum = float(np.percentile(finite, percentile))
        return _expand_clim(minimum, maximum)

    if percentile == 100.0:
        minimum = float(np.min(finite))
        maximum = float(np.max(finite))
    else:
        tail = 0.5 * (100.0 - percentile)
        minimum, maximum = (
            float(v) for v in np.percentile(finite, (tail, 100.0 - tail))
        )
    return _expand_clim(minimum, maximum)


def scalar_color_limits(
        values: np.ndarray,
        *,
        percentile: float = 95.0,
        zero_min: bool = False,
        symmetric: bool = False,
) -> tuple[float, float]:
    """Return finite display limits for host scalar samples.

    By default use the central ``percentile`` percent of finite samples.
    ``symmetric=True`` uses the selected percentile of absolute values around
    zero; ``percentile=100`` includes every finite sample. ``zero_min=True``
    anchors error-like quantities at zero. These two policies are exclusive.
    NaNs/infinities are ignored and constant ranges are expanded.
    """
    return _robust_clim(values, percentile=percentile, zero_min=zero_min, symmetric=symmetric)


def _safe_clim(values: np.ndarray) -> tuple[float, float]:
    """Return a robust finite color range, expanding constants for PyVista."""
    return _robust_clim(values, percentile=95.0)


def _mesh_overlay_line_width(mesh: DGMesh) -> float:
    """Choose a PyVista wireframe width that does not saturate dense meshes."""
    elements = int(mesh.num_tri)
    if elements >= 50_000:
        return 0.25
    if elements >= 10_000:
        return 0.4
    if elements >= 2_000:
        return 0.65
    return 1.0


def resolve_field_plot_resolution(
        requested_resolution: int | None,
        *,
        order: int,
        num_elements: int,
        default: int = 20,
        coarse_element_limit: int = 130,
) -> int:
    """Choose a plot grid with a polynomial-degree minimum on coarse meshes."""
    requested = int(default if requested_resolution is None else requested_resolution)
    if requested < 2:
        raise ValueError("plot resolution must be at least 2")
    if int(num_elements) <= int(coarse_element_limit):
        return max(requested, 2 * int(order) + 3, 3)
    return requested


def resolve_postprocessed_plot_resolution(
        requested_resolution: int | None,
        *,
        order: int,
        num_elements: int,
        default: int = 20,
) -> int:
    """Choose a plot grid able to display degree-``order + 1`` data."""
    base = resolve_field_plot_resolution(
        requested_resolution, order=order, num_elements=num_elements, default=default,
    )
    return resolve_field_plot_resolution(
        max(base, 2 * (int(order) + 1) + 3),
        order=int(order) + 1,
        num_elements=num_elements,
        default=default,
    )


def contour_levels_for_order(order: int) -> int:
    """Choose enough Matplotlib contour bands for a degree-``order`` field."""
    return min(256, max(128, 24 * (int(order) + 1)))


def _triangle_grid_point_count(resolution: int) -> int:
    """Return the number of points produced by :func:`reference_plot_points`."""
    resolution = int(resolution)
    return resolution * (resolution + 1) // 2


def _auto_exact_plot_resolution(
        *,
        num_elements: int,
        max_total_points: int = 5_000_000,
        max_resolution: int = 100,
) -> int:
    """Choose a dense exact-plot resolution without unbounded memory growth."""
    num_elements = max(1, int(num_elements))
    max_points_per_element = max(1, int(max_total_points) // num_elements)
    candidate = int((sqrt(8.0 * max_points_per_element + 1.0) - 1.0) // 2)
    return max(2, min(int(max_resolution), candidate))


def resolve_exact_plot_resolution(
        exact_resolution: int | str | None,
        *,
        numerical_resolution: int,
        num_elements: int,
        max_total_points: int = 5_000_000,
        max_resolution: int = 100,
) -> int:
    """Resolve an exact-panel plotting resolution policy.

    ``None`` and ``"same"`` use the numerical plotting resolution. ``"auto"``
    chooses a mesh-size-capped dense resolution that is independent of the
    solution polynomial order. Integer values, including numeric strings from
    command-line arguments, are interpreted directly.
    """
    if exact_resolution is None:
        return int(numerical_resolution)
    if isinstance(exact_resolution, str):
        policy = exact_resolution.strip().lower()
        if policy == "same":
            return int(numerical_resolution)
        if policy == "auto":
            return _auto_exact_plot_resolution(
                num_elements=num_elements,
                max_total_points=max_total_points,
                max_resolution=max_resolution,
            )
        try:
            exact_resolution = int(policy)
        except ValueError as exc:
            raise ValueError("exact_resolution must be an integer, 'same', 'auto', or None") from exc
    else:
        exact_resolution = int(exact_resolution)
    if exact_resolution < 2:
        raise ValueError("exact_resolution must be at least 2")
    return exact_resolution


def _resolve_exact_plot_resolution(
        exact_resolution: int | str | None,
        *,
        numerical_resolution: int,
        num_elements: int,
) -> int:
    """Backward-compatible private wrapper for exact-panel resolution."""
    return resolve_exact_plot_resolution(
        exact_resolution,
        numerical_resolution=numerical_resolution,
        num_elements=num_elements,
    )


def reference_plot_connectivity(reference_points: np.ndarray) -> np.ndarray:
    r"""Triangulate plotting points on :math:`\hat K`.

    The result has shape ``(num_plot_triangles, 3)`` and can be offset per
    element to build a refined discontinuous visualization mesh.
    """
    points = np.asarray(reference_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("reference_points must have shape (num_points, 2)")
    if points.shape[0] < 3:
        raise ValueError("at least three reference points are required")
    return np.ascontiguousarray(Delaunay(points).simplices, dtype=np.int64)


def sample_field_on_elements(
        field: DGField,
        *,
        resolution: int = 20,
        reference_points: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Sample a scalar DG field on every element.

    Parameters
    ----------
    field
        Scalar DG field to sample.
    resolution
        Number of points per reference coordinate direction.  Ignored when
        ``reference_points`` is supplied.
    reference_points
        Optional custom points on :math:`\hat K` with shape
        ``(num_plot_points, 2)``.

    Returns
    -------
    reference_points
        The reference sampling points.
    physical_points
        Mapped physical coordinates with shape
        ``(num_elements, num_plot_points, 2)``.
    values
        Field values with shape ``(num_elements, num_plot_points)``.
    """
    if reference_points is None:
        reference_points = reference_plot_points(resolution)
    else:
        reference_points = np.ascontiguousarray(reference_points, dtype=np.float64)
    physical_points = field.space.mesh.map_reference_points(reference_points)
    values = field.values_at_ref(reference_points)
    return reference_points, physical_points, values


def sample_callable_on_elements(
        mesh: DGMesh,
        function: Callable,
        *,
        resolution: int = 20,
        reference_points: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Sample a scalar callable on every physical element.

    Parameters
    ----------
    mesh
        Mesh whose affine element maps are used for sampling.
    function
        Callable ``function(x, y)`` evaluated on mapped physical points.
    resolution
        Number of points per reference coordinate direction.  Ignored when
        ``reference_points`` is supplied.
    reference_points
        Optional custom points on :math:`\hat K`.

    Returns
    -------
    reference_points
        Reference sampling points.
    physical_points
        Mapped physical coordinates with shape ``(num_elements, num_points, 2)``.
    values
        Callable values normalized to shape ``(num_elements, num_points)``.
    """
    if reference_points is None:
        reference_points = reference_plot_points(resolution)
    else:
        reference_points = np.ascontiguousarray(reference_points, dtype=np.float64)
    physical_points = mesh.map_reference_points(reference_points)
    values = function(physical_points[:, :, 0], physical_points[:, :, 1])
    values = _normalize_sample_values(values, mesh.num_tri, reference_points.shape[0])
    return reference_points, physical_points, values


def refined_field_polydata(
        field: DGField,
        *,
        resolution: int = 20,
        reference_points: np.ndarray | None = None,
        values: np.ndarray | None = None,
        scalar_name: str | None = None,
):
    """Build a discontinuous refined :class:`pyvista.PolyData` for a DG field.

    The geometry is refined only for visualization.  It does not change the
    field or its owning :class:`~hybridge.core.space.DGSpace`.
    """
    if reference_points is None:
        reference_points = reference_plot_points(resolution)
    else:
        reference_points = np.ascontiguousarray(reference_points, dtype=np.float64)
    if values is None:
        sampled_values = field.values_at_ref(reference_points)
        values = sampled_values
    else:
        values = _normalize_sample_values(
            values,
            field.space.mesh.num_tri,
            reference_points.shape[0],
        )

    name = field.name if scalar_name is None else str(scalar_name)
    return refined_sample_polydata(
        field.space.mesh,
        reference_points,
        values,
        scalar_name=name,
    )


def refined_sample_polydata(
        mesh: DGMesh,
        reference_points: np.ndarray,
        values: np.ndarray,
        *,
        scalar_name: str,
):
    """Build refined :class:`pyvista.PolyData` from mesh-only scalar samples.

    Unlike :func:`refined_field_polydata`, this helper never evaluates a
    :class:`~hybridge.core.space.DGField` basis.  It is therefore appropriate for exact
    reference functions whose visualization should depend only on the physical
    mesh and the requested sampling density, not on the DG polynomial order.
    """
    pv = _require_pyvista()
    reference_points = np.ascontiguousarray(reference_points, dtype=np.float64)
    values = _normalize_sample_values(values, mesh.num_tri, reference_points.shape[0])
    physical_points = mesh.map_reference_points(reference_points)
    reference_triangles = reference_plot_connectivity(reference_points)
    points_per_element = reference_points.shape[0]
    triangle_offsets = np.repeat(
        np.arange(mesh.num_tri, dtype=np.int64) * points_per_element,
        reference_triangles.shape[0],
    )
    refined_triangles = np.tile(reference_triangles, (mesh.num_tri, 1)) + triangle_offsets[:, None]
    refined_faces = np.insert(refined_triangles, 0, 3, axis=1).ravel()

    refined_points = np.zeros((mesh.num_tri * points_per_element, 3), dtype=np.float64)
    refined_points[:, :2] = physical_points.reshape(-1, 2)

    polydata = pv.PolyData(refined_points, refined_faces)
    polydata.point_data[str(scalar_name)] = np.asarray(values, dtype=np.float64).reshape(-1)
    return polydata


def coarse_mesh_polydata(mesh: DGMesh):
    """Build a wireframe-ready :class:`pyvista.PolyData` from a :class:`DGMesh`."""
    pv = _require_pyvista()
    points = np.zeros((mesh.node_coords.shape[0], 3), dtype=np.float64)
    points[:, :2] = mesh.node_coords
    faces = np.insert(mesh.triangles, 0, 3, axis=1).ravel()
    return pv.PolyData(points, faces)


def add_field_to_plotter(
        plotter,
        field: DGField,
        *,
        resolution: int = 20,
        reference_points: np.ndarray | None = None,
        values: np.ndarray | None = None,
        scalar_name: str | None = None,
        title: str | None = None,
        subplot: tuple[int, int] | None = None,
        show_mesh: bool = True,
        cmap: str = "viridis",
        clim: tuple[float, float] | None = None,
        scalar_bar_args: dict | None = None,
        show_edges: bool = False,
        mesh_color: str = "black",
        mesh_opacity: float = 0.45,
        mesh_line_width: float | None = None,
        show_grid: bool = True,
        title_position: str = "upper_edge",
        title_font_size: int = 11,
        return_actor: bool = False,
):
    """Add a sampled DG field to an existing PyVista plotter.

    This is the lowest-level plotting helper intended for custom layouts.  It
    returns the refined field mesh so callers may inspect or reuse the sampled
    scalar array. With ``return_actor=True``, return ``(mesh, actor)`` so live
    viewers can update the scalar range without recreating the actor.
    """
    if reference_points is None:
        reference_points = reference_plot_points(resolution)
    else:
        reference_points = np.ascontiguousarray(reference_points, dtype=np.float64)
    if values is None:
        values = field.values_at_ref(reference_points)
    return add_samples_to_plotter(
        plotter, field.space.mesh, reference_points, values,
        scalar_name=field.name if scalar_name is None else str(scalar_name),
        title=title, subplot=subplot, show_mesh=show_mesh, cmap=cmap, clim=clim,
        scalar_bar_args=scalar_bar_args, show_edges=show_edges,
        mesh_color=mesh_color, mesh_opacity=mesh_opacity, mesh_line_width=mesh_line_width,
        show_grid=show_grid, title_position=title_position, title_font_size=title_font_size,
        return_actor=return_actor,
    )


def add_samples_to_plotter(
        plotter,
        mesh: DGMesh,
        reference_points: np.ndarray,
        values: np.ndarray,
        *,
        scalar_name: str,
        title: str | None = None,
        subplot: tuple[int, int] | None = None,
        show_mesh: bool = True,
        cmap: str = "viridis",
        clim: tuple[float, float] | None = None,
        scalar_bar_args: dict | None = None,
        show_edges: bool = False,
        mesh_color: str = "black",
        mesh_opacity: float = 0.45,
        mesh_line_width: float | None = None,
        show_grid: bool = True,
        title_position: str = "upper_edge",
        title_font_size: int = 11,
        return_actor: bool = False,
):
    """Add mesh-only scalar samples to an existing PyVista plotter.

    This is intended for exact/reference callables.  It maps the supplied
    reference points with the mesh geometry and never touches a DG basis or a
    :class:`DGField`, so the rendered data are independent of polynomial order.
    With ``return_actor=True``, return ``(mesh, actor)`` instead of only the mesh.
    """
    if subplot is not None:
        plotter.subplot(*subplot)

    refined_mesh = refined_sample_polydata(
        mesh,
        reference_points,
        values,
        scalar_name=scalar_name,
    )
    scalar_values = refined_mesh.point_data[scalar_name]
    if clim is None:
        clim = _safe_clim(scalar_values)

    actor = plotter.add_mesh(
        refined_mesh,
        scalars=scalar_name,
        cmap=cmap,
        clim=clim,
        show_edges=show_edges,
        scalar_bar_args=scalar_bar_args,
    )
    if show_mesh:
        plotter.add_mesh(
            coarse_mesh_polydata(mesh),
            style="wireframe",
            color=mesh_color,
            line_width=_mesh_overlay_line_width(mesh) if mesh_line_width is None else float(mesh_line_width),
            opacity=mesh_opacity,
        )
    if title:
        plotter.add_text(title, position=title_position, font_size=title_font_size, shadow=False)
    plotter.enable_parallel_projection()
    plotter.view_xy()
    if show_grid:
        plotter.show_grid(color=(100, 100, 100, 0.15))
    return (refined_mesh, actor) if return_actor else refined_mesh


def plot_field(
        field: DGField,
        *,
        resolution: int = 20,
        title: str | None = None,
        show_mesh: bool = True,
        show: bool = True,
        off_screen: bool = False,
        window_size: tuple[int, int] = (900, 700),
        cmap: str = "viridis",
        clim: tuple[float, float] | None = None,
        scalar_name: str | None = None,
):
    """Plot one scalar DG field and return the PyVista plotter."""
    pv = _require_pyvista()
    plotter = pv.Plotter(window_size=list(window_size), off_screen=off_screen)
    scalar_bar_args = {
        "vertical": False,
        "width": 0.55,
        "height": 0.08,
        "position_x": 0.225,
        "position_y": 0.02,
    }
    add_field_to_plotter(
        plotter,
        field,
        resolution=resolution,
        title=field.name if title is None else title,
        show_mesh=show_mesh,
        cmap=cmap,
        clim=clim,
        scalar_name=scalar_name,
        scalar_bar_args=scalar_bar_args,
    )
    if show:
        plotter.show()
    return plotter


def plot_fields(
        fields: Sequence[DGField],
        *,
        resolution: int = 20,
        titles: Sequence[str] | None = None,
        shape: tuple[int, int] | None = None,
        show_mesh: bool = True,
        show: bool = True,
        off_screen: bool = False,
        window_size: tuple[int, int] = (1600, 700),
        cmap: str = "viridis",
        share_clim: bool = False,
):
    """Plot several scalar DG fields in one PyVista window.

    ``share_clim=True`` uses one color range across all panels, which is useful
    for comparing fields in the same units.
    """
    field_tuple = tuple(fields)
    if not field_tuple:
        raise ValueError("at least one DG field is required")
    if titles is None:
        titles = tuple(field.name for field in field_tuple)
    else:
        titles = tuple(titles)
    if len(titles) != len(field_tuple):
        raise ValueError("titles must have the same length as fields")

    if shape is None:
        shape = (1, len(field_tuple))
    rows, columns = shape
    if rows * columns < len(field_tuple):
        raise ValueError("shape does not have enough panels for all fields")

    reference_points = reference_plot_points(resolution)
    shared_clim = None
    if share_clim:
        shared_values = np.concatenate([field.values_at_ref(reference_points).reshape(-1) for field in field_tuple])
        shared_clim = _robust_clim(shared_values)

    pv = _require_pyvista()
    plotter = pv.Plotter(shape=shape, window_size=list(window_size), off_screen=off_screen)
    scalar_bar_args = {
        "vertical": False,
        "width": 0.55,
        "height": 0.08,
        "position_x": 0.225,
        "position_y": 0.02,
    }
    for index, (field, title) in enumerate(zip(field_tuple, titles)):
        add_field_to_plotter(
            plotter,
            field,
            resolution=resolution,
            reference_points=reference_points,
            title=title,
            subplot=(index // columns, index % columns),
            show_mesh=show_mesh,
            cmap=cmap,
            clim=shared_clim,
            scalar_bar_args=scalar_bar_args,
        )
    if len(field_tuple) > 1:
        plotter.link_views()
    if show:
        plotter.show()
    return plotter


def matplotlib_discontinuous_triangulation(mesh: DGMesh, reference_points: np.ndarray):
    """Build a Matplotlib triangulation with duplicated vertices per DG element.

    The returned triangulation is suitable for DG visualizations because every
    physical element owns its own copy of the refined reference grid.  Neighboring
    elements therefore do not share Matplotlib vertices and discontinuous values
    are not interpolated across element boundaries.
    """
    import matplotlib.tri as mtri

    reference_points = np.ascontiguousarray(reference_points, dtype=np.float64)
    physical_points = mesh.map_reference_points(reference_points)
    points_per_element = reference_points.shape[0]
    reference_triangles = reference_plot_connectivity(reference_points)
    triangle_offsets = np.repeat(
        np.arange(mesh.num_tri, dtype=np.int64) * points_per_element,
        reference_triangles.shape[0],
    )
    triangles = np.tile(reference_triangles, (mesh.num_tri, 1)) + triangle_offsets[:, None]
    points = physical_points.reshape(-1, 2)
    return mtri.Triangulation(points[:, 0], points[:, 1], triangles)


def add_matplotlib_mesh(ax, mesh: DGMesh, *, color: str = "black", linewidth: float = 0.65,
                        alpha: float = 0.55, bounds=None):
    """Overlay the physical mesh, optionally selecting a rectangular close-up.

    The array-only implementation is in hybridge.io.figures.add_matplotlib_mesh.
    """
    from hybridge.io.figures import add_matplotlib_mesh as overlay
    return overlay(ax, mesh, color=color, linewidth=linewidth, alpha=alpha, bounds=bounds)


def _matplotlib_backend_is_noninteractive(backend: str) -> bool:
    """Return True when the backend cannot open interactive windows."""
    backend = str(backend).lower()
    return (
        backend in {"agg", "pdf", "ps", "svg", "template", "cairo"}
        or backend.endswith("agg")
        and not backend.startswith(("qt", "tk", "gtk", "wx", "macosx"))
        or "inline" in backend
    )


def _matplotlib_pyplot(*, show: bool):
    """Import pyplot, switching away from non-interactive backends when showing."""
    import matplotlib

    if (
        show
        and _matplotlib_backend_is_noninteractive(matplotlib.get_backend())
        and "MPLBACKEND" not in os.environ
    ):
        errors = []
        for backend in ("QtAgg", "TkAgg", "GTK3Agg", "WXAgg", "MacOSX"):
            try:
                matplotlib.use(backend, force=True)
                import matplotlib.pyplot as plt

                return plt
            except Exception as exc:  # pragma: no cover - backend availability is environment-specific.
                errors.append(f"{backend}: {exc}")
        message = "\n".join(errors)
        raise RuntimeError(
            "Matplotlib is using a non-interactive backend and no interactive backend could be activated. "
            "Install PyQt/PySide or Tk support, or run with an interactive backend such as "
            "`MPLBACKEND=QtAgg python scripts/diffusion_reaction/run_cases.py ... --plot`.\n"
            f"Tried backends:\n{message}"
        )

    import matplotlib.pyplot as plt

    return plt


def plot_scalar_sample_panels_matplotlib(
        mesh: DGMesh,
        panels: Sequence[tuple],
        *,
        suptitle: str | None = None,
        show_mesh: bool = True,
        cmap: str = "jet",
        levels: int | Sequence[float] = 64,
        clim: tuple[float, float] | None = None,
        share_clim: bool = True,
        show: bool = True,
        figsize: tuple[float, float] | None = None,
):
    """Plot scalar per-element samples using Matplotlib discontinuous contours.

    Each panel is ``(title, reference_points, values)`` or
    ``(title, reference_points, values, options)``.  ``values`` may be a scalar,
    one value per reference point, or an array with shape
    ``(num_elements, num_points)``.  Per-panel ``options`` may set ``cmap``,
    ``levels``, ``clim``, ``extend``, ``show_mesh``, ``robust_percentile``,
    and ``zero_min``.  Vertices are duplicated per element so discontinuous
    DG fields are not averaged across element boundaries.

    Parameters
    ----------
    mesh
        Mesh used to map all supplied reference-point grids.
    panels
        Sequence of panel tuples.  Panels may use different reference grids,
        which is useful when exact/reference data should be sampled more densely
        than polynomial DG fields.
    clim
        Optional shared color limits.  When provided with integer ``levels`` and
        ``share_clim=True``, the contour levels span exactly this interval.
        Automatic limits use the central 95 percent of finite values by
        default.  Error panels can set ``zero_min=True`` to use a zero lower
        limit and a percentile-based upper limit.
    show
        If true, call :func:`matplotlib.pyplot.show`.  The active Matplotlib
        backend must be interactive for a window to appear.
    """
    plt = _matplotlib_pyplot(show=show)

    panel_tuple = tuple(panels)
    if not panel_tuple:
        raise ValueError("at least one panel is required")
    if figsize is None:
        figsize = (5.0 * len(panel_tuple), 4.8)

    normalized_panels = []
    for panel in panel_tuple:
        if len(panel) == 3:
            title, reference_points, values = panel
            panel_options = {}
        elif len(panel) == 4:
            title, reference_points, values, panel_options = panel
            panel_options = {} if panel_options is None else dict(panel_options)
        else:
            raise ValueError(
                "each panel must be (title, reference_points, values) "
                "or include an options dict"
            )
        reference_points = np.ascontiguousarray(reference_points, dtype=np.float64)
        normalized_values = _normalize_sample_values(values, mesh.num_tri, reference_points.shape[0])
        normalized_panels.append((title, reference_points, normalized_values, panel_options))

    colorbar_option_keys = {
        "cmap",
        "levels",
        "clim",
        "extend",
        "robust_percentile",
        "zero_min",
    }
    use_shared_colorbar = bool(share_clim) and not any(
        colorbar_option_keys.intersection(options) for _, _, _, options in normalized_panels
    )

    def _resolve_clim(
            values: np.ndarray,
            requested_clim=None,
            *,
            percentile: float = 95.0,
            zero_min: bool = False,
    ) -> tuple[float, float]:
        """Resolve explicit or data-derived color limits for a plot field."""
        if requested_clim is None:
            return _robust_clim(values, percentile=percentile, zero_min=zero_min)
        minimum, maximum = float(requested_clim[0]), float(requested_clim[1])
        return _expand_clim(minimum, maximum)

    shared_range = None
    if share_clim:
        if clim is None:
            shared_values = np.concatenate(
                [values.reshape(-1) for _, _, values, _ in normalized_panels]
            )
            shared_range = _resolve_clim(shared_values)
        else:
            shared_range = _resolve_clim(np.array([0.0]), clim)

    fig, axes = plt.subplots(1, len(normalized_panels), figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes)
    contour = None
    panel_contours = []
    triangulations: dict[tuple[tuple[int, ...], bytes], object] = {}
    for ax, (title, reference_points, values, panel_options) in zip(axes, normalized_panels):
        key = (reference_points.shape, reference_points.tobytes())
        triangulation = triangulations.get(key)
        if triangulation is None:
            triangulation = matplotlib_discontinuous_triangulation(mesh, reference_points)
            triangulations[key] = triangulation
        panel_levels = panel_options.get("levels", levels)
        panel_percentile = float(panel_options.get("robust_percentile", 95.0))
        panel_zero_min = bool(panel_options.get("zero_min", False))
        if "clim" in panel_options:
            panel_minimum, panel_maximum = _resolve_clim(
                values,
                panel_options.get("clim"),
                percentile=panel_percentile,
                zero_min=panel_zero_min,
            )
        elif shared_range is not None:
            panel_minimum, panel_maximum = shared_range
        else:
            panel_minimum, panel_maximum = _resolve_clim(
                values,
                percentile=panel_percentile,
                zero_min=panel_zero_min,
            )
        if isinstance(panel_levels, int):
            panel_levels = np.linspace(panel_minimum, panel_maximum, int(panel_levels))
        contour = ax.tricontourf(
            triangulation,
            values.reshape(-1),
            levels=panel_levels,
            cmap=panel_options.get("cmap", cmap),
            extend=panel_options.get("extend", "both"),
            vmin=panel_minimum,
            vmax=panel_maximum,
        )
        contour.set_clim(panel_minimum, panel_maximum)
        panel_contours.append((ax, contour))
        if panel_options.get("show_mesh", show_mesh):
            add_matplotlib_mesh(ax, mesh)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    if use_shared_colorbar and contour is not None:
        fig.colorbar(
            contour,
            ax=axes.ravel().tolist(),
            shrink=0.76,
            fraction=0.045,
            pad=0.035,
            location="right",
        )
    elif not use_shared_colorbar:
        for ax, panel_contour in panel_contours:
            fig.colorbar(
                panel_contour,
                ax=ax,
                shrink=0.72,
                fraction=0.045,
                pad=0.035,
                location="right",
            )
    if show and not _matplotlib_backend_is_noninteractive(plt.get_backend()):
        plt.show()
    return fig


def plot_scalar_raster_panels_matplotlib(
        panels: Sequence[tuple],
        bounds: tuple[float, float, float, float],
        *,
        suptitle: str | None = None,
        cmap: str = "viridis",
        clim: tuple[float, float] | None = None,
        share_clim: bool = True,
        symmetric: bool = False,
        robust_percentile: float = 95.0,
        show: bool = True,
        figsize: tuple[float, float] | None = None,
):
    """Plot scalar rasters with physical extents and optional shared color limits.

    Each panel is ``(title, values)`` or ``(title, values, options)`` with a
    nonempty 2D host array. NaNs, infinities, and masked pixels remain masked,
    preserving mesh holes. Row zero is at the top, matching ``RasterGeometry``;
    no interpolation is applied across pixels or discontinuities. ``bounds``
    is ``(xmin, xmax, ymin, ymax)`` as returned by that geometry.

    Per-panel options may override ``cmap``, ``clim``, ``symmetric``, and
    ``robust_percentile``. With ``share_clim=True``, panels use shared limits
    unless overridden. A single colorbar is used only when no panel overrides
    color options; otherwise each panel gets a colorbar.
    Returns the figure for callers to save or further annotate.
    """
    bounds = tuple(float(v) for v in bounds)
    if (len(bounds) != 4 or not np.isfinite(bounds).all()
            or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]):
        raise ValueError("bounds must be finite (xmin, xmax, ymin, ymax) with positive spans")
    normalized = []
    for panel in panels:
        if len(panel) not in (2, 3):
            raise ValueError("each panel must be (title, values) or include an options dict")
        title, values = panel[:2]
        options = dict(panel[2] or {}) if len(panel) == 3 else {}
        values = np.ma.masked_invalid(np.ma.asarray(values, dtype=np.float64))
        if values.ndim != 2 or not values.size:
            raise ValueError("raster values must be a nonempty two-dimensional array")
        normalized.append((title, values, options))
    if not normalized:
        raise ValueError("at least one panel is required")
    shared_range = clim
    if share_clim and shared_range is None:
        shared_range = scalar_color_limits(
            np.concatenate([values.compressed() for _, values, _ in normalized]),
            percentile=robust_percentile, symmetric=symmetric,
        )
    color_options = {"cmap", "clim", "symmetric", "robust_percentile"}
    shared_colorbar = share_clim and not any(color_options.intersection(o) for _, _, o in normalized)
    plt = _matplotlib_pyplot(show=show)
    fig, axes = plt.subplots(
        1, len(normalized), squeeze=False,
        figsize=figsize or (5.0 * len(normalized), 4.8), constrained_layout=True,
    )
    for ax, (title, values, options) in zip(axes.flat, normalized):
        limits = options.get("clim", shared_range)
        if limits is None or (
            options.get("clim") is None and {"symmetric", "robust_percentile"}.intersection(options)
        ):
            limits = scalar_color_limits(
                values.compressed(), percentile=options.get("robust_percentile", robust_percentile),
                symmetric=options.get("symmetric", symmetric),
            )
        limits = _expand_clim(float(limits[0]), float(limits[1]))
        image = ax.imshow(
            values, extent=bounds, origin="upper", interpolation="nearest",
            cmap=options.get("cmap", cmap), vmin=limits[0], vmax=limits[1],
        )
        ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
        if not shared_colorbar:
            fig.colorbar(image, ax=ax, shrink=0.7)
    if shared_colorbar:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7)
    if suptitle:
        fig.suptitle(suptitle)
    if show and not _matplotlib_backend_is_noninteractive(plt.get_backend()):
        plt.show()
    return fig


def _plot_postprocessed_comparison(
        field: DGField,
        postprocessed: DGField,
        exact_solution: Callable,
        *,
        resolution: int,
        exact_resolution: int | str | None,
        title: str,
        show_mesh: bool,
        show: bool,
        off_screen: bool,
        window_size: tuple[int, int],
        use_matplotlib: bool,
):
    """Plot the HDG solution, its postprocessed field and the exact solution on one scale."""
    mesh = field.space.mesh
    post_resolution = resolve_postprocessed_plot_resolution(
        resolution, order=field.space.order, num_elements=mesh.num_tri,
    )
    numerical_points, _, numerical_values = sample_field_on_elements(field, resolution=resolution)
    post_points, _, post_values = sample_field_on_elements(postprocessed, resolution=post_resolution)
    exact_points, _, exact_values = sample_callable_on_elements(
        mesh,
        exact_solution,
        resolution=_resolve_exact_plot_resolution(
            exact_resolution, numerical_resolution=post_resolution, num_elements=mesh.num_tri,
        ),
    )
    titles = (
        f"HDG solution, p = {field.space.order}\nL2 error {field.l2_error(exact_solution):.1e}",
        f"Postprocessed, p = {postprocessed.space.order}\n"
        f"L2 error {postprocessed.l2_error(exact_solution):.1e}",
        "Exact solution",
    )
    if use_matplotlib:
        return plot_scalar_sample_panels_matplotlib(
            mesh,
            (
                (titles[0], numerical_points, numerical_values),
                (titles[1], post_points, post_values),
                (titles[2], exact_points, exact_values, {"show_mesh": False}),
            ),
            suptitle=title or None,
            show_mesh=show_mesh,
            cmap="viridis",
            levels=contour_levels_for_order(postprocessed.space.order),
            share_clim=True,
            show=show,
        )

    pv = _require_pyvista()
    clim = _robust_clim(np.concatenate(
        (numerical_values.reshape(-1), post_values.reshape(-1), exact_values.reshape(-1))
    ))
    plotter = pv.Plotter(shape=(1, 3), window_size=list(window_size), off_screen=off_screen)
    scalar_bar_args = {
        "vertical": False,
        "width": 0.55,
        "height": 0.08,
        "position_x": 0.225,
        "position_y": 0.02,
    }
    panels = ((field, numerical_points, numerical_values), (postprocessed, post_points, post_values))
    for column, (panel_field, panel_points, values) in enumerate(panels):
        add_field_to_plotter(
            plotter,
            panel_field,
            reference_points=panel_points,
            values=values,
            scalar_name=f"field_{column}",
            title=titles[column] if column or not title else f"{titles[column]}\n{title}",
            subplot=(0, column),
            show_mesh=show_mesh,
            cmap="viridis",
            clim=clim,
            scalar_bar_args=scalar_bar_args,
        )
    add_samples_to_plotter(
        plotter,
        mesh,
        exact_points,
        exact_values,
        scalar_name="field_2",
        title=titles[2],
        subplot=(0, 2),
        show_mesh=False,
        cmap="viridis",
        clim=clim,
        scalar_bar_args=scalar_bar_args,
    )
    plotter.link_views()
    if show:
        plotter.show()
    return plotter


def plot_solution_comparison(
        field: DGField,
        exact_solution: Callable,
        *,
        postprocessed: DGField | None = None,
        resolution: int = 20,
        exact_resolution: int | str | None = None,
        title: str = "",
        show_mesh: bool = True,
        show: bool = True,
        off_screen: bool = False,
        window_size: tuple[int, int] = (1800, 650),
        backend: str = "auto",
):
    """Plot numerical solution, exact solution, and absolute error.

    This is the solver-oriented helper used by :mod:`hybridge.solvers.advection_reaction`.  For a
    generic single-field plot use :func:`plot_field`.

    ``resolution`` controls the numerical and error panels.  ``exact_resolution``
    controls only the exact reference panel; use ``"auto"`` for a denser exact
    sampling that is independent of the DG polynomial order.  ``None`` preserves the historical
    behavior and samples the exact panel on the same grid as the numerical panel.

    With ``postprocessed`` (for example ``result.postprocessed_field`` from
    ``hdg_postprocess="primal"``), the panels are the HDG solution, the
    postprocessed field and the exact solution, on one color scale, each
    numerical panel titled with its L2 error.

    ``backend`` is ``"matplotlib"``, ``"pyvista"``, or ``"auto"`` (Matplotlib
    for meshes of at most 130 triangles, PyVista otherwise).
    """
    if backend not in ("auto", "matplotlib", "pyvista"):
        raise ValueError("backend must be 'auto', 'matplotlib' or 'pyvista'")
    use_matplotlib = backend == "matplotlib" or (backend == "auto" and field.space.mesh.num_tri <= 130)
    if postprocessed is not None:
        return _plot_postprocessed_comparison(
            field,
            postprocessed,
            exact_solution,
            resolution=resolution,
            exact_resolution=exact_resolution,
            title=title,
            show_mesh=show_mesh,
            show=show,
            off_screen=off_screen,
            window_size=window_size,
            use_matplotlib=use_matplotlib,
        )
    reference_points, physical_points, numerical_values = sample_field_on_elements(
        field,
        resolution=resolution,
    )
    exact_values_for_error = exact_solution(physical_points[:, :, 0], physical_points[:, :, 1])
    exact_values_for_error = _normalize_sample_values(
        exact_values_for_error,
        field.space.mesh.num_tri,
        reference_points.shape[0],
    )
    absolute_error = np.abs(numerical_values - exact_values_for_error)

    exact_panel_resolution = _resolve_exact_plot_resolution(
        exact_resolution,
        numerical_resolution=resolution,
        num_elements=field.space.mesh.num_tri,
    )
    exact_reference_points, _, exact_display_values = sample_callable_on_elements(
        field.space.mesh,
        exact_solution,
        resolution=exact_panel_resolution,
    )
    if use_matplotlib:
        return plot_scalar_sample_panels_matplotlib(
            field.space.mesh,
            (
                ("Numerical solution", reference_points, numerical_values),
                ("Exact solution", exact_reference_points, exact_display_values, {"show_mesh": False}),
                (
                    "Absolute error",
                    reference_points,
                    absolute_error,
                    {"cmap": "magma", "zero_min": True},
                ),
            ),
            suptitle=title or None,
            show_mesh=show_mesh,
            cmap="jet",
            levels=contour_levels_for_order(field.space.order),
            share_clim=False,
            show=show,
        )

    pv = _require_pyvista()
    field_clim = _robust_clim(np.concatenate((numerical_values.reshape(-1), exact_display_values.reshape(-1))))
    error_clim = _robust_clim(absolute_error, zero_min=True)

    plotter = pv.Plotter(shape=(1, 3), window_size=list(window_size), off_screen=off_screen)
    scalar_bar_args = {
        "vertical": False,
        "width": 0.55,
        "height": 0.08,
        "position_x": 0.225,
        "position_y": 0.02,
    }
    panels = (
        ("Numerical solution", reference_points, numerical_values, field_clim, "viridis"),
        ("Exact solution", exact_reference_points, exact_display_values, field_clim, "viridis"),
        ("Absolute error", reference_points, absolute_error, error_clim, "magma"),
    )
    for column, (panel_title, panel_reference_points, values, clim, cmap) in enumerate(panels):
        display_title = panel_title if column != 0 or not title else f"{panel_title}\n{title}"
        if column == 1:
            add_samples_to_plotter(
                plotter,
                field.space.mesh,
                panel_reference_points,
                values,
                scalar_name=f"field_{column}",
                title=display_title,
                subplot=(0, column),
                show_mesh=False,
                cmap=cmap,
                clim=clim,
                scalar_bar_args=scalar_bar_args,
            )
        else:
            add_field_to_plotter(
                plotter,
                field,
                reference_points=panel_reference_points,
                values=values,
                scalar_name=f"field_{column}",
                title=display_title,
                subplot=(0, column),
                show_mesh=show_mesh,
                cmap=cmap,
                clim=clim,
                scalar_bar_args=scalar_bar_args,
            )
    plotter.link_views()
    if show:
        plotter.show()
    return plotter


__all__ = [
    "add_field_to_plotter",
    "add_matplotlib_mesh",
    "add_samples_to_plotter",
    "coarse_mesh_polydata",
    "contour_levels_for_order",
    "matplotlib_discontinuous_triangulation",
    "plot_field",
    "plot_fields",
    "plot_scalar_sample_panels_matplotlib",
    "plot_solution_comparison",
    "reference_plot_connectivity",
    "resolve_exact_plot_resolution",
    "resolve_field_plot_resolution",
    "resolve_postprocessed_plot_resolution",
    "refined_field_polydata",
    "refined_sample_polydata",
    "sample_callable_on_elements",
    "sample_field_on_elements",
]
