"""PyVista plotting helpers for :mod:`dgfem` fields.

The helpers in this module sample each DG element independently.  Vertices on
shared mesh edges are intentionally duplicated so discontinuities remain
visible instead of being averaged by the rendering backend.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.spatial import Delaunay

from .mesh import DGMesh
from .space import DGField

def _require_pyvista():
    """Import PyVista lazily so non-plotting code has no plotting dependency."""
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("dgfem plotting helpers require pyvista") from exc
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


def _safe_clim(values: np.ndarray) -> tuple[float, float]:
    """Return a finite color range, expanding constants for PyVista."""
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        return 0.0, 1.0
    if minimum == maximum:
        return minimum, minimum + 1.0
    return minimum, maximum


def reference_plot_points(resolution: int) -> np.ndarray:
    r"""Return a uniform plotting grid on :math:`\hat K`.

    The reference triangle is
    :math:`\hat K=\{(r,s)\in[-1,1]^2:\ s\le -r\}`.  The returned array has
    shape ``(num_plot_points, 2)`` and is C-contiguous.
    """
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    axis = np.linspace(-1.0, 1.0, int(resolution))
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    inside = yy <= -xx
    return np.ascontiguousarray(np.column_stack((xx[inside], yy[inside])), dtype=np.float64)


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
    field or its owning :class:`~dgfem.space.DGSpace`.
    """
    pv = _require_pyvista()
    reference_points, physical_points, sampled_values = sample_field_on_elements(
        field,
        resolution=resolution,
        reference_points=reference_points,
    )
    if values is None:
        values = sampled_values
    else:
        values = _normalize_sample_values(
            values,
            field.space.mesh.num_tri,
            reference_points.shape[0],
        )

    reference_triangles = reference_plot_connectivity(reference_points)
    points_per_element = reference_points.shape[0]
    triangle_offsets = np.repeat(
        np.arange(field.space.mesh.num_tri, dtype=np.int64) * points_per_element,
        reference_triangles.shape[0],
    )
    refined_triangles = np.tile(reference_triangles, (field.space.mesh.num_tri, 1)) + triangle_offsets[:, None]
    refined_faces = np.insert(refined_triangles, 0, 3, axis=1).ravel()

    refined_points = np.zeros((field.space.mesh.num_tri * points_per_element, 3), dtype=np.float64)
    refined_points[:, :2] = physical_points.reshape(-1, 2)

    name = field.name if scalar_name is None else str(scalar_name)
    mesh = pv.PolyData(refined_points, refined_faces)
    mesh.point_data[name] = np.asarray(values, dtype=np.float64).reshape(-1)
    return mesh


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
):
    """Add a sampled DG field to an existing PyVista plotter.

    This is the lowest-level plotting helper intended for custom layouts.  It
    returns the refined field mesh so callers may inspect or reuse the sampled
    scalar array.
    """
    if subplot is not None:
        plotter.subplot(*subplot)

    scalar = field.name if scalar_name is None else str(scalar_name)
    refined_mesh = refined_field_polydata(
        field,
        resolution=resolution,
        reference_points=reference_points,
        values=values,
        scalar_name=scalar,
    )
    scalar_values = refined_mesh.point_data[scalar]
    if clim is None:
        clim = _safe_clim(scalar_values)

    plotter.add_mesh(
        refined_mesh,
        scalars=scalar,
        cmap=cmap,
        clim=clim,
        show_edges=show_edges,
        scalar_bar_args=scalar_bar_args,
    )
    if show_mesh:
        plotter.add_mesh(
            coarse_mesh_polydata(field.space.mesh),
            style="wireframe",
            color=mesh_color,
            line_width=1.0,
            opacity=mesh_opacity,
        )
    if title:
        plotter.add_text(title, position="upper_edge", font_size=11, shadow=False)
    plotter.enable_parallel_projection()
    plotter.view_xy()
    plotter.show_grid(color=(100, 100, 100, 0.15))
    return refined_mesh


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
        limits = [_safe_clim(field.values_at_ref(reference_points)) for field in field_tuple]
        shared_clim = min(lo for lo, _ in limits), max(hi for _, hi in limits)

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


def plot_solution_comparison(
        field: DGField,
        exact_solution: Callable,
        *,
        resolution: int = 20,
        title: str = "",
        show_mesh: bool = True,
        show: bool = True,
        off_screen: bool = False,
        window_size: tuple[int, int] = (1800, 650),
):
    """Plot numerical solution, exact solution, and absolute error.

    This is the solver-oriented helper used by :mod:`dgfem.adv_rea`.  For a
    generic single-field plot use :func:`plot_field`.
    """
    pv = _require_pyvista()
    reference_points, physical_points, numerical_values = sample_field_on_elements(
        field,
        resolution=resolution,
    )
    exact_values = exact_solution(physical_points[:, :, 0], physical_points[:, :, 1])
    exact_values = _normalize_sample_values(
        exact_values,
        field.space.mesh.num_tri,
        reference_points.shape[0],
    )
    absolute_error = np.abs(numerical_values - exact_values)

    field_min = float(min(np.min(numerical_values), np.min(exact_values)))
    field_max = float(max(np.max(numerical_values), np.max(exact_values)))
    if field_min == field_max:
        field_max = field_min + 1.0

    plotter = pv.Plotter(shape=(1, 3), window_size=list(window_size), off_screen=off_screen)
    scalar_bar_args = {
        "vertical": False,
        "width": 0.55,
        "height": 0.08,
        "position_x": 0.225,
        "position_y": 0.02,
    }
    panels = (
        ("Numerical solution", numerical_values, (field_min, field_max), "viridis"),
        ("Exact solution", exact_values, (field_min, field_max), "viridis"),
        ("Absolute error", absolute_error, None, "magma"),
    )
    for column, (panel_title, values, clim, cmap) in enumerate(panels):
        display_title = panel_title if column != 0 or not title else f"{panel_title}\n{title}"
        add_field_to_plotter(
            plotter,
            field,
            reference_points=reference_points,
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
    "coarse_mesh_polydata",
    "plot_field",
    "plot_fields",
    "plot_solution_comparison",
    "reference_plot_connectivity",
    "reference_plot_points",
    "refined_field_polydata",
    "sample_field_on_elements",
]
