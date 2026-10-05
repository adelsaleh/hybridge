#!/usr/bin/env python3
"""Render high-order 3-D PyVista surfaces of ``T`` and ``phi_T``.

The two homogeneous-Dirichlet Poisson problems are

    -Delta T     = 1,
    -Delta phi_T = 1_{c1 < T < c2},

where ``c1 = alpha1*max(T)`` and ``c2 = alpha2*max(T)``.  The defaults
``alpha1=0.6`` and ``alpha2=0.7`` match the torsion window used by
``dolfinx_torsion_initialized_window_reduced_optimization.py``.

The selectable scalar panels are ``T``, ``rho_T``, and ``phi_T``.  When the
density panel is enabled, ``rho_T`` and the right-hand side for ``phi_T`` use
a mollified logistic window; otherwise the crisp indicator is retained.  The
available geometries are the convex polygon, five-node star, narrow-back
Pac-Man, the repository's ITER wall, the smooth non-convex domain with a
circular hole, and the smooth horseshoe used by
``plot_poisson_torsion_geometries.py``.

By default, a preliminary torsion solve identifies cells intersecting the
threshold band.  The bulk mesh starts coarser than the requested band size;
band cells are then locally refined before the final solves.  The discontinuous
indicator is also integrated with a higher-order quadrature rule there.
``--show-mesh`` displays this band-focused mesh in a separate window.
"""

from __future__ import annotations

# Support direct execution as well as python -m from a checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

import argparse
import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hdgfem_torsion_pyvista_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/hdgfem_torsion_pyvista_cache")

import gmsh
import numpy as np
import pyvista as pv
import ufl
from dolfinx import fem, mesh as dmesh, plot
from mpi4py import MPI


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "projects/diocotron/runs" / "torsion_pyvista"
DEFAULT_MESH_SIZES = {
    "convex-polygon": 0.030,
    "pacman": 0.020,
    "star": 0.035,
    "iter": 0.070,
    "nonconvex-hole": 0.035,
    "horseshoe": 0.020,
}
PACMAN_RADIUS = 1.0
PACMAN_MOUTH_HALF_ANGLE = 0.38
PACMAN_TIP_X = -0.78
PACMAN_TIP_TO_BACK_DISTANCE = PACMAN_RADIUS + PACMAN_TIP_X


def _add_polygon(points: np.ndarray, mesh_size: float) -> list[int]:
    point_tags = [
        gmsh.model.geo.addPoint(float(x), float(y), 0.0, mesh_size)
        for x, y in points
    ]
    return [
        gmsh.model.geo.addLine(point_tags[i], point_tags[(i + 1) % len(point_tags)])
        for i in range(len(point_tags))
    ]


def _add_smooth_loop(points: np.ndarray, mesh_size: float) -> int:
    point_tags = [
        gmsh.model.geo.addPoint(float(x), float(y), 0.0, mesh_size)
        for x, y in points
    ]
    return gmsh.model.geo.addSpline(point_tags + [point_tags[0]])


def _add_circle_loop(cx: float, cy: float, radius: float, mesh_size: float) -> list[int]:
    center = gmsh.model.geo.addPoint(cx, cy, 0.0, mesh_size)
    angles = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
    points = [
        gmsh.model.geo.addPoint(
            cx + radius * math.cos(angle),
            cy + radius * math.sin(angle),
            0.0,
            mesh_size,
        )
        for angle in angles
    ]
    return [
        gmsh.model.geo.addCircleArc(points[i], center, points[(i + 1) % 4])
        for i in range(4)
    ]


def generate_mesh(geometry_name: str, mesh_size: float, path: Path, verbosity: int) -> None:
    """Generate a supported figure geometry as a linear triangle mesh."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", verbosity)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.45 * mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

        if geometry_name == "iter":
            geo_path = REPO_ROOT / "iter.geo"
            if not geo_path.exists():
                raise FileNotFoundError(f"ITER geometry not found: {geo_path}")
            gmsh.open(str(geo_path))
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
        else:
            gmsh.model.add(geometry_name)
            if geometry_name == "pacman":
                center = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, mesh_size)
                upper = gmsh.model.geo.addPoint(
                    PACMAN_RADIUS * math.cos(PACMAN_MOUTH_HALF_ANGLE),
                    PACMAN_RADIUS * math.sin(PACMAN_MOUTH_HALF_ANGLE),
                    0.0,
                    mesh_size,
                )
                back = gmsh.model.geo.addPoint(-PACMAN_RADIUS, 0.0, 0.0, mesh_size)
                lower = gmsh.model.geo.addPoint(
                    PACMAN_RADIUS * math.cos(PACMAN_MOUTH_HALF_ANGLE),
                    -PACMAN_RADIUS * math.sin(PACMAN_MOUTH_HALF_ANGLE),
                    0.0,
                    mesh_size,
                )
                tip = gmsh.model.geo.addPoint(PACMAN_TIP_X, 0.0, 0.0, mesh_size)
                upper_arc = gmsh.model.geo.addCircleArc(upper, center, back)
                lower_arc = gmsh.model.geo.addCircleArc(back, center, lower)
                lower_mouth = gmsh.model.geo.addLine(lower, tip)
                upper_mouth = gmsh.model.geo.addLine(tip, upper)
                boundary = [upper_arc, lower_arc, lower_mouth, upper_mouth]
                outer_loop = gmsh.model.geo.addCurveLoop(boundary)
                surface = gmsh.model.geo.addPlaneSurface([outer_loop])
                physical_boundary = boundary
            elif geometry_name == "convex-polygon":
                theta = np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False) + 0.10
                radii = np.array(
                    [1.04, 0.94, 1.10, 0.98, 1.06, 0.96, 1.08, 0.92, 1.02]
                )
                points = np.column_stack(
                    (radii * np.cos(theta), 0.82 * radii * np.sin(theta))
                )
                boundary = _add_polygon(points, mesh_size)
                outer_loop = gmsh.model.geo.addCurveLoop(boundary)
                surface = gmsh.model.geo.addPlaneSurface([outer_loop])
                physical_boundary = boundary
            elif geometry_name == "star":
                theta = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False) + 0.5 * np.pi
                radii = np.empty(10)
                radii[0::2] = 1.06
                radii[1::2] = 0.44
                points = np.column_stack((radii * np.cos(theta), radii * np.sin(theta)))
                boundary = _add_polygon(points, mesh_size)
                outer_loop = gmsh.model.geo.addCurveLoop(boundary)
                surface = gmsh.model.geo.addPlaneSurface([outer_loop])
                physical_boundary = boundary
            elif geometry_name == "nonconvex-hole":
                theta = np.linspace(0.0, 2.0 * np.pi, 220, endpoint=False)
                radius = 1.0 + 0.34 * np.cos(3.0 * theta) - 0.08 * np.sin(2.0 * theta)
                outer_points = np.column_stack(
                    (radius * np.cos(theta), 0.86 * radius * np.sin(theta))
                )
                outer_spline = _add_smooth_loop(outer_points, mesh_size)
                hole_arcs = _add_circle_loop(0.20, 0.03, 0.22, 0.75 * mesh_size)
                outer_loop = gmsh.model.geo.addCurveLoop([outer_spline])
                hole_loop = gmsh.model.geo.addCurveLoop(hole_arcs)
                surface = gmsh.model.geo.addPlaneSurface([outer_loop, hole_loop])
                physical_boundary = [outer_spline, *hole_arcs]
            elif geometry_name == "horseshoe":
                gap_half_angle = 0.48
                outer_radius = 1.16
                inner_radius = 0.46
                outer_theta = np.linspace(
                    gap_half_angle,
                    2.0 * np.pi - gap_half_angle,
                    170,
                )
                inner_theta = np.linspace(
                    2.0 * np.pi - gap_half_angle,
                    gap_half_angle,
                    130,
                )
                cap_steps = 28

                outer = np.column_stack(
                    (outer_radius * np.cos(outer_theta), outer_radius * np.sin(outer_theta))
                )
                lower_cap_r = np.linspace(outer_radius, inner_radius, cap_steps)
                lower_cap = np.column_stack(
                    (
                        lower_cap_r * np.cos(2.0 * np.pi - gap_half_angle),
                        lower_cap_r * np.sin(2.0 * np.pi - gap_half_angle),
                    )
                )
                inner = np.column_stack(
                    (inner_radius * np.cos(inner_theta), inner_radius * np.sin(inner_theta))
                )
                upper_cap_r = np.linspace(inner_radius, outer_radius, cap_steps)
                upper_cap = np.column_stack(
                    (
                        upper_cap_r * np.cos(gap_half_angle),
                        upper_cap_r * np.sin(gap_half_angle),
                    )
                )
                points = np.vstack(
                    (outer, lower_cap[1:], inner[1:], upper_cap[1:-1])
                )
                points[:, 0] += 0.10
                points[:, 1] *= 0.92
                points = np.column_stack((points[:, 1], -points[:, 0]))
                boundary_spline = _add_smooth_loop(points, mesh_size)
                outer_loop = gmsh.model.geo.addCurveLoop([boundary_spline])
                surface = gmsh.model.geo.addPlaneSurface([outer_loop])
                physical_boundary = [boundary_spline]
            else:  # protected by argparse; retained for direct function calls
                raise ValueError(f"unknown geometry {geometry_name!r}")

            gmsh.model.geo.synchronize()
            gmsh.model.addPhysicalGroup(2, [surface], 1, "Omega")
            gmsh.model.addPhysicalGroup(1, physical_boundary, 2, "Dirichlet")

        gmsh.model.mesh.generate(2)
        try:
            gmsh.model.mesh.optimize("Netgen")
        except Exception:
            gmsh.model.mesh.optimize()
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


def solve_torsion(
    domain,
    degree: int,
    alpha1: float,
    alpha2: float,
    quadrature_degree: int,
    *,
    prefix: str,
):
    """Solve the torsion problem and return its threshold band."""
    # Reuse the assembly path used by the reduced-optimization script so its
    # boundary treatment and PETSc configuration remain consistent.
    from projects.diocotron.dolfinx.torsion.equilibrium.window_fit import (
        boundary_bc,
        global_minmax,
        solve_linear_form,
    )

    V = fem.functionspace(domain, ("Lagrange", degree))
    trial = ufl.TrialFunction(V)
    test = ufl.TestFunction(V)
    dx = ufl.Measure(
        "dx",
        domain=domain,
        metadata={"quadrature_degree": quadrature_degree},
    )
    stiffness = ufl.inner(ufl.grad(trial), ufl.grad(test)) * dx
    bc = boundary_bc(V)

    torsion = fem.Function(V, name="T")
    solve_linear_form(
        stiffness,
        1.0 * test * dx,
        torsion,
        [bc],
        prefix=prefix,
        solver="mumps",
        ksp_type=None,
        rtol=1.0e-12,
        atol=1.0e-14,
        max_it=None,
        verbosity=0,
    )
    _, tmax = global_minmax(domain.comm, torsion)
    c1 = alpha1 * tmax
    c2 = alpha2 * tmax
    return torsion, c1, c2, stiffness, test, bc


def band_cell_mask(
    torsion: fem.Function,
    c1: float,
    c2: float,
    padding_fraction: float,
) -> np.ndarray:
    """Mark cells whose nodal value range intersects the padded torsion band."""
    V = torsion.function_space
    cell_dofs = np.asarray(V.dofmap.list, dtype=np.int64)
    cell_values = np.asarray(torsion.x.array.real, dtype=np.float64)[cell_dofs]
    cell_min = np.min(cell_values, axis=1)
    cell_max = np.max(cell_values, axis=1)
    padding = padding_fraction * (c2 - c1)
    return (cell_max >= c1 - padding) & (cell_min <= c2 + padding)


def refine_band_mesh(
    domain,
    torsion: fem.Function,
    c1: float,
    c2: float,
    padding_fraction: float,
):
    """Locally refine every triangle intersecting the padded torsion band."""
    marked_cells = np.flatnonzero(
        band_cell_mask(torsion, c1, c2, padding_fraction)
    ).astype(np.int32)
    if marked_cells.size == 0:
        return domain, 0

    tdim = domain.topology.dim
    domain.topology.create_entities(1)
    domain.topology.create_connectivity(tdim, 1)
    marked_edges = dmesh.compute_incident_entities(
        domain.topology, marked_cells, tdim, 1
    )
    refined, _, _ = dmesh.refine(
        domain,
        marked_edges,
        option=dmesh.RefinementOption.parent_cell,
    )
    refined.name = domain.name
    refined.topology.create_entities(refined.topology.dim - 1)
    refined.topology.create_connectivity(refined.topology.dim - 1, refined.topology.dim)
    refined.topology.create_connectivity(refined.topology.dim, refined.topology.dim - 1)
    return refined, int(marked_cells.size)


def solve_fields(
    domain,
    degree: int,
    alpha1: float,
    alpha2: float,
    quadrature_degree: int,
    band_quadrature_degree: int,
    band_padding_fraction: float,
    mollify_density: bool,
    rho_epsilon_ratio: float,
):
    """Solve ``T`` and ``phi_T``, using extra quadrature in the buffered band."""
    from projects.diocotron.dolfinx.torsion.equilibrium.window_fit import (
        solve_linear_form,
        window_ufl,
    )

    torsion, c1, c2, stiffness, test, bc = solve_torsion(
        domain,
        degree,
        alpha1,
        alpha2,
        quadrature_degree,
        prefix="torsion_plot_final_",
    )

    rho_epsilon = rho_epsilon_ratio * (c2 - c1) if mollify_density else None
    if rho_epsilon is None:
        density = ufl.conditional(
            ufl.gt(torsion, c1),
            ufl.conditional(ufl.lt(torsion, c2), 1.0, 0.0),
            0.0,
        )
    else:
        density = window_ufl(torsion, c1, c2, rho_epsilon, 1.0)
    band_mask = band_cell_mask(torsion, c1, c2, band_padding_fraction)
    cells = np.arange(band_mask.size, dtype=np.int32)
    cell_tags = dmesh.meshtags(
        domain,
        domain.topology.dim,
        cells,
        band_mask.astype(np.int32),
    )
    tagged_dx = ufl.Measure("dx", domain=domain, subdomain_data=cell_tags)
    rhs = (
        density
        * test
        * tagged_dx(0, metadata={"quadrature_degree": quadrature_degree})
        + density
        * test
        * tagged_dx(1, metadata={"quadrature_degree": band_quadrature_degree})
    )
    phi_t = fem.Function(torsion.function_space, name="phi_T")
    solve_linear_form(
        stiffness,
        rhs,
        phi_t,
        [bc],
        prefix="phi_t_plot_",
        solver="mumps",
        ksp_type=None,
        rtol=1.0e-12,
        atol=1.0e-14,
        max_it=None,
        verbosity=0,
    )
    return torsion, phi_t, c1, c2, band_mask, rho_epsilon


def high_order_grid(
    torsion: fem.Function,
    phi_t: fem.Function,
    c1: float,
    c2: float,
    rho_epsilon: float | None,
    subdivisions: int,
) -> pv.DataSet:
    """Build a tessellated VTK grid containing ``T``, ``rho_T``, and ``phi_T``."""
    topology, cell_types, points = plot.vtk_mesh(torsion.function_space)
    grid = pv.UnstructuredGrid(topology, cell_types, points)
    torsion_values = np.asarray(torsion.x.array.real, dtype=np.float64)
    grid.point_data["T"] = torsion_values
    grid.point_data["phi_T"] = np.asarray(phi_t.x.array.real, dtype=np.float64)
    if subdivisions > 0:
        if np.all(np.asarray(cell_types) == int(pv.CellType.TRIANGLE)):
            # VTK tessellation does not add sampling points to already-linear
            # triangles. Subdivide those for reliable crisp-band plotting.
            grid = grid.extract_surface(algorithm="dataset_surface").triangulate().subdivide(
                subdivisions,
                subfilter="linear",
            )
        else:
            grid = grid.tessellate(max_n_subdivide=subdivisions, merge_points=True)
    plotted_torsion = np.asarray(grid.point_data["T"], dtype=np.float64)
    if rho_epsilon is None:
        # Classify the tessellated T values rather than tessellating a
        # high-order binary field, which can overshoot near the jumps.
        density_values = (
            (plotted_torsion > c1) & (plotted_torsion < c2)
        ).astype(np.float64)
    else:
        from projects.diocotron.dolfinx.torsion.equilibrium.window_fit import window_numpy

        density_values = window_numpy(
            plotted_torsion,
            c1,
            c2,
            rho_epsilon,
            1.0,
        )
    grid.point_data["rho_T"] = np.clip(density_values, 0.0, 1.0)
    return grid


def show_mesh_window(domain, band_mask: np.ndarray, *, show_grid: bool) -> None:
    """Show the locally refined finite-element mesh in a separate window."""
    topology, cell_types, points = plot.vtk_mesh(domain)
    grid = pv.UnstructuredGrid(topology, cell_types, points)
    plotter = pv.Plotter(title="Band-refined finite-element mesh")
    plotter.set_background("white")
    plotter.add_mesh(
        grid,
        color="#f3f3f3",
        show_edges=True,
        edge_color="#303030",
        line_width=0.65,
    )
    marked_cells = np.flatnonzero(band_mask)
    if marked_cells.size:
        plotter.add_mesh(
            grid.extract_cells(marked_cells),
            color="#ffb74d",
            opacity=0.32,
            show_edges=True,
            edge_color="#7a4b00",
            line_width=0.9,
        )
    plotter.add_text(
        "Refined buffered torsion-band cells",
        position="upper_edge",
        font_size=16,
        color="#111111",
    )
    plotter.add_axes()
    if show_grid:
        plotter.show_grid(
            color="#505050",
            grid="back",
            location="outer",
            ticks="both",
            xtitle="x",
            ytitle="y",
            ztitle="",
        )
    plotter.view_xy()
    plotter.camera.parallel_projection = True
    plotter.reset_camera()
    plotter.enable_trackball_style()
    plotter.add_camera_orientation_widget(animate=True)
    plotter.show()


def contour_values(
    value_max: float,
    contour_levels: int,
    *,
    upper_contour_levels: int = 0,
    upper_contour_start: float = 0.72,
    explicit_contour_levels: np.ndarray | None = None,
) -> np.ndarray:
    """Construct base and peak-focused contour values with vectorized NumPy."""
    if explicit_contour_levels is not None:
        return np.asarray(explicit_contour_levels, dtype=np.float64)
    levels = np.linspace(0.0, value_max, contour_levels + 2)[1:-1]
    if upper_contour_levels:
        s = np.linspace(0.0, 1.0, upper_contour_levels + 2)[1:-1]
        upper_fractions = upper_contour_start + (1.0 - upper_contour_start) * np.sqrt(s)
        levels = np.unique(np.concatenate((levels, value_max * upper_fractions)))
    return levels


def selected_panel_specs(
    *,
    geometry_name: str,
    c1: float,
    c2: float,
    rho_epsilon: float | None,
    phi_top_contours: int,
    phi_top_start: float,
    plot_t: bool,
    plot_rho_t: bool,
    plot_phi_t: bool,
) -> list[dict[str, object]]:
    """Return the small, ordered descriptor list shared by both plotters."""
    panels: list[dict[str, object]] = []
    if plot_t:
        panels.append(
            {
                "scalar": "T",
                "title": f"Torsion T ({geometry_name})",
                "cmap": "viridis",
                "threshold_levels": (c1, c2),
            }
        )
    if plot_rho_t:
        panels.append(
            {
                "scalar": "rho_T",
                "title": (
                    "Mollified density rho_T"
                    if rho_epsilon is not None
                    else "Density rho_T"
                ),
                "cmap": "cividis",
                "explicit_contour_levels": np.array([0.10, 0.25, 0.50, 0.75, 0.90]),
            }
        )
    if plot_phi_t:
        panels.append(
            {
                "scalar": "phi_T",
                "title": "Potential phi_T",
                "cmap": "plasma",
                "upper_contour_levels": phi_top_contours,
                "upper_contour_start": phi_top_start,
            }
        )
    return panels


def _add_surface_panel(
    plotter: pv.Plotter,
    grid: pv.DataSet,
    *,
    position: tuple[int, int],
    scalar: str,
    title: str,
    cmap: str,
    contour_levels: int,
    height_fraction: float,
    threshold_levels: tuple[float, float] | None = None,
    upper_contour_levels: int = 0,
    upper_contour_start: float = 0.72,
    show_grid: bool = False,
    explicit_contour_levels: np.ndarray | None = None,
    plot_dimension: str = "3d",
    threshold_scalar: str = "T",
) -> None:
    plotter.subplot(*position)
    values = np.asarray(grid.point_data[scalar], dtype=np.float64)
    value_max = float(np.max(values))
    xy_span = max(grid.bounds[1] - grid.bounds[0], grid.bounds[3] - grid.bounds[2])
    warp_factor = height_fraction * xy_span / max(value_max, np.finfo(float).eps)
    is_2d = plot_dimension == "2d"
    surface = grid if is_2d else grid.warp_by_scalar(scalar, factor=warp_factor)

    levels = contour_values(
        value_max,
        contour_levels,
        upper_contour_levels=upper_contour_levels,
        upper_contour_start=upper_contour_start,
        explicit_contour_levels=explicit_contour_levels,
    )
    contours = grid.contour(isosurfaces=levels, scalars=scalar)
    if not is_2d:
        contours = contours.warp_by_scalar(scalar, factor=warp_factor)
    plotter.add_mesh(
        surface,
        scalars=scalar,
        cmap=cmap,
        clim=(0.0, value_max),
        smooth_shading=not is_2d,
        show_edges=False,
        ambient=0.24,
        diffuse=0.76,
        specular=0.18,
        scalar_bar_args={
            "title": scalar,
            "vertical": True,
            "position_x": 0.84,
            "position_y": 0.10,
            "width": 0.065,
            "height": 0.78,
            "title_font_size": 24,
            "label_font_size": 16,
            "fmt": "%.2e",
        },
    )
    if contours.n_points:
        plotter.add_mesh(contours, color="#161616", line_width=2.0, opacity=0.78)

    if threshold_levels is not None:
        threshold_colors = ("#00e5ff", "#ffea00")
        for level, color in zip(threshold_levels, threshold_colors, strict=True):
            line = grid.contour(isosurfaces=[level], scalars=threshold_scalar)
            if is_2d and line.n_points:
                # Keep the band limits above the filled surface and the regular
                # contours.  Coplanar VTK actors can otherwise z-fight and make
                # these important lines look broken or blurred.
                line = line.copy(deep=True)
                line.points[:, 2] += 1.0e-4 * xy_span
            elif not is_2d:
                line = line.warp_by_scalar(scalar, factor=warp_factor)
            if line.n_points:
                # A narrow dark casing preserves contrast over every colormap
                # while leaving the cyan/yellow limits crisp and identifiable.
                plotter.add_mesh(line, color="#101010", line_width=7.0)
                plotter.add_mesh(line, color=color, line_width=4.0)

    plotter.add_text(title, position="upper_edge", font_size=18, color="#111111")
    plotter.add_axes(line_width=2, labels_off=False)
    if show_grid:
        plotter.show_grid(
            color="#505050",
            grid="back",
            location="outer",
            ticks="both",
            xtitle="x",
            ytitle="y",
            ztitle="" if is_2d else "display height",
        )
    if is_2d:
        plotter.view_xy()
    else:
        plotter.view_isometric()
    plotter.camera.parallel_projection = True


def render(
    grid: pv.DataSet,
    output: Path,
    *,
    geometry_name: str,
    c1: float,
    c2: float,
    rho_epsilon: float | None,
    contour_levels: int,
    phi_top_contours: int,
    phi_top_start: float,
    height_fraction: float,
    window_size: tuple[int, int],
    show_grid: bool,
    plot_t: bool,
    plot_rho_t: bool,
    plot_phi_t: bool,
    plot_dimension: str,
    save: bool,
    off_screen: bool,
) -> None:
    """Render the selected scalar fields as linked panels in one PyVista plot."""
    panels = selected_panel_specs(
        geometry_name=geometry_name,
        c1=c1,
        c2=c2,
        rho_epsilon=rho_epsilon,
        phi_top_contours=phi_top_contours,
        phi_top_start=phi_top_start,
        plot_t=plot_t,
        plot_rho_t=plot_rho_t,
        plot_phi_t=plot_phi_t,
    )

    positions = [(0, index) for index in range(len(panels))]
    plotter = pv.Plotter(
        shape=(1, len(panels)),
        off_screen=off_screen,
        window_size=window_size,
    )
    plotter.set_background("#f7f7f4")
    for position, panel in zip(positions, panels, strict=True):
        _add_surface_panel(
            plotter,
            grid,
            position=position,
            scalar=str(panel["scalar"]),
            title=str(panel["title"]),
            cmap=str(panel["cmap"]),
            contour_levels=contour_levels,
            height_fraction=height_fraction,
            threshold_levels=(
                (c1, c2)
                if plot_dimension == "2d"
                else panel.get("threshold_levels")
            ),
            upper_contour_levels=int(panel.get("upper_contour_levels", 0)),
            upper_contour_start=float(panel.get("upper_contour_start", 0.72)),
            show_grid=show_grid,
            explicit_contour_levels=panel.get("explicit_contour_levels"),
            plot_dimension=plot_dimension,
            threshold_scalar="T",
        )
    plotter.link_views()
    for position in positions:
        plotter.subplot(*position)
        plotter.reset_camera()
        plotter.camera.zoom(0.93)
        if not off_screen:
            plotter.add_camera_orientation_widget(animate=True)
    if not off_screen:
        if plot_dimension == "2d":
            plotter.enable_image_style()
        else:
            plotter.enable_trackball_style()

        def apply_to_cameras(action) -> None:
            for position in positions:
                plotter.subplot(*position)
                action(plotter)
            plotter.render()

        plotter.add_key_event("r", lambda: apply_to_cameras(lambda p: p.reset_camera()))
        plotter.add_key_event("t", lambda: apply_to_cameras(lambda p: p.view_xy()))
        plotter.add_key_event("i", lambda: apply_to_cameras(lambda p: p.view_isometric()))
        plotter.add_key_event("+", lambda: apply_to_cameras(lambda p: p.camera.zoom(1.15)))
        plotter.add_key_event("-", lambda: apply_to_cameras(lambda p: p.camera.zoom(0.87)))
        plotter.subplot(0, 0)
        zoom_state = [1.0]

        def set_zoom(value: float) -> None:
            value = float(value)
            ratio = value / zoom_state[0]
            zoom_state[0] = value
            apply_to_cameras(lambda p: p.camera.zoom(ratio))

        plotter.add_slider_widget(
            set_zoom,
            (0.5, 2.0),
            value=1.0,
            title="Camera zoom",
            pointa=(0.24, 0.08),
            pointb=(0.72, 0.08),
            interaction_event="always",
        )
        plotter.add_text(
            "Mouse: rotate/pan/zoom   R: reset   T: top   I: isometric   +/-: zoom",
            position="lower_edge",
            font_size=10,
            color="#202020",
        )
    plotter.enable_anti_aliasing("ssaa")
    if save:
        output.parent.mkdir(parents=True, exist_ok=True)
    if not off_screen:
        # ``show`` initializes an on-screen render window. Passing the
        # screenshot here avoids calling ``screenshot`` before that window
        # exists when --show and --save are combined.
        plotter.show(
            auto_close=False,
            screenshot=str(output) if save else False,
        )
    elif save:
        plotter.screenshot(str(output), transparent_background=False)
    plotter.close()


def save_matplotlib_plot(
    grid: pv.DataSet,
    output: Path,
    *,
    geometry_name: str,
    c1: float,
    c2: float,
    rho_epsilon: float | None,
    contour_levels: int,
    phi_top_contours: int,
    phi_top_start: float,
    height_fraction: float,
    show_grid: bool,
    plot_t: bool,
    plot_rho_t: bool,
    plot_phi_t: bool,
    plot_dimension: str,
    dpi: int,
    show: bool,
    backend: str | None,
) -> None:
    """Save vectorized Matplotlib trisurfaces matching the PyVista panels."""
    import matplotlib

    if backend is not None:
        matplotlib.use(backend, force=True)
    elif not show:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    panels = selected_panel_specs(
        geometry_name=geometry_name,
        c1=c1,
        c2=c2,
        rho_epsilon=rho_epsilon,
        phi_top_contours=phi_top_contours,
        phi_top_start=phi_top_start,
        plot_t=plot_t,
        plot_rho_t=plot_rho_t,
        plot_phi_t=plot_phi_t,
    )
    surface = (
        grid
        if isinstance(grid, pv.PolyData)
        else grid.extract_surface(algorithm="dataset_surface")
    ).triangulate()
    points = np.asarray(surface.points, dtype=np.float64)
    triangles = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    xy_span = max(np.ptp(points[:, 0]), np.ptp(points[:, 1]))

    fig = plt.figure(figsize=(6.2 * len(panels), 5.6), dpi=dpi, constrained_layout=True)
    # This loop has at most three iterations and is required to construct a
    # dynamic number of Matplotlib axes; all point/cell work remains vectorized.
    for index, panel in enumerate(panels, start=1):
        if plot_dimension == "2d":
            ax = fig.add_subplot(1, len(panels), index)
        else:
            ax = fig.add_subplot(
                1,
                len(panels),
                index,
                projection="3d",
                computed_zorder=False,
            )
        scalar = str(panel["scalar"])
        values = np.asarray(surface.point_data[scalar], dtype=np.float64)
        value_max = float(np.max(values))
        levels = contour_values(
            value_max,
            contour_levels,
            upper_contour_levels=int(panel.get("upper_contour_levels", 0)),
            upper_contour_start=float(panel.get("upper_contour_start", 0.72)),
            explicit_contour_levels=panel.get("explicit_contour_levels"),
        )
        if plot_dimension == "2d":
            image = ax.tripcolor(
                triangulation,
                values,
                shading="gouraud",
                cmap=str(panel["cmap"]),
                vmin=0.0,
                vmax=value_max,
            )
            ax.tricontour(
                triangulation,
                values,
                levels=levels,
                colors="#101010",
                linewidths=0.65,
                alpha=0.82,
                zorder=10,
            )
            torsion_values = np.asarray(surface.point_data["T"], dtype=np.float64)
            ax.tricontour(
                triangulation,
                torsion_values,
                levels=(c1, c2),
                colors="#101010",
                linewidths=3.2,
                zorder=29,
            )
            ax.tricontour(
                triangulation,
                torsion_values,
                levels=(c1, c2),
                colors=("#00e5ff", "#ffea00"),
                linewidths=2.0,
                zorder=30,
            )
            fig.colorbar(image, ax=ax, shrink=0.78, pad=0.025, label=scalar)
            ax.set_title(str(panel["title"]))
            ax.set_aspect("equal", adjustable="box")
            if show_grid:
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.grid(True, linewidth=0.4, alpha=0.45)
            else:
                ax.set_axis_off()
            continue

        warp_factor = height_fraction * xy_span / max(value_max, np.finfo(float).eps)
        heights = warp_factor * values
        surface_artist = ax.plot_trisurf(
            triangulation,
            heights,
            cmap=str(panel["cmap"]),
            vmin=0.0,
            vmax=warp_factor * value_max,
            linewidth=0.0,
            antialiased=False,
            shade=True,
        )
        surface_artist.set_zorder(1)
        contour_offset = 0.006 * height_fraction * xy_span
        contour_grid = surface.contour(isosurfaces=levels, scalars=scalar)
        encoded_lines = np.asarray(contour_grid.lines, dtype=np.int64)
        if encoded_lines.size:
            line_cells = encoded_lines.reshape(-1, 3)
            if not np.all(line_cells[:, 0] == 2):
                raise RuntimeError("expected two-point line cells from triangular contours")
            endpoint_ids = line_cells[:, 1:]
            contour_segments = np.asarray(
                contour_grid.points[endpoint_ids],
                dtype=np.float64,
            ).copy()
            contour_scalars = np.asarray(
                contour_grid.point_data[scalar][endpoint_ids],
                dtype=np.float64,
            )
            contour_segments[:, :, 2] = (
                warp_factor * contour_scalars + contour_offset
            )
            contour_collection = Line3DCollection(
                contour_segments,
                colors="#101010",
                linewidths=1.15,
                alpha=0.92,
            )
            contour_collection.set_sort_zpos(float(np.max(heights) + contour_offset))
            contour_collection.set_zorder(20)
            ax.add_collection3d(contour_collection)
        thresholds = panel.get("threshold_levels")
        if thresholds is not None:
            threshold_values = np.asarray(thresholds, dtype=np.float64)
            threshold_grid = surface.contour(
                isosurfaces=threshold_values,
                scalars=scalar,
            )
            threshold_lines = np.asarray(threshold_grid.lines, dtype=np.int64)
            if threshold_lines.size:
                threshold_cells = threshold_lines.reshape(-1, 3)
                endpoint_ids = threshold_cells[:, 1:]
                threshold_segments = np.asarray(
                    threshold_grid.points[endpoint_ids],
                    dtype=np.float64,
                ).copy()
                threshold_scalars = np.asarray(
                    threshold_grid.point_data[scalar][endpoint_ids],
                    dtype=np.float64,
                )
                threshold_segments[:, :, 2] = (
                    warp_factor * threshold_scalars + 1.5 * contour_offset
                )
                segment_values = np.mean(threshold_scalars, axis=1)
                color_ids = np.argmin(
                    np.abs(segment_values[:, None] - threshold_values[None, :]),
                    axis=1,
                )
                threshold_colors = np.asarray(("#00e5ff", "#ffea00"))[color_ids]
                threshold_collection = Line3DCollection(
                    threshold_segments,
                    colors=threshold_colors,
                    linewidths=2.4,
                )
                threshold_collection.set_sort_zpos(
                    float(np.max(heights) + 1.5 * contour_offset)
                )
                threshold_collection.set_zorder(25)
                ax.add_collection3d(threshold_collection)
        mappable = ScalarMappable(
            norm=Normalize(vmin=0.0, vmax=value_max),
            cmap=str(panel["cmap"]),
        )
        fig.colorbar(mappable, ax=ax, shrink=0.68, pad=0.03, label=scalar)
        ax.set_title(str(panel["title"]))
        ax.set_proj_type("ortho")
        ax.view_init(elev=28.0, azim=-58.0)
        ax.set_box_aspect((np.ptp(points[:, 0]), np.ptp(points[:, 1]), height_fraction * xy_span))
        if show_grid:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("display height")
            ax.grid(True, linewidth=0.4, alpha=0.45)
        else:
            ax.set_axis_off()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="#f7f7f4")
    if show:
        plt.show(block=True)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry",
        choices=tuple(DEFAULT_MESH_SIZES),
        default="star",
        help="domain to mesh and solve (default: star)",
    )
    parser.add_argument(
        "--mesh-size",
        type=float,
        default=None,
        help=(
            "target size near the band; defaults to 0.020 (horseshoe/Pac-Man), "
            "0.030 (convex polygon), 0.035 (star/hole), or 0.070 (ITER)"
        ),
    )
    parser.add_argument(
        "--bulk-coarsening-factor",
        type=float,
        default=None,
        help="initial bulk size divided by --mesh-size; default is 2^band-refinements",
    )
    parser.add_argument("--order", type=int, default=6, help="Lagrange degree (default: 6)")
    parser.add_argument("--alpha1", type=float, default=0.60)
    parser.add_argument("--alpha2", type=float, default=0.70)
    parser.add_argument("--quadrature-degree", type=int, default=None)
    parser.add_argument(
        "--band-quadrature-degree",
        type=int,
        default=None,
        help="quadrature degree in cells near c1<T<c2; default is higher than the bulk degree",
    )
    parser.add_argument(
        "--band-refinements",
        type=int,
        default=1,
        help="local mesh-refinement passes in and around the torsion band (default: 1)",
    )
    parser.add_argument(
        "--band-padding-fraction",
        type=float,
        default=0.35,
        help=(
            "padding on each side of [c1,c2] used for both mesh adaptation "
            "and high quadrature, as a fraction of c2-c1 (default: 0.35)"
        ),
    )
    parser.add_argument("--contours", type=int, default=28)
    parser.add_argument(
        "--phi-top-contours",
        type=int,
        default=20,
        help="extra phi_T contours biased toward its maximum (default: 20)",
    )
    parser.add_argument(
        "--phi-top-start",
        type=float,
        default=0.72,
        help="fraction of max(phi_T) above which extra contours are added (default: 0.72)",
    )
    parser.add_argument(
        "--rho-epsilon-ratio",
        type=float,
        default=0.08,
        help="mollifier epsilon divided by c2-c1 when rho_T is plotted (default: 0.08)",
    )
    parser.add_argument(
        "--tessellation",
        type=int,
        default=None,
        help="VTK subdivision level; default is 2 with rho_T or 3 without it",
    )
    parser.add_argument("--height-fraction", type=float, default=0.42)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=1800)
    parser.add_argument(
        "--plot-dimension",
        choices=("2d", "3d"),
        default="3d",
        help="render flat 2-D fields or warped 3-D surfaces (default: 3d)",
    )
    parser.add_argument(
        "--plot-t",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include the torsion T panel (default: enabled)",
    )
    parser.add_argument(
        "--plot-rho-t",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include mollified rho_T and use it for phi_T (default: enabled)",
    )
    parser.add_argument(
        "--plot-phi-t",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include the initialized potential phi_T panel (default: enabled)",
    )
    parser.add_argument(
        "--grid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show coordinate grids on the scalar panels and optional mesh window",
    )
    parser.add_argument(
        "--matplotlib",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable Matplotlib output; --show also opens its interactive window",
    )
    parser.add_argument(
        "--pyvista",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable the PyVista renderer/window (default: enabled)",
    )
    parser.add_argument(
        "--matplotlib-output",
        type=Path,
        default=None,
        help="Matplotlib PNG path; default appends _matplotlib to the PyVista filename",
    )
    parser.add_argument("--matplotlib-dpi", type=int, default=240)
    parser.add_argument(
        "--matplotlib-backend",
        default=None,
        help="optional interactive backend, for example QtAgg or TkAgg",
    )
    parser.add_argument("--gmsh-verbosity", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG path; default filename lists the selected scalar fields",
    )
    parser.add_argument(
        "--save",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save the PNG (default: enabled; use --no-save for display only)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="open an interactive PyVista window (independent of --save)",
    )
    parser.add_argument(
        "--show-mesh",
        action="store_true",
        help="open the band-refined mesh in a separate PyVista window",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("this visualization script must be run with one MPI rank")
    if args.mesh_size is not None and args.mesh_size <= 0.0:
        raise ValueError("--mesh-size must be positive")
    if args.bulk_coarsening_factor is not None and args.bulk_coarsening_factor < 1.0:
        raise ValueError("--bulk-coarsening-factor must be at least 1")
    if args.order < 1:
        raise ValueError("--order must be at least 1")
    if not 0.0 <= args.alpha1 < args.alpha2 <= 1.0:
        raise ValueError("require 0 <= --alpha1 < --alpha2 <= 1")
    if args.quadrature_degree is not None and args.quadrature_degree < 1:
        raise ValueError("--quadrature-degree must be positive")
    if args.band_quadrature_degree is not None and args.band_quadrature_degree < 1:
        raise ValueError("--band-quadrature-degree must be positive")
    if args.band_refinements < 0:
        raise ValueError("--band-refinements must be nonnegative")
    if args.band_padding_fraction < 0.0:
        raise ValueError("--band-padding-fraction must be nonnegative")
    if args.contours < 2:
        raise ValueError("--contours must be at least 2")
    if args.phi_top_contours < 0:
        raise ValueError("--phi-top-contours must be nonnegative")
    if not 0.0 < args.phi_top_start < 1.0:
        raise ValueError("require 0 < --phi-top-start < 1")
    if args.rho_epsilon_ratio <= 0.0:
        raise ValueError("--rho-epsilon-ratio must be positive")
    if args.tessellation is not None and args.tessellation < 0:
        raise ValueError("--tessellation must be nonnegative")
    if args.height_fraction <= 0.0:
        raise ValueError("--height-fraction must be positive")
    if args.width < 640 or args.height < 480:
        raise ValueError("the render window must be at least 640 x 480")
    if args.matplotlib_dpi < 72:
        raise ValueError("--matplotlib-dpi must be at least 72")
    if not any((args.plot_t, args.plot_rho_t, args.plot_phi_t)):
        raise ValueError("at least one of --plot-t, --plot-rho-t, or --plot-phi-t must be enabled")
    pyvista_requested = args.pyvista and (args.save or args.show)
    matplotlib_requested = args.matplotlib
    if not pyvista_requested and not args.show_mesh and not matplotlib_requested:
        raise ValueError(
            "nothing to do: enable PyVista save/show, --show-mesh, or --matplotlib"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    mesh_size = args.mesh_size or DEFAULT_MESH_SIZES[args.geometry]
    bulk_coarsening_factor = (
        args.bulk_coarsening_factor
        if args.bulk_coarsening_factor is not None
        else float(2**args.band_refinements)
    )
    bulk_mesh_size = mesh_size * bulk_coarsening_factor
    quadrature_degree = args.quadrature_degree or max(2 * args.order + 8, 20)
    band_quadrature_degree = args.band_quadrature_degree or max(
        quadrature_degree + 12,
        4 * args.order + 20,
    )
    if band_quadrature_degree < quadrature_degree:
        raise ValueError("--band-quadrature-degree must not be smaller than the bulk degree")
    tessellation = args.tessellation if args.tessellation is not None else (2 if args.plot_rho_t else 3)
    selected_fields = [
        token
        for enabled, token in (
            (args.plot_t, "T"),
            (args.plot_rho_t, "rhoT"),
            (args.plot_phi_t, "phiT"),
        )
        if enabled
    ]
    output = args.output or (
        DEFAULT_OUTPUT_DIR
        / (
            f"{args.geometry}_{'_'.join(selected_fields)}_"
            f"{args.plot_dimension}_p{args.order}.png"
        )
    )
    matplotlib_output = args.matplotlib_output or output.with_name(
        f"{output.stem}_matplotlib.png"
    )

    # The mesh is an implementation detail of this plotting run, so keep it
    # temporary and leave only the requested visualization behind.
    with tempfile.TemporaryDirectory(prefix="torsion_pyvista_") as temp_dir:
        mesh_path = Path(temp_dir) / f"{args.geometry}.msh"
        generate_mesh(args.geometry, bulk_mesh_size, mesh_path, args.gmsh_verbosity)

        from projects.diocotron.dolfinx.torsion.equilibrium.window_fit import read_mesh_with_meshio

        domain = read_mesh_with_meshio(mesh_path, MPI.COMM_WORLD)
        for refinement in range(args.band_refinements):
            preview_torsion, preview_c1, preview_c2, _, _, _ = solve_torsion(
                domain,
                args.order,
                args.alpha1,
                args.alpha2,
                quadrature_degree,
                prefix=f"torsion_plot_adapt_{refinement}_",
            )
            old_cells = domain.topology.index_map(domain.topology.dim).size_global
            band_width = preview_c2 - preview_c1
            buffer_low = preview_c1 - args.band_padding_fraction * band_width
            buffer_high = preview_c2 + args.band_padding_fraction * band_width
            domain, marked_cells = refine_band_mesh(
                domain,
                preview_torsion,
                preview_c1,
                preview_c2,
                args.band_padding_fraction,
            )
            new_cells = domain.topology.index_map(domain.topology.dim).size_global
            print(
                f"band_refinement={refinement + 1} marked_cells={marked_cells} "
                f"buffer=[{buffer_low:.6e},{buffer_high:.6e}] "
                f"cells={old_cells}->{new_cells}"
            )
            if marked_cells == 0:
                break

        torsion, phi_t, c1, c2, band_mask, rho_epsilon = solve_fields(
            domain,
            args.order,
            args.alpha1,
            args.alpha2,
            quadrature_degree,
            band_quadrature_degree,
            args.band_padding_fraction,
            args.plot_rho_t,
            args.rho_epsilon_ratio,
        )
        grid = high_order_grid(
            torsion,
            phi_t,
            c1,
            c2,
            rho_epsilon,
            tessellation,
        )
        if args.pyvista and (args.save or args.show):
            render(
                grid,
                output,
                geometry_name=args.geometry,
                c1=c1,
                c2=c2,
                rho_epsilon=rho_epsilon,
                contour_levels=args.contours,
                phi_top_contours=args.phi_top_contours,
                phi_top_start=args.phi_top_start,
                height_fraction=args.height_fraction,
                window_size=(args.width, args.height),
                show_grid=args.grid,
                plot_t=args.plot_t,
                plot_rho_t=args.plot_rho_t,
                plot_phi_t=args.plot_phi_t,
                plot_dimension=args.plot_dimension,
                save=args.save,
                off_screen=not args.show,
            )
        if args.matplotlib:
            save_matplotlib_plot(
                grid,
                matplotlib_output,
                geometry_name=args.geometry,
                c1=c1,
                c2=c2,
                rho_epsilon=rho_epsilon,
                contour_levels=args.contours,
                phi_top_contours=args.phi_top_contours,
                phi_top_start=args.phi_top_start,
                height_fraction=args.height_fraction,
                show_grid=args.grid,
                plot_t=args.plot_t,
                plot_rho_t=args.plot_rho_t,
                plot_phi_t=args.plot_phi_t,
                plot_dimension=args.plot_dimension,
                dpi=args.matplotlib_dpi,
                show=args.show,
                backend=args.matplotlib_backend,
            )
        if args.show_mesh:
            show_mesh_window(domain, band_mask, show_grid=args.grid)

    cells = domain.topology.index_map(domain.topology.dim).size_global
    dofs = torsion.function_space.dofmap.index_map.size_global
    print(
        f"geometry={args.geometry} band_mesh_size={mesh_size:g} "
        f"bulk_mesh_size={bulk_mesh_size:g} cells={cells} order={args.order} dofs={dofs}"
    )
    print(f"Tmax={np.max(torsion.x.array.real):.12e} c1={c1:.12e} c2={c2:.12e}")
    print(f"phi_T_max={np.max(phi_t.x.array.real):.12e}")
    density_mode = "mollified" if rho_epsilon is not None else "crisp"
    print(
        f"rho_T_mode={density_mode} rho_epsilon={rho_epsilon} "
        f"plot_tessellation={tessellation}"
    )
    print(
        f"band_cells={np.count_nonzero(band_mask)} "
        f"quadrature_degree={quadrature_degree} "
        f"band_quadrature_degree={band_quadrature_degree}"
    )
    if args.pyvista and args.save:
        print(f"figure={output.resolve()}")
    if args.matplotlib:
        print(f"matplotlib_figure={matplotlib_output.resolve()}")
    if args.show:
        shown = [
            name
            for enabled, name in (
                (args.pyvista, "pyvista"),
                (args.matplotlib, "matplotlib"),
            )
            if enabled
        ]
        print(f"interactive_windows={','.join(shown)}")


if __name__ == "__main__":
    main()
