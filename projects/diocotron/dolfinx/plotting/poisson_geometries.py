#!/usr/bin/env python3
"""Generate DOLFINx Poisson/torsion figures on several nontrivial geometries.

The script solves

    -Delta u = 1 in Omega,  u = 0 on dOmega

with continuous high-order Lagrange elements, then writes high-resolution 2D
filled-contour figures for each geometry with both Matplotlib and PyVista.
"""

from __future__ import annotations

# Support direct execution as well as python -m from a checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hdgfem_poisson_figures_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/hdgfem_poisson_figures_cache")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import gmsh
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import ufl
from dolfinx import fem, geometry, mesh, plot
from dolfinx.fem import petsc as fem_petsc
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
from petsc4py import PETSc

from projects.diocotron.dolfinx.geometry.canonical import build_gmsh_model as build_canonical_gmsh_model


@dataclass(frozen=True)
class GeometryCase:
    slug: str
    title: str
    h: float
    mesh_path: Path | None = None


CASES = (
    GeometryCase("polygonal_convex", "Polygonal Convex Domain", 0.030),
    GeometryCase("five_node_star", "Five-Node Star-Shaped Domain", 0.022),
    GeometryCase("pacman", "Narrow-Back Pac-Man Domain", 0.020),
    GeometryCase("smooth_nonconvex_hole", "Smooth Non-Convex Domain With Hole", 0.024),
    GeometryCase("horseshoe", "Horseshoe Domain", 0.020),
    GeometryCase("iter_wall", "ITER Wall Boundary", 0.070),
)

CASE_BY_SLUG = {case.slug: case for case in CASES}
PACMAN_RADIUS = 1.0
PACMAN_MOUTH_HALF_ANGLE = 0.38
PACMAN_TIP_X = -0.78
PACMAN_TIP_TO_BACK_DISTANCE = PACMAN_RADIUS + PACMAN_TIP_X


def _add_polygon(points: np.ndarray, h: float) -> tuple[list[int], list[int]]:
    point_tags = [gmsh.model.geo.addPoint(float(x), float(y), 0.0, h) for x, y in points]
    line_tags = []
    for i, start in enumerate(point_tags):
        end = point_tags[(i + 1) % len(point_tags)]
        line_tags.append(gmsh.model.geo.addLine(start, end))
    return point_tags, line_tags


def _add_smooth_loop(points: np.ndarray, h: float) -> tuple[list[int], int]:
    point_tags = [gmsh.model.geo.addPoint(float(x), float(y), 0.0, h) for x, y in points]
    spline = gmsh.model.geo.addSpline(point_tags + [point_tags[0]])
    return point_tags, spline


def _add_circle_loop(cx: float, cy: float, radius: float, h: float) -> tuple[list[int], list[int]]:
    center = gmsh.model.geo.addPoint(cx, cy, 0.0, h)
    angles = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)
    points = [gmsh.model.geo.addPoint(cx + radius * math.cos(a), cy + radius * math.sin(a), 0.0, h) for a in angles]
    arcs = [
        gmsh.model.geo.addCircleArc(points[0], center, points[1]),
        gmsh.model.geo.addCircleArc(points[1], center, points[2]),
        gmsh.model.geo.addCircleArc(points[2], center, points[3]),
        gmsh.model.geo.addCircleArc(points[3], center, points[0]),
    ]
    return points, arcs


def build_gmsh_model(case: GeometryCase) -> None:
    if case.slug in {"pacman", "horseshoe"}:
        build_canonical_gmsh_model(case.slug, case.h, verbosity=1, algorithm=6)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Netgen")
        return

    gmsh.model.add(case.slug)
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.45 * case.h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", case.h)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)

    if case.slug == "pacman":
        center = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, case.h)
        upper = gmsh.model.geo.addPoint(
            PACMAN_RADIUS * math.cos(PACMAN_MOUTH_HALF_ANGLE),
            PACMAN_RADIUS * math.sin(PACMAN_MOUTH_HALF_ANGLE),
            0.0,
            case.h,
        )
        back = gmsh.model.geo.addPoint(-PACMAN_RADIUS, 0.0, 0.0, case.h)
        lower = gmsh.model.geo.addPoint(
            PACMAN_RADIUS * math.cos(PACMAN_MOUTH_HALF_ANGLE),
            -PACMAN_RADIUS * math.sin(PACMAN_MOUTH_HALF_ANGLE),
            0.0,
            case.h,
        )
        tip = gmsh.model.geo.addPoint(PACMAN_TIP_X, 0.0, 0.0, case.h)
        upper_arc = gmsh.model.geo.addCircleArc(upper, center, back)
        lower_arc = gmsh.model.geo.addCircleArc(back, center, lower)
        lower_mouth = gmsh.model.geo.addLine(lower, tip)
        upper_mouth = gmsh.model.geo.addLine(tip, upper)
        boundary = [upper_arc, lower_arc, lower_mouth, upper_mouth]
        outer_loop = gmsh.model.geo.addCurveLoop(boundary)
        surface = gmsh.model.geo.addPlaneSurface([outer_loop])
        physical_boundary = boundary

    elif case.slug == "polygonal_convex":
        theta = np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False) + 0.10
        radii = np.array([1.04, 0.94, 1.10, 0.98, 1.06, 0.96, 1.08, 0.92, 1.02])
        points = np.column_stack((radii * np.cos(theta), 0.82 * radii * np.sin(theta)))
        _, boundary = _add_polygon(points, case.h)
        outer_loop = gmsh.model.geo.addCurveLoop(boundary)
        surface = gmsh.model.geo.addPlaneSurface([outer_loop])
        physical_boundary = boundary

    elif case.slug == "five_node_star":
        theta = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False) + 0.5 * math.pi
        radii = np.empty(10)
        radii[0::2] = 1.06
        radii[1::2] = 0.44
        points = np.column_stack((radii * np.cos(theta), radii * np.sin(theta)))
        _, boundary = _add_polygon(points, case.h)
        outer_loop = gmsh.model.geo.addCurveLoop(boundary)
        surface = gmsh.model.geo.addPlaneSurface([outer_loop])
        physical_boundary = boundary

    elif case.slug == "smooth_nonconvex_hole":
        theta = np.linspace(0.0, 2.0 * np.pi, 220, endpoint=False)
        radius = 1.0 + 0.34 * np.cos(3.0 * theta) - 0.08 * np.sin(2.0 * theta)
        x = radius * np.cos(theta)
        y = 0.86 * radius * np.sin(theta)
        _, outer_spline = _add_smooth_loop(np.column_stack((x, y)), case.h)
        _, hole_arcs = _add_circle_loop(0.20, 0.03, 0.22, 0.75 * case.h)
        outer_loop = gmsh.model.geo.addCurveLoop([outer_spline])
        hole_loop = gmsh.model.geo.addCurveLoop(hole_arcs)
        surface = gmsh.model.geo.addPlaneSurface([outer_loop, hole_loop])
        physical_boundary = [outer_spline, *hole_arcs]

    elif case.slug == "horseshoe":
        gap_half_angle = 0.48
        outer_radius = 1.16
        inner_radius = 0.46
        outer_theta = np.linspace(gap_half_angle, 2.0 * np.pi - gap_half_angle, 170)
        inner_theta = np.linspace(2.0 * np.pi - gap_half_angle, gap_half_angle, 130)
        cap_steps = 28

        outer = np.column_stack((outer_radius * np.cos(outer_theta), outer_radius * np.sin(outer_theta)))
        lower_cap_r = np.linspace(outer_radius, inner_radius, cap_steps)
        lower_cap = np.column_stack(
            (lower_cap_r * np.cos(2.0 * np.pi - gap_half_angle), lower_cap_r * np.sin(2.0 * np.pi - gap_half_angle))
        )
        inner = np.column_stack((inner_radius * np.cos(inner_theta), inner_radius * np.sin(inner_theta)))
        upper_cap_r = np.linspace(inner_radius, outer_radius, cap_steps)
        upper_cap = np.column_stack((upper_cap_r * np.cos(gap_half_angle), upper_cap_r * np.sin(gap_half_angle)))

        points = np.vstack((outer, lower_cap[1:], inner[1:], upper_cap[1:-1]))
        points[:, 0] += 0.10
        points[:, 1] *= 0.92
        points = np.column_stack((points[:, 1], -points[:, 0]))
        _, boundary_spline = _add_smooth_loop(points, case.h)
        outer_loop = gmsh.model.geo.addCurveLoop([boundary_spline])
        surface = gmsh.model.geo.addPlaneSurface([outer_loop])
        physical_boundary = [boundary_spline]

    else:
        raise ValueError(f"unknown case {case.slug}")

    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [surface], 1, "Omega")
    gmsh.model.addPhysicalGroup(1, physical_boundary, 2, "Dirichlet")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Netgen")


def create_domain(case: GeometryCase):
    if case.mesh_path is not None:
        if not case.mesh_path.exists():
            raise FileNotFoundError(
                f"{case.mesh_path} does not exist. Generate it first with "
                "python -m projects.diocotron.dolfinx.geometry.canonical iter "
                f"--mesh-size {case.h} --output {case.mesh_path}"
            )
        mesh_data = gmshio.read_from_msh(case.mesh_path, MPI.COMM_WORLD, rank=0, gdim=2)
        domain = mesh_data.mesh
        domain.name = case.slug
        tdim = domain.topology.dim
        domain.topology.create_connectivity(tdim - 1, tdim)
        domain.topology.create_connectivity(tdim, tdim - 1)
        return domain

    gmsh.initialize()
    try:
        if case.slug == "iter_wall":
            build_canonical_gmsh_model("iter", case.h, verbosity=1, algorithm=6)
            gmsh.model.mesh.generate(2)
            gmsh.model.mesh.optimize("Netgen")
        else:
            build_gmsh_model(case)
        mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
        domain = mesh_data.mesh
    finally:
        gmsh.finalize()

    domain.name = case.slug
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    domain.topology.create_connectivity(tdim, tdim - 1)
    return domain


def solve_poisson(domain, degree: int, prefix: str) -> fem.Function:
    V = fem.functionspace(domain, ("Lagrange", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain, PETSc.ScalarType(1.0))

    facets = mesh.locate_entities_boundary(
        domain, domain.topology.dim - 1, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, domain.topology.dim - 1, facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    problem = fem_petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix=f"{prefix}_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    uh = problem.solve()
    uh.name = "u"
    uh.x.scatter_forward()
    return uh


def sample_function_at_points(function: fem.Function, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if points_xy.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=bool)

    domain = function.function_space.mesh
    points = np.zeros((points_xy.shape[0], 3), dtype=np.float64)
    points[:, :2] = points_xy
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(domain, candidates, points)

    cells = np.full(points.shape[0], -1, dtype=np.int32)
    valid: list[int] = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            cells[i] = int(links[0])
            valid.append(i)

    values = np.zeros(points.shape[0], dtype=np.float64)
    inside = cells >= 0
    if valid:
        valid_idx = np.asarray(valid, dtype=np.int32)
        evaluated = function.eval(points[valid_idx], cells[valid_idx])
        values[valid_idx] = np.asarray(evaluated, dtype=np.float64).reshape(len(valid_idx), -1)[:, 0]
    return values, inside


def dense_plot_data(uh: fem.Function, pixels: int) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    coords = uh.function_space.mesh.geometry.x
    xmin, ymin = np.min(coords[:, :2], axis=0)
    xmax, ymax = np.max(coords[:, :2], axis=0)
    width = float(xmax - xmin)
    height = float(ymax - ymin)
    pad = 0.015 * max(width, height)
    xmin -= pad
    xmax += pad
    ymin -= pad
    ymax += pad

    if width >= height:
        nx = int(pixels)
        ny = max(2, int(round(pixels * height / width)))
    else:
        ny = int(pixels)
        nx = max(2, int(round(pixels * width / height)))

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    values, inside = sample_function_at_points(uh, np.column_stack((X.ravel(), Y.ravel())))
    Z = np.ma.array(values.reshape(Y.shape), mask=~inside.reshape(Y.shape))
    return X, Y, Z


def high_order_pyvista_grid(uh: fem.Function) -> pv.UnstructuredGrid:
    topology, cell_types, points = plot.vtk_mesh(uh.function_space)
    grid = pv.UnstructuredGrid(topology, cell_types, points)
    grid.point_data["u"] = np.asarray(uh.x.array.real, dtype=float)
    return grid


def save_matplotlib_panel(
    path: Path,
    case: GeometryCase,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ma.MaskedArray,
    clim: tuple[float, float],
    filled_levels: int,
    line_levels: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 6.8), dpi=320, constrained_layout=True)
    contour = ax.contourf(
        X,
        Y,
        Z,
        levels=np.linspace(clim[0], clim[1], filled_levels),
        cmap="jet",
        vmin=clim[0],
        vmax=clim[1],
        corner_mask=True,
    )
    ax.contour(
        X,
        Y,
        Z,
        levels=np.linspace(clim[0], clim[1], line_levels),
        colors="black",
        linewidths=0.24,
        alpha=0.62,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.018)
    cbar.set_label("u", rotation=0, labelpad=10)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_pyvista_panel(
    path: Path,
    grid: pv.UnstructuredGrid,
    clim: tuple[float, float],
    line_levels: int,
    camera_zoom: float,
) -> None:
    contours = grid.contour(isosurfaces=np.linspace(clim[0], clim[1], line_levels), scalars="u")
    plotter = pv.Plotter(off_screen=True, window_size=(2478, 1974))
    plotter.set_background("white")
    plotter.add_mesh(
        grid,
        scalars="u",
        cmap="jet",
        clim=clim,
        show_edges=False,
        scalar_bar_args={
            "title": "u",
            "vertical": True,
            "position_x": 0.89,
            "position_y": 0.08,
            "width": 0.055,
            "height": 0.84,
            "title_font_size": 42,
            "label_font_size": 34,
            "fmt": "%.5f",
        },
    )
    plotter.add_mesh(contours, color="black", line_width=1.6, opacity=0.72)
    plotter.view_xy()
    plotter.camera.parallel_projection = True
    plotter.camera.zoom(camera_zoom)
    plotter.enable_anti_aliasing("ssaa")
    plotter.screenshot(str(path), transparent_background=False)
    plotter.close()


def mesh_stats(domain) -> dict[str, int]:
    tdim = domain.topology.dim
    vertex_map = domain.topology.index_map(0)
    cell_map = domain.topology.index_map(tdim)
    return {
        "vertices": vertex_map.size_local + vertex_map.num_ghosts,
        "cells": cell_map.size_local + cell_map.num_ghosts,
    }


def run_case(
    case: GeometryCase,
    degree: int,
    output_dir: Path,
    filled_levels: int,
    line_levels: int,
    pixels: int,
    plotters: str,
) -> dict[str, object]:
    domain = create_domain(case)
    uh = solve_poisson(domain, degree, case.slug)
    stats = mesh_stats(domain)

    u_values = np.asarray(uh.x.array.real, dtype=float)
    clim = (0.0, float(np.max(u_values)))

    matplotlib_path = output_dir / f"{case.slug}_matplotlib_2d.png"
    pyvista_path = output_dir / f"{case.slug}_pyvista_2d.png"

    if plotters in ("both", "matplotlib"):
        X, Y, Z = dense_plot_data(uh, pixels)
        save_matplotlib_panel(matplotlib_path, case, X, Y, Z, clim, filled_levels, line_levels)
    if plotters in ("both", "pyvista"):
        grid = high_order_pyvista_grid(uh)
        camera_zoom = 1.06 if case.slug == "iter_wall" else 1.18
        save_pyvista_panel(pyvista_path, grid, clim, line_levels, camera_zoom)

    return {
        "case": case.slug,
        "title": case.title,
        "mesh_size": case.h,
        "mesh_path": None if case.mesh_path is None else str(case.mesh_path),
        "degree": degree,
        "filled_levels": filled_levels,
        "line_levels": line_levels,
        "plot_pixels": pixels,
        "vertices": stats["vertices"],
        "cells": stats["cells"],
        "dofs": int(uh.function_space.dofmap.index_map.size_local * uh.function_space.dofmap.index_map_bs),
        "u_min": float(np.min(u_values)),
        "u_max": float(np.max(u_values)),
        "matplotlib_2d": str(matplotlib_path),
        "pyvista_2d": str(pyvista_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--filled-levels", type=int, default=320)
    parser.add_argument("--line-levels", type=int, default=36)
    parser.add_argument("--plot-pixels", type=int, default=950)
    parser.add_argument("--plotters", choices=("both", "matplotlib", "pyvista"), default="both")
    parser.add_argument("--cases", nargs="+", choices=sorted(CASE_BY_SLUG), default=[case.slug for case in CASES])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "projects/diocotron/runs" / "poisson_torsion_geometries",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for slug in args.cases:
        case = CASE_BY_SLUG[slug]
        result = run_case(
            case,
            args.degree,
            args.output_dir,
            args.filled_levels,
            args.line_levels,
            args.plot_pixels,
            args.plotters,
        )
        results.append(result)
        print(
            f"{case.slug}: cells={result['cells']} dofs={result['dofs']} "
            f"u_max={result['u_max']:.12e} -> {result['matplotlib_2d']} and {result['pyvista_2d']}",
            flush=True,
        )

    metadata_path = args.output_dir / "poisson_figure_metadata.json"
    if metadata_path.exists():
        metadata_by_case = {
            entry["case"]: entry
            for entry in json.loads(metadata_path.read_text(encoding="utf-8"))
        }
    else:
        metadata_by_case = {}
    metadata_by_case.update({entry["case"]: entry for entry in results})
    metadata = [metadata_by_case[case.slug] for case in CASES if case.slug in metadata_by_case]
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"metadata: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
