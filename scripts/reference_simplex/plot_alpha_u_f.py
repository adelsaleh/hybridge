"""Solve ``alpha u = f`` by HDG on the reference simplex alone and plot ``u`` against ``u_h`` in 3D.

Usage::

    python scripts/reference_simplex/plot_alpha_u_f.py [--problem FILE] [--degrees 1 2 4 8] [--dim 3]
        [--backend matplotlib|pyvista] [--basis orthonormal] [--trace lagrange_warp_blend]
        [--tau 1] [--extra-degree 10] [--out DIR] [--no-show]

``FILE`` is a Python module that defines ``alpha`` and ``f`` (and optionally
``DIM``, ``DEGREES``, ``TAU``); the default is
``scripts/reference_simplex/alpha_u_f_problem.py``, which documents the
format. There is no mesh: the domain is the reference triangle (``dim = 2``)
or the reference tetrahedron (``dim = 3``), and ``u_h`` is one polynomial of
degree ``p``. For each degree ``p`` the script

1. builds ``ReferenceElementData`` of degree ``p``;
2. forms the local HDG blocks of
   ``(alpha u, v)_K + <tau (u - uhat), v>_dK = (f, v)_K``, with the volume mass
   ``int alpha phi_i phi_j`` and the load ``int f phi_i`` computed on a rule exact
   for degree ``2p + extra_degree``;
3. takes the trace ``uhat`` on every face as the L2 projection of the exact
   solution ``u = f / alpha`` onto the trace space (all faces are boundary
   faces), and solves ``A u_h = F + B uhat``;
4. computes the L2 projection ``Pi_p u`` (best approximation in ``P_p``) and
   the errors ``||u - u_h||_L2``, ``||u - Pi_p u||_L2`` and ``max |u - u_h|``.

The results are shown in one window (needs a display; ``--no-show`` skips it),
with a slider that selects the degree:

- ``--backend matplotlib`` (default): ``u``, ``u_h`` and ``u - u_h`` in three
  rotatable 3D views (horizontal cross-sections ``xi_3 = const`` of the
  tetrahedron, or surfaces ``z = u(x, y)`` over the triangle), ``u`` and ``u_h``
  along a segment through the centroid, and the errors against ``p``;
- ``--backend pyvista``: the same three fields on cross-sections of a
  tetrahedral mesh of the reference tetrahedron (warped surfaces on the
  triangle), with linked views.

Files are written only with ``--out DIR``: ``alpha_u_f.pdf`` (one page per
degree and a summary page), one PNG per page and ``errors.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "tests"))

from hybridge.core.quadrature import ReferenceElementData, _evaluate_basis  # noqa: E402
from hybridge.core.simplex.reference import simplex_volume_rule  # noqa: E402
from hybridge.io import reference_plots as rp  # noqa: E402
from hybridge.io.figures import publication_style  # noqa: E402
from simplex_hdg_helpers import reaction_blocks, trace_coefficients, volume_load, weighted_mass  # noqa: E402

DEFAULT_PROBLEM = Path(__file__).resolve().with_name("alpha_u_f_problem.py")
# Segment through the centroid used for the line plots: midpoints of two opposite
# edges of the tetrahedron, and from the midpoint of edge 0 to vertex 2 on the triangle.
SEGMENTS = {2: (np.array([0.0, -1.0]), np.array([-1.0, 1.0])),
            3: (np.array([0.0, -1.0, -1.0]), np.array([-1.0, 0.0, 0.0]))}


class Problem:
    """``alpha`` and ``f`` read from a problem module, with ``u = f / alpha``."""

    def __init__(self, path: Path):
        """Import the module at ``path`` and check that it defines ``alpha`` and ``f``."""
        spec = importlib.util.spec_from_file_location("alpha_u_f_problem_module", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot import a problem module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in ("alpha", "f"):
            if not hasattr(module, name):
                raise ValueError(f"the problem module {path} must define {name!r}")
        self.path = path
        self.module = module
        self.dim = int(getattr(module, "DIM", 3))
        self.degrees = tuple(int(p) for p in getattr(module, "DEGREES", (1, 2, 3, 4)))
        self.tau = float(getattr(module, "TAU", 1.0))

    def alpha(self, x: np.ndarray) -> np.ndarray:
        """Values of ``alpha`` at the points ``x`` ``(n, dim)``; raises ``ValueError`` if one is not positive."""
        source = self.module.alpha
        values = source(x) if callable(source) else source
        values = np.broadcast_to(np.asarray(values, dtype=np.float64), (x.shape[0],))
        if np.any(values <= 0.0):
            raise ValueError("alpha must be positive at every point; the local matrix is then positive definite")
        return values

    def f(self, x: np.ndarray) -> np.ndarray:
        """Values of ``f`` at the points ``x`` ``(n, dim)``."""
        return np.broadcast_to(np.asarray(self.module.f(x), dtype=np.float64), (x.shape[0],))

    def u(self, x: np.ndarray) -> np.ndarray:
        """Exact solution ``u = f / alpha`` at the points ``x``."""
        return self.f(x) / self.alpha(x)


def solve(problem: Problem, dim: int, order: int, *, basis: str, trace: str, tau: float, extra_degree: int):
    """Solve ``alpha u = f`` by HDG on the reference simplex with ``u_h`` of degree ``order``.

    Returns a dictionary with the reference data, the coefficients of ``u_h`` and
    of the L2 projection ``Pi_p u`` in the volume basis, and the L2 errors of both
    measured on a rule exact for degree ``2 order + extra_degree``.
    """
    degree = 2 * order + extra_degree
    if dim == 2:
        reference = ReferenceElementData.triangle(order, basis_type=basis, face_degree=degree)
    else:
        reference = ReferenceElementData.tetrahedron(order, basis_type=basis, face_degree=degree)
    rule = simplex_volume_rule(dim, degree)
    alpha_mass = weighted_mass(reference, problem.alpha, rule)
    a, b, _, _ = reaction_blocks(reference, trace, alpha=0.0, tau=tau, alpha_mass=alpha_mass)
    load = volume_load(reference, problem.f, rule)
    uhat = trace_coefficients(reference, trace, problem.u)
    coefficients = np.linalg.solve(a, load + b @ uhat)
    projection = np.linalg.solve(weighted_mass(reference, 1.0, rule), volume_load(reference, problem.u, rule))
    points, weights = rule
    basis_values = np.asarray(_evaluate_basis(reference.basis_type, order, points), dtype=np.float64)
    exact = problem.u(points)
    error_hdg = float(np.sqrt(weights @ (exact - basis_values @ coefficients) ** 2))
    error_projection = float(np.sqrt(weights @ (exact - basis_values @ projection) ** 2))
    return {"reference": reference, "coefficients": coefficients, "projection": projection,
            "l2_error": error_hdg, "l2_projection_error": error_projection,
            "condition_A": float(np.linalg.cond(a))}


def evaluate(reference: ReferenceElementData, coefficients: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Values of the polynomial with ``coefficients`` in the basis of ``reference`` at ``points``."""
    return np.asarray(_evaluate_basis(reference.basis_type, reference.order, points), dtype=np.float64) @ coefficients


def triangle_lattice(corners: np.ndarray, subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Points ``(n, k)`` and triangles ``(m, 3)`` of a uniform lattice of the triangle with the three ``corners``."""
    a, b = np.meshgrid(np.arange(subdivisions + 1), np.arange(subdivisions + 1), indexing="ij")
    keep = (a + b) <= subdivisions
    a, b = a[keep], b[keep]
    index = -np.ones((subdivisions + 1, subdivisions + 1), dtype=np.int64)
    index[a, b] = np.arange(a.size)
    triangles = []
    for i in range(subdivisions):
        for j in range(subdivisions - i):
            triangles.append((index[i, j], index[i + 1, j], index[i, j + 1]))
            if i + j + 1 < subdivisions:
                triangles.append((index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]))
    s, t = a / subdivisions, b / subdivisions
    points = corners[0] + s[:, None] * (corners[1] - corners[0]) + t[:, None] * (corners[2] - corners[0])
    return points, np.asarray(triangles, dtype=np.int64)


def slice_corners(height: float) -> np.ndarray:
    """Corners of the cross-section ``xi_3 = height`` of the reference tetrahedron, shape ``(3, 3)``."""
    c = float(height)
    return np.array([[-1.0, -1.0, c], [-c, -1.0, c], [-1.0, -c, c]])


def color_scale(values: np.ndarray, *, signed: bool):
    """Colormap and normalization: diverging and symmetric when ``signed`` or when values change sign, else the blue ramp."""
    from matplotlib.colors import Normalize

    low, high = float(np.min(values)), float(np.max(values))
    if signed or (low < 0.0 < high):
        bound = max(abs(low), abs(high), 1e-300)
        return rp.diverging_colormap(), Normalize(-bound, bound)
    return rp.sequential_colormap(), Normalize(low, high if high > low else low + 1e-12)


def draw_tetrahedron_slices(ax, field, heights, cmap, norm, subdivisions: int = 24) -> None:
    """Draw ``field`` (callable on points ``(n, 3)``) on horizontal cross-sections of the reference tetrahedron."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    rp.draw_simplex_outline(ax, np.array([[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]),
                            ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)))
    for height in heights:
        points, triangles = triangle_lattice(slice_corners(height), subdivisions)
        values = field(points)
        colors = cmap(norm(values[triangles].mean(axis=1)))
        ax.add_collection3d(Poly3DCollection(points[triangles], facecolors=colors, edgecolors="none", alpha=0.92))
    labels = {"A": (-1.0, -1.0, -1.0), "B": (1.0, -1.0, -1.0), "C": (-1.0, 1.0, -1.0), "D": (-1.0, -1.0, 1.0)}
    for text, position in labels.items():
        ax.text(*position, " " + text, fontsize=7, color=rp.TEXT_SECONDARY)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    # Seen from outside the slanted face (normal (1, 1, 1)), above the cross-sections.
    ax.view_init(elev=28, azim=35)
    ax.set_axis_off()


def draw_triangle_surface(ax, field, cmap, norm, subdivisions: int = 40) -> None:
    """Draw the surface ``z = field(x, y)`` over the reference triangle, colored by ``z``."""
    points, triangles = triangle_lattice(np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]]), subdivisions)
    values = field(points)
    surface = ax.plot_trisurf(points[:, 0], points[:, 1], values, triangles=triangles, cmap=cmap, norm=norm,
                              linewidth=0.0, antialiased=True)
    surface.set_array(values[triangles].mean(axis=1))
    ax.view_init(elev=28, azim=-62)
    ax.set_xlabel("xi_1", labelpad=-8)
    ax.set_ylabel("xi_2", labelpad=-8)
    ax.tick_params(labelsize=6, pad=-2)


def segment_points(dim: int, count: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Parameter ``s`` in ``[0, 1]`` and points of the segment of ``SEGMENTS[dim]``."""
    start, end = SEGMENTS[dim]
    s = np.linspace(0.0, 1.0, count)
    return s, start + s[:, None] * (end - start)


def degree_page(problem: Problem, dim: int, order: int, result: dict, heights):
    """Figure for one degree: ``u``, ``u_h`` and ``u - u_h`` in 3D, and both along a segment."""
    from matplotlib.cm import ScalarMappable

    reference, coefficients = result["reference"], result["coefficients"]

    def u_h(points):
        """Values of the HDG solution ``u_h``."""
        return evaluate(reference, coefficients, points)

    def error(points):
        """Values of ``u - u_h``."""
        return problem.u(points) - u_h(points)

    if dim == 3:
        probe = np.vstack([triangle_lattice(slice_corners(h), 12)[0] for h in heights])
    else:
        probe = triangle_lattice(np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]]), 40)[0]
    cmap, norm = color_scale(problem.u(probe), signed=False)
    error_cmap, error_norm = color_scale(error(probe), signed=True)
    figure = rp.new_figure(15.0, 4.6)
    titles = ("exact u = f / alpha", f"HDG u_h, p = {order}", "error u - u_h")
    fields = (problem.u, u_h, error)
    axes = []
    for column, (title, field) in enumerate(zip(titles, fields)):
        ax = figure.add_subplot(1, 4, column + 1, projection="3d")
        scale = (error_cmap, error_norm) if column == 2 else (cmap, norm)
        if dim == 3:
            draw_tetrahedron_slices(ax, field, heights, *scale)
        else:
            draw_triangle_surface(ax, field, *scale)
        ax.set_title(title)
        axes.append(ax)
    figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes[:2], shrink=0.6, label="u, u_h")
    figure.colorbar(ScalarMappable(norm=error_norm, cmap=error_cmap), ax=axes[2], shrink=0.6, label="u - u_h")
    s, points = segment_points(dim)
    ax = figure.add_subplot(1, 4, 4)
    rp.plot_series(ax, {"u": s, "u_h": s}, {"u": problem.u(points), "u_h": u_h(points)}, logy=False,
                   xlabel="position on the segment", ylabel="value",
                   title=f"L2 error {result['l2_error']:.2e} (best {result['l2_projection_error']:.2e})")
    for line in ax.lines:
        line.set_marker("")
    ax.lines[0].set_linewidth(3.0)
    ax.legend(frameon=False)
    name = "tetrahedron" if dim == 3 else "triangle"
    figure.suptitle(f"alpha u = f on the reference {name}, degree p = {order} (tau = {problem.tau:g})")
    return figure


def summary_page(problem: Problem, dim: int, degrees, results):
    """Figure with the errors against the degree and ``u_h`` of several degrees along the segment."""
    figure = rp.new_figure(11.0, 4.2)
    ax = figure.add_subplot(1, 2, 1)
    rp.plot_series(ax, list(degrees), {
        "||u - u_h|| (HDG)": [results[p]["l2_error"] for p in degrees],
        "||u - Pi_p u|| (best)": [results[p]["l2_projection_error"] for p in degrees],
        "max |u - u_h|": [results[p]["max_error"] for p in degrees],
    }, xlabel="degree p of u_h", ylabel="error", title="Errors against the degree")
    shown = list(degrees) if len(degrees) <= 7 else [degrees[int(round(k))] for k in np.linspace(0, len(degrees) - 1, 7)]
    s, points = segment_points(dim)
    series = {"u": problem.u(points)}
    for p in shown:
        series[f"u_h, p={p}"] = evaluate(results[p]["reference"], results[p]["coefficients"], points)
    ax = figure.add_subplot(1, 2, 2)
    rp.plot_series(ax, s, series, logy=False, xlabel="position on the segment", ylabel="value",
                   title="u and u_h along the segment through the centroid")
    for line in ax.lines:
        line.set_marker("")
    ax.lines[0].set_linewidth(3.0)
    ax.legend(frameon=False)
    figure.suptitle(f"alpha u = f: problem {problem.path.name}")
    return figure


def write_files(problem: Problem, dim: int, degrees, results: dict, args) -> None:
    """Write ``alpha_u_f.pdf`` (a page per degree and a summary), the PNG pages and ``errors.json`` to ``args.out``."""
    from matplotlib.backends.backend_pdf import PdfPages

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with publication_style(), PdfPages(out / "alpha_u_f.pdf", metadata={"Creator": "HYBRIDGE"}) as pdf:
        figures = [degree_page(problem, dim, p, results[p], args.heights) for p in degrees]
        figures.append(summary_page(problem, dim, degrees, results))
        for index, figure in enumerate(figures):
            pdf.savefig(figure)
            name = f"degree-{degrees[index]:02d}.png" if index < len(degrees) else "summary.png"
            figure.savefig(out / name, dpi=140)
    record = {"problem": str(problem.path), "dim": dim, "basis": args.basis, "trace": args.trace, "tau": problem.tau,
              "extra_degree": args.extra_degree,
              "degrees": {str(p): {k: v for k, v in results[p].items() if isinstance(v, float)} for p in degrees}}
    with open(out / "errors.json", "w") as handle:
        json.dump(record, handle, indent=1)
    print(f"wrote {out / 'alpha_u_f.pdf'}, {len(figures)} PNG pages and errors.json")


def fields_for_degree(problem: Problem, result: dict):
    """Callables ``(u, u_h, u - u_h)`` on points ``(n, dim)`` for one solved degree."""
    reference, coefficients = result["reference"], result["coefficients"]

    def u_h(points):
        """Values of the HDG solution ``u_h``."""
        return evaluate(reference, coefficients, points)

    def error(points):
        """Values of ``u - u_h``."""
        return problem.u(points) - u_h(points)

    return problem.u, u_h, error


def show_matplotlib(problem: Problem, dim: int, degrees, results: dict, heights) -> None:
    """Open one window: three rotatable 3D views, the segment plot, the errors against p and a degree slider."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.widgets import Slider

    if dim == 3:
        probe = np.vstack([triangle_lattice(slice_corners(h), 12)[0] for h in heights])
    else:
        probe = triangle_lattice(np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]]), 40)[0]
    cmap, norm = color_scale(problem.u(probe), signed=False)
    figure = plt.figure(figsize=(15.0, 8.8), layout="constrained")
    figure.canvas.manager.set_window_title(f"alpha u = f: {problem.path.name}")
    grid = figure.add_gridspec(3, 3, height_ratios=(1.0, 0.75, 0.06))
    views = [figure.add_subplot(grid[0, k], projection="3d") for k in range(3)]
    if hasattr(views[0], "shareview"):
        for ax in views[1:]:
            ax.shareview(views[0])
    figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=views[:2], shrink=0.6, label="u, u_h")
    error_bar = figure.colorbar(ScalarMappable(norm=norm, cmap=rp.diverging_colormap()), ax=views[2], shrink=0.6,
                                label="u - u_h")
    segment_ax = figure.add_subplot(grid[1, 0:2])
    errors_ax = figure.add_subplot(grid[1, 2])
    slider_ax = figure.add_subplot(grid[2, :])
    s, segment = segment_points(dim)
    rp.plot_series(errors_ax, list(degrees), {
        "||u - u_h|| (HDG)": [results[p]["l2_error"] for p in degrees],
        "||u - Pi_p u|| (best)": [results[p]["l2_projection_error"] for p in degrees],
        "max |u - u_h|": [results[p]["max_error"] for p in degrees],
    }, xlabel="degree p", ylabel="error", title="Errors against the degree")
    marker = errors_ax.axvline(degrees[0], color=rp.TEXT_SECONDARY, linewidth=1.0, linestyle="--")
    rp.plot_series(segment_ax, s, {"u": problem.u(segment), "u_h": problem.u(segment)}, logy=False,
                   xlabel="position on the segment through the centroid", ylabel="value")
    for line in segment_ax.lines:
        line.set_marker("")
    segment_ax.lines[0].set_linewidth(3.0)
    segment_ax.legend(frameon=False)

    def draw(index: int) -> None:
        """Draw the three views and the segment plot for ``degrees[index]``."""
        p = degrees[index]
        u, u_h, error = fields_for_degree(problem, results[p])
        error_cmap, error_norm = color_scale(error(probe), signed=True)
        error_bar.mappable.set_norm(error_norm)
        error_bar.update_normal(error_bar.mappable)
        for ax, field, scale, title in zip(
                views, (u, u_h, error), ((cmap, norm), (cmap, norm), (error_cmap, error_norm)),
                ("exact u = f / alpha", f"HDG u_h, p = {p}", f"error u - u_h, p = {p}")):
            elev, azim = ax.elev, ax.azim
            ax.cla()
            if dim == 3:
                draw_tetrahedron_slices(ax, field, heights, *scale)
            else:
                draw_triangle_surface(ax, field, *scale)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(title)
        segment_ax.lines[1].set_ydata(u_h(segment))
        segment_ax.relim()
        segment_ax.autoscale_view()
        segment_ax.set_title(f"p = {p}: L2 error {results[p]['l2_error']:.2e} "
                             f"(best {results[p]['l2_projection_error']:.2e}), max {results[p]['max_error']:.2e}")
        marker.set_xdata([p, p])
        figure.canvas.draw_idle()

    slider = Slider(slider_ax, "degree", 0, len(degrees) - 1, valinit=0, valstep=1)
    slider.valtext.set_text(f"p = {degrees[0]}")

    def on_change(value) -> None:
        """Redraw for the degree chosen with the slider."""
        index = int(round(value))
        slider.valtext.set_text(f"p = {degrees[index]}")
        draw(index)

    slider.on_changed(on_change)
    for ax in views:
        ax.view_init(elev=28, azim=35) if dim == 3 else ax.view_init(elev=28, azim=-62)
    draw(0)
    name = "tetrahedron" if dim == 3 else "triangle"
    figure.suptitle(f"alpha u = f on the reference {name} ({problem.path.name}, tau = {problem.tau:g}); "
                    "drag a 3D view to rotate, move the slider to change p")
    plt.show()


def show_pyvista(problem: Problem, dim: int, degrees, results: dict, heights) -> None:
    """Open one PyVista window with u, u_h and u - u_h side by side (linked views) and a degree slider."""
    import pyvista as pv

    if dim == 3:
        corners = np.array([[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
        n = 28
        lattice = np.array([(i, j, k) for i in range(n + 1) for j in range(n + 1 - i) for k in range(n + 1 - i - j)])
        points = -1.0 + 2.0 * lattice / n
        mesh = pv.PolyData(points).delaunay_3d()
        coordinates = np.asarray(mesh.points, dtype=np.float64)
        edges = pv.PolyData(corners, lines=np.hstack([[2, a, b] for a, b in
                                                     ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))]))
    else:
        points, triangles = triangle_lattice(np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]]), 48)
        faces = np.hstack([np.full((len(triangles), 1), 3), triangles]).ravel()
        mesh = pv.PolyData(np.column_stack((points, np.zeros(len(points)))), faces)
        coordinates = np.asarray(mesh.points, dtype=np.float64)[:, :2]
        edges = None
    exact = problem.u(coordinates)
    bound = float(np.abs(exact).max())
    plotter = pv.Plotter(shape=(1, 3), window_size=(1650, 650), title=f"alpha u = f: {problem.path.name}")
    plotter.set_background("white")
    cmap = rp.sequential_colormap() if exact.min() >= 0.0 else rp.diverging_colormap()
    u_limits = (float(exact.min()), float(exact.max())) if exact.min() >= 0.0 else (-bound, bound)

    def show_field(column: int, name: str, values: np.ndarray, colormap, limits, title: str) -> None:
        """Draw ``values`` in subplot ``column`` (cross-sections in 3D, a warped surface in 2D)."""
        plotter.subplot(0, column)
        grid = mesh.copy()
        grid.point_data[name] = values
        if dim == 3:
            shown = grid.slice_along_axis(n=len(heights), axis="z", bounds=(-1, 1, -1, 1, min(heights), max(heights)))
            plotter.add_mesh(edges, color="#52514e", line_width=2, name=f"edges{column}")
        else:
            scale = 1.0 / max(np.abs(values).max(), 1e-300)
            shown = grid.warp_by_scalar(name, factor=scale)
        plotter.add_mesh(shown, scalars=name, cmap=colormap, clim=limits, name=f"field{column}", lighting=False,
                         scalar_bar_args={"title": name, "color": "black", "vertical": True, "fmt": "%.3g",
                                          "position_x": 0.86, "width": 0.07, "label_font_size": 12})
        if dim == 3:
            plotter.add_point_labels(edges.points, ["A", "B", "C", "D"], font_size=12, text_color="black",
                                     shape=None, show_points=False, name=f"labels{column}")
        plotter.add_text(title, font_size=10, color="black", name=f"title{column}")

    slider = {}

    def update(value) -> None:
        """Redraw u_h and u - u_h for the degree nearest to the slider value; the slider snaps to that degree."""
        p = min(degrees, key=lambda degree: abs(degree - value))
        if "widget" in slider:
            slider["widget"].GetRepresentation().SetValue(p)
        _, u_h, error = fields_for_degree(problem, results[p])
        values = u_h(coordinates)
        errors = exact - values
        error_bound = max(float(np.abs(errors).max()), 1e-300)
        show_field(1, "u_h", values, cmap, u_limits, f"HDG u_h, p = {p}: L2 error {results[p]['l2_error']:.2e}")
        show_field(2, "u - u_h", errors, rp.diverging_colormap(), (-error_bound, error_bound),
                   f"error u - u_h, p = {p} (max {results[p]['max_error']:.2e})")
        plotter.render()

    show_field(0, "u", exact, cmap, u_limits, "exact u = f / alpha")
    update(degrees[0])
    plotter.subplot(0, 1)
    if len(degrees) > 1:
        slider["widget"] = plotter.add_slider_widget(
            update, rng=[degrees[0], degrees[-1]], value=degrees[0], title="degree p", fmt="%.0f",
            pointa=(0.1, 0.08), pointb=(0.9, 0.08), color="black")
    plotter.link_views()
    plotter.view_vector((1.0, 0.75, 0.75)) if dim == 3 else plotter.view_vector((0.3, -1.0, 0.9))
    plotter.show()


def main(argv=None) -> int:
    """Parse the options, solve for every degree, write the figures and ``errors.json``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--problem", type=Path, default=DEFAULT_PROBLEM, help="module defining alpha and f")
    parser.add_argument("--degrees", type=int, nargs="+", help="degrees p of u_h (default: DEGREES of the module)")
    parser.add_argument("--dim", type=int, choices=(2, 3), help="2: triangle, 3: tetrahedron (default: DIM)")
    parser.add_argument("--basis", default="orthonormal", help="volume basis of u_h")
    parser.add_argument("--trace", default="lagrange_warp_blend", help="trace basis on the faces")
    parser.add_argument("--tau", type=float, help="stabilization tau > 0 (default: TAU of the module)")
    parser.add_argument("--extra-degree", type=int, default=10,
                        help="rules are exact for degree 2p + this, for non-polynomial alpha and f")
    parser.add_argument("--heights", type=float, nargs="+", default=(-0.9, -0.55, -0.2, 0.15),
                        help="xi_3 of the cross-sections drawn on the tetrahedron")
    parser.add_argument("--backend", choices=("matplotlib", "pyvista"), default="matplotlib",
                        help="window: matplotlib 3D axes or a PyVista render window")
    parser.add_argument("--out", type=Path, help="also write alpha_u_f.pdf, PNG pages and errors.json here")
    parser.add_argument("--no-show", action="store_true", help="do not open a window")
    args = parser.parse_args(argv)

    problem = Problem(args.problem.resolve())
    dim = args.dim or problem.dim
    degrees = tuple(sorted(set(args.degrees or problem.degrees)))
    if args.tau is not None:
        problem.tau = args.tau
    if problem.tau <= 0.0:
        raise ValueError("tau must be positive")
    show = not args.no_show
    if show and not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        if args.out is None:
            raise SystemExit("no display (DISPLAY is not set): pass --out DIR to write the figures instead")
        print("no display (DISPLAY is not set): writing files only")
        show = False

    results = {}
    probe = np.vstack([segment_points(dim, 400)[1],
                       simplex_volume_rule(dim, 2 * max(degrees) + args.extra_degree)[0]])
    for p in degrees:
        start = time.perf_counter()
        results[p] = solve(problem, dim, p, basis=args.basis, trace=args.trace, tau=problem.tau,
                           extra_degree=args.extra_degree)
        results[p]["max_error"] = float(np.abs(problem.u(probe) - evaluate(results[p]["reference"],
                                                                             results[p]["coefficients"], probe)).max())
        print(f"p = {p:2d}: ||u - u_h|| = {results[p]['l2_error']:.3e}, ||u - Pi_p u|| = "
              f"{results[p]['l2_projection_error']:.3e}, max |u - u_h| = {results[p]['max_error']:.3e}, "
              f"cond(A) = {results[p]['condition_A']:.2e} ({time.perf_counter() - start:.2f} s)", flush=True)

    if args.out is not None:
        write_files(problem, dim, degrees, results, args)
    if show:
        viewer = show_pyvista if args.backend == "pyvista" else show_matplotlib
        viewer(problem, dim, degrees, results, args.heights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
