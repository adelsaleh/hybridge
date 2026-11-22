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
import scipy.linalg

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(REPOSITORY / "tests"))

from hybridge.core.quadrature import ReferenceElementData, _evaluate_basis, _evaluate_simplex_basis  # noqa: E402
from hybridge.core.simplex.reference import reference_simplex, simplex_volume_rule  # noqa: E402
from hybridge.io import reference_plots as rp  # noqa: E402
from hybridge.io.figures import publication_style  # noqa: E402
from simplex_hdg_helpers import (  # noqa: E402
    basis_blocks, reaction_blocks, trace_coefficients, volume_load, weighted_mass,
)

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
    of the L2 projection ``Pi_p u`` in the volume basis, the L2 errors of both
    measured on a rule exact for degree ``2 order + extra_degree``, and
    ``max_error``, the largest ``|u - u_h|`` at the points of that rule and of the
    segment of the line plots. The volume integrals and the errors are computed
    block by block over the points of the rule (:func:`basis_blocks`), so the
    basis is never tabulated on the whole rule at once. ``A`` is symmetric
    positive definite (``alpha > 0``, ``tau > 0``) and is solved by Cholesky.
    """
    degree = 2 * order + extra_degree
    if dim == 2:
        reference = ReferenceElementData.triangle(order, basis_type=basis, face_degree=degree)
    else:
        reference = ReferenceElementData.tetrahedron(order, basis_type=basis, face_degree=degree)
    rule = simplex_volume_rule(dim, degree)
    alpha_mass = weighted_mass(reference, problem.alpha, rule)
    a, b, _, _ = reaction_blocks(reference, trace, alpha=0.0, tau=tau, alpha_mass=alpha_mass)
    loads = volume_load(reference, lambda x: np.column_stack((problem.f(x), problem.u(x))), rule)
    uhat = trace_coefficients(reference, trace, problem.u)
    factor = scipy.linalg.cho_factor(a)
    coefficients = scipy.linalg.cho_solve(factor, loads[:, 0] + b @ uhat)
    projection = scipy.linalg.cho_solve(scipy.linalg.cho_factor(reference.exact_mass.astype(np.float64)), loads[:, 1])
    both = np.column_stack((coefficients, projection))
    squares = np.zeros(2)
    segment = segment_points(dim, 400)[1]
    max_error = float(np.abs(problem.u(segment) - evaluate(reference, coefficients, segment)).max())
    for points, weights, values in basis_blocks(reference, rule):
        errors = problem.u(points)[:, None] - values @ both
        squares += weights @ errors**2
        max_error = max(max_error, float(np.abs(errors[:, 0]).max()))
    error_hdg, error_projection = (float(value) for value in np.sqrt(squares))
    result = {"reference": reference, "coefficients": coefficients, "projection": projection, "uhat": uhat,
              "trace": trace, "l2_error": error_hdg, "l2_projection_error": error_projection,
              "max_error": max_error, "condition_A": spd_condition_number(a)}
    face_points, face_weights = reference._generic_face_rule
    trace_squares = volume_squares = 0.0
    for f in range(reference.num_faces):
        x, uhat_values = trace_values(result, dim, f, face_points)
        exact_values = problem.u(x)
        ratio = reference.face_measure_ratios[f]
        trace_squares += ratio * face_weights @ (exact_values - uhat_values) ** 2
        volume_squares += ratio * face_weights @ (exact_values - evaluate(reference, coefficients, x)) ** 2
    result["trace_l2_error"] = float(np.sqrt(trace_squares))
    result["face_l2_error"] = float(np.sqrt(volume_squares))
    return result


def spd_condition_number(matrix: np.ndarray) -> float:
    """Return the 2-norm condition number ``lambda_max / lambda_min`` of a symmetric positive definite matrix.

    All eigenvalues are computed (``scipy.linalg.eigh``, ``O(n^3)``: 0.4 s for
    ``n = 2925``, ``p = 24``). A Lanczos estimate of the two extreme eigenvalues
    looks cheaper but is not: the smallest eigenvalues of ``A`` are clustered
    (the interior modes all lie near ``alpha lambda_min(M)``), and it took 37 s at
    ``p = 24``.
    """
    eigenvalues = scipy.linalg.eigh(matrix, eigvals_only=True)
    return float(eigenvalues[-1] / eigenvalues[0])


def trace_values(result: dict, dim: int, face: int, face_points: np.ndarray):
    """Points ``x`` of face ``face`` and the values of the trace ``uhat_h`` there.

    ``face_points`` are coordinates on the reference face, shape ``(n, dim - 1)``;
    ``x = F_face(s)`` has shape ``(n, dim)``. The trace on face ``f`` is
    ``sum_a uhat[f, a] mu_a(s)`` in the face's own vertex order (orientation 0).
    """
    reference = result["reference"]
    face_dof = reference.face_dof
    mu = np.asarray(_evaluate_simplex_basis(result["trace"], dim - 1, reference.order, face_points, gradients=False),
                    dtype=np.float64)
    x = reference.simplex.face_map(face, face_points)
    return x, mu @ result["uhat"][face * face_dof:(face + 1) * face_dof]


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


def draw_tetrahedron_slices(ax, field, heights, cmap, norm, subdivisions: int = 24, *, faces: bool = False,
                            face_opacity: float = 0.4) -> None:
    """Draw ``field`` (callable on points ``(n, 3)``) on horizontal cross-sections of the reference tetrahedron.

    With ``faces`` the same field is also drawn on the four faces, with opacity
    ``face_opacity`` so that the cross-sections stay visible, and the faces are
    labelled. Cross-sections and faces form one collection, so all their
    triangles are sorted by depth together and the picture stays right when the
    view is rotated.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    polygons, colors = [], []
    for height in heights:
        points, triangles = triangle_lattice(slice_corners(height), subdivisions)
        values = field(points)
        polygons.append(points[triangles])
        rgba = cmap(norm(values[triangles].mean(axis=1)))
        rgba[:, 3] = 0.95
        colors.append(rgba)
    if faces:
        samples, triangles = face_samples(3, subdivisions)
        for f in range(4):
            points = reference_simplex(3).face_map(f, samples)
            values = field(points)
            polygons.append(points[triangles])
            rgba = cmap(norm(values[triangles].mean(axis=1)))
            rgba[:, 3] = face_opacity
            colors.append(rgba)
    ax.computed_zorder = False
    ax.add_collection3d(Poly3DCollection(np.concatenate(polygons), facecolors=np.concatenate(colors),
                                         edgecolors="none", zorder=1))
    rp.draw_simplex_outline(ax, TETRAHEDRON_VERTICES, ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)))
    for name, position in zip(VERTEX_NAMES, TETRAHEDRON_VERTICES):
        ax.text(*position, " " + name, fontsize=7, color=rp.TEXT_SECONDARY, zorder=5)
    if faces:
        for name, position in zip(face_names(3), face_label_positions(3)):
            ax.text(*position, name, fontsize=8, color="black", ha="center", va="center", zorder=10,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    # Seen from outside the slanted face (normal (1, 1, 1)), above the cross-sections.
    ax.view_init(elev=28, azim=35)
    ax.set_axis_off()


def draw_triangle_surface(ax, field, cmap, norm, subdivisions: int = 40, *, faces: bool = False) -> None:
    """Draw the surface ``z = field(x, y)`` over the reference triangle, colored by ``z``.

    With ``faces`` the values on the three faces (edges) are drawn as labelled
    curves along the boundary of the surface.
    """
    points, triangles = triangle_lattice(REFERENCE_TRIANGLE, subdivisions)
    values = field(points)
    surface = ax.plot_trisurf(points[:, 0], points[:, 1], values, triangles=triangles, cmap=cmap, norm=norm,
                              linewidth=0.0, antialiased=True, alpha=0.85 if faces else 1.0)
    surface.set_array(values[triangles].mean(axis=1))
    if faces:
        samples, _ = face_samples(2)
        for f, name in enumerate(face_names(2)):
            x = reference_simplex(2).face_map(f, samples)
            edge_values = field(x)
            ax.plot(x[:, 0], x[:, 1], edge_values, color=rp.CATEGORICAL[f], linewidth=2.5, zorder=10)
            middle = len(edge_values) // 2
            ax.text(x[middle, 0], x[middle, 1], edge_values[middle], " " + name, fontsize=8, color="black", zorder=11)
    ax.view_init(elev=28, azim=-62)
    ax.set_xlabel("xi_1", labelpad=-8)
    ax.set_ylabel("xi_2", labelpad=-8)
    ax.tick_params(labelsize=6, pad=-2)


def segment_points(dim: int, count: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Parameter ``s`` in ``[0, 1]`` and points of the segment of ``SEGMENTS[dim]``."""
    start, end = SEGMENTS[dim]
    s = np.linspace(0.0, 1.0, count)
    return s, start + s[:, None] * (end - start)


REFERENCE_TRIANGLE = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
TETRAHEDRON_VERTICES = np.array([[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
VERTEX_NAMES = ("A", "B", "C", "D")


def face_names(dim: int) -> list[str]:
    """Labels ``"F<f> <vertices>"`` of the faces, in the numbering of ``reference_simplex(dim).face_vertices``.

    Tetrahedron: face ``f`` is opposite vertex ``f`` (F0 BCD, F1 ACD, F2 ABD, F3 ABC).
    Triangle: face ``f`` is the edge from vertex ``f`` to vertex ``f + 1`` (F0 AB, F1 BC, F2 CA).
    """
    return [f"F{f} " + "".join(VERTEX_NAMES[v] for v in vertices)
            for f, vertices in enumerate(reference_simplex(dim).face_vertices)]


def face_label_positions(dim: int, offset: float = 0.15) -> np.ndarray:
    """Positions ``(dim + 1, dim)`` for face labels: face centroids moved ``offset`` outward along the face normals."""
    simplex = reference_simplex(dim)
    centroids = np.array([simplex.vertices[list(vertices)].mean(axis=0) for vertices in simplex.face_vertices])
    return centroids + offset * np.asarray(simplex.outward_normals)


def face_samples(dim: int, subdivisions: int = 24):
    """Sample points on the reference face: a triangle lattice ``(points (n, 2), triangles)`` in 3D, a line in 2D."""
    if dim == 3:
        return triangle_lattice(REFERENCE_TRIANGLE, subdivisions)
    return np.linspace(-1.0, 1.0, 201)[:, None], None


FACE_FIELDS = {"volume": "u - u_h", "trace": "u - uhat_h"}


def face_errors(problem: Problem, result: dict, dim: int, subdivisions: int = 24, kind: str = "trace"):
    """For every face ``f``: points ``x``, reference-face points ``s`` and an error at ``x``.

    ``kind = "trace"`` gives ``u - uhat_h`` (the trace unknown of face ``f``);
    ``kind = "volume"`` gives ``u - u_h`` (the element polynomial restricted to the face).
    """
    samples, _ = face_samples(dim, subdivisions)
    out = []
    for f in range(dim + 1):
        x, uhat_values = trace_values(result, dim, f, samples)
        approximation = uhat_values if kind == "trace" else evaluate(result["reference"], result["coefficients"], x)
        out.append((x, samples, problem.u(x) - approximation))
    return out


def draw_faces_3d(ax, problem: Problem, result: dict, dim: int, cmap, norm, kind: str = "trace") -> None:
    """Draw an error on the faces (see :func:`face_errors`): colored triangles on the tetrahedron, curves on the triangle."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if dim == 3:
        _, triangles = face_samples(dim)
        polygons, colors = [], []
        for x, _, values in face_errors(problem, result, dim, kind=kind):
            polygons.append(x[triangles])
            colors.append(cmap(norm(values[triangles].mean(axis=1))))
        # One collection, so that its triangles are sorted by depth together; the labels are drawn
        # after it (computed_zorder off), so they stay readable from every direction.
        ax.computed_zorder = False
        ax.add_collection3d(Poly3DCollection(np.concatenate(polygons), facecolors=np.concatenate(colors),
                                             edgecolors="none", alpha=0.97, zorder=1))
        rp.draw_simplex_outline(ax, TETRAHEDRON_VERTICES, ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)))
        for name, position in zip(VERTEX_NAMES, TETRAHEDRON_VERTICES):
            ax.text(*position, " " + name, fontsize=7, color=rp.TEXT_SECONDARY, zorder=5)
        for name, position in zip(face_names(3), face_label_positions(3)):
            ax.text(*position, name, fontsize=8, color="black", ha="center", va="center", zorder=10,
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.8})
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        return
    closed = np.vstack((REFERENCE_TRIANGLE, REFERENCE_TRIANGLE[:1]))
    ax.plot(closed[:, 0], closed[:, 1], np.zeros(4), color=rp.OUTLINE, linewidth=0.8)
    for f, ((x, _, values), name) in enumerate(zip(face_errors(problem, result, dim, kind=kind), face_names(2))):
        ax.plot(x[:, 0], x[:, 1], values, color=rp.CATEGORICAL[f], linewidth=2.0)
        middle = len(values) // 2
        ax.text(x[middle, 0], x[middle, 1], values[middle], " " + name, fontsize=8, color="black")
    ax.set_xlabel("xi_1", labelpad=-8)
    ax.set_ylabel("xi_2", labelpad=-8)
    ax.tick_params(labelsize=6, pad=-2)


def draw_faces_flat(ax, problem: Problem, result: dict, dim: int, cmap, norm, kind: str = "trace") -> None:
    """Draw an error (see :func:`face_errors`) on all faces at once: the unfolded tetrahedron, or one curve per edge."""
    if dim == 3:
        face_vertices = reference_simplex(3).face_vertices
        _, net = rp.tetrahedron_net(face_vertices)
        _, triangles = face_samples(dim)
        triangle = reference_simplex(2)
        for f, (_, s, values) in enumerate(face_errors(problem, result, dim, kind=kind)):
            xy = triangle.barycentric(s) @ net[f]
            ax.tripcolor(xy[:, 0], xy[:, 1], triangles, facecolors=values[triangles].mean(axis=1), cmap=cmap,
                         norm=norm, shading="flat")
            closed = np.vstack((net[f], net[f][:1]))
            ax.plot(closed[:, 0], closed[:, 1], color=rp.OUTLINE, linewidth=0.8)
            for v, corner in zip(face_vertices[f], net[f]):
                ax.text(corner[0], corner[1], VERTEX_NAMES[v], fontsize=7, color=rp.TEXT_SECONDARY)
            centre = net[f].mean(axis=0)
            ax.text(centre[0], centre[1], face_names(3)[f], fontsize=8, color="black", ha="center", va="center",
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.75})
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"{FACE_FIELDS[kind]} on the unfolded faces")
        return
    edges = face_names(2)
    errors = face_errors(problem, result, dim, kind=kind)
    rp.plot_series(ax, {name: s[:, 0] for name, (_, s, _) in zip(edges, errors)},
                   {name: values for name, (_, _, values) in zip(edges, errors)}, logy=False,
                   xlabel="edge parameter t (from first to second vertex)", ylabel=FACE_FIELDS[kind],
                   title=f"{FACE_FIELDS[kind]} on the faces (edges)")
    ax.set_ylim(norm.vmin, norm.vmax)
    for line in ax.lines:
        line.set_marker("")
    ax.axhline(0.0, color=rp.TEXT_SECONDARY, linewidth=0.6)
    ax.legend(frameon=False)


def build_view(figure, problem: Problem, dim: int, degrees, results: dict, heights, *, slider: bool,
               face_opacity: float = 0.4):
    """Lay out the five 3D views, the segment, the flat faces and the errors on ``figure``; return ``draw(index)``.

    Top row: ``u``, ``u_h`` and ``u - u_h``, each on the cross-sections of the
    tetrahedron and on its faces in the same view (faces with opacity
    ``face_opacity``; on the triangle: a surface with its three edge curves), and
    ``u - uhat_h`` on the faces.
    Bottom row: ``u`` and ``u_h`` along the segment through the centroid,
    ``u - u_h`` and ``u - uhat_h`` on the unfolded faces, and the errors against ``p``. With ``slider`` a thin third row is left for the degree
    slider; the function returns ``(draw, slider_ax)``.
    """
    from matplotlib.cm import ScalarMappable

    if dim == 3:
        probe = np.vstack([triangle_lattice(slice_corners(h), 12)[0] for h in heights])
    else:
        probe = triangle_lattice(REFERENCE_TRIANGLE, 40)[0]
    cmap, norm = color_scale(problem.u(probe), signed=False)
    rows = (1.0, 0.8, 0.06) if slider else (1.0, 0.8)
    outer = figure.add_gridspec(len(rows), 1, height_ratios=rows)
    top = outer[0].subgridspec(1, 4)
    bottom = outer[1].subgridspec(1, 5)
    views = [figure.add_subplot(top[0, k], projection="3d") for k in range(4)]
    if hasattr(views[0], "shareview"):
        for ax in views[1:]:
            ax.shareview(views[0])
    figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=views[:2], shrink=0.6, label="u, u_h")
    segment_ax = figure.add_subplot(bottom[0, 0:2])
    flat_axes = {"volume": figure.add_subplot(bottom[0, 2]), "trace": figure.add_subplot(bottom[0, 3])}
    errors_ax = figure.add_subplot(bottom[0, 4])
    # One color bar per axes group of one grid row; a bar spanning both rows breaks the layout.
    error_bar = figure.colorbar(ScalarMappable(norm=norm, cmap=rp.diverging_colormap()), ax=views[2], shrink=0.6,
                                label="u - u_h")
    trace_bar = figure.colorbar(ScalarMappable(norm=norm, cmap=rp.diverging_colormap()), ax=views[3], shrink=0.6,
                                label="u - uhat_h")
    bars = {"volume": [error_bar], "trace": [trace_bar]}
    if dim == 3:
        for kind, label in (("volume", "u - u_h"), ("trace", "u - uhat_h")):
            bars[kind].append(figure.colorbar(ScalarMappable(norm=norm, cmap=rp.diverging_colormap()),
                                              ax=flat_axes[kind], shrink=0.8, label=label))
    s, segment = segment_points(dim)
    rp.plot_series(errors_ax, list(degrees), {
        "||u - u_h|| (HDG)": [results[p]["l2_error"] for p in degrees],
        "||u - Pi_p u|| (best)": [results[p]["l2_projection_error"] for p in degrees],
        "max |u - u_h|": [results[p]["max_error"] for p in degrees],
        "||u - u_h|| on the faces": [results[p]["face_l2_error"] for p in degrees],
        "||u - uhat_h|| on the faces": [results[p]["trace_l2_error"] for p in degrees],
    }, xlabel="degree p", ylabel="error", title="Errors against the degree")
    marker = errors_ax.axvline(degrees[0], color=rp.TEXT_SECONDARY, linewidth=1.0, linestyle="--")
    rp.plot_series(segment_ax, s, {"u": problem.u(segment), "u_h": problem.u(segment)}, logy=False,
                   xlabel="position on the segment through the centroid", ylabel="value")
    for line in segment_ax.lines:
        line.set_marker("")
    segment_ax.lines[0].set_linewidth(3.0)
    segment_ax.legend(frameon=False)
    default_view = (28, 35) if dim == 3 else (28, -62)
    for ax in views:
        ax.view_init(*default_view)

    def draw(index: int) -> None:
        """Draw every panel for ``degrees[index]``, keeping the current 3D viewing angle."""
        p = degrees[index]
        result = results[p]
        u, u_h, error = fields_for_degree(problem, result)
        on_faces = {kind: np.concatenate([v for _, _, v in face_errors(problem, result, dim, 8, kind=kind)])
                    for kind in FACE_FIELDS}
        error_cmap, error_norm = color_scale(np.concatenate([error(probe), on_faces["volume"]]), signed=True)
        trace_cmap, trace_norm = color_scale(on_faces["trace"], signed=True)
        scales = {"volume": (error_cmap, error_norm), "trace": (trace_cmap, trace_norm)}
        for kind, scale in (("volume", error_norm), ("trace", trace_norm)):
            for bar in bars[kind]:
                bar.mappable.set_norm(scale)
                bar.update_normal(bar.mappable)
        panels = (
            (u, (cmap, norm), "exact u = f / alpha, inside and on the faces", True),
            (u_h, (cmap, norm), f"HDG u_h inside and on the faces, p = {p}", True),
            (error, (error_cmap, error_norm), f"error u - u_h inside and on the faces, p = {p}", True),
        )
        for ax, (field, scale, title, with_faces) in zip(views[:3], panels):
            elev, azim = ax.elev, ax.azim
            ax.cla()
            if dim == 3:
                draw_tetrahedron_slices(ax, field, heights, *scale, faces=with_faces, face_opacity=face_opacity)
            else:
                draw_triangle_surface(ax, field, *scale, faces=with_faces)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(title)
        elev, azim = views[3].elev, views[3].azim
        views[3].cla()
        draw_faces_3d(views[3], problem, result, dim, *scales["trace"], kind="trace")
        views[3].view_init(elev=elev, azim=azim)
        views[3].set_title(f"u - uhat_h on the faces, p = {p}")
        for kind in FACE_FIELDS:
            flat_axes[kind].cla()
            draw_faces_flat(flat_axes[kind], problem, result, dim, *scales[kind], kind=kind)
        segment_ax.lines[1].set_ydata(u_h(segment))
        segment_ax.relim()
        segment_ax.autoscale_view()
        segment_ax.set_title(f"u and u_h, p = {p}: ||u - u_h|| = {result['l2_error']:.2e} "
                             f"(best {result['l2_projection_error']:.2e})")
        marker.set_xdata([p, p])

    name = "tetrahedron" if dim == 3 else "triangle"
    figure.suptitle(f"alpha u = f on the reference {name} ({problem.path.name}, tau = {problem.tau:g})"
                    + ("; drag a 3D view to rotate, move the slider to change p" if slider else ""))
    if slider:
        return draw, figure.add_subplot(outer[2])
    return draw


def summary_page(problem: Problem, dim: int, degrees, results):
    """Figure with the errors against the degree and ``u_h`` of several degrees along the segment."""
    figure = rp.new_figure(11.0, 4.2)
    ax = figure.add_subplot(1, 2, 1)
    rp.plot_series(ax, list(degrees), {
        "||u - u_h|| (HDG)": [results[p]["l2_error"] for p in degrees],
        "||u - Pi_p u|| (best)": [results[p]["l2_projection_error"] for p in degrees],
        "max |u - u_h|": [results[p]["max_error"] for p in degrees],
        "||u - u_h|| on the faces": [results[p]["face_l2_error"] for p in degrees],
        "||u - uhat_h|| on the faces": [results[p]["trace_l2_error"] for p in degrees],
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
        for index, p in enumerate(degrees):
            figure = rp.new_figure(20.0, 9.0)
            build_view(figure, problem, dim, degrees, results, args.heights, slider=False,
                       face_opacity=args.face_opacity)(index)
            pdf.savefig(figure)
            figure.savefig(out / f"degree-{p:02d}.png", dpi=120)
        figure = summary_page(problem, dim, degrees, results)
        pdf.savefig(figure)
        figure.savefig(out / "summary.png", dpi=140)
    record = {"problem": str(problem.path), "dim": dim, "basis": args.basis, "trace": args.trace, "tau": problem.tau,
              "extra_degree": args.extra_degree,
              "degrees": {str(p): {k: v for k, v in results[p].items() if isinstance(v, float)} for p in degrees}}
    with open(out / "errors.json", "w") as handle:
        json.dump(record, handle, indent=1)
    print(f"wrote {out / 'alpha_u_f.pdf'}, {len(degrees) + 1} PNG pages and errors.json")


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


def show_matplotlib(problem: Problem, dim: int, degrees, results: dict, heights, face_opacity: float = 0.4) -> None:
    """Open one window with the panels of :func:`build_view` and a slider that selects the degree."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    figure = plt.figure(figsize=(20.0, 9.5), layout="constrained")
    figure.canvas.manager.set_window_title(f"alpha u = f: {problem.path.name}")
    draw, slider_ax = build_view(figure, problem, dim, degrees, results, heights, slider=True,
                                 face_opacity=face_opacity)
    slider = Slider(slider_ax, "degree", 0, len(degrees) - 1, valinit=0, valstep=1)
    slider.valtext.set_text(f"p = {degrees[0]}")

    def on_change(value) -> None:
        """Redraw for the degree chosen with the slider."""
        index = int(round(value))
        slider.valtext.set_text(f"p = {degrees[index]}")
        draw(index)
        figure.canvas.draw_idle()

    slider.on_changed(on_change)
    draw(0)
    plt.show()


def show_pyvista(problem: Problem, dim: int, degrees, results: dict, heights, face_opacity: float = 0.4) -> None:
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
    # The faces as one mesh whose points are not shared between faces, because uhat_h is a separate
    # polynomial on every face (it may jump across the edges where two faces meet).
    samples, face_triangles = face_samples(dim, 24)
    face_blocks = []
    for f in range(dim + 1):
        x = reference_simplex(dim).face_map(f, samples)
        face_blocks.append(np.column_stack((x, np.zeros(len(x)))) if dim == 2 else x)
    count = len(samples)
    if dim == 3:
        cells = np.vstack([np.hstack([np.full((len(face_triangles), 1), 3), face_triangles + f * count])
                           for f in range(4)]).ravel()
        face_mesh = pv.PolyData(np.vstack(face_blocks), cells)
    else:
        lines = np.hstack([np.concatenate(([count], np.arange(count) + f * count)) for f in range(3)])
        face_mesh = pv.PolyData(np.vstack(face_blocks), lines=lines)
    face_coordinates = np.asarray(face_mesh.points, dtype=np.float64)[:, :dim]
    plotter = pv.Plotter(shape=(1, 4), window_size=(2200, 650), title=f"alpha u = f: {problem.path.name}")
    plotter.set_background("white")
    cmap = rp.sequential_colormap() if exact.min() >= 0.0 else rp.diverging_colormap()
    u_limits = (float(exact.min()), float(exact.max())) if exact.min() >= 0.0 else (-bound, bound)

    def show_field(column: int, name: str, values: np.ndarray, colormap, limits, title: str,
                   face_values=None) -> None:
        """Draw ``values`` in subplot ``column`` (cross-sections in 3D, a warped surface in 2D).

        ``face_values`` (values at the points of the face mesh) are drawn on the faces
        in the same subplot and color scale, with opacity ``face_opacity`` in 3D and
        as tubes along the edges in 2D.
        """
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
        if face_values is not None:
            on_faces = face_mesh.copy()
            on_faces.point_data[name] = face_values
            if dim == 2:
                lifted = on_faces.points.copy()
                lifted[:, 2] = face_values / max(np.abs(values).max(), 1e-300)
                on_faces.points = lifted
            plotter.add_mesh(on_faces, scalars=name, cmap=colormap, clim=limits, name=f"faces{column}",
                             lighting=False, opacity=face_opacity if dim == 3 else 1.0, line_width=6,
                             render_lines_as_tubes=dim == 2, show_scalar_bar=False)
            positions = face_label_positions(dim, 0.12)
            if dim == 2:
                positions = np.column_stack((positions, np.zeros(len(positions))))
            plotter.add_point_labels(positions, face_names(dim), font_size=13, text_color="black",
                                     shape="rounded_rect", shape_color="white", shape_opacity=0.7,
                                     show_points=False, name=f"facelabels{column}", always_visible=True)
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
        face_values = {kind: np.concatenate([v for _, _, v in face_errors(problem, results[p], dim, 24, kind=kind)])
                       for kind in FACE_FIELDS}
        error_bound = max(float(np.abs(errors).max()), float(np.abs(face_values["volume"]).max()), 1e-300)
        show_field(1, "u_h", values, cmap, u_limits,
                   f"HDG u_h inside and on the faces, p = {p}: L2 error {results[p]['l2_error']:.2e}",
                   face_values=u_h(face_coordinates))
        show_field(2, "u - u_h", errors, rp.diverging_colormap(), (-error_bound, error_bound),
                   f"error u - u_h inside and on the faces, p = {p} (max {results[p]['max_error']:.2e})",
                   face_values=face_values["volume"])
        bounds = {"volume": error_bound, "trace": max(float(np.abs(face_values["trace"]).max()), 1e-300)}
        norms = {"volume": results[p]["face_l2_error"], "trace": results[p]["trace_l2_error"]}
        for column, kind in ((3, "trace"),):
            name = FACE_FIELDS[kind]
            plotter.subplot(0, column)
            shown = face_mesh.copy()
            shown.point_data[name] = face_values[kind]
            plotter.add_mesh(shown, scalars=name, cmap=rp.diverging_colormap(), clim=(-bounds[kind], bounds[kind]),
                             name=f"faces{column}", lighting=False, line_width=6, render_lines_as_tubes=dim == 2,
                             scalar_bar_args={"title": name + " ", "color": "black", "vertical": True,
                                              "fmt": "%.3g", "position_x": 0.86, "width": 0.07, "label_font_size": 12})
            if dim == 3:
                plotter.add_mesh(edges, color="#52514e", line_width=2, name=f"edges{column}")
                plotter.add_point_labels(edges.points, ["A", "B", "C", "D"], font_size=12, text_color="black",
                                         shape=None, show_points=False, name=f"labels{column}")
            positions = face_label_positions(dim, 0.12)
            if dim == 2:
                positions = np.column_stack((positions, np.zeros(len(positions))))
            plotter.add_point_labels(positions, face_names(dim), font_size=13, text_color="black",
                                     shape="rounded_rect", shape_color="white", shape_opacity=0.7, show_points=False,
                                     name=f"facelabels{column}", always_visible=True)
            plotter.add_text(f"{name} on the faces, p = {p} (L2 {norms[kind]:.2e})", font_size=10, color="black",
                             name=f"title{column}")
        plotter.render()

    show_field(0, "u", exact, cmap, u_limits, "exact u = f / alpha, inside and on the faces",
               face_values=problem.u(face_coordinates))
    update(degrees[0])
    plotter.subplot(0, 1)
    if len(degrees) > 1:
        slider["widget"] = plotter.add_slider_widget(
            update, rng=[degrees[0], degrees[-1]], value=degrees[0], title="degree p", fmt="%.0f",
            pointa=(0.1, 0.08), pointb=(0.9, 0.08), color="black")
    plotter.link_views()
    plotter.view_vector((1.0, 0.75, 0.75)) if dim == 3 else plotter.view_vector((0.3, -1.0, 0.9))
    plotter.show()


def limit_blas_threads(count: int) -> None:
    """Limit the threads of the BLAS/LAPACK library under NumPy and SciPy to ``count``.

    On a node shared with other jobs, OpenBLAS with one thread per CPU makes a
    dense ``1000 x 1000`` solve hundreds of times slower than with 8 threads: its
    threads wait on each other while other processes hold some of the cores.
    Uses ``threadpoolctl`` when it is installed; otherwise prints how to set
    ``OPENBLAS_NUM_THREADS`` before starting Python.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        print(f"threadpoolctl is not installed: start with OPENBLAS_NUM_THREADS={count} to limit the BLAS threads")
        return
    threadpool_limits(limits=max(1, int(count)), user_api="blas")


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
    parser.add_argument("--blas-threads", type=int, default=min(8, os.cpu_count() or 1),
                        help="threads of the BLAS/LAPACK library for the dense products and solves (default "
                             "min(8, CPUs); more threads are slower on a shared node)")
    parser.add_argument("--face-opacity", type=float, default=0.4,
                        help="opacity of the faces drawn over the cross-sections in the u - u_h view")
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

    limit_blas_threads(args.blas_threads)
    results = {}
    for p in degrees:
        start = time.perf_counter()
        results[p] = solve(problem, dim, p, basis=args.basis, trace=args.trace, tau=problem.tau,
                           extra_degree=args.extra_degree)
        print(f"p = {p:2d}: ||u - u_h|| = {results[p]['l2_error']:.3e}, ||u - Pi_p u|| = "
              f"{results[p]['l2_projection_error']:.3e}, max |u - u_h| = {results[p]['max_error']:.3e}, "
              f"cond(A) = {results[p]['condition_A']:.2e} ({time.perf_counter() - start:.2f} s)", flush=True)

    if args.out is not None:
        write_files(problem, dim, degrees, results, args)
    if show:
        viewer = show_pyvista if args.backend == "pyvista" else show_matplotlib
        viewer(problem, dim, degrees, results, args.heights, face_opacity=args.face_opacity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
