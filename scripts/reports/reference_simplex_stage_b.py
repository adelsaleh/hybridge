"""Stage B pages of the reference-simplex report: HDG for ``alpha u = f`` on affine cells, patches and Kuhn meshes.

:func:`add_stage_b_pages` draws four pages into an open PDF (any object with a
``savefig(figure)`` method) and stores the numbers behind them in
``summary["stage_b"]``:

1. ``u_h`` on the two-cell patches and the cube patches: on triangles, ``u_h`` and
   ``u - u_h`` over the patch; on tetrahedra, horizontal cross-sections drawn in 3D
   and one cross-section of the Kuhn cube in its plane;
2. the orientation matrices ``R_sigma``, defined by
   ``mu_a(sigma s) = sum_b R_sigma[a, b] mu_b(s)``, of a nodal and the orthonormal
   trace basis on a tetrahedron face and on a triangle edge;
3. L2 errors of ``u_h`` and of the L2 projection against ``h`` and against ``p`` on
   Kuhn box meshes of ``[-1, 1]^d``, with the observed rates;
4. the checks of ``tests/test_simplex_hdg_reaction_geometry.py`` as numbers: the
   largest polynomial reproduction error, the extreme eigenvalues of the global
   condensed matrix, and the largest change of ``S`` and ``u_h`` under relabeling
   of the cells and under a change of volume basis.

The problem is ``alpha u = f`` with ``alpha = 1 + |x|^2 / 2``. Pages 1 and 3 use the
smooth solution ``u = exp((x_1 + ... + x_d) / 2) cos(pi x_1 / 2)``, its values as
boundary data and ``tau = 1``; page 4 uses the data of the tests (``tau = 2.3``).
Meshes, data and the solver come from ``tests/simplex_hdg_mesh.py``, which the
report script puts on ``sys.path``.
"""

from __future__ import annotations

import numpy as np

from hybridge.core.simplex.reference import compose_permutations, face_permutations, reference_simplex
from hybridge.io import reference_plots as rp
from simplex_hdg_helpers import random_polynomial
from simplex_hdg_mesh import (
    MESH_NAMES,
    cube_patch,
    evaluate_at_points,
    evaluate_cells,
    face_rule,
    face_trace_projection,
    l2_error,
    named_mesh,
    quadratic_alpha,
    reaction_convergence,
    reference_element,
    relabel_cells,
    sample_points,
    smooth_solution,
    smooth_source,
    solve_polynomial_problem,
    solve_reaction,
    two_cell_patch,
    volume_rule,
)

VOLUME_BASES = ("orthonormal", "dub_orth", "bernstein", "hier_C0", "lagrange_warp_blend", "lagrange_fekete")
TRACE_BASES = ("lagrange_warp_blend", "orthonormal")
CHECK_ORDERS = {2: (0, 1, 2, 4), 3: (0, 1, 2, 3)}
ALPHA, TAU = 1.7, 2.3
H_SUBDIVISIONS = {2: (2, 4, 8, 16), 3: (1, 2, 4, 6)}
P_ORDERS = {2: tuple(range(1, 11)), 3: tuple(range(1, 8))}


def solve_smooth(mesh, order: int, basis: str = "orthonormal", trace: str = "orthonormal"):
    """Solve the smooth problem (tau = 1) on ``mesh`` at degree ``order``; return ``(reference, solution, L2 error)``."""
    reference = reference_element(mesh.dim, order, basis)
    rule = volume_rule(mesh.dim, 2 * order + 6)
    solution = solve_reaction(mesh, reference, trace, alpha=quadratic_alpha, tau=1.0, source=smooth_source,
                              boundary=smooth_solution, rule=rule, face_rule=face_rule(mesh.dim, 2 * order + 6))
    return reference, solution, l2_error(mesh, reference, solution.u, smooth_solution, rule)


def triangle_lattice(n: int = 24) -> np.ndarray:
    """Points ``(i, j)/n`` in barycentric coordinates of the reference triangle, ``i + j <= n``; shape ``((n+1)(n+2)/2, 2)``.

    The points on the edges lie exactly on them, so a plot of a cell ends at its edges.
    """
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    keep = i + j <= n
    bary = np.column_stack((n - i[keep] - j[keep], i[keep], j[keep])) / n
    return reference_simplex(2).from_barycentric(bary)


def draw_triangle_patch(ax, mesh, reference, coefficients, *, error: bool, cmap, norm, title: str):
    """Draw ``u_h`` (or ``u - u_h`` with ``error``) over a triangle mesh, with the cell edges and the vertex numbers.

    Every cell is sampled on the same lattice of the reference triangle; the
    lattice triangles of all cells form one triangulation drawn with Gouraud
    shading, so a jump of ``u_h`` between two cells stays visible. Returns the image.
    """
    import matplotlib.tri as mtri

    lattice = triangle_lattice()
    triangles = mtri.Triangulation(lattice[:, 0], lattice[:, 1]).triangles
    points = mesh.map_points(lattice)  # (n_K, n, 2)
    values = evaluate_cells(mesh, reference, coefficients, points)
    if error:
        values = smooth_solution(points.reshape(-1, 2)).reshape(values.shape) - values
    offsets = (np.arange(mesh.num_cells) * len(lattice))[:, None, None]
    image = ax.tripcolor(points[:, :, 0].ravel(), points[:, :, 1].ravel(), (triangles[None] + offsets).reshape(-1, 3),
                         values.ravel(), shading="gouraud", cmap=cmap, norm=norm)
    from matplotlib.collections import LineCollection

    corners = mesh.vertices[mesh.cells]  # (n_K, 3, 2)
    edges = np.stack((corners, np.roll(corners, -1, axis=1)), axis=2).reshape(-1, 2, 2)
    ax.add_collection(LineCollection(edges, colors=rp.OUTLINE, linewidths=0.7))
    lower, upper = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    pad = 0.02 * (upper - lower)
    ax.set_xlim(lower[0] - pad[0], upper[0] + pad[0])
    ax.set_ylim(lower[1] - pad[1], upper[1] + pad[1])
    for number, position in enumerate(mesh.vertices):
        ax.annotate(str(number), position, textcoords="offset points", xytext=(3, 3), fontsize=7,
                    color=rp.TEXT_SECONDARY)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)
    return image


def cross_section(mesh, reference, coefficients, height: float, resolution: int = 70):
    """Sample ``u_h`` on the plane ``x_3 = height`` of a tetrahedral mesh.

    A ``resolution x resolution`` grid over the bounding box of the mesh is cut
    into two triangles per square; ``u_h`` is evaluated at the centroid of every
    triangle whose centroid lies in a cell. Returns the triangle corners
    ``(m, 3, 3)``, the centroids ``(m, 3)`` and the values ``(m,)``.
    """
    lower, upper = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    s, t = np.meshgrid(np.linspace(lower[0], upper[0], resolution), np.linspace(lower[1], upper[1], resolution),
                       indexing="ij")
    grid = np.stack((s, t, np.full_like(s, height)), axis=-1).reshape(-1, 3)
    index = np.arange(resolution * resolution).reshape(resolution, resolution)
    a, b, c, d = index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[:-1, 1:].ravel(), index[1:, 1:].ravel()
    triangles = np.concatenate((np.stack((a, b, d), axis=1), np.stack((a, d, c), axis=1)))
    corners = grid[triangles]
    centroids = corners.mean(axis=1)
    values = evaluate_at_points(mesh, reference, coefficients, centroids)
    keep = np.isfinite(values)
    return corners[keep], centroids[keep], values[keep]


def draw_tetrahedron_sections(ax3d, mesh, reference, coefficients, heights, *, cmap, norm, title: str) -> None:
    """Draw the edges of every tetrahedron and the cross-sections ``x_3 = height`` of ``u_h`` on 3D axes."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    polygons, colors = [], []
    for height in heights:
        corners, _, values = cross_section(mesh, reference, coefficients, height, resolution=45)
        polygons.append(corners)
        colors.append(cmap(norm(values)))
    ax3d.computed_zorder = False
    ax3d.add_collection3d(Poly3DCollection(np.concatenate(polygons), facecolors=np.concatenate(colors),
                                           edgecolors="none", zorder=1))
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    segments = mesh.vertices[mesh.cells[:, [p for pair in pairs for p in pair]]].reshape(-1, 2, 3)
    ax3d.add_collection3d(Line3DCollection(segments, colors=rp.OUTLINE, linewidths=0.5, zorder=2))
    lower, upper = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    ax3d.set_xlim(lower[0], upper[0])
    ax3d.set_ylim(lower[1], upper[1])
    ax3d.set_zlim(lower[2], upper[2])
    ax3d.set_box_aspect(tuple(upper - lower))
    ax3d.view_init(elev=20, azim=-58)
    ax3d.set_axis_off()
    ax3d.set_title(title)


def draw_section_in_plane(ax, values, corners, *, cmap, norm, title: str):
    """Draw one cross-section (from :func:`cross_section`) in its plane, one flat color per grid triangle."""
    from matplotlib.collections import PolyCollection

    collection = PolyCollection(corners[:, :, :2], array=values, cmap=cmap, norm=norm, edgecolors="face",
                                linewidths=0.3, antialiased=False)
    ax.add_collection(collection)
    ax.set_xlim(corners[:, :, 0].min(), corners[:, :, 0].max())
    ax.set_ylim(corners[:, :, 1].min(), corners[:, :, 1].max())
    ax.set_aspect("equal")
    ax.set_xlabel("x_1")
    ax.set_ylabel("x_2")
    ax.set_title(title)
    return collection


def _largest_value(dim: int) -> float:
    """Largest value of :func:`smooth_solution` on a grid of ``[-1, 1]^dim`` (the top of the color scale of ``u_h``)."""
    axis = np.linspace(-1.0, 1.0, 41)
    points = np.stack(np.meshgrid(*([axis] * dim), indexing="ij"), axis=-1).reshape(-1, dim)
    return float(smooth_solution(points).max())


def page_solutions(pdf, summary, order: int = 2) -> None:
    """Page 1: u_h and u - u_h on the two-cell and cube patches, triangles and tetrahedra."""
    from matplotlib.colors import Normalize

    figure = rp.new_figure(11.0, 8.5)
    grid = figure.add_gridspec(2, 4)
    sequential, diverging = rp.sequential_colormap(), rp.diverging_colormap()
    data = {}

    patches_2d = (("two cells", two_cell_patch(2)), ("square, four cells", cube_patch(2)))
    value_norm = Normalize(0.0, _largest_value(2))
    for k, (name, mesh) in enumerate(patches_2d):
        reference, solution, error = solve_smooth(mesh, order)
        points = mesh.map_points(triangle_lattice())
        values = evaluate_cells(mesh, reference, solution.u, points)
        bound = float(np.abs(smooth_solution(points.reshape(-1, 2)).reshape(values.shape) - values).max())
        ax = figure.add_subplot(grid[0, 2 * k])
        image = draw_triangle_patch(ax, mesh, reference, solution.u, error=False, cmap=sequential, norm=value_norm,
                                    title=f"triangles, {name}:\nu_h")
        if k == 1:
            figure.colorbar(image, ax=ax, shrink=0.75, label="u_h")
        ax = figure.add_subplot(grid[0, 2 * k + 1])
        image = draw_triangle_patch(ax, mesh, reference, solution.u, error=True, cmap=diverging,
                                    norm=Normalize(-bound, bound), title=f"u - u_h\n(L2 error {error:.2e})")
        figure.colorbar(image, ax=ax, shrink=0.75, format="%.0e")
        data[f"triangles, {name}"] = {"l2_error": error, "max_abs_error": bound,
                                      "orientations": mesh.orientations.tolist()}

    two = two_cell_patch(3)
    value_norm = Normalize(0.0, _largest_value(3))
    reference, solution, error = solve_smooth(two, order)
    ax3d = figure.add_subplot(grid[1, 0], projection="3d")
    draw_tetrahedron_sections(ax3d, two, reference, solution.u, (-0.7, -0.3, 0.1, 0.5), cmap=sequential,
                              norm=value_norm, title=f"tetrahedra, two cells: u_h\n(L2 error {error:.2e})")
    data["tetrahedra, two cells"] = {"l2_error": error, "orientations": two.orientations.tolist()}

    cube = cube_patch(3)
    reference, solution, error = solve_smooth(cube, order)
    ax3d = figure.add_subplot(grid[1, 1], projection="3d")
    draw_tetrahedron_sections(ax3d, cube, reference, solution.u, (-0.6, 0.0, 0.6), cmap=sequential,
                              norm=value_norm, title=f"Kuhn cube, six cells: u_h\n(L2 error {error:.2e})")
    height = 0.2
    corners, centroids, values = cross_section(cube, reference, solution.u, height, resolution=121)
    difference = smooth_solution(centroids) - values
    bound = float(np.abs(difference).max())
    ax = figure.add_subplot(grid[1, 2])
    image = draw_section_in_plane(ax, values, corners, cmap=sequential, norm=value_norm,
                                  title=f"Kuhn cube, section x_3 = {height}:\nu_h")
    figure.colorbar(image, ax=ax, shrink=0.75, label="u_h")
    ax = figure.add_subplot(grid[1, 3])
    image = draw_section_in_plane(ax, difference, corners, cmap=diverging, norm=Normalize(-bound, bound),
                                  title=f"Kuhn cube, section x_3 = {height}:\nu - u_h")
    figure.colorbar(image, ax=ax, shrink=0.75, format="%.0e")
    data["tetrahedra, Kuhn cube"] = {"l2_error": error, "section_height": height, "section_max_abs_error": bound}

    figure.suptitle(f"Stage B: u_h of alpha u = f on patches, p = {order}, tau = 1, alpha = 1 + |x|^2/2")
    figure.supxlabel(
        "Triangles: cells (0, 1, 2) and (3, 2, 1) share edge {1, 2}, seen reversed by the second cell. "
        "Tetrahedra: cells (0, 1, 2, 3) and (2, 3, 1, 4) share face {1, 2, 3}, seen through the 3-cycle (1, 2, 0) "
        "by the second cell.", fontsize=8)
    pdf.savefig(figure)
    summary.setdefault("stage_b", {})["patches"] = {"order": order, **data}


def page_orientation_matrices(pdf, summary) -> None:
    """Page 2: the orientation matrices R_sigma of a nodal and the orthonormal trace basis, and their identities."""
    from matplotlib.colors import Normalize

    figure = rp.new_figure(11.0, 6.5)
    grid = figure.add_gridspec(3, 6)
    norm = Normalize(-1.0, 1.0)
    residuals = {}
    image = None
    rows = ((3, 2, "lagrange_warp_blend"), (3, 2, "orthonormal"))
    for row, (dim, order, trace) in enumerate(rows):
        tables = reference_element(dim, order, "orthonormal").trace_tables(trace)
        rotations = tables.orientation_matrices.astype(np.float64)
        for o, sigma in enumerate(face_permutations(dim)):
            ax = figure.add_subplot(grid[row, o])
            image = ax.imshow(rotations[o], cmap=rp.diverging_colormap(), norm=norm, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"sigma = {sigma}")
            if o == 0:
                ax.set_ylabel(f"face p = {order}\n{_trace_label(trace)}")
        residuals[f"tetrahedron p={order} {trace}"] = _orientation_identities(dim, tables)
    for k, trace in enumerate(TRACE_BASES):
        tables = reference_element(2, 3, "orthonormal").trace_tables(trace)
        rotations = tables.orientation_matrices.astype(np.float64)
        for o, sigma in enumerate(face_permutations(2)):
            ax = figure.add_subplot(grid[2, 2 * k + o])
            image = ax.imshow(rotations[o], cmap=rp.diverging_colormap(), norm=norm, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"edge p = 3, sigma = {sigma}", fontsize=8)
            ax.set_xlabel("nodal (Gauss-Lobatto)" if trace == "lagrange_warp_blend" else trace)
        residuals[f"triangle p=3 {trace}"] = _orientation_identities(2, tables)
    ax = figure.add_subplot(grid[2, 4:])
    lines = ["mu_a(sigma s) = sum_b R[a, b] mu_b(s), where sigma maps",
             "local face vertex i to global face vertex sigma[i].",
             "Largest residuals over the bases shown:"]
    worst = {key: max(value[key] for value in residuals.values()) for key in ("composition", "mass", "coupling")}
    lines += [f"  R_(a o b) - R_a R_b: {worst['composition']:.1e}",
              f"  R G R^T - G: {worst['mass']:.1e}",
              f"  coupling[o] - coupling[0] R_o^T: {worst['coupling']:.1e}"]
    ax.axis("off")
    ax.text(0.0, 0.5, "\n".join(lines), fontsize=7, va="center", ha="left", wrap=True)
    figure.colorbar(image, ax=figure.axes[:-1], shrink=0.5, label="entry of R")
    figure.suptitle("Orientation matrices of the trace bases (tetrahedron face p = 2, triangle edge p = 3)")
    pdf.savefig(figure)
    summary.setdefault("stage_b", {})["orientation_matrices"] = residuals


def _trace_label(trace: str) -> str:
    """Short name of a trace basis for a figure label."""
    return "nodal (warp and blend)" if trace == "lagrange_warp_blend" else trace


def _orientation_identities(dim: int, tables) -> dict:
    """Largest residuals of ``R_(a o b) = R_a R_b``, ``R G R^T = G`` and ``coupling[o] = coupling[0] R_o^T``."""
    sigmas = face_permutations(dim)
    rotations = tables.orientation_matrices.astype(np.float64)
    mass = tables.trace_mass.astype(np.float64)
    coupling = tables.element_trace_coupling.astype(np.float64)
    composition = max(np.abs(rotations[sigmas.index(compose_permutations(a, b))] - rotations[i] @ rotations[j]).max()
                      for i, a in enumerate(sigmas) for j, b in enumerate(sigmas))
    return {
        "composition": float(composition),
        "mass": float(max(np.abs(r @ mass @ r.T - mass).max() for r in rotations)),
        "coupling": float(max(np.abs(coupling[o] - np.einsum("fia,ba->fib", coupling[0], r)).max()
                              for o, r in enumerate(rotations))),
    }


def page_convergence(pdf, summary) -> None:
    """Page 3: L2 errors against h and p on Kuhn box meshes, with rates, and their table."""
    figure = rp.new_figure(11.0, 8.5)
    grid = figure.add_gridspec(2, 6, height_ratios=(1.0, 1.0))
    result = {"h": {}, "p": {}}
    rows = {2: [], 3: []}
    for column, dim in enumerate((2, 3)):
        ax = figure.add_subplot(grid[0, 2 * column:2 * column + 2])
        subdivisions = H_SUBDIVISIONS[dim]
        sizes = 2.0 / np.asarray(subdivisions, dtype=float)
        for index, order in enumerate((1, 2, 3)):
            errors, projection_errors = reaction_convergence(dim, order, subdivisions)
            rates = np.log(np.divide(errors[:-1], errors[1:])) / np.log(sizes[:-1] / sizes[1:])
            color, marker = rp.CATEGORICAL[index], rp.MARKERS[index]
            ax.loglog(sizes, errors, color=color, marker=marker, markersize=4,
                      label=f"p = {order}: u_h, last rate {rates[-1]:.2f}")
            ax.loglog(sizes, projection_errors, color=color, linestyle="--", linewidth=1.0,
                      label=f"p = {order}: L2 projection")
            result["h"][f"d={dim} p={order}"] = {"N": list(subdivisions), "h": sizes.tolist(), "error": errors,
                                                 "projection_error": projection_errors, "rates": rates.tolist()}
            for k, n in enumerate(subdivisions):
                rows[dim].append([str(order), str(n), f"{sizes[k]:.3f}", f"{errors[k]:.3e}",
                                  "" if k == 0 else f"{rates[k - 1]:.2f}", f"{projection_errors[k]:.3e}",
                                  f"{errors[k] / projection_errors[k]:.2f}"])
        ax.grid(True, which="major", linewidth=0.4)
        ax.minorticks_off()
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{size:.3g}" for size in sizes])
        ax.set_xlabel("h = 2 / N (cube edge)")
        ax.set_ylabel("L2 error")
        ax.set_title(("triangles" if dim == 2 else "tetrahedra") + ", Kuhn box meshes")
        ax.legend(frameon=False, fontsize=6)
    ax = figure.add_subplot(grid[0, 4:])
    series, xs = {}, {}
    for dim in (2, 3):
        orders = P_ORDERS[dim]
        errors = [reaction_convergence(dim, order, (2,))[0][0] for order in orders]
        name = "triangles, N = 2" if dim == 2 else "tetrahedra, N = 2"
        series[name], xs[name] = errors, list(orders)
        result["p"][name] = {"orders": list(orders), "error": errors}
    rp.plot_series(ax, xs, series, xlabel="p", ylabel="L2 error", title="Error against the degree")
    for column, dim in enumerate((2, 3)):
        ax = figure.add_subplot(grid[1, 3 * column:3 * column + 3])
        rp.plot_text_table(ax, rows[dim], ["p", "N", "h", "error of u_h", "rate", "projection error", "ratio"],
                           title=("triangles" if dim == 2 else "tetrahedra") + ": L2 errors (tau = 1)")
    figure.supxlabel("ratio = error of u_h / error of the L2 projection. With tau fixed, the jump term weighs tau / h "
                     "against the mass: the ratio grows from the projection value towards that of the continuous "
                     "interpolant, and the rate approaches p + 1 from below.", fontsize=7)
    figure.suptitle("Stage B: convergence of u_h for alpha = 1 + |x|^2/2 and u = exp(sum x / 2) cos(pi x_1 / 2)")
    pdf.savefig(figure)
    summary.setdefault("stage_b", {})["convergence"] = result


def mesh_checks(dim: int, name: str) -> dict:
    """The numbers behind the checks of the test file on one mesh (see :func:`page_checks`)."""
    mesh = named_mesh(name, dim)
    points = sample_points(mesh)
    reproduction, trace_error = 0.0, 0.0
    for order in CHECK_ORDERS[dim]:
        exact = random_polynomial(dim, order, seed=17 + order)
        expected = exact(points.reshape(-1, dim)).reshape(points.shape[:2])
        for basis in VOLUME_BASES:
            reference = reference_element(dim, order, basis)
            for trace in TRACE_BASES:
                solution = solve_reaction(mesh, reference, trace, alpha=ALPHA, tau=TAU,
                                          source=lambda x: ALPHA * exact(x), boundary=exact)
                values = evaluate_cells(mesh, reference, solution.u, points)
                reproduction = max(reproduction, float(np.abs(values - expected).max()))
                traces = face_trace_projection(mesh, reference, trace, exact)
                trace_error = max(trace_error, float(np.abs(solution.uhat - traces).max()))
    order = max(CHECK_ORDERS[dim])
    eigenvalues = np.linalg.eigvalsh(
        solve_polynomial_problem(mesh, reference_element(dim, order, "bernstein"), "lagrange_warp_blend",
                                 tau=TAU).matrix.toarray())
    relabel_matrix, relabel_values = 0.0, 0.0
    basis_matrix, basis_values = 0.0, 0.0
    for trace in TRACE_BASES:
        reference = reference_element(dim, 2, "orthonormal")
        base = solve_polynomial_problem(mesh, reference, trace, tau=TAU)
        base_values = evaluate_cells(mesh, reference, base.u, points)
        scale = np.abs(base.matrix).max()
        for seed in range(3):
            for keep_sign in (True, False):
                other = relabel_cells(mesh, seed=100 + seed, keep_sign=keep_sign)
                solution = solve_polynomial_problem(other, reference, trace, tau=TAU)
                relabel_matrix = max(relabel_matrix, float(np.abs((solution.matrix - base.matrix).toarray()).max() / scale))
                values = evaluate_cells(other, reference, solution.u, points)
                relabel_values = max(relabel_values, float(np.abs(values - base_values).max()))
        for basis in VOLUME_BASES[1:]:
            other_reference = reference_element(dim, 2, basis)
            solution = solve_polynomial_problem(mesh, other_reference, trace, tau=TAU)
            basis_matrix = max(basis_matrix, float(np.abs((solution.matrix - base.matrix).toarray()).max() / scale))
            values = evaluate_cells(mesh, other_reference, solution.u, points)
            basis_values = max(basis_values, float(np.abs(values - base_values).max()))
    return {
        "cells": mesh.num_cells, "faces": mesh.num_faces,
        "orientations_present": sorted(set(mesh.orientations.ravel().tolist())),
        "reproduction_max_error": reproduction, "trace_max_error": trace_error,
        "S_min_eigenvalue": float(eigenvalues.min()), "S_max_eigenvalue": float(eigenvalues.max()),
        "S_order": order,
        "relabel_S_relative_change": relabel_matrix, "relabel_u_h_change": relabel_values,
        "volume_basis_S_relative_change": basis_matrix, "volume_basis_u_h_change": basis_values,
    }


def page_checks(pdf, summary) -> None:
    """Page 4: the checks of the test file as numbers, one row per mesh and dimension."""
    rows, data = [], {}
    for dim in (2, 3):
        for name in MESH_NAMES:
            numbers = mesh_checks(dim, name)
            data[f"d={dim} {name}"] = numbers
            rows.append([str(dim), name, str(numbers["cells"]), str(numbers["faces"]),
                         " ".join(str(o) for o in numbers["orientations_present"]),
                         f"{numbers['reproduction_max_error']:.1e}", f"{numbers['trace_max_error']:.1e}",
                         f"{numbers['S_min_eigenvalue']:.2e}", f"{numbers['S_max_eigenvalue']:.2e}",
                         f"{numbers['relabel_S_relative_change']:.1e}", f"{numbers['relabel_u_h_change']:.1e}",
                         f"{numbers['volume_basis_S_relative_change']:.1e}", f"{numbers['volume_basis_u_h_change']:.1e}"])
    figure = rp.new_figure(11.0, 4.0)
    ax = figure.add_subplot(1, 1, 1)
    rp.plot_text_table(ax, rows, ["d", "mesh", "cells", "faces", "orientations", "max |u_h - u*|",
                                  "max |uhat - P u*|", "min eig S", "max eig S", "relabel: dS / |S|",
                                  "relabel: du_h", "basis: dS / |S|", "basis: du_h"],
                       title="Checks of tests/test_simplex_hdg_reaction_geometry.py (alpha = 1.7 or 1 + |x|^2/2, tau = 2.3)")
    orders = ", ".join(f"d = {d}: p in {CHECK_ORDERS[d]}" for d in (2, 3))
    figure.supxlabel(
        f"Reproduction: u* of degree p ({orders}), six volume bases, two trace bases, alpha = 1.7.\n"
        "Eigenvalues of S (all faces): highest p, Bernstein volume basis, nodal trace basis, alpha = 1 + |x|^2/2.\n"
        "Relabeling: three random relabelings that keep the sign of det A_K and three that need not, p = 2; "
        "basis: the six volume bases against the orthonormal one, p = 2.\n"
        "dS / |S|: largest entry of the change of the global condensed matrix over its largest entry; "
        "du_h: largest change of u_h at sample points.", fontsize=7)
    pdf.savefig(figure)
    summary.setdefault("stage_b", {})["checks"] = data


def add_stage_b_pages(pdf, summary) -> None:
    """Draw the four stage B pages into ``pdf`` and store their numbers in ``summary["stage_b"]``."""
    page_solutions(pdf, summary)
    page_orientation_matrices(pdf, summary)
    page_convergence(pdf, summary)
    page_checks(pdf, summary)


__all__ = ["add_stage_b_pages"]
