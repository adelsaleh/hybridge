"""Stage C pages of the reference-simplex report: HDG for ``beta . grad u + alpha u = f`` on affine cells and Kuhn meshes.

:func:`add_stage_c_pages` draws four pages into an open PDF (any object with a
``savefig(figure)`` method) and stores the numbers behind them in
``summary["stage_c"]``:

1. ``u_h`` and ``u - u_h`` on two triangle meshes of the square and on two tetrahedral
   meshes of the cube (horizontal cross-sections drawn in 3D, and one cross-section
   of the Kuhn cube in its plane);
2. L2 errors of ``u_h`` and of the L2 projection against ``h`` and against ``p`` on
   Kuhn box meshes of ``[-1, 1]^d``, with the observed rates;
3. the condition number of the trace matrix ``S_II`` of the interior faces (the
   matrix that is solved) against ``h`` and against ``p``;
4. the checks of ``tests/test_simplex_hdg_advection_geometry.py`` as numbers: blocks of
   single cells against quadrature on the cell, polynomial reproduction, ``cond(S_II)``,
   the largest change of ``S`` and ``u_h`` under relabeling of the cells and under a
   change of volume basis, and the difference with ``solve_advection_reaction_hdg`` in 2D.

The problem is ``beta . grad u + alpha u = f`` with constant ``beta`` (``(1, 0.45)`` on
triangles, ``(1, 0.65, 0.3)`` on tetrahedra, oblique to every mesh face) and the upwind
``tau = |beta . n|`` unless a page says otherwise. Pages 1 to 3 use ``alpha = 1``, the
smooth solution ``u = exp((x_1 + ... + x_d) / 2) cos(pi x_1 / 2)`` and its values as
boundary data on every boundary face; page 4 uses the data of the tests (``alpha = 1.7``).
Meshes, data and the solver come from ``tests/simplex_hdg_mesh.py`` and
``tests/simplex_hdg_transport.py``, which the report script puts on ``sys.path``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg

from hybridge.io import reference_plots as rp
from reference_simplex_stage_b import (
    _largest_value,
    cross_section,
    draw_section_in_plane,
    draw_tetrahedron_sections,
    draw_triangle_patch,
    triangle_lattice,
)
from simplex_hdg_helpers import random_polynomial
from simplex_hdg_mesh import (
    CELL_KINDS,
    MESH_NAMES,
    cell_load,
    cell_vertices,
    evaluate_cells,
    face_rule,
    face_trace_projection,
    kuhn_box_mesh,
    l2_error,
    named_mesh,
    one_cell_mesh,
    polynomial_data,
    reference_element,
    relabel_cells,
    sample_points,
    simplex_mesh,
    smooth_solution,
    solve_condensed,
    two_dimensional_meshes,
    volume_rule,
)
from simplex_hdg_transport import (
    direct_cell_blocks,
    interior_trace_matrix,
    mesh_transport_blocks,
    polynomial_gradient,
    smooth_transport_source,
    solve_transport,
    transport_convergence,
)

VOLUME_BASES = ("orthonormal", "dub_orth", "bernstein", "hier_C0", "lagrange_equispaced", "lagrange_warp_blend",
                "lagrange_recursive", "lagrange_fekete", "lagrange_fekete_free")
TRACE_BASES = ("lagrange_warp_blend", "orthonormal")
STABILIZATIONS = (None, "lax-friedrichs", 2.3)
CHECK_ORDERS = {2: (0, 1, 2, 4), 3: (0, 1, 2, 3)}
BETA = {2: np.array([1.0, 0.45]), 3: np.array([1.0, 0.65, 0.3])}
ALPHA = 1.7
H_SUBDIVISIONS = {2: (4, 8, 16, 32), 3: (2, 4, 6, 8)}
P_ORDERS = {2: tuple(range(1, 11)), 3: tuple(range(1, 8))}


def solve_smooth(mesh, order: int, basis: str = "orthonormal", trace: str = "orthonormal"):
    """Solve the smooth problem (alpha = 1, upwind tau) on ``mesh`` at degree ``order``; return ``(reference, solution, L2 error)``."""
    beta = BETA[mesh.dim]
    reference = reference_element(mesh.dim, order, basis)
    rule = volume_rule(mesh.dim, 2 * order + 6)
    solution = solve_transport(mesh, reference, trace, beta=beta, alpha=1.0, source=smooth_transport_source(beta, 1.0),
                               boundary=smooth_solution, rule=rule, face_rule=face_rule(mesh.dim, 2 * order + 6))
    return reference, solution, l2_error(mesh, reference, solution.u, smooth_solution, rule)


def interior_condition_estimate(mesh, solution) -> float:
    """Estimate of the 1-norm condition number of ``S_II`` (``scipy.sparse.linalg.onenormest`` of ``S_II`` and of its inverse by LU)."""
    n_t = solution.uhat.shape[1]
    free = np.flatnonzero(np.repeat(~mesh.boundary_faces, n_t))
    matrix = solution.matrix[free][:, free].tocsc()
    factor = scipy.sparse.linalg.splu(matrix)
    inverse = scipy.sparse.linalg.LinearOperator(matrix.shape, matvec=factor.solve,
                                                 rmatvec=lambda x: factor.solve(x, trans="T"), dtype=np.float64)
    return float(scipy.sparse.linalg.onenormest(matrix) * scipy.sparse.linalg.onenormest(inverse))


def _draw_velocity(ax, beta) -> None:
    """Draw the direction of ``beta`` as an arrow in the upper left corner of ``ax``."""
    direction = np.asarray(beta, dtype=float) / np.linalg.norm(beta)
    start = np.array([0.06, 0.9])
    ax.annotate("", xy=start + 0.12 * direction, xytext=start, xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": rp.TEXT_SECONDARY, "linewidth": 1.2})
    ax.text(start[0], start[1] + 0.03, "beta", transform=ax.transAxes, fontsize=7, color=rp.TEXT_SECONDARY)


def page_solutions(pdf, summary, order: int = 2) -> None:
    """Page 1: u_h and u - u_h on two triangle meshes and two tetrahedral meshes."""
    from matplotlib.colors import Normalize

    figure = rp.new_figure(11.0, 8.5)
    grid = figure.add_gridspec(2, 4)
    sequential, diverging = rp.sequential_colormap(), rp.diverging_colormap()
    data = {}
    meshes_2d = (("Kuhn mesh, N = 4", kuhn_box_mesh(2, 4)), ("perturbed, relabeled, N = 2", named_mesh("perturbed box", 2)))
    value_norm = Normalize(0.0, _largest_value(2))
    for k, (name, mesh) in enumerate(meshes_2d):
        reference, solution, error = solve_smooth(mesh, order)
        points = mesh.map_points(triangle_lattice())
        values = evaluate_cells(mesh, reference, solution.u, points)
        bound = float(np.abs(smooth_solution(points.reshape(-1, 2)).reshape(values.shape) - values).max())
        ax = figure.add_subplot(grid[0, 2 * k])
        image = draw_triangle_patch(ax, mesh, reference, solution.u, error=False, cmap=sequential, norm=value_norm,
                                    title=f"triangles, {name}:\nu_h")
        _draw_velocity(ax, BETA[2])
        if k == 1:
            figure.colorbar(image, ax=ax, shrink=0.75, label="u_h")
        ax = figure.add_subplot(grid[0, 2 * k + 1])
        image = draw_triangle_patch(ax, mesh, reference, solution.u, error=True, cmap=diverging,
                                    norm=Normalize(-bound, bound), title=f"u - u_h\n(L2 error {error:.2e})")
        figure.colorbar(image, ax=ax, shrink=0.75, format="%.0e")
        data[f"triangles, {name}"] = {"cells": mesh.num_cells, "l2_error": error}

    value_norm = Normalize(0.0, _largest_value(3))
    perturbed = named_mesh("perturbed box", 3)
    reference, solution, error = solve_smooth(perturbed, order)
    ax3d = figure.add_subplot(grid[1, 0], projection="3d")
    draw_tetrahedron_sections(ax3d, perturbed, reference, solution.u, (-0.6, 0.0, 0.6), cmap=sequential,
                              norm=value_norm, title=f"tetrahedra, perturbed, N = 2:\nu_h (L2 error {error:.2e})")
    data["tetrahedra, perturbed and relabeled, N = 2"] = {"cells": perturbed.num_cells, "l2_error": error}

    cube = kuhn_box_mesh(3, 1)
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

    figure.suptitle(f"Stage C: u_h of beta . grad u + u = f, p = {order}, upwind tau = abs(beta . n)")
    figure.supxlabel(
        "beta = (1, 0.45) on triangles, (1, 0.65, 0.3) on tetrahedra; u = exp(sum x / 2) cos(pi x_1 / 2). The boundary "
        "trace is the projection of u on every boundary face;\nit does not enter u_h on outflow faces. u_h is "
        "discontinuous: its jumps between cells are visible in the error plots.", fontsize=8)
    pdf.savefig(figure)
    summary.setdefault("stage_c", {})["solutions"] = {"order": order, **data}


def page_convergence(pdf, summary) -> None:
    """Page 2: L2 errors against h and p on Kuhn box meshes, with rates, and their table."""
    figure = rp.new_figure(11.0, 8.5)
    grid = figure.add_gridspec(2, 6, height_ratios=(1.0, 1.0))
    result = {"h": {}, "p": {}}
    rows = {2: [], 3: []}
    for column, dim in enumerate((2, 3)):
        ax = figure.add_subplot(grid[0, 2 * column:2 * column + 2])
        subdivisions = H_SUBDIVISIONS[dim]
        sizes = 2.0 / np.asarray(subdivisions, dtype=float)
        for index, order in enumerate((1, 2, 3)):
            errors, projection_errors = transport_convergence(dim, order, subdivisions, beta=BETA[dim])
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
        errors = [transport_convergence(dim, order, (2,), beta=BETA[dim])[0][0] for order in orders]
        name = "triangles, N = 2" if dim == 2 else "tetrahedra, N = 2"
        series[name], xs[name] = errors, list(orders)
        result["p"][name] = {"orders": list(orders), "error": errors}
    rp.plot_series(ax, xs, series, xlabel="p", ylabel="L2 error", title="Error against the degree")
    for column, dim in enumerate((2, 3)):
        ax = figure.add_subplot(grid[1, 3 * column:3 * column + 3])
        rp.plot_text_table(ax, rows[dim], ["p", "N", "h", "error of u_h", "rate", "projection error", "ratio"],
                           title=("triangles" if dim == 2 else "tetrahedra") + ": L2 errors (upwind tau, alpha = 1)")
    figure.supxlabel("ratio = error of u_h / error of the L2 projection. Upwind DG, which this HDG method equals for "
                     "tau = abs(beta . n), is proven to converge at rate p + 1/2 in general; on these meshes the rate is p + 1.",
                     fontsize=7)
    figure.suptitle("Stage C: convergence of u_h for beta . grad u + u = f and u = exp(sum x / 2) cos(pi x_1 / 2)")
    pdf.savefig(figure)
    summary.setdefault("stage_c", {})["convergence"] = result


def page_conditioning(pdf, summary) -> None:
    """Page 3: the 1-norm condition number of S_II against h (p = 1..3) and against p (N = 2) on Kuhn box meshes."""
    figure = rp.new_figure(11.0, 4.5)
    result = {"h": {}, "p": {}}
    for column, dim in enumerate((2, 3)):
        ax = figure.add_subplot(1, 3, column + 1)
        subdivisions = H_SUBDIVISIONS[dim]
        sizes = 2.0 / np.asarray(subdivisions, dtype=float)
        series = {}
        for order in (1, 2, 3):
            values = []
            for n in subdivisions:
                mesh = kuhn_box_mesh(dim, n)
                _, solution, _ = solve_smooth(mesh, order)
                values.append(interior_condition_estimate(mesh, solution))
            series[f"p = {order}"] = values
            result["h"][f"d={dim} p={order}"] = {"N": list(subdivisions), "condition": values}
        rp.plot_series(ax, sizes, series, xlabel="h = 2 / N", ylabel="cond_1(S_II)",
                       title=("triangles" if dim == 2 else "tetrahedra") + ": against h")
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{size:.3g}" for size in sizes])
    ax = figure.add_subplot(1, 3, 3)
    series, xs = {}, {}
    for dim in (2, 3):
        mesh = kuhn_box_mesh(dim, 2)
        values = [interior_condition_estimate(mesh, solve_smooth(mesh, order)[1]) for order in P_ORDERS[dim]]
        name = "triangles, N = 2" if dim == 2 else "tetrahedra, N = 2"
        series[name], xs[name] = values, list(P_ORDERS[dim])
        result["p"][name] = {"orders": list(P_ORDERS[dim]), "condition": values}
    rp.plot_series(ax, xs, series, xlabel="p", ylabel="cond_1(S_II)", title="against the degree")
    figure.suptitle("Stage C: condition number of the interior trace matrix (orthonormal bases, upwind tau)")
    figure.supxlabel("cond_1 = norm_1(S_II) norm_1(S_II^{-1}), estimated by scipy.sparse.linalg.onenormest with an LU "
                     "factorization of S_II.", fontsize=7)
    pdf.savefig(figure)
    summary.setdefault("stage_c", {})["conditioning"] = result


def mesh_checks(dim: int, name: str) -> dict:
    """The numbers behind the mesh checks of the test file on one mesh (see :func:`page_checks`)."""
    mesh = named_mesh(name, dim)
    beta = BETA[dim]
    points = sample_points(mesh)
    reproduction, trace_error = 0.0, 0.0
    for order in CHECK_ORDERS[dim]:
        exact = random_polynomial(dim, order, seed=17 + order)
        gradient = polynomial_gradient(dim, order, seed=17 + order)
        expected = exact(points.reshape(-1, dim)).reshape(points.shape[:2])
        cases = [(basis, None) for basis in VOLUME_BASES] + [("orthonormal", s) for s in STABILIZATIONS[1:]]
        for basis, stabilization in cases:
            reference = reference_element(dim, order, basis)
            for trace in TRACE_BASES:
                solution = solve_transport(mesh, reference, trace, beta=beta, alpha=ALPHA,
                                           source=lambda x: gradient(x) @ beta + ALPHA * exact(x), boundary=exact,
                                           stabilization=stabilization)
                values = evaluate_cells(mesh, reference, solution.u, points)
                reproduction = max(reproduction, float(np.abs(values - expected).max()))
                traces = face_trace_projection(mesh, reference, trace, exact)
                trace_error = max(trace_error, float(np.abs(solution.uhat - traces).max()))

    def polynomial_problem(other_mesh, reference, trace, stabilization=None):
        """Solve with the polynomial data of the tests (degree p + 2, exact rules)."""
        source, boundary, rule, rule_on_faces = polynomial_data(dim, reference.order)
        return solve_transport(other_mesh, reference, trace, beta=beta, alpha=ALPHA, source=source, boundary=boundary,
                               stabilization=stabilization, rule=rule, face_rule=rule_on_faces)

    condition = 0.0
    if not mesh.boundary_faces.all():
        for order in CHECK_ORDERS[dim]:
            for trace in TRACE_BASES:
                for stabilization in STABILIZATIONS:
                    solution = polynomial_problem(mesh, reference_element(dim, order, "bernstein"), trace, stabilization)
                    condition = max(condition, float(np.linalg.cond(interior_trace_matrix(mesh, solution))))
    relabel_matrix, relabel_values = 0.0, 0.0
    basis_matrix, basis_values = 0.0, 0.0
    for trace in TRACE_BASES:
        reference = reference_element(dim, 2, "orthonormal")
        base = polynomial_problem(mesh, reference, trace)
        base_values = evaluate_cells(mesh, reference, base.u, points)
        scale = np.abs(base.matrix).max()
        for seed in range(3):
            for keep_sign in (True, False):
                other = relabel_cells(mesh, seed=100 + seed, keep_sign=keep_sign)
                solution = polynomial_problem(other, reference, trace)
                relabel_matrix = max(relabel_matrix, float(np.abs((solution.matrix - base.matrix).toarray()).max() / scale))
                values = evaluate_cells(other, reference, solution.u, points)
                relabel_values = max(relabel_values, float(np.abs(values - base_values).max()))
        for basis in VOLUME_BASES[1:]:
            other_reference = reference_element(dim, 2, basis)
            solution = polynomial_problem(mesh, other_reference, trace)
            basis_matrix = max(basis_matrix, float(np.abs((solution.matrix - base.matrix).toarray()).max() / scale))
            values = evaluate_cells(mesh, other_reference, solution.u, points)
            basis_values = max(basis_values, float(np.abs(values - base_values).max()))
    return {
        "cells": mesh.num_cells, "faces": mesh.num_faces,
        "orientations_present": sorted(set(mesh.orientations.ravel().tolist())),
        "min_abs_beta_dot_n": float(np.abs(mesh.normals @ beta).min()),
        "reproduction_max_error": reproduction, "trace_max_error": trace_error,
        "S_II_max_condition_2norm": condition,
        "relabel_S_relative_change": relabel_matrix, "relabel_u_h_change": relabel_values,
        "volume_basis_S_relative_change": basis_matrix, "volume_basis_u_h_change": basis_values,
    }


def cell_block_residuals(dim: int) -> dict:
    """Largest relative difference of A, B, C, D with quadrature on the physical cell, per cell kind (as in the test)."""
    rng = np.random.default_rng(7)
    residuals = {}
    for kind in CELL_KINDS:
        worst = 0.0
        for order in (0, 1, 2, 3, 4):
            for basis in ("orthonormal", "bernstein", "lagrange_warp_blend"):
                reference = reference_element(dim, order, basis)
                for seed in range(2):
                    vertices = cell_vertices(dim, kind, 10 * seed + order)
                    mesh, numbers = one_cell_mesh(vertices, seed + order)
                    velocity = rng.standard_normal(dim)
                    velocity /= np.linalg.norm(velocity)
                    for trace in TRACE_BASES:
                        for stabilization in STABILIZATIONS:
                            blocks = mesh_transport_blocks(mesh, reference, trace, beta=velocity, alpha=ALPHA,
                                                           stabilization=stabilization)
                            direct = direct_cell_blocks(vertices, numbers, reference, trace, velocity, ALPHA,
                                                        stabilization)
                            scale = np.abs(direct[0]).max()
                            for ours, theirs in zip(blocks, direct):
                                worst = max(worst, float(np.abs(ours[0] - theirs).max() / max(scale, np.abs(theirs).max())))
        residuals[kind] = worst
    return residuals


def two_dimensional_parity() -> dict:
    """Largest difference of u_h and uhat with ``solve_advection_reaction_hdg`` (eliminate) on the two 2D meshes."""
    from hybridge.core.space import DGSpace
    from hybridge.solvers.advection_reaction import solve_advection_reaction_hdg

    beta = BETA[2]
    source = smooth_transport_source(beta, ALPHA)
    result = {}
    for mesh_name, dg_mesh in two_dimensional_meshes().items():
        mesh = simplex_mesh(dg_mesh.node_coords, dg_mesh.triangles)
        against = dg_mesh.edges[:, 0] > dg_mesh.edges[:, 1]
        index = {tuple(sorted(edge)): e for e, edge in enumerate(dg_mesh.edges.tolist())}
        edge_of_face = np.array([index[tuple(face)] for face in mesh.faces.tolist()])
        worst_u, worst_trace = 0.0, 0.0
        for order in (1, 2, 4):
            boundary = random_polynomial(2, order, seed=41)
            for basis in ("bernstein", "hier_C0", "dub_orth"):
                reference = reference_element(2, order, basis)
                moments = cell_load(mesh, reference, source, volume_rule(2, 2 * order + 8))
                space = DGSpace(dg_mesh, order, basis_type=basis)
                for kind, trace in (("legacy-lagrange", "lagrange_warp_blend"), ("legendre-modal", "dub_orth")):
                    rotation = reference.trace_tables(trace).orientation_matrices.astype(np.float64)
                    for stabilization in STABILIZATIONS:
                        blocks = mesh_transport_blocks(mesh, reference, trace, beta=beta, alpha=ALPHA,
                                                       stabilization=stabilization)
                        ours = solve_condensed(mesh, blocks, (moments, np.zeros((mesh.num_cells, blocks[1].shape[2]))),
                                               face_trace_projection(mesh, reference, trace, boundary))
                        present = solve_advection_reaction_hdg(
                            moments, (lambda x, y: np.full_like(x, beta[0]), lambda x, y: np.full_like(x, beta[1])),
                            ALPHA, lambda x, y: boundary(np.column_stack((np.ravel(x), np.ravel(y)))).reshape(np.shape(x)),
                            space, solver="direct", boundary_mode="eliminate", advection_stabilization=stabilization,
                            trace_basis=kind, verbose=False)
                        scale = max(1.0, np.abs(ours.u).max())
                        worst_u = max(worst_u, float(np.abs(np.asarray(present.field.coeffs) - ours.u).max() / scale))
                        traces = np.asarray(present.trace).reshape(dg_mesh.num_edg, -1)
                        traces = np.where(against[:, None], traces @ rotation[1].T, traces)
                        worst_trace = max(worst_trace, float(np.abs(traces[edge_of_face] - ours.uhat).max() / scale))
        result[mesh_name] = {"u_h_relative_difference": worst_u, "uhat_relative_difference": worst_trace}
    return result


def page_checks(pdf, summary) -> None:
    """Page 4: the checks of the test file as numbers."""
    rows, data = [], {}
    for dim in (2, 3):
        for name in MESH_NAMES:
            numbers = mesh_checks(dim, name)
            data[f"d={dim} {name}"] = numbers
            rows.append([str(dim), name, str(numbers["cells"]), str(numbers["faces"]),
                         " ".join(str(o) for o in numbers["orientations_present"]),
                         f"{numbers['min_abs_beta_dot_n']:.2f}",
                         f"{numbers['reproduction_max_error']:.1e}", f"{numbers['trace_max_error']:.1e}",
                         f"{numbers['S_II_max_condition_2norm']:.1f}" if numbers["S_II_max_condition_2norm"] else "-",
                         f"{numbers['relabel_S_relative_change']:.1e}", f"{numbers['relabel_u_h_change']:.1e}",
                         f"{numbers['volume_basis_S_relative_change']:.1e}", f"{numbers['volume_basis_u_h_change']:.1e}"])
    cells = {f"d={dim}": cell_block_residuals(dim) for dim in (2, 3)}
    parity = two_dimensional_parity()
    figure = rp.new_figure(11.0, 6.5)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.4, 1.0))
    rp.plot_text_table(figure.add_subplot(grid[0, :]), rows,
                       ["d", "mesh", "cells", "faces", "orientations", "min abs(beta.n)", "max abs(u_h - u*)",
                        "max abs(uhat - P u*)", "max cond_2 S_II", "relabel: dS / max S", "relabel: du_h",
                        "basis: dS / max S", "basis: du_h"],
                       title="Mesh checks of tests/test_simplex_hdg_advection_geometry.py (alpha = 1.7)")
    rp.plot_text_table(figure.add_subplot(grid[1, 0]),
                       [[kind] + [f"{cells[f'd={dim}'][kind]:.1e}" for dim in (2, 3)] for kind in CELL_KINDS],
                       ["cell", "triangle", "tetrahedron"],
                       title="One cell: max abs(block - quadrature on the cell) / max abs(A)")
    rp.plot_text_table(figure.add_subplot(grid[1, 1]),
                       [[name, f"{value['u_h_relative_difference']:.1e}", f"{value['uhat_relative_difference']:.1e}"]
                        for name, value in parity.items()],
                       ["2D mesh", "du_h / max u_h", "duhat / max u_h"],
                       title="2D: difference with solve_advection_reaction_hdg")
    figure.supxlabel(
        "Reproduction: u* of degree p (d = 2: p in (0, 1, 2, 4); d = 3: p in (0, 1, 2, 3)), nine volume bases with "
        "the upwind tau and the orthonormal basis with tau = 2 abs(beta.n) and tau = 2.3, two trace bases.\n"
        "cond_2 S_II: largest over the same degrees, both trace bases and the three stabilizations (Bernstein volume "
        "basis).\nRelabeling: three random relabelings that keep the sign of det A_K and three that need not, p = 2; "
        "basis: the nine volume bases against the orthonormal one, p = 2.\n"
        "dS / max S: largest entry of the change of the matrix of all faces over its largest entry; du_h: largest "
        "change of u_h at sample points.\n"
        "One cell: p <= 4, three volume bases, two trace bases, the three stabilizations, a random unit velocity. "
        "2D: p in (1, 2, 4), three volume bases, two trace bases, three stabilizations.", fontsize=7)
    pdf.savefig(figure)
    summary.setdefault("stage_c", {})["checks"] = {"meshes": data, "cells": cells, "two_dimensional_parity": parity}


def add_stage_c_pages(pdf, summary) -> None:
    """Draw the four stage C pages into ``pdf`` and store their numbers in ``summary["stage_c"]``."""
    page_solutions(pdf, summary)
    page_convergence(pdf, summary)
    page_conditioning(pdf, summary)
    page_checks(pdf, summary)


__all__ = ["add_stage_c_pages"]
