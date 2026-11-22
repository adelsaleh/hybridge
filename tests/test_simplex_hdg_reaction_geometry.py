"""Stage B: HDG for ``alpha u = f`` on affine triangles and tetrahedra, patches and Kuhn box meshes.

The local problem on a cell ``K`` is ``(alpha u, v)_K + <tau (u - uhat), v>_dK = (f, v)_K``
and the transmission row on an interior face is ``sum_K <tau (u - uhat), mu>_F = 0``.
Its blocks are the reference blocks of stage A with the cell Jacobian ``J_K`` and the
face-measure ratios ``r_{K,f}``, and with the coupling of the orientation through which
the cell sees each face (``tests/simplex_hdg_mesh.py`` computes the geometry and the
orientations; ``tests/simplex_hdg_helpers.py`` the blocks).

With the same ``tau`` on both sides, the transmission row gives ``uhat = P_F avg(u)``
on an interior face (``P_F`` the L2 projection onto ``P_p(F)``, which keeps the
traces of ``P_p`` functions on a flat face), so ``u_h`` minimizes

    E(v) = ||v - u||_alpha^2 + (tau / 2) sum_{F interior} ||[[v]]||_F^2 + tau sum_{F on dOmega} ||v - P_F g||_F^2

over the discontinuous ``P_p`` functions (``u`` the exact solution, ``g`` the boundary
data, ``||w||_alpha^2 = (alpha w, w)``). Hence a polynomial ``u`` of degree ``p`` is
reproduced exactly, the global condensed matrix is symmetric positive definite, and
``||u_h - u||_alpha^2 <= E(I_h u) <= ||I_h u - u||_alpha^2 + tau ||I_h u - u||_dOmega^2 = O(h^{2p+2})``
for the continuous ``P_p`` interpolant ``I_h u``.
"""

from __future__ import annotations

from itertools import permutations
from math import factorial

import numpy as np
import pytest

from hybridge.core.quadrature import _evaluate_basis, _evaluate_simplex_basis
from hybridge.core.simplex.reference import face_permutations, reference_simplex
from simplex_hdg_helpers import random_polynomial, reaction_blocks
from simplex_hdg_mesh import (
    CELL_KINDS,
    MESH_NAMES,
    cell_load,
    cell_vertices,
    evaluate_cells,
    face_trace_projection,
    named_mesh,
    one_cell_mesh,
    polynomial_data,
    quadratic_alpha,
    reaction_convergence,
    reference_coordinates,
    reference_element,
    relabel_cells,
    sample_points,
    simplex_mesh,
    smooth_source,
    solve_condensed,
    solve_polynomial_problem,
    solve_reaction,
    two_cell_patch,
    two_dimensional_meshes,
    volume_rule,
)

VOLUME_BASES = ("orthonormal", "dub_orth", "bernstein", "hier_C0", "lagrange_warp_blend", "lagrange_fekete")
TRACE_BASES = ("lagrange_warp_blend", "orthonormal")
ORDERS = {2: (0, 1, 2, 4), 3: (0, 1, 2, 3)}
ALPHA = 1.7
TAU = 2.3


# ---------------------------------------------------------------------------
# (a) One affine cell: geometry and blocks against direct quadrature
# ---------------------------------------------------------------------------


def _direct_blocks(vertices, numbers, reference, trace, alpha, tau):
    """Blocks of ``alpha u = f`` on one physical cell by quadrature on the cell itself.

    Volume points are ``x_q = sum_k lambda_k(xi_q) X_k`` with weights scaled by
    ``|K| / |K^|`` (``|K| = |det(X_1 - X_0, ...)| / d!``); basis values are taken at the
    reference point found from the barycentric coordinates of ``x_q``. Face ``f`` is
    parametrized with its vertices sorted by mesh number (the global order) and its
    weights scaled by ``|F| / |F^|`` (``|F|``: edge length or half the cross product).
    The trace basis is evaluated at the reference-face points without permutation.
    """
    dim = vertices.shape[1]
    simplex = reference_simplex(dim)
    order = reference.order
    points, weights = reference._generic_volume_rule
    measure = abs(np.linalg.det((vertices[1:] - vertices[0]).T)) / factorial(dim)
    x = simplex.barycentric(points) @ vertices
    phi = np.asarray(_evaluate_basis(reference.basis_type, order, reference_coordinates(vertices, x)), dtype=np.float64)
    w = weights * measure / simplex.measure
    a = alpha * phi.T @ (w[:, None] * phi)
    face_points, face_weights = reference._generic_face_rule
    face_simplex = reference_simplex(dim - 1)
    mu = np.asarray(_evaluate_simplex_basis(trace, dim - 1, order, face_points, gradients=False), dtype=np.float64)
    columns, diagonal = [], []
    for local in simplex.face_vertices:
        corners = vertices[sorted(local, key=lambda k: numbers[k])]
        if dim == 2:
            area = np.linalg.norm(corners[1] - corners[0])
        else:
            area = 0.5 * np.linalg.norm(np.cross(corners[1] - corners[0], corners[2] - corners[0]))
        xf = face_simplex.barycentric(face_points) @ corners
        phi_f = np.asarray(_evaluate_basis(reference.basis_type, order, reference_coordinates(vertices, xf)), dtype=np.float64)
        wf = face_weights * area / face_simplex.measure
        a += tau * phi_f.T @ (wf[:, None] * phi_f)
        columns.append(tau * phi_f.T @ (wf[:, None] * mu))
        diagonal.append(tau * mu.T @ (wf[:, None] * mu))
    b = np.concatenate(columns, axis=1)
    n_t = mu.shape[1]
    d = np.zeros((len(diagonal) * n_t,) * 2)
    for f, block in enumerate(diagonal):
        d[f * n_t:(f + 1) * n_t, f * n_t:(f + 1) * n_t] = block
    return a, b, b.T, d


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("kind", CELL_KINDS)
def test_cell_geometry_equals_direct_formulas(dim, kind):
    """J_K, r_{K,f}, the outward normals and the orientations of one cell equal formulas computed from its vertices."""
    for seed in range(4):
        vertices = cell_vertices(dim, kind, seed)
        mesh, numbers = one_cell_mesh(vertices, seed)
        simplex = reference_simplex(dim)
        volume = abs(np.linalg.det((vertices[1:] - vertices[0]).T)) / factorial(dim)
        assert np.isclose(mesh.jacobians[0], volume / simplex.measure, rtol=1e-13)
        centroid = vertices.mean(axis=0)
        for f, local in enumerate(simplex.face_vertices):
            corners = vertices[list(local)]
            if dim == 2:
                area = np.linalg.norm(corners[1] - corners[0])
            else:
                area = 0.5 * np.linalg.norm(np.cross(corners[1] - corners[0], corners[2] - corners[0]))
            assert np.isclose(mesh.face_ratios[0, f], area / reference_simplex(dim - 1).measure, rtol=1e-12)
            normal = mesh.normals[0, f]
            assert np.isclose(np.linalg.norm(normal), 1.0)
            assert np.abs((corners[1:] - corners[0]) @ normal).max() <= 1e-12 * np.abs(corners).max()
            assert normal @ (corners.mean(axis=0) - centroid) > 0.0
            sigma = face_permutations(dim)[mesh.orientations[0, f]]
            glob = sorted(numbers[list(local)])
            assert all(glob[sigma[i]] == numbers[local[i]] for i in range(dim))
        assert np.abs(np.einsum("f,fd->d", mesh.face_ratios[0], mesh.normals[0])).max() <= 1e-12 * mesh.face_ratios.max()


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("kind", CELL_KINDS)
@pytest.mark.parametrize("trace", TRACE_BASES)
def test_cell_blocks_equal_direct_quadrature(dim, kind, trace):
    """The blocks from the reference tables with J_K, r_{K,f} and the orientations equal quadrature on the cell."""
    for order in (0, 1, 2, 3, 4):
        for basis in ("orthonormal", "bernstein", "lagrange_warp_blend"):
            reference = reference_element(dim, order, basis)
            for seed in range(2):
                vertices = cell_vertices(dim, kind, 10 * seed + order)
                mesh, numbers = one_cell_mesh(vertices, seed + order)
                blocks = reaction_blocks(reference, trace, alpha=ALPHA, tau=TAU, orientations=mesh.orientations[0],
                                         jacobian=mesh.jacobians[0], face_ratios=mesh.face_ratios[0])
                direct = _direct_blocks(vertices, numbers, reference, trace, ALPHA, TAU)
                for name, ours, theirs in zip("ABCD", blocks, direct):
                    scale = np.abs(theirs).max()
                    assert np.abs(ours - theirs).max() <= 1e-11 * scale, (order, basis, name)


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("kind", CELL_KINDS)
def test_one_cell_reproduces_polynomials(dim, kind):
    """On one physical cell with uhat = u* on every face, the local solve returns every u* of degree p."""
    for order in ORDERS[dim]:
        for basis in ("orthonormal", "hier_C0", "lagrange_fekete"):
            reference = reference_element(dim, order, basis)
            vertices = cell_vertices(dim, kind, order)
            mesh, _ = one_cell_mesh(vertices, order)
            exact = random_polynomial(dim, order, seed=order + 3)
            solution = solve_reaction(mesh, reference, "lagrange_warp_blend", alpha=ALPHA, tau=TAU,
                                      source=lambda x: ALPHA * exact(x), boundary=exact)
            points = sample_points(mesh)
            values = evaluate_cells(mesh, reference, solution.u, points)
            expected = exact(points.reshape(-1, dim)).reshape(values.shape)
            a = reaction_blocks(reference, "lagrange_warp_blend", alpha=ALPHA, tau=TAU,
                                jacobian=mesh.jacobians[0], face_ratios=mesh.face_ratios[0])[0]
            scale = max(1.0, np.abs(expected).max())
            assert np.abs(values - expected).max() <= (1e-14 * np.linalg.cond(a) + 1e-12) * scale, (order, basis)


# ---------------------------------------------------------------------------
# (b) Patches and the Kuhn box: reproduction, positive definiteness, relabeling, basis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("name", MESH_NAMES)
@pytest.mark.parametrize("basis", VOLUME_BASES)
@pytest.mark.parametrize("trace", TRACE_BASES)
def test_mesh_reproduces_polynomials(dim, name, basis, trace):
    """For u* of degree p, alpha constant and uhat = u* on the boundary, u_h = u* and uhat = P_F u* on every face."""
    mesh = named_mesh(name, dim)
    points = sample_points(mesh)
    for order in ORDERS[dim]:
        reference = reference_element(dim, order, basis)
        exact = random_polynomial(dim, order, seed=17 + order)
        solution = solve_reaction(mesh, reference, trace, alpha=ALPHA, tau=TAU,
                                  source=lambda x: ALPHA * exact(x), boundary=exact)
        assert solution.solve is None or solution.solve.converged
        values = evaluate_cells(mesh, reference, solution.u, points)
        expected = exact(points.reshape(-1, dim)).reshape(values.shape)
        a = reaction_blocks(reference, trace, alpha=ALPHA, tau=TAU, jacobian=mesh.jacobians,
                            face_ratios=mesh.face_ratios)[0]
        condition = np.linalg.cond(a).max()
        scale = max(1.0, np.abs(expected).max())
        tolerance = (1e-14 * condition + 1e-12) * scale
        assert np.abs(values - expected).max() <= tolerance, (order, np.abs(values - expected).max(), condition)
        traces = face_trace_projection(mesh, reference, trace, exact)
        assert np.abs(solution.uhat - traces).max() <= tolerance


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("name", MESH_NAMES)
@pytest.mark.parametrize("trace", TRACE_BASES)
def test_global_condensed_matrix_is_symmetric_positive_definite(dim, name, trace):
    """The global trace matrix S = sum_K S_K (all faces) is symmetric positive definite, for constant and variable alpha."""
    mesh = named_mesh(name, dim)
    for order in ORDERS[dim]:
        reference = reference_element(dim, order, "bernstein")
        source, boundary, rule, rule_on_faces = polynomial_data(dim, order)
        for alpha in (ALPHA, quadratic_alpha):
            solution = solve_reaction(mesh, reference, trace, alpha=alpha, tau=TAU, source=source,
                                      boundary=boundary, rule=rule, face_rule=rule_on_faces)
            matrix = solution.matrix.toarray()
            scale = np.abs(matrix).max()
            assert np.abs(matrix - matrix.T).max() <= 1e-13 * scale
            eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
            assert eigenvalues.min() > 1e-8 * eigenvalues.max(), (order, eigenvalues.min(), eigenvalues.max())


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("name", MESH_NAMES)
@pytest.mark.parametrize("keep_sign", (True, False), ids=("even", "any"))
def test_solution_does_not_change_under_relabeling(dim, name, keep_sign):
    """Permuting the local vertex order of every cell leaves S, uhat and u_h (at physical points) unchanged.

    ``even`` keeps the sign of det A_K; ``any`` also allows odd permutations, which
    make det A_K negative. The faces keep their global order and numbering, so the
    global matrices are compared entry by entry.
    """
    mesh = named_mesh(name, dim)
    points = sample_points(mesh)
    for order in (1, 2, 3):
        for trace in TRACE_BASES:
            reference = reference_element(dim, order, "orthonormal")
            base = solve_polynomial_problem(mesh, reference, trace, tau=TAU)
            base_values = evaluate_cells(mesh, reference, base.u, points)
            for seed in range(3):
                relabeled = relabel_cells(mesh, seed=100 + seed, keep_sign=keep_sign)
                assert np.array_equal(relabeled.faces, mesh.faces)
                solution = solve_polynomial_problem(relabeled, reference, trace, tau=TAU)
                scale = np.abs(base.matrix).max()
                assert np.abs((solution.matrix - base.matrix).toarray()).max() <= 1e-12 * scale
                assert np.abs(solution.uhat - base.uhat).max() <= 1e-11 * max(1.0, np.abs(base.uhat).max())
                values = evaluate_cells(relabeled, reference, solution.u, points)
                assert np.abs(values - base_values).max() <= 1e-11 * max(1.0, np.abs(base_values).max())


@pytest.mark.parametrize("dim", (2, 3))
def test_every_orientation_of_the_shared_face(dim):
    """On the two-cell patch, every local order of either cell gives the same S and u_h; all d! orientations occur."""
    mesh = two_cell_patch(dim)
    shared = int(np.flatnonzero(~mesh.boundary_faces)[0])
    points = sample_points(mesh)
    order = 2
    for trace in TRACE_BASES:
        reference = reference_element(dim, order, "hier_C0")
        base = solve_polynomial_problem(mesh, reference, trace, tau=TAU)
        base_values = evaluate_cells(mesh, reference, base.u, points)
        seen = set()
        for cell in (0, 1):
            for local in permutations(range(dim + 1)):
                cells = mesh.cells.copy()
                cells[cell] = mesh.cells[cell, list(local)]
                relabeled = simplex_mesh(mesh.vertices, cells)
                side = int(np.flatnonzero(relabeled.cell_faces[cell] == shared)[0])
                seen.add(int(relabeled.orientations[cell, side]))
                solution = solve_polynomial_problem(relabeled, reference, trace, tau=TAU)
                assert np.abs((solution.matrix - base.matrix).toarray()).max() <= 1e-12 * np.abs(base.matrix).max()
                values = evaluate_cells(relabeled, reference, solution.u, points)
                assert np.abs(values - base_values).max() <= 1e-11 * max(1.0, np.abs(base_values).max())
        assert seen == set(range(factorial(dim)))


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("name", MESH_NAMES)
@pytest.mark.parametrize("trace", TRACE_BASES)
def test_condensed_matrix_and_solution_do_not_depend_on_the_volume_basis(dim, name, trace):
    """S, uhat and u_h (at physical points) are the same for every volume basis."""
    mesh = named_mesh(name, dim)
    points = sample_points(mesh)
    for order in ORDERS[dim][1:]:
        results = {}
        for basis in VOLUME_BASES:
            reference = reference_element(dim, order, basis)
            solution = solve_polynomial_problem(mesh, reference, trace, tau=TAU)
            results[basis] = (solution, evaluate_cells(mesh, reference, solution.u, points))
        base, base_values = results["orthonormal"]
        scale = np.abs(base.matrix).max()
        for basis, (solution, values) in results.items():
            assert np.abs((solution.matrix - base.matrix).toarray()).max() <= 1e-11 * scale, (order, basis)
            assert np.abs(solution.uhat - base.uhat).max() <= 1e-10 * max(1.0, np.abs(base.uhat).max())
            assert np.abs(values - base_values).max() <= 1e-10 * max(1.0, np.abs(base_values).max())


# ---------------------------------------------------------------------------
# (c) Convergence on the Kuhn box for a smooth solution and variable alpha
# ---------------------------------------------------------------------------


H_SUBDIVISIONS = {2: (4, 8, 16), 3: (2, 4, 6)}


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("order", (1, 2, 3))
def test_h_convergence_on_the_kuhn_box(dim, order):
    """The L2 error of u_h decreases at rate close to p + 1 and stays within a factor 2.5 of the L2 projection error.

    With tau fixed, the jump term weighs ``tau / h`` against the mass, so the error
    moves from that of the L2 projection (coarse meshes) towards that of the
    continuous interpolant (fine meshes); both are O(h^{p+1}), and the observed rate
    approaches p + 1 from below (2.8 to 2.9 at p = 2 on these meshes).
    """
    subdivisions = H_SUBDIVISIONS[dim]
    errors, projection_errors = reaction_convergence(dim, order, subdivisions)
    rates = np.log(np.divide(errors[:-1], errors[1:])) / np.log(np.divide(subdivisions[1:], subdivisions[:-1]))
    assert rates[-1] >= order + 1 - 0.3, (errors, rates)
    assert np.all(np.divide(errors, projection_errors) <= 2.5), (errors, projection_errors)


@pytest.mark.parametrize("dim", (2, 3))
def test_p_convergence_on_the_kuhn_box(dim):
    """At N = 2 the L2 error of u_h decreases by a factor of at least 5 per degree (exponential convergence in p)."""
    orders = range(1, 9) if dim == 2 else range(1, 7)
    errors = [reaction_convergence(dim, order, (2,))[0][0] for order in orders]
    assert np.all(np.divide(errors[:-1], errors[1:]) >= 5.0), errors
    assert errors[-1] <= (1e-8 if dim == 2 else 1e-5)


# ---------------------------------------------------------------------------
# (d) Two-dimensional parity with hybridge.hdg.matrices and solve_advection_reaction_hdg
# ---------------------------------------------------------------------------


TWO_D_TRACES = (("legacy-lagrange", "lagrange_warp_blend"), ("legendre-modal", "dub_orth"))


@pytest.mark.parametrize("order", (0, 1, 2, 4))
@pytest.mark.parametrize("basis", ("bernstein", "hier_C0", "dub_orth"))
@pytest.mark.parametrize("kind, trace", TWO_D_TRACES)
def test_two_dimensional_blocks_equal_the_present_assembly(order, basis, kind, trace):
    """On 2D meshes the blocks equal those of hdg/matrices.py at zero velocity, after the change of edge orientation.

    ``hdg/matrices.py`` gives ``B`` in the local orientation of each element (our
    orientation 0) and ``C`` in the orientation of ``DGMesh.edges`` (the first element
    that lists the edge). Ours use the increasing vertex order. With ``R`` the
    reversal matrix of the trace basis: ``B = B_present R_o^T`` (``o`` our orientation)
    and ``C = R_e C_present`` (``R_e = R`` when ``DGMesh.edges`` runs against the
    increasing order).
    """
    from hybridge.core.space import DGSpace
    from hybridge.hdg.matrices import (
        advection_trace_lift_from_stabilization,
        boundary_mass_from_trace_stabilization,
        element_boundary_mats_from_trace_weight,
    )

    for dg_mesh in two_dimensional_meshes().values():
        mesh = simplex_mesh(dg_mesh.node_coords, dg_mesh.triangles)
        space = DGSpace(dg_mesh, order, basis_type=basis)
        trace_space = space.trace_space(kind)
        tau_face = np.full((dg_mesh.num_tri, 3, trace_space.quads.size), TAU)
        boundary_mass = boundary_mass_from_trace_stabilization(space, tau_face, trace_space=trace_space)
        element_boundary = element_boundary_mats_from_trace_weight(space, tau_face, trace_space=trace_space)
        lift = advection_trace_lift_from_stabilization(space, tau_face, trace_space=trace_space)

        reference = reference_element(2, order, basis)
        a, b, c, _ = reaction_blocks(reference, trace, alpha=ALPHA, tau=TAU, orientations=mesh.orientations,
                                     jacobian=mesh.jacobians, face_ratios=mesh.face_ratios)
        rotation = reference.trace_tables(trace).orientation_matrices.astype(np.float64)
        n_t = rotation.shape[1]
        a_present = ALPHA * dg_mesh.aff_jacs[:, None, None] * space.quad_data.MKrf + boundary_mass
        assert np.allclose(mesh.jacobians, dg_mesh.aff_jacs, rtol=1e-14)
        assert np.abs(a - a_present).max() <= 1e-12 * np.abs(a).max()
        edges = dg_mesh.edges[dg_mesh.loc2glob_edge]  # (n_K, 3, 2) in the DGMesh direction
        against = (edges[:, :, 0] > edges[:, :, 1]).astype(np.int64)
        present = np.where(dg_mesh.orientations, 0, 1)
        assert np.array_equal(mesh.orientations, present ^ against)
        b_present = element_boundary.reshape(dg_mesh.num_tri, -1, 3, n_t)
        c_present = lift  # (n_K, 3, n_t, n_u)
        expected_b = np.einsum("kifa,kfba->kifb", b_present, rotation[mesh.orientations])
        expected_c = np.einsum("kfab,kfbi->kfai", rotation[against], c_present)
        assert np.abs(b - expected_b.reshape(b.shape)).max() <= 1e-12 * np.abs(b).max()
        assert np.abs(c - expected_c.reshape(c.shape)).max() <= 1e-12 * np.abs(c).max()


@pytest.mark.parametrize("order", (1, 2, 4))
@pytest.mark.parametrize("basis", ("bernstein", "dub_orth"))
@pytest.mark.parametrize("kind, trace", TWO_D_TRACES)
def test_two_dimensional_solution_equals_solve_advection_reaction_hdg(order, basis, kind, trace):
    """With beta = 0 and a constant tau, ``solve_advection_reaction_hdg`` solves the same system: equal u_h and uhat.

    Both solves get the same source moments (a non-polynomial source on a fine
    rule) and polynomial boundary data of degree p, which both project exactly.
    """
    from hybridge.core.space import DGSpace
    from hybridge.solvers.advection_reaction import solve_advection_reaction_hdg

    boundary = random_polynomial(2, order, seed=41)
    for dg_mesh in two_dimensional_meshes().values():
        mesh = simplex_mesh(dg_mesh.node_coords, dg_mesh.triangles)
        reference = reference_element(2, order, basis)
        moments = cell_load(mesh, reference, smooth_source, volume_rule(2, 2 * order + 8))
        blocks = reaction_blocks(reference, trace, alpha=ALPHA, tau=TAU, orientations=mesh.orientations,
                                 jacobian=mesh.jacobians, face_ratios=mesh.face_ratios)
        ours = solve_condensed(mesh, blocks, (moments, np.zeros((mesh.num_cells, blocks[1].shape[2]))),
                               face_trace_projection(mesh, reference, trace, boundary))
        space = DGSpace(dg_mesh, order, basis_type=basis)
        result = solve_advection_reaction_hdg(
            moments, (lambda x, y: 0.0 * x, lambda x, y: 0.0 * x), ALPHA,
            lambda x, y: boundary(np.column_stack((x.ravel(), y.ravel()))).reshape(np.shape(x)),
            space, solver="direct", boundary_mode="eliminate", advection_stabilization=TAU,
            trace_basis=kind, verbose=False)
        scale = max(1.0, np.abs(ours.u).max())
        assert np.abs(np.asarray(result.field.coeffs) - ours.u).max() <= 1e-11 * scale
        rotation = reference.trace_tables(trace).orientation_matrices.astype(np.float64)
        present = np.asarray(result.trace).reshape(dg_mesh.num_edg, -1)
        against = dg_mesh.edges[:, 0] > dg_mesh.edges[:, 1]
        present = np.where(against[:, None], present @ rotation[1].T, present)
        # Our faces are the sorted vertex pairs; match them to the DGMesh edges.
        index = {tuple(sorted(edge)): e for e, edge in enumerate(dg_mesh.edges.tolist())}
        order_ = np.array([index[tuple(face)] for face in mesh.faces.tolist()])
        assert np.abs(present[order_] - ours.uhat).max() <= 1e-11 * scale
