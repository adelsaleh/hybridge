"""Stage C, phase 2: HDG for ``beta . grad u + alpha u = f`` on affine triangles and tetrahedra, patches and Kuhn meshes.

The local blocks are those of ``tests/simplex_hdg_transport.py`` (the transport form of
``assemble_transport_numpy``) with the geometry of ``tests/simplex_hdg_mesh.py``: the
inverse ``A_K^{-1}`` of the affine map, so that ``beta . grad v = (A_K^{-1} beta) . grad^ v^``,
the Jacobian ``|det A_K|``, the face ratios ``r_{K,f}``, the unit outward normals and the
orientation of every cell face. The velocity ``beta`` and the reaction ``alpha > 0`` are
constant and ``beta`` has a nonzero normal component on every face of every mesh used here,
so the upwind ``tau = |beta . n|`` is positive on every face.

Boundary data: the trace on every boundary face is the L2 projection of ``g``, as in
``boundary_mode="eliminate"`` of ``solve_advection_reaction_hdg``. With the upwind ``tau``
an outflow face has ``gamma = tau - beta.n = 0``, so this prescribed trace does not enter
``u_h`` there; an inflow face carries the flux ``beta.n g + |beta.n| (u - g)``.

On an interior face the transmission row gives ``uhat = (tau_1 u_1 + tau_2 u_2) / (tau_1 + tau_2)``
(the average for ``tau = k |beta.n|``), and the flux ``beta.n uhat + tau (u - uhat)`` seen by
either cell is ``beta.n`` times the value of the upwind cell: the method is the upwind
discontinuous Galerkin method, whose L2 error is ``O(h^{p+1/2})`` in general and
``O(h^{p+1})`` on many meshes.
"""

from __future__ import annotations

from itertools import permutations
from math import factorial

import numpy as np
import pytest

from simplex_hdg_helpers import random_polynomial
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
    reference_element,
    relabel_cells,
    sample_points,
    simplex_mesh,
    smooth_solution,
    solve_condensed,
    two_cell_patch,
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

VOLUME_BASES = (
    "orthonormal",
    "dub_orth",
    "bernstein",
    "hier_C0",
    "lagrange_equispaced",
    "lagrange_warp_blend",
    "lagrange_recursive",
    "lagrange_fekete",
    "lagrange_fekete_free",
)
TRACE_BASES = ("lagrange_warp_blend", "orthonormal")
ORDERS = {2: (0, 1, 2, 4), 3: (0, 1, 2, 3)}
ALPHA = 1.7
# Oblique to every face of the meshes used here: |beta . n| >= 0.18 on all of them.
BETA = {2: np.array([1.0, 0.45]), 3: np.array([1.0, 0.65, 0.3])}
# Upwind tau = |beta.n| (hybridge's default), tau = 2 |beta.n|, and an absolute tau.
STABILIZATIONS = (None, "lax-friedrichs", 2.3)


def _polynomial_source(dim: int, order: int, seed: int, beta):
    """Return ``(u*, f)`` with ``u*`` a random polynomial of degree ``order`` and ``f = beta . grad u* + ALPHA u*``."""
    exact = random_polynomial(dim, order, seed=seed)
    gradient = polynomial_gradient(dim, order, seed=seed)
    return exact, lambda x: gradient(x) @ beta + ALPHA * exact(x)


def _solve_polynomial_problem(mesh, reference, trace, *, beta, stabilization=None):
    """Solve with the data of :func:`simplex_hdg_mesh.polynomial_data` (degree p + 2, rules exact for every product)."""
    source, boundary, rule, rule_on_faces = polynomial_data(mesh.dim, reference.order)
    return solve_transport(mesh, reference, trace, beta=beta, alpha=ALPHA, source=source, boundary=boundary,
                           stabilization=stabilization, rule=rule, face_rule=rule_on_faces)


# ---------------------------------------------------------------------------
# (a) One affine cell: blocks against direct quadrature on the physical cell
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("kind", CELL_KINDS)
@pytest.mark.parametrize("trace", TRACE_BASES)
def test_cell_blocks_equal_direct_quadrature(dim, kind, trace):
    """A, B, C, D from the reference tables and the cell geometry equal quadrature on the physical cell.

    Covers random, rotated, thin, sliver and reflected (``det A_K < 0``) cells, random
    face orientations, three volume bases, p <= 4, the three stabilizations, and a
    random velocity of unit size, so that ``A_K^{-1} beta`` and the physical normals
    enter with every sign.
    """
    rng = np.random.default_rng(7)
    for order in (0, 1, 2, 3, 4):
        for basis in ("orthonormal", "bernstein", "lagrange_warp_blend"):
            reference = reference_element(dim, order, basis)
            for seed in range(2):
                vertices = cell_vertices(dim, kind, 10 * seed + order)
                mesh, numbers = one_cell_mesh(vertices, seed + order)
                beta = rng.standard_normal(dim)
                beta /= np.linalg.norm(beta)
                for stabilization in STABILIZATIONS:
                    blocks = mesh_transport_blocks(mesh, reference, trace, beta=beta, alpha=ALPHA,
                                                   stabilization=stabilization)
                    direct = direct_cell_blocks(vertices, numbers, reference, trace, beta, ALPHA, stabilization)
                    for name, ours, theirs in zip("ABCD", blocks, direct):
                        scale = max(np.abs(theirs).max(), np.abs(direct[0]).max())
                        error = np.abs(ours[0] - theirs).max()
                        assert error <= 1e-11 * scale, (order, basis, seed, stabilization, name, error / scale)


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("kind", CELL_KINDS)
def test_one_cell_reproduces_polynomials(dim, kind):
    """On one physical cell with uhat = P_F u* on every face, the local solve returns every u* of degree p."""
    for order in ORDERS[dim]:
        for basis in ("orthonormal", "hier_C0", "lagrange_fekete"):
            reference = reference_element(dim, order, basis)
            mesh, _ = one_cell_mesh(cell_vertices(dim, kind, order), order)
            exact, source = _polynomial_source(dim, order, order + 3, BETA[dim])
            points = sample_points(mesh)
            expected = exact(points.reshape(-1, dim)).reshape(points.shape[:2])
            for stabilization in STABILIZATIONS:
                solution = solve_transport(mesh, reference, "lagrange_warp_blend", beta=BETA[dim], alpha=ALPHA,
                                           source=source, boundary=exact, stabilization=stabilization)
                values = evaluate_cells(mesh, reference, solution.u, points)
                a = mesh_transport_blocks(mesh, reference, "lagrange_warp_blend", beta=BETA[dim], alpha=ALPHA,
                                          stabilization=stabilization)[0][0]
                scale = max(1.0, np.abs(expected).max())
                tolerance = (1e-14 * np.linalg.cond(a) + 1e-12) * scale
                assert np.abs(values - expected).max() <= tolerance, (order, basis, stabilization)


# ---------------------------------------------------------------------------
# (b) Patches and Kuhn meshes: reproduction, nonsingular trace system, relabeling, basis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("name", MESH_NAMES)
def test_velocity_is_oblique_to_every_face(dim, name):
    """|beta . n| >= 0.18 on every cell face of the test meshes, so tau > 0 and every interior D block is invertible."""
    mesh = named_mesh(name, dim)
    assert np.abs(mesh.normals @ BETA[dim]).min() >= 0.18


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("name", MESH_NAMES)
@pytest.mark.parametrize("trace", TRACE_BASES)
def test_mesh_reproduces_polynomials(dim, name, trace):
    """For u* of degree p and f = beta . grad u* + alpha u*, u_h = u* and uhat = P_F u* on every face.

    Every volume basis with the upwind tau; the other two stabilizations with the
    orthonormal volume basis.
    """
    mesh = named_mesh(name, dim)
    points = sample_points(mesh)
    for order in ORDERS[dim]:
        exact, source = _polynomial_source(dim, order, 17 + order, BETA[dim])
        expected = exact(points.reshape(-1, dim)).reshape(points.shape[:2])
        cases = [(basis, None) for basis in VOLUME_BASES] + [("orthonormal", s) for s in STABILIZATIONS[1:]]
        for basis, stabilization in cases:
            reference = reference_element(dim, order, basis)
            solution = solve_transport(mesh, reference, trace, beta=BETA[dim], alpha=ALPHA, source=source,
                                       boundary=exact, stabilization=stabilization)
            assert solution.solve is None or solution.solve.converged
            values = evaluate_cells(mesh, reference, solution.u, points)
            a = mesh_transport_blocks(mesh, reference, trace, beta=BETA[dim], alpha=ALPHA,
                                      stabilization=stabilization)[0]
            scale = max(1.0, np.abs(expected).max())
            tolerance = (1e-14 * np.linalg.cond(a).max() + 1e-12) * scale
            error = np.abs(values - expected).max()
            assert error <= tolerance, (order, basis, stabilization, error)
            traces = face_trace_projection(mesh, reference, trace, exact)
            assert np.abs(solution.uhat - traces).max() <= tolerance, (order, basis, stabilization)


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("name", MESH_NAMES)
def test_interior_trace_matrix_is_nonsingular(dim, name):
    """The trace matrix S_II of the interior faces (the one that is solved) is nonsingular; its condition stays moderate.

    ``S_II`` is not symmetric. Its 2-norm condition number is at most ``1e6`` for p <= 4
    (2D) and p <= 3 (3D) with the nodal and the orthonormal trace basis; the largest
    value found is 174 (perturbed Kuhn box, d = 3; see the report). With the upwind
    tau the outflow boundary faces have zero columns in the matrix of all faces,
    which is therefore singular; those faces are prescribed and not solved for.
    """
    mesh = named_mesh(name, dim)
    if mesh.boundary_faces.all():
        pytest.skip("no interior face")
    for order in ORDERS[dim]:
        for trace in TRACE_BASES:
            for stabilization in STABILIZATIONS:
                solution = _solve_polynomial_problem(mesh, reference_element(dim, order, "bernstein"), trace,
                                                     beta=BETA[dim], stabilization=stabilization)
                singular_values = np.linalg.svd(interior_trace_matrix(mesh, solution), compute_uv=False)
                condition = singular_values[0] / singular_values[-1]
                assert condition <= 1e6, (order, trace, stabilization, condition)


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("name", MESH_NAMES)
@pytest.mark.parametrize("keep_sign", (True, False), ids=("even", "any"))
def test_solution_does_not_change_under_relabeling(dim, name, keep_sign):
    """Permuting the local vertex order of every cell leaves S, uhat and u_h (at physical points) unchanged.

    ``even`` keeps the sign of det A_K; ``any`` also allows odd permutations, which make
    det A_K negative. The faces keep their global order and numbering, so the global
    matrices are compared entry by entry.
    """
    mesh = named_mesh(name, dim)
    points = sample_points(mesh)
    for order in (1, 2, 3):
        for trace in TRACE_BASES:
            reference = reference_element(dim, order, "orthonormal")
            for stabilization in (None, 2.3):
                base = _solve_polynomial_problem(mesh, reference, trace, beta=BETA[dim], stabilization=stabilization)
                base_values = evaluate_cells(mesh, reference, base.u, points)
                scale = np.abs(base.matrix).max()
                for seed in range(3):
                    relabeled = relabel_cells(mesh, seed=100 + seed, keep_sign=keep_sign)
                    solution = _solve_polynomial_problem(relabeled, reference, trace, beta=BETA[dim],
                                                         stabilization=stabilization)
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
    for trace in TRACE_BASES:
        reference = reference_element(dim, 2, "hier_C0")
        base = _solve_polynomial_problem(mesh, reference, trace, beta=BETA[dim])
        base_values = evaluate_cells(mesh, reference, base.u, points)
        seen = set()
        for cell in (0, 1):
            for local in permutations(range(dim + 1)):
                cells = mesh.cells.copy()
                cells[cell] = mesh.cells[cell, list(local)]
                relabeled = simplex_mesh(mesh.vertices, cells)
                side = int(np.flatnonzero(relabeled.cell_faces[cell] == shared)[0])
                seen.add(int(relabeled.orientations[cell, side]))
                solution = _solve_polynomial_problem(relabeled, reference, trace, beta=BETA[dim])
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
            solution = _solve_polynomial_problem(mesh, reference, trace, beta=BETA[dim])
            results[basis] = (solution, evaluate_cells(mesh, reference, solution.u, points))
        base, base_values = results["orthonormal"]
        scale = np.abs(base.matrix).max()
        for basis, (solution, values) in results.items():
            assert np.abs((solution.matrix - base.matrix).toarray()).max() <= 1e-11 * scale, (order, basis)
            assert np.abs(solution.uhat - base.uhat).max() <= 1e-10 * max(1.0, np.abs(base.uhat).max())
            assert np.abs(values - base_values).max() <= 1e-10 * max(1.0, np.abs(base_values).max())


# ---------------------------------------------------------------------------
# (c) Convergence on Kuhn box meshes for a smooth solution
# ---------------------------------------------------------------------------


H_SUBDIVISIONS = {2: (4, 8, 16), 3: (2, 4, 6)}


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("order", (1, 2, 3))
def test_h_convergence_on_the_kuhn_box(dim, order):
    """The L2 error of u_h decreases at rate p + 1 (last rate >= p + 0.9) and stays within a factor 1.7 of the L2 projection.

    ``u = exp((x_1 + ... + x_d) / 2) cos(pi x_1 / 2)``, ``alpha = 1``, upwind ``tau``,
    ``beta = (1, 0.45)`` or ``(1, 0.65, 0.3)``, oblique to every face of the Kuhn
    mesh. The theory of upwind DG guarantees only ``p + 1/2``; these meshes give
    ``p + 1``: rates 1.90 to 4.01 up to N = 32 in 2D and N = 8 in 3D, with the
    error at most 1.62 times the projection error (see the report).
    """
    subdivisions = H_SUBDIVISIONS[dim]
    errors, projection_errors = transport_convergence(dim, order, subdivisions, beta=BETA[dim])
    rates = np.log(np.divide(errors[:-1], errors[1:])) / np.log(np.divide(subdivisions[1:], subdivisions[:-1]))
    assert rates[-1] >= order + 0.9, (errors, rates)
    assert np.all(np.divide(errors, projection_errors) <= 1.7), (errors, projection_errors)


@pytest.mark.parametrize("dim", (2, 3))
def test_p_convergence_on_the_kuhn_box(dim):
    """At N = 2 the L2 error of u_h decreases by a factor of at least 4 per degree (exponential convergence in p)."""
    orders = range(1, 9) if dim == 2 else range(1, 7)
    errors = [transport_convergence(dim, order, (2,), beta=BETA[dim])[0][0] for order in orders]
    assert np.all(np.divide(errors[:-1], errors[1:]) >= 4.0), errors


# ---------------------------------------------------------------------------
# (d) Two-dimensional parity with solve_advection_reaction_hdg
# ---------------------------------------------------------------------------


TWO_D_TRACES = (("legacy-lagrange", "lagrange_warp_blend"), ("legendre-modal", "dub_orth"))


@pytest.mark.parametrize("order", (1, 2, 4))
@pytest.mark.parametrize("basis", ("bernstein", "hier_C0", "dub_orth"))
@pytest.mark.parametrize("kind, trace", TWO_D_TRACES)
def test_two_dimensional_solution_equals_solve_advection_reaction_hdg(order, basis, kind, trace):
    """For beta != 0, ``solve_advection_reaction_hdg`` (NumPy, eliminate) solves the same system: equal u_h and uhat.

    On ``rectangle_mesh`` and a perturbed Kuhn ``DGMesh``, with the velocity as two
    constant callables and the stabilizations ``None`` (upwind), ``"lax-friedrichs"``
    and the absolute ``tau = 2.3``. Both solves get the same source moments (the
    smooth source on a fine rule) and polynomial boundary data of degree p, which
    both project exactly. hybridge orients an edge by ``DGMesh.edges``; the trace of
    an edge whose ``DGMesh`` direction runs against the increasing vertex order is
    converted with the reversal matrix.
    """
    from hybridge.core.space import DGSpace
    from hybridge.solvers.advection_reaction import solve_advection_reaction_hdg

    beta = BETA[2]
    boundary = random_polynomial(2, order, seed=41)
    source = smooth_transport_source(beta, ALPHA)
    for dg_mesh in two_dimensional_meshes().values():
        mesh = simplex_mesh(dg_mesh.node_coords, dg_mesh.triangles)
        reference = reference_element(2, order, basis)
        moments = cell_load(mesh, reference, source, volume_rule(2, 2 * order + 8))
        rotation = reference.trace_tables(trace).orientation_matrices.astype(np.float64)
        against = dg_mesh.edges[:, 0] > dg_mesh.edges[:, 1]
        index = {tuple(sorted(edge)): e for e, edge in enumerate(dg_mesh.edges.tolist())}
        edge_of_face = np.array([index[tuple(face)] for face in mesh.faces.tolist()])
        space = DGSpace(dg_mesh, order, basis_type=basis)
        for stabilization in STABILIZATIONS:
            blocks = mesh_transport_blocks(mesh, reference, trace, beta=beta, alpha=ALPHA, stabilization=stabilization)
            ours = solve_condensed(mesh, blocks, (moments, np.zeros((mesh.num_cells, blocks[1].shape[2]))),
                                   face_trace_projection(mesh, reference, trace, boundary))
            result = solve_advection_reaction_hdg(
                moments, (lambda x, y: np.full_like(x, beta[0]), lambda x, y: np.full_like(x, beta[1])), ALPHA,
                lambda x, y: boundary(np.column_stack((np.ravel(x), np.ravel(y)))).reshape(np.shape(x)),
                space, solver="direct", boundary_mode="eliminate", advection_stabilization=stabilization,
                trace_basis=kind, verbose=False)
            scale = max(1.0, np.abs(ours.u).max())
            assert np.abs(np.asarray(result.field.coeffs) - ours.u).max() <= 1e-11 * scale, stabilization
            present = np.asarray(result.trace).reshape(dg_mesh.num_edg, -1)
            present = np.where(against[:, None], present @ rotation[1].T, present)
            assert np.abs(present[edge_of_face] - ours.uhat).max() <= 1e-11 * scale, stabilization
