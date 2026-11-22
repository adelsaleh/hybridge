"""Stage A: HDG for ``alpha u = f`` on the reference triangle and the reference tetrahedron.

The local problem on one element is ``(alpha u, v)_K + <tau (u - uhat), v>_dK = (f, v)_K``
with the transmission row ``<tau (u - uhat), mu>_dK``. Its blocks use only the mass,
face-mass, element-trace coupling and trace-mass tables of ``ReferenceElementData``,
so these tests check those tables before any derivative or geometry enters.
"""

from __future__ import annotations

import numpy as np
import pytest

from hybridge.core.quadrature import ReferenceElementData, _evaluate_basis
from simplex_hdg_helpers import (
    block_diagonal,
    random_interior_points,
    random_polynomial,
    reaction_blocks,
    schur_complement,
    trace_coefficients,
    volume_load,
)

VOLUME_BASES = (
    "dub_orth",
    "orthonormal",
    "bernstein",
    "hier_C0",
    "lagrange_equispaced",
    "lagrange_warp_blend",
    "lagrange_recursive",
    "lagrange_fekete",
    "lagrange_fekete_free",
)
TRACE_BASES = ("lagrange_warp_blend", "orthonormal")
ORDERS = {2: (0, 1, 2, 3, 5, 8, 10), 3: (0, 1, 2, 3, 4, 6, 8)}
ALPHA = 1.7
TAU = 2.3


def _reference(dim: int, order: int, basis: str) -> ReferenceElementData:
    """Reference data of the triangle (``dim = 2``) or tetrahedron (``dim = 3``)."""
    if dim == 2:
        return ReferenceElementData.triangle(order, basis_type=basis)
    return ReferenceElementData.tetrahedron(order, basis_type=basis)


def _cases(max_order_3d: int | None = None):
    """All (dim, order, volume basis, trace basis) combinations of this stage."""
    for dim, orders in ORDERS.items():
        for order in orders:
            if dim == 3 and max_order_3d is not None and order > max_order_3d:
                continue
            for basis in VOLUME_BASES:
                for trace in TRACE_BASES:
                    yield pytest.param(dim, order, basis, trace, id=f"d{dim}-p{order}-{basis}-{trace}")


@pytest.mark.parametrize("dim, order, basis, trace", list(_cases()))
def test_local_solve_reproduces_polynomials(dim, order, basis, trace):
    """With f = alpha u* and uhat = u* on the faces, the local solve returns u* for u* of degree p."""
    reference = _reference(dim, order, basis)
    exact = random_polynomial(dim, order, seed=11 * dim + order)
    a, b, _, _ = reaction_blocks(reference, trace, alpha=ALPHA, tau=TAU)
    load = volume_load(reference, lambda x: ALPHA * exact(x))
    uhat = trace_coefficients(reference, trace, exact)
    coefficients = np.linalg.solve(a, load + b @ uhat)
    points = random_interior_points(dim, 50, seed=3)
    values = np.asarray(_evaluate_basis(reference.basis_type, order, points), dtype=np.float64) @ coefficients
    scale = max(1.0, np.abs(exact(points)).max())
    tolerance = 1e-14 * np.linalg.cond(a) * scale + 1e-12 * scale
    assert np.abs(values - exact(points)).max() <= tolerance


@pytest.mark.parametrize("dim, order, basis, trace", list(_cases()))
def test_local_and_condensed_matrices_are_symmetric_positive_definite(dim, order, basis, trace):
    """A and S = D - C A^{-1} B are symmetric positive definite for alpha, tau > 0."""
    reference = _reference(dim, order, basis)
    a, b, c, d = reaction_blocks(reference, trace, alpha=ALPHA, tau=TAU)
    s = schur_complement(a, b, c, d)
    for matrix in (a, s):
        scale = np.abs(matrix).max()
        assert np.abs(matrix - matrix.T).max() <= 1e-12 * scale * max(1.0, np.linalg.cond(a) * 1e-3)
        assert np.linalg.eigvalsh(0.5 * (matrix + matrix.T)).min() > 0.0
    assert np.allclose(c, b.T, rtol=0, atol=1e-13 * np.abs(b).max())


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("trace", TRACE_BASES)
def test_condensed_matrix_does_not_depend_on_the_volume_basis(dim, trace):
    """S is the same for every volume basis: a change of volume basis cancels in C A^{-1} B."""
    for order in ORDERS[dim]:
        matrices = {}
        for basis in VOLUME_BASES:
            reference = _reference(dim, order, basis)
            a, b, c, d = reaction_blocks(reference, trace, alpha=ALPHA, tau=TAU)
            matrices[basis] = (schur_complement(a, b, c, d), np.linalg.cond(a))
        base = matrices["orthonormal"][0]
        for basis, (matrix, condition) in matrices.items():
            error = np.abs(matrix - base).max() / np.abs(base).max()
            assert error <= 1e-14 * condition + 1e-12, (order, basis, error, condition)


@pytest.mark.parametrize("dim", (2, 3))
def test_change_of_trace_basis_is_a_congruence(dim):
    """For trace bases mu' = T mu on every face, S' = T S T^T with T block diagonal over the faces."""
    for order in ORDERS[dim][1:5]:
        reference = _reference(dim, order, "orthonormal")
        points, _ = reference._generic_face_rule
        from hybridge.core.quadrature import _evaluate_simplex_basis

        mu = np.asarray(_evaluate_simplex_basis("orthonormal", dim - 1, order, points, gradients=False), dtype=np.float64)
        nodal = np.asarray(_evaluate_simplex_basis("lagrange_warp_blend", dim - 1, order, points, gradients=False), dtype=np.float64)
        change = np.linalg.lstsq(mu, nodal, rcond=None)[0].T  # nodal_a = sum_b change[a, b] mu_b
        s_modal = schur_complement(*reaction_blocks(reference, "orthonormal", alpha=ALPHA, tau=TAU))
        s_nodal = schur_complement(*reaction_blocks(reference, "lagrange_warp_blend", alpha=ALPHA, tau=TAU))
        t = block_diagonal(change, reference.num_faces)
        assert np.abs(s_nodal - t @ s_modal @ t.T).max() <= 1e-11 * np.abs(s_nodal).max()


@pytest.mark.parametrize("dim", (2, 3))
def test_zero_stabilization_gives_the_l2_projection(dim):
    """At tau = 0 the local solve is the L2 projection of f / alpha; for small tau the gap is O(tau)."""
    order = 3
    reference = _reference(dim, order, "orthonormal")

    def source(x):
        """A smooth non-polynomial source."""
        return np.exp(0.3 * x.sum(axis=1)) * np.cos(x[:, 0])

    load = volume_load(reference, source)
    projection = np.linalg.solve(ALPHA * reference.exact_mass.astype(np.float64), load)
    gaps = []
    for tau in (0.0, 1e-3, 1e-4):
        a, b, _, _ = reaction_blocks(reference, "orthonormal", alpha=ALPHA, tau=tau)
        uhat = trace_coefficients(reference, "orthonormal", lambda x: np.zeros(len(x)))
        coefficients = np.linalg.solve(a, load + b @ uhat)
        gaps.append(np.abs(coefficients - projection).max())
    assert gaps[0] <= 1e-14
    assert gaps[1] > 0.0
    assert 8.0 < gaps[1] / gaps[2] < 12.0


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("trace", TRACE_BASES)
def test_face_orientation_changes_the_condensed_matrix_by_the_orientation_matrices(dim, trace):
    """With every face seen through sigma_o, S_o = R S_id R^T (R = blockdiag of orientation_matrices[o])."""
    for order in (1, 2, 3, 4):
        reference = _reference(dim, order, "orthonormal")
        tables = reference.trace_tables(trace)
        base = schur_complement(*reaction_blocks(reference, trace, alpha=ALPHA, tau=TAU))
        exact = random_polynomial(dim, order, seed=5)
        for o in range(1, reference.num_face_orientations):
            orientations = [o] * reference.num_faces
            a, b, c, d = reaction_blocks(reference, trace, alpha=ALPHA, tau=TAU, orientations=orientations)
            rotation = block_diagonal(tables.orientation_matrices[o].astype(np.float64), reference.num_faces)
            s = schur_complement(a, b, c, d)
            assert np.abs(s - rotation @ base @ rotation.T).max() <= 1e-11 * np.abs(base).max()
            load = volume_load(reference, lambda x: ALPHA * exact(x))
            uhat = trace_coefficients(reference, trace, exact, orientations=orientations)
            coefficients = np.linalg.solve(a, load + b @ uhat)
            points = random_interior_points(dim, 20, seed=9)
            values = np.asarray(_evaluate_basis("orthonormal", order, points), dtype=np.float64) @ coefficients
            assert np.abs(values - exact(points)).max() <= 1e-10 * max(1.0, np.abs(exact(points)).max())


@pytest.mark.parametrize("order", (0, 1, 2, 3, 5))
@pytest.mark.parametrize("basis", ("bernstein", "hier_C0", "dub_orth"))
@pytest.mark.parametrize("kind, trace", (("legacy-lagrange", "lagrange_warp_blend"), ("legendre-modal", "dub_orth")))
def test_triangle_blocks_equal_the_present_two_dimensional_assembly(order, basis, kind, trace):
    """On a one-element mesh of the reference triangle, the blocks equal those of hdg/matrices.py at zero velocity.

    The mesh element is the reference triangle itself, so the Jacobian is 1 and
    ``jacs_el_fc`` equals the face-measure ratios. With beta = 0 the advection
    weight gamma = tau - beta.n equals tau, and the transport blocks of
    ``hybridge.hdg.matrices`` are exactly the blocks of ``alpha u = f``.
    """
    from hybridge.core.mesh import DGMesh
    from hybridge.core.space import DGSpace
    from hybridge.hdg.matrices import (
        advection_trace_lift_from_stabilization,
        boundary_mass_from_trace_stabilization,
        element_boundary_mats_from_trace_weight,
    )

    mesh = DGMesh.from_arrays(np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]]), np.array([[0, 1, 2]]))
    space = DGSpace(mesh, order, basis_type=basis)
    trace_space = space.trace_space(kind)
    assert np.allclose(mesh.jacs_el_fc[0], ReferenceElementData.triangle(order, basis_type=basis).face_measure_ratios)
    tau_face = np.full((1, 3, trace_space.quads.size), TAU)
    boundary_mass = boundary_mass_from_trace_stabilization(space, tau_face, trace_space=trace_space)[0]
    element_boundary = element_boundary_mats_from_trace_weight(space, tau_face, trace_space=trace_space)[0]
    lift = advection_trace_lift_from_stabilization(space, tau_face, trace_space=trace_space)[0]

    reference = ReferenceElementData.triangle(order, basis_type=basis)
    a, b, c, _ = reaction_blocks(reference, trace, alpha=ALPHA, tau=TAU)
    a_present = ALPHA * mesh.aff_jacs[0] * space.quad_data.MKrf + boundary_mass
    scale = np.abs(a).max()
    assert np.abs(a - a_present).max() <= 1e-12 * scale
    assert np.abs(b - element_boundary).max() <= 1e-12 * np.abs(b).max()
    assert np.abs(c - lift.reshape(c.shape)).max() <= 1e-12 * np.abs(c).max()
