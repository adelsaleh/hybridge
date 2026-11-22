"""Stage C, phase 1: HDG for ``beta . grad u + alpha u = f`` on the reference triangle and tetrahedron.

The derivative tables ``D^k_ij = int_K^ phi_i d_k phi_j`` of ``ReferenceElementData``
are checked first (integration by parts, exact differentiation of polynomials, change
of basis) for every volume basis and ``p <= 8``. The local HDG blocks of the transport
form of hybridge, built in ``simplex_hdg_transport`` from the reference tables only,
are then checked on the reference element (``J = I``): polynomial reproduction,
invertibility of the local matrix, independence of the condensed matrix from the
volume basis, change of trace basis, face orientations, and, on the triangle, equality
with the blocks of the NumPy transport assembly ``assemble_transport_numpy``.

The velocity is constant and has a nonzero normal component on every face, so the
upwind ``tau = |beta . n|`` is positive on every face.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from hybridge.core.quadrature import ReferenceElementData, _evaluate_basis, _evaluate_simplex_basis
from simplex_hdg_helpers import (
    block_diagonal,
    random_interior_points,
    random_polynomial,
    schur_complement,
    trace_coefficients,
    volume_load,
)
from simplex_hdg_transport import (
    advection_reaction_blocks,
    element_geometry,
    local_solve,
    polynomial_gradient,
    transport_face_weights,
    volume_loads,
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
TABLE_ORDERS = tuple(range(9))
HDG_ORDERS = {2: (0, 1, 2, 3, 5, 8), 3: (0, 1, 2, 3, 4, 6)}
ALPHA = 1.7
# beta . n on the faces of the reference triangle: (0.4, 0.3/sqrt 2, -0.7);
# of the reference tetrahedron: (0.7/sqrt 3, -0.6, 0.35, -0.45). Positive values are outflow faces.
BETA = {2: np.array([0.7, -0.4]), 3: np.array([0.6, -0.35, 0.45])}
# Upwind tau = |beta.n| (hybridge's default), tau = 2 |beta.n|, and an absolute tau.
STABILIZATIONS = (None, "lax-friedrichs", 2.3)


@lru_cache(maxsize=None)
def _reference(dim: int, order: int, basis: str) -> ReferenceElementData:
    """Reference data of the triangle (``dim = 2``) or tetrahedron (``dim = 3``), shared by the tests of this file."""
    if dim == 2:
        return ReferenceElementData.triangle(order, basis_type=basis)
    return ReferenceElementData.tetrahedron(order, basis_type=basis)


def _blocks(reference, trace, *, beta=None, stabilization=None, orientations=None):
    """``(A, B, C, D)`` of one reference element, without the element axis."""
    beta = BETA[reference.dim] if beta is None else beta
    blocks = advection_reaction_blocks(
        reference, trace, beta=beta, alpha=ALPHA, stabilization=stabilization, orientations=orientations
    )
    return tuple(block[0] for block in blocks)


def _change_of_basis(dim: int, order: int, basis_from: str, basis_to: str) -> np.ndarray:
    """Return ``T`` with ``phi_to(x) = phi_from(x) T`` (row vectors of basis values), from a least-squares fit at random points."""
    count = len(_evaluate_basis("orthonormal", order, np.zeros((1, dim)))[0])
    points = random_interior_points(dim, 3 * count + 20, seed=6)
    phi_from = np.asarray(_evaluate_basis(basis_from, order, points), dtype=np.float64)
    phi_to = np.asarray(_evaluate_basis(basis_to, order, points), dtype=np.float64)
    change = np.linalg.lstsq(phi_from, phi_to, rcond=None)[0]
    assert np.abs(phi_from @ change - phi_to).max() <= 1e-11 * np.abs(phi_to).max()
    return change


# ---------------------------------------------------------------------------
# Derivative tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("basis", VOLUME_BASES)
def test_integration_by_parts_identity_of_the_derivative_tables(dim, basis):
    """D^k + (D^k)^T = sum_f n_{f,k} r_f E^f for p <= 8 (the divergence theorem applied to phi_i phi_j)."""
    for order in TABLE_ORDERS:
        reference = _reference(dim, order, basis)
        derivative = reference.derivative_matrices.astype(np.float64)
        boundary = np.einsum(
            "fk,f,fij->kij",
            reference.reference_normals,
            reference.face_measure_ratios,
            reference.face_mass.astype(np.float64),
        )
        error = np.abs(derivative + derivative.transpose(0, 2, 1) - boundary).max()
        assert error <= 1e-12 * max(1.0, np.abs(boundary).max()), (order, error)


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("basis", VOLUME_BASES)
def test_derivative_tables_differentiate_polynomials_exactly(dim, basis):
    """For u = sum_j c_j phi_j of degree p <= 8, M^{-1} D^k c are the coefficients of d_k u.

    ``c`` is the L2 projection of a random polynomial ``u`` of degree ``p``, which is
    ``u`` itself; ``d_k u`` has degree ``p - 1``, so its L2 projection is also exact,
    and it is compared with the analytic derivative at random points.
    """
    points = random_interior_points(dim, 60, seed=5)
    for order in TABLE_ORDERS:
        reference = _reference(dim, order, basis)
        mass = reference.exact_mass.astype(np.float64)
        derivative = reference.derivative_matrices.astype(np.float64)
        exact = random_polynomial(dim, order, seed=4)
        gradient = polynomial_gradient(dim, order, seed=4)(points)
        coefficients = np.linalg.solve(mass, volume_load(reference, exact))
        derivative_coefficients = np.linalg.solve(mass, (derivative @ coefficients).T)  # (n_u, dim)
        values = np.asarray(_evaluate_basis(reference.basis_type, order, points), dtype=np.float64) @ derivative_coefficients
        scale = max(1.0, np.abs(gradient).max())
        error = np.abs(values - gradient).max()
        assert error <= 1e-13 * np.linalg.cond(mass) * scale, (order, error)


@pytest.mark.parametrize("dim", (2, 3))
def test_change_of_volume_basis_transforms_the_derivative_tables_by_congruence(dim):
    """With phi_b = phi_a T, D^k_b = T^T D^k_a T for every basis b and p <= 8 (a = orthonormal)."""
    for order in TABLE_ORDERS:
        base = _reference(dim, order, "orthonormal").derivative_matrices.astype(np.float64)
        for basis in VOLUME_BASES:
            change = _change_of_basis(dim, order, "orthonormal", basis)
            derivative = _reference(dim, order, basis).derivative_matrices.astype(np.float64)
            error = np.abs(change.T @ base @ change - derivative).max()
            assert error <= 1e-12 * max(1.0, np.abs(derivative).max()), (order, basis, error)


# ---------------------------------------------------------------------------
# HDG on the reference element
# ---------------------------------------------------------------------------


def _hdg_cases():
    """All (dim, order, volume basis, trace basis) combinations of the local HDG checks."""
    for dim, orders in HDG_ORDERS.items():
        for order in orders:
            for basis in VOLUME_BASES:
                for trace in TRACE_BASES:
                    yield pytest.param(dim, order, basis, trace, id=f"d{dim}-p{order}-{basis}-{trace}")


@pytest.mark.parametrize("dim, order, basis, trace", list(_hdg_cases()))
def test_local_solve_reproduces_polynomials_and_the_exact_flux(dim, order, basis, trace):
    """With f = beta . grad u* + alpha u* and uhat = u* on the faces, the local solve returns u* (degree p).

    The local equation then holds for u* by integration by parts, for every
    stabilization. The transmission residual ``D uhat - C u`` is the exact
    flux ``-<beta.n u*, mu>_F = -(beta.n_f) r_f G uhat_f`` on every face.
    """
    reference = _reference(dim, order, basis)
    beta = BETA[dim]
    exact = random_polynomial(dim, order, seed=11 * dim + order)
    gradient = polynomial_gradient(dim, order, seed=11 * dim + order)
    loads = volume_loads(reference, lambda x: gradient(x) @ beta + ALPHA * exact(x))
    uhat = trace_coefficients(reference, trace, exact)
    points = random_interior_points(dim, 50, seed=3)
    basis_values = np.asarray(_evaluate_basis(reference.basis_type, order, points), dtype=np.float64)
    scale = max(1.0, np.abs(exact(points)).max())
    normal_flux = reference.reference_normals @ beta
    flux_mass = block_diagonal(reference.trace_tables(trace).trace_mass.astype(np.float64), reference.num_faces)
    flux_weights = np.repeat(normal_flux * reference.face_measure_ratios, reference.face_dof)
    for stabilization in STABILIZATIONS:
        a, b, c, d = _blocks(reference, trace, stabilization=stabilization)
        coefficients = local_solve(a[None], b[None], loads, uhat[None])[0]
        tolerance = 1e-14 * np.linalg.cond(a) * scale + 1e-12 * scale
        assert np.abs(basis_values @ coefficients - exact(points)).max() <= tolerance, stabilization
        flux = d @ uhat - c @ coefficients
        expected = -(flux_weights[:, None] * flux_mass) @ uhat
        assert np.abs(flux - expected).max() <= tolerance * max(1.0, np.abs(c).max()), stabilization


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("basis", VOLUME_BASES)
def test_local_matrix_is_invertible_with_a_positive_symmetric_part(dim, basis):
    """The symmetric part of A is alpha M + sum_f (tau_f - beta.n_f / 2) r_f E^f, so u^T A u >= alpha u^T M u.

    Integration by parts gives ``-(u, beta.grad u) = -<beta.n u, u>_dK / 2``; the face
    weights ``tau_f - beta.n_f / 2`` are at least ``|beta.n_f| / 2`` for every
    stabilization used here. Hence the smallest eigenvalue of ``sym(A)`` relative to
    ``M`` is at least ``alpha`` (equal to it when a polynomial of degree p vanishes on
    the boundary, ``p >= d + 1``), and ``sigma_min(A) >= alpha lambda_min(M)``.
    """
    beta = BETA[dim]
    for order in HDG_ORDERS[dim]:
        reference = _reference(dim, order, basis)
        mass = reference.exact_mass.astype(np.float64)
        face_mass = reference.face_mass.astype(np.float64)
        cholesky = np.linalg.cholesky(mass)
        mass_condition = np.linalg.cond(mass)
        normal_flux = (reference.reference_normals @ beta)[None]
        for stabilization in STABILIZATIONS:
            a = _blocks(reference, "orthonormal", stabilization=stabilization)[0]
            tau, _ = transport_face_weights(normal_flux, stabilization)
            weights = (tau[0] - 0.5 * normal_flux[0]) * reference.face_measure_ratios
            assert weights.min() >= 0.5 * np.abs(normal_flux).min() * reference.face_measure_ratios.min()
            symmetric = 0.5 * (a + a.T)
            expected = ALPHA * mass + np.einsum("f,fij->ij", weights, face_mass)
            assert np.abs(symmetric - expected).max() <= 1e-12 * np.abs(expected).max(), (order, stabilization)
            whitened = np.linalg.solve(cholesky, np.linalg.solve(cholesky, symmetric).T)
            smallest = np.linalg.eigvalsh(0.5 * (whitened + whitened.T)).min()
            assert smallest >= ALPHA * (1.0 - 1e-12 * mass_condition), (order, stabilization, smallest)
            sigma_min = np.linalg.svd(a, compute_uv=False).min()
            assert sigma_min >= ALPHA * np.linalg.eigvalsh(mass).min() * (1.0 - 1e-12 * mass_condition)


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("trace", TRACE_BASES)
def test_condensed_matrix_does_not_depend_on_the_volume_basis(dim, trace):
    """S = D - C A^{-1} B is the same for every volume basis: A' = T^T A T, B' = T^T B and C' = C T cancel."""
    for order in HDG_ORDERS[dim]:
        for stabilization in STABILIZATIONS:
            matrices = {}
            for basis in VOLUME_BASES:
                a, b, c, d = _blocks(_reference(dim, order, basis), trace, stabilization=stabilization)
                matrices[basis] = (schur_complement(a, b, c, d), np.linalg.cond(a))
            base = matrices["orthonormal"][0]
            for basis, (matrix, condition) in matrices.items():
                error = np.abs(matrix - base).max() / np.abs(base).max()
                assert error <= 1e-14 * condition + 1e-12, (order, stabilization, basis, error, condition)


@pytest.mark.parametrize("dim", (2, 3))
def test_change_of_trace_basis_is_a_congruence(dim):
    """For trace bases mu' = T mu on every face, S' = T S T^T with T block diagonal over the faces (S is not symmetric)."""
    for order in HDG_ORDERS[dim][1:5]:
        reference = _reference(dim, order, "orthonormal")
        points, _ = reference._generic_face_rule
        mu = np.asarray(_evaluate_simplex_basis("orthonormal", dim - 1, order, points, gradients=False), dtype=np.float64)
        nodal = np.asarray(
            _evaluate_simplex_basis("lagrange_warp_blend", dim - 1, order, points, gradients=False), dtype=np.float64
        )
        change = block_diagonal(np.linalg.lstsq(mu, nodal, rcond=None)[0].T, reference.num_faces)
        for stabilization in STABILIZATIONS:
            s_modal = schur_complement(*_blocks(reference, "orthonormal", stabilization=stabilization))
            s_nodal = schur_complement(*_blocks(reference, "lagrange_warp_blend", stabilization=stabilization))
            assert np.abs(s_nodal - change @ s_modal @ change.T).max() <= 1e-11 * np.abs(s_nodal).max()


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("trace", TRACE_BASES)
def test_face_orientation_changes_the_condensed_matrix_by_the_orientation_matrices(dim, trace):
    """With face f seen through sigma_{o_f}, S_o = R S_id R^T, R = blockdiag_f(orientation_matrices[o_f]).

    ``orientation_matrices[o]`` is the matrix ``R_o`` with
    ``mu_a(sigma_o s) = sum_b R_o[a, b] mu_b(s)``, so ``B_o = B_id R^T``,
    ``C_o = R C_id`` and ``D_o = R D_id R^T``. Checked for every orientation on
    all faces at once (the reversal on the triangle, the 3-cycles ``(1, 2, 0)``
    and ``(2, 0, 1)`` and the transpositions on the tetrahedron) and for one
    assignment with a different orientation on each face; the local solve with
    oriented trace data still reproduces a polynomial of degree p.
    """
    count = 6 if dim == 3 else 2
    assignments = [[o] * (dim + 1) for o in range(1, count)] + [[(f + 1) % count for f in range(dim + 1)]]
    for order in (1, 2, 3, 4):
        reference = _reference(dim, order, "orthonormal")
        rotations = reference.trace_tables(trace).orientation_matrices.astype(np.float64)
        exact = random_polynomial(dim, order, seed=5)
        gradient = polynomial_gradient(dim, order, seed=5)
        loads = volume_loads(reference, lambda x: gradient(x) @ BETA[dim] + ALPHA * exact(x))
        points = random_interior_points(dim, 20, seed=9)
        basis_values = np.asarray(_evaluate_basis("orthonormal", order, points), dtype=np.float64)
        for stabilization in STABILIZATIONS:
            base = schur_complement(*_blocks(reference, trace, stabilization=stabilization))
            for orientations in assignments:
                a, b, c, d = _blocks(reference, trace, stabilization=stabilization, orientations=orientations)
                n_t = reference.face_dof
                rotation = np.zeros((len(orientations) * n_t,) * 2)
                for f, o in enumerate(orientations):
                    rotation[f * n_t:(f + 1) * n_t, f * n_t:(f + 1) * n_t] = rotations[o]
                s = schur_complement(a, b, c, d)
                assert np.abs(s - rotation @ base @ rotation.T).max() <= 1e-11 * np.abs(base).max(), orientations
                uhat = trace_coefficients(reference, trace, exact, orientations=orientations)
                coefficients = local_solve(a[None], b[None], loads, uhat[None])[0]
                values = basis_values @ coefficients
                assert np.abs(values - exact(points)).max() <= 1e-10 * max(1.0, np.abs(exact(points)).max())


@pytest.mark.parametrize("dim", (2, 3))
def test_blocks_of_stacked_elements_equal_the_blocks_of_each_element(dim):
    """One call on E elements returns, for each element, the blocks of a call on that element alone.

    The geometric inputs are arbitrary arrays (not those of a real element); this
    checks only the element-batched layout that phase 2 will use, including
    broadcasting of inputs given for one element.
    """
    reference = _reference(dim, 2, "bernstein")
    rng = np.random.default_rng(2)
    count = 4
    inputs = {
        "beta": rng.standard_normal((count, dim)),
        "alpha": rng.uniform(0.5, 2.0, count),
        "inverse_jacobians": rng.standard_normal((count, dim, dim)),
        "determinants": rng.uniform(0.1, 1.0, count),
        "face_ratios": rng.uniform(0.5, 2.0, (count, dim + 1)),
        "normals": rng.standard_normal((count, dim + 1, dim)),
        "orientations": rng.integers(0, reference.num_face_orientations, (count, dim + 1)),
    }
    for trace in TRACE_BASES:
        for stabilization in STABILIZATIONS:
            stacked = advection_reaction_blocks(reference, trace, stabilization=stabilization, **inputs)
            for e in range(count):
                single = advection_reaction_blocks(
                    reference, trace, stabilization=stabilization, **{k: v[e] for k, v in inputs.items()}
                )
                for block_stacked, block_single in zip(stacked, single):
                    assert block_single.shape[0] == 1
                    assert np.array_equal(block_stacked[e], block_single[0])
    broadcast = element_geometry(reference, beta=inputs["beta"], alpha=1.0)
    assert broadcast["inverse_jacobians"].shape == (count, dim, dim)
    assert np.array_equal(broadcast["normals"][3], reference.reference_normals)
    jacobians = rng.standard_normal((count, dim, dim))
    offsets = rng.standard_normal((count, dim))
    exact = random_polynomial(dim, 2, seed=1)
    loads = volume_loads(reference, exact, jacobians=jacobians, offsets=offsets)
    for e in range(count):
        mapped = volume_load(reference, lambda xi: exact(offsets[e] + xi @ jacobians[e].T))
        assert np.allclose(loads[e], abs(np.linalg.det(jacobians[e])) * mapped, rtol=1e-13, atol=1e-13)


# ---------------------------------------------------------------------------
# Parity with the two-dimensional transport assembly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", (0, 1, 2, 3, 5))
@pytest.mark.parametrize("basis", ("bernstein", "hier_C0", "dub_orth"))
@pytest.mark.parametrize(
    "kind, trace",
    (("legacy-lagrange", "lagrange_warp_blend"), ("legendre-modal", "dub_orth"), ("bernstein", "bernstein")),
)
@pytest.mark.parametrize("velocity", ("callables", "dg-field"))
def test_triangle_blocks_equal_the_numpy_transport_assembly(order, basis, kind, trace, velocity):
    """On the reference triangle, A, B, C, D and F equal those of ``assemble_transport_numpy`` for beta != 0.

    The mesh has two triangles: element 0 is the reference triangle (Jacobian 1,
    ``jacs_el_fc`` equal to the face-measure ratios, every face orientation
    positive) and element 1 is its mirror image across face 1, so that face 1 of
    element 0 is an interior face. Only element 0 is compared; element 1 needs
    the geometry of phase 2. From the NumPy transport driver the test
    takes ``local_solver`` (``A^{-1}``), ``element_boundary_mats`` (``B``),
    ``trace_lift`` (``C``) and the source moments (``F``) of element 0, and the
    ``gamma``-weighted trace mass of its face 1 (block ``D`` of that face) from
    ``advection_interior_trace_mass_blocks_from_weight``, which the driver
    calls. The velocity is given either as two constant callables (upwind
    ``tau = |beta.n|``) or as a constant DG field (``"conflict-averaged-upwind"``
    by default, which is the same here); both signs of beta are used, so face 1
    is an outflow face once and an inflow face once.
    """
    from hybridge.core.mesh import DGMesh
    from hybridge.core.space import DGSpace
    from hybridge.hdg import condensation
    from hybridge.hdg.coefficients import _prepare_beta_data
    from hybridge.hdg.matrices import advection_interior_trace_mass_blocks_from_weight
    from hybridge.hdg.stabilization import advection_trace_weights_from_normal_flux, resolve_transport_stabilization
    from hybridge.solvers.advection_host import assemble_transport_numpy
    from hybridge.solvers.advection_stages import TransportAssemblyInputs

    vertices = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
    mesh = DGMesh.from_arrays(vertices, np.array([[0, 1, 2], [1, 3, 2]]))
    space = DGSpace(mesh, order, basis_type=basis)
    trace_space = space.trace_space(kind)
    reference = ReferenceElementData.triangle(order, basis_type=basis)
    assert mesh.aff_jacs[0] == 1.0 and np.all(mesh.orientations[0])
    assert np.allclose(mesh.jacs_el_fc[0], reference.face_measure_ratios, rtol=0, atol=1e-15)
    interior = list(zip(mesh.interior_elements.tolist(), mesh.interior_faces.tolist())).index((0, 1))
    n_t = trace_space.edg_dof

    for sign in (1.0, -1.0):
        beta = sign * BETA[2]
        exact = random_polynomial(2, order, seed=1)
        gradient = polynomial_gradient(2, order, seed=1)

        def source(x, y):
            """f = beta . grad u* + alpha u* at physical points given as two coordinate arrays."""
            points = np.stack((np.ravel(x), np.ravel(y)), axis=1)
            return (gradient(points) @ beta + ALPHA * exact(points)).reshape(np.shape(x))

        if velocity == "callables":
            beta_input = (lambda x, y: np.full_like(x, beta[0]), lambda x, y: np.full_like(x, beta[1]))
        else:
            beta_input = (space.constant(beta[0]), space.constant(beta[1]))
        for stabilization in STABILIZATIONS:
            resolved = resolve_transport_stabilization(stabilization, beta_input)
            beta_h, beta_dot_normal, beta_callables = _prepare_beta_data(beta_input, space, trace_space=trace_space)
            inputs = TransportAssemblyInputs(
                space=space, trace_space=trace_space, source_data=condensation.source_moments(source, space),
                beta_h=beta_h, beta_callables=beta_callables, beta_dot_normal=beta_dot_normal, reaction_h=ALPHA,
                boundary_condition=condensation.normalize_boundary_condition(0.0), boundary_mode="penalty",
                boundary_penalty=1e20, advection_stabilization=resolved, verbosity=0, detail_timings={},
            )
            present = assemble_transport_numpy(inputs)
            _, gamma_face = advection_trace_weights_from_normal_flux(space, beta_dot_normal, resolved, trace_space=trace_space)
            interior_mass = advection_interior_trace_mass_blocks_from_weight(space, gamma_face, trace_space=trace_space)

            a, b, c, d = _blocks(reference, trace, beta=beta, stabilization=stabilization)
            loads = volume_loads(reference, lambda x: gradient(x) @ beta + ALPHA * exact(x))[0]
            label = (sign, stabilization)
            # The driver keeps only A^{-1}; its inverse differs from A by about cond(A) times the rounding unit.
            a_present = np.linalg.inv(present.local_solver[0])
            assert np.abs(a_present - a).max() <= (1e-15 * np.linalg.cond(a) + 1e-12) * np.abs(a).max(), label
            assert np.abs(present.element_boundary_mats[0] - b).max() <= 1e-12 * np.abs(b).max(), label
            assert np.abs(present.trace_lift[0].reshape(c.shape) - c).max() <= 1e-12 * np.abs(c).max(), label
            assert np.abs(inputs.source_data[0] - loads).max() <= 1e-12 * np.abs(loads).max(), label
            face_block = d[n_t:2 * n_t, n_t:2 * n_t]
            assert np.abs(interior_mass[interior] - face_block).max() <= 1e-12 * np.abs(d).max(), label
