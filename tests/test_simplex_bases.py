"""Tests of the modal bases on the interval, triangle and tetrahedron (:mod:`hybridge.core.simplex.polynomials`).

The bases are ``dub_orth`` (orthogonal), ``orthonormal``, ``bernstein`` and
``hier_C0`` (Karniadakis-Sherwin modified basis). The tests check the mass
matrices on exact quadrature rules, the span, the vanishing of the modified modes
on the faces, the gradients against Richardson-extrapolated central differences
(at the vertices and on the edge where the collapsed coordinates are undefined
too), the scaled Jacobi recurrence, the modified basis against the listing of
Karniadakis and Sherwin, 2nd ed., Appendix D (triangle p. 604, tetrahedron
pp. 609-610, with its vertex C corrected), and the corrected vertex gradients of
the triangle kernels of :mod:`hybridge.core.basis`.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pytest
from scipy.special import eval_jacobi

from hybridge.core import basis as legacy
from hybridge.core.simplex import polynomials as P
from hybridge.core.simplex.reference import gauss_jacobi_collapsed_rule, reference_simplex

NAMES = P.MODAL_BASIS_NAMES
DIMS = (1, 2, 3)


def _barycentric(points: np.ndarray) -> np.ndarray:
    """Barycentric coordinates ``(n, d + 1)`` of reference points ``(n, d)``: lambda_k = (1 + xi_k)/2, lambda_0 = 1 - sum."""
    tail = 0.5 * (1.0 + points)
    return np.column_stack((1.0 - tail.sum(axis=1), tail))


def _random_points(dim: int, count: int, seed: int) -> np.ndarray:
    """``count`` points drawn uniformly in barycentric coordinates strictly inside the reference ``dim``-simplex."""
    lam = np.random.default_rng(seed).dirichlet(np.ones(dim + 1), size=count)
    return reference_simplex(dim).from_barycentric(lam)


def _singular_edge_points(count: int) -> np.ndarray:
    """Points of the tetrahedron edge xi_2 + xi_3 = 0 from (-1,-1,1) to (-1,1,-1), its end points included."""
    s = np.linspace(0.0, 1.0, count)
    return np.column_stack((-np.ones(count), -1.0 + 2.0 * s, 1.0 - 2.0 * s))


def _special_points(dim: int) -> np.ndarray:
    """The vertices of the reference simplex and, for ``dim = 3``, points on the singular edge xi_2 + xi_3 = 0."""
    points = [np.asarray(reference_simplex(dim).vertices)]
    if dim == 3:
        points.append(_singular_edge_points(5))
    return np.vstack(points)


def _richardson_gradient(name: str, dim: int, order: int, points: np.ndarray, h: float) -> np.ndarray:
    """Gradient ``(n, modes, dim)`` from Richardson-extrapolated central differences (4 D(h/2) - D(h)) / 3.

    The differences are taken along the directions ``(1, -1)`` and ``(0, 1)`` in two
    dimensions, so that no stencil point lies on the line xi_2 = 1 where the
    collapsed coordinate of the triangle kernels of :mod:`hybridge.core.basis`
    divides by zero; along the coordinate axes otherwise. The directional
    derivatives are converted to the gradient by solving with the direction matrix.
    """
    directions = np.array([[1.0, -1.0], [0.0, 1.0]]) if dim == 2 else np.eye(dim)

    def central(step: float, direction: np.ndarray) -> np.ndarray:
        """Central difference (f(x + step d) - f(x - step d)) / (2 step) of all modes at all points."""
        plus = P.modal_basis(name, dim, order, points + step * direction)
        minus = P.modal_basis(name, dim, order, points - step * direction)
        return (plus - minus) / (2.0 * step)

    directional = np.stack(
        [(4.0 * central(0.5 * h, d) - central(h, d)) / 3.0 for d in directions], axis=-1
    )
    return np.einsum("nmj,kj->nmk", directional, np.linalg.inv(directions))


def _ks_collapsed(points: np.ndarray) -> np.ndarray:
    """Collapsed coordinates eta_k = (lambda_k - s_{k-1}) / s_k of points ``(n, d)`` (defined where s_k != 0)."""
    lam = _barycentric(points)
    partial = np.cumsum(lam, axis=1)
    return (lam[:, 1:] - partial[:, :-1]) / partial[:, 1:]


def _book_tetrahedron_mode(row: np.ndarray, points: np.ndarray) -> np.ndarray:
    """One mode of the tetrahedron modified basis as printed in K&S, Appendix D.1.2.4 (pp. 609-610), at interior points.

    ``row`` is a row of ``polynomials._modified_table(3, p)``: entity vertices and
    indices (n_1, n_2, n_3), which are the book's p, q or r as listed in the
    docstring of ``_modified_table``. Vertex C is the corrected
    (1+eta_2)/2 (1-eta_3)/2; the printed (1-eta_1)/2 (1+eta_2)/2 (1-eta_3)/2 is
    tested in ``test_book_vertex_modes``. The book's faces "ABE", "ACE", "BCE" are
    ABD, ACD, BCD.
    """
    eta = _ks_collapsed(points)
    a1, a2, a3 = (0.5 * (1.0 - eta[:, k]) for k in range(3))
    b1, b2, b3 = (0.5 * (1.0 + eta[:, k]) for k in range(3))
    level, v0, v1, v2, v3, n1, n2, n3 = (int(x) for x in row)
    vertices = (v0, v1, v2, v3)[: level + 1]

    def jac(n: int, alpha: float, z: np.ndarray) -> np.ndarray:
        """Jacobi polynomial P_n^{(alpha, 1)}(z)."""
        return eval_jacobi(n, alpha, 1.0, z)

    if vertices == (0,):
        return a1 * a2 * a3
    if vertices == (1,):
        return b1 * a2 * a3
    if vertices == (2,):
        return b2 * a3
    if vertices == (3,):
        return b3
    if vertices == (0, 1):  # edge AB, p = n1
        return a1 * b1 * jac(n1 - 1, 1, eta[:, 0]) * a2 ** (n1 + 1) * a3 ** (n1 + 1)
    if vertices in ((0, 2), (1, 2)):  # edges AC, BC, q = n1
        first = a1 if vertices == (0, 2) else b1
        return first * a2 * b2 * jac(n1 - 1, 1, eta[:, 1]) * a3 ** (n1 + 1)
    if vertices in ((0, 3), (1, 3)):  # edges AD, BD, r = n1
        first = a1 if vertices == (0, 3) else b1
        return first * a2 * a3 * b3 * jac(n1 - 1, 1, eta[:, 2])
    if vertices == (2, 3):  # edge CD, r = n1
        return b2 * a3 * b3 * jac(n1 - 1, 1, eta[:, 2])
    if vertices == (0, 1, 2):  # face ABC, (p, q) = (n1, n2)
        return (a1 * b1 * jac(n1 - 1, 1, eta[:, 0]) * a2 ** (n1 + 1) * b2 * jac(n2 - 1, 2 * n1 + 1, eta[:, 1])
                * a3 ** (n1 + n2 + 1))
    if vertices == (0, 1, 3):  # face ABD (printed "ABE"), (p, r) = (n1, n2)
        return (a1 * b1 * jac(n1 - 1, 1, eta[:, 0]) * a2 ** (n1 + 1) * a3 ** (n1 + 1) * b3
                * jac(n2 - 1, 2 * n1 + 1, eta[:, 2]))
    if vertices in ((0, 2, 3), (1, 2, 3)):  # faces ACD, BCD (printed "ACE", "BCE"), (q, r) = (n1, n2)
        first = a1 if vertices == (0, 2, 3) else b1
        return (first * a2 * b2 * jac(n1 - 1, 1, eta[:, 1]) * a3 ** (n1 + 1) * b3
                * jac(n2 - 1, 2 * n1 + 1, eta[:, 2]))
    assert vertices == (0, 1, 2, 3)  # interior, (p, q, r) = (n1, n2, n3)
    return (a1 * b1 * jac(n1 - 1, 1, eta[:, 0]) * a2 ** (n1 + 1) * b2 * jac(n2 - 1, 2 * n1 + 1, eta[:, 1])
            * a3 ** (n1 + n2 + 1) * b3 * jac(n3 - 1, 2 * (n1 + n2) + 1, eta[:, 2]))


def _book_triangle_interior(p: int, q: int, alpha: float, points: np.ndarray) -> np.ndarray:
    """Triangle interior mode of K&S p. 604 with the second Jacobi weight ``(alpha, 1)``; the book has alpha = 2p + 1.

    (1-eta_1)/2 (1+eta_1)/2 P^{(1,1)}_{p-1}(eta_1) ((1-eta_2)/2)^{p+1} (1+eta_2)/2 P^{(alpha,1)}_{q-1}(eta_2).
    """
    eta = _ks_collapsed(points)
    return (
        0.25 * (1.0 - eta[:, 0]) * (1.0 + eta[:, 0]) * eval_jacobi(p - 1, 1.0, 1.0, eta[:, 0])
        * (0.5 * (1.0 - eta[:, 1])) ** (p + 1) * 0.5 * (1.0 + eta[:, 1]) * eval_jacobi(q - 1, alpha, 1.0, eta[:, 1])
    )



# ---------------------------------------------------------------------------
# Scaled Jacobi recurrence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha, beta", [(0.0, 0.0), (1.0, 1.0), (3.0, 0.0), (7.0, 1.0), (0.5, 2.5)])
def test_scaled_jacobi_equals_jacobi_at_t_one(alpha, beta):
    """At t = 1 the scaled polynomials and their x-derivatives equal ``basis._jacobi_p`` and ``_jacobi_derivative``."""
    order = 12
    x = np.linspace(-1.0, 1.0, 9)
    values, d_dx, _ = P.scaled_jacobi(order, alpha, beta, x, np.ones_like(x))
    expected = np.array([[legacy._jacobi_p(k, alpha, beta, xx) for k in range(order + 1)] for xx in x])
    expected_dx = np.array([[legacy._jacobi_derivative(k, alpha, beta, xx) for k in range(order + 1)] for xx in x])
    assert values.shape == (x.size, order + 1)
    np.testing.assert_allclose(values, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(d_dx, expected_dx, rtol=1e-12, atol=1e-10)


def test_scaled_jacobi_homogeneity_and_euler_identity():
    """Q_k(s x, s t) = s^k Q_k(x, t); x dQ/dx + t dQ/dt = k Q; dQ/dt equals differences; at t = 0, Q_k = c_k x^k.

    c_k = C(2k + a + b, k) / 2^k is the leading coefficient of P_k^{(a,b)}.
    """
    rng = np.random.default_rng(3)
    order, alpha, beta = 9, 5.0, 1.0
    x = rng.uniform(-1.0, 1.0, 20)
    t = rng.uniform(0.0, 1.0, 20)
    values, d_dx, d_dt = P.scaled_jacobi(order, alpha, beta, x, t)
    k = np.arange(order + 1)
    s = 0.37
    scaled, _, _ = P.scaled_jacobi(order, alpha, beta, s * x, s * t)
    np.testing.assert_allclose(scaled, s ** k * values, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(x[:, None] * d_dx + t[:, None] * d_dt, k * values, rtol=1e-11, atol=1e-12)

    def central(step: float) -> np.ndarray:
        """Central difference in t, (Q(x, t + step) - Q(x, t - step)) / (2 step)."""
        plus, _, _ = P.scaled_jacobi(order, alpha, beta, x, t + step)
        minus, _, _ = P.scaled_jacobi(order, alpha, beta, x, t - step)
        return (plus - minus) / (2.0 * step)

    richardson = (4.0 * central(5e-4) - central(1e-3)) / 3.0
    np.testing.assert_allclose(d_dt, richardson, rtol=0.0, atol=1e-9 * np.abs(d_dt).max())
    at_zero, _, _ = P.scaled_jacobi(order, alpha, beta, x, np.zeros_like(x))
    leading = np.array([comb(2 * n + int(alpha + beta), n) / 2.0 ** n for n in k])
    np.testing.assert_allclose(at_zero, leading * x[:, None] ** k, rtol=1e-12, atol=1e-14)
    grid_values, _, _ = P.scaled_jacobi(3, 0.0, 0.0, np.zeros((2, 3)), np.ones((2, 3)))
    assert grid_values.shape == (2, 3, 4)


# ---------------------------------------------------------------------------
# Dimensions, labels and arguments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", DIMS)
def test_dimensions_labels_and_shapes(dim):
    """Every basis has C(p + d, d) modes, one label per mode, float64 values (n, modes) and gradients (n, modes, d)."""
    points = _random_points(dim, 7, seed=dim)
    for name in NAMES:
        for order in range(7):
            count = P.modal_dimension(dim, order)
            assert count == comb(order + dim, dim)
            labels = P.modal_mode_labels(name, dim, order)
            assert len(labels) == count and len(set(labels)) == count
            values = P.modal_basis(name, dim, order, points)
            gradients = P.modal_gradients(name, dim, order, points)
            assert values.shape == (7, count) and values.dtype == np.float64
            assert gradients.shape == (7, count, dim) and gradients.dtype == np.float64


def test_hier_c0_entity_counts_in_three_dimensions():
    """The tetrahedron modified basis has 4 vertex, 6(p-1) edge, 4(p-1)(p-2)/2 face and (p-1)(p-2)(p-3)/6 interior modes."""
    for order in range(1, 9):
        kinds = [label[0] for label in P.modal_mode_labels("hier_C0", 3, order)]
        assert kinds.count("vertex") == 4
        assert kinds.count("edge") == 6 * (order - 1)
        assert kinds.count("face") == 4 * (order - 1) * (order - 2) // 2
        assert kinds.count("interior") == (order - 1) * (order - 2) * (order - 3) // 6
        assert kinds == sorted(kinds, key=("vertex", "edge", "face", "interior").index)


def test_invalid_arguments_raise():
    """Unknown names, dimensions other than 1, 2, 3, negative orders and wrong point shapes raise ``ValueError``."""
    points = np.zeros((2, 3))
    with pytest.raises(ValueError):
        P.modal_basis("lagrange", 3, 2, points)
    with pytest.raises(ValueError):
        P.modal_basis("dub_orth", 4, 2, np.zeros((2, 4)))
    with pytest.raises(ValueError):
        P.modal_basis("dub_orth", 3, -1, points)
    with pytest.raises(ValueError):
        P.modal_gradients("dub_orth", 3, 2, np.zeros((2, 2)))
    with pytest.raises(ValueError):
        P.scaled_jacobi(3, -1.0, 0.0, 0.0, 1.0)
    assert P.modal_basis("HIER_c0", 1, 2, np.zeros(4)).shape == (4, 3)


@pytest.mark.parametrize("name", NAMES)
def test_parallel_kernels_equal_serial_kernels(name):
    """``parallel=True`` gives the same values and gradients as the serial kernels."""
    for dim in DIMS:
        points = _random_points(dim, 40, seed=11)
        np.testing.assert_allclose(
            P.modal_basis(name, dim, 5, points, parallel=True),
            P.modal_basis(name, dim, 5, points),
            rtol=1e-14,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            P.modal_gradients(name, dim, 5, points, parallel=True),
            P.modal_gradients(name, dim, 5, points),
            rtol=1e-14,
            atol=1e-14,
        )
    values, d_dx, d_dt = P.scaled_jacobi(6, 2.0, 1.0, np.linspace(-1, 1, 5), np.ones(5), parallel=True)
    np.testing.assert_allclose(values, P.scaled_jacobi(6, 2.0, 1.0, np.linspace(-1, 1, 5), np.ones(5))[0])


# ---------------------------------------------------------------------------
# Mass matrices, legacy parity and span
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", DIMS)
def test_orthonormal_mass_matrix_is_identity(dim):
    """The orthonormal basis has mass matrix I to 1e-12 for p = 0..10 on a rule exact for degree 2p."""
    for order in range(11):
        points, weights = gauss_jacobi_collapsed_rule(dim, 2 * order)
        values = P.modal_basis("orthonormal", dim, order, points)
        mass = values.T @ (weights[:, None] * values)
        np.testing.assert_allclose(mass, np.eye(values.shape[1]), rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("dim", DIMS)
def test_dub_orth_mass_matrix_is_diagonal_with_closed_form_norms(dim):
    """``dub_orth`` has a diagonal mass matrix with entries 1 / orthonormal_scale^2 = prod_k 2 / (2 N_k + k)."""
    for order in (0, 1, 4, 8):
        points, weights = gauss_jacobi_collapsed_rule(dim, 2 * order)
        values = P.modal_basis("dub_orth", dim, order, points)
        mass = values.T @ (weights[:, None] * values)
        expected = np.diag(1.0 / P.orthonormal_scale(dim, order) ** 2)
        np.testing.assert_allclose(mass, expected, rtol=0.0, atol=1e-12 * np.abs(mass).max())
    assert P.modal_basis("dub_orth", dim, 0, np.zeros((1, dim)))[0, 0] == 1.0


def test_triangle_orthonormal_is_dub_orth_times_closed_form_diagonal():
    """On the triangle, orthonormal = legacy Dubiner * sqrt((2p + 1)(p + q + 1) / 2), values and gradients."""
    points = np.vstack((_random_points(2, 30, seed=5), reference_simplex(2).vertices))
    for order in range(9):
        scale = np.array([np.sqrt((2 * p + 1) * (p + q + 1) / 2.0) for p, q in legacy.dubiner_pq_order(order)])
        np.testing.assert_allclose(P.orthonormal_scale(2, order), scale, rtol=1e-15)
        np.testing.assert_allclose(
            P.modal_basis("orthonormal", 2, order, points),
            legacy.evaluate_dubiner_basis(order, points) * scale,
            rtol=1e-15,
            atol=0.0,
        )
        np.testing.assert_allclose(
            P.modal_gradients("orthonormal", 2, order, points),
            legacy.evaluate_dubiner_gradients(order, points) * scale[None, :, None],
            rtol=1e-15,
            atol=0.0,
        )


def test_triangle_and_interval_bases_are_the_legacy_bases():
    """2D bases are the kernels of ``hybridge.core.basis``; 1D ``dub_orth`` and ``bernstein`` are the edge bases.

    The barycentric kernels of this module, run in one and two dimensions, give
    the same ``dub_orth`` and ``bernstein`` values to rounding.
    """
    points2 = _random_points(2, 25, seed=6)
    x = np.linspace(-1.0, 1.0, 11)
    for order in range(8):
        assert P.modal_mode_labels("dub_orth", 2, order) == legacy.dubiner_pq_order(order)
        assert P.modal_mode_labels("bernstein", 2, order) == legacy.bernstein_exponents(order)
        np.testing.assert_array_equal(
            P.modal_basis("dub_orth", 2, order, points2), legacy.evaluate_dubiner_basis(order, points2)
        )
        np.testing.assert_array_equal(
            P.modal_basis("bernstein", 2, order, points2), legacy.evaluate_bernstein_basis(order, points2)
        )
        np.testing.assert_array_equal(
            P.modal_basis("hier_C0", 2, order, points2), legacy.evaluate_hierarchical_c0_basis(order, points2)
        )
        np.testing.assert_array_equal(P.modal_basis("dub_orth", 1, order, x), P._legendre_edge_basis(order, x).T)
        np.testing.assert_array_equal(P.modal_basis("bernstein", 1, order, x), P._bernstein_edge_basis(order, x).T)
        index = P._orthogonal_index_array(2, order)
        values, _ = P._orthogonal_kernel(points2, order, index, P.modal_dimension(2, order), False)
        np.testing.assert_allclose(values, legacy.evaluate_dubiner_basis(order, points2), rtol=1e-12, atol=1e-12)
        exponents, coefficients = P._bernstein_arrays(2, order)
        values, _ = P._bernstein_kernel(points2, order, exponents, coefficients, False)
        np.testing.assert_allclose(values, legacy.evaluate_bernstein_basis(order, points2), rtol=1e-13, atol=1e-15)
        index = P._orthogonal_index_array(1, order)
        values, _ = P._orthogonal_kernel(x[:, None], order, index, order + 1, False)
        np.testing.assert_allclose(values, P._legendre_edge_basis(order, x).T, rtol=1e-13, atol=1e-13)
        exponents, coefficients = P._bernstein_arrays(1, order)
        values, _ = P._bernstein_kernel(x[:, None], order, exponents, coefficients, False)
        np.testing.assert_allclose(values, P._bernstein_edge_basis(order, x).T, rtol=1e-13, atol=1e-15)


def test_interval_hier_c0_is_the_modified_basis():
    """1D ``hier_C0`` is (1-x)/2, (1+x)/2, then (1-x)(1+x)/4 P^{(1,1)}_{j-1}(x), j = 1..p-1 (K&S)."""
    x = np.linspace(-1.0, 1.0, 13)
    for order in range(1, 9):
        expected = [0.5 * (1 - x), 0.5 * (1 + x)]
        expected += [0.25 * (1 - x) * (1 + x) * eval_jacobi(j - 1, 1.0, 1.0, x) for j in range(1, order)]
        values = P.modal_basis("hier_C0", 1, order, x)
        np.testing.assert_allclose(values, np.column_stack(expected), rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("dim", DIMS)
def test_all_bases_span_the_same_space(dim):
    """Each basis equals the orthonormal basis times an invertible matrix T: Phi_b = Phi_o T to 1e-12 at random points."""
    for order in (2, 5, 7):
        points = _random_points(dim, 2 * P.modal_dimension(dim, order) + 5, seed=order)
        reference = P.modal_basis("orthonormal", dim, order, points)
        for name in NAMES:
            values = P.modal_basis(name, dim, order, points)
            transform, *_ = np.linalg.lstsq(reference, values, rcond=None)
            residual = np.linalg.norm(reference @ transform - values) / np.linalg.norm(values)
            assert residual < 1e-12, (name, order, residual)
            assert np.linalg.matrix_rank(transform) == values.shape[1]


@pytest.mark.parametrize("dim", DIMS)
def test_bernstein_partition_of_unity_and_nonnegative(dim):
    """Bernstein values are >= 0 in the simplex and sum to 1; their gradients sum to 0."""
    points = np.vstack((_random_points(dim, 30, seed=8), reference_simplex(dim).vertices))
    for order in (0, 1, 3, 7):
        values = P.modal_basis("bernstein", dim, order, points)
        assert values.min() >= 0.0
        np.testing.assert_allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-13)
        np.testing.assert_allclose(P.modal_gradients("bernstein", dim, order, points).sum(axis=1), 0.0, atol=1e-11)


def test_tetrahedron_dub_orth_on_bottom_face_equals_triangle_dub_orth():
    """At xi_3 = -1 the modes (p, q, 0) of the tetrahedron equal the triangle modes (p, q), with their xi_1, xi_2 derivatives."""
    points2 = np.vstack((_random_points(2, 30, seed=9), [[1.0, -1.0], [-1.0, -1.0], [0.0, 0.0]]))
    points3 = np.column_stack((points2, -np.ones(len(points2))))
    for order in range(9):
        labels3 = P.modal_mode_labels("dub_orth", 3, order)
        columns = [labels3.index((p, q, 0)) for p, q in legacy.dubiner_pq_order(order)]
        values3 = P.modal_basis("dub_orth", 3, order, points3)[:, columns]
        np.testing.assert_allclose(values3, legacy.evaluate_dubiner_basis(order, points2), rtol=1e-12, atol=1e-12)
        gradients3 = P.modal_gradients("dub_orth", 3, order, points3)[:, columns, :2]
        np.testing.assert_allclose(
            gradients3, legacy.evaluate_dubiner_gradients(order, points2), rtol=1e-12, atol=1e-10
        )


def test_tetrahedron_dub_orth_equals_collapsed_product():
    """At interior points, 3D ``dub_orth`` equals the K&S product in collapsed coordinates.

    The product is P_p(eta_1) ((1-eta_2)/2)^p P_q^{(2p+1,0)}(eta_2) ((1-eta_3)/2)^{p+q} P_r^{(2p+2q+2,0)}(eta_3).
    """
    points = _random_points(3, 40, seed=10)
    eta = _ks_collapsed(points)
    order = 7
    values = P.modal_basis("dub_orth", 3, order, points)
    for column, (p, q, r) in enumerate(P.modal_mode_labels("dub_orth", 3, order)):
        expected = (
            eval_jacobi(p, 0.0, 0.0, eta[:, 0])
            * (0.5 * (1.0 - eta[:, 1])) ** p
            * eval_jacobi(q, 2.0 * p + 1.0, 0.0, eta[:, 1])
            * (0.5 * (1.0 - eta[:, 2])) ** (p + q)
            * eval_jacobi(r, 2.0 * p + 2.0 * q + 2.0, 0.0, eta[:, 2])
        )
        np.testing.assert_allclose(values[:, column], expected, rtol=1e-11, atol=1e-11)


# ---------------------------------------------------------------------------
# Modified (hier_C0) basis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim, vertex", [(d, v) for d in DIMS for v in range(d + 1)])
def test_hier_c0_vertex_mode_is_barycentric_coordinate(dim, vertex):
    """The ``hier_C0`` mode of ``vertex`` equals lambda_vertex (value and gradient): 1 at it, 0 at the other vertices."""
    simplex = reference_simplex(dim)
    points = np.vstack((_random_points(dim, 25, seed=vertex), simplex.vertices, _special_points(dim)))
    grad_lambda = np.vstack((-0.5 * np.ones((1, dim)), 0.5 * np.eye(dim)))
    for order in (1, 2, 5):
        column = P.modal_mode_labels("hier_C0", dim, order).index(("vertex", vertex, 0))
        values = P.modal_basis("hier_C0", dim, order, points)[:, column]
        gradients = P.modal_gradients("hier_C0", dim, order, points)[:, column]
        np.testing.assert_allclose(values, _barycentric(points)[:, vertex], rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(gradients, np.broadcast_to(grad_lambda[vertex], gradients.shape), atol=1e-15)
        at_vertices = P.modal_basis("hier_C0", dim, order, simplex.vertices)[:, column]
        np.testing.assert_allclose(at_vertices, np.eye(dim + 1)[vertex], atol=1e-15)


_BOOK_VERTEX_CASES = [
    # (dim, vertex name, vertex number, printed formula in collapsed coordinates, what the printed formula equals)
    (2, "A", 0, lambda e: 0.25 * (1 - e[:, 0]) * (1 - e[:, 1]), lambda lam: lam[:, 0]),
    (2, "B", 1, lambda e: 0.25 * (1 + e[:, 0]) * (1 - e[:, 1]), lambda lam: lam[:, 1]),
    (2, "C", 2, lambda e: 0.5 * (1 + e[:, 1]), lambda lam: lam[:, 2]),
    (3, "A", 0, lambda e: 0.125 * (1 - e[:, 0]) * (1 - e[:, 1]) * (1 - e[:, 2]), lambda lam: lam[:, 0]),
    (3, "B", 1, lambda e: 0.125 * (1 + e[:, 0]) * (1 - e[:, 1]) * (1 - e[:, 2]), lambda lam: lam[:, 1]),
    (
        3,
        "C",
        2,
        lambda e: 0.125 * (1 - e[:, 0]) * (1 + e[:, 1]) * (1 - e[:, 2]),
        lambda lam: lam[:, 0] * lam[:, 2] / (lam[:, 0] + lam[:, 1]),
    ),
    (3, "D", 3, lambda e: 0.5 * (1 + e[:, 2]), lambda lam: lam[:, 3]),
]


@pytest.mark.parametrize(
    "dim, name, vertex, printed, equals", _BOOK_VERTEX_CASES, ids=[f"{c[0]}d-{c[1]}" for c in _BOOK_VERTEX_CASES]
)
def test_book_vertex_modes(dim, name, vertex, printed, equals):
    """The vertex modes printed in K&S, Appendix D (triangle p. 604, tetrahedron p. 609), against lambda_vertex.

    Every printed vertex mode equals lambda_v except the tetrahedron vertex C,
    printed as (1-eta_1)/2 (1+eta_2)/2 (1-eta_3)/2: it equals
    lambda_0 lambda_2 / (lambda_0 + lambda_1), a rational function that is not 1 at
    C. Without the factor (1-eta_1)/2 it is (1+eta_2)/2 (1-eta_3)/2 = lambda_2, which
    is the ``hier_C0`` mode of vertex 2.
    """
    order = 4
    points = _random_points(dim, 40, seed=20 + vertex)
    lam = _barycentric(points)
    eta = _ks_collapsed(points)
    np.testing.assert_allclose(printed(eta), equals(lam), rtol=1e-12, atol=1e-14)
    is_barycentric = np.allclose(printed(eta), lam[:, vertex], rtol=1e-12, atol=1e-14)
    assert is_barycentric == ((dim, name) != (3, "C"))
    if (dim, name) == (3, "C"):
        corrected = 0.25 * (1 + eta[:, 1]) * (1 - eta[:, 2])
        np.testing.assert_allclose(corrected, lam[:, 2], rtol=1e-12, atol=1e-14)
    column = P.modal_mode_labels("hier_C0", dim, order).index(("vertex", vertex, 0))
    values = P.modal_basis("hier_C0", dim, order, points)[:, column]
    np.testing.assert_allclose(values, lam[:, vertex], rtol=1e-12, atol=1e-14)


def _contained(label, dim: int, face_vertices: set[int]) -> bool:
    """Whether the entity of a ``hier_C0`` label lies in the closed face with vertex set ``face_vertices``."""
    simplex = reference_simplex(dim)
    kind, entity, _ = label
    if kind == "vertex":
        vertices = {entity}
    elif kind == "edge":
        vertices = set(simplex.edges[entity])
    elif kind == "face":
        vertices = set(simplex.face_vertices[entity])
    else:
        vertices = set(range(dim + 1))
    return vertices <= face_vertices


@pytest.mark.parametrize("dim", DIMS)
def test_hier_c0_modes_vanish_on_faces_not_containing_their_entity(dim):
    """On face f, each ``hier_C0`` mode whose vertex, edge or face is not in f vanishes; C(p + d - 1, d - 1) modes do not."""
    simplex = reference_simplex(dim)
    for order in (1, 2, 3, 4, 6):
        labels = P.modal_mode_labels("hier_C0", dim, order)
        for f, corners in enumerate(simplex.face_vertices):
            if dim == 1:
                face_points = simplex.vertices[list(corners)]
            else:
                face_points = simplex.face_map(f, _random_points(dim - 1, 30, seed=f))
            values = P.modal_basis("hier_C0", dim, order, face_points)
            inside = np.array([_contained(label, dim, set(corners)) for label in labels])
            assert np.abs(values[:, ~inside]).max(initial=0.0) < 1e-13
            assert inside.sum() == comb(order + dim - 1, dim - 1)
            assert np.all(np.abs(values[:, inside]).max(axis=0) > 1e-6)


def test_tetrahedron_hier_c0_equals_book_listing():
    """Every 3D ``hier_C0`` mode equals the K&S Appendix D.1.2.4 listing (pp. 609-610, vertex C corrected) at interior points."""
    points = _random_points(3, 40, seed=30)
    for order in range(1, 8):
        table = P._modified_table(3, order)[0]
        values = P.modal_basis("hier_C0", 3, order, points)
        expected = np.column_stack([_book_tetrahedron_mode(row, points) for row in table])
        np.testing.assert_allclose(values, expected, rtol=1e-12, atol=1e-13)


def test_tetrahedron_hier_c0_on_bottom_face_is_book_triangle_basis():
    """At xi_3 = -1, the 3D modes of face ABC and of its vertices and edges are the triangle basis of K&S p. 604.

    The vertex and edge modes equal the triangle ``hier_C0`` of
    ``hybridge.core.basis`` (whose edge 2, from C to A, lists the degrees in
    decreasing order). The face modes equal the book's triangle interior modes with
    the Jacobi weight (2p + 1, 1); the legacy interior modes are the same products
    with the weight (2p - 1, 1). Both sets span the same space.
    """
    points2 = _random_points(2, 40, seed=31)
    points3 = np.column_stack((points2, -np.ones(len(points2))))
    for order in range(1, 8):
        labels3 = P.modal_mode_labels("hier_C0", 3, order)
        table = P._modified_table(3, order)[0]
        values3 = P.modal_basis("hier_C0", 3, order, points3)
        legacy_values = legacy.evaluate_hierarchical_c0_basis(order, points2)
        legacy_modes = legacy.hierarchical_c0_mode_indexing(order)
        on_face = [m for m, label in enumerate(labels3) if _contained(label, 3, {0, 1, 2})]
        assert len(on_face) == legacy_values.shape[1]
        legacy_columns = list(range(3))
        for e3, e2 in ((0, 0), (1, 1), (2, 2)):
            for k in range(order - 1):
                local = k if e2 < 2 else order - 2 - k
                np.testing.assert_allclose(
                    values3[:, labels3.index(("edge", e3, k))], legacy_values[:, 3 + e2 * (order - 1) + local],
                    rtol=1e-12, atol=1e-14,
                )
        np.testing.assert_allclose(values3[:, :3], legacy_values[:, legacy_columns], rtol=1e-12, atol=1e-14)
        for column in on_face:
            if labels3[column][0] == "face":
                p, q = int(table[column, 5]), int(table[column, 6])
                expected = _book_triangle_interior(p, q, 2.0 * p + 1.0, points2)
                np.testing.assert_allclose(values3[:, column], expected, rtol=1e-12, atol=1e-14)
        for m in range(3 * order, legacy_values.shape[1]):
            p, q = (int(i) for i in legacy_modes[m])
            expected = _book_triangle_interior(p, q, 2.0 * p - 1.0, points2)
            np.testing.assert_allclose(legacy_values[:, m], expected, rtol=1e-12, atol=1e-14)
        transform, *_ = np.linalg.lstsq(legacy_values, values3[:, on_face], rcond=None)
        assert np.linalg.norm(legacy_values @ transform - values3[:, on_face]) < 1e-12 * np.linalg.norm(values3)
        assert np.linalg.matrix_rank(transform) == len(on_face)



# ---------------------------------------------------------------------------
# Gradients and singular points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("name", NAMES)
def test_gradients_equal_richardson_differences(name, dim):
    """Analytic gradients equal extrapolated central differences at random points, the vertices and (3D) the singular edge."""
    points = np.vstack((_random_points(dim, 12, seed=dim), _special_points(dim)))
    for order in (1, 3, 6):
        analytic = P.modal_gradients(name, dim, order, points)
        numeric = _richardson_gradient(name, dim, order, points, h=1e-3)
        scale = max(1.0, np.abs(analytic).max())
        assert np.abs(analytic - numeric).max() <= 1e-8 * scale, (name, dim, order)


@pytest.mark.parametrize("dim", (2, 3))
@pytest.mark.parametrize("name", NAMES)
def test_values_at_singular_points_equal_interior_limits(name, dim):
    """At the points where the collapsed coordinates are undefined, values and gradients are the limits from inside.

    The points are the vertex (-1, 1) of the triangle, and the vertex (-1,-1,1) and
    the edge xi_2 + xi_3 = 0 of the tetrahedron. They are approached along the
    segment to the centroid at distances h = 1e-4, 1e-5, 1e-6; the differences
    must decrease linearly in h.
    """
    singular = np.array([[-1.0, 1.0]]) if dim == 2 else _singular_edge_points(5)
    centroid = np.full(dim, -1.0 + 2.0 / (dim + 1))
    order = 6
    values0 = P.modal_basis(name, dim, order, singular)
    gradients0 = P.modal_gradients(name, dim, order, singular)
    value_gaps, gradient_gaps = [], []
    for h in (1e-4, 1e-5, 1e-6):
        nearby = singular + h * (centroid - singular)
        value_gaps.append(np.abs(P.modal_basis(name, dim, order, nearby) - values0).max())
        gradient_gaps.append(np.abs(P.modal_gradients(name, dim, order, nearby) - gradients0).max())
    scale = max(1.0, np.abs(gradients0).max())
    for gaps in (value_gaps, gradient_gaps):
        assert gaps[0] < 1e-1 * scale
        assert gaps[1] <= 0.2 * gaps[0] + 1e-9 * scale
        assert gaps[2] <= 0.2 * gaps[1] + 1e-9 * scale


def test_triangle_dubiner_vertex_gradients_equal_barycentric_kernel():
    """At the triangle vertices, the legacy Dubiner gradients equal the barycentric kernel of this module in 2D."""
    vertices = np.asarray(reference_simplex(2).vertices)
    for order in range(1, 9):
        index = P._orthogonal_index_array(2, order)
        _, barycentric = P._orthogonal_kernel(vertices, order, index, P.modal_dimension(2, order), True)
        np.testing.assert_allclose(
            legacy.evaluate_dubiner_gradients(order, vertices), barycentric, rtol=1e-13, atol=1e-12
        )
