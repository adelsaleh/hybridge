"""Reference simplices of dimension 1, 2 and 3: geometry, face orientations and quadrature.

Checks :mod:`hybridge.core.simplex.reference`: vertices, faces, face maps, measure
ratios and outward normals; the action of the face-vertex permutations on face
points, with a 3-cycle that separates a permutation from its inverse; exactness
of every quadrature rule for all monomials of total degree ``<= k``, ``k <= 20``,
against the exact integrals of ``monomial_integral``; and the index permutations
that the face orientations induce on the symmetric face rules.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import product
from math import factorial

import numpy as np
import pytest

from hybridge.core.simplex.reference import (
    ReferenceSimplex,
    compose_permutations,
    face_permutations,
    face_quadrature_permutation,
    gauss_jacobi_collapsed_rule,
    inverse_permutation,
    monomial_integral,
    permute_face_points,
    point_permutation,
    reference_simplex,
    simplex_face_rule,
    simplex_volume_rule,
    triangle_edge_points,
)

DIMENSIONS = (1, 2, 3)
MAX_DEGREE = 20
RULES = {
    "gauss_jacobi_collapsed_rule": gauss_jacobi_collapsed_rule,
    "simplex_volume_rule": simplex_volume_rule,
}


def _random_interior_points(dim: int, count: int, seed: int = 0) -> np.ndarray:
    """Return ``count`` random points strictly inside the reference ``dim``-simplex, shape ``(count, dim)``."""
    rng = np.random.default_rng(seed)
    lam = rng.dirichlet(np.ones(dim + 1), size=count)
    return 2.0 * lam[:, 1:] - 1.0


def _reference_face_vertices(dim: int) -> np.ndarray:
    """Vertices of the reference face of the ``dim``-simplex, shape ``(dim, dim - 1)``."""
    if dim == 1:
        return np.zeros((1, 0))
    return np.asarray(reference_simplex(dim - 1).vertices)


@lru_cache(maxsize=None)
def _monomials(dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Exponents ``(M, dim)`` of all monomials of total degree ``<= MAX_DEGREE`` and their exact integrals ``(M,)``."""
    exponents = sorted(
        (a for a in product(range(MAX_DEGREE + 1), repeat=dim) if sum(a) <= MAX_DEGREE),
        key=lambda a: (sum(a), a),
    )
    integrals = [float(monomial_integral(a)) for a in exponents]
    return np.asarray(exponents, dtype=np.int64), np.asarray(integrals)


def _monomial_quadrature(points: np.ndarray, weights: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    r"""Apply a rule to the monomials :math:`\prod_k \xi_k^{a_k}`; ``exponents`` ``(M, d)`` gives ``(M,)`` sums."""
    powers = points[None, :, :] ** np.arange(MAX_DEGREE + 1)[:, None, None]  # (degree, nq, d)
    values = np.ones((exponents.shape[0], points.shape[0]))
    for k in range(points.shape[1]):
        values *= powers[exponents[:, k], :, k]
    return values @ weights


# --- geometry -----------------------------------------------------------------


def test_reference_simplex_tables() -> None:
    """Vertices, measures, face and edge numbering, measure ratios and normals equal the documented values."""
    interval, triangle, tetrahedron = (reference_simplex(d) for d in DIMENSIONS)
    np.testing.assert_array_equal(interval.vertices, [[-1.0], [1.0]])
    np.testing.assert_array_equal(triangle.vertices, [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]])
    np.testing.assert_array_equal(
        tetrahedron.vertices, [[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]
    )
    assert [s.measure for s in (interval, triangle, tetrahedron)] == [2.0, 2.0, 4.0 / 3.0]
    assert interval.face_vertices == ((0,), (1,))
    assert triangle.face_vertices == ((0, 1), (1, 2), (2, 0))
    assert tetrahedron.face_vertices == ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
    assert interval.edges == ((0, 1),)
    assert triangle.edges == triangle.face_vertices
    assert tetrahedron.edges == ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
    np.testing.assert_array_equal(interval.face_measure_ratios, [1.0, 1.0])
    np.testing.assert_allclose(triangle.face_measure_ratios, [1.0, np.sqrt(2.0), 1.0], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(tetrahedron.face_measure_ratios, [np.sqrt(3.0), 1.0, 1.0, 1.0], rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(interval.outward_normals, [[-1.0], [1.0]])
    np.testing.assert_allclose(
        triangle.outward_normals, [[0.0, -1.0], [2**-0.5, 2**-0.5], [-1.0, 0.0]], rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        tetrahedron.outward_normals,
        [[3**-0.5] * 3, [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        rtol=0.0,
        atol=1e-15,
    )


def test_reference_simplex_is_shared_read_only_and_rejects_other_dimensions() -> None:
    """One instance per dimension, arrays not writable, and ValueError for dimensions other than 1, 2, 3."""
    for dim in DIMENSIONS:
        simplex = reference_simplex(dim)
        assert isinstance(simplex, ReferenceSimplex)
        assert simplex is reference_simplex(dim)
        assert simplex.dim == dim
        for array in (simplex.vertices, simplex.face_measure_ratios, simplex.outward_normals):
            assert array.dtype == np.float64
            assert not array.flags.writeable
    for dim in (0, 4, 2.5):
        with pytest.raises(ValueError):
            reference_simplex(dim)


@pytest.mark.parametrize("dim", DIMENSIONS)
def test_face_map_sends_reference_face_vertices_to_face_vertices(dim: int) -> None:
    """Vertex ``i`` of the reference face goes to vertex ``face_vertices[f][i]`` of the simplex."""
    simplex = reference_simplex(dim)
    corners = _reference_face_vertices(dim)
    for f, face in enumerate(simplex.face_vertices):
        np.testing.assert_allclose(simplex.face_map(f, corners), simplex.vertices[list(face)], rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("dim", DIMENSIONS)
def test_face_jacobian_is_the_affine_part_of_the_face_map_and_gives_the_ratio(dim: int) -> None:
    """``face_map(f, s) = face_map(f, 0) + J_f s``, ``sqrt(det J^T J) = ratio_f`` and ``J_f^T n_f = 0``."""
    simplex = reference_simplex(dim)
    s = _random_interior_points(dim - 1, 7, seed=dim) if dim > 1 else np.zeros((7, 0))
    origin = np.zeros((1, dim - 1))
    for f in range(dim + 1):
        jacobian = simplex.face_jacobian(f)
        assert jacobian.shape == (dim, dim - 1)
        np.testing.assert_allclose(
            simplex.face_map(f, s), simplex.face_map(f, origin) + s @ jacobian.T, rtol=0.0, atol=1e-14
        )
        ratio = np.sqrt(np.linalg.det(jacobian.T @ jacobian)) if dim > 1 else 1.0
        np.testing.assert_allclose(simplex.face_measure_ratios[f], ratio, rtol=1e-15)
        np.testing.assert_allclose(jacobian.T @ simplex.outward_normals[f], 0.0, atol=1e-15)


@pytest.mark.parametrize("dim", DIMENSIONS)
def test_outward_normals_are_unit_outward_and_satisfy_the_divergence_theorem(dim: int) -> None:
    """``|n_f| = 1``, ``n_f`` points away from the opposite vertex, and ``sum_f ratio_f n_f = 0``."""
    simplex = reference_simplex(dim)
    np.testing.assert_allclose(np.linalg.norm(simplex.outward_normals, axis=1), 1.0, rtol=0.0, atol=1e-15)
    for f, face in enumerate(simplex.face_vertices):
        (opposite,) = set(range(dim + 1)) - set(face)
        assert simplex.outward_normals[f] @ (simplex.vertices[face[0]] - simplex.vertices[opposite]) > 0.0
        centroid_gap = simplex.vertices[list(face)].mean(axis=0) - simplex.vertices.mean(axis=0)
        assert simplex.outward_normals[f] @ centroid_gap > 0.0
    total = simplex.face_measure_ratios @ simplex.outward_normals
    np.testing.assert_allclose(total, 0.0, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("dim", DIMENSIONS)
def test_barycentric_round_trip(dim: int) -> None:
    """Barycentric coordinates sum to one, equal ``delta_ij`` at the vertices and invert ``from_barycentric``."""
    simplex = reference_simplex(dim)
    np.testing.assert_allclose(simplex.barycentric(simplex.vertices), np.eye(dim + 1), rtol=0.0, atol=1e-15)
    points = _random_interior_points(dim, 50)
    lam = simplex.barycentric(points)
    assert lam.shape == (50, dim + 1)
    assert np.all(lam > 0.0)
    np.testing.assert_allclose(lam.sum(axis=1), 1.0, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(simplex.from_barycentric(lam), points, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(lam @ simplex.vertices, points, rtol=0.0, atol=1e-15)
    with pytest.raises(ValueError):
        simplex.barycentric(np.zeros((3, dim + 1)))


def test_triangle_face_map_equals_the_legacy_edge_map() -> None:
    """On the triangle, ``face_map(f, t)`` equals ``triangle_edge_points(t)[:, f]`` for every edge."""
    simplex = reference_simplex(2)
    t = np.concatenate(([-1.0, 1.0, 0.0], np.polynomial.legendre.leggauss(9)[0], np.linspace(-1.0, 1.0, 11)))
    legacy = triangle_edge_points(t)
    for f in range(3):
        np.testing.assert_allclose(simplex.face_map(f, t[:, None]), legacy[:, f], rtol=0.0, atol=2e-16)
        # The reversed legacy edge, evaluated at -t, is the orientation sigma = (1, 0).
        np.testing.assert_allclose(
            simplex.face_map(f, permute_face_points((1, 0), t[:, None])),
            triangle_edge_points(-t)[:, f],
            rtol=0.0,
            atol=1e-15,
        )


# --- face orientations ---------------------------------------------------------


def test_face_permutations_list_identity_first() -> None:
    """``d!`` permutations, identity first, in ``itertools.permutations`` order."""
    assert face_permutations(1) == ((0,),)
    assert face_permutations(2) == ((0, 1), (1, 0))
    assert face_permutations(3) == ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


@pytest.mark.parametrize("dim", DIMENSIONS)
def test_face_permutations_form_a_group(dim: int) -> None:
    """The set is closed under composition and inversion, and the point action is a homomorphism."""
    group = face_permutations(dim)
    identity = group[0]
    points = _random_interior_points(dim - 1, 9, seed=3) if dim > 1 else np.zeros((9, 0))
    for a in group:
        inverse = inverse_permutation(a)
        assert inverse in group
        assert compose_permutations(a, inverse) == identity
        assert compose_permutations(inverse, a) == identity
        for b in group:
            ab = compose_permutations(a, b)
            assert ab in group
            np.testing.assert_allclose(
                permute_face_points(ab, points),
                permute_face_points(a, permute_face_points(b, points)),
                rtol=0.0,
                atol=1e-15,
            )
    np.testing.assert_array_equal(permute_face_points(identity, points), points)


def test_composition_order_and_three_cycle_distinguish_sigma_from_its_inverse() -> None:
    """``compose(a, b)[i] = a[b[i]]``; the 3-cycle (1, 2, 0) moves local vertex 0 to global vertex 1."""
    assert compose_permutations((1, 0, 2), (0, 2, 1)) == (1, 2, 0)
    assert compose_permutations((0, 2, 1), (1, 0, 2)) == (2, 0, 1)
    sigma = (1, 2, 0)
    assert inverse_permutation(sigma) == (2, 0, 1)
    triangle = reference_simplex(2)
    # Local barycentric coordinates (0.5, 0.3, 0.2): lambda_glob[sigma[i]] = lambda_loc[i]
    # gives (0.2, 0.5, 0.3), i.e. xi = (0.0, -0.4); the inverse gives (0.3, 0.2, 0.5), xi = (-0.6, 0.0).
    point = triangle.from_barycentric(np.array([[0.5, 0.3, 0.2]]))
    np.testing.assert_allclose(permute_face_points(sigma, point), [[0.0, -0.4]], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        permute_face_points(inverse_permutation(sigma), point), [[-0.6, 0.0]], rtol=0.0, atol=1e-15
    )
    # Local vertex 0 lies at global vertex sigma[0] = 1.
    np.testing.assert_allclose(
        permute_face_points(sigma, triangle.vertices), triangle.vertices[list(sigma)], rtol=0.0, atol=1e-15
    )


def test_two_dimensional_reversal_maps_t_to_minus_t() -> None:
    """On a triangle edge, ``sigma = (1, 0)`` maps ``t`` to ``-t``."""
    t = np.linspace(-1.0, 1.0, 13)[:, None]
    np.testing.assert_allclose(permute_face_points((1, 0), t), -t, rtol=0.0, atol=1e-15)


def test_orientation_describes_the_same_physical_point_on_a_shared_face() -> None:
    """With ``g_{sigma[i]} = m_i``, the local and the global face maps agree after ``permute_face_points``.

    A tetrahedron face is given by three points; the "global" order lists them
    as ``g`` and an element lists them as ``m``. The point with local reference
    coordinates ``s`` is the point with global coordinates
    ``permute_face_points(sigma, s)``.
    """
    rng = np.random.default_rng(5)
    corners = rng.normal(size=(3, 3))
    s = _random_interior_points(2, 11, seed=6)

    def physical(order: tuple[int, ...], points: np.ndarray) -> np.ndarray:
        """Map reference-triangle points to the face with vertex ``i`` at ``corners[order[i]]``, shape ``(n, 3)``."""
        return reference_simplex(2).barycentric(points) @ corners[list(order)]

    global_order = (0, 1, 2)
    for sigma in face_permutations(3):
        local_order = tuple(global_order[sigma[i]] for i in range(3))
        np.testing.assert_allclose(
            physical(local_order, s), physical(global_order, permute_face_points(sigma, s)), rtol=0.0, atol=1e-14
        )


@pytest.mark.parametrize(
    ("dim", "degree"),
    [(1, 0), (1, 6)]
    + [(2, k) for k in (0, 1, 2, 5, 10, 13, 20)]
    + [(3, k) for k in (0, 1, 2, 4, 7, 8, 13, 14, 15, 17, 20)],
)
def test_face_orientations_permute_the_points_of_the_face_rules(dim: int, degree: int) -> None:
    """Every orientation maps the face rule onto itself: ``p`` is a permutation and keeps the weights."""
    points, weights = simplex_face_rule(dim, degree)
    group = face_permutations(dim)
    tables = {}
    for sigma in group:
        p = face_quadrature_permutation(dim, points, sigma)
        assert p.dtype == np.int64
        np.testing.assert_array_equal(np.sort(p), np.arange(len(weights)))
        np.testing.assert_allclose(points[p], permute_face_points(sigma, points), rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(weights[p], weights, rtol=1e-14, atol=0.0)
        tables[sigma] = p
    np.testing.assert_array_equal(tables[group[0]], np.arange(len(weights)))
    for a in group:
        for b in group:
            np.testing.assert_array_equal(tables[compose_permutations(a, b)], tables[a][tables[b]])


def test_point_permutation_matches_coincident_points_and_reports_mismatch() -> None:
    """A reordering is recovered, repeated points are paired in index order, and a mismatch raises."""
    rng = np.random.default_rng(1)
    target = rng.normal(size=(20, 3))
    order = rng.permutation(20)
    p = point_permutation(target[order] + 1e-14, target)
    np.testing.assert_array_equal(p, order)
    # Repeated points: the copies are paired in increasing index order on both sides.
    doubled = np.vstack((target[:3], target[:3]))
    np.testing.assert_array_equal(point_permutation(doubled, doubled), np.arange(6))
    p = point_permutation(doubled[::-1], doubled)
    np.testing.assert_array_equal(p, [2, 1, 0, 5, 4, 3])
    np.testing.assert_array_equal(doubled[p], doubled[::-1])
    with pytest.raises(ValueError, match="largest distance"):
        point_permutation(target + 1e-6, target)
    with pytest.raises(ValueError):
        point_permutation(target[:5], target)


def test_face_quadrature_permutation_rejects_a_non_symmetric_rule() -> None:
    """The collapsed triangle rule is not invariant under the 3-cycle, so ValueError is raised."""
    points, _ = gauss_jacobi_collapsed_rule(2, 4)
    with pytest.raises(ValueError, match="largest distance"):
        face_quadrature_permutation(3, points, (1, 2, 0))


# --- quadrature -------------------------------------------------------------------


def test_jacobi_roots_use_the_weight_one_minus_x_to_the_alpha() -> None:
    """``roots_jacobi(n, alpha, 0)`` integrates against ``(1 - x)^alpha``: checked on 1 and on x."""
    from scipy.special import roots_jacobi

    for alpha in (0, 1, 2):
        x, w = roots_jacobi(4, float(alpha), 0.0)
        np.testing.assert_allclose(w.sum(), 2.0 ** (alpha + 1) / (alpha + 1), rtol=1e-14)
        first_moment = {0: 0.0, 1: -2.0 / 3.0, 2: -4.0 / 3.0}[alpha]
        np.testing.assert_allclose(w @ x, first_moment, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("rule_name", sorted(RULES))
@pytest.mark.parametrize("dim", DIMENSIONS)
def test_rules_integrate_every_monomial_through_their_degree(rule_name: str, dim: int) -> None:
    """For ``k = 0..20``, the rule integrates all monomials of total degree ``<= k`` exactly (to rounding)."""
    exponents, exact = _monomials(dim)
    for degree in range(MAX_DEGREE + 1):
        points, weights = RULES[rule_name](dim, degree)
        selected = exponents.sum(axis=1) <= degree
        got = _monomial_quadrature(points, weights, exponents[selected])
        np.testing.assert_allclose(got, exact[selected], rtol=0.0, atol=1e-13, err_msg=f"degree {degree}")


@pytest.mark.parametrize("rule_name", sorted(RULES))
@pytest.mark.parametrize("dim", DIMENSIONS)
def test_rules_have_positive_weights_summing_to_the_measure_and_interior_points(rule_name: str, dim: int) -> None:
    """Weights are positive ``float64`` and sum to ``2^d/d!``; every point has positive barycentric coordinates."""
    simplex = reference_simplex(dim)
    for degree in range(MAX_DEGREE + 1):
        points, weights = RULES[rule_name](dim, degree)
        assert points.dtype == np.float64 and weights.dtype == np.float64
        assert points.shape == (weights.size, dim)
        assert np.all(weights > 0.0)
        np.testing.assert_allclose(weights.sum(), simplex.measure, rtol=1e-14)
        assert np.all(simplex.barycentric(points) > 0.0)


@pytest.mark.parametrize("dim", DIMENSIONS)
def test_collapsed_rule_uses_ceil_half_k_plus_one_points_per_direction(dim: int) -> None:
    """``gauss_jacobi_collapsed_rule(d, k)`` has ``ceil((k + 1)/2)^d`` points."""
    for degree in range(MAX_DEGREE + 1):
        points, _ = gauss_jacobi_collapsed_rule(dim, degree)
        assert points.shape[0] == (-(-(degree + 1) // 2)) ** dim


def test_face_rules_are_the_volume_rules_one_dimension_lower() -> None:
    """``simplex_face_rule(d, k) = simplex_volume_rule(d - 1, k)``; for ``d = 1`` one point of weight 1."""
    points, weights = simplex_face_rule(1, 7)
    assert points.shape == (1, 0)
    np.testing.assert_array_equal(weights, [1.0])
    for dim in (2, 3):
        for degree in (0, 3, 14, 15):
            face_points, face_weights = simplex_face_rule(dim, degree)
            volume_points, volume_weights = simplex_volume_rule(dim - 1, degree)
            np.testing.assert_array_equal(face_points, volume_points)
            np.testing.assert_array_equal(face_weights, volume_weights)


def test_rules_reject_negative_degrees() -> None:
    """A negative or non-integer exact degree raises ValueError."""
    for rule in (gauss_jacobi_collapsed_rule, simplex_volume_rule, simplex_face_rule):
        with pytest.raises(ValueError):
            rule(2, -1)
        with pytest.raises(ValueError):
            rule(2, 1.5)


# --- exact monomial integrals -------------------------------------------------------


def test_monomial_integral_spot_values() -> None:
    """Measures and a few integrals computed by hand on the interval, triangle and tetrahedron."""
    assert monomial_integral((0,)) == 2
    assert monomial_integral((0, 0)) == 2
    assert monomial_integral((0, 0, 0)) == Fraction(4, 3)
    for dim in DIMENSIONS:
        assert monomial_integral((0,) * dim) == Fraction(2**dim, factorial(dim))
    assert monomial_integral((1,)) == 0
    assert monomial_integral((2,)) == Fraction(2, 3)
    # Triangle: integral of x = area * centroid = 2 * (-1/3); x^2 = int_{-1}^{1} x^2 (1 - x) dx; xy = 0.
    assert monomial_integral((1, 0)) == Fraction(-2, 3)
    assert monomial_integral((2, 0)) == Fraction(2, 3)
    assert monomial_integral((1, 1)) == 0
    # Tetrahedron: integral of xi_1 = (4/3) * (-1/2); xi_3^2 = int_{-1}^{1} z^2 (1 - z)^2 / 2 dz;
    # xi_1 xi_2 xi_3 from the Dirichlet formula.
    assert monomial_integral((1, 0, 0)) == Fraction(-2, 3)
    assert monomial_integral((0, 0, 2)) == Fraction(8, 15)
    assert monomial_integral((1, 1, 1)) == Fraction(-2, 45)
