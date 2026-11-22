"""Node families, Fekete tables, Lagrange bases and Lebesgue constants of ``hybridge.core.simplex.nodes``."""

from __future__ import annotations

from itertools import permutations
from math import comb
from pathlib import Path

import numpy as np
import pytest

from hybridge.core.conforming import lattice_reference_nodes
from hybridge.core.simplex import nodes as N
from hybridge.core.simplex.reference import _legendre_gauss_lobatto, point_permutation, reference_simplex

FIXTURE = Path(__file__).resolve().parent / "data" / "simplex_basix_reference.npz"
GLL_FAMILIES = ("warp_blend", "recursive", "fekete", "fekete_free")
DEGREES = {1: (0, 1, 2, 5, 9), 2: (0, 1, 2, 3, 5, 8, 12, 15), 3: (0, 1, 2, 3, 4, 6, 8, 10)}
CELL = {2: "triangle", 3: "tetrahedron"}


def _cases(families=N.NODE_FAMILIES):
    """Parameters ``(family, dim, order)`` over the degrees of ``DEGREES``."""
    return [(family, d, p) for family in families for d, orders in DEGREES.items() for p in orders]


def _barycentric(dim, x):
    """Barycentric coordinates ``(n, dim + 1)`` of reference points ``x`` ``(n, dim)``."""
    return reference_simplex(dim).barycentric(np.asarray(x).reshape(-1, dim))


def _log_abs_det(family, dim, order):
    """``log|det V|`` of the Vandermonde matrix of a node family in the orthonormal basis."""
    sign, value = np.linalg.slogdet(N.vandermonde(family, dim, order))
    assert sign != 0.0
    return value


def _legacy_lattice(k):
    """The equispaced triangle lattice as computed by ``lattice_reference_nodes`` before it moved (reference copy)."""
    vertices = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
    nodes = [vertex for vertex in vertices]
    for edge in range(3):
        start, end = vertices[edge], vertices[(edge + 1) % 3]
        for j in range(1, k):
            nodes.append(start + (j / k) * (end - start))
    for b in range(1, k):
        for a in range(1, k - b):
            nodes.append(np.array([-1.0 + 2.0 * a / k, -1.0 + 2.0 * b / k]))
    return np.ascontiguousarray(np.asarray(nodes, dtype=np.float64))


@pytest.mark.parametrize("family, dim, order", _cases())
def test_lagrange_basis_is_nodal_and_reproduces_linear_functions(family, dim, order):
    """l_i(x_j) = delta_ij at the nodes; at random points sum_i grad l_i = 0 and sum_i x_i grad l_i = I."""
    x = N.simplex_nodes(family, dim, order)
    assert x.shape == (comb(order + dim, dim), dim) and x.dtype == np.float64
    values = N.lagrange_basis(family, dim, order, x)
    np.testing.assert_allclose(values, np.eye(x.shape[0]), atol=1e-9)
    rng = np.random.default_rng(order + 10 * dim)
    points = reference_simplex(dim).from_barycentric(rng.dirichlet(np.ones(dim + 1), size=7))
    gradients = N.lagrange_gradients(family, dim, order, points)
    assert gradients.shape == (7, x.shape[0], dim)
    # sum_i grad l_i = 0 and sum_i x_i grad l_i = identity whenever order >= 1.
    np.testing.assert_allclose(gradients.sum(axis=1), 0.0, atol=1e-8)
    if order >= 1:
        np.testing.assert_allclose(np.einsum("im,qin->qmn", x, gradients), np.broadcast_to(np.eye(dim), (7, dim, dim)), atol=1e-8)


@pytest.mark.parametrize("family, dim, order", _cases())
def test_node_sets_are_invariant_under_vertex_permutations(family, dim, order):
    """Every permutation of the barycentric coordinates maps the node set onto itself (as a set, 1e-12)."""
    x = N.simplex_nodes(family, dim, order)
    lam = _barycentric(dim, x)
    assert lam.min() > -1e-14
    simplex = reference_simplex(dim)
    for sigma in permutations(range(dim + 1)):
        point_permutation(simplex.from_barycentric(lam[:, list(sigma)]), x, atol=1e-12)


@pytest.mark.parametrize("family", [f for f in N.NODE_FAMILIES if f != "fekete_free"])
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_nodes_are_ordered_by_entity_with_the_expected_counts(family, dim):
    """Counts C(p-1, k) per k-simplex, and nodes ordered vertices, edges (first to second vertex), faces, interior."""
    simplex = reference_simplex(dim)
    for order in range(1, 9):
        counts = N.node_entity_counts(dim, order)
        assert counts == tuple(comb(order - 1, k) for k in range(dim + 1))
        assert counts[:4] == (1, order - 1, (order - 1) * (order - 2) // 2, (order - 1) * (order - 2) * (order - 3) // 6)[: dim + 1]
        assert sum(comb(dim + 1, k + 1) * c for k, c in enumerate(counts)) == comb(order + dim, dim)
        x = N.simplex_nodes(family, dim, order)
        if dim == 1:
            assert np.all(np.diff(x[:, 0]) > 0)
            continue
        support = (_barycentric(dim, x) > 1e-12).sum(axis=1) - 1  # dimension of the carrying sub-simplex
        expected = np.concatenate([np.full(comb(dim + 1, k + 1) * c, k) for k, c in enumerate(counts)])
        np.testing.assert_array_equal(support, expected)
        np.testing.assert_array_equal(x[: dim + 1], simplex.vertices)
        for e, (a, b) in enumerate(simplex.edges):
            block = _barycentric(dim, x[dim + 1 + e * (order - 1) : dim + 1 + (e + 1) * (order - 1)])
            assert np.all(block[:, [k for k in range(dim + 1) if k not in (a, b)]] < 1e-12)
            assert np.all(np.diff(block[:, b]) > 0)  # from the first to the second vertex
    assert N.node_entity_counts(dim, 0) == (0,) * dim + (1,)
    np.testing.assert_allclose(N.simplex_nodes(family, dim, 0), simplex.vertices.mean(axis=0, keepdims=True))


@pytest.mark.parametrize("order", range(0, 13))
def test_one_dimensional_nodes_are_the_ascending_gll_points(order):
    """On the interval every family except equispaced returns the Gauss-Lobatto-Legendre points in increasing order."""
    gll = _legendre_gauss_lobatto(order + 1)[0]
    for family in GLL_FAMILIES:
        np.testing.assert_array_equal(N.simplex_nodes(family, 1, order)[:, 0], gll)
    if order:
        np.testing.assert_allclose(N.simplex_nodes("equispaced", 1, order)[:, 0], np.linspace(-1.0, 1.0, order + 1), atol=1e-15)


@pytest.mark.parametrize("family", ["equispaced", "recursive", "fekete", "warp_blend"])
@pytest.mark.parametrize("order", range(1, 11))
def test_tetrahedron_faces_carry_the_triangle_set(family, order):
    """Face blocks are the face maps of the triangle interior nodes; warp_blend differs from p = 4 on (3D alpha)."""
    simplex = reference_simplex(3)
    x = N.simplex_nodes(family, 3, order)
    triangle_interior = N.simplex_nodes(family, 2, order)[3 * order :]
    count = (order - 1) * (order - 2) // 2
    offset = 4 + 6 * (order - 1)
    gaps = []
    for f in range(4):
        block = x[offset + f * count : offset + (f + 1) * count]
        if family == "warp_blend" and order >= 4:
            # Nodes3D warps faces with the 3D blending parameter: the restriction differs from Nodes2D.
            gaps.append(np.abs(block - simplex.face_map(f, triangle_interior)).max())
            continue
        np.testing.assert_allclose(block, simplex.face_map(f, triangle_interior), atol=1e-14)
        # The face restriction, read in the face's own coordinates, is the whole triangle set.
        on_face = x[np.abs(_barycentric(3, x)[:, [v for v in range(4) if v not in simplex.face_vertices[f]][0]]) < 1e-12]
        face_lam = _barycentric(3, on_face)[:, list(simplex.face_vertices[f])]
        point_permutation(reference_simplex(2).from_barycentric(face_lam), N.simplex_nodes(family, 2, order), atol=1e-13)
    if gaps:
        # alpha_2D(4) = 0.1001 and alpha_3D(4) = 0.1002 move the face nodes by about 1e-7; above p = 4 by 1e-4 or more.
        assert min(gaps) > (1e-8 if order == 4 else 1e-4)


def test_fekete_table_entries_are_stationary_and_increase_the_determinant():
    """Every stored entry: small projected gradient, determinant above its start, warp_blend and (free) fekete."""
    table = N.fekete_table()
    assert set(table["generator"]) >= {"hybridge_commit", "numpy", "scipy", "nodes_module_sha256"}
    keys = {f"{v}/d{d}/p{p}" for v in N.FEKETE_VARIANTS for d, top in N.FEKETE_TABLE_MAX_ORDER.items() for p in range(1, top + 1)}
    assert keys <= set(table["entries"])
    for key in keys:
        entry = table["entries"][key]
        d, p, variant = entry["dim"], entry["order"], entry["variant"]
        scale = max(1.0, abs(entry["log_abs_det"]))
        assert entry["projected_gradient_norm"] <= 1e-8 * scale, key
        assert entry["log_abs_det"] >= entry["start_log_abs_det"] - 1e-12 * scale, key
        assert abs(_log_abs_det(variant, d, p) - entry["log_abs_det"]) <= 1e-10 * scale, key
        assert entry["lebesgue_estimate"] >= 1.0 and entry["optimizer"]["status"] in (0, 2)
        best_reference = _log_abs_det("warp_blend", d, p)
        if variant == "fekete_free":
            best_reference = max(best_reference, table["entries"][f"fekete/d{d}/p{p}"]["log_abs_det"])
            assert entry["log_abs_det"] >= best_reference - 1e-10 * scale, key
        elif d == 2:
            assert entry["log_abs_det"] >= best_reference - 1e-10 * scale, key


@pytest.mark.parametrize("dim, order", [(2, 5), (2, 12), (3, 5), (3, 9)])
@pytest.mark.parametrize("variant", N.FEKETE_VARIANTS)
def test_fekete_nodes_satisfy_the_first_order_conditions(variant, dim, order):
    """Recompute the projected gradient of log|det V| at stored nodes, and the determinant of the recorded start."""
    entry = N.fekete_table()["entries"][f"{variant}/d{dim}/p{order}"]
    x = N.simplex_nodes(variant, dim, order)
    free = N._interior_orbit if variant == "fekete" else N._any_orbit
    orbits = N._detect_orbits(dim, x, free)
    objective = N._FeketeObjective(dim, order, orbits, x.shape[0])
    value, gradient = objective(objective.start)
    assert abs(-value - entry["log_abs_det"]) <= 1e-10 * abs(value)
    assert np.abs(objective.projected_gradient(objective.start, gradient)).max() <= 1e-8 * abs(value)
    # The start set recorded in the entry has a smaller determinant.
    if variant == "fekete":
        triangle = N.fekete_table()["entries"][f"fekete/d2/p{order}"] if dim == 3 else None
        start = N._face_compatible_start(dim, order, entry["start"].split()[0], triangle)
        sign, start_value = np.linalg.slogdet(N._orthonormal_values(dim, order, start))
        assert sign != 0.0 and start_value <= entry["log_abs_det"] + 1e-12 * abs(value)
        np.testing.assert_allclose(start_value, entry["start_log_abs_det"], rtol=1e-12)


@pytest.mark.parametrize("dim", [2, 3])
def test_stored_fekete_table_is_reproduced_at_degree_four(dim):
    """Recomputing degree 4 gives the stored orbit types, nodes (1e-10) and log|det V| (1e-13 relative)."""
    table = N.fekete_table()["entries"]
    triangle = N.compute_fekete_entry(2, 4, "fekete") if dim == 3 else None
    fekete = N.compute_fekete_entry(dim, 4, "fekete", triangle_entry=triangle)
    free = N.compute_fekete_entry(dim, 4, "fekete_free", fekete_entry=fekete)
    for entry in (fekete, free):
        stored = table[f"{entry['variant']}/d{dim}/p4"]
        assert [o["type"] for o in entry["orbits"]] == [o["type"] for o in stored["orbits"]]
        np.testing.assert_allclose(N._nodes_from_entry(entry), N.simplex_nodes(entry["variant"], dim, 4), atol=1e-10)
        np.testing.assert_allclose(entry["log_abs_det"], stored["log_abs_det"], rtol=1e-13)


def test_fekete_sets_above_the_table_are_computed_with_a_warning():
    """A degree above the table is computed on request with a RuntimeWarning and is nodal."""
    order = N.FEKETE_TABLE_MAX_ORDER[2] + 1
    with pytest.warns(RuntimeWarning, match="computed in"):
        x = N.simplex_nodes("fekete", 2, order)
    np.testing.assert_allclose(N.lagrange_basis("fekete", 2, order, x), np.eye(x.shape[0]), atol=1e-9)
    assert _log_abs_det("fekete", 2, order) > _log_abs_det("warp_blend", 2, order)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("family, key", [("recursive", "gll_isaac"), ("equispaced", "equispaced")])
def test_recursive_and_equispaced_nodes_equal_basix(dim, family, key):
    """recursive and equispaced equal basix gll_isaac and equispaced lattices after x = 2 X - 1 (as sets, 1e-13)."""
    data = np.load(FIXTURE)
    for order in range(1, 9):
        basix_points = 2.0 * data[f"{CELL[dim]}_{key}_p{order}"] - 1.0  # x = 2 X - 1
        point_permutation(N.simplex_nodes(family, dim, order), basix_points, atol=1e-13)


@pytest.mark.parametrize("dim, order", [(2, 6), (2, 9), (3, 6), (3, 7)])
def test_warp_blend_lebesgue_constant_is_below_equispaced(dim, order):
    """Lebesgue constant of warp_blend below equispaced, and not below the maximum on the degree-6p lattice."""
    warp = N.lebesgue_constant("warp_blend", dim, order)
    assert warp < N.lebesgue_constant("equispaced", dim, order)
    assert warp >= 1.0
    # The estimate refines the lattice maximum and cannot fall below it.
    x = reference_simplex(dim).from_barycentric(N._lattice_multi_indices(dim, 6 * order) / (6.0 * order))
    assert warp >= np.abs(N.lagrange_basis("warp_blend", dim, order, x)).sum(axis=1).max() - 1e-12


def test_lebesgue_constant_of_low_degrees():
    """Lebesgue constants 1 for degrees 0 and 1, and 5/3 for the quadratic triangle lattice."""
    assert N.lebesgue_constant("equispaced", 2, 0) == 1.0
    for dim in (1, 2, 3):
        np.testing.assert_allclose(N.lebesgue_constant("recursive", dim, 1), 1.0, atol=1e-12)
    np.testing.assert_allclose(N.lebesgue_constant("equispaced", 2, 2), 5.0 / 3.0, atol=1e-9)


@pytest.mark.parametrize("degree", range(1, 13))
def test_conforming_lattice_is_unchanged(degree):
    """lattice_reference_nodes equals, bit for bit, the expressions of its former implementation."""
    nodes = lattice_reference_nodes(degree)
    assert nodes.tobytes() == _legacy_lattice(degree).tobytes()
    assert nodes.dtype == np.float64 and nodes.flags.c_contiguous and nodes.flags.writeable


def test_invalid_arguments_raise():
    """Unknown family, dimension, degree or point shape raise ValueError."""
    with pytest.raises(ValueError, match="unknown node family"):
        N.simplex_nodes("gauss", 2, 3)
    with pytest.raises(ValueError, match="dimension"):
        N.simplex_nodes("equispaced", 4, 3)
    with pytest.raises(ValueError, match="degree"):
        N.simplex_nodes("equispaced", 2, -1)
    with pytest.raises(ValueError, match="shape"):
        N.lagrange_basis("equispaced", 3, 2, np.zeros((4, 2)))
    with pytest.raises(ValueError, match="at least 1"):
        lattice_reference_nodes(0)
