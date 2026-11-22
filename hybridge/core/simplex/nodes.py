r"""Interpolation nodes on the reference simplices, nodal Lagrange bases and Lebesgue constants.

The reference simplices are those of :func:`~hybridge.core.simplex.reference.reference_simplex`:
the interval ``[-1, 1]``, the triangle ``(-1,-1), (1,-1), (-1,1)`` and the
tetrahedron ``(-1,-1,-1), (1,-1,-1), (-1,1,-1), (-1,-1,1)``, with barycentric
coordinates :math:`\lambda_k = (1 + \xi_k)/2` for ``k >= 1`` and
:math:`\lambda_0 = 1 - \sum_{k \ge 1} \lambda_k`.

A node set of degree ``p`` on the ``d``-simplex has :math:`n_u = \binom{p+d}{d}`
points, the dimension of the polynomials :math:`\mathbb P_p` of total degree at
most ``p``. Five families are provided (:data:`NODE_FAMILIES`); all are invariant
under the :math:`(d+1)!` permutations of the vertices:

- ``"equispaced"``: the lattice :math:`\lambda = \alpha / p` over the
  multi-indices :math:`\alpha \in \mathbb N^{d+1}`, :math:`|\alpha| = p`.
- ``"warp_blend"``: the warp and blend nodes of Warburton (J. Eng. Math. 56, 2006)
  as written in Hesthaven and Warburton, *Nodal Discontinuous Galerkin Methods*
  (2008), ``Nodes2D`` and ``Nodes3D``: the lattice is moved so that each edge
  carries the Gauss-Lobatto-Legendre (GLL) points, with the blending parameter
  :math:`\alpha` of their optimized tables for ``p <= 15`` (``5/3`` in 2D and
  ``1`` in 3D above). ``Nodes3D`` moves the faces with the 3D parameter, so the
  face nodes of the tetrahedron are not the triangle ``warp_blend`` nodes for
  ``p >= 4`` (they are for ``p <= 3``).
- ``"recursive"``: the recursive nodes of Isaac (SIAM J. Sci. Comput. 42, 2020)
  built from the GLL points of the interval; the nodes of each face are the
  triangle nodes of the same family.
- ``"fekete"``: edges carry the GLL points, the faces of the tetrahedron carry the
  triangle ``"fekete"`` nodes, and the interior nodes maximize
  :math:`\log|\det V|` over symmetric orbits (see below).
- ``"fekete_free"``: every node, boundary nodes included, moves within the closed
  simplex to maximize :math:`\log|\det V|` over symmetric orbits.

Here :math:`V_{ij} = \psi_j(x_i)` is the Vandermonde matrix of the nodes
:math:`x_i` in the :math:`L^2`-orthonormal basis :math:`\psi_j` of
:math:`\mathbb P_p` (``modal_basis("orthonormal", ...)``). On the interval every
family except ``"equispaced"`` is the GLL set, which is also the Fekete set of the
interval (Fejer).

**Node order.** For ``d = 1`` the nodes are in increasing order. For ``d >= 2``
they are ordered by the sub-simplex whose relative interior contains them:
the vertices in vertex order; then the interior nodes of each edge, in the edge
order of ``reference_simplex(d).edges``, listed from the edge's first to its
second vertex; for ``d = 3`` the interior nodes of each face, in the order of
``face_vertices``, obtained by applying ``ReferenceSimplex.face_map`` to the
interior nodes of the triangle set (so the triangle vertex ``i`` goes to the
``i``-th listed vertex of the face); last the interior nodes of the simplex. In
every family the node with lattice multi-index :math:`\alpha` keeps the place of
the equispaced node :math:`\alpha / p`: interior multi-indices are listed with
:math:`\alpha_d` varying slowest and :math:`\alpha_1` fastest. ``"fekete_free"``
keeps the order of the set it starts from, even when a boundary node has moved
into the interior. Degree 0 is the single centroid.

**Fekete computation.** The start set is split into orbits of the vertex
permutation group :math:`S_{d+1}`; each orbit is the set of permutations of one
barycentric vector with a fixed pattern of equal coordinates (types ``S3``,
``S21``, ``S111`` on the triangle; ``S4``, ``S31``, ``S22``, ``S211``, ``S1111``
on the tetrahedron) and is described by 0 to 3 parameters in a box, so that the
nodes stay in the closed simplex and the set stays symmetric. The objective
:math:`\log|\det V|` is maximized with ``scipy.optimize.minimize`` (L-BFGS-B)
using the exact gradient

.. math::

    \frac{\partial \log|\det V|}{\partial x_i}
    = \sum_j (V^{-1})_{ji}\, \nabla\psi_j(x_i) = \nabla \ell_i(x_i),

chained through the orbit parameters, followed by a few Newton steps when the
projected gradient is not yet small. The sets of degree ``p <= 15`` (triangle)
and ``p <= 10`` (tetrahedron) are stored in ``data/fekete.json``, written by
``scripts/dev/compute_fekete_points.py``; higher degrees are computed on request,
kept in memory for the rest of the process, and announced with a
``RuntimeWarning`` that gives the time taken.

For every degree of the stored table, both starts of ``"fekete_free"`` (the
``"fekete"`` set and ``"warp_blend"``) end at the ``"fekete"`` set, equal to
``1e-13``: within the orbit structure of these starts, the face-compatible set is
a stationary point of the problem with free boundary nodes, whose boundary orbits
stay on the boundary of the simplex. This is a computed result; the code does not
assume it, and ``"fekete_free"`` is computed separately.

**Lagrange bases.** The nodal basis :math:`\ell_i` with :math:`\ell_i(x_j) =
\delta_{ij}` is :math:`\ell(x) = \psi(x) V^{-1}` (row vectors), evaluated by
solving with the LU factors of :math:`V` (``scipy.linalg.lu_factor``), which are
computed once per family, dimension and degree. All arithmetic is in ``float64``.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import warnings
from functools import lru_cache
from importlib import resources
from itertools import combinations, permutations
from math import comb

import numpy as np
import scipy.linalg
import scipy.optimize

from hybridge.core.simplex.polynomials import _as_points, modal_basis, modal_gradients
from hybridge.core.simplex.reference import _legendre_gauss_lobatto, point_permutation, reference_simplex

NODE_FAMILIES = ("equispaced", "warp_blend", "recursive", "fekete", "fekete_free")
FEKETE_VARIANTS = ("fekete", "fekete_free")
# Degrees stored in data/fekete.json, per dimension.
FEKETE_TABLE_MAX_ORDER = {2: 15, 3: 10}

# Blending parameters alpha of Hesthaven and Warburton (2008), Nodes2D (alpopt) and
# Nodes3D (alphastore), indexed by the degree p = 1..15.
_WARP_ALPHA_2D = (0.0, 0.0, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999, 1.2832, 1.3648, 1.4773, 1.4959, 1.5743,
                  1.5770, 1.6223, 1.6258)
_WARP_ALPHA_3D = (0.0, 0.0, 0.0, 0.1002, 1.1332, 1.5608, 1.3413, 1.2577, 1.1603, 1.10153, 0.6080, 0.4523,
                  0.8856, 0.8717, 0.9655)
_WARP_ALPHA_ABOVE = {2: 5.0 / 3.0, 3: 1.0}

# Symmetric orbits. A type is named by the multiplicities of the distinct
# barycentric coordinates of its points; its "labels" give, for each of the d + 1
# coordinates of the representative, the index of its value in the vector of
# distinct values computed by _orbit_values.
_ORBIT_LABELS = {
    "S3": (0, 0, 0),
    "S21": (0, 0, 1),
    "S111": (0, 1, 2),
    "S4": (0, 0, 0, 0),
    "S31": (0, 0, 0, 1),
    "S22": (0, 0, 1, 1),
    "S211": (0, 0, 1, 2),
    "S1111": (0, 1, 2, 3),
}
_ORBIT_TYPE_BY_MULTIPLICITY = {
    (3,): "S3", (2, 1): "S21", (1, 1, 1): "S111",
    (4,): "S4", (3, 1): "S31", (2, 2): "S22", (2, 1, 1): "S211", (1, 1, 1, 1): "S1111",
}
_ORBIT_BOUNDS = {
    "S3": (), "S21": ((0.0, 0.5),), "S111": ((0.0, 1.0), (0.0, 1.0)),
    "S4": (), "S31": ((0.0, 1.0 / 3.0),), "S22": ((0.0, 0.25),), "S211": ((0.0, 1.0), (0.0, 1.0)),
    "S1111": ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
}
_ORBIT_TOLERANCE = 1.0e-9
_LEBESGUE_CHUNK = 4096

_COMPUTED_FEKETE: dict[tuple[str, int, int], dict] = {}
_COMPUTED_FEKETE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Argument checks and entity counts
# ---------------------------------------------------------------------------


def _check_arguments(family, dim, order) -> tuple[str, int, int]:
    """Return ``(family, dim, order)`` as ``(str, int, int)``; raise ``ValueError`` for an unknown family,
    a dimension other than 1, 2, 3 or a negative or non-integer degree."""
    if family not in NODE_FAMILIES:
        raise ValueError(f"unknown node family {family!r}; expected one of {NODE_FAMILIES}")
    d = int(dim)
    if d != dim or d not in (1, 2, 3):
        raise ValueError(f"the reference simplex has dimension 1, 2 or 3; got {dim!r}")
    p = int(order)
    if p != order or p < 0:
        raise ValueError(f"the polynomial degree must be an integer >= 0; got {order!r}")
    return family, d, p


def node_entity_counts(dim, order) -> tuple[int, ...]:
    r"""Number of nodes in the relative interior of each ``k``-dimensional sub-simplex, ``k = 0, ..., dim``.

    For degree ``p >= 1`` the count for one ``k``-simplex is :math:`\binom{p-1}{k}`:
    1 per vertex, ``p - 1`` per edge, ``(p-1)(p-2)/2`` per triangle and
    ``(p-1)(p-2)(p-3)/6`` per tetrahedron, the same for every family of
    :data:`NODE_FAMILIES` (for ``"fekete_free"`` it counts the nodes of the start
    set, see the module docstring). The result has length ``dim + 1``; entry
    ``dim`` is the number of interior nodes of the simplex. For ``p = 0`` the only
    node is the centroid and the result is ``(0, ..., 0, 1)``. With
    :math:`\binom{d+1}{k+1}` sub-simplices of dimension ``k``, the total is
    :math:`\binom{p+d}{d}`.
    """
    _, d, p = _check_arguments("equispaced", dim, order)
    if p == 0:
        return (0,) * d + (1,)
    return tuple(comb(p - 1, k) for k in range(d + 1))


# ---------------------------------------------------------------------------
# Lattice multi-indices in node order
# ---------------------------------------------------------------------------


def _interior_multi_indices(dim: int, order: int) -> np.ndarray:
    r"""Multi-indices :math:`\alpha \in \mathbb N^{d+1}` with :math:`|\alpha| = p` and every entry >= 1.

    Shape ``(C(p - 1, d), d + 1)``, dtype ``int64``. Entry 0 is
    :math:`p - \alpha_1 - \dots - \alpha_d`; rows are listed with :math:`\alpha_d`
    varying slowest and :math:`\alpha_1` fastest, the order of the interior nodes of
    the legacy triangle lattice of :mod:`hybridge.core.conforming`.
    """
    p = int(order)
    if p < dim + 1:
        return np.zeros((0, dim + 1), dtype=np.int64)
    grids = np.meshgrid(*([np.arange(1, p)] * dim), indexing="ij")
    tail = np.stack([g.ravel() for g in grids], axis=1)
    tail = tail[tail.sum(axis=1) <= p - 1]
    # numpy.lexsort sorts by its last key first: alpha_d slowest, alpha_1 fastest.
    tail = tail[np.lexsort(tail.T)]
    return np.ascontiguousarray(np.column_stack((p - tail.sum(axis=1), tail)), dtype=np.int64)


def _lattice_multi_indices(dim: int, degree: int) -> np.ndarray:
    r"""All multi-indices :math:`\alpha \in \mathbb N^{d+1}` with :math:`|\alpha| = m`, ``m = degree``.

    Shape ``(C(m + d, d), d + 1)``, dtype ``int64``, in no particular order; the
    points :math:`\lambda = \alpha/m` form the equispaced lattice of degree ``m``.
    """
    m = int(degree)
    grids = np.meshgrid(*([np.arange(m + 1)] * dim), indexing="ij")
    tail = np.stack([g.ravel() for g in grids], axis=1)
    tail = tail[tail.sum(axis=1) <= m]
    return np.ascontiguousarray(np.column_stack((m - tail.sum(axis=1), tail)), dtype=np.int64)


# ---------------------------------------------------------------------------
# One-dimensional families
# ---------------------------------------------------------------------------


def _gll_points(order: int) -> np.ndarray:
    """The ``order + 1`` Gauss-Lobatto-Legendre points of ``[-1, 1]`` in increasing order, shape ``(order + 1,)``."""
    points, _ = _legendre_gauss_lobatto(int(order) + 1)
    return np.asarray(points, dtype=np.float64)


@lru_cache(maxsize=None)
def _gll_unit_table(order: int) -> np.ndarray:
    """Table ``T`` of shape ``(order + 1, order + 1)`` with ``T[n, j] = (1 + t_j)/2`` for the GLL points
    ``t_0 < ... < t_n`` of degree ``n`` (``n + 1`` points), ``j <= n``; entries with ``j > n`` are zero.

    Row ``n`` is the GLL family of degree ``n`` on ``[0, 1]`` used by the recursive construction.
    """
    table = np.zeros((order + 1, order + 1), dtype=np.float64)
    for n in range(order + 1):
        table[n, : n + 1] = 0.5 * (1.0 + _gll_points(n))
    table.setflags(write=False)
    return table


def _edge_parameters(family: str, order: int) -> np.ndarray:
    """Increasing positions ``s_j`` in ``(0, 1)`` of the interior nodes of an edge, shape ``(order - 1,)``.

    ``j / order`` for ``"equispaced"``; ``(1 + t_j) / 2`` for the interior GLL points ``t_j`` otherwise.
    """
    if family == "equispaced":
        return np.arange(1, order) / order
    return 0.5 * (1.0 + _gll_points(order)[1:-1])


# ---------------------------------------------------------------------------
# Warp and blend (Hesthaven and Warburton 2008)
# ---------------------------------------------------------------------------


def _warp_alpha(dim: int, order: int) -> float:
    """Blending parameter alpha of Nodes2D (``dim = 2``) or Nodes3D (``dim = 3``) for degree ``order``."""
    table = _WARP_ALPHA_2D if dim == 2 else _WARP_ALPHA_3D
    return table[order - 1] if order <= len(table) else _WARP_ALPHA_ABOVE[dim]


def _warp_factor(order: int, r: np.ndarray) -> np.ndarray:
    r"""Edge warp of degree ``p`` divided by :math:`1 - r^2`, at ``r`` in ``[-1, 1]`` (any shape).

    With the equispaced points :math:`e_i = -1 + 2i/p` and the GLL points
    :math:`t_i`, the warp :math:`w(r) = \sum_i (t_i - e_i)\,L_i(r)` (:math:`L_i` the
    Lagrange polynomials on the :math:`e_i`) vanishes at :math:`r = \pm 1`, and

    .. math::

        q(r) = \frac{w(r)}{1 - r^2}
        = \sum_{i=1}^{p-1} \frac{t_i - e_i}{1 - e_i^2}
          \prod_{\substack{j=1 \\ j \ne i}}^{p-1} \frac{r - e_j}{e_i - e_j}

    is a polynomial; it is the function ``Warpfactor`` of Nodes2D and
    ``evalwarp`` of Nodes3D. The loops run over the ``p - 1`` interior indices.
    """
    p = int(order)
    equispaced = -1.0 + 2.0 * np.arange(p + 1) / p
    gll = _gll_points(p)
    result = np.zeros_like(r, dtype=np.float64)
    for i in range(1, p):
        term = np.full_like(result, (gll[i] - equispaced[i]) / (1.0 - equispaced[i] ** 2))
        for j in range(1, p):
            if j != i:
                term = term * (r - equispaced[j]) / (equispaced[i] - equispaced[j])
        result = result + term
    return result


def _face_warp(order: int, alpha: float, lam: np.ndarray, face: tuple[int, ...]) -> np.ndarray:
    r"""Barycentric displacement of the warp of one triangle, shape ``lam.shape``.

    ``lam`` ``(n, d + 1)`` are barycentric coordinates and ``face`` lists the three
    coordinates of a triangle (the whole simplex for ``d = 2``, a face for
    ``d = 3``). For each edge ``(i, j)`` of the triangle, with ``k`` its third
    vertex, the node moves along the edge direction by

    .. math::

        w_{ij} = 4\lambda_i\lambda_j\, q(\lambda_j - \lambda_i)\,(1 + (\alpha\lambda_k)^2),

    measured on an equilateral triangle of side 2, which is the barycentric
    displacement :math:`w_{ij}(e_j - e_i)/2`. This is ``Nodes2D`` and
    ``evalshift`` of ``Nodes3D`` without the equilateral coordinates.
    """
    shift = np.zeros_like(lam)
    for i, j in combinations(face, 2):
        (k,) = set(face) - {i, j}
        w = 4.0 * lam[:, i] * lam[:, j] * _warp_factor(order, lam[:, j] - lam[:, i]) * (1.0 + (alpha * lam[:, k]) ** 2)
        shift[:, j] += 0.5 * w
        shift[:, i] -= 0.5 * w
    return shift


def _warp_blend_barycentric(dim: int, order: int, multi_indices: np.ndarray) -> np.ndarray:
    r"""Barycentric coordinates ``(n, d + 1)`` of the warp and blend nodes of the given lattice multi-indices.

    ``d = 2``: :math:`\lambda = \alpha/p` plus the triangle warp of
    :func:`_face_warp` (Nodes2D). ``d = 3``: :math:`\lambda = \alpha/p` plus, for
    each face ``a`` (opposite vertex ``a``, vertices ``b, c, e``), its triangle warp
    times the blend

    .. math::

        \beta_a = \frac{(1 + (\alpha\lambda_a)^2)\,\lambda_b\lambda_c\lambda_e}
                       {(\lambda_b + \lambda_a/2)(\lambda_c + \lambda_a/2)(\lambda_e + \lambda_a/2)}

    (Nodes3D); :math:`\beta_a = 1` on face ``a`` and 0 on the other faces. The
    formula is used for face and interior nodes; edge nodes are set to the GLL
    points by the caller.
    """
    lam = multi_indices / float(order)
    alpha = _warp_alpha(dim, order)
    if dim == 2:
        return lam + _face_warp(order, alpha, lam, (0, 1, 2))
    shift = np.zeros_like(lam)
    for a in range(4):
        others = tuple(k for k in range(4) if k != a)
        b, c, e = others
        numerator = lam[:, b] * lam[:, c] * lam[:, e]
        denominator = (lam[:, b] + 0.5 * lam[:, a]) * (lam[:, c] + 0.5 * lam[:, a]) * (lam[:, e] + 0.5 * lam[:, a])
        blend = numerator.copy()
        positive = denominator > 1.0e-10
        blend[positive] = (1.0 + (alpha * lam[positive, a]) ** 2) * numerator[positive] / denominator[positive]
        shift += blend[:, None] * _face_warp(order, alpha, lam, others)
    return lam + shift


# ---------------------------------------------------------------------------
# Recursive nodes (Isaac 2020)
# ---------------------------------------------------------------------------


def _recursive_barycentric(multi_indices: np.ndarray, table: np.ndarray) -> np.ndarray:
    r"""Barycentric coordinates of the recursive nodes of Isaac (2020) for rows of multi-indices.

    ``multi_indices`` ``(n, m + 1)`` may have different sums :math:`|\alpha|`
    per row; ``table`` is :func:`_gll_unit_table`, whose row ``k`` is the GLL family
    :math:`x_{k,0} < \dots < x_{k,k}` of degree ``k`` on ``[0, 1]``. With
    :math:`k = |\alpha|`:

    .. math::

        b_1(\alpha) = (x_{k,\alpha_0},\, x_{k,\alpha_1}), \qquad
        b_m(\alpha) = \frac{\sum_{i=0}^{m} x_{k,\,k-\alpha_i}\;
                      \iota_i\, b_{m-1}(\alpha_{\setminus i})}
                     {\sum_{i=0}^{m} x_{k,\,k-\alpha_i}},

    where :math:`\alpha_{\setminus i}` deletes entry ``i`` and :math:`\iota_i`
    inserts a zero coordinate at position ``i``. Rows are processed together; the
    loop runs over the ``m + 1`` coordinates.
    """
    alpha = np.asarray(multi_indices, dtype=np.int64)
    total = alpha.sum(axis=1)
    m = alpha.shape[1] - 1
    if m == 1:
        return np.column_stack((table[total, alpha[:, 0]], table[total, alpha[:, 1]]))
    result = np.zeros(alpha.shape, dtype=np.float64)
    weight_sum = np.zeros(alpha.shape[0], dtype=np.float64)
    for i in range(m + 1):
        weight = table[total, total - alpha[:, i]]
        sub = _recursive_barycentric(np.delete(alpha, i, axis=1), table)
        result[:, :i] += weight[:, None] * sub[:, :i]
        result[:, i + 1 :] += weight[:, None] * sub[:, i:]
        weight_sum += weight
    return result / weight_sum[:, None]


# ---------------------------------------------------------------------------
# Assembly of a node set by entity
# ---------------------------------------------------------------------------


def _interior_barycentric(family: str, dim: int, order: int) -> np.ndarray:
    """Barycentric coordinates ``(C(order - 1, dim), dim + 1)`` of the interior nodes of a lattice family.

    ``family`` is ``"equispaced"``, ``"warp_blend"`` or ``"recursive"``; rows follow
    :func:`_interior_multi_indices`.
    """
    alpha = _interior_multi_indices(dim, order)
    if family == "equispaced":
        return alpha / float(order)
    if family == "warp_blend":
        return _warp_blend_barycentric(dim, order, alpha)
    return _recursive_barycentric(alpha, _gll_unit_table(order))


def _assemble_nodes(dim: int, order: int, edge_parameters: np.ndarray, face_points, interior_points) -> np.ndarray:
    """Concatenate vertices, edge, face and interior nodes into the node order of the module docstring.

    ``edge_parameters`` ``(p - 1,)`` are the positions ``s`` in ``(0, 1)`` of the edge
    nodes; edge ``(a, b)`` gets ``v_a + s (v_b - v_a)``. ``face_points`` (``d = 3``)
    is either an array ``((p-1)(p-2)/2, 2)`` of triangle interior nodes in reference
    coordinates, mapped to each face with ``ReferenceSimplex.face_map``, or a list of
    four arrays ``((p-1)(p-2)/2, 3)`` already on the faces. ``interior_points`` has
    shape ``(C(p - 1, d), d)``. Returns ``(C(p + d, d), d)`` ``float64``.
    """
    simplex = reference_simplex(dim)
    vertices = np.asarray(simplex.vertices, dtype=np.float64)
    blocks = [vertices[k][None, :] for k in range(dim + 1)]
    for a, b in simplex.edges:
        start, end = vertices[a], vertices[b]
        blocks.append(start[None, :] + edge_parameters[:, None] * (end - start)[None, :])
    if dim == 3:
        for f in range(4):
            if isinstance(face_points, np.ndarray):
                blocks.append(simplex.face_map(f, face_points))
            else:
                blocks.append(np.asarray(face_points[f], dtype=np.float64))
    blocks.append(np.asarray(interior_points, dtype=np.float64).reshape(-1, dim))
    nodes = np.ascontiguousarray(np.vstack(blocks), dtype=np.float64)
    expected = comb(order + dim, dim)
    if nodes.shape != (expected, dim):  # pragma: no cover - construction invariant
        raise AssertionError(f"node set has shape {nodes.shape}; expected ({expected}, {dim})")
    return nodes


def _lattice_family_nodes(family: str, dim: int, order: int) -> np.ndarray:
    """Nodes ``(C(p + d, d), d)`` of ``"equispaced"``, ``"warp_blend"`` or ``"recursive"`` for ``d >= 2``, ``p >= 1``.

    Edges carry :func:`_edge_parameters`. The faces of the tetrahedron carry the
    interior triangle nodes of the same family through the face maps, except for
    ``"warp_blend"``, whose face nodes come from the Nodes3D formula. The 2D
    equispaced nodes are computed with the expressions of the legacy
    ``lattice_reference_nodes`` (``v_a + (j/p)(v_b - v_a)`` on edges,
    ``-1 + 2 alpha_k / p`` inside), so that they agree with it bit for bit.
    """
    simplex = reference_simplex(dim)
    edges = _edge_parameters(family, order)
    if family == "equispaced":
        interior = -1.0 + 2.0 * _interior_multi_indices(dim, order)[:, 1:] / order
    else:
        interior = simplex.from_barycentric(_interior_barycentric(family, dim, order))
    faces = None
    if dim == 3:
        if family == "warp_blend":
            faces = []
            face_alpha = _interior_multi_indices(2, order)
            for f, corners in enumerate(simplex.face_vertices):
                alpha = np.zeros((face_alpha.shape[0], 4), dtype=np.int64)
                alpha[:, list(corners)] = face_alpha
                faces.append(simplex.from_barycentric(_warp_blend_barycentric(3, order, alpha)))
        else:
            faces = _lattice_family_nodes(family, 2, order)[3 * order :]
    return _assemble_nodes(dim, order, edges, faces, interior)


def _centroid(dim: int) -> np.ndarray:
    """The centroid of the reference ``dim``-simplex, shape ``(1, dim)``."""
    return np.asarray(reference_simplex(dim).vertices, dtype=np.float64).mean(axis=0, keepdims=True)


@lru_cache(maxsize=None)
def _cached_nodes(family: str, dim: int, order: int) -> np.ndarray:
    """Read-only node array of :func:`simplex_nodes` (checked arguments), computed once per process."""
    if order == 0:
        nodes = _centroid(dim)
    elif dim == 1:
        if family == "equispaced":
            nodes = (-1.0 + 2.0 * np.arange(order + 1) / order)[:, None]
        else:
            nodes = _gll_points(order)[:, None]
    elif family in FEKETE_VARIANTS:
        nodes = _nodes_from_entry(_fekete_entry(dim, order, family))
    else:
        nodes = _lattice_family_nodes(family, dim, order)
    nodes = np.ascontiguousarray(nodes, dtype=np.float64)
    nodes.setflags(write=False)
    return nodes


def simplex_nodes(family, dim, order) -> np.ndarray:
    r"""Interpolation nodes of degree ``order`` of one family on the reference ``dim``-simplex.

    ``family`` is one of :data:`NODE_FAMILIES`, ``dim`` is 1, 2 or 3 and
    ``order >= 0``. Returns a new ``float64`` array of shape
    ``(C(order + dim, dim), dim)`` in reference coordinates, ordered as described in
    the module docstring (increasing for ``dim = 1``; vertices, edges, faces,
    interior for ``dim >= 2``). Degree 0 returns the centroid. For ``dim = 2``,
    ``"equispaced"`` equals :func:`hybridge.core.conforming.lattice_reference_nodes`
    bit for bit. The Fekete families of degree above :data:`FEKETE_TABLE_MAX_ORDER`
    are computed on the first request (see the module docstring), which can take
    minutes. Raises ``ValueError`` for an unknown family, dimension or degree.
    """
    family, d, p = _check_arguments(family, dim, order)
    return np.array(_cached_nodes(family, d, p))


# ---------------------------------------------------------------------------
# Vandermonde matrix and Lagrange bases
# ---------------------------------------------------------------------------


def _orthonormal_values(dim: int, order: int, points: np.ndarray, parallel: bool = False) -> np.ndarray:
    """Orthonormal modal basis of degree ``order`` at ``points`` ``(n, dim)``; ``float64``, shape ``(n, n_u)``.

    ``parallel=True`` evaluates the points in parallel (Numba threads).
    """
    return np.asarray(modal_basis("orthonormal", dim, order, points, parallel=parallel), dtype=np.float64)


def _orthonormal_gradients(dim: int, order: int, points: np.ndarray) -> np.ndarray:
    """Reference gradients of the orthonormal modal basis at ``points`` ``(n, dim)``; ``float64``, ``(n, n_u, dim)``."""
    return np.asarray(modal_gradients("orthonormal", dim, order, points), dtype=np.float64)


@lru_cache(maxsize=None)
def _cached_vandermonde_lu(family: str, dim: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """LU factors ``(lu, piv)`` of ``scipy.linalg.lu_factor`` of the Vandermonde matrix, computed once per process."""
    matrix = _orthonormal_values(dim, order, _cached_nodes(family, dim, order))
    lu, piv = scipy.linalg.lu_factor(matrix, check_finite=False)
    lu.setflags(write=False)
    piv.setflags(write=False)
    return lu, piv


def vandermonde(family, dim, order) -> np.ndarray:
    r"""Vandermonde matrix :math:`V_{ij} = \psi_j(x_i)` of the nodes of one family, shape ``(n_u, n_u)``.

    :math:`x_i` are the rows of :func:`simplex_nodes` and :math:`\psi_j` the
    :math:`L^2(\hat K)`-orthonormal modal basis of degree ``order``
    (``modal_basis("orthonormal", dim, order, x)``), with ``n_u = C(order + dim, dim)``.
    """
    family, d, p = _check_arguments(family, dim, order)
    return _orthonormal_values(d, p, _cached_nodes(family, d, p))


def lagrange_basis(family, dim, order, points) -> np.ndarray:
    r"""Nodal Lagrange basis :math:`\ell_i` of one node family, evaluated at ``points``.

    ``points`` has shape ``(n, dim)`` (or ``(n,)`` for ``dim = 1``). Returns
    ``(n, n_u)`` with entry ``[q, i]`` equal to :math:`\ell_i(x_q)`, where
    :math:`\ell_i \in \mathbb P_p` and :math:`\ell_i(x_j) = \delta_{ij}` on the nodes
    :math:`x_j` of :func:`simplex_nodes`. The values are :math:`\psi(x_q) V^{-1}`,
    obtained by solving :math:`V^T \ell(x_q)^T = \psi(x_q)^T` with the stored LU
    factors of the Vandermonde matrix :math:`V` (:func:`vandermonde`).
    """
    family, d, p = _check_arguments(family, dim, order)
    x = _as_points(points, d)
    values = _orthonormal_values(d, p, x)
    solved = scipy.linalg.lu_solve(_cached_vandermonde_lu(family, d, p), values.T, trans=1, check_finite=False)
    return np.ascontiguousarray(solved.T)


def lagrange_gradients(family, dim, order, points) -> np.ndarray:
    r"""Reference gradients of the nodal Lagrange basis of one node family at ``points``.

    Returns ``(n, n_u, dim)`` with entry ``[q, i, m]`` equal to
    :math:`\partial \ell_i / \partial \xi_m (x_q)`, computed as
    :math:`\partial_m \psi(x_q) V^{-1}` with the stored LU factors of :math:`V`.
    ``points`` is as in :func:`lagrange_basis`.
    """
    family, d, p = _check_arguments(family, dim, order)
    x = _as_points(points, d)
    gradients = _orthonormal_gradients(d, p, x)
    n, n_u, _ = gradients.shape
    rhs = gradients.transpose(1, 0, 2).reshape(n_u, n * d)
    solved = scipy.linalg.lu_solve(_cached_vandermonde_lu(family, d, p), rhs, trans=1, check_finite=False)
    return np.ascontiguousarray(solved.reshape(n_u, n, d).transpose(1, 0, 2))


# ---------------------------------------------------------------------------
# Lebesgue constant
# ---------------------------------------------------------------------------


def _project_to_simplex(dim: int, x: np.ndarray) -> np.ndarray:
    """Map points ``(n, dim)`` into the closed simplex: negative barycentric coordinates are set to zero and
    the coordinates are divided by their sum. Points of the simplex are unchanged."""
    simplex = reference_simplex(dim)
    lam = np.clip(simplex.barycentric(x), 0.0, None)
    return simplex.from_barycentric(lam / lam.sum(axis=1, keepdims=True))


def _lebesgue_function(dim: int, order: int, lu_piv, x: np.ndarray) -> np.ndarray:
    r"""Lebesgue function :math:`\Lambda(x) = \sum_i |\ell_i(x)|` at points ``(n, dim)``, shape ``(n,)``.

    ``lu_piv`` are the LU factors of the Vandermonde matrix of the nodes; points are
    processed in blocks of 4096 to bound the memory of the modal values, and blocks
    of more than 256 points are evaluated in parallel.
    """
    result = np.empty(x.shape[0], dtype=np.float64)
    for start in range(0, x.shape[0], _LEBESGUE_CHUNK):
        block = x[start : start + _LEBESGUE_CHUNK]
        values = _orthonormal_values(dim, order, block, parallel=block.shape[0] > 256)
        ell = scipy.linalg.lu_solve(lu_piv, values.T, trans=1, check_finite=False)
        result[start : start + block.shape[0]] = np.abs(ell).sum(axis=0)
    return result


def _lebesgue_estimate(dim: int, order: int, nodes: np.ndarray, resolution: int | None = None) -> float:
    r"""Lower estimate of the Lebesgue constant :math:`\max_{\hat K} \sum_i |\ell_i|` of the nodes ``(n_u, dim)``.

    The Lebesgue function is evaluated on the equispaced lattice of degree
    ``resolution`` (default ``max(6 order, 6)``); from the ten lattice points with the
    largest values, Nelder-Mead (``scipy.optimize.minimize``) maximizes it over the
    simplex (points outside are mapped back by :func:`_project_to_simplex`). The
    largest value found is returned; the true maximum can only be larger.
    """
    if order == 0:
        return 1.0
    lu_piv = scipy.linalg.lu_factor(_orthonormal_values(dim, order, nodes), check_finite=False)
    m = max(6 * order, 6) if resolution is None else int(resolution)
    simplex = reference_simplex(dim)
    lattice = simplex.from_barycentric(_lattice_multi_indices(dim, m) / float(m))
    values = _lebesgue_function(dim, order, lu_piv, lattice)
    best = float(values.max())
    starts = lattice[np.argsort(values)[::-1][:10]]
    step = 1.0 / m

    def negative_lebesgue(y: np.ndarray) -> float:
        """Minus the Lebesgue function at the point ``y`` ``(dim,)`` mapped into the simplex."""
        return -float(_lebesgue_function(dim, order, lu_piv, _project_to_simplex(dim, y[None, :]))[0])

    for start in starts:
        initial = np.vstack((start, start + step * np.eye(dim)))
        result = scipy.optimize.minimize(
            negative_lebesgue, start, method="Nelder-Mead",
            options={"initial_simplex": initial, "xatol": 1.0e-10, "fatol": 1.0e-13, "maxiter": 400 * dim},
        )
        best = max(best, -float(result.fun))
    return best


def lebesgue_constant(family, dim, order, *, resolution=None) -> float:
    r"""Estimate of the Lebesgue constant :math:`\Lambda_p = \max_{x \in \hat K} \sum_i |\ell_i(x)|` of a node family.

    The Lagrange basis :math:`\ell_i` is that of :func:`lagrange_basis`. The
    Lebesgue function is evaluated on the equispaced lattice of degree
    ``resolution`` (default ``6 * order``, at least 6) and then maximized locally by
    Nelder-Mead from the ten lattice points with the largest values. The result is
    a lower bound of :math:`\Lambda_p`, close to it when the lattice is fine enough
    to sample every peak of the Lebesgue function. Degree 0 returns 1. Raises
    ``ValueError`` when ``resolution`` is not an integer >= 1.
    """
    family, d, p = _check_arguments(family, dim, order)
    if resolution is not None and (int(resolution) != resolution or resolution < 1):
        raise ValueError(f"resolution must be an integer >= 1; got {resolution!r}")
    return _lebesgue_estimate(d, p, _cached_nodes(family, d, p), resolution)


# ---------------------------------------------------------------------------
# Symmetric orbits
# ---------------------------------------------------------------------------


def _orbit_values(kind: str, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Distinct barycentric values of an orbit representative and their derivatives in the parameters.

    ``theta`` holds the parameters of the orbit type ``kind``, each in the interval
    of ``_ORBIT_BOUNDS``. Returns ``values`` ``(n_values,)`` and ``jacobian``
    ``(n_values, len(theta))``; the representative is ``values[_ORBIT_LABELS[kind]]``
    and its ``d + 1`` coordinates sum to one:

    - ``S3``: ``(1/3)``; ``S4``: ``(1/4)``;
    - ``S21``: ``(a, 1 - 2a)``; ``S31``: ``(a, 1 - 3a)``; ``S22``: ``(a, 1/2 - a)``;
    - ``S111``: ``(u, (1-u) v, (1-u)(1-v))``; ``S211``: ``(u/2, (1-u) v, (1-u)(1-v))``;
    - ``S1111``: ``(u, (1-u) v, (1-u)(1-v) w, (1-u)(1-v)(1-w))``.

    The last three are collapsed coordinates: the box ``[0, 1]^k`` covers the whole
    closed simplex of representatives.
    """
    t = np.asarray(theta, dtype=np.float64)
    if kind in ("S3", "S4"):
        return np.array([1.0 / len(_ORBIT_LABELS[kind])]), np.zeros((1, 0))
    if kind in ("S21", "S31", "S22"):
        a = t[0]
        if kind == "S21":
            return np.array([a, 1.0 - 2.0 * a]), np.array([[1.0], [-2.0]])
        if kind == "S31":
            return np.array([a, 1.0 - 3.0 * a]), np.array([[1.0], [-3.0]])
        return np.array([a, 0.5 - a]), np.array([[1.0], [-1.0]])
    if kind in ("S111", "S211"):
        u, v = t
        scale = 0.5 if kind == "S211" else 1.0
        values = np.array([scale * u, (1.0 - u) * v, (1.0 - u) * (1.0 - v)])
        jacobian = np.array([[scale, 0.0], [-v, 1.0 - u], [-(1.0 - v), -(1.0 - u)]])
        return values, jacobian
    u, v, w = t
    values = np.array([u, (1.0 - u) * v, (1.0 - u) * (1.0 - v) * w, (1.0 - u) * (1.0 - v) * (1.0 - w)])
    jacobian = np.array([
        [1.0, 0.0, 0.0],
        [-v, 1.0 - u, 0.0],
        [-(1.0 - v) * w, -(1.0 - u) * w, (1.0 - u) * (1.0 - v)],
        [-(1.0 - v) * (1.0 - w), -(1.0 - u) * (1.0 - w), -(1.0 - u) * (1.0 - v)],
    ])
    return values, jacobian


def _orbit_parameters(kind: str, representative: np.ndarray) -> np.ndarray:
    """Parameters ``theta`` of :func:`_orbit_values` whose representative is ``representative`` ``(d + 1,)``.

    The representative must follow the pattern ``_ORBIT_LABELS[kind]`` (equal
    coordinates where the labels are equal). The result is clipped to the bounds of
    the type; a collapsed coordinate whose factor ``1 - u`` vanishes is set to 1/2.
    """
    r = np.asarray(representative, dtype=np.float64)

    def ratio(numerator: float, denominator: float) -> float:
        """``numerator / denominator``, or 1/2 when the denominator is below ``1e-14``."""
        return numerator / denominator if denominator > 1.0e-14 else 0.5

    if kind in ("S3", "S4"):
        theta = []
    elif kind in ("S21", "S31", "S22"):
        theta = [r[0]]
    elif kind == "S111":
        theta = [r[0], ratio(r[1], 1.0 - r[0])]
    elif kind == "S211":
        theta = [2.0 * r[0], ratio(r[2], 1.0 - 2.0 * r[0])]
    else:
        u = r[0]
        v = ratio(r[1], 1.0 - u)
        theta = [u, v, ratio(r[2], (1.0 - u) * (1.0 - v))]
    bounds = np.asarray(_ORBIT_BOUNDS[kind], dtype=np.float64).reshape(-1, 2)
    return np.clip(np.asarray(theta, dtype=np.float64), bounds[:, 0], bounds[:, 1])


@lru_cache(maxsize=None)
def _orbit_label_permutations(kind: str) -> np.ndarray:
    """Distinct permutations of the labels of ``kind``, sorted lexicographically, shape ``(orbit size, d + 1)``.

    Point ``k`` of an orbit with distinct values ``values`` is ``values[result[k]]``;
    the orbit sizes are 1, 3, 6 (triangle) and 1, 4, 6, 12, 24 (tetrahedron).
    """
    result = np.asarray(sorted(set(permutations(_ORBIT_LABELS[kind]))), dtype=np.int64)
    result.setflags(write=False)
    return result


def _orbit_points(kind: str, representative: np.ndarray) -> np.ndarray:
    """Barycentric coordinates ``(orbit size, d + 1)`` of the orbit of ``representative`` in canonical order.

    The order is that of :func:`_orbit_label_permutations`; ``representative``
    follows the pattern ``_ORBIT_LABELS[kind]``.
    """
    labels = _ORBIT_LABELS[kind]
    first = np.asarray([labels.index(value) for value in range(max(labels) + 1)], dtype=np.int64)
    return np.asarray(representative, dtype=np.float64)[first[_orbit_label_permutations(kind)]]


def _detect_orbits(dim: int, nodes: np.ndarray, free) -> list[dict]:
    r"""Split a node set invariant under :math:`S_{d+1}` into orbits.

    ``nodes`` ``(n, dim)`` are reference coordinates; ``free(representative)``
    returns whether an orbit may move. Points are grouped by their sorted
    barycentric coordinates (tolerance ``1e-9``); the type follows from the
    multiplicities of equal coordinates. Each orbit is a dict with ``"type"``,
    ``"theta"`` (start parameters), ``"nodes"`` (node indices of the canonical orbit
    points of :func:`_orbit_points`, an ``int64`` array) and ``"free"``. Raises
    ``ValueError`` when the set is not symmetric.
    """
    lam = reference_simplex(dim).barycentric(nodes)
    keys = np.sort(lam, axis=1)
    # Node i belongs to the group of the first node j whose sorted coordinates agree with its own.
    close = np.max(np.abs(keys[:, None, :] - keys[None, :, :]), axis=2) < _ORBIT_TOLERANCE
    first = np.argmax(close, axis=1)
    groups = [np.flatnonzero(first == leader).tolist() for leader in np.unique(first)]
    orbits = []
    for members in groups:
        values = keys[members[0]]
        distinct: list[list[float]] = []
        for value in values:
            if distinct and abs(value - distinct[-1][0]) < _ORBIT_TOLERANCE:
                distinct[-1].append(value)
            else:
                distinct.append([value])
        multiplicity = tuple(sorted((len(group) for group in distinct), reverse=True))
        kind = _ORBIT_TYPE_BY_MULTIPLICITY[multiplicity]
        means = [float(np.mean(group)) for group in distinct]
        counts = [len(group) for group in distinct]
        if kind in ("S21", "S31"):
            repeated = means[int(np.argmax(counts))]
            single = means[int(np.argmin(counts))]
            representative = [repeated] * (dim) + [single]
        elif kind == "S22":
            representative = [means[0], means[0], means[1], means[1]]
        elif kind == "S211":
            repeated = means[int(np.argmax(counts))]
            singles = sorted(value for value, count in zip(means, counts) if count == 1)
            representative = [repeated, repeated] + singles
        else:
            representative = sorted(values.tolist())
        theta = _orbit_parameters(kind, np.asarray(representative))
        canonical = _orbit_points(kind, _orbit_values(kind, theta)[0][list(_ORBIT_LABELS[kind])])
        if canonical.shape[0] != len(members):
            raise ValueError(
                f"the node set is not invariant under vertex permutations: an orbit of type {kind} has "
                f"{len(members)} points instead of {canonical.shape[0]}"
            )
        match = point_permutation(canonical, lam[members], atol=1.0e-8)
        orbits.append({
            "type": kind,
            "theta": theta,
            "nodes": np.asarray(members, dtype=np.int64)[match],
            "free": bool(free(np.asarray(representative))),
        })
    return orbits


# ---------------------------------------------------------------------------
# Fekete optimization
# ---------------------------------------------------------------------------


class _FeketeObjective:
    r"""Minus :math:`\log|\det V|` of a symmetric node set as a function of the free orbit parameters.

    ``orbits`` are those of :func:`_detect_orbits` for a set of ``n`` nodes of
    degree ``order`` on the ``dim``-simplex. The parameters of the free orbits are
    concatenated in orbit order into one vector :math:`\theta`; the fixed orbits keep
    their start positions. :meth:`__call__` returns the value and the exact
    gradient for ``scipy.optimize.minimize(..., jac=True)``.
    """

    def __init__(self, dim: int, order: int, orbits: list[dict], num_nodes: int) -> None:
        """Store the orbits, the barycentric coordinates of the fixed nodes and the parameter slices and bounds."""
        self.dim = dim
        self.order = order
        self.orbits = orbits
        self.num_nodes = num_nodes
        self.fixed = np.zeros((num_nodes, dim + 1), dtype=np.float64)
        self.slices: list[tuple[int, slice]] = []
        bounds: list[tuple[float, float]] = []
        start: list[float] = []
        offset = 0
        for index, orbit in enumerate(orbits):
            values, _ = _orbit_values(orbit["type"], orbit["theta"])
            if orbit["free"] and len(orbit["theta"]) > 0:
                size = len(orbit["theta"])
                self.slices.append((index, slice(offset, offset + size)))
                bounds.extend(_ORBIT_BOUNDS[orbit["type"]])
                start.extend(orbit["theta"])
                offset += size
            else:
                self.fixed[orbit["nodes"]] = values[_orbit_label_permutations(orbit["type"])]
        self.bounds = np.asarray(bounds, dtype=np.float64).reshape(-1, 2)
        self.start = np.asarray(start, dtype=np.float64)

    def barycentric(self, theta: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Barycentric coordinates ``(n, d + 1)`` of all nodes and, per free orbit, the derivatives
        ``(orbit size, d + 1, k)`` of its points in its ``k`` parameters."""
        lam = self.fixed.copy()
        derivatives = []
        for index, part in self.slices:
            orbit = self.orbits[index]
            labels = _orbit_label_permutations(orbit["type"])
            values, jacobian = _orbit_values(orbit["type"], theta[part])
            lam[orbit["nodes"]] = values[labels]
            derivatives.append(jacobian[labels])
        return lam, derivatives

    def log_det_and_point_gradient(self, lam: np.ndarray) -> tuple[float, np.ndarray]:
        r"""Return :math:`\log|\det V|` and its gradient ``(n, d)`` in the reference coordinates of the nodes.

        Row ``i`` of the gradient is :math:`\sum_j (V^{-1})_{ji} \nabla\psi_j(x_i)`, the
        diagonal of :math:`\nabla\psi\, V^{-1}`, obtained by one LU solve with
        ``n * d`` right-hand sides. Returns ``(-inf, zeros)`` when ``V`` is singular.
        """
        x = reference_simplex(self.dim).from_barycentric(lam)
        matrix = _orthonormal_values(self.dim, self.order, x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)
            lu, piv = scipy.linalg.lu_factor(matrix, check_finite=False)
        diagonal = np.abs(np.diag(lu))
        n, d = x.shape
        if not np.all(np.isfinite(diagonal)) or np.any(diagonal == 0.0):
            return -np.inf, np.zeros((n, d))
        gradients = _orthonormal_gradients(self.dim, self.order, x)
        rhs = gradients.transpose(1, 0, 2).reshape(n, n * d)
        solved = scipy.linalg.lu_solve((lu, piv), rhs, trans=1, check_finite=False).reshape(n, n, d)
        rows = np.arange(n)
        return float(np.log(diagonal).sum()), solved[rows, rows, :]

    def gradient_from_points(self, point_gradient: np.ndarray, derivatives: list[np.ndarray]) -> np.ndarray:
        r"""Chain the gradient ``(n, d)`` in the reference coordinates to the parameter vector.

        :math:`\xi_k = 2\lambda_k - 1` gives :math:`\partial/\partial\lambda_k = 2\,\partial/\partial\xi_k`
        for ``k >= 1`` (the parametrizations keep :math:`\sum_k \lambda_k = 1`).
        """
        result = np.zeros(self.start.size, dtype=np.float64)
        for (index, part), derivative in zip(self.slices, derivatives):
            nodes = self.orbits[index]["nodes"]
            result[part] = np.einsum("mk,mkt->t", 2.0 * point_gradient[nodes], derivative[:, 1:, :])
        return result

    def __call__(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        r"""Value :math:`-\log|\det V|` and gradient at the parameters ``theta``.

        Returns ``1e300`` and a zero gradient when ``V`` is singular (coinciding nodes).
        """
        lam, derivatives = self.barycentric(theta)
        value, point_gradient = self.log_det_and_point_gradient(lam)
        if not np.isfinite(value):
            return 1.0e300, np.zeros_like(theta)
        return -value, -self.gradient_from_points(point_gradient, derivatives)

    def projected_gradient(self, theta: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        r"""Projected gradient :math:`\theta - \mathrm{clip}(\theta - g, l, u)` of the problem with bounds ``l, u``.

        Its maximum norm vanishes exactly at the stationary points (Karush-Kuhn-Tucker
        points) of the bound-constrained problem.
        """
        if theta.size == 0:
            return np.zeros(0)
        return theta - np.clip(theta - gradient, self.bounds[:, 0], self.bounds[:, 1])


def _newton_polish(
    objective: _FeketeObjective, theta: np.ndarray, target: float, max_steps: int = 20
) -> tuple[np.ndarray, int]:
    r"""Newton steps on the parameters not held at a bound, until the projected gradient is below ``target``.

    The Hessian of the free parameters is formed by differences of the exact
    gradient at ``theta +- 1e-6`` (one-sided at a bound, so that no node leaves the
    simplex); each step solves with it (shifted to be
    positive definite when needed), is clipped to the bounds and is halved until it
    is accepted. A step is accepted when it lowers :math:`-\log|\det V|`, or when the
    change of :math:`-\log|\det V|` is within rounding (``1e-13`` relative) and the
    projected gradient decreases: near the maximum the value no longer tells the
    steps apart. Returns the parameters and the number of accepted steps.
    """
    accepted = 0
    value, gradient = objective(theta)
    for _ in range(max_steps):
        projected = objective.projected_gradient(theta, gradient)
        norm = np.max(np.abs(projected), initial=0.0)
        if norm <= target:
            break
        lower, upper = objective.bounds[:, 0], objective.bounds[:, 1]
        held = ((theta <= lower + 1.0e-14) & (gradient > 0.0)) | ((theta >= upper - 1.0e-14) & (gradient < 0.0))
        free = np.flatnonzero(~held)
        if free.size == 0:
            break
        hessian = np.empty((free.size, free.size), dtype=np.float64)
        for column, index in enumerate(free):
            plus, minus = theta.copy(), theta.copy()
            plus[index] = min(theta[index] + 1.0e-6, upper[index])
            minus[index] = max(theta[index] - 1.0e-6, lower[index])
            width = plus[index] - minus[index]
            hessian[:, column] = (objective(plus)[1][free] - objective(minus)[1][free]) / width
        hessian = 0.5 * (hessian + hessian.T)
        smallest = np.linalg.eigvalsh(hessian)[0]
        if smallest <= 0.0:
            hessian = hessian + (1.0e-8 - smallest) * np.eye(free.size)
        direction = np.zeros_like(theta)
        direction[free] = -np.linalg.solve(hessian, gradient[free])
        length = 1.0
        improved = False
        for _ in range(30):
            trial = np.clip(theta + length * direction, lower, upper)
            trial_value, trial_gradient = objective(trial)
            trial_norm = np.max(np.abs(objective.projected_gradient(trial, trial_gradient)), initial=0.0)
            within_rounding = abs(trial_value - value) <= 1.0e-13 * max(1.0, abs(value))
            if trial_value < value - 1.0e-13 * max(1.0, abs(value)) or (within_rounding and trial_norm < norm):
                theta, value, gradient = trial, trial_value, trial_gradient
                improved = True
                accepted += 1
                break
            length *= 0.5
        if not improved:
            break
    return theta, accepted


def _optimize_start(dim: int, order: int, nodes: np.ndarray, free) -> dict:
    r"""Maximize :math:`\log|\det V|` over the free symmetric orbits of the start set ``nodes`` ``(n_u, dim)``.

    ``free(representative)`` selects the orbits that move (see
    :func:`_detect_orbits`). L-BFGS-B runs on the objective scaled so that its start
    gradient has maximum norm 0.01, with ``ftol = 0`` and a projected-gradient
    tolerance of ``1e-13 max(1, |log det V|)`` before scaling (at most 20000
    iterations); if the projected gradient then exceeds ``1e-12 max(1, |log det V|)``,
    :func:`_newton_polish` continues. Returns a dict
    with the orbits (``"type"``, ``"barycentric"`` representative,
    ``"nodes"``, ``"free"``), ``"nodes_array"`` (the optimized nodes, reference
    coordinates), ``"log_abs_det"``, ``"start_log_abs_det"``,
    ``"projected_gradient_norm"`` (maximum norm, in the parameters) and
    ``"optimizer"`` (scipy status, message, iterations, evaluations, Newton steps).
    """
    orbits = _detect_orbits(dim, nodes, free)
    objective = _FeketeObjective(dim, order, orbits, nodes.shape[0])
    start_value, _ = objective(objective.start)
    theta = objective.start
    optimizer = {"status": 0, "message": "no free parameters", "iterations": 0, "evaluations": 1, "newton_steps": 0}
    if theta.size:
        _, start_gradient = objective(theta)
        scale = 1.0e-2 / max(float(np.max(np.abs(start_gradient))), 1.0e-12)

        def scaled_objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            """``scale`` times the value and gradient of ``objective`` at ``parameters``.

            The first L-BFGS-B step moves by minus the gradient; with this scaling it
            changes no parameter by more than 0.01, instead of jumping to a corner of
            the box where nodes coincide.
            """
            value, gradient = objective(parameters)
            return scale * value, scale * gradient

        result = scipy.optimize.minimize(
            scaled_objective, theta, jac=True, method="L-BFGS-B", bounds=objective.bounds,
            options={"ftol": 0.0, "gtol": 1.0e-13 * scale * max(1.0, abs(start_value)), "maxiter": 20000,
                     "maxfun": 100000, "maxcor": 30, "maxls": 50},
        )
        theta = np.asarray(result.x, dtype=np.float64)
        optimizer = {
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "evaluations": int(result.nfev),
            "newton_steps": 0,
        }
        value, gradient = objective(theta)
        target = 1.0e-12 * max(1.0, abs(value))
        if np.max(np.abs(objective.projected_gradient(theta, gradient))) > target:
            theta, optimizer["newton_steps"] = _newton_polish(objective, theta, target)
    value, gradient = objective(theta)
    lam, _ = objective.barycentric(theta)
    parts = dict(objective.slices)
    stored = []
    for index, orbit in enumerate(orbits):
        parameters = theta[parts[index]] if index in parts else orbit["theta"]
        values, _ = _orbit_values(orbit["type"], parameters)
        stored.append({
            "type": orbit["type"],
            "barycentric": values[list(_ORBIT_LABELS[orbit["type"]])],
            "nodes": orbit["nodes"],
            "free": orbit["free"],
        })
    return {
        "orbits": stored,
        "nodes_array": reference_simplex(dim).from_barycentric(lam),
        "log_abs_det": -float(value),
        "start_log_abs_det": -float(start_value),
        "projected_gradient_norm": float(np.max(np.abs(objective.projected_gradient(theta, gradient)), initial=0.0)),
        "optimizer": optimizer,
    }


def _face_compatible_start(dim: int, order: int, interior_family: str, triangle_entry: dict | None) -> np.ndarray:
    """Start set of ``"fekete"``: GLL edges, (``dim = 3``) the triangle ``"fekete"`` faces of ``triangle_entry``
    mapped by the face maps, and the interior nodes of ``interior_family`` (``"warp_blend"`` or ``"recursive"``).

    Returns reference coordinates ``(C(order + dim, dim), dim)`` in the node order of the module docstring.
    """
    simplex = reference_simplex(dim)
    interior = simplex.from_barycentric(_interior_barycentric(interior_family, dim, order))
    faces = None
    if dim == 3:
        faces = _nodes_from_entry(triangle_entry)[3 * order :]
    return _assemble_nodes(dim, order, _edge_parameters("fekete", order), faces, interior)


def _interior_orbit(representative: np.ndarray) -> bool:
    """Whether every barycentric coordinate of ``representative`` exceeds ``1e-9`` (an orbit inside the simplex)."""
    return bool(np.all(np.asarray(representative) > _ORBIT_TOLERANCE))


def _any_orbit(representative: np.ndarray) -> bool:
    """Return ``True`` for every orbit: in ``"fekete_free"`` all orbits move, boundary orbits included."""
    return True


def compute_fekete_entry(dim, order, variant, *, triangle_entry=None, fekete_entry=None) -> dict:
    r"""Compute one Fekete node set; return it in the format of the entries of ``data/fekete.json``.

    ``variant`` is ``"fekete"`` or ``"fekete_free"``, ``dim`` is 2 or 3 and
    ``order >= 1``.

    - ``"fekete"`` starts from the face-compatible sets of
      :func:`_face_compatible_start` with the interior nodes of ``"warp_blend"`` and
      of ``"recursive"``, moves only the interior orbits and keeps the start with the
      larger final determinant. For ``dim = 3`` the faces come from
      ``triangle_entry`` (the 2D ``"fekete"`` entry of the same degree; by default
      the stored or computed one).
    - ``"fekete_free"`` starts from the ``"fekete"`` set (``fekete_entry``; by
      default the stored or computed one) and from ``"warp_blend"``, moves every
      orbit and keeps the larger final determinant.

    The returned dict holds ``dim``, ``order``, ``variant``, ``start`` (the name of
    the start set kept), ``orbits`` (each with ``type``, ``barycentric``
    representative of ``d + 1`` floats, ``nodes`` (indices of the canonical orbit
    points) and ``free``), ``log_abs_det``, ``start_log_abs_det``,
    ``projected_gradient_norm``, ``lebesgue_estimate``, ``optimizer`` and
    ``seconds``; arrays are converted to lists.
    """
    d = int(dim)
    p = int(order)
    if variant not in FEKETE_VARIANTS or d not in (2, 3) or p < 1:
        raise ValueError(
            f"expected variant in {FEKETE_VARIANTS}, dim 2 or 3 and order >= 1; got {variant!r}, {dim!r}, {order!r}"
        )
    started = time.perf_counter()
    candidates = []
    if variant == "fekete":
        if d == 3 and triangle_entry is None:
            triangle_entry = _fekete_entry(2, p, "fekete")
        for interior_family in ("warp_blend", "recursive"):
            start = _face_compatible_start(d, p, interior_family, triangle_entry)
            candidates.append((f"{interior_family} interior", _optimize_start(d, p, start, _interior_orbit)))
    else:
        if fekete_entry is None:
            fekete_entry = _fekete_entry(d, p, "fekete")
        starts = (
            ("fekete", _nodes_from_entry(fekete_entry)),
            ("warp_blend", _lattice_family_nodes("warp_blend", d, p)),
        )
        for name, start in starts:
            candidates.append((name, _optimize_start(d, p, start, _any_orbit)))
    name, best = max(candidates, key=lambda item: item[1]["log_abs_det"])
    entry = {
        "dim": d,
        "order": p,
        "variant": variant,
        "start": name,
        "orbits": [
            {"type": o["type"], "barycentric": [float(v) for v in o["barycentric"]],
             "nodes": [int(i) for i in o["nodes"]], "free": bool(o["free"])}
            for o in best["orbits"]
        ],
        "log_abs_det": best["log_abs_det"],
        "start_log_abs_det": best["start_log_abs_det"],
        "other_starts": {other: c["log_abs_det"] for other, c in candidates if other != name},
        "projected_gradient_norm": best["projected_gradient_norm"],
        "optimizer": best["optimizer"],
    }
    entry["lebesgue_estimate"] = _lebesgue_estimate(d, p, _nodes_from_entry(entry))
    entry["seconds"] = time.perf_counter() - started
    return entry


def _nodes_from_entry(entry: dict) -> np.ndarray:
    """Reference coordinates ``(C(p + d, d), d)`` of the nodes of a Fekete entry (orbit representatives and indices)."""
    dim, order = int(entry["dim"]), int(entry["order"])
    lam = np.full((comb(order + dim, dim), dim + 1), np.nan, dtype=np.float64)
    for orbit in entry["orbits"]:
        lam[np.asarray(orbit["nodes"], dtype=np.int64)] = _orbit_points(orbit["type"], np.asarray(orbit["barycentric"]))
    if np.isnan(lam).any():  # pragma: no cover - table invariant
        raise ValueError(f"the Fekete entry of dimension {dim} and degree {order} does not cover every node")
    return reference_simplex(dim).from_barycentric(lam)


@lru_cache(maxsize=1)
def fekete_table() -> dict:
    """Contents of the package file ``hybridge/core/simplex/data/fekete.json`` (a dict), read once.

    Keys: ``"description"``, ``"generator"`` (provenance) and ``"entries"``, a dict
    from ``"<variant>/d<dim>/p<order>"`` to the entries of :func:`compute_fekete_entry`.
    Returns ``{"entries": {}}`` when the file is absent.
    """
    try:
        text = resources.files("hybridge.core.simplex").joinpath("data/fekete.json").read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"entries": {}}
    return json.loads(text)


def _fekete_entry(dim: int, order: int, variant: str) -> dict:
    """Fekete entry of the package table, or one computed now and kept in memory (with a ``RuntimeWarning``)."""
    key = f"{variant}/d{dim}/p{order}"
    stored = fekete_table()["entries"].get(key)
    if stored is not None:
        return stored
    with _COMPUTED_FEKETE_LOCK:
        cached = _COMPUTED_FEKETE.get((variant, dim, order))
    if cached is not None:
        return cached
    started = time.perf_counter()
    entry = compute_fekete_entry(dim, order, variant)
    elapsed = time.perf_counter() - started
    warnings.warn(
        f"the {variant!r} nodes of degree {order} on the {('triangle', 'tetrahedron')[dim - 2]} are not in "
        f"data/fekete.json; they were computed in {elapsed:.1f} s and are kept in memory for this process",
        RuntimeWarning,
        stacklevel=4,
    )
    with _COMPUTED_FEKETE_LOCK:
        _COMPUTED_FEKETE[(variant, dim, order)] = entry
    return entry


def _source_sha256() -> str:
    """SHA-256 of the source text of this module, recorded with a generated Fekete table."""
    return hashlib.sha256(resources.files("hybridge.core.simplex").joinpath("nodes.py").read_bytes()).hexdigest()


__all__ = [
    "FEKETE_TABLE_MAX_ORDER",
    "FEKETE_VARIANTS",
    "NODE_FAMILIES",
    "compute_fekete_entry",
    "fekete_table",
    "lagrange_basis",
    "lagrange_gradients",
    "lebesgue_constant",
    "node_entity_counts",
    "simplex_nodes",
    "vandermonde",
]
