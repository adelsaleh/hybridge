r"""Reference simplices of dimension 1, 2 and 3 and their quadrature rules.

This module is the low-level home of the reference-element geometry of
:mod:`hybridge`. It imports nothing from the rest of :mod:`hybridge.core`;
:mod:`hybridge.core.quadrature` and :mod:`hybridge.core.space` build on it.

The reference triangle is
:math:`\hat K = \operatorname{conv}\{(-1,-1), (1,-1), (-1,1)\}`. The triangle
rules below (tabulated Dunavant rules, the collapsed Gauss-Legendre rule and
the one-dimensional Gauss and Gauss-Lobatto rules) were moved here unchanged
from :mod:`hybridge.core.quadrature`, which still exposes them under their
original names.

The second part of the module describes the reference simplices of dimension
``d = 1, 2, 3`` (:func:`reference_simplex`): the interval ``[-1, 1]``, the
triangle above and the tetrahedron
:math:`\operatorname{conv}\{(-1,-1,-1), (1,-1,-1), (-1,1,-1), (-1,-1,1)\}`,
with their faces, face maps and unit outward normals, the permutations of the
vertices of a face (:func:`face_permutations`, :func:`permute_face_points`) and
quadrature rules exact for a given total degree
(:func:`gauss_jacobi_collapsed_rule`, :func:`simplex_volume_rule`,
:func:`simplex_face_rule`). The reference simplex of every dimension has the
vertex ``(-1, ..., -1)`` and the vertices ``-1 + 2 e_k``, and measure ``2^d / d!``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import permutations
from math import comb, factorial

import numpy as np

from hybridge.runtime.precision import REAL_DTYPE


# Positive-weight Dunavant rules on the unit triangle.  Each entry is
# ``(barycentric representative, weight per orbit point)``; permutations are
# expanded below and weights are scaled by two for our area-two reference
# triangle.
_DUNAVANT_RULE_BLOCKS = {
    2: (((2 / 3, 1 / 6, 1 / 6), 1 / 3),),
    4: (
        ((0.108103018168070, 0.445948490915965, 0.445948490915965), 0.223381589678011),
        ((0.816847572980459, 0.091576213509771, 0.091576213509771), 0.109951743655322),
    ),
    6: (
        ((0.501426509658179, 0.249286745170910, 0.249286745170910), 0.116786275726379),
        ((0.873821971016996, 0.063089014491502, 0.063089014491502), 0.050844906370207),
        ((0.053145049844816, 0.310352451033785, 0.636502499121399), 0.082851075618374),
    ),
    8: (
        ((1 / 3, 1 / 3, 1 / 3), 0.144315607677787),
        ((0.081414823414554, 0.459292588292723, 0.459292588292723), 0.095091634267285),
        ((0.658861384496480, 0.170569307751760, 0.170569307751760), 0.103217370534718),
        ((0.898905543365938, 0.050547228317031, 0.050547228317031), 0.032458497623198),
        ((0.008394777409958, 0.263112829634638, 0.728492392955404), 0.027230314174435),
    ),
    10: (
        ((1 / 3, 1 / 3, 1 / 3), 0.090817990382754),
        ((0.028844733232685, 0.485577633383657, 0.485577633383657), 0.036725957756467),
        ((0.781036849029926, 0.109481575485037, 0.109481575485037), 0.045321059435528),
        ((0.141707219414880, 0.307939838764121, 0.550352941820999), 0.072757916845420),
        ((0.025003534762686, 0.246672560639903, 0.728323904597411), 0.028327242531057),
        ((0.009540815400299, 0.066803251012200, 0.923655933587500), 0.009421666963733),
    ),
    12: (
        ((0.023565220452390, 0.488217389773805, 0.488217389773805), 0.025731066440455),
        ((0.120551215411080, 0.439724392294460, 0.439724392294460), 0.043692544538038),
        ((0.457579229975768, 0.271210385012116, 0.271210385012116), 0.062858224217885),
        ((0.744847708916828, 0.127576145541586, 0.127576145541586), 0.034796112930709),
        ((0.957365299093580, 0.021317350453210, 0.021317350453210), 0.006166261051559),
        ((0.115343494534698, 0.275713269685514, 0.608943235779788), 0.040371557766381),
        ((0.022838332222257, 0.281325580989940, 0.695836086787803), 0.022356773202303),
        ((0.025734050548330, 0.116251915907597, 0.858014033544073), 0.017316231108659),
    ),
    14: (
        ((0.022072179275643, 0.488963910362179, 0.488963910362179), 0.021883581369429),
        ((0.164710561319092, 0.417644719340454, 0.417644719340454), 0.032788353544125),
        ((0.453044943382323, 0.273477528308839, 0.273477528308839), 0.051774104507292),
        ((0.645588935174913, 0.177205532412543, 0.177205532412543), 0.042162588736993),
        ((0.876400233818255, 0.061799883090873, 0.061799883090873), 0.014433699669777),
        ((0.961218077502598, 0.019390961248701, 0.019390961248701), 0.004923403602400),
        ((0.057124757403648, 0.172266687821356, 0.770608554774996), 0.024665753212564),
        ((0.092916249356972, 0.336861459796345, 0.570222290846683), 0.038571510787061),
        ((0.014646950055654, 0.298372882136258, 0.686980167808088), 0.014436308113534),
        ((0.001268330932872, 0.118974497696957, 0.879757171370171), 0.005010228838501),
    ),
}


def _duffy_points_for_degree(exact_degree: int) -> int:
    """Return the 1D Gauss count making the collapsed rule exact for ``P_exact_degree``.

    The Duffy Jacobian adds one degree in the collapsed direction, so ``n``
    points are exact when ``2n - 1 >= exact_degree + 1``.
    """
    return (int(exact_degree) + 3) // 2


def _compact_dunavant_degree(exact_degree: int) -> int | None:
    """Return the smallest tabulated admissible Dunavant degree >= ``exact_degree``."""
    degrees = [degree for degree in sorted(_DUNAVANT_RULE_BLOCKS) if degree >= exact_degree]
    return degrees[0] if degrees else None


def _symmetric_triangle_quadrature(
        order: int, exact_degree: int | None = None, *, dtype=REAL_DTYPE
) -> tuple[np.ndarray, np.ndarray]:
    """Return a positive-weight symmetric rule exact for ``P_exact_degree`` (default ``2*order``).

    The smallest tabulated compact Dunavant rule of at least that degree is
    used when available (even degrees up to 14).  Otherwise a minimally exact
    Duffy rule is expanded into complete barycentric permutation orbits.
    Orbit symmetrization preserves the original rule's polynomial exactness
    and positivity while providing a fully symmetric rule for any degree.
    The arrays are computed and returned in ``dtype`` (default ``REAL_DTYPE``);
    the reference-simplex tables pass ``float64``.
    """
    exact_degree = max(2, 2 * order) if exact_degree is None else max(1, int(exact_degree))
    compact = _compact_dunavant_degree(exact_degree)
    blocks = None if compact is None else _DUNAVANT_RULE_BLOCKS[compact]
    if blocks is None:
        points, base_weights = _triangle_quadrature(order, _duffy_points_for_degree(exact_degree), dtype=dtype)
        barycentric = np.column_stack(
            (
                -0.5 * (points[:, 0] + points[:, 1]),
                0.5 * (points[:, 0] + 1.0),
                0.5 * (points[:, 1] + 1.0),
            )
        )
        orbit_points = []
        orbit_weights = []
        for bary, weight in zip(barycentric, base_weights):
            orbit = sorted(set(permutations(map(float, bary))))
            orbit = np.asarray(orbit, dtype=dtype)
            orbit_points.append(
                np.column_stack((-1.0 + 2.0 * orbit[:, 1], -1.0 + 2.0 * orbit[:, 2]))
            )
            orbit_weights.append(np.full(len(orbit), weight / len(orbit), dtype=dtype))
        return (
            np.ascontiguousarray(np.vstack(orbit_points)),
            np.ascontiguousarray(np.concatenate(orbit_weights)),
        )

    barycentric = []
    weights = []
    for bary, weight in blocks:
        orbit = sorted(set(permutations(map(float, bary))))
        barycentric.extend(orbit)
        weights.extend([2.0 * weight] * len(orbit))
    barycentric = np.asarray(barycentric, dtype=dtype)
    points = np.column_stack((-1.0 + 2.0 * barycentric[:, 1], -1.0 + 2.0 * barycentric[:, 2]))
    return np.ascontiguousarray(points), np.ascontiguousarray(weights, dtype=dtype)


def _quadrature_point_count(order: int, num_1d: int | None) -> int:
    """Resolve and validate the one-dimensional quadrature point count."""
    if num_1d is None:
        return max(2 * order + 2, 2)
    count = int(num_1d)
    if count < 1:
        raise ValueError("quadrature point count must be positive")
    return count


def _triangle_quadrature(
        order: int, num_1d: int | None = None, *, dtype=REAL_DTYPE
) -> tuple[np.ndarray, np.ndarray]:
    """Build a collapsed tensor-product Gauss rule on the reference triangle.

    A tensor Gauss-Legendre rule on ``[-1, 1]^2`` is mapped to the reference
    triangle with the Duffy-style transform

    ``x = -1 + 2*u*(1-v)``, ``y = -1 + 2*u*v``.

    The returned points have shape ``(num_quads, 2)`` and weights have shape
    ``(num_quads,)``.  The weights integrate on the reference triangle only;
    callers multiply by each physical element Jacobian when assembling. Both are
    returned in ``dtype`` (default ``REAL_DTYPE``).
    """
    num_1d = _quadrature_point_count(order, num_1d)
    s, ws = np.polynomial.legendre.leggauss(num_1d)
    t, wt = np.polynomial.legendre.leggauss(num_1d)
    ss, tt = np.meshgrid(s, t, indexing="xy")
    u = 0.5 * (ss + 1.0)
    v = 0.5 * (tt + 1.0)
    x = -1.0 + 2.0 * u * (1.0 - v)
    y = -1.0 + 2.0 * u * v
    weights = (ws[None, :] * wt[:, None] * u).ravel()
    points = np.stack((x.ravel(), y.ravel()), axis=1)
    return np.ascontiguousarray(points, dtype=dtype), np.ascontiguousarray(weights, dtype=dtype)


def _edge_quadrature(order: int, num_1d: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build a one-dimensional Gauss rule for reference edges.

    The same rule is used on all three reference faces.  Points and weights are
    returned in the canonical interval coordinate ``t in [-1, 1]`` with shape
    ``(num_face_quads,)``.
    """
    num_1d = _quadrature_point_count(order, num_1d)
    points, weights = np.polynomial.legendre.leggauss(num_1d)
    return np.ascontiguousarray(points, dtype=REAL_DTYPE), np.ascontiguousarray(weights, dtype=REAL_DTYPE)


def _legendre_gauss_lobatto(num_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Legendre-Gauss-Lobatto nodes and weights on ``[-1, 1]``."""
    if num_points < 1:
        raise ValueError("Gauss-Lobatto rule needs at least one point")
    if num_points == 1:
        return np.array([0.0]), np.array([2.0])
    if num_points == 2:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    poly = np.polynomial.legendre.Legendre.basis(num_points - 1)
    roots = np.real_if_close(poly.deriv().roots(), tol=1000)
    if np.iscomplexobj(roots):
        raise ArithmeticError("Legendre derivative produced non-real Gauss-Lobatto nodes")
    interior = np.sort(np.asarray(roots, dtype=REAL_DTYPE))
    points = np.concatenate(([-1.0], interior, [1.0]))
    values = poly(points)
    weights = 2.0 / ((num_points - 1) * num_points * values * values)
    return np.ascontiguousarray(points, dtype=REAL_DTYPE), np.ascontiguousarray(weights, dtype=REAL_DTYPE)


def triangle_edge_points(edge_points_1d: np.ndarray) -> np.ndarray:
    """Map points ``t`` of ``[-1, 1]`` onto the three edges of the reference triangle.

    Returns an array of shape ``(num_points, 3, 2)``. Edge ``f`` runs from vertex ``f``
    to vertex ``(f + 1) mod 3`` of ``(-1,-1), (1,-1), (-1,1)``, with ``t = -1`` at the
    first vertex:

    ``0``: bottom edge ``(t, -1)``; ``1``: diagonal edge ``(-t, t)``; ``2``: left edge
    ``(-1, -t)``.

    Evaluating at ``-t`` gives the same edges traversed in the opposite direction,
    which is how the reversed edge orientation is tabulated. The input is converted
    to ``REAL_DTYPE``. This one function replaces three identical copies formerly in
    :mod:`hybridge.core.quadrature` (``_edge_points``, ``_reference_edge_points_from_1d``)
    and :mod:`hybridge.core.space` (``_reference_edge_points``), which remain as aliases.
    """
    t = np.asarray(edge_points_1d, dtype=REAL_DTYPE)
    return np.ascontiguousarray(
        np.stack(
            (
                np.stack((t, -np.ones_like(t)), axis=1),
                np.stack((-t, t), axis=1),
                np.stack((-np.ones_like(t), -t), axis=1),
            ),
            axis=1,
        )
    )


def monomial_integral(exponents) -> Fraction:
    r"""Return the exact integral of :math:`\prod_k \xi_k^{a_k}` over the reference simplex.

    ``exponents`` holds the nonnegative powers ``(a_1, ..., a_d)`` of the reference
    coordinates; its length ``d`` (1, 2 or 3) selects the interval ``[-1, 1]``, the
    reference triangle or the reference tetrahedron, all with vertex ``(-1, ..., -1)``
    and the unit vectors shifted by ``-1``. Each coordinate is written as
    :math:`\xi_k = 2\lambda_k - 1` in barycentric coordinates and the integral is
    expanded with the Dirichlet formula

    .. math::

        \int_{\hat K} \lambda_1^{b_1}\cdots\lambda_d^{b_d}\,d\xi
        = 2^d \frac{b_1!\cdots b_d!}{(b_1+\cdots+b_d+d)!},

    in which :math:`2^d/d!` is the measure of the reference simplex. The result is a
    :class:`fractions.Fraction`, so it carries no rounding error.
    """
    powers = tuple(int(a) for a in exponents)
    if not powers or any(a < 0 for a in powers):
        raise ValueError(f"exponents must be a nonempty tuple of nonnegative integers; got {exponents!r}")
    dim = len(powers)
    expansions = [
        [(b, comb(a, b) * (-1) ** (a - b) * 2 ** b) for b in range(a + 1)]
        for a in powers
    ]
    total = Fraction(0)

    def accumulate(axis: int, coefficient: int, chosen: tuple[int, ...]) -> None:
        """Add the terms of the binomial expansion of coordinates ``axis, ..., d-1``."""
        nonlocal total
        if axis == dim:
            numerator = 1
            for b in chosen:
                numerator *= factorial(b)
            total += Fraction(coefficient * 2 ** dim * numerator, factorial(sum(chosen) + dim))
            return
        for b, c in expansions[axis]:
            accumulate(axis + 1, coefficient * c, chosen + (b,))

    accumulate(0, 1, ())
    return total


# ---------------------------------------------------------------------------
# Reference simplices of dimension 1, 2 and 3
# ---------------------------------------------------------------------------

_SIMPLEX_VERTICES = {
    1: ((-1.0,), (1.0,)),
    2: ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)),
    3: ((-1.0, -1.0, -1.0), (1.0, -1.0, -1.0), (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0)),
}
# Triangle faces keep the legacy numbering (face f joins vertex f to vertex
# (f + 1) mod 3); tetrahedron face f is opposite vertex f, vertices increasing.
_SIMPLEX_FACE_VERTICES = {
    1: ((0,), (1,)),
    2: ((0, 1), (1, 2), (2, 0)),
    3: ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)),
}
_SIMPLEX_EDGES = {
    1: ((0, 1),),
    2: ((0, 1), (1, 2), (2, 0)),
    3: ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)),
}


def _check_dimension(dim) -> int:
    """Return ``dim`` as an ``int``; raise ``ValueError`` unless it is 1, 2 or 3."""
    value = int(dim)
    if value != dim or value not in _SIMPLEX_VERTICES:
        raise ValueError(f"the reference simplex has dimension 1, 2 or 3; got {dim!r}")
    return value


def _check_exact_degree(exact_degree) -> int:
    """Return ``exact_degree`` as an ``int``; raise ``ValueError`` if it is not an integer >= 0."""
    value = int(exact_degree)
    if value != exact_degree or value < 0:
        raise ValueError(f"exact_degree must be an integer >= 0; got {exact_degree!r}")
    return value


def _point_array(points, num_coordinates: int, name: str) -> np.ndarray:
    """Return ``points`` as a ``float64`` array of shape ``(n, num_coordinates)``.

    Raises ``ValueError`` when the array is not two-dimensional with
    ``num_coordinates`` columns (``num_coordinates = 0`` is allowed: the points of
    a zero-dimensional face have no coordinates).
    """
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != num_coordinates:
        raise ValueError(f"{name} must have shape (n, {num_coordinates}); got shape {array.shape}")
    return array


def _barycentric_coordinates(points: np.ndarray) -> np.ndarray:
    r"""Barycentric coordinates in the reference ``m``-simplex, for any ``m >= 0``.

    ``points`` has shape ``(n, m)``. The result has shape ``(n, m + 1)`` with
    :math:`\lambda_k = (1 + \xi_k)/2` for ``k = 1, ..., m`` and
    :math:`\lambda_0 = 1 - \sum_{k \ge 1} \lambda_k`; column ``k`` belongs to vertex
    ``k`` (vertex 0 is ``(-1, ..., -1)``, vertex ``k`` is ``-1 + 2 e_k``). For
    ``m = 0`` the result is a column of ones.
    """
    tail = 0.5 * (1.0 + points)
    head = 1.0 - tail.sum(axis=1, keepdims=True)
    return np.ascontiguousarray(np.concatenate((head, tail), axis=1))


def _coordinates_from_barycentric(lam: np.ndarray) -> np.ndarray:
    r"""Reference coordinates :math:`\xi_k = 2\lambda_k - 1`, ``k = 1, ..., m``, from ``lam`` of shape ``(n, m + 1)``.

    Column 0 (:math:`\lambda_0`) is not read; it is determined by the others when
    the coordinates sum to one. The result has shape ``(n, m)``.
    """
    return np.ascontiguousarray(2.0 * lam[:, 1:] - 1.0)


@dataclass(frozen=True, eq=False)
class ReferenceSimplex:
    r"""Geometry of the reference simplex :math:`\hat K` of dimension ``dim`` (1, 2 or 3).

    Obtain instances from :func:`reference_simplex`; they are shared, and their
    arrays are read-only ``float64`` arrays.

    - ``vertices`` ``(d + 1, d)``: vertex 0 is ``(-1, ..., -1)`` and vertex ``k``
      is ``-1 + 2 e_k``. ``measure`` is :math:`|\hat K| = 2^d / d!`.
    - ``face_vertices``: for each of the ``d + 1`` faces, the indices of its ``d``
      vertices, in the order used by :meth:`face_map`. Interval:
      ``((0,), (1,))``. Triangle: ``((0, 1), (1, 2), (2, 0))``, face ``f`` joins
      vertex ``f`` to vertex ``(f + 1) mod 3`` (bottom, diagonal and left edge).
      Tetrahedron: face ``f`` is opposite vertex ``f``, vertices increasing,
      ``((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))``.
    - ``edges``: the vertex pairs of the edges. Interval ``((0, 1),)``; triangle
      the same pairs as its faces; tetrahedron
      ``((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))``.
    - ``face_measure_ratios`` ``(d + 1,)``: :math:`|F_f| / |\hat F|`, where the
      reference face :math:`\hat F` is the reference simplex of dimension
      ``d - 1`` (measure :math:`2^{d-1}/(d-1)!`; for ``d = 1`` a face is a point
      and the ratio is 1). Triangle ``(1, sqrt 2, 1)``, tetrahedron
      ``(sqrt 3, 1, 1, 1)``. It equals :math:`\sqrt{\det(J_f^T J_f)}` with
      :math:`J_f` from :meth:`face_jacobian`.
    - ``outward_normals`` ``(d + 1, d)``: unit outward normal :math:`n_f` of face
      ``f``, equal to :math:`-\nabla\lambda_v / |\nabla\lambda_v|` for the vertex
      ``v`` not on the face. The divergence theorem gives
      :math:`\sum_f r_f n_f = 0` for the ratios :math:`r_f`.
    """

    dim: int
    vertices: np.ndarray
    measure: float
    face_vertices: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, int], ...]
    face_measure_ratios: np.ndarray
    outward_normals: np.ndarray

    def _face_index(self, f) -> int:
        """Return ``f`` as an ``int``; raise ``ValueError`` unless ``0 <= f <= dim``."""
        index = int(f)
        if index != f or not 0 <= index <= self.dim:
            raise ValueError(f"face index must be in 0..{self.dim}; got {f!r}")
        return index

    def barycentric(self, points) -> np.ndarray:
        r"""Barycentric coordinates of ``points`` (shape ``(n, d)``), shape ``(n, d + 1)``.

        :math:`\lambda_k = (1 + \xi_k)/2` for ``k = 1, ..., d`` and
        :math:`\lambda_0 = 1 - \sum_{k \ge 1} \lambda_k`; column ``k`` belongs to
        ``vertices[k]``. All coordinates are positive if and only if the point lies
        in the interior of the simplex.
        """
        return _barycentric_coordinates(_point_array(points, self.dim, "points"))

    def from_barycentric(self, lam) -> np.ndarray:
        r"""Reference coordinates :math:`\xi_k = 2\lambda_k - 1` from barycentric coordinates ``lam`` ``(n, d + 1)``.

        Returns shape ``(n, d)``. Column 0 of ``lam`` is not read: the result is the
        point :math:`\sum_k \lambda_k v_k` when ``lam`` sums to one.
        """
        return _coordinates_from_barycentric(_point_array(lam, self.dim + 1, "lam"))

    def face_map(self, f, face_points) -> np.ndarray:
        r"""Map points of the reference face :math:`\hat F` onto face ``f`` of the simplex.

        ``face_points`` has shape ``(n, d - 1)`` (shape ``(n, 0)`` for the
        interval). The image is

        .. math::

            x(\hat s) = \sum_{i=0}^{d-1} v_{g_i}\,\hat\lambda_i(\hat s),

        where ``(g_0, ..., g_{d-1}) = face_vertices[f]`` and :math:`\hat\lambda_i`
        are the barycentric coordinates of :math:`\hat s` in the reference simplex
        of dimension ``d - 1`` (same vertex numbering one dimension lower). So vertex
        ``i`` of :math:`\hat F` goes to ``vertices[g_i]``. On the triangle this is
        :math:`x(t) = v_f (1 - t)/2 + v_{f+1} (1 + t)/2`, the map of
        :func:`triangle_edge_points`. Returns shape ``(n, d)``.
        """
        index = self._face_index(f)
        s = _point_array(face_points, self.dim - 1, "face_points")
        corners = self.vertices[list(self.face_vertices[index])]
        return np.ascontiguousarray(_barycentric_coordinates(s) @ corners)

    def face_jacobian(self, f) -> np.ndarray:
        r"""Constant Jacobian :math:`J_f = \partial x / \partial \hat s` of :meth:`face_map`, shape ``(d, d - 1)``.

        Column ``j`` is :math:`(v_{g_{j+1}} - v_{g_0}) / 2` with
        ``(g_0, ..., g_{d-1}) = face_vertices[f]``.
        """
        index = self._face_index(f)
        corners = self.vertices[list(self.face_vertices[index])]
        return np.ascontiguousarray((0.5 * (corners[1:] - corners[0])).T)


def _read_only(array: np.ndarray) -> np.ndarray:
    """Return a C-contiguous ``float64`` copy of ``array`` that cannot be written."""
    result = np.array(array, dtype=np.float64, order="C")
    result.setflags(write=False)
    return result


def reference_simplex(dim) -> ReferenceSimplex:
    """Return the reference simplex of dimension ``dim`` (1, 2 or 3); one shared instance per dimension.

    See :class:`ReferenceSimplex` for the vertex, face and edge numbering. Raises
    ``ValueError`` for any other dimension.
    """
    return _build_reference_simplex(_check_dimension(dim))


@lru_cache(maxsize=None)
def _build_reference_simplex(dim: int) -> ReferenceSimplex:
    """Compute the vertices, faces, edges, face measure ratios and unit outward normals of the ``dim``-simplex."""
    vertices = np.asarray(_SIMPLEX_VERTICES[dim], dtype=np.float64)
    face_vertices = _SIMPLEX_FACE_VERTICES[dim]
    # Gradients of the barycentric coordinates: grad lambda_0 = -(1/2) (1, ..., 1),
    # grad lambda_k = (1/2) e_k.
    gradients = np.vstack((-0.5 * np.ones((1, dim)), 0.5 * np.eye(dim)))
    ratios = np.empty(dim + 1, dtype=np.float64)
    normals = np.empty((dim + 1, dim), dtype=np.float64)
    for f, corners in enumerate(face_vertices):
        (opposite,) = set(range(dim + 1)) - set(corners)
        normals[f] = -gradients[opposite] / np.linalg.norm(gradients[opposite]) + 0.0  # + 0.0 turns -0.0 into 0.0
        jacobian = (0.5 * (vertices[list(corners[1:])] - vertices[corners[0]])).T
        ratios[f] = 1.0 if dim == 1 else np.sqrt(np.linalg.det(jacobian.T @ jacobian))
    return ReferenceSimplex(
        dim=dim,
        vertices=_read_only(vertices),
        measure=2.0**dim / factorial(dim),
        face_vertices=face_vertices,
        edges=_SIMPLEX_EDGES[dim],
        face_measure_ratios=_read_only(ratios),
        outward_normals=_read_only(normals),
    )


# ---------------------------------------------------------------------------
# Permutations of the vertices of a face
# ---------------------------------------------------------------------------


def _check_permutation(sigma, size: int) -> tuple[int, ...]:
    """Return ``sigma`` as a tuple of ints; raise ``ValueError`` unless it is a permutation of ``range(size)``."""
    values = tuple(int(i) for i in sigma)
    if sorted(values) != list(range(size)):
        raise ValueError(f"expected a permutation of range({size}); got {tuple(sigma)!r}")
    return values


def face_permutations(dim) -> tuple[tuple[int, ...], ...]:
    r"""All permutations of the ``dim`` vertices of a face of the reference ``dim``-simplex.

    A face of a ``d``-simplex has ``d`` vertices, so there are ``d!``
    permutations of ``(0, ..., d - 1)``: the identity first, then the order of
    :func:`itertools.permutations`. Interval ``((0,),)``; triangle
    ``((0, 1), (1, 0))``; tetrahedron ``((0, 1, 2), (0, 2, 1), (1, 0, 2),
    (1, 2, 0), (2, 0, 1), (2, 1, 0))``. Index ``o`` in this tuple is the
    orientation number ``o``.

    **Convention.** A face shared by two elements has one global order of its
    vertices (for example increasing mesh vertex numbers), and each element lists
    the same vertices in its own local order (the order of
    ``ReferenceSimplex.face_vertices``). The orientation :math:`\sigma` maps local
    face-vertex index ``i`` to global face-vertex index ``sigma[i]``: if the
    element lists the mesh vertices :math:`(m_0, \dots, m_{d-1})` and the global
    order is :math:`(g_0, \dots, g_{d-1})`, then :math:`g_{\sigma[i]} = m_i`. A
    point of the reference face with barycentric coordinates
    :math:`\lambda^{loc}` with respect to the local order has the coordinates
    :math:`\lambda^{glob}[\sigma[i]] = \lambda^{loc}[i]` with respect to the
    global order (:func:`permute_face_points`). On the triangle,
    ``sigma = (1, 0)`` maps the edge coordinate ``t`` to ``-t``, the reversed
    edge orientation of the legacy tables.
    """
    dim = _check_dimension(dim)
    return tuple(permutations(range(dim)))


def permute_face_points(sigma, face_points) -> np.ndarray:
    r"""Coordinates, in the global vertex order of a face, of points given in the local order.

    ``face_points`` has shape ``(n, m)`` with ``m = d - 1`` for a face of a
    ``d``-simplex, and ``sigma`` is a permutation of ``range(m + 1)`` with the
    convention of :func:`face_permutations`. Each point is converted to its local
    barycentric coordinates :math:`\lambda^{loc}` (:meth:`ReferenceSimplex.barycentric`
    one dimension lower), the global coordinates
    :math:`\lambda^{glob}[\sigma[i]] = \lambda^{loc}[i]` are formed, and the
    reference coordinates :math:`\xi_k = 2\lambda^{glob}_k - 1`, ``k = 1, ..., m``,
    are returned with shape ``(n, m)``. The identity returns an exact copy.
    For ``m = 1``, ``sigma = (1, 0)`` returns ``-t``. Successive calls compose as
    ``permute_face_points(a, permute_face_points(b, s)) =
    permute_face_points(compose_permutations(a, b), s)``.
    """
    s = np.asarray(face_points, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"face_points must have shape (n, m); got shape {s.shape}")
    sigma = _check_permutation(sigma, s.shape[1] + 1)
    if sigma == tuple(range(len(sigma))):
        return np.array(s, order="C")
    local = _barycentric_coordinates(s)
    glob = np.empty_like(local)
    glob[:, list(sigma)] = local
    return _coordinates_from_barycentric(glob)


def compose_permutations(a, b) -> tuple[int, ...]:
    r"""Composition :math:`a \circ b` of two permutations of the same length: ``result[i] = a[b[i]]``.

    ``b`` is applied first and ``a`` second, so the point action of
    :func:`permute_face_points` is a group homomorphism:
    ``permute_face_points(compose_permutations(a, b), s)`` equals
    ``permute_face_points(a, permute_face_points(b, s))``.
    """
    a = _check_permutation(a, len(tuple(a)))
    b = _check_permutation(b, len(a))
    return tuple(a[i] for i in b)


def inverse_permutation(a) -> tuple[int, ...]:
    """Inverse permutation ``inv`` with ``inv[a[i]] = i`` for every ``i``.

    ``compose_permutations(a, inv)`` and ``compose_permutations(inv, a)`` are
    the identity.
    """
    a = _check_permutation(a, len(tuple(a)))
    inverse = [0] * len(a)
    for i, image in enumerate(a):
        inverse[image] = i
    return tuple(inverse)


# ---------------------------------------------------------------------------
# Quadrature rules exact for a total degree
# ---------------------------------------------------------------------------


def _gauss_points_per_direction(exact_degree: int) -> int:
    """Number ``n = ceil((k + 1) / 2)`` of Gauss points per direction, so that ``2n - 1 >= k``."""
    return exact_degree // 2 + 1


def gauss_jacobi_collapsed_rule(dim, exact_degree) -> tuple[np.ndarray, np.ndarray]:
    r"""Collapsed Gauss-Jacobi rule on the reference ``dim``-simplex, exact for total degree ``exact_degree``.

    Returns ``points`` ``(n^d, d)`` and ``weights`` ``(n^d,)`` in ``float64``, with
    ``n = ceil((k + 1) / 2)`` points per direction and ``k = exact_degree``.
    Direction ``j = 1, ..., d`` uses the ``n`` Gauss-Jacobi points
    :math:`\eta_j` and weights of ``scipy.special.roots_jacobi(n, j - 1, 0)``,
    whose weight function is :math:`(1 - \eta)^{j-1}`. The collapsed (Duffy,
    Stroud) map is

    .. math::

        \lambda_j = \frac{1 + \eta_j}{2} \prod_{i > j} \frac{1 - \eta_i}{2},
        \qquad \xi_j = 2\lambda_j - 1,

    i.e. :math:`\xi_1 = (1+\eta_1)(1-\eta_2)/2 - 1`, :math:`\xi_2 = \eta_2` on the
    triangle and :math:`\xi_1 = (1+\eta_1)(1-\eta_2)(1-\eta_3)/4 - 1`,
    :math:`\xi_2 = (1+\eta_2)(1-\eta_3)/2 - 1`, :math:`\xi_3 = \eta_3` on the
    tetrahedron. Its Jacobian :math:`\prod_j ((1-\eta_j)/2)^{j-1}` is carried by
    the Gauss-Jacobi weight functions, so the weights are the products of the
    one-dimensional weights times :math:`2^{-d(d-1)/2}`. A polynomial of total
    degree ``k`` in :math:`\xi` has degree at most ``k`` in each :math:`\eta_j`,
    which ``n`` Gauss points integrate exactly (``2n - 1 >= k``). For ``d = 1``
    this is the Gauss-Legendre rule. The weights are positive and sum to
    :math:`2^d/d!`, and every point lies strictly inside the simplex. The points
    are ordered with :math:`\eta_1` varying slowest and :math:`\eta_d` fastest.
    """
    from scipy.special import roots_jacobi

    dim = _check_dimension(dim)
    n = _gauss_points_per_direction(_check_exact_degree(exact_degree))
    rules = [roots_jacobi(n, float(alpha), 0.0) for alpha in range(dim)]
    eta = np.stack([g.ravel() for g in np.meshgrid(*[x for x, _ in rules], indexing="ij")], axis=1)
    weights = np.ones(eta.shape[0], dtype=np.float64)
    for g in np.meshgrid(*[w for _, w in rules], indexing="ij"):
        weights *= g.ravel()
    weights *= 2.0 ** (-(dim * (dim - 1) // 2))
    lam = np.empty_like(eta)
    collapse = np.ones(eta.shape[0], dtype=np.float64)
    for j in reversed(range(dim)):
        lam[:, j] = 0.5 * (1.0 + eta[:, j]) * collapse
        collapse = collapse * (0.5 * (1.0 - eta[:, j]))
    points = 2.0 * lam - 1.0
    return np.ascontiguousarray(points, dtype=np.float64), np.ascontiguousarray(weights, dtype=np.float64)


def _merge_coincident_points(points: np.ndarray, weights: np.ndarray, atol: float = 1.0e-12):
    """Merge quadrature points closer than ``atol`` (max norm) into one point carrying their summed weight.

    ``points`` has shape ``(n, m)`` and ``weights`` shape ``(n,)``. Points are
    grouped by the connected components of the graph that joins every pair within
    ``atol``; each group becomes its mean point with the sum of its weights. A rule
    and its merged form integrate every polynomial to the same value up to a
    change of order ``atol`` in the points. Returns ``(points, weights)`` in
    ``float64``, in order of first appearance.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    pairs = cKDTree(points).query_pairs(r=atol, p=np.inf, output_type="ndarray")
    if pairs.size == 0:
        return points, weights
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points),) * 2)
    _, labels = connected_components(graph, directed=False)
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    groups = rank[inverse]
    counts = np.bincount(groups)
    merged_weights = np.bincount(groups, weights=weights)
    merged_points = np.stack([np.bincount(groups, weights=points[:, j]) / counts for j in range(points.shape[1])], axis=1)
    return np.ascontiguousarray(merged_points), np.ascontiguousarray(merged_weights)


def simplex_volume_rule(dim, exact_degree) -> tuple[np.ndarray, np.ndarray]:
    """Positive-weight rule on the reference ``dim``-simplex exact for total degree ``exact_degree``.

    Returns ``points`` ``(nq, d)`` and ``weights`` ``(nq,)`` in ``float64``; the
    weights sum to ``2^d / d!``. With ``k = exact_degree``:

    - ``d = 1``: Gauss-Legendre with ``ceil((k + 1) / 2)`` points (``numpy``
      ``leggauss``), symmetric under ``t -> -t``;
    - ``d = 2``: the rule of ``_symmetric_triangle_quadrature(max(1, (k + 1) // 2),
      exact_degree=k)``, invariant under the six permutations of the triangle
      vertices: a tabulated Dunavant rule for ``k <= 14`` (rule of the smallest
      tabulated even degree ``>= max(k, 1)``), above that a collapsed
      Gauss-Legendre rule expanded into full permutation orbits. Above degree 14
      that expansion lists most orbits twice (its base rule is already symmetric
      under exchanging vertices 1 and 2); coincident points are merged here and
      their weights added, which halves the point count without changing the
      rule. The points are computed in ``float64`` whatever ``REAL_DTYPE`` is;
    - ``d = 3``: :func:`gauss_jacobi_collapsed_rule` (not symmetric).
    """
    dim = _check_dimension(dim)
    k = _check_exact_degree(exact_degree)
    if dim == 1:
        points, weights = np.polynomial.legendre.leggauss(_gauss_points_per_direction(k))
        points = points[:, None]
    elif dim == 2:
        points, weights = _symmetric_triangle_quadrature(max(1, (k + 1) // 2), exact_degree=k, dtype=np.float64)
        points, weights = _merge_coincident_points(points, weights)
    else:
        points, weights = gauss_jacobi_collapsed_rule(3, k)
    return np.ascontiguousarray(points, dtype=np.float64), np.ascontiguousarray(weights, dtype=np.float64)


def simplex_face_rule(dim, exact_degree) -> tuple[np.ndarray, np.ndarray]:
    """Rule on the reference face of the ``dim``-simplex, exact for total degree ``exact_degree``.

    The reference face is the reference simplex of dimension ``d - 1``, and the
    rule is ``simplex_volume_rule(d - 1, exact_degree)``: points ``(nq, d - 1)``
    and weights ``(nq,)`` summing to the face measure ``2^(d-1) / (d-1)!``. Both
    face rules are invariant under every permutation of the face vertices (the
    symmetric Gauss-Legendre rule for ``d = 2``, the symmetric triangle rule for
    ``d = 3``), so :func:`face_quadrature_permutation` applies to them. For
    ``d = 1`` a face is a point: one point with no coordinates (shape ``(1, 0)``)
    and weight 1.
    """
    dim = _check_dimension(dim)
    k = _check_exact_degree(exact_degree)
    if dim == 1:
        return np.zeros((1, 0), dtype=np.float64), np.ones(1, dtype=np.float64)
    return simplex_volume_rule(dim - 1, k)


def point_permutation(points_from, points_to, atol: float = 1.0e-12) -> np.ndarray:
    """Index array ``p`` with ``points_to[p[i]]`` equal to ``points_from[i]`` within ``atol``, for every ``i``.

    Both arrays have the same shape ``(n, m)``; ``p`` has shape ``(n,)``, dtype
    ``int64``, and is a permutation of ``range(n)``. Points are compared in the
    maximum norm. Points of ``points_to`` closer than ``atol`` to each other
    (connected by chains of such pairs) form one group of copies of the same
    point. Each point of ``points_from`` is sent to the group of its nearest point
    of ``points_to``, and inside a group the points of ``points_from`` are paired
    with the copies in increasing index order on both sides. This makes ``p``
    unique even when a rule repeats a point, and for an orientation ``sigma``
    acting on a rule whose copies are exact, the permutations compose like the
    orientations (see :func:`face_quadrature_permutation`). Raises
    ``ValueError``, stating the largest distance, when a group receives a number
    of points different from its number of copies or when a matched pair is
    farther apart than ``atol``.
    """
    source = np.asarray(points_from, dtype=np.float64)
    target = np.asarray(points_to, dtype=np.float64)
    if source.ndim != 2 or source.shape != target.shape:
        raise ValueError(
            f"points_from and points_to must have the same shape (n, m); got {source.shape} and {target.shape}"
        )
    n, m = source.shape
    if n == 0 or m == 0:
        return np.arange(n, dtype=np.int64)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import KDTree

    tree = KDTree(target)
    pairs = tree.query_pairs(atol, p=np.inf, output_type="ndarray")
    if pairs.size:
        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, target_group = connected_components(graph, directed=False)
    else:
        target_group = np.arange(n)
    nearest_distance, nearest = tree.query(source, k=1, p=np.inf)
    source_group = target_group[nearest]
    indices = np.arange(n)
    source_order = np.lexsort((indices, source_group))
    target_order = np.lexsort((indices, target_group))
    if not np.array_equal(source_group[source_order], target_group[target_order]):
        raise ValueError(
            "points_to is not a permutation of points_from within "
            f"atol={atol:g}: some point of points_to is the nearest point of more points of points_from "
            f"than it has copies; the largest distance to a nearest point is {float(np.max(nearest_distance)):.3e}"
        )
    index = np.empty(n, dtype=np.int64)
    index[source_order] = target_order
    mismatch = float(np.max(np.abs(source - target[index])))
    if not mismatch <= atol:
        raise ValueError(
            "points_to is not a permutation of points_from within "
            f"atol={atol:g}: the largest distance between matched points is {mismatch:.3e}"
        )
    return index


def face_quadrature_permutation(dim, face_points, sigma) -> np.ndarray:
    r"""Index permutation ``p`` of a face rule induced by the face orientation ``sigma``.

    ``face_points`` ``(nq, d - 1)`` are the points of a rule on the reference face
    of the ``dim``-simplex whose point set is invariant under ``sigma`` (the rules
    of :func:`simplex_face_rule`), and ``sigma`` is a permutation of
    ``range(dim)`` (:func:`face_permutations`). The result is
    ``point_permutation(permute_face_points(sigma, face_points), face_points)``,
    shape ``(nq,)``, dtype ``int64``:

    .. math::

        \text{face\_points}[p[q]] = \text{permute\_face\_points}(\sigma,
        \text{face\_points})[q],

    so ``p[q]`` is the index of the rule point at which local point ``q`` lies when
    the face is read in its global vertex order. If ``values[..., j]`` holds a
    function :math:`\mu` defined in the global order at ``face_points[j]``, then
    ``values[..., p]`` holds :math:`\mu` at the element's local points, and a face
    integral :math:`\int_{\hat F} \phi\,(\mu \circ \sigma)` is
    :math:`\sum_q w_q \phi(\hat s_q)\,\text{values}[p[q]]`. The symmetric rules
    also satisfy ``weights[p] = weights``. Composition: the permutation of
    ``compose_permutations(a, b)`` is ``p_a[p_b]``. This holds also for the
    triangle rules of exact degree above 14, which contain most points twice
    (the collapsed Gauss-Legendre rule they are expanded from is already
    symmetric under exchanging vertices 1 and 2): :func:`point_permutation`
    pairs the two copies in index order. Raises ``ValueError`` when the rule is
    not invariant under ``sigma`` (tolerance ``1e-12``).
    """
    dim = _check_dimension(dim)
    points = _point_array(face_points, dim - 1, "face_points")
    sigma = _check_permutation(sigma, dim)
    return point_permutation(permute_face_points(sigma, points), points)


__all__ = [
    "ReferenceSimplex",
    "compose_permutations",
    "face_permutations",
    "face_quadrature_permutation",
    "gauss_jacobi_collapsed_rule",
    "inverse_permutation",
    "monomial_integral",
    "permute_face_points",
    "point_permutation",
    "reference_simplex",
    "simplex_face_rule",
    "simplex_volume_rule",
    "triangle_edge_points",
]
