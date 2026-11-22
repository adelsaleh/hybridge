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
"""

from __future__ import annotations

from fractions import Fraction
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


def _symmetric_triangle_quadrature(order: int, exact_degree: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return a positive-weight symmetric rule exact for ``P_exact_degree`` (default ``2*order``).

    The smallest tabulated compact Dunavant rule of at least that degree is
    used when available (even degrees up to 14).  Otherwise a minimally exact
    Duffy rule is expanded into complete barycentric permutation orbits.
    Orbit symmetrization preserves the original rule's polynomial exactness
    and positivity while providing a fully symmetric rule for any degree.
    """
    exact_degree = max(2, 2 * order) if exact_degree is None else max(1, int(exact_degree))
    compact = _compact_dunavant_degree(exact_degree)
    blocks = None if compact is None else _DUNAVANT_RULE_BLOCKS[compact]
    if blocks is None:
        points, base_weights = _triangle_quadrature(order, _duffy_points_for_degree(exact_degree))
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
            orbit = np.asarray(orbit, dtype=REAL_DTYPE)
            orbit_points.append(
                np.column_stack((-1.0 + 2.0 * orbit[:, 1], -1.0 + 2.0 * orbit[:, 2]))
            )
            orbit_weights.append(np.full(len(orbit), weight / len(orbit), dtype=REAL_DTYPE))
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
    barycentric = np.asarray(barycentric, dtype=REAL_DTYPE)
    points = np.column_stack((-1.0 + 2.0 * barycentric[:, 1], -1.0 + 2.0 * barycentric[:, 2]))
    return np.ascontiguousarray(points), np.ascontiguousarray(weights, dtype=REAL_DTYPE)


def _quadrature_point_count(order: int, num_1d: int | None) -> int:
    """Resolve and validate the one-dimensional quadrature point count."""
    if num_1d is None:
        return max(2 * order + 2, 2)
    count = int(num_1d)
    if count < 1:
        raise ValueError("quadrature point count must be positive")
    return count


def _triangle_quadrature(order: int, num_1d: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build a collapsed tensor-product Gauss rule on the reference triangle.

    A tensor Gauss-Legendre rule on ``[-1, 1]^2`` is mapped to the reference
    triangle with the Duffy-style transform

    ``x = -1 + 2*u*(1-v)``, ``y = -1 + 2*u*v``.

    The returned points have shape ``(num_quads, 2)`` and weights have shape
    ``(num_quads,)``.  The weights integrate on the reference triangle only;
    callers multiply by each physical element Jacobian when assembling.
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
    return np.ascontiguousarray(points, dtype=REAL_DTYPE), np.ascontiguousarray(weights, dtype=REAL_DTYPE)


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


__all__ = ["monomial_integral", "triangle_edge_points"]
