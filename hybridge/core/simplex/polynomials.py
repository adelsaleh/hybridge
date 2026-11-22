r"""Polynomial bases on reference simplices of dimension 1, 2 and 3.

The one-dimensional bases used for triangle edges (Lagrange polynomials on given
nodes, Legendre polynomials and Bernstein polynomials) were moved here unchanged
from :mod:`hybridge.core.quadrature` and :mod:`hybridge.core.space`, which still
expose them under their original names.

The rest of the module tabulates four modal bases of the space :math:`P_p` of
polynomials of total degree at most ``p`` on the reference simplex of dimension
``d = 1, 2, 3``: the interval ``[-1, 1]``, the triangle
``conv{(-1,-1), (1,-1), (-1,1)}`` and the tetrahedron
``conv{(-1,-1,-1), (1,-1,-1), (-1,1,-1), (-1,-1,1)}``. The space has
``C(p + d, d)`` functions (:func:`modal_dimension`). The bases follow Karniadakis
and Sherwin, *Spectral/hp Element Methods for CFD*, 2nd ed., chapter 3 (K&S
below):

``dub_orth``
    The orthogonal (Dubiner) basis, not scaled to unit norm; the mass matrix is diagonal.
``orthonormal``
    ``dub_orth`` times the constant of :func:`orthonormal_scale`; the mass matrix
    is the identity.
``bernstein``
    :math:`B_\alpha = \frac{p!}{\alpha_0!\cdots\alpha_d!}\lambda_0^{\alpha_0}\cdots\lambda_d^{\alpha_d}`,
    :math:`|\alpha| = p`.
``hier_C0``
    The modified (C0) basis: one mode per vertex, then edge, face and interior
    modes that vanish on every face of the simplex that does not contain their
    edge or face (formulas in :func:`_modified_kernel`, order in
    :func:`_modified_table`).

On the tetrahedron, ``hier_C0`` is the modified basis listed in K&S, Appendix
D.1.2.4 (pp. 609-610), written in barycentric coordinates (formulas in
:func:`_modified_kernel`); the vertices A, B, C, D of the book are the vertices
0, 1, 2, 3 here. Its Jacobi weights are :math:`(1,1)` along an edge,
:math:`(2p+1,1)` for the second factor of a face mode and :math:`(2p+2q+1,1)` for
the third factor of an interior mode (eqs. 3.2.15-3.2.17, p. 111). Two misprints
of that listing are corrected:

- Vertex C is printed as :math:`\frac{1-\eta_1}{2}\frac{1+\eta_2}{2}\frac{1-\eta_3}{2}`,
  which equals :math:`\lambda_0\lambda_2/(\lambda_0+\lambda_1)`: a rational
  function, not 1 at vertex C and not continuous there. The factor
  :math:`(1-\eta_1)/2` does not belong there: the construction of Section 3.2.3.3
  (pp. 114-117) adds the two modes collapsed onto C, and
  :math:`(1-\eta_1)/2 + (1+\eta_1)/2 = 1`. The vertex mode used here is
  :math:`\frac{1+\eta_2}{2}\frac{1-\eta_3}{2} = \lambda_2`. The other printed
  vertex modes (triangle A, B, C on p. 604; tetrahedron A, B, D) equal
  :math:`\lambda_v`.
- The three faces through the top vertex are labelled ABE, ACE and BCE; the
  tetrahedron has no vertex E, and they are the faces ABD, ACD and BCD. Their
  formulas are correct.

The triangle ``hier_C0`` of :mod:`hybridge.core.basis` is unchanged. Its vertex and
edge modes are those of the book (p. 604), with a decreasing degree along edge 2,
but its interior modes use the Jacobi weight :math:`(2p-1,1)` instead of the
book's :math:`(2p+1,1)`. The restriction of the tetrahedron basis to the face
:math:`\xi_3 = -1` is the book's triangle basis, so it equals the triangle
``hier_C0`` only in span.

Notation used by all docstrings below. The barycentric coordinates are
:math:`\lambda_k = (1 + \xi_k)/2` for ``k = 1, ..., d`` and
:math:`\lambda_0 = 1 - \sum_{k \ge 1}\lambda_k`; :math:`\lambda_v` is 1 at vertex
``v`` and 0 at the other vertices. The partial sums are
:math:`s_k = \lambda_0 + \cdots + \lambda_k`, so :math:`s_d = 1`. The collapsed
coordinates of K&S are :math:`\eta_k = (\lambda_k - s_{k-1}) / s_k`; on the
tetrahedron :math:`\eta_1 = 2(1+\xi_1)/(-\xi_2-\xi_3) - 1`,
:math:`\eta_2 = 2(1+\xi_2)/(1-\xi_3) - 1`, :math:`\eta_3 = \xi_3`, and
:math:`(1-\eta_k)/2 = s_{k-1}/s_k`, :math:`(1+\eta_k)/2 = \lambda_k/s_k`. They
are undefined where :math:`s_k = 0`: on the triangle at the vertex ``(-1, 1)``,
on the tetrahedron on the edge :math:`\xi_2 + \xi_3 = 0` (from ``(-1,-1,1)`` to
``(-1,1,-1)``) and at the vertex ``(-1,-1,1)``. Nodal sets and face quadrature
rules contain such points. The bases in two and three dimensions are therefore
written with the scaled Jacobi polynomials

.. math::

    Q_n^{(a,b)}(x, t) = t^n P_n^{(a,b)}(x / t),

which are polynomials in ``(x, t)``, homogeneous of degree ``n``, computed by the
three-term recurrence without any division by ``t`` (:func:`scaled_jacobi`;
R. C. Kirby, ACM TOMS 37, 2010). A K&S product such as
:math:`P_p(\eta_1)((1-\eta_2)/2)^p = Q_p^{(0,0)}(\lambda_1-\lambda_0, s_1)/s_2^p`
then becomes a product of :math:`Q` evaluated at barycentric coordinates in which
every power of :math:`s_k` cancels, so values and gradients are computed at every
point, singular points of the collapsed coordinates included. Gradients are
computed by the chain rule through the barycentric coordinates
(:math:`\nabla\lambda_0 = -\tfrac12(1, \ldots, 1)`,
:math:`\nabla\lambda_k = \tfrac12 e_k`), not by finite differences.

On the triangle, ``dub_orth``, ``bernstein`` and ``hier_C0`` are the existing
kernels of :mod:`hybridge.core.basis` (same values and same order of the modes).
In one dimension, ``dub_orth`` and ``bernstein`` are :func:`_legendre_edge_basis`
and :func:`_bernstein_edge_basis`. Every evaluator returns ``float64`` arrays: the
values have shape ``(num_points, n)`` and the reference gradients
:math:`\partial\phi/\partial\xi_k` shape ``(num_points, n, d)``. The tabulation is
done by Numba kernels over the points (serial by default, so that they can be
called from parallel loops; ``parallel=True`` runs the points in parallel).
"""

from __future__ import annotations

from functools import lru_cache
from math import comb, factorial, sqrt

import numba as nb
import numpy as np

from hybridge.core import basis as _basis2d
from hybridge.core.simplex.reference import reference_simplex
from hybridge.runtime.precision import REAL_DTYPE


def _lagrange_basis(nodes: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Evaluate 1D nodal Lagrange basis functions at ``points``."""
    nodes = np.asarray(nodes, dtype=REAL_DTYPE)
    points = np.asarray(points, dtype=REAL_DTYPE)
    values = np.ones((nodes.size, points.size), dtype=REAL_DTYPE)
    for i in range(nodes.size):
        for j in range(nodes.size):
            if i != j:
                values[i] *= (points - nodes[j]) / (nodes[i] - nodes[j])
    return np.ascontiguousarray(values)


def _bernstein_edge_basis(order: int, points: np.ndarray) -> np.ndarray:
    """Tabulate the Bernstein trace basis at one-dimensional edge points."""
    from math import factorial

    r = 0.5 * (np.asarray(points, dtype=REAL_DTYPE) + 1.0)
    values = np.empty((order + 1, r.size), dtype=REAL_DTYPE)
    for j in range(order + 1):
        coeff = factorial(order) / (factorial(j) * factorial(order - j))
        values[j] = coeff * (1.0 - r) ** (order - j) * r ** j
    return np.ascontiguousarray(values)


def _legendre_edge_basis(order: int, points: np.ndarray) -> np.ndarray:
    """Tabulate the Legendre trace basis at one-dimensional edge points."""
    points = np.asarray(points, dtype=REAL_DTYPE)
    values = np.empty((order + 1, points.size), dtype=REAL_DTYPE)
    for j in range(order + 1):
        values[j] = np.polynomial.legendre.Legendre.basis(j)(points)
    return np.ascontiguousarray(values)


MODAL_BASIS_NAMES = ("dub_orth", "orthonormal", "bernstein", "hier_C0")


# ---------------------------------------------------------------------------
# Scaled Jacobi polynomials t^n P_n^{(a,b)}(x/t)
# ---------------------------------------------------------------------------


@nb.njit(cache=True, fastmath=True)
def _scaled_jacobi_fill(order, alpha, beta, x, t, values, d_dx, d_dt):
    r"""Write :math:`Q_k = t^k P_k^{(\alpha,\beta)}(x/t)`, :math:`\partial_x Q_k`, :math:`\partial_t Q_k`, ``k = 0..order``.

    ``x`` and ``t`` are scalars; ``values``, ``d_dx`` and ``d_dt`` are 1D arrays of
    length at least ``order + 1`` whose first ``order + 1`` entries are
    overwritten. The recurrence is the Jacobi three-term recurrence multiplied by
    :math:`t^n`: :math:`Q_0 = 1`,
    :math:`Q_1 = ((\alpha+\beta+2)x + (\alpha-\beta)t)/2`, and for ``n >= 2``, with
    :math:`c = 2n + \alpha + \beta`,

    .. math::

        2n(n+\alpha+\beta)(c-2)\,Q_n
        = (c-1)\big(c(c-2)\,x + (\alpha^2-\beta^2)\,t\big) Q_{n-1}
        - 2(n+\alpha-1)(n+\beta-1)\,c\,t^2 Q_{n-2}.

    The derivatives follow by differentiating this recurrence. Nothing is divided
    by ``t``, so ``t = 0`` is allowed. Requires :math:`\alpha, \beta > -1`.
    """
    values[0] = 1.0
    d_dx[0] = 0.0
    d_dt[0] = 0.0
    if order < 1:
        return
    values[1] = 0.5 * ((alpha + beta + 2.0) * x + (alpha - beta) * t)
    d_dx[1] = 0.5 * (alpha + beta + 2.0)
    d_dt[1] = 0.5 * (alpha - beta)
    tt = t * t
    for n in range(2, order + 1):
        nf = float(n)
        c = 2.0 * nf + alpha + beta
        inverse = 1.0 / (2.0 * nf * (nf + alpha + beta) * (c - 2.0))
        a_x = (c - 1.0) * c * (c - 2.0)
        a_t = (c - 1.0) * (alpha * alpha - beta * beta)
        a_2 = 2.0 * (nf + alpha - 1.0) * (nf + beta - 1.0) * c
        linear = a_x * x + a_t * t
        values[n] = (linear * values[n - 1] - a_2 * tt * values[n - 2]) * inverse
        d_dx[n] = (a_x * values[n - 1] + linear * d_dx[n - 1] - a_2 * tt * d_dx[n - 2]) * inverse
        d_dt[n] = (
            a_t * values[n - 1]
            + linear * d_dt[n - 1]
            - a_2 * (2.0 * t * values[n - 2] + tt * d_dt[n - 2])
        ) * inverse


@nb.njit(cache=True, fastmath=True)
def _scaled_jacobi_kernel(order, alpha, beta, x, t):
    r"""Tabulate :math:`Q_k^{(\alpha,\beta)}(x_i, t_i)` and its derivatives in ``x`` and ``t`` (1D arrays ``x``, ``t``).

    Returns three arrays of shape ``(x.size, order + 1)``: values, :math:`\partial_x`
    and :math:`\partial_t`, column ``k`` for degree ``k``.
    """
    num_points = x.shape[0]
    values = np.empty((num_points, order + 1), dtype=np.float64)
    d_dx = np.empty((num_points, order + 1), dtype=np.float64)
    d_dt = np.empty((num_points, order + 1), dtype=np.float64)
    for i in nb.prange(num_points):
        _scaled_jacobi_fill(order, alpha, beta, x[i], t[i], values[i], d_dx[i], d_dt[i])
    return values, d_dx, d_dt


_scaled_jacobi_kernel_parallel = nb.njit(parallel=True, cache=True, fastmath=True)(
    getattr(_scaled_jacobi_kernel, "py_func", _scaled_jacobi_kernel)
)


def scaled_jacobi(order: int, alpha: float, beta: float, x, t, *, parallel: bool = False):
    r"""Scaled Jacobi polynomials :math:`Q_k(x, t) = t^k P_k^{(\alpha,\beta)}(x/t)`, ``k = 0..order``, and derivatives.

    ``x`` and ``t`` are broadcast to a common shape ``S``. Returns
    ``(values, d_dx, d_dt)``, each a ``float64`` array of shape ``S + (order + 1,)``
    holding :math:`Q_k`, :math:`\partial Q_k/\partial x` and
    :math:`\partial Q_k/\partial t`. :math:`Q_k` is a polynomial in ``(x, t)``,
    homogeneous of degree ``k`` (:math:`Q_k(sx, st) = s^k Q_k(x, t)`), equal to
    :math:`P_k^{(\alpha,\beta)}(x)` at ``t = 1``. It is computed by the three-term
    recurrence of :func:`_scaled_jacobi_fill`, which does not divide by ``t``, so
    ``t = 0`` gives the leading coefficient times :math:`x^k`. Requires
    ``order >= 0`` and :math:`\alpha, \beta > -1`.
    """
    order = _check_order(order)
    if not (alpha > -1.0 and beta > -1.0):
        raise ValueError(f"Jacobi parameters must be > -1; got alpha={alpha!r}, beta={beta!r}")
    xb, tb = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(t, dtype=np.float64))
    shape = xb.shape
    kernel = _scaled_jacobi_kernel_parallel if parallel else _scaled_jacobi_kernel
    values, d_dx, d_dt = kernel(
        order,
        float(alpha),
        float(beta),
        np.ascontiguousarray(xb.reshape(-1)),
        np.ascontiguousarray(tb.reshape(-1)),
    )
    return tuple(np.ascontiguousarray(a.reshape(shape + (order + 1,))) for a in (values, d_dx, d_dt))


# ---------------------------------------------------------------------------
# Compiled kernels of the modal bases (dimension 1, 2 or 3 from points.shape[1])
# ---------------------------------------------------------------------------


@nb.njit(cache=True, fastmath=True)
def _scaled_jacobi_values(order, alpha, beta, x, t, values):
    r"""Write :math:`Q_k = t^k P_k^{(\alpha,\beta)}(x/t)` for ``k = 0..order`` into ``values[:order + 1]``.

    The values of :func:`_scaled_jacobi_fill` with the same arithmetic, without the
    two derivative recurrences; used when no gradient is requested.
    """
    values[0] = 1.0
    if order < 1:
        return
    values[1] = 0.5 * ((alpha + beta + 2.0) * x + (alpha - beta) * t)
    tt = t * t
    for n in range(2, order + 1):
        nf = float(n)
        c = 2.0 * nf + alpha + beta
        inverse = 1.0 / (2.0 * nf * (nf + alpha + beta) * (c - 2.0))
        a_x = (c - 1.0) * c * (c - 2.0)
        a_t = (c - 1.0) * (alpha * alpha - beta * beta)
        a_2 = 2.0 * (nf + alpha - 1.0) * (nf + beta - 1.0) * c
        values[n] = ((a_x * x + a_t * t) * values[n - 1] - a_2 * tt * values[n - 2]) * inverse


# The kernels below loop over blocks of _POINTS_PER_TASK consecutive points (the
# parallel loop runs over blocks) and allocate their work arrays once per block:
# allocating them once per point made the parallel kernels allocation-bound.
_POINTS_PER_TASK = 64


@nb.njit(cache=True, fastmath=True)
def _orthogonal_kernel(points, order, mode_index, scale, want_gradients):
    r"""Tabulate the orthogonal basis ``dub_orth`` times ``scale`` and, if ``want_gradients``, its reference gradients.

    ``points`` has shape ``(num_points, d)``, ``d = 1, 2, 3``. Mode
    :math:`(n_1, \ldots, n_d)` is

    .. math::

        \phi = \prod_{k=1}^{d} Q_{n_k}^{(\beta_k, 0)}(x_k, t_k), \qquad
        \beta_k = 2(n_1 + \cdots + n_{k-1}) + k - 1,

    with :math:`x_k = \lambda_k - s_{k-1} = \xi_k + \sum_{j>k}\lambda_j` and
    :math:`t_k = s_k = 1 - \sum_{j>k}\lambda_j` (so :math:`t_d = 1`); this equals the
    K&S product
    :math:`\prod_k P_{n_k}^{(\beta_k,0)}(\eta_k)((1-\eta_k)/2)^{n_1+\cdots+n_{k-1}}`.
    Its column is ``mode_index[n_1, n_2, n_3]`` (missing indices are 0), an int64
    array of shape ``(order + 1,) * 3``, and column ``m`` is multiplied by
    ``scale[m]`` (``scale`` has shape ``(num_modes,)``: ones for ``dub_orth``,
    :func:`orthonormal_scale` for ``orthonormal``). For each point the polynomials
    of level 1 are computed once, those of level 2 once per :math:`n_1`, and those
    of level 3 once per :math:`(n_1, n_2)`. Returns ``values``
    ``(num_points, num_modes)`` and ``gradients`` ``(num_points, num_modes, d)``
    (shape ``(0, num_modes, d)`` when ``want_gradients`` is false).
    """
    num_points = points.shape[0]
    dim = points.shape[1]
    num_modes = scale.shape[0]
    values = np.empty((num_points, num_modes), dtype=np.float64)
    if want_gradients:
        gradients = np.empty((num_points, num_modes, dim), dtype=np.float64)
    else:
        gradients = np.empty((0, num_modes, dim), dtype=np.float64)
    num_tasks = (num_points + _POINTS_PER_TASK - 1) // _POINTS_PER_TASK
    for task in nb.prange(num_tasks):
        x = np.empty(3, dtype=np.float64)
        t = np.empty(3, dtype=np.float64)
        gx = np.zeros((3, 3), dtype=np.float64)
        gt = np.zeros((3, 3), dtype=np.float64)
        for k in range(dim):
            gx[k, k] = 1.0
            for c in range(k + 1, dim):
                gx[k, c] = 0.5
                gt[k, c] = -0.5
        av = np.empty(order + 1, dtype=np.float64)
        ax = np.empty(order + 1, dtype=np.float64)
        at = np.empty(order + 1, dtype=np.float64)
        bv = np.empty(order + 1, dtype=np.float64)
        bx = np.empty(order + 1, dtype=np.float64)
        bt = np.empty(order + 1, dtype=np.float64)
        cv = np.empty(order + 1, dtype=np.float64)
        cx = np.empty(order + 1, dtype=np.float64)
        ct = np.empty(order + 1, dtype=np.float64)
        ga = np.zeros(3, dtype=np.float64)
        gb = np.zeros(3, dtype=np.float64)
        stop = min(num_points, (task + 1) * _POINTS_PER_TASK)
        for i in range(task * _POINTS_PER_TASK, stop):
            tail = 0.0
            for k in range(dim - 1, -1, -1):
                xi = points[i, k]
                x[k] = xi + tail
                t[k] = 1.0 - tail
                tail += 0.5 * (1.0 + xi)
            if want_gradients:
                _scaled_jacobi_fill(order, 0.0, 0.0, x[0], t[0], av, ax, at)
            else:
                _scaled_jacobi_values(order, 0.0, 0.0, x[0], t[0], av)
            for p in range(order + 1):
                if want_gradients:
                    for c in range(dim):
                        ga[c] = ax[p] * gx[0, c] + at[p] * gt[0, c]
                if dim == 1:
                    col = mode_index[p, 0, 0]
                    values[i, col] = av[p] * scale[col]
                    if want_gradients:
                        gradients[i, col, 0] = ga[0] * scale[col]
                    continue
                if want_gradients:
                    _scaled_jacobi_fill(order - p, 2.0 * p + 1.0, 0.0, x[1], t[1], bv, bx, bt)
                else:
                    _scaled_jacobi_values(order - p, 2.0 * p + 1.0, 0.0, x[1], t[1], bv)
                for q in range(order - p + 1):
                    if want_gradients:
                        for c in range(dim):
                            gb[c] = bx[q] * gx[1, c] + bt[q] * gt[1, c]
                    ab = av[p] * bv[q]
                    if dim == 2:
                        col = mode_index[p, q, 0]
                        values[i, col] = ab * scale[col]
                        if want_gradients:
                            for c in range(2):
                                gradients[i, col, c] = (ga[c] * bv[q] + av[p] * gb[c]) * scale[col]
                        continue
                    if want_gradients:
                        _scaled_jacobi_fill(order - p - q, 2.0 * (p + q) + 2.0, 0.0, x[2], t[2], cv, cx, ct)
                    else:
                        _scaled_jacobi_values(order - p - q, 2.0 * (p + q) + 2.0, 0.0, x[2], t[2], cv)
                    for r in range(order - p - q + 1):
                        col = mode_index[p, q, r]
                        s = scale[col]
                        values[i, col] = ab * cv[r] * s
                        if want_gradients:
                            for c in range(3):
                                gc = cx[r] * gx[2, c] + ct[r] * gt[2, c]
                                gradients[i, col, c] = ((ga[c] * bv[q] + av[p] * gb[c]) * cv[r] + ab * gc) * s
    return values, gradients


_orthogonal_kernel_parallel = nb.njit(parallel=True, cache=True, fastmath=True)(
    getattr(_orthogonal_kernel, "py_func", _orthogonal_kernel)
)


@nb.njit(cache=True, fastmath=True)
def _bernstein_kernel(points, order, exponents, coefficients, want_gradients):
    r"""Tabulate the Bernstein basis :math:`c_\alpha \prod_v \lambda_v^{\alpha_v}` and, if ``want_gradients``, gradients.

    ``points`` has shape ``(num_points, d)``; ``exponents`` ``(num_modes, d + 1)``
    holds :math:`\alpha` (column ``v`` is the power of :math:`\lambda_v`) and
    ``coefficients`` ``(num_modes,)`` the multinomial coefficients
    :math:`c_\alpha = p!/(\alpha_0!\cdots\alpha_d!)`. The gradient is
    :math:`\sum_v c_\alpha \alpha_v \lambda_v^{\alpha_v - 1}\prod_{w\ne v}\lambda_w^{\alpha_w}\nabla\lambda_v`.
    Returns ``values`` ``(num_points, num_modes)`` and ``gradients``
    ``(num_points, num_modes, d)`` (shape ``(0, num_modes, d)`` without gradients).
    """
    num_points = points.shape[0]
    dim = points.shape[1]
    num_modes = exponents.shape[0]
    values = np.empty((num_points, num_modes), dtype=np.float64)
    if want_gradients:
        gradients = np.zeros((num_points, num_modes, dim), dtype=np.float64)
    else:
        gradients = np.zeros((0, num_modes, dim), dtype=np.float64)
    num_tasks = (num_points + _POINTS_PER_TASK - 1) // _POINTS_PER_TASK
    for task in nb.prange(num_tasks):
        lam = np.empty(dim + 1, dtype=np.float64)
        powers = np.empty((dim + 1, order + 1), dtype=np.float64)
        stop = min(num_points, (task + 1) * _POINTS_PER_TASK)
        for i in range(task * _POINTS_PER_TASK, stop):
            tail = 0.0
            for k in range(dim):
                lam[k + 1] = 0.5 * (points[i, k] + 1.0)
                tail += lam[k + 1]
            lam[0] = 1.0 - tail
            for v in range(dim + 1):
                powers[v, 0] = 1.0
                for e in range(1, order + 1):
                    powers[v, e] = powers[v, e - 1] * lam[v]
            for m in range(num_modes):
                value = coefficients[m]
                for v in range(dim + 1):
                    value *= powers[v, exponents[m, v]]
                values[i, m] = value
                if want_gradients:
                    for v in range(dim + 1):
                        e = exponents[m, v]
                        if e == 0:
                            continue
                        term = coefficients[m] * e * powers[v, e - 1]
                        for w in range(dim + 1):
                            if w != v:
                                term *= powers[w, exponents[m, w]]
                        if v == 0:
                            for c in range(dim):
                                gradients[i, m, c] -= 0.5 * term
                        else:
                            gradients[i, m, v - 1] += 0.5 * term
    return values, gradients


_bernstein_kernel_parallel = nb.njit(parallel=True, cache=True, fastmath=True)(
    getattr(_bernstein_kernel, "py_func", _bernstein_kernel)
)


@nb.njit(cache=True, fastmath=True)
def _modified_kernel(points, order, modes, want_gradients):
    r"""Tabulate the modified C0 basis (``hier_C0``) in barycentric form and, if ``want_gradients``, its gradients.

    ``points`` has shape ``(num_points, d)``. Row ``m`` of the int64 array ``modes``
    ``(num_modes, 8)`` is ``(level, v_0, v_1, v_2, v_3, n_1, n_2, n_3)``:
    ``level = -1`` is the constant 1 (the basis of degree 0); otherwise the mode
    belongs to the entity with vertices :math:`v_0 < \cdots < v_{level}` and is

    .. math::

        \phi = \prod_{j=0}^{level} \lambda_{v_j}
        \prod_{k=1}^{level} Q_{n_k - 1}^{(\alpha_k, 1)}(\lambda_{v_k} - s_{v_k - 1}, s_{v_k}),
        \qquad s_m = \lambda_0 + \cdots + \lambda_m,\quad
        \alpha_k = 2(n_1 + \cdots + n_{k-1}) + 1,

    with :math:`n_k \ge 1`; ``level = 0`` gives the vertex mode :math:`\lambda_{v_0}`.
    The partial sums run over all vertices up to :math:`v_k` (not only those of
    the entity) and :math:`s_d = 1`; with them the mode is the K&S product in
    collapsed coordinates, since :math:`(1 \pm \eta_k)/2` are ratios of these sums
    (see the module docstring). The gradient is accumulated factor by factor with
    the product rule. Returns
    ``values`` ``(num_points, num_modes)`` and ``gradients``
    ``(num_points, num_modes, d)`` (shape ``(0, num_modes, d)`` without gradients).
    """
    num_points = points.shape[0]
    dim = points.shape[1]
    num_modes = modes.shape[0]
    values = np.empty((num_points, num_modes), dtype=np.float64)
    if want_gradients:
        gradients = np.zeros((num_points, num_modes, dim), dtype=np.float64)
    else:
        gradients = np.zeros((0, num_modes, dim), dtype=np.float64)
    num_tasks = (num_points + _POINTS_PER_TASK - 1) // _POINTS_PER_TASK
    for task in nb.prange(num_tasks):
        lam = np.empty(dim + 1, dtype=np.float64)
        # Gradients of lambda_k and of the partial sums s_j = 1 - (lambda_{j+1} + ... + lambda_d)
        # (s_d = 1 exactly, s_0 = lambda_0); they are the same at every point.
        glam = np.zeros((dim + 1, dim), dtype=np.float64)
        gpartial = np.zeros((dim + 1, dim), dtype=np.float64)
        for k in range(dim):
            glam[0, k] = -0.5
            glam[k + 1, k] = 0.5
        for j in range(dim + 1):
            for c in range(j, dim):
                gpartial[j, c] = -0.5
        partial = np.empty(dim + 1, dtype=np.float64)
        qv = np.empty(order + 1, dtype=np.float64)
        qx = np.empty(order + 1, dtype=np.float64)
        qt = np.empty(order + 1, dtype=np.float64)
        g = np.empty(dim, dtype=np.float64)
        stop = min(num_points, (task + 1) * _POINTS_PER_TASK)
        for i in range(task * _POINTS_PER_TASK, stop):
            tail = 0.0
            for k in range(dim):
                lam[k + 1] = 0.5 * (points[i, k] + 1.0)
                tail += lam[k + 1]
            lam[0] = 1.0 - tail
            tail = 0.0
            for j in range(dim, -1, -1):
                partial[j] = 1.0 - tail
                if j >= 1:
                    tail += lam[j]
            for m in range(num_modes):
                level = modes[m, 0]
                if level < 0:
                    values[i, m] = 1.0
                    continue
                value = 1.0
                for c in range(dim):
                    g[c] = 0.0
                for j in range(level + 1):
                    v = modes[m, 1 + j]
                    factor = lam[v]
                    for c in range(dim):
                        g[c] = g[c] * factor + value * glam[v, c]
                    value *= factor
                alpha = 1.0
                for k in range(1, level + 1):
                    v = modes[m, 1 + k]
                    n = modes[m, 4 + k]
                    if want_gradients:
                        _scaled_jacobi_fill(n - 1, alpha, 1.0, lam[v] - partial[v - 1], partial[v], qv, qx, qt)
                    else:
                        _scaled_jacobi_values(n - 1, alpha, 1.0, lam[v] - partial[v - 1], partial[v], qv)
                    factor = qv[n - 1]
                    if want_gradients:
                        for c in range(dim):
                            d_factor = qx[n - 1] * (glam[v, c] - gpartial[v - 1, c]) + qt[n - 1] * gpartial[v, c]
                            g[c] = g[c] * factor + value * d_factor
                    value *= factor
                    alpha += 2.0 * n
                values[i, m] = value
                if want_gradients:
                    for c in range(dim):
                        gradients[i, m, c] = g[c]
    return values, gradients


_modified_kernel_parallel = nb.njit(parallel=True, cache=True, fastmath=True)(
    getattr(_modified_kernel, "py_func", _modified_kernel)
)


# ---------------------------------------------------------------------------
# Mode numbering
# ---------------------------------------------------------------------------


def _check_order(order) -> int:
    """Return ``order`` as an ``int``; raise ``ValueError`` unless it is an integer >= 0."""
    value = int(order)
    if value != order or value < 0:
        raise ValueError(f"polynomial order must be an integer >= 0; got {order!r}")
    return value


def _check_dim(dim) -> int:
    """Return ``dim`` as an ``int``; raise ``ValueError`` unless it is 1, 2 or 3."""
    value = int(dim)
    if value != dim or value not in (1, 2, 3):
        raise ValueError(f"the reference simplex has dimension 1, 2 or 3; got {dim!r}")
    return value


def _check_name(name: str) -> str:
    """Return the entry of ``MODAL_BASIS_NAMES`` equal to ``name`` up to case and spaces, else raise ``ValueError``."""
    key = str(name).strip().lower()
    for candidate in MODAL_BASIS_NAMES:
        if candidate.lower() == key:
            return candidate
    raise ValueError(f"unknown modal basis {name!r}; the modal bases are {', '.join(MODAL_BASIS_NAMES)}")


def _as_points(points, dim: int) -> np.ndarray:
    """Return ``points`` as a C-contiguous ``float64`` array of shape ``(num_points, dim)``.

    For ``dim = 1`` a 1D array of shape ``(num_points,)`` is also accepted. Raises
    ``ValueError`` for any other shape.
    """
    array = np.asarray(points, dtype=np.float64)
    if dim == 1 and array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[1] != dim:
        raise ValueError(f"points must have shape (num_points, {dim}); got {array.shape}")
    return np.ascontiguousarray(array)


def modal_dimension(dim, order) -> int:
    """Return ``C(order + dim, dim)``, the dimension of the polynomials of degree ``<= order`` in ``dim`` variables."""
    return comb(_check_order(order) + _check_dim(dim), _check_dim(dim))


@lru_cache(maxsize=None)
def _orthogonal_indices(dim: int, order: int) -> tuple[tuple[int, ...], ...]:
    """Return the multi-indices ``(n_1, ..., n_dim)`` of ``dub_orth`` in column order.

    The order is by total degree ``n = n_1 + ... + n_dim``, then lexicographic:
    ``(j,)`` for ``dim = 1``; :func:`hybridge.core.basis.dubiner_pq_order`
    (``for n: for p in 0..n: q = n - p``) for ``dim = 2``; and
    ``for n: for p in 0..n: for q in 0..n-p: r = n - p - q`` for ``dim = 3``.
    """
    if dim == 1:
        return tuple((j,) for j in range(order + 1))
    if dim == 2:
        return tuple(tuple(int(i) for i in pq) for pq in _basis2d.dubiner_pq_order(order))
    return tuple(
        (p, q, n - p - q)
        for n in range(order + 1)
        for p in range(n + 1)
        for q in range(n - p + 1)
    )


@lru_cache(maxsize=None)
def _orthogonal_index_array(dim: int, order: int) -> np.ndarray:
    """Return the int64 array ``A`` ``(order + 1,) * 3``; ``A[n_1, n_2, n_3]`` is the column of that mode, else -1."""
    index = np.full((order + 1,) * 3, -1, dtype=np.int64)
    for column, multi in enumerate(_orthogonal_indices(dim, order)):
        padded = tuple(multi) + (0,) * (3 - dim)
        index[padded] = column
    return index


@lru_cache(maxsize=None)
def _orthonormal_scale_array(dim: int, order: int) -> np.ndarray:
    """Return the float64 array of :func:`orthonormal_scale` (computed once per ``(dim, order)``)."""
    scale = np.empty(comb(order + dim, dim), dtype=np.float64)
    for column, multi in enumerate(_orthogonal_indices(dim, order)):
        factor = 1.0
        partial = 0
        for k, n in enumerate(multi, start=1):
            partial += n
            factor *= (2 * partial + k) / 2.0
        scale[column] = sqrt(factor)
    return scale


@lru_cache(maxsize=None)
def _unit_scale_array(dim: int, order: int) -> np.ndarray:
    """Return ones of shape ``(C(order + dim, dim),)``: the column scale of ``dub_orth`` in :func:`_orthogonal_kernel`."""
    return np.ones(comb(order + dim, dim), dtype=np.float64)


# From this many values (points times modes) on, the basis evaluators of
# hybridge.core.quadrature and hybridge.core.simplex.nodes use the parallel kernels;
# below it, starting the Numba threads costs more than it saves.
PARALLEL_MIN_VALUES = 1 << 16


def use_parallel_kernel(num_points: int, num_modes: int) -> bool:
    """Return whether ``num_points * num_modes`` values are enough to evaluate them with the parallel kernels."""
    return int(num_points) * int(num_modes) >= PARALLEL_MIN_VALUES


def orthonormal_scale(dim, order) -> np.ndarray:
    r"""Return the constants ``c`` ``(C(order + dim, dim),)`` with ``orthonormal = dub_orth * c``, in column order.

    The squared norm of the ``dub_orth`` mode :math:`(n_1, \ldots, n_d)` on the
    reference simplex follows from the Jacobi norms
    :math:`\int_{-1}^1 (1-x)^\beta (P_n^{(\beta,0)})^2 dx = 2^{\beta+1}/(2n+\beta+1)`
    in collapsed coordinates:

    .. math::

        \|\phi\|^2 = \prod_{k=1}^{d} \frac{2}{2N_k + k}, \qquad
        N_k = n_1 + \cdots + n_k,

    so :math:`c = \prod_k \sqrt{(2N_k + k)/2}`. Interval: :math:`\sqrt{(2j+1)/2}`;
    triangle: :math:`\sqrt{(2p+1)(p+q+1)/2}`; tetrahedron:
    :math:`\sqrt{(2p+1)(p+q+1)(2p+2q+2r+3)/4}`. A copy is returned.
    """
    return _orthonormal_scale_array(_check_dim(dim), _check_order(order)).copy()


@lru_cache(maxsize=None)
def _bernstein_exponent_tuples(dim: int, order: int) -> tuple[tuple[int, ...], ...]:
    """Return the Bernstein exponents ``(alpha_0, ..., alpha_dim)`` (powers of ``lambda_0..lambda_dim``) in column order.

    ``dim = 1``: ``(order - j, j)`` for ``j = 0..order``, the order of
    :func:`_bernstein_edge_basis`. ``dim = 2``:
    :func:`hybridge.core.basis.bernstein_exponents`. ``dim = 3``:
    ``for i: for j: for k: (i, j, k, order - i - j - k)``, the 2D order extended.
    """
    if dim == 1:
        return tuple((order - j, j) for j in range(order + 1))
    if dim == 2:
        return tuple(tuple(int(i) for i in e) for e in _basis2d.bernstein_exponents(order))
    return tuple(
        (i, j, k, order - i - j - k)
        for i in range(order + 1)
        for j in range(order + 1 - i)
        for k in range(order + 1 - i - j)
    )


@lru_cache(maxsize=None)
def _bernstein_arrays(dim: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the Bernstein exponents as an int64 array ``(n, dim + 1)`` and the multinomial coefficients ``(n,)``."""
    exponents = np.asarray(_bernstein_exponent_tuples(dim, order), dtype=np.int64).reshape(-1, dim + 1)
    coefficients = np.empty(exponents.shape[0], dtype=np.float64)
    for m, alpha in enumerate(exponents):
        denominator = 1
        for a in alpha:
            denominator *= factorial(int(a))
        coefficients[m] = factorial(order) / denominator
    return np.ascontiguousarray(exponents), coefficients


@lru_cache(maxsize=None)
def _modified_table(dim: int, order: int) -> tuple[np.ndarray, tuple[tuple[str, int, int], ...]]:
    """Return the rows of :func:`_modified_kernel` and the labels ``(kind, entity, local index)``, ``dim`` 1 or 3.

    Degree 0: one constant mode, label ``("interior", 0, 0)``. Degree ``p >= 1``:

    - ``("vertex", v, 0)``: :math:`\\lambda_v`, ``v = 0..dim``;
    - ``("edge", e, k)``, ``dim = 3``: edge ``e = (a, b)`` of
      ``reference_simplex(3).edges``, ``k = 0..p-2``, :math:`n_1 = k + 1`
      (degree ``k + 2``);
    - ``("face", f, k)``, ``dim = 3``: face ``f = (a, b, c)`` of
      ``reference_simplex(3).face_vertices``; the indices :math:`(n_1, n_2)`, both
      ``>= 1``, are ordered by ``n = n_1 + n_2 = 2..p-1``, then :math:`n_1`
      (degree ``n + 1``);
    - ``("interior", 0, k)``: for ``dim = 1`` the modes of the interval with
      :math:`n_1 = k + 1`, for ``dim = 3`` the indices :math:`(n_1, n_2, n_3) \\ge 1`
      ordered by ``n = 3..p-1``, then :math:`n_1`, then :math:`n_2`.

    In the notation of K&S, Appendix D.1.2.4, :math:`n_1` is the index ``p`` of edge
    AB, ``q`` of edges BC and AC, and ``r`` of edges AD, BD and CD;
    :math:`(n_1, n_2)` is ``(p, q)`` on face ABC (face 3), ``(p, r)`` on ABD (face 2)
    and ``(q, r)`` on ACD and BCD (faces 1 and 0); :math:`(n_1, n_2, n_3)` is
    ``(p, q, r)`` in the interior.
    """
    rows: list[tuple[int, ...]] = []
    labels: list[tuple[str, int, int]] = []
    if order == 0:
        rows.append((-1, 0, 0, 0, 0, 0, 0, 0))
        labels.append(("interior", 0, 0))
    elif dim == 1:
        for v in range(2):
            rows.append((0, v, 0, 0, 0, 0, 0, 0))
            labels.append(("vertex", v, 0))
        for k in range(order - 1):
            rows.append((1, 0, 1, 0, 0, k + 1, 0, 0))
            labels.append(("interior", 0, k))
    else:
        simplex = reference_simplex(3)
        for v in range(4):
            rows.append((0, v, 0, 0, 0, 0, 0, 0))
            labels.append(("vertex", v, 0))
        for e, (a, b) in enumerate(simplex.edges):
            for k in range(order - 1):
                rows.append((1, a, b, 0, 0, k + 1, 0, 0))
                labels.append(("edge", e, k))
        for f, (a, b, c) in enumerate(simplex.face_vertices):
            local = 0
            for n in range(2, order):
                for n1 in range(1, n):
                    rows.append((2, a, b, c, 0, n1, n - n1, 0))
                    labels.append(("face", f, local))
                    local += 1
        local = 0
        for n in range(3, order):
            for n1 in range(1, n - 1):
                for n2 in range(1, n - n1):
                    rows.append((3, 0, 1, 2, 3, n1, n2, n - n1 - n2))
                    labels.append(("interior", 0, local))
                    local += 1
    return np.ascontiguousarray(np.asarray(rows, dtype=np.int64)), tuple(labels)


@lru_cache(maxsize=None)
def _legacy_modified_labels(order: int) -> tuple[tuple[str, int, int], ...]:
    """Return ``(kind, entity, local index)`` for the modes of the triangle ``hier_C0`` of :mod:`hybridge.core.basis`.

    The legacy order (:func:`hybridge.core.basis.hierarchical_c0_mode_indexing`) is
    vertices 0, 1, 2; then ``order - 1`` modes on each edge ``e = 0, 1, 2`` (edge
    ``e`` joins vertex ``e`` and vertex ``(e + 1) mod 3``); then the interior modes.
    """
    if order == 0:
        return (("interior", 0, 0),)
    labels = [("vertex", v, 0) for v in range(3)]
    labels += [("edge", e, k) for e in range(3) for k in range(order - 1)]
    labels += [("interior", 0, k) for k in range((order - 1) * (order - 2) // 2)]
    return tuple(labels)


def modal_mode_labels(name: str, dim, order) -> tuple:
    """Return one label per column of :func:`modal_basis`, in column order.

    ``dub_orth`` and ``orthonormal``: the multi-index ``(n_1, ..., n_dim)`` of
    :func:`_orthogonal_indices` (degrees of the Jacobi factors; the total degree is
    their sum). ``bernstein``: the exponents ``(alpha_0, ..., alpha_dim)`` of
    :math:`\\lambda_0, ..., \\lambda_{dim}`. ``hier_C0``: ``(kind, entity, local
    index)`` with ``kind`` one of ``"vertex"``, ``"edge"``, ``"face"``,
    ``"interior"`` (the interior of the interval, triangle or tetrahedron), and
    ``entity`` the vertex, edge or face number of
    :func:`hybridge.core.simplex.reference.reference_simplex` (0 for the interior).
    """
    name, dim, order = _check_name(name), _check_dim(dim), _check_order(order)
    if name in ("dub_orth", "orthonormal"):
        return _orthogonal_indices(dim, order)
    if name == "bernstein":
        return _bernstein_exponent_tuples(dim, order)
    if dim == 2:
        return _legacy_modified_labels(order)
    return _modified_table(dim, order)[1]


def _tabulate(name: str, dim: int, order: int, points: np.ndarray, want_gradients: bool, parallel: bool) -> np.ndarray:
    """Return the values ``(num_points, n)`` or, if ``want_gradients``, the gradients ``(num_points, n, dim)`` of ``name``.

    ``name``, ``dim`` and ``order`` are already checked and ``points`` has shape
    ``(num_points, dim)`` in ``float64``. Triangle ``dub_orth``, ``bernstein`` and
    ``hier_C0`` call :mod:`hybridge.core.basis`; one-dimensional ``dub_orth`` and
    ``bernstein`` values call the edge bases of this module; everything else calls
    the kernels above.
    """
    if dim == 2 and name != "hier_C0":
        kind = "dub_orth" if name == "orthonormal" else name
        legacy = {
            ("dub_orth", False): _basis2d.evaluate_dubiner_basis,
            ("dub_orth", True): _basis2d.evaluate_dubiner_gradients,
            ("bernstein", False): _basis2d.evaluate_bernstein_basis,
            ("bernstein", True): _basis2d.evaluate_bernstein_gradients,
        }
        legacy_parallel = {
            ("dub_orth", False): _basis2d.evaluate_dubiner_basis_parallel,
            ("dub_orth", True): _basis2d.evaluate_dubiner_gradients_parallel,
            ("bernstein", False): _basis2d.evaluate_bernstein_basis_parallel,
            ("bernstein", True): _basis2d.evaluate_bernstein_gradients_parallel,
        }
        evaluate = (legacy_parallel if parallel else legacy)[(kind, want_gradients)]
        result = np.asarray(evaluate(order, points), dtype=np.float64)
    elif dim == 2:
        if want_gradients:
            evaluate = (
                _basis2d.evaluate_hierarchical_c0_gradients_parallel
                if parallel
                else _basis2d.evaluate_hierarchical_c0_gradients
            )
        else:
            evaluate = (
                _basis2d.evaluate_hierarchical_c0_basis_parallel
                if parallel
                else _basis2d.evaluate_hierarchical_c0_basis
            )
        return np.ascontiguousarray(evaluate(order, points), dtype=np.float64)
    elif dim == 1 and not want_gradients and name != "hier_C0":
        edge = _bernstein_edge_basis if name == "bernstein" else _legendre_edge_basis
        result = np.asarray(edge(order, points[:, 0]).T, dtype=np.float64)
    elif name in ("dub_orth", "orthonormal"):
        kernel = _orthogonal_kernel_parallel if parallel else _orthogonal_kernel
        scale = _orthonormal_scale_array(dim, order) if name == "orthonormal" else _unit_scale_array(dim, order)
        values, gradients = kernel(points, order, _orthogonal_index_array(dim, order), scale, want_gradients)
        return gradients if want_gradients else values  # the kernel applied the orthonormal scale
    elif name == "bernstein":
        exponents, coefficients = _bernstein_arrays(dim, order)
        kernel = _bernstein_kernel_parallel if parallel else _bernstein_kernel
        values, gradients = kernel(points, order, exponents, coefficients, want_gradients)
        result = gradients if want_gradients else values
    else:
        kernel = _modified_kernel_parallel if parallel else _modified_kernel
        values, gradients = kernel(points, order, _modified_table(dim, order)[0], want_gradients)
        result = gradients if want_gradients else values
    if name == "orthonormal":
        scale = _orthonormal_scale_array(dim, order)
        result = result * (scale[None, :, None] if want_gradients else scale[None, :])
    return np.ascontiguousarray(result, dtype=np.float64)


def modal_basis(name: str, dim, order, points, *, parallel: bool = False) -> np.ndarray:
    """Values of the modal basis ``name`` of degree ``order`` on the reference ``dim``-simplex at ``points``.

    ``name`` is one of ``MODAL_BASIS_NAMES`` (case is ignored); ``points`` has shape
    ``(num_points, dim)`` (for ``dim = 1`` also ``(num_points,)``). Returns a
    ``float64`` array of shape ``(num_points, modal_dimension(dim, order))``; column
    ``m`` is the mode labelled ``modal_mode_labels(name, dim, order)[m]``. The
    points may lie anywhere, inside or outside the simplex, including the vertices
    and the points where the collapsed coordinates are undefined; the modes are
    polynomials and are evaluated without division. ``parallel=True`` runs the
    points in parallel. Raises ``ValueError`` for an unknown name, a dimension
    other than 1, 2, 3, a negative order or a wrong point shape.
    """
    name, dim, order = _check_name(name), _check_dim(dim), _check_order(order)
    return _tabulate(name, dim, order, _as_points(points, dim), False, bool(parallel))


def modal_gradients(name: str, dim, order, points, *, parallel: bool = False) -> np.ndarray:
    r"""Reference gradients :math:`\partial\phi_m/\partial\xi_k` of the modal basis ``name`` at ``points``.

    Arguments as in :func:`modal_basis`. Returns a ``float64`` array of shape
    ``(num_points, modal_dimension(dim, order), dim)``; entry ``[i, m, k]`` is the
    derivative of mode ``m`` with respect to :math:`\xi_{k+1}` at point ``i``,
    computed from the analytic expressions of the modes (chain rule through the
    barycentric coordinates and the scaled Jacobi recurrence).
    """
    name, dim, order = _check_name(name), _check_dim(dim), _check_order(order)
    return _tabulate(name, dim, order, _as_points(points, dim), True, bool(parallel))


__all__ = [
    "MODAL_BASIS_NAMES",
    "modal_basis",
    "modal_dimension",
    "modal_gradients",
    "modal_mode_labels",
    "orthonormal_scale",
    "scaled_jacobi",
]
