r"""Reference-triangle basis functions for :mod:`hdgfem`.

The package supports three local polynomial bases on
:math:`\hat K = \operatorname{conv}\{(-1,-1),(1,-1),(-1,1)\}`:

``bernstein``
    Total-degree Bernstein basis in barycentric coordinates.
``hier_C0``
    The legacy hierarchical modified C0 basis.
``dub_orth``
    Dubiner/Koornwinder-style hierarchical basis matching the legacy
    collapsed-coordinate formulas.

The public evaluators are small validation wrappers. By default the actual
tabulation is done by serial Numba kernels with ``fastmath=True`` so these
kernels can be called safely from larger parallel transfer/adaptivity loops.
Explicit ``*_parallel`` wrappers are provided for top-level bulk tabulation.
"""

from __future__ import annotations

from functools import lru_cache
from math import factorial

import numba as nb
import numpy as np

BASIS_BERNSTEIN = 0
BASIS_HIERARCHICAL_C0 = 1
BASIS_DUBINER = 2


def _as_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (num_points, 2); got {pts.shape}")
    return np.ascontiguousarray(pts)


def reference_barycentric(points: np.ndarray) -> np.ndarray:
    """Convert reference coordinates to barycentric coordinates."""
    pts = _as_points(points)
    bary = np.empty((pts.shape[0], 3), dtype=np.float64)
    bary[:, 1] = 0.5 * (pts[:, 0] + 1.0)
    bary[:, 2] = 0.5 * (pts[:, 1] + 1.0)
    bary[:, 0] = 1.0 - bary[:, 1] - bary[:, 2]
    return np.ascontiguousarray(bary)


@lru_cache(maxsize=None)
def bernstein_exponents(order: int) -> tuple[tuple[int, int, int], ...]:
    """Return exponent triples ``(i,j,k)`` with ``i+j+k=order``."""
    return tuple(
        (i, j, order - i - j)
        for i in range(order + 1)
        for j in range(order + 1 - i)
    )


@lru_cache(maxsize=None)
def _bernstein_exponent_array(order: int) -> np.ndarray:
    return np.asarray(bernstein_exponents(order), dtype=np.int64)


@lru_cache(maxsize=None)
def bernstein_coefficients(order: int) -> np.ndarray:
    """Return multinomial coefficients for the Bernstein basis."""
    coeffs = [
        factorial(order) / (factorial(i) * factorial(j) * factorial(k))
        for i, j, k in bernstein_exponents(order)
    ]
    return np.ascontiguousarray(np.asarray(coeffs, dtype=np.float64))


@nb.njit(cache=True, fastmath=True)
def _pow_int(base: float, exponent: int) -> float:
    value = 1.0
    for _ in range(exponent):
        value *= base
    return value


@nb.njit(cache=True, fastmath=True)
def _jacobi_p(n: int, alpha: float, beta: float, x: float) -> float:
    if n <= 0:
        return 1.0
    if n == 1:
        return 0.5 * (alpha + beta + 2.0) * x + 0.5 * (alpha - beta)

    p_nm2 = 1.0
    p_nm1 = 0.5 * (alpha + beta + 2.0) * x + 0.5 * (alpha - beta)
    for k in range(1, n):
        kf = float(k)
        a = (2.0 * kf + alpha + beta + 1.0) * (2.0 * kf + alpha + beta + 2.0) / (
            2.0 * (kf + 1.0) * (kf + alpha + beta + 1.0)
        )
        b = (beta * beta - alpha * alpha) * (2.0 * kf + alpha + beta + 1.0) / (
            2.0 * (kf + 1.0) * (kf + alpha + beta + 1.0) * (2.0 * kf + alpha + beta)
        )
        c = (kf + alpha) * (kf + beta) * (2.0 * kf + alpha + beta + 2.0) / (
            (kf + 1.0) * (kf + alpha + beta + 1.0) * (2.0 * kf + alpha + beta)
        )
        p_n = (a * x - b) * p_nm1 - c * p_nm2
        p_nm2 = p_nm1
        p_nm1 = p_n
    return p_nm1


@nb.njit(cache=True, fastmath=True)
def _jacobi_derivative(n: int, alpha: float, beta: float, x: float) -> float:
    if n <= 0:
        return 0.0
    return 0.5 * (n + alpha + beta + 1.0) * _jacobi_p(n - 1, alpha + 1.0, beta + 1.0, x)


@nb.njit(cache=True, fastmath=True)
def _jacobi_derivative_array_kernel(n: int, alpha: float, beta: float, values: np.ndarray) -> np.ndarray:
    # ``nb.prange`` is intentional here: the default dispatcher below is
    # compiled without ``parallel=True``, while the explicit ``*_parallel``
    # dispatcher reuses the same implementation for top-level bulk tabulation.
    out = np.empty_like(values)
    for i in nb.prange(values.size):
        out.flat[i] = _jacobi_derivative(n, alpha, beta, values.flat[i])
    return out


_jacobi_derivative_array_kernel_parallel = nb.njit(parallel=True, cache=True, fastmath=True)(
    getattr(_jacobi_derivative_array_kernel, "py_func", _jacobi_derivative_array_kernel)
)


@nb.njit(cache=True, fastmath=True)
def _bernstein_basis_kernel(points: np.ndarray, exponents: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    num_points = points.shape[0]
    num_modes = exponents.shape[0]
    out = np.empty((num_points, num_modes), dtype=np.float64)
    for q in nb.prange(num_points):
        l1 = 0.5 * (points[q, 0] + 1.0)
        l2 = 0.5 * (points[q, 1] + 1.0)
        l0 = 1.0 - l1 - l2
        for m in range(num_modes):
            out[q, m] = (
                coeffs[m]
                * _pow_int(l0, exponents[m, 0])
                * _pow_int(l1, exponents[m, 1])
                * _pow_int(l2, exponents[m, 2])
            )
    return out


_bernstein_basis_kernel_parallel = nb.njit(parallel=True, cache=True, fastmath=True)(
    getattr(_bernstein_basis_kernel, "py_func", _bernstein_basis_kernel)
)


@nb.njit(cache=True, fastmath=True)
def _bernstein_gradient_kernel(points: np.ndarray, exponents: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    num_points = points.shape[0]
    num_modes = exponents.shape[0]
    out = np.zeros((num_points, num_modes, 2), dtype=np.float64)
    for q in nb.prange(num_points):
        lambda0 = 1.0 - 0.5 * (points[q, 0] + 1.0) - 0.5 * (points[q, 1] + 1.0)
        lambda1 = 0.5 * (points[q, 0] + 1.0)
        lambda2 = 0.5 * (points[q, 1] + 1.0)
        for m in range(num_modes):
            for component in range(3):
                power = exponents[m, component]
                if power == 0:
                    continue
                if component == 0:
                    dlambda_x = -0.5
                    dlambda_y = -0.5
                elif component == 1:
                    dlambda_x = 0.5
                    dlambda_y = 0.0
                else:
                    dlambda_x = 0.0
                    dlambda_y = 0.5
                product = coeffs[m] * power
                for other in range(3):
                    exponent = exponents[m, other]
                    if other == component:
                        exponent -= 1
                    if other == 0:
                        value = lambda0
                    elif other == 1:
                        value = lambda1
                    else:
                        value = lambda2
                    product *= _pow_int(value, exponent)
                out[q, m, 0] += product * dlambda_x
                out[q, m, 1] += product * dlambda_y
    return out


_bernstein_gradient_kernel_parallel = nb.njit(parallel=True, cache=True, fastmath=True)(
    getattr(_bernstein_gradient_kernel, "py_func", _bernstein_gradient_kernel)
)


def evaluate_bernstein_basis(order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate all degree-``order`` Bernstein basis functions."""
    return np.ascontiguousarray(
        _bernstein_basis_kernel(
            _as_points(points),
            _bernstein_exponent_array(order),
            bernstein_coefficients(order),
        )
    )


def evaluate_bernstein_basis_parallel(order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate the Bernstein basis with top-level point parallelism."""
    return np.ascontiguousarray(
        _bernstein_basis_kernel_parallel(
            _as_points(points),
            _bernstein_exponent_array(order),
            bernstein_coefficients(order),
        )
    )


def evaluate_bernstein_gradients(order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate reference-coordinate gradients of the Bernstein basis."""
    return np.ascontiguousarray(
        _bernstein_gradient_kernel(
            _as_points(points),
            _bernstein_exponent_array(order),
            bernstein_coefficients(order),
        )
    )


def evaluate_bernstein_gradients_parallel(order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate Bernstein gradients with top-level point parallelism."""
    return np.ascontiguousarray(
        _bernstein_gradient_kernel_parallel(
            _as_points(points),
            _bernstein_exponent_array(order),
            bernstein_coefficients(order),
        )
    )


def jacobi_derivative(n: int, alpha: float, beta: float, x: np.ndarray) -> np.ndarray:
    """Vectorized wrapper for :math:`dP_n^{(\alpha,\beta)}/dx`."""
    return _jacobi_derivative_array_kernel(int(n), float(alpha), float(beta), np.asarray(x, dtype=np.float64))


def jacobi_derivative_parallel(n: int, alpha: float, beta: float, x: np.ndarray) -> np.ndarray:
    """Evaluate :math:`dP_n^{(\alpha,\beta)}/dx` with top-level parallelism."""
    return _jacobi_derivative_array_kernel_parallel(
        int(n),
        float(alpha),
        float(beta),
        np.asarray(x, dtype=np.float64),
    )


@lru_cache(maxsize=None)
def hierarchical_c0_mode_indexing(order: int) -> np.ndarray:
    """Return the legacy hierarchical C0 mode ordering."""
    if order < 0:
        raise ValueError("order must be nonnegative")
    if order == 0:
        return np.array(((0, 0),), dtype=np.int64)

    num_modes = (order + 1) * (order + 2) // 2
    indices = np.zeros((num_modes, 2), dtype=np.int64)
    indices[0] = (0, 0)
    indices[1] = (order, 0)
    indices[2] = (0, order)

    for i in range(order - 1):
        indices[3 + i] = (i + 1, 0)
        indices[3 + (order - 1) + i] = (order - i - 1, i + 1)
        indices[3 + 2 * (order - 1) + i] = (0, order - 1 - i)

    if order >= 3:
        cursor = 3 * order
        for q in range(1, order):
            for p in range(1, order - q):
                indices[cursor] = (p, q)
                cursor += 1
    return np.ascontiguousarray(indices)


@nb.njit(cache=True, fastmath=True)
def _hierarchical_c0_basis_kernel(order: int, points: np.ndarray, modes: np.ndarray) -> np.ndarray:
    num_points = points.shape[0]
    num_modes = modes.shape[0]
    out = np.empty((num_points, num_modes), dtype=np.float64)
    if order == 0:
        for q in nb.prange(num_points):
            out[q, 0] = 1.0
        return out

    for qpt in nb.prange(num_points):
        xi1 = points[qpt, 0]
        xi2 = points[qpt, 1]
        denom = 1.0 - xi2
        eta = -1.0 if abs(denom) <= 1e-14 else 2.0 * (1.0 + xi1) / denom - 1.0
        w_em = 0.5 * (1.0 - eta)
        w_ep = 0.5 * (1.0 + eta)
        w_xm = 0.5 * (1.0 - xi2)
        w_xp = 0.5 * (1.0 + xi2)
        for m in range(num_modes):
            p = modes[m, 0]
            r = modes[m, 1]
            if p == 0 and r == 0:
                out[qpt, m] = w_em * w_xm
            elif p == order and r == 0:
                out[qpt, m] = w_ep * w_xm
            elif p == 0 and r == order:
                out[qpt, m] = w_xp
            elif r == 0:
                out[qpt, m] = (w_em * w_ep) * _jacobi_p(p - 1, 1.0, 1.0, eta) * _pow_int(w_xm, p + 1)
            elif r == order - p:
                out[qpt, m] = w_ep * (w_xm * w_xp) * _jacobi_p(r - 1, 1.0, 1.0, xi2)
            elif p == 0:
                out[qpt, m] = w_em * (w_xm * w_xp) * _jacobi_p(r - 1, 1.0, 1.0, xi2)
            else:
                out[qpt, m] = (
                    (w_em * w_ep)
                    * _jacobi_p(p - 1, 1.0, 1.0, eta)
                    * _pow_int(w_xm, p + 1)
                    * w_xp
                    * _jacobi_p(r - 1, 2.0 * p - 1.0, 1.0, xi2)
                )
    return out


_hierarchical_c0_basis_kernel_parallel = nb.njit(parallel=True, cache=True, fastmath=True)(
    getattr(_hierarchical_c0_basis_kernel, "py_func", _hierarchical_c0_basis_kernel)
)


@nb.njit(cache=True, fastmath=True)
def _hierarchical_c0_gradient_kernel(order: int, points: np.ndarray, modes: np.ndarray) -> np.ndarray:
    num_points = points.shape[0]
    num_modes = modes.shape[0]
    out = np.empty((num_points, num_modes, 2), dtype=np.float64)
    if order == 0:
        out[:, :, :] = 0.0
        return out

    for qpt in nb.prange(num_points):
        xi1 = points[qpt, 0]
        xi2 = points[qpt, 1]
        denom = 1.0 - xi2
        if abs(denom) <= 1e-14:
            eta = -1.0
            eta_x1 = 0.0
            eta_x2 = 0.0
        else:
            eta = 2.0 * (1.0 + xi1) / denom - 1.0
            eta_x1 = 2.0 / denom
            eta_x2 = 2.0 * (1.0 + xi1) / (denom * denom)
        w_em = 0.5 * (1.0 - eta)
        w_ep = 0.5 * (1.0 + eta)
        w_xm = 0.5 * (1.0 - xi2)
        w_xp = 0.5 * (1.0 + xi2)
        d = w_xm * w_xp
        d_dxi2 = -0.5 * xi2

        for m in range(num_modes):
            p = modes[m, 0]
            r = modes[m, 1]
            if p == 0 and r == 0:
                out[qpt, m, 0] = -0.5
                out[qpt, m, 1] = -0.5
            elif p == order and r == 0:
                out[qpt, m, 0] = 0.5
                out[qpt, m, 1] = 0.0
            elif p == 0 and r == order:
                out[qpt, m, 0] = 0.0
                out[qpt, m, 1] = 0.5
            elif r == 0:
                jp = _jacobi_p(p - 1, 1.0, 1.0, eta)
                djp = _jacobi_derivative(p - 1, 1.0, 1.0, eta)
                term = (-eta / 2.0) * jp + (1.0 - eta * eta) / 4.0 * djp
                out[qpt, m, 0] = eta_x1 * term * _pow_int(w_xm, p + 1)
                a = w_em * w_ep * jp
                da = (-0.5 * eta * eta_x2) * jp + ((1.0 - eta * eta) / 4.0) * eta_x2 * djp
                b = _pow_int(w_xm, p + 1)
                db = -(p + 1) * 0.5 * _pow_int(w_xm, p)
                out[qpt, m, 1] = da * b + a * db
            elif r == order - p:
                jq = _jacobi_p(r - 1, 1.0, 1.0, xi2)
                djq = _jacobi_derivative(r - 1, 1.0, 1.0, xi2)
                out[qpt, m, 0] = 0.5 * eta_x1 * d * jq
                out[qpt, m, 1] = (0.5 * eta_x2) * d * jq + w_ep * d_dxi2 * jq + w_ep * d * djq
            elif p == 0:
                jq = _jacobi_p(r - 1, 1.0, 1.0, xi2)
                djq = _jacobi_derivative(r - 1, 1.0, 1.0, xi2)
                out[qpt, m, 0] = -0.5 * eta_x1 * d * jq
                out[qpt, m, 1] = (-0.5 * eta_x2) * d * jq + w_em * d_dxi2 * jq + w_em * d * djq
            else:
                jp = _jacobi_p(p - 1, 1.0, 1.0, eta)
                djp = _jacobi_derivative(p - 1, 1.0, 1.0, eta)
                jq = _jacobi_p(r - 1, 2.0 * p - 1.0, 1.0, xi2)
                djq = _jacobi_derivative(r - 1, 2.0 * p - 1.0, 1.0, xi2)
                term = (-eta / 2.0) * jp + (1.0 - eta * eta) / 4.0 * djp
                out[qpt, m, 0] = eta_x1 * term * _pow_int(w_xm, p + 1) * w_xp * jq
                w = w_em * w_ep
                dw = -0.5 * eta_x2 * w_ep + 0.5 * eta_x2 * w_em
                a = w * jp
                da = dw * jp + w * djp * eta_x2
                b = _pow_int(w_xm, p + 1)
                db = -(p + 1) * 0.5 * _pow_int(w_xm, p)
                out[qpt, m, 1] = da * b * w_xp * jq + a * db * w_xp * jq + a * b * 0.5 * jq + a * b * w_xp * djq
    return out


_hierarchical_c0_gradient_kernel_parallel = nb.njit(parallel=True, cache=True, fastmath=True)(
    getattr(_hierarchical_c0_gradient_kernel, "py_func", _hierarchical_c0_gradient_kernel)
)


def evaluate_hierarchical_c0_basis(order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate the legacy hierarchical modified C0 basis."""
    return np.ascontiguousarray(
        _hierarchical_c0_basis_kernel(
            int(order),
            _as_points(points),
            hierarchical_c0_mode_indexing(int(order)),
        )
    )


def evaluate_hierarchical_c0_basis_parallel(order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate the hierarchical C0 basis with top-level point parallelism."""
    return np.ascontiguousarray(
        _hierarchical_c0_basis_kernel_parallel(
            int(order),
            _as_points(points),
            hierarchical_c0_mode_indexing(int(order)),
        )
    )


def evaluate_hierarchical_c0_gradients(order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate reference-coordinate gradients of the hierarchical C0 basis."""
    return np.ascontiguousarray(
        _hierarchical_c0_gradient_kernel(
            int(order),
            _as_points(points),
            hierarchical_c0_mode_indexing(int(order)),
        )
    )


def evaluate_hierarchical_c0_gradients_parallel(order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate hierarchical C0 gradients with top-level point parallelism."""
    return np.ascontiguousarray(
        _hierarchical_c0_gradient_kernel_parallel(
            int(order),
            _as_points(points),
            hierarchical_c0_mode_indexing(int(order)),
        )
    )


@lru_cache(maxsize=None)
def dubiner_pq_order(order: int) -> tuple[tuple[int, int], ...]:
    """Return the legacy Dubiner ordering ``(p,q)`` by total degree."""
    return tuple((p, n - p) for n in range(order + 1) for p in range(n + 1))


@lru_cache(maxsize=None)
def _dubiner_pq_array(order: int) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(dubiner_pq_order(order), dtype=np.int64))


@nb.njit(cache=True, fastmath=True)
def _dubiner_basis_kernel(points: np.ndarray, pq: np.ndarray) -> np.ndarray:
    num_points = points.shape[0]
    num_modes = pq.shape[0]
    out = np.empty((num_points, num_modes), dtype=np.float64)
    for qpt in nb.prange(num_points):
        xi = points[qpt, 0]
        eta = points[qpt, 1]
        at_vertex = abs(eta - 1.0) <= 1e-14 and abs(xi + 1.0) <= 1e-14
        a = -1.0
        fac_base = 0.0
        if not at_vertex:
            one_minus_eta = 1.0 - eta
            a = 2.0 * (1.0 + xi) / one_minus_eta - 1.0
            fac_base = 0.5 * one_minus_eta
        for m in range(num_modes):
            p = pq[m, 0]
            r = pq[m, 1]
            if at_vertex:
                out[qpt, m] = float(r + 1) if p == 0 else 0.0
            else:
                out[qpt, m] = (
                    _jacobi_p(p, 0.0, 0.0, a)
                    * _pow_int(fac_base, p)
                    * _jacobi_p(r, 2.0 * p + 1.0, 0.0, eta)
                )
    return out


_dubiner_basis_kernel_parallel = nb.njit(parallel=True, cache=True, fastmath=True)(
    getattr(_dubiner_basis_kernel, "py_func", _dubiner_basis_kernel)
)


@nb.njit(cache=True, fastmath=True)
def _dubiner_gradient_kernel(points: np.ndarray, pq: np.ndarray) -> np.ndarray:
    num_points = points.shape[0]
    num_modes = pq.shape[0]
    out = np.zeros((num_points, num_modes, 2), dtype=np.float64)
    for qpt in nb.prange(num_points):
        xi = points[qpt, 0]
        eta = points[qpt, 1]
        at_vertex = abs(eta - 1.0) <= 1e-14 and abs(xi + 1.0) <= 1e-14
        a = -1.0
        fac_base = 0.0
        da_dxi = 0.0
        da_deta = 0.0
        one_minus_eta = 1.0 - eta
        if not at_vertex:
            a = 2.0 * (1.0 + xi) / one_minus_eta - 1.0
            fac_base = 0.5 * one_minus_eta
            da_dxi = 2.0 / one_minus_eta
            da_deta = (a + 1.0) / one_minus_eta
        for m in range(num_modes):
            p = pq[m, 0]
            r = pq[m, 1]
            if at_vertex:
                out[qpt, m, 0] = float(r + 1) if p == 1 else 0.0
                if p == 0:
                    out[qpt, m, 1] = 0.5 * r * (r + 2)
                elif p == 1:
                    out[qpt, m, 1] = 0.5 * (r + 1)
                else:
                    out[qpt, m, 1] = 0.0
            else:
                fac = _pow_int(fac_base, p)
                t = _jacobi_p(p, 0.0, 0.0, a)
                dt = _jacobi_derivative(p, 0.0, 0.0, a)
                u = _jacobi_p(r, 2.0 * p + 1.0, 0.0, eta)
                du = _jacobi_derivative(r, 2.0 * p + 1.0, 0.0, eta)
                dfac = 0.0 if p == 0 else -(p / one_minus_eta) * fac
                out[qpt, m, 0] = dt * da_dxi * fac * u
                out[qpt, m, 1] = dt * da_deta * fac * u + t * dfac * u + t * fac * du
    return out


_dubiner_gradient_kernel_parallel = nb.njit(parallel=True, cache=True, fastmath=True)(
    getattr(_dubiner_gradient_kernel, "py_func", _dubiner_gradient_kernel)
)


def evaluate_dubiner_basis(order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate all Dubiner basis functions."""
    return np.ascontiguousarray(_dubiner_basis_kernel(_as_points(points), _dubiner_pq_array(int(order))))


def evaluate_dubiner_basis_parallel(order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate all Dubiner basis functions with top-level point parallelism."""
    return np.ascontiguousarray(_dubiner_basis_kernel_parallel(_as_points(points), _dubiner_pq_array(int(order))))


def evaluate_dubiner_gradients(order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate reference-coordinate gradients of all Dubiner basis functions."""
    return np.ascontiguousarray(_dubiner_gradient_kernel(_as_points(points), _dubiner_pq_array(int(order))))


def evaluate_dubiner_gradients_parallel(order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate Dubiner gradients with top-level point parallelism."""
    return np.ascontiguousarray(_dubiner_gradient_kernel_parallel(_as_points(points), _dubiner_pq_array(int(order))))


@nb.njit(cache=True, fastmath=True)
def _eval_bernstein_modal_value(
        coeffs: np.ndarray,
        xi: float,
        eta: float,
        exponents: np.ndarray,
        multinomial: np.ndarray,
) -> float:
    lambda1 = 0.5 * (xi + 1.0)
    lambda2 = 0.5 * (eta + 1.0)
    lambda0 = 1.0 - lambda1 - lambda2
    value = 0.0
    for m in range(coeffs.size):
        value += (
            coeffs[m]
            * multinomial[m]
            * _pow_int(lambda0, exponents[m, 0])
            * _pow_int(lambda1, exponents[m, 1])
            * _pow_int(lambda2, exponents[m, 2])
        )
    return value


@nb.njit(cache=True, fastmath=True)
def _eval_hierarchical_c0_mode_value(order: int, xi1: float, xi2: float, p: int, r: int) -> float:
    if order == 0:
        return 1.0
    denom = 1.0 - xi2
    eta = -1.0 if abs(denom) <= 1e-14 else 2.0 * (1.0 + xi1) / denom - 1.0
    w_em = 0.5 * (1.0 - eta)
    w_ep = 0.5 * (1.0 + eta)
    w_xm = 0.5 * (1.0 - xi2)
    w_xp = 0.5 * (1.0 + xi2)
    if p == 0 and r == 0:
        return w_em * w_xm
    if p == order and r == 0:
        return w_ep * w_xm
    if p == 0 and r == order:
        return w_xp
    if r == 0:
        return (w_em * w_ep) * _jacobi_p(p - 1, 1.0, 1.0, eta) * _pow_int(w_xm, p + 1)
    if r == order - p:
        return w_ep * (w_xm * w_xp) * _jacobi_p(r - 1, 1.0, 1.0, xi2)
    if p == 0:
        return w_em * (w_xm * w_xp) * _jacobi_p(r - 1, 1.0, 1.0, xi2)
    return (
        (w_em * w_ep)
        * _jacobi_p(p - 1, 1.0, 1.0, eta)
        * _pow_int(w_xm, p + 1)
        * w_xp
        * _jacobi_p(r - 1, 2.0 * p - 1.0, 1.0, xi2)
    )


@nb.njit(cache=True, fastmath=True)
def _eval_hierarchical_c0_modal_value(
        coeffs: np.ndarray,
        xi: float,
        eta: float,
        order: int,
        modes: np.ndarray,
) -> float:
    value = 0.0
    for m in range(coeffs.size):
        value += coeffs[m] * _eval_hierarchical_c0_mode_value(order, xi, eta, modes[m, 0], modes[m, 1])
    return value


@nb.njit(cache=True, fastmath=True)
def _eval_dubiner_mode_value(xi: float, eta: float, p: int, r: int) -> float:
    at_vertex = abs(eta - 1.0) <= 1e-14 and abs(xi + 1.0) <= 1e-14
    if at_vertex:
        return float(r + 1) if p == 0 else 0.0
    one_minus_eta = 1.0 - eta
    a = 2.0 * (1.0 + xi) / one_minus_eta - 1.0
    fac_base = 0.5 * one_minus_eta
    return _jacobi_p(p, 0.0, 0.0, a) * _pow_int(fac_base, p) * _jacobi_p(r, 2.0 * p + 1.0, 0.0, eta)


@nb.njit(cache=True, fastmath=True)
def _eval_dubiner_modal_value(coeffs: np.ndarray, xi: float, eta: float, pq: np.ndarray) -> float:
    value = 0.0
    for m in range(coeffs.size):
        value += coeffs[m] * _eval_dubiner_mode_value(xi, eta, pq[m, 0], pq[m, 1])
    return value


@nb.njit(cache=True, fastmath=True)
def evaluate_modal_value(
        basis_kind: int,
        order: int,
        coeffs: np.ndarray,
        xi: float,
        eta: float,
        bernstein_exps: np.ndarray,
        bernstein_coeffs: np.ndarray,
        hierarchical_modes: np.ndarray,
        dubiner_pq: np.ndarray,
) -> float:
    """Evaluate one modal DG field at one reference point."""
    if basis_kind == BASIS_BERNSTEIN:
        return _eval_bernstein_modal_value(coeffs, xi, eta, bernstein_exps, bernstein_coeffs)
    if basis_kind == BASIS_HIERARCHICAL_C0:
        return _eval_hierarchical_c0_modal_value(coeffs, xi, eta, order, hierarchical_modes)
    return _eval_dubiner_modal_value(coeffs, xi, eta, dubiner_pq)


def basis_kind(name: str) -> int:
    """Return the integer basis id used by compiled transfer kernels."""
    normalized = name.strip().lower()
    if normalized in {"bernstein", "bern"}:
        return BASIS_BERNSTEIN
    if normalized in {"hier_c0", "hierarchical_c0", "c0", "cg"}:
        return BASIS_HIERARCHICAL_C0
    if normalized in {"dub_orth", "dubiner", "dubiner_orthonormal"}:
        return BASIS_DUBINER
    raise ValueError(f"unsupported basis type {name!r}")


def modal_eval_payload(name: str, order: int):
    """Return compact basis metadata for compiled pointwise modal evaluation."""
    kind = basis_kind(name)
    empty_i3 = np.empty((0, 3), dtype=np.int64)
    empty_i2 = np.empty((0, 2), dtype=np.int64)
    empty_f = np.empty(0, dtype=np.float64)
    if kind == BASIS_BERNSTEIN:
        return kind, _bernstein_exponent_array(order), bernstein_coefficients(order), empty_i2, empty_i2
    if kind == BASIS_HIERARCHICAL_C0:
        return kind, empty_i3, empty_f, hierarchical_c0_mode_indexing(order), empty_i2
    return kind, empty_i3, empty_f, empty_i2, _dubiner_pq_array(order)


def basis_values(reference, reference_points: np.ndarray) -> np.ndarray:
    """Evaluate all basis functions for ``reference`` at ``reference_points``."""
    return reference.basis_at(reference_points)


def gradient_values(reference, reference_points: np.ndarray) -> np.ndarray:
    """Evaluate all reference gradients for ``reference`` at ``reference_points``."""
    return reference.gradients_at(reference_points)


__all__ = [
    "BASIS_BERNSTEIN",
    "BASIS_DUBINER",
    "BASIS_HIERARCHICAL_C0",
    "basis_values",
    "basis_kind",
    "bernstein_coefficients",
    "bernstein_exponents",
    "dubiner_pq_order",
    "evaluate_bernstein_basis",
    "evaluate_bernstein_basis_parallel",
    "evaluate_bernstein_gradients",
    "evaluate_bernstein_gradients_parallel",
    "evaluate_dubiner_basis",
    "evaluate_dubiner_basis_parallel",
    "evaluate_dubiner_gradients",
    "evaluate_dubiner_gradients_parallel",
    "evaluate_hierarchical_c0_basis",
    "evaluate_hierarchical_c0_basis_parallel",
    "evaluate_hierarchical_c0_gradients",
    "evaluate_hierarchical_c0_gradients_parallel",
    "evaluate_modal_value",
    "gradient_values",
    "hierarchical_c0_mode_indexing",
    "jacobi_derivative",
    "jacobi_derivative_parallel",
    "modal_eval_payload",
    "reference_barycentric",
]
