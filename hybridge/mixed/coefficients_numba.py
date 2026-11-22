"""Element-parallel Numba twins of the host diffusion-tensor table helpers.

Each kernel repeats the NumPy arithmetic of ``hybridge.mixed.coefficients`` per
sample in the same order (no fastmath), so its tables are bitwise identical to
the chunked NumPy path; parallelism is over elements. Tables are ``(K, m, 4)``
with trailing components 00, 01, 10, 11.
"""
from __future__ import annotations

import numpy as np

from hybridge.runtime.optional import njit, prange


@njit(cache=True, parallel=True)
def stack_components(k00, k01, k10, k11):
    """``np.stack((k00, k01, k10, k11), axis=-1)`` of ``(K, m)`` component tables."""
    num_elements, num_points = k00.shape
    out = np.empty((num_elements, num_points, 4))
    for e in prange(num_elements):
        for q in range(num_points):
            out[e, q, 0] = k00[e, q]
            out[e, q, 1] = k01[e, q]
            out[e, q, 2] = k10[e, q]
            out[e, q, 3] = k11[e, q]
    return out


@njit(cache=True, parallel=True)
def worst_status(values):
    """2 if any sample is nonfinite, else 1 if any symmetric part is not positive definite, else 0."""
    num_elements, num_points = values.shape[0], values.shape[1]
    status = np.zeros(num_elements, dtype=np.int64)
    for e in prange(num_elements):
        worst = 0
        for q in range(num_points):
            a, b, c, d = values[e, q, 0], values[e, q, 1], values[e, q, 2], values[e, q, 3]
            if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c) and np.isfinite(d)):
                worst = 2
                break
            scale = max(max(abs(a), abs(b)), max(abs(c), abs(d)))
            if scale == 0.:
                worst = 1
                continue
            a, b, c, d = a / scale, b / scale, c / scale, d / scale
            off = .5 * (b + c)
            if a <= 0. or d <= 0. or a * d - off * off <= 0.:
                worst = 1
        status[e] = worst
    return status.max() if num_elements else 0


@njit(cache=True, parallel=True)
def diffusion_kinds(values):
    """``DIFFUSION_KINDS`` index of each element's samples (same precedence as ``_diffusion_kinds``)."""
    num_elements, num_points = values.shape[0], values.shape[1]
    kinds = np.empty(num_elements, dtype=np.int64)
    for e in prange(num_elements):
        constant = diagonal = isotropic = symmetric = True
        for q in range(num_points):
            for j in range(4):
                if values[e, q, j] != values[e, 0, j]:
                    constant = False
            if values[e, q, 1] != 0. or values[e, q, 2] != 0.:
                diagonal = False
            if values[e, q, 0] != values[e, q, 3]:
                isotropic = False
            if values[e, q, 1] != values[e, q, 2]:
                symmetric = False
        isotropic = isotropic and diagonal
        if constant and isotropic:
            kinds[e] = 0
        elif constant and diagonal:
            kinds[e] = 1
        elif constant:
            kinds[e] = 2
        elif isotropic:
            kinds[e] = 3
        elif diagonal:
            kinds[e] = 4
        elif symmetric:
            kinds[e] = 5
        else:
            kinds[e] = 6
    return kinds


@njit(cache=True, parallel=True)
def inverse_values(values):
    """Scale-aware 2x2 inverses, as ``_inverse``; returns them and whether all are finite."""
    num_elements, num_points = values.shape[0], values.shape[1]
    out = np.empty(values.shape)
    finite = np.ones(num_elements, dtype=np.bool_)
    for e in prange(num_elements):
        for q in range(num_points):
            a, b, c, d = values[e, q, 0], values[e, q, 1], values[e, q, 2], values[e, q, 3]
            scale = max(max(abs(a), abs(b)), max(abs(c), abs(d)))
            a, b, c, d = a / scale, b / scale, c / scale, d / scale
            det = a * d - b * c
            out[e, q, 0] = d / det / scale
            out[e, q, 1] = -b / det / scale
            out[e, q, 2] = -c / det / scale
            out[e, q, 3] = a / det / scale
            for j in range(4):
                if not np.isfinite(out[e, q, j]):
                    finite[e] = False
    return out, finite.all()
