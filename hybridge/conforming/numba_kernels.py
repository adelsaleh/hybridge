"""Numba kernels of the condensed CG Poisson setup: one element per iteration, parallel over elements."""
from __future__ import annotations

import numpy as np
from numba import prange

from hybridge.hdg.numba_common import cholesky_factor_inplace, cholesky_solve_inplace
from hybridge.runtime.optional import njit

__all__ = ["condense_cg_elements"]


@njit(cache=True, parallel=True)
def condense_cg_elements(metric, jacobian, reference, num_skeleton_local, schur, bubble_inverse, bubble_coupling,
                         status):
    """Element stiffness, interior Cholesky and Schur complement onto the skeleton nodes.

    ``K_e = |J| (M00 S00 + M01 (S01 + S10) + M11 S11)`` with ``M = A^{-1} A^{-T}`` (``metric``) and the
    reference tables ``reference[a, b] = S^{ab}``; the first ``num_skeleton_local`` local nodes are the
    vertex and edge nodes. Writes ``schur = K_bb - K_bi K_ii^{-1} K_ib``, ``bubble_inverse = K_ii^{-1}``
    and ``bubble_coupling = K_ii^{-1} K_ib``; ``status[e]`` is the Cholesky status (0 on success).
    """
    num_elements = metric.shape[0]
    n = reference.shape[2]
    nb = num_skeleton_local
    ni = n - nb
    for e in prange(num_elements):
        m00 = metric[e, 0, 0]
        m01 = metric[e, 0, 1]
        m11 = metric[e, 1, 1]
        jac = jacobian[e]
        local = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                local[i, j] = jac * (m00 * reference[0, 0, i, j]
                                     + m01 * (reference[0, 1, i, j] + reference[1, 0, i, j])
                                     + m11 * reference[1, 1, i, j])
        if ni == 0:
            for i in range(nb):
                for j in range(nb):
                    schur[e, i, j] = local[i, j]
            status[e] = 0
            continue
        factor = np.empty((ni, ni))
        for i in range(ni):
            for j in range(ni):
                factor[i, j] = local[nb + i, nb + j]
        code = cholesky_factor_inplace(factor, 1e-10)
        status[e] = code
        if code != 0:
            continue
        rhs = np.zeros((ni, nb + ni))
        for i in range(ni):
            for j in range(nb):
                rhs[i, j] = local[nb + i, j]
            rhs[i, nb + i] = 1.0
        cholesky_solve_inplace(factor, rhs)
        for i in range(ni):
            for j in range(nb):
                bubble_coupling[e, i, j] = rhs[i, j]
            for j in range(ni):
                bubble_inverse[e, i, j] = rhs[i, nb + j]
        for i in range(nb):
            for j in range(nb):
                value = local[i, j]
                for m in range(ni):
                    value -= local[i, nb + m] * rhs[m, j]
                schur[e, i, j] = value
