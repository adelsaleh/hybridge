"""Inline Numba helpers shared by the DR and ADR mixed HDG kernels.

DR is ADR with beta = 0: both condense the same ``[u, q_x, q_y]`` local system
through the scalar Schur block. The families differ only in how the ``u`` row
is built (DR: exact projected reaction and tau face tables; ADR: sampled
reaction, advection and ``tau_total`` tables), so the condensation and the
column solve live here once.
"""

from __future__ import annotations

from hdgfem.hdg.numba_common import (
    cholesky_solve_inplace,
    lu_factor_inplace,
    lu_solve_inplace,
)
from hdgfem.runtime.optional import njit


@njit(cache=True, inline="always", fastmath=True)
def _finish_diffusion_condensation(
        schur, d0, d1, mn0, mn1, kd0, kd1, mass_inverse, jac, diffusion,
):
    """Add the mixed diffusive contribution to the scalar Schur block."""
    nel = schur.shape[0]
    for i in range(nel):
        for j in range(nel):
            value0 = 0.0
            value1 = 0.0
            for k in range(nel):
                value0 += mass_inverse[i, k] * d0[k, j]
                value1 += mass_inverse[i, k] * d1[k, j]
            kd0[i, j] = value0
            kd1[i, j] = value1
    jac_inv = diffusion / jac
    for i in range(nel):
        for j in range(nel):
            value0 = 0.0
            value1 = 0.0
            for k in range(nel):
                value0 += mn0[i, k] * kd0[k, j]
                value1 += mn1[i, k] * kd1[k, j]
            schur[i, j] += jac_inv * (value0 + value1)


@njit(cache=True, inline="always", fastmath=True)
def _solve_mixed_columns(
        local_columns,
        schur_matrix,
        d0,
        d1,
        mn0,
        mn1,
        rhs0,
        rhs1,
        rhs2,
        red_rhs,
        tmp1,
        tmp2,
        pivots,
        mass_inverse,
        aff_jac,
        factor_kind=0,
        cached_factor=None,
        cached_pivots=None,
        factor_element=0,
        diffusion=1.0,
):
    """Apply the mixed local inverse to all RHS columns (DR and ADR).

    Condenses ``q`` into the scalar Schur block ``schur_matrix``, solves for
    ``u`` (freshly factored, or with a cached LU/Cholesky factor) and recovers
    ``q = (kappa / J) M^{-1} (D u - rhs_q)`` for scalar ``kappa = diffusion``.
    """
    nel = mass_inverse.shape[0]
    ncols = rhs0.shape[1]
    jac_inverse = diffusion / aff_jac

    for i in range(nel):
        for column in range(ncols):
            value1 = 0.0
            value2 = 0.0
            for k in range(nel):
                value1 += mass_inverse[i, k] * rhs1[k, column]
                value2 += mass_inverse[i, k] * rhs2[k, column]
            tmp1[i, column] = value1
            tmp2[i, column] = value2

    for i in range(nel):
        for column in range(ncols):
            value = rhs0[i, column]
            acc0 = 0.0
            acc1 = 0.0
            for k in range(nel):
                acc0 += mn0[i, k] * tmp1[k, column]
                acc1 += mn1[i, k] * tmp2[k, column]
            red_rhs[i, column] = value + jac_inverse * (acc0 + acc1)

    if cached_factor is None:
        lu_factor_inplace(schur_matrix, pivots)
        lu_solve_inplace(schur_matrix, pivots, red_rhs)
    elif factor_kind == 2:
        cholesky_solve_inplace(cached_factor[factor_element], red_rhs)
    elif cached_pivots is not None:
        lu_solve_inplace(cached_factor[factor_element], cached_pivots[factor_element], red_rhs)

    for i in range(nel):
        for column in range(ncols):
            u_value = red_rhs[i, column]
            local_columns[i, column] = u_value

    for i in range(nel):
        for column in range(ncols):
            value0 = 0.0
            value1 = 0.0
            for j in range(nel):
                value0 += d0[i, j] * red_rhs[j, column]
                value1 += d1[i, j] * red_rhs[j, column]
            tmp1[i, column] = value0 - rhs1[i, column]
            tmp2[i, column] = value1 - rhs2[i, column]

    for i in range(nel):
        for column in range(ncols):
            value0 = 0.0
            value1 = 0.0
            for k in range(nel):
                value0 += mass_inverse[i, k] * tmp1[k, column]
                value1 += mass_inverse[i, k] * tmp2[k, column]
            local_columns[nel + i, column] = jac_inverse * value0
            local_columns[2 * nel + i, column] = jac_inverse * value1
