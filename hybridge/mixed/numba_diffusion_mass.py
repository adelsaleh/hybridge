"""Small element-local inverse diffusion mass actions for fused HDG kernels."""
import numpy as np
from hybridge.runtime.optional import njit
from hybridge.hdg.numba_common import (
    lu_factor_inplace,
    lu_solve_inplace,
    cholesky_factor_inplace,
    cholesky_solve_inplace,
)


@njit(cache=True)
def factor_diffusion_mass(factor, pivots, inverse_values, kind, basis, weights, jac):
    """Factor weighted inverse-diffusion mass blocks for one variable element.

    Isotropic and diagonal tensors need one or two scalar Cholesky factors.
    Symmetric full tensors use a coupled Cholesky factor; general tensors use
    pivoted LU. Return Cholesky failure status without raising inside prange.
    """
    n = basis.shape[0]
    for i in range(n):
        for j in range(n):
            a = b = c = d = 0.0
            for q in range(weights.size):
                w = jac * weights[q] * basis[i, q] * basis[j, q]
                a += w * inverse_values[q, 0]
                if kind != 3:
                    d += w * inverse_values[q, 3]
                if kind >= 5:
                    b += w * inverse_values[q, 1]
                    c += w * inverse_values[q, 2]
            factor[i, j] = a
            if kind == 4:
                factor[i, n+j] = d
            elif kind >= 5:
                factor[i, n+j] = b
                factor[n+i, j] = c
                factor[n+i, n+j] = d
    if kind == 3:
        return cholesky_factor_inplace(factor)
    if kind == 4:
        status = cholesky_factor_inplace(factor[:, :n])
        if status != 0:
            return status
        return cholesky_factor_inplace(factor[:, n:])
    if kind == 5:
        return cholesky_factor_inplace(factor)
    lu_factor_inplace(factor, pivots)
    return 0


@njit(cache=True)
def apply_inverse_diffusion_mass(rhs, work, kind, factor, pivots, tensor, mass_inverse, jac):
    """Apply the flux mass inverse in-place to stacked x/y RHS columns."""
    n = mass_inverse.shape[0]
    if kind <= 2:
        for i in range(n):
            for column in range(rhs.shape[1]):
                x = y = 0.0
                for j in range(n):
                    x += mass_inverse[i, j] * rhs[j, column]
                    y += mass_inverse[i, j] * rhs[n+j, column]
                work[i, column] = (tensor[0]*x + tensor[1]*y) / jac
                work[n+i, column] = (tensor[2]*x + tensor[3]*y) / jac
        rhs[:, :] = work
    elif kind == 3:
        cholesky_solve_inplace(factor, rhs[:n])
        cholesky_solve_inplace(factor, rhs[n:])
    elif kind == 4:
        cholesky_solve_inplace(factor[:, :n], rhs[:n])
        cholesky_solve_inplace(factor[:, n:], rhs[n:])
    elif kind == 5:
        cholesky_solve_inplace(factor, rhs)
    else:
        lu_solve_inplace(factor, pivots, rhs)
