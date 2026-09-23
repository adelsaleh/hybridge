"""Parallel FP64 local ADR operations without coefficient projection.

Small element-local BLAS/LAPACK calls use one BLAS thread; parallelism is
across elements. No fastmath: residual/nonfinite diagnostics remain meaningful.
"""
import numpy as np

from .common import njit, prange


def warmup():
    """Compile/cache the small FP64 signatures outside assembly timing."""
    basis = np.ones((2, 4))
    weighted_mass(np.ones((1, 4)), basis, np.ones(4), np.ones(1))
    advection_matrices(np.ones((1, 4, 2)), np.ones((1, 2, 2)),
                       basis, np.ones((2, 2, 4)), np.ones(4))
    inverse = local_inverses(np.eye(6)[None].copy())
    condense(inverse, np.ones((1, 6, 6)), np.ones((1, 3, 2, 6)),
             np.ones((1, 3, 2, 2)), np.ones((1, 6)), np.ones((1, 3), dtype=np.bool_))


@njit(cache=True, parallel=True)
def weighted_mass(values, basis, weights, jacobians):
    """Integrate sampled weights using O(P Q), not O(P² Q), scratch per thread."""
    ne, nq = values.shape
    p = basis.shape[0]
    out = np.empty((ne, p, p))
    trial = np.ascontiguousarray(basis.T)
    for e in prange(ne):
        test = np.empty((p, nq))
        for i in range(p):
            for q in range(nq):
                test[i, q] = basis[i, q] * weights[q] * values[e, q] * jacobians[e]
        out[e] = test @ trial
    return out


@njit(cache=True, parallel=True)
def advection_matrices(beta, scaled_inverse_transpose, basis, derivatives, weights):
    """Integrate beta dot grad(test) times trial on the unchanged quadrature."""
    ne, nq, _ = beta.shape
    p = basis.shape[0]
    out = np.empty((ne, p, p))
    trial = np.ascontiguousarray(basis.T)
    for e in prange(ne):
        test = np.empty((p, nq))
        for q in range(nq):
            b0 = (beta[e, q, 0] * scaled_inverse_transpose[e, 0, 0]
                  + beta[e, q, 1] * scaled_inverse_transpose[e, 1, 0])
            b1 = (beta[e, q, 0] * scaled_inverse_transpose[e, 0, 1]
                  + beta[e, q, 1] * scaled_inverse_transpose[e, 1, 1])
            for i in range(p):
                test[i, q] = weights[q] * (b0 * derivatives[0, i, q] + b1 * derivatives[1, i, q])
        out[e] = test @ trial
    return out


@njit(cache=True, parallel=True)
def local_inverses(matrices):
    """LAPACK inverses in parallel, retaining the existing reconstruction cache."""
    out = np.empty_like(matrices)
    for e in prange(matrices.shape[0]):
        out[e] = np.linalg.inv(matrices[e])
    return out


@njit(cache=True, parallel=True)
def condense(inverses, boundary, lift, face_mass, source, orientations):
    """Race-free elemental Schur blocks and RHS, with global face orientation."""
    ne, nf, f, n = lift.shape
    blocks = np.empty((ne, nf, nf, f, f))
    rhs = np.empty((ne, nf, f))
    for e in prange(ne):
        response = inverses[e] @ boundary[e]
        forcing = inverses[e] @ source[e]
        for row in range(nf):
            schur = lift[e, row] @ response
            rhs[e, row] = lift[e, row] @ forcing
            for col in range(nf):
                for i in range(f):
                    for j in range(f):
                        local_j = j if orientations[e, col] else f - 1 - j
                        value = -schur[i, col * f + local_j]
                        if row == col:
                            value += face_mass[e, row, i, j]
                        blocks[e, row, col, i, j] = value
    return blocks, rhs
