"""Small reusable Numba helpers for :mod:`hdgfem` kernels.

The functions here are copied in spirit from the legacy ``hdg_numba_helpers``
module, but they deliberately avoid mesh-specific geometry reconstruction.  New
``hdgfem`` kernels receive precomputed geometry from :class:`hdgfem.core.mesh.DGMesh`.
"""

from __future__ import annotations

try:  # pragma: no cover - availability depends on the runtime environment.
    import numba as nb
except ImportError:  # pragma: no cover
    nb = None


NUMBA_AVAILABLE = nb is not None
prange = nb.prange if nb is not None else range


def njit(*args, **kwargs):
    """Return ``numba.njit`` when available, otherwise a no-op decorator."""
    if nb is None:
        def decorator(function):
            return function

        return decorator
    return nb.njit(*args, **kwargs)


@njit(cache=True, inline="always")
def map_edge_dof_bool(is_positive_orientation: bool, local_dof: int, edge_dof: int) -> int:
    """Map a local edge dof through the element-edge orientation."""
    return local_dof if is_positive_orientation else edge_dof - 1 - local_dof


@njit(cache=True, inline="always")
def zero_matrix(matrix):
    """Set a small dense matrix to zero in-place."""
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = 0.0


@njit(cache=True, inline="always")
def zero_vector(vector):
    """Set a small dense vector to zero in-place."""
    for i in range(vector.shape[0]):
        vector[i] = 0.0


@njit(cache=True)
def lu_factor_inplace(matrix, pivots):
    """In-place LU factorization with partial pivoting for small dense blocks."""
    n = matrix.shape[0]
    for k in range(n):
        pivot = k
        max_value = abs(matrix[k, k])
        for i in range(k + 1, n):
            value = abs(matrix[i, k])
            if value > max_value:
                max_value = value
                pivot = i
        pivots[k] = pivot
        if pivot != k:
            for j in range(n):
                tmp = matrix[k, j]
                matrix[k, j] = matrix[pivot, j]
                matrix[pivot, j] = tmp

        diagonal = matrix[k, k]
        if abs(diagonal) < 1e-30:
            diagonal = 1e-30 if diagonal >= 0.0 else -1e-30
            matrix[k, k] = diagonal

        for i in range(k + 1, n):
            matrix[i, k] /= diagonal
            multiplier = matrix[i, k]
            for j in range(k + 1, n):
                matrix[i, j] -= multiplier * matrix[k, j]


@njit(cache=True)
def lu_solve_inplace(lu_matrix, pivots, rhs):
    """Solve ``LU x = rhs`` in-place for one or more RHS columns."""
    n = lu_matrix.shape[0]
    num_rhs = rhs.shape[1]

    for k in range(n):
        pivot = pivots[k]
        if pivot != k:
            for column in range(num_rhs):
                tmp = rhs[k, column]
                rhs[k, column] = rhs[pivot, column]
                rhs[pivot, column] = tmp

    for i in range(n):
        for column in range(num_rhs):
            value = rhs[i, column]
            for j in range(i):
                value -= lu_matrix[i, j] * rhs[j, column]
            rhs[i, column] = value

    for i in range(n - 1, -1, -1):
        for column in range(num_rhs):
            value = rhs[i, column]
            for j in range(i + 1, n):
                value -= lu_matrix[i, j] * rhs[j, column]
            rhs[i, column] = value / lu_matrix[i, i]


__all__ = [
    "NUMBA_AVAILABLE",
    "lu_factor_inplace",
    "lu_solve_inplace",
    "map_edge_dof_bool",
    "njit",
    "prange",
    "zero_matrix",
    "zero_vector",
]
