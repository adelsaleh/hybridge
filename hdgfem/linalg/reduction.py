"""hdgfem.linalg.reduction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from hdgfem.runtime.precision import REAL_DTYPE
from dataclasses import dataclass
from hdgfem.linalg.results import validate_global_system_inputs

from hdgfem.runtime.optional import require_cupy



@dataclass(frozen=True)
class KnownDofReduction:
    """Reduced COO system obtained by eliminating prescribed dofs."""

    rows: NDArray
    cols: NDArray
    data: NDArray
    rhs: NDArray
    free_mask: NDArray
    known_mask: NDArray
    known_values: NDArray
    old_to_new: NDArray


def eliminate_known_dofs(
    row_indices: NDArray,
    col_indices: NDArray,
    matrix_values: NDArray,
    rhs: NDArray,
    known_mask: NDArray,
    known_values: NDArray,
) -> KnownDofReduction:
    r"""Eliminate prescribed unknowns from a COO linear system.

    Given :math:`Ax=b` and known entries :math:`x_k=g`, this returns the
    reduced free-dof system

    .. math::

        A_{ff} x_f = b_f - A_{fk} g.

    The implementation is fully vectorized over the COO triplets.  Rows whose
    unknown is prescribed are dropped, columns whose unknown is prescribed are
    accumulated into the reduced RHS, and free/free entries are remapped to
    compact reduced indices.
    """
    row_indices = np.asarray(row_indices)
    col_indices = np.asarray(col_indices)
    matrix_values = np.asarray(matrix_values, dtype=REAL_DTYPE)
    rhs = np.asarray(rhs, dtype=REAL_DTYPE)
    known_mask = np.asarray(known_mask, dtype=bool)
    known_values = np.asarray(known_values, dtype=REAL_DTYPE)

    if rhs.ndim != 1:
        raise ValueError("rhs must be one-dimensional")
    system_size = rhs.size
    if known_mask.shape != (system_size,):
        raise ValueError(f"known_mask must have shape ({system_size},), got {known_mask.shape}")
    if known_values.shape != (system_size,):
        raise ValueError(f"known_values must have shape ({system_size},), got {known_values.shape}")
    validate_global_system_inputs(
        row_indices,
        col_indices,
        matrix_values,
        rhs,
        system_size,
    )

    free_mask = ~known_mask
    old_to_new = np.full(system_size, -1, dtype=np.int64)
    old_to_new[free_mask] = np.arange(np.count_nonzero(free_mask), dtype=np.int64)

    row_is_free = free_mask[row_indices]
    col_is_free = free_mask[col_indices]
    free_free = row_is_free & col_is_free
    free_known = row_is_free & ~col_is_free

    reduced_rows = old_to_new[row_indices[free_free]]
    reduced_cols = old_to_new[col_indices[free_free]]
    reduced_data = matrix_values[free_free].copy()
    reduced_rhs = rhs[free_mask].copy()

    if np.any(free_known):
        np.add.at(
            reduced_rhs,
            old_to_new[row_indices[free_known]],
            -matrix_values[free_known] * known_values[col_indices[free_known]],
        )

    return KnownDofReduction(
        rows=np.ascontiguousarray(reduced_rows),
        cols=np.ascontiguousarray(reduced_cols),
        data=np.ascontiguousarray(reduced_data),
        rhs=np.ascontiguousarray(reduced_rhs),
        free_mask=np.ascontiguousarray(free_mask),
        known_mask=np.ascontiguousarray(known_mask),
        known_values=np.ascontiguousarray(known_values),
        old_to_new=np.ascontiguousarray(old_to_new),
    )


def update_known_dof_rhs(row_indices, col_indices, matrix_values, rhs, known_values, reduction):
    """Refresh a reduced RHS/boundary while retaining the fixed reduced operator.

    This is shared by diffusion and frozen-operator transport solves. The full
    COO matrix is needed only for the free/known boundary-column contribution.
    """
    rhs = np.asarray(rhs, dtype=REAL_DTYPE)
    known_values = np.asarray(known_values, dtype=REAL_DTYPE).ravel()
    if rhs.shape != reduction.free_mask.shape or known_values.shape != rhs.shape:
        raise ValueError("RHS and known values must match the full reduction size")
    rows, cols = np.asarray(row_indices), np.asarray(col_indices)
    data = np.asarray(matrix_values, dtype=REAL_DTYPE)
    reduced_rhs = rhs[reduction.free_mask].copy()
    free_known = reduction.free_mask[rows] & reduction.known_mask[cols]
    np.add.at(reduced_rhs, reduction.old_to_new[rows[free_known]],
              -data[free_known]*known_values[cols[free_known]])
    return KnownDofReduction(
        rows=reduction.rows, cols=reduction.cols, data=reduction.data,
        rhs=np.ascontiguousarray(reduced_rhs), free_mask=reduction.free_mask,
        known_mask=reduction.known_mask, known_values=np.ascontiguousarray(known_values),
        old_to_new=reduction.old_to_new,
    )


def expand_known_dofs(reduced_solution: NDArray, reduction: KnownDofReduction) -> NDArray:
    """Expand a reduced solution by reinserting prescribed dof values."""
    reduced_solution = np.asarray(reduced_solution, dtype=REAL_DTYPE)
    expected_shape = (np.count_nonzero(reduction.free_mask),)
    if reduced_solution.shape != expected_shape:
        raise ValueError(f"reduced_solution must have shape {expected_shape}; got {reduced_solution.shape}")
    full_solution = reduction.known_values.copy()
    full_solution[reduction.free_mask] = reduced_solution
    return np.ascontiguousarray(full_solution)


def expand_known_dofs_cupy(reduced_solution, reduction: KnownDofReduction):
    """Expand a reduced trace vector on-device without a host round trip."""
    cupy = require_cupy()
    reduced_cp = cupy.asarray(reduced_solution, dtype=REAL_DTYPE)
    expected_size = int(np.count_nonzero(reduction.free_mask))
    if reduced_cp.shape != (expected_size,):
        raise ValueError(f"reduced_solution must have shape ({expected_size},); got {reduced_cp.shape}")
    full = cupy.asarray(reduction.known_values, dtype=REAL_DTYPE).copy()
    full[cupy.asarray(reduction.free_mask)] = reduced_cp
    return cupy.ascontiguousarray(full)
