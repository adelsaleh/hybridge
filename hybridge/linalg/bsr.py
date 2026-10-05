"""Host utilities for sparse matrices with dense square blocks."""

from __future__ import annotations

import numpy as np
from scipy.sparse import bsr_matrix, isspmatrix_bsr


def face_dense_to_bsr(blocks: np.ndarray, neighbors: np.ndarray) -> bsr_matrix:
    """Convert padded face-neighbor blocks without scalar COO index arrays."""
    blocks, neighbors = np.asarray(blocks), np.asarray(neighbors)
    if blocks.ndim != 4 or blocks.shape[2] != blocks.shape[3]:
        raise ValueError("blocks must have shape (rows, slots, b, b)")
    if neighbors.shape != blocks.shape[:2] or not np.issubdtype(neighbors.dtype, np.integer):
        raise ValueError("neighbors must be an integer (rows, slots) array")
    rows, _, size, _ = blocks.shape
    if size < 1 or np.any(neighbors < -1) or np.any(neighbors >= rows):
        raise ValueError("invalid block size or neighbor index")
    valid = neighbors >= 0
    indptr = np.r_[0, np.cumsum(valid.sum(axis=1), dtype=np.int64)]
    result = bsr_matrix((blocks[valid], neighbors[valid], indptr),
                        shape=(rows*size, rows*size))
    result.sum_duplicates()
    result.sort_indices()
    return result


def principal_bsr_submatrix(
    matrix: bsr_matrix,
    selected_block_rows: np.ndarray,
) -> bsr_matrix:
    """Extract an exact principal block submatrix in the supplied row order.

    The input must be square with square BSR blocks and sorted, duplicate-free
    block column indices. selected_block_rows contains unique integer block
    row IDs. Its order defines both the rows and columns of the result.

    Numeric blocks and structural zeros are preserved without expansion to
    scalar CSR storage. Remapped column indices are sorted within every output
    row. Returned data and index arrays are independent of the input.
    """
    if not isspmatrix_bsr(matrix):
        raise TypeError("matrix must be a SciPy BSR matrix")
    block_size, block_columns = matrix.blocksize
    if matrix.shape[0] != matrix.shape[1] or block_size != block_columns:
        raise ValueError("matrix and its BSR blocks must be square")
    if not matrix.has_canonical_format:
        raise ValueError("matrix must have sorted BSR indices without duplicates")

    selected = np.asarray(selected_block_rows)
    if selected.ndim != 1:
        raise ValueError("selected_block_rows must be one-dimensional")
    # An empty Python list has float dtype but is an unambiguous empty selection.
    if selected.size and not np.issubdtype(selected.dtype, np.integer):
        raise TypeError("selected_block_rows must contain integer block row IDs")
    num_block_rows = matrix.shape[0] // block_size
    if np.any(selected < 0) or np.any(selected >= num_block_rows):
        raise ValueError("selected_block_rows contains an out-of-range block row")
    selected = np.asarray(selected, dtype=np.int64)
    if np.unique(selected).size != selected.size:
        raise ValueError("selected_block_rows must not contain duplicates")

    global_to_local = np.full(num_block_rows, -1, dtype=np.int64)
    global_to_local[selected] = np.arange(selected.size, dtype=np.int64)
    indptr = np.zeros(selected.size + 1, dtype=np.int64)
    columns = []
    source_positions = []
    for local_row, global_row in enumerate(selected):
        begin, end = matrix.indptr[global_row : global_row + 2]
        local_columns = global_to_local[matrix.indices[begin:end]]
        active = local_columns >= 0
        active_columns = local_columns[active]
        order = np.argsort(active_columns, kind="stable")
        columns.append(active_columns[order])
        source_positions.append(
            (begin + np.flatnonzero(active))[order]
        )
        indptr[local_row + 1] = indptr[local_row] + active_columns.size

    if selected.size:
        indices = np.concatenate(columns)
        positions = np.concatenate(source_positions)
    else:
        indices = np.empty(0, dtype=np.int64)
        positions = np.empty(0, dtype=np.int64)
    return bsr_matrix(
        (matrix.data[positions], indices, indptr),
        shape=(selected.size * block_size, selected.size * block_size),
    )


__all__ = ["face_dense_to_bsr", "principal_bsr_submatrix"]
