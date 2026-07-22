"""On-the-fly upwind block Gauss-Seidel preconditioner construction.

This module is an experimental host-side bridge between HDG trace assembly and
the reusable :mod:`hdgfem.linalg.upwind_block_gs` preconditioner.  The existing
builder scans an already assembled, upwind-ordered CSR matrix.  The routines
here either consume reduced scalar COO triplets with an explicit upwind-SCC
permutation or consume assembly-emitted dense edge-block COO entries that are
already in upwind order.

The implementation is intentionally conservative: it constructs the same
``UpwindBlockGSPreconditioner`` object used by the CSR path and supports only
the forward sweep.  It is a host benchmark target before moving the same idea
inside GPU assembly kernels.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:  # pragma: no cover - exercised only when numba is available.
    from numba import njit, prange
except ImportError:  # pragma: no cover
    njit = None
    prange = range

from .upwind_block_gs import UpwindBlockGSPreconditioner, UpwindBlockGSStats


@dataclass(frozen=True)
class UpwindBlockGSOnTheFlyTimings:
    """Wall-clock timings for triplet-driven preconditioner construction."""

    validation_seconds: float
    bounded_fill_seconds: float
    row_ptr_seconds: float
    compact_seconds: float
    block_sort_seconds: float
    pattern_seconds: float
    diagonal_inverse_seconds: float
    warmup_seconds: float
    build_seconds: float


def _require_numba() -> None:
    if njit is None:
        raise RuntimeError("triplet-driven upwind block-GS construction requires numba")


def _njit(*args, **kwargs):
    if njit is None:  # pragma: no cover
        def decorator(function):
            return function

        return decorator
    return njit(*args, **kwargs)


@_njit(cache=True)
def _invert_permutation_kernel(permutation, inverse_permutation):
    invalid_entries = 0
    num_entries = permutation.size
    seen = np.zeros(num_entries, dtype=np.uint8)

    for ordered_index in range(num_entries):
        natural_index = permutation[ordered_index]
        if natural_index < 0 or natural_index >= num_entries:
            invalid_entries += 1
            continue
        if seen[natural_index] != 0:
            invalid_entries += 1
        seen[natural_index] = 1
        inverse_permutation[natural_index] = ordered_index

    for natural_index in range(num_entries):
        if seen[natural_index] == 0:
            invalid_entries += 1

    return invalid_entries


@_njit(cache=True)
def _find_or_insert_bounded_col(col_table, counts, row_block, col_block, max_couplings_per_block):
    count = counts[row_block]
    for slot in range(count):
        if col_table[row_block, slot] == col_block:
            return slot, 0

    if count >= max_couplings_per_block:
        return -1, -1

    col_table[row_block, count] = col_block
    counts[row_block] = count + 1
    return count, 1


@_njit(cache=True)
def _fill_bounded_forward_blocks_from_block_coo(
        block_rows,
        block_cols,
        block_values,
        block_levels,
        max_couplings_per_block,
        lower_col_table,
        same_col_table,
        downstream_col_table,
        lower_counts,
        same_counts,
        downstream_counts,
        diagonal_blocks,
        lower_values_bounded,
):
    overflow_count = 0
    num_block_entries = block_rows.size
    block_size = diagonal_blocks.shape[1]

    for entry in range(num_block_entries):
        row_block = block_rows[entry]
        col_block = block_cols[entry]

        if row_block == col_block:
            for i in range(block_size):
                for j in range(block_size):
                    diagonal_blocks[row_block, i, j] += block_values[entry, i, j]
            continue

        col_level = block_levels[col_block]
        row_level = block_levels[row_block]
        if col_level < row_level:
            slot, inserted = _find_or_insert_bounded_col(
                lower_col_table,
                lower_counts,
                row_block,
                col_block,
                max_couplings_per_block,
            )
            if inserted < 0:
                overflow_count += 1
                continue
            for i in range(block_size):
                for j in range(block_size):
                    lower_values_bounded[row_block, slot, i, j] += block_values[entry, i, j]
        elif col_level == row_level:
            _slot, inserted = _find_or_insert_bounded_col(
                same_col_table,
                same_counts,
                row_block,
                col_block,
                max_couplings_per_block,
            )
            if inserted < 0:
                overflow_count += 1
        else:
            _slot, inserted = _find_or_insert_bounded_col(
                downstream_col_table,
                downstream_counts,
                row_block,
                col_block,
                max_couplings_per_block,
            )
            if inserted < 0:
                overflow_count += 1

    return overflow_count


@_njit(cache=True)
def _scale_diagonal_and_bounded_lower_blocks(diagonal_blocks, lower_values_bounded, lower_counts, row_scale):
    zero_diagonal_count = 0
    num_blocks = diagonal_blocks.shape[0]
    block_size = diagonal_blocks.shape[1]

    for block in range(num_blocks):
        count = lower_counts[block]
        block_base = block * block_size
        for row_dof in range(block_size):
            diagonal = diagonal_blocks[block, row_dof, row_dof]
            if diagonal == 0.0:
                scale = 1.0
                zero_diagonal_count += 1
            else:
                scale = 1.0 / diagonal
            row_scale[block_base + row_dof] = scale

            for col_dof in range(block_size):
                diagonal_blocks[block, row_dof, col_dof] *= scale

            for slot in range(count):
                for col_dof in range(block_size):
                    lower_values_bounded[block, slot, row_dof, col_dof] *= scale

    return zero_diagonal_count


@_njit(cache=True)
def _fill_bounded_forward_blocks(
        row_indices,
        col_indices,
        matrix_values,
        inverse_permutation,
        row_scale,
        block_levels,
        block_size,
        max_couplings_per_block,
        lower_col_table,
        same_col_table,
        downstream_col_table,
        lower_counts,
        same_counts,
        downstream_counts,
        diagonal_blocks,
        lower_values_bounded,
):
    lower_triplets = 0
    same_triplets = 0
    downstream_triplets = 0
    overflow_count = 0
    num_triplets = row_indices.size

    for triplet in range(num_triplets):
        natural_row = row_indices[triplet]
        ordered_row = inverse_permutation[natural_row]
        ordered_col = inverse_permutation[col_indices[triplet]]
        row_block = ordered_row // block_size
        col_block = ordered_col // block_size
        local_row = ordered_row - row_block * block_size
        local_col = ordered_col - col_block * block_size
        value = matrix_values[triplet] * row_scale[natural_row]

        if row_block == col_block:
            diagonal_blocks[row_block, local_row, local_col] += value
            continue

        col_level = block_levels[col_block]
        row_level = block_levels[row_block]
        if col_level < row_level:
            lower_triplets += 1
            slot, inserted = _find_or_insert_bounded_col(
                lower_col_table,
                lower_counts,
                row_block,
                col_block,
                max_couplings_per_block,
            )
            if inserted < 0:
                overflow_count += 1
                continue
            lower_values_bounded[row_block, slot, local_row, local_col] += value
        elif col_level == row_level:
            same_triplets += 1
            _slot, inserted = _find_or_insert_bounded_col(
                same_col_table,
                same_counts,
                row_block,
                col_block,
                max_couplings_per_block,
            )
            if inserted < 0:
                overflow_count += 1
        else:
            downstream_triplets += 1
            _slot, inserted = _find_or_insert_bounded_col(
                downstream_col_table,
                downstream_counts,
                row_block,
                col_block,
                max_couplings_per_block,
            )
            if inserted < 0:
                overflow_count += 1

    return lower_triplets, same_triplets, downstream_triplets, overflow_count


@_njit(cache=True)
def _compact_bounded_lower_blocks(
        lower_counts,
        lower_col_table,
        lower_values_bounded,
        lower_row_ptr,
        lower_col_ind,
        lower_values,
):
    block_size = lower_values.shape[1]
    num_blocks = lower_counts.size

    for block in range(num_blocks):
        out_begin = lower_row_ptr[block]
        count = lower_counts[block]
        for slot in range(count):
            out_pos = out_begin + slot
            lower_col_ind[out_pos] = lower_col_table[block, slot]
            for i in range(block_size):
                for j in range(block_size):
                    lower_values[out_pos, i, j] = lower_values_bounded[block, slot, i, j]


@_njit(cache=True)
def _insert_relation_key(table, key):
    capacity = table.size
    slot = key % capacity

    for _probe in range(capacity):
        current = table[slot]
        if current == key:
            return 0
        if current == -1:
            table[slot] = key
            return 1
        slot += 1
        if slot == capacity:
            slot = 0

    return -1


@_njit(cache=True)
def _insert_or_get_relation_position(key_table, position_table, key, position):
    capacity = key_table.size
    slot = key % capacity

    for _probe in range(capacity):
        current = key_table[slot]
        if current == key:
            return position_table[slot], 0
        if current == -1:
            key_table[slot] = key
            position_table[slot] = position
            return position, 1
        slot += 1
        if slot == capacity:
            slot = 0

    return -1, -1


@_njit(cache=True)
def _count_unique_block_relations(
        row_indices,
        col_indices,
        inverse_permutation,
        block_size,
        block_levels,
        num_blocks,
        lower_key_table,
        same_key_table,
        downstream_key_table,
        retained_counts,
        same_counts,
        downstream_counts,
):
    lower_triplets = 0
    same_triplets = 0
    downstream_triplets = 0
    hash_overflow_count = 0
    num_triplets = row_indices.size

    for triplet in range(num_triplets):
        ordered_row = inverse_permutation[row_indices[triplet]]
        ordered_col = inverse_permutation[col_indices[triplet]]
        row_block = ordered_row // block_size
        col_block = ordered_col // block_size

        if row_block == col_block:
            continue

        key = row_block * num_blocks + col_block
        col_level = block_levels[col_block]
        row_level = block_levels[row_block]
        if col_level < row_level:
            lower_triplets += 1
            inserted = _insert_relation_key(lower_key_table, key)
            if inserted == 1:
                retained_counts[row_block] += 1
            elif inserted < 0:
                hash_overflow_count += 1
        elif col_level == row_level:
            same_triplets += 1
            inserted = _insert_relation_key(same_key_table, key)
            if inserted == 1:
                same_counts[row_block] += 1
            elif inserted < 0:
                hash_overflow_count += 1
        else:
            downstream_triplets += 1
            inserted = _insert_relation_key(downstream_key_table, key)
            if inserted == 1:
                downstream_counts[row_block] += 1
            elif inserted < 0:
                hash_overflow_count += 1

    return lower_triplets, same_triplets, downstream_triplets, hash_overflow_count


@_njit(cache=True)
def _fill_forward_blocks(
        row_indices,
        col_indices,
        matrix_values,
        inverse_permutation,
        row_scale,
        lower_row_ptr,
        next_lower_pos,
        lower_key_table,
        lower_position_table,
        lower_col_ind,
        block_levels,
        num_blocks,
        block_size,
        diagonal_blocks,
        lower_values,
):
    overflow_count = 0
    num_triplets = row_indices.size

    for triplet in range(num_triplets):
        natural_row = row_indices[triplet]
        ordered_row = inverse_permutation[natural_row]
        ordered_col = inverse_permutation[col_indices[triplet]]
        row_block = ordered_row // block_size
        col_block = ordered_col // block_size
        local_row = ordered_row - row_block * block_size
        local_col = ordered_col - col_block * block_size
        value = matrix_values[triplet] * row_scale[natural_row]

        if row_block == col_block:
            diagonal_blocks[row_block, local_row, local_col] += value
            continue

        if block_levels[col_block] >= block_levels[row_block]:
            continue

        next_pos = next_lower_pos[row_block]
        key = row_block * num_blocks + col_block
        pos, inserted = _insert_or_get_relation_position(
            lower_key_table,
            lower_position_table,
            key,
            next_pos,
        )
        if inserted < 0:
            overflow_count += 1
            continue
        if inserted == 1:
            if pos >= lower_row_ptr[row_block + 1]:
                overflow_count += 1
                continue
            next_lower_pos[row_block] = pos + 1
            lower_col_ind[pos] = col_block

        lower_values[pos, local_row, local_col] += value

    return overflow_count


@_njit(cache=True)
def _sort_lower_blocks_by_column(lower_row_ptr, lower_col_ind, lower_values):
    block_size = lower_values.shape[1]
    num_blocks = lower_row_ptr.size - 1

    for block in range(num_blocks):
        start = lower_row_ptr[block]
        stop = lower_row_ptr[block + 1]
        for pos in range(start + 1, stop):
            current = pos
            while current > start and lower_col_ind[current - 1] > lower_col_ind[current]:
                tmp_col = lower_col_ind[current - 1]
                lower_col_ind[current - 1] = lower_col_ind[current]
                lower_col_ind[current] = tmp_col

                for i in range(block_size):
                    for j in range(block_size):
                        tmp_value = lower_values[current - 1, i, j]
                        lower_values[current - 1, i, j] = lower_values[current, i, j]
                        lower_values[current, i, j] = tmp_value

                current -= 1


@_njit(cache=True, parallel=True)
def _scale_ordered_coo_rows_kernel(row_indices, matrix_values, rhs, row_scale, scaled_values, scaled_rhs):
    for entry in prange(matrix_values.size):
        scaled_values[entry] = matrix_values[entry] * row_scale[row_indices[entry]]
    for row in prange(rhs.size):
        scaled_rhs[row] = rhs[row] * row_scale[row]


def _level_width_array(level_widths: Sequence[int] | object) -> np.ndarray:
    if hasattr(level_widths, "widths"):
        level_widths = getattr(level_widths, "widths")
    widths = np.asarray(tuple(level_widths), dtype=np.int64)
    if widths.ndim != 1:
        raise ValueError("level_widths must be one-dimensional")
    if widths.size == 0:
        raise ValueError("level_widths must contain at least one level")
    if np.any(widths <= 0):
        raise ValueError("level_widths must be strictly positive")
    return np.ascontiguousarray(widths)


def _level_offsets_and_block_levels(widths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    level_offsets = np.empty(widths.size + 1, dtype=np.int64)
    level_offsets[0] = 0
    np.cumsum(widths, out=level_offsets[1:])
    block_levels = np.repeat(np.arange(widths.size, dtype=np.int64), widths)
    return np.ascontiguousarray(level_offsets), np.ascontiguousarray(block_levels)


def _inverse_permutation(permutation: np.ndarray, system_size: int) -> np.ndarray:
    if permutation.shape != (system_size,):
        raise ValueError(
            f"permutation must have shape ({system_size},), got {permutation.shape}"
        )
    if permutation.size == 0:
        return np.empty(0, dtype=np.int64)

    inverse = np.empty(system_size, dtype=np.int64)
    invalid_entries = _invert_permutation_kernel(permutation, inverse)
    if int(invalid_entries) != 0:
        raise ValueError("permutation must contain each scalar dof exactly once")
    return np.ascontiguousarray(inverse)



def _relation_hash_capacity(num_blocks: int, multiplier: int = 8) -> int:
    target = max(16, int(num_blocks) * int(multiplier))
    return 1 << (target - 1).bit_length()

def _resolved_apply_mode(apply_mode: str, widths: np.ndarray, parallel_min_width: int) -> str:
    if apply_mode not in {"auto", "serial", "parallel"}:
        raise ValueError("apply_mode must be 'auto', 'serial', or 'parallel'")
    if parallel_min_width <= 0:
        raise ValueError("parallel_min_width must be positive")
    if apply_mode == "auto":
        return "parallel" if int(np.max(widths)) >= int(parallel_min_width) else "serial"
    return apply_mode


def build_forward_upwind_block_gs_from_coo(
        row_indices,
        col_indices,
        matrix_values,
        system_size: int,
        *,
        permutation,
        level_widths: Sequence[int] | object,
        block_size: int,
        row_scale=None,
        diagonal_regularization: float = 0.0,
        apply_mode: str = "auto",
        parallel_min_width: int = 1024,
        bounded_max_couplings_per_block: int = 6,
        warm_start: bool = True,
) -> UpwindBlockGSPreconditioner:
    """Build a forward upwind block-GS preconditioner from COO triplets.

    Parameters
    ----------
    row_indices, col_indices, matrix_values
        Reduced trace COO triplets in the natural scalar trace-DOF ordering.
        Duplicate triplets are accumulated into dense edge-block entries.
    system_size
        Number of scalar trace unknowns in the reduced system.
    permutation
        Scalar permutation that maps ordered scalar rows to natural scalar rows,
        matching ``matrix[permutation][:, permutation]``.
    level_widths
        Upwind-SCC topological level widths in edge-block units.
    block_size
        Number of scalar trace degrees of freedom per edge.
    row_scale
        Optional left row scaling in natural scalar ordering.  Passing
        ``1 / diag(A)`` makes the accumulated blocks match the diagonally
        scaled matrix ``D^-1 A`` used by the existing solver helper.
    diagonal_regularization, apply_mode, parallel_min_width, warm_start
        Same meaning as in :func:`build_upwind_block_gs_preconditioner`.
    bounded_max_couplings_per_block
        Maximum number of unique lower, same-level, or downstream neighbor edge
        blocks expected per row edge in the one-pass bounded host builder.
    """
    _require_numba()
    build_start = time.perf_counter()

    validation_start = time.perf_counter()
    system_size = int(system_size)
    block_size = int(block_size)
    if system_size < 0:
        raise ValueError(f"system_size must be non-negative, got {system_size}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if system_size % block_size != 0:
        raise ValueError(
            f"system_size {system_size} is not divisible by block_size={block_size}"
        )

    rows = np.ascontiguousarray(row_indices, dtype=np.int64)
    cols = np.ascontiguousarray(col_indices, dtype=np.int64)
    values = np.ascontiguousarray(matrix_values, dtype=np.float64)
    if rows.ndim != 1 or cols.ndim != 1 or values.ndim != 1:
        raise ValueError("row_indices, col_indices, and matrix_values must be one-dimensional")
    if rows.shape != cols.shape or rows.shape != values.shape:
        raise ValueError("row_indices, col_indices, and matrix_values must have matching shapes")
    if rows.size:
        if int(np.min(rows)) < 0 or int(np.max(rows)) >= system_size:
            raise ValueError("row_indices contain entries outside the system")
        if int(np.min(cols)) < 0 or int(np.max(cols)) >= system_size:
            raise ValueError("col_indices contain entries outside the system")

    permutation_array = np.ascontiguousarray(permutation, dtype=np.int64)
    inverse_permutation = _inverse_permutation(permutation_array, system_size)

    if row_scale is None:
        row_scale_array = np.ones(system_size, dtype=np.float64)
    else:
        row_scale_array = np.ascontiguousarray(row_scale, dtype=np.float64)
        if row_scale_array.shape != (system_size,):
            raise ValueError(
                f"row_scale must have shape ({system_size},), got {row_scale_array.shape}"
            )

    widths = _level_width_array(level_widths)
    level_offsets, block_levels = _level_offsets_and_block_levels(widths)
    num_blocks = system_size // block_size
    if int(level_offsets[-1]) != num_blocks:
        raise ValueError(
            "level_widths sum does not match matrix block count: "
            f"{int(level_offsets[-1])} != {num_blocks}"
        )
    resolved_mode = _resolved_apply_mode(apply_mode, widths, parallel_min_width)
    validation_seconds = time.perf_counter() - validation_start

    max_couplings = int(bounded_max_couplings_per_block)
    if max_couplings <= 0:
        raise ValueError("bounded_max_couplings_per_block must be positive")

    pattern_start = time.perf_counter()
    bounded_fill_start = time.perf_counter()
    retained_counts = np.zeros(num_blocks, dtype=np.int64)
    same_counts = np.zeros(num_blocks, dtype=np.int64)
    downstream_counts = np.zeros(num_blocks, dtype=np.int64)
    lower_col_table = np.full((num_blocks, max_couplings), -1, dtype=np.int64)
    same_col_table = np.full((num_blocks, max_couplings), -1, dtype=np.int64)
    downstream_col_table = np.full((num_blocks, max_couplings), -1, dtype=np.int64)
    diagonal_blocks = np.zeros((num_blocks, block_size, block_size), dtype=np.float64)
    lower_values_bounded = np.zeros((num_blocks, max_couplings, block_size, block_size), dtype=np.float64)
    lower_triplets, same_triplets, downstream_triplets, bounded_overflow = _fill_bounded_forward_blocks(
        rows,
        cols,
        values,
        inverse_permutation,
        row_scale_array,
        block_levels,
        block_size,
        max_couplings,
        lower_col_table,
        same_col_table,
        downstream_col_table,
        retained_counts,
        same_counts,
        downstream_counts,
        diagonal_blocks,
        lower_values_bounded,
    )
    bounded_fill_seconds = time.perf_counter() - bounded_fill_start
    if int(bounded_overflow) != 0:
        raise RuntimeError(
            "bounded upwind block pattern overflowed; increase "
            "bounded_max_couplings_per_block "
            f"above {max_couplings} (overflow count={int(bounded_overflow)})"
        )

    row_ptr_start = time.perf_counter()
    lower_row_ptr = np.empty(num_blocks + 1, dtype=np.int64)
    lower_row_ptr[0] = 0
    np.cumsum(retained_counts, out=lower_row_ptr[1:])
    retained_total = int(lower_row_ptr[-1])
    dropped_same = int(np.sum(same_counts))
    downstream_total = int(np.sum(downstream_counts))
    row_ptr_seconds = time.perf_counter() - row_ptr_start

    lower_col_ind = np.empty(retained_total, dtype=np.int64)
    lower_values = np.zeros((retained_total, block_size, block_size), dtype=np.float64)
    compact_start = time.perf_counter()
    _compact_bounded_lower_blocks(
        retained_counts,
        lower_col_table,
        lower_values_bounded,
        lower_row_ptr,
        lower_col_ind,
        lower_values,
    )
    compact_seconds = time.perf_counter() - compact_start
    del lower_values_bounded, lower_col_table, same_col_table, downstream_col_table

    sort_start = time.perf_counter()
    _sort_lower_blocks_by_column(lower_row_ptr, lower_col_ind, lower_values)
    block_sort_seconds = time.perf_counter() - sort_start
    pattern_seconds = time.perf_counter() - pattern_start

    if diagonal_regularization != 0.0:
        diag_ids = np.arange(block_size)
        diagonal_blocks[:, diag_ids, diag_ids] += float(diagonal_regularization)

    inverse_start = time.perf_counter()
    try:
        diagonal_inverse = np.linalg.inv(diagonal_blocks)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "failed to invert at least one edge-block diagonal while building "
            "the triplet-driven upwind block-GS preconditioner"
        ) from exc
    diagonal_inverse_seconds = time.perf_counter() - inverse_start

    offdiag_total = retained_total + downstream_total + dropped_same
    dropped_fraction = (
        0.0
        if offdiag_total == 0
        else (dropped_same + downstream_total) / offdiag_total
    )

    upper_row_ptr = np.zeros(num_blocks + 1, dtype=np.int64)
    upper_col_ind = np.empty(0, dtype=np.int64)
    upper_values = np.empty((0, block_size, block_size), dtype=np.float64)
    warmup_seconds = 0.0
    stats = UpwindBlockGSStats(
        num_blocks=int(num_blocks),
        block_size=int(block_size),
        num_levels=int(widths.size),
        max_width=int(np.max(widths)),
        median_width=float(np.median(widths)),
        mean_width=float(np.mean(widths)),
        retained_block_couplings=retained_total,
        downstream_block_couplings=downstream_total,
        dropped_same_level_couplings=dropped_same,
        dropped_downstream_couplings=downstream_total,
        dropped_coupling_fraction=float(dropped_fraction),
        apply_mode=resolved_mode,
        sweep="forward",
        csr_prepare_seconds=0.0,
        coupling_count_seconds=pattern_seconds,
        block_fill_seconds=bounded_fill_seconds + compact_seconds + block_sort_seconds,
        diagonal_inverse_seconds=diagonal_inverse_seconds,
        warmup_seconds=warmup_seconds,
        build_seconds=0.0,
    )

    preconditioner = UpwindBlockGSPreconditioner(
        level_offsets=level_offsets,
        lower_row_ptr=lower_row_ptr,
        lower_col_ind=lower_col_ind,
        lower_values=lower_values,
        upper_row_ptr=upper_row_ptr,
        upper_col_ind=upper_col_ind,
        upper_values=upper_values,
        diagonal_blocks=diagonal_blocks,
        diagonal_inverse=diagonal_inverse,
        stats=stats,
        apply_mode=resolved_mode,
        sweep="forward",
    )

    if warm_start:
        warmup_start = time.perf_counter()
        preconditioner @ np.zeros(preconditioner.shape[1], dtype=np.float64)
        warmup_seconds = time.perf_counter() - warmup_start
        preconditioner.reset_timing()

    build_seconds = time.perf_counter() - build_start
    preconditioner.stats = UpwindBlockGSStats(
        num_blocks=stats.num_blocks,
        block_size=stats.block_size,
        num_levels=stats.num_levels,
        max_width=stats.max_width,
        median_width=stats.median_width,
        mean_width=stats.mean_width,
        retained_block_couplings=stats.retained_block_couplings,
        downstream_block_couplings=stats.downstream_block_couplings,
        dropped_same_level_couplings=stats.dropped_same_level_couplings,
        dropped_downstream_couplings=stats.dropped_downstream_couplings,
        dropped_coupling_fraction=stats.dropped_coupling_fraction,
        apply_mode=stats.apply_mode,
        sweep=stats.sweep,
        csr_prepare_seconds=stats.csr_prepare_seconds,
        coupling_count_seconds=stats.coupling_count_seconds,
        block_fill_seconds=stats.block_fill_seconds,
        diagonal_inverse_seconds=stats.diagonal_inverse_seconds,
        warmup_seconds=warmup_seconds,
        build_seconds=build_seconds,
    )
    preconditioner.onfly_timings = UpwindBlockGSOnTheFlyTimings(
        validation_seconds=validation_seconds,
        bounded_fill_seconds=bounded_fill_seconds,
        row_ptr_seconds=row_ptr_seconds,
        compact_seconds=compact_seconds,
        block_sort_seconds=block_sort_seconds,
        pattern_seconds=pattern_seconds,
        diagonal_inverse_seconds=diagonal_inverse_seconds,
        warmup_seconds=warmup_seconds,
        build_seconds=build_seconds,
    )
    preconditioner.onfly_strategy = "bounded"
    preconditioner.row_scale = np.ascontiguousarray(row_scale_array, dtype=np.float64)
    preconditioner.onfly_max_couplings_per_block = int(max_couplings)
    preconditioner.onfly_triplet_counts = {
        "lower": int(lower_triplets),
        "same_level": int(same_triplets),
        "downstream": int(downstream_triplets),
    }
    return preconditioner


def build_forward_upwind_block_gs_from_ordered_block_coo(
        block_rows,
        block_cols,
        block_values,
        num_blocks: int,
        *,
        level_widths: Sequence[int] | object,
        diagonal_regularization: float = 0.0,
        apply_mode: str = "auto",
        parallel_min_width: int = 1024,
        bounded_max_couplings_per_block: int = 6,
        warm_start: bool = True,
) -> UpwindBlockGSPreconditioner:
    """Build a forward upwind block-GS preconditioner from ordered block COO.

    ``block_rows`` and ``block_cols`` are reduced trace edge-block ids, not
    scalar dof ids.  They are expected to already be in the upwind-SCC order
    associated with ``level_widths``.  ``block_values`` contains unscaled dense
    edge blocks.  The builder computes the same left Jacobi row scaling used by
    :func:`hdgfem.linalg.system.diagonal_scale_system` from the accumulated
    diagonal blocks before compacting and inverting the block diagonal.
    """
    _require_numba()
    build_start = time.perf_counter()

    validation_start = time.perf_counter()
    num_blocks = int(num_blocks)
    if num_blocks < 0:
        raise ValueError(f"num_blocks must be non-negative, got {num_blocks}")

    rows = np.ascontiguousarray(block_rows, dtype=np.int64)
    cols = np.ascontiguousarray(block_cols, dtype=np.int64)
    values = np.ascontiguousarray(block_values, dtype=np.float64)
    if rows.ndim != 1 or cols.ndim != 1:
        raise ValueError("block_rows and block_cols must be one-dimensional")
    if rows.shape != cols.shape:
        raise ValueError("block_rows and block_cols must have matching shapes")
    if values.ndim != 3 or values.shape[0] != rows.size or values.shape[1] != values.shape[2]:
        raise ValueError(
            "block_values must have shape (num_block_entries, block_size, block_size)"
        )
    block_size = int(values.shape[1])
    if block_size <= 0:
        raise ValueError("block_values must contain non-empty square blocks")
    if rows.size:
        if int(np.min(rows)) < 0 or int(np.max(rows)) >= num_blocks:
            raise ValueError("block_rows contain entries outside the block system")
        if int(np.min(cols)) < 0 or int(np.max(cols)) >= num_blocks:
            raise ValueError("block_cols contain entries outside the block system")

    widths = _level_width_array(level_widths)
    level_offsets, block_levels = _level_offsets_and_block_levels(widths)
    if int(level_offsets[-1]) != num_blocks:
        raise ValueError(
            "level_widths sum does not match block count: "
            f"{int(level_offsets[-1])} != {num_blocks}"
        )
    resolved_mode = _resolved_apply_mode(apply_mode, widths, parallel_min_width)
    max_couplings = int(bounded_max_couplings_per_block)
    if max_couplings <= 0:
        raise ValueError("bounded_max_couplings_per_block must be positive")
    validation_seconds = time.perf_counter() - validation_start

    pattern_start = time.perf_counter()
    bounded_fill_start = time.perf_counter()
    retained_counts = np.zeros(num_blocks, dtype=np.int64)
    same_counts = np.zeros(num_blocks, dtype=np.int64)
    downstream_counts = np.zeros(num_blocks, dtype=np.int64)
    lower_col_table = np.full((num_blocks, max_couplings), -1, dtype=np.int64)
    same_col_table = np.full((num_blocks, max_couplings), -1, dtype=np.int64)
    downstream_col_table = np.full((num_blocks, max_couplings), -1, dtype=np.int64)
    diagonal_blocks = np.zeros((num_blocks, block_size, block_size), dtype=np.float64)
    lower_values_bounded = np.zeros((num_blocks, max_couplings, block_size, block_size), dtype=np.float64)
    bounded_overflow = _fill_bounded_forward_blocks_from_block_coo(
        rows,
        cols,
        values,
        block_levels,
        max_couplings,
        lower_col_table,
        same_col_table,
        downstream_col_table,
        retained_counts,
        same_counts,
        downstream_counts,
        diagonal_blocks,
        lower_values_bounded,
    )
    row_scale = np.empty(num_blocks * block_size, dtype=np.float64)
    zero_diagonal_count = _scale_diagonal_and_bounded_lower_blocks(
        diagonal_blocks,
        lower_values_bounded,
        retained_counts,
        row_scale,
    )
    bounded_fill_seconds = time.perf_counter() - bounded_fill_start
    if int(bounded_overflow) != 0:
        raise RuntimeError(
            "bounded block-COO upwind pattern overflowed; increase "
            "bounded_max_couplings_per_block "
            f"above {max_couplings} (overflow count={int(bounded_overflow)})"
        )

    row_ptr_start = time.perf_counter()
    lower_row_ptr = np.empty(num_blocks + 1, dtype=np.int64)
    lower_row_ptr[0] = 0
    np.cumsum(retained_counts, out=lower_row_ptr[1:])
    retained_total = int(lower_row_ptr[-1])
    dropped_same = int(np.sum(same_counts))
    downstream_total = int(np.sum(downstream_counts))
    row_ptr_seconds = time.perf_counter() - row_ptr_start

    lower_col_ind = np.empty(retained_total, dtype=np.int64)
    lower_values = np.zeros((retained_total, block_size, block_size), dtype=np.float64)
    compact_start = time.perf_counter()
    _compact_bounded_lower_blocks(
        retained_counts,
        lower_col_table,
        lower_values_bounded,
        lower_row_ptr,
        lower_col_ind,
        lower_values,
    )
    compact_seconds = time.perf_counter() - compact_start
    del lower_values_bounded, lower_col_table, same_col_table, downstream_col_table

    sort_start = time.perf_counter()
    _sort_lower_blocks_by_column(lower_row_ptr, lower_col_ind, lower_values)
    block_sort_seconds = time.perf_counter() - sort_start
    pattern_seconds = time.perf_counter() - pattern_start

    if diagonal_regularization != 0.0:
        diag_ids = np.arange(block_size)
        diagonal_blocks[:, diag_ids, diag_ids] += float(diagonal_regularization)

    inverse_start = time.perf_counter()
    try:
        diagonal_inverse = np.linalg.inv(diagonal_blocks)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "failed to invert at least one edge-block diagonal while building "
            "the block-COO upwind block-GS preconditioner"
        ) from exc
    diagonal_inverse_seconds = time.perf_counter() - inverse_start

    offdiag_total = retained_total + downstream_total + dropped_same
    dropped_fraction = (
        0.0
        if offdiag_total == 0
        else (dropped_same + downstream_total) / offdiag_total
    )

    upper_row_ptr = np.zeros(num_blocks + 1, dtype=np.int64)
    upper_col_ind = np.empty(0, dtype=np.int64)
    upper_values = np.empty((0, block_size, block_size), dtype=np.float64)
    stats = UpwindBlockGSStats(
        num_blocks=int(num_blocks),
        block_size=int(block_size),
        num_levels=int(widths.size),
        max_width=int(np.max(widths)),
        median_width=float(np.median(widths)),
        mean_width=float(np.mean(widths)),
        retained_block_couplings=retained_total,
        downstream_block_couplings=downstream_total,
        dropped_same_level_couplings=dropped_same,
        dropped_downstream_couplings=downstream_total,
        dropped_coupling_fraction=float(dropped_fraction),
        apply_mode=resolved_mode,
        sweep="forward",
        csr_prepare_seconds=0.0,
        coupling_count_seconds=pattern_seconds,
        block_fill_seconds=bounded_fill_seconds + compact_seconds + block_sort_seconds,
        diagonal_inverse_seconds=diagonal_inverse_seconds,
        warmup_seconds=0.0,
        build_seconds=0.0,
    )

    preconditioner = UpwindBlockGSPreconditioner(
        level_offsets=level_offsets,
        lower_row_ptr=lower_row_ptr,
        lower_col_ind=lower_col_ind,
        lower_values=lower_values,
        upper_row_ptr=upper_row_ptr,
        upper_col_ind=upper_col_ind,
        upper_values=upper_values,
        diagonal_blocks=diagonal_blocks,
        diagonal_inverse=diagonal_inverse,
        stats=stats,
        apply_mode=resolved_mode,
        sweep="forward",
    )

    warmup_seconds = 0.0
    if warm_start:
        warmup_start = time.perf_counter()
        preconditioner @ np.zeros(preconditioner.shape[1], dtype=np.float64)
        warmup_seconds = time.perf_counter() - warmup_start
        preconditioner.reset_timing()

    build_seconds = time.perf_counter() - build_start
    preconditioner.stats = UpwindBlockGSStats(
        num_blocks=stats.num_blocks,
        block_size=stats.block_size,
        num_levels=stats.num_levels,
        max_width=stats.max_width,
        median_width=stats.median_width,
        mean_width=stats.mean_width,
        retained_block_couplings=stats.retained_block_couplings,
        downstream_block_couplings=stats.downstream_block_couplings,
        dropped_same_level_couplings=stats.dropped_same_level_couplings,
        dropped_downstream_couplings=stats.dropped_downstream_couplings,
        dropped_coupling_fraction=stats.dropped_coupling_fraction,
        apply_mode=stats.apply_mode,
        sweep=stats.sweep,
        csr_prepare_seconds=stats.csr_prepare_seconds,
        coupling_count_seconds=stats.coupling_count_seconds,
        block_fill_seconds=stats.block_fill_seconds,
        diagonal_inverse_seconds=stats.diagonal_inverse_seconds,
        warmup_seconds=warmup_seconds,
        build_seconds=build_seconds,
    )
    preconditioner.onfly_timings = UpwindBlockGSOnTheFlyTimings(
        validation_seconds=validation_seconds,
        bounded_fill_seconds=bounded_fill_seconds,
        row_ptr_seconds=row_ptr_seconds,
        compact_seconds=compact_seconds,
        block_sort_seconds=block_sort_seconds,
        pattern_seconds=pattern_seconds,
        diagonal_inverse_seconds=diagonal_inverse_seconds,
        warmup_seconds=warmup_seconds,
        build_seconds=build_seconds,
    )
    preconditioner.onfly_strategy = "ordered-block-coo"
    preconditioner.row_scale = np.ascontiguousarray(row_scale, dtype=np.float64)
    preconditioner.onfly_max_couplings_per_block = int(max_couplings)
    preconditioner.onfly_triplet_counts = {
        "block_entries": int(rows.size),
        "zero_diagonal_rows": int(zero_diagonal_count),
    }
    return preconditioner


def scale_ordered_trace_coo_from_block_gs(row_indices, matrix_values, rhs, preconditioner):
    """Scale ordered trace COO/RHS using the row scale computed by block-GS setup.

    The ordered block-COO preconditioner builder already accumulates diagonal
    blocks and computes the left Jacobi scale needed by the trace solve.  This
    helper reuses that scale instead of scanning the block stream a second time.
    It returns new contiguous arrays and leaves the unscaled assembly output
    untouched.
    """
    _require_numba()
    row_scale = getattr(preconditioner, "row_scale", None)
    if row_scale is None:
        raise ValueError("preconditioner does not expose row_scale; rebuild it with the on-fly builder")

    rows = np.ascontiguousarray(row_indices, dtype=np.int64)
    values = np.ascontiguousarray(matrix_values, dtype=np.float64)
    rhs_array = np.ascontiguousarray(rhs, dtype=np.float64)
    row_scale_array = np.ascontiguousarray(row_scale, dtype=np.float64)
    if row_scale_array.shape != rhs_array.shape:
        raise ValueError(
            f"row_scale shape {row_scale_array.shape} does not match RHS shape {rhs_array.shape}"
        )
    if rows.shape != values.shape:
        raise ValueError("row_indices and matrix_values must have matching shapes")

    scaled_values = np.empty_like(values)
    scaled_rhs = np.empty_like(rhs_array)
    _scale_ordered_coo_rows_kernel(rows, values, rhs_array, row_scale_array, scaled_values, scaled_rhs)
    return np.ascontiguousarray(scaled_values), np.ascontiguousarray(scaled_rhs)


__all__ = [
    "UpwindBlockGSOnTheFlyTimings",
    "build_forward_upwind_block_gs_from_coo",
    "build_forward_upwind_block_gs_from_ordered_block_coo",
    "scale_ordered_trace_coo_from_block_gs",
]
