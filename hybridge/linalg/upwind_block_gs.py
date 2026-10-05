"""Level-scheduled upwind block Gauss-Seidel preconditioners.

The routines here build a reusable SciPy ``LinearOperator`` from an already
assembled, upwind-ordered CSR trace matrix.  The first implementation is a
conservative benchmark tool: it keeps diagonal edge blocks and strictly
upstream cross-level block couplings, while dropping same-level and downstream
couplings so that each topological level can be applied in parallel.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import LinearOperator

try:  # pragma: no cover - exercised only when numba is available.
    from numba import njit, prange
except ImportError:  # pragma: no cover
    njit = None
    prange = range


@dataclass(frozen=True)
class UpwindBlockGSStats:
    """Structural diagnostics for an upwind block-GS preconditioner."""

    num_blocks: int
    block_size: int
    num_levels: int
    max_width: int
    median_width: float
    mean_width: float
    retained_block_couplings: int
    downstream_block_couplings: int
    dropped_same_level_couplings: int
    dropped_downstream_couplings: int
    dropped_coupling_fraction: float
    apply_mode: str
    sweep: str
    csr_prepare_seconds: float
    coupling_count_seconds: float
    block_fill_seconds: float
    diagonal_inverse_seconds: float
    warmup_seconds: float
    build_seconds: float


def _require_numba():
    """Return Numba or raise the optional-dependency error for this backend."""
    if njit is None:
        raise RuntimeError("upwind block Gauss-Seidel preconditioner requires numba")


def _njit(*args, **kwargs):
    """Compile with Numba when available or provide an identity decorator."""
    if njit is None:  # pragma: no cover
        def decorator(function):
            """Return the decorated function unchanged when Numba is unavailable."""
            return function

        return decorator
    return njit(*args, **kwargs)


@_njit(cache=True)
def _count_block_couplings(indptr, indices, num_blocks, block_size, block_levels):
    """Count lower, diagonal, and upper block couplings in matrix COO data."""
    retained = np.zeros(num_blocks, dtype=np.int64)
    same_level = np.zeros(num_blocks, dtype=np.int64)
    downstream = np.zeros(num_blocks, dtype=np.int64)

    retained_seen = np.full(num_blocks, -1, dtype=np.int64)
    same_seen = np.full(num_blocks, -1, dtype=np.int64)
    downstream_seen = np.full(num_blocks, -1, dtype=np.int64)

    for block in range(num_blocks):
        row_level = block_levels[block]
        row_begin = block * block_size
        row_end = row_begin + block_size
        for row in range(row_begin, row_end):
            for scalar_pos in range(indptr[row], indptr[row + 1]):
                col_block = indices[scalar_pos] // block_size
                if col_block == block:
                    continue

                col_level = block_levels[col_block]
                if col_level < row_level:
                    if retained_seen[col_block] != block:
                        retained_seen[col_block] = block
                        retained[block] += 1
                elif col_level == row_level:
                    if same_seen[col_block] != block:
                        same_seen[col_block] = block
                        same_level[block] += 1
                else:
                    if downstream_seen[col_block] != block:
                        downstream_seen[col_block] = block
                        downstream[block] += 1

    return retained, same_level, downstream


@_njit(cache=True)
def _fill_block_data(
        indptr,
        indices,
        data,
        lower_row_ptr,
        lower_col_ind,
        lower_values,
        upper_row_ptr,
        upper_col_ind,
        upper_values,
        diagonal_blocks,
        block_size,
        block_levels,
):
    """Partition the scalar matrix into diagonal and ordered block couplings."""
    num_blocks = lower_row_ptr.size - 1
    lower_seen = np.full(num_blocks, -1, dtype=np.int64)
    upper_seen = np.full(num_blocks, -1, dtype=np.int64)
    lower_positions = np.empty(num_blocks, dtype=np.int64)
    upper_positions = np.empty(num_blocks, dtype=np.int64)

    for block in range(num_blocks):
        row_level = block_levels[block]
        next_lower_pos = lower_row_ptr[block]
        next_upper_pos = upper_row_ptr[block]
        row_begin = block * block_size

        for local_row in range(block_size):
            row = row_begin + local_row
            for scalar_pos in range(indptr[row], indptr[row + 1]):
                col = indices[scalar_pos]
                col_block = col // block_size
                local_col = col - col_block * block_size
                value = data[scalar_pos]

                if col_block == block:
                    diagonal_blocks[block, local_row, local_col] += value
                elif block_levels[col_block] < row_level:
                    if lower_seen[col_block] != block:
                        lower_seen[col_block] = block
                        lower_positions[col_block] = next_lower_pos
                        lower_col_ind[next_lower_pos] = col_block
                        next_lower_pos += 1
                    lower_values[lower_positions[col_block], local_row, local_col] += value
                elif block_levels[col_block] > row_level:
                    if upper_seen[col_block] != block:
                        upper_seen[col_block] = block
                        upper_positions[col_block] = next_upper_pos
                        upper_col_ind[next_upper_pos] = col_block
                        next_upper_pos += 1
                    upper_values[upper_positions[col_block], local_row, local_col] += value


@_njit(cache=True, parallel=True)
def _forward_level_sweep(
        level_offsets,
        block_row_ptr,
        block_col_ind,
        block_values,
        diagonal_inverse,
        vector,
        out,
        block_size,
):
    """Apply one forward Gauss-Seidel sweep over block levels."""

    num_levels = level_offsets.size - 1

    for level in range(num_levels):
        start = level_offsets[level]
        stop = level_offsets[level + 1]
        for block in prange(start, stop):
            block_base = block * block_size
            residual = np.empty(block_size, dtype=np.float64)

            for i in range(block_size):
                residual[i] = vector[block_base + i]

            for block_pos in range(block_row_ptr[block], block_row_ptr[block + 1]):
                col_block = block_col_ind[block_pos]
                col_base = col_block * block_size
                for i in range(block_size):
                    coupling = 0.0
                    for j in range(block_size):
                        coupling += block_values[block_pos, i, j] * out[col_base + j]
                    residual[i] -= coupling

            for i in range(block_size):
                value = 0.0
                for j in range(block_size):
                    value += diagonal_inverse[block, i, j] * residual[j]
                out[block_base + i] = value


@_njit(cache=True)
def _forward_level_sweep_serial(
        level_offsets,
        block_row_ptr,
        block_col_ind,
        block_values,
        diagonal_inverse,
        vector,
        out,
        block_size,
):
    """Apply a serial forward sweep over block levels."""

    num_levels = level_offsets.size - 1

    for level in range(num_levels):
        start = level_offsets[level]
        stop = level_offsets[level + 1]
        for block in range(start, stop):
            block_base = block * block_size
            residual = np.empty(block_size, dtype=np.float64)

            for i in range(block_size):
                residual[i] = vector[block_base + i]

            for block_pos in range(block_row_ptr[block], block_row_ptr[block + 1]):
                col_block = block_col_ind[block_pos]
                col_base = col_block * block_size
                for i in range(block_size):
                    coupling = 0.0
                    for j in range(block_size):
                        coupling += block_values[block_pos, i, j] * out[col_base + j]
                    residual[i] -= coupling

            for i in range(block_size):
                value = 0.0
                for j in range(block_size):
                    value += diagonal_inverse[block, i, j] * residual[j]
                out[block_base + i] = value


@_njit(cache=True, parallel=True)
def _backward_level_sweep(
        level_offsets,
        block_row_ptr,
        block_col_ind,
        block_values,
        diagonal_inverse,
        vector,
        out,
        block_size,
):
    """Apply one parallel backward Gauss-Seidel sweep over block levels."""
    num_levels = level_offsets.size - 1

    for level in range(num_levels - 1, -1, -1):
        start = level_offsets[level]
        stop = level_offsets[level + 1]
        for block in prange(start, stop):
            block_base = block * block_size
            residual = np.empty(block_size, dtype=np.float64)

            for i in range(block_size):
                residual[i] = vector[block_base + i]

            for block_pos in range(block_row_ptr[block], block_row_ptr[block + 1]):
                col_block = block_col_ind[block_pos]
                col_base = col_block * block_size
                for i in range(block_size):
                    coupling = 0.0
                    for j in range(block_size):
                        coupling += block_values[block_pos, i, j] * out[col_base + j]
                    residual[i] -= coupling

            for i in range(block_size):
                value = 0.0
                for j in range(block_size):
                    value += diagonal_inverse[block, i, j] * residual[j]
                out[block_base + i] = value


@_njit(cache=True)
def _backward_level_sweep_serial(
        level_offsets,
        block_row_ptr,
        block_col_ind,
        block_values,
        diagonal_inverse,
        vector,
        out,
        block_size,
):
    """Apply one serial backward Gauss-Seidel sweep over block levels."""
    num_levels = level_offsets.size - 1

    for level in range(num_levels - 1, -1, -1):
        start = level_offsets[level]
        stop = level_offsets[level + 1]
        for block in range(start, stop):
            block_base = block * block_size
            residual = np.empty(block_size, dtype=np.float64)

            for i in range(block_size):
                residual[i] = vector[block_base + i]

            for block_pos in range(block_row_ptr[block], block_row_ptr[block + 1]):
                col_block = block_col_ind[block_pos]
                col_base = col_block * block_size
                for i in range(block_size):
                    coupling = 0.0
                    for j in range(block_size):
                        coupling += block_values[block_pos, i, j] * out[col_base + j]
                    residual[i] -= coupling

            for i in range(block_size):
                value = 0.0
                for j in range(block_size):
                    value += diagonal_inverse[block, i, j] * residual[j]
                out[block_base + i] = value


@_njit(cache=True)
def _apply_diagonal_blocks(diagonal_blocks, vector, out, block_size):
    """Apply inverse diagonal blocks independently to a block vector."""
    num_blocks = diagonal_blocks.shape[0]
    for block in range(num_blocks):
        block_base = block * block_size
        for i in range(block_size):
            value = 0.0
            for j in range(block_size):
                value += diagonal_blocks[block, i, j] * vector[block_base + j]
            out[block_base + i] = value


class UpwindBlockGSPreconditioner(LinearOperator):
    """SciPy ``LinearOperator`` applying a level-scheduled block-GS sweep."""

    def __init__(
            self,
            *,
            level_offsets: np.ndarray,
            lower_row_ptr: np.ndarray,
            lower_col_ind: np.ndarray,
            lower_values: np.ndarray,
            upper_row_ptr: np.ndarray,
            upper_col_ind: np.ndarray,
            upper_values: np.ndarray,
            diagonal_blocks: np.ndarray,
            diagonal_inverse: np.ndarray,
            stats: UpwindBlockGSStats,
            apply_mode: str,
            sweep: str,
    ):
        """Construct and validate a level-scheduled block-GS operator."""
        self.level_offsets = np.ascontiguousarray(level_offsets, dtype=np.int64)
        self.lower_row_ptr = np.ascontiguousarray(lower_row_ptr, dtype=np.int64)
        self.lower_col_ind = np.ascontiguousarray(lower_col_ind, dtype=np.int64)
        self.lower_values = np.ascontiguousarray(lower_values, dtype=np.float64)
        self.upper_row_ptr = np.ascontiguousarray(upper_row_ptr, dtype=np.int64)
        self.upper_col_ind = np.ascontiguousarray(upper_col_ind, dtype=np.int64)
        self.upper_values = np.ascontiguousarray(upper_values, dtype=np.float64)
        self.diagonal_blocks = np.ascontiguousarray(diagonal_blocks, dtype=np.float64)
        self.diagonal_inverse = np.ascontiguousarray(diagonal_inverse, dtype=np.float64)
        self.stats = stats
        self.apply_mode = apply_mode
        self.sweep = sweep

        self.apply_count = 0
        self.apply_seconds = 0.0
        self.local_solve_seconds = 0.0
        self.reduce_seconds = 0.0
        self.copy_seconds = 0.0

        shape = (stats.num_blocks * stats.block_size, stats.num_blocks * stats.block_size)
        super().__init__(dtype=np.dtype(np.float64), shape=shape)

    def _matvec(self, vector):
        """Apply the block Gauss-Seidel operator to one vector."""
        vector = np.ascontiguousarray(vector, dtype=np.float64)
        if vector.shape != (self.shape[1],):
            raise ValueError(f"vector must have shape ({self.shape[1]},), got {vector.shape}")

        out = np.empty_like(vector)
        start = time.perf_counter()
        if self.apply_mode == "parallel":
            _forward_level_sweep(
                self.level_offsets,
                self.lower_row_ptr,
                self.lower_col_ind,
                self.lower_values,
                self.diagonal_inverse,
                vector,
                out,
                self.stats.block_size,
            )
        else:
            _forward_level_sweep_serial(
                self.level_offsets,
                self.lower_row_ptr,
                self.lower_col_ind,
                self.lower_values,
                self.diagonal_inverse,
                vector,
                out,
                self.stats.block_size,
            )
        if self.sweep == "forward_backward":
            diagonal_out = np.empty_like(vector)
            _apply_diagonal_blocks(self.diagonal_blocks, out, diagonal_out, self.stats.block_size)
            if self.apply_mode == "parallel":
                _backward_level_sweep(
                    self.level_offsets,
                    self.upper_row_ptr,
                    self.upper_col_ind,
                    self.upper_values,
                    self.diagonal_inverse,
                    diagonal_out,
                    out,
                    self.stats.block_size,
                )
            else:
                _backward_level_sweep_serial(
                    self.level_offsets,
                    self.upper_row_ptr,
                    self.upper_col_ind,
                    self.upper_values,
                    self.diagonal_inverse,
                    diagonal_out,
                    out,
                    self.stats.block_size,
                )
        elapsed = time.perf_counter() - start

        self.apply_count += 1
        self.apply_seconds += elapsed
        self.local_solve_seconds += elapsed
        return out

    def _matmat(self, matrix):
        """Apply the block Gauss-Seidel operator to multiple vectors."""
        matrix = np.asarray(matrix, dtype=np.float64)
        columns = [self._matvec(matrix[:, col]) for col in range(matrix.shape[1])]
        return np.column_stack(columns)

    def reset_timing(self) -> None:
        """Reset application counters after optional JIT warm-up."""
        self.apply_count = 0
        self.apply_seconds = 0.0
        self.local_solve_seconds = 0.0
        self.reduce_seconds = 0.0
        self.copy_seconds = 0.0


def _level_width_array(level_widths: Sequence[int] | object) -> np.ndarray:
    """Normalize and validate the number of blocks in each level."""
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
    """Build cumulative offsets and per-block level ids from level widths."""
    level_offsets = np.empty(widths.size + 1, dtype=np.int64)
    level_offsets[0] = 0
    np.cumsum(widths, out=level_offsets[1:])
    block_levels = np.repeat(np.arange(widths.size, dtype=np.int64), widths)
    return np.ascontiguousarray(level_offsets), np.ascontiguousarray(block_levels)


def build_upwind_block_gs_preconditioner(
        matrix: scipy.sparse.spmatrix | scipy.sparse.sparray,
        *,
        block_size: int,
        level_widths: Sequence[int] | object,
        diagonal_regularization: float = 0.0,
        apply_mode: str = "auto",
        parallel_min_width: int = 1024,
        sweep: str = "forward",
        warm_start: bool = True,
) -> UpwindBlockGSPreconditioner:
    """Build a level-scheduled upwind block Gauss-Seidel preconditioner.

    Parameters
    ----------
    matrix
        Upwind-ordered square sparse matrix.
    block_size
        Number of trace degrees of freedom per edge block.
    level_widths
        Topological level widths in edge-block units.  A
        ``LevelWidthDiagnostics`` instance may be supplied directly.
    diagonal_regularization
        Optional value added to each dense diagonal block before inversion.
    apply_mode
        ``"serial"``, ``"parallel"``, or ``"auto"``.  The parallel sweep uses
        ``prange`` within each topological level; this only pays off when
        levels are wide enough to amortize thread scheduling.
    parallel_min_width
        Minimum maximum-level width used by ``apply_mode="auto"`` before the
        parallel sweep is selected.
    sweep
        ``"forward"`` keeps only upstream couplings. ``"forward_backward"``
        applies an SSOR-like forward sweep, block-diagonal multiply, and
        backward sweep using downstream couplings.
    warm_start
        Apply the preconditioner once to a zero vector so Numba compilation is
        charged to setup time rather than the Krylov solve.
    """
    _require_numba()
    build_start = time.perf_counter()

    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if apply_mode not in {"auto", "serial", "parallel"}:
        raise ValueError("apply_mode must be 'auto', 'serial', or 'parallel'")
    if sweep not in {"forward", "forward_backward"}:
        raise ValueError("sweep must be 'forward' or 'forward_backward'")
    if parallel_min_width <= 0:
        raise ValueError("parallel_min_width must be positive")
    if not scipy.sparse.issparse(matrix):
        raise TypeError("matrix must be a SciPy sparse matrix or sparse array")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square, got shape {matrix.shape}")
    if matrix.shape[0] % block_size != 0:
        raise ValueError(
            f"matrix size {matrix.shape[0]} is not divisible by block_size={block_size}"
        )

    widths = _level_width_array(level_widths)
    level_offsets, block_levels = _level_offsets_and_block_levels(widths)
    num_blocks = matrix.shape[0] // block_size
    if int(level_offsets[-1]) != num_blocks:
        raise ValueError(
            "level_widths sum does not match matrix block count: "
            f"{int(level_offsets[-1])} != {num_blocks}"
        )

    csr_start = time.perf_counter()
    csr = matrix.tocsr().astype(np.float64, copy=False)
    csr.sum_duplicates()
    csr.sort_indices()

    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    csr_prepare_seconds = time.perf_counter() - csr_start

    count_start = time.perf_counter()
    retained_counts, same_counts, downstream_counts = _count_block_couplings(
        indptr,
        indices,
        num_blocks,
        int(block_size),
        block_levels,
    )
    coupling_count_seconds = time.perf_counter() - count_start

    lower_row_ptr = np.empty(num_blocks + 1, dtype=np.int64)
    lower_row_ptr[0] = 0
    np.cumsum(retained_counts, out=lower_row_ptr[1:])
    upper_row_ptr = np.empty(num_blocks + 1, dtype=np.int64)
    upper_row_ptr[0] = 0
    np.cumsum(downstream_counts, out=upper_row_ptr[1:])

    retained_total = int(lower_row_ptr[-1])
    downstream_total = int(upper_row_ptr[-1])
    lower_col_ind = np.empty(retained_total, dtype=np.int64)
    lower_values = np.zeros((retained_total, block_size, block_size), dtype=np.float64)
    upper_col_ind = np.empty(downstream_total, dtype=np.int64)
    upper_values = np.zeros((downstream_total, block_size, block_size), dtype=np.float64)
    diagonal_blocks = np.zeros((num_blocks, block_size, block_size), dtype=np.float64)

    fill_start = time.perf_counter()
    _fill_block_data(
        indptr,
        indices,
        data,
        lower_row_ptr,
        lower_col_ind,
        lower_values,
        upper_row_ptr,
        upper_col_ind,
        upper_values,
        diagonal_blocks,
        int(block_size),
        block_levels,
    )
    block_fill_seconds = time.perf_counter() - fill_start

    if diagonal_regularization != 0.0:
        diag_ids = np.arange(block_size)
        diagonal_blocks[:, diag_ids, diag_ids] += float(diagonal_regularization)

    inverse_start = time.perf_counter()
    try:
        diagonal_inverse = np.linalg.inv(diagonal_blocks)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "failed to invert at least one edge-block diagonal while building "
            "the upwind block-GS preconditioner"
        ) from exc
    diagonal_inverse_seconds = time.perf_counter() - inverse_start

    dropped_same = int(np.sum(same_counts))
    dropped_downstream = downstream_total if sweep == "forward" else 0
    offdiag_total = retained_total + downstream_total + dropped_same
    dropped_fraction = (
        0.0
        if offdiag_total == 0
        else (dropped_same + dropped_downstream) / offdiag_total
    )
    resolved_apply_mode = apply_mode
    if resolved_apply_mode == "auto":
        resolved_apply_mode = "parallel" if int(np.max(widths)) >= int(parallel_min_width) else "serial"

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
        dropped_downstream_couplings=dropped_downstream,
        dropped_coupling_fraction=float(dropped_fraction),
        apply_mode=resolved_apply_mode,
        sweep=sweep,
        csr_prepare_seconds=csr_prepare_seconds,
        coupling_count_seconds=coupling_count_seconds,
        block_fill_seconds=block_fill_seconds,
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
        apply_mode=resolved_apply_mode,
        sweep=sweep,
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

    return preconditioner


__all__ = [
    "UpwindBlockGSPreconditioner",
    "UpwindBlockGSStats",
    "build_upwind_block_gs_preconditioner",
]
