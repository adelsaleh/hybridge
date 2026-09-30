"""CuPy level-scheduled upwind block Gauss-Seidel preconditioner.

This is an experimental GPU analogue of :mod:`hdgfem.linalg.upwind_block_gs`.
Setup currently reuses the CPU block extraction path and transfers compact
block data to the device.  Application stays on the GPU through a Cupyx
``LinearOperator`` and uses one custom CuPy RawKernel launch per topological
level to fuse block coupling accumulation and diagonal-block solves.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.sparse

from hdgfem.linalg.upwind_block_gs import UpwindBlockGSStats, build_upwind_block_gs_preconditioner


_UPWIND_FORWARD_LEVEL_KERNEL_TEMPLATE = r"""
extern "C" __global__ void upwind_forward_level(
        const long long block_start,
        const long long block_stop,
        const long long* __restrict__ lower_row_ptr,
        const long long* __restrict__ lower_col_ind,
        const {scalar}* __restrict__ lower_values,
        const {scalar}* __restrict__ diagonal_inverse,
        const {scalar}* __restrict__ vector,
        {scalar}* __restrict__ out,
        const int block_size)
{{
    const long long local_block = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long width = block_stop - block_start;
    if (local_block >= width) {{
        return;
    }}

    const long long block = block_start + local_block;
    const long long block_base = block * block_size;
    {scalar} residual[16];

    for (int i = 0; i < block_size; ++i) {{
        residual[i] = vector[block_base + i];
    }}

    for (long long pos = lower_row_ptr[block]; pos < lower_row_ptr[block + 1]; ++pos) {{
        const long long col_block = lower_col_ind[pos];
        const long long col_base = col_block * block_size;
        const long long value_base = pos * block_size * block_size;
        for (int i = 0; i < block_size; ++i) {{
            {scalar} coupling = ({scalar})0;
            const long long row_base = value_base + i * block_size;
            for (int j = 0; j < block_size; ++j) {{
                coupling += lower_values[row_base + j] * out[col_base + j];
            }}
            residual[i] -= coupling;
        }}
    }}

    const long long diag_base = block * block_size * block_size;
    for (int i = 0; i < block_size; ++i) {{
        {scalar} value = ({scalar})0;
        const long long row_base = diag_base + i * block_size;
        for (int j = 0; j < block_size; ++j) {{
            value += diagonal_inverse[row_base + j] * residual[j];
        }}
        out[block_base + i] = value;
    }}
}}
"""


def _forward_level_kernel_source(dtype) -> str:
    """Build the raw CUDA source for one forward level sweep."""
    return _UPWIND_FORWARD_LEVEL_KERNEL_TEMPLATE.format(
        scalar="float" if dtype == np.dtype(np.float32) else "double"
    )


@dataclass(frozen=True)
class CupyUpwindBlockGSStats:
    """Diagnostics for the experimental CuPy upwind block-GS preconditioner."""

    host_stats: UpwindBlockGSStats
    host_setup_seconds: float
    device_transfer_seconds: float
    operator_build_seconds: float


class CupyUpwindBlockGSPreconditioner:
    """Apply a level-scheduled block lower solve on the GPU."""

    def __init__(
            self,
            *,
            cupy,
            cupyx_linalg,
            level_offsets,
            lower_row_ptr,
            lower_col_ind,
            lower_values,
            diagonal_inverse,
            host_level_offsets,
            dtype,
            stats: CupyUpwindBlockGSStats,
    ):
        """Initialize this object."""
        self.cupy = cupy
        self.level_offsets = level_offsets
        self.lower_row_ptr = lower_row_ptr
        self.lower_col_ind = lower_col_ind
        self.lower_values = lower_values
        self.diagonal_inverse = diagonal_inverse
        self.host_level_offsets = np.ascontiguousarray(host_level_offsets, dtype=np.int64)
        self.dtype = np.dtype(dtype)
        self.cupy_dtype = cupy.dtype(dtype)
        self.stats = stats
        self.block_size = int(stats.host_stats.block_size)
        self.num_blocks = int(stats.host_stats.num_blocks)
        self.shape = (self.num_blocks * self.block_size, self.num_blocks * self.block_size)

        if self.block_size > 16:
            raise ValueError("RawKernel upwind block-GS supports block_size <= 16")
        self.threads_per_block = 128
        self._forward_level_kernel = cupy.RawKernel(
            _forward_level_kernel_source(self.dtype),
            "upwind_forward_level",
        )

        self.apply_count = 0
        self.apply_seconds = 0.0
        self.local_solve_seconds = 0.0
        self.reduce_seconds = 0.0
        self.copy_seconds = 0.0

        self.operator = cupyx_linalg.LinearOperator(
            self.shape,
            matvec=self.matvec,
            dtype=self.cupy_dtype,
        )
        self.operator.stats = stats
        self.operator.apply_count = 0
        self.operator.apply_seconds = 0.0
        self.operator.local_solve_seconds = 0.0
        self.operator.reduce_seconds = 0.0
        self.operator.copy_seconds = 0.0
        self.operator._upwind_block_gs_impl = self

    def _record_apply(self, elapsed: float) -> None:
        """Record one preconditioner application and its elapsed device time."""
        self.apply_count += 1
        self.apply_seconds += elapsed
        self.local_solve_seconds += elapsed
        self.operator.apply_count = self.apply_count
        self.operator.apply_seconds = self.apply_seconds
        self.operator.local_solve_seconds = self.local_solve_seconds
        self.operator.reduce_seconds = self.reduce_seconds
        self.operator.copy_seconds = self.copy_seconds

    def reset_timing(self) -> None:
        """Reset accumulated preconditioner call counts and timings."""
        self.apply_count = 0
        self.apply_seconds = 0.0
        self.local_solve_seconds = 0.0
        self.reduce_seconds = 0.0
        self.copy_seconds = 0.0
        self.operator.apply_count = 0
        self.operator.apply_seconds = 0.0
        self.operator.local_solve_seconds = 0.0
        self.operator.reduce_seconds = 0.0
        self.operator.copy_seconds = 0.0

    def matvec(self, vector):
        """Apply a matrix-vector product."""
        cupy = self.cupy
        start_time = time.perf_counter()
        vec = cupy.asarray(vector, dtype=self.cupy_dtype).reshape(-1)
        if vec.size != self.shape[1]:
            raise ValueError(f"vector must have size {self.shape[1]}, got {vec.size}")
        out = cupy.empty_like(vec)

        for level in range(self.host_level_offsets.size - 1):
            block_start = int(self.host_level_offsets[level])
            block_stop = int(self.host_level_offsets[level + 1])
            width = block_stop - block_start
            if width <= 0:
                continue
            grid = ((width + self.threads_per_block - 1) // self.threads_per_block,)
            self._forward_level_kernel(
                grid,
                (self.threads_per_block,),
                (
                    np.int64(block_start),
                    np.int64(block_stop),
                    self.lower_row_ptr,
                    self.lower_col_ind,
                    self.lower_values,
                    self.diagonal_inverse,
                    vec,
                    out,
                    np.int32(self.block_size),
                ),
            )

        elapsed = time.perf_counter() - start_time
        self._record_apply(elapsed)
        return out.reshape(vector.shape)


def cupy_upwind_block_gs_from_host_preconditioner(
        host_prec,
        *,
        dtype=None,
        warm_start: bool = True,
):
    """Transfer an existing host upwind block-GS preconditioner to CuPy.

    ``host_prec`` must be an :class:`UpwindBlockGSPreconditioner` built for the
    already scaled and upwind-ordered matrix.  This routine does not rebuild any
    sparsity or block data on the host; it only transfers the compact level,
    lower-coupling, and inverse-diagonal arrays to the current CUDA device and
    returns the Cupyx ``LinearOperator`` used by iterative solvers.
    """
    from hdgfem.runtime.optional import require_cupy, require_cupyx_sparse_linalg

    cupy = require_cupy()
    cupyx_linalg = require_cupyx_sparse_linalg()
    cupy_dtype = cupy.dtype(cupy.float64 if dtype is None else dtype)
    np_dtype = np.dtype(cupy.asnumpy(cupy.empty((), dtype=cupy_dtype)).dtype)
    if np_dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError("upwind block-GS dtype must be float32 or float64")
    if getattr(host_prec, "sweep", "forward") != "forward":
        raise ValueError("CuPy upwind block-GS currently supports only forward sweeps")

    build_start = time.perf_counter()
    transfer_start = time.perf_counter()
    level_offsets = cupy.asarray(host_prec.level_offsets, dtype=cupy.int64)
    lower_row_ptr = cupy.asarray(host_prec.lower_row_ptr, dtype=cupy.int64)
    lower_col_ind = cupy.asarray(host_prec.lower_col_ind, dtype=cupy.int64)
    lower_values = cupy.asarray(host_prec.lower_values, dtype=cupy_dtype)
    diagonal_inverse = cupy.asarray(host_prec.diagonal_inverse, dtype=cupy_dtype)
    cupy.cuda.get_current_stream().synchronize()
    device_transfer_seconds = time.perf_counter() - transfer_start

    stats = CupyUpwindBlockGSStats(
        host_stats=host_prec.stats,
        host_setup_seconds=float(host_prec.stats.build_seconds),
        device_transfer_seconds=device_transfer_seconds,
        operator_build_seconds=time.perf_counter() - build_start,
    )
    preconditioner = CupyUpwindBlockGSPreconditioner(
        cupy=cupy,
        cupyx_linalg=cupyx_linalg,
        level_offsets=level_offsets,
        lower_row_ptr=lower_row_ptr,
        lower_col_ind=lower_col_ind,
        lower_values=lower_values,
        diagonal_inverse=diagonal_inverse,
        host_level_offsets=host_prec.level_offsets,
        dtype=np_dtype,
        stats=stats,
    )
    if warm_start:
        warmup = cupy.zeros(preconditioner.shape[0], dtype=cupy_dtype)
        preconditioner.matvec(warmup)
        cupy.cuda.get_current_stream().synchronize()
        preconditioner.reset_timing()
    return preconditioner.operator


def build_cupy_upwind_block_gs_preconditioner(
        matrix: scipy.sparse.spmatrix | scipy.sparse.sparray,
        *,
        block_size: int,
        level_widths: Sequence[int] | object,
        diagonal_regularization: float = 0.0,
        dtype=None,
):
    """Build a Cupyx ``LinearOperator`` for an upwind block-GS sweep."""
    from hdgfem.runtime.optional import require_cupy, require_cupyx_sparse_linalg

    cupy = require_cupy()
    cupyx_linalg = require_cupyx_sparse_linalg()
    cupy_dtype = cupy.dtype(cupy.float64 if dtype is None else dtype)
    np_dtype = np.dtype(cupy.asnumpy(cupy.empty((), dtype=cupy_dtype)).dtype)
    if np_dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError("upwind block-GS dtype must be float32 or float64")

    build_start = time.perf_counter()
    host_start = time.perf_counter()
    host_prec = build_upwind_block_gs_preconditioner(
        matrix,
        block_size=block_size,
        level_widths=level_widths,
        diagonal_regularization=diagonal_regularization,
        apply_mode="serial",
        sweep="forward",
        warm_start=False,
    )
    host_setup_seconds = time.perf_counter() - host_start

    transfer_start = time.perf_counter()
    level_offsets = cupy.asarray(host_prec.level_offsets, dtype=cupy.int64)
    lower_row_ptr = cupy.asarray(host_prec.lower_row_ptr, dtype=cupy.int64)
    lower_col_ind = cupy.asarray(host_prec.lower_col_ind, dtype=cupy.int64)
    lower_values = cupy.asarray(host_prec.lower_values, dtype=cupy_dtype)
    diagonal_inverse = cupy.asarray(host_prec.diagonal_inverse, dtype=cupy_dtype)
    cupy.cuda.get_current_stream().synchronize()
    device_transfer_seconds = time.perf_counter() - transfer_start

    stats = CupyUpwindBlockGSStats(
        host_stats=host_prec.stats,
        host_setup_seconds=host_setup_seconds,
        device_transfer_seconds=device_transfer_seconds,
        operator_build_seconds=time.perf_counter() - build_start,
    )
    preconditioner = CupyUpwindBlockGSPreconditioner(
        cupy=cupy,
        cupyx_linalg=cupyx_linalg,
        level_offsets=level_offsets,
        lower_row_ptr=lower_row_ptr,
        lower_col_ind=lower_col_ind,
        lower_values=lower_values,
        diagonal_inverse=diagonal_inverse,
        host_level_offsets=host_prec.level_offsets,
        dtype=np_dtype,
        stats=stats,
    )
    warmup = cupy.zeros(preconditioner.shape[0], dtype=cupy_dtype)
    preconditioner.matvec(warmup)
    cupy.cuda.get_current_stream().synchronize()
    preconditioner.reset_timing()
    return preconditioner.operator


__all__ = [
    "CupyUpwindBlockGSPreconditioner",
    "CupyUpwindBlockGSStats",
    "build_cupy_upwind_block_gs_preconditioner",
    "cupy_upwind_block_gs_from_host_preconditioner",
]
