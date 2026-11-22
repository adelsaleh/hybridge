"""hybridge.linalg.gpu.sparse."""

from __future__ import annotations

import numpy as np
import scipy.sparse.linalg
from typing import Any
from hybridge.runtime.precision import REAL_DTYPE, real_raw_kernel
from hybridge.runtime.optional import require_cupy, require_cupyx_sparse

from dataclasses import dataclass



def scipy_csr_to_cupy(matrix: scipy.sparse.spmatrix | scipy.sparse.sparray, *, dtype=None):
    """Convert a SciPy sparse matrix to a CuPy CSR matrix."""
    cupy = require_cupy()
    sparse = require_cupyx_sparse()
    if dtype is None:
        dtype = REAL_DTYPE
    csr = matrix.tocsr()
    csr.sum_duplicates()
    return sparse.csr_matrix(
        (
            cupy.asarray(csr.data, dtype=dtype),
            cupy.asarray(csr.indices, dtype=cupy.int32),
            cupy.asarray(csr.indptr, dtype=cupy.int32),
        ),
        shape=csr.shape,
    )


def scipy_coo_to_cupy_csr(row_indices, col_indices, matrix_values, shape, *, dtype=None):
    """Copy host COO triplets to the GPU and construct CSR on the device."""
    cupy = require_cupy()
    sparse = require_cupyx_sparse()
    if dtype is None:
        dtype = REAL_DTYPE
    coo = sparse.coo_matrix(
        (
            cupy.asarray(matrix_values, dtype=dtype),
            (
                cupy.asarray(row_indices, dtype=cupy.int32),
                cupy.asarray(col_indices, dtype=cupy.int32),
            ),
        ),
        shape=shape,
    )
    csr = coo.tocsr()
    csr.sum_duplicates()
    cupy.cuda.get_current_stream().synchronize()
    return csr


_CSR_ROW_SCALE_SOURCE = r"""
extern "C" __global__ void diagonal_scale_csr_rows(
        const int* __restrict__ indptr,
        const int* __restrict__ indices,
        double* __restrict__ data,
        double* __restrict__ rhs,
        double* __restrict__ row_diagonal,
        const long long nrows)
{
    const long long row = blockIdx.x;
    if (row >= nrows) {
        return;
    }
    const int start = indptr[row];
    const int end = indptr[row + 1];
    __shared__ double row_scale;
    if (threadIdx.x == 0) {
        double diagonal = 0.0;
        double row_max = 0.0;
        for (int p = start; p < end; ++p) {
            row_max = fmax(row_max, fabs(data[p]));
            if (indices[p] == (int)row) {
                diagonal += data[p];
            }
        }
        double value = diagonal;
        if (!isfinite(value) || fabs(value) <= 1.0e-10 * row_max) {
            value = row_max;
        }
        if (!isfinite(value) || value == 0.0) {
            value = 1.0;
        }
        row_scale = value;
        row_diagonal[row] = value;
        rhs[row] /= value;
    }
    __syncthreads();
    const double inverse = 1.0 / row_scale;
    for (int p = start + threadIdx.x; p < end; p += blockDim.x) {
        data[p] *= inverse;
    }
}
"""


_CSR_ROW_SCALE_KERNELS: dict[int, Any] = {}


def diagonal_scale_cupy_csr_rows_in_place(matrix, rhs):
    """Apply left Jacobi row scaling to a CuPy CSR matrix and RHS in place.

    Each row is divided by its diagonal entry. A diagonal that is non-finite
    or at most ``1e-10`` times the row's largest magnitude falls back to that
    maximum, and an all-zero row is left unscaled. Returns the per-row scale.
    """
    cupy = require_cupy()
    device_id = int(cupy.cuda.runtime.getDevice())
    kernel = _CSR_ROW_SCALE_KERNELS.get(device_id)
    if kernel is None:
        kernel = real_raw_kernel(_CSR_ROW_SCALE_SOURCE, "diagonal_scale_csr_rows")
        _CSR_ROW_SCALE_KERNELS[device_id] = kernel
    nrows = int(rhs.size)
    diagonal = cupy.empty(nrows, dtype=REAL_DTYPE)
    if nrows:
        kernel(
            (nrows,),
            (128,),
            (
                matrix.indptr,
                matrix.indices,
                matrix.data,
                rhs,
                diagonal,
                np.int64(nrows),
            ),
        )
    return diagonal


_CSR_SYMMETRIC_DIAGONAL_SOURCE = r"""
extern "C" __global__ void csr_inverse_sqrt_diagonal(
        const int* __restrict__ indptr,
        const int* __restrict__ indices,
        const double* __restrict__ data,
        double* __restrict__ inverse_sqrt_diagonal,
        const long long nrows)
{
    const long long row = blockIdx.x;
    if (row >= nrows || threadIdx.x != 0) {
        return;
    }
    const int start = indptr[row];
    const int end = indptr[row + 1];
    double value = 0.0;
    double row_max = 0.0;
    for (int p = start; p < end; ++p) {
        row_max = fmax(row_max, fabs(data[p]));
        if (indices[p] == (int)row) {
            value += data[p];
        }
    }
    value = fabs(value);
    // Same robust estimate as diagonal_scale_csr_rows: a tiny or non-finite
    // diagonal falls back to the row maximum, and an empty row to 1.
    if (!isfinite(value) || value <= 1.0e-10 * row_max) {
        value = row_max;
    }
    if (!isfinite(value) || value == 0.0) {
        value = 1.0;
    }
    inverse_sqrt_diagonal[row] = 1.0 / sqrt(value);
}

extern "C" __global__ void symmetric_scale_csr_rows(
        const int* __restrict__ indptr,
        const int* __restrict__ indices,
        double* __restrict__ data,
        double* __restrict__ rhs,
        const double* __restrict__ inverse_sqrt_diagonal,
        const long long nrows)
{
    const long long row = blockIdx.x;
    if (row >= nrows) {
        return;
    }
    const int start = indptr[row];
    const int end = indptr[row + 1];
    const double row_scale = inverse_sqrt_diagonal[row];
    if (threadIdx.x == 0) {
        rhs[row] *= row_scale;
    }
    for (int p = start + threadIdx.x; p < end; p += blockDim.x) {
        data[p] *= row_scale * inverse_sqrt_diagonal[indices[p]];
    }
}
"""


_CSR_SYMMETRIC_SCALE_KERNELS: dict[int, tuple[Any, Any]] = {}


def symmetric_scale_cupy_csr_in_place(matrix, rhs):
    """Apply symmetric Jacobi scaling ``D^-1/2 A D^-1/2`` to CSR/RHS in place.

    The returned vector is ``D^-1/2``. After solving the scaled system for
    ``y``, recover the physical unknown by multiplying ``x = D^-1/2 y``.
    """
    cupy = require_cupy()
    device_id = int(cupy.cuda.runtime.getDevice())
    kernels = _CSR_SYMMETRIC_SCALE_KERNELS.get(device_id)
    if kernels is None:
        diag_kernel = real_raw_kernel(_CSR_SYMMETRIC_DIAGONAL_SOURCE, "csr_inverse_sqrt_diagonal")
        scale_kernel = real_raw_kernel(_CSR_SYMMETRIC_DIAGONAL_SOURCE, "symmetric_scale_csr_rows")
        kernels = (diag_kernel, scale_kernel)
        _CSR_SYMMETRIC_SCALE_KERNELS[device_id] = kernels
    diag_kernel, scale_kernel = kernels
    nrows = int(rhs.size)
    inverse_sqrt_diagonal = cupy.empty(nrows, dtype=REAL_DTYPE)
    if nrows:
        diag_kernel(
            (nrows,),
            (128,),
            (matrix.indptr, matrix.indices, matrix.data, inverse_sqrt_diagonal, np.int64(nrows)),
        )
        scale_kernel(
            (nrows,),
            (128,),
            (matrix.indptr, matrix.indices, matrix.data, rhs, inverse_sqrt_diagonal, np.int64(nrows)),
        )
    return inverse_sqrt_diagonal


@dataclass(frozen=True)
class _DeviceCsrMatrixView:
    """Device-owned scalar CSR arrays accepted by PyAMGX Matrix.upload."""

    data: Any
    indices: Any
    indptr: Any
    shape: tuple[int, int]


@dataclass(frozen=True)
class _DeviceBsrMatrixView:
    """Device-owned face-BSR arrays accepted directly by PyAMGX."""

    data: Any
    indices: Any
    indptr: Any
    shape: tuple[int, int]
    block_size: int


_DEVICE_BSR_MATVEC_SOURCE = r"""
extern "C" __global__ void device_bsr_matvec(
        const int* __restrict__ indptr,
        const int* __restrict__ indices,
        const double* __restrict__ data,
        const double* __restrict__ x,
        double* __restrict__ y,
        const int num_block_rows,
        const int block_size)
{
    const int scalar_row = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_rows = num_block_rows * block_size;
    if (scalar_row >= num_rows) {
        return;
    }
    const int block_row = scalar_row / block_size;
    const int row_dof = scalar_row - block_row * block_size;
    double value = 0.0;
    for (int block = indptr[block_row]; block < indptr[block_row + 1]; ++block) {
        const int column_base = indices[block] * block_size;
        const long long data_base = (
            (long long)block * block_size + row_dof
        ) * block_size;
        for (int col_dof = 0; col_dof < block_size; ++col_dof) {
            value += data[data_base + col_dof] * x[column_base + col_dof];
        }
    }
    y[scalar_row] = value;
}
"""


_DEVICE_BSR_MATVEC_KERNELS: dict[int, Any] = {}


def _as_cupyx_csr_matrix(matrix, sparse, cp):
    """Expose a scalar compressed device view as a Cupyx CSR matrix."""
    if isinstance(matrix, _DeviceCsrMatrixView):
        return sparse.csr_matrix(
            (matrix.data, matrix.indices, matrix.indptr),
            shape=matrix.shape,
            dtype=REAL_DTYPE,
        )
    return matrix


_DEVICE_BSR_TO_SCALAR_CSR_SOURCE = r"""
extern "C" __global__ void device_bsr_to_scalar_csr(
        const int* __restrict__ block_indptr,
        const int* __restrict__ block_indices,
        const double* __restrict__ block_data,
        int* __restrict__ scalar_indptr,
        int* __restrict__ scalar_indices,
        double* __restrict__ scalar_data,
        const int num_block_rows,
        const int block_size)
{
    const int scalar_row = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_scalar_rows = num_block_rows * block_size;
    if (scalar_row > num_scalar_rows) {
        return;
    }
    if (scalar_row == num_scalar_rows) {
        scalar_indptr[scalar_row] =
            block_indptr[num_block_rows] * block_size * block_size;
        return;
    }
    const int block_row = scalar_row / block_size;
    const int row_dof = scalar_row - block_row * block_size;
    const int block_begin = block_indptr[block_row];
    const int block_end = block_indptr[block_row + 1];
    const int blocks_in_row = block_end - block_begin;
    const int scalar_begin =
        block_begin * block_size * block_size
        + row_dof * blocks_in_row * block_size;
    scalar_indptr[scalar_row] = scalar_begin;
    int output = scalar_begin;
    for (int block = block_begin; block < block_end; ++block) {
        const int scalar_column = block_indices[block] * block_size;
        const long long data_begin =
            ((long long)block * block_size + row_dof) * block_size;
        for (int column_dof = 0; column_dof < block_size; ++column_dof) {
            scalar_indices[output] = scalar_column + column_dof;
            scalar_data[output] = block_data[data_begin + column_dof];
            ++output;
        }
    }
}
"""


_DEVICE_BSR_TO_SCALAR_CSR_KERNELS: dict[int, Any] = {}


def _scalarize_device_bsr_matrix(matrix: _DeviceBsrMatrixView, sparse, cp):
    """Expand a face-BSR view to scalar CUDA CSR without host staging."""
    block_size = int(matrix.block_size)
    num_block_rows = int(matrix.shape[0] // block_size)
    scalar_indptr = cp.empty(matrix.shape[0] + 1, dtype=cp.int32)
    scalar_indices = cp.empty(int(matrix.data.size), dtype=cp.int32)
    scalar_data = cp.empty(int(matrix.data.size), dtype=REAL_DTYPE)
    device_id = int(cp.cuda.runtime.getDevice())
    kernel = _DEVICE_BSR_TO_SCALAR_CSR_KERNELS.get(device_id)
    if kernel is None:
        kernel = real_raw_kernel(
            _DEVICE_BSR_TO_SCALAR_CSR_SOURCE, "device_bsr_to_scalar_csr"
        )
        _DEVICE_BSR_TO_SCALAR_CSR_KERNELS[device_id] = kernel
    threads = 256
    rows_with_terminal = int(matrix.shape[0]) + 1
    kernel(
        ((rows_with_terminal + threads - 1) // threads,),
        (threads,),
        (
            matrix.indptr,
            matrix.indices,
            matrix.data,
            scalar_indptr,
            scalar_indices,
            scalar_data,
            np.int32(num_block_rows),
            np.int32(block_size),
        ),
    )
    return sparse.csr_matrix(
        (scalar_data, scalar_indices, scalar_indptr),
        shape=matrix.shape,
        dtype=REAL_DTYPE,
    )


def _device_compressed_matvec(matrix, vector, sparse, cp):
    """Apply a device CSR or face-BSR matrix without host materialization."""
    if not isinstance(matrix, _DeviceBsrMatrixView):
        return _as_cupyx_csr_matrix(matrix, sparse, cp) @ vector
    output = cp.empty(matrix.shape[0], dtype=REAL_DTYPE)
    threads = 256
    blocks = (matrix.shape[0] + threads - 1) // threads
    device_id = int(cp.cuda.runtime.getDevice())
    kernel = _DEVICE_BSR_MATVEC_KERNELS.get(device_id)
    if kernel is None:
        kernel = real_raw_kernel(_DEVICE_BSR_MATVEC_SOURCE, "device_bsr_matvec")
        _DEVICE_BSR_MATVEC_KERNELS[device_id] = kernel
    kernel(
        (blocks,),
        (threads,),
        (
            matrix.indptr,
            matrix.indices,
            matrix.data,
            vector,
            output,
            np.int32(matrix.shape[0] // matrix.block_size),
            np.int32(matrix.block_size),
        ),
    )
    return output


def _assembly_device_csr_matrix(assembly, cp, sparse):
    """Build a device CSR matrix view from assembled COO or CSR data."""
    system_size = int(assembly.rhs.size)
    matrix_format = getattr(assembly, "matrix_format", "coo")
    if matrix_format == "csr":
        if assembly.indptr is None or assembly.indices is None:
            raise RuntimeError("CSR assembly is missing indptr/indices")
        return _DeviceCsrMatrixView(
            data=assembly.data,
            indices=assembly.indices.astype(cp.int32, copy=False),
            indptr=assembly.indptr.astype(cp.int32, copy=False),
            shape=(system_size, system_size),
        )
    if matrix_format == "bsr":
        if assembly.indptr is None or assembly.indices is None:
            raise RuntimeError("BSR assembly is missing block indptr/indices")
        block_size = int(assembly.data.shape[-1])
        return _DeviceBsrMatrixView(
            data=assembly.data,
            indices=assembly.indices.astype(cp.int32, copy=False),
            indptr=assembly.indptr.astype(cp.int32, copy=False),
            shape=(system_size, system_size),
            block_size=block_size,
        )

    matrix = sparse.coo_matrix(
        (assembly.data, (assembly.rows.astype(cp.int32), assembly.cols.astype(cp.int32))),
        shape=(system_size, system_size),
        dtype=REAL_DTYPE,
    ).tocsr()
    matrix.sum_duplicates()
    if matrix.indices.dtype != cp.int32 or matrix.indptr.dtype != cp.int32:
        matrix = sparse.csr_matrix(
            (
                matrix.data,
                matrix.indices.astype(cp.int32, copy=False),
                matrix.indptr.astype(cp.int32, copy=False),
            ),
            shape=matrix.shape,
            dtype=REAL_DTYPE,
        )
    return matrix


_CSR_ROW_UNSCALE_SOURCE = r"""
extern "C" __global__ void restore_left_scaled_csr_rows(
        const int* __restrict__ indptr,
        double* __restrict__ data,
        const double* __restrict__ row_diagonal,
        const long long nrows)
{
    const long long row = blockIdx.x;
    if (row >= nrows) {
        return;
    }
    const int start = indptr[row];
    const int end = indptr[row + 1];
    const double row_scale = row_diagonal[row];
    for (int p = start + threadIdx.x; p < end; p += blockDim.x) {
        data[p] *= row_scale;
    }
}

extern "C" __global__ void restore_symmetric_scaled_csr_rows(
        const int* __restrict__ indptr,
        const int* __restrict__ indices,
        double* __restrict__ data,
        const double* __restrict__ inverse_sqrt_diagonal,
        const long long nrows)
{
    const long long row = blockIdx.x;
    if (row >= nrows) {
        return;
    }
    const int start = indptr[row];
    const int end = indptr[row + 1];
    const double row_scale = inverse_sqrt_diagonal[row];
    for (int p = start + threadIdx.x; p < end; p += blockDim.x) {
        data[p] /= row_scale * inverse_sqrt_diagonal[indices[p]];
    }
}
"""


_CSR_ROW_UNSCALE_KERNELS: dict[int, tuple[Any, Any]] = {}


def _restore_scaled_csr_rows_in_place(
    matrix,
    *,
    row_diagonal=None,
    inverse_sqrt_diagonal=None,
):
    """Restore CSR values after left or symmetric device scaling."""
    if row_diagonal is None and inverse_sqrt_diagonal is None:
        return
    if row_diagonal is not None and inverse_sqrt_diagonal is not None:
        raise ValueError("exactly one CSR scaling vector may be restored")
    cp = require_cupy()
    device_id = int(cp.cuda.runtime.getDevice())
    kernels = _CSR_ROW_UNSCALE_KERNELS.get(device_id)
    if kernels is None:
        left_kernel = real_raw_kernel(_CSR_ROW_UNSCALE_SOURCE, "restore_left_scaled_csr_rows")
        symmetric_kernel = real_raw_kernel(
            _CSR_ROW_UNSCALE_SOURCE,
            "restore_symmetric_scaled_csr_rows",
        )
        kernels = (left_kernel, symmetric_kernel)
        _CSR_ROW_UNSCALE_KERNELS[device_id] = kernels
    nrows = int(matrix.shape[0])
    if not nrows:
        return
    left_kernel, symmetric_kernel = kernels
    if row_diagonal is not None:
        left_kernel(
            (nrows,),
            (128,),
            (matrix.indptr, matrix.data, row_diagonal, np.int64(nrows)),
        )
    else:
        symmetric_kernel(
            (nrows,),
            (128,),
            (matrix.indptr, matrix.indices, matrix.data, inverse_sqrt_diagonal, np.int64(nrows)),
        )


# Left row scaling of a face-BSR matrix. A block row's values (nnzb_row, b, b) are one
# contiguous slab, so a group of LANES threads per block row reads and writes it coalesced,
# GROUPS block rows per CUDA block. The lanes first find each scalar row's diagonal entry and
# largest magnitude, the rule of ``diagonal_scale_cupy_csr_rows_in_place``, summing and comparing in
# the same order, so every launch shape gives bitwise the result of the earlier
# one-block-per-scalar-row kernel. The face block size b = p+1 (2..10 for p = 1..9) is a runtime
# argument, any b <= 32; the launch shape is tuned per b (see ``_bsr_row_scale_choice``).
_BSR_ROW_SCALE_TEMPLATE = r"""
#define LANES @LANES@
#define GROUPS @GROUPS@
#define MAX_BLOCK 32

extern "C" __global__ void diagonal_scale_bsr_block_rows(
        const int* __restrict__ indptr,
        const int* __restrict__ indices,
        double* __restrict__ data,
        double* __restrict__ rhs,
        double* __restrict__ row_diagonal,
        const int num_block_rows,
        const int block_size)
{
    __shared__ double inverse_scale[GROUPS][MAX_BLOCK];
    const int group = threadIdx.x / LANES;
    const int lane = threadIdx.x % LANES;
    const int block_row = blockIdx.x * GROUPS + group;
    // No early return: groups share warps, and every lane must reach __syncwarp.
    const bool active = block_row < num_block_rows;
    const int b = block_size;
    const int start = active ? indptr[block_row] : 0;
    const int end = active ? indptr[block_row + 1] : 0;
    if (active) {
        for (int row_dof = lane; row_dof < b; row_dof += LANES) {
            double diagonal = 0.0;
            double row_max = 0.0;
            for (int block = start; block < end; ++block) {
                const long long base = ((long long)block * b + row_dof) * b;
                const bool diagonal_block = indices[block] == block_row;
                for (int col_dof = 0; col_dof < b; ++col_dof) {
                    const double entry = data[base + col_dof];
                    row_max = fmax(row_max, fabs(entry));
                    if (diagonal_block && col_dof == row_dof) {
                        diagonal += entry;
                    }
                }
            }
            double value = diagonal;
            if (!isfinite(value) || fabs(value) <= 1.0e-10 * row_max) {
                value = row_max;
            }
            if (!isfinite(value) || value == 0.0) {
                value = 1.0;
            }
            const long long row = (long long)block_row * b + row_dof;
            row_diagonal[row] = value;
            rhs[row] /= value;
            inverse_scale[group][row_dof] = 1.0 / value;
        }
    }
    __syncthreads();
    if (active) {
        const int block_entries = b * b;
        const long long slab = (long long)start * block_entries;
        const int entries = (end - start) * block_entries;
        for (int entry = lane; entry < entries; entry += LANES) {
            data[slab + entry] *= inverse_scale[group][(entry % block_entries) / b];
        }
    }
}

extern "C" __global__ void restore_left_scaled_bsr_block_rows(
        const int* __restrict__ indptr,
        double* __restrict__ data,
        const double* __restrict__ row_diagonal,
        const int num_block_rows,
        const int block_size)
{
    __shared__ double row_scale[GROUPS][MAX_BLOCK];
    const int group = threadIdx.x / LANES;
    const int lane = threadIdx.x % LANES;
    const int block_row = blockIdx.x * GROUPS + group;
    const bool active = block_row < num_block_rows;
    const int b = block_size;
    if (active) {
        for (int row_dof = lane; row_dof < b; row_dof += LANES) {
            row_scale[group][row_dof] = row_diagonal[(long long)block_row * b + row_dof];
        }
    }
    __syncthreads();
    if (active) {
        const int start = indptr[block_row];
        const int end = indptr[block_row + 1];
        const int block_entries = b * b;
        const long long slab = (long long)start * block_entries;
        const int entries = (end - start) * block_entries;
        for (int entry = lane; entry < entries; entry += LANES) {
            data[slab + entry] *= row_scale[group][(entry % block_entries) / b];
        }
    }
}
"""

_BSR_SCALE_MAX_BLOCK = 32
# (lanes per block row, block rows per CUDA block); the first is the untuned default.
_BSR_SCALE_SHAPES = ((32, 4), (32, 2), (32, 8), (16, 4), (16, 8), (16, 16), (8, 8), (8, 16))
_BSR_SCALE_SAMPLE_ROWS = 65536
_BSR_SCALE_SAMPLE_BYTES = 128 * 1024**2

_BSR_ROW_SCALE_KERNELS: dict[tuple, tuple[Any, Any]] = {}
_BSR_ROW_SCALE_CHOICES: dict[tuple, tuple[int, int]] = {}


def _bsr_scale_shape_name(shape) -> str:
    """Name a launch shape for the tuner and the kernel log."""
    lanes, groups = shape
    return f"lanes{lanes}x{groups}"


def _bsr_row_scale_kernels(shape):
    """Return the cached scale/restore kernels compiled for one launch shape."""
    cp = require_cupy()
    lanes, groups = shape
    key = (int(cp.cuda.runtime.getDevice()), lanes, groups)
    kernels = _BSR_ROW_SCALE_KERNELS.get(key)
    if kernels is None:
        source = _BSR_ROW_SCALE_TEMPLATE.replace("@LANES@", str(lanes)).replace("@GROUPS@", str(groups))
        kernels = (
            real_raw_kernel(source, "diagonal_scale_bsr_block_rows"),
            real_raw_kernel(source, "restore_left_scaled_bsr_block_rows"),
        )
        _BSR_ROW_SCALE_KERNELS[key] = kernels
    return kernels


def _bsr_scale_launch(num_block_rows: int, shape):
    """Return the grid and block of one launch shape."""
    lanes, groups = shape
    return (max(1, -(-int(num_block_rows) // groups)),), (lanes * groups,)


def _launch_bsr_row_scale(shape, indptr, indices, data, rhs, diagonal, num_block_rows, block_size):
    """Scale ``num_block_rows`` block rows in place with one launch shape."""
    grid, block = _bsr_scale_launch(num_block_rows, shape)
    scale_kernel, _ = _bsr_row_scale_kernels(shape)
    scale_kernel(grid, block, (indptr, indices, data, rhs, diagonal,
                               np.int32(num_block_rows), np.int32(block_size)))


def _bsr_row_scale_choice(matrix: _DeviceBsrMatrixView):
    """Return the measured fastest launch shape for this block size, machine and precision.

    The tuner times every shape on a private copy of the first block rows (at most 65,536 rows or
    128 MB), checks that they agree, and caches the choice on disk; later processes only compile it.
    """
    cp = require_cupy()
    block_size = int(matrix.block_size)
    num_block_rows = int(matrix.shape[0]) // block_size
    key = (int(cp.cuda.runtime.getDevice()), block_size, str(matrix.data.dtype),
           max(1, num_block_rows).bit_length())
    choice = _BSR_ROW_SCALE_CHOICES.get(key)
    if choice is not None:
        return choice
    from hybridge.runtime.cuda_tuning import fingerprint, select_variant
    from hybridge.linalg.gpu.solver_tuning import module_source_fingerprint

    shapes = {_bsr_scale_shape_name(shape): shape for shape in _BSR_SCALE_SHAPES}
    trial = {}

    def trial_prepare(name, handle):
        """Copy the sample once, on a cache miss only; sizing it reads two scalars, not indptr."""
        if not trial:
            # The first block rows keep global column indices consistent with their row numbers.
            bytes_per_block = block_size * block_size * int(matrix.data.itemsize)
            budget_blocks = max(1, _BSR_SCALE_SAMPLE_BYTES // bytes_per_block)
            head = matrix.indptr[: min(num_block_rows, _BSR_SCALE_SAMPLE_ROWS) + 1]
            limit = cp.asarray(budget_blocks, dtype=head.dtype)
            rows = max(1, int(cp.searchsorted(head, limit, side="right").item()) - 1)
            blocks = int(matrix.indptr[rows].item())
            trial.update(
                rows=rows,
                indptr=matrix.indptr[: rows + 1], indices=matrix.indices[:blocks],
                original=matrix.data[:blocks].copy(), data=matrix.data[:blocks].copy(),
                original_rhs=cp.arange(1, rows * block_size + 1, dtype=matrix.data.dtype),
                rhs=cp.empty(rows * block_size, dtype=matrix.data.dtype),
                diagonal=cp.empty(rows * block_size, dtype=matrix.data.dtype),
            )

    def before(name, handle):
        """Restore the unscaled sample outside the timed launch."""
        trial["data"][...] = trial["original"]
        trial["rhs"][...] = trial["original_rhs"]

    def run(name, handle):
        """Scale the sample with one launch shape."""
        _launch_bsr_row_scale(handle, trial["indptr"], trial["indices"], trial["data"], trial["rhs"],
                              trial["diagonal"], trial["rows"], block_size)
        return trial["data"], trial["rhs"], trial["diagonal"]

    def prepare(name):
        """Compile one launch shape."""
        _bsr_row_scale_kernels(shapes[name])
        return shapes[name]

    name, _, _ = select_variant(
        cupy=cp,
        identity=dict(operation="bsr-row-scale", order=block_size - 1, block_size=block_size,
                      dtype=str(matrix.data.dtype), layout="bsr-row-major",
                      block_rows_bucket=key[3], sample_rows=_BSR_SCALE_SAMPLE_ROWS,
                      sample_bytes=_BSR_SCALE_SAMPLE_BYTES,
                      source=fingerprint([module_source_fingerprint(__file__)])),
        candidates=list(shapes), prepare=prepare, trial_prepare=trial_prepare, before=before, run=run,
        rtol=0.0,
    )
    trial.clear()
    choice = shapes[name]
    _BSR_ROW_SCALE_CHOICES[key] = choice
    return choice


def _check_bsr_scale_block_size(matrix: _DeviceBsrMatrixView) -> None:
    """Reject block sizes the kernels do not cover."""
    block_size = int(matrix.block_size)
    if not 1 <= block_size <= _BSR_SCALE_MAX_BLOCK:
        raise ValueError(
            f"BSR row scaling supports block sizes 1..{_BSR_SCALE_MAX_BLOCK}, got {block_size}"
        )


def _diagonal_scale_bsr_rows_in_place(matrix: _DeviceBsrMatrixView, rhs):
    """Apply the scalar CSR left-scaling rule directly to face-BSR values."""
    cp = require_cupy()
    nrows = int(rhs.size)
    diagonal = cp.empty(nrows, dtype=REAL_DTYPE)
    if nrows:
        _check_bsr_scale_block_size(matrix)
        _launch_bsr_row_scale(_bsr_row_scale_choice(matrix), matrix.indptr, matrix.indices, matrix.data,
                              rhs, diagonal, nrows // matrix.block_size, matrix.block_size)
    return diagonal


def _restore_left_scaled_bsr_rows_in_place(
    matrix: _DeviceBsrMatrixView,
    row_diagonal,
) -> None:
    """Restore face-BSR values after scalar-row left scaling, with the scaling's launch shape."""
    nrows = int(matrix.shape[0])
    if not nrows:
        return
    _check_bsr_scale_block_size(matrix)
    shape = _bsr_row_scale_choice(matrix)
    num_block_rows = nrows // matrix.block_size
    grid, block = _bsr_scale_launch(num_block_rows, shape)
    _, restore_kernel = _bsr_row_scale_kernels(shape)
    restore_kernel(grid, block, (matrix.indptr, matrix.data, row_diagonal,
                                 np.int32(num_block_rows), np.int32(matrix.block_size)))
