"""Experimental Legendre face-BSR operators for HDG Poisson.

CUDA 13 generic-BSR ``cusparseSpMV`` is preferred.  CuPy 14.1 does not expose
``cusparseCreateBsr``, so this module owns a narrow binding around CuPy's
cuSPARSE handle.  A row-owned RawKernel is the explicit fallback and benchmark
control.  This is an internal prototype, not a public backend contract.
"""

from __future__ import annotations

from hybridge.runtime.precision import REAL_DTYPE, REAL_ITEMSIZE, real_raw_kernel

import ctypes
from dataclasses import dataclass
from typing import Any

import numpy as np

from hybridge.runtime.optional import require_cupy


_CUSPARSE_SUCCESS = 0
_CUSPARSE_OPERATION_NON_TRANSPOSE = 0
_CUSPARSE_INDEX_32I = 2
_CUSPARSE_INDEX_BASE_ZERO = 0
_CUSPARSE_ORDER_ROW = 2
_CUSPARSE_SPMV_ALG_DEFAULT = 0
_CUDA_REAL = 0 if REAL_ITEMSIZE == 4 else 1
_GENERIC_BSR_MIN_RUNTIME = 13010


class CusparseBsrUnavailable(RuntimeError):
    """Raised when generic cuSPARSE BSR SpMV cannot be used safely."""


def _array_module(array):
    """Return NumPy or CuPy without importing CuPy on host-only paths."""
    try:
        cp = require_cupy()
    except Exception:
        return np
    return cp.get_array_module(array)


def legendre_orthonormal_scales(block_size: int, *, xp=np):
    """Return ``S_jj=sqrt((2*j+1)/2)`` for the normalized modal basis."""
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError("block_size must be positive")
    modes = xp.arange(block_size, dtype=REAL_DTYPE)
    return xp.sqrt((2.0 * modes + 1.0) / 2.0)


def transform_legendre_bsr_to_orthonormal(data, rhs):
    """Return ``S A S``, ``S b``, and ``diag(S)`` for face-major BSR."""
    if data.ndim != 3 or data.shape[1] != data.shape[2]:
        raise ValueError("data must have shape (nnzb, block_size, block_size)")
    block_size = int(data.shape[1])
    if rhs.ndim != 1 or rhs.size % block_size:
        raise ValueError("rhs length must be divisible by the BSR block size")
    xp = _array_module(data)
    scales = legendre_orthonormal_scales(block_size, xp=xp)
    matrix = xp.ascontiguousarray(
        data * scales[None, :, None] * scales[None, None, :]
    )
    vector = xp.ascontiguousarray(
        (rhs.reshape((-1, block_size)) * scales[None, :]).reshape(-1)
    )
    return matrix, vector, scales


def modal_degree_schedule(degree: int, mode: str = "halve") -> tuple[int, ...]:
    """Return a strictly decreasing p schedule terminating at zero."""
    degree = int(degree)
    if degree < 1:
        raise ValueError("degree must be at least one")
    normalized = str(mode).replace("_", "-").lower()
    if normalized in {"direct", "direct-to-zero", "p-to-zero"}:
        return degree, 0
    if normalized not in {"halve", "halving"}:
        raise ValueError("schedule mode must be 'halve' or 'direct-to-zero'")
    result = [degree]
    while result[-1] > 1:
        next_degree = max(1, result[-1] // 2)
        if next_degree != result[-1]:
            result.append(next_degree)
    result.append(0)
    return tuple(result)


def principal_modal_bsr_data(data, coarse_block_size: int):
    """Extract nested-modal Galerkin principal blocks from all BSR entries."""
    if data.ndim != 3 or data.shape[1] != data.shape[2]:
        raise ValueError("data must have shape (nnzb, block_size, block_size)")
    coarse_block_size = int(coarse_block_size)
    if not 1 <= coarse_block_size <= int(data.shape[1]):
        raise ValueError("coarse_block_size must be in [1, fine_block_size]")
    return _array_module(data).ascontiguousarray(
        data[:, :coarse_block_size, :coarse_block_size]
    )


def restrict_modal(vector, *, num_faces: int, fine_block_size: int, coarse_block_size: int):
    """Restrict by retaining the lowest orthonormal Legendre modes."""
    num_faces = int(num_faces)
    fine_block_size = int(fine_block_size)
    coarse_block_size = int(coarse_block_size)
    if not 1 <= coarse_block_size <= fine_block_size:
        raise ValueError("invalid modal restriction block sizes")
    if vector.ndim != 1 or vector.size != num_faces * fine_block_size:
        raise ValueError("fine vector shape is incompatible with the transfer")
    xp = _array_module(vector)
    return xp.ascontiguousarray(
        vector.reshape((num_faces, fine_block_size))[:, :coarse_block_size]
    ).reshape(-1)


def prolong_modal(vector, *, num_faces: int, fine_block_size: int, coarse_block_size: int):
    """Prolong by low-mode injection and zero high modes."""
    num_faces = int(num_faces)
    fine_block_size = int(fine_block_size)
    coarse_block_size = int(coarse_block_size)
    if not 1 <= coarse_block_size <= fine_block_size:
        raise ValueError("invalid modal prolongation block sizes")
    if vector.ndim != 1 or vector.size != num_faces * coarse_block_size:
        raise ValueError("coarse vector shape is incompatible with the transfer")
    xp = _array_module(vector)
    output = xp.zeros((num_faces, fine_block_size), dtype=vector.dtype)
    output[:, :coarse_block_size] = vector.reshape((num_faces, coarse_block_size))
    return output.reshape(-1)


def diagonal_block_positions(indptr, indices, local_positions=None):
    """Return compressed positions of diagonal face blocks."""
    num_rows = int(indptr.size - 1)
    if local_positions is not None:
        if local_positions.ndim != 1 or int(local_positions.size) != num_rows:
            raise ValueError("local diagonal positions must have one entry per row")
        positions = indptr[:-1] + local_positions
        xp = _array_module(positions)
        valid = xp.all(indices[positions] == xp.arange(num_rows, dtype=indices.dtype))
        if hasattr(valid, "get"):
            valid = valid.get()
        if not bool(valid):
            raise ValueError("direct BSR diagonal slots do not point to diagonal blocks")
        return positions

    indptr_host = np.asarray(indptr)
    indices_host = np.asarray(indices)
    positions = np.empty(num_rows, dtype=np.int32)
    for row in range(num_rows):
        start, stop = int(indptr_host[row]), int(indptr_host[row + 1])
        matches = np.flatnonzero(indices_host[start:stop] == row)
        if matches.size != 1:
            raise ValueError(f"block row {row} must contain one diagonal block")
        positions[row] = start + int(matches[0])
    return positions


def _configure_cusparse(lib) -> None:
    """Declare the narrow generic-BSR ctypes interface."""
    ptr = ctypes.c_void_p
    cint = ctypes.c_int
    i64 = ctypes.c_int64
    lib.cusparseCreateBsr.argtypes = [
        ctypes.POINTER(ptr), i64, i64, i64, i64, i64,
        ptr, ptr, ptr, cint, cint, cint, cint, cint,
    ]
    lib.cusparseCreateBsr.restype = cint
    lib.cusparseCreateCsr.argtypes = [
        ctypes.POINTER(ptr), i64, i64, i64,
        ptr, ptr, ptr, cint, cint, cint, cint,
    ]
    lib.cusparseCreateCsr.restype = cint
    lib.cusparseDestroySpMat.argtypes = [ptr]
    lib.cusparseDestroySpMat.restype = cint
    lib.cusparseCreateDnVec.argtypes = [ctypes.POINTER(ptr), i64, ptr, cint]
    lib.cusparseCreateDnVec.restype = cint
    lib.cusparseDestroyDnVec.argtypes = [ptr]
    lib.cusparseDestroyDnVec.restype = cint
    lib.cusparseDnVecSetValues.argtypes = [ptr, ptr]
    lib.cusparseDnVecSetValues.restype = cint
    common = [ptr, cint, ptr, ptr, ptr, ptr, ptr, cint, cint]
    lib.cusparseSpMV_bufferSize.argtypes = common + [ctypes.POINTER(ctypes.c_size_t)]
    lib.cusparseSpMV_bufferSize.restype = cint
    lib.cusparseSpMV_preprocess.argtypes = common + [ptr]
    lib.cusparseSpMV_preprocess.restype = cint
    lib.cusparseSpMV.argtypes = common + [ptr]
    lib.cusparseSpMV.restype = cint
    lib.cusparseSetStream.argtypes = [ptr, ptr]
    lib.cusparseSetStream.restype = cint
    if hasattr(lib, "cusparseGetErrorString"):
        lib.cusparseGetErrorString.argtypes = [cint]
        lib.cusparseGetErrorString.restype = ctypes.c_char_p


def _load_cusparse():
    """Load the cuSPARSE ABI already loaded by active CuPy."""
    errors = []
    for candidate in ("libcusparse.so.12", "libcusparse.so"):
        try:
            lib = ctypes.CDLL(candidate)
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        if not hasattr(lib, "cusparseCreateBsr"):
            errors.append(f"{candidate}: cusparseCreateBsr is absent")
            continue
        _configure_cusparse(lib)
        return lib
    raise CusparseBsrUnavailable("; ".join(errors) or "cuSPARSE is unavailable")


class _CusparseGenericBsrOperator:
    """Matrix-owned CUDA 13 BSR descriptors, preprocessing, and workspace."""

    def __init__(self, indptr, indices, data, *, shape=None):
        """Create and preprocess descriptors for one fixed BSR matrix."""
        self.cp = require_cupy()
        runtime = int(self.cp.cuda.runtime.runtimeGetVersion())
        if runtime < _GENERIC_BSR_MIN_RUNTIME:
            raise CusparseBsrUnavailable(
                f"generic BSR SpMV needs CUDA runtime >=13010; found {runtime}"
            )
        self.lib = _load_cusparse()
        self.indptr, self.indices, self.data = indptr, indices, data
        self.block_rows = int(indptr.size - 1)
        is_csr = data.ndim == 1
        if not is_csr and (data.ndim != 3 or data.shape[1] != data.shape[2]):
            raise ValueError("BSR data must contain square blocks")
        self.block_size = 1 if is_csr else int(data.shape[1])
        self.size = self.block_rows * self.block_size
        self.shape = (self.size, self.size) if shape is None else tuple(map(int, shape))
        if (len(self.shape) != 2 or self.shape[0] != self.size
                or self.shape[1] < 0 or self.shape[1] % self.block_size):
            raise ValueError("shape must match block rows and have a block-aligned column count")
        if (data.dtype != REAL_DTYPE or indices.dtype != np.int32
                or indptr.dtype != np.int32
                or not all(a.flags.c_contiguous for a in (data, indices, indptr))):
            raise ValueError("Require contiguous real values and int32 indices/row pointers")
        if data.shape[0] != indices.size:
            raise ValueError("data and indices counts differ")
        self.block_cols = self.shape[1] // self.block_size
        self.input_size = self.shape[1]
        self.handle = ctypes.c_void_p(int(self.cp.cuda.device.get_cusparse_handle()))
        self.stream_ptr: int | None = None
        self.matrix = ctypes.c_void_p()
        self.x_descriptor = ctypes.c_void_p()
        self.y_descriptor = ctypes.c_void_p()
        self.workspace = None
        self.workspace_size = 0
        self.closed = False
        self.alpha = (ctypes.c_float if REAL_ITEMSIZE == 4 else ctypes.c_double)(1.0)
        self.beta = (ctypes.c_float if REAL_ITEMSIZE == 4 else ctypes.c_double)(0.0)
        self.scratch_x = self.cp.empty(self.input_size, dtype=REAL_DTYPE)
        self.scratch_y = self.cp.empty(self.size, dtype=REAL_DTYPE)
        try:
            create = self.lib.cusparseCreateCsr if is_csr else self.lib.cusparseCreateBsr
            dimensions = [self.block_rows, self.block_cols, int(indices.size)]
            if not is_csr:
                dimensions += [self.block_size, self.block_size]
            types = [_CUSPARSE_INDEX_32I, _CUSPARSE_INDEX_32I,
                     _CUSPARSE_INDEX_BASE_ZERO, _CUDA_REAL]
            if not is_csr:
                types += [_CUSPARSE_ORDER_ROW]
            self._check(create(
                ctypes.byref(self.matrix), *dimensions,
                ctypes.c_void_p(int(indptr.data.ptr)),
                ctypes.c_void_p(int(indices.data.ptr)),
                ctypes.c_void_p(int(data.data.ptr)),
                *types,
            ), "cusparseCreateCsr" if is_csr else "cusparseCreateBsr")
            self._check(self.lib.cusparseCreateDnVec(
                ctypes.byref(self.x_descriptor), self.input_size,
                ctypes.c_void_p(int(self.scratch_x.data.ptr)), _CUDA_REAL,
            ), "cusparseCreateDnVec(x)")
            self._check(self.lib.cusparseCreateDnVec(
                ctypes.byref(self.y_descriptor), self.size,
                ctypes.c_void_p(int(self.scratch_y.data.ptr)), _CUDA_REAL,
            ), "cusparseCreateDnVec(y)")
            self._set_stream()
            workspace_size = ctypes.c_size_t()
            self._check(self.lib.cusparseSpMV_bufferSize(
                self.handle, _CUSPARSE_OPERATION_NON_TRANSPOSE, self._alpha_ptr,
                self.matrix, self.x_descriptor, self._beta_ptr,
                self.y_descriptor, _CUDA_REAL, _CUSPARSE_SPMV_ALG_DEFAULT,
                ctypes.byref(workspace_size),
            ), "cusparseSpMV_bufferSize(BSR)")
            self.workspace_size = int(workspace_size.value)
            if self.workspace_size:
                self.workspace = self.cp.empty(self.workspace_size, dtype=self.cp.uint8)
            self._check(self.lib.cusparseSpMV_preprocess(
                self.handle, _CUSPARSE_OPERATION_NON_TRANSPOSE, self._alpha_ptr,
                self.matrix, self.x_descriptor, self._beta_ptr,
                self.y_descriptor, _CUDA_REAL, _CUSPARSE_SPMV_ALG_DEFAULT,
                self._workspace_ptr,
            ), "cusparseSpMV_preprocess(BSR)")
        except Exception as exc:
            self.close(suppress_errors=True)
            if isinstance(exc, CusparseBsrUnavailable):
                raise
            raise CusparseBsrUnavailable(str(exc)) from exc

    @property
    def _alpha_ptr(self):
        """Return the host scalar-one pointer required by cuSPARSE."""
        return ctypes.cast(ctypes.byref(self.alpha), ctypes.c_void_p)

    @property
    def _beta_ptr(self):
        """Return the host scalar-zero pointer required by cuSPARSE."""
        return ctypes.cast(ctypes.byref(self.beta), ctypes.c_void_p)

    @property
    def _workspace_ptr(self):
        """Return the cached external-workspace device pointer."""
        return ctypes.c_void_p(0 if self.workspace is None else int(self.workspace.data.ptr))

    def _message(self, status: int) -> str:
        """Return a readable cuSPARSE status message."""
        if hasattr(self.lib, "cusparseGetErrorString"):
            value = self.lib.cusparseGetErrorString(int(status))
            if value:
                return value.decode("utf-8", errors="replace")
        return f"status {status}"

    def _check(self, status: int, operation: str) -> None:
        """Raise an availability error for a failed cuSPARSE call."""
        if int(status) != _CUSPARSE_SUCCESS:
            raise CusparseBsrUnavailable(f"{operation} failed: {self._message(status)}")

    def _set_stream(self) -> None:
        """Bind and enforce the stream used to preprocess this descriptor."""
        stream = self.cp.cuda.get_current_stream()
        current = int(stream.ptr)
        if self.stream_ptr is None:
            self.stream_ptr = current
        elif current != self.stream_ptr:
            raise RuntimeError(
                "generic-BSR preprocessing is tied to its setup stream; "
                "construct a stream-local operator instead of reusing it"
            )
        self._check(
            self.lib.cusparseSetStream(self.handle, ctypes.c_void_p(current)),
            "cusparseSetStream",
        )

    def matvec(self, x, out=None):
        """Apply the cached BSR descriptor asynchronously."""
        if self.closed:
            raise RuntimeError("cannot apply a closed cuSPARSE BSR operator")
        x = self.cp.asarray(x, dtype=REAL_DTYPE)
        if x.ndim != 1 or int(x.size) != self.input_size:
            raise ValueError(f"x must have shape ({self.input_size},)")
        if not x.flags.c_contiguous:
            x = self.cp.ascontiguousarray(x)
        if out is None:
            out = self.cp.empty(self.size, dtype=REAL_DTYPE)
        elif (out.ndim != 1 or int(out.size) != self.size
              or out.dtype != REAL_DTYPE or not out.flags.c_contiguous):
            raise ValueError(f"out must be contiguous at the selected real precision with shape ({self.size},)")
        if int(x.data.ptr) == int(out.data.ptr):
            raise ValueError("x and out must not alias")
        self._set_stream()
        self._check(self.lib.cusparseDnVecSetValues(
            self.x_descriptor, ctypes.c_void_p(int(x.data.ptr))),
            "cusparseDnVecSetValues(x)")
        self._check(self.lib.cusparseDnVecSetValues(
            self.y_descriptor, ctypes.c_void_p(int(out.data.ptr))),
            "cusparseDnVecSetValues(y)")
        self._check(self.lib.cusparseSpMV(
            self.handle, _CUSPARSE_OPERATION_NON_TRANSPOSE, self._alpha_ptr,
            self.matrix, self.x_descriptor, self._beta_ptr, self.y_descriptor,
            _CUDA_REAL, _CUSPARSE_SPMV_ALG_DEFAULT, self._workspace_ptr,
        ), "cusparseSpMV(BSR)")
        return out

    def close(self, *, suppress_errors: bool = False) -> None:
        """Destroy owned descriptors but not CuPy's shared handle."""
        if self.closed:
            return
        first_error = None
        for descriptor, destroy in (
            (self.y_descriptor, "cusparseDestroyDnVec"),
            (self.x_descriptor, "cusparseDestroyDnVec"),
            (self.matrix, "cusparseDestroySpMat"),
        ):
            if descriptor and descriptor.value:
                try:
                    status = getattr(self.lib, destroy)(descriptor)
                    if int(status) != _CUSPARSE_SUCCESS:
                        raise RuntimeError(f"{destroy} returned status {status}")
                except Exception as exc:
                    first_error = first_error or exc
        self.matrix = self.x_descriptor = self.y_descriptor = ctypes.c_void_p()
        self.workspace = self.scratch_x = self.scratch_y = None
        self.closed = True
        if first_error is not None and not suppress_errors:
            raise first_error

    def __del__(self):
        """Perform best-effort cleanup during interpreter shutdown."""
        try:
            self.close(suppress_errors=True)
        except Exception:
            pass


class _CusparseGenericCsrOperator(_CusparseGenericBsrOperator):
    """Generic CSR SpMV sharing BSR's descriptors and preallocation contract."""

    def __init__(self, indptr, indices, data, *, shape):
        """Require one-dimensional CSR data, then set up the shared generic-SpMV descriptors."""
        if data.ndim != 1:
            raise ValueError("CSR data must be one-dimensional")
        super().__init__(indptr, indices, data, shape=shape)


_FUSED_FACE_BLOCK_JACOBI_STEP = r"""
extern "C" __global__
void legendre_face_bsr_block_jacobi_step(
        const int num_block_rows,
        const int* __restrict__ indptr,
        const int* __restrict__ indices,
        const double* __restrict__ data,
        const double* __restrict__ diagonal_inverse,
        const double* __restrict__ rhs,
        const double* __restrict__ x,
        const double weight,
        double* __restrict__ output)
{
    const int row = (int)blockIdx.x;
    const int lane = (int)threadIdx.x;
    if (row >= num_block_rows) return;

    double ax = 0.0;
    for (int pos = indptr[row]; pos < indptr[row + 1]; ++pos) {
        const int col = indices[pos];
        const long long block_base =
            (long long)pos * BLOCK_SIZE * BLOCK_SIZE;
        const long long x_base = (long long)col * BLOCK_SIZE;
#pragma unroll
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            const double owned_x = lane == j ? x[x_base + j] : 0.0;
            const double x_j = __shfl_sync(0xffffffffu, owned_x, j);
            if (lane < BLOCK_SIZE) {
                ax += data[block_base + lane * BLOCK_SIZE + j] * x_j;
            }
        }
    }

    const long long row_base = (long long)row * BLOCK_SIZE;
    const double residual = lane < BLOCK_SIZE ? rhs[row_base + lane] - ax : 0.0;
    double update = 0.0;
    const long long diagonal_base =
        (long long)row * BLOCK_SIZE * BLOCK_SIZE;
#pragma unroll
    for (int j = 0; j < BLOCK_SIZE; ++j) {
        const double residual_j = __shfl_sync(0xffffffffu, residual, j);
        if (lane < BLOCK_SIZE) {
            update += diagonal_inverse[
                diagonal_base + lane * BLOCK_SIZE + j
            ] * residual_j;
        }
    }
    if (lane < BLOCK_SIZE) {
        output[row_base + lane] = x[row_base + lane] + weight * update;
    }
}
"""
_FUSED_BLOCK_JACOBI_KERNEL_CACHE: dict[int, Any] = {}


_FUSED_FACE_BLOCK_JACOBI_ZERO_START = r"""
extern "C" __global__
void legendre_face_block_jacobi_zero_start(
        const int num_block_rows,
        const double* __restrict__ diagonal_inverse,
        const double* __restrict__ rhs,
        const double weight,
        double* __restrict__ output)
{
    const int row = (int)blockIdx.x;
    const int lane = (int)threadIdx.x;
    if (row >= num_block_rows) return;

    const long long row_base = (long long)row * BLOCK_SIZE;
    const double rhs_i = lane < BLOCK_SIZE ? rhs[row_base + lane] : 0.0;
    double update = 0.0;
    const long long diagonal_base =
        (long long)row * BLOCK_SIZE * BLOCK_SIZE;
#pragma unroll
    for (int j = 0; j < BLOCK_SIZE; ++j) {
        const double rhs_j = __shfl_sync(0xffffffffu, rhs_i, j);
        if (lane < BLOCK_SIZE) {
            update += diagonal_inverse[
                diagonal_base + lane * BLOCK_SIZE + j
            ] * rhs_j;
        }
    }
    if (lane < BLOCK_SIZE) {
        output[row_base + lane] = weight * update;
    }
}
"""
_FUSED_BLOCK_JACOBI_ZERO_START_KERNEL_CACHE: dict[int, Any] = {}


_PACKED_FACE_BSR_KERNELS = r"""
// Packed face-BSR kernels: one thread per scalar row, GROUP_ROWS lanes per
// block row, GROUPS = 32 / GROUP_ROWS block rows per warp; the remaining
// 32 % GROUP_ROWS lanes idle.  The warp-per-block-row kernels above keep only
// BLOCK_SIZE of 32 lanes busy and their one-warp blocks cap sm_80+ occupancy
// at 32 warps per SM.  Matrix blocks are read transposed, data_t[pos][j][i] =
// A_pos[i][j], so each group loads contiguous values.  Every row sum runs in
// the warp-per-row order (block position outer, column inner), so the step
// and zero-start kernels are bitwise equal to theirs and the SpMV and
// restricted residual to the raw row-owned SpMV. The multiply-adds are explicit fma() calls:
// the compiler contracts the reference kernels' `a += b * c` to fma, but may split an
// implicit one here (it did at BLOCK_SIZE 1), which breaks the bitwise match.
#define GROUPS (32 / GROUP_ROWS)

__device__ __forceinline__ int packed_group_row(
        const int num_block_rows, int* lane_in_group, int* group_base)
{
    const int lane = (int)(threadIdx.x & 31u);
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int group = lane / GROUP_ROWS;
    *lane_in_group = lane - group * GROUP_ROWS;
    *group_base = group * GROUP_ROWS;
    const int row = warp * GROUPS + group;
    return (group < GROUPS && row < num_block_rows) ? row : -1;
}

// Row i of block row `row` times x; with coarse_correction, x is read as
// x + P e, where P injects the coarse_block lowest modes of e.
__device__ __forceinline__ double packed_row_product(
        const int row, const int i,
        const int* __restrict__ indptr,
        const int* __restrict__ indices,
        const double* __restrict__ data_t,
        const double* __restrict__ x,
        const double* __restrict__ coarse_correction,
        const int coarse_block)
{
    double ax = 0.0;
    const int stop = __ldg(indptr + row + 1);
    for (int pos = __ldg(indptr + row); pos < stop; ++pos) {
        const int col = __ldg(indices + pos);
        const double* __restrict__ block =
            data_t + (long long)pos * BLOCK_SIZE * BLOCK_SIZE + i;
        const double* __restrict__ x_block = x + (long long)col * BLOCK_SIZE;
        if (coarse_correction == nullptr) {
#pragma unroll
            for (int j = 0; j < BLOCK_SIZE; ++j) {
                ax = fma(__ldg(block + j * BLOCK_SIZE), __ldg(x_block + j), ax);
            }
        } else {
            const double* __restrict__ e_block =
                coarse_correction + (long long)col * coarse_block;
#pragma unroll
            for (int j = 0; j < BLOCK_SIZE; ++j) {
                double x_j = __ldg(x_block + j);
                if (j < coarse_block) x_j += __ldg(e_block + j);
                ax = fma(__ldg(block + j * BLOCK_SIZE), x_j, ax);
            }
        }
    }
    return ax;
}

extern "C" __global__
void packed_face_bsr_spmv(
        const int num_block_rows,
        const int* __restrict__ indptr,
        const int* __restrict__ indices,
        const double* __restrict__ data_t,
        const double* __restrict__ x,
        double* __restrict__ y)
{
    int i, base;
    const int row = packed_group_row(num_block_rows, &i, &base);
    if (row < 0) return;
    y[(long long)row * BLOCK_SIZE + i] = packed_row_product(
        row, i, indptr, indices, data_t, x, nullptr, 0);
}

// coarse_rhs[row, i] = (rhs - A x)[row, i] for the COARSE_BLOCK lowest modes:
// residual, subtraction and modal restriction in one pass that reads only the
// retained rows of each block.  Compiled with GROUP_ROWS = COARSE_BLOCK.
extern "C" __global__
void packed_face_bsr_restricted_residual(
        const int num_block_rows,
        const int* __restrict__ indptr,
        const int* __restrict__ indices,
        const double* __restrict__ data_t,
        const double* __restrict__ rhs,
        const double* __restrict__ x,
        double* __restrict__ coarse_rhs)
{
    int i, base;
    const int row = packed_group_row(num_block_rows, &i, &base);
    if (row < 0) return;
    const double ax = packed_row_product(
        row, i, indptr, indices, data_t, x, nullptr, 0);
    coarse_rhs[(long long)row * COARSE_BLOCK + i] =
        rhs[(long long)row * BLOCK_SIZE + i] - ax;
}

// output = x + weight * D^{-1} (rhs - A x), with x optionally read as x + P e
// (fusing the coarse-correction prolongation into the first post-smoothing
// stage).  diagonal_inverse_t holds the transposed inverse diagonal blocks.
extern "C" __global__
void packed_face_bsr_block_jacobi_step(
        const int num_block_rows,
        const int* __restrict__ indptr,
        const int* __restrict__ indices,
        const double* __restrict__ data_t,
        const double* __restrict__ diagonal_inverse_t,
        const double* __restrict__ rhs,
        const double* __restrict__ x,
        const double* __restrict__ coarse_correction,
        const int coarse_block,
        const double weight,
        double* __restrict__ output)
{
    int i, base;
    const int row = packed_group_row(num_block_rows, &i, &base);
    double residual = 0.0;
    double x_i = 0.0;
    if (row >= 0) {
        const long long row_index = (long long)row * BLOCK_SIZE + i;
        residual = rhs[row_index] - packed_row_product(
            row, i, indptr, indices, data_t, x, coarse_correction, coarse_block);
        x_i = x[row_index];
        if (coarse_correction != nullptr && i < coarse_block) {
            x_i += coarse_correction[(long long)row * coarse_block + i];
        }
    }
    // All lanes, idle ones included, take part in the group exchange.
    double update = 0.0;
    const double* __restrict__ inverse =
        diagonal_inverse_t + (long long)(row < 0 ? 0 : row) * BLOCK_SIZE * BLOCK_SIZE + i;
#pragma unroll
    for (int j = 0; j < BLOCK_SIZE; ++j) {
        const double residual_j = __shfl_sync(0xffffffffu, residual, (base + j) & 31);
        if (row >= 0) update = fma(__ldg(inverse + j * BLOCK_SIZE), residual_j, update);
    }
    if (row >= 0) {
        output[(long long)row * BLOCK_SIZE + i] = fma(weight, update, x_i);
    }
}

extern "C" __global__
void packed_face_block_jacobi_zero_start(
        const int num_block_rows,
        const double* __restrict__ diagonal_inverse_t,
        const double* __restrict__ rhs,
        const double weight,
        double* __restrict__ output)
{
    int i, base;
    const int row = packed_group_row(num_block_rows, &i, &base);
    if (row < 0) return;
    const double* __restrict__ inverse =
        diagonal_inverse_t + (long long)row * BLOCK_SIZE * BLOCK_SIZE + i;
    const double* __restrict__ rhs_block = rhs + (long long)row * BLOCK_SIZE;
    double update = 0.0;
#pragma unroll
    for (int j = 0; j < BLOCK_SIZE; ++j) {
        update = fma(__ldg(inverse + j * BLOCK_SIZE), __ldg(rhs_block + j), update);
    }
    output[(long long)row * BLOCK_SIZE + i] = weight * update;
}
"""
_PACKED_MODULE_CACHE: dict[tuple[int, int], Any] = {}
PACKED_THREADS_PER_BLOCK = 256


def _packed_module(block_size: int, group_rows: int):
    """Compile (once) the packed kernels for one block and group size."""
    key = (int(block_size), int(group_rows))
    module = _PACKED_MODULE_CACHE.get(key)
    if module is None:
        from hybridge.runtime.precision import real_raw_module

        source = (
            f"#define BLOCK_SIZE {key[0]}\n#define GROUP_ROWS {key[1]}\n"
            f"#define COARSE_BLOCK {key[1]}\n" + _PACKED_FACE_BSR_KERNELS
        )
        module = real_raw_module(code=source, options=("--std=c++17",))
        _PACKED_MODULE_CACHE[key] = module
    return module


def packed_grid(num_block_rows: int, group_rows: int,
                threads: int = PACKED_THREADS_PER_BLOCK) -> int:
    """Return the packed-kernel grid size for one launch."""
    rows_per_block = (int(threads) // 32) * (32 // int(group_rows))
    return max(1, -(-int(num_block_rows) // rows_per_block))


class PackedFaceBsrKernels:
    """Prebound packed kernels for one fixed face-BSR level.

    Holds the block-transposed matrix and inverse diagonal so the smoother loop
    launches kernels without per-call validation. Pointers are fixed after
    construction, which also makes the launches capturable in a CUDA graph.
    """

    def __init__(self, operator: "LegendreFaceBsrOperator", diagonal_inverse=None,
                 *, coarse_block_size: int | None = None,
                 threads: int = PACKED_THREADS_PER_BLOCK):
        """Transpose the level blocks and compile the level's kernels."""
        cp = require_cupy()
        self.cp = cp
        self.block_rows = int(operator.block_rows)
        self.block_size = int(operator.block_size)
        if not 1 <= self.block_size <= 32:
            raise NotImplementedError("packed face-BSR kernels support block sizes 1..32")
        expected = (self.block_rows, self.block_size, self.block_size)
        if diagonal_inverse is not None and tuple(diagonal_inverse.shape) != expected:
            raise ValueError(f"diagonal_inverse must have shape {expected}")
        if int(threads) % 32 or not 32 <= int(threads) <= 1024:
            raise ValueError("packed threads must be a multiple of 32 in [32, 1024]")
        self.indptr, self.indices = operator.indptr, operator.indices
        existing = getattr(getattr(operator, "_implementation", None), "packed", None)
        self.owns_data_t = existing is None
        self.data_t = (cp.ascontiguousarray(operator.data.transpose(0, 2, 1))
                       if existing is None else existing.data_t)
        self.diagonal_inverse_t = None if diagonal_inverse is None else cp.ascontiguousarray(
            cp.asarray(diagonal_inverse, dtype=REAL_DTYPE).transpose(0, 2, 1)
        )
        module = _packed_module(self.block_size, self.block_size)
        self._spmv = module.get_function("packed_face_bsr_spmv")
        self._step = module.get_function("packed_face_bsr_block_jacobi_step")
        self._zero = module.get_function("packed_face_block_jacobi_zero_start")
        self._grid = (packed_grid(self.block_rows, self.block_size, threads),)
        self._block = (int(threads),)
        self._rows = np.int32(self.block_rows)
        self._null = np.uint64(0)
        self.coarse_block_size = None
        if coarse_block_size is not None:
            self.coarse_block_size = int(coarse_block_size)
            if not 1 <= self.coarse_block_size <= self.block_size:
                raise ValueError("coarse_block_size must be in [1, block_size]")
            coarse = _packed_module(self.block_size, self.coarse_block_size)
            self._restrict = coarse.get_function("packed_face_bsr_restricted_residual")
            self._restrict_grid = (packed_grid(self.block_rows, self.coarse_block_size, threads),)

    @property
    def nbytes(self) -> int:
        """Return device bytes of the transposed copies."""
        return int((self.data_t.nbytes if self.owns_data_t else 0)
                   + (0 if self.diagonal_inverse_t is None else self.diagonal_inverse_t.nbytes))

    def spmv(self, x, out):
        """``out = A x``."""
        self._spmv(self._grid, self._block, (
            self._rows, self.indptr, self.indices, self.data_t, x, out))
        return out

    def step(self, rhs, x, weight, out, coarse_correction=None):
        """``out = x' + weight D^{-1}(rhs - A x')`` with ``x' = x + P e``."""
        if coarse_correction is None:
            correction, coarse_block = self._null, np.int32(0)
        else:
            correction, coarse_block = coarse_correction, np.int32(self.coarse_block_size)
        self._step(self._grid, self._block, (
            self._rows, self.indptr, self.indices, self.data_t,
            self.diagonal_inverse_t, rhs, x, correction, coarse_block,
            REAL_DTYPE(weight), out))
        return out

    def zero_start(self, rhs, weight, out):
        """``out = weight D^{-1} rhs``."""
        self._zero(self._grid, self._block, (
            self._rows, self.diagonal_inverse_t, rhs, REAL_DTYPE(weight), out))
        return out

    def restricted_residual(self, rhs, x, out):
        """``out = R (rhs - A x)``, keeping the coarse level's modes."""
        self._restrict(self._restrict_grid, self._block, (
            self._rows, self.indptr, self.indices, self.data_t, rhs, x, out))
        return out


_RAW_FACE_BSR_SPMV = r"""
extern "C" __global__
void legendre_face_bsr_spmv(
        const int num_block_rows,
        const int* __restrict__ indptr,
        const int* __restrict__ indices,
        const double* __restrict__ data,
        const double* __restrict__ x,
        double* __restrict__ y)
{
    const int row = (int)blockIdx.x;
    if (row >= num_block_rows) return;
    for (int i = (int)threadIdx.x; i < BLOCK_SIZE; i += (int)blockDim.x) {
        double sum = 0.0;
        for (int pos = indptr[row]; pos < indptr[row + 1]; ++pos) {
            const int col = indices[pos];
            const long long base = (long long)pos * BLOCK_SIZE * BLOCK_SIZE;
            const long long xbase = (long long)col * BLOCK_SIZE;
#pragma unroll
            for (int j = 0; j < BLOCK_SIZE; ++j) {
                sum += data[base + i * BLOCK_SIZE + j] * x[xbase + j];
            }
        }
        y[(long long)row * BLOCK_SIZE + i] = sum;
    }
}
"""
_RAW_KERNEL_CACHE: dict[int, Any] = {}


class _RawFaceBsrOperator:
    """Row-owned fixed-block RawKernel fallback for block sizes 1..10."""

    def __init__(self, indptr, indices, data):
        """Compile or reuse the block-size-specialized fallback kernel."""
        self.cp = require_cupy()
        self.indptr, self.indices, self.data = indptr, indices, data
        self.block_rows = int(indptr.size - 1)
        self.block_size = int(data.shape[1])
        self.size = self.block_rows * self.block_size
        if not 1 <= self.block_size <= 10:
            raise NotImplementedError("raw face-BSR SpMV supports block sizes 1..10")
        kernel = _RAW_KERNEL_CACHE.get(self.block_size)
        if kernel is None:
            source = f"#define BLOCK_SIZE {self.block_size}\n" + _RAW_FACE_BSR_SPMV
            kernel = real_raw_kernel(source, "legendre_face_bsr_spmv")
            _RAW_KERNEL_CACHE[self.block_size] = kernel
        self.kernel = kernel

    def matvec(self, x, out=None):
        """Apply the fallback kernel asynchronously."""
        x = self.cp.asarray(x, dtype=REAL_DTYPE)
        if x.ndim != 1 or int(x.size) != self.size:
            raise ValueError(f"x must have shape ({self.size},)")
        if not x.flags.c_contiguous:
            x = self.cp.ascontiguousarray(x)
        if out is None:
            out = self.cp.empty_like(x)
        elif (out.ndim != 1 or int(out.size) != self.size
              or out.dtype != REAL_DTYPE or not out.flags.c_contiguous):
            raise ValueError(f"out must be contiguous at the selected real precision with shape ({self.size},)")
        if int(x.data.ptr) == int(out.data.ptr):
            raise ValueError("x and out must not alias")
        self._launch(x, out)
        return out

    def _launch(self, x, out):
        """Launch the unchanged row-owned SpMV after vector validation."""
        self.kernel((self.block_rows,), (32,), (
            np.int32(self.block_rows), self.indptr, self.indices, self.data, x, out,
        ))

    def close(self, *, suppress_errors: bool = False) -> None:
        """Match the descriptor-owning implementation lifecycle."""


class _PackedFaceBsrOperator(_RawFaceBsrOperator):
    """Packed SpMV with the same vector validation as the legacy raw operator."""

    def __init__(self, operator, *, threads=PACKED_THREADS_PER_BLOCK):
        """Retain the selected packed layout and its measured launch size."""
        self.cp = require_cupy()
        self.block_rows, self.block_size = operator.block_rows, operator.block_size
        self.size = self.block_rows * self.block_size
        self.packed = PackedFaceBsrKernels(operator, threads=threads)

    def _launch(self, x, out):
        """Apply the packed layout through the shared vector contract."""
        self.packed.spmv(x, out)


def _fused_smoother_kernel(block_size: int, *, zero: bool = False):
    """Compile the existing warp-owned smoother without allocating trial data."""
    if not 1 <= int(block_size) <= 10:
        raise NotImplementedError("fused block-Jacobi supports block sizes 1..10")
    cache = (_FUSED_BLOCK_JACOBI_ZERO_START_KERNEL_CACHE if zero
             else _FUSED_BLOCK_JACOBI_KERNEL_CACHE)
    kernel = cache.get(block_size)
    if kernel is None:
        source = (_FUSED_FACE_BLOCK_JACOBI_ZERO_START if zero else _FUSED_FACE_BLOCK_JACOBI_STEP)
        name = "legendre_face_block_jacobi_zero_start" if zero else "legendre_face_bsr_block_jacobi_step"
        kernel = real_raw_kernel(f"#define BLOCK_SIZE {block_size}\n" + source, name)
        cache[block_size] = kernel
    return kernel


def select_face_bsr_kernels(operator, diagonal_inverse=None, *, reuse=None):
    """Choose same-precision SpMV or legacy/packed Richardson applications.

    Smoother calibration returns a layout and launch size only. Chebyshev
    weights, sweeps, transfers, coarse correction and Krylov updates are fixed.
    """
    from hybridge.linalg.gpu.solver_tuning import (
        MAX_SAMPLE_MATRIX_BYTES, MAX_SAMPLE_ROWS, SAMPLING_VERSION,
        module_source_fingerprint, sample_bsr, screen_kernel,
    )
    from hybridge.linalg.gpu import solver_tuning
    from hybridge.runtime.cuda_tuning import (
        CandidateUnavailable, device_signature, fingerprint, select_variant,
    )

    cp = require_cupy()
    b = int(operator.block_size)
    smoother = diagonal_inverse is not None
    if not 1 <= b <= 10:
        raise NotImplementedError("automatic face-BSR kernels support block sizes 1..10")
    names = ["legacy", "packed:256", "packed:128", "packed:512"]
    if not smoother and b > 1:
        names.append("cusparse")
    device = device_signature(cp)
    samples, resources, descriptors = {}, {}, []

    def inputs():
        """Allocate bounded coefficient/graph samples only for a cache miss."""
        if not samples:
            samples.update(sample_bsr(cp, operator.indptr, operator.indices, operator.data,
                                      diagonal_inverse))
        return samples

    def prepare(name):
        """Compile and resource-screen an equivalent same-precision candidate."""
        if name == "cusparse":
            if int(cp.cuda.runtime.runtimeGetVersion()) < _GENERIC_BSR_MIN_RUNTIME:
                raise CandidateUnavailable("generic BSR SpMV requires CUDA runtime >=13010")
            try:
                _load_cusparse()
            except CusparseBsrUnavailable as exc:
                raise CandidateUnavailable(str(exc)) from exc
            resources[name] = {}
            return {}
        threads = 32 if name == "legacy" else int(name.split(":")[1])
        if name == "legacy":
            if smoother:
                kernels = (_fused_smoother_kernel(b), _fused_smoother_kernel(b, zero=True))
            else:
                if b not in _RAW_KERNEL_CACHE:
                    _RAW_KERNEL_CACHE[b] = real_raw_kernel(
                        f"#define BLOCK_SIZE {b}\n" + _RAW_FACE_BSR_SPMV, "legendre_face_bsr_spmv")
                kernels = (_RAW_KERNEL_CACHE[b],)
        else:
            try:
                module = _packed_module(b, b)
                functions = (("packed_face_bsr_block_jacobi_step", "packed_face_block_jacobi_zero_start")
                             if smoother else ("packed_face_bsr_spmv",))
                kernels = tuple(module.get_function(name) for name in functions)
            except Exception as exc:
                raise CandidateUnavailable(f"compilation: {exc}") from exc
            free, _ = cp.cuda.runtime.memGetInfo()
            shared_layout = getattr(getattr(operator, "_implementation", None), "packed", None)
            needed = (int(operator.data.nbytes) if shared_layout is None else 0)
            needed += 0 if not smoother else int(diagonal_inverse.nbytes)
            if needed > int(free) * .8:
                raise CandidateUnavailable("insufficient space for persistent packed layout")
        resources[name] = dict(threads=threads,
                               attributes=[screen_kernel(cp, k, threads, device) for k in kernels])
        return kernels

    def trial_prepare(name, handle):
        """Build trial layouts and optional descriptors before event timing."""
        d = inputs()
        if name == "cusparse":
            try:
                descriptor = _CusparseGenericBsrOperator(
                    d["indptr"], d["indices"], d["data"], shape=(d["rows"] * b, int(d["x"].size)))
            except CusparseBsrUnavailable as exc:
                raise CandidateUnavailable(str(exc)) from exc
            descriptors.append(descriptor)
            handle["descriptor"] = descriptor
            resources[name] = dict(workspace_bytes=descriptor.workspace_size)
        elif name.startswith("packed:") and "data_t" not in d:
            d["data_t"] = cp.ascontiguousarray(d["data"].transpose(0, 2, 1))
            if smoother:
                d["inverse_t"] = cp.ascontiguousarray(d["inverse"].transpose(0, 2, 1))

    def run(name, handle):
        """Apply a candidate to isolated SpMV or smoother test vectors."""
        d = inputs()
        rows = np.int32(d["rows"])
        if name == "cusparse":
            return handle["descriptor"].matvec(d["x"], out=d["out"])
        if name == "legacy":
            grid, block = (int(rows),), (32,)
            if smoother:
                handle[0](grid, block, (rows, d["indptr"], d["indices"], d["data"],
                                       d["inverse"], d["rhs"], d["x"], REAL_DTYPE(.375), d["out"]))
                handle[1](grid, block, (rows, d["inverse"], d["rhs"], REAL_DTYPE(.375), d["zero_out"]))
            else:
                handle[0](grid, block, (rows, d["indptr"], d["indices"], d["data"], d["x"], d["out"]))
        else:
            threads = int(name.split(":")[1])
            grid, block = (packed_grid(int(rows), b, threads),), (threads,)
            if smoother:
                handle[0](grid, block, (rows, d["indptr"], d["indices"], d["data_t"],
                                       d["inverse_t"], d["rhs"], d["x"], np.uint64(0), np.int32(0),
                                       REAL_DTYPE(.375), d["out"]))
                handle[1](grid, block, (rows, d["inverse_t"], d["rhs"], REAL_DTYPE(.375), d["zero_out"]))
            else:
                handle[0](grid, block, (rows, d["indptr"], d["indices"], d["data_t"], d["x"], d["out"]))
        return (d["out"], d["zero_out"]) if smoother else d["out"]

    try:
        name, _, report = select_variant(
            cupy=cp, identity=dict(operation="face-bsr-smoother" if smoother else "face-bsr-spmv",
                shape=list(operator.data.shape), block_rows=int(operator.block_rows),
                dtype=str(operator.data.dtype), layout="bsr-row-major", reuse=reuse or "iterative-fixed-matrix",
                sample_rows=MAX_SAMPLE_ROWS, sample_matrix_bytes=MAX_SAMPLE_MATRIX_BYTES,
                sampling=SAMPLING_VERSION, source=fingerprint([module_source_fingerprint(__file__),
                                                              module_source_fingerprint(solver_tuning.__file__)])),
            candidates=names, prepare=prepare, trial_prepare=trial_prepare, run=run,
            describe=lambda name, _: dict(resources[name], layout_setup_in_timing=False),
        )
    finally:
        for descriptor in descriptors:
            descriptor.close()
    return name, report


@dataclass
class LegendreFaceBsrOperator:
    """Device face-BSR matrix with measured equivalent SpMV dispatch.

    ``auto`` calibrates legacy, packed and supported cuSPARSE applications.
    ``legacy`` retains the historical cuSPARSE-first fallback policy.
    Explicit ``raw-cuda``, ``packed-raw-cuda`` and ``cusparse`` bypass tuning.
    """

    indptr: Any
    indices: Any
    data: Any
    backend: str = "auto"

    def __post_init__(self):
        """Validate BSR arrays and select the requested SpMV backend."""
        cp = require_cupy()
        self.indptr = cp.ascontiguousarray(self.indptr, dtype=cp.int32)
        self.indices = cp.ascontiguousarray(self.indices, dtype=cp.int32)
        self.data = cp.ascontiguousarray(self.data, dtype=REAL_DTYPE)
        if self.indptr.ndim != 1 or self.indices.ndim != 1:
            raise ValueError("BSR indptr and indices must be one-dimensional")
        if self.data.ndim != 3 or self.data.shape[1] != self.data.shape[2]:
            raise ValueError("BSR data must have shape (nnzb, b, b)")
        if int(self.data.shape[0]) != int(self.indices.size):
            raise ValueError("BSR data and index counts disagree")
        self.block_rows = int(self.indptr.size - 1)
        self.block_size = int(self.data.shape[1])
        self.shape = (self.block_rows * self.block_size,) * 2
        requested = str(self.backend).replace("_", "-").lower()
        if requested not in {"auto", "legacy", "cusparse", "raw", "raw-cuda", "packed", "packed-raw-cuda"}:
            raise ValueError("backend must be auto, legacy, cusparse, raw-cuda, or packed-raw-cuda")
        self.backend_requested = requested
        self.tuning_report = None
        self.fallback_reason = None
        packed_threads = PACKED_THREADS_PER_BLOCK
        if requested == "auto" and 1 <= self.block_size <= 10:
            variant, self.tuning_report = select_face_bsr_kernels(self)
            if variant.startswith("packed:"):
                packed_threads = int(variant.split(":")[1])
                requested = "packed-raw-cuda"
            else:
                requested = "raw-cuda" if variant == "legacy" else "auto"
        if requested in {"packed", "packed-raw-cuda"}:
            self._implementation = _PackedFaceBsrOperator(self, threads=packed_threads)
            self.backend_used = "packed-raw-cuda"
            if self.tuning_report is not None:
                self.tuning_report = dict(self.tuning_report,
                                          persistent_layout_bytes=self._implementation.packed.nbytes)
            return
        if requested == "legacy":
            requested = "auto"
        if requested in {"auto", "cusparse"} and self.block_size == 1:
            reason = (
                "CUDA generic BSR SpMV rejects 1x1 blocks; use the scalar CSR "
                "coarse solver or the row-owned fallback"
            )
            if requested == "cusparse":
                raise CusparseBsrUnavailable(reason)
            self.fallback_reason = reason
            self._implementation = _RawFaceBsrOperator(
                self.indptr, self.indices, self.data
            )
            self.backend_used = "raw-cuda"
        elif requested in {"auto", "cusparse"}:
            try:
                self._implementation = _CusparseGenericBsrOperator(
                    self.indptr, self.indices, self.data
                )
                self.backend_used = "cusparse-generic-bsr"
            except CusparseBsrUnavailable as exc:
                if requested == "cusparse":
                    raise
                self.fallback_reason = str(exc)
                self._implementation = _RawFaceBsrOperator(
                    self.indptr, self.indices, self.data
                )
                self.backend_used = "raw-cuda"
        else:
            self._implementation = _RawFaceBsrOperator(
                self.indptr, self.indices, self.data
            )
            self.backend_used = "raw-cuda"

    def matvec(self, vector, out=None):
        """Return ``A @ vector`` on the active stream."""
        return self._implementation.matvec(vector, out=out)

    def fused_block_jacobi_step(
            self, rhs, correction, diagonal_inverse, weight: float, out=None,
    ):
        """Apply one dense face-block Jacobi--Richardson stage.

        One warp owns one block row, loads each neighboring vector block once,
        consumes the dense BSR blocks cooperatively, and applies the dense
        inverse diagonal block without materializing a residual or update.
        """
        cp = require_cupy()
        rhs = cp.asarray(rhs, dtype=REAL_DTYPE)
        correction = cp.asarray(correction, dtype=REAL_DTYPE)
        diagonal_inverse = cp.asarray(diagonal_inverse, dtype=REAL_DTYPE)
        size = int(self.shape[0])
        for name, vector in (("rhs", rhs), ("correction", correction)):
            if (
                vector.ndim != 1 or int(vector.size) != size
                or not vector.flags.c_contiguous
            ):
                raise ValueError(
                    f"{name} must be contiguous at the selected real precision with shape ({size},)"
                )
        expected_diagonal_shape = (
            self.block_rows, self.block_size, self.block_size
        )
        if (
            tuple(diagonal_inverse.shape) != expected_diagonal_shape
            or not diagonal_inverse.flags.c_contiguous
        ):
            raise ValueError(
                "diagonal_inverse must be contiguous at the selected real precision with shape "
                f"{expected_diagonal_shape}"
            )
        weight = float(weight)
        if not np.isfinite(weight):
            raise ValueError("weight must be finite")
        if out is None:
            out = cp.empty_like(correction)
        elif (
            out.ndim != 1 or int(out.size) != size
            or out.dtype != REAL_DTYPE or not out.flags.c_contiguous
        ):
            raise ValueError(
                f"out must be contiguous at the selected real precision with shape ({size},)"
            )
        if int(out.data.ptr) in {int(rhs.data.ptr), int(correction.data.ptr)}:
            raise ValueError("out must not alias rhs or correction")
        kernel = _FUSED_BLOCK_JACOBI_KERNEL_CACHE.get(self.block_size)
        if kernel is None:
            if not 1 <= self.block_size <= 10:
                raise NotImplementedError(
                    "fused block-Jacobi supports block sizes 1..10"
                )
            source = (
                f"#define BLOCK_SIZE {self.block_size}\n"
                + _FUSED_FACE_BLOCK_JACOBI_STEP
            )
            kernel = real_raw_kernel(
                source, "legendre_face_bsr_block_jacobi_step"
            )
            _FUSED_BLOCK_JACOBI_KERNEL_CACHE[self.block_size] = kernel
        kernel((self.block_rows,), (32,), (
            np.int32(self.block_rows), self.indptr, self.indices, self.data,
            diagonal_inverse, rhs, correction, REAL_DTYPE(weight), out,
        ))
        return out

    def fused_block_jacobi_zero_start(
            self, rhs, diagonal_inverse, weight: float, out=None,
    ):
        """Apply the first Jacobi--Richardson stage from zero.

        For a provably zero correction, ``A @ correction`` vanishes and the
        update is exactly ``weight * diagonal_inverse @ rhs``. This dedicated
        warp-owned kernel therefore avoids reading or traversing the face-BSR
        operator during the first pre-smoothing stage.
        """
        cp = require_cupy()
        rhs = cp.asarray(rhs, dtype=REAL_DTYPE)
        diagonal_inverse = cp.asarray(diagonal_inverse, dtype=REAL_DTYPE)
        size = int(self.shape[0])
        if (
            rhs.ndim != 1 or int(rhs.size) != size
            or not rhs.flags.c_contiguous
        ):
            raise ValueError(
                f"rhs must be contiguous at the selected real precision with shape ({size},)"
            )
        expected_diagonal_shape = (
            self.block_rows, self.block_size, self.block_size
        )
        if (
            tuple(diagonal_inverse.shape) != expected_diagonal_shape
            or not diagonal_inverse.flags.c_contiguous
        ):
            raise ValueError(
                "diagonal_inverse must be contiguous at the selected real precision with shape "
                f"{expected_diagonal_shape}"
            )
        weight = float(weight)
        if not np.isfinite(weight):
            raise ValueError("weight must be finite")
        if out is None:
            out = cp.empty_like(rhs)
        elif (
            out.ndim != 1 or int(out.size) != size
            or out.dtype != REAL_DTYPE or not out.flags.c_contiguous
        ):
            raise ValueError(
                f"out must be contiguous at the selected real precision with shape ({size},)"
            )
        if int(out.data.ptr) == int(rhs.data.ptr):
            raise ValueError("out must not alias rhs")
        kernel = _FUSED_BLOCK_JACOBI_ZERO_START_KERNEL_CACHE.get(
            self.block_size
        )
        if kernel is None:
            if not 1 <= self.block_size <= 10:
                raise NotImplementedError(
                    "fused block-Jacobi supports block sizes 1..10"
                )
            source = (
                f"#define BLOCK_SIZE {self.block_size}\n"
                + _FUSED_FACE_BLOCK_JACOBI_ZERO_START
            )
            kernel = real_raw_kernel(
                source, "legendre_face_block_jacobi_zero_start"
            )
            _FUSED_BLOCK_JACOBI_ZERO_START_KERNEL_CACHE[
                self.block_size
            ] = kernel
        kernel((self.block_rows,), (32,), (
            np.int32(self.block_rows), diagonal_inverse, rhs,
            REAL_DTYPE(weight), out,
        ))
        return out

    def diagonal_blocks(self, positions):
        """Return a contiguous copy of diagonal blocks."""
        cp = require_cupy()
        positions = cp.asarray(positions, dtype=cp.int32)
        if positions.ndim != 1 or int(positions.size) != self.block_rows:
            raise ValueError("diagonal positions must have one entry per block row")
        return cp.ascontiguousarray(self.data[positions])

    def close(self) -> None:
        """Release descriptors and workspace."""
        self._implementation.close()

    def __matmul__(self, vector):
        """Apply the matrix through Python at-operator syntax."""
        return self.matvec(vector)


__all__ = [
    "CusparseBsrUnavailable",
    "LegendreFaceBsrOperator",
    "diagonal_block_positions",
    "legendre_orthonormal_scales",
    "modal_degree_schedule",
    "principal_modal_bsr_data",
    "prolong_modal",
    "restrict_modal",
    "transform_legendre_bsr_to_orthonormal",
]
