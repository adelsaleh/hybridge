"""Experimental Legendre face-BSR operators for HDG Poisson.

CUDA 13 generic-BSR ``cusparseSpMV`` is preferred.  CuPy 14.1 does not expose
``cusparseCreateBsr``, so this module owns a narrow binding around CuPy's
cuSPARSE handle.  A row-owned RawKernel is the explicit fallback and benchmark
control.  This is an internal prototype, not a public backend contract.
"""

from __future__ import annotations

from hdgfem.runtime.precision import REAL_DTYPE, REAL_ITEMSIZE, real_raw_kernel

import ctypes
from dataclasses import dataclass
from typing import Any

import numpy as np

from hdgfem.runtime.optional import require_cupy


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
        self.kernel((self.block_rows,), (32,), (
            np.int32(self.block_rows), self.indptr, self.indices, self.data, x, out,
        ))
        return out

    def close(self, *, suppress_errors: bool = False) -> None:
        """Match the descriptor-owning implementation lifecycle."""


@dataclass
class LegendreFaceBsrOperator:
    """Device face-BSR matrix with cuSPARSE-first SpMV dispatch."""

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
        if requested not in {"auto", "cusparse", "raw", "raw-cuda"}:
            raise ValueError("backend must be auto, cusparse, or raw-cuda")
        self.fallback_reason = None
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
