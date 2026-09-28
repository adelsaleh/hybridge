"""Reusable FP64 GPU solves from explicit SuperLU-compatible L/U factors.

No factorization is performed here. SpSM analysis uses CuPy's bindings; SpSV
and capture-compatible solves use HDGFEM's existing ctypes loader, without a build.
The factorization convention is Pr*A*Pc=L*U, with unit diagonal L and explicit
diagonal U. Callers must incorporate any additional factorization scaling.
"""
from __future__ import annotations

import ctypes
import time

import numpy as np

from .cupy import require_cupy


def superlu_gather_indices(perm_r, perm_c):
    """Return input/output gather indices for the SuperLU permutation convention."""
    rows, columns = np.asarray(perm_r), np.asarray(perm_c)
    if rows.ndim != 1 or columns.shape != rows.shape:
        raise ValueError("Row and column permutations must be equal-length vectors")
    expected = np.arange(rows.size)
    for permutation in (rows, columns):
        if permutation.dtype.kind not in "iu" or not np.array_equal(np.sort(permutation), expected):
            raise ValueError("Invalid factor permutation")
    dtype = np.int32 if rows.size < np.iinfo(np.int32).max else np.int64
    return np.argsort(rows).astype(dtype), columns.astype(dtype, copy=False)


def _check_status(status, operation):
    if status:
        raise RuntimeError(f"{operation} failed with cuSPARSE status {status}")


def _triangular_library():
    from .legendre_face_bsr import _load_cusparse

    lib = _load_cusparse()
    pointer, integer = ctypes.c_void_p, ctypes.c_int
    common = [pointer, integer, pointer, pointer, pointer, pointer, integer, integer, pointer]
    signatures = {
        "cusparseSpSV_createDescr": [ctypes.POINTER(pointer)],
        "cusparseSpSV_destroyDescr": [pointer],
        "cusparseSpSV_bufferSize": common + [ctypes.POINTER(ctypes.c_size_t)],
        "cusparseSpSV_analysis": common + [pointer],
        "cusparseSpSV_solve": common,
        "cusparseSpSM_solve": [pointer, integer, integer, pointer, pointer,
                               pointer, pointer, integer, integer, pointer],
    }
    for name, arguments in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = arguments, integer
    return lib


class _TriangularAnalysis:
    def __init__(self, matrix, rhs, output, *, lower, method, reserve_bytes):
        from cupyx import cusparse as descriptors
        from cupy_backends.cuda.libs import cusparse

        self.cp = require_cupy()
        self.backend, self.method = cusparse, method
        self.handle = self.cp.cuda.device.get_cusparse_handle()
        self.matrix, self.rhs, self.output = matrix, rhs, output
        self.alpha = np.array(1., dtype=np.float64)
        self.descriptors = []
        self.descr = None
        self.buffer = None
        self.lib = None
        try:
            self.lib = _triangular_library()
            a = descriptors.SpMatDescriptor.create(matrix)
            self.descriptors.append(a)
            a.set_attribute(cusparse.CUSPARSE_SPMAT_FILL_MODE,
                            cusparse.CUSPARSE_FILL_MODE_LOWER if lower else cusparse.CUSPARSE_FILL_MODE_UPPER)
            a.set_attribute(cusparse.CUSPARSE_SPMAT_DIAG_TYPE,
                            cusparse.CUSPARSE_DIAG_TYPE_UNIT if lower else cusparse.CUSPARSE_DIAG_TYPE_NON_UNIT)
            cusparse.setPointerMode(self.handle, cusparse.CUSPARSE_POINTER_MODE_HOST)
            if method == "spsm":
                b = descriptors.DnMatDescriptor.create(rhs.reshape(-1, 1))
                self.descriptors.append(b)
                x = descriptors.DnMatDescriptor.create(output.reshape(-1, 1))
                self.descriptors.append(x)
                self.descr = cusparse.spSM_createDescr()
                self.arguments = (self.handle, 0, 0, self.alpha.ctypes.data,
                                  a.desc, b.desc, x.desc, 1, 0, self.descr)
                size = cusparse.spSM_bufferSize(*self.arguments)
            elif method == "spsv":
                b = descriptors.DnVecDescriptor.create(rhs)
                self.descriptors.append(b)
                x = descriptors.DnVecDescriptor.create(output)
                self.descriptors.append(x)
                self.descr = ctypes.c_void_p()
                _check_status(self.lib.cusparseSpSV_createDescr(ctypes.byref(self.descr)), "SpSV create")
                self.arguments = (self.handle, 0, self.alpha.ctypes.data,
                                  a.desc, b.desc, x.desc, 1, 0, self.descr)
                size_pointer = ctypes.c_size_t()
                _check_status(self.lib.cusparseSpSV_bufferSize(*self.arguments, ctypes.byref(size_pointer)),
                              "SpSV buffer size")
                size = size_pointer.value
            else:
                raise ValueError(f"Unsupported triangular method: {method}")
            self.workspace_bytes = int(size)
            free = self.cp.cuda.runtime.memGetInfo()[0]
            if size + reserve_bytes > free:
                raise MemoryError(f"Triangular analysis needs {size / 1024**3:.2f} GiB; "
                                  f"free device memory is {free / 1024**3:.2f} GiB")
            self.buffer = self.cp.empty(max(1, size), dtype=self.cp.int8)
            if method == "spsm":
                cusparse.spSM_analysis(*self.arguments, self.buffer.data.ptr)
            else:
                self._set_stream()
                _check_status(self.lib.cusparseSpSV_analysis(*self.arguments, self.buffer.data.ptr), "SpSV analysis")
        except Exception:
            self.close()
            raise

    def _set_stream(self):
        # CuPy's setStream wrapper rejects capture even though these CUDA
        # triangular solve APIs support it. The installed native ABI does not.
        _check_status(self.lib.cusparseSetStream(self.handle, self.cp.cuda.get_current_stream().ptr),
                      "cuSPARSE set stream")

    def execute(self):
        # SpSM's output must be zero-initialized, including after a previous RHS.
        self.output.fill(0)
        self._set_stream()
        if self.method == "spsm":
            _check_status(self.lib.cusparseSpSM_solve(*self.arguments), "SpSM solve")
        else:
            _check_status(self.lib.cusparseSpSV_solve(*self.arguments), "SpSV solve")

    def close(self):
        if self.descr is not None:
            if self.method == "spsm":
                self.backend.spSM_destroyDescr(self.descr)
            elif self.lib is not None and self.descr.value:
                _check_status(self.lib.cusparseSpSV_destroyDescr(self.descr), "SpSV destroy")
            self.descr = None
        for descriptor in reversed(self.descriptors):
            if descriptor.desc is not None:
                descriptor.destroy(descriptor.desc)
                descriptor.desc = None
        self.descriptors.clear()
        self.buffer = None


class ReusableCuPyLUSolve:
    """Apply explicit FP64 LU with resident buffers and optional CUDA graph replay.

    ``method`` is ``spsv``, ``spsm`` or the high-level ``cupyx`` baseline.
    Analysis is retained by the first two methods. ``solve`` returns a buffer
    overwritten by the next call, unless an independent ``out`` is supplied.
    This object owns mutable workspaces and is not safe for concurrent calls.
    Close it after outstanding solves finish, preferably with a context manager.
    """

    def __init__(self, lower, upper, perm_r, perm_c, *, method="spsv", graph=False, reserve_bytes=0):
        from cupyx.scipy.sparse import csr_matrix

        cp = self.cp = require_cupy()
        if method not in {"spsv", "spsm", "cupyx"} or (graph and method == "cupyx"):
            raise ValueError("Graph replay requires cached spsv or spsm")
        if lower.shape != upper.shape or lower.shape[0] != lower.shape[1]:
            raise ValueError("Factors must be square and have matching shapes")
        for factor in (lower, upper):
            if not isinstance(factor, csr_matrix) or factor.dtype != np.dtype(np.float64):
                raise TypeError("Factors must be FP64 CuPy CSR matrices")
            if not factor.has_canonical_format:
                raise ValueError("Canonical CSR factors are required")
        rows, columns = superlu_gather_indices(perm_r, perm_c)
        if rows.size != lower.shape[0]:
            raise ValueError("Factor and permutation sizes differ")
        self.lower, self.upper, self.method = lower, upper, method
        self.closed = False
        self.graph = None
        self.triangular = []
        self._last_stream = cp.cuda.get_current_stream()
        self._vectors = []
        self.rows = self.columns = None
        started = time.perf_counter()
        try:
            self.rows, self.columns = cp.asarray(rows), cp.asarray(columns)
            self._vectors = [cp.zeros(rows.size, dtype=cp.float64) for _ in range(5)]
            self.input, self.permuted, self.y, self.z, self.output = self._vectors
            if method != "cupyx":
                for factor, rhs, result, is_lower in (
                        (lower, self.permuted, self.y, True), (upper, self.y, self.z, False)):
                    self.triangular.append(_TriangularAnalysis(
                        factor, rhs, result, lower=is_lower, method=method, reserve_bytes=reserve_bytes))
            cp.cuda.get_current_stream().synchronize()
            self.analysis_seconds = time.perf_counter() - started
            self.workspace_bytes = (sum(value.nbytes for value in self._vectors)
                                    + self.rows.nbytes + self.columns.nbytes
                                    + sum(item.workspace_bytes for item in self.triangular))
            self.graph_setup_seconds = 0.
            if graph:
                started = time.perf_counter()
                stream = cp.cuda.Stream(non_blocking=True)
                self._last_stream = stream
                with stream:
                    self._launch()  # Warm every operation before capture.
                    stream.synchronize()
                    stream.begin_capture()
                    try:
                        self._launch()
                        self.graph = stream.end_capture()
                    except Exception:
                        try:
                            stream.end_capture()
                        except Exception:
                            pass
                        raise
                stream.synchronize()
                self.graph_setup_seconds = time.perf_counter() - started
        except Exception:
            self.close()
            raise

    def _launch(self):
        cp = self.cp
        cp.take(self.input, self.rows, out=self.permuted)
        if self.method == "cupyx":
            from cupyx.scipy.sparse.linalg import spsolve_triangular
            y = spsolve_triangular(self.lower, self.permuted, lower=True, unit_diagonal=True)
            z = spsolve_triangular(self.upper, y, lower=False)
        else:
            for item in self.triangular:
                item.execute()
            z = self.z
        cp.take(z, self.columns, out=self.output)

    def solve(self, rhs, *, out=None):
        if self.closed:
            raise RuntimeError("LU solver is closed")
        if not isinstance(rhs, self.cp.ndarray) or rhs.shape != self.input.shape or rhs.dtype != self.input.dtype:
            raise ValueError("RHS must be a matching FP64 device vector")
        if out is not None and (not isinstance(out, self.cp.ndarray)
                                or out.shape != rhs.shape or out.dtype != rhs.dtype):
            raise ValueError("Output must be a matching FP64 device vector")
        self._last_stream = self.cp.cuda.get_current_stream()
        self.cp.copyto(self.input, rhs)
        if self.graph is None:
            self._launch()
        else:
            self.graph.launch(stream=self._last_stream)
        if out is not None:
            self.cp.copyto(out, self.output)
            return out
        return self.output

    def close(self):
        if not self.closed:
            self._last_stream.synchronize()
            self.graph = None
            for item in reversed(self.triangular):
                item.close()
            self.triangular.clear()
            self._vectors.clear()
            self.input = self.permuted = self.y = self.z = self.output = None
            self.rows = self.columns = None
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
