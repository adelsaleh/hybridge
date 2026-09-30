"""Scoped, verified MKL thread settings and a pattern-reusing solver for PyPardiso."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import os
import time

import numpy as np


@contextmanager
def pardiso_thread_limit(threads=16):
    """Set and verify the loaded backend's thread limit, restoring it on exit.

    ``threads='all'`` uses every affinity-visible CPU. Small hosts use all
    available CPUs when fewer than the requested number are available. The
    existing solver lock protects this process-wide MKL setting. This reports
    the actual backend limit; CPU activity must be measured around the work.
    """
    from hdgfem.linalg.direct import _import_pypardiso, _PYPARDISO_LOCK

    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    requested = available if threads == "all" else min(int(threads), available)
    if requested <= 0:
        raise ValueError("Pardiso thread count must be positive")
    with _PYPARDISO_LOCK:
        library = _import_pypardiso().ps.libmkl
        getter, setter = library.MKL_Get_Max_Threads, library.MKL_Set_Num_Threads
        get_dynamic, set_dynamic = library.MKL_Get_Dynamic, library.MKL_Set_Dynamic
        for function in (getter, get_dynamic):
            function.argtypes, function.restype = [], ctypes.c_int
        for function in (setter, set_dynamic):
            function.argtypes, function.restype = [ctypes.c_int], None
        previous, dynamic = int(getter()), int(get_dynamic())
        try:
            set_dynamic(0)
            setter(requested)
            actual = int(getter())
            if actual != requested:
                raise RuntimeError(f"MKL thread limit {actual} != requested {requested}")
            yield actual
        finally:
            setter(previous)
            set_dynamic(dynamic)


class ReusablePardisoSolver:
    """One oneMKL PARDISO instance for repeated nonsymmetric solves on a fixed COO pattern.

    The first solve (and any solve whose COO ``rows``/``cols`` differ from the
    stored pattern) builds the CSR pattern and a gather map with the parallel
    kernels of :mod:`hdgfem.kernels.sparse_pattern` and runs the reordering
    and symbolic analysis (phase 11). Every solve then gathers the new values
    (duplicates summed in COO order, finiteness checked on the way) and runs
    only the numerical factorization (phase 22) and the solve (phase 33). If a
    factorization with a reused analysis fails, the analysis is redone once.
    Weighted matching/scaling permutations computed in the analysis are reused
    with the new values; the physical residual ``b - A x`` is recomputed and
    reported for every solve. MKL is called directly with 1-based ``int32``
    index arrays cached per pattern. ``threads`` scopes the MKL thread count
    (``None`` keeps the current setting). Call :meth:`close` to free memory.
    """

    def __init__(self, *, threads=None):
        self.threads = threads
        self._solver = None
        self._rows = self._cols = None
        self._order = self._segments = self._ia = self._ja = None
        self._size = None
        self.analysis_count = 0
        self.factorization_count = 0
        self.last_threads = None

    def _pattern(self, rows, cols, size):
        """Build and store the CSR pattern, gather map and compact copy of a new COO pattern."""
        from hdgfem.linalg.sparse_pattern import build_coo_csr_pattern, index_dtype

        self._rows = self._cols = None
        indptr, indices, self._order, self._segments = build_coo_csr_pattern(rows, cols, size)
        if not np.diff(indptr).all():
            raise ValueError("Matrix A is singular, because it contains empty row(s)")
        if indices.size >= np.iinfo(np.int32).max:
            raise ValueError("PARDISO (32-bit interface) needs fewer than 2**31 nonzeros")
        self._ia = indptr.astype(np.int32) + np.int32(1)
        self._ja = indices.astype(np.int32) + np.int32(1)
        dtype = index_dtype(int(size) + 1)
        self._rows, self._cols = rows.astype(dtype), cols.astype(dtype)
        self._size = int(size)

    def _same_pattern(self, rows, cols, size) -> bool:
        """Whether ``(rows, cols, size)`` is the stored pattern (compared in parallel)."""
        from hdgfem.linalg.sparse_pattern import coo_pattern_mismatches

        return (self._rows is not None and self._size == int(size) and rows.shape == self._rows.shape
                and cols.shape == self._cols.shape
                and coo_pattern_mismatches(rows, cols, self._rows, self._cols) == 0)

    def _call(self, phase, values, rhs):
        """One MKL PARDISO call on the cached 1-based pattern (``pypardiso``'s call without index copies)."""
        from hdgfem.linalg.direct import _import_pypardiso

        solver = self._solver
        x = np.zeros_like(rhs)
        error = ctypes.c_int32(0)
        int_p, double_p = ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_double)
        solver._mkl_pardiso(solver.pt.ctypes.data_as(ctypes.POINTER(solver._pt_type[0])),
                            ctypes.byref(ctypes.c_int32(1)), ctypes.byref(ctypes.c_int32(1)),
                            ctypes.byref(ctypes.c_int32(solver.mtype)), ctypes.byref(ctypes.c_int32(phase)),
                            ctypes.byref(ctypes.c_int32(self._size)), values.ctypes.data_as(double_p),
                            self._ia.ctypes.data_as(int_p), self._ja.ctypes.data_as(int_p),
                            solver.perm.ctypes.data_as(int_p), ctypes.byref(ctypes.c_int32(1)),
                            solver.iparm.ctypes.data_as(int_p), ctypes.byref(ctypes.c_int32(solver.msglvl)),
                            rhs.ctypes.data_as(double_p), x.ctypes.data_as(double_p), ctypes.byref(error))
        if error.value != 0:
            raise _import_pypardiso().PyPardisoError(error.value)
        return x

    def solve_coo(self, rows, cols, data, rhs, size, *, rtol=0.0, atol=0.0, raise_on_nonconvergence=True):
        """Solve the COO system (duplicates summed) and return a :class:`SolveResult`."""
        from contextlib import nullcontext
        from hdgfem.linalg.sparse_pattern import csr_residual_norm, csr_values_from_coo
        from hdgfem.linalg.results import (
                    LinearSolveError,
                    SolveResult,
                    finalize_solve_result,
                    residual_diagnostics,
                )
        from hdgfem.linalg.direct import _import_pypardiso, _PYPARDISO_LOCK
        from hdgfem.linalg.results import _validate_finite_array

        rhs = np.ascontiguousarray(rhs, dtype=np.float64)
        _validate_finite_array(rhs, "rhs")
        if rhs.shape != (int(size),):
            raise ValueError(f"rhs has shape {rhs.shape}, expected ({int(size)},)")
        rows, cols = np.ascontiguousarray(rows), np.ascontiguousarray(cols)
        start = time.perf_counter()
        same = self._same_pattern(rows, cols, size)
        if not same:
            self._pattern(rows, cols, size)
        values, nonfinite = csr_values_from_coo(data, self._order, self._segments)
        if nonfinite:
            _validate_finite_array(np.asarray(data), "matrix data")
            raise ValueError("matrix data sums to non-finite values")
        assembly_elapsed = time.perf_counter() - start
        pypardiso = _import_pypardiso()
        context = pardiso_thread_limit(self.threads) if self.threads is not None else nullcontext(None)
        start = time.perf_counter()
        with _PYPARDISO_LOCK, context as actual_threads:
            self.last_threads = actual_threads
            if self._solver is None:
                self._solver = pypardiso.PyPardisoSolver(mtype=11)
                self._solver.set_iparm(12, 0)   # CSR input, no transposed solve

            analysed = reused = False
            try:
                if not same or self.analysis_count == 0:
                    self._call(11, values, rhs)
                    self.analysis_count += 1
                    analysed = True
                reused = not analysed
                self._call(22, values, rhs)
                x = self._call(33, values, rhs)
            except Exception as error:
                if analysed:
                    self._solver.free_memory(everything=True)
                    self._solver, self._rows = None, None
                    raise LinearSolveError(f"pypardiso solve failed: {error}") from error
                self._call(11, values, rhs)  # the reused analysis did not suit the new values: redo it once
                self.analysis_count += 1
                reused = False
                self._call(22, values, rhs)
                x = self._call(33, values, rhs)
            self.factorization_count += 1
        solve_elapsed = time.perf_counter() - start
        residual_norm = float(csr_residual_norm(self._ia, self._ja, values, x, rhs, 1))
        rhs_norm, relative, target = residual_diagnostics(residual_norm, rhs, rtol=rtol, atol=atol)
        result = SolveResult(
            x=x, residual_norm=residual_norm, rhs_norm=rhs_norm, relative_residual_norm=relative,
            residual_target=target, solver_residual_norm=residual_norm, solver_rhs_norm=rhs_norm,
            solver_relative_residual_norm=relative, solver_residual_target=target,
            physical_residual_norm=residual_norm, physical_rhs_norm=rhs_norm,
            physical_relative_residual_norm=relative, physical_residual_target=target,
            rtol=rtol, atol=atol, info=0, preconditioner=None,
            total_elapsed_seconds=assembly_elapsed + solve_elapsed, scale_elapsed_seconds=0.0,
            preconditioner_elapsed_seconds=0.0, solve_elapsed_seconds=solve_elapsed,
            matrix_assembly_elapsed_seconds=assembly_elapsed,
        )
        result.pardiso_analysis_reused = reused
        result.pardiso_threads = actual_threads
        return finalize_solve_result(result, backend="pypardiso-reused-analysis", backend_info=0,
                                     backend_success=True, raise_on_nonconvergence=raise_on_nonconvergence)

    def close(self):
        """Release PARDISO memory and the stored pattern."""
        if self._solver is not None:
            from hdgfem.linalg.direct import _PYPARDISO_LOCK
            with _PYPARDISO_LOCK:
                try:
                    self._solver.free_memory(everything=True)
                except Exception:
                    pass
        self._solver = self._rows = self._cols = None
