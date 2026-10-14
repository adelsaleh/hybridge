"""hybridge.linalg.amgx.host."""

from __future__ import annotations

from hybridge.runtime.precision import AMGX_MODE, REAL_DTYPE
from typing import Any
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import copy
from hybridge.runtime.optional import require_cupy, require_pyamgx

import numpy as np
import scipy.sparse.linalg
import time
from numpy.typing import NDArray
from hybridge.linalg.amgx.errors import as_amgx_capacity_error, destroy_amgx_objects
from hybridge.linalg.results import (
    SolveResult,
    compute_residual_norm,
    diagonal_scale_system,
    finalize_solve_result,
    residual_diagnostics,
    restricted_residual_diagnostics,
)
from hybridge.linalg.results import (
    _normalize_diagnostic_rows,
    _solver_print,
    _validate_finite_array,
    _validate_solver_controls,
)


_PYAMGX_RUNTIME_INITIALIZED = False


def initialize_pyamgx_once():
    """Initialize AMGX once per process.

    PyAMGX/AMGX 2.5 does not tolerate repeated ``initialize()`` calls in one
    Python process because plugin readers are registered globally.
    Register a flushed output callback on first initialization so native
    progress remains visible when stdout is redirected to a terminal log.
    """
    global _PYAMGX_RUNTIME_INITIALIZED
    amgx = require_pyamgx()
    if not _PYAMGX_RUNTIME_INITIALIZED:
        amgx.initialize()
        _PYAMGX_RUNTIME_INITIALIZED = True
        # AMGX's default printf callback becomes block-buffered under terminal
        # log pipes. Its public callback emits each native iteration row now,
        # rather than leaving it in C stdout until a failed solve is unwound.
        from hybridge.runtime.terminal import (
                    flush_native_stdio,
                    write_native_solver_output,
                )

        flush_native_stdio()
        amgx.register_print_callback(write_native_solver_output)
    return amgx


class _PyAMGXSharedResourceManager:
    """Own every AMGX ``Resources`` handle in the process and when it is destroyed.

    AMGX's ``Resources`` destructor tears down process-global state: the
    device and pinned memory pools and the cuSPARSE/cuBLAS handles that every
    other AMGX object uses (``free_resources`` in AMGX ``src/resources.cu``).
    Destroying one ``Resources`` while objects made from another are alive
    frees their memory under them. AMGX prints "trying to free non-empty
    temporary device pool", later solves read freed memory, and destroying
    those objects calls ``cudaFree`` on an interior pool pointer. That throws
    ``amgx::amgx_exception`` from a C++ destructor, and ``std::terminate``
    aborts the process (exit 134), possibly long after the faulty destroy.

    Every AMGX object hybridge creates therefore hangs off a lease from here:

    - ``acquire``/``release``: one shared, reference-counted ``Resources``
      built from the first owner's configuration, whose resource-level keys
      (``exception_handling``, ``device_mem_pool_*``) then apply to every
      sharer. Device solvers, the one-shot host solve and the dtype probe
      use it.
    - ``acquire_private``/``release_private`` (via ``private_amgx_resources``):
      a ``Resources`` from the caller's own configuration, for callers that
      need their own resource-level settings.

    No ``Resources`` is destroyed while a lease of either kind is outstanding.
    A released private one is parked and the shared one stays reusable; the
    last release destroys them all. Lease holders destroy their matrices,
    vectors, solvers and configs before releasing.
    """

    def __init__(self):
        """Initialize the instance."""
        self.pyamgx = None
        self.resource_cfg = None
        self.rsrc = None
        self.refcount = 0
        self.private_leases = 0
        self._parked = []

    def acquire(self, pyamgx, resource_config: dict):
        """Acquire the shared AMGX resource handle and increment its owner count."""
        initialize_pyamgx_once()
        if self.rsrc is None:
            self.pyamgx = pyamgx
            try:
                self.resource_cfg = pyamgx.Config()
                self.resource_cfg.create_from_dict(copy.deepcopy(resource_config))
                self.rsrc = pyamgx.Resources()
                self.rsrc.create_simple(self.resource_cfg)
            except Exception as exc:
                capacity_error = as_amgx_capacity_error(
                    exc, phase="resource acquisition", cp=require_cupy(), pyamgx=pyamgx
                )
                # A handle whose creation failed owns no global state.
                self._discard_shared(suppress_errors=True)
                if capacity_error is not None and capacity_error is not exc:
                    raise capacity_error from exc
                raise
        self.refcount += 1
        return self.rsrc

    def release(self, *, suppress_errors: bool = False) -> None:
        """Release one shared owner; destroy the handles once no lease remains."""
        if self.refcount > 0:
            self.refcount -= 1
        self._destroy_if_idle(suppress_errors=suppress_errors)

    def acquire_private(self, pyamgx, resource_config: dict):
        """Create a ``Resources`` from ``resource_config``; return the ``(resources, config)`` lease."""
        initialize_pyamgx_once()
        cfg = rsrc = None
        try:
            cfg = pyamgx.Config()
            cfg.create_from_dict(copy.deepcopy(resource_config))
            rsrc = pyamgx.Resources()
            rsrc.create_simple(cfg)
        except Exception:
            destroy_amgx_objects((rsrc, cfg), suppress_errors=True)
            raise
        self.private_leases += 1
        return rsrc, cfg

    def release_private(self, lease, *, suppress_errors: bool = False) -> None:
        """Park a private lease; destroy the handles once no lease remains."""
        if self.private_leases > 0:
            self.private_leases -= 1
        self._parked.append(tuple(lease))
        self._destroy_if_idle(suppress_errors=suppress_errors)

    def _destroy_if_idle(self, *, suppress_errors: bool) -> None:
        """Destroy every parked and shared handle once no lease is outstanding."""
        if self.refcount or self.private_leases:
            return
        # Each Resources goes before the Config it was created from.
        objects = [obj for lease in reversed(self._parked) for obj in lease]
        self._parked.clear()
        self._discard_shared(suppress_errors=suppress_errors, objects=objects)

    def _discard_shared(self, *, suppress_errors: bool, objects=()) -> None:
        """Destroy ``objects`` and the shared handle, then forget the shared handle."""
        try:
            destroy_amgx_objects(
                (*objects, self.rsrc, self.resource_cfg), suppress_errors=suppress_errors
            )
        finally:
            self.rsrc = None
            self.resource_cfg = None
            self.pyamgx = None


_AMGX_SHARED_RESOURCES = _PyAMGXSharedResourceManager()


@contextmanager
def private_amgx_resources(resource_config: Mapping[str, Any]) -> Iterator[Any]:
    """Yield an AMGX ``Resources`` built from ``resource_config`` under a private lease.

    Use this, never ``pyamgx.Resources().create_simple``, when a caller needs
    its own resource-level settings (e.g. ``device_mem_pool_enabled``).
    Destroy every object made from it before the block exits. The handle is
    destroyed only once no other hybridge AMGX lease is outstanding (see
    ``_PyAMGXSharedResourceManager``).
    """
    amgx = initialize_pyamgx_once()
    manager = _AMGX_SHARED_RESOURCES
    lease = manager.acquire_private(amgx, dict(resource_config))
    failed = True
    try:
        yield lease[0]
        failed = False
    finally:
        manager.release_private(lease, suppress_errors=failed)


_PYAMGX_DTYPE_SUPPORT: dict[str, bool] = {}


def pyamgx_supports_real_dtype(dtype) -> bool:
    """Whether the loaded PyAMGX binding accepts ``dtype`` arrays in its matching mode.

    The default build of the PyAMGX fork hard-codes float64 arrays, so FP32
    (``dFFI``) needs the mode-aware binding from
    ``scripts/dev/build_pyamgx_precision.py``. The probe uploads one tiny
    vector once per process per dtype, on the shared AMGX resources. Only
    the binding's dtype rejection counts as unsupported; any other failure
    propagates.
    """
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("PyAMGX real arrays are float32 or float64")
    if dtype.str not in _PYAMGX_DTYPE_SUPPORT:
        amgx = initialize_pyamgx_once()
        mode = "dFFI" if dtype == np.dtype(np.float32) else "dDDI"
        manager = _AMGX_SHARED_RESOURCES
        resources = manager.acquire(amgx, {"config_version": 2, "solver": {"solver": "BICGSTAB"}})
        vector = None
        try:
            vector = amgx.Vector().create(resources, mode=mode)
            try:
                vector.upload(np.zeros(2, dtype=dtype), block_dim=1)
                supported = True
            except ValueError as error:
                if "dtype" not in str(error):
                    raise
                supported = False
        finally:
            try:
                destroy_amgx_objects((vector,))
            finally:
                manager.release()
        _PYAMGX_DTYPE_SUPPORT[dtype.str] = supported
    return _PYAMGX_DTYPE_SUPPORT[dtype.str]


def default_pyamgx_config(*, tolerance: float, maxiter: int | None, verbose: bool | int = 0) -> dict[str, Any]:
    """Return the default AMGX BICGSTAB+AMG configuration."""
    monitor = int(bool(verbose) and int(verbose) >= 3)
    return {
        "config_version": 2,
        "determinism_flag": 1,
        "exception_handling": 1,
        "solver": {
            "solver": "BICGSTAB",
            "monitor_residual": monitor,
            "convergence": "RELATIVE_INI_CORE",
            "tolerance": float(tolerance),
            "max_iters": int(maxiter) if maxiter is not None else 1500,
            "obtain_timings": int(bool(verbose) and int(verbose) >= 2),
            "preconditioner": {
                "solver": "AMG",
                "algorithm": "CLASSICAL",
                "selector": "PMIS",
                "cycle": "V",
                "strength_threshold": 0.5,
                "coarse_solver": "DENSE_LU_SOLVER",
                "presweeps": 2,
                "postsweeps": 2,
                "max_levels": 50,
            },
        },
    }


def solve_pyamgx_csr(
        matrix,
        rhs,
        *,
        initial_guess=None,
        config: Mapping[str, Any] | None = None,
        tolerance: float = 1e-13,
        maxiter: int | None = None,
        verbose: bool | int = 0,
        return_info: bool = False,
):
    """Solve a CuPy CSR system with PyAMGX and optionally return native diagnostics.

    The one-shot objects live on the shared AMGX resources: destroying a
    private ``Resources`` here would free the memory of every live device
    solver (see ``_PyAMGXSharedResourceManager``).
    """
    cupy = require_cupy()
    amgx = initialize_pyamgx_once()
    amgx_config = default_pyamgx_config(tolerance=tolerance, maxiter=maxiter, verbose=verbose)
    if config is not None:
        amgx_config = dict(config)

    info = {"amgx_status": "unknown", "amgx_iterations": None, "residual_history": ()}
    manager = _AMGX_SHARED_RESOURCES
    cfg = rsrc = mat = vec_b = vec_x = solver = None
    failed = True
    failure_phase = "solution allocation"
    try:
        rhs_cp = cupy.asarray(rhs, dtype=REAL_DTYPE)
        if initial_guess is None:
            x_cp = cupy.zeros_like(rhs_cp)
        else:
            x_cp = cupy.asarray(initial_guess, dtype=REAL_DTYPE).copy()
            if x_cp.shape != rhs_cp.shape:
                raise ValueError(f"initial_guess must have shape {rhs_cp.shape}; got {x_cp.shape}")
        failure_phase = "configuration creation"
        cfg = amgx.Config()
        cfg.create_from_dict(amgx_config)
        failure_phase = "resource acquisition"
        rsrc = manager.acquire(amgx, amgx_config)
        failure_phase = "solver-object creation"
        mat = amgx.Matrix()
        mat.create(rsrc, mode=AMGX_MODE)
        vec_b = amgx.Vector()
        vec_b.create(rsrc, mode=AMGX_MODE)
        vec_x = amgx.Vector()
        vec_x.create(rsrc, mode=AMGX_MODE)
        solver = amgx.Solver()
        solver.create(rsrc, cfg, mode=AMGX_MODE)
        failure_phase = "matrix upload"
        mat.upload_CSR(matrix)
        failure_phase = "vector upload"
        vec_b.upload_raw(rhs_cp.data.ptr, rhs_cp.size)
        vec_x.upload_raw(x_cp.data.ptr, x_cp.size)
        failure_phase = "solver setup"
        solver.setup(mat)
        failure_phase = "solver iteration"
        solver.solve(vec_b, vec_x)
        failure_phase = "solution download"
        vec_x.download_raw(x_cp.data.ptr)
        cupy.cuda.get_current_stream().synchronize()

        try:
            info["amgx_status"] = str(solver.status)
        except Exception:
            pass
        try:
            info["amgx_iterations"] = int(solver.iterations_number)
        except Exception:
            pass
        store_residual_history = bool(
            amgx_config.get("solver", {}).get("store_res_history", 0)
        )
        if store_residual_history and info["amgx_iterations"] is not None:
            history = []
            first = max(0, info["amgx_iterations"] - 63)
            for iteration in range(first, info["amgx_iterations"] + 1):
                try:
                    history.append(float(solver.get_residual(iteration)))
                except Exception:
                    history = []
                    break
            info["residual_history"] = tuple(history)
        failed = False
    except Exception as exc:
        capacity_error = as_amgx_capacity_error(
            exc, phase=failure_phase, cp=cupy, pyamgx=amgx
        )
        if capacity_error is not None and capacity_error is not exc:
            raise capacity_error from exc
        raise
    finally:
        cleanup_error = None
        try:
            destroy_amgx_objects((solver, vec_x, vec_b, mat, cfg))
        except Exception as exc:
            cleanup_error = exc
        if rsrc is not None:
            try:
                manager.release()
            except Exception as exc:
                cleanup_error = cleanup_error or exc
        if cleanup_error is not None and not failed:
            raise cleanup_error
    return (x_cp, info) if return_info else x_cp


def solve_pyamgx_system(
    matrix: scipy.sparse.spmatrix | scipy.sparse.sparray,
    rhs: NDArray,
    *,
    initial_guess: NDArray | None = None,
    config: Mapping[str, Any] | None = None,
    rtol: float = 1e-13,
    atol: float = 0.0,
    maxiter: int | None = None,
    scale_system: bool = True,
    raise_on_nonconvergence: bool = False,
    verbose: bool | int = 0,
    diagnostic_rows: NDArray | None = None,
    diagnostic_label: str | None = None,
) -> SolveResult:
    """Solve an assembled sparse system with PyAMGX.

    The solve happens on the GPU. Diagnostics are computed on the host using the
    same conventions as the SciPy and PETSc paths.
    """
    _validate_solver_controls(rtol=rtol, atol=atol, maxiter=maxiter)
    total_start = time.time()
    from hybridge.runtime.optional import asnumpy
    from hybridge.linalg.gpu.sparse import scipy_csr_to_cupy

    physical_matrix = matrix.tocsr()
    physical_rhs = np.asarray(rhs, dtype=REAL_DTYPE)
    if physical_matrix.shape != (physical_rhs.size, physical_rhs.size):
        raise ValueError(
            f"matrix must have shape ({physical_rhs.size}, {physical_rhs.size}), "
            f"got {physical_matrix.shape}"
        )
    _validate_finite_array(np.asarray(physical_matrix.data), "matrix data")
    _validate_finite_array(physical_rhs, "rhs")
    diagnostic_rows = _normalize_diagnostic_rows(diagnostic_rows, physical_rhs.size)
    if initial_guess is not None:
        initial_guess = np.asarray(initial_guess, dtype=REAL_DTYPE)
        if initial_guess.shape != physical_rhs.shape:
            raise ValueError(f"initial_guess must have shape {physical_rhs.shape}; got {initial_guess.shape}")
        _validate_finite_array(initial_guess, "initial_guess")

    scale_start = time.time()
    if scale_system:
        solve_matrix, solve_rhs = diagonal_scale_system(physical_matrix, physical_rhs, copy_matrix=True)
    else:
        solve_matrix = physical_matrix
        solve_rhs = physical_rhs
    scale_elapsed_seconds = time.time() - scale_start

    matrix_start = time.time()
    matrix_cp = scipy_csr_to_cupy(solve_matrix)
    matrix_elapsed_seconds = time.time() - matrix_start
    _solver_print(
        verbose,
        2,
        "  PyAMGX CSR matrix copied to device in %.5fs with nnz=%d",
        matrix_elapsed_seconds,
        solve_matrix.nnz,
    )

    initial_residual_norm = None
    if initial_guess is not None:
        initial_residual_norm = compute_residual_norm(physical_matrix, initial_guess, physical_rhs)
        _solver_print(verbose, 2, "  initial residual from supplied guess: %.3e", initial_residual_norm)

    solve_start = time.time()
    solution_cp, amgx_info = solve_pyamgx_csr(
        matrix_cp,
        solve_rhs,
        initial_guess=initial_guess,
        config=config,
        tolerance=rtol,
        maxiter=maxiter,
        verbose=verbose,
        return_info=True,
    )
    solution = np.ascontiguousarray(asnumpy(solution_cp), dtype=REAL_DTYPE)
    solution_is_finite = bool(np.all(np.isfinite(solution)))
    solve_elapsed_seconds = time.time() - solve_start

    solver_residual = solve_matrix @ solution - solve_rhs
    solver_residual_norm = float(np.linalg.norm(solver_residual))
    solver_rhs_norm, solver_relative_residual_norm, solver_residual_target = residual_diagnostics(
        solver_residual_norm,
        solve_rhs,
        rtol=rtol,
        atol=atol,
    )

    physical_residual = physical_matrix @ solution - physical_rhs
    physical_residual_norm = float(np.linalg.norm(physical_residual))
    physical_rhs_norm, physical_relative_residual_norm, physical_residual_target = residual_diagnostics(
        physical_residual_norm,
        physical_rhs,
        rtol=rtol,
        atol=atol,
    )
    diagnostic_residual_norm, diagnostic_rhs_norm, diagnostic_relative_residual_norm, diagnostic_residual_target = (
        restricted_residual_diagnostics(physical_residual, physical_rhs, diagnostic_rows, rtol=rtol, atol=atol)
    )

    total_elapsed_seconds = time.time() - total_start
    _solver_print(
        verbose,
        1,
        "  PyAMGX finished in %.5fs with solver_rel=%.3e",
        solve_elapsed_seconds,
        solver_relative_residual_norm,
    )
    _solver_print(
        verbose,
        2,
        "  residuals: solver_abs=%.3e, solver_target=%.3e, physical_abs=%.3e, physical_rel=%.3e",
        solver_residual_norm,
        solver_residual_target,
        physical_residual_norm,
        physical_relative_residual_norm,
    )
    _solver_print(
        verbose,
        2,
        "  timings: scale=%.5fs, cupy_matrix=%.5fs, amgx=%.5fs, total=%.5fs",
        scale_elapsed_seconds,
        matrix_elapsed_seconds,
        solve_elapsed_seconds,
        total_elapsed_seconds,
    )

    native_status = str(amgx_info.get("amgx_status", "unknown"))
    normalized_status = native_status.lower().replace("-", "_").replace(" ", "_")
    backend_success = normalized_status == "unknown" or not any(
        marker in normalized_status
        for marker in ("fail", "diverg", "not_converged", "notconverged")
    )
    result = SolveResult(
        x=solution,
        residual_norm=solver_residual_norm,
        info=0 if backend_success else 1,
        preconditioner=None,
        total_elapsed_seconds=total_elapsed_seconds,
        scale_elapsed_seconds=scale_elapsed_seconds,
        preconditioner_elapsed_seconds=matrix_elapsed_seconds,
        solve_elapsed_seconds=solve_elapsed_seconds,
        iteration_count=amgx_info.get("amgx_iterations"),
        initial_residual_norm=initial_residual_norm,
        rhs_norm=solver_rhs_norm,
        relative_residual_norm=solver_relative_residual_norm,
        residual_target=solver_residual_target,
        solver_residual_norm=solver_residual_norm,
        solver_rhs_norm=solver_rhs_norm,
        solver_relative_residual_norm=solver_relative_residual_norm,
        solver_residual_target=solver_residual_target,
        physical_residual_norm=physical_residual_norm,
        physical_rhs_norm=physical_rhs_norm,
        physical_relative_residual_norm=physical_relative_residual_norm,
        physical_residual_target=physical_residual_target,
        diagnostic_residual_label=diagnostic_label,
        diagnostic_residual_norm=diagnostic_residual_norm,
        diagnostic_rhs_norm=diagnostic_rhs_norm,
        diagnostic_relative_residual_norm=diagnostic_relative_residual_norm,
        diagnostic_residual_target=diagnostic_residual_target,
        rtol=rtol,
        atol=atol,
    )
    return finalize_solve_result(
        result,
        backend="pyamgx",
        backend_info=native_status,
        backend_success=backend_success,
        solution_is_finite=solution_is_finite,
        residual_history=amgx_info.get("residual_history"),
        raise_on_nonconvergence=raise_on_nonconvergence,
    )
