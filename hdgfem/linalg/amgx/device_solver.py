"""hdgfem.linalg.amgx.device_solver."""

from __future__ import annotations

import copy
import numpy as np
import time
import warnings
from hdgfem.runtime.precision import AMGX_MODE, REAL_DTYPE, audit_arrays
from typing import Any
from hdgfem.linalg.results import (
    LinearSolveCapacityError,
    LinearSolveConvergenceError,
    SolveResult,
    finalize_solve_result,
)
from hdgfem.linalg.gpu.sparse import (
    _DeviceBsrMatrixView,
    _DeviceCsrMatrixView,
    _as_cupyx_csr_matrix,
    _assembly_device_csr_matrix,
    _device_compressed_matvec,
    _diagonal_scale_bsr_rows_in_place,
    diagonal_scale_cupy_csr_rows_in_place as _diagonal_scale_csr_rows_in_place,
    _restore_left_scaled_bsr_rows_in_place,
    _restore_scaled_csr_rows_in_place,
    _scalarize_device_bsr_matrix,
    symmetric_scale_cupy_csr_in_place,
)
from hdgfem.linalg.amgx.errors import (
    as_amgx_capacity_error as _as_amgx_capacity_error,
    destroy_amgx_objects,
)
from hdgfem.linalg.amgx.config import format_amgx_configuration
from hdgfem.linalg.amgx.host import initialize_pyamgx_once
from dataclasses import replace
from hdgfem.runtime.optional import require_cupy, require_cupyx_sparse, require_pyamgx


def _residual_stats_cp(residual, rhs, *, rtol: float, atol: float):
    """Compute device residual norms and the requested convergence target."""
    cp = require_cupy()
    residual_norm = float(cp.linalg.norm(residual).get())
    rhs_norm = float(cp.linalg.norm(rhs).get())
    relative = residual_norm / rhs_norm if rhs_norm > 0.0 else residual_norm
    target = max(float(rtol) * rhs_norm, float(atol))
    return residual_norm, rhs_norm, relative, target


_AMGX_REUSABLE_SOLVERS = []


class _PyAMGXSharedResourceManager:
    """Own one process-wide AMGX Resources handle shared by live solvers."""

    def __init__(self):
        """Initialize the instance."""
        self.pyamgx = None
        self.resource_cfg = None
        self.rsrc = None
        self.refcount = 0

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
                capacity_error = _as_amgx_capacity_error(
                    exc, phase="resource acquisition", cp=require_cupy(), pyamgx=pyamgx
                )
                self.release(suppress_errors=True)
                if capacity_error is not None and capacity_error is not exc:
                    raise capacity_error from exc
                raise
        self.refcount += 1
        return self.rsrc

    def release(self, *, suppress_errors: bool = False) -> None:
        """Release one owner and destroy shared AMGX resources when unused."""
        if self.refcount > 0:
            self.refcount -= 1
        if self.refcount != 0:
            return
        try:
            destroy_amgx_objects(
                (self.rsrc, self.resource_cfg), suppress_errors=suppress_errors
            )
        finally:
            self.rsrc = None
            self.resource_cfg = None
            self.pyamgx = None


_AMGX_SHARED_RESOURCES = _PyAMGXSharedResourceManager()



def _amgx_config_for_solve(*, config=None, tolerance: float = 1e-13, maxiter: int | None = None, verbose: bool | int = 0, fixed_amg_cycles: int | None = None):
    """Build an AMGX solver configuration with normalized controls and diagnostics."""
    from hdgfem.linalg.amgx.host import default_pyamgx_config

    if config is None:
        amgx_config = default_pyamgx_config(tolerance=tolerance, maxiter=maxiter, verbose=verbose)
    else:
        amgx_config = copy.deepcopy(config)
        solver_config = amgx_config.setdefault("solver", {})
        if "tolerance" not in solver_config:
            solver_config["tolerance"] = float(tolerance)
        if maxiter is not None:
            solver_config["max_iters"] = int(maxiter)

    verbose_level = 1 if isinstance(verbose, bool) and verbose else (0 if not verbose else int(verbose))
    solver_config = amgx_config.setdefault("solver", {})
    if fixed_amg_cycles is not None:
        if (isinstance(fixed_amg_cycles, bool) or not isinstance(fixed_amg_cycles, int)
                or fixed_amg_cycles <= 0):
            raise ValueError("fixed_amg_cycles must be a positive integer")
        if solver_config.get("solver") != "AMG":
            raise ValueError("fixed_amg_cycles requires an AMG solver")
        solver_config["max_iters"] = fixed_amg_cycles
        solver_config["monitor_residual"] = 0
        solver_config["store_res_history"] = 0
    else:
        solver_config["monitor_residual"] = 1
        solver_config.setdefault("store_res_history", 1)
    if verbose_level == 2 or verbose_level >= 4:
        solver_config["obtain_timings"] = 1
    if verbose_level >= 3:
        solver_config["print_solve_stats"] = 1
        # Honor an explicit cadence, including the transport guard's compact
        # default. The native loop always prints the last completed iteration.
        solver_config.setdefault("print_solve_stats_interval", 1)
    return amgx_config


def _amgx_relative_residual_check_rtol(config, fallback: float) -> float:
    """Return the configured relative solver-system validation tolerance."""
    solver = {} if config is None else config.get("solver", {})
    convergence = str(solver.get("convergence", "")).upper()
    if convergence.startswith("RELATIVE"):
        value = float(solver.get("tolerance", fallback))
        if np.isfinite(value) and value >= 0.0:
            return value
    return float(fallback)


def _validate_amgx_block_configuration(config, block_dim: int) -> None:
    """Reject known scalar-only solvers before invoking native block setup."""
    if block_dim <= 1:
        return

    def check(node, path: str) -> None:
        """Reject ``CHEBYSHEV_POLY`` in a solver, preconditioner or smoother for block matrices."""
        solver_name = node.get("solver") if isinstance(node, dict) else node
        if isinstance(solver_name, str) and solver_name.upper() == "CHEBYSHEV_POLY":
            raise ValueError(
                f"AMGX {path} uses scalar-only CHEBYSHEV_POLY with "
                f"block size {block_dim}; use CHEBYSHEV with a "
                "block-compatible preconditioner or scalarize the matrix"
            )
        if isinstance(node, dict):
            # These solvers act on the same matrix. A coarse_solver may act
            # on scalar-expanded levels, so its block size is not known here.
            for key in ("solver", "preconditioner", "smoother"):
                if key in node:
                    check(node[key], f"{path}.{key}")

    check(config, "config")


class PyAMGXCsrDeviceSolver:
    """Reusable PyAMGX CSR solver for a fixed device-resident matrix."""

    def __init__(
            self,
            *,
            config=None,
            tolerance: float = 1e-13,
            maxiter: int | None = None,
            verbose: bool | int = 0,
            reusable: bool = False,
            fixed_amg_cycles: int | None = None,
    ):
        """Initialize a solve, or an explicitly fixed number of AMG cycles.

        Fixed cycles disable tolerance-based early exit for preconditioner use.
        Ordinary solver calls retain convergence monitoring; history is optional.
        """
        self.cp = require_cupy()
        self.pyamgx = require_pyamgx()
        self.config_dict = _amgx_config_for_solve(config=config, tolerance=tolerance, maxiter=maxiter, verbose=verbose, fixed_amg_cycles=fixed_amg_cycles)
        self.verbose_level = (
            1
            if isinstance(verbose, bool) and verbose
            else (0 if not verbose else int(verbose))
        )
        self.cfg = self.rsrc = self.mat = self.vec_b = self.vec_x = self.solver = None
        self.shape = None
        self.size = None
        self.block_rows = None
        self.block_dim = 1
        self.last_matrix_upload_elapsed_seconds = 0.0
        self.last_solver_setup_elapsed_seconds = 0.0
        self.last_coefficients_replace_elapsed_seconds = 0.0
        self.setup_count = 0
        self.coefficients_replace_count = 0
        self.is_setup = False
        self.closed = False
        self.reusable = bool(reusable)
        self._shared_resources_acquired = False
        failure_phase = "resource acquisition"
        try:
            self.rsrc = _AMGX_SHARED_RESOURCES.acquire(self.pyamgx, self.config_dict)
            self._shared_resources_acquired = True
            failure_phase = "configuration creation"
            self.cfg = self.pyamgx.Config()
            self.cfg.create_from_dict(self.config_dict)
            failure_phase = "solver-object creation"
            self.mat = self.pyamgx.Matrix()
            self.mat.create(self.rsrc, mode=AMGX_MODE)
            self.vec_b = self.pyamgx.Vector()
            self.vec_b.create(self.rsrc, mode=AMGX_MODE)
            self.vec_x = self.pyamgx.Vector()
            self.vec_x.create(self.rsrc, mode=AMGX_MODE)
            self.solver = self.pyamgx.Solver()
            self.solver.create(self.rsrc, self.cfg, mode=AMGX_MODE)
            if self.reusable:
                _AMGX_REUSABLE_SOLVERS.append(self)
        except Exception as exc:
            capacity_error = _as_amgx_capacity_error(
                exc, phase=failure_phase, cp=self.cp, pyamgx=self.pyamgx
            )
            self.close(suppress_errors=True)
            if capacity_error is not None and capacity_error is not exc:
                raise capacity_error from exc
            raise

    def setup(self, matrix) -> float:
        """Upload and set up a fixed device CSR or face-BSR matrix in AMGX."""
        if self.closed:
            raise RuntimeError("cannot set up a closed PyAMGXCsrDeviceSolver")
        setup_start = time.perf_counter()
        self.last_matrix_upload_elapsed_seconds = 0.0
        self.last_solver_setup_elapsed_seconds = 0.0
        failure_phase = "matrix upload"
        try:
            block_dim = int(getattr(matrix, "block_size", 1))
            if block_dim < 1 or matrix.shape[0] % block_dim or matrix.shape[1] % block_dim:
                raise ValueError(
                    f"matrix shape {matrix.shape} is incompatible with block size {block_dim}"
                )
            _validate_amgx_block_configuration(self.config_dict, block_dim)
            block_shape = (
                int(matrix.shape[0] // block_dim),
                int(matrix.shape[1] // block_dim),
            )
            matrix_upload_start = time.perf_counter()
            audit_arrays('amgx-matrix-upload', matrix)
            self.mat.upload(
                matrix.indptr,
                matrix.indices,
                matrix.data,
                block_dims=[block_dim, block_dim],
                shape=block_shape,
            )
            self.cp.cuda.get_current_stream().synchronize()
            self.last_matrix_upload_elapsed_seconds = time.perf_counter() - matrix_upload_start
            failure_phase = "solver setup"
            solver_setup_start = time.perf_counter()
            self.solver.setup(self.mat)
            failure_phase = "setup synchronization"
            self.cp.cuda.get_current_stream().synchronize()
            self.last_solver_setup_elapsed_seconds = time.perf_counter() - solver_setup_start
            self.shape = tuple(matrix.shape)
            self.size = int(matrix.shape[0])
            self.block_rows = int(block_shape[0])
            self.block_dim = block_dim
            self.is_setup = True
            self.setup_count += 1
        except Exception as exc:
            capacity_error = _as_amgx_capacity_error(
                exc, phase=failure_phase, cp=self.cp, pyamgx=self.pyamgx
            )
            self.close(suppress_errors=True)
            if capacity_error is not None and capacity_error is not exc:
                raise capacity_error from exc
            raise
        return time.perf_counter() - setup_start

    def replace_coefficients(self, matrix) -> float:
        """Replace fixed-pattern coefficients while retaining solver setup state."""
        if self.closed or not self.is_setup:
            raise RuntimeError(
                "PyAMGXCsrDeviceSolver must be set up before coefficient replacement"
            )
        block_dim = int(getattr(matrix, "block_size", 1))
        if tuple(matrix.shape) != self.shape or block_dim != self.block_dim:
            raise ValueError(
                "replacement matrix shape/block size does not match the cached AMGX matrix"
            )
        expected_values = int(self.mat.get_nnz()) * block_dim * block_dim
        if int(matrix.data.size) != expected_values:
            raise ValueError(
                "replacement matrix nonzero count does not match the cached AMGX pattern"
            )
        started = time.perf_counter()
        try:
            self.mat.replace_coefficients(matrix.data)
            self.cp.cuda.get_current_stream().synchronize()
        except Exception as exc:
            capacity_error = _as_amgx_capacity_error(
                exc, phase="coefficient replacement", cp=self.cp, pyamgx=self.pyamgx
            )
            self.close(suppress_errors=True)
            if capacity_error is not None and capacity_error is not exc:
                raise capacity_error from exc
            raise
        elapsed = time.perf_counter() - started
        self.last_coefficients_replace_elapsed_seconds = elapsed
        self.last_matrix_upload_elapsed_seconds = elapsed
        self.last_solver_setup_elapsed_seconds = 0.0
        self.coefficients_replace_count += 1
        return elapsed

    def solve(self, rhs, *, initial_guess=None):
        """Solve the configured AMGX system for one device RHS."""
        if self.closed or not self.is_setup:
            raise RuntimeError("PyAMGXCsrDeviceSolver must be set up before solve()")
        if tuple(rhs.shape) != (self.size,):
            raise ValueError(f"rhs must have shape ({self.size},); got {rhs.shape}")
        info = {"amgx_status": "unknown", "amgx_iterations": None, "residual_history": ()}
        if self.verbose_level == 2 or self.verbose_level >= 4:
            print(format_amgx_configuration(self.config_dict), flush=True)
        solve_start = time.perf_counter()
        failure_phase = "solution allocation"
        try:
            if initial_guess is None:
                x = self.cp.zeros_like(rhs)
                zero_initial_guess = True
            else:
                x = self.cp.asarray(initial_guess, dtype=REAL_DTYPE).copy()
                if tuple(x.shape) != tuple(rhs.shape):
                    raise ValueError(f"initial_guess must have shape {rhs.shape}; got {x.shape}")
                zero_initial_guess = False
            failure_phase = "vector upload"
            audit_arrays('amgx-vector-upload', rhs, x)
            self.vec_b.upload_raw(rhs.data.ptr, self.block_rows, self.block_dim)
            self.vec_x.upload_raw(x.data.ptr, self.block_rows, self.block_dim)
            failure_phase = "solver iteration"
            self.solver.solve(self.vec_b, self.vec_x, zero_initial_guess=zero_initial_guess)
            failure_phase = "solution download"
            self.vec_x.download_raw(x.data.ptr)
            self.cp.cuda.get_current_stream().synchronize()
        except Exception as exc:
            capacity_error = _as_amgx_capacity_error(
                exc, phase=failure_phase, cp=self.cp, pyamgx=self.pyamgx
            )
            if (
                failure_phase == "solution allocation"
                and capacity_error is None
                and isinstance(exc, (TypeError, ValueError))
            ):
                # Malformed guesses do not invalidate a healthy hierarchy.
                raise
            self.close(suppress_errors=True)
            if capacity_error is not None and capacity_error is not exc:
                raise capacity_error from exc
            raise
        solve_elapsed = time.perf_counter() - solve_start
        try:
            info["amgx_status"] = str(self.solver.status)
        except Exception:
            pass
        try:
            info["amgx_iterations"] = int(self.solver.iterations_number)
        except Exception:
            pass
        store_residual_history = bool(
            self.config_dict.get("solver", {}).get("store_res_history", 0)
        )
        if store_residual_history and info["amgx_iterations"] is not None:
            history = []
            first = max(0, info["amgx_iterations"] - 63)
            for iteration in range(first, info["amgx_iterations"] + 1):
                try:
                    history.append(float(self.solver.get_residual(iteration)))
                except Exception:
                    history = []
                    break
            info["residual_history"] = tuple(history)
        info["amgx_setup_elapsed_seconds"] = 0.0
        info["amgx_solve_elapsed_seconds"] = solve_elapsed
        return x, info

    def close(self, *, suppress_errors: bool = False) -> None:
        """Destroy owned AMGX objects and release the shared resource handle."""
        if self.closed:
            return
        first_error = None
        try:
            destroy_amgx_objects((self.solver, self.vec_x, self.vec_b, self.mat, self.cfg))
        except Exception as exc:
            first_error = exc
        if self._shared_resources_acquired:
            try:
                _AMGX_SHARED_RESOURCES.release()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
            self._shared_resources_acquired = False
        self.solver = self.mat = self.vec_x = self.vec_b = self.rsrc = self.cfg = None
        self._hdgfem_fixed_operator = None
        self.is_setup = False
        self.closed = True
        if first_error is not None and not suppress_errors:
            raise first_error

    def __del__(self):
        """Best-effort cleanup for an unclosed AMGX solver instance."""
        try:
            self.close(suppress_errors=True)
        except Exception:
            pass


def _pyamgx_solve_csr_device(
        matrix,
        rhs,
        *,
        initial_guess=None,
        config=None,
        tolerance: float = 1e-13,
        maxiter: int | None = None,
        verbose: bool | int = 0,
):
    """Solve a device CSR system with PyAMGX without staging through host CSR."""
    solver = PyAMGXCsrDeviceSolver(config=config, tolerance=tolerance, maxiter=maxiter, verbose=verbose)
    try:
        setup_elapsed = solver.setup(matrix)
        x, info = solver.solve(rhs, initial_guess=initial_guess)
    finally:
        solver.close()
    info["amgx_setup_elapsed_seconds"] = setup_elapsed
    info["amgx_matrix_upload_elapsed_seconds"] = solver.last_matrix_upload_elapsed_seconds
    info["amgx_solver_setup_elapsed_seconds"] = solver.last_solver_setup_elapsed_seconds
    return x, info


def _normalize_device_scale_mode(value) -> str:
    """Normalize public boolean/string scaling controls for device solves."""
    if isinstance(value, (bool, np.bool_)):
        return "left" if bool(value) else "none"
    normalized = str(value).replace("_", "-").lower()
    aliases = {"on": "left", "off": "none", "true": "left", "false": "none"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"none", "left", "symmetric"}:
        raise ValueError("scale_system must be bool or one of 'none', 'left', 'symmetric'")
    return normalized


def _solve_reduced_system_amgx_device_once(
    assembly,
    *,
    config=None,
    tolerance: float = 1e-13,
    check_rtol: float | None = None,
    solver_check_rtol: float | None = None,
    atol: float = 0.0,
    maxiter: int | None = None,
    initial_guess=None,
    reusable_solver: PyAMGXCsrDeviceSolver | None = None,
    scale_system: bool | str = True,
    scalarize_bsr: bool = False,
    replace_reusable_coefficients: bool = False,
    raise_on_nonconvergence: bool = True,
    materialize_host_solution: bool = True,
    verbose: bool | int = 0,
):
    """Solve a CUDA-resident reduced trace system with device CSR and PyAMGX."""
    cp = require_cupy()
    sparse = require_cupyx_sparse()
    total_start = time.perf_counter()
    system_size = int(assembly.rhs.size)
    solve_tolerance = float(tolerance)
    result_check_rtol = solve_tolerance if check_rtol is None else float(check_rtol)
    solver_result_check_rtol = (
        result_check_rtol
        if solver_check_rtol is None
        else float(solver_check_rtol)
    )
    if not np.isfinite(solve_tolerance) or solve_tolerance < 0.0:
        raise ValueError(f"tolerance must be finite and non-negative, got {tolerance}")
    if not np.isfinite(result_check_rtol) or result_check_rtol < 0.0:
        raise ValueError(f"check_rtol must be finite and non-negative, got {check_rtol}")
    if not np.isfinite(solver_result_check_rtol) or solver_result_check_rtol < 0.0:
        raise ValueError(
            "solver_check_rtol must be finite and non-negative, got "
            f"{solver_check_rtol}"
        )
    if not np.isfinite(atol) or float(atol) < 0.0:
        raise ValueError(f"atol must be finite and non-negative, got {atol}")
    if maxiter is not None and int(maxiter) <= 0:
        raise ValueError(f"maxiter must be positive when provided, got {maxiter}")
    if not bool(cp.all(cp.isfinite(assembly.data)).get()):
        raise ValueError("device matrix contains non-finite values")
    if not bool(cp.all(cp.isfinite(assembly.rhs)).get()):
        raise ValueError("device rhs contains non-finite values")
    if initial_guess is not None and not bool(cp.all(cp.isfinite(cp.asarray(initial_guess))).get()):
        raise ValueError("initial_guess contains non-finite values")

    matrix_start = time.perf_counter()
    matrix_format = getattr(assembly, "matrix_format", "coo")
    if matrix_format == "csr":
        if assembly.indptr is None or assembly.indices is None:
            raise RuntimeError("CSR assembly is missing indptr/indices")
        matrix = _DeviceCsrMatrixView(
            data=assembly.data,
            indices=assembly.indices.astype(cp.int32, copy=False),
            indptr=assembly.indptr.astype(cp.int32, copy=False),
            shape=(system_size, system_size),
        )
    elif matrix_format == "bsr":
        if assembly.indptr is None or assembly.indices is None:
            raise RuntimeError("BSR assembly is missing block indptr/indices")
        if assembly.data.ndim != 3 or assembly.data.shape[1] != assembly.data.shape[2]:
            raise RuntimeError(
                f"BSR data must have shape (nnzb, block_size, block_size); got {assembly.data.shape}"
            )
        block_dim = int(assembly.data.shape[1])
        if system_size % block_dim:
            raise RuntimeError(
                f"system size {system_size} is not divisible by BSR block size {block_dim}"
            )
        matrix = _DeviceBsrMatrixView(
            data=assembly.data,
            indices=assembly.indices.astype(cp.int32, copy=False),
            indptr=assembly.indptr.astype(cp.int32, copy=False),
            shape=(system_size, system_size),
            block_size=block_dim,
        )
    else:
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
    bsr_scalarized = bool(scalarize_bsr and isinstance(matrix, _DeviceBsrMatrixView))
    if bsr_scalarized:
        matrix = _scalarize_device_bsr_matrix(matrix, sparse, cp)
    cp.cuda.get_current_stream().synchronize()
    matrix_elapsed = time.perf_counter() - matrix_start

    physical_rhs = assembly.rhs
    scale_mode = _normalize_device_scale_mode(scale_system)
    if isinstance(matrix, _DeviceBsrMatrixView) and scale_mode == "symmetric":
        raise ValueError(
            "device BSR solves do not support symmetric scaling; "
            "use scale_system='left' or False"
        )
    row_diagonal = None
    inverse_sqrt_diagonal = None
    scale_start = time.perf_counter()
    if scale_mode == "left":
        solve_matrix = matrix
        solve_rhs = physical_rhs.copy()
        if isinstance(solve_matrix, _DeviceBsrMatrixView):
            row_diagonal = _diagonal_scale_bsr_rows_in_place(solve_matrix, solve_rhs)
        else:
            row_diagonal = _diagonal_scale_csr_rows_in_place(solve_matrix, solve_rhs)
        solve_initial_guess = initial_guess
    elif scale_mode == "symmetric":
        solve_matrix = matrix
        solve_rhs = physical_rhs.copy()
        inverse_sqrt_diagonal = symmetric_scale_cupy_csr_in_place(solve_matrix, solve_rhs)
        solve_initial_guess = (
            None if initial_guess is None else cp.asarray(initial_guess) / inverse_sqrt_diagonal
        )
    else:
        solve_matrix = matrix
        solve_rhs = physical_rhs
        solve_initial_guess = initial_guess
    cp.cuda.get_current_stream().synchronize()
    scale_elapsed = time.perf_counter() - scale_start
    matrix_is_scaled = row_diagonal is not None or inverse_sqrt_diagonal is not None

    def restore_scaled_matrix() -> float:
        """Restore the physical CSR coefficients after an in-place scaled solve."""
        nonlocal matrix_is_scaled
        if not matrix_is_scaled:
            return 0.0
        restore_start = time.perf_counter()
        if isinstance(solve_matrix, _DeviceBsrMatrixView):
            _restore_left_scaled_bsr_rows_in_place(solve_matrix, row_diagonal)
        else:
            _restore_scaled_csr_rows_in_place(
                solve_matrix,
                row_diagonal=row_diagonal,
                inverse_sqrt_diagonal=inverse_sqrt_diagonal,
            )
        cp.cuda.get_current_stream().synchronize()
        matrix_is_scaled = False
        return time.perf_counter() - restore_start

    try:
        amgx_call_start = time.perf_counter()
        if reusable_solver is None:
            x_cp, amgx_info = _pyamgx_solve_csr_device(
                solve_matrix,
                solve_rhs,
                initial_guess=solve_initial_guess,
                config=config,
                tolerance=solve_tolerance,
                maxiter=maxiter,
                verbose=verbose,
            )
        else:
            setup_elapsed = 0.0
            matrix_upload_elapsed = 0.0
            solver_setup_elapsed = 0.0
            preconditioner_reused = bool(reusable_solver.is_setup)
            if not reusable_solver.is_setup:
                setup_elapsed = reusable_solver.setup(solve_matrix)
                matrix_upload_elapsed = reusable_solver.last_matrix_upload_elapsed_seconds
                solver_setup_elapsed = reusable_solver.last_solver_setup_elapsed_seconds
            elif replace_reusable_coefficients:
                matrix_upload_elapsed = reusable_solver.replace_coefficients(solve_matrix)
                preconditioner_reused = True
            x_cp, amgx_info = reusable_solver.solve(solve_rhs, initial_guess=solve_initial_guess)
            amgx_info["amgx_setup_elapsed_seconds"] = setup_elapsed
            amgx_info["amgx_matrix_upload_elapsed_seconds"] = matrix_upload_elapsed
            amgx_info["amgx_solver_setup_elapsed_seconds"] = solver_setup_elapsed
            amgx_info["amgx_preconditioner_reused"] = preconditioner_reused
        audit_arrays('amgx-solution', x_cp)
        solver_x_cp = x_cp
        if inverse_sqrt_diagonal is not None:
            x_cp = inverse_sqrt_diagonal * solver_x_cp
        amgx_call_elapsed = time.perf_counter() - amgx_call_start
    except BaseException as exc:
        try:
            restore_scaled_matrix()
        except Exception as restore_exc:
            if not isinstance(exc, LinearSolveCapacityError):
                raise
            # A depleted or failed CUDA runtime can also reject restoration.
            # Preserve the terminal native phase and its pre-cleanup counters.
            exc.matrix_restore_error = f"{type(restore_exc).__name__}: {restore_exc}"
        raise

    try:
        finite_start = time.perf_counter()
        solution_is_finite = bool(cp.all(cp.isfinite(x_cp)).get())
        finite_elapsed = time.perf_counter() - finite_start

        solver_residual_start = time.perf_counter()
        solver_residual = (
            _device_compressed_matvec(solve_matrix, solver_x_cp, sparse, cp)
            - solve_rhs
        )
        solver_residual_norm, solver_rhs_norm, solver_relative, solver_target = _residual_stats_cp(
            solver_residual,
            solve_rhs,
            rtol=solver_result_check_rtol,
            atol=atol,
        )
        solver_residual_elapsed = time.perf_counter() - solver_residual_start

        # Check b-A*x after restoring the physical coefficients. Undoing the
        # residual scaling algebraically can hide cancellation/roundoff errors.
        unscale_elapsed = restore_scaled_matrix()
        physical_residual_start = time.perf_counter()
        if scale_mode == "none":
            # The solver and physical systems are identical. Reuse the SpMV
            # and norms, but retain the independent physical acceptance target.
            physical_residual_norm = solver_residual_norm
            physical_rhs_norm = solver_rhs_norm
            physical_relative = solver_relative
            physical_target = max(float(result_check_rtol) * physical_rhs_norm, float(atol))
        else:
            physical_residual = _device_compressed_matvec(matrix, x_cp, sparse, cp) - physical_rhs
            physical_residual_norm, physical_rhs_norm, physical_relative, physical_target = _residual_stats_cp(
                physical_residual,
                physical_rhs,
                rtol=result_check_rtol,
                atol=atol,
            )
        physical_residual_elapsed = time.perf_counter() - physical_residual_start
        validation_elapsed = finite_elapsed + solver_residual_elapsed + physical_residual_elapsed
    except BaseException:
        restore_scaled_matrix()
        raise
    native_status = str(amgx_info.get("amgx_status", "unknown"))
    normalized_status = native_status.lower().replace("-", "_").replace(" ", "_")
    backend_success = normalized_status == "unknown" or not any(
        marker in normalized_status
        for marker in ("fail", "diverg", "not_converged", "notconverged")
    )
    total_elapsed = time.perf_counter() - total_start
    verbose_level = 1 if isinstance(verbose, bool) and verbose else (0 if not verbose else int(verbose))
    if verbose_level == 2 or verbose_level >= 4:
        print("  PyAMGX device solve timings:", flush=True)
        print(f"    matrix view: {matrix_elapsed:.5f}s", flush=True)
        print(f"    row scaling: {scale_elapsed:.5f}s", flush=True)
        print(f"    matrix unscale: {unscale_elapsed:.5f}s", flush=True)
        print(f"    setup: {amgx_info['amgx_setup_elapsed_seconds']:.5f}s", flush=True)
        print(
            f"      matrix upload: {amgx_info['amgx_matrix_upload_elapsed_seconds']:.5f}s",
            flush=True,
        )
        print(f"      solver setup: {amgx_info['amgx_solver_setup_elapsed_seconds']:.5f}s", flush=True)
        print(f"    iterate: {amgx_info['amgx_solve_elapsed_seconds']:.5f}s", flush=True)
        print(f"    amgx call total: {amgx_call_elapsed:.5f}s", flush=True)
        print(f"    validation: {validation_elapsed:.5f}s", flush=True)
        print(f"    solver relative residual: {solver_relative:.3e}", flush=True)
    elif verbose_level:
        print(
            f"  PyAMGX: matrix={matrix_elapsed:.5f}s scale={scale_elapsed:.5f}s "
            f"setup={amgx_info['amgx_setup_elapsed_seconds']:.5f}s "
            f"solve={amgx_info['amgx_solve_elapsed_seconds']:.5f}s rel={solver_relative:.3e}",
            flush=True,
        )
    result = SolveResult(
        x=cp.asnumpy(x_cp) if materialize_host_solution else None,
        residual_norm=solver_residual_norm,
        info=0 if backend_success else 1,
        preconditioner=None,
        total_elapsed_seconds=total_elapsed,
        scale_elapsed_seconds=scale_elapsed,
        preconditioner_elapsed_seconds=matrix_elapsed + amgx_info["amgx_setup_elapsed_seconds"],
        solve_elapsed_seconds=amgx_info["amgx_solve_elapsed_seconds"],
        iteration_count=amgx_info.get("amgx_iterations"),
        rhs_norm=solver_rhs_norm,
        relative_residual_norm=solver_relative,
        residual_target=solver_target,
        solver_residual_norm=solver_residual_norm,
        solver_rhs_norm=solver_rhs_norm,
        solver_relative_residual_norm=solver_relative,
        solver_residual_target=solver_target,
        physical_residual_norm=physical_residual_norm,
        physical_rhs_norm=physical_rhs_norm,
        physical_relative_residual_norm=physical_relative,
        physical_residual_target=physical_target,
        rtol=result_check_rtol,
        atol=atol,
    )
    result.cupyx_solver = "pyamgx-device"
    result.device_scale_mode = scale_mode
    result.amgx_csr_elapsed_seconds = matrix_elapsed
    result.amgx_bsr_scalarized = bsr_scalarized
    result.amgx_preconditioner_reused = bool(
        amgx_info.get("amgx_preconditioner_reused", False)
    )
    result.amgx_matrix_unscale_elapsed_seconds = unscale_elapsed
    result.amgx_setup_elapsed_seconds = amgx_info["amgx_setup_elapsed_seconds"]
    result.amgx_matrix_upload_elapsed_seconds = amgx_info["amgx_matrix_upload_elapsed_seconds"]
    result.amgx_solver_setup_elapsed_seconds = amgx_info["amgx_solver_setup_elapsed_seconds"]
    result.amgx_solve_elapsed_seconds = amgx_info["amgx_solve_elapsed_seconds"]
    result.amgx_call_elapsed_seconds = amgx_call_elapsed
    result.amgx_overhead_elapsed_seconds = max(
        0.0,
        amgx_call_elapsed - amgx_info["amgx_setup_elapsed_seconds"] - amgx_info["amgx_solve_elapsed_seconds"],
    )
    result.solve_finite_check_elapsed_seconds = finite_elapsed
    result.solve_solver_residual_elapsed_seconds = solver_residual_elapsed
    result.solve_physical_residual_elapsed_seconds = physical_residual_elapsed
    result.solve_validation_elapsed_seconds = validation_elapsed
    result.solve_accounted_elapsed_seconds = (
        matrix_elapsed + scale_elapsed + amgx_call_elapsed + validation_elapsed + unscale_elapsed
    )
    result.solve_unaccounted_elapsed_seconds = max(0.0, total_elapsed - result.solve_accounted_elapsed_seconds)
    result.solve_global_overhead_elapsed_seconds = result.solve_unaccounted_elapsed_seconds
    result = finalize_solve_result(
        result,
        backend="pyamgx-device",
        backend_info=native_status,
        backend_success=backend_success,
        solution_is_finite=solution_is_finite,
        residual_history=amgx_info.get("residual_history"),
        raise_on_nonconvergence=raise_on_nonconvergence,
    )
    return result, x_cp


def _solve_reduced_system_cusolver_qr_device_once(
    assembly,
    *,
    tolerance: float = 1e-13,
    check_rtol: float | None = None,
    atol: float = 0.0,
    materialize_host_solution: bool = True,
):
    """Last-resort device sparse QR with an explicit physical residual check.

    Uses CuPy's cuSOLVER sparse QR binding, not a host sparse solve. BSR is
    expanded on device. The direct solver receives its own canonical CSR copy
    and the original, unscaled system; it never reuses a failed iterate.
    """
    from cupyx.cusolver import csrlsvqr

    cp, sparse = require_cupy(), require_cupyx_sparse()
    started = time.perf_counter()
    rtol = float(tolerance if check_rtol is None else check_rtol)
    if not np.isfinite(rtol) or rtol < 0.0 or not np.isfinite(atol) or atol < 0.0:
        raise ValueError("direct residual tolerances must be finite and nonnegative")
    if not bool(cp.all(cp.isfinite(assembly.data)).get()) or not bool(cp.all(cp.isfinite(assembly.rhs)).get()):
        raise ValueError("device direct system contains non-finite values")
    matrix = _assembly_device_csr_matrix(assembly, cp, sparse)
    scalarized = isinstance(matrix, _DeviceBsrMatrixView)
    if max(int(matrix.shape[0]), int(matrix.data.size)) > np.iinfo(np.int32).max:
        raise ValueError("cuSOLVER sparse QR requires matrix size and scalar nnz to fit int32")
    csr = (_scalarize_device_bsr_matrix(matrix, sparse, cp) if scalarized
           else _as_cupyx_csr_matrix(matrix, sparse, cp).copy())
    csr.sum_duplicates()
    csr.sort_indices()
    rhs = cp.ascontiguousarray(assembly.rhs).copy()
    cp.cuda.get_current_stream().synchronize()
    setup_elapsed = time.perf_counter() - started
    solve_start = time.perf_counter()
    # tol is a pivot-singularity threshold, not the residual tolerance. Reject
    # a singularity warning as well as explicit backend failures.
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="(?i).*singular.*")
        solution = csrlsvqr(csr, rhs, tol=0.0, reorder=1)
    cp.cuda.get_current_stream().synchronize()
    solve_elapsed = time.perf_counter() - solve_start
    residual = _device_compressed_matvec(matrix, solution, sparse, cp) - assembly.rhs
    norm, rhs_norm, relative, target = _residual_stats_cp(residual, assembly.rhs, rtol=rtol, atol=atol)
    finite = bool(cp.all(cp.isfinite(solution)).get())
    result = SolveResult(
        x=cp.asnumpy(solution) if materialize_host_solution else None,
        residual_norm=norm, info=0, preconditioner=None,
        total_elapsed_seconds=time.perf_counter() - started,
        preconditioner_elapsed_seconds=setup_elapsed, solve_elapsed_seconds=solve_elapsed,
        rhs_norm=rhs_norm, relative_residual_norm=relative, residual_target=target,
        solver_residual_norm=norm, solver_rhs_norm=rhs_norm,
        solver_relative_residual_norm=relative, solver_residual_target=target,
        physical_residual_norm=norm, physical_rhs_norm=rhs_norm,
        physical_relative_residual_norm=relative, physical_residual_target=target,
        rtol=rtol, atol=atol,
    )
    result.device_scale_mode = "none"
    result.amgx_bsr_scalarized = scalarized
    result.cupyx_solver = "cusolver-qr-device"
    return finalize_solve_result(
        result, backend="cusolver-qr-device", backend_info=0,
        backend_success=True, solution_is_finite=finite, raise_on_nonconvergence=False,
    ), solution


def _device_transport_matrix_diagnostics(assembly) -> dict[str, float | int]:
    """Cheap row-scale diagnostics on failure; these are not condition estimates."""
    cp, sparse = require_cupy(), require_cupyx_sparse()
    matrix = _assembly_device_csr_matrix(assembly, cp, sparse)
    if isinstance(matrix, (_DeviceBsrMatrixView, _DeviceCsrMatrixView)):
        absolute_matrix = replace(matrix, data=cp.abs(matrix.data))
    else:
        absolute_matrix = matrix.copy()
        absolute_matrix.data = cp.abs(absolute_matrix.data)
    row_l1 = _device_compressed_matvec(
        absolute_matrix, cp.ones(matrix.shape[1], dtype=REAL_DTYPE), sparse, cp,
    )
    zero_rows = cp.count_nonzero(row_l1 == 0)
    values = cp.asnumpy(cp.stack((cp.min(row_l1), cp.max(row_l1), zero_rows)))
    return {"matrix_size": int(matrix.shape[0]), "matrix_scalar_stored_entries": int(matrix.data.size),
            "matrix_row_l1_min": float(values[0]), "matrix_row_l1_max": float(values[1]),
            "matrix_zero_rows": int(values[2])}


def solve_reduced_system_amgx_device(
    assembly,
    *,
    config=None,
    retry_attempts=None,
    tolerance: float = 1e-13,
    check_rtol: float | None = None,
    atol: float = 0.0,
    maxiter: int | None = None,
    initial_guess=None,
    reusable_solver: PyAMGXCsrDeviceSolver | None = None,
    retry_solver_cache: dict[Any, PyAMGXCsrDeviceSolver] | None = None,
    retry_seed_solution=None,
    retry_seed_label: str | None = None,
    cache_fixed_operator: bool = False,
    reuse_primary_preconditioner: bool = False,
    scale_system: bool | str = True,
    raise_on_nonconvergence: bool = True,
    materialize_host_solution: bool = True,
    verbose: bool | int = 0,
    failure_snapshot=None,
):
    """Solve one assembled device system, optionally retrying without reassembly.

    ``retry_seed_solution`` initializes the wrapper's single best-candidate
    slot from a failed upstream solver.  Its physical residual is evaluated
    here, and no vector history is retained.

    With ``reusable_solver`` and ``reuse_primary_preconditioner=True``, an
    already set-up solver receives the new matrix through
    ``replace_coefficients`` and keeps its previous setup (a possibly stale
    preconditioner); callers force a fresh setup by clearing ``is_setup``.

    After every attempt fails, the raised ``LinearSolveConvergenceError``
    carries ``save_transport_snapshot(path)``, which writes the failed system
    through ``failure_snapshot`` (default
    :func:`hdgfem.linalg.failure_snapshot.save_system_snapshot`).
    """
    def fixed_solver(key, solver_config):
        """Keep native allocations; refresh numeric setup only for a new matrix."""
        if retry_solver_cache is None:
            raise ValueError("fixed-operator AMGX reuse requires an owning solver cache")
        key = ("fixed-operator", key)
        active = retry_solver_cache.get(key)
        if active is None or active.closed:
            active = PyAMGXCsrDeviceSolver(config=solver_config, tolerance=tolerance,
                                           maxiter=maxiter, verbose=verbose, reusable=True)
            retry_solver_cache[key] = active
        if getattr(active, "_hdgfem_fixed_operator", None) is not assembly.data:
            active.is_setup = False
            active._hdgfem_fixed_operator = assembly.data
        return active

    if cache_fixed_operator:
        if reusable_solver is not None:
            raise ValueError("supply either reusable_solver or cache_fixed_operator")
        reusable_solver = fixed_solver("primary", config)
    configured_retries = tuple(retry_attempts or ())
    if retry_seed_solution is not None and not raise_on_nonconvergence:
        raise ValueError("retry_seed_solution requires raise_on_nonconvergence=True")
    max_attempts = 8
    if 1 + len(configured_retries) > max_attempts:
        raise ValueError(f"AMGX solve supports at most {max_attempts} bounded attempts")
    attempts = [
        {
            "label": "primary-stage-guess" if initial_guess is not None else "primary-zero",
            "backend": "amgx",
            "config": config,
            "initial_guess": initial_guess,
            "scale_system": _normalize_device_scale_mode(scale_system),
            "reusable_solver": reusable_solver,
            "use_best_solution": False,
            "residual_correction": False,
            "scalarize_bsr": False,
            "reuse_preconditioner": False,
            "solver_cache_key": None,
        }
    ]
    for index, retry in enumerate(configured_retries, start=1):
        backend = str(retry.get("backend", "amgx"))
        if backend not in {"amgx", "cusolver-qr"}:
            raise ValueError(f"unsupported device retry backend {backend!r}")
        if backend == "cusolver-qr" and (
            index != len(configured_retries)
            or retry.get("residual_correction", False)
            or retry.get("reuse_preconditioner", False)
            or _normalize_device_scale_mode(retry.get("scale_system", False)) != "none"
        ):
            raise ValueError("cusolver-qr must be the final, unscaled direct retry without preconditioner reuse or correction")
        use_initial_guess = bool(retry.get("use_initial_guess", True))
        reuse_primary_solver = bool(retry.get("reuse_primary_solver", False))
        if reuse_primary_solver and (
            backend != "amgx"
            or retry.get("scalarize_bsr", False)
            or retry.get("reuse_preconditioner", False)
            or retry.get("residual_correction", False)
        ):
            raise ValueError(
                "reuse_primary_solver requires a nonscalarized AMGX retry "
                "without correction or separate preconditioner reuse"
            )
        retry_config = retry.get("config", config)
        retry_scale_mode = _normalize_device_scale_mode(
            retry.get("scale_system", False if backend == "cusolver-qr" else scale_system)
        )
        if reuse_primary_solver and (
            retry_config != config
            or retry_scale_mode != _normalize_device_scale_mode(scale_system)
        ):
            raise ValueError(
                "reuse_primary_solver requires the primary configuration and scaling"
            )
        attempts.append(
            {
                "label": str(retry.get("label", f"retry-{index}")),
                "backend": backend,
                "config": retry_config,
                "initial_guess": initial_guess if use_initial_guess else None,
                "scale_system": retry_scale_mode,
                "reusable_solver": reusable_solver if reuse_primary_solver else None,
                "reuse_primary_solver": reuse_primary_solver,
                "use_best_solution": bool(retry.get("use_best_solution", False)),
                "residual_correction": bool(retry.get("residual_correction", False)),
                "scalarize_bsr": bool(retry.get("scalarize_bsr", False)),
                "reuse_preconditioner": bool(
                    retry.get("reuse_preconditioner", False)
                ),
                "solver_cache_key": retry.get("solver_cache_key"),
            }
        )

    if len(attempts) == 1 and retry_seed_solution is None:
        return _solve_reduced_system_amgx_device_once(
            assembly,
            config=config,
            tolerance=tolerance,
            check_rtol=check_rtol,
            solver_check_rtol=_amgx_relative_residual_check_rtol(
                config, tolerance if check_rtol is None else check_rtol
            ),
            atol=atol,
            maxiter=maxiter,
            initial_guess=initial_guess,
            reusable_solver=reusable_solver,
            scale_system=scale_system,
            replace_reusable_coefficients=bool(
                reuse_primary_preconditioner and reusable_solver is not None
                and not reusable_solver.closed and reusable_solver.is_setup),
            raise_on_nonconvergence=raise_on_nonconvergence,
            materialize_host_solution=materialize_host_solution,
            verbose=verbose,
        )

    cp = require_cupy()
    retry_wrapper_start = time.perf_counter()
    attempt_log = []
    last_result = None
    last_error = None
    best_result = None
    best_solution = None
    last_solution = None
    best_score = float("inf")
    seed_metrics = None
    verbose_level = 1 if isinstance(verbose, bool) and verbose else (0 if not verbose else int(verbose))
    if retry_seed_solution is not None:
        sparse = require_cupyx_sparse()
        seed = cp.asarray(retry_seed_solution, dtype=REAL_DTYPE)
        if seed.ndim != 1 or int(seed.size) != int(assembly.rhs.size):
            raise ValueError(
                "retry_seed_solution must have the reduced-system shape "
                f"({int(assembly.rhs.size)},)"
            )
        if bool(cp.all(cp.isfinite(seed)).get()):
            seed_matrix = _assembly_device_csr_matrix(assembly, cp, sparse)
            seed_residual = (
                _device_compressed_matvec(seed_matrix, seed, sparse, cp)
                - assembly.rhs
            )
            seed_check_rtol = (
                float(tolerance) if check_rtol is None else float(check_rtol)
            )
            residual_norm, rhs_norm, relative_residual, residual_target = (
                _residual_stats_cp(
                    seed_residual, assembly.rhs,
                    rtol=seed_check_rtol, atol=atol,
                )
            )
            seed_score = residual_norm / max(residual_target, 1.0e-300)
            if all(np.isfinite(value) for value in (
                residual_norm, rhs_norm, relative_residual, residual_target, seed_score,
            )):
                best_solution = seed.copy()
                best_score = seed_score
                seed_metrics = {
                    "label": retry_seed_label or "upstream-best",
                    "physical_residual": residual_norm,
                    "physical_rhs_norm": rhs_norm,
                    "physical_relative_residual": relative_residual,
                    "physical_target": residual_target,
                }

    def attach_seed_metrics(result) -> None:
        """Copy the retry seed's physical residual metrics onto ``result`` when both exist."""
        if result is None or seed_metrics is None:
            return
        result.amgx_retry_seed_label = seed_metrics["label"]
        result.amgx_retry_seed_physical_residual = seed_metrics["physical_residual"]
        result.amgx_retry_seed_physical_rhs_norm = seed_metrics["physical_rhs_norm"]
        result.amgx_retry_seed_physical_relative_residual = seed_metrics[
            "physical_relative_residual"
        ]
        result.amgx_retry_seed_physical_target = seed_metrics["physical_target"]

    try:
        for index, attempt in enumerate(attempts, start=1):
            try:
                attempt_initial_guess = (
                    best_solution
                    if attempt["use_best_solution"] and best_solution is not None
                    else attempt["initial_guess"]
                )
                solve_assembly = assembly
                base_solution = None
                residual_matrix = None
                if attempt["residual_correction"] and best_solution is not None:
                    sparse = require_cupyx_sparse()
                    residual_matrix = _assembly_device_csr_matrix(assembly, cp, sparse)
                    correction_rhs = assembly.rhs - _device_compressed_matvec(
                        residual_matrix, best_solution, sparse, cp
                    )
                    solve_assembly = replace(assembly, rhs=correction_rhs)
                    base_solution = best_solution
                    attempt_initial_guess = None

                attempt_solver = attempt["reusable_solver"]
                if (
                    attempt.get("reuse_primary_solver", False)
                    and attempt_solver is not None
                    and attempt_solver.closed
                ):
                    # Setup/iteration exceptions close the primary's AMGX
                    # objects. Retry with a fresh ephemeral solver in that case.
                    attempt_solver = None
                replace_coefficients = False
                if (index == 1 and reuse_primary_preconditioner and attempt_solver is not None
                        and not attempt_solver.closed):
                    replace_coefficients = bool(attempt_solver.is_setup)
                if cache_fixed_operator and attempt["backend"] == "amgx":
                    attempt_solver = fixed_solver("primary" if index == 1 else index-1, attempt["config"])
                elif (
                    attempt["reuse_preconditioner"]
                    and retry_solver_cache is not None
                ):
                    cache_key = attempt["solver_cache_key"]
                    if cache_key is None:
                        cache_key = attempt["label"]
                    attempt_solver = retry_solver_cache.get(cache_key)
                    if attempt_solver is None or attempt_solver.closed:
                        attempt_solver = PyAMGXCsrDeviceSolver(
                            config=attempt["config"],
                            tolerance=tolerance,
                            maxiter=maxiter,
                            verbose=verbose,
                            reusable=True,
                        )
                        retry_solver_cache[cache_key] = attempt_solver
                    replace_coefficients = bool(attempt_solver.is_setup)

                if attempt["backend"] == "cusolver-qr":
                    result, solution = _solve_reduced_system_cusolver_qr_device_once(
                        assembly, tolerance=tolerance, check_rtol=check_rtol, atol=atol,
                        materialize_host_solution=materialize_host_solution,
                    )
                else:
                    result, solution = _solve_reduced_system_amgx_device_once(
                        solve_assembly,
                        config=attempt["config"],
                        tolerance=tolerance,
                        check_rtol=check_rtol,
                        solver_check_rtol=_amgx_relative_residual_check_rtol(
                            attempt["config"],
                            tolerance if check_rtol is None else check_rtol,
                        ),
                        atol=atol,
                        maxiter=maxiter,
                        initial_guess=attempt_initial_guess,
                        reusable_solver=attempt_solver,
                        scale_system=attempt["scale_system"],
                        scalarize_bsr=attempt["scalarize_bsr"],
                        replace_reusable_coefficients=replace_coefficients,
                        raise_on_nonconvergence=False,
                        materialize_host_solution=materialize_host_solution,
                        verbose=verbose,
                    )
                if base_solution is not None:
                    result.amgx_correction_info = int(result.info)
                    result.amgx_correction_relative_residual_norm = result.solver_relative_residual_norm
                    solution = base_solution + solution
                    combined_residual = (
                        _device_compressed_matvec(residual_matrix, solution, sparse, cp)
                        - assembly.rhs
                    )
                    result_check_rtol = float(tolerance) if check_rtol is None else float(check_rtol)
                    residual_norm, rhs_norm, relative_residual, residual_target = _residual_stats_cp(
                        combined_residual,
                        assembly.rhs,
                        rtol=result_check_rtol,
                        atol=atol,
                    )
                    solution_is_finite = bool(cp.all(cp.isfinite(solution)).get())
                    result.x = cp.asnumpy(solution) if materialize_host_solution else None
                    result.residual_norm = residual_norm
                    result.rhs_norm = rhs_norm
                    result.relative_residual_norm = relative_residual
                    result.residual_target = residual_target
                    result.solver_residual_norm = residual_norm
                    result.solver_rhs_norm = rhs_norm
                    result.solver_relative_residual_norm = relative_residual
                    result.solver_residual_target = residual_target
                    result.physical_residual_norm = residual_norm
                    result.physical_rhs_norm = rhs_norm
                    result.physical_relative_residual_norm = relative_residual
                    result.physical_residual_target = residual_target
                    result.amgx_residual_correction = True
                    result = finalize_solve_result(
                        result,
                        backend=result.backend or "pyamgx-device",
                        backend_info=result.backend_info,
                        backend_success=result.failure_reason not in {"backend-nonconvergence", "backend-divergence"},
                        solution_is_finite=solution_is_finite,
                        residual_history=result.residual_history,
                        raise_on_nonconvergence=False,
                    )

                finite = bool(cp.all(cp.isfinite(solution)).get())
                success = finite and result.converged
                score = float(result.physical_residual_norm) / max(
                    float(result.physical_residual_target), 1.0e-300
                )
                if finite and np.isfinite(score) and score < best_score:
                    best_result = result
                    best_solution = solution.copy()
                    best_score = score
                attempt_log.append(
                    {
                        "attempt": index,
                        "label": attempt["label"],
                        "success": success,
                        "finite": finite,
                        "scale_system": attempt["scale_system"],
                        "used_best_solution": bool(
                            attempt["use_best_solution"] and attempt_initial_guess is not None
                        ),
                        "residual_correction": base_solution is not None,
                        "scalarized_bsr": bool(
                            getattr(result, "amgx_bsr_scalarized", False)
                        ),
                        "preconditioner_reused": bool(
                            getattr(result, "amgx_preconditioner_reused", False)
                        ),
                        "primary_solver_reused": bool(
                            attempt.get("reuse_primary_solver", False)
                            and attempt_solver is not None
                        ),
                        "status": result.status,
                        "failure_reason": result.failure_reason,
                        "backend_info": result.backend_info,
                        "iterations": result.iteration_count,
                        "relative_residual": result.solver_relative_residual_norm,
                        "residual": result.solver_residual_norm,
                        "target": result.solver_residual_target,
                        "backend": result.backend,
                        "physical_relative_residual": result.physical_relative_residual_norm,
                        "physical_residual": result.physical_residual_norm,
                        "physical_target": result.physical_residual_target,
                    }
                )
                last_result = result
                last_solution = solution
                if verbose_level and len(attempts) > 1:
                    print(
                        f"  AMGX attempt {index}/{len(attempts)} {attempt['label']}: "
                        f"{'accepted' if success else 'rejected'} "
                        f"rel={result.solver_relative_residual_norm:.3e} "
                        f"true_rel={result.physical_relative_residual_norm:.3e} "
                        f"status={result.status} iterations={result.iteration_count}",
                        flush=True,
                    )
                if success:
                    result.amgx_attempts = tuple(attempt_log)
                    result.amgx_attempt_count = index
                    attach_seed_metrics(result)
                    return result, solution
            except Exception as exc:
                capacity_error = _as_amgx_capacity_error(
                    exc, phase="AMGX call", cp=cp, pyamgx=None
                )
                entry = {
                    "attempt": index,
                    "label": attempt["label"],
                    "success": False,
                    "finite": False,
                    "scale_system": attempt["scale_system"],
                    "residual_correction": attempt["residual_correction"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if capacity_error is not None:
                    entry["terminal_capacity_failure"] = True
                    entry["phase"] = capacity_error.phase
                    attempt_log.append(entry)
                    capacity_error.amgx_attempts = tuple(attempt_log)
                    capacity_error.amgx_attempt_count = index
                    if verbose_level:
                        print(
                            f"  AMGX attempt {index}/{len(attempts)} {attempt['label']}: "
                            f"terminal capacity failure ({capacity_error})",
                            flush=True,
                        )
                    raise capacity_error
                last_error = exc
                attempt_log.append(entry)
                if verbose_level:
                    print(
                        f"  AMGX attempt {index}/{len(attempts)} {attempt['label']}: "
                        f"failed ({type(exc).__name__}: {exc})",
                        flush=True,
                    )
    finally:
        retry_wrapper_elapsed = time.perf_counter() - retry_wrapper_start
        timed_result = last_result if last_result is not None else best_result
        if timed_result is not None:
            attach_seed_metrics(timed_result)
            timed_result.amgx_retry_matrix_backup_elapsed_seconds = 0.0
            timed_result.amgx_retry_matrix_restore_elapsed_seconds = 0.0
            timed_result.amgx_retry_matrix_restore_count = 0
            timed_result.amgx_retry_matrix_backup_bytes = 0
            timed_result.amgx_retry_wrapper_elapsed_seconds = retry_wrapper_elapsed
            attempt_elapsed = float(getattr(timed_result, "total_elapsed_seconds", 0.0) or 0.0)
            timed_result.amgx_retry_outer_overhead_elapsed_seconds = max(
                0.0, retry_wrapper_elapsed - attempt_elapsed
            )
        if verbose_level == 2 or verbose_level >= 4:
            print("  AMGX retry-wrapper timings:", flush=True)
            print("    matrix backup to host: disabled", flush=True)
            print(f"    wrapper total: {retry_wrapper_elapsed:.5f}s", flush=True)

    if not raise_on_nonconvergence and (best_result is not None or last_result is not None):
        selected_result = best_result or last_result
        selected_solution = best_solution if best_result is not None else last_solution
        selected_result.amgx_attempts = tuple(attempt_log)
        selected_result.amgx_attempt_count = len(attempt_log)
        return selected_result, selected_solution
    failed_result = best_result or last_result
    if failed_result is not None:
        failed_result.amgx_attempts = tuple(attempt_log)
        failed_result.amgx_attempt_count = len(attempt_log)
        attach_seed_metrics(failed_result)
    details = "; ".join(
        f"{entry['label']}: {entry.get('error') or entry.get('failure_reason') or 'residual target not met'}"
        for entry in attempt_log
    )
    message = f"pyamgx-device solve exhausted {len(attempts)} bounded attempts: {details}"
    error = LinearSolveConvergenceError(message, result=failed_result)
    error.amgx_attempts = tuple(attempt_log)
    if seed_metrics is not None:
        error.amgx_retry_seed = dict(seed_metrics)
    # Preserve the failed system for the application error handler. Successful
    # solves incur no host transfer or snapshot allocation.
    # ``failure_snapshot(path, assembly, initial_guess=, best_solution=)``
    # writes the archive; equation families pass a richer writer (transport:
    # hdgfem.transport.diagnostics.save_transport_failure_snapshot).
    from hdgfem.linalg.failure_snapshot import save_system_snapshot

    snapshot_writer = failure_snapshot or save_system_snapshot

    def save_failure_snapshot(path):
        """Write a failure snapshot of this assembly with its initial guess and best solution."""
        return snapshot_writer(
            path, assembly, initial_guess=initial_guess, best_solution=best_solution,
        )

    error.save_transport_snapshot = save_failure_snapshot
    try:
        error.matrix_diagnostics = _device_transport_matrix_diagnostics(assembly)
    except Exception as diagnostic_error:
        error.matrix_diagnostics = {"error": f"{type(diagnostic_error).__name__}: {diagnostic_error}"}
    if last_error is not None:
        raise error from last_error
    raise error
