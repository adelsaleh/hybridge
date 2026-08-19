"""Inexpensive CuPy prototype of modal face-block p-multigrid.

The implementation deliberately favors transparent numerical operations over
kernel fusion.  It validates nested Legendre p-coarsening, block-polynomial
smoothing, a reusable scalar p=0 AMG correction, and the SPD/PCG contract before
production integration.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..backends.cupy import require_cupy
from ..backends.legendre_face_bsr import (
    LegendreFaceBsrOperator,
    modal_degree_schedule,
    principal_modal_bsr_data,
)


@dataclass(frozen=True)
class FacePmgLevelDiagnostics:
    """Setup diagnostics for one p-level."""

    degree: int
    block_size: int
    lambda_max: float | None
    lambda_low: float | None
    spmv_backend: str
    spmv_fallback_reason: str | None
    smoother_backend: str


@dataclass
class _FacePmgLevel:
    degree: int
    operator: LegendreFaceBsrOperator
    diagonal: Any
    diagonal_inverse: Any
    chebyshev_weights: tuple[float, ...]
    lambda_max: float | None
    lambda_low: float | None
    smoother_backend: str

    @property
    def block_size(self) -> int:
        """Return the number of retained face modes."""
        return int(self.degree + 1)


@dataclass
class _FacePmgWorkspace:
    """Persistent vectors owned by one nonterminal p-level."""

    correction: Any
    scratch: Any
    residual: Any
    coarse_rhs: Any

    @property
    def nbytes(self) -> int:
        """Return the device storage owned by this level workspace."""
        return int(sum(
            array.nbytes
            for array in (
                self.correction, self.scratch, self.residual, self.coarse_rhs
            )
            if array is not None
        ))


@dataclass(frozen=True)
class PrototypePcgResult:
    """Result and independently evaluated residuals from a prototype Krylov solve."""

    solution: Any
    converged: bool
    iterations: int
    residual_norm: float
    rhs_norm: float
    relative_residual: float
    residual_over_initial: float
    target: float
    history: tuple[float, ...]
    elapsed_seconds: float


class AmgxScalarVcycle:
    """One reusable scalar classical-AMG application on the p=0 face graph."""

    def __init__(self, operator: LegendreFaceBsrOperator, *, config, verbose: int = 0):
        """Build one reusable AMGX hierarchy for the scalar face operator."""
        if int(operator.block_size) != 1:
            raise ValueError("the scalar AMG coarse operator must have block size one")
        from ..backends.advection_cuda import PyAMGXCsrDeviceSolver

        self.operator = operator
        self.config = copy.deepcopy(config)
        self.solver = PyAMGXCsrDeviceSolver(
            config=self.config,
            tolerance=1.0e-30,
            maxiter=1,
            verbose=verbose,
            reusable=True,
        )
        self.setup_seconds = float(self.solver.setup(operator))
        self.apply_count = 0
        self.apply_seconds = 0.0
        self.last_info = None

    def __call__(self, rhs):
        """Apply exactly one zero-initialized AMGX cycle."""
        started = time.perf_counter()
        result, info = self.solver.solve(rhs, initial_guess=None)
        self.apply_seconds += time.perf_counter() - started
        self.apply_count += 1
        self.last_info = info
        return result

    def close(self) -> None:
        """Release AMGX objects and their cached hierarchy."""
        self.solver.close()


class CupyxCgScalarSolve:
    """Diagnostic p=0 solve using CuPyX CSR CG rather than AMGX."""

    def __init__(self, operator: LegendreFaceBsrOperator, *, tolerance=1.0e-12, maxiter=1000):
        """Create the diagnostic scalar CuPyX CSR solver."""
        if int(operator.block_size) != 1:
            raise ValueError("the scalar diagnostic solver needs block size one")
        from ..backends.cupy import require_cupyx_sparse, require_cupyx_sparse_linalg

        sparse = require_cupyx_sparse()
        self.linalg = require_cupyx_sparse_linalg()
        self.matrix = sparse.csr_matrix(
            (operator.data.reshape(-1), operator.indices, operator.indptr),
            shape=operator.shape,
        )
        self.tolerance = float(tolerance)
        self.maxiter = int(maxiter)
        self.setup_seconds = 0.0
        self.apply_count = 0
        self.apply_seconds = 0.0
        self.last_info = None

    def __call__(self, rhs):
        """Solve the scalar coarse equation to the configured tolerance."""
        started = time.perf_counter()
        try:
            result, info = self.linalg.cg(
                self.matrix, rhs, x0=None, rtol=self.tolerance,
                atol=0.0, maxiter=self.maxiter,
            )
        except TypeError:
            result, info = self.linalg.cg(
                self.matrix, rhs, x0=None, tol=self.tolerance,
                maxiter=self.maxiter,
            )
        self.apply_seconds += time.perf_counter() - started
        self.apply_count += 1
        self.last_info = int(info)
        if int(info) != 0:
            raise RuntimeError(f"CuPyX p=0 CG failed with info={info}")
        return result

    def close(self) -> None:
        """Release references owned by the diagnostic solver."""
        self.matrix = None


def symmetric_scalar_amgx_config(base_config: dict | None = None) -> dict:
    """Return the legacy nodal-derived scalar-cycle ablation.

    The historical name is retained for the experimental runner, but numerical
    self-adjointness must be checked; the inherited Chebyshev preset is not
    symmetric on the production modal p=0 graph.
    """
    if base_config is None:
        amg = {
            "solver": "AMG",
            "algorithm": "CLASSICAL",
            "selector": "PMIS",
            "strength": "AHAT",
            "strength_threshold": 0.25,
            "interpolator": "D2",
            "cycle": "V",
            "smoother": {"solver": "JACOBI_L1", "max_iters": 1},
            "coarse_solver": "DENSE_LU_SOLVER",
            "max_levels": 100,
            "dense_lu_num_rows": 2048,
            "dense_lu_max_rows": 4096,
            "coarsest_sweeps": 2,
            "aggressive_levels": 1,
            "interp_max_elements": 4,
        }
    else:
        source = copy.deepcopy(base_config).get("solver", {})
        preconditioner = source.get("preconditioner")
        amg = copy.deepcopy(preconditioner if isinstance(preconditioner, dict) else source)
        amg["solver"] = "AMG"
    amg.update({
        "presweeps": 1,
        "postsweeps": 1,
        "max_iters": 1,
        "error_scaling": 0,
        "tolerance": 1.0e-30,
        "convergence": "ABSOLUTE",
        "norm": "L2",
        # PyAMGXCsrDeviceSolver records history for diagnostics, so AMGX requires monitoring.
        "monitor_residual": 1,
        "print_solve_stats": 0,
        "obtain_timings": 0,
    })
    return {"config_version": 2, "solver": amg}


def scalar_p0_amgx_config() -> dict:
    """Return a dedicated fixed-work scalar-AMG cycle for the p=0 face graph.

    This configuration deliberately does not inherit the full-order nodal trace
    preset. The p=0 operator has one scalar constant mode per face, so it uses
    one diagonal L1-Jacobi pre/post sweep and no aggressive first level.
    """
    amg = {
        "solver": "AMG",
        "algorithm": "CLASSICAL",
        "selector": "PMIS",
        "strength": "AHAT",
        "strength_threshold": 0.25,
        "interpolator": "D2",
        "cycle": "V",
        "presweeps": 1,
        "postsweeps": 1,
        "smoother": {"solver": "JACOBI_L1", "max_iters": 1},
        "coarse_solver": "DENSE_LU_SOLVER",
        "max_iters": 1,
        "max_levels": 100,
        "dense_lu_num_rows": 2048,
        "dense_lu_max_rows": 4096,
        "coarsest_sweeps": 2,
        "aggressive_levels": 0,
        "interp_max_elements": 4,
        "error_scaling": 0,
        "tolerance": 1.0e-30,
        "convergence": "ABSOLUTE",
        "norm": "L2",
        # PyAMGXCsrDeviceSolver records history for diagnostics, so AMGX
        # requires residual monitoring even for a fixed one-cycle apply.
        "monitor_residual": 1,
        "print_solve_stats": 0,
        "obtain_timings": 0,
    }
    return {"config_version": 2, "solver": amg}


def _block_action(blocks, vector, block_size: int):
    """Apply a batch of dense face blocks to a face-major vector."""
    cp = require_cupy()
    shaped = vector.reshape((-1, int(block_size), 1))
    return cp.matmul(blocks, shaped).reshape(-1)


def _estimate_preconditioned_lambda_max(
        operator,
        diagonal,
        diagonal_inverse,
        *,
        iterations: int,
        safety_factor: float,
) -> float:
    """Estimate the largest generalized eigenvalue of ``A x=lambda M x``."""
    cp = require_cupy()
    size = int(operator.shape[0])
    block_size = int(operator.block_size)
    phase = cp.arange(size, dtype=cp.float64)
    vector = cp.sin(0.731 * (phase + 1.0)) + cp.cos(0.193 * (phase + 0.5))
    vector /= cp.linalg.norm(vector)
    for _ in range(max(2, int(iterations))):
        applied = operator.matvec(vector)
        vector = _block_action(diagonal_inverse, applied, block_size)
        norm = cp.linalg.norm(vector)
        vector /= norm
    applied = operator.matvec(vector)
    mass_applied = _block_action(diagonal, vector, block_size)
    estimate = cp.real(cp.vdot(vector, applied) / cp.vdot(vector, mass_applied))
    value = float(estimate.get()) * float(safety_factor)
    if not np.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"invalid block-preconditioned spectral estimate {value}")
    return value


def _chebyshev_richardson_weights(order: int, low: float, high: float) -> tuple[float, ...]:
    """Return reciprocal Chebyshev roots for a fixed Richardson polynomial."""
    order = int(order)
    if order < 1:
        raise ValueError("Chebyshev order must be positive")
    if not 0.0 < low < high:
        raise ValueError("Chebyshev interval must satisfy 0 < low < high")
    indices = np.arange(order, dtype=np.float64)
    roots = (
        0.5 * (high + low)
        - 0.5 * (high - low)
        * np.cos((2.0 * indices + 1.0) * np.pi / (2.0 * order))
    )
    return tuple(float(value) for value in 1.0 / roots)


class FaceBlockPmgPrototype:
    """Nested-modal p-multigrid V-cycle over one direct face-BSR graph."""

    def __init__(
            self,
            *,
            indptr,
            indices,
            orthonormal_data,
            degree: int,
            diagonal_positions,
            schedule: str = "halve",
            chebyshev_order: int = 3,
            lambda_low_fraction: float = 0.1,
            lambda_safety_factor: float = 1.2,
            power_iterations: int = 12,
            presweeps: int = 1,
            postsweeps: int = 1,
            spmv_backend: str = "auto",
            smoother_backend: str = "auto",
            coarse_factory: Callable[[LegendreFaceBsrOperator], Callable] | None = None,
    ):
        """Build every nested p-level and the reusable p-zero correction."""
        cp = require_cupy()
        self.cp = cp
        self.indptr = cp.ascontiguousarray(indptr, dtype=cp.int32)
        self.indices = cp.ascontiguousarray(indices, dtype=cp.int32)
        self.orthonormal_data = cp.ascontiguousarray(orthonormal_data, dtype=cp.float64)
        self.degree_schedule = modal_degree_schedule(degree, schedule)
        self.num_faces = int(self.indptr.size - 1)
        self.diagonal_positions = cp.ascontiguousarray(diagonal_positions, dtype=cp.int32)
        self.chebyshev_order = int(chebyshev_order)
        self.presweeps = int(presweeps)
        self.postsweeps = int(postsweeps)
        if self.presweeps < 0 or self.postsweeps < 0:
            raise ValueError("pre/post sweep counts must be nonnegative")
        if self.presweeps + self.postsweeps == 0:
            raise ValueError("at least one smoothing application is required")
        if not 0.0 < float(lambda_low_fraction) < 1.0:
            raise ValueError("lambda_low_fraction must lie in (0,1)")
        requested_smoother = str(smoother_backend).replace("_", "-").lower()
        if requested_smoother not in {"auto", "cupy", "fused-raw-cuda"}:
            raise ValueError(
                "smoother_backend must be auto, cupy, or fused-raw-cuda"
            )
        self.smoother_backend_requested = requested_smoother
        self.levels: list[_FacePmgLevel] = []
        for level_index, level_degree in enumerate(self.degree_schedule):
            block_size = int(level_degree + 1)
            level_data = principal_modal_bsr_data(self.orthonormal_data, block_size)
            operator = LegendreFaceBsrOperator(
                self.indptr, self.indices, level_data, backend=spmv_backend
            )
            diagonal = operator.diagonal_blocks(self.diagonal_positions)
            diagonal = cp.ascontiguousarray(0.5 * (diagonal + diagonal.transpose(0, 2, 1)))
            try:
                diagonal_inverse = cp.linalg.inv(diagonal)
            except Exception as exc:
                operator.close()
                raise RuntimeError(
                    f"failed to invert p={level_degree} face diagonal blocks"
                ) from exc
            diagonal_inverse = cp.ascontiguousarray(
                0.5 * (diagonal_inverse + diagonal_inverse.transpose(0, 2, 1))
            )
            if level_index + 1 == len(self.degree_schedule):
                lambda_max = lambda_low = None
                weights: tuple[float, ...] = ()
            else:
                lambda_max = _estimate_preconditioned_lambda_max(
                    operator,
                    diagonal,
                    diagonal_inverse,
                    iterations=power_iterations,
                    safety_factor=lambda_safety_factor,
                )
                lambda_low = float(lambda_low_fraction) * lambda_max
                weights = _chebyshev_richardson_weights(
                    self.chebyshev_order, lambda_low, lambda_max
                )
            level_smoother = "coarse"
            if level_index + 1 < len(self.degree_schedule):
                level_smoother = requested_smoother
                if level_smoother == "auto":
                    level_smoother = (
                        "fused-raw-cuda" if block_size <= 10 else "cupy"
                    )
            self.levels.append(_FacePmgLevel(
                degree=level_degree,
                operator=operator,
                diagonal=diagonal,
                diagonal_inverse=diagonal_inverse,
                chebyshev_weights=weights,
                lambda_max=lambda_max,
                lambda_low=lambda_low,
                smoother_backend=level_smoother,
            ))
        if int(self.levels[-1].block_size) != 1:
            self.close()
            raise AssertionError("the final p-level must contain only the constant mode")
        if coarse_factory is None:
            self.close()
            raise ValueError("coarse_factory must provide the p=0 correction")
        self.coarse_solver = coarse_factory(self.levels[-1].operator)
        self._workspaces = []
        for level_index, level in enumerate(self.levels[:-1]):
            coarse_level = self.levels[level_index + 1]
            fine_size = self.num_faces * level.block_size
            coarse_size = self.num_faces * coarse_level.block_size
            self._workspaces.append(_FacePmgWorkspace(
                correction=cp.empty(fine_size, dtype=cp.float64),
                scratch=cp.empty(fine_size, dtype=cp.float64),
                residual=cp.empty(fine_size, dtype=cp.float64),
                coarse_rhs=cp.empty(coarse_size, dtype=cp.float64),
            ))
        self._apply_active = False

    @property
    def fine_operator(self) -> LegendreFaceBsrOperator:
        """Return the finest normalized modal operator."""
        return self.levels[0].operator

    @property
    def workspace_bytes(self) -> int:
        """Return persistent device bytes used by V-cycle work vectors."""
        return int(sum(workspace.nbytes for workspace in self._workspaces))

    @property
    def diagnostics(self) -> tuple[FacePmgLevelDiagnostics, ...]:
        """Return immutable level setup diagnostics."""
        return tuple(FacePmgLevelDiagnostics(
            degree=level.degree,
            block_size=level.block_size,
            lambda_max=level.lambda_max,
            lambda_low=level.lambda_low,
            spmv_backend=level.operator.backend_used,
            spmv_fallback_reason=level.operator.fallback_reason,
            smoother_backend=level.smoother_backend,
        ) for level in self.levels)

    def _smooth(
            self, level: _FacePmgLevel, rhs, correction, *, reverse: bool,
            zero_initial: bool = False, scratch=None,
    ):
        """Apply fixed block-preconditioned Richardson-Chebyshev stages."""
        weights = level.chebyshev_weights[::-1] if reverse else level.chebyshev_weights
        if level.smoother_backend == "fused-raw-cuda":
            if scratch is None:
                scratch = self.cp.empty_like(correction)
            if int(scratch.data.ptr) == int(correction.data.ptr):
                raise ValueError("smoother correction and scratch must not alias")
            first_stage = True
            for _ in range(self.postsweeps if reverse else self.presweeps):
                for weight in weights:
                    if zero_initial and first_stage:
                        level.operator.fused_block_jacobi_zero_start(
                            rhs, level.diagonal_inverse, weight, out=scratch
                        )
                    else:
                        level.operator.fused_block_jacobi_step(
                            rhs, correction, level.diagonal_inverse, weight,
                            out=scratch,
                        )
                    correction, scratch = scratch, correction
                    first_stage = False
            return correction
        first_stage = True
        for _ in range(self.postsweeps if reverse else self.presweeps):
            for weight in weights:
                if zero_initial and first_stage:
                    correction = weight * _block_action(
                        level.diagonal_inverse, rhs, level.block_size
                    )
                else:
                    residual = rhs - level.operator.matvec(correction)
                    update = _block_action(
                        level.diagonal_inverse, residual, level.block_size
                    )
                    correction = correction + weight * update
                first_stage = False
        return correction

    @staticmethod
    def _alternate_buffer(workspace: _FacePmgWorkspace, correction):
        """Return the persistent ping-pong buffer not holding correction."""
        if int(correction.data.ptr) == int(workspace.correction.data.ptr):
            return workspace.scratch
        if int(correction.data.ptr) == int(workspace.scratch.data.ptr):
            return workspace.correction
        raise RuntimeError("fused smoother returned an unowned correction buffer")

    def _apply_level(self, level_index: int, rhs):
        """Apply one recursive zero-initialized p-multigrid V-cycle."""
        if level_index + 1 == len(self.levels):
            return self.coarse_solver(rhs)
        level = self.levels[level_index]
        coarse_level = self.levels[level_index + 1]
        workspace = self._workspaces[level_index]
        correction = workspace.correction
        if self.presweeps:
            correction = self._smooth(
                level, rhs, correction, reverse=False, zero_initial=True,
                scratch=workspace.scratch,
            )
        else:
            correction.fill(0.0)
        level.operator.matvec(correction, out=workspace.residual)
        self.cp.subtract(rhs, workspace.residual, out=workspace.residual)
        fine_residual = workspace.residual.reshape(
            (self.num_faces, level.block_size)
        )
        coarse_rhs = workspace.coarse_rhs.reshape(
            (self.num_faces, coarse_level.block_size)
        )
        coarse_rhs[...] = fine_residual[:, :coarse_level.block_size]
        coarse_error = self._apply_level(
            level_index + 1, workspace.coarse_rhs
        )
        correction_view = correction.reshape(
            (self.num_faces, level.block_size)
        )
        correction_view[:, :coarse_level.block_size] += coarse_error.reshape(
            (self.num_faces, coarse_level.block_size)
        )
        if self.postsweeps:
            scratch = None
            if level.smoother_backend == "fused-raw-cuda":
                scratch = self._alternate_buffer(workspace, correction)
            correction = self._smooth(
                level, rhs, correction, reverse=True, scratch=scratch
            )
        return correction

    def apply(self, rhs):
        """Apply one serial V-cycle and return its borrowed workspace output."""
        rhs = self.cp.asarray(rhs, dtype=self.cp.float64)
        if rhs.ndim != 1 or int(rhs.size) != int(self.fine_operator.shape[0]):
            raise ValueError(f"rhs must have shape ({self.fine_operator.shape[0]},)")
        if self._apply_active:
            raise RuntimeError("FaceBlockPmgPrototype.apply is not reentrant")
        self._apply_active = True
        try:
            return self._apply_level(0, rhs)
        finally:
            self._apply_active = False

    def symmetry_defect(self) -> float:
        """Measure ``|x^T B y-y^T B x|`` with deterministic device vectors."""
        size = int(self.fine_operator.shape[0])
        modes = self.cp.arange(size, dtype=self.cp.float64)
        x = self.cp.sin(0.37 * (modes + 1.0))
        y = self.cp.cos(0.23 * (modes + 0.5))
        bx = self.apply(x).copy()
        by = self.apply(y)
        left = self.cp.vdot(x, by)
        right = self.cp.vdot(y, bx)
        denominator = self.cp.linalg.norm(x) * self.cp.linalg.norm(by)
        denominator += self.cp.linalg.norm(y) * self.cp.linalg.norm(bx)
        return float((self.cp.abs(left - right) / self.cp.maximum(
            denominator, self.cp.finfo(self.cp.float64).tiny
        )).get())

    def positive_action_sample(self) -> float:
        """Return a sampled normalized ``x^T B x`` curvature value."""
        size = int(self.fine_operator.shape[0])
        modes = self.cp.arange(size, dtype=self.cp.float64)
        x = self.cp.sin(0.41 * (modes + 1.25))
        bx = self.apply(x)
        value = self.cp.real(self.cp.vdot(x, bx))
        scale = self.cp.linalg.norm(x) * self.cp.linalg.norm(bx)
        return float((value / self.cp.maximum(
            scale, self.cp.finfo(self.cp.float64).tiny
        )).get())

    def close(self) -> None:
        """Release coarse hierarchy and all p-level SpMV descriptors."""
        coarse = getattr(self, "coarse_solver", None)
        if coarse is not None and hasattr(coarse, "close"):
            coarse.close()
        for level in getattr(self, "levels", ()):
            level.operator.close()
        self._workspaces = []
        self.levels = []


def solve_pcg_prototype(
        operator: LegendreFaceBsrOperator,
        rhs,
        preconditioner: FaceBlockPmgPrototype,
        *,
        rtol: float = 1.0e-8,
        atol: float = 0.0,
        maxiter: int = 500,
        true_residual_every: int = 10,
) -> PrototypePcgResult:
    """Solve with FP64 PCG and the repository's true-residual contract."""
    cp = require_cupy()
    rhs = cp.asarray(rhs, dtype=cp.float64)
    if rhs.ndim != 1 or int(rhs.size) != int(operator.shape[0]):
        raise ValueError(f"rhs must have shape ({operator.shape[0]},)")
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("rtol and atol must be nonnegative")
    x = cp.zeros_like(rhs)
    residual = rhs.copy()
    rhs_norm = float(cp.linalg.norm(rhs).get())
    initial_norm = float(cp.linalg.norm(residual).get())
    target = max(float(atol), float(rtol) * rhs_norm)
    history = [initial_norm]
    if initial_norm <= target:
        return PrototypePcgResult(
            x, True, 0, initial_norm, rhs_norm,
            initial_norm / max(rhs_norm, np.finfo(float).tiny),
            0.0 if initial_norm == 0.0 else 1.0,
            target, tuple(history), 0.0,
        )
    started = time.perf_counter()
    z = preconditioner.apply(residual)
    rho = cp.real(cp.vdot(residual, z))
    rho_value = float(rho.get())
    if not np.isfinite(rho_value) or rho_value <= 0.0:
        raise RuntimeError(f"preconditioner is not positive on the initial residual: {rho_value}")
    direction = z.copy()
    converged = False
    iterations = 0
    for iteration in range(1, int(maxiter) + 1):
        applied = operator.matvec(direction)
        curvature = cp.real(cp.vdot(direction, applied))
        curvature_value = float(curvature.get())
        if not np.isfinite(curvature_value) or curvature_value <= 0.0:
            raise RuntimeError(
                f"PCG lost positive A-curvature at iteration {iteration}: {curvature_value}"
            )
        alpha = rho / curvature
        x = x + alpha * direction
        residual = residual - alpha * applied
        if true_residual_every > 0 and iteration % int(true_residual_every) == 0:
            residual = rhs - operator.matvec(x)
        residual_norm = float(cp.linalg.norm(residual).get())
        history.append(residual_norm)
        iterations = iteration
        if residual_norm <= target:
            converged = True
            break
        z = preconditioner.apply(residual)
        rho_new = cp.real(cp.vdot(residual, z))
        rho_new_value = float(rho_new.get())
        if not np.isfinite(rho_new_value) or rho_new_value <= 0.0:
            raise RuntimeError(
                f"preconditioner lost positive curvature at iteration {iteration}: "
                f"{rho_new_value}"
            )
        direction = z + (rho_new / rho) * direction
        rho = rho_new
    true_residual = rhs - operator.matvec(x)
    cp.cuda.get_current_stream().synchronize()
    residual_norm = float(cp.linalg.norm(true_residual).get())
    elapsed = time.perf_counter() - started
    relative = residual_norm / max(rhs_norm, np.finfo(float).tiny)
    over_initial = residual_norm / max(initial_norm, np.finfo(float).tiny)
    converged = bool(converged and residual_norm <= target)
    return PrototypePcgResult(
        solution=x,
        converged=converged,
        iterations=iterations,
        residual_norm=residual_norm,
        rhs_norm=rhs_norm,
        relative_residual=relative,
        residual_over_initial=over_initial,
        target=target,
        history=tuple(history),
        elapsed_seconds=elapsed,
    )


def solve_pcgf_prototype(
        operator: LegendreFaceBsrOperator,
        rhs,
        preconditioner: FaceBlockPmgPrototype,
        *,
        rtol: float = 1.0e-8,
        atol: float = 0.0,
        maxiter: int = 500,
        true_residual_every: int = 10,
) -> PrototypePcgResult:
    """Solve with the AMGX PCGF recurrence and true FP64 residual checks.

    PCGF is a diagnostic outer method while the p=0 correction is not proven
    self-adjoint. The implementation mirrors AMGX's flexible beta update,
    ``z_new.T @ (r_new-r_old) / (r_old.T @ z_old)``, and still rejects loss of
    positive operator or preconditioner curvature.
    """
    cp = require_cupy()
    rhs = cp.asarray(rhs, dtype=cp.float64)
    if rhs.ndim != 1 or int(rhs.size) != int(operator.shape[0]):
        raise ValueError(f"rhs must have shape ({operator.shape[0]},)")
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("rtol and atol must be nonnegative")
    x = cp.zeros_like(rhs)
    residual = rhs.copy()
    rhs_norm = float(cp.linalg.norm(rhs).get())
    initial_norm = float(cp.linalg.norm(residual).get())
    target = max(float(atol), float(rtol) * rhs_norm)
    history = [initial_norm]
    if initial_norm <= target:
        return PrototypePcgResult(
            x, True, 0, initial_norm, rhs_norm,
            initial_norm / max(rhs_norm, np.finfo(float).tiny),
            0.0 if initial_norm == 0.0 else 1.0,
            target, tuple(history), 0.0,
        )
    started = time.perf_counter()
    z = preconditioner.apply(residual)
    rho = cp.real(cp.vdot(residual, z))
    rho_value = float(rho.get())
    if not np.isfinite(rho_value) or rho_value <= 0.0:
        raise RuntimeError(
            f"preconditioner is not positive on the initial residual: {rho_value}"
        )
    direction = z.copy()
    converged = False
    iterations = 0
    for iteration in range(1, int(maxiter) + 1):
        applied = operator.matvec(direction)
        curvature = cp.real(cp.vdot(applied, direction))
        curvature_value = float(curvature.get())
        if not np.isfinite(curvature_value) or curvature_value <= 0.0:
            raise RuntimeError(
                f"PCGF lost positive A-curvature at iteration {iteration}: "
                f"{curvature_value}"
            )
        alpha = rho / curvature
        previous_residual = residual
        x = x + alpha * direction
        residual = residual - alpha * applied
        if true_residual_every > 0 and iteration % int(true_residual_every) == 0:
            residual = rhs - operator.matvec(x)
        residual_norm = float(cp.linalg.norm(residual).get())
        history.append(residual_norm)
        iterations = iteration
        if residual_norm <= target:
            converged = True
            break
        residual_delta = residual - previous_residual
        z_new = preconditioner.apply(residual)
        rho_new = cp.real(cp.vdot(residual, z_new))
        rho_new_value = float(rho_new.get())
        if not np.isfinite(rho_new_value) or rho_new_value <= 0.0:
            raise RuntimeError(
                f"preconditioner lost positive curvature at iteration {iteration}: "
                f"{rho_new_value}"
            )
        beta = cp.real(cp.vdot(z_new, residual_delta)) / rho
        direction = z_new + beta * direction
        z = z_new
        rho = rho_new
    true_residual = rhs - operator.matvec(x)
    cp.cuda.get_current_stream().synchronize()
    residual_norm = float(cp.linalg.norm(true_residual).get())
    elapsed = time.perf_counter() - started
    relative = residual_norm / max(rhs_norm, np.finfo(float).tiny)
    over_initial = residual_norm / max(initial_norm, np.finfo(float).tiny)
    converged = bool(converged and residual_norm <= target)
    return PrototypePcgResult(
        solution=x,
        converged=converged,
        iterations=iterations,
        residual_norm=residual_norm,
        rhs_norm=rhs_norm,
        relative_residual=relative,
        residual_over_initial=over_initial,
        target=target,
        history=tuple(history),
        elapsed_seconds=elapsed,
    )


__all__ = [
    "AmgxScalarVcycle",
    "CupyxCgScalarSolve",
    "FaceBlockPmgPrototype",
    "FacePmgLevelDiagnostics",
    "PrototypePcgResult",
    "scalar_p0_amgx_config",
    "solve_pcgf_prototype",
    "solve_pcg_prototype",
    "symmetric_scalar_amgx_config",
]
