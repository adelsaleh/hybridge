"""Inexpensive CuPy prototype of modal face-block p-multigrid.

The implementation deliberately favors transparent numerical operations over
kernel fusion.  It validates nested Legendre p-coarsening, block-polynomial
smoothing, a reusable scalar p=0 AMG correction, and the SPD/PCG contract before
production integration.
"""

from __future__ import annotations

from hdgfem.runtime.precision import REAL_DTYPE, audit_arrays

import copy
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from hdgfem.runtime.optional import require_cupy
from hdgfem.linalg.system import residual_history_is_stagnated
from hdgfem.linalg.face_hp_policy import (
    scalar_p0_amgx_config,
    robust_scalar_p0_amgx_config,
    face_hp_mg_preconditioner_parameters,
)
from hdgfem.backends.legendre_face_bsr import (
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


@dataclass(frozen=True)
class FaceBlockHpMgPcgResult:
    """Validated result from the reusable production FB-HP-MG PCG solver."""

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
    best_iteration: int | None = None
    terminal_residual_norm: float | None = None
    true_residual_check_count: int = 0
    returned_best_iterate: bool = False
    breakdown_reason: str | None = None
    residual_restart_count: int = 0


def _format_fb_hp_mg_pcg_stats(
        *, degree: int, diagnostics: tuple[FacePmgLevelDiagnostics, ...],
        history: tuple[float, ...], iterations: int, residual_norm: float,
        rhs_norm: float, target: float, true_residual_every: int,
        workspace_bytes: int, coarse_apply_count: int,
        coarse_apply_seconds: float,
        best_iteration: int | None = None,
        terminal_residual_norm: float | None = None,
        returned_best_iterate: bool = False,
        preconditioner_policy: str = "standard",
        section: str = "all",
        residual_restart_count: int = 0,
        breakdown_reason: str | None = None,
        original_matrix_residual: bool = False,
) -> str:
    """Format native telemetry, optionally just the live header or final summary."""
    initial_norm = history[0] if history else residual_norm
    tiny = np.finfo(REAL_DTYPE).tiny
    schedule = " -> ".join(f"p={item.degree}" for item in diagnostics)
    fine = diagnostics[0] if diagnostics else None
    smoother = "unknown" if fine is None else fine.smoother_backend
    spmv = "unknown" if fine is None else fine.spmv_backend
    lines = [
        "  FB-HP-MG-PCG convergence (native outer solver):",
        f"    fine action: {spmv} face BSR, block={int(degree) + 1}",
        (
            f"    hierarchy: {schedule}; smoother={smoother}; "
            f"policy={preconditioner_policy}"
        ),
        "    coarse correction: scalar AMGX p=0, one fixed V-cycle/application",
        "    outer recurrence: flexible PCGF; restart on significant residual gap",
        (
            "    residual: assembly-basis L2; recursive between true refreshes "
            f"(every {int(true_residual_every)} iterations and at exit)"
        ),
        f"    target: {target:.6e}",
        "    iter       residual        res/rhs    res/initial       sample",
        "    --------------------------------------------------------------",
    ]
    if original_matrix_residual:
        lines.insert(6, "    true residual action: original assembled matrix")
    if section == "header":
        return "\n".join(lines)
    if section not in {"all", "footer"}:
        raise ValueError(f"unknown PCGF report section: {section}")
    displayed = list(history[:int(iterations) + 1])
    if not displayed:
        displayed = [residual_norm]
    displayed[-1] = (
        residual_norm
        if terminal_residual_norm is None
        else terminal_residual_norm
    )
    for iteration, value in enumerate(displayed):
        is_true = (
            iteration == 0
            or iteration == int(iterations)
            or (
                int(true_residual_every) > 0
                and iteration % int(true_residual_every) == 0
            )
        )
        lines.append(_format_fb_hp_mg_pcg_row(
            iteration, value, rhs_norm, initial_norm, is_true=is_true,
        ))
    if section == "footer":
        lines = []
    lines.extend((
        "    --------------------------------------------------------------",
        f"    Total iterations: {int(iterations)}",
        (
            f"    {'Returned' if returned_best_iterate else 'Final'} true residual: "
            f"{residual_norm:.6e} "
            f"({residual_norm / max(rhs_norm, tiny):.3e} of RHS)"
        ),
        f"    Native workspace: {workspace_bytes / (1024.0 ** 3):.3f} GiB",
        (
            f"    Coarse AMGX: applications={int(coarse_apply_count)} "
            f"elapsed={coarse_apply_seconds:.5f}s; hierarchy=reused"
        ),
    ))
    if terminal_residual_norm is not None:
        lines.insert(-2, (
            f"    Terminal true residual: {terminal_residual_norm:.6e}; "
            f"best checkpoint: iter {int(best_iteration or 0)}"
            + (" (returned)" if returned_best_iterate else "")
        ))
    lines.append(f"    Residual restarts: {int(residual_restart_count)}")
    if breakdown_reason is not None:
        lines.append(f"    Stopped: {breakdown_reason}")
    return "\n".join(lines)


def _format_fb_hp_mg_pcg_row(
        iteration: int, value: float, rhs_norm: float, initial_norm: float,
        *, is_true: bool,
) -> str:
    """Format one immediately available residual sample for live output."""
    tiny = np.finfo(REAL_DTYPE).tiny
    sample = "true" if is_true else "recursive"
    return (
        f"    {iteration:4d}  {value:14.6e}  "
        f"{value / max(rhs_norm, tiny):13.3e}  "
        f"{value / max(initial_norm, tiny):13.3e}  {sample:>11}"
    )


def _pcgf_beta(cp, preconditioned_residual, residual_delta, previous_rho):
    """AMGX's flexible CG update, shared by native and prototype iterations."""
    return cp.real(cp.vdot(preconditioned_residual, residual_delta)) / previous_rho


class AmgxScalarVcycle:
    """One reusable scalar classical-AMG application on the p=0 face graph."""

    def __init__(self, operator: LegendreFaceBsrOperator, *, config, verbose: int = 0):
        """Build one reusable AMGX hierarchy for the scalar face operator."""
        if int(operator.block_size) != 1:
            raise ValueError("the scalar AMG coarse operator must have block size one")
        from hdgfem.backends.advection_cuda import PyAMGXCsrDeviceSolver

        self.operator = operator
        self.config = copy.deepcopy(config)
        self.solver = PyAMGXCsrDeviceSolver(
            config=self.config,
            tolerance=1.0e-30,
            maxiter=1,
            verbose=verbose,
            reusable=True,
            fixed_amg_cycles=1,
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
        from hdgfem.runtime.optional import (
                    require_cupyx_sparse,
                    require_cupyx_sparse_linalg,
                )

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
    phase = cp.arange(size, dtype=REAL_DTYPE)
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
    indices = np.arange(order, dtype=REAL_DTYPE)
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
        self.orthonormal_data = cp.ascontiguousarray(orthonormal_data, dtype=REAL_DTYPE)
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
                correction=cp.empty(fine_size, dtype=REAL_DTYPE),
                scratch=cp.empty(fine_size, dtype=REAL_DTYPE),
                residual=cp.empty(fine_size, dtype=REAL_DTYPE),
                coarse_rhs=cp.empty(coarse_size, dtype=REAL_DTYPE),
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
        rhs = self.cp.asarray(rhs, dtype=REAL_DTYPE)
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
        modes = self.cp.arange(size, dtype=REAL_DTYPE)
        x = self.cp.sin(0.37 * (modes + 1.0))
        y = self.cp.cos(0.23 * (modes + 0.5))
        bx = self.apply(x).copy()
        by = self.apply(y)
        left = self.cp.vdot(x, by)
        right = self.cp.vdot(y, bx)
        denominator = self.cp.linalg.norm(x) * self.cp.linalg.norm(by)
        denominator += self.cp.linalg.norm(y) * self.cp.linalg.norm(bx)
        return float((self.cp.abs(left - right) / self.cp.maximum(
            denominator, self.cp.finfo(REAL_DTYPE).tiny
        )).get())

    def positive_action_sample(self) -> float:
        """Return a sampled normalized ``x^T B x`` curvature value."""
        size = int(self.fine_operator.shape[0])
        modes = self.cp.arange(size, dtype=REAL_DTYPE)
        x = self.cp.sin(0.41 * (modes + 1.25))
        bx = self.apply(x)
        value = self.cp.real(self.cp.vdot(x, bx))
        scale = self.cp.linalg.norm(x) * self.cp.linalg.norm(bx)
        return float((value / self.cp.maximum(
            scale, self.cp.finfo(REAL_DTYPE).tiny
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


class FaceBlockHpMgPcgSolver:
    """Reusable native face-BSR p-multigrid flexible-CG backend.

    Inputs use the HDG Legendre assembly basis. Setup performs the congruence
    transformation to orthonormal modal coordinates once, builds the selected
    symmetric p-hierarchy, and retains all Krylov and multigrid workspaces.
    Solutions are mapped back to the assembly basis for HDG reconstruction.
    """

    def __init__(
            self, *, indptr, indices, data, degree: int, diagonal_positions,
            symmetry_limit: float = max(1.0e-10, 32 * np.finfo(REAL_DTYPE).eps), curvature_limit: float = 0.0,
            preconditioner_policy: str = "standard",
            preconditioner_tuning: dict | None = None,
            verbose: int = 0,
    ):
        """Build and validate the fixed native hierarchy."""
        from hdgfem.backends.legendre_face_bsr import legendre_orthonormal_scales

        cp = require_cupy()
        self.cp = cp
        self.verbose = max(0, int(verbose))
        self.degree = int(degree)
        self.block_size = self.degree + 1
        self.preconditioner_policy = str(preconditioner_policy).replace("_", "-").lower()
        policy = face_hp_mg_preconditioner_parameters(
            self.preconditioner_policy, overrides=preconditioner_tuning,
        )
        self.preconditioner_parameters = policy
        if not 4 <= self.degree <= 6:
            raise ValueError(
                "FB-HP-MG-PCG currently supports polynomial degrees 4 through 6"
            )
        if data.ndim != 3 or tuple(data.shape[1:]) != (
            self.block_size, self.block_size,
        ):
            raise ValueError(
                "FB-HP-MG-PCG requires face BSR blocks with shape "
                f"({self.block_size}, {self.block_size})"
            )
        started = time.perf_counter()
        self.scales = cp.ascontiguousarray(
            legendre_orthonormal_scales(self.block_size, xp=cp)
        )
        self.orthonormal_data = cp.ascontiguousarray(
            data * self.scales[None, :, None] * self.scales[None, None, :]
        )
        self.preconditioner = None
        try:
            self.preconditioner = FaceBlockPmgPrototype(
                indptr=indptr,
                indices=indices,
                orthonormal_data=self.orthonormal_data,
                degree=self.degree,
                diagonal_positions=diagonal_positions,
                schedule=policy["schedule"],
                chebyshev_order=policy["chebyshev_order"],
                presweeps=policy["presweeps"],
                postsweeps=policy["postsweeps"],
                spmv_backend="auto",
                smoother_backend="fused-raw-cuda",
                coarse_factory=lambda operator: AmgxScalarVcycle(
                    operator,
                    config=policy["coarse_config"],
                    verbose=0,
                ),
            )
            self.symmetry_defect = float(self.preconditioner.symmetry_defect())
            self.positive_curvature = float(
                self.preconditioner.positive_action_sample()
            )
            if (
                not np.isfinite(self.symmetry_defect)
                or self.symmetry_defect > float(symmetry_limit)
            ):
                raise RuntimeError(
                    "FB-HP-MG symmetry gate failed: defect "
                    f"{self.symmetry_defect:.3e} exceeds {float(symmetry_limit):.3e}"
                )
            if (
                not np.isfinite(self.positive_curvature)
                or self.positive_curvature <= float(curvature_limit)
            ):
                raise RuntimeError(
                    "FB-HP-MG curvature gate failed: sampled normalized curvature "
                    f"is {self.positive_curvature:.3e}"
                )
        except Exception:
            self.close()
            raise

        size = int(self.preconditioner.fine_operator.shape[0])
        self._rhs_orthonormal = cp.empty(size, dtype=REAL_DTYPE)
        self._x = cp.empty(size, dtype=REAL_DTYPE)
        self._residual = cp.empty(size, dtype=REAL_DTYPE)
        self._residual_delta = cp.empty(size, dtype=REAL_DTYPE)
        self._direction = cp.empty(size, dtype=REAL_DTYPE)
        self._applied = cp.empty(size, dtype=REAL_DTYPE)
        self._best_x = cp.empty(size, dtype=REAL_DTYPE)
        self._assembly_solution = cp.empty(size, dtype=REAL_DTYPE)
        self.setup_seconds = time.perf_counter() - started
        self.solve_count = 0
        self.setup_count = 1

    @property
    def fine_operator(self) -> LegendreFaceBsrOperator:
        """Return the cached generic-cuSPARSE finest operator."""
        return self.preconditioner.fine_operator

    @property
    def workspace_bytes(self) -> int:
        """Return persistent native Krylov and V-cycle workspace bytes."""
        krylov = sum(
            int(array.nbytes)
            for array in (
                self._rhs_orthonormal, self._x, self._residual, self._residual_delta,
                self._direction, self._applied, self._best_x,
                self._assembly_solution,
            )
        )
        return krylov + int(self.preconditioner.workspace_bytes)

    def _assembly_norm_device(self, orthonormal_vector):
        """Return the assembly-basis norm as a device scalar."""
        view = orthonormal_vector.reshape((-1, self.block_size))
        workspace = self._assembly_solution.reshape((-1, self.block_size))
        self.cp.divide(view, self.scales[None, :], out=workspace)
        return self.cp.linalg.norm(self._assembly_solution)

    def _assembly_norm(self, orthonormal_vector) -> float:
        """Return the norm after mapping a modal RHS/residual by ``S^-1``."""
        return float(self._assembly_norm_device(orthonormal_vector).get())

    def _refresh_true_residual(self, rhs, *, assembly_matvec, out):
        """Refresh ``S*(b-A*x)`` and return its assembly-basis norm on device.

        When the original action is supplied, measure the norm before mapping
        the residual into Krylov coordinates. This avoids the roundoff mismatch
        between the original coefficients and the stored congruence ``S*A*S``.
        Only scratch vectors are used; no extra iterate checkpoint is retained.
        """
        cp = self.cp
        if assembly_matvec is None:
            self.fine_operator.matvec(self._x, out=out)
            cp.subtract(self._rhs_orthonormal, out, out=out)
            return self._assembly_norm_device(out)
        cp.multiply(
            self._x.reshape((-1, self.block_size)), self.scales[None, :],
            out=self._assembly_solution.reshape((-1, self.block_size)),
        )
        cp.subtract(rhs, assembly_matvec(self._assembly_solution), out=out)
        norm = cp.linalg.norm(out)
        view = out.reshape((-1, self.block_size))
        cp.multiply(view, self.scales[None, :], out=view)
        return norm

    def solve(
            self, rhs, *, initial_guess=None, rtol: float = 1.0e-8,
            atol: float = 0.0, maxiter: int = 500,
            true_residual_every: int = 10,
            store_residual_history: bool = True,
            residual_gap_restart: float = 0.1,
            stagnation_checks: int = 6,
            assembly_matvec: Callable[[Any], Any] | None = None,
    ) -> FaceBlockHpMgPcgResult:
        """Solve with PCGF, retaining one best checked iterate for recovery.

        True refreshes that materially change the recursive residual restart
        the search direction. Six true checks without 1% improvement in the
        best norm end an unproductive attempt, without relaxing the tolerance.
        ``stagnation_checks=0`` disables that safeguard.
        ``true_residual_every=0`` omits periodic refreshes but keeps initial,
        convergence-candidate and terminal checks. ``store_residual_history=False``
        omits the returned iteration history without changing stopping tests.

        ``assembly_matvec(x)`` optionally applies the original assembled matrix
        to an assembly-basis vector. Every true check (including convergence and
        checkpoint selection) then uses that action, while Krylov products keep
        using the transformed operator. A rejected convergence check refreshes
        the residual and restarts PCGF instead of abandoning the hierarchy.
        """
        cp = self.cp
        rhs = cp.asarray(rhs, dtype=REAL_DTYPE)
        size = int(self.fine_operator.shape[0])
        if rhs.ndim != 1 or int(rhs.size) != size:
            raise ValueError(f"rhs must have shape ({size},)")
        if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0.0 or atol < 0.0:
            raise ValueError("rtol and atol must be finite and nonnegative")
        if int(true_residual_every) != true_residual_every or true_residual_every < 0:
            raise ValueError("true_residual_every must be a nonnegative integer")
        if not np.isfinite(residual_gap_restart) or residual_gap_restart <= 0.0:
            raise ValueError("residual_gap_restart must be finite and positive")
        if int(stagnation_checks) != stagnation_checks or (stagnation_checks != 0 and stagnation_checks < 2):
            raise ValueError("stagnation_checks must be zero or an integer at least two")
        cp.multiply(
            rhs.reshape((-1, self.block_size)),
            self.scales[None, :],
            out=self._rhs_orthonormal.reshape((-1, self.block_size)),
        )
        if initial_guess is None:
            self._x.fill(0.0)
        else:
            guess = cp.asarray(initial_guess, dtype=REAL_DTYPE)
            if guess.ndim != 1 or int(guess.size) != size:
                raise ValueError(f"initial_guess must have shape ({size},)")
            self._x[...] = (
                guess.reshape((-1, self.block_size)) / self.scales[None, :]
            ).reshape(-1)

        audit_arrays("poisson-krylov-workspace", self, rhs)
        coarse_solver = self.preconditioner.coarse_solver
        coarse_count_before = int(getattr(coarse_solver, "apply_count", 0))
        coarse_seconds_before = float(getattr(coarse_solver, "apply_seconds", 0.0))
        started = time.perf_counter()
        initial_norm_device = self._refresh_true_residual(
            rhs, assembly_matvec=assembly_matvec, out=self._residual,
        )
        rhs_norm, initial_norm = (
            float(value)
            for value in cp.stack(
                (cp.linalg.norm(rhs), initial_norm_device)
            ).get()
        )
        target = max(float(atol), float(rtol) * rhs_norm)
        history = [initial_norm] if store_residual_history else []
        if not np.isfinite(rhs_norm) or not np.isfinite(initial_norm):
            raise ValueError("FB-HP-MG requires finite RHS and initial residual norms")
        best_residual_norm = initial_norm
        best_iteration = 0
        true_residual_check_count = 1
        self._best_x[...] = self._x
        residual_restart_count = 0
        recent_best = deque([initial_norm], maxlen=max(2, int(stagnation_checks)))

        if self.verbose >= 3:
            print(_format_fb_hp_mg_pcg_stats(
                degree=self.degree, diagnostics=self.preconditioner.diagnostics,
                history=tuple(history), iterations=0, residual_norm=initial_norm,
                rhs_norm=rhs_norm, target=target, true_residual_every=true_residual_every,
                workspace_bytes=0, coarse_apply_count=0, coarse_apply_seconds=0.,
                preconditioner_policy=self.preconditioner_policy, section="header",
                original_matrix_residual=assembly_matvec is not None,
            ), flush=True)
            print(_format_fb_hp_mg_pcg_row(
                0, initial_norm, rhs_norm, initial_norm, is_true=True,
            ), flush=True)

        def retain_true_checkpoint(iteration: int, value: float) -> None:
            """Keep exactly one device iterate: the best true-residual point."""
            nonlocal best_residual_norm, best_iteration
            if np.isfinite(value) and value < best_residual_norm:
                best_residual_norm = value
                best_iteration = int(iteration)
                self._best_x[...] = self._x

        converged = initial_norm <= target
        iterations = 0
        breakdown_reason = None
        if not converged:
            z = self.preconditioner.apply(self._residual)
            rho = cp.real(cp.vdot(self._residual, z))
            rho_value = float(rho.get())
            if not np.isfinite(rho_value) or rho_value <= 0.0:
                raise RuntimeError(
                    "FB-HP-MG preconditioner is not positive on the initial "
                    f"residual: {rho_value}"
                )
            self._direction[...] = z
            rho_iteration = 0
            for iteration in range(1, int(maxiter) + 1):
                self.fine_operator.matvec(self._direction, out=self._applied)
                curvature = cp.real(cp.vdot(self._direction, self._applied))
                self._residual_delta[...] = self._residual
                alpha = rho / curvature
                self._x += alpha * self._direction
                self._residual -= alpha * self._applied
                is_true = (
                    true_residual_every > 0
                    and iteration % int(true_residual_every) == 0
                )
                gap_norm_device = None
                if is_true:
                    # _applied is now free: use it for b-A*x while retaining
                    # the recursive residual long enough to measure its gap.
                    true_norm_device = self._refresh_true_residual(
                        rhs, assembly_matvec=assembly_matvec, out=self._applied,
                    )
                    self._residual -= self._applied
                    gap_norm_device = self._assembly_norm_device(self._residual)
                    self._residual[...] = self._applied
                    true_residual_check_count += 1
                check_scalars = [
                    (true_norm_device if is_true
                     else self._assembly_norm_device(self._residual)),
                    curvature, rho,
                ]
                if is_true:
                    check_scalars.append(gap_norm_device)
                checked_values = tuple(
                    float(value)
                    for value in cp.stack(check_scalars).get()
                )
                residual_norm, curvature_value, rho_value = checked_values[:3]
                if store_residual_history:
                    history.append(residual_norm)
                iterations = iteration
                restart_direction = False
                if is_true:
                    gap_norm = checked_values[3]
                    restart_direction = (
                        not np.isfinite(gap_norm)
                        or gap_norm > residual_gap_restart * max(
                            residual_norm, np.finfo(REAL_DTYPE).tiny,
                        )
                    )
                if residual_norm <= target and not is_true:
                    residual_norm = float(self._refresh_true_residual(
                        rhs, assembly_matvec=assembly_matvec, out=self._residual,
                    ).get())
                    true_residual_check_count += 1
                    if store_residual_history:
                        history[-1] = residual_norm
                    is_true = True
                    restart_direction = residual_norm > target
                if self.verbose >= 3:
                    print(_format_fb_hp_mg_pcg_row(
                        iteration, residual_norm, rhs_norm, initial_norm, is_true=is_true,
                    ), flush=True)
                if not np.isfinite(rho_value) or rho_value <= 0.0:
                    breakdown_reason = (
                        "FB-HP-MG preconditioner lost positive curvature at "
                        f"iteration {rho_iteration}: {rho_value}"
                    )
                    break
                if not np.isfinite(curvature_value) or curvature_value <= 0.0:
                    breakdown_reason = (
                        "FB-HP-MG-PCG lost positive A-curvature at iteration "
                        f"{iteration}: {curvature_value}"
                    )
                    break
                if not np.isfinite(residual_norm):
                    breakdown_reason = f"non-finite residual at iteration {iteration}"
                    break
                if is_true:
                    retain_true_checkpoint(iteration, residual_norm)
                    recent_best.append(best_residual_norm)
                if residual_norm <= target:
                    converged = True
                    break
                if is_true and stagnation_checks and residual_history_is_stagnated(
                    recent_best, window=int(stagnation_checks), relative_improvement=0.01,
                ):
                    breakdown_reason = (
                        f"true-residual stagnation over {int(stagnation_checks)} checks; "
                        f"best {best_residual_norm:.6e} exceeds target {target:.6e}"
                    )
                    break
                if iteration == int(maxiter):
                    break
                cp.subtract(self._residual, self._residual_delta, out=self._residual_delta)
                try:
                    z = self.preconditioner.apply(self._residual)
                except RuntimeError as exc:
                    breakdown_reason = (
                        "FB-HP-MG preconditioner application failed after "
                        f"iteration {iteration}: {exc}"
                    )
                    break
                rho_new = cp.real(cp.vdot(self._residual, z))
                if restart_direction:
                    self._direction[...] = z
                    residual_restart_count += 1
                else:
                    self._direction *= _pcgf_beta(cp, z, self._residual_delta, rho)
                    self._direction += z
                rho = rho_new
                rho_iteration = iteration

        terminal_residual_norm = float(self._refresh_true_residual(
            rhs, assembly_matvec=assembly_matvec, out=self._residual,
        ).get())
        true_residual_check_count += 1
        if history:
            history[-1] = terminal_residual_norm
        retain_true_checkpoint(iterations, terminal_residual_norm)
        returned_best_iterate = bool(
            not np.isfinite(terminal_residual_norm)
            or best_residual_norm < terminal_residual_norm
        )
        if returned_best_iterate:
            self._x[...] = self._best_x
        residual_norm = best_residual_norm
        elapsed = time.perf_counter() - started
        relative = residual_norm / max(rhs_norm, np.finfo(REAL_DTYPE).tiny)
        over_initial = residual_norm / max(initial_norm, np.finfo(REAL_DTYPE).tiny)
        converged = bool(breakdown_reason is None and residual_norm <= target)
        self._assembly_solution[...] = (
            self._x.reshape((-1, self.block_size)) * self.scales[None, :]
        ).reshape(-1)
        self.solve_count += 1
        result = FaceBlockHpMgPcgResult(
            solution=self._assembly_solution,
            converged=converged,
            iterations=iterations,
            residual_norm=residual_norm,
            rhs_norm=rhs_norm,
            relative_residual=relative,
            residual_over_initial=over_initial,
            target=target,
            history=tuple(history),
            elapsed_seconds=elapsed,
            best_iteration=best_iteration,
            terminal_residual_norm=terminal_residual_norm,
            true_residual_check_count=true_residual_check_count,
            returned_best_iterate=returned_best_iterate,
            breakdown_reason=breakdown_reason,
            residual_restart_count=residual_restart_count,
        )
        if self.verbose >= 3:
            print(_format_fb_hp_mg_pcg_stats(
                degree=self.degree,
                diagnostics=self.preconditioner.diagnostics,
                history=result.history,
                iterations=result.iterations,
                residual_norm=result.residual_norm,
                rhs_norm=result.rhs_norm,
                target=result.target,
                true_residual_every=true_residual_every,
                workspace_bytes=self.workspace_bytes,
                coarse_apply_count=(
                    int(getattr(coarse_solver, "apply_count", 0))
                    - coarse_count_before
                ),
                coarse_apply_seconds=(
                    float(getattr(coarse_solver, "apply_seconds", 0.0))
                    - coarse_seconds_before
                ),
                best_iteration=result.best_iteration,
                terminal_residual_norm=result.terminal_residual_norm,
                returned_best_iterate=result.returned_best_iterate,
                preconditioner_policy=self.preconditioner_policy,
                section="footer",
                residual_restart_count=residual_restart_count,
                breakdown_reason=breakdown_reason,
                original_matrix_residual=assembly_matvec is not None,
            ), flush=True)
        return result

    def close(self) -> None:
        """Release the scalar AMGX hierarchy and generic-BSR descriptors."""
        preconditioner = getattr(self, "preconditioner", None)
        if preconditioner is not None:
            preconditioner.close()
        self.preconditioner = None


def solve_pcg_prototype(
        operator: LegendreFaceBsrOperator,
        rhs,
        preconditioner: FaceBlockPmgPrototype,
        *,
        rtol: float = 1.0e-8,
        atol: float = 0.0,
        maxiter: int = 500,
        true_residual_every: int = 10,
        initial_guess=None,
) -> PrototypePcgResult:
    """Solve with PCG at the selected precision and the repository's true-residual contract."""
    cp = require_cupy()
    rhs = cp.asarray(rhs, dtype=REAL_DTYPE)
    if rhs.ndim != 1 or int(rhs.size) != int(operator.shape[0]):
        raise ValueError(f"rhs must have shape ({operator.shape[0]},)")
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("rtol and atol must be nonnegative")
    if initial_guess is None:
        x = cp.zeros_like(rhs)
    else:
        x = cp.asarray(initial_guess, dtype=REAL_DTYPE).copy()
        if x.shape != rhs.shape:
            raise ValueError(
                f"initial_guess must have shape {rhs.shape}; got {x.shape}"
            )
    residual = rhs - operator.matvec(x)
    rhs_norm = float(cp.linalg.norm(rhs).get())
    initial_norm = float(cp.linalg.norm(residual).get())
    target = max(float(atol), float(rtol) * rhs_norm)
    history = [initial_norm]
    if initial_norm <= target:
        return PrototypePcgResult(
            x, True, 0, initial_norm, rhs_norm,
            initial_norm / max(rhs_norm, np.finfo(REAL_DTYPE).tiny),
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
    relative = residual_norm / max(rhs_norm, np.finfo(REAL_DTYPE).tiny)
    over_initial = residual_norm / max(initial_norm, np.finfo(REAL_DTYPE).tiny)
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
        initial_guess=None,
) -> PrototypePcgResult:
    """Solve with the AMGX PCGF recurrence and true residual checks at the selected precision.

    PCGF is a diagnostic outer method while the p=0 correction is not proven
    self-adjoint. The implementation mirrors AMGX's flexible beta update,
    ``z_new.T @ (r_new-r_old) / (r_old.T @ z_old)``, and still rejects loss of
    positive operator or preconditioner curvature. An optional initial guess
    is copied; the convergence target remains relative to the original RHS.
    """
    cp = require_cupy()
    rhs = cp.asarray(rhs, dtype=REAL_DTYPE)
    if rhs.ndim != 1 or int(rhs.size) != int(operator.shape[0]):
        raise ValueError(f"rhs must have shape ({operator.shape[0]},)")
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("rtol and atol must be nonnegative")
    if initial_guess is None:
        x = cp.zeros_like(rhs)
        residual = rhs.copy()
    else:
        x = cp.asarray(initial_guess, dtype=REAL_DTYPE).copy()
        if x.shape != rhs.shape:
            raise ValueError(
                f"initial_guess must have shape {rhs.shape}; got {x.shape}"
            )
        residual = rhs - operator.matvec(x)
    rhs_norm = float(cp.linalg.norm(rhs).get())
    initial_norm = float(cp.linalg.norm(residual).get())
    target = max(float(atol), float(rtol) * rhs_norm)
    history = [initial_norm]
    if initial_norm <= target:
        return PrototypePcgResult(
            x, True, 0, initial_norm, rhs_norm,
            initial_norm / max(rhs_norm, np.finfo(REAL_DTYPE).tiny),
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
        beta = _pcgf_beta(cp, z_new, residual_delta, rho)
        direction = z_new + beta * direction
        z = z_new
        rho = rho_new
    true_residual = rhs - operator.matvec(x)
    cp.cuda.get_current_stream().synchronize()
    residual_norm = float(cp.linalg.norm(true_residual).get())
    elapsed = time.perf_counter() - started
    relative = residual_norm / max(rhs_norm, np.finfo(REAL_DTYPE).tiny)
    over_initial = residual_norm / max(initial_norm, np.finfo(REAL_DTYPE).tiny)
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
    "FaceBlockHpMgPcgResult",
    "FaceBlockHpMgPcgSolver",
    "FacePmgLevelDiagnostics",
    "PrototypePcgResult",
    "face_hp_mg_preconditioner_parameters",
    "robust_scalar_p0_amgx_config",
    "scalar_p0_amgx_config",
    "solve_pcgf_prototype",
    "solve_pcg_prototype",
    "symmetric_scalar_amgx_config",
]
