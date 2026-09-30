"""Production-oriented CUDA GMRES interface.

This module keeps the experimental kernels and numerical choices explicit while
providing conservative defaults for real HDG solves.  It wraps the low-level
restarted GMRES implementation with:

* finite-value checks;
* exact true-residual replacement at every restart boundary;
* restart-cycle stagnation and divergence detection;
* automatic CGS-to-CGS2 fallback when the measured Arnoldi basis loses too
  much orthogonality;
* reusable GMRES storage; and
* an optional exception on unsuccessful termination.

Operator/preconditioner construction and persistent architecture autotuning are
kept separate.  This makes the solver useful immediately with any object that
implements the device operator protocols in :mod:`hdgfem.linalg.gpu.gmres`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Literal

import numpy as np

from hdgfem.runtime.optional import require_cupy_device
from hdgfem.linalg.gpu.gmres import (
    CuPyGMRESResult,
    CuPyGMRESWorkspace,
    CuPyOrthogonalization,
    CuPyRestartedGMRESSolver,
    DeviceMatvecOperator,
    DevicePreconditioner,
    _validate_restart_parameters,
    _validate_robustness_parameters,
)

FallbackThreshold = float | Literal["auto"] | None


@dataclass(frozen=True)
class CuPyProductionGMRESOptions:
    """Numerical and safety settings for production GPU GMRES.

    The defaults retain the fast one-pass CGS implementation.  When
    ``cgs2_fallback_threshold='auto'``, the solver measures the Arnoldi Gram
    matrix once per restart cycle while CGS is active and switches subsequent
    cycles to CGS2 when the maximum off-diagonal entry exceeds a dtype-aware
    threshold.  The automatic thresholds are ``1e-8`` for float64 and
    ``1e-3`` for float32.
    """

    restart: int = 75
    max_iterations: int | None = 2000
    rtol: float = 1.0e-8
    atol: float = 0.0
    orthogonalization: CuPyOrthogonalization = "cgs"
    breakdown_tolerance: float | None = None
    check_finite: bool = True
    stagnation_cycles: int | None = 8
    stagnation_tolerance: float = 1.0e-3
    divergence_factor: float = 1.0e6
    cgs2_fallback_threshold: FallbackThreshold = "auto"
    raise_on_failure: bool = False

    def with_overrides(self, **overrides: Any) -> "CuPyProductionGMRESOptions":
        """Return a validated-field-name copy with selected changes."""

        valid = {item.name for item in fields(type(self))}
        unknown = sorted(set(overrides) - valid)
        if unknown:
            raise TypeError(
                "unknown production GMRES option(s): " + ", ".join(unknown)
            )
        return replace(self, **overrides)

    def resolved_fallback_threshold(self, dtype: Any) -> float | None:
        """Return the validated orthogonality fallback threshold."""
        value = self.cgs2_fallback_threshold
        if value is None or self.orthogonalization != "cgs":
            return None
        if value == "auto":
            host_dtype = np.dtype(dtype)
            if host_dtype == np.dtype(np.float64):
                return 1.0e-8
            if host_dtype == np.dtype(np.float32):
                return 1.0e-3
            raise TypeError("production GPU GMRES supports float32 and float64 only")
        return float(value)

    def validate(self, *, num_dofs: int, dtype: Any) -> None:
        """Validate shapes, dtypes, devices, and solver parameters."""
        _validate_restart_parameters(
            restart=self.restart,
            max_iterations=self.max_iterations,
            num_dofs=num_dofs,
            rtol=self.rtol,
            atol=self.atol,
            breakdown_tolerance=self.breakdown_tolerance,
        )
        _validate_robustness_parameters(
            check_finite=self.check_finite,
            stagnation_cycles=self.stagnation_cycles,
            stagnation_tolerance=self.stagnation_tolerance,
            divergence_factor=self.divergence_factor,
            cgs2_fallback_threshold=self.resolved_fallback_threshold(dtype),
        )
        if self.orthogonalization not in ("mgs", "mgs2", "cgs", "cgs2"):
            raise ValueError(
                "orthogonalization must be one of mgs, mgs2, cgs, or cgs2"
            )
        if (
            self.orthogonalization != "cgs"
            and self.cgs2_fallback_threshold not in (None, "auto")
        ):
            raise ValueError(
                "cgs2_fallback_threshold is only valid with "
                "orthogonalization='cgs'"
            )


class CuPyGMRESFailure(RuntimeError):
    """Raised when a production solve is configured to fail loudly."""

    def __init__(self, result: CuPyGMRESResult) -> None:
        """Initialize the instance and validate persistent storage."""
        self.result = result
        super().__init__(
            f"GPU GMRES terminated with status={result.status!r} after "
            f"{result.iterations} iterations: {result.termination_reason}; "
            f"relative residual={result.relative_residual:.3e}"
        )


class CuPyProductionGMRESSolver:
    """Reusable robust GPU GMRES solver for sequential HDG systems."""

    def __init__(
        self,
        operator: DeviceMatvecOperator,
        *,
        preconditioner: DevicePreconditioner | None = None,
        options: CuPyProductionGMRESOptions | None = None,
        workspace: CuPyGMRESWorkspace | None = None,
    ) -> None:
        """Initialize the instance and validate persistent storage."""
        cp = require_cupy_device()
        self.operator = operator
        self.preconditioner = preconditioner
        self.options = options or CuPyProductionGMRESOptions()
        dtype = cp.dtype(operator.dtype)
        self.options.validate(num_dofs=int(operator.num_dofs), dtype=dtype.name)
        fallback = self.options.resolved_fallback_threshold(dtype.name)

        self._solver = CuPyRestartedGMRESSolver(
            operator,
            restart=self.options.restart,
            max_iterations=self.options.max_iterations,
            rtol=self.options.rtol,
            atol=self.options.atol,
            preconditioner=preconditioner,
            orthogonalization=self.options.orthogonalization,
            breakdown_tolerance=self.options.breakdown_tolerance,
            check_finite=self.options.check_finite,
            stagnation_cycles=self.options.stagnation_cycles,
            stagnation_tolerance=self.options.stagnation_tolerance,
            divergence_factor=self.options.divergence_factor,
            cgs2_fallback_threshold=fallback,
            workspace=workspace,
        )

    @property
    def workspace(self) -> CuPyGMRESWorkspace:
        """Return the persistent GMRES workspace."""
        return self._solver.workspace

    @property
    def workspace_device_bytes(self) -> int:
        """Return persistent device-workspace storage in bytes."""
        return self._solver.workspace_device_bytes

    @property
    def resolved_cgs2_fallback_threshold(self) -> float | None:
        """Return the validated CGS2 fallback threshold."""
        return self._solver.cgs2_fallback_threshold

    def solve(
        self,
        rhs: Any,
        *,
        x0: Any | None = None,
        solution_out: Any | None = None,
        profiler: Any | None = None,
        monitor_orthogonality: bool = False,
        raise_on_failure: bool | None = None,
    ) -> CuPyGMRESResult:
        """Solve one system and optionally raise on unsuccessful termination."""

        result = self._solver.solve(
            rhs,
            x0=x0,
            solution_out=solution_out,
            profiler=profiler,
            monitor_orthogonality=monitor_orthogonality,
        )
        should_raise = (
            self.options.raise_on_failure
            if raise_on_failure is None
            else bool(raise_on_failure)
        )
        if should_raise and not result.converged:
            raise CuPyGMRESFailure(result)
        return result


__all__ = [
    "CuPyGMRESFailure",
    "CuPyProductionGMRESOptions",
    "CuPyProductionGMRESSolver",
]
