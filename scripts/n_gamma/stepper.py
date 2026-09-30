"""Decoupled n-Gamma D-BDF2 stepper in a poloidal plane (plan: docs/development/plans/n_gamma_d_bdf2.md).

``geometry`` is required: ``"cartesian"`` solves the poloidal-plane model on
``(x, y)`` with the plain divergence (weight ``W = 1``); ``"axisymmetric"``
solves the toroidally symmetric model on ``(R, Z)`` multiplied by ``W = R``.
One step performs exactly two linear scalar HDG ADR solves with two reusable
:class:`hdgfem.AdvectionDiffusionReactionHDGSolver` objects:

1. density ``n^{k+1}`` with reaction ``W*alpha``, frozen advection
   ``W*u* b_p`` and source ``W*(S_n + h_n)``;
2. momentum ``Gamma^{k+1}`` with the same advection and reaction and source
   ``W*(S_Gamma + h_Gamma - c_s^2 b_p . grad n_h^{k+1})`` from the new density.

``alpha``, the histories ``h_w`` and the extrapolations are

    Euler startup: alpha = 1/dt,      h_w = w^k/dt,                  n* = n^k
    BDF2:          alpha = 3/(2 dt),  h_w = (4 w^k - w^{k-1})/(2 dt), n* = 2 n^k - n^{k-1}

and likewise for ``Gamma*``. Omitting the previous pair selects one Euler
startup step; afterwards BDF2 is used. A failed linear solve is retried once
with ``fallback_options`` (for example a stronger AMGX config) at the same
step and ``dt``, and the retry is recorded. Fields, traces, time and history are
committed only after both solves succeed with finite results; otherwise
:class:`StepRejected` is raised and the state is unchanged. Sources and
boundary data come from factories evaluated at the new time and must not
depend on the timestep or numerical history; they take mesh coordinates.
On the host path, ``compiled`` (a
:class:`scripts.n_gamma.compiled.CompiledCoefficients` of the same case)
replaces the NumPy coefficients and source factories by Numba-compiled ones.
Returned ADR fluxes carry the weight ``W``. There are no Newton/Picard iterations.
"""
from __future__ import annotations

from dataclasses import dataclass
import time as _time
from typing import Any, Callable

import numpy as np

from hdgfem import AdvectionDiffusionReactionHDGSolver, DGField, DGSpace, field_linear_combination

from . import coefficients as nc
from .diagnostics import sampled_minimum

SourceFactory = Callable[[float], Callable]


@dataclass(frozen=True)
class SolveSummary:
    """Reduced linear-solve outcome of one ADR solve."""

    status: str | None
    iterations: int | None
    relative_residual: float | None
    fallback: bool = False
    setup_reused: bool = False              # AMGX setup/preconditioner kept from an earlier solve
    analysis_reused: bool = False           # PARDISO reordering/analysis kept from an earlier solve
    static_reused: bool = False             # diffusion data, stabilization, pattern (and mass factors)
    reconstruction_reused: bool = False     # reconstruction from assembly factors/columns


@dataclass(frozen=True)
class NGammaStepDiagnostics:
    """Once-per-step record: stage, floor clamps, solves, sampled minimum and timings."""

    step: int
    time: float
    dt: float
    stage: str
    alpha: float
    floor: nc.DensityFloorReport
    density_solve: SolveSummary | None = None
    momentum_solve: SolveSummary | None = None
    min_density_sampled: float | None = None
    density_seconds: float = 0.
    momentum_seconds: float = 0.
    total_seconds: float = 0.
    accepted: bool = True
    reason: str | None = None


@dataclass(frozen=True)
class NGammaStepResult:
    """Both ADR results of an accepted step and its diagnostics."""

    density: Any
    momentum: Any
    diagnostics: NGammaStepDiagnostics


class StepRejected(RuntimeError):
    """A failed solve or nonfinite state; the stepper state was not modified."""

    def __init__(self, reason: str, diagnostics: NGammaStepDiagnostics | None):
        super().__init__(reason)
        self.diagnostics = diagnostics


def _summary(result, fallback: bool = False) -> SolveSummary:
    """Reduce an ADR result to its solve outcome and the caches it reused."""
    solve = getattr(result, "global_solve_result", None)
    details = (result.timings.details or {}) if result is not None else {}
    flag = lambda *keys: any(details.get(key) == 1. for key in keys)
    cached = dict(analysis_reused=flag("host.pardiso_analysis_reused"),
                  static_reused=flag("host.static_reused", "raw.coefficients.diffusion.cached"),
                  reconstruction_reused=flag("numba.local_columns_reused", "raw.reconstruction.cached_factors"))
    if solve is None:
        return SolveSummary(None, None, None, fallback, **cached)
    relative = next((value for value in (solve.physical_relative_residual_norm, solve.relative_residual_norm)
                     if value is not None), None)
    return SolveSummary(solve.status, solve.iteration_count, None if relative is None else float(relative), fallback,
                        bool(getattr(solve, "amgx_preconditioner_reused", False)), **cached)


def _finite(field: DGField, device: bool) -> bool:
    if field.constant_value is not None:
        return bool(np.isfinite(field.constant_value))
    if device and field.device_coefficients_materialized():
        import cupy as cp
        return bool(cp.isfinite(field._first_device_coefficients()).all())
    return bool(np.isfinite(field.coeffs).all())


class NGammaBDF2Stepper:
    """Advance ``(n, Gamma)`` with the decoupled semi-implicit D-BDF2 scheme."""

    def __init__(
            self,
            space: DGSpace,
            density: DGField,
            momentum: DGField,
            *,
            dt: float,
            time: float,
            geometry: str,
            b_poloidal: Callable,
            diffusion: float,
            viscosity: float,
            density_floor: float,
            source_density: SourceFactory,
            source_momentum: SourceFactory,
            boundary_density: SourceFactory,
            boundary_momentum: SourceFactory,
            sound_speed: float = 1.,
            previous_density: DGField | None = None,
            previous_momentum: DGField | None = None,
            options: dict[str, Any] | None = None,
            density_options: dict[str, Any] | None = None,
            momentum_options: dict[str, Any] | None = None,
            fallback_options: dict[str, Any] | None = None,
            compiled=None,
    ):
        if not float(dt) > 0.:
            raise ValueError("dt must be positive")
        if (previous_density is None) != (previous_momentum is None):
            raise ValueError("previous_density and previous_momentum must be given together")
        self.space = space
        self.dt = float(dt)
        self.time = float(time)
        self.geometry = nc.check_geometry(geometry)
        self.b_poloidal = b_poloidal
        self.sound_speed = float(sound_speed)
        self.density_floor = float(density_floor)
        if not self.density_floor > 0.:
            raise ValueError("density_floor must be positive")
        self.sources = (source_density, source_momentum)
        self.boundaries = (boundary_density, boundary_momentum)
        base = dict(hdg_postprocess="none", verbose=False)
        base.update(options or {})
        self.device = str(base.get("assembly_backend", "numba")).replace("_", "-") == "raw-cuda"
        self.trace_space = space.trace_space(base.get("trace_basis", "legacy-lagrange"))
        self.compiled = None if self.device else compiled
        if self.compiled is not None and self.compiled.geometry != self.geometry:
            raise ValueError("compiled coefficients belong to a different geometry")
        tensor = (self.compiled.diffusion_tensor if self.compiled is not None
                  else lambda value: nc.diffusion_tensor(value, b_poloidal, geometry=geometry))
        density_settings = {**base, "diffusion": tensor(diffusion), **(density_options or {})}
        momentum_settings = {**base, "diffusion": tensor(viscosity), **(momentum_options or {})}
        self.solvers = (AdvectionDiffusionReactionHDGSolver(space, **density_settings),
                        AdvectionDiffusionReactionHDGSolver(space, **momentum_settings))
        self.current = (self._resident(density), self._resident(momentum))
        self.previous = None if previous_density is None else (
            self._resident(previous_density), self._resident(previous_momentum))
        self.fallback_options = dict(fallback_options or {})
        self.traces = (None, None)
        self.step_count = 0

    def close(self) -> None:
        """Release the persistent solver state (cached AMGX solvers) of both equations."""
        for solver in self.solvers:
            solver.close()

    def _resident(self, field: DGField) -> DGField:
        """Cache a device copy so combinations and evaluations stay on the GPU."""
        if self.device and field.constant_value is None and not field.device_coefficients_materialized():
            from hdgfem.core.device import as_cupy_coefficients, as_cupy_space
            as_cupy_coefficients(field, as_cupy_space(field.space))
        return field

    @property
    def stage(self) -> str:
        """``"euler"`` until one previous level exists, then ``"bdf2"``."""
        return "euler" if self.previous is None else "bdf2"

    def _solve(self, solver):
        """Solve; after a failure retry once with ``fallback_options`` (same step and dt), then restore."""
        try:
            return solver.solve(), False
        except Exception:  # noqa: BLE001 - any solve failure may use the fallback
            if not self.fallback_options:
                raise
        saved = {key: getattr(solver.options, key) for key in self.fallback_options}
        try:
            return solver.solve(**self.fallback_options), True
        finally:
            solver.with_options(**saved)

    def _density_source(self, history, t_new):
        if self.compiled is not None:
            return self.compiled.density_source(history, t_new)
        return nc.density_source(self.space, self.sources[0](t_new), history, geometry=self.geometry)

    def _momentum_source(self, history, density, t_new):
        if self.compiled is not None:
            return self.compiled.momentum_source(history, density, t_new)
        return nc.momentum_source(self.space, self.sources[1](t_new), history, density, self.b_poloidal,
                                  self.sound_speed, geometry=self.geometry)

    def _combine(self, terms, name):
        return field_linear_combination(self.space, terms, name=name)

    def advance(self, *, postprocess: str | None = None) -> NGammaStepResult:
        """Take one step; raise :class:`StepRejected` without changing state on failure.

        ``postprocess`` (``"primal"``, ``"flux"`` or ``"both"``) enables HDG
        post-processing for this step only, for example the final step of a run.
        """
        if postprocess is None or postprocess == "none":
            return self._advance()
        saved = [solver.options.hdg_postprocess for solver in self.solvers]
        for solver in self.solvers:
            solver.with_options(hdg_postprocess=postprocess)
        try:
            return self._advance()
        finally:
            for solver, value in zip(self.solvers, saved):
                solver.with_options(hdg_postprocess=value)

    def _advance(self) -> NGammaStepResult:
        """One step with the current solver options."""
        start = _time.perf_counter()
        dt, t_new, stage = self.dt, self.time + self.dt, self.stage
        (n_k, gamma_k) = self.current
        if stage == "euler":
            alpha = 1./dt
            n_star, gamma_star = n_k, gamma_k
            h_n = self._combine([(1./dt, n_k)], "h_n")
            h_gamma = self._combine([(1./dt, gamma_k)], "h_Gamma")
        else:
            n_km1, gamma_km1 = self.previous
            alpha = 1.5/dt
            n_star = self._combine([(2., n_k), (-1., n_km1)], "n_star")
            gamma_star = self._combine([(2., gamma_k), (-1., gamma_km1)], "Gamma_star")
            h_n = self._combine([(2./dt, n_k), (-.5/dt, n_km1)], "h_n")
            h_gamma = self._combine([(2./dt, gamma_k), (-.5/dt, gamma_km1)], "h_Gamma")
        step = self.step_count + 1

        def rejected(reason, floor=None, **fields):
            diagnostics = None if floor is None else NGammaStepDiagnostics(
                step, t_new, dt, stage, alpha, floor, accepted=False, reason=reason,
                total_seconds=_time.perf_counter() - start, **fields)
            return StepRejected(reason, diagnostics)

        try:
            floor = nc.density_floor_diagnostics(n_star, self.space, self.trace_space, self.density_floor,
                                                 device=self.device)
        except FloatingPointError as error:
            raise rejected(str(error)) from error
        if self.compiled is not None:
            beta = self.compiled.advection(n_star, gamma_star)
        else:
            beta = nc.advection(self.space, n_star, gamma_star, self.b_poloidal, self.density_floor,
                                geometry=self.geometry)
        reaction = nc.reaction(alpha, geometry=self.geometry)  # sampled to arrays before any kernel
        density_solver, momentum_solver = self.solvers
        try:
            density_start = _time.perf_counter()
            density_solver.set_problem(self._density_source(h_n, t_new), beta, reaction, self.boundaries[0](t_new))
            density_result, density_fallback = self._solve(density_solver)
            density_seconds = _time.perf_counter() - density_start
        except Exception as error:  # noqa: BLE001 - any solve failure rejects the step
            raise rejected(f"density solve failed: {error}", floor) from error
        n_new = density_result.field
        if not _finite(n_new, self.device):
            raise rejected("density solution is not finite", floor,
                           density_solve=_summary(density_result, density_fallback))
        try:
            momentum_start = _time.perf_counter()
            momentum_solver.set_problem(self._momentum_source(h_gamma, n_new, t_new), beta, reaction,
                                        self.boundaries[1](t_new))
            momentum_result, momentum_fallback = self._solve(momentum_solver)
            momentum_seconds = _time.perf_counter() - momentum_start
        except Exception as error:  # noqa: BLE001
            raise rejected(f"momentum solve failed: {error}", floor,
                           density_solve=_summary(density_result, density_fallback),
                           density_seconds=density_seconds) from error
        gamma_new = momentum_result.field
        if not _finite(gamma_new, self.device):
            raise rejected("momentum solution is not finite", floor,
                           density_solve=_summary(density_result, density_fallback),
                           momentum_solve=_summary(momentum_result, momentum_fallback))
        diagnostics = NGammaStepDiagnostics(
            step, t_new, dt, stage, alpha, floor, _summary(density_result, density_fallback),
            _summary(momentum_result, momentum_fallback),
            sampled_minimum(n_new, self.space, self.trace_space, device=self.device),
            density_seconds, momentum_seconds, _time.perf_counter() - start)
        # Commit only now: both solves succeeded with finite fields.
        self.previous = self.current
        self.current = (n_new, gamma_new)
        self.traces = (density_result.trace, momentum_result.trace)
        self.time = t_new
        self.step_count = step
        return NGammaStepResult(density_result, momentum_result, diagnostics)


__all__ = ["NGammaBDF2Stepper", "NGammaStepDiagnostics", "NGammaStepResult", "SolveSummary", "StepRejected"]
