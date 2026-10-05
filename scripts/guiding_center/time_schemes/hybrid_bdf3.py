"""Shared third-order startup, histories, stage guesses and timing for H1/H2.

H1 predicts density with AB3 before one BDF3 transport solve. H2 predicts
with extrapolated-drift BDF3, updates Poisson, then corrects with BDF3.
"""
from __future__ import annotations

import math
from time import perf_counter

from hybridge.runtime.logging import timed_call
from .recovery import (
    StepRecoveryWork, with_poisson_recovery, recovery_options, poisson_tau,
)
from scripts.guiding_center.poisson.poisson_recovery import PoissonStageFailure
from hybridge.core.field_ops import (
    perpendicular_vector_field,
    solution_field,
    solution_trace,
)


def bdf3_source(space, densities, dt):
    """Return the BDF3 history source and drift scale for constant positive dt."""
    if len(densities) < 3:
        raise ValueError("BDF3 requires three accepted densities")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("BDF3 requires a finite positive constant dt")
    source = (18/11) * densities[0] - (9/11) * densities[1] + (2/11) * densities[2]
    source.name = "rho_bdf3_source_h"
    return source, 6*dt/11


def bdf3_predictor_data(space, densities, dt):
    """Build a third-order endpoint initial guess and the shared BDF3 RHS."""
    source, scale = bdf3_source(space, densities, dt)
    predictor = 3. * densities[0] - 3. * densities[1] + densities[2]
    predictor.name = "rho_extrapolated_guess_h"
    return predictor, source, scale


def ab3_predict(space, densities, residuals, dt):
    """Use newest-first accepted histories, with the physical (unscaled) RHS."""
    if len(densities) < 3 or len(residuals) < 3:
        raise ValueError("H1-BDF3 requires three accepted densities and residuals")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("H1-BDF3 requires a finite positive constant dt")
    predictor = (densities[0] + (23*dt/12) * residuals[0]
                 - (16*dt/12) * residuals[1] + (5*dt/12) * residuals[2])
    predictor.name = "rho_ab3_predictor_h"
    source, scale = bdf3_source(space, densities, dt)
    return predictor, source, scale


from scripts.guiding_center.time_schemes.stage_support import closest_trace, GuidingCenterStep as HybridBDF3Step


class HybridBDF3Stepper:
    """Own accepted history and reusable trace/device or explicit-residual data.

    Solver callbacks receive explicit stage guesses. Histories are committed
    only after endpoint transport, Poisson and optional residual work succeeds.
    Predictor states never enter the multistep history.
    """

    scheme = None

    def __init__(self, space, dt, density, poisson_result, residual, *,
                 density_boundary, potential_boundary, phase_verbosity=0, detail_verbosity=0,
                 startup_method="si-euler-extrap3", poisson_solver=None,
                 poisson_tau_retry_factor=2., poisson_tau_max_retries=4,
                 recovery_verbosity=0, recovery_record=None):
        """Initialize owning accepted histories and the shared trace/residual workspace."""
        if self.scheme not in {"h1-bdf3", "h2-bdf3"}:
            raise ValueError("use H1BDF3Stepper or H2BDF3Stepper")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError(f"{self.scheme} requires a finite positive constant dt")
        if startup_method not in {"si-euler-extrap3", "ssprk3"}:
            raise ValueError("unknown hybrid BDF3 startup method")
        self.startup_method = startup_method
        self.space, self.dt, self.time = space, float(dt), 0.0
        self.residual = residual
        self.recovery_options = recovery_options(poisson_tau_retry_factor, poisson_tau_max_retries,
                                                 recovery_verbosity, recovery_record)
        self.phase_verbosity, self.detail_verbosity = phase_verbosity, detail_verbosity
        self.density_boundary, self.potential_boundary = density_boundary, potential_boundary
        start = perf_counter()
        self.densities = [density.copy(name="rho_history_h")]
        self.potential_trace = solution_trace(poisson_result, space, reduced=False).copy()
        work = StepRecoveryWork(self, poisson_solver)
        work.accepted_start(self.densities[0], self.potential_trace)
        self.drift = perpendicular_vector_field(poisson_result.flux, 1, space)
        self.uses_residual_history = self.scheme == "h1-bdf3"
        self.initial_residual_count = int(self.uses_residual_history or startup_method == "ssprk3")
        if self.initial_residual_count:
            while True:
                try:
                    initial_rhs, self.density_trace = work.evaluate(self.densities[0], self.drift, 0.)
                    break
                except PoissonStageFailure as failure:
                    if poisson_solver is None:
                        raise failure.error from failure
                    poisson_result, self.potential_trace = work.repair(failure)
                    self.drift = perpendicular_vector_field(poisson_result.flux, 1, space)
            self.residuals = [initial_rhs]
        else:
            self.residuals = []
        # Prime the existing projection cache and establish device residency.
        projected = residual.project_trace(density)
        if not self.initial_residual_count:
            self.density_trace = projected
        self.drifts = [self.drift]
        self.potential_trace = solution_trace(poisson_result, space, reduced=False).copy()
        residual.synchronize()
        self.setup_time = perf_counter()-start
        self.initial_poisson_result = poisson_result
        self.initial_poisson_retry_wall_time = work.poisson_wall
        self.initial_recovery_metrics = work.metrics()
        self.initial_residual_count = work.residual_count
        self.accepted_poisson_tau = poisson_tau(poisson_solver)

    def _timed(self, label, verbosity, function):
        """Use the runner's output helper and include completed device work."""
        def synchronized_call():
            """Finish pending device work inside the measured interval."""
            value = function()
            self.residual.synchronize()
            return value
        return timed_call(f"[gc:{self.scheme}] {label}", verbosity, synchronized_call)

    def _rebuild_poisson_history(self, work):
        """Refresh all drift/residual histories without changing accepted densities."""
        drifts, residuals = [], []
        need_rhs = self.uses_residual_history or self.startup_method == "ssprk3"
        for index, density in enumerate(self.densities):
            stage_time = self.time-index*self.dt
            result, trace = work.poisson(density, stage_time, "rebuild accepted history",
                                         guess=self.potential_trace)
            drift = perpendicular_vector_field(result.flux, 1., self.space)
            drifts.append(drift)
            if need_rhs:
                rhs, density_trace = work.evaluate(density, drift, stage_time)
                residuals.append(rhs)
                if index == 0:
                    self.density_trace = density_trace
            if index == 0:
                potential_trace = trace
        self.drift = drifts[0]
        self.drifts = drifts if not self.uses_residual_history else drifts[:1]
        self.residuals = residuals
        self.potential_trace = potential_trace
        work.synchronize()

    @with_poisson_recovery
    def advance(self, poisson_solver, solve_transport, *, endpoint_postprocess=None):
        """Advance using linear solves; startup passes stage_time/stage keywords."""
        space, dt, t = self.space, self.dt, self.time
        endpoint = t+dt
        startup = len(self.densities) < 3
        poisson_results, transport_results = [], []
        candidates = [(t, self.potential_trace)]
        guess_times, stage_times, stage_labels, stage_wall_times = [], [], [], []
        poisson_wall = transport_wall = beta_time = residual_time = 0.0
        residual_count = 0
        prediction_time = trace_projection_time = 0.0
        transport_guess_time = None
        transport_stage_times, transport_guess_times, transport_labels = [], [], []

        def poisson(density, stage_time, label, *, final=False):
            """Solve one stage with the nearest trace guess and record its cost."""
            nonlocal poisson_wall
            self.residual.synchronize()
            guess, guess_time = closest_trace(candidates, stage_time)

            def solve():
                """Solve the stage Poisson equation and retain an owning trace."""
                return self._recovery_work.poisson(
                    density, stage_time, label, guess=guess, final=final, guess_time=guess_time)

            (result, trace), elapsed = self._timed(
                f"{label} Poisson t={stage_time:.6f} guess_t={guess_time:.6f}",
                self.phase_verbosity, solve,
            )
            candidates.append((stage_time, trace))
            guess_times.append(guess_time)
            stage_times.append(stage_time)
            stage_labels.append(label)
            stage_wall_times.append(elapsed)
            poisson_results.append(result)
            poisson_wall += elapsed
            return result, trace, guess

        def rhs(density, result, stage_time):
            """Evaluate physical drift and explicit residual at the requested stage time."""
            nonlocal residual_time, residual_count, beta_time
            beta, elapsed = self._timed(
                f"drift t={stage_time:.6f}", self.detail_verbosity,
                lambda: perpendicular_vector_field(result.flux, 1, space),
            )
            beta_time += elapsed
            (value, trace), elapsed = self._timed(
                f"explicit HDG residual t={stage_time:.6f}", self.detail_verbosity,
                lambda: self._recovery_work.evaluate(density, beta, stage_time),
            )
            residual_time += elapsed
            residual_count += 1
            return value, trace, beta

        current = self.densities[0]
        current_rhs = self.residuals[0] if self.residuals else None
        transport_result = source = beta = transport_guess = None
        scale = 0.0
        if startup and self.startup_method == "ssprk3":
            # SSPRK3 stage times are t, t+dt, t+dt/2. Each stage obtains
            # its own Poisson field and time-dependent density boundary.
            y1 = current + dt * current_rhs
            p1, _, _ = poisson(y1, endpoint, "SSPRK3 stage 1")
            f1, _, _ = rhs(y1, p1, endpoint)
            y2 = .75 * current + .25 * y1 + (.25*dt) * f1
            p2, _, _ = poisson(y2, t+.5*dt, "SSPRK3 stage 2")
            f2, _, _ = rhs(y2, p2, t+.5*dt)
            density = (1/3) * current + (2/3) * y2 + (2*dt/3) * f2
            density.name = "rho_h"
        elif startup:
            # Richardson extrapolation of the existing first-order SI Euler:
            # independent paths with 1, 2 and 3 substeps over the same interval.
            # The weights cancel the 1/m and 1/m^2 endpoint error terms.
            endpoints = []
            density_candidates = [(t, current)]
            for substeps in (1, 2, 3):
                self._recovery_work.checkpoint = self._recovery_work.start_checkpoint
                branch_density, branch_drift = current, self.drift
                scale = dt/substeps
                for substep in range(1, substeps+1):
                    stage_time = endpoint if substep == substeps else t+substep*scale
                    label = f"SI-Euler extrap3 path {substeps} stage {substep}"
                    source = branch_density
                    beta, elapsed = self._timed(
                        f"{label} scaled drift", self.detail_verbosity,
                        lambda: scale * branch_drift,
                    )
                    beta_time += elapsed
                    start = perf_counter()
                    guess_density, transport_guess_time = closest_trace(density_candidates, stage_time)
                    transport_guess, elapsed = self._timed(
                        f"{label} transport guess", self.detail_verbosity,
                        lambda: self.residual.project_trace(guess_density),
                    )
                    trace_projection_time += elapsed
                    transport_result, _ = self._timed(
                        f"{label} transport t={stage_time:.6f} guess_t={transport_guess_time:.6f}",
                        self.phase_verbosity,
                        lambda: solve_transport(source, beta, transport_guess, scale,
                                                stage_time=stage_time, stage=label),
                    )
                    # Solver buffers may be reused by another path. Retain
                    # owning branch endpoints and nearest-time guess states.
                    branch_density = solution_field(transport_result, space).copy(
                        name="rho_startup_stage_h")
                    self.residual.synchronize()
                    transport_wall += perf_counter()-start
                    transport_results.append(transport_result)
                    transport_stage_times.append(stage_time)
                    transport_guess_times.append(transport_guess_time)
                    transport_labels.append(label)
                    density_candidates.append((stage_time, branch_density))
                    branch_poisson, _, _ = poisson(branch_density, stage_time, label)
                    branch_drift, elapsed = self._timed(
                        f"{label} drift", self.detail_verbosity,
                        lambda: perpendicular_vector_field(branch_poisson.flux, 1, space),
                    )
                    beta_time += elapsed
                endpoints.append(branch_density)
            density, prediction_time = self._timed(
                "third-order SI-Euler endpoint extrapolation", self.detail_verbosity,
                lambda: .5 * endpoints[0] - 4. * endpoints[1] + 4.5 * endpoints[2],
            )
            density.name = "rho_extrap3_h"
        elif self.uses_residual_history:
            (predictor, source, scale), prediction_time = self._timed(
                "AB3 endpoint prediction and BDF3 source", self.detail_verbosity,
                lambda: ab3_predict(space, self.densities, self.residuals, dt),
            )
            predicted_poisson, _, _ = poisson(predictor, endpoint, "AB3 predictor")
            beta, elapsed = self._timed(
                "BDF3 scaled drift", self.detail_verbosity,
                lambda: perpendicular_vector_field(predicted_poisson.flux, scale, space),
            )
            beta_time += elapsed
            start = perf_counter()
            # The AB3 density is already at the target time. Project it with
            # the actual trace basis instead of reusing the previous trace.
            transport_guess, trace_projection_time = self._timed(
                "projecting AB3 endpoint transport guess", self.detail_verbosity,
                lambda: self.residual.project_trace(predictor),
            )
            transport_result, _ = self._timed(
                f"BDF3 transport t={endpoint:.6f} guess_t={endpoint:.6f}",
                self.phase_verbosity, lambda: solve_transport(source, beta, transport_guess, scale),
            )
            density = solution_field(transport_result, space, name="rho_h")
            transport_results.append(transport_result)
            transport_guess_time = endpoint
            transport_stage_times.append(endpoint)
            transport_guess_times.append(endpoint)
            transport_labels.append("BDF3")
            self.residual.synchronize()
            transport_wall += perf_counter()-start

        else:
            # H2: the predictor and corrector solve the same BDF3 density
            # equation, first with extrapolated drift, then with the drift
            # of the predicted endpoint. No explicit transport residual.
            (predictor, source, scale), prediction_time = self._timed(
                "extrapolated endpoint guess and BDF3 source", self.detail_verbosity,
                lambda: bdf3_predictor_data(space, self.densities, dt),
            )
            beta, elapsed = self._timed(
                "BDF3 predictor extrapolated drift", self.detail_verbosity,
                lambda: ((3*scale) * self.drifts[0] - (3*scale) * self.drifts[1]
                         + scale * self.drifts[2]),
            )
            beta_time += elapsed
            start = perf_counter()
            transport_guess, trace_projection_time = self._timed(
                "projecting extrapolated endpoint transport guess", self.detail_verbosity,
                lambda: self.residual.project_trace(predictor),
            )
            predicted_transport, _ = self._timed(
                f"BDF3 predictor transport t={endpoint:.6f} guess_t={endpoint:.6f}",
                self.phase_verbosity,
                lambda: solve_transport(source, beta, transport_guess, scale,
                                        stage_time=endpoint, stage="BDF3 predictor"),
            )
            predicted_density = solution_field(predicted_transport, space).copy(
                name="rho_bdf3_predictor_h")
            # The actual predictor trace is at the same endpoint. Detach it
            # before another solve can overwrite the solver's work buffers.
            transport_guess = solution_trace(predicted_transport, space, reduced=True).copy()
            self.residual.synchronize()
            transport_wall += perf_counter()-start
            transport_results.append(predicted_transport)
            predicted_poisson, _, _ = poisson(predicted_density, endpoint, "BDF3 predictor")
            beta, elapsed = self._timed(
                "BDF3 corrector scaled drift", self.detail_verbosity,
                lambda: perpendicular_vector_field(predicted_poisson.flux, scale, space),
            )
            beta_time += elapsed
            transport_result, elapsed = self._timed(
                f"BDF3 corrector transport t={endpoint:.6f} guess_t={endpoint:.6f}",
                self.phase_verbosity,
                lambda: solve_transport(source, beta, transport_guess, scale,
                                        stage_time=endpoint, stage="BDF3 corrector"),
            )
            transport_wall += elapsed
            density = solution_field(transport_result, space, name="rho_h")
            transport_results.append(transport_result)
            transport_guess_time = endpoint
            transport_stage_times.extend([endpoint, endpoint])
            transport_guess_times.extend([endpoint, endpoint])
            transport_labels.extend(["BDF3 predictor", "BDF3 corrector"])

        result, potential_trace, poisson_guess = poisson(density, endpoint, "accepted endpoint", final=True)
        if self.uses_residual_history or (startup and self.startup_method == "ssprk3"):
            accepted_rhs, residual_trace, accepted_drift = rhs(density, result, endpoint)
        else:
            accepted_rhs = None
            accepted_drift, elapsed = self._timed(
                "accepted endpoint drift", self.detail_verbosity,
                lambda: perpendicular_vector_field(result.flux, 1, space),
            )
            beta_time += elapsed
            if startup:
                residual_trace, elapsed = self._timed(
                    "projecting extrapolated startup endpoint trace", self.detail_verbosity,
                    lambda: self.residual.project_trace(density),
                )
                trace_projection_time += elapsed
        density_trace = (residual_trace if startup else
                         solution_trace(transport_result, space, reduced=True).copy())
        accepted_density = density.copy(name="rho_h")
        # Complete all device work before committing any accepted history.
        self.residual.synchronize()
        self.densities = [accepted_density, *self.densities[:2]]
        self.residuals = ([accepted_rhs, *self.residuals[:2]] if accepted_rhs is not None else [])
        self.drifts = [accepted_drift, *self.drifts[:2]] if not self.uses_residual_history else [accepted_drift]
        self.drift = accepted_drift
        self.potential_trace = potential_trace
        self.density_trace = density_trace
        self.time = endpoint
        return HybridBDF3Step(
            density=accepted_density, density_trace=density_trace, poisson_result=result,
            potential_trace=potential_trace, transport_result=transport_result,
            transport_source=source, transport_beta=beta, transport_initial_guess=transport_guess,
            poisson_initial_guess=poisson_guess, transport_results=transport_results,
            poisson_results=poisson_results, transport_wall_time=transport_wall,
            poisson_wall_time=poisson_wall, beta_build_time=beta_time,
            metrics={
                f"{self.scheme.replace('-', '_')}_startup": startup,
                f"{self.scheme.replace('-', '_')}_startup_method": self.startup_method if startup else None,
                "transport_time_order": 3, "transport_beta_scale": scale,
                "explicit_residual_count": residual_count, "explicit_residual_time": residual_time,
                "explicit_residual_backend": self.residual.backend,
                f"{self.scheme.replace('-', '_')}_history_count": len(self.densities),
                "density_predictor_time": prediction_time,
                "transport_trace_projection_time": trace_projection_time,
                "poisson_stage_times": stage_times,
                "poisson_stage_labels": stage_labels,
                "poisson_stage_wall_times": stage_wall_times,
                "poisson_stage_initial_guess_times": guess_times,
                "poisson_initial_guess_time": guess_times[-1],
                "transport_stage_times": transport_stage_times,
                "transport_stage_initial_guess_times": transport_guess_times,
                "transport_stage_labels": transport_labels,
                "transport_initial_guess_time": transport_guess_time,
                "transport_initial_guess_kind": ("closest-stage-density-projection" if startup else
                                                 ("ab3-endpoint-projection" if self.uses_residual_history else
                                                  "bdf3-predictor-trace")) if transport_results else None,
            },
        )
