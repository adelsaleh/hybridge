"""Kennedy--Carpenter ARK3(2)4L[2]SA with a frozen upwind HDG operator.

The implicit split uses beta at the accepted start of the step. The explicit
split is the difference of complete semidiscrete HDG operators, including their
upwind stabilization and algebraic trace, evaluated at each stage density.
Coefficients: SUNDIALS ARKODE Butcher tables ARK324L2SA (ERK and DIRK).
"""
from __future__ import annotations

import math
from time import perf_counter
import numpy as np

from hdgfem.core.field_ops import (
    perpendicular_vector_field, solution_field, solution_trace, field_l2_norm,
)
from hdgfem.runtime.logging import timed_call
from scripts.guiding_center.time_schemes.stage_support import closest_trace, GuidingCenterStep
from scripts.guiding_center.poisson.poisson_recovery import (
    PoissonCheckpoint, PoissonStageRankFailure, PoissonTauRecovery,
    raise_for_poisson_rank_failure, raise_for_poisson_transport_failure,
)


GAMMA = 1767732205903/4055673282236
EXPLICIT = np.array([
    [0, 0, 0, 0],
    [1767732205903/2027836641118, 0, 0, 0],
    [5535828885825/10492691773637, 788022342437/10882634858940, 0, 0],
    [6485989280629/16251701735622, -4246266847089/9704473918619,
     10755448449292/10357097424841, 0],
])
IMPLICIT = np.array([
    [0, 0, 0, 0],
    [GAMMA, GAMMA, 0, 0],
    [2746238789719/10658868560708, -640167445237/6845629431997, GAMMA, 0],
    [1471266399579/7840856788654, -4482444167858/7529755066697,
     11266239266428/11593286722821, GAMMA],
])
WEIGHTS = IMPLICIT[-1].copy()
EMBEDDED_WEIGHTS = np.array([
    2756255671327/12835298489170, -10771552573575/22201958757719,
    9247589265047/10645013368117, 2193209047091/5459859503100,
])
ABSCISSAE = np.array([0, 2*GAMMA, 3/5, 1])
for _table in (EXPLICIT, IMPLICIT, WEIGHTS, EMBEDDED_WEIGHTS, ABSCISSAE):
    _table.flags.writeable = False


class IMEXARK3Stepper:
    """Three linear transport solves sharing one assembled/factored operator.

    ``solve_transport`` receives ``reuse_operator=True`` for stages 3 and 4.
    All stage fields/traces are owning values before solver buffers are reused.
    Accepted state is committed only after every solve and device operation.
    The embedded second-order estimate is diagnostic; dt remains fixed.
    """

    scheme = "imex-ark3"
    initial_residual_count = 1

    def __init__(self, space, dt, density, poisson_result, residual, *,
                 density_boundary, potential_boundary, phase_verbosity=0, detail_verbosity=0,
                 poisson_solver=None, poisson_tau_retry_factor=2., poisson_tau_max_retries=4,
                 recovery_verbosity=0, recovery_record=None, density_diagnostics=None):
        """Initialize accepted state and the work needed for bounded stage recovery."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("IMEX-ARK3 requires a finite positive dt")
        self.space, self.dt, self.time = space, float(dt), 0.
        self.residual = residual
        self.density_boundary, self.potential_boundary = density_boundary, potential_boundary
        self.phase_verbosity, self.detail_verbosity = phase_verbosity, detail_verbosity
        self.density_diagnostics = density_diagnostics
        self.recovery_options = dict(factor=poisson_tau_retry_factor,
            max_retries=poisson_tau_max_retries, verbosity=recovery_verbosity, record=recovery_record)
        start = perf_counter()
        self.density = density.copy(name="rho_h")
        self.potential_trace = solution_trace(poisson_result, space, reduced=False).copy()
        self.density_trace = None
        work = _ARKWork(self, poisson_solver)
        checkpoint = PoissonCheckpoint(self.density, self.potential_trace, 0., "initial state")
        try:
            self.rhs, self.density_trace, self.drift = work.evaluate(poisson_result, checkpoint)
        except PoissonStageRankFailure as failure:
            if poisson_solver is None:
                raise failure.error from failure
            poisson_result, checkpoint, values = work.repair(failure)
            self.rhs, self.density_trace, self.drift = values
            self.potential_trace = checkpoint.trace
        self.initial_residual_count = work.residual_count
        self.initial_poisson_result = poisson_result
        self.initial_recovery_metrics = work.recovery_metrics()
        self.initial_poisson_retry_results = work.poisson_results
        self.initial_poisson_retry_wall_time = sum(work.poisson_walls)
        self.accepted_poisson_tau = _poisson_tau(poisson_solver)
        residual.synchronize()
        self.setup_time = perf_counter()-start

    def _timed(self, label, verbosity, function):
        """Include completed device work in the runner's standard stage timing."""
        def synchronized():
            """Complete pending device work before returning the timed result."""
            value = function()
            self.residual.synchronize()
            return value
        return timed_call(f"[gc:{self.scheme}] {label}", verbosity, synchronized)

    def advance(self, poisson_solver, solve_transport, *, endpoint_postprocess=None):
        """Retry numerical transport failure or proven rank loss without partial commits.

        Re-evaluate the failed Poisson checkpoint first. Rebuild the start field
        and replay the entire unaccepted RK step so one tau defines every F_i.
        Traces from discarded attempts are used only as nearest-time guesses.
        """
        work = _ARKWork(self, poisson_solver, endpoint_postprocess)
        restart = (self.accepted_poisson_tau is not None
                   and self.accepted_poisson_tau != _poisson_tau(poisson_solver))
        while True:
            try:
                if restart:
                    result, checkpoint, _ = work.poisson(self.density, self.time, "restart at accepted state")
                    rhs, density_trace, drift = work.evaluate(result, checkpoint)
                else:
                    checkpoint = PoissonCheckpoint(self.density, self.potential_trace,
                                                    self.time, "accepted start")
                    rhs, density_trace, drift = self.rhs, self.density_trace, self.drift
                outcome, accepted_rhs, accepted_drift = self._advance(
                    work, solve_transport, checkpoint, rhs, density_trace, drift)
            except PoissonStageRankFailure as failure:
                work.repair(failure)
                restart = True
                continue
            self.residual.synchronize()
            self.time += self.dt
            self.density, self.drift = outcome.density, accepted_drift
            self.rhs, self.density_trace = accepted_rhs, outcome.density_trace
            self.potential_trace = outcome.potential_trace
            self.accepted_poisson_tau = _poisson_tau(poisson_solver)
            return outcome

    def _advance(self, work, solve_transport, checkpoint, start_rhs, start_trace, start_drift):
        """One atomic ARK attempt. Work accounting survives a rejected attempt."""
        s, dt, t = self.space, self.dt, self.time
        endpoint, scale = t+dt, GAMMA*dt
        frozen = work.timed("beta", "frozen scaled drift", self.detail_verbosity,
            lambda: scale * start_drift)
        full_rhs, implicit_rhs, explicit_rhs = [start_rhs], [start_rhs], [None]
        # Prefer a regenerated start trace over a previous tau at the same time.
        work.transport_candidates.append((t, start_trace))
        for i in range(1, 4):
            stage_time = endpoint if i == 3 else t+ABSCISSAE[i]*dt
            label = f"ARK stage {i+1}"
            terms = []
            for j in range(i):
                if IMPLICIT[i, j]:
                    terms.append((dt*IMPLICIT[i, j], implicit_rhs[j]))
                if j and EXPLICIT[i, j]:
                    terms.append((dt*EXPLICIT[i, j], explicit_rhs[j]))
            source = work.timed("combination", f"{label} source", self.detail_verbosity,
                lambda: sum((weight * field for weight, field in terms), start=self.density))
            source.name = "rho_ark_source_h"
            guess, guess_time = closest_trace(work.transport_candidates, stage_time)
            work.operator_reuses += int(i > 1)
            work.operator_assemblies += int(i == 1)
            work.transport_times.append(stage_time); work.transport_guesses.append(guess_time)
            work.transport_labels.append(label)
            try:
                result = work.timed("transport",
                    f"{label} transport t={stage_time:.6f} guess_t={guess_time:.6f}"
                    + (" [cached operator]" if i > 1 else " [new operator]"),
                    self.phase_verbosity,
                    lambda: solve_transport(source, frozen, guess, scale, stage_time=stage_time,
                                             stage=label, reuse_operator=i > 1))
            except Exception as error:
                work.transport_failures.append(dict(stage=label, time=stage_time,
                    wall_time=work.transport_walls[-1], error=str(error),
                    diagnostics=getattr(error, "transport_diagnostics", {})))
                raise_for_poisson_transport_failure(error, checkpoint, label + " frozen transport")
                raise
            work.transport_results.append(result)
            stage_density = solution_field(result, s).copy()
            work.check_density(stage_density, stage_time, label)
            work.transport_candidates.append((stage_time, solution_trace(result, s, reduced=True).copy()))
            stage_poisson, stage_checkpoint, _ = work.poisson(stage_density, stage_time, label)
            stage_rhs, _, _ = work.evaluate(stage_poisson, stage_checkpoint)
            implicit = (stage_density - source) / scale
            correction = stage_rhs - implicit
            full_rhs.append(stage_rhs); implicit_rhs.append(implicit); explicit_rhs.append(correction)

        density = work.timed("combination", "accepted density", self.detail_verbosity,
            lambda: sum(((dt*w) * f for w, f in zip(WEIGHTS, full_rhs)), start=self.density))
        density.name = "rho_h"
        work.check_density(density, endpoint, "accepted endpoint")
        error = sum(((dt*(w-v)) * f for w, v, f in zip(WEIGHTS, EMBEDDED_WEIGHTS, full_rhs)),
                    start=0. * self.density)
        error.name = "rho_ark_embedded_error_h"
        error_l2, density_l2 = field_l2_norm(error), field_l2_norm(density)
        if not math.isfinite(error_l2) or not math.isfinite(density_l2):
            raise FloatingPointError("IMEX-ARK3 produced a nonfinite embedded estimate or density")
        embedded_relative = error_l2/density_l2 if density_l2 else error_l2
        accepted_poisson, accepted_checkpoint, poisson_guess = work.poisson(
            density, endpoint, "accepted endpoint", final=True)
        accepted_rhs, density_trace, drift = work.evaluate(accepted_poisson, accepted_checkpoint)
        return GuidingCenterStep(
            density=density, density_trace=density_trace, poisson_result=accepted_poisson,
            potential_trace=accepted_checkpoint.trace, transport_result=result,
            transport_source=source, transport_beta=frozen, transport_initial_guess=guess,
            poisson_initial_guess=poisson_guess, transport_results=work.transport_results,
            poisson_results=work.poisson_results, transport_wall_time=sum(work.transport_walls),
            poisson_wall_time=sum(work.poisson_walls), beta_build_time=work.walls["beta"],
            metrics={
                "transport_time_order": 3, "transport_beta_scale": scale,
                "imex_ark3_embedded_order": 2, "imex_ark3_embedded_error_l2": error_l2,
                "imex_ark3_embedded_error_relative": embedded_relative,
                "imex_ark3_operator_assemblies": work.operator_assemblies,
                "imex_ark3_operator_reuses": work.operator_reuses,
                "explicit_residual_count": work.residual_count,
                "explicit_residual_time": work.walls["residual"],
                "explicit_residual_backend": self.residual.backend,
                "density_predictor_time": work.walls["combination"],
                "poisson_stage_times": work.poisson_times, "poisson_stage_labels": work.poisson_labels,
                "poisson_stage_wall_times": work.poisson_walls,
                "poisson_stage_initial_guess_times": work.poisson_guesses,
                "poisson_initial_guess_time": work.poisson_guesses[-1],
                "transport_stage_times": work.transport_times, "transport_stage_labels": work.transport_labels,
                "transport_stage_wall_times": work.transport_walls,
                "transport_stage_initial_guess_times": work.transport_guesses,
                "transport_initial_guess_time": work.transport_guesses[-1],
                "transport_initial_guess_kind": "closest-stage-trace",
                **work.recovery_metrics(),
                **work.positivity_metrics(),
            },
        ), accepted_rhs, drift


def _poisson_tau(solver):
    """Allow lightweight solver doubles while reporting the configured tau."""
    options = getattr(solver, "options", None)
    value = getattr(options, "stabilization", None)
    return None if value is None else float(value)


class _ARKWork:
    """Stage accounting and owning warm starts across bounded ARK retries."""

    def __init__(self, stepper, poisson_solver, endpoint_postprocess=None):
        """Initialize per-attempt timing, checkpoint candidates and recovery accounting."""
        self.stepper, self.solver = stepper, poisson_solver
        self.endpoint_postprocess = endpoint_postprocess or {}
        self.recovery = PoissonTauRecovery(**stepper.recovery_options)
        self.poisson_candidates = [(stepper.time, stepper.potential_trace)]
        self.transport_candidates = ([] if stepper.density_trace is None else
                                     [(stepper.time, stepper.density_trace)])
        self.poisson_results, self.transport_results = [], []
        self.transport_failures = []
        self.poisson_times, self.poisson_guesses, self.poisson_labels, self.poisson_walls = [], [], [], []
        self.transport_times, self.transport_guesses, self.transport_labels, self.transport_walls = [], [], [], []
        self.walls = dict(beta=0., residual=0., combination=0., poisson=0., transport=0., positivity=0.)
        self.density_checks = []
        self.residual_count = self.residual_failures = 0
        self.operator_assemblies = self.operator_reuses = 0
        self.recovery_wall = 0.
        self.rejected_transport_count = self.rejected_poisson_count = 0

    def timed(self, kind, label, verbosity, function):
        """Accumulate completed stage work, including failed attempts."""
        start = perf_counter()
        try:
            value, _ = self.stepper._timed(label, verbosity, function)
            return value
        finally:
            elapsed = perf_counter()-start
            self.walls[kind] += elapsed
            if kind in {"poisson", "transport"}:
                getattr(self, kind+"_walls").append(elapsed)

    def check_density(self, density, stage_time, label):
        """Record optional positivity diagnostics for this stage density."""
        if self.stepper.density_diagnostics is None:
            return
        values = self.timed("positivity", label + " positivity", self.stepper.detail_verbosity,
                            lambda: self.stepper.density_diagnostics(density))
        self.density_checks.append(dict(stage=label, time=stage_time,
            tau_attempt=len(self.recovery.events), **values))

    def positivity_metrics(self):
        """Summarize the accepted endpoint and all audited stage densities."""
        if not self.density_checks:
            return {}
        # Include discarded stages in the audit and identify their tau attempt.
        # The final entry is the accepted endpoint of the successful attempt.
        endpoint = {k: v for k, v in self.density_checks[-1].items()
                    if k not in {"stage", "time", "tau_attempt"}}
        return dict(**endpoint, positivity_stage_checks=list(self.density_checks),
                    positivity_stage_min=min(x["rho_min_checked"] for x in self.density_checks),
                    positivity_stage_negative_mass_max=max(x["rho_negative_mass_quadrature"]
                                                           for x in self.density_checks),
                    positivity_stage_time=self.walls["positivity"])

    def poisson(self, density, stage_time, label, *, final=False):
        """Solve a stage Poisson problem and retain its owning checkpoint."""
        stepper = self.stepper
        guess, guess_time = closest_trace(self.poisson_candidates, stage_time)
        def solve():
            """Solve this Poisson RHS and detach the resulting trace from solver buffers."""
            self.solver.set_source(density)
            self.solver.set_boundary_condition(stepper.potential_boundary(stage_time))
            result = self.solver.solve(initial_guess=guess,
                **(self.endpoint_postprocess if final else {}))
            return result, solution_trace(result, stepper.space, reduced=False).copy()
        result, trace = self.timed("poisson",
            f"{label} Poisson t={stage_time:.6f} guess_t={guess_time:.6f}", stepper.phase_verbosity, solve)
        self.poisson_results.append(result)
        self.poisson_candidates.append((stage_time, trace))
        self.poisson_times.append(stage_time); self.poisson_guesses.append(guess_time)
        self.poisson_labels.append(label)
        return result, PoissonCheckpoint(density, trace, stage_time, label, final), guess

    def evaluate(self, result, checkpoint):
        """Evaluate the physical residual and attach provenance on proven rank loss."""
        stepper = self.stepper
        drift = self.timed("beta", f"{checkpoint.label} drift", stepper.detail_verbosity,
            lambda: perpendicular_vector_field(result.flux, 1, stepper.space))
        try:
            rhs, trace = self.timed("residual", f"{checkpoint.label} explicit HDG residual",
                stepper.detail_verbosity,
                lambda: stepper.residual.evaluate(checkpoint.density, drift,
                                                  stepper.density_boundary(checkpoint.time)))
        except Exception as error:
            self.residual_failures += 1
            raise_for_poisson_rank_failure(error, checkpoint, checkpoint.label + " explicit HDG residual")
            raise
        self.residual_count += 1
        return rhs, trace, drift

    def repair(self, failure):
        """Increase tau and rebuild the failed Poisson checkpoint within the retry limit."""
        from hdgfem.diagnostics import solver_diagnostics_snapshot
        # Drop rejected solution/system arrays, retaining the package metric
        # interface and owning traces already stored in the candidate lists.
        for prefix in ("transport", "poisson"):
            results = getattr(self, prefix+"_results")
            for index, result in enumerate(results):
                if hasattr(result, "field") and hasattr(result, "timings"):
                    results[index] = solver_diagnostics_snapshot(result)
            setattr(self, "rejected_"+prefix+"_count", len(results))
        # Tracebacks otherwise keep the rejected RK sources and residuals alive.
        failure.error.__traceback__ = None
        failure.__traceback__ = None
        start = perf_counter()
        try:
            while True:
                self.recovery.increase(self.solver, failure, step_time=self.stepper.time)
                checkpoint = failure.checkpoint
                # Preserve the exact-time potential trace before with_options
                # destroys the solver-owned cache. Every checkpoint owns it.
                self.poisson_candidates.append((checkpoint.time, checkpoint.trace))
                result, checkpoint, _ = self.poisson(checkpoint.density, checkpoint.time,
                    checkpoint.label + " tau retry", final=checkpoint.final)
                try:
                    values = self.evaluate(result, checkpoint)
                except PoissonStageRankFailure as next_failure:
                    failure = next_failure
                else:
                    return result, checkpoint, values
        finally:
            self.recovery_wall += perf_counter()-start

    def recovery_metrics(self):
        """Return stage counts, rejected work and stabilization recovery diagnostics."""
        return {"poisson_tau": _poisson_tau(self.solver),
                "poisson_tau_retry_count": len(self.recovery.events),
                "poisson_tau_retry_events": list(self.recovery.events),
                "poisson_tau_recovery_wall_time": self.recovery_wall,
                "explicit_residual_failed_count": self.residual_failures,
                "transport_attempt_count": len(self.transport_walls),
                "transport_failed_stage_count": len(self.transport_failures),
                "transport_failed_stage_wall_time": sum(item["wall_time"] for item in self.transport_failures),
                "transport_failed_stage_diagnostics": self.transport_failures,
                "transport_rejected_stage_count": self.rejected_transport_count,
                "poisson_rejected_stage_count": self.rejected_poisson_count}
