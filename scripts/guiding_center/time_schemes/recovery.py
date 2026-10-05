"""Shared transactional replay and Poisson checkpoints for temporal steppers."""
from functools import wraps
from time import perf_counter

from hybridge.core.field_ops import solution_trace
from hybridge.core.transfer import project_same_mesh_field
from hybridge.core.space import DGField
from hybridge.runtime.logging import timed_call
from scripts.guiding_center.poisson.poisson_recovery import (
    PoissonCheckpoint, PoissonStageFailure, PoissonTauRecovery,
    raise_for_poisson_rank_failure, raise_for_poisson_transport_failure,
)


def poisson_tau(solver):
    value = getattr(getattr(solver, "options", None), "stabilization", None)
    return None if value is None else float(value)


def recovery_options(factor=2., max_retries=4, verbosity=0, record=None):
    options = dict(factor=factor, max_retries=max_retries, verbosity=verbosity, record=record)
    PoissonTauRecovery(**options)
    return options


class StepRecoveryWork:
    """Keep exact Poisson provenance and failed-work accounting outside accepted state."""

    def __init__(self, stepper, solver, endpoint_postprocess=None):
        self.stepper, self.solver = stepper, solver
        self.endpoint_postprocess = endpoint_postprocess or {}
        self.recovery = PoissonTauRecovery(**stepper.recovery_options)
        self.checkpoint = self.start_checkpoint = None
        self.poisson_results, self.transport_results = [], []
        self.poisson_times, self.poisson_labels, self.poisson_walls = [], [], []
        self.poisson_guess_times = []
        self.residual_wall = 0.
        self.poisson_wall = self.transport_wall = self.recovery_wall = 0.
        self.transport_attempts = self.residual_count = self.residual_failures = 0
        self.transport_failures = []
        self.rejected_transport_count = self.rejected_poisson_count = 0

    def accepted_start(self, density, trace):
        self.start_checkpoint = self.checkpoint = PoissonCheckpoint(
            density, trace, self.stepper.time, "accepted start")

    def synchronize(self):
        workspace = getattr(self.stepper, "residual", None)
        if workspace is not None:
            workspace.synchronize()

    def poisson(self, density, stage_time, label, *, guess, final=False, guess_time=None):
        """Rebuild through the public solver API and retain owning source and trace."""
        start = perf_counter()
        poisson_space = getattr(self.solver, "space", self.stepper.space)
        source = project_same_mesh_field(density, poisson_space) if isinstance(density, DGField) else density
        self.solver.set_source(source)
        self.solver.set_boundary_condition(self.stepper.potential_boundary(stage_time))
        result = self.solver.solve(initial_guess=guess,
            **(self.endpoint_postprocess if final else {}))
        self.last_post_poisson_start = perf_counter()
        trace, self.last_trace_wall = timed_call(
            "[gc] updating accepted potential trace",
            self.stepper.detail_verbosity if final else 0,
            lambda: solution_trace(result, poisson_space, reduced=False).copy())
        source = density.copy(name="rho_checkpoint_h")
        self.synchronize()
        elapsed = perf_counter()-start
        self.poisson_wall += elapsed
        self.poisson_results.append(result)
        self.poisson_times.append(stage_time)
        self.poisson_labels.append(label)
        self.poisson_guess_times.append(guess_time)
        self.poisson_walls.append(elapsed)
        self.checkpoint = PoissonCheckpoint(source, trace, stage_time, label, final)
        return result, trace

    def transport(self, solve, *args, **kwargs):
        """Wrap every transport callback, including startup and corrector solves."""
        stage = kwargs.get("stage") or self.stepper.scheme
        start = perf_counter()
        self.transport_attempts += 1
        try:
            result = solve(*args, **kwargs)
            self.synchronize()
        except Exception as error:
            self.transport_failures.append(dict(stage=stage, error=str(error),
                wall_time=perf_counter()-start,
                diagnostics=getattr(error, "transport_diagnostics", {})))
            raise_for_poisson_transport_failure(error, self.checkpoint, stage)
            raise
        finally:
            self.transport_wall += perf_counter()-start
        self.transport_results.append(result)
        return result

    def evaluate(self, density, drift, stage_time):
        """Attach the preceding Poisson solve to explicit trace-rank failures."""
        start = perf_counter()
        try:
            result = self.stepper.residual.evaluate(
                density, drift, self.stepper.density_boundary(stage_time))
            self.synchronize()
        except Exception as error:
            self.residual_failures += 1
            raise_for_poisson_rank_failure(error, self.checkpoint, "explicit HDG residual")
            raise
        finally:
            self.residual_wall += perf_counter()-start
        self.residual_count += 1
        return result

    def repair(self, failure):
        """Repeat the failed checkpoint first; the caller then rebuilds histories."""
        from hybridge.diagnostics.solver import solver_diagnostics_snapshot
        for kind in ("poisson", "transport"):
            results = getattr(self, kind+"_results")
            for index, result in enumerate(results):
                if hasattr(result, "field") and hasattr(result, "timings"):
                    results[index] = solver_diagnostics_snapshot(result)
            setattr(self, "rejected_"+kind+"_count", len(results))
        failure.error.__traceback__ = None
        failure.__traceback__ = None
        start = perf_counter()
        try:
            self.recovery.increase(self.solver, failure, step_time=self.stepper.time)
            checkpoint = failure.checkpoint
            return self.poisson(checkpoint.density, checkpoint.time,
                checkpoint.label + " tau retry", guess=checkpoint.trace, final=checkpoint.final,
                guess_time=checkpoint.time)
        finally:
            self.recovery_wall += perf_counter()-start

    def metrics(self):
        return dict(poisson_tau=poisson_tau(self.solver),
            poisson_tau_retry_count=len(self.recovery.events),
            poisson_tau_retry_events=list(self.recovery.events),
            poisson_tau_recovery_wall_time=self.recovery_wall,
            transport_attempt_count=self.transport_attempts,
            transport_failed_stage_count=len(self.transport_failures),
            transport_failed_stage_wall_time=sum(x['wall_time'] for x in self.transport_failures),
            transport_failed_stage_diagnostics=self.transport_failures,
            transport_rejected_stage_count=self.rejected_transport_count,
            poisson_rejected_stage_count=self.rejected_poisson_count,
            explicit_residual_failed_count=self.residual_failures)


def with_poisson_recovery(advance):
    """Replay one complete unaccepted step; restore accepted state on any terminal error."""
    @wraps(advance)
    def wrapped(self, poisson_solver, solve_transport, *, endpoint_postprocess=None):
        if self.accepted_poisson_tau is None:
            self.accepted_poisson_tau = poisson_tau(poisson_solver)
        saved = self.__dict__.copy()
        work = StepRecoveryWork(self, poisson_solver, endpoint_postprocess)
        restart = self.accepted_poisson_tau != poisson_tau(poisson_solver)
        try:
            while True:
                self.__dict__.update(saved)
                self._recovery_work = work
                try:
                    if restart:
                        start = perf_counter()
                        try:
                            self._rebuild_poisson_history(work)
                        finally:
                            work.recovery_wall += perf_counter()-start
                    density = self.densities[0] if hasattr(self, "densities") else self.density
                    work.accepted_start(density, self.potential_trace)
                    outcome = advance(self, poisson_solver,
                        lambda *a, **kw: work.transport(solve_transport, *a, **kw),
                        endpoint_postprocess=endpoint_postprocess)
                except PoissonStageFailure as failure:
                    work.repair(failure)
                    restart = True
                    continue
                outcome.metrics.update(work.metrics())
                outcome.poisson_results = work.poisson_results
                outcome.transport_results = work.transport_results
                outcome.poisson_wall_time = work.poisson_wall
                outcome.transport_wall_time = max(outcome.transport_wall_time, work.transport_wall)
                if hasattr(self, "residual"):
                    outcome.metrics.update(explicit_residual_count=work.residual_count,
                                           explicit_residual_time=work.residual_wall)
                if work.recovery.events:
                    outcome.metrics.update(
                        poisson_stage_initial_guess_times=work.poisson_guess_times,
                        poisson_stage_times=work.poisson_times,
                        poisson_stage_labels=work.poisson_labels,
                        poisson_stage_wall_times=work.poisson_walls)
                self.accepted_poisson_tau = poisson_tau(poisson_solver)
                return outcome
        except BaseException:
            self.__dict__.clear()
            self.__dict__.update(saved)
            raise
        finally:
            self.__dict__.pop("_recovery_work", None)
    return wrapped
