"""Bounded Poisson-stabilization recovery for numerical transport failures.

This controls existing solvers; it does not implement numerical kernels or
replace a singular transport solve with a regularized/host solve.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from hdgfem.linalg.transport_diagnostics import transport_rank_failure_details


@dataclass
class PoissonCheckpoint:
    """Owning source/trace at the Poisson evaluation which supplied a drift."""

    density: object
    trace: object
    time: float
    label: str
    final: bool = False


class PoissonStageFailure(Exception):
    """Internal backtracking signal with the owning Poisson checkpoint."""

    def __init__(self, error, checkpoint, stage):
        """Attach the failed stage and owning Poisson checkpoint."""
        self.error, self.checkpoint, self.stage = error, checkpoint, stage
        self.details = transport_rank_failure_details(error)
        self.reason = "trace-rank-loss" if self.details is not None else "transport-solve-failure"
        super().__init__(str(error))


# Compatibility for callers that specifically handle trace-rank failures.
PoissonStageRankFailure = PoissonStageFailure


def raise_for_poisson_rank_failure(error, checkpoint, stage):
    """Attach field provenance only when diagnostics prove active rank loss."""
    if transport_rank_failure_details(error) is not None:
        raise PoissonStageRankFailure(error, checkpoint, stage) from error


def is_transport_solve_failure(error):
    """Recognize numerical failures across backends; do not retry capacity/config errors."""
    from hdgfem.linalg.system import LinearSolveError, LinearSolveCapacityError
    from numpy.linalg import LinAlgError

    cause = error.__cause__
    if isinstance(error, (LinearSolveCapacityError, MemoryError)) or isinstance(
            cause, (LinearSolveCapacityError, MemoryError)):
        return False
    if any(token in str(error).lower() for token in ("out of memory", "out-of-memory", "insufficient memory")):
        return False
    if transport_rank_failure_details(error) is not None:
        return True
    if isinstance(error, (LinearSolveError, LinAlgError, FloatingPointError)):
        return True
    # SuperLU and some GPU wrappers expose numerical factorization failures as
    # RuntimeError. Restrict this to numerical messages, not arbitrary runtime bugs.
    if isinstance(error, (RuntimeError, ValueError)):
        message = str(error).lower()
        if isinstance(cause, (LinAlgError, LinearSolveError)):
            return True
        return any(token in message for token in (
            "singular", "zero pivot", "zero-pivot", "factorization failed",
            "factorisation failed", "failed to converge", "did not converge",
            "diverged", "divergence", "numerical breakdown",
            "matrix contains non-finite", "matrix data contains non-finite",
        ))
    return False


def raise_for_poisson_transport_failure(error, checkpoint, stage):
    """Backtrack a failed transport solve independently of the configured backend."""
    if is_transport_solve_failure(error):
        raise PoissonStageFailure(error, checkpoint, stage) from error


class PoissonTauRecovery:
    """Retain increased tau, invalidate operator caches via the public API.

    One instance bounds all recovery attempts in an unaccepted timestep. The
    caller repeats the checkpoint Poisson solve, then replays dependent stages.
    """

    def __init__(self, *, factor=2., max_retries=4, verbosity=0, record=None):
        """Validate retry limits and initialize stabilization recovery accounting."""
        if not math.isfinite(factor) or factor <= 1:
            raise ValueError("poisson_tau_retry_factor must be finite and greater than one")
        if isinstance(max_retries, bool) or int(max_retries) != max_retries or max_retries < 0:
            raise ValueError("poisson_tau_max_retries must be a nonnegative integer")
        self.factor, self.max_retries = float(factor), int(max_retries)
        self.verbosity, self.record = verbosity, record
        self.events = []

    def increase(self, solver, failure, *, step_time):
        """Increase stabilization through the solver API, or propagate exhaustion."""
        from hdgfem.solvers.stabilization import resolve_diffusion_stabilization

        old = float(resolve_diffusion_stabilization(
            solver.options.stabilization, solver.options.diffusion, solver.space))
        new = old*self.factor
        event = {"step_start_time": float(step_time), "stage": failure.stage,
                 "poisson_time": float(failure.checkpoint.time),
                 "poisson_stage": failure.checkpoint.label,
                 "retry": len(self.events)+1, "tau_before": old, "tau_after": new,
                 "rank_diagnostics": failure.details, "reason": failure.reason,
                 "error_type": type(failure.error).__name__, "error": str(failure.error)}
        exhausted = len(self.events) >= self.max_retries
        invalid = not math.isfinite(old) or old <= 0 or not math.isfinite(new)
        if exhausted or invalid:
            event.update(status="exhausted" if exhausted else "invalid_tau", tau_after=old)
            if self.record is not None:
                self.record(event)
            reason = (f"Poisson tau recovery exhausted after {len(self.events)} retries"
                      if exhausted else "Poisson tau recovery requires finite positive scalar tau")
            failure.error.add_note(reason + "; accepted state was not advanced.")
            if self.verbosity:
                print(f"\n[gc:poisson-retry] {reason}; tau={old:g}", flush=True)
            raise failure.error from failure
        # This clears numerical factors, matrix and native/AMGX hierarchies;
        # Compatible tau-independent raw flux recovery data also remain cached.
        solver.with_options(stabilization=new)
        event["status"] = "retry"
        self.events.append(event)
        if self.record is not None:
            self.record(event)
        if self.verbosity:
            print(f"\n[gc:poisson-retry] {failure.stage}: edges={(failure.details or {}).get('edges', [])} "
                  f"tau={old:g} -> {new:g}; repeat {failure.checkpoint.label} "
                  f"Poisson t={failure.checkpoint.time:.6f}; replay unaccepted timestep",
                  flush=True)
        return new
