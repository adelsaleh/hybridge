"""Solver-independent adaptive step control for source homotopy.

The finite-element runners own their state vectors, residuals, tangent solves,
and Newton correctors.  This module owns only the continuation parameter and
its accept/reject policy, which keeps the numerical backends from duplicating
subtle rollback and step-size bookkeeping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceHomotopySchedule:
    """Configuration for adaptive continuation from zero to ``target_lambda``."""

    initial_step: float = 0.5
    min_step: float = 1.0e-3
    max_step: float = 0.5
    step_grow: float = 1.5
    step_shrink: float = 0.5
    max_attempts: int = 64
    target_lambda: float = 1.0
    easy_newton_iterations: int = 2

    def __post_init__(self) -> None:
        values = (self.initial_step, self.min_step, self.max_step)
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("homotopy step sizes must be positive and finite")
        if self.min_step > self.max_step:
            raise ValueError("homotopy min_step must not exceed max_step")
        if not math.isfinite(self.step_grow) or self.step_grow <= 1.0:
            raise ValueError("homotopy step_grow must exceed one")
        if (
            not math.isfinite(self.step_shrink)
            or not 0.0 < self.step_shrink < 1.0
        ):
            raise ValueError("homotopy step_shrink must lie strictly between zero and one")
        if self.max_attempts < 1:
            raise ValueError("homotopy max_attempts must be positive")
        if not math.isfinite(self.target_lambda) or not 0.0 < self.target_lambda <= 1.0:
            raise ValueError("homotopy target_lambda must lie in (0,1]")
        if self.easy_newton_iterations < 0:
            raise ValueError("easy_newton_iterations must be nonnegative")


@dataclass(frozen=True)
class SourceHomotopyTrial:
    """One proposed continuation step from the last accepted state."""

    attempt: int
    lambda_old: float
    lambda_trial: float
    delta_lambda: float


class AdaptiveSourceHomotopy:
    """Track accepted continuation states and adaptive trial step sizes.

    A rejected trial never changes :attr:`lambda_value`.  Its attempted
    ``delta_lambda`` is halved (or otherwise scaled by ``step_shrink``), so a
    caller can restore its state vector and retry from exactly the last
    accepted solution.
    """

    _lambda_tolerance = 1.0e-14

    def __init__(self, schedule: SourceHomotopySchedule):
        self.schedule = schedule
        self.lambda_value = 0.0
        self.step = min(
            max(float(schedule.initial_step), float(schedule.min_step)),
            float(schedule.max_step),
        )
        self.attempts = 0
        self.accepted_steps = 0
        self.rejected_steps = 0
        self._pending: SourceHomotopyTrial | None = None

    @property
    def complete(self) -> bool:
        """Return whether the configured target lambda has been accepted."""
        return self.lambda_value >= self.schedule.target_lambda - self._lambda_tolerance

    @property
    def exhausted(self) -> bool:
        """Return whether no further trial is permitted by the schedule."""
        return self.attempts >= self.schedule.max_attempts

    @property
    def below_minimum_step(self) -> bool:
        """Return whether rejection has reduced the next step below its floor."""
        return self.step < self.schedule.min_step * (1.0 - 1.0e-12)

    def next_trial(self) -> SourceHomotopyTrial:
        """Create the next trial without changing the last accepted lambda."""
        if self._pending is not None:
            raise RuntimeError("the pending homotopy trial must be accepted or rejected")
        if self.complete:
            raise RuntimeError("the homotopy target has already been reached")
        if self.exhausted:
            raise RuntimeError("the homotopy attempt budget is exhausted")
        if self.below_minimum_step:
            raise RuntimeError("the homotopy step is below its configured minimum")

        delta = min(self.step, self.schedule.target_lambda - self.lambda_value)
        trial = SourceHomotopyTrial(
            attempt=self.attempts,
            lambda_old=self.lambda_value,
            lambda_trial=self.lambda_value + delta,
            delta_lambda=delta,
        )
        self.attempts += 1
        self._pending = trial
        return trial

    def accept(
        self,
        trial: SourceHomotopyTrial,
        *,
        newton_iterations: int,
        backtracks: int,
    ) -> None:
        """Accept a fully corrected stage and optionally grow the next step."""
        self._require_pending(trial)
        self.lambda_value = min(trial.lambda_trial, self.schedule.target_lambda)
        self.accepted_steps += 1
        self._pending = None
        if (
            newton_iterations <= self.schedule.easy_newton_iterations
            and backtracks == 0
        ):
            self.step = min(
                self.schedule.max_step,
                self.step * self.schedule.step_grow,
            )

    def reject(self, trial: SourceHomotopyTrial) -> None:
        """Reject a stage and shrink the attempted increment for the retry."""
        self._require_pending(trial)
        self.rejected_steps += 1
        self.step = trial.delta_lambda * self.schedule.step_shrink
        self._pending = None

    def retry_after_external_update(self, trial: SourceHomotopyTrial) -> None:
        """Resolve a failed trial without shrinking its continuation increment.

        A coupled continuation method may change an auxiliary control while
        remaining at the last accepted ``lambda``.  Once that update has been
        strictly corrected, retrying the same source increment is preferable
        to immediately shortening it.  This method only updates bookkeeping:
        the caller remains responsible for rolling back the state and for
        verifying the external update before requesting the retry.
        """
        self._require_pending(trial)
        self.rejected_steps += 1
        self.step = trial.delta_lambda
        self._pending = None

    def _require_pending(self, trial: SourceHomotopyTrial) -> None:
        if self._pending != trial:
            raise RuntimeError("homotopy trial does not match the pending proposal")
