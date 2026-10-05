"""Pure policy helpers for bounded residual-trend Newton continuation.

This module intentionally depends only on the Python standard library.  The
optimizer can therefore unit-test the iteration-budget decision independently
of MPI, PETSc, DOLFINx, and mesh construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class NewtonBudgetDecision:
    """Decision made when a nonlinear solve exhausts its current budget."""

    extend: bool
    next_budget: int
    reason: str
    geometric_contraction: float = math.nan


@dataclass(frozen=True)
class NewtonStallForecast:
    """Rolling prediction of whether damped Newton can finish in budget.

    The forecast is deliberately dimensionless: it uses only ratios of
    accepted residuals and the corresponding line-search damping factors.
    ``consecutive_exceedances`` is returned to the caller so separate Newton
    solves do not share patience state.
    """

    stalled: bool
    eligible: bool
    reason: str
    geometric_contraction: float = math.nan
    predicted_remaining: float = math.inf
    remaining_budget: int = 0
    consecutive_exceedances: int = 0


def newton_hard_ceiling(initial_budget: int, ceiling_factor: float) -> int:
    """Return a finite hard ceiling no smaller than the initial budget."""
    if initial_budget < 1:
        raise ValueError("initial Newton budget must be positive")
    if not math.isfinite(ceiling_factor) or ceiling_factor < 1.0:
        raise ValueError("Newton soft-cap ceiling factor must be finite and >= 1")
    return max(initial_budget, int(math.ceil(initial_budget * ceiling_factor)))


def decide_newton_budget_extension(
        residuals: Sequence[float],
        *,
        initial_budget: int,
        current_budget: int,
        hard_ceiling: int,
        chunk: int,
        trend_window: int,
        maximum_contraction: float,
        soft_cap_enabled: bool = True,
        hard_cap_reason: str = "",
) -> NewtonBudgetDecision:
    """Decide whether a spent Newton budget may grow by one bounded chunk.

    ``maximum_contraction`` is the largest accepted geometric mean of the
    recent ratios ``r[k+1]/r[k]``.  It must be below one, so merely finite or
    imperceptibly decreasing residuals do not buy unbounded work.  Every
    residual in the window must also decrease strictly.
    """
    if initial_budget < 1 or current_budget < initial_budget:
        raise ValueError("invalid Newton iteration budgets")
    if hard_ceiling < current_budget:
        raise ValueError("hard ceiling is below the current Newton budget")
    if chunk < 1:
        raise ValueError("Newton soft-cap chunk must be positive")
    if trend_window < 2:
        raise ValueError("Newton trend window must contain at least two steps")
    if not math.isfinite(maximum_contraction) or not (0.0 < maximum_contraction < 1.0):
        raise ValueError("maximum Newton contraction must lie strictly in (0, 1)")

    hard_reason = str(hard_cap_reason).strip()
    if hard_reason:
        return NewtonBudgetDecision(False, current_budget, f"hard_policy:{hard_reason}")
    if not soft_cap_enabled:
        return NewtonBudgetDecision(False, current_budget, "soft_cap_disabled")
    if current_budget >= hard_ceiling:
        return NewtonBudgetDecision(False, current_budget, "hard_ceiling_reached")

    values = tuple(float(value) for value in residuals)
    required = trend_window + 1
    if len(values) < required:
        return NewtonBudgetDecision(False, current_budget, "insufficient_trend_history")
    recent = values[-required:]
    if not all(math.isfinite(value) and value > 0.0 for value in recent):
        return NewtonBudgetDecision(False, current_budget, "nonfinite_or_nonpositive_residual")

    ratios = tuple(new / old for old, new in zip(recent, recent[1:]))
    if any(ratio >= 1.0 for ratio in ratios):
        return NewtonBudgetDecision(False, current_budget, "lost_downward_trend")
    geometric = math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))
    if geometric > maximum_contraction:
        return NewtonBudgetDecision(
            False,
            current_budget,
            "weak_contraction",
            geometric,
        )

    next_budget = min(current_budget + chunk, hard_ceiling)
    return NewtonBudgetDecision(True, next_budget, "contracting_toward_tolerance", geometric)


def forecast_newton_stall(
        residuals: Sequence[float],
        damping_factors: Sequence[float],
        *,
        tolerance: float,
        completed_iterations: int,
        available_budget: int,
        window: int = 4,
        patience: int = 3,
        consecutive_exceedances: int = 0,
) -> NewtonStallForecast:
    """Forecast remaining accepted updates from a rolling contraction rate.

    A window is eligible only when every one of its accepted Newton updates
    was damped (``alpha < 1``).  This avoids terminating ordinary full-step
    Newton convergence merely because an early global phase contracted
    slowly.  For an eligible window, the geometric mean ``q`` of the recent
    residual ratios predicts

    ``ceil(log(tolerance/current_residual) / log(q))``

    remaining iterations.  Three consecutive over-budget predictions are
    required by default.  Any ineligible or affordable window resets the
    patience counter.

    ``available_budget`` is supplied by the nonlinear driver because that
    driver owns the existing soft-cap policy.  It should be the current cap
    unless the observed trend qualifies for bounded extension, in which case
    it should be the finite hard ceiling.
    """
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Newton stall tolerance must be positive and finite")
    if completed_iterations < 0:
        raise ValueError("completed Newton iterations must be nonnegative")
    if available_budget < completed_iterations:
        raise ValueError("available Newton budget is below completed iterations")
    if window < 2:
        raise ValueError("Newton stall window must contain at least two updates")
    if patience < 1:
        raise ValueError("Newton stall patience must be positive")
    if consecutive_exceedances < 0:
        raise ValueError("Newton stall patience state must be nonnegative")

    values = tuple(float(value) for value in residuals)
    alphas = tuple(float(value) for value in damping_factors)
    if len(values) != len(alphas) + 1:
        raise ValueError("residual history must contain one more entry than damping history")
    if len(alphas) < window:
        return NewtonStallForecast(
            False,
            False,
            "insufficient_history",
            remaining_budget=available_budget - completed_iterations,
        )

    recent_residuals = values[-(window + 1):]
    recent_alphas = alphas[-window:]
    remaining = available_budget - completed_iterations
    if not all(math.isfinite(value) and value > 0.0 for value in recent_residuals):
        return NewtonStallForecast(
            False,
            False,
            "nonfinite_or_nonpositive_residual",
            remaining_budget=remaining,
        )
    if not all(math.isfinite(alpha) and 0.0 < alpha < 1.0 for alpha in recent_alphas):
        return NewtonStallForecast(
            False,
            False,
            "window_contains_full_step",
            remaining_budget=remaining,
        )

    ratios = tuple(
        new / old for old, new in zip(recent_residuals, recent_residuals[1:])
    )
    geometric = math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))
    current = recent_residuals[-1]
    if current <= tolerance:
        predicted = 0.0
    elif geometric <= 0.0 or not math.isfinite(geometric):
        predicted = math.inf
    elif geometric >= 1.0:
        predicted = math.inf
    else:
        # Subtract logarithms instead of forming tolerance/current directly;
        # the quotient can underflow to zero for otherwise valid finite
        # residual scales.
        predicted = float(
            max(
                0,
                math.ceil(
                    (math.log(tolerance) - math.log(current))
                    / math.log(geometric)
                ),
            )
        )

    exceeds = not math.isfinite(predicted) or predicted > remaining
    next_count = consecutive_exceedances + 1 if exceeds else 0
    return NewtonStallForecast(
        stalled=bool(exceeds and next_count >= patience),
        eligible=True,
        reason="forecast_exceeds_budget" if exceeds else "forecast_within_budget",
        geometric_contraction=geometric,
        predicted_remaining=predicted,
        remaining_budget=remaining,
        consecutive_exceedances=next_count,
    )
