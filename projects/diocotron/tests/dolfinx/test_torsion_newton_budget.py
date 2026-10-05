"""Dependency-light tests for residual-trend Newton budget continuation."""

from __future__ import annotations

import math

import pytest

from projects.diocotron.dolfinx.torsion.equilibrium.newton_budget import (
    decide_newton_budget_extension,
    forecast_newton_stall,
    newton_hard_ceiling,
)


def decide(residuals, **overrides):
    options = {
        "initial_budget": 4,
        "current_budget": 4,
        "hard_ceiling": 12,
        "chunk": 3,
        "trend_window": 4,
        "maximum_contraction": 0.98,
    }
    options.update(overrides)
    return decide_newton_budget_extension(residuals, **options)


def test_contracting_residual_extends_by_one_bounded_chunk() -> None:
    decision = decide([1.0, 0.5, 0.25, 0.125, 0.0625])

    assert decision.extend
    assert decision.next_budget == 7
    assert decision.reason == "contracting_toward_tolerance"
    assert decision.geometric_contraction == pytest.approx(0.5)


def test_extension_is_clipped_at_finite_hard_ceiling() -> None:
    decision = decide(
        [1.0, 0.4, 0.16, 0.064, 0.0256],
        current_budget=10,
        hard_ceiling=12,
        chunk=5,
    )

    assert decision.extend
    assert decision.next_budget == 12

    stopped = decide(
        [1.0, 0.4, 0.16, 0.064, 0.0256],
        current_budget=12,
        hard_ceiling=12,
    )
    assert not stopped.extend
    assert stopped.reason == "hard_ceiling_reached"


def test_weak_or_lost_downward_trend_does_not_extend() -> None:
    weak = decide([1.0, 0.99, 0.98, 0.97, 0.96])
    assert not weak.extend
    assert weak.reason == "weak_contraction"
    assert weak.geometric_contraction > 0.98

    lost = decide([1.0, 0.6, 0.3, 0.31, 0.15])
    assert not lost.extend
    assert lost.reason == "lost_downward_trend"


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -1.0, 0.0])
def test_nonfinite_or_nonpositive_history_does_not_extend(bad_value: float) -> None:
    decision = decide([1.0, 0.5, bad_value, 0.2, 0.1])
    assert not decision.extend
    assert decision.reason == "nonfinite_or_nonpositive_residual"


def test_hard_policies_override_a_contracting_residual() -> None:
    residuals = [1.0, 0.5, 0.25, 0.125, 0.0625]

    trial = decide(
        residuals,
        hard_ceiling=4,
        hard_cap_reason="outer_trial_projection",
    )
    assert not trial.extend
    assert trial.reason == "hard_policy:outer_trial_projection"

    disabled = decide(residuals, hard_ceiling=4, soft_cap_enabled=False)
    assert not disabled.extend
    assert disabled.reason == "soft_cap_disabled"


def test_history_must_cover_the_configured_recent_window() -> None:
    decision = decide([1.0, 0.5, 0.25, 0.125])
    assert not decision.extend
    assert decision.reason == "insufficient_trend_history"


def test_hard_ceiling_rounds_up_and_validates_inputs() -> None:
    assert newton_hard_ceiling(40, 4.0) == 160
    assert newton_hard_ceiling(3, 1.1) == 4

    with pytest.raises(ValueError):
        newton_hard_ceiling(0, 4.0)
    with pytest.raises(ValueError):
        newton_hard_ceiling(4, 0.99)


def test_stall_forecast_is_scale_free_and_requires_patience() -> None:
    residuals = [1.0, 0.999, 0.998, 0.997, 0.996]
    alphas = [0.5, 0.25, 0.5, 0.125]

    first = forecast_newton_stall(
        residuals,
        alphas,
        tolerance=1.0e-12,
        completed_iterations=4,
        available_budget=100,
        window=4,
        patience=3,
    )
    scaled = forecast_newton_stall(
        [value * 1.0e7 for value in residuals],
        alphas,
        tolerance=1.0e-5,
        completed_iterations=4,
        available_budget=100,
        window=4,
        patience=3,
    )

    assert first.eligible
    assert not first.stalled
    assert first.consecutive_exceedances == 1
    assert scaled.geometric_contraction == pytest.approx(first.geometric_contraction)
    assert scaled.predicted_remaining == first.predicted_remaining

    second = forecast_newton_stall(
        residuals,
        alphas,
        tolerance=1.0e-12,
        completed_iterations=4,
        available_budget=100,
        window=4,
        patience=3,
        consecutive_exceedances=first.consecutive_exceedances,
    )
    third = forecast_newton_stall(
        residuals,
        alphas,
        tolerance=1.0e-12,
        completed_iterations=4,
        available_budget=100,
        window=4,
        patience=3,
        consecutive_exceedances=second.consecutive_exceedances,
    )
    assert not second.stalled
    assert third.stalled
    assert third.reason == "forecast_exceeds_budget"


def test_stall_forecast_full_newton_step_resets_patience() -> None:
    decision = forecast_newton_stall(
        [1.0, 0.9, 0.8, 0.7, 0.6],
        [0.5, 0.5, 1.0, 0.5],
        tolerance=1.0e-12,
        completed_iterations=4,
        available_budget=10,
        consecutive_exceedances=2,
    )

    assert not decision.eligible
    assert not decision.stalled
    assert decision.consecutive_exceedances == 0
    assert decision.reason == "window_contains_full_step"


def test_stall_forecast_respects_available_soft_cap_budget() -> None:
    residuals = [1.0, 0.8, 0.64, 0.512, 0.4096]
    alphas = [0.5, 0.5, 0.5, 0.5]
    short = forecast_newton_stall(
        residuals,
        alphas,
        tolerance=1.0e-12,
        completed_iterations=4,
        available_budget=100,
        patience=1,
    )
    extended = forecast_newton_stall(
        residuals,
        alphas,
        tolerance=1.0e-12,
        completed_iterations=4,
        available_budget=1000,
        patience=1,
    )

    assert short.stalled
    assert not extended.stalled
    assert extended.reason == "forecast_within_budget"


def test_stall_forecast_handles_extreme_finite_residual_scales() -> None:
    decision = forecast_newton_stall(
        [1.0e300, 9.0e299, 8.1e299, 7.29e299, 6.561e299],
        [0.5, 0.5, 0.5, 0.5],
        tolerance=1.0e-300,
        completed_iterations=4,
        available_budget=20_000,
        patience=1,
    )

    assert decision.eligible
    assert math.isfinite(decision.predicted_remaining)
    assert not decision.stalled


def test_logged_difficult_basin_is_rejected_after_three_forecasts() -> None:
    # Early accepted updates from outer_2_trial in the motivating high-DOF
    # run.  The old corrector continued for 651 iterations; the rolling policy
    # has enough evidence to reject this trajectory after only eight updates.
    residuals = [
        2.369982e-3,
        2.001780e-3,
        1.733741e-3,
        1.715734e-3,
        1.645327e-3,
        1.641028e-3,
        1.632239e-3,
        1.613373e-3,
        1.589971e-3,
    ]
    alphas = [0.25, 0.5, 0.03125, 0.03125, 6.104e-5, 0.001953, 0.007812, 0.01562]
    consecutive = 0
    decision = None
    for count in range(5, len(residuals) + 1):
        decision = forecast_newton_stall(
            residuals[:count],
            alphas[:count - 1],
            tolerance=1.0e-12,
            completed_iterations=count - 1,
            available_budget=1200,
            window=4,
            patience=3,
            consecutive_exceedances=consecutive,
        )
        consecutive = decision.consecutive_exceedances
        if decision.stalled:
            break

    assert decision is not None
    assert decision.stalled
    assert count - 1 == 8
    assert decision.predicted_remaining > decision.remaining_budget
