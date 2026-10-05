"""Unit tests for solver-independent source-homotopy step control."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[4] / "projects/diocotron/dolfinx"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from projects.diocotron.dolfinx.torsion.initialization.source_homotopy import (  # noqa: E402
    AdaptiveSourceHomotopy,
    SourceHomotopySchedule,
)


def test_easy_schedule_uses_half_then_one() -> None:
    controller = AdaptiveSourceHomotopy(SourceHomotopySchedule())

    first = controller.next_trial()
    assert first.lambda_old == pytest.approx(0.0)
    assert first.lambda_trial == pytest.approx(0.5)
    assert first.delta_lambda == pytest.approx(0.5)
    controller.accept(first, newton_iterations=2, backtracks=0)

    second = controller.next_trial()
    assert second.lambda_old == pytest.approx(0.5)
    assert second.lambda_trial == pytest.approx(1.0)
    controller.accept(second, newton_iterations=1, backtracks=0)

    assert controller.complete is True
    assert controller.accepted_steps == 2
    assert controller.rejected_steps == 0


def test_rejection_halves_attempted_delta_without_moving_lambda() -> None:
    controller = AdaptiveSourceHomotopy(SourceHomotopySchedule())

    failed = controller.next_trial()
    controller.reject(failed)
    assert controller.lambda_value == pytest.approx(0.0)
    assert controller.step == pytest.approx(0.25)

    quarter = controller.next_trial()
    assert quarter.lambda_trial == pytest.approx(0.25)
    controller.accept(quarter, newton_iterations=3, backtracks=0)
    assert controller.step == pytest.approx(0.25)

    half = controller.next_trial()
    assert half.lambda_old == pytest.approx(0.25)
    assert half.lambda_trial == pytest.approx(0.5)
    controller.accept(half, newton_iterations=2, backtracks=0)
    assert controller.step == pytest.approx(0.375)
    assert controller.next_trial().lambda_trial == pytest.approx(0.875)


def test_rejections_eventually_cross_minimum_step() -> None:
    schedule = SourceHomotopySchedule(min_step=0.125)
    controller = AdaptiveSourceHomotopy(schedule)

    for expected_step in (0.25, 0.125, 0.0625):
        trial = controller.next_trial()
        controller.reject(trial)
        assert controller.step == pytest.approx(expected_step)

    assert controller.below_minimum_step is True
    with pytest.raises(RuntimeError, match="below its configured minimum"):
        controller.next_trial()


def test_pending_trial_must_be_resolved_before_next_proposal() -> None:
    controller = AdaptiveSourceHomotopy(SourceHomotopySchedule())
    controller.next_trial()
    with pytest.raises(RuntimeError, match="pending homotopy trial"):
        controller.next_trial()


def test_external_update_retries_same_increment_without_moving_lambda() -> None:
    controller = AdaptiveSourceHomotopy(SourceHomotopySchedule(initial_step=0.2))

    failed = controller.next_trial()
    controller.retry_after_external_update(failed)

    assert controller.lambda_value == pytest.approx(0.0)
    assert controller.step == pytest.approx(0.2)
    assert controller.rejected_steps == 1
    retry = controller.next_trial()
    assert retry.lambda_old == pytest.approx(0.0)
    assert retry.lambda_trial == pytest.approx(0.2)


def test_configurable_strict_corrector_budget_allows_step_recovery() -> None:
    schedule = SourceHomotopySchedule(
        initial_step=0.1,
        max_step=0.4,
        step_grow=2.0,
        easy_newton_iterations=6,
    )
    controller = AdaptiveSourceHomotopy(schedule)

    first = controller.next_trial()
    controller.accept(first, newton_iterations=6, backtracks=0)
    assert controller.step == pytest.approx(0.2)

    second = controller.next_trial()
    controller.accept(second, newton_iterations=7, backtracks=0)
    assert controller.step == pytest.approx(0.2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_step": 0.0},
        {"min_step": 0.6, "max_step": 0.5},
        {"step_grow": 1.0},
        {"step_shrink": 1.0},
        {"max_attempts": 0},
        {"easy_newton_iterations": -1},
    ],
)
def test_schedule_rejects_invalid_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SourceHomotopySchedule(**kwargs)
