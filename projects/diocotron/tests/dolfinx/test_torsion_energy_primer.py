"""Tests for the exact primitive used by the energy-descent primer."""

from __future__ import annotations

import pytest

from projects.diocotron.dolfinx.torsion.initialization.energy_primer import (
    LOGISTIC_CLIP,
    classify_picard_spectrum,
    clipped_softplus,
    residual_progress_is_material,
    window_activity,
    window_primitive,
)


@pytest.mark.parametrize("value", [-80.0, -20.0, -1.0, 0.0, 2.0, 30.0, 80.0])
def test_window_primitive_derivative_matches_clipped_activity(value: float) -> None:
    c1 = -0.3
    c2 = 0.7
    epsilon = 0.4
    step = 1.0e-6
    derivative = (
        window_primitive(value + step, c1, c2, epsilon)
        - window_primitive(value - step, c1, c2, epsilon)
    ) / (2.0 * step)

    assert derivative == pytest.approx(
        window_activity(value, c1, c2, epsilon), abs=2.0e-8
    )


def test_clipped_softplus_is_continuous_at_both_clip_points() -> None:
    step = 1.0e-8
    for clip in (-LOGISTIC_CLIP, LOGISTIC_CLIP):
        center = clipped_softplus(clip)
        assert clipped_softplus(clip - step) == pytest.approx(center, abs=2.0e-8)
        assert clipped_softplus(clip + step) == pytest.approx(center, abs=2.0e-8)


def test_window_primitive_rejects_nonpositive_epsilon() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        window_primitive(0.0, 0.1, 0.2, 0.0)


def test_residual_progress_requires_material_hminus1_reduction() -> None:
    assert residual_progress_is_material(
        1.0, 0.94, minimum_relative_reduction=0.05, tolerance=1.0e-8
    )
    assert not residual_progress_is_material(
        1.0, 0.96, minimum_relative_reduction=0.05, tolerance=1.0e-8
    )
    assert residual_progress_is_material(
        1.0, 1.0e-9, minimum_relative_reduction=0.50, tolerance=1.0e-8
    )
    assert not residual_progress_is_material(
        1.0, float("nan"), minimum_relative_reduction=0.05, tolerance=1.0e-8
    )


@pytest.mark.parametrize(
    (
        "mu_min", "mu_max", "error_min", "error_max", "status",
        "energy_minimum", "contracting",
    ),
    [
        (-0.5, 0.8, 0.0, 0.0, "CONTRACTING", True, True),
        (-1.2, 0.8, 0.0, 0.0, "NONCONTRACTING", True, False),
        (-0.4, 1.1, 0.0, 0.0, "NONCONTRACTING", False, False),
        (-0.5, 0.99, 0.0, 0.02, "UNCERTAIN", None, None),
    ],
)
def test_picard_spectrum_classifies_minimum_and_contraction_separately(
    mu_min: float,
    mu_max: float,
    error_min: float,
    error_max: float,
    status: str,
    energy_minimum: bool | None,
    contracting: bool | None,
) -> None:
    measured_status, measured_minimum, measured_contraction, radius = (
        classify_picard_spectrum(
            mu_min,
            mu_max,
            error_min=error_min,
            error_max=error_max,
        )
    )

    assert measured_status == status
    assert measured_minimum is energy_minimum
    assert measured_contraction is contracting
    assert radius >= max(abs(mu_min), abs(mu_max))
