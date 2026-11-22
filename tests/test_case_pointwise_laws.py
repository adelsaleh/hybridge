"""Compiled case coefficients (hybridge PointwiseLaw) agree with the catalogue's NumPy closures."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

from hybridge.cases.closed_loop_pointwise import closed_loop_pointwise_case
from scripts.advection_diffusion_reaction.cases import catalogue, closed_loop_stress_cases as stress
from scripts.advection_diffusion_reaction.cases import oscillatory_cases as oscillatory


def _points(shape, annulus=False, seed=7):
    rng = np.random.default_rng(seed)
    if not annulus:
        return rng.uniform(-1, 1, shape), rng.uniform(-1, 1, shape)
    radius, angle = rng.uniform(0.7, 1.0, shape), rng.uniform(0, 2*np.pi, shape)
    return radius*np.cos(angle), radius*np.sin(angle)


def _assert_close(compiled, reference, x, y, tolerance=1e-13):
    if not callable(reference):
        assert compiled == reference
        return
    expected = np.asarray(reference(x, y), dtype=float)
    scale = max(1.0, float(np.max(np.abs(expected))))
    np.testing.assert_allclose(compiled(x, y), expected, rtol=0, atol=tolerance*scale)


@pytest.mark.parametrize("key", [*catalogue.SINE_CASES, *oscillatory.VARIANTS, "variable_velocity"])
def test_compiled_case_matches_numpy(key):
    reference, compiled = catalogue.CASE_DEFINITIONS[key].build(), catalogue.compiled_case(key)
    x, y = _points((37, 5))  # 2-D input: laws flatten and pad any shape
    for name in ("source", "boundary_condition", "exact"):
        _assert_close(getattr(compiled, name), getattr(reference, name), x, y)
    for component, expected in zip(compiled.beta, reference.beta):
        _assert_close(component, expected, x, y)
    assert compiled.reaction == reference.reaction
    assert compiled.diffusion is reference.diffusion or np.array_equal(compiled.diffusion, reference.diffusion)
    assert compiled.exact_flux is not None


def test_compiled_case_none_without_laws():
    assert catalogue.compiled_case("disk") is None


@pytest.mark.parametrize("variant", ["trap", "cross", "orthogonal"])
def test_closed_loop_laws_match_numpy(variant):
    parameters = stress.StressParameters(geometry="annulus", variant=variant, reaction=1e-3,
                                         **stress.LEVELS["main"])
    normalization = 1.0 if variant == "orthogonal" else 3.7
    reference, _ = stress.make_case(parameters, normalization)
    fields = {k: getattr(parameters, k) for k in ("variant", "epsilon", "speed", "reaction", "neck_width")}
    compiled, _ = closed_loop_pointwise_case(**fields, normalization=normalization)
    x, y = _points(301, annulus=True)
    # The corrugated-coordinate derivatives cancel heavily (epsilon = 1e-6), so compiled and NumPy
    # evaluation orders and libm differ by ~5e-13 of the largest sample (~5e-11 pointwise).
    close = lambda compiled, expected: _assert_close(compiled, expected, x, y, tolerance=1e-10)
    close(compiled["source"], reference["source"])
    close(compiled["boundary_condition"], reference["boundary_condition"])
    for component, expected in zip(compiled["beta"], reference["beta"]):
        close(component, expected)
    for row, expected_row in zip(compiled["diffusion"], reference["diffusion"]):
        for component, expected in zip(row, expected_row):
            close(component, expected)
