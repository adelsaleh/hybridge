"""Raised-degree Poisson in the guiding-center runtime: drift space and configuration rule.

A Poisson flux of degree q > p must reach the DG(p) transport in its own space (cross-space
drift tables, no projection); fluxes of degree <= p keep the density space, so offsets 0 and
-1 are unchanged. Host-only checks.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hybridge import DGSpace, rectangle_mesh
from hybridge.core.field_ops import perpendicular_vector_field
from hybridge.core.space import VectorDGField
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime.configuration import _validate_config
from scripts.guiding_center.time_schemes.stage_support import transport_drift

FLUX = (lambda x, y: np.sin(x + 2.0 * y), lambda x, y: x * y - y)


@pytest.mark.parametrize("flux_order", [1, 2, 4])
def test_transport_advection_field_space(flux_order):
    mesh = rectangle_mesh(3, 2)
    space = DGSpace(mesh, 2)
    flux_space = space if flux_order == 2 else DGSpace(mesh, flux_order)
    flux = VectorDGField(FLUX, flux_space)
    drift = transport_drift(flux, 0.5, space)
    target = space if flux_order <= space.order else flux_space
    assert all(component.space is target for component in drift.components)
    expected = perpendicular_vector_field(flux, 0.5, target)
    for actual, reference in zip(drift.components, expected.components):
        np.testing.assert_array_equal(actual.coeffs, reference.coeffs)


@pytest.mark.parametrize("scheme", ["si-euler", "si-bdf2", "si-bdf3", "imex-ark3"])
@pytest.mark.parametrize("offset", [1, 2])
def test_raised_poisson_degree_is_accepted(scheme, offset):
    base = preset_by_key("rho_helm_wave_raw_cuda_amgx_accuracy")
    _validate_config(replace(base, time_scheme=scheme, poisson_order_offset=offset))


@pytest.mark.parametrize("changes, message", [
    (dict(poisson_order_offset=-2), "must be -1, 0, or a raised degree"),
    (dict(poisson_order_offset=1, time_scheme="predictor-corrector"), "wired for"),
    (dict(poisson_order_offset=1, time_scheme="imex-ark3", transport_electric_field="compatible"), "raw LDG-H flux"),
    (dict(poisson_order_offset=-1, time_scheme="imex-ark3"), "require si-bdf2 or si-bdf3"),
])
def test_poisson_degree_rejections(changes, message):
    base = preset_by_key("rho_helm_wave_raw_cuda_amgx_accuracy")
    with pytest.raises(ValueError, match=message):
        _validate_config(replace(base, **changes))
