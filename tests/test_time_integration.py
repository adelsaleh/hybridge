"""Reusable BDF2 transport algebra; no PDE solves or time integration."""

import numpy as np
import pytest

from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace, VectorDGField
from hdgfem.core.time_integration import bdf2_transport_data
from hdgfem.runtime.precision import REAL_DTYPE


@pytest.fixture
def fields():
    space = DGSpace(rectangle_mesh(1, 1), 1)
    arrays = np.random.default_rng(11).normal(size=(6, *space.shape)).astype(REAL_DTYPE)
    values = [space.field(a.copy()) for a in arrays]
    return values[0], VectorDGField(values[1:3]), values[3], VectorDGField(values[4:6])


@pytest.mark.parametrize("startup", [True, False])
def test_bdf2_algebra_outputs_own_coefficients_and_leave_inputs_unchanged(fields, startup):
    field, velocity, previous, previous_velocity = fields
    inputs = [field, *velocity.components, previous, *previous_velocity.components]
    originals = [value.coeffs.copy() for value in inputs]
    history = {} if startup else dict(previous_field=previous, previous_velocity=previous_velocity)
    source, beta, effective_dt = bdf2_transport_data(field, velocity, .3, **history)
    assert effective_dt == pytest.approx(.3 if startup else .2)
    tolerance = 20*np.finfo(REAL_DTYPE).eps
    np.testing.assert_allclose(source.coeffs, originals[0] if startup else (4*originals[0]-originals[3])/3,
                               atol=tolerance, rtol=tolerance)
    for i, component in enumerate(beta.components):
        expected = .3*originals[i+1] if startup else .2*(2*originals[i+1]-originals[i+4])
        np.testing.assert_allclose(component.coeffs, expected, atol=tolerance, rtol=tolerance)
        assert component.space is velocity.components[i].space
        component.coeffs[:] = 0
    assert source.space is field.space
    source.coeffs[:] = 0
    for value, original in zip(inputs, originals):
        np.testing.assert_array_equal(value.coeffs, original)


def test_bdf2_keeps_constant_coefficients_lazy(fields):
    space = fields[0].space
    field, previous = space.constant(2), space.constant(1)
    velocity = VectorDGField((space.constant(3), space.zeros()))
    source, beta, _ = bdf2_transport_data(field, velocity, .3, previous_field=previous, previous_velocity=velocity)
    assert source.constant_value == pytest.approx(7/3)
    assert not source.coefficients_materialized
    assert beta.components[0].constant_value == pytest.approx(.6)
    assert beta.components[1].is_zero
    assert not any(component.coefficients_materialized for component in beta.components)


@pytest.mark.parametrize("dt", [0, -1, np.nan, np.inf])
def test_bdf2_rejects_invalid_step(fields, dt):
    with pytest.raises(ValueError, match="finite and positive"):
        bdf2_transport_data(*fields[:2], dt)


@pytest.mark.parametrize("dt", [True, "0.1", 1j])
def test_bdf2_rejects_nonreal_step(fields, dt):
    with pytest.raises(TypeError, match="real number"):
        bdf2_transport_data(*fields[:2], dt)


def test_bdf2_validates_types_mesh_and_history(fields):
    field, velocity, previous, previous_velocity = fields
    with pytest.raises(TypeError, match="DGField"):
        bdf2_transport_data(None, velocity, .1)
    with pytest.raises(TypeError, match="VectorDGField"):
        bdf2_transport_data(field, field, .1)
    with pytest.raises(ValueError, match="two components"):
        bdf2_transport_data(field, VectorDGField((field,)), .1)
    for history in (dict(previous_field=previous), dict(previous_velocity=previous_velocity)):
        with pytest.raises(ValueError, match="both previous"):
            bdf2_transport_data(field, velocity, .1, **history)
    other = DGSpace(rectangle_mesh(1, 1), 1)
    with pytest.raises(ValueError, match="same mesh"):
        bdf2_transport_data(field, VectorDGField((other.zeros(), other.zeros())), .1)
    with pytest.raises(ValueError, match="same mesh"):
        bdf2_transport_data(field, velocity, .1, previous_field=other.zeros(), previous_velocity=velocity)
    high = DGSpace(field.space.mesh, 2)
    with pytest.raises(ValueError, match="same polynomial order"):
        bdf2_transport_data(field, velocity, .1, previous_field=high.zeros(), previous_velocity=velocity)


@pytest.mark.parametrize("startup", [True, False])
def test_bdf2_device_parity_without_host_materialization(fields, startup):
    cp = pytest.importorskip("cupy")
    try:
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip("no CUDA device")
    except cp.cuda.runtime.CUDARuntimeError as exc:
        pytest.skip(f"CUDA unavailable: {exc}")
    from hdgfem.core.device import field_from_cupy_coefficients

    field, velocity, previous, previous_velocity = fields
    inputs = [field, *velocity.components, previous, *previous_velocity.components]
    original_device = [cp.asarray(value.coeffs) for value in inputs]
    device = [field_from_cupy_coefficients(value.space, coefficients)
              for value, coefficients in zip(inputs, original_device)]
    histories = {} if startup else dict(previous_field=previous, previous_velocity=previous_velocity)
    expected_source, expected_beta, _ = bdf2_transport_data(field, velocity, .3, **histories)
    device_histories = {} if startup else dict(previous_field=device[3], previous_velocity=VectorDGField(device[4:6]))
    source, beta, _ = bdf2_transport_data(device[0], VectorDGField(device[1:3]), .3, **device_histories)
    tolerance = 20*np.finfo(REAL_DTYPE).eps
    for result, expected in zip((source, *beta.components), (expected_source, *expected_beta.components)):
        assert not result.coefficients_materialized
        coefficients = result._device_coefficients_for(cp.cuda.Device().id)
        np.testing.assert_allclose(cp.asnumpy(coefficients), expected.coeffs, atol=tolerance, rtol=tolerance)
        assert all(coefficients.data.ptr != original.data.ptr for original in original_device)
        coefficients.fill(0)
    for result, original, expected in zip(device, original_device, inputs):
        assert not result.coefficients_materialized
        np.testing.assert_array_equal(cp.asnumpy(original), expected.coeffs)
