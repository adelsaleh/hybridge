"""Reusable time-discretization algebra for DG transport fields.

Velocities are DG vector fields (:class:`~hybridge.core.space.VectorDGField`) or advection fields given
at the quadrature points (:class:`~hybridge.core.quadrature_coefficients.QuadratureVectorField`);
the history levels of one call must all have the same kind and layout.
"""

from __future__ import annotations

import math
from numbers import Real

from hybridge.core.field_ops import field_linear_combination
from hybridge.core.quadrature_coefficients import QuadratureVectorField
from hybridge.core.space import DGField, VectorDGField


def _checked_current(field, velocity, dt) -> float:
    """Validate the current field, its velocity and the step; return float dt."""
    if not isinstance(field, DGField):
        raise TypeError("field must be a DGField")
    if not isinstance(velocity, (VectorDGField, QuadratureVectorField)):
        raise TypeError("velocity must be a VectorDGField or a QuadratureVectorField")
    if velocity.dim != 2:
        raise ValueError("velocity must have two components")
    if not isinstance(dt, Real) or isinstance(dt, bool):
        raise TypeError("dt must be a real number")
    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if isinstance(velocity, QuadratureVectorField):
        field.space.assert_same_mesh(velocity.space)
        return dt
    for component in velocity.components:
        field.space.assert_same_mesh(component.space)
    return dt


def _check_history(field, velocity, history_field, history_velocity, label) -> None:
    """Require one history level to match the current field and velocity bases."""
    if not isinstance(history_field, DGField):
        raise TypeError(f"{label}_field must be a DGField")
    if isinstance(velocity, QuadratureVectorField):
        if not isinstance(history_velocity, QuadratureVectorField):
            raise TypeError(f"{label}_velocity must be a QuadratureVectorField like velocity")
        field.space.assert_coefficient_compatible(history_field.space)
        history_velocity.check(velocity.space, velocity.trace_space)
        return
    if not isinstance(history_velocity, VectorDGField):
        raise TypeError(f"{label}_velocity must be a VectorDGField")
    field.space.assert_coefficient_compatible(history_field.space)
    if history_velocity.dim != velocity.dim:
        raise ValueError(f"{label} velocity must have the same dimension as velocity")
    for current, history in zip(velocity.components, history_velocity.components):
        current.space.assert_coefficient_compatible(history.space)


def _scaled_velocity_combination(effective_dt, weights, velocities, name):
    """Return ``effective_dt * sum_i weights[i] * velocities[i]``, integer weights combined first.

    DG velocities combine componentwise with :func:`field_linear_combination`; quadrature
    advection fields with :meth:`QuadratureVectorField.linear_combination`, which keeps a single-valued
    normal flux bitwise antisymmetric.
    """
    if isinstance(velocities[0], QuadratureVectorField):
        combined = effective_dt * QuadratureVectorField.linear_combination(list(zip(weights, velocities)), name=name)
        combined.name = name
        return combined
    return VectorDGField(tuple(
        effective_dt*field_linear_combination(components[0].space, list(zip(weights, components)))
        for components in zip(*(velocity.components for velocity in velocities))
    ), name=name)


def bdf2_transport_data(
    field: DGField,
    velocity: VectorDGField,
    dt: float,
    *,
    previous_field: DGField | None = None,
    previous_velocity: VectorDGField | None = None,
) -> tuple[DGField, VectorDGField, float]:
    """Return history source, scaled velocity, and the effective implicit step.

    With no history, use Euler: ``source = field`` and ``beta = dt*velocity``.
    Otherwise constant-step BDF2 uses ``source = (4*field-previous_field)/3``
    and ``beta = (2*dt/3)*(2*velocity-previous_velocity)``. Pass both histories
    from the same accepted endpoint, at the same time step. The caller owns
    history updates and commits them only after its coupled solves succeed.

    All fields must share a mesh; each history must match its current field's
    basis and degree. Outputs own their coefficients and preserve host/device
    residency through the shared field algebra. Inputs are never modified.
    """
    dt = _checked_current(field, velocity, dt)
    if (previous_field is None) != (previous_velocity is None):
        raise ValueError("BDF2 requires both previous field and previous velocity, or neither")
    if previous_field is None:
        source = field.copy(name="bdf2_source_h")
        beta = dt*velocity
        beta.name = "bdf2_beta_h"
        return source, beta, dt
    _check_history(field, velocity, previous_field, previous_velocity, "previous")
    effective_dt = 2*dt/3
    # Form the integer-weighted extrapolations before scaling. Besides
    # matching the established scheme, this avoids extra cancellation from
    # separately rounded fractional weights in single precision.
    source = field_linear_combination(field.space, [(4, field), (-1, previous_field)])/3
    source.name = "bdf2_source_h"
    beta = _scaled_velocity_combination(effective_dt, (2, -1), (velocity, previous_velocity), "bdf2_beta_h")
    return source, beta, effective_dt


def bdf3_transport_data(
    field: DGField,
    velocity: VectorDGField,
    dt: float,
    *,
    previous_field: DGField | None = None,
    previous_velocity: VectorDGField | None = None,
    older_field: DGField | None = None,
    older_velocity: VectorDGField | None = None,
) -> tuple[DGField, VectorDGField, float]:
    """Return history source, scaled velocity, and the effective implicit step.

    Constant-step BDF3 uses ``source = (18*field-9*previous_field+2*older_field)/11``
    and ``beta = (6*dt/11)*(3*velocity-3*previous_velocity+older_velocity)``.
    Without the older level this is :func:`bdf2_transport_data` (BDF2, or
    Euler without any history), so callers can ramp the order during startup.
    Each level needs both its field and velocity, from one accepted endpoint;
    an older level requires the previous one. The caller owns history updates
    and commits them only after its coupled solves succeed.

    Mesh, basis and residency rules match :func:`bdf2_transport_data`.
    Outputs own their coefficients and inputs are never modified.
    """
    if (older_field is None) != (older_velocity is None):
        raise ValueError("BDF3 requires both older field and older velocity, or neither")
    if older_field is None:
        return bdf2_transport_data(field, velocity, dt, previous_field=previous_field,
                                   previous_velocity=previous_velocity)
    if previous_field is None or previous_velocity is None:
        raise ValueError("BDF3 requires both previous field and previous velocity with an older level")
    dt = _checked_current(field, velocity, dt)
    _check_history(field, velocity, previous_field, previous_velocity, "previous")
    _check_history(field, velocity, older_field, older_velocity, "older")
    effective_dt = 6*dt/11
    # As for BDF2, combine integer weights before the single fractional scale.
    source = field_linear_combination(
        field.space, [(18, field), (-9, previous_field), (2, older_field)])/11
    source.name = "bdf3_source_h"
    beta = _scaled_velocity_combination(effective_dt, (3, -3, 1), (velocity, previous_velocity, older_velocity),
                                        "bdf3_beta_h")
    return source, beta, effective_dt


__all__ = ["bdf2_transport_data", "bdf3_transport_data"]
