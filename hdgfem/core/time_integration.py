"""Reusable time-discretization algebra for DG transport fields."""

from __future__ import annotations

import math
from numbers import Real

from hdgfem.core.field_ops import field_linear_combination
from hdgfem.core.space import DGField, VectorDGField


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
    if not isinstance(field, DGField):
        raise TypeError("field must be a DGField")
    if not isinstance(velocity, VectorDGField):
        raise TypeError("velocity must be a VectorDGField")
    if velocity.dim != 2:
        raise ValueError("velocity must have two components")
    if not isinstance(dt, Real) or isinstance(dt, bool):
        raise TypeError("dt must be a real number")
    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if (previous_field is None) != (previous_velocity is None):
        raise ValueError("BDF2 requires both previous field and previous velocity, or neither")
    for component in velocity.components:
        field.space.assert_same_mesh(component.space)
    if previous_field is None:
        source = field.copy(name="bdf2_source_h")
        beta = dt*velocity
        beta.name = "bdf2_beta_h"
        return source, beta, dt
    if not isinstance(previous_field, DGField):
        raise TypeError("previous_field must be a DGField")
    if not isinstance(previous_velocity, VectorDGField):
        raise TypeError("previous_velocity must be a VectorDGField")
    field.space.assert_coefficient_compatible(previous_field.space)
    if previous_velocity.dim != velocity.dim:
        raise ValueError("previous velocity must have the same dimension as velocity")
    for current, previous in zip(velocity.components, previous_velocity.components):
        current.space.assert_coefficient_compatible(previous.space)
    effective_dt = 2*dt/3
    # Form the integer-weighted extrapolations before scaling. Besides
    # matching the established scheme, this avoids extra cancellation from
    # separately rounded fractional weights in single precision.
    source = field_linear_combination(field.space, [(4, field), (-1, previous_field)])/3
    source.name = "bdf2_source_h"
    beta = VectorDGField(tuple(
        effective_dt*field_linear_combination(current.space, [(2, current), (-1, previous)])
        for current, previous in zip(velocity.components, previous_velocity.components)
    ), name="bdf2_beta_h")
    return source, beta, effective_dt


__all__ = ["bdf2_transport_data"]
