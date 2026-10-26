"""Shared stage result, transport drift and nearest-time warm-start selection."""
from dataclasses import dataclass
import math
from typing import Any
from hybridge.core.field_ops import perpendicular_vector_field, trace_linear_combination
from hybridge.core.quadrature_coefficients import QuadratureVectorField


def drift_flux(result):
    """Return what supplies the transport drift of a Poisson ``result``: its flux, or an exact drift.

    A pointwise exact-drift run (``runtime/exact_drift.py``) attaches ``result.exact_drift``, the exact
    drift at the transport's quadrature points (:class:`~hybridge.core.quadrature_coefficients.QuadratureVectorField`);
    every other run uses ``result.flux``. Either goes to :func:`transport_drift`.
    """
    exact = getattr(result, "exact_drift", None)
    return result.flux if exact is None else exact


def transport_drift(flux, scale, space, *, name="beta_h"):
    """Return the transport drift ``scale * flux^perp`` of a Poisson flux.

    A flux of degree at most that of the density ``space`` is represented in
    ``space`` (exact embedding: Poisson offsets 0 and -1). A raised-degree flux
    (Poisson degree q > p) keeps its own space; the transport assembly and
    residual contract it with cross-space drift tables at the density's
    quadrature points, without projection (``hybridge.transport.advection_field``).
    A :class:`~hybridge.core.quadrature_coefficients.QuadratureVectorField` (see
    :func:`drift_flux`) already is the drift ``(-q_y, q_x)`` and is only scaled.
    """
    if isinstance(flux, QuadratureVectorField):
        drift = flux * float(scale)
        drift.name = name
        return drift
    flux_order = flux.components[0].space.order
    return perpendicular_vector_field(flux, scale, space if flux_order <= space.order else None, name=name)


def closest_trace(candidates, target_time):
    """Choose the closest time; break ties in favor of the newest evaluation."""
    distances = [abs(time-target_time) for time, _ in candidates]
    nearest = min(distances)
    rounding = 4*max(math.ulp(target_time), *(math.ulp(time) for time, _ in candidates))
    # Stage midpoints may round toward either endpoint. Treat only distances
    # indistinguishable at floating-point resolution as ties.
    index = next(i for i in range(len(candidates)-1, -1, -1)
                 if distances[i] <= nearest+rounding)
    guess_time, trace = candidates[index]
    return trace, guess_time


@dataclass
class GuidingCenterStep:
    """Accepted density/field state with all stage solves and diagnostic metadata."""

    density: Any
    density_trace: Any
    poisson_result: Any
    potential_trace: Any
    transport_result: Any
    transport_source: Any
    transport_beta: Any
    transport_initial_guess: Any
    poisson_initial_guess: Any
    transport_results: list
    poisson_results: list
    transport_wall_time: float
    poisson_wall_time: float
    beta_build_time: float
    metrics: dict
    transport_boundary: Any = None
    potential_trace_time: float = 0.0
    post_poisson_start: float | None = None



def _fixed_operator_trace_predictor(current, previous=None, older=None):
    """Predict the next trace from up to three accepted fixed-operator solves."""
    if older is not None:
        return (
            trace_linear_combination(
                [(3.0, current), (-3.0, previous), (1.0, older)]
            ),
            2,
        )
    if previous is not None:
        return (
            trace_linear_combination([(2.0, current), (-1.0, previous)]),
            1,
        )
    return current, 0

