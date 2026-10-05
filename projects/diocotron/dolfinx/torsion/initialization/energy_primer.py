"""Dependency-light scalar helpers for the equilibrium energy primer.

The DOLFINx runner builds the equivalent UFL expressions.  Keeping the scalar
definition here makes the clipped primitive and its derivative independently
testable without importing MPI, PETSc, or DOLFINx.
"""

from __future__ import annotations

import math


LOGISTIC_CLIP = 50.0
_SOFTPLUS_AT_NEGATIVE_CLIP = math.log1p(math.exp(-LOGISTIC_CLIP))
_SOFTPLUS_AT_POSITIVE_CLIP = math.log1p(math.exp(LOGISTIC_CLIP))


def clipped_logistic(value: float) -> float:
    """Return the logistic with the same exact tails used by the UFL source."""
    z = float(value)
    if z < -LOGISTIC_CLIP:
        return 0.0
    if z > LOGISTIC_CLIP:
        return 1.0
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    exponential = math.exp(z)
    return exponential / (1.0 + exponential)


def clipped_softplus(value: float) -> float:
    """Primitive whose derivative is :func:`clipped_logistic` everywhere.

    Outside ``[-50, 50]`` the primitive is continued with the constant slope
    of the clipped logistic: zero on the left and one on the right.  The
    endpoint values make the continuation continuous.
    """
    z = float(value)
    if z < -LOGISTIC_CLIP:
        return _SOFTPLUS_AT_NEGATIVE_CLIP
    if z > LOGISTIC_CLIP:
        return _SOFTPLUS_AT_POSITIVE_CLIP + (z - LOGISTIC_CLIP)
    if z >= 0.0:
        return z + math.log1p(math.exp(-z))
    return math.log1p(math.exp(z))


def window_primitive(value: float, c1: float, c2: float, epsilon: float) -> float:
    """Return a primitive of the clipped two-threshold activity window."""
    eps = float(epsilon)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("window primitive requires positive finite epsilon")
    s = float(value)
    return eps * (
        clipped_softplus((s - float(c1)) / eps)
        - clipped_softplus((s - float(c2)) / eps)
    )


def window_activity(value: float, c1: float, c2: float, epsilon: float) -> float:
    """Return the clipped activity differentiated from ``window_primitive``."""
    eps = float(epsilon)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("window activity requires positive finite epsilon")
    s = float(value)
    return (
        clipped_logistic((s - float(c1)) / eps)
        - clipped_logistic((s - float(c2)) / eps)
    )


def residual_progress_is_material(
    residual_before: float,
    residual_after: float,
    *,
    minimum_relative_reduction: float,
    tolerance: float,
) -> bool:
    """Return whether a primer step improves the dual residual sufficiently.

    Reaching the requested tolerance is always sufficient. Otherwise the new
    residual must be smaller by minimum_relative_reduction.
    """
    before = float(residual_before)
    after = float(residual_after)
    reduction = float(minimum_relative_reduction)
    tol = float(tolerance)
    if not all(math.isfinite(value) for value in (before, after, reduction, tol)):
        return False
    if before < 0.0 or after < 0.0 or tol <= 0.0 or not 0.0 <= reduction < 1.0:
        return False
    if after <= tol:
        return True
    return after <= (1.0 - reduction) * before


def classify_picard_spectrum(
    mu_min: float,
    mu_max: float,
    *,
    error_min: float = 0.0,
    error_max: float = 0.0,
) -> tuple[str, bool | None, bool | None, float]:
    """Classify local energy minimality and undamped-Picard contraction.

    A strict local energy minimum needs only mu_max < 1; undamped Picard
    additionally needs mu_min > -1. Eigenvalue error estimates are used
    conservatively, producing UNCERTAIN when their intervals touch either
    boundary.
    """
    values = tuple(float(value) for value in (mu_min, mu_max, error_min, error_max))
    if not all(math.isfinite(value) for value in values):
        return "EIGEN_NOT_CONVERGED", None, None, math.nan
    lower, upper, err_lower, err_upper = values
    if lower > upper or err_lower < 0.0 or err_upper < 0.0:
        return "INVALID_SPECTRUM", None, None, math.nan

    lower_interval = (lower - err_lower, lower + err_lower)
    upper_interval = (upper - err_upper, upper + err_upper)
    radius_bound = max(
        abs(lower_interval[0]),
        abs(lower_interval[1]),
        abs(upper_interval[0]),
        abs(upper_interval[1]),
    )

    if upper_interval[1] < 1.0:
        energy_minimum: bool | None = True
    elif upper_interval[0] >= 1.0:
        energy_minimum = False
    else:
        energy_minimum = None

    if lower_interval[0] > -1.0 and upper_interval[1] < 1.0:
        contracting: bool | None = True
    elif lower_interval[1] <= -1.0 or upper_interval[0] >= 1.0:
        contracting = False
    else:
        contracting = None

    if contracting is True:
        status = "CONTRACTING"
    elif contracting is False:
        status = "NONCONTRACTING"
    else:
        status = "UNCERTAIN"
    return status, energy_minimum, contracting, radius_bound
