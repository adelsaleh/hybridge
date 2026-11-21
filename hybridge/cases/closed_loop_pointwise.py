"""Closed-loop stress coefficients as compiled, parallel :class:`~hybridge.core.pointwise.PointwiseLaw` objects.

The formulas are :mod:`hybridge.cases.closed_loop_coefficients`; this module only binds them to
the coefficient-callable contract of the stationary ADR solvers, so host sampling runs in
hybridge's parallel pointwise kernels instead of NumPy on one core. Each law evaluates one
component (no bundled multi-component input exists yet; see TODO.md, Coefficient Inputs).
"""
from __future__ import annotations

import math

from hybridge.cases.closed_loop_coefficients import closed_loop_exact, closed_loop_velocity, closed_loop_volume
from hybridge.core.pointwise import pointwise_law

VARIANT_CODES = {"trap": 0.0, "cross": 1.0, "orthogonal": 2.0}


# v = (hole_radius, epsilon, speed/normalization, reaction, variant), the closed_loop_* parameters.
def _kxx(x, y, t, v):
    """Evaluate the xx diffusion component for the closed-loop pointwise law."""
    return closed_loop_volume(x, y, (v[0], v[1], v[2], v[3], v[4]))[0]


def _kxy(x, y, t, v):
    """Evaluate the xy diffusion component for the closed-loop pointwise law."""
    return closed_loop_volume(x, y, (v[0], v[1], v[2], v[3], v[4]))[1]


def _kyy(x, y, t, v):
    """Evaluate the yy diffusion component for the closed-loop pointwise law."""
    return closed_loop_volume(x, y, (v[0], v[1], v[2], v[3], v[4]))[2]


def _source(x, y, t, v):
    """Evaluate the manufactured closed-loop source."""
    return closed_loop_volume(x, y, (v[0], v[1], v[2], v[3], v[4]))[5]


def _beta_x(x, y, t, v):
    """Evaluate the x component of closed-loop advection."""
    return closed_loop_velocity(x, y, (v[0], v[1], v[2], v[3], v[4]))[0]


def _beta_y(x, y, t, v):
    """Evaluate the y component of closed-loop advection."""
    return closed_loop_velocity(x, y, (v[0], v[1], v[2], v[3], v[4]))[1]


def _exact(x, y, t, v):
    """Evaluate the exact closed-loop solution."""
    return closed_loop_exact(x, y, (v[0], v[1], v[2], v[3], v[4]))


def closed_loop_pointwise_case(*, variant, epsilon, speed, reaction, neck_width=0.02, normalization=1.0,
                               geometry="annulus"):
    """Return ``(coefficient kwargs, exact)`` for a corrugated-annulus stress case.

    The kwargs are ``source, diffusion=((kxx, kxy), (kxy, kyy)), beta, reaction,
    boundary_condition``, as for the NumPy case factories. ``normalization`` is the
    frozen velocity normalization; the hole radius is ``0.65 - neck_width``.
    """
    if geometry != "annulus":
        raise NotImplementedError("compiled closed-loop laws cover the annulus geometry only")
    if not math.isfinite(normalization) or normalization <= 0:
        raise ValueError("A frozen positive velocity normalization is required")
    params = (0.65-neck_width, epsilon, speed/normalization, reaction, VARIANT_CODES[variant])
    law = lambda function, name: pointwise_law(function, params=params, name=f"closed_loop_{variant}_{name}")
    exact = law(_exact, "exact")
    kxx, kxy, kyy = law(_kxx, "kxx"), law(_kxy, "kxy"), law(_kyy, "kyy")
    return dict(source=law(_source, "source"), diffusion=((kxx, kxy), (kxy, kyy)),
                beta=(law(_beta_x, "beta_x"), law(_beta_y, "beta_y")), reaction=reaction,
                boundary_condition=exact), exact


__all__ = ["VARIANT_CODES", "closed_loop_pointwise_case"]
