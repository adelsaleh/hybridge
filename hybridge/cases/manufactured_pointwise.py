"""Manufactured ADR coefficients as compiled, parallel :class:`~hybridge.core.pointwise.PointwiseLaw` objects.

Three families of the ADR case catalogue (``scripts/advection_diffusion_reaction/cases``), whose
NumPy closures sample on one core:

* ``sine``: ``u = sin(pi x) sin(pi y)`` with a constant tensor, velocity and reaction
  (legacy ``trigonometric``, ``advection_dominated``, ``anisotropic``);
* ``cellular``: the two-mode Fourier solution with a constant drift plus the divergence-free
  cellular flow (``oscillatory_rhs``, ``cellular*``);
* ``quadratic``: ``u = 1 + x^2 + 2 y^2`` with ``beta = (1 + x, -1/2 + y)`` (legacy
  ``variable_velocity``).

The formulas are ``register_jitable`` (NumPy-callable too) and the sources are the conservative
``-div(K grad u) + div(beta u) + reaction u``. Each law evaluates one component (no bundled
multi-component input exists yet; see TODO.md, Coefficient Inputs).
"""
from __future__ import annotations

import numpy as np

from hybridge.core.pointwise import pointwise_law

try:
    from numba.extending import register_jitable
except ImportError:
    def register_jitable(function):
        """Return ``function`` unchanged when Numba is unavailable (plain NumPy evaluation)."""
        return function


@register_jitable
def sine_mode(x, y):
    """Return ``u, ux, uy, uxx, uxy, uyy`` of ``u = sin(pi x) sin(pi y)``."""
    pi = np.pi
    sx, cx, sy, cy = np.sin(pi*x), np.cos(pi*x), np.sin(pi*y), np.cos(pi*y)
    u = sx*sy
    return u, pi*cx*sy, pi*sx*cy, -pi*pi*u, pi*pi*cx*cy, -pi*pi*u


@register_jitable
def fourier_mode(x, y, amplitude, m, n):
    """Return ``u, ux, uy, uxx, uxy, uyy`` of ``amplitude sin(pi m x) sin(pi n y)``."""
    a, b = np.pi*m, np.pi*n
    sx, cx, sy, cy = np.sin(a*x), np.cos(a*x), np.sin(b*y), np.cos(b*y)
    u = amplitude*sx*sy
    return u, amplitude*a*cx*sy, amplitude*b*sx*cy, -a*a*u, amplitude*a*b*cx*cy, -b*b*u


@register_jitable
def cellular_velocity(x, y, amplitude, wave, drift_x, drift_y):
    """Return the constant drift plus the cellular flow of streamfunction ``sin(wave x) sin(wave y)``."""
    return (drift_x+amplitude*np.sin(wave*x)*np.cos(wave*y),
            drift_y-amplitude*np.cos(wave*x)*np.sin(wave*y))


@register_jitable
def _tensor_source(k00, k01, k11, beta_x, beta_y, reaction, u, ux, uy, uxx, uxy, uyy):
    """``-div(K grad u) + beta . grad u + reaction u`` for a constant ``K`` and ``div(beta) = 0``."""
    return -k00*uxx-2*k01*uxy-k11*uyy+beta_x*ux+beta_y*uy+reaction*u


# sine: v = (k00, k01, k11, reaction, beta_x, beta_y).
def _sine_exact(x, y, t, v):
    """Evaluate the exact sine-mode solution."""
    return sine_mode(x, y)[0]


def _sine_source(x, y, t, v):
    """Evaluate the tensor-diffusion source for the sine-mode solution."""
    u, ux, uy, uxx, uxy, uyy = sine_mode(x, y)
    return _tensor_source(v[0], v[1], v[2], v[4], v[5], v[3], u, ux, uy, uxx, uxy, uyy)


# cellular: v = (k00, k01, k11, reaction, drift_x, drift_y, amplitude, wave, a1, m1, n1, a2, m2, n2).
@register_jitable
def _cellular_solution(x, y, v):
    """Combine two Fourier modes and their first and second derivatives."""
    u, ux, uy, uxx, uxy, uyy = fourier_mode(x, y, v[8], v[9], v[10])
    u2, ux2, uy2, uxx2, uxy2, uyy2 = fourier_mode(x, y, v[11], v[12], v[13])
    return u+u2, ux+ux2, uy+uy2, uxx+uxx2, uxy+uxy2, uyy+uyy2


def _cellular_exact(x, y, t, v):
    """Evaluate the exact cellular-flow solution."""
    return _cellular_solution(x, y, v)[0]


def _cellular_beta_x(x, y, t, v):
    """Evaluate the x component of cellular advection."""
    return cellular_velocity(x, y, v[6], v[7], v[4], v[5])[0]


def _cellular_beta_y(x, y, t, v):
    """Evaluate the y component of cellular advection."""
    return cellular_velocity(x, y, v[6], v[7], v[4], v[5])[1]


def _cellular_source(x, y, t, v):
    """Evaluate the manufactured source for cellular advection and tensor diffusion."""
    u, ux, uy, uxx, uxy, uyy = _cellular_solution(x, y, v)
    beta_x, beta_y = cellular_velocity(x, y, v[6], v[7], v[4], v[5])
    return _tensor_source(v[0], v[1], v[2], beta_x, beta_y, v[3], u, ux, uy, uxx, uxy, uyy)


# quadratic: v = (diffusion, reaction).
def _quadratic_exact(x, y, t, v):
    """Evaluate the quadratic manufactured solution."""
    return 1+x*x+2*y*y


def _quadratic_beta_x(x, y, t, v):
    """Evaluate the x component of the quadratic case velocity."""
    return 1+x


def _quadratic_beta_y(x, y, t, v):
    """Evaluate the y component of the quadratic case velocity."""
    return -.5+y


def _quadratic_source(x, y, t, v):
    # -k (uxx + uyy) + beta . grad u + div(beta) u + reaction u, with div(beta) = 2.
    """Evaluate the conservative ADR source for the quadratic solution."""
    return -6*v[0]+(1+x)*2*x+(-.5+y)*4*y+(2+v[1])*(1+x*x+2*y*y)


def _tensor(diffusion):
    """Normalize a scalar or symmetric diffusion tensor into its three components."""
    tensor = np.asarray(diffusion, dtype=float)
    tensor = np.eye(2)*tensor if tensor.ndim == 0 else tensor
    if tensor.shape != (2, 2) or tensor[0, 1] != tensor[1, 0]:
        raise ValueError("a constant symmetric 2x2 diffusion tensor (or scalar) is required")
    return float(tensor[0, 0]), float(tensor[0, 1]), float(tensor[1, 1])


def sine_pointwise_case(*, diffusion, reaction, beta=(1.0, 0.5)):
    """Return ``(coefficient kwargs, exact)`` for ``u = sin(pi x) sin(pi y)``.

    The kwargs are ``source, diffusion, beta, reaction, boundary_condition``; ``diffusion`` (a
    constant scalar or symmetric tensor), ``beta`` and ``reaction`` are constants and passed through.
    """
    params = (*_tensor(diffusion), reaction, *beta)
    exact = pointwise_law(_sine_exact, params=params, name="sine_exact")
    return dict(source=pointwise_law(_sine_source, params=params, name="sine_source"), diffusion=diffusion,
                beta=tuple(beta), reaction=reaction, boundary_condition=exact), exact


def cellular_pointwise_case(*, diffusion, reaction, amplitude, frequency, modes, drift=(1.0, 0.5)):
    """Return ``(coefficient kwargs, exact)`` for the two-mode Fourier solution in a cellular flow.

    ``modes`` are the two ``(amplitude, m, n)`` triples; the velocity is ``drift`` plus the cellular
    flow of ``amplitude`` and wave number ``pi frequency`` (the constant ``drift`` when the amplitude
    is zero). ``diffusion`` is a constant scalar or symmetric tensor and passed through.
    """
    (first, second) = modes
    params = (*_tensor(diffusion), reaction, *drift, amplitude, np.pi*frequency, *first, *second)
    law = lambda function, name: pointwise_law(function, params=params, name=f"cellular_{name}")
    exact = law(_cellular_exact, "exact")
    beta = tuple(drift) if amplitude == 0 else (law(_cellular_beta_x, "beta_x"), law(_cellular_beta_y, "beta_y"))
    return dict(source=law(_cellular_source, "source"), diffusion=diffusion, beta=beta, reaction=reaction,
                boundary_condition=exact), exact


def quadratic_pointwise_case(*, diffusion=1.0, reaction=0.5):
    """Return ``(coefficient kwargs, exact)`` for ``u = 1 + x^2 + 2 y^2``, ``beta = (1 + x, -1/2 + y)``."""
    params = (float(diffusion), float(reaction))
    law = lambda function, name: pointwise_law(function, params=params, name=f"quadratic_{name}")
    exact = law(_quadratic_exact, "exact")
    return dict(source=law(_quadratic_source, "source"), diffusion=diffusion,
                beta=(law(_quadratic_beta_x, "beta_x"), law(_quadratic_beta_y, "beta_y")), reaction=reaction,
                boundary_condition=exact), exact


__all__ = ["cellular_pointwise_case", "cellular_velocity", "fourier_mode", "quadratic_pointwise_case",
           "sine_mode", "sine_pointwise_case"]
