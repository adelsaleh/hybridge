"""ADR coefficients of the decoupled n-Gamma D-BDF2 step in a poloidal plane.

Two geometries share one formulation with a measure weight ``W``:

* ``"cartesian"``: the poloidal-plane model on ``Omega_{x,y} subset R^2`` with
  the plain divergence; mesh coordinates are ``(x, y)`` and ``W = 1``;
* ``"axisymmetric"``: the toroidally symmetric model; mesh coordinates are
  ``(R, Z)``, the divergence is ``(1/R) d_R(R F_R) + d_Z F_Z`` and every
  equation is multiplied by ``W = R`` (the first mesh coordinate).

One step solves two scalar HDG ADR problems with

    P = I - b_p b_p^T,   beta = W u* b_p,   K_n = W D P,   K_Gamma = W mu P,
    reaction W*alpha,
    density source   W (S_n + h_n),
    momentum source  W (S_Gamma + h_Gamma - c_s^2 b_p . grad_p n_h^{k+1}),

where ``u* = Gamma*/max(n*, n_floor)`` is evaluated pointwise (volume,
per-incidence face and recovery points) from the extrapolated DG fields,
``h_w`` is the BDF history combination and ``n_h^{k+1}`` is the newly solved
density. ``b_p`` is a callable of the mesh coordinates, used as supplied
without renormalization (``|b_p| < 1`` keeps ``P`` positive definite).

Coefficient callables take NumPy or CuPy coordinate arrays; element-local
terms are :class:`hdgfem.ElementCoefficient` objects evaluated with
``xp=cupy`` on the raw-CUDA path, so no field leaves the device. Returned ADR
fluxes carry the weight ``W``. On the host the pointwise combinations run in
element chunks on the ``hdgfem.core.host_threads`` pool.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from hdgfem import DGField, DGSpace, ElementCoefficient, field_gradient_at_ref, field_values_at_ref
from hdgfem.core.element_coefficients import physical_points
from hdgfem.core.host_threads import elementwise

PoloidalField = Callable[..., tuple]
GEOMETRIES = ("cartesian", "axisymmetric")


def check_geometry(geometry: str) -> str:
    """Validate and return ``geometry``; there is deliberately no default."""
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry must be one of {GEOMETRIES}; got {geometry!r}")
    return geometry


def measure_weight(geometry: str, first, second):
    """Return ``W``: ``1`` (scalar) for Cartesian, the first coordinate ``R`` for axisymmetric."""
    return first if check_geometry(geometry) == "axisymmetric" else 1.


def projector_components(b_poloidal: PoloidalField, first, second):
    """Return ``(p00, p01, p11)`` of ``P = I - b_p b_p^T`` at mesh coordinates."""
    b_1, b_2 = b_poloidal(first, second)
    return 1. - b_1*b_1, -b_1*b_2, 1. - b_2*b_2


def diffusion_tensor(coefficient: float, b_poloidal: PoloidalField, *, geometry: str):
    """Return the symmetric ``W*coefficient*P`` as ``(k00, k01, k11)`` callables.

    The three-component form is classified as ``variable-symmetric``; it is
    elliptic wherever ``W > 0`` and ``|b_p| < 1``.
    """
    check_geometry(geometry)
    value = float(coefficient)
    if not value > 0.:
        raise ValueError("diffusion coefficient must be positive")

    def component(index):
        def evaluate(first, second):
            weight = measure_weight(geometry, first, second)
            return value*weight*projector_components(b_poloidal, first, second)[index]
        return evaluate
    return component(0), component(1), component(2)


def reaction(alpha: float, *, geometry: str):
    """Return the reaction ``W*alpha``: a scalar (Cartesian) or ``R*alpha`` callable."""
    value = float(alpha)
    if check_geometry(geometry) == "cartesian":
        return value
    return lambda first, second: value*first


def _device(xp) -> bool:
    return xp is not np


def _pointwise(device, function, *arrays):
    """``function(*arrays)``; host arrays are evaluated in parallel element chunks."""
    return function(*arrays) if device else elementwise(function, *arrays)


def _coordinates(space, points, xp):
    mapped = physical_points(space, points, xp=xp)
    return mapped[..., 0], mapped[..., 1]


def advection(space: DGSpace, n_star: DGField, gamma_star: DGField, b_poloidal: PoloidalField,
              density_floor: float, *, geometry: str) -> ElementCoefficient:
    """Return ``beta = W*(Gamma*/max(n*, floor))*b_p`` evaluated pointwise.

    Only the velocity denominator is clamped; the stored fields are untouched.
    Use :func:`density_floor_diagnostics` for the clamp counts of a step.
    """
    check_geometry(geometry)
    floor = float(density_floor)
    if not floor > 0.:
        raise ValueError("density floor must be positive")

    def function(points, *, xp, t=None):
        device = _device(xp)
        first, second = _coordinates(space, points, xp)
        density = field_values_at_ref(n_star, points, device=device)
        momentum = field_values_at_ref(gamma_star, points, device=device)

        def combine(first, second, density, momentum):
            velocity = momentum/xp.maximum(density, floor)
            velocity = velocity*measure_weight(geometry, first, second)
            b_1, b_2 = b_poloidal(first, second)
            return xp.stack((velocity*b_1, velocity*b_2), axis=-1)
        return _pointwise(device, combine, first, second, density, momentum)
    return ElementCoefficient(function, space.mesh, 2, "W_u_star_b_p")


def density_source(space: DGSpace, source: Callable, history: DGField, *, geometry: str) -> ElementCoefficient:
    """Return ``W*(S_n + h_n)``; ``source`` takes mesh coordinates and is bound to the new time."""
    check_geometry(geometry)

    def function(points, *, xp, t=None):
        first, second = _coordinates(space, points, xp)

        def combine(first, second, history_values):
            return measure_weight(geometry, first, second)*(source(first, second) + history_values)
        return _pointwise(_device(xp), combine, first, second,
                          field_values_at_ref(history, points, device=_device(xp)))
    return ElementCoefficient(function, space.mesh, 1, "density_source")


def momentum_source(space: DGSpace, source: Callable, history: DGField, density: DGField,
                    b_poloidal: PoloidalField, sound_speed: float = 1., *, geometry: str) -> ElementCoefficient:
    """Return ``W*(S_Gamma + h_Gamma - c_s^2 b_p . grad n_h)`` with the new density ``n_h``."""
    check_geometry(geometry)
    cs2 = float(sound_speed)**2

    def function(points, *, xp, t=None):
        device = _device(xp)
        first, second = _coordinates(space, points, xp)
        d_first, d_second = field_gradient_at_ref(density, points, device=device)

        def combine(first, second, history_values, d_first, d_second):
            b_1, b_2 = b_poloidal(first, second)
            values = source(first, second) + history_values - cs2*(b_1*d_first + b_2*d_second)
            return measure_weight(geometry, first, second)*values
        return _pointwise(device, combine, first, second, field_values_at_ref(history, points, device=device),
                          d_first, d_second)
    return ElementCoefficient(function, space.mesh, 1, "momentum_source")


@dataclass(frozen=True)
class DensityFloorReport:
    """Per-step velocity-denominator diagnostics (sampled, not certified bounds)."""

    floor: float
    min_extrapolated_density: float
    volume_clamps: int
    face_clamps: int
    volume_points: int
    face_points: int


def density_floor_diagnostics(n_star: DGField, space: DGSpace, trace_space, density_floor: float, *,
                              device: bool = False) -> DensityFloorReport:
    """Sample ``n*`` on the assembly volume and face points and count floor clamps.

    Raises ``FloatingPointError`` for nonfinite extrapolated densities: the
    floor never repairs a nonfinite state. Only four scalars leave the device.
    """
    from hdgfem.assembly.matrices_numpy import _reference_edge_points_from_1d

    face = np.asarray(_reference_edge_points_from_1d(trace_space.quads)).reshape(-1, 2)
    volume_values = field_values_at_ref(n_star, space.quad_data.Krf_quads, device=device)
    face_values = field_values_at_ref(n_star, face, device=device)
    xp = np
    if device:
        from hdgfem.backends.cupy import require_cupy
        xp = require_cupy()
    floor = float(density_floor)
    stats = xp.stack((xp.minimum(volume_values.min(), face_values.min()),
                      (volume_values < floor).sum().astype(xp.float64),
                      (face_values < floor).sum().astype(xp.float64),
                      xp.isfinite(volume_values).all().astype(xp.float64)
                      * xp.isfinite(face_values).all().astype(xp.float64)))
    values = stats.get() if device else stats
    if values[3] != 1.:
        raise FloatingPointError("extrapolated density is not finite")
    return DensityFloorReport(floor, float(values[0]), int(values[1]), int(values[2]),
                              int(volume_values.size), int(face_values.size))


__all__ = [
    "DensityFloorReport",
    "GEOMETRIES",
    "advection",
    "check_geometry",
    "density_floor_diagnostics",
    "density_source",
    "diffusion_tensor",
    "measure_weight",
    "momentum_source",
    "projector_components",
    "reaction",
]
