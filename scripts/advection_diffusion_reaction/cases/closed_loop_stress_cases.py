"""Analytic coefficients for the September 2026 ADR stress proposal.

NumPy only: importing this module never meshes, assembles, solves or JITs.
All derivatives are analytic; finite differences are reserved for tests.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import importlib.util
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

GEOMETRIES = ("annulus", "square")
VARIANTS = ("trap", "cross", "orthogonal")
LEVELS = {
    "entry": dict(epsilon=1e-4, speed=20.0, neck_width=0.04),
    "main": dict(epsilon=1e-6, speed=50.0, neck_width=0.02),
    "severe": dict(epsilon=1e-8, speed=100.0, neck_width=0.015),
}


@dataclass(frozen=True)
class StressParameters:
    variant: str = "trap"
    epsilon: float = 1e-6
    speed: float = 50.0
    neck_width: float = 0.02
    reaction: float = 1e-3
    geometry: str = "annulus"

    def __post_init__(self):
        if self.geometry not in GEOMETRIES:
            raise ValueError(f"Unknown stress geometry: {self.geometry}")
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown stress variant: {self.variant}")
        if not 0 < self.epsilon <= 1 or not math.isfinite(self.epsilon):
            raise ValueError("epsilon must be finite and in (0, 1]")
        if not math.isfinite(self.speed) or self.speed <= 0:
            raise ValueError("speed must be positive and finite")
        if not 0 < self.neck_width < 0.65:
            raise ValueError("neck_width must be in (0, 0.65)")
        if not math.isfinite(self.reaction) or self.reaction <= 0:
            raise ValueError("reaction must be positive and finite")

    @property
    def hole_radius(self):
        return 0.65 - self.neck_width

    def to_dict(self):
        values = asdict(self)
        if self.geometry == "annulus":
            values.pop("geometry")  # Preserve existing frozen annular records.
        else:
            values.pop("neck_width")  # Not a square-domain parameter.
        return values


@lru_cache(maxsize=1)
def square_coefficients():
    """Load package formulas without selecting a solver worktree or compiling."""
    name = "_hdgfem_square_stress_coefficients"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[3]/"hdgfem/core/square_stress_coefficients.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def square_parameters(parameters, normalization=1.0):
    return (parameters.epsilon, parameters.speed/normalization,
            parameters.reaction, VARIANTS.index(parameters.variant))


def case_exact_data(x, y, parameters):
    """Geometry-aware exact derivatives; the old annular API remains unchanged."""
    if parameters.geometry == "square":
        return square_coefficients().square_exact_data(x, y)
    return exact_data(x, y, parameters.hole_radius)


def polar_points(rho, phi, hole_radius):
    """Map independent normalized-radius/angle samples to the ideal annulus."""
    radius = hole_radius + np.asarray(rho) * (1 + 0.35*np.cos(9*np.asarray(phi)) - hole_radius)
    return radius*np.cos(phi), radius*np.sin(phi)


def coordinates(x, y, hole_radius):
    """Return rho and chi as (value, dx, dy, dxx, dxy, dyy)."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    r = np.hypot(x, y)
    if np.any(r == 0):
        raise ValueError("Polar stress coefficients are undefined at the origin")
    phi = np.arctan2(y, x)
    width = 1 + 0.35*np.cos(9*phi) - hole_radius
    wp, wpp = -3.15*np.sin(9*phi), -28.35*np.cos(9*phi)
    rho = (r-hole_radius)/width
    rx, ry, px, py = x/r, y/r, -y/r**2, x/r**2
    rxx, rxy, ryy = y*y/r**3, -x*y/r**3, x*x/r**3
    pxx, pxy, pyy = 2*x*y/r**4, (y*y-x*x)/r**4, -2*x*y/r**4
    dr, dp = 1/width, -rho*wp/width
    drp, dpp = -wp/width**2, rho*(2*(wp/width)**2-wpp/width)
    gx, gy = dr*rx+dp*px, dr*ry+dp*py
    hxx = 2*drp*rx*px+dpp*px*px+dr*rxx+dp*pxx
    hxy = drp*(rx*py+ry*px)+dpp*px*py+dr*rxy+dp*pxy
    hyy = 2*drp*ry*py+dpp*py*py+dr*ryy+dp*pyy
    rho_data = (rho, gx, gy, hxx, hxy, hyy)
    a, q = 2*np.pi, 0.05*np.sin(5*phi)
    qp, qpp = 0.25*np.cos(5*phi), -1.25*np.sin(5*phi)
    sn, cs = np.sin(a*rho), np.cos(a*rho)
    fr, fp = 1+a*q*cs, qp*sn
    frr, frp, fpp = -a*a*q*sn, a*qp*cs, qpp*sn
    chi = (rho+q*sn, fr*gx+fp*px, fr*gy+fp*py,
           fr*hxx+fp*pxx+frr*gx*gx+2*frp*gx*px+fpp*px*px,
           fr*hxy+fp*pxy+frr*gx*gy+frp*(gx*py+gy*px)+fpp*px*py,
           fr*hyy+fp*pyy+frr*gy*gy+2*frp*gy*py+fpp*py*py)
    return rho_data, chi


def exact_data(x, y, hole_radius):
    """Manufactured primal value, gradient and Hessian in Cartesian coordinates."""
    _, (chi, cx, cy, cxx, cxy, cyy) = coordinates(x, y, hole_radius)
    a = 2*np.pi
    u, du, ddu = np.sin(a*chi), a*np.cos(a*chi), -a*a*np.sin(a*chi)
    data = [u, du*cx, du*cy, du*cxx+ddu*cx*cx,
            du*cxy+ddu*cx*cy, du*cyy+ddu*cy*cy]
    for amplitude, m, n in ((0.25, 6, 5), (0.25*0.35, 11, 9)):
        a, b = m*np.pi, n*np.pi
        sx, cx, sy, cy = np.sin(a*x), np.cos(a*x), np.sin(b*y), np.cos(b*y)
        v = amplitude*sx*sy
        terms = (v, amplitude*a*cx*sy, amplitude*b*sx*cy,
                 -a*a*v, amplitude*a*b*cx*cy, -b*b*v)
        data = [old+term for old, term in zip(data, terms)]
    return tuple(data)


def diffusion_data(x, y, parameters):
    """Return Kxx, Kxy, Kyy and the two components of div(K)."""
    if parameters.geometry == "square":
        return square_coefficients().square_diffusion_data(x, y, square_parameters(parameters))
    eps = parameters.epsilon
    if parameters.variant == "orthogonal":
        c, s = np.cos(np.pi/7), np.sin(np.pi/7)
        zero = np.zeros(np.broadcast(x, y).shape)
        return (zero+c*c+eps*s*s, zero+(1-eps)*c*s,
                zero+s*s+eps*c*c, zero, zero)
    _, (_, gx, gy, hxx, hxy, hyy) = coordinates(x, y, parameters.hole_radius)
    norm = np.hypot(gx, gy)
    bx, by = gy/norm, -gx/norm
    nx, ny = (gx*hxx+gy*hxy)/norm, (gx*hxy+gy*hyy)/norm
    bxx, bxy = hxy/norm-bx*nx/norm, hyy/norm-bx*ny/norm
    byx, byy = -hxx/norm-by*nx/norm, -hxy/norm-by*ny/norm
    scale = 1-eps
    return (eps+scale*bx*bx, scale*bx*by, eps+scale*by*by,
            scale*(2*bx*bxx+bxy*by+bx*byy),
            scale*(bxx*by+bx*byx+2*by*byy))


def unscaled_velocity(x, y, parameters):
    """Return grad-perp(psi), or g(s)e_t for the orthogonal variant."""
    if parameters.geometry == "square":
        values = (parameters.epsilon, 1.0, parameters.reaction, VARIANTS.index(parameters.variant))
        return square_coefficients().square_velocity(x, y, values)
    if parameters.variant == "orthogonal":
        c, s = np.cos(np.pi/7), np.sin(np.pi/7)
        g = (1+0.8*np.sin(13*np.pi*(c*x+s*y)))/1.8
        return -s*g, c*g
    rho_data, chi_data = coordinates(x, y, parameters.hole_radius)
    if parameters.variant == "trap":
        chi, cx, cy = chi_data[:3]
        factor = np.sin(8*np.pi*chi)
        return factor*cy, -factor*cx
    rho, rx, ry = rho_data[:3]
    envelope = 16*rho**2*(1-rho)**2
    derivative = 32*rho*(1-rho)*(1-2*rho)
    value = dx = dy = 0.0
    for amplitude, m, n, denominator in ((1.0, 13, 11, 13), (0.35, 17, 19, 19)):
        a, b, d = m*np.pi, n*np.pi, denominator*np.pi
        sx, cx, sy, cy = np.sin(a*x), np.cos(a*x), np.sin(b*y), np.cos(b*y)
        value = value+amplitude*sx*sy/d
        dx = dx+amplitude*a*cx*sy/d
        dy = dy+amplitude*b*sx*cy/d
    return derivative*ry*value+envelope*dy, -(derivative*rx*value+envelope*dx)


def estimate_normalization(parameters, *, rtol=1e-3, max_refinements=5):
    """Empirically converge peak speed on nested grids independent of FEM meshes.

    Require two successive relative peak changes below rtol. Store the entire
    ladder and refuse unconverged estimates; this is not a certified upper bound.
    """
    if not math.isfinite(rtol) or rtol <= 0 or max_refinements < 2:
        raise ValueError("Normalization needs positive rtol and at least two refinements")
    if parameters.variant == "orthogonal":
        return dict(value=1.0, converged=True, method="analytic", history=[])
    if parameters.geometry == "square":
        return estimate_square_normalization(parameters, rtol=rtol, max_refinements=max_refinements)
    history, stable = [], 0
    for refinement in range(max_refinements+1):
        nphi, nrho = 256*2**refinement, 64*2**refinement
        radial = np.linspace(0, 1, nrho+1)[None, :]
        peak = 0.0
        for start in range(0, nphi, 32):
            phi = (2*np.pi*np.arange(start, min(start+32, nphi))/nphi)[:, None]
            x, y = polar_points(radial, phi, parameters.hole_radius)
            vx, vy = unscaled_velocity(x, y, parameters)
            peak = max(peak, float(np.max(np.hypot(vx, vy))))
        change = abs(peak-history[-1]["peak"])/peak if history else None
        history.append(dict(angular_points=nphi, radial_intervals=nrho, peak=peak,
                            relative_change=change))
        stable = stable+1 if change is not None and change <= rtol else 0
        if stable >= 2:
            return dict(value=peak, converged=True, method="nested polar sampling",
                        rtol=rtol, history=history,
                        note="Empirical convergence, not a certified continuum maximum; frozen across h/p.")
    raise RuntimeError(f"Velocity normalization did not converge: {history}")


def estimate_square_normalization(parameters, *, rtol, max_refinements):
    """Freeze a peak-speed estimate on nested Cartesian grids, not FEM nodes."""
    history, stable = [], 0
    for refinement in range(max_refinements+1):
        intervals = 128*2**refinement
        axis = np.linspace(-1, 1, intervals+1)
        peak = 0.0
        for start in range(0, len(axis), 32):
            vx, vy = unscaled_velocity(axis[None, :], axis[start:start+32, None], parameters)
            peak = max(peak, float(np.max(np.hypot(vx, vy))))
        if not math.isfinite(peak) or peak <= 0:
            raise ValueError("Square normalization has nonpositive or nonfinite speed")
        change = abs(peak-history[-1]["peak"])/peak if history else None
        history.append(dict(intervals_per_axis=intervals, peak=peak, relative_change=change))
        stable = stable+1 if change is not None and change <= rtol else 0
        if stable >= 2:
            return dict(value=peak, converged=True, method="nested Cartesian sampling",
                        rtol=rtol, history=history,
                        note="Empirical convergence, not a certified continuum maximum; frozen across h/p.")
    raise RuntimeError(f"Velocity normalization did not converge: {history}")


def make_case(parameters, normalization):
    """Return the established ADR factory contract (coefficient kwargs, exact)."""
    if not math.isfinite(normalization) or normalization <= 0:
        raise ValueError("A frozen positive velocity normalization is required")
    if parameters.variant == "orthogonal" and normalization != 1:
        raise ValueError("Orthogonal transport uses its analytic normalization of one")

    def exact(x, y):
        return case_exact_data(x, y, parameters)[0]

    def velocity(x, y):
        vx, vy = unscaled_velocity(x, y, parameters)
        scale = parameters.speed/normalization
        return scale*vx, scale*vy

    def source(x, y):
        u, ux, uy, uxx, uxy, uyy = case_exact_data(x, y, parameters)
        kxx, kxy, kyy, divx, divy = diffusion_data(x, y, parameters)
        vx, vy = velocity(x, y)
        return (-kxx*uxx-2*kxy*uxy-kyy*uyy-divx*ux-divy*uy
                +vx*ux+vy*uy+parameters.reaction*u)

    def component(index):
        return lambda x, y: diffusion_data(x, y, parameters)[index]

    kxx, kxy, kyy = (component(i) for i in range(3))
    return dict(source=source, diffusion=((kxx, kxy), (kxy, kyy)),
                beta=(lambda x, y: velocity(x, y)[0], lambda x, y: velocity(x, y)[1]),
                reaction=parameters.reaction, boundary_condition=exact), exact
