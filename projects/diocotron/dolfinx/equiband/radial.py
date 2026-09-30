"""Independent one-dimensional disk reference (not a production hot path)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import brentq
from .nonlinearities import Window


def sharp_ring_seed(delta, radius=1., inner_fraction=0.50):
    """Analytical sharp annulus, used only as an initial guess for smooth solves."""
    a = inner_fraction*radius
    def width(b):
        return (b*b-a*a)/4-a*a/2*np.log(b/a)
    if delta >= width(radius):
        raise ValueError("no analytical seed for this inner radius and threshold width")
    b = brentq(lambda b: width(b)-delta, a*(1+1e-8), radius)
    lower = (b*b-a*a)/2*np.log(radius/b)
    upper = lower+delta
    m = (lower+upper)/2
    def profile(r):
        r = np.asarray(r)
        safe = np.maximum(r, a)
        inside = upper-(safe*safe-a*a)/4+a*a/2*np.log(safe/a)
        outside = (b*b-a*a)/2*np.log(radius/np.maximum(r, b))
        value = np.where(r < a, upper, np.where(r < b, inside, outside))
        derivative = np.where(r < a, 0., np.where(r < b, -safe/2+a*a/(2*safe), -(b*b-a*a)/(2*safe)))
        return np.array([value, derivative])
    return m, profile


@dataclass
class RadialReference:
    solution: object
    m: float
    radius: float
    thresholds: tuple[float, float]

    def values(self, r):
        return self.solution.sol(r)[0]

    def crossings(self):
        lo, hi = self.thresholds
        return np.array([brentq(lambda r: float(self.values(r)-level), 0., self.radius)
                         for level in (hi, self.m, lo)])

    @property
    def distance(self):
        return self.crossings()[1]/self.radius


def solve_radial(band, radius=1., m=None, initial=None, tolerance=1e-8):
    seed_m, seed_profile = sharp_ring_seed(band.threshold_width_delta, radius)
    m = seed_m if m is None else float(m)
    r = np.linspace(0., radius, 400)
    y = seed_profile(r) if initial is None else initial.solution.sol(r)
    window = Window(band)
    def fun(r, y):
        return np.vstack((y[1], -window.value(y[0], m)))
    def jac(r, y):
        result = np.zeros((2, 2, len(r)))
        result[0, 1], result[1, 0] = 1., -window.derivative(y[0], m)
        return result
    solution = solve_bvp(fun, lambda a, b: np.array([a[1], b[0]]), r, y,
                         S=np.array([[0., 0.], [0., -1.]]), fun_jac=jac,
                         tol=tolerance, max_nodes=40000)
    if not solution.success:
        raise RuntimeError("RADIAL_REFERENCE_NOT_CONVERGED: " + solution.message)
    return RadialReference(solution, m, radius, band.thresholds(m))
