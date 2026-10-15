"""Axisymmetric hub, edge waves and the synchronism window of the guiding-center magnetron.

DIAG (no time integration). Model of
:func:`scripts.guiding_center.cases.guiding_center_cases.magnetron`: -lap phi = rho
on r_c < r < r_a, phi(r_c) = 0, phi(r_a) = -V + V1 cos(N(theta - W t)), drift
u_theta = -d_r phi, u_r = (1/r) d_theta phi, hub rho0 = rho_c chi(r).

* :class:`HubEquilibrium`: the axisymmetric state, its enclosed charge Q(r),
  the induced cathode charge Q_c (zero for space-charge-limited emission) and
  the angular drift Omega(r) = (Q_c + Q(r))/(2 pi r^2).
* :meth:`HubEquilibrium.edge_wave_speed`: the angular phase velocity of the
  Kelvin (diocotron) wave of mode m on a sharp hub edge, Omega(r_h) - rho_c G_m(r_h, r_h),
  G_m the annulus Green function with Dirichlet walls.
* :meth:`HubEquilibrium.channel_threshold`: in the frame of the wave, electrons
  follow the contours of psi = phi0(r) + W r^2/2 + V1 g_N(r) cos(N theta'),
  g_N the harmonic extension of the anode wave. A contour joins the hub top to
  the anode iff max(A - V1 g) <= min(A + V1 g) over [r_h, r_a], A = phi0 + W r^2/2;
  the smallest such V1 is the drift-model synchronism (Buneman-Hartree)
  threshold without the space charge of the spokes.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["HubEquilibrium"]


class HubEquilibrium:
    """Axisymmetric magnetron hub on a Gauss-Legendre radial grid."""

    def __init__(self, parameters: dict, *, panels: int = 400, points: int = 8):
        p = dict(parameters)
        self.rc, self.ra = float(p["cathode"]), float(p.get("anode", 1.0))
        self.rh, self.rho_c, self.delta = float(p["hub_top"]), float(p["hub_density"]), float(p["hub_edge"])
        self.V, self.N = float(p["voltage"]), int(p.get("wave_mode", 0))
        nodes, weights = np.polynomial.legendre.leggauss(points)
        breaks = np.linspace(self.rc, self.ra, panels + 1)
        self.r = np.concatenate([(a + b) / 2 + (b - a) * nodes / 2 for a, b in zip(breaks, breaks[1:])])
        self.w = np.concatenate([weights * (b - a) / 2 for a, b in zip(breaks, breaks[1:])])
        self.rho = self.density(self.r)
        self.Q = 2 * math.pi * np.cumsum(self.rho * self.r * self.w)  # sorted grid: running integral
        self.Q -= math.pi * self.rho * self.r * self.w  # midpoint correction of the running sum
        log_ratio = math.log(self.ra / self.rc)
        # phi(ra) - phi(rc) = -V and -r phi' = (Q_c + Q)/(2 pi)
        self.Qc = (2 * math.pi * self.V - np.sum(self.Q / self.r * self.w)) / log_ratio
        self.omega = (self.Qc + self.Q) / (2 * math.pi * self.r ** 2)
        self.phi = -np.cumsum((self.Qc + self.Q) / (2 * math.pi * self.r) * self.w)

    def density(self, r):
        return self.rho_c * 0.5 * (1.0 - np.tanh((np.asarray(r) - self.rh) / self.delta))

    def density_derivative(self, r):
        return -self.rho_c * 0.5 / self.delta / np.cosh((np.asarray(r) - self.rh) / self.delta) ** 2

    def angular_velocity(self, r):
        return np.interp(r, self.r, self.omega)

    def wave_profile(self, m: int | None = None, r=None):
        """Harmonic extension g_m(r) of cos(m theta) on the anode (zero on the cathode)."""
        m = self.N if m is None else int(m)
        r = self.r if r is None else np.asarray(r)
        a = (self.rc / self.ra) ** (2 * m)
        return ((r / self.ra) ** m - a * (self.ra / r) ** m) / (1 - a)

    def green_diagonal(self, m: int, r):
        """G_m(r, r) of -(1/r)(r g')' + m^2 g/r^2 with Dirichlet walls at r_c and r_a."""
        r = np.asarray(r, dtype=float)
        a = (self.rc / self.ra) ** (2 * m)
        return (1 - (self.rc / r) ** (2 * m)) * (1 - (r / self.ra) ** (2 * m)) / (2 * m * (1 - a))

    def _operator(self, m: int):
        R, S = self.r[:, None], self.r[None, :]
        lo, hi = np.minimum(R, S), np.maximum(R, S)
        a = self.rc ** (2 * m) / self.ra ** (2 * m)
        green = ((lo / self.ra) ** m - a * (self.ra / lo) ** m) * ((self.ra / hi) ** m - (hi / self.ra) ** m) \
            / (2 * m * (1 - a))
        operator = (m * self.density_derivative(self.r) / self.r)[:, None] * green * (self.r * self.w)[None, :]
        operator[np.diag_indices_from(operator)] += m * self.omega
        return operator

    def growth_rates(self, modes=range(1, 13)) -> dict[int, float]:
        """Largest Im(omega) of the free modes (zero: the attached hub is stable)."""
        return {int(m): float(np.linalg.eigvals(self._operator(int(m))).imag.max()) for m in modes}

    def edge_wave_speed(self, m: int | None = None) -> float:
        """Angular phase velocity of the Kelvin wave of mode m on a sharp hub edge.

        For the smooth edge the wave is a quasi-mode: every W in the range of
        Omega over the hub is resonant with a critical layer inside it.
        """
        m = self.N if m is None else int(m)
        return float(self.angular_velocity(self.rh) - self.rho_c * self.green_diagonal(m, self.rh))

    def channel_threshold(self, speed: float, *, mode: int | None = None) -> float:
        """Smallest V1 for which a wave-frame contour joins the hub top to the anode."""
        g = self.wave_profile(mode)
        outside = self.r >= self.rh
        A = self.phi[outside] + 0.5 * float(speed) * self.r[outside] ** 2
        gs = g[outside]
        low, high = 0.0, 100.0 * max(self.V, 1e-12)
        for _ in range(80):
            middle = 0.5 * (low + high)
            if (A - middle * gs).max() <= (A + middle * gs).min():
                high = middle
            else:
                low = middle
        return high

    def synchronism_window(self, amplitude: float, *, mode: int | None = None, speeds=None):
        """Wave speeds W whose channel threshold is below ``amplitude`` (endpoints, or None)."""
        speeds = np.linspace(0.3 * self.omega.max(), 1.3 * self.omega.max(), 401) if speeds is None \
            else np.asarray(speeds)
        thresholds = np.array([self.channel_threshold(W, mode=mode) for W in speeds])
        open_ = speeds[thresholds <= float(amplitude)]
        return None if open_.size == 0 else (float(open_.min()), float(open_.max()))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cathode", type=float, default=0.5)
    parser.add_argument("--hub-top", type=float, default=0.65)
    parser.add_argument("--hub-density", type=float, default=1.0)
    parser.add_argument("--hub-edge", type=float, default=0.01)
    parser.add_argument("--voltage", type=float, default=0.1)
    parser.add_argument("--wave-mode", type=int, default=6)
    parser.add_argument("--amplitudes", default="0.0125,0.025,0.05")
    args = parser.parse_args()
    hub = HubEquilibrium(dict(cathode=args.cathode, hub_top=args.hub_top, hub_density=args.hub_density,
                              hub_edge=args.hub_edge, voltage=args.voltage, wave_mode=args.wave_mode))
    print(f"hub charge {hub.Q[-1]:.4f}, cathode charge {hub.Qc:.4f} (vacuum {2*math.pi*hub.V/math.log(hub.ra/hub.rc):.4f})")
    print("Omega: " + ", ".join(f"{r:g}:{hub.angular_velocity(r):.4f}" for r in (hub.rc, hub.rh, .8, 1.)))
    print(f"max free growth rate (m=1..12): {max(hub.growth_rates().values()):.2e}")
    print(f"sharp-edge Kelvin wave of mode {hub.N}: W = {hub.edge_wave_speed():.4f}; "
          f"Omega over the hub {hub.omega[hub.r <= hub.rh].min():.4f}..{hub.omega[hub.r <= hub.rh].max():.4f}")
    for amplitude in (float(a) for a in args.amplitudes.split(",")):
        print(f"V1={amplitude:g} (V1/V={amplitude/hub.V:.3f}): synchronism window W in {hub.synchronism_window(amplitude)}")


if __name__ == "__main__":
    main()
