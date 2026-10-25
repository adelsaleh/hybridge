"""Guiding-center case definitions used by the fixed-mesh runner."""

from __future__ import annotations

from hybridge.runtime.precision import REAL_DTYPE

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

ScalarCallable = Callable[[Any, Any], Any]
TimeScalarCallable = Callable[[Any, Any, float], Any]


def _zero_like_xy(x, y):
    """Return a backend-compatible zero with coordinate broadcasting."""
    return 0.0 * x + 0.0 * y


def _array_namespace(*arrays):
    """Return NumPy or CuPy for the provided arrays without requiring CuPy."""
    for array in arrays:
        module = type(array).__module__.split(".", 1)[0]
        if module == "cupy" or hasattr(array, "__cuda_array_interface__"):
            import cupy

            return cupy
    return np


def _xy_arrays(x, y):
    xp = _array_namespace(x, y)
    return xp, xp.asarray(x, dtype=REAL_DTYPE), xp.asarray(y, dtype=REAL_DTYPE)


@dataclass(frozen=True)
class GuidingCenterCase:
    """Concrete time-dependent data for one guiding-center run."""

    key: str
    description: str
    initial_density: ScalarCallable
    potential_boundary: TimeScalarCallable
    density_boundary: TimeScalarCallable | None
    density_transport_boundary_mode: str
    default_domain: str
    potential_boundary_constant: float | None = None
    density_is_vorticity: bool = False
    equilibrium_density: ScalarCallable | None = None
    equilibrium_radial_derivative: Callable[[Any], Any] | None = None
    exact_density: TimeScalarCallable | None = None
    exact_potential: TimeScalarCallable | None = None
    negative_laplacian_potential: TimeScalarCallable | None = None
    exact_flux: Callable[[Any, Any, float], tuple[Any, Any]] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    exact_density_gradient: Callable[[Any, Any, float], tuple[Any, Any]] | None = None
    exact_potential_gradient: Callable[[Any, Any, float], tuple[Any, Any]] | None = None
    # Sharp radial band on an interface-conforming mesh: the runner projects
    # ``band_modulation`` and multiplies it by the elementwise band indicator
    # (element centroid radius in [inner, outer]), which is exact because no
    # element crosses the band edges. ``band_modulation(x, y, perturbed)``.
    conforming_band: tuple[float, float] | None = None
    band_modulation: Callable[[Any, Any, bool], Any] | None = None
    # Forcing of the density equation d_t rho + v.grad rho = S(x, y, t) + alpha*rho:
    # an explicit source independent of rho and a constant reaction coefficient alpha.
    # ``density_source_mass(t)`` is the exact injected mass int_0^t int S, when known;
    # S vanishes outside ``density_source_window``; a separable source
    # S = density_source_rate(t) * density_source_profile(x, y) is projected once.
    density_source: TimeScalarCallable | None = None
    density_reaction: float = 0.0
    density_source_mass: Callable[[float], float] | None = None
    density_source_window: tuple[float, float] | None = None
    density_source_rate: Callable[[float], float] | None = None
    density_source_profile: ScalarCallable | None = None
    # A space-dependent reaction coefficient adds density_reaction_rate(t) *
    # density_reaction_profile(x, y) to alpha (rate 1 when None), e.g. the
    # electron-impact ionization rate nu(x) of a prescribed neutral gas.
    density_reaction_profile: ScalarCallable | None = None
    density_reaction_rate: Callable[[float], float] | None = None
    # Electrode potentials: pairs (g_k(x, y), c_k(t)); the wall potential is
    # potential_boundary + sum_k c_k(t) g_k. Poisson is solved with
    # potential_boundary alone and the discrete harmonic fields of the g_k are
    # added (scripts.guiding_center.runtime.electrodes), so a non-equipotential,
    # time-dependent wall keeps the zero-Dirichlet Poisson path.
    wall_potentials: tuple = ()
    # Record the transport outflux through the walls (zero inflow data only).
    wall_flux_diagnostics: bool = False

    imported_equilibrium: Any = None

    @property
    def has_density_forcing(self) -> bool:
        """Return whether the density equation has a source or a reaction term."""
        return (self.density_source is not None or self.density_reaction != 0.0
                or self.density_reaction_profile is not None)

    @property
    def has_exact_solution(self) -> bool:
        """Return whether exact time-dependent density and potential are available."""
        return self.exact_density is not None and self.exact_potential is not None

    def initial_density_at(self) -> ScalarCallable:
        """Return the initial-density callable."""
        return self.initial_density

    def potential_boundary_at(self, time: float) -> ScalarCallable:
        """Return a two-argument potential Dirichlet boundary callable at ``time``."""
        t = float(time)
        boundary = lambda x, y: self.potential_boundary(x, y, t)
        if self.potential_boundary_constant is not None:
            boundary._hybridge_constant_value = float(self.potential_boundary_constant)
        boundary._hybridge_time = t  # read by the electrode superposition
        return boundary

    def wall_coefficients(self, time: float) -> tuple[float, ...]:
        """Electrode weights c_k(t) of :attr:`wall_potentials`."""
        return tuple(float(rate(float(time))) for _, rate in self.wall_potentials)

    def wall_potential_at(self, time: float) -> ScalarCallable:
        """Complete wall potential at ``time``: potential_boundary + sum_k c_k(t) g_k."""
        t = float(time)
        terms = [(c, g) for c, (g, _) in zip(self.wall_coefficients(t), self.wall_potentials) if c != 0.0]

        def boundary(x, y):
            value = self.potential_boundary(x, y, t)
            for weight, pattern in terms:
                value = value + weight * pattern(x, y)
            return value

        return boundary

    def density_boundary_at(self, time: float) -> ScalarCallable | None:
        """Return a two-argument density boundary callable at ``time`` when prescribed."""
        if self.density_boundary is None:
            return None
        t = float(time)
        return lambda x, y: self.density_boundary(x, y, t)

    def exact_density_at(self, time: float) -> ScalarCallable | None:
        """Return a two-argument exact-density callable at ``time`` when available."""
        if self.exact_density is None:
            return None
        t = float(time)
        return lambda x, y: self.exact_density(x, y, t)

    def exact_potential_at(self, time: float) -> ScalarCallable | None:
        """Return a two-argument exact-potential callable at ``time`` when available."""
        if self.exact_potential is None:
            return None
        t = float(time)
        return lambda x, y: self.exact_potential(x, y, t)

    def exact_density_gradient_at(self, time: float) -> Callable | None:
        """Return the analytic physical density gradient at the requested time."""
        if self.exact_density_gradient is None:
            return None
        return lambda x, y: self.exact_density_gradient(x, y, float(time))

    def exact_potential_gradient_at(self, time: float) -> Callable | None:
        """Return the analytic physical potential gradient at the requested time."""
        if self.exact_potential_gradient is None:
            return None
        return lambda x, y: self.exact_potential_gradient(x, y, float(time))


@dataclass(frozen=True)
class GuidingCenterCaseDefinition:
    """Factory metadata for a guiding-center benchmark case."""

    key: str
    description: str
    factory: Callable[..., GuidingCenterCase]
    default_domain: str = "rectangle"
    default_params: dict[str, Any] = field(default_factory=dict)

    def build(self, **params: Any) -> GuidingCenterCase:
        """Build a concrete case, merging registry defaults with user parameters."""
        merged = dict(self.default_params)
        merged.update(params)
        case = self.factory(**merged)
        if case.key != self.key:
            raise ValueError(f"case factory for {self.key!r} returned {case.key!r}")
        return case


def rho_eq_gaussian_annulus(x, y, *, r0: float = 0.45, sigma: float = 0.03):
    """Legacy Gaussian annular equilibrium density profile."""
    xp, x_arr, y_arr = _xy_arrays(x, y)
    r = xp.sqrt(x_arr * x_arr + y_arr * y_arr)
    width = float(sigma)
    if width <= 0.0:
        raise ValueError("sigma must be positive")
    return xp.exp(-((r - float(r0)) ** 2) / (2.0 * width * width))


def diocotron_gaussian_annulus(
        *,
        k: int = 3,
        eps: float = 0.05,
        r0: float = 0.45,
        sigma: float = 0.03,
        theta_shift: float = 0.0,
        truncate: bool = False,
) -> GuidingCenterCase:
    """Return the legacy Gaussian-annulus diocotron perturbation case."""
    mode = int(k)
    amplitude = float(eps)
    radius0 = float(r0)
    width = float(sigma)
    shift = float(theta_shift)

    def equilibrium(x, y):
        return rho_eq_gaussian_annulus(x, y, r0=radius0, sigma=width)

    def initial_density(x, y):
        xp, x_arr, y_arr = _xy_arrays(x, y)
        theta = xp.arctan2(y_arr, x_arr) - shift
        return equilibrium(x_arr, y_arr) * (1.0 + amplitude * xp.cos(mode * theta))

    return GuidingCenterCase(
        key="diocotron_gaussian_annulus",
        description="Legacy Gaussian-annulus diocotron perturbation with one azimuthal mode, zero potential boundary, and zero density flux.",
        initial_density=initial_density,
        potential_boundary=lambda x, y, t: _zero_like_xy(x, y),
        density_boundary=None,
        density_transport_boundary_mode="zero-flux",
        default_domain="disc",
        potential_boundary_constant=0.0,
        equilibrium_density=equilibrium,
        parameters={"k": mode, "eps": amplitude, "r0": radius0, "sigma": width, "theta_shift": shift},
    )


def rho_eq_annular_band(
        x,
        y,
        *,
        s_minus: float = 0.79,
        s_plus: float = 0.80,
        rho_bar: float = 1.0,
        edge_width: float = 0.0,
):
    """Annular-band equilibrium, optionally with smooth tanh transitions."""
    inner = float(s_minus)
    outer = float(s_plus)
    transition = float(edge_width)
    if not 0.0 <= inner < outer:
        raise ValueError("expected 0 <= s_minus < s_plus")
    if transition < 0.0:
        raise ValueError("edge_width must be nonnegative")
    xp, x_arr, y_arr = _xy_arrays(x, y)
    r = xp.sqrt(x_arr * x_arr + y_arr * y_arr)
    if transition > 0.0:
        return 0.5 * float(rho_bar) * (
            xp.tanh((r - inner) / transition) - xp.tanh((r - outer) / transition)
        )
    return xp.where((r >= inner) & (r <= outer), float(rho_bar), 0.0)


def rho_eq_super_gaussian_annulus(
        x,
        y,
        *,
        s_bar: float = 0.795,
        s_d: float = 0.005,
        p: float = 10.0,
        rho_bar: float = 1.0,
        truncate: bool = False,
):
    """Super-Gaussian annulus ``rho_bar * exp(-abs((r-s_bar)/s_d)**p)``."""
    center = float(s_bar)
    scale = float(s_d)
    power = float(p)
    if center < 0.0:
        raise ValueError("s_bar must be nonnegative")
    if scale <= 0.0:
        raise ValueError("s_d must be positive")
    if power <= 0.0:
        raise ValueError("p must be positive")
    xp, x_arr, y_arr = _xy_arrays(x, y)
    r = xp.sqrt(x_arr * x_arr + y_arr * y_arr)
    radius = xp.abs((r - center) / scale)
    if truncate:
        # Literal compact support of Zoni--Guclu (2019), equation (35).
        # Clip before exponentiation to avoid overflow outside the annulus.
        value = xp.exp(-xp.minimum(radius, 1.0) ** power)
        return float(rho_bar) * xp.where((r >= center-scale) & (r <= center+scale), value, 0.0)
    return float(rho_bar) * xp.exp(-radius ** power)



def diocotron_k(
        *,
        k: int = 3,
        modes: tuple[int, ...] | list[int] | None = None,
        phase_scale: float = 0.0,
        epsilon: float = 0.05,
        eps: float | None = None,
        s_minus: float = 0.79,
        s_plus: float = 0.80,
        rho_bar: float = 1.0,
        edge_width: float = 0.0,
        s_bar: float | None = None,
        s_d: float | None = None,
        p: float | None = None,
        theta_shift: float = 0.0,
        truncate: bool = False,
) -> GuidingCenterCase:
    """Return an annular-band diocotron perturbation case.

    By default ``k`` is the sole azimuthal mode. When ``modes`` is supplied,
    the perturbation is ``eps * sum(cos(m * theta + phase_scale * m**2))``;
    ``k`` then selects the reference mode for the runner's harmonic diagnostics.
    The amplitude is per mode. Supplying ``p`` selects the radial
    super-Gaussian profile
    ``exp(-abs((r-s_bar)/s_d)**p)``. By default, ``s_bar`` and ``s_d`` are
    inferred from the midpoint and half-width of ``s_minus``/``s_plus``.
    """
    mode = int(k)
    if mode < 1:
        raise ValueError("k must be at least 1")
    active_modes = (mode,) if modes is None else tuple(modes)
    if not active_modes or any(int(m) != m or m < 1 for m in active_modes):
        raise ValueError("modes must contain positive integers")
    active_modes = tuple(int(m) for m in active_modes)
    if len(set(active_modes)) != len(active_modes):
        raise ValueError("modes must not contain duplicates")
    phase = float(phase_scale)
    if not np.isfinite(phase):
        raise ValueError("phase_scale must be finite")
    amplitude = float(epsilon if eps is None else eps)
    inner = float(s_minus)
    outer = float(s_plus)
    density_level = float(rho_bar)
    transition = float(edge_width)
    radial_center = 0.5 * (inner + outer) if s_bar is None else float(s_bar)
    radial_scale = 0.5 * (outer - inner) if s_d is None else float(s_d)
    radial_power = None if p is None else float(p)
    shift = float(theta_shift)

    def equilibrium(x, y):
        if radial_power is not None:
            return rho_eq_super_gaussian_annulus(
                x,
                y,
                s_bar=radial_center,
                s_d=radial_scale,
                p=radial_power,
                rho_bar=density_level,
                truncate=truncate,
            )
        return rho_eq_annular_band(
            x,
            y,
            s_minus=inner,
            s_plus=outer,
            rho_bar=density_level,
            edge_width=transition,
        )

    def radial_derivative(r):
        xp = _array_namespace(r)
        r = xp.asarray(r, dtype=REAL_DTYPE)
        if radial_power is not None:
            u = (r-radial_center)/radial_scale
            return (-radial_power/radial_scale * xp.sign(u) * xp.abs(u)**(radial_power-1)
                    * equilibrium(r, 0*r))
        lower, upper = (r-inner)/transition, (r-outer)/transition
        return .5*density_level/transition * (xp.tanh(upper)**2-xp.tanh(lower)**2)

    def initial_density(x, y):
        xp, x_arr, y_arr = _xy_arrays(x, y)
        theta = xp.arctan2(y_arr, x_arr) - shift
        modulation = 1.0 + amplitude * xp.cos(
            active_modes[0] * theta + phase * active_modes[0] ** 2
        )
        for azimuthal_mode in active_modes[1:]:
            modulation = modulation + amplitude * xp.cos(
                azimuthal_mode * theta + phase * azimuthal_mode ** 2
            )
        return equilibrium(x_arr, y_arr) * modulation

    return GuidingCenterCase(
        key="diocotron_k",
        description=f"Annular-band diocotron perturbation with azimuthal modes {active_modes}, zero potential boundary, and zero density flux.",
        initial_density=initial_density,
        potential_boundary=lambda x, y, t: _zero_like_xy(x, y),
        density_boundary=None,
        density_transport_boundary_mode="zero-flux",
        default_domain="disc",
        potential_boundary_constant=0.0,
        equilibrium_density=equilibrium,
        equilibrium_radial_derivative=(radial_derivative if
            (radial_power is not None and not truncate) or
            (radial_power is None and transition > 0) else None),
        parameters={
            "k": mode,
            "modes": active_modes,
            "phase_scale": phase,
            "epsilon": amplitude,
            "s_minus": inner,
            "s_plus": outer,
            "rho_bar": density_level,
            "edge_width": transition,
            "s_bar": radial_center,
            "s_d": radial_scale,
            "p": radial_power,
            "theta_shift": shift,
            "truncate": bool(truncate),
        },
    )


def sin2_window(start: float, duration: float):
    """Unit-integral injection window w(t) = (2/T) sin^2(pi (t-t0)/T) on [t0, t0+T] and its primitive W(t)."""
    start, duration = float(start), float(duration)
    if not duration > 0:
        raise ValueError("injection duration must be positive")

    def rate(t):
        tau = float(t) - start
        if tau <= 0.0 or tau >= duration:
            return 0.0
        return 2.0 / duration * math.sin(math.pi * tau / duration) ** 2

    def primitive(t):
        tau = min(max(float(t) - start, 0.0), duration)
        return (tau - duration / (2.0 * math.pi) * math.sin(2.0 * math.pi * tau / duration)) / duration

    return rate, primitive


def gaussian_injection(*, charges, centers=None, width=0.06, start=0.0, duration=4.0, angular_velocity=0.0,
                       inner_wall=0.0, wall=1.0) -> dict[str, Any]:
    """Explicit density source of normalized Gaussian injectors, as GuidingCenterCase fields.

    S(x, t) = w(t) sum_k Q_k exp(-|x - c_k(t)|^2/sigma^2)/(pi sigma^2), with the
    window of :func:`sin2_window` and centres c_k rotating about the origin at
    ``angular_velocity``. Each injector must lie at least 5 sigma from every wall,
    so it injects Q_k up to exp(-25) (a centred one exactly 1 - exp(-wall^2/sigma^2)).
    Returns {} without charges; otherwise the forcing fields plus a JSON ``record``.
    """
    charges = tuple(float(q) for q in charges)
    if not charges:
        return {}
    centers = tuple((0.0, 0.0) for _ in charges) if centers is None else tuple(
        (float(x), float(y)) for x, y in centers)
    width, omega = float(width), float(angular_velocity)
    if len(centers) != len(charges) or not width > 0:
        raise ValueError("one centre per injection charge and a positive width are required")
    sigma2 = width * width
    fractions = []
    for x, y in centers:
        radius = math.hypot(x, y)
        if wall - radius < 5 * width or (inner_wall > 0 and radius - inner_wall < 5 * width):
            raise ValueError("injectors must lie at least 5 widths from every wall")
        fractions.append(1.0 - math.exp(-wall ** 2 / sigma2) if radius == 0.0 else 1.0)
    rate, primitive = sin2_window(start, duration)
    total = sum(q * f for q, f in zip(charges, fractions))

    def blobs(x, y, t):
        xp, x_arr, y_arr = _xy_arrays(x, y)
        cos_t, sin_t = math.cos(omega * t), math.sin(omega * t)
        result = 0.0 * x_arr
        for q, (cx, cy) in zip(charges, centers):
            px, py = cos_t * cx - sin_t * cy, sin_t * cx + cos_t * cy
            result = result + q / (math.pi * sigma2) * xp.exp(-((x_arr - px) ** 2 + (y_arr - py) ** 2) / sigma2)
        return result

    def source(x, y, t):
        weight = rate(t)
        return weight * blobs(x, y, t) if weight else 0.0 * _xy_arrays(x, y)[1]

    fields = dict(
        density_source=source,
        density_source_mass=lambda t: total * primitive(t),
        density_source_window=(float(start), float(start) + float(duration)),
        record=dict(charges=list(charges), centers=[list(c) for c in centers], width=width,
                    start=float(start), duration=float(duration), angular_velocity=omega, total_charge=total),
    )
    if omega == 0.0:
        fields.update(density_source_rate=rate, density_source_profile=lambda x, y: blobs(x, y, 0.0))
    return fields


def smooth_switch(start: float, duration: float):
    """Switch-on r(t) = sin^2(pi (t-t0)/(2T)) on [t0, t0+T], 0 before and 1 after, and its primitive R(t)."""
    start, duration = float(start), float(duration)
    if duration < 0:
        raise ValueError("switch-on duration must be nonnegative")

    def rate(t):
        tau = float(t) - start
        if tau <= 0.0:
            return 0.0
        if tau >= duration:
            return 1.0
        return math.sin(0.5 * math.pi * tau / duration) ** 2

    def primitive(t):
        tau = max(float(t) - start, 0.0)
        if tau >= duration:
            return tau - 0.5 * duration
        return 0.5 * tau - duration / (2.0 * math.pi) * math.sin(math.pi * tau / duration)

    return rate, primitive


def pulse_train(period: float, duty: float = 0.5, ramp: float | None = None):
    """Periodic smooth pulses r(t) in [0, 1], e.g. a pulsed emission rate.

    Each period starts with a pulse of length ``duty * period`` that rises and falls as sin^2 over
    ``ramp`` (default a tenth of the pulse) and is 0 for the rest of the period.
    """
    period, duty = float(period), float(duty)
    if not (period > 0.0 and 0.0 < duty <= 1.0):
        raise ValueError("pulse_train needs period > 0 and 0 < duty <= 1")
    on = duty * period
    ramp = 0.1 * on if ramp is None else float(ramp)
    if not 0.0 <= 2.0 * ramp <= on:
        raise ValueError("pulse_train needs 0 <= 2 * ramp <= duty * period")

    def rate(t):
        tau = math.fmod(float(t), period)
        if tau < 0.0:
            tau += period
        if tau >= on:
            return 0.0
        if ramp > 0.0 and tau < ramp:
            return math.sin(0.5 * math.pi * tau / ramp) ** 2
        if ramp > 0.0 and tau > on - ramp:
            return math.sin(0.5 * math.pi * (on - tau) / ramp) ** 2
        return 1.0

    return rate


def gaussian_ionization(*, rate, center=(0.0, 0.0), width=None, start=0.0, ramp=0.0) -> dict[str, Any]:
    """Electron-impact ionization S = nu(x) rho of a prescribed neutral gas, as GuidingCenterCase fields.

    nu(x, t) = r(t) * rate * exp(-|x - c|^2/width^2), with a uniform gas when
    ``width`` is None and the switch-on r(t) of :func:`smooth_switch` (at
    ``start`` over ``ramp``). The source acts only where electrons already are,
    so it amplifies the density it meets. A uniform gas present from t = 0 is
    the constant reaction ``density_reaction``, for which rho(x, t) =
    exp(nu t) r(x, (exp(nu t) - 1)/nu) maps every forced solution onto the
    unforced one r. Returns {} without a rate; otherwise the forcing fields, a
    JSON ``record`` and ``exponent(x, y, t)`` = int_0^t nu, so that an
    axisymmetric equilibrium rho0 evolves exactly as rho0 exp(exponent).
    """
    rate = float(rate)
    if rate == 0.0:
        return {}
    cx, cy = (float(c) for c in center)
    uniform = width is None
    if not uniform and not float(width) > 0:
        raise ValueError("the ionization width must be positive")
    switch, primitive = smooth_switch(start, ramp)
    record = dict(rate=rate, center=[cx, cy], width=None if uniform else float(width),
                  start=float(start), ramp=float(ramp))

    def profile(x, y):
        xp, x_arr, y_arr = _xy_arrays(x, y)
        if uniform:
            return rate + 0.0 * x_arr
        return rate * xp.exp(-((x_arr - cx) ** 2 + (y_arr - cy) ** 2) / float(width) ** 2)

    def exponent(x, y, t):
        return profile(x, y) * primitive(t)

    if uniform and float(start) <= 0.0 and float(ramp) == 0.0:
        return dict(density_reaction=rate, record=record, exponent=exponent)
    return dict(density_reaction_profile=profile, density_reaction_rate=switch, record=record, exponent=exponent)


def diocotron_annulus(
        *,
        k: int = 3,
        modes: tuple[int, ...] | list[int] | None = None,
        phase_scale: float = 0.0,
        epsilon: float = 1.0e-6,
        eps: float | None = None,
        inner_wall: float = 0.1,
        wall: float = 1.0,
        s_minus: float = 0.4,
        s_plus: float = 0.5,
        rho_bar: float = 1.0,
        edge_width: float = 0.0,
        s_bar: float | None = None,
        s_d: float | None = None,
        p: float | None = None,
        theta_shift: float = 0.0,
        truncate: bool = False,
        conforming: bool = True,
        band_mesh_size: float | None = None,
        band_grading: float = 0.25,
        refine_band: tuple[float, float] | None = None,
        injection_charges: tuple[float, ...] | list[float] = (),
        injection_centers: tuple[tuple[float, float], ...] | list | None = None,
        injection_width: float = 0.06,
        injection_start: float = 0.0,
        injection_duration: float = 4.0,
        injection_angular_velocity: float = 0.0,
        ionization_rate: float = 0.0,
        ionization_center: tuple[float, float] = (0.0, 0.0),
        ionization_width: float | None = None,
        ionization_start: float = 0.0,
        ionization_ramp: float = 0.0,
) -> GuidingCenterCase:
    """Diocotron band between an inner wall and an outer wall, phi = 0 on both.

    This is the semi-Lagrangian benchmark geometry of Madaule et al. (2013) and
    Crouseilles et al. (2014) ("Dirichlet for all modes"), scaled to an outer
    wall of radius ``wall`` (lengths enter only through ratios; times are
    unchanged). ``inner_wall=0`` gives the disk. Profiles are those of
    :func:`diocotron_k`. A sharp band (``p=None``, ``edge_width=0``) with
    ``conforming=True`` is meshed with embedded circles at ``s_minus`` and
    ``s_plus`` and projected elementwise, so the initial step has no Gibbs
    oscillations; so is a truncated super-Gaussian (``p`` with ``truncate=True``),
    whose cutoff jumps are then exact element edges. ``band_mesh_size`` refines
    the mesh on ``refine_band`` (default ``(s_minus, s_plus)``); the run's mesh
    size applies far from it (see :func:`hybridge.core.mesh.gmsh_annulus_mesh`).

    ``injection_charges`` adds an explicit density source (independent of rho):
    normalized Gaussians of width ``injection_width`` at ``injection_centers``
    (default: the centre), each injecting its charge with the unit-integral
    window (2/T) sin^2(pi (t-t0)/T) on [t0, t0+T] = ``injection_start`` +
    [0, ``injection_duration``], rotating about the centre at
    ``injection_angular_velocity`` (see :func:`gaussian_injection`).

    ``ionization_rate`` adds the ionization S = nu(x) rho of a neutral gas
    cloud of that peak rate (Gaussian of ``ionization_width`` at
    ``ionization_center``; uniform without a width), switched on at
    ``ionization_start`` over ``ionization_ramp`` (see
    :func:`gaussian_ionization`). Without a perturbation (``epsilon=0``), no
    injection and an axisymmetric gas, rho0(r) exp(int_0^t nu) is the exact
    density.
    """
    inner_wall, wall = float(inner_wall), float(wall)
    if not 0.0 <= inner_wall < float(s_minus) < float(s_plus) < wall:
        raise ValueError("require 0 <= inner_wall < s_minus < s_plus < wall")
    base = diocotron_k(
        k=k, modes=modes, phase_scale=phase_scale, epsilon=epsilon, eps=eps,
        s_minus=s_minus, s_plus=s_plus, rho_bar=rho_bar, edge_width=edge_width,
        s_bar=s_bar, s_d=s_d, p=p, theta_shift=theta_shift, truncate=truncate,
    )
    parameters = dict(base.parameters)
    step = parameters["p"] is None and parameters["edge_width"] == 0.0
    cut_profile = parameters["p"] is not None and parameters["truncate"]
    band = (parameters["s_minus"], parameters["s_plus"]) if ((step or cut_profile) and conforming) else None
    active_modes = tuple(parameters["modes"])
    amplitude, phase = parameters["epsilon"], parameters["phase_scale"]
    shift, level = parameters["theta_shift"], parameters["rho_bar"]

    def radial_profile(x_arr, y_arr):
        if not cut_profile:
            return level + 0.0 * x_arr
        # The untruncated super-Gaussian; the band indicator applies the cutoff.
        return rho_eq_super_gaussian_annulus(
            x_arr, y_arr, s_bar=parameters["s_bar"], s_d=parameters["s_d"], p=parameters["p"],
            rho_bar=level, truncate=False)

    def band_modulation(x, y, perturbed=True):
        xp, x_arr, y_arr = _xy_arrays(x, y)
        profile = radial_profile(x_arr, y_arr)
        if not perturbed:
            return profile
        theta = xp.arctan2(y_arr, x_arr) - shift
        modulation = 1.0 + 0.0 * x_arr
        for azimuthal_mode in active_modes:
            modulation = modulation + amplitude * xp.cos(azimuthal_mode * theta + phase * azimuthal_mode ** 2)
        return profile * modulation

    parameters.update(
        inner_wall=inner_wall,
        wall=wall,
        conforming=bool(band is not None),
        geometry={
            "inner_radius": inner_wall,
            "outer_radius": wall,
            "interface_radii": band or (),
            "band": ((tuple(float(r) for r in refine_band) if refine_band is not None
                      else (parameters["s_minus"], parameters["s_plus"])) if band_mesh_size is not None else None),
            "band_mesh_size": None if band_mesh_size is None else float(band_mesh_size),
            "band_grading": float(band_grading),
        },
    )
    wall_text = f"inner wall r={inner_wall:g}, " if inner_wall > 0 else ""
    forcing = gaussian_injection(
        charges=injection_charges, centers=injection_centers, width=injection_width,
        start=injection_start, duration=injection_duration, angular_velocity=injection_angular_velocity,
        inner_wall=inner_wall, wall=wall)
    if forcing:
        parameters["injection"] = forcing.pop("record")
    ionization = gaussian_ionization(
        rate=ionization_rate, center=ionization_center, width=ionization_width,
        start=ionization_start, ramp=ionization_ramp)
    exact = {}
    if ionization:
        parameters["ionization"] = ionization.pop("record")
        exponent = ionization.pop("exponent")
        axisymmetric = ionization_width is None or tuple(float(c) for c in ionization_center) == (0.0, 0.0)
        if amplitude == 0.0 and not forcing and axisymmetric:
            equilibrium = base.equilibrium_density

            def exact_density(x, y, t):
                xp, x_arr, y_arr = _xy_arrays(x, y)
                return equilibrium(x_arr, y_arr) * xp.exp(exponent(x_arr, y_arr, t))

            exact = dict(exact_density=exact_density)
    return replace(
        base,
        **forcing,
        **ionization,
        **exact,
        key="diocotron_annulus",
        description=(f"Diocotron band {parameters['s_minus']:g}<r<{parameters['s_plus']:g}, {wall_text}"
                     f"outer wall r={wall:g}, phi=0 on every wall, modes {active_modes}."),
        default_domain="annulus",
        parameters=parameters,
        conforming_band=band,
        band_modulation=band_modulation if band is not None else None,
    )


def magnetron(
        *,
        cathode: float = 0.5,
        anode: float = 1.0,
        voltage: float = 0.1,
        hub_top: float = 0.65,
        hub_density: float = 1.0,
        hub_edge: float = 0.01,
        emission_rate: float = 1.0,
        emission_top: float | None = None,
        initial_hub: bool = True,
        wave_mode: int = 6,
        wave_amplitude: float = 0.025,
        wave_speed: float = 0.29,
        wave_start: float = 0.0,
        wave_ramp: float = 20.0,
        band_mesh_size: float | None = None,
        refine_band: tuple[float, float] | None = None,
        band_grading: float = 0.25,
        anode_geometry: str = "circle",
        back_radius: float = 1.35,
        vane_fraction: float = 0.5,
        corner_mesh_size: float | None = None,
        corner_grading: float = 0.3,
        vane_width: float | None = None,
        tip_radius: float | None = None,
        cathode_mesh_size: float | None = None,
        cathode_grading: float = 0.3,
        curvature_points: int | None = None,
        anode_mesh_size: float | None = None,
        anode_grading: float = 0.3,
        emission_profile: str = "cos2",
        hall_parameter: float | None = None,
        electron_temperature: float = 0.0,
        emission_period: float | None = None,
        emission_duty: float = 0.5,
        emission_ramp: float | None = None,
        emission_phase: float = 0.0,
) -> GuidingCenterCase:
    r"""Crossed-field (magnetron) diode in the guiding-center limit: emission, spokes, anode current.

    Electrons of density rho >= 0 drift between a cathode r = ``cathode`` and an
    anode r = ``anode``. With -lap phi = rho (phi is minus the physical potential
    in model units), the drift u = (d_y phi, -d_x phi) is E x B for B along +z:
    u_theta = -d_r phi and u_r = (1/r) d_theta phi. The cathode is at phi = 0 and
    the anode at

        phi = -V + V1 s(t) cos(N (theta - W t)),

    a DC voltage ``voltage`` (anode positive in physical units, so the vacuum drift
    co-rotates with the electron space charge) and a wave of ``wave_mode`` = N
    periods turning at the angular velocity ``wave_speed`` = W: the RF field of
    the anode cavities, prescribed, with amplitude ``wave_amplitude`` = V1
    switched on by :func:`smooth_switch` over ``wave_ramp``.

    Space-charge-limited emission keeps a Brillouin-like hub of density
    ``hub_density`` = rho_c on the cathode,

        S + alpha rho = kappa chi(r) (rho_c - rho),  chi = (1 - tanh((r - r_h)/delta))/2,

    a source kappa rho_c chi and a reaction -kappa chi rho (``emission_rate`` =
    kappa, ``hub_top`` = r_h, ``hub_edge`` = delta). ``emission_top`` = r_e < r_h
    confines the emission to a cathode layer: chi = 1 below r_e - 2 delta, a
    cos^2 taper and exactly 0 from r_e on (a tanh tail would keep filling the
    hub edge); the hub above it is then fed only by the non-axisymmetric drift. The
    hub starts full (``initial_hub``, density rho_c chi(r) with r_h) or empty.

    The anode wave makes the wall non-equipotential: u.n = -(N V1 s/r) sin(N(theta - W t))
    on the anode, so the drift crosses it, electrons leave where u.n > 0 (the
    anode current) and empty space enters elsewhere (inflow data rho = 0). Near
    synchronism (W close to the edge-wave speed of the hub) the wave draws the
    hub into N spokes that carry the current. The wall potential enters through
    ``wall_potentials`` (discrete harmonic fields; Poisson keeps zero Dirichlet data).

    ``anode_geometry="vanes"``: a vane-type anode block (domain ``vane-anode``,
    :func:`hybridge.core.mesh.gmsh_vane_anode_mesh`) with 2N vanes, tips at
    r = ``anode``, slots back to ``back_radius``, each vane covering
    ``vane_fraction`` of the pitch. The drive is the pi mode of the vane
    resonators: vane k at -V + (-1)^k V1 s(t) cos(N W t), a standing wave whose
    co-rotating half turns at W; the back wall of each slot carries the cavity
    voltage, linear in theta between its two vanes, so the vanes are
    equipotential (u.n = 0) and the electrons leave through the slot back walls.
    The vane tips are re-entrant 270-degree corners, where the field is
    singular (~ r^(-1/3)); ``corner_mesh_size`` grades the mesh there.

    ``vane_width``: parallel-sided vanes of that width instead of radial ones
    (``vane_fraction`` unused); the pi-mode pattern puts a boundary point on vane k
    when it lies within w/2 of the vane axis, and ramps linearly in theta across
    each slot back wall between the back corners.

    ``tip_radius`` (with ``vane_width``): round the two tip corners of every vane by a fillet of that
    radius (:func:`hybridge.core.mesh.vane_tip_fillets`), so the field has no corner singularity.

    ``cathode_mesh_size``/``cathode_grading`` (vanes): grade the mesh towards the cathode, where a
    pulsed cathode condition opens a diffusive layer of width ~sqrt(D t) at each switch.

    ``curvature_points`` (vanes): bound the mesh size on the curved walls by 2 pi R / points, so the
    affine chords stay close to the circles (geometric error ~ h^2 / 8R).

    ``emission_profile`` (with ``emission_top``): the taper of the emission layer between
    r_e - 2 delta and r_e, ``"cos2"`` (default, C^1) or ``"smooth"`` (C^infinity, the compact step
    f(1-s)/(f(s)+f(1-s)) with f(a) = exp(-1/a)); both are 1 below and exactly 0 above the taper.

    ``hall_parameter`` = h (with ``electron_temperature`` = theta > 0): electrons
    colliding elastically with a background gas, below the ionization threshold, so
    no ions are created. They drift with u_E + q/h (q = -grad phi points towards the
    anode) and diffuse with D = theta/h (Einstein relation). The density then takes
    an advection-diffusion-reaction solve with absorbing (rho = 0) walls
    (``scripts.guiding_center.runtime.density_adr``); h -> infinity is the
    collisionless diode above.

    ``emission_period``: pulsed emission. The source and the reaction of the
    emission layer follow :func:`pulse_train` (``emission_duty``, ``emission_ramp``).
    ``emission_phase``: the train is evaluated at t + phase, e.g. half a pulse so that the run starts
    with the emission on and the cathode data rho_c P(0) = rho_c match the initial hub.
    """
    cathode, anode, V = float(cathode), float(anode), float(voltage)
    width = None if vane_width is None else float(vane_width)
    if width is not None and anode_geometry != "vanes":
        raise ValueError("vane_width needs anode_geometry='vanes'")
    r_h, rho_c, delta, kappa = float(hub_top), float(hub_density), float(hub_edge), float(emission_rate)
    N, V1, W = int(wave_mode), float(wave_amplitude), float(wave_speed)
    if not 0.0 < cathode < r_h < anode:
        raise ValueError("require 0 < cathode < hub_top < anode")
    if delta <= 0.0 or rho_c < 0.0 or kappa < 0.0 or N < 0:
        raise ValueError("hub_edge must be positive; hub_density, emission_rate and wave_mode nonnegative")
    middle_squared = (0.5 * (cathode + anode)) ** 2

    r_e = r_h if emission_top is None else float(emission_top)
    if not cathode < r_e <= r_h:
        raise ValueError("require cathode < emission_top <= hub_top")

    def layer(top):
        def profile(x, y):
            xp, x_arr, y_arr = _xy_arrays(x, y)
            return 0.5 * (1.0 - xp.tanh((xp.sqrt(x_arr * x_arr + y_arr * y_arr) - top) / delta))
        return profile

    if emission_profile not in {"cos2", "smooth"}:
        raise ValueError("emission_profile must be 'cos2' or 'smooth'")

    def compact_layer(x, y):  # 1 below r_e - 2 delta, a C^1 or C^infinity taper, exactly 0 from r_e on
        xp, x_arr, y_arr = _xy_arrays(x, y)
        s = xp.clip((xp.sqrt(x_arr * x_arr + y_arr * y_arr) - (r_e - 2.0 * delta)) / (2.0 * delta), 0.0, 1.0)
        if emission_profile == "cos2":
            return xp.cos(0.5 * math.pi * s) ** 2

        def bump(a):
            return xp.where(a > 0.0, xp.exp(-1.0 / xp.maximum(a, 1.0e-300)), 0.0)
        rising, falling = bump(s), bump(1.0 - s)
        return falling / (rising + falling)

    hub = layer(r_h)
    emitter = hub if emission_top is None else compact_layer

    def anode_pattern(function):
        def pattern(x, y):
            xp, x_arr, y_arr = _xy_arrays(x, y)
            on_anode = x_arr * x_arr + y_arr * y_arr > middle_squared
            return xp.where(on_anode, function(xp, xp.arctan2(y_arr, x_arr)), 0.0 * x_arr)
        return pattern

    switch, _ = smooth_switch(wave_start, wave_ramp)
    walls = [(anode_pattern(lambda xp, theta: -V + 0.0 * theta), lambda t: 1.0)]
    vanes = anode_geometry == "vanes"
    if anode_geometry not in {"circle", "vanes"}:
        raise ValueError("anode_geometry must be 'circle' or 'vanes'")
    if vanes:
        if N < 1 or not anode < float(back_radius) or not 0.0 < float(vane_fraction) < 1.0:
            raise ValueError("a vane anode needs wave_mode >= 1, back_radius > anode and 0 < vane_fraction < 1")
        pitch, half = 2.0 * math.pi / (2 * N), math.pi * float(vane_fraction) / (2 * N)
        if width is not None:
            if not (0.0 < width < 2.0 * anode and 2.0 * math.asin(width / (2.0 * anode)) < pitch):
                raise ValueError("vane_width must be positive and leave a gap between neighbouring vane tips")
            half_back = math.asin(width / (2.0 * float(back_radius)))

        def pi_mode(xp, theta):  # (-1)^k on vane k; linear across each slot back wall
            index = xp.rint(theta / pitch)
            sign = 1.0 - 2.0 * xp.mod(index, 2.0)
            offset = xp.abs(theta - index * pitch)
            ramp = 1.0 - 2.0 * (offset - half) / (pitch - 2.0 * half)
            return sign * xp.where(offset <= half, 1.0, ramp)

        def parallel_pi_mode(x, y):  # parallel-sided vanes: on vane k within w/2 of its axis
            xp, x_arr, y_arr = _xy_arrays(x, y)
            radius_sq = x_arr * x_arr + y_arr * y_arr
            theta = xp.arctan2(y_arr, x_arr)
            index = xp.rint(theta / pitch)
            sign = 1.0 - 2.0 * xp.mod(index, 2.0)
            offset = xp.abs(theta - index * pitch)
            on_vane = xp.sqrt(radius_sq) * xp.sin(offset) <= 0.5 * width * (1.0 + 1e-9)
            ramp = 1.0 - 2.0 * (offset - half_back) / (pitch - 2.0 * half_back)
            return xp.where(radius_sq > middle_squared, sign * xp.where(on_vane, 1.0, ramp), 0.0 * x_arr)

        if V1 != 0.0:
            pattern = anode_pattern(pi_mode) if width is None else parallel_pi_mode
            walls.append((pattern, lambda t: V1 * switch(t) * math.cos(N * W * t)))
    elif N > 0 and V1 != 0.0:
        walls += [(anode_pattern(lambda xp, theta: xp.cos(N * theta)),
                   lambda t: V1 * switch(t) * math.cos(N * W * t)),
                  (anode_pattern(lambda xp, theta: xp.sin(N * theta)),
                   lambda t: V1 * switch(t) * math.sin(N * W * t))]
    emission = kappa * rho_c
    forcing = {}
    pulse = None if emission_period is None else pulse_train(emission_period, emission_duty, emission_ramp)
    if pulse is not None and float(emission_phase) != 0.0:
        unshifted, phase = pulse, float(emission_phase)

        def pulse(t):
            return unshifted(t + phase)
    elif float(emission_phase) != 0.0:
        raise ValueError("emission_phase needs emission_period")
    if kappa > 0.0 and pulse is None:
        forcing = dict(
            density_source=lambda x, y, t: emission * emitter(x, y),
            density_source_rate=lambda t: 1.0,
            density_source_profile=lambda x, y: emission * emitter(x, y),
            density_reaction_profile=lambda x, y: -kappa * emitter(x, y),
        )
    elif kappa > 0.0:  # pulsed emission: source and reaction follow the same pulse train
        forcing = dict(
            density_source=lambda x, y, t: emission * pulse(t) * emitter(x, y),
            density_source_rate=pulse,
            density_source_profile=lambda x, y: emission * emitter(x, y),
            density_reaction_profile=lambda x, y: -kappa * emitter(x, y),
            density_reaction_rate=pulse,
        )
    parameters = dict(
        cathode=cathode, anode=anode, voltage=V, hub_top=r_h, hub_density=rho_c, hub_edge=delta,
        emission_rate=kappa, emission_top=r_e, initial_hub=bool(initial_hub), wave_mode=N, wave_amplitude=V1, wave_speed=W,
        wave_start=float(wave_start), wave_ramp=float(wave_ramp),
        vacuum_cathode_charge=2.0 * math.pi * V / math.log(anode / cathode),
        anode_geometry=anode_geometry,
        geometry={
            "inner_radius": cathode,
            "outer_radius": float(back_radius) if vanes else anode,
            "interface_radii": (),
            "band": ((tuple(float(r) for r in refine_band) if refine_band is not None else (r_h - 0.05, anode))
                     if band_mesh_size is not None else None),
            "band_mesh_size": None if band_mesh_size is None else float(band_mesh_size),
            "band_grading": float(band_grading),
        },
    )
    if vanes:
        parameters.update(back_radius=float(back_radius), vanes=2 * N, vane_fraction=float(vane_fraction))
        parameters["geometry"].update(anode_radius=anode, back_radius=float(back_radius), vanes=2 * N,
                                      vane_fraction=float(vane_fraction),
                                      corner_mesh_size=None if corner_mesh_size is None else float(corner_mesh_size),
                                      corner_grading=float(corner_grading))
        if width is not None:
            parameters.update(vane_width=width)
            parameters["geometry"].update(vane_width=width)
        if cathode_mesh_size is not None:
            parameters["geometry"].update(cathode_mesh_size=float(cathode_mesh_size),
                                          cathode_grading=float(cathode_grading))
        if curvature_points:
            parameters["geometry"].update(curvature_points=int(curvature_points))
        if anode_mesh_size is not None:  # outflow layers ~D/|u.n| where the charge reaches the anode
            parameters["geometry"].update(anode_mesh_size=float(anode_mesh_size), anode_grading=float(anode_grading))
        if tip_radius is not None:
            if width is None:
                raise ValueError("tip_radius needs parallel-sided vanes (vane_width)")
            parameters.update(tip_radius=float(tip_radius))
            parameters["geometry"].update(tip_radius=float(tip_radius))
    elif tip_radius is not None:
        raise ValueError("tip_radius needs anode_geometry='vanes' with vane_width")
    if emission_profile != "cos2":
        if emission_top is None:
            raise ValueError("emission_profile needs emission_top")
        parameters.update(emission_profile=emission_profile)
    if hall_parameter is not None:
        hall, theta_e = float(hall_parameter), float(electron_temperature)
        if not (math.isfinite(hall) and hall > 0.0 and math.isfinite(theta_e) and theta_e > 0.0):
            raise ValueError("a collisional magnetron needs hall_parameter > 0 and electron_temperature > 0")
        parameters["collisional"] = {"hall_parameter": hall, "electron_temperature": theta_e,
                                     "diffusion": theta_e / hall}
        # All-Dirichlet density data of the ADR solve: the anode absorbs (0); the cathode emits at the
        # hub density rho_c (times the emission pulse), consistent with the emission layer next to it.
        emitting = pulse if pulse is not None else (lambda t: 1.0)

        def collisional_density_boundary(x, y, t):
            xp, x_arr, y_arr = _xy_arrays(x, y)
            on_cathode = x_arr * x_arr + y_arr * y_arr < middle_squared
            return xp.where(on_cathode, rho_c * emitting(t), 0.0 * x_arr)
    if pulse is not None:
        parameters.update(emission_period=float(emission_period), emission_duty=float(emission_duty),
                          emission_ramp=None if emission_ramp is None else float(emission_ramp))
        if float(emission_phase) != 0.0:
            parameters.update(emission_phase=float(emission_phase))
    return GuidingCenterCase(
        key="magnetron",
        description=(f"Magnetron diode {cathode:g}<r<{anode:g}: hub rho_c={rho_c:g} to r={r_h:g}, V={V:g}, "
                     f"anode wave N={N}, V1={V1:g}, W={W:g}; emission rate {kappa:g}."),
        initial_density=(lambda x, y: rho_c * hub(x, y)) if initial_hub else _zero_like_xy,
        potential_boundary=lambda x, y, t: _zero_like_xy(x, y),
        potential_boundary_constant=0.0,
        density_boundary=(collisional_density_boundary if hall_parameter is not None
                          else lambda x, y, t: _zero_like_xy(x, y)),
        density_transport_boundary_mode="eliminate",
        default_domain="vane-anode" if vanes else "annulus",
        parameters=parameters,
        wall_potentials=tuple(walls),
        wall_flux_diagnostics=True,
        **forcing,
    )


def euler_vortex_gas(
        *,
        counts: tuple[int, ...] | list[int] = (192, 96, 48, 24),
        sigmas: tuple[float, ...] | list[float] = (0.008, 0.016, 0.032, 0.064),
        amplitude: float = 4.0,
        seed: int = 17,
        center_radius: float = 0.96,
) -> GuidingCenterCase:
    r"""Signed Gaussian vortex gas for unforced Euler flow on the unit disk.

    ``rho(x) = sum(a_j * exp(-|x-c_j|**2/(2*sigma_j**2)))``.
    Centers are sampled uniformly in disk area, independently at each scale.
    Each scale has equal numbers of positive and negative blobs, with random
    magnitudes in ``[0.8, 1.2]*amplitude``. The negative amplitudes are rescaled
    per scale to cancel the *disk* circulation, accounting for Gaussian tails
    outside the wall. No positivity constraint or radial envelope is applied.
    ``phi=0`` on the wall supplies the impermeable Euler boundary; vorticity
    itself need not vanish there. The seed fixes the same field on every mesh.
    """
    from scipy.stats import ncx2

    counts, sigmas = tuple(counts), tuple(sigmas)
    if not counts or len(counts) != len(sigmas):
        raise ValueError("counts and sigmas must have the same nonzero length")
    if any(not np.isfinite(n) or int(n) != n or n < 2 or n % 2 for n in counts):
        raise ValueError("counts must contain positive even integers")
    counts = tuple(int(n) for n in counts)
    sigmas = tuple(float(sigma) for sigma in sigmas)
    if any(not np.isfinite(sigma) or sigma <= 0 for sigma in sigmas):
        raise ValueError("sigmas must be finite and positive")
    level, radius = float(amplitude), float(center_radius)
    if not np.isfinite(level) or level <= 0:
        raise ValueError("amplitude must be finite and positive")
    if not np.isfinite(radius) or not 0 < radius < 1:
        raise ValueError("center_radius must be between zero and one")
    if not np.isfinite(seed) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    blobs = []
    for count, width in zip(counts, sigmas):
        r = radius * np.sqrt(rng.random(count))
        theta = rng.uniform(0.0, 2.0*np.pi, count)
        strengths = level * rng.uniform(0.8, 1.2, count)
        negative = rng.permutation(count)[:count//2]
        strengths[negative] *= -1.0
        # Integral of a normalized displaced 2D Gaussian over the unit disk.
        disk_fraction = ncx2.cdf(1.0/width**2, df=2, nc=(r/width)**2)
        positive = strengths > 0
        strengths[negative] *= -float(np.dot(strengths[positive], disk_fraction[positive])) / float(
            np.dot(strengths[negative], disk_fraction[negative])
        )
        blobs.extend(zip(r*np.cos(theta), r*np.sin(theta), np.full(count, width), strengths))
    # Python scalars preserve the configured precision on NumPy/CuPy inputs.
    blobs = tuple(tuple(float(value) for value in blob) for blob in blobs)

    def initial_density(x, y):
        xp, x_arr, y_arr = _xy_arrays(x, y)
        x_arr, y_arr = xp.broadcast_arrays(x_arr, y_arr)
        density = xp.zeros_like(x_arr)
        for cx, cy, width, strength in blobs:
            density += strength * xp.exp(
                -((x_arr-cx)**2 + (y_arr-cy)**2) / (2.0*width**2)
            )
        return density

    return GuidingCenterCase(
        key="euler_vortex_gas",
        description="Domain-filling signed Gaussian vortices at multiple scales for decaying 2D Euler turbulence.",
        initial_density=initial_density,
        potential_boundary=lambda x, y, t: _zero_like_xy(x, y),
        density_boundary=None,
        density_transport_boundary_mode="zero-flux",
        default_domain="disc",
        potential_boundary_constant=0.0,
        density_is_vorticity=True,
        parameters={
            "counts": counts, "sigmas": sigmas, "amplitude": level,
            "seed": int(seed), "center_radius": radius,
        },
    )


def positive_turbulence(
        *,
        counts: tuple[int, ...] | list[int] = (192, 96, 48, 24),
        sigmas: tuple[float, ...] | list[float] = (0.008, 0.016, 0.032, 0.064),
        amplitude: float = 4.0,
        seed: int = 17,
        cutoff: float = 8.0,
        wall_gap: float = 0.04,
        geometry: str = "disc",
        geometry_params: dict | None = None,
        fft_grid_shape: tuple[int, int] | list[int] | None = None,
) -> GuidingCenterCase:
    r"""Nonnegative multiscale guiding-center density / Euler vorticity.

    All Gaussian strengths are positive. Each blob is truncated at ``cutoff``
    standard deviations and sampled with its entire support at least
    ``wall_gap`` inside the selected wall. On the default unit disk, ``rho_0``
    is therefore identically zero in the annulus
    ``1-wall_gap < sqrt(x**2+y**2) <= 1``. The analytic initial density is
    nonnegative everywhere and has no vortex cores or tails at the wall. Up to
    the sign convention in the Poisson equation, the same scalar is one-signed
    Euler vorticity. The seed fixes the same field on every mesh.

    ``fft_grid_shape=(nx, ny)`` selects an approximate FFT-convolved field
    with smooth nonnegative grid reconstruction. Additional center clearance
    preserves the requested zero-density wall band after grid spreading.
    """
    from hybridge.core.geometry import DiskDomain, shaped_domain
    from hybridge.cases.profiles import sample_gaussian_blob_field

    counts, sigmas = tuple(counts), tuple(sigmas)
    geometry_params = dict(geometry_params or {})
    if geometry == "disc":
        if geometry_params:
            raise ValueError("geometry_params are only supported for shaped domains")
        domain = DiskDomain()
        domain_label = "unit disk"
    else:
        domain = shaped_domain(geometry, **geometry_params)
        domain_label = "ITER" if geometry == "iter" else geometry
    density = sample_gaussian_blob_field(
        domain, counts, sigmas,
        amplitude=amplitude,
        seed=seed,
        cutoff=cutoff,
        wall_clearance=wall_gap,
        strength_mode="positive",
        fft_grid_shape=fft_grid_shape,
    )
    return GuidingCenterCase(
        key="positive_turbulence",
        description=(
            "Nonnegative multiscale Gaussian guiding-center density (equivalently, "
            f"one-signed Euler vorticity) in the {domain_label}, with compact "
            "support away from the wall."
            + (" Uses FFT convolution and cubic B-spline reconstruction."
               if fft_grid_shape is not None else "")
        ),
        initial_density=density,
        potential_boundary=lambda x, y, t: _zero_like_xy(x, y),
        density_boundary=None,
        density_transport_boundary_mode="zero-flux",
        default_domain=geometry,
        potential_boundary_constant=0.0,
        parameters={
            "counts": tuple(int(count) for count in counts),
            "sigmas": tuple(float(sigma) for sigma in sigmas),
            "amplitude": float(amplitude),
            "seed": int(seed),
            "cutoff": float(cutoff),
            "wall_gap": float(wall_gap),
            "geometry": geometry_params,
            "geometry_name": geometry,
            **({"initial_profile": "fft_gaussian",
                "fft_grid_shape": density.grid_shape,
                "fft_grid_spacing": tuple(float(value) for value in density.spacing),
                "fft_support_padding": density.support_padding}
               if fft_grid_shape is not None else {}),
        },
    )


def euler_shaped_vortex_gas(
        *, geometry: str = "horseshoe", geometry_params: dict | None = None,
        counts: tuple[int, ...] | list[int] = (192, 96, 48, 24),
        sigmas: tuple[float, ...] | list[float] = (0.004, 0.008, 0.016, 0.032),
        amplitude: float = 4.0, seed: int = 17,
) -> GuidingCenterCase:
    """Signed multiscale gas in a horseshoe, supplied ITER wall, or Pac-Man domain."""
    from hybridge.core.geometry import shaped_domain
    from hybridge.cases.profiles import sample_gaussian_blob_field

    geometry_params = dict(geometry_params or {})
    domain = shaped_domain(geometry, **geometry_params)
    density = sample_gaussian_blob_field(domain, counts, sigmas, amplitude=amplitude, seed=seed)
    return GuidingCenterCase(
        key="euler_shaped_vortex_gas",
        description=f"Multiscale signed Euler vortex gas in the {geometry} domain.",
        initial_density=density,
        potential_boundary=lambda x, y, t: _zero_like_xy(x, y),
        density_boundary=None, density_transport_boundary_mode="zero-flux",
        default_domain=geometry, potential_boundary_constant=0.0, density_is_vorticity=True,
        parameters={
            "counts": tuple(counts), "sigmas": tuple(sigmas), "amplitude": float(amplitude),
            "seed": int(seed), "geometry": geometry_params, "gaussian_cutoff": density.cutoff,
        },
    )


def euler_star_vortex_gas(
        *,
        counts: tuple[int, ...] | list[int] = (192, 96, 48, 24),
        sigmas: tuple[float, ...] | list[float] = (0.008, 0.016, 0.032, 0.064),
        amplitude: float = 4.0,
        seed: int = 17,
        star_radius: float = 1.0,
        star_amplitude: float = 0.35,
        star_mode: int = 5,
        hole_radius: float = 0.30,
        boundary_points: int = 500,
) -> GuidingCenterCase:
    """Multiscale signed vortex gas in a sampled star around a circular island.

    The outer radius is ``star_radius + star_amplitude*cos(star_mode*theta)``.
    Each positive Gaussian has an equal negative partner at a random nonzero
    symmetry rotation of the domain. Their integrals cancel on this domain,
    including wall-truncated tails, before mesh/projection error. The field
    itself is not forced to have rotational symmetry. Centers lie at least
    two core widths from both walls; the Gaussian tails are not truncated.

    Potential is fixed to zero on both walls, with zero density boundary flux.
    No independent circulation around the island is prescribed.
    """
    counts, sigmas = tuple(counts), tuple(sigmas)
    if not counts or len(counts) != len(sigmas):
        raise ValueError("counts and sigmas must have the same nonzero length")
    if any(not np.isfinite(n) or int(n) != n or n < 2 or n % 2 for n in counts):
        raise ValueError("counts must contain positive even integers")
    counts = tuple(int(n) for n in counts)
    sigmas = tuple(float(sigma) for sigma in sigmas)
    if any(not np.isfinite(sigma) or sigma <= 0.0 for sigma in sigmas):
        raise ValueError("sigmas must be finite and positive")
    level, radius, modulation, hole = map(float, (amplitude, star_radius, star_amplitude, hole_radius))
    if not all(np.isfinite(v) for v in (level, radius, modulation, hole)):
        raise ValueError("amplitude and geometry parameters must be finite")
    if level <= 0.0 or radius <= abs(modulation):
        raise ValueError("amplitude must be positive and star_radius must exceed abs(star_amplitude)")
    if not np.isfinite(star_mode) or int(star_mode) != star_mode or star_mode < 2:
        raise ValueError("star_mode must be an integer of at least two")
    mode = int(star_mode)
    if (not np.isfinite(boundary_points) or int(boundary_points) != boundary_points
            or boundary_points < max(8, 4 * mode) or boundary_points % mode):
        raise ValueError("boundary_points must be a sufficiently large multiple of star_mode")
    nboundary = int(boundary_points)
    if not 0.0 < hole < (radius - abs(modulation)) * np.cos(np.pi / nboundary):
        raise ValueError("hole_radius must be positive and strictly inside the sampled star")
    if not np.isfinite(seed) or int(seed) != seed or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if 4.0 * max(sigmas) >= radius + abs(modulation) - hole:
        raise ValueError("sigmas are too large to place cores two widths from both walls")

    angles = 2.0 * np.pi * np.arange(nboundary) / nboundary
    radii = radius + modulation * np.cos(mode * angles)
    vertices = radii[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    edges = np.roll(vertices, -1, axis=0) - vertices
    edge_sq = np.sum(edges * edges, axis=1)
    rng = np.random.default_rng(int(seed))
    blobs = []
    for count, width in zip(counts, sigmas):
        pairs = 0
        # Rejection from a disk gives uniform area sampling in the fluid's
        # eligible interior, including the lobes. Bound attempts for bad inputs.
        for _ in range(max(10000, 100 * count)):
            theta = rng.uniform(0.0, 2.0 * np.pi)
            r = (radius + abs(modulation)) * np.sqrt(rng.random())
            if r <= hole + 2.0 * width:
                continue
            direction = np.array([np.cos(theta), np.sin(theta)])
            sector = min(int(theta * nboundary / (2.0 * np.pi)), nboundary - 1)
            a, e = vertices[sector], edges[sector]
            wall_r = (a[0] * e[1] - a[1] * e[0]) / (direction[0] * e[1] - direction[1] * e[0])
            if r >= wall_r:
                continue
            center = r * direction
            offset = center - vertices
            fraction = np.clip(np.sum(offset * edges, axis=1) / edge_sq, 0.0, 1.0)
            distance_sq = np.sum((offset - fraction[:, None] * edges)**2, axis=1)
            if np.min(distance_sq) <= (2.0 * width)**2:
                continue
            partner_angle = theta + 2.0 * np.pi * rng.integers(1, mode) / mode
            strength = float(level * rng.uniform(0.8, 1.2))
            blobs.append((float(center[0]), float(center[1]), width, strength))
            blobs.append((float(r * np.cos(partner_angle)), float(r * np.sin(partner_angle)), width, -strength))
            pairs += 1
            if 2 * pairs == count:
                break
        else:
            raise ValueError("could not place vortex cores with two-width wall clearance; reduce sigmas")
    blobs = tuple(blobs)

    def initial_density(x, y):
        xp, x_arr, y_arr = _xy_arrays(x, y)
        x_arr, y_arr = xp.broadcast_arrays(x_arr, y_arr)
        density = xp.zeros_like(x_arr)
        for cx, cy, width, strength in blobs:
            density += strength * xp.exp(-((x_arr-cx)**2 + (y_arr-cy)**2) / (2.0*width**2))
        return density

    return GuidingCenterCase(
        key="euler_star_vortex_gas",
        description="Multiscale signed vortex gas in a nonconvex star with a circular island.",
        initial_density=initial_density,
        potential_boundary=lambda x, y, t: _zero_like_xy(x, y),
        density_boundary=None,
        density_transport_boundary_mode="zero-flux",
        default_domain="smooth-star",
        potential_boundary_constant=0.0,
        density_is_vorticity=True,
        parameters={
            "counts": counts, "sigmas": sigmas, "amplitude": level, "seed": int(seed),
            "geometry": {
                "radius": radius, "amplitude": modulation, "mode": mode,
                "hole_radius": hole, "boundary_points": nboundary,
            },
        },
    )


def spiral_sheet(
        *,
        turns: int = 5,
        r_inner: float = 0.12,
        r_outer: float = 0.75,
        sigma: float = 0.005,
        rho_bar: float = 1.0,
        theta_shift: float = 0.0,
) -> GuidingCenterCase:
    r"""Gaussian convolution of a finite Archimedean spiral density sheet.

    For ``gamma(a) = (r_inner + b*a) * (cos(a), sin(a))``,
    ``0 <= a <= 2*pi*turns`` and ``b = (r_outer-r_inner)/(2*pi*turns)``,
    the density is ``rho_bar/(sqrt(2*pi)*sigma)`` times the integral of
    ``exp(-|x-gamma(a)|**2/(2*sigma**2))`` along the curve. Thus the
    transverse Gaussian has approximately peak ``rho_bar`` and width
    ``sigma``, including smoothly rounded, half-height tips.

    Evaluate with 64-point Gauss-Legendre quadrature on nearby angular
    intervals. Radial and angular distance bounds discard only Gaussian
    tails beyond nine sigma, avoiding a points-by-entire-curve allocation.
    The same evaluation supports NumPy and CuPy arrays.
    """
    if not np.isfinite(turns) or int(turns) != turns or turns < 1:
        raise ValueError("turns must be a positive integer")
    count = int(turns)
    inner, outer = float(r_inner), float(r_outer)
    width, level, shift = float(sigma), float(rho_bar), float(theta_shift)
    if not all(np.isfinite(v) for v in (inner, outer, width, level, shift)):
        raise ValueError("spiral parameters must be finite")
    if not 0.0 < inner < outer < 1.0:
        raise ValueError("expected 0 < r_inner < r_outer < 1 on the unit disk")
    if width <= 0.0 or level <= 0.0:
        raise ValueError("sigma and rho_bar must be positive")
    end = 2.0 * np.pi * count
    pitch = (outer - inner) / end
    cutoff = 9.0 * width
    nodes, weights = np.polynomial.legendre.leggauss(64)
    normalization = float(level / (np.sqrt(2.0 * np.pi) * width))

    def initial_density(x, y):
        xp, x_arr, y_arr = _xy_arrays(x, y)
        x_arr, y_arr = xp.broadcast_arrays(x_arr, y_arr)
        shape = x_arr.shape
        x_flat, y_flat = x_arr.ravel(), y_arr.ravel()
        radius = xp.hypot(x_flat, y_flat)
        density = xp.zeros_like(radius)
        active = (radius >= inner - cutoff) & (radius <= outer + cutoff)
        r = radius[active]
        theta = xp.mod(xp.arctan2(y_flat[active], x_flat[active]) - shift, 2.0 * np.pi)
        radial_lo = xp.maximum(0.0, (r - cutoff - inner) / pitch)
        radial_hi = xp.minimum(end, (r + cutoff - inner) / pitch)
        # |x-gamma|^2 = (r-R)^2 + 4*r*R*sin((a-theta)/2)^2.
        # On the radial interval R >= max(r_inner, r-cutoff), so this
        # angular window retains every curve point within nine sigma.
        denominator = 2.0 * xp.sqrt(r * xp.maximum(inner, r - cutoff))
        angular_half = 2.0 * xp.arcsin(xp.minimum(
            1.0, cutoff / xp.maximum(denominator, np.finfo(REAL_DTYPE).tiny),
        ))
        values = xp.zeros_like(r)
        # Include copies on both sides of the angular seam for both tips.
        for winding in range(-1, count + 1):
            center = theta + 2.0 * np.pi * winding
            lo = xp.maximum(radial_lo, center - angular_half)
            hi = xp.minimum(radial_hi, center + angular_half)
            nearby = hi > lo
            midpoint = 0.5 * (lo[nearby] + hi[nearby])
            half = 0.5 * (hi[nearby] - lo[nearby])
            local_r, local_center = r[nearby], center[nearby]
            integral = xp.zeros_like(local_r)
            for node, weight in zip(nodes, weights):
                alpha = midpoint + float(node) * half
                spiral_r = inner + pitch * alpha
                distance_sq = (local_r - spiral_r) ** 2 + (
                    4.0 * local_r * spiral_r * xp.sin(0.5 * (alpha - local_center)) ** 2
                )
                integral += float(weight) * xp.sqrt(spiral_r**2 + pitch**2) * xp.exp(
                    -distance_sq / (2.0 * width**2)
                )
            values[nearby] += normalization * half * integral
        density[active] = values
        return density.reshape(shape)

    return GuidingCenterCase(
        key="spiral_sheet",
        description="Thin Gaussian Archimedean spiral sheet with zero potential boundary and zero density flux.",
        initial_density=initial_density,
        potential_boundary=lambda x, y, t: _zero_like_xy(x, y),
        density_boundary=None,
        density_transport_boundary_mode="zero-flux",
        default_domain="disc",
        potential_boundary_constant=0.0,
        parameters={
            "turns": count, "r_inner": inner, "r_outer": outer,
            "sigma": width, "rho_bar": level, "theta_shift": shift,
        },
    )


def rho_helm_wave(
        *,
        U: float = 1.0,
        kx: float = 1.0,
        ky: float = 1.0,
        A: float = 1.0,
) -> GuidingCenterCase:
    r"""Return the manufactured Helmholtz-wave guiding-center pair.

    ``phi = A cos(theta) + U y`` and ``rho = A |k|^2 cos(theta)`` with
    ``theta = kx (x - U t) + ky y`` satisfy ``-Delta(phi) = rho`` and the
    guiding-center transport equation ``rho_t + q^perp . grad(rho) = 0`` with
    ``q = -grad(phi)``. ``A = 1/|k|`` keeps the crest-parallel drift of unit
    amplitude for large wave numbers (high-resolution study); ``A = 1`` is the
    legacy case.
    """
    velocity = float(U)
    wave_x = float(kx)
    wave_y = float(ky)
    amplitude = float(A)
    laplace_factor = wave_x * wave_x + wave_y * wave_y

    def phase(x, y, t):
        xp, x_arr, y_arr = _xy_arrays(x, y)
        return wave_x * (x_arr - velocity * float(t)) + wave_y * y_arr

    def potential(x, y, t):
        xp, x_arr, y_arr = _xy_arrays(x, y)
        return amplitude * xp.cos(phase(x_arr, y_arr, t)) + velocity * y_arr

    def density(x, y, t):
        xp, x_arr, y_arr = _xy_arrays(x, y)
        return amplitude * laplace_factor * xp.cos(phase(x_arr, y_arr, t))

    def flux(x, y, t):
        xp, x_arr, y_arr = _xy_arrays(x, y)
        s = phase(x_arr, y_arr, t)
        return amplitude * wave_x * xp.sin(s), amplitude * wave_y * xp.sin(s) - velocity

    def density_gradient(x, y, t):
        xp, x_arr, y_arr = _xy_arrays(x, y)
        sine = xp.sin(phase(x_arr, y_arr, t))
        return -amplitude * laplace_factor * wave_x * sine, -amplitude * laplace_factor * wave_y * sine

    def potential_gradient(x, y, t):
        qx, qy = flux(x, y, t)
        return -qx, -qy

    return GuidingCenterCase(
        key="rho_helm_wave",
        description="Legacy manufactured rho/phi Helmholtz wave with exact nonzero boundary data.",
        initial_density=lambda x, y: density(x, y, 0.0),
        potential_boundary=potential,
        density_boundary=density,
        density_transport_boundary_mode="eliminate",
        default_domain="rectangle",
        exact_density=density,
        exact_potential=potential,
        negative_laplacian_potential=density,
        exact_flux=flux,
        exact_density_gradient=density_gradient,
        exact_potential_gradient=potential_gradient,
        parameters={"U": velocity, "kx": wave_x, "ky": wave_y, "A": amplitude},
    )


# Cross-shaped polygon (arms |x| <= 0.6 or |y| <= 0.6 inside (-1, 1)^2): four re-entrant
# 270-degree corners at (+-0.6, +-0.6), area 3.36.
CROSS_POLYGON = (
    (-0.6, -1.0), (0.6, -1.0), (0.6, -0.6), (1.0, -0.6), (1.0, 0.6), (0.6, 0.6),
    (0.6, 1.0), (-0.6, 1.0), (-0.6, 0.6), (-1.0, 0.6), (-1.0, -0.6), (-0.6, -0.6),
)


def orbiting_vortex(
        *,
        rho0: float = 2.0 * np.pi,
        A: float = 40.0,
        R: float = 0.2,
        m: int = 2,
        d: float = 0.7,
        theta0: float = 0.0,
) -> GuidingCenterCase:
    r"""Compact vortex of finite regularity orbiting in a uniform background (exact, unforced).

    ``rho = rho0 + A (1 - s)_+^m`` with ``s = |x - c(t)|^2 / R^2`` and
    ``phi = -rho0 |x|^2 / 4 + Psi(s)``, ``-Delta Psi = A (1 - s)_+^m``. The background
    rotates rigidly with ``Omega = rho0 / 2``, the radial vortex does not advect itself,
    so the vortex centre ``c(t) = d (cos(theta0 + Omega t), sin(theta0 + Omega t))``
    orbits the origin and ``rho_t + u . grad(rho) = 0`` holds exactly with
    ``u = (d_y phi, -d_x phi)``. The density is ``C^{m-1}`` with a jump of its m-th
    radial derivative on ``|x - c| = R``: ``rho`` lies in ``H^{m+1/2-eps}`` only.
    Boundary data (Dirichlet ``phi``, inflow ``rho``) come from the exact solution,
    so any polygon containing part of the orbit is admissible; the default is the
    cross ``CROSS_POLYGON``; with the defaults the vortex starts in the right arm, covers the
    re-entrant corner (0.6, 0.6) at ``Omega t = pi / 4`` (distance 0.15 < R) and reaches the
    top arm at ``Omega t = pi / 2`` (t = 0.5).
    """
    background, amplitude, radius = float(rho0), float(A), float(R)
    smoothness, orbit, angle = int(m), float(d), float(theta0)
    if smoothness != m or smoothness < 1:
        raise ValueError("m must be a positive integer")
    if radius <= 0.0:
        raise ValueError("R must be positive")
    omega = 0.5 * background
    # Psi(s) = -A R^2/(4(m+1)) [H_{m+1}(min(s, 1)) + log(max(s, 1))],
    # H_n(s) = sum_{j=1}^n (1 - (1 - s)^j) / j;  Psi'(r)/r = -A/(2(m+1)) G(s),
    # G(s) = sum_{j=0}^m (1 - s)^j inside, 1/s outside (cancellation-free at s -> 0).
    psi_scale = -amplitude * radius * radius / (4.0 * (smoothness + 1))
    slope_scale = -amplitude / (2.0 * (smoothness + 1))

    def centre(t):
        phase = angle + omega * float(t)
        return orbit * np.cos(phase), orbit * np.sin(phase)

    def local(x, y, t):
        xp, x_arr, y_arr = _xy_arrays(x, y)
        cx, cy = centre(t)
        dx, dy = x_arr - cx, y_arr - cy
        s = (dx * dx + dy * dy) / (radius * radius)
        return xp, x_arr, y_arr, dx, dy, s

    def density(x, y, t):
        xp, _, _, _, _, s = local(x, y, t)
        return background + amplitude * xp.maximum(1.0 - s, 0.0) ** smoothness

    def potential(x, y, t):
        xp, x_arr, y_arr, _, _, s = local(x, y, t)
        w = 1.0 - xp.minimum(s, 1.0)
        harmonic = sum((1.0 - w**j) / j for j in range(1, smoothness + 2))
        return (-0.25 * background * (x_arr * x_arr + y_arr * y_arr)
                + psi_scale * (harmonic + xp.log(xp.maximum(s, 1.0))))

    def potential_gradient(x, y, t):
        xp, x_arr, y_arr, dx, dy, s = local(x, y, t)
        w = xp.maximum(1.0 - s, 0.0)
        inside = sum(w**j for j in range(smoothness + 1))
        g = slope_scale * xp.where(s < 1.0, inside, 1.0 / xp.maximum(s, 1.0))
        return -0.5 * background * x_arr + g * dx, -0.5 * background * y_arr + g * dy

    def flux(x, y, t):
        gx, gy = potential_gradient(x, y, t)
        return -gx, -gy

    def density_gradient(x, y, t):
        xp, _, _, dx, dy, s = local(x, y, t)
        w = xp.maximum(1.0 - s, 0.0)
        factor = xp.where(s < 1.0, -2.0 * amplitude * smoothness * w ** (smoothness - 1) / (radius * radius), 0.0)
        return factor * dx, factor * dy

    return GuidingCenterCase(
        key="orbiting_vortex",
        description=("Compact C^{m-1} vortex orbiting in a rigidly rotating background; "
                     "exact solution, exact nonzero boundary data."),
        initial_density=lambda x, y: density(x, y, 0.0),
        potential_boundary=potential,
        density_boundary=density,
        density_transport_boundary_mode="eliminate",
        default_domain="polygon",
        exact_density=density,
        exact_potential=potential,
        negative_laplacian_potential=density,
        exact_flux=flux,
        exact_density_gradient=density_gradient,
        exact_potential_gradient=potential_gradient,
        parameters={"rho0": background, "A": amplitude, "R": radius, "m": smoothness, "d": orbit,
                    "theta0": angle, "geometry": {"vertices": CROSS_POLYGON}},
    )


# Cases whose density boundary data are the exact solution (transport must eliminate them).
EXACT_BOUNDARY_CASES = frozenset({"rho_helm_wave", "orbiting_vortex"})


def imported_equilibrium(**params):
    """Load an archived potential/window without requiring a DOLFINx runtime."""
    from scripts.guiding_center.cases.imported_equilibrium import imported_equilibrium as factory
    return factory(**params)


CASE_DEFINITIONS: dict[str, GuidingCenterCaseDefinition] = {
    "imported_equilibrium": GuidingCenterCaseDefinition(
        key="imported_equilibrium", description="Archived CG2 equiband on its original polygon",
        factory=imported_equilibrium, default_domain="imported"),
    "diocotron_gaussian_annulus": GuidingCenterCaseDefinition(
        key="diocotron_gaussian_annulus",
        description="Legacy Gaussian-annulus diocotron perturbation; default azimuthal mode k=3.",
        factory=diocotron_gaussian_annulus,
        default_domain="disc",
        default_params={"k": 3},
    ),
    "diocotron_k": GuidingCenterCaseDefinition(
        key="diocotron_k",
        description="Annular-band single-mode diocotron perturbation; default azimuthal mode k=3.",
        factory=diocotron_k,
        default_domain="disc",
        default_params={"k": 3},
    ),
    "euler_vortex_gas": GuidingCenterCaseDefinition(
        key="euler_vortex_gas",
        description="360 signed Gaussian vortices at four core sizes, spread across the unit disk.",
        factory=euler_vortex_gas,
        default_domain="disc",
    ),
    "positive_turbulence": GuidingCenterCaseDefinition(
        key="positive_turbulence",
        description=(
            "360 positive Gaussian density blobs at four core sizes, compactly "
            "supported away from the unit-disk wall."
        ),
        factory=positive_turbulence,
        default_domain="disc",
    ),
    "euler_star_vortex_gas": GuidingCenterCaseDefinition(
        key="euler_star_vortex_gas",
        description="360 signed Gaussian vortices in a five-lobed nonconvex star with a circular hole.",
        factory=euler_star_vortex_gas,
        default_domain="smooth-star",
    ),
    "euler_shaped_vortex_gas": GuidingCenterCaseDefinition(
        key="euler_shaped_vortex_gas",
        description="Multiscale signed vortex gas in a horseshoe, supplied ITER wall, or Pac-Man domain.",
        factory=euler_shaped_vortex_gas,
        default_domain="horseshoe",
    ),
    "spiral_sheet": GuidingCenterCaseDefinition(
        key="spiral_sheet",
        description="Gaussian-smoothed finite Archimedean spiral; default five turns with sigma=0.005.",
        factory=spiral_sheet,
        default_domain="disc",
    ),
    "rho_helm_wave": GuidingCenterCaseDefinition(
        key="rho_helm_wave",
        description="Manufactured Helmholtz-wave density/potential pair from the legacy guiding-center tests.",
        factory=rho_helm_wave,
        default_domain="rectangle",
    ),
    "orbiting_vortex": GuidingCenterCaseDefinition(
        key="orbiting_vortex",
        description="Exact compact C^{m-1} vortex orbiting in a rotating background, cross-shaped polygon.",
        factory=orbiting_vortex,
        default_domain="polygon",
    ),
    "diocotron_annulus": GuidingCenterCaseDefinition(
        key="diocotron_annulus",
        description=("Diocotron band between an inner and an outer wall (semi-Lagrangian benchmark "
                     "geometry, scaled to an outer wall of radius 1); interface-conforming sharp band."),
        factory=diocotron_annulus,
        default_domain="annulus",
        default_params={"k": 3},
    ),
    "magnetron": GuidingCenterCaseDefinition(
        key="magnetron",
        description=("Crossed-field diode: emission-fed hub on a cathode, rotating anode wave, spokes and "
                     "anode current through a non-equipotential wall."),
        factory=magnetron,
        default_domain="annulus",
    ),
}

CASE_BY_KEY = CASE_DEFINITIONS


def case_definition_by_key(key: str) -> GuidingCenterCaseDefinition:
    """Return a guiding-center case definition by key."""
    try:
        return CASE_DEFINITIONS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(CASE_DEFINITIONS))
        raise ValueError(f"unknown guiding-center case {key!r}; valid cases are {valid}") from exc


__all__ = [
    "CASE_BY_KEY",
    "CASE_DEFINITIONS",
    "GuidingCenterCase",
    "GuidingCenterCaseDefinition",
    "case_definition_by_key",
    "CROSS_POLYGON",
    "EXACT_BOUNDARY_CASES",
    "orbiting_vortex",
    "diocotron_gaussian_annulus",
    "diocotron_k",
    "rho_eq_annular_band",
    "rho_eq_gaussian_annulus",
    "rho_eq_super_gaussian_annulus",
    "rho_helm_wave",
    "spiral_sheet",
    "euler_vortex_gas",
    "positive_turbulence",
    "euler_star_vortex_gas",
    "euler_shaped_vortex_gas",
    "diocotron_annulus",
    "magnetron",
]
