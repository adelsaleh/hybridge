"""Guiding-center case definitions used by the fixed-mesh runner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

ScalarCallable = Callable[[Any, Any], Any]
TimeScalarCallable = Callable[[Any, Any, float], Any]


def _zero_like_xy(x, y):
    """Return a backend-compatible zero with coordinate broadcasting."""
    return 0.0 * x + 0.0 * y


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
    equilibrium_density: ScalarCallable | None = None
    exact_density: TimeScalarCallable | None = None
    exact_potential: TimeScalarCallable | None = None
    negative_laplacian_potential: TimeScalarCallable | None = None
    exact_flux: Callable[[Any, Any, float], tuple[Any, Any]] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

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
        return lambda x, y: self.potential_boundary(x, y, t)

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


def rho_eq_diocotron(x, y, *, r0: float = 0.45, sigma: float = 0.03):
    """Classical annular Gaussian density profile used by the legacy diocotron test."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r = np.sqrt(x * x + y * y)
    return np.exp(-((r - float(r0)) ** 2) / (2.0 * float(sigma) ** 2))


def diocotron_k(
        *,
        k: int = 3,
        eps: float = 0.05,
        r0: float = 0.45,
        sigma: float = 0.03,
) -> GuidingCenterCase:
    """Return the fixed-disc diocotron perturbation case."""
    mode = int(k)
    amplitude = float(eps)
    radius0 = float(r0)
    width = float(sigma)

    def equilibrium(x, y):
        return rho_eq_diocotron(x, y, r0=radius0, sigma=width)

    def initial_density(x, y):
        theta = np.arctan2(y, x)
        return (1.0 + amplitude * np.cos(mode * theta)) * equilibrium(x, y)

    return GuidingCenterCase(
        key="diocotron_k",
        description="Classical fixed-disc diocotron perturbation with zero potential boundary and zero density flux.",
        initial_density=initial_density,
        potential_boundary=lambda x, y, t: _zero_like_xy(x, y),
        density_boundary=None,
        density_transport_boundary_mode="zero-flux",
        default_domain="disc",
        equilibrium_density=equilibrium,
        parameters={"k": mode, "eps": amplitude, "r0": radius0, "sigma": width},
    )


def diocotron_azimuthal_perturbation(
        theta,
        *,
        m_min: int = 3,
        n_modes: int = 100,
        theta_shift: float = 0.0,
        xp=np,
):
    """Return an equal-amplitude broadband azimuthal perturbation."""
    if int(m_min) < 1:
        raise ValueError("m_min must be at least 1")
    if int(n_modes) < 1:
        raise ValueError("n_modes must be at least 1")

    theta = xp.asarray(theta)
    psi = theta - float(theta_shift)
    psi = (psi + xp.pi) % (2.0 * xp.pi) - xp.pi

    mode_min = int(m_min)
    mode_count = int(n_modes)
    mode_max = mode_min + mode_count - 1
    mean_mode = 0.5 * (mode_min + mode_max)
    envelope = xp.sinc(mode_count * psi / (2.0 * xp.pi)) / xp.sinc(psi / (2.0 * xp.pi))
    return envelope * xp.cos(mean_mode * psi)


def diocotron_equilibrium_polar(
        r,
        *,
        s_minus: float = 0.79,
        s_plus: float = 0.80,
        rho_bar: float = 1.0,
        xp=np,
):
    """Sharp annular equilibrium ``rho_0(r)``."""
    inner = float(s_minus)
    outer = float(s_plus)
    if not 0.0 <= inner < outer:
        raise ValueError("expected 0 <= s_minus < s_plus")
    r = xp.asarray(r)
    return xp.where((r >= inner) & (r <= outer), float(rho_bar), 0.0)


def diocotron_perturbation_polar(
        r,
        theta,
        *,
        s_minus: float = 0.79,
        s_plus: float = 0.80,
        rho_bar: float = 1.0,
        epsilon: float = 2.0e-2,
        m_min: int = 3,
        n_modes: int = 100,
        theta_shift: float = 0.0,
        xp=np,
):
    """Return the broadband perturbation ``rho_1(r, theta, 0)``."""
    rho_0 = diocotron_equilibrium_polar(
        r,
        s_minus=s_minus,
        s_plus=s_plus,
        rho_bar=rho_bar,
        xp=xp,
    )
    eta = diocotron_azimuthal_perturbation(
        theta,
        m_min=m_min,
        n_modes=n_modes,
        theta_shift=theta_shift,
        xp=xp,
    )
    return float(epsilon) * rho_0 * eta


def diocotron_initial_density_polar(
        r,
        theta,
        *,
        s_minus: float = 0.79,
        s_plus: float = 0.80,
        rho_bar: float = 1.0,
        epsilon: float = 2.0e-2,
        m_min: int = 3,
        n_modes: int = 100,
        theta_shift: float = 0.0,
        xp=np,
):
    """Return ``rho_0(r) * (1 + epsilon * eta(theta))``."""
    rho_0 = diocotron_equilibrium_polar(
        r,
        s_minus=s_minus,
        s_plus=s_plus,
        rho_bar=rho_bar,
        xp=xp,
    )
    eta = diocotron_azimuthal_perturbation(
        theta,
        m_min=m_min,
        n_modes=n_modes,
        theta_shift=theta_shift,
        xp=xp,
    )
    return rho_0 * (1.0 + float(epsilon) * eta)


def diocotron_initial_density_cartesian(
        x,
        y,
        *,
        s_minus: float = 0.79,
        s_plus: float = 0.80,
        rho_bar: float = 1.0,
        epsilon: float = 2.0e-2,
        m_min: int = 3,
        n_modes: int = 100,
        theta_shift: float = 0.0,
        xp=np,
):
    """Cartesian broadband diocotron density wrapper.  Pass ``xp=cupy`` for CuPy arrays."""
    x = xp.asarray(x)
    y = xp.asarray(y)
    r = xp.sqrt(x * x + y * y)
    theta = xp.arctan2(y, x)
    return diocotron_initial_density_polar(
        r,
        theta,
        s_minus=s_minus,
        s_plus=s_plus,
        rho_bar=rho_bar,
        epsilon=epsilon,
        m_min=m_min,
        n_modes=n_modes,
        theta_shift=theta_shift,
        xp=xp,
    )


def rho_eq_diocotron_annulus(
        x,
        y,
        *,
        s_minus: float = 0.79,
        s_plus: float = 0.80,
        rho_bar: float = 1.0,
):
    """Sharp annular diocotron equilibrium density profile."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    r = np.sqrt(x * x + y * y)
    return diocotron_equilibrium_polar(
        r,
        s_minus=s_minus,
        s_plus=s_plus,
        rho_bar=rho_bar,
    )


def diocotron_broadband(
        *,
        s_minus: float = 0.79,
        s_plus: float = 0.80,
        rho_bar: float = 1.0,
        epsilon: float = 2.0e-2,
        eps: float | None = None,
        m_min: int = 3,
        n_modes: int = 100,
        theta_shift: float = 0.0,
) -> GuidingCenterCase:
    """Return the sharp-annulus broadband diocotron perturbation case."""
    inner = float(s_minus)
    outer = float(s_plus)
    density_level = float(rho_bar)
    amplitude = float(epsilon if eps is None else eps)
    mode_min = int(m_min)
    mode_count = int(n_modes)
    shift = float(theta_shift)

    def equilibrium(x, y):
        return rho_eq_diocotron_annulus(
            x,
            y,
            s_minus=inner,
            s_plus=outer,
            rho_bar=density_level,
        )

    def initial_density(x, y):
        return diocotron_initial_density_cartesian(
            x,
            y,
            s_minus=inner,
            s_plus=outer,
            rho_bar=density_level,
            epsilon=amplitude,
            m_min=mode_min,
            n_modes=mode_count,
            theta_shift=shift,
        )

    return GuidingCenterCase(
        key="diocotron_broadband",
        description="Sharp-annulus broadband diocotron perturbation with modes m_min..m_min+n_modes-1.",
        initial_density=initial_density,
        potential_boundary=lambda x, y, t: _zero_like_xy(x, y),
        density_boundary=None,
        density_transport_boundary_mode="zero-flux",
        default_domain="disc",
        equilibrium_density=equilibrium,
        parameters={
            "s_minus": inner,
            "s_plus": outer,
            "rho_bar": density_level,
            "epsilon": amplitude,
            "m_min": mode_min,
            "n_modes": mode_count,
            "theta_shift": shift,
        },
    )


def rho_helm_wave(
        *,
        U: float = 1.0,
        kx: float = 1.0,
        ky: float = 1.0,
) -> GuidingCenterCase:
    r"""Return the legacy manufactured Helmholtz-wave guiding-center pair.

    The exact potential and density satisfy ``-Delta(phi) = rho`` and the
    guiding-center transport equation ``rho_t + q^perp . grad(rho) = 0`` with
    ``q = -grad(phi)``.
    """
    velocity = float(U)
    wave_x = float(kx)
    wave_y = float(ky)
    laplace_factor = wave_x * wave_x + wave_y * wave_y

    def phase(x, y, t):
        return wave_x * (np.asarray(x, dtype=np.float64) - velocity * float(t)) + wave_y * np.asarray(y, dtype=np.float64)

    def potential(x, y, t):
        return np.cos(phase(x, y, t)) + velocity * np.asarray(y, dtype=np.float64)

    def density(x, y, t):
        return laplace_factor * np.cos(phase(x, y, t))

    def flux(x, y, t):
        s = phase(x, y, t)
        return wave_x * np.sin(s), wave_y * np.sin(s) - velocity

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
        parameters={"U": velocity, "kx": wave_x, "ky": wave_y},
    )


CASE_DEFINITIONS: dict[str, GuidingCenterCaseDefinition] = {
    "diocotron_k": GuidingCenterCaseDefinition(
        key="diocotron_k",
        description="Fixed-disc diocotron perturbation; default azimuthal mode k=3.",
        factory=diocotron_k,
        default_domain="disc",
        default_params={"k": 3},
    ),
    "diocotron_broadband": GuidingCenterCaseDefinition(
        key="diocotron_broadband",
        description="Sharp-annulus broadband diocotron perturbation; default modes m=3,...,102.",
        factory=diocotron_broadband,
        default_domain="disc",
        default_params={"m_min": 3, "n_modes": 100},
    ),
    "rho_helm_wave": GuidingCenterCaseDefinition(
        key="rho_helm_wave",
        description="Manufactured Helmholtz-wave density/potential pair from the legacy guiding-center tests.",
        factory=rho_helm_wave,
        default_domain="rectangle",
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
    "diocotron_azimuthal_perturbation",
    "diocotron_broadband",
    "diocotron_equilibrium_polar",
    "diocotron_initial_density_cartesian",
    "diocotron_initial_density_polar",
    "diocotron_k",
    "diocotron_perturbation_polar",
    "rho_eq_diocotron",
    "rho_eq_diocotron_annulus",
    "rho_helm_wave",
]
