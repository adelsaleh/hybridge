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
    return xp, xp.asarray(x, dtype=xp.float64), xp.asarray(y, dtype=xp.float64)


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
        boundary = lambda x, y: self.potential_boundary(x, y, t)
        if self.potential_boundary_constant is not None:
            boundary._hdgfem_constant_value = float(self.potential_boundary_constant)
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
    return float(rho_bar) * xp.exp(-xp.abs((r - center) / scale) ** power)



def diocotron_k(
        *,
        k: int = 3,
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
) -> GuidingCenterCase:
    """Return an annular-band single-mode diocotron perturbation case.

    ``k`` is the azimuthal mode. Supplying ``p`` selects the radial
    super-Gaussian profile
    ``exp(-abs((r-s_bar)/s_d)**p)``. By default, ``s_bar`` and ``s_d`` are
    inferred from the midpoint and half-width of ``s_minus``/``s_plus``.
    """
    mode = int(k)
    if mode < 1:
        raise ValueError("k must be at least 1")
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
            )
        return rho_eq_annular_band(
            x,
            y,
            s_minus=inner,
            s_plus=outer,
            rho_bar=density_level,
            edge_width=transition,
        )

    def initial_density(x, y):
        xp, x_arr, y_arr = _xy_arrays(x, y)
        theta = xp.arctan2(y_arr, x_arr) - shift
        return equilibrium(x_arr, y_arr) * (1.0 + amplitude * xp.cos(mode * theta))

    return GuidingCenterCase(
        key="diocotron_k",
        description=f"Annular-band diocotron perturbation with azimuthal mode k={mode}, zero potential boundary, and zero density flux.",
        initial_density=initial_density,
        potential_boundary=lambda x, y, t: _zero_like_xy(x, y),
        density_boundary=None,
        density_transport_boundary_mode="zero-flux",
        default_domain="disc",
        potential_boundary_constant=0.0,
        equilibrium_density=equilibrium,
        parameters={
            "k": mode,
            "epsilon": amplitude,
            "s_minus": inner,
            "s_plus": outer,
            "rho_bar": density_level,
            "edge_width": transition,
            "s_bar": radial_center,
            "s_d": radial_scale,
            "p": radial_power,
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
        xp, x_arr, y_arr = _xy_arrays(x, y)
        return wave_x * (x_arr - velocity * float(t)) + wave_y * y_arr

    def potential(x, y, t):
        xp, x_arr, y_arr = _xy_arrays(x, y)
        return xp.cos(phase(x_arr, y_arr, t)) + velocity * y_arr

    def density(x, y, t):
        xp, x_arr, y_arr = _xy_arrays(x, y)
        return laplace_factor * xp.cos(phase(x_arr, y_arr, t))

    def flux(x, y, t):
        xp, x_arr, y_arr = _xy_arrays(x, y)
        s = phase(x_arr, y_arr, t)
        return wave_x * xp.sin(s), wave_y * xp.sin(s) - velocity

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
    "diocotron_gaussian_annulus",
    "diocotron_k",
    "rho_eq_annular_band",
    "rho_eq_gaussian_annulus",
    "rho_eq_super_gaussian_annulus",
    "rho_helm_wave",
]
