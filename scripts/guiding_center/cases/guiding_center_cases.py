"""Guiding-center case definitions used by the fixed-mesh runner."""

from __future__ import annotations

from hdgfem.runtime.precision import REAL_DTYPE

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
        profile_path: str | None = None,
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

    ``geometry="smooth-star"`` is the README showcase domain: a five-lobed
    star around a circular island (``geometry_params`` override the radius,
    lobe amplitude, mode, island radius, boundary points and Gmsh threads).
    It requires ``profile_path``, a profile saved by
    :meth:`hdgfem.cases.profiles.GaussianBlobField.save` (relative paths are
    taken from the repository root); the stored blobs are reused exactly and
    their cutoff must equal ``cutoff``. ``counts``, ``sigmas``, ``seed`` and
    ``wall_gap`` then only describe how that profile was sampled.
    """
    from hdgfem.core.geometry import DiskDomain, shaped_domain
    from hdgfem.cases.profiles import sample_gaussian_blob_field

    counts, sigmas = tuple(counts), tuple(sigmas)
    geometry_params = dict(geometry_params or {})
    if geometry == "smooth-star":
        from pathlib import Path
        from hdgfem.cases.profiles import GaussianBlobField

        if profile_path is None or fft_grid_shape is not None:
            raise ValueError("smooth-star positive turbulence requires profile_path and no FFT grid")
        geometry_params = {"radius": 1.0, "amplitude": 0.35, "mode": 5, "hole_radius": 0.3,
                           "boundary_points": 500, "num_threads": 16, **geometry_params}
        path = Path(profile_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        if not path.is_file():
            raise FileNotFoundError(f"saved Gaussian-blob profile not found: {path}")
        density = GaussianBlobField.load(path, amplitude=amplitude)
        if density.cutoff != float(cutoff):
            raise ValueError(f"{path} was sampled with cutoff {density.cutoff:g}, not {cutoff:g}")
        widths, counts = np.unique(density.sigmas, return_counts=True)
        counts, sigmas = tuple(int(n) for n in counts), tuple(float(w) for w in widths)
        domain_label = "five-lobed star around a circular island (README showcase profile)"
    elif geometry == "disc":
        if geometry_params:
            raise ValueError("geometry_params are only supported for shaped domains")
        domain = DiskDomain()
        domain_label = "unit disk"
    else:
        domain = shaped_domain(geometry, **geometry_params)
        domain_label = "ITER" if geometry == "iter" else geometry
    if profile_path is not None and geometry != "smooth-star":
        raise ValueError("profile_path is only supported with geometry='smooth-star'")
    if geometry != "smooth-star":
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
            **({"profile_path": str(profile_path)} if profile_path is not None else {}),
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
    from hdgfem.core.geometry import shaped_domain
    from hdgfem.cases.profiles import sample_gaussian_blob_field

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

    def density_gradient(x, y, t):
        xp, x_arr, y_arr = _xy_arrays(x, y)
        sine = xp.sin(phase(x_arr, y_arr, t))
        return -laplace_factor * wave_x * sine, -laplace_factor * wave_y * sine

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
    "spiral_sheet",
    "euler_vortex_gas",
    "positive_turbulence",
    "euler_star_vortex_gas",
    "euler_shaped_vortex_gas",
]
