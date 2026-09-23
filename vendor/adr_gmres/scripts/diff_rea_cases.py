"""Manufactured diffusion-reaction cases used by development drivers and tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DiffusionReactionProblem:
    """Manufactured diffusion-reaction data.

    ``exact_flux`` is the conservative flux ``q=-kappa grad u`` used by the
    runner to report raw and postprocessed flux errors.  Iteration intentionally
    yields only ``(diffusion, reaction, source, exact)`` so older tests and
    scripts that unpack four values continue to work.
    """

    diffusion: Any
    reaction: Callable
    source: Callable
    exact: Callable
    exact_flux: Callable

    def __iter__(self):
        """Preserve legacy four-value unpacking: diffusion, reaction, source, exact."""
        yield self.diffusion
        yield self.reaction
        yield self.source
        yield self.exact


ProblemTuple = DiffusionReactionProblem


@dataclass(frozen=True)
class DiffusionReactionCase:
    """Metadata and factory for a manufactured diffusion-reaction problem."""

    legacy_id: int
    key: str
    name: str
    factory: Callable[..., ProblemTuple]
    default_domain: str = "rectangle"

    def build(self, **params) -> ProblemTuple:
        """Return manufactured PDE data for this case."""
        return self.factory(**params)


def zero_coefficient(x, y):
    """Zero callable with NumPy broadcasting semantics."""
    return 0.0 * x * y


def identity_diffusion_tensor():
    """Identity diffusion tensor in symmetric component form ``(k00, k01, k11)``."""
    return 1.0, 0.0, 1.0


def _identity_flux(gradx: Callable, grady: Callable) -> Callable:
    """Return conservative identity-diffusion flux ``q=-grad u``."""
    return lambda x, y: (-gradx(x, y), -grady(x, y))


def quadratic_poisson_case() -> ProblemTuple:
    """Quadratic exact solution with identity diffusion and no reaction."""
    gradx = lambda x, y: 2.0 * x + 0.0 * y
    grady = lambda x, y: 2.0 * y + 0.0 * x
    return DiffusionReactionProblem(
        diffusion=identity_diffusion_tensor(),
        reaction=zero_coefficient,
        source=lambda x, y: -4.0 + 0.0 * x * y,
        exact=lambda x, y: 1.0 + x**2 + y**2,
        exact_flux=_identity_flux(gradx, grady),
    )


def exponential_bubble_poisson_case() -> ProblemTuple:
    """Exponential polynomial bubble with identity diffusion and no reaction."""
    exact = lambda x, y: np.exp(x - y) * x * (1.0 - x) * y * (1.0 - y)

    def gradx(x, y):
        return np.exp(x - y) * y * (1.0 - y) * ((1.0 - 2.0 * x) + x * (1.0 - x))

    def grady(x, y):
        return np.exp(x - y) * x * (1.0 - x) * ((1.0 - 2.0 * y) - y * (1.0 - y))

    return DiffusionReactionProblem(
        diffusion=identity_diffusion_tensor(),
        reaction=zero_coefficient,
        source=lambda x, y: -2.0 * x * (y - 1.0) * (y - 2.0 * x + x * y + 2.0) * np.exp(x - y),
        exact=exact,
        exact_flux=_identity_flux(gradx, grady),
    )


def trigonometric_poisson_case() -> ProblemTuple:
    """Smooth trigonometric exact solution, usually run on a disk."""
    exact = lambda x, y: np.sin(x**2 + y**2) + np.sin(x * y)
    gradx = lambda x, y: 2.0 * x * np.cos(x**2 + y**2) + y * np.cos(x * y)
    grady = lambda x, y: 2.0 * y * np.cos(x**2 + y**2) + x * np.cos(x * y)
    return DiffusionReactionProblem(
        diffusion=identity_diffusion_tensor(),
        reaction=zero_coefficient,
        source=lambda x, y: -(
            4.0 * np.cos(x**2 + y**2)
            - (x**2 + y**2) * (np.sin(x * y) + 4.0 * np.sin(x**2 + y**2))
        ),
        exact=exact,
        exact_flux=_identity_flux(gradx, grady),
    )


def quadratic_variable_reaction_case() -> ProblemTuple:
    """Quadratic exact solution with a smooth variable reaction coefficient."""

    def reaction(x, y):
        return np.cos(3.0 * np.pi * x) + np.cos(3.0 * np.pi * y) + 2.0

    gradx = lambda x, y: 2.0 * x + 0.0 * y
    grady = lambda x, y: 2.0 * y + 0.0 * x
    return DiffusionReactionProblem(
        diffusion=identity_diffusion_tensor(),
        reaction=reaction,
        source=lambda x, y: -4.0 + reaction(x, y) * (x**2 + y**2),
        exact=lambda x, y: x**2 + y**2,
        exact_flux=_identity_flux(gradx, grady),
    )


def lshape_singular_harmonic_case() -> ProblemTuple:
    """Reentrant-corner singular harmonic solution for the L-shaped domain."""
    alpha = 2.0 / 3.0

    def exact(x, y):
        return (x**2 + y**2) ** (alpha / 2.0) * np.sin(alpha * (np.arctan2(y, x) + np.pi / 2.0))

    def exact_flux(x, y):
        radius = np.sqrt(x**2 + y**2)
        safe_radius = np.maximum(radius, np.finfo(float).tiny)
        angle = np.arctan2(y, x) + np.pi / 2.0
        factor = alpha * safe_radius ** (alpha - 1.0)
        sin_term = np.sin(alpha * angle)
        cos_term = np.cos(alpha * angle)
        ux = factor * (sin_term * x / safe_radius - cos_term * y / safe_radius)
        uy = factor * (sin_term * y / safe_radius + cos_term * x / safe_radius)
        return -ux, -uy

    return DiffusionReactionProblem(
        diffusion=identity_diffusion_tensor(),
        reaction=zero_coefficient,
        source=zero_coefficient,
        exact=exact,
        exact_flux=exact_flux,
    )


def _tensor_sine_data(m: int = 1, n: int = 1):
    r"""Return tensor sine manufactured data and exact gradients on ``[-1, 1]^2``."""
    a = 0.5 * int(m) * np.pi
    b = 0.5 * int(n) * np.pi

    def exact(x, y):
        return np.sin(a * (x + 1.0)) * np.sin(b * (y + 1.0))

    def gradx(x, y):
        return a * np.cos(a * (x + 1.0)) * np.sin(b * (y + 1.0))

    def grady(x, y):
        return b * np.sin(a * (x + 1.0)) * np.cos(b * (y + 1.0))

    def reaction(x, y):
        return 1.0 + x**2 + y**2

    def source(x, y):
        u = exact(x, y)
        ux = gradx(x, y)
        uy = grady(x, y)
        uxx = -(a**2) * u
        uyy = -(b**2) * u
        uxy = a * b * np.cos(a * (x + 1.0)) * np.cos(b * (y + 1.0))
        div_kappa_grad_u = (
            (2.0 + x**2) * uxx
            + x * y * uxy
            + (3.0 + y**2) * uyy
            + 2.5 * x * ux
            + 2.5 * y * uy
        )
        return -div_kappa_grad_u + reaction(x, y) * u

    def k11(x, y):
        return 2.0 + x**2

    def k12(x, y):
        return 0.5 * x * y

    def k22(x, y):
        return 3.0 + y**2

    return (k11, k12, k22), reaction, source, exact, gradx, grady


def tensor_sine_diffusion_reaction_case(m: int = 1, n: int = 1) -> ProblemTuple:
    """Smooth manufactured tensor-diffusion reaction case."""
    diffusion, reaction, source, exact, gradx, grady = _tensor_sine_data(m, n)
    k11, k12, k22 = diffusion

    def exact_flux(x, y):
        return (
            -(k11(x, y) * gradx(x, y) + k12(x, y) * grady(x, y)),
            -(k12(x, y) * gradx(x, y) + k22(x, y) * grady(x, y)),
        )

    return DiffusionReactionProblem(
        diffusion=diffusion,
        reaction=reaction,
        source=source,
        exact=exact,
        exact_flux=exact_flux,
    )


def tensor_sine_exact_gradients(m: int = 1, n: int = 1) -> tuple[Callable, Callable]:
    """Return exact gradient callables for :func:`tensor_sine_diffusion_reaction_case`."""
    _, _, _, _, gradx, grady = _tensor_sine_data(m, n)
    return gradx, grady


def rotated_anisotropic_sine_case(
    anisotropy_ratio: float = 1.0e3,
    angle_degrees: float = 30.0,
    m: int = 1,
    n: int = 1,
) -> ProblemTuple:
    r"""Sine solution with a constant rotated anisotropic diffusion tensor.

    The tensor eigenvalues are ``1`` and ``1 / anisotropy_ratio``.  Rotating
    its principal axes exercises the mixed derivative term and prevents the
    mesh axes from hiding anisotropy-related errors.  The exact solution
    vanishes on the boundary of ``[-1, 1]^2``.
    """

    ratio = float(anisotropy_ratio)
    angle = float(angle_degrees)
    if not np.isfinite(ratio) or ratio < 1.0:
        raise ValueError("anisotropy_ratio must be finite and at least one")
    if not np.isfinite(angle):
        raise ValueError("angle_degrees must be finite")
    if isinstance(m, bool) or int(m) != m or int(m) <= 0:
        raise ValueError("m must be a positive integer")
    if isinstance(n, bool) or int(n) != n or int(n) <= 0:
        raise ValueError("n must be a positive integer")

    theta = np.deg2rad(angle)
    cosine = float(np.cos(theta))
    sine = float(np.sin(theta))
    parallel = 1.0
    perpendicular = 1.0 / ratio
    k00 = parallel * cosine**2 + perpendicular * sine**2
    k01 = (parallel - perpendicular) * cosine * sine
    k11 = parallel * sine**2 + perpendicular * cosine**2

    alpha = 0.5 * int(m) * np.pi
    beta = 0.5 * int(n) * np.pi

    def exact(x, y):
        return np.sin(alpha * (x + 1.0)) * np.sin(beta * (y + 1.0))

    def gradx(x, y):
        return alpha * np.cos(alpha * (x + 1.0)) * np.sin(beta * (y + 1.0))

    def grady(x, y):
        return beta * np.sin(alpha * (x + 1.0)) * np.cos(beta * (y + 1.0))

    def source(x, y):
        u = exact(x, y)
        uxx = -(alpha**2) * u
        uyy = -(beta**2) * u
        uxy = (
            alpha
            * beta
            * np.cos(alpha * (x + 1.0))
            * np.cos(beta * (y + 1.0))
        )
        return -(k00 * uxx + 2.0 * k01 * uxy + k11 * uyy)

    def exact_flux(x, y):
        ux = gradx(x, y)
        uy = grady(x, y)
        return (-(k00 * ux + k01 * uy), -(k01 * ux + k11 * uy))

    return DiffusionReactionProblem(
        diffusion=(k00, k01, k11),
        reaction=zero_coefficient,
        source=source,
        exact=exact,
        exact_flux=exact_flux,
    )


CASE_DEFINITIONS: tuple[DiffusionReactionCase, ...] = (
    DiffusionReactionCase(0, "quadratic-poisson", "quadratic Poisson", quadratic_poisson_case),
    DiffusionReactionCase(2, "exponential-bubble", "exponential bubble Poisson", exponential_bubble_poisson_case),
    DiffusionReactionCase(3, "trigonometric-poisson", "trigonometric Poisson", trigonometric_poisson_case, "disc"),
    DiffusionReactionCase(
        5,
        "quadratic-variable-reaction",
        "quadratic variable-reaction",
        quadratic_variable_reaction_case,
    ),
    DiffusionReactionCase(6, "lshape-singular", "L-shape singular harmonic", lshape_singular_harmonic_case, "lshape"),
    DiffusionReactionCase(
        7,
        "tensor-sine",
        "tensor sine diffusion-reaction",
        tensor_sine_diffusion_reaction_case,
    ),
    DiffusionReactionCase(
        8,
        "rotated-anisotropic-sine",
        "rotated strongly anisotropic sine diffusion",
        rotated_anisotropic_sine_case,
        "structured-rectangle",
    ),
)

CASE_BY_KEY = {case.key: case for case in CASE_DEFINITIONS}
CASE_BY_LEGACY_ID = {case.legacy_id: case for case in CASE_DEFINITIONS}


def case_definition_by_key(key: str) -> DiffusionReactionCase:
    """Return case metadata by descriptive key."""
    try:
        return CASE_BY_KEY[key]
    except KeyError as exc:
        valid = ", ".join(sorted(CASE_BY_KEY))
        raise ValueError(f"unknown diffusion-reaction case {key!r}; valid cases are {valid}") from exc


def case_definition_by_legacy_id(legacy_id: int) -> DiffusionReactionCase:
    """Return case metadata by legacy numeric ID."""
    try:
        return CASE_BY_LEGACY_ID[int(legacy_id)]
    except KeyError as exc:
        valid = ", ".join(str(key) for key in sorted(CASE_BY_LEGACY_ID))
        raise ValueError(f"supported legacy diffusion-reaction tests are {valid}") from exc


def case_by_key(key: str, **params) -> ProblemTuple:
    """Return manufactured PDE data by descriptive key."""
    return case_definition_by_key(key).build(**params)


def case_by_legacy_id(legacy_id: int, **params) -> ProblemTuple:
    """Return manufactured PDE data by legacy numeric ID."""
    return case_definition_by_legacy_id(legacy_id).build(**params)


def legacy_case_factories() -> dict[str, Callable[[], ProblemTuple]]:
    """Return factories keyed by legacy benchmark names like ``test0``."""
    return {f"test{case.legacy_id}": case.factory for case in CASE_DEFINITIONS if case.legacy_id != 7}


__all__ = [
    "CASE_BY_KEY",
    "CASE_BY_LEGACY_ID",
    "CASE_DEFINITIONS",
    "DiffusionReactionCase",
    "DiffusionReactionProblem",
    "case_by_key",
    "case_by_legacy_id",
    "case_definition_by_key",
    "case_definition_by_legacy_id",
    "exponential_bubble_poisson_case",
    "identity_diffusion_tensor",
    "legacy_case_factories",
    "lshape_singular_harmonic_case",
    "quadratic_poisson_case",
    "quadratic_variable_reaction_case",
    "rotated_anisotropic_sine_case",
    "tensor_sine_diffusion_reaction_case",
    "tensor_sine_exact_gradients",
    "trigonometric_poisson_case",
    "zero_coefficient",
]
