"""Backend-neutral stabilization policies and geometric scale helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..core.space import DGSpace


DomainLength = float | Literal["auto"] | None


def compute_domain_length(
        domain_measure: float,
        boundary_measure: float,
        dim: int,
) -> float:
    r"""Return the global diffusive length :math:`d|\Omega|/|\partial\Omega|`."""
    measure = float(domain_measure)
    boundary = float(boundary_measure)
    dimension = int(dim)
    if not np.isfinite(measure) or measure <= 0.0:
        raise ValueError("domain_measure must be finite and positive")
    if not np.isfinite(boundary) or boundary <= 0.0:
        raise ValueError("boundary_measure must be finite and positive")
    if dimension <= 0:
        raise ValueError("dim must be positive")
    return dimension * measure / boundary


def mesh_domain_measures(space_or_mesh) -> tuple[float, float]:
    """Return physical area and exterior-boundary length for an affine triangle mesh."""
    mesh = space_or_mesh.mesh if isinstance(space_or_mesh, DGSpace) else space_or_mesh
    domain_measure = 2.0 * float(np.sum(np.asarray(mesh.aff_jacs, dtype=np.float64)))
    boundary_measure = 2.0 * float(
        np.sum(np.asarray(mesh.edge_jacs, dtype=np.float64)[mesh.bnd_edges_inds])
    )
    if not np.isfinite(domain_measure) or domain_measure <= 0.0:
        raise ValueError("mesh domain measure must be finite and positive")
    if not np.isfinite(boundary_measure) or boundary_measure <= 0.0:
        raise ValueError("mesh boundary measure must be finite and positive")
    return domain_measure, boundary_measure


def automatic_domain_length(space_or_mesh) -> float:
    r"""Compute :math:`2|\Omega|/|\partial\Omega|` for a two-dimensional mesh."""
    domain_measure, boundary_measure = mesh_domain_measures(space_or_mesh)
    return compute_domain_length(domain_measure, boundary_measure, dim=2)


def geometric_diffusion_tau(
        kappa_normal: float,
        domain_length: float,
        gamma_d: float = 1.0,
) -> float:
    r"""Return :math:`\gamma_d\kappa_n/L_\Omega` after validating its scale."""
    kappa = float(kappa_normal)
    length = float(domain_length)
    gamma = float(gamma_d)
    if not np.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("normal diffusivity must be finite and positive")
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("domain_length must be finite and positive")
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma_d must be finite and positive")
    return gamma * kappa / length


@dataclass(frozen=True)
class GlobalLengthDiffusion:
    r"""Mesh- and degree-independent policy :math:`\tau_d=\gamma_d\kappa/L_\Omega`.

    ``domain_length=None`` and ``"auto"`` both select
    :func:`automatic_domain_length`. The current implementation accepts a
    positive scalar or constant isotropic tensor diffusion coefficient; variable
    and anisotropic face-normal diffusivity remain separate qualification steps.
    """

    gamma_d: float = 1.0
    domain_length: DomainLength = "auto"

    @property
    def mode(self) -> str:
        """Return the stable public policy name."""
        return "global_length"

    def resolved_domain_length(self, space: DGSpace) -> float:
        """Return the explicit or geometry-derived physical length."""
        if self.domain_length is None or (
            isinstance(self.domain_length, str)
            and self.domain_length.strip().lower() == "auto"
        ):
            return automatic_domain_length(space)
        length = float(self.domain_length)
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("domain_length must be finite and positive or 'auto'")
        return length

    def resolve(self, diffusion: Any, space: DGSpace) -> float:
        """Resolve this policy for positive constant isotropic diffusion."""
        kappa = constant_isotropic_diffusivity(diffusion)
        return geometric_diffusion_tau(
            kappa,
            self.resolved_domain_length(space),
            self.gamma_d,
        )


def constant_isotropic_diffusivity(diffusion: Any) -> float:
    """Return scalar kappa from scalar or constant isotropic tensor data."""
    if np.isscalar(diffusion):
        return float(diffusion)
    try:
        values = np.asarray(diffusion, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise NotImplementedError(
            "global-length stabilization requires constant isotropic diffusion"
        ) from exc
    if values.shape == (3,):
        kappa_00, kappa_01, kappa_11 = (float(value) for value in values)
        matrix = np.array(
            ((kappa_00, kappa_01), (kappa_01, kappa_11)),
            dtype=np.float64,
        )
    elif values.shape == (2, 2):
        matrix = values
    else:
        raise NotImplementedError(
            "global-length stabilization requires constant isotropic diffusion"
        )
    scale = max(1.0, float(np.max(np.abs(matrix))))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    if (
        abs(float(matrix[0, 1])) > tolerance
        or abs(float(matrix[1, 0])) > tolerance
        or abs(float(matrix[0, 0] - matrix[1, 1])) > tolerance
    ):
        raise NotImplementedError(
            "global-length stabilization for anisotropic diffusion requires "
            "the deferred face-normal diffusivity policy"
        )
    return float(0.5 * (matrix[0, 0] + matrix[1, 1]))


def is_global_length_diffusion(value: Any) -> bool:
    """Return whether ``value`` selects the global physical-length policy."""
    if isinstance(value, GlobalLengthDiffusion):
        return True
    return isinstance(value, str) and value.strip().lower().replace("-", "_") == "global_length"


def resolve_diffusion_stabilization(
        stabilization: Any,
        diffusion: Any,
        space: DGSpace,
) -> Any:
    """Lower a built-in diffusion policy while preserving every explicit input."""
    if isinstance(stabilization, GlobalLengthDiffusion):
        return stabilization.resolve(diffusion, space)
    if is_global_length_diffusion(stabilization):
        return GlobalLengthDiffusion().resolve(diffusion, space)
    return stabilization


__all__ = [
    "GlobalLengthDiffusion",
    "automatic_domain_length",
    "compute_domain_length",
    "constant_isotropic_diffusivity",
    "geometric_diffusion_tau",
    "is_global_length_diffusion",
    "mesh_domain_measures",
    "resolve_diffusion_stabilization",
]
