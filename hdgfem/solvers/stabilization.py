"""Backend-neutral stabilization policies and geometric scale helpers."""

from __future__ import annotations

from hdgfem.runtime.precision import REAL_DTYPE

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from hdgfem.core.space import DGSpace, DGField


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
    domain_measure = 2.0 * float(np.sum(np.asarray(mesh.aff_jacs, dtype=REAL_DTYPE)))
    boundary_measure = 2.0 * float(
        np.sum(np.asarray(mesh.edge_jacs, dtype=REAL_DTYPE)[mesh.bnd_edges_inds])
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
    :func:`automatic_domain_length`. A positive scalar or constant isotropic
    tensor uses a scalar tau. Other
    elliptic tensors use the maximum sampled normal diffusivity per incidence.
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

    def resolve(self, diffusion: Any, space: DGSpace, *, device: bool = False) -> float | np.ndarray:
        """Resolve scalar or sidewise normal-diffusivity stabilization.

        ``device=True`` samples the sidewise normal diffusivity on the device
        (CuPy result); the scalar case is unchanged.
        """
        try:
            kappa = constant_isotropic_diffusivity(diffusion)
        except NotImplementedError:
            from hdgfem.assembly.diffusion_coefficients import normal_diffusivity_on_faces
            scale = geometric_diffusion_tau(1., self.resolved_domain_length(space), self.gamma_d)
            return scale * normal_diffusivity_on_faces(diffusion, space, device=device)
        return geometric_diffusion_tau(
            kappa,
            self.resolved_domain_length(space),
            self.gamma_d,
        )


def constant_isotropic_diffusivity(diffusion: Any) -> float:
    """Return scalar kappa from scalar or constant isotropic tensor data."""
    if np.isscalar(diffusion):
        return float(diffusion)
    if isinstance(diffusion, DGField):
        if diffusion.constant_value is not None:
            return float(diffusion.constant_value)
        raise NotImplementedError("diffusion is not constant isotropic")
    try:
        values = np.asarray(diffusion, dtype=REAL_DTYPE)
    except (TypeError, ValueError) as exc:
        raise NotImplementedError(
            "global-length stabilization requires constant isotropic diffusion"
        ) from exc
    if values.ndim == 0:
        return float(values)
    if values.shape == (3,):
        kappa_00, kappa_01, kappa_11 = (float(value) for value in values)
        matrix = np.array(
            ((kappa_00, kappa_01), (kappa_01, kappa_11)),
            dtype=REAL_DTYPE,
        )
    elif values.shape == (4,):
        matrix = values.reshape(2, 2)
    elif values.shape == (2, 2):
        matrix = values
    else:
        raise NotImplementedError(
            "global-length stabilization requires constant isotropic diffusion"
        )
    if (
        float(matrix[0, 1]) != 0.0
        or float(matrix[1, 0]) != 0.0
        or float(matrix[0, 0]) != float(matrix[1, 1])
    ):
        raise NotImplementedError(
            "diffusion is anisotropic; use incidence normal diffusivity"
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
        *,
        device: bool = False,
) -> Any:
    """Lower a built-in diffusion policy while preserving every explicit input."""
    if isinstance(stabilization, GlobalLengthDiffusion):
        return stabilization.resolve(diffusion, space, device=device)
    if is_global_length_diffusion(stabilization):
        return GlobalLengthDiffusion().resolve(diffusion, space, device=device)
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


def is_lax_friedrichs(stabilization):
    """Whether advection uses tau=2*abs(beta.n) instead of abs(beta.n).

    The extra penalty keeps outflow-side trace columns active for a
    discontinuous velocity. It changes the numerical flux, not Poisson tau.
    """
    return isinstance(stabilization, str) and stabilization == "lax-friedrichs"


@dataclass(frozen=True)
class ScaledUpwind:
    """Advection stabilization tau = factor * abs(beta.n).

    A plain scalar stabilization still denotes an absolute tau, not a factor.
    Factors below one are allowed but do not provide the usual upwind bound.
    """

    factor: float = 1.0

    def __post_init__(self):
        factor = float(self.factor)
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("upwind factor must be finite and positive")
        object.__setattr__(self, "factor", factor)


def upwind_factor(stabilization):
    """Return the upwind multiplier, or None for an explicit tau policy."""
    if stabilization is None or is_conflict_averaged_upwind(stabilization):
        return 1.0
    if isinstance(stabilization, ScaledUpwind):
        return stabilization.factor
    if is_lax_friedrichs(stabilization):
        return 2.0
    return None


__all__ += ["ScaledUpwind", "upwind_factor"]


def is_conflict_averaged_upwind(stabilization):
    """Whether to repair double-outflow face nodes before applying upwind."""
    return isinstance(stabilization, str) and stabilization == "conflict-averaged-upwind"


def conflict_averaged_normal_pair(a, b, *, xp=np):
    """Return effective outward velocities at orientation-matched face nodes.

    Only nonnegative pairs with positive sum are changed. No tolerance or
    clipping is used. This repairs the numerical face flux, not the volume
    velocity, and is neither an H(div) reconstruction nor an energy guarantee.
    """
    conflict = (a >= 0) & (b >= 0) & ((a + b) > 0)
    average = (a - b) * 0.5
    return xp.where(conflict, average, a), xp.where(conflict, -average, b)


def effective_advection_normal_flux(normal, mesh, stabilization, *, xp=np):
    """Apply a face policy using cached incidence and global edge orientation.

    NumPy and CuPy use the same gathering operation; device callers pass the
    cached device mesh. Boundary samples and the input array are unchanged.
    """
    if not is_conflict_averaged_upwind(stabilization):
        return normal
    slots = mesh.edge_side_indices[mesh.int_edges_inds]
    orientations = mesh.orientations.reshape(-1)
    values = normal.reshape(-1, normal.shape[-1])
    left, right = slots[:, 0], slots[:, 1]
    a = xp.where(orientations[left, None], values[left], values[left, ::-1])
    b = xp.where(orientations[right, None], values[right], values[right, ::-1])
    a, b = conflict_averaged_normal_pair(a, b, xp=xp)
    result = values.copy()
    result[left] = xp.where(orientations[left, None], a, a[:, ::-1])
    result[right] = xp.where(orientations[right, None], b, b[:, ::-1])
    return result.reshape(normal.shape)


def gauge_inactive_advection_trace_blocks(blocks, tau, mesh, *, xp=np):
    """Add identity once to existing side mass blocks of exactly inactive faces.

    Both incident sides must have identically zero effective tau. The zero
    physical lift already supplies zero RHS and zero row/column couplings.
    There is no clipping, rank regularization, or host/device synchronization.
    """
    elements, faces = mesh.interior_elements, mesh.interior_faces
    edges = mesh.loc2glob_edge[elements, faces]
    slots = mesh.edge_side_indices[edges]
    flat = tau.reshape(-1, tau.shape[-1])
    inactive = xp.all(flat[slots[:, 0]] == 0, axis=1) & xp.all(flat[slots[:, 1]] == 0, axis=1)
    canonical = slots[:, 0] == (3 * elements + faces)
    blocks += (inactive & canonical)[:, None, None] * xp.eye(blocks.shape[-1], dtype=blocks.dtype)
    return blocks


__all__ += ["is_conflict_averaged_upwind", "conflict_averaged_normal_pair",
            "effective_advection_normal_flux", "gauge_inactive_advection_trace_blocks"]
