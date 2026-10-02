"""Backend-neutral stabilization policies and geometric scale helpers."""

from __future__ import annotations


from dataclasses import dataclass

import numpy as np

from hdgfem.core.space import DGSpace

from hdgfem.core.space import DGTraceSpace
from hdgfem.hdg.coefficients import _face_quadrature_values_from_scalar_input
from hdgfem.hdg.trace_maps import _trace_ref
from hdgfem.hdg.coefficients import _require_normal_flux


__all__ = [
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


def resolve_transport_stabilization(stabilization, beta):
    """Select corrected upwind for DG velocities without sampling or transfers.

    DG fields and coefficient arrays have discontinuous element semantics.
    Two analytic callables keep sidewise upwind. Explicit policies override
    the default; ``"upwind"`` requests the original unit-factor scheme.
    """
    from hdgfem.hdg.coefficients import _is_callable_beta

    if isinstance(stabilization, str) and stabilization == "upwind":
        return ScaledUpwind(1.)
    if stabilization is None and beta is not None and not _is_callable_beta(beta):
        return "conflict-averaged-upwind"
    return stabilization


def upwind_factor(stabilization):
    """Return the upwind multiplier, or None for an explicit tau policy."""
    if (stabilization is None or is_conflict_averaged_upwind(stabilization)
            or (isinstance(stabilization, str) and stabilization == "upwind")):
        return 1.0
    if isinstance(stabilization, ScaledUpwind):
        return stabilization.factor
    if is_lax_friedrichs(stabilization):
        return 2.0
    return None


__all__ += ["ScaledUpwind", "upwind_factor", "resolve_transport_stabilization"]


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


def advection_trace_stabilization_values(
        test_space: DGSpace,
        beta_dot_normal: np.ndarray,
        stabilization=None,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    r"""Return side-quadrature advection stabilization values.

    ``stabilization=None`` selects the upwind choice
    :math:`\tau_{K,F}=|\beta_h\cdot n_K|`.  Explicit scalar, callable, DG field,
    coefficient-array, or already evaluated face data specify absolute tau.
    ``ScaledUpwind(factor)`` selects ``factor*abs(beta_h.n)``;
    ``"lax-friedrichs"`` is the factor-two alias. Arrays have shape
    ``(num_elements, 3, num_face_quads)`` without averaging across an interior
    edge for these policies. ``"conflict-averaged-upwind"`` instead uses
    effective velocities from the shared interior double-outflow repair.
    """
    trace_ref = _trace_ref(test_space, trace_space)
    beta_dot_normal = _require_normal_flux(beta_dot_normal, test_space, trace_space=trace_ref)
    beta_dot_normal = effective_advection_normal_flux(beta_dot_normal, test_space.mesh, stabilization)
    factor = upwind_factor(stabilization)
    if factor is not None:
        return np.ascontiguousarray(factor * np.abs(beta_dot_normal))
    return _face_quadrature_values_from_scalar_input(
        stabilization,
        test_space,
        "advection_stabilization",
        trace_space=trace_ref,
    )


def advection_trace_weights_from_normal_flux(
        test_space: DGSpace,
        beta_dot_normal: np.ndarray,
        stabilization=None,
        *,
        trace_space: DGTraceSpace | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the side weights ``tau`` and ``gamma=tau-beta_h\cdot n``.

    ``tau`` multiplies the element-side value ``u_h`` in the trace conservation
    equation, while ``gamma`` multiplies the trace unknown ``\widehat u_h``.
    """
    trace_ref = _trace_ref(test_space, trace_space)
    beta_dot_normal = _require_normal_flux(beta_dot_normal, test_space, trace_space=trace_ref)
    tau = advection_trace_stabilization_values(
        test_space,
        beta_dot_normal,
        stabilization,
        trace_space=trace_ref,
    )
    gamma = tau - effective_advection_normal_flux(beta_dot_normal, test_space.mesh, stabilization)
    return np.ascontiguousarray(tau), np.ascontiguousarray(gamma)
