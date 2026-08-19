"""Stationary advection-diffusion-reaction HDG reference algebra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import hdg
from . import matrices_numpy as matrices
from ..core.space import DGSpace, DGTraceSpace, VectorDGField
from ..linalg.system import KnownDofReduction, eliminate_known_dofs
from ..solvers.diffusion_reaction import (
    _local_solver_pre_mats,
    diffusion_element_boundary_mats,
    diffusion_inverse_mass_blocks,
    diffusion_trace_lift,
)
from ..solvers.stabilization import resolve_diffusion_stabilization
from ..backends.numba import beta_values_on_volume, reaction_values_on_volume


@dataclass(frozen=True)
class ADRPreparedData:
    """Coefficient samples and face moments shared by all ADR backends."""

    source_rhs: np.ndarray
    reaction_values: np.ndarray
    beta_values: np.ndarray
    beta_dot_normal: np.ndarray
    tau_advection: np.ndarray
    tau_diffusion: np.ndarray
    tau_total: np.ndarray
    gamma: np.ndarray
    u_boundary_mass: np.ndarray
    normal_mass_x: np.ndarray
    normal_mass_y: np.ndarray
    element_boundary: np.ndarray
    trace_lift: np.ndarray
    interior_gamma_mass: np.ndarray
    d0_reference: np.ndarray
    d1_reference: np.ndarray


@dataclass(frozen=True)
class ADRNumpyAssembly:
    """NumPy reference local data and reduced trace system."""

    trace_system: hdg.TraceSystem
    reduction: KnownDofReduction
    local_solver: np.ndarray
    prepared: ADRPreparedData


def recommended_diffusion_stabilization(
        space: DGSpace,
        diffusion=1.0,
        *,
        penalty_constant: float = 1.0,
) -> np.ndarray:
    r"""Return :math:`C_\tau(p+1)^2\kappa_n/h_F` on element sides.

    The first implementation supports the identity or a positive scalar
    diffusion coefficient.  ``h_F`` is the element altitude normal to the
    side, namely ``2*det(J_K)/J_F`` in the mesh's reference scaling.
    """
    if not np.isscalar(diffusion):
        raise NotImplementedError(
            "automatic ADR diffusion stabilization currently requires scalar diffusion; "
            "provide diffusion_stabilization explicitly for tensor diffusion"
        )
    kappa = float(diffusion)
    if not np.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("diffusion must be finite and positive")
    constant = float(penalty_constant)
    if not np.isfinite(constant) or constant <= 0.0:
        raise ValueError("diffusion_penalty_constant must be finite and positive")
    mesh = space.mesh
    h_normal = 2.0 * mesh.aff_jacs[:, None] / mesh.jacs_el_fc
    return np.ascontiguousarray(constant * (space.order + 1) ** 2 * kappa / h_normal)


def normalize_diffusion_stabilization(
        stabilization,
        space: DGSpace,
        *,
        diffusion=1.0,
        penalty_constant: float = 1.0,
) -> np.ndarray:
    """Return positive sidewise diffusion stabilization constants."""
    stabilization = resolve_diffusion_stabilization(stabilization, diffusion, space)
    legacy_inverse_h = stabilization is None or (
        isinstance(stabilization, str)
        and stabilization.strip().lower().replace("_", "-")
        in {"auto", "recommended", "inverse-h"}
    )
    if legacy_inverse_h:
        tau = recommended_diffusion_stabilization(
            space, diffusion, penalty_constant=penalty_constant
        )
    elif np.isscalar(stabilization):
        tau = np.full((space.mesh.num_tri, 3), float(stabilization), dtype=np.float64)
    else:
        tau = np.asarray(stabilization, dtype=np.float64)
        if tau.shape == (space.mesh.num_tri,):
            tau = np.broadcast_to(tau[:, None], (space.mesh.num_tri, 3)).copy()
        elif tau.shape != (space.mesh.num_tri, 3):
            raise ValueError(
                "diffusion_stabilization must be scalar, 'global_length', "
                "'inverse-h', or have shape "
                f"({space.mesh.num_tri}, 3); got {tau.shape}"
            )
    if np.any(~np.isfinite(tau)) or np.any(tau <= 0.0):
        raise ValueError("diffusion stabilization must be finite and strictly positive")
    return np.ascontiguousarray(tau)


def _normal_flux(beta, space: DGSpace, trace_space: DGTraceSpace) -> np.ndarray:
    """Evaluate beta dot the outward element normal on trace quadrature."""
    if isinstance(beta, VectorDGField):
        return matrices.advective_boundary_normal(beta, space, trace_space=trace_space)
    from ..solvers.advection_reaction import _prepare_beta_data

    _field, values, _callables = _prepare_beta_data(beta, space, trace_space=trace_space)
    return np.ascontiguousarray(values)


def prepare_adr_data(
        source,
        reaction,
        beta,
        space: DGSpace,
        *,
        diffusion=1.0,
        advection_stabilization=None,
        diffusion_stabilization="global_length",
        diffusion_penalty_constant: float = 1.0,
        trace_space: DGTraceSpace | None = None,
) -> ADRPreparedData:
    """Sample coefficients and assemble the common ADR face moment tables."""
    trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
    beta_dot_normal = _normal_flux(beta, space, trace_ref)
    tau_advection = matrices.advection_trace_stabilization_values(
        space,
        beta_dot_normal,
        advection_stabilization,
        trace_space=trace_ref,
    )
    tau_diffusion = normalize_diffusion_stabilization(
        diffusion_stabilization,
        space,
        diffusion=diffusion,
        penalty_constant=diffusion_penalty_constant,
    )
    tau_total = np.ascontiguousarray(tau_advection + tau_diffusion[:, :, None])
    gamma = np.ascontiguousarray(tau_total - beta_dot_normal)

    source_rhs = hdg.source_moments(source, space)
    reaction_samples = reaction_values_on_volume(reaction, space)
    beta_field = beta if isinstance(beta, VectorDGField) else hdg.as_vector_field(beta, space)
    beta_samples = beta_values_on_volume(beta_field, None, space)

    u_boundary_mass = matrices.boundary_mass_from_trace_stabilization(
        space, tau_total, trace_space=trace_ref
    )
    _d0, _d1, _zero, normal_mass_x, normal_mass_y, _jinv = _local_solver_pre_mats(
        0.0, 0.0, space
    )
    diffusion_boundary = diffusion_element_boundary_mats(
        0.0, space, trace_space=trace_ref
    )
    element_boundary = diffusion_boundary.copy()
    element_boundary[:, :space.el_dof] = matrices.element_boundary_mats_from_trace_weight(
        space, gamma, trace_space=trace_ref
    )

    trace_lift = diffusion_trace_lift(0.0, space, trace_space=trace_ref)
    trace_lift[..., :space.el_dof] = matrices.advection_trace_lift_from_stabilization(
        space, tau_total, trace_space=trace_ref
    )
    interior_gamma_mass = matrices.advection_interior_trace_mass_blocks_from_weight(
        space, gamma, trace_space=trace_ref
    )
    from ..solvers.diffusion_reaction import _reference_derivative_matrices

    d0_reference, d1_reference = _reference_derivative_matrices(space)
    return ADRPreparedData(
        source_rhs=np.ascontiguousarray(source_rhs),
        reaction_values=np.ascontiguousarray(reaction_samples),
        beta_values=np.ascontiguousarray(beta_samples),
        beta_dot_normal=np.ascontiguousarray(beta_dot_normal),
        tau_advection=np.ascontiguousarray(tau_advection),
        tau_diffusion=np.ascontiguousarray(tau_diffusion),
        tau_total=tau_total,
        gamma=gamma,
        u_boundary_mass=np.ascontiguousarray(u_boundary_mass),
        normal_mass_x=np.ascontiguousarray(normal_mass_x),
        normal_mass_y=np.ascontiguousarray(normal_mass_y),
        element_boundary=np.ascontiguousarray(element_boundary),
        trace_lift=np.ascontiguousarray(trace_lift),
        interior_gamma_mass=np.ascontiguousarray(interior_gamma_mass),
        d0_reference=np.ascontiguousarray(d0_reference),
        d1_reference=np.ascontiguousarray(d1_reference),
    )


def local_solvers_numpy(
        prepared: ADRPreparedData,
        space: DGSpace,
        *,
        diffusion=1.0,
) -> np.ndarray:
    """Build dense mixed ADR local inverses as the reference implementation."""
    q = space.quad_data
    nel = space.el_dof
    jac = space.mesh.aff_jacs
    inv_t = space.mesh.inv_aff_mats_t
    beta = prepared.beta_values
    reaction = prepared.reaction_values
    basis = q.bas_of_quads
    gradients = q.dbas_of_quads
    weights = q.Krf_w
    grad_x = (
        inv_t[:, 0, 0, None, None] * gradients[None, 0]
        + inv_t[:, 0, 1, None, None] * gradients[None, 1]
    )
    grad_y = (
        inv_t[:, 1, 0, None, None] * gradients[None, 0]
        + inv_t[:, 1, 1, None, None] * gradients[None, 1]
    )
    reaction_mass = jac[:, None, None] * np.einsum(
        "Kq,iq,jq,q->Kij", reaction, basis, basis, weights, optimize=True
    )
    advection = jac[:, None, None] * np.einsum(
        "Kqd,Kdiq,jq,q->Kij",
        beta,
        np.stack((grad_x, grad_y), axis=1),
        basis,
        weights,
        optimize=True,
    )

    d0, d1, _zero, normal_x, normal_y, _jinv = _local_solver_pre_mats(0.0, 0.0, space)
    g00, g01, g10, g11 = diffusion_inverse_mass_blocks(diffusion, space)
    local = np.zeros((space.mesh.num_tri, 3 * nel, 3 * nel), dtype=np.float64)
    blocks = local.reshape(space.mesh.num_tri, 3, nel, 3, nel)
    blocks[:, 0, :, 0, :] = reaction_mass - advection + prepared.u_boundary_mass
    blocks[:, 0, :, 1, :] = normal_x - d0
    blocks[:, 0, :, 2, :] = normal_y - d1
    blocks[:, 1, :, 0, :] = d0
    blocks[:, 1, :, 1, :] = -g00
    blocks[:, 1, :, 2, :] = -g01
    blocks[:, 2, :, 0, :] = d1
    blocks[:, 2, :, 1, :] = -g10
    blocks[:, 2, :, 2, :] = -g11
    return np.ascontiguousarray(np.linalg.inv(local))


def _reduce_all_dirichlet(
        full: hdg.TraceSystem,
        space: DGSpace,
        trace_space: DGTraceSpace,
) -> tuple[hdg.TraceSystem, KnownDofReduction]:
    """Eliminate every exterior trace degree of freedom."""
    ntr = trace_space.edg_dof
    known = np.zeros(space.mesh.num_edg * ntr, dtype=bool)
    for edge in space.mesh.bnd_edges_inds:
        known[edge * ntr:(edge + 1) * ntr] = True
    known_values = np.asarray(full.boundary_trace, dtype=np.float64).reshape(-1)
    reduction = eliminate_known_dofs(
        full.rows, full.cols, full.data, full.rhs, known, known_values
    )
    reduced = hdg.TraceSystem(
        rows=reduction.rows,
        cols=reduction.cols,
        data=reduction.data,
        rhs=reduction.rhs,
        boundary_trace=full.boundary_trace,
    )
    return reduced, reduction


def assemble_numpy(
        prepared: ADRPreparedData,
        boundary_condition,
        space: DGSpace,
        *,
        diffusion=1.0,
        trace_space: DGTraceSpace | None = None,
) -> ADRNumpyAssembly:
    """Assemble the boundary-eliminated NumPy ADR reference system."""
    trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
    local_solver = local_solvers_numpy(prepared, space, diffusion=diffusion)
    source_block = np.zeros((space.mesh.num_tri, 3 * space.el_dof), dtype=np.float64)
    source_block[:, :space.el_dof] = prepared.source_rhs
    trace_blocks = hdg.element_to_trace_matrix_from_lift(
        prepared.trace_lift,
        local_solver,
        prepared.element_boundary,
        space,
        trace_space=trace_ref,
    )
    rows, cols = hdg.trace_matrix_indices(
        space, interior_mass_mode="face", trace_space=trace_ref
    )
    data = hdg.trace_matrix_data(
        trace_blocks,
        space,
        1.0,
        interior_mass_mode="face",
        interior_mass_blocks=prepared.interior_gamma_mass,
        trace_space=trace_ref,
    )
    rhs, boundary_trace = hdg.trace_rhs_from_lift(
        prepared.trace_lift,
        source_block,
        local_solver,
        boundary_condition,
        space,
        1.0,
        trace_space=trace_ref,
    )
    full = hdg.TraceSystem(rows=rows, cols=cols, data=data, rhs=rhs, boundary_trace=boundary_trace)
    reduced, reduction = _reduce_all_dirichlet(full, space, trace_ref)
    return ADRNumpyAssembly(reduced, reduction, local_solver, prepared)


__all__ = [
    "ADRPreparedData",
    "ADRNumpyAssembly",
    "assemble_numpy",
    "local_solvers_numpy",
    "normalize_diffusion_stabilization",
    "prepare_adr_data",
    "recommended_diffusion_stabilization",
]
