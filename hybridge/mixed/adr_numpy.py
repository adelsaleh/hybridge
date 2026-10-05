"""ADR NumPy reference: the shared mixed local inverse and assembler with advection terms."""

from __future__ import annotations

import numpy as np
from hybridge.core.space import DGSpace, DGTraceSpace
from hybridge.linalg.reduction import KnownDofReduction, eliminate_known_dofs
from hybridge.mixed.local_numpy import (
    _local_solver_pre_mats,
    assemble_mixed_trace_system,
    mixed_local_inverse,
)
from dataclasses import dataclass
from hybridge.hdg import condensation as hdg
from hybridge.mixed.adr_preparation import ADRPreparedData


@dataclass(frozen=True)
class ADRNumpyAssembly:
    """NumPy reference local data and reduced trace system."""

    trace_system: hdg.TraceSystem
    reduction: KnownDofReduction
    local_solver: np.ndarray
    prepared: ADRPreparedData


def local_solvers_numpy(
        prepared: ADRPreparedData,
        space: DGSpace,
        *,
        diffusion=1.0,
) -> np.ndarray:
    """Build dense mixed ADR local inverses as the reference implementation."""
    if prepared.u_boundary_mass is None:
        raise ValueError("NumPy ADR requires dense_local_matrices=True during preparation")
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

    d0, d1, _zero, normal_x, normal_y, jacs_inv = _local_solver_pre_mats(0.0, 0.0, space)
    u_block = reaction_mass - advection + prepared.u_boundary_mass
    return mixed_local_inverse(u_block, d0, d1, normal_x, normal_y, jacs_inv, space, diffusion=diffusion)


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
    # Boundary rows are eliminated below, so their penalty value is irrelevant.
    full = assemble_mixed_trace_system(
        local_solver,
        prepared.element_boundary,
        prepared.trace_lift,
        prepared.interior_gamma_mass,
        source_block,
        boundary_condition,
        space,
        boundary_penalty=1.0,
        trace_space=trace_ref,
    )
    reduced, reduction = _reduce_all_dirichlet(full, space, trace_ref)
    return ADRNumpyAssembly(reduced, reduction, local_solver, prepared)
