"""hdgfem.mixed.adr_numpy."""

from __future__ import annotations

import numpy as np
from hdgfem.core.space import DGSpace, DGTraceSpace
from hdgfem.linalg.reduction import KnownDofReduction, eliminate_known_dofs
from hdgfem.mixed.local_numpy import (
    _local_solver_pre_mats,
    diffusion_inverse_mass_blocks,
)
from dataclasses import dataclass
from hdgfem.hdg import condensation as hdg
from hdgfem.mixed.adr_preparation import ADRPreparedData


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
