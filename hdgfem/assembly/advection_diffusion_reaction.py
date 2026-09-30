"""Stationary advection-diffusion-reaction HDG reference algebra."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from hdgfem.assembly import hdg
from hdgfem.assembly import matrices_numpy as matrices
from hdgfem.core.element_coefficients import ElementCoefficient
from hdgfem.core.host_threads import parallel_copy
from hdgfem.core.space import DGSpace, DGTraceSpace, VectorDGField
from hdgfem.linalg.system import KnownDofReduction, eliminate_known_dofs
from hdgfem.solvers.diffusion_reaction import (
    _local_solver_pre_mats,
    diffusion_element_boundary_mats,
    diffusion_inverse_mass_blocks,
    diffusion_trace_lift,
)
from hdgfem.solvers.stabilization import resolve_diffusion_stabilization
from hdgfem.backends.numba import beta_values_on_volume, reaction_values_on_volume


@dataclass(frozen=True)
class ADRPreparedData:
    """Coefficient samples and face moments shared by all ADR backends.

    ``backends.adr_coefficients_cupy.prepare_adr_data_cupy`` fills the sample
    tables with CuPy arrays for raw-CUDA assembly; ``tau_diffusion`` and the
    reference tables stay on the host.
    """

    source_rhs: np.ndarray
    reaction_values: np.ndarray
    beta_values: np.ndarray
    beta_dot_normal: np.ndarray
    tau_advection: np.ndarray
    tau_diffusion: np.ndarray
    tau_total: np.ndarray
    gamma: np.ndarray
    u_boundary_mass: np.ndarray | None
    normal_mass_x: np.ndarray | None
    normal_mass_y: np.ndarray | None
    element_boundary: np.ndarray | None
    trace_lift: np.ndarray | None
    interior_gamma_mass: np.ndarray | None
    d0_reference: np.ndarray
    d1_reference: np.ndarray
    diffusion_stabilization_law: object = None
    face_quadrature: np.ndarray | None = None
    sample_time: float | None = None
    preparation_seconds: float = 0.0


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
        device: bool = False,
) -> np.ndarray:
    r"""Return :math:`C_\tau(p+1)^2\kappa_n/h_F` on element sides.

    Normal diffusivity is the maximum sampled value of n^T kappa n on
    each element-side incidence; it preserves discontinuous coefficients.  ``h_F`` is the element altitude normal to the
    side, namely ``2*det(J_K)/J_F`` in the mesh's reference scaling.
    ``device=True`` evaluates on the device and returns a CuPy array.
    """
    from hdgfem.assembly.diffusion_coefficients import normal_diffusivity_on_faces
    constant = float(penalty_constant)
    if not np.isfinite(constant) or constant <= 0.0:
        raise ValueError("diffusion_penalty_constant must be finite and positive")
    kappa = normal_diffusivity_on_faces(diffusion, space, device=device)
    xp, mesh = np, space.mesh
    if device:
        from hdgfem.backends.cupy import as_cupy_space, require_cupy
        xp, mesh = require_cupy(), as_cupy_space(space).mesh
    h_normal = 2.0 * mesh.aff_jacs[:, None] / mesh.jacs_el_fc
    return xp.ascontiguousarray(constant * (space.order + 1) ** 2 * kappa / h_normal)


def normalize_diffusion_stabilization(
        stabilization,
        space: DGSpace,
        *,
        diffusion=1.0,
        penalty_constant: float = 1.0,
        trace_space: DGTraceSpace | None = None,
        t=None,
        device: bool = False,
) -> np.ndarray:
    """Return positive incidence constants or spatial face samples.

    ``device=True`` evaluates every built-in policy, callable law and DG field
    on the device and returns a CuPy array (see ``backends.coefficients_cupy``).
    """
    stabilization = resolve_diffusion_stabilization(stabilization, diffusion, space, device=device)
    legacy_inverse_h = stabilization is None or (
        isinstance(stabilization, str)
        and stabilization.strip().lower().replace("_", "-")
        in {"auto", "recommended", "inverse-h"}
    )
    xp = np
    if device:
        from hdgfem.backends.cupy import require_cupy
        xp = require_cupy()
    if legacy_inverse_h:
        tau = recommended_diffusion_stabilization(
            space, diffusion, penalty_constant=penalty_constant, device=device
        )
    elif hasattr(stabilization, "__cuda_array_interface__"):
        # Global-length sidewise constants already resolved on the device.
        if stabilization.shape != (space.mesh.num_tri, 3):
            raise ValueError("device diffusion stabilization must have shape (num_elements, 3)")
        tau = stabilization
    else:
        if device:
            from hdgfem.backends.coefficients_cupy import face_samples_cupy
            tau = face_samples_cupy(stabilization, space, label="diffusion_stabilization",
                                    trace_space=trace_space, t=t)
        else:
            tau = matrices._face_quadrature_values_from_scalar_input(
                stabilization, space, "diffusion_stabilization", trace_space=trace_space, t=t)
        # Preserve the established public shape for incidence-constant inputs.
        if not callable(stabilization) and not hasattr(stabilization, "space"):
            if np.ndim(stabilization) < 3:
                tau = tau[:, :, 0]
    if bool(xp.any(~xp.isfinite(tau))) or bool(xp.any(tau <= 0.0)):
        raise ValueError("diffusion stabilization must be finite and strictly positive")
    return xp.ascontiguousarray(tau)


def diffusion_stabilization_on_trace(prepared, space, trace_space, *, device=False):
    """Resample the retained spatial law; never interpolate quadrature tables."""
    law = prepared.diffusion_stabilization_law
    if law is None:
        law = prepared.tau_diffusion
    if hasattr(law, 'ndim') and law.ndim == 3:
        if prepared.face_quadrature is None or not np.array_equal(
                prepared.face_quadrature, trace_space.quads):
            raise ValueError("diffusion stabilization quadrature table is incompatible with recovery quadrature; supply a spatial law")
    xp = np
    if device:
        from hdgfem.backends.cupy import require_cupy
        from hdgfem.backends.coefficients_cupy import face_samples_cupy
        xp = require_cupy()
        values = face_samples_cupy(law, space, label="diffusion_stabilization",
                                  trace_space=trace_space, t=prepared.sample_time)
    else:
        if hasattr(law, '__cuda_array_interface__'):
            from hdgfem.backends.cupy import asnumpy
            law = asnumpy(law)
        values = matrices._face_quadrature_values_from_scalar_input(
            law, space, "diffusion_stabilization", trace_space=trace_space, t=prepared.sample_time)
    if xp.any(~xp.isfinite(values)) or xp.any(values <= 0.):
        raise ValueError("diffusion stabilization must be finite and strictly positive")
    return values


def element_beta_normal(beta, space: DGSpace, trace_space: DGTraceSpace, *, xp=np, t=None):
    """Return ``beta.n`` of an element-local vector coefficient, shape ``(K, 3, nfq)``.

    Each element sees its own values on its faces, so ``beta.n`` may jump
    across a face.
    """
    if beta.components != 2:
        raise ValueError("element-local advection coefficient must have two components")
    face = beta.face_values(space, trace_space, xp=xp, t=t)
    if xp is np:
        normals = space.mesh.normals
    else:
        from hdgfem.backends.cupy import as_cupy_space
        normals = as_cupy_space(space).mesh.normals
    normal = face[..., 0] * normals[:, :, 0, None] + face[..., 1] * normals[:, :, 1, None]
    return xp.ascontiguousarray(normal)


def element_beta_samples(beta, space: DGSpace, trace_space: DGTraceSpace, *, xp=np, t=None):
    """Return ``(beta.n, beta)`` of an element-local vector coefficient.

    ``beta.n`` is :func:`element_beta_normal`; volume samples have shape
    ``(K, nq, 2)``. Nothing is projected into the solution space.
    """
    return (element_beta_normal(beta, space, trace_space, xp=xp, t=t),
            beta.volume_values(space, xp=xp, t=t))


def _normal_flux(beta, space: DGSpace, trace_space: DGTraceSpace, *, t=None) -> np.ndarray:
    """Evaluate beta dot the outward element normal on trace quadrature."""
    if isinstance(beta, ElementCoefficient):
        return element_beta_normal(beta, space, trace_space, t=t)
    if isinstance(beta, VectorDGField):
        return matrices.advective_boundary_normal(beta, space, trace_space=trace_space)
    from hdgfem.solvers.advection_reaction import _prepare_beta_data

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
        dense_local_matrices: bool = True,
        t=None,
        static: dict | None = None,
) -> ADRPreparedData:
    """Sample coefficients and assemble the common ADR face moment tables.

    ``static`` is an optional cross-call cache owned by a reusable solver for
    one space, trace space, diffusion and non-callable diffusion
    stabilization: it keeps the diffusion stabilization and the diffusion-only
    local blocks (normal masses, reference derivatives, diffusion boundary and
    lift blocks), which do not depend on the advection, reaction or source.
    """
    preparation_start = time.perf_counter()
    trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space

    def cached(name, build):
        """Return a static block, building and storing it on first use."""
        if static is None:
            return build()
        if name not in static:
            static[name] = build()
        return static[name]
    beta_dot_normal = _normal_flux(beta, space, trace_ref, t=t)
    tau_advection = matrices.advection_trace_stabilization_values(
        space,
        beta_dot_normal,
        advection_stabilization,
        trace_space=trace_ref,
    )
    def diffusion_tau():
        """The diffusion stabilization (time-dependent only for callable laws)."""
        return normalize_diffusion_stabilization(
            diffusion_stabilization,
            space,
            diffusion=diffusion,
            penalty_constant=diffusion_penalty_constant, trace_space=trace_ref, t=t,
        )
    tau_diffusion = diffusion_tau() if callable(diffusion_stabilization) else cached("tau_diffusion", diffusion_tau)
    tau_samples = tau_diffusion[:, :, None] if tau_diffusion.ndim == 2 else tau_diffusion
    tau_total = np.ascontiguousarray(tau_advection + tau_samples)
    gamma = np.ascontiguousarray(tau_total - beta_dot_normal)

    # Element-local coefficients resolve to quadrature values at this level.
    if isinstance(source, ElementCoefficient):
        source_rhs = hdg.source_moments_from_values(source.volume_values(space, t=t), space)
    else:
        source_rhs = hdg.source_moments(source, space)
    if isinstance(reaction, ElementCoefficient):
        reaction = reaction.volume_values(space, t=t)
    reaction_samples = reaction_values_on_volume(reaction, space)
    if isinstance(beta, ElementCoefficient):
        beta_samples = beta.volume_values(space, t=t)
    else:
        beta_field = beta if isinstance(beta, VectorDGField) else hdg.as_vector_field(beta, space)
        beta_samples = beta_values_on_volume(beta_field, None, space)

    if dense_local_matrices:
        u_boundary_mass = matrices.boundary_mass_from_trace_stabilization(
            space, tau_total, trace_space=trace_ref
        )
        normal_mass_x, normal_mass_y = cached(
            "normal_mass", lambda: _local_solver_pre_mats(0.0, 0.0, space)[3:5])
        diffusion_boundary = cached(
            "diffusion_boundary", lambda: diffusion_element_boundary_mats(0.0, space, trace_space=trace_ref))
        element_boundary = parallel_copy(diffusion_boundary)
        element_boundary[:, :space.el_dof] = matrices.element_boundary_mats_from_trace_weight(
            space, gamma, trace_space=trace_ref
        )

        trace_lift = parallel_copy(cached("diffusion_trace_lift",
                                          lambda: diffusion_trace_lift(0.0, space, trace_space=trace_ref)))
        trace_lift[..., :space.el_dof] = matrices.advection_trace_lift_from_stabilization(
            space, tau_total, trace_space=trace_ref
        )
        interior_gamma_mass = matrices.advection_interior_trace_mass_blocks_from_weight(
            space, gamma, trace_space=trace_ref
        )
    else:
        u_boundary_mass = normal_mass_x = normal_mass_y = None
        element_boundary = trace_lift = interior_gamma_mass = None
    from hdgfem.solvers.diffusion_reaction import _reference_derivative_matrices

    d0_reference, d1_reference = cached("reference_derivatives", lambda: _reference_derivative_matrices(space))
    return ADRPreparedData(
        source_rhs=np.ascontiguousarray(source_rhs),
        reaction_values=np.ascontiguousarray(reaction_samples),
        beta_values=np.ascontiguousarray(beta_samples),
        beta_dot_normal=np.ascontiguousarray(beta_dot_normal),
        tau_advection=np.ascontiguousarray(tau_advection),
        tau_diffusion=np.ascontiguousarray(tau_diffusion),
        tau_total=tau_total,
        gamma=gamma,
        u_boundary_mass=None if u_boundary_mass is None else np.ascontiguousarray(u_boundary_mass),
        normal_mass_x=None if normal_mass_x is None else np.ascontiguousarray(normal_mass_x),
        normal_mass_y=None if normal_mass_y is None else np.ascontiguousarray(normal_mass_y),
        element_boundary=None if element_boundary is None else np.ascontiguousarray(element_boundary),
        trace_lift=None if trace_lift is None else np.ascontiguousarray(trace_lift),
        interior_gamma_mass=None if interior_gamma_mass is None else np.ascontiguousarray(interior_gamma_mass),
        d0_reference=np.ascontiguousarray(d0_reference),
        d1_reference=np.ascontiguousarray(d1_reference),
        diffusion_stabilization_law=(diffusion_stabilization
                                    if callable(diffusion_stabilization) or hasattr(diffusion_stabilization, "space")
                                    else tau_diffusion),
        face_quadrature=trace_ref.quads.copy(), sample_time=t,
        preparation_seconds=time.perf_counter() - preparation_start,
    )


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


__all__ = [
    "ADRPreparedData",
    "ADRNumpyAssembly",
    "assemble_numpy",
    "local_solvers_numpy",
    "normalize_diffusion_stabilization",
    "prepare_adr_data",
    "recommended_diffusion_stabilization",
]
