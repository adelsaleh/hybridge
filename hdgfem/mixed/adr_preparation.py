"""Stationary advection-diffusion-reaction HDG reference algebra."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from hdgfem.hdg import condensation as hdg
from hdgfem.hdg import matrices
import hdgfem.hdg.coefficients as hdg_coefficients
import hdgfem.hdg.stabilization as hdg_stabilization
from hdgfem.core.element_coefficients import ElementCoefficient
from hdgfem.runtime.threads import parallel_copy
from hdgfem.core.space import DGSpace, DGTraceSpace, VectorDGField
from hdgfem.mixed.local_numpy import (
    _local_solver_pre_mats,
    diffusion_element_boundary_mats,
    diffusion_trace_lift,
)
from hdgfem.mixed.stabilization import resolve_diffusion_stabilization
from hdgfem.hdg.coefficients import beta_values_on_volume, reaction_values_on_volume


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
    from hdgfem.mixed.coefficients import normal_diffusivity_on_faces
    constant = float(penalty_constant)
    if not np.isfinite(constant) or constant <= 0.0:
        raise ValueError("diffusion_penalty_constant must be finite and positive")
    kappa = normal_diffusivity_on_faces(diffusion, space, device=device)
    xp, mesh = np, space.mesh
    if device:
        from hdgfem.core.device import as_cupy_space
        from hdgfem.runtime.optional import require_cupy
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
        from hdgfem.runtime.optional import require_cupy
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
            from hdgfem.hdg.coefficients_device import face_samples_cupy
            tau = face_samples_cupy(stabilization, space, label="diffusion_stabilization",
                                    trace_space=trace_space, t=t)
        else:
            tau = hdg_coefficients._face_quadrature_values_from_scalar_input(
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
        from hdgfem.runtime.optional import require_cupy
        from hdgfem.hdg.coefficients_device import face_samples_cupy
        xp = require_cupy()
        values = face_samples_cupy(law, space, label="diffusion_stabilization",
                                  trace_space=trace_space, t=prepared.sample_time)
    else:
        if hasattr(law, '__cuda_array_interface__'):
            from hdgfem.runtime.optional import asnumpy
            law = asnumpy(law)
        values = hdg_coefficients._face_quadrature_values_from_scalar_input(
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
        from hdgfem.core.device import as_cupy_space
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
        return hdg_coefficients.advective_boundary_normal(beta, space, trace_space=trace_space)
    from hdgfem.hdg.coefficients import _prepare_beta_data

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
    tau_advection = hdg_stabilization.advection_trace_stabilization_values(
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
    from hdgfem.hdg.reference import _reference_derivative_matrices

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


__all__ = [
    "ADRPreparedData",
    "normalize_diffusion_stabilization",
    "prepare_adr_data",
    "recommended_diffusion_stabilization",
]
