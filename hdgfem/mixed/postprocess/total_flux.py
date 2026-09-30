"""hdgfem.mixed.postprocess.total_flux."""

from __future__ import annotations

import numpy as np
from hdgfem.mixed.adr_preparation import ADRPreparedData
from typing import Any
from hdgfem.core.space import DGField, DGSpace, VectorDGField
from hdgfem.core.element_coefficients import ElementCoefficient
from hdgfem.mixed.postprocess.flux import (
    _build_hdg_postprocess_cache,
    _normalize_flux_postprocess_space,
    _postprocess_rt_flux_from_samples,
)
from hdgfem.hdg import condensation as hdg


def _beta_field(beta, space: DGSpace) -> VectorDGField:
    """Normalize beta to a two-component DG field for postprocessing."""
    if isinstance(beta, VectorDGField):
        return beta
    return hdg.as_vector_field(beta, space)


def _adr_postprocess_samples(
        beta,
        prepared: ADRPreparedData,
        space: DGSpace,
        post_space: DGSpace,
        advection_stabilization,
        *, xp=np,
) -> tuple[Any, Any, Any]:
    """Sample beta and total stabilization on degree-p+1 quadrature rules."""
    if xp is np:
        coefficients = lambda field: field.coeffs
    else:
        from hdgfem.core.device import as_cupy_coefficients, as_cupy_space
        coefficients = lambda field: as_cupy_coefficients(field, as_cupy_space(field.space))
    qpost = post_space.quad_data
    nqf = qpost.weights_JGL.size
    face_points = qpost.pts_fc.reshape(-1, 2)
    if isinstance(beta, ElementCoefficient):
        # Element-local beta is evaluated directly on the recovery quadrature.
        beta_volume = beta.values_at_ref(qpost.Krf_quads, xp=xp, t=prepared.sample_time)
        beta_face = beta.face_values_at_ref(qpost.pts_fc, xp=xp, t=prepared.sample_time)
    else:
        beta_h = _beta_field(beta, space)
        beta_volume = xp.empty((space.mesh.num_tri, qpost.Krf_w.size, 2), dtype=xp.float64)
        beta_face = xp.empty((space.mesh.num_tri, 3, nqf, 2), dtype=xp.float64)
        for component, field in enumerate(beta_h.components):
            volume_basis = xp.asarray(field.space.basis_at(qpost.Krf_quads))
            face_basis = xp.asarray(field.space.basis_at(face_points)).reshape(
                nqf, 3, field.space.el_dof
            ).transpose(1, 2, 0)
            beta_volume[..., component] = coefficients(field) @ volume_basis.T
            beta_face[..., component] = xp.einsum(
                "Ki,fiq->Kfq", coefficients(field), face_basis, optimize=True
            )

    normals = xp.asarray(space.mesh.normals)
    beta_n = (
        beta_face[..., 0] * normals[..., 0, None]
        + beta_face[..., 1] * normals[..., 1, None]
    )
    # Upwind-family policies (None, ScaledUpwind, "lax-friedrichs",
    # "conflict-averaged-upwind") use the same rule as ADR assembly:
    # factor * |beta.n|, with the conflict-averaged interior repair.
    from hdgfem.hdg.stabilization import effective_advection_normal_flux, upwind_factor

    factor = upwind_factor(advection_stabilization)
    if factor is not None:
        if xp is np:
            mesh = space.mesh
        else:
            from hdgfem.core.device import as_cupy_space
            mesh = as_cupy_space(space).mesh
        tau_adv = factor * xp.abs(
            effective_advection_normal_flux(beta_n, mesh, advection_stabilization, xp=xp)
        )
    elif np.isscalar(advection_stabilization) and not isinstance(advection_stabilization, str):
        tau_adv = xp.full_like(beta_n, float(advection_stabilization))
    elif isinstance(advection_stabilization, DGField):
        tau_basis = xp.asarray(advection_stabilization.space.basis_at(face_points)).reshape(
            nqf, 3, advection_stabilization.space.el_dof
        ).transpose(1, 2, 0)
        tau_adv = xp.einsum(
            "Ki,fiq->Kfq", coefficients(advection_stabilization), tau_basis, optimize=True
        )
    elif callable(advection_stabilization):
        mapped = xp.asarray(space.mesh.map_reference_points(face_points))
        raw = xp.asarray(
            advection_stabilization(mapped[..., 0], mapped[..., 1]), dtype=xp.float64
        )
        if raw.ndim == 0:
            tau_adv = xp.broadcast_to(raw, beta_n.shape)
        else:
            tau_adv = xp.broadcast_to(raw, (space.mesh.num_tri, 3 * nqf)).reshape(
                space.mesh.num_tri, nqf, 3
            ).transpose(0, 2, 1).copy()
    else:
        raw = xp.asarray(advection_stabilization, dtype=xp.float64)
        if raw.shape == space.shape:
            tau_basis = xp.asarray(space.basis_at(face_points)).reshape(
                nqf, 3, space.el_dof
            ).transpose(1, 2, 0)
            tau_adv = xp.einsum("Ki,fiq->Kfq", raw, tau_basis, optimize=True)
        elif raw.shape == (space.mesh.num_tri, 3):
            tau_adv = xp.broadcast_to(raw[:, :, None], beta_n.shape)
        elif raw.shape == beta_n.shape:
            tau_adv = raw
        else:
            raise ValueError(
                "ADR postprocessing needs advection stabilization as None, scalar, "
                "callable, DGField, DG coefficients, per-face constants, or values "
                "on its degree-p+1 face quadrature"
            )
    from hdgfem.mixed.adr_preparation import diffusion_stabilization_on_trace
    tau_diff = diffusion_stabilization_on_trace(
        prepared, space, post_space.trace_space("bernstein"), device=xp is not np)
    tau_total = tau_adv + xp.asarray(tau_diff)
    if xp.any(~xp.isfinite(tau_total)):
        raise ValueError("ADR postprocessing stabilization must be finite")
    return (
        xp.ascontiguousarray(beta_volume),
        xp.ascontiguousarray(beta_face),
        xp.ascontiguousarray(tau_total),
    )


def _project_total_flux(
        local_unknowns: np.ndarray,
        prepared: ADRPreparedData,
        space: DGSpace,
) -> VectorDGField:
    """Project q_h plus beta_h u_h into the degree-p vector DG space."""
    xp = np
    if hasattr(local_unknowns, "__cuda_array_interface__"):
        from hdgfem.runtime.optional import require_cupy
        from hdgfem.core.device import field_from_cupy_coefficients
        xp = require_cupy()
    blocks = local_unknowns.reshape(space.mesh.num_tri, 3, space.el_dof)
    basis = xp.asarray(space.quad_data.bas_of_quads)
    beta_values = xp.asarray(prepared.beta_values)
    weighted_phi = xp.asarray(space.quad_data.weighted_phi)
    mass_inverse = xp.asarray(space.quad_data.MKrf_inv)
    u = blocks[:, 0] @ basis
    qx = blocks[:, 1] @ basis
    qy = blocks[:, 2] @ basis
    values_x = qx + beta_values[..., 0] * u
    values_y = qy + beta_values[..., 1] * u
    coeffs_x = (values_x @ weighted_phi) @ mass_inverse
    coeffs_y = (values_y @ weighted_phi) @ mass_inverse
    if xp is not np:
        return VectorDGField(tuple(field_from_cupy_coefficients(space, c, name="q_h_plus_beta_u_h")
                                   for c in (coeffs_x, coeffs_y)), name="q_h_plus_beta_u_h")
    return (space * space).field((coeffs_x, coeffs_y), name="q_h_plus_beta_u_h")


def _adr_recovery_cache(space, trace_ref, *, want_flux=False):
    """Reuse geometry/reference recovery tables, never PDE coefficient factors."""
    cache = _build_hdg_postprocess_cache(
        space, trace_ref, want_primal=False, want_flux=want_flux,
        cache=getattr(space, '_adr_recovery_cache', None))
    space._adr_recovery_cache = cache
    return cache


def _postprocess_total_flux(
        local_unknowns: np.ndarray,
        trace: np.ndarray,
        beta,
        prepared: ADRPreparedData,
        space: DGSpace,
        trace_ref,
        advection_stabilization,
        flux_postprocess_space="l2_closest",
        postprocessing_backend="numba",
) -> VectorDGField:
    """Recover the total flux by full-space or RT_p normal-moment matching."""
    from hdgfem.mixed.adr_numba_kernels import solve_adr_total_flux_postprocess_kernel
    from hdgfem.mixed.postprocess.flux import _trace_basis_at

    xp = np
    if postprocessing_backend == "cupy":
        from hdgfem.runtime.optional import require_cupy
        xp = require_cupy()
    local_unknowns = xp.asarray(local_unknowns)
    flux_space = _normalize_flux_postprocess_space(flux_postprocess_space)
    cache = _adr_recovery_cache(
        space,
        trace_ref,
        want_flux=flux_space == "l2_closest" and xp is np,
    )
    if flux_space == "l2_closest" and xp is np and (
        cache.flux_ainv_constraint_t is None or cache.flux_schur_lu is None
    ):
        raise RuntimeError("ADR total-flux postprocess factorization is unavailable")
    post = cache.post_space
    qpost = post.quad_data
    nqf = qpost.weights_JGL.size
    face_points = qpost.pts_fc.reshape(-1, 2)
    base_face = xp.asarray(space.basis_at(face_points)).reshape(
        nqf, 3, space.el_dof
    ).transpose(1, 2, 0)
    post_face = xp.asarray(qpost.bas_of_bd_quads)
    trace_basis = xp.asarray(_trace_basis_at(trace_ref, qpost.quads_JGL))
    if xp is np:
        local_trace = trace_ref.element_coefficients(trace).reshape(
            space.mesh.num_tri, 3, trace_ref.edg_dof)
    else:
        from hdgfem.hdg.condensation_device import element_traces_cupy
        local_trace = element_traces_cupy(trace, space, trace_space=trace_ref).reshape(
            space.mesh.num_tri, 3, trace_ref.edg_dof)
    blocks = local_unknowns.reshape(space.mesh.num_tri, 3, space.el_dof)
    u_face = xp.einsum("Ki,fiq->Kfq", blocks[:, 0], base_face, optimize=True)
    qx_face = xp.einsum("Ki,fiq->Kfq", blocks[:, 1], base_face, optimize=True)
    qy_face = xp.einsum("Ki,fiq->Kfq", blocks[:, 2], base_face, optimize=True)
    hat_face = xp.einsum("Kfa,aq->Kfq", local_trace, trace_basis, optimize=True)

    beta_volume, beta_face, tau_total = _adr_postprocess_samples(
        beta, prepared, space, post, advection_stabilization, xp=xp
    )
    base_volume = xp.asarray(space.basis_at(qpost.Krf_quads))
    u_volume = blocks[:, 0] @ base_volume.T
    qx_volume = blocks[:, 1] @ base_volume.T
    qy_volume = blocks[:, 2] @ base_volume.T
    total_volume_values = xp.ascontiguousarray(
        xp.stack(
            (
                qx_volume + beta_volume[..., 0] * u_volume,
                qy_volume + beta_volume[..., 1] * u_volume,
            ),
            axis=0,
        )
    )
    normals = xp.asarray(space.mesh.normals)
    beta_n = (
        beta_face[..., 0] * normals[..., 0, None]
        + beta_face[..., 1] * normals[..., 1, None]
    )
    numerical = (
        normals[..., 0, None] * qx_face
        + normals[..., 1, None] * qy_face
        + beta_n * hat_face
        + tau_total * (u_face - hat_face)
    )

    if flux_space == "RT_projection":
        return _postprocess_rt_flux_from_samples(
            total_volume_values,
            numerical,
            space,
            post,
            backend=postprocessing_backend,
            name="total_flux_h_star_rt_p",
            materialize_host=postprocessing_backend != "cupy",
        )

    if xp is not np:
        from hdgfem.mixed.postprocess.flux_cupy import postprocess_total_flux_l2_cupy
        return postprocess_total_flux_l2_cupy(total_volume_values, numerical, space, cache)

    base_weighted = qpost.Krf_w[:, None] * base_volume
    base_coeffs = np.empty(
        (2, space.mesh.num_tri, space.el_dof), dtype=np.float64
    )
    base_coeffs[0] = (
        total_volume_values[0] @ base_weighted @ space.quad_data.MKrf_inv
    )
    base_coeffs[1] = (
        total_volume_values[1] @ base_weighted @ space.quad_data.MKrf_inv
    )

    q0x = base_coeffs[0] @ cache.base_to_post_mass.T @ qpost.MKrf_inv
    q0y = base_coeffs[1] @ cache.base_to_post_mass.T @ qpost.MKrf_inv
    q0x_face = xp.einsum("Ki,fiq->Kfq", q0x, post_face, optimize=True)
    q0y_face = xp.einsum("Ki,fiq->Kfq", q0y, post_face, optimize=True)
    current = normals[..., 0, None] * q0x_face + normals[..., 1, None] * q0y_face
    gap = xp.einsum(
        "Kf,Kfq,aq,q->Kfa",
        space.mesh.jacs_el_fc,
        numerical - current,
        qpost.bas1d_of_ref_edg_qds,
        qpost.weights_JGL,
        optimize=True,
    ).reshape(space.mesh.num_tri, -1)
    coeffs = np.empty((2, space.mesh.num_tri, post.el_dof), dtype=np.float64)
    solve_adr_total_flux_postprocess_kernel(
        coeffs,
        xp.ascontiguousarray(base_coeffs),
        xp.ascontiguousarray(gap),
        xp.ascontiguousarray(space.mesh.aff_jacs),
        xp.ascontiguousarray(qpost.MKrf_inv),
        cache.base_to_post_mass,
        cache.interior_low_to_base,
        cache.interior_low_to_post,
        cache.flux_ainv_constraint_t,
        cache.flux_schur_lu,
        cache.flux_schur_pivots,
    )
    return (post * post).field((coeffs[0], coeffs[1]), name="total_flux_h_star")


def _postprocess_primal_from_total_flux(
        local_unknowns: np.ndarray,
        total_flux_star: VectorDGField,
        beta,
        prepared: ADRPreparedData,
        space: DGSpace,
        trace_ref,
        advection_stabilization,
        diffusion,
        postprocessing_backend="numba",
) -> DGField:
    """Recover u_h^* through the coupled ADR local Neumann HDG problem."""
    from hdgfem.mixed.adr_numba_kernels import (
            solve_adr_primal_from_total_flux_postprocess_kernel,
        )

    from hdgfem.mixed.coefficients import (
            sample_diffusion_tensor,
            inverse_diffusion_values,
        )
    cache = _adr_recovery_cache(space, trace_ref)
    post = cache.post_space
    qpost = post.quad_data
    if postprocessing_backend == "cupy":
        from hdgfem.mixed.postprocess.flux_cupy import postprocess_primal_cupy
        from hdgfem.runtime.optional import require_cupy
        samples = _adr_postprocess_samples(
            beta, prepared, space, post, advection_stabilization, xp=require_cupy())
        return postprocess_primal_cupy(local_unknowns, total_flux_star, space, cache,
                                      samples, diffusion)
    beta_volume, beta_face, tau_total = _adr_postprocess_samples(
        beta, prepared, space, post, advection_stabilization
    )
    total_coeffs = np.ascontiguousarray(
        np.stack(
            (
                total_flux_star.components[0].coeffs,
                total_flux_star.components[1].coeffs,
            ),
            axis=0,
        )
    )
    base_primal = np.ascontiguousarray(
        local_unknowns.reshape(space.mesh.num_tri, 3, space.el_dof)[:, 0]
    )
    coeffs = np.empty(post.shape, dtype=np.float64)
    solve_adr_primal_from_total_flux_postprocess_kernel(
        coeffs,
        base_primal,
        total_coeffs,
        np.ascontiguousarray(space.mesh.aff_jacs),
        np.ascontiguousarray(space.mesh.inv_aff_mats_t),
        np.ascontiguousarray(space.mesh.jacs_el_fc),
        np.ascontiguousarray(space.mesh.normals),
        np.ascontiguousarray(qpost.Krf_w),
        np.ascontiguousarray(qpost.phi),
        np.ascontiguousarray(qpost.gphi),
        np.ascontiguousarray(qpost.weights_JGL),
        np.ascontiguousarray(qpost.bas_of_bd_quads),
        np.ascontiguousarray(qpost.bas1d_of_ref_edg_qds),
        beta_volume,
        beta_face,
        tau_total,
        cache.mean_base,
        inverse_diffusion_values(sample_diffusion_tensor(diffusion, post)),
    )
    return post.field(coeffs, name="u_h_star")
