"""CuPy postprocessing kernels for stationary ADR HDG."""

from __future__ import annotations

import numpy as np

from hdgfem.runtime.optional import require_cupy


def solve_adr_rt_total_flux_postprocess_cupy(
        total_flux_values: np.ndarray,
        numerical_normal_flux: np.ndarray,
        aff_mats: np.ndarray,
        aff_jacs: np.ndarray,
        jacs_el_fc: np.ndarray,
        normals: np.ndarray,
        volume_weights: np.ndarray,
        face_weights: np.ndarray,
        post_weighted_basis: np.ndarray,
        post_mass_inverse: np.ndarray,
        base_volume_basis: np.ndarray,
        base_face_basis: np.ndarray,
        radial_volume: np.ndarray,
        radial_face: np.ndarray,
        low_volume_basis: np.ndarray,
        face_test_basis: np.ndarray,
        *, materialize_host: bool = True,
) -> np.ndarray:
    r"""Reconstruct ``RT_p`` total fluxes with batched device dense solves.

    This is the CuPy mirror of
    ``solve_adr_rt_total_flux_postprocess_kernel``. Static reference and mesh
    tables are uploaded once per call, all element moment matrices are formed
    and solved in batches on the active CUDA device. Final degree-``p+1``
    coefficients stay on device when ``materialize_host=False``; the default
    preserves the host-returning helper contract.
    """
    cp = require_cupy()
    total_values = cp.asarray(total_flux_values)
    numerical = cp.asarray(numerical_normal_flux)
    affine = cp.asarray(aff_mats)
    jacobian = cp.asarray(aff_jacs)
    face_jacobian = cp.asarray(jacs_el_fc)
    normal = cp.asarray(normals)
    volume_w = cp.asarray(volume_weights)
    face_w = cp.asarray(face_weights)
    weighted_post = cp.asarray(post_weighted_basis)
    post_mass_inv = cp.asarray(post_mass_inverse)
    base_volume = cp.asarray(base_volume_basis)
    base_face = cp.asarray(base_face_basis)
    radial_vol = cp.asarray(radial_volume)
    radial_fc = cp.asarray(radial_face)
    low_volume = cp.asarray(low_volume_basis)
    face_test = cp.asarray(face_test_basis)

    num_elements = affine.shape[0]
    base_dof = base_volume.shape[1]
    enrichment_dof = radial_vol.shape[1]
    rt_dof = 2 * base_dof + enrichment_dof
    volume_quads = volume_w.size
    face_quads = face_w.size
    scaled_affine = affine / jacobian[:, None, None]

    rt_volume = cp.empty(
        (num_elements, 2, rt_dof, volume_quads), dtype=cp.float64
    )
    base_volume_t = base_volume.T
    rt_volume[:, 0, :base_dof] = (
        scaled_affine[:, 0, 0, None, None] * base_volume_t[None]
    )
    rt_volume[:, 1, :base_dof] = (
        scaled_affine[:, 1, 0, None, None] * base_volume_t[None]
    )
    rt_volume[:, 0, base_dof : 2 * base_dof] = (
        scaled_affine[:, 0, 1, None, None] * base_volume_t[None]
    )
    rt_volume[:, 1, base_dof : 2 * base_dof] = (
        scaled_affine[:, 1, 1, None, None] * base_volume_t[None]
    )
    rt_volume[:, :, 2 * base_dof :] = cp.einsum(
        "Kac,cjq->Kajq", scaled_affine, radial_vol, optimize=True
    )

    rt_face = cp.empty(
        (num_elements, 2, rt_dof, 3, face_quads), dtype=cp.float64
    )
    base_face_dof_first = base_face.transpose(1, 0, 2)
    rt_face[:, 0, :base_dof] = (
        scaled_affine[:, 0, 0, None, None, None]
        * base_face_dof_first[None]
    )
    rt_face[:, 1, :base_dof] = (
        scaled_affine[:, 1, 0, None, None, None]
        * base_face_dof_first[None]
    )
    rt_face[:, 0, base_dof : 2 * base_dof] = (
        scaled_affine[:, 0, 1, None, None, None]
        * base_face_dof_first[None]
    )
    rt_face[:, 1, base_dof : 2 * base_dof] = (
        scaled_affine[:, 1, 1, None, None, None]
        * base_face_dof_first[None]
    )
    rt_face[:, :, 2 * base_dof :] = cp.einsum(
        "Kac,cfjq->Kajfq", scaled_affine, radial_fc, optimize=True
    )

    rt_normal = cp.einsum(
        "Kadfq,Kfa->Kfdq", rt_face, normal, optimize=True
    )
    face_matrix = cp.einsum(
        "Kf,q,iq,Kfdq->Kfid",
        face_jacobian,
        face_w,
        face_test,
        rt_normal,
        optimize=True,
    ).reshape(num_elements, -1, rt_dof)
    face_rhs = cp.einsum(
        "Kf,q,iq,Kfq->Kfi",
        face_jacobian,
        face_w,
        face_test,
        numerical,
        optimize=True,
    ).reshape(num_elements, -1)

    interior_matrix = cp.einsum(
        "K,q,qi,Kadq->Kiad",
        jacobian,
        volume_w,
        low_volume,
        rt_volume,
        optimize=True,
    )
    interior_rhs = cp.einsum(
        "K,q,qi,Kaq->Kia",
        jacobian,
        volume_w,
        low_volume,
        total_values.transpose(1, 0, 2),
        optimize=True,
    )

    moment_matrix = cp.concatenate(
        (
            face_matrix,
            interior_matrix[:, :, 0],
            interior_matrix[:, :, 1],
        ),
        axis=1,
    )
    moment_rhs = cp.concatenate(
        (face_rhs, interior_rhs[:, :, 0], interior_rhs[:, :, 1]), axis=1
    )
    rt_coeffs = cp.linalg.solve(moment_matrix, moment_rhs[..., None])[..., 0]

    rt_values = cp.einsum(
        "Kd,Kadq->Kaq", rt_coeffs, rt_volume, optimize=True
    )
    post_moments = cp.einsum(
        "Kaq,qi->Kai", rt_values, weighted_post, optimize=True
    )
    post_coeffs = cp.einsum(
        "Kai,ij->Kaj", post_moments, post_mass_inv, optimize=True
    )
    result = cp.ascontiguousarray(post_coeffs.transpose(1, 0, 2))
    return cp.asnumpy(result) if materialize_host else result


__all__ = ["solve_adr_rt_total_flux_postprocess_cupy"]


def postprocess_total_flux_l2_cupy(total_values, numerical, space, cache):
    """Apply the host-equivalent constrained minimum-distance flux recovery."""
    from hdgfem.core.device import field_from_cupy_coefficients
    from hdgfem.core.space import VectorDGField

    cp = require_cupy()
    post = cache.post_space
    q = post.quad_data
    n, d = space.mesh.num_tri, post.el_dof
    jac = cp.asarray(space.mesh.aff_jacs)
    normals = cp.asarray(space.mesh.normals)
    face_jac = cp.asarray(space.mesh.jacs_el_fc)
    mass_inv = cp.asarray(q.MKrf_inv)
    base_basis = cp.asarray(space.basis_at(q.Krf_quads))
    base = total_values @ (cp.asarray(q.Krf_w)[:, None] * base_basis) @ cp.asarray(space.quad_data.MKrf_inv)
    initial = base @ cp.asarray(cache.base_to_post_mass).T @ mass_inv
    face_moments = cp.asarray(q.face_element_test_trace_trial)
    face_constraints = cp.einsum('Kf,Kfc,fia->Kfaci', face_jac, normals, face_moments).reshape(n, -1, 2*d)
    low = cp.asarray(cache.interior_low_to_post)
    low_base = cp.asarray(cache.interior_low_to_base)
    low_dof = low.shape[0]
    interior = cp.zeros((n, 2*low_dof, 2*d))
    interior[:, :low_dof, :d] = jac[:, None, None] * low
    interior[:, low_dof:, d:] = jac[:, None, None] * low
    constraints = cp.concatenate((face_constraints, interior), axis=1)
    face_target = cp.einsum('Kf,Kfq,aq,q->Kfa', face_jac, numerical,
                            cp.asarray(q.bas1d_of_ref_edg_qds), cp.asarray(q.weights_JGL)).reshape(n, -1)
    interior_target = (base @ low_base.T).transpose(1, 0, 2).reshape(n, -1) * jac[:, None]
    target = cp.concatenate((face_target, interior_target), axis=1)
    initial = initial.transpose(1, 0, 2).reshape(n, 2*d)
    lift = cp.concatenate((mass_inv @ constraints[:, :, :d].transpose(0, 2, 1),
                           mass_inv @ constraints[:, :, d:].transpose(0, 2, 1)), axis=1) / jac[:, None, None]
    gap = target - (constraints @ initial[..., None])[..., 0]
    correction = cp.linalg.solve(constraints @ lift, gap[..., None])
    result = (initial + (lift @ correction)[..., 0]).reshape(n, 2, d)
    return VectorDGField(tuple(field_from_cupy_coefficients(post, result[:, c], name='total_flux_h_star')
                               for c in range(2)), name='total_flux_h_star')


def _primal_system_cupy(local_unknowns, total_flux, space, cache, samples, diffusion):
    """Independent contraction reference for the coupled ADR Neumann system.

    Volume, face and mean equations match the Numba reference, including the
    total numerical flux and the scalar Neumann multiplier.
    """
    from hdgfem.core.device import as_cupy_coefficients, as_cupy_space

    cp = require_cupy()
    post = cache.post_space
    q = post.quad_data
    n, d, e = space.mesh.num_tri, post.el_dof, q.edg_dof
    offset, rows = 3*d, 3*d + 3*e + 1
    matrix = cp.zeros((n, rows, rows))
    rhs = cp.zeros((n, rows))
    jac = cp.asarray(space.mesh.aff_jacs)
    normal = cp.asarray(space.mesh.normals)
    face_jac = cp.asarray(space.mesh.jacs_el_fc)
    phi, weights = cp.asarray(q.phi), cp.asarray(q.Krf_w)
    gradient = cp.einsum('Kab,qib->Kqia', cp.asarray(space.mesh.inv_aff_mats_t), cp.asarray(q.gphi))
    beta_volume, beta_face, tau = samples
    flux = cp.stack([as_cupy_coefficients(f, as_cupy_space(f.space)) for f in total_flux.components], axis=1)
    values = flux @ phi.T
    from hdgfem.mixed.coefficients import (
            sample_diffusion_tensor,
            inverse_diffusion_values,
        )
    inverse = inverse_diffusion_values(sample_diffusion_tensor(diffusion, post, device=True))
    for c in range(2):
        constitutive = slice(c*d, (c+1)*d)
        flux_columns = slice((c+1)*d, (c+2)*d)
        grad_mass = cp.einsum('K,q,Kqi,qj->Kij', jac, weights, gradient[..., c], phi)
        matrix[:, constitutive, :d] = -grad_mass
        for component in range(2):
            matrix[:, constitutive, (component+1)*d:(component+2)*d] = cp.einsum(
                'K,q,Kq,qi,qj->Kij', jac, weights, inverse[..., 2*c+component], phi, phi)
        matrix[:, 2*d:3*d, flux_columns] = -grad_mass
    matrix[:, 2*d:3*d, :d] = -cp.einsum('K,q,Kqia,Kqa,qj->Kij', jac, weights, gradient, beta_volume, phi)
    rhs[:, 2*d:3*d] = -cp.einsum('K,q,Kqia,Kaq->Ki', jac, weights, gradient, values)
    mean = jac[:, None] * (weights @ phi)
    matrix[:, 2*d:3*d, -1] = mean
    matrix[:, -1, :d] = mean
    base = cp.asarray(local_unknowns).reshape(n, 3, space.el_dof)[:, 0]
    rhs[:, -1] = jac * (base @ cp.asarray(cache.mean_base))
    face_phi = cp.asarray(q.bas_of_bd_quads)
    trace_phi = cp.asarray(q.bas1d_of_ref_edg_qds)
    face_weights = cp.asarray(q.weights_JGL)
    for face in range(3):
        p = face_phi[face]
        scale = face_jac[:, face, None] * face_weights
        bn = cp.einsum('Kqa,Ka->Kq', beta_face[:, face], normal[:, face])
        normal_flux = cp.einsum('Kai,Ka,iq->Kq', flux, normal[:, face], p)
        trace_slice = slice(offset + face*e, offset + (face+1)*e)
        mass_face = cp.einsum('Kq,iq,jq->Kij', scale, p, p)
        cross = cp.einsum('Kq,iq,aq->Kia', scale, p, trace_phi)
        matrix[:, 2*d:3*d, :d] += cp.einsum('Kq,Kq,iq,jq->Kij', scale, tau[:, face], p, p)
        matrix[:, 2*d:3*d, trace_slice] += cp.einsum('Kq,Kq,iq,aq->Kia', scale, bn-tau[:, face], p, trace_phi)
        matrix[:, trace_slice, :d] += cp.einsum('Kq,Kq,aq,iq->Kai', scale, tau[:, face], trace_phi, p)
        matrix[:, trace_slice, trace_slice] += cp.einsum('Kq,Kq,aq,bq->Kab', scale, bn-tau[:, face], trace_phi, trace_phi)
        for c in range(2):
            nc = normal[:, face, c, None, None]
            matrix[:, c*d:(c+1)*d, trace_slice] += nc * cross
            matrix[:, 2*d:3*d, (c+1)*d:(c+2)*d] += nc * mass_face
            matrix[:, trace_slice, (c+1)*d:(c+2)*d] += nc * cross.transpose(0, 2, 1)
        rhs[:, 2*d:3*d] += cp.einsum('Kq,Kq,iq->Ki', scale, normal_flux, p)
        rhs[:, trace_slice] += cp.einsum('Kq,Kq,aq->Ka', scale, normal_flux, trace_phi)
    return matrix, rhs


def postprocess_primal_cupy(local_unknowns, total_flux, space, cache, samples, diffusion):
    """Recover tensor ADR primal coefficients using fused assembly and batched LU."""
    from hdgfem.mixed.coefficients import (
            sample_diffusion_tensor,
            inverse_diffusion_values,
        )
    from hdgfem.mixed.postprocess.primal_raw_cuda import primal_system_raw_cuda
    from hdgfem.core.device import field_from_cupy_coefficients

    cp = require_cupy()
    post = cache.post_space
    inverse = inverse_diffusion_values(sample_diffusion_tensor(diffusion, post, device=True))
    matrix, rhs = primal_system_raw_cuda(local_unknowns, total_flux, space, cache, samples, inverse)
    result = cp.linalg.solve(matrix, rhs[..., None])[:, :post.el_dof, 0]
    return field_from_cupy_coefficients(post, result, name='u_h_star')
