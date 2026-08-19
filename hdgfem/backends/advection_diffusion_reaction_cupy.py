"""CuPy postprocessing kernels for stationary ADR HDG."""

from __future__ import annotations

import numpy as np

from .cupy import require_cupy


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
) -> np.ndarray:
    r"""Reconstruct ``RT_p`` total fluxes with batched device dense solves.

    This is the CuPy mirror of
    ``solve_adr_rt_total_flux_postprocess_kernel``. Static reference and mesh
    tables are uploaded once per call, all element moment matrices are formed
    and solved in batches on the active CUDA device, and only the final
    degree-``p+1`` component coefficients are materialized on the host.
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
    return cp.asnumpy(post_coeffs.transpose(1, 0, 2))


__all__ = ["solve_adr_rt_total_flux_postprocess_cupy"]
