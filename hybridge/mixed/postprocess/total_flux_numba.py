"""Element-parallel Numba kernels for the host ADR total-flux and primal recovery inputs.

They replace the stacked NumPy products of ``total_flux`` (BLAS/einsum calls that run
on one OpenBLAS thread in the host campaigns) with explicit per-element loops; sums
are over the same terms, so results match the NumPy path to round-off. No BLAS is
called inside ``prange`` (scipy's OpenBLAS aborts beyond its caller limit).
"""
from __future__ import annotations

import numpy as np

from hybridge.runtime.optional import njit, prange


@njit(cache=True, parallel=True)
def values_at_points(coeffs, basis):
    """``coeffs @ basis.T``: ``(K, P)`` coefficients at ``(Q, P)`` basis rows -> ``(K, Q)``."""
    num_elements, num_dofs = coeffs.shape
    num_points = basis.shape[0]
    out = np.empty((num_elements, num_points))
    for e in prange(num_elements):
        for q in range(num_points):
            total = 0.
            for i in range(num_dofs):
                total += coeffs[e, i] * basis[q, i]
            out[e, q] = total
    return out


@njit(cache=True, parallel=True)
def values_on_faces(coeffs, face_basis):
    """``einsum('Ki,fiq->Kfq')``: coefficients at ``(3, P, Q)`` face basis values -> ``(K, 3, Q)``."""
    num_elements, num_dofs = coeffs.shape
    num_faces, num_points = face_basis.shape[0], face_basis.shape[2]
    out = np.empty((num_elements, num_faces, num_points))
    for e in prange(num_elements):
        for f in range(num_faces):
            for q in range(num_points):
                total = 0.
                for i in range(num_dofs):
                    total += coeffs[e, i] * face_basis[f, i, q]
                out[e, f, q] = total
    return out


@njit(cache=True, parallel=True)
def project_total_flux(blocks, basis, beta_values, weighted_phi, mass_inverse):
    """Coefficients of q_h + beta u_h projected on the degree-p space, ``(2, K, P)``."""
    num_elements, _, num_dofs = blocks.shape
    num_points = basis.shape[1]
    out = np.empty((2, num_elements, num_dofs))
    for e in prange(num_elements):
        moments = np.zeros((2, num_dofs))
        for q in range(num_points):
            u = qx = qy = 0.
            for i in range(num_dofs):
                u += blocks[e, 0, i] * basis[i, q]
                qx += blocks[e, 1, i] * basis[i, q]
                qy += blocks[e, 2, i] * basis[i, q]
            vx = qx + beta_values[e, q, 0] * u
            vy = qy + beta_values[e, q, 1] * u
            for j in range(num_dofs):
                moments[0, j] += vx * weighted_phi[q, j]
                moments[1, j] += vy * weighted_phi[q, j]
        for c in range(2):
            for j in range(num_dofs):
                total = 0.
                for i in range(num_dofs):
                    total += moments[c, i] * mass_inverse[i, j]
                out[c, e, j] = total
    return out


@njit(cache=True, parallel=True)
def total_flux_recovery_inputs(blocks, local_trace, base_face, trace_basis, base_volume, volume_weights,
                               beta_volume, beta_face, tau_total, normals, mass_inverse, base_to_post_mass,
                               post_mass_inverse, post_face, face_jacobians, edge_basis, edge_weights):
    """``base_coeffs`` (2, K, P) and the face-moment ``gap`` (K, 3*A) of the l2-closest flux recovery.

    Per element: base-space projection of q_h + beta u_h from the recovery volume rule, its
    lift to the post space, and the misfit of its normal trace against the numerical total
    flux ``n.q_h + (beta.n) u_hat + tau (u_h - u_hat)`` in the edge moments.
    """
    num_elements, _, num_dofs = blocks.shape
    num_faces, num_face_points = base_face.shape[0], base_face.shape[2]
    num_trace = local_trace.shape[2]
    num_volume = base_volume.shape[0]
    num_post = post_mass_inverse.shape[0]
    num_moments = edge_basis.shape[0]
    base_coeffs = np.empty((2, num_elements, num_dofs))
    gap = np.empty((num_elements, num_faces * num_moments))
    for e in prange(num_elements):
        moments = np.zeros((2, num_dofs))
        for q in range(num_volume):
            u = qx = qy = 0.
            for i in range(num_dofs):
                u += blocks[e, 0, i] * base_volume[q, i]
                qx += blocks[e, 1, i] * base_volume[q, i]
                qy += blocks[e, 2, i] * base_volume[q, i]
            vx = qx + beta_volume[e, q, 0] * u
            vy = qy + beta_volume[e, q, 1] * u
            for j in range(num_dofs):
                weight = volume_weights[q] * base_volume[q, j]
                moments[0, j] += vx * weight
                moments[1, j] += vy * weight
        lifted = np.zeros((2, num_post))
        for c in range(2):
            for j in range(num_dofs):
                total = 0.
                for i in range(num_dofs):
                    total += moments[c, i] * mass_inverse[i, j]
                base_coeffs[c, e, j] = total
            for k in range(num_post):
                total = 0.
                for j in range(num_dofs):
                    total += base_coeffs[c, e, j] * base_to_post_mass[k, j]
                lifted[c, k] = total
        q0 = np.zeros((2, num_post))
        for c in range(2):
            for j in range(num_post):
                total = 0.
                for k in range(num_post):
                    total += lifted[c, k] * post_mass_inverse[k, j]
                q0[c, j] = total
        for f in range(num_faces):
            nx, ny = normals[e, f, 0], normals[e, f, 1]
            misfit = np.empty(num_face_points)
            for q in range(num_face_points):
                u = qx = qy = 0.
                for i in range(num_dofs):
                    u += blocks[e, 0, i] * base_face[f, i, q]
                    qx += blocks[e, 1, i] * base_face[f, i, q]
                    qy += blocks[e, 2, i] * base_face[f, i, q]
                hat = 0.
                for a in range(num_trace):
                    hat += local_trace[e, f, a] * trace_basis[a, q]
                q0x = q0y = 0.
                for k in range(num_post):
                    q0x += q0[0, k] * post_face[f, k, q]
                    q0y += q0[1, k] * post_face[f, k, q]
                beta_n = beta_face[e, f, q, 0] * nx + beta_face[e, f, q, 1] * ny
                numerical = nx * qx + ny * qy + beta_n * hat + tau_total[e, f, q] * (u - hat)
                misfit[q] = numerical - (nx * q0x + ny * q0y)
            for a in range(num_moments):
                total = 0.
                for q in range(num_face_points):
                    total += misfit[q] * edge_basis[a, q] * edge_weights[q]
                gap[e, f * num_moments + a] = face_jacobians[e, f] * total
    return base_coeffs, gap
