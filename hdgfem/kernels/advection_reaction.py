r"""Numba kernels for advection-reaction HDG local assembly.

These kernels assemble element-local quantities from already prepared geometry,
reference tables, and coefficient values.  They intentionally do not know about
``DGSpace`` or ``DGField`` objects; the adapter in :mod:`hdgfem.backends.numba`
owns that conversion.
"""

from __future__ import annotations

from hdgfem.kernels.common import njit


@njit(cache=True)
def assemble_local_mats_and_boundary_kernel(
        local_mats,
        element_boundary_mats,
        aff_jacs,
        inv_aff_mats_t,
        jacs_el_fc,
        volume_basis,
        volume_gradients,
        volume_weights,
        face_basis,
        weighted_face_basis,
        weighted_trace_basis,
        beta_volume_values,
        beta_normal_flux,
        tau_face_values,
        reaction_values,
):
    r"""Assemble local advection-reaction blocks and element trace couplings.

    Parameters are raw contiguous arrays.  Important shapes are:

    ``local_mats``
        ``(nK, nel, nel)`` output for
        :math:`\int_K r\phi_i\phi_j
        -(\beta\cdot\nabla\phi_i)\phi_j\,dx
        +\int_{\partial K}\tau\phi_i\phi_j\,ds`.
    ``element_boundary_mats``
        ``(nK, nel, 3*ntr)`` output for the
        :math:`(\tau-\beta\cdot n)\widehat u_h` trace coupling.
    ``volume_basis``
        ``(nq, nel)``.
    ``volume_gradients``
        ``(nq, nel, 2)`` reference gradients.
    ``face_basis`` and ``weighted_face_basis``
        ``(3, nel, nqf)``.
    ``weighted_trace_basis``
        ``(ntr, nqf)``.
    """
    num_elements = local_mats.shape[0]
    nel = local_mats.shape[1]
    ntr = weighted_trace_basis.shape[0]
    nq = volume_weights.shape[0]
    nqf = weighted_trace_basis.shape[1]

    for element in range(num_elements):
        for i in range(nel):
            for j in range(nel):
                local_mats[element, i, j] = 0.0
            for column in range(3 * ntr):
                element_boundary_mats[element, i, column] = 0.0

        jac = aff_jacs[element]
        inv_t00 = inv_aff_mats_t[element, 0, 0]
        inv_t01 = inv_aff_mats_t[element, 0, 1]
        inv_t10 = inv_aff_mats_t[element, 1, 0]
        inv_t11 = inv_aff_mats_t[element, 1, 1]

        for q in range(nq):
            weight = volume_weights[q]
            scaled_weight = jac * weight
            bx = beta_volume_values[element, q, 0]
            by = beta_volume_values[element, q, 1]
            reaction = reaction_values[element, q]

            for i in range(nel):
                grad_ref_0 = volume_gradients[q, i, 0]
                grad_ref_1 = volume_gradients[q, i, 1]
                grad_x = inv_t00 * grad_ref_0 + inv_t01 * grad_ref_1
                grad_y = inv_t10 * grad_ref_0 + inv_t11 * grad_ref_1
                beta_dot_grad = bx * grad_x + by * grad_y
                phi_i = volume_basis[q, i]

                for j in range(nel):
                    phi_j = volume_basis[q, j]
                    local_mats[element, i, j] += scaled_weight * (
                        reaction * phi_i * phi_j - beta_dot_grad * phi_j
                    )

        for face in range(3):
            face_jac = jacs_el_fc[element, face]
            column_offset = face * ntr
            for qf in range(nqf):
                normal_flux = beta_normal_flux[element, face, qf]
                tau = tau_face_values[element, face, qf]
                boundary_weight = face_jac * tau
                trace_weight = face_jac * (tau - normal_flux)

                for i in range(nel):
                    phi_i = face_basis[face, i, qf]
                    weighted_phi_i_factor = boundary_weight * phi_i
                    trace_phi_i_factor = trace_weight * phi_i

                    for j in range(nel):
                        local_mats[element, i, j] += weighted_phi_i_factor * weighted_face_basis[face, j, qf]

                    for j in range(ntr):
                        element_boundary_mats[element, i, column_offset + j] += (
                            trace_phi_i_factor * weighted_trace_basis[j, qf]
                        )


__all__ = ["assemble_local_mats_and_boundary_kernel"]
