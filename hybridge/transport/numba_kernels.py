r"""Fused Numba kernels for projected advection-reaction HDG assembly.

The kernels in this module assemble the global trace COO matrix directly from
``DGSpace`` reference data and already-projected coefficient arrays.  They do
not materialize dense element-local tensors in Python.  This mirrors the fast
legacy Numba solvers while using the new :mod:`hybridge` mesh/reference layout.

Quadrature-point coefficients (:mod:`hybridge.core.quadrature_coefficients`)
enter through ``pointwise_beta`` (the advection field at the volume points
``beta_volume_values`` (K, NQV, 2) and its normal flux ``beta_normal_flux``
(K, 3, NQF)) and ``pointwise_reaction`` (``reaction_values`` (K, NQV)); both
require the per-point volume operator. Unused inputs are 1-element dummies, so
each kernel keeps one type signature for both modes.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised only when numba is installed.
    from numba import prange
except ImportError:  # pragma: no cover
    prange = range

from hybridge.hdg.numba_common import (
    lu_factor_inplace,
    lu_solve_inplace,
    zero_matrix,
)
from hybridge.runtime.optional import njit
from hybridge.hdg.numba_common import _trace_local_dof, _trace_orientation_sign

# Elements per parallel work item of the fused kernels (scratch is per chunk).
_ELEMENT_CHUNK = 16


@njit(cache=True, inline="always", fastmath=True)
def _eval_scalar_coeff(coeffs, basis_values, element, point, nel):
    """Evaluate a projected or scalar coefficient at one volume point."""
    value = 0.0
    for k in range(nel):
        value += coeffs[element, k] * basis_values[point, k]
    return value


@njit(cache=True, inline="always", fastmath=True)
def _eval_scalar_face_coeff(coeffs, face_basis, element, face, point, nel):
    """Evaluate a projected or scalar coefficient at one face point."""
    value = 0.0
    for k in range(nel):
        value += coeffs[element, k] * face_basis[face, k, point]
    return value


@njit(cache=True, inline="always", fastmath=True)
def _face_normal_flux(beta_coeffs, beta_face_basis, normals, element, face, point, nb,
                      pointwise_beta, beta_normal_flux):
    """Evaluate the normal advection flux at one element face point.

    ``beta_face_basis`` is the advection field basis on the solution face quadrature and
    ``nb`` its size; the advection field may live in another DG space than the solution.
    With ``pointwise_beta`` the flux is read from ``beta_normal_flux`` (K, 3, NQF).
    """
    if pointwise_beta:
        return beta_normal_flux[element, face, point]
    beta_x = _eval_scalar_face_coeff(beta_coeffs[0], beta_face_basis, element, face, point, nb)
    beta_y = _eval_scalar_face_coeff(beta_coeffs[1], beta_face_basis, element, face, point, nb)
    return beta_x * normals[element, face, 0] + beta_y * normals[element, face, 1]


@njit(cache=True, inline="always", fastmath=True)
def _advection_tau(tau_kind, tau_scalar, tau_coeffs, face_basis, element, face, point, nel, normal_flux):
    """Return the upwind stabilization value for a normal flux."""
    if tau_kind == 3:
        return tau_scalar * abs(normal_flux)
    if tau_kind == 0:
        return abs(normal_flux)
    if tau_kind == 1:
        return tau_scalar
    return _eval_scalar_face_coeff(tau_coeffs, face_basis, element, face, point, nel)


@njit(cache=True, inline="always", fastmath=True)
def _projected_source_moment(source_data, source_kind, mass_matrix, element, i, nel):
    """Evaluate one projected source moment on an element."""
    if source_kind == 0:
        return 0.0
    if source_kind == 1:
        return source_data[0, i]
    value = 0.0
    for k in range(nel):
        value += source_data[element, k] * mass_matrix[k, i]
    return value


@njit(cache=True, fastmath=True)
def _assemble_face_trace_weights(
        tau_face_values,
        gamma_face_values,
        element,
        normals,
        face_basis,
        beta_face_basis,
        beta_coeffs,
        tau_kind,
        tau_scalar,
        tau_coeffs,
        pointwise_beta,
        beta_normal_flux,
):
    r"""Cache ``tau`` and ``gamma=tau-beta.n`` on every side quadrature node."""
    nel = face_basis.shape[1]
    nb = beta_face_basis.shape[1]
    nqf = face_basis.shape[2]
    for face in range(3):
        for qf in range(nqf):
            normal_flux = _face_normal_flux(beta_coeffs, beta_face_basis, normals, element, face, qf, nb,
                                            pointwise_beta, beta_normal_flux)
            tau = _advection_tau(
                tau_kind,
                tau_scalar,
                tau_coeffs,
                face_basis,
                element,
                face,
                qf,
                nel,
                normal_flux,
            )
            tau_face_values[element, face, qf] = tau
            gamma_face_values[element, face, qf] = tau - normal_flux


@njit(cache=True, parallel=True, fastmath=True)
def assemble_face_trace_weights_kernel(
        tau_face_values,
        gamma_face_values,
        normals,
        face_basis,
        beta_face_basis,
        beta_coeffs,
        tau_kind,
        tau_scalar,
        tau_coeffs,
        pointwise_beta,
        beta_normal_flux,
):
    r"""Fill side trace weights for every element before fused assembly."""
    num_elements = tau_face_values.shape[0]
    for element in prange(num_elements):
        _assemble_face_trace_weights(
            tau_face_values,
            gamma_face_values,
            element,
            normals,
            face_basis,
            beta_face_basis,
            beta_coeffs,
            tau_kind,
            tau_scalar,
            tau_coeffs,
            pointwise_beta,
            beta_normal_flux,
        )


@njit(cache=True, inline="always")
def _assemble_conflict_face_trace_weights(tau, gamma, element, normals, beta_face_basis,
                                         beta_coeffs, loc2glob_edge, orientations,
                                         edge_side_indices, zero_boundary_flux):
    """Build this element's repaired weights using only immutable neighbor data."""
    gauge = np.zeros(3, dtype=np.bool_)
    _conflict_face_trace_weights_into(gauge, tau, gamma, element, normals, beta_face_basis,
                                      beta_coeffs, loc2glob_edge, orientations,
                                      edge_side_indices, zero_boundary_flux,
                                      False, np.zeros((1, 1, 1), dtype=np.float64))
    return gauge


@njit(cache=True, inline="always")
def _conflict_face_trace_weights_into(gauge, tau, gamma, element, normals, beta_face_basis,
                                      beta_coeffs, loc2glob_edge, orientations,
                                      edge_side_indices, zero_boundary_flux,
                                      pointwise_beta, beta_normal_flux):
    """Write this element's repaired weights and its inactive-face gauge flags.

    With ``pointwise_beta`` both sides' normal fluxes come from ``beta_normal_flux``.
    """
    nb, nqf = beta_face_basis.shape[1], beta_face_basis.shape[2]
    for face in range(3):
        side = 3 * element + face
        edge = loc2glob_edge[element, face]
        left, right = edge_side_indices[edge, 0], edge_side_indices[edge, 1]
        other = right if left == side else left
        inactive = other >= 0
        for qf in range(nqf):
            a = _face_normal_flux(beta_coeffs, beta_face_basis, normals, element, face, qf, nb,
                                  pointwise_beta, beta_normal_flux)
            if other >= 0:
                k, f = other // 3, other % 3
                q = qf if orientations[element, face] == orientations[k, f] else nqf - 1 - qf
                b = _face_normal_flux(beta_coeffs, beta_face_basis, normals, k, f, q, nb,
                                      pointwise_beta, beta_normal_flux)
                if a >= 0 and b >= 0 and a + b > 0:
                    a = (a - b) * 0.5
                    b = -a
                inactive = inactive and a == 0 and b == 0
            elif zero_boundary_flux:
                a = 0.0
            tau[element, face, qf] = abs(a)
            gamma[element, face, qf] = abs(a) - a
        gauge[face] = inactive and left == side


@njit(cache=True, fastmath=True)
def _assemble_weighted_trace_lift_side(
        lift,
        element,
        face,
        is_positive_orientation,
        jacs_el_fc,
        face_basis,
        face_weights,
        trace_basis,
        tau_face_values,
        trace_orientation_mode,
):
    """Assemble ``int_F tau mu phi`` for one element side."""
    ntr = trace_basis.shape[0]
    nel = face_basis.shape[1]
    nqf = face_weights.shape[0]
    face_jac = jacs_el_fc[element, face]
    for row_dof in range(ntr):
        local_row_dof = _trace_local_dof(is_positive_orientation, row_dof, ntr, trace_orientation_mode)
        row_sign = _trace_orientation_sign(is_positive_orientation, row_dof, trace_orientation_mode)
        for i in range(nel):
            value = 0.0
            for qf in range(nqf):
                value += (
                    face_jac
                    * tau_face_values[element, face, qf]
                    * face_weights[qf]
                    * row_sign
                    * trace_basis[local_row_dof, qf]
                    * face_basis[face, i, qf]
                )
            lift[row_dof, i] = value


@njit(cache=True, fastmath=True)
def _weighted_trace_mass_value(
        element,
        face,
        is_positive_orientation,
        row_dof,
        col_dof,
        jacs_el_fc,
        face_basis,
        face_weights,
        trace_basis,
        gamma_face_values,
        trace_orientation_mode,
):
    """Return ``int_F (tau-beta.n) mu_row mu_col`` for one side block entry."""
    ntr = trace_basis.shape[0]
    nqf = face_weights.shape[0]
    local_row_dof = _trace_local_dof(is_positive_orientation, row_dof, ntr, trace_orientation_mode)
    local_col_dof = _trace_local_dof(is_positive_orientation, col_dof, ntr, trace_orientation_mode)
    row_sign = _trace_orientation_sign(is_positive_orientation, row_dof, trace_orientation_mode)
    col_sign = _trace_orientation_sign(is_positive_orientation, col_dof, trace_orientation_mode)
    face_jac = jacs_el_fc[element, face]
    value = 0.0
    for qf in range(nqf):
        value += (
            face_jac
            * gamma_face_values[element, face, qf]
            * face_weights[qf]
            * row_sign
            * col_sign
            * trace_basis[local_row_dof, qf]
            * trace_basis[local_col_dof, qf]
        )
    return value


@njit(cache=True, fastmath=True)
def _assemble_projected_local_system(
        local_matrix,
        local_rhs,
        element,
        aff_jacs,
        inv_aff_mats_t,
        jacs_el_fc,
        normals,
        mass_matrix,
        reaction_triples,
        advection_tensor,
        face_basis,
        face_weights,
        trace_basis,
        source_coeffs,
        source_kind,
        beta_coeffs,
        tau_face_values,
        gamma_face_values,
        reaction_coeffs,
        reaction_scalar,
        reaction_is_scalar,
        volume_quadrature,
        vq_weights,
        vq_phi,
        vq_dphi0,
        vq_dphi1,
        vq_psi,
        vq_h,
        pointwise_beta,
        beta_volume_values,
        pointwise_reaction,
        reaction_values,
):
    r"""Assemble one element-local HDG operator and trace/source RHS columns.

    ``advection_tensor`` is ``(2, NB, NEL, NEL)`` for an advection field with ``NB``
    basis functions; the advection field space may differ from the solution space.
    With ``volume_quadrature`` the volume operator is instead summed over the
    solution volume rule (weights ``vq_weights``, basis ``vq_phi``, reference
    gradients ``vq_dphi0/1``, advection basis ``vq_psi``) as rank-1 updates
    ``A += h_q phi(q)^T``; ``vq_h`` (``NEL``) is caller-owned scratch. Both
    forms integrate the same discrete operator. With ``pointwise_beta`` the
    physical advection field is read at each volume point from ``beta_volume_values``
    (K, NQV, 2), and with ``pointwise_reaction`` the reaction from
    ``reaction_values`` (K, NQV); both need ``volume_quadrature``.
    """
    nel = local_matrix.shape[0]
    nb = beta_coeffs.shape[2]
    ntr = trace_basis.shape[0]
    nqf = face_weights.shape[0]

    zero_matrix(local_matrix)
    for i in range(nel):
        for j in range(3 * ntr + 1):
            local_rhs[i, j] = 0.0

    jac = aff_jacs[element]
    inv_t00 = inv_aff_mats_t[element, 0, 0]
    inv_t01 = inv_aff_mats_t[element, 0, 1]
    inv_t10 = inv_aff_mats_t[element, 1, 0]
    inv_t11 = inv_aff_mats_t[element, 1, 1]
    source_column = 3 * ntr

    for i in range(nel):
        source_value = _projected_source_moment(source_coeffs, source_kind, mass_matrix, element, i, nel)
        local_rhs[i, source_column] = jac * source_value

    for face in range(3):
        face_jac = jacs_el_fc[element, face]
        column_offset = face * ntr
        for qf in range(nqf):
            mass_weight = face_jac * tau_face_values[element, face, qf] * face_weights[qf]
            trace_weight = face_jac * gamma_face_values[element, face, qf] * face_weights[qf]
            for i in range(nel):
                phi_i = face_basis[face, i, qf]
                mass_factor = mass_weight * phi_i
                trace_factor = trace_weight * phi_i
                for j in range(nel):
                    local_matrix[i, j] += mass_factor * face_basis[face, j, qf]
                for j in range(ntr):
                    local_rhs[i, column_offset + j] += trace_factor * trace_basis[j, qf]

    # Reaction mass, then minus the volume advection term.  Each coefficient
    # mode k scales one contiguous (NEL, NEL) tensor slice, so the inner loops
    # stream through memory; the advection field is folded into reference coordinates
    # once per mode (b_D = jac * beta . inv_t[:, D]).
    if reaction_is_scalar:
        scale = reaction_scalar * jac
        for i in range(nel):
            for j in range(nel):
                local_matrix[i, j] += scale * mass_matrix[i, j]
    if volume_quadrature:
        for q in range(vq_weights.shape[0]):
            beta_x = 0.0
            beta_y = 0.0
            if pointwise_beta:
                beta_x = beta_volume_values[element, q, 0]
                beta_y = beta_volume_values[element, q, 1]
            else:
                for k in range(nb):
                    psi = vq_psi[q, k]
                    beta_x += beta_coeffs[0, element, k] * psi
                    beta_y += beta_coeffs[1, element, k] * psi
            weight = jac * vq_weights[q]
            b0 = weight * (beta_x * inv_t00 + beta_y * inv_t10)
            b1 = weight * (beta_x * inv_t01 + beta_y * inv_t11)
            r = 0.0
            if not reaction_is_scalar:
                if pointwise_reaction:
                    r = reaction_values[element, q]
                else:
                    for k in range(nel):
                        r += reaction_coeffs[element, k] * vq_phi[q, k]
                r *= weight
            for i in range(nel):
                vq_h[i] = r * vq_phi[q, i] - b0 * vq_dphi0[q, i] - b1 * vq_dphi1[q, i]
            for i in range(nel):
                h = vq_h[i]
                for j in range(nel):
                    local_matrix[i, j] += h * vq_phi[q, j]
        return
    if not reaction_is_scalar:
        for k in range(nel):
            scale = jac * reaction_coeffs[element, k]
            for i in range(nel):
                for j in range(nel):
                    local_matrix[i, j] += scale * reaction_triples[k, i, j]
    for k in range(nb):
        beta_x_k = beta_coeffs[0, element, k]
        beta_y_k = beta_coeffs[1, element, k]
        b0 = jac * (beta_x_k * inv_t00 + beta_y_k * inv_t10)
        b1 = jac * (beta_x_k * inv_t01 + beta_y_k * inv_t11)
        for i in range(nel):
            for j in range(nel):
                local_matrix[i, j] -= b0 * advection_tensor[0, k, i, j] + b1 * advection_tensor[1, k, i, j]


@njit(cache=True, parallel=True, fastmath=True)
def assemble_projected_trace_system_kernel(
        rows,
        cols,
        data,
        rhs_indices,
        rhs_values,
        loc2glob_edge,
        orientations,
        loc2oriented_ref_face,
        interior_side_index,
        edge_to_solve_edge,
        int_edges,
        bnd_edges,
        edge_jacs,
        aff_jacs,
        inv_aff_mats_t,
        jacs_el_fc,
        normals,
        mass_matrix,
        reaction_triples,
        advection_tensor,
        face_basis,
        beta_face_basis,
        face_weights,
        trace_basis,
        edge_mass,
        oriented_lifts,
        trace_orientation_mode,
        source_coeffs,
        source_kind,
        beta_coeffs,
        tau_face_values,
        gamma_face_values,
        reaction_coeffs,
        reaction_scalar,
        reaction_is_scalar,
        boundary_trace,
        boundary_penalty,
        conflict_averaged,
        edge_side_indices,
        zero_boundary_flux,
        volume_quadrature,
        vq_weights,
        vq_phi,
        vq_dphi0,
        vq_dphi1,
        vq_psi,
        pointwise_beta,
        beta_normal_flux,
        beta_volume_values,
        pointwise_reaction,
        reaction_values,
):
    r"""Assemble the projected-coefficient HDG trace system in COO form.

    The COO layout follows the side-weighted advection trace equation: first
    all interior element-side Schur flux blocks, then one ``tau-beta.n`` mass
    block per interior side, then one diagonal penalty entry per boundary trace
    dof.
    """
    num_elements = loc2glob_edge.shape[0]
    nel = mass_matrix.shape[0]
    ntr = trace_basis.shape[0]
    n_int = int_edges.shape[0]
    n_bnd = bnd_edges.shape[0]
    n_flux = n_int * 2 * ntr * 3 * ntr
    n_mass = n_int * 2 * ntr * ntr

    # Elements are processed in chunks so the dense scratch below is
    # allocated once per chunk, not once per element.
    num_chunks = (num_elements + _ELEMENT_CHUNK - 1) // _ELEMENT_CHUNK
    for chunk in prange(num_chunks):
        gauge = np.zeros(3, dtype=np.bool_)
        local_matrix = np.empty((nel, nel), dtype=np.float64)
        local_rhs_columns = np.empty((nel, 3 * ntr + 1), dtype=np.float64)
        weighted_lift = np.empty((ntr, nel), dtype=np.float64)
        pivots = np.empty(nel, dtype=np.int64)
        vq_h = np.empty(nel, dtype=np.float64)
        for element in range(chunk * _ELEMENT_CHUNK, min(num_elements, (chunk + 1) * _ELEMENT_CHUNK)):
            for face in range(3):
                gauge[face] = False
            if conflict_averaged:
                _conflict_face_trace_weights_into(
                    gauge, tau_face_values, gamma_face_values, element, normals, beta_face_basis,
                    beta_coeffs, loc2glob_edge, orientations, edge_side_indices, zero_boundary_flux,
                    pointwise_beta, beta_normal_flux)

            _assemble_projected_local_system(
                local_matrix,
                local_rhs_columns,
                element,
                aff_jacs,
                inv_aff_mats_t,
                jacs_el_fc,
                normals,
                mass_matrix,
                reaction_triples,
                advection_tensor,
                face_basis,
                face_weights,
                trace_basis,
                source_coeffs,
                source_kind,
                beta_coeffs,
                tau_face_values,
                gamma_face_values,
                reaction_coeffs,
                reaction_scalar,
                reaction_is_scalar,
                volume_quadrature,
                vq_weights,
                vq_phi,
                vq_dphi0,
                vq_dphi1,
                vq_psi,
                vq_h,
                pointwise_beta,
                beta_volume_values,
                pointwise_reaction,
                reaction_values,
            )

            lu_factor_inplace(local_matrix, pivots)
            lu_solve_inplace(local_matrix, pivots, local_rhs_columns)

            for row_face in range(3):
                rhs_base = (element * 3 + row_face) * ntr
                row_edge = loc2glob_edge[element, row_face]
                side_id = interior_side_index[element, row_face]
                if side_id < 0:
                    for row_dof in range(ntr):
                        rhs_indices[rhs_base + row_dof] = 0
                        rhs_values[rhs_base + row_dof] = 0.0
                    continue

                row_is_positive = orientations[element, row_face]
                row_solve_edge = edge_to_solve_edge[row_edge]
                _assemble_weighted_trace_lift_side(
                    weighted_lift,
                    element,
                    row_face,
                    row_is_positive,
                    jacs_el_fc,
                    face_basis,
                    face_weights,
                    trace_basis,
                    tau_face_values,
                    trace_orientation_mode,
                )
                for row_dof in range(ntr):
                    rhs_value = 0.0
                    for i in range(nel):
                        lift_value = weighted_lift[row_dof, i]
                        rhs_value += lift_value * local_rhs_columns[i, 3 * ntr]
                    rhs_indices[rhs_base + row_dof] = row_solve_edge * ntr + row_dof
                    rhs_values[rhs_base + row_dof] = rhs_value

                    for col_face in range(3):
                        col_edge = loc2glob_edge[element, col_face]
                        col_solve_edge = edge_to_solve_edge[col_edge]
                        col_is_positive = orientations[element, col_face]
                        for col_dof in range(ntr):
                            local_col_dof = _trace_local_dof(col_is_positive, col_dof, ntr, trace_orientation_mode)
                            col_sign = _trace_orientation_sign(col_is_positive, col_dof, trace_orientation_mode)
                            column = col_face * ntr + local_col_dof
                            schur_value = 0.0
                            for i in range(nel):
                                lift_value = weighted_lift[row_dof, i]
                                schur_value += lift_value * local_rhs_columns[i, column]

                            out = (((side_id * 3 + col_face) * ntr + row_dof) * ntr + col_dof)
                            rows[out] = row_solve_edge * ntr + row_dof
                            cols[out] = col_solve_edge * ntr + col_dof
                            data[out] = -col_sign * schur_value

                    mass_base = n_flux + side_id * ntr * ntr
                    for col_dof in range(ntr):
                        out = mass_base + row_dof * ntr + col_dof
                        rows[out] = row_solve_edge * ntr + row_dof
                        cols[out] = row_solve_edge * ntr + col_dof
                        data[out] = _weighted_trace_mass_value(
                            element,
                            row_face,
                            row_is_positive,
                            row_dof,
                            col_dof,
                            jacs_el_fc,
                            face_basis,
                            face_weights,
                            trace_basis,
                            gamma_face_values,
                            trace_orientation_mode,
                        )
                        if gauge[row_face] and row_dof == col_dof:
                            data[out] += 1.0

    boundary_matrix_offset = n_flux + n_mass
    boundary_rhs_offset = num_elements * 3 * ntr
    for edge_pos in prange(n_bnd):
        edge = bnd_edges[edge_pos]
        solve_edge = edge_to_solve_edge[edge]
        for i in range(ntr):
            matrix_out = boundary_matrix_offset + edge_pos * ntr + i
            dof = solve_edge * ntr + i
            rows[matrix_out] = dof
            cols[matrix_out] = dof
            data[matrix_out] = boundary_penalty

            rhs_out = boundary_rhs_offset + edge_pos * ntr + i
            rhs_indices[rhs_out] = dof
            rhs_values[rhs_out] = boundary_penalty * boundary_trace[edge, i]


@njit(cache=True, parallel=True, fastmath=True)
def assemble_projected_trace_system_eliminated_kernel(
        rows,
        cols,
        data,
        rhs_indices,
        rhs_values,
        block_rows,
        block_cols,
        block_data,
        block_side_offsets,
        emit_block_coo,
        loc2glob_edge,
        orientations,
        loc2oriented_ref_face,
        interior_side_index,
        edge_to_solve_edge,
        free_edges,
        side_flux_offsets,
        edge_jacs,
        aff_jacs,
        inv_aff_mats_t,
        jacs_el_fc,
        normals,
        mass_matrix,
        reaction_triples,
        advection_tensor,
        face_basis,
        beta_face_basis,
        face_weights,
        trace_basis,
        edge_mass,
        oriented_lifts,
        trace_orientation_mode,
        source_coeffs,
        source_kind,
        beta_coeffs,
        tau_face_values,
        gamma_face_values,
        reaction_coeffs,
        reaction_scalar,
        reaction_is_scalar,
        boundary_trace,
        conflict_averaged,
        edge_side_indices,
        zero_boundary_flux,
        volume_quadrature,
        vq_weights,
        vq_phi,
        vq_dphi0,
        vq_dphi1,
        vq_psi,
        pointwise_beta,
        beta_normal_flux,
        beta_volume_values,
        pointwise_reaction,
        reaction_values,
):
    r"""Assemble a boundary-eliminated projected HDG trace system in COO form.

    Only interior trace rows and columns are emitted.  Schur terms that couple
    an interior row to a prescribed boundary trace column are moved directly to
    the reduced RHS, matching ``b_f - A_fk g`` without a post-assembly sparse
    COO scan.
    """
    num_elements = loc2glob_edge.shape[0]
    nel = mass_matrix.shape[0]
    ntr = trace_basis.shape[0]

    # Elements are processed in chunks so the dense scratch below is
    # allocated once per chunk, not once per element.
    num_chunks = (num_elements + _ELEMENT_CHUNK - 1) // _ELEMENT_CHUNK
    for chunk in prange(num_chunks):
        gauge = np.zeros(3, dtype=np.bool_)
        local_matrix = np.empty((nel, nel), dtype=np.float64)
        local_rhs_columns = np.empty((nel, 3 * ntr + 1), dtype=np.float64)
        weighted_lift = np.empty((ntr, nel), dtype=np.float64)
        pivots = np.empty(nel, dtype=np.int64)
        vq_h = np.empty(nel, dtype=np.float64)
        for element in range(chunk * _ELEMENT_CHUNK, min(num_elements, (chunk + 1) * _ELEMENT_CHUNK)):
            for face in range(3):
                gauge[face] = False
            if conflict_averaged:
                _conflict_face_trace_weights_into(
                    gauge, tau_face_values, gamma_face_values, element, normals, beta_face_basis,
                    beta_coeffs, loc2glob_edge, orientations, edge_side_indices, zero_boundary_flux,
                    pointwise_beta, beta_normal_flux)

            _assemble_projected_local_system(
                local_matrix,
                local_rhs_columns,
                element,
                aff_jacs,
                inv_aff_mats_t,
                jacs_el_fc,
                normals,
                mass_matrix,
                reaction_triples,
                advection_tensor,
                face_basis,
                face_weights,
                trace_basis,
                source_coeffs,
                source_kind,
                beta_coeffs,
                tau_face_values,
                gamma_face_values,
                reaction_coeffs,
                reaction_scalar,
                reaction_is_scalar,
                volume_quadrature,
                vq_weights,
                vq_phi,
                vq_dphi0,
                vq_dphi1,
                vq_psi,
                vq_h,
                pointwise_beta,
                beta_volume_values,
                pointwise_reaction,
                reaction_values,
            )

            lu_factor_inplace(local_matrix, pivots)
            lu_solve_inplace(local_matrix, pivots, local_rhs_columns)

            for row_face in range(3):
                rhs_base = (element * 3 + row_face) * ntr
                row_edge = loc2glob_edge[element, row_face]
                row_solve_edge = edge_to_solve_edge[row_edge]
                side_id = interior_side_index[element, row_face]
                if side_id < 0 or row_solve_edge < 0:
                    for row_dof in range(ntr):
                        rhs_indices[rhs_base + row_dof] = 0
                        rhs_values[rhs_base + row_dof] = 0.0
                    continue

                row_is_positive = orientations[element, row_face]
                side_base = side_flux_offsets[side_id]
                block_side_base = 0
                if emit_block_coo:
                    block_side_base = block_side_offsets[side_id]
                _assemble_weighted_trace_lift_side(
                    weighted_lift,
                    element,
                    row_face,
                    row_is_positive,
                    jacs_el_fc,
                    face_basis,
                    face_weights,
                    trace_basis,
                    tau_face_values,
                    trace_orientation_mode,
                )
                for row_dof in range(ntr):
                    rhs_value = 0.0
                    for i in range(nel):
                        lift_value = weighted_lift[row_dof, i]
                        rhs_value += lift_value * local_rhs_columns[i, 3 * ntr]

                    col_block_pos = 0
                    for col_face in range(3):
                        col_edge = loc2glob_edge[element, col_face]
                        col_solve_edge = edge_to_solve_edge[col_edge]
                        col_is_positive = orientations[element, col_face]
                        if col_solve_edge >= 0:
                            block_out = 0
                            if emit_block_coo:
                                block_out = block_side_base + col_block_pos
                                block_rows[block_out] = row_solve_edge
                                block_cols[block_out] = col_solve_edge
                            for col_dof in range(ntr):
                                local_col_dof = _trace_local_dof(col_is_positive, col_dof, ntr, trace_orientation_mode)
                                col_sign = _trace_orientation_sign(col_is_positive, col_dof, trace_orientation_mode)
                                column = col_face * ntr + local_col_dof
                                schur_value = 0.0
                                for i in range(nel):
                                    lift_value = weighted_lift[row_dof, i]
                                    schur_value += lift_value * local_rhs_columns[i, column]

                                out = side_base + (col_block_pos * ntr + row_dof) * ntr + col_dof
                                rows[out] = row_solve_edge * ntr + row_dof
                                cols[out] = col_solve_edge * ntr + col_dof
                                data[out] = -col_sign * schur_value
                                if emit_block_coo:
                                    block_data[block_out, row_dof, col_dof] = -col_sign * schur_value
                            col_block_pos += 1
                        else:
                            for col_dof in range(ntr):
                                local_col_dof = _trace_local_dof(col_is_positive, col_dof, ntr, trace_orientation_mode)
                                col_sign = _trace_orientation_sign(col_is_positive, col_dof, trace_orientation_mode)
                                column = col_face * ntr + local_col_dof
                                schur_value = 0.0
                                for i in range(nel):
                                    lift_value = weighted_lift[row_dof, i]
                                    schur_value += lift_value * local_rhs_columns[i, column]
                                rhs_value += col_sign * schur_value * boundary_trace[col_edge, col_dof]

                    rhs_indices[rhs_base + row_dof] = row_solve_edge * ntr + row_dof
                    rhs_values[rhs_base + row_dof] = rhs_value

                    mass_base = side_flux_offsets[side_flux_offsets.shape[0] - 1] + side_id * ntr * ntr
                    mass_block_out = 0
                    if emit_block_coo:
                        mass_block_out = block_side_offsets[block_side_offsets.shape[0] - 1] + side_id
                        block_rows[mass_block_out] = row_solve_edge
                        block_cols[mass_block_out] = row_solve_edge
                    for col_dof in range(ntr):
                        out = mass_base + row_dof * ntr + col_dof
                        mass_value = _weighted_trace_mass_value(
                            element,
                            row_face,
                            row_is_positive,
                            row_dof,
                            col_dof,
                            jacs_el_fc,
                            face_basis,
                            face_weights,
                            trace_basis,
                            gamma_face_values,
                            trace_orientation_mode,
                        )
                        if gauge[row_face] and row_dof == col_dof:
                            mass_value += 1.0
                        rows[out] = row_solve_edge * ntr + row_dof
                        cols[out] = row_solve_edge * ntr + col_dof
                        data[out] = mass_value
                        if emit_block_coo:
                            block_data[mass_block_out, row_dof, col_dof] = mass_value


@njit(cache=True, parallel=True, fastmath=True)
def reconstruct_projected_field_kernel(
        coeffs,
        trace,
        loc2glob_edge,
        orientations,
        aff_jacs,
        inv_aff_mats_t,
        jacs_el_fc,
        normals,
        mass_matrix,
        reaction_triples,
        advection_tensor,
        face_basis,
        beta_face_basis,
        face_weights,
        trace_basis,
        trace_orientation_mode,
        source_coeffs,
        source_kind,
        beta_coeffs,
        tau_face_values,
        gamma_face_values,
        reaction_coeffs,
        reaction_scalar,
        reaction_is_scalar,
        conflict_averaged,
        edge_side_indices,
        zero_boundary_flux,
        volume_quadrature,
        vq_weights,
        vq_phi,
        vq_dphi0,
        vq_dphi1,
        vq_psi,
        pointwise_beta,
        beta_normal_flux,
        beta_volume_values,
        pointwise_reaction,
        reaction_values,
):
    r"""Recover element coefficients by rebuilding and solving local systems."""
    num_elements = loc2glob_edge.shape[0]
    nel = coeffs.shape[1]
    ntr = trace_basis.shape[0]

    # Elements are processed in chunks so the dense scratch below is
    # allocated once per chunk, not once per element.
    num_chunks = (num_elements + _ELEMENT_CHUNK - 1) // _ELEMENT_CHUNK
    for chunk in prange(num_chunks):
        gauge = np.zeros(3, dtype=np.bool_)
        local_matrix = np.empty((nel, nel), dtype=np.float64)
        local_rhs_columns = np.empty((nel, 3 * ntr + 1), dtype=np.float64)
        solve_rhs = np.empty((nel, 1), dtype=np.float64)
        pivots = np.empty(nel, dtype=np.int64)
        vq_h = np.empty(nel, dtype=np.float64)
        for element in range(chunk * _ELEMENT_CHUNK, min(num_elements, (chunk + 1) * _ELEMENT_CHUNK)):
            for face in range(3):
                gauge[face] = False
            if conflict_averaged:
                _conflict_face_trace_weights_into(
                    gauge, tau_face_values, gamma_face_values, element, normals, beta_face_basis,
                    beta_coeffs, loc2glob_edge, orientations, edge_side_indices, zero_boundary_flux,
                    pointwise_beta, beta_normal_flux)

            _assemble_projected_local_system(
                local_matrix,
                local_rhs_columns,
                element,
                aff_jacs,
                inv_aff_mats_t,
                jacs_el_fc,
                normals,
                mass_matrix,
                reaction_triples,
                advection_tensor,
                face_basis,
                face_weights,
                trace_basis,
                source_coeffs,
                source_kind,
                beta_coeffs,
                tau_face_values,
                gamma_face_values,
                reaction_coeffs,
                reaction_scalar,
                reaction_is_scalar,
                volume_quadrature,
                vq_weights,
                vq_phi,
                vq_dphi0,
                vq_dphi1,
                vq_psi,
                vq_h,
                pointwise_beta,
                beta_volume_values,
                pointwise_reaction,
                reaction_values,
            )

            for i in range(nel):
                value = local_rhs_columns[i, 3 * ntr]
                for face in range(3):
                    edge = loc2glob_edge[element, face]
                    is_positive = orientations[element, face]
                    for j in range(ntr):
                        global_trace_dof = _trace_local_dof(is_positive, j, ntr, trace_orientation_mode)
                        trace_sign = _trace_orientation_sign(is_positive, j, trace_orientation_mode)
                        global_dof = edge * ntr + global_trace_dof
                        value += local_rhs_columns[i, face * ntr + j] * trace_sign * trace[global_dof]
                solve_rhs[i, 0] = value

            lu_factor_inplace(local_matrix, pivots)
            lu_solve_inplace(local_matrix, pivots, solve_rhs)
            for i in range(nel):
                coeffs[element, i] = solve_rhs[i, 0]


__all__ = [
    "assemble_face_trace_weights_kernel",
    "assemble_projected_trace_system_eliminated_kernel",
    "assemble_projected_trace_system_kernel",
    "reconstruct_projected_field_kernel",
]
