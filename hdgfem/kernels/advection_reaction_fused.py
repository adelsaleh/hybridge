r"""Fused Numba kernels for projected advection-reaction HDG assembly.

The kernels in this module assemble the global trace COO matrix directly from
``DGSpace`` reference data and already-projected coefficient arrays.  They do
not materialize dense element-local tensors in Python.  This mirrors the fast
legacy Numba solvers while using the new :mod:`hdgfem` mesh/reference layout.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised only when numba is installed.
    from numba import prange
except ImportError:  # pragma: no cover
    prange = range

from hdgfem.kernels.common import lu_factor_inplace, lu_solve_inplace, map_edge_dof_bool, njit, zero_matrix


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
def _face_normal_flux(beta_coeffs, face_basis, normals, element, face, point, nel):
    """Evaluate the normal advection flux at one element face point."""
    beta_x = _eval_scalar_face_coeff(beta_coeffs[0], face_basis, element, face, point, nel)
    beta_y = _eval_scalar_face_coeff(beta_coeffs[1], face_basis, element, face, point, nel)
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
        beta_coeffs,
        tau_kind,
        tau_scalar,
        tau_coeffs,
):
    r"""Cache ``tau`` and ``gamma=tau-beta.n`` on every side quadrature node."""
    nel = face_basis.shape[1]
    nqf = face_basis.shape[2]
    for face in range(3):
        for qf in range(nqf):
            normal_flux = _face_normal_flux(beta_coeffs, face_basis, normals, element, face, qf, nel)
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
        beta_coeffs,
        tau_kind,
        tau_scalar,
        tau_coeffs,
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
            beta_coeffs,
            tau_kind,
            tau_scalar,
            tau_coeffs,
        )


@njit(cache=True, inline="always")
def _assemble_conflict_face_trace_weights(tau, gamma, element, normals, face_basis,
                                         beta_coeffs, loc2glob_edge, orientations,
                                         edge_side_indices, zero_boundary_flux):
    """Build this element's repaired weights using only immutable neighbor data."""
    nel, nqf = face_basis.shape[1], face_basis.shape[2]
    gauge = np.zeros(3, dtype=np.bool_)
    for face in range(3):
        side = 3 * element + face
        edge = loc2glob_edge[element, face]
        left, right = edge_side_indices[edge, 0], edge_side_indices[edge, 1]
        other = right if left == side else left
        inactive = other >= 0
        for qf in range(nqf):
            a = _face_normal_flux(beta_coeffs, face_basis, normals, element, face, qf, nel)
            if other >= 0:
                k, f = other // 3, other % 3
                q = qf if orientations[element, face] == orientations[k, f] else nqf - 1 - qf
                b = _face_normal_flux(beta_coeffs, face_basis, normals, k, f, q, nel)
                if a >= 0 and b >= 0 and a + b > 0:
                    a = (a - b) * 0.5
                    b = -a
                inactive = inactive and a == 0 and b == 0
            elif zero_boundary_flux:
                a = 0.0
            tau[element, face, qf] = abs(a)
            gamma[element, face, qf] = abs(a) - a
        gauge[face] = inactive and left == side
    return gauge


@njit(cache=True, inline="always")
def _trace_local_dof(is_positive_orientation, dof, edge_dof, trace_orientation_mode):
    """Map a global trace dof into local orientation for nodal/modal traces."""
    if trace_orientation_mode == 1:
        return dof
    return map_edge_dof_bool(is_positive_orientation, dof, edge_dof)


@njit(cache=True, inline="always")
def _trace_orientation_sign(is_positive_orientation, dof, trace_orientation_mode):
    """Return the coefficient sign for the selected trace orientation rule."""
    if trace_orientation_mode == 1 and (not is_positive_orientation) and dof % 2 == 1:
        return -1.0
    return 1.0


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
):
    r"""Assemble one element-local HDG operator and trace/source RHS columns."""
    nel = local_matrix.shape[0]
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

        for j in range(nel):
            if reaction_is_scalar:
                value = reaction_scalar * jac * mass_matrix[i, j]
            else:
                reaction_value = 0.0
                for k in range(nel):
                    reaction_value += reaction_coeffs[element, k] * reaction_triples[k, i, j]
                value = jac * reaction_value

            advection_value = 0.0
            for k in range(nel):
                beta_x_k = beta_coeffs[0, element, k]
                beta_y_k = beta_coeffs[1, element, k]
                advection_value += (
                    beta_x_k
                    * (
                        inv_t00 * advection_tensor[0, k, i, j]
                        + inv_t01 * advection_tensor[1, k, i, j]
                    )
                    + beta_y_k
                    * (
                        inv_t10 * advection_tensor[0, k, i, j]
                        + inv_t11 * advection_tensor[1, k, i, j]
                    )
                )
            local_matrix[i, j] = value - jac * advection_value

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

    for element in prange(num_elements):
        gauge = np.zeros(3, dtype=np.bool_)
        if conflict_averaged:
            gauge = _assemble_conflict_face_trace_weights(
                tau_face_values, gamma_face_values, element, normals, face_basis,
                beta_coeffs, loc2glob_edge, orientations, edge_side_indices, zero_boundary_flux)

        local_matrix = np.empty((nel, nel), dtype=np.float64)
        local_rhs_columns = np.empty((nel, 3 * ntr + 1), dtype=np.float64)
        weighted_lift = np.empty((ntr, nel), dtype=np.float64)

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
        )

        pivots = np.empty(nel, dtype=np.int64)
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

    for element in prange(num_elements):
        gauge = np.zeros(3, dtype=np.bool_)
        if conflict_averaged:
            gauge = _assemble_conflict_face_trace_weights(
                tau_face_values, gamma_face_values, element, normals, face_basis,
                beta_coeffs, loc2glob_edge, orientations, edge_side_indices, zero_boundary_flux)

        local_matrix = np.empty((nel, nel), dtype=np.float64)
        local_rhs_columns = np.empty((nel, 3 * ntr + 1), dtype=np.float64)
        weighted_lift = np.empty((ntr, nel), dtype=np.float64)

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
        )

        pivots = np.empty(nel, dtype=np.int64)
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
):
    r"""Recover element coefficients by rebuilding and solving local systems."""
    num_elements = loc2glob_edge.shape[0]
    nel = coeffs.shape[1]
    ntr = trace_basis.shape[0]

    for element in prange(num_elements):
        if conflict_averaged:
            _assemble_conflict_face_trace_weights(
                tau_face_values, gamma_face_values, element, normals, face_basis,
                beta_coeffs, loc2glob_edge, orientations, edge_side_indices, zero_boundary_flux)

        local_matrix = np.empty((nel, nel), dtype=np.float64)
        local_rhs_columns = np.empty((nel, 3 * ntr + 1), dtype=np.float64)

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
        )

        solve_rhs = np.empty((nel, 1), dtype=np.float64)
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

        pivots = np.empty(nel, dtype=np.int64)
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
