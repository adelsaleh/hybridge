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

from .common import lu_factor_inplace, lu_solve_inplace, map_edge_dof_bool, njit, zero_matrix


@njit(cache=True, inline="always", fastmath=True)
def _eval_scalar_coeff(coeffs, basis_values, element, point, nel):
    value = 0.0
    for k in range(nel):
        value += coeffs[element, k] * basis_values[point, k]
    return value


@njit(cache=True, inline="always", fastmath=True)
def _eval_scalar_face_coeff(coeffs, face_basis, element, face, point, nel):
    value = 0.0
    for k in range(nel):
        value += coeffs[element, k] * face_basis[face, k, point]
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
        beta_coeffs,
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
        source_value = 0.0
        for k in range(nel):
            source_value += source_coeffs[element, k] * mass_matrix[k, i]
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
        normal_x = normals[element, face, 0]
        normal_y = normals[element, face, 1]
        face_jac = jacs_el_fc[element, face]
        column_offset = face * ntr
        for qf in range(nqf):
            beta_x = _eval_scalar_face_coeff(beta_coeffs[0], face_basis, element, face, qf, nel)
            beta_y = _eval_scalar_face_coeff(beta_coeffs[1], face_basis, element, face, qf, nel)
            normal_flux = beta_x * normal_x + beta_y * normal_y
            abs_flux = abs(normal_flux)
            mass_weight = face_jac * abs_flux * face_weights[qf]
            trace_weight = face_jac * (abs_flux - normal_flux) * face_weights[qf]

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
        source_coeffs,
        beta_coeffs,
        reaction_coeffs,
        reaction_scalar,
        reaction_is_scalar,
        boundary_trace,
        boundary_penalty,
):
    r"""Assemble the projected-coefficient HDG trace system in COO form.

    The COO layout is intentionally identical to
    :func:`hdgfem.assembly.hdg.trace_matrix_indices`: first all interior
    element-side flux blocks, then one interior edge mass block per edge, then
    one diagonal penalty entry per boundary trace dof.
    """
    num_elements = loc2glob_edge.shape[0]
    nel = mass_matrix.shape[0]
    ntr = trace_basis.shape[0]
    n_int = int_edges.shape[0]
    n_bnd = bnd_edges.shape[0]
    n_flux = n_int * 2 * ntr * 3 * ntr
    n_mass = n_int * ntr * ntr

    for element in prange(num_elements):
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
            beta_coeffs,
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

            oriented_face = loc2oriented_ref_face[element, row_face]
            lift_scale = 0.5 * jacs_el_fc[element, row_face]
            row_solve_edge = edge_to_solve_edge[row_edge]
            for row_dof in range(ntr):
                rhs_value = 0.0
                for i in range(nel):
                    lift_value = lift_scale * oriented_lifts[oriented_face, row_dof, i]
                    rhs_value += lift_value * local_rhs_columns[i, 3 * ntr]
                rhs_indices[rhs_base + row_dof] = row_solve_edge * ntr + row_dof
                rhs_values[rhs_base + row_dof] = rhs_value

                for col_face in range(3):
                    col_edge = loc2glob_edge[element, col_face]
                    col_solve_edge = edge_to_solve_edge[col_edge]
                    col_is_positive = orientations[element, col_face]
                    for col_dof in range(ntr):
                        local_col_dof = map_edge_dof_bool(col_is_positive, col_dof, ntr)
                        column = col_face * ntr + local_col_dof
                        schur_value = 0.0
                        for i in range(nel):
                            lift_value = lift_scale * oriented_lifts[oriented_face, row_dof, i]
                            schur_value += lift_value * local_rhs_columns[i, column]

                        out = (((side_id * 3 + col_face) * ntr + row_dof) * ntr + col_dof)
                        rows[out] = row_solve_edge * ntr + row_dof
                        cols[out] = col_solve_edge * ntr + col_dof
                        data[out] = -schur_value

    mass_offset = n_flux
    for edge_pos in prange(n_int):
        edge = int_edges[edge_pos]
        solve_edge = edge_to_solve_edge[edge]
        edge_scale = edge_jacs[edge]
        base = mass_offset + edge_pos * ntr * ntr
        for i in range(ntr):
            row = solve_edge * ntr + i
            for j in range(ntr):
                out = base + i * ntr + j
                rows[out] = row
                cols[out] = solve_edge * ntr + j
                data[out] = edge_scale * edge_mass[i, j]

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
        source_coeffs,
        beta_coeffs,
        reaction_coeffs,
        reaction_scalar,
        reaction_is_scalar,
        boundary_trace,
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
    n_free_edges = free_edges.shape[0]

    for element in prange(num_elements):
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
            beta_coeffs,
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

            oriented_face = loc2oriented_ref_face[element, row_face]
            lift_scale = 0.5 * jacs_el_fc[element, row_face]
            side_base = side_flux_offsets[side_id]
            for row_dof in range(ntr):
                rhs_value = 0.0
                for i in range(nel):
                    lift_value = lift_scale * oriented_lifts[oriented_face, row_dof, i]
                    rhs_value += lift_value * local_rhs_columns[i, 3 * ntr]

                col_block_pos = 0
                for col_face in range(3):
                    col_edge = loc2glob_edge[element, col_face]
                    col_solve_edge = edge_to_solve_edge[col_edge]
                    col_is_positive = orientations[element, col_face]
                    if col_solve_edge >= 0:
                        for col_dof in range(ntr):
                            local_col_dof = map_edge_dof_bool(col_is_positive, col_dof, ntr)
                            column = col_face * ntr + local_col_dof
                            schur_value = 0.0
                            for i in range(nel):
                                lift_value = lift_scale * oriented_lifts[oriented_face, row_dof, i]
                                schur_value += lift_value * local_rhs_columns[i, column]

                            out = side_base + (col_block_pos * ntr + row_dof) * ntr + col_dof
                            rows[out] = row_solve_edge * ntr + row_dof
                            cols[out] = col_solve_edge * ntr + col_dof
                            data[out] = -schur_value
                        col_block_pos += 1
                    else:
                        for col_dof in range(ntr):
                            local_col_dof = map_edge_dof_bool(col_is_positive, col_dof, ntr)
                            column = col_face * ntr + local_col_dof
                            schur_value = 0.0
                            for i in range(nel):
                                lift_value = lift_scale * oriented_lifts[oriented_face, row_dof, i]
                                schur_value += lift_value * local_rhs_columns[i, column]
                            rhs_value += schur_value * boundary_trace[col_edge, col_dof]

                rhs_indices[rhs_base + row_dof] = row_solve_edge * ntr + row_dof
                rhs_values[rhs_base + row_dof] = rhs_value

    mass_offset = side_flux_offsets[side_flux_offsets.shape[0] - 1]
    for edge_pos in prange(n_free_edges):
        edge = free_edges[edge_pos]
        solve_edge = edge_to_solve_edge[edge]
        edge_scale = edge_jacs[edge]
        base = mass_offset + edge_pos * ntr * ntr
        for i in range(ntr):
            row = solve_edge * ntr + i
            for j in range(ntr):
                out = base + i * ntr + j
                rows[out] = row
                cols[out] = solve_edge * ntr + j
                data[out] = edge_scale * edge_mass[i, j]


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
        source_coeffs,
        beta_coeffs,
        reaction_coeffs,
        reaction_scalar,
        reaction_is_scalar,
):
    r"""Recover element coefficients by rebuilding and solving local systems."""
    num_elements = loc2glob_edge.shape[0]
    nel = coeffs.shape[1]
    ntr = trace_basis.shape[0]

    for element in prange(num_elements):
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
            beta_coeffs,
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
                    global_dof = edge * ntr + map_edge_dof_bool(is_positive, j, ntr)
                    value += local_rhs_columns[i, face * ntr + j] * trace[global_dof]
            solve_rhs[i, 0] = value

        pivots = np.empty(nel, dtype=np.int64)
        lu_factor_inplace(local_matrix, pivots)
        lu_solve_inplace(local_matrix, pivots, solve_rhs)
        for i in range(nel):
            coeffs[element, i] = solve_rhs[i, 0]


__all__ = [
    "assemble_projected_trace_system_eliminated_kernel",
    "assemble_projected_trace_system_kernel",
    "reconstruct_projected_field_kernel",
]
