r"""Experimental fused Numba kernels for the tensor-diffusion ``test7`` case.

This module is intentionally specialized.  It hard-codes the manufactured
``test7`` tensor, reaction, and source so we can validate a tensor-aware fused
diffusion local solve without first designing the generic coefficient adapter.
"""

from __future__ import annotations

import numpy as np

from .common import lu_factor_inplace, lu_solve_inplace, map_edge_dof_bool, njit, prange
from .diff_rea_fused import _diffusion_lift_dot


@njit(cache=True, inline="always", fastmath=True)
def _test7_kappa_inverse(x, y):
    k00 = 2.0 + x * x
    k01 = 0.5 * x * y
    k11 = 3.0 + y * y
    det = k00 * k11 - k01 * k01
    return k11 / det, -k01 / det, -k01 / det, k00 / det


@njit(cache=True, inline="always", fastmath=True)
def _test7_reaction(x, y):
    return 1.0 + x * x + y * y


@njit(cache=True, inline="always", fastmath=True)
def _test7_exact(x, y, a, b):
    return np.sin(a * (x + 1.0)) * np.sin(b * (y + 1.0))


@njit(cache=True, inline="always", fastmath=True)
def _test7_source(x, y, a, b):
    u = _test7_exact(x, y, a, b)
    ux = a * np.cos(a * (x + 1.0)) * np.sin(b * (y + 1.0))
    uy = b * np.sin(a * (x + 1.0)) * np.cos(b * (y + 1.0))
    uxx = -(a * a) * u
    uyy = -(b * b) * u
    uxy = a * b * np.cos(a * (x + 1.0)) * np.cos(b * (y + 1.0))

    div_kappa_grad_u = (
        (2.0 + x * x) * uxx
        + x * y * uxy
        + (3.0 + y * y) * uyy
        + 2.5 * x * ux
        + 2.5 * y * uy
    )
    return -div_kappa_grad_u + _test7_reaction(x, y) * u


@njit(cache=True, parallel=True, fastmath=True)
def build_test7_boundary_trace_kernel(
        boundary_trace,
        boundary_edges,
        edges,
        node_coords,
        edge_quads,
        edge_weights,
        edge_basis,
        edge_mass_inverse,
        a,
        b,
):
    """Fill boundary trace coefficients by evaluating exact test7 data in Numba."""
    ntr = edge_basis.shape[0]
    nq = edge_weights.shape[0]

    for boundary_pos in prange(boundary_edges.shape[0]):
        edge = boundary_edges[boundary_pos]
        node0 = edges[edge, 0]
        node1 = edges[edge, 1]
        x0 = node_coords[node0, 0]
        y0 = node_coords[node0, 1]
        x1 = node_coords[node1, 0]
        y1 = node_coords[node1, 1]

        rhs = np.empty(ntr, dtype=np.float64)
        for i in range(ntr):
            rhs[i] = 0.0

        for q in range(nq):
            t = edge_quads[q]
            x = 0.5 * ((1.0 - t) * x0 + (1.0 + t) * x1)
            y = 0.5 * ((1.0 - t) * y0 + (1.0 + t) * y1)
            value = _test7_exact(x, y, a, b)
            weight = edge_weights[q]
            for i in range(ntr):
                rhs[i] += value * weight * edge_basis[i, q]

        for j in range(ntr):
            coeff = 0.0
            for i in range(ntr):
                coeff += rhs[i] * edge_mass_inverse[i, j]
            boundary_trace[edge, j] = coeff


@njit(cache=True, inline="always", fastmath=True)
def _physical_quad_point(aff_mats, aff_vecs, element, ref_x, ref_y):
    x = aff_mats[element, 0, 0] * ref_x + aff_mats[element, 0, 1] * ref_y + aff_vecs[element, 0]
    y = aff_mats[element, 1, 0] * ref_x + aff_mats[element, 1, 1] * ref_y + aff_vecs[element, 1]
    return x, y


@njit(cache=True, inline="always", fastmath=True)
def _zero_matrix(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            matrix[i, j] = 0.0


@njit(cache=True, inline="always", fastmath=True)
def _build_test7_tensor_operator(
        schur_matrix,
        d0,
        d1,
        mn0,
        mn1,
        flux_lu,
        flux_pivots,
        flux_d_response,
        element,
        aff_mats,
        aff_vecs,
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        ref_quads,
        ref_weights,
        basis_values,
        face_element_mass,
        d0_reference,
        d1_reference,
):
    """Build tensor local blocks and factor the flux mass block for one element."""
    nel = d0.shape[0]
    two_nel = 2 * nel
    aff00 = aff_mats[element, 0, 0]
    aff01 = aff_mats[element, 0, 1]
    aff10 = aff_mats[element, 1, 0]
    aff11 = aff_mats[element, 1, 1]
    jac = aff_jacs[element]

    _zero_matrix(flux_lu)

    for i in range(nel):
        for j in range(nel):
            d0[i, j] = aff11 * d0_reference[i, j] - aff10 * d1_reference[i, j]
            d1[i, j] = -aff01 * d0_reference[i, j] + aff00 * d1_reference[i, j]

            normal_x_value = 0.0
            normal_y_value = 0.0
            tau_value = 0.0
            for face in range(3):
                face_mass_value = face_element_mass[face, i, j]
                face_scale = jacs_el_fc[element, face]
                tau_value += tau[element, face] * face_scale * face_mass_value
                normal_x_value += face_scale * normals[element, face, 0] * face_mass_value
                normal_y_value += face_scale * normals[element, face, 1] * face_mass_value

            mn0[i, j] = normal_x_value - d0[i, j]
            mn1[i, j] = normal_y_value - d1[i, j]
            schur_matrix[i, j] = tau_value

    for q in range(ref_weights.shape[0]):
        x, y = _physical_quad_point(aff_mats, aff_vecs, element, ref_quads[q, 0], ref_quads[q, 1])
        inv00, inv01, inv10, inv11 = _test7_kappa_inverse(x, y)
        reaction_value = _test7_reaction(x, y)
        jac_weight = jac * ref_weights[q]
        for i in range(nel):
            phi_i_weight = jac_weight * basis_values[i, q]
            for j in range(nel):
                weight = phi_i_weight * basis_values[j, q]
                schur_matrix[i, j] += weight * reaction_value
                flux_lu[i, j] += weight * inv00
                flux_lu[i, nel + j] += weight * inv01
                flux_lu[nel + i, j] += weight * inv10
                flux_lu[nel + i, nel + j] += weight * inv11

    for i in range(nel):
        for j in range(nel):
            flux_d_response[i, j] = d0[i, j]
            flux_d_response[nel + i, j] = d1[i, j]

    lu_factor_inplace(flux_lu, flux_pivots)
    lu_solve_inplace(flux_lu, flux_pivots, flux_d_response)

    for i in range(nel):
        for j in range(nel):
            value = 0.0
            for k in range(nel):
                value += mn0[i, k] * flux_d_response[k, j]
                value += mn1[i, k] * flux_d_response[nel + k, j]
            schur_matrix[i, j] += value


@njit(cache=True, inline="always", fastmath=True)
def _build_test7_tensor_rhs_columns(
        rhs0,
        rhs1,
        rhs2,
        element,
        aff_mats,
        aff_vecs,
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        ref_quads,
        ref_weights,
        basis_values,
        face_element_trace,
        a,
        b,
):
    """Build local RHS columns for all trace dofs plus one source column."""
    nel = rhs0.shape[0]
    ntr = face_element_trace.shape[2]
    trace_cols = 3 * ntr
    for i in range(nel):
        for column in range(trace_cols + 1):
            rhs0[i, column] = 0.0
            rhs1[i, column] = 0.0
            rhs2[i, column] = 0.0

    jac = aff_jacs[element]
    for q in range(ref_weights.shape[0]):
        x, y = _physical_quad_point(aff_mats, aff_vecs, element, ref_quads[q, 0], ref_quads[q, 1])
        source_value = jac * ref_weights[q] * _test7_source(x, y, a, b)
        for i in range(nel):
            rhs0[i, trace_cols] += source_value * basis_values[i, q]

    for face in range(3):
        face_scale = jacs_el_fc[element, face]
        normal_x = normals[element, face, 0]
        normal_y = normals[element, face, 1]
        for trace_dof in range(ntr):
            column = face * ntr + trace_dof
            for i in range(nel):
                coupling = face_scale * face_element_trace[face, i, trace_dof]
                rhs0[i, column] = tau[element, face] * coupling
                rhs1[i, column] = normal_x * coupling
                rhs2[i, column] = normal_y * coupling


@njit(cache=True, inline="always", fastmath=True)
def _solve_test7_tensor_columns(
        local_columns,
        schur_matrix,
        d0,
        d1,
        mn0,
        mn1,
        rhs0,
        rhs1,
        rhs2,
        red_rhs,
        flux_lu,
        flux_pivots,
        flux_rhs,
        schur_pivots,
):
    """Apply the tensor mixed local inverse to all RHS columns."""
    nel = d0.shape[0]
    ncols = rhs0.shape[1]

    for i in range(nel):
        for column in range(ncols):
            flux_rhs[i, column] = rhs1[i, column]
            flux_rhs[nel + i, column] = rhs2[i, column]
    lu_solve_inplace(flux_lu, flux_pivots, flux_rhs)

    for i in range(nel):
        for column in range(ncols):
            value = rhs0[i, column]
            for k in range(nel):
                value += mn0[i, k] * flux_rhs[k, column]
                value += mn1[i, k] * flux_rhs[nel + k, column]
            red_rhs[i, column] = value

    lu_factor_inplace(schur_matrix, schur_pivots)
    lu_solve_inplace(schur_matrix, schur_pivots, red_rhs)

    for i in range(nel):
        for column in range(ncols):
            local_columns[i, column] = red_rhs[i, column]

    for i in range(nel):
        for column in range(ncols):
            value0 = 0.0
            value1 = 0.0
            for j in range(nel):
                value0 += d0[i, j] * red_rhs[j, column]
                value1 += d1[i, j] * red_rhs[j, column]
            flux_rhs[i, column] = value0 - rhs1[i, column]
            flux_rhs[nel + i, column] = value1 - rhs2[i, column]
    lu_solve_inplace(flux_lu, flux_pivots, flux_rhs)

    for i in range(nel):
        for column in range(ncols):
            local_columns[nel + i, column] = flux_rhs[i, column]
            local_columns[2 * nel + i, column] = flux_rhs[nel + i, column]


@njit(cache=True, inline="always", fastmath=True)
def _assemble_test7_tensor_local_columns(
        local_columns,
        element,
        aff_mats,
        aff_vecs,
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        ref_quads,
        ref_weights,
        basis_values,
        face_element_mass,
        face_element_trace,
        d0_reference,
        d1_reference,
        a,
        b,
):
    """Build solved local columns for the test7 tensor diffusion problem."""
    nel = basis_values.shape[0]
    ncols = local_columns.shape[1]
    two_nel = 2 * nel
    schur_matrix = np.empty((nel, nel), dtype=np.float64)
    d0 = np.empty((nel, nel), dtype=np.float64)
    d1 = np.empty((nel, nel), dtype=np.float64)
    mn0 = np.empty((nel, nel), dtype=np.float64)
    mn1 = np.empty((nel, nel), dtype=np.float64)
    flux_lu = np.empty((two_nel, two_nel), dtype=np.float64)
    flux_d_response = np.empty((two_nel, nel), dtype=np.float64)
    rhs0 = np.empty((nel, ncols), dtype=np.float64)
    rhs1 = np.empty((nel, ncols), dtype=np.float64)
    rhs2 = np.empty((nel, ncols), dtype=np.float64)
    red_rhs = np.empty((nel, ncols), dtype=np.float64)
    flux_rhs = np.empty((two_nel, ncols), dtype=np.float64)
    flux_pivots = np.empty(two_nel, dtype=np.int64)
    schur_pivots = np.empty(nel, dtype=np.int64)

    _build_test7_tensor_operator(
        schur_matrix,
        d0,
        d1,
        mn0,
        mn1,
        flux_lu,
        flux_pivots,
        flux_d_response,
        element,
        aff_mats,
        aff_vecs,
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        ref_quads,
        ref_weights,
        basis_values,
        face_element_mass,
        d0_reference,
        d1_reference,
    )
    _build_test7_tensor_rhs_columns(
        rhs0,
        rhs1,
        rhs2,
        element,
        aff_mats,
        aff_vecs,
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        ref_quads,
        ref_weights,
        basis_values,
        face_element_trace,
        a,
        b,
    )
    _solve_test7_tensor_columns(
        local_columns,
        schur_matrix,
        d0,
        d1,
        mn0,
        mn1,
        rhs0,
        rhs1,
        rhs2,
        red_rhs,
        flux_lu,
        flux_pivots,
        flux_rhs,
        schur_pivots,
    )

@njit(cache=True, parallel=True, fastmath=True)
def assemble_test7_tensor_trace_system_eliminated_kernel(
        rows,
        cols,
        data,
        rhs_indices,
        rhs_values,
        loc2glob_edge,
        orientations,
        loc2oriented_face_coupling,
        interior_side_index,
        edge_to_solve_edge,
        valid_elements,
        valid_faces,
        side_flux_offsets,
        aff_mats,
        aff_vecs,
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        ref_quads,
        ref_weights,
        basis_values,
        face_element_mass,
        face_element_trace,
        edge_mass,
        oriented_lifts,
        d0_reference,
        d1_reference,
        boundary_trace,
        a,
        b,
):
    """Fully fused reduced trace assembly for the hard-coded test7 tensor."""
    num_elements = loc2glob_edge.shape[0]
    nel = basis_values.shape[0]
    ntr = edge_mass.shape[0]
    trace_cols = 3 * ntr

    for element in prange(num_elements):
        local_columns = np.empty((3 * nel, trace_cols + 1), dtype=np.float64)
        _assemble_test7_tensor_local_columns(
            local_columns,
            element,
            aff_mats,
            aff_vecs,
            aff_jacs,
            jacs_el_fc,
            normals,
            tau,
            ref_quads,
            ref_weights,
            basis_values,
            face_element_mass,
            face_element_trace,
            d0_reference,
            d1_reference,
            a,
            b,
        )

        for row_face in range(3):
            rhs_base = (element * 3 + row_face) * ntr
            side_id = interior_side_index[element, row_face]
            row_edge = loc2glob_edge[element, row_face]
            row_solve_edge = edge_to_solve_edge[row_edge]
            if side_id < 0 or row_solve_edge < 0:
                for row_dof in range(ntr):
                    rhs_indices[rhs_base + row_dof] = 0
                    rhs_values[rhs_base + row_dof] = 0.0
                continue

            side_base = side_flux_offsets[side_id]
            for row_dof in range(ntr):
                rhs_value = _diffusion_lift_dot(
                    local_columns,
                    oriented_lifts,
                    loc2oriented_face_coupling,
                    normals,
                    tau,
                    jacs_el_fc,
                    element,
                    row_face,
                    row_dof,
                    trace_cols,
                    nel,
                )

                col_block_pos = 0
                for col_face in range(3):
                    col_edge = loc2glob_edge[element, col_face]
                    col_solve_edge = edge_to_solve_edge[col_edge]
                    col_is_positive = orientations[element, col_face]
                    if col_solve_edge >= 0:
                        for col_dof in range(ntr):
                            local_col_dof = map_edge_dof_bool(col_is_positive, col_dof, ntr)
                            column = col_face * ntr + local_col_dof
                            schur_value = _diffusion_lift_dot(
                                local_columns,
                                oriented_lifts,
                                loc2oriented_face_coupling,
                                normals,
                                tau,
                                jacs_el_fc,
                                element,
                                row_face,
                                row_dof,
                                column,
                                nel,
                            )
                            out = side_base + (col_block_pos * ntr + row_dof) * ntr + col_dof
                            rows[out] = row_solve_edge * ntr + row_dof
                            cols[out] = col_solve_edge * ntr + col_dof
                            data[out] = -schur_value
                        col_block_pos += 1
                    else:
                        for col_dof in range(ntr):
                            local_col_dof = map_edge_dof_bool(col_is_positive, col_dof, ntr)
                            column = col_face * ntr + local_col_dof
                            schur_value = _diffusion_lift_dot(
                                local_columns,
                                oriented_lifts,
                                loc2oriented_face_coupling,
                                normals,
                                tau,
                                jacs_el_fc,
                                element,
                                row_face,
                                row_dof,
                                column,
                                nel,
                            )
                            rhs_value += schur_value * boundary_trace[col_edge, col_dof]

                rhs_indices[rhs_base + row_dof] = row_solve_edge * ntr + row_dof
                rhs_values[rhs_base + row_dof] = rhs_value

    mass_offset = side_flux_offsets[side_flux_offsets.shape[0] - 1]
    for side_pos in prange(valid_elements.shape[0]):
        element = valid_elements[side_pos]
        face = valid_faces[side_pos]
        edge = loc2glob_edge[element, face]
        solve_edge = edge_to_solve_edge[edge]
        scale = tau[element, face] * jacs_el_fc[element, face]
        base = mass_offset + side_pos * ntr * ntr
        for i in range(ntr):
            row = solve_edge * ntr + i
            for j in range(ntr):
                out = base + i * ntr + j
                rows[out] = row
                cols[out] = solve_edge * ntr + j
                data[out] = scale * edge_mass[i, j]


@njit(cache=True, inline="always", fastmath=True)
def _build_test7_tensor_reconstruction_rhs(
        rhs0,
        rhs1,
        rhs2,
        element,
        trace,
        loc2glob_edge,
        orientations,
        aff_mats,
        aff_vecs,
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        ref_quads,
        ref_weights,
        basis_values,
        face_element_trace,
        a,
        b,
):
    """Build one local RHS from exact source and the solved trace."""
    nel = basis_values.shape[0]
    ntr = face_element_trace.shape[2]
    for i in range(nel):
        rhs0[i, 0] = 0.0
        rhs1[i, 0] = 0.0
        rhs2[i, 0] = 0.0

    jac = aff_jacs[element]
    for q in range(ref_weights.shape[0]):
        x, y = _physical_quad_point(aff_mats, aff_vecs, element, ref_quads[q, 0], ref_quads[q, 1])
        source_value = jac * ref_weights[q] * _test7_source(x, y, a, b)
        for i in range(nel):
            rhs0[i, 0] += source_value * basis_values[i, q]

    for face in range(3):
        edge = loc2glob_edge[element, face]
        is_positive = orientations[element, face]
        face_scale = jacs_el_fc[element, face]
        normal_x = normals[element, face, 0]
        normal_y = normals[element, face, 1]
        for trace_dof in range(ntr):
            global_dof = edge * ntr + map_edge_dof_bool(is_positive, trace_dof, ntr)
            trace_value = trace[global_dof]
            for i in range(nel):
                coupling = face_scale * face_element_trace[face, i, trace_dof] * trace_value
                rhs0[i, 0] += tau[element, face] * coupling
                rhs1[i, 0] += normal_x * coupling
                rhs2[i, 0] += normal_y * coupling


@njit(cache=True, parallel=True, fastmath=True)
def reconstruct_test7_tensor_local_unknowns_kernel(
        local_unknowns,
        trace,
        loc2glob_edge,
        orientations,
        aff_mats,
        aff_vecs,
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        ref_quads,
        ref_weights,
        basis_values,
        face_element_mass,
        face_element_trace,
        d0_reference,
        d1_reference,
        a,
        b,
):
    """Recover mixed local unknowns for the hard-coded test7 tensor."""
    num_elements = loc2glob_edge.shape[0]
    nel = basis_values.shape[0]

    for element in prange(num_elements):
        local_columns = np.empty((3 * nel, 1), dtype=np.float64)
        two_nel = 2 * nel
        schur_matrix = np.empty((nel, nel), dtype=np.float64)
        d0 = np.empty((nel, nel), dtype=np.float64)
        d1 = np.empty((nel, nel), dtype=np.float64)
        mn0 = np.empty((nel, nel), dtype=np.float64)
        mn1 = np.empty((nel, nel), dtype=np.float64)
        flux_lu = np.empty((two_nel, two_nel), dtype=np.float64)
        flux_d_response = np.empty((two_nel, nel), dtype=np.float64)
        rhs0 = np.empty((nel, 1), dtype=np.float64)
        rhs1 = np.empty((nel, 1), dtype=np.float64)
        rhs2 = np.empty((nel, 1), dtype=np.float64)
        red_rhs = np.empty((nel, 1), dtype=np.float64)
        flux_rhs = np.empty((two_nel, 1), dtype=np.float64)
        flux_pivots = np.empty(two_nel, dtype=np.int64)
        schur_pivots = np.empty(nel, dtype=np.int64)

        _build_test7_tensor_operator(
            schur_matrix,
            d0,
            d1,
            mn0,
            mn1,
            flux_lu,
            flux_pivots,
            flux_d_response,
            element,
            aff_mats,
            aff_vecs,
            aff_jacs,
            jacs_el_fc,
            normals,
            tau,
            ref_quads,
            ref_weights,
            basis_values,
            face_element_mass,
            d0_reference,
            d1_reference,
        )
        _build_test7_tensor_reconstruction_rhs(
            rhs0,
            rhs1,
            rhs2,
            element,
            trace,
            loc2glob_edge,
            orientations,
            aff_mats,
            aff_vecs,
            aff_jacs,
            jacs_el_fc,
            normals,
            tau,
            ref_quads,
            ref_weights,
            basis_values,
            face_element_trace,
            a,
            b,
        )
        _solve_test7_tensor_columns(
            local_columns,
            schur_matrix,
            d0,
            d1,
            mn0,
            mn1,
            rhs0,
            rhs1,
            rhs2,
            red_rhs,
            flux_lu,
            flux_pivots,
            flux_rhs,
            schur_pivots,
        )
        for i in range(3 * nel):
            local_unknowns[element, i] = local_columns[i, 0]


__all__ = [
    "assemble_test7_tensor_trace_system_eliminated_kernel",
    "build_test7_boundary_trace_kernel",
    "reconstruct_test7_tensor_local_unknowns_kernel",
]
