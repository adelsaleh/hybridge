r"""Fused Numba kernels for diffusion-reaction HDG trace assembly.

The kernels here consume already-built mixed local diffusion solvers.  They
assemble the boundary-eliminated trace system directly, so prescribed Dirichlet
trace coefficients are moved into the reduced RHS instead of being imposed by a
large diagonal penalty.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised only when numba is installed.
    from numba import prange
except ImportError:  # pragma: no cover
    prange = range

from .common import lu_factor_inplace, lu_solve_inplace, map_edge_dof_bool, njit


@njit(cache=True, inline="always", fastmath=True)
def _apply_local_solver_columns(local_columns, local_solver, boundary_mats, source_rhs, element):
    """Fill local solution columns for trace columns plus one source column."""
    rows = local_solver.shape[1]
    trace_cols = boundary_mats.shape[2]
    source_col = trace_cols
    for i in range(rows):
        for col in range(trace_cols + 1):
            value = 0.0
            for j in range(rows):
                rhs_value = source_rhs[element, j] if col == source_col else boundary_mats[element, j, col]
                value += local_solver[element, i, j] * rhs_value
            local_columns[i, col] = value


@njit(cache=True, inline="always", fastmath=True)
def _build_projected_diffusion_operator(
        schur_matrix,
        d0,
        d1,
        mn0,
        mn1,
        k_d0,
        k_d1,
        element,
        aff_mats,
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        mass_matrix,
        mass_inverse,
        reaction_triples,
        face_element_mass,
        d0_reference,
        d1_reference,
        reaction_coeffs,
        reaction_scalar,
        reaction_is_scalar,
):
    """Build the condensed scalar diffusion operator for one element."""
    nel = mass_matrix.shape[0]
    aff00 = aff_mats[element, 0, 0]
    aff01 = aff_mats[element, 0, 1]
    aff10 = aff_mats[element, 1, 0]
    aff11 = aff_mats[element, 1, 1]
    jac = aff_jacs[element]

    for i in range(nel):
        for j in range(nel):
            d0[i, j] = aff11 * d0_reference[i, j] - aff10 * d1_reference[i, j]
            d1[i, j] = -aff01 * d0_reference[i, j] + aff00 * d1_reference[i, j]

            if reaction_is_scalar:
                reaction_value = reaction_scalar * jac * mass_matrix[i, j]
            else:
                weighted = 0.0
                for k in range(nel):
                    weighted += reaction_coeffs[element, k] * reaction_triples[k, i, j]
                reaction_value = jac * weighted

            normal_x_value = 0.0
            normal_y_value = 0.0
            tau_value = reaction_value
            for face in range(3):
                face_mass_value = face_element_mass[face, i, j]
                face_scale = jacs_el_fc[element, face]
                tau_value += tau[element, face] * face_scale * face_mass_value
                normal_x_value += face_scale * normals[element, face, 0] * face_mass_value
                normal_y_value += face_scale * normals[element, face, 1] * face_mass_value

            mn0[i, j] = normal_x_value - d0[i, j]
            mn1[i, j] = normal_y_value - d1[i, j]
            schur_matrix[i, j] = tau_value

    for i in range(nel):
        for j in range(nel):
            value0 = 0.0
            value1 = 0.0
            for k in range(nel):
                value0 += mass_inverse[i, k] * d0[k, j]
                value1 += mass_inverse[i, k] * d1[k, j]
            k_d0[i, j] = value0
            k_d1[i, j] = value1

    jac_inverse = 1.0 / jac
    for i in range(nel):
        for j in range(nel):
            value0 = 0.0
            value1 = 0.0
            for k in range(nel):
                value0 += mn0[i, k] * k_d0[k, j]
                value1 += mn1[i, k] * k_d1[k, j]
            schur_matrix[i, j] += jac_inverse * (value0 + value1)


@njit(cache=True, inline="always", fastmath=True)
def _build_projected_diffusion_rhs_columns(
        rhs0,
        rhs1,
        rhs2,
        element,
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        mass_matrix,
        face_element_trace,
        source_coeffs,
):
    """Build local RHS columns for trace dofs plus one source column."""
    nel = mass_matrix.shape[0]
    ntr = face_element_trace.shape[2]
    trace_cols = 3 * ntr
    for i in range(nel):
        for column in range(trace_cols + 1):
            rhs0[i, column] = 0.0
            rhs1[i, column] = 0.0
            rhs2[i, column] = 0.0

    jac = aff_jacs[element]
    for i in range(nel):
        source_value = 0.0
        for k in range(nel):
            source_value += source_coeffs[element, k] * mass_matrix[k, i]
        rhs0[i, trace_cols] = jac * source_value

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
def _solve_projected_diffusion_columns(
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
        tmp1,
        tmp2,
        pivots,
        mass_inverse,
        aff_jac,
):
    """Apply the mixed diffusion local inverse to all RHS columns."""
    nel = mass_inverse.shape[0]
    ncols = rhs0.shape[1]
    jac_inverse = 1.0 / aff_jac

    for i in range(nel):
        for column in range(ncols):
            value1 = 0.0
            value2 = 0.0
            for k in range(nel):
                value1 += mass_inverse[i, k] * rhs1[k, column]
                value2 += mass_inverse[i, k] * rhs2[k, column]
            tmp1[i, column] = value1
            tmp2[i, column] = value2

    for i in range(nel):
        for column in range(ncols):
            value = rhs0[i, column]
            acc0 = 0.0
            acc1 = 0.0
            for k in range(nel):
                acc0 += mn0[i, k] * tmp1[k, column]
                acc1 += mn1[i, k] * tmp2[k, column]
            red_rhs[i, column] = value + jac_inverse * (acc0 + acc1)

    lu_factor_inplace(schur_matrix, pivots)
    lu_solve_inplace(schur_matrix, pivots, red_rhs)

    for i in range(nel):
        for column in range(ncols):
            u_value = red_rhs[i, column]
            local_columns[i, column] = u_value

    for i in range(nel):
        for column in range(ncols):
            value0 = 0.0
            value1 = 0.0
            for j in range(nel):
                value0 += d0[i, j] * red_rhs[j, column]
                value1 += d1[i, j] * red_rhs[j, column]
            tmp1[i, column] = value0 - rhs1[i, column]
            tmp2[i, column] = value1 - rhs2[i, column]

    for i in range(nel):
        for column in range(ncols):
            value0 = 0.0
            value1 = 0.0
            for k in range(nel):
                value0 += mass_inverse[i, k] * tmp1[k, column]
                value1 += mass_inverse[i, k] * tmp2[k, column]
            local_columns[nel + i, column] = jac_inverse * value0
            local_columns[2 * nel + i, column] = jac_inverse * value1


@njit(cache=True, inline="always", fastmath=True)
def _assemble_projected_diffusion_local_columns(
        local_columns,
        element,
        aff_mats,
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        mass_matrix,
        mass_inverse,
        reaction_triples,
        face_element_mass,
        face_element_trace,
        d0_reference,
        d1_reference,
        source_coeffs,
        reaction_coeffs,
        reaction_scalar,
        reaction_is_scalar,
):
    """Build solved local columns for projected diffusion-reaction assembly."""
    nel = mass_matrix.shape[0]
    ncols = local_columns.shape[1]
    schur_matrix = np.empty((nel, nel), dtype=np.float64)
    d0 = np.empty((nel, nel), dtype=np.float64)
    d1 = np.empty((nel, nel), dtype=np.float64)
    mn0 = np.empty((nel, nel), dtype=np.float64)
    mn1 = np.empty((nel, nel), dtype=np.float64)
    k_d0 = np.empty((nel, nel), dtype=np.float64)
    k_d1 = np.empty((nel, nel), dtype=np.float64)
    rhs0 = np.empty((nel, ncols), dtype=np.float64)
    rhs1 = np.empty((nel, ncols), dtype=np.float64)
    rhs2 = np.empty((nel, ncols), dtype=np.float64)
    red_rhs = np.empty((nel, ncols), dtype=np.float64)
    tmp1 = np.empty((nel, ncols), dtype=np.float64)
    tmp2 = np.empty((nel, ncols), dtype=np.float64)
    pivots = np.empty(nel, dtype=np.int64)

    _build_projected_diffusion_operator(
        schur_matrix,
        d0,
        d1,
        mn0,
        mn1,
        k_d0,
        k_d1,
        element,
        aff_mats,
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        mass_matrix,
        mass_inverse,
        reaction_triples,
        face_element_mass,
        d0_reference,
        d1_reference,
        reaction_coeffs,
        reaction_scalar,
        reaction_is_scalar,
    )
    _build_projected_diffusion_rhs_columns(
        rhs0,
        rhs1,
        rhs2,
        element,
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        mass_matrix,
        face_element_trace,
        source_coeffs,
    )
    _solve_projected_diffusion_columns(
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
        tmp1,
        tmp2,
        pivots,
        mass_inverse,
        aff_jacs[element],
    )


@njit(cache=True, inline="always", fastmath=True)
def _build_projected_tensor_diffusion_operator(
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
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        mass_matrix,
        reaction_triples,
        face_element_mass,
        d0_reference,
        d1_reference,
        reaction_coeffs,
        reaction_scalar,
        reaction_is_scalar,
        inv00_coeffs,
        inv01_coeffs,
        inv10_coeffs,
        inv11_coeffs,
):
    """Build projected tensor local blocks and factor the flux mass block."""
    nel = mass_matrix.shape[0]
    aff00 = aff_mats[element, 0, 0]
    aff01 = aff_mats[element, 0, 1]
    aff10 = aff_mats[element, 1, 0]
    aff11 = aff_mats[element, 1, 1]
    jac = aff_jacs[element]

    for i in range(nel):
        for j in range(nel):
            d0[i, j] = aff11 * d0_reference[i, j] - aff10 * d1_reference[i, j]
            d1[i, j] = -aff01 * d0_reference[i, j] + aff00 * d1_reference[i, j]

            if reaction_is_scalar:
                reaction_value = reaction_scalar * jac * mass_matrix[i, j]
            else:
                weighted_reaction = 0.0
                for k in range(nel):
                    weighted_reaction += reaction_coeffs[element, k] * reaction_triples[k, i, j]
                reaction_value = jac * weighted_reaction

            normal_x_value = 0.0
            normal_y_value = 0.0
            tau_value = reaction_value
            for face in range(3):
                face_mass_value = face_element_mass[face, i, j]
                face_scale = jacs_el_fc[element, face]
                tau_value += tau[element, face] * face_scale * face_mass_value
                normal_x_value += face_scale * normals[element, face, 0] * face_mass_value
                normal_y_value += face_scale * normals[element, face, 1] * face_mass_value

            mn0[i, j] = normal_x_value - d0[i, j]
            mn1[i, j] = normal_y_value - d1[i, j]
            schur_matrix[i, j] = tau_value

            weighted00 = 0.0
            weighted01 = 0.0
            weighted10 = 0.0
            weighted11 = 0.0
            for k in range(nel):
                triple = reaction_triples[k, i, j]
                weighted00 += inv00_coeffs[element, k] * triple
                weighted01 += inv01_coeffs[element, k] * triple
                weighted10 += inv10_coeffs[element, k] * triple
                weighted11 += inv11_coeffs[element, k] * triple
            flux_lu[i, j] = jac * weighted00
            flux_lu[i, nel + j] = jac * weighted01
            flux_lu[nel + i, j] = jac * weighted10
            flux_lu[nel + i, nel + j] = jac * weighted11

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
def _solve_projected_tensor_diffusion_columns(
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
    """Apply the projected tensor mixed local inverse to all RHS columns."""
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
def _assemble_projected_tensor_diffusion_local_columns(
        local_columns,
        element,
        aff_mats,
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        mass_matrix,
        reaction_triples,
        face_element_mass,
        face_element_trace,
        d0_reference,
        d1_reference,
        source_coeffs,
        reaction_coeffs,
        reaction_scalar,
        reaction_is_scalar,
        inv00_coeffs,
        inv01_coeffs,
        inv10_coeffs,
        inv11_coeffs,
):
    """Build solved local columns for projected tensor diffusion assembly."""
    nel = mass_matrix.shape[0]
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

    _build_projected_tensor_diffusion_operator(
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
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        mass_matrix,
        reaction_triples,
        face_element_mass,
        d0_reference,
        d1_reference,
        reaction_coeffs,
        reaction_scalar,
        reaction_is_scalar,
        inv00_coeffs,
        inv01_coeffs,
        inv10_coeffs,
        inv11_coeffs,
    )
    _build_projected_diffusion_rhs_columns(
        rhs0,
        rhs1,
        rhs2,
        element,
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        mass_matrix,
        face_element_trace,
        source_coeffs,
    )
    _solve_projected_tensor_diffusion_columns(
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


@njit(cache=True, inline="always", fastmath=True)
def _diffusion_lift_dot(
        local_columns,
        oriented_lifts,
        loc2oriented_face_coupling,
        normals,
        tau,
        jacs_el_fc,
        element,
        face,
        row_dof,
        column,
        nel,
):
    """Dot one oriented diffusion trace-lift row with a local solution column."""
    oriented_face = loc2oriented_face_coupling[element, face]
    scale = jacs_el_fc[element, face]
    tau_scale = tau[element, face]
    nx = normals[element, face, 0]
    ny = normals[element, face, 1]
    value = 0.0
    for i in range(nel):
        lift = scale * oriented_lifts[oriented_face, row_dof, i]
        value += tau_scale * lift * local_columns[i, column]
        value += nx * lift * local_columns[nel + i, column]
        value += ny * lift * local_columns[2 * nel + i, column]
    return value


@njit(cache=True, parallel=True, fastmath=True)
def assemble_diffusion_trace_system_eliminated_kernel(
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
        jacs_el_fc,
        normals,
        tau,
        edge_mass,
        oriented_lifts,
        local_solver,
        element_boundary_mats,
        source_rhs,
        boundary_trace,
):
    r"""Assemble a reduced diffusion-reaction HDG trace system in COO form.

    The emitted matrix contains only free trace dofs.  Couplings from a free
    row to a prescribed boundary trace column are added to the RHS with the
    same sign convention as generic known-dof elimination.
    """
    num_elements = loc2glob_edge.shape[0]
    nel = local_solver.shape[1] // 3
    ntr = edge_mass.shape[0]
    trace_cols = 3 * ntr

    for element in prange(num_elements):
        local_columns = np.empty((3 * nel, trace_cols + 1), dtype=np.float64)
        _apply_local_solver_columns(local_columns, local_solver, element_boundary_mats, source_rhs, element)

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


@njit(cache=True, parallel=True, fastmath=True)
def assemble_diffusion_trace_rhs_eliminated_kernel(
        rhs_indices,
        rhs_values,
        loc2glob_edge,
        orientations,
        loc2oriented_face_coupling,
        interior_side_index,
        edge_to_solve_edge,
        jacs_el_fc,
        normals,
        tau,
        oriented_lifts,
        local_solver,
        element_boundary_mats,
        source_rhs,
        boundary_trace,
):
    """Assemble only the reduced RHS for a cached diffusion trace matrix."""
    num_elements = loc2glob_edge.shape[0]
    nel = local_solver.shape[1] // 3
    ntr = element_boundary_mats.shape[2] // 3
    trace_cols = 3 * ntr

    for element in prange(num_elements):
        local_columns = np.empty((3 * nel, trace_cols + 1), dtype=np.float64)
        _apply_local_solver_columns(local_columns, local_solver, element_boundary_mats, source_rhs, element)

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

                for col_face in range(3):
                    col_edge = loc2glob_edge[element, col_face]
                    col_solve_edge = edge_to_solve_edge[col_edge]
                    if col_solve_edge >= 0:
                        continue
                    col_is_positive = orientations[element, col_face]
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


@njit(cache=True, parallel=True, fastmath=True)
def assemble_projected_diffusion_trace_system_eliminated_kernel(
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
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        mass_matrix,
        mass_inverse,
        reaction_triples,
        face_element_mass,
        face_element_trace,
        edge_mass,
        oriented_lifts,
        d0_reference,
        d1_reference,
        source_coeffs,
        reaction_coeffs,
        reaction_scalar,
        reaction_is_scalar,
        boundary_trace,
):
    """Fully fused projected diffusion trace assembly with strong trace BCs."""
    num_elements = loc2glob_edge.shape[0]
    nel = mass_matrix.shape[0]
    ntr = edge_mass.shape[0]
    trace_cols = 3 * ntr

    for element in prange(num_elements):
        local_columns = np.empty((3 * nel, trace_cols + 1), dtype=np.float64)
        _assemble_projected_diffusion_local_columns(
            local_columns,
            element,
            aff_mats,
            aff_jacs,
            jacs_el_fc,
            normals,
            tau,
            mass_matrix,
            mass_inverse,
            reaction_triples,
            face_element_mass,
            face_element_trace,
            d0_reference,
            d1_reference,
            source_coeffs,
            reaction_coeffs,
            reaction_scalar,
            reaction_is_scalar,
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


@njit(cache=True, parallel=True, fastmath=True)
def assemble_projected_tensor_diffusion_trace_system_eliminated_kernel(
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
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        mass_matrix,
        reaction_triples,
        face_element_mass,
        face_element_trace,
        edge_mass,
        oriented_lifts,
        d0_reference,
        d1_reference,
        source_coeffs,
        reaction_coeffs,
        reaction_scalar,
        reaction_is_scalar,
        inv00_coeffs,
        inv01_coeffs,
        inv10_coeffs,
        inv11_coeffs,
        boundary_trace,
):
    """Fully fused projected tensor-diffusion trace assembly with strong BCs."""
    num_elements = loc2glob_edge.shape[0]
    nel = mass_matrix.shape[0]
    ntr = edge_mass.shape[0]
    trace_cols = 3 * ntr

    for element in prange(num_elements):
        local_columns = np.empty((3 * nel, trace_cols + 1), dtype=np.float64)
        _assemble_projected_tensor_diffusion_local_columns(
            local_columns,
            element,
            aff_mats,
            aff_jacs,
            jacs_el_fc,
            normals,
            tau,
            mass_matrix,
            reaction_triples,
            face_element_mass,
            face_element_trace,
            d0_reference,
            d1_reference,
            source_coeffs,
            reaction_coeffs,
            reaction_scalar,
            reaction_is_scalar,
            inv00_coeffs,
            inv01_coeffs,
            inv10_coeffs,
            inv11_coeffs,
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


@njit(cache=True, parallel=True, fastmath=True)
def assemble_projected_diffusion_trace_rhs_eliminated_kernel(
        rhs_indices,
        rhs_values,
        loc2glob_edge,
        orientations,
        loc2oriented_face_coupling,
        interior_side_index,
        edge_to_solve_edge,
        aff_mats,
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        mass_matrix,
        mass_inverse,
        reaction_triples,
        face_element_mass,
        face_element_trace,
        oriented_lifts,
        d0_reference,
        d1_reference,
        source_coeffs,
        reaction_coeffs,
        reaction_scalar,
        reaction_is_scalar,
        boundary_trace,
):
    """Fully fused reduced RHS assembly for a cached projected trace matrix."""
    num_elements = loc2glob_edge.shape[0]
    nel = mass_matrix.shape[0]
    ntr = face_element_trace.shape[2]
    trace_cols = 3 * ntr

    for element in prange(num_elements):
        local_columns = np.empty((3 * nel, trace_cols + 1), dtype=np.float64)
        _assemble_projected_diffusion_local_columns(
            local_columns,
            element,
            aff_mats,
            aff_jacs,
            jacs_el_fc,
            normals,
            tau,
            mass_matrix,
            mass_inverse,
            reaction_triples,
            face_element_mass,
            face_element_trace,
            d0_reference,
            d1_reference,
            source_coeffs,
            reaction_coeffs,
            reaction_scalar,
            reaction_is_scalar,
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

                for col_face in range(3):
                    col_edge = loc2glob_edge[element, col_face]
                    col_solve_edge = edge_to_solve_edge[col_edge]
                    if col_solve_edge >= 0:
                        continue
                    col_is_positive = orientations[element, col_face]
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


@njit(cache=True, inline="always", fastmath=True)
def _build_projected_diffusion_reconstruction_rhs(
        rhs0,
        rhs1,
        rhs2,
        element,
        trace,
        loc2glob_edge,
        orientations,
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        mass_matrix,
        face_element_trace,
        source_coeffs,
):
    """Build one local RHS from source coefficients and the solved trace."""
    nel = mass_matrix.shape[0]
    ntr = face_element_trace.shape[2]
    for i in range(nel):
        rhs0[i, 0] = 0.0
        rhs1[i, 0] = 0.0
        rhs2[i, 0] = 0.0

    jac = aff_jacs[element]
    for i in range(nel):
        source_value = 0.0
        for k in range(nel):
            source_value += source_coeffs[element, k] * mass_matrix[k, i]
        rhs0[i, 0] = jac * source_value

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
def reconstruct_projected_diffusion_local_unknowns_kernel(
        local_unknowns,
        trace,
        loc2glob_edge,
        orientations,
        aff_mats,
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        mass_matrix,
        mass_inverse,
        reaction_triples,
        face_element_mass,
        face_element_trace,
        d0_reference,
        d1_reference,
        source_coeffs,
        reaction_coeffs,
        reaction_scalar,
        reaction_is_scalar,
):
    """Recover mixed local unknowns by solving projected local systems."""
    num_elements = loc2glob_edge.shape[0]
    nel = mass_matrix.shape[0]

    for element in prange(num_elements):
        local_columns = np.empty((3 * nel, 1), dtype=np.float64)
        schur_matrix = np.empty((nel, nel), dtype=np.float64)
        d0 = np.empty((nel, nel), dtype=np.float64)
        d1 = np.empty((nel, nel), dtype=np.float64)
        mn0 = np.empty((nel, nel), dtype=np.float64)
        mn1 = np.empty((nel, nel), dtype=np.float64)
        k_d0 = np.empty((nel, nel), dtype=np.float64)
        k_d1 = np.empty((nel, nel), dtype=np.float64)
        rhs0 = np.empty((nel, 1), dtype=np.float64)
        rhs1 = np.empty((nel, 1), dtype=np.float64)
        rhs2 = np.empty((nel, 1), dtype=np.float64)
        red_rhs = np.empty((nel, 1), dtype=np.float64)
        tmp1 = np.empty((nel, 1), dtype=np.float64)
        tmp2 = np.empty((nel, 1), dtype=np.float64)
        pivots = np.empty(nel, dtype=np.int64)

        _build_projected_diffusion_operator(
            schur_matrix,
            d0,
            d1,
            mn0,
            mn1,
            k_d0,
            k_d1,
            element,
            aff_mats,
            aff_jacs,
            jacs_el_fc,
            normals,
            tau,
            mass_matrix,
            mass_inverse,
            reaction_triples,
            face_element_mass,
            d0_reference,
            d1_reference,
            reaction_coeffs,
            reaction_scalar,
            reaction_is_scalar,
        )
        _build_projected_diffusion_reconstruction_rhs(
            rhs0,
            rhs1,
            rhs2,
            element,
            trace,
            loc2glob_edge,
            orientations,
            aff_jacs,
            jacs_el_fc,
            normals,
            tau,
            mass_matrix,
            face_element_trace,
            source_coeffs,
        )
        _solve_projected_diffusion_columns(
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
            tmp1,
            tmp2,
            pivots,
            mass_inverse,
            aff_jacs[element],
        )
        for i in range(3 * nel):
            local_unknowns[element, i] = local_columns[i, 0]


@njit(cache=True, parallel=True, fastmath=True)
def reconstruct_projected_tensor_diffusion_local_unknowns_kernel(
        local_unknowns,
        trace,
        loc2glob_edge,
        orientations,
        aff_mats,
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        mass_matrix,
        reaction_triples,
        face_element_mass,
        face_element_trace,
        d0_reference,
        d1_reference,
        source_coeffs,
        reaction_coeffs,
        reaction_scalar,
        reaction_is_scalar,
        inv00_coeffs,
        inv01_coeffs,
        inv10_coeffs,
        inv11_coeffs,
):
    """Recover mixed local unknowns from projected tensor local systems."""
    num_elements = loc2glob_edge.shape[0]
    nel = mass_matrix.shape[0]

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

        _build_projected_tensor_diffusion_operator(
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
            aff_jacs,
            jacs_el_fc,
            normals,
            tau,
            mass_matrix,
            reaction_triples,
            face_element_mass,
            d0_reference,
            d1_reference,
            reaction_coeffs,
            reaction_scalar,
            reaction_is_scalar,
            inv00_coeffs,
            inv01_coeffs,
            inv10_coeffs,
            inv11_coeffs,
        )
        _build_projected_diffusion_reconstruction_rhs(
            rhs0,
            rhs1,
            rhs2,
            element,
            trace,
            loc2glob_edge,
            orientations,
            aff_jacs,
            jacs_el_fc,
            normals,
            tau,
            mass_matrix,
            face_element_trace,
            source_coeffs,
        )
        _solve_projected_tensor_diffusion_columns(
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


@njit(cache=True, parallel=True, fastmath=True)
def reconstruct_diffusion_local_unknowns_kernel(
        local_unknowns,
        trace,
        loc2glob_edge,
        orientations,
        local_solver,
        element_boundary_mats,
        source_rhs,
):
    """Recover mixed local unknowns from a full trace vector."""
    num_elements = loc2glob_edge.shape[0]
    rows = local_solver.shape[1]
    ntr = element_boundary_mats.shape[2] // 3

    for element in prange(num_elements):
        local_trace = np.empty(3 * ntr, dtype=np.float64)
        for face in range(3):
            edge = loc2glob_edge[element, face]
            is_positive = orientations[element, face]
            for local_dof in range(ntr):
                global_dof = edge * ntr + map_edge_dof_bool(is_positive, local_dof, ntr)
                local_trace[face * ntr + local_dof] = trace[global_dof]

        rhs = np.empty(rows, dtype=np.float64)
        for i in range(rows):
            value = source_rhs[element, i]
            for j in range(3 * ntr):
                value += element_boundary_mats[element, i, j] * local_trace[j]
            rhs[i] = value

        for i in range(rows):
            value = 0.0
            for j in range(rows):
                value += local_solver[element, i, j] * rhs[j]
            local_unknowns[element, i] = value


@njit(cache=True, parallel=True, fastmath=True)
def factor_hdiv_flux_min_distance_postprocess_kernel(
        ainv_constraint_t,
        schur_lu,
        schur_pivots,
        aff_jacs,
        jacs_el_fc,
        normals,
        mass_inverse,
        face_post_trace,
        interior_low_to_post,
):
    """Factor local Schur complements for the constrained flux projection.

    For each element, this prepares the minimum-distance correction operator
    for a degree ``p+1`` vector flux.  Constraints are face normal-flux moments
    against ``P_{p+1}(F)`` plus interior component moments against
    ``P_{p-1}(K)``.  The solve kernel reuses the stored ``A^{-1} C^T`` blocks
    and LU-factored constraint Schur complements.
    """
    num_elements = aff_jacs.shape[0]
    post_el_dof = mass_inverse.shape[0]
    post_edg_dof = face_post_trace.shape[2]
    low_dof = interior_low_to_post.shape[0]
    face_rows = 3 * post_edg_dof
    constraints = face_rows + 2 * low_dof
    vector_dof = 2 * post_el_dof

    for element in prange(num_elements):
        constraint_matrix = np.empty((constraints, vector_dof), dtype=np.float64)
        for row in range(constraints):
            for col in range(vector_dof):
                constraint_matrix[row, col] = 0.0

        for face in range(3):
            scale = jacs_el_fc[element, face]
            nx = normals[element, face, 0]
            ny = normals[element, face, 1]
            for trace_dof in range(post_edg_dof):
                row = face * post_edg_dof + trace_dof
                for j in range(post_el_dof):
                    moment = scale * face_post_trace[face, j, trace_dof]
                    constraint_matrix[row, j] = nx * moment
                    constraint_matrix[row, post_el_dof + j] = ny * moment

        jac = aff_jacs[element]
        for i in range(low_dof):
            row_x = face_rows + i
            row_y = face_rows + low_dof + i
            for j in range(post_el_dof):
                moment = jac * interior_low_to_post[i, j]
                constraint_matrix[row_x, j] = moment
                constraint_matrix[row_y, post_el_dof + j] = moment

        jac_inverse = 1.0 / jac
        for constraint in range(constraints):
            for i in range(post_el_dof):
                value_x = 0.0
                value_y = 0.0
                for k in range(post_el_dof):
                    value_x += mass_inverse[i, k] * constraint_matrix[constraint, k]
                    value_y += mass_inverse[i, k] * constraint_matrix[constraint, post_el_dof + k]
                ainv_constraint_t[element, i, constraint] = jac_inverse * value_x
                ainv_constraint_t[element, post_el_dof + i, constraint] = jac_inverse * value_y

        local_schur = schur_lu[element]
        for row in range(constraints):
            for col in range(constraints):
                value = 0.0
                for j in range(vector_dof):
                    value += constraint_matrix[row, j] * ainv_constraint_t[element, j, col]
                local_schur[row, col] = value
        lu_factor_inplace(local_schur, schur_pivots[element])


@njit(cache=True, parallel=True, fastmath=True)
def solve_hdiv_flux_min_distance_postprocess_kernel(
        flux_coeffs,
        local_unknowns,
        trace,
        loc2glob_edge,
        orientations,
        aff_jacs,
        jacs_el_fc,
        normals,
        tau,
        mass_inverse,
        base_to_post_mass,
        face_base_to_post,
        trace_base_to_post,
        face_post_trace,
        interior_low_to_base,
        interior_low_to_post,
        ainv_constraint_t,
        schur_lu,
        schur_pivots,
):
    """Apply constrained minimum-distance H(div)-type flux post-processing.

    The unconstrained starting point is the ``L2`` projection of the raw HDG
    flux into ``[P_{p+1}]^2``.  The stored Schur factors add the minimum
    mass-norm correction that matches the HDG numerical normal flux on each
    face and preserves low-order interior flux moments.
    """
    num_elements = loc2glob_edge.shape[0]
    base_el_dof = base_to_post_mass.shape[1]
    post_el_dof = base_to_post_mass.shape[0]
    base_edg_dof = trace_base_to_post.shape[0]
    post_edg_dof = trace_base_to_post.shape[1]
    low_dof = interior_low_to_base.shape[0]
    face_rows = 3 * post_edg_dof
    constraints = face_rows + 2 * low_dof

    for element in prange(num_elements):
        q0x = np.empty(post_el_dof, dtype=np.float64)
        q0y = np.empty(post_el_dof, dtype=np.float64)
        projection_rhs_x = np.empty(post_el_dof, dtype=np.float64)
        projection_rhs_y = np.empty(post_el_dof, dtype=np.float64)

        for i in range(post_el_dof):
            value_x = 0.0
            value_y = 0.0
            for j in range(base_el_dof):
                mass = base_to_post_mass[i, j]
                value_x += mass * local_unknowns[element, base_el_dof + j]
                value_y += mass * local_unknowns[element, 2 * base_el_dof + j]
            projection_rhs_x[i] = value_x
            projection_rhs_y[i] = value_y

        for i in range(post_el_dof):
            value_x = 0.0
            value_y = 0.0
            for j in range(post_el_dof):
                value_x += mass_inverse[i, j] * projection_rhs_x[j]
                value_y += mass_inverse[i, j] * projection_rhs_y[j]
            q0x[i] = value_x
            q0y[i] = value_y

        constraint_gap = np.empty((constraints, 1), dtype=np.float64)
        for i in range(constraints):
            constraint_gap[i, 0] = 0.0

        for face in range(3):
            edge = loc2glob_edge[element, face]
            is_positive = orientations[element, face]
            scale = jacs_el_fc[element, face]
            nx = normals[element, face, 0]
            ny = normals[element, face, 1]
            tau_face = tau[element, face]

            for trace_dof in range(post_edg_dof):
                u_face = 0.0
                qx_face = 0.0
                qy_face = 0.0
                for j in range(base_el_dof):
                    moment = face_base_to_post[face, j, trace_dof]
                    u_face += local_unknowns[element, j] * moment
                    qx_face += local_unknowns[element, base_el_dof + j] * moment
                    qy_face += local_unknowns[element, 2 * base_el_dof + j] * moment

                trace_face = 0.0
                for j in range(base_edg_dof):
                    global_dof = edge * base_edg_dof + map_edge_dof_bool(is_positive, j, base_edg_dof)
                    trace_face += trace[global_dof] * trace_base_to_post[j, trace_dof]

                q0_face = 0.0
                for j in range(post_el_dof):
                    moment = face_post_trace[face, j, trace_dof]
                    q0_face += nx * q0x[j] * moment + ny * q0y[j] * moment

                row = face * post_edg_dof + trace_dof
                target = scale * (nx * qx_face + ny * qy_face + tau_face * (u_face - trace_face))
                constraint_gap[row, 0] = target - scale * q0_face

        jac = aff_jacs[element]
        for i in range(low_dof):
            target_x = 0.0
            target_y = 0.0
            current_x = 0.0
            current_y = 0.0
            for j in range(base_el_dof):
                moment = interior_low_to_base[i, j]
                target_x += local_unknowns[element, base_el_dof + j] * moment
                target_y += local_unknowns[element, 2 * base_el_dof + j] * moment
            for j in range(post_el_dof):
                moment = interior_low_to_post[i, j]
                current_x += q0x[j] * moment
                current_y += q0y[j] * moment
            constraint_gap[face_rows + i, 0] = jac * (target_x - current_x)
            constraint_gap[face_rows + low_dof + i, 0] = jac * (target_y - current_y)

        lu_solve_inplace(schur_lu[element], schur_pivots[element], constraint_gap)

        for i in range(post_el_dof):
            value_x = q0x[i]
            value_y = q0y[i]
            for c in range(constraints):
                value_x += ainv_constraint_t[element, i, c] * constraint_gap[c, 0]
                value_y += ainv_constraint_t[element, post_el_dof + i, c] * constraint_gap[c, 0]
            flux_coeffs[0, element, i] = value_x
            flux_coeffs[1, element, i] = value_y


@njit(cache=True, parallel=True, fastmath=True)
def factor_primal_postprocess_kernel(
        primal_lu,
        primal_pivots,
        aff_jacs,
        inv_aff_mats_t,
        stiffness_rr,
        stiffness_rs,
        stiffness_ss,
        mean_post,
):
    """Factor local scalar HDG primal post-processing matrices.

    The element stiffness matrix is assembled from precomputed reference
    stiffness tensors and the affine element metric.  This avoids the expensive
    quadrature loop over every pair of degree ``p+1`` basis functions.
    """
    num_elements = aff_jacs.shape[0]
    post_el_dof = stiffness_rr.shape[0]
    rows = post_el_dof + 1

    for element in prange(num_elements):
        matrix = primal_lu[element]
        for i in range(rows):
            for j in range(rows):
                matrix[i, j] = 0.0

        inv00 = inv_aff_mats_t[element, 0, 0]
        inv01 = inv_aff_mats_t[element, 0, 1]
        inv10 = inv_aff_mats_t[element, 1, 0]
        inv11 = inv_aff_mats_t[element, 1, 1]
        jac = aff_jacs[element]
        metric_rr = inv00 * inv00 + inv10 * inv10
        metric_rs = inv00 * inv01 + inv10 * inv11
        metric_ss = inv01 * inv01 + inv11 * inv11

        for i in range(post_el_dof):
            for j in range(post_el_dof):
                matrix[i, j] = jac * (
                    metric_rr * stiffness_rr[i, j]
                    + metric_rs * stiffness_rs[i, j]
                    + metric_ss * stiffness_ss[i, j]
                )

        for i in range(post_el_dof):
            mean = jac * mean_post[i]
            matrix[i, post_el_dof] = mean
            matrix[post_el_dof, i] = mean

        lu_factor_inplace(matrix, primal_pivots[element])


@njit(cache=True, parallel=True, fastmath=True)
def solve_primal_postprocess_kernel(
        primal_coeffs,
        local_unknowns,
        aff_jacs,
        inv_aff_mats_t,
        weights,
        base_basis_on_post_quads,
        post_grad,
        mean_base,
        inv00_values,
        inv01_values,
        inv10_values,
        inv11_values,
        primal_lu,
        primal_pivots,
):
    """Recover ``u_h^*`` from the HDG mixed flux using local scalar solves.

    The right-hand side represents ``-(kappa^{-1} q_h, grad w)`` plus the mean
    constraint.  Raw flux values are evaluated once per quadrature point and
    reused for all postprocess test functions.
    """
    num_elements = aff_jacs.shape[0]
    base_el_dof = base_basis_on_post_quads.shape[1]
    post_el_dof = post_grad.shape[1]
    quad_count = weights.shape[0]
    rows = post_el_dof + 1

    for element in prange(num_elements):
        rhs = np.empty((rows, 1), dtype=np.float64)
        cqx_values = np.empty(quad_count, dtype=np.float64)
        cqy_values = np.empty(quad_count, dtype=np.float64)
        for i in range(rows):
            rhs[i, 0] = 0.0

        inv00 = inv_aff_mats_t[element, 0, 0]
        inv01 = inv_aff_mats_t[element, 0, 1]
        inv10 = inv_aff_mats_t[element, 1, 0]
        inv11 = inv_aff_mats_t[element, 1, 1]
        jac = aff_jacs[element]

        for q in range(quad_count):
            qx_value = 0.0
            qy_value = 0.0
            for j in range(base_el_dof):
                basis_value = base_basis_on_post_quads[q, j]
                qx_value += local_unknowns[element, base_el_dof + j] * basis_value
                qy_value += local_unknowns[element, 2 * base_el_dof + j] * basis_value
            cqx_values[q] = inv00_values[element, q] * qx_value + inv01_values[element, q] * qy_value
            cqy_values[q] = inv10_values[element, q] * qx_value + inv11_values[element, q] * qy_value

        for i in range(post_el_dof):
            value = 0.0
            for q in range(quad_count):
                gi0 = post_grad[q, i, 0]
                gi1 = post_grad[q, i, 1]
                gix = inv00 * gi0 + inv01 * gi1
                giy = inv10 * gi0 + inv11 * gi1
                value += weights[q] * (cqx_values[q] * gix + cqy_values[q] * giy)
            rhs[i, 0] = -jac * value

        mean_value = 0.0
        for j in range(base_el_dof):
            mean_value += local_unknowns[element, j] * mean_base[j]
        rhs[post_el_dof, 0] = jac * mean_value

        lu_solve_inplace(primal_lu[element], primal_pivots[element], rhs)
        for i in range(post_el_dof):
            primal_coeffs[element, i] = rhs[i, 0]


__all__ = [
    "assemble_diffusion_trace_rhs_eliminated_kernel",
    "assemble_diffusion_trace_system_eliminated_kernel",
    "assemble_projected_diffusion_trace_rhs_eliminated_kernel",
    "assemble_projected_diffusion_trace_system_eliminated_kernel",
    "assemble_projected_tensor_diffusion_trace_system_eliminated_kernel",
    "factor_hdiv_flux_min_distance_postprocess_kernel",
    "factor_primal_postprocess_kernel",
    "reconstruct_projected_diffusion_local_unknowns_kernel",
    "reconstruct_projected_tensor_diffusion_local_unknowns_kernel",
    "reconstruct_diffusion_local_unknowns_kernel",
    "solve_hdiv_flux_min_distance_postprocess_kernel",
    "solve_primal_postprocess_kernel",
]
