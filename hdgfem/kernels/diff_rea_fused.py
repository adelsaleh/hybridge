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


__all__ = [
    "assemble_diffusion_trace_rhs_eliminated_kernel",
    "assemble_diffusion_trace_system_eliminated_kernel",
    "assemble_projected_diffusion_trace_rhs_eliminated_kernel",
    "assemble_projected_diffusion_trace_system_eliminated_kernel",
    "reconstruct_projected_diffusion_local_unknowns_kernel",
    "reconstruct_diffusion_local_unknowns_kernel",
]
