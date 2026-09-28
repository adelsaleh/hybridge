"""Fused multithreaded Numba kernels for stationary ADR HDG.

The coefficient inputs are values/moments sampled by the Python adapter.  In
particular, they are deliberately not same-space DG coefficient tables.  This
keeps the element kernel independent of the approximation spaces used for the
source, reaction, and velocity fields.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised when Numba is installed.
    from numba import prange
except ImportError:  # pragma: no cover
    prange = range

from .common import lu_factor_inplace, lu_solve_inplace, map_edge_dof_bool, njit
from .diffusion_mass import factor_diffusion_mass, apply_inverse_diffusion_mass


@njit(cache=True, inline="always")
def _trace_local_dof(positive, dof, edge_dof, orientation_mode):
    """Map a global trace coordinate to its element-local coordinate."""
    if orientation_mode == 1:
        return dof
    return map_edge_dof_bool(positive, dof, edge_dof)


@njit(cache=True, inline="always")
def _trace_sign(positive, dof, orientation_mode):
    """Return the modal orientation sign for one trace coordinate."""
    if orientation_mode == 1 and (not positive) and dof % 2 == 1:
        return -1.0
    return 1.0


@njit(cache=True, inline="always", fastmath=True)
def _build_local_operator(
        schur,
        d0,
        d1,
        mn0,
        mn1,
        kd0,
        kd1,
        element,
        aff_mats,
        aff_jacs,
        jacs_el_fc,
        normals,
        mass_inverse,
        basis,
        gradients,
        weights,
        reaction_values,
        beta_values,
        u_boundary_mass,
        d0_reference,
        d1_reference,
):
    """Build the condensed scalar ADR operator for one element."""
    nel = basis.shape[0]
    nq = weights.shape[0]
    aff00 = aff_mats[element, 0, 0]
    aff01 = aff_mats[element, 0, 1]
    aff10 = aff_mats[element, 1, 0]
    aff11 = aff_mats[element, 1, 1]
    jac = aff_jacs[element]
    inv00 = aff11 / jac
    inv01 = -aff10 / jac
    inv10 = -aff01 / jac
    inv11 = aff00 / jac

    for i in range(nel):
        for j in range(nel):
            d0v = aff11 * d0_reference[i, j] - aff10 * d1_reference[i, j]
            d1v = -aff01 * d0_reference[i, j] + aff00 * d1_reference[i, j]
            d0[i, j] = d0v
            d1[i, j] = d1v

            reaction = 0.0
            advection = 0.0
            for q in range(nq):
                phi_i = basis[i, q]
                phi_j = basis[j, q]
                grad_x = inv00 * gradients[0, i, q] + inv01 * gradients[1, i, q]
                grad_y = inv10 * gradients[0, i, q] + inv11 * gradients[1, i, q]
                weight = weights[q]
                reaction += weight * reaction_values[element, q] * phi_i * phi_j
                advection += weight * phi_j * (
                    beta_values[element, q, 0] * grad_x
                    + beta_values[element, q, 1] * grad_y
                )
            schur[i, j] = u_boundary_mass[element, i, j] + jac * (reaction - advection)

    # M_n-D^T is supplied directly: form it from the oriented physical normal
    # boundary matrices passed in u_boundary_mass's companion arrays later.
    # Here mn0/mn1 temporarily contain M_n and are converted by the caller.


@njit(cache=True, inline="always", fastmath=True)
def _finish_diffusion_condensation(
        schur, d0, d1, mn0, mn1, kd0, kd1, mass_inverse, jac, diffusion,
):
    """Add the mixed diffusive contribution to the scalar Schur block."""
    nel = schur.shape[0]
    for i in range(nel):
        for j in range(nel):
            value0 = 0.0
            value1 = 0.0
            for k in range(nel):
                value0 += mass_inverse[i, k] * d0[k, j]
                value1 += mass_inverse[i, k] * d1[k, j]
            kd0[i, j] = value0
            kd1[i, j] = value1
    jac_inv = diffusion / jac
    for i in range(nel):
        for j in range(nel):
            value0 = 0.0
            value1 = 0.0
            for k in range(nel):
                value0 += mn0[i, k] * kd0[k, j]
                value1 += mn1[i, k] * kd1[k, j]
            schur[i, j] += jac_inv * (value0 + value1)


@njit(cache=True, inline="always", fastmath=True)
def _solve_columns(
        local_columns,
        schur,
        d0,
        d1,
        mn0,
        mn1,
        rhs0,
        rhs1,
        rhs2,
        reduced_rhs,
        tmp1,
        tmp2,
        pivots,
        mass_inverse,
        jac,
        diffusion,
):
    """Solve all local trace/source columns and recover the mixed flux."""
    nel = schur.shape[0]
    ncols = rhs0.shape[1]
    jac_inv = diffusion / jac
    for i in range(nel):
        for col in range(ncols):
            x = 0.0
            y = 0.0
            for k in range(nel):
                x += mass_inverse[i, k] * rhs1[k, col]
                y += mass_inverse[i, k] * rhs2[k, col]
            tmp1[i, col] = x
            tmp2[i, col] = y
    for i in range(nel):
        for col in range(ncols):
            x = 0.0
            y = 0.0
            for k in range(nel):
                x += mn0[i, k] * tmp1[k, col]
                y += mn1[i, k] * tmp2[k, col]
            reduced_rhs[i, col] = rhs0[i, col] + jac_inv * (x + y)

    lu_factor_inplace(schur, pivots)
    lu_solve_inplace(schur, pivots, reduced_rhs)
    for i in range(nel):
        for col in range(ncols):
            u = reduced_rhs[i, col]
            local_columns[i, col] = u
    for i in range(nel):
        for col in range(ncols):
            value0 = 0.0
            value1 = 0.0
            for j in range(nel):
                value0 += d0[i, j] * reduced_rhs[j, col]
                value1 += d1[i, j] * reduced_rhs[j, col]
            tmp1[i, col] = value0 - rhs1[i, col]
            tmp2[i, col] = value1 - rhs2[i, col]
    for i in range(nel):
        for col in range(ncols):
            value0 = 0.0
            value1 = 0.0
            for k in range(nel):
                value0 += mass_inverse[i, k] * tmp1[k, col]
                value1 += mass_inverse[i, k] * tmp2[k, col]
            local_columns[nel + i, col] = jac_inv * value0
            local_columns[2 * nel + i, col] = jac_inv * value1


@njit(cache=True, inline="always", fastmath=True)
def _build_scalar_element_columns(
        local_columns,
        element,
        aff_mats,
        aff_jacs,
        jacs_el_fc,
        normals,
        mass_inverse,
        basis,
        gradients,
        weights,
        reaction_values,
        beta_values,
        u_boundary_mass,
        normal_mass_x,
        normal_mass_y,
        d0_reference,
        d1_reference,
        element_boundary,
        source_rhs,
        diffusion,
):
    """Build and solve every local trace/source response for one element."""
    nel = basis.shape[0]
    ncols = element_boundary.shape[2] + 1
    schur = np.empty((nel, nel), dtype=np.float64)
    d0 = np.empty((nel, nel), dtype=np.float64)
    d1 = np.empty((nel, nel), dtype=np.float64)
    mn0 = np.empty((nel, nel), dtype=np.float64)
    mn1 = np.empty((nel, nel), dtype=np.float64)
    kd0 = np.empty((nel, nel), dtype=np.float64)
    kd1 = np.empty((nel, nel), dtype=np.float64)
    rhs0 = np.empty((nel, ncols), dtype=np.float64)
    rhs1 = np.empty((nel, ncols), dtype=np.float64)
    rhs2 = np.empty((nel, ncols), dtype=np.float64)
    reduced_rhs = np.empty((nel, ncols), dtype=np.float64)
    tmp1 = np.empty((nel, ncols), dtype=np.float64)
    tmp2 = np.empty((nel, ncols), dtype=np.float64)
    pivots = np.empty(nel, dtype=np.int64)

    _build_local_operator(
        schur, d0, d1, mn0, mn1, kd0, kd1, element,
        aff_mats, aff_jacs, jacs_el_fc, normals, mass_inverse,
        basis, gradients, weights, reaction_values, beta_values,
        u_boundary_mass, d0_reference, d1_reference,
    )
    for i in range(nel):
        for j in range(nel):
            mn0[i, j] = normal_mass_x[element, i, j] - d0[i, j]
            mn1[i, j] = normal_mass_y[element, i, j] - d1[i, j]
    _finish_diffusion_condensation(
        schur, d0, d1, mn0, mn1, kd0, kd1, mass_inverse,
        aff_jacs[element], diffusion,
    )

    trace_cols = ncols - 1
    for i in range(nel):
        for col in range(trace_cols):
            rhs0[i, col] = element_boundary[element, i, col]
            rhs1[i, col] = element_boundary[element, nel + i, col]
            rhs2[i, col] = element_boundary[element, 2 * nel + i, col]
        rhs0[i, trace_cols] = source_rhs[element, i]
        rhs1[i, trace_cols] = 0.0
        rhs2[i, trace_cols] = 0.0
    _solve_columns(
        local_columns, schur, d0, d1, mn0, mn1,
        rhs0, rhs1, rhs2, reduced_rhs, tmp1, tmp2, pivots,
        mass_inverse, aff_jacs[element], diffusion,
    )


@njit(cache=True, inline="never")
def _build_element_columns(
        local_columns, element, aff_mats, aff_jacs, jacs_el_fc, normals,
        mass_inverse, basis, gradients, weights, reaction_values, beta_values,
        u_boundary_mass, normal_mass_x, normal_mass_y, d0_reference, d1_reference,
        element_boundary, source_rhs, diffusion_kinds, diffusion_constants,
        inverse_diffusion, status):
    """Dispatch by exact tensor structure while retaining scalar Schur algebra."""
    kind = diffusion_kinds[element]
    constant_index = np.int64(0) if diffusion_constants.shape[0] == 1 else np.int64(element)
    if kind == 0:
        _build_scalar_element_columns(
            local_columns, element, aff_mats, aff_jacs, jacs_el_fc, normals,
            mass_inverse, basis, gradients, weights, reaction_values, beta_values,
            u_boundary_mass, normal_mass_x, normal_mass_y, d0_reference, d1_reference,
            element_boundary, source_rhs, diffusion_constants[constant_index, 0])
        status[element] = 0
        return
    n = basis.shape[0]
    ncols = local_columns.shape[1]
    schur = np.empty((n, n), dtype=np.float64)
    dx = np.empty((n, n), dtype=np.float64)
    dy = np.empty((n, n), dtype=np.float64)
    nx = np.empty((n, n), dtype=np.float64)
    ny = np.empty((n, n), dtype=np.float64)
    # The shared volume builder only uses its first three output arrays.
    _build_local_operator(
        schur, dx, dy, nx, ny, nx, ny, element, aff_mats, aff_jacs,
        jacs_el_fc, normals, mass_inverse, basis, gradients, weights,
        reaction_values, beta_values, u_boundary_mass, d0_reference, d1_reference)
    for i in range(n):
        for j in range(n):
            nx[i, j] = normal_mass_x[element, i, j] - dx[i, j]
            ny[i, j] = normal_mass_y[element, i, j] - dy[i, j]
    rows = 0 if kind <= 2 else (n if kind <= 4 else 2*n)
    columns = 0 if kind <= 2 else (n if kind == 3 else 2*n)
    factor = np.empty((rows, columns), dtype=np.float64)
    flux_pivots = np.empty(2*n if kind == 6 else 0, dtype=np.int64)
    status[element] = 0
    if kind >= 3:
        status[element] = factor_diffusion_mass(
            factor, flux_pivots, inverse_diffusion[element], kind, basis,
            weights, aff_jacs[element])
        if status[element] != 0:
            local_columns[:, :] = np.nan
            return
    derivative = np.empty((2*n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            derivative[i, j] = dx[i, j]
            derivative[n+i, j] = dy[i, j]
    work = np.empty_like(derivative)
    apply_inverse_diffusion_mass(derivative, work, kind, factor, flux_pivots,
                                diffusion_constants[constant_index], mass_inverse, aff_jacs[element])
    for i in range(n):
        for j in range(n):
            value = 0.0
            for k in range(n):
                value += nx[i, k]*derivative[k, j] + ny[i, k]*derivative[n+k, j]
            schur[i, j] += value
    flux_rhs = np.empty((2*n, ncols), dtype=np.float64)
    red = np.empty((n, ncols), dtype=np.float64)
    for i in range(n):
        for col in range(ncols-1):
            red[i, col] = element_boundary[element, i, col]
            flux_rhs[i, col] = element_boundary[element, n+i, col]
            flux_rhs[n+i, col] = element_boundary[element, 2*n+i, col]
        red[i, ncols-1] = source_rhs[element, i]
        flux_rhs[i, ncols-1] = 0.0
        flux_rhs[n+i, ncols-1] = 0.0
    work_rhs = np.empty_like(flux_rhs)
    apply_inverse_diffusion_mass(flux_rhs, work_rhs, kind, factor, flux_pivots,
                                diffusion_constants[constant_index], mass_inverse, aff_jacs[element])
    for i in range(n):
        for col in range(ncols):
            for k in range(n):
                red[i, col] += nx[i, k]*flux_rhs[k, col] + ny[i, k]*flux_rhs[n+k, col]
    pivots = np.empty(n, dtype=np.int64)
    lu_factor_inplace(schur, pivots)
    lu_solve_inplace(schur, pivots, red)
    for i in range(n):
        for col in range(ncols):
            local_columns[i, col] = red[i, col]
            x = -flux_rhs[i, col]
            y = -flux_rhs[n+i, col]
            for j in range(n):
                x += derivative[i, j]*red[j, col]
                y += derivative[n+i, j]*red[j, col]
            local_columns[n+i, col] = x
            local_columns[2*n+i, col] = y


@njit(cache=True, inline="always", fastmath=True)
def _lift_dot(trace_lift, local_columns, element, face, row_dof, column):
    """Apply one total-flux transmission row to a local response column."""
    value = 0.0
    for i in range(local_columns.shape[0]):
        value += trace_lift[element, face, row_dof, i] * local_columns[i, column]
    return value


@njit(cache=True, parallel=True, fastmath=True)
def assemble_projected_adr_trace_system_eliminated_kernel(
        rows,
        cols,
        data,
        rhs_indices,
        rhs_values,
        loc2glob_edge,
        orientations,
        interior_side_index,
        edge_to_solve_edge,
        side_flux_offsets,
        aff_mats,
        aff_jacs,
        jacs_el_fc,
        normals,
        mass_inverse,
        basis,
        gradients,
        weights,
        reaction_values,
        beta_values,
        u_boundary_mass,
        normal_mass_x,
        normal_mass_y,
        d0_reference,
        d1_reference,
        element_boundary,
        trace_lift,
        interior_gamma_mass,
        source_rhs,
        boundary_trace,
        trace_orientation_mode,
        diffusion_kinds, diffusion_constants, inverse_diffusion, status,
):
    """Assemble the all-Dirichlet reduced stationary ADR trace system."""
    num_elements = loc2glob_edge.shape[0]
    nel = basis.shape[0]
    ntr = boundary_trace.shape[1]
    trace_cols = 3 * ntr
    mass_offset = side_flux_offsets[side_flux_offsets.shape[0] - 1]

    for element in prange(num_elements):
        local_columns = np.empty((3 * nel, trace_cols + 1), dtype=np.float64)
        _build_element_columns(
            local_columns, element, aff_mats, aff_jacs, jacs_el_fc, normals,
            mass_inverse, basis, gradients, weights, reaction_values, beta_values,
            u_boundary_mass, normal_mass_x, normal_mass_y,
            d0_reference, d1_reference, element_boundary, source_rhs,
            diffusion_kinds, diffusion_constants, inverse_diffusion, status,
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
            mass_base = mass_offset + side_id * ntr * ntr
            for i in range(ntr):
                for j in range(ntr):
                    out = mass_base + i * ntr + j
                    rows[out] = row_solve_edge * ntr + i
                    cols[out] = row_solve_edge * ntr + j
                    data[out] = interior_gamma_mass[side_id, i, j]
            side_base = side_flux_offsets[side_id]
            for row_dof in range(ntr):
                rhs_value = _lift_dot(
                    trace_lift, local_columns, element, row_face, row_dof, trace_cols
                )
                col_block_pos = 0
                for col_face in range(3):
                    col_edge = loc2glob_edge[element, col_face]
                    col_solve_edge = edge_to_solve_edge[col_edge]
                    positive = orientations[element, col_face]
                    if col_solve_edge >= 0:
                        for col_dof in range(ntr):
                            local_dof = _trace_local_dof(
                                positive, col_dof, ntr, trace_orientation_mode
                            )
                            sign = _trace_sign(positive, col_dof, trace_orientation_mode)
                            column = col_face * ntr + local_dof
                            value = sign * _lift_dot(
                                trace_lift, local_columns, element, row_face, row_dof, column
                            )
                            out = side_base + (col_block_pos * ntr + row_dof) * ntr + col_dof
                            rows[out] = row_solve_edge * ntr + row_dof
                            cols[out] = col_solve_edge * ntr + col_dof
                            data[out] = -value
                        col_block_pos += 1
                    else:
                        for col_dof in range(ntr):
                            local_dof = _trace_local_dof(
                                positive, col_dof, ntr, trace_orientation_mode
                            )
                            sign = _trace_sign(positive, col_dof, trace_orientation_mode)
                            column = col_face * ntr + local_dof
                            value = sign * _lift_dot(
                                trace_lift, local_columns, element, row_face, row_dof, column
                            )
                            rhs_value += value * boundary_trace[col_edge, col_dof]
                rhs_indices[rhs_base + row_dof] = row_solve_edge * ntr + row_dof
                rhs_values[rhs_base + row_dof] = rhs_value


@njit(cache=True, parallel=True, fastmath=True)
def reconstruct_projected_adr_local_unknowns_kernel(
        local_unknowns,
        trace,
        loc2glob_edge,
        orientations,
        aff_mats,
        aff_jacs,
        jacs_el_fc,
        normals,
        mass_inverse,
        basis,
        gradients,
        weights,
        reaction_values,
        beta_values,
        u_boundary_mass,
        normal_mass_x,
        normal_mass_y,
        d0_reference,
        d1_reference,
        element_boundary,
        source_rhs,
        trace_orientation_mode,
        diffusion_kinds, diffusion_constants, inverse_diffusion, status,
):
    """Reconstruct ``[u_h,q_x,q_y]`` from the full trace."""
    num_elements = loc2glob_edge.shape[0]
    nel = basis.shape[0]
    ntr = element_boundary.shape[2] // 3
    for element in prange(num_elements):
        columns = np.empty((3 * nel, 3 * ntr + 1), dtype=np.float64)
        _build_element_columns(
            columns, element, aff_mats, aff_jacs, jacs_el_fc, normals,
            mass_inverse, basis, gradients, weights, reaction_values, beta_values,
            u_boundary_mass, normal_mass_x, normal_mass_y,
            d0_reference, d1_reference, element_boundary, source_rhs,
            diffusion_kinds, diffusion_constants, inverse_diffusion, status,
        )
        source_col = 3 * ntr
        for i in range(3 * nel):
            value = columns[i, source_col]
            for face in range(3):
                edge = loc2glob_edge[element, face]
                positive = orientations[element, face]
                for dof in range(ntr):
                    local_dof = _trace_local_dof(positive, dof, ntr, trace_orientation_mode)
                    sign = _trace_sign(positive, dof, trace_orientation_mode)
                    value += columns[i, face * ntr + local_dof] * sign * trace[edge * ntr + dof]
            local_unknowns[element, i] = value


@njit(cache=True, parallel=True, fastmath=True)
def solve_adr_total_flux_postprocess_kernel(
        flux_coeffs,
        base_flux_coeffs,
        face_constraint_gaps,
        aff_jacs,
        mass_inverse,
        base_to_post_mass,
        interior_low_to_base,
        interior_low_to_post,
        ainv_constraint_t,
        schur_lu,
        schur_pivots,
):
    """Apply the constrained degree-p+1 total-flux recovery.

    Face entries in ``face_constraint_gaps`` are the physical numerical-flux
    moments minus the same moments of the unconstrained projected total flux.
    This lets coefficient/face sampling remain outside the hot solve kernel.
    """
    num_elements = aff_jacs.shape[0]
    base_el_dof = base_to_post_mass.shape[1]
    post_el_dof = base_to_post_mass.shape[0]
    face_rows = face_constraint_gaps.shape[1]
    low_dof = interior_low_to_base.shape[0]
    constraints = face_rows + 2 * low_dof
    for element in prange(num_elements):
        projection_x = np.empty(post_el_dof, dtype=np.float64)
        projection_y = np.empty(post_el_dof, dtype=np.float64)
        q0x = np.empty(post_el_dof, dtype=np.float64)
        q0y = np.empty(post_el_dof, dtype=np.float64)
        for i in range(post_el_dof):
            value_x = 0.0
            value_y = 0.0
            for j in range(base_el_dof):
                moment = base_to_post_mass[i, j]
                value_x += moment * base_flux_coeffs[0, element, j]
                value_y += moment * base_flux_coeffs[1, element, j]
            projection_x[i] = value_x
            projection_y[i] = value_y
        for i in range(post_el_dof):
            value_x = 0.0
            value_y = 0.0
            for j in range(post_el_dof):
                value_x += mass_inverse[i, j] * projection_x[j]
                value_y += mass_inverse[i, j] * projection_y[j]
            q0x[i] = value_x
            q0y[i] = value_y

        gap = np.empty((constraints, 1), dtype=np.float64)
        for i in range(face_rows):
            gap[i, 0] = face_constraint_gaps[element, i]
        jac = aff_jacs[element]
        for i in range(low_dof):
            target_x = 0.0
            target_y = 0.0
            current_x = 0.0
            current_y = 0.0
            for j in range(base_el_dof):
                target_x += interior_low_to_base[i, j] * base_flux_coeffs[0, element, j]
                target_y += interior_low_to_base[i, j] * base_flux_coeffs[1, element, j]
            for j in range(post_el_dof):
                current_x += interior_low_to_post[i, j] * q0x[j]
                current_y += interior_low_to_post[i, j] * q0y[j]
            gap[face_rows + i, 0] = jac * (target_x - current_x)
            gap[face_rows + low_dof + i, 0] = jac * (target_y - current_y)

        lu_solve_inplace(schur_lu[element], schur_pivots[element], gap)
        for i in range(post_el_dof):
            value_x = q0x[i]
            value_y = q0y[i]
            for c in range(constraints):
                value_x += ainv_constraint_t[element, i, c] * gap[c, 0]
                value_y += ainv_constraint_t[element, post_el_dof + i, c] * gap[c, 0]
            flux_coeffs[0, element, i] = value_x
            flux_coeffs[1, element, i] = value_y


@njit(cache=True, parallel=True, fastmath=True)
def solve_adr_rt_total_flux_postprocess_kernel(
        flux_coeffs,
        total_flux_values,
        numerical_normal_flux,
        aff_mats,
        aff_jacs,
        jacs_el_fc,
        normals,
        volume_weights,
        face_weights,
        post_weighted_basis,
        post_mass_inverse,
        base_volume_basis,
        base_face_basis,
        radial_volume,
        radial_face,
        low_volume_basis,
        face_test_basis,
):
    r"""Reconstruct the total flux in ``RT_p=[P_p]^2+x P_p``.

    The square set of Raviart--Thomas degrees of freedom consists of normal
    moments against ``P_p(F)`` on all three faces and component moments against
    ``P_{p-1}(K)`` in the interior.  Their targets are respectively the total
    ADR numerical flux and the directly sampled volume flux
    ``q_h + beta_h*u_h``.  The resulting
    physical RT polynomial is stored in the existing componentwise degree-p+1
    DG space.
    """
    num_elements = aff_jacs.shape[0]
    volume_quads = volume_weights.shape[0]
    face_quads = face_weights.shape[0]
    base_el_dof = base_volume_basis.shape[1]
    post_el_dof = post_mass_inverse.shape[0]
    face_dof = face_test_basis.shape[0]
    low_dof = low_volume_basis.shape[1]
    enrichment_dof = radial_volume.shape[1]
    rt_dof = 2 * base_el_dof + enrichment_dof
    face_rows = 3 * face_dof

    for element in prange(num_elements):
        rt_volume_x = np.empty((rt_dof, volume_quads), dtype=np.float64)
        rt_volume_y = np.empty((rt_dof, volume_quads), dtype=np.float64)
        rt_face_x = np.empty((rt_dof, 3, face_quads), dtype=np.float64)
        rt_face_y = np.empty((rt_dof, 3, face_quads), dtype=np.float64)
        matrix = np.zeros((rt_dof, rt_dof), dtype=np.float64)
        rhs = np.zeros((rt_dof, 1), dtype=np.float64)
        pivots = np.empty(rt_dof, dtype=np.int64)

        jac = aff_jacs[element]
        inv_jac = 1.0 / jac
        a00 = aff_mats[element, 0, 0]
        a01 = aff_mats[element, 0, 1]
        a10 = aff_mats[element, 1, 0]
        a11 = aff_mats[element, 1, 1]

        # Contravariant Piola images of [P_p]^2 and x times homogeneous P_p.
        for j in range(base_el_dof):
            for q in range(volume_quads):
                value = base_volume_basis[q, j] * inv_jac
                rt_volume_x[j, q] = a00 * value
                rt_volume_y[j, q] = a10 * value
                rt_volume_x[base_el_dof + j, q] = a01 * value
                rt_volume_y[base_el_dof + j, q] = a11 * value
            for face in range(3):
                for q in range(face_quads):
                    value = base_face_basis[face, j, q] * inv_jac
                    rt_face_x[j, face, q] = a00 * value
                    rt_face_y[j, face, q] = a10 * value
                    rt_face_x[base_el_dof + j, face, q] = a01 * value
                    rt_face_y[base_el_dof + j, face, q] = a11 * value

        enrichment_offset = 2 * base_el_dof
        for j in range(enrichment_dof):
            basis_index = enrichment_offset + j
            for q in range(volume_quads):
                rx = radial_volume[0, j, q]
                ry = radial_volume[1, j, q]
                rt_volume_x[basis_index, q] = inv_jac * (a00 * rx + a01 * ry)
                rt_volume_y[basis_index, q] = inv_jac * (a10 * rx + a11 * ry)
            for face in range(3):
                for q in range(face_quads):
                    rx = radial_face[0, face, j, q]
                    ry = radial_face[1, face, j, q]
                    rt_face_x[basis_index, face, q] = inv_jac * (a00 * rx + a01 * ry)
                    rt_face_y[basis_index, face, q] = inv_jac * (a10 * rx + a11 * ry)

        # Face-normal RT degrees of freedom.
        for face in range(3):
            face_scale = jacs_el_fc[element, face]
            nx = normals[element, face, 0]
            ny = normals[element, face, 1]
            for i in range(face_dof):
                row = face * face_dof + i
                target = 0.0
                for q in range(face_quads):
                    weight_test = face_scale * face_weights[q] * face_test_basis[i, q]
                    target += weight_test * numerical_normal_flux[element, face, q]
                    for j in range(rt_dof):
                        matrix[row, j] += weight_test * (
                            nx * rt_face_x[j, face, q]
                            + ny * rt_face_y[j, face, q]
                        )
                rhs[row, 0] = target

        # Interior component moments against P_{p-1}(K).
        for i in range(low_dof):
            row_x = face_rows + i
            row_y = face_rows + low_dof + i
            target_x = 0.0
            target_y = 0.0
            for q in range(volume_quads):
                weight_test = jac * volume_weights[q] * low_volume_basis[q, i]
                target_x += weight_test * total_flux_values[0, element, q]
                target_y += weight_test * total_flux_values[1, element, q]
                for j in range(rt_dof):
                    matrix[row_x, j] += weight_test * rt_volume_x[j, q]
                    matrix[row_y, j] += weight_test * rt_volume_y[j, q]
            rhs[row_x, 0] = target_x
            rhs[row_y, 0] = target_y

        lu_factor_inplace(matrix, pivots)
        lu_solve_inplace(matrix, pivots, rhs)

        # Exact componentwise representation in the degree-p+1 storage space.
        projection_x = np.zeros(post_el_dof, dtype=np.float64)
        projection_y = np.zeros(post_el_dof, dtype=np.float64)
        for i in range(post_el_dof):
            for q in range(volume_quads):
                value_x = 0.0
                value_y = 0.0
                for j in range(rt_dof):
                    value_x += rhs[j, 0] * rt_volume_x[j, q]
                    value_y += rhs[j, 0] * rt_volume_y[j, q]
                projection_x[i] += post_weighted_basis[q, i] * value_x
                projection_y[i] += post_weighted_basis[q, i] * value_y
        for i in range(post_el_dof):
            value_x = 0.0
            value_y = 0.0
            for j in range(post_el_dof):
                value_x += post_mass_inverse[i, j] * projection_x[j]
                value_y += post_mass_inverse[i, j] * projection_y[j]
            flux_coeffs[0, element, i] = value_x
            flux_coeffs[1, element, i] = value_y


@njit(cache=True, parallel=True, fastmath=True)
def solve_adr_primal_from_total_flux_postprocess_kernel(
        primal_coeffs,
        base_primal_coeffs,
        total_flux_coeffs,
        aff_jacs,
        inv_aff_mats_t,
        jacs_el_fc,
        normals,
        volume_weights,
        volume_basis,
        volume_gradients,
        face_weights,
        face_basis,
        face_trace_basis,
        beta_volume,
        beta_face,
        tau_total,
        mean_base,
        diffusion,
):
    r"""Solve the coupled degree-p+1 ADR Neumann postprocess on each element.

    The unknowns are ``(u_h^*, q_h^*, phi_h^*, eta)``.  The first three
    blocks satisfy the local mixed HDG equations with total numerical flux

    ``q_h^*.n + (beta.n) phi_h^* + tau (u_h^* - phi_h^*)``.

    Its volume-divergence and boundary-normal moments are prescribed by the
    already reconstructed conservative total flux ``q_h^{T,*}``.  ``eta`` is
    the scalar Neumann multiplier paired with the mean constraint
    ``(u_h^*,1)_K=(u_h,1)_K``.
    """
    num_elements = aff_jacs.shape[0]
    post_el_dof = volume_basis.shape[1]
    post_edg_dof = face_trace_basis.shape[0]
    base_el_dof = base_primal_coeffs.shape[1]
    volume_quads = volume_weights.shape[0]
    face_quads = face_weights.shape[0]
    trace_offset = 3 * post_el_dof
    multiplier = trace_offset + 3 * post_edg_dof
    rows = multiplier + 1

    for element in prange(num_elements):
        matrix = np.zeros((rows, rows), dtype=np.float64)
        rhs = np.zeros((rows, 1), dtype=np.float64)
        pivots = np.empty(rows, dtype=np.int64)
        jac = aff_jacs[element]
        inv00 = inv_aff_mats_t[element, 0, 0]
        inv01 = inv_aff_mats_t[element, 0, 1]
        inv10 = inv_aff_mats_t[element, 1, 0]
        inv11 = inv_aff_mats_t[element, 1, 1]

        # Volume terms in the constitutive and conservation equations.
        for q in range(volume_quads):
            scale = jac * volume_weights[q]
            total_x = 0.0
            total_y = 0.0
            for j in range(post_el_dof):
                basis_j = volume_basis[q, j]
                total_x += total_flux_coeffs[0, element, j] * basis_j
                total_y += total_flux_coeffs[1, element, j] * basis_j

            for i in range(post_el_dof):
                gi0 = volume_gradients[q, i, 0]
                gi1 = volume_gradients[q, i, 1]
                gix = inv00 * gi0 + inv01 * gi1
                giy = inv10 * gi0 + inv11 * gi1

                # First-order constitutive equations, q^*=-kappa grad(u^*).
                for j in range(post_el_dof):
                    basis_j = volume_basis[q, j]
                    mass = scale * volume_basis[q, i] * basis_j / diffusion
                    matrix[i, post_el_dof + j] += mass
                    matrix[post_el_dof + i, 2 * post_el_dof + j] += mass
                    matrix[i, j] -= scale * gix * basis_j
                    matrix[post_el_dof + i, j] -= scale * giy * basis_j

                conservation_row = 2 * post_el_dof + i
                for j in range(post_el_dof):
                    basis_j = volume_basis[q, j]
                    matrix[conservation_row, post_el_dof + j] -= scale * gix * basis_j
                    matrix[conservation_row, 2 * post_el_dof + j] -= scale * giy * basis_j
                    matrix[conservation_row, j] -= scale * basis_j * (
                        beta_volume[element, q, 0] * gix
                        + beta_volume[element, q, 1] * giy
                    )
                rhs[conservation_row, 0] -= scale * (total_x * gix + total_y * giy)
                matrix[conservation_row, multiplier] += scale * volume_basis[q, i]

        # Boundary terms and the prescribed normal moments of q_h^{T,*}.
        for face in range(3):
            face_scale = jacs_el_fc[element, face]
            nx = normals[element, face, 0]
            ny = normals[element, face, 1]
            face_trace_offset = trace_offset + face * post_edg_dof
            for q in range(face_quads):
                scale = face_scale * face_weights[q]
                beta_n = (
                    beta_face[element, face, q, 0] * nx
                    + beta_face[element, face, q, 1] * ny
                )
                tau = tau_total[element, face, q]
                total_normal = 0.0
                for j in range(post_el_dof):
                    basis_j = face_basis[face, j, q]
                    total_normal += (
                        nx * total_flux_coeffs[0, element, j]
                        + ny * total_flux_coeffs[1, element, j]
                    ) * basis_j

                for i in range(post_el_dof):
                    test_i = face_basis[face, i, q]
                    conservation_row = 2 * post_el_dof + i
                    for j in range(post_el_dof):
                        trial_j = face_basis[face, j, q]
                        boundary_mass = scale * test_i * trial_j
                        matrix[conservation_row, post_el_dof + j] += nx * boundary_mass
                        matrix[conservation_row, 2 * post_el_dof + j] += ny * boundary_mass
                        matrix[conservation_row, j] += tau * boundary_mass
                    for a in range(post_edg_dof):
                        trace_trial = face_trace_basis[a, q]
                        matrix[conservation_row, face_trace_offset + a] += (
                            scale * test_i * trace_trial * (beta_n - tau)
                        )
                    rhs[conservation_row, 0] += scale * total_normal * test_i

                for i in range(post_el_dof):
                    test_i = face_basis[face, i, q]
                    for a in range(post_edg_dof):
                        trace_trial = face_trace_basis[a, q]
                        trace_mass = scale * test_i * trace_trial
                        matrix[i, face_trace_offset + a] += nx * trace_mass
                        matrix[post_el_dof + i, face_trace_offset + a] += ny * trace_mass

                for a in range(post_edg_dof):
                    boundary_row = trace_offset + face * post_edg_dof + a
                    test_a = face_trace_basis[a, q]
                    for j in range(post_el_dof):
                        trial_j = face_basis[face, j, q]
                        boundary_mass = scale * test_a * trial_j
                        matrix[boundary_row, post_el_dof + j] += nx * boundary_mass
                        matrix[boundary_row, 2 * post_el_dof + j] += ny * boundary_mass
                        matrix[boundary_row, j] += tau * boundary_mass
                    for b in range(post_edg_dof):
                        matrix[boundary_row, face_trace_offset + b] += (
                            scale * test_a * face_trace_basis[b, q] * (beta_n - tau)
                        )
                    rhs[boundary_row, 0] += scale * total_normal * test_a

        # Replace the Neumann null mode by the raw-primal element mean.
        mean_value = 0.0
        for j in range(base_el_dof):
            mean_value += base_primal_coeffs[element, j] * mean_base[j]
        rhs[multiplier, 0] = jac * mean_value
        for j in range(post_el_dof):
            mean = 0.0
            for q in range(volume_quads):
                mean += volume_weights[q] * volume_basis[q, j]
            matrix[multiplier, j] = jac * mean

        lu_factor_inplace(matrix, pivots)
        lu_solve_inplace(matrix, pivots, rhs)
        for i in range(post_el_dof):
            primal_coeffs[element, i] = rhs[i, 0]


__all__ = [
    "assemble_projected_adr_trace_system_eliminated_kernel",
    "reconstruct_projected_adr_local_unknowns_kernel",
    "solve_adr_primal_from_total_flux_postprocess_kernel",
    "solve_adr_rt_total_flux_postprocess_kernel",
    "solve_adr_total_flux_postprocess_kernel",
]
