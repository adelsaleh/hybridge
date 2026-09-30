"""Fused multithreaded Numba kernels for stationary ADR HDG.

The coefficient inputs are values/moments sampled by the Python adapter.  In
particular, they are deliberately not same-space DG coefficient tables.  This
keeps the element kernel independent of the approximation spaces used for the
source, reaction, and velocity fields.

The face tables of ``prepare_adr_data(dense_local_matrices=True)`` (boundary
mass, normal masses, element-boundary coupling, trace lift and interior trace
masses) are built per element from the face samples ``tau_total`` and
``gamma`` by :func:`_element_face_tables`, so no ``(K, ...)`` dense table is
materialized on the host.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised when Numba is installed.
    from numba import prange
except ImportError:  # pragma: no cover
    prange = range

from hdgfem.hdg.numba_common import (
    lu_factor_inplace,
    lu_solve_inplace,
)
from hdgfem.runtime.optional import njit
from hdgfem.mixed.numba_common import _finish_diffusion_condensation, _solve_mixed_columns

from hdgfem.mixed.numba_diffusion_mass import (
    factor_diffusion_mass,
    apply_inverse_diffusion_mass,
)
from hdgfem.hdg.numba_common import _trace_local_dof, _trace_orientation_sign


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
            schur[i, j] = u_boundary_mass[i, j] + jac * (reaction - advection)

    # M_n-D^T is supplied directly: form it from the oriented physical normal
    # boundary matrices passed in u_boundary_mass's companion arrays later.
    # Here mn0/mn1 temporarily contain M_n and are converted by the caller.


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
    ncols = element_boundary.shape[1] + 1
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
            mn0[i, j] = normal_mass_x[i, j] - d0[i, j]
            mn1[i, j] = normal_mass_y[i, j] - d1[i, j]
    _finish_diffusion_condensation(
        schur, d0, d1, mn0, mn1, kd0, kd1, mass_inverse,
        aff_jacs[element], diffusion,
    )

    trace_cols = ncols - 1
    for i in range(nel):
        for col in range(trace_cols):
            rhs0[i, col] = element_boundary[i, col]
            rhs1[i, col] = element_boundary[nel + i, col]
            rhs2[i, col] = element_boundary[2 * nel + i, col]
        rhs0[i, trace_cols] = source_rhs[element, i]
        rhs1[i, trace_cols] = 0.0
        rhs2[i, trace_cols] = 0.0
    _solve_mixed_columns(
        local_columns, schur, d0, d1, mn0, mn1,
        rhs0, rhs1, rhs2, reduced_rhs, tmp1, tmp2, pivots,
        mass_inverse, aff_jacs[element], diffusion=diffusion,
    )


@njit(cache=True, inline="never")
def _build_element_columns(
        local_columns, element, aff_mats, aff_jacs, jacs_el_fc, normals,
        mass_inverse, basis, gradients, weights, reaction_values, beta_values,
        u_boundary_mass, normal_mass_x, normal_mass_y, d0_reference, d1_reference,
        element_boundary, source_rhs, diffusion_kinds, diffusion_constants,
        inverse_diffusion, status):
    """Dispatch by exact tensor structure while retaining scalar Schur algebra.

    ``u_boundary_mass``, ``normal_mass_x/y`` and ``element_boundary`` are this
    element's tables from :func:`_element_face_tables`.
    """
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
            nx[i, j] = normal_mass_x[i, j] - dx[i, j]
            ny[i, j] = normal_mass_y[i, j] - dy[i, j]
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
            red[i, col] = element_boundary[i, col]
            flux_rhs[i, col] = element_boundary[n+i, col]
            flux_rhs[n+i, col] = element_boundary[2*n+i, col]
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
def _lift_dot(trace_lift, local_columns, face, row_dof, column):
    """Apply one total-flux transmission row to a local response column."""
    value = 0.0
    for i in range(local_columns.shape[0]):
        value += trace_lift[face, row_dof, i] * local_columns[i, column]
    return value


@njit(cache=True, inline="always", fastmath=True)
def _element_face_tables(
        element, orientations, jacs_el_fc, normals, tau_total, gamma,
        face_basis, weighted_face_basis, weighted_edge_basis, oriented_edge_basis,
        face_weights, oriented_restriction, face_mass,
        u_boundary_mass, normal_mass_x, normal_mass_y, element_boundary, trace_lift,
):
    """One element's face tables, as ``prepare_adr_data(dense_local_matrices=True)`` builds them.

    ``u_boundary_mass`` is the tau-weighted face mass, ``normal_mass_x/y`` the
    normal-weighted face masses, ``element_boundary`` the ``(3*nel, 3*ntr)``
    coupling (gamma rows in local face orientation, then the normal rows) and
    ``trace_lift`` the ``(3, ntr, 3*nel)`` lift in global edge orientation
    (tau columns, then the normal columns). ``oriented_restriction`` is
    ``DGTraceSpace.face_trace_test_element_trial_oriented`` and ``face_mass``
    ``ReferenceElementData.face_element_test_element_trial``.
    """
    nel = face_basis.shape[1]
    nq = face_weights.shape[0]
    ntr = weighted_edge_basis.shape[0]
    tau_basis = np.empty((nel, nq), dtype=np.float64)
    for i in range(nel):
        for j in range(nel):
            u_boundary_mass[i, j] = 0.0
            normal_mass_x[i, j] = 0.0
            normal_mass_y[i, j] = 0.0
    for f in range(3):
        jac = jacs_el_fc[element, f]
        nx = normals[element, f, 0] * jac
        ny = normals[element, f, 1] * jac
        positive = orientations[element, f]
        orientation = 0 if positive else 1
        coupling = f if positive else f + 3
        for i in range(nel):
            for q in range(nq):
                tau_basis[i, q] = jac * tau_total[element, f, q] * face_basis[f, i, q]
        for i in range(nel):
            for j in range(nel):
                value = 0.0
                for q in range(nq):
                    value += tau_basis[i, q] * weighted_face_basis[f, j, q]
                u_boundary_mass[i, j] += value
                normal_mass_x[i, j] += nx * face_mass[f, i, j]
                normal_mass_y[i, j] += ny * face_mass[f, i, j]
            for a in range(ntr):
                value = 0.0
                for q in range(nq):
                    value += gamma[element, f, q] * face_basis[f, i, q] * weighted_edge_basis[a, q]
                column = f * ntr + a
                element_boundary[i, column] = jac * value
                element_boundary[nel + i, column] = nx * oriented_restriction[f, a, i]
                element_boundary[2 * nel + i, column] = ny * oriented_restriction[f, a, i]
        for a in range(ntr):
            for i in range(nel):
                value = 0.0
                for q in range(nq):
                    value += tau_basis[i, q] * oriented_edge_basis[orientation, a, q] * face_weights[q]
                trace_lift[f, a, i] = value
                trace_lift[f, a, nel + i] = nx * oriented_restriction[coupling, a, i]
                trace_lift[f, a, 2 * nel + i] = ny * oriented_restriction[coupling, a, i]


@njit(cache=True, inline="always", fastmath=True)
def _side_trace_mass(element, face, orientations, jacs_el_fc, gamma, oriented_edge_basis, face_weights, out):
    """Interior trace mass of one element side, weighted by ``gamma = tau - beta.n``."""
    ntr = out.shape[0]
    nq = face_weights.shape[0]
    orientation = 0 if orientations[element, face] else 1
    jac = jacs_el_fc[element, face]
    for a in range(ntr):
        for b in range(ntr):
            value = 0.0
            for q in range(nq):
                value += (gamma[element, face, q] * oriented_edge_basis[orientation, a, q]
                          * oriented_edge_basis[orientation, b, q] * face_weights[q])
            out[a, b] = jac * value


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
        tau_total,
        gamma,
        face_basis,
        weighted_face_basis,
        weighted_edge_basis,
        oriented_edge_basis,
        face_weights,
        oriented_restriction,
        face_mass,
        d0_reference,
        d1_reference,
        source_rhs,
        boundary_trace,
        trace_orientation_mode,
        diffusion_kinds, diffusion_constants, inverse_diffusion, status,
        column_cache, store_columns,
):
    """Assemble the all-Dirichlet reduced stationary ADR trace system.

    Face tables are built per element from ``tau_total`` and ``gamma``
    (:func:`_element_face_tables`). With ``store_columns`` every element's
    local solution columns ``(3*nel, 3*ntr + 1)`` are written to
    ``column_cache`` for :func:`reconstruct_adr_from_local_columns_kernel`.
    """
    num_elements = loc2glob_edge.shape[0]
    nel = basis.shape[0]
    ntr = boundary_trace.shape[1]
    trace_cols = 3 * ntr
    mass_offset = side_flux_offsets[side_flux_offsets.shape[0] - 1]

    for element in prange(num_elements):
        if store_columns:
            local_columns = column_cache[element]
        else:
            local_columns = np.empty((3 * nel, trace_cols + 1), dtype=np.float64)
        u_boundary_mass = np.empty((nel, nel), dtype=np.float64)
        normal_mass_x = np.empty((nel, nel), dtype=np.float64)
        normal_mass_y = np.empty((nel, nel), dtype=np.float64)
        element_boundary = np.empty((3 * nel, trace_cols), dtype=np.float64)
        trace_lift = np.empty((3, ntr, 3 * nel), dtype=np.float64)
        side_mass = np.empty((ntr, ntr), dtype=np.float64)
        _element_face_tables(
            element, orientations, jacs_el_fc, normals, tau_total, gamma,
            face_basis, weighted_face_basis, weighted_edge_basis, oriented_edge_basis,
            face_weights, oriented_restriction, face_mass,
            u_boundary_mass, normal_mass_x, normal_mass_y, element_boundary, trace_lift,
        )
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
            _side_trace_mass(element, row_face, orientations, jacs_el_fc, gamma,
                             oriented_edge_basis, face_weights, side_mass)
            for i in range(ntr):
                for j in range(ntr):
                    out = mass_base + i * ntr + j
                    rows[out] = row_solve_edge * ntr + i
                    cols[out] = row_solve_edge * ntr + j
                    data[out] = side_mass[i, j]
            side_base = side_flux_offsets[side_id]
            for row_dof in range(ntr):
                rhs_value = _lift_dot(
                    trace_lift, local_columns, row_face, row_dof, trace_cols
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
                            sign = _trace_orientation_sign(positive, col_dof, trace_orientation_mode)
                            column = col_face * ntr + local_dof
                            value = sign * _lift_dot(
                                trace_lift, local_columns, row_face, row_dof, column
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
                            sign = _trace_orientation_sign(positive, col_dof, trace_orientation_mode)
                            column = col_face * ntr + local_dof
                            value = sign * _lift_dot(
                                trace_lift, local_columns, row_face, row_dof, column
                            )
                            rhs_value += value * boundary_trace[col_edge, col_dof]
                rhs_indices[rhs_base + row_dof] = row_solve_edge * ntr + row_dof
                rhs_values[rhs_base + row_dof] = rhs_value


@njit(cache=True, parallel=True, fastmath=True)
def reconstruct_adr_from_local_columns_kernel(
        local_unknowns, trace, columns, loc2glob_edge, orientations, trace_orientation_mode):
    """Reconstruct ``[u_h,q_x,q_y]`` from the assembly's stored local solution columns."""
    num_elements = loc2glob_edge.shape[0]
    rows = columns.shape[1]
    ntr = (columns.shape[2] - 1) // 3
    source_col = 3 * ntr
    for element in prange(num_elements):
        for i in range(rows):
            value = columns[element, i, source_col]
            for face in range(3):
                edge = loc2glob_edge[element, face]
                positive = orientations[element, face]
                for dof in range(ntr):
                    local_dof = _trace_local_dof(positive, dof, ntr, trace_orientation_mode)
                    sign = _trace_orientation_sign(positive, dof, trace_orientation_mode)
                    value += columns[element, i, face * ntr + local_dof] * sign * trace[edge * ntr + dof]
            local_unknowns[element, i] = value


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
        tau_total,
        gamma,
        face_basis,
        weighted_face_basis,
        weighted_edge_basis,
        oriented_edge_basis,
        face_weights,
        oriented_restriction,
        face_mass,
        d0_reference,
        d1_reference,
        source_rhs,
        trace_orientation_mode,
        diffusion_kinds, diffusion_constants, inverse_diffusion, status,
):
    """Reconstruct ``[u_h,q_x,q_y]`` from the full trace (face tables built per element)."""
    num_elements = loc2glob_edge.shape[0]
    nel = basis.shape[0]
    ntr = weighted_edge_basis.shape[0]
    for element in prange(num_elements):
        columns = np.empty((3 * nel, 3 * ntr + 1), dtype=np.float64)
        u_boundary_mass = np.empty((nel, nel), dtype=np.float64)
        normal_mass_x = np.empty((nel, nel), dtype=np.float64)
        normal_mass_y = np.empty((nel, nel), dtype=np.float64)
        element_boundary = np.empty((3 * nel, 3 * ntr), dtype=np.float64)
        trace_lift = np.empty((3, ntr, 3 * nel), dtype=np.float64)
        _element_face_tables(
            element, orientations, jacs_el_fc, normals, tau_total, gamma,
            face_basis, weighted_face_basis, weighted_edge_basis, oriented_edge_basis,
            face_weights, oriented_restriction, face_mass,
            u_boundary_mass, normal_mass_x, normal_mass_y, element_boundary, trace_lift,
        )
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
                    sign = _trace_orientation_sign(positive, dof, trace_orientation_mode)
                    value += columns[i, face * ntr + local_dof] * sign * trace[edge * ntr + dof]
            local_unknowns[element, i] = value


__all__ = [
    "assemble_projected_adr_trace_system_eliminated_kernel",
    "reconstruct_adr_from_local_columns_kernel",
    "reconstruct_projected_adr_local_unknowns_kernel",
]
