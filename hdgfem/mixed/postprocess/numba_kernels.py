"""Numba kernels for mixed HDG (DR and ADR) p+1 postprocessing.

Flux recovery is shared by DR and ADR (DR is ADR with beta = 0):

* ``factor_hdiv_flux_min_distance_postprocess_kernel`` factors the constraint
  Schur complement once per mesh and order.
* l2_closest recovery shares the projection and constrained correction
  (inline helpers ``_project_flux_to_post``, ``_apply_min_distance_correction``)
  between two entry kernels: ``solve_flux_min_distance_postprocess_kernel``
  takes host-sampled face gaps (ADR total flux), and
  ``solve_diffusion_flux_min_distance_postprocess_kernel`` computes DR gaps in
  registers from exact moment tables (β = 0, face-constant tau).
* ``solve_rt_flux_postprocess_kernel`` performs RT_p moment matching.

The primal recoveries differ between the equations and stay separate:
``factor_primal_postprocess_kernel``/``solve_primal_postprocess_kernel`` (DR,
Stenberg-type on the raw flux) and
``solve_adr_primal_from_total_flux_postprocess_kernel`` (ADR, local mixed
problem on the recovered flux).
"""

from __future__ import annotations

import numpy as np
from hdgfem.hdg.numba_common import (
    _trace_local_dof,
    _trace_orientation_sign,
    lu_factor_inplace,
    lu_solve_inplace,
)
from hdgfem.runtime.optional import njit, prange


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
def solve_hdiv_flux_primal_reference_min_distance_postprocess_kernel(
        flux_coeffs,
        local_unknowns,
        trace,
        primal_coeffs,
        loc2glob_edge,
        orientations,
        aff_jacs,
        inv_aff_mats_t,
        jacs_el_fc,
        normals,
        tau,
        post_grad_project_r,
        post_grad_project_s,
        face_base_to_post,
        trace_base_to_post,
        face_post_trace,
        interior_low_to_base,
        interior_low_to_post,
        ainv_constraint_t,
        schur_lu,
        schur_pivots,
        trace_orientation_mode,
):
    """Apply constrained H(div)-type flux post-processing using ``-grad(u*)``.

    This variant is used for identity diffusion when primal post-processing is
    also requested.  The unconstrained starting point is the degree ``p+1`` L2
    projection of ``-grad(u_h_star)``.  The stored Schur factors then add the
    minimum mass-norm correction that enforces the same HDG numerical normal
    flux moments and low-order raw-flux interior moments as the generic flux
    postprocessor.
    """
    num_elements = loc2glob_edge.shape[0]
    base_el_dof = face_base_to_post.shape[1]
    post_el_dof = post_grad_project_r.shape[0]
    base_edg_dof = trace_base_to_post.shape[0]
    post_edg_dof = trace_base_to_post.shape[1]
    low_dof = interior_low_to_base.shape[0]
    face_rows = 3 * post_edg_dof
    constraints = face_rows + 2 * low_dof

    for element in prange(num_elements):
        q0x = np.empty(post_el_dof, dtype=np.float64)
        q0y = np.empty(post_el_dof, dtype=np.float64)
        grad_r = np.empty(post_el_dof, dtype=np.float64)
        grad_s = np.empty(post_el_dof, dtype=np.float64)

        for i in range(post_el_dof):
            value_r = 0.0
            value_s = 0.0
            for j in range(post_el_dof):
                value_r += post_grad_project_r[i, j] * primal_coeffs[element, j]
                value_s += post_grad_project_s[i, j] * primal_coeffs[element, j]
            grad_r[i] = value_r
            grad_s[i] = value_s

        inv00 = inv_aff_mats_t[element, 0, 0]
        inv01 = inv_aff_mats_t[element, 0, 1]
        inv10 = inv_aff_mats_t[element, 1, 0]
        inv11 = inv_aff_mats_t[element, 1, 1]
        for i in range(post_el_dof):
            q0x[i] = -(inv00 * grad_r[i] + inv01 * grad_s[i])
            q0y[i] = -(inv10 * grad_r[i] + inv11 * grad_s[i])

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
                    global_dof = edge * base_edg_dof + _trace_local_dof(
                        is_positive,
                        j,
                        base_edg_dof,
                        trace_orientation_mode,
                    )
                    sign = _trace_orientation_sign(is_positive, j, trace_orientation_mode)
                    trace_face += sign * trace[global_dof] * trace_base_to_post[j, trace_dof]

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


@njit(cache=True, inline="always", fastmath=True)
def _project_flux_to_post(bx, by, base_to_post_mass, mass_inverse, q0x, q0y):
    """Unconstrained ``[P_{p+1}]^2`` projection ``q0`` of one element's degree-p flux."""
    post_el_dof = base_to_post_mass.shape[0]
    base_el_dof = base_to_post_mass.shape[1]
    rhs_x = np.empty(post_el_dof, dtype=np.float64)
    rhs_y = np.empty(post_el_dof, dtype=np.float64)
    for i in range(post_el_dof):
        value_x = 0.0
        value_y = 0.0
        for j in range(base_el_dof):
            mass = base_to_post_mass[i, j]
            value_x += mass * bx[j]
            value_y += mass * by[j]
        rhs_x[i] = value_x
        rhs_y[i] = value_y
    for i in range(post_el_dof):
        value_x = 0.0
        value_y = 0.0
        for j in range(post_el_dof):
            value_x += mass_inverse[i, j] * rhs_x[j]
            value_y += mass_inverse[i, j] * rhs_y[j]
        q0x[i] = value_x
        q0y[i] = value_y


@njit(cache=True, inline="always", fastmath=True)
def _apply_min_distance_correction(
        element, bx, by, q0x, q0y, gap, face_rows, jac,
        interior_low_to_base, interior_low_to_post,
        ainv_constraint_t, schur_lu, schur_pivots, flux_coeffs,
):
    """Close one element's constraint gaps with the minimum mass-norm correction.

    ``gap[:face_rows]`` must already hold the face rows (numerical-flux moments
    minus those of ``q0``). This fills the low-order interior rows, which
    preserve the interior moments of the degree-p flux ``(bx, by)``, solves
    with the stored Schur factors and writes ``q0 + A^{-1} C^T gap``.
    """
    base_el_dof = bx.shape[0]
    post_el_dof = q0x.shape[0]
    low_dof = interior_low_to_base.shape[0]
    constraints = face_rows + 2 * low_dof
    for i in range(low_dof):
        target_x = 0.0
        target_y = 0.0
        current_x = 0.0
        current_y = 0.0
        for j in range(base_el_dof):
            target_x += interior_low_to_base[i, j] * bx[j]
            target_y += interior_low_to_base[i, j] * by[j]
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
def solve_flux_min_distance_postprocess_kernel(
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
    """l2_closest flux recovery from sampled face gaps (ADR total flux).

    ``base_flux_coeffs`` ``(2, K, base_el_dof)`` is the degree-p flux (for ADR
    the projection of ``q_h + beta u_h``) and ``face_constraint_gaps``
    ``(K, 3 * post_edg_dof)`` the numerical-flux face moments minus those of
    its ``[P_{p+1}]^2`` projection, sampled on the host.
    """
    num_elements = aff_jacs.shape[0]
    post_el_dof = base_to_post_mass.shape[0]
    face_rows = face_constraint_gaps.shape[1]
    constraints = face_rows + 2 * interior_low_to_base.shape[0]
    for element in prange(num_elements):
        q0x = np.empty(post_el_dof, dtype=np.float64)
        q0y = np.empty(post_el_dof, dtype=np.float64)
        bx = base_flux_coeffs[0, element]
        by = base_flux_coeffs[1, element]
        _project_flux_to_post(bx, by, base_to_post_mass, mass_inverse, q0x, q0y)
        gap = np.empty((constraints, 1), dtype=np.float64)
        for i in range(face_rows):
            gap[i, 0] = face_constraint_gaps[element, i]
        _apply_min_distance_correction(
            element, bx, by, q0x, q0y, gap, face_rows, aff_jacs[element],
            interior_low_to_base, interior_low_to_post,
            ainv_constraint_t, schur_lu, schur_pivots, flux_coeffs,
        )


@njit(cache=True, parallel=True, fastmath=True)
def solve_diffusion_flux_min_distance_postprocess_kernel(
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
        trace_orientation_mode,
):
    """l2_closest flux recovery for diffusion-reaction (β = 0 fast path).

    Same projection and correction as ``solve_flux_min_distance_postprocess_kernel``
    (shared inline helpers), with the base flux ``q_h`` read from the local
    unknowns and the face gaps ``q_h.n + tau (u_h - lambda_h)`` computed in
    registers from exact moment tables and face-constant ``tau`` ``(K, 3)``.
    """
    num_elements = loc2glob_edge.shape[0]
    base_el_dof = base_to_post_mass.shape[1]
    post_el_dof = base_to_post_mass.shape[0]
    base_edg_dof = trace_base_to_post.shape[0]
    post_edg_dof = trace_base_to_post.shape[1]
    face_rows = 3 * post_edg_dof
    constraints = face_rows + 2 * interior_low_to_base.shape[0]

    for element in prange(num_elements):
        q0x = np.empty(post_el_dof, dtype=np.float64)
        q0y = np.empty(post_el_dof, dtype=np.float64)
        bx = local_unknowns[element, base_el_dof:2 * base_el_dof]
        by = local_unknowns[element, 2 * base_el_dof:3 * base_el_dof]
        _project_flux_to_post(bx, by, base_to_post_mass, mass_inverse, q0x, q0y)

        gap = np.empty((constraints, 1), dtype=np.float64)
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
                    qx_face += bx[j] * moment
                    qy_face += by[j] * moment
                trace_face = 0.0
                for j in range(base_edg_dof):
                    global_dof = edge * base_edg_dof + _trace_local_dof(
                        is_positive, j, base_edg_dof, trace_orientation_mode,
                    )
                    sign = _trace_orientation_sign(is_positive, j, trace_orientation_mode)
                    trace_face += sign * trace[global_dof] * trace_base_to_post[j, trace_dof]
                q0_face = 0.0
                for j in range(post_el_dof):
                    moment = face_post_trace[face, j, trace_dof]
                    q0_face += nx * q0x[j] * moment + ny * q0y[j] * moment
                target = scale * (nx * qx_face + ny * qy_face + tau_face * (u_face - trace_face))
                gap[face * post_edg_dof + trace_dof, 0] = target - scale * q0_face

        _apply_min_distance_correction(
            element, bx, by, q0x, q0y, gap, face_rows, aff_jacs[element],
            interior_low_to_base, interior_low_to_post,
            ainv_constraint_t, schur_lu, schur_pivots, flux_coeffs,
        )


@njit(cache=True, parallel=True, fastmath=True)
def solve_rt_flux_postprocess_kernel(
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
        inverse_diffusion,
):
    r"""Solve the coupled degree-p+1 ADR Neumann postprocess on each element.

    ``inverse_diffusion`` contains validated K^-1 samples (K, nq, 4) in
    component order 00,01,10,11 at the recovery quadrature, including both
    off-diagonal blocks for nonsymmetric elliptic tensors.

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
                    mass = scale * volume_basis[q, i] * basis_j
                    matrix[i, post_el_dof + j] += mass * inverse_diffusion[element, q, 0]
                    matrix[i, 2 * post_el_dof + j] += mass * inverse_diffusion[element, q, 1]
                    matrix[post_el_dof + i, post_el_dof + j] += mass * inverse_diffusion[element, q, 2]
                    matrix[post_el_dof + i, 2 * post_el_dof + j] += mass * inverse_diffusion[element, q, 3]
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
