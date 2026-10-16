"""CuPy postprocessing kernels for stationary ADR and diffusion-reaction (Poisson) HDG."""

from __future__ import annotations

import numpy as np

from hybridge.runtime.optional import require_cupy


def solve_adr_rt_total_flux_postprocess_cupy(
        total_flux_values: np.ndarray,
        numerical_normal_flux: np.ndarray,
        aff_mats: np.ndarray,
        aff_jacs: np.ndarray,
        jacs_el_fc: np.ndarray,
        normals: np.ndarray,
        volume_weights: np.ndarray,
        face_weights: np.ndarray,
        post_weighted_basis: np.ndarray,
        post_mass_inverse: np.ndarray,
        base_volume_basis: np.ndarray,
        base_face_basis: np.ndarray,
        radial_volume: np.ndarray,
        radial_face: np.ndarray,
        low_volume_basis: np.ndarray,
        face_test_basis: np.ndarray,
        *, materialize_host: bool = True,
) -> np.ndarray:
    r"""Reconstruct ``RT_p`` total fluxes with batched device dense solves.

    This is the CuPy mirror of
    ``solve_rt_flux_postprocess_kernel``. Static reference and mesh
    tables are uploaded once per call, all element moment matrices are formed
    and solved in batches on the active CUDA device. Final degree-``p+1``
    coefficients stay on device when ``materialize_host=False``; the default
    preserves the host-returning helper contract.
    """
    cp = require_cupy()
    total_values = cp.asarray(total_flux_values)
    numerical = cp.asarray(numerical_normal_flux)
    affine = cp.asarray(aff_mats)
    jacobian = cp.asarray(aff_jacs)
    face_jacobian = cp.asarray(jacs_el_fc)
    normal = cp.asarray(normals)
    volume_w = cp.asarray(volume_weights)
    face_w = cp.asarray(face_weights)
    weighted_post = cp.asarray(post_weighted_basis)
    post_mass_inv = cp.asarray(post_mass_inverse)
    base_volume = cp.asarray(base_volume_basis)
    base_face = cp.asarray(base_face_basis)
    radial_vol = cp.asarray(radial_volume)
    radial_fc = cp.asarray(radial_face)
    low_volume = cp.asarray(low_volume_basis)
    face_test = cp.asarray(face_test_basis)

    num_elements = affine.shape[0]
    base_dof = base_volume.shape[1]
    enrichment_dof = radial_vol.shape[1]
    rt_dof = 2 * base_dof + enrichment_dof
    volume_quads = volume_w.size
    face_quads = face_w.size
    scaled_affine = affine / jacobian[:, None, None]

    rt_volume = cp.empty(
        (num_elements, 2, rt_dof, volume_quads), dtype=cp.float64
    )
    base_volume_t = base_volume.T
    rt_volume[:, 0, :base_dof] = (
        scaled_affine[:, 0, 0, None, None] * base_volume_t[None]
    )
    rt_volume[:, 1, :base_dof] = (
        scaled_affine[:, 1, 0, None, None] * base_volume_t[None]
    )
    rt_volume[:, 0, base_dof : 2 * base_dof] = (
        scaled_affine[:, 0, 1, None, None] * base_volume_t[None]
    )
    rt_volume[:, 1, base_dof : 2 * base_dof] = (
        scaled_affine[:, 1, 1, None, None] * base_volume_t[None]
    )
    rt_volume[:, :, 2 * base_dof :] = cp.einsum(
        "Kac,cjq->Kajq", scaled_affine, radial_vol, optimize=True
    )

    rt_face = cp.empty(
        (num_elements, 2, rt_dof, 3, face_quads), dtype=cp.float64
    )
    base_face_dof_first = base_face.transpose(1, 0, 2)
    rt_face[:, 0, :base_dof] = (
        scaled_affine[:, 0, 0, None, None, None]
        * base_face_dof_first[None]
    )
    rt_face[:, 1, :base_dof] = (
        scaled_affine[:, 1, 0, None, None, None]
        * base_face_dof_first[None]
    )
    rt_face[:, 0, base_dof : 2 * base_dof] = (
        scaled_affine[:, 0, 1, None, None, None]
        * base_face_dof_first[None]
    )
    rt_face[:, 1, base_dof : 2 * base_dof] = (
        scaled_affine[:, 1, 1, None, None, None]
        * base_face_dof_first[None]
    )
    rt_face[:, :, 2 * base_dof :] = cp.einsum(
        "Kac,cfjq->Kajfq", scaled_affine, radial_fc, optimize=True
    )

    rt_normal = cp.einsum(
        "Kadfq,Kfa->Kfdq", rt_face, normal, optimize=True
    )
    face_matrix = cp.einsum(
        "Kf,q,iq,Kfdq->Kfid",
        face_jacobian,
        face_w,
        face_test,
        rt_normal,
        optimize=True,
    ).reshape(num_elements, -1, rt_dof)
    face_rhs = cp.einsum(
        "Kf,q,iq,Kfq->Kfi",
        face_jacobian,
        face_w,
        face_test,
        numerical,
        optimize=True,
    ).reshape(num_elements, -1)

    interior_matrix = cp.einsum(
        "K,q,qi,Kadq->Kiad",
        jacobian,
        volume_w,
        low_volume,
        rt_volume,
        optimize=True,
    )
    interior_rhs = cp.einsum(
        "K,q,qi,Kaq->Kia",
        jacobian,
        volume_w,
        low_volume,
        total_values.transpose(1, 0, 2),
        optimize=True,
    )

    moment_matrix = cp.concatenate(
        (
            face_matrix,
            interior_matrix[:, :, 0],
            interior_matrix[:, :, 1],
        ),
        axis=1,
    )
    moment_rhs = cp.concatenate(
        (face_rhs, interior_rhs[:, :, 0], interior_rhs[:, :, 1]), axis=1
    )
    rt_coeffs = cp.linalg.solve(moment_matrix, moment_rhs[..., None])[..., 0]

    rt_values = cp.einsum(
        "Kd,Kadq->Kaq", rt_coeffs, rt_volume, optimize=True
    )
    post_moments = cp.einsum(
        "Kaq,qi->Kai", rt_values, weighted_post, optimize=True
    )
    post_coeffs = cp.einsum(
        "Kai,ij->Kaj", post_moments, post_mass_inv, optimize=True
    )
    result = cp.ascontiguousarray(post_coeffs.transpose(1, 0, 2))
    return cp.asnumpy(result) if materialize_host else result


__all__ = ["postprocess_diffusion_primal_cupy", "solve_adr_rt_total_flux_postprocess_cupy"]


def postprocess_total_flux_l2_cupy(total_values, numerical, space, cache):
    """Apply the host-equivalent constrained minimum-distance flux recovery."""
    from hybridge.core.device import field_from_cupy_coefficients
    from hybridge.core.space import VectorDGField

    cp = require_cupy()
    post = cache.post_space
    q = post.quad_data
    n, d = space.mesh.num_tri, post.el_dof
    jac = cp.asarray(space.mesh.aff_jacs)
    normals = cp.asarray(space.mesh.normals)
    face_jac = cp.asarray(space.mesh.jacs_el_fc)
    mass_inv = cp.asarray(q.MKrf_inv)
    base_basis = cp.asarray(space.basis_at(q.Krf_quads))
    base = total_values @ (cp.asarray(q.Krf_w)[:, None] * base_basis) @ cp.asarray(space.quad_data.MKrf_inv)
    initial = base @ cp.asarray(cache.base_to_post_mass).T @ mass_inv
    face_moments = cp.asarray(q.face_element_test_trace_trial)
    face_constraints = cp.einsum('Kf,Kfc,fia->Kfaci', face_jac, normals, face_moments).reshape(n, -1, 2*d)
    low = cp.asarray(cache.interior_low_to_post)
    low_base = cp.asarray(cache.interior_low_to_base)
    low_dof = low.shape[0]
    interior = cp.zeros((n, 2*low_dof, 2*d))
    interior[:, :low_dof, :d] = jac[:, None, None] * low
    interior[:, low_dof:, d:] = jac[:, None, None] * low
    constraints = cp.concatenate((face_constraints, interior), axis=1)
    face_target = cp.einsum('Kf,Kfq,aq,q->Kfa', face_jac, numerical,
                            cp.asarray(q.bas1d_of_ref_edg_qds), cp.asarray(q.weights_JGL)).reshape(n, -1)
    interior_target = (base @ low_base.T).transpose(1, 0, 2).reshape(n, -1) * jac[:, None]
    target = cp.concatenate((face_target, interior_target), axis=1)
    initial = initial.transpose(1, 0, 2).reshape(n, 2*d)
    lift = cp.concatenate((mass_inv @ constraints[:, :, :d].transpose(0, 2, 1),
                           mass_inv @ constraints[:, :, d:].transpose(0, 2, 1)), axis=1) / jac[:, None, None]
    gap = target - (constraints @ initial[..., None])[..., 0]
    correction = cp.linalg.solve(constraints @ lift, gap[..., None])
    result = (initial + (lift @ correction)[..., 0]).reshape(n, 2, d)
    return VectorDGField(tuple(field_from_cupy_coefficients(post, result[:, c], name='total_flux_h_star')
                               for c in range(2)), name='total_flux_h_star')


def _primal_system_cupy(local_unknowns, total_flux, space, cache, samples, diffusion):
    """Independent contraction reference for the coupled ADR Neumann system.

    Volume, face and mean equations match the Numba reference, including the
    total numerical flux and the scalar Neumann multiplier.
    """
    from hybridge.core.device import as_cupy_coefficients, as_cupy_space

    cp = require_cupy()
    post = cache.post_space
    q = post.quad_data
    n, d, e = space.mesh.num_tri, post.el_dof, q.edg_dof
    offset, rows = 3*d, 3*d + 3*e + 1
    matrix = cp.zeros((n, rows, rows))
    rhs = cp.zeros((n, rows))
    jac = cp.asarray(space.mesh.aff_jacs)
    normal = cp.asarray(space.mesh.normals)
    face_jac = cp.asarray(space.mesh.jacs_el_fc)
    phi, weights = cp.asarray(q.phi), cp.asarray(q.Krf_w)
    gradient = cp.einsum('Kab,qib->Kqia', cp.asarray(space.mesh.inv_aff_mats_t), cp.asarray(q.gphi))
    beta_volume, beta_face, tau = samples
    flux = cp.stack([as_cupy_coefficients(f, as_cupy_space(f.space)) for f in total_flux.components], axis=1)
    values = flux @ phi.T
    from hybridge.mixed.coefficients import (
            sample_diffusion_tensor,
            inverse_diffusion_values,
        )
    inverse = inverse_diffusion_values(sample_diffusion_tensor(diffusion, post, device=True))
    for c in range(2):
        constitutive = slice(c*d, (c+1)*d)
        flux_columns = slice((c+1)*d, (c+2)*d)
        grad_mass = cp.einsum('K,q,Kqi,qj->Kij', jac, weights, gradient[..., c], phi)
        matrix[:, constitutive, :d] = -grad_mass
        for component in range(2):
            matrix[:, constitutive, (component+1)*d:(component+2)*d] = cp.einsum(
                'K,q,Kq,qi,qj->Kij', jac, weights, inverse[..., 2*c+component], phi, phi)
        matrix[:, 2*d:3*d, flux_columns] = -grad_mass
    matrix[:, 2*d:3*d, :d] = -cp.einsum('K,q,Kqia,Kqa,qj->Kij', jac, weights, gradient, beta_volume, phi)
    rhs[:, 2*d:3*d] = -cp.einsum('K,q,Kqia,Kaq->Ki', jac, weights, gradient, values)
    mean = jac[:, None] * (weights @ phi)
    matrix[:, 2*d:3*d, -1] = mean
    matrix[:, -1, :d] = mean
    base = cp.asarray(local_unknowns).reshape(n, 3, space.el_dof)[:, 0]
    rhs[:, -1] = jac * (base @ cp.asarray(cache.mean_base))
    face_phi = cp.asarray(q.bas_of_bd_quads)
    trace_phi = cp.asarray(q.bas1d_of_ref_edg_qds)
    face_weights = cp.asarray(q.weights_JGL)
    for face in range(3):
        p = face_phi[face]
        scale = face_jac[:, face, None] * face_weights
        bn = cp.einsum('Kqa,Ka->Kq', beta_face[:, face], normal[:, face])
        normal_flux = cp.einsum('Kai,Ka,iq->Kq', flux, normal[:, face], p)
        trace_slice = slice(offset + face*e, offset + (face+1)*e)
        mass_face = cp.einsum('Kq,iq,jq->Kij', scale, p, p)
        cross = cp.einsum('Kq,iq,aq->Kia', scale, p, trace_phi)
        matrix[:, 2*d:3*d, :d] += cp.einsum('Kq,Kq,iq,jq->Kij', scale, tau[:, face], p, p)
        matrix[:, 2*d:3*d, trace_slice] += cp.einsum('Kq,Kq,iq,aq->Kia', scale, bn-tau[:, face], p, trace_phi)
        matrix[:, trace_slice, :d] += cp.einsum('Kq,Kq,aq,iq->Kai', scale, tau[:, face], trace_phi, p)
        matrix[:, trace_slice, trace_slice] += cp.einsum('Kq,Kq,aq,bq->Kab', scale, bn-tau[:, face], trace_phi, trace_phi)
        for c in range(2):
            nc = normal[:, face, c, None, None]
            matrix[:, c*d:(c+1)*d, trace_slice] += nc * cross
            matrix[:, 2*d:3*d, (c+1)*d:(c+2)*d] += nc * mass_face
            matrix[:, trace_slice, (c+1)*d:(c+2)*d] += nc * cross.transpose(0, 2, 1)
        rhs[:, 2*d:3*d] += cp.einsum('Kq,Kq,iq->Ki', scale, normal_flux, p)
        rhs[:, trace_slice] += cp.einsum('Kq,Kq,aq->Ka', scale, normal_flux, trace_phi)
    return matrix, rhs


def postprocess_primal_cupy(local_unknowns, total_flux, space, cache, samples, diffusion):
    """Recover tensor ADR primal coefficients using fused assembly and batched LU."""
    from hybridge.mixed.coefficients import (
            sample_diffusion_tensor,
            inverse_diffusion_values,
        )
    from hybridge.mixed.postprocess.primal_raw_cuda import primal_system_raw_cuda
    from hybridge.core.device import field_from_cupy_coefficients

    cp = require_cupy()
    post = cache.post_space
    inverse = inverse_diffusion_values(sample_diffusion_tensor(diffusion, post, device=True))
    matrix, rhs = primal_system_raw_cuda(local_unknowns, total_flux, space, cache, samples, inverse)
    result = cp.linalg.solve(matrix, rhs[..., None])[:, :post.el_dof, 0]
    return field_from_cupy_coefficients(post, result, name='u_h_star')


# Device memory for one batch of local primal systems (elements x rows x rows doubles) or of per-call temporaries.
_PRIMAL_BATCH_BYTES = 256 * 1024 * 1024
# Environment override (bytes) of the device memory the cached primal operators may take; default 25 % of the device.
_PRIMAL_CACHE_ENV = "HYBRIDGE_PRIMAL_CACHE_MAX_BYTES"
# One thread per (element k, coefficient i). "pq": u*_ki = s sum_j (P^T_kji qx_kj + Q^T_kji qy_kj) + a_ki mean_k, with
# P = A R_x, Q = A R_y (A the inverse of the local system, R the reference right-hand-side maps); "inverse":
# u*_ki = sum_j A^T_kji rhs_kj + a_ki mean_k. mean_k = sum_j u_kj mean_base_j. Transposed per-element storage
# (index i fastest) keeps the reads coalesced.
_PRIMAL_APPLY_SOURCE = r"""
extern "C" __global__ void primal_apply_pq(
        const double* __restrict__ p_t, const double* __restrict__ q_t, const double* __restrict__ mean_column,
        const double* __restrict__ unknowns, const double* __restrict__ mean_base, const double scale,
        const long long num_elements, const int d, const int e, double* __restrict__ out) {
    const long long tid = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_elements * d) return;
    const long long k = tid / d;
    const int i = (int)(tid - k * d);
    const double* u = unknowns + 3LL * e * k;
    const double* p = p_t + (long long)e * d * k + i;
    const double* q = q_t + (long long)e * d * k + i;
    double mean = 0.0, sum = 0.0;
    for (int j = 0; j < e; ++j) {
        mean += u[j] * mean_base[j];
        sum += p[(long long)d * j] * u[e + j] + q[(long long)d * j] * u[2 * e + j];
    }
    out[(long long)d * k + i] = scale * sum + mean_column[(long long)d * k + i] * mean;
}

extern "C" __global__ void primal_apply_inverse(
        const double* __restrict__ inverse_t, const double* __restrict__ mean_column,
        const double* __restrict__ rhs, const double* __restrict__ unknowns, const double* __restrict__ mean_base,
        const long long num_elements, const int d, const int e, double* __restrict__ out) {
    const long long tid = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_elements * d) return;
    const long long k = tid / d;
    const int i = (int)(tid - k * d);
    const double* u = unknowns + 3LL * e * k;
    const double* r = rhs + (long long)d * k;
    const double* a = inverse_t + (long long)d * d * k + i;
    double mean = 0.0, sum = 0.0;
    for (int j = 0; j < e; ++j) mean += u[j] * mean_base[j];
    for (int j = 0; j < d; ++j) sum += a[(long long)d * j] * r[j];
    out[(long long)d * k + i] = sum + mean_column[(long long)d * k + i] * mean;
}
"""


_PRIMAL_KERNELS: dict[int, tuple] = {}


def _primal_kernels(cp):
    """Compiled ``primal_apply_pq`` / ``primal_apply_inverse`` kernels of the current device (NVRTC, compiled once)."""
    device = cp.cuda.runtime.getDevice()
    kernels = _PRIMAL_KERNELS.get(device)
    if kernels is None:
        module = cp.RawModule(code=_PRIMAL_APPLY_SOURCE)
        kernels = _PRIMAL_KERNELS[device] = (module.get_function("primal_apply_pq"),
                                             module.get_function("primal_apply_inverse"))
    return kernels


def _primal_cache_limit(cp) -> int:
    """Bytes the cached primal operators may take: ``$HYBRIDGE_PRIMAL_CACHE_MAX_BYTES``, else 25 % of the device."""
    import os

    value = os.environ.get(_PRIMAL_CACHE_ENV)
    if value:
        return int(float(value))
    return int(cp.cuda.Device().mem_info[1] // 4)


def _primal_device_state(cp, space, cache):
    """Device tables of the primal postprocess, uploaded once per device into ``cache``.

    ``c_r``/``c_s`` (and ``c_rs_t = [c_r; c_s]^T``) map one flux component's coefficients to the
    right-hand side: ``c_r[i, j] = sum_q w_q d_r w_i(x_q) phi_j(x_q)`` (reference derivatives of the
    degree-``p+1`` test functions against the degree-``p`` basis), so a scalar ``kappa`` needs no
    quadrature-point values.
    """
    device = cp.cuda.runtime.getDevice()
    states = getattr(cache, "_diffusion_primal_device_states", None)
    if states is None:
        states = cache._diffusion_primal_device_states = {}
    state = states.get(device)
    if state is None:
        q_post = cache.post_space.quad_data
        weights = np.asarray(q_post.Krf_w, dtype=np.float64)
        basis = np.asarray(cache.base_basis_on_post_quads, dtype=np.float64)
        grad_r = np.asarray(q_post.gphi[:, :, 0], dtype=np.float64)
        grad_s = np.asarray(q_post.gphi[:, :, 1], dtype=np.float64)
        c_r, c_s = grad_r.T @ (weights[:, None] * basis), grad_s.T @ (weights[:, None] * basis)
        state = states[device] = {name: cp.asarray(value, dtype=cp.float64, order="C") for name, value in (
            ("stiffness_rr", cache.primal_stiffness_rr), ("stiffness_rs", cache.primal_stiffness_rs),
            ("stiffness_ss", cache.primal_stiffness_ss), ("mean_post", cache.mean_post),
            ("mean_base", cache.mean_base), ("basis_t", basis.T), ("weights", weights), ("grad_r", grad_r),
            ("grad_s", grad_s), ("c_r", c_r), ("c_s", c_s), ("c_rs_t", np.concatenate((c_r, c_s)).T),
            ("jacobians", space.mesh.aff_jacs), ("inverse_t", space.mesh.inv_aff_mats_t))}
        state["operator"] = None
        state["operator_unavailable"] = False
    return state


def _primal_matrices(cp, state, start, stop):
    """Local systems ``[[K, m], [m^T, 0]]`` of elements ``start:stop``; the mean row is not scaled by
    the element area, which keeps the matrix balanced (the multiplier is not used)."""
    post_el_dof = state["mean_post"].shape[0]
    rows = post_el_dof + 1
    jac, inv = state["jacobians"][start:stop], state["inverse_t"][start:stop]
    inv00, inv01, inv10, inv11 = inv[:, 0, 0], inv[:, 0, 1], inv[:, 1, 0], inv[:, 1, 1]
    matrix = cp.zeros((stop - start, rows, rows), dtype=cp.float64)
    matrix[:, :post_el_dof, :post_el_dof] = jac[:, None, None] * (
        (inv00 * inv00 + inv10 * inv10)[:, None, None] * state["stiffness_rr"]
        + (inv00 * inv01 + inv10 * inv11)[:, None, None] * state["stiffness_rs"]
        + (inv01 * inv01 + inv11 * inv11)[:, None, None] * state["stiffness_ss"])
    matrix[:, :post_el_dof, post_el_dof] = state["mean_post"]
    matrix[:, post_el_dof, :post_el_dof] = state["mean_post"]
    return matrix


def _primal_operator(cp, state, num_elements, base_el_dof):
    """Cached per-element primal operator (geometry only, so built once), or ``None``.

    ``("pq", P^T, Q^T, a)`` when ``2 e + 1`` columns per coefficient fit the cache limit, else
    ``("inverse", A^T, a)`` (``A`` the inverse block, ``a`` the mean-multiplier column), else ``None``:
    the limit is exceeded, the build runs out of device memory, or the explicit inverses of the first
    batch miss the 1e-8 accuracy check. Then every call solves the batched systems.
    """
    if state["operator"] is not None or state["operator_unavailable"]:
        return state["operator"]
    import time

    started = time.perf_counter()
    d, e = state["mean_post"].shape[0], int(base_el_dof)
    rows, limit = d + 1, _primal_cache_limit(cp)
    kind = ("pq" if num_elements * d * (2 * e + 1) * 8 <= limit else
            "inverse" if num_elements * d * (d + 1) * 8 <= limit else None)
    if kind is None:
        state["operator_unavailable"] = True
        return None
    try:
        mean_column = cp.empty((num_elements, d), dtype=cp.float64)
        if kind == "pq":
            p_t = cp.empty((num_elements, e, d), dtype=cp.float64)
            q_t = cp.empty((num_elements, e, d), dtype=cp.float64)
        else:
            inverse_t = cp.empty((num_elements, d, d), dtype=cp.float64)
        batch = max(1, _PRIMAL_BATCH_BYTES // (8 * (rows * rows + 4 * d * e)))
        for start in range(0, num_elements, batch):
            stop = min(num_elements, start + batch)
            matrix = _primal_matrices(cp, state, start, stop)
            inverse = cp.linalg.inv(matrix)
            if start == 0 and float(cp.abs(matrix @ inverse - cp.eye(rows)).max()) > 1.0e-8:
                state["operator_unavailable"] = True
                return None
            block = inverse[:, :d, :d]
            mean_column[start:stop] = inverse[:, :d, d]
            if kind == "pq":
                # R_x = -J (G00 c_r + G01 c_s), R_y = -J (G10 c_r + G11 c_s) for kappa = 1.
                jac, geometry = state["jacobians"][start:stop], state["inverse_t"][start:stop]
                g = [(-jac * geometry[:, i, j])[:, None, None] for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
                p_t[start:stop] = (block @ (g[0] * state["c_r"] + g[1] * state["c_s"])).transpose(0, 2, 1)
                q_t[start:stop] = (block @ (g[2] * state["c_r"] + g[3] * state["c_s"])).transpose(0, 2, 1)
            else:
                inverse_t[start:stop] = block.transpose(0, 2, 1)
    except cp.cuda.memory.OutOfMemoryError:
        state["operator_unavailable"] = True
        cp.get_default_memory_pool().free_all_blocks()
        return None
    state["operator"] = ("pq", p_t, q_t, mean_column) if kind == "pq" else ("inverse", inverse_t, mean_column)
    cp.cuda.get_current_stream().synchronize()
    state["build_seconds"] = time.perf_counter() - started
    return state["operator"]


def postprocess_diffusion_primal_cupy(local_unknowns, space, cache, diffusion=1.0, *, name="u_h_star"):
    """Degree-``p+1`` primal postprocessing of a diffusion-reaction HDG solution on the CUDA device.

    Device port of ``solve_primal_postprocess_kernel``: on each element, ``u*`` in ``P_{p+1}``
    satisfies ``(grad u*, grad w) = -(kappa^{-1} q_h, grad w)`` for every ``w`` and has the mean
    of ``u_h``. ``local_unknowns`` holds ``[u_h | q_x | q_y]`` per element (shape
    ``(K, 3 el_dof)``, host or device); ``cache`` is the host postprocessing cache
    (``mixed.postprocess.flux._new_hdg_postprocess_cache``), which keeps the device state.

    The local systems depend on the mesh only, so the first call builds a per-element operator
    (``_primal_operator``) and later calls only apply it. For a scalar ``kappa``: the ``pq`` operator
    maps the flux coefficients straight to ``u*`` in one kernel (no GEMM, no temporaries); the
    ``inverse`` operator (when ``pq`` does not fit) needs one GEMM for the right-hand side first. A
    non-scalar ``kappa``, or no cached operator, solves the batched local systems on every call.
    Returns a device-resident ``DGField`` in ``cache.post_space``.
    """
    from hybridge.core.device import field_from_cupy_coefficients
    from hybridge.mixed.coefficients import _diffusion_is_identity, _inverse_diffusion_values

    cp = require_cupy()
    post = cache.post_space
    base_el_dof, post_el_dof = int(space.el_dof), int(post.el_dof)
    num_elements = int(space.mesh.num_tri)
    unknowns = cp.ascontiguousarray(cp.asarray(local_unknowns, dtype=cp.float64))
    if tuple(unknowns.shape) != (num_elements, 3 * base_el_dof):
        raise ValueError(f"local_unknowns must have shape {(num_elements, 3 * base_el_dof)}; got {unknowns.shape}")
    state = _primal_device_state(cp, space, cache)
    if _diffusion_is_identity(diffusion) or np.isscalar(diffusion):
        scale = 1.0 if _diffusion_is_identity(diffusion) else 1.0 / float(diffusion)
        sampled = None
    else:  # sampled kappa^{-1} at the postprocess quadrature points, shape (K, n_q) per entry
        scale = 1.0
        sampled = tuple(cp.asarray(value, dtype=cp.float64) for value in _inverse_diffusion_values(diffusion, post))
    coefficients = cp.empty((num_elements, post_el_dof), dtype=cp.float64)
    operator = None if sampled is not None else _primal_operator(cp, state, num_elements, base_el_dof)
    # Path of this call for the solver's timing details: 2 pq, 1 inverse, 0 per-call solve.
    cache._diffusion_primal_info = {
        "postprocess.primal.operator": 0.0 if operator is None else (2.0 if operator[0] == "pq" else 1.0),
        "postprocess.primal.cache_bytes": 0.0 if operator is None else float(sum(a.nbytes for a in operator[1:])),
        "postprocess.primal.build_seconds": float(state.pop("build_seconds", 0.0)),
    }
    if operator is not None:
        apply_pq, apply_inverse = _primal_kernels(cp)
        threads = num_elements * post_el_dof
        grid = ((threads + 255) // 256,)
        if operator[0] == "pq":
            apply_pq(grid, (256,), (operator[1], operator[2], operator[3], unknowns, state["mean_base"],
                                    np.float64(scale), np.int64(num_elements), np.int32(post_el_dof),
                                    np.int32(base_el_dof), coefficients))
            return field_from_cupy_coefficients(post, coefficients, name=name)
        fluxes = unknowns[:, base_el_dof:].reshape(2 * num_elements, base_el_dof) @ state["c_rs_t"]
        along = fluxes.reshape(num_elements, 2, 2 * post_el_dof)
        geometry = state["inverse_t"]
        factor = (-scale * state["jacobians"])[:, None]
        rhs = cp.ascontiguousarray(factor * (
            geometry[:, 0, 0][:, None] * along[:, 0, :post_el_dof]
            + geometry[:, 0, 1][:, None] * along[:, 0, post_el_dof:]
            + geometry[:, 1, 0][:, None] * along[:, 1, :post_el_dof]
            + geometry[:, 1, 1][:, None] * along[:, 1, post_el_dof:]))
        apply_inverse(grid, (256,), (operator[1], operator[2], rhs, unknowns, state["mean_base"],
                                     np.int64(num_elements), np.int32(post_el_dof), np.int32(base_el_dof),
                                     coefficients))
        return field_from_cupy_coefficients(post, coefficients, name=name)
    batch = max(1, _PRIMAL_BATCH_BYTES // (8 * (post_el_dof + 1) ** 2))
    for start in range(0, num_elements, batch):
        stop = min(num_elements, start + batch)
        jac, inv = state["jacobians"][start:stop], state["inverse_t"][start:stop]
        inv00, inv01, inv10, inv11 = (inv[:, i, j][:, None] for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)))
        block = unknowns[start:stop]
        if sampled is None:
            # rhs_i = -J/kappa (inv00 C_r q_x + inv01 C_s q_x + inv10 C_r q_y + inv11 C_s q_y)_i
            along_x = block[:, base_el_dof:2 * base_el_dof] @ state["c_rs_t"]
            along_y = block[:, 2 * base_el_dof:] @ state["c_rs_t"]
            rhs = -(scale * jac)[:, None] * (
                inv00 * along_x[:, :post_el_dof] + inv01 * along_x[:, post_el_dof:]
                + inv10 * along_y[:, :post_el_dof] + inv11 * along_y[:, post_el_dof:])
        else:
            qx = block[:, base_el_dof:2 * base_el_dof] @ state["basis_t"]
            qy = block[:, 2 * base_el_dof:] @ state["basis_t"]
            i00, i01, i10, i11 = (value[start:stop] for value in sampled)
            cqx, cqy = i00 * qx + i01 * qy, i10 * qx + i11 * qy
            # cq . grad w = (inv00 cqx + inv10 cqy) d_r w + (inv01 cqx + inv11 cqy) d_s w
            along_r = (inv00 * cqx + inv10 * cqy) * state["weights"]
            along_s = (inv01 * cqx + inv11 * cqy) * state["weights"]
            rhs = -jac[:, None] * (along_r @ state["grad_r"] + along_s @ state["grad_s"])
        full = cp.concatenate((rhs, (block[:, :base_el_dof] @ state["mean_base"])[:, None]), axis=1)
        coefficients[start:stop] = cp.linalg.solve(_primal_matrices(cp, state, start, stop),
                                                   full[:, :, None])[:, :post_el_dof, 0]
    return field_from_cupy_coefficients(post, coefficients, name=name)
