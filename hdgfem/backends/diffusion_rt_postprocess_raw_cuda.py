"""Raw-CUDA RT_p flux postprocessing for diffusion HDG."""

from __future__ import annotations

from hdgfem.precision import REAL_DTYPE, real_raw_kernel

import numpy as np

from hdgfem.backends.cupy import require_cupy

_KERNEL_CACHE = {}

_KERNEL_BODY = r'''
__device__ __forceinline__ double rt_volume_component(
        const int K, const int component, const int dof, const int q,
        const double *affine, const double *jacobian,
        const double *base_volume, const double *radial_volume) {
    const double inv_jac = 1.0 / jacobian[K];
    if (dof < BASE_DOF) {
        return affine[(K * 2 + component) * 2] * inv_jac
             * base_volume[q * BASE_DOF + dof];
    }
    if (dof < 2 * BASE_DOF) {
        const int local = dof - BASE_DOF;
        return affine[(K * 2 + component) * 2 + 1] * inv_jac
             * base_volume[q * BASE_DOF + local];
    }
    const int local = dof - 2 * BASE_DOF;
    double value = 0.0;
    for (int ref_component = 0; ref_component < 2; ++ref_component) {
        value += affine[(K * 2 + component) * 2 + ref_component] * inv_jac
            * radial_volume[(ref_component * ENRICH_DOF + local) * VOLUME_QUADS + q];
    }
    return value;
}
__device__ __forceinline__ double rt_face_component(
        const int K, const int component, const int dof, const int face,
        const int q, const double *affine, const double *jacobian,
        const double *base_face, const double *radial_face) {
    const double inv_jac = 1.0 / jacobian[K];
    if (dof < BASE_DOF) {
        return affine[(K * 2 + component) * 2] * inv_jac
            * base_face[(face * BASE_DOF + dof) * FACE_QUADS + q];
    }
    if (dof < 2 * BASE_DOF) {
        const int local = dof - BASE_DOF;
        return affine[(K * 2 + component) * 2 + 1] * inv_jac
            * base_face[(face * BASE_DOF + local) * FACE_QUADS + q];
    }
    const int local = dof - 2 * BASE_DOF;
    double value = 0.0;
    for (int ref_component = 0; ref_component < 2; ++ref_component) {
        value += affine[(K * 2 + component) * 2 + ref_component] * inv_jac
            * radial_face[((ref_component * 3 + face) * ENRICH_DOF + local)
                          * FACE_QUADS + q];
    }
    return value;
}
extern "C" __global__ void diffusion_rt_flux_postprocess_raw(
        const int num_elements, const double *total_values,
        const double *numerical, const double *affine, const double *jacobian,
        const double *face_jacobian, const double *normals,
        const double *volume_weights, const double *face_weights,
        const double *weighted_post, const double *post_mass_inverse,
        const double *base_volume, const double *base_face,
        const double *radial_volume, const double *radial_face,
        const double *low_volume, const double *face_test,
        double *output, int *status) {
    const int K = blockIdx.x;
    const int tid = threadIdx.x;
    if (K >= num_elements) return;
    extern __shared__ double shared[];
    double *matrix = shared;
    double *rhs = matrix + RT_DOF * RT_DOF;
    double *factors = rhs + RT_DOF;
    double *moments = factors + RT_DOF;
    for (int flat = tid; flat < RT_DOF * RT_DOF; flat += blockDim.x) {
        const int row = flat / RT_DOF;
        const int dof = flat - row * RT_DOF;
        double value = 0.0;
        if (row < 3 * FACE_TEST_DOF) {
            const int face = row / FACE_TEST_DOF;
            const int test = row - face * FACE_TEST_DOF;
            for (int q = 0; q < FACE_QUADS; ++q) {
                double normal_value = 0.0;
                for (int component = 0; component < 2; ++component) {
                    normal_value += rt_face_component(
                        K, component, dof, face, q, affine, jacobian,
                        base_face, radial_face)
                        * normals[(K * 3 + face) * 2 + component];
                }
                value += face_jacobian[K * 3 + face] * face_weights[q]
                    * face_test[test * FACE_QUADS + q] * normal_value;
            }
        } else {
            const int local_row = row - 3 * FACE_TEST_DOF;
            const int component = local_row / LOW_DOF;
            const int test = local_row - component * LOW_DOF;
            for (int q = 0; q < VOLUME_QUADS; ++q) {
                value += jacobian[K] * volume_weights[q]
                    * low_volume[q * LOW_DOF + test]
                    * rt_volume_component(K, component, dof, q, affine,
                        jacobian, base_volume, radial_volume);
            }
        }
        matrix[flat] = value;
    }
    for (int row = tid; row < RT_DOF; row += blockDim.x) {
        double value = 0.0;
        if (row < 3 * FACE_TEST_DOF) {
            const int face = row / FACE_TEST_DOF;
            const int test = row - face * FACE_TEST_DOF;
            for (int q = 0; q < FACE_QUADS; ++q) {
                value += face_jacobian[K * 3 + face] * face_weights[q]
                    * face_test[test * FACE_QUADS + q]
                    * numerical[(K * 3 + face) * FACE_QUADS + q];
            }
        } else {
            const int local_row = row - 3 * FACE_TEST_DOF;
            const int component = local_row / LOW_DOF;
            const int test = local_row - component * LOW_DOF;
            for (int q = 0; q < VOLUME_QUADS; ++q) {
                value += jacobian[K] * volume_weights[q]
                    * low_volume[q * LOW_DOF + test]
                    * total_values[(component * num_elements + K)
                                   * VOLUME_QUADS + q];
            }
        }
        rhs[row] = value;
    }
    __syncthreads();
    if (tid == 0) status[K] = 0;
    for (int pivot = 0; pivot < RT_DOF; ++pivot) {
        if (tid == 0) {
            int best = pivot;
            double best_value = fabs(matrix[pivot * RT_DOF + pivot]);
            for (int row = pivot + 1; row < RT_DOF; ++row) {
                const double candidate = fabs(matrix[row * RT_DOF + pivot]);
                if (candidate > best_value) {
                    best = row;
                    best_value = candidate;
                }
            }
            if (!(best_value > 1.0e-30) || !isfinite(best_value)) {
                status[K] = pivot + 1;
            } else if (best != pivot) {
                for (int col = pivot; col < RT_DOF; ++col) {
                    const double temporary = matrix[pivot * RT_DOF + col];
                    matrix[pivot * RT_DOF + col] = matrix[best * RT_DOF + col];
                    matrix[best * RT_DOF + col] = temporary;
                }
                const double temporary = rhs[pivot];
                rhs[pivot] = rhs[best];
                rhs[best] = temporary;
            }
        }
        __syncthreads();
        if (status[K] != 0) return;
        for (int row = pivot + 1 + tid; row < RT_DOF; row += blockDim.x) {
            factors[row] = matrix[row * RT_DOF + pivot]
                / matrix[pivot * RT_DOF + pivot];
        }
        __syncthreads();
        const int trailing = RT_DOF - pivot - 1;
        for (int flat = tid; flat < trailing * trailing; flat += blockDim.x) {
            const int row = pivot + 1 + flat / trailing;
            const int col = pivot + 1 + flat % trailing;
            matrix[row * RT_DOF + col] -= factors[row]
                * matrix[pivot * RT_DOF + col];
        }
        for (int row = pivot + 1 + tid; row < RT_DOF; row += blockDim.x) {
            rhs[row] -= factors[row] * rhs[pivot];
            matrix[row * RT_DOF + pivot] = 0.0;
        }
        __syncthreads();
    }
    if (tid == 0) {
        for (int row = RT_DOF - 1; row >= 0; --row) {
            double value = rhs[row];
            for (int col = row + 1; col < RT_DOF; ++col) {
                value -= matrix[row * RT_DOF + col] * factors[col];
            }
            factors[row] = value / matrix[row * RT_DOF + row];
        }
    }
    __syncthreads();
    for (int flat = tid; flat < 2 * POST_DOF; flat += blockDim.x) {
        const int component = flat / POST_DOF;
        const int test = flat - component * POST_DOF;
        double moment = 0.0;
        for (int q = 0; q < VOLUME_QUADS; ++q) {
            double value = 0.0;
            for (int dof = 0; dof < RT_DOF; ++dof) {
                value += factors[dof] * rt_volume_component(K, component,
                    dof, q, affine, jacobian, base_volume, radial_volume);
            }
            moment += value * weighted_post[q * POST_DOF + test];
        }
        moments[flat] = moment;
    }
    __syncthreads();
    for (int flat = tid; flat < 2 * POST_DOF; flat += blockDim.x) {
        const int component = flat / POST_DOF;
        const int coefficient = flat - component * POST_DOF;
        double value = 0.0;
        for (int test = 0; test < POST_DOF; ++test) {
            value += moments[component * POST_DOF + test]
                * post_mass_inverse[test * POST_DOF + coefficient];
        }
        output[(component * num_elements + K) * POST_DOF + coefficient] = value;
    }
}
'''


def _kernel_source(*, base_dof, enrich_dof, volume_quads, face_quads,
                   face_test_dof, low_dof, post_dof):
    """Specialize the RT_p CUDA source and return its local-system size."""
    rt_dof = 2 * int(base_dof) + int(enrich_dof)
    definitions = (
        f"#define BASE_DOF {int(base_dof)}\n"
        f"#define ENRICH_DOF {int(enrich_dof)}\n"
        f"#define RT_DOF {rt_dof}\n"
        f"#define VOLUME_QUADS {int(volume_quads)}\n"
        f"#define FACE_QUADS {int(face_quads)}\n"
        f"#define FACE_TEST_DOF {int(face_test_dof)}\n"
        f"#define LOW_DOF {int(low_dof)}\n"
        f"#define POST_DOF {int(post_dof)}\n"
    )
    return definitions + _KERNEL_BODY, rt_dof


def solve_diffusion_rt_flux_postprocess_raw_cuda(
        total_flux_values, numerical_normal_flux, aff_mats, aff_jacs,
        jacs_el_fc, normals, volume_weights, face_weights,
        post_weighted_basis, post_mass_inverse, base_volume_basis,
        base_face_basis, radial_volume, radial_face, low_volume_basis,
        face_test_basis):
    """Reconstruct RT_p flux coefficients with one CUDA block per element."""
    cp = require_cupy()
    arrays = [
        total_flux_values, numerical_normal_flux, aff_mats, aff_jacs,
        jacs_el_fc, normals, volume_weights, face_weights,
        post_weighted_basis, post_mass_inverse, base_volume_basis,
        base_face_basis, radial_volume, radial_face, low_volume_basis,
        face_test_basis,
    ]
    arrays = [
        cp.ascontiguousarray(cp.asarray(value), dtype=REAL_DTYPE)
        for value in arrays
    ]
    (total_values, numerical, affine, jacobian, face_jacobian, normal,
     volume_w, face_w, weighted_post, post_mass_inv, base_volume,
     base_face, radial_vol, radial_fc, low_volume, face_test) = arrays
    num_elements = int(affine.shape[0])
    base_dof = int(base_volume.shape[1])
    enrich_dof = int(radial_vol.shape[1])
    post_dof = int(weighted_post.shape[1])
    key = (
        base_dof, enrich_dof, int(volume_w.size), int(face_w.size),
        int(face_test.shape[0]), int(low_volume.shape[1]), post_dof,
    )
    cached = _KERNEL_CACHE.get(key)
    if cached is None:
        source, rt_dof = _kernel_source(
            base_dof=base_dof, enrich_dof=enrich_dof,
            volume_quads=volume_w.size, face_quads=face_w.size,
            face_test_dof=face_test.shape[0], low_dof=low_volume.shape[1],
            post_dof=post_dof,
        )
        kernel = real_raw_kernel(source, "diffusion_rt_flux_postprocess_raw")
        cached = kernel, rt_dof
        _KERNEL_CACHE[key] = cached
    kernel, rt_dof = cached
    output = cp.empty((2, num_elements, post_dof), dtype=REAL_DTYPE)
    status = cp.zeros(num_elements, dtype=cp.int32)
    shared_bytes = (rt_dof * rt_dof + 2 * rt_dof + 2 * post_dof) * 8
    kernel(
        (num_elements,), (64,),
        (np.int32(num_elements), total_values, numerical, affine, jacobian,
         face_jacobian, normal, volume_w, face_w, weighted_post,
         post_mass_inv, base_volume, base_face, radial_vol, radial_fc,
         low_volume, face_test, output, status),
        shared_mem=shared_bytes,
    )
    failed = cp.flatnonzero(status)
    if int(failed.size):
        first = int(failed[0].get())
        pivot = int(status[first].get()) - 1
        raise RuntimeError(
            "raw-CUDA RT_p postprocess singular moment matrix at element "
            f"{first}, pivot {pivot}"
        )
    return cp.asnumpy(output)


__all__ = ["solve_diffusion_rt_flux_postprocess_raw_cuda"]
