
"""Experimental Raw CUDA kernels for diffusion-reaction HDG assembly.

The first implementation targets the standalone CUDA diffusion runner
benchmark path: identity diffusion, scalar zero reaction, legacy-lagrange or legendre-modal
trace coordinates, and p <= 6.  It keeps the existing CuPy source and boundary
trace evaluation, then fuses the expensive element-local HDG condensation,
boundary-column elimination, and COO/RHS emission into Raw CUDA kernels.

The serial Raw CUDA kernel is intentionally written as a readable baseline: one
CUDA thread owns one element and performs the same small dense algebra as the
fused Numba kernel.  The cooperative path keeps the same element ownership, but
uses all threads in the block to assemble the local Schur matrix, factor its
trailing updates, solve all trace/source columns, recover flux columns, and emit
the reduced matrix directly as COO or CSR.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .cupy import as_cupy_space, require_cupy
from .raw_cuda import RawCudaBlockSize, resolve_raw_cuda_block_size


@dataclass(frozen=True)
class RawDiffusionAssemblyResult:
    """Device-side reduced trace system assembled by the Raw CUDA path."""

    rows: Any | None
    cols: Any | None
    data: Any
    rhs: Any
    source_rhs: Any
    boundary_trace: Any
    d0_reference: Any
    d1_reference: Any
    face_element_mass: Any
    timings: dict[str, float]
    indptr: Any | None = None
    indices: Any | None = None
    matrix_format: str = 'coo'
    csr_pattern: Any | None = None


_RAW_TRACE_ORIENTATION_HELPERS = r"""
__device__ __forceinline__ int raw_trace_local_dof(
        const bool positive,
        const int dof,
        const int edge_dof)
{
#if TRACE_ORIENTATION_MODE == 1
    return dof;
#else
    return positive ? dof : (edge_dof - 1 - dof);
#endif
}

__device__ __forceinline__ double raw_trace_orientation_sign(
        const bool positive,
        const int dof)
{
#if TRACE_ORIENTATION_MODE == 1
    return ((!positive) && ((dof & 1) == 1)) ? -1.0 : 1.0;
#else
    return 1.0;
#endif
}
"""

_RAW_ASSEMBLY_TEMPLATE = r"""
// Solve one condensed element column against an already-factorized scalar Schur
// matrix.  All arrays are owned by one element block.  The routine deliberately
// keeps the algebra explicit instead of calling cuSOLVER/cuBLAS batched helpers:
// the goal of this RawKernel path is to fuse local construction, solve, boundary
// elimination, and COO emission without materializing large global temporaries.
__device__ __forceinline__ void solve_condensed_column_raw(
        double* __restrict__ schur_lu,
        const int* __restrict__ pivots,
        const double* __restrict__ d0,
        const double* __restrict__ d1,
        const double* __restrict__ mn0,
        const double* __restrict__ mn1,
        double* __restrict__ rhs0,
        double* __restrict__ qx,
        double* __restrict__ qy,
        double* __restrict__ u,
        double* __restrict__ tmp1,
        double* __restrict__ tmp2,
        const double* __restrict__ mass_inverse,
        const double* __restrict__ face_element_trace,
        const double* __restrict__ source_rhs,
        const double* __restrict__ jacs_el_fc,
        const double* __restrict__ normals,
        const double tau,
        const double jac_inv,
        const long long element,
        const int is_source_column,
        const int col_face,
        const int local_trace_dof)
{
    // Build the mixed RHS column.  Trace columns use local face coordinates;
    // global edge orientation is handled by the caller through local_trace_dof.
    for (int i = 0; i < NEL; ++i) {
        rhs0[i] = 0.0;
        qx[i] = 0.0;
        qy[i] = 0.0;
    }
    if (is_source_column) {
        for (int i = 0; i < NEL; ++i) {
            rhs0[i] = source_rhs[element * 3 * NEL + i];
        }
    } else {
        const double face_scale = jacs_el_fc[element * 3 + col_face];
        const double normal_x = normals[(element * 3 + col_face) * 2 + 0];
        const double normal_y = normals[(element * 3 + col_face) * 2 + 1];
        for (int i = 0; i < NEL; ++i) {
            const double coupling = face_scale * face_element_trace[(col_face * NEL + i) * NTR + local_trace_dof];
            rhs0[i] = tau * coupling;
            qx[i] = normal_x * coupling;
            qy[i] = normal_y * coupling;
        }
    }

    // Condense flux equations into the scalar Schur RHS:
    // rhs_u + J^{-1} * ((N_x-D_x) M^{-1} rhs_qx + (N_y-D_y) M^{-1} rhs_qy).
    // tmp1/tmp2 hold M^{-1} rhs_qx/y, then are reused below for q recovery.
    for (int i = 0; i < NEL; ++i) {
        double value1 = 0.0;
        double value2 = 0.0;
        for (int k = 0; k < NEL; ++k) {
            value1 += mass_inverse[i * NEL + k] * qx[k];
            value2 += mass_inverse[i * NEL + k] * qy[k];
        }
        tmp1[i] = value1;
        tmp2[i] = value2;
    }
    for (int i = 0; i < NEL; ++i) {
        double acc0 = 0.0;
        double acc1 = 0.0;
        for (int k = 0; k < NEL; ++k) {
            acc0 += mn0[i * NEL + k] * tmp1[k];
            acc1 += mn1[i * NEL + k] * tmp2[k];
        }
        u[i] = rhs0[i] + jac_inv * (acc0 + acc1);
    }

    // Apply the same row permutations used during in-place LU factorization,
    // then perform forward/back substitution.  No explicit inverse is formed.
    for (int k = 0; k < NEL; ++k) {
        const int pivot = pivots[k];
        if (pivot != k) {
            const double swap_value = u[k];
            u[k] = u[pivot];
            u[pivot] = swap_value;
        }
    }
    for (int i = 0; i < NEL; ++i) {
        double value = u[i];
        for (int j = 0; j < i; ++j) {
            value -= schur_lu[i * NEL + j] * u[j];
        }
        u[i] = value;
    }
    for (int i = NEL - 1; i >= 0; --i) {
        double value = u[i];
        for (int j = i + 1; j < NEL; ++j) {
            value -= schur_lu[i * NEL + j] * u[j];
        }
        u[i] = value / schur_lu[i * NEL + i];
    }

    // Recover qx/qy for this column in the same storage originally used for the
    // mixed flux RHS.  The emitted HDG flux only needs u, qx, and qy values.
    for (int i = 0; i < NEL; ++i) {
        double value0 = 0.0;
        double value1 = 0.0;
        for (int j = 0; j < NEL; ++j) {
            value0 += d0[i * NEL + j] * u[j];
            value1 += d1[i * NEL + j] * u[j];
        }
        tmp1[i] = value0 - qx[i];
        tmp2[i] = value1 - qy[i];
    }
    for (int i = 0; i < NEL; ++i) {
        double value0 = 0.0;
        double value1 = 0.0;
        for (int k = 0; k < NEL; ++k) {
            value0 += mass_inverse[i * NEL + k] * tmp1[k];
            value1 += mass_inverse[i * NEL + k] * tmp2[k];
        }
        qx[i] = jac_inv * value0;
        qy[i] = jac_inv * value1;
    }
}

// Evaluate the contribution of one solved local column to one oriented trace
// test row.  This is the row-side HDG numerical flux pairing used for both
// source RHS emission and trace matrix/boundary-column emission.
__device__ __forceinline__ double trace_flux_value_raw(
        const double* __restrict__ u,
        const double* __restrict__ qx,
        const double* __restrict__ qy,
        const double* __restrict__ oriented_lifts,
        const double* __restrict__ jacs_el_fc,
        const double* __restrict__ normals,
        const long long* __restrict__ loc2oriented_face_coupling,
        const double tau,
        const long long element,
        const int row_face,
        const int row_dof)
{
    const long long oriented_face = loc2oriented_face_coupling[element * 3 + row_face];
    const double scale = jacs_el_fc[element * 3 + row_face];
    const double nx = normals[(element * 3 + row_face) * 2 + 0];
    const double ny = normals[(element * 3 + row_face) * 2 + 1];
    double value = 0.0;
    for (int i = 0; i < NEL; ++i) {
        const double lift = scale * oriented_lifts[(oriented_face * NTR + row_dof) * NEL + i];
        value += tau * lift * u[i];
        value += nx * lift * qx[i];
        value += ny * lift * qy[i];
    }
    return value;
}

extern "C" __global__ void assemble_diffusion_raw(
        long long* __restrict__ rows,
        long long* __restrict__ cols,
        double* __restrict__ data,
        double* __restrict__ rhs,
        const long long* __restrict__ loc2glob_edge,
        const bool* __restrict__ orientations,
        const long long* __restrict__ loc2oriented_face_coupling,
        const long long* __restrict__ interior_side_index,
        const long long* __restrict__ edge_to_solve_edge,
        const long long* __restrict__ int_edges,
        const long long* __restrict__ side_flux_offsets,
        const double* __restrict__ aff_mats,
        const double* __restrict__ aff_jacs,
        const double* __restrict__ jacs_el_fc,
        const double* __restrict__ edge_jacs,
        const double* __restrict__ normals,
        const double* __restrict__ mass_matrix,
        const double* __restrict__ mass_inverse,
        const double* __restrict__ face_element_mass,
        const double* __restrict__ face_element_trace,
        const double* __restrict__ edge_mass,
        const double* __restrict__ oriented_lifts,
        const double* __restrict__ d0_reference,
        const double* __restrict__ d1_reference,
        const double* __restrict__ source_rhs,
        const double* __restrict__ boundary_trace,
        const double tau,
        const long long num_elements,
        const long long num_int_edges,
        const long long n_flux)
{
    // Shared arrays are private to this element block.  The p=6-safe v2 layout
    // stores only the factorized Schur matrix, geometry-dependent derivative
    // blocks, and one active RHS/solution column.  It intentionally avoids the
    // old all-columns layout, which exceeded the Quadro RTX 6000 shared-memory
    // limit at p=6.
    extern __shared__ unsigned char shared_raw[];
    double* shared = reinterpret_cast<double*>(shared_raw);
    double* schur_matrix = shared;
    double* d0 = schur_matrix + (NEL * NEL);
    double* d1 = d0 + (NEL * NEL);
    double* mn0 = d1 + (NEL * NEL);
    double* mn1 = mn0 + (NEL * NEL);
    double* k_d0 = mn1 + (NEL * NEL);
    double* k_d1 = k_d0 + (NEL * NEL);
    double* rhs0 = k_d1 + (NEL * NEL);
    double* qx = rhs0 + NEL;
    double* qy = qx + NEL;
    double* u = qy + NEL;
    double* tmp1 = u + NEL;
    double* tmp2 = tmp1 + NEL;
    int* pivots = reinterpret_cast<int*>(tmp2 + NEL);

    const long long element = blockIdx.x;

    if (element < num_elements && threadIdx.x == 0) {
        const double aff00 = aff_mats[(element * 2 + 0) * 2 + 0];
        const double aff01 = aff_mats[(element * 2 + 0) * 2 + 1];
        const double aff10 = aff_mats[(element * 2 + 1) * 2 + 0];
        const double aff11 = aff_mats[(element * 2 + 1) * 2 + 1];
        const double jac = aff_jacs[element];
        const double jac_inv = 1.0 / jac;

        // Build the scalar Schur operator S once for this element.  d0/d1 are
        // physical derivative matrices, while mn0/mn1 are the boundary-normal
        // mass terms minus those derivatives.
        for (int i = 0; i < NEL; ++i) {
            for (int j = 0; j < NEL; ++j) {
                const long long ij = i * NEL + j;
                const double d0v = aff11 * d0_reference[ij] - aff10 * d1_reference[ij];
                const double d1v = -aff01 * d0_reference[ij] + aff00 * d1_reference[ij];
                double normal_x_value = 0.0;
                double normal_y_value = 0.0;
                double tau_value = 0.0;
                for (int face = 0; face < 3; ++face) {
                    const double face_mass_value = face_element_mass[(face * NEL + i) * NEL + j];
                    const double face_scale = jacs_el_fc[element * 3 + face];
                    tau_value += tau * face_scale * face_mass_value;
                    normal_x_value += face_scale * normals[(element * 3 + face) * 2 + 0] * face_mass_value;
                    normal_y_value += face_scale * normals[(element * 3 + face) * 2 + 1] * face_mass_value;
                }
                d0[ij] = d0v;
                d1[ij] = d1v;
                mn0[ij] = normal_x_value - d0v;
                mn1[ij] = normal_y_value - d1v;
                schur_matrix[ij] = tau_value;
            }
        }

        // Precompute M^{-1}D once.  This costs two extra NEL x NEL shared
        // arrays, but p=6 still fits below this GPU's 64 KiB opt-in shared
        // memory limit and avoids an otherwise dominant O(NEL^4) recomputation.
        for (int i = 0; i < NEL; ++i) {
            for (int j = 0; j < NEL; ++j) {
                double value0 = 0.0;
                double value1 = 0.0;
                for (int k = 0; k < NEL; ++k) {
                    value0 += mass_inverse[i * NEL + k] * d0[k * NEL + j];
                    value1 += mass_inverse[i * NEL + k] * d1[k * NEL + j];
                }
                k_d0[i * NEL + j] = value0;
                k_d1[i * NEL + j] = value1;
            }
        }
        for (int i = 0; i < NEL; ++i) {
            for (int j = 0; j < NEL; ++j) {
                double value0 = 0.0;
                double value1 = 0.0;
                for (int k = 0; k < NEL; ++k) {
                    value0 += mn0[i * NEL + k] * k_d0[k * NEL + j];
                    value1 += mn1[i * NEL + k] * k_d1[k * NEL + j];
                }
                schur_matrix[i * NEL + j] += jac_inv * (value0 + value1);
            }
        }

        // In-place dense partial-pivot LU.  The pivot sequence is reused for
        // every trace/source column solved below; no local inverse is formed.
        for (int k = 0; k < NEL; ++k) {
            int pivot = k;
            double max_value = fabs(schur_matrix[k * NEL + k]);
            for (int i = k + 1; i < NEL; ++i) {
                const double value = fabs(schur_matrix[i * NEL + k]);
                if (value > max_value) {
                    max_value = value;
                    pivot = i;
                }
            }
            pivots[k] = pivot;
            if (pivot != k) {
                for (int j = 0; j < NEL; ++j) {
                    const double tmp = schur_matrix[k * NEL + j];
                    schur_matrix[k * NEL + j] = schur_matrix[pivot * NEL + j];
                    schur_matrix[pivot * NEL + j] = tmp;
                }
            }
            double diagonal = schur_matrix[k * NEL + k];
            if (fabs(diagonal) < 1.0e-30) {
                diagonal = diagonal >= 0.0 ? 1.0e-30 : -1.0e-30;
                schur_matrix[k * NEL + k] = diagonal;
            }
            for (int i = k + 1; i < NEL; ++i) {
                schur_matrix[i * NEL + k] /= diagonal;
                const double multiplier = schur_matrix[i * NEL + k];
                for (int j = k + 1; j < NEL; ++j) {
                    schur_matrix[i * NEL + j] -= multiplier * schur_matrix[k * NEL + j];
                }
            }
        }

        // Source column: solve once and add its numerical flux contribution to
        // the reduced RHS for every interior row side of this element.
        solve_condensed_column_raw(
            schur_matrix, pivots, d0, d1, mn0, mn1, rhs0, qx, qy, u, tmp1, tmp2,
            mass_inverse, face_element_trace, source_rhs, jacs_el_fc, normals,
            tau, jac_inv, element, 1, 0, 0);
        for (int row_face = 0; row_face < 3; ++row_face) {
            const long long side_id = interior_side_index[element * 3 + row_face];
            const long long row_edge = loc2glob_edge[element * 3 + row_face];
            const long long row_solve_edge = edge_to_solve_edge[row_edge];
            if (side_id < 0 || row_solve_edge < 0) {
                continue;
            }
            for (int row_dof = 0; row_dof < NTR; ++row_dof) {
                const double rhs_value = trace_flux_value_raw(
                    u, qx, qy, oriented_lifts, jacs_el_fc, normals,
                    loc2oriented_face_coupling, tau, element, row_face, row_dof);
                atomicAdd(&rhs[row_solve_edge * NTR + row_dof], rhs_value);
            }
        }

        // Trace columns: solve one local trace coordinate at a time, then either
        // write its free-column COO entries or immediately eliminate the known
        // boundary trace value into the RHS.  This is the memory-saving step that
        // replaces the old all-local-columns array.
        for (int col_face = 0; col_face < 3; ++col_face) {
            const long long col_edge = loc2glob_edge[element * 3 + col_face];
            const long long col_solve_edge = edge_to_solve_edge[col_edge];
            const bool positive = orientations[element * 3 + col_face];
            int col_block_pos = 0;
            for (int face = 0; face < col_face; ++face) {
                const long long edge = loc2glob_edge[element * 3 + face];
                if (edge_to_solve_edge[edge] >= 0) {
                    col_block_pos += 1;
                }
            }
            for (int col_dof = 0; col_dof < NTR; ++col_dof) {
                const int local_col_dof = raw_trace_local_dof(positive, col_dof, NTR);
                const double col_sign = raw_trace_orientation_sign(positive, col_dof);
                solve_condensed_column_raw(
                    schur_matrix, pivots, d0, d1, mn0, mn1, rhs0, qx, qy, u, tmp1, tmp2,
                    mass_inverse, face_element_trace, source_rhs, jacs_el_fc, normals,
                    tau, jac_inv, element, 0, col_face, local_col_dof);

                for (int row_face = 0; row_face < 3; ++row_face) {
                    const long long side_id = interior_side_index[element * 3 + row_face];
                    const long long row_edge = loc2glob_edge[element * 3 + row_face];
                    const long long row_solve_edge = edge_to_solve_edge[row_edge];
                    if (side_id < 0 || row_solve_edge < 0) {
                        continue;
                    }
                    const long long side_base = side_flux_offsets[side_id];
                    for (int row_dof = 0; row_dof < NTR; ++row_dof) {
                        const double schur_value = col_sign * trace_flux_value_raw(
                            u, qx, qy, oriented_lifts, jacs_el_fc, normals,
                            loc2oriented_face_coupling, tau, element, row_face, row_dof);
                        if (col_solve_edge >= 0) {
                            const long long out = side_base + ((long long)col_block_pos * NTR + row_dof) * NTR + col_dof;
                            rows[out] = row_solve_edge * NTR + row_dof;
                            cols[out] = col_solve_edge * NTR + col_dof;
                            data[out] = -schur_value;
                        } else {
                            const double trace_value = boundary_trace[col_edge * NTR + col_dof];
                            atomicAdd(&rhs[row_solve_edge * NTR + row_dof], schur_value * trace_value);
                        }
                    }
                }
            }
        }
    }

    // Emit the interior trace mass contribution once per global interior edge.
    // This part is independent of the element-local solves and is appended after
    // the flux COO segment using the precomputed n_flux offset.
    if (element < num_int_edges && threadIdx.x == 0) {
        const long long edge = int_edges[element];
        const long long solve_edge = edge_to_solve_edge[edge];
        const double scale = 2.0 * tau * edge_jacs[edge];
        const long long base = n_flux + element * NTR * NTR;
        for (int i = 0; i < NTR; ++i) {
            for (int j = 0; j < NTR; ++j) {
                const long long out = base + i * NTR + j;
                rows[out] = solve_edge * NTR + i;
                cols[out] = solve_edge * NTR + j;
                data[out] = scale * edge_mass[i * NTR + j];
            }
        }
    }
}
"""


_RAW_ASSEMBLY_COOP_TEMPLATE = r"""
#ifndef RAW_MATRIX_CSR
#define RAW_MATRIX_CSR 0
#endif
#ifndef RAW_RHS_ONLY
#define RAW_RHS_ONLY 0
#endif
#ifndef RAW_SOURCE_ONLY
#define RAW_SOURCE_ONLY 0
#endif

__device__ __forceinline__ void factor_diffusion_schur_lu_coop_raw(
        double* __restrict__ schur_lu,
        int* __restrict__ pivots)
{
    const int tid = threadIdx.x;
    for (int k = 0; k < NEL; ++k) {
        if (tid == 0) {
            int pivot = k;
            double max_value = fabs(schur_lu[k * NEL + k]);
            for (int i = k + 1; i < NEL; ++i) {
                const double value = fabs(schur_lu[i * NEL + k]);
                if (value > max_value) {
                    max_value = value;
                    pivot = i;
                }
            }
            pivots[k] = pivot;
            if (pivot != k) {
                for (int j = 0; j < NEL; ++j) {
                    const double tmp = schur_lu[k * NEL + j];
                    schur_lu[k * NEL + j] = schur_lu[pivot * NEL + j];
                    schur_lu[pivot * NEL + j] = tmp;
                }
            }
            double diagonal = schur_lu[k * NEL + k];
            if (fabs(diagonal) < 1.0e-30) {
                diagonal = diagonal >= 0.0 ? 1.0e-30 : -1.0e-30;
                schur_lu[k * NEL + k] = diagonal;
            }
            for (int i = k + 1; i < NEL; ++i) {
                schur_lu[i * NEL + k] /= diagonal;
            }
        }
        __syncthreads();

        const int width = NEL - k - 1;
        for (int idx = tid; idx < width * width; idx += blockDim.x) {
            const int i = k + 1 + idx / width;
            const int j = k + 1 + idx - (idx / width) * width;
            schur_lu[i * NEL + j] -= schur_lu[i * NEL + k] * schur_lu[k * NEL + j];
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void solve_diffusion_all_columns_coop_raw(
        const double* __restrict__ schur_lu,
        const int* __restrict__ pivots,
        double* __restrict__ columns)
{
    const int tid = threadIdx.x;
    for (int k = 0; k < NEL; ++k) {
        const int pivot = pivots[k];
        if (pivot != k) {
            for (int col = tid; col < NCOLS; col += blockDim.x) {
                const double tmp = columns[k * NCOLS + col];
                columns[k * NCOLS + col] = columns[pivot * NCOLS + col];
                columns[pivot * NCOLS + col] = tmp;
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < NEL; ++i) {
        for (int col = tid; col < NCOLS; col += blockDim.x) {
            double value = columns[i * NCOLS + col];
            for (int j = 0; j < i; ++j) {
                value -= schur_lu[i * NEL + j] * columns[j * NCOLS + col];
            }
            columns[i * NCOLS + col] = value;
        }
        __syncthreads();
    }
    for (int i = NEL - 1; i >= 0; --i) {
        for (int col = tid; col < NCOLS; col += blockDim.x) {
            double value = columns[i * NCOLS + col];
            for (int j = i + 1; j < NEL; ++j) {
                value -= schur_lu[i * NEL + j] * columns[j * NCOLS + col];
            }
            columns[i * NCOLS + col] = value / schur_lu[i * NEL + i];
        }
        __syncthreads();
    }
}

extern "C" __global__ void assemble_diffusion_raw_coop(
        long long* __restrict__ rows,
        long long* __restrict__ cols,
        const int* __restrict__ csr_indptr,
        double* __restrict__ data,
        double* __restrict__ rhs,
        const long long* __restrict__ loc2glob_edge,
        const bool* __restrict__ orientations,
        const long long* __restrict__ loc2oriented_face_coupling,
        const long long* __restrict__ interior_side_index,
        const long long* __restrict__ edge_to_solve_edge,
        const long long* __restrict__ int_edges,
        const long long* __restrict__ side_flux_offsets,
        const int* __restrict__ side_csr_block_pos,
        const int* __restrict__ mass_csr_block_pos,
        const double* __restrict__ aff_mats,
        const double* __restrict__ aff_jacs,
        const double* __restrict__ jacs_el_fc,
        const double* __restrict__ edge_jacs,
        const double* __restrict__ normals,
        const double* __restrict__ mass_matrix,
        const double* __restrict__ mass_inverse,
        const double* __restrict__ face_element_mass,
        const double* __restrict__ face_element_trace,
        const double* __restrict__ edge_mass,
        const double* __restrict__ oriented_lifts,
        const double* __restrict__ d0_reference,
        const double* __restrict__ d1_reference,
        const double* __restrict__ source_rhs,
        const double* __restrict__ boundary_trace,
        const double tau,
        const long long num_elements,
        const long long num_int_edges,
        const long long n_flux)
{
    extern __shared__ unsigned char shared_raw[];
    double* shared = reinterpret_cast<double*>(shared_raw);
    double* schur_matrix = shared;
    double* d0 = schur_matrix + (NEL * NEL);
    double* d1 = d0 + (NEL * NEL);
    double* mn0 = d1 + (NEL * NEL);
    double* mn1 = mn0 + (NEL * NEL);
    double* k_d0 = mn1 + (NEL * NEL);
    double* k_d1 = k_d0 + (NEL * NEL);
    double* local_rhs = k_d1 + (NEL * NEL);
    double* qx_cols = local_rhs + (NEL * NCOLS);
    double* qy_cols = qx_cols + (NEL * NCOLS);
    double* tmp_cols = qy_cols + (NEL * NCOLS);
    int* pivots = reinterpret_cast<int*>(tmp_cols + (NEL * NCOLS));

    const int tid = threadIdx.x;
    const long long element = blockIdx.x;
    const int source_col = NCOLS - 1;

    if (element < num_elements) {
        const double aff00 = aff_mats[(element * 2 + 0) * 2 + 0];
        const double aff01 = aff_mats[(element * 2 + 0) * 2 + 1];
        const double aff10 = aff_mats[(element * 2 + 1) * 2 + 0];
        const double aff11 = aff_mats[(element * 2 + 1) * 2 + 1];
        const double jac = aff_jacs[element];
        const double jac_inv = 1.0 / jac;

        for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
            const int i = idx / NEL;
            const int j = idx - i * NEL;
            const double d0v = aff11 * d0_reference[idx] - aff10 * d1_reference[idx];
            const double d1v = -aff01 * d0_reference[idx] + aff00 * d1_reference[idx];
            double normal_x_value = 0.0;
            double normal_y_value = 0.0;
            double tau_value = 0.0;
            for (int face = 0; face < 3; ++face) {
                const double face_mass_value = face_element_mass[(face * NEL + i) * NEL + j];
                const double face_scale = jacs_el_fc[element * 3 + face];
                tau_value += tau * face_scale * face_mass_value;
                normal_x_value += face_scale * normals[(element * 3 + face) * 2 + 0] * face_mass_value;
                normal_y_value += face_scale * normals[(element * 3 + face) * 2 + 1] * face_mass_value;
            }
            d0[idx] = d0v;
            d1[idx] = d1v;
            mn0[idx] = normal_x_value - d0v;
            mn1[idx] = normal_y_value - d1v;
            schur_matrix[idx] = tau_value;
        }
        __syncthreads();

        for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
            const int i = idx / NEL;
            const int j = idx - i * NEL;
            double value0 = 0.0;
            double value1 = 0.0;
            for (int k = 0; k < NEL; ++k) {
                value0 += mass_inverse[i * NEL + k] * d0[k * NEL + j];
                value1 += mass_inverse[i * NEL + k] * d1[k * NEL + j];
            }
            k_d0[idx] = value0;
            k_d1[idx] = value1;
        }
        __syncthreads();

        for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
            const int i = idx / NEL;
            const int j = idx - i * NEL;
            double value0 = 0.0;
            double value1 = 0.0;
            for (int k = 0; k < NEL; ++k) {
                value0 += mn0[i * NEL + k] * k_d0[k * NEL + j];
                value1 += mn1[i * NEL + k] * k_d1[k * NEL + j];
            }
            schur_matrix[idx] += jac_inv * (value0 + value1);
        }

        for (int idx = tid; idx < NEL * NCOLS; idx += blockDim.x) {
            const int i = idx / NCOLS;
            const int col = idx - i * NCOLS;
            double rhs_value = 0.0;
            double qx_value = 0.0;
            double qy_value = 0.0;
            if (col == source_col) {
                rhs_value = source_rhs[element * 3 * NEL + i];
            } else {
                const int col_face = col / NTR;
                const int local_trace_dof = col - col_face * NTR;
                const double face_scale = jacs_el_fc[element * 3 + col_face];
                const double coupling = face_scale * face_element_trace[(col_face * NEL + i) * NTR + local_trace_dof];
                rhs_value = tau * coupling;
                qx_value = normals[(element * 3 + col_face) * 2 + 0] * coupling;
                qy_value = normals[(element * 3 + col_face) * 2 + 1] * coupling;
            }
            local_rhs[idx] = rhs_value;
            qx_cols[idx] = qx_value;
            qy_cols[idx] = qy_value;
        }
        __syncthreads();

        for (int idx = tid; idx < NEL * NCOLS; idx += blockDim.x) {
            const int i = idx / NCOLS;
            const int col = idx - i * NCOLS;
            double value = 0.0;
            for (int k = 0; k < NEL; ++k) {
                value += mass_inverse[i * NEL + k] * qx_cols[k * NCOLS + col];
            }
            tmp_cols[idx] = value;
        }
        __syncthreads();
        for (int idx = tid; idx < NEL * NCOLS; idx += blockDim.x) {
            const int i = idx / NCOLS;
            const int col = idx - i * NCOLS;
            double value = 0.0;
            for (int k = 0; k < NEL; ++k) {
                value += mn0[i * NEL + k] * tmp_cols[k * NCOLS + col];
            }
            local_rhs[idx] += jac_inv * value;
        }
        __syncthreads();

        for (int idx = tid; idx < NEL * NCOLS; idx += blockDim.x) {
            const int i = idx / NCOLS;
            const int col = idx - i * NCOLS;
            double value = 0.0;
            for (int k = 0; k < NEL; ++k) {
                value += mass_inverse[i * NEL + k] * qy_cols[k * NCOLS + col];
            }
            tmp_cols[idx] = value;
        }
        __syncthreads();
        for (int idx = tid; idx < NEL * NCOLS; idx += blockDim.x) {
            const int i = idx / NCOLS;
            const int col = idx - i * NCOLS;
            double value = 0.0;
            for (int k = 0; k < NEL; ++k) {
                value += mn1[i * NEL + k] * tmp_cols[k * NCOLS + col];
            }
            local_rhs[idx] += jac_inv * value;
        }
        __syncthreads();

        factor_diffusion_schur_lu_coop_raw(schur_matrix, pivots);
        solve_diffusion_all_columns_coop_raw(schur_matrix, pivots, local_rhs);

        for (int idx = tid; idx < NEL * NCOLS; idx += blockDim.x) {
            const int i = idx / NCOLS;
            const int col = idx - i * NCOLS;
            double value = 0.0;
            for (int j = 0; j < NEL; ++j) {
                value += d0[i * NEL + j] * local_rhs[j * NCOLS + col];
            }
            tmp_cols[idx] = value - qx_cols[idx];
        }
        __syncthreads();
        for (int idx = tid; idx < NEL * NCOLS; idx += blockDim.x) {
            const int i = idx / NCOLS;
            const int col = idx - i * NCOLS;
            double value = 0.0;
            for (int k = 0; k < NEL; ++k) {
                value += mass_inverse[i * NEL + k] * tmp_cols[k * NCOLS + col];
            }
            qx_cols[idx] = jac_inv * value;
        }
        __syncthreads();

        for (int idx = tid; idx < NEL * NCOLS; idx += blockDim.x) {
            const int i = idx / NCOLS;
            const int col = idx - i * NCOLS;
            double value = 0.0;
            for (int j = 0; j < NEL; ++j) {
                value += d1[i * NEL + j] * local_rhs[j * NCOLS + col];
            }
            tmp_cols[idx] = value - qy_cols[idx];
        }
        __syncthreads();
        for (int idx = tid; idx < NEL * NCOLS; idx += blockDim.x) {
            const int i = idx / NCOLS;
            const int col = idx - i * NCOLS;
            double value = 0.0;
            for (int k = 0; k < NEL; ++k) {
                value += mass_inverse[i * NEL + k] * tmp_cols[k * NCOLS + col];
            }
            qy_cols[idx] = jac_inv * value;
        }
        __syncthreads();

        for (int task = tid; task < 3 * NTR; task += blockDim.x) {
            const int row_face = task / NTR;
            const int row_dof = task - row_face * NTR;
            const long long side_id = interior_side_index[element * 3 + row_face];
            const long long row_edge = loc2glob_edge[element * 3 + row_face];
            const long long row_solve_edge = edge_to_solve_edge[row_edge];
            if (side_id < 0 || row_solve_edge < 0) {
                continue;
            }

#if !RAW_MATRIX_CSR
            const long long side_base = side_flux_offsets[side_id];
#endif
            const long long oriented_face = loc2oriented_face_coupling[element * 3 + row_face];
            const double scale = jacs_el_fc[element * 3 + row_face];
            const double nx = normals[(element * 3 + row_face) * 2 + 0];
            const double ny = normals[(element * 3 + row_face) * 2 + 1];
            double lift_values[NEL];
            double rhs_value = 0.0;
            for (int i = 0; i < NEL; ++i) {
                const double lift = scale * oriented_lifts[(oriented_face * NTR + row_dof) * NEL + i];
                lift_values[i] = lift;
                rhs_value += tau * lift * local_rhs[i * NCOLS + source_col];
                rhs_value += nx * lift * qx_cols[i * NCOLS + source_col];
                rhs_value += ny * lift * qy_cols[i * NCOLS + source_col];
            }

#if !RAW_SOURCE_ONLY
            int col_block_pos = 0;
            for (int col_face = 0; col_face < 3; ++col_face) {
                const long long col_edge = loc2glob_edge[element * 3 + col_face];
                const long long col_solve_edge = edge_to_solve_edge[col_edge];
                const bool positive = orientations[element * 3 + col_face];
                for (int col_dof = 0; col_dof < NTR; ++col_dof) {
                    const int local_col_dof = raw_trace_local_dof(positive, col_dof, NTR);
                    const double col_sign = raw_trace_orientation_sign(positive, col_dof);
                    const int column = col_face * NTR + local_col_dof;
                    double schur_value = 0.0;
                    for (int i = 0; i < NEL; ++i) {
                        const double lift = lift_values[i];
                        schur_value += tau * lift * local_rhs[i * NCOLS + column];
                        schur_value += nx * lift * qx_cols[i * NCOLS + column];
                        schur_value += ny * lift * qy_cols[i * NCOLS + column];
                    }
                    schur_value *= col_sign;
                    if (col_solve_edge >= 0) {
#if !RAW_RHS_ONLY
#if RAW_MATRIX_CSR
                        const int block_pos = side_csr_block_pos[side_id * 3 + col_face];
                        const long long row = row_solve_edge * NTR + row_dof;
                        const long long out = (long long)csr_indptr[row] + ((long long)block_pos * NTR + col_dof);
                        atomicAdd(&data[out], -schur_value);
#else
                        const long long out = side_base + ((long long)col_block_pos * NTR + row_dof) * NTR + col_dof;
                        rows[out] = row_solve_edge * NTR + row_dof;
                        cols[out] = col_solve_edge * NTR + col_dof;
                        data[out] = -schur_value;
#endif
#endif
                    } else {
                        rhs_value += schur_value * boundary_trace[col_edge * NTR + col_dof];
                    }
                }
                if (col_solve_edge >= 0) {
                    col_block_pos += 1;
                }
            }
#endif
            atomicAdd(&rhs[row_solve_edge * NTR + row_dof], rhs_value);
        }
    }

    if (element < num_int_edges) {
        const long long edge = int_edges[element];
        const long long solve_edge = edge_to_solve_edge[edge];
        const double scale = 2.0 * tau * edge_jacs[edge];
#if RAW_MATRIX_CSR
        const int block_pos = mass_csr_block_pos[solve_edge];
#endif
        for (int idx = tid; idx < NTR * NTR; idx += blockDim.x) {
            const int i = idx / NTR;
            const int j = idx - i * NTR;
#if !RAW_RHS_ONLY
#if RAW_MATRIX_CSR
            const long long row = solve_edge * NTR + i;
            const long long out = (long long)csr_indptr[row] + ((long long)block_pos * NTR + j);
            atomicAdd(&data[out], scale * edge_mass[i * NTR + j]);
#else
            const long long base = n_flux + element * NTR * NTR;
            const long long out = base + idx;
            rows[out] = solve_edge * NTR + i;
            cols[out] = solve_edge * NTR + j;
            data[out] = scale * edge_mass[i * NTR + j];
#endif
#endif
        }
    }
}
"""


def _raw_assembly_csr_template() -> str:
    """Convert the raw diffusion assembly template from COO emission to CSR updates."""
    source = _RAW_ASSEMBLY_TEMPLATE
    source = source.replace(
        'extern "C" __global__ void assemble_diffusion_raw(\n        long long* __restrict__ rows,\n        long long* __restrict__ cols,\n        double* __restrict__ data,',
        'extern "C" __global__ void assemble_diffusion_raw_csr(\n        const int* __restrict__ csr_indptr,\n        double* __restrict__ data,',
    )
    source = source.replace(
        '        const long long* __restrict__ int_edges,\n        const long long* __restrict__ side_flux_offsets,',
        '        const long long* __restrict__ int_edges,\n        const int* __restrict__ side_csr_block_pos,\n        const int* __restrict__ mass_csr_block_pos,',
    )
    source = source.replace(
        '                    const long long side_base = side_flux_offsets[side_id];\n                    for (int row_dof = 0; row_dof < NTR; ++row_dof) {',
        '                    const int block_pos = col_solve_edge >= 0 ? side_csr_block_pos[side_id * 3 + col_face] : -1;\n                    for (int row_dof = 0; row_dof < NTR; ++row_dof) {',
    )
    source = source.replace(
        '                            const long long out = side_base + ((long long)col_block_pos * NTR + row_dof) * NTR + col_dof;\n                            rows[out] = row_solve_edge * NTR + row_dof;\n                            cols[out] = col_solve_edge * NTR + col_dof;\n                            data[out] = -schur_value;',
        '                            const long long row = row_solve_edge * NTR + row_dof;\n                            const long long out = (long long)csr_indptr[row] + ((long long)block_pos * NTR + col_dof);\n                            atomicAdd(&data[out], -schur_value);',
    )
    source = source.replace(
        '        const long long base = n_flux + element * NTR * NTR;\n        for (int i = 0; i < NTR; ++i) {\n            for (int j = 0; j < NTR; ++j) {\n                const long long out = base + i * NTR + j;\n                rows[out] = solve_edge * NTR + i;\n                cols[out] = solve_edge * NTR + j;\n                data[out] = scale * edge_mass[i * NTR + j];\n            }\n        }',
        '        const int block_pos = mass_csr_block_pos[solve_edge];\n        for (int i = 0; i < NTR; ++i) {\n            const long long row = solve_edge * NTR + i;\n            for (int j = 0; j < NTR; ++j) {\n                const long long out = (long long)csr_indptr[row] + ((long long)block_pos * NTR + j);\n                atomicAdd(&data[out], scale * edge_mass[i * NTR + j]);\n            }\n        }',
    )
    return source


_RAW_RECONSTRUCT_TEMPLATE = r"""
extern "C" __global__ void reconstruct_diffusion_raw(
        double* __restrict__ uh,
        double* __restrict__ local_unknowns,
        const int write_local_unknowns,
        const double* __restrict__ trace,
        const long long* __restrict__ loc2glob_edge,
        const long long* __restrict__ loc2oriented_face_coupling,
        const double* __restrict__ aff_mats,
        const double* __restrict__ aff_jacs,
        const double* __restrict__ jacs_el_fc,
        const double* __restrict__ normals,
        const double* __restrict__ mass_inverse,
        const double* __restrict__ face_element_mass,
        const double* __restrict__ face_element_trace,
        const double* __restrict__ d0_reference,
        const double* __restrict__ d1_reference,
        const double* __restrict__ source_rhs,
        const double tau,
        const long long num_elements)
{
    // Reconstruction is the single-RHS version of the same element-local solve.
    // It consumes the full trace vector, including prescribed boundary trace
    // dofs, and writes only primal u_h coefficients.
    extern __shared__ unsigned char shared_raw[];
    double* shared = reinterpret_cast<double*>(shared_raw);
    double* schur_matrix = shared;
    double* d0 = schur_matrix + (NEL * NEL);
    double* d1 = d0 + (NEL * NEL);
    double* mn0 = d1 + (NEL * NEL);
    double* mn1 = mn0 + (NEL * NEL);
    double* k_d0 = mn1 + (NEL * NEL);
    double* k_d1 = k_d0 + (NEL * NEL);
    double* rhs0 = k_d1 + (NEL * NEL);
    double* rhs1 = rhs0 + NEL;
    double* rhs2 = rhs1 + NEL;
    double* red_rhs = rhs2 + NEL;
    double* tmp1 = red_rhs + NEL;
    double* tmp2 = tmp1 + NEL;
    int* pivots = reinterpret_cast<int*>(tmp2 + NEL);

    const long long element = blockIdx.x;
    if (element >= num_elements || threadIdx.x != 0) {
        return;
    }

    const double aff00 = aff_mats[(element * 2 + 0) * 2 + 0];
    const double aff01 = aff_mats[(element * 2 + 0) * 2 + 1];
    const double aff10 = aff_mats[(element * 2 + 1) * 2 + 0];
    const double aff11 = aff_mats[(element * 2 + 1) * 2 + 1];
    const double jac_inv = 1.0 / aff_jacs[element];

    for (int i = 0; i < NEL; ++i) {
        rhs0[i] = source_rhs[element * 3 * NEL + i];
        rhs1[i] = 0.0;
        rhs2[i] = 0.0;
        for (int j = 0; j < NEL; ++j) {
            const long long ij = i * NEL + j;
            const double d0v = aff11 * d0_reference[ij] - aff10 * d1_reference[ij];
            const double d1v = -aff01 * d0_reference[ij] + aff00 * d1_reference[ij];
            double normal_x_value = 0.0;
            double normal_y_value = 0.0;
            double tau_value = 0.0;
            for (int face = 0; face < 3; ++face) {
                const double face_mass_value = face_element_mass[(face * NEL + i) * NEL + j];
                const double face_scale = jacs_el_fc[element * 3 + face];
                tau_value += tau * face_scale * face_mass_value;
                normal_x_value += face_scale * normals[(element * 3 + face) * 2 + 0] * face_mass_value;
                normal_y_value += face_scale * normals[(element * 3 + face) * 2 + 1] * face_mass_value;
            }
            d0[ij] = d0v;
            d1[ij] = d1v;
            mn0[ij] = normal_x_value - d0v;
            mn1[ij] = normal_y_value - d1v;
            schur_matrix[ij] = tau_value;
        }
    }

    // Add trace contribution B * lambda to the local RHS.  Use the same
    // oriented lift table as Schur assembly; a simple dof reversal is not
    // sufficiently explicit once loc2oriented_face_coupling has encoded both
    // local face id and edge orientation.
    for (int face = 0; face < 3; ++face) {
        const long long edge = loc2glob_edge[element * 3 + face];
        const long long oriented_face = loc2oriented_face_coupling[element * 3 + face];
        const double face_scale = jacs_el_fc[element * 3 + face];
        const double normal_x = normals[(element * 3 + face) * 2 + 0];
        const double normal_y = normals[(element * 3 + face) * 2 + 1];
        for (int trace_dof = 0; trace_dof < NTR; ++trace_dof) {
            const double trace_value = trace[edge * NTR + trace_dof];
            for (int i = 0; i < NEL; ++i) {
                const double coupling = face_scale * face_element_trace[(oriented_face * NTR + trace_dof) * NEL + i];
                rhs0[i] += tau * coupling * trace_value;
                rhs1[i] += normal_x * coupling * trace_value;
                rhs2[i] += normal_y * coupling * trace_value;
            }
        }
    }

    for (int i = 0; i < NEL; ++i) {
        for (int j = 0; j < NEL; ++j) {
            double value0 = 0.0;
            double value1 = 0.0;
            for (int k = 0; k < NEL; ++k) {
                value0 += mass_inverse[i * NEL + k] * d0[k * NEL + j];
                value1 += mass_inverse[i * NEL + k] * d1[k * NEL + j];
            }
            k_d0[i * NEL + j] = value0;
            k_d1[i * NEL + j] = value1;
        }
    }
    for (int i = 0; i < NEL; ++i) {
        for (int j = 0; j < NEL; ++j) {
            double value0 = 0.0;
            double value1 = 0.0;
            for (int k = 0; k < NEL; ++k) {
                value0 += mn0[i * NEL + k] * k_d0[k * NEL + j];
                value1 += mn1[i * NEL + k] * k_d1[k * NEL + j];
            }
            schur_matrix[i * NEL + j] += jac_inv * (value0 + value1);
        }
    }

    for (int i = 0; i < NEL; ++i) {
        double value1 = 0.0;
        double value2 = 0.0;
        for (int k = 0; k < NEL; ++k) {
            value1 += mass_inverse[i * NEL + k] * rhs1[k];
            value2 += mass_inverse[i * NEL + k] * rhs2[k];
        }
        tmp1[i] = value1;
        tmp2[i] = value2;
    }
    for (int i = 0; i < NEL; ++i) {
        double acc0 = 0.0;
        double acc1 = 0.0;
        for (int k = 0; k < NEL; ++k) {
            acc0 += mn0[i * NEL + k] * tmp1[k];
            acc1 += mn1[i * NEL + k] * tmp2[k];
        }
        red_rhs[i] = rhs0[i] + jac_inv * (acc0 + acc1);
    }

    for (int k = 0; k < NEL; ++k) {
        int pivot = k;
        double max_value = fabs(schur_matrix[k * NEL + k]);
        for (int i = k + 1; i < NEL; ++i) {
            const double value = fabs(schur_matrix[i * NEL + k]);
            if (value > max_value) { max_value = value; pivot = i; }
        }
        pivots[k] = pivot;
        if (pivot != k) {
            for (int j = 0; j < NEL; ++j) {
                const double tmp = schur_matrix[k * NEL + j];
                schur_matrix[k * NEL + j] = schur_matrix[pivot * NEL + j];
                schur_matrix[pivot * NEL + j] = tmp;
            }
        }
        double diagonal = schur_matrix[k * NEL + k];
        if (fabs(diagonal) < 1.0e-30) {
            diagonal = diagonal >= 0.0 ? 1.0e-30 : -1.0e-30;
            schur_matrix[k * NEL + k] = diagonal;
        }
        for (int i = k + 1; i < NEL; ++i) {
            schur_matrix[i * NEL + k] /= diagonal;
            const double multiplier = schur_matrix[i * NEL + k];
            for (int j = k + 1; j < NEL; ++j) {
                schur_matrix[i * NEL + j] -= multiplier * schur_matrix[k * NEL + j];
            }
        }
    }
    for (int k = 0; k < NEL; ++k) {
        const int pivot = pivots[k];
        if (pivot != k) {
            const double tmp = red_rhs[k];
            red_rhs[k] = red_rhs[pivot];
            red_rhs[pivot] = tmp;
        }
    }
    for (int i = 0; i < NEL; ++i) {
        double value = red_rhs[i];
        for (int j = 0; j < i; ++j) { value -= schur_matrix[i * NEL + j] * red_rhs[j]; }
        red_rhs[i] = value;
    }
    for (int i = NEL - 1; i >= 0; --i) {
        double value = red_rhs[i];
        for (int j = i + 1; j < NEL; ++j) { value -= schur_matrix[i * NEL + j] * red_rhs[j]; }
        red_rhs[i] = value / schur_matrix[i * NEL + i];
    }
    if (write_local_unknowns) {
        for (int i = 0; i < NEL; ++i) {
            double value0 = 0.0;
            double value1 = 0.0;
            for (int j = 0; j < NEL; ++j) {
                value0 += d0[i * NEL + j] * red_rhs[j];
                value1 += d1[i * NEL + j] * red_rhs[j];
            }
            tmp1[i] = value0 - rhs1[i];
            tmp2[i] = value1 - rhs2[i];
        }
        for (int i = 0; i < NEL; ++i) {
            double qx_value = 0.0;
            double qy_value = 0.0;
            for (int k = 0; k < NEL; ++k) {
                qx_value += mass_inverse[i * NEL + k] * tmp1[k];
                qy_value += mass_inverse[i * NEL + k] * tmp2[k];
            }
            local_unknowns[element * 3 * NEL + i] = red_rhs[i];
            local_unknowns[element * 3 * NEL + NEL + i] = jac_inv * qx_value;
            local_unknowns[element * 3 * NEL + 2 * NEL + i] = jac_inv * qy_value;
            uh[element * NEL + i] = red_rhs[i];
        }
    } else {
        for (int i = 0; i < NEL; ++i) {
            uh[element * NEL + i] = red_rhs[i];
        }
    }
}
"""


_RAW_RECONSTRUCT_COOP_TEMPLATE = r"""
__device__ __forceinline__ void factor_diffusion_reconstruct_lu_coop_raw(
        double* __restrict__ schur_lu,
        int* __restrict__ pivots)
{
    const int tid = threadIdx.x;
    for (int k = 0; k < NEL; ++k) {
        if (tid == 0) {
            int pivot = k;
            double max_value = fabs(schur_lu[k * NEL + k]);
            for (int i = k + 1; i < NEL; ++i) {
                const double value = fabs(schur_lu[i * NEL + k]);
                if (value > max_value) {
                    max_value = value;
                    pivot = i;
                }
            }
            pivots[k] = pivot;
            if (pivot != k) {
                for (int j = 0; j < NEL; ++j) {
                    const double tmp = schur_lu[k * NEL + j];
                    schur_lu[k * NEL + j] = schur_lu[pivot * NEL + j];
                    schur_lu[pivot * NEL + j] = tmp;
                }
            }
            double diagonal = schur_lu[k * NEL + k];
            if (fabs(diagonal) < 1.0e-30) {
                diagonal = diagonal >= 0.0 ? 1.0e-30 : -1.0e-30;
                schur_lu[k * NEL + k] = diagonal;
            }
            for (int i = k + 1; i < NEL; ++i) {
                schur_lu[i * NEL + k] /= diagonal;
            }
        }
        __syncthreads();

        const int width = NEL - k - 1;
        for (int idx = tid; idx < width * width; idx += blockDim.x) {
            const int i = k + 1 + idx / width;
            const int j = k + 1 + idx - (idx / width) * width;
            schur_lu[i * NEL + j] -= schur_lu[i * NEL + k] * schur_lu[k * NEL + j];
        }
        __syncthreads();
    }
}

extern "C" __global__ void reconstruct_diffusion_raw_coop(
        double* __restrict__ uh,
        double* __restrict__ local_unknowns,
        const int write_local_unknowns,
        const double* __restrict__ trace,
        const long long* __restrict__ loc2glob_edge,
        const long long* __restrict__ loc2oriented_face_coupling,
        const double* __restrict__ aff_mats,
        const double* __restrict__ aff_jacs,
        const double* __restrict__ jacs_el_fc,
        const double* __restrict__ normals,
        const double* __restrict__ mass_inverse,
        const double* __restrict__ face_element_mass,
        const double* __restrict__ face_element_trace,
        const double* __restrict__ d0_reference,
        const double* __restrict__ d1_reference,
        const double* __restrict__ source_rhs,
        const double tau,
        const long long num_elements)
{
    extern __shared__ unsigned char shared_raw[];
    double* shared = reinterpret_cast<double*>(shared_raw);
    double* schur_matrix = shared;
    double* d0 = schur_matrix + (NEL * NEL);
    double* d1 = d0 + (NEL * NEL);
    double* mn0 = d1 + (NEL * NEL);
    double* mn1 = mn0 + (NEL * NEL);
    double* k_d0 = mn1 + (NEL * NEL);
    double* k_d1 = k_d0 + (NEL * NEL);
    double* rhs0 = k_d1 + (NEL * NEL);
    double* rhs1 = rhs0 + NEL;
    double* rhs2 = rhs1 + NEL;
    double* red_rhs = rhs2 + NEL;
    double* tmp1 = red_rhs + NEL;
    double* tmp2 = tmp1 + NEL;
    int* pivots = reinterpret_cast<int*>(tmp2 + NEL);

    const int tid = threadIdx.x;
    const long long element = blockIdx.x;
    if (element >= num_elements) {
        return;
    }

    const double aff00 = aff_mats[(element * 2 + 0) * 2 + 0];
    const double aff01 = aff_mats[(element * 2 + 0) * 2 + 1];
    const double aff10 = aff_mats[(element * 2 + 1) * 2 + 0];
    const double aff11 = aff_mats[(element * 2 + 1) * 2 + 1];
    const double jac_inv = 1.0 / aff_jacs[element];

    for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
        const int i = idx / NEL;
        const int j = idx - i * NEL;
        const double d0v = aff11 * d0_reference[idx] - aff10 * d1_reference[idx];
        const double d1v = -aff01 * d0_reference[idx] + aff00 * d1_reference[idx];
        double normal_x_value = 0.0;
        double normal_y_value = 0.0;
        double tau_value = 0.0;
        for (int face = 0; face < 3; ++face) {
            const double face_mass_value = face_element_mass[(face * NEL + i) * NEL + j];
            const double face_scale = jacs_el_fc[element * 3 + face];
            tau_value += tau * face_scale * face_mass_value;
            normal_x_value += face_scale * normals[(element * 3 + face) * 2 + 0] * face_mass_value;
            normal_y_value += face_scale * normals[(element * 3 + face) * 2 + 1] * face_mass_value;
        }
        d0[idx] = d0v;
        d1[idx] = d1v;
        mn0[idx] = normal_x_value - d0v;
        mn1[idx] = normal_y_value - d1v;
        schur_matrix[idx] = tau_value;
    }

    for (int i = tid; i < NEL; i += blockDim.x) {
        double value0 = source_rhs[element * 3 * NEL + i];
        double value1 = 0.0;
        double value2 = 0.0;
        for (int face = 0; face < 3; ++face) {
            const long long edge = loc2glob_edge[element * 3 + face];
            const long long oriented_face = loc2oriented_face_coupling[element * 3 + face];
            const double face_scale = jacs_el_fc[element * 3 + face];
            const double normal_x = normals[(element * 3 + face) * 2 + 0];
            const double normal_y = normals[(element * 3 + face) * 2 + 1];
            for (int trace_dof = 0; trace_dof < NTR; ++trace_dof) {
                const double trace_value = trace[edge * NTR + trace_dof];
                const double coupling = face_scale * face_element_trace[(oriented_face * NTR + trace_dof) * NEL + i];
                value0 += tau * coupling * trace_value;
                value1 += normal_x * coupling * trace_value;
                value2 += normal_y * coupling * trace_value;
            }
        }
        rhs0[i] = value0;
        rhs1[i] = value1;
        rhs2[i] = value2;
    }
    __syncthreads();

    for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
        const int i = idx / NEL;
        const int j = idx - i * NEL;
        double value0 = 0.0;
        double value1 = 0.0;
        for (int k = 0; k < NEL; ++k) {
            value0 += mass_inverse[i * NEL + k] * d0[k * NEL + j];
            value1 += mass_inverse[i * NEL + k] * d1[k * NEL + j];
        }
        k_d0[idx] = value0;
        k_d1[idx] = value1;
    }
    __syncthreads();

    for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
        const int i = idx / NEL;
        const int j = idx - i * NEL;
        double value0 = 0.0;
        double value1 = 0.0;
        for (int k = 0; k < NEL; ++k) {
            value0 += mn0[i * NEL + k] * k_d0[k * NEL + j];
            value1 += mn1[i * NEL + k] * k_d1[k * NEL + j];
        }
        schur_matrix[idx] += jac_inv * (value0 + value1);
    }

    for (int i = tid; i < NEL; i += blockDim.x) {
        double value1 = 0.0;
        double value2 = 0.0;
        for (int k = 0; k < NEL; ++k) {
            value1 += mass_inverse[i * NEL + k] * rhs1[k];
            value2 += mass_inverse[i * NEL + k] * rhs2[k];
        }
        tmp1[i] = value1;
        tmp2[i] = value2;
    }
    __syncthreads();

    for (int i = tid; i < NEL; i += blockDim.x) {
        double acc0 = 0.0;
        double acc1 = 0.0;
        for (int k = 0; k < NEL; ++k) {
            acc0 += mn0[i * NEL + k] * tmp1[k];
            acc1 += mn1[i * NEL + k] * tmp2[k];
        }
        red_rhs[i] = rhs0[i] + jac_inv * (acc0 + acc1);
    }
    __syncthreads();

    factor_diffusion_reconstruct_lu_coop_raw(schur_matrix, pivots);

    if (tid == 0) {
        for (int k = 0; k < NEL; ++k) {
            const int pivot = pivots[k];
            if (pivot != k) {
                const double tmp = red_rhs[k];
                red_rhs[k] = red_rhs[pivot];
                red_rhs[pivot] = tmp;
            }
        }
        for (int i = 0; i < NEL; ++i) {
            double value = red_rhs[i];
            for (int j = 0; j < i; ++j) {
                value -= schur_matrix[i * NEL + j] * red_rhs[j];
            }
            red_rhs[i] = value;
        }
        for (int i = NEL - 1; i >= 0; --i) {
            double value = red_rhs[i];
            for (int j = i + 1; j < NEL; ++j) {
                value -= schur_matrix[i * NEL + j] * red_rhs[j];
            }
            red_rhs[i] = value / schur_matrix[i * NEL + i];
        }
    }
    __syncthreads();

    if (write_local_unknowns) {
        for (int i = tid; i < NEL; i += blockDim.x) {
            double value0 = 0.0;
            double value1 = 0.0;
            for (int j = 0; j < NEL; ++j) {
                value0 += d0[i * NEL + j] * red_rhs[j];
                value1 += d1[i * NEL + j] * red_rhs[j];
            }
            tmp1[i] = value0 - rhs1[i];
            tmp2[i] = value1 - rhs2[i];
        }
        __syncthreads();

        for (int i = tid; i < NEL; i += blockDim.x) {
            double qx_value = 0.0;
            double qy_value = 0.0;
            for (int k = 0; k < NEL; ++k) {
                qx_value += mass_inverse[i * NEL + k] * tmp1[k];
                qy_value += mass_inverse[i * NEL + k] * tmp2[k];
            }
            local_unknowns[element * 3 * NEL + i] = red_rhs[i];
            local_unknowns[element * 3 * NEL + NEL + i] = jac_inv * qx_value;
            local_unknowns[element * 3 * NEL + 2 * NEL + i] = jac_inv * qy_value;
            uh[element * NEL + i] = red_rhs[i];
        }
    } else {
        for (int i = tid; i < NEL; i += blockDim.x) {
            uh[element * NEL + i] = red_rhs[i];
        }
    }
}
"""


def _kernel_source(
        template: str,
        *,
        nel: int,
        ntr: int,
        ncols: int,
        matrix_format: str = 'coo',
        trace_orientation_mode: int = 0,
        rhs_only: bool = False,
        source_only: bool = False,
) -> str:
    """Build a parameterized CUDA kernel source for raw diffusion assembly."""
    matrix_format = str(matrix_format).lower()
    if matrix_format not in {'coo', 'csr'}:
        raise ValueError("matrix_format must be 'coo' or 'csr'")
    trace_orientation_mode = int(trace_orientation_mode)
    if trace_orientation_mode not in {0, 1}:
        raise ValueError("trace_orientation_mode must be 0 or 1")
    prefix = (
        f"#define RAW_MATRIX_CSR {1 if matrix_format == 'csr' else 0}\n"
        f"#define RAW_RHS_ONLY {1 if rhs_only else 0}\n"
        f"#define RAW_SOURCE_ONLY {1 if source_only else 0}\n"
        f"#define TRACE_ORIENTATION_MODE {trace_orientation_mode}\n"
    )
    source = template.replace('NEL', str(int(nel))).replace('NTR', str(int(ntr))).replace('NCOLS', str(int(ncols)))
    return prefix + _RAW_TRACE_ORIENTATION_HELPERS + source


def _shared_sizes(nel: int, ntr: int) -> tuple[int, int]:
    # Serial assembly stores one RHS/solution column at a time.
    """Compute serial assembly and reconstruction shared-memory requirements."""
    assembly_doubles = 7 * nel * nel + 6 * nel
    assembly_bytes = assembly_doubles * 8 + nel * 4 + 256
    reconstruct_doubles = 7 * nel * nel + 6 * nel
    reconstruct_bytes = reconstruct_doubles * 8 + nel * 4 + 256
    return assembly_bytes, reconstruct_bytes


def _coop_shared_sizes(nel: int, ntr: int, *, ncols: int | None = None) -> int:
    """Compute cooperative assembly shared-memory requirements."""
    # Cooperative assembly stores all condensed trace/source columns plus one
    # temporary column slab for flux recovery.  For p=6 this is just under the
    # 64 KiB opt-in shared-memory limit on the original benchmark GPU.
    ncols = 3 * ntr + 1 if ncols is None else int(ncols)
    assembly_doubles = 7 * nel * nel + 4 * nel * ncols
    return assembly_doubles * 8 + nel * 4 + 256


def _compile_kernel(cupy, source: str, name: str, shared_bytes: int):
    """Compile a raw CUDA kernel and request its dynamic shared-memory budget."""
    kernel = cupy.RawKernel(source, name, options=('--std=c++11',))
    try:
        kernel.max_dynamic_shared_size_bytes = int(shared_bytes)
    except Exception:
        pass
    return kernel


def _edge_to_solve_edge(mesh) -> np.ndarray:
    """Map interior global edges to contiguous reduced solve-edge ids."""
    edge_is_free = np.ones(mesh.num_edg, dtype=bool)
    edge_is_free[mesh.bnd_edges_inds] = False
    free_edges = np.flatnonzero(edge_is_free).astype(np.int64)
    edge_to_solve = np.full(mesh.num_edg, -1, dtype=np.int64)
    edge_to_solve[free_edges] = np.arange(free_edges.size, dtype=np.int64)
    return np.ascontiguousarray(edge_to_solve)


def _interior_side_index(mesh) -> np.ndarray:
    """Map each interior element side to its contiguous side index."""
    index = np.full((mesh.num_tri, 3), -1, dtype=np.int64)
    index[mesh.interior_elements, mesh.interior_faces] = np.arange(mesh.interior_elements.size, dtype=np.int64)
    return np.ascontiguousarray(index)


def _side_flux_offsets(mesh, edge_to_solve_edge: np.ndarray, edg_dof: int) -> np.ndarray:
    """Compute packed flux-block offsets for all interior element sides."""
    face_is_free = edge_to_solve_edge[mesh.loc2glob_edge] >= 0
    side_col_counts = np.count_nonzero(face_is_free[mesh.interior_elements], axis=1).astype(np.int64)
    offsets = np.empty(side_col_counts.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(side_col_counts * edg_dof * edg_dof, out=offsets[1:])
    return np.ascontiguousarray(offsets)


def _raw_trace_orientation_mode(trace_ref) -> int:
    """Select the raw CUDA orientation rule for the active trace basis."""
    kind = getattr(trace_ref, 'kind', '')
    if kind == 'legacy-lagrange' and getattr(trace_ref, 'nodal', False):
        return 0
    if kind == 'legendre-modal' and not getattr(trace_ref, 'nodal', True):
        return 1
    raise ValueError('raw CUDA diffusion assembly currently supports legacy-lagrange nodal and legendre-modal trace bases')


def validate_raw_cuda_supported(cspace, trace_ref) -> None:
    """Validate degree and trace orientation for raw diffusion kernels.

    Raw diffusion kernels currently compile local storage for at most 28
    scalar element DOFs (``p <= 6``) and support nodal legacy-Lagrange or modal
    Legendre traces. The shared capability preflight checks the broader backend
    combination; this helper enforces kernel-shape limits at launch setup.

    Raises
    ------
    ValueError
        If the trace representation or local element size is unsupported.
    """
    _raw_trace_orientation_mode(trace_ref)
    if cspace.el_dof > 28:
        raise ValueError('raw CUDA diffusion assembly currently supports p <= 6 (el_dof <= 28)')



def assemble_projected_diffusion_trace_rhs_eliminated_raw_cuda(
        *,
        source_rhs,
        boundary_trace,
        cspace,
        trace_ref,
        d0_reference,
        d1_reference,
        face_element_mass,
        tau: float,
        csr_pattern,
        block_size: RawCudaBlockSize = "auto",
) -> RawDiffusionAssemblyResult:
    """Assemble only the reduced RHS for a cached raw-CUDA CSR diffusion operator."""
    cupy = require_cupy()
    validate_raw_cuda_supported(cspace, trace_ref)
    if csr_pattern is None:
        raise ValueError('csr_pattern is required for raw-CUDA cached RHS assembly')
    block_size = resolve_raw_cuda_block_size(
        block_size, equation="diffusion-reaction", order=cspace.order
    )
    if block_size not in {1, 32, 64, 128}:
        raise ValueError('raw CUDA diffusion block_size must be one of 1, 32, 64, 128')
    timings: dict[str, float] = {}
    mesh_h = cspace.host.mesh
    nel = int(cspace.el_dof)
    ntr = int(cspace.edg_dof)
    boundary_is_zero = bool(cupy.all(boundary_trace == 0.0).get())
    ncols = 1 if boundary_is_zero else 3 * ntr + 1
    assembly_shared = _shared_sizes(nel, ntr)[0] if block_size == 1 else _coop_shared_sizes(nel, ntr, ncols=ncols)
    trace_orientation_mode = _raw_trace_orientation_mode(trace_ref)

    edge_to_solve = csr_pattern.edge_to_solve_edge
    side_index = csr_pattern.interior_side_index
    indptr = csr_pattern.indptr
    indices = csr_pattern.indices
    side_map_arg = csr_pattern.side_csr_block_pos
    mass_map_arg = csr_pattern.mass_csr_block_pos

    zero_start = time.perf_counter()
    dummy_data = cupy.zeros(1 if block_size != 1 else indices.size, dtype=cupy.float64)
    rhs = cupy.zeros(mesh_h.int_edges_inds.size * ntr, dtype=cupy.float64)
    boundary_trace_full = cupy.zeros((mesh_h.num_edg, ntr), dtype=cupy.float64)
    if mesh_h.bnd_edges_inds.size:
        boundary_trace_full[cspace.mesh.bnd_edges_inds] = boundary_trace
    cupy.cuda.get_current_stream().synchronize()
    timings['raw.cached_rhs_zero'] = time.perf_counter() - zero_start

    start = time.perf_counter()
    if block_size == 1:
        source = _kernel_source(
            _raw_assembly_csr_template(),
            nel=nel,
            ntr=ntr,
            ncols=ncols,
            matrix_format='csr',
            trace_orientation_mode=trace_orientation_mode,
        )
        kernel = _compile_kernel(cupy, source, 'assemble_diffusion_raw_csr', assembly_shared)
        kernel_args = (
            indptr,
            dummy_data,
            rhs,
            cspace.mesh.loc2glob_edge,
            cspace.mesh.orientations,
            cspace.mesh.loc2oriented_face_coupling,
            side_index,
            edge_to_solve,
            cspace.mesh.int_edges_inds,
            side_map_arg,
            mass_map_arg,
            cspace.mesh.aff_mats,
            cspace.mesh.aff_jacs,
            cspace.mesh.jacs_el_fc,
            cspace.mesh.edge_jacs,
            cspace.mesh.normals,
            cspace.quad_data.MKrf,
            cspace.quad_data.MKrf_inv,
            face_element_mass,
            trace_ref.face_element_test_trace_trial,
            trace_ref.M_rf_fc,
            trace_ref.face_trace_test_element_trial_oriented,
            d0_reference,
            d1_reference,
            source_rhs,
            boundary_trace_full.reshape(-1),
            np.float64(tau),
            np.int64(cspace.mesh.num_tri),
            np.int64(cspace.mesh.int_edges_inds.size),
            np.int64(0),
        )
    else:
        dummy_i64 = cupy.empty(1, dtype=cupy.int64)
        dummy_i32 = cupy.empty(1, dtype=cupy.int32)
        source = _kernel_source(
            _RAW_ASSEMBLY_COOP_TEMPLATE,
            nel=nel,
            ntr=ntr,
            ncols=ncols,
            matrix_format='csr',
            trace_orientation_mode=trace_orientation_mode,
            rhs_only=True,
            source_only=boundary_is_zero,
        )
        kernel = _compile_kernel(cupy, source, 'assemble_diffusion_raw_coop', assembly_shared)
        kernel_args = (
            dummy_i64,
            dummy_i64,
            indptr,
            dummy_data,
            rhs,
            cspace.mesh.loc2glob_edge,
            cspace.mesh.orientations,
            cspace.mesh.loc2oriented_face_coupling,
            side_index,
            edge_to_solve,
            cspace.mesh.int_edges_inds,
            dummy_i64,
            side_map_arg,
            mass_map_arg,
            cspace.mesh.aff_mats,
            cspace.mesh.aff_jacs,
            cspace.mesh.jacs_el_fc,
            cspace.mesh.edge_jacs,
            cspace.mesh.normals,
            cspace.quad_data.MKrf,
            cspace.quad_data.MKrf_inv,
            face_element_mass,
            trace_ref.face_element_test_trace_trial,
            trace_ref.M_rf_fc,
            trace_ref.face_trace_test_element_trial_oriented,
            d0_reference,
            d1_reference,
            source_rhs,
            boundary_trace_full.reshape(-1),
            np.float64(tau),
            np.int64(cspace.mesh.num_tri),
            np.int64(cspace.mesh.int_edges_inds.size),
            np.int64(0),
        )
    grid = (max(int(cspace.mesh.num_tri), int(cspace.mesh.int_edges_inds.size)),)
    kernel(grid, (block_size,), kernel_args, shared_mem=int(assembly_shared))
    cupy.cuda.get_current_stream().synchronize()
    timings['raw.cached_rhs_kernel'] = time.perf_counter() - start
    timings['raw.block_size'] = float(block_size)
    timings['raw.total'] = timings.get('raw.cached_rhs_zero', 0.0) + timings.get('raw.cached_rhs_kernel', 0.0)
    return RawDiffusionAssemblyResult(
        rows=None,
        cols=None,
        data=None,
        rhs=rhs,
        source_rhs=source_rhs,
        boundary_trace=boundary_trace,
        d0_reference=d0_reference,
        d1_reference=d1_reference,
        face_element_mass=face_element_mass,
        timings=timings,
        indptr=indptr,
        indices=indices,
        matrix_format='csr',
        csr_pattern=csr_pattern,
    )


def assemble_projected_diffusion_trace_system_eliminated_raw_cuda(
        *,
        source_rhs,
        boundary_trace,
        cspace,
        trace_ref,
        d0_reference,
        d1_reference,
        face_element_mass,
        tau: float,
        matrix_format: str = 'coo',
        block_size: RawCudaBlockSize = "auto",
) -> RawDiffusionAssemblyResult:
    """Assemble the reduced trace system/RHS with a Raw CUDA fused element loop."""
    cupy = require_cupy()
    validate_raw_cuda_supported(cspace, trace_ref)
    matrix_format = str(matrix_format).lower()
    if matrix_format not in {'coo', 'csr'}:
        raise ValueError("matrix_format must be 'coo' or 'csr'")
    block_size = resolve_raw_cuda_block_size(
        block_size, equation="diffusion-reaction", order=cspace.order
    )
    if block_size not in {1, 32, 64, 128}:
        raise ValueError('raw CUDA diffusion block_size must be one of 1, 32, 64, 128')
    timings: dict[str, float] = {}
    mesh_h = cspace.host.mesh
    nel = int(cspace.el_dof)
    ntr = int(cspace.edg_dof)
    ncols = 3 * ntr + 1
    assembly_shared = _shared_sizes(nel, ntr)[0] if block_size == 1 else _coop_shared_sizes(nel, ntr)
    trace_orientation_mode = _raw_trace_orientation_mode(trace_ref)

    csr_pattern = None
    indptr = indices = None
    if matrix_format == 'csr':
        from .advection_raw_cuda import build_reduced_csr_pattern_raw

        start = time.perf_counter()
        csr_pattern = build_reduced_csr_pattern_raw(cspace, timings)
        edge_to_solve = csr_pattern.edge_to_solve_edge
        side_index = csr_pattern.interior_side_index
        indptr = csr_pattern.indptr
        indices = csr_pattern.indices
        cupy.cuda.get_current_stream().synchronize()
        timings['raw.map_setup'] = timings.get('raw.csr_pattern.total', 0.0)
        timings['raw.csr_pattern.wrapper'] = time.perf_counter() - start

        zero_start = time.perf_counter()
        rows = cols = None
        data = cupy.zeros(indices.size, dtype=cupy.float64)
        rhs = cupy.zeros(mesh_h.int_edges_inds.size * ntr, dtype=cupy.float64)
        boundary_trace_full = cupy.zeros((mesh_h.num_edg, ntr), dtype=cupy.float64)
        if mesh_h.bnd_edges_inds.size:
            boundary_trace_full[cspace.mesh.bnd_edges_inds] = boundary_trace
        cupy.cuda.get_current_stream().synchronize()
        timings['raw.csr_zero'] = time.perf_counter() - zero_start
        n_flux = 0
        side_map_arg = csr_pattern.side_csr_block_pos
        mass_map_arg = csr_pattern.mass_csr_block_pos
    else:
        start = time.perf_counter()
        edge_to_solve_h = _edge_to_solve_edge(mesh_h)
        side_index_h = _interior_side_index(mesh_h)
        side_offsets_h = _side_flux_offsets(mesh_h, edge_to_solve_h, ntr)
        n_flux = int(side_offsets_h[-1])
        n_mass = int(mesh_h.int_edges_inds.size * ntr * ntr)
        nnz = n_flux + n_mass
        edge_to_solve = cupy.asarray(edge_to_solve_h, dtype=cupy.int64)
        side_index = cupy.asarray(side_index_h, dtype=cupy.int64)
        side_map_arg = cupy.asarray(side_offsets_h, dtype=cupy.int64)
        mass_map_arg = None
        cupy.cuda.get_current_stream().synchronize()
        timings['raw.map_setup'] = time.perf_counter() - start

        rows = cupy.empty(nnz, dtype=cupy.int64)
        cols = cupy.empty_like(rows)
        data = cupy.empty(nnz, dtype=cupy.float64)
        rhs = cupy.zeros(mesh_h.int_edges_inds.size * ntr, dtype=cupy.float64)
        boundary_trace_full = cupy.zeros((mesh_h.num_edg, ntr), dtype=cupy.float64)
        if mesh_h.bnd_edges_inds.size:
            boundary_trace_full[cspace.mesh.bnd_edges_inds] = boundary_trace

    start = time.perf_counter()
    if block_size == 1:
        if matrix_format == 'csr':
            source = _kernel_source(
                _raw_assembly_csr_template(),
                nel=nel,
                ntr=ntr,
                ncols=ncols,
                matrix_format=matrix_format,
                trace_orientation_mode=trace_orientation_mode,
            )
            kernel = _compile_kernel(cupy, source, 'assemble_diffusion_raw_csr', assembly_shared)
            kernel_args = (
                indptr,
                data,
                rhs,
                cspace.mesh.loc2glob_edge,
                cspace.mesh.orientations,
                cspace.mesh.loc2oriented_face_coupling,
                side_index,
                edge_to_solve,
                cspace.mesh.int_edges_inds,
                side_map_arg,
                mass_map_arg,
                cspace.mesh.aff_mats,
                cspace.mesh.aff_jacs,
                cspace.mesh.jacs_el_fc,
                cspace.mesh.edge_jacs,
                cspace.mesh.normals,
                cspace.quad_data.MKrf,
                cspace.quad_data.MKrf_inv,
                face_element_mass,
                trace_ref.face_element_test_trace_trial,
                trace_ref.M_rf_fc,
                trace_ref.face_trace_test_element_trial_oriented,
                d0_reference,
                d1_reference,
                source_rhs,
                boundary_trace_full.reshape(-1),
                np.float64(tau),
                np.int64(cspace.mesh.num_tri),
                np.int64(cspace.mesh.int_edges_inds.size),
                np.int64(n_flux),
            )
        else:
            source = _kernel_source(
                _RAW_ASSEMBLY_TEMPLATE,
                nel=nel,
                ntr=ntr,
                ncols=ncols,
                matrix_format=matrix_format,
                trace_orientation_mode=trace_orientation_mode,
            )
            kernel = _compile_kernel(cupy, source, 'assemble_diffusion_raw', assembly_shared)
            kernel_args = (
                rows,
                cols,
                data,
                rhs,
                cspace.mesh.loc2glob_edge,
                cspace.mesh.orientations,
                cspace.mesh.loc2oriented_face_coupling,
                side_index,
                edge_to_solve,
                cspace.mesh.int_edges_inds,
                side_map_arg,
                cspace.mesh.aff_mats,
                cspace.mesh.aff_jacs,
                cspace.mesh.jacs_el_fc,
                cspace.mesh.edge_jacs,
                cspace.mesh.normals,
                cspace.quad_data.MKrf,
                cspace.quad_data.MKrf_inv,
                face_element_mass,
                trace_ref.face_element_test_trace_trial,
                trace_ref.M_rf_fc,
                trace_ref.face_trace_test_element_trial_oriented,
                d0_reference,
                d1_reference,
                source_rhs,
                boundary_trace_full.reshape(-1),
                np.float64(tau),
                np.int64(cspace.mesh.num_tri),
                np.int64(cspace.mesh.int_edges_inds.size),
                np.int64(n_flux),
            )
    else:
        dummy_i64 = cupy.empty(1, dtype=cupy.int64)
        dummy_i32 = cupy.empty(1, dtype=cupy.int32)
        if matrix_format == 'csr':
            rows_arg = dummy_i64
            cols_arg = dummy_i64
            csr_indptr_arg = indptr
            side_offsets_arg = dummy_i64
            side_csr_arg = side_map_arg
            mass_csr_arg = mass_map_arg
        else:
            rows_arg = rows
            cols_arg = cols
            csr_indptr_arg = dummy_i32
            side_offsets_arg = side_map_arg
            side_csr_arg = dummy_i32
            mass_csr_arg = dummy_i32
        source = _kernel_source(
            _RAW_ASSEMBLY_COOP_TEMPLATE,
            nel=nel,
            ntr=ntr,
            ncols=ncols,
            matrix_format=matrix_format,
            trace_orientation_mode=trace_orientation_mode,
        )
        kernel = _compile_kernel(cupy, source, 'assemble_diffusion_raw_coop', assembly_shared)
        kernel_args = (
            rows_arg,
            cols_arg,
            csr_indptr_arg,
            data,
            rhs,
            cspace.mesh.loc2glob_edge,
            cspace.mesh.orientations,
            cspace.mesh.loc2oriented_face_coupling,
            side_index,
            edge_to_solve,
            cspace.mesh.int_edges_inds,
            side_offsets_arg,
            side_csr_arg,
            mass_csr_arg,
            cspace.mesh.aff_mats,
            cspace.mesh.aff_jacs,
            cspace.mesh.jacs_el_fc,
            cspace.mesh.edge_jacs,
            cspace.mesh.normals,
            cspace.quad_data.MKrf,
            cspace.quad_data.MKrf_inv,
            face_element_mass,
            trace_ref.face_element_test_trace_trial,
            trace_ref.M_rf_fc,
            trace_ref.face_trace_test_element_trial_oriented,
            d0_reference,
            d1_reference,
            source_rhs,
            boundary_trace_full.reshape(-1),
            np.float64(tau),
            np.int64(cspace.mesh.num_tri),
            np.int64(cspace.mesh.int_edges_inds.size),
            np.int64(n_flux),
        )
    grid = (max(int(cspace.mesh.num_tri), int(cspace.mesh.int_edges_inds.size)),)
    kernel(grid, (block_size,), kernel_args, shared_mem=int(assembly_shared))
    cupy.cuda.get_current_stream().synchronize()
    timings['raw.kernel' if matrix_format == 'coo' else 'raw.csr_kernel'] = time.perf_counter() - start
    timings['raw.block_size'] = float(block_size)
    timings['raw.total'] = (
        timings.get('raw.map_setup', 0.0)
        + timings.get('raw.kernel', 0.0)
        + timings.get('raw.csr_zero', 0.0)
        + timings.get('raw.csr_kernel', 0.0)
    )
    return RawDiffusionAssemblyResult(
        rows=rows,
        cols=cols,
        data=data,
        rhs=rhs,
        source_rhs=source_rhs,
        boundary_trace=boundary_trace,
        d0_reference=d0_reference,
        d1_reference=d1_reference,
        face_element_mass=face_element_mass,
        timings=timings,
        indptr=indptr,
        indices=indices,
        matrix_format=matrix_format,
        csr_pattern=csr_pattern,
    )


def reconstruct_projected_diffusion_field_raw_cuda(
        *,
        trace,
        source_rhs,
        cspace,
        trace_ref,
        d0_reference,
        d1_reference,
        face_element_mass,
        tau: float,
        block_size: RawCudaBlockSize = "auto",
        return_local_unknowns: bool = False,
):
    """Recover primal coefficients, optionally with full mixed local unknowns."""
    cupy = require_cupy()
    validate_raw_cuda_supported(cspace, trace_ref)
    block_size = resolve_raw_cuda_block_size(
        block_size, equation="diffusion-reaction", order=cspace.order
    )
    if block_size not in {1, 32, 64, 128}:
        raise ValueError('raw CUDA diffusion reconstruction block_size must be one of 1, 32, 64, 128')
    nel = int(cspace.el_dof)
    ntr = int(cspace.edg_dof)
    _, reconstruct_shared = _shared_sizes(nel, ntr)
    uh = cupy.empty((cspace.mesh.num_tri, nel), dtype=cupy.float64)
    local_unknowns = (
        cupy.empty((cspace.mesh.num_tri, 3 * nel), dtype=cupy.float64)
        if return_local_unknowns
        else cupy.empty(1, dtype=cupy.float64)
    )
    template = _RAW_RECONSTRUCT_TEMPLATE if block_size == 1 else _RAW_RECONSTRUCT_COOP_TEMPLATE
    kernel_name = 'reconstruct_diffusion_raw' if block_size == 1 else 'reconstruct_diffusion_raw_coop'
    source = _kernel_source(
        template,
        nel=nel,
        ntr=ntr,
        ncols=1,
        trace_orientation_mode=_raw_trace_orientation_mode(trace_ref),
    )
    kernel = _compile_kernel(cupy, source, kernel_name, reconstruct_shared)
    start = time.perf_counter()
    kernel(
        (int(cspace.mesh.num_tri),),
        (block_size,),
        (
            uh,
            local_unknowns,
            np.int32(1 if return_local_unknowns else 0),
            trace.reshape(-1),
            cspace.mesh.loc2glob_edge,
            cspace.mesh.loc2oriented_face_coupling,
            cspace.mesh.aff_mats,
            cspace.mesh.aff_jacs,
            cspace.mesh.jacs_el_fc,
            cspace.mesh.normals,
            cspace.quad_data.MKrf_inv,
            face_element_mass,
            trace_ref.face_trace_test_element_trial_oriented,
            d0_reference,
            d1_reference,
            source_rhs,
            np.float64(tau),
            np.int64(cspace.mesh.num_tri),
        ),
        shared_mem=int(reconstruct_shared),
    )
    cupy.cuda.get_current_stream().synchronize()
    elapsed = time.perf_counter() - start
    uh = cupy.ascontiguousarray(uh)
    if return_local_unknowns:
        return uh, cupy.ascontiguousarray(local_unknowns), elapsed
    return uh, elapsed


_RAW_PRIMAL_POSTPROCESS_TEMPLATE = r"""
__device__ __forceinline__ void reduce_post_rows_sum_raw(double* __restrict__ scratch)
{
    const int tid = threadIdx.x;
    int active = POST_ROWS;
    while (active > 1) {
        const int stride = (active + 1) >> 1;
        if (tid < active - stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
        active = stride;
    }
}

__device__ __forceinline__ void factor_primal_postprocess_lu_coop_raw(
        double* __restrict__ matrix,
        int* __restrict__ pivots)
{
    const int tid = threadIdx.x;
    for (int k = 0; k < POST_ROWS; ++k) {
        if (tid == 0) {
            int pivot = k;
            double max_value = fabs(matrix[k * POST_ROWS + k]);
            for (int i = k + 1; i < POST_ROWS; ++i) {
                const double value = fabs(matrix[i * POST_ROWS + k]);
                if (value > max_value) {
                    max_value = value;
                    pivot = i;
                }
            }
            pivots[k] = pivot;
            if (pivot != k) {
                for (int j = 0; j < POST_ROWS; ++j) {
                    const double tmp = matrix[k * POST_ROWS + j];
                    matrix[k * POST_ROWS + j] = matrix[pivot * POST_ROWS + j];
                    matrix[pivot * POST_ROWS + j] = tmp;
                }
            }
            double diagonal = matrix[k * POST_ROWS + k];
            if (fabs(diagonal) < 1.0e-30) {
                diagonal = diagonal >= 0.0 ? 1.0e-30 : -1.0e-30;
                matrix[k * POST_ROWS + k] = diagonal;
            }
            for (int i = k + 1; i < POST_ROWS; ++i) {
                matrix[i * POST_ROWS + k] /= diagonal;
            }
        }
        __syncthreads();

        const int width = POST_ROWS - k - 1;
        for (int idx = tid; idx < width * width; idx += blockDim.x) {
            const int i = k + 1 + idx / width;
            const int j = k + 1 + idx - (idx / width) * width;
            matrix[i * POST_ROWS + j] -= matrix[i * POST_ROWS + k] * matrix[k * POST_ROWS + j];
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void solve_primal_postprocess_rhs_coop_raw(
        const double* __restrict__ matrix,
        const int* __restrict__ pivots,
        double* __restrict__ rhs,
        double* __restrict__ scratch)
{
    const int tid = threadIdx.x;
    for (int k = 0; k < POST_ROWS; ++k) {
        const int pivot = pivots[k];
        if (tid == 0 && pivot != k) {
            const double tmp = rhs[k];
            rhs[k] = rhs[pivot];
            rhs[pivot] = tmp;
        }
        __syncthreads();
    }

    for (int i = 0; i < POST_ROWS; ++i) {
        double partial = 0.0;
        for (int j = tid; j < i; j += blockDim.x) {
            partial += matrix[i * POST_ROWS + j] * rhs[j];
        }
        if (tid < POST_ROWS) {
            scratch[tid] = partial;
        }
        __syncthreads();
        reduce_post_rows_sum_raw(scratch);
        if (tid == 0) {
            rhs[i] -= scratch[0];
        }
        __syncthreads();
    }

    for (int i = POST_ROWS - 1; i >= 0; --i) {
        double partial = 0.0;
        for (int j = i + 1 + tid; j < POST_ROWS; j += blockDim.x) {
            partial += matrix[i * POST_ROWS + j] * rhs[j];
        }
        if (tid < POST_ROWS) {
            scratch[tid] = partial;
        }
        __syncthreads();
        reduce_post_rows_sum_raw(scratch);
        if (tid == 0) {
            rhs[i] = (rhs[i] - scratch[0]) / matrix[i * POST_ROWS + i];
        }
        __syncthreads();
    }
}

extern "C" __global__ void primal_postprocess_diffusion_raw(
        double* __restrict__ coeffs,
        const double* __restrict__ local_unknowns,
        const double* __restrict__ aff_jacs,
        const double* __restrict__ inv_aff_mats_t,
        const double* __restrict__ weights,
        const double* __restrict__ base_basis_on_post_quads,
        const double* __restrict__ post_grad,
        const double* __restrict__ mean_base,
        const double* __restrict__ mean_post,
        const double* __restrict__ stiffness_rr,
        const double* __restrict__ stiffness_rs,
        const double* __restrict__ stiffness_ss,
        const int inverse_mode,
        const double inv00_scalar,
        const double inv01_scalar,
        const double inv10_scalar,
        const double inv11_scalar,
        const double* __restrict__ inv00_values,
        const double* __restrict__ inv01_values,
        const double* __restrict__ inv10_values,
        const double* __restrict__ inv11_values,
        const long long num_elements)
{
    extern __shared__ unsigned char shared_raw[];
    double* shared = reinterpret_cast<double*>(shared_raw);
    double* matrix = shared;
    double* rhs = matrix + (POST_ROWS * POST_ROWS);
    double* cqx = rhs + POST_ROWS;
    double* cqy = cqx + POST_NQ;
    double* scratch = cqy + POST_NQ;
    int* pivots = reinterpret_cast<int*>(scratch + POST_ROWS);

    const int tid = threadIdx.x;
    const long long element = blockIdx.x;
    if (element >= num_elements) {
        return;
    }

    const double jac = aff_jacs[element];
    const double inv00g = inv_aff_mats_t[(element * 2 + 0) * 2 + 0];
    const double inv01g = inv_aff_mats_t[(element * 2 + 0) * 2 + 1];
    const double inv10g = inv_aff_mats_t[(element * 2 + 1) * 2 + 0];
    const double inv11g = inv_aff_mats_t[(element * 2 + 1) * 2 + 1];
    const double metric_rr = inv00g * inv00g + inv10g * inv10g;
    const double metric_rs = inv00g * inv01g + inv10g * inv11g;
    const double metric_ss = inv01g * inv01g + inv11g * inv11g;

    for (int idx = tid; idx < POST_ROWS * POST_ROWS; idx += blockDim.x) {
        matrix[idx] = 0.0;
    }
    for (int i = tid; i < POST_ROWS; i += blockDim.x) {
        rhs[i] = 0.0;
    }
    __syncthreads();

    for (int idx = tid; idx < POST_NEL * POST_NEL; idx += blockDim.x) {
        const int i = idx / POST_NEL;
        const int j = idx - i * POST_NEL;
        matrix[i * POST_ROWS + j] = jac * (
            metric_rr * stiffness_rr[idx]
            + metric_rs * stiffness_rs[idx]
            + metric_ss * stiffness_ss[idx]
        );
    }
    for (int i = tid; i < POST_NEL; i += blockDim.x) {
        const double mean = jac * mean_post[i];
        matrix[i * POST_ROWS + POST_NEL] = mean;
        matrix[POST_NEL * POST_ROWS + i] = mean;
    }
    __syncthreads();

    for (int q = tid; q < POST_NQ; q += blockDim.x) {
        double qx = 0.0;
        double qy = 0.0;
        for (int j = 0; j < BASE_NEL; ++j) {
            const double basis = base_basis_on_post_quads[q * BASE_NEL + j];
            qx += local_unknowns[element * 3 * BASE_NEL + BASE_NEL + j] * basis;
            qy += local_unknowns[element * 3 * BASE_NEL + 2 * BASE_NEL + j] * basis;
        }
        if (inverse_mode == 0) {
            cqx[q] = inv00_scalar * qx + inv01_scalar * qy;
            cqy[q] = inv10_scalar * qx + inv11_scalar * qy;
        } else {
            cqx[q] = inv00_values[element * POST_NQ + q] * qx + inv01_values[element * POST_NQ + q] * qy;
            cqy[q] = inv10_values[element * POST_NQ + q] * qx + inv11_values[element * POST_NQ + q] * qy;
        }
    }
    __syncthreads();

    for (int i = tid; i < POST_NEL; i += blockDim.x) {
        double value = 0.0;
        for (int q = 0; q < POST_NQ; ++q) {
            const double grad_r = post_grad[(q * POST_NEL + i) * 2 + 0];
            const double grad_s = post_grad[(q * POST_NEL + i) * 2 + 1];
            const double grad_x = inv00g * grad_r + inv01g * grad_s;
            const double grad_y = inv10g * grad_r + inv11g * grad_s;
            value += weights[q] * (cqx[q] * grad_x + cqy[q] * grad_y);
        }
        rhs[i] = -jac * value;
    }
    if (tid == 0) {
        double mean_value = 0.0;
        for (int j = 0; j < BASE_NEL; ++j) {
            mean_value += local_unknowns[element * 3 * BASE_NEL + j] * mean_base[j];
        }
        rhs[POST_NEL] = jac * mean_value;
    }
    __syncthreads();

    factor_primal_postprocess_lu_coop_raw(matrix, pivots);
    solve_primal_postprocess_rhs_coop_raw(matrix, pivots, rhs, scratch);

    for (int i = tid; i < POST_NEL; i += blockDim.x) {
        coeffs[element * POST_NEL + i] = rhs[i];
    }
}
"""


def _raw_primal_postprocess_source(base_nel: int, post_nel: int, post_nq: int) -> str:
    """Instantiate the raw CUDA primal postprocessing kernel source."""
    return (
        _RAW_PRIMAL_POSTPROCESS_TEMPLATE
        .replace('BASE_NEL', str(int(base_nel)))
        .replace('POST_NEL', str(int(post_nel)))
        .replace('POST_ROWS', str(int(post_nel) + 1))
        .replace('POST_NQ', str(int(post_nq)))
    )


def raw_cuda_primal_postprocess_row_fit_order(block_size: int) -> int:
    """Return largest base degree whose degree ``p+1`` local rows fit in one block."""
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError('block_size must be positive')
    max_order = -1
    for order in range(64):
        post_el_dof = (order + 2) * (order + 3) // 2
        if post_el_dof + 1 > block_size:
            break
        max_order = order
    return max_order


def raw_cuda_primal_postprocess_supported_order(block_size: int) -> int:
    """Return the current raw-CUDA diffusion primal-postprocess degree limit."""
    return min(6, raw_cuda_primal_postprocess_row_fit_order(block_size))


def _select_raw_primal_postprocess_block_size(order: int, requested_block_size: int) -> int:
    """Choose a valid CUDA block size for primal postprocessing."""
    post_el_dof = (int(order) + 2) * (int(order) + 3) // 2
    post_rows = post_el_dof + 1
    allowed = (32, 64, 128)
    requested = max(32, int(requested_block_size))
    for block_size in allowed:
        if block_size >= requested and block_size >= post_rows:
            return block_size
    raise ValueError(
        f'raw CUDA primal postprocess needs at least {post_rows} threads per element for p={order}; '
        'maximum supported block size is 128'
    )


def _raw_primal_postprocess_shared_size(post_el_dof: int, post_nq: int) -> int:
    """Compute dynamic shared memory required by primal postprocessing."""
    rows = int(post_el_dof) + 1
    doubles = rows * rows + rows + 2 * int(post_nq) + rows
    return doubles * 8 + rows * 4 + 256


def _as_scalar_or_none(value) -> float | None:
    """Return a finite scalar coefficient or None for non-scalar data."""
    if np.isscalar(value):
        return float(value)
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.shape == ():
        return float(array)
    return None


def _constant_inverse_diffusion_components(diffusion) -> tuple[float, float, float, float] | None:
    """Return inverse tensor components for a constant diffusion coefficient."""
    scalar = _as_scalar_or_none(diffusion)
    if scalar is not None:
        if scalar <= 0.0:
            raise ValueError('diffusion scalar must be positive')
        inv = 1.0 / scalar
        return inv, 0.0, 0.0, inv

    try:
        array = np.asarray(diffusion, dtype=np.float64)
    except (TypeError, ValueError):
        array = None
    if array is not None and array.shape == (2, 2):
        k00, k01 = float(array[0, 0]), float(array[0, 1])
        k10, k11 = float(array[1, 0]), float(array[1, 1])
    elif isinstance(diffusion, (tuple, list)):
        if len(diffusion) == 3:
            k00 = _as_scalar_or_none(diffusion[0])
            k01 = _as_scalar_or_none(diffusion[1])
            k11 = _as_scalar_or_none(diffusion[2])
            k10 = k01
        elif len(diffusion) == 4:
            k00 = _as_scalar_or_none(diffusion[0])
            k01 = _as_scalar_or_none(diffusion[1])
            k10 = _as_scalar_or_none(diffusion[2])
            k11 = _as_scalar_or_none(diffusion[3])
        elif (
            len(diffusion) == 2
            and all(isinstance(row, (tuple, list)) and len(row) == 2 for row in diffusion)
        ):
            k00 = _as_scalar_or_none(diffusion[0][0])
            k01 = _as_scalar_or_none(diffusion[0][1])
            k10 = _as_scalar_or_none(diffusion[1][0])
            k11 = _as_scalar_or_none(diffusion[1][1])
        else:
            return None
        if None in {k00, k01, k10, k11}:
            return None
    else:
        return None

    det = k00 * k11 - k01 * k10
    if det <= 0.0:
        raise ValueError(f'diffusion tensor must be positive definite; determinant is {det}')
    return k11 / det, -k01 / det, -k10 / det, k00 / det


def postprocess_projected_diffusion_primal_raw_cuda(
        local_unknowns,
        cspace,
        diffusion=1.0,
        *,
        trace_space=None,
        block_size: RawCudaBlockSize = "auto",
        name: str = 'u_h_star',
        timings: dict[str, float] | None = None,
        cache=None,
):
    """Recover the HDG primal postprocessed field with a per-element Raw CUDA kernel.

    The current raw diffusion path is validated for base degree ``p <= 6``.  At
    ``p=6`` the degree-7 postprocess solve has 36 scalar coefficients plus one
    mean row, so block sizes 64 and 128 provide at least one thread per local
    row.  A 128-thread block would fit that row-wise local work through
    ``p=13``, but the assembled raw diffusion solve remains capped at ``p<=6``.
    """
    cupy = require_cupy()
    if int(cspace.el_dof) > 28:
        raise ValueError('raw CUDA primal postprocess currently supports p <= 6 (el_dof <= 28)')
    timings = {} if timings is None else timings
    stream = cupy.cuda.get_current_stream()
    stream.synchronize()
    total_start = time.perf_counter()

    setup_start = time.perf_counter()
    from ..solvers.diffusion_reaction import _new_hdg_postprocess_cache

    trace_ref = cspace.host.trace_space('legacy-lagrange') if trace_space is None else trace_space
    if cache is None or cache.base_space is not cspace.host or cache.trace_space is not trace_ref:
        cache = _new_hdg_postprocess_cache(cspace.host, trace_ref)
    cpost_space = as_cupy_space(cache.post_space, device=cspace.device_id)
    local_unknowns = cupy.ascontiguousarray(cupy.asarray(local_unknowns, dtype=cupy.float64))
    expected_unknowns = (cspace.mesh.num_tri, 3 * cspace.el_dof)
    if tuple(local_unknowns.shape) != expected_unknowns:
        raise ValueError(f'local_unknowns must have shape {expected_unknowns}; got {local_unknowns.shape}')

    base_el_dof = int(cspace.el_dof)
    post_el_dof = int(cache.post_space.el_dof)
    post_nq = int(cache.post_space.quad_data.Krf_w.shape[0])
    block_size = resolve_raw_cuda_block_size(
        block_size, equation="diffusion-reaction", order=cspace.order
    )
    effective_block_size = _select_raw_primal_postprocess_block_size(cspace.order, block_size)
    shared_bytes = _raw_primal_postprocess_shared_size(post_el_dof, post_nq)

    weights = cupy.asarray(cache.post_space.quad_data.Krf_w, dtype=cupy.float64)
    base_basis = cupy.asarray(cache.base_basis_on_post_quads, dtype=cupy.float64)
    post_grad = cupy.asarray(cache.post_space.quad_data.gphi, dtype=cupy.float64)
    mean_base = cupy.asarray(cache.mean_base, dtype=cupy.float64)
    mean_post = cupy.asarray(cache.mean_post, dtype=cupy.float64)
    stiffness_rr = cupy.asarray(cache.primal_stiffness_rr, dtype=cupy.float64)
    stiffness_rs = cupy.asarray(cache.primal_stiffness_rs, dtype=cupy.float64)
    stiffness_ss = cupy.asarray(cache.primal_stiffness_ss, dtype=cupy.float64)
    inverse_constants = _constant_inverse_diffusion_components(diffusion)
    if inverse_constants is None:
        from ..solvers.diffusion_reaction import _inverse_diffusion_values

        inv00_h, inv01_h, inv10_h, inv11_h = _inverse_diffusion_values(diffusion, cache.post_space)
        inverse_mode = np.int32(1)
        inv00_scalar = inv01_scalar = inv10_scalar = inv11_scalar = np.float64(0.0)
        inv00 = cupy.asarray(inv00_h, dtype=cupy.float64)
        inv01 = cupy.asarray(inv01_h, dtype=cupy.float64)
        inv10 = cupy.asarray(inv10_h, dtype=cupy.float64)
        inv11 = cupy.asarray(inv11_h, dtype=cupy.float64)
    else:
        inverse_mode = np.int32(0)
        inv00_scalar, inv01_scalar, inv10_scalar, inv11_scalar = map(np.float64, inverse_constants)
        inv00 = inv01 = inv10 = inv11 = weights
    coeffs = cupy.empty((cspace.mesh.num_tri, post_el_dof), dtype=cupy.float64)
    stream.synchronize()
    timings['postprocess.primal.raw_cuda.setup'] = time.perf_counter() - setup_start

    kernel_source = _raw_primal_postprocess_source(base_el_dof, post_el_dof, post_nq)
    kernel = _compile_kernel(cupy, kernel_source, 'primal_postprocess_diffusion_raw', shared_bytes)
    solve_start = time.perf_counter()
    kernel(
        (int(cspace.mesh.num_tri),),
        (int(effective_block_size),),
        (
            coeffs,
            local_unknowns,
            cspace.mesh.aff_jacs,
            cspace.mesh.inv_aff_mats_t,
            weights,
            base_basis,
            post_grad,
            mean_base,
            mean_post,
            stiffness_rr,
            stiffness_rs,
            stiffness_ss,
            inverse_mode,
            inv00_scalar,
            inv01_scalar,
            inv10_scalar,
            inv11_scalar,
            inv00,
            inv01,
            inv10,
            inv11,
            np.int64(cspace.mesh.num_tri),
        ),
        shared_mem=int(shared_bytes),
    )
    stream.synchronize()
    timings['postprocess.primal.raw_cuda.kernel'] = time.perf_counter() - solve_start
    timings['postprocess.primal.raw_cuda.total'] = time.perf_counter() - total_start
    timings['postprocess.primal.raw_cuda.block_size'] = float(effective_block_size)
    timings['postprocess.primal.raw_cuda.shared_bytes'] = float(shared_bytes)
    timings['postprocess.primal.raw_cuda.row_fit_order'] = float(
        raw_cuda_primal_postprocess_row_fit_order(effective_block_size)
    )
    timings['postprocess.primal.raw_cuda.supported_order'] = float(
        raw_cuda_primal_postprocess_supported_order(effective_block_size)
    )
    return cpost_space.field(cupy.ascontiguousarray(coeffs), name=name), cache


__all__ = [
    'RawDiffusionAssemblyResult',
    'assemble_projected_diffusion_trace_rhs_eliminated_raw_cuda',
    'assemble_projected_diffusion_trace_system_eliminated_raw_cuda',
    'postprocess_projected_diffusion_primal_raw_cuda',
    'raw_cuda_primal_postprocess_row_fit_order',
    'raw_cuda_primal_postprocess_supported_order',
    'reconstruct_projected_diffusion_field_raw_cuda',
    'validate_raw_cuda_supported',
]
