
"""Experimental Raw CUDA kernels for diffusion-reaction HDG assembly.

The first implementation targets the standalone ``scripts/run_diff_rea_gpu4_hdg.py``
benchmark path: identity diffusion, scalar zero reaction, nodal legacy-lagrange
trace coordinates, and p <= 6.  It keeps the existing CuPy source and boundary
trace evaluation, then fuses the expensive element-local HDG condensation,
boundary-column elimination, and COO/RHS emission into Raw CUDA kernels.

The CUDA kernels are intentionally written as a readable baseline: one CUDA
thread owns one element and performs the same small dense algebra as the fused
Numba kernel.  The assembly kernel factors the local scalar Schur matrix once,
solves trace/source columns one at a time to stay within p=6 shared-memory limits,
and emits the reduced COO/RHS directly.  This is correct and useful for launch
fusion experiments, but the next performance pass should make one CUDA block
cooperate on one element.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .cupy import require_cupy


@dataclass(frozen=True)
class RawDiffusionAssemblyResult:
    """Device-side reduced trace system assembled by the Raw CUDA path."""

    rows: Any
    cols: Any
    data: Any
    rhs: Any
    source_rhs: Any
    boundary_trace: Any
    d0_reference: Any
    d1_reference: Any
    face_element_mass: Any
    timings: dict[str, float]


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
                const int local_col_dof = positive ? col_dof : (NTR - 1 - col_dof);
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
                        const double schur_value = trace_flux_value_raw(
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

_RAW_RECONSTRUCT_TEMPLATE = r"""
extern "C" __global__ void reconstruct_diffusion_raw(
        double* __restrict__ uh,
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
    for (int i = 0; i < NEL; ++i) {
        uh[element * NEL + i] = red_rhs[i];
    }
}
"""


def _kernel_source(template: str, *, nel: int, ntr: int, ncols: int) -> str:
    return template.replace('NEL', str(int(nel))).replace('NTR', str(int(ntr))).replace('NCOLS', str(int(ncols)))


def _shared_sizes(nel: int, ntr: int) -> tuple[int, int]:
    # Assembly v2 stores one RHS/solution column at a time.  The previous
    # all-columns layout exceeded the 64 KiB opt-in shared-memory limit on the
    # Quadro RTX 6000 at p=6; this formula intentionally mirrors the CUDA shared
    # layout in assemble_diffusion_raw so unsupported degrees fail predictably.
    assembly_doubles = 7 * nel * nel + 6 * nel
    assembly_bytes = assembly_doubles * 8 + nel * 4 + 256
    reconstruct_doubles = 7 * nel * nel + 6 * nel
    reconstruct_bytes = reconstruct_doubles * 8 + nel * 4 + 256
    return assembly_bytes, reconstruct_bytes


def _compile_kernel(cupy, source: str, name: str, shared_bytes: int):
    kernel = cupy.RawKernel(source, name, options=('--std=c++11',))
    try:
        kernel.max_dynamic_shared_size_bytes = int(shared_bytes)
    except Exception:
        pass
    return kernel


def _edge_to_solve_edge(mesh) -> np.ndarray:
    edge_is_free = np.ones(mesh.num_edg, dtype=bool)
    edge_is_free[mesh.bnd_edges_inds] = False
    free_edges = np.flatnonzero(edge_is_free).astype(np.int64)
    edge_to_solve = np.full(mesh.num_edg, -1, dtype=np.int64)
    edge_to_solve[free_edges] = np.arange(free_edges.size, dtype=np.int64)
    return np.ascontiguousarray(edge_to_solve)


def _interior_side_index(mesh) -> np.ndarray:
    index = np.full((mesh.num_tri, 3), -1, dtype=np.int64)
    index[mesh.interior_elements, mesh.interior_faces] = np.arange(mesh.interior_elements.size, dtype=np.int64)
    return np.ascontiguousarray(index)


def _side_flux_offsets(mesh, edge_to_solve_edge: np.ndarray, edg_dof: int) -> np.ndarray:
    face_is_free = edge_to_solve_edge[mesh.loc2glob_edge] >= 0
    side_col_counts = np.count_nonzero(face_is_free[mesh.interior_elements], axis=1).astype(np.int64)
    offsets = np.empty(side_col_counts.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(side_col_counts * edg_dof * edg_dof, out=offsets[1:])
    return np.ascontiguousarray(offsets)


def validate_raw_cuda_supported(cspace, trace_ref) -> None:
    if not getattr(trace_ref, 'nodal', False) or getattr(trace_ref, 'kind', '') != 'legacy-lagrange':
        raise ValueError('raw CUDA diffusion assembly currently supports only legacy-lagrange nodal trace basis')
    if cspace.el_dof > 28:
        raise ValueError('raw CUDA diffusion assembly currently supports p <= 6 (el_dof <= 28)')


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
) -> RawDiffusionAssemblyResult:
    """Assemble the reduced trace COO/RHS with a Raw CUDA fused element loop."""
    cupy = require_cupy()
    validate_raw_cuda_supported(cspace, trace_ref)
    timings: dict[str, float] = {}
    mesh_h = cspace.host.mesh
    nel = int(cspace.el_dof)
    ntr = int(cspace.edg_dof)
    ncols = 3 * ntr + 1
    assembly_shared, _ = _shared_sizes(nel, ntr)

    start = time.perf_counter()
    edge_to_solve_h = _edge_to_solve_edge(mesh_h)
    side_index_h = _interior_side_index(mesh_h)
    side_offsets_h = _side_flux_offsets(mesh_h, edge_to_solve_h, ntr)
    n_flux = int(side_offsets_h[-1])
    n_mass = int(mesh_h.int_edges_inds.size * ntr * ntr)
    nnz = n_flux + n_mass
    edge_to_solve = cupy.asarray(edge_to_solve_h, dtype=cupy.int64)
    side_index = cupy.asarray(side_index_h, dtype=cupy.int64)
    side_offsets = cupy.asarray(side_offsets_h, dtype=cupy.int64)
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
    source = _kernel_source(_RAW_ASSEMBLY_TEMPLATE, nel=nel, ntr=ntr, ncols=ncols)
    kernel = _compile_kernel(cupy, source, 'assemble_diffusion_raw', assembly_shared)
    grid = (max(int(cspace.mesh.num_tri), int(cspace.mesh.int_edges_inds.size)),)
    kernel(
        grid,
        (1,),
        (
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
            side_offsets,
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
        ),
        shared_mem=int(assembly_shared),
    )
    cupy.cuda.get_current_stream().synchronize()
    timings['raw.kernel'] = time.perf_counter() - start
    timings['raw.total'] = sum(timings.values())
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
):
    """Recover primal element coefficients with the Raw CUDA local solve."""
    cupy = require_cupy()
    validate_raw_cuda_supported(cspace, trace_ref)
    nel = int(cspace.el_dof)
    ntr = int(cspace.edg_dof)
    _, reconstruct_shared = _shared_sizes(nel, ntr)
    uh = cupy.empty((cspace.mesh.num_tri, nel), dtype=cupy.float64)
    source = _kernel_source(_RAW_RECONSTRUCT_TEMPLATE, nel=nel, ntr=ntr, ncols=1)
    kernel = _compile_kernel(cupy, source, 'reconstruct_diffusion_raw', reconstruct_shared)
    start = time.perf_counter()
    kernel(
        (int(cspace.mesh.num_tri),),
        (1,),
        (
            uh,
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
    return cupy.ascontiguousarray(uh), time.perf_counter() - start


__all__ = [
    'RawDiffusionAssemblyResult',
    'assemble_projected_diffusion_trace_system_eliminated_raw_cuda',
    'reconstruct_projected_diffusion_field_raw_cuda',
    'validate_raw_cuda_supported',
]
