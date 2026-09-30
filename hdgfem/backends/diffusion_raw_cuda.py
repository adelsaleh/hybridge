
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

from hdgfem.runtime.precision import REAL_DTYPE, REAL_ITEMSIZE

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from hdgfem.core.device import as_cupy_space
from hdgfem.runtime.optional import require_cupy
from hdgfem.hdg.cuda.launch import RawCudaBlockSize, resolve_raw_cuda_block_size
from hdgfem.hdg.trace_maps import (
    _edge_to_solve_edge,
    _interior_side_index,
    _side_flux_offsets,
)
from hdgfem.hdg.cuda.launch import _compile_kernel, _compile_kernel_timed


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
    schur_lu: Any | None = None
    schur_pivots: Any | None = None
    local_factor_key: tuple[Any, ...] | None = None
    local_factor_bytes: int = 0
    local_factor_kind: str = "none"


def _allocate_local_schur_factors(cupy, *, num_elements: int, nel: int, factor_kind: str):
    """Allocate persistent per-element Schur factors with a clear OOM error."""
    if factor_kind != "schur-lu":
        raise ValueError("raw CUDA local factors support only 'schur-lu'")
    lu_bytes = int(num_elements) * int(nel) * int(nel) * np.dtype(REAL_DTYPE).itemsize
    pivot_bytes = int(num_elements) * int(nel) * np.dtype(np.int32).itemsize
    required = lu_bytes + pivot_bytes
    free, _total = cupy.cuda.runtime.memGetInfo()
    if int(free) < required:
        raise MemoryError(
            "raw-CUDA local Schur-LU cache requires "
            f"{required} bytes ({required / (1024 ** 3):.3f} GiB), but only "
            f"{int(free)} bytes ({int(free) / (1024 ** 3):.3f} GiB) are available"
        )
    try:
        schur_lu = cupy.empty((num_elements, nel, nel), dtype=REAL_DTYPE)
        schur_pivots = cupy.empty((num_elements, nel), dtype=cupy.int32)
    except Exception as exc:
        out_of_memory = getattr(cupy.cuda.memory, "OutOfMemoryError", ())
        if out_of_memory and isinstance(exc, out_of_memory):
            raise MemoryError(
                "raw-CUDA local Schur-LU cache allocation failed for "
                f"{required} bytes ({required / (1024 ** 3):.3f} GiB); "
                f"the device reported {int(free)} bytes free before allocation"
            ) from exc
        raise
    return schur_lu, schur_pivots, required


def _validate_local_schur_factors(cached_raw, *, num_elements: int, nel: int, local_factor_key=None):
    """Return cached factors after validating ownership metadata and shape."""
    schur_lu = cached_raw.schur_lu
    schur_pivots = cached_raw.schur_pivots
    if schur_lu is None or schur_pivots is None:
        raise ValueError("raw-CUDA cached Schur factors were requested but are unavailable")
    cached_kind = str(getattr(cached_raw, "local_factor_kind", "schur-lu"))
    expected_lu = (int(num_elements), int(nel), int(nel))
    if cached_kind != "schur-lu":
        raise ValueError("raw CUDA cached local factors must use 'schur-lu'")
    expected_pivots = (int(num_elements), int(nel))
    if tuple(schur_lu.shape) != expected_lu or tuple(schur_pivots.shape) != expected_pivots:
        raise ValueError(
            "raw-CUDA cached Schur factor shape mismatch: "
            f"expected {expected_lu}/{expected_pivots}, got "
            f"{tuple(schur_lu.shape)}/{tuple(schur_pivots.shape)}"
        )
    if local_factor_key is not None and cached_raw.local_factor_key != local_factor_key:
        raise ValueError("raw-CUDA cached Schur factor key does not match the current operator")
    return schur_lu, schur_pivots


from hdgfem.hdg.cuda.raw_source import (
    RAW_TRACE_ORIENTATION_HELPERS as _RAW_TRACE_ORIENTATION_HELPERS,
    RAW_COOPERATIVE_SOLVES,
)

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
        double* __restrict__ schur_lu_cache,
        int* __restrict__ schur_pivot_cache,
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
#if RAW_USE_CACHED_FACTORS
                schur_matrix[ij] = schur_lu_cache[element * NEL * NEL + ij];
#else
                schur_matrix[ij] = tau_value;
#endif
            }
        }

#if !RAW_USE_CACHED_FACTORS
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

#endif

#if !RAW_USE_CACHED_FACTORS
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

#if RAW_WRITE_CACHED_FACTORS
        for (int idx = 0; idx < NEL * NEL; ++idx) {
            schur_lu_cache[element * NEL * NEL + idx] = schur_matrix[idx];
        }
        for (int k = 0; k < NEL; ++k) {
            schur_pivot_cache[element * NEL + k] = pivots[k];
        }
#endif
#else
        for (int k = 0; k < NEL; ++k) {
            pivots[k] = schur_pivot_cache[element * NEL + k];
        }
#endif

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
#ifndef RAW_MATRIX_BSR
#define RAW_MATRIX_BSR 0
#endif
#ifndef RAW_RHS_ONLY
#define RAW_RHS_ONLY 0
#endif
#ifndef RAW_SOURCE_ONLY
#define RAW_SOURCE_ONLY 0
#endif
#ifndef RAW_BATCHED_FULL
#define RAW_BATCHED_FULL 0
#endif
#ifndef RAW_BATCH_COLS
#define RAW_BATCH_COLS 8
#endif

""" + RAW_COOPERATIVE_SOLVES + r"""extern "C" __global__ void assemble_diffusion_raw_coop(
        long long* __restrict__ rows,
        long long* __restrict__ cols,
        const int* __restrict__ csr_indptr,
        double* __restrict__ data,
        double* __restrict__ rhs,
        double* __restrict__ schur_lu_cache,
        int* __restrict__ schur_pivot_cache,
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
        const double* __restrict__ mass_inverse_d0_reference,
        const double* __restrict__ mass_inverse_d1_reference,
        const double* __restrict__ mass_inverse_face_element_trace,
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
#if RAW_SOURCE_ONLY
    // The source-only path builds x/y contributions sequentially and reuses
    // these three matrix workspaces.  This keeps p=6 below half an SM's shared
    // memory, allowing two resident blocks on a 64 KiB/SM device.
    double* mn0 = d0 + (NEL * NEL);
    double* k_d0 = mn0 + (NEL * NEL);
    double* local_rhs = k_d0 + (NEL * NEL);
#elif RAW_BATCHED_FULL
    // Retain both normal-minus-derivative matrices, but reuse the fourth
    // matrix workspace for Schur construction and flux recovery.  Only a
    // small batch of condensed columns is live at once.
    double* mn0 = d0;
    double* mn1 = mn0 + (NEL * NEL);
    double* k_d0 = mn1 + (NEL * NEL);
    double* local_rhs = k_d0 + (NEL * NEL);
#else
    double* d1 = d0 + (NEL * NEL);
    double* mn0 = d1 + (NEL * NEL);
    double* mn1 = mn0 + (NEL * NEL);
    double* k_d0 = mn1 + (NEL * NEL);
    double* k_d1 = k_d0 + (NEL * NEL);
    double* local_rhs = k_d1 + (NEL * NEL);
#endif
#if RAW_BATCHED_FULL
    double* tmp_cols = local_rhs + (NEL * RAW_BATCH_COLS);
    double* row_values = tmp_cols + (NEL * RAW_BATCH_COLS);
    int* pivots = reinterpret_cast<int*>(row_values + (3 * NTR * RAW_BATCH_COLS));
#else
    double* qx_cols = local_rhs + (NEL * NCOLS);
    double* qy_cols = qx_cols + (NEL * NCOLS);
    double* tmp_cols = qy_cols + (NEL * NCOLS);
    int* pivots = reinterpret_cast<int*>(tmp_cols + (NEL * NCOLS));
#endif

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

#if RAW_SOURCE_ONLY
        // Build the two directional Schur contributions sequentially through
        // one derivative, normal-minus-derivative, and M^{-1}D workspace.
        for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
            const int i = idx / NEL;
            const int j = idx - i * NEL;
            const double derivative = aff11 * d0_reference[idx] - aff10 * d1_reference[idx];
            double normal_value = 0.0;
            double tau_value = 0.0;
            for (int face = 0; face < 3; ++face) {
                const double face_mass_value = face_element_mass[(face * NEL + i) * NEL + j];
                const double face_scale = jacs_el_fc[element * 3 + face];
                tau_value += tau * face_scale * face_mass_value;
                normal_value += face_scale * normals[(element * 3 + face) * 2 + 0] * face_mass_value;
            }
            d0[idx] = derivative;
            mn0[idx] = normal_value - derivative;
#if RAW_USE_CACHED_FACTORS
            schur_matrix[idx] = schur_lu_cache[element * NEL * NEL + idx];
#else
            schur_matrix[idx] = tau_value;
#endif
        }
        __syncthreads();
#if !RAW_USE_CACHED_FACTORS
        for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
            const int i = idx / NEL;
            const int j = idx - i * NEL;
            double value = 0.0;
            for (int k = 0; k < NEL; ++k) {
                value += mass_inverse[i * NEL + k] * d0[k * NEL + j];
            }
            k_d0[idx] = value;
        }
        __syncthreads();
        for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
            const int i = idx / NEL;
            const int j = idx - i * NEL;
            double value = 0.0;
            for (int k = 0; k < NEL; ++k) {
                value += mn0[i * NEL + k] * k_d0[k * NEL + j];
            }
            schur_matrix[idx] += jac_inv * value;
        }
        __syncthreads();

        for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
            const int i = idx / NEL;
            const int j = idx - i * NEL;
            const double derivative = -aff01 * d0_reference[idx] + aff00 * d1_reference[idx];
            double normal_value = 0.0;
            for (int face = 0; face < 3; ++face) {
                const double face_mass_value = face_element_mass[(face * NEL + i) * NEL + j];
                const double face_scale = jacs_el_fc[element * 3 + face];
                normal_value += face_scale * normals[(element * 3 + face) * 2 + 1] * face_mass_value;
            }
            d0[idx] = derivative;
            mn0[idx] = normal_value - derivative;
        }
        __syncthreads();
        for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
            const int i = idx / NEL;
            const int j = idx - i * NEL;
            double value = 0.0;
            for (int k = 0; k < NEL; ++k) {
                value += mass_inverse[i * NEL + k] * d0[k * NEL + j];
            }
            k_d0[idx] = value;
        }
        __syncthreads();
        for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
            const int i = idx / NEL;
            const int j = idx - i * NEL;
            double value = 0.0;
            for (int k = 0; k < NEL; ++k) {
                value += mn0[i * NEL + k] * k_d0[k * NEL + j];
            }
            schur_matrix[idx] += jac_inv * value;
        }
        __syncthreads();
#endif
#elif RAW_BATCHED_FULL
        // Build the factor once while retaining only the two matrices needed
        // to condense subsequent batches of trace/source columns.
        for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
            const int i = idx / NEL;
            const int j = idx - i * NEL;
            const double derivative_x = aff11 * d0_reference[idx] - aff10 * d1_reference[idx];
            const double derivative_y = -aff01 * d0_reference[idx] + aff00 * d1_reference[idx];
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
            mn0[idx] = normal_x_value - derivative_x;
            mn1[idx] = normal_y_value - derivative_y;
#if RAW_USE_CACHED_FACTORS
            schur_matrix[idx] = schur_lu_cache[element * NEL * NEL + idx];
#else
            schur_matrix[idx] = tau_value;
#endif
        }
        __syncthreads();
#if !RAW_USE_CACHED_FACTORS
        for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
            const int i = idx / NEL;
            const int j = idx - i * NEL;
            k_d0[idx] = aff11 * mass_inverse_d0_reference[idx]
                      - aff10 * mass_inverse_d1_reference[idx];
        }
        __syncthreads();
        for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
            const int i = idx / NEL;
            const int j = idx - i * NEL;
            double value = 0.0;
            for (int k = 0; k < NEL; ++k) {
                value += mn0[i * NEL + k] * k_d0[k * NEL + j];
            }
            schur_matrix[idx] += jac_inv * value;
        }
        __syncthreads();
        for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
            const int i = idx / NEL;
            const int j = idx - i * NEL;
            k_d0[idx] = -aff01 * mass_inverse_d0_reference[idx]
                       + aff00 * mass_inverse_d1_reference[idx];
        }
        __syncthreads();
        for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
            const int i = idx / NEL;
            const int j = idx - i * NEL;
            double value = 0.0;
            for (int k = 0; k < NEL; ++k) {
                value += mn1[i * NEL + k] * k_d0[k * NEL + j];
            }
            schur_matrix[idx] += jac_inv * value;
        }
        __syncthreads();
#endif
#else
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
#if RAW_USE_CACHED_FACTORS
            schur_matrix[idx] = schur_lu_cache[element * NEL * NEL + idx];
#else
            schur_matrix[idx] = tau_value;
#endif
        }
        __syncthreads();
#if !RAW_USE_CACHED_FACTORS

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

#endif

#endif

#if RAW_SOURCE_ONLY
        for (int i = tid; i < NEL; i += blockDim.x) {
            local_rhs[i * NCOLS + source_col] = source_rhs[element * 3 * NEL + i];
        }
        __syncthreads();
#elif RAW_BATCHED_FULL
        // Column data is prepared below one batch at a time.
#else
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
#endif

#if RAW_USE_CACHED_FACTORS
        for (int k = tid; k < NEL; k += blockDim.x) {
            pivots[k] = schur_pivot_cache[element * NEL + k];
        }
        __syncthreads();
#else
        factor_diffusion_schur_lu_coop_raw(schur_matrix, pivots);
#if RAW_WRITE_CACHED_FACTORS
        for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
            schur_lu_cache[element * NEL * NEL + idx] = schur_matrix[idx];
        }
        for (int k = tid; k < NEL; k += blockDim.x) {
            schur_pivot_cache[element * NEL + k] = pivots[k];
        }
        __syncthreads();
#endif
#endif
#if RAW_BATCHED_FULL
        for (int batch_start = 0; batch_start < NCOLS; batch_start += RAW_BATCH_COLS) {
            const int batch_cols = min(RAW_BATCH_COLS, NCOLS - batch_start);

            // Form the scalar local right-hand sides for this batch.
            for (int idx = tid; idx < NEL * RAW_BATCH_COLS; idx += blockDim.x) {
                const int i = idx / RAW_BATCH_COLS;
                const int batch_col = idx - i * RAW_BATCH_COLS;
                if (batch_col < batch_cols) {
                    const int col = batch_start + batch_col;
                    double value = 0.0;
                    if (col == source_col) {
                        value = source_rhs[element * 3 * NEL + i];
                    } else {
                        const int col_face = col / NTR;
                        const int local_trace_dof = col - col_face * NTR;
                        const double face_scale = jacs_el_fc[element * 3 + col_face];
                        const double coupling = face_scale
                            * face_element_trace[(col_face * NEL + i) * NTR + local_trace_dof];
                        value = tau * coupling;
                    }
                    local_rhs[idx] = value;
                }
            }
            __syncthreads();

            // Add (N_x-D_x) M^{-1} qhat_x to the scalar equations.
            for (int idx = tid; idx < NEL * RAW_BATCH_COLS; idx += blockDim.x) {
                const int i = idx / RAW_BATCH_COLS;
                const int batch_col = idx - i * RAW_BATCH_COLS;
                if (batch_col < batch_cols) {
                    const int col = batch_start + batch_col;
                    double value = 0.0;
                    if (col != source_col) {
                        const int col_face = col / NTR;
                        const int local_trace_dof = col - col_face * NTR;
                        const double face_scale = jacs_el_fc[element * 3 + col_face];
                        const double normal = normals[(element * 3 + col_face) * 2 + 0];
                        value = face_scale * normal
                            * mass_inverse_face_element_trace[
                                (col_face * NEL + i) * NTR + local_trace_dof];
                    }
                    tmp_cols[idx] = value;
                }
            }
            __syncthreads();
            for (int idx = tid; idx < NEL * RAW_BATCH_COLS; idx += blockDim.x) {
                const int i = idx / RAW_BATCH_COLS;
                const int batch_col = idx - i * RAW_BATCH_COLS;
                if (batch_col < batch_cols) {
                    double value = 0.0;
                    for (int k = 0; k < NEL; ++k) {
                        value += mn0[i * NEL + k] * tmp_cols[k * RAW_BATCH_COLS + batch_col];
                    }
                    local_rhs[idx] += jac_inv * value;
                }
            }
            __syncthreads();

            // Add (N_y-D_y) M^{-1} qhat_y.
            for (int idx = tid; idx < NEL * RAW_BATCH_COLS; idx += blockDim.x) {
                const int i = idx / RAW_BATCH_COLS;
                const int batch_col = idx - i * RAW_BATCH_COLS;
                if (batch_col < batch_cols) {
                    const int col = batch_start + batch_col;
                    double value = 0.0;
                    if (col != source_col) {
                        const int col_face = col / NTR;
                        const int local_trace_dof = col - col_face * NTR;
                        const double face_scale = jacs_el_fc[element * 3 + col_face];
                        const double normal = normals[(element * 3 + col_face) * 2 + 1];
                        value = face_scale * normal
                            * mass_inverse_face_element_trace[
                                (col_face * NEL + i) * NTR + local_trace_dof];
                    }
                    tmp_cols[idx] = value;
                }
            }
            __syncthreads();
            for (int idx = tid; idx < NEL * RAW_BATCH_COLS; idx += blockDim.x) {
                const int i = idx / RAW_BATCH_COLS;
                const int batch_col = idx - i * RAW_BATCH_COLS;
                if (batch_col < batch_cols) {
                    double value = 0.0;
                    for (int k = 0; k < NEL; ++k) {
                        value += mn1[i * NEL + k] * tmp_cols[k * RAW_BATCH_COLS + batch_col];
                    }
                    local_rhs[idx] += jac_inv * value;
                }
            }
            __syncthreads();

            solve_diffusion_column_batch_coop_raw(
                schur_matrix, pivots, local_rhs, batch_cols);

            // Start each numerical-flux row with tau * <test, u_h>.
            for (int idx = tid; idx < 3 * NTR * RAW_BATCH_COLS; idx += blockDim.x) {
                const int task = idx / RAW_BATCH_COLS;
                const int batch_col = idx - task * RAW_BATCH_COLS;
                if (batch_col < batch_cols) {
                    const int row_face = task / NTR;
                    const int row_dof = task - row_face * NTR;
                    const long long oriented_face = loc2oriented_face_coupling[element * 3 + row_face];
                    const double scale = jacs_el_fc[element * 3 + row_face];
                    double value = 0.0;
                    for (int i = 0; i < NEL; ++i) {
                        const double lift = scale
                            * oriented_lifts[(oriented_face * NTR + row_dof) * NEL + i];
                        value += tau * lift * local_rhs[i * RAW_BATCH_COLS + batch_col];
                    }
                    row_values[idx] = value;
                }
            }
            __syncthreads();

            // Recover q_x into the reusable matrix workspace.
            for (int idx = tid; idx < NEL * RAW_BATCH_COLS; idx += blockDim.x) {
                const int i = idx / RAW_BATCH_COLS;
                const int batch_col = idx - i * RAW_BATCH_COLS;
                if (batch_col < batch_cols) {
                    const int col = batch_start + batch_col;
                    double value = 0.0;
                    for (int j = 0; j < NEL; ++j) {
                        const int derivative_idx = i * NEL + j;
                        const double derivative = aff11 * d0_reference[derivative_idx]
                                                - aff10 * d1_reference[derivative_idx];
                        value += derivative * local_rhs[j * RAW_BATCH_COLS + batch_col];
                    }
                    if (col != source_col) {
                        const int col_face = col / NTR;
                        const int local_trace_dof = col - col_face * NTR;
                        const double face_scale = jacs_el_fc[element * 3 + col_face];
                        const double coupling = face_scale
                            * face_element_trace[(col_face * NEL + i) * NTR + local_trace_dof];
                        value -= normals[(element * 3 + col_face) * 2 + 0] * coupling;
                    }
                    tmp_cols[idx] = value;
                }
            }
            __syncthreads();
            for (int idx = tid; idx < NEL * RAW_BATCH_COLS; idx += blockDim.x) {
                const int i = idx / RAW_BATCH_COLS;
                const int batch_col = idx - i * RAW_BATCH_COLS;
                if (batch_col < batch_cols) {
                    double value = 0.0;
                    for (int k = 0; k < NEL; ++k) {
                        value += mass_inverse[i * NEL + k]
                               * tmp_cols[k * RAW_BATCH_COLS + batch_col];
                    }
                    k_d0[idx] = jac_inv * value;
                }
            }
            __syncthreads();
            for (int idx = tid; idx < 3 * NTR * RAW_BATCH_COLS; idx += blockDim.x) {
                const int task = idx / RAW_BATCH_COLS;
                const int batch_col = idx - task * RAW_BATCH_COLS;
                if (batch_col < batch_cols) {
                    const int row_face = task / NTR;
                    const int row_dof = task - row_face * NTR;
                    const long long oriented_face = loc2oriented_face_coupling[element * 3 + row_face];
                    const double scale = jacs_el_fc[element * 3 + row_face];
                    const double nx = normals[(element * 3 + row_face) * 2 + 0];
                    double value = 0.0;
                    for (int i = 0; i < NEL; ++i) {
                        const double lift = scale
                            * oriented_lifts[(oriented_face * NTR + row_dof) * NEL + i];
                        value += nx * lift * k_d0[i * RAW_BATCH_COLS + batch_col];
                    }
                    row_values[idx] += value;
                }
            }
            __syncthreads();

            // Recover q_y and complete the numerical-flux rows.
            for (int idx = tid; idx < NEL * RAW_BATCH_COLS; idx += blockDim.x) {
                const int i = idx / RAW_BATCH_COLS;
                const int batch_col = idx - i * RAW_BATCH_COLS;
                if (batch_col < batch_cols) {
                    const int col = batch_start + batch_col;
                    double value = 0.0;
                    for (int j = 0; j < NEL; ++j) {
                        const int derivative_idx = i * NEL + j;
                        const double derivative = -aff01 * d0_reference[derivative_idx]
                                                + aff00 * d1_reference[derivative_idx];
                        value += derivative * local_rhs[j * RAW_BATCH_COLS + batch_col];
                    }
                    if (col != source_col) {
                        const int col_face = col / NTR;
                        const int local_trace_dof = col - col_face * NTR;
                        const double face_scale = jacs_el_fc[element * 3 + col_face];
                        const double coupling = face_scale
                            * face_element_trace[(col_face * NEL + i) * NTR + local_trace_dof];
                        value -= normals[(element * 3 + col_face) * 2 + 1] * coupling;
                    }
                    tmp_cols[idx] = value;
                }
            }
            __syncthreads();
            for (int idx = tid; idx < NEL * RAW_BATCH_COLS; idx += blockDim.x) {
                const int i = idx / RAW_BATCH_COLS;
                const int batch_col = idx - i * RAW_BATCH_COLS;
                if (batch_col < batch_cols) {
                    double value = 0.0;
                    for (int k = 0; k < NEL; ++k) {
                        value += mass_inverse[i * NEL + k]
                               * tmp_cols[k * RAW_BATCH_COLS + batch_col];
                    }
                    k_d0[idx] = jac_inv * value;
                }
            }
            __syncthreads();
            for (int idx = tid; idx < 3 * NTR * RAW_BATCH_COLS; idx += blockDim.x) {
                const int task = idx / RAW_BATCH_COLS;
                const int batch_col = idx - task * RAW_BATCH_COLS;
                if (batch_col < batch_cols) {
                    const int row_face = task / NTR;
                    const int row_dof = task - row_face * NTR;
                    const long long oriented_face = loc2oriented_face_coupling[element * 3 + row_face];
                    const double scale = jacs_el_fc[element * 3 + row_face];
                    const double ny = normals[(element * 3 + row_face) * 2 + 1];
                    double value = 0.0;
                    for (int i = 0; i < NEL; ++i) {
                        const double lift = scale
                            * oriented_lifts[(oriented_face * NTR + row_dof) * NEL + i];
                        value += ny * lift * k_d0[i * RAW_BATCH_COLS + batch_col];
                    }
                    row_values[idx] += value;
                }
            }
            __syncthreads();

            // Scatter only the columns present in this batch.
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
                double rhs_value = 0.0;
                if (source_col >= batch_start && source_col < batch_start + batch_cols) {
                    rhs_value = row_values[task * RAW_BATCH_COLS + source_col - batch_start];
                }
                int col_block_pos = 0;
                for (int col_face = 0; col_face < 3; ++col_face) {
                    const long long col_edge = loc2glob_edge[element * 3 + col_face];
                    const long long col_solve_edge = edge_to_solve_edge[col_edge];
                    const bool positive = orientations[element * 3 + col_face];
                    for (int col_dof = 0; col_dof < NTR; ++col_dof) {
                        const int local_col_dof = raw_trace_local_dof(positive, col_dof, NTR);
                        const double col_sign = raw_trace_orientation_sign(positive, col_dof);
                        const int column = col_face * NTR + local_col_dof;
                        if (column < batch_start || column >= batch_start + batch_cols) {
                            continue;
                        }
                        const double schur_value = col_sign
                            * row_values[task * RAW_BATCH_COLS + column - batch_start];
                        if (col_solve_edge >= 0) {
#if !RAW_RHS_ONLY
#if RAW_MATRIX_CSR
                            const int block_pos = side_csr_block_pos[side_id * 3 + col_face];
#if RAW_MATRIX_BSR
                            const long long out = (
                                ((long long)csr_indptr[row_solve_edge] + block_pos) * NTR
                                + row_dof) * NTR + col_dof;
#else
                            const long long row = row_solve_edge * NTR + row_dof;
                            const long long out = (long long)csr_indptr[row]
                                                + ((long long)block_pos * NTR + col_dof);
#endif
                            atomicAdd(&data[out], -schur_value);
#else
                            const long long out = side_base
                                + ((long long)col_block_pos * NTR + row_dof) * NTR + col_dof;
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
                atomicAdd(&rhs[row_solve_edge * NTR + row_dof], rhs_value);
            }
            __syncthreads();
        }
#else
        solve_diffusion_all_columns_coop_raw(schur_matrix, pivots, local_rhs);

#if RAW_SOURCE_ONLY
        for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
            d0[idx] = aff11 * d0_reference[idx] - aff10 * d1_reference[idx];
        }
        __syncthreads();
        for (int idx = tid; idx < NEL * NCOLS; idx += blockDim.x) {
            const int i = idx / NCOLS;
            const int col = idx - i * NCOLS;
            double value = 0.0;
            for (int j = 0; j < NEL; ++j) {
                value += d0[i * NEL + j] * local_rhs[j * NCOLS + col];
            }
            tmp_cols[idx] = value;
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

        for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
            d0[idx] = -aff01 * d0_reference[idx] + aff00 * d1_reference[idx];
        }
        __syncthreads();
        for (int idx = tid; idx < NEL * NCOLS; idx += blockDim.x) {
            const int i = idx / NCOLS;
            const int col = idx - i * NCOLS;
            double value = 0.0;
            for (int j = 0; j < NEL; ++j) {
                value += d0[i * NEL + j] * local_rhs[j * NCOLS + col];
            }
            tmp_cols[idx] = value;
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
#else
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

#endif

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
#if RAW_MATRIX_BSR
                        const long long out = (
                            ((long long)csr_indptr[row_solve_edge] + block_pos) * NTR
                            + row_dof) * NTR + col_dof;
#else
                        const long long row = row_solve_edge * NTR + row_dof;
                        const long long out = (long long)csr_indptr[row] + ((long long)block_pos * NTR + col_dof);
#endif
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
#endif
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
#if RAW_MATRIX_BSR
            const long long out = (
                ((long long)csr_indptr[solve_edge] + block_pos) * NTR + i) * NTR + j;
#else
            const long long row = solve_edge * NTR + i;
            const long long out = (long long)csr_indptr[row] + ((long long)block_pos * NTR + j);
#endif
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
        const double* __restrict__ schur_lu_cache,
        const int* __restrict__ schur_pivot_cache,
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
#if RAW_USE_CACHED_FACTORS
            schur_matrix[ij] = schur_lu_cache[element * NEL * NEL + ij];
#else
            schur_matrix[ij] = tau_value;
#endif
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

#if !RAW_USE_CACHED_FACTORS
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

#endif

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

#if !RAW_USE_CACHED_FACTORS
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
#else
    for (int k = 0; k < NEL; ++k) {
        pivots[k] = schur_pivot_cache[element * NEL + k];
    }
#endif
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

__device__ __forceinline__ void solve_diffusion_lu_column_raw(
        const double* __restrict__ factor,
        const int* __restrict__ pivots,
        double* __restrict__ rhs)
{
    // A single condensed RHS has strict row dependencies.  Keeping the whole
    // solve in one thread avoids 3*NEL block barriers and warp reductions.
    if (threadIdx.x == 0) {
        for (int k = 0; k < NEL; ++k) {
            const int pivot = pivots[k];
            if (pivot != k) {
                const double tmp = rhs[k];
                rhs[k] = rhs[pivot];
                rhs[pivot] = tmp;
            }
        }
        for (int i = 0; i < NEL; ++i) {
            double value = rhs[i];
            for (int j = 0; j < i; ++j) {
                value -= factor[i * NEL + j] * rhs[j];
            }
            rhs[i] = value;
        }
        for (int i = NEL - 1; i >= 0; --i) {
            double value = rhs[i];
            for (int j = i + 1; j < NEL; ++j) {
                value -= factor[i * NEL + j] * rhs[j];
            }
            rhs[i] = value / factor[i * NEL + i];
        }
    }
    __syncthreads();
}

extern "C" __global__ void reconstruct_diffusion_raw_coop(
        double* __restrict__ uh,
        double* __restrict__ local_unknowns,
        const int write_local_unknowns,
        const double* __restrict__ schur_lu_cache,
        const int* __restrict__ schur_pivot_cache,
        const double* __restrict__ trace,
        const long long* __restrict__ loc2glob_edge,
        const long long* __restrict__ loc2oriented_face_coupling,
        const double* __restrict__ aff_mats,
        const double* __restrict__ aff_jacs,
        const double* __restrict__ jacs_el_fc,
        const double* __restrict__ normals,
        const double* __restrict__ mass_inverse,
        const double* __restrict__ mass_inverse_d0_reference,
        const double* __restrict__ mass_inverse_d1_reference,
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
    double* mn0 = schur_matrix + (NEL * NEL);
    double* mn1 = mn0 + (NEL * NEL);
    double* matrix_work = mn1 + (NEL * NEL);
    double* rhs0 = matrix_work + (NEL * NEL);
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
        mn0[idx] = normal_x_value - d0v;
        mn1[idx] = normal_y_value - d1v;
#if RAW_USE_CACHED_FACTORS
        schur_matrix[idx] = schur_lu_cache[element * NEL * NEL + idx];
#else
        schur_matrix[idx] = tau_value;
#endif
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
#if !RAW_USE_CACHED_FACTORS

    for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
        matrix_work[idx] = aff11 * mass_inverse_d0_reference[idx]
                         - aff10 * mass_inverse_d1_reference[idx];
    }
    __syncthreads();
    for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
        const int i = idx / NEL;
        const int j = idx - i * NEL;
        double value = 0.0;
        for (int k = 0; k < NEL; ++k) {
            value += mn0[i * NEL + k] * matrix_work[k * NEL + j];
        }
        schur_matrix[idx] += jac_inv * value;
    }
    __syncthreads();
    for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
        matrix_work[idx] = -aff01 * mass_inverse_d0_reference[idx]
                          + aff00 * mass_inverse_d1_reference[idx];
    }
    __syncthreads();
    for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
        const int i = idx / NEL;
        const int j = idx - i * NEL;
        double value = 0.0;
        for (int k = 0; k < NEL; ++k) {
            value += mn1[i * NEL + k] * matrix_work[k * NEL + j];
        }
        schur_matrix[idx] += jac_inv * value;
    }
    __syncthreads();

#endif

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

#if RAW_USE_CACHED_FACTORS
    for (int k = tid; k < NEL; k += blockDim.x) {
        pivots[k] = schur_pivot_cache[element * NEL + k];
    }
    __syncthreads();
#else
    factor_diffusion_reconstruct_lu_coop_raw(schur_matrix, pivots);
#endif
    solve_diffusion_lu_column_raw(schur_matrix, pivots, red_rhs);

    if (write_local_unknowns) {
        for (int i = tid; i < NEL; i += blockDim.x) {
            double qx_value = -tmp1[i];
            double qy_value = -tmp2[i];
            for (int j = 0; j < NEL; ++j) {
                const double u_value = red_rhs[j];
                qx_value += (
                    aff11 * mass_inverse_d0_reference[i * NEL + j]
                    - aff10 * mass_inverse_d1_reference[i * NEL + j]
                ) * u_value;
                qy_value += (
                    -aff01 * mass_inverse_d0_reference[i * NEL + j]
                    + aff00 * mass_inverse_d1_reference[i * NEL + j]
                ) * u_value;
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


def _batched_full_column_count(nel: int, ntr: int, ncols: int) -> int:
    """Largest full-assembly batch that keeps shared memory at most 32 KiB."""
    nel = int(nel)
    ntr = int(ntr)
    ncols = int(ncols)
    fixed_bytes = nel * 4 + 256
    available_doubles = max(0, (32 * 1024 - fixed_bytes) // 8)
    per_column_doubles = 2 * nel + 3 * ntr
    by_shared_memory = max(
        1,
        (available_doubles - 4 * nel * nel) // per_column_doubles,
    )
    # Recovered fluxes alias one NEL-by-NEL matrix workspace.
    return max(1, min(ncols, nel, by_shared_memory))


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
        batched_full: bool = False,
        use_cached_factors: bool = False,
        write_cached_factors: bool = False,
) -> str:
    """Build a parameterized CUDA kernel source for raw diffusion assembly."""
    matrix_format = str(matrix_format).lower()
    if matrix_format not in {'coo', 'csr', 'bsr'}:
        raise ValueError("matrix_format must be 'coo', 'csr', or 'bsr'")
    trace_orientation_mode = int(trace_orientation_mode)
    if trace_orientation_mode not in {0, 1}:
        raise ValueError("trace_orientation_mode must be 0 or 1")
    prefix = (
        f"#define RAW_MATRIX_CSR {1 if matrix_format in {'csr', 'bsr'} else 0}\n"
        f"#define RAW_MATRIX_BSR {1 if matrix_format == 'bsr' else 0}\n"
        f"#define RAW_RHS_ONLY {1 if rhs_only else 0}\n"
        f"#define RAW_SOURCE_ONLY {1 if source_only else 0}\n"
        f"#define RAW_BATCHED_FULL {1 if batched_full else 0}\n"
        f"#define RAW_BATCH_COLS {_batched_full_column_count(nel, ntr, ncols)}\n"
        f"#define RAW_USE_CACHED_FACTORS {1 if use_cached_factors else 0}\n"
        f"#define RAW_WRITE_CACHED_FACTORS {1 if write_cached_factors else 0}\n"
        f"#define TRACE_ORIENTATION_MODE {trace_orientation_mode}\n"
    )
    source = template.replace('NEL', str(int(nel))).replace('NTR', str(int(ntr))).replace('NCOLS', str(int(ncols)))
    return prefix + _RAW_TRACE_ORIENTATION_HELPERS + source


def _shared_sizes(nel: int, ntr: int) -> tuple[int, int]:
    # Serial assembly stores one RHS/solution column at a time.
    """Compute serial assembly and reconstruction shared-memory requirements."""
    assembly_doubles = 7 * nel * nel + 6 * nel
    assembly_bytes = assembly_doubles * REAL_ITEMSIZE + nel * 4 + 256
    reconstruct_doubles = 7 * nel * nel + 6 * nel
    reconstruct_bytes = reconstruct_doubles * REAL_ITEMSIZE + nel * 4 + 256
    return assembly_bytes, reconstruct_bytes


def _coop_reconstruct_shared_size(nel: int) -> int:
    """Shared memory for compact cooperative mixed-field reconstruction."""
    reconstruct_doubles = 4 * nel * nel + 6 * nel
    return reconstruct_doubles * REAL_ITEMSIZE + nel * 4 + 256


def _coop_shared_sizes(
        nel: int,
        ntr: int,
        *,
        ncols: int | None = None,
        source_only: bool = False,
        batched_full: bool = False,
) -> int:
    """Compute cooperative assembly shared-memory requirements."""
    ncols = 3 * ntr + 1 if ncols is None else int(ncols)
    if source_only:
        # Source-only RHS assembly reuses one matrix workspace for each spatial
        # direction instead of retaining all six derivative intermediates.
        assembly_doubles = 4 * nel * nel + 4 * nel * ncols
    elif batched_full:
        # Full assembly retains the Schur factor and two directional coupling
        # matrices, then condenses the widest column batch that remains at or
        # below 32 KiB.  Limiting it to NEL lets a matrix workspace hold fluxes.
        batch_cols = _batched_full_column_count(nel, ntr, ncols)
        assembly_doubles = (
            4 * nel * nel
            + 2 * nel * batch_cols
            + 3 * ntr * batch_cols
        )
    else:
        assembly_doubles = 7 * nel * nel + 4 * nel * ncols
    return assembly_doubles * REAL_ITEMSIZE + nel * 4 + 256


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
        cached_factors: RawDiffusionAssemblyResult | None = None,
        local_factor_key: tuple[Any, ...] | None = None,
) -> RawDiffusionAssemblyResult:
    """Assemble only the reduced RHS for a cached compressed diffusion operator."""
    cupy = require_cupy()
    raw_wall_start = time.perf_counter()
    validate_raw_cuda_supported(cspace, trace_ref)
    if csr_pattern is None:
        raise ValueError('csr_pattern is required for raw-CUDA cached RHS assembly')
    matrix_format = str(getattr(csr_pattern, "matrix_format", "csr")).lower()
    if matrix_format not in {"csr", "bsr"}:
        raise ValueError("cached raw-CUDA RHS assembly requires a CSR or BSR face graph")
    block_size = resolve_raw_cuda_block_size(
        block_size, equation="diffusion-reaction", order=cspace.order
    )
    if block_size not in {1, 32, 64, 128}:
        raise ValueError('raw CUDA diffusion block_size must be one of 1, 32, 64, 128')
    if matrix_format == "bsr" and block_size == 1:
        raise ValueError("raw CUDA cached BSR RHS assembly requires a cooperative block size")
    timings: dict[str, float] = {}
    mesh_h = cspace.host.mesh
    nel = int(cspace.el_dof)
    ntr = int(cspace.edg_dof)
    boundary_is_zero = bool(cupy.all(boundary_trace == 0.0).get())
    ncols = 1 if boundary_is_zero else 3 * ntr + 1
    assembly_shared = (
        _shared_sizes(nel, ntr)[0]
        if block_size == 1
        else _coop_shared_sizes(nel, ntr, ncols=ncols, source_only=boundary_is_zero)
    )
    trace_orientation_mode = _raw_trace_orientation_mode(trace_ref)
    use_cached_factors = cached_factors is not None
    factor_kind = str(getattr(cached_factors, "local_factor_kind", "schur-lu")) if use_cached_factors else "schur-lu"
    if factor_kind != "schur-lu":
        raise ValueError("raw CUDA cached local factors must use 'schur-lu'")
    if use_cached_factors:
        schur_lu, schur_pivots = _validate_local_schur_factors(
            cached_factors, num_elements=int(cspace.mesh.num_tri), nel=nel, local_factor_key=local_factor_key
        )
        local_factor_bytes = int(cached_factors.local_factor_bytes)
    else:
        schur_lu = cupy.empty(1, dtype=REAL_DTYPE)
        schur_pivots = cupy.empty(1, dtype=cupy.int32)
        local_factor_bytes = 0

    edge_to_solve = csr_pattern.edge_to_solve_edge
    side_index = csr_pattern.interior_side_index
    indptr = csr_pattern.indptr
    indices = csr_pattern.indices
    side_map_arg = csr_pattern.side_csr_block_pos
    mass_map_arg = csr_pattern.mass_csr_block_pos

    timings['raw.setup'] = time.perf_counter() - raw_wall_start
    zero_start = time.perf_counter()
    dummy_data = cupy.zeros(1 if block_size != 1 else indices.size, dtype=REAL_DTYPE)
    rhs = cupy.zeros(mesh_h.int_edges_inds.size * ntr, dtype=REAL_DTYPE)
    boundary_trace_full = cupy.zeros((mesh_h.num_edg, ntr), dtype=REAL_DTYPE)
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
            use_cached_factors=use_cached_factors,
        )
        kernel, kernel_jit = _compile_kernel_timed(cupy, source, 'assemble_diffusion_raw_csr', assembly_shared)
        kernel_args = (
            indptr,
            dummy_data,
            rhs,
            schur_lu,
            schur_pivots,
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
            REAL_DTYPE(tau),
            np.int64(cspace.mesh.num_tri),
            np.int64(cspace.mesh.int_edges_inds.size),
            np.int64(0),
        )
    else:
        dummy_i64 = cupy.empty(1, dtype=cupy.int64)
        dummy_i32 = cupy.empty(1, dtype=cupy.int32)
        kernel_template = _RAW_ASSEMBLY_COOP_TEMPLATE
        kernel_name = 'assemble_diffusion_raw_coop'
        if matrix_format == 'bsr':
            kernel_name = 'assemble_diffusion_raw_coop_bsr'
            kernel_template = kernel_template.replace(
                'void assemble_diffusion_raw_coop(',
                'void assemble_diffusion_raw_coop_bsr(',
            )
        source = _kernel_source(
            kernel_template,
            nel=nel,
            ntr=ntr,
            ncols=ncols,
            matrix_format=matrix_format,
            trace_orientation_mode=trace_orientation_mode,
            rhs_only=True,
            source_only=boundary_is_zero,
            use_cached_factors=use_cached_factors,
        )
        kernel, kernel_jit = _compile_kernel_timed(cupy, source, kernel_name, assembly_shared)
        kernel_args = (
            dummy_i64,
            dummy_i64,
            indptr,
            dummy_data,
            rhs,
            schur_lu,
            schur_pivots,
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
            d0_reference,
            d1_reference,
            trace_ref.face_element_test_trace_trial,
            face_element_mass,
            trace_ref.face_element_test_trace_trial,
            trace_ref.M_rf_fc,
            trace_ref.face_trace_test_element_trial_oriented,
            d0_reference,
            d1_reference,
            source_rhs,
            boundary_trace_full.reshape(-1),
            REAL_DTYPE(tau),
            np.int64(cspace.mesh.num_tri),
            np.int64(cspace.mesh.int_edges_inds.size),
            np.int64(0),
        )
    stream = cupy.cuda.get_current_stream()
    stream.synchronize()
    timings['raw.kernel.jit'] = kernel_jit
    timings['raw.kernel.prepare'] = max(0.0, time.perf_counter() - start - kernel_jit)
    grid = (max(int(cspace.mesh.num_tri), int(cspace.mesh.int_edges_inds.size)),)
    begin = cupy.cuda.Event()
    end = cupy.cuda.Event()
    launch_wall_start = time.perf_counter()
    begin.record(stream)
    kernel(grid, (block_size,), kernel_args, shared_mem=int(assembly_shared))
    end.record(stream)
    end.synchronize()
    device_seconds = cupy.cuda.get_elapsed_time(begin, end) / 1000.0
    launch_wall_seconds = time.perf_counter() - launch_wall_start
    timings['raw.kernel.device'] = device_seconds
    timings['raw.kernel.wall'] = launch_wall_seconds
    timings['raw.cached_rhs_kernel'] = device_seconds
    timings['raw.block_size'] = float(block_size)
    timings['raw.local_factors.reused'] = float(use_cached_factors)
    timings['raw.local_factors.bytes'] = float(local_factor_bytes)
    timings['raw.total'] = (
        timings.get('raw.setup', 0.0)
        + timings.get('raw.cached_rhs_zero', 0.0)
        + timings.get('raw.kernel.jit', 0.0)
        + timings.get('raw.kernel.prepare', 0.0)
        + timings.get('raw.kernel.wall', 0.0)
    )
    timings['raw.wall_total'] = time.perf_counter() - raw_wall_start
    timings['raw.unaccounted'] = max(
        0.0, timings['raw.wall_total'] - timings['raw.total']
    )
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
        matrix_format=matrix_format,
        csr_pattern=csr_pattern,
        schur_lu=None if not use_cached_factors else schur_lu,
        schur_pivots=None if not use_cached_factors else schur_pivots,
        local_factor_key=None if not use_cached_factors else cached_factors.local_factor_key,
        local_factor_bytes=local_factor_bytes,
        local_factor_kind=factor_kind if use_cached_factors else "none",
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
        cache_local_factors: bool = False,
        local_factor_kind: str = "schur-lu",
        local_factor_key: tuple[Any, ...] | None = None,
) -> RawDiffusionAssemblyResult:
    """Assemble the reduced trace system/RHS with a Raw CUDA fused element loop."""
    cupy = require_cupy()
    raw_wall_start = time.perf_counter()
    validate_raw_cuda_supported(cspace, trace_ref)
    matrix_format = str(matrix_format).lower()
    if matrix_format not in {'coo', 'csr', 'bsr'}:
        raise ValueError("matrix_format must be 'coo', 'csr', or 'bsr'")
    if local_factor_kind != "schur-lu":
        raise ValueError("raw CUDA local_factor_kind must be 'schur-lu'")
    block_size = resolve_raw_cuda_block_size(
        block_size, equation="diffusion-reaction", order=cspace.order
    )
    if block_size not in {1, 32, 64, 128}:
        raise ValueError('raw CUDA diffusion block_size must be one of 1, 32, 64, 128')
    if matrix_format == 'bsr' and block_size == 1:
        raise ValueError("raw CUDA BSR assembly requires a cooperative block_size of 32, 64, or 128")
    timings: dict[str, float] = {}
    mesh_h = cspace.host.mesh
    nel = int(cspace.el_dof)
    ntr = int(cspace.edg_dof)
    ncols = 3 * ntr + 1
    assembly_shared = (
        _shared_sizes(nel, ntr)[0]
        if block_size == 1
        else _coop_shared_sizes(nel, ntr, batched_full=True)
    )
    trace_orientation_mode = _raw_trace_orientation_mode(trace_ref)
    reference_precompute_start = time.perf_counter()
    if block_size != 1:
        mass_inverse = cspace.quad_data.MKrf_inv
        mass_inverse_d0_reference = cupy.ascontiguousarray(mass_inverse @ d0_reference)
        mass_inverse_d1_reference = cupy.ascontiguousarray(mass_inverse @ d1_reference)
        mass_inverse_face_element_trace = cupy.ascontiguousarray(
            cupy.einsum(
                "ik,fkj->fij",
                mass_inverse,
                trace_ref.face_element_test_trace_trial,
                optimize=True,
            )
        )
    cupy.cuda.get_current_stream().synchronize()
    timings['raw.reference_precompute'] = time.perf_counter() - reference_precompute_start
    factor_allocation_start = time.perf_counter()
    if cache_local_factors:
        schur_lu, schur_pivots, local_factor_bytes = _allocate_local_schur_factors(
            cupy, num_elements=int(cspace.mesh.num_tri), nel=nel, factor_kind=local_factor_kind
        )
    else:
        schur_lu = cupy.empty(1, dtype=REAL_DTYPE)
        schur_pivots = cupy.empty(1, dtype=cupy.int32)
        local_factor_bytes = 0
    timings['raw.local_factors.allocate'] = time.perf_counter() - factor_allocation_start
    timings['raw.local_factors.bytes'] = float(local_factor_bytes)
    timings['raw.local_factors.created'] = float(bool(cache_local_factors))

    csr_pattern = None
    indptr = indices = None
    if matrix_format in {'csr', 'bsr'}:
        from hdgfem.hdg.cuda.pattern import build_reduced_csr_pattern_raw

        start = time.perf_counter()
        csr_pattern = build_reduced_csr_pattern_raw(
            cspace, timings, matrix_format=matrix_format
        )
        edge_to_solve = csr_pattern.edge_to_solve_edge
        side_index = csr_pattern.interior_side_index
        indptr = csr_pattern.indptr
        indices = csr_pattern.indices
        cupy.cuda.get_current_stream().synchronize()
        timings['raw.map_setup'] = timings.get('raw.csr_pattern.total', 0.0)
        timings[f'raw.{matrix_format}_pattern.wrapper'] = time.perf_counter() - start

        zero_start = time.perf_counter()
        rows = cols = None
        data = (
            cupy.zeros((csr_pattern.num_blocks, ntr, ntr), dtype=REAL_DTYPE)
            if matrix_format == 'bsr'
            else cupy.zeros(indices.size, dtype=REAL_DTYPE)
        )
        rhs = cupy.zeros(mesh_h.int_edges_inds.size * ntr, dtype=REAL_DTYPE)
        boundary_trace_full = cupy.zeros((mesh_h.num_edg, ntr), dtype=REAL_DTYPE)
        if mesh_h.bnd_edges_inds.size:
            boundary_trace_full[cspace.mesh.bnd_edges_inds] = boundary_trace
        cupy.cuda.get_current_stream().synchronize()
        timings[f'raw.{matrix_format}_zero'] = time.perf_counter() - zero_start
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
        data = cupy.empty(nnz, dtype=REAL_DTYPE)
        rhs = cupy.zeros(mesh_h.int_edges_inds.size * ntr, dtype=REAL_DTYPE)
        boundary_trace_full = cupy.zeros((mesh_h.num_edg, ntr), dtype=REAL_DTYPE)
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
                write_cached_factors=bool(cache_local_factors),
            )
            kernel, kernel_jit = _compile_kernel_timed(cupy, source, 'assemble_diffusion_raw_csr', assembly_shared)
            kernel_args = (
                indptr,
                data,
                rhs,
                schur_lu,
                schur_pivots,
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
                REAL_DTYPE(tau),
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
                write_cached_factors=bool(cache_local_factors),
            )
            kernel, kernel_jit = _compile_kernel_timed(cupy, source, 'assemble_diffusion_raw', assembly_shared)
            kernel_args = (
                rows,
                cols,
                data,
                rhs,
                schur_lu,
                schur_pivots,
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
                REAL_DTYPE(tau),
                np.int64(cspace.mesh.num_tri),
                np.int64(cspace.mesh.int_edges_inds.size),
                np.int64(n_flux),
            )
    else:
        dummy_i64 = cupy.empty(1, dtype=cupy.int64)
        dummy_i32 = cupy.empty(1, dtype=cupy.int32)
        if matrix_format in {'csr', 'bsr'}:
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
        kernel_template = _RAW_ASSEMBLY_COOP_TEMPLATE
        kernel_name = 'assemble_diffusion_raw_coop'
        if matrix_format == 'bsr':
            kernel_name = 'assemble_diffusion_raw_coop_bsr'
            kernel_template = kernel_template.replace(
                'void assemble_diffusion_raw_coop(',
                'void assemble_diffusion_raw_coop_bsr(',
            )
        source = _kernel_source(
            kernel_template,
            nel=nel,
            ntr=ntr,
            ncols=ncols,
            matrix_format=matrix_format,
            trace_orientation_mode=trace_orientation_mode,
            batched_full=True,
            write_cached_factors=bool(cache_local_factors),
        )
        kernel, kernel_jit = _compile_kernel_timed(cupy, source, kernel_name, assembly_shared)
        kernel_args = (
            rows_arg,
            cols_arg,
            csr_indptr_arg,
            data,
            rhs,
            schur_lu,
            schur_pivots,
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
            mass_inverse_d0_reference,
            mass_inverse_d1_reference,
            mass_inverse_face_element_trace,
            face_element_mass,
            trace_ref.face_element_test_trace_trial,
            trace_ref.M_rf_fc,
            trace_ref.face_trace_test_element_trial_oriented,
            d0_reference,
            d1_reference,
            source_rhs,
            boundary_trace_full.reshape(-1),
            REAL_DTYPE(tau),
            np.int64(cspace.mesh.num_tri),
            np.int64(cspace.mesh.int_edges_inds.size),
            np.int64(n_flux),
        )
    stream = cupy.cuda.get_current_stream()
    stream.synchronize()
    timings['raw.kernel.jit'] = kernel_jit
    timings['raw.kernel.prepare'] = max(0.0, time.perf_counter() - start - kernel_jit)
    grid = (max(int(cspace.mesh.num_tri), int(cspace.mesh.int_edges_inds.size)),)
    begin = cupy.cuda.Event()
    end = cupy.cuda.Event()
    launch_wall_start = time.perf_counter()
    begin.record(stream)
    kernel(grid, (block_size,), kernel_args, shared_mem=int(assembly_shared))
    end.record(stream)
    end.synchronize()
    device_seconds = cupy.cuda.get_elapsed_time(begin, end) / 1000.0
    launch_wall_seconds = time.perf_counter() - launch_wall_start
    timings['raw.kernel.device'] = device_seconds
    timings['raw.kernel.wall'] = launch_wall_seconds
    timings['raw.kernel' if matrix_format == 'coo' else f'raw.{matrix_format}_kernel'] = device_seconds
    timings['raw.block_size'] = float(block_size)
    map_wall = timings.get(
        f'raw.{matrix_format}_pattern.wrapper', timings.get('raw.map_setup', 0.0)
    )
    timings['raw.total'] = (
        timings.get('raw.reference_precompute', 0.0)
        + timings.get('raw.local_factors.allocate', 0.0)
        + map_wall
        + timings.get(f'raw.{matrix_format}_zero', 0.0)
        + timings.get('raw.kernel.jit', 0.0)
        + timings.get('raw.kernel.prepare', 0.0)
        + timings.get('raw.kernel.wall', 0.0)
    )
    timings['raw.wall_total'] = time.perf_counter() - raw_wall_start
    timings['raw.unaccounted'] = max(
        0.0, timings['raw.wall_total'] - timings['raw.total']
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
        schur_lu=None if not cache_local_factors else schur_lu,
        schur_pivots=None if not cache_local_factors else schur_pivots,
        local_factor_key=local_factor_key if cache_local_factors else None,
        local_factor_bytes=local_factor_bytes,
        local_factor_kind=local_factor_kind if cache_local_factors else "none",
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
        cached_factors: RawDiffusionAssemblyResult | None = None,
        local_factor_key: tuple[Any, ...] | None = None,
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
    use_cached_factors = cached_factors is not None
    factor_kind = str(getattr(cached_factors, "local_factor_kind", "schur-lu")) if use_cached_factors else "schur-lu"
    if factor_kind != "schur-lu":
        raise ValueError("raw CUDA reconstruction supports only cached 'schur-lu' factors")
    if use_cached_factors:
        schur_lu, schur_pivots = _validate_local_schur_factors(
            cached_factors, num_elements=int(cspace.mesh.num_tri), nel=nel, local_factor_key=local_factor_key
        )
    else:
        schur_lu = cupy.empty(1, dtype=REAL_DTYPE)
        schur_pivots = cupy.empty(1, dtype=cupy.int32)
    reconstruct_shared = (
        _shared_sizes(nel, ntr)[1]
        if block_size == 1
        else _coop_reconstruct_shared_size(nel)
    )
    if block_size == 1:
        reference_args = (
            cspace.quad_data.MKrf_inv,
            face_element_mass,
        )
    else:
        mass_inverse = cspace.quad_data.MKrf_inv
        reference_args = (
            mass_inverse,
            cupy.ascontiguousarray(mass_inverse @ d0_reference),
            cupy.ascontiguousarray(mass_inverse @ d1_reference),
            face_element_mass,
        )
    uh = cupy.empty((cspace.mesh.num_tri, nel), dtype=REAL_DTYPE)
    local_unknowns = (
        cupy.empty((cspace.mesh.num_tri, 3 * nel), dtype=REAL_DTYPE)
        if return_local_unknowns
        else cupy.empty(1, dtype=REAL_DTYPE)
    )
    template = _RAW_RECONSTRUCT_TEMPLATE if block_size == 1 else _RAW_RECONSTRUCT_COOP_TEMPLATE
    kernel_name = 'reconstruct_diffusion_raw' if block_size == 1 else 'reconstruct_diffusion_raw_coop'
    source = _kernel_source(
        template,
        nel=nel,
        ntr=ntr,
        ncols=1,
        trace_orientation_mode=_raw_trace_orientation_mode(trace_ref),
        use_cached_factors=use_cached_factors,
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
            schur_lu,
            schur_pivots,
            trace.reshape(-1),
            cspace.mesh.loc2glob_edge,
            cspace.mesh.loc2oriented_face_coupling,
            cspace.mesh.aff_mats,
            cspace.mesh.aff_jacs,
            cspace.mesh.jacs_el_fc,
            cspace.mesh.normals,
            *reference_args,
            trace_ref.face_trace_test_element_trial_oriented,
            d0_reference,
            d1_reference,
            source_rhs,
            REAL_DTYPE(tau),
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
    return doubles * REAL_ITEMSIZE + rows * 4 + 256


def _as_scalar_or_none(value) -> float | None:
    """Return a finite scalar coefficient or None for non-scalar data."""
    if np.isscalar(value):
        return float(value)
    try:
        array = np.asarray(value, dtype=REAL_DTYPE)
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
        array = np.asarray(diffusion, dtype=REAL_DTYPE)
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
    from hdgfem.solvers.diffusion_reaction import _new_hdg_postprocess_cache

    trace_ref = cspace.host.trace_space('legacy-lagrange') if trace_space is None else trace_space
    if cache is None or cache.base_space is not cspace.host or cache.trace_space is not trace_ref:
        cache = _new_hdg_postprocess_cache(cspace.host, trace_ref)
    cpost_space = as_cupy_space(cache.post_space, device=cspace.device_id)
    local_unknowns = cupy.ascontiguousarray(cupy.asarray(local_unknowns, dtype=REAL_DTYPE))
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

    weights = cupy.asarray(cache.post_space.quad_data.Krf_w, dtype=REAL_DTYPE)
    base_basis = cupy.asarray(cache.base_basis_on_post_quads, dtype=REAL_DTYPE)
    post_grad = cupy.asarray(cache.post_space.quad_data.gphi, dtype=REAL_DTYPE)
    mean_base = cupy.asarray(cache.mean_base, dtype=REAL_DTYPE)
    mean_post = cupy.asarray(cache.mean_post, dtype=REAL_DTYPE)
    stiffness_rr = cupy.asarray(cache.primal_stiffness_rr, dtype=REAL_DTYPE)
    stiffness_rs = cupy.asarray(cache.primal_stiffness_rs, dtype=REAL_DTYPE)
    stiffness_ss = cupy.asarray(cache.primal_stiffness_ss, dtype=REAL_DTYPE)
    inverse_constants = _constant_inverse_diffusion_components(diffusion)
    if inverse_constants is None:
        from hdgfem.solvers.diffusion_reaction import _inverse_diffusion_values

        inv00_h, inv01_h, inv10_h, inv11_h = _inverse_diffusion_values(diffusion, cache.post_space)
        inverse_mode = np.int32(1)
        inv00_scalar = inv01_scalar = inv10_scalar = inv11_scalar = REAL_DTYPE(0.0)
        inv00 = cupy.asarray(inv00_h, dtype=REAL_DTYPE)
        inv01 = cupy.asarray(inv01_h, dtype=REAL_DTYPE)
        inv10 = cupy.asarray(inv10_h, dtype=REAL_DTYPE)
        inv11 = cupy.asarray(inv11_h, dtype=REAL_DTYPE)
    else:
        inverse_mode = np.int32(0)
        inv00_scalar, inv01_scalar, inv10_scalar, inv11_scalar = map(REAL_DTYPE, inverse_constants)
        inv00 = inv01 = inv10 = inv11 = weights
    coeffs = cupy.empty((cspace.mesh.num_tri, post_el_dof), dtype=REAL_DTYPE)
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
