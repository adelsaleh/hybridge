"""Raw CUDA solve-and-emit kernels for advection-reaction HDG assembly.

This module is the advection counterpart to ``diffusion_raw_cuda``.  The first
implementation intentionally reuses the runner's existing CuPy construction of
per-element local matrices, element-to-trace RHS columns, source moments, and
boundary trace values.  The Raw CUDA kernel then fuses the remaining expensive
assembly stages:

* factor one element-local advection-reaction matrix in shared memory,
* solve all local trace/source columns without materializing a global solved
  tensor,
* emit the boundary-eliminated reduced trace COO matrix directly,
* move known boundary trace-column action directly into the reduced RHS, and
* append the interior trace-mass blocks.

The kernels are heavily commented because there are now two assembly depths in
this file.  The precomputed path accepts local matrices/RHS columns built by
CuPy and fuses only the local solve/COO emission.  The fused-local path mirrors
``hdgfem.transport.numba_kernels``: it consumes already-projected coefficients,
builds the element-local operator/RHS in shared memory, solves it, and emits the
reduced trace matrix without ever materializing dense local tensors globally.

The fused path also exposes two LU policies.  ``lu_mode="safe"`` is the
default and keeps the historical, validated factorization ordering.  The
experimental ``lu_mode="coop"`` path uses a shared-memory pivot reduction and
parallel multiplier/trailing-update work while deliberately keeping row swaps on
thread 0; earlier fully parallel row-swap variants reproduced illegal-address
failures in the fused kernel.  The fused kernels are validated for legacy-lagrange and legendre-modal
trace bases through p <= 7, with p=8--9 available for explicit experimental
qualification, while the precomputed raw path keeps its original p <= 6 guard.
"""

from __future__ import annotations

from hdgfem.runtime.precision import REAL_DTYPE, REAL_ITEMSIZE

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from hdgfem.runtime.optional import require_cupy, require_cupyx_sparse
from hdgfem.hdg.cuda.raw_source import RAW_COOP_LU_FACTOR
from hdgfem.hdg.cuda.launch import RawCudaBlockSize, resolve_raw_cuda_block_size
from hdgfem.hdg.trace_maps import (
    _edge_to_solve_edge,
    _interior_side_index,
    _side_flux_offsets,
)
from hdgfem.hdg.cuda.launch import _compile_kernel, _compile_kernel_timed
from hdgfem.hdg.cuda.pattern import (
    ReducedTraceCsrPattern,
    _edge_to_solve_edge_device,
    build_reduced_csr_pattern_raw,
)


@dataclass(frozen=True)
class RawAdvectionAssemblyResult:
    """Device-side reduced trace system and reconstruction inputs.

    ``lu_mode`` records the fused raw CUDA factorization policy so reconstruction
    uses the same local solve variant as assembly.  Precomputed raw assembly
    leaves it at the default ``"safe"`` value.
    """

    data: Any
    rhs: Any
    boundary_trace: Any
    timings: dict[str, float]
    rows: Any | None = None
    cols: Any | None = None
    indptr: Any | None = None
    indices: Any | None = None
    matrix_format: str = "coo"
    csr_pattern: ReducedTraceCsrPattern | None = None
    local_mats: Any | None = None
    element_boundary: Any | None = None
    source_rhs: Any | None = None
    source_coeffs: Any | None = None
    beta_coeffs: Any | None = None
    reaction_coeffs: Any | None = None
    reaction_scalar: float = 0.0
    reaction_is_scalar: bool = False
    advection_tensor: Any | None = None
    advection_sparse_offsets: Any | None = None
    advection_sparse_modes: Any | None = None
    advection_sparse_values0: Any | None = None
    advection_sparse_values1: Any | None = None
    use_sparse_advection: bool = False
    mass_is_diagonal: bool = False
    local_response: Any | None = None
    lu_mode: str = "safe"
    zero_boundary_flux: bool = False
    advection_stabilization: object = None


@dataclass
class RawAdvectionFactorWorkspace:
    """Owning device LU/pivot/lift storage for repeated transport right-hand sides.

    Assembly refreshes numerical values; shape-compatible allocations survive
    coefficient updates and are reused throughout a frozen RK step.
    """

    local_lu: Any | None = None
    pivots: Any | None = None
    trace_lift: Any | None = None
    signature: tuple | None = None

    def ensure(self, cupy, *, num_elements, nel, ntr):
        """Allocate only when the device, precision, or discrete shape changes."""
        signature = (int(cupy.cuda.runtime.getDevice()), num_elements, nel, ntr, str(REAL_DTYPE))
        if signature == self.signature:
            return
        self.local_lu = cupy.empty((num_elements, nel, nel), dtype=REAL_DTYPE)
        self.pivots = cupy.empty((num_elements, nel), dtype=cupy.int32)
        self.trace_lift = cupy.zeros((num_elements, 3, ntr, nel), dtype=REAL_DTYPE)
        self.signature = signature

    @property
    def nbytes(self):
        """Return additional bytes retained for RHS-only solves."""
        return sum(int(getattr(a, "nbytes", 0)) for a in (self.local_lu, self.pivots, self.trace_lift))


_RAW_LU_SCRATCH_THREADS = 128


def build_reduced_csr_pattern_cupy_reference(cspace, timings: dict[str, float] | None = None) -> ReducedTraceCsrPattern:
    """Reference CuPy reduced trace CSR pattern builder used by tests."""
    cupy = require_cupy()
    sparse = require_cupyx_sparse()
    mesh = cspace.mesh
    ntr = int(cspace.edg_dof)
    num_free = int(mesh.int_edges_inds.size)
    num_sides = int(mesh.interior_elements.size)
    system_size = num_free * ntr
    local_timings: dict[str, float] = {}
    total_start = time.perf_counter()

    start = time.perf_counter()
    edge_to_solve = _edge_to_solve_edge_device(cspace)
    side_ids = cupy.arange(num_sides, dtype=cupy.int64)
    interior_side_index = cupy.full(cspace.mesh.num_tri * 3, -1, dtype=cupy.int64)
    interior_side_index[mesh.interior_elements * 3 + mesh.interior_faces] = side_ids
    row_edges = mesh.loc2glob_edge[mesh.interior_elements, mesh.interior_faces]
    row_solve = edge_to_solve[row_edges]
    col_solve = edge_to_solve[mesh.loc2glob_edge[mesh.interior_elements]]
    valid = col_solve >= 0
    row_solve_b = cupy.broadcast_to(row_solve[:, None], col_solve.shape)
    pair_keys = row_solve_b[valid] * num_free + col_solve[valid]
    diag = cupy.arange(num_free, dtype=cupy.int64)
    pair_keys = cupy.concatenate((pair_keys, diag * num_free + diag))
    block_keys = cupy.unique(pair_keys)
    block_rows = block_keys // num_free
    block_cols = block_keys - block_rows * num_free
    block_counts = cupy.bincount(block_rows, minlength=num_free).astype(cupy.int32)
    block_indptr = cupy.empty(num_free + 1, dtype=cupy.int32)
    block_indptr[0] = 0
    cupy.cumsum(block_counts, out=block_indptr[1:])
    block_neighbors = block_cols.astype(cupy.int32)
    num_blocks = int(block_indptr[-1].get())
    local_timings['raw.csr_pattern.reference_blocks'] = time.perf_counter() - start

    start = time.perf_counter()
    i_grid, j_grid = cupy.meshgrid(cupy.arange(ntr, dtype=cupy.int64), cupy.arange(ntr, dtype=cupy.int64), indexing='ij')
    scalar_rows = (block_rows[:, None, None] * ntr + i_grid[None, :, :]).ravel()
    scalar_cols = (block_cols[:, None, None] * ntr + j_grid[None, :, :]).ravel()
    coo = sparse.coo_matrix(
        (cupy.ones(scalar_rows.size, dtype=REAL_DTYPE), (scalar_rows.astype(cupy.int32), scalar_cols.astype(cupy.int32))),
        shape=(system_size, system_size),
        dtype=REAL_DTYPE,
    )
    csr = coo.tocsr()
    csr.sum_duplicates()
    indptr = csr.indptr.astype(cupy.int32, copy=False)
    indices = csr.indices.astype(cupy.int32, copy=False)
    local_timings['raw.csr_pattern.reference_scalar'] = time.perf_counter() - start

    start = time.perf_counter()
    side_csr_block_pos = cupy.full((num_sides, 3), -1, dtype=cupy.int32)
    emission_keys = row_solve_b * num_free + col_solve
    emission_valid = valid & (row_solve_b >= 0)
    emission_pos = cupy.searchsorted(block_keys, emission_keys[emission_valid])
    emission_rows = row_solve_b[emission_valid]
    side_csr_block_pos[emission_valid] = (emission_pos - block_indptr[emission_rows]).astype(cupy.int32)
    mass_keys = diag * num_free + diag
    mass_pos = cupy.searchsorted(block_keys, mass_keys) - block_indptr[diag]
    mass_csr_block_pos = mass_pos.astype(cupy.int32)
    cupy.cuda.get_current_stream().synchronize()
    local_timings['raw.csr_pattern.reference_maps'] = time.perf_counter() - start
    local_timings['raw.csr_pattern.reference_total'] = time.perf_counter() - total_start
    if timings is not None:
        timings.update(local_timings)
    return ReducedTraceCsrPattern(
        indptr=indptr,
        indices=indices,
        block_indptr=block_indptr,
        block_neighbors=block_neighbors,
        side_csr_block_pos=side_csr_block_pos.reshape(-1),
        mass_csr_block_pos=mass_csr_block_pos,
        edge_to_solve_edge=edge_to_solve,
        interior_side_index=interior_side_index,
        num_blocks=num_blocks,
        timings=local_timings,
    )


def assert_reduced_csr_patterns_equal(reference: ReducedTraceCsrPattern, actual: ReducedTraceCsrPattern) -> None:
    """Raise ``AssertionError`` if two device CSR patterns differ."""
    cupy = require_cupy()
    fields = (
        'indptr',
        'indices',
        'block_indptr',
        'block_neighbors',
        'side_csr_block_pos',
        'mass_csr_block_pos',
    )
    for name in fields:
        ref = getattr(reference, name)
        got = getattr(actual, name)
        if ref.shape != got.shape:
            raise AssertionError(f'{name} shape differs: {ref.shape} != {got.shape}')
        if not bool(cupy.all(ref == got).get()):
            diff = int(cupy.argmax(ref != got).get())
            raise AssertionError(f'{name} differs at flattened index {diff}')

_RAW_ASSEMBLY_TEMPLATE = r"""
extern "C" __global__ void assemble_advection_raw(
        long long* __restrict__ rows,
        long long* __restrict__ cols,
        double* __restrict__ data,
        double* __restrict__ rhs,
        const double* __restrict__ local_mats,
        const double* __restrict__ element_boundary,
        const double* __restrict__ source_rhs,
        const double* __restrict__ boundary_trace,
        const long long* __restrict__ loc2glob_edge,
        const bool* __restrict__ orientations,
        const long long* __restrict__ loc2oriented_face_coupling,
        const long long* __restrict__ interior_side_index,
        const long long* __restrict__ edge_to_solve_edge,
        const long long* __restrict__ int_edges,
        const long long* __restrict__ side_flux_offsets,
        const double* __restrict__ jacs_el_fc,
        const double* __restrict__ side_mass_blocks,
        const double* __restrict__ trace_lift,
        const long long num_elements,
        const long long num_int_edges,
        const long long n_flux)
{
    // One element owns one CUDA block.  v1 uses only thread 0 inside that block,
    // but shared memory keeps the local matrix, RHS columns, and LU pivots close
    // to the executing SM and avoids writing solved element columns globally.
    extern __shared__ unsigned char shared_raw[];
    double* shared = reinterpret_cast<double*>(shared_raw);
    double* local_lu = shared;                         // NEL x NEL local matrix, factorized in place
    double* local_rhs = local_lu + (NEL * NEL);        // NEL x (3*NTR+1) trace/source RHS columns
    int* pivots = reinterpret_cast<int*>(local_rhs + (NEL * NCOLS));

    const long long element = blockIdx.x;

    if (element < num_elements && threadIdx.x == 0) {
        // Copy the element-local operator and RHS columns into shared memory.
        // The local RHS layout matches the CuPy path exactly: three face blocks
        // of trace columns followed by one source column.
        for (int i = 0; i < NEL; ++i) {
            for (int j = 0; j < NEL; ++j) {
                local_lu[i * NEL + j] = local_mats[(element * NEL + i) * NEL + j];
            }
            for (int col = 0; col < NCOLS; ++col) {
                if (col == NCOLS - 1) {
                    local_rhs[i * NCOLS + col] = source_rhs[element * NEL + i];
                } else {
                    local_rhs[i * NCOLS + col] = element_boundary[(element * NEL + i) * (3 * NTR) + col];
                }
            }
        }

        // Dense partial-pivot LU.  As in the diffusion raw backend, this is a
        // solve-based path: the local inverse is never formed or materialized.
        for (int k = 0; k < NEL; ++k) {
            int pivot = k;
            double max_value = fabs(local_lu[k * NEL + k]);
            for (int i = k + 1; i < NEL; ++i) {
                const double value = fabs(local_lu[i * NEL + k]);
                if (value > max_value) {
                    max_value = value;
                    pivot = i;
                }
            }
            pivots[k] = pivot;
            if (pivot != k) {
                for (int j = 0; j < NEL; ++j) {
                    const double tmp = local_lu[k * NEL + j];
                    local_lu[k * NEL + j] = local_lu[pivot * NEL + j];
                    local_lu[pivot * NEL + j] = tmp;
                }
            }
            double diagonal = local_lu[k * NEL + k];
            if (fabs(diagonal) < 1.0e-30) {
                diagonal = diagonal >= 0.0 ? 1.0e-30 : -1.0e-30;
                local_lu[k * NEL + k] = diagonal;
            }
            for (int i = k + 1; i < NEL; ++i) {
                local_lu[i * NEL + k] /= diagonal;
                const double multiplier = local_lu[i * NEL + k];
                for (int j = k + 1; j < NEL; ++j) {
                    local_lu[i * NEL + j] -= multiplier * local_lu[k * NEL + j];
                }
            }
        }

        // Apply pivot row swaps to every RHS column, then solve L and U.  After
        // this block local_rhs contains local_mats^{-1} * [B_trace, f].
        for (int k = 0; k < NEL; ++k) {
            const int pivot = pivots[k];
            if (pivot != k) {
                for (int col = 0; col < NCOLS; ++col) {
                    const double tmp = local_rhs[k * NCOLS + col];
                    local_rhs[k * NCOLS + col] = local_rhs[pivot * NCOLS + col];
                    local_rhs[pivot * NCOLS + col] = tmp;
                }
            }
        }
        for (int i = 0; i < NEL; ++i) {
            for (int col = 0; col < NCOLS; ++col) {
                double value = local_rhs[i * NCOLS + col];
                for (int j = 0; j < i; ++j) {
                    value -= local_lu[i * NEL + j] * local_rhs[j * NCOLS + col];
                }
                local_rhs[i * NCOLS + col] = value;
            }
        }
        for (int i = NEL - 1; i >= 0; --i) {
            for (int col = 0; col < NCOLS; ++col) {
                double value = local_rhs[i * NCOLS + col];
                for (int j = i + 1; j < NEL; ++j) {
                    value -= local_lu[i * NEL + j] * local_rhs[j * NCOLS + col];
                }
                local_rhs[i * NCOLS + col] = value / local_lu[i * NEL + i];
            }
        }

        // Emit Schur flux blocks.  Row-side orientation is encoded by
        // The row side uses the precomputed tau-weighted global-orientation lift.  Column-side orientation
        // uses the trace-basis transform selected at kernel compile time:
        // nodal bases reverse dof order, modal Legendre bases apply parity signs.
        for (int row_face = 0; row_face < 3; ++row_face) {
            const long long side_id = interior_side_index[element * 3 + row_face];
            const long long row_edge = loc2glob_edge[element * 3 + row_face];
            const long long row_solve_edge = edge_to_solve_edge[row_edge];
            if (side_id < 0 || row_solve_edge < 0) {
                continue;
            }
            const long long side_base = side_flux_offsets[side_id];
            for (int row_dof = 0; row_dof < NTR; ++row_dof) {
                double rhs_value = 0.0;
                for (int i = 0; i < NEL; ++i) {
                    const double lift = trace_lift[((element * 3 + row_face) * NTR + row_dof) * NEL + i];
                    rhs_value += lift * local_rhs[i * NCOLS + (NCOLS - 1)];
                }

                int col_block_pos = 0;
                for (int col_face = 0; col_face < 3; ++col_face) {
                    const long long col_edge = loc2glob_edge[element * 3 + col_face];
                    const long long col_solve_edge = edge_to_solve_edge[col_edge];
                    const bool positive = orientations[element * 3 + col_face];
                    for (int col_dof = 0; col_dof < NTR; ++col_dof) {
                        const int column = raw_trace_column_index(col_face, positive, col_dof);
                        const double trace_sign = raw_trace_orientation_sign(positive, col_dof);
                        double schur_value = 0.0;
                        for (int i = 0; i < NEL; ++i) {
                            const double lift = trace_lift[((element * 3 + row_face) * NTR + row_dof) * NEL + i];
                            schur_value += lift * local_rhs[i * NCOLS + column];
                        }
                        schur_value *= trace_sign;
                        if (col_solve_edge >= 0) {
                            const long long out = side_base + ((long long)col_block_pos * NTR + row_dof) * NTR + col_dof;
                            rows[out] = row_solve_edge * NTR + row_dof;
                            cols[out] = col_solve_edge * NTR + col_dof;
                            data[out] = -schur_value;
                        } else {
                            rhs_value += schur_value * boundary_trace[col_edge * NTR + col_dof];
                        }
                    }
                    if (col_solve_edge >= 0) {
                        col_block_pos += 1;
                    }
                }
                const long long mass_base = n_flux + side_id * NTR * NTR + row_dof * NTR;
                for (int col_dof = 0; col_dof < NTR; ++col_dof) {
                    const long long out = mass_base + col_dof;
                    rows[out] = row_solve_edge * NTR + row_dof;
                    cols[out] = row_solve_edge * NTR + col_dof;
                    data[out] = side_mass_blocks[side_id * NTR * NTR + row_dof * NTR + col_dof];
                }
                atomicAdd(&rhs[row_solve_edge * NTR + row_dof], rhs_value);
            }
        }
    }

}
"""


_RAW_ASSEMBLY_COOP_TEMPLATE = r"""
extern "C" __global__ void assemble_advection_raw_coop(
        long long* __restrict__ rows,
        long long* __restrict__ cols,
        double* __restrict__ data,
        double* __restrict__ rhs,
        const double* __restrict__ local_mats,
        const double* __restrict__ element_boundary,
        const double* __restrict__ source_rhs,
        const double* __restrict__ boundary_trace,
        const long long* __restrict__ loc2glob_edge,
        const bool* __restrict__ orientations,
        const long long* __restrict__ loc2oriented_face_coupling,
        const long long* __restrict__ interior_side_index,
        const long long* __restrict__ edge_to_solve_edge,
        const long long* __restrict__ int_edges,
        const long long* __restrict__ side_flux_offsets,
        const double* __restrict__ jacs_el_fc,
        const double* __restrict__ side_mass_blocks,
        const double* __restrict__ trace_lift,
        const long long num_elements,
        const long long num_int_edges,
        const long long n_flux)
{
    // Cooperative version: one CUDA block still owns one element, but all
    // threads in the block participate in the dense local work.  This is the
    // first step beyond the serial baseline.  Pivot selection remains on thread
    // 0 for simplicity; row swaps, eliminations, RHS solves, and COO emission
    // are spread across the block.
    extern __shared__ unsigned char shared_raw[];
    double* shared = reinterpret_cast<double*>(shared_raw);
    double* local_lu = shared;                         // NEL x NEL matrix, factorized in place
    double* local_rhs = local_lu + (NEL * NEL);        // NEL x NCOLS solved RHS columns
    int* pivots = reinterpret_cast<int*>(local_rhs + (NEL * NCOLS));

    const int tid = threadIdx.x;
    const long long element = blockIdx.x;

    if (element < num_elements) {
        // Cooperative copy into shared memory.  The RHS columns are laid out so
        // each column is an element-boundary trace column, except the last one
        // which is the element source column.
        for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
            const int i = idx / NEL;
            const int j = idx - i * NEL;
            local_lu[idx] = local_mats[(element * NEL + i) * NEL + j];
        }
        for (int idx = tid; idx < NEL * NCOLS; idx += blockDim.x) {
            const int i = idx / NCOLS;
            const int col = idx - i * NCOLS;
            local_rhs[idx] = (col == NCOLS - 1)
                ? source_rhs[element * NEL + i]
                : element_boundary[(element * NEL + i) * (3 * NTR) + col];
        }
        __syncthreads();

        // Partial-pivot LU.  Pivot search is serial for now because NEL <= 28;
        // the expensive trailing update is parallelized across the block.
        for (int k = 0; k < NEL; ++k) {
            if (tid == 0) {
                int pivot = k;
                double max_value = fabs(local_lu[k * NEL + k]);
                for (int i = k + 1; i < NEL; ++i) {
                    const double value = fabs(local_lu[i * NEL + k]);
                    if (value > max_value) {
                        max_value = value;
                        pivot = i;
                    }
                }
                pivots[k] = pivot;
            }
            __syncthreads();

            const int pivot = pivots[k];
            if (pivot != k) {
                for (int j = tid; j < NEL; j += blockDim.x) {
                    const double tmp = local_lu[k * NEL + j];
                    local_lu[k * NEL + j] = local_lu[pivot * NEL + j];
                    local_lu[pivot * NEL + j] = tmp;
                }
            }
            __syncthreads();

            if (tid == 0) {
                double diagonal = local_lu[k * NEL + k];
                if (fabs(diagonal) < 1.0e-30) {
                    diagonal = diagonal >= 0.0 ? 1.0e-30 : -1.0e-30;
                    local_lu[k * NEL + k] = diagonal;
                }
            }
            __syncthreads();

            const double diagonal = local_lu[k * NEL + k];
            for (int i = k + 1 + tid; i < NEL; i += blockDim.x) {
                local_lu[i * NEL + k] /= diagonal;
            }
            __syncthreads();

            const int width = NEL - k - 1;
            for (int idx = tid; idx < width * width; idx += blockDim.x) {
                const int i = k + 1 + idx / width;
                const int j = k + 1 + idx - (idx / width) * width;
                local_lu[i * NEL + j] -= local_lu[i * NEL + k] * local_lu[k * NEL + j];
            }
            __syncthreads();
        }

        // Apply the saved row permutations to all RHS columns.  Each k needs a
        // synchronization because later swaps depend on earlier row positions.
        for (int k = 0; k < NEL; ++k) {
            const int pivot = pivots[k];
            if (pivot != k) {
                for (int col = tid; col < NCOLS; col += blockDim.x) {
                    const double tmp = local_rhs[k * NCOLS + col];
                    local_rhs[k * NCOLS + col] = local_rhs[pivot * NCOLS + col];
                    local_rhs[pivot * NCOLS + col] = tmp;
                }
            }
            __syncthreads();
        }

        // Forward and backward triangular solves.  Rows are sequential, but RHS
        // columns are independent and distributed over the block.  For p=6 there
        // are 22 columns, so 32-thread blocks already cover this stage well.
        for (int i = 0; i < NEL; ++i) {
            for (int col = tid; col < NCOLS; col += blockDim.x) {
                double value = local_rhs[i * NCOLS + col];
                for (int j = 0; j < i; ++j) {
                    value -= local_lu[i * NEL + j] * local_rhs[j * NCOLS + col];
                }
                local_rhs[i * NCOLS + col] = value;
            }
            __syncthreads();
        }
        for (int i = NEL - 1; i >= 0; --i) {
            for (int col = tid; col < NCOLS; col += blockDim.x) {
                double value = local_rhs[i * NCOLS + col];
                for (int j = i + 1; j < NEL; ++j) {
                    value -= local_lu[i * NEL + j] * local_rhs[j * NCOLS + col];
                }
                local_rhs[i * NCOLS + col] = value / local_lu[i * NEL + i];
            }
            __syncthreads();
        }

        // Emit rows in parallel over the 3*NTR row test functions.  Each row
        // task owns all column blocks for that row, so matrix writes are unique.
        // RHS writes still use atomicAdd because two element sides can
        // contribute to the same global trace row.
        const int row_tasks = 3 * NTR;
        for (int task = tid; task < row_tasks; task += blockDim.x) {
            const int row_face = task / NTR;
            const int row_dof = task - row_face * NTR;
            const long long side_id = interior_side_index[element * 3 + row_face];
            const long long row_edge = loc2glob_edge[element * 3 + row_face];
            const long long row_solve_edge = edge_to_solve_edge[row_edge];
            if (side_id < 0 || row_solve_edge < 0) {
                continue;
            }

            const long long side_base = side_flux_offsets[side_id];

            // This thread owns one Schur row for this element side.  The same
            // tau-weighted lift vector is dotted with the solved source column
            // and then with every solved trace column.
            double lift_values[NEL];
            double rhs_value = 0.0;
            for (int i = 0; i < NEL; ++i) {
                const double lift = trace_lift[((element * 3 + row_face) * NTR + row_dof) * NEL + i];
                lift_values[i] = lift;
                rhs_value += lift * local_rhs[i * NCOLS + (NCOLS - 1)];
            }

            int col_block_pos = 0;
            for (int col_face = 0; col_face < 3; ++col_face) {
                const long long col_edge = loc2glob_edge[element * 3 + col_face];
                const long long col_solve_edge = edge_to_solve_edge[col_edge];
                const bool positive = orientations[element * 3 + col_face];
                for (int col_dof = 0; col_dof < NTR; ++col_dof) {
                    const int column = raw_trace_column_index(col_face, positive, col_dof);
                    const double trace_sign = raw_trace_orientation_sign(positive, col_dof);
                    double schur_value = 0.0;
                    for (int i = 0; i < NEL; ++i) {
                        schur_value += lift_values[i] * local_rhs[i * NCOLS + column];
                    }
                    schur_value *= trace_sign;
                    if (col_solve_edge >= 0) {
                        const long long out = side_base + ((long long)col_block_pos * NTR + row_dof) * NTR + col_dof;
                        rows[out] = row_solve_edge * NTR + row_dof;
                        cols[out] = col_solve_edge * NTR + col_dof;
                        data[out] = -schur_value;
                    } else {
                        rhs_value += schur_value * boundary_trace[col_edge * NTR + col_dof];
                    }
                }
                if (col_solve_edge >= 0) {
                    col_block_pos += 1;
                }
            }
            const long long mass_base = n_flux + side_id * NTR * NTR + row_dof * NTR;
            for (int col_dof = 0; col_dof < NTR; ++col_dof) {
                const long long out = mass_base + col_dof;
                rows[out] = row_solve_edge * NTR + row_dof;
                cols[out] = row_solve_edge * NTR + col_dof;
                data[out] = side_mass_blocks[side_id * NTR * NTR + row_dof * NTR + col_dof];
            }
            atomicAdd(&rhs[row_solve_edge * NTR + row_dof], rhs_value);
        }
    }

}
"""


_RAW_FUSED_TEMPLATE = r"""
// The fused-local raw path is intentionally close to
// hdgfem.transport.numba_kernels._assemble_projected_local_system.  All
// coefficient fields have already been projected into the element basis before
// this kernel is launched.  The kernel therefore only needs small reference
// tensors and per-element geometry, and no Python/CuPy dense local tensors are
// materialized.

#ifndef RAW_LU_MODE_COOP
#define RAW_LU_MODE_COOP 0
#endif
#ifndef RAW_LU_SCRATCH_THREADS
#define RAW_LU_SCRATCH_THREADS 128
#endif

__device__ __forceinline__ double raw_side_normal(
        const double* beta_coeffs, const double* face_basis, const double* normals,
        const long long num_elements, const long long side, const int qf)
{
    const long long element = side / 3;
    const int face = side % 3;
    double bx = 0.0, by = 0.0;
    for (int k = 0; k < NEL; ++k) {
        const double phi = face_basis[(face * NEL + k) * NQF + qf];
        bx += beta_coeffs[element * NEL + k] * phi;
        by += beta_coeffs[(num_elements + element) * NEL + k] * phi;
    }
    return bx * normals[side * 2] + by * normals[side * 2 + 1];
}

__device__ __forceinline__ bool raw_inactive_face_gauge(
        const double* tau_face, const double* beta_coeffs, const double* face_basis,
        const double* normals, const long long* edge_side_indices,
        const long long num_elements, const long long element, const int face,
        const long long edge)
{
#if RAW_CONFLICT_AVERAGED_UPWIND
    if (edge_side_indices[edge * 2] != element * 3 + face) return false;
    const long long other = edge_side_indices[edge * 2 + 1];
    if (other < 0) return false;
    for (int q = 0; q < NQF; ++q)
        if (tau_face[face * NQF + q] != 0.0) return false;
    // If our effective velocity is zero, the pair is inactive exactly when
    // the other raw velocity is nonnegative (equal outflow or two zeros).
    // Read immutable coefficients, never another block's weights.
    for (int q = 0; q < NQF; ++q)
        if (raw_side_normal(beta_coeffs, face_basis, normals, num_elements, other, q) < 0.0)
            return false;
    return true;
#else
    return false;
#endif
}

__device__ __forceinline__ void assemble_projected_local_advection_raw(
        double* __restrict__ local_lu,
        double* __restrict__ local_rhs,
        double* __restrict__ tau_face,
        double* __restrict__ gamma_face,
        double* __restrict__ source_cache,
        double* __restrict__ beta_x_cache,
        double* __restrict__ beta_y_cache,
        double* __restrict__ beta_ref0_cache,
        double* __restrict__ beta_ref1_cache,
        double* __restrict__ reaction_cache,
        const double* __restrict__ aff_jacs,
        const double* __restrict__ inv_aff_mats_t,
        const double* __restrict__ jacs_el_fc,
        const double* __restrict__ normals,
        const double* __restrict__ mass_matrix,
        const double* __restrict__ reaction_triples,
        const double* __restrict__ advection_tensor,
        const int* __restrict__ advection_sparse_offsets,
        const int* __restrict__ advection_sparse_modes,
        const double* __restrict__ advection_sparse_values0,
        const double* __restrict__ advection_sparse_values1,
        const int use_sparse_advection,
        const int mass_is_diagonal,
        const double* __restrict__ face_basis,
        const double* __restrict__ face_weights,
        const double* __restrict__ trace_basis,
        const double* __restrict__ source_coeffs,
        const double* __restrict__ beta_coeffs,
        const double* __restrict__ reaction_coeffs,
        const double reaction_scalar,
        const int reaction_is_scalar,
        const long long* __restrict__ loc2glob_edge,
        const long long* __restrict__ edge_side_indices,
        const bool* __restrict__ orientations,
        const long long* __restrict__ edge_to_solve_edge,
        const int zero_boundary_flux,
        const long long num_elements,
        const long long element)
{
    const int tid = threadIdx.x;
    const double jac = aff_jacs[element];
    const double inv_t00 = inv_aff_mats_t[(element * 2 + 0) * 2 + 0];
    const double inv_t01 = inv_aff_mats_t[(element * 2 + 0) * 2 + 1];
    const double inv_t10 = inv_aff_mats_t[(element * 2 + 1) * 2 + 0];
    const double inv_t11 = inv_aff_mats_t[(element * 2 + 1) * 2 + 1];

    // ---------------------------------------------------------------------
    // Per-element coefficient cache
    // ---------------------------------------------------------------------
    // Coefficients arrive in global arrays shaped by field, element, and local
    // basis index.  The fused assembly needs the same beta/source entries in
    // several independent blocks below:
    //
    //   * tau_face:      |beta.n| on every face quadrature node, including
    //                    discontinuous fluxes when beta is DG-elementwise
    //                    discontinuous, premultiplied by face_jac * w_q.
    //   * gamma_face:    tau - beta.n on every face quadrature node, with the
    //                    same face_jac * w_q factor.
    //   * source column: M * source_coeffs.
    //   * volume matrix: beta . grad(phi_i) for every (i, j),
    //   * reaction mass: optional projected reaction field.
    //
    // Pulling these short vectors into shared memory once per element removes a
    // large number of repeated global loads.  The transformed beta caches also
    // fold the affine inverse into the coefficient vector:
    //
    //   beta_ref0[k] = beta_x[k] * inv_t00 + beta_y[k] * inv_t10
    //   beta_ref1[k] = beta_x[k] * inv_t01 + beta_y[k] * inv_t11
    //
    // so the inner volume loop only does two coefficient/tensor products per k
    // instead of rereading beta and remultiplying geometry for every matrix
    // entry.
    for (int k = tid; k < NEL; k += blockDim.x) {
        const double beta_x = beta_coeffs[((0 * num_elements + element) * NEL) + k];
        const double beta_y = beta_coeffs[((1 * num_elements + element) * NEL) + k];
        source_cache[k] = source_coeffs[element * NEL + k];
        beta_x_cache[k] = beta_x;
        beta_y_cache[k] = beta_y;
        beta_ref0_cache[k] = beta_x * inv_t00 + beta_y * inv_t10;
        beta_ref1_cache[k] = beta_x * inv_t01 + beta_y * inv_t11;
        if (!reaction_is_scalar) {
            reaction_cache[k] = reaction_coeffs[element * NEL + k];
        }
    }
    __syncthreads();

    // Reset every local RHS column.  Columns 0..3*NTR-1 are local trace columns;
    // the last column is the projected source moment.  These columns will be
    // overwritten by the local solve in-place, so they never leave shared memory.
    for (int idx = tid; idx < NEL * NCOLS; idx += blockDim.x) {
        local_rhs[idx] = 0.0;
    }

    // Precompute per-side trace weights on each local face quadrature point once
    // per element.
    // This removes the worst repeated coefficient evaluation from boundary mass
    // and trace RHS construction while keeping the data in shared memory.
    for (int idx = tid; idx < 3 * NQF; idx += blockDim.x) {
        const int face = idx / NQF;
        const int qf = idx - face * NQF;
        double beta_x = 0.0;
        double beta_y = 0.0;
        for (int k = 0; k < NEL; ++k) {
            const double phi = face_basis[(face * NEL + k) * NQF + qf];
            beta_x += beta_x_cache[k] * phi;
            beta_y += beta_y_cache[k] * phi;
        }
        const double normal_x = normals[(element * 3 + face) * 2 + 0];
        const double normal_y = normals[(element * 3 + face) * 2 + 1];
        double normal_flux = beta_x * normal_x + beta_y * normal_y;
#if RAW_CONFLICT_AVERAGED_UPWIND
        const long long side = element * 3 + face;
        const long long edge = loc2glob_edge[side];
        const long long left = edge_side_indices[edge * 2];
        const long long right = edge_side_indices[edge * 2 + 1];
        const long long other = left == side ? right : left;
        if (other >= 0) {
            const int other_q = orientations[side] == orientations[other] ? qf : NQF - 1 - qf;
            const double neighbor = raw_side_normal(beta_coeffs, face_basis, normals,
                                                    num_elements, other, other_q);
            if (normal_flux >= 0.0 && neighbor >= 0.0 && normal_flux + neighbor > 0.0)
                normal_flux = (normal_flux - neighbor) * 0.5;
        }
#endif
        double tau = RAW_ADVECTION_TAU_FACTOR * fabs(normal_flux);
        double gamma = tau - normal_flux;
        if (zero_boundary_flux) {
            const long long edge = loc2glob_edge[element * 3 + face];
            if (edge_to_solve_edge[edge] < 0) {
                tau = 0.0;
                gamma = 0.0;
            }
        }
        // Store quadrature-weighted values: every consumer integrates
        // face_jac * w_q * (tau or gamma), and a zero tau stays zero.
        const double face_measure = jacs_el_fc[element * 3 + face] * face_weights[qf];
        tau_face[idx] = tau * face_measure;
        gamma_face[idx] = gamma * face_measure;
    }
    __syncthreads();
    // Source moments use the projected source coefficient vector: J * M * f_h.
    // The index order mass_matrix[k, i] intentionally matches the Numba fused
    // implementation, even though M is symmetric for the usual bases.
    const int source_col = 3 * NTR;
    for (int i = tid; i < NEL; i += blockDim.x) {
        double source_value = 0.0;
        if (mass_is_diagonal) {
            source_value = source_cache[i] * mass_matrix[i * NEL + i];
        } else {
            for (int k = 0; k < NEL; ++k) {
                source_value += source_cache[k] * mass_matrix[k * NEL + i];
            }
        }
        local_rhs[i * NCOLS + source_col] = jac * source_value;
    }

    // Volume operator: reaction mass minus beta . grad contribution.  Reaction
    // can be a scalar or a projected element field.  The advection tensor stores
    // int phi_k * phi_j * grad_D(phi_i), shaped D,k,i,j.
    for (int idx = tid; idx < NEL * NEL; idx += blockDim.x) {
        const int i = idx / NEL;
        const int j = idx - i * NEL;
        double value;
        if (reaction_is_scalar) {
            value = (mass_is_diagonal && i != j)
                ? 0.0 : reaction_scalar * jac * mass_matrix[i * NEL + j];
        } else {
            double reaction_value = 0.0;
            for (int k = 0; k < NEL; ++k) {
                reaction_value += reaction_cache[k] * reaction_triples[(k * NEL + i) * NEL + j];
            }
            value = jac * reaction_value;
        }

        double advection_value = 0.0;
        if (use_sparse_advection) {
            const int entry = i * NEL + j;
            for (int pos = advection_sparse_offsets[entry];
                    pos < advection_sparse_offsets[entry + 1]; ++pos) {
                const int k = advection_sparse_modes[pos];
                advection_value +=
                    beta_ref0_cache[k] * advection_sparse_values0[pos]
                    + beta_ref1_cache[k] * advection_sparse_values1[pos];
            }
        } else {
            for (int k = 0; k < NEL; ++k) {
                const double adv0 = advection_tensor[((0 * NEL + k) * NEL + i) * NEL + j];
                const double adv1 = advection_tensor[((1 * NEL + k) * NEL + i) * NEL + j];
                advection_value += beta_ref0_cache[k] * adv0 + beta_ref1_cache[k] * adv1;
            }
        }
        local_lu[idx] = value - jac * advection_value;
    }
    __syncthreads();

    // Upwind boundary mass: int_{faces} |beta.n| phi_i phi_j.  The matrix is
    // symmetric, so each thread owns one upper-triangle entry and writes it and
    // its mirror; no atomics are needed inside the shared local matrix.  This
    // halves the FP64 work, which bounds this kernel on FP64-limited devices.
    for (int idx = tid; idx < NEL * (NEL + 1) / 2; idx += blockDim.x) {
        int i = 0;
        int j = idx;
        while (j >= NEL - i) {
            j -= NEL - i;
            ++i;
        }
        j += i;
        double value = 0.0;
        for (int face = 0; face < 3; ++face) {
            const double* phi_i = face_basis + (face * NEL + i) * NQF;
            const double* phi_j = face_basis + (face * NEL + j) * NQF;
            for (int qf = 0; qf < NQF; ++qf)
                value = fma(tau_face[face * NQF + qf] * phi_i[qf], phi_j[qf], value);
        }
        local_lu[i * NEL + j] += value;
        if (i != j)
            local_lu[j * NEL + i] += value;
    }

    // Local trace RHS columns: int_{face} (|beta.n| - beta.n) phi_i mu_j.  The
    // column basis is stored in local trace orientation.  Global edge orientation
    // is handled later when solved columns are emitted or when reconstruction
    // multiplies by the full trace vector.
    for (int idx = tid; idx < NEL * 3 * NTR; idx += blockDim.x) {
        const int i = idx / (3 * NTR);
        const int col = idx - i * (3 * NTR);
        const int face = col / NTR;
        const int trace_dof = col - face * NTR;
        double value = 0.0;
        for (int qf = 0; qf < NQF; ++qf) {
            const double phi_i = face_basis[(face * NEL + i) * NQF + qf];
            const double mu_j = trace_basis[trace_dof * NQF + qf];
            value = fma(gamma_face[face * NQF + qf] * phi_i, mu_j, value);
        }
        local_rhs[i * NCOLS + col] = value;
    }
    __syncthreads();
}

__device__ __forceinline__ void factor_local_lu_raw(
        double* __restrict__ local_lu,
        int* __restrict__ pivots)
{
    const int tid = threadIdx.x;
    for (int k = 0; k < NEL; ++k) {
        if (tid == 0) {
            int pivot = k;
            double max_value = fabs(local_lu[k * NEL + k]);
            for (int i = k + 1; i < NEL; ++i) {
                const double value = fabs(local_lu[i * NEL + k]);
                if (value > max_value) {
                    max_value = value;
                    pivot = i;
                }
            }
            pivots[k] = pivot;
        }
        __syncthreads();

        const int pivot = pivots[k];
        if (pivot != k) {
            for (int j = tid; j < NEL; j += blockDim.x) {
                const double tmp = local_lu[k * NEL + j];
                local_lu[k * NEL + j] = local_lu[pivot * NEL + j];
                local_lu[pivot * NEL + j] = tmp;
            }
        }
        __syncthreads();

        if (tid == 0) {
            double diagonal = local_lu[k * NEL + k];
            if (fabs(diagonal) < 1.0e-30) {
                diagonal = diagonal >= 0.0 ? 1.0e-30 : -1.0e-30;
                local_lu[k * NEL + k] = diagonal;
            }
        }
        __syncthreads();

        const double diagonal = local_lu[k * NEL + k];
        for (int i = k + 1 + tid; i < NEL; i += blockDim.x) {
            local_lu[i * NEL + k] /= diagonal;
        }
        __syncthreads();

        const int width = NEL - k - 1;
        for (int idx = tid; idx < width * width; idx += blockDim.x) {
            const int i = k + 1 + idx / width;
            const int j = k + 1 + idx - (idx / width) * width;
            local_lu[i * NEL + j] -= local_lu[i * NEL + k] * local_lu[k * NEL + j];
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void solve_all_columns_raw(
        const double* __restrict__ local_lu,
        const int* __restrict__ pivots,
        double* __restrict__ local_rhs)
{
    const int tid = threadIdx.x;
    for (int k = 0; k < NEL; ++k) {
        const int pivot = pivots[k];
        if (pivot != k) {
            for (int col = tid; col < NCOLS; col += blockDim.x) {
                const double tmp = local_rhs[k * NCOLS + col];
                local_rhs[k * NCOLS + col] = local_rhs[pivot * NCOLS + col];
                local_rhs[pivot * NCOLS + col] = tmp;
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < NEL; ++i) {
        for (int col = tid; col < NCOLS; col += blockDim.x) {
            double value = local_rhs[i * NCOLS + col];
            for (int j = 0; j < i; ++j) {
                value -= local_lu[i * NEL + j] * local_rhs[j * NCOLS + col];
            }
            local_rhs[i * NCOLS + col] = value;
        }
        __syncthreads();
    }
    for (int i = NEL - 1; i >= 0; --i) {
        for (int col = tid; col < NCOLS; col += blockDim.x) {
            double value = local_rhs[i * NCOLS + col];
            for (int j = i + 1; j < NEL; ++j) {
                value -= local_lu[i * NEL + j] * local_rhs[j * NCOLS + col];
            }
            local_rhs[i * NCOLS + col] = value / local_lu[i * NEL + i];
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void solve_column_zero_raw(
        const double* __restrict__ local_lu,
        const int* __restrict__ pivots,
        double* __restrict__ local_rhs)
{
    const int tid = threadIdx.x;
    for (int k = 0; k < NEL; ++k) {
        const int pivot = pivots[k];
        if (pivot != k && tid == 0) {
            const double tmp = local_rhs[k * NCOLS];
            local_rhs[k * NCOLS] = local_rhs[pivot * NCOLS];
            local_rhs[pivot * NCOLS] = tmp;
        }
        __syncthreads();
    }

    for (int i = 0; i < NEL; ++i) {
        if (tid == 0) {
            double value = local_rhs[i * NCOLS];
            for (int j = 0; j < i; ++j) {
                value -= local_lu[i * NEL + j] * local_rhs[j * NCOLS];
            }
            local_rhs[i * NCOLS] = value;
        }
        __syncthreads();
    }
    for (int i = NEL - 1; i >= 0; --i) {
        if (tid == 0) {
            double value = local_rhs[i * NCOLS];
            for (int j = i + 1; j < NEL; ++j) {
                value -= local_lu[i * NEL + j] * local_rhs[j * NCOLS];
            }
            local_rhs[i * NCOLS] = value / local_lu[i * NEL + i];
        }
        __syncthreads();
    }
}


""" + RAW_COOP_LU_FACTOR + r"""__device__ __forceinline__ void factor_local_lu_coop_pivot_scale_raw(
        double* __restrict__ local_lu,
        int* __restrict__ pivots,
        double* __restrict__ pivot_abs_values,
        int* __restrict__ pivot_rows)
{
    // Experimental cooperative LU for the fused raw kernel.  The safe LU
    // above remains the default because the earlier aggressive helper exposed
    // intermittent value corruption in the fused path.  This version makes the
    // synchronization contract explicit for every pivot step:
    //
    //   1. all threads scan a strided subset of the active pivot column,
    //   2. a shared-memory reduction publishes one deterministic pivot row,
    //   3. thread 0 performs the selected row swap, preserving the stable handoff,
    //   4. thread 0 clamps a tiny diagonal exactly as the safe helper does,
    //   5. the multiplier column and trailing Schur update are parallelized.
    //
    // pivot_abs_values and pivot_rows are block-sized scratch arrays.  Python
    // currently limits raw block sizes to 1, 32, 64, or 128, and the shared
    // workspace reserves RAW_LU_SCRATCH_THREADS entries so tid-indexed scratch
    // writes stay in bounds.
    const int tid = threadIdx.x;
    for (int k = 0; k < NEL; ++k) {
        double local_max = -1.0;
        int local_pivot = k;
        for (int i = k + tid; i < NEL; i += blockDim.x) {
            const double value = fabs(local_lu[i * NEL + k]);
            if (value > local_max || (value == local_max && i < local_pivot)) {
                local_max = value;
                local_pivot = i;
            }
        }
        pivot_abs_values[tid] = local_max;
        pivot_rows[tid] = local_pivot;
        __syncthreads();

        // blockDim.x is constrained to powers of two plus the serial value 1.
        // Ties choose the lower row index, matching the deterministic serial
        // search order whenever equal absolute pivot values appear.
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const double candidate_value = pivot_abs_values[tid + stride];
                const int candidate_row = pivot_rows[tid + stride];
                const double current_value = pivot_abs_values[tid];
                const int current_row = pivot_rows[tid];
                if (candidate_value > current_value ||
                        (candidate_value == current_value && candidate_row < current_row)) {
                    pivot_abs_values[tid] = candidate_value;
                    pivot_rows[tid] = candidate_row;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            const int pivot = pivot_rows[0];
            pivots[k] = pivot;
            // Keep row swaps serialized.  Parallel row swaps, including a staged
            // scratch variant, reproduced the historical illegal-address failure
            // in the fully fused kernel.  The surrounding cooperative pivot
            // search, multiplier scaling, and trailing update still move the
            // expensive LU work onto the block while preserving this handoff.
            if (pivot != k) {
                for (int j = 0; j < NEL; ++j) {
                    const double tmp = local_lu[k * NEL + j];
                    local_lu[k * NEL + j] = local_lu[pivot * NEL + j];
                    local_lu[pivot * NEL + j] = tmp;
                }
            }

            double diagonal = local_lu[k * NEL + k];
            if (fabs(diagonal) < 1.0e-30) {
                diagonal = diagonal >= 0.0 ? 1.0e-30 : -1.0e-30;
                local_lu[k * NEL + k] = diagonal;
            }
        }
        // Publish the stored pivot for the later triangular solve and the
        // completed row swap/diagonal clamp for the multiplier update.
        __syncthreads();

        const double diagonal = local_lu[k * NEL + k];
        for (int i = k + 1 + tid; i < NEL; i += blockDim.x) {
            local_lu[i * NEL + k] /= diagonal;
        }
        __syncthreads();

        const int width = NEL - k - 1;
        for (int idx = tid; idx < width * width; idx += blockDim.x) {
            const int i = k + 1 + idx / width;
            const int j = k + 1 + idx - (idx / width) * width;
            local_lu[i * NEL + j] -= local_lu[i * NEL + k] * local_lu[k * NEL + j];
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void factor_local_lu_serial_raw(
        double* __restrict__ local_lu,
        int* __restrict__ pivots)
{
    // Correctness baseline for the full fused kernel: the local operator was
    // assembled cooperatively, but the small dense LU is currently performed by
    // one thread.  The earlier cooperative LU/triangular-solve path is kept in
    // this file for the next optimization pass, but the serial factor/solve avoids
    // the value corruption seen on larger meshes while still eliminating global
    // local_mats/element_boundary/source_rhs tensors.
    if (threadIdx.x == 0) {
        for (int k = 0; k < NEL; ++k) {
            int pivot = k;
            double max_value = fabs(local_lu[k * NEL + k]);
            for (int i = k + 1; i < NEL; ++i) {
                const double value = fabs(local_lu[i * NEL + k]);
                if (value > max_value) {
                    max_value = value;
                    pivot = i;
                }
            }
            pivots[k] = pivot;
            if (pivot != k) {
                for (int j = 0; j < NEL; ++j) {
                    const double tmp = local_lu[k * NEL + j];
                    local_lu[k * NEL + j] = local_lu[pivot * NEL + j];
                    local_lu[pivot * NEL + j] = tmp;
                }
            }
            double diagonal = local_lu[k * NEL + k];
            if (fabs(diagonal) < 1.0e-30) {
                diagonal = diagonal >= 0.0 ? 1.0e-30 : -1.0e-30;
                local_lu[k * NEL + k] = diagonal;
            }
            for (int i = k + 1; i < NEL; ++i) {
                local_lu[i * NEL + k] /= diagonal;
                const double multiplier = local_lu[i * NEL + k];
                for (int j = k + 1; j < NEL; ++j) {
                    local_lu[i * NEL + j] -= multiplier * local_lu[k * NEL + j];
                }
            }
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void solve_all_columns_serial_raw(
        const double* __restrict__ local_lu,
        const int* __restrict__ pivots,
        double* __restrict__ local_rhs)
{
    if (threadIdx.x == 0) {
        for (int k = 0; k < NEL; ++k) {
            const int pivot = pivots[k];
            if (pivot != k) {
                for (int col = 0; col < NCOLS; ++col) {
                    const double tmp = local_rhs[k * NCOLS + col];
                    local_rhs[k * NCOLS + col] = local_rhs[pivot * NCOLS + col];
                    local_rhs[pivot * NCOLS + col] = tmp;
                }
            }
        }
        for (int i = 0; i < NEL; ++i) {
            for (int col = 0; col < NCOLS; ++col) {
                double value = local_rhs[i * NCOLS + col];
                for (int j = 0; j < i; ++j) {
                    value -= local_lu[i * NEL + j] * local_rhs[j * NCOLS + col];
                }
                local_rhs[i * NCOLS + col] = value;
            }
        }
        for (int i = NEL - 1; i >= 0; --i) {
            for (int col = 0; col < NCOLS; ++col) {
                double value = local_rhs[i * NCOLS + col];
                for (int j = i + 1; j < NEL; ++j) {
                    value -= local_lu[i * NEL + j] * local_rhs[j * NCOLS + col];
                }
                local_rhs[i * NCOLS + col] = value / local_lu[i * NEL + i];
            }
        }
    }
    __syncthreads();
}

extern "C" __global__ void assemble_advection_raw_fused(
        long long* __restrict__ rows,
        long long* __restrict__ cols,
        double* __restrict__ data,
        double* __restrict__ rhs,
        double* __restrict__ local_response,
        double* __restrict__ cached_lu,
        int* __restrict__ cached_pivots,
        double* __restrict__ cached_lift,
        const int cache_factors,
        const long long* __restrict__ loc2glob_edge,
        const long long* __restrict__ edge_side_indices,
        const bool* __restrict__ orientations,
        const long long* __restrict__ loc2oriented_face_coupling,
        const long long* __restrict__ interior_side_index,
        const long long* __restrict__ edge_to_solve_edge,
        const long long* __restrict__ int_edges,
        const long long* __restrict__ side_flux_offsets,
        const double* __restrict__ aff_jacs,
        const double* __restrict__ inv_aff_mats_t,
        const double* __restrict__ jacs_el_fc,
        const double* __restrict__ edge_jacs,
        const double* __restrict__ normals,
        const double* __restrict__ mass_matrix,
        const double* __restrict__ reaction_triples,
        const double* __restrict__ advection_tensor,
        const int* __restrict__ advection_sparse_offsets,
        const int* __restrict__ advection_sparse_modes,
        const double* __restrict__ advection_sparse_values0,
        const double* __restrict__ advection_sparse_values1,
        const int use_sparse_advection,
        const int mass_is_diagonal,
        const double* __restrict__ face_basis,
        const double* __restrict__ face_weights,
        const double* __restrict__ trace_basis,
        const double* __restrict__ edge_mass,
        const double* __restrict__ oriented_lifts,
        const double* __restrict__ source_coeffs,
        const double* __restrict__ beta_coeffs,
        const double* __restrict__ reaction_coeffs,
        const double* __restrict__ boundary_trace,
        const double reaction_scalar,
        const int reaction_is_scalar,
        const int store_local_response,
        const int zero_boundary_flux,
        const long long num_elements,
        const long long num_int_edges,
        const long long n_flux)
{
    extern __shared__ unsigned char shared_raw[];
    double* shared = reinterpret_cast<double*>(shared_raw);
    double* local_lu = shared;                         // NEL x NEL local operator, factored in place
    double* local_rhs = local_lu + (NEL * NEL);        // NEL x (3*NTR+1) trace/source RHS columns
    double* tau_face = local_rhs + (NEL * NCOLS);      // 3 x NQF trace tau values
    double* gamma_face = tau_face + (3 * NQF);         // 3 x NQF trace gamma values
    double* source_cache = gamma_face + (3 * NQF);     // NEL projected source coefficients
    double* beta_x_cache = source_cache + NEL;         // NEL projected beta_x coefficients
    double* beta_y_cache = beta_x_cache + NEL;         // NEL projected beta_y coefficients
    double* beta_ref0_cache = beta_y_cache + NEL;      // NEL beta coefficients transformed by inv_aff_mats_t column 0
    double* beta_ref1_cache = beta_ref0_cache + NEL;   // NEL beta coefficients transformed by inv_aff_mats_t column 1
    double* reaction_cache = beta_ref1_cache + NEL;    // NEL projected reaction coefficients when reaction is nonscalar
    double* schur_lift_rows = reaction_cache + NEL;    // p>=8: 3*NTR x NEL spill-free Schur lifts
#if NEL >= 45
    double* post_lift_workspace = schur_lift_rows + (3 * NTR * NEL);
#else
    double* post_lift_workspace = schur_lift_rows;
#endif
#if RAW_LU_MODE_COOP
    double* lu_pivot_abs = post_lift_workspace;        // block scratch for cooperative pivot reduction
    int* pivots = reinterpret_cast<int*>(lu_pivot_abs + RAW_LU_SCRATCH_THREADS);
    int* lu_pivot_rows = pivots + NEL;                 // block scratch for cooperative pivot row reduction
#else
    int* pivots = reinterpret_cast<int*>(post_lift_workspace);
#endif

    const int tid = threadIdx.x;
    const long long element = blockIdx.x;

    if (element < num_elements) {
        assemble_projected_local_advection_raw(
            local_lu, local_rhs, tau_face, gamma_face,
            source_cache, beta_x_cache, beta_y_cache,
            beta_ref0_cache, beta_ref1_cache, reaction_cache,
            aff_jacs, inv_aff_mats_t, jacs_el_fc, normals,
            mass_matrix, reaction_triples, advection_tensor,
            advection_sparse_offsets, advection_sparse_modes,
            advection_sparse_values0, advection_sparse_values1,
            use_sparse_advection, mass_is_diagonal,
            face_basis, face_weights, trace_basis,
            source_coeffs, beta_coeffs, reaction_coeffs,
            reaction_scalar, reaction_is_scalar, loc2glob_edge, edge_side_indices, orientations, edge_to_solve_edge,
            zero_boundary_flux, num_elements, element);
#if RAW_LU_MODE_COOP
        factor_local_lu_coop_pivot_scale_raw(local_lu, pivots, lu_pivot_abs, lu_pivot_rows);
#else
        factor_local_lu_coop_raw(local_lu, pivots);
#endif
        solve_all_columns_raw(local_lu, pivots, local_rhs);
        if (cache_factors) {
            for (int entry = tid; entry < NEL * NEL; entry += blockDim.x)
                cached_lu[element * NEL * NEL + entry] = local_lu[entry];
            for (int entry = tid; entry < NEL; entry += blockDim.x)
                cached_pivots[element * NEL + entry] = pivots[entry];
        }

        if (store_local_response) {
            for (int entry = tid; entry < NEL * NCOLS; entry += blockDim.x) {
                local_response[element * (NEL * NCOLS) + entry] = local_rhs[entry];
            }
        }
        __syncthreads();

        // Emit one row task per local trace test function.  Row-side orientation
        // is encoded by loc2oriented_face_coupling and oriented_lifts; column-side
        // orientation uses the compile-time trace-basis transform.
        for (int task = tid; task < 3 * NTR; task += blockDim.x) {
            const int row_face = task / NTR;
            const int row_dof = task - row_face * NTR;
            const long long side_id = interior_side_index[element * 3 + row_face];
            const long long row_edge = loc2glob_edge[element * 3 + row_face];
            const long long row_solve_edge = edge_to_solve_edge[row_edge];
            if (side_id < 0 || row_solve_edge < 0) {
                continue;
            }

            const bool gauge_face = raw_inactive_face_gauge(
                tau_face, beta_coeffs, face_basis, normals, edge_side_indices,
                num_elements, element, row_face, row_edge);
            const long long side_base = side_flux_offsets[side_id];
            const bool row_positive_lift = orientations[element * 3 + row_face];
            const int row_local_lift_dof = raw_local_trace_dof(row_positive_lift, row_dof);
            const double row_lift_sign = raw_trace_orientation_sign(row_positive_lift, row_dof);

            // This thread owns one Schur row for this element side.  Build the
            // tau-weighted global-orientation lift on the fly from face
            // quadrature values, matching the NumPy/CUDA precomputed lift.
#if NEL < 45
            double lift_values[NEL];
#endif
            double rhs_value = 0.0;
            for (int i = 0; i < NEL; ++i) {
                double lift = 0.0;
                for (int qf = 0; qf < NQF; ++qf) {
                    const double mu = trace_basis[row_local_lift_dof * NQF + qf];
                    const double phi_i = face_basis[(row_face * NEL + i) * NQF + qf];
                    lift = fma(tau_face[row_face * NQF + qf] * mu, phi_i, lift);
                }
                lift *= row_lift_sign;
#if NEL >= 45
                schur_lift_rows[task * NEL + i] = lift;
#else
                lift_values[i] = lift;
#endif
                if (cache_factors)
                    cached_lift[(element * 3 * NTR + task) * NEL + i] = lift;
                rhs_value += lift * local_rhs[i * NCOLS + (NCOLS - 1)];
            }

            int col_block_pos = 0;
            for (int col_face = 0; col_face < 3; ++col_face) {
                const long long col_edge = loc2glob_edge[element * 3 + col_face];
                const long long col_solve_edge = edge_to_solve_edge[col_edge];
                const bool positive = orientations[element * 3 + col_face];
                for (int col_dof = 0; col_dof < NTR; ++col_dof) {
                    const int column = raw_trace_column_index(col_face, positive, col_dof);
                    const double trace_sign = raw_trace_orientation_sign(positive, col_dof);
                    double schur_value = 0.0;
                    for (int i = 0; i < NEL; ++i) {
#if NEL >= 45
                        schur_value += schur_lift_rows[task * NEL + i]
                            * local_rhs[i * NCOLS + column];
#else
                        schur_value += lift_values[i] * local_rhs[i * NCOLS + column];
#endif
                    }
                    schur_value *= trace_sign;
                    if (col_solve_edge >= 0) {
                        const long long out = side_base + ((long long)col_block_pos * NTR + row_dof) * NTR + col_dof;
                        rows[out] = row_solve_edge * NTR + row_dof;
                        cols[out] = col_solve_edge * NTR + col_dof;
                        data[out] = -schur_value;
                    } else {
                        rhs_value += schur_value * boundary_trace[col_edge * NTR + col_dof];
                    }
                }
                if (col_solve_edge >= 0) {
                    col_block_pos += 1;
                }
            }
            const bool row_positive_mass = orientations[element * 3 + row_face];
            const int row_local_mass_dof = raw_local_trace_dof(row_positive_mass, row_dof);
            const double row_mass_sign = raw_trace_orientation_sign(row_positive_mass, row_dof);
            const long long mass_base = n_flux + side_id * NTR * NTR + row_dof * NTR;
            for (int col_dof = 0; col_dof < NTR; ++col_dof) {
                const int col_local_mass_dof = raw_local_trace_dof(row_positive_mass, col_dof);
                const double col_mass_sign = raw_trace_orientation_sign(row_positive_mass, col_dof);
                double mass_value = (gauge_face && row_dof == col_dof) ? 1.0 : 0.0;
                for (int qf = 0; qf < NQF; ++qf) {
                    const double mu_row = row_mass_sign * trace_basis[row_local_mass_dof * NQF + qf];
                    const double mu_col = col_mass_sign * trace_basis[col_local_mass_dof * NQF + qf];
                    mass_value = fma(gamma_face[row_face * NQF + qf] * mu_row, mu_col, mass_value);
                }
                const long long out = mass_base + col_dof;
                rows[out] = row_solve_edge * NTR + row_dof;
                cols[out] = row_solve_edge * NTR + col_dof;
                data[out] = mass_value;
            }
            atomicAdd(&rhs[row_solve_edge * NTR + row_dof], rhs_value);
        }
    }

}

extern "C" __global__ void reconstruct_advection_raw_fused(
        double* __restrict__ uh,
        const double* __restrict__ trace,
        const long long* __restrict__ loc2glob_edge,
        const long long* __restrict__ edge_side_indices,
        const long long* __restrict__ edge_to_solve_edge,
        const bool* __restrict__ orientations,
        const double* __restrict__ aff_jacs,
        const double* __restrict__ inv_aff_mats_t,
        const double* __restrict__ jacs_el_fc,
        const double* __restrict__ normals,
        const double* __restrict__ mass_matrix,
        const double* __restrict__ reaction_triples,
        const double* __restrict__ advection_tensor,
        const int* __restrict__ advection_sparse_offsets,
        const int* __restrict__ advection_sparse_modes,
        const double* __restrict__ advection_sparse_values0,
        const double* __restrict__ advection_sparse_values1,
        const int use_sparse_advection,
        const int mass_is_diagonal,
        const double* __restrict__ face_basis,
        const double* __restrict__ face_weights,
        const double* __restrict__ trace_basis,
        const double* __restrict__ source_coeffs,
        const double* __restrict__ beta_coeffs,
        const double* __restrict__ reaction_coeffs,
        const double reaction_scalar,
        const int reaction_is_scalar,
        const int zero_boundary_flux,
        const long long num_elements)
{
    extern __shared__ unsigned char shared_raw[];
    double* shared = reinterpret_cast<double*>(shared_raw);
    double* local_lu = shared;                         // NEL x NEL local operator, factored in place
    double* local_rhs = local_lu + (NEL * NEL);        // NEL x (3*NTR+1) trace/source RHS columns
    double* tau_face = local_rhs + (NEL * NCOLS);      // 3 x NQF trace tau values
    double* gamma_face = tau_face + (3 * NQF);         // 3 x NQF trace gamma values
    double* source_cache = gamma_face + (3 * NQF);     // NEL projected source coefficients
    double* beta_x_cache = source_cache + NEL;         // NEL projected beta_x coefficients
    double* beta_y_cache = beta_x_cache + NEL;         // NEL projected beta_y coefficients
    double* beta_ref0_cache = beta_y_cache + NEL;      // NEL beta coefficients transformed by inv_aff_mats_t column 0
    double* beta_ref1_cache = beta_ref0_cache + NEL;   // NEL beta coefficients transformed by inv_aff_mats_t column 1
    double* reaction_cache = beta_ref1_cache + NEL;    // NEL projected reaction coefficients when reaction is nonscalar
#if RAW_LU_MODE_COOP
    double* lu_pivot_abs = reaction_cache + NEL;       // block scratch for cooperative pivot reduction
    int* pivots = reinterpret_cast<int*>(lu_pivot_abs + RAW_LU_SCRATCH_THREADS);
    int* lu_pivot_rows = pivots + NEL;                 // block scratch for cooperative pivot row reduction
#else
    int* pivots = reinterpret_cast<int*>(reaction_cache + NEL);
#endif

    const int tid = threadIdx.x;
    const long long element = blockIdx.x;
    if (element >= num_elements) {
        return;
    }

    assemble_projected_local_advection_raw(
        local_lu, local_rhs, tau_face, gamma_face,
        source_cache, beta_x_cache, beta_y_cache,
        beta_ref0_cache, beta_ref1_cache, reaction_cache,
        aff_jacs, inv_aff_mats_t, jacs_el_fc, normals,
        mass_matrix, reaction_triples, advection_tensor,
            advection_sparse_offsets, advection_sparse_modes,
            advection_sparse_values0, advection_sparse_values1,
            use_sparse_advection, mass_is_diagonal,
        face_basis, face_weights, trace_basis,
        source_coeffs, beta_coeffs, reaction_coeffs,
        reaction_scalar, reaction_is_scalar, loc2glob_edge, edge_side_indices, orientations, edge_to_solve_edge,
            zero_boundary_flux, num_elements, element);

    // Collapse the trace/source columns into one physical reconstruction RHS:
    // f + B(lambda).  The global trace vector uses global edge orientation, so
    // negative local orientations are mapped into local trace orientation with
    // the compile-time trace-basis transform.
    for (int i = tid; i < NEL; i += blockDim.x) {
        double value = local_rhs[i * NCOLS + (NCOLS - 1)];
        for (int face = 0; face < 3; ++face) {
            const long long edge = loc2glob_edge[element * 3 + face];
            const bool positive = orientations[element * 3 + face];
            for (int dof = 0; dof < NTR; ++dof) {
                const int column = raw_trace_column_index(face, positive, dof);
                const double trace_value = raw_oriented_trace_value(positive, dof, trace[edge * NTR + dof]);
                value += local_rhs[i * NCOLS + column] * trace_value;
            }
        }
        local_rhs[i * NCOLS] = value;
    }
    __syncthreads();

#if RAW_LU_MODE_COOP
    factor_local_lu_coop_pivot_scale_raw(local_lu, pivots, lu_pivot_abs, lu_pivot_rows);
#else
    factor_local_lu_coop_raw(local_lu, pivots);
#endif
    solve_column_zero_raw(local_lu, pivots, local_rhs);

    for (int i = tid; i < NEL; i += blockDim.x) {
        uh[element * NEL + i] = local_rhs[i * NCOLS];
    }
}
"""


_RAW_RESPONSE_RECONSTRUCT_TEMPLATE = r"""
extern "C" __global__ void reconstruct_advection_from_response_raw(
        double* __restrict__ uh,
        const double* __restrict__ local_response,
        const double* __restrict__ trace,
        const long long* __restrict__ loc2glob_edge,
        const bool* __restrict__ orientations,
        const long long num_elements)
{
    const long long element = blockIdx.x;
    if (element >= num_elements) return;
    const int tid = threadIdx.x;
    const long long response_base = element * (long long)(NEL * NCOLS);
    for (int i = tid; i < NEL; i += blockDim.x) {
        double value = local_response[response_base + i * NCOLS + (NCOLS - 1)];
        for (int face = 0; face < 3; ++face) {
            const long long edge = loc2glob_edge[element * 3 + face];
            const bool positive = orientations[element * 3 + face];
            for (int dof = 0; dof < NTR; ++dof) {
                const int column = raw_trace_column_index(face, positive, dof);
                const double trace_value = raw_oriented_trace_value(
                    positive, dof, trace[edge * NTR + dof]);
                value += local_response[response_base + i * NCOLS + column] * trace_value;
            }
        }
        uh[element * NEL + i] = value;
    }
}
"""

_RAW_RECONSTRUCT_TEMPLATE = r"""
extern "C" __global__ void reconstruct_advection_raw(
        double* __restrict__ uh,
        const double* __restrict__ trace,
        const double* __restrict__ local_mats,
        const double* __restrict__ element_boundary,
        const double* __restrict__ source_rhs,
        const long long* __restrict__ loc2glob_edge,
        const bool* __restrict__ orientations,
        const long long num_elements)
{
    // Reconstruction solves one element-local RHS: f + B * lambda.  The trace
    // vector is in global edge orientation, so negative local faces use the
    // compile-time trace-basis transform before multiplying by the element-local
    // boundary matrix.
    extern __shared__ unsigned char shared_raw[];
    double* shared = reinterpret_cast<double*>(shared_raw);
    double* local_lu = shared;
    double* rhs = local_lu + (NEL * NEL);
    int* pivots = reinterpret_cast<int*>(rhs + NEL);

    const long long element = blockIdx.x;
    if (element >= num_elements || threadIdx.x != 0) {
        return;
    }

    for (int i = 0; i < NEL; ++i) {
        rhs[i] = source_rhs[element * NEL + i];
        for (int j = 0; j < NEL; ++j) {
            local_lu[i * NEL + j] = local_mats[(element * NEL + i) * NEL + j];
        }
        for (int face = 0; face < 3; ++face) {
            const long long edge = loc2glob_edge[element * 3 + face];
            const bool positive = orientations[element * 3 + face];
            for (int dof = 0; dof < NTR; ++dof) {
                const int column = raw_trace_column_index(face, positive, dof);
                const double trace_value = raw_oriented_trace_value(positive, dof, trace[edge * NTR + dof]);
                rhs[i] += element_boundary[(element * NEL + i) * (3 * NTR) + column] * trace_value;
            }
        }
    }

    for (int k = 0; k < NEL; ++k) {
        int pivot = k;
        double max_value = fabs(local_lu[k * NEL + k]);
        for (int i = k + 1; i < NEL; ++i) {
            const double value = fabs(local_lu[i * NEL + k]);
            if (value > max_value) { max_value = value; pivot = i; }
        }
        pivots[k] = pivot;
        if (pivot != k) {
            for (int j = 0; j < NEL; ++j) {
                const double tmp = local_lu[k * NEL + j];
                local_lu[k * NEL + j] = local_lu[pivot * NEL + j];
                local_lu[pivot * NEL + j] = tmp;
            }
        }
        double diagonal = local_lu[k * NEL + k];
        if (fabs(diagonal) < 1.0e-30) {
            diagonal = diagonal >= 0.0 ? 1.0e-30 : -1.0e-30;
            local_lu[k * NEL + k] = diagonal;
        }
        for (int i = k + 1; i < NEL; ++i) {
            local_lu[i * NEL + k] /= diagonal;
            const double multiplier = local_lu[i * NEL + k];
            for (int j = k + 1; j < NEL; ++j) {
                local_lu[i * NEL + j] -= multiplier * local_lu[k * NEL + j];
            }
        }
    }
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
        for (int j = 0; j < i; ++j) { value -= local_lu[i * NEL + j] * rhs[j]; }
        rhs[i] = value;
    }
    for (int i = NEL - 1; i >= 0; --i) {
        double value = rhs[i];
        for (int j = i + 1; j < NEL; ++j) { value -= local_lu[i * NEL + j] * rhs[j]; }
        rhs[i] = value / local_lu[i * NEL + i];
    }
    for (int i = 0; i < NEL; ++i) {
        uh[element * NEL + i] = rhs[i];
    }
}
"""


def _normalize_raw_lu_mode(lu_mode: str) -> str:
    """Normalize raw LU mode to supported values."""
    normalized = str(lu_mode).replace('-', '_').lower()
    if normalized not in {'safe', 'coop'}:
        raise ValueError("raw CUDA LU mode must be one of 'safe' or 'coop'")
    return normalized


def _raw_trace_orientation_mode(trace_ref) -> str:
    """Return the edge-orientation transform used by raw CUDA trace columns.

    Parameters
    ----------
    trace_ref : object
        Trace reference data with ``kind`` and ``nodal`` attributes. Supported
        kinds are ``legacy-lagrange`` and ``legendre-modal``.

    Returns
    -------
    mode : {"nodal", "modal"}
        ``"nodal"`` means negative orientations reverse trace dof order.
        ``"modal"`` means negative orientations multiply Legendre mode ``j`` by
        ``(-1)**j`` without changing the dof index.

    Raises
    ------
    ValueError
        If the trace basis does not have a raw CUDA orientation rule.
    """
    kind = str(getattr(trace_ref, 'kind', ''))
    nodal = bool(getattr(trace_ref, 'nodal', False))
    if kind == 'legacy-lagrange' and nodal:
        return 'nodal'
    if kind == 'legendre-modal' and not nodal:
        return 'modal'
    raise ValueError(
        'raw CUDA advection assembly currently supports only legacy-lagrange '
        'nodal and legendre-modal trace bases'
    )


def _kernel_source(
        template: str,
        *,
        nel: int,
        ntr: int,
        ncols: int,
        nqf: int | None = None,
        lu_mode: str = 'safe',
        trace_orientation: str = 'nodal',
        advection_stabilization=None,
) -> str:
    """Build an instantiated CUDA kernel source string for advection assembly."""
    source = template.replace('NEL', str(int(nel))).replace('NTR', str(int(ntr))).replace('NCOLS', str(int(ncols)))
    if nqf is not None:
        source = source.replace('NQF', str(int(nqf)))
    normalized_lu_mode = _normalize_raw_lu_mode(lu_mode)
    if trace_orientation not in {'nodal', 'modal'}:
        raise ValueError("trace_orientation must be 'nodal' or 'modal'")
    from hdgfem.hdg.stabilization import upwind_factor, is_conflict_averaged_upwind
    factor = upwind_factor(advection_stabilization)
    if factor is None:
        raise ValueError("Unsupported raw CUDA advection stabilization")
    prefix = (
        f"#define RAW_ADVECTION_TAU_FACTOR {factor:.17g}\n"
        f"#define RAW_CONFLICT_AVERAGED_UPWIND {int(is_conflict_averaged_upwind(advection_stabilization))}\n"
        f"#define RAW_LU_MODE_COOP {1 if normalized_lu_mode == 'coop' else 0}\n"

        f"#define RAW_LU_SCRATCH_THREADS {_RAW_LU_SCRATCH_THREADS}\n"
        f"#define RAW_TRACE_ORIENTATION_MODAL {1 if trace_orientation == 'modal' else 0}\n"
        f"#define RAW_NTR {int(ntr)}\n"
        "__device__ __forceinline__ int raw_local_trace_dof(const bool positive, const int global_dof) {\n"
        "#if RAW_TRACE_ORIENTATION_MODAL\n"
        "    return global_dof;\n"
        "#else\n"
        "    return positive ? global_dof : (RAW_NTR - 1 - global_dof);\n"
        "#endif\n"
        "}\n"
        "__device__ __forceinline__ double raw_trace_orientation_sign(const bool positive, const int global_dof) {\n"
        "#if RAW_TRACE_ORIENTATION_MODAL\n"
        "    return (positive || ((global_dof & 1) == 0)) ? 1.0 : -1.0;\n"
        "#else\n"
        "    return 1.0;\n"
        "#endif\n"
        "}\n"
        "__device__ __forceinline__ int raw_trace_column_index(const int face, const bool positive, const int global_dof) {\n"
        "    return face * RAW_NTR + raw_local_trace_dof(positive, global_dof);\n"
        "}\n"
        "__device__ __forceinline__ double raw_oriented_trace_value(const bool positive, const int global_dof, const double value) {\n"
        "    return raw_trace_orientation_sign(positive, global_dof) * value;\n"
        "}\n"
    )
    return prefix + source


def _shared_sizes(nel: int, ntr: int) -> tuple[int, int]:
    """Compute dynamic shared-memory sizes for advection assembly and reconstruction."""
    ncols = 3 * ntr + 1
    assembly_doubles = nel * nel + nel * ncols
    assembly_bytes = assembly_doubles * REAL_ITEMSIZE + nel * 4 + 256
    reconstruct_doubles = nel * nel + nel
    reconstruct_bytes = reconstruct_doubles * REAL_ITEMSIZE + nel * 4 + 256
    return assembly_bytes, reconstruct_bytes


def _fused_shared_sizes(nel: int, ntr: int, nqf: int, *, lu_mode: str = 'safe') -> tuple[int, int]:
    """Compute shared-memory byte sizes for fused advection kernels."""
    ncols = 3 * ntr + 1
    normalized_lu_mode = _normalize_raw_lu_mode(lu_mode)
    # The fused kernels keep the local operator, all local trace/source columns,
    # trace tau and gamma on face quadrature points, and six short per-element
    # coefficient caches: source, beta_x, beta_y, transformed beta_ref0,
    # transformed beta_ref1, and optional reaction coefficients.  The reaction
    # cache is reserved even for scalar reaction so the CUDA shared-memory
    # layout is compile-time fixed.
    # The experimental cooperative LU mode adds fixed block-sized scratch arrays
    # for pivot absolute values and pivot row indices.  p=8--9 with the supported trace bases remain below the target
    # device opt-in shared-memory limit.
    base_doubles = nel * nel + nel * ncols + 6 * nqf + 6 * nel
    assembly_doubles = base_doubles
    # At p>=8 the fused Schur loop would otherwise materialize one NEL-entry
    # lift vector in every participating thread.  NVRTC lowers that
    # dynamically-indexed array to thread-local memory (global DRAM).  The LU
    # and solved columns already limit these high-order kernels to one CTA per
    # SM, so stage all 3*NTR lift rows in additional block-shared storage
    # without reducing their resident-block occupancy.
    if nel >= 45:
        assembly_doubles += 3 * ntr * nel
    reconstruct_doubles = base_doubles
    ints = nel
    if normalized_lu_mode == 'coop':
        assembly_doubles += _RAW_LU_SCRATCH_THREADS
        reconstruct_doubles += _RAW_LU_SCRATCH_THREADS
        ints += _RAW_LU_SCRATCH_THREADS
    assembly_bytes = assembly_doubles * REAL_ITEMSIZE + ints * 4 + 256
    reconstruct_bytes = reconstruct_doubles * REAL_ITEMSIZE + ints * 4 + 256
    return assembly_bytes, reconstruct_bytes


def _apply_fused_assembly_launch_bounds(
        source: str, *, kernel_name: str, nel: int, block_size: int
) -> str:
    """Apply the measured p6/128-thread occupancy specialization."""
    if int(nel) != 28 or int(block_size) != 128:
        return source
    needle = f'extern "C" __global__ void {kernel_name}('
    replacement = f'extern "C" __global__ __launch_bounds__(128, 3) void {kernel_name}('
    if source.count(needle) != 1:
        raise RuntimeError(f"expected one {kernel_name} entry point for launch-bound specialization")
    return source.replace(needle, replacement, 1)


def _raw_fused_csr_template() -> str:
    """Return the fused raw template with the assembly entry point writing CSR."""
    source = _RAW_FUSED_TEMPLATE
    start = source.index('extern "C" __global__ void assemble_advection_raw_fused(')
    end = source.index('\n\nextern "C" __global__ void reconstruct_advection_raw_fused', start)
    kernel = source[start:end]
    kernel = kernel.replace(
        'extern "C" __global__ void assemble_advection_raw_fused(\n        long long* __restrict__ rows,\n        long long* __restrict__ cols,\n        double* __restrict__ data,\n        double* __restrict__ rhs,',
        'extern "C" __global__ void assemble_advection_raw_fused_csr(\n        const int* __restrict__ csr_indptr,\n        double* __restrict__ data,\n        double* __restrict__ rhs,',
        1,
    )
    kernel = kernel.replace(
        '        const long long* __restrict__ interior_side_index,\n        const long long* __restrict__ edge_to_solve_edge,\n        const long long* __restrict__ int_edges,\n        const long long* __restrict__ side_flux_offsets,',
        '        const long long* __restrict__ interior_side_index,\n        const long long* __restrict__ edge_to_solve_edge,\n        const long long* __restrict__ int_edges,\n        const int* __restrict__ side_csr_block_pos,\n        const int* __restrict__ mass_csr_block_pos,',
        1,
    )
    kernel = kernel.replace(
        '        const int zero_boundary_flux,\n        const long long num_elements,\n        const long long num_int_edges,\n        const long long n_flux)',
        '        const int zero_boundary_flux,\n        const long long num_elements,\n        const long long num_int_edges)',
        1,
    )
    kernel = kernel.replace(
        '            const long long side_base = side_flux_offsets[side_id];\n',
        '',
        1,
    )
    kernel = kernel.replace(
        '                        const long long out = side_base + ((long long)col_block_pos * NTR + row_dof) * NTR + col_dof;\n                        rows[out] = row_solve_edge * NTR + row_dof;\n                        cols[out] = col_solve_edge * NTR + col_dof;\n                        data[out] = -schur_value;',
        '                        const int block_pos = side_csr_block_pos[side_id * 3 + col_face];\n                        const long long row = row_solve_edge * NTR + row_dof;\n                        const long long out = (long long)csr_indptr[row] + ((long long)block_pos * NTR + col_dof);\n                        atomicAdd(&data[out], -schur_value);',
        1,
    )
    kernel = kernel.replace(
        '            const bool row_positive_mass = orientations[element * 3 + row_face];\n            const int row_local_mass_dof = raw_local_trace_dof(row_positive_mass, row_dof);\n            const double row_mass_sign = raw_trace_orientation_sign(row_positive_mass, row_dof);\n            const long long mass_base = n_flux + side_id * NTR * NTR + row_dof * NTR;\n            for (int col_dof = 0; col_dof < NTR; ++col_dof) {\n                const int col_local_mass_dof = raw_local_trace_dof(row_positive_mass, col_dof);\n                const double col_mass_sign = raw_trace_orientation_sign(row_positive_mass, col_dof);\n                double mass_value = (gauge_face && row_dof == col_dof) ? 1.0 : 0.0;\n                for (int qf = 0; qf < NQF; ++qf) {\n                    const double mu_row = row_mass_sign * trace_basis[row_local_mass_dof * NQF + qf];\n                    const double mu_col = col_mass_sign * trace_basis[col_local_mass_dof * NQF + qf];\n                    mass_value = fma(gamma_face[row_face * NQF + qf] * mu_row, mu_col, mass_value);\n                }\n                const long long out = mass_base + col_dof;\n                rows[out] = row_solve_edge * NTR + row_dof;\n                cols[out] = row_solve_edge * NTR + col_dof;\n                data[out] = mass_value;\n            }',
        '            const int mass_block_pos = mass_csr_block_pos[row_solve_edge];\n            if (mass_block_pos >= 0) {\n                const bool row_positive_mass = orientations[element * 3 + row_face];\n                const int row_local_mass_dof = raw_local_trace_dof(row_positive_mass, row_dof);\n                const double row_mass_sign = raw_trace_orientation_sign(row_positive_mass, row_dof);\n                const long long row = row_solve_edge * NTR + row_dof;\n                for (int col_dof = 0; col_dof < NTR; ++col_dof) {\n                    const int col_local_mass_dof = raw_local_trace_dof(row_positive_mass, col_dof);\n                    const double col_mass_sign = raw_trace_orientation_sign(row_positive_mass, col_dof);\n                    double mass_value = (gauge_face && row_dof == col_dof) ? 1.0 : 0.0;\n                    for (int qf = 0; qf < NQF; ++qf) {\n                        const double mu_row = row_mass_sign * trace_basis[row_local_mass_dof * NQF + qf];\n                        const double mu_col = col_mass_sign * trace_basis[col_local_mass_dof * NQF + qf];\n                        mass_value = fma(gamma_face[row_face * NQF + qf] * mu_row, mu_col, mass_value);\n                    }\n                    const long long out = (long long)csr_indptr[row] + ((long long)mass_block_pos * NTR + col_dof);\n                    atomicAdd(&data[out], mass_value);\n                }\n            }',
        1,
    )
    return source[:start] + kernel + source[end:]


def _raw_fused_bsr_template() -> str:
    """Return the fused raw template with the assembly entry point writing face BSR."""
    source = _raw_fused_csr_template()
    replacements = (
        (
            "assemble_advection_raw_fused_csr",
            "assemble_advection_raw_fused_bsr",
        ),
        (
            "const long long out = (long long)csr_indptr[row] + "
            "((long long)block_pos * NTR + col_dof);",
            "const long long out = (((long long)csr_indptr[row_solve_edge] + "
            "block_pos) * NTR + row_dof) * NTR + col_dof;",
        ),
        (
            "const long long out = (long long)csr_indptr[row] + "
            "((long long)mass_block_pos * NTR + col_dof);",
            "const long long out = (((long long)csr_indptr[row_solve_edge] + "
            "mass_block_pos) * NTR + row_dof) * NTR + col_dof;",
        ),
    )
    for scalar_fragment, bsr_fragment in replacements:
        if source.count(scalar_fragment) != 1:
            raise RuntimeError("failed to specialize fused advection assembly for BSR")
        source = source.replace(scalar_fragment, bsr_fragment, 1)

    # Distinct off-diagonal face blocks are owned by exactly one triangle, so
    # assign them directly. The diagonal is shared by the two adjacent elements.
    # Stage each element's diagonal Schur row in the now-dead coefficient cache,
    # then combine it with tangent mass in the original uniform mass loop.
    schur_write = "                        atomicAdd(&data[out], -schur_value);"
    staged_bsr_write = r"""                        if (col_solve_edge == row_solve_edge) {
                            source_cache[task * NTR + col_dof] = schur_value;
                        } else {
                            data[out] = -schur_value;
                        }"""
    if source.count(schur_write) != 1:
        raise RuntimeError("failed to stage diagonal advection BSR writes")
    source = source.replace(schur_write, staged_bsr_write, 1)

    mass_write = "                    atomicAdd(&data[out], mass_value);"
    combined_diagonal_write = (
        "                    atomicAdd(&data[out], "
        "-source_cache[task * NTR + col_dof] + mass_value);"
    )
    if source.count(mass_write) != 1:
        raise RuntimeError("failed to fuse diagonal advection Schur and mass writes")
    source = source.replace(mass_write, combined_diagonal_write, 1)
    return source


def validate_raw_cuda_supported(
        cspace,
        trace_ref,
        *,
        max_el_dof: int = 28,
        max_order: int = 6,
        label: str = 'raw CUDA advection assembly',
) -> None:
    """Validate polynomial degree and trace orientation for raw advection.

    Parameters are CUDA mirrors rather than public ``DGSpace`` objects because
    this preflight is called immediately before kernel setup. A supported trace
    must be either nodal legacy Lagrange or modal Legendre, and ``el_dof`` must
    fit the statically bounded local arrays compiled into the kernels.

    Raises
    ------
    ValueError
        If the trace basis/orientation or polynomial degree is unsupported.
    """
    try:
        _raw_trace_orientation_mode(trace_ref)
    except ValueError as exc:
        raise ValueError(f'{label} currently supports only legacy-lagrange nodal and legendre-modal trace bases') from exc
    if cspace.el_dof > max_el_dof:
        raise ValueError(f'{label} currently supports p <= {max_order} (el_dof <= {max_el_dof})')


def assemble_projected_advection_trace_system_eliminated_raw_cuda(
        *,
        local_mats,
        element_boundary,
        source_rhs,
        boundary_trace,
        side_mass_blocks,
        trace_lift,
        cspace,
        trace_ref,
        block_size: RawCudaBlockSize = "auto",
) -> RawAdvectionAssemblyResult:
    """Assemble the reduced advection trace COO/RHS with a Raw CUDA solve kernel."""
    cupy = require_cupy()
    validate_raw_cuda_supported(cspace, trace_ref)
    timings: dict[str, float] = {}
    mesh_h = cspace.host.mesh
    nel = int(cspace.el_dof)
    ntr = int(cspace.edg_dof)
    ncols = 3 * ntr + 1
    block_size = resolve_raw_cuda_block_size(
        block_size, equation="advection-reaction", order=cspace.order
    )
    if block_size not in {1, 32, 64, 128}:
        raise ValueError('raw CUDA advection block_size must be one of 1, 32, 64, 128')
    assembly_shared, _ = _shared_sizes(nel, ntr)

    start = time.perf_counter()
    edge_to_solve_h = _edge_to_solve_edge(mesh_h)
    side_index_h = _interior_side_index(mesh_h)
    side_offsets_h = _side_flux_offsets(mesh_h, edge_to_solve_h, ntr)
    n_flux = int(side_offsets_h[-1])
    n_mass = int(mesh_h.interior_elements.size * ntr * ntr)
    nnz = n_flux + n_mass
    edge_to_solve = cupy.asarray(edge_to_solve_h, dtype=cupy.int64)
    side_index = cupy.asarray(side_index_h, dtype=cupy.int64)
    side_offsets = cupy.asarray(side_offsets_h, dtype=cupy.int64)
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
    template = _RAW_ASSEMBLY_TEMPLATE if block_size == 1 else _RAW_ASSEMBLY_COOP_TEMPLATE
    kernel_name = 'assemble_advection_raw' if block_size == 1 else 'assemble_advection_raw_coop'
    source = _kernel_source(
        template, nel=nel, ntr=ntr, ncols=ncols,
        trace_orientation=_raw_trace_orientation_mode(trace_ref),
    )
    kernel = _compile_kernel(cupy, source, kernel_name, assembly_shared)
    grid = (max(int(cspace.mesh.num_tri), int(cspace.mesh.int_edges_inds.size)),)
    kernel(
        grid,
        (block_size,),
        (
            rows,
            cols,
            data,
            rhs,
            local_mats,
            element_boundary,
            source_rhs,
            boundary_trace_full.reshape(-1),
            cspace.mesh.loc2glob_edge,
            cspace.mesh.orientations,
            cspace.mesh.loc2oriented_face_coupling,
            side_index,
            edge_to_solve,
            cspace.mesh.int_edges_inds,
            side_offsets,
            cspace.mesh.jacs_el_fc,
            side_mass_blocks,
            trace_lift,
            np.int64(cspace.mesh.num_tri),
            np.int64(cspace.mesh.int_edges_inds.size),
            np.int64(n_flux),
        ),
        shared_mem=int(assembly_shared),
    )
    cupy.cuda.get_current_stream().synchronize()
    timings['raw.kernel'] = time.perf_counter() - start
    timings['raw.block_size'] = float(block_size)
    timings['raw.total'] = (
        timings.get('raw.map_setup', 0.0)
        + timings.get('raw.kernel', 0.0)
        + timings.get('raw.csr_zero', 0.0)
        + timings.get('raw.csr_kernel', 0.0)
    )
    return RawAdvectionAssemblyResult(
        rows=rows,
        cols=cols,
        data=data,
        rhs=rhs,
        local_mats=local_mats,
        element_boundary=element_boundary,
        source_rhs=source_rhs,
        boundary_trace=boundary_trace,
        timings=timings,
    )


def assemble_projected_advection_trace_system_eliminated_raw_cuda_fused(
        *,
        source_coeffs,
        beta_coeffs,
        reaction_coeffs,
        reaction_scalar: float,
        reaction_is_scalar: bool,
        boundary_trace,
        cspace,
        trace_ref,
        advection_tensor,
        advection_sparse_offsets,
        advection_sparse_modes,
        advection_sparse_values0,
        advection_sparse_values1,
        use_sparse_advection: bool,
        mass_is_diagonal: bool,
        block_size: RawCudaBlockSize = "auto",
        lu_mode: str = 'safe',
        matrix_format: str = 'coo',
        zero_boundary_flux: bool = False,
        advection_stabilization=None,
        local_response=None,
        cache_local_response: bool = True,
        factor_workspace: RawAdvectionFactorWorkspace | None = None,
) -> RawAdvectionAssemblyResult:
    """Assemble the reduced advection trace system with fused projected local assembly."""
    cupy = require_cupy()
    lu_mode = _normalize_raw_lu_mode(lu_mode)
    validate_raw_cuda_supported(
        cspace,
        trace_ref,
        max_el_dof=55,
        max_order=9,
        label='fused raw CUDA advection assembly',
    )
    timings: dict[str, float] = {}
    raw_wall_start = time.perf_counter()
    mesh_h = cspace.host.mesh
    nel = int(cspace.el_dof)
    ntr = int(cspace.edg_dof)
    nqf = int(trace_ref.weights.size)
    ncols = 3 * ntr + 1
    block_size = resolve_raw_cuda_block_size(
        block_size, equation="advection-reaction", order=cspace.order
    )
    if block_size not in {1, 32, 64, 128}:
        raise ValueError('raw CUDA advection block_size must be one of 1, 32, 64, 128')
    assembly_shared, _ = _fused_shared_sizes(nel, ntr, nqf, lu_mode=lu_mode)

    source_coeffs = cupy.ascontiguousarray(source_coeffs, dtype=REAL_DTYPE)
    beta_coeffs = cupy.ascontiguousarray(beta_coeffs, dtype=REAL_DTYPE)
    if reaction_is_scalar:
        reaction_coeffs = cupy.empty(1, dtype=REAL_DTYPE)
    else:
        reaction_coeffs = cupy.ascontiguousarray(reaction_coeffs, dtype=REAL_DTYPE)
    advection_tensor = cupy.ascontiguousarray(advection_tensor, dtype=REAL_DTYPE)
    advection_sparse_offsets = cupy.ascontiguousarray(advection_sparse_offsets, dtype=cupy.int32)
    advection_sparse_modes = cupy.ascontiguousarray(advection_sparse_modes, dtype=cupy.int32)
    advection_sparse_values0 = cupy.ascontiguousarray(advection_sparse_values0, dtype=REAL_DTYPE)
    advection_sparse_values1 = cupy.ascontiguousarray(advection_sparse_values1, dtype=REAL_DTYPE)
    use_sparse_advection = bool(use_sparse_advection)
    mass_is_diagonal = bool(mass_is_diagonal)
    matrix_format = str(matrix_format).lower()
    if matrix_format not in {'coo', 'csr', 'bsr'}:
        raise ValueError("matrix_format must be 'coo', 'csr', or 'bsr'")
    zero_boundary_flux = bool(zero_boundary_flux)
    cache_local_response = bool(cache_local_response)
    response_shape = (int(cspace.mesh.num_tri), nel, ncols)
    if cache_local_response:
        if local_response is None:
            local_response = cupy.empty(response_shape, dtype=REAL_DTYPE)
        else:
            if tuple(local_response.shape) != response_shape:
                raise ValueError(
                    "local_response workspace must have shape "
                    f"{response_shape}; got {tuple(local_response.shape)}"
                )
            if local_response.dtype != REAL_DTYPE or not local_response.flags.c_contiguous:
                raise ValueError("local_response workspace must be C-contiguous float64")
    else:
        local_response = cupy.empty(1, dtype=REAL_DTYPE)
    if factor_workspace is not None:
        if not cache_local_response:
            raise ValueError("factor caching requires local response caching")
        previous_lu = factor_workspace.local_lu
        factor_workspace.ensure(cupy, num_elements=int(cspace.mesh.num_tri), nel=nel, ntr=ntr)
        reused = previous_lu is not None and factor_workspace.local_lu is previous_lu
        factor_args = (factor_workspace.local_lu, factor_workspace.pivots,
                       factor_workspace.trace_lift, np.int32(1))
        timings["raw.factors.workspace.reused"] = float(reused)
        timings["raw.factors.workspace.bytes"] = float(factor_workspace.nbytes)
    else:
        factor_args = (local_response, cupy.empty(1, dtype=cupy.int32), local_response, np.int32(0))
    if zero_boundary_flux:
        boundary_trace = cupy.zeros((mesh_h.bnd_edges_inds.size, ntr), dtype=REAL_DTYPE)
    else:
        boundary_trace = cupy.ascontiguousarray(boundary_trace, dtype=REAL_DTYPE)
    cupy.cuda.get_current_stream().synchronize()
    timings['raw.input_prepare'] = time.perf_counter() - raw_wall_start

    csr_pattern = None
    indptr = indices = None
    if matrix_format in {'csr', 'bsr'}:
        start = time.perf_counter()
        csr_pattern = build_reduced_csr_pattern_raw(
            cspace,
            timings,
            matrix_format=matrix_format,
        )
        edge_to_solve = csr_pattern.edge_to_solve_edge
        side_index = csr_pattern.interior_side_index
        indptr = csr_pattern.indptr
        indices = csr_pattern.indices
        cupy.cuda.get_current_stream().synchronize()
        timings['raw.map_setup'] = timings.get('raw.csr_pattern.total', 0.0)
        timings['raw.csr_pattern.wrapper'] = time.perf_counter() - start
        rows = cols = None
        zero_start = time.perf_counter()
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
    else:
        start = time.perf_counter()
        edge_to_solve_h = _edge_to_solve_edge(mesh_h)
        side_index_h = _interior_side_index(mesh_h)
        side_offsets_h = _side_flux_offsets(mesh_h, edge_to_solve_h, ntr)
        n_flux = int(side_offsets_h[-1])
        n_mass = int(mesh_h.interior_elements.size * ntr * ntr)
        nnz = n_flux + n_mass
        edge_to_solve = cupy.asarray(edge_to_solve_h, dtype=cupy.int64)
        side_index = cupy.asarray(side_index_h, dtype=cupy.int64)
        side_offsets = cupy.asarray(side_offsets_h, dtype=cupy.int64)
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
    if matrix_format in {'csr', 'bsr'}:
        kernel_name = f'assemble_advection_raw_fused_{matrix_format}'
        template = (
            _raw_fused_bsr_template()
            if matrix_format == 'bsr'
            else _raw_fused_csr_template()
        )
        source = _kernel_source(
            template, nel=nel, ntr=ntr, ncols=ncols, nqf=nqf,
            lu_mode=lu_mode, trace_orientation=_raw_trace_orientation_mode(trace_ref),
            advection_stabilization=advection_stabilization,
        )
        source = _apply_fused_assembly_launch_bounds(
            source, kernel_name=kernel_name, nel=nel, block_size=block_size
        )
        compile_start = time.perf_counter()
        kernel, kernel_jit = _compile_kernel_timed(
            cupy, source, kernel_name, assembly_shared
        )
        timings['raw.kernel.prepare'] = compile_start - start
        timings['raw.kernel.jit'] = kernel_jit
        stream = cupy.cuda.get_current_stream()
        begin = cupy.cuda.Event()
        end = cupy.cuda.Event()
        launch_wall_start = time.perf_counter()
        begin.record(stream)
        grid = (max(int(cspace.mesh.num_tri), int(cspace.mesh.int_edges_inds.size)),)
        kernel(
            grid,
            (block_size,),
            (
                indptr,
                data,
                rhs,
                local_response,
                *factor_args,
                cspace.mesh.loc2glob_edge,
                cspace.mesh.edge_side_indices,
                cspace.mesh.orientations,
                cspace.mesh.loc2oriented_face_coupling,
                side_index,
                edge_to_solve,
                cspace.mesh.int_edges_inds,
                csr_pattern.side_csr_block_pos,
                csr_pattern.mass_csr_block_pos,
                cspace.mesh.aff_jacs,
                cspace.mesh.inv_aff_mats_t,
                cspace.mesh.jacs_el_fc,
                cspace.mesh.edge_jacs,
                cspace.mesh.normals,
                cspace.quad_data.MKrf,
                cspace.quad_data.weighted_triple_phi_flat,
                advection_tensor,
                advection_sparse_offsets,
                advection_sparse_modes,
                advection_sparse_values0,
                advection_sparse_values1,
                np.int32(1 if use_sparse_advection else 0),
                np.int32(1 if mass_is_diagonal else 0),
                trace_ref.bas_of_bd_quads,
                trace_ref.weights,
                trace_ref.bas1d_of_ref_edg_qds,
                trace_ref.M_rf_fc,
                trace_ref.face_trace_test_element_trial_oriented,
                source_coeffs,
                beta_coeffs,
                reaction_coeffs,
                boundary_trace_full.reshape(-1),
                REAL_DTYPE(reaction_scalar),
                np.int32(1 if reaction_is_scalar else 0),
                np.int32(1 if cache_local_response else 0),
                np.int32(1 if zero_boundary_flux else 0),
                np.int64(cspace.mesh.num_tri),
                np.int64(cspace.mesh.int_edges_inds.size),
            ),
            shared_mem=int(assembly_shared),
        )
    else:
        source = _kernel_source(
            _RAW_FUSED_TEMPLATE, nel=nel, ntr=ntr, ncols=ncols, nqf=nqf,
            lu_mode=lu_mode, trace_orientation=_raw_trace_orientation_mode(trace_ref),
            advection_stabilization=advection_stabilization,
        )
        source = _apply_fused_assembly_launch_bounds(
            source, kernel_name='assemble_advection_raw_fused', nel=nel, block_size=block_size
        )
        compile_start = time.perf_counter()
        kernel, kernel_jit = _compile_kernel_timed(
            cupy, source, 'assemble_advection_raw_fused', assembly_shared
        )
        timings['raw.kernel.prepare'] = compile_start - start
        timings['raw.kernel.jit'] = kernel_jit
        stream = cupy.cuda.get_current_stream()
        begin = cupy.cuda.Event()
        end = cupy.cuda.Event()
        launch_wall_start = time.perf_counter()
        begin.record(stream)
        grid = (max(int(cspace.mesh.num_tri), int(cspace.mesh.int_edges_inds.size)),)
        kernel(
            grid,
            (block_size,),
            (
                rows,
                cols,
                data,
                rhs,
                local_response,
                *factor_args,
                cspace.mesh.loc2glob_edge,
                cspace.mesh.edge_side_indices,
                cspace.mesh.orientations,
                cspace.mesh.loc2oriented_face_coupling,
                side_index,
                edge_to_solve,
                cspace.mesh.int_edges_inds,
                side_offsets,
                cspace.mesh.aff_jacs,
                cspace.mesh.inv_aff_mats_t,
                cspace.mesh.jacs_el_fc,
                cspace.mesh.edge_jacs,
                cspace.mesh.normals,
                cspace.quad_data.MKrf,
                cspace.quad_data.weighted_triple_phi_flat,
                advection_tensor,
                advection_sparse_offsets,
                advection_sparse_modes,
                advection_sparse_values0,
                advection_sparse_values1,
                np.int32(1 if use_sparse_advection else 0),
                np.int32(1 if mass_is_diagonal else 0),
                trace_ref.bas_of_bd_quads,
                trace_ref.weights,
                trace_ref.bas1d_of_ref_edg_qds,
                trace_ref.M_rf_fc,
                trace_ref.face_trace_test_element_trial_oriented,
                source_coeffs,
                beta_coeffs,
                reaction_coeffs,
                boundary_trace_full.reshape(-1),
                REAL_DTYPE(reaction_scalar),
                np.int32(1 if reaction_is_scalar else 0),
                np.int32(1 if cache_local_response else 0),
                np.int32(1 if zero_boundary_flux else 0),
                np.int64(cspace.mesh.num_tri),
                np.int64(cspace.mesh.int_edges_inds.size),
                np.int64(n_flux),
            ),
            shared_mem=int(assembly_shared),
        )
    end.record(stream)
    end.synchronize()
    device_seconds = cupy.cuda.get_elapsed_time(begin, end) / 1000.0
    launch_wall_seconds = time.perf_counter() - launch_wall_start
    timings['raw.kernel.device'] = device_seconds
    timings['raw.kernel.wall'] = launch_wall_seconds
    timings[
        'raw.kernel' if matrix_format == 'coo' else f'raw.{matrix_format}_kernel'
    ] = device_seconds
    timings['raw.block_size'] = float(block_size)
    map_wall = timings.get('raw.csr_pattern.wrapper', timings.get('raw.map_setup', 0.0))
    timings['raw.total'] = (
        timings.get('raw.input_prepare', 0.0)
        + map_wall
        + timings.get(f'raw.{matrix_format}_zero', 0.0)
        + timings.get('raw.kernel.prepare', 0.0)
        + timings.get('raw.kernel.jit', 0.0)
        + timings.get('raw.kernel.wall', 0.0)
    )
    timings['raw.wall_total'] = time.perf_counter() - raw_wall_start
    timings['raw.unaccounted'] = max(
        0.0, timings['raw.wall_total'] - timings['raw.total']
    )
    return RawAdvectionAssemblyResult(
        rows=rows,
        cols=cols,
        data=data,
        rhs=rhs,
        boundary_trace=boundary_trace,
        timings=timings,
        indptr=indptr,
        indices=indices,
        matrix_format=matrix_format,
        csr_pattern=csr_pattern,
        source_coeffs=source_coeffs,
        beta_coeffs=beta_coeffs,
        reaction_coeffs=reaction_coeffs,
        reaction_scalar=float(reaction_scalar),
        reaction_is_scalar=bool(reaction_is_scalar),
        advection_tensor=advection_tensor,
        advection_sparse_offsets=advection_sparse_offsets,
        advection_sparse_modes=advection_sparse_modes,
        advection_sparse_values0=advection_sparse_values0,
        advection_sparse_values1=advection_sparse_values1,
        use_sparse_advection=use_sparse_advection,
        mass_is_diagonal=mass_is_diagonal,
        local_response=local_response if cache_local_response else None,
        lu_mode=lu_mode,
        zero_boundary_flux=zero_boundary_flux,
        advection_stabilization=advection_stabilization,
    )


def reconstruct_projected_advection_field_raw_cuda(
        *,
        trace,
        local_mats,
        element_boundary,
        source_rhs,
        cspace,
        trace_ref,
):
    """Recover primal element coefficients with the Raw CUDA local advection solve."""
    cupy = require_cupy()
    validate_raw_cuda_supported(cspace, trace_ref)
    nel = int(cspace.el_dof)
    ntr = int(cspace.edg_dof)
    _, reconstruct_shared = _shared_sizes(nel, ntr)
    uh = cupy.empty((cspace.mesh.num_tri, nel), dtype=REAL_DTYPE)
    source = _kernel_source(
        _RAW_RECONSTRUCT_TEMPLATE, nel=nel, ntr=ntr, ncols=1,
        trace_orientation=_raw_trace_orientation_mode(trace_ref),
    )
    kernel = _compile_kernel(cupy, source, 'reconstruct_advection_raw', reconstruct_shared)
    start = time.perf_counter()
    kernel(
        (int(cspace.mesh.num_tri),),
        (1,),
        (
            uh,
            trace.reshape(-1),
            local_mats,
            element_boundary,
            source_rhs,
            cspace.mesh.loc2glob_edge,
            cspace.mesh.orientations,
            np.int64(cspace.mesh.num_tri),
        ),
        shared_mem=int(reconstruct_shared),
    )
    cupy.cuda.get_current_stream().synchronize()
    return cupy.ascontiguousarray(uh), time.perf_counter() - start


def reconstruct_projected_advection_field_raw_cuda_fused(
        *,
        trace,
        source_coeffs,
        beta_coeffs,
        reaction_coeffs,
        reaction_scalar: float,
        reaction_is_scalar: bool,
        cspace,
        trace_ref,
        advection_tensor,
        advection_sparse_offsets,
        advection_sparse_modes,
        advection_sparse_values0,
        advection_sparse_values1,
        use_sparse_advection: bool,
        mass_is_diagonal: bool,
        block_size: RawCudaBlockSize = "auto",
        lu_mode: str = 'safe',
        zero_boundary_flux: bool = False,
        advection_stabilization=None,
):
    """Recover primal coefficients with fused projected local reconstruction."""
    cupy = require_cupy()
    lu_mode = _normalize_raw_lu_mode(lu_mode)
    validate_raw_cuda_supported(
        cspace,
        trace_ref,
        max_el_dof=55,
        max_order=9,
        label='fused raw CUDA advection reconstruction',
    )
    nel = int(cspace.el_dof)
    ntr = int(cspace.edg_dof)
    nqf = int(trace_ref.weights.size)
    ncols = 3 * ntr + 1
    block_size = resolve_raw_cuda_block_size(
        block_size, equation="advection-reaction", order=cspace.order
    )
    if block_size not in {1, 32, 64, 128}:
        raise ValueError('raw CUDA advection block_size must be one of 1, 32, 64, 128')
    _, reconstruct_shared = _fused_shared_sizes(nel, ntr, nqf, lu_mode=lu_mode)
    if reaction_is_scalar:
        reaction_coeffs = cupy.empty(1, dtype=REAL_DTYPE)
    else:
        reaction_coeffs = cupy.ascontiguousarray(reaction_coeffs, dtype=REAL_DTYPE)
    source_coeffs = cupy.ascontiguousarray(source_coeffs, dtype=REAL_DTYPE)
    beta_coeffs = cupy.ascontiguousarray(beta_coeffs, dtype=REAL_DTYPE)
    advection_tensor = cupy.ascontiguousarray(advection_tensor, dtype=REAL_DTYPE)
    advection_sparse_offsets = cupy.ascontiguousarray(advection_sparse_offsets, dtype=cupy.int32)
    advection_sparse_modes = cupy.ascontiguousarray(advection_sparse_modes, dtype=cupy.int32)
    advection_sparse_values0 = cupy.ascontiguousarray(advection_sparse_values0, dtype=REAL_DTYPE)
    advection_sparse_values1 = cupy.ascontiguousarray(advection_sparse_values1, dtype=REAL_DTYPE)
    use_sparse_advection = bool(use_sparse_advection)
    mass_is_diagonal = bool(mass_is_diagonal)
    zero_boundary_flux = bool(zero_boundary_flux)
    edge_to_solve_edge = _edge_to_solve_edge_device(cspace) if zero_boundary_flux else cupy.empty(1, dtype=cupy.int64)
    uh = cupy.empty((cspace.mesh.num_tri, nel), dtype=REAL_DTYPE)
    source = _kernel_source(
        _RAW_FUSED_TEMPLATE, nel=nel, ntr=ntr, ncols=ncols, nqf=nqf,
        lu_mode=lu_mode, trace_orientation=_raw_trace_orientation_mode(trace_ref),
        advection_stabilization=advection_stabilization,
    )
    kernel = _compile_kernel(cupy, source, 'reconstruct_advection_raw_fused', reconstruct_shared)
    start = time.perf_counter()
    kernel(
        (int(cspace.mesh.num_tri),),
        (block_size,),
        (
            uh,
            trace.reshape(-1),
            cspace.mesh.loc2glob_edge,
            cspace.mesh.edge_side_indices,
            edge_to_solve_edge,
            cspace.mesh.orientations,
            cspace.mesh.aff_jacs,
            cspace.mesh.inv_aff_mats_t,
            cspace.mesh.jacs_el_fc,
            cspace.mesh.normals,
            cspace.quad_data.MKrf,
            cspace.quad_data.weighted_triple_phi_flat,
            advection_tensor,
            advection_sparse_offsets,
            advection_sparse_modes,
            advection_sparse_values0,
            advection_sparse_values1,
            np.int32(1 if use_sparse_advection else 0),
            np.int32(1 if mass_is_diagonal else 0),
            trace_ref.bas_of_bd_quads,
            trace_ref.weights,
            trace_ref.bas1d_of_ref_edg_qds,
            source_coeffs,
            beta_coeffs,
            reaction_coeffs,
            REAL_DTYPE(reaction_scalar),
            np.int32(1 if reaction_is_scalar else 0),
            np.int32(1 if zero_boundary_flux else 0),
            np.int64(cspace.mesh.num_tri),
        ),
        shared_mem=int(reconstruct_shared),
    )
    cupy.cuda.get_current_stream().synchronize()
    return cupy.ascontiguousarray(uh), time.perf_counter() - start


def reconstruct_projected_advection_field_from_response_raw_cuda(
        *, trace, local_response, cspace, trace_ref,
        block_size: RawCudaBlockSize = "auto",
):
    """Recover coefficients from cached ``A_e^-1 [B_e, f_e]`` columns."""
    cupy = require_cupy()
    validate_raw_cuda_supported(
        cspace,
        trace_ref,
        max_el_dof=55,
        max_order=9,
        label="raw CUDA advection response reconstruction",
    )
    nel = int(cspace.el_dof)
    ntr = int(cspace.edg_dof)
    ncols = 3 * ntr + 1
    expected = (int(cspace.mesh.num_tri), nel, ncols)
    if tuple(local_response.shape) != expected:
        raise ValueError(
            f"local_response must have shape {expected}; got {tuple(local_response.shape)}"
        )
    block_size = resolve_raw_cuda_block_size(
        block_size, equation="advection-reaction", order=cspace.order
    )
    source = _kernel_source(
        _RAW_RESPONSE_RECONSTRUCT_TEMPLATE,
        nel=nel,
        ntr=ntr,
        ncols=ncols,
        trace_orientation=_raw_trace_orientation_mode(trace_ref),
    )
    kernel = _compile_kernel(
        cupy, source, "reconstruct_advection_from_response_raw", 0
    )
    uh = cupy.empty((cspace.mesh.num_tri, nel), dtype=REAL_DTYPE)
    stream = cupy.cuda.get_current_stream()
    begin = cupy.cuda.Event()
    end = cupy.cuda.Event()
    begin.record(stream)
    kernel(
        (int(cspace.mesh.num_tri),),
        (int(block_size),),
        (
            uh,
            local_response,
            trace.reshape(-1),
            cspace.mesh.loc2glob_edge,
            cspace.mesh.orientations,
            np.int64(cspace.mesh.num_tri),
        ),
    )
    end.record(stream)
    end.synchronize()
    return cupy.ascontiguousarray(uh), cupy.cuda.get_elapsed_time(begin, end) / 1000.0


_RAW_CACHED_SOURCE_TEMPLATE = r"""
extern "C" __global__ void solve_advection_cached_source(
        const double* __restrict__ element_lu,
        const int* __restrict__ element_pivots,
        const double* __restrict__ source_moments,
        double* __restrict__ local_response,
        double* __restrict__ solved_source,
        const long long num_elements)
{
    extern __shared__ unsigned char shared_raw[];
    double* lu = reinterpret_cast<double*>(shared_raw);
    double* rhs = lu + NEL * NEL;
    int* pivots = reinterpret_cast<int*>(rhs + NEL * NCOLS);
    const long long element = blockIdx.x;
    if (element >= num_elements) return;
    for (int i = threadIdx.x; i < NEL * NEL; i += blockDim.x)
        lu[i] = element_lu[element * NEL * NEL + i];
    for (int i = threadIdx.x; i < NEL; i += blockDim.x) {
        pivots[i] = element_pivots[element * NEL + i];
        rhs[i * NCOLS] = source_moments[element * NEL + i];
    }
    __syncthreads();
    solve_column_zero_raw(lu, pivots, rhs);
    for (int i = threadIdx.x; i < NEL; i += blockDim.x) {
        const double value = rhs[i * NCOLS];
        solved_source[element * NEL + i] = value;
        local_response[(element * NEL + i) * NCOLS + NCOLS - 1] = value;
    }
}
"""


def solve_cached_advection_source_raw(source_moments, local_response, factors, cspace, trace_ref):
    """Apply cached element LU factors using the existing triangular solve kernel.

    Only the source column of the local response is replaced; trace response
    columns and the condensed global operator remain untouched.
    """
    cupy = require_cupy()
    nel, ntr = int(cspace.el_dof), int(cspace.edg_dof)
    ncols = 3*ntr+1
    begin = _RAW_FUSED_TEMPLATE.index("__device__ __forceinline__ void solve_column_zero_raw(")
    end = _RAW_FUSED_TEMPLATE.index("__device__", begin+12)
    template = _RAW_FUSED_TEMPLATE[begin:end] + _RAW_CACHED_SOURCE_TEMPLATE
    shared = (nel*nel + nel*ncols)*REAL_ITEMSIZE + nel*4
    source = _kernel_source(template, nel=nel, ntr=ntr, ncols=ncols)
    kernel = _compile_kernel(cupy, source, "solve_advection_cached_source", shared)
    solved = cupy.empty((cspace.mesh.num_tri, nel), dtype=REAL_DTYPE)
    kernel((int(cspace.mesh.num_tri),), (32,),
           (factors.local_lu, factors.pivots, source_moments, local_response,
            solved, np.int64(cspace.mesh.num_tri)), shared_mem=shared)
    return solved


__all__ = [
    "RawAdvectionAssemblyResult",
    "RawAdvectionFactorWorkspace",
    "solve_cached_advection_source_raw",
    "build_reduced_csr_pattern_cupy_reference",
    "assert_reduced_csr_patterns_equal",
    "assemble_projected_advection_trace_system_eliminated_raw_cuda",
    "assemble_projected_advection_trace_system_eliminated_raw_cuda_fused",
    "reconstruct_projected_advection_field_raw_cuda",
    "reconstruct_projected_advection_field_raw_cuda_fused",
    "reconstruct_projected_advection_field_from_response_raw_cuda",
    "_raw_trace_orientation_mode",
    "validate_raw_cuda_supported",
]
