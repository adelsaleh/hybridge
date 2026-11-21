"""Tri-stage raw-CUDA local-elimination assembly for advection face BSR.

TSLE-BSR ("tri-stage local elimination", selected with
``raw_local_assembly="split3"``) assembles the same statically condensed
advection-reaction HDG trace system as the fused raw assembler in
:mod:`hybridge.transport.raw_cuda`. The discretization is identical,
and matrix, RHS, and local response agree exactly with the fused path. The
difference is scheduling: the fused kernel builds, factors, and condenses each
element in one launch. TSLE splits that work into three launches, and each
launch gets its own shared-memory footprint and block size.

Local problem and condensation
------------------------------
For element ``e`` with ``NEL`` volume dofs and ``NTR`` trace dofs per face:

    A_e u_e = B_e lambda_e + f_e

* ``A_e`` = reaction mass - (beta . grad) volume term + upwind face mass
  ``int_{dK} tau phi_i phi_j``, with ``tau = factor * |beta.n|``.
* ``B_e`` has ``3*NTR`` columns, ``int_F gamma phi_i mu_j`` with
  ``gamma = tau - beta.n``. The columns are ordered by face and stored in
  *local* (element-side) trace orientation.
* ``f_e = J * M * source_coeffs`` is the projected source moment.

Each interior face row of the global trace system is

    sum_e [ G_e - L_e A_e^{-1} B_e ] lambda = sum_e L_e A_e^{-1} f_e

where ``L_e = int_F tau mu phi`` (the "lift" rows) and ``G_e = int_F gamma mu
mu`` (the trace mass). Boundary (Dirichlet) trace columns are eliminated to
the RHS using ``boundary_trace``.

Stages (one CUDA block per element in every stage)
--------------------------------------------------
1. ``advection_tsle_build``: runs the shared device routine
   ``assemble_projected_local_advection_raw`` and stores ``A_e``,
   ``[B_e | f_e]``, and the per-face quadrature-weighted ``tau``/``gamma``
   tables in the global workspace.
2. ``advection_tsle_solve``: reloads ``A_e`` into shared memory, factors it
   with the cooperative scaled-pivot LU, and overwrites the response in place
   with ``A_e^{-1} [B_e | f_e]``. The LU factors are discarded. The workspace
   keeps the unfactored ``A_e``, so this path cannot do RHS-only re-solves
   and does not fill ``RawAdvectionFactorWorkspace``.
3. ``advection_tsle_scatter_bsr``: forms ``L_e``, the condensed Schur blocks
   and RHS, applies trace orientation and boundary elimination, and writes the
   face-BSR ``data``/``rhs`` arrays in place.

Device array layouts (selected ``REAL_DTYPE``, FP64 by default, C order)
---------------------------------------------------------
* ``local_operator``: ``(E, NEL, NEL)``, which holds ``A_e``.
* ``local_response``: ``(E, NEL, NCOLS)`` with ``NCOLS = 3*NTR + 1``. Column
  ``face*NTR + j`` is local trace dof ``j`` of ``face``; the last column is the
  source. It holds ``[B_e | f_e]`` after build and ``A_e^{-1}[B_e | f_e]``
  after solve. It is returned as ``local_response`` for
  ``reconstruct_projected_advection_field_from_response_raw_cuda``.
* ``face_flux``: ``(E, 2, 3, NQF)``, where ``[e, 0]`` is ``tau`` and
  ``[e, 1]`` is ``gamma`` at each face quadrature node, both premultiplied by
  the face Jacobian and quadrature weight (as stored by the shared routine).
* ``data``: ``(num_blocks, NTR, NTR)`` face-BSR values over the pattern from
  ``build_reduced_csr_pattern_raw(..., matrix_format="bsr")``; ``rhs`` has
  shape ``(num_interior_edges * NTR,)``.

The persistent workspace costs about ``E * (NEL**2 + NEL*NCOLS + 6*NQF) * 8``
bytes. That is several GiB at p>=7 on large meshes; see
``docs/backends/raw_cuda.md`` for measured sizes, the qualified p range, and
the fused-versus-TSLE timings behind the p>=7 recommendation.

Launch sizes
------------
``block_size="auto"`` uses persistent, correctness-checked stage selection.
The shared default 30-second/eight-candidate budget is divided into 10 seconds
and 3/3/2 candidates for build/solve/scatter, including each conservative
baseline. Trials use at most 512 actual elements on a compact isolated mesh;
there are no full-mesh finalist launches. Artificial cut boundaries use zero
trace values. Cache hits skip all sample preparation and benchmarking.
All stages share one elapsed deadline; an overrun leaves later stages at
their legal conservative sizes. Unavailable trial memory also uses those
sizes without persisting an unmeasured result.
Compiled kernels remain cached in ``_TSLE_MODULE_CACHE``.
"""

from __future__ import annotations

from hybridge.runtime.precision import REAL_DTYPE, REAL_ITEMSIZE, real_raw_module, cuda_source

from dataclasses import dataclass
import hashlib
import time
from typing import Any

import numpy as np

from hybridge.transport.raw_cuda import (
    RawAdvectionAssemblyResult,
    _RAW_FUSED_TEMPLATE,
    _fused_advection_inputs,
    _kernel_source,
    _pointwise_launch_inputs,
    _volume_launch_inputs,
    _raw_trace_orientation_mode,
    validate_raw_cuda_supported,
)
from hybridge.hdg.cuda.pattern import build_reduced_csr_pattern_raw
from hybridge.runtime.optional import require_cupy
from hybridge.hdg.cuda.launch import RawCudaBlockSize


_TSLE_KERNEL_NAMES = (
    "advection_tsle_build",
    "advection_tsle_solve",
    "advection_tsle_scatter_bsr",
)
# Powers of two only: the cooperative LU pivot search reduces by halving
# blockDim.x. Block size 1 (the serial baseline of the fused path) is not
# offered. Unlike the fused kernel, the TSLE solve sizes its pivot scratch by
# blockDim.x rather than RAW_LU_SCRATCH_THREADS, which makes 256 legal here.
_TSLE_BLOCK_CANDIDATES = (32, 64, 128, 256)
# Complete compact trial meshes keep conflict/gauge neighbor reads bounded.
_TSLE_TUNE_LIMIT = 512


@dataclass
class RawAdvectionTsleWorkspace:
    """Persistent selected-precision element workspaces used by TSLE-BSR.

    The arrays carry stage-to-stage data between the three kernels; see the
    module docstring for their layouts. Pass one instance to repeated
    assemblies (``raw_tsle_workspace`` in ``assemble_reduced_system_cuda``) to
    keep the allocations and their identities across coefficient updates. The
    values are overwritten on every assembly.

    Attributes
    ----------
    local_operator : cupy.ndarray or None
        ``(E, NEL, NEL)`` unfactored local operators ``A_e``.
    local_response : cupy.ndarray or None
        ``(E, NEL, 3*NTR + 1)``, holding ``[B_e | f_e]`` after the build stage
        and ``A_e^{-1} [B_e | f_e]`` after the solve stage.
    face_flux : cupy.ndarray or None
        ``(E, 2, 3, NQF)`` per-face ``tau`` (index 0) and ``gamma`` (index 1)
        quadrature tables, premultiplied by face Jacobian and weight.
    signature : tuple of int or None
        ``(device, E, NEL, NCOLS, NQF)`` of the current allocation.
    tensor : RawAdvectionTsleTensorWorkspace or None
        The workspace of ``raw_local_assembly="tensor"``
        (:mod:`hybridge.transport.tsle_tensor`), created on first use, so the
        solver's one workspace serves either TSLE path.
    """

    local_operator: Any | None = None
    local_response: Any | None = None
    face_flux: Any | None = None
    signature: tuple[int, ...] | None = None
    tensor: Any | None = None
    volume_gemm: Any | None = None

    def ensure(self, cupy, *, num_elements: int, nel: int, ncols: int, nqf: int) -> None:
        """Allocate stage arrays when their device/discrete signature changes."""
        signature = (
            int(cupy.cuda.runtime.getDevice()),
            int(num_elements),
            int(nel),
            int(ncols),
            int(nqf),
        )
        if self.signature == signature:
            return
        self.local_operator = cupy.empty((num_elements, nel, nel), dtype=REAL_DTYPE)
        self.local_response = cupy.empty((num_elements, nel, ncols), dtype=REAL_DTYPE)
        self.face_flux = cupy.empty((num_elements, 2, 3, nqf), dtype=REAL_DTYPE)
        self.signature = signature

    @property
    def nbytes(self) -> int:
        """Return bytes owned by the persistent stage workspace."""
        return sum(
            int(getattr(array, "nbytes", 0))
            for array in (self.local_operator, self.local_response, self.face_flux, self.tensor, self.volume_gemm)
        )

    def clear(self) -> None:
        """Release references to all persistent device arrays."""
        self.local_operator = None
        self.local_response = None
        self.face_flux = None
        self.signature = None
        self.volume_gemm = None
        if self.tensor is not None:
            self.tensor.clear()


@dataclass(frozen=True)
class _TsleKernels:
    """Compiled stage kernels for one discrete signature.

    ``jit_seconds`` is the compile time of this call: nonzero only when the
    module was actually compiled, ``0.0`` on a cache hit.
    """

    build: Any
    solve: Any
    scatter: Any
    jit_seconds: float
    source_identity: str = ""


# (device, NEL, NTR, NQF, trace orientation, upwind factor, conflict-averaged
# flag) -> compiled kernels. All of these are compile-time constants.
_TSLE_MODULE_CACHE: dict[tuple[Any, ...], _TsleKernels] = {}


# The template is appended to _RAW_FUSED_TEMPLATE and instantiated by
# _kernel_source, which substitutes NEL, NTR, NCOLS, and NQF textually and
# prepends the stabilization/orientation macros and the raw_*trace* helpers.
_TSLE_KERNEL_TEMPLATE = r"""
// ---------------------------------------------------------------------------
// Stage 1: element operator construction.
// ---------------------------------------------------------------------------
// One block per element. The shared routine fills the shared-memory local
// operator A_e, the unsolved columns [B_e | f_e], and the per-face tau/gamma
// tables exactly as the fused kernel does. They are then copied to the global
// workspace for the next stages. zero_boundary_flux zeroes tau/gamma on
// boundary (non-solve) faces inside the shared routine.
//
// Dynamic shared layout (doubles), sized by _shared_bytes_build:
//   local_operator   NEL*NEL
//   local_response   NEL*NCOLS
//   tau_face         3*NQF
//   gamma_face       3*NQF
//   coefficient caches: source and reaction (NEL each), beta_x, beta_y and
//   beta in reference coordinates 0/1 (NBETA each: the advection field may live in its
//   own DG space)
extern "C" __global__ void advection_tsle_build(
        double* __restrict__ element_operator,
        double* __restrict__ element_response,
        double* __restrict__ element_face_flux,
        const long long* __restrict__ loc2glob_edge,
        const long long* __restrict__ edge_side_indices,
        const bool* __restrict__ orientations,
        const long long* __restrict__ edge_to_solve_edge,
        const double* __restrict__ aff_jacs,
        const double* __restrict__ inv_aff_mats_t,
        const double* __restrict__ jacs_el_fc,
        const double* __restrict__ normals,
        const double* __restrict__ mass_matrix,
        const double* __restrict__ reaction_triples,
        const double* __restrict__ advection_tensor,
        const double* __restrict__ volume_table,
        const int* __restrict__ advection_sparse_offsets,
        const int* __restrict__ advection_sparse_modes,
        const double* __restrict__ advection_sparse_values0,
        const double* __restrict__ advection_sparse_values1,
        const int use_sparse_advection,
        const int mass_is_diagonal,
        const double* __restrict__ face_basis,
        const double* __restrict__ beta_face_basis,
        const double* __restrict__ face_weights,
        const double* __restrict__ trace_basis,
        const double* __restrict__ source_coeffs,
        const double* __restrict__ beta_coeffs,
        const double* __restrict__ reaction_coeffs,
        const double* __restrict__ beta_volume_values,
        const double* __restrict__ beta_normal_flux,
        const double* __restrict__ reaction_values,
        const double reaction_scalar,
        const int reaction_is_scalar,
        const int zero_boundary_flux,
        const long long num_elements)
{
    extern __shared__ unsigned char shared_raw[];
    double* local_operator = reinterpret_cast<double*>(shared_raw);
    double* local_response = local_operator + NEL * NEL;
    double* tau_face = local_response + NEL * NCOLS;
    double* gamma_face = tau_face + 3 * NQF;
    double* source_cache = gamma_face + 3 * NQF;
    double* beta_x_cache = source_cache + NEL;
    double* beta_y_cache = beta_x_cache + NBETA;
    double* beta_ref0_cache = beta_y_cache + NBETA;
    double* beta_ref1_cache = beta_ref0_cache + NBETA;
    double* reaction_cache = beta_ref1_cache + NBETA;

    const long long element = blockIdx.x;
    if (element >= num_elements) {
        return;
    }
    assemble_projected_local_advection_raw(
        local_operator, local_response, tau_face, gamma_face,
        source_cache, beta_x_cache, beta_y_cache,
        beta_ref0_cache, beta_ref1_cache, reaction_cache,
        aff_jacs, inv_aff_mats_t, jacs_el_fc, normals,
        mass_matrix, reaction_triples, advection_tensor, volume_table,
        advection_sparse_offsets, advection_sparse_modes,
        advection_sparse_values0, advection_sparse_values1,
        use_sparse_advection, mass_is_diagonal,
        face_basis, beta_face_basis, face_weights, trace_basis,
        source_coeffs, beta_coeffs, reaction_coeffs, beta_volume_values, beta_normal_flux, reaction_values,
        reaction_scalar, reaction_is_scalar, loc2glob_edge, edge_side_indices, orientations, edge_to_solve_edge,
        zero_boundary_flux, num_elements, element);

    for (int idx = threadIdx.x; idx < NEL * NEL; idx += blockDim.x) {
#if RAW_VOLUME_PRELOADED
        // The volume pre-pass rewrites the volume operator before every build
        // launch, so adding the face terms here stays idempotent.
        element_operator[element * (NEL * NEL) + idx] += local_operator[idx];
#else
        element_operator[element * (NEL * NEL) + idx] = local_operator[idx];
#endif
    }
    for (int idx = threadIdx.x; idx < NEL * NCOLS; idx += blockDim.x) {
        element_response[element * (NEL * NCOLS) + idx] = local_response[idx];
    }
    // The shared routine ends with __syncthreads(), so every entry below is
    // final. Face-flux layout per element: [tau(3, NQF) | gamma(3, NQF)].
    for (int idx = threadIdx.x; idx < 3 * NQF; idx += blockDim.x) {
        const long long base = element * (6 * NQF);
        element_face_flux[base + idx] = tau_face[idx];
        element_face_flux[base + 3 * NQF + idx] = gamma_face[idx];
    }
}


// ---------------------------------------------------------------------------
// Stage 2: pivoted local solve.
// ---------------------------------------------------------------------------
// One block per element. Loads A_e and [B_e | f_e], factors A_e with the
// cooperative scaled-pivot LU, and solves all NCOLS columns at once. Only
// the response is written back; element_operator keeps the unfactored A_e
// and the LU factors are discarded. Re-running this kernel is therefore
// destructive for element_response (autotuning rebuilds before each trial).
//
// Dynamic shared layout, sized by _shared_bytes_solve:
//   local_operator   NEL*NEL doubles
//   local_response   NEL*NCOLS doubles
//   pivot_abs        blockDim.x doubles  (pivot-search reduction scratch)
//   pivots           NEL ints
//   pivot_rows       blockDim.x ints     (pivot-search reduction scratch)
// The scratch arrays are sized by the actual block size, so blocks larger
// than RAW_LU_SCRATCH_THREADS are safe here. blockDim.x must be a power of two
// for the tree reduction.
extern "C" __global__ void advection_tsle_solve(
        const double* __restrict__ element_operator,
        double* __restrict__ element_response,
        const long long num_elements)
{
    extern __shared__ unsigned char shared_raw[];
    double* local_operator = reinterpret_cast<double*>(shared_raw);
    double* local_response = local_operator + NEL * NEL;
    double* pivot_abs = local_response + NEL * NCOLS;
    int* pivots = reinterpret_cast<int*>(pivot_abs + blockDim.x);
    int* pivot_rows = pivots + NEL;

    const long long element = blockIdx.x;
    if (element >= num_elements) {
        return;
    }
    for (int idx = threadIdx.x; idx < NEL * NEL; idx += blockDim.x) {
        local_operator[idx] = element_operator[element * (NEL * NEL) + idx];
    }
    for (int idx = threadIdx.x; idx < NEL * NCOLS; idx += blockDim.x) {
        local_response[idx] = element_response[element * (NEL * NCOLS) + idx];
    }
    __syncthreads();

    // Both helpers synchronize internally after each step; the solved response
    // is complete when solve_all_columns_raw returns.
    factor_local_lu_coop_pivot_scale_raw(
        local_operator, pivots, pivot_abs, pivot_rows);
    solve_all_columns_raw(local_operator, pivots, local_response);

    for (int idx = threadIdx.x; idx < NEL * NCOLS; idx += blockDim.x) {
        element_response[element * (NEL * NCOLS) + idx] = local_response[idx];
    }
}


// ---------------------------------------------------------------------------
// Stage 3: static condensation and face-BSR scatter.
// ---------------------------------------------------------------------------
// One block per element. Each thread owns one "task" = one (row_face,
// row_dof) trace row of this element, which gives 3*NTR tasks (30 at p=9).
// Threads beyond that have no work. Faces whose edge is not a solve edge
// (Dirichlet boundary, or zero-flux boundary) have no trace row and are
// skipped.
//
// For a task, with R = A_e^{-1}[B_e | f_e] from stage 2:
//   lift[i]       = int_F tau * mu_row * phi_i               (row of L_e)
//   schur[c]      = lift . R[:, c]  for every trace column c (row of L A^-1 B)
//   rhs_value     = lift . R[:, source] + sum_{boundary c} schur[c] * g_c
//   off-diagonal  data[row, col block]  = -schur          (plain store)
//   diagonal      data[row, row block] += -schur + G_e    (atomicAdd)
//   rhs[row]     += rhs_value                             (atomicAdd)
//
// Write conflicts: an off-diagonal block (row edge != col edge) couples two
// edges of the same triangle. Two distinct edges share at most one triangle,
// so exactly one element (CUDA block) writes it, and a plain store is safe.
// The diagonal block and RHS row of an interior edge receive one
// contribution from each of the two adjacent elements, so they accumulate
// with atomics; the host therefore zero-fills data and rhs before launch.
//
// Orientation: R stores trace columns in local (element-side) orientation,
// while the BSR blocks use the global edge orientation. raw_local_trace_dof
// maps a global dof to its local index, and raw_trace_orientation_sign gives
// the sign (always +1 for nodal traces, (-1)^j on reversed modal traces).
//
// Dynamic shared layout (doubles), sized by _shared_bytes_scatter:
//   lift_rows        3*NTR*NEL   one L_e row per task, reused for 3*NTR columns
//   diagonal_schur   3*NTR*NTR   same-edge Schur block, staged so that Schur
//                                and trace mass go out in one atomicAdd
extern "C" __global__ void advection_tsle_scatter_bsr(
        const int* __restrict__ bsr_indptr,
        double* __restrict__ data,
        double* __restrict__ rhs,
        const double* __restrict__ element_response,
        const double* __restrict__ element_face_flux,
        const long long* __restrict__ loc2glob_edge,
        const long long* __restrict__ edge_side_indices,
        const bool* __restrict__ orientations,
        const long long* __restrict__ interior_side_index,
        const long long* __restrict__ edge_to_solve_edge,
        const int* __restrict__ side_bsr_block_pos,
        const int* __restrict__ diagonal_bsr_block_pos,
        const double* __restrict__ face_basis,
        const double* __restrict__ trace_basis,
        const double* __restrict__ boundary_trace,
        const long long num_elements)
{
    extern __shared__ unsigned char shared_raw[];
    double* lift_rows = reinterpret_cast<double*>(shared_raw);
    double* diagonal_schur = lift_rows + (3 * NTR * NEL);
    const int tid = threadIdx.x;
    const long long element = blockIdx.x;
    if (element >= num_elements) {
        return;
    }

    for (int idx = tid; idx < 3 * NTR * NTR; idx += blockDim.x) {
        diagonal_schur[idx] = 0.0;
    }
    __syncthreads();

    const double* response = element_response + element * (NEL * NCOLS);
    const double* tau_face = element_face_flux + element * (6 * NQF);
    const double* gamma_face = tau_face + 3 * NQF;

    for (int task = tid; task < 3 * NTR; task += blockDim.x) {
        const int row_face = task / NTR;
        const int row_dof = task - row_face * NTR;  // global-orientation dof
        // side_id indexes the pattern's per-(side, col_face) block positions;
        // it is -1 when this face has no trace row in the reduced system.
        const long long side_id = interior_side_index[element * 3 + row_face];
        const long long row_edge = loc2glob_edge[element * 3 + row_face];
        const long long row_solve_edge = edge_to_solve_edge[row_edge];
        if (side_id < 0 || row_solve_edge < 0) {
            continue;
        }

        // Gauge for inactive faces under conflict-averaged upwinding: when
        // tau vanishes on both sides of an interior edge, the trace mass G
        // and the lifts are zero and the row would be singular. The first
        // side (edge_side_indices[2*edge]) then adds an identity to the
        // diagonal block, which pins that trace. The test is equivalent to
        // the fused kernel's raw_inactive_face_gauge, but it reads the
        // neighbor's precomputed tau instead of re-evaluating beta.n. The
        // all-zero test does not depend on quadrature node order, so no
        // orientation flip is needed.
        bool gauge_face = false;
#if RAW_CONFLICT_AVERAGED_UPWIND
        const long long other = edge_side_indices[row_edge * 2 + 1];
        gauge_face = edge_side_indices[row_edge * 2] == element * 3 + row_face && other >= 0;
        if (gauge_face) {
            // The build kernel has completed; all face weights are immutable.
            const double* other_tau = element_face_flux + (other / 3) * (6 * NQF) + (other % 3) * NQF;
            for (int q = 0; q < NQF; ++q)
                gauge_face = gauge_face && tau_face[row_face * NQF + q] == 0.0 && other_tau[q] == 0.0;
        }
#endif
        const bool row_positive = orientations[element * 3 + row_face];
        const int row_local_dof = raw_local_trace_dof(row_positive, row_dof);
        const double row_sign = raw_trace_orientation_sign(row_positive, row_dof);

        // Lift row L_e[row, :] into shared memory, and the source part of
        // the condensed RHS, lift . (A^-1 f). face_flux already carries the
        // face Jacobian and quadrature weight, and the row sign factors out.
        double rhs_value = 0.0;
        for (int i = 0; i < NEL; ++i) {
            double lift = 0.0;
            for (int qf = 0; qf < NQF; ++qf) {
                const double mu = trace_basis[row_local_dof * NQF + qf];
                const double phi = face_basis[(row_face * NEL + i) * NQF + qf];
                lift = fma(tau_face[row_face * NQF + qf] * mu, phi, lift);
            }
            lift *= row_sign;
            lift_rows[task * NEL + i] = lift;
            rhs_value += lift * response[i * NCOLS + (NCOLS - 1)];
        }

        // Schur row L_e A_e^{-1} B_e, one global-orientation column at a time.
        // Solve-edge columns go to BSR (the same-edge block is staged), and
        // boundary columns move to the RHS with the prescribed trace g.
        for (int col_face = 0; col_face < 3; ++col_face) {
            const long long col_edge = loc2glob_edge[element * 3 + col_face];
            const long long col_solve_edge = edge_to_solve_edge[col_edge];
            const bool col_positive = orientations[element * 3 + col_face];
            for (int col_dof = 0; col_dof < NTR; ++col_dof) {
                const int column = raw_trace_column_index(col_face, col_positive, col_dof);
                const double col_sign = raw_trace_orientation_sign(col_positive, col_dof);
                double schur_value = 0.0;
                for (int i = 0; i < NEL; ++i) {
                    schur_value += lift_rows[task * NEL + i]
                        * response[i * NCOLS + column];
                }
                schur_value *= col_sign;
                if (col_solve_edge >= 0) {
                    const int block_pos = side_bsr_block_pos[side_id * 3 + col_face];
                    const long long out = (((long long)bsr_indptr[row_solve_edge]
                        + block_pos) * NTR + row_dof) * NTR + col_dof;
                    if (col_solve_edge == row_solve_edge) {
                        diagonal_schur[task * NTR + col_dof] = schur_value;
                    } else {
                        data[out] = -schur_value;
                    }
                } else {
                    rhs_value += schur_value * boundary_trace[col_edge * NTR + col_dof];
                }
            }
        }

        // Diagonal block: G_e (this side's upwind trace mass, both indices in
        // this face's orientation) minus the staged same-edge Schur block,
        // plus the optional gauge identity. The pattern always contains a self
        // block for a solve edge; the -1 check is defensive.
        const int diagonal_pos = diagonal_bsr_block_pos[row_solve_edge];
        if (diagonal_pos >= 0) {
            for (int col_dof = 0; col_dof < NTR; ++col_dof) {
                const int col_local_dof = raw_local_trace_dof(row_positive, col_dof);
                const double col_sign = raw_trace_orientation_sign(row_positive, col_dof);
                double mass_value = (gauge_face && row_dof == col_dof) ? 1.0 : 0.0;
                for (int qf = 0; qf < NQF; ++qf) {
                    const double mu_row = row_sign
                        * trace_basis[row_local_dof * NQF + qf];
                    const double mu_col = col_sign
                        * trace_basis[col_local_dof * NQF + qf];
                    mass_value = fma(gamma_face[row_face * NQF + qf] * mu_row,
                                     mu_col, mass_value);
                }
                const long long out = (((long long)bsr_indptr[row_solve_edge]
                    + diagonal_pos) * NTR + row_dof) * NTR + col_dof;
                atomicAdd(&data[out],
                    -diagonal_schur[task * NTR + col_dof] + mass_value);
            }
        }
        atomicAdd(&rhs[row_solve_edge * NTR + row_dof], rhs_value);
    }
}
"""


# Shared-memory sizes mirror the layouts documented above each kernel. The
# trailing 256 bytes are alignment slack. Build/scatter conservatively reserve
# their FP64 footprints even in an FP32 process; solve uses REAL_ITEMSIZE.


def _shared_bytes_build(nel: int, ncols: int, nqf: int, nbeta: int | None = None) -> int:
    """Return dynamic shared bytes for local-operator construction.

    Operator ``nel**2``, response ``nel*ncols``, tau/gamma ``6*nqf``, source
    and reaction caches (``nel`` each) and four advection caches (``nbeta`` each,
    default ``nel``). The size does not depend on block size.
    """
    nbeta = int(nel) if nbeta is None else int(nbeta)
    return int((nel * nel + nel * ncols + 6 * nqf + 2 * nel + 4 * nbeta) * 8 + 256)


def _shared_bytes_solve(nel: int, ncols: int, block_size: int) -> int:
    """Return dynamic shared bytes for cooperative LU/all-column solve.

    Operator and response as in the build stage, plus the ``nel`` pivot
    indices and two ``block_size``-long pivot-search scratch arrays. This is
    the only stage whose footprint grows with the block size.
    """
    doubles = nel * nel + nel * ncols + block_size
    ints = nel + block_size
    return int(doubles * REAL_ITEMSIZE + ints * 4 + 256)


def _shared_bytes_scatter(nel: int, ntr: int) -> int:
    """Return dynamic shared bytes for lift/Schur/BSR scatter.

    ``3*ntr`` lift rows of length ``nel`` plus the staged ``3*ntr x ntr``
    same-edge Schur rows. The response stays in global memory.
    """
    return int((3 * ntr * nel + 3 * ntr * ntr) * 8 + 256)


def _compile_kernels(
        cupy, *, nel: int, ntr: int, nqf: int, trace_orientation: str, advection_stabilization=None,
        nbeta: int | None = None, nqv: int | None = None, pointwise_beta: bool = False,
        pointwise_reaction: bool = False, volume_preloaded: bool = False,
) -> _TsleKernels:
    """Compile or reuse the three kernels for one discrete CUDA signature.

    The source is ``_RAW_FUSED_TEMPLATE`` (the shared local-assembly, LU, and
    solve device routines) followed by ``_TSLE_KERNEL_TEMPLATE``, instantiated
    with ``lu_mode="coop"``. Dimensions, trace orientation, the upwind
    ``tau`` factor, and the conflict-averaged flag are compile-time constants
    and form the cache key, with the quadrature-point coefficient flags
    (``pointwise_beta``, ``pointwise_reaction``).

    Each kernel's dynamic shared-memory limit is raised to the largest stage
    requirement at the largest candidate block. That opts in beyond the 48 KiB
    default on devices that allow it. A failure to raise the limit is ignored
    here. The ``"auto"`` path drops candidates that exceed the device limit in
    :func:`_select_launch_blocks`; an explicit ``block_size`` that does not fit
    fails at launch.

    Raises
    ------
    ValueError
        If ``advection_stabilization`` is not an upwind policy.
    """
    device_id = int(cupy.cuda.runtime.getDevice())
    from hybridge.hdg.stabilization import upwind_factor, is_conflict_averaged_upwind
    factor = upwind_factor(advection_stabilization)
    if factor is None:
        raise ValueError("Unsupported TSLE advection stabilization")
    nbeta = int(nel) if nbeta is None else int(nbeta)
    key = (device_id, int(nel), int(ntr), int(nqf), str(trace_orientation), factor,
           is_conflict_averaged_upwind(advection_stabilization), nbeta, nqv, bool(pointwise_beta),
           bool(pointwise_reaction), bool(volume_preloaded))
    cached = _TSLE_MODULE_CACHE.get(key)
    if cached is not None:
        return _TsleKernels(cached.build, cached.solve, cached.scatter, 0.0, cached.source_identity)

    source = _kernel_source(
        _RAW_FUSED_TEMPLATE + "\n" + _TSLE_KERNEL_TEMPLATE,
        nel=nel,
        ntr=ntr,
        ncols=3 * ntr + 1,
        nqf=nqf,
        lu_mode="coop",
        trace_orientation=trace_orientation,
        advection_stabilization=advection_stabilization,
        nbeta=nbeta,
        nqv=nqv,
        pointwise_beta=pointwise_beta,
        pointwise_reaction=pointwise_reaction,
    )
    if volume_preloaded:
        source = "#define RAW_VOLUME_PRELOADED 1\n" + source
    started = time.perf_counter()
    module = real_raw_module(
        code=source,
        options=("--std=c++11", "--generate-line-info"),
        name_expressions=_TSLE_KERNEL_NAMES,
    )
    module.compile()
    kernels = _TsleKernels(
        module.get_function(_TSLE_KERNEL_NAMES[0]),
        module.get_function(_TSLE_KERNEL_NAMES[1]),
        module.get_function(_TSLE_KERNEL_NAMES[2]),
        time.perf_counter() - started,
        hashlib.sha256(cuda_source(source).encode()).hexdigest(),
    )
    max_shared = max(
        _shared_bytes_build(nel, 3 * ntr + 1, nqf, nbeta),
        _shared_bytes_solve(nel, 3 * ntr + 1, max(_TSLE_BLOCK_CANDIDATES)),
        _shared_bytes_scatter(nel, ntr),
    )
    for kernel in (kernels.build, kernels.solve, kernels.scatter):
        try:
            kernel.max_dynamic_shared_size_bytes = int(max_shared)
        except Exception:
            pass
    _TSLE_MODULE_CACHE[key] = kernels
    return kernels


def _select_launch_blocks(
        cupy, *, requested: RawCudaBlockSize, identity, kernels: _TsleKernels,
        make_trial, nel: int, ntr: int, ncols: int, nqf: int,
        timings: dict[str, float], nbeta: int | None = None,
) -> tuple[int, int, int]:
    """Select persistent stage sizes, or retain an explicit size without trials."""
    if not (requested is None or (isinstance(requested, str) and requested.lower() == "auto")):
        block_size = int(requested)
        if block_size not in _TSLE_BLOCK_CANDIDATES:
            raise ValueError("TSLE-BSR block size must be 'auto' or one of 32, 64, 128, 256")
        return block_size, block_size, block_size
    from hybridge.transport.tsle_tuning import select_stage_blocks
    return select_stage_blocks(cupy=cupy, identity=identity, kernels=kernels,
        shared_bytes=_stage_shared_bytes(nel, ntr, ncols, nqf, nbeta),
        make_trial=make_trial, nel=nel, timings=timings)


def _stage_shared_bytes(nel, ntr, ncols, nqf, nbeta):
    """Keep production and calibration launch footprints identical."""
    return dict(build=lambda block: _shared_bytes_build(nel, ncols, nqf, nbeta),
                solve=lambda block: _shared_bytes_solve(nel, ncols, block),
                scatter=lambda block: _shared_bytes_scatter(nel, ntr))


def _tuning_space_identity(space):
    """Read basis/quadrature metadata from the public space's reference data."""
    reference = space.reference
    return dict(order=space.order, basis=reference.basis_type,
                quadrature=dict(volume=reference.volume_quadrature,
                                volume_1d=reference.volume_quad_1d,
                                volume_degree=reference.volume_degree,
                                edge_1d=reference.edge_quad_1d))


def assemble_projected_advection_trace_system_eliminated_tsle_bsr(
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
        matrix_format: str = "bsr",
        zero_boundary_flux: bool = False,
        workspace: RawAdvectionTsleWorkspace | None = None,
        cache_local_response: bool = True,
        advection_stabilization=None,
        beta_face_basis=None,
        volume_table=None,
        beta_volume_values=None,
        beta_normal_flux=None,
        reaction_values=None,
        _capture_tuning_inputs: bool = False,
) -> RawAdvectionAssemblyResult:
    """Assemble a reduced face-BSR system with the TSLE three-stage pipeline.

    This is the ``raw_local_assembly="split3"`` branch of
    ``advection_cuda.assemble_reduced_system_cuda``, which prepares all
    projected coefficients and reference tensors. The inputs and the returned
    result match
    ``assemble_projected_advection_trace_system_eliminated_raw_cuda_fused``,
    except that there are no ``lu_mode``, ``local_response``, or
    ``factor_workspace`` arguments.

    Parameters
    ----------
    source_coeffs, beta_coeffs : array_like
        Projected source ``(E, NEL)`` and advection field ``(2, E, NB)`` coefficients;
        the advection field may live in another DG space (``NB != NEL``), described by
        ``beta_face_basis`` and ``advection_tensor``.
    reaction_coeffs, reaction_scalar, reaction_is_scalar
        Either a projected ``(E, NEL)`` reaction field, or a constant
        ``reaction_scalar`` when ``reaction_is_scalar`` is true (the array is
        then ignored).
    boundary_trace : array_like
        ``(num_boundary_edges, NTR)`` prescribed trace values in global edge
        orientation. Ignored (replaced by zeros) when ``zero_boundary_flux``.
    cspace, trace_ref
        CuPy element space and trace reference data. The trace basis must be
        legacy-lagrange nodal or legendre-modal, with p <= 9.
    advection_tensor, advection_sparse_*, use_sparse_advection
        Dense ``(2, NB, NEL, NEL)`` reference advection tensor, or its
        CSR-like sparse form, selected by ``use_sparse_advection``.
    mass_is_diagonal : bool
        Lets the kernel skip off-diagonal reference-mass products.
    block_size : {"auto", 32, 64, 128, 256}
        ``"auto"`` (or ``None``) tunes each stage separately; an integer
        applies to all three stages.
    matrix_format : {"bsr"}
        Only face BSR is supported.
    zero_boundary_flux : bool
        Zero ``tau``/``gamma`` on boundary faces (no inflow or outflow flux)
        instead of eliminating a prescribed boundary trace.
    workspace : RawAdvectionTsleWorkspace, optional
        Reused stage storage. A temporary one is allocated when omitted.
    cache_local_response : bool
        Return the solved response ``A_e^{-1}[B_e | f_e]`` (a view of
        ``workspace.local_response``) for reconstruction. When the workspace
        is reused, the next assembly overwrites that view.
    advection_stabilization
        Upwind stabilization policy; see ``hybridge.hdg.stabilization``.
    beta_face_basis : array_like, optional
        ``(3, NB, NQF)`` advection basis on the trace face quadrature; ``None``
        for an advection field in the solution space.
    volume_table : array_like, optional
        Packed volume quadrature table selecting the quadrature volume
        operator (``hybridge.transport.raw_cuda.volume_quadrature_layout``).
    beta_volume_values, beta_normal_flux, reaction_values : array_like, optional
        Quadrature-point advection field ``(E, NQV, 2)`` with its normal flux ``(E, 3, NQF)``
        and reaction ``(E, NQV)``, as in the fused assembly; they need
        ``volume_table``.

    Returns
    -------
    RawAdvectionAssemblyResult
        ``data`` ``(num_blocks, NTR, NTR)``, ``rhs``, ``indptr``/``indices``
        (the BSR block graph), the pattern, the device inputs used, and
        ``lu_mode="coop"``. ``timings`` includes ``raw.tsle.{build,solve,
        scatter,device}`` CUDA-event seconds, the per-stage block sizes, JIT
        and autotune costs, and workspace bytes. It also fills the generic
        ``raw.*`` keys shared with the fused path.

    Raises
    ------
    ValueError
        For an unsupported trace basis or p > 9, a non-BSR ``matrix_format``,
        an invalid ``block_size``, or an unsupported stabilization.
    """
    cupy = require_cupy()
    validate_raw_cuda_supported(
        cspace,
        trace_ref,
        max_el_dof=55,
        max_order=9,
        label="TSLE-BSR advection assembly",
    )
    if str(matrix_format).lower() != "bsr":
        raise ValueError("TSLE-BSR requires matrix_format='bsr'")

    timings: dict[str, float] = {}
    wall_start = time.perf_counter()
    mesh_h = cspace.host.mesh
    num_elements = int(cspace.mesh.num_tri)
    nel = int(cspace.el_dof)
    ntr = int(cspace.edg_dof)
    nqf = int(trace_ref.weights.size)
    ncols = 3 * ntr + 1  # 3 faces x NTR trace columns + 1 source column

    # The kernels take raw pointers, so every input must be a contiguous array
    # of the dtype they expect. The coerced arrays are returned in the result
    # so that reconstruction reuses the same device buffers. A scalar reaction
    # passes a 1-element dummy that the kernel never reads.
    source_coeffs = cupy.ascontiguousarray(source_coeffs, dtype=REAL_DTYPE)
    (beta_volume_values, beta_normal_flux, reaction_values,
     pointwise_beta, pointwise_reaction) = _pointwise_launch_inputs(
        cupy, cspace, trace_ref, beta_volume_values, beta_normal_flux, reaction_values)
    beta_coeffs, beta_face_basis, advection_tensor, nbeta = _fused_advection_inputs(
        cupy, beta_coeffs, beta_face_basis, advection_tensor, cspace, trace_ref, pointwise_beta)
    if (pointwise_beta or pointwise_reaction) and volume_table is None:
        raise ValueError("quadrature-point coefficients need the quadrature volume table (volume_table)")
    volume_table, nqv = _volume_launch_inputs(cupy, volume_table, nel, nbeta)
    if pointwise_reaction and reaction_is_scalar:
        raise ValueError("reaction_values give a field reaction; set reaction_is_scalar=False")
    reaction_coeffs = (
        cupy.empty(1, dtype=REAL_DTYPE)
        if reaction_is_scalar or pointwise_reaction
        else cupy.ascontiguousarray(reaction_coeffs, dtype=REAL_DTYPE)
    )
    advection_sparse_offsets = cupy.ascontiguousarray(advection_sparse_offsets, dtype=cupy.int32)
    advection_sparse_modes = cupy.ascontiguousarray(advection_sparse_modes, dtype=cupy.int32)
    advection_sparse_values0 = cupy.ascontiguousarray(advection_sparse_values0, dtype=REAL_DTYPE)
    advection_sparse_values1 = cupy.ascontiguousarray(advection_sparse_values1, dtype=REAL_DTYPE)

    if zero_boundary_flux:
        boundary_trace = cupy.zeros((mesh_h.bnd_edges_inds.size, ntr), dtype=REAL_DTYPE)
    else:
        boundary_trace = cupy.ascontiguousarray(boundary_trace, dtype=REAL_DTYPE)

    pattern_start = time.perf_counter()
    pattern = build_reduced_csr_pattern_raw(cspace, timings, matrix_format="bsr")
    timings["raw.tsle.pattern.wrapper"] = time.perf_counter() - pattern_start
    data = cupy.zeros((pattern.num_blocks, ntr, ntr), dtype=REAL_DTYPE)
    rhs = cupy.zeros(mesh_h.int_edges_inds.size * ntr, dtype=REAL_DTYPE)
    # The scatter kernel looks up boundary values by global edge id, so expand
    # the boundary-edge table to all edges (zero on interior edges, which are
    # never read).
    boundary_trace_full = cupy.zeros((mesh_h.num_edg, ntr), dtype=REAL_DTYPE)
    if mesh_h.bnd_edges_inds.size:
        boundary_trace_full[cspace.mesh.bnd_edges_inds] = boundary_trace

    owned_workspace = workspace is None
    if workspace is None:
        workspace = RawAdvectionTsleWorkspace()
    workspace.ensure(
        cupy,
        num_elements=num_elements,
        nel=nel,
        ncols=ncols,
        nqf=nqf,
    )
    timings["raw.tsle.workspace.bytes"] = float(workspace.nbytes)
    cupy.cuda.get_current_stream().synchronize()
    timings["raw.input_prepare"] = time.perf_counter() - wall_start

    trace_orientation = _raw_trace_orientation_mode(trace_ref)
    from hybridge.transport import tsle_volume_gemm
    volume = tsle_volume_gemm.volume_mode()
    preloaded = volume != "inline" and tsle_volume_gemm.volume_prepass_applies(
        nqv=nqv, pointwise_beta=pointwise_beta, pointwise_reaction=pointwise_reaction)
    if preloaded and REAL_ITEMSIZE != 8:
        raise ValueError("preloaded split3 volume assembly requires float64; use HYBRIDGE_RAW_SPLIT3_VOLUME=inline")
    kernels = _compile_kernels(
        cupy,
        nel=nel,
        ntr=ntr,
        nqf=nqf,
        trace_orientation=trace_orientation,
        advection_stabilization=advection_stabilization,
        nbeta=nbeta,
        nqv=nqv,
        pointwise_beta=pointwise_beta,
        pointwise_reaction=pointwise_reaction,
        volume_preloaded=preloaded,
    )
    timings["raw.tsle.jit"] = float(kernels.jit_seconds)

    # Argument tuples follow the kernel signatures in _TSLE_KERNEL_TEMPLATE
    # position by position. Scalars are wrapped in explicit NumPy types so
    # that CuPy passes int/long long/double exactly as declared.
    build_args = (
        workspace.local_operator,
        workspace.local_response,
        workspace.face_flux,
        cspace.mesh.loc2glob_edge,
        cspace.mesh.edge_side_indices,
        cspace.mesh.orientations,
        pattern.edge_to_solve_edge,
        cspace.mesh.aff_jacs,
        cspace.mesh.inv_aff_mats_t,
        cspace.mesh.jacs_el_fc,
        cspace.mesh.normals,
        cspace.quad_data.MKrf,
        cspace.quad_data.weighted_triple_phi_flat,
        advection_tensor,
        volume_table,
        advection_sparse_offsets,
        advection_sparse_modes,
        advection_sparse_values0,
        advection_sparse_values1,
        np.int32(1 if use_sparse_advection else 0),
        np.int32(1 if mass_is_diagonal else 0),
        trace_ref.bas_of_bd_quads,
        beta_face_basis,
        trace_ref.weights,
        trace_ref.bas1d_of_ref_edg_qds,
        source_coeffs,
        beta_coeffs,
        reaction_coeffs,
        beta_volume_values,
        beta_normal_flux,
        reaction_values,
        REAL_DTYPE(reaction_scalar),
        np.int32(1 if reaction_is_scalar else 0),
        np.int32(1 if zero_boundary_flux else 0),
        np.int64(num_elements),
    )
    solve_args = (
        workspace.local_operator,
        workspace.local_response,
        np.int64(num_elements),
    )
    build_kernel = kernels.build
    if preloaded:
        # The build stage becomes "volume GEMM for the launched elements, then
        # the build kernel adding the face terms" (tsle_volume_gemm).
        workspace.volume_gemm = tsle_volume_gemm.prepare_volume_gemm(
            cupy, mode=volume, operator=workspace.local_operator, existing=workspace.volume_gemm,
            num_elements=num_elements, nel=nel, nbeta=nbeta, advection_tensor=advection_tensor,
            mass_matrix=cspace.quad_data.MKrf, reaction_triples=cspace.quad_data.weighted_triple_phi_flat,
            mass_is_diagonal=bool(mass_is_diagonal), beta_coeffs=beta_coeffs, aff_jacs=cspace.mesh.aff_jacs,
            inv_aff_mats_t=cspace.mesh.inv_aff_mats_t, reaction_coeffs=reaction_coeffs,
            reaction_scalar=float(reaction_scalar), reaction_is_scalar=bool(reaction_is_scalar), timings=timings)
        kernels = _TsleKernels(tsle_volume_gemm.PreloadedBuild(build_kernel, workspace.volume_gemm),
                               kernels.solve, kernels.scatter, kernels.jit_seconds, kernels.source_identity)
        timings["raw.tsle.workspace.bytes"] = float(workspace.nbytes)
    timings["raw.tsle.volume.preloaded"] = float(preloaded)
    scatter_args = (
        pattern.indptr,
        data,
        rhs,
        workspace.local_response,
        workspace.face_flux,
        cspace.mesh.loc2glob_edge,
        cspace.mesh.edge_side_indices,
        cspace.mesh.orientations,
        pattern.interior_side_index,
        pattern.edge_to_solve_edge,
        pattern.side_csr_block_pos,
        pattern.mass_csr_block_pos,
        trace_ref.bas_of_bd_quads,
        trace_ref.bas1d_of_ref_edg_qds,
        boundary_trace_full.reshape(-1),
        np.int64(num_elements),
    )

    from hybridge.hdg.stabilization import upwind_factor, is_conflict_averaged_upwind
    if _capture_tuning_inputs:
        # Only the lazy trial factory uses this private preparation return.
        # It owns a compact mesh and all mutable arrays; no trial executes on
        # the production workspace, even for scatter or preloaded volume GEMM.
        from hybridge.transport.tsle_tuning import StageTrial
        return StageTrial(cupy, kernels,
            dict(build=build_args, solve=solve_args, scatter=scatter_args),
            _stage_shared_bytes(nel, ntr, ncols, nqf, nbeta), num_elements,
            trace_orientation, is_conflict_averaged_upwind(advection_stabilization))

    tuning_key = dict(adapter="transport-split3-stages-v1", source=kernels.source_identity,
        precision=np.dtype(REAL_DTYPE).name, nel=nel, ntr=ntr, nqf=nqf, nbeta=nbeta, nqv=nqv,
        **_tuning_space_identity(cspace.host), trace_orientation=trace_orientation,
        stabilization=dict(factor=upwind_factor(advection_stabilization),
            conflict_averaged=is_conflict_averaged_upwind(advection_stabilization)),
        sparse_advection=bool(use_sparse_advection), diagonal_mass=bool(mass_is_diagonal),
        scalar_reaction=bool(reaction_is_scalar), zero_boundary_flux=bool(zero_boundary_flux),
        pointwise_beta=bool(pointwise_beta), pointwise_reaction=bool(pointwise_reaction),
        elements_bucket=max(1, int(num_elements)).bit_length(),
        sample_elements=min(num_elements, _TSLE_TUNE_LIMIT), sample_policy="element-prefix-zero-cut-boundary-v1",
        volume=volume if preloaded else "inline",
        volume_source=hashlib.sha256(tsle_volume_gemm._SOURCE.encode()).hexdigest() if preloaded else None)

    def make_trial():
        """Prepare bounded same-precision scratch only on a calibration miss."""
        from hybridge.transport.tuning_sample import sample_raw_options
        from hybridge.transport.tsle_tuning import TuningScratchUnavailable
        # Account for stage storage, saved parity outputs, LU RHS restoration,
        # dense-oracle copies, compact BSR data and coefficient/geometry tables.
        # The check is lazy, so a cached calibration needs none of this memory.
        estimated = min(num_elements, _TSLE_TUNE_LIMIT) * (
            4 * (nel*nel + nel*ncols + 6*nqf) + 18*ntr*ntr
            + 8*nel + 4*nbeta + 3*ntr + 256) * REAL_ITEMSIZE
        free, _ = cupy.cuda.runtime.memGetInfo()
        available = int(free) + int(cupy.get_default_memory_pool().free_bytes())
        if estimated > available:
            raise TuningScratchUnavailable("bounded split3 calibration scratch does not fit")
        try:
            options, _ = sample_raw_options(cupy, dict(
                source_coeffs=source_coeffs, beta_coeffs=beta_coeffs,
                reaction_coeffs=reaction_coeffs, reaction_scalar=reaction_scalar,
                reaction_is_scalar=reaction_is_scalar, boundary_trace=boundary_trace,
                cspace=cspace, trace_ref=trace_ref, advection_tensor=advection_tensor,
                advection_sparse_offsets=advection_sparse_offsets, advection_sparse_modes=advection_sparse_modes,
                advection_sparse_values0=advection_sparse_values0, advection_sparse_values1=advection_sparse_values1,
                use_sparse_advection=use_sparse_advection, mass_is_diagonal=mass_is_diagonal,
                block_size=32, matrix_format="bsr", zero_boundary_flux=zero_boundary_flux,
                advection_stabilization=advection_stabilization, beta_face_basis=beta_face_basis,
                volume_table=volume_table if nqv is not None else None,
                beta_volume_values=beta_volume_values if pointwise_beta else None,
                beta_normal_flux=beta_normal_flux if pointwise_beta else None,
                reaction_values=reaction_values if pointwise_reaction else None), limit=_TSLE_TUNE_LIMIT)
            return assemble_projected_advection_trace_system_eliminated_tsle_bsr(
                **options, _capture_tuning_inputs=True)
        except cupy.cuda.memory.OutOfMemoryError as exc:
            raise TuningScratchUnavailable("split3 calibration allocation failed") from exc

    selected = _select_launch_blocks(
        cupy,
        requested=block_size,
        identity=tuning_key,
        kernels=kernels,
        make_trial=make_trial,
        nel=nel,
        ntr=ntr,
        ncols=ncols,
        nqf=nqf,
        timings=timings,
        nbeta=nbeta,
    )
    build_block, solve_block, scatter_block = selected
    timings["raw.tsle.build.block_size"] = float(build_block)
    timings["raw.tsle.solve.block_size"] = float(solve_block)
    timings["raw.tsle.scatter.block_size"] = float(scatter_block)
    from hybridge.hdg.cuda.launch import record_kernel_resources
    for stage, kernel, block, shared in (
            ("build", build_kernel, build_block, _shared_bytes_build(nel, ncols, nqf, nbeta)),
            ("solve", kernels.solve, solve_block, _shared_bytes_solve(nel, ncols, solve_block)),
            ("scatter", kernels.scatter, scatter_block, _shared_bytes_scatter(nel, ntr))):
        record_kernel_resources(cupy, kernel, block, shared, timings, f"raw.tsle.{stage}.resource.")
    # The response reconstruction kernel retains the shared raw-CUDA public
    # launch policy (at most 128 threads). TSLE's independent stage choices are
    # recorded above and may legitimately select 256.
    timings["raw.block_size"] = float(min(build_block, 128))

    # Production launch. Scatter accumulates atomically, so both outputs are
    # cleared first. Calibration used only its isolated compact mesh. All
    # three stages are queued back to back on the current stream; the four
    # events bracket each stage for the per-stage device timings.
    data.fill(0.0)
    rhs.fill(0.0)
    stream = cupy.cuda.get_current_stream()
    events = [cupy.cuda.Event() for _ in range(4)]
    volume_done = cupy.cuda.Event()
    launch_wall_start = time.perf_counter()
    events[0].record(stream)
    if preloaded:
        workspace.volume_gemm(num_elements)
    volume_done.record(stream)
    build_kernel(
        (num_elements,),
        (build_block,),
        build_args,
        shared_mem=_shared_bytes_build(nel, ncols, nqf, nbeta),
    )
    events[1].record(stream)
    kernels.solve(
        (num_elements,),
        (solve_block,),
        solve_args,
        shared_mem=_shared_bytes_solve(nel, ncols, solve_block),
    )
    events[2].record(stream)
    kernels.scatter(
        (num_elements,),
        (scatter_block,),
        scatter_args,
        shared_mem=_shared_bytes_scatter(nel, ntr),
    )
    events[3].record(stream)
    events[3].synchronize()
    timings["raw.tsle.build"] = cupy.cuda.get_elapsed_time(events[0], events[1]) / 1000.0
    if preloaded:
        timings["raw.tsle.volume"] = cupy.cuda.get_elapsed_time(events[0], volume_done) / 1000.0
    timings["raw.tsle.solve"] = cupy.cuda.get_elapsed_time(events[1], events[2]) / 1000.0
    timings["raw.tsle.scatter"] = cupy.cuda.get_elapsed_time(events[2], events[3]) / 1000.0
    timings["raw.tsle.device"] = cupy.cuda.get_elapsed_time(events[0], events[3]) / 1000.0
    # Generic keys shared with the fused path, so that solver logging and
    # benchmarks can compare the two assemblers without special cases.
    timings["raw.bsr_kernel"] = timings["raw.tsle.device"]
    timings["raw.kernel.device"] = timings["raw.tsle.device"]
    timings["raw.kernel.wall"] = time.perf_counter() - launch_wall_start
    timings["raw.total"] = time.perf_counter() - wall_start
    timings["raw.wall_total"] = timings["raw.total"]
    timings["raw.unaccounted"] = 0.0

    response = workspace.local_response if cache_local_response else None
    # A temporary workspace needs no explicit lifetime handling: events[3] was
    # synchronized above, so no queued kernel still reads it, and the result
    # holds local_response when it is cached. owned_workspace is informational.
    _ = owned_workspace
    return RawAdvectionAssemblyResult(
        advection_stabilization=advection_stabilization,
        data=data,
        rhs=rhs,
        boundary_trace=boundary_trace,
        timings=timings,
        rows=None,
        cols=None,
        indptr=pattern.indptr,
        indices=pattern.indices,
        matrix_format="bsr",
        csr_pattern=pattern,
        source_coeffs=source_coeffs,
        beta_coeffs=beta_coeffs,
        beta_face_basis=beta_face_basis,
        volume_table=volume_table if nqv is not None else None,
        reaction_coeffs=reaction_coeffs,
        reaction_scalar=float(reaction_scalar),
        reaction_is_scalar=bool(reaction_is_scalar),
        advection_tensor=advection_tensor,
        advection_sparse_offsets=advection_sparse_offsets,
        advection_sparse_modes=advection_sparse_modes,
        advection_sparse_values0=advection_sparse_values0,
        advection_sparse_values1=advection_sparse_values1,
        use_sparse_advection=bool(use_sparse_advection),
        mass_is_diagonal=bool(mass_is_diagonal),
        local_response=response,
        lu_mode="coop",
        zero_boundary_flux=bool(zero_boundary_flux),
        beta_volume_values=beta_volume_values if pointwise_beta else None,
        beta_normal_flux=beta_normal_flux if pointwise_beta else None,
        reaction_values=reaction_values if pointwise_reaction else None,
        pointwise_beta=pointwise_beta,
        pointwise_reaction=pointwise_reaction,
    )


def split3_workspace_bytes(num_elements: int, nel: int, ntr: int, nqf: int) -> int:
    """Return the bytes of a :class:`RawAdvectionTsleWorkspace` for this discrete signature."""
    ncols = 3 * int(ntr) + 1
    return int(num_elements) * (int(nel) ** 2 + int(nel) * ncols + 6 * int(nqf)) * REAL_ITEMSIZE


def split3_memory_sizes(space, trace_space, workspace: RawAdvectionTsleWorkspace | None = None) -> tuple[int, int]:
    """Return ``(workspace_bytes, available_bytes)`` for the memory-aware split3 choice.

    Available memory is the driver's free device memory plus the free blocks
    of CuPy's default memory pool. A workspace already allocated for this
    signature counts as available, so a solver that chose split3 keeps it.
    """
    cupy = require_cupy()
    nqf = int(trace_space.weights.size)
    needed = split3_workspace_bytes(space.mesh.num_tri, space.el_dof, space.order + 1, nqf)
    free, _total = cupy.cuda.runtime.memGetInfo()
    available = int(free) + int(cupy.get_default_memory_pool().free_bytes())
    signature = (int(cupy.cuda.runtime.getDevice()), int(space.mesh.num_tri), int(space.el_dof),
                 3 * (space.order + 1) + 1, nqf)
    if workspace is not None and workspace.signature == signature:
        available += workspace.nbytes
    return needed, available


def clear_tsle_runtime_caches() -> None:
    """Clear process caches, retaining persistent calibration records on disk.

    Use HYBRIDGE_CUDA_TUNING=refresh to recalibrate. Existing numerical
    workspaces are unaffected.
    """
    from hybridge.runtime.cuda_tuning import clear_tuning_cache
    _TSLE_MODULE_CACHE.clear()
    clear_tuning_cache()


__all__ = [
    "RawAdvectionTsleWorkspace",
    "assemble_projected_advection_trace_system_eliminated_tsle_bsr",
    "clear_tsle_runtime_caches",
    "split3_memory_sizes",
    "split3_workspace_bytes",
]
