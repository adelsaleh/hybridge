"""Tensorized TSLE assembly for advection face BSR (experimental).

TSLE-BSR (:mod:`hybridge.transport.tsle_bsr`, ``raw_local_assembly="split3"``)
builds every element operator inside a raw kernel that saturates the FP64 pipe
on FP64-limited GPUs. This variant keeps the same discretization and the same
three-stage split, but moves the dense work onto library contractions and
batched LU in a selectable working precision. ``precision="float32"`` is the
``split3-fp32-tensor`` prototype tracked in ``TODO.md``; ``"float64"`` runs the
same algebra in FP64 as its accuracy baseline.

Stages
------
1. Build. An FP64 kernel evaluates the quadrature-weighted face tables
   ``tau_w``/``gamma_w`` (``face_jac * w_q`` times ``tau``/``gamma``) with the
   arithmetic of the fused/TSLE routine: upwind factor, conflict averaging, and
   zero-flux faces. A second FP64 kernel forms the per-element coefficient
   rows, which are rounded once to the working precision:

       x_A[e] = [-J b_ref0 | -J b_ref1 | J r | tau_w]             (K_A values)
       A_e    = x_A[e] . Y_A,      Y_A = [T_0; T_1; R; P]         (K_A x NEL^2)

   ``T_D[k, (i,j)]`` is the reference advection tensor, ``b_ref`` the velocity
   coefficients in reference coordinates, ``R`` the reaction mass (one ``M`` row
   for a scalar reaction, ``NEL`` triple-product rows for a field), and
   ``P[(f,q), (i,j)] = phi_fi(q) phi_fj(q)`` the boundary mass. The trace
   columns ``B_e[i, (f,t)] = sum_q gamma_w[e,f,q] phi_fi(q) mu_t(q)`` and the
   source column ``f_e = J M^T s_e`` are two more contractions.
2. Solve. Batched pivoted LU of ``A_e`` and solution of all ``3*NTR + 1``
   columns: the TSLE cooperative kernel compiled in the working precision
   (``"coop"``), cuBLAS ``getrf/getrsBatched`` (``"cublas"``), MAGMA
   batched LU (``"magma"``), or the fused register Gauss-Jordan solve of
   :mod:`hybridge.linalg.gpu.fused_batched_solve` (``"fused"``, FP64 only:
   one block per element, partial pivoting, ``A_e`` left intact, the
   response written in place in the layout stage 3 reads).
3. Condense. Lift rows ``L_e[(f,t), i] = sum_q tau_w mu_t phi_fi``, Schur rows
   ``S_e = L_e R_e`` with ``R_e = A_e^{-1}[B_e | f_e]``, and trace masses
   ``G_e[f, a, b] = sum_q gamma_w mu_a mu_b``, all in local (element-side)
   trace orientation. An FP64 scatter kernel then applies orientation,
   boundary elimination, and the inactive-face gauge, and accumulates the
   face-BSR ``data``/``rhs`` in FP64.

Contraction engines
-------------------
``contraction="cutensor"`` calls cuTENSOR through ``cupyx.cutensor`` with
plans built once per workspace. Face-batched products (``B``, ``L``, ``G``)
use a mode shared by both operands and the output, so no zero blocks are
multiplied, and results are written straight into strided views of the stage
buffers. ``compute`` selects the cuTENSOR compute descriptor: ``"default"``
(32F or 64F), ``"3xtf32"`` (FP32-accurate tensor-core emulation), or
``"tf32"``. ``contraction="cublas"`` uses dense GEMMs with block-structured
reference tables. ``compute="bf16x9"`` (FP32 only) uses cuBLAS
``CUBLAS_COMPUTE_32F_EMULATED_16BFX9``.

Layouts
-------
``"coop"`` works on C-order ``A_e`` ``(E, NEL, NEL)`` and responses
``(E, NEL, NCOLS)``. The library solvers work on column-major data: stage 1
writes ``A_e`` as its C-order transpose and the response as ``(E, NCOLS, NEL)``,
so no transposition pass is needed.
"""

from __future__ import annotations

import hashlib
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hybridge.hdg.cuda.pattern import build_reduced_csr_pattern_raw
from hybridge.linalg.gpu.cublas_batched import BatchedLUWorkspace, lu_solve_batched_cublas
from hybridge.runtime.optional import require_cupy, require_cutensor
from hybridge.runtime.precision import specialize_real_source
from hybridge.transport.raw_cuda import (
    RawAdvectionAssemblyResult,
    _RAW_FUSED_TEMPLATE,
    _kernel_source,
    _raw_trace_orientation_mode,
    validate_raw_cuda_supported,
)
from hybridge.transport.tsle_bsr import _TSLE_KERNEL_TEMPLATE

TENSOR_PRECISIONS = ("float32", "float64")
TENSOR_CONTRACTIONS = ("cutensor", "cublas")
TENSOR_COMPUTES = ("default", "3xtf32", "tf32", "bf16x9")
TENSOR_LOCAL_SOLVERS = ("auto", "coop", "cublas", "magma", "fused")

# Candidate block sizes for the cooperative solve (powers of two, as required
# by its pivot reduction) and the prefix grid used to time them.
_COOP_BLOCKS = (32, 64, 128, 256)
_COOP_TUNE_LIMIT = 32_768
_SCATTER_BLOCK = 128
_ELEMENTWISE_BLOCK = 256
# cuBLAS values not exported by CuPy: CUBLAS_COMPUTE_32F_EMULATED_16BFX9
# (cublas_api.h, CUDA 13.0), CUDA_R_32F, CUBLAS_GEMM_DEFAULT.
_CUBLAS_COMPUTE_32F_EMULATED_16BFX9 = 78
_CUDA_R_32F = 0
_CUBLAS_GEMM_DEFAULT = -1

# (device, NEL, NTR, NQF, orientation, stabilization, working dtype) ->
# compiled kernels; and coop tuning results keyed like TSLE's.
_MODULE_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_COOP_TUNING_CACHE: dict[tuple[Any, ...], int] = {}


# The template is appended to _RAW_FUSED_TEMPLATE (for raw_side_normal and the
# stabilization macros) and instantiated by _kernel_source, which replaces the
# upper-case dimension tokens textually; identifiers below therefore avoid
# upper-case words that contain them. These kernels stay FP64; only the
# arrays typed WORK_REAL use the working precision.
_TENSOR_KERNEL_TEMPLATE = r"""
// One thread per (element, face, face quadrature node): quadrature-weighted
// tau/gamma with exactly the arithmetic of assemble_projected_local_advection_raw.
// Output layout (E, 3, NQF).
extern "C" __global__ void advection_tsle_tensor_face_weights(
        double* __restrict__ tau_out,
        double* __restrict__ gamma_out,
        const double* __restrict__ beta_coeffs,
        const double* __restrict__ face_basis,
        const double* __restrict__ face_weights,
        const double* __restrict__ normals,
        const double* __restrict__ jacs_el_fc,
        const long long* __restrict__ loc2glob_edge,
        const long long* __restrict__ edge_side_indices,
        const bool* __restrict__ orientations,
        const long long* __restrict__ edge_to_solve_edge,
        const int zero_boundary_flux,
        const long long num_elements)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements * 3 * NQF) {
        return;
    }
    const long long element = idx / (3 * NQF);
    const int local = (int)(idx - element * 3 * NQF);
    const int face = local / NQF;
    const int qf = local - face * NQF;
    double beta_x = 0.0;
    double beta_y = 0.0;
    for (int k = 0; k < NEL; ++k) {
        const double phi = face_basis[(face * NEL + k) * NQF + qf];
        beta_x += beta_coeffs[element * NEL + k] * phi;
        beta_y += beta_coeffs[(num_elements + element) * NEL + k] * phi;
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
        const long long edge_id = loc2glob_edge[element * 3 + face];
        if (edge_to_solve_edge[edge_id] < 0) {
            tau = 0.0;
            gamma = 0.0;
        }
    }
    const double face_measure = jacs_el_fc[element * 3 + face] * face_weights[qf];
    tau_out[idx] = tau * face_measure;
    gamma_out[idx] = gamma * face_measure;
}


// One thread per (chunk element, k < NEL): FP64 coefficient rows rounded once
// to WORK_REAL. Chunk element e is mesh element element_offset + e; the rows
// are chunk-local, every other array is mesh-wide (total_elements).
//   x_operator (E, operator_width) = [-J b_ref0 | -J b_ref1 | reaction | tau_w]
//   x_rhs      (E, 3*NQF + NEL)    = [gamma_w | J s]
// reaction is J*r (reaction_width = NEL) or J*reaction_scalar (width 1).
extern "C" __global__ void advection_tsle_tensor_rows(
        WORK_REAL* __restrict__ x_operator,
        WORK_REAL* __restrict__ x_rhs,
        const double* __restrict__ tau_w,
        const double* __restrict__ gamma_w,
        const double* __restrict__ aff_jacs,
        const double* __restrict__ inv_aff_mats_t,
        const double* __restrict__ source_coeffs,
        const double* __restrict__ beta_coeffs,
        const double* __restrict__ reaction_coeffs,
        const double reaction_scalar,
        const int reaction_is_scalar,
        const int operator_width,
        const long long num_elements,
        const long long element_offset,
        const long long total_elements)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements * NEL) {
        return;
    }
    const long long element = idx / NEL;
    const long long mesh_element = element_offset + element;
    const int k = (int)(idx - element * NEL);
    const double jac = aff_jacs[mesh_element];
    const double inv_t00 = inv_aff_mats_t[(mesh_element * 2 + 0) * 2 + 0];
    const double inv_t01 = inv_aff_mats_t[(mesh_element * 2 + 0) * 2 + 1];
    const double inv_t10 = inv_aff_mats_t[(mesh_element * 2 + 1) * 2 + 0];
    const double inv_t11 = inv_aff_mats_t[(mesh_element * 2 + 1) * 2 + 1];
    const double beta_x = beta_coeffs[mesh_element * NEL + k];
    const double beta_y = beta_coeffs[(total_elements + mesh_element) * NEL + k];
    WORK_REAL* row = x_operator + element * operator_width;
    row[k] = (WORK_REAL)(-jac * (beta_x * inv_t00 + beta_y * inv_t10));
    row[NEL + k] = (WORK_REAL)(-jac * (beta_x * inv_t01 + beta_y * inv_t11));
    int tail = 2 * NEL;
    if (reaction_is_scalar) {
        if (k == 0) {
            row[tail] = (WORK_REAL)(reaction_scalar * jac);
        }
        tail += 1;
    } else {
        row[tail + k] = (WORK_REAL)(jac * reaction_coeffs[mesh_element * NEL + k]);
        tail += NEL;
    }
    WORK_REAL* rhs_row = x_rhs + element * (3 * NQF + NEL);
    for (int q = k; q < 3 * NQF; q += NEL) {
        row[tail + q] = (WORK_REAL)tau_w[mesh_element * 3 * NQF + q];
        rhs_row[q] = (WORK_REAL)gamma_w[mesh_element * 3 * NQF + q];
    }
    rhs_row[3 * NQF + k] = (WORK_REAL)(jac * source_coeffs[mesh_element * NEL + k]);
}


// One thread per (element, trace row): the TSLE scatter applied to the
// precomputed Schur rows S_e = L_e A_e^{-1} [B_e | f_e] (E, 3*NTR, NCOLS) and
// trace masses G_e (E, 3, NTR, NTR), both in local trace orientation. Values
// are promoted to FP64 before orientation signs, boundary elimination, and
// accumulation. The same-edge (diagonal) block receives -S + G (+ gauge) in one
// atomicAdd per entry; other blocks are written by exactly one element. Schur
// rows and trace masses are chunk-local (mesh element element_offset + e).
extern "C" __global__ void advection_tsle_tensor_scatter_bsr(
        const int* __restrict__ bsr_indptr,
        double* __restrict__ data,
        double* __restrict__ rhs,
        const WORK_REAL* __restrict__ schur,
        const WORK_REAL* __restrict__ trace_mass,
        const double* __restrict__ tau_w,
        const long long* __restrict__ loc2glob_edge,
        const long long* __restrict__ edge_side_indices,
        const bool* __restrict__ orientations,
        const long long* __restrict__ interior_side_index,
        const long long* __restrict__ edge_to_solve_edge,
        const int* __restrict__ side_bsr_block_pos,
        const int* __restrict__ diagonal_bsr_block_pos,
        const double* __restrict__ boundary_trace,
        const long long num_elements,
        const long long element_offset)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements * 3 * NTR) {
        return;
    }
    const long long element = idx / (3 * NTR);
    const long long mesh_element = element_offset + element;
    const int task = (int)(idx - element * 3 * NTR);
    const int row_face = task / NTR;
    const int row_dof = task - row_face * NTR;
    const long long side_id = interior_side_index[mesh_element * 3 + row_face];
    const long long row_edge = loc2glob_edge[mesh_element * 3 + row_face];
    const long long row_solve_edge = edge_to_solve_edge[row_edge];
    if (side_id < 0 || row_solve_edge < 0) {
        return;
    }
    bool gauge_face = false;
#if RAW_CONFLICT_AVERAGED_UPWIND
    const long long other = edge_side_indices[row_edge * 2 + 1];
    gauge_face = edge_side_indices[row_edge * 2] == mesh_element * 3 + row_face && other >= 0;
    if (gauge_face) {
        const double* own_tau = tau_w + mesh_element * 3 * NQF + row_face * NQF;
        const double* other_tau = tau_w + (other / 3) * 3 * NQF + (other % 3) * NQF;
        for (int q = 0; q < NQF; ++q)
            gauge_face = gauge_face && own_tau[q] == 0.0 && other_tau[q] == 0.0;
    }
#endif
    const bool row_positive = orientations[mesh_element * 3 + row_face];
    const int row_local_dof = raw_local_trace_dof(row_positive, row_dof);
    const double row_sign = raw_trace_orientation_sign(row_positive, row_dof);
    const WORK_REAL* schur_row = schur + (element * 3 * NTR + row_face * NTR + row_local_dof) * NCOLS;
    double rhs_value = row_sign * (double)schur_row[NCOLS - 1];
    const int diagonal_pos = diagonal_bsr_block_pos[row_solve_edge];
    for (int col_face = 0; col_face < 3; ++col_face) {
        const long long col_edge = loc2glob_edge[mesh_element * 3 + col_face];
        const long long col_solve_edge = edge_to_solve_edge[col_edge];
        const bool col_positive = orientations[mesh_element * 3 + col_face];
        for (int col_dof = 0; col_dof < NTR; ++col_dof) {
            const int column = raw_trace_column_index(col_face, col_positive, col_dof);
            const double col_sign = raw_trace_orientation_sign(col_positive, col_dof);
            const double schur_value = row_sign * col_sign * (double)schur_row[column];
            if (col_solve_edge < 0) {
                rhs_value += schur_value * boundary_trace[col_edge * NTR + col_dof];
            } else if (col_solve_edge == row_solve_edge) {
                if (diagonal_pos >= 0) {
                    const int col_local_dof = raw_local_trace_dof(row_positive, col_dof);
                    double value = row_sign * col_sign * (double)trace_mass[
                        ((element * 3 + row_face) * NTR + row_local_dof) * NTR + col_local_dof];
                    if (gauge_face && row_dof == col_dof) {
                        value += 1.0;
                    }
                    const long long out = (((long long)bsr_indptr[row_solve_edge]
                        + diagonal_pos) * NTR + row_dof) * NTR + col_dof;
                    atomicAdd(&data[out], value - schur_value);
                }
            } else {
                const int block_pos = side_bsr_block_pos[side_id * 3 + col_face];
                const long long out = (((long long)bsr_indptr[row_solve_edge]
                    + block_pos) * NTR + row_dof) * NTR + col_dof;
                data[out] = -schur_value;
            }
        }
    }
    atomicAdd(&rhs[row_solve_edge * NTR + row_dof], rhs_value);
}
"""

_KERNEL_NAMES = (
    "advection_tsle_tensor_face_weights",
    "advection_tsle_tensor_rows",
    "advection_tsle_tensor_scatter_bsr",
)


class _CutensorContraction:
    """A cuTENSOR contraction ``C = A * B`` planned once for fixed operands.

    Descriptors encode shapes, strides, dtype, and pointer alignment, so a plan
    is reusable for any buffers with the same layout (the workspace keeps its
    buffers, which keeps the alignment too).
    """

    def __init__(self, cupy, a, mode_a: str, b, mode_b: str, c, mode_c: str, compute_desc: int):
        """Create descriptors and a plan for ``c[mode_c] = a[mode_a] * b[mode_b]``."""
        cutensor = require_cutensor()
        from cupy_backends.cuda.libs import cutensor as library

        self._library = library
        self._handle = cutensor._get_handle()
        descriptors = tuple(cutensor.create_tensor_descriptor(x) for x in (a, b, c))
        operation = cutensor.create_contraction(
            descriptors[0], cutensor.create_mode(*mode_a), library.OP_IDENTITY,
            descriptors[1], cutensor.create_mode(*mode_b), library.OP_IDENTITY,
            descriptors[2], cutensor.create_mode(*mode_c), library.OP_IDENTITY,
            compute_desc,
        )
        preference = cutensor.create_plan_preference(
            algo=library.ALGO_DEFAULT, jit_mode=library.JIT_MODE_NONE)
        size = library.estimateWorkspaceSize(
            self._handle.ptr, operation.ptr, preference.ptr, library.WORKSPACE_RECOMMENDED)
        self._plan = cutensor.create_plan(operation, preference, ws_limit=size)
        self._scratch = cupy.empty(max(int(size), 1), dtype=cupy.int8)
        self._scratch_size = int(size)
        scalar = np.float64 if c.dtype == np.float64 else np.float32
        self._alpha = cutensor._Scalar(1.0, scalar)
        self._beta = cutensor._Scalar(0.0, scalar)
        self._keep = (descriptors, operation, preference)

    def __call__(self, a, b, c) -> None:
        """Run the planned contraction on buffers laid out like the planned ones."""
        self._library.contract(
            self._handle.ptr, self._plan.ptr, self._alpha.ptr, a.data.ptr, b.data.ptr,
            self._beta.ptr, c.data.ptr, c.data.ptr, self._scratch.data.ptr, self._scratch_size)


def _gemm_bf16x9(cupy, x, y, out) -> None:
    """Row-major FP32 ``out = x @ y`` with cuBLAS BF16x9 FP32 emulation.

    ``x`` is ``(m, k)`` and ``y`` is ``(k, n)``, both C-contiguous; ``out`` is a
    C-contiguous ``(m, n)`` buffer. Computed as the column-major
    ``out^T = y^T x^T``.
    """
    from cupy_backends.cuda.libs import cublas

    m, k = (int(v) for v in x.shape)
    n = int(y.shape[1])
    one = np.ones(1, dtype=np.float32)
    zero = np.zeros(1, dtype=np.float32)
    handle = cupy.cuda.device.get_cublas_handle()
    mode = cublas.getPointerMode(handle)
    cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    try:
        cublas.gemmEx(
            handle, cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_N, n, m, k,
            one.ctypes.data, y.data.ptr, _CUDA_R_32F, n, x.data.ptr, _CUDA_R_32F, k,
            zero.ctypes.data, out.data.ptr, _CUDA_R_32F, n,
            _CUBLAS_COMPUTE_32F_EMULATED_16BFX9, _CUBLAS_GEMM_DEFAULT)
    finally:
        cublas.setPointerMode(handle, mode)


@dataclass
class RawAdvectionTsleTensorWorkspace:
    """Persistent buffers, reference tables, plans, and LU state.

    Buffers are reallocated when the discrete signature, working precision,
    engine, or solver layout changes, and reused (values overwritten)
    otherwise. ``nbytes`` counts every owned device buffer.
    """

    signature: tuple | None = None
    tables: dict[str, Any] = field(default_factory=dict)
    buffers: dict[str, Any] = field(default_factory=dict)
    contractions: dict[str, Any] = field(default_factory=dict)
    lu: BatchedLUWorkspace = field(default_factory=BatchedLUWorkspace)

    @property
    def nbytes(self) -> int:
        """Return the bytes of every owned device array."""
        arrays = list(self.tables.values()) + list(self.buffers.values())
        for name in ("pivots", "info", "matrix_pointers", "rhs_pointers", "pivot_pointers"):
            arrays.append(getattr(self.lu, name))
        return sum(int(getattr(array, "nbytes", 0)) for array in arrays if array is not None)

    def clear(self) -> None:
        """Release every device array and plan."""
        self.signature = None
        self.tables.clear()
        self.buffers.clear()
        self.contractions.clear()
        self.lu = BatchedLUWorkspace()


def _compile(cupy, *, nel: int, ntr: int, nqf: int, trace_orientation: str, advection_stabilization, dtype) -> dict[str, Any]:
    """Compile the FP64 helper kernels and the working-precision coop solve."""
    from hybridge.hdg.stabilization import is_conflict_averaged_upwind, upwind_factor

    factor = upwind_factor(advection_stabilization)
    if factor is None:
        raise ValueError("Unsupported tensor TSLE advection stabilization")
    key = (int(cupy.cuda.runtime.getDevice()), nel, ntr, nqf, trace_orientation, factor,
           is_conflict_averaged_upwind(advection_stabilization), np.dtype(dtype).name)
    cached = _MODULE_CACHE.get(key)
    if cached is not None:
        return dict(cached, jit_seconds=0.0)
    started = time.perf_counter()
    options = dict(nel=nel, ntr=ntr, ncols=3 * ntr + 1, nqf=nqf, lu_mode="coop",
                   trace_orientation=trace_orientation, advection_stabilization=advection_stabilization)
    work_type = "float" if np.dtype(dtype) == np.float32 else "double"
    helper_source = f"#define WORK_REAL {work_type}\n" + _kernel_source(
        _RAW_FUSED_TEMPLATE + "\n" + _TENSOR_KERNEL_TEMPLATE, **options)
    helpers = cupy.RawModule(code=helper_source, options=("--std=c++11",), name_expressions=_KERNEL_NAMES)
    solve_source = specialize_real_source(
        _kernel_source(_RAW_FUSED_TEMPLATE + "\n" + _TSLE_KERNEL_TEMPLATE, **options), dtype)
    solve_module = cupy.RawModule(code=solve_source, options=("--std=c++11",),
                                  name_expressions=("advection_tsle_solve",))
    kernels = {name.removeprefix("advection_tsle_tensor_"): helpers.get_function(name) for name in _KERNEL_NAMES}
    kernels["solve"] = solve_module.get_function("advection_tsle_solve")
    itemsize = np.dtype(dtype).itemsize
    kernels["solve"].max_dynamic_shared_size_bytes = int(
        (nel * nel + nel * (3 * ntr + 1) + max(_COOP_BLOCKS)) * itemsize + (nel + max(_COOP_BLOCKS)) * 4 + 256)
    _MODULE_CACHE[key] = kernels
    return dict(kernels, jit_seconds=time.perf_counter() - started)


def _solve_shared_bytes(nel: int, ncols: int, block: int, itemsize: int) -> int:
    """Dynamic shared bytes of the cooperative solve (see ``_shared_bytes_solve``)."""
    return int((nel * nel + nel * ncols + block) * itemsize + (nel + block) * 4 + 256)


def _reference_tables(cupy, *, cspace, trace_ref, advection_tensor, reaction_is_scalar: bool,
                      mass_is_diagonal: bool, dtype, column_major: bool, contraction: str) -> dict[str, Any]:
    """Build the static reference tables of stages 1 and 3 in FP64, then cast.

    ``column_major`` stores the dense-GEMM operator columns as ``(j, i)`` and
    response columns as ``(c, i)`` for the library LU solvers; the cuTENSOR
    tables keep their natural order and the plans permute the output.
    """
    nel, ntr = int(cspace.el_dof), int(cspace.edg_dof)
    nqf = int(trace_ref.weights.size)
    ncols = 3 * ntr + 1
    q = cspace.quad_data
    phi = cupy.asarray(trace_ref.bas_of_bd_quads, dtype=np.float64).reshape(3, nel, nqf)
    mu = cupy.asarray(trace_ref.bas1d_of_ref_edg_qds, dtype=np.float64).reshape(ntr, nqf)
    mass = cupy.asarray(q.MKrf, dtype=np.float64).reshape(nel, nel)
    if mass_is_diagonal:
        mass = cupy.diag(cupy.diag(mass))
    advection = cupy.asarray(advection_tensor, dtype=np.float64).reshape(2 * nel, nel, nel)
    if reaction_is_scalar:
        reaction = mass[None, :, :]
    else:
        reaction = cupy.asarray(q.weighted_triple_phi_flat, dtype=np.float64).reshape(nel, nel, nel)
    boundary = cupy.einsum("fiq,fjq->fqij", phi, phi).reshape(3 * nqf, nel, nel)
    operator = cupy.concatenate((advection, reaction, boundary), axis=0)
    if column_major and contraction != "cutensor":
        # Plain GEMMs write row-major (i, j) columns; cuTENSOR transposes
        # through its output modes instead.
        operator = operator.transpose(0, 2, 1)
    tables = {"operator": operator}
    phi_mu = cupy.einsum("fiq,tq->fqit", phi, mu)            # (3, NQF, NEL, NTR)
    mu_mu = cupy.einsum("aq,bq->qab", mu, mu)                # (NQF, NTR, NTR)
    if contraction == "cutensor":
        tables.update(phi_mu=phi_mu, source=mass, mu_mu=mu_mu)
    else:
        # Dense block-structured tables for plain GEMMs. Response rows are
        # [gamma_w (3*NQF) | J s (NEL)] against (i, c) or (c, i) columns.
        response = cupy.zeros((3 * nqf + nel, nel, ncols), dtype=np.float64)
        for face in range(3):
            response[face * nqf:(face + 1) * nqf, :, face * ntr:(face + 1) * ntr] = phi_mu[face]
        response[3 * nqf:, :, ncols - 1] = mass
        if column_major:
            response = response.transpose(0, 2, 1)
        lift = cupy.zeros((3 * nqf, 3, ntr, nel), dtype=np.float64)
        trace_mass = cupy.zeros((3 * nqf, 3, ntr, ntr), dtype=np.float64)
        for face in range(3):
            lift[face * nqf:(face + 1) * nqf, face] = phi_mu[face].transpose(0, 2, 1)
            trace_mass[face * nqf:(face + 1) * nqf, face] = mu_mu
        tables.update(response=response, lift=lift, trace_mass=trace_mass)
    return {
        name: cupy.ascontiguousarray(table.reshape(table.shape[0], -1) if name in {"operator", "response", "lift", "trace_mass"} else table, dtype=dtype)
        for name, table in tables.items()
    }


TENSOR_CHUNK_BYTES = 16 * 2 ** 30


def _chunk_bytes_per_element(nel: int, ntr: int, nqf: int) -> int:
    """FP64 bytes of the chunk-sized stage buffers, pivots and pointers per element."""
    ncols = 3 * ntr + 1
    reals = ((3 * nel + 3 * nqf) + (3 * nqf + nel)   # coefficient rows
             + nel * nel + nel * ncols                # operator, response
             + 3 * ntr * nel + 3 * ntr * ncols        # lift, Schur rows
             + 3 * ntr * ntr)                         # trace masses
    return 8 * reals + 4 * (nel + 1) + 3 * 8


def tensor_chunk_elements(num_elements: int, nel: int, ntr: int, nqf: int, *,
                          chunk_elements: int | None = None) -> int:
    """Return the elements per tensor TSLE pass.

    ``chunk_elements``, else ``HYBRIDGE_RAW_TENSOR_CHUNK_ELEMENTS``, else equal
    chunks whose stage buffers fit ``HYBRIDGE_RAW_TENSOR_CHUNK_GIB`` (16 GiB by
    default; one chunk up to about 270k elements at p=9).
    """
    import os

    num_elements = max(int(num_elements), 1)
    if chunk_elements is None:
        value = os.environ.get("HYBRIDGE_RAW_TENSOR_CHUNK_ELEMENTS", "").strip()
        chunk_elements = int(value) if value else None
    if chunk_elements is None:
        value = os.environ.get("HYBRIDGE_RAW_TENSOR_CHUNK_GIB", "").strip()
        budget = float(value) * 2 ** 30 if value else TENSOR_CHUNK_BYTES
        limit = max(1, int(budget // _chunk_bytes_per_element(int(nel), int(ntr), int(nqf))))
        chunks = -(-num_elements // limit)
        chunk_elements = -(-num_elements // chunks)
    if int(chunk_elements) < 1:
        raise ValueError("tensor chunk_elements must be positive")
    return min(int(chunk_elements), num_elements)


def tensor_workspace_bytes(num_elements: int, nel: int, ntr: int, nqf: int, *,
                           cache_local_response: bool = True, chunk_elements: int | None = None) -> int:
    """Return the FP64 device bytes of one tensor TSLE assembly for this signature.

    Counts the persistent stage buffers of :class:`RawAdvectionTsleTensorWorkspace`
    (with the field-reaction coefficient width, an upper bound) for one chunk
    (:func:`tensor_chunk_elements`), the mesh-wide face weights, the batched-LU
    pivots and pointers, and, with ``cache_local_response``, the FP64 response
    copy returned for reconstruction. Reference tables and the output ``data``
    are not counted (split3 does not count them either).
    """
    nel, ntr, nqf = int(nel), int(ntr), int(nqf)
    ncols = 3 * ntr + 1
    chunk = tensor_chunk_elements(num_elements, nel, ntr, nqf, chunk_elements=chunk_elements)
    mesh_wide = 8 * (6 * nqf + (nel * ncols if cache_local_response else 0))  # tau_w, gamma_w, response copy
    return int(num_elements) * mesh_wide + chunk * _chunk_bytes_per_element(nel, ntr, nqf)


def tensor_memory_sizes(space, trace_space, workspace=None) -> tuple[int, int]:
    """Return ``(workspace_bytes, available_bytes)`` for the memory-aware tensor choice.

    Mirrors :func:`hybridge.transport.tsle_bsr.split3_memory_sizes`: available
    memory is the driver's free memory plus CuPy's free pool blocks, plus a
    tensor workspace already allocated for this element count and size.
    """
    cupy = require_cupy()
    nqf = int(trace_space.weights.size)
    num_elements, nel = int(space.mesh.num_tri), int(space.el_dof)
    needed = tensor_workspace_bytes(num_elements, nel, space.order + 1, nqf)
    free, _total = cupy.cuda.runtime.memGetInfo()
    available = int(free) + int(cupy.get_default_memory_pool().free_bytes())
    tensor = getattr(workspace, "tensor", None)
    if tensor is not None and tensor.signature is not None and tensor.signature[3:5] == (num_elements, nel):
        available += tensor.nbytes
    return needed, available


def _device_compute_capability() -> str | None:
    """Return the current CUDA device's compute capability (e.g. ``"90"``), or ``None``."""
    try:
        import cupy

        return str(cupy.cuda.Device().compute_capability)
    except Exception:  # noqa: BLE001 - no device or no CuPy
        return None


def default_tensor_engines() -> tuple[str, str]:
    """Return contraction preference and measured local-solve selection.

    ``auto`` compares equivalent same-precision local solvers on actual sampled
    systems. Environment overrides retain explicit engine selection.
    """
    import os

    contraction = os.environ.get("HYBRIDGE_RAW_TENSOR_CONTRACTION", "").strip().lower()
    if not contraction:
        try:
            require_cutensor()
            contraction = "cutensor"
        except Exception:  # noqa: BLE001 - any load failure falls back to cuBLAS
            contraction = "cublas"
    from hybridge.runtime.env import getenv
    local_solver = getenv("RAW_TENSOR_LOCAL_SOLVER", "auto").strip().lower() or "auto"
    if contraction not in TENSOR_CONTRACTIONS:
        raise ValueError(f"HYBRIDGE_RAW_TENSOR_CONTRACTION must be one of {TENSOR_CONTRACTIONS}")
    if local_solver not in TENSOR_LOCAL_SOLVERS:
        raise ValueError(f"HYBRIDGE_RAW_TENSOR_LOCAL_SOLVER must be one of {TENSOR_LOCAL_SOLVERS}")
    return contraction, local_solver


def _strided(cupy, base, shape, strides, offset: int = 0):
    """Return a strided view of ``base`` with element strides and offset."""
    itemsize = base.dtype.itemsize
    flat = base.reshape(-1)[offset:]
    return cupy.lib.stride_tricks.as_strided(flat, shape=shape, strides=tuple(s * itemsize for s in strides))


def _compute_descriptor(dtype, compute: str) -> int:
    """Return the cuTENSOR compute descriptor for ``dtype`` and ``compute``."""
    require_cutensor()
    from cupy_backends.cuda.libs import cutensor as library

    if np.dtype(dtype) == np.float64:
        if compute != "default":
            raise ValueError("float64 tensor TSLE supports compute='default' only")
        return library.COMPUTE_DESC_64F
    return {"default": library.COMPUTE_DESC_32F, "3xtf32": library.COMPUTE_DESC_3xTF32,
            "tf32": library.COMPUTE_DESC_TF32}[compute]


def assemble_projected_advection_trace_system_eliminated_tsle_tensor(
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
        advection_sparse_offsets=None,
        advection_sparse_modes=None,
        advection_sparse_values0=None,
        advection_sparse_values1=None,
        use_sparse_advection: bool = False,
        mass_is_diagonal: bool = False,
        block_size="auto",
        matrix_format: str = "bsr",
        zero_boundary_flux: bool = False,
        workspace: RawAdvectionTsleTensorWorkspace | None = None,
        cache_local_response: bool = True,
        advection_stabilization=None,
        beta_face_basis=None,
        volume_table=None,
        beta_volume_values=None,
        beta_normal_flux=None,
        reaction_values=None,
        precision: str = "float32",
        contraction: str = "cutensor",
        compute: str = "default",
        local_solver: str = "coop",
        chunk_elements: int | None = None,
) -> RawAdvectionAssemblyResult:
    """Assemble the reduced face-BSR advection system with tensorized TSLE.

    Inputs and the returned result match
    :func:`hybridge.transport.tsle_bsr.assemble_projected_advection_trace_system_eliminated_tsle_bsr`
    (the sparse advection arguments are accepted and ignored: the dense
    reference tensor is contracted directly). ``block_size`` applies to the
    cooperative solve only.

    Parameters
    ----------
    beta_volume_values, beta_normal_flux, reaction_values : None
        Accepted for input compatibility with split3. Quadrature-point
        coefficients are unsupported and raise ``NotImplementedError``;
        use fused or split3 assembly for those inputs.
    precision : {"float32", "float64"}
        Working precision of the contractions, local LU, and Schur rows. Face
        tables and coefficient rows are formed in FP64; ``data`` and ``rhs``
        are always accumulated in FP64.
    contraction : {"cutensor", "cublas"}
        Engine for the stage 1 and stage 3 products.
    compute : {"default", "3xtf32", "tf32", "bf16x9"}
        ``"3xtf32"``/``"tf32"`` need ``contraction="cutensor"`` and FP32;
        ``"bf16x9"`` needs ``contraction="cublas"`` and FP32.
    local_solver : {"coop", "cublas", "magma", "fused"}
        Stage 2 batched LU/solve. ``"fused"`` needs ``precision="float64"``.
    chunk_elements : int, optional
        Elements per pass of stages 1-3 and the scatter (see
        :func:`tensor_chunk_elements`). The stage buffers are sized for one
        chunk, so the workspace stays bounded on large meshes.

    Returns
    -------
    RawAdvectionAssemblyResult
        FP64 ``data``/``rhs`` and, when ``cache_local_response`` is set, the
        FP64 response ``(E, NEL, NCOLS)`` for reconstruction. ``timings``
        holds ``raw.tsle_tensor.{prepare,build,solve,condense,scatter,device}``
        CUDA-event seconds, ``raw.tsle_tensor.workspace.bytes``, and the generic
        ``raw.kernel.device``.
    """
    if any(value is not None for value in (beta_volume_values, beta_normal_flux, reaction_values)):
        raise NotImplementedError(
            "tensor TSLE does not support quadrature-point coefficients; use "
            "raw_local_assembly='fused' or 'split3'"
        )
    cupy = require_cupy()
    validate_raw_cuda_supported(cspace, trace_ref, max_el_dof=55, max_order=9,
                                label="tensor TSLE advection assembly")
    if str(matrix_format).lower() not in {"bsr", "auto"}:
        raise ValueError("tensor TSLE requires matrix_format='bsr'")
    # beta_face_basis is accepted for input parity with split3; only the
    # solution-space table (a same-space advection) is supported here.
    # volume_table (the quadrature form of the same volume operator) is
    # accepted for input parity with split3 and ignored: this path builds the
    # operator by its own tensor contraction.
    del volume_table
    if tuple(getattr(beta_coeffs, "shape", ())[2:]) != (int(cspace.el_dof),) or (
            beta_face_basis is not None and beta_face_basis is not trace_ref.bas_of_bd_quads):
        raise NotImplementedError(
            "tensor TSLE supports only an advection field in the solution space; use "
            "raw_local_assembly='fused' or 'split3' for an advection field from another DG space"
        )
    if precision not in TENSOR_PRECISIONS:
        raise ValueError(f"precision must be one of {TENSOR_PRECISIONS}")
    if contraction not in TENSOR_CONTRACTIONS:
        raise ValueError(f"contraction must be one of {TENSOR_CONTRACTIONS}")
    if local_solver not in TENSOR_LOCAL_SOLVERS:
        raise ValueError(f"local_solver must be one of {TENSOR_LOCAL_SOLVERS}")
    if compute not in TENSOR_COMPUTES:
        raise ValueError(f"compute must be one of {TENSOR_COMPUTES}")
    if compute in {"3xtf32", "tf32"} and (contraction != "cutensor" or precision != "float32"):
        raise ValueError(f"compute={compute!r} needs contraction='cutensor' and precision='float32'")
    if compute == "bf16x9" and (contraction != "cublas" or precision != "float32"):
        raise ValueError("compute='bf16x9' needs contraction='cublas' and precision='float32'")
    if local_solver == "magma":
        from hybridge.linalg.gpu.magma_batched import lu_solve_batched_magma
    if local_solver == "fused":
        if precision != "float64":
            raise ValueError("local_solver='fused' needs precision='float64'")
        from hybridge.linalg.gpu.fused_batched_solve import solve_batched_fused

    dtype = np.dtype(precision)
    timings: dict[str, float] = {}
    wall_start = time.perf_counter()
    mesh_h = cspace.host.mesh
    num_elements = int(cspace.mesh.num_tri)
    nel, ntr = int(cspace.el_dof), int(cspace.edg_dof)
    nqf = int(trace_ref.weights.size)
    ncols = 3 * ntr + 1
    column_major = local_solver != "coop"
    chunk = tensor_chunk_elements(num_elements, nel, ntr, nqf, chunk_elements=chunk_elements)

    source_coeffs = cupy.ascontiguousarray(source_coeffs, dtype=np.float64)
    beta_coeffs = cupy.ascontiguousarray(beta_coeffs, dtype=np.float64)
    reaction_coeffs = (cupy.empty(1, dtype=np.float64) if reaction_is_scalar
                       else cupy.ascontiguousarray(reaction_coeffs, dtype=np.float64))
    if zero_boundary_flux:
        boundary_trace = cupy.zeros((mesh_h.bnd_edges_inds.size, ntr), dtype=np.float64)
    else:
        boundary_trace = cupy.ascontiguousarray(boundary_trace, dtype=np.float64)
    pattern = build_reduced_csr_pattern_raw(cspace, timings, matrix_format="bsr")
    data = cupy.zeros((pattern.num_blocks, ntr, ntr), dtype=np.float64)
    rhs = cupy.zeros(mesh_h.int_edges_inds.size * ntr, dtype=np.float64)
    boundary_trace_full = cupy.zeros((mesh_h.num_edg, ntr), dtype=np.float64)
    if mesh_h.bnd_edges_inds.size:
        boundary_trace_full[cspace.mesh.bnd_edges_inds] = boundary_trace

    trace_orientation = _raw_trace_orientation_mode(trace_ref)
    kernels = _compile(cupy, nel=nel, ntr=ntr, nqf=nqf, trace_orientation=trace_orientation,
                       advection_stabilization=advection_stabilization, dtype=dtype)
    timings["raw.tsle_tensor.jit"] = float(kernels["jit_seconds"])

    # Workspace: mesh-wide face weights (the upwind gauge reads neighbours across
    # chunks), chunk-sized stage buffers, tables, and plans for this signature.
    workspace = RawAdvectionTsleTensorWorkspace() if workspace is None else workspace
    operator_width = 2 * nel + (1 if reaction_is_scalar else nel) + 3 * nqf
    signature = (int(cupy.cuda.runtime.getDevice()), id(cspace), id(trace_ref), num_elements, nel, ntr, nqf,
                 dtype.name, contraction, compute, local_solver, bool(reaction_is_scalar), bool(mass_is_diagonal),
                 chunk)
    if workspace.signature != signature:
        workspace.clear()
        workspace.tables.update(_reference_tables(
            cupy, cspace=cspace, trace_ref=trace_ref, advection_tensor=advection_tensor,
            reaction_is_scalar=bool(reaction_is_scalar), mass_is_diagonal=bool(mass_is_diagonal),
            dtype=dtype, column_major=column_major, contraction=contraction))
        empty = lambda *shape, real=dtype: cupy.empty(shape, dtype=real)  # noqa: E731
        workspace.buffers.update(
            tau_w=empty(num_elements, 3, nqf, real=np.float64),
            gamma_w=empty(num_elements, 3, nqf, real=np.float64),
            x_operator=empty(chunk, operator_width),
            x_rhs=empty(chunk, 3 * nqf + nel),
            operator=empty(chunk, nel, nel),
            response=empty(chunk, ncols, nel) if column_major else empty(chunk, nel, ncols),
            lift=empty(chunk, 3 * ntr, nel),
            schur=empty(chunk, 3 * ntr, ncols),
            trace_mass=empty(chunk, 3, ntr, ntr),
        )
        if local_solver in {"fused", "auto"}:
            workspace.buffers["info"] = cupy.zeros(chunk, dtype=np.int32)
        workspace.signature = signature
    tables, buffers = workspace.tables, workspace.buffers

    if column_major:
        trace_modes, operator_modes, response_modes = "efti", "eji", "eci"
    else:
        trace_modes, operator_modes, response_modes = "eift", "eij", "eic"

    def views(count):
        """Prefix views of the chunk buffers for ``count`` elements, with strided aliases."""
        x_op, x_rhs = buffers["x_operator"][:count], buffers["x_rhs"][:count]
        response = buffers["response"][:count]
        view = dict(
            count=count, x_op=x_op, x_rhs=x_rhs, response=response, operator=buffers["operator"][:count],
            lift=buffers["lift"][:count], schur=buffers["schur"][:count],
            trace_mass=buffers["trace_mass"][:count],
            tau=_strided(cupy, x_op, (count, 3, nqf), (operator_width, nqf, 1), operator_width - 3 * nqf),
            gamma=_strided(cupy, x_rhs, (count, 3, nqf), (3 * nqf + nel, nqf, 1)),
            source=_strided(cupy, x_rhs, (count, nel), (3 * nqf + nel, 1), 3 * nqf),
        )
        if column_major:
            view["trace_columns"] = _strided(cupy, response, (count, 3, ntr, nel), (ncols * nel, ntr * nel, nel, 1))
            view["source_column"] = _strided(cupy, response, (count, nel), (ncols * nel, 1), (ncols - 1) * nel)
        else:
            view["trace_columns"] = _strided(cupy, response, (count, nel, 3, ntr), (nel * ncols, ncols, ntr, 1))
            view["source_column"] = _strided(cupy, response, (count, nel), (nel * ncols, ncols), ncols - 1)
        if "info" in buffers:
            view["info"] = buffers["info"][:count]
        return view

    if contraction == "cutensor":
        descriptor = _compute_descriptor(dtype, compute)
        operator_table = tables["operator"].reshape(operator_width, nel, nel)

        def plans_for(v):
            """cuTENSOR plans for chunks of ``v['count']`` elements (at most two sizes)."""
            plans = workspace.contractions.get(v["count"])
            if plans is None:
                lift = v["lift"].reshape(v["count"], 3, ntr, nel)
                plans = workspace.contractions[v["count"]] = dict(
                    operator=_CutensorContraction(cupy, v["x_op"], "eK", operator_table, "Kij",
                                                  v["operator"], operator_modes, descriptor),
                    trace=_CutensorContraction(cupy, v["gamma"], "efq", tables["phi_mu"], "fqit",
                                               v["trace_columns"], trace_modes, descriptor),
                    source=_CutensorContraction(cupy, v["source"], "ek", tables["source"], "ki",
                                                v["source_column"], "ei", descriptor),
                    lift=_CutensorContraction(cupy, v["tau"], "efq", tables["phi_mu"], "fqit",
                                              lift, "efti", descriptor),
                    schur=_CutensorContraction(cupy, v["lift"], "eri", v["response"], response_modes,
                                               v["schur"], "erc", descriptor),
                    trace_mass=_CutensorContraction(cupy, v["gamma"], "efq", tables["mu_mu"], "qab",
                                                    v["trace_mass"], "efab", descriptor),
                )
            return plans

        def build_products(v):
            """Stage 1 contractions: operator, trace columns, and source column."""
            plans = plans_for(v)
            plans["operator"](v["x_op"], operator_table, v["operator"])
            plans["trace"](v["gamma"], tables["phi_mu"], v["trace_columns"])
            plans["source"](v["source"], tables["source"], v["source_column"])

        def condense_products(v):
            """Stage 3 contractions: lift rows, Schur rows, and trace masses."""
            plans = plans_for(v)
            plans["lift"](v["tau"], tables["phi_mu"], v["lift"].reshape(v["count"], 3, ntr, nel))
            plans["schur"](v["lift"], v["response"], v["schur"])
            plans["trace_mass"](v["gamma"], tables["mu_mu"], v["trace_mass"])
    else:
        def gemm(x, y, out):
            """Row-major ``out = x @ y`` with the selected cuBLAS compute mode."""
            if compute == "bf16x9":
                _gemm_bf16x9(cupy, cupy.ascontiguousarray(x), y, out)
            else:
                cupy.matmul(x, y, out=out)

        def build_products(v):
            """Stage 1 GEMMs: operator and response columns (dense block tables)."""
            gemm(v["x_op"], tables["operator"], v["operator"].reshape(v["count"], -1))
            gemm(v["x_rhs"], tables["response"], v["response"].reshape(v["count"], -1))

        def condense_products(v):
            """Stage 3 GEMMs: lift rows, batched Schur rows, and trace masses."""
            count = v["count"]
            tau_rows = cupy.ascontiguousarray(v["tau"].reshape(count, 3 * nqf))
            gamma_rows = v["x_rhs"][:, :3 * nqf]
            gemm(tau_rows, tables["lift"], v["lift"].reshape(count, -1))
            right = v["response"].transpose(0, 2, 1) if column_major else v["response"]
            cupy.matmul(v["lift"], right, out=v["schur"])
            gemm(cupy.ascontiguousarray(gamma_rows), tables["trace_mass"], v["trace_mass"].reshape(count, -1))

    # Stage 2 launcher.
    itemsize = dtype.itemsize
    if local_solver == "coop":
        tune_key = (int(cupy.cuda.runtime.getDevice()), nel, ntr, dtype.name, num_elements)
        block = None if block_size in (None, "auto") else int(block_size)
        if block is not None and block not in _COOP_BLOCKS:
            raise ValueError(f"coop block_size must be 'auto' or one of {_COOP_BLOCKS}")

        def solve_launch(count, rows, block):
            """Launch the cooperative solve on the first ``count`` elements."""
            kernels["solve"]((count,), (block,), (buffers["operator"], rows, np.int64(count)),
                             shared_mem=_solve_shared_bytes(nel, ncols, block, itemsize))
    elif local_solver not in {"fused", "auto"}:
        solve_lu = lu_solve_batched_cublas if local_solver == "cublas" else lu_solve_batched_magma

    def face_weights_launch():
        """Launch the FP64 face-weight kernel on every mesh element."""
        total = num_elements * 3 * nqf
        kernels["face_weights"](
            ((total + _ELEMENTWISE_BLOCK - 1) // _ELEMENTWISE_BLOCK,), (_ELEMENTWISE_BLOCK,),
            (buffers["tau_w"], buffers["gamma_w"], beta_coeffs, trace_ref.bas_of_bd_quads, trace_ref.weights,
             cspace.mesh.normals, cspace.mesh.jacs_el_fc, cspace.mesh.loc2glob_edge, cspace.mesh.edge_side_indices,
             cspace.mesh.orientations, pattern.edge_to_solve_edge, np.int32(1 if zero_boundary_flux else 0),
             np.int64(num_elements)))

    def rows_launch(offset, v):
        """Launch the FP64 coefficient-row kernel on one chunk."""
        total = v["count"] * nel
        kernels["rows"](
            ((total + _ELEMENTWISE_BLOCK - 1) // _ELEMENTWISE_BLOCK,), (_ELEMENTWISE_BLOCK,),
            (v["x_op"], v["x_rhs"], buffers["tau_w"], buffers["gamma_w"], cspace.mesh.aff_jacs,
             cspace.mesh.inv_aff_mats_t, source_coeffs, beta_coeffs, reaction_coeffs, np.float64(reaction_scalar),
             np.int32(1 if reaction_is_scalar else 0), np.int32(operator_width), np.int64(v["count"]),
             np.int64(offset), np.int64(num_elements)))

    def scatter_launch(offset, v):
        """Scatter one chunk's Schur rows and trace masses into the face BSR system."""
        total = v["count"] * 3 * ntr
        kernels["scatter_bsr"](
            ((total + _SCATTER_BLOCK - 1) // _SCATTER_BLOCK,), (_SCATTER_BLOCK,),
            (pattern.indptr, data, rhs, v["schur"], v["trace_mass"], buffers["tau_w"],
             cspace.mesh.loc2glob_edge, cspace.mesh.edge_side_indices, cspace.mesh.orientations,
             pattern.interior_side_index, pattern.edge_to_solve_edge, pattern.side_csr_block_pos,
             pattern.mass_csr_block_pos, boundary_trace_full.reshape(-1), np.int64(v["count"]),
             np.int64(offset)))

    # Pick the coop block on a prefix grid of the first chunk; each trial reloads the prefix response.
    if local_solver == "coop":
        block = block if block is not None else _COOP_TUNING_CACHE.get(tune_key)
        if block is None and num_elements:
            first = views(min(chunk, num_elements))
            face_weights_launch()
            rows_launch(0, first)
            build_products(first)
            count = min(first["count"], _COOP_TUNE_LIMIT)
            original = first["response"][:count].copy()
            scratch = cupy.empty_like(original)
            properties = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.runtime.getDevice())
            shared_limit = int(properties.get("sharedMemPerBlockOptin", 48 * 1024))
            results = {}
            for candidate in _COOP_BLOCKS:
                if _solve_shared_bytes(nel, ncols, candidate, itemsize) > shared_limit:
                    continue
                samples = []
                for repeat in range(4):
                    scratch[...] = original
                    begin, end = cupy.cuda.Event(), cupy.cuda.Event()
                    begin.record()
                    solve_launch(count, scratch, candidate)
                    end.record()
                    end.synchronize()
                    if repeat:
                        samples.append(cupy.cuda.get_elapsed_time(begin, end))
                results[candidate] = statistics.median(samples)
            block = min(results, key=results.get)
            _COOP_TUNING_CACHE[tune_key] = block
            del original, scratch
        timings["raw.tsle_tensor.solve.block_size"] = float(block or 0)

    # Calibrate before timing production stages. The generic selector samples
    # real systems into private arrays and never factors the build buffers.
    if local_solver == "auto":
        from hybridge.linalg.gpu.tuned_batched_solve import select_batched_solver, solve_selected
        from hybridge.runtime.cuda_tuning import CandidateUnavailable, kernel_attributes
        props = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.runtime.getDevice())
        shared_limit = int(props.get("sharedMemPerBlockOptin") or props["sharedMemPerBlock"])
        coop_storage = {}
        def cooperative(a, b, info, threads):
            """Adapt column-major systems to the native cooperative layout."""
            shared = _solve_shared_bytes(nel, ncols, threads, itemsize)
            if shared > shared_limit or threads > kernel_attributes(cupy, kernels['solve'])['max_threads_per_block']:
                raise CandidateUnavailable('cooperative solve resource limits')
            if a is None:
                return
            # cuBLAS/fused arrays are column-major within a system; the
            # cooperative kernel uses row-major. Include these copies in its
            # measured cost and reuse the allocations within a chunk size.
            count = a.shape[0]
            if coop_storage.get('count') != count:
                coop_storage.update(count=count, a=cupy.empty((count, nel, nel), dtype=dtype),
                                    b=cupy.empty((count, nel, ncols), dtype=dtype))
            ar, br = coop_storage['a'], coop_storage['b']
            ar[...] = a.transpose(0, 2, 1)
            br[...] = b.transpose(0, 2, 1)
            kernels['solve']((count,), (threads,), (ar, br, np.int64(count)), shared_mem=shared)
            b[...] = br.transpose(0, 2, 1)
            if info is not None:
                info[...] = (~cupy.all(cupy.isfinite(br), axis=(1, 2))).astype(cupy.int32)
        if num_elements:
            first = views(min(chunk, num_elements, 2048))
            def prepare_sample():
                """Build a bounded prefix only when calibration needs inputs."""
                face_weights_launch()
                rows_launch(0, first)
                build_products(first)
            selected_solver, tuning_report = select_batched_solver(
                first['operator'], first['response'], cooperative=cooperative,
                prepare_inputs=prepare_sample,
                cooperative_source=hashlib.sha256((_TSLE_KERNEL_TEMPLATE + _RAW_FUSED_TEMPLATE).encode()).hexdigest(),
                workload=dict(order=cspace.order, element_basis=cspace.host.reference.basis_type,
                              trace_basis=trace_ref.kind, reaction_is_scalar=bool(reaction_is_scalar),
                              contraction=contraction, compute=compute,
                              chunk_bucket=max(0, (chunk-1).bit_length()),
                              mesh_bucket=max(0, (num_elements-1).bit_length())))
        else:
            selected_solver, tuning_report = 'cublas', dict(reused=False, wall_seconds=0.)
        timings['raw.tsle_tensor.autotune.wall'] = tuning_report['wall_seconds']
        timings['raw.tsle_tensor.autotune.reused'] = float(tuning_report['reused'])
        # Numeric IDs preserve the timings mapping's float-only contract;
        # complete variant/configuration and rejection records live in the cache.
        timings['raw.tsle_tensor.solve.algorithm'] = float(
            {'cublas': 0, 'magma': 1, 'coop': 2, 'fused': 3}[selected_solver.split(':')[0]])

    def solve_stage(v):
        """Stage 2: factor every ``A_e`` of the chunk and overwrite its response in place."""
        if local_solver == "auto":
            solve_selected(selected_solver, v['operator'], v['response'], info=v['info'],
                           workspace=workspace.lu, cooperative=cooperative)
        elif local_solver == "coop":
            solve_launch(v["count"], v["response"], block)
        elif local_solver == "fused":
            solve_batched_fused(v["operator"], v["response"], info=v["info"])
        else:
            solve_lu(v["operator"], v["response"], trans=False, workspace=workspace.lu, check_info=False)

    local_response = (cupy.empty((num_elements, nel, ncols), dtype=np.float64)
                      if cache_local_response else None)
    timings["raw.input_prepare"] = time.perf_counter() - wall_start
    stream = cupy.cuda.get_current_stream()
    stages = ("prepare", "build", "solve", "condense", "scatter")
    marks = []
    info_nonzero = cupy.zeros((), dtype=np.int64)
    launch_start = time.perf_counter()
    weights = [cupy.cuda.Event(), cupy.cuda.Event()]
    weights[0].record(stream)
    face_weights_launch()
    weights[1].record(stream)
    for offset in range(0, num_elements, chunk):
        v = views(min(chunk, num_elements - offset))
        events = [cupy.cuda.Event() for _ in range(6)]
        events[0].record(stream)
        rows_launch(offset, v)
        events[1].record(stream)
        build_products(v)
        events[2].record(stream)
        solve_stage(v)
        events[3].record(stream)
        condense_products(v)
        events[4].record(stream)
        scatter_launch(offset, v)
        events[5].record(stream)
        marks.append(events)
        if local_solver in {"fused", "auto"}:
            info_nonzero += cupy.count_nonzero(v["info"])
        elif local_solver != "coop":
            info_nonzero += cupy.count_nonzero(workspace.lu.info)
        if local_response is not None:
            solved = v["response"].transpose(0, 2, 1) if column_major else v["response"]
            local_response[offset:offset + v["count"]] = solved
    done = cupy.cuda.Event()
    done.record(stream)
    done.synchronize()
    seconds = dict.fromkeys(stages, 0.0)
    seconds["prepare"] = cupy.cuda.get_elapsed_time(*weights) / 1000.0
    for events in marks:
        for index, name in enumerate(stages):
            seconds[name] += cupy.cuda.get_elapsed_time(events[index], events[index + 1]) / 1000.0
    for name in stages:
        timings[f"raw.tsle_tensor.{name}"] = seconds[name]
    timings["raw.tsle_tensor.device"] = sum(seconds.values())
    timings["raw.tsle_tensor.chunks"] = float(len(marks))
    timings["raw.tsle_tensor.chunk_elements"] = float(chunk)
    timings["raw.tsle_tensor.workspace.bytes"] = float(workspace.nbytes)
    timings["raw.tsle_tensor.local_response.bytes"] = float(0 if local_response is None else local_response.nbytes)
    if local_solver != "coop":
        timings["raw.tsle_tensor.solve.info_nonzero"] = float(info_nonzero)
    if local_solver != "coop" and bool(info_nonzero.item()):
        raise np.linalg.LinAlgError('Tensor local solve reported a failed system')
    timings["raw.bsr_kernel"] = timings["raw.tsle_tensor.device"]
    timings["raw.kernel.device"] = timings["raw.tsle_tensor.device"]
    timings["raw.kernel.wall"] = time.perf_counter() - launch_start
    timings["raw.total"] = time.perf_counter() - wall_start
    timings["raw.wall_total"] = timings["raw.total"]
    return RawAdvectionAssemblyResult(
        advection_stabilization=advection_stabilization,
        data=data,
        rhs=rhs,
        boundary_trace=boundary_trace,
        timings=timings,
        indptr=pattern.indptr,
        indices=pattern.indices,
        matrix_format="bsr",
        csr_pattern=pattern,
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
        use_sparse_advection=bool(use_sparse_advection),
        mass_is_diagonal=bool(mass_is_diagonal),
        local_response=local_response,
        lu_mode="coop",
        zero_boundary_flux=bool(zero_boundary_flux),
    )


__all__ = [
    "RawAdvectionTsleTensorWorkspace",
    "default_tensor_engines",
    "tensor_chunk_elements",
    "tensor_memory_sizes",
    "tensor_workspace_bytes",
    "TENSOR_COMPUTES",
    "TENSOR_CONTRACTIONS",
    "TENSOR_LOCAL_SOLVERS",
    "TENSOR_PRECISIONS",
    "assemble_projected_advection_trace_system_eliminated_tsle_tensor",
]
