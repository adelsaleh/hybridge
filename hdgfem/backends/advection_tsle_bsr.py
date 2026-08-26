"""Tri-stage raw-CUDA local-elimination assembly for advection face BSR.

TSLE-BSR deliberately separates the element operator construction, pivoted
local solves, and Schur/scatter work.  The mathematical kernels are shared with
the established fused assembler; only storage and launch geometry differ.
"""

from __future__ import annotations

from dataclasses import dataclass
import statistics
import time
from typing import Any

import numpy as np

from .advection_raw_cuda import (
    RawAdvectionAssemblyResult,
    _RAW_FUSED_TEMPLATE,
    _kernel_source,
    _raw_trace_orientation_mode,
    build_reduced_csr_pattern_raw,
    validate_raw_cuda_supported,
)
from .cupy import require_cupy
from .raw_cuda import RawCudaBlockSize


_TSLE_KERNEL_NAMES = (
    "advection_tsle_build",
    "advection_tsle_solve",
    "advection_tsle_scatter_bsr",
)
_TSLE_BLOCK_CANDIDATES = (32, 64, 128, 256)
_TSLE_TUNE_LIMIT = 32_768
_TSLE_TUNE_WARMUPS = 1
_TSLE_TUNE_REPEATS = 3


@dataclass
class RawAdvectionTsleWorkspace:
    """Persistent FP64 element workspaces used by TSLE-BSR."""

    local_operator: Any | None = None
    local_response: Any | None = None
    face_flux: Any | None = None
    signature: tuple[int, ...] | None = None

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
        self.local_operator = cupy.empty((num_elements, nel, nel), dtype=cupy.float64)
        self.local_response = cupy.empty((num_elements, nel, ncols), dtype=cupy.float64)
        self.face_flux = cupy.empty((num_elements, 2, 3, nqf), dtype=cupy.float64)
        self.signature = signature

    @property
    def nbytes(self) -> int:
        """Return bytes owned by the persistent stage workspace."""
        return sum(
            int(getattr(array, "nbytes", 0))
            for array in (self.local_operator, self.local_response, self.face_flux)
        )

    def clear(self) -> None:
        """Release references to all persistent device arrays."""
        self.local_operator = None
        self.local_response = None
        self.face_flux = None
        self.signature = None


@dataclass(frozen=True)
class _TsleKernels:
    build: Any
    solve: Any
    scatter: Any
    jit_seconds: float


_TSLE_MODULE_CACHE: dict[tuple[Any, ...], _TsleKernels] = {}
_TSLE_TUNING_CACHE: dict[tuple[Any, ...], tuple[int, int, int]] = {}


_TSLE_KERNEL_TEMPLATE = r"""
extern "C" __global__ void advection_tsle_build(
        double* __restrict__ element_operator,
        double* __restrict__ element_response,
        double* __restrict__ element_face_flux,
        const long long* __restrict__ loc2glob_edge,
        const long long* __restrict__ edge_to_solve_edge,
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
    double* local_operator = reinterpret_cast<double*>(shared_raw);
    double* local_response = local_operator + NEL * NEL;
    double* tau_face = local_response + NEL * NCOLS;
    double* gamma_face = tau_face + 3 * NQF;
    double* source_cache = gamma_face + 3 * NQF;
    double* beta_x_cache = source_cache + NEL;
    double* beta_y_cache = beta_x_cache + NEL;
    double* beta_ref0_cache = beta_y_cache + NEL;
    double* beta_ref1_cache = beta_ref0_cache + NEL;
    double* reaction_cache = beta_ref1_cache + NEL;

    const long long element = blockIdx.x;
    if (element >= num_elements) {
        return;
    }
    assemble_projected_local_advection_raw(
        local_operator, local_response, tau_face, gamma_face,
        source_cache, beta_x_cache, beta_y_cache,
        beta_ref0_cache, beta_ref1_cache, reaction_cache,
        aff_jacs, inv_aff_mats_t, jacs_el_fc, normals,
        mass_matrix, reaction_triples, advection_tensor,
        advection_sparse_offsets, advection_sparse_modes,
        advection_sparse_values0, advection_sparse_values1,
        use_sparse_advection, mass_is_diagonal,
        face_basis, face_weights, trace_basis,
        source_coeffs, beta_coeffs, reaction_coeffs,
        reaction_scalar, reaction_is_scalar, loc2glob_edge, edge_to_solve_edge,
        zero_boundary_flux, num_elements, element);

    for (int idx = threadIdx.x; idx < NEL * NEL; idx += blockDim.x) {
        element_operator[element * (NEL * NEL) + idx] = local_operator[idx];
    }
    for (int idx = threadIdx.x; idx < NEL * NCOLS; idx += blockDim.x) {
        element_response[element * (NEL * NCOLS) + idx] = local_response[idx];
    }
    for (int idx = threadIdx.x; idx < 3 * NQF; idx += blockDim.x) {
        const long long base = element * (6 * NQF);
        element_face_flux[base + idx] = tau_face[idx];
        element_face_flux[base + 3 * NQF + idx] = gamma_face[idx];
    }
}


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

    factor_local_lu_coop_pivot_scale_raw(
        local_operator, pivots, pivot_abs, pivot_rows);
    solve_all_columns_raw(local_operator, pivots, local_response);

    for (int idx = threadIdx.x; idx < NEL * NCOLS; idx += blockDim.x) {
        element_response[element * (NEL * NCOLS) + idx] = local_response[idx];
    }
}


extern "C" __global__ void advection_tsle_scatter_bsr(
        const int* __restrict__ bsr_indptr,
        double* __restrict__ data,
        double* __restrict__ rhs,
        const double* __restrict__ element_response,
        const double* __restrict__ element_face_flux,
        const long long* __restrict__ loc2glob_edge,
        const bool* __restrict__ orientations,
        const long long* __restrict__ interior_side_index,
        const long long* __restrict__ edge_to_solve_edge,
        const int* __restrict__ side_bsr_block_pos,
        const int* __restrict__ diagonal_bsr_block_pos,
        const double* __restrict__ jacs_el_fc,
        const double* __restrict__ face_basis,
        const double* __restrict__ face_weights,
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
        const int row_dof = task - row_face * NTR;
        const long long side_id = interior_side_index[element * 3 + row_face];
        const long long row_edge = loc2glob_edge[element * 3 + row_face];
        const long long row_solve_edge = edge_to_solve_edge[row_edge];
        if (side_id < 0 || row_solve_edge < 0) {
            continue;
        }

        const bool row_positive = orientations[element * 3 + row_face];
        const int row_local_dof = raw_local_trace_dof(row_positive, row_dof);
        const double row_sign = raw_trace_orientation_sign(row_positive, row_dof);
        const double face_jac = jacs_el_fc[element * 3 + row_face];
        double rhs_value = 0.0;
        for (int i = 0; i < NEL; ++i) {
            double lift = 0.0;
            for (int qf = 0; qf < NQF; ++qf) {
                const double mu = row_sign * trace_basis[row_local_dof * NQF + qf];
                const double phi = face_basis[(row_face * NEL + i) * NQF + qf];
                lift += face_jac * tau_face[row_face * NQF + qf]
                    * face_weights[qf] * mu * phi;
            }
            lift_rows[task * NEL + i] = lift;
            rhs_value += lift * response[i * NCOLS + (NCOLS - 1)];
        }

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

        const int diagonal_pos = diagonal_bsr_block_pos[row_solve_edge];
        if (diagonal_pos >= 0) {
            for (int col_dof = 0; col_dof < NTR; ++col_dof) {
                const int col_local_dof = raw_local_trace_dof(row_positive, col_dof);
                const double col_sign = raw_trace_orientation_sign(row_positive, col_dof);
                double mass_value = 0.0;
                for (int qf = 0; qf < NQF; ++qf) {
                    const double mu_row = row_sign
                        * trace_basis[row_local_dof * NQF + qf];
                    const double mu_col = col_sign
                        * trace_basis[col_local_dof * NQF + qf];
                    mass_value += face_jac * gamma_face[row_face * NQF + qf]
                        * face_weights[qf] * mu_row * mu_col;
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


def _shared_bytes_build(nel: int, ncols: int, nqf: int) -> int:
    """Return dynamic shared bytes for local-operator construction."""
    return int((nel * nel + nel * ncols + 6 * nqf + 6 * nel) * 8 + 256)


def _shared_bytes_solve(nel: int, ncols: int, block_size: int) -> int:
    """Return dynamic shared bytes for cooperative LU/all-column solve."""
    doubles = nel * nel + nel * ncols + block_size
    ints = nel + block_size
    return int(doubles * 8 + ints * 4 + 256)


def _shared_bytes_scatter(nel: int, ntr: int) -> int:
    """Return dynamic shared bytes for lift/Schur/BSR scatter."""
    return int((3 * ntr * nel + 3 * ntr * ntr) * 8 + 256)


def _device_property(properties: dict, name: str, default: int) -> int:
    """Read a CUDA property across CuPy string/byte-key variants."""
    value = properties.get(name, properties.get(name.encode(), default))
    return int(value)


def _compile_kernels(cupy, *, nel: int, ntr: int, nqf: int, trace_orientation: str) -> _TsleKernels:
    """Compile or reuse the three kernels for one discrete CUDA signature."""
    device_id = int(cupy.cuda.runtime.getDevice())
    key = (device_id, int(nel), int(ntr), int(nqf), str(trace_orientation))
    cached = _TSLE_MODULE_CACHE.get(key)
    if cached is not None:
        return _TsleKernels(cached.build, cached.solve, cached.scatter, 0.0)

    source = _kernel_source(
        _RAW_FUSED_TEMPLATE + "\n" + _TSLE_KERNEL_TEMPLATE,
        nel=nel,
        ntr=ntr,
        ncols=3 * ntr + 1,
        nqf=nqf,
        lu_mode="coop",
        trace_orientation=trace_orientation,
    )
    started = time.perf_counter()
    module = cupy.RawModule(
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
    )
    max_shared = max(
        _shared_bytes_build(nel, 3 * ntr + 1, nqf),
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


def _event_seconds(cupy, launch) -> float:
    """Time one launch on the current stream with CUDA events."""
    stream = cupy.cuda.get_current_stream()
    begin = cupy.cuda.Event()
    end = cupy.cuda.Event()
    begin.record(stream)
    launch()
    end.record(stream)
    end.synchronize()
    return float(cupy.cuda.get_elapsed_time(begin, end)) / 1000.0


def _candidate_median(
        cupy,
        prepare,
        launch,
        *,
        warmups: int = _TSLE_TUNE_WARMUPS,
        repeats: int = _TSLE_TUNE_REPEATS,
) -> float:
    """Return a prepared launch's median CUDA-event time."""
    samples: list[float] = []
    for repeat in range(warmups + repeats):
        prepare()
        elapsed = _event_seconds(cupy, launch)
        if repeat >= warmups:
            samples.append(elapsed)
    return float(statistics.median(samples))


def _select_launch_blocks(
        cupy,
        *,
        requested: RawCudaBlockSize,
        tune_key: tuple[Any, ...],
        tune_elements: int,
        num_elements: int,
        kernels: _TsleKernels,
        build_args: tuple[Any, ...],
        solve_args: tuple[Any, ...],
        scatter_args: tuple[Any, ...],
        data,
        rhs,
        nel: int,
        ntr: int,
        ncols: int,
        nqf: int,
        timings: dict[str, float],
) -> tuple[int, int, int]:
    """Select explicit or sampled/full-mesh stage launch sizes."""
    if not (requested is None or (isinstance(requested, str) and requested.lower() == "auto")):
        block_size = int(requested)
        if block_size not in _TSLE_BLOCK_CANDIDATES:
            raise ValueError("TSLE-BSR block size must be 'auto' or one of 32, 64, 128, 256")
        return block_size, block_size, block_size

    cached = _TSLE_TUNING_CACHE.get(tune_key)
    if cached is not None:
        timings["raw.tsle.autotune.reused"] = 1.0
        return cached

    properties = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.runtime.getDevice())
    max_threads = _device_property(properties, "maxThreadsPerBlock", 1024)
    max_shared = _device_property(properties, "sharedMemPerBlockOptin", 0)
    if max_shared <= 0:
        max_shared = _device_property(properties, "sharedMemPerBlock", 48 * 1024)
    candidates = [
        value
        for value in _TSLE_BLOCK_CANDIDATES
        if value <= max_threads
        and _shared_bytes_build(nel, ncols, nqf) <= max_shared
        and _shared_bytes_solve(nel, ncols, value) <= max_shared
        and _shared_bytes_scatter(nel, ntr) <= max_shared
    ]
    if not candidates:
        raise RuntimeError("no TSLE-BSR launch candidate fits this CUDA device")

    tune_start = time.perf_counter()
    grid = (int(tune_elements),)
    full_grid = (int(num_elements),)
    validate_full = num_elements > tune_elements

    build_times: dict[int, float] = {}
    for block in candidates:
        launch = lambda block=block: kernels.build(
            grid, (block,), build_args,
            shared_mem=_shared_bytes_build(nel, ncols, nqf),
        )
        build_times[block] = _candidate_median(cupy, lambda: None, launch)
        timings[f"raw.tsle.autotune.build.{block}"] = build_times[block]
    build_finalists = sorted(build_times, key=build_times.get)[:2]
    if validate_full:
        full_build_times: dict[int, float] = {}
        for block in build_finalists:
            launch = lambda block=block: kernels.build(
                full_grid, (block,), build_args,
                shared_mem=_shared_bytes_build(nel, ncols, nqf),
            )
            full_build_times[block] = _candidate_median(cupy, lambda: None, launch)
            timings[f"raw.tsle.autotune.full.build.{block}"] = full_build_times[block]
        build_block = min(full_build_times, key=full_build_times.get)
    else:
        build_block = build_finalists[0]

    def prepare_solve() -> None:
        """Rebuild sampled local systems before a destructive solve trial."""
        kernels.build(
            grid, (build_block,), build_args,
            shared_mem=_shared_bytes_build(nel, ncols, nqf),
        )

    solve_times: dict[int, float] = {}
    for block in candidates:
        launch = lambda block=block: kernels.solve(
            grid, (block,), solve_args,
            shared_mem=_shared_bytes_solve(nel, ncols, block),
        )
        solve_times[block] = _candidate_median(cupy, prepare_solve, launch)
        timings[f"raw.tsle.autotune.solve.{block}"] = solve_times[block]
    solve_finalists = sorted(solve_times, key=solve_times.get)[:2]
    if validate_full:
        def prepare_solve_full() -> None:
            """Rebuild full-mesh local systems before a solve finalist."""
            kernels.build(
                full_grid, (build_block,), build_args,
                shared_mem=_shared_bytes_build(nel, ncols, nqf),
            )

        full_solve_times: dict[int, float] = {}
        for block in solve_finalists:
            launch = lambda block=block: kernels.solve(
                full_grid, (block,), solve_args,
                shared_mem=_shared_bytes_solve(nel, ncols, block),
            )
            full_solve_times[block] = _candidate_median(cupy, prepare_solve_full, launch)
            timings[f"raw.tsle.autotune.full.solve.{block}"] = full_solve_times[block]
        solve_block = min(full_solve_times, key=full_solve_times.get)
    else:
        solve_block = solve_finalists[0]

    def prepare_scatter() -> None:
        """Prepare sampled solved columns and zero scatter outputs."""
        kernels.build(
            grid, (build_block,), build_args,
            shared_mem=_shared_bytes_build(nel, ncols, nqf),
        )
        kernels.solve(
            grid, (solve_block,), solve_args,
            shared_mem=_shared_bytes_solve(nel, ncols, solve_block),
        )
        data.fill(0.0)
        rhs.fill(0.0)

    scatter_times: dict[int, float] = {}
    for block in candidates:
        launch = lambda block=block: kernels.scatter(
            grid, (block,), scatter_args,
            shared_mem=_shared_bytes_scatter(nel, ntr),
        )
        scatter_times[block] = _candidate_median(cupy, prepare_scatter, launch)
        timings[f"raw.tsle.autotune.scatter.{block}"] = scatter_times[block]
    scatter_finalists = sorted(scatter_times, key=scatter_times.get)[:2]
    if validate_full:
        def prepare_scatter_full() -> None:
            """Prepare full-mesh solved columns and zero scatter outputs."""
            kernels.build(
                full_grid, (build_block,), build_args,
                shared_mem=_shared_bytes_build(nel, ncols, nqf),
            )
            kernels.solve(
                full_grid, (solve_block,), solve_args,
                shared_mem=_shared_bytes_solve(nel, ncols, solve_block),
            )
            data.fill(0.0)
            rhs.fill(0.0)

        full_scatter_times: dict[int, float] = {}
        for block in scatter_finalists:
            launch = lambda block=block: kernels.scatter(
                full_grid, (block,), scatter_args,
                shared_mem=_shared_bytes_scatter(nel, ntr),
            )
            full_scatter_times[block] = _candidate_median(cupy, prepare_scatter_full, launch)
            timings[f"raw.tsle.autotune.full.scatter.{block}"] = full_scatter_times[block]
        scatter_block = min(full_scatter_times, key=full_scatter_times.get)
    else:
        scatter_block = scatter_finalists[0]

    selected = (build_block, solve_block, scatter_block)
    _TSLE_TUNING_CACHE[tune_key] = selected
    cupy.cuda.get_current_stream().synchronize()
    timings["raw.tsle.autotune.reused"] = 0.0
    timings["raw.tsle.autotune.wall"] = time.perf_counter() - tune_start
    return selected


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
) -> RawAdvectionAssemblyResult:
    """Assemble a reduced face-BSR system with the TSLE three-stage pipeline."""
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
    ncols = 3 * ntr + 1

    source_coeffs = cupy.ascontiguousarray(source_coeffs, dtype=cupy.float64)
    beta_coeffs = cupy.ascontiguousarray(beta_coeffs, dtype=cupy.float64)
    reaction_coeffs = (
        cupy.empty(1, dtype=cupy.float64)
        if reaction_is_scalar
        else cupy.ascontiguousarray(reaction_coeffs, dtype=cupy.float64)
    )
    advection_tensor = cupy.ascontiguousarray(advection_tensor, dtype=cupy.float64)
    advection_sparse_offsets = cupy.ascontiguousarray(advection_sparse_offsets, dtype=cupy.int32)
    advection_sparse_modes = cupy.ascontiguousarray(advection_sparse_modes, dtype=cupy.int32)
    advection_sparse_values0 = cupy.ascontiguousarray(advection_sparse_values0, dtype=cupy.float64)
    advection_sparse_values1 = cupy.ascontiguousarray(advection_sparse_values1, dtype=cupy.float64)

    if zero_boundary_flux:
        boundary_trace = cupy.zeros((mesh_h.bnd_edges_inds.size, ntr), dtype=cupy.float64)
    else:
        boundary_trace = cupy.ascontiguousarray(boundary_trace, dtype=cupy.float64)

    pattern_start = time.perf_counter()
    pattern = build_reduced_csr_pattern_raw(cspace, timings, matrix_format="bsr")
    timings["raw.tsle.pattern.wrapper"] = time.perf_counter() - pattern_start
    data = cupy.zeros((pattern.num_blocks, ntr, ntr), dtype=cupy.float64)
    rhs = cupy.zeros(mesh_h.int_edges_inds.size * ntr, dtype=cupy.float64)
    boundary_trace_full = cupy.zeros((mesh_h.num_edg, ntr), dtype=cupy.float64)
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
    kernels = _compile_kernels(
        cupy,
        nel=nel,
        ntr=ntr,
        nqf=nqf,
        trace_orientation=trace_orientation,
    )
    timings["raw.tsle.jit"] = float(kernels.jit_seconds)

    build_args = (
        workspace.local_operator,
        workspace.local_response,
        workspace.face_flux,
        cspace.mesh.loc2glob_edge,
        pattern.edge_to_solve_edge,
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
        np.float64(reaction_scalar),
        np.int32(1 if reaction_is_scalar else 0),
        np.int32(1 if zero_boundary_flux else 0),
        np.int64(num_elements),
    )
    solve_args = (
        workspace.local_operator,
        workspace.local_response,
        np.int64(num_elements),
    )
    scatter_args = (
        pattern.indptr,
        data,
        rhs,
        workspace.local_response,
        workspace.face_flux,
        cspace.mesh.loc2glob_edge,
        cspace.mesh.orientations,
        pattern.interior_side_index,
        pattern.edge_to_solve_edge,
        pattern.side_csr_block_pos,
        pattern.mass_csr_block_pos,
        cspace.mesh.jacs_el_fc,
        trace_ref.bas_of_bd_quads,
        trace_ref.weights,
        trace_ref.bas1d_of_ref_edg_qds,
        boundary_trace_full.reshape(-1),
        np.int64(num_elements),
    )

    properties = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.runtime.getDevice())
    tuning_key = (
        int(cupy.cuda.runtime.getDevice()),
        _device_property(properties, "major", 0),
        _device_property(properties, "minor", 0),
        nel,
        ntr,
        nqf,
        trace_orientation,
        bool(use_sparse_advection),
        bool(mass_is_diagonal),
        bool(reaction_is_scalar),
        bool(zero_boundary_flux),
        min(num_elements, _TSLE_TUNE_LIMIT),
        num_elements,
    )
    selected = _select_launch_blocks(
        cupy,
        requested=block_size,
        tune_key=tuning_key,
        tune_elements=min(num_elements, _TSLE_TUNE_LIMIT),
        num_elements=num_elements,
        kernels=kernels,
        build_args=build_args,
        solve_args=solve_args,
        scatter_args=scatter_args,
        data=data,
        rhs=rhs,
        nel=nel,
        ntr=ntr,
        ncols=ncols,
        nqf=nqf,
        timings=timings,
    )
    build_block, solve_block, scatter_block = selected
    timings["raw.tsle.build.block_size"] = float(build_block)
    timings["raw.tsle.solve.block_size"] = float(solve_block)
    timings["raw.tsle.scatter.block_size"] = float(scatter_block)
    # The response reconstruction kernel retains the shared raw-CUDA public
    # launch policy (at most 128 threads). TSLE's independent stage choices are
    # recorded above and may legitimately select 256.
    timings["raw.block_size"] = float(min(build_block, 128))

    data.fill(0.0)
    rhs.fill(0.0)
    stream = cupy.cuda.get_current_stream()
    events = [cupy.cuda.Event() for _ in range(4)]
    launch_wall_start = time.perf_counter()
    events[0].record(stream)
    kernels.build(
        (num_elements,),
        (build_block,),
        build_args,
        shared_mem=_shared_bytes_build(nel, ncols, nqf),
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
    timings["raw.tsle.solve"] = cupy.cuda.get_elapsed_time(events[1], events[2]) / 1000.0
    timings["raw.tsle.scatter"] = cupy.cuda.get_elapsed_time(events[2], events[3]) / 1000.0
    timings["raw.tsle.device"] = cupy.cuda.get_elapsed_time(events[0], events[3]) / 1000.0
    timings["raw.bsr_kernel"] = timings["raw.tsle.device"]
    timings["raw.kernel.device"] = timings["raw.tsle.device"]
    timings["raw.kernel.wall"] = time.perf_counter() - launch_wall_start
    timings["raw.total"] = time.perf_counter() - wall_start
    timings["raw.wall_total"] = timings["raw.total"]
    timings["raw.unaccounted"] = 0.0

    response = workspace.local_response if cache_local_response else None
    # Keep the temporary workspace alive through all queued work even for the
    # one-shot call. The final event above has completed before this scope exits.
    _ = owned_workspace
    return RawAdvectionAssemblyResult(
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
    )


def clear_tsle_runtime_caches() -> None:
    """Clear process-local kernel/tuning caches for deterministic tests."""
    _TSLE_MODULE_CACHE.clear()
    _TSLE_TUNING_CACHE.clear()


__all__ = [
    "RawAdvectionTsleWorkspace",
    "assemble_projected_advection_trace_system_eliminated_tsle_bsr",
    "clear_tsle_runtime_caches",
]
