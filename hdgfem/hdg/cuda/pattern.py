"""hdgfem.hdg.cuda.pattern."""

from __future__ import annotations

import numpy as np
import time
from typing import Any
from hdgfem.hdg.cuda.launch import _compile_kernel
from dataclasses import dataclass
from hdgfem.runtime.optional import require_cupy


@dataclass(frozen=True)
class ReducedTraceCsrPattern:
    """Device-side reduced trace face-block graph and lookup maps.

    ``block_indptr``/``block_neighbors`` always describe the compressed face
    graph. For scalar CSR, ``indptr``/``indices`` contain its scalar expansion;
    for BSR they alias the block graph directly.
    """

    indptr: Any
    indices: Any
    block_indptr: Any
    block_neighbors: Any
    side_csr_block_pos: Any
    mass_csr_block_pos: Any
    edge_to_solve_edge: Any
    interior_side_index: Any
    num_blocks: int
    timings: dict[str, float]
    matrix_format: str = "csr"


_CSR_MAX_INCIDENT_SIDES = 4


_CSR_MAX_NEIGHBORS = 12


_RAW_CSR_PATTERN_TEMPLATE = r"""
#define CSR_MAX_INCIDENT_SIDES __CSR_MAX_INCIDENT_SIDES__
#define CSR_MAX_NEIGHBORS __CSR_MAX_NEIGHBORS__
#define CSR_NTR __CSR_NTR__

extern "C" __global__ void build_csr_pattern_incident(
        const long long* __restrict__ interior_elements,
        const long long* __restrict__ interior_faces,
        const long long* __restrict__ loc2glob_edge,
        const long long* __restrict__ edge_to_solve_edge,
        int* __restrict__ incident_counts,
        long long* __restrict__ incident_sides,
        long long* __restrict__ interior_side_index,
        int* __restrict__ overflow,
        const long long num_sides)
{
    const long long side = blockIdx.x * blockDim.x + threadIdx.x;
    if (side >= num_sides) {
        return;
    }
    const long long element = interior_elements[side];
    const long long face = interior_faces[side];
    interior_side_index[element * 3 + face] = side;
    const long long edge = loc2glob_edge[element * 3 + face];
    const long long solve_edge = edge_to_solve_edge[edge];
    if (solve_edge < 0) {
        return;
    }
    const int slot = atomicAdd(&incident_counts[solve_edge], 1);
    if (slot < CSR_MAX_INCIDENT_SIDES) {
        incident_sides[solve_edge * CSR_MAX_INCIDENT_SIDES + slot] = side;
    } else {
        overflow[0] = 1;
    }
}

__device__ __forceinline__ void csr_add_unique_neighbor(
        long long* __restrict__ neighbors,
        int* __restrict__ count,
        int* __restrict__ overflow,
        const long long value)
{
    for (int i = 0; i < *count; ++i) {
        if (neighbors[i] == value) {
            return;
        }
    }
    if (*count < CSR_MAX_NEIGHBORS) {
        neighbors[*count] = value;
        *count += 1;
    } else {
        overflow[0] = 1;
    }
}

extern "C" __global__ void build_csr_pattern_neighbors(
        const long long* __restrict__ interior_elements,
        const long long* __restrict__ loc2glob_edge,
        const long long* __restrict__ edge_to_solve_edge,
        const int* __restrict__ incident_counts,
        const long long* __restrict__ incident_sides,
        int* __restrict__ block_counts,
        long long* __restrict__ fixed_neighbors,
        int* __restrict__ overflow,
        const long long num_free_edges)
{
    const long long row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_free_edges) {
        return;
    }

    long long neighbors[CSR_MAX_NEIGHBORS];
    int count = 0;
    csr_add_unique_neighbor(neighbors, &count, overflow, row);

    const int incident_count = incident_counts[row];
    const int capped_incident_count = incident_count < CSR_MAX_INCIDENT_SIDES ? incident_count : CSR_MAX_INCIDENT_SIDES;
    for (int slot = 0; slot < capped_incident_count; ++slot) {
        const long long side = incident_sides[row * CSR_MAX_INCIDENT_SIDES + slot];
        if (side < 0) {
            continue;
        }
        const long long element = interior_elements[side];
        for (int col_face = 0; col_face < 3; ++col_face) {
            const long long col_edge = loc2glob_edge[element * 3 + col_face];
            const long long col_solve_edge = edge_to_solve_edge[col_edge];
            if (col_solve_edge >= 0) {
                csr_add_unique_neighbor(neighbors, &count, overflow, col_solve_edge);
            }
        }
    }

    for (int i = 1; i < count; ++i) {
        const long long value = neighbors[i];
        int j = i - 1;
        while (j >= 0 && neighbors[j] > value) {
            neighbors[j + 1] = neighbors[j];
            --j;
        }
        neighbors[j + 1] = value;
    }

    block_counts[row] = count;
    for (int i = 0; i < CSR_MAX_NEIGHBORS; ++i) {
        fixed_neighbors[row * CSR_MAX_NEIGHBORS + i] = i < count ? neighbors[i] : -1;
    }
}

extern "C" __global__ void finalize_csr_pattern_rows(
        const int* __restrict__ block_counts,
        const int* __restrict__ block_indptr,
        const long long* __restrict__ fixed_neighbors,
        int* __restrict__ block_neighbors,
        int* __restrict__ indptr,
        int* __restrict__ indices,
        int* __restrict__ mass_csr_block_pos,
        const long long num_free_edges)
{
    const long long row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_free_edges) {
        return;
    }
    const int count = block_counts[row];
    const int block_start = block_indptr[row];
    const int scalar_base = block_start * CSR_NTR * CSR_NTR;
    int mass_pos = -1;

    for (int p = 0; p < count; ++p) {
        const int neighbor = (int)fixed_neighbors[row * CSR_MAX_NEIGHBORS + p];
        block_neighbors[block_start + p] = neighbor;
        if (neighbor == row) {
            mass_pos = p;
        }
    }
    mass_csr_block_pos[row] = mass_pos;

    for (int i = 0; i < CSR_NTR; ++i) {
        const int scalar_row = (int)(row * CSR_NTR + i);
        const int row_start = scalar_base + i * count * CSR_NTR;
        indptr[scalar_row] = row_start;
        for (int p = 0; p < count; ++p) {
            const int neighbor = (int)fixed_neighbors[row * CSR_MAX_NEIGHBORS + p];
            for (int j = 0; j < CSR_NTR; ++j) {
                indices[row_start + p * CSR_NTR + j] = neighbor * CSR_NTR + j;
            }
        }
    }
    if (row == num_free_edges - 1) {
        indptr[num_free_edges * CSR_NTR] = (block_indptr[num_free_edges]) * CSR_NTR * CSR_NTR;
    }
}

extern "C" __global__ void finalize_bsr_pattern_rows(
        const int* __restrict__ block_counts,
        const int* __restrict__ block_indptr,
        const long long* __restrict__ fixed_neighbors,
        int* __restrict__ block_neighbors,
        int* __restrict__ mass_csr_block_pos,
        const long long num_free_edges)
{
    const long long row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_free_edges) {
        return;
    }
    const int count = block_counts[row];
    const int block_start = block_indptr[row];
    int mass_pos = -1;
    for (int p = 0; p < count; ++p) {
        const int neighbor = (int)fixed_neighbors[row * CSR_MAX_NEIGHBORS + p];
        block_neighbors[block_start + p] = neighbor;
        if (neighbor == row) {
            mass_pos = p;
        }
    }
    mass_csr_block_pos[row] = mass_pos;
}

extern "C" __global__ void finalize_csr_pattern_side_positions(
        const long long* __restrict__ interior_elements,
        const long long* __restrict__ interior_faces,
        const long long* __restrict__ loc2glob_edge,
        const long long* __restrict__ edge_to_solve_edge,
        const int* __restrict__ block_counts,
        const long long* __restrict__ fixed_neighbors,
        int* __restrict__ side_csr_block_pos,
        const long long num_sides)
{
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sides * 3) {
        return;
    }
    const long long side = idx / 3;
    const int col_face = (int)(idx - side * 3);
    const long long element = interior_elements[side];
    const long long row_face = interior_faces[side];
    const long long row_edge = loc2glob_edge[element * 3 + row_face];
    const long long row_solve_edge = edge_to_solve_edge[row_edge];
    const long long col_edge = loc2glob_edge[element * 3 + col_face];
    const long long col_solve_edge = edge_to_solve_edge[col_edge];
    int position = -1;
    if (row_solve_edge >= 0 && col_solve_edge >= 0) {
        const int count = block_counts[row_solve_edge];
        for (int p = 0; p < count; ++p) {
            if (fixed_neighbors[row_solve_edge * CSR_MAX_NEIGHBORS + p] == col_solve_edge) {
                position = p;
                break;
            }
        }
    }
    side_csr_block_pos[side * 3 + col_face] = position;
}
"""


def _csr_pattern_source(ntr: int, max_incident: int = _CSR_MAX_INCIDENT_SIDES, max_neighbors: int = _CSR_MAX_NEIGHBORS) -> str:
    """Build the C++ source for CSR pattern kernels."""
    return (
        _RAW_CSR_PATTERN_TEMPLATE
        .replace('__CSR_NTR__', str(int(ntr)))
        .replace('__CSR_MAX_INCIDENT_SIDES__', str(int(max_incident)))
        .replace('__CSR_MAX_NEIGHBORS__', str(int(max_neighbors)))
    )


def _edge_to_solve_edge_device(cspace):
    """Build the device map from global edges to reduced solve edges."""
    cupy = require_cupy()
    edge_to_solve = cupy.full(cspace.mesh.num_edg, -1, dtype=cupy.int64)
    edge_to_solve[cspace.mesh.int_edges_inds] = cupy.arange(cspace.mesh.int_edges_inds.size, dtype=cupy.int64)
    return edge_to_solve


def build_reduced_csr_pattern_raw(
        cspace,
        timings: dict[str, float] | None = None,
        *,
        matrix_format: str = "csr",
) -> ReducedTraceCsrPattern:
    """Build the reduced trace scalar-CSR or face-BSR pattern on the device."""
    cupy = require_cupy()
    matrix_format = str(matrix_format).lower()
    if matrix_format not in {"csr", "bsr"}:
        raise ValueError("matrix_format must be 'csr' or 'bsr'")
    mesh = cspace.mesh
    ntr = int(cspace.edg_dof)
    num_free = int(mesh.int_edges_inds.size)
    num_sides = int(mesh.interior_elements.size)
    system_size = num_free * ntr
    local_timings: dict[str, float] = {}
    total_start = time.perf_counter()

    start = time.perf_counter()
    edge_to_solve = _edge_to_solve_edge_device(cspace)
    incident_counts = cupy.zeros(num_free, dtype=cupy.int32)
    incident_sides = cupy.full((num_free, _CSR_MAX_INCIDENT_SIDES), -1, dtype=cupy.int64)
    interior_side_index = cupy.full(cspace.mesh.num_tri * 3, -1, dtype=cupy.int64)
    overflow = cupy.zeros(1, dtype=cupy.int32)
    source = _csr_pattern_source(ntr)
    incident_kernel = _compile_kernel(cupy, source, 'build_csr_pattern_incident', 0)
    threads = 256
    incident_kernel(
        ((num_sides + threads - 1) // threads,),
        (threads,),
        (
            mesh.interior_elements,
            mesh.interior_faces,
            mesh.loc2glob_edge,
            edge_to_solve,
            incident_counts,
            incident_sides.reshape(-1),
            interior_side_index,
            overflow,
            np.int64(num_sides),
        ),
    )
    cupy.cuda.get_current_stream().synchronize()
    local_timings['raw.csr_pattern.incident'] = time.perf_counter() - start

    start = time.perf_counter()
    block_counts = cupy.empty(num_free, dtype=cupy.int32)
    fixed_neighbors = cupy.empty((num_free, _CSR_MAX_NEIGHBORS), dtype=cupy.int64)
    neighbors_kernel = _compile_kernel(cupy, source, 'build_csr_pattern_neighbors', 0)
    neighbors_kernel(
        ((num_free + threads - 1) // threads,),
        (threads,),
        (
            mesh.interior_elements,
            mesh.loc2glob_edge,
            edge_to_solve,
            incident_counts,
            incident_sides.reshape(-1),
            block_counts,
            fixed_neighbors.reshape(-1),
            overflow,
            np.int64(num_free),
        ),
    )
    cupy.cuda.get_current_stream().synchronize()
    local_timings['raw.csr_pattern.neighbors'] = time.perf_counter() - start

    if int(overflow[0].get()) != 0:
        raise RuntimeError(
            'raw CSR pattern exceeded fixed local topology capacity; increase _CSR_MAX_INCIDENT_SIDES or _CSR_MAX_NEIGHBORS'
        )

    start = time.perf_counter()
    block_indptr = cupy.empty(num_free + 1, dtype=cupy.int32)
    block_indptr[0] = 0
    cupy.cumsum(block_counts, out=block_indptr[1:])
    num_blocks = int(block_indptr[-1].get())
    local_timings['raw.csr_pattern.cumsum'] = time.perf_counter() - start

    start = time.perf_counter()
    block_neighbors = cupy.empty(num_blocks, dtype=cupy.int32)
    side_csr_block_pos = cupy.empty(num_sides * 3, dtype=cupy.int32)
    mass_csr_block_pos = cupy.empty(num_free, dtype=cupy.int32)
    if matrix_format == "csr":
        scalar_nnz = num_blocks * ntr * ntr
        indptr = cupy.empty(system_size + 1, dtype=cupy.int32)
        indices = cupy.empty(scalar_nnz, dtype=cupy.int32)
        finalize_rows = _compile_kernel(cupy, source, 'finalize_csr_pattern_rows', 0)
        finalize_rows(
            ((num_free + threads - 1) // threads,),
            (threads,),
            (
                block_counts,
                block_indptr,
                fixed_neighbors.reshape(-1),
                block_neighbors,
                indptr,
                indices,
                mass_csr_block_pos,
                np.int64(num_free),
            ),
        )
    else:
        finalize_rows = _compile_kernel(cupy, source, 'finalize_bsr_pattern_rows', 0)
        finalize_rows(
            ((num_free + threads - 1) // threads,),
            (threads,),
            (
                block_counts,
                block_indptr,
                fixed_neighbors.reshape(-1),
                block_neighbors,
                mass_csr_block_pos,
                np.int64(num_free),
            ),
        )
        indptr = block_indptr
        indices = block_neighbors
    finalize_positions = _compile_kernel(cupy, source, 'finalize_csr_pattern_side_positions', 0)
    finalize_positions(
        ((num_sides * 3 + threads - 1) // threads,),
        (threads,),
        (
            mesh.interior_elements,
            mesh.interior_faces,
            mesh.loc2glob_edge,
            edge_to_solve,
            block_counts,
            fixed_neighbors.reshape(-1),
            side_csr_block_pos,
            np.int64(num_sides),
        ),
    )
    cupy.cuda.get_current_stream().synchronize()
    local_timings[
        'raw.csr_pattern.expand' if matrix_format == "csr" else 'raw.bsr_pattern.finalize'
    ] = time.perf_counter() - start
    local_timings['raw.csr_pattern.total'] = time.perf_counter() - total_start
    if timings is not None:
        timings.update(local_timings)
    return ReducedTraceCsrPattern(
        indptr=indptr,
        indices=indices,
        block_indptr=block_indptr,
        block_neighbors=block_neighbors,
        side_csr_block_pos=side_csr_block_pos,
        mass_csr_block_pos=mass_csr_block_pos,
        edge_to_solve_edge=edge_to_solve,
        interior_side_index=interior_side_index,
        num_blocks=num_blocks,
        timings=local_timings,
        matrix_format=matrix_format,
    )
