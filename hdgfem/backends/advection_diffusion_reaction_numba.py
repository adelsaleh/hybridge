"""DGSpace adapter for the fused stationary ADR Numba kernels."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..assembly import hdg
from ..assembly.advection_diffusion_reaction import ADRPreparedData
from ..core.space import DGSpace, DGTraceSpace
from ..kernels import NUMBA_AVAILABLE
from ..kernels.advection_diffusion_reaction_fused import (
    assemble_projected_adr_trace_system_eliminated_kernel,
    reconstruct_projected_adr_local_unknowns_kernel,
)
from ..linalg.system import KnownDofReduction
from .numba import (
    _boundary_reduction_maps,
    _interior_side_index,
    _reduction_with_system,
    _trace_orientation_mode,
)


@dataclass(frozen=True)
class NumbaADRTraceAssembly:
    """Reduced ADR trace system assembled by the fused host kernel."""

    trace_system: hdg.TraceSystem
    reduction: KnownDofReduction
    timings: dict[str, float]


def assemble_projected_adr_trace_system_eliminated_numba(
        prepared: ADRPreparedData,
        boundary_condition,
        space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
        diffusion: float = 1.0,
) -> NumbaADRTraceAssembly:
    """Assemble the all-Dirichlet reduced ADR trace system with ``prange``."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")
    timings: dict[str, float] = {}
    start = time.perf_counter()
    trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
    orientation_mode = _trace_orientation_mode(trace_ref)
    boundary_trace = hdg.boundary_trace_coefficients(
        boundary_condition, space, trace_space=trace_ref
    )
    edge_to_solve, free_edges, reduction_template = _boundary_reduction_maps(
        space, boundary_trace, None, trace_space=trace_ref
    )
    mesh = space.mesh
    ntr = trace_ref.edg_dof
    valid_elements = np.ascontiguousarray(mesh.interior_elements, dtype=np.int64)
    valid_faces = np.ascontiguousarray(mesh.interior_faces, dtype=np.int64)
    face_is_free = edge_to_solve[mesh.loc2glob_edge] >= 0
    counts = np.count_nonzero(face_is_free[valid_elements], axis=1).astype(np.int64)
    offsets = np.empty(counts.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts * ntr * ntr, out=offsets[1:])
    nnz = int(offsets[-1]) + valid_elements.size * ntr * ntr
    timings["reduction_map"] = time.perf_counter() - start

    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty(nnz, dtype=np.int64)
    data = np.empty(nnz, dtype=np.float64)
    rhs_indices = np.empty(mesh.num_tri * 3 * ntr, dtype=np.int64)
    rhs_values = np.empty(mesh.num_tri * 3 * ntr, dtype=np.float64)

    start = time.perf_counter()
    assemble_projected_adr_trace_system_eliminated_kernel(
        rows,
        cols,
        data,
        rhs_indices,
        rhs_values,
        np.ascontiguousarray(mesh.loc2glob_edge, dtype=np.int64),
        np.ascontiguousarray(mesh.orientations, dtype=np.bool_),
        _interior_side_index(space),
        np.ascontiguousarray(edge_to_solve, dtype=np.int64),
        np.ascontiguousarray(offsets),
        np.ascontiguousarray(mesh.aff_mats, dtype=np.float64),
        np.ascontiguousarray(mesh.aff_jacs, dtype=np.float64),
        np.ascontiguousarray(mesh.jacs_el_fc, dtype=np.float64),
        np.ascontiguousarray(mesh.normals, dtype=np.float64),
        np.ascontiguousarray(space.quad_data.MKrf_inv, dtype=np.float64),
        np.ascontiguousarray(space.quad_data.bas_of_quads, dtype=np.float64),
        np.ascontiguousarray(space.quad_data.dbas_of_quads, dtype=np.float64),
        np.ascontiguousarray(space.quad_data.Krf_w, dtype=np.float64),
        prepared.reaction_values,
        prepared.beta_values,
        prepared.u_boundary_mass,
        prepared.normal_mass_x,
        prepared.normal_mass_y,
        prepared.d0_reference,
        prepared.d1_reference,
        prepared.element_boundary,
        prepared.trace_lift,
        prepared.interior_gamma_mass,
        prepared.source_rhs,
        np.ascontiguousarray(boundary_trace, dtype=np.float64),
        int(orientation_mode),
        float(diffusion),
    )
    timings["kernel"] = time.perf_counter() - start
    start = time.perf_counter()
    rhs = np.zeros(free_edges.size * ntr, dtype=np.float64)
    np.add.at(rhs, rhs_indices, rhs_values)
    timings["rhs_finalization"] = time.perf_counter() - start
    timings["total"] = sum(timings.values())
    reduction = _reduction_with_system(reduction_template, rows, cols, data, rhs)
    return NumbaADRTraceAssembly(
        trace_system=hdg.TraceSystem(rows, cols, data, rhs, boundary_trace),
        reduction=reduction,
        timings=timings,
    )


def reconstruct_projected_adr_local_unknowns_numba(
        trace,
        prepared: ADRPreparedData,
        space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
        diffusion: float = 1.0,
) -> np.ndarray:
    """Reconstruct mixed ADR element fields in the fused host kernel."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")
    trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
    orientation_mode = _trace_orientation_mode(trace_ref)
    mesh = space.mesh
    out = np.empty((mesh.num_tri, 3 * space.el_dof), dtype=np.float64)
    reconstruct_projected_adr_local_unknowns_kernel(
        out,
        np.ascontiguousarray(trace, dtype=np.float64),
        np.ascontiguousarray(mesh.loc2glob_edge, dtype=np.int64),
        np.ascontiguousarray(mesh.orientations, dtype=np.bool_),
        np.ascontiguousarray(mesh.aff_mats, dtype=np.float64),
        np.ascontiguousarray(mesh.aff_jacs, dtype=np.float64),
        np.ascontiguousarray(mesh.jacs_el_fc, dtype=np.float64),
        np.ascontiguousarray(mesh.normals, dtype=np.float64),
        np.ascontiguousarray(space.quad_data.MKrf_inv, dtype=np.float64),
        np.ascontiguousarray(space.quad_data.bas_of_quads, dtype=np.float64),
        np.ascontiguousarray(space.quad_data.dbas_of_quads, dtype=np.float64),
        np.ascontiguousarray(space.quad_data.Krf_w, dtype=np.float64),
        prepared.reaction_values,
        prepared.beta_values,
        prepared.u_boundary_mass,
        prepared.normal_mass_x,
        prepared.normal_mass_y,
        prepared.d0_reference,
        prepared.d1_reference,
        prepared.element_boundary,
        prepared.source_rhs,
        int(orientation_mode),
        float(diffusion),
    )
    return np.ascontiguousarray(out)


__all__ = [
    "NumbaADRTraceAssembly",
    "assemble_projected_adr_trace_system_eliminated_numba",
    "reconstruct_projected_adr_local_unknowns_numba",
]
