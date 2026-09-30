"""Experimental Numba adapter for the hard-coded tensor-diffusion test7 case."""

from __future__ import annotations

import time

import numpy as np

from hdgfem.hdg import condensation as hdg_assembly
from hdgfem.core.space import DGSpace
from hdgfem.kernels import NUMBA_AVAILABLE
from scripts.diffusion_reaction.experiments.test7_fused_kernels import (
    assemble_test7_tensor_trace_system_eliminated_kernel,
    build_test7_boundary_trace_kernel,
    reconstruct_test7_tensor_local_unknowns_kernel,
)
from hdgfem.backends.numba import (
    NumbaDiffusionTraceAssembly,
    _interior_side_index,
    _normalize_diffusion_stabilization,
    _reference_diffusion_derivative_matrices,
)
from hdgfem.hdg.trace_maps import _boundary_reduction_maps, _reduction_with_system


def _test7_frequencies(m: int = 1, n: int = 1) -> tuple[float, float]:
    return 0.5 * int(m) * np.pi, 0.5 * int(n) * np.pi


def assemble_test7_tensor_trace_system_eliminated_numba(
        stabilization,
        space: DGSpace,
        *,
        m: int = 1,
        n: int = 1,
        edge_order: np.ndarray | None = None,
) -> NumbaDiffusionTraceAssembly:
    """Assemble the reduced trace system for the hard-coded ``test7`` tensor."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("test7 fused tensor assembly requires numba")

    timings: dict[str, float] = {}
    start = time.perf_counter()
    tau = _normalize_diffusion_stabilization(stabilization, space)
    interior_side_index = _interior_side_index(space.mesh)
    d0_reference, d1_reference = _reference_diffusion_derivative_matrices(space)
    a, b = _test7_frequencies(m, n)
    timings["preparation"] = time.perf_counter() - start

    mesh = space.mesh
    q = space.quad_data

    start = time.perf_counter()
    boundary_trace = np.zeros((mesh.num_edg, q.edg_dof), dtype=np.float64)
    build_test7_boundary_trace_kernel(
        boundary_trace,
        np.ascontiguousarray(mesh.bnd_edges_inds, dtype=np.int64),
        np.ascontiguousarray(mesh.edges, dtype=np.int64),
        np.ascontiguousarray(mesh.node_coords, dtype=np.float64),
        np.ascontiguousarray(q.quads_JGL, dtype=np.float64),
        np.ascontiguousarray(q.weights_JGL, dtype=np.float64),
        np.ascontiguousarray(q.bas1d_of_ref_edg_qds, dtype=np.float64),
        np.ascontiguousarray(np.linalg.inv(q.M_rf_fc), dtype=np.float64),
        float(a),
        float(b),
    )
    timings["boundary_trace"] = time.perf_counter() - start

    start = time.perf_counter()
    edge_to_solve_edge, free_edges, reduction_template = _boundary_reduction_maps(space, boundary_trace, edge_order)
    valid_elements = np.ascontiguousarray(mesh.interior_elements, dtype=np.int64)
    valid_faces = np.ascontiguousarray(mesh.interior_faces, dtype=np.int64)
    face_is_free = edge_to_solve_edge[mesh.loc2glob_edge] >= 0
    side_col_counts = np.count_nonzero(face_is_free[valid_elements], axis=1).astype(np.int64)
    side_flux_offsets = np.empty(side_col_counts.size + 1, dtype=np.int64)
    side_flux_offsets[0] = 0
    np.cumsum(side_col_counts * q.edg_dof * q.edg_dof, out=side_flux_offsets[1:])
    n_flux = int(side_flux_offsets[-1])
    n_mass = valid_elements.size * q.edg_dof * q.edg_dof
    nnz = n_flux + n_mass
    timings["reduction_map"] = time.perf_counter() - start

    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty_like(rows)
    data = np.empty(nnz, dtype=np.float64)
    rhs_indices = np.empty(mesh.num_tri * 3 * q.edg_dof, dtype=np.int64)
    rhs_values = np.empty_like(rhs_indices, dtype=np.float64)

    start = time.perf_counter()
    assemble_test7_tensor_trace_system_eliminated_kernel(
        rows,
        cols,
        data,
        rhs_indices,
        rhs_values,
        np.ascontiguousarray(mesh.loc2glob_edge, dtype=np.int64),
        np.ascontiguousarray(mesh.orientations, dtype=np.bool_),
        np.ascontiguousarray(mesh.loc2oriented_face_coupling, dtype=np.int64),
        interior_side_index,
        np.ascontiguousarray(edge_to_solve_edge, dtype=np.int64),
        valid_elements,
        valid_faces,
        np.ascontiguousarray(side_flux_offsets, dtype=np.int64),
        np.ascontiguousarray(mesh.aff_mats, dtype=np.float64),
        np.ascontiguousarray(mesh.aff_vecs, dtype=np.float64),
        np.ascontiguousarray(mesh.aff_jacs, dtype=np.float64),
        np.ascontiguousarray(mesh.jacs_el_fc, dtype=np.float64),
        np.ascontiguousarray(mesh.normals, dtype=np.float64),
        tau,
        np.ascontiguousarray(q.Krf_quads, dtype=np.float64),
        np.ascontiguousarray(q.Krf_w, dtype=np.float64),
        np.ascontiguousarray(q.bas_of_quads, dtype=np.float64),
        np.ascontiguousarray(q.face_element_test_element_trial, dtype=np.float64),
        np.ascontiguousarray(q.face_element_test_trace_trial, dtype=np.float64),
        np.ascontiguousarray(q.M_rf_fc, dtype=np.float64),
        np.ascontiguousarray(q.face_trace_test_element_trial_oriented, dtype=np.float64),
        d0_reference,
        d1_reference,
        np.ascontiguousarray(boundary_trace, dtype=np.float64),
        float(a),
        float(b),
    )
    timings["kernel"] = time.perf_counter() - start

    start = time.perf_counter()
    rhs = np.zeros(free_edges.size * q.edg_dof, dtype=np.float64)
    np.add.at(rhs, rhs_indices, rhs_values)
    timings["rhs_finalization"] = time.perf_counter() - start
    timings["total"] = sum(timings.values())

    reduction = _reduction_with_system(reduction_template, rows, cols, data, rhs)
    return NumbaDiffusionTraceAssembly(
        trace_system=hdg_assembly.TraceSystem(
            rows=rows,
            cols=cols,
            data=data,
            rhs=rhs,
            boundary_trace=boundary_trace,
        ),
        timings=timings,
        reduction=reduction,
    )


def reconstruct_test7_tensor_local_unknowns_numba(
        trace: np.ndarray,
        stabilization,
        space: DGSpace,
        *,
        m: int = 1,
        n: int = 1,
) -> np.ndarray:
    """Recover local mixed unknowns for the hard-coded ``test7`` tensor."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("test7 fused tensor reconstruction requires numba")

    mesh = space.mesh
    q = space.quad_data
    trace = np.ascontiguousarray(np.asarray(trace, dtype=np.float64))
    expected_trace_shape = (mesh.num_edg * q.edg_dof,)
    if trace.shape != expected_trace_shape:
        raise ValueError(f"trace must have shape {expected_trace_shape}; got {trace.shape}")

    tau = _normalize_diffusion_stabilization(stabilization, space)
    d0_reference, d1_reference = _reference_diffusion_derivative_matrices(space)
    a, b = _test7_frequencies(m, n)
    local_unknowns = np.empty((mesh.num_tri, 3 * q.el_dof), dtype=np.float64)
    reconstruct_test7_tensor_local_unknowns_kernel(
        local_unknowns,
        trace,
        np.ascontiguousarray(mesh.loc2glob_edge, dtype=np.int64),
        np.ascontiguousarray(mesh.orientations, dtype=np.bool_),
        np.ascontiguousarray(mesh.aff_mats, dtype=np.float64),
        np.ascontiguousarray(mesh.aff_vecs, dtype=np.float64),
        np.ascontiguousarray(mesh.aff_jacs, dtype=np.float64),
        np.ascontiguousarray(mesh.jacs_el_fc, dtype=np.float64),
        np.ascontiguousarray(mesh.normals, dtype=np.float64),
        tau,
        np.ascontiguousarray(q.Krf_quads, dtype=np.float64),
        np.ascontiguousarray(q.Krf_w, dtype=np.float64),
        np.ascontiguousarray(q.bas_of_quads, dtype=np.float64),
        np.ascontiguousarray(q.face_element_test_element_trial, dtype=np.float64),
        np.ascontiguousarray(q.face_element_test_trace_trial, dtype=np.float64),
        d0_reference,
        d1_reference,
        float(a),
        float(b),
    )
    return local_unknowns


__all__ = [
    "assemble_test7_tensor_trace_system_eliminated_numba",
    "reconstruct_test7_tensor_local_unknowns_numba",
]
