"""DGSpace adapter for the fused stationary ADR Numba kernels.

The kernels build every element's face tables from the face samples
``prepared.tau_total`` and ``prepared.gamma`` and small reference tables, so
``prepare_adr_data(dense_local_matrices=False)`` is sufficient.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from hybridge.hdg import condensation as hdg
from hybridge.mixed.adr_preparation import ADRPreparedData
from hybridge.mixed.coefficients import PreparedDiffusion, prepare_diffusion
from hybridge.core.space import DGSpace, DGTraceSpace
from hybridge.runtime.optional import NUMBA_AVAILABLE
from hybridge.mixed.adr_numba_kernels import (
    assemble_projected_adr_trace_system_eliminated_kernel,
    reconstruct_adr_from_local_columns_kernel,
    reconstruct_projected_adr_local_unknowns_kernel,
)
from hybridge.linalg.reduction import KnownDofReduction
from hybridge.hdg.trace_maps import (
    _boundary_reduction_maps,
    _reduction_with_system,
    _trace_orientation_mode,
)
from hybridge.hdg.trace_maps import _interior_side_index


@dataclass(frozen=True)
class NumbaADRTraceAssembly:
    """Reduced ADR trace system assembled by the fused host kernel."""

    trace_system: hdg.TraceSystem
    reduction: KnownDofReduction
    timings: dict[str, float]
    local_columns: np.ndarray | None = None


def assemble_projected_adr_trace_system_eliminated_numba(
        prepared: ADRPreparedData,
        boundary_condition,
        space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
        diffusion=1.0,
        diffusion_data: PreparedDiffusion | None = None,
        local_columns: np.ndarray | None = None,
) -> NumbaADRTraceAssembly:
    """Assemble the all-Dirichlet reduced ADR trace system with ``prange``.

    ``local_columns`` (shape ``(K, 3*el_dof, 3*edg_dof + 1)``) receives every
    element's local solution columns, so reconstruction of the same system can
    use :func:`reconstruct_projected_adr_local_unknowns_numba` with
    ``local_columns`` instead of rebuilding the local problems.
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")
    timings: dict[str, float] = {}
    start = time.perf_counter()
    trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
    orientation_mode = _trace_orientation_mode(trace_ref)
    tensor = prepare_diffusion(diffusion, space) if diffusion_data is None else diffusion_data
    status = np.zeros(space.mesh.num_tri, dtype=np.int64)
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
        _interior_side_index(space.mesh),
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
        *_face_table_arguments(prepared, space, trace_ref),
        prepared.d0_reference,
        prepared.d1_reference,
        prepared.source_rhs,
        np.ascontiguousarray(boundary_trace, dtype=np.float64),
        int(orientation_mode),
        tensor.kinds, tensor.constants, tensor.inverse_values, status,
        _column_buffer(local_columns, space, ntr), local_columns is not None,
    )
    if np.any(status):
        raise ValueError("ADR inverse-diffusion mass factorization failed on elements " + str(np.flatnonzero(status)))
    timings["kernel"] = time.perf_counter() - start
    start = time.perf_counter()
    rhs = np.zeros(free_edges.size * ntr, dtype=np.float64)
    np.add.at(rhs, rhs_indices, rhs_values)
    timings["rhs_finalization"] = time.perf_counter() - start
    timings["total"] = sum(timings.values())
    timings.update({f"diffusion.{name}.elements": float(count) for name, count in tensor.counts.items()})
    reduction = _reduction_with_system(reduction_template, rows, cols, data, rhs)
    return NumbaADRTraceAssembly(
        trace_system=hdg.TraceSystem(rows, cols, data, rhs, boundary_trace),
        reduction=reduction,
        timings=timings,
        local_columns=local_columns,
    )


def _face_table_arguments(prepared: ADRPreparedData, space: DGSpace, trace_ref: DGTraceSpace) -> tuple:
    """Face samples and reference tables from which the kernels build each element's face tables."""
    if prepared.face_quadrature is not None and not np.array_equal(prepared.face_quadrature, trace_ref.quads):
        raise ValueError("prepared ADR face samples use a different trace quadrature than trace_space")
    contiguous = lambda array: np.ascontiguousarray(array, dtype=np.float64)
    return (
        contiguous(prepared.tau_total),
        contiguous(prepared.gamma),
        contiguous(trace_ref.bas_of_bd_quads),
        contiguous(trace_ref.weighted_bas_of_bd_quads),
        contiguous(trace_ref.weighted_bas1d_of_ref_edg_qds),
        contiguous(trace_ref.oriented_basis_table),
        contiguous(trace_ref.weights),
        contiguous(trace_ref.face_trace_test_element_trial_oriented),
        contiguous(space.quad_data.face_element_test_element_trial),
    )


def _column_buffer(local_columns, space: DGSpace, ntr: int) -> np.ndarray:
    """Validate a caller-owned local-column buffer, or return a 1-element placeholder."""
    if local_columns is None:
        return np.empty((1, 1, 1), dtype=np.float64)
    expected = (space.mesh.num_tri, 3 * space.el_dof, 3 * ntr + 1)
    if local_columns.shape != expected or local_columns.dtype != np.float64 or not local_columns.flags.c_contiguous:
        raise ValueError(f"local_columns must be a C-contiguous float64 array of shape {expected}")
    return local_columns


def reconstruct_projected_adr_local_unknowns_numba(
        trace,
        prepared: ADRPreparedData,
        space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
        diffusion=1.0,
        diffusion_data: PreparedDiffusion | None = None,
        local_columns: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct mixed ADR element fields in the fused host kernel.

    With ``local_columns`` from the assembly of the same prepared system, the
    fields are a contraction of those columns with the trace; otherwise every
    local problem is rebuilt and solved.
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")
    trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
    orientation_mode = _trace_orientation_mode(trace_ref)
    if local_columns is not None:
        mesh = space.mesh
        out = np.empty((mesh.num_tri, 3 * space.el_dof), dtype=np.float64)
        reconstruct_adr_from_local_columns_kernel(
            out, np.ascontiguousarray(trace, dtype=np.float64), _column_buffer(local_columns, space, trace_ref.edg_dof),
            np.ascontiguousarray(mesh.loc2glob_edge, dtype=np.int64),
            np.ascontiguousarray(mesh.orientations, dtype=np.bool_), int(orientation_mode))
        return out
    tensor = prepare_diffusion(diffusion, space) if diffusion_data is None else diffusion_data
    status = np.zeros(space.mesh.num_tri, dtype=np.int64)
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
        *_face_table_arguments(prepared, space, trace_ref),
        prepared.d0_reference,
        prepared.d1_reference,
        prepared.source_rhs,
        int(orientation_mode),
        tensor.kinds, tensor.constants, tensor.inverse_values, status,
    )
    if np.any(status):
        raise ValueError("ADR inverse-diffusion mass factorization failed on elements " + str(np.flatnonzero(status)))
    return np.ascontiguousarray(out)


__all__ = [
    "NumbaADRTraceAssembly",
    "assemble_projected_adr_trace_system_eliminated_numba",
    "reconstruct_projected_adr_local_unknowns_numba",
]
