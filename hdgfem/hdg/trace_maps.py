"""hdgfem.hdg.trace_maps."""

from __future__ import annotations

from hdgfem.core.space import DGSpace, DGTraceSpace

import numpy as np
from hdgfem.linalg.system import KnownDofReduction



def _trace_ref(space: DGSpace, trace_space: DGTraceSpace | None = None) -> DGTraceSpace:
    """Return the requested trace reference, defaulting to the legacy trace basis."""
    return space.trace_space("legacy-lagrange") if trace_space is None else trace_space


def _trace_orientation_mode(trace_ref: DGTraceSpace) -> int:
    """Return the fused-kernel edge orientation mode for a trace basis."""
    if trace_ref.kind == "legacy-lagrange" and trace_ref.nodal:
        return 0
    if trace_ref.kind == "legendre-modal" and not trace_ref.nodal:
        return 1
    raise NotImplementedError(
        "assembly_backend='numba' currently supports trace_basis='legacy-lagrange' "
        "and trace_basis='legendre-modal'"
    )


def _edge_order_to_solve_map(
        num_edges: int,
        active_edges: np.ndarray,
        edge_order: np.ndarray | None,
) -> np.ndarray:
    """Map global mesh edges to contiguous solve-edge positions."""
    active_edges = np.asarray(active_edges, dtype=np.int64)
    if edge_order is None:
        ordered_edges = active_edges
    else:
        ordered_edges = np.asarray(edge_order, dtype=np.int64)
        if ordered_edges.ndim != 1:
            raise ValueError("edge_order must be one-dimensional")
        if ordered_edges.shape != active_edges.shape:
            raise ValueError(
                f"edge_order must have shape {active_edges.shape} for this solve system; "
                f"got {ordered_edges.shape}"
            )

    if ordered_edges.size and (ordered_edges.min() < 0 or ordered_edges.max() >= num_edges):
        raise ValueError("edge_order contains edge ids outside the mesh")

    active_mask = np.zeros(num_edges, dtype=bool)
    active_mask[active_edges] = True
    seen = np.zeros(num_edges, dtype=bool)
    seen[ordered_edges] = True
    if not np.array_equal(active_mask, seen):
        raise ValueError("edge_order must contain every active solve edge exactly once")

    edge_to_solve_edge = np.full(num_edges, -1, dtype=np.int64)
    edge_to_solve_edge[ordered_edges] = np.arange(ordered_edges.size, dtype=np.int64)
    return np.ascontiguousarray(edge_to_solve_edge)


def _boundary_reduction_maps(
        space: DGSpace,
        boundary_trace: np.ndarray,
        edge_order: np.ndarray | None = None,
        *,
        trace_space: DGTraceSpace | None = None,
) -> tuple[np.ndarray, np.ndarray, KnownDofReduction]:
    """Return edge and dof maps for direct boundary elimination."""
    mesh = space.mesh
    trace_ref = _trace_ref(space, trace_space)
    edg_dof = trace_ref.edg_dof

    edge_is_free = np.ones(mesh.num_edg, dtype=bool)
    edge_is_free[mesh.bnd_edges_inds] = False
    free_edges = np.flatnonzero(edge_is_free).astype(np.int64)

    edge_to_solve_edge = _edge_order_to_solve_map(mesh.num_edg, free_edges, edge_order)

    system_size = mesh.num_edg * edg_dof
    free_mask = np.repeat(edge_is_free, edg_dof)
    known_mask = ~free_mask
    old_to_new = np.full(system_size, -1, dtype=np.int64)
    old_to_new[free_mask] = np.arange(np.count_nonzero(free_mask), dtype=np.int64)

    reduction = KnownDofReduction(
        rows=np.empty(0, dtype=np.int64),
        cols=np.empty(0, dtype=np.int64),
        data=np.empty(0, dtype=np.float64),
        rhs=np.empty(np.count_nonzero(free_mask), dtype=np.float64),
        free_mask=np.ascontiguousarray(free_mask),
        known_mask=np.ascontiguousarray(known_mask),
        known_values=np.ascontiguousarray(boundary_trace.ravel(), dtype=np.float64),
        old_to_new=np.ascontiguousarray(old_to_new),
    )
    return edge_to_solve_edge, np.ascontiguousarray(free_edges), reduction


def _reduction_with_system(
        reduction: KnownDofReduction,
        rows: np.ndarray,
        cols: np.ndarray,
        data: np.ndarray,
        rhs: np.ndarray,
) -> KnownDofReduction:
    """Attach assembled reduced COO arrays to a precomputed reduction map."""
    return KnownDofReduction(
        rows=rows,
        cols=cols,
        data=data,
        rhs=rhs,
        free_mask=reduction.free_mask,
        known_mask=reduction.known_mask,
        known_values=reduction.known_values,
        old_to_new=reduction.old_to_new,
    )


def _edge_to_solve_edge(mesh) -> np.ndarray:
    """Map interior global edges to contiguous reduced solve-edge ids."""
    edge_is_free = np.ones(mesh.num_edg, dtype=bool)
    edge_is_free[mesh.bnd_edges_inds] = False
    free_edges = np.flatnonzero(edge_is_free).astype(np.int64)
    edge_to_solve = np.full(mesh.num_edg, -1, dtype=np.int64)
    edge_to_solve[free_edges] = np.arange(free_edges.size, dtype=np.int64)
    return np.ascontiguousarray(edge_to_solve)


def _interior_side_index(mesh) -> np.ndarray:
    """Map each interior element side to its contiguous side index."""
    index = np.full((mesh.num_tri, 3), -1, dtype=np.int64)
    index[mesh.interior_elements, mesh.interior_faces] = np.arange(mesh.interior_elements.size, dtype=np.int64)
    return np.ascontiguousarray(index)


def _side_flux_offsets(mesh, edge_to_solve_edge: np.ndarray, edg_dof: int) -> np.ndarray:
    """Compute per-side offsets used for side-by-side flux indexing."""
    face_is_free = edge_to_solve_edge[mesh.loc2glob_edge] >= 0
    side_col_counts = np.count_nonzero(face_is_free[mesh.interior_elements], axis=1).astype(np.int64)
    offsets = np.empty(side_col_counts.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(side_col_counts * edg_dof * edg_dof, out=offsets[1:])
    return np.ascontiguousarray(offsets)
