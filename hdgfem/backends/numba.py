"""DGSpace-aware Numba assembly adapter for advection-reaction HDG.

This module is the boundary between the public :mod:`hdgfem` abstractions and
pure ndarray Numba kernels.  It intentionally keeps Python callables and
``DGField`` objects out of the kernels.
"""

from __future__ import annotations

import time
import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from hdgfem.hdg import condensation as hdg_assembly
from hdgfem.kernels import NUMBA_AVAILABLE
from hdgfem.linalg.reduction import KnownDofReduction
from hdgfem.kernels.diffusion_reaction_fused import (
    factor_projected_diffusion_schur_kernel,
    assemble_diffusion_trace_rhs_eliminated_kernel,
    assemble_diffusion_trace_system_eliminated_kernel,
    assemble_projected_diffusion_trace_rhs_eliminated_kernel,
    assemble_projected_diffusion_trace_system_eliminated_kernel,
    assemble_projected_tensor_diffusion_trace_system_eliminated_kernel,
    reconstruct_diffusion_local_unknowns_kernel,
    reconstruct_projected_diffusion_local_unknowns_kernel,
    reconstruct_projected_tensor_diffusion_local_unknowns_kernel,
)
from hdgfem.core.space import DGField, DGSpace, DGTraceSpace
from hdgfem.hdg.trace_maps import _trace_ref
from hdgfem.hdg.trace_maps import (
    _interior_side_index,
    _boundary_reduction_maps,
    _reduction_with_system,
    _trace_orientation_mode,
)
from hdgfem.hdg.coefficients import (
    _reaction_coefficients,
    _same_space_field_coefficients,
    _source_coefficients,
)


@dataclass(frozen=True)
class NumbaDiffusionTraceAssembly:
    """Reduced diffusion-reaction trace system assembled by Numba."""

    trace_system: hdg_assembly.TraceSystem
    timings: dict[str, float]
    reduction: KnownDofReduction


@dataclass(frozen=True)
class NumbaDiffusionSchurCache:
    """Persistent identity-diffusion scalar factors; no mixed matrix batches."""

    factors: np.ndarray
    pivots: np.ndarray
    factor_kind: str
    operator_key: str
    construction_seconds: float

    @property
    def local_factor_bytes(self) -> int:
        """Return retained factor and pivot storage in bytes."""
        return self.factors.nbytes + self.pivots.nbytes


def _diffusion_schur_inputs(reaction, stabilization, space):
    """Prepare operator-only inputs shared by factor construction and validation."""
    mesh, q = space.mesh, space.quad_data
    coeffs, scalar, is_scalar = _reaction_coefficients(reaction, space)
    d0, d1 = _reference_diffusion_derivative_matrices(space)
    arrays = tuple(np.ascontiguousarray(value, dtype=np.float64) for value in (
        mesh.aff_mats, mesh.aff_jacs, mesh.jacs_el_fc, mesh.normals,
        _normalize_diffusion_stabilization(stabilization, space),
        q.MKrf, q.MKrf_inv,
        q.weighted_triple_phi_flat.reshape(q.el_dof, q.el_dof, q.el_dof),
        q.face_element_test_element_trial, d0, d1, coeffs))
    digest = hashlib.sha256()
    for value in arrays:
        digest.update(str(value.shape).encode())
        if value.size:
            digest.update(memoryview(value).cast("B"))
    digest.update(repr((float(scalar), bool(is_scalar))).encode())
    return arrays + (float(scalar), bool(is_scalar)), digest.hexdigest()


def build_diffusion_schur_cache_numba(reaction, stabilization, space, *, factor_kind="schur-lu"):
    """Construct reusable LU or Cholesky factors for the scalar local Schur blocks.

    Geometry, reference data, reaction and stabilization participate in the
    cache signature. Source and Dirichlet values can change without refactoring.
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Schur factor caching requires numba")
    if factor_kind not in {"schur-lu", "schur-cholesky"}:
        raise ValueError("factor_kind must be 'schur-lu' or 'schur-cholesky'")
    start = time.perf_counter()
    inputs, key = _diffusion_schur_inputs(reaction, stabilization, space)
    if any(not np.all(np.isfinite(value)) for value in inputs[:-2]) or not np.isfinite(inputs[-2]):
        raise ValueError("Schur factor inputs must be finite")
    if factor_kind == "schur-cholesky" and np.any(inputs[4] <= 0.0):
        raise ValueError("Schur-Cholesky caching requires strictly positive stabilization")
    count, nel = space.mesh.num_tri, space.quad_data.el_dof
    factors = np.empty((count, nel, nel), dtype=np.float64)
    pivots = np.zeros((count, nel if factor_kind == "schur-lu" else 0), dtype=np.int64)
    status = np.empty(count, dtype=np.int64)
    factor_projected_diffusion_schur_kernel(
        factors, pivots, status, 2 if factor_kind == "schur-cholesky" else 1, *inputs)
    failed = np.flatnonzero(status)
    if failed.size:
        element = int(failed[0])
        raise ValueError(f"Schur Cholesky failed on element {element}: "
                         f"status={status[element]} (nonfinite/asymmetric or not positive definite)")
    return NumbaDiffusionSchurCache(factors, pivots, factor_kind, key, time.perf_counter() - start)


def _diffusion_schur_cache_arguments(cache, reaction, stabilization, space):
    """Reject stale factors before dispatch; return normalized kernel arguments."""
    if cache is None:
        return 0, None, None
    if not isinstance(cache, NumbaDiffusionSchurCache):
        raise TypeError("cached_factors must be a NumbaDiffusionSchurCache")
    count, nel = space.mesh.num_tri, space.quad_data.el_dof
    if cache.factor_kind not in {"schur-lu", "schur-cholesky"}:
        raise ValueError("invalid Schur factor cache policy")
    width = nel if cache.factor_kind == "schur-lu" else 0
    if (cache.factors.shape != (count, nel, nel)
            or cache.pivots.shape != (count, width)
            or cache.factors.dtype != np.float64 or cache.pivots.dtype != np.int64
            or not cache.factors.flags.c_contiguous or not cache.pivots.flags.c_contiguous):
        raise ValueError("invalid Schur factor cache shape, dtype or layout")
    _, key = _diffusion_schur_inputs(reaction, stabilization, space)
    if key != cache.operator_key:
        raise ValueError("stale diffusion Schur cache: operator inputs changed")
    return (2 if cache.factor_kind == "schur-cholesky" else 1), cache.factors, cache.pivots


def _projected_tensor_component_coefficients(component, space: DGSpace, label: str) -> np.ndarray:
    """Return DG coefficients for internally projected tensor components."""
    if isinstance(component, DGField):
        return _same_space_field_coefficients(component, space, label)
    values = np.asarray(component, dtype=np.float64)
    if values.shape == space.shape:
        return np.ascontiguousarray(values, dtype=np.float64)
    raise TypeError(f"{label} must be a DGField or coefficient array with shape {space.shape}")


def _projected_tensor_coefficients(tensor, space: DGSpace, label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return four same-space coefficient arrays for a projected tensor."""
    if not isinstance(tensor, (tuple, list)) or len(tensor) != 4:
        raise TypeError(f"{label} must be a 4-tuple of projected coefficient arrays")
    return (
        _projected_tensor_component_coefficients(tensor[0], space, f"{label}[0,0]"),
        _projected_tensor_component_coefficients(tensor[1], space, f"{label}[0,1]"),
        _projected_tensor_component_coefficients(tensor[2], space, f"{label}[1,0]"),
        _projected_tensor_component_coefficients(tensor[3], space, f"{label}[1,1]"),
    )


def _diffusion_face_element_trace(trace_ref: DGTraceSpace) -> np.ndarray:
    """Return ``(face, element-test, trace-trial)`` diffusion coupling table."""
    return np.ascontiguousarray(
        trace_ref.face_trace_test_element_trial_oriented[:3].transpose(0, 2, 1),
        dtype=np.float64,
    )


def _reference_diffusion_derivative_matrices(space: DGSpace) -> tuple[np.ndarray, np.ndarray]:
    """Return reference derivative matrices in diffusion local-solver layout."""
    q = space.quad_data
    d0 = np.einsum("q,iq,jq->ij", q.Krf_w, q.bas_of_quads, q.dbas_of_quads[0], optimize=True)
    d1 = np.einsum("q,iq,jq->ij", q.Krf_w, q.bas_of_quads, q.dbas_of_quads[1], optimize=True)
    return np.ascontiguousarray(d0.T), np.ascontiguousarray(d1.T)


def _normalize_diffusion_stabilization(stabilization, space: DGSpace) -> np.ndarray:
    """Return diffusion stabilization values with shape ``(num_elements, 3)``."""
    mesh = space.mesh
    if np.isscalar(stabilization):
        return np.full((mesh.num_tri, 3), float(stabilization), dtype=np.float64)
    tau = np.asarray(stabilization, dtype=np.float64)
    if tau.shape == (mesh.num_tri,):
        return np.ascontiguousarray(np.broadcast_to(tau[:, None], (mesh.num_tri, 3)))
    if tau.shape != (mesh.num_tri, 3):
        raise ValueError(f"stabilization must be scalar or have shape ({mesh.num_tri}, 3); got {tau.shape}")
    return np.ascontiguousarray(tau)


def assemble_projected_diffusion_trace_system_eliminated_numba(
        source,
        reaction,
        boundary_condition: Callable,
        stabilization,
        space: DGSpace,
        *,
        edge_order: np.ndarray | None = None,
        trace_space: DGTraceSpace | None = None,
        cached_factors: NumbaDiffusionSchurCache | None = None,
) -> NumbaDiffusionTraceAssembly:
    """Assemble the reduced projected diffusion-reaction trace system in Numba."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")

    timings: dict[str, float] = {}
    start = time.perf_counter()
    factor_arguments = _diffusion_schur_cache_arguments(
        cached_factors, reaction, stabilization, space)
    trace_ref = _trace_ref(space, trace_space)
    trace_orientation_mode = _trace_orientation_mode(trace_ref)
    source_coeffs, source_kind = _source_coefficients(source, space)
    reaction_coeffs, reaction_scalar, reaction_is_scalar = _reaction_coefficients(reaction, space)
    tau = _normalize_diffusion_stabilization(stabilization, space)
    interior_side_index = _interior_side_index(space.mesh)
    d0_reference, d1_reference = _reference_diffusion_derivative_matrices(space)
    face_element_trace = _diffusion_face_element_trace(trace_ref)
    timings["coefficient_validation"] = time.perf_counter() - start

    start = time.perf_counter()
    boundary_trace = hdg_assembly.boundary_trace_coefficients(boundary_condition, space, trace_space=trace_ref)
    mesh = space.mesh
    q = space.quad_data
    edg_dof = trace_ref.edg_dof
    edge_to_solve_edge, free_edges, reduction_template = _boundary_reduction_maps(
        space,
        boundary_trace,
        edge_order,
        trace_space=trace_ref,
    )
    valid_elements = np.ascontiguousarray(mesh.interior_elements, dtype=np.int64)
    valid_faces = np.ascontiguousarray(mesh.interior_faces, dtype=np.int64)
    face_is_free = edge_to_solve_edge[mesh.loc2glob_edge] >= 0
    side_col_counts = np.count_nonzero(face_is_free[valid_elements], axis=1).astype(np.int64)
    side_flux_offsets = np.empty(side_col_counts.size + 1, dtype=np.int64)
    side_flux_offsets[0] = 0
    np.cumsum(side_col_counts * edg_dof * edg_dof, out=side_flux_offsets[1:])
    n_flux = int(side_flux_offsets[-1])
    n_mass = valid_elements.size * edg_dof * edg_dof
    nnz = n_flux + n_mass
    timings["reduction_map"] = time.perf_counter() - start

    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty_like(rows)
    data = np.empty(nnz, dtype=np.float64)
    rhs_indices = np.empty(mesh.num_tri * 3 * edg_dof, dtype=np.int64)
    rhs_values = np.empty_like(rhs_indices, dtype=np.float64)

    start = time.perf_counter()
    assemble_projected_diffusion_trace_system_eliminated_kernel(
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
        np.ascontiguousarray(mesh.aff_jacs, dtype=np.float64),
        np.ascontiguousarray(mesh.jacs_el_fc, dtype=np.float64),
        np.ascontiguousarray(mesh.normals, dtype=np.float64),
        tau,
        np.ascontiguousarray(q.MKrf, dtype=np.float64),
        np.ascontiguousarray(q.MKrf_inv, dtype=np.float64),
        np.ascontiguousarray(q.weighted_triple_phi_flat.reshape(q.el_dof, q.el_dof, q.el_dof), dtype=np.float64),
        np.ascontiguousarray(q.face_element_test_element_trial, dtype=np.float64),
        face_element_trace,
        np.ascontiguousarray(trace_ref.M_rf_fc, dtype=np.float64),
        np.ascontiguousarray(trace_ref.face_trace_test_element_trial_oriented, dtype=np.float64),
        d0_reference,
        d1_reference,
        source_coeffs,
        int(source_kind),
        np.ascontiguousarray(reaction_coeffs, dtype=np.float64),
        float(reaction_scalar),
        bool(reaction_is_scalar),
        np.ascontiguousarray(boundary_trace, dtype=np.float64),
        int(trace_orientation_mode),
        *factor_arguments,
    )
    timings["kernel"] = time.perf_counter() - start

    start = time.perf_counter()
    rhs = np.zeros(free_edges.size * edg_dof, dtype=np.float64)
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


def assemble_projected_tensor_diffusion_trace_system_eliminated_numba(
        source,
        reaction,
        diffusion_inverse,
        boundary_condition: Callable,
        stabilization,
        space: DGSpace,
        *,
        edge_order: np.ndarray | None = None,
        trace_space: DGTraceSpace | None = None,
) -> NumbaDiffusionTraceAssembly:
    """Assemble the reduced projected tensor diffusion-reaction trace system."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")

    timings: dict[str, float] = {}
    start = time.perf_counter()
    trace_ref = _trace_ref(space, trace_space)
    trace_orientation_mode = _trace_orientation_mode(trace_ref)
    source_coeffs, source_kind = _source_coefficients(source, space)
    reaction_coeffs, reaction_scalar, reaction_is_scalar = _reaction_coefficients(reaction, space)
    inv00_coeffs, inv01_coeffs, inv10_coeffs, inv11_coeffs = _projected_tensor_coefficients(
        diffusion_inverse,
        space,
        "diffusion_inverse",
    )
    tau = _normalize_diffusion_stabilization(stabilization, space)
    interior_side_index = _interior_side_index(space.mesh)
    d0_reference, d1_reference = _reference_diffusion_derivative_matrices(space)
    face_element_trace = _diffusion_face_element_trace(trace_ref)
    timings["coefficient_validation"] = time.perf_counter() - start

    start = time.perf_counter()
    boundary_trace = hdg_assembly.boundary_trace_coefficients(boundary_condition, space, trace_space=trace_ref)
    mesh = space.mesh
    q = space.quad_data
    edg_dof = trace_ref.edg_dof
    edge_to_solve_edge, free_edges, reduction_template = _boundary_reduction_maps(
        space,
        boundary_trace,
        edge_order,
        trace_space=trace_ref,
    )
    valid_elements = np.ascontiguousarray(mesh.interior_elements, dtype=np.int64)
    valid_faces = np.ascontiguousarray(mesh.interior_faces, dtype=np.int64)
    face_is_free = edge_to_solve_edge[mesh.loc2glob_edge] >= 0
    side_col_counts = np.count_nonzero(face_is_free[valid_elements], axis=1).astype(np.int64)
    side_flux_offsets = np.empty(side_col_counts.size + 1, dtype=np.int64)
    side_flux_offsets[0] = 0
    np.cumsum(side_col_counts * edg_dof * edg_dof, out=side_flux_offsets[1:])
    n_flux = int(side_flux_offsets[-1])
    n_mass = valid_elements.size * edg_dof * edg_dof
    nnz = n_flux + n_mass
    timings["reduction_map"] = time.perf_counter() - start

    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty_like(rows)
    data = np.empty(nnz, dtype=np.float64)
    rhs_indices = np.empty(mesh.num_tri * 3 * edg_dof, dtype=np.int64)
    rhs_values = np.empty_like(rhs_indices, dtype=np.float64)

    start = time.perf_counter()
    assemble_projected_tensor_diffusion_trace_system_eliminated_kernel(
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
        np.ascontiguousarray(mesh.aff_jacs, dtype=np.float64),
        np.ascontiguousarray(mesh.jacs_el_fc, dtype=np.float64),
        np.ascontiguousarray(mesh.normals, dtype=np.float64),
        tau,
        np.ascontiguousarray(q.MKrf, dtype=np.float64),
        np.ascontiguousarray(q.weighted_triple_phi_flat.reshape(q.el_dof, q.el_dof, q.el_dof), dtype=np.float64),
        np.ascontiguousarray(q.face_element_test_element_trial, dtype=np.float64),
        face_element_trace,
        np.ascontiguousarray(trace_ref.M_rf_fc, dtype=np.float64),
        np.ascontiguousarray(trace_ref.face_trace_test_element_trial_oriented, dtype=np.float64),
        d0_reference,
        d1_reference,
        source_coeffs,
        int(source_kind),
        np.ascontiguousarray(reaction_coeffs, dtype=np.float64),
        float(reaction_scalar),
        bool(reaction_is_scalar),
        inv00_coeffs,
        inv01_coeffs,
        inv10_coeffs,
        inv11_coeffs,
        np.ascontiguousarray(boundary_trace, dtype=np.float64),
        int(trace_orientation_mode),
    )
    timings["kernel"] = time.perf_counter() - start

    start = time.perf_counter()
    rhs = np.zeros(free_edges.size * edg_dof, dtype=np.float64)
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


def assemble_projected_diffusion_trace_rhs_eliminated_numba(
        source,
        reaction,
        boundary_condition: Callable,
        stabilization,
        space: DGSpace,
        *,
        edge_order: np.ndarray | None = None,
        trace_space: DGTraceSpace | None = None,
        cached_factors: NumbaDiffusionSchurCache | None = None,
) -> tuple[np.ndarray, np.ndarray, KnownDofReduction, dict[str, float]]:
    """Assemble only the reduced RHS for a cached projected diffusion matrix."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")

    timings: dict[str, float] = {}
    start = time.perf_counter()
    factor_arguments = _diffusion_schur_cache_arguments(
        cached_factors, reaction, stabilization, space)
    trace_ref = _trace_ref(space, trace_space)
    trace_orientation_mode = _trace_orientation_mode(trace_ref)
    source_coeffs, source_kind = _source_coefficients(source, space)
    reaction_coeffs, reaction_scalar, reaction_is_scalar = _reaction_coefficients(reaction, space)
    tau = _normalize_diffusion_stabilization(stabilization, space)
    interior_side_index = _interior_side_index(space.mesh)
    d0_reference, d1_reference = _reference_diffusion_derivative_matrices(space)
    face_element_trace = _diffusion_face_element_trace(trace_ref)
    boundary_trace = hdg_assembly.boundary_trace_coefficients(boundary_condition, space, trace_space=trace_ref)
    mesh = space.mesh
    q = space.quad_data
    edg_dof = trace_ref.edg_dof
    edge_to_solve_edge, free_edges, reduction_template = _boundary_reduction_maps(
        space,
        boundary_trace,
        edge_order,
        trace_space=trace_ref,
    )
    timings["preparation"] = time.perf_counter() - start

    rhs_indices = np.empty(mesh.num_tri * 3 * edg_dof, dtype=np.int64)
    rhs_values = np.empty_like(rhs_indices, dtype=np.float64)

    start = time.perf_counter()
    assemble_projected_diffusion_trace_rhs_eliminated_kernel(
        rhs_indices,
        rhs_values,
        np.ascontiguousarray(mesh.loc2glob_edge, dtype=np.int64),
        np.ascontiguousarray(mesh.orientations, dtype=np.bool_),
        np.ascontiguousarray(mesh.loc2oriented_face_coupling, dtype=np.int64),
        interior_side_index,
        np.ascontiguousarray(edge_to_solve_edge, dtype=np.int64),
        np.ascontiguousarray(mesh.aff_mats, dtype=np.float64),
        np.ascontiguousarray(mesh.aff_jacs, dtype=np.float64),
        np.ascontiguousarray(mesh.jacs_el_fc, dtype=np.float64),
        np.ascontiguousarray(mesh.normals, dtype=np.float64),
        tau,
        np.ascontiguousarray(q.MKrf, dtype=np.float64),
        np.ascontiguousarray(q.MKrf_inv, dtype=np.float64),
        np.ascontiguousarray(q.weighted_triple_phi_flat.reshape(q.el_dof, q.el_dof, q.el_dof), dtype=np.float64),
        np.ascontiguousarray(q.face_element_test_element_trial, dtype=np.float64),
        face_element_trace,
        np.ascontiguousarray(trace_ref.face_trace_test_element_trial_oriented, dtype=np.float64),
        d0_reference,
        d1_reference,
        source_coeffs,
        int(source_kind),
        np.ascontiguousarray(reaction_coeffs, dtype=np.float64),
        float(reaction_scalar),
        bool(reaction_is_scalar),
        np.ascontiguousarray(boundary_trace, dtype=np.float64),
        int(trace_orientation_mode),
        *factor_arguments,
    )
    timings["kernel"] = time.perf_counter() - start

    start = time.perf_counter()
    rhs = np.zeros(free_edges.size * edg_dof, dtype=np.float64)
    np.add.at(rhs, rhs_indices, rhs_values)
    timings["rhs_finalization"] = time.perf_counter() - start
    timings["total"] = sum(timings.values())
    reduction = _reduction_with_system(
        reduction_template,
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float64),
        rhs,
    )
    return rhs, boundary_trace, reduction, timings


def assemble_diffusion_trace_system_eliminated_numba(
        local_solver: np.ndarray,
        element_boundary_mats: np.ndarray,
        source_rhs: np.ndarray,
        boundary_condition: Callable,
        stabilization,
        space: DGSpace,
        *,
        edge_order: np.ndarray | None = None,
        trace_space: DGTraceSpace | None = None,
) -> NumbaDiffusionTraceAssembly:
    """Assemble a boundary-eliminated diffusion-reaction trace system.

    This adapter mirrors the strong-boundary path used by the advection
    backend: only non-boundary trace dofs appear in the emitted COO matrix, and
    contributions from prescribed boundary trace columns are accumulated
    directly into the reduced RHS.
    """
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")

    timings: dict[str, float] = {}
    start = time.perf_counter()
    mesh = space.mesh
    q = space.quad_data
    trace_ref = _trace_ref(space, trace_space)
    trace_orientation_mode = _trace_orientation_mode(trace_ref)
    edg_dof = trace_ref.edg_dof
    local_solver = np.ascontiguousarray(local_solver, dtype=np.float64)
    element_boundary_mats = np.ascontiguousarray(element_boundary_mats, dtype=np.float64)
    source_rhs = np.ascontiguousarray(source_rhs, dtype=np.float64)
    tau = _normalize_diffusion_stabilization(stabilization, space)
    expected_solver = (mesh.num_tri, 3 * q.el_dof, 3 * q.el_dof)
    expected_boundary = (mesh.num_tri, 3 * q.el_dof, 3 * edg_dof)
    expected_rhs = (mesh.num_tri, 3 * q.el_dof)
    if local_solver.shape != expected_solver:
        raise ValueError(f"local_solver must have shape {expected_solver}; got {local_solver.shape}")
    if element_boundary_mats.shape != expected_boundary:
        raise ValueError(f"element_boundary_mats must have shape {expected_boundary}; got {element_boundary_mats.shape}")
    if source_rhs.shape != expected_rhs:
        raise ValueError(f"source_rhs must have shape {expected_rhs}; got {source_rhs.shape}")
    timings["input_validation"] = time.perf_counter() - start

    start = time.perf_counter()
    boundary_trace = hdg_assembly.boundary_trace_coefficients(boundary_condition, space, trace_space=trace_ref)
    edge_to_solve_edge, free_edges, reduction_template = _boundary_reduction_maps(
        space,
        boundary_trace,
        edge_order,
        trace_space=trace_ref,
    )
    valid_elements = np.ascontiguousarray(mesh.interior_elements, dtype=np.int64)
    valid_faces = np.ascontiguousarray(mesh.interior_faces, dtype=np.int64)
    face_is_free = edge_to_solve_edge[mesh.loc2glob_edge] >= 0
    side_col_counts = np.count_nonzero(face_is_free[valid_elements], axis=1).astype(np.int64)
    side_flux_offsets = np.empty(side_col_counts.size + 1, dtype=np.int64)
    side_flux_offsets[0] = 0
    np.cumsum(side_col_counts * edg_dof * edg_dof, out=side_flux_offsets[1:])
    n_flux = int(side_flux_offsets[-1])
    n_mass = valid_elements.size * edg_dof * edg_dof
    nnz = n_flux + n_mass
    timings["reduction_map"] = time.perf_counter() - start

    rows = np.empty(nnz, dtype=np.int64)
    cols = np.empty_like(rows)
    data = np.empty(nnz, dtype=np.float64)
    rhs_indices = np.empty(mesh.num_tri * 3 * edg_dof, dtype=np.int64)
    rhs_values = np.empty_like(rhs_indices, dtype=np.float64)

    start = time.perf_counter()
    assemble_diffusion_trace_system_eliminated_kernel(
        rows,
        cols,
        data,
        rhs_indices,
        rhs_values,
        np.ascontiguousarray(mesh.loc2glob_edge, dtype=np.int64),
        np.ascontiguousarray(mesh.orientations, dtype=np.bool_),
        np.ascontiguousarray(mesh.loc2oriented_face_coupling, dtype=np.int64),
        _interior_side_index(space.mesh),
        np.ascontiguousarray(edge_to_solve_edge, dtype=np.int64),
        valid_elements,
        valid_faces,
        side_flux_offsets,
        np.ascontiguousarray(mesh.jacs_el_fc, dtype=np.float64),
        np.ascontiguousarray(mesh.normals, dtype=np.float64),
        tau,
        np.ascontiguousarray(trace_ref.M_rf_fc, dtype=np.float64),
        np.ascontiguousarray(trace_ref.face_trace_test_element_trial_oriented, dtype=np.float64),
        local_solver,
        element_boundary_mats,
        source_rhs,
        np.ascontiguousarray(boundary_trace, dtype=np.float64),
        int(trace_orientation_mode),
    )
    timings["kernel"] = time.perf_counter() - start

    start = time.perf_counter()
    rhs = np.zeros(free_edges.size * edg_dof, dtype=np.float64)
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


def assemble_diffusion_trace_rhs_eliminated_numba(
        local_solver: np.ndarray,
        element_boundary_mats: np.ndarray,
        source_rhs: np.ndarray,
        boundary_condition: Callable,
        stabilization,
        space: DGSpace,
        *,
        edge_order: np.ndarray | None = None,
        trace_space: DGTraceSpace | None = None,
) -> tuple[np.ndarray, np.ndarray, KnownDofReduction, dict[str, float]]:
    """Assemble only the reduced RHS for a cached diffusion trace operator."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")

    timings: dict[str, float] = {}
    start = time.perf_counter()
    mesh = space.mesh
    trace_ref = _trace_ref(space, trace_space)
    trace_orientation_mode = _trace_orientation_mode(trace_ref)
    edg_dof = trace_ref.edg_dof
    local_solver = np.ascontiguousarray(local_solver, dtype=np.float64)
    element_boundary_mats = np.ascontiguousarray(element_boundary_mats, dtype=np.float64)
    source_rhs = np.ascontiguousarray(source_rhs, dtype=np.float64)
    tau = _normalize_diffusion_stabilization(stabilization, space)
    boundary_trace = hdg_assembly.boundary_trace_coefficients(boundary_condition, space, trace_space=trace_ref)
    edge_to_solve_edge, free_edges, reduction_template = _boundary_reduction_maps(
        space,
        boundary_trace,
        edge_order,
        trace_space=trace_ref,
    )
    timings["preparation"] = time.perf_counter() - start

    rhs_indices = np.empty(mesh.num_tri * 3 * edg_dof, dtype=np.int64)
    rhs_values = np.empty_like(rhs_indices, dtype=np.float64)

    start = time.perf_counter()
    assemble_diffusion_trace_rhs_eliminated_kernel(
        rhs_indices,
        rhs_values,
        np.ascontiguousarray(mesh.loc2glob_edge, dtype=np.int64),
        np.ascontiguousarray(mesh.orientations, dtype=np.bool_),
        np.ascontiguousarray(mesh.loc2oriented_face_coupling, dtype=np.int64),
        _interior_side_index(space.mesh),
        np.ascontiguousarray(edge_to_solve_edge, dtype=np.int64),
        np.ascontiguousarray(mesh.jacs_el_fc, dtype=np.float64),
        np.ascontiguousarray(mesh.normals, dtype=np.float64),
        tau,
        np.ascontiguousarray(trace_ref.face_trace_test_element_trial_oriented, dtype=np.float64),
        local_solver,
        element_boundary_mats,
        source_rhs,
        np.ascontiguousarray(boundary_trace, dtype=np.float64),
        int(trace_orientation_mode),
    )
    timings["kernel"] = time.perf_counter() - start

    start = time.perf_counter()
    rhs = np.zeros(free_edges.size * edg_dof, dtype=np.float64)
    np.add.at(rhs, rhs_indices, rhs_values)
    timings["rhs_finalization"] = time.perf_counter() - start
    timings["total"] = sum(timings.values())
    reduction = _reduction_with_system(
        reduction_template,
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.float64),
        rhs,
    )
    return rhs, boundary_trace, reduction, timings


def reconstruct_diffusion_local_unknowns_numba(
        trace: np.ndarray,
        source_rhs: np.ndarray,
        local_solver: np.ndarray,
        element_boundary_mats: np.ndarray,
        space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    """Recover mixed diffusion local unknowns with the Numba backend."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")

    mesh = space.mesh
    q = space.quad_data
    trace_ref = _trace_ref(space, trace_space)
    trace_orientation_mode = _trace_orientation_mode(trace_ref)
    edg_dof = trace_ref.edg_dof
    trace = np.ascontiguousarray(np.asarray(trace, dtype=np.float64))
    expected_trace_shape = (mesh.num_edg * edg_dof,)
    if trace.shape != expected_trace_shape:
        raise ValueError(f"trace must have shape {expected_trace_shape}; got {trace.shape}")

    local_solver = np.ascontiguousarray(local_solver, dtype=np.float64)
    element_boundary_mats = np.ascontiguousarray(element_boundary_mats, dtype=np.float64)
    source_rhs = np.ascontiguousarray(source_rhs, dtype=np.float64)
    local_unknowns = np.empty((mesh.num_tri, 3 * q.el_dof), dtype=np.float64)
    reconstruct_diffusion_local_unknowns_kernel(
        local_unknowns,
        trace,
        np.ascontiguousarray(mesh.loc2glob_edge, dtype=np.int64),
        np.ascontiguousarray(mesh.orientations, dtype=np.bool_),
        local_solver,
        element_boundary_mats,
        source_rhs,
        int(trace_orientation_mode),
    )
    return local_unknowns


def reconstruct_projected_diffusion_local_unknowns_numba(
        trace: np.ndarray,
        source,
        reaction,
        stabilization,
        space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
        cached_factors: NumbaDiffusionSchurCache | None = None,
) -> np.ndarray:
    """Recover mixed diffusion local unknowns with fully fused projected kernels."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")

    factor_arguments = _diffusion_schur_cache_arguments(
        cached_factors, reaction, stabilization, space)
    trace_ref = _trace_ref(space, trace_space)
    trace_orientation_mode = _trace_orientation_mode(trace_ref)
    source_coeffs, source_kind = _source_coefficients(source, space)
    reaction_coeffs, reaction_scalar, reaction_is_scalar = _reaction_coefficients(reaction, space)
    tau = _normalize_diffusion_stabilization(stabilization, space)
    d0_reference, d1_reference = _reference_diffusion_derivative_matrices(space)
    face_element_trace = _diffusion_face_element_trace(trace_ref)
    mesh = space.mesh
    q = space.quad_data
    edg_dof = trace_ref.edg_dof
    trace = np.ascontiguousarray(np.asarray(trace, dtype=np.float64))
    expected_trace_shape = (mesh.num_edg * edg_dof,)
    if trace.shape != expected_trace_shape:
        raise ValueError(f"trace must have shape {expected_trace_shape}; got {trace.shape}")

    local_unknowns = np.empty((mesh.num_tri, 3 * q.el_dof), dtype=np.float64)
    reconstruct_projected_diffusion_local_unknowns_kernel(
        local_unknowns,
        trace,
        np.ascontiguousarray(mesh.loc2glob_edge, dtype=np.int64),
        np.ascontiguousarray(mesh.orientations, dtype=np.bool_),
        np.ascontiguousarray(mesh.aff_mats, dtype=np.float64),
        np.ascontiguousarray(mesh.aff_jacs, dtype=np.float64),
        np.ascontiguousarray(mesh.jacs_el_fc, dtype=np.float64),
        np.ascontiguousarray(mesh.normals, dtype=np.float64),
        tau,
        np.ascontiguousarray(q.MKrf, dtype=np.float64),
        np.ascontiguousarray(q.MKrf_inv, dtype=np.float64),
        np.ascontiguousarray(q.weighted_triple_phi_flat.reshape(q.el_dof, q.el_dof, q.el_dof), dtype=np.float64),
        np.ascontiguousarray(q.face_element_test_element_trial, dtype=np.float64),
        face_element_trace,
        d0_reference,
        d1_reference,
        source_coeffs,
        int(source_kind),
        np.ascontiguousarray(reaction_coeffs, dtype=np.float64),
        float(reaction_scalar),
        bool(reaction_is_scalar),
        int(trace_orientation_mode),
        *factor_arguments,
    )
    return local_unknowns


def reconstruct_projected_tensor_diffusion_local_unknowns_numba(
        trace: np.ndarray,
        source,
        reaction,
        diffusion_inverse,
        stabilization,
        space: DGSpace,
        *,
        trace_space: DGTraceSpace | None = None,
) -> np.ndarray:
    """Recover mixed local unknowns with projected tensor fused kernels."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError("assembly_backend='numba' requires numba")

    trace_ref = _trace_ref(space, trace_space)
    trace_orientation_mode = _trace_orientation_mode(trace_ref)
    source_coeffs, source_kind = _source_coefficients(source, space)
    reaction_coeffs, reaction_scalar, reaction_is_scalar = _reaction_coefficients(reaction, space)
    inv00_coeffs, inv01_coeffs, inv10_coeffs, inv11_coeffs = _projected_tensor_coefficients(
        diffusion_inverse,
        space,
        "diffusion_inverse",
    )
    tau = _normalize_diffusion_stabilization(stabilization, space)
    d0_reference, d1_reference = _reference_diffusion_derivative_matrices(space)
    face_element_trace = _diffusion_face_element_trace(trace_ref)
    mesh = space.mesh
    q = space.quad_data
    edg_dof = trace_ref.edg_dof
    trace = np.ascontiguousarray(np.asarray(trace, dtype=np.float64))
    expected_trace_shape = (mesh.num_edg * edg_dof,)
    if trace.shape != expected_trace_shape:
        raise ValueError(f"trace must have shape {expected_trace_shape}; got {trace.shape}")

    local_unknowns = np.empty((mesh.num_tri, 3 * q.el_dof), dtype=np.float64)
    reconstruct_projected_tensor_diffusion_local_unknowns_kernel(
        local_unknowns,
        trace,
        np.ascontiguousarray(mesh.loc2glob_edge, dtype=np.int64),
        np.ascontiguousarray(mesh.orientations, dtype=np.bool_),
        np.ascontiguousarray(mesh.aff_mats, dtype=np.float64),
        np.ascontiguousarray(mesh.aff_jacs, dtype=np.float64),
        np.ascontiguousarray(mesh.jacs_el_fc, dtype=np.float64),
        np.ascontiguousarray(mesh.normals, dtype=np.float64),
        tau,
        np.ascontiguousarray(q.MKrf, dtype=np.float64),
        np.ascontiguousarray(q.weighted_triple_phi_flat.reshape(q.el_dof, q.el_dof, q.el_dof), dtype=np.float64),
        np.ascontiguousarray(q.face_element_test_element_trial, dtype=np.float64),
        face_element_trace,
        d0_reference,
        d1_reference,
        source_coeffs,
        int(source_kind),
        np.ascontiguousarray(reaction_coeffs, dtype=np.float64),
        float(reaction_scalar),
        bool(reaction_is_scalar),
        inv00_coeffs,
        inv01_coeffs,
        inv10_coeffs,
        inv11_coeffs,
        int(trace_orientation_mode),
    )
    return local_unknowns


__all__ = [
    "NumbaDiffusionTraceAssembly",
    "assemble_diffusion_trace_rhs_eliminated_numba",
    "assemble_diffusion_trace_system_eliminated_numba",
    "assemble_projected_diffusion_trace_rhs_eliminated_numba",
    "assemble_projected_diffusion_trace_system_eliminated_numba",
    "assemble_projected_tensor_diffusion_trace_system_eliminated_numba",
    "reconstruct_diffusion_local_unknowns_numba",
    "reconstruct_projected_diffusion_local_unknowns_numba",
    "reconstruct_projected_tensor_diffusion_local_unknowns_numba",
]
