"""Self-contained DG field evaluation and transfer utilities."""

from __future__ import annotations

import time
from hdgfem.precision import REAL_DTYPE
from dataclasses import dataclass

import numba as nb
import numpy as np
from scipy.spatial import cKDTree

from hdgfem.core import basis as basis_module
from hdgfem.core.space import DGField, DGSpace, VectorDGField, VectorDGSpace


@dataclass(frozen=True)
class TransferDiagnostics:
    """Timing and point-location diagnostics for DG transfer."""

    n_target_points: int
    n_located_points: int
    n_missed_points: int
    n_duplicate_hits: int
    locate_seconds: float
    refmap_seconds: float
    eval_seconds: float
    project_seconds: float
    total_seconds: float


@dataclass(frozen=True)
class TransferPlan:
    """Reusable geometric transfer plan from one DG mesh to another."""

    qds_flat: np.ndarray
    pt_idx: np.ndarray
    src_idx: np.ndarray
    xi_src: np.ndarray
    n_target_points: int
    n_located_points: int
    n_missed_points: int
    n_duplicate_hits: int
    locate_seconds: float
    refmap_seconds: float


@nb.njit(parallel=True, cache=True, fastmath=True)
def _locate_points_from_candidates_kernel(
        points: np.ndarray,
        candidate_elements: np.ndarray,
        aff_vecs: np.ndarray,
        inv_aff_mats: np.ndarray,
        src_idx: np.ndarray,
        xi_src: np.ndarray,
        eps: float,
) -> None:
    """Select the first candidate triangle containing each point."""
    n_points = points.shape[0]
    n_candidates = candidate_elements.shape[1]
    for point in nb.prange(n_points):
        x = points[point, 0]
        y = points[point, 1]
        src_idx[point] = -1
        xi_src[point, 0] = np.nan
        xi_src[point, 1] = np.nan
        for candidate in range(n_candidates):
            element = candidate_elements[point, candidate]
            dx = x - aff_vecs[element, 0]
            dy = y - aff_vecs[element, 1]
            xi = inv_aff_mats[element, 0, 0] * dx + inv_aff_mats[element, 0, 1] * dy
            eta = inv_aff_mats[element, 1, 0] * dx + inv_aff_mats[element, 1, 1] * dy
            if xi >= -1.0 - eps and eta >= -1.0 - eps and xi + eta <= eps:
                src_idx[point] = element
                xi_src[point, 0] = xi
                xi_src[point, 1] = eta
                break


@nb.njit(parallel=True, cache=True, fastmath=True)
def _locate_points_exhaustive_kernel(
        points: np.ndarray,
        aff_vecs: np.ndarray,
        inv_aff_mats: np.ndarray,
        src_idx: np.ndarray,
        xi_src: np.ndarray,
        eps: float,
) -> None:
    """Locate points by checking every element, used only for KD-tree misses."""
    n_points = points.shape[0]
    n_elements = aff_vecs.shape[0]
    for point in nb.prange(n_points):
        x = points[point, 0]
        y = points[point, 1]
        src_idx[point] = -1
        xi_src[point, 0] = np.nan
        xi_src[point, 1] = np.nan
        for element in range(n_elements):
            dx = x - aff_vecs[element, 0]
            dy = y - aff_vecs[element, 1]
            xi = inv_aff_mats[element, 0, 0] * dx + inv_aff_mats[element, 0, 1] * dy
            eta = inv_aff_mats[element, 1, 0] * dx + inv_aff_mats[element, 1, 1] * dy
            if xi >= -1.0 - eps and eta >= -1.0 - eps and xi + eta <= eps:
                src_idx[point] = element
                xi_src[point, 0] = xi
                xi_src[point, 1] = eta
                break


def _normalize_candidate_indices(indices: np.ndarray, n_points: int) -> np.ndarray:
    """Normalize nearest-neighbor candidate ids to a two-dimensional array."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim == 1:
        indices = indices.reshape(n_points, 1)
    return np.ascontiguousarray(indices, dtype=np.int64)


def _locate_points_in_mesh(
        points: np.ndarray,
        mesh,
        *,
        neighbors: int = 32,
        retry_neighbors: int = 128,
        eps: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, float]:
    """Locate physical points in a mesh using nearest-centroid candidates.

    The expensive containment and physical-to-reference mapping work is done in
    a parallel numba kernel.  A second wider candidate search is used only for
    points missed by the first pass.
    """
    points = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (num_points, 2); got {points.shape}")
    n_points = points.shape[0]
    if n_points == 0:
        empty_i = np.empty(0, dtype=np.int64)
        empty_x = np.empty((0, 2), dtype=np.float64)
        return empty_i, empty_i, empty_x, 0, 0.0, 0.0

    aff_vecs = np.ascontiguousarray(mesh.aff_vecs, dtype=np.float64)
    inv_aff_mats = np.ascontiguousarray(mesh.inv_aff_mats, dtype=np.float64)
    centroids = np.ascontiguousarray(np.mean(mesh.element_vertices, axis=1), dtype=np.float64)
    tree_start = time.perf_counter()
    tree = cKDTree(centroids)
    locate_start = time.perf_counter()
    k = min(max(1, int(neighbors)), mesh.num_tri)
    _, candidate_elements = tree.query(points, k=k, workers=-1)
    candidate_elements = _normalize_candidate_indices(candidate_elements, n_points)

    src_all = np.full(n_points, -1, dtype=np.int64)
    xi_all = np.empty((n_points, 2), dtype=np.float64)
    _locate_points_from_candidates_kernel(
        points,
        candidate_elements,
        aff_vecs,
        inv_aff_mats,
        src_all,
        xi_all,
        float(eps),
    )

    missed_mask = src_all < 0
    if np.any(missed_mask) and retry_neighbors > k:
        missed_points = np.ascontiguousarray(points[missed_mask], dtype=np.float64)
        retry_k = min(max(k + 1, int(retry_neighbors)), mesh.num_tri)
        _, retry_candidates = tree.query(missed_points, k=retry_k, workers=-1)
        retry_candidates = _normalize_candidate_indices(retry_candidates, missed_points.shape[0])
        retry_src = np.full(missed_points.shape[0], -1, dtype=np.int64)
        retry_xi = np.empty((missed_points.shape[0], 2), dtype=np.float64)
        _locate_points_from_candidates_kernel(
            missed_points,
            retry_candidates,
            aff_vecs,
            inv_aff_mats,
            retry_src,
            retry_xi,
            float(eps),
        )
        missed_indices = np.nonzero(missed_mask)[0]
        found_retry = retry_src >= 0
        src_all[missed_indices[found_retry]] = retry_src[found_retry]
        xi_all[missed_indices[found_retry]] = retry_xi[found_retry]

    missed_mask = src_all < 0
    if np.any(missed_mask):
        missed_points = np.ascontiguousarray(points[missed_mask], dtype=np.float64)
        exhaustive_src = np.full(missed_points.shape[0], -1, dtype=np.int64)
        exhaustive_xi = np.empty((missed_points.shape[0], 2), dtype=np.float64)
        _locate_points_exhaustive_kernel(
            missed_points,
            aff_vecs,
            inv_aff_mats,
            exhaustive_src,
            exhaustive_xi,
            float(eps),
        )
        missed_indices = np.nonzero(missed_mask)[0]
        found_exhaustive = exhaustive_src >= 0
        src_all[missed_indices[found_exhaustive]] = exhaustive_src[found_exhaustive]
        xi_all[missed_indices[found_exhaustive]] = exhaustive_xi[found_exhaustive]

    located_mask = src_all >= 0
    pt_idx = np.ascontiguousarray(np.nonzero(located_mask)[0].astype(np.int64))
    src_idx = np.ascontiguousarray(src_all[located_mask], dtype=np.int64)
    xi_src = np.ascontiguousarray(xi_all[located_mask], dtype=np.float64)
    locate_seconds = time.perf_counter() - locate_start
    tree_seconds = locate_start - tree_start
    return pt_idx, src_idx, xi_src, 0, tree_seconds + locate_seconds, 0.0


@nb.njit(parallel=True, cache=True, fastmath=True)
def _evaluate_source_values_kernel(
        coeffs: np.ndarray,
        xi_src: np.ndarray,
        src_idx: np.ndarray,
        basis_kind: int,
        order: int,
        bernstein_exps: np.ndarray,
        bernstein_coeffs: np.ndarray,
        hierarchical_modes: np.ndarray,
        dubiner_pq: np.ndarray,
        values: np.ndarray,
) -> None:
    """Evaluate source DG coefficients at located source-reference points."""
    for point in nb.prange(xi_src.shape[0]):
        values[point] = basis_module.evaluate_modal_value(
            basis_kind,
            order,
            coeffs[src_idx[point]],
            xi_src[point, 0],
            xi_src[point, 1],
            bernstein_exps,
            bernstein_coeffs,
            hierarchical_modes,
            dubiner_pq,
        )


def _evaluate_coefficients_at_source_points(field: DGField, xi_src: np.ndarray, src_idx: np.ndarray) -> np.ndarray:
    """Evaluate a DG field at located source-element reference points."""
    if xi_src.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    basis_kind, bernstein_exps, bernstein_coeffs, hierarchical_modes, dubiner_pq = basis_module.modal_eval_payload(
        field.space.quad_data.basis_type,
        field.space.quad_data.order,
    )
    values = np.empty(xi_src.shape[0], dtype=np.float64)
    _evaluate_source_values_kernel(
        field.coeffs,
        np.ascontiguousarray(xi_src, dtype=np.float64),
        np.ascontiguousarray(src_idx, dtype=np.int64),
        basis_kind,
        int(field.space.quad_data.order),
        bernstein_exps,
        bernstein_coeffs,
        hierarchical_modes,
        dubiner_pq,
        values,
    )
    return values


def build_transfer_plan(
        source: DGSpace,
        target: DGSpace,
        *,
        verbose: bool = True,
) -> TransferPlan:
    """Build a reusable geometric transfer plan from ``source`` to ``target``."""
    start = time.perf_counter()
    qds_flat = target.mesh.flatten_mapped_reference_points(target.quad_data.Krf_quads)

    pt_idx, src_idx, xi_src, duplicate_hits, locate_seconds, refmap_seconds = _locate_points_in_mesh(
        qds_flat,
        source.mesh,
    )
    missed = int(qds_flat.shape[0] - pt_idx.size)

    plan = TransferPlan(
        qds_flat=np.ascontiguousarray(qds_flat),
        pt_idx=np.ascontiguousarray(pt_idx, dtype=np.int64),
        src_idx=np.ascontiguousarray(src_idx, dtype=np.int64),
        xi_src=np.ascontiguousarray(xi_src, dtype=np.float64),
        n_target_points=int(qds_flat.shape[0]),
        n_located_points=int(pt_idx.size),
        n_missed_points=missed,
        n_duplicate_hits=duplicate_hits,
        locate_seconds=locate_seconds,
        refmap_seconds=refmap_seconds,
    )
    if verbose:
        print(
            f"    hdgfem transfer plan        {time.perf_counter() - start:8.3f}s | "
            f"pts {plan.n_located_points}/{plan.n_target_points}, missed={plan.n_missed_points}"
        )
    return plan


def _project_values_on_target(values: np.ndarray, target: DGSpace) -> np.ndarray:
    """Project target-quadrature values into target DG coefficients."""
    rhs = (values * target.quad_data.Krf_w[None, :]) @ target.quad_data.bas_of_quads.T
    coeffs = rhs @ target.quad_data.MKrf_inv
    return np.ascontiguousarray(coeffs, dtype=np.float64)


def project_same_mesh_field(field: DGField, target: DGSpace, *, name: str | None = None) -> DGField:
    """L2-project between degrees on one mesh, retaining host/device residency.

    The reference projection is cached per pair of spaces. Integration uses
    the higher-degree space so restriction does not alias source modes into
    the lower-degree RHS. Identical spaces return the original field unless
    a different name is requested.
    """
    source = field.space
    source.assert_same_mesh(target)
    name = field.name if name is None else name
    if source is target:
        return field if name == field.name else field.copy(name=name)
    if field.constant_value is not None:
        return target.constant(field.constant_value, name=name)
    cache = getattr(target, "_same_mesh_projection_cache", None)
    if cache is None:
        cache = target._same_mesh_projection_cache = {}
    if source not in cache:
        integration = source if source.order >= target.order else target
        points, weights = integration.quad_data.Krf_quads, integration.quad_data.Krf_w
        cross_mass = source.basis_at(points).T @ (weights[:, None] * target.basis_at(points))
        matrix = np.ascontiguousarray(cross_mass @ target.quad_data.MKrf_inv, dtype=REAL_DTYPE)
        cache[source] = (matrix, {})
    matrix, device_matrices = cache[source]
    devices = field._device_coeffs or {}
    if devices:
        from hdgfem.backends.cupy import field_from_cupy_coefficients, require_cupy
        cp = require_cupy()
        device_id = min(devices)
        with cp.cuda.Device(device_id):
            if device_id not in device_matrices:
                device_matrices[device_id] = cp.asarray(matrix)
            coefficients = field._device_coefficients_for(device_id) @ device_matrices[device_id]
            return field_from_cupy_coefficients(
                target, cp.ascontiguousarray(coefficients), device=device_id, name=name)
    return target.field(np.ascontiguousarray(field.coeffs @ matrix), name=name)


def project_field(
        field: DGField,
        target: DGSpace,
        *,
        plan: TransferPlan | None = None,
        verbose: bool = True,
) -> tuple[DGField, TransferDiagnostics]:
    """L2-project ``field`` into ``target``."""
    total_start = time.perf_counter()
    source = field.space
    if source.mesh is target.mesh:
        t0 = time.perf_counter()
        projected = project_same_mesh_field(field, target)
        if projected is field:
            projected = field.copy()
        eval_seconds = 0.0
        project_seconds = time.perf_counter() - t0
        n_points = target.mesh.num_tri * max(source, target, key=lambda s: s.order).quad_data.Krf_w.size
        diag = TransferDiagnostics(
            n_target_points=int(n_points),
            n_located_points=int(n_points),
            n_missed_points=0,
            n_duplicate_hits=0,
            locate_seconds=0.0,
            refmap_seconds=0.0,
            eval_seconds=eval_seconds,
            project_seconds=project_seconds,
            total_seconds=time.perf_counter() - total_start,
        )
        return projected, diag

    if plan is None:
        plan = build_transfer_plan(source, target, verbose=verbose)

    t0 = time.perf_counter()
    values_flat = _evaluate_coefficients_at_source_points(field, plan.xi_src, plan.src_idx)
    eval_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    q_target = target.quad_data.Krf_quads.shape[0]
    target_values_flat = np.zeros(plan.n_target_points, dtype=np.float64)
    target_values_flat[plan.pt_idx] = values_flat
    target_values = target_values_flat.reshape(target.mesh.num_tri, q_target)
    coeffs = _project_values_on_target(target_values, target)
    project_seconds = time.perf_counter() - t0

    diag = TransferDiagnostics(
        n_target_points=plan.n_target_points,
        n_located_points=plan.n_located_points,
        n_missed_points=plan.n_missed_points,
        n_duplicate_hits=plan.n_duplicate_hits,
        locate_seconds=plan.locate_seconds,
        refmap_seconds=plan.refmap_seconds,
        eval_seconds=eval_seconds,
        project_seconds=project_seconds,
        total_seconds=time.perf_counter() - total_start,
    )
    if verbose:
        print(
            f"    hdgfem transfer scalar      {diag.total_seconds:8.3f}s | "
            f"pts {diag.n_located_points}/{diag.n_target_points}, missed={diag.n_missed_points}"
        )
    return target.field(coeffs, name=field.name), diag


def transfer_field(
        field: DGField,
        target: DGSpace,
        *,
        plan: TransferPlan | None = None,
        name: str | None = None,
        verbose: bool = True,
        warn_on_miss: bool = True,
) -> tuple[DGField, TransferDiagnostics]:
    """Project ``field`` into ``target`` and optionally rename the result.

    Parameters
    ----------
    field
        Scalar DG field to evaluate on the target space quadrature points.
    target
        DG space on the destination mesh.
    plan
        Optional precomputed transfer plan from ``field.space`` to ``target``.
        Supplying a plan avoids rebuilding point-location data when several
        fields are transferred between the same spaces.
    name
        Optional output field name.  If omitted, the source field name is kept.
    verbose
        If true, print the underlying projection diagnostics.
    warn_on_miss
        If true, print a warning when target quadrature points could not be
        located in the source mesh.

    Returns
    -------
    projected
        Scalar DG field in ``target``.
    diagnostics
        Point-location, evaluation, and projection timing/count diagnostics.

    Notes
    -----
    This is a convenience wrapper for adaptive workflows that repeatedly
    transfer named DG state fields to a newly generated mesh.  The returned
    diagnostics are the same as :func:`project_field`.
    """
    projected, diagnostics = project_field(field, target, plan=plan, verbose=verbose)
    if name is not None and name != projected.name:
        projected = target.field(projected.coeffs, name=name)
    if warn_on_miss and diagnostics.n_missed_points:
        print(
            f"TRANSFER_WARNING field={field.name} missed="
            f"{diagnostics.n_missed_points}/{diagnostics.n_target_points}",
            flush=True,
        )
    return projected, diagnostics


def project_vector_field(
        field: VectorDGField,
        target: VectorDGSpace,
        *,
        plan: TransferPlan | None = None,
        verbose: bool = True,
):
    """Project a vector DG field component-wise."""
    if field.dim != target.dim:
        raise ValueError(f"target dimension {target.dim} does not match field dimension {field.dim}")
    active_plan = plan
    if active_plan is None and field.components[0].space.mesh is not target.components[0].mesh:
        active_plan = build_transfer_plan(field.components[0].space, target.components[0], verbose=verbose)
    outputs = []
    diagnostics = []
    for component, target_space in zip(field.components, target.components):
        projected, diag = project_field(component, target_space, plan=active_plan, verbose=verbose)
        outputs.append(projected)
        diagnostics.append(diag)
    return VectorDGField(tuple(outputs), name=field.name), diagnostics


def evaluate_field_at_points(field: DGField, points_xy: np.ndarray, *, missing=np.nan) -> np.ndarray:
    """Evaluate a scalar DG field at arbitrary physical points."""
    points = np.asarray(points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points_xy must have shape (num_points, 2); got {points.shape}")
    pt_idx, src_idx, xi_src, _, _, _ = _locate_points_in_mesh(points, field.space.mesh)
    result = np.full(points.shape[0], missing, dtype=np.float64)
    if pt_idx.size == 0:
        return result
    result[pt_idx] = _evaluate_coefficients_at_source_points(field, xi_src, src_idx)
    return result
