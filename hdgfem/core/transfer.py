"""Self-contained DG field evaluation and transfer utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numba as nb
import numpy as np

from . import basis as basis_module
from .space import DGField, DGSpace, VectorDGField, VectorDGSpace


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


def _points_in_triangles(points: np.ndarray, triangles_xy: np.ndarray, *, eps: float = 1e-12):
    """Locate points in triangles using vectorized barycentric tests per cell."""
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles_xy, dtype=np.float64)
    point_ids: list[np.ndarray] = []
    triangle_ids: list[np.ndarray] = []
    for tri_id, tri in enumerate(triangles):
        a, b, c = tri
        mat = np.column_stack((b - a, c - a))
        det = np.linalg.det(mat)
        if abs(det) < 1e-30:
            continue
        inv = np.linalg.inv(mat)
        uv = (points - a) @ inv.T
        u = uv[:, 0]
        v = uv[:, 1]
        w = 1.0 - u - v
        mask = (u >= -eps) & (v >= -eps) & (w >= -eps)
        ids = np.nonzero(mask)[0]
        if ids.size:
            point_ids.append(ids)
            triangle_ids.append(np.full(ids.size, tri_id, dtype=np.int64))
    if not point_ids:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    pt_idx = np.concatenate(point_ids)
    tri_idx = np.concatenate(triangle_ids)
    order = np.lexsort((tri_idx, pt_idx))
    return pt_idx[order], tri_idx[order]


def _deduplicate_hits(pt_idx_all: np.ndarray, src_idx_all: np.ndarray):
    if pt_idx_all.size == 0:
        return pt_idx_all, src_idx_all, 0
    duplicate_hits = int(pt_idx_all.size - np.unique(pt_idx_all).size)
    keep = np.concatenate(([True], pt_idx_all[1:] != pt_idx_all[:-1]))
    return pt_idx_all[keep], src_idx_all[keep], duplicate_hits


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

    t0 = time.perf_counter()
    pt_idx_all, src_idx_all = _points_in_triangles(qds_flat, source.mesh.element_vertices)
    locate_seconds = time.perf_counter() - t0
    pt_idx, src_idx, duplicate_hits = _deduplicate_hits(pt_idx_all, src_idx_all)
    missed = int(qds_flat.shape[0] - pt_idx.size)

    t0 = time.perf_counter()
    xi_src = (
        source.mesh.physical_to_reference(qds_flat[pt_idx], src_idx)
        if pt_idx.size
        else np.empty((0, 2), dtype=np.float64)
    )
    refmap_seconds = time.perf_counter() - t0

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
    rhs = (values * target.quad_data.Krf_w[None, :]) @ target.quad_data.bas_of_quads.T
    coeffs = rhs @ target.quad_data.MKrf_inv
    return np.ascontiguousarray(coeffs, dtype=np.float64)


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
        values = field.values_at_ref(target.quad_data.Krf_quads)
        eval_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        coeffs = _project_values_on_target(values, target)
        project_seconds = time.perf_counter() - t0
        diag = TransferDiagnostics(
            n_target_points=int(values.size),
            n_located_points=int(values.size),
            n_missed_points=0,
            n_duplicate_hits=0,
            locate_seconds=0.0,
            refmap_seconds=0.0,
            eval_seconds=eval_seconds,
            project_seconds=project_seconds,
            total_seconds=time.perf_counter() - total_start,
        )
        return target.field(coeffs, name=field.name), diag

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
    pt_idx_all, src_idx_all = _points_in_triangles(points, field.space.mesh.element_vertices)
    pt_idx, src_idx, _ = _deduplicate_hits(pt_idx_all, src_idx_all)
    result = np.full(points.shape[0], missing, dtype=np.float64)
    if pt_idx.size == 0:
        return result
    xi_src = field.space.mesh.physical_to_reference(points[pt_idx], src_idx)
    result[pt_idx] = _evaluate_coefficients_at_source_points(field, xi_src, src_idx)
    return result
