"""CuPy assembly helpers for diffusion-reaction HDG trace systems."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..assembly import hdg as hdg_assembly
from ..core.space import DGField
from .cupy import as_cupy_coefficients, as_cupy_space, require_cupy
from .raw_cuda import RawCudaBlockSize
from .diffusion_raw_cuda import (
    RawDiffusionAssemblyResult,
    assemble_projected_diffusion_trace_rhs_eliminated_raw_cuda,
    assemble_projected_diffusion_trace_system_eliminated_raw_cuda,
    validate_raw_cuda_supported,
)


RAW_CUDA_MAX_EL_DOF = 28


@dataclass(frozen=True)
class TraceReferenceData:
    """Device trace-reference tables used by the GPU diffusion backends."""

    kind: str
    nodal: bool
    interpolation_nodes: Any
    quads: Any
    weights: Any
    bas_of_bd_quads: Any
    bas1d_of_ref_edg_qds: Any
    weighted_bas_of_bd_quads: Any
    weighted_bas1d_of_ref_edg_qds: Any
    face_element_test_trace_trial: Any
    face_trace_test_element_trial_oriented: Any
    M_rf_fc: Any


@dataclass(frozen=True)
class CupyDiffusionTraceAssembly:
    """Reduced diffusion trace system assembled on the CUDA device."""

    rows: Any | None
    cols: Any | None
    data: Any
    rhs: Any
    boundary_trace: Any
    timings: dict[str, float]
    matrix_format: str = "coo"
    indptr: Any | None = None
    indices: Any | None = None
    local_lhs: Any | None = None
    element_boundary_mats: Any | None = None
    source_rhs: Any | None = None
    raw_assembly: RawDiffusionAssemblyResult | None = None


def _as_scalar_or_none(value) -> float | None:
    """Return a finite scalar coefficient or None for non-scalar data."""
    if np.isscalar(value):
        return float(value)
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.shape == ():
        return float(array)
    return None


def _constant_inverse_diffusion_components(diffusion) -> tuple[float, float, float, float] | None:
    """Return inverse tensor components for a constant diffusion coefficient."""
    scalar = _as_scalar_or_none(diffusion)
    if scalar is not None:
        if scalar <= 0.0:
            raise ValueError("diffusion scalar must be positive")
        inv = 1.0 / scalar
        return inv, 0.0, 0.0, inv

    try:
        array = np.asarray(diffusion, dtype=np.float64)
    except (TypeError, ValueError):
        array = None
    if array is not None and array.shape == (2, 2):
        k00, k01 = float(array[0, 0]), float(array[0, 1])
        k10, k11 = float(array[1, 0]), float(array[1, 1])
    elif isinstance(diffusion, (tuple, list)):
        if len(diffusion) == 3:
            k00 = _as_scalar_or_none(diffusion[0])
            k01 = _as_scalar_or_none(diffusion[1])
            k11 = _as_scalar_or_none(diffusion[2])
            k10 = k01
        elif len(diffusion) == 4:
            k00 = _as_scalar_or_none(diffusion[0])
            k01 = _as_scalar_or_none(diffusion[1])
            k10 = _as_scalar_or_none(diffusion[2])
            k11 = _as_scalar_or_none(diffusion[3])
        elif (
            len(diffusion) == 2
            and all(isinstance(row, (tuple, list)) and len(row) == 2 for row in diffusion)
        ):
            k00 = _as_scalar_or_none(diffusion[0][0])
            k01 = _as_scalar_or_none(diffusion[0][1])
            k10 = _as_scalar_or_none(diffusion[1][0])
            k11 = _as_scalar_or_none(diffusion[1][1])
        else:
            return None
        if None in {k00, k01, k10, k11}:
            return None
    else:
        return None

    det = k00 * k11 - k01 * k10
    if det <= 0.0:
        raise ValueError(f"diffusion tensor must be positive definite; determinant is {det}")
    return k11 / det, -k01 / det, -k10 / det, k00 / det


def legendre_gauss_lobatto(num_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Legendre-Gauss-Lobatto points and weights on [-1, 1]."""
    if num_points < 2:
        raise ValueError("Gauss-Lobatto rule needs at least two points")
    if num_points == 2:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    poly = np.polynomial.legendre.Legendre.basis(num_points - 1)
    interior = np.sort(poly.deriv().roots())
    points = np.concatenate(([-1.0], interior, [1.0]))
    values = poly(points)
    weights = 2.0 / ((num_points - 1) * num_points * values * values)
    return np.ascontiguousarray(points, dtype=np.float64), np.ascontiguousarray(weights, dtype=np.float64)


def lagrange_basis(nodes: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Evaluate one-dimensional Lagrange basis functions."""
    nodes = np.asarray(nodes, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    values = np.ones((nodes.size, points.size), dtype=np.float64)
    for i in range(nodes.size):
        for j in range(nodes.size):
            if i != j:
                values[i] *= (points - nodes[j]) / (nodes[i] - nodes[j])
    return np.ascontiguousarray(values)


def bernstein_edge_basis(order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate Bernstein edge basis functions on [-1, 1]."""
    from math import factorial

    r = 0.5 * (points + 1.0)
    values = np.empty((order + 1, points.size), dtype=np.float64)
    for j in range(order + 1):
        coeff = factorial(order) / (factorial(j) * factorial(order - j))
        values[j] = coeff * (1.0 - r) ** (order - j) * r ** j
    return np.ascontiguousarray(values)


def legendre_edge_basis(order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate modal Legendre edge basis functions on [-1, 1]."""
    values = np.empty((order + 1, points.size), dtype=np.float64)
    for j in range(order + 1):
        values[j] = np.polynomial.legendre.Legendre.basis(j)(points)
    return np.ascontiguousarray(values)


def edge_points(edge_points_1d: np.ndarray) -> np.ndarray:
    """Return reference-triangle face coordinates for 1D edge coordinates."""
    t = edge_points_1d
    return np.ascontiguousarray(
        np.stack(
            (
                np.stack((t, -np.ones_like(t)), axis=1),
                np.stack((-t, t), axis=1),
                np.stack((-np.ones_like(t), -t), axis=1),
            ),
            axis=1,
        )
    )


def build_trace_reference(cspace, kind: str) -> TraceReferenceData:
    """Build device trace-reference tables for the requested trace basis."""
    cupy = require_cupy()
    space = cspace.host
    order = int(cspace.order)
    normalized = str(kind).replace("-", "_").lower()
    if normalized == "bernstein":
        q = cspace.quad_data
        host_q = space.quad_data
        edge_quads = host_q.quads_JGL
        edge_weights = host_q.weights_JGL
        face_basis = host_q.bas_of_bd_quads
        negative_face_points = edge_points(-edge_quads)
        negative_face_basis = space.basis_at(negative_face_points.reshape(-1, 2)).reshape(
            edge_quads.size, 3, space.el_dof
        ).transpose(1, 2, 0)
        edge_basis = bernstein_edge_basis(order, edge_quads)
        weighted_face_basis = np.ascontiguousarray(face_basis * edge_weights[None, None, :])
        weighted_edge_basis = np.ascontiguousarray(edge_basis * edge_weights[None, :])
        face_coupling = np.einsum("q,fiq,jq->fij", edge_weights, face_basis, edge_basis, optimize=True)
        face_coupling_reversed = np.einsum(
            "q,fiq,jq->fij",
            edge_weights,
            negative_face_basis,
            edge_basis,
            optimize=True,
        )
        trace_lift = np.ascontiguousarray(
            np.concatenate((face_coupling.transpose(0, 2, 1), face_coupling_reversed.transpose(0, 2, 1)), axis=0)
        )
        edge_mass = np.einsum("q,iq,jq->ij", edge_weights, edge_basis, edge_basis, optimize=True)
        return TraceReferenceData(
            kind="bernstein",
            nodal=False,
            interpolation_nodes=cupy.asarray(edge_quads, dtype=cupy.float64),
            quads=cupy.asarray(edge_quads, dtype=cupy.float64),
            weights=cupy.asarray(edge_weights, dtype=cupy.float64),
            bas_of_bd_quads=cupy.asarray(face_basis, dtype=cupy.float64),
            bas1d_of_ref_edg_qds=cupy.asarray(edge_basis, dtype=cupy.float64),
            weighted_bas_of_bd_quads=cupy.asarray(weighted_face_basis, dtype=cupy.float64),
            weighted_bas1d_of_ref_edg_qds=cupy.asarray(weighted_edge_basis, dtype=cupy.float64),
            face_element_test_trace_trial=cupy.asarray(face_coupling, dtype=cupy.float64),
            face_trace_test_element_trial_oriented=cupy.asarray(trace_lift, dtype=cupy.float64),
            M_rf_fc=cupy.asarray(np.ascontiguousarray(edge_mass), dtype=cupy.float64),
        )
    if normalized not in {"legacy_lagrange", "legendre_modal"}:
        raise ValueError("trace basis must be 'legacy-lagrange', 'legendre-modal', or 'bernstein'")

    interpolation_nodes, _ = legendre_gauss_lobatto(order + 1)
    edge_quads, edge_weights = legendre_gauss_lobatto(2 * order + 1)
    face_points = edge_points(edge_quads)
    face_basis = space.basis_at(face_points.reshape(-1, 2)).reshape(edge_quads.size, 3, space.el_dof).transpose(1, 2, 0)
    if normalized == "legacy_lagrange":
        edge_basis = lagrange_basis(interpolation_nodes, edge_quads)
        edge_basis_reversed = lagrange_basis(interpolation_nodes, -edge_quads)
        nodal = True
        kind_out = "legacy-lagrange"
    else:
        edge_basis = legendre_edge_basis(order, edge_quads)
        edge_basis_reversed = legendre_edge_basis(order, -edge_quads)
        nodal = False
        kind_out = "legendre-modal"
    weighted_face_basis = np.ascontiguousarray(face_basis * edge_weights[None, None, :])
    weighted_edge_basis = np.ascontiguousarray(edge_basis * edge_weights[None, :])
    face_coupling = np.einsum("q,fiq,jq->fij", edge_weights, face_basis, edge_basis, optimize=True)
    face_coupling_reversed = np.einsum("q,fiq,jq->fij", edge_weights, face_basis, edge_basis_reversed, optimize=True)
    trace_lift = np.ascontiguousarray(
        np.concatenate((face_coupling.transpose(0, 2, 1), face_coupling_reversed.transpose(0, 2, 1)), axis=0)
    )
    edge_mass = np.einsum("q,iq,jq->ij", edge_weights, edge_basis, edge_basis, optimize=True)
    return TraceReferenceData(
        kind=kind_out,
        nodal=nodal,
        interpolation_nodes=cupy.asarray(interpolation_nodes, dtype=cupy.float64),
        quads=cupy.asarray(edge_quads, dtype=cupy.float64),
        weights=cupy.asarray(edge_weights, dtype=cupy.float64),
        bas_of_bd_quads=cupy.asarray(np.ascontiguousarray(face_basis), dtype=cupy.float64),
        bas1d_of_ref_edg_qds=cupy.asarray(edge_basis, dtype=cupy.float64),
        weighted_bas_of_bd_quads=cupy.asarray(weighted_face_basis, dtype=cupy.float64),
        weighted_bas1d_of_ref_edg_qds=cupy.asarray(weighted_edge_basis, dtype=cupy.float64),
        face_element_test_trace_trial=cupy.asarray(np.ascontiguousarray(face_coupling), dtype=cupy.float64),
        face_trace_test_element_trial_oriented=cupy.asarray(trace_lift, dtype=cupy.float64),
        M_rf_fc=cupy.asarray(np.ascontiguousarray(edge_mass), dtype=cupy.float64),
    )


def mapped_quads_cupy(cspace):
    """Return mapped volume quadrature points on device."""
    cupy = require_cupy()
    mesh = cspace.mesh
    q = cspace.quad_data
    return cupy.einsum("Krc,qc->Krq", mesh.aff_mats, q.Krf_quads) + mesh.aff_vecs[:, :, None]


def _require_same_space_dg_field(value, cspace, label: str, backend: str) -> DGField:
    """Require a DGField bound to the exact space used by device assembly."""
    host_space = cspace.host
    if isinstance(value, DGField):
        value.space.assert_same_mesh(host_space)
        if value.space is not host_space:
            raise ValueError(f"{label} must live in the same DGSpace object for assembly_backend='{backend}'")
        return value
    if callable(value):
        raise TypeError(
            f"assembly_backend='{backend}' requires {label} to be a DGField; "
            f"project callables first with space.project_callable(...)."
        )
    if np.isscalar(value):
        raise TypeError(
            f"assembly_backend='{backend}' requires {label} to be a DGField; "
            f"use space.zeros(...) or space.constant(...) for constants."
        )
    raise TypeError(
        f"assembly_backend='{backend}' requires {label} to be a DGField; "
        f"wrap coefficient arrays with space.field(...)."
    )


def _raw_cuda_reaction_is_zero(reaction, cspace) -> bool:
    """Return whether raw CUDA can treat the reaction coefficient as exactly zero."""
    if not isinstance(reaction, DGField):
        return False
    constant_value = reaction.constant_value
    if constant_value is not None:
        return constant_value == 0.0
    cached = reaction._device_coefficients_for(cspace.device_id)
    if cached is not None:
        cupy = require_cupy()
        return bool(cupy.all(cached == 0.0).get())
    return reaction.is_zero


def _validate_raw_cuda_source(source, cspace):
    """Validate source data accepted by the raw CUDA diffusion path."""
    if isinstance(source, DGField):
        source.space.assert_same_mesh(cspace.host)
        if source.space is not cspace.host:
            raise ValueError("source must live in the same DGSpace object for assembly_backend='raw-cuda'")
        return source
    if np.isscalar(source) or callable(source):
        return source
    raise TypeError(
        "assembly_backend='raw-cuda' requires source to be a DGField, scalar, or CuPy-compatible callable"
    )


def reaction_mass_cupy(reaction, cspace):
    """Assemble reaction mass matrices on device."""
    cupy = require_cupy()
    mesh = cspace.mesh
    q = cspace.quad_data
    if np.isscalar(reaction):
        scalar = float(reaction)
        if scalar == 0.0:
            return 0.0
        return scalar * mesh.aff_jacs[:, None, None] * q.MKrf[None, ...]
    if isinstance(reaction, DGField):
        reaction.space.assert_same_mesh(cspace.host)
        constant_value = reaction.constant_value
        if constant_value is not None:
            if constant_value == 0.0:
                return 0.0
            return constant_value * mesh.aff_jacs[:, None, None] * q.MKrf[None, ...]
        coeffs = as_cupy_coefficients(reaction, cspace)
        values = coeffs @ q.bas_of_quads
    else:
        points = mapped_quads_cupy(cspace)
        values = cupy.asarray(reaction(points[:, 0, :], points[:, 1, :]), dtype=cupy.float64)
    scaled = values * mesh.aff_jacs[:, None]
    flat = scaled @ q.weighted_phi_phi_flat
    return cupy.ascontiguousarray(flat.reshape(mesh.num_tri, cspace.el_dof, cspace.el_dof))


def reference_derivative_mats(cspace):
    """Return device reference derivative matrices in diffusion layout."""
    cupy = require_cupy()
    q = cspace.quad_data
    d0 = cupy.einsum("q,iq,jq->ij", q.Krf_w, q.bas_of_quads, q.dbas_of_quads[0], optimize=True)
    d1 = cupy.einsum("q,iq,jq->ij", q.Krf_w, q.bas_of_quads, q.dbas_of_quads[1], optimize=True)
    return cupy.ascontiguousarray(d0.T), cupy.ascontiguousarray(d1.T)


def face_element_mass(trace_ref):
    """Return device face element mass tables."""
    cupy = require_cupy()
    return cupy.einsum(
        "fiq,fjq->fij",
        trace_ref.bas_of_bd_quads,
        trace_ref.weighted_bas_of_bd_quads,
        optimize=True,
    )


def local_lhs_mats_cupy(reaction, cspace, trace_ref, tau: float):
    """Assemble device mixed local diffusion-reaction matrices."""
    cupy = require_cupy()
    mesh = cspace.mesh
    q = cspace.quad_data
    el_dof = cspace.el_dof
    m_rea = reaction_mass_cupy(reaction, cspace)
    d0t, d1t = reference_derivative_mats(cspace)
    face_mass = face_element_mass(trace_ref)
    local_lhs = cupy.zeros((mesh.num_tri, 3 * el_dof, 3 * el_dof), dtype=cupy.float64)
    blocks = local_lhs.reshape((mesh.num_tri, 3, el_dof, 3, el_dof))
    blocks[:, 0, :, 0, :] = m_rea + cupy.sum(float(tau) * mesh.jacs_el_fc[..., None, None] * face_mass[None, ...], axis=1)
    blocks[:, 1, :, 1, :] = -mesh.aff_jacs[:, None, None] * q.MKrf[None, ...]
    blocks[:, 2, :, 2, :] = -mesh.aff_jacs[:, None, None] * q.MKrf[None, ...]
    blocks[:, 0, :, 1, :] = (
        cupy.sum((mesh.jacs_el_fc * mesh.normals[..., 0])[..., None, None] * face_mass[None, ...], axis=1)
        - mesh.aff_mats[:, 1, 1][:, None, None] * d0t[None, ...]
        + mesh.aff_mats[:, 1, 0][:, None, None] * d1t[None, ...]
    )
    blocks[:, 0, :, 2, :] = (
        cupy.sum((mesh.jacs_el_fc * mesh.normals[..., 1])[..., None, None] * face_mass[None, ...], axis=1)
        + mesh.aff_mats[:, 0, 1][:, None, None] * d0t[None, ...]
        - mesh.aff_mats[:, 0, 0][:, None, None] * d1t[None, ...]
    )
    blocks[:, 1, :, 0, :] = mesh.aff_mats[:, 1, 1][:, None, None] * d0t[None, ...] - mesh.aff_mats[:, 1, 0][:, None, None] * d1t[None, ...]
    blocks[:, 2, :, 0, :] = -mesh.aff_mats[:, 0, 1][:, None, None] * d0t[None, ...] + mesh.aff_mats[:, 0, 0][:, None, None] * d1t[None, ...]
    return cupy.ascontiguousarray(local_lhs)


def element_boundary_mats_cupy(cspace, trace_ref, tau: float):
    """Assemble device element-to-trace coupling matrices."""
    cupy = require_cupy()
    mesh = cspace.mesh
    el_dof = cspace.el_dof
    edg_dof = cspace.edg_dof
    result = cupy.zeros((mesh.num_tri, 3 * el_dof, 3 * edg_dof), dtype=cupy.float64)
    blocks = result.reshape((mesh.num_tri, 3, el_dof, 3, edg_dof))
    oriented = trace_ref.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling]
    coupling = oriented.transpose(0, 3, 1, 2)
    scaled = mesh.jacs_el_fc[:, None, :, None] * coupling
    blocks[:, 0] = float(tau) * scaled
    blocks[:, 1] = mesh.normals[..., 0][:, None, :, None] * scaled
    blocks[:, 2] = mesh.normals[..., 1][:, None, :, None] * scaled
    return cupy.ascontiguousarray(result)


def source_moments_cupy(source: Callable, cspace):
    """Assemble block source moments on device."""
    cupy = require_cupy()
    mesh = cspace.mesh
    q = cspace.quad_data
    rhs = cupy.zeros((mesh.num_tri, 3 * cspace.el_dof), dtype=cupy.float64)
    if np.isscalar(source):
        ref_moments = cupy.asarray(cspace.host._constant_reference_moments(float(source)))
        rhs[:, : cspace.el_dof] = mesh.aff_jacs[:, None] * ref_moments[None, :]
        return cupy.ascontiguousarray(rhs)
    if isinstance(source, DGField):
        source.space.assert_same_mesh(cspace.host)
        constant_value = source.constant_value
        if constant_value is not None:
            ref_moments = cupy.asarray(cspace.host._constant_reference_moments(constant_value))
            rhs[:, : cspace.el_dof] = mesh.aff_jacs[:, None] * ref_moments[None, :]
            return cupy.ascontiguousarray(rhs)
        coeffs = as_cupy_coefficients(source, cspace)
        values = coeffs @ q.bas_of_quads
    else:
        points = mapped_quads_cupy(cspace)
        values = cupy.asarray(source(points[:, 0, :], points[:, 1, :]), dtype=cupy.float64)
    rhs[:, : cspace.el_dof] = mesh.aff_jacs[:, None] * cupy.einsum(
        "Kq,iq,q->Ki",
        values,
        q.bas_of_quads,
        q.Krf_w,
        optimize=True,
    )
    return cupy.ascontiguousarray(rhs)


def solve_local_mats(local_lhs, rhs):
    """Solve batched dense local systems on device."""
    cupy = require_cupy()
    return cupy.ascontiguousarray(cupy.linalg.solve(local_lhs, rhs))


def b_trace_mats_cupy(cspace, trace_ref, tau: float):
    """Assemble device trace lift matrices."""
    cupy = require_cupy()
    mesh = cspace.mesh
    el_dof = cspace.el_dof
    edg_dof = cspace.edg_dof
    lift = mesh.jacs_el_fc[..., None, None] * trace_ref.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling]
    result = cupy.zeros((mesh.num_tri, 3, edg_dof, 3 * el_dof), dtype=cupy.float64)
    result[..., :el_dof] = float(tau) * lift
    result[..., el_dof : 2 * el_dof] = lift * mesh.normals[..., 0, None, None]
    result[..., 2 * el_dof :] = lift * mesh.normals[..., 1, None, None]
    return cupy.ascontiguousarray(result)


def trace_blocks_cupy(b_el_fc, solved_el_bd, cspace, trace_ref):
    """Form element trace blocks on device."""
    cupy = require_cupy()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    blocks = b_el_fc @ solved_el_bd[:, None, :, :]
    blocks = blocks.reshape(mesh.num_tri, 3, edg_dof, 3, edg_dof)
    return cupy.ascontiguousarray(blocks.swapaxes(2, 3))


def trace_data_cupy(trace_blocks, cspace, trace_ref, tau: float):
    """Return reduced trace COO values on device before boundary elimination."""
    cupy = require_cupy()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    n_flux = valid_elements.size * 3 * edg_dof * edg_dof
    n_mass = mesh.int_edges_inds.size * edg_dof * edg_dof
    data = cupy.empty(n_flux + n_mass, dtype=cupy.float64)
    data[:n_flux] = -trace_blocks[valid_elements, valid_faces].ravel()
    data[n_flux:] = (2.0 * float(tau) * mesh.edge_jacs[mesh.int_edges_inds, None, None] * trace_ref.M_rf_fc[None]).ravel()
    return data


def face_rhs_cupy(b_el_fc, solved_src, cspace):
    """Assemble per-element-side RHS fluxes on device."""
    cupy = require_cupy()
    return cupy.ascontiguousarray((b_el_fc @ solved_src[:, None, :, None]).squeeze(-1))


def setup_reduced_indices(cspace):
    """Build reduced COO row/column arrays on device."""
    cupy = require_cupy()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    n_flux = valid_elements.size * 3 * edg_dof * edg_dof
    n_mass = mesh.int_edges_inds.size * edg_dof * edg_dof
    rows = cupy.empty(n_flux + n_mass, dtype=cupy.int64)
    cols = cupy.empty_like(rows)
    i_grid, j_grid = cupy.meshgrid(cupy.arange(edg_dof, dtype=cupy.int64), cupy.arange(edg_dof, dtype=cupy.int64), indexing="ij")
    row_edges = mesh.loc2glob_edge[valid_elements, valid_faces]
    col_edges = mesh.loc2glob_edge[valid_elements]
    rows[:n_flux] = cupy.broadcast_to(
        row_edges[:, None, None, None] * edg_dof + i_grid[None, None, :, :],
        (valid_elements.size, 3, edg_dof, edg_dof),
    ).ravel()
    cols[:n_flux] = (col_edges[:, :, None, None] * edg_dof + j_grid[None, None, :, :]).ravel()
    local = cupy.arange(edg_dof, dtype=cupy.int64)
    l0 = cupy.broadcast_to(local[:, None], (edg_dof, edg_dof)).ravel()
    l1 = cupy.broadcast_to(local[None, :], (edg_dof, edg_dof)).ravel()
    rows[n_flux:] = (mesh.int_edges_inds[:, None] * edg_dof + l0).ravel()
    cols[n_flux:] = (mesh.int_edges_inds[:, None] * edg_dof + l1).ravel()
    return rows, cols


def boundary_trace_values_cupy(boundary_condition: Callable, cspace, trace_ref):
    """Evaluate/project compact boundary trace values on device."""
    cupy = require_cupy()
    mesh = cspace.mesh
    if mesh.bnd_edges_inds.size == 0:
        return cupy.empty((0, cspace.edg_dof), dtype=cupy.float64)
    edge_coords = mesh.node_coords[mesh.edges[mesh.bnd_edges_inds]]
    t = trace_ref.interpolation_nodes if trace_ref.nodal else trace_ref.quads
    points = 0.5 * (
        (1.0 - t)[None, :, None] * edge_coords[:, 0:1, :]
        + (1.0 + t)[None, :, None] * edge_coords[:, 1:2, :]
    )
    values = cupy.asarray(boundary_condition(points[..., 0], points[..., 1]), dtype=cupy.float64)
    num_points = int(t.size)
    expected_shape = (int(mesh.bnd_edges_inds.size), num_points)
    if values.ndim == 0:
        values = cupy.full(expected_shape, float(values), dtype=cupy.float64)
    elif values.shape == (num_points,):
        values = cupy.broadcast_to(values[None, :], expected_shape)
    if values.shape != expected_shape:
        raise ValueError(
            "boundary_condition must return a scalar, edge-point vector, or "
            f"{expected_shape} array; got {values.shape}"
        )
    if trace_ref.nodal:
        return cupy.ascontiguousarray(values)
    rhs = (values * trace_ref.weights[None, :]) @ trace_ref.bas1d_of_ref_edg_qds.T
    return cupy.linalg.solve(trace_ref.M_rf_fc, rhs.T).T


def build_dof_maps(cspace):
    """Build device maps for boundary elimination."""
    cupy = require_cupy()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    edge_to_boundary_slot = cupy.full(mesh.num_edg, -1, dtype=cupy.int64)
    edge_to_boundary_slot[mesh.bnd_edges_inds] = cupy.arange(mesh.bnd_edges_inds.size, dtype=cupy.int64)
    full_to_reduced = cupy.full(mesh.num_edg * edg_dof, -1, dtype=cupy.int64)
    local = cupy.arange(edg_dof, dtype=cupy.int64)
    full_to_reduced[(mesh.int_edges_inds[:, None] * edg_dof + local[None, :]).ravel()] = (
        cupy.arange(mesh.int_edges_inds.size, dtype=cupy.int64)[:, None] * edg_dof + local[None, :]
    ).ravel()
    return edge_to_boundary_slot, full_to_reduced


def eliminate_boundary_cupy(rows, cols, data, rhs, boundary_trace, maps, cspace):
    """Eliminate boundary trace columns from a device COO system."""
    cupy = require_cupy()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    edge_to_boundary_slot, full_to_reduced = maps
    row_r = rows.reshape((rows.size // edg_dof, edg_dof))
    col_r = cols.reshape((cols.size // edg_dof, edg_dof))
    data_r = data.reshape((data.size // edg_dof, edg_dof))
    col_edges = col_r[:, 0] // edg_dof
    boundary_slots = edge_to_boundary_slot[col_edges]
    keep = cupy.where(boundary_slots < 0)[0]
    remove = cupy.where(boundary_slots >= 0)[0]
    keep_count = int(keep.size)
    reduced_rows = cupy.empty(keep_count * edg_dof, dtype=cupy.int64)
    reduced_cols = cupy.empty_like(reduced_rows)
    reduced_data = cupy.empty(keep_count * edg_dof, dtype=cupy.float64)
    reduced_rows.reshape((keep_count, edg_dof))[:] = full_to_reduced[row_r[keep]]
    reduced_cols.reshape((keep_count, edg_dof))[:] = full_to_reduced[col_r[keep]]
    reduced_data.reshape((keep_count, edg_dof))[:] = data_r[keep]
    if remove.size:
        row_ids = row_r[remove, 0]
        slots = boundary_slots[remove]
        cupy.add.at(rhs, row_ids, cupy.sum(-data_r[remove] * boundary_trace[slots], axis=1))
    reduced_rhs = rhs.reshape((mesh.num_edg, edg_dof))[mesh.int_edges_inds].ravel()
    return reduced_rows, reduced_cols, reduced_data, reduced_rhs


def compact_boundary_trace_to_full(boundary_trace, cspace):
    """Expand compact boundary-edge trace values to a full edge table."""
    cupy = require_cupy()
    full = cupy.zeros((cspace.mesh.num_edg, cspace.edg_dof), dtype=cupy.float64)
    if cspace.mesh.bnd_edges_inds.size:
        full[cspace.mesh.bnd_edges_inds] = boundary_trace
    return cupy.ascontiguousarray(full)


def raw_cuda_diffusion_fallback_reason(cspace, trace_ref) -> tuple[str, str] | None:
    """Return a human-readable reason when raw CUDA diffusion is unsupported."""
    trace_kind = str(getattr(trace_ref, "kind", "unknown"))
    supported_trace = (
        trace_kind == "legacy-lagrange" and getattr(trace_ref, "nodal", False)
    ) or (
        trace_kind == "legendre-modal" and not getattr(trace_ref, "nodal", True)
    )
    if not supported_trace:
        label = f"raw-cuda trace={trace_kind} unsupported"
        detail = (
            "raw CUDA diffusion assembly supports legacy-lagrange nodal and legendre-modal trace bases; "
            f"got trace={trace_kind}"
        )
        return label, detail
    if int(cspace.el_dof) <= RAW_CUDA_MAX_EL_DOF:
        return None
    order = int(cspace.order)
    label = f"raw-cuda p={order} > 6"
    detail = (
        f"raw CUDA diffusion assembly supports p <= 6 "
        f"(el_dof <= {RAW_CUDA_MAX_EL_DOF}); got p={order}, el_dof={int(cspace.el_dof)}"
    )
    return label, detail


def assemble_projected_diffusion_trace_system_eliminated_cupy(
        source,
        reaction,
        boundary_condition: Callable,
        stabilization: float,
        space,
        *,
        trace_basis: str = "legacy-lagrange",
        trace_ref=None,
) -> CupyDiffusionTraceAssembly:
    """Assemble a reduced diffusion trace system using CuPy operations."""
    cupy = require_cupy()
    timings: dict[str, float] = {}
    cspace = as_cupy_space(space)
    if trace_ref is None:
        trace_ref = build_trace_reference(cspace, trace_basis)
    maps = build_dof_maps(cspace)

    start = time.perf_counter()
    rows, cols = setup_reduced_indices(cspace)
    local_lhs = local_lhs_mats_cupy(reaction, cspace, trace_ref, float(stabilization))
    b_el_fc = b_trace_mats_cupy(cspace, trace_ref, float(stabilization))
    element_boundary = element_boundary_mats_cupy(cspace, trace_ref, float(stabilization))
    source_rhs = source_moments_cupy(source, cspace)
    local_rhs = cupy.concatenate((element_boundary, source_rhs[..., None]), axis=2)
    solved = solve_local_mats(local_lhs, local_rhs)
    solved_el_bd = solved[:, :, : 3 * cspace.edg_dof]
    solved_src = solved[:, :, 3 * cspace.edg_dof]
    blocks = trace_blocks_cupy(b_el_fc, solved_el_bd, cspace, trace_ref)
    data = trace_data_cupy(blocks, cspace, trace_ref, float(stabilization))
    faces = face_rhs_cupy(b_el_fc, solved_src, cspace)
    rhs_full = cupy.zeros(cspace.mesh.num_edg * cspace.edg_dof, dtype=cupy.float64)
    rhs_full_r = rhs_full.reshape((cspace.mesh.num_edg, cspace.edg_dof))
    cupy.add.at(
        rhs_full_r,
        cspace.mesh.loc2glob_edge[cspace.mesh.interior_elements, cspace.mesh.interior_faces],
        faces[cspace.mesh.interior_elements, cspace.mesh.interior_faces],
    )
    boundary_trace = boundary_trace_values_cupy(boundary_condition, cspace, trace_ref)
    rows, cols, data, rhs = eliminate_boundary_cupy(rows, cols, data, rhs_full, boundary_trace, maps, cspace)
    cupy.cuda.get_current_stream().synchronize()
    timings["total"] = time.perf_counter() - start
    return CupyDiffusionTraceAssembly(
        rows=rows,
        cols=cols,
        data=data,
        rhs=rhs,
        boundary_trace=boundary_trace,
        timings=timings,
        matrix_format="coo",
        local_lhs=local_lhs,
        element_boundary_mats=element_boundary,
        source_rhs=source_rhs,
    )


def assemble_projected_diffusion_trace_rhs_eliminated_raw_cupy(
        source,
        reaction,
        boundary_condition: Callable,
        stabilization: float,
        space,
        *,
        cached_raw: RawDiffusionAssemblyResult,
        trace_basis: str = "legacy-lagrange",
        block_size: RawCudaBlockSize = "auto",
        trace_ref=None,
) -> CupyDiffusionTraceAssembly:
    """Assemble only the reduced RHS for a cached raw-CUDA CSR diffusion operator."""
    cupy = require_cupy()
    timings: dict[str, float] = {}
    cspace = as_cupy_space(space)
    source = _validate_raw_cuda_source(source, cspace)
    reaction = _require_same_space_dg_field(reaction, cspace, "reaction", "raw-cuda")
    if not _raw_cuda_reaction_is_zero(reaction, cspace):
        raise NotImplementedError("raw CUDA diffusion assembly currently supports only zero reaction")
    if trace_ref is None:
        trace_ref = build_trace_reference(cspace, trace_basis)
    fallback = raw_cuda_diffusion_fallback_reason(cspace, trace_ref)
    if fallback is not None:
        _, detail = fallback
        raise NotImplementedError(detail)
    validate_raw_cuda_supported(cspace, trace_ref)
    if str(cached_raw.matrix_format).lower() != "csr" or cached_raw.csr_pattern is None:
        raise ValueError("raw-CUDA cached RHS assembly requires a cached CSR raw assembly")

    start = time.perf_counter()
    source_rhs = source_moments_cupy(source, cspace)
    boundary_trace = boundary_trace_values_cupy(boundary_condition, cspace, trace_ref)
    raw = assemble_projected_diffusion_trace_rhs_eliminated_raw_cuda(
        source_rhs=source_rhs,
        boundary_trace=boundary_trace,
        cspace=cspace,
        trace_ref=trace_ref,
        d0_reference=cached_raw.d0_reference,
        d1_reference=cached_raw.d1_reference,
        face_element_mass=cached_raw.face_element_mass,
        tau=float(stabilization),
        csr_pattern=cached_raw.csr_pattern,
        block_size=block_size,
    )
    cupy.cuda.get_current_stream().synchronize()
    timings.update(raw.timings)
    timings["total"] = time.perf_counter() - start
    return CupyDiffusionTraceAssembly(
        rows=None,
        cols=None,
        data=cached_raw.data,
        rhs=raw.rhs,
        boundary_trace=raw.boundary_trace,
        timings=timings,
        matrix_format="csr",
        indptr=cached_raw.indptr,
        indices=cached_raw.indices,
        source_rhs=source_rhs,
        raw_assembly=RawDiffusionAssemblyResult(
            rows=None,
            cols=None,
            data=cached_raw.data,
            rhs=raw.rhs,
            source_rhs=source_rhs,
            boundary_trace=raw.boundary_trace,
            d0_reference=cached_raw.d0_reference,
            d1_reference=cached_raw.d1_reference,
            face_element_mass=cached_raw.face_element_mass,
            timings=timings,
            indptr=cached_raw.indptr,
            indices=cached_raw.indices,
            matrix_format="csr",
            csr_pattern=cached_raw.csr_pattern,
        ),
    )


def assemble_projected_diffusion_trace_system_eliminated_raw_cupy(
        source,
        reaction,
        boundary_condition: Callable,
        stabilization: float,
        space,
        *,
        trace_basis: str = "legacy-lagrange",
        matrix_format: str = "coo",
        block_size: RawCudaBlockSize = "auto",
        trace_ref=None,
) -> CupyDiffusionTraceAssembly:
    """Assemble a reduced diffusion trace system with the raw CUDA backend."""
    cupy = require_cupy()
    timings: dict[str, float] = {}
    cspace = as_cupy_space(space)
    source = _validate_raw_cuda_source(source, cspace)
    reaction = _require_same_space_dg_field(reaction, cspace, "reaction", "raw-cuda")
    if not _raw_cuda_reaction_is_zero(reaction, cspace):
        raise NotImplementedError("raw CUDA diffusion assembly currently supports only zero reaction")
    if trace_ref is None:
        trace_ref = build_trace_reference(cspace, trace_basis)
    fallback = raw_cuda_diffusion_fallback_reason(cspace, trace_ref)
    if fallback is not None:
        _, detail = fallback
        raise NotImplementedError(detail)
    validate_raw_cuda_supported(cspace, trace_ref)

    start = time.perf_counter()
    source_rhs = source_moments_cupy(source, cspace)
    boundary_trace = boundary_trace_values_cupy(boundary_condition, cspace, trace_ref)
    d0_reference, d1_reference = reference_derivative_mats(cspace)
    face_mass = face_element_mass(trace_ref)
    raw = assemble_projected_diffusion_trace_system_eliminated_raw_cuda(
        source_rhs=source_rhs,
        boundary_trace=boundary_trace,
        cspace=cspace,
        trace_ref=trace_ref,
        d0_reference=d0_reference,
        d1_reference=d1_reference,
        face_element_mass=face_mass,
        tau=float(stabilization),
        matrix_format=matrix_format,
        block_size=block_size,
    )
    cupy.cuda.get_current_stream().synchronize()
    timings.update(raw.timings)
    timings["total"] = time.perf_counter() - start
    return CupyDiffusionTraceAssembly(
        rows=raw.rows,
        cols=raw.cols,
        data=raw.data,
        rhs=raw.rhs,
        boundary_trace=raw.boundary_trace,
        timings=timings,
        matrix_format=raw.matrix_format,
        indptr=raw.indptr,
        indices=raw.indices,
        source_rhs=source_rhs,
        raw_assembly=raw,
    )



def _postprocess_reference_cache(space, trace_space, cache):
    """Return host reference tables for primal postprocessing without host solves."""
    from ..solvers.diffusion_reaction import _new_hdg_postprocess_cache

    trace_ref = space.trace_space("legacy-lagrange") if trace_space is None else trace_space
    if cache is None or cache.base_space is not space or cache.trace_space is not trace_ref:
        cache = _new_hdg_postprocess_cache(space, trace_ref)
    return cache


def postprocess_projected_diffusion_primal_cupy(
        local_unknowns,
        space,
        diffusion=1.0,
        *,
        trace_space=None,
        cache=None,
        name: str = "u_h_star",
        timings: dict[str, float] | None = None,
):
    """Recover the HDG primal postprocessed field on the CUDA device with CuPy.

    The host is used only to build reference tables and sample the optional
    inverse-diffusion coefficient table.  The per-element matrix assembly, RHS
    construction, and local solves run on device, and the returned ``DGField``
    stores its coefficient table on the active CUDA device until host access is
    requested.
    """
    cupy = require_cupy()
    timings = {} if timings is None else timings
    cspace = as_cupy_space(space)
    stream = cupy.cuda.get_current_stream()
    stream.synchronize()
    total_start = time.perf_counter()

    setup_start = time.perf_counter()
    cache = _postprocess_reference_cache(space, trace_space, cache)
    cpost_space = as_cupy_space(cache.post_space, device=cspace.device_id)
    local_unknowns = cupy.ascontiguousarray(cupy.asarray(local_unknowns, dtype=cupy.float64))
    expected_unknowns = (cspace.mesh.num_tri, 3 * cspace.el_dof)
    if tuple(local_unknowns.shape) != expected_unknowns:
        raise ValueError(f"local_unknowns must have shape {expected_unknowns}; got {local_unknowns.shape}")

    base_el_dof = int(cspace.el_dof)
    post_el_dof = int(cache.post_space.el_dof)
    rows = post_el_dof + 1
    num_elements = int(cspace.mesh.num_tri)
    q_post = cpost_space.quad_data

    stiffness_rr = cupy.asarray(cache.primal_stiffness_rr, dtype=cupy.float64)
    stiffness_rs = cupy.asarray(cache.primal_stiffness_rs, dtype=cupy.float64)
    stiffness_ss = cupy.asarray(cache.primal_stiffness_ss, dtype=cupy.float64)
    mean_post = cupy.asarray(cache.mean_post, dtype=cupy.float64)
    mean_base = cupy.asarray(cache.mean_base, dtype=cupy.float64)
    base_basis_t = cupy.asarray(cache.base_basis_on_post_quads.T, dtype=cupy.float64)
    weights = q_post.Krf_w
    post_grad = q_post.gphi

    inverse_constants = _constant_inverse_diffusion_components(diffusion)
    if inverse_constants is None:
        from ..solvers.diffusion_reaction import _inverse_diffusion_values

        inv00_h, inv01_h, inv10_h, inv11_h = _inverse_diffusion_values(diffusion, cache.post_space)
        inv00 = cupy.asarray(inv00_h, dtype=cupy.float64)
        inv01 = cupy.asarray(inv01_h, dtype=cupy.float64)
        inv10 = cupy.asarray(inv10_h, dtype=cupy.float64)
        inv11 = cupy.asarray(inv11_h, dtype=cupy.float64)
    else:
        inv00, inv01, inv10, inv11 = map(float, inverse_constants)
    stream.synchronize()
    timings["postprocess.primal.cupy.setup"] = time.perf_counter() - setup_start

    solve_start = time.perf_counter()
    inv_t = cspace.mesh.inv_aff_mats_t
    aff_jacs = cspace.mesh.aff_jacs
    inv00_geom = inv_t[:, 0, 0]
    inv01_geom = inv_t[:, 0, 1]
    inv10_geom = inv_t[:, 1, 0]
    inv11_geom = inv_t[:, 1, 1]
    metric_rr = inv00_geom * inv00_geom + inv10_geom * inv10_geom
    metric_rs = inv00_geom * inv01_geom + inv10_geom * inv11_geom
    metric_ss = inv01_geom * inv01_geom + inv11_geom * inv11_geom

    matrix = cupy.zeros((num_elements, rows, rows), dtype=cupy.float64)
    matrix[:, :post_el_dof, :post_el_dof] = aff_jacs[:, None, None] * (
        metric_rr[:, None, None] * stiffness_rr[None, :, :]
        + metric_rs[:, None, None] * stiffness_rs[None, :, :]
        + metric_ss[:, None, None] * stiffness_ss[None, :, :]
    )
    mean_rows = aff_jacs[:, None] * mean_post[None, :]
    matrix[:, :post_el_dof, post_el_dof] = mean_rows
    matrix[:, post_el_dof, :post_el_dof] = mean_rows

    qx_values = local_unknowns[:, base_el_dof:2 * base_el_dof] @ base_basis_t
    qy_values = local_unknowns[:, 2 * base_el_dof:3 * base_el_dof] @ base_basis_t
    cqx_values = inv00 * qx_values + inv01 * qy_values
    cqy_values = inv10 * qx_values + inv11 * qy_values

    rhs = cupy.zeros((num_elements, rows), dtype=cupy.float64)
    for quad in range(int(weights.shape[0])):
        grad_r = post_grad[quad, :, 0]
        grad_s = post_grad[quad, :, 1]
        grad_x = inv00_geom[:, None] * grad_r[None, :] + inv01_geom[:, None] * grad_s[None, :]
        grad_y = inv10_geom[:, None] * grad_r[None, :] + inv11_geom[:, None] * grad_s[None, :]
        rhs[:, :post_el_dof] += weights[quad] * (
            cqx_values[:, quad, None] * grad_x
            + cqy_values[:, quad, None] * grad_y
        )
    rhs[:, :post_el_dof] *= -aff_jacs[:, None]
    rhs[:, post_el_dof] = aff_jacs * (local_unknowns[:, :base_el_dof] @ mean_base)

    solution = cupy.linalg.solve(matrix, rhs[:, :, None]).squeeze(-1)
    coeffs = cupy.ascontiguousarray(solution[:, :post_el_dof])
    stream.synchronize()
    timings["postprocess.primal.cupy.solve"] = time.perf_counter() - solve_start
    timings["postprocess.primal.cupy.total"] = time.perf_counter() - total_start
    return cpost_space.field(coeffs, name=name), cache


__all__ = [
    "CupyDiffusionTraceAssembly",
    "TraceReferenceData",
    "assemble_projected_diffusion_trace_system_eliminated_cupy",
    "assemble_projected_diffusion_trace_rhs_eliminated_raw_cupy",
    "assemble_projected_diffusion_trace_system_eliminated_raw_cupy",
    "boundary_trace_values_cupy",
    "build_dof_maps",
    "build_trace_reference",
    "compact_boundary_trace_to_full",
    "element_boundary_mats_cupy",
    "face_element_mass",
    "reference_derivative_mats",
    "postprocess_projected_diffusion_primal_cupy",
    "raw_cuda_diffusion_fallback_reason",
    "source_moments_cupy",
]
