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
from .cupy_diff_rea_raw import (
    RawDiffusionAssemblyResult,
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
    if not getattr(trace_ref, "nodal", False) or trace_kind != "legacy-lagrange":
        label = f"raw-cuda trace={trace_kind} unsupported"
        detail = (
            "raw CUDA diffusion assembly supports only legacy-lagrange nodal trace basis; "
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


def assemble_projected_diffusion_trace_system_eliminated_raw_cupy(
        source,
        reaction,
        boundary_condition: Callable,
        stabilization: float,
        space,
        *,
        trace_basis: str = "legacy-lagrange",
        matrix_format: str = "coo",
        block_size: int = 1,
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


__all__ = [
    "CupyDiffusionTraceAssembly",
    "TraceReferenceData",
    "assemble_projected_diffusion_trace_system_eliminated_cupy",
    "assemble_projected_diffusion_trace_system_eliminated_raw_cupy",
    "boundary_trace_values_cupy",
    "build_dof_maps",
    "build_trace_reference",
    "compact_boundary_trace_to_full",
    "element_boundary_mats_cupy",
    "face_element_mass",
    "reference_derivative_mats",
    "raw_cuda_diffusion_fallback_reason",
    "source_moments_cupy",
]
