"""Reusable HDG static-condensation and trace-assembly helpers.

This module contains the mesh/trace glue that is independent of the concrete
PDE local operator.  Local element matrices still come from problem-specific
code, usually via :mod:`dgfem.hdg_mats`; once those matrices and element RHS
moments are available, the functions here assemble the global trace system and
recover element coefficients.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import hdg_mats
from .global_system import SolveResult, solve_global_system
from .space import DGField, DGSpace, VectorDGField


@dataclass(frozen=True)
class TraceSystem:
    """COO representation of an assembled HDG trace system."""

    rows: np.ndarray
    cols: np.ndarray
    data: np.ndarray
    rhs: np.ndarray
    boundary_trace: np.ndarray | None = None


def as_vector_field(beta, space: DGSpace) -> VectorDGField:
    """Normalize two-component advection data to a :class:`VectorDGField`.

    Accepted inputs are already-built vector fields, coefficient arrays accepted
    by ``space * space``, or a tuple of two callables that are projected into
    ``space``.
    """
    if isinstance(beta, VectorDGField):
        if beta.dim != 2:
            raise ValueError("advection field must have two components")
        beta.components[0].space.assert_same_mesh(space)
        beta.components[1].space.assert_same_mesh(space)
        return beta

    vector_space = space * space
    if isinstance(beta, np.ndarray):
        return vector_space.field(beta, name="beta_h")
    if isinstance(beta, tuple) and len(beta) == 2 and all(callable(component) for component in beta):
        return vector_space.field(
            (
                space.project_callable(beta[0], name="beta_x").coeffs,
                space.project_callable(beta[1], name="beta_y").coeffs,
            ),
            name="beta_h",
        )
    raise TypeError("beta must be a VectorDGField, coefficient array, or tuple of two callables")


def reaction_mass(reaction, space: DGSpace) -> np.ndarray:
    """Assemble reaction mass matrices, with an exact constant fast path."""
    if np.isscalar(reaction):
        return float(reaction) * space.mesh.aff_jacs[:, None, None] * space.quad_data.MKrf[None, :, :]
    if isinstance(reaction, DGField):
        return hdg_mats.mass_from_field(space, reaction)
    if callable(reaction):
        return space.weighted_mass(reaction)

    values = np.asarray(reaction, dtype=np.float64)
    if values.shape == (space.mesh.num_tri, space.quad_data.Krf_w.shape[0]):
        out = np.empty((space.mesh.num_tri, space.el_dof, space.el_dof), dtype=np.float64)
        return hdg_mats.set_weighted_mass_from_values(out, values, space)
    if values.shape == (space.mesh.num_tri, space.el_dof):
        return hdg_mats.mass_from_field(space, space.field(values, name="reaction"))
    raise TypeError("reaction must be a scalar, callable, quadrature values, or DG coefficient array")


def source_moments(source, space: DGSpace) -> np.ndarray:
    r"""Compute element moments :math:`\int_K f\phi_i\,dx`."""
    if isinstance(source, DGField):
        source.space.assert_same_mesh(space)
        if source.space is space:
            rhs = source.coeffs @ space.quad_data.MKrf
            rhs *= space.mesh.aff_jacs[:, None]
            return np.ascontiguousarray(rhs, dtype=np.float64)
        values = source.values_at_ref(space.quad_data.Krf_quads)
    elif callable(source):
        points = space.mapped_quads()
        values = source(points[:, :, 0], points[:, :, 1])
    else:
        array = np.asarray(source, dtype=np.float64)
        if array.shape == space.shape:
            return np.ascontiguousarray(array)
        if array.shape == (space.mesh.num_tri, space.quad_data.Krf_w.shape[0]):
            values = array
        else:
            raise TypeError("source must be a DGField, callable, source moments, or quadrature values")

    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0:
        values = np.full((space.mesh.num_tri, space.quad_data.Krf_w.shape[0]), float(values))
    elif values.shape == (space.quad_data.Krf_w.shape[0],):
        values = np.broadcast_to(values[None, :], (space.mesh.num_tri, values.size))
    if values.shape != (space.mesh.num_tri, space.quad_data.Krf_w.shape[0]):
        raise ValueError(
            "source values must have shape "
            f"({space.mesh.num_tri}, {space.quad_data.Krf_w.shape[0]}); got {values.shape}"
        )

    rhs = values @ space.quad_data.weighted_phi
    rhs *= space.mesh.aff_jacs[:, None]
    return np.ascontiguousarray(rhs, dtype=np.float64)


def boundary_trace_coefficients(boundary_condition: Callable, space: DGSpace) -> np.ndarray:
    """Project Dirichlet data onto the trace basis on boundary edges."""
    mesh = space.mesh
    q = space.quad_data
    trace_coeffs = np.zeros((mesh.num_edg, q.edg_dof), dtype=np.float64)
    if mesh.bnd_edges_inds.size == 0:
        return trace_coeffs

    edge_vertices = mesh.node_coords[mesh.edges[mesh.bnd_edges_inds]]
    t = q.quads_JGL
    points = 0.5 * (
        (1.0 - t)[None, :, None] * edge_vertices[:, 0:1, :]
        + (1.0 + t)[None, :, None] * edge_vertices[:, 1:2, :]
    )
    values = boundary_condition(points[:, :, 0], points[:, :, 1])
    values = np.asarray(values, dtype=np.float64)
    num_face_quads = q.weights_JGL.size
    if values.ndim == 0:
        values = np.full((mesh.bnd_edges_inds.size, num_face_quads), float(values))
    elif values.shape == (num_face_quads,):
        values = np.broadcast_to(values[None, :], (mesh.bnd_edges_inds.size, num_face_quads))
    if values.shape != (mesh.bnd_edges_inds.size, num_face_quads):
        raise ValueError(
            "boundary_condition must return a scalar, face-quadrature vector, or "
            f"({mesh.bnd_edges_inds.size}, {num_face_quads}) array; got {values.shape}"
        )

    rhs = (values * q.weights_JGL[None, :]) @ q.bas1d_of_ref_edg_qds.T
    trace_coeffs[mesh.bnd_edges_inds] = rhs @ np.linalg.inv(q.M_rf_fc)
    return np.ascontiguousarray(trace_coeffs)


def trace_matrix_indices(space: DGSpace) -> tuple[np.ndarray, np.ndarray]:
    """Return COO row/column indices for the full trace system."""
    mesh = space.mesh
    edg_dof = space.quad_data.edg_dof
    n_interior_flux = mesh.int_edges_inds.size * 2 * edg_dof * 3 * edg_dof
    n_interior_mass = mesh.int_edges_inds.size * edg_dof * edg_dof
    n_boundary = mesh.bnd_edges_inds.size * edg_dof
    rows = np.empty(n_interior_flux + n_interior_mass + n_boundary, dtype=np.int64)
    cols = np.empty_like(rows)

    i_grid, j_grid = np.meshgrid(np.arange(edg_dof), np.arange(edg_dof), indexing="ij")
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    row_edges = mesh.sigma[valid_elements, valid_faces]
    col_edges = mesh.sigma[valid_elements]
    rows[:n_interior_flux] = np.broadcast_to(
        row_edges[:, None, None, None] * edg_dof + i_grid[None, None, :, :],
        (valid_elements.size, 3, edg_dof, edg_dof),
    ).ravel()
    cols[:n_interior_flux] = (col_edges[:, :, None, None] * edg_dof + j_grid[None, None, :, :]).ravel()

    l0 = np.broadcast_to(np.arange(edg_dof)[:, None], (edg_dof, edg_dof)).ravel()
    l1 = np.broadcast_to(np.arange(edg_dof)[None, :], (edg_dof, edg_dof)).ravel()
    offset = n_interior_flux
    rows[offset:offset + n_interior_mass] = (mesh.int_edges_inds[:, None] * edg_dof + l0).ravel()
    cols[offset:offset + n_interior_mass] = (mesh.int_edges_inds[:, None] * edg_dof + l1).ravel()

    offset += n_interior_mass
    boundary_dofs = mesh.bnd_edges_inds[:, None] * edg_dof + np.arange(edg_dof)[None, :]
    rows[offset:] = boundary_dofs.ravel()
    cols[offset:] = boundary_dofs.ravel()
    return rows, cols


def element_to_trace_matrix(local_solver: np.ndarray, element_boundary_mats: np.ndarray, space: DGSpace) -> np.ndarray:
    """Assemble oriented element-to-trace Schur complement blocks."""
    mesh = space.mesh
    q = space.quad_data
    edge_lift = mesh.jacs_el_fc[..., None, None] / 2.0 * q.MKrfe_lst[mesh.sigma_1]
    schur = edge_lift @ (local_solver @ element_boundary_mats)[:, None, :, :]
    edg_dof = q.edg_dof
    schur = schur.reshape(mesh.num_tri, 3, edg_dof, 3, edg_dof)
    negative_elements, negative_faces = np.nonzero(~mesh.orientations)
    schur[negative_elements, :, :, negative_faces, :] = schur[negative_elements, :, :, negative_faces, ::-1]
    return np.ascontiguousarray(schur.swapaxes(2, 3))


def trace_matrix_data(trace_blocks: np.ndarray, space: DGSpace, boundary_penalty: float) -> np.ndarray:
    """Return COO data values matching :func:`trace_matrix_indices`."""
    mesh = space.mesh
    q = space.quad_data
    edg_dof = q.edg_dof
    n_interior_flux = mesh.int_edges_inds.size * 2 * edg_dof * 3 * edg_dof
    n_interior_mass = mesh.int_edges_inds.size * edg_dof * edg_dof
    n_boundary = mesh.bnd_edges_inds.size * edg_dof
    data = np.empty(n_interior_flux + n_interior_mass + n_boundary, dtype=np.float64)

    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    data[:n_interior_flux] = -trace_blocks[valid_elements, valid_faces].ravel()

    edge_jacs = mesh.edge_jacs[mesh.int_edges_inds]
    offset = n_interior_flux
    data[offset:offset + n_interior_mass] = (edge_jacs[:, None, None] * q.M_rf_fc[None, :, :]).ravel()

    offset += n_interior_mass
    data[offset:] = boundary_penalty
    return data


def global_rhs(
        source_rhs: np.ndarray,
        local_solver: np.ndarray,
        boundary_condition: Callable,
        space: DGSpace,
        boundary_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the full trace RHS and known boundary trace coefficients."""
    mesh = space.mesh
    q = space.quad_data
    edge_lift = mesh.jacs_el_fc[..., None, None] / 2.0 * q.MKrfe_lst[mesh.sigma_1]
    face_rhs = (edge_lift @ (local_solver @ source_rhs[..., None])[:, None, :, :]).squeeze(-1)

    rhs = np.zeros((mesh.num_edg, q.edg_dof), dtype=np.float64)
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    if valid_elements.size:
        np.add.at(rhs, mesh.sigma[valid_elements, valid_faces], face_rhs[valid_elements, valid_faces])

    boundary_trace = boundary_trace_coefficients(boundary_condition, space)
    rhs[mesh.bnd_edges_inds] = boundary_penalty * boundary_trace[mesh.bnd_edges_inds]
    return rhs.ravel(), boundary_trace


def assemble_trace_system(
        local_solver: np.ndarray,
        element_boundary_mats: np.ndarray,
        source_rhs: np.ndarray,
        boundary_condition: Callable,
        space: DGSpace,
        *,
        boundary_penalty: float = 1e20,
) -> TraceSystem:
    """Assemble the global HDG trace system from condensed element data."""
    trace_blocks = element_to_trace_matrix(local_solver, element_boundary_mats, space)
    rows, cols = trace_matrix_indices(space)
    data = trace_matrix_data(trace_blocks, space, boundary_penalty)
    rhs, boundary_trace = global_rhs(source_rhs, local_solver, boundary_condition, space, boundary_penalty)
    return TraceSystem(rows=rows, cols=cols, data=data, rhs=rhs, boundary_trace=boundary_trace)


def solve_trace_system(
        rows: np.ndarray,
        cols: np.ndarray,
        data: np.ndarray,
        rhs: np.ndarray,
        *,
        solver: str | None = "BICGSTAB",
        preconditioner="ilu",
        rtol: float = 1e-13,
        atol: float = 0.0,
        maxiter: int | None = None,
        verbose: bool | int = False,
) -> SolveResult:
    """Solve the sparse trace system from COO triplets."""
    solver_name = "direct" if solver is None or str(solver).lower() == "direct" else str(solver)
    return solve_global_system(
        rows,
        cols,
        data,
        rhs,
        rhs.size,
        solver=solver_name,
        preconditioner=preconditioner,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        scale_system=True,
        scale_matrix_in_place=True,
        raise_on_nonconvergence=True,
        verbose=bool(verbose),
    )


def reconstruct_field(
        trace: np.ndarray,
        source_rhs: np.ndarray,
        local_solver: np.ndarray,
        element_boundary_mats: np.ndarray,
        space: DGSpace,
        *,
        name: str = "u_h",
) -> DGField:
    """Recover element coefficients from the solved trace vector."""
    mesh = space.mesh
    edg_dof = space.quad_data.edg_dof
    element_traces = trace.reshape(mesh.num_edg, edg_dof)[mesh.sigma].copy()
    element_traces[~mesh.orientations] = element_traces[~mesh.orientations][:, ::-1]
    element_traces = element_traces.reshape(mesh.num_tri, 3 * edg_dof)
    coeffs = local_solver @ (source_rhs[..., None] + element_boundary_mats @ element_traces[..., None])
    return space.field(coeffs.squeeze(-1), name=name)


__all__ = [
    "TraceSystem",
    "as_vector_field",
    "assemble_trace_system",
    "boundary_trace_coefficients",
    "element_to_trace_matrix",
    "global_rhs",
    "reaction_mass",
    "reconstruct_field",
    "solve_trace_system",
    "source_moments",
    "trace_matrix_data",
    "trace_matrix_indices",
]
