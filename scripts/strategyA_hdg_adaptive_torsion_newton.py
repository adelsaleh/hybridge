"""Adaptive HDG Strategy A Newton solve on the native smooth star domain.

This script mirrors ``ff/stratA/strategyA_adaptive_torsion_newton.edp`` at the
algorithmic level while keeping the HDG unknowns and residuals explicit:

* torsion-designed initializer on the smooth star;
* optional pre-adaptation from the designed torsion band;
* epsilon continuation for the semilinear window;
* one scheduled remesh per epsilon stage after accepted Newton steps;
* HDG residual assembled from the element and trace equations.

The adaptive remesh is a Python/Gmsh equivalent of FreeFEM ``adaptmesh``: it
uses the same scalar indicator and schedule, then transfers DG fields with the
projector in ``hdgfem.core.transfer``.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hdgfem.assembly import hdg as hdg_assembly
from hdgfem.assembly.hdg_gram import CondensedHDGGramInverse
from hdgfem.core.mesh import DGMesh, gmsh_smooth_star_mesh
from hdgfem.core.space import DGField, DGSpace
from hdgfem.core.transfer import build_transfer_plan, project_field
from hdgfem.io.plot import plot_fields
from hdgfem.solvers.diff_rea import (
    DiffusionReactionHDGOptions,
    DiffusionReactionHDGSolver,
    _local_solver_pre_mats,
    _normalize_tau,
    diffusion_element_boundary_mats,
)


@dataclass
class StrategyParameters:
    """FreeFEM default parameters for the native star Strategy A run."""

    alpha_t1: float = 0.60
    alpha_t2: float = 0.70
    eps_t_ratio: float = 0.06
    rho_amp: float = 1.0
    beta_phi1: float = 0.60
    beta_phi2: float = 0.70
    eps_phi_ratios: tuple[float, ...] = (0.11, 0.08, 0.06)
    max_it: int = 70
    tol_res: float = 1.0e-10
    tol_newton: float = 1.0e-10
    beta_ls: float = 0.5
    armijo_c: float = 1.0e-6
    alpha_min: float = 1.0e-7
    max_backtrack: int = 30
    mu_shift: float = 2.0
    mu_min: float = 0.20
    mu_max: float = 30.0
    mass_floor_fraction: float = 0.01
    rho_max_floor: float = 0.02
    max_stagnation: int = 5
    stagnation_tol: float = 1.0e-5
    active_threshold: float = 0.05
    plateau_threshold: float = 0.90
    pre_adapt_design: bool = True
    use_adapt: bool = True
    adapt_after_last_eps: bool = False
    adapt_after_accepted: int = 5
    adapt_grad_weight: float = 10.0
    adapt_err: float = 0.02
    adapt_errg: float = 0.01
    adapt_ratio: float = 1.8
    adapt_anisomax: float = 10.0
    adapt_nbvx: int = 500000
    adapt_nbsmooth: int = 3


@dataclass
class State:
    """Current HDG nonlinear state."""

    u: DGField
    flux: np.ndarray
    trace: np.ndarray
    rho: DGField
    residual: np.ndarray
    residual_coeff_l2: float
    residual_hdg: float
    residual_hdg_squared: float


@dataclass
class DesignState:
    """Torsion-designed fields that must survive remeshing."""

    torsion: DGField
    rho_design: DGField
    phi_design: DGField
    c1_t: float
    c2_t: float
    eps_t: float


def zero_boundary(x, y):
    return np.zeros_like(x, dtype=np.float64)


def one_source(x, y):
    return np.ones_like(x, dtype=np.float64)


def logistic(z: np.ndarray, eps: float) -> np.ndarray:
    zz = np.asarray(z, dtype=np.float64) / float(eps)
    out = np.empty_like(zz)
    out[zz > 50.0] = 1.0
    out[zz < -50.0] = 0.0
    mask = (zz >= -50.0) & (zz <= 50.0)
    out[mask] = 1.0 / (1.0 + np.exp(-zz[mask]))
    return out


def softplus(z: np.ndarray, eps: float) -> np.ndarray:
    zz = np.asarray(z, dtype=np.float64) / float(eps)
    out = np.empty_like(zz)
    out[zz > 50.0] = z[zz > 50.0]
    out[zz < -50.0] = eps * np.exp(zz[zz < -50.0])
    mask = (zz >= -50.0) & (zz <= 50.0)
    out[mask] = eps * np.log1p(np.exp(zz[mask]))
    return out


def window_values(values: np.ndarray, c1: float, c2: float, eps: float, amp: float) -> np.ndarray:
    return amp * (logistic(values - c1, eps) - logistic(values - c2, eps))


def window_derivative(values: np.ndarray, c1: float, c2: float, eps: float, amp: float) -> np.ndarray:
    s1 = logistic(values - c1, eps)
    s2 = logistic(values - c2, eps)
    return amp * (s1 * (1.0 - s1) - s2 * (1.0 - s2)) / eps


def window_primitive(values: np.ndarray, c1: float, c2: float, eps: float, amp: float) -> np.ndarray:
    return amp * (softplus(values - c1, eps) - softplus(values - c2, eps))


def project_quadrature_values(space: DGSpace, values: np.ndarray, *, name: str) -> DGField:
    rhs = np.asarray(values, dtype=np.float64) @ space.quad_data.weighted_phi
    coeffs = rhs @ space.quad_data.MKrf_inv
    return space.field(np.ascontiguousarray(coeffs), name=name)


def field_from_moments(space: DGSpace, moments: np.ndarray, *, name: str) -> DGField:
    moments = np.asarray(moments, dtype=np.float64)
    expected = (space.mesh.num_tri, space.el_dof)
    if moments.shape != expected:
        raise ValueError(f"moments must have shape {expected}; got {moments.shape}")
    coeffs = (moments / space.mesh.aff_jacs[:, None]) @ space.quad_data.MKrf_inv
    return space.field(np.ascontiguousarray(coeffs), name=name)


def scalar_moments_from_values(space: DGSpace, values: np.ndarray) -> np.ndarray:
    rhs = np.asarray(values, dtype=np.float64) @ space.quad_data.weighted_phi
    rhs *= space.mesh.aff_jacs[:, None]
    return np.ascontiguousarray(rhs)


def block_source_from_scalar_moments(moments: np.ndarray, space: DGSpace) -> np.ndarray:
    out = np.zeros((space.mesh.num_tri, 3 * space.el_dof), dtype=np.float64)
    out[:, :space.el_dof] = moments
    return out


def flux_coefficients(result) -> np.ndarray:
    coeffs = np.asarray(result.flux.as_component_first(), dtype=np.float64)
    if coeffs.shape[0] != 2:
        raise ValueError(f"expected two flux components; got shape {coeffs.shape}")
    return np.ascontiguousarray(coeffs)


def vector_fields_from_flux(space: DGSpace, flux: np.ndarray, *, name: str):
    flux = np.asarray(flux, dtype=np.float64)
    expected = (2, space.mesh.num_tri, space.el_dof)
    if flux.shape != expected:
        raise ValueError(f"flux must have shape {expected}; got {flux.shape}")
    return (
        space.field(np.ascontiguousarray(flux[0]), name=f"{name}_x"),
        space.field(np.ascontiguousarray(flux[1]), name=f"{name}_y"),
    )


def mesh_edge_min_max(mesh: DGMesh) -> tuple[float, float]:
    lengths = np.linalg.norm(mesh.node_coords[mesh.edges[:, 1]] - mesh.node_coords[mesh.edges[:, 0]], axis=1)
    return float(np.min(lengths)), float(np.max(lengths))


def mass_from_values(space: DGSpace, values: np.ndarray) -> float:
    return float(np.einsum("K,Kq,q->", space.mesh.aff_jacs, values, space.quad_data.Krf_w, optimize=True))


def l2_from_values(space: DGSpace, values: np.ndarray) -> float:
    return float(np.sqrt(np.einsum("K,Kq,q->", space.mesh.aff_jacs, values * values, space.quad_data.Krf_w, optimize=True)))


def h1_flux_jump_norm(field: DGField, flux_coeffs: np.ndarray, trace: np.ndarray) -> tuple[float, float, float]:
    """Return ``(||q|| + ||u-uhat||, ||q||, ||u-uhat||)`` for an HDG state."""
    space = field.space
    mesh = space.mesh
    q = space.quad_data
    flux_coeffs = np.asarray(flux_coeffs, dtype=np.float64)
    qx_values = flux_coeffs[0] @ q.bas_of_quads
    qy_values = flux_coeffs[1] @ q.bas_of_quads
    flux_l2 = float(np.sqrt(np.einsum(
        "K,Kq,q->",
        mesh.aff_jacs,
        qx_values * qx_values + qy_values * qy_values,
        q.Krf_w,
        optimize=True,
    )))

    local_trace = hdg_assembly.element_traces(trace, space).reshape(mesh.num_tri, 3, q.edg_dof)
    u_face = np.einsum("Ki,fiq->Kfq", field.coeffs, q.bas_of_bd_quads, optimize=True)
    trace_face = np.einsum("Kfa,aq->Kfq", local_trace, q.bas1d_of_ref_edg_qds, optimize=True)
    jump_l2 = float(np.sqrt(np.einsum(
        "Kf,Kfq,q->",
        mesh.jacs_el_fc,
        (u_face - trace_face) * (u_face - trace_face),
        q.weights_JGL,
        optimize=True,
    )))
    return flux_l2 + jump_l2, flux_l2, jump_l2


def trace_from_field_faces(field: DGField) -> np.ndarray:
    """Project element face values to one trace polynomial per mesh edge."""
    space = field.space
    mesh = space.mesh
    q = space.quad_data
    rhs = np.zeros((mesh.num_edg, q.edg_dof), dtype=np.float64)
    mass = np.zeros((mesh.num_edg, q.edg_dof, q.edg_dof), dtype=np.float64)
    oriented = q.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling]
    face_rhs = mesh.jacs_el_fc[:, :, None] * np.einsum("Kfai,Ki->Kfa", oriented, field.coeffs, optimize=True)
    face_mass = mesh.jacs_el_fc[:, :, None, None] * q.M_rf_fc[None, None, :, :]
    for face in range(3):
        edges = mesh.loc2glob_edge[:, face]
        interior = np.isin(edges, mesh.int_edges_inds)
        np.add.at(rhs, edges[interior], face_rhs[interior, face])
        np.add.at(mass, edges[interior], face_mass[interior, face])

    trace = np.zeros((mesh.num_edg, q.edg_dof), dtype=np.float64)
    for edge in mesh.int_edges_inds:
        trace[edge] = np.linalg.solve(mass[edge], rhs[edge])
    return trace.reshape(-1)


def hdg_residual(
        field: DGField,
        flux_coeffs: np.ndarray,
        trace: np.ndarray,
        *,
        source_values: np.ndarray,
        stabilization: float,
) -> np.ndarray:
    """Assemble the nonlinear HDG residual from local and trace equations."""
    space = field.space
    mesh = space.mesh
    q = space.quad_data
    tau = _normalize_tau(stabilization, space)
    d0, d1, m_tau, m_n0, m_n1, _ = _local_solver_pre_mats(0.0, tau, space)
    element_boundary = diffusion_element_boundary_mats(tau, space)
    source = block_source_from_scalar_moments(scalar_moments_from_values(space, source_values), space)
    local_trace = hdg_assembly.element_traces(trace, space)

    flux_coeffs = np.asarray(flux_coeffs, dtype=np.float64)
    expected_flux_shape = (2, mesh.num_tri, q.el_dof)
    if flux_coeffs.shape != expected_flux_shape:
        raise ValueError(f"flux_coeffs must have shape {expected_flux_shape}; got {flux_coeffs.shape}")
    qx_coeffs = np.ascontiguousarray(flux_coeffs[0])
    qy_coeffs = np.ascontiguousarray(flux_coeffs[1])

    local = np.zeros((mesh.num_tri, 3 * q.el_dof), dtype=np.float64)
    local[:, :q.el_dof] = (
        np.einsum("Kij,Kj->Ki", m_tau, field.coeffs, optimize=True)
        + np.einsum("Kij,Kj->Ki", m_n0 - d0, qx_coeffs, optimize=True)
        + np.einsum("Kij,Kj->Ki", m_n1 - d1, qy_coeffs, optimize=True)
    )
    mass = mesh.aff_jacs[:, None, None] * q.MKrf[None, :, :]
    local[:, q.el_dof:2 * q.el_dof] = (
        np.einsum("Kij,Kj->Ki", d0, field.coeffs, optimize=True)
        - np.einsum("Kij,Kj->Ki", mass, qx_coeffs, optimize=True)
    )
    local[:, 2 * q.el_dof:] = (
        np.einsum("Kij,Kj->Ki", d1, field.coeffs, optimize=True)
        - np.einsum("Kij,Kj->Ki", mass, qy_coeffs, optimize=True)
    )
    local -= np.einsum("Kij,Kj->Ki", element_boundary, local_trace, optimize=True)
    local -= source

    trace_residual_full = np.zeros((mesh.num_edg, q.edg_dof), dtype=np.float64)
    oriented = q.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling]
    u_lift = np.einsum("Kfai,Ki->Kfa", oriented, field.coeffs, optimize=True)
    qx_lift = np.einsum("Kfai,Ki->Kfa", oriented, qx_coeffs, optimize=True)
    qy_lift = np.einsum("Kfai,Ki->Kfa", oriented, qy_coeffs, optimize=True)
    trace_by_edge = trace.reshape(mesh.num_edg, q.edg_dof)
    for local_face in range(3):
        edges = mesh.loc2glob_edge[:, local_face]
        face_contrib = mesh.jacs_el_fc[:, local_face, None] * (
            mesh.normals[:, local_face, 0, None] * qx_lift[:, local_face]
            + mesh.normals[:, local_face, 1, None] * qy_lift[:, local_face]
            + tau[:, local_face, None] * u_lift[:, local_face]
            - tau[:, local_face, None] * (trace_by_edge[edges] @ q.M_rf_fc.T)
        )
        np.add.at(trace_residual_full, edges, face_contrib)

    interior_trace = trace_residual_full[mesh.int_edges_inds].reshape(-1)
    return np.concatenate((local.reshape(-1), interior_trace))


def mixed_u_block_rhs_from_residual(residual: np.ndarray, space: DGSpace) -> np.ndarray:
    local_size = space.mesh.num_tri * 3 * space.el_dof
    local = np.asarray(residual[:local_size], dtype=np.float64).reshape(space.mesh.num_tri, 3 * space.el_dof)
    return np.ascontiguousarray(-local[:, :space.el_dof])


def physical_stiffness_blocks(space: DGSpace) -> np.ndarray:
    mesh = space.mesh
    q = space.quad_data
    gradients = np.einsum(
        "KcD,Diq->Kciq",
        mesh.inv_aff_mats_t,
        q.dbas_of_quads,
        optimize=True,
    )
    return np.einsum(
        "K,q,Kciq,Kcjq->Kij",
        mesh.aff_jacs,
        q.Krf_w,
        gradients,
        gradients,
        optimize=True,
    )


def build_flux_jump_gram_inverse(
        space: DGSpace,
        *,
        cg_rtol: float,
        cg_atol: float,
        cg_maxiter: int | None,
        verbose_every: int,
        verify_residual: bool,
) -> CondensedHDGGramInverse:
    """Build an SPD HDG dual-norm inverse.

    The requested primal HDG control is flux plus trace jump.  Since ``q`` and
    ``u`` are independent coordinates in the residual vector, the pure
    flux-jump Gram has continuous primal zero modes.  The primal stiffness
    block is therefore included as the standard SPD realization of this norm.
    """
    start = time.perf_counter()
    mesh = space.mesh
    q = space.quad_data
    tau_jac = mesh.jacs_el_fc
    face_uu = tau_jac[:, :, None, None] * q.face_element_test_element_trial[None, :, :, :]
    u_block = np.ascontiguousarray(physical_stiffness_blocks(space) + np.sum(face_uu, axis=1), dtype=np.float64)

    try:
        u_block_inverse = np.ascontiguousarray(np.linalg.inv(u_block), dtype=np.float64)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "flux-jump HDG Gram is singular for this space; this script is intended for p=2"
        ) from exc

    oriented = q.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling]
    face_u_trace = np.ascontiguousarray(
        tau_jac[:, :, None, None] * oriented.swapaxes(2, 3),
        dtype=np.float64,
    )
    face_trace_trace = np.ascontiguousarray(
        tau_jac[:, :, None, None] * q.M_rf_fc[None, None, :, :],
        dtype=np.float64,
    )

    edge_to_free = np.full(mesh.num_edg, -1, dtype=np.int64)
    edge_to_free[mesh.int_edges_inds] = np.arange(mesh.int_edges_inds.size)
    face_free_edges = np.ascontiguousarray(edge_to_free[mesh.loc2glob_edge], dtype=np.int64)
    interior_face_mask = face_free_edges.reshape(-1) >= 0
    interior_face_flat = np.flatnonzero(interior_face_mask).astype(np.int64)
    face_free_flat = np.ascontiguousarray(face_free_edges.reshape(-1)[interior_face_flat], dtype=np.int64)

    a_inv_b = np.einsum("Kij,Kfja->Kfia", u_block_inverse, face_u_trace, optimize=True)
    same_face_schur = face_trace_trace - np.einsum("Kfia,Kfib->Kfab", face_u_trace, a_inv_b, optimize=True)
    edge_blocks = np.zeros((mesh.int_edges_inds.size, q.edg_dof, q.edg_dof), dtype=np.float64)
    np.add.at(
        edge_blocks,
        face_free_flat,
        same_face_schur.reshape(-1, q.edg_dof, q.edg_dof)[interior_face_flat],
    )
    edge_block_inverse = np.ascontiguousarray(np.linalg.inv(edge_blocks), dtype=np.float64)

    element_count = mesh.num_tri
    element_dofs = q.el_dof
    edge_dofs = q.edg_dof
    local_dofs = element_count * 3 * element_dofs
    trace_dofs = mesh.int_edges_inds.size * edge_dofs
    compatible_sparse_nnz = max(
        element_count * (3 * element_dofs * element_dofs + 6 * element_dofs * edge_dofs)
        + mesh.int_edges_inds.size * edge_dofs * edge_dofs,
        1,
    )
    local_storage = (
        u_block.size
        + u_block_inverse.size
        + face_u_trace.size
        + face_trace_trace.size
        + edge_block_inverse.size
    )
    return CondensedHDGGramInverse(
        local_dofs=local_dofs,
        trace_dofs=trace_dofs,
        element_count=element_count,
        element_dofs=element_dofs,
        edge_dofs=edge_dofs,
        aff_jacs=np.ascontiguousarray(mesh.aff_jacs, dtype=np.float64),
        reference_mass=np.ascontiguousarray(q.MKrf, dtype=np.float64),
        reference_mass_inverse=np.ascontiguousarray(q.MKrf_inv, dtype=np.float64),
        u_block=u_block,
        u_block_inverse=u_block_inverse,
        face_u_trace=face_u_trace,
        face_trace_trace=face_trace_trace,
        face_free_edges=face_free_edges,
        interior_face_flat=interior_face_flat,
        face_free_flat=face_free_flat,
        edge_block_inverse=edge_block_inverse,
        setup_seconds=time.perf_counter() - start,
        default_rtol=float(cg_rtol),
        default_atol=float(cg_atol),
        default_maxiter=cg_maxiter,
        verbose_every=int(verbose_every),
        verify_residual=bool(verify_residual),
        fill_ratio=float(local_storage / compatible_sparse_nnz),
    )


def _extract_gmsh_model_to_mesh(gmsh, *, write_path: Path | None = None) -> DGMesh:
    if write_path is not None:
        gmsh.write(str(write_path))
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    if node_tags.size == 0:
        raise RuntimeError("Gmsh generated no nodes")
    order = np.argsort(node_tags)
    sorted_tags = node_tags[order]
    coords = node_coords.reshape(-1, 3)[order, :2]
    tag_to_index = {int(tag): index for index, tag in enumerate(sorted_tags)}
    _, tri_node_tags = gmsh.model.mesh.getElementsByType(2)
    if tri_node_tags.size == 0:
        raise RuntimeError("Gmsh generated no triangular elements")
    triangles = np.fromiter(
        (tag_to_index[int(tag)] for tag in tri_node_tags),
        dtype=np.int64,
        count=tri_node_tags.size,
    ).reshape(-1, 3)
    return DGMesh(coords, triangles)


def gmsh_star_mesh_with_size_callback(
        *,
        boundary_points: int,
        radius: float,
        amplitude: float,
        mode: int,
        hmin: float,
        hmax: float,
        size_callback,
        verbosity: int,
        algorithm: int | None,
        write_path: Path | None = None,
) -> DGMesh:
    import gmsh

    started_gmsh = not gmsh.isInitialized()
    if started_gmsh:
        gmsh.initialize()
    else:
        gmsh.clear()
    try:
        gmsh.model.add("adaptive_smooth_star")
        gmsh.option.setNumber("General.Verbosity", int(verbosity))
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.MeshSizeMin", float(hmin))
        gmsh.option.setNumber("Mesh.MeshSizeMax", float(hmax))
        if algorithm is not None:
            gmsh.option.setNumber("Mesh.Algorithm", int(algorithm))

        theta = np.linspace(0.0, 2.0 * np.pi, int(boundary_points), endpoint=False)
        rr = float(radius) + float(amplitude) * np.cos(int(mode) * theta)
        vertices = np.column_stack((rr * np.cos(theta), rr * np.sin(theta)))
        occ = gmsh.model.occ
        points = [occ.addPoint(float(x), float(y), 0.0, float(hmax)) for x, y in vertices]
        lines = [occ.addLine(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
        loop = occ.addCurveLoop(lines)
        surface = occ.addPlaneSurface([loop])
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [surface], name="adaptive_smooth_star")
        gmsh.model.mesh.setSizeCallback(size_callback)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.removeSizeCallback()
        return _extract_gmsh_model_to_mesh(gmsh, write_path=write_path)
    finally:
        if started_gmsh:
            gmsh.finalize()


def adaptive_remesh(
        space: DGSpace,
        indicator: DGField,
        args: argparse.Namespace,
        *,
        hmin: float,
        hmax: float,
) -> tuple[DGMesh, dict[str, float]]:
    """Generate a Gmsh mesh from the current scalar indicator."""
    from scipy.spatial import cKDTree

    points = space.mesh.flatten_mapped_reference_points(space.quad_data.Krf_quads)
    raw_values = np.asarray(indicator.values(), dtype=np.float64).reshape(-1)
    if raw_values.size == 0:
        raise ValueError("empty adaptivity indicator")
    lo = float(np.quantile(raw_values, args.adapt_low_quantile))
    hi = float(np.quantile(raw_values, args.adapt_high_quantile))
    if not np.isfinite(hi - lo) or hi <= lo:
        hi = float(np.max(raw_values))
        lo = float(np.min(raw_values))
    scale = max(hi - lo, 1.0e-30)
    scores = np.clip((raw_values - lo) / scale, 0.0, 1.0)
    tree = cKDTree(points)
    k = max(1, int(args.adapt_callback_neighbors))
    power = float(args.adapt_size_power)

    def callback(_dim, _tag, x, y, z, lc):
        del z, lc
        distances, indices = tree.query((x, y), k=k)
        distances = np.atleast_1d(distances)
        indices = np.atleast_1d(indices)
        weights = 1.0 / np.maximum(distances, 1.0e-12)
        score = float(np.dot(weights, scores[indices]) / np.sum(weights))
        score = min(max(score, 0.0), 1.0) ** power
        return float(np.clip(hmax - (hmax - hmin) * score, hmin, hmax))

    start = time.perf_counter()
    mesh = gmsh_star_mesh_with_size_callback(
        boundary_points=args.star_n,
        radius=args.star_r0,
        amplitude=args.star_amp,
        mode=args.star_mode,
        hmin=hmin,
        hmax=hmax,
        size_callback=callback,
        verbosity=args.gmsh_verbosity,
        algorithm=args.gmsh_algorithm,
    )
    elapsed = time.perf_counter() - start
    return mesh, {
        "indicator_min": float(np.min(raw_values)),
        "indicator_max": float(np.max(raw_values)),
        "indicator_qlo": lo,
        "indicator_qhi": hi,
        "elapsed": elapsed,
    }


def solve_hdg_with_fallback(
        source,
        reaction,
        boundary_condition,
        space: DGSpace,
        args: argparse.Namespace,
        *,
        diffusion: float = 1.0,
        stabilization: float = 1.0,
        initial_guess=None,
        problem_label: str,
):
    attempts = []
    if not args.skip_petsc:
        attempts.append(("petsc_bicgstab_gamg", {
            "solver": "petsc",
            "petsc_preset": "bicgstab_gamg",
            "preconditioner": None,
            "solver_rtol": args.linear_rtol,
            "solver_atol": args.linear_atol,
            "maxiter": args.linear_maxiter,
        }))
    attempts.extend([
        ("scipy_ilu_bicgstab", {
            "solver": "BICGSTAB",
            "preconditioner": "ilu",
            "solver_rtol": args.linear_rtol,
            "solver_atol": args.linear_atol,
            "maxiter": args.linear_maxiter,
            "ilu_drop_tol": args.ilu_drop_tol,
            "ilu_fill_factor": args.ilu_fill_factor,
        }),
        ("scipy_direct", {
            "solver": "direct",
            "preconditioner": None,
            "solver_rtol": args.linear_rtol,
            "solver_atol": args.linear_atol,
        }),
    ])

    last_error = None
    for label, kwargs in attempts:
        print(f"SOLVER_TRY problem={problem_label} label={label}", flush=True)
        try:
            options = DiffusionReactionHDGOptions(
                diffusion=diffusion,
                stabilization=stabilization,
                boundary_mode="eliminate",
                assembly_backend=args.hdg_assembly_backend,
                local_solver_backend=args.hdg_local_solver_backend,
                initial_guess=initial_guess,
                verbose=args.hdg_solver_verbose,
                scale_system=True,
                **kwargs,
            )
            result = DiffusionReactionHDGSolver(space, options=options).solve(
                source=source,
                reaction=reaction,
                boundary_condition=boundary_condition,
            )
            solve = result.global_solve_result
            print(
                f"SOLVER_OK problem={problem_label} label={label} "
                f"iters={None if solve is None else solve.iteration_count} "
                f"rel={np.nan if solve is None else solve.relative_residual_norm:.3e}",
                flush=True,
            )
            return result, label
        except Exception as exc:
            last_error = exc
            print(f"SOLVER_FAIL problem={problem_label} label={label} error={type(exc).__name__}: {exc}", flush=True)
            if "petsc4py is not importable" in str(exc):
                continue
    raise RuntimeError(f"all HDG solver attempts failed for {problem_label}") from last_error


def build_state(
        space: DGSpace,
        u: DGField,
        flux: np.ndarray,
        trace: np.ndarray,
        *,
        c1_phi: float,
        c2_phi: float,
        eps_phi: float,
        rho_amp: float,
        tau: float,
        gram_inverse: CondensedHDGGramInverse,
        gram_rtol: float,
        gram_atol: float,
        gram_maxiter: int | None,
) -> State:
    rho_values = window_values(u.values(), c1_phi, c2_phi, eps_phi, rho_amp)
    rho = project_quadrature_values(space, rho_values, name="rho")
    residual = hdg_residual(u, flux, trace, source_values=rho_values, stabilization=tau)
    coeff_l2 = float(np.linalg.norm(residual))
    residual_hdg2, diag = gram_inverse.dual_norm_squared(
        residual,
        rtol=gram_rtol,
        atol=gram_atol,
        maxiter=gram_maxiter,
    )
    if diag.info != 0:
        print(
            f"GRAM_WARNING info={diag.info} iterations={diag.iterations} rel={diag.relative_residual:.3e}",
            flush=True,
        )
    return State(
        u=u,
        flux=flux,
        trace=trace,
        rho=rho,
        residual=residual,
        residual_coeff_l2=coeff_l2,
        residual_hdg=float(np.sqrt(max(residual_hdg2, 0.0))),
        residual_hdg_squared=float(residual_hdg2),
    )


def compute_metrics(
        state: State,
        design: DesignState,
        *,
        c2_phi: float,
        eps_phi: float,
        params: StrategyParameters,
) -> dict[str, float]:
    space = state.u.space
    rho_values = state.rho.values()
    rho_design_values = design.rho_design.values()
    rho_design_l2 = max(design.rho_design.l2_norm(), 1.0e-30)
    u_values = state.u.values()
    dx_u, dy_u = state.u.grad_values()
    active = rho_values > params.active_threshold * params.rho_amp
    plateau = rho_values > params.plateau_threshold * params.rho_amp
    active_area = mass_from_values(space, active.astype(np.float64))
    plateau_area = mass_from_values(space, plateau.astype(np.float64))
    rel_design = l2_from_values(space, rho_values - rho_design_values) / rho_design_l2
    return {
        "min_u": float(np.min(u_values)),
        "max_u": float(np.max(u_values)),
        "max_rho": float(np.max(rho_values)),
        "mass_rho": mass_from_values(space, rho_values),
        "rho_l2": l2_from_values(space, rho_values),
        "energy_phi": float(np.sqrt(np.einsum(
            "K,Kq,q->",
            space.mesh.aff_jacs,
            dx_u * dx_u + dy_u * dy_u,
            space.quad_data.Krf_w,
            optimize=True,
        ))),
        "active_area": active_area,
        "plateau_area": plateau_area,
        "plateau_frac": plateau_area / max(active_area, 1.0e-30),
        "rel_rho_design": rel_design,
        "annular_phi_minus_c2": float(np.max(u_values) - c2_phi),
    }


def write_newton_row(writer: csv.DictWriter, **row) -> None:
    writer.writerow(row)


def transfer_field(field: DGField, target_space: DGSpace, plan, *, name: str) -> DGField:
    transferred, diag = project_field(field, target_space, plan=plan, verbose=False)
    if diag.n_missed_points:
        print(
            f"TRANSFER_WARNING field={field.name} missed={diag.n_missed_points}/{diag.n_target_points}",
            flush=True,
        )
    return target_space.field(transferred.coeffs, name=name)


def make_adapt_indicator_from_rho(rho: DGField, eps: float, grad_weight: float, *, name: str) -> DGField:
    dx_rho, dy_rho = rho.grad_values()
    values = rho.values() + float(grad_weight) * float(eps) * np.sqrt(dx_rho * dx_rho + dy_rho * dy_rho)
    return project_quadrature_values(rho.space, values, name=name)


class PyVistaStrategyPlotter:
    """PyVista plotting and frame writer using ``hdgfem.io.plot`` helpers."""

    def __init__(
            self,
            args: argparse.Namespace,
            *,
            run_tag: str,
            run_dir: Path,
            frame_writer: csv.DictWriter,
    ) -> None:
        self.args = args
        self.run_tag = run_tag
        self.run_dir = run_dir
        self.frame_writer = frame_writer
        self.frame_counter = 0
        self.frame_dir = args.frame_dir or (run_dir / "frames")
        if args.save_frames:
            self.frame_dir.mkdir(parents=True, exist_ok=True)

    @property
    def active(self) -> bool:
        return bool(self.args.plot or self.args.save_frames)

    def _plot(
            self,
            fields: list[DGField],
            titles: list[str],
            *,
            save_path: Path | None,
            show: bool,
            window_size: tuple[int, int],
    ) -> None:
        if not fields:
            return
        try:
            plotter = plot_fields(
                fields,
                titles=titles,
                resolution=self.args.plot_resolution,
                shape=(1, len(fields)),
                show_mesh=True,
                show=show,
                off_screen=save_path is not None or self.args.plot_off_screen,
                window_size=window_size,
                share_clim=False,
            )
            if save_path is not None:
                plotter.screenshot(str(save_path))
            plotter.close()
        except Exception as exc:
            print(f"PLOT_SKIP stage={titles[0] if titles else 'unknown'} error={type(exc).__name__}: {exc}", flush=True)

    def emit(
            self,
            fields: list[DGField],
            titles: list[str],
            *,
            stage: str,
            ieps,
            k,
            eps_phi: float,
            residual: float,
            metrics: dict[str, float],
            token: str,
            save: bool,
            show: bool,
    ) -> None:
        if not self.active:
            return
        save_path = None
        if self.args.save_frames and save:
            save_path = self.frame_dir / f"{self.run_tag}_frame_{self.frame_counter:04d}_{token}.png"
        if save_path is not None:
            self._plot(
                fields,
                titles,
                save_path=save_path,
                show=False,
                window_size=(self.args.frame_window_width, self.args.frame_window_height),
            )
            self.frame_writer.writerow({
                "frame": self.frame_counter,
                "runTag": self.run_tag,
                "stage": stage,
                "ieps": ieps,
                "k": k,
                "nt": fields[0].space.mesh.num_tri if fields else "NA",
                "ndof": fields[0].space.ndof if fields else "NA",
                "epsPhi": eps_phi,
                "resHdg": residual,
                "massRho": metrics.get("mass_rho", ""),
                "maxRho": metrics.get("max_rho", ""),
                "activeArea": metrics.get("active_area", ""),
                "plateauArea": metrics.get("plateau_area", ""),
                "relRhoDesign": metrics.get("rel_rho_design", ""),
                "filename": save_path,
            })
            self.frame_counter += 1
        if self.args.plot and show:
            self._plot(
                fields,
                titles,
                save_path=None,
                show=not self.args.plot_off_screen,
                window_size=(self.args.plot_window_width, self.args.plot_window_height),
            )


def remesh_design_pre_newton(
        space: DGSpace,
        design: DesignState,
        args: argparse.Namespace,
        params: StrategyParameters,
        *,
        initial_hmin: float,
        initial_hmax: float,
) -> tuple[DGSpace, DesignState, object, str, dict[str, float]]:
    indicator = make_adapt_indicator_from_rho(
        design.rho_design,
        design.eps_t,
        params.adapt_grad_weight,
        name="pre_adapt_indicator",
    )
    old_space = space
    old_nt = old_space.mesh.num_tri
    old_ndof = old_space.ndof
    old_mass = mass_from_values(space, design.rho_design.values())
    new_mesh, info = adaptive_remesh(old_space, indicator, args, hmin=initial_hmin, hmax=initial_hmax)
    new_space = DGSpace(new_mesh, args.order, basis_type=args.basis)
    plan = build_transfer_plan(old_space, new_space, verbose=False)
    torsion = transfer_field(design.torsion, new_space, plan, name="T")
    rho_design_values = window_values(torsion.values(), design.c1_t, design.c2_t, design.eps_t, params.rho_amp)
    rho_design = project_quadrature_values(new_space, rho_design_values, name="rhoDesign")
    phi_result, phi_solver = solve_hdg_with_fallback(
        rho_design,
        0.0,
        zero_boundary,
        new_space,
        args,
        diffusion=1.0,
        stabilization=args.hdg_tau,
        problem_label="phi_design_preadapt",
    )
    new_design = DesignState(
        torsion=torsion,
        rho_design=rho_design,
        phi_design=phi_result.field,
        c1_t=design.c1_t,
        c2_t=design.c2_t,
        eps_t=design.eps_t,
    )
    new_mass = mass_from_values(new_space, rho_design.values())
    info.update({
        "nt_old": old_nt,
        "ndof_old": old_ndof,
        "nt_new": new_mesh.num_tri,
        "ndof_new": new_space.ndof,
        "mass_before": old_mass,
        "mass_after": new_mass,
        "mass_rel": abs(new_mass - old_mass) / max(abs(old_mass), 1.0e-30),
    })
    return new_space, new_design, phi_result, phi_solver, info


def scheduled_adapt(
        space: DGSpace,
        state: State,
        design: DesignState,
        args: argparse.Namespace,
        params: StrategyParameters,
        *,
        eps_phi: float,
        c1_phi: float,
        c2_phi: float,
        initial_hmin: float,
        initial_hmax: float,
        gram_args: dict,
) -> tuple[DGSpace, State, DesignState, CondensedHDGGramInverse, dict[str, float]]:
    old_space = space
    old_metrics = {
        "nt_old": old_space.mesh.num_tri,
        "ndof_old": old_space.ndof,
        "mass_before": mass_from_values(old_space, state.rho.values()),
        "max_rho_before": float(np.max(state.rho.values())),
    }
    indicator = make_adapt_indicator_from_rho(
        state.rho,
        eps_phi,
        params.adapt_grad_weight,
        name="scheduled_adapt_indicator",
    )
    new_mesh, info = adaptive_remesh(old_space, indicator, args, hmin=initial_hmin, hmax=initial_hmax)
    new_space = DGSpace(new_mesh, args.order, basis_type=args.basis)
    plan = build_transfer_plan(old_space, new_space, verbose=False)

    torsion = transfer_field(design.torsion, new_space, plan, name="T")
    phi_design = transfer_field(design.phi_design, new_space, plan, name="phiDesign")
    u = transfer_field(state.u, new_space, plan, name="phi")
    qx_old, qy_old = vector_fields_from_flux(old_space, state.flux, name="flux")
    qx = transfer_field(qx_old, new_space, plan, name="flux_x")
    qy = transfer_field(qy_old, new_space, plan, name="flux_y")
    flux = np.ascontiguousarray(np.stack((qx.coeffs, qy.coeffs), axis=0))
    trace = trace_from_field_faces(u)

    rho_design_values = window_values(torsion.values(), design.c1_t, design.c2_t, design.eps_t, params.rho_amp)
    rho_design = project_quadrature_values(new_space, rho_design_values, name="rhoDesign")
    new_design = DesignState(
        torsion=torsion,
        rho_design=rho_design,
        phi_design=phi_design,
        c1_t=design.c1_t,
        c2_t=design.c2_t,
        eps_t=design.eps_t,
    )
    gram_inverse = build_flux_jump_gram_inverse(new_space, **gram_args)
    new_state = build_state(
        new_space,
        u,
        flux,
        trace,
        c1_phi=c1_phi,
        c2_phi=c2_phi,
        eps_phi=eps_phi,
        rho_amp=params.rho_amp,
        tau=args.hdg_tau,
        gram_inverse=gram_inverse,
        gram_rtol=args.gram_cg_rtol,
        gram_atol=args.gram_cg_atol,
        gram_maxiter=args.gram_cg_maxiter,
    )
    new_metrics = {
        "nt_new": new_mesh.num_tri,
        "ndof_new": new_space.ndof,
        "mass_after": mass_from_values(new_space, new_state.rho.values()),
        "max_rho_after": float(np.max(new_state.rho.values())),
    }
    mass_rel = abs(new_metrics["mass_after"] - old_metrics["mass_before"]) / max(abs(old_metrics["mass_before"]), 1.0e-30)
    max_rel = abs(new_metrics["max_rho_after"] - old_metrics["max_rho_before"]) / max(abs(old_metrics["max_rho_before"]), 1.0e-30)
    info.update(old_metrics)
    info.update(new_metrics)
    info.update({"mass_rel": mass_rel, "max_rho_rel": max_rel})
    return new_space, new_state, new_design, gram_inverse, info


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--mesh-size", type=float, default=0.08)
    parser.add_argument("--star-n", type=int, default=260)
    parser.add_argument("--star-r0", type=float, default=1.5)
    parser.add_argument("--star-amp", type=float, default=0.32)
    parser.add_argument("--star-mode", type=int, default=5)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--basis", default="dub_orth", choices=("bernstein", "hier_C0", "dub_orth"))
    parser.add_argument("--hdg-tau", type=float, default=1.0)
    parser.add_argument("--hdg-assembly-backend", choices=("numpy", "numba", "auto"), default="numba")
    parser.add_argument("--hdg-local-solver-backend", choices=("numpy", "numba"), default="numba")
    parser.add_argument("--hdg-solver-verbose", type=int, default=0)
    parser.add_argument("--gmsh-verbosity", type=int, default=0)
    parser.add_argument("--gmsh-algorithm", type=int, default=None)
    parser.add_argument("--skip-petsc", action="store_true")
    parser.add_argument("--linear-rtol", type=float, default=1.0e-10)
    parser.add_argument("--linear-atol", type=float, default=1.0e-12)
    parser.add_argument("--linear-maxiter", type=int, default=None)
    parser.add_argument("--ilu-drop-tol", type=float, default=1.0e-10)
    parser.add_argument("--ilu-fill-factor", type=float, default=50.0)
    parser.add_argument("--gram-cg-rtol", type=float, default=1.0e-9)
    parser.add_argument("--gram-cg-atol", type=float, default=1.0e-12)
    parser.add_argument("--gram-cg-maxiter", type=int, default=250)
    parser.add_argument("--gram-cg-verbose-every", type=int, default=0)
    parser.add_argument("--gram-no-verify", action="store_true")
    parser.add_argument("--eps-ratios", default=None, help="comma-separated override for epsilon continuation")
    parser.add_argument("--max-it", type=int, default=None)
    parser.add_argument("--tol-res", type=float, default=None)
    parser.add_argument("--tol-newton", type=float, default=None)
    parser.add_argument("--no-preadapt", action="store_true")
    parser.add_argument("--no-adapt", action="store_true")
    parser.add_argument("--adapt-after-last-eps", action="store_true")
    parser.add_argument("--adapt-after-accepted", type=int, default=None)
    parser.add_argument("--adapt-grad-weight", type=float, default=None)
    parser.add_argument("--adapt-low-quantile", type=float, default=0.10)
    parser.add_argument("--adapt-high-quantile", type=float, default=0.98)
    parser.add_argument("--adapt-callback-neighbors", type=int, default=8)
    parser.add_argument("--adapt-size-power", type=float, default=1.0)
    parser.add_argument("--plot", action="store_true", help="show PyVista plot windows at enabled stages")
    parser.add_argument("--plot-off-screen", action="store_true", help="render plot windows off-screen")
    parser.add_argument("--plot-resolution", type=int, default=10)
    parser.add_argument("--plot-window-width", type=int, default=1600)
    parser.add_argument("--plot-window-height", type=int, default=700)
    parser.add_argument("--plot-initial", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-design", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-newton", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-newton-every", type=int, default=5)
    parser.add_argument("--plot-adapt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-frames", action="store_true", help="save PyVista PNG frames at enabled stages")
    parser.add_argument("--frame-dir", type=Path, default=None)
    parser.add_argument("--frame-every", type=int, default=None)
    parser.add_argument("--frame-design", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frame-adapt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frame-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frame-window-width", type=int, default=1600)
    parser.add_argument("--frame-window-height", type=int, default=700)
    return parser.parse_args(argv)


def configure_params(args: argparse.Namespace) -> StrategyParameters:
    params = StrategyParameters()
    if args.eps_ratios:
        params.eps_phi_ratios = tuple(float(x.strip()) for x in args.eps_ratios.split(",") if x.strip())
    if args.max_it is not None:
        params.max_it = int(args.max_it)
    if args.tol_res is not None:
        params.tol_res = float(args.tol_res)
    if args.tol_newton is not None:
        params.tol_newton = float(args.tol_newton)
    if args.no_preadapt:
        params.pre_adapt_design = False
    if args.no_adapt:
        params.use_adapt = False
    if args.adapt_after_last_eps:
        params.adapt_after_last_eps = True
    if args.adapt_after_accepted is not None:
        params.adapt_after_accepted = int(args.adapt_after_accepted)
    if args.adapt_grad_weight is not None:
        params.adapt_grad_weight = float(args.adapt_grad_weight)
    return params


def csv_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_tag = args.run_tag or f"strategyA_hdg_adapt_star_{timestamp}"
    run_dir = args.run_dir or (REPO_ROOT / "run_logs" / run_tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    return (
        run_dir / f"{run_tag}_newton.csv",
        run_dir / f"{run_tag}_adapt.csv",
        run_dir / f"{run_tag}_frames.csv",
        run_dir / f"{run_tag}_summary.txt",
    )


def run_strategy(args: argparse.Namespace) -> State:
    params = configure_params(args)
    if args.order != 2:
        raise ValueError("this comparison script is intentionally restricted to p=2 for now")

    newton_csv, adapt_csv, frame_csv, summary_path = csv_paths(args)
    run_tag = newton_csv.name.removesuffix("_newton.csv")
    run_dir = newton_csv.parent
    frame_every = args.frame_every if args.frame_every is not None else args.plot_newton_every
    total_start = time.perf_counter()
    print("========== START STRATEGY A HDG ADAPTIVE TORSION NEWTON ==========")
    print(f"RUN_TAG {run_tag}")
    print(f"NEWTON_CSV {newton_csv}")
    print(f"ADAPT_CSV {adapt_csv}")
    print(f"FRAME_CSV {frame_csv}")

    newton_fields = [
        "record", "runTag", "ieps", "epsPhiRatio", "epsPhi", "k", "nt", "ndof",
        "resHdg", "resHdg2", "resCoeffL2", "stepHdg", "stepFluxL2", "stepJumpL2",
        "alpha", "bt", "muShift", "minU", "maxU", "maxRho", "massRho", "rhoL2",
        "energyPhi", "activeArea", "plateauArea", "plateauFrac", "relRhoDesign",
        "annularPhiMinusC2", "solveTime", "metricTime", "stepTime", "solver", "status",
    ]
    adapt_fields = [
        "record", "runTag", "ieps", "k", "ntOld", "ndofOld", "ntNew", "ndofNew",
        "eps", "hMin", "hMax", "indicatorMin", "indicatorMax", "indicatorQlo",
        "indicatorQhi", "massBefore", "massAfter", "massRel", "maxRhoRel",
        "adaptTime", "status",
    ]
    frame_fields = [
        "frame", "runTag", "stage", "ieps", "k", "nt", "ndof", "epsPhi", "resHdg",
        "massRho", "maxRho", "activeArea", "plateauArea", "relRhoDesign", "filename",
    ]

    with (
        newton_csv.open("w", newline="", encoding="utf-8") as newton_handle,
        adapt_csv.open("w", newline="", encoding="utf-8") as adapt_handle,
        frame_csv.open("w", newline="", encoding="utf-8") as frame_handle,
    ):
        newton_writer = csv.DictWriter(newton_handle, fieldnames=newton_fields)
        adapt_writer = csv.DictWriter(adapt_handle, fieldnames=adapt_fields)
        frame_writer = csv.DictWriter(frame_handle, fieldnames=frame_fields)
        newton_writer.writeheader()
        adapt_writer.writeheader()
        frame_writer.writeheader()
        plotter = PyVistaStrategyPlotter(args, run_tag=run_tag, run_dir=run_dir, frame_writer=frame_writer)

        mesh = gmsh_smooth_star_mesh(
            args.mesh_size,
            boundary_points=args.star_n,
            radius=args.star_r0,
            amplitude=args.star_amp,
            mode=args.star_mode,
            verbosity=args.gmsh_verbosity,
            algorithm=args.gmsh_algorithm,
        )
        initial_hmin, initial_hmax = mesh_edge_min_max(mesh)
        space = DGSpace(mesh, args.order, basis_type=args.basis)
        print(
            f"GEOMETRY smooth_star starN={args.star_n} r0={args.star_r0} "
            f"amp={args.star_amp} mode={args.star_mode} meshSize={args.mesh_size}"
        )
        print(
            f"INITIAL_MESH nt={mesh.num_tri} ndof={space.ndof} hMin={initial_hmin:.6e} "
            f"hMax={initial_hmax:.6e}"
        )
        if args.plot_initial:
            mesh_field = space.field(np.zeros(space.shape, dtype=np.float64), name="mesh")
            plotter.emit(
                [mesh_field],
                ["Initial mesh"],
                stage="INITIAL_MESH",
                ieps=-1,
                k=-1,
                eps_phi=0.0,
                residual=0.0,
                metrics={},
                token="initial_mesh",
                save=False,
                show=True,
            )

        torsion_result, torsion_solver = solve_hdg_with_fallback(
            one_source,
            0.0,
            zero_boundary,
            space,
            args,
            diffusion=1.0,
            stabilization=args.hdg_tau,
            problem_label="torsion",
        )
        torsion = torsion_result.field
        t_values = torsion.values()
        t_max = float(np.max(t_values))
        if t_max <= 1.0e-14:
            raise RuntimeError("torsion maximum is too small")
        c1_t = params.alpha_t1 * t_max
        c2_t = params.alpha_t2 * t_max
        eps_t = params.eps_t_ratio * (c2_t - c1_t)
        rho_design_values = window_values(t_values, c1_t, c2_t, eps_t, params.rho_amp)
        rho_design = project_quadrature_values(space, rho_design_values, name="rhoDesign")
        print(
            f"TORSION solver={torsion_solver} Tmax={t_max:.6e} c1T={c1_t:.6e} "
            f"c2T={c2_t:.6e} epsT={eps_t:.6e}"
        )
        if args.plot_design:
            plotter.emit(
                [torsion, rho_design],
                ["Torsion T", "Torsion-designed rho"],
                stage="TORSION_DESIGN",
                ieps=-1,
                k=-1,
                eps_phi=eps_t,
                residual=0.0,
                metrics={"mass_rho": mass_from_values(space, rho_design.values()), "max_rho": float(np.max(rho_design.values()))},
                token="torsion_design",
                save=bool(args.frame_design),
                show=True,
            )

        phi_result, phi_solver = solve_hdg_with_fallback(
            rho_design,
            0.0,
            zero_boundary,
            space,
            args,
            diffusion=1.0,
            stabilization=args.hdg_tau,
            initial_guess=torsion_result.trace,
            problem_label="phi_design",
        )
        design = DesignState(
            torsion=torsion,
            rho_design=rho_design,
            phi_design=phi_result.field,
            c1_t=c1_t,
            c2_t=c2_t,
            eps_t=eps_t,
        )
        print(
            f"PHI_DESIGN solver={phi_solver} max={np.max(design.phi_design.values()):.6e} "
            f"rhoDesignMass={mass_from_values(space, rho_design.values()):.6e}"
        )
        if args.plot_design:
            plotter.emit(
                [design.phi_design, design.rho_design],
                ["Poisson initializer phiDesign", "rhoDesign"],
                stage="PHI_DESIGN",
                ieps=-1,
                k=-1,
                eps_phi=eps_t,
                residual=0.0,
                metrics={"mass_rho": mass_from_values(space, design.rho_design.values()), "max_rho": float(np.max(design.rho_design.values()))},
                token="phi_design",
                save=bool(args.frame_design),
                show=True,
            )

        if params.pre_adapt_design:
            if args.plot_adapt:
                pre_indicator = make_adapt_indicator_from_rho(
                    design.rho_design,
                    design.eps_t,
                    params.adapt_grad_weight,
                    name="pre_adapt_indicator_plot",
                )
                plotter.emit(
                    [design.rho_design, pre_indicator],
                    ["Before pre-adapt: rhoDesign", "Pre-adapt indicator"],
                    stage="BEFORE_PREADAPT",
                    ieps=-1,
                    k=-1,
                    eps_phi=eps_t,
                    residual=0.0,
                    metrics={"mass_rho": mass_from_values(space, design.rho_design.values()), "max_rho": float(np.max(design.rho_design.values()))},
                    token="before_preadapt",
                    save=bool(args.frame_adapt),
                    show=True,
                )
            print("PREADAPT_START", flush=True)
            old_nt, old_ndof = space.mesh.num_tri, space.ndof
            space, design, phi_result, phi_solver, info = remesh_design_pre_newton(
                space,
                design,
                args,
                params,
                initial_hmin=initial_hmin,
                initial_hmax=initial_hmax,
            )
            print(
                f"PREADAPT_DONE ntOld={old_nt} ntNew={space.mesh.num_tri} "
                f"ndofOld={old_ndof} ndofNew={space.ndof} massRel={info['mass_rel']:.3e} "
                f"time={info['elapsed']:.3f}"
            )
            adapt_writer.writerow({
                "record": "PREADAPT",
                "runTag": run_tag,
                "ieps": -1,
                "k": -1,
                "ntOld": info["nt_old"],
                "ndofOld": info["ndof_old"],
                "ntNew": info["nt_new"],
                "ndofNew": info["ndof_new"],
                "eps": eps_t,
                "hMin": initial_hmin,
                "hMax": initial_hmax,
                "indicatorMin": info["indicator_min"],
                "indicatorMax": info["indicator_max"],
                "indicatorQlo": info["indicator_qlo"],
                "indicatorQhi": info["indicator_qhi"],
                "massBefore": info["mass_before"],
                "massAfter": info["mass_after"],
                "massRel": info["mass_rel"],
                "maxRhoRel": "",
                "adaptTime": info["elapsed"],
                "status": "PREADAPT_DONE",
            })
            if args.plot_adapt:
                plotter.emit(
                    [design.rho_design, design.phi_design],
                    ["After pre-adapt: rhoDesign", "After pre-adapt: phiDesign"],
                    stage="AFTER_PREADAPT",
                    ieps=-1,
                    k=-1,
                    eps_phi=eps_t,
                    residual=0.0,
                    metrics={"mass_rho": mass_from_values(space, design.rho_design.values()), "max_rho": float(np.max(design.rho_design.values()))},
                    token="after_preadapt",
                    save=bool(args.frame_adapt),
                    show=True,
                )

        phi_design_values = design.phi_design.values()
        phi_max = float(np.max(phi_design_values))
        if phi_max <= 1.0e-14:
            raise RuntimeError("phiDesign maximum is too small")
        c1_phi = params.beta_phi1 * phi_max
        c2_phi = params.beta_phi2 * phi_max
        width_phi = c2_phi - c1_phi
        eps_phi = params.eps_phi_ratios[0] * width_phi
        u = design.phi_design
        flux = flux_coefficients(phi_result)
        if phi_result.field.space is not space:
            # Pre-adaptation rebuilt phi_result on the new space; this branch is
            # defensive for future refactors.
            flux = np.zeros((2, space.mesh.num_tri, space.el_dof), dtype=np.float64)
        trace = phi_result.trace.copy()
        if trace.size != space.mesh.num_edg * space.quad_data.edg_dof:
            trace = trace_from_field_faces(u)

        gram_args = {
            "cg_rtol": args.gram_cg_rtol,
            "cg_atol": args.gram_cg_atol,
            "cg_maxiter": args.gram_cg_maxiter,
            "verbose_every": args.gram_cg_verbose_every,
            "verify_residual": not args.gram_no_verify,
        }
        gram_inverse = build_flux_jump_gram_inverse(space, **gram_args)
        state = build_state(
            space,
            u,
            flux,
            trace,
            c1_phi=c1_phi,
            c2_phi=c2_phi,
            eps_phi=eps_phi,
            rho_amp=params.rho_amp,
            tau=args.hdg_tau,
            gram_inverse=gram_inverse,
            gram_rtol=args.gram_cg_rtol,
            gram_atol=args.gram_cg_atol,
            gram_maxiter=args.gram_cg_maxiter,
        )
        setup_metrics = compute_metrics(state, design, c2_phi=c2_phi, eps_phi=eps_phi, params=params)
        write_newton_row(
            newton_writer,
            record="SETUP",
            runTag=run_tag,
            ieps="NA",
            epsPhiRatio="NA",
            epsPhi=eps_phi,
            k="NA",
            nt=space.mesh.num_tri,
            ndof=space.ndof,
            resHdg=state.residual_hdg,
            resHdg2=state.residual_hdg_squared,
            resCoeffL2=state.residual_coeff_l2,
            stepHdg="NA",
            stepFluxL2="NA",
            stepJumpL2="NA",
            alpha="NA",
            bt="NA",
            muShift=params.mu_shift,
            minU=setup_metrics["min_u"],
            maxU=setup_metrics["max_u"],
            maxRho=setup_metrics["max_rho"],
            massRho=setup_metrics["mass_rho"],
            rhoL2=setup_metrics["rho_l2"],
            energyPhi=setup_metrics["energy_phi"],
            activeArea=setup_metrics["active_area"],
            plateauArea=setup_metrics["plateau_area"],
            plateauFrac=setup_metrics["plateau_frac"],
            relRhoDesign=setup_metrics["rel_rho_design"],
            annularPhiMinusC2=setup_metrics["annular_phi_minus_c2"],
            solveTime="NA",
            metricTime="NA",
            stepTime="NA",
            solver="NA",
            status="INITIAL",
        )
        if args.plot_design:
            plotter.emit(
                [state.u, state.rho, design.rho_design, design.phi_design],
                ["Initial phi", "Initial rho=f(phi)", "rhoDesign", "phiDesign"],
                stage="INIT",
                ieps=-1,
                k=-1,
                eps_phi=eps_phi,
                residual=state.residual_hdg,
                metrics=setup_metrics,
                token="init",
                save=bool(args.frame_design),
                show=True,
            )

        mu_shift = params.mu_shift
        for ieps, eps_ratio in enumerate(params.eps_phi_ratios):
            eps_phi = eps_ratio * width_phi
            mu_shift = max(mu_shift, 2.0)
            state = build_state(
                space,
                state.u,
                state.flux,
                state.trace,
                c1_phi=c1_phi,
                c2_phi=c2_phi,
                eps_phi=eps_phi,
                rho_amp=params.rho_amp,
                tau=args.hdg_tau,
                gram_inverse=gram_inverse,
                gram_rtol=args.gram_cg_rtol,
                gram_atol=args.gram_cg_atol,
                gram_maxiter=args.gram_cg_maxiter,
            )
            metrics = compute_metrics(state, design, c2_phi=c2_phi, eps_phi=eps_phi, params=params)
            mass_initial = metrics["mass_rho"]
            rho_design_mass = mass_from_values(space, design.rho_design.values())
            mass_floor = params.mass_floor_fraction * max(mass_initial, rho_design_mass)
            print(
                f"EPS_START ieps={ieps} epsRatio={eps_ratio} epsPhi={eps_phi:.6e} "
                f"nt={space.mesh.num_tri} ndof={space.ndof} resHdg={state.residual_hdg:.6e} "
                f"resCoeff={state.residual_coeff_l2:.6e} mass={mass_initial:.6e}"
            )
            write_newton_row(
                newton_writer,
                record="EPS_START",
                runTag=run_tag,
                ieps=ieps,
                epsPhiRatio=eps_ratio,
                epsPhi=eps_phi,
                k="NA",
                nt=space.mesh.num_tri,
                ndof=space.ndof,
                resHdg=state.residual_hdg,
                resHdg2=state.residual_hdg_squared,
                resCoeffL2=state.residual_coeff_l2,
                stepHdg="NA",
                stepFluxL2="NA",
                stepJumpL2="NA",
                alpha="NA",
                bt="NA",
                muShift=mu_shift,
                minU=metrics["min_u"],
                maxU=metrics["max_u"],
                maxRho=metrics["max_rho"],
                massRho=metrics["mass_rho"],
                rhoL2=metrics["rho_l2"],
                energyPhi=metrics["energy_phi"],
                activeArea=metrics["active_area"],
                plateauArea=metrics["plateau_area"],
                plateauFrac=metrics["plateau_frac"],
                relRhoDesign=metrics["rel_rho_design"],
                annularPhiMinusC2=metrics["annular_phi_minus_c2"],
                solveTime="NA",
                metricTime="NA",
                stepTime="NA",
                solver="NA",
                status="EPS_START",
            )

            reject_count = 0
            stagnation_count = 0
            accepted_this_eps = 0
            adapted_this_eps = False

            for k in range(params.max_it):
                step_start = time.perf_counter()
                rho_values = state.rho.values()
                df_values = window_derivative(state.u.values(), c1_phi, c2_phi, eps_phi, params.rho_amp)
                source_moments = mixed_u_block_rhs_from_residual(state.residual, space)
                reaction_values = -df_values
                source_h = field_from_moments(space, source_moments, name=f"newton_source_{ieps}_{k}")
                reaction_h = project_quadrature_values(space, reaction_values, name=f"newton_reaction_{ieps}_{k}")

                solve_start = time.perf_counter()
                correction, solver_label = solve_hdg_with_fallback(
                    source_h,
                    reaction_h,
                    zero_boundary,
                    space,
                    args,
                    diffusion=1.0 + mu_shift,
                    stabilization=args.hdg_tau,
                    initial_guess=np.zeros_like(state.trace),
                    problem_label=f"newton_ieps{ieps}_k{k}",
                )
                solve_time = time.perf_counter() - solve_start
                du = correction.field
                flux_du = flux_coefficients(correction)
                trace_du = correction.trace
                step_norm, step_flux_l2, step_jump_l2 = h1_flux_jump_norm(du, flux_du, trace_du)

                if state.residual_hdg < params.tol_res or step_norm < params.tol_newton:
                    metrics = compute_metrics(state, design, c2_phi=c2_phi, eps_phi=eps_phi, params=params)
                    write_newton_row(
                        newton_writer,
                        record="NEWTON",
                        runTag=run_tag,
                        ieps=ieps,
                        epsPhiRatio=eps_ratio,
                        epsPhi=eps_phi,
                        k=k,
                        nt=space.mesh.num_tri,
                        ndof=space.ndof,
                        resHdg=state.residual_hdg,
                        resHdg2=state.residual_hdg_squared,
                        resCoeffL2=state.residual_coeff_l2,
                        stepHdg=step_norm,
                        stepFluxL2=step_flux_l2,
                        stepJumpL2=step_jump_l2,
                        alpha=0.0,
                        bt=0,
                        muShift=mu_shift,
                        minU=metrics["min_u"],
                        maxU=metrics["max_u"],
                        maxRho=metrics["max_rho"],
                        massRho=metrics["mass_rho"],
                        rhoL2=metrics["rho_l2"],
                        energyPhi=metrics["energy_phi"],
                        activeArea=metrics["active_area"],
                        plateauArea=metrics["plateau_area"],
                        plateauFrac=metrics["plateau_frac"],
                        relRhoDesign=metrics["rel_rho_design"],
                        annularPhiMinusC2=metrics["annular_phi_minus_c2"],
                        solveTime=solve_time,
                        metricTime=0.0,
                        stepTime=time.perf_counter() - step_start,
                        solver=solver_label,
                        status="CONVERGED",
                    )
                    print(f"STEP ieps={ieps} k={k} resHdg={state.residual_hdg:.6e} status=CONVERGED")
                    break

                old_state = state
                old_merit = old_state.residual_hdg
                alpha = 1.0
                n_backtrack = 0
                accepted = False
                best_trial = None

                while alpha >= params.alpha_min and n_backtrack <= params.max_backtrack:
                    trial_u = space.field(old_state.u.coeffs + alpha * du.coeffs, name="phi")
                    trial_flux = old_state.flux + alpha * flux_du
                    trial_trace = old_state.trace + alpha * trace_du
                    trial_rho_values = window_values(trial_u.values(), c1_phi, c2_phi, eps_phi, params.rho_amp)
                    trial_mass = mass_from_values(space, trial_rho_values)
                    trial_max_rho = float(np.max(trial_rho_values))
                    branch_ok = trial_mass >= mass_floor and trial_max_rho >= params.rho_max_floor
                    if branch_ok:
                        trial_state = build_state(
                            space,
                            trial_u,
                            trial_flux,
                            trial_trace,
                            c1_phi=c1_phi,
                            c2_phi=c2_phi,
                            eps_phi=eps_phi,
                            rho_amp=params.rho_amp,
                            tau=args.hdg_tau,
                            gram_inverse=gram_inverse,
                            gram_rtol=args.gram_cg_rtol,
                            gram_atol=args.gram_cg_atol,
                            gram_maxiter=args.gram_cg_maxiter,
                        )
                        merit = trial_state.residual_hdg
                    else:
                        trial_state = None
                        merit = math.inf
                    armijo = branch_ok and merit <= (1.0 - params.armijo_c * alpha) * old_merit
                    if armijo:
                        best_trial = trial_state
                        accepted = True
                        break
                    alpha *= params.beta_ls
                    n_backtrack += 1

                if not accepted:
                    reject_count += 1
                    mu_shift = min(params.mu_max, 2.0 * mu_shift)
                    metrics = compute_metrics(old_state, design, c2_phi=c2_phi, eps_phi=eps_phi, params=params)
                    write_newton_row(
                        newton_writer,
                        record="NEWTON",
                        runTag=run_tag,
                        ieps=ieps,
                        epsPhiRatio=eps_ratio,
                        epsPhi=eps_phi,
                        k=k,
                        nt=space.mesh.num_tri,
                        ndof=space.ndof,
                        resHdg=old_state.residual_hdg,
                        resHdg2=old_state.residual_hdg_squared,
                        resCoeffL2=old_state.residual_coeff_l2,
                        stepHdg=step_norm,
                        stepFluxL2=step_flux_l2,
                        stepJumpL2=step_jump_l2,
                        alpha=alpha,
                        bt=n_backtrack,
                        muShift=mu_shift,
                        minU=metrics["min_u"],
                        maxU=metrics["max_u"],
                        maxRho=metrics["max_rho"],
                        massRho=metrics["mass_rho"],
                        rhoL2=metrics["rho_l2"],
                        energyPhi=metrics["energy_phi"],
                        activeArea=metrics["active_area"],
                        plateauArea=metrics["plateau_area"],
                        plateauFrac=metrics["plateau_frac"],
                        relRhoDesign=metrics["rel_rho_design"],
                        annularPhiMinusC2=metrics["annular_phi_minus_c2"],
                        solveTime=solve_time,
                        metricTime=0.0,
                        stepTime=time.perf_counter() - step_start,
                        solver=solver_label,
                        status="FAIL_LS",
                    )
                    print(
                        f"STEP ieps={ieps} k={k} resHdg={old_state.residual_hdg:.6e} "
                        f"alpha={alpha:.3e} bt={n_backtrack} mu={mu_shift:.3e} status=FAIL_LS"
                    )
                    if reject_count >= 5 or mu_shift >= params.mu_max:
                        print(f"EPS_STOP ieps={ieps} reason=too_many_failed_steps")
                        break
                    continue

                reject_count = 0
                accepted_this_eps += 1
                state = best_trial
                metrics_start = time.perf_counter()
                metrics = compute_metrics(state, design, c2_phi=c2_phi, eps_phi=eps_phi, params=params)
                metric_time = time.perf_counter() - metrics_start
                write_newton_row(
                    newton_writer,
                    record="NEWTON",
                    runTag=run_tag,
                    ieps=ieps,
                    epsPhiRatio=eps_ratio,
                    epsPhi=eps_phi,
                    k=k,
                    nt=space.mesh.num_tri,
                    ndof=space.ndof,
                    resHdg=state.residual_hdg,
                    resHdg2=state.residual_hdg_squared,
                    resCoeffL2=state.residual_coeff_l2,
                    stepHdg=step_norm,
                    stepFluxL2=step_flux_l2,
                    stepJumpL2=step_jump_l2,
                    alpha=alpha,
                    bt=n_backtrack,
                    muShift=mu_shift,
                    minU=metrics["min_u"],
                    maxU=metrics["max_u"],
                    maxRho=metrics["max_rho"],
                    massRho=metrics["mass_rho"],
                    rhoL2=metrics["rho_l2"],
                    energyPhi=metrics["energy_phi"],
                    activeArea=metrics["active_area"],
                    plateauArea=metrics["plateau_area"],
                    plateauFrac=metrics["plateau_frac"],
                    relRhoDesign=metrics["rel_rho_design"],
                    annularPhiMinusC2=metrics["annular_phi_minus_c2"],
                    solveTime=solve_time,
                    metricTime=metric_time,
                    stepTime=time.perf_counter() - step_start,
                    solver=solver_label,
                    status="ACCEPT",
                )
                print(
                    f"STEP ieps={ieps} k={k} resHdg={state.residual_hdg:.6e} "
                    f"resCoeff={state.residual_coeff_l2:.6e} alpha={alpha:.3e} bt={n_backtrack} "
                    f"mu={mu_shift:.3e} mass={metrics['mass_rho']:.6e} status=ACCEPT"
                )
                if args.plot_newton and args.plot_newton_every > 0 and k % args.plot_newton_every == 0:
                    plotter.emit(
                        [state.u, state.rho],
                        [f"Newton phi ieps={ieps} k={k}", "rho=f(phi)"],
                        stage="ACCEPT",
                        ieps=ieps,
                        k=k,
                        eps_phi=eps_phi,
                        residual=state.residual_hdg,
                        metrics=metrics,
                        token=f"ieps_{ieps}_k_{k}_accept",
                        save=bool(args.save_frames and frame_every is not None and frame_every > 0 and k % frame_every == 0),
                        show=True,
                    )

                if (
                    params.use_adapt
                    and not adapted_this_eps
                    and accepted_this_eps >= params.adapt_after_accepted
                    and (ieps < len(params.eps_phi_ratios) - 1 or params.adapt_after_last_eps)
                ):
                    print(f"ADAPT_SCHEDULE_START ieps={ieps} k={k} accepted={accepted_this_eps}", flush=True)
                    if args.plot_adapt:
                        adapt_indicator = make_adapt_indicator_from_rho(
                            state.rho,
                            eps_phi,
                            params.adapt_grad_weight,
                            name="scheduled_adapt_indicator_plot",
                        )
                        plotter.emit(
                            [state.u, state.rho, adapt_indicator],
                            [f"Before adapt phi ieps={ieps} k={k}", "Before adapt rho", "Adapt indicator"],
                            stage="BEFORE_ADAPT",
                            ieps=ieps,
                            k=k,
                            eps_phi=eps_phi,
                            residual=state.residual_hdg,
                            metrics=metrics,
                            token=f"ieps_{ieps}_k_{k}_before_adapt",
                            save=bool(args.frame_adapt),
                            show=True,
                        )
                    space, state, design, gram_inverse, adapt_info = scheduled_adapt(
                        space,
                        state,
                        design,
                        args,
                        params,
                        eps_phi=eps_phi,
                        c1_phi=c1_phi,
                        c2_phi=c2_phi,
                        initial_hmin=initial_hmin,
                        initial_hmax=initial_hmax,
                        gram_args=gram_args,
                    )
                    adapted_this_eps = True
                    print(
                        f"ADAPT_SCHEDULE_DONE ieps={ieps} k={k} ntNew={space.mesh.num_tri} "
                        f"ndofNew={space.ndof} massRel={adapt_info['mass_rel']:.3e} "
                        f"time={adapt_info['elapsed']:.3f}"
                    )
                    if args.plot_adapt:
                        post_adapt_metrics = compute_metrics(state, design, c2_phi=c2_phi, eps_phi=eps_phi, params=params)
                        plotter.emit(
                            [state.u, state.rho, design.rho_design],
                            [f"After adapt phi ieps={ieps} k={k}", "After adapt rho", "rhoDesign"],
                            stage="AFTER_ADAPT",
                            ieps=ieps,
                            k=k,
                            eps_phi=eps_phi,
                            residual=state.residual_hdg,
                            metrics=post_adapt_metrics,
                            token=f"ieps_{ieps}_k_{k}_after_adapt",
                            save=bool(args.frame_adapt),
                            show=True,
                        )
                    adapt_writer.writerow({
                        "record": "ADAPT",
                        "runTag": run_tag,
                        "ieps": ieps,
                        "k": k,
                        "ntOld": adapt_info["nt_old"],
                        "ndofOld": adapt_info["ndof_old"],
                        "ntNew": adapt_info["nt_new"],
                        "ndofNew": adapt_info["ndof_new"],
                        "eps": eps_phi,
                        "hMin": initial_hmin,
                        "hMax": initial_hmax,
                        "indicatorMin": adapt_info["indicator_min"],
                        "indicatorMax": adapt_info["indicator_max"],
                        "indicatorQlo": adapt_info["indicator_qlo"],
                        "indicatorQhi": adapt_info["indicator_qhi"],
                        "massBefore": adapt_info["mass_before"],
                        "massAfter": adapt_info["mass_after"],
                        "massRel": adapt_info["mass_rel"],
                        "maxRhoRel": adapt_info["max_rho_rel"],
                        "adaptTime": adapt_info["elapsed"],
                        "status": "SCHEDULED",
                    })

                if n_backtrack <= 1:
                    mu_shift = max(params.mu_min, 0.85 * mu_shift)
                elif n_backtrack >= 8:
                    mu_shift = min(params.mu_max, 1.5 * mu_shift)

                rel_drop = abs(old_merit - state.residual_hdg) / max(old_merit, 1.0e-30)
                if rel_drop < params.stagnation_tol:
                    stagnation_count += 1
                    mu_shift = min(params.mu_max, 1.25 * mu_shift)
                    if stagnation_count >= params.max_stagnation:
                        print(f"EPS_STOP ieps={ieps} reason=repeated_stagnation")
                        break
                else:
                    stagnation_count = 0

            metrics = compute_metrics(state, design, c2_phi=c2_phi, eps_phi=eps_phi, params=params)
            write_newton_row(
                newton_writer,
                record="EPS_END",
                runTag=run_tag,
                ieps=ieps,
                epsPhiRatio=eps_ratio,
                epsPhi=eps_phi,
                k="NA",
                nt=space.mesh.num_tri,
                ndof=space.ndof,
                resHdg=state.residual_hdg,
                resHdg2=state.residual_hdg_squared,
                resCoeffL2=state.residual_coeff_l2,
                stepHdg="NA",
                stepFluxL2="NA",
                stepJumpL2="NA",
                alpha="NA",
                bt="NA",
                muShift=mu_shift,
                minU=metrics["min_u"],
                maxU=metrics["max_u"],
                maxRho=metrics["max_rho"],
                massRho=metrics["mass_rho"],
                rhoL2=metrics["rho_l2"],
                energyPhi=metrics["energy_phi"],
                activeArea=metrics["active_area"],
                plateauArea=metrics["plateau_area"],
                plateauFrac=metrics["plateau_frac"],
                relRhoDesign=metrics["rel_rho_design"],
                annularPhiMinusC2=metrics["annular_phi_minus_c2"],
                solveTime="NA",
                metricTime="NA",
                stepTime="NA",
                solver="NA",
                status="EPS_END",
            )
            print(f"EPS_END ieps={ieps} resHdg={state.residual_hdg:.6e} mass={metrics['mass_rho']:.6e}")
            if args.plot_newton and frame_every is not None and frame_every > 0:
                plotter.emit(
                    [state.u, state.rho, design.rho_design],
                    [f"Epsilon end phi ieps={ieps}", "rho=f(phi)", "rhoDesign"],
                    stage="EPS_END",
                    ieps=ieps,
                    k=-1,
                    eps_phi=eps_phi,
                    residual=state.residual_hdg,
                    metrics=metrics,
                    token=f"ieps_{ieps}_eps_end",
                    save=bool(args.save_frames),
                    show=False,
                )

        metrics = compute_metrics(state, design, c2_phi=c2_phi, eps_phi=eps_phi, params=params)
        write_newton_row(
            newton_writer,
            record="FINAL",
            runTag=run_tag,
            ieps="NA",
            epsPhiRatio="NA",
            epsPhi=eps_phi,
            k="NA",
            nt=space.mesh.num_tri,
            ndof=space.ndof,
            resHdg=state.residual_hdg,
            resHdg2=state.residual_hdg_squared,
            resCoeffL2=state.residual_coeff_l2,
            stepHdg="NA",
            stepFluxL2="NA",
            stepJumpL2="NA",
            alpha="NA",
            bt="NA",
            muShift=mu_shift,
            minU=metrics["min_u"],
            maxU=metrics["max_u"],
            maxRho=metrics["max_rho"],
            massRho=metrics["mass_rho"],
            rhoL2=metrics["rho_l2"],
            energyPhi=metrics["energy_phi"],
            activeArea=metrics["active_area"],
            plateauArea=metrics["plateau_area"],
            plateauFrac=metrics["plateau_frac"],
            relRhoDesign=metrics["rel_rho_design"],
            annularPhiMinusC2=metrics["annular_phi_minus_c2"],
            solveTime="NA",
            metricTime="NA",
            stepTime="NA",
            solver="NA",
            status="FINAL",
        )
        if args.plot_final:
            plotter.emit(
                [design.torsion, design.rho_design, design.phi_design, state.u, state.rho],
                ["Torsion T", "rhoDesign", "phiDesign", "Final phi", "Final rho=f(phi)"],
                stage="FINAL",
                ieps=len(params.eps_phi_ratios),
                k=-1,
                eps_phi=eps_phi,
                residual=state.residual_hdg,
                metrics=metrics,
                token="final",
                save=bool(args.frame_final),
                show=True,
            )

    elapsed = time.perf_counter() - total_start
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"runTag {run_tag}\n")
        handle.write(f"nt {space.mesh.num_tri}\n")
        handle.write(f"ndof {space.ndof}\n")
        handle.write(f"order {args.order}\n")
        handle.write(f"resHdg {state.residual_hdg}\n")
        handle.write(f"resCoeffL2 {state.residual_coeff_l2}\n")
        handle.write(f"massRho {metrics['mass_rho']}\n")
        handle.write(f"maxRho {metrics['max_rho']}\n")
        handle.write(f"timeTotal {elapsed}\n")
    print(
        f"FINAL resHdg={state.residual_hdg:.6e} resCoeff={state.residual_coeff_l2:.6e} "
        f"maxPhi={metrics['max_u']:.6e} maxRho={metrics['max_rho']:.6e} "
        f"mass={metrics['mass_rho']:.6e}"
    )
    print(f"SUMMARY {summary_path}")
    print(f"TIME_TOTAL {elapsed:.3f}")
    print("========== END STRATEGY A HDG ADAPTIVE TORSION NEWTON ==========")
    return state


def main() -> State:
    return run_strategy(parse_args())


if __name__ == "__main__":
    main()
