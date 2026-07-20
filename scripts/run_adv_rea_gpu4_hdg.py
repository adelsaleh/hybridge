#!/usr/bin/env python3
"""Standalone CuPy/PyAMGX advection-reaction HDG runner using hdgfem core data.

This script mirrors the fast legacy 2d/adv_rea_vec_gpu4.py path, but sources
mesh, reference-element, and basis data from hdgfem. It is intentionally
independent of scripts/run_adv_rea_cases.py so the GPU assembly/solve path can
be iterated on without disturbing the preset runner.

For advection assembly, ``--assembly-backend raw-cuda --raw-local-assembly
fused`` is the current memory-scaling path.  It supports the default validated
``--raw-lu-mode safe`` and an opt-in experimental ``--raw-lu-mode coop`` mode
that coordinates threads during the fused local LU stage.  The cooperative mode
is restricted to the fused raw path and is documented in
``run_logs/raw_cuda_fused_coop_lu_findings_20260720.md``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hdgfem.core.mesh import gmsh_rectangle_mesh, rectangle_mesh
from hdgfem.core.quadrature import ReferenceElementData
from hdgfem.core.space import DGSpace
from hdgfem.backends.cupy import as_cupy_space, require_cupy, require_cupyx_sparse, require_pyamgx
from hdgfem.backends.cupy_adv_rea_raw import (
    assemble_projected_advection_trace_system_eliminated_raw_cuda,
    assemble_projected_advection_trace_system_eliminated_raw_cuda_fused,
    reconstruct_projected_advection_field_raw_cuda,
    reconstruct_projected_advection_field_raw_cuda_fused,
)
from hdgfem.io.output import pretty_print_sections
from hdgfem.linalg.ordering import upwind_scc_trace_ordering
from scripts.adv_rea_cases import case_definition_by_key


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "amgx"
DEFAULT_AMGX_CONFIG_PATH = CONFIG_DIR / "adv_rea_gpu4_hdg_bicgstab_ilu0_amg.json"


AMGX_CONFIG = {
    "config_version": 2,
    "determinism_flag": 1,
    "exception_handling": 1,
    "solver": {
        "solver": "BICGSTAB",
        "monitor_residual": 1,
        "convergence": "RELATIVE_INI_CORE",
        "tolerance": 1.0e-14,
        "max_iters": 1500,
        "print_solve_stats": 0,
        "obtain_timings": 0,
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "CLASSICAL",
            "selector": "PMIS",
            "cycle": "W",
            "strength_threshold": 0.5,
            "max_iters": 1,
            "max_levels": 20,
            "interp_max_elements": 7,
            "interp_truncation_factor": 0.03,
            "smoother": {
                "solver": "ILU0",
                "max_row_sum": 1.0e5,
            },
            "presweeps": 4,
            "postsweeps": 4,
            "coarsest_sweeps": 2,
            "coarse_solver": "DENSE_LU_SOLVER",
            "print_grid_stats": 0,
        },
    },
}


@dataclass(frozen=True)
class TraceReferenceData:
    kind: str
    nodal: bool
    interpolation_nodes: object
    quads: object
    weights: object
    bas_of_bd_quads: object
    bas1d_of_ref_edg_qds: object
    weighted_bas_of_bd_quads: object
    weighted_bas1d_of_ref_edg_qds: object
    face_trace_test_element_trial_oriented: object
    M_rf_fc: object


def legendre_gauss_lobatto(num_points: int) -> tuple[np.ndarray, np.ndarray]:
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
    nodes = np.asarray(nodes, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    values = np.ones((nodes.size, points.size), dtype=np.float64)
    for i in range(nodes.size):
        for j in range(nodes.size):
            if i != j:
                values[i] *= (points - nodes[j]) / (nodes[i] - nodes[j])
    return np.ascontiguousarray(values)


def bernstein_edge_basis(order: int, points: np.ndarray) -> np.ndarray:
    from math import factorial

    r = 0.5 * (points + 1.0)
    values = np.empty((order + 1, points.size), dtype=np.float64)
    for j in range(order + 1):
        coeff = factorial(order) / (factorial(j) * factorial(order - j))
        values[j] = coeff * (1.0 - r) ** (order - j) * r ** j
    return np.ascontiguousarray(values)



def legendre_edge_basis(order: int, points: np.ndarray) -> np.ndarray:
    values = np.empty((order + 1, points.size), dtype=np.float64)
    for j in range(order + 1):
        values[j] = np.polynomial.legendre.Legendre.basis(j)(points)
    return np.ascontiguousarray(values)


def edge_points(edge_points_1d: np.ndarray) -> np.ndarray:
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
    cp = require_cupy()
    space = cspace.host
    order = cspace.order
    normalized = kind.replace("-", "_").lower()
    if normalized == "modern":
        q = cspace.quad_data
        return TraceReferenceData(
            kind="modern",
            nodal=False,
            interpolation_nodes=q.rf_edg_lag_nodes,
            quads=q.quads_JGL,
            weights=q.weights_JGL,
            bas_of_bd_quads=q.bas_of_bd_quads,
            bas1d_of_ref_edg_qds=q.bas1d_of_ref_edg_qds,
            weighted_bas_of_bd_quads=q.weighted_bas_of_bd_quads,
            weighted_bas1d_of_ref_edg_qds=q.weighted_bas1d_of_ref_edg_qds,
            face_trace_test_element_trial_oriented=q.face_trace_test_element_trial_oriented,
            M_rf_fc=q.M_rf_fc,
        )
    if normalized not in {"legacy_lagrange", "legendre_modal"}:
        raise ValueError("trace basis must be 'legacy-lagrange', 'legendre-modal', or 'modern'")

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
    # Keep the element face orientation fixed and reverse the trace coordinate.
    face_coupling_reversed = np.einsum("q,fiq,jq->fij", edge_weights, face_basis, edge_basis_reversed, optimize=True)
    trace_lift = np.ascontiguousarray(
        np.concatenate((face_coupling.transpose(0, 2, 1), face_coupling_reversed.transpose(0, 2, 1)), axis=0)
    )
    edge_mass = np.einsum("q,iq,jq->ij", edge_weights, edge_basis, edge_basis, optimize=True)
    return TraceReferenceData(
        kind=kind_out,
        nodal=nodal,
        interpolation_nodes=cp.asarray(interpolation_nodes, dtype=cp.float64),
        quads=cp.asarray(edge_quads, dtype=cp.float64),
        weights=cp.asarray(edge_weights, dtype=cp.float64),
        bas_of_bd_quads=cp.asarray(np.ascontiguousarray(face_basis), dtype=cp.float64),
        bas1d_of_ref_edg_qds=cp.asarray(edge_basis, dtype=cp.float64),
        weighted_bas_of_bd_quads=cp.asarray(weighted_face_basis, dtype=cp.float64),
        weighted_bas1d_of_ref_edg_qds=cp.asarray(weighted_edge_basis, dtype=cp.float64),
        face_trace_test_element_trial_oriented=cp.asarray(trace_lift, dtype=cp.float64),
        M_rf_fc=cp.asarray(np.ascontiguousarray(edge_mass), dtype=cp.float64),
    )


TIMINGS: dict[str, float] = {}


def record_timing(key: str, seconds: float) -> float:
    seconds = float(seconds)
    TIMINGS[key] = TIMINGS.get(key, 0.0) + seconds
    return seconds


def timing_value(key: str) -> float:
    return TIMINGS.get(key, 0.0)


def timing_percent(seconds: float, total_seconds: float, digits: int = 1) -> str:
    percent = 100.0 * float(seconds) / float(total_seconds) if total_seconds > 0.0 else 0.0
    return f"{float(seconds):.{digits}f} ({percent:.1f}%)"


def compact_config_name(value: object) -> str:
    if value is None:
        return "embedded default"
    text = str(value)
    return Path(text).name if text else "embedded default"


def maybe_restore_stdout(original_stdout, quiet_handle, stdout_fd_copy=None) -> None:
    if stdout_fd_copy is not None:
        try:
            import ctypes

            ctypes.CDLL(None).fflush(None)
        except Exception:
            pass
        os.dup2(stdout_fd_copy, 1)
        os.close(stdout_fd_copy)
    if original_stdout is not None:
        sys.stdout = original_stdout
    if quiet_handle is not None:
        quiet_handle.close()


def pretty_print(items, title: str = "Results", pad_lines: int = 1, default_fmt: str = ".5g") -> None:
    formatted = []
    for label, value, fmt in items:
        if isinstance(value, str):
            rendered = value
        else:
            rendered = format(value, fmt or default_fmt)
        formatted.append((label, rendered))
    label_width = max(len(label) for label, _ in formatted)
    value_width = max(len(value) for _, value in formatted)
    print("\n" * pad_lines, end="")
    print(title)
    print("=" * (label_width + value_width + 3))
    for label, value in formatted:
        print(f"{label:<{label_width}} : {value:>{value_width}}")
    print("=" * (label_width + value_width + 3))


def sync_time(cp, start: float, key: str) -> float:
    cp.cuda.get_current_stream().synchronize()
    return record_timing(key, time.perf_counter() - start)


def print_done(seconds: float) -> None:
    print(f"done in {seconds:.5f}s", flush=True)


def mapped_quads_cupy(cspace):
    cp = require_cupy()
    mesh = cspace.mesh
    q = cspace.quad_data
    return cp.einsum("Krc,qc->Krq", mesh.aff_mats, q.Krf_quads) + mesh.aff_vecs[:, :, None]


def project_callable_cupy(func: Callable, cspace, label: str, timing_key: str):
    cp = require_cupy()
    print(label, end="", flush=True)
    start = time.perf_counter()
    q = cspace.quad_data
    points = mapped_quads_cupy(cspace)
    values = cp.asarray(func(points[:, 0, :], points[:, 1, :]), dtype=cp.float64)
    rhs_t = cp.einsum("Kq,iq,q->iK", values, q.bas_of_quads, q.Krf_w)
    coeffs = cp.linalg.solve(q.MKrf, rhs_t).T
    elapsed = sync_time(cp, start, timing_key)
    print_done(elapsed)
    return cp.ascontiguousarray(coeffs)




def source_coefficients_cupy(source: Callable, cspace):
    return project_callable_cupy(source, cspace, "  source projection ... ", "projection.source")


def reaction_coefficients_cupy(reaction, cspace):
    cp = require_cupy()
    if isinstance(reaction, (int, float, np.integer, np.floating)):
        return cp.empty(1, dtype=cp.float64), float(reaction), True
    coeffs = project_callable_cupy(reaction, cspace, "  reaction projection ... ", "projection.reaction")
    return coeffs, 0.0, False


def reference_advection_tensor_cupy(cspace):
    cp = require_cupy()
    print("  reference advection tensor ... ", end="", flush=True)
    start = time.perf_counter()
    q = cspace.host.quad_data
    tensor = np.einsum(
        "q,qk,qj,qiD->Dkij",
        q.Krf_w,
        q.phi,
        q.phi,
        q.gphi,
        optimize=True,
    )
    result = cp.asarray(np.ascontiguousarray(tensor, dtype=np.float64))
    elapsed = sync_time(cp, start, "assembly.reference_advection_tensor")
    print_done(elapsed)
    return result


def source_moments_cupy(source: Callable, cspace):
    cp = require_cupy()
    print("  source moments ... ", end="", flush=True)
    start = time.perf_counter()
    mesh = cspace.mesh
    q = cspace.quad_data
    points = mapped_quads_cupy(cspace)
    values = cp.asarray(source(points[:, 0, :], points[:, 1, :]), dtype=cp.float64)
    rhs = mesh.aff_jacs[:, None] * cp.einsum("Kq,iq,q->Ki", values, q.bas_of_quads, q.Krf_w)
    elapsed = sync_time(cp, start, "source_moments")
    print_done(elapsed)
    return cp.ascontiguousarray(rhs)


def beta_dot_normal_from_coeffs(beta_coeffs, cspace, trace_ref):
    cp = require_cupy()
    mesh = cspace.mesh
    q = cspace.quad_data
    return cp.einsum(
        "dKi,Kfd,fiq->Kfq",
        beta_coeffs,
        mesh.normals,
        trace_ref.bas_of_bd_quads,
        optimize=True,
    )


def reaction_mass_cupy(reaction: Callable, cspace):
    cp = require_cupy()
    print("  reaction mass ... ", end="", flush=True)
    start = time.perf_counter()
    mesh = cspace.mesh
    q = cspace.quad_data
    points = mapped_quads_cupy(cspace)
    values = cp.asarray(reaction(points[:, 0, :], points[:, 1, :]), dtype=cp.float64)
    scaled = values * mesh.aff_jacs[:, None]
    flat = scaled @ q.weighted_phi_phi_flat
    result = flat.reshape(mesh.num_tri, cspace.el_dof, cspace.el_dof)
    elapsed = sync_time(cp, start, "local.reaction_mass")
    print_done(elapsed)
    return result


def boundary_mass_cupy(beta_dot_normal, cspace, trace_ref):
    cp = require_cupy()
    print("  boundary mass ... ", end="", flush=True)
    start = time.perf_counter()
    mesh = cspace.mesh
    q = cspace.quad_data
    result = cp.einsum(
        "Kf,Kfq,fiq,fjq->Kij",
        mesh.jacs_el_fc,
        cp.abs(beta_dot_normal),
        trace_ref.bas_of_bd_quads,
        trace_ref.weighted_bas_of_bd_quads,
        optimize=True,
    )
    elapsed = sync_time(cp, start, "local.boundary_mass")
    print_done(elapsed)
    return result


def advection_mats_cupy(beta_coeffs, cspace):
    cp = require_cupy()
    print("  advection mats ... ", end="", flush=True)
    start = time.perf_counter()
    mesh = cspace.mesh
    q = cspace.quad_data
    beta_values = cp.einsum("dKi,iq->Kqd", beta_coeffs, q.bas_of_quads, optimize=True)
    scaled_inv_t = mesh.aff_jacs[:, None, None] * mesh.inv_aff_mats_t
    result = cp.einsum(
        "Kqd,KdD,jq,Diq,q->Kij",
        beta_values,
        scaled_inv_t,
        q.bas_of_quads,
        q.dbas_of_quads,
        q.Krf_w,
        optimize=True,
    )
    elapsed = sync_time(cp, start, "local.advection")
    print_done(elapsed)
    return result


def element_boundary_mats_cupy(beta_dot_normal, cspace, trace_ref):
    cp = require_cupy()
    print("  element boundary mats ... ", end="", flush=True)
    start = time.perf_counter()
    mesh = cspace.mesh
    q = cspace.quad_data
    flux_weight = cp.abs(beta_dot_normal) - beta_dot_normal
    result = cp.einsum(
        "Kf,Kfq,fiq,jq->Kifj",
        mesh.jacs_el_fc,
        flux_weight,
        trace_ref.bas_of_bd_quads,
        trace_ref.weighted_bas1d_of_ref_edg_qds,
        optimize=True,
    ).reshape(mesh.num_tri, cspace.el_dof, 3 * cspace.edg_dof)
    elapsed = sync_time(cp, start, "local.element_boundary")
    print_done(elapsed)
    return cp.ascontiguousarray(result)


def local_mats_cupy(reaction, beta_coeffs, beta_dot_normal, cspace, trace_ref):
    return reaction_mass_cupy(reaction, cspace) + boundary_mass_cupy(beta_dot_normal, cspace, trace_ref) - advection_mats_cupy(beta_coeffs, cspace)


def solve_local_mats(local_mats, rhs, label: str, key: str):
    cp = require_cupy()
    print(label, end="", flush=True)
    start = time.perf_counter()
    result = cp.linalg.solve(local_mats, rhs)
    elapsed = sync_time(cp, start, key)
    print_done(elapsed)
    return cp.ascontiguousarray(result)


def setup_reduced_indices(cspace):
    cp = require_cupy()
    print("  reduced COO indices ... ", end="", flush=True)
    start = time.perf_counter()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    n_flux = valid_elements.size * 3 * edg_dof * edg_dof
    n_mass = mesh.int_edges_inds.size * edg_dof * edg_dof
    rows = cp.empty(n_flux + n_mass, dtype=cp.int64)
    cols = cp.empty_like(rows)
    i_grid, j_grid = cp.meshgrid(cp.arange(edg_dof, dtype=cp.int64), cp.arange(edg_dof, dtype=cp.int64), indexing="ij")
    row_edges = mesh.loc2glob_edge[valid_elements, valid_faces]
    col_edges = mesh.loc2glob_edge[valid_elements]
    rows[:n_flux] = cp.broadcast_to(
        row_edges[:, None, None, None] * edg_dof + i_grid[None, None, :, :],
        (valid_elements.size, 3, edg_dof, edg_dof),
    ).ravel()
    cols[:n_flux] = (col_edges[:, :, None, None] * edg_dof + j_grid[None, None, :, :]).ravel()
    local = cp.arange(edg_dof, dtype=cp.int64)
    l0 = cp.broadcast_to(local[:, None], (edg_dof, edg_dof)).ravel()
    l1 = cp.broadcast_to(local[None, :], (edg_dof, edg_dof)).ravel()
    rows[n_flux:] = (mesh.int_edges_inds[:, None] * edg_dof + l0).ravel()
    cols[n_flux:] = (mesh.int_edges_inds[:, None] * edg_dof + l1).ravel()
    elapsed = sync_time(cp, start, "assembly.indices")
    print_done(elapsed)
    return rows, cols


def trace_blocks_cupy(solved_el_bd_mats, cspace, trace_ref):
    cp = require_cupy()
    print("  trace blocks ... ", end="", flush=True)
    start = time.perf_counter()
    mesh = cspace.mesh
    q = cspace.quad_data
    edg_dof = cspace.edg_dof
    trace_lift = (
        mesh.jacs_el_fc[..., None, None]
        / 2.0
        * trace_ref.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling]
    )
    blocks = trace_lift @ solved_el_bd_mats[:, None, :, :]
    blocks = blocks.reshape(mesh.num_tri, 3, edg_dof, 3, edg_dof)
    if mesh.num_negative_orientations:
        neg = blocks[mesh.negative_orientation_elements, :, :, mesh.negative_orientation_faces, :]
        if trace_ref.kind == "legendre-modal":
            signs = cp.where(cp.arange(edg_dof, dtype=cp.int64) % 2 == 0, 1.0, -1.0)
            blocks[mesh.negative_orientation_elements, :, :, mesh.negative_orientation_faces, :] = neg * signs
        else:
            blocks[mesh.negative_orientation_elements, :, :, mesh.negative_orientation_faces, :] = neg[..., ::-1]
    result = cp.ascontiguousarray(blocks.swapaxes(2, 3))
    elapsed = sync_time(cp, start, "assembly.trace_blocks")
    print_done(elapsed)
    return result


def trace_data_cupy(trace_blocks, cspace, trace_ref):
    cp = require_cupy()
    print("  COO data ... ", end="", flush=True)
    start = time.perf_counter()
    mesh = cspace.mesh
    q = cspace.quad_data
    edg_dof = cspace.edg_dof
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    n_flux = valid_elements.size * 3 * edg_dof * edg_dof
    n_mass = mesh.int_edges_inds.size * edg_dof * edg_dof
    data = cp.empty(n_flux + n_mass, dtype=cp.float64)
    data[:n_flux] = -trace_blocks[valid_elements, valid_faces].ravel()
    data[n_flux:] = (mesh.edge_jacs[mesh.int_edges_inds, None, None] * trace_ref.M_rf_fc[None]).ravel()
    elapsed = sync_time(cp, start, "assembly.data")
    print_done(elapsed)
    return data


def face_rhs_cupy(solved_src, cspace, trace_ref):
    cp = require_cupy()
    print("  face RHS ... ", end="", flush=True)
    start = time.perf_counter()
    mesh = cspace.mesh
    q = cspace.quad_data
    trace_lift = (
        mesh.jacs_el_fc[..., None, None]
        / 2.0
        * trace_ref.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling]
    )
    result = (trace_lift @ solved_src[:, None, :, :]).squeeze(-1)
    elapsed = sync_time(cp, start, "assembly.rhs_faces")
    print_done(elapsed)
    return result


def boundary_trace_values_cupy(exact: Callable, cspace, trace_ref):
    cp = require_cupy()
    mesh = cspace.mesh
    q = cspace.quad_data
    if mesh.bnd_edges_inds.size == 0:
        return cp.empty((0, cspace.edg_dof), dtype=cp.float64)
    edge_coords = mesh.node_coords[mesh.edges[mesh.bnd_edges_inds]]
    t = trace_ref.interpolation_nodes if trace_ref.nodal else trace_ref.quads
    points = 0.5 * ((1.0 - t)[None, :, None] * edge_coords[:, 0:1, :] + (1.0 + t)[None, :, None] * edge_coords[:, 1:2, :])
    values = cp.asarray(exact(points[..., 0], points[..., 1]), dtype=cp.float64)
    if trace_ref.nodal:
        return cp.ascontiguousarray(values)
    rhs = (values * trace_ref.weights[None, :]) @ trace_ref.bas1d_of_ref_edg_qds.T
    return cp.linalg.solve(trace_ref.M_rf_fc, rhs.T).T


def build_dof_maps(cspace):
    cp = require_cupy()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    edge_to_boundary_slot = cp.full(mesh.num_edg, -1, dtype=cp.int64)
    edge_to_boundary_slot[mesh.bnd_edges_inds] = cp.arange(mesh.bnd_edges_inds.size, dtype=cp.int64)
    full_to_reduced = cp.full(mesh.num_edg * edg_dof, -1, dtype=cp.int64)
    local = cp.arange(edg_dof, dtype=cp.int64)
    full_to_reduced[(mesh.int_edges_inds[:, None] * edg_dof + local[None, :]).ravel()] = (
        cp.arange(mesh.int_edges_inds.size, dtype=cp.int64)[:, None] * edg_dof + local[None, :]
    ).ravel()
    return edge_to_boundary_slot, full_to_reduced


def eliminate_boundary_cupy(rows, cols, data, rhs, boundary_trace, maps, cspace):
    cp = require_cupy()
    print("  boundary elimination ... ", end="", flush=True)
    start = time.perf_counter()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    edge_to_boundary_slot, full_to_reduced = maps
    row_r = rows.reshape((rows.size // edg_dof, edg_dof))
    col_r = cols.reshape((cols.size // edg_dof, edg_dof))
    data_r = data.reshape((data.size // edg_dof, edg_dof))
    col_edges = col_r[:, 0] // edg_dof
    boundary_slots = edge_to_boundary_slot[col_edges]
    keep = cp.where(boundary_slots < 0)[0]
    remove = cp.where(boundary_slots >= 0)[0]
    keep_count = int(keep.size)
    reduced_rows = cp.empty(keep_count * edg_dof, dtype=cp.int64)
    reduced_cols = cp.empty_like(reduced_rows)
    reduced_data = cp.empty(keep_count * edg_dof, dtype=cp.float64)
    reduced_rows.reshape((keep_count, edg_dof))[:] = full_to_reduced[row_r[keep]]
    reduced_cols.reshape((keep_count, edg_dof))[:] = full_to_reduced[col_r[keep]]
    reduced_data.reshape((keep_count, edg_dof))[:] = data_r[keep]
    if remove.size:
        row_ids = row_r[remove, 0]
        slots = boundary_slots[remove]
        cp.add.at(rhs, row_ids, cp.sum(-data_r[remove] * boundary_trace[slots], axis=1))
    reduced_rhs = rhs.reshape((mesh.num_edg, edg_dof))[mesh.int_edges_inds].ravel()
    elapsed = sync_time(cp, start, "assembly.boundary_elimination")
    print_done(elapsed)
    return reduced_rows, reduced_cols, reduced_data, reduced_rhs


def assemble_reduced_system(
        source,
        reaction,
        exact,
        beta_coeffs,
        beta_dot_normal,
        maps,
        cspace,
        trace_ref,
        backend: str = "cupy",
        raw_block_size: int = 1,
        raw_local_assembly: str = "precomputed",
        raw_lu_mode: str = "safe",
):
    cp = require_cupy()
    if backend == "raw-cuda":
        return assemble_reduced_system_raw_cuda(
            source,
            reaction,
            exact,
            beta_coeffs,
            beta_dot_normal,
            cspace,
            trace_ref,
            raw_block_size=raw_block_size,
            raw_local_assembly=raw_local_assembly,
            raw_lu_mode=raw_lu_mode,
        )
    if beta_dot_normal is None:
        raise ValueError("beta_dot_normal is required for CuPy assembly")
    print("assembling reduced trace system (hdgfem/cupy gpu4-style) ...", flush=True)
    start_total = time.perf_counter()
    rows, cols = setup_reduced_indices(cspace)
    local_mats = local_mats_cupy(reaction, beta_coeffs, beta_dot_normal, cspace, trace_ref)
    element_boundary = element_boundary_mats_cupy(beta_dot_normal, cspace, trace_ref)
    source_rhs = source_moments_cupy(source, cspace)
    local_rhs = cp.concatenate((element_boundary, source_rhs[..., None]), axis=2)
    solved = solve_local_mats(local_mats, local_rhs, "  local solve trace/source ... ", "local.solve.assembly")
    solved_el_bd = solved[:, :, : 3 * cspace.edg_dof]
    solved_src = solved[:, :, 3 * cspace.edg_dof :]
    blocks = trace_blocks_cupy(solved_el_bd, cspace, trace_ref)
    data = trace_data_cupy(blocks, cspace, trace_ref)
    rhs_full = cp.zeros(cspace.mesh.num_edg * cspace.edg_dof, dtype=cp.float64)
    rhs_full_r = rhs_full.reshape((cspace.mesh.num_edg, cspace.edg_dof))
    faces = face_rhs_cupy(solved_src, cspace, trace_ref)
    cp.add.at(
        rhs_full_r,
        cspace.mesh.loc2glob_edge[cspace.mesh.interior_elements, cspace.mesh.interior_faces],
        faces[cspace.mesh.interior_elements, cspace.mesh.interior_faces],
    )
    boundary_trace = boundary_trace_values_cupy(exact, cspace, trace_ref)
    rows, cols, data, rhs = eliminate_boundary_cupy(rows, cols, data, rhs_full, boundary_trace, maps, cspace)
    total = sync_time(cp, start_total, "assembly.total")
    print(f"assembly completed in {total:.5f}s", flush=True)
    return rows, cols, data, rhs, local_mats, element_boundary, source_rhs, boundary_trace, None


def assemble_reduced_system_raw_cuda(
        source,
        reaction,
        exact,
        beta_coeffs,
        beta_dot_normal,
        cspace,
        trace_ref,
        raw_block_size: int = 1,
        raw_local_assembly: str = "precomputed",
        raw_lu_mode: str = "safe",
):
    cp = require_cupy()
    if raw_local_assembly not in {"precomputed", "fused"}:
        raise ValueError("raw_local_assembly must be 'precomputed' or 'fused'")
    if raw_lu_mode not in {"safe", "coop"}:
        raise ValueError("raw_lu_mode must be 'safe' or 'coop'")
    if raw_lu_mode != "safe" and raw_local_assembly != "fused":
        raise ValueError("raw_lu_mode='coop' is only supported with raw_local_assembly='fused'")
    print(
        f"assembling reduced trace system (raw CUDA {raw_local_assembly} advection gpu4-style) ...",
        flush=True,
    )
    start_total = time.perf_counter()
    if raw_local_assembly == "fused":
        source_coeffs = source_coefficients_cupy(source, cspace)
        reaction_coeffs, reaction_scalar, reaction_is_scalar = reaction_coefficients_cupy(reaction, cspace)
        advection_tensor = reference_advection_tensor_cupy(cspace)
        boundary_trace = boundary_trace_values_cupy(exact, cspace, trace_ref)
        raw = assemble_projected_advection_trace_system_eliminated_raw_cuda_fused(
            source_coeffs=source_coeffs,
            beta_coeffs=beta_coeffs,
            reaction_coeffs=reaction_coeffs,
            reaction_scalar=reaction_scalar,
            reaction_is_scalar=reaction_is_scalar,
            boundary_trace=boundary_trace,
            cspace=cspace,
            trace_ref=trace_ref,
            advection_tensor=advection_tensor,
            block_size=raw_block_size,
            lu_mode=raw_lu_mode,
        )
    else:
        if beta_dot_normal is None:
            raise ValueError("beta_dot_normal is required for precomputed raw CUDA assembly")
        local_mats = local_mats_cupy(reaction, beta_coeffs, beta_dot_normal, cspace, trace_ref)
        element_boundary = element_boundary_mats_cupy(beta_dot_normal, cspace, trace_ref)
        source_rhs = source_moments_cupy(source, cspace)
        boundary_trace = boundary_trace_values_cupy(exact, cspace, trace_ref)
        raw = assemble_projected_advection_trace_system_eliminated_raw_cuda(
            local_mats=local_mats,
            element_boundary=element_boundary,
            source_rhs=source_rhs,
            boundary_trace=boundary_trace,
            cspace=cspace,
            trace_ref=trace_ref,
            block_size=raw_block_size,
        )
    for key, value in raw.timings.items():
        record_timing(f"assembly.{key}", value)
    total = sync_time(cp, start_total, "assembly.total")
    print(f"  raw CUDA local assembly: {raw_local_assembly}", flush=True)
    print(f"  raw CUDA LU mode: {raw.lu_mode}", flush=True)
    print(f"  raw CUDA block size: {int(raw.timings.get('raw.block_size', float(raw_block_size)))}", flush=True)
    print(f"  raw CUDA map/setup: {raw.timings.get('raw.map_setup', 0.0):.5f}s", flush=True)
    print(f"  raw CUDA solve/emit kernel: {raw.timings.get('raw.kernel', 0.0):.5f}s", flush=True)
    print(f"assembly completed in {total:.5f}s", flush=True)
    return raw.rows, raw.cols, raw.data, raw.rhs, raw.local_mats, raw.element_boundary, raw.source_rhs, raw.boundary_trace, raw


def load_amgx_config(args):
    config_path = Path(args.amgx_config).expanduser() if args.amgx_config else DEFAULT_AMGX_CONFIG_PATH
    if config_path.exists():
        with config_path.open() as handle:
            config = json.load(handle)
        return config, config_path
    if args.amgx_config:
        raise FileNotFoundError(f"AMGX config file not found: {config_path}")
    return copy.deepcopy(AMGX_CONFIG), None

def solve_amgx(rows, cols, data, rhs, args, permutation=None):
    cp = require_cupy()
    cpsp = require_cupyx_sparse()
    pyamgx = require_pyamgx()
    print("solving global system with PyAMGX ...", flush=True)
    solve_total_start = time.perf_counter()
    n = int(rhs.size)
    solve_info = {
        "solver": str(args.amgx_solver),
        "tolerance": float(args.amgx_tolerance),
        "maxiter": int(args.amgx_maxiter),
        "global_dof": n,
        "permuted": permutation is not None,
    }
    if permutation is not None:
        perm_start = time.perf_counter()
        perm = cp.asarray(permutation, dtype=cp.int64)
        inverse = cp.empty_like(perm)
        inverse[perm] = cp.arange(perm.size, dtype=cp.int64)
        rows = inverse[rows]
        cols = inverse[cols]
        rhs = rhs[perm]
        perm_elapsed = sync_time(cp, perm_start, "solve.permutation")
        print(f"  symmetric upwind permutation: {perm_elapsed:.5f}s", flush=True)
    else:
        perm = None
    scale_start = time.perf_counter()
    matrix = cpsp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=cp.float64).tocsr()
    matrix.sum_duplicates()
    diagonal = matrix.diagonal()
    diagonal[diagonal == 0.0] = 1.0
    d_inv = cpsp.diags(1.0 / diagonal, format="csr", dtype=cp.float64)
    matrix = d_inv @ matrix
    rhs = d_inv @ rhs
    scale_elapsed = sync_time(cp, scale_start, "solve.csr_scale")
    solve_info["nnz"] = int(matrix.nnz)
    print(f"  CSR assembly + diagonal scaling: {scale_elapsed:.5f}s, nnz={matrix.nnz:,}", flush=True)

    config, config_path = load_amgx_config(args)
    solve_info["amgx_config"] = str(config_path) if config_path is not None else "embedded default"
    preconditioner = config.get("solver", {}).get("preconditioner", {})
    smoother = preconditioner.get("smoother", {}) if isinstance(preconditioner, dict) else {}
    preconditioner_name = str(preconditioner.get("solver", "none")) if isinstance(preconditioner, dict) else "none"
    smoother_name = str(smoother.get("solver", "")) if isinstance(smoother, dict) else ""
    solve_info["preconditioner"] = f"{preconditioner_name}/{smoother_name}" if smoother_name else preconditioner_name
    print(f"  AMGX config: {config_path if config_path is not None else 'embedded default'}", flush=True)
    amgx_monitor_requested = os.environ.get("HDGFEM_GPU4_AMGX_MONITOR", "0") == "1"
    verbose_amgx = int(amgx_monitor_requested)
    suppress_amgx_stdout = not amgx_monitor_requested
    config["solver"]["solver"] = str(args.amgx_solver)
    config["solver"]["tolerance"] = float(args.amgx_tolerance)
    config["solver"]["max_iters"] = int(args.amgx_maxiter)
    config["solver"]["print_solve_stats"] = verbose_amgx
    config["solver"]["obtain_timings"] = verbose_amgx
    config["solver"]["preconditioner"]["print_grid_stats"] = verbose_amgx

    amgx_stdout_fd = amgx_stdout_handle = None
    if suppress_amgx_stdout:
        amgx_stdout_handle = open(os.devnull, "w")
        amgx_stdout_fd = os.dup(1)
        os.dup2(amgx_stdout_handle.fileno(), 1)
    try:
        pyamgx.initialize()
    finally:
        maybe_restore_stdout(None, amgx_stdout_handle, amgx_stdout_fd)
    cfg = rsrc = mat = vec_b = vec_x = solver = None
    try:
        setup_start = time.perf_counter()
        cfg = pyamgx.Config().create_from_dict(config)
        rsrc = pyamgx.Resources().create_simple(cfg)
        mat = pyamgx.Matrix().create(rsrc, mode="dDDI")
        vec_b = pyamgx.Vector().create(rsrc, mode="dDDI")
        vec_x = pyamgx.Vector().create(rsrc, mode="dDDI")
        mat.upload_CSR(matrix)
        vec_b.upload_raw(rhs.data.ptr, rhs.size)
        trace = cp.zeros(rhs.size, dtype=cp.float64)
        vec_x.upload_raw(trace.data.ptr, trace.size)
        solver = pyamgx.Solver().create(rsrc, cfg)
        solver.setup(mat)
        setup_elapsed = sync_time(cp, setup_start, "solve.amgx_setup")
        print(f"  PyAMGX setup/upload: {setup_elapsed:.5f}s", flush=True)

        amgx_start = time.perf_counter()
        solver.solve(vec_b, vec_x)
        solve_elapsed = sync_time(cp, amgx_start, "solve.amgx_solve")
        try:
            solve_info["amgx_status"] = str(solver.status)
        except Exception:
            solve_info["amgx_status"] = "unknown"
        try:
            solve_info["amgx_iterations"] = int(solver.iterations_number)
        except Exception:
            solve_info["amgx_iterations"] = -1
        vec_x.download_raw(trace.data.ptr)
        cp.cuda.get_current_stream().synchronize()
        residual = cp.linalg.norm(matrix @ trace - rhs)
        rhs_norm = cp.linalg.norm(rhs)
        residual_value = float(residual.get())
        rhs_norm_value = float(rhs_norm.get())
        rel_residual = residual_value / rhs_norm_value if rhs_norm_value else float("nan")
        solve_info["scaled_abs_res"] = residual_value
        solve_info["scaled_rhs_norm"] = rhs_norm_value
        solve_info["scaled_rel_res"] = rel_residual
        print(f"  PyAMGX solve: {solve_elapsed:.5f}s, scaled_rel_res={rel_residual:.3e}", flush=True)
        if perm is not None:
            unpermuted = cp.empty_like(trace)
            unpermuted[perm] = trace
            trace = unpermuted
    finally:
        for obj in (solver, mat, vec_x, vec_b, rsrc, cfg):
            if obj is not None:
                try:
                    obj.destroy()
                except AttributeError:
                    pass
        amgx_stdout_fd = amgx_stdout_handle = None
        if suppress_amgx_stdout:
            amgx_stdout_handle = open(os.devnull, "w")
            amgx_stdout_fd = os.dup(1)
            os.dup2(amgx_stdout_handle.fileno(), 1)
        try:
            pyamgx.finalize()
        finally:
            maybe_restore_stdout(None, amgx_stdout_handle, amgx_stdout_fd)
    solve_info["total"] = record_timing("solve.total", time.perf_counter() - solve_total_start)
    return trace, solve_info

def reconstruct_trace(trace_reduced, boundary_trace, cspace):
    cp = require_cupy()
    print("augmenting traces ... ", end="", flush=True)
    start = time.perf_counter()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    trace = cp.empty(mesh.num_edg * edg_dof, dtype=cp.float64)
    trace_r = trace.reshape((mesh.num_edg, edg_dof))
    trace_r[mesh.int_edges_inds] = trace_reduced.reshape((mesh.int_edges_inds.size, edg_dof))
    trace_r[mesh.bnd_edges_inds] = boundary_trace
    elapsed = sync_time(cp, start, "reconstruct.trace")
    print_done(elapsed)
    return trace


def reconstruct_field(trace, source, reaction, beta_coeffs, beta_dot_normal, cspace, trace_ref):
    cp = require_cupy()
    print("reconstructing element field ...", flush=True)
    start = time.perf_counter()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    trace_r = trace.reshape((mesh.num_edg, edg_dof))
    element_traces = trace_r[mesh.loc2glob_edge, :]
    if mesh.num_negative_orientations:
        neg = element_traces[mesh.negative_orientation_elements, mesh.negative_orientation_faces]
        if trace_ref.kind == "legendre-modal":
            signs = cp.where(cp.arange(edg_dof, dtype=cp.int64) % 2 == 0, 1.0, -1.0)
            element_traces[mesh.negative_orientation_elements, mesh.negative_orientation_faces] = neg * signs
        else:
            element_traces[mesh.negative_orientation_elements, mesh.negative_orientation_faces] = neg[:, ::-1]
    element_traces = element_traces.reshape((mesh.num_tri, 3 * edg_dof))
    local_mats = local_mats_cupy(reaction, beta_coeffs, beta_dot_normal, cspace, trace_ref)
    element_boundary = element_boundary_mats_cupy(beta_dot_normal, cspace, trace_ref)
    source_rhs = source_moments_cupy(source, cspace)
    rhs = source_rhs[..., None] + element_boundary @ element_traces[..., None]
    uh = solve_local_mats(local_mats, rhs, "  local solve reconstruction ... ", "local.solve.reconstruction").squeeze(-1)
    elapsed = sync_time(cp, start, "reconstruct.field")
    print_done(elapsed)
    return cp.ascontiguousarray(uh)


def reconstruct_field_raw_cuda(trace, raw_assembly, cspace, trace_ref):
    cp = require_cupy()
    print("reconstructing element field (raw CUDA) ... ", end="", flush=True)
    start = time.perf_counter()
    if raw_assembly.local_mats is None:
        block_size = int(raw_assembly.timings.get("raw.block_size", 32.0))
        uh, kernel_elapsed = reconstruct_projected_advection_field_raw_cuda_fused(
            trace=trace,
            source_coeffs=raw_assembly.source_coeffs,
            beta_coeffs=raw_assembly.beta_coeffs,
            reaction_coeffs=raw_assembly.reaction_coeffs,
            reaction_scalar=raw_assembly.reaction_scalar,
            reaction_is_scalar=raw_assembly.reaction_is_scalar,
            cspace=cspace,
            trace_ref=trace_ref,
            advection_tensor=raw_assembly.advection_tensor,
            block_size=block_size,
            lu_mode=raw_assembly.lu_mode,
        )
    else:
        uh, kernel_elapsed = reconstruct_projected_advection_field_raw_cuda(
            trace=trace,
            local_mats=raw_assembly.local_mats,
            element_boundary=raw_assembly.element_boundary,
            source_rhs=raw_assembly.source_rhs,
            cspace=cspace,
            trace_ref=trace_ref,
        )
    record_timing("local.solve.reconstruction", kernel_elapsed)
    elapsed = sync_time(cp, start, "reconstruct.field")
    print_done(elapsed)
    return uh


def evaluate_errors(uh, exact: Callable, cspace, plot_resolution: int, error_volume_quad_1d: int | None = None):
    cp = require_cupy()
    print("computing plot/error data ... ", end="", flush=True)
    start = time.perf_counter()
    mesh = cspace.mesh
    q = cspace.quad_data
    if error_volume_quad_1d is None:
        err_points = q.Krf_quads
        err_weights = q.Krf_w
        err_basis = q.bas_of_quads
    else:
        err_ref = ReferenceElementData.triangle(
            cspace.order,
            basis_type=cspace.host.quad_data.basis_type,
            volume_quad_1d=int(error_volume_quad_1d),
            edge_quad_1d=cspace.host.quad_data.edge_quad_1d,
        )
        err_points = cp.asarray(err_ref.Krf_quads, dtype=cp.float64)
        err_weights = cp.asarray(err_ref.Krf_w, dtype=cp.float64)
        err_basis = cp.asarray(err_ref.bas_of_quads, dtype=cp.float64)
    mapped = cp.einsum("Krc,qc->Krq", mesh.aff_mats, err_points) + mesh.aff_vecs[:, :, None]
    exact_q = cp.asarray(exact(mapped[:, 0, :], mapped[:, 1, :]), dtype=cp.float64)
    uh_q = uh @ err_basis
    l2 = cp.sqrt(cp.einsum("K,Kq,q->", mesh.aff_jacs, (uh_q - exact_q) ** 2, err_weights, optimize=True))

    grid = cp.linspace(-1.0, 1.0, int(plot_resolution), endpoint=True, dtype=cp.float64)
    xx, yy = cp.meshgrid(grid, grid)
    cond = yy <= -xx
    ref_points = cp.column_stack((xx[cond], yy[cond]))
    basis_plot = cp.asarray(cspace.host.basis_at(cp.asnumpy(ref_points)), dtype=cp.float64)
    mapped_plot = cp.einsum("Krc,pc->Krp", mesh.aff_mats, ref_points) + mesh.aff_vecs[:, :, None]
    exact_plot = cp.asarray(exact(mapped_plot[:, 0, :], mapped_plot[:, 1, :]), dtype=cp.float64)
    uh_plot = uh @ basis_plot.T
    abs_err = cp.abs(uh_plot - exact_plot)
    linf = cp.max(abs_err)
    avg_max = cp.average(cp.max(abs_err, axis=-1))
    max_element = cp.argmax(cp.max(abs_err, axis=-1))
    elapsed = sync_time(cp, start, "plot_error")
    print_done(elapsed)
    return float(l2.get()), float(linf.get()), float(avg_max.get()), int(max_element.get())


def print_run_summary(
    args,
    mesh,
    space,
    cspace,
    l2: float,
    linf: float,
    avg_max: float,
    max_element: int,
    total_seconds: float,
    solve_info: dict,
) -> None:
    assembly_total = timing_value("assembly.total")
    solve_total = timing_value("solve.total")
    reconstruct_total = timing_value("reconstruct.trace") + timing_value("reconstruct.field")
    mesh_total = timing_value("mesh.generate")
    setup_total = timing_value("space.setup") + timing_value("gpu.setup")
    projection_total = timing_value("projection.beta") + timing_value("projection.source") + timing_value("projection.reaction")
    global_dof = int(solve_info.get("global_dof", cspace.mesh.int_edges_inds.size * cspace.edg_dof))
    raw_mode = args.raw_local_assembly if args.assembly_backend == "raw-cuda" else "n/a"
    raw_lu = args.raw_lu_mode if args.assembly_backend == "raw-cuda" else "n/a"
    raw_block = str(args.raw_block_size) if args.assembly_backend == "raw-cuda" else "n/a"
    volume_quad = str(args.volume_quad_1d) if args.volume_quad_1d is not None else "default"
    edge_quad = str(args.edge_quad_1d) if args.edge_quad_1d is not None else "default"
    error_quad = str(args.error_volume_quad_1d) if args.error_volume_quad_1d is not None else "same"

    sections = [
        (
            "Run / mesh",
            [
                ("case", args.case, "s"),
                ("mesh", args.mesh_type, "s"),
                ("mesh size", args.mesh_size, ".3g"),
                ("p", args.order, ",d"),
                ("triangles", mesh.num_tri, ",d"),
                ("edges", mesh.num_edg, ",d"),
                ("trace dofs", global_dof, ",d"),
            ],
        ),
        (
            "Options",
            [
                ("basis", args.basis, "s"),
                ("trace basis", args.trace_basis, "s"),
                ("volume quad 1d", volume_quad, "s"),
                ("edge quad 1d", edge_quad, "s"),
                ("error quad 1d", error_quad, "s"),
                ("assembly backend", args.assembly_backend, "s"),
                ("raw local", raw_mode, "s"),
                ("raw LU", raw_lu, "s"),
                ("raw block", raw_block, "s"),
                ("trace ordering", args.trace_ordering, "s"),
            ],
        ),
        (
            "Solver",
            [
                ("solver", solve_info.get("solver", "unknown"), "s"),
                ("preconditioner", solve_info.get("preconditioner", "unknown"), "s"),
                ("AMGX config", compact_config_name(solve_info.get("amgx_config")), "s"),
                ("matrix nnz", int(solve_info.get("nnz", 0)), ",d"),
                ("scaled rel res", float(solve_info.get("scaled_rel_res", float("nan"))), ".3e"),
                ("AMGX status", solve_info.get("amgx_status", "unknown"), "s"),
                ("AMGX iterations", int(solve_info.get("amgx_iterations", -1)), ",d"),
                ("tol", float(solve_info.get("tolerance", args.amgx_tolerance)), ".1e"),
            ],
        ),
        (
            "Errors",
            [
                ("theoretical h^(p+1)", mesh.h ** (args.order + 1), ".3e"),
                ("L2 error", l2, ".3e"),
                ("Linf error", linf, ".3e"),
                ("avg max error", avg_max, ".3e"),
                ("max-error element", max_element, ",d"),
            ],
        ),
        (
            "Timings",
            [
                ("mesh generation (s)", timing_percent(mesh_total, total_seconds), "s"),
                ("setup (s)", timing_percent(setup_total, total_seconds), "s"),
                ("projections (s)", timing_percent(projection_total, total_seconds), "s"),
                ("assembly (s)", timing_percent(assembly_total, total_seconds), "s"),
                ("global solve (s)", timing_percent(solve_total, total_seconds), "s"),
                ("reconstruct (s)", timing_percent(reconstruct_total, total_seconds), "s"),
                ("plot/error (s)", timing_percent(timing_value("plot_error"), total_seconds), "s"),
                ("total (s)", total_seconds, ".2f"),
            ],
        ),
    ]
    pretty_print_sections(sections, title="HDGFEM GPU4 Advection-Reaction Solve Summary")


def print_detailed_runtime_summary(total_seconds: float, solve_info: dict) -> None:
    assembly_total = timing_value("assembly.total")
    solve_total = timing_value("solve.total")
    reconstruct_total = timing_value("reconstruct.trace") + timing_value("reconstruct.field")
    items = [
        ("mesh generation (s)", timing_value("mesh.generate"), ".5f"),
        ("space/reference setup (s)", timing_value("space.setup"), ".5f"),
        ("GPU mirror/setup (s)", timing_value("gpu.setup"), ".5f"),
        ("advection projection (s)", timing_value("projection.beta"), ".5f"),
        ("source projection (s)", timing_value("projection.source"), ".5f"),
        ("reaction projection (s)", timing_value("projection.reaction"), ".5f"),
        ("assembly total (s)", assembly_total, ".5f"),
        ("  raw block size", timing_value("assembly.raw.block_size"), ".0f"),
        ("  raw map/setup", timing_value("assembly.raw.map_setup"), ".5f"),
        ("  reference adv tensor", timing_value("assembly.reference_advection_tensor"), ".5f"),
        ("  raw solve/emit kernel", timing_value("assembly.raw.kernel"), ".5f"),
        ("  raw total", timing_value("assembly.raw.total"), ".5f"),
        ("  index arrays", timing_value("assembly.indices"), ".5f"),
        ("  local matrices", timing_value("local.reaction_mass") + timing_value("local.boundary_mass") + timing_value("local.advection"), ".5f"),
        ("  local solve", timing_value("local.solve.assembly"), ".5f"),
        ("  trace blocks/data", timing_value("assembly.trace_blocks") + timing_value("assembly.data"), ".5f"),
        ("  RHS/source", timing_value("source_moments") + timing_value("assembly.rhs_faces"), ".5f"),
        ("  boundary elimination", timing_value("assembly.boundary_elimination"), ".5f"),
        ("trace ordering (s)", timing_value("trace_ordering"), ".5f"),
        ("global solve total (s)", solve_total, ".5f"),
        ("  permutation", timing_value("solve.permutation"), ".5f"),
        ("  CSR+scaling", timing_value("solve.csr_scale"), ".5f"),
        ("  AMGX setup", timing_value("solve.amgx_setup"), ".5f"),
        ("  AMGX solve", timing_value("solve.amgx_solve"), ".5f"),
        ("AMGX config", compact_config_name(solve_info.get("amgx_config")), "s"),
        ("AMGX solver", solve_info.get("solver", "unknown"), ""),
        ("AMGX preconditioner", solve_info.get("preconditioner", "unknown"), ""),
        ("AMGX tolerance", float(solve_info.get("tolerance", float("nan"))), ".3e"),
        ("AMGX maxiter", int(solve_info.get("maxiter", 0)), ",d"),
        ("matrix nnz", int(solve_info.get("nnz", 0)), ",d"),
        ("AMGX status", solve_info.get("amgx_status", "unknown"), "s"),
        ("AMGX iterations", int(solve_info.get("amgx_iterations", -1)), ",d"),
        ("scaled abs residual", float(solve_info.get("scaled_abs_res", float("nan"))), ".3e"),
        ("scaled rhs norm", float(solve_info.get("scaled_rhs_norm", float("nan"))), ".3e"),
        ("scaled rel residual", float(solve_info.get("scaled_rel_res", float("nan"))), ".3e"),
        ("reconstruct total (s)", reconstruct_total, ".5f"),
        ("  trace augment", timing_value("reconstruct.trace"), ".5f"),
        ("  field kernel/solve", timing_value("reconstruct.field"), ".5f"),
        ("  local reconstruction solve", timing_value("local.solve.reconstruction"), ".5f"),
        ("plot/error eval (s)", timing_value("plot_error"), ".5f"),
        ("total measured (s)", total_seconds, ".5f"),
    ]
    pretty_print(items, title="HDGFEM GPU4 Detailed Timing / Solver Info")

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="test2_legacy_gpu3", help="case key from scripts/adv_rea_cases.py")
    parser.add_argument("--order", "-o", type=int, default=6)
    parser.add_argument("--mesh-size", "-ms", type=float, default=0.01)
    parser.add_argument("--mesh-type", "-mt", choices=("rectangle", "structured-rectangle"), default="rectangle")
    parser.add_argument("--nx", type=int, default=128, help="structured rectangle cells in x")
    parser.add_argument("--ny", type=int, default=None, help="structured rectangle cells in y")
    parser.add_argument("--basis", default="dub_orth", choices=("hier_C0", "hierarchical_c0", "bernstein", "dub_orth"))
    parser.add_argument("--volume-quad-1d", type=int, default=None, help="1D volume quadrature count for assembly; default is hdgfem's order-based rule")
    parser.add_argument("--edge-quad-1d", type=int, default=None, help="1D edge quadrature count for modern trace basis")
    parser.add_argument("--error-volume-quad-1d", type=int, default=None, help="independent 1D volume quadrature count for L2 error evaluation")
    parser.add_argument("--trace-basis", choices=("legacy-lagrange", "legendre-modal", "modern"), default="legacy-lagrange")
    parser.add_argument("--assembly-backend", choices=("cupy", "raw-cuda"), default="cupy")
    parser.add_argument("--raw-local-assembly", choices=("precomputed", "fused"), default="precomputed", help="For --assembly-backend raw-cuda, choose whether CuPy precomputes local tensors or the RawKernel builds them from projected coefficients")
    parser.add_argument("--raw-lu-mode", choices=("safe", "coop"), default="safe", help="LU factorization mode for fused raw CUDA local assembly; 'safe' is the validated default, 'coop' is experimental")
    parser.add_argument("--raw-block-size", type=int, choices=(1, 32, 64, 128), default=32, help="CUDA block size for --assembly-backend raw-cuda; 1 keeps the serial baseline")
    parser.add_argument("--plot-resolution", "-pr", type=int, default=20)
    parser.add_argument("--gmsh-verbosity", type=int, default=0)
    parser.add_argument("--gmsh-algorithm", type=int, default=None)
    parser.add_argument("--gmsh-num-threads", type=int, default=None, help="optional positive CPU thread count passed to Gmsh")
    parser.add_argument("--trace-ordering", choices=("none", "upwind-scc"), default="none")
    parser.add_argument("--trace-ordering-flux-tolerance", type=float, default=0.0)
    parser.add_argument("--amgx-config", default=None, help="Path to a PyAMGX JSON config; defaults to configs/amgx working config when present")
    parser.add_argument("--amgx-solver", default="BICGSTAB", help="AMGX Krylov solver, e.g. BICGSTAB, GMRES, FGMRES")
    parser.add_argument("--amgx-tolerance", type=float, default=1.0e-14)
    parser.add_argument("--amgx-maxiter", type=int, default=1500)
    parser.add_argument("--show-cupy-config", action="store_true")
    parser.add_argument("--verbosity", "-v", type=int, choices=(0, 1, 2), default=1, help="Logging verbosity: 0 final summary only, 1 progress plus summary, 2 detailed GPU timing and solver diagnostics")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.raw_lu_mode != "safe" and (args.assembly_backend != "raw-cuda" or args.raw_local_assembly != "fused"):
        parser.error("--raw-lu-mode coop requires --assembly-backend raw-cuda --raw-local-assembly fused")
    cp = require_cupy()
    require_cupyx_sparse()
    require_pyamgx()
    # Match the diffusion GPU4 runner: PyAMGX allocates directly through CUDA,
    # so do not let CuPy retain large freed assembly temporaries in its memory
    # pool before AMGX uploads the global matrix.
    cp.cuda.set_allocator(None)
    cp.cuda.set_pinned_memory_allocator(None)
    if args.show_cupy_config:
        cp.show_config()

    original_stdout = None
    quiet_handle = None
    stdout_fd_copy = None
    if args.verbosity == 0:
        original_stdout = sys.stdout
        quiet_handle = open(os.devnull, "w")
        stdout_fd_copy = os.dup(1)
        sys.stdout = quiet_handle
        os.dup2(quiet_handle.fileno(), 1)

    run_start = time.perf_counter()
    print("\n----- Standalone HDGFEM GPU4-Style Advection-Reaction Solve -----")
    print(
        f"case={args.case}, mesh={args.mesh_type}, mesh_size={args.mesh_size:g}, "
        f"order={args.order}, basis={args.basis}, volume_quad_1d={args.volume_quad_1d}, "
        f"assembly_backend={args.assembly_backend}, raw_local_assembly={args.raw_local_assembly}, "
        f"raw_lu_mode={args.raw_lu_mode}, raw_block_size={args.raw_block_size}"
    )

    case = case_definition_by_key(args.case)
    beta_x, beta_y, reaction, source, exact = case.build()

    print("generating mesh ... ", end="", flush=True)
    start = time.perf_counter()
    if args.mesh_type == "structured-rectangle":
        mesh = rectangle_mesh(args.nx, args.ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    else:
        mesh = gmsh_rectangle_mesh(
            args.mesh_size,
            xlim=(-1.0, 1.0),
            ylim=(-1.0, 1.0),
            verbosity=args.gmsh_verbosity,
            algorithm=args.gmsh_algorithm,
            num_threads=args.gmsh_num_threads,
        )
    mesh_time = record_timing("mesh.generate", time.perf_counter() - start)
    print_done(mesh_time)

    print("building DG space/reference data ... ", end="", flush=True)
    start = time.perf_counter()
    space = DGSpace(
        mesh,
        args.order,
        basis_type=args.basis,
        volume_quad_1d=args.volume_quad_1d,
        edge_quad_1d=args.edge_quad_1d,
    )
    space_time = record_timing("space.setup", time.perf_counter() - start)
    print_done(space_time)
    print(f"h={mesh.h:.2e}, h^(p+1)={mesh.h ** (args.order + 1):.2e}, triangles={mesh.num_tri:,}")

    print("copying static data to GPU ... ", end="", flush=True)
    start = time.perf_counter()
    cspace = as_cupy_space(space)
    maps = build_dof_maps(cspace)
    trace_ref = build_trace_reference(cspace, args.trace_basis)
    gpu_time = sync_time(cp, start, "gpu.setup")
    print_done(gpu_time)

    beta_0 = project_callable_cupy(beta_x, cspace, "projecting beta_x ... ", "projection.beta")
    beta_1 = project_callable_cupy(beta_y, cspace, "projecting beta_y ... ", "projection.beta")
    beta_coeffs = cp.ascontiguousarray(cp.stack((beta_0, beta_1), axis=0))

    raw_fused = args.assembly_backend == "raw-cuda" and args.raw_local_assembly == "fused"
    if raw_fused and args.trace_ordering == "upwind-scc":
        print("fused raw CUDA assembly ignores upwind SCC ordering; using trace_ordering=none", flush=True)
        args.trace_ordering = "none"

    beta_dot_normal = None
    if not raw_fused or args.trace_ordering == "upwind-scc":
        beta_dot_normal = beta_dot_normal_from_coeffs(beta_coeffs, cspace, trace_ref)
        cp.cuda.get_current_stream().synchronize()

    trace_permutation = None
    if args.trace_ordering == "upwind-scc":
        print("computing upwind SCC trace ordering ... ", end="", flush=True)
        ordering_start = time.perf_counter()
        ordering = upwind_scc_trace_ordering(
            space.mesh,
            cp.asnumpy(beta_dot_normal),
            cspace.edg_dof,
            active_edges=space.mesh.int_edges_inds,
            flux_tolerance=args.trace_ordering_flux_tolerance,
        )
        elapsed = record_timing("trace_ordering", time.perf_counter() - ordering_start)
        trace_permutation = ordering.dof_permutation
        d = ordering.diagnostics
        levels = d.level_widths
        print(f"done in {elapsed:.5f}s", flush=True)
        print(
            "  upwind SCC graph: "
            f"nodes={d.num_nodes:,}, edges={d.num_directed_edges:,}, "
            f"components={d.num_components:,}, largest={d.largest_component_size:,}, cyclic_nodes={d.cyclic_nodes:,}",
            flush=True,
        )
        print(
            "  upwind level widths: "
            f"levels={levels.num_levels:,}, max={levels.max_width:,}, median={levels.median_width:.1f}, "
            f"mean={levels.mean_width:.1f}, top10_fraction={levels.top10_width_fraction:.3f}",
            flush=True,
        )

    rows, cols, data, rhs, _local_mats, _element_boundary, _source_rhs, boundary_trace, raw_assembly = assemble_reduced_system(
        source,
        reaction,
        exact,
        beta_coeffs,
        beta_dot_normal,
        maps,
        cspace,
        trace_ref,
        backend=args.assembly_backend,
        raw_block_size=args.raw_block_size,
        raw_local_assembly=args.raw_local_assembly,
        raw_lu_mode=args.raw_lu_mode,
    )
    trace_reduced, solve_info = solve_amgx(rows, cols, data, rhs, args, permutation=trace_permutation)
    trace = reconstruct_trace(trace_reduced, boundary_trace, cspace)
    if args.assembly_backend == "raw-cuda":
        uh = reconstruct_field_raw_cuda(trace, raw_assembly, cspace, trace_ref)
    else:
        uh = reconstruct_field(trace, source, reaction, beta_coeffs, beta_dot_normal, cspace, trace_ref)
    l2, linf, avg_max, max_element = evaluate_errors(
        uh,
        exact,
        cspace,
        args.plot_resolution,
        error_volume_quad_1d=args.error_volume_quad_1d,
    )

    total = time.perf_counter() - run_start
    maybe_restore_stdout(original_stdout, quiet_handle, stdout_fd_copy)
    original_stdout = None
    quiet_handle = None
    stdout_fd_copy = None
    print_run_summary(args, mesh, space, cspace, l2, linf, avg_max, max_element, total, solve_info)
    if args.verbosity >= 2:
        print_detailed_runtime_summary(total, solve_info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
