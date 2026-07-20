#!/usr/bin/env python3
"""Standalone CuPy/PyAMGX diffusion-reaction HDG runner using hdgfem core data.

This mirrors the solve-based legacy ``2d/diff_rea_gpu_v4.py`` path, but uses
modern hdgfem mesh, reference-element, quadrature, and basis data.  It is kept
independent of ``scripts/run_diff_rea_cases.py`` so the GPU path can be tested
and tuned without touching the preset runner.
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

from hdgfem.backends.cupy import as_cupy_space, require_cupy, require_cupyx_sparse, require_pyamgx
from hdgfem.backends.cupy_diff_rea_raw import (
    assemble_projected_diffusion_trace_system_eliminated_raw_cuda,
    reconstruct_projected_diffusion_field_raw_cuda,
)
from hdgfem.core.mesh import gmsh_disc_mesh, gmsh_lshape_mesh, gmsh_rectangle_mesh, gmsh_triangle_mesh, rectangle_mesh
from hdgfem.core.quadrature import ReferenceElementData
from hdgfem.core.space import DGSpace
from scripts.diff_rea_cases import case_definition_by_key


LEGACY_V4_BASELINE = {
    "command": "2d/diff_rea_gpu_v4.py -o 6 -ms 0.05 -test 3 -pr 12",
    "triangles": 72969,
    "global_dof": 763973,
    "setup": 1.488,
    "global_solve": 1.875,
    "amgx_solve": 1.159,
    "reconstruct_solve": 0.441,
    "l2_error": 9.95e-10,
    "linf_error": 1.852e-08,
    "wall": 10.35,
}


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "amgx"
CLASSICAL_AMG_CONFIG_PATH = CONFIG_DIR / "diff_rea_gpu4_hdg_pcgf_classical_amg.json"
CHEB_L1_AMG_CONFIG_PATH = CONFIG_DIR / "diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json"
CHEBPOLY4_L1_AMG_CONFIG_PATH = CONFIG_DIR / "diff_rea_gpu4_hdg_pcgf_chebpoly4_l1_aggressive.json"
DEFAULT_AMGX_CONFIG_PATH = CHEB_L1_AMG_CONFIG_PATH


RECOMMENDED_PROFILES = {
    "nodal": {
        "best": (
            "legacy-lagrange trace + PCGF + Chebyshev/L1 aggressive AMG",
            CHEB_L1_AMG_CONFIG_PATH,
            "fastest average solve path in p=4..7 coarse sweeps and p=6 fine solve-time tests",
        ),
        "second best": (
            "legacy-lagrange trace + PCGF + ChebPoly4/L1 aggressive AMG",
            CHEBPOLY4_L1_AMG_CONFIG_PATH,
            "similar residuals; often close, but setup is heavier",
        ),
        "most robust": (
            "legacy-lagrange trace + PCGF + classical V-cycle GS AMG",
            CLASSICAL_AMG_CONFIG_PATH,
            "conservative SPD path; slower at some degrees but reliable",
        ),
    },
    "modal": {
        "best": (
            "legendre-modal trace + BICGSTAB + classical AMG",
            CLASSICAL_AMG_CONFIG_PATH,
            "best practical modal default; modal PCGF remains preconditioner-sensitive",
        ),
        "second best": (
            "legendre-modal trace + PCGF + ChebPoly4/L1 aggressive AMG",
            CHEBPOLY4_L1_AMG_CONFIG_PATH,
            "can improve modal PCGF accuracy, but setup/solve cost is not predictable",
        ),
        "most robust": (
            "legendre-modal trace + BICGSTAB + classical AMG",
            CLASSICAL_AMG_CONFIG_PATH,
            "robust across the failing modal PCGF cases checked so far",
        ),
    },
}


RECOMMENDATION_TEXT = """recommended profiles:
  nodal best:        --trace-basis legacy-lagrange --amgx-solver PCGF --amgx-config configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json
  nodal second:      --trace-basis legacy-lagrange --amgx-solver PCGF --amgx-config configs/amgx/diff_rea_gpu4_hdg_pcgf_chebpoly4_l1_aggressive.json
  nodal robust:      --trace-basis legacy-lagrange --amgx-solver PCGF --amgx-config configs/amgx/diff_rea_gpu4_hdg_pcgf_classical_amg.json
  modal best/robust: --trace-basis legendre-modal --amgx-solver BICGSTAB --amgx-config configs/amgx/diff_rea_gpu4_hdg_pcgf_classical_amg.json
  modal PCGF test:   --trace-basis legendre-modal --amgx-solver PCGF --amgx-config configs/amgx/diff_rea_gpu4_hdg_pcgf_chebpoly4_l1_aggressive.json
"""


AMGX_CONFIG = {
    "config_version": 2,
    "solver": {
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "CLASSICAL",
            "selector": "PMIS",
            "strength": "AHAT",
            "strength_threshold": 0.75,
            "interpolator": "D2",
            "cycle": "V",
            "presweeps": 1,
            "postsweeps": 1,
            "smoother": {"solver": "MULTICOLOR_GS", "symmetric_GS": 1},
            "coarse_solver": "DENSE_LU_SOLVER",
            "max_iters": 1,
            "max_levels": 100,
            "print_grid_stats": 1,
            "dense_lu_num_rows": 2048,
            "dense_lu_max_rows": 4096,
            "coarsest_sweeps": 1,
        },
        "solver": "PCGF",
        "tolerance": 1.0e-13,
        "max_iters": 2000,
        "convergence": "RELATIVE_INI",
        "norm": "L2",
        "print_solve_stats": 1,
        "monitor_residual": 1,
        "store_res_history": 1,
        "obtain_timings": 1,
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
    face_element_test_trace_trial: object
    face_trace_test_element_trial_oriented: object
    M_rf_fc: object


TIMINGS: dict[str, float] = {}


def record_timing(key: str, seconds: float) -> float:
    seconds = float(seconds)
    TIMINGS[key] = TIMINGS.get(key, 0.0) + seconds
    return seconds


def timing_value(key: str) -> float:
    return TIMINGS.get(key, 0.0)


def sync_time(cp, start: float, key: str) -> float:
    cp.cuda.get_current_stream().synchronize()
    return record_timing(key, time.perf_counter() - start)


def print_done(seconds: float) -> None:
    print(f"done in {seconds:.5f}s", flush=True)


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
        face_element_test_trace_trial = q.face_trace_test_element_trial_oriented[:3].transpose(0, 2, 1)
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
            face_element_test_trace_trial=cp.ascontiguousarray(face_element_test_trace_trial),
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
        face_element_test_trace_trial=cp.asarray(np.ascontiguousarray(face_coupling), dtype=cp.float64),
        face_trace_test_element_trial_oriented=cp.asarray(trace_lift, dtype=cp.float64),
        M_rf_fc=cp.asarray(np.ascontiguousarray(edge_mass), dtype=cp.float64),
    )


def mapped_quads_cupy(cspace):
    cp = require_cupy()
    mesh = cspace.mesh
    q = cspace.quad_data
    return cp.einsum("Krc,qc->Krq", mesh.aff_mats, q.Krf_quads) + mesh.aff_vecs[:, :, None]


def _callable_is_zero(func: Callable, cp) -> bool:
    try:
        x = cp.asarray([0.0, 0.37], dtype=cp.float64)
        y = cp.asarray([0.0, -0.21], dtype=cp.float64)
        vals = cp.asarray(func(x, y), dtype=cp.float64)
        return bool(cp.all(vals == 0.0).get())
    except Exception:
        return False


def reaction_mass_cupy(reaction: Callable, cspace):
    cp = require_cupy()
    print("  reaction mass ... ", end="", flush=True)
    start = time.perf_counter()
    mesh = cspace.mesh
    q = cspace.quad_data
    if _callable_is_zero(reaction, cp):
        elapsed = sync_time(cp, start, "local.reaction_mass")
        print_done(elapsed)
        return 0.0
    points = mapped_quads_cupy(cspace)
    values = cp.asarray(reaction(points[:, 0, :], points[:, 1, :]), dtype=cp.float64)
    scaled = values * mesh.aff_jacs[:, None]
    flat = scaled @ q.weighted_phi_phi_flat
    result = flat.reshape(mesh.num_tri, cspace.el_dof, cspace.el_dof)
    elapsed = sync_time(cp, start, "local.reaction_mass")
    print_done(elapsed)
    return result


def reference_derivative_mats(cspace):
    cp = require_cupy()
    q = cspace.quad_data
    d0 = cp.einsum("q,iq,jq->ij", q.Krf_w, q.bas_of_quads, q.dbas_of_quads[0], optimize=True)
    d1 = cp.einsum("q,iq,jq->ij", q.Krf_w, q.bas_of_quads, q.dbas_of_quads[1], optimize=True)
    return cp.ascontiguousarray(d0.T), cp.ascontiguousarray(d1.T)


def face_element_mass(trace_ref):
    cp = require_cupy()
    return cp.einsum(
        "fiq,fjq->fij",
        trace_ref.bas_of_bd_quads,
        trace_ref.weighted_bas_of_bd_quads,
        optimize=True,
    )


def local_lhs_mats_cupy(reaction: Callable, cspace, trace_ref, tau: float):
    cp = require_cupy()
    print("  local mixed LHS matrices ... ", end="", flush=True)
    start = time.perf_counter()
    mesh = cspace.mesh
    q = cspace.quad_data
    el_dof = cspace.el_dof
    M_rea = reaction_mass_cupy(reaction, cspace)
    D0T, D1T = reference_derivative_mats(cspace)
    face_mass = face_element_mass(trace_ref)
    local_lhs = cp.zeros((mesh.num_tri, 3 * el_dof, 3 * el_dof), dtype=cp.float64)
    blocks = local_lhs.reshape((mesh.num_tri, 3, el_dof, 3, el_dof))
    blocks[:, 0, :, 0, :] = M_rea + cp.sum(float(tau) * mesh.jacs_el_fc[..., None, None] * face_mass[None, ...], axis=1)
    blocks[:, 1, :, 1, :] = -mesh.aff_jacs[:, None, None] * q.MKrf[None, ...]
    blocks[:, 2, :, 2, :] = -mesh.aff_jacs[:, None, None] * q.MKrf[None, ...]
    blocks[:, 0, :, 1, :] = (
        cp.sum((mesh.jacs_el_fc * mesh.normals[..., 0])[..., None, None] * face_mass[None, ...], axis=1)
        - mesh.aff_mats[:, 1, 1][:, None, None] * D0T[None, ...]
        + mesh.aff_mats[:, 1, 0][:, None, None] * D1T[None, ...]
    )
    blocks[:, 0, :, 2, :] = (
        cp.sum((mesh.jacs_el_fc * mesh.normals[..., 1])[..., None, None] * face_mass[None, ...], axis=1)
        + mesh.aff_mats[:, 0, 1][:, None, None] * D0T[None, ...]
        - mesh.aff_mats[:, 0, 0][:, None, None] * D1T[None, ...]
    )
    blocks[:, 1, :, 0, :] = mesh.aff_mats[:, 1, 1][:, None, None] * D0T[None, ...] - mesh.aff_mats[:, 1, 0][:, None, None] * D1T[None, ...]
    blocks[:, 2, :, 0, :] = -mesh.aff_mats[:, 0, 1][:, None, None] * D0T[None, ...] + mesh.aff_mats[:, 0, 0][:, None, None] * D1T[None, ...]
    elapsed = sync_time(cp, start, "local.lhs")
    print_done(elapsed)
    return cp.ascontiguousarray(local_lhs)


def element_boundary_mats_cupy(cspace, trace_ref, tau: float):
    cp = require_cupy()
    print("  poisson boundary matrices ... ", end="", flush=True)
    start = time.perf_counter()
    mesh = cspace.mesh
    el_dof = cspace.el_dof
    edg_dof = cspace.edg_dof
    result = cp.zeros((mesh.num_tri, 3 * el_dof, 3 * edg_dof), dtype=cp.float64)
    blocks = result.reshape((mesh.num_tri, 3, el_dof, 3, edg_dof))
    oriented = trace_ref.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling]
    coupling = oriented.transpose(0, 3, 1, 2)
    scaled = mesh.jacs_el_fc[:, None, :, None] * coupling
    blocks[:, 0] = float(tau) * scaled
    blocks[:, 1] = mesh.normals[..., 0][:, None, :, None] * scaled
    blocks[:, 2] = mesh.normals[..., 1][:, None, :, None] * scaled
    elapsed = sync_time(cp, start, "local.element_boundary")
    print_done(elapsed)
    return cp.ascontiguousarray(result)


def source_moments_cupy(source: Callable, cspace):
    cp = require_cupy()
    print("  source moments ... ", end="", flush=True)
    start = time.perf_counter()
    mesh = cspace.mesh
    q = cspace.quad_data
    points = mapped_quads_cupy(cspace)
    values = cp.asarray(source(points[:, 0, :], points[:, 1, :]), dtype=cp.float64)
    rhs = cp.zeros((mesh.num_tri, 3 * cspace.el_dof), dtype=cp.float64)
    rhs[:, : cspace.el_dof] = mesh.aff_jacs[:, None] * cp.einsum("Kq,iq,q->Ki", values, q.bas_of_quads, q.Krf_w, optimize=True)
    elapsed = sync_time(cp, start, "source_moments")
    print_done(elapsed)
    return cp.ascontiguousarray(rhs)


def solve_local_mats(local_lhs, rhs, label: str, key: str):
    cp = require_cupy()
    print(label, end="", flush=True)
    start = time.perf_counter()
    result = cp.linalg.solve(local_lhs, rhs)
    elapsed = sync_time(cp, start, key)
    print_done(elapsed)
    return cp.ascontiguousarray(result)


def b_trace_mats_cupy(cspace, trace_ref, tau: float):
    cp = require_cupy()
    print("  B trace matrices ... ", end="", flush=True)
    start = time.perf_counter()
    mesh = cspace.mesh
    el_dof = cspace.el_dof
    edg_dof = cspace.edg_dof
    lift = mesh.jacs_el_fc[..., None, None] * trace_ref.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling]
    result = cp.zeros((mesh.num_tri, 3, edg_dof, 3 * el_dof), dtype=cp.float64)
    result[..., :el_dof] = float(tau) * lift
    result[..., el_dof : 2 * el_dof] = lift * mesh.normals[..., 0, None, None]
    result[..., 2 * el_dof :] = lift * mesh.normals[..., 1, None, None]
    elapsed = sync_time(cp, start, "assembly.b_trace")
    print_done(elapsed)
    return cp.ascontiguousarray(result)


def trace_blocks_cupy(B_el_fc, solved_el_bd, cspace, trace_ref):
    cp = require_cupy()
    print("  trace blocks ... ", end="", flush=True)
    start = time.perf_counter()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    blocks = B_el_fc @ solved_el_bd[:, None, :, :]
    blocks = blocks.reshape(mesh.num_tri, 3, edg_dof, 3, edg_dof)
    result = cp.ascontiguousarray(blocks.swapaxes(2, 3))
    elapsed = sync_time(cp, start, "assembly.trace_blocks")
    print_done(elapsed)
    return result


def trace_data_cupy(trace_blocks, cspace, trace_ref, tau: float):
    cp = require_cupy()
    print("  COO data ... ", end="", flush=True)
    start = time.perf_counter()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    valid_elements = mesh.interior_elements
    valid_faces = mesh.interior_faces
    n_flux = valid_elements.size * 3 * edg_dof * edg_dof
    n_mass = mesh.int_edges_inds.size * edg_dof * edg_dof
    data = cp.empty(n_flux + n_mass, dtype=cp.float64)
    data[:n_flux] = -trace_blocks[valid_elements, valid_faces].ravel()
    data[n_flux:] = (2.0 * float(tau) * mesh.edge_jacs[mesh.int_edges_inds, None, None] * trace_ref.M_rf_fc[None]).ravel()
    elapsed = sync_time(cp, start, "assembly.data")
    print_done(elapsed)
    return data


def face_rhs_cupy(B_el_fc, solved_src, cspace):
    cp = require_cupy()
    print("  face RHS ... ", end="", flush=True)
    start = time.perf_counter()
    result = (B_el_fc @ solved_src[:, None, :, None]).squeeze(-1)
    elapsed = sync_time(cp, start, "assembly.rhs_faces")
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


def boundary_trace_values_cupy(exact: Callable, cspace, trace_ref):
    cp = require_cupy()
    mesh = cspace.mesh
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


def assemble_reduced_system(source, reaction, exact, maps, cspace, trace_ref, tau: float, backend: str = "cupy"):
    cp = require_cupy()
    if backend == "raw-cuda":
        return assemble_reduced_system_raw_cuda(source, exact, cspace, trace_ref, tau)

    print("assembling reduced trace system (hdgfem/cupy diffusion gpu4-style) ...", flush=True)
    start_total = time.perf_counter()
    rows, cols = setup_reduced_indices(cspace)
    local_lhs = local_lhs_mats_cupy(reaction, cspace, trace_ref, tau)
    B_el_fc = b_trace_mats_cupy(cspace, trace_ref, tau)
    element_boundary = element_boundary_mats_cupy(cspace, trace_ref, tau)
    source_rhs = source_moments_cupy(source, cspace)
    local_rhs = cp.concatenate((element_boundary, source_rhs[..., None]), axis=2)
    solved = solve_local_mats(local_lhs, local_rhs, "  local solve trace/source ... ", "local.solve.assembly")
    solved_el_bd = solved[:, :, : 3 * cspace.edg_dof]
    solved_src = solved[:, :, 3 * cspace.edg_dof]
    del solved, local_rhs
    blocks = trace_blocks_cupy(B_el_fc, solved_el_bd, cspace, trace_ref)
    data = trace_data_cupy(blocks, cspace, trace_ref, tau)
    faces = face_rhs_cupy(B_el_fc, solved_src, cspace)
    del blocks, solved_el_bd, solved_src
    rhs_full = cp.zeros(cspace.mesh.num_edg * cspace.edg_dof, dtype=cp.float64)
    rhs_full_r = rhs_full.reshape((cspace.mesh.num_edg, cspace.edg_dof))
    cp.add.at(
        rhs_full_r,
        cspace.mesh.loc2glob_edge[cspace.mesh.interior_elements, cspace.mesh.interior_faces],
        faces[cspace.mesh.interior_elements, cspace.mesh.interior_faces],
    )
    boundary_trace = boundary_trace_values_cupy(exact, cspace, trace_ref)
    rows, cols, data, rhs = eliminate_boundary_cupy(rows, cols, data, rhs_full, boundary_trace, maps, cspace)
    total = sync_time(cp, start_total, "assembly.total")
    print(f"assembly completed in {total:.5f}s", flush=True)
    return rows, cols, data, rhs, local_lhs, element_boundary, source_rhs, boundary_trace, None


def assemble_reduced_system_raw_cuda(source, exact, cspace, trace_ref, tau: float):
    cp = require_cupy()
    print("assembling reduced trace system (raw CUDA fused diffusion gpu4-style) ...", flush=True)
    start_total = time.perf_counter()
    source_rhs = source_moments_cupy(source, cspace)
    boundary_trace = boundary_trace_values_cupy(exact, cspace, trace_ref)
    D0T, D1T = reference_derivative_mats(cspace)
    face_mass = face_element_mass(trace_ref)
    raw = assemble_projected_diffusion_trace_system_eliminated_raw_cuda(
        source_rhs=source_rhs,
        boundary_trace=boundary_trace,
        cspace=cspace,
        trace_ref=trace_ref,
        d0_reference=D0T,
        d1_reference=D1T,
        face_element_mass=face_mass,
        tau=tau,
    )
    for key, value in raw.timings.items():
        record_timing(f"assembly.{key}", value)
    total = sync_time(cp, start_total, "assembly.total")
    print(f"  raw CUDA map/setup: {raw.timings.get('raw.map_setup', 0.0):.5f}s", flush=True)
    print(f"  raw CUDA fused kernel: {raw.timings.get('raw.kernel', 0.0):.5f}s", flush=True)
    print(f"assembly completed in {total:.5f}s", flush=True)
    return raw.rows, raw.cols, raw.data, raw.rhs, None, None, raw.source_rhs, raw.boundary_trace, raw



def short_config_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path(__file__).resolve().parents[1]))
    except ValueError:
        return str(path)


def print_recommended_profiles(args) -> None:
    trace_family = "modal" if args.trace_basis in {"legendre-modal", "modern"} else "nodal"
    print("\nRecommended diffusion GPU4 profiles")
    print("=" * 42)
    for family in ("nodal", "modal"):
        marker = " (selected trace family)" if family == trace_family else ""
        print(f"{family}{marker}:")
        for rank, (label, config_path, note) in RECOMMENDED_PROFILES[family].items():
            print(f"  {rank:<11}: {label}")
            print(f"               config={short_config_path(config_path)}; {note}")
    print("=" * 42)


def load_amgx_config(args):
    config_path = Path(args.amgx_config).expanduser() if args.amgx_config else DEFAULT_AMGX_CONFIG_PATH
    if config_path.exists():
        with config_path.open() as handle:
            config = json.load(handle)
        return config, config_path
    if args.amgx_config:
        raise FileNotFoundError(f"AMGX config file not found: {config_path}")
    return copy.deepcopy(AMGX_CONFIG), None

def solve_amgx(rows, cols, data, rhs, args):
    cp = require_cupy()
    cpsp = require_cupyx_sparse()
    pyamgx = require_pyamgx()
    print("solving global system with PyAMGX ...", flush=True)
    solve_total_start = time.perf_counter()
    n = int(rhs.size)
    csr_start = time.perf_counter()
    matrix = cpsp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=cp.float64).tocsr()
    matrix.sum_duplicates()
    csr_elapsed = sync_time(cp, csr_start, "solve.csr_scale")
    print(f"  CSR assembly without diagonal scaling: {csr_elapsed:.5f}s, nnz={matrix.nnz:,}", flush=True)

    config, config_path = load_amgx_config(args)
    print(f"  AMGX config: {config_path if config_path is not None else 'embedded default'}", flush=True)
    config["solver"]["solver"] = str(args.amgx_solver)
    config["solver"]["tolerance"] = float(args.amgx_tolerance)
    config["solver"]["max_iters"] = int(args.amgx_maxiter)
    verbose_amgx = int(os.environ.get("HDGFEM_GPU4_AMGX_MONITOR", "1") == "1")
    solver_name = str(args.amgx_solver).upper()
    # AMGX CG-family solvers need residual monitoring even in quiet mode;
    # otherwise they can run to max_iters after convergence and produce NaNs.
    needs_residual_monitor = solver_name in {"CG", "PCG", "PCGF"}
    config["solver"]["print_solve_stats"] = verbose_amgx
    config["solver"]["monitor_residual"] = int(verbose_amgx or needs_residual_monitor)
    config["solver"]["store_res_history"] = verbose_amgx
    config["solver"]["obtain_timings"] = verbose_amgx
    config["solver"]["preconditioner"]["print_grid_stats"] = verbose_amgx

    pyamgx.initialize()
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
        vec_x.download_raw(trace.data.ptr)
        cp.cuda.get_current_stream().synchronize()
        residual = cp.linalg.norm(matrix @ trace - rhs)
        rhs_norm = cp.linalg.norm(rhs)
        residual_value = float(residual.get())
        rhs_norm_value = float(rhs_norm.get())
        rel_residual = residual_value / rhs_norm_value if rhs_norm_value else float("nan")
        print(f"  PyAMGX solve: {solve_elapsed:.5f}s, scaled_rel_res={rel_residual:.3e}", flush=True)
    finally:
        for obj in (solver, mat, vec_x, vec_b, rsrc, cfg):
            if obj is not None:
                try:
                    obj.destroy()
                except AttributeError:
                    pass
        pyamgx.finalize()
    record_timing("solve.total", time.perf_counter() - solve_total_start)
    return trace


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


def reconstruct_field(trace, local_lhs, element_boundary, source_rhs, cspace, trace_ref):
    cp = require_cupy()
    print("reconstructing element field ...", flush=True)
    start = time.perf_counter()
    mesh = cspace.mesh
    edg_dof = cspace.edg_dof
    trace_r = trace.reshape((mesh.num_edg, edg_dof))
    element_traces = trace_r[mesh.loc2glob_edge, :]
    element_traces = element_traces.reshape((mesh.num_tri, 3 * edg_dof))
    rhs = source_rhs[..., None] + element_boundary @ element_traces[..., None]
    solution = solve_local_mats(local_lhs, rhs, "  local solve reconstruction ... ", "local.solve.reconstruction").squeeze(-1)
    uh = solution.reshape((mesh.num_tri, 3, cspace.el_dof))[:, 0]
    elapsed = sync_time(cp, start, "reconstruct.field")
    print_done(elapsed)
    return cp.ascontiguousarray(uh)


def reconstruct_field_raw_cuda(trace, raw_assembly, cspace, trace_ref, tau: float):
    cp = require_cupy()
    print("reconstructing element field (raw CUDA) ... ", end="", flush=True)
    start = time.perf_counter()
    uh, kernel_elapsed = reconstruct_projected_diffusion_field_raw_cuda(
        trace=trace,
        source_rhs=raw_assembly.source_rhs,
        cspace=cspace,
        trace_ref=trace_ref,
        d0_reference=raw_assembly.d0_reference,
        d1_reference=raw_assembly.d1_reference,
        face_element_mass=raw_assembly.face_element_mass,
        tau=tau,
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=RECOMMENDATION_TEXT,
    )
    parser.add_argument("--case", default="trigonometric-poisson", help="case key from scripts/diff_rea_cases.py")
    parser.add_argument("--order", "-o", type=int, default=6)
    parser.add_argument("--mesh-size", "-ms", type=float, default=0.05)
    parser.add_argument("--mesh-type", "-mt", choices=("auto", "disc", "rectangle", "unit-rectangle", "triangle", "lshape", "structured-rectangle"), default="auto")
    parser.add_argument("--disc-radius", type=float, default=None)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--basis", default="dub_orth", choices=("hier_C0", "hierarchical_c0", "bernstein", "dub_orth"))
    parser.add_argument("--volume-quad-1d", type=int, default=None)
    parser.add_argument("--edge-quad-1d", type=int, default=None)
    parser.add_argument("--error-volume-quad-1d", type=int, default=None)
    parser.add_argument("--trace-basis", choices=("legacy-lagrange", "legendre-modal", "modern"), default="legacy-lagrange")
    parser.add_argument("--assembly-backend", choices=("cupy", "raw-cuda"), default="cupy")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--plot-resolution", "-pr", type=int, default=12)
    parser.add_argument("--gmsh-verbosity", type=int, default=0)
    parser.add_argument("--gmsh-algorithm", type=int, default=None)
    parser.add_argument(
        "--amgx-config",
        default=None,
        help="Path to a PyAMGX JSON config; omitted means the nodal best Chebyshev/L1 profile",
    )
    parser.add_argument("--amgx-solver", default="PCGF")
    parser.add_argument("--amgx-tolerance", type=float, default=1.0e-13)
    parser.add_argument("--amgx-maxiter", type=int, default=2000)
    parser.add_argument("--show-cupy-config", action="store_true")
    return parser


def build_mesh(args, case):
    domain = args.mesh_type
    if domain == "auto":
        domain = case.default_domain
    if domain == "structured-rectangle":
        return rectangle_mesh(args.nx, args.ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0)), domain
    if domain == "unit-rectangle":
        return gmsh_rectangle_mesh(args.mesh_size, xlim=(0.0, 1.0), ylim=(0.0, 1.0), verbosity=args.gmsh_verbosity, algorithm=args.gmsh_algorithm), domain
    if domain == "rectangle":
        return gmsh_rectangle_mesh(args.mesh_size, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), verbosity=args.gmsh_verbosity, algorithm=args.gmsh_algorithm), domain
    if domain == "disc":
        radius = args.disc_radius if args.disc_radius is not None else (5.0 if case.key == "trigonometric-poisson" else 1.0)
        return gmsh_disc_mesh(args.mesh_size, center=(0.0, 0.0), radius=radius, verbosity=args.gmsh_verbosity, algorithm=args.gmsh_algorithm), domain
    if domain == "lshape":
        return gmsh_lshape_mesh(args.mesh_size, verbosity=args.gmsh_verbosity, algorithm=args.gmsh_algorithm), domain
    return gmsh_triangle_mesh(args.mesh_size, vertices=((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)), verbosity=args.gmsh_verbosity, algorithm=args.gmsh_algorithm), "triangle"


def print_runtime_summary(total_seconds: float) -> None:
    mesh_total = timing_value("mesh.generate")
    space_total = timing_value("space.setup")
    gpu_setup_total = timing_value("gpu.setup")
    setup_total = mesh_total + space_total + gpu_setup_total
    assembly_total = timing_value("assembly.total")
    csr_total = timing_value("solve.csr_scale")
    amgx_setup_total = timing_value("solve.amgx_setup")
    amgx_solve_total = timing_value("solve.amgx_solve")
    solve_total = timing_value("solve.total")
    reconstruct_total = timing_value("reconstruct.trace") + timing_value("reconstruct.field")
    plot_total = timing_value("plot_error")
    amgx_total = csr_total + amgx_setup_total + amgx_solve_total
    gpu_core_total = assembly_total + amgx_total + reconstruct_total
    accounted_total = setup_total + assembly_total + solve_total + reconstruct_total + plot_total

    def pct(seconds: float) -> str:
        return f"{100.0 * seconds / total_seconds:.1f}%" if total_seconds else "nan%"

    items = [
        ("mesh generation (s)", mesh_total, ".3f"),
        ("space/reference setup (s)", space_total, ".3f"),
        ("GPU mirror/setup (s)", gpu_setup_total, ".3f"),
        ("host/setup subtotal (s)", setup_total, ".3f"),
        ("host/setup share", pct(setup_total), ""),
        ("assembly total (s)", assembly_total, ".3f"),
        ("assembly share", pct(assembly_total), ""),
        ("  raw map/setup", timing_value("assembly.raw.map_setup"), ".3f"),
        ("  raw fused kernel", timing_value("assembly.raw.kernel"), ".3f"),
        ("  raw total", timing_value("assembly.raw.total"), ".3f"),
        ("  index arrays", timing_value("assembly.indices"), ".3f"),
        ("  local LHS", timing_value("local.lhs"), ".3f"),
        ("  local solve", timing_value("local.solve.assembly"), ".3f"),
        (
            "  B/trace blocks/data",
            timing_value("assembly.b_trace") + timing_value("assembly.trace_blocks") + timing_value("assembly.data"),
            ".3f",
        ),
        ("  RHS/source", timing_value("source_moments") + timing_value("assembly.rhs_faces"), ".3f"),
        ("  boundary elimination", timing_value("assembly.boundary_elimination"), ".3f"),
        ("global solve total (s)", solve_total, ".3f"),
        ("global solve share", pct(solve_total), ""),
        ("  CSR assembly", csr_total, ".3f"),
        ("  AMGX setup/upload", amgx_setup_total, ".3f"),
        ("  AMGX solve", amgx_solve_total, ".3f"),
        ("  AMGX subtotal", amgx_total, ".3f"),
        ("reconstruct total (s)", reconstruct_total, ".3f"),
        ("reconstruct share", pct(reconstruct_total), ""),
        ("  trace augmentation", timing_value("reconstruct.trace"), ".3f"),
        ("  local solve recon", timing_value("local.solve.reconstruction"), ".3f"),
        ("plot/error eval (s)", plot_total, ".3f"),
        ("plot/error share", pct(plot_total), ""),
        ("GPU core subtotal (s)", gpu_core_total, ".3f"),
        ("GPU core share", pct(gpu_core_total), ""),
        ("accounted subtotal (s)", accounted_total, ".3f"),
        ("total measured (s)", total_seconds, ".3f"),
    ]
    pretty_print(items, title="HDGFEM Diffusion GPU4-Style Timing Summary")


def print_baseline_comparison() -> None:
    items = [
        ("baseline command", LEGACY_V4_BASELINE["command"], ""),
        ("baseline setup (s)", LEGACY_V4_BASELINE["setup"], ".3f"),
        ("this assembly / baseline", timing_value("assembly.total") / LEGACY_V4_BASELINE["setup"], ".3f"),
        ("baseline AMGX solve (s)", LEGACY_V4_BASELINE["amgx_solve"], ".3f"),
        ("this AMGX / baseline", timing_value("solve.amgx_solve") / LEGACY_V4_BASELINE["amgx_solve"], ".3f"),
        ("baseline wall (s)", LEGACY_V4_BASELINE["wall"], ".3f"),
    ]
    pretty_print(items, title="Remembered Legacy Diffusion V4 Baseline")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cp = require_cupy()
    require_cupyx_sparse()
    require_pyamgx()
    # Match the legacy gpu_v4 runner: do not let CuPy keep freed blocks in its
    # memory pool while PyAMGX reports device memory usage.
    cp.cuda.set_allocator(None)
    cp.cuda.set_pinned_memory_allocator(None)
    if args.show_cupy_config:
        cp.show_config()

    run_start = time.perf_counter()
    print("\n----- Standalone HDGFEM GPU4-Style Diffusion-Reaction Solve -----")
    print(
        f"case={args.case}, mesh={args.mesh_type}, mesh_size={args.mesh_size:g}, "
        f"order={args.order}, basis={args.basis}, trace_basis={args.trace_basis}, "
        f"volume_quad_1d={args.volume_quad_1d}, assembly_backend={args.assembly_backend}"
    )
    print_recommended_profiles(args)

    case = case_definition_by_key(args.case)
    problem = case.build()
    diffusion, reaction, source, exact = problem
    if diffusion != (1.0, 0.0, 1.0):
        raise NotImplementedError("this gpu4-style runner currently supports identity diffusion only")

    print("generating mesh ... ", end="", flush=True)
    start = time.perf_counter()
    mesh, domain = build_mesh(args, case)
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

    rows, cols, data, rhs, local_lhs, element_boundary, source_rhs, boundary_trace, raw_assembly = assemble_reduced_system(
        source,
        reaction,
        exact,
        maps,
        cspace,
        trace_ref,
        args.tau,
        backend=args.assembly_backend,
    )
    trace_reduced = solve_amgx(rows, cols, data, rhs, args)
    trace = reconstruct_trace(trace_reduced, boundary_trace, cspace)
    if args.assembly_backend == "raw-cuda":
        uh = reconstruct_field_raw_cuda(trace, raw_assembly, cspace, trace_ref, args.tau)
    else:
        uh = reconstruct_field(trace, local_lhs, element_boundary, source_rhs, cspace, trace_ref)
    l2, linf, avg_max, max_element = evaluate_errors(
        uh,
        exact,
        cspace,
        args.plot_resolution,
        error_volume_quad_1d=args.error_volume_quad_1d,
    )

    total = time.perf_counter() - run_start
    global_dof = int(cspace.mesh.int_edges_inds.size * cspace.edg_dof)
    items = [
        ("case", args.case, ""),
        ("domain", domain, ""),
        ("basis", args.basis, ""),
        ("trace basis", args.trace_basis, ""),
        ("volume quad points", int(space.quad_data.Krf_w.size), ",d"),
        ("error quad", args.error_volume_quad_1d or "same", ""),
        ("order", args.order, ",d"),
        ("triangles", mesh.num_tri, ",d"),
        ("edges", mesh.num_edg, ",d"),
        ("interior edges", int(mesh.int_edges_inds.size), ",d"),
        ("global dof", global_dof, ",d"),
        ("mesh size", args.mesh_size, ".4f"),
        ("h", mesh.h, ".3e"),
        ("h^(p+1)", mesh.h ** (args.order + 1), ".3e"),
        ("L2 error", l2, ".3e"),
        ("Linf error", linf, ".3e"),
        ("avg max error", avg_max, ".3e"),
        ("max-error element", max_element, ",d"),
    ]
    pretty_print(items, title="Solve Summary")
    print_runtime_summary(total)
    print_baseline_comparison()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
