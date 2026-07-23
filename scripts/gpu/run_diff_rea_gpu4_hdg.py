#!/usr/bin/env python3
"""Standalone CuPy/PyAMGX diffusion-reaction HDG runner using hdgfem core data.

This mirrors the solve-based legacy ``2d/diff_rea_gpu_v4.py`` path, but uses
modern hdgfem mesh, reference-element, quadrature, and basis data.  It is kept
independent of ``scripts/diffusion_reaction/run_diff_rea_cases.py`` so the GPU
path can be tested and tuned without touching the preset runner.
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
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from hdgfem.backends.cupy import as_cupy_space, require_cupy, require_cupyx_sparse, require_pyamgx
from hdgfem.backends.cupy_diff_rea_raw import (
    assemble_projected_diffusion_trace_system_eliminated_raw_cuda,
    reconstruct_projected_diffusion_field_raw_cuda,
)
from hdgfem.core.mesh import gmsh_disc_mesh, gmsh_lshape_mesh, gmsh_rectangle_mesh, gmsh_triangle_mesh, rectangle_mesh
from hdgfem.core.quadrature import ReferenceElementData
from hdgfem.core.space import DGSpace
from hdgfem.io.output import pretty_print_sections
from scripts.diffusion_reaction.diff_rea_cases import case_definition_by_key


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "amgx"
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
RUN_METADATA: dict[str, object] = {}
VERBOSITY = 1
RAW_CUDA_MAX_EL_DOF = 28


def set_verbosity(value: int) -> None:
    global VERBOSITY
    VERBOSITY = int(value)


def log(message: str = "", *, level: int = 1, end: str = "\n", flush: bool = True) -> None:
    if VERBOSITY >= int(level):
        print(message, end=end, flush=flush)


def record_timing(key: str, seconds: float) -> float:
    seconds = float(seconds)
    TIMINGS[key] = TIMINGS.get(key, 0.0) + seconds
    return seconds


def timing_value(key: str) -> float:
    return TIMINGS.get(key, 0.0)


def sync_time(cp, start: float, key: str) -> float:
    cp.cuda.get_current_stream().synchronize()
    return record_timing(key, time.perf_counter() - start)


def print_done(seconds: float, *, level: int = 1) -> None:
    log(f"done in {seconds:.5f}s", level=level)


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
            "q,fiq,jq->fij", edge_weights, negative_face_basis, edge_basis, optimize=True
        )
        trace_lift = np.ascontiguousarray(
            np.concatenate((face_coupling.transpose(0, 2, 1), face_coupling_reversed.transpose(0, 2, 1)), axis=0)
        )
        edge_mass = np.einsum("q,iq,jq->ij", edge_weights, edge_basis, edge_basis, optimize=True)
        return TraceReferenceData(
            kind="bernstein",
            nodal=False,
            interpolation_nodes=cp.asarray(edge_quads, dtype=cp.float64),
            quads=cp.asarray(edge_quads, dtype=cp.float64),
            weights=cp.asarray(edge_weights, dtype=cp.float64),
            bas_of_bd_quads=cp.asarray(face_basis, dtype=cp.float64),
            bas1d_of_ref_edg_qds=cp.asarray(edge_basis, dtype=cp.float64),
            weighted_bas_of_bd_quads=cp.asarray(weighted_face_basis, dtype=cp.float64),
            weighted_bas1d_of_ref_edg_qds=cp.asarray(weighted_edge_basis, dtype=cp.float64),
            face_element_test_trace_trial=cp.asarray(face_coupling, dtype=cp.float64),
            face_trace_test_element_trial_oriented=cp.asarray(trace_lift, dtype=cp.float64),
            M_rf_fc=cp.asarray(np.ascontiguousarray(edge_mass), dtype=cp.float64),
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
    log("  reaction mass ... ", end="")
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
    log("  local mixed LHS matrices ... ", end="")
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
    log("  poisson boundary matrices ... ", end="")
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
    log("  source moments ... ", end="")
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
    log(label, end="")
    start = time.perf_counter()
    result = cp.linalg.solve(local_lhs, rhs)
    elapsed = sync_time(cp, start, key)
    print_done(elapsed)
    return cp.ascontiguousarray(result)


def b_trace_mats_cupy(cspace, trace_ref, tau: float):
    cp = require_cupy()
    log("  B trace matrices ... ", end="")
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
    log("  trace blocks ... ", end="")
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
    log("  COO data ... ", end="")
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
    log("  face RHS ... ", end="")
    start = time.perf_counter()
    result = (B_el_fc @ solved_src[:, None, :, None]).squeeze(-1)
    elapsed = sync_time(cp, start, "assembly.rhs_faces")
    print_done(elapsed)
    return cp.ascontiguousarray(result)


def setup_reduced_indices(cspace):
    cp = require_cupy()
    log("  reduced COO indices ... ", end="")
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
    log("  boundary elimination ... ", end="")
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


def raw_cuda_diffusion_fallback_reason(cspace, trace_ref) -> tuple[str, str] | None:
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


def assemble_reduced_system(source, reaction, exact, maps, cspace, trace_ref, tau: float, backend: str = "cupy", raw_matrix_format: str = "coo", raw_block_size: int = 1):
    cp = require_cupy()
    requested_backend = str(backend)
    RUN_METADATA["requested_assembly_backend"] = requested_backend
    if requested_backend == "raw-cuda":
        fallback = raw_cuda_diffusion_fallback_reason(cspace, trace_ref)
        if fallback is None:
            RUN_METADATA["assembly_backend"] = "raw-cuda"
            RUN_METADATA.pop("assembly_fallback", None)
            RUN_METADATA.pop("assembly_fallback_detail", None)
            return assemble_reduced_system_raw_cuda(
                source,
                exact,
                cspace,
                trace_ref,
                tau,
                matrix_format=raw_matrix_format,
                block_size=raw_block_size,
            )
        fallback_label, fallback_detail = fallback
        RUN_METADATA["assembly_backend"] = "cupy"
        RUN_METADATA["assembly_fallback"] = fallback_label
        RUN_METADATA["assembly_fallback_detail"] = fallback_detail
        log(f"raw CUDA diffusion assembly fallback: {fallback_detail}; using CuPy assembly.")
    else:
        RUN_METADATA["assembly_backend"] = requested_backend
        RUN_METADATA.pop("assembly_fallback", None)
        RUN_METADATA.pop("assembly_fallback_detail", None)

    log("assembling reduced trace system (hdgfem/cupy diffusion gpu4-style) ...")
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
    log(f"assembly completed in {total:.5f}s")
    return rows, cols, data, rhs, local_lhs, element_boundary, source_rhs, boundary_trace, None


def assemble_reduced_system_raw_cuda(source, exact, cspace, trace_ref, tau: float, *, matrix_format: str = "coo", block_size: int = 1):
    cp = require_cupy()
    log("assembling reduced trace system (raw CUDA fused diffusion gpu4-style) ...")
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
        matrix_format=matrix_format,
        block_size=block_size,
    )
    for key, value in raw.timings.items():
        record_timing(f"assembly.{key}", value)
    total = sync_time(cp, start_total, "assembly.total")
    log(f"  raw CUDA matrix format: {raw.matrix_format}")
    log(f"  raw CUDA block size: {int(raw.timings.get('raw.block_size', 1.0))}")
    log(f"  raw CUDA map/setup: {raw.timings.get('raw.map_setup', 0.0):.5f}s")
    if raw.matrix_format == "csr":
        log(f"  raw CUDA csr zero: {raw.timings.get('raw.csr_zero', 0.0):.5f}s")
        log(f"  raw CUDA fused CSR kernel: {raw.timings.get('raw.csr_kernel', 0.0):.5f}s")
    else:
        log(f"  raw CUDA fused kernel: {raw.timings.get('raw.kernel', 0.0):.5f}s")
    log(f"assembly completed in {total:.5f}s")
    return raw.rows, raw.cols, raw.data, raw.rhs, None, None, raw.source_rhs, raw.boundary_trace, raw



def short_config_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path(__file__).resolve().parents[2]))
    except ValueError:
        return str(path)


def print_recommended_profiles(args) -> None:
    if VERBOSITY < 2:
        return
    trace_family = "modal" if args.trace_basis in {"legendre-modal", "bernstein"} else "nodal"
    log("\nRecommended diffusion GPU4 profiles", level=2)
    log("=" * 42, level=2)
    for family in ("nodal", "modal"):
        marker = " (selected trace family)" if family == trace_family else ""
        log(f"{family}{marker}:", level=2)
        for rank, (label, config_path, note) in RECOMMENDED_PROFILES[family].items():
            log(f"  {rank:<11}: {label}", level=2)
            log(f"               config={short_config_path(config_path)}; {note}", level=2)
    log("=" * 42, level=2)


def load_amgx_config(args):
    config_path = Path(args.amgx_config).expanduser() if args.amgx_config else DEFAULT_AMGX_CONFIG_PATH
    if config_path.exists():
        with config_path.open() as handle:
            config = json.load(handle)
        return config, config_path
    if args.amgx_config:
        raise FileNotFoundError(f"AMGX config file not found: {config_path}")
    return copy.deepcopy(AMGX_CONFIG), None

def _cupyx_csr_from_coo(rows, cols, data, rhs):
    cp = require_cupy()
    cpsp = require_cupyx_sparse()
    n = int(rhs.size)
    matrix = cpsp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=cp.float64).tocsr()
    matrix.sum_duplicates()
    return _cupyx_csr_from_arrays(matrix.data, matrix.indptr, matrix.indices, shape=matrix.shape)


def _cupyx_csr_from_arrays(data, indptr, indices, *, shape):
    cp = require_cupy()
    cpsp = require_cupyx_sparse()
    if indices.dtype != cp.int32 or indptr.dtype != cp.int32:
        indices = indices.astype(cp.int32, copy=False)
        indptr = indptr.astype(cp.int32, copy=False)
    return cpsp.csr_matrix((data, indices, indptr), shape=shape, dtype=cp.float64)


def _upload_device_csr_to_amgx(mat, matrix) -> None:
    mat.upload(matrix.indptr, matrix.indices, matrix.data, shape=matrix.shape)


def solve_amgx(rows, cols, data, rhs, args, *, indptr=None, indices=None):
    cp = require_cupy()
    pyamgx = require_pyamgx()
    log("solving global system with PyAMGX ...")
    solve_total_start = time.perf_counter()
    csr_start = time.perf_counter()
    if indptr is None or indices is None:
        matrix = _cupyx_csr_from_coo(rows, cols, data, rhs)
        csr_elapsed = sync_time(cp, csr_start, "solve.csr_scale")
        matrix_format = "coo->csr"
        log(f"  CSR assembly without diagonal scaling: {csr_elapsed:.5f}s, nnz={matrix.nnz:,}")
    else:
        matrix = _cupyx_csr_from_arrays(data, indptr, indices, shape=(int(rhs.size), int(rhs.size)))
        csr_elapsed = sync_time(cp, csr_start, "solve.csr_scale")
        matrix_format = "direct csr"
        log(f"  direct CSR view setup: {csr_elapsed:.5f}s, nnz={matrix.nnz:,}")
    RUN_METADATA["matrix_nnz"] = int(matrix.nnz)
    RUN_METADATA["solve_matrix_format"] = matrix_format

    config, config_path = load_amgx_config(args)
    log(f"  AMGX config: {config_path if config_path is not None else 'embedded default'}")
    config["solver"]["solver"] = str(args.amgx_solver)
    config["solver"]["tolerance"] = float(args.amgx_tolerance)
    config["solver"]["max_iters"] = int(args.amgx_maxiter)
    monitor_env = os.environ.get("HDGFEM_GPU4_AMGX_MONITOR")
    verbose_amgx = int((VERBOSITY >= 2) if monitor_env is None else monitor_env == "1")
    solver_name = str(args.amgx_solver).upper()
    RUN_METADATA["amgx_config"] = str(config_path) if config_path is not None else "embedded default"
    RUN_METADATA["amgx_solver"] = solver_name
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
        phase_start = time.perf_counter()
        cfg = pyamgx.Config().create_from_dict(config)
        rsrc = pyamgx.Resources().create_simple(cfg)
        mat = pyamgx.Matrix().create(rsrc, mode="dDDI")
        vec_b = pyamgx.Vector().create(rsrc, mode="dDDI")
        vec_x = pyamgx.Vector().create(rsrc, mode="dDDI")
        cp.cuda.get_current_stream().synchronize()
        setup_objects_elapsed = record_timing("solve.amgx_setup.objects", time.perf_counter() - phase_start)

        phase_start = time.perf_counter()
        _upload_device_csr_to_amgx(mat, matrix)
        cp.cuda.get_current_stream().synchronize()
        matrix_upload_elapsed = record_timing("solve.amgx_setup.matrix_upload", time.perf_counter() - phase_start)

        phase_start = time.perf_counter()
        vec_b.upload_raw(rhs.data.ptr, rhs.size)
        trace = cp.zeros(rhs.size, dtype=cp.float64)
        vec_x.upload_raw(trace.data.ptr, trace.size)
        cp.cuda.get_current_stream().synchronize()
        vector_upload_elapsed = record_timing("solve.amgx_setup.vector_upload", time.perf_counter() - phase_start)

        phase_start = time.perf_counter()
        solver = pyamgx.Solver().create(rsrc, cfg)
        solver.setup(mat)
        cp.cuda.get_current_stream().synchronize()
        solver_setup_elapsed = record_timing("solve.amgx_setup.solver_setup", time.perf_counter() - phase_start)

        setup_elapsed = record_timing("solve.amgx_setup", time.perf_counter() - setup_start)
        log(f"  PyAMGX setup/upload: {setup_elapsed:.5f}s")
        log(
            "    setup breakdown: "
            f"objects={setup_objects_elapsed:.5f}s "
            f"matrix_upload={matrix_upload_elapsed:.5f}s "
            f"vectors={vector_upload_elapsed:.5f}s "
            f"solver_setup={solver_setup_elapsed:.5f}s",
            level=2,
        )

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
        RUN_METADATA["amgx_scaled_rel_residual"] = rel_residual
        RUN_METADATA["amgx_residual_norm"] = residual_value
        RUN_METADATA["amgx_rhs_norm"] = rhs_norm_value
        log(f"  PyAMGX solve: {solve_elapsed:.5f}s, scaled_rel_res={rel_residual:.3e}")
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
    log("augmenting traces ... ", end="")
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
    log("reconstructing element field ...")
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


def reconstruct_field_raw_cuda(trace, raw_assembly, cspace, trace_ref, tau: float, raw_block_size: int = 1):
    cp = require_cupy()
    log("reconstructing element field (raw CUDA) ... ", end="")
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
        block_size=raw_block_size,
    )
    record_timing("local.solve.reconstruction", kernel_elapsed)
    elapsed = sync_time(cp, start, "reconstruct.field")
    print_done(elapsed)
    return uh


def evaluate_errors(uh, exact: Callable, cspace, plot_resolution: int, error_volume_quad_1d: int | None = None):
    cp = require_cupy()
    log("computing plot/error data ... ", end="")
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
    parser.add_argument("--case", default="trigonometric-poisson", help="case key from scripts/diffusion_reaction/diff_rea_cases.py")
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
    parser.add_argument("--trace-basis", choices=("legacy-lagrange", "legendre-modal", "bernstein"), default="legacy-lagrange")
    parser.add_argument("--assembly-backend", choices=("cupy", "raw-cuda"), default="cupy")
    parser.add_argument("--raw-matrix-format", choices=("coo", "csr"), default="coo")
    parser.add_argument("--raw-block-size", type=int, choices=(1, 32, 64, 128), default=1)
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
    parser.add_argument("--verbosity", "-v", type=int, choices=(0, 1, 2), default=1)
    return parser


def build_mesh(args, case):
    domain = args.mesh_type
    if domain == "auto":
        domain = case.default_domain
    log_cache = int(getattr(args, "verbosity", 1)) >= 1
    if domain == "structured-rectangle":
        return rectangle_mesh(args.nx, args.ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0)), domain
    if domain == "unit-rectangle":
        return gmsh_rectangle_mesh(
            args.mesh_size,
            xlim=(0.0, 1.0),
            ylim=(0.0, 1.0),
            verbosity=args.gmsh_verbosity,
            algorithm=args.gmsh_algorithm,
            log_cache=log_cache,
        ), domain
    if domain == "rectangle":
        return gmsh_rectangle_mesh(
            args.mesh_size,
            xlim=(-1.0, 1.0),
            ylim=(-1.0, 1.0),
            verbosity=args.gmsh_verbosity,
            algorithm=args.gmsh_algorithm,
            log_cache=log_cache,
        ), domain
    if domain == "disc":
        radius = args.disc_radius if args.disc_radius is not None else (5.0 if case.key == "trigonometric-poisson" else 1.0)
        return gmsh_disc_mesh(
            args.mesh_size,
            center=(0.0, 0.0),
            radius=radius,
            verbosity=args.gmsh_verbosity,
            algorithm=args.gmsh_algorithm,
            log_cache=log_cache,
        ), domain
    if domain == "lshape":
        return gmsh_lshape_mesh(
            args.mesh_size,
            verbosity=args.gmsh_verbosity,
            algorithm=args.gmsh_algorithm,
            log_cache=log_cache,
        ), domain
    return gmsh_triangle_mesh(
        args.mesh_size,
        vertices=((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)),
        verbosity=args.gmsh_verbosity,
        algorithm=args.gmsh_algorithm,
        log_cache=log_cache,
    ), "triangle"


def _display_optional_int(value) -> str:
    return "default" if value is None else f"{int(value):,d}"


def _display_amgx_config() -> str:
    value = RUN_METADATA.get("amgx_config", "embedded default")
    if value == "embedded default":
        return "embedded default"
    return short_config_path(Path(str(value)))


def _format_timing_cell(seconds: float, total_seconds: float) -> str:
    seconds = float(seconds)
    if total_seconds > 0.0:
        return f"{seconds:.3f}s ({100.0 * seconds / total_seconds:.1f}%)"
    return f"{seconds:.3f}s"


def print_final_summary(
        *,
        args,
        domain: str,
        mesh,
        space,
        cspace,
        l2: float,
        linf: float,
        avg_max: float,
        max_element: int,
        total_seconds: float,
) -> None:
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
    global_dof = int(cspace.mesh.int_edges_inds.size * cspace.edg_dof)
    raw_kernel_total = timing_value("assembly.raw.csr_kernel") or timing_value("assembly.raw.kernel")
    amgx_subtotal = csr_total + amgx_setup_total + amgx_solve_total
    effective_backend = str(RUN_METADATA.get("assembly_backend", args.assembly_backend))
    requested_backend = str(RUN_METADATA.get("requested_assembly_backend", args.assembly_backend))
    fallback_label = RUN_METADATA.get("assembly_fallback")

    run_options = [
        ("case", args.case, "s"),
        ("domain", domain, "s"),
        ("backend", effective_backend, "s"),
    ]
    if requested_backend != effective_backend:
        run_options.append(("requested backend", requested_backend, "s"))
    if fallback_label:
        run_options.append(("fallback", str(fallback_label), "s"))
    run_options.extend(
        [
            ("basis", args.basis, "s"),
            ("trace basis", args.trace_basis, "s"),
            ("raw matrix", args.raw_matrix_format if effective_backend == "raw-cuda" else "n/a", "s"),
            ("raw block", str(args.raw_block_size) if effective_backend == "raw-cuda" else "n/a", "s"),
            ("verbosity", args.verbosity, ",d"),
        ]
    )
    mesh_details = [
        ("order", args.order, ",d"),
        ("mesh size", args.mesh_size, ".4f"),
        ("h", mesh.h, ".3e"),
        ("h^(p+1)", mesh.h ** (args.order + 1), ".3e"),
        ("triangles", mesh.num_tri, ",d"),
        ("edges", mesh.num_edg, ",d"),
        ("interior edges", int(mesh.int_edges_inds.size), ",d"),
        ("global dof", global_dof, ",d"),
        ("element dof", int(cspace.el_dof), ",d"),
        ("edge dof", int(cspace.edg_dof), ",d"),
        ("volume quad pts", int(space.quad_data.Krf_w.size), ",d"),
        ("error quad 1d", _display_optional_int(args.error_volume_quad_1d), "s"),
    ]
    solver_details = [
        ("config", _display_amgx_config(), "s"),
        ("solver", str(RUN_METADATA.get("amgx_solver", args.amgx_solver)), "s"),
        ("tol", args.amgx_tolerance, ".1e"),
        ("maxiter", args.amgx_maxiter, ",d"),
        ("matrix input", str(RUN_METADATA.get("solve_matrix_format", "unknown")), "s"),
        ("matrix nnz", int(RUN_METADATA.get("matrix_nnz", 0)), ",d"),
        ("scaled residual", float(RUN_METADATA.get("amgx_scaled_rel_residual", float("nan"))), ".3e"),
    ]
    errors = [
        ("L2", l2, ".3e"),
        ("Linf", linf, ".3e"),
        ("avg max", avg_max, ".3e"),
        ("max element", max_element, ",d"),
    ]
    timings = [
        ("setup subtotal", _format_timing_cell(setup_total, total_seconds), "s"),
        ("mesh setup", _format_timing_cell(mesh_total, total_seconds), "s"),
        ("space setup", _format_timing_cell(space_total, total_seconds), "s"),
        ("GPU setup", _format_timing_cell(gpu_setup_total, total_seconds), "s"),
        ("assembly", _format_timing_cell(assembly_total, total_seconds), "s"),
    ]
    if effective_backend == "raw-cuda":
        timings.extend(
            [
                ("raw map/setup", _format_timing_cell(timing_value("assembly.raw.map_setup"), total_seconds), "s"),
                ("raw zero", _format_timing_cell(timing_value("assembly.raw.csr_zero"), total_seconds), "s"),
                ("raw kernel", _format_timing_cell(raw_kernel_total, total_seconds), "s"),
                ("raw total", _format_timing_cell(timing_value("assembly.raw.total"), total_seconds), "s"),
            ]
        )
    else:
        timings.extend(
            [
                ("index arrays", _format_timing_cell(timing_value("assembly.indices"), total_seconds), "s"),
                ("local LHS", _format_timing_cell(timing_value("local.lhs"), total_seconds), "s"),
                ("local solve", _format_timing_cell(timing_value("local.solve.assembly"), total_seconds), "s"),
                (
                    "trace blocks/data",
                    _format_timing_cell(
                        timing_value("assembly.b_trace")
                        + timing_value("assembly.trace_blocks")
                        + timing_value("assembly.data"),
                        total_seconds,
                    ),
                    "s",
                ),
                (
                    "RHS/source",
                    _format_timing_cell(timing_value("source_moments") + timing_value("assembly.rhs_faces"), total_seconds),
                    "s",
                ),
                ("boundary elim", _format_timing_cell(timing_value("assembly.boundary_elimination"), total_seconds), "s"),
            ]
        )
    timings.extend(
        [
            ("global solve", _format_timing_cell(solve_total, total_seconds), "s"),
            ("CSR/view", _format_timing_cell(csr_total, total_seconds), "s"),
            ("AMGX setup", _format_timing_cell(amgx_setup_total, total_seconds), "s"),
            ("AMGX solve", _format_timing_cell(amgx_solve_total, total_seconds), "s"),
            ("AMGX subtotal", _format_timing_cell(amgx_subtotal, total_seconds), "s"),
            ("reconstruct", _format_timing_cell(reconstruct_total, total_seconds), "s"),
            ("local recon solve", _format_timing_cell(timing_value("local.solve.reconstruction"), total_seconds), "s"),
            ("plot/error", _format_timing_cell(plot_total, total_seconds), "s"),
            ("total measured", f"{total_seconds:.3f}s", "s"),
        ]
    )

    sections = [
        ("Run / Options", run_options),
        ("Mesh / DOF", mesh_details),
        ("Solver", solver_details),
        ("Errors", errors),
        ("Timings", timings),
    ]
    pretty_print_sections(sections, title="HDGFEM GPU4 Diffusion-Reaction Solve Summary")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    set_verbosity(args.verbosity)
    TIMINGS.clear()
    RUN_METADATA.clear()
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
    log("\n----- Standalone HDGFEM GPU4-Style Diffusion-Reaction Solve -----")
    log(
        f"case={args.case}, mesh={args.mesh_type}, mesh_size={args.mesh_size:g}, "
        f"order={args.order}, basis={args.basis}, trace_basis={args.trace_basis}, "
        f"volume_quad_1d={args.volume_quad_1d}, assembly_backend={args.assembly_backend}, "
        f"raw_matrix_format={args.raw_matrix_format}, raw_block_size={args.raw_block_size}"
    )
    print_recommended_profiles(args)
    if case_definition_by_key is None:
        raise ModuleNotFoundError("Case definitions are unavailable in this repository checkout")

    case = case_definition_by_key(args.case)
    problem = case.build()
    diffusion, reaction, source, exact = problem
    if diffusion != (1.0, 0.0, 1.0):
        raise NotImplementedError("this gpu4-style runner currently supports identity diffusion only")

    log("generating mesh ... ", end="")
    start = time.perf_counter()
    mesh, domain = build_mesh(args, case)
    mesh_time = record_timing("mesh.generate", time.perf_counter() - start)
    print_done(mesh_time)

    log("building DG space/reference data ... ", end="")
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
    log(f"h={mesh.h:.2e}, h^(p+1)={mesh.h ** (args.order + 1):.2e}, triangles={mesh.num_tri:,}")

    log("copying static data to GPU ... ", end="")
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
        raw_matrix_format=args.raw_matrix_format,
        raw_block_size=args.raw_block_size,
    )
    trace_reduced = solve_amgx(
        rows,
        cols,
        data,
        rhs,
        args,
        indptr=getattr(raw_assembly, "indptr", None),
        indices=getattr(raw_assembly, "indices", None),
    )
    trace = reconstruct_trace(trace_reduced, boundary_trace, cspace)
    if raw_assembly is not None:
        uh = reconstruct_field_raw_cuda(trace, raw_assembly, cspace, trace_ref, args.tau, args.raw_block_size)
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
    print_final_summary(
        args=args,
        domain=domain,
        mesh=mesh,
        space=space,
        cspace=cspace,
        l2=l2,
        linf=linf,
        avg_max=avg_max,
        max_element=max_element,
        total_seconds=total,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
