#!/usr/bin/env python3
"""Standalone CuPy/PyAMGX diffusion-reaction HDG runner using hdgfem core data.

This mirrors the solve-based legacy ``2d/diff_rea_gpu_v4.py`` path, but uses
modern hdgfem mesh, reference-element, quadrature, and basis data.  It is kept
independent of ``scripts/diffusion_reaction/run_cases.py`` so the GPU
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
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from hdgfem.backends.cupy import (
    as_cupy_coefficients,
    as_cupy_space,
    diagonal_scale_cupy_csr_rows_in_place,
    require_cupy,
    require_cupyx_sparse,
    require_pyamgx,
    symmetric_scale_cupy_csr_in_place,
)
from hdgfem.backends.raw_cuda import resolve_raw_cuda_block_size
from hdgfem.backends.diffusion_cupy import (
    assemble_projected_diffusion_trace_system_eliminated_raw_cupy,
    postprocess_projected_diffusion_primal_cupy,
)
from hdgfem.backends.diffusion_raw_cuda import (
    postprocess_projected_diffusion_primal_raw_cuda,
    reconstruct_projected_diffusion_field_raw_cuda,
)
from hdgfem.core.mesh import gmsh_disc_mesh, gmsh_lshape_mesh, gmsh_rectangle_mesh, gmsh_triangle_mesh, rectangle_mesh
from hdgfem.core.quadrature import ReferenceElementData
from hdgfem.core.space import DGField, DGSpace
from hdgfem.io.output import pretty_print_sections
from hdgfem.io.plot import (
    _require_pyvista,
    add_samples_to_plotter,
    resolve_exact_plot_resolution,
    sample_callable_on_elements,
    sample_field_on_elements,
)
from hdgfem.solvers.diffusion_reaction import _new_hdg_postprocess_cache, _postprocess_diffusion_solution
from scripts.diffusion_reaction.cases import case_definition_by_key, zero_coefficient


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


@dataclass(frozen=True)
class DeviceCsrScaleResult:
    matrix: Any
    rhs: Any
    mode: str
    elapsed_seconds: float
    row_diagonal: Any | None = None
    inverse_sqrt_diagonal: Any | None = None


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


def timing_prefix_total(prefix: str) -> float:
    return sum(value for key, value in TIMINGS.items() if key.startswith(prefix))


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


def reaction_mass_cupy(reaction: Callable, cspace):
    cp = require_cupy()
    log("  reaction mass ... ", end="")
    start = time.perf_counter()
    mesh = cspace.mesh
    q = cspace.quad_data
    if np.isscalar(reaction):
        scalar = float(reaction)
        if scalar == 0.0:
            elapsed = sync_time(cp, start, "local.reaction_mass")
            print_done(elapsed)
            return 0.0
        result = scalar * mesh.aff_jacs[:, None, None] * q.MKrf[None, ...]
    elif isinstance(reaction, DGField):
        reaction.space.assert_same_mesh(cspace.host)
        constant_value = reaction.constant_value
        if constant_value is not None:
            if constant_value == 0.0:
                elapsed = sync_time(cp, start, "local.reaction_mass")
                print_done(elapsed)
                return 0.0
            result = constant_value * mesh.aff_jacs[:, None, None] * q.MKrf[None, ...]
        else:
            coeffs = as_cupy_coefficients(reaction, cspace)
            values = coeffs @ q.bas_of_quads
            scaled = values * mesh.aff_jacs[:, None]
            flat = scaled @ q.weighted_phi_phi_flat
            result = flat.reshape(mesh.num_tri, cspace.el_dof, cspace.el_dof)
    else:
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
    if np.isscalar(source):
        values = cp.full((mesh.num_tri, q.Krf_w.size), float(source), dtype=cp.float64)
    elif isinstance(source, DGField):
        source.space.assert_same_mesh(cspace.host)
        coeffs = as_cupy_coefficients(source, cspace)
        values = coeffs @ q.bas_of_quads
    else:
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
    num_points = int(t.size)
    expected_shape = (int(mesh.bnd_edges_inds.size), num_points)
    if values.ndim == 0:
        values = cp.full(expected_shape, float(values), dtype=cp.float64)
    elif values.shape == (num_points,):
        values = cp.broadcast_to(values[None, :], expected_shape)
    if values.shape != expected_shape:
        raise ValueError(
            "boundary_condition must return a scalar, edge-point vector, or "
            f"{expected_shape} array; got {values.shape}"
        )
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


def _raw_cuda_diffusion_source_input(source, space: DGSpace):
    if isinstance(source, DGField):
        source.space.assert_same_mesh(space)
        if source.space is not space:
            raise ValueError("source_h must live in the runner DGSpace object for raw-cuda assembly")
        return source
    if callable(source) or np.isscalar(source):
        return source
    return space.field(source, name="source_h")


def _raw_cuda_diffusion_reaction_field(reaction, space: DGSpace) -> DGField:
    if isinstance(reaction, DGField):
        reaction.space.assert_same_mesh(space)
        if reaction.space is not space:
            raise ValueError("reaction_h must live in the runner DGSpace object for raw-cuda assembly")
        return reaction
    if reaction is zero_coefficient:
        return space.zeros(name="reaction_h")
    if np.isscalar(reaction):
        return space.constant(float(reaction), name="reaction_h")
    if callable(reaction):
        raise TypeError(
            "raw-cuda diffusion assembly requires reaction_h as a DGField and currently supports only zero reaction; "
            "use space.zeros(name='reaction_h') for pure Poisson cases."
        )
    return space.field(reaction, name="reaction_h")


def _prepare_raw_cuda_diffusion_inputs(source, reaction, space: DGSpace):
    log("preparing raw CUDA coefficient inputs ... ", end="")
    total_start = time.perf_counter()

    start = time.perf_counter()
    source_input = _raw_cuda_diffusion_source_input(source, space)
    source_elapsed = record_timing("assembly.coefficient_prep.source", time.perf_counter() - start)

    start = time.perf_counter()
    reaction_h = _raw_cuda_diffusion_reaction_field(reaction, space)
    reaction_elapsed = record_timing("assembly.coefficient_prep.reaction", time.perf_counter() - start)

    total_elapsed = record_timing("assembly.coefficient_prep", time.perf_counter() - total_start)
    print_done(total_elapsed)
    log(f"  source input prep: {source_elapsed:.5f}s", level=2)
    log(f"  reaction field prep: {reaction_elapsed:.5f}s", level=2)
    return source_input, reaction_h


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
            source_input, reaction_h = _prepare_raw_cuda_diffusion_inputs(source, reaction, cspace.host)
            return assemble_reduced_system_raw_cuda(
                source_input,
                reaction_h,
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

    log("assembling reduced trace system (hdgfem/cupy diffusion CUDA) ...")
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


def assemble_reduced_system_raw_cuda(source, reaction, exact, cspace, trace_ref, tau: float, *, matrix_format: str = "coo", block_size: int = 1):
    cp = require_cupy()
    log("assembling reduced trace system (raw CUDA fused diffusion CUDA) ...")
    start_total = time.perf_counter()
    assembly = assemble_projected_diffusion_trace_system_eliminated_raw_cupy(
        source,
        reaction,
        exact,
        tau,
        cspace,
        trace_basis=getattr(trace_ref, "kind", "legacy-lagrange"),
        matrix_format=matrix_format,
        block_size=block_size,
        trace_ref=trace_ref,
    )
    raw = assembly.raw_assembly
    if raw is None:
        raise RuntimeError("raw CUDA diffusion helper did not return raw assembly metadata")
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
    log("\nRecommended diffusion CUDA profiles", level=2)
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


def normalize_scale_system(value: str) -> str:
    value = str(value).lower()
    if value == "on":
        return "left"
    if value == "off":
        return "none"
    if value not in {"none", "left", "symmetric"}:
        raise ValueError(f"unknown scale-system value: {value}")
    return value


def resolve_scale_system(args) -> str:
    return normalize_scale_system(args.scale_system)


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


def prepare_scaled_device_csr_for_solve(
        matrix,
        rhs,
        scale_system: str,
        *,
        timing_key: str | None = "solve.scale",
) -> DeviceCsrScaleResult:
    """Return the device CSR/RHS that should be uploaded to AMGX.

    Scaling stays fully on device.  For symmetric scaling this prepares
    ``D^-1/2 A D^-1/2`` and ``D^-1/2 b``; callers recover the physical trace by
    multiplying the scaled unknown by the returned ``inverse_sqrt_diagonal``.
    """
    cp = require_cupy()
    mode = normalize_scale_system(scale_system)
    row_diagonal = None
    inverse_sqrt_diagonal = None
    if mode == "none":
        return DeviceCsrScaleResult(matrix=matrix, rhs=rhs, mode=mode, elapsed_seconds=0.0)

    start = time.perf_counter()
    solve_matrix = _cupyx_csr_from_arrays(
        matrix.data.copy(),
        matrix.indptr,
        matrix.indices,
        shape=matrix.shape,
    )
    solve_rhs = rhs.copy()
    if mode == "left":
        row_diagonal = diagonal_scale_cupy_csr_rows_in_place(solve_matrix, solve_rhs)
    elif mode == "symmetric":
        inverse_sqrt_diagonal = symmetric_scale_cupy_csr_in_place(solve_matrix, solve_rhs)
    else:
        raise ValueError(f"unknown scale-system value: {scale_system}")
    cp.cuda.get_current_stream().synchronize()
    elapsed = time.perf_counter() - start
    if timing_key is not None:
        record_timing(timing_key, elapsed)
    return DeviceCsrScaleResult(
        matrix=solve_matrix,
        rhs=solve_rhs,
        mode=mode,
        elapsed_seconds=elapsed,
        row_diagonal=row_diagonal,
        inverse_sqrt_diagonal=inverse_sqrt_diagonal,
    )


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
        log(f"  CSR assembly: {csr_elapsed:.5f}s, nnz={matrix.nnz:,}")
    else:
        matrix = _cupyx_csr_from_arrays(data, indptr, indices, shape=(int(rhs.size), int(rhs.size)))
        csr_elapsed = sync_time(cp, csr_start, "solve.csr_scale")
        matrix_format = "direct csr"
        log(f"  direct CSR view setup: {csr_elapsed:.5f}s, nnz={matrix.nnz:,}")
    RUN_METADATA["matrix_nnz"] = int(matrix.nnz)
    RUN_METADATA["solve_matrix_format"] = matrix_format

    scale_system = resolve_scale_system(args)
    RUN_METADATA["solve_scale_system"] = scale_system
    scaled_system = prepare_scaled_device_csr_for_solve(matrix, rhs, scale_system, timing_key="solve.scale")
    solve_matrix = scaled_system.matrix
    solve_rhs = scaled_system.rhs
    row_diagonal = scaled_system.row_diagonal
    inverse_sqrt_diagonal = scaled_system.inverse_sqrt_diagonal
    scale_elapsed = scaled_system.elapsed_seconds
    RUN_METADATA["solve_scale_seconds"] = scale_elapsed
    RUN_METADATA["solve_scale_vector"] = (
        "row diagonal" if row_diagonal is not None
        else "inverse sqrt diagonal" if inverse_sqrt_diagonal is not None
        else "none"
    )
    if scale_system != "none":
        log(f"  {scale_system} diagonal scaling: {scale_elapsed:.5f}s")
    else:
        log("  diagonal scaling: off", level=2)

    config, config_path = load_amgx_config(args)
    log(f"  AMGX config: {config_path if config_path is not None else 'embedded default'}")
    config["solver"]["solver"] = str(args.amgx_solver)
    config["solver"]["tolerance"] = float(args.amgx_tolerance)
    config["solver"]["max_iters"] = int(args.amgx_maxiter)
    monitor_env = os.environ.get("HDGFEM_CUDA_AMGX_MONITOR")
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
        _upload_device_csr_to_amgx(mat, solve_matrix)
        cp.cuda.get_current_stream().synchronize()
        matrix_upload_elapsed = record_timing("solve.amgx_setup.matrix_upload", time.perf_counter() - phase_start)

        phase_start = time.perf_counter()
        vec_b.upload_raw(solve_rhs.data.ptr, solve_rhs.size)
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
        try:
            RUN_METADATA["amgx_status"] = str(solver.status)
        except Exception:
            RUN_METADATA["amgx_status"] = "unknown"
        try:
            RUN_METADATA["amgx_iterations"] = int(solver.iterations_number)
        except Exception:
            RUN_METADATA["amgx_iterations"] = None
        vec_x.download_raw(trace.data.ptr)
        cp.cuda.get_current_stream().synchronize()
        solver_residual_vec = solve_matrix @ trace - solve_rhs
        solver_residual = cp.linalg.norm(solver_residual_vec)
        solver_rhs_norm = cp.linalg.norm(solve_rhs)
        solver_residual_value = float(solver_residual.get())
        solver_rhs_norm_value = float(solver_rhs_norm.get())
        solver_rel_residual = solver_residual_value / solver_rhs_norm_value if solver_rhs_norm_value else float("nan")
        if inverse_sqrt_diagonal is not None:
            trace *= inverse_sqrt_diagonal
            physical_residual = cp.linalg.norm(solver_residual_vec / inverse_sqrt_diagonal)
            physical_rhs_norm = cp.linalg.norm(solve_rhs / inverse_sqrt_diagonal)
            physical_residual_value = float(physical_residual.get())
            physical_rhs_norm_value = float(physical_rhs_norm.get())
            physical_rel_residual = (
                physical_residual_value / physical_rhs_norm_value if physical_rhs_norm_value else float("nan")
            )
        elif row_diagonal is not None:
            physical_residual = cp.linalg.norm(row_diagonal * solver_residual_vec)
            physical_rhs_norm = cp.linalg.norm(row_diagonal * solve_rhs)
            physical_residual_value = float(physical_residual.get())
            physical_rhs_norm_value = float(physical_rhs_norm.get())
            physical_rel_residual = (
                physical_residual_value / physical_rhs_norm_value if physical_rhs_norm_value else float("nan")
            )
        else:
            physical_residual_value = solver_residual_value
            physical_rhs_norm_value = solver_rhs_norm_value
            physical_rel_residual = solver_rel_residual
        RUN_METADATA["amgx_scaled_rel_residual"] = solver_rel_residual
        RUN_METADATA["amgx_solver_rel_residual"] = solver_rel_residual
        RUN_METADATA["amgx_physical_rel_residual"] = physical_rel_residual
        RUN_METADATA["amgx_residual_norm"] = solver_residual_value
        RUN_METADATA["amgx_rhs_norm"] = solver_rhs_norm_value
        RUN_METADATA["amgx_physical_residual_norm"] = physical_residual_value
        RUN_METADATA["amgx_physical_rhs_norm"] = physical_rhs_norm_value
        iteration_text = RUN_METADATA.get("amgx_iterations")
        iteration_text = "unknown" if iteration_text is None else f"{int(iteration_text):,d}"
        log(
            f"  PyAMGX solve: {solve_elapsed:.5f}s, iterations={iteration_text}, "
            f"solver_rel_res={solver_rel_residual:.3e}, physical_rel_res={physical_rel_residual:.3e}"
        )
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


def reconstruct_field(trace, local_lhs, element_boundary, source_rhs, cspace, trace_ref, *, return_local_unknowns: bool = False):
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
    local_unknowns = cp.ascontiguousarray(solution.reshape((mesh.num_tri, 3 * cspace.el_dof)))
    uh = cp.ascontiguousarray(local_unknowns.reshape((mesh.num_tri, 3, cspace.el_dof))[:, 0])
    elapsed = sync_time(cp, start, "reconstruct.field")
    print_done(elapsed)
    if return_local_unknowns:
        return uh, local_unknowns
    return uh


def reconstruct_field_raw_cuda(
        trace,
        raw_assembly,
        cspace,
        trace_ref,
        tau: float,
        raw_block_size: int = 1,
        *,
        return_local_unknowns: bool = False,
):
    cp = require_cupy()
    log("reconstructing element field (raw CUDA) ... ", end="")
    start = time.perf_counter()
    result = reconstruct_projected_diffusion_field_raw_cuda(
        trace=trace,
        source_rhs=raw_assembly.source_rhs,
        cspace=cspace,
        trace_ref=trace_ref,
        d0_reference=raw_assembly.d0_reference,
        d1_reference=raw_assembly.d1_reference,
        face_element_mass=raw_assembly.face_element_mass,
        tau=tau,
        block_size=raw_block_size,
        return_local_unknowns=return_local_unknowns,
    )
    if return_local_unknowns:
        uh, local_unknowns, kernel_elapsed = result
    else:
        uh, kernel_elapsed = result
        local_unknowns = None
    record_timing("local.solve.reconstruction", kernel_elapsed)
    elapsed = sync_time(cp, start, "reconstruct.field")
    print_done(elapsed)
    if return_local_unknowns:
        return uh, local_unknowns
    return uh


def _host_source_field_for_projected_reconstruction(source, space: DGSpace) -> DGField:
    """Return a same-space source field for host-side projected reconstruction."""
    if isinstance(source, DGField):
        source.space.assert_same_mesh(space)
        if source.space is not space:
            raise ValueError("source_h must live in the runner DGSpace object for raw-cuda postprocessing")
        return source
    if np.isscalar(source):
        return space.constant(float(source), name="source_h")
    if callable(source):
        return space.project_callable(source, name="source_h")
    return space.field(source, name="source_h")


def _host_reaction_field_for_projected_reconstruction(reaction, space: DGSpace) -> DGField:
    """Return a same-space reaction field for host-side projected reconstruction."""
    if isinstance(reaction, DGField):
        reaction.space.assert_same_mesh(space)
        if reaction.space is not space:
            raise ValueError("reaction_h must live in the runner DGSpace object for raw-cuda postprocessing")
        return reaction
    if reaction is zero_coefficient:
        return space.zeros(name="reaction_h")
    if np.isscalar(reaction):
        return space.constant(float(reaction), name="reaction_h")
    if callable(reaction):
        raise TypeError(
            "raw-cuda primal postprocess plotting requires reaction_h as a DGField or exact zero/constant; "
            "project callable reactions before using raw-cuda postprocess plotting."
        )
    return space.field(reaction, name="reaction_h")


def postprocess_primal_field_for_plot(
        *,
        trace,
        local_unknowns_device,
        source,
        reaction,
        exact: Callable,
        space: DGSpace,
        trace_basis: str,
        diffusion,
        tau: float,
        backend: str,
        cspace,
        raw_block_size: int,
        postprocess_cache=None,
) -> DGField:
    """Build ``u_h^*`` for plotting from GPU solve data."""
    cp = require_cupy()
    backend = str(backend).lower()
    if backend not in {"host", "cupy", "raw-cuda"}:
        raise ValueError("postprocess backend must be 'host', 'cupy', or 'raw-cuda'")
    RUN_METADATA["postprocess_backend"] = backend
    trace_space = space.trace_space(trace_basis)
    log(f"post-processing primal field ({backend}) ... ", end="")
    start = time.perf_counter()

    if backend == "cupy":
        if local_unknowns_device is None:
            raise RuntimeError("CuPy primal postprocess requires device local unknowns")
        device_timings: dict[str, float] = {}
        postprocessed_field, _ = postprocess_projected_diffusion_primal_cupy(
            local_unknowns_device,
            space,
            diffusion,
            trace_space=trace_space,
            timings=device_timings,
            cache=postprocess_cache,
        )
        for key, value in device_timings.items():
            record_timing(key, value)

    elif backend == "raw-cuda":
        if local_unknowns_device is None:
            raise RuntimeError("raw CUDA primal postprocess requires device local unknowns")
        device_timings = {}
        postprocessed_field, _ = postprocess_projected_diffusion_primal_raw_cuda(
            local_unknowns_device,
            cspace,
            diffusion,
            trace_space=trace_space,
            block_size=raw_block_size,
            timings=device_timings,
            cache=postprocess_cache,
        )
        for key, value in device_timings.items():
            record_timing(key, value)
        block = int(device_timings.get("postprocess.primal.raw_cuda.block_size", raw_block_size))
        row_fit = int(device_timings.get("postprocess.primal.raw_cuda.row_fit_order", -1))
        supported = int(device_timings.get("postprocess.primal.raw_cuda.supported_order", -1))
        RUN_METADATA["postprocess_raw_cuda_block_size"] = block
        RUN_METADATA["postprocess_raw_cuda_row_fit_order"] = row_fit
        RUN_METADATA["postprocess_raw_cuda_supported_order"] = supported

    else:
        trace_host = cp.asnumpy(trace)
        if local_unknowns_device is None:
            from hdgfem.backends.numba import reconstruct_projected_diffusion_local_unknowns_numba

            source_h = _host_source_field_for_projected_reconstruction(source, space)
            reaction_h = _host_reaction_field_for_projected_reconstruction(reaction, space)
            reconstruct_start = time.perf_counter()
            local_unknowns_host = reconstruct_projected_diffusion_local_unknowns_numba(
                trace_host,
                source_h,
                reaction_h,
                tau,
                space,
                trace_space=trace_space,
            )
            record_timing("postprocess.primal.host_reconstruct", time.perf_counter() - reconstruct_start)
        else:
            local_unknowns_host = cp.asnumpy(local_unknowns_device)

        postprocessed_field, _, _ = _postprocess_diffusion_solution(
            local_unknowns_host,
            trace_host,
            space,
            tau,
            diffusion,
            "primal",
            trace_space=trace_space,
            cache=postprocess_cache,
        )
        if postprocessed_field is None:
            raise RuntimeError("primal HDG postprocessing did not produce a field")

    elapsed = record_timing("postprocess.primal", time.perf_counter() - start)
    print_done(elapsed)
    if backend == "raw-cuda":
        log(
            f"  raw CUDA primal postprocess block size: "
            f"{RUN_METADATA.get('postprocess_raw_cuda_block_size')}; "
            f"row-fit p<={RUN_METADATA.get('postprocess_raw_cuda_row_fit_order')}, "
            f"current raw path p<={RUN_METADATA.get('postprocess_raw_cuda_supported_order')}",
            level=1,
        )

    if postprocessed_field.device_coefficients_materialized() and not postprocessed_field.coefficients_materialized:
        traffic_start = time.perf_counter()
        _ = postprocessed_field.coeffs
        record_timing("traffic.device_to_host.postprocess_primal_coeffs", time.perf_counter() - traffic_start)

    log("computing postprocessed primal error diagnostics ... ", end="", level=2)
    error_start = time.perf_counter()
    points = postprocessed_field.space.mapped_quads()
    exact_values = exact(points[:, :, 0], points[:, :, 1])
    post_diff = postprocessed_field.values() - exact_values
    RUN_METADATA["postprocess_primal_l2"] = float(
        np.sqrt(
            np.einsum(
                "K,Kq,q->",
                postprocessed_field.space.mesh.aff_jacs,
                post_diff * post_diff,
                postprocessed_field.space.quad_data.Krf_w,
                optimize=True,
            )
        )
    )
    RUN_METADATA["postprocess_primal_linf"] = float(np.max(np.abs(post_diff)))
    diagnostic_elapsed = record_timing("diagnostics.postprocess_primal_error_host", time.perf_counter() - error_start)
    print_done(diagnostic_elapsed, level=2)
    return postprocessed_field


def sample_postprocessed_primal_for_plot(
        postprocessed_field: DGField,
        exact: Callable,
        *,
        resolution: int,
) -> dict[str, np.ndarray]:
    """Sample ``u_h^*`` and exact values on the postprocessed-space plot grid."""
    reference_points, _, values = sample_field_on_elements(postprocessed_field, resolution=resolution)
    _, _, exact_values = sample_callable_on_elements(
        postprocessed_field.space.mesh,
        exact,
        reference_points=reference_points,
    )
    return {
        "reference_points": reference_points,
        "postprocessed_values": values,
        "exact_values_for_error": exact_values,
    }


def effective_plot_resolution(requested_resolution: int, order: int, num_elements: int) -> int:
    """Use a polynomial-degree minimum for coarse per-element plots."""
    resolution = int(requested_resolution)
    if int(num_elements) <= 130:
        return max(resolution, 2 * int(order) + 3, 3)
    return resolution


def matplotlib_contour_levels(order: int) -> int:
    """Choose enough contour bands for coarse per-element degree-``order`` plots."""
    return min(256, max(128, 24 * (int(order) + 1)))


def postprocess_plot_resolution(numerical_resolution: int, order: int, num_elements: int) -> int:
    """Choose a plotting resolution that can show degree ``p+1`` postprocess data."""
    return effective_plot_resolution(max(int(numerical_resolution), 2 * (int(order) + 1) + 3), int(order) + 1, num_elements)


def evaluate_errors(
        uh,
        exact: Callable,
        cspace,
        plot_resolution: int,
        error_volume_quad_1d: int | None = None,
        *,
        return_plot_samples: bool = False,
):
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
    traffic_start = time.perf_counter()
    ref_points_host = cp.asnumpy(ref_points)
    record_timing("traffic.device_to_host.plot_reference_points", time.perf_counter() - traffic_start)
    basis_plot_host = cspace.host.basis_at(ref_points_host)
    traffic_start = time.perf_counter()
    basis_plot = cp.asarray(basis_plot_host, dtype=cp.float64)
    cp.cuda.get_current_stream().synchronize()
    record_timing("traffic.host_to_device.plot_basis", time.perf_counter() - traffic_start)
    mapped_plot = cp.einsum("Krc,pc->Krp", mesh.aff_mats, ref_points) + mesh.aff_vecs[:, :, None]
    exact_plot = cp.asarray(exact(mapped_plot[:, 0, :], mapped_plot[:, 1, :]), dtype=cp.float64)
    uh_plot = uh @ basis_plot.T
    abs_err = cp.abs(uh_plot - exact_plot)
    linf = cp.max(abs_err)
    avg_max = cp.average(cp.max(abs_err, axis=-1))
    max_element = cp.argmax(cp.max(abs_err, axis=-1))
    elapsed = sync_time(cp, start, "plot_error")
    print_done(elapsed)
    traffic_start = time.perf_counter()
    metrics = float(l2.get()), float(linf.get()), float(avg_max.get()), int(max_element.get())
    record_timing("traffic.device_to_host.error_metrics", time.perf_counter() - traffic_start)
    if not return_plot_samples:
        return metrics
    traffic_start = time.perf_counter()
    plot_samples = {
        "reference_points": cp.asnumpy(ref_points),
        "numerical_values": cp.asnumpy(uh_plot),
        "exact_values_for_error": cp.asnumpy(exact_plot),
    }
    record_timing("traffic.device_to_host.plot_samples", time.perf_counter() - traffic_start)
    return (*metrics, plot_samples)


def plot_sampled_solution_comparison(
        mesh,
        exact_solution: Callable,
        plot_samples: dict[str, np.ndarray],
        *,
        numerical_resolution: int,
        exact_resolution: int | str | None,
        polynomial_order: int | None = None,
        postprocessed_plot_samples: dict[str, np.ndarray] | None = None,
        title: str = "",
        show_mesh: bool = True,
        show: bool = True,
        off_screen: bool = False,
):
    """Plot device-sampled numerical/exact/error data using host plot helpers."""
    reference_points = np.ascontiguousarray(plot_samples["reference_points"], dtype=np.float64)
    numerical_values = np.asarray(plot_samples["numerical_values"], dtype=np.float64)
    exact_values_for_error = np.asarray(plot_samples["exact_values_for_error"], dtype=np.float64)
    absolute_error = np.abs(numerical_values - exact_values_for_error)

    post_reference_points = None
    postprocessed_values = None
    postprocess_error = None
    if postprocessed_plot_samples is not None:
        post_reference_points = np.ascontiguousarray(postprocessed_plot_samples["reference_points"], dtype=np.float64)
        postprocessed_values = np.asarray(postprocessed_plot_samples["postprocessed_values"], dtype=np.float64)
        post_exact_values = np.asarray(postprocessed_plot_samples["exact_values_for_error"], dtype=np.float64)
        postprocess_error = np.abs(postprocessed_values - post_exact_values)

    displayed_error = absolute_error if postprocess_error is None else postprocess_error

    def _error_clim_from_percentile(values: np.ndarray, percentile: float = 95.0) -> tuple[float, float]:
        finite = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 0.0, 1.0
        error_upper = float(np.percentile(finite, percentile))
        if not np.isfinite(error_upper) or error_upper <= 0.0:
            error_upper = 1.0
        return 0.0, error_upper

    error_clim = _error_clim_from_percentile(displayed_error)

    exact_panel_resolution = resolve_exact_plot_resolution(
        exact_resolution,
        numerical_resolution=numerical_resolution,
        num_elements=mesh.num_tri,
    )
    exact_reference_points, _, exact_display_values = sample_callable_on_elements(
        mesh,
        exact_solution,
        resolution=exact_panel_resolution,
    )
    if mesh.num_tri <= 130:
        from hdgfem.io.plot import plot_scalar_sample_panels_matplotlib

        panels = [("Numerical solution", reference_points, numerical_values)]
        if postprocessed_values is not None:
            panels.append(("Postprocessed primal", post_reference_points, postprocessed_values))
        panels.extend(
            [
                ("Exact solution", exact_reference_points, exact_display_values, {"show_mesh": False}),
                (
                    "Absolute error" if postprocess_error is None else "Postprocessed absolute error",
                    reference_points if postprocess_error is None else post_reference_points,
                    displayed_error,
                    {"cmap": "magma", "zero_min": True},
                ),
            ]
        )
        return plot_scalar_sample_panels_matplotlib(
            mesh,
            panels,
            suptitle=title or None,
            show_mesh=show_mesh,
            cmap="jet",
            levels=matplotlib_contour_levels(
                (int(polynomial_order) + (1 if postprocessed_values is not None else 0))
                if polynomial_order is not None
                else max((int(numerical_resolution) - 3) // 2, 0)
            ),
            share_clim=False,
            show=show,
        )

    pv = _require_pyvista()

    panel_count = 4 if postprocessed_values is not None else 3
    plotter = pv.Plotter(shape=(1, panel_count), window_size=[600 * panel_count, 650], off_screen=off_screen)
    scalar_bar_args = {
        "vertical": False,
        "width": 0.55,
        "height": 0.08,
        "position_x": 0.225,
        "position_y": 0.02,
    }
    panels = [("Numerical solution", reference_points, numerical_values, None, "viridis", True)]
    if postprocessed_values is not None:
        panels.append(("Postprocessed primal", post_reference_points, postprocessed_values, None, "viridis", True))
    panels.extend(
        [
            ("Exact solution", exact_reference_points, exact_display_values, None, "viridis", False),
            (
                "Absolute error" if postprocess_error is None else "Postprocessed absolute error",
                reference_points if postprocess_error is None else post_reference_points,
                displayed_error,
                error_clim,
                "magma",
                True,
            ),
        ]
    )
    for column, (panel_title, panel_reference_points, values, clim, cmap, panel_show_mesh) in enumerate(panels):
        display_title = panel_title if column != 0 or not title else f"{panel_title}\n{title}"
        add_samples_to_plotter(
            plotter,
            mesh,
            panel_reference_points,
            values,
            scalar_name=f"field_{column}",
            title=display_title,
            subplot=(0, column),
            show_mesh=show_mesh and panel_show_mesh,
            cmap=cmap,
            clim=clim,
            scalar_bar_args=scalar_bar_args,
        )
    plotter.link_views()
    if show:
        plotter.show()
    return plotter


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=RECOMMENDATION_TEXT,
    )
    parser.add_argument("--case", default="trigonometric-poisson", help="case key from scripts/diffusion_reaction/cases.py")
    parser.add_argument("--order", "-o", type=int, default=6)
    parser.add_argument("--mesh-size", "-ms", type=float, default=0.05)
    parser.add_argument("--mesh-type", "-mt", choices=("auto", "disc", "rectangle", "unit-rectangle", "triangle", "lshape", "structured-rectangle"), default="auto")
    parser.add_argument("--disc-radius", type=float, default=None)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--basis", default="dub_orth", choices=("hier_C0", "hierarchical_c0", "bernstein", "dub_orth"))
    parser.add_argument("--volume-quadrature", choices=("auto", "symmetric", "duffy"), default="auto", help="triangle volume quadrature family; --volume-quad-1d forces the legacy Duffy rule")
    parser.add_argument("--volume-quad-1d", type=int, default=None, help="legacy Duffy 1D point count; when set, overrides --volume-quadrature")
    parser.add_argument("--edge-quad-1d", type=int, default=None)
    parser.add_argument("--error-volume-quad-1d", type=int, default=None)
    parser.add_argument("--trace-basis", choices=("legacy-lagrange", "legendre-modal", "bernstein"), default="legacy-lagrange")
    parser.add_argument("--assembly-backend", choices=("cupy", "raw-cuda"), default="cupy")
    parser.add_argument("--raw-matrix-format", choices=("coo", "csr"), default="coo")
    parser.add_argument("--raw-block-size", choices=("auto", "1", "32", "64", "128"), default="auto")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--plot", action="store_true", help="show numerical/exact/error plots after the summary")
    parser.add_argument(
        "--plot-postprocess-primal",
        action="store_true",
        help="include HDG primal postprocessed u_h^* in the solution plot; implies --plot",
    )
    parser.add_argument(
        "--postprocess-backend",
        choices=("auto", "host", "cupy", "raw-cuda"),
        default="auto",
        help="primal postprocess backend for --plot-postprocess-primal; auto follows the assembly backend",
    )
    parser.add_argument("--plot-resolution", "-pr", type=int, default=12, help="plot/error sampling resolution; coarse meshes use a polynomial-degree minimum")
    parser.add_argument(
        "--exact-plot-resolution",
        default="auto",
        help="exact-solution panel resolution: integer, 'auto', or 'same'",
    )
    parser.add_argument("--hide-mesh", action="store_true", help="hide mesh overlay in plots")
    parser.add_argument("--gmsh-verbosity", type=int, default=0)
    parser.add_argument("--gmsh-algorithm", type=int, default=None)
    parser.add_argument(
        "--amgx-config",
        default=None,
        help="Path to a PyAMGX JSON config; omitted means the nodal best Chebyshev/L1 profile",
    )
    parser.add_argument("--amgx-solver", default="PCGF", help="AMGX Krylov solver; overrides the solver entry loaded from --amgx-config")
    parser.add_argument("--amgx-tolerance", type=float, default=1.0e-13)
    parser.add_argument(
        "--scale-system",
        choices=("symmetric", "left", "on", "off"),
        default="off",
        help="diagonal scaling before AMGX upload; 'on' is an alias for legacy left row scaling",
    )
    parser.add_argument("--amgx-maxiter", type=int, default=2000)
    parser.add_argument("--show-cupy-config", action="store_true")
    parser.add_argument("--verbosity", "-v", type=int, choices=(0, 1, 2), default=1)
    return parser


def resolve_primal_postprocess_backend(args) -> str:
    """Resolve the requested primal postprocess backend."""
    backend = str(args.postprocess_backend).lower()
    if backend == "auto":
        return "raw-cuda" if args.assembly_backend == "raw-cuda" else "cupy"
    return backend


def prepare_primal_postprocess_setup(space: DGSpace, cspace, trace_basis: str, backend: str):
    """Prebuild degree p+1 postprocess reference data during setup."""
    backend = str(backend).lower()
    if backend == "none":
        return None
    log("building primal postprocess p+1 reference data ... ", end="")
    cp = require_cupy()
    start = time.perf_counter()
    host_start = time.perf_counter()
    trace_space = space.trace_space(trace_basis)
    cache = _new_hdg_postprocess_cache(space, trace_space)
    record_timing("postprocess.reference_host_setup", time.perf_counter() - host_start)
    if backend in {"cupy", "raw-cuda"}:
        traffic_start = time.perf_counter()
        as_cupy_space(cache.post_space, device=cspace.device_id)
        cp.cuda.get_current_stream().synchronize()
        record_timing("traffic.host_to_device.postprocess_reference", time.perf_counter() - traffic_start)
    elapsed = record_timing("postprocess.reference_setup", time.perf_counter() - start)
    print_done(elapsed)
    return cache


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


def _timing_row(label: str, seconds: float, total_seconds: float) -> tuple[str, str, str]:
    return label, _format_timing_cell(seconds, total_seconds), "s"


def _nonzero_timing_rows(items: list[tuple[str, float]], total_seconds: float) -> list[tuple[str, str, str]]:
    return [
        _timing_row(label, seconds, total_seconds)
        for label, seconds in items
        if abs(float(seconds)) > 0.0
    ]


def print_detailed_timing_table(args, total_seconds: float) -> None:
    """Print backend-specific timing details before the compact solve summary."""
    if int(getattr(args, "verbosity", 1)) < 2:
        return
    effective_backend = str(RUN_METADATA.get("assembly_backend", args.assembly_backend))
    raw_kernel_total = timing_value("assembly.raw.csr_kernel") or timing_value("assembly.raw.kernel")
    csr_total = timing_value("solve.csr_scale")
    sections: list[tuple[str, list[tuple[str, str, str]]]] = []

    setup_rows = _nonzero_timing_rows(
        [
            ("mesh generate", timing_value("mesh.generate")),
            ("space/reference", timing_value("space.setup")),
            ("post p+1 ref", timing_value("postprocess.reference_setup")),
            ("post p+1 host", timing_value("postprocess.reference_host_setup")),
        ],
        total_seconds,
    )
    if setup_rows:
        sections.append(("Setup", setup_rows))

    if effective_backend == "raw-cuda":
        assembly_items = [
            ("coefficient prep", timing_value("assembly.coefficient_prep")),
            ("source prep", timing_value("assembly.coefficient_prep.source")),
            ("reaction prep", timing_value("assembly.coefficient_prep.reaction")),
            ("raw map/setup", timing_value("assembly.raw.map_setup")),
            ("raw csr zero", timing_value("assembly.raw.csr_zero")),
            ("raw kernel", raw_kernel_total),
            ("raw total", timing_value("assembly.raw.total")),
            ("assembly total", timing_value("assembly.total")),
        ]
    else:
        assembly_items = [
            ("index arrays", timing_value("assembly.indices")),
            ("local LHS", timing_value("local.lhs")),
            ("local solve", timing_value("local.solve.assembly")),
            ("trace lift", timing_value("assembly.b_trace")),
            ("trace blocks", timing_value("assembly.trace_blocks")),
            ("global data", timing_value("assembly.data")),
            ("source moments", timing_value("source_moments")),
            ("RHS faces", timing_value("assembly.rhs_faces")),
            ("boundary elim", timing_value("assembly.boundary_elimination")),
            ("assembly total", timing_value("assembly.total")),
        ]
    assembly_rows = _nonzero_timing_rows(assembly_items, total_seconds)
    if assembly_rows:
        sections.append(("Assembly", assembly_rows))

    solve_rows = _nonzero_timing_rows(
        [
            ("CSR/view", csr_total),
            ("diagonal scaling", timing_value("solve.scale")),
            ("AMGX setup/upload", timing_value("solve.amgx_setup")),
            ("AMGX solve", timing_value("solve.amgx_solve")),
            ("solve total", timing_value("solve.total")),
            ("trace augment", timing_value("reconstruct.trace")),
            ("field reconstruct", timing_value("reconstruct.field")),
            ("local recon solve", timing_value("local.solve.reconstruction")),
        ],
        total_seconds,
    )
    if solve_rows:
        sections.append(("Solve / Recon", solve_rows))

    post_rows = _nonzero_timing_rows(
        [
            ("post cupy setup", timing_value("postprocess.primal.cupy.setup")),
            ("post cupy solve", timing_value("postprocess.primal.cupy.solve")),
            ("post cupy total", timing_value("postprocess.primal.cupy.total")),
            ("post cuda setup", timing_value("postprocess.primal.raw_cuda.setup")),
            ("post cuda kernel", timing_value("postprocess.primal.raw_cuda.kernel")),
            ("post cuda total", timing_value("postprocess.primal.raw_cuda.total")),
            ("post host recon", timing_value("postprocess.primal.host_reconstruct")),
            ("post total", timing_value("postprocess.primal")),
        ],
        total_seconds,
    )
    if post_rows:
        sections.append(("HDG Postprocess", post_rows))

    diagnostic_rows = _nonzero_timing_rows(
        [
            ("post errors host", timing_value("diagnostics.postprocess_primal_error_host")),
            ("solution error/sample", timing_value("plot_error")),
        ],
        total_seconds,
    )
    if diagnostic_rows:
        sections.append(("Diagnostics", diagnostic_rows))

    traffic_rows = _nonzero_timing_rows(
        [
            ("H2D space static", timing_value("traffic.host_to_device.space_static")),
            ("H2D maps/trace", timing_value("traffic.host_to_device.maps_trace_reference")),
            ("H2D post p+1 ref", timing_value("traffic.host_to_device.postprocess_reference")),
            ("H2D plot basis", timing_value("traffic.host_to_device.plot_basis")),
            ("D2H post coeffs", timing_value("traffic.device_to_host.postprocess_primal_coeffs")),
            ("D2H plot ref pts", timing_value("traffic.device_to_host.plot_reference_points")),
            ("D2H error metrics", timing_value("traffic.device_to_host.error_metrics")),
            ("D2H plot samples", timing_value("traffic.device_to_host.plot_samples")),
            ("traffic total", timing_prefix_total("traffic.")),
        ],
        total_seconds,
    )
    if traffic_rows:
        sections.append(("Host/Device Traffic", traffic_rows))

    if sections:
        pretty_print_sections(sections, title="HDGFEM CUDA Detailed Timings")


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
    post_ref_total = timing_value("postprocess.reference_setup")
    setup_total = space_total + gpu_setup_total + post_ref_total
    assembly_total = timing_value("assembly.total")
    csr_total = timing_value("solve.csr_scale")
    amgx_setup_total = timing_value("solve.amgx_setup")
    amgx_solve_total = timing_value("solve.amgx_solve")
    solve_total = timing_value("solve.total")
    reconstruct_total = timing_value("reconstruct.trace") + timing_value("reconstruct.field")
    post_host_l2_total = timing_value("diagnostics.postprocess_primal_error_host")
    postprocess_hdg_total = timing_value("postprocess.primal")
    plot_total = timing_value("plot_error")
    traffic_total = timing_prefix_total("traffic.")
    global_dof = int(cspace.mesh.int_edges_inds.size * cspace.edg_dof)
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
            ("volume quad", str(space.quad_data.volume_quadrature), "s"),
            ("raw matrix", args.raw_matrix_format if effective_backend == "raw-cuda" else "n/a", "s"),
            ("raw block", str(args.raw_block_size) if effective_backend == "raw-cuda" else "n/a", "s"),
            ("postprocess", str(RUN_METADATA.get("postprocess_backend", "none")), "s"),
            (
                "post raw block",
                str(RUN_METADATA.get("postprocess_raw_cuda_block_size", "n/a"))
                if RUN_METADATA.get("postprocess_backend") == "raw-cuda"
                else "n/a",
                "s",
            ),
            ("verbosity", args.verbosity, ",d"),
        ]
    )
    mesh_details = [
        ("order", args.order, ",d"),
        ("mesh size", args.mesh_size, ".4f"),
        ("h", mesh.h, ".3e"),
        ("triangles", mesh.num_tri, ",d"),
        ("edges", mesh.num_edg, ",d"),
        ("interior edges", int(mesh.int_edges_inds.size), ",d"),
        ("global dof", global_dof, ",d"),
        ("element dof", int(cspace.el_dof), ",d"),
        ("edge dof", int(cspace.edg_dof), ",d"),
        ("volume quad pts", int(space.quad_data.Krf_w.size), ",d"),
        ("volume quad 1d", _display_optional_int(args.volume_quad_1d), "s"),
        ("error quad 1d", _display_optional_int(args.error_volume_quad_1d), "s"),
    ]
    solver_details = [
        ("config", _display_amgx_config(), "s"),
        ("solver", str(RUN_METADATA.get("amgx_solver", args.amgx_solver)), "s"),
        ("tol", args.amgx_tolerance, ".1e"),
        ("maxiter", args.amgx_maxiter, ",d"),
        ("iterations", _display_optional_int(RUN_METADATA.get("amgx_iterations")), "s"),
        ("matrix input", str(RUN_METADATA.get("solve_matrix_format", "unknown")), "s"),
        ("scaling", str(RUN_METADATA.get("solve_scale_system", "unknown")), "s"),
        ("matrix nnz", int(RUN_METADATA.get("matrix_nnz", 0)), ",d"),
        ("solver residual", float(RUN_METADATA.get("amgx_solver_rel_residual", float("nan"))), ".3e"),
        ("physical residual", float(RUN_METADATA.get("amgx_physical_rel_residual", float("nan"))), ".3e"),
    ]
    errors = [
        ("theory h^(p+1)", mesh.h ** (args.order + 1), ".3e"),
        ("L2", l2, ".3e"),
        ("Linf", linf, ".3e"),
    ]
    if "postprocess_primal_l2" in RUN_METADATA:
        errors.append(("post primal L2", float(RUN_METADATA["postprocess_primal_l2"]), ".3e"))
    if "postprocess_primal_linf" in RUN_METADATA:
        errors.append(("post primal Linf", float(RUN_METADATA["postprocess_primal_linf"]), ".3e"))
    errors.extend(
        [
            ("avg max", avg_max, ".3e"),
            ("max element", max_element, ",d"),
        ]
    )
    timings = [
        ("mesh", _format_timing_cell(mesh_total, total_seconds), "s"),
        ("setup", _format_timing_cell(setup_total, total_seconds), "s"),
        ("assembly", _format_timing_cell(assembly_total, total_seconds), "s"),
        ("global solve", _format_timing_cell(solve_total, total_seconds), "s"),
        ("reconstruction", _format_timing_cell(reconstruct_total, total_seconds), "s"),
        ("HDG postprocess", _format_timing_cell(postprocess_hdg_total, total_seconds), "s"),
        ("total measured", f"{total_seconds:.3f}s", "s"),
    ]

    sections = [
        ("Run / Options", run_options),
        ("Mesh / DOF", mesh_details),
        ("Solver", solver_details),
        ("Errors", errors),
        ("Timings", timings),
    ]
    pretty_print_sections(sections, title="HDGFEM CUDA Diffusion-Reaction Solve Summary")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.assembly_backend == "raw-cuda":
        args.raw_block_size = resolve_raw_cuda_block_size(
            args.raw_block_size,
            equation="diffusion-reaction",
            order=args.order,
        )
    postprocess_backend = resolve_primal_postprocess_backend(args) if args.plot_postprocess_primal else "none"
    if args.plot_postprocess_primal:
        args.plot = True
    set_verbosity(args.verbosity)
    TIMINGS.clear()
    RUN_METADATA.clear()
    cp = require_cupy()
    require_cupyx_sparse()
    require_pyamgx()
    RUN_METADATA["postprocess_backend"] = postprocess_backend
    # Match the legacy gpu_v4 runner: do not let CuPy keep freed blocks in its
    # memory pool while PyAMGX reports device memory usage.
    cp.cuda.set_allocator(None)
    cp.cuda.set_pinned_memory_allocator(None)
    if args.show_cupy_config:
        cp.show_config()

    run_start = time.perf_counter()
    log("\n----- Standalone HDGFEM CUDA-Style Diffusion-Reaction Solve -----")
    log(
        f"case={args.case}, mesh={args.mesh_type}, mesh_size={args.mesh_size:g}, "
        f"order={args.order}, basis={args.basis}, trace_basis={args.trace_basis}, "
        f"volume_quadrature={args.volume_quadrature}, volume_quad_1d={args.volume_quad_1d}, "
        f"assembly_backend={args.assembly_backend}, "
        f"raw_matrix_format={args.raw_matrix_format}, raw_block_size={args.raw_block_size}"
    )
    print_recommended_profiles(args)
    if case_definition_by_key is None:
        raise ModuleNotFoundError("Case definitions are unavailable in this repository checkout")

    case = case_definition_by_key(args.case)
    problem = case.build()
    diffusion, reaction, source, exact = problem
    if diffusion != (1.0, 0.0, 1.0):
        raise NotImplementedError("this CUDA runner currently supports identity diffusion only")

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
        volume_quadrature=args.volume_quadrature,
        volume_quad_1d=args.volume_quad_1d,
        edge_quad_1d=args.edge_quad_1d,
    )
    space_time = record_timing("space.setup", time.perf_counter() - start)
    print_done(space_time)
    plot_resolution = effective_plot_resolution(args.plot_resolution, space.order, mesh.num_tri)
    log(f"h={mesh.h:.2e}, h^(p+1)={mesh.h ** (args.order + 1):.2e}, triangles={mesh.num_tri:,}")

    log("copying static data to GPU ... ", end="")
    start = time.perf_counter()
    traffic_start = time.perf_counter()
    cspace = as_cupy_space(space)
    cp.cuda.get_current_stream().synchronize()
    record_timing("traffic.host_to_device.space_static", time.perf_counter() - traffic_start)
    traffic_start = time.perf_counter()
    maps = build_dof_maps(cspace)
    trace_ref = build_trace_reference(cspace, args.trace_basis)
    cp.cuda.get_current_stream().synchronize()
    record_timing("traffic.host_to_device.maps_trace_reference", time.perf_counter() - traffic_start)
    gpu_time = record_timing("gpu.setup", time.perf_counter() - start)
    print_done(gpu_time)

    postprocess_cache = None
    if args.plot_postprocess_primal:
        postprocess_cache = prepare_primal_postprocess_setup(space, cspace, args.trace_basis, postprocess_backend)

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
    local_unknowns_device = None
    needs_device_local_unknowns = args.plot_postprocess_primal and postprocess_backend in {"cupy", "raw-cuda"}
    if raw_assembly is not None:
        if needs_device_local_unknowns:
            uh, local_unknowns_device = reconstruct_field_raw_cuda(
                trace,
                raw_assembly,
                cspace,
                trace_ref,
                args.tau,
                args.raw_block_size,
                return_local_unknowns=True,
            )
        else:
            uh = reconstruct_field_raw_cuda(trace, raw_assembly, cspace, trace_ref, args.tau, args.raw_block_size)
    else:
        if args.plot_postprocess_primal:
            uh, local_unknowns_device = reconstruct_field(
                trace,
                local_lhs,
                element_boundary,
                source_rhs,
                cspace,
                trace_ref,
                return_local_unknowns=True,
            )
        else:
            uh = reconstruct_field(trace, local_lhs, element_boundary, source_rhs, cspace, trace_ref)

    postprocessed_plot_samples = None
    if args.plot_postprocess_primal:
        postprocessed_field = postprocess_primal_field_for_plot(
            trace=trace,
            local_unknowns_device=local_unknowns_device,
            source=source,
            reaction=reaction,
            exact=exact,
            space=space,
            trace_basis=args.trace_basis,
            diffusion=diffusion,
            tau=args.tau,
            backend=postprocess_backend,
            cspace=cspace,
            raw_block_size=args.raw_block_size,
            postprocess_cache=postprocess_cache,
        )
        postprocessed_plot_samples = sample_postprocessed_primal_for_plot(
            postprocessed_field,
            exact,
            resolution=postprocess_plot_resolution(plot_resolution, space.order, mesh.num_tri),
        )

    error_result = evaluate_errors(
        uh,
        exact,
        cspace,
        plot_resolution,
        error_volume_quad_1d=args.error_volume_quad_1d,
        return_plot_samples=args.plot,
    )
    if args.plot:
        l2, linf, avg_max, max_element, plot_samples = error_result
    else:
        l2, linf, avg_max, max_element = error_result
        plot_samples = None

    total = time.perf_counter() - run_start
    print_detailed_timing_table(args, total)
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

    if args.plot:
        log("plotting solution comparison ... ", end="")
        plot_start = time.perf_counter()
        plot_title = f"{args.case}, p={space.order}, elements={mesh.num_tri:,}, L2={l2:.2e}"
        plot_sampled_solution_comparison(
            mesh,
            exact,
            plot_samples,
            numerical_resolution=plot_resolution,
            exact_resolution=args.exact_plot_resolution,
            polynomial_order=space.order,
            postprocessed_plot_samples=postprocessed_plot_samples,
            title=plot_title,
            show_mesh=not args.hide_mesh,
        )
        print_done(time.perf_counter() - plot_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
