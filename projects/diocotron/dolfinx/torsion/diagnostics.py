#!/usr/bin/env python3
"""Measure variation of ``phi_T`` around the torsion level set ``T=c1``.

For each requested geometry this script solves

    -Delta T = 1,                 T = 0 on the boundary,
    -Delta phi_T = rho_T(T),      phi_T = 0 on the boundary,

with the same mesh adaptation, quadrature, and optional mollified density as
``plot_torsion_phi_t_pyvista.py``.  A tessellated contour supplies an initial
level-set curve.  Dense arc-length samples are then Newton projected onto the
degree-p finite-element level set and degree-(p-1) discontinuous projections
of the exact FE gradients are evaluated at those points.

The primary oscillation amplitude is ``max(phi_T)-min(phi_T)`` on ``T=c1``.
The script also reports its relative value, total variation from both ordered
samples and ``integral |d phi_T/ds| ds``, the corresponding effective
full-oscillation counts ``TV/(2*amplitude)``, and normal-derivative variation.
JSON and CSV outputs contain all metrics and discretization details needed to
reproduce a run.
"""

from __future__ import annotations

# Support direct execution as well as python -m from a checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

import argparse
import csv
import gc
import json
import math
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hdgfem_torsion_oscillations_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/hdgfem_torsion_oscillations_cache")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import basix
import numpy as np
import pyvista as pv
import ufl
from dolfinx import fem, geometry
from mpi4py import MPI

from projects.diocotron.dolfinx.plotting.torsion import (
    DEFAULT_MESH_SIZES,
    PACMAN_MOUTH_HALF_ANGLE,
    PACMAN_RADIUS,
    PACMAN_TIP_TO_BACK_DISTANCE,
    PACMAN_TIP_X,
    band_cell_mask,
    generate_mesh,
    high_order_grid,
    refine_band_mesh,
    solve_fields,
    solve_torsion,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = REPO_ROOT / "projects/diocotron/runs" / "torsion_phi_t_oscillations_p6.json"
REQUESTED_GEOMETRIES = ("iter", "star", "horseshoe", "convex-polygon", "pacman")


def gradient_function(function: fem.Function) -> fem.Function:
    """Represent the exact cellwise FE gradient in a discontinuous space."""
    degree = function.function_space.element.basix_element.degree
    value_size = function.function_space.mesh.geometry.dim
    space = fem.functionspace(
        function.function_space.mesh,
        ("DG", max(degree - 1, 0), (value_size,)),
    )
    result = fem.Function(space)
    expression = fem.Expression(ufl.grad(function), space.element.interpolation_points)
    result.interpolate(expression)
    result.x.scatter_forward()
    return result


def evaluate(function: fem.Function, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a scalar or vector FE function and return values and cell ids."""
    domain = function.function_space.mesh
    points = np.zeros((len(points_xy), 3), dtype=np.float64)
    points[:, :2] = points_xy
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    collisions = geometry.compute_colliding_cells(domain, candidates, points)
    cells = np.full(len(points), -1, dtype=np.int32)
    for point_index in range(len(points)):
        links = collisions.links(point_index)
        if len(links):
            cells[point_index] = int(links[0])
    valid = cells >= 0
    value_size = function.function_space.value_size
    values = np.full((len(points), value_size), np.nan, dtype=np.float64)
    if np.any(valid):
        raw = function.eval(points[valid], cells[valid])
        values[valid] = np.asarray(raw, dtype=np.float64).reshape(-1, value_size)
    return values, cells


def evaluate_in_cells(
    function: fem.Function, points_xy: np.ndarray, cells: np.ndarray
) -> np.ndarray:
    """Evaluate using prescribed cells, ensuring consistent traces on interfaces."""
    points = np.zeros((len(points_xy), 3), dtype=np.float64)
    points[:, :2] = points_xy
    raw = function.eval(points, cells)
    return np.asarray(raw, dtype=np.float64).reshape(
        len(points), function.function_space.value_size
    )


def line_components(polyline: pv.PolyData) -> list[np.ndarray]:
    """Decode VTK line cells into ordered point arrays."""
    encoded = np.asarray(polyline.lines, dtype=np.int64)
    components: list[np.ndarray] = []
    offset = 0
    while offset < len(encoded):
        count = int(encoded[offset])
        ids = encoded[offset + 1 : offset + 1 + count]
        offset += count + 1
        if count >= 2:
            points = np.asarray(polyline.points[ids, :2], dtype=np.float64)
            if np.linalg.norm(points[-1] - points[0]) > 1.0e-12:
                points = np.vstack((points, points[0]))
            components.append(points)
    return components


def resample_closed_components(
    components: list[np.ndarray], total_points: int
) -> tuple[np.ndarray, np.ndarray, list[slice]]:
    """Resample closed polylines approximately uniformly in arc length."""
    lengths = np.asarray(
        [np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) for points in components],
        dtype=np.float64,
    )
    if not len(lengths) or np.any(lengths <= 0.0):
        raise RuntimeError("T=c1 contour did not contain a usable closed component")
    counts = np.maximum(512, np.rint(total_points * lengths / np.sum(lengths)).astype(int))
    sampled: list[np.ndarray] = []
    component_ids: list[int] = []
    slices: list[slice] = []
    start = 0
    for component_id, (points, count) in enumerate(zip(components, counts, strict=True)):
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        targets = np.linspace(0.0, cumulative[-1], int(count), endpoint=False)
        segment_ids = np.minimum(
            np.searchsorted(cumulative, targets, side="right") - 1,
            len(segment_lengths) - 1,
        )
        fractions = (targets - cumulative[segment_ids]) / segment_lengths[segment_ids]
        values = points[segment_ids] + fractions[:, None] * (
            points[segment_ids + 1] - points[segment_ids]
        )
        sampled.append(values)
        component_ids.extend([component_id] * len(values))
        slices.append(slice(start, start + len(values)))
        start += len(values)
    return np.vstack(sampled), np.asarray(component_ids, dtype=np.int32), slices


def project_to_level_set(
    points: np.ndarray,
    torsion: fem.Function,
    grad_t: fem.Function,
    c1: float,
    iterations: int,
) -> tuple[np.ndarray, float]:
    """Apply safeguarded normal Newton corrections to place samples on ``T=c1``."""
    projected = points.copy()
    for _ in range(iterations):
        t_values, t_cells = evaluate(torsion, projected)
        valid = t_cells >= 0
        if not np.all(valid):
            raise RuntimeError(
                f"level-set projection lost {np.count_nonzero(~valid)} of {len(valid)} samples"
            )
        # T and its discontinuous gradient must use the same cell trace when a
        # point lies on an element interface.
        gradients = evaluate_in_cells(grad_t, projected, t_cells)
        gradient_norm_sq = np.sum(gradients * gradients, axis=1)
        valid &= gradient_norm_sq > 100.0 * np.finfo(float).eps
        if not np.all(valid):
            raise RuntimeError(
                f"level-set projection lost {np.count_nonzero(~valid)} of {len(valid)} samples"
            )
        residual = t_values[:, 0] - c1
        active = np.abs(residual) > 1.0e-14 * max(1.0, abs(c1))
        if not np.any(active):
            break
        correction = residual / gradient_norm_sq
        pending = np.flatnonzero(active)
        scales = np.ones(len(projected), dtype=np.float64)
        for _backtrack in range(14):
            candidates = projected[pending] - (
                scales[pending] * correction[pending]
            )[:, None] * gradients[pending]
            candidate_values, candidate_cells = evaluate(torsion, candidates)
            candidate_residual = np.abs(candidate_values[:, 0] - c1)
            accepted = (candidate_cells >= 0) & (
                candidate_residual <= np.abs(residual[pending]) * (1.0 + 1.0e-12)
            )
            if np.any(accepted):
                projected[pending[accepted]] = candidates[accepted]
            pending = pending[~accepted]
            if not len(pending):
                break
            scales[pending] *= 0.5
        if len(pending):
            raise RuntimeError(
                f"level-set projection could not improve {len(pending)} of "
                f"{len(projected)} samples after backtracking"
            )
    residual, cells = evaluate(torsion, projected)
    if np.any(cells < 0):
        raise RuntimeError("projected level-set samples left the mesh")
    return projected, float(np.max(np.abs(residual[:, 0] - c1)))


def quadrature_weights(points: np.ndarray, slices: list[slice]) -> tuple[np.ndarray, float]:
    """Return periodic trapezoidal arc-length weights for each component."""
    weights = np.empty(len(points), dtype=np.float64)
    total_length = 0.0
    for component_slice in slices:
        component = points[component_slice]
        forward = np.linalg.norm(np.roll(component, -1, axis=0) - component, axis=1)
        backward = np.roll(forward, 1)
        weights[component_slice] = 0.5 * (forward + backward)
        total_length += float(np.sum(forward))
    return weights, total_length


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.dot(values, weights) / np.sum(weights))


def weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    return math.sqrt(float(np.dot(values * values, weights) / np.sum(weights)))


def weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    """Compute weighted empirical quantiles with sorted cumulative weights."""
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return np.interp(probabilities, cumulative, sorted_values)


def sample_outer_region_gradients(
    torsion: fem.Function,
    phi_t: fem.Function,
    c1: float,
    *,
    quadrature_degree: int,
    cell_chunk_size: int = 1024,
) -> dict[str, object]:
    """Sample exact FE gradients throughout the region ``T<c1``.

    Quadrature points are used as a deterministic, element-aware fine grid.
    The returned fractions are area weighted.  Since the torsion solution is
    positive in the continuous problem, ``T<c1`` is the region between the
    boundary and the requested level set (up to discretization error).
    """
    domain = torsion.function_space.mesh
    grad_t = gradient_function(torsion)
    grad_phi = gradient_function(phi_t)
    reference_points, reference_weights = basix.make_quadrature(
        basix.CellType.triangle, quadrature_degree
    )
    geometry_dofmap = np.asarray(domain.geometry.dofmaps[0], dtype=np.int32)
    if geometry_dofmap.shape[1] != 3:
        raise RuntimeError("outer-region sampler currently expects linear triangles")
    num_cells = domain.topology.index_map(domain.topology.dim).size_local

    point_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    grad_t_chunks: list[np.ndarray] = []
    grad_phi_chunks: list[np.ndarray] = []
    t_chunks: list[np.ndarray] = []
    for cell_start in range(0, num_cells, cell_chunk_size):
        cell_stop = min(cell_start + cell_chunk_size, num_cells)
        cell_ids = np.arange(cell_start, cell_stop, dtype=np.int32)
        vertices = np.asarray(
            domain.geometry.x[geometry_dofmap[cell_ids], :2], dtype=np.float64
        )
        jacobian_0 = vertices[:, 1] - vertices[:, 0]
        jacobian_1 = vertices[:, 2] - vertices[:, 0]
        physical = (
            vertices[:, None, 0]
            + reference_points[None, :, 0, None] * jacobian_0[:, None]
            + reference_points[None, :, 1, None] * jacobian_1[:, None]
        )
        determinant = np.abs(
            jacobian_0[:, 0] * jacobian_1[:, 1]
            - jacobian_0[:, 1] * jacobian_1[:, 0]
        )
        points = np.ascontiguousarray(physical.reshape(-1, 2))
        point_cells = np.repeat(cell_ids, len(reference_points))
        t_values = evaluate_in_cells(torsion, points, point_cells)[:, 0]
        in_outer_region = t_values < c1
        if not np.any(in_outer_region):
            continue
        weights = (determinant[:, None] * reference_weights[None, :]).reshape(-1)
        point_chunks.append(points[in_outer_region])
        weight_chunks.append(weights[in_outer_region])
        grad_t_chunks.append(
            evaluate_in_cells(grad_t, points, point_cells)[in_outer_region]
        )
        grad_phi_chunks.append(
            evaluate_in_cells(grad_phi, points, point_cells)[in_outer_region]
        )
        t_chunks.append(t_values[in_outer_region])

    if not point_chunks:
        raise RuntimeError("no quadrature samples were found in T<c1")
    points = np.concatenate(point_chunks)
    weights = np.concatenate(weight_chunks)
    t_values = np.concatenate(t_chunks)
    t_gradients = np.concatenate(grad_t_chunks)
    phi_gradients = np.concatenate(grad_phi_chunks)
    phi_gradient_norm = np.linalg.norm(phi_gradients, axis=1)
    t_gradient_norm = np.linalg.norm(t_gradients, axis=1)
    inward_derivative = np.sum(phi_gradients * t_gradients, axis=1)
    normalized_inward_derivative = inward_derivative / t_gradient_norm
    scale_norm = float(np.max(phi_gradient_norm))
    scale_dot = float(np.max(np.abs(inward_derivative)))
    norm_tolerance = 1.0e-10 * max(scale_norm, np.finfo(float).eps)
    dot_tolerance = 1.0e-10 * max(scale_dot, np.finfo(float).eps)
    probabilities = np.array([0.0, 0.001, 0.01, 0.05, 0.50])
    norm_quantiles = weighted_quantiles(phi_gradient_norm, weights, probabilities)
    dot_quantiles = weighted_quantiles(inward_derivative, weights, probabilities)
    normalized_quantiles = weighted_quantiles(
        normalized_inward_derivative, weights, probabilities
    )
    min_norm_id = int(np.argmin(phi_gradient_norm))
    min_dot_id = int(np.argmin(inward_derivative))
    total_weight = float(np.sum(weights))

    def area_fraction(mask: np.ndarray) -> float:
        return float(np.sum(weights[mask]) / total_weight)

    return {
        "outer_region_definition": "quadrature samples with T<c1",
        "outer_region_quadrature_degree": quadrature_degree,
        "outer_region_sample_points": len(points),
        "outer_region_sampled_area": total_weight,
        "outer_region_sampled_T_min": float(np.min(t_values)),
        "outer_region_sampled_T_max": float(np.max(t_values)),
        "outer_region_grad_phi_norm_min": float(phi_gradient_norm[min_norm_id]),
        "outer_region_grad_phi_norm_quantiles_0_0p1_1_5_50pct": norm_quantiles.tolist(),
        "outer_region_grad_phi_norm_min_point": points[min_norm_id].tolist(),
        "outer_region_grad_phi_norm_numerical_zero_tolerance": norm_tolerance,
        "outer_region_grad_phi_norm_area_fraction_le_tolerance": area_fraction(
            phi_gradient_norm <= norm_tolerance
        ),
        "outer_region_grad_phi_dot_grad_T_min": float(inward_derivative[min_dot_id]),
        "outer_region_grad_phi_dot_grad_T_quantiles_0_0p1_1_5_50pct": dot_quantiles.tolist(),
        "outer_region_grad_phi_dot_grad_T_min_point": points[min_dot_id].tolist(),
        "outer_region_grad_phi_dot_grad_T_sign_tolerance": dot_tolerance,
        "outer_region_grad_phi_dot_grad_T_area_fraction_le_zero": area_fraction(
            inward_derivative <= 0.0
        ),
        "outer_region_grad_phi_dot_grad_T_area_fraction_lt_negative_tolerance": area_fraction(
            inward_derivative < -dot_tolerance
        ),
        "outer_region_inward_unit_derivative_min": float(
            np.min(normalized_inward_derivative)
        ),
        "outer_region_inward_unit_derivative_quantiles_0_0p1_1_5_50pct": (
            normalized_quantiles.tolist()
        ),
    }


def measure_level_set(
    torsion: fem.Function,
    phi_t: fem.Function,
    c1: float,
    c2: float,
    rho_epsilon: float | None,
    *,
    contour_subdivisions: int,
    sample_points: int,
    projection_iterations: int,
) -> dict[str, float | int]:
    """Extract, project, and measure the ``T=c1`` contour."""
    grid = high_order_grid(
        torsion,
        phi_t,
        c1,
        c2,
        rho_epsilon,
        contour_subdivisions,
    )
    contour = grid.contour(isosurfaces=[c1], scalars="T").strip(join=True)
    components = line_components(contour)
    points, _, slices = resample_closed_components(components, sample_points)

    grad_t = gradient_function(torsion)
    grad_phi = gradient_function(phi_t)
    points, max_level_residual = project_to_level_set(
        points,
        torsion,
        grad_t,
        c1,
        projection_iterations,
    )
    phi_values, phi_cells = evaluate(phi_t, points)
    t_gradients, t_gradient_cells = evaluate(grad_t, points)
    phi_gradients, phi_gradient_cells = evaluate(grad_phi, points)
    if np.any((phi_cells < 0) | (t_gradient_cells < 0) | (phi_gradient_cells < 0)):
        raise RuntimeError("failed to evaluate one or more projected contour samples")

    weights, contour_length = quadrature_weights(points, slices)
    phi_values = phi_values[:, 0]
    t_gradient_norm = np.linalg.norm(t_gradients, axis=1)
    unit_normal = t_gradients / t_gradient_norm[:, None]
    unit_tangent = np.column_stack((-unit_normal[:, 1], unit_normal[:, 0]))
    tangent_derivative = np.sum(phi_gradients * unit_tangent, axis=1)
    normal_derivative = np.sum(phi_gradients * unit_normal, axis=1)

    phi_min = float(np.min(phi_values))
    phi_max = float(np.max(phi_values))
    phi_mean = weighted_mean(phi_values, weights)
    amplitude = phi_max - phi_min
    gradient_total_variation = float(np.dot(np.abs(tangent_derivative), weights))
    sample_total_variation = 0.0
    for component_slice in slices:
        component_phi = phi_values[component_slice]
        sample_total_variation += float(
            np.sum(np.abs(np.roll(component_phi, -1) - component_phi))
        )
    denominator = max(abs(phi_mean), np.finfo(float).eps)
    gradient_effective_count = (
        gradient_total_variation / (2.0 * amplitude) if amplitude > 0.0 else 0.0
    )
    sample_effective_count = (
        sample_total_variation / (2.0 * amplitude) if amplitude > 0.0 else 0.0
    )
    normal_mean = weighted_mean(normal_derivative, weights)
    normal_centered = normal_derivative - normal_mean
    return {
        "contour_components": len(components),
        "sample_points": len(points),
        "contour_length": contour_length,
        "max_abs_T_minus_c1": max_level_residual,
        "phi_min": phi_min,
        "phi_max": phi_max,
        "phi_mean": phi_mean,
        "phi_peak_to_peak": amplitude,
        "phi_peak_to_peak_relative_to_abs_mean": amplitude / denominator,
        "phi_centered_rms": weighted_rms(phi_values - phi_mean, weights),
        "phi_tangential_derivative_rms": weighted_rms(tangent_derivative, weights),
        "phi_total_variation_from_gradient": gradient_total_variation,
        "phi_total_variation_from_samples": sample_total_variation,
        "effective_full_oscillations_from_gradient": gradient_effective_count,
        "effective_full_oscillations_from_samples": sample_effective_count,
        "normal_derivative_mean": normal_mean,
        "normal_derivative_min": float(np.min(normal_derivative)),
        "normal_derivative_max": float(np.max(normal_derivative)),
        "normal_derivative_peak_to_peak": float(np.ptp(normal_derivative)),
        "normal_derivative_centered_rms": weighted_rms(normal_centered, weights),
        "min_abs_grad_T": float(np.min(t_gradient_norm)),
    }


def solve_and_measure(name: str, args: argparse.Namespace) -> dict[str, object]:
    mesh_size = (args.mesh_size or DEFAULT_MESH_SIZES[name]) * args.mesh_scale
    bulk_factor = args.bulk_coarsening_factor or float(2**args.band_refinements)
    quadrature_degree = args.quadrature_degree or max(2 * args.order + 8, 20)
    band_quadrature_degree = args.band_quadrature_degree or max(
        quadrature_degree + 12, 4 * args.order + 20
    )
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"torsion_osc_{name}_") as temp_dir:
        mesh_path = Path(temp_dir) / f"{name}.msh"
        generate_mesh(name, mesh_size * bulk_factor, mesh_path, args.gmsh_verbosity)
        from projects.diocotron.dolfinx.torsion.equilibrium.window_fit import read_mesh_with_meshio

        domain = read_mesh_with_meshio(mesh_path, MPI.COMM_WORLD)
        refinement_history: list[dict[str, int]] = []
        for refinement in range(args.band_refinements):
            preview_t, preview_c1, preview_c2, _, _, _ = solve_torsion(
                domain,
                args.order,
                args.alpha1,
                args.alpha2,
                quadrature_degree,
                prefix=f"torsion_osc_{name}_adapt_{refinement}_",
            )
            old_cells = domain.topology.index_map(domain.topology.dim).size_global
            domain, marked_cells = refine_band_mesh(
                domain,
                preview_t,
                preview_c1,
                preview_c2,
                args.band_padding_fraction,
            )
            new_cells = domain.topology.index_map(domain.topology.dim).size_global
            refinement_history.append(
                {"marked_cells": marked_cells, "cells_before": old_cells, "cells_after": new_cells}
            )
            print(
                f"geometry={name} refinement={refinement + 1} "
                f"marked={marked_cells} cells={old_cells}->{new_cells}",
                flush=True,
            )
            del preview_t
            if marked_cells == 0:
                break

        torsion, phi_t, c1, c2, band_mask, rho_epsilon = solve_fields(
            domain,
            args.order,
            args.alpha1,
            args.alpha2,
            quadrature_degree,
            band_quadrature_degree,
            args.band_padding_fraction,
            args.mollify_density,
            args.rho_epsilon_ratio,
        )
        metrics = measure_level_set(
            torsion,
            phi_t,
            c1,
            c2,
            rho_epsilon,
            contour_subdivisions=args.contour_subdivisions,
            sample_points=args.sample_points,
            projection_iterations=args.projection_iterations,
        )
        outer_region_metrics = sample_outer_region_gradients(
            torsion,
            phi_t,
            c1,
            quadrature_degree=args.outer_region_quadrature_degree,
        )
        cells = domain.topology.index_map(domain.topology.dim).size_global
        dofs = torsion.function_space.dofmap.index_map.size_global
        result: dict[str, object] = {
            "geometry": name,
            "geometry_parameters": (
                {
                    "radius": PACMAN_RADIUS,
                    "mouth_half_angle_radians": PACMAN_MOUTH_HALF_ANGLE,
                    "mouth_tip_x": PACMAN_TIP_X,
                    "tip_to_back_distance": PACMAN_TIP_TO_BACK_DISTANCE,
                }
                if name == "pacman"
                else None
            ),
            "order": args.order,
            "mesh_size": mesh_size,
            "bulk_mesh_size": mesh_size * bulk_factor,
            "cells": cells,
            "dofs": dofs,
            "alpha1": args.alpha1,
            "alpha2": args.alpha2,
            "c1": c1,
            "c2": c2,
            "T_max": c2 / args.alpha2,
            "phi_T_domain_max": float(np.max(phi_t.x.array.real)),
            "density_mode": "mollified" if rho_epsilon is not None else "crisp",
            "rho_epsilon": rho_epsilon,
            "band_cells": int(np.count_nonzero(band_mask)),
            "quadrature_degree": quadrature_degree,
            "band_quadrature_degree": band_quadrature_degree,
            "band_refinements": len(refinement_history),
            "refinement_history": refinement_history,
            "contour_subdivisions": args.contour_subdivisions,
            "projection_iterations_requested": args.projection_iterations,
            "elapsed_seconds": time.perf_counter() - started,
            **metrics,
            # Keep a structured copy in JSON for discoverability.  The flat
            # keys below remain part of the CSV/backward-compatible schema.
            "outer_region_gradient_sampling": outer_region_metrics,
            **outer_region_metrics,
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry",
        choices=("all", *REQUESTED_GEOMETRIES),
        default="all",
        help="geometry to measure (default: all supported measurement geometries)",
    )
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--mesh-size", type=float, default=None)
    parser.add_argument(
        "--mesh-scale",
        type=float,
        default=1.0,
        help="multiply each geometry's target mesh size by this factor (default: 1)",
    )
    parser.add_argument("--bulk-coarsening-factor", type=float, default=None)
    parser.add_argument("--band-refinements", type=int, default=2)
    parser.add_argument("--band-padding-fraction", type=float, default=0.35)
    parser.add_argument("--alpha1", type=float, default=0.60)
    parser.add_argument("--alpha2", type=float, default=0.70)
    parser.add_argument("--quadrature-degree", type=int, default=None)
    parser.add_argument("--band-quadrature-degree", type=int, default=None)
    parser.add_argument(
        "--mollify-density",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="match the plotting script's default mollified rho_T (default: enabled)",
    )
    parser.add_argument("--rho-epsilon-ratio", type=float, default=0.08)
    parser.add_argument("--contour-subdivisions", type=int, default=3)
    parser.add_argument("--sample-points", type=int, default=40000)
    parser.add_argument("--projection-iterations", type=int, default=12)
    parser.add_argument(
        "--outer-region-quadrature-degree",
        type=int,
        default=16,
        help="quadrature-grid degree for sampling gradients in T<c1 (default: 16)",
    )
    parser.add_argument("--gmsh-verbosity", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("this diagnostic currently requires one MPI rank")
    if args.order < 2:
        raise ValueError("--order must be at least 2 for gradient measurement")
    if args.mesh_size is not None and args.mesh_size <= 0.0:
        raise ValueError("--mesh-size must be positive")
    if args.mesh_scale <= 0.0:
        raise ValueError("--mesh-scale must be positive")
    if args.bulk_coarsening_factor is not None and args.bulk_coarsening_factor < 1.0:
        raise ValueError("--bulk-coarsening-factor must be at least 1")
    if args.band_refinements < 0:
        raise ValueError("--band-refinements must be nonnegative")
    if not 0.0 < args.alpha1 < args.alpha2 <= 1.0:
        raise ValueError("require 0 < --alpha1 < --alpha2 <= 1")
    if args.sample_points < 1000:
        raise ValueError("--sample-points must be at least 1000")
    if args.contour_subdivisions < 1:
        raise ValueError("--contour-subdivisions must be positive")
    if args.projection_iterations < 1:
        raise ValueError("--projection-iterations must be positive")
    if args.outer_region_quadrature_degree < 1:
        raise ValueError("--outer-region-quadrature-degree must be positive")


def write_outputs(path: Path, results: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metric_definition": (
            "Primary amplitude=max(phi_T)-min(phi_T) on T=c1; gradients are exact "
            "cellwise derivatives of the degree-p FE fields evaluated after Newton projection."
        ),
        "outer_region_gradient_sampling_definition": {
            "region": "physical quadrature points satisfying T<c1 (between the domain boundary and T=c1)",
            "nonvanishing_test": "|grad(phi_T)| > 0",
            "inward_monotonicity_test": "grad(phi_T) dot grad(T) > 0",
            "sampling": (
                "Exact cellwise finite-element gradients sampled on a deterministic "
                "high-order triangular quadrature grid; reported fractions are area weighted."
            ),
            "qualification": (
                "Point sampling is numerical evidence and does not constitute a proof of "
                "strict positivity at every point or a uniform bound up to polygonal corners."
            ),
        },
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    flat_results = [
        {
            key: value
            for key, value in result.items()
            if key not in ("refinement_history", "outer_region_gradient_sampling")
        }
        for result in results
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_results[0]))
        writer.writeheader()
        writer.writerows(flat_results)
    print(f"json={path.resolve()}")
    print(f"csv={csv_path.resolve()}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    names = REQUESTED_GEOMETRIES if args.geometry == "all" else (args.geometry,)
    results: list[dict[str, object]] = []
    for name in names:
        result = solve_and_measure(name, args)
        results.append(result)
        print(
            f"RESULT geometry={name} phi_peak_to_peak={result['phi_peak_to_peak']:.12e} "
            f"relative={result['phi_peak_to_peak_relative_to_abs_mean']:.12e} "
            f"gradient_TV={result['phi_total_variation_from_gradient']:.12e} "
            f"sample_TV={result['phi_total_variation_from_samples']:.12e} "
            f"effective_full_oscillations="
            f"{result['effective_full_oscillations_from_samples']:.6f}",
            flush=True,
        )
        gc.collect()
    write_outputs(args.output, results)


if __name__ == "__main__":
    main()
