#!/usr/bin/env python3
"""Compare trace-basis and scaling effects for diffusion HDG matrices.

The script is intentionally orchestration-only: assembly and solves go through
:class:`hdgfem.DiffusionReactionHDGSolver`, while sparse scaling uses the
public :mod:`hdgfem.linalg` API.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import eigsh

from hdgfem import DGSpace, DiffusionReactionHDGSolver
from hdgfem.core.mesh import (
    gmsh_disc_mesh,
    gmsh_lshape_mesh,
    gmsh_rectangle_mesh,
    gmsh_triangle_mesh,
    rectangle_mesh,
)
from hdgfem.linalg.amgx.config import load_amgx_config
from hdgfem.io.output import pretty_print_sections
from hdgfem.linalg import assemble_global_matrix, scale_sparse_system
from scripts.diffusion_reaction.cases import case_definition_by_key


TRACE_BASIS_CHOICES = ("legacy-lagrange", "legendre-modal", "bernstein")
SCALE_CHOICES = ("off", "none", "left", "on", "symmetric")
DEFAULT_AMGX_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json"
)


def _csv_values(value: str, *, choices: tuple[str, ...], label: str) -> list[str]:
    """Parse a comma-separated option and reject unsupported values."""
    values = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in values if item not in choices]
    if invalid:
        raise ValueError(f"invalid {label}: {', '.join(invalid)}; expected one of {', '.join(choices)}")
    return values


def _scale_mode(value: str) -> str:
    """Normalize compatibility aliases for sparse-system scaling."""
    return {"off": "none", "on": "left"}.get(value, value)


def _build_mesh(args, case):
    """Build the domain selected by case metadata and CLI options."""
    domain = case.default_domain if args.mesh_type == "auto" else args.mesh_type
    common = {
        "verbosity": args.gmsh_verbosity,
        "algorithm": args.gmsh_algorithm,
        "log_cache": args.verbosity >= 1,
    }
    if domain == "structured-rectangle":
        return rectangle_mesh(args.nx, args.ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0)), domain
    if domain == "unit-rectangle":
        return gmsh_rectangle_mesh(args.mesh_size, xlim=(0.0, 1.0), ylim=(0.0, 1.0), **common), domain
    if domain == "rectangle":
        return gmsh_rectangle_mesh(args.mesh_size, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), **common), domain
    if domain == "disc":
        radius = args.disc_radius if args.disc_radius is not None else (5.0 if case.key == "trigonometric-poisson" else 1.0)
        return gmsh_disc_mesh(args.mesh_size, args.mesh_size, center=(0.0, 0.0), radius=radius, **common), domain
    if domain == "lshape":
        return gmsh_lshape_mesh(args.mesh_size, **common), domain
    return gmsh_triangle_mesh(
        args.mesh_size,
        vertices=((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)),
        **common,
    ), "triangle"


def _assembly_matrix(assembly) -> scipy.sparse.csr_array:
    """Convert a public diffusion assembly result to host CSR."""
    size = int(assembly.rhs.size)
    if assembly.indptr is not None and assembly.indices is not None:
        return scipy.sparse.csr_array(
            (assembly.data, assembly.indices, assembly.indptr),
            shape=(size, size),
        )
    if assembly.rows is None or assembly.cols is None:
        raise ValueError("assembly result contains neither CSR indices nor COO indices")
    return assemble_global_matrix(assembly.rows, assembly.cols, assembly.data, size)


def _finite_percentiles(values: np.ndarray) -> dict[str, float]:
    """Return robust percentiles for finite values."""
    finite = np.asarray(values, dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"p5": float("nan"), "p50": float("nan"), "p95": float("nan")}
    p5, p50, p95 = np.percentile(finite, (5.0, 50.0, 95.0))
    return {"p5": float(p5), "p50": float(p50), "p95": float(p95)}


def _matrix_diagnostics(matrix, rhs) -> dict[str, Any]:
    """Compute backend-independent algebraic diagnostics for a CSR system."""
    started = time.perf_counter()
    diagonal = np.asarray(matrix.diagonal(), dtype=np.float64)
    row_norm = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    col_norm = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel())
    difference = matrix - matrix.T
    matrix_norm = float(np.linalg.norm(matrix.data))
    symmetry_norm = float(np.linalg.norm(difference.data)) if difference.nnz else 0.0
    return {
        "n": int(matrix.shape[0]),
        "nnz": int(matrix.nnz),
        "diag_min": float(np.min(diagonal)) if diagonal.size else float("nan"),
        "diag_max": float(np.max(diagonal)) if diagonal.size else float("nan"),
        "abs_diag_min": float(np.min(np.abs(diagonal))) if diagonal.size else float("nan"),
        "abs_diag_max": float(np.max(np.abs(diagonal))) if diagonal.size else float("nan"),
        "diag_zero_count": int(np.count_nonzero(diagonal == 0.0)),
        "diag_negative_count": int(np.count_nonzero(diagonal < 0.0)),
        "row_norm": _finite_percentiles(row_norm),
        "col_norm": _finite_percentiles(col_norm),
        "rhs_norm": float(np.linalg.norm(rhs)),
        "symmetry_defect": symmetry_norm / matrix_norm if matrix_norm else float("nan"),
        "stats_elapsed_seconds": time.perf_counter() - started,
    }


def _condition_estimate(matrix, stats: dict[str, Any], args) -> dict[str, Any]:
    """Estimate the spectral condition number for small symmetric systems."""
    size = int(matrix.shape[0])
    if args.condition_max_dof <= 0:
        return {"condition_status": "disabled"}
    if size > args.condition_max_dof:
        return {"condition_status": f"skipped: n={size:,d} > {args.condition_max_dof:,d}"}
    if not math.isfinite(stats["symmetry_defect"]) or stats["symmetry_defect"] > args.symmetry_tolerance:
        return {"condition_status": "skipped: matrix is not sufficiently symmetric"}
    started = time.perf_counter()
    try:
        symmetric = 0.5 * (matrix + matrix.T)
        if size == 1:
            lambda_min = lambda_max = float(symmetric[0, 0])
        else:
            lambda_max = float(eigsh(symmetric, k=1, which="LA", return_eigenvectors=False, tol=args.eig_tol, maxiter=args.eig_maxiter)[0])
            lambda_min = float(eigsh(symmetric, k=1, which="SA", return_eigenvectors=False, tol=args.eig_tol, maxiter=args.eig_maxiter)[0])
        return {
            "condition_status": "ok" if lambda_min > 0.0 else "nonpositive-min-eigenvalue",
            "lambda_min": lambda_min,
            "lambda_max": lambda_max,
            "condition_estimate": abs(lambda_max / lambda_min) if lambda_min else float("inf"),
            "condition_elapsed_seconds": time.perf_counter() - started,
        }
    except Exception as exc:
        return {"condition_status": f"failed: {exc}", "condition_elapsed_seconds": time.perf_counter() - started}


def _diagnostic_rows(result: dict[str, Any]) -> list[tuple[str, Any, str]]:
    """Build pretty-printer rows for one trace/scaling combination."""
    row = result["row_norm"]
    col = result["col_norm"]
    rows = [
        ("trace", result["trace_basis"], "s"),
        ("scale", result["scale"], "s"),
        ("n", result["n"], ",d"),
        ("nnz", result["nnz"], ",d"),
        ("diag min", result["diag_min"], ".3e"),
        ("diag max", result["diag_max"], ".3e"),
        ("abs diag min", result["abs_diag_min"], ".3e"),
        ("abs diag max", result["abs_diag_max"], ".3e"),
        ("zero diag", result["diag_zero_count"], ",d"),
        ("negative diag", result["diag_negative_count"], ",d"),
        ("row p05/p50/p95", f"{row['p5']:.3e} / {row['p50']:.3e} / {row['p95']:.3e}", "s"),
        ("col p05/p50/p95", f"{col['p5']:.3e} / {col['p50']:.3e} / {col['p95']:.3e}", "s"),
        ("sym defect", result["symmetry_defect"], ".3e"),
        ("rhs norm", result["rhs_norm"], ".3e"),
        ("scale time", result["scale_elapsed_seconds"], ".3f"),
        ("stats time", result["stats_elapsed_seconds"], ".3f"),
        ("cond status", result["condition_status"], "s"),
    ]
    if "condition_estimate" in result:
        rows.extend([
            ("lambda min", result["lambda_min"], ".3e"),
            ("lambda max", result["lambda_max"], ".3e"),
            ("cond est", result["condition_estimate"], ".3e"),
        ])
    if result.get("solve_enabled"):
        rows.extend([
            ("iterations", result.get("iterations"), "s"),
            ("solver rel", result.get("solver_relative_residual"), ".3e"),
            ("physical rel", result.get("physical_relative_residual"), ".3e"),
            ("solve time", result.get("solve_elapsed_seconds"), ".3f"),
        ])
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="trigonometric-poisson")
    parser.add_argument("--order", "-o", type=int, default=6)
    parser.add_argument("--mesh-size", "-ms", type=float, default=0.05)
    parser.add_argument("--mesh-type", "-mt", choices=("auto", "disc", "rectangle", "unit-rectangle", "triangle", "lshape", "structured-rectangle"), default="auto")
    parser.add_argument("--disc-radius", type=float, default=None)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--basis", default="dub_orth", choices=("hier_C0", "hierarchical_c0", "bernstein", "dub_orth"))
    parser.add_argument("--volume-quadrature", choices=("auto", "symmetric", "duffy"), default="auto")
    parser.add_argument("--volume-quad-1d", type=int, default=None)
    parser.add_argument("--edge-quad-1d", type=int, default=None)
    parser.add_argument("--trace-basis", choices=TRACE_BASIS_CHOICES, default=None)
    parser.add_argument("--trace-bases", default="legacy-lagrange,legendre-modal")
    parser.add_argument("--assembly-backend", choices=("cupy", "raw-cuda"), default="raw-cuda")
    parser.add_argument("--raw-matrix-format", choices=("coo", "csr"), default="csr")
    parser.add_argument("--raw-block-size", choices=("auto", "1", "32", "64", "128"), default="128")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--scales", default="off,symmetric")
    parser.add_argument("--scale-system", choices=SCALE_CHOICES, default=None)
    parser.add_argument("--solve", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amgx-config", default=None)
    parser.add_argument("--amgx-solver", default="PCGF")
    parser.add_argument("--amgx-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--amgx-maxiter", type=int, default=2000)
    parser.add_argument("--condition-max-dof", type=int, default=5000)
    parser.add_argument("--symmetry-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--eig-tol", type=float, default=1.0e-8)
    parser.add_argument("--eig-maxiter", type=int, default=20000)
    parser.add_argument("--gmsh-verbosity", type=int, default=0)
    parser.add_argument("--gmsh-algorithm", type=int, default=None)
    parser.add_argument("--show-cupy-config", action="store_true")
    parser.add_argument("--verbosity", "-v", type=int, choices=(0, 1, 2), default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run matrix diagnostics and optional package-owned solves."""
    args = build_arg_parser().parse_args(argv)
    trace_bases = [args.trace_basis] if args.trace_basis else _csv_values(
        args.trace_bases, choices=TRACE_BASIS_CHOICES, label="trace basis"
    )
    requested_scales = [args.scale_system] if args.scale_system else _csv_values(
        args.scales, choices=SCALE_CHOICES, label="scale mode"
    )
    scales = [_scale_mode(value) for value in requested_scales]
    if args.show_cupy_config:
        from hdgfem.runtime.optional import require_cupy

        require_cupy().show_config()

    case = case_definition_by_key(args.case)
    problem = case.build()
    mesh, domain = _build_mesh(args, case)
    space = DGSpace(
        mesh,
        args.order,
        basis_type=args.basis,
        volume_quadrature=args.volume_quadrature,
        volume_quad_1d=args.volume_quad_1d,
        edge_quad_1d=args.edge_quad_1d,
    )
    config, config_path = load_amgx_config(
        args.amgx_config,
        default_path=DEFAULT_AMGX_CONFIG_PATH,
        solver=args.amgx_solver,
        tolerance=args.amgx_tolerance,
        maxiter=args.amgx_maxiter,
        verbose=args.verbosity,
    )
    if args.verbosity:
        print(
            f"domain={domain}, h={mesh.h:.3e}, triangles={mesh.num_tri:,}, "
            f"config={config_path}, backend={args.assembly_backend}"
        )

    all_results: list[dict[str, Any]] = []
    for trace_basis in trace_bases:
        solver = DiffusionReactionHDGSolver(
            space,
            source=problem.source,
            reaction=problem.reaction,
            boundary_condition=problem.exact,
            diffusion=problem.diffusion,
            stabilization=args.tau,
            solver="amgx",
            solver_rtol=args.amgx_tolerance,
            maxiter=args.amgx_maxiter,
            amgx_config=config,
            assembly_backend=args.assembly_backend,
            trace_basis=trace_basis,
            raw_matrix_format=args.raw_matrix_format,
            raw_block_size=args.raw_block_size,
            boundary_mode="eliminate",
            verbose=args.verbosity,
        )
        assembly = solver.assemble_global_matrix()
        matrix = _assembly_matrix(assembly)
        for scale in scales:
            scale_started = time.perf_counter()
            scaled_matrix, scaled_rhs, _ = scale_sparse_system(matrix, assembly.rhs, scale)
            scale_elapsed = time.perf_counter() - scale_started
            stats = _matrix_diagnostics(scaled_matrix, scaled_rhs)
            stats.update(_condition_estimate(scaled_matrix, stats, args))
            stats.update({
                "trace_basis": trace_basis,
                "scale": scale,
                "scale_elapsed_seconds": scale_elapsed,
                "solve_enabled": args.solve,
            })
            if args.solve:
                result = solver.solve(scale_system=scale)
                solve = result.global_solve_result
                stats.update({
                    "iterations": None if solve is None else solve.iteration_count,
                    "solver_relative_residual": None if solve is None else solve.solver_relative_residual_norm,
                    "physical_relative_residual": None if solve is None else solve.physical_relative_residual_norm,
                    "solve_elapsed_seconds": None if solve is None else solve.total_elapsed_seconds,
                })
            all_results.append(stats)
            pretty_print_sections(
                [(f"trace={trace_basis}, scale={scale}", _diagnostic_rows(stats))],
                title="HDGFEM Diffusion Matrix Diagnostics",
            )

    comparison = []
    for result in all_results:
        label = f"{result['trace_basis']} / {result['scale']}"
        comparison.extend([
            (f"{label} symmetry", result["symmetry_defect"], ".3e"),
            (f"{label} row p95", result["row_norm"]["p95"], ".3e"),
            (f"{label} col p95", result["col_norm"]["p95"], ".3e"),
        ])
        if result.get("solve_enabled"):
            comparison.append((f"{label} iterations", result.get("iterations"), "s"))
        if "condition_estimate" in result:
            comparison.append((f"{label} condition", result["condition_estimate"], ".3e"))
    pretty_print_sections([("Comparison", comparison)], title="Scaling Comparison Summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
