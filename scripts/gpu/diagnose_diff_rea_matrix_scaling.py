#!/usr/bin/env python3
"""Diagnose device scaling for GPU diffusion-reaction trace matrices.

The module intentionally stays outside the solver stack.  It reuses the GPU
runner assembly path, builds a device CSR matrix, applies optional device-side
scaling to a copy, and reports algebraic diagnostics for nodal/modal trace
comparisons.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from hdgfem.core.space import DGSpace
from hdgfem.io.output import pretty_print_sections
from scripts.gpu import run_diff_rea_gpu4_hdg as runner
from scripts.diffusion_reaction.diff_rea_cases import case_definition_by_key


TRACE_BASIS_CHOICES = ("legacy-lagrange", "legendre-modal", "bernstein")
SCALE_CHOICES = ("off", "none", "left", "on", "symmetric")


def _csv_values(value: str | None, *, default: tuple[str, ...], choices: tuple[str, ...], label: str) -> list[str]:
    if value is None or str(value).strip() == "":
        values = list(default)
    else:
        values = [item.strip() for item in str(value).split(",") if item.strip()]
    invalid = [item for item in values if item not in choices]
    if invalid:
        raise ValueError(f"invalid {label}: {', '.join(invalid)}; expected one of {', '.join(choices)}")
    return values


def _fmt_float(value: float | None, fmt: str = ".3e") -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return str(value)
    return format(value, fmt)


def _finite_stats(values: np.ndarray, percentiles: tuple[float, ...] = (0.0, 1.0, 5.0, 50.0, 95.0, 99.0, 100.0)) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"p{p:g}": float("nan") for p in percentiles}
    result: dict[str, float] = {}
    for p in percentiles:
        result[f"p{p:g}"] = float(np.percentile(values, p))
    return result


def _device_percentiles(cp, values, percentiles: tuple[float, ...] = (0.0, 1.0, 5.0, 50.0, 95.0, 99.0, 100.0)) -> dict[str, float]:
    return _finite_stats(cp.asnumpy(values), percentiles)


def _row_norms(cp, matrix):
    nrows = int(matrix.shape[0])
    row_sq = cp.zeros(nrows, dtype=cp.float64)
    counts = matrix.indptr[1:] - matrix.indptr[:-1]
    nonempty = counts > 0
    if int(cp.count_nonzero(nonempty).get()):
        starts = matrix.indptr[:-1][nonempty]
        row_sq[nonempty] = cp.add.reduceat(matrix.data * matrix.data, starts)
    return cp.sqrt(row_sq)


def _column_norms(cp, matrix):
    ncols = int(matrix.shape[1])
    col_sq = cp.bincount(
        matrix.indices.astype(cp.int64, copy=False),
        weights=matrix.data * matrix.data,
        minlength=ncols,
    )
    if int(col_sq.size) > ncols:
        col_sq = col_sq[:ncols]
    return cp.sqrt(col_sq)


def build_device_csr(rows, cols, data, rhs, raw_assembly):
    cp = runner.require_cupy()
    start = time.perf_counter()
    indptr = getattr(raw_assembly, "indptr", None)
    indices = getattr(raw_assembly, "indices", None)
    if indptr is not None and indices is not None:
        matrix = runner._cupyx_csr_from_arrays(data, indptr, indices, shape=(int(rhs.size), int(rhs.size)))
        matrix_input = "direct csr"
    else:
        matrix = runner._cupyx_csr_from_coo(rows, cols, data, rhs)
        matrix_input = "coo->csr"
    cp.cuda.get_current_stream().synchronize()
    return matrix, matrix_input, time.perf_counter() - start


def matrix_diagnostics(matrix, rhs) -> dict[str, Any]:
    cp = runner.require_cupy()
    start = time.perf_counter()
    diag = matrix.diagonal()
    abs_diag = cp.abs(diag)
    row_norm = _row_norms(cp, matrix)
    col_norm = _column_norms(cp, matrix)
    data_norm = cp.linalg.norm(matrix.data)
    rhs_norm = cp.linalg.norm(rhs)
    cp.cuda.get_current_stream().synchronize()
    basic_elapsed = time.perf_counter() - start

    symmetry_start = time.perf_counter()
    symmetry_error = None
    try:
        diff = matrix - matrix.T
        diff.sum_duplicates()
        cp.cuda.get_current_stream().synchronize()
        symmetry_norm = float(cp.linalg.norm(diff.data).get()) if int(diff.nnz) else 0.0
        matrix_norm = float(data_norm.get())
        symmetry_defect = symmetry_norm / matrix_norm if matrix_norm else float("nan")
        symmetry_nnz = int(diff.nnz)
    except Exception as exc:  # pragma: no cover - depends on cupyx sparse backend behavior
        cp.cuda.get_current_stream().synchronize()
        symmetry_norm = float("nan")
        symmetry_defect = float("nan")
        symmetry_nnz = -1
        symmetry_error = str(exc)
    symmetry_elapsed = time.perf_counter() - symmetry_start

    diag_host = cp.asnumpy(diag)
    abs_diag_host = np.abs(diag_host)
    row_stats = _device_percentiles(cp, row_norm)
    col_stats = _device_percentiles(cp, col_norm)
    return {
        "n": int(matrix.shape[0]),
        "nnz": int(matrix.nnz),
        "diag_min": float(np.nanmin(diag_host)) if diag_host.size else float("nan"),
        "diag_max": float(np.nanmax(diag_host)) if diag_host.size else float("nan"),
        "abs_diag_min": float(np.nanmin(abs_diag_host)) if abs_diag_host.size else float("nan"),
        "abs_diag_max": float(np.nanmax(abs_diag_host)) if abs_diag_host.size else float("nan"),
        "diag_zero_count": int(cp.count_nonzero(diag == 0.0).get()),
        "diag_negative_count": int(cp.count_nonzero(diag < 0.0).get()),
        "row_norm": row_stats,
        "col_norm": col_stats,
        "matrix_frobenius": float(data_norm.get()),
        "rhs_norm": float(rhs_norm.get()),
        "symmetry_norm": symmetry_norm,
        "symmetry_defect": symmetry_defect,
        "symmetry_nnz": symmetry_nnz,
        "symmetry_error": symmetry_error,
        "stats_elapsed_seconds": basic_elapsed,
        "symmetry_elapsed_seconds": symmetry_elapsed,
    }


def condition_estimate(matrix, stats: dict[str, Any], args) -> dict[str, Any]:
    n = int(matrix.shape[0])
    if int(args.condition_max_dof) <= 0:
        return {"condition_status": "disabled"}
    if n > int(args.condition_max_dof):
        return {"condition_status": f"skipped: n={n:,d} > {int(args.condition_max_dof):,d}"}
    symmetry_defect = float(stats.get("symmetry_defect", float("nan")))
    if not math.isfinite(symmetry_defect) or symmetry_defect > float(args.symmetry_tolerance):
        return {"condition_status": f"skipped: symmetry_defect={_fmt_float(symmetry_defect)}"}
    try:
        from scipy.sparse.linalg import eigsh
    except Exception as exc:  # pragma: no cover - optional dependency
        return {"condition_status": f"skipped: scipy unavailable ({exc})"}

    start = time.perf_counter()
    try:
        host = matrix.get()
        host = 0.5 * (host + host.T)
        if n == 1:
            lam_min = lam_max = float(host[0, 0])
        else:
            lam_max = float(eigsh(host, k=1, which="LA", return_eigenvectors=False, tol=args.eig_tol, maxiter=args.eig_maxiter)[0])
            lam_min = float(eigsh(host, k=1, which="SA", return_eigenvectors=False, tol=args.eig_tol, maxiter=args.eig_maxiter)[0])
        cond = abs(lam_max / lam_min) if lam_min != 0.0 else float("inf")
        status = "ok" if lam_min > 0.0 else "nonpositive-min-eigenvalue"
        return {
            "condition_status": status,
            "lambda_min": lam_min,
            "lambda_max": lam_max,
            "condition_estimate": cond,
            "condition_elapsed_seconds": time.perf_counter() - start,
        }
    except Exception as exc:
        return {
            "condition_status": f"failed: {exc}",
            "condition_elapsed_seconds": time.perf_counter() - start,
        }


def solve_for_scale(rows, cols, data, rhs, raw_assembly, args, scale: str) -> dict[str, Any]:
    solve_args = argparse.Namespace(**vars(args))
    solve_args.scale_system = scale
    before = dict(runner.TIMINGS)
    runner.solve_amgx(
        rows,
        cols,
        data,
        rhs,
        solve_args,
        indptr=getattr(raw_assembly, "indptr", None),
        indices=getattr(raw_assembly, "indices", None),
    )
    after = dict(runner.TIMINGS)
    return {
        "amgx_iterations": runner.RUN_METADATA.get("amgx_iterations"),
        "amgx_solver_rel_residual": runner.RUN_METADATA.get("amgx_solver_rel_residual"),
        "amgx_physical_rel_residual": runner.RUN_METADATA.get("amgx_physical_rel_residual"),
        "solve_scale_seconds": runner.RUN_METADATA.get("solve_scale_seconds"),
        "solve_csr_seconds": after.get("solve.csr_scale", 0.0) - before.get("solve.csr_scale", 0.0),
        "amgx_setup_seconds": after.get("solve.amgx_setup", 0.0) - before.get("solve.amgx_setup", 0.0),
        "amgx_solve_seconds": after.get("solve.amgx_solve", 0.0) - before.get("solve.amgx_solve", 0.0),
        "solve_total_seconds": after.get("solve.total", 0.0) - before.get("solve.total", 0.0),
    }


def diagnostic_rows(result: dict[str, Any]) -> list[tuple[str, Any, str]]:
    row = result["row_norm"]
    col = result["col_norm"]
    rows: list[tuple[str, Any, str]] = [
        ("trace", result["trace_basis"], "s"),
        ("scale", result["scale"], "s"),
        ("matrix input", result["matrix_input"], "s"),
        ("n", result["n"], ",d"),
        ("nnz", result["nnz"], ",d"),
        ("diag min", result["diag_min"], ".3e"),
        ("diag max", result["diag_max"], ".3e"),
        ("abs diag min", result["abs_diag_min"], ".3e"),
        ("abs diag max", result["abs_diag_max"], ".3e"),
        ("zero diag", result["diag_zero_count"], ",d"),
        ("negative diag", result["diag_negative_count"], ",d"),
        ("row p05", row["p5"], ".3e"),
        ("row p50", row["p50"], ".3e"),
        ("row p95", row["p95"], ".3e"),
        ("col p05", col["p5"], ".3e"),
        ("col p50", col["p50"], ".3e"),
        ("col p95", col["p95"], ".3e"),
        ("sym defect", result["symmetry_defect"], ".3e"),
        ("rhs norm", result["rhs_norm"], ".3e"),
        ("scale time", result["scale_elapsed_seconds"], ".3f"),
        ("stats time", result["stats_elapsed_seconds"], ".3f"),
        ("sym time", result["symmetry_elapsed_seconds"], ".3f"),
        ("cond status", result.get("condition_status", "n/a"), "s"),
    ]
    if "lambda_min" in result:
        rows.extend(
            [
                ("lambda min", result["lambda_min"], ".3e"),
                ("lambda max", result["lambda_max"], ".3e"),
                ("cond est", result["condition_estimate"], ".3e"),
            ]
        )
    if result.get("solve_enabled"):
        rows.extend(
            [
                ("AMGX iter", result.get("amgx_iterations"), "s"),
                ("solver rel", result.get("amgx_solver_rel_residual"), ".3e"),
                ("physical rel", result.get("amgx_physical_rel_residual"), ".3e"),
                ("AMGX setup", result.get("amgx_setup_seconds"), ".3f"),
                ("AMGX solve", result.get("amgx_solve_seconds"), ".3f"),
            ]
        )
    if result.get("symmetry_error"):
        rows.append(("symmetry error", result["symmetry_error"], "s"))
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--trace-basis", choices=TRACE_BASIS_CHOICES, default=None, help="single trace basis; overrides --trace-bases")
    parser.add_argument("--trace-bases", default="legacy-lagrange,legendre-modal", help="comma-separated trace basis list")
    parser.add_argument("--assembly-backend", choices=("cupy", "raw-cuda"), default="raw-cuda")
    parser.add_argument("--raw-matrix-format", choices=("coo", "csr"), default="csr")
    parser.add_argument("--raw-block-size", type=int, choices=(1, 32, 64, 128), default=128)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--scales", default="off,symmetric", help="comma-separated scale modes: off/none,left/on,symmetric")
    parser.add_argument("--scale-system", choices=SCALE_CHOICES, default=None, help="single scale mode; overrides --scales")
    parser.add_argument("--solve", action=argparse.BooleanOptionalAction, default=True, help="run AMGX for every diagnostic scale")
    parser.add_argument("--amgx-config", default=str(runner.DEFAULT_AMGX_CONFIG_PATH))
    parser.add_argument("--amgx-solver", default="PCGF")
    parser.add_argument("--amgx-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--amgx-maxiter", type=int, default=2000)
    parser.add_argument("--condition-max-dof", type=int, default=5000, help="host eigsh threshold; 0 disables condition estimates")
    parser.add_argument("--symmetry-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--eig-tol", type=float, default=1.0e-8)
    parser.add_argument("--eig-maxiter", type=int, default=20000)
    parser.add_argument("--gmsh-verbosity", type=int, default=0)
    parser.add_argument("--gmsh-algorithm", type=int, default=None)
    parser.add_argument("--show-cupy-config", action="store_true")
    parser.add_argument("--verbosity", "-v", type=int, choices=(0, 1, 2), default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    trace_bases = [args.trace_basis] if args.trace_basis else _csv_values(
        args.trace_bases,
        default=("legacy-lagrange", "legendre-modal"),
        choices=TRACE_BASIS_CHOICES,
        label="trace basis",
    )
    scales = [args.scale_system] if args.scale_system else _csv_values(
        args.scales,
        default=("off", "symmetric"),
        choices=SCALE_CHOICES,
        label="scale mode",
    )

    runner.set_verbosity(args.verbosity)
    cp = runner.require_cupy()
    runner.require_cupyx_sparse()
    if args.solve:
        runner.require_pyamgx()
    cp.cuda.set_allocator(None)
    cp.cuda.set_pinned_memory_allocator(None)
    if args.show_cupy_config:
        cp.show_config()

    case = case_definition_by_key(args.case)
    diffusion, reaction, source, exact = case.build()
    if diffusion != (1.0, 0.0, 1.0):
        raise NotImplementedError("diagnostics currently support identity diffusion only")

    runner.log("\n----- HDGFEM GPU Diffusion Matrix Scaling Diagnostics -----")
    runner.log(
        f"case={args.case}, mesh={args.mesh_type}, mesh_size={args.mesh_size:g}, order={args.order}, "
        f"basis={args.basis}, trace_bases={','.join(trace_bases)}, scales={','.join(scales)}, "
        f"backend={args.assembly_backend}, raw_matrix={args.raw_matrix_format}"
    )

    start = time.perf_counter()
    runner.log("generating mesh ... ", end="")
    mesh, domain = runner.build_mesh(args, case)
    runner.print_done(time.perf_counter() - start)

    start = time.perf_counter()
    runner.log("building DG space/reference data ... ", end="")
    space = DGSpace(
        mesh,
        args.order,
        basis_type=args.basis,
        volume_quadrature=args.volume_quadrature,
        volume_quad_1d=args.volume_quad_1d,
        edge_quad_1d=args.edge_quad_1d,
    )
    runner.print_done(time.perf_counter() - start)
    runner.log(f"domain={domain}, h={mesh.h:.3e}, triangles={mesh.num_tri:,}")

    start = time.perf_counter()
    runner.log("copying static data to GPU ... ", end="")
    cspace = runner.as_cupy_space(space)
    maps = runner.build_dof_maps(cspace)
    cp.cuda.get_current_stream().synchronize()
    runner.print_done(time.perf_counter() - start)

    all_results: list[dict[str, Any]] = []
    for trace_basis in trace_bases:
        runner.TIMINGS.clear()
        runner.RUN_METADATA.clear()
        args_for_trace = copy.copy(args)
        args_for_trace.trace_basis = trace_basis
        trace_ref = runner.build_trace_reference(cspace, trace_basis)
        rows, cols, data, rhs, _local_lhs, _element_boundary, _source_rhs, _boundary_trace, raw_assembly = runner.assemble_reduced_system(
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
        matrix, matrix_input, csr_elapsed = build_device_csr(rows, cols, data, rhs, raw_assembly)
        assembly_backend = str(runner.RUN_METADATA.get("assembly_backend", args.assembly_backend))
        for scale in scales:
            normalized_scale = runner.normalize_scale_system(scale)
            scaled = runner.prepare_scaled_device_csr_for_solve(matrix, rhs, normalized_scale, timing_key=None)
            stats = matrix_diagnostics(scaled.matrix, scaled.rhs)
            stats.update(condition_estimate(scaled.matrix, stats, args))
            stats.update(
                {
                    "trace_basis": trace_basis,
                    "scale": normalized_scale,
                    "requested_scale": scale,
                    "basis": args.basis,
                    "assembly_backend": assembly_backend,
                    "matrix_input": matrix_input,
                    "csr_build_seconds": csr_elapsed,
                    "scale_elapsed_seconds": scaled.elapsed_seconds,
                    "solve_enabled": bool(args.solve),
                }
            )
            if args.solve:
                stats.update(solve_for_scale(rows, cols, data, rhs, raw_assembly, args_for_trace, scale))
            all_results.append(stats)
            title = f"Diagnostics: trace={trace_basis}, scale={normalized_scale}"
            pretty_print_sections([(title, diagnostic_rows(stats))], title="HDGFEM Diffusion Matrix Diagnostics")

    comparison_rows: list[tuple[str, Any, str]] = []
    for result in all_results:
        label = f"{result['trace_basis']} / {result['scale']}"
        comparison_rows.extend(
            [
                (f"{label} sym", result["symmetry_defect"], ".3e"),
                (f"{label} row95", result["row_norm"]["p95"], ".3e"),
                (f"{label} col95", result["col_norm"]["p95"], ".3e"),
            ]
        )
        if result.get("solve_enabled"):
            comparison_rows.append((f"{label} iter", result.get("amgx_iterations"), "s"))
        if "condition_estimate" in result:
            comparison_rows.append((f"{label} cond", result["condition_estimate"], ".3e"))
    if comparison_rows:
        pretty_print_sections([("Comparison", comparison_rows)], title="Scaling Comparison Summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
