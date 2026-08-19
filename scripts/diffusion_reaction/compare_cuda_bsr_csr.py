#!/usr/bin/env python3
"""Compare direct raw-CUDA face-BSR and scalar-CSR HDG Poisson paths.

The default is a radius-5 unstructured Gmsh disk with approximately 150,000
triangles at p=6. The CSR and BSR configurations are independently selectable;
repeated runs can alternate their execution order and emit machine-readable
CSV timing and parity records.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hdgfem import (
    DGSpace,
    DiffusionReactionHDGSolver,
    evaluate_scalar_error,
    gmsh_disc_mesh,
    rectangle_mesh,
)
from hdgfem.backends.cupy import require_cupy
from hdgfem.io.config import describe_amgx_preconditioner, describe_amgx_solver, load_amgx_config
from hdgfem.solvers.stabilization import GlobalLengthDiffusion
from scripts.diffusion_reaction.cases import trigonometric_poisson_case


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "amgx"
DEFAULT_CSR_CONFIG = CONFIG_DIR / "diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json"
DEFAULT_BSR_AGGREGATION_CONFIG = (
    CONFIG_DIR / "diff_rea_gpu4_hdg_pcgf_aggregation_block_jacobi_bsr.json"
)
DEFAULT_BSR_BLOCK_JACOBI_CONFIG = (
    CONFIG_DIR / "diff_rea_gpu4_hdg_pcgf_block_jacobi_bsr.json"
)


def _default_bsr_config(order: int) -> Path:
    """Choose an AMGX config compatible with the face block size p+1."""
    if 1 <= order <= 4:
        return DEFAULT_BSR_AGGREGATION_CONFIG
    return DEFAULT_BSR_BLOCK_JACOBI_CONFIG


@dataclass(frozen=True)
class BenchmarkRow:
    repeat: int
    matrix_format: str
    config: str
    solver: str
    preconditioner: str
    matrix_bytes: int
    pattern_bytes: int
    assembly_seconds: float
    kernel_seconds: float
    amgx_setup_seconds: float
    amgx_solve_seconds: float
    reconstruction_seconds: float
    total_seconds: float
    iterations: int
    relative_residual: float
    l2_error: float | None
    coefficients: np.ndarray


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        choices=("disk", "structured-rectangle"),
        default="disk",
    )
    parser.add_argument("--mesh-size", type=float, default=0.0345)
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--nx", type=int, default=275)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--order", "-p", type=int, default=6)
    parser.add_argument("--only", choices=("both", "csr", "bsr"), default="both")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--format-order",
        choices=("csr-first", "bsr-first", "alternate"),
        default="csr-first",
        help="execution order within each repeat when --only=both",
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--allow-small", action="store_true")
    parser.add_argument("--minimum-triangles", type=int, default=150_000)
    parser.add_argument("--basis", default="dub_orth")
    parser.add_argument(
        "--trace-basis",
        choices=("legacy-lagrange", "legendre-modal"),
        default="legacy-lagrange",
    )
    parser.add_argument(
        "--raw-block-size", choices=("auto", "32", "64", "128"), default="128"
    )
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--csr-amgx-config", type=Path, default=DEFAULT_CSR_CONFIG)
    parser.add_argument(
        "--bsr-amgx-config",
        type=Path,
        default=None,
        help="override the degree-dependent BSR AMGX config",
    )
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--maxiter", type=int, default=10_000)
    parser.add_argument("--evaluate-error", action="store_true")
    parser.add_argument("--error-volume-quad-1d", type=int, default=None)
    parser.add_argument("--verbosity", "-v", type=int, choices=(0, 1, 2), default=1)
    return parser


def _format_bytes(value: int) -> str:
    return f"{value / (1024.0 ** 2):.1f} MiB"


def _matrix_storage(solver: DiffusionReactionHDGSolver) -> tuple[int, int]:
    cached = solver._raw_cuda_assembly_cache
    if cached is None or cached.raw_assembly is None:
        raise RuntimeError("raw-CUDA matrix cache is unavailable")
    raw = cached.raw_assembly
    if raw.indptr is None or raw.indices is None or raw.data is None:
        raise RuntimeError("compressed raw-CUDA assembly is incomplete")
    return int(raw.data.nbytes), int(raw.indptr.nbytes + raw.indices.nbytes)


def _run_one(
    matrix_format: str,
    *,
    repeat: int,
    space,
    problem,
    tau,
    config,
    config_path,
    args,
):
    solver = DiffusionReactionHDGSolver(
        space,
        diffusion=problem.diffusion,
        stabilization=tau,
        solver="amgx",
        solver_rtol=args.rtol,
        maxiter=args.maxiter,
        scale_system=False,
        amgx_config=config,
        cache_device_matrix=True,
        assembly_backend="raw-cuda",
        trace_basis=args.trace_basis,
        raw_matrix_format=matrix_format,
        raw_block_size=args.raw_block_size,
        boundary_mode="eliminate",
        hdg_postprocess="none",
        verbose=args.verbosity,
    )
    solver.set_problem(problem.source, problem.reaction, problem.exact)
    started = time.perf_counter()
    try:
        result = solver.solve()
        require_cupy().cuda.get_current_stream().synchronize()
        wall_seconds = time.perf_counter() - started
        info = result.global_solve_result
        if info is None:
            raise RuntimeError("AMGX diagnostics are missing")
        details = result.timings.details or {}
        matrix_bytes, pattern_bytes = _matrix_storage(solver)
        coefficients = np.array(result.field.coeffs, copy=True)
        l2_error = None
        if args.evaluate_error:
            l2_error = evaluate_scalar_error(
                result.field,
                problem.exact,
                volume_quad_1d=args.error_volume_quad_1d,
                include_samples=False,
            ).metrics.l2
        return BenchmarkRow(
            repeat=repeat,
            matrix_format=matrix_format,
            config="embedded" if config_path is None else config_path.name,
            solver=describe_amgx_solver(config),
            preconditioner=describe_amgx_preconditioner(config),
            matrix_bytes=matrix_bytes,
            pattern_bytes=pattern_bytes,
            assembly_seconds=float(result.timings.trace_assembly),
            kernel_seconds=float(
                details.get(f"raw.assembly.raw.{matrix_format}_kernel", 0.0)
            ),
            amgx_setup_seconds=float(details.get("solve.amgx.setup", 0.0)),
            amgx_solve_seconds=float(details.get("solve.amgx.solve", 0.0)),
            reconstruction_seconds=float(result.timings.reconstruction),
            total_seconds=float(max(result.timings.total, wall_seconds)),
            iterations=int(info.iteration_count),
            relative_residual=float(info.physical_relative_residual_norm),
            l2_error=l2_error,
            coefficients=coefficients,
        )
    finally:
        solver.clear_cache()


def _coefficient_difference(csr: BenchmarkRow, bsr: BenchmarkRow) -> float:
    scale = max(np.linalg.norm(csr.coefficients.ravel()), np.finfo(float).tiny)
    return float(np.linalg.norm((bsr.coefficients - csr.coefficients).ravel()) / scale)


def _print_results(rows, *, domain, triangles, order, trace_dofs, tau):
    print("\nRaw-CUDA diffusion trace format comparison")
    print(f"  domain         : {domain}")
    print(f"  triangles      : {triangles:,}")
    print(f"  polynomial p   : {order}")
    print(f"  face block size: {order + 1}")
    print(f"  trace DOFs     : {trace_dofs:,}")
    print(f"  tau_d          : {tau:.8g}\n")
    print(
        f"{'rep':>3} {'fmt':<4} {'AMGX solver / preconditioner':<31} {'iter':>7} "
        f"{'residual':>10} {'assembly':>9} {'kernel':>9} {'setup':>9} "
        f"{'solve':>9} {'reconstruct':>11} {'total':>9} {'matrix':>11} {'pattern':>10}"
    )
    print("-" * 155)
    for row in rows:
        label = f"{row.solver}/{row.preconditioner}"
        print(
            f"{row.repeat:3d} {row.matrix_format:<4} {label:<31.31} {row.iterations:7d} "
            f"{row.relative_residual:10.3e} {row.assembly_seconds:9.3f} "
            f"{row.kernel_seconds:9.3f} {row.amgx_setup_seconds:9.3f} "
            f"{row.amgx_solve_seconds:9.3f} {row.reconstruction_seconds:11.3f} "
            f"{row.total_seconds:9.3f} {_format_bytes(row.matrix_bytes):>11} "
            f"{_format_bytes(row.pattern_bytes):>10}"
        )
        print(f"     config: {row.config}")
        if row.l2_error is not None:
            print(f"     manufactured L2 error: {row.l2_error:.6e}")
    for repeat in sorted({row.repeat for row in rows}):
        by_format = {
            row.matrix_format: row for row in rows if row.repeat == repeat
        }
        if {"csr", "bsr"} <= by_format.keys():
            csr, bsr = by_format["csr"], by_format["bsr"]
            difference = _coefficient_difference(csr, bsr)
            print(
                f"\n  repeat {repeat} BSR/CSR primal coefficient relative "
                f"difference: {difference:.3e}"
            )
            if bsr.amgx_solve_seconds > 0.0:
                print(
                    "  AMGX solve speed ratio (CSR / BSR): "
                    f"{csr.amgx_solve_seconds / bsr.amgx_solve_seconds:.3f}"
                )
            print(
                "  compressed-pattern byte ratio (BSR / CSR): "
                f"{bsr.pattern_bytes / csr.pattern_bytes:.4f}"
            )
    if len({row.repeat for row in rows}) > 1:
        print("\n  medians over repeats")
        for matrix_format in ("csr", "bsr"):
            selected = [row for row in rows if row.matrix_format == matrix_format]
            if not selected:
                continue
            print(
                f"    {matrix_format}: setup="
                f"{np.median([row.amgx_setup_seconds for row in selected]):.6f}s, "
                f"solve={np.median([row.amgx_solve_seconds for row in selected]):.6f}s, "
                f"total={np.median([row.total_seconds for row in selected]):.6f}s"
            )


def _write_csv(path, rows, *, domain, triangles, order, trace_dofs, tau):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "domain",
        "triangles",
        "order",
        "face_block_size",
        "trace_dofs",
        "tau",
        "repeat",
        "matrix_format",
        "config",
        "solver",
        "preconditioner",
        "matrix_bytes",
        "pattern_bytes",
        "assembly_seconds",
        "kernel_seconds",
        "amgx_setup_seconds",
        "amgx_solve_seconds",
        "reconstruction_seconds",
        "total_seconds",
        "iterations",
        "relative_residual",
        "l2_error",
        "coefficient_relative_difference",
    )
    differences = {}
    for repeat in sorted({row.repeat for row in rows}):
        by_format = {
            row.matrix_format: row for row in rows if row.repeat == repeat
        }
        if {"csr", "bsr"} <= by_format.keys():
            differences[repeat] = _coefficient_difference(
                by_format["csr"], by_format["bsr"]
            )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "domain": domain,
                    "triangles": triangles,
                    "order": order,
                    "face_block_size": order + 1,
                    "trace_dofs": trace_dofs,
                    "tau": tau,
                    "repeat": row.repeat,
                    "matrix_format": row.matrix_format,
                    "config": row.config,
                    "solver": row.solver,
                    "preconditioner": row.preconditioner,
                    "matrix_bytes": row.matrix_bytes,
                    "pattern_bytes": row.pattern_bytes,
                    "assembly_seconds": row.assembly_seconds,
                    "kernel_seconds": row.kernel_seconds,
                    "amgx_setup_seconds": row.amgx_setup_seconds,
                    "amgx_solve_seconds": row.amgx_solve_seconds,
                    "reconstruction_seconds": row.reconstruction_seconds,
                    "total_seconds": row.total_seconds,
                    "iterations": row.iterations,
                    "relative_residual": row.relative_residual,
                    "l2_error": row.l2_error,
                    "coefficient_relative_difference": differences.get(row.repeat),
                }
            )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ny = args.nx if args.ny is None else args.ny
    if args.order < 1 or args.order > 6:
        raise ValueError("this raw-CUDA diffusion benchmark supports 1 <= p <= 6")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    if args.domain == "disk":
        if args.mesh_size <= 0.0:
            raise ValueError("mesh-size must be positive")
        if args.radius <= 0.0:
            raise ValueError("radius must be positive")
        domain_label = (
            f"unstructured disk, radius={args.radius:g}, "
            f"mesh-size={args.mesh_size:g}"
        )
        print(f"building {domain_label} ...", flush=True)
        mesh = gmsh_disc_mesh(
            args.mesh_size,
            center=(0.0, 0.0),
            radius=args.radius,
            verbosity=0,
        )
    else:
        if args.nx <= 0 or ny <= 0:
            raise ValueError("nx and ny must be positive")
        domain_label = f"structured rectangle, {args.nx} x {ny}"
        print(f"building {domain_label} ...", flush=True)
        mesh = rectangle_mesh(
            args.nx,
            ny,
            xlim=(-1.0, 1.0),
            ylim=(-1.0, 1.0),
        )
    if not args.allow_small and mesh.num_tri <= args.minimum_triangles:
        raise ValueError(
            f"mesh has {mesh.num_tri:,} triangles; expected more than "
            f"{args.minimum_triangles:,}. Increase --nx/--ny or pass --allow-small."
        )
    print(f"building p={args.order} DG space for {mesh.num_tri:,} triangles ...", flush=True)
    space = DGSpace(mesh, args.order, basis_type=args.basis)
    problem = trigonometric_poisson_case()
    tau = float(
        args.tau
        if args.tau is not None
        else GlobalLengthDiffusion(gamma_d=1.0, domain_length="auto").resolve(
            problem.diffusion, space
        )
    )
    rows = []
    cp = require_cupy()
    for repeat in range(1, args.repeats + 1):
        if args.only == "both":
            bsr_first = args.format_order == "bsr-first" or (
                args.format_order == "alternate" and repeat % 2 == 0
            )
            requested = ("bsr", "csr") if bsr_first else ("csr", "bsr")
        else:
            requested = (args.only,)
        for matrix_format in requested:
            selected = args.csr_amgx_config
            if matrix_format == "bsr":
                selected = args.bsr_amgx_config or _default_bsr_config(args.order)
            config, config_path = load_amgx_config(
                selected,
                tolerance=args.rtol,
                maxiter=args.maxiter,
                verbose=args.verbosity,
            )
            print(
                f"\nrepeat {repeat}: running raw-CUDA {matrix_format.upper()} with "
                f"{describe_amgx_solver(config)}/"
                f"{describe_amgx_preconditioner(config)} ...",
                flush=True,
            )
            rows.append(
                _run_one(
                    matrix_format,
                    repeat=repeat,
                    space=space,
                    problem=problem,
                    tau=tau,
                    config=config,
                    config_path=config_path,
                    args=args,
                )
            )
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
    trace_dofs = int(mesh.int_edges_inds.size * space.layout.edg_dof)
    _print_results(
        rows,
        domain=domain_label,
        triangles=mesh.num_tri,
        order=space.order,
        trace_dofs=trace_dofs,
        tau=tau,
    )
    if args.csv is not None:
        _write_csv(
            args.csv,
            rows,
            domain=domain_label,
            triangles=mesh.num_tri,
            order=space.order,
            trace_dofs=trace_dofs,
            tau=tau,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
