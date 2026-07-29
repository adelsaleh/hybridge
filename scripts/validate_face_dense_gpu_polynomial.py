"""Validate and benchmark harmonic-Ritz polynomial preconditioning on CUDA.

A degree sweep can use one shared Arnoldi probe at the largest requested
polynomial degree.  Every smaller candidate is then extracted from a leading
Hessenberg submatrix, avoiding repeated global matvec/preconditioner setup
work.  CUDA JIT initialization is reported separately from numerical setup.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from hdgfem.assembly import hdg as hdg_assembly
from hdgfem.backends.cupy import require_cupy_device
from hdgfem.backends.cupy_face_dense import CuPyFaceDenseOperator
from hdgfem.backends.cupy_gmres import restarted_gmres_cupy
from hdgfem.backends.cupy_polynomial import (
    CuPyPolynomialPreconditioner,
    initialize_polynomial_kernels_cupy,
    setup_polynomial_arnoldi_probe_cupy,
)
from hdgfem.backends.cupy_preconditionners import (
    CuPyFaceAdditiveSchwarzPreconditioner,
    CuPyFaceBlockJacobiPreconditioner,
)
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers.diff_rea import diffusion_element_boundary_mats, local_solvers
from hdgfem.solvers.diff_rea_face_dense import assemble_diffusion_face_dense_components
from scripts.diff_rea_cases import quadratic_poisson_case


METHODS = ("none", "block_jacobi", "asm", "poly", "bj_poly", "asm_poly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, default=64)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--degrees", nargs="+", type=int, default=[2, 4, 6, 8, 10])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=["asm", "asm_poly"],
    )
    parser.add_argument(
        "--setup-mode",
        choices=("shared", "independent"),
        default="shared",
        help="reuse one maximum-degree Arnoldi probe or rebuild each degree",
    )
    parser.add_argument(
        "--rhs-counts",
        nargs="+",
        type=int,
        default=[1, 5, 20],
        help="numbers of right-hand sides used for setup amortization summaries",
    )
    parser.add_argument("--restart", type=int, default=100)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument(
        "--outer-orthogonalization",
        choices=("mgs", "mgs2", "cgs", "cgs2"),
        default="cgs",
    )
    parser.add_argument(
        "--setup-orthogonalization",
        choices=("mgs", "mgs2", "cgs", "cgs2"),
        default="cgs2",
    )
    parser.add_argument(
        "--local-solver",
        choices=("cpu_inverse", "gpu_inverse", "cublas_inverse"),
        default="cublas_inverse",
    )
    parser.add_argument(
        "--preconditioner-application",
        choices=("matmul", "raw"),
        default="raw",
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--apply-warmup", type=int, default=3)
    parser.add_argument("--apply-repeats", type=int, default=30)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--output-prefix", type=Path, default=None)
    args = parser.parse_args()
    if args.mesh <= 0 or args.order < 0:
        parser.error("mesh must be positive and order non-negative")
    if any(degree <= 0 for degree in args.degrees):
        parser.error("all polynomial degrees must be positive")
    if any(count <= 0 for count in args.rhs_counts):
        parser.error("all rhs-counts must be positive")
    if args.restart <= 0 or args.max_iterations <= 0:
        parser.error("restart and max-iterations must be positive")
    if args.apply_warmup < 0 or args.apply_repeats <= 0:
        parser.error("invalid application benchmark counts")
    args.degrees = sorted(set(args.degrees))
    args.rhs_counts = sorted(set(args.rhs_counts))
    return args


def assemble_case(mesh_size: int, order: int):
    diffusion, reaction, source, boundary_condition = quadratic_poisson_case()
    space = DGSpace(
        rectangle_mesh(mesh_size, mesh_size),
        order,
        basis_type="dub_orth",
    )
    stabilization = 1.3
    local_solver = local_solvers(
        reaction,
        stabilization,
        space,
        backend="numpy",
        diffusion=diffusion,
    )
    element_boundary_mats = diffusion_element_boundary_mats(stabilization, space)
    source_rhs = hdg_assembly.block_source_moments(
        source,
        space,
        num_blocks=3,
        source_block=0,
    )
    assembly = assemble_diffusion_face_dense_components(
        local_solver,
        element_boundary_mats,
        source_rhs,
        boundary_condition,
        stabilization,
        space,
    )
    return space, assembly


def _device_name(cp: Any, device_id: int) -> str:
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    name = properties.get("name", properties.get(b"name", "unknown"))
    return name.decode(errors="replace") if isinstance(name, bytes) else str(name)


def benchmark_apply(
    cp: Any,
    preconditioner: Any,
    vector: Any,
    warmup: int,
    repeats: int,
) -> float:
    output = cp.empty_like(vector)
    for _ in range(warmup):
        preconditioner.apply_into(vector, output)
    cp.cuda.get_current_stream().synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = cp.cuda.Event()
        stop = cp.cuda.Event()
        start.record()
        preconditioner.apply_into(vector, output)
        stop.record()
        stop.synchronize()
        samples.append(float(cp.cuda.get_elapsed_time(start, stop)))
    return float(np.median(samples))


def timed_solve(
    cp: Any,
    operator: Any,
    rhs: Any,
    preconditioner: Any,
    args: argparse.Namespace,
):
    cp.cuda.get_current_stream().synchronize()
    start = perf_counter()
    result = restarted_gmres_cupy(
        operator,
        rhs,
        restart=args.restart,
        max_iterations=args.max_iterations,
        rtol=args.rtol,
        atol=args.atol,
        preconditioner=preconditioner,
        orthogonalization=args.outer_orthogonalization,
    )
    cp.cuda.get_current_stream().synchronize()
    return result, 1.0e3 * (perf_counter() - start)


def write_outputs(prefix: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump({"metadata": metadata, "rows": rows}, file, indent=2)
    return csv_path, json_path


def _print_amortized_selection(
    rows: list[dict[str, Any]], rhs_counts: list[int]
) -> None:
    polynomial_methods = sorted(
        {row["method"] for row in rows if int(row.get("degree", 0)) > 0}
    )
    if not polynomial_methods:
        return
    print("\nBest converged degree by amortized setup + solve time")
    print("-" * 67)
    for method in polynomial_methods:
        candidates = [
            row
            for row in rows
            if row["method"] == method and row["status"] == "converged"
        ]
        if not candidates:
            print(f"{method:10s}: no converged candidate")
            continue
        summaries: list[str] = []
        for rhs_count in rhs_counts:
            winner = min(
                candidates,
                key=lambda row: float(row["solve_ms"])
                + float(row["total_setup_ms"]) / rhs_count,
            )
            amortized = float(winner["solve_ms"]) + float(
                winner["total_setup_ms"]
            ) / rhs_count
            summaries.append(
                f"N={rhs_count}: P={winner['degree']} ({amortized:.3f} ms/RHS)"
            )
        print(f"{method:10s}: " + "; ".join(summaries))


def main() -> None:
    args = parse_args()
    cp = require_cupy_device()
    device_id = int(cp.cuda.Device().id) if args.device is None else int(args.device)

    with cp.cuda.Device(device_id):
        space, assembly = assemble_case(args.mesh, args.order)
        system = assembly.eliminated_system
        operator = CuPyFaceDenseOperator.from_system(
            system,
            implementation="raw",
            device_id=device_id,
        )
        rhs = operator.to_device(system.rhs)

        need_bj = any(method in {"block_jacobi", "bj_poly"} for method in args.methods)
        need_asm = any(method in {"asm", "asm_poly"} for method in args.methods)
        block_jacobi = None
        asm = None
        if need_bj:
            block_jacobi = CuPyFaceBlockJacobiPreconditioner.from_system(
                system,
                device_id=device_id,
                local_solver=args.local_solver,
                application=args.preconditioner_application,
            )
        if need_asm:
            asm = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
                system,
                assembly.element_blocks,
                space.mesh.loc2glob_edge,
                device_id=device_id,
                local_solver=args.local_solver,
                application=args.preconditioner_application,
            )

        cp.cuda.get_current_stream().synchronize()
        jit_start = perf_counter()
        initialize_polynomial_kernels_cupy(
            dtype=operator.dtype,
            device_id=device_id,
        )
        kernel_initialization_ms = 1.0e3 * (perf_counter() - jit_start)

        print("GPU harmonic-Ritz polynomial preconditioner study")
        print("=" * 79)
        print(f"Device / dofs       : {_device_name(cp, device_id)} / {system.num_dofs}")
        print(f"Mesh / order        : {args.mesh}x{args.mesh} / p={args.order}")
        print(f"Restart / tolerance : {args.restart} / {args.rtol:.1e}")
        print(
            f"Outer / setup orth. : {args.outer_orthogonalization} / "
            f"{args.setup_orthogonalization}"
        )
        print(f"Polynomial setup    : {args.setup_mode}")
        print(f"One-time kernel JIT : {kernel_initialization_ms:.3f} ms")
        print()
        print(
            "method    degree probe[ms] cand[ms] apply[ms] iter cycles "
            "solve[ms] cold[ms]   relres      status   setup-orthF"
        )
        print("-" * 126)

        rows: list[dict[str, Any]] = []
        baseline_methods = [
            method
            for method in args.methods
            if not method.endswith("_poly") and method != "poly"
        ]
        for method in baseline_methods:
            preconditioner = {
                "none": None,
                "block_jacobi": block_jacobi,
                "asm": asm,
            }[method]
            apply_ms = (
                float("nan")
                if preconditioner is None
                else benchmark_apply(
                    cp,
                    preconditioner,
                    rhs,
                    args.apply_warmup,
                    args.apply_repeats,
                )
            )
            result, solve_ms = timed_solve(cp, operator, rhs, preconditioner, args)
            row = {
                "method": method,
                "degree": 0,
                "requested_degree": 0,
                "probe_ms": 0.0,
                "candidate_setup_ms": 0.0,
                "total_setup_ms": 0.0,
                "cold_total_ms": solve_ms,
                "apply_median_ms": apply_ms,
                "iterations": result.iterations,
                "restart_cycles": result.restart_cycles,
                "solve_ms": solve_ms,
                "relative_residual": result.relative_residual,
                "status": result.status,
                "setup_orthogonality_frobenius": float("nan"),
                "minimum_root_magnitude": float("nan"),
                "maximum_root_magnitude": float("nan"),
            }
            rows.append(row)
            print(
                f"{method:10s} {0:6d} {0.0:9.3f} {0.0:8.3f} {apply_ms:9.4f} "
                f"{result.iterations:4d} {result.restart_cycles:6d} "
                f"{solve_ms:9.3f} {solve_ms:8.3f} {result.relative_residual:10.3e} "
                f"{result.status:>11s}"
            )

        polynomial_methods = [
            method
            for method in args.methods
            if method == "poly" or method.endswith("_poly")
        ]
        for method in polynomial_methods:
            base = {"poly": None, "bj_poly": block_jacobi, "asm_poly": asm}[method]
            probe = None
            shared_probe_ms = 0.0
            if args.setup_mode == "shared":
                cp.cuda.get_current_stream().synchronize()
                probe_start = perf_counter()
                probe = setup_polynomial_arnoldi_probe_cupy(
                    operator,
                    maximum_degree=max(args.degrees),
                    base_preconditioner=base,
                    seed=args.seed,
                    orthogonalization=args.setup_orthogonalization,
                )
                cp.cuda.get_current_stream().synchronize()
                shared_probe_ms = 1.0e3 * (perf_counter() - probe_start)

            for degree in args.degrees:
                cp.cuda.get_current_stream().synchronize()
                candidate_start = perf_counter()
                if probe is None:
                    polynomial = CuPyPolynomialPreconditioner.from_operator(
                        operator,
                        degree=degree,
                        base_preconditioner=base,
                        seed=args.seed,
                        setup_orthogonalization=args.setup_orthogonalization,
                    )
                    cp.cuda.get_current_stream().synchronize()
                    independent_setup_ms = 1.0e3 * (
                        perf_counter() - candidate_start
                    )
                    probe_ms = independent_setup_ms
                    candidate_setup_ms = 0.0
                    total_setup_ms = independent_setup_ms
                else:
                    polynomial = CuPyPolynomialPreconditioner.from_probe(
                        operator,
                        probe=probe,
                        degree=degree,
                        base_preconditioner=base,
                    )
                    cp.cuda.get_current_stream().synchronize()
                    candidate_setup_ms = 1.0e3 * (
                        perf_counter() - candidate_start
                    )
                    probe_ms = shared_probe_ms
                    total_setup_ms = probe_ms + candidate_setup_ms

                apply_ms = benchmark_apply(
                    cp,
                    polynomial,
                    rhs,
                    args.apply_warmup,
                    args.apply_repeats,
                )
                result, solve_ms = timed_solve(cp, operator, rhs, polynomial, args)
                setup = polynomial.setup
                assert setup is not None
                root_magnitudes = np.abs(polynomial.roots)
                cold_total_ms = total_setup_ms + solve_ms
                row = {
                    "method": method,
                    "degree": polynomial.degree,
                    "requested_degree": degree,
                    "probe_ms": probe_ms,
                    "candidate_setup_ms": candidate_setup_ms,
                    "total_setup_ms": total_setup_ms,
                    "cold_total_ms": cold_total_ms,
                    "apply_median_ms": apply_ms,
                    "iterations": result.iterations,
                    "restart_cycles": result.restart_cycles,
                    "solve_ms": solve_ms,
                    "relative_residual": result.relative_residual,
                    "status": result.status,
                    "setup_orthogonality_frobenius": (
                        setup.frobenius_orthogonality_defect
                    ),
                    "setup_maximum_offdiagonal": setup.maximum_offdiagonal,
                    "minimum_root_magnitude": float(np.min(root_magnitudes)),
                    "maximum_root_magnitude": float(np.max(root_magnitudes)),
                    "setup_matvec_count": setup.matvec_count,
                    "setup_base_preconditioner_count": (
                        setup.base_preconditioner_count
                    ),
                    "shared_probe_requested_degree": (
                        0 if probe is None else probe.requested_degree
                    ),
                    "shared_probe_matvec_count": (
                        0 if probe is None else probe.matvec_count
                    ),
                    "matvecs_per_application": polynomial.matvecs_per_application,
                    "base_calls_per_application": (
                        polynomial.base_preconditioner_calls_per_application
                    ),
                }
                for rhs_count in args.rhs_counts:
                    row[f"amortized_ms_per_rhs_{rhs_count}"] = (
                        solve_ms + total_setup_ms / rhs_count
                    )
                rows.append(row)
                print(
                    f"{method:10s} {polynomial.degree:6d} {probe_ms:9.3f} "
                    f"{candidate_setup_ms:8.3f} {apply_ms:9.4f} "
                    f"{result.iterations:4d} {result.restart_cycles:6d} "
                    f"{solve_ms:9.3f} {cold_total_ms:8.3f} "
                    f"{result.relative_residual:10.3e} {result.status:>11s} "
                    f"{setup.frobenius_orthogonality_defect:13.3e}"
                )

        _print_amortized_selection(rows, args.rhs_counts)

        if args.output_prefix is not None:
            metadata = {
                "device": _device_name(cp, device_id),
                "device_id": device_id,
                "mesh": args.mesh,
                "order": args.order,
                "num_dofs": system.num_dofs,
                "restart": args.restart,
                "max_iterations": args.max_iterations,
                "rtol": args.rtol,
                "outer_orthogonalization": args.outer_orthogonalization,
                "setup_orthogonalization": args.setup_orthogonalization,
                "setup_mode": args.setup_mode,
                "rhs_counts": args.rhs_counts,
                "seed": args.seed,
                "kernel_initialization_ms": kernel_initialization_ms,
            }
            csv_path, json_path = write_outputs(args.output_prefix, rows, metadata)
            print(f"\nCSV report : {csv_path}")
            print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()