"""Benchmark reusable GPU GMRES workspaces over multiple right-hand sides.

The matrix, additive-Schwarz preconditioner, shared harmonic-Ritz probe, and
polynomial preconditioner are constructed once.  Sequential scaled right-hand
sides are then solved either through the legacy per-call allocation path or a
reusable :class:`CuPyRestartedGMRESSolver` with preallocated solution vectors.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from hdgfem.backends.cupy import require_cupy_device
from hdgfem.backends.cupy_face_dense import CuPyFaceDenseOperator
from hdgfem.backends.cupy_gmres import (
    CuPyRestartedGMRESSolver,
    restarted_gmres_cupy,
)
from hdgfem.backends.cupy_polynomial import (
    CuPyPolynomialPreconditioner,
    initialize_polynomial_kernels_cupy,
    setup_polynomial_arnoldi_probe_cupy,
)
from hdgfem.backends.cupy_preconditionners import (
    CuPyFaceAdditiveSchwarzPreconditioner,
)
from scripts.validate_face_dense_gpu_polynomial import assemble_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, default=64)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--degree", type=int, default=18)
    parser.add_argument("--restart", type=int, default=100)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument(
        "--rhs-counts", nargs="+", type=int, default=[1, 2, 5, 10, 20]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1729)
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
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--output-prefix", type=Path, default=None)
    args = parser.parse_args()
    if args.mesh <= 0 or args.order < 0:
        parser.error("mesh must be positive and order non-negative")
    if args.degree <= 0 or args.restart <= 0 or args.max_iterations <= 0:
        parser.error("degree, restart, and max-iterations must be positive")
    if args.repeats <= 0 or any(value <= 0 for value in args.rhs_counts):
        parser.error("repeats and rhs-counts must be positive")
    args.rhs_counts = sorted(set(args.rhs_counts))
    return args


def _device_name(cp: Any, device_id: int) -> str:
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    name = properties.get("name", properties.get(b"name", "unknown"))
    return name.decode(errors="replace") if isinstance(name, bytes) else str(name)


def _synchronize(cp: Any) -> None:
    cp.cuda.get_current_stream().synchronize()


def _time_sequence(cp: Any, operation, repeats: int) -> tuple[float, list[Any]]:
    samples: list[float] = []
    last_results: list[Any] = []
    for _ in range(repeats):
        _synchronize(cp)
        start = perf_counter()
        last_results = operation()
        _synchronize(cp)
        samples.append(1.0e3 * (perf_counter() - start))
    return float(np.median(samples)), last_results


def _write_outputs(
    prefix: Path,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump({"metadata": metadata, "rows": rows}, file, indent=2)
    return csv_path, json_path


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
        base_rhs = operator.to_device(system.rhs).reshape(-1)
        asm = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
            system,
            assembly.element_blocks,
            space.mesh.loc2glob_edge,
            device_id=device_id,
            local_solver=args.local_solver,
            application=args.preconditioner_application,
        )
        initialize_polynomial_kernels_cupy(
            dtype=operator.dtype,
            device_id=device_id,
        )
        _synchronize(cp)
        probe_start = perf_counter()
        probe = setup_polynomial_arnoldi_probe_cupy(
            operator,
            maximum_degree=args.degree,
            base_preconditioner=asm,
            seed=args.seed,
            orthogonalization=args.setup_orthogonalization,
        )
        _synchronize(cp)
        probe_ms = 1.0e3 * (perf_counter() - probe_start)
        candidate_start = perf_counter()
        polynomial = CuPyPolynomialPreconditioner.from_probe(
            operator,
            probe=probe,
            degree=args.degree,
            base_preconditioner=asm,
        )
        _synchronize(cp)
        candidate_ms = 1.0e3 * (perf_counter() - candidate_start)

        maximum_rhs = max(args.rhs_counts)
        rhs_batch = cp.empty((maximum_rhs, operator.num_dofs), dtype=operator.dtype)
        solution_batch = cp.empty_like(rhs_batch)
        scales = 1.0 + 0.05 * cp.arange(maximum_rhs, dtype=operator.dtype)
        rhs_batch[:] = scales[:, None] * base_rhs[None, :]

        reusable_solver = CuPyRestartedGMRESSolver(
            operator,
            restart=args.restart,
            max_iterations=args.max_iterations,
            rtol=args.rtol,
            atol=args.atol,
            preconditioner=polynomial,
            orthogonalization=args.outer_orthogonalization,
        )

        # Warm both execution paths after all CUDA kernels and memory-pool
        # allocations have been initialized.
        reusable_solver.solve(rhs_batch[0], solution_out=solution_batch[0])
        restarted_gmres_cupy(
            operator,
            rhs_batch[0],
            restart=args.restart,
            max_iterations=args.max_iterations,
            rtol=args.rtol,
            atol=args.atol,
            preconditioner=polynomial,
            orthogonalization=args.outer_orthogonalization,
        )
        _synchronize(cp)

        rows: list[dict[str, Any]] = []
        print("GPU GMRES reusable-workspace multi-RHS study")
        print("=" * 78)
        print(f"Device / dofs       : {_device_name(cp, device_id)} / {operator.num_dofs}")
        print(f"Mesh / order        : {args.mesh}x{args.mesh} / p={args.order}")
        print(f"Polynomial degree   : {args.degree}")
        print(f"Restart / tolerance : {args.restart} / {args.rtol:.1e}")
        print(f"Shared setup        : {probe_ms + candidate_ms:.3f} ms")
        print(
            "Workspace storage   : "
            f"{reusable_solver.workspace_device_bytes / 2**20:.2f} MiB"
        )
        print()
        print(
            "rhs  mode       total[ms]  ms/RHS  iter/RHS  max-relres   status"
        )
        print("-" * 78)

        for rhs_count in args.rhs_counts:
            def legacy_operation() -> list[Any]:
                results = []
                for index in range(rhs_count):
                    results.append(
                        restarted_gmres_cupy(
                            operator,
                            rhs_batch[index],
                            restart=args.restart,
                            max_iterations=args.max_iterations,
                            rtol=args.rtol,
                            atol=args.atol,
                            preconditioner=polynomial,
                            orthogonalization=args.outer_orthogonalization,
                        )
                    )
                return results

            def reused_operation() -> list[Any]:
                results = []
                for index in range(rhs_count):
                    results.append(
                        reusable_solver.solve(
                            rhs_batch[index],
                            solution_out=solution_batch[index],
                        )
                    )
                return results

            for mode, operation in (
                ("legacy", legacy_operation),
                ("reused", reused_operation),
            ):
                total_ms, results = _time_sequence(cp, operation, args.repeats)
                average_iterations = float(
                    np.mean([result.iterations for result in results])
                )
                maximum_relative_residual = float(
                    max(result.relative_residual for result in results)
                )
                converged = all(result.converged for result in results)
                row = {
                    "rhs_count": rhs_count,
                    "mode": mode,
                    "total_ms": total_ms,
                    "milliseconds_per_rhs": total_ms / rhs_count,
                    "average_iterations": average_iterations,
                    "maximum_relative_residual": maximum_relative_residual,
                    "status": "converged" if converged else "failed",
                    "probe_ms": probe_ms,
                    "candidate_ms": candidate_ms,
                    "workspace_device_bytes": reusable_solver.workspace_device_bytes,
                }
                rows.append(row)
                print(
                    f"{rhs_count:3d}  {mode:9s} {total_ms:10.3f} "
                    f"{total_ms / rhs_count:7.3f} {average_iterations:9.1f} "
                    f"{maximum_relative_residual:11.3e} {row['status']:>9s}"
                )

        metadata = {
            "device": _device_name(cp, device_id),
            "device_id": device_id,
            "mesh": args.mesh,
            "order": args.order,
            "num_dofs": int(operator.num_dofs),
            "degree": args.degree,
            "restart": args.restart,
            "rtol": args.rtol,
            "probe_ms": probe_ms,
            "candidate_ms": candidate_ms,
            "workspace_device_bytes": reusable_solver.workspace_device_bytes,
        }
        if args.output_prefix is not None:
            csv_path, json_path = _write_outputs(args.output_prefix, rows, metadata)
            print(f"\nCSV report : {csv_path}")
            print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()
