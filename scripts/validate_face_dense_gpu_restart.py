"""Study GMRES restart size and Arnoldi orthogonality on one CUDA HDG case.

The authoritative time-to-solution is measured with orthogonality monitoring
turned off.  A second, untimed solve may compute one small Gram matrix per
restart cycle to quantify loss of orthogonality without contaminating the main
timing.
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
from hdgfem.backends.cupy_preconditionners import (
    CuPyFaceAdditiveSchwarzPreconditioner,
    CuPyFaceBlockJacobiPreconditioner,
)
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers.diff_rea import (
    diffusion_element_boundary_mats,
    local_solvers,
)
from hdgfem.solvers.diff_rea_face_dense import (
    assemble_diffusion_face_dense_components,
)
from scripts.diff_rea_cases import quadratic_poisson_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, default=64)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument(
        "--restarts",
        nargs="+",
        type=int,
        default=[10, 20, 30, 50, 75, 100],
    )
    parser.add_argument(
        "--orthogonalizations",
        nargs="+",
        choices=("cgs", "cgs2", "mgs", "mgs2"),
        default=["cgs", "cgs2"],
    )
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--preconditioner",
        choices=("none", "block_jacobi", "asm"),
        default="asm",
    )
    parser.add_argument(
        "--local-solver",
        choices=("cpu_inverse", "gpu_inverse", "cublas_inverse"),
        default="cublas_inverse",
    )
    parser.add_argument(
        "--skip-orthogonality",
        action="store_true",
        help="Skip the separate diagnostic solve that forms restart Gram matrices.",
    )
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--output-prefix", type=Path, default=None)
    args = parser.parse_args()

    if args.mesh <= 0 or args.order < 0:
        parser.error("mesh must be positive and order non-negative")
    if any(value <= 0 for value in args.restarts):
        parser.error("all restart dimensions must be positive")
    if args.max_iterations <= 0:
        parser.error("max-iterations must be positive")
    if args.warmup < 0:
        parser.error("warmup must be non-negative")
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
    element_boundary_mats = diffusion_element_boundary_mats(
        stabilization,
        space,
    )
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


def krylov_basis_mebibytes(restart: int, num_dofs: int, dtype: Any) -> float:
    """Storage of the ``(restart+1) x num_dofs`` Arnoldi basis."""

    return (restart + 1) * num_dofs * np.dtype(dtype).itemsize / 2.0**20


def _maximum_metrics(records) -> tuple[float, float, float]:
    if not records:
        return float("nan"), float("nan"), float("nan")
    return (
        max(record.frobenius_defect for record in records),
        max(record.maximum_offdiagonal for record in records),
        max(record.maximum_diagonal_error for record in records),
    )


def _device_name(cp: Any, device_id: int) -> str:
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    name = properties.get("name", properties.get(b"name", "unknown"))
    return name.decode(errors="replace") if isinstance(name, bytes) else str(name)


def _write_outputs(prefix: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]):
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


def main() -> None:
    args = parse_args()
    cp = require_cupy_device()
    device_id = int(cp.cuda.Device().id) if args.device is None else args.device

    with cp.cuda.Device(device_id):
        space, assembly = assemble_case(args.mesh, args.order)
        system = assembly.eliminated_system
        operator = CuPyFaceDenseOperator.from_system(
            system,
            implementation="raw",
            device_id=device_id,
        )
        preconditioner = None
        if args.preconditioner == "block_jacobi":
            preconditioner = CuPyFaceBlockJacobiPreconditioner.from_system(
                system,
                device_id=device_id,
                local_solver=args.local_solver,
                application="raw",
            )
        elif args.preconditioner == "asm":
            preconditioner = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
                system,
                assembly.element_blocks,
                space.mesh.loc2glob_edge,
                device_id=device_id,
                local_solver=args.local_solver,
                application="raw",
            )
        rhs = operator.to_device(system.rhs)

        rows: list[dict[str, Any]] = []
        host_solutions: dict[tuple[str, int], np.ndarray] = {}
        for mode in args.orthogonalizations:
            for restart in args.restarts:
                def solve(*, monitor: bool = False):
                    return restarted_gmres_cupy(
                        operator,
                        rhs,
                        restart=restart,
                        max_iterations=args.max_iterations,
                        rtol=args.rtol,
                        atol=args.atol,
                        preconditioner=preconditioner,
                        orthogonalization=mode,
                        monitor_orthogonality=monitor,
                    )

                for _ in range(args.warmup):
                    solve()
                operator.synchronize()
                start = perf_counter()
                result = solve()
                operator.synchronize()
                elapsed_ms = 1.0e3 * (perf_counter() - start)

                records = ()
                if not args.skip_orthogonality:
                    diagnostic = solve(monitor=True)
                    operator.synchronize()
                    records = diagnostic.orthogonality_records
                    if diagnostic.status != result.status:
                        raise RuntimeError(
                            "diagnostic and timed solves returned different statuses"
                        )
                frobenius, offdiagonal, diagonal = _maximum_metrics(records)
                host_solutions[(mode, restart)] = operator.to_host(result.solution)
                rows.append(
                    {
                        "orthogonalization": mode,
                        "restart": restart,
                        "time_ms": elapsed_ms,
                        "iterations": result.iterations,
                        "restart_cycles": result.restart_cycles,
                        "relative_residual": result.relative_residual,
                        "status": result.status,
                        "basis_mib": krylov_basis_mebibytes(
                            restart,
                            system.num_dofs,
                            np.dtype(system.blocks.dtype),
                        ),
                        "max_orthogonality_frobenius": frobenius,
                        "max_orthogonality_offdiagonal": offdiagonal,
                        "max_orthogonality_diagonal_error": diagonal,
                        "orthogonality_cycles": len(records),
                    }
                )

        converged = [row for row in rows if row["status"] == "converged"]
        reference_row = min(
            converged or rows,
            key=lambda row: (row["relative_residual"], -row["restart"]),
        )
        reference_key = (
            str(reference_row["orthogonalization"]),
            int(reference_row["restart"]),
        )
        reference_solution = host_solutions[reference_key]
        reference_norm = max(
            float(np.linalg.norm(reference_solution)),
            np.finfo(np.float64).eps,
        )
        for row in rows:
            key = (str(row["orthogonalization"]), int(row["restart"]))
            row["solution_difference"] = float(
                np.linalg.norm(host_solutions[key] - reference_solution)
                / reference_norm
            )

        print("GPU GMRES restart and orthogonality study")
        print("=" * 58)
        print(f"Device / dofs   : {_device_name(cp, device_id)} / {system.num_dofs}")
        print(f"Mesh / order    : {args.mesh}x{args.mesh} / p={args.order}")
        print(f"Preconditioner  : {args.preconditioner}")
        print(f"Tolerance       : {args.rtol:.1e}")
        print()
        print(
            "orth  restart  time[ms]  iter cycles   relres      status  "
            "basis[MiB]   ||I-VVt||F  max-offdiag  sol-diff"
        )
        print("-" * 111)
        for row in rows:
            print(
                f"{row['orthogonalization']:5s} {row['restart']:7d} "
                f"{row['time_ms']:9.3f} {row['iterations']:5d} "
                f"{row['restart_cycles']:6d} {row['relative_residual']:10.3e} "
                f"{row['status']:>12s} {row['basis_mib']:10.2f} "
                f"{row['max_orthogonality_frobenius']:12.3e} "
                f"{row['max_orthogonality_offdiagonal']:12.3e} "
                f"{row['solution_difference']:9.2e}"
            )

        if args.output_prefix is not None:
            metadata = {
                "device": _device_name(cp, device_id),
                "mesh": args.mesh,
                "order": args.order,
                "num_dofs": system.num_dofs,
                "preconditioner": args.preconditioner,
                "rtol": args.rtol,
                "max_iterations": args.max_iterations,
                "reference": {
                    "orthogonalization": reference_key[0],
                    "restart": reference_key[1],
                },
            }
            csv_path, json_path = _write_outputs(
                args.output_prefix,
                rows,
                metadata,
            )
            print()
            print(f"CSV report : {csv_path}")
            print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()
