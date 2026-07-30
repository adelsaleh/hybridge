"""Systematic CUDA profiling for the face-dense HDG solver.

The script keeps correctness kernels unchanged and measures:

* CPU face-dense assembly;
* operator setup and host-to-device transfer;
* neighbour gather, dense face product, and complete matvec;
* block-Jacobi application;
* ASM restriction, local dense operation, prolongation, and total application;
* uninstrumented GMRES time-to-solution;
* one separately instrumented GMRES solve with operation attribution.

Example
-------

.. code-block:: bash

   PYTHONPATH=. python scripts/profile_face_dense_gpu.py \\
       --orders 1 2 3 4 \\
       --meshes 8 16 32 \\
       --boundary-mode eliminate \\
       --output-prefix results/face_dense_gpu
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
from hdgfem.backends.cupy_profiling import (
    CuPyGMRESProfiler,
    benchmark_cuda_call,
    profile_additive_schwarz,
    profile_block_jacobi,
    profile_face_dense_operator,
    time_synchronized_setup,
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--meshes", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--basis", default="dub_orth")
    parser.add_argument(
        "--boundary-mode",
        choices=("eliminate", "penalty"),
        default="eliminate",
    )
    parser.add_argument("--boundary-penalty", type=float, default=1.0e6)
    parser.add_argument("--stabilization", type=float, default=1.3)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
    )
    parser.add_argument(
        "--matvec-implementations",
        nargs="+",
        choices=("matmul", "raw", "raw_fused"),
        default=["raw_fused", "raw", "matmul"],
    )
    parser.add_argument(
        "--gmres-matvec-implementation",
        choices=("matmul", "raw", "raw_fused"),
        default="raw_fused",
        help="Operator used inside GMRES; independent of reporting order.",
    )
    parser.add_argument(
        "--preconditioner-application",
        choices=("matmul", "raw"),
        default="raw",
        help="Dense inverse application used by block-Jacobi and, by default, ASM.",
    )
    parser.add_argument(
        "--asm-application",
        choices=("matmul", "raw", "fused"),
        default=None,
        help=(
            "ASM-specific application path. 'fused' combines restriction and "
            "local inverse multiplication, then uses race-free prolongation."
        ),
    )
    parser.add_argument(
        "--preconditioners",
        nargs="+",
        choices=("none", "block_jacobi", "asm"),
        default=["none", "block_jacobi", "asm"],
    )
    parser.add_argument(
        "--local-solver",
        choices=("cpu_inverse", "gpu_inverse", "cublas_inverse", "gpu_solve"),
        default="cublas_inverse",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--restart", type=int, default=50)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument(
        "--orthogonalization",
        choices=("mgs", "mgs2", "cgs", "cgs2"),
        default=None,
        help=(
            "Arnoldi orthogonalization. When omitted, the legacy "
            "--reorthogonalize flag selects mgs2; otherwise mgs is used."
        ),
    )
    parser.add_argument(
        "--reorthogonalize",
        action="store_true",
        help="Legacy alias for --orthogonalization mgs2.",
    )
    parser.add_argument("--gmres-warmup", type=int, default=1)
    parser.add_argument("--skip-gmres", action="store_true")
    parser.add_argument("--skip-detailed-gmres", action="store_true")
    parser.add_argument("--detailed-max-iterations", type=int, default=100)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("face_dense_gpu_profile"),
    )
    args = parser.parse_args()

    if any(order < 0 for order in args.orders):
        parser.error("polynomial orders must be non-negative")
    if any(mesh <= 0 for mesh in args.meshes):
        parser.error("mesh sizes must be positive")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("warmup must be non-negative and repeats positive")
    if args.gmres_warmup < 0:
        parser.error("gmres-warmup must be non-negative")
    if args.detailed_max_iterations <= 0:
        parser.error("detailed-max-iterations must be positive")
    if args.orthogonalization is not None and args.reorthogonalize:
        parser.error(
            "--orthogonalization and --reorthogonalize cannot be combined"
        )
    if args.gmres_matvec_implementation not in args.matvec_implementations:
        parser.error(
            "--gmres-matvec-implementation must also appear in "
            "--matvec-implementations"
        )
    if args.local_solver == "gpu_solve" and args.preconditioner_application == "raw":
        parser.error("raw preconditioner application requires a precomputed inverse")
    return args


def _device_name(cp: Any, device_id: int) -> str:
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    name = properties.get("name", properties.get(b"name", "unknown"))
    if isinstance(name, bytes):
        return name.decode(errors="replace")
    return str(name)


def _assemble_case(
    *,
    mesh_size: int,
    order: int,
    basis: str,
    stabilization: float,
    boundary_penalty: float,
):
    diffusion, reaction, source, boundary_condition = quadratic_poisson_case()
    space = DGSpace(
        rectangle_mesh(mesh_size, mesh_size),
        order,
        basis_type=basis,
    )

    start = perf_counter()
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
        boundary_penalty=boundary_penalty,
    )
    assembly_ms = 1.0e3 * (perf_counter() - start)
    return space, assembly, assembly_ms


def _timing_fields(prefix: str, timing) -> dict[str, Any]:
    return {
        f"{prefix}_minimum_ms": timing.minimum_ms,
        f"{prefix}_median_ms": timing.median_ms,
        f"{prefix}_mean_ms": timing.mean_ms,
        f"{prefix}_std_ms": timing.standard_deviation_ms,
        f"{prefix}_p90_ms": timing.p90_ms,
    }


def _common_row(
    *,
    device_name: str,
    boundary_mode: str,
    mesh_size: int,
    order: int,
    space: DGSpace,
    system: Any,
    dtype_name: str,
    assembly_ms: float,
) -> dict[str, Any]:
    return {
        "device": device_name,
        "boundary_mode": boundary_mode,
        "mesh_nx": mesh_size,
        "mesh_ny": mesh_size,
        "polynomial_order": order,
        "num_elements": int(space.mesh.num_tri),
        "num_faces_global": int(space.mesh.num_edg),
        "num_system_faces": int(system.num_rows),
        "block_size": int(system.block_size),
        "num_slots": int(system.num_slots),
        "num_dofs": int(system.num_dofs),
        "dtype": dtype_name,
        "cpu_assembly_ms": assembly_ms,
    }


def _time_gmres_wall(cp: Any, solver_call, *, warmup: int) -> tuple[Any, float]:
    stream = cp.cuda.get_current_stream()
    for _ in range(warmup):
        solver_call()
    stream.synchronize()
    start = perf_counter()
    result = solver_call()
    stream.synchronize()
    return result, 1.0e3 * (perf_counter() - start)


def _write_outputs(
    prefix: Path,
    rows: list[dict[str, Any]],
    detailed: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")

    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload = {
        "metadata": metadata,
        "summary_rows": rows,
        "detailed_gmres_profiles": detailed,
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    return csv_path, json_path


def main() -> None:
    args = _parse_args()
    cp = require_cupy_device()
    device_id = int(cp.cuda.Device().id) if args.device is None else int(args.device)
    dtype = np.dtype(args.dtype)
    device_name = _device_name(cp, device_id)

    rows: list[dict[str, Any]] = []
    detailed_profiles: list[dict[str, Any]] = []

    print("Face-dense HDG CUDA profiling")
    print("=" * 40)
    print(f"Device          : {device_name} (id={device_id})")
    print(f"Boundary mode   : {args.boundary_mode}")
    print(f"Local solver    : {args.local_solver}")
    print(f"GMRES matvec    : {args.gmres_matvec_implementation}")
    resolved_orthogonalization = (
        args.orthogonalization
        if args.orthogonalization is not None
        else ("mgs2" if args.reorthogonalize else "mgs")
    )
    resolved_asm_application = (
        args.preconditioner_application
        if args.asm_application is None
        else args.asm_application
    )
    print(f"BJ apply        : {args.preconditioner_application}")
    print(f"ASM apply       : {resolved_asm_application}")
    print(f"Orthogonalize   : {resolved_orthogonalization}")
    print(f"Warmup/repeats  : {args.warmup}/{args.repeats}")
    print()

    with cp.cuda.Device(device_id):
        for mesh_size in args.meshes:
            for order in args.orders:
                space, assembly, assembly_ms = _assemble_case(
                    mesh_size=mesh_size,
                    order=order,
                    basis=args.basis,
                    stabilization=args.stabilization,
                    boundary_penalty=args.boundary_penalty,
                )
                system = (
                    assembly.eliminated_system
                    if args.boundary_mode == "eliminate"
                    else assembly.penalty_system
                )
                common = _common_row(
                    device_name=device_name,
                    boundary_mode=args.boundary_mode,
                    mesh_size=mesh_size,
                    order=order,
                    space=space,
                    system=system,
                    dtype_name=dtype.name,
                    assembly_ms=assembly_ms,
                )
                print(
                    f"mesh={mesh_size:4d}x{mesh_size:<4d} p={order}  "
                    f"NE={space.mesh.num_tri:7d}  dofs={system.num_dofs:9d}"
                )

                operators: dict[str, Any] = {}
                for implementation in args.matvec_implementations:
                    operator, setup = time_synchronized_setup(
                        lambda implementation=implementation: (
                            CuPyFaceDenseOperator.from_system(
                                system,
                                implementation=implementation,
                                dtype=dtype,
                                device_id=device_id,
                            )
                        ),
                        device_id=device_id,
                    )
                    operators[implementation] = operator
                    rng = np.random.default_rng(20260727 + mesh_size * 31 + order)
                    x = operator.to_device(
                        rng.standard_normal(system.rhs.shape).astype(dtype, copy=False)
                    )
                    out = cp.empty_like(x)
                    profile = profile_face_dense_operator(
                        operator,
                        x,
                        out,
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                    row = {
                        **common,
                        "record_type": "operator",
                        "implementation": implementation,
                        "setup_ms": setup.elapsed_ms,
                        "estimated_flops": profile.estimated_flops,
                        "median_gflops": profile.median_gflops,
                        "matrix_bytes": profile.matrix_bytes,
                        "vector_bytes": profile.vector_bytes,
                        **_timing_fields("total", profile.total),
                        **_timing_fields("gather", profile.gather),
                        **_timing_fields("dense", profile.dense_product),
                    }
                    rows.append(row)
                    print(
                        f"  matvec {implementation:6s}: "
                        f"{profile.total.median_ms:9.4f} ms  "
                        f"gather={profile.gather.median_ms:8.4f}  "
                        f"dense={profile.dense_product.median_ms:8.4f}"
                    )

                primary_operator = operators[args.gmres_matvec_implementation]
                rhs_gpu = primary_operator.to_device(system.rhs.astype(dtype, copy=False))
                random_gpu = primary_operator.to_device(
                    np.random.default_rng(991 + mesh_size + order)
                    .standard_normal(system.rhs.shape)
                    .astype(dtype, copy=False)
                )
                preconditioners: dict[str, Any | None] = {"none": None}

                if "block_jacobi" in args.preconditioners:
                    bj, setup = time_synchronized_setup(
                        lambda: CuPyFaceBlockJacobiPreconditioner.from_system(
                            system,
                            device_id=device_id,
                            dtype=dtype,
                            local_solver=args.local_solver,
                            application=args.preconditioner_application,
                        ),
                        device_id=device_id,
                    )
                    preconditioners["block_jacobi"] = bj
                    out = cp.empty_like(random_gpu)
                    timing = profile_block_jacobi(
                        bj,
                        random_gpu,
                        out,
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                    rows.append(
                        {
                            **common,
                            "record_type": "preconditioner",
                            "preconditioner": "block_jacobi",
                            "local_solver": args.local_solver,
                            "application": args.preconditioner_application,
                            "setup_ms": setup.elapsed_ms,
                            "allocates_during_apply": bj.allocates_during_apply,
                            "maximum_inverse_residual": bj.maximum_inverse_residual,
                            **_timing_fields("total", timing),
                        }
                    )
                    print(
                        f"  block-Jacobi  : {timing.median_ms:9.4f} ms  "
                        f"setup={setup.elapsed_ms:9.3f} ms"
                    )

                if "asm" in args.preconditioners:
                    asm, setup = time_synchronized_setup(
                        lambda: CuPyFaceAdditiveSchwarzPreconditioner.from_system(
                            system,
                            assembly.element_blocks,
                            space.mesh.loc2glob_edge,
                            device_id=device_id,
                            dtype=dtype,
                            local_solver=args.local_solver,
                            application=resolved_asm_application,
                        ),
                        device_id=device_id,
                    )
                    preconditioners["asm"] = asm
                    out = cp.empty_like(random_gpu)
                    timing = profile_additive_schwarz(
                        asm,
                        random_gpu,
                        out,
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                    rows.append(
                        {
                            **common,
                            "record_type": "preconditioner",
                            "preconditioner": "asm",
                            "local_solver": args.local_solver,
                            "application": resolved_asm_application,
                            "setup_ms": setup.elapsed_ms,
                            "allocates_during_apply": asm.allocates_during_apply,
                            "workspace_bytes": asm.workspace_bytes,
                            "restricted_workspace_bytes": asm.restricted_workspace_bytes,
                            "race_free_prolongation": asm.uses_race_free_prolongation,
                            "maximum_inverse_residual": asm.maximum_inverse_residual,
                            **_timing_fields("total", timing.total),
                            **_timing_fields("restriction", timing.restriction),
                            **_timing_fields("local_solve", timing.local_solve),
                            **_timing_fields("prolongation", timing.prolongation),
                        }
                    )
                    print(
                        f"  ASM           : {timing.total.median_ms:9.4f} ms  "
                        f"restrict={timing.restriction.median_ms:8.4f}  "
                        f"local={timing.local_solve.median_ms:8.4f}  "
                        f"prolong={timing.prolongation.median_ms:8.4f}"
                    )

                if not args.skip_gmres:
                    for name in args.preconditioners:
                        preconditioner = preconditioners.get(name)
                        if name != "none" and preconditioner is None:
                            continue

                        def solve(*, profiler=None, max_iterations=None):
                            return restarted_gmres_cupy(
                                primary_operator,
                                rhs_gpu,
                                restart=args.restart,
                                max_iterations=(
                                    args.max_iterations
                                    if max_iterations is None
                                    else max_iterations
                                ),
                                rtol=args.rtol,
                                atol=args.atol,
                                preconditioner=preconditioner,
                                orthogonalization=args.orthogonalization,
                                reorthogonalize=args.reorthogonalize,
                                profiler=profiler,
                            )

                        result, solve_ms = _time_gmres_wall(
                            cp,
                            lambda: solve(profiler=None),
                            warmup=args.gmres_warmup,
                        )
                        profiled_result = None
                        profile_summary = None
                        if not args.skip_detailed_gmres:
                            profiler = CuPyGMRESProfiler(device_id=device_id)
                            profiled_result = solve(
                                profiler=profiler,
                                max_iterations=min(
                                    args.max_iterations,
                                    args.detailed_max_iterations,
                                ),
                            )
                            profile_summary = profiler.finalize()

                        rows.append(
                            {
                                **common,
                                "record_type": "gmres",
                                "implementation": primary_operator.implementation,
                                "preconditioner": name,
                                "local_solver": (
                                    "none" if preconditioner is None else args.local_solver
                                ),
                                "preconditioner_application": (
                                    "none" if preconditioner is None
                                    else args.preconditioner_application
                                ),
                                "solve_wall_ms": solve_ms,
                                "converged": result.converged,
                                "status": result.status,
                                "iterations": result.iterations,
                                "restart_cycles": result.restart_cycles,
                                "relative_residual": result.relative_residual,
                                "matvec_count": result.matvec_count,
                                "preconditioner_count": result.preconditioner_count,
                                "dot_count": result.dot_count,
                                "axpy_count": result.axpy_count,
                                "norm_count": result.norm_count,
                                "basis_update_count": result.basis_update_count,
                                "orthogonalization": result.orthogonalization,
                                "basis_projection_count": (
                                    result.basis_projection_count
                                ),
                                "basis_correction_count": (
                                    result.basis_correction_count
                                ),
                                "coefficient_d2h_count": (
                                    result.coefficient_d2h_count
                                ),
                                "profiled_iterations": (
                                    None if profiled_result is None else profiled_result.iterations
                                ),
                                "profiled_total_gpu_operation_ms": (
                                    None
                                    if profile_summary is None
                                    else profile_summary.total_gpu_operation_ms
                                ),
                                "profiled_cpu_small_system_ms": (
                                    None
                                    if profile_summary is None
                                    else profile_summary.total_cpu_small_system_ms
                                ),
                            }
                        )
                        if profiled_result is not None and profile_summary is not None:
                            detailed_profiles.append(
                                {
                                    **common,
                                    "preconditioner": name,
                                    "solve_wall_ms_uninstrumented": solve_ms,
                                    "detailed_max_iterations": min(
                                        args.max_iterations,
                                        args.detailed_max_iterations,
                                    ),
                                    "result": {
                                        "converged": result.converged,
                                        "status": result.status,
                                        "iterations": result.iterations,
                                        "restart_cycles": result.restart_cycles,
                                        "relative_residual": result.relative_residual,
                                        "orthogonalization": result.orthogonalization,
                                        "basis_projection_count": (
                                            result.basis_projection_count
                                        ),
                                        "basis_correction_count": (
                                            result.basis_correction_count
                                        ),
                                        "coefficient_d2h_count": (
                                            result.coefficient_d2h_count
                                        ),
                                    },
                                    "profiled_result": {
                                        "converged": profiled_result.converged,
                                        "status": profiled_result.status,
                                        "iterations": profiled_result.iterations,
                                        "restart_cycles": profiled_result.restart_cycles,
                                        "relative_residual": profiled_result.relative_residual,
                                        "orthogonalization": (
                                            profiled_result.orthogonalization
                                        ),
                                        "basis_projection_count": (
                                            profiled_result.basis_projection_count
                                        ),
                                        "basis_correction_count": (
                                            profiled_result.basis_correction_count
                                        ),
                                        "coefficient_d2h_count": (
                                            profiled_result.coefficient_d2h_count
                                        ),
                                    },
                                    "operation_profile": profile_summary.to_dict(),
                                }
                            )
                        print(
                            f"  GMRES {name:12s}: {solve_ms:10.3f} ms  "
                            f"iters={result.iterations:5d}  "
                            f"relres={result.relative_residual:.3e}  "
                            f"status={result.status}"
                        )
                print()

    metadata = {
        "device_id": device_id,
        "device_name": device_name,
        "orders": args.orders,
        "meshes": args.meshes,
        "basis": args.basis,
        "boundary_mode": args.boundary_mode,
        "boundary_penalty": args.boundary_penalty,
        "stabilization": args.stabilization,
        "dtype": dtype.name,
        "matvec_implementations": args.matvec_implementations,
        "gmres_matvec_implementation": args.gmres_matvec_implementation,
        "preconditioner_application": args.preconditioner_application,
        "preconditioners": args.preconditioners,
        "local_solver": args.local_solver,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "restart": args.restart,
        "max_iterations": args.max_iterations,
        "rtol": args.rtol,
        "atol": args.atol,
        "orthogonalization": resolved_orthogonalization,
        "reorthogonalize": args.reorthogonalize,
        "skip_detailed_gmres": args.skip_detailed_gmres,
        "detailed_max_iterations": args.detailed_max_iterations,
    }
    csv_path, json_path = _write_outputs(
        args.output_prefix,
        rows,
        detailed_profiles,
        metadata,
    )
    print(f"CSV report : {csv_path}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()