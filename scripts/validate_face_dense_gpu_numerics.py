"""Strong CUDA numerical-regression matrix for the face-dense HDG solver.

The default cases are intentionally small enough for a direct NumPy reference.
They validate the GPU assembly, fused and two-stage matvecs, boundary modes,
preconditioners, and true-residual GMRES stopping criterion in one report.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from hdgfem.assembly.face_dense import (
    face_dense_matvec,
    normalize_penalty_rows,
)
from hdgfem.backends.cupy import require_cupy_device
from hdgfem.backends.cupy_assembly import CuPyGlobalFaceAssembler
from hdgfem.backends.cupy_face_dense import CuPyFaceDenseOperator
from hdgfem.backends.cupy_gmres import restarted_gmres_cupy
from hdgfem.backends.cupy_polynomial import CuPyPolynomialPreconditioner
from hdgfem.backends.cupy_preconditionners import (
    CuPyFaceAdditiveSchwarzPreconditioner,
    CuPyFaceBlockJacobiPreconditioner,
)
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers.diff_rea_face_dense import solve_diffusion_face_dense_direct
from scripts.diff_rea_cases import quadratic_poisson_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--orders", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--boundary-modes",
        nargs="+",
        choices=("eliminate", "penalty"),
        default=["eliminate", "penalty"],
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=("float32", "float64"),
        default=["float64"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("none", "block_jacobi", "asm", "asm_poly"),
        default=["none", "asm", "asm_poly"],
    )
    parser.add_argument("--polynomial-degree", type=int, default=8)
    parser.add_argument("--restart", type=int, default=50)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--boundary-penalty", type=float, default=1.0e6)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/gpu_numerical_regression"),
    )
    return parser.parse_args()


def relative_error(value: np.ndarray, reference: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(reference)), np.finfo(reference.dtype).eps)
    return float(np.linalg.norm(value - reference) / denominator)


def method_preconditioner(
    method: str,
    *,
    operator: Any,
    system: Any,
    assembly: Any,
    space: DGSpace,
    dtype: np.dtype,
):
    if method == "none":
        return None
    if method == "block_jacobi":
        return CuPyFaceBlockJacobiPreconditioner.from_system(
            system,
            dtype=dtype,
            device_id=operator.device_id,
            local_solver="cublas_inverse",
            application="raw",
        )
    asm = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
        system,
        assembly.element_blocks,
        space.mesh.loc2glob_edge,
        dtype=dtype,
        device_id=operator.device_id,
        local_solver="cublas_inverse",
        application="raw",
    )
    if method == "asm":
        return asm
    return CuPyPolynomialPreconditioner.from_operator(
        operator,
        degree=args_global.polynomial_degree,
        base_preconditioner=asm,
        setup_orthogonalization="cgs2",
    )


def main() -> None:
    global args_global
    args_global = parse_args()
    args = args_global
    cp = require_cupy_device()
    device_id = int(cp.cuda.Device().id) if args.device is None else int(args.device)
    device_name = cp.cuda.runtime.getDeviceProperties(device_id)["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    diffusion, reaction, source, boundary = quadratic_poisson_case()

    print("Strong GPU numerical validation matrix")
    print("=" * 88)
    print(f"Device: {device_name} (id={device_id})")

    for mesh_size in args.meshes:
        for order in args.orders:
            space = DGSpace(
                rectangle_mesh(mesh_size, mesh_size),
                order,
                basis_type="dub_orth",
            )
            for boundary_mode in args.boundary_modes:
                direct = solve_diffusion_face_dense_direct(
                    source,
                    reaction,
                    boundary,
                    space,
                    diffusion=diffusion,
                    stabilization=1.3,
                    boundary_mode=boundary_mode,
                    boundary_penalty=args.boundary_penalty,
                )
                assembly = direct.assembly
                system = direct.system

                for dtype_name in args.dtypes:
                    dtype = np.dtype(dtype_name)
                    case_rtol = max(args.rtol, 2.0e-5) if dtype == np.float32 else args.rtol
                    with cp.cuda.Device(device_id):
                        assembler = CuPyGlobalFaceAssembler.from_topology(
                            space.mesh.loc2glob_edge,
                            assembly.topology,
                            block_size=assembly.element_blocks.shape[-1],
                            dtype=dtype,
                            active_row_faces=space.mesh.interior_face_mask,
                            device_id=device_id,
                        )
                        element_device = cp.asarray(assembly.element_blocks, dtype=dtype)
                        gpu_global = assembler.assemble(element_device)
                        operator_raw = CuPyFaceDenseOperator.from_system(
                            system,
                            implementation="raw",
                            dtype=dtype,
                            device_id=device_id,
                        )
                        operator_fused = CuPyFaceDenseOperator.from_system(
                            system,
                            implementation="raw_fused",
                            dtype=dtype,
                            device_id=device_id,
                        )
                        rng = np.random.default_rng(
                            100000 * mesh_size + 100 * order + (0 if boundary_mode == "eliminate" else 1)
                        )
                        x_host = rng.standard_normal(system.num_dofs).astype(dtype)
                        x_device = operator_fused.to_device(x_host)
                        raw_value = operator_raw.matvec(x_device)
                        fused_value = operator_fused.matvec(x_device)
                        cp.cuda.get_current_stream().synchronize()
                        gpu_global_host = cp.asnumpy(gpu_global)
                        raw_host = cp.asnumpy(raw_value)
                        fused_host = cp.asnumpy(fused_value)

                    assembly_error = relative_error(
                        gpu_global_host,
                        assembly.interior_row_blocks.astype(dtype),
                    )
                    matvec_reference = face_dense_matvec(
                        system.blocks.astype(dtype),
                        system.neighbors,
                        x_host,
                    )
                    raw_error = relative_error(raw_host, matvec_reference)
                    fused_error = relative_error(fused_host, matvec_reference)
                    fused_raw_error = relative_error(fused_host, raw_host)

                    print(
                        f"mesh={mesh_size:3d} p={order} {boundary_mode:9s} "
                        f"{dtype_name:7s} dofs={system.num_dofs:6d} "
                        f"assembly={assembly_error:.2e} fused={fused_error:.2e}"
                    )

                    assembly_limit = 2.0e-5 if dtype == np.float32 else 2.0e-13
                    matvec_limit = 5.0e-5 if dtype == np.float32 else 5.0e-13
                    if assembly_error > assembly_limit:
                        failures.append(
                            f"assembly error {assembly_error:.3e} for mesh={mesh_size}, p={order}, "
                            f"mode={boundary_mode}, dtype={dtype_name}"
                        )
                    if max(raw_error, fused_error, fused_raw_error) > matvec_limit:
                        failures.append(
                            f"matvec error for mesh={mesh_size}, p={order}, mode={boundary_mode}, "
                            f"dtype={dtype_name}"
                        )

                    for method in args.methods:
                        solve_system = system
                        solve_operator = operator_fused
                        if boundary_mode == "penalty" and method in ("none", "block_jacobi"):
                            # Remove the artificial penalty scale from the
                            # unpreconditioned GMRES stopping norm.
                            boundary_faces = np.flatnonzero(
                                assembly.topology.incidence_count == 1
                            )
                            solve_system = normalize_penalty_rows(
                                system,
                                boundary_faces,
                                boundary_penalty=args.boundary_penalty,
                            )
                            solve_operator = CuPyFaceDenseOperator.from_system(
                                solve_system,
                                implementation="raw_fused",
                                dtype=dtype,
                                device_id=device_id,
                            )

                        preconditioner = method_preconditioner(
                            method,
                            operator=solve_operator,
                            system=solve_system,
                            assembly=assembly,
                            space=space,
                            dtype=dtype,
                        )
                        rhs = solve_operator.to_device(
                            solve_system.rhs.astype(dtype)
                        )
                        result = restarted_gmres_cupy(
                            solve_operator,
                            rhs,
                            restart=min(args.restart, solve_system.num_dofs),
                            max_iterations=args.max_iterations,
                            rtol=case_rtol,
                            preconditioner=preconditioner,
                            orthogonalization="cgs2",
                        )
                        solution_host = solve_operator.to_host(
                            result.solution
                        ).reshape(-1)
                        direct_solution = direct.system_solution.astype(dtype).reshape(-1)
                        solution_error = relative_error(solution_host, direct_solution)
                        row = {
                            "device": device_name,
                            "mesh": mesh_size,
                            "order": order,
                            "boundary_mode": boundary_mode,
                            "dtype": dtype_name,
                            "num_dofs": system.num_dofs,
                            "assembly_relative_error": assembly_error,
                            "raw_matvec_relative_error": raw_error,
                            "fused_matvec_relative_error": fused_error,
                            "fused_raw_relative_difference": fused_raw_error,
                            "method": method,
                            "polynomial_degree": args.polynomial_degree if method == "asm_poly" else 0,
                            "iterations": result.iterations,
                            "cycles": result.restart_cycles,
                            "relative_residual": result.relative_residual,
                            "solution_relative_error": solution_error,
                            "status": result.status,
                        }
                        rows.append(row)
                        print(
                            f"    {method:12s} iter={result.iterations:5d} "
                            f"relres={result.relative_residual:.2e} "
                            f"solerr={solution_error:.2e} {result.status}"
                        )
                        residual_limit = 5.0 * case_rtol
                        solution_limit = 2.0e-3 if dtype == np.float32 else max(5.0e-7, 100.0 * case_rtol)
                        if not result.converged or result.relative_residual > residual_limit:
                            failures.append(
                                f"solver {method} did not meet residual for mesh={mesh_size}, p={order}, "
                                f"mode={boundary_mode}, dtype={dtype_name}"
                            )
                        if solution_error > solution_limit:
                            failures.append(
                                f"solver {method} solution error {solution_error:.3e} for mesh={mesh_size}, "
                                f"p={order}, mode={boundary_mode}, dtype={dtype_name}"
                            )

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    if rows:
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    json_path.write_text(json.dumps({"rows": rows, "failures": failures}, indent=2))
    print(f"\nCSV report : {csv_path}")
    print(f"JSON report: {json_path}")
    if failures:
        print("\nValidation failures")
        print("-" * 40)
        for failure in failures:
            print(f"- {failure}")
        if args.strict:
            raise SystemExit(1)
    else:
        print("\nAll configured numerical checks passed.")


if __name__ == "__main__":
    main()
