"""Microbenchmark face-dense diffusion primitives on the radius-5 disk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from hdgfem.assembly import hdg as hdg_assembly
from hdgfem.backends.cupy import require_cupy_device
from hdgfem.backends.cupy_face_dense import CuPyFaceDenseOperator
from hdgfem.backends.cupy_gmres import restarted_gmres_cupy
from hdgfem.backends.cupy_polynomial import CuPyPolynomialPreconditioner
from hdgfem.backends.cupy_preconditionners import CuPyFaceAdditiveSchwarzPreconditioner
from hdgfem.backends.cupy_profiling import (
    CuPyGMRESProfiler,
    benchmark_cuda_call,
    profile_additive_schwarz,
    profile_face_dense_operator,
)
from hdgfem.core.mesh import gmsh_disc_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers.diffusion_reaction import diffusion_element_boundary_mats, local_solvers
from hdgfem.solvers.diffusion_face_dense import assemble_diffusion_face_dense_components
from scripts.diffusion_reaction.cases import trigonometric_poisson_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-size", type=float, default=0.075)
    parser.add_argument("--orders", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--degrees", nargs="+", type=int, default=[2, 4, 8, 12, 18])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mesh_size <= 0 or any(order < 1 for order in args.orders):
        parser.error("mesh size and orders must be positive")
    if any(degree < 1 for degree in args.degrees):
        parser.error("polynomial degrees must be positive")
    return args


def synchronized_wall_ms(cp, function):
    cp.cuda.get_current_stream().synchronize()
    start = perf_counter()
    result = function()
    cp.cuda.get_current_stream().synchronize()
    return result, 1.0e3 * (perf_counter() - start)


def relative_difference(cp, left, right) -> float:
    numerator = float(cp.linalg.norm(left.reshape(-1) - right.reshape(-1)).get())
    denominator = max(float(cp.linalg.norm(right.reshape(-1)).get()), np.finfo(float).eps)
    return numerator / denominator


def main() -> None:
    args = parse_args()
    cp = require_cupy_device()
    device_id = int(cp.cuda.Device().id)
    dtype = np.dtype("float64")
    problem = trigonometric_poisson_case()
    diffusion, reaction, source, boundary = problem
    tau = 1.0

    mesh = gmsh_disc_mesh(
        args.mesh_size,
        center=(0.0, 0.0),
        radius=5.0,
        verbosity=0,
    )
    triangles = int(mesh.num_elements)
    if triangles > 100_000:
        raise RuntimeError(f"refusing mesh with {triangles} elements (>100000)")

    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    device_name = properties.get("name", properties.get(b"name", "unknown"))
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    payload = {
        "metadata": {
            "device": device_name,
            "mesh_size": args.mesh_size,
            "triangles": triangles,
            "radius": 5.0,
            "basis": "dub_orth",
            "quadrature": "2p",
            "tau": tau,
            "rtol": args.rtol,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "orders": [],
    }
    print(f"mesh triangles={triangles} device={device_name}", flush=True)

    with cp.cuda.Device(device_id):
        for order in args.orders:
            print(f"p={order}: assembling", flush=True)
            assembly_start = perf_counter()
            space = DGSpace(
                mesh,
                order,
                basis_type="dub_orth",
                volume_quad_1d=2 * order,
                edge_quad_1d=2 * order,
            )
            local_solver = local_solvers(reaction, tau, space, diffusion=diffusion)
            boundary_mats = diffusion_element_boundary_mats(tau, space)
            source_rhs = hdg_assembly.block_source_moments(
                source, space, num_blocks=3, source_block=0
            )
            assembly = assemble_diffusion_face_dense_components(
                local_solver,
                boundary_mats,
                source_rhs,
                boundary,
                tau,
                space,
                boundary_penalty=1.0e6,
            )
            system = assembly.eliminated_system
            assembly_ms = 1.0e3 * (perf_counter() - assembly_start)
            rhs = cp.asarray(system.rhs.reshape(-1), dtype=dtype)
            output = cp.empty_like(rhs)

            order_row = {
                "order": order,
                "dofs": int(system.num_dofs),
                "block_size": int(system.block_size),
                "num_slots": int(system.num_slots),
                "assembly_wall_ms": assembly_ms,
                "operators": {},
                "asm": {},
                "polynomials": [],
            }

            operators = {}
            reference_output = None
            for implementation in ("matmul", "raw", "raw_fused"):
                operator = CuPyFaceDenseOperator.from_system(
                    system,
                    implementation=implementation,
                    dtype=dtype,
                    device_id=device_id,
                )
                profile = profile_face_dense_operator(
                    operator, rhs, output, warmup=args.warmup, repeats=args.repeats
                )
                operator.matvec_into(rhs, output)
                cp.cuda.get_current_stream().synchronize()
                current = output.copy()
                difference = 0.0 if reference_output is None else relative_difference(cp, current, reference_output)
                if reference_output is None:
                    reference_output = current
                operators[implementation] = operator
                order_row["operators"][implementation] = {
                    **profile.to_dict(),
                    "relative_difference": difference,
                }

            fastest_operator_name = min(
                order_row["operators"],
                key=lambda name: order_row["operators"][name]["total"]["median_ms"],
            )
            operator = operators[fastest_operator_name]
            order_row["fastest_operator"] = fastest_operator_name

            asms = {}
            asm_reference = None
            for application in ("matmul", "raw", "fused"):
                preconditioner = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
                    system,
                    assembly.element_blocks,
                    space.mesh.loc2glob_edge,
                    dtype=dtype,
                    device_id=device_id,
                    local_solver="cublas_inverse",
                    application=application,
                )
                profile = profile_additive_schwarz(
                    preconditioner, rhs, output, warmup=args.warmup, repeats=args.repeats
                )
                preconditioner.apply_into(rhs, output)
                cp.cuda.get_current_stream().synchronize()
                current = output.copy()
                difference = 0.0 if asm_reference is None else relative_difference(cp, current, asm_reference)
                if asm_reference is None:
                    asm_reference = current
                asms[application] = preconditioner
                order_row["asm"][application] = {
                    **profile.to_dict(),
                    "relative_difference": difference,
                }

            fastest_asm_name = min(
                order_row["asm"],
                key=lambda name: order_row["asm"][name]["total"]["median_ms"],
            )
            base_preconditioner = asms[fastest_asm_name]
            order_row["fastest_asm"] = fastest_asm_name

            degree18 = None
            for degree in args.degrees:
                setup_start = perf_counter()
                polynomial = CuPyPolynomialPreconditioner.from_operator(
                    operator,
                    degree=degree,
                    base_preconditioner=base_preconditioner,
                    seed=1729,
                    setup_orthogonalization="cgs2",
                )
                cp.cuda.get_current_stream().synchronize()
                setup_ms = 1.0e3 * (perf_counter() - setup_start)
                apply_stats = benchmark_cuda_call(
                    lambda: polynomial.apply_into(rhs, output),
                    warmup=args.warmup,
                    repeats=args.repeats,
                    device_id=device_id,
                )

                # Compile and initialize the GMRES path without including it in
                # the measured solve.
                restarted_gmres_cupy(
                    operator,
                    rhs,
                    restart=100,
                    max_iterations=1,
                    rtol=args.rtol,
                    preconditioner=polynomial,
                    orthogonalization="cgs",
                )
                result, solve_ms = synchronized_wall_ms(
                    cp,
                    lambda: restarted_gmres_cupy(
                        operator,
                        rhs,
                        restart=100,
                        max_iterations=args.max_iterations,
                        rtol=args.rtol,
                        preconditioner=polynomial,
                        orthogonalization="cgs",
                    ),
                )
                polynomial_row = {
                    "degree": degree,
                    "effective_degree": int(polynomial.degree),
                    "setup_ms": setup_ms,
                    "apply": apply_stats.to_dict(),
                    "iterations": int(result.iterations),
                    "restart_cycles": int(result.restart_cycles),
                    "solve_wall_ms": solve_ms,
                    "relative_residual": float(result.relative_residual),
                    "status": result.status,
                    "matvecs_per_application": int(polynomial.matvecs_per_application),
                    "asm_calls_per_application": int(polynomial.base_preconditioner_calls_per_application),
                }
                order_row["polynomials"].append(polynomial_row)
                print(
                    f"p={order} d={degree}: apply={apply_stats.median_ms:.3f} ms "
                    f"solve={solve_ms:.1f} ms it={result.iterations} {result.status}",
                    flush=True,
                )
                if degree == 18:
                    degree18 = polynomial

            if degree18 is not None:
                profiler = CuPyGMRESProfiler(device_id=device_id)
                profiled_result = restarted_gmres_cupy(
                    operator,
                    rhs,
                    restart=100,
                    max_iterations=args.max_iterations,
                    rtol=args.rtol,
                    preconditioner=degree18,
                    orthogonalization="cgs",
                    profiler=profiler,
                )
                order_row["degree18_profile"] = {
                    "iterations": int(profiled_result.iterations),
                    "status": profiled_result.status,
                    **profiler.finalize().to_dict(),
                }

            payload["orders"].append(order_row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
            print(
                f"p={order}: operator={fastest_operator_name} ASM={fastest_asm_name}",
                flush=True,
            )

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
