"""Compare MGS/MGS2 with batched CGS/CGS2 on one CUDA HDG system.

The script is intended for synchronization and time-to-solution studies.  It
uses the raw face-dense operator and raw ASM inverse application by default,
then changes only the Arnoldi orthogonalization strategy.
"""

from __future__ import annotations

import argparse
from time import perf_counter

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
    parser.add_argument("--mesh", type=int, default=32)
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--restart", type=int, default=30)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("mgs", "mgs2", "cgs", "cgs2"),
        default=["mgs", "mgs2", "cgs", "cgs2"],
    )
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
    parser.add_argument("--device", type=int, default=None)
    args = parser.parse_args()
    if args.mesh <= 0 or args.order < 0:
        parser.error("mesh must be positive and order non-negative")
    if args.restart <= 0 or args.max_iterations <= 0:
        parser.error("restart and max-iterations must be positive")
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


def relative_difference(left: np.ndarray, right: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(right)), np.finfo(np.float64).eps)
    return float(np.linalg.norm(left - right) / denominator)


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
        results = {}
        times_ms = {}
        for mode in args.modes:
            solve = lambda: restarted_gmres_cupy(
                operator,
                rhs,
                restart=args.restart,
                max_iterations=args.max_iterations,
                rtol=args.rtol,
                preconditioner=preconditioner,
                orthogonalization=mode,
            )
            for _ in range(args.warmup):
                solve()
            operator.synchronize()
            start = perf_counter()
            result = solve()
            operator.synchronize()
            elapsed_ms = 1.0e3 * (perf_counter() - start)
            results[mode] = result
            times_ms[mode] = elapsed_ms

        reference_mode = "cgs2" if "cgs2" in results else args.modes[0]
        reference_solution = operator.to_host(results[reference_mode].solution)

        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        device_name = properties.get("name", properties.get(b"name", "unknown"))
        if isinstance(device_name, bytes):
            device_name = device_name.decode(errors="replace")

        print("GPU GMRES orthogonalization comparison")
        print("=" * 48)
        print(f"Device / dofs       : {device_name} / {system.num_dofs}")
        print(f"Mesh / order        : {args.mesh}x{args.mesh} / p={args.order}")
        print(f"Restart / tolerance : {args.restart} / {args.rtol:.1e}")
        print(f"Preconditioner      : {args.preconditioner}")
        print()
        print(
            "mode   time [ms]  iter  cycles   relres      DOT  AXPY  "
            "proj  corr  D2H  solution-diff"
        )
        print("-" * 91)
        for mode in args.modes:
            result = results[mode]
            solution = operator.to_host(result.solution)
            difference = relative_difference(solution, reference_solution)
            print(
                f"{mode:5s} {times_ms[mode]:10.3f} "
                f"{result.iterations:5d} {result.restart_cycles:7d} "
                f"{result.relative_residual:10.3e} "
                f"{result.dot_count:6d} {result.axpy_count:5d} "
                f"{result.basis_projection_count:5d} "
                f"{result.basis_correction_count:5d} "
                f"{result.coefficient_d2h_count:4d} "
                f"{difference:13.3e}"
            )


if __name__ == "__main__":
    main()
