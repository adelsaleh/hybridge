"""Validate the production-oriented robust GPU GMRES interface.

The script assembles the established CPU-local/GPU-global face-dense system,
constructs the selected GPU preconditioner, and reports restart-cycle
convergence safeguards.  It does not materialize the global dense matrix.
"""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from hdgfem.hdg import condensation as hdg_assembly
from hdgfem.runtime.optional import require_cupy_device
from hdgfem.backends.cupy_face_dense import CuPyFaceDenseOperator
from hdgfem.backends.cupy_polynomial import CuPyPolynomialPreconditioner
from hdgfem.backends.cupy_preconditionners import (
    CuPyFaceAdditiveSchwarzPreconditioner,
    CuPyFaceBlockJacobiPreconditioner,
)
from hdgfem.backends.cupy_solver import (
    CuPyProductionGMRESOptions,
    CuPyProductionGMRESSolver,
)
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers.diffusion_reaction import (
    diffusion_element_boundary_mats,
    local_solvers,
)
from hdgfem.solvers.diffusion_face_dense import (
    assemble_diffusion_face_dense_components,
)
from scripts.diffusion_reaction.cases import quadratic_poisson_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, default=64)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--restart", type=int, default=75)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument(
        "--operator", choices=("raw", "raw_fused", "matmul"), default="raw"
    )
    parser.add_argument(
        "--preconditioner",
        choices=("none", "block_jacobi", "asm", "asm_poly"),
        default="asm_poly",
    )
    parser.add_argument(
        "--asm-application", choices=("raw", "fused"), default="fused"
    )
    parser.add_argument("--polynomial-degree", type=int, default=18)
    parser.add_argument(
        "--orthogonalization",
        choices=("cgs", "cgs2", "mgs", "mgs2"),
        default="cgs",
    )
    parser.add_argument(
        "--fallback-threshold",
        default="auto",
        help="CGS max-offdiagonal threshold, 'auto', or 'none'",
    )
    parser.add_argument("--stagnation-cycles", type=int, default=8)
    parser.add_argument("--stagnation-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--divergence-factor", type=float, default=1.0e6)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--monitor-orthogonality", action="store_true")
    return parser.parse_args()


def fallback_value(text: str):
    value = text.strip().lower()
    if value == "auto":
        return "auto"
    if value in ("none", "off", "disabled"):
        return None
    return float(value)


def assemble_case(mesh_size: int, order: int):
    diffusion, reaction, source, boundary = quadratic_poisson_case()
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
        boundary,
        stabilization,
        space,
        boundary_penalty=1.0e6,
    )
    return space, assembly


def main() -> None:
    args = parse_args()
    cp = require_cupy_device()
    device_id = int(cp.cuda.Device().id) if args.device is None else int(args.device)

    space, assembly = assemble_case(args.mesh, args.order)
    system = assembly.eliminated_system

    with cp.cuda.Device(device_id):
        operator = CuPyFaceDenseOperator.from_system(
            system,
            implementation=args.operator,
            device_id=device_id,
        )
        preconditioner = None
        if args.preconditioner == "block_jacobi":
            preconditioner = CuPyFaceBlockJacobiPreconditioner.from_system(
                system,
                device_id=device_id,
                local_solver="cublas_inverse",
                application="raw",
            )
        elif args.preconditioner in ("asm", "asm_poly"):
            asm = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
                system,
                assembly.element_blocks,
                space.mesh.loc2glob_edge,
                device_id=device_id,
                local_solver="cublas_inverse",
                application=args.asm_application,
            )
            preconditioner = asm
            if args.preconditioner == "asm_poly":
                preconditioner = CuPyPolynomialPreconditioner.from_operator(
                    operator,
                    degree=args.polynomial_degree,
                    base_preconditioner=asm,
                    setup_orthogonalization="cgs2",
                )

        options = CuPyProductionGMRESOptions(
            restart=args.restart,
            max_iterations=args.max_iterations,
            rtol=args.rtol,
            atol=args.atol,
            orthogonalization=args.orthogonalization,
            stagnation_cycles=args.stagnation_cycles,
            stagnation_tolerance=args.stagnation_tolerance,
            divergence_factor=args.divergence_factor,
            cgs2_fallback_threshold=fallback_value(args.fallback_threshold),
        )
        solver = CuPyProductionGMRESSolver(
            operator,
            preconditioner=preconditioner,
            options=options,
        )
        rhs = operator.to_device(system.rhs)

        cp.cuda.get_current_stream().synchronize()
        start = perf_counter()
        result = solver.solve(
            rhs,
            monitor_orthogonality=args.monitor_orthogonality,
        )
        cp.cuda.get_current_stream().synchronize()
        elapsed_ms = 1.0e3 * (perf_counter() - start)

    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")

    print("Production GPU GMRES validation")
    print("=" * 88)
    print(f"Device / dofs       : {device_name} / {system.num_dofs}")
    print(f"Mesh / order        : {args.mesh}x{args.mesh} / p={args.order}")
    print(f"Operator / prec.    : {args.operator} / {args.preconditioner}")
    print(f"Restart / tolerance : {args.restart} / {args.rtol:.1e}")
    print(f"Solve time          : {elapsed_ms:.3f} ms")
    print(
        f"Status              : {result.status} — {result.termination_reason}"
    )
    print(
        f"Iterations / cycles : {result.iterations} / {result.restart_cycles}"
    )
    print(f"True relative res.  : {result.relative_residual:.3e}")
    print(f"CGS→CGS2 fallbacks  : {result.fallback_count}")
    print(
        "Residual replacements: "
        f"{result.true_residual_recomputations} "
        "(initial plus one per completed cycle)"
    )
    print(f"Workspace           : {solver.workspace_device_bytes / 2**20:.3f} MiB")
    print()
    print(
        "cycle  iterations  orth  true-start    true-end      reduction  "
        "stagnation  fallback"
    )
    print("-" * 88)
    for record in result.cycle_records:
        print(
            f"{record.restart_cycle:5d} "
            f"{record.iteration_start:5d}-{record.iteration_end:<5d} "
            f"{record.orthogonalization:5s} "
            f"{record.true_residual_start:12.4e} "
            f"{record.true_residual_end:12.4e} "
            f"{record.residual_reduction:10.3e} "
            f"{record.stagnation_count:10d} "
            f"{str(record.switched_to_cgs2):>8s}"
        )

    if not result.converged:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
