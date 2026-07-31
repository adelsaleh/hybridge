"""Validate the ordinary diffusion-reaction API with the GPU face-dense solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers.diff_rea import solve_diffusion_reaction_hdg
from hdgfem.solvers.diff_rea_gpu import DiffusionReactionGPUOptions
from scripts.diff_rea_cases import quadratic_poisson_case


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, default=64)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--boundary-mode", choices=("eliminate", "penalty"), default="eliminate")
    parser.add_argument("--operator", choices=("auto", "raw", "raw_fused", "matmul"), default="auto")
    parser.add_argument("--preconditioner", choices=("none", "block_jacobi", "asm", "asm_poly"), default="asm_poly")
    parser.add_argument("--asm-application", choices=("auto", "matmul", "raw", "fused"), default="auto")
    parser.add_argument("--polynomial-degree", type=int, default=18)
    parser.add_argument("--restart", type=int, default=75)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--cache-file")
    parser.add_argument("--force-retune", action="store_true")
    parser.add_argument("--compare-direct", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.compare_direct and args.mesh > 8:
        parser.error("--compare-direct is restricted to mesh <= 8")

    space = DGSpace(
        rectangle_mesh(args.mesh, args.mesh),
        args.order,
        basis_type="dub_orth",
    )
    diffusion, reaction, source, exact = quadratic_poisson_case()
    gpu_options = DiffusionReactionGPUOptions(
        operator=args.operator,
        preconditioner=args.preconditioner,
        asm_application=args.asm_application,
        polynomial_degree=args.polynomial_degree,
        restart=args.restart,
        autotune=True,
        autotune_cache_file=args.cache_file,
        autotune_force=args.force_retune,
    )

    start = time.perf_counter()
    result = solve_diffusion_reaction_hdg(
        source,
        reaction,
        exact,
        space,
        diffusion=diffusion,
        stabilization=1.0,
        solver="gpu_face_dense",
        solver_rtol=args.rtol,
        maxiter=args.max_iterations,
        boundary_mode=args.boundary_mode,
        gpu_options=gpu_options,
        verbose=False,
    )
    wall_seconds = time.perf_counter() - start
    diagnostic = result.gpu_diagnostics
    gmres = diagnostic.gmres_result

    reference_trace_error = None
    reference_field_error = None
    if args.compare_direct:
        direct = solve_diffusion_reaction_hdg(
            source,
            reaction,
            exact,
            space,
            diffusion=diffusion,
            stabilization=1.0,
            solver="direct",
            preconditioner=None,
            boundary_mode=args.boundary_mode,
            verbose=False,
        )
        reference_trace_error = float(
            np.linalg.norm(result.trace - direct.trace)
            / max(np.linalg.norm(direct.trace), np.finfo(np.float64).tiny)
        )
        reference_field_error = float(
            np.linalg.norm(result.field.coeffs - direct.field.coeffs)
            / max(np.linalg.norm(direct.field.coeffs), np.finfo(np.float64).tiny)
        )

    payload = {
        "device": diagnostic.device_name,
        "mesh": args.mesh,
        "order": args.order,
        "dofs": int(gmres.solution.size),
        "boundary_mode": args.boundary_mode,
        "operator": diagnostic.operator,
        "preconditioner": diagnostic.preconditioner,
        "asm_application": diagnostic.asm_application,
        "polynomial_degree": diagnostic.polynomial_degree,
        "autotune_cache_hit": diagnostic.autotune_cache_hit,
        "autotune_cache_key": diagnostic.autotune_cache_key,
        "iterations": gmres.iterations,
        "restart_cycles": gmres.restart_cycles,
        "relative_residual": gmres.relative_residual,
        "status": gmres.status,
        "termination_reason": gmres.termination_reason,
        "fallback_count": gmres.fallback_count,
        "solver_seconds": diagnostic.solve_seconds,
        "total_wall_seconds": wall_seconds,
        "face_assembly_seconds": diagnostic.face_assembly_seconds,
        "autotune_seconds": diagnostic.autotune_seconds,
        "operator_setup_seconds": diagnostic.operator_setup_seconds,
        "preconditioner_setup_seconds": diagnostic.preconditioner_setup_seconds,
        "workspace_mib": diagnostic.workspace_device_bytes / 2**20,
        "trace_error_vs_direct": reference_trace_error,
        "field_error_vs_direct": reference_field_error,
    }

    print("Integrated diffusion-reaction GPU API validation")
    print("=" * 72)
    print(f"Device / dofs       : {payload['device']} / {payload['dofs']}")
    print(f"Mesh / order        : {args.mesh}x{args.mesh} / p={args.order}")
    print(
        "Operator / prec.    : "
        f"{payload['operator']} / {payload['preconditioner']}"
        + ("" if payload["asm_application"] is None else f" ({payload['asm_application']})")
    )
    print(f"Cache hit           : {payload['autotune_cache_hit']}")
    print(f"Iterations / cycles : {payload['iterations']} / {payload['restart_cycles']}")
    print(f"True relative res.  : {payload['relative_residual']:.3e}")
    print(f"Status              : {payload['status']} — {payload['termination_reason']}")
    print(f"Solve / total       : {payload['solver_seconds']*1e3:.3f} / {wall_seconds*1e3:.3f} ms")
    print(f"Workspace           : {payload['workspace_mib']:.3f} MiB")
    if reference_trace_error is not None:
        print(f"Trace vs direct     : {reference_trace_error:.3e}")
        print(f"Field vs direct     : {reference_field_error:.3e}")

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"JSON report         : {path}")
    return 0 if gmres.converged else 1


if __name__ == "__main__":
    raise SystemExit(main())
