"""Tune face-dense matvec and additive-Schwarz kernels on one CUDA device."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hdgfem.assembly import hdg as hdg_assembly
from hdgfem.backends.cupy_autotune import autotune_face_dense_gpu_cached
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers.diff_rea import diffusion_element_boundary_mats, local_solvers
from hdgfem.solvers.diff_rea_face_dense import assemble_diffusion_face_dense_components
from scripts.diff_rea_cases import quadratic_poisson_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, default=64)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--boundary-mode", choices=("eliminate", "penalty"), default="eliminate")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cache-file", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--force-retune", action="store_true")
    args = parser.parse_args()

    space = DGSpace(rectangle_mesh(args.mesh, args.mesh), args.order, basis_type="dub_orth")
    diffusion, reaction, source, boundary = quadratic_poisson_case()
    tau = 1.3
    local_solver = local_solvers(reaction, tau, space, diffusion=diffusion)
    boundary_mats = diffusion_element_boundary_mats(tau, space)
    source_rhs = hdg_assembly.block_source_moments(source, space, num_blocks=3, source_block=0)
    assembly = assemble_diffusion_face_dense_components(
        local_solver,
        boundary_mats,
        source_rhs,
        boundary,
        tau,
        space,
        boundary_penalty=1.0e6,
    )
    system = assembly.eliminated_system if args.boundary_mode == "eliminate" else assembly.penalty_system
    dtype = np.float32 if args.dtype == "float32" else np.float64

    cached = autotune_face_dense_gpu_cached(
        system,
        element_blocks=assembly.element_blocks,
        loc2glob_face=space.mesh.loc2glob_edge,
        dtype=dtype,
        device_id=args.device,
        polynomial_order=args.order,
        warmup=args.warmup,
        repeats=args.repeats,
        cache_path=args.cache_file,
        use_cache=not args.no_cache,
        force_retune=args.force_retune,
    )
    result = cached.result

    print("Face-dense CUDA autotuning")
    print("=" * 68)
    print(f"Device / dofs : {result.device_name} / {result.num_dofs}")
    print(f"Mesh / order  : {args.mesh}x{args.mesh} / p={args.order}")
    print(f"dtype         : {result.dtype}")
    print(f"cache         : {'hit' if cached.cache_hit else 'miss'}")
    if cached.cache_path is not None:
        print(f"cache file    : {cached.cache_path}")
    print(f"cache key     : {cached.key.cache_id[:16]}")
    print()
    print("operator candidate   median[ms] minimum[ms] workspace[MiB] relerr")
    for row in result.operator_candidates:
        print(
            f"{row.name:18s} {row.median_ms:10.4f} {row.minimum_ms:11.4f} "
            f"{row.workspace_bytes / 2**20:14.3f} {row.relative_error:.3e}"
        )
    print(f"selected operator: {result.operator_choice}")
    print()
    print("ASM candidate        median[ms] minimum[ms] workspace[MiB] relerr")
    for row in result.asm_candidates:
        print(
            f"{row.name:18s} {row.median_ms:10.4f} {row.minimum_ms:11.4f} "
            f"{row.workspace_bytes / 2**20:14.3f} {row.relative_error:.3e}"
        )
    print(f"selected ASM path: {result.asm_choice}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_hit": cached.cache_hit,
            "cache_path": None if cached.cache_path is None else str(cached.cache_path),
            "cache_key": cached.key.to_dict(),
            "result": result.to_dict(),
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nJSON report: {args.output}")


if __name__ == "__main__":
    main()