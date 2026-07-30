"""Compare three-stage and two-kernel fused GPU additive Schwarz paths."""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from hdgfem.backends.cupy import require_cupy_device
from hdgfem.backends.cupy_face_dense import CuPyFaceDenseOperator
from hdgfem.backends.cupy_gmres import restarted_gmres_cupy
from hdgfem.backends.cupy_polynomial import (
    CuPyPolynomialPreconditioner,
    initialize_polynomial_kernels_cupy,
)
from hdgfem.backends.cupy_preconditionners import CuPyFaceAdditiveSchwarzPreconditioner
from hdgfem.backends.cupy_profiling import profile_additive_schwarz
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.linalg.additive_schwarz import build_face_additive_schwarz_preconditioner
from hdgfem.solvers.diff_rea_face_dense import solve_diffusion_face_dense_direct
from scripts.diff_rea_cases import quadratic_poisson_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=int, default=64)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument(
        "--boundary-mode", choices=("eliminate", "penalty"), default="eliminate"
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--operator", choices=("raw", "raw_fused"), default="raw"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--restart", type=int, default=75)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--polynomial-degree", type=int, default=18)
    parser.add_argument("--skip-polynomial", action="store_true")
    parser.add_argument("--device", type=int, default=None)
    return parser.parse_args()


def synchronized_solve(cp, operator, rhs, preconditioner, args):
    cp.cuda.get_current_stream().synchronize()
    start = perf_counter()
    result = restarted_gmres_cupy(
        operator,
        rhs,
        restart=args.restart,
        max_iterations=args.max_iterations,
        rtol=args.rtol,
        preconditioner=preconditioner,
        orthogonalization="cgs",
    )
    cp.cuda.get_current_stream().synchronize()
    return result, 1.0e3 * (perf_counter() - start)


def main() -> None:
    args = parse_args()
    cp = require_cupy_device()
    dtype = np.dtype(args.dtype)
    device_id = int(cp.cuda.Device().id) if args.device is None else int(args.device)

    diffusion, reaction, source, boundary = quadratic_poisson_case()
    space = DGSpace(
        rectangle_mesh(args.mesh, args.mesh),
        args.order,
        basis_type="dub_orth",
    )
    direct = solve_diffusion_face_dense_direct(
        source,
        reaction,
        boundary,
        space,
        diffusion=diffusion,
        stabilization=1.3,
        boundary_mode=args.boundary_mode,
        boundary_penalty=1.0e6,
    )
    system = direct.system
    cpu = build_face_additive_schwarz_preconditioner(
        system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
    )

    with cp.cuda.Device(device_id):
        operator = CuPyFaceDenseOperator.from_system(
            system,
            implementation=args.operator,
            dtype=dtype,
            device_id=device_id,
        )
        raw = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
            system,
            direct.assembly.element_blocks,
            space.mesh.loc2glob_edge,
            dtype=dtype,
            device_id=device_id,
            local_solver="cublas_inverse",
            application="raw",
        )
        fused = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
            system,
            direct.assembly.element_blocks,
            space.mesh.loc2glob_edge,
            dtype=dtype,
            device_id=device_id,
            local_solver="cublas_inverse",
            application="fused",
        )

        rng = np.random.default_rng(20260730)
        x_host = rng.standard_normal(system.rhs.shape).astype(dtype)
        x = cp.asarray(x_host)
        raw_out = cp.empty_like(x)
        fused_out = cp.empty_like(x)
        raw_profile = profile_additive_schwarz(
            raw, x, raw_out, warmup=args.warmup, repeats=args.repeats
        )
        fused_profile = profile_additive_schwarz(
            fused, x, fused_out, warmup=args.warmup, repeats=args.repeats
        )
        raw.apply_into(x, raw_out)
        fused.apply_into(x, fused_out)
        cp.cuda.get_current_stream().synchronize()
        raw_host = cp.asnumpy(raw_out)
        fused_host = cp.asnumpy(fused_out)

        rhs = operator.to_device(system.rhs.astype(dtype, copy=False))
        raw_result, raw_solve_ms = synchronized_solve(cp, operator, rhs, raw, args)
        fused_result, fused_solve_ms = synchronized_solve(cp, operator, rhs, fused, args)

        polynomial_rows = []
        if not args.skip_polynomial:
            initialize_polynomial_kernels_cupy(dtype=operator.dtype, device_id=device_id)
            for name, base in (("raw", raw), ("fused", fused)):
                cp.cuda.get_current_stream().synchronize()
                setup_start = perf_counter()
                polynomial = CuPyPolynomialPreconditioner.from_operator(
                    operator,
                    degree=args.polynomial_degree,
                    base_preconditioner=base,
                    setup_orthogonalization="cgs2",
                )
                cp.cuda.get_current_stream().synchronize()
                setup_ms = 1.0e3 * (perf_counter() - setup_start)
                result, solve_ms = synchronized_solve(
                    cp, operator, rhs, polynomial, args
                )
                polynomial_rows.append((name, setup_ms, solve_ms, result))

    cpu_reference = cpu.apply(x_host)
    denominator = max(float(np.linalg.norm(cpu_reference)), np.finfo(dtype).eps)
    fused_cpu_error = float(np.linalg.norm(fused_host - cpu_reference) / denominator)
    raw_fused_error = float(
        np.linalg.norm(fused_host - raw_host)
        / max(float(np.linalg.norm(raw_host)), np.finfo(dtype).eps)
    )

    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()

    print("GPU fused additive-Schwarz validation")
    print("=" * 82)
    print(f"Device / dofs       : {device_name} / {system.num_dofs}")
    print(f"Mesh / order        : {args.mesh}x{args.mesh} / p={args.order}")
    print(f"Boundary / dtype    : {args.boundary_mode} / {dtype.name}")
    print(f"Operator            : {args.operator}")
    print()
    print("application  total[ms] restrict[ms] local[ms] prolong[ms] workspace[MiB]")
    print("-" * 82)
    for name, preconditioner, profile in (
        ("raw", raw, raw_profile),
        ("fused", fused, fused_profile),
    ):
        print(
            f"{name:11s} {profile.total.median_ms:9.4f} "
            f"{profile.restriction.median_ms:12.4f} "
            f"{profile.local_solve.median_ms:9.4f} "
            f"{profile.prolongation.median_ms:11.4f} "
            f"{preconditioner.workspace_bytes / 2**20:14.3f}"
        )
    print()
    print(f"standalone speedup  : {raw_profile.total.median_ms / fused_profile.total.median_ms:.3f} x")
    print(f"workspace reduction : {(raw.workspace_bytes - fused.workspace_bytes) / 2**20:.3f} MiB")
    print(f"fused-vs-CPU error  : {fused_cpu_error:.3e}")
    print(f"fused-vs-raw error  : {raw_fused_error:.3e}")
    print()
    print("GMRES ASM")
    print("-" * 82)
    print(
        f"raw   : {raw_solve_ms:9.3f} ms  iter={raw_result.iterations:5d} "
        f"relres={raw_result.relative_residual:.3e} {raw_result.status}"
    )
    print(
        f"fused : {fused_solve_ms:9.3f} ms  iter={fused_result.iterations:5d} "
        f"relres={fused_result.relative_residual:.3e} {fused_result.status}"
    )
    if polynomial_rows:
        print()
        print(f"GMRES ASM-polynomial, degree {args.polynomial_degree}")
        print("-" * 82)
        for name, setup_ms, solve_ms, result in polynomial_rows:
            print(
                f"{name:6s}: setup={setup_ms:8.3f} ms solve={solve_ms:9.3f} ms "
                f"iter={result.iterations:4d} relres={result.relative_residual:.3e} "
                f"{result.status}"
            )


if __name__ == "__main__":
    main()
