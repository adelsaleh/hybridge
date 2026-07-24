"""Compare CPU inverse transfer, GPU inversion, and batched GPU solves."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from hdgfem.backends.cupy import require_cupy_device
from hdgfem.backends.cupy_preconditionners import (
    CuPyFaceAdditiveSchwarzPreconditioner,
    CuPyFaceBlockJacobiPreconditioner,
)
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.linalg.additive_schwarz import (
    build_face_additive_schwarz_preconditioner,
)
from hdgfem.linalg.block_jacobi import build_face_block_jacobi_preconditioner
from hdgfem.solvers.diff_rea_face_dense import solve_diffusion_face_dense_direct
from scripts.diff_rea_cases import quadratic_poisson_case


def relative_difference(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(right)), np.finfo(np.float64).eps)
    return float(np.linalg.norm(left - right) / scale)


def time_gpu_call(cp, function, *, warmup: int = 5, repeats: int = 30) -> float:
    for _ in range(warmup):
        function()
    cp.cuda.get_current_stream().synchronize()

    start = cp.cuda.Event()
    stop = cp.cuda.Event()
    start.record()
    for _ in range(repeats):
        function()
    stop.record()
    stop.synchronize()
    return float(cp.cuda.get_elapsed_time(start, stop) / repeats)


def timed_setup(cp, builder):
    cp.cuda.get_current_stream().synchronize()
    start = perf_counter()
    result = builder()
    cp.cuda.get_current_stream().synchronize()
    return result, 1.0e3 * (perf_counter() - start)


def run_case(boundary_mode: str) -> None:
    cp = require_cupy_device()
    space = DGSpace(
        rectangle_mesh(8, 8),
        2,
        basis_type="dub_orth",
    )
    diffusion, reaction, source, exact = quadratic_poisson_case()
    direct = solve_diffusion_face_dense_direct(
        source,
        reaction,
        exact,
        space,
        diffusion=diffusion,
        stabilization=1.3,
        boundary_mode=boundary_mode,
        boundary_penalty=1.0e6,
    )
    system = direct.system

    cpu_bj = build_face_block_jacobi_preconditioner(system)
    cpu_asm = build_face_additive_schwarz_preconditioner(
        system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
    )
    rng = np.random.default_rng(20260723)
    vector = rng.standard_normal(system.rhs.shape)
    vector_gpu = cp.asarray(vector)

    print(f"Boundary mode: {boundary_mode}")
    print(f"System dofs  : {system.num_dofs}")
    print()

    print("Block-Jacobi")
    print("-------------")
    expected_bj = cpu_bj.apply(vector)
    for mode in ("cpu_inverse", "gpu_inverse", "gpu_solve"):
        preconditioner, setup_ms = timed_setup(
            cp,
            lambda mode=mode: CuPyFaceBlockJacobiPreconditioner.from_system(
                system,
                local_solver=mode,
            ),
        )
        output = cp.empty_like(vector_gpu)
        apply_ms = time_gpu_call(
            cp,
            lambda: preconditioner.apply_into(vector_gpu, output),
        )
        error = relative_difference(cp.asnumpy(output), expected_bj)
        residual = preconditioner.maximum_inverse_residual
        residual_text = "n/a" if residual is None else f"{residual:.3e}"
        print(
            f"{mode:12s} setup={setup_ms:9.3f} ms  "
            f"apply={apply_ms:9.4f} ms  error={error:.3e}  "
            f"inv_res={residual_text:>9s}  "
            f"allocates={preconditioner.allocates_during_apply}"
        )
    print()

    print("Additive Schwarz")
    print("-----------------")
    expected_asm = cpu_asm.apply(vector)
    for mode in ("cpu_inverse", "gpu_inverse", "gpu_solve"):
        preconditioner, setup_ms = timed_setup(
            cp,
            lambda mode=mode: CuPyFaceAdditiveSchwarzPreconditioner.from_system(
                system,
                direct.assembly.element_blocks,
                space.mesh.loc2glob_edge,
                local_solver=mode,
            ),
        )
        output = cp.empty_like(vector_gpu)
        apply_ms = time_gpu_call(
            cp,
            lambda: preconditioner.apply_into(vector_gpu, output),
        )
        error = relative_difference(cp.asnumpy(output), expected_asm)
        residual = preconditioner.maximum_inverse_residual
        residual_text = "n/a" if residual is None else f"{residual:.3e}"
        print(
            f"{mode:12s} setup={setup_ms:9.3f} ms  "
            f"apply={apply_ms:9.4f} ms  error={error:.3e}  "
            f"inv_res={residual_text:>9s}  "
            f"allocates={preconditioner.allocates_during_apply}"
        )
    print()


def main() -> None:
    cp = require_cupy_device()
    print("GPU local factorization and solve comparison")
    print("=" * 45)
    print(f"CUDA device: {cp.cuda.Device()}")
    print()
    for mode in ("eliminate", "penalty"):
        run_case(mode)


if __name__ == "__main__":
    main()
