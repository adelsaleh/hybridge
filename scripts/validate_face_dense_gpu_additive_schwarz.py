"""Validate CUDA additive Schwarz against the CPU reference and direct solve."""

from __future__ import annotations

import numpy as np

from hdgfem.backends.cupy import require_cupy_device
from hdgfem.backends.cupy_face_dense import CuPyFaceDenseOperator
from hdgfem.backends.cupy_gmres import restarted_gmres_cupy
from hdgfem.backends.cupy_preconditionners import (
    CuPyFaceAdditiveSchwarzPreconditioner,
)
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.linalg.additive_schwarz import (
    build_face_additive_schwarz_preconditioner,
)
from hdgfem.solvers.diff_rea_face_dense import solve_diffusion_face_dense_direct
from scripts.diff_rea_cases import quadratic_poisson_case


def relative_difference(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(right)), np.finfo(np.float64).eps)
    return float(np.linalg.norm(left - right) / scale)


def run_case(boundary_mode: str) -> None:
    cp = require_cupy_device()
    mesh_size = 6 if boundary_mode == "eliminate" else 4
    space = DGSpace(
        rectangle_mesh(mesh_size, mesh_size),
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

    cpu_asm = build_face_additive_schwarz_preconditioner(
        system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
    )
    operator = CuPyFaceDenseOperator.from_system(system, implementation="matmul")
    gpu_asm = CuPyFaceAdditiveSchwarzPreconditioner.from_system(
        system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
        device_id=operator.device_id,
        dtype=np.dtype(operator.dtype.name),
    )

    rng = np.random.default_rng(20260722)
    vector = rng.standard_normal(system.rhs.shape)
    expected_application = cpu_asm.apply(vector)
    vector_device = operator.to_device(vector)
    computed_application = operator.to_host(gpu_asm.apply(vector_device))

    rhs_device = operator.to_device(system.rhs)
    result = restarted_gmres_cupy(
        operator,
        rhs_device,
        restart=30,
        max_iterations=1000,
        rtol=1.0e-10,
        preconditioner=gpu_asm,
        reorthogonalize=True,
    )
    operator.synchronize()
    solution = operator.to_host(result.solution)

    print(f"Boundary mode                 : {boundary_mode}")
    print(f"CUDA device                   : {cp.cuda.Device(operator.device_id)}")
    print(f"System dofs                   : {system.num_dofs}")
    print(
        "Elements / local size         : "
        f"{gpu_asm.num_elements} / {gpu_asm.local_size}"
    )
    print(
        "GPU ASM application difference: "
        f"{relative_difference(computed_application, expected_application):.3e}"
    )
    print(f"GMRES status                  : {result.status}")
    print(
        "Arnoldi iterations / cycles   : "
        f"{result.iterations} / {result.restart_cycles}"
    )
    print(f"True relative residual        : {result.relative_residual:.3e}")
    print(
        "Difference from direct solve    : "
        f"{relative_difference(solution.reshape(-1), direct.system_solution):.3e}"
    )
    print(f"Preconditioner applications   : {result.preconditioner_count}")
    print()


def main() -> None:
    print("Face-dense CUDA additive-Schwarz validation")
    print("=" * 46)
    for boundary_mode in ("eliminate", "penalty"):
        run_case(boundary_mode)


if __name__ == "__main__":
    main()
