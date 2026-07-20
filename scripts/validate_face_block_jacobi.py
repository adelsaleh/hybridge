"""Compare unpreconditioned and block-Jacobi face-dense GMRES on CPU."""

from __future__ import annotations

import numpy as np

from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.linalg import (
    build_face_block_jacobi_preconditioner,
    solve_face_dense_gmres,
)
from hdgfem.solvers.diff_rea_face_dense import solve_diffusion_face_dense_direct
from scripts.diff_rea_cases import quadratic_poisson_case


def run_case(boundary_mode: str) -> None:
    mesh_size = 5 if boundary_mode == "eliminate" else 3
    order = 2
    space = DGSpace(
        rectangle_mesh(mesh_size, mesh_size),
        order,
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

    plain = solve_face_dense_gmres(
        direct.system,
        restart=30,
        max_iterations=800,
        rtol=1.0e-10,
        reorthogonalize=True,
    )
    block_jacobi = build_face_block_jacobi_preconditioner(direct.system)
    preconditioned = solve_face_dense_gmres(
        direct.system,
        restart=30,
        max_iterations=800,
        rtol=1.0e-10,
        preconditioner=block_jacobi,
        reorthogonalize=True,
    )

    reference_scale = max(float(np.linalg.norm(direct.system_solution)), 1.0)
    preconditioned_difference = float(
        np.linalg.norm(
            preconditioned.solution.reshape(-1) - direct.system_solution
        )
        / reference_scale
    )

    print(f"Boundary mode                 : {boundary_mode}")
    print(f"System dofs                   : {direct.system.num_dofs}")
    print(f"Maximum inverse residual      : {block_jacobi.maximum_inverse_residual:.3e}")
    print(f"Plain GMRES status            : {plain.status}")
    print(f"Plain Arnoldi iterations      : {plain.iterations}")
    print(f"Plain restart cycles          : {plain.restart_cycles}")
    print(f"Plain true relative residual  : {plain.relative_residual:.3e}")
    print(f"BJ-GMRES status               : {preconditioned.status}")
    print(f"BJ Arnoldi iterations         : {preconditioned.iterations}")
    print(f"BJ restart cycles             : {preconditioned.restart_cycles}")
    print(f"BJ true relative residual     : {preconditioned.relative_residual:.3e}")
    print(f"BJ difference from direct     : {preconditioned_difference:.3e}")
    print(f"BJ preconditioner calls       : {preconditioned.preconditioner_count}")
    print()


def main() -> None:
    print("Face-dense CPU block-Jacobi validation")
    print("=" * 39)
    run_case("eliminate")
    run_case("penalty")


if __name__ == "__main__":
    main()