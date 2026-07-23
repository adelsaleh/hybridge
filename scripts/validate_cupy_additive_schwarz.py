"""Compare plain, block-Jacobi, and additive-Schwarz face GMRES on CPU."""

from __future__ import annotations

import numpy as np

from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.linalg import (
    build_face_additive_schwarz_preconditioner,
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
    bj_result = solve_face_dense_gmres(
        direct.system,
        restart=30,
        max_iterations=800,
        rtol=1.0e-10,
        preconditioner=block_jacobi,
        reorthogonalize=True,
    )
    additive_schwarz = build_face_additive_schwarz_preconditioner(
        direct.system,
        direct.assembly.element_blocks,
        space.mesh.loc2glob_edge,
    )
    asm_result = solve_face_dense_gmres(
        direct.system,
        restart=30,
        max_iterations=800,
        rtol=1.0e-10,
        preconditioner=additive_schwarz,
        reorthogonalize=True,
    )

    reference_scale = max(float(np.linalg.norm(direct.system_solution)), 1.0)
    asm_difference = float(
        np.linalg.norm(asm_result.solution.reshape(-1) - direct.system_solution)
        / reference_scale
    )

    print(f"Boundary mode                 : {boundary_mode}")
    print(f"System dofs                   : {direct.system.num_dofs}")
    print(f"ASM local matrix size         : {additive_schwarz.local_size}")
    print(f"ASM maximum inverse residual  : {additive_schwarz.maximum_inverse_residual:.3e}")
    print(f"Plain iterations / cycles     : {plain.iterations} / {plain.restart_cycles}")
    print(f"Plain relative residual       : {plain.relative_residual:.3e}")
    print(f"BJ iterations / cycles        : {bj_result.iterations} / {bj_result.restart_cycles}")
    print(f"BJ relative residual          : {bj_result.relative_residual:.3e}")
    print(f"ASM iterations / cycles       : {asm_result.iterations} / {asm_result.restart_cycles}")
    print(f"ASM relative residual         : {asm_result.relative_residual:.3e}")
    print(f"ASM difference from direct    : {asm_difference:.3e}")
    print(f"ASM preconditioner calls      : {asm_result.preconditioner_count}")
    print()


def main() -> None:
    print("Face-dense CPU additive-Schwarz validation")
    print("=" * 43)
    run_case("eliminate")
    run_case("penalty")


if __name__ == "__main__":
    main()
