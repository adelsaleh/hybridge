"""Run a compact restarted-GMRES validation report for face-dense systems."""

from __future__ import annotations

import numpy as np

from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.linalg.gmres import solve_face_dense_gmres
from hdgfem.solvers.diff_rea_face_dense import solve_diffusion_face_dense_direct
from scripts.diff_rea_cases import quadratic_poisson_case


def run_case(boundary_mode: str) -> None:
    mesh_size = 3 if boundary_mode == "eliminate" else 2
    order = 2 if boundary_mode == "eliminate" else 1
    penalty = 1.0e4

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
        boundary_penalty=penalty,
    )
    gmres = solve_face_dense_gmres(
        direct.system,
        restart=20,
        max_iterations=250,
        rtol=1.0e-11,
        reorthogonalize=True,
    )

    reference_norm = max(float(np.linalg.norm(direct.system_solution)), 1.0)
    solution_difference = float(
        np.linalg.norm(gmres.solution.reshape(-1) - direct.system_solution)
        / reference_norm
    )

    print(f"Boundary mode             : {boundary_mode}")
    print(f"System dofs               : {direct.system.num_dofs}")
    print(f"GMRES status              : {gmres.status}")
    print(f"Arnoldi iterations        : {gmres.iterations}")
    print(f"Restart cycles            : {gmres.restart_cycles}")
    print(f"True relative residual    : {gmres.relative_residual:.3e}")
    print(f"Difference from direct    : {solution_difference:.3e}")
    print(f"Matrix-vector calls       : {gmres.matvec_count}")
    print()


def main() -> None:
    print("Face-dense restarted-GMRES validation")
    print("=" * 41)
    run_case("eliminate")
    run_case("penalty")


if __name__ == "__main__":
    main()