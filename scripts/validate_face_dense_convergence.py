"""Independent manufactured-solution validation for face-dense assembly.

Run from the repository root with::

    PYTHONPATH=. python scripts/validate_face_dense_convergence.py

The script never calls the COO trace assembler.  It materializes the small
face-dense matrix only to obtain reference direct solutions for convergence
checks.
"""

from __future__ import annotations

import numpy as np

from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.solvers.diff_rea_face_dense import solve_diffusion_face_dense_direct
from scripts.diff_rea_cases import (
    exponential_bubble_poisson_case,
    quadratic_poisson_case,
    tensor_sine_diffusion_reaction_case,
)


def solve_case(problem_factory, mesh_size: int, order: int, stabilization: float):
    space = DGSpace(
        rectangle_mesh(mesh_size, mesh_size),
        order,
        basis_type="dub_orth",
    )
    diffusion, reaction, source, exact = problem_factory()
    result = solve_diffusion_face_dense_direct(
        source,
        reaction,
        exact,
        space,
        diffusion=diffusion,
        stabilization=stabilization,
        boundary_mode="eliminate",
    )
    return result.field.l2_error(exact), result.relative_residual, result.system.num_dofs


def report_convergence(name: str, problem_factory, stabilization: float) -> None:
    print(f"\n{name}")
    print("  p    n      trace dofs          L2 error       rate      residual")
    print("  -    -      ----------          --------       ----      --------")
    for order in range(4):
        previous_error = None
        for mesh_size in (2, 4, 8):
            error, residual, num_dofs = solve_case(
                problem_factory,
                mesh_size,
                order,
                stabilization,
            )
            rate = "   -" if previous_error is None else f"{np.log2(previous_error / error):6.3f}"
            print(
                f" {order:2d}  {mesh_size:3d}  {num_dofs:14d}  "
                f"{error:16.8e}  {rate}  {residual:10.3e}"
            )
            previous_error = error


def report_quadratic_reproduction() -> None:
    print("Quadratic polynomial reproduction (p=2, 2x2 mesh)")
    diffusion, reaction, source, exact = quadratic_poisson_case()
    for mode in ("penalty", "eliminate"):
        space = DGSpace(rectangle_mesh(2, 2), 2, basis_type="dub_orth")
        result = solve_diffusion_face_dense_direct(
            source,
            reaction,
            exact,
            space,
            diffusion=diffusion,
            stabilization=1.3,
            boundary_mode=mode,
            boundary_penalty=1.0e8,
        )
        print(
            f"  {mode:9s}: L2 error={result.field.l2_error(exact):.3e}, "
            f"relative residual={result.relative_residual:.3e}"
        )


def main() -> None:
    report_quadratic_reproduction()
    report_convergence(
        "Smooth exponential-bubble Poisson problem",
        exponential_bubble_poisson_case,
        1.0,
    )
    report_convergence(
        "Variable tensor diffusion-reaction problem",
        tensor_sine_diffusion_reaction_case,
        4.0,
    )


if __name__ == "__main__":
    main()