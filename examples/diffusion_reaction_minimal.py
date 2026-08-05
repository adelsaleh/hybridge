"""Minimal end-to-end HDG diffusion-reaction solve on the host."""

from __future__ import annotations

from hdgfem import DGSpace, rectangle_mesh, solve_diffusion_reaction_hdg


def main() -> None:
    """Build, solve, and verify a manufactured diffusion-reaction problem."""
    mesh = rectangle_mesh(6, 6, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    space = DGSpace(mesh, 2, basis_type="dub_orth")

    exact = lambda x, y: 1.0 + x**2 + y**2
    reaction = lambda x, y: 0.0 * x
    source = lambda x, y: -4.0 + 0.0 * x

    result = solve_diffusion_reaction_hdg(
        source,
        reaction,
        exact,
        space,
        diffusion=1.0,
        stabilization=1.0,
        solver="direct",
        preconditioner=None,
        boundary_mode="eliminate",
        verbose=False,
    )

    error = result.field.l2_error(exact)
    linear_solve = result.global_solve_result
    if linear_solve is None or not linear_solve.converged:
        raise RuntimeError("the global trace solve did not converge")
    if error > 1.0e-10:
        raise RuntimeError(f"unexpected L2 error: {error:.3e}")

    print(f"elements: {mesh.num_tri}")
    print(f"trace dofs: {result.trace.size}")
    print(f"L2 error: {error:.3e}")
    print(
        "physical relative residual: "
        f"{linear_solve.physical_relative_residual_norm:.3e}"
    )


if __name__ == "__main__":
    main()
