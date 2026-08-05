"""Minimal end-to-end HDG advection-reaction solve on the host."""

from __future__ import annotations

from hdgfem import DGSpace, rectangle_mesh, solve_advection_reaction_hdg


def main() -> None:
    """Build, solve, and verify a manufactured advection-reaction problem."""
    mesh = rectangle_mesh(6, 6, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
    space = DGSpace(mesh, 2, basis_type="dub_orth")

    exact = lambda x, y: 1.0 + x + y
    beta_x = lambda x, y: 1.0 + 0.0 * x
    beta_y = lambda x, y: 0.5 + 0.0 * y
    reaction = lambda x, y: 2.0 + 0.0 * x
    source = lambda x, y: 1.5 + 2.0 * exact(x, y)

    result = solve_advection_reaction_hdg(
        source,
        (beta_x, beta_y),
        reaction,
        exact,
        space,
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
