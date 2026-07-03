from __future__ import annotations

import numpy as np
import pytest

from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGField, DGSpace
from hdgfem.solvers.diff_rea import (
    DiffusionReactionHDGSolver as DiffReaSolver,
    solve_diffusion_reaction_hdg,
)
from scripts.diff_rea_cases import (
    quadratic_poisson_case,
    tensor_sine_diffusion_reaction_case,
    tensor_sine_exact_gradients,
)


def _space(order: int = 2) -> DGSpace:
    return DGSpace(rectangle_mesh(1, 1), order, basis_type="dub_orth")


def _callable_problem():
    _, reaction, source, exact = quadratic_poisson_case()
    return source, reaction, exact


def _projected_problem(space: DGSpace):
    _, reaction, source, exact = quadratic_poisson_case()
    return (
        DGField(source, space, name="source_h"),
        DGField(reaction, space, name="reaction_h"),
        exact,
    )


def _assert_solver_cache_matches_result(solver: DiffReaSolver, result) -> None:
    assert solver.result is result
    assert solver.field is result.field
    assert solver.flux is result.flux
    assert solver.trace is result.trace
    assert solver.timings is result.timings
    assert solver.local_unknowns is result.local_unknowns
    assert solver.rows is result.matrix_rows
    assert solver.cols is result.matrix_cols
    assert solver.data is result.matrix_data
    assert solver.rhs is result.rhs
    assert solver.solve_rows is result.solve_matrix_rows
    assert solver.solve_cols is result.solve_matrix_cols
    assert solver.solve_data is result.solve_matrix_data
    assert solver.solve_rhs is result.solve_rhs
    assert solver.boundary_trace is result.boundary_trace
    assert solver.reduction is result.reduction
    assert solver.local_solver is result.local_solver
    assert solver.element_boundary_mats is result.element_boundary_mats
    assert solver.global_solve_result is result.global_solve_result


@pytest.mark.parametrize("boundary_mode", ("penalty", "eliminate"))
def test_diff_rea_solver_matches_function_for_callable_problem(boundary_mode: str) -> None:
    space = _space()
    source, reaction, exact = _callable_problem()
    kwargs = {
        "stabilization": 1.0,
        "solver": "direct",
        "preconditioner": None,
        "boundary_mode": boundary_mode,
        "verbose": False,
    }

    function_result = solve_diffusion_reaction_hdg(source, reaction, exact, space, **kwargs)
    solver = DiffReaSolver(space, source=source, reaction=reaction, boundary_condition=exact, **kwargs)
    class_result = solver.solve()

    np.testing.assert_allclose(class_result.trace, function_result.trace, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(class_result.field.coeffs, function_result.field.coeffs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        class_result.flux.as_component_first(),
        function_result.flux.as_component_first(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(class_result.local_unknowns, function_result.local_unknowns, rtol=1e-12, atol=1e-12)
    assert class_result.boundary_mode == boundary_mode
    _assert_solver_cache_matches_result(solver, class_result)


def test_diff_rea_solver_discrete_problem_and_source_update_match_function() -> None:
    space = _space(order=1)
    source_h, reaction_h, exact = _projected_problem(space)
    kwargs = {
        "stabilization": 1.0,
        "solver": "direct",
        "preconditioner": None,
        "boundary_mode": "eliminate",
        "verbose": False,
    }

    function_result = solve_diffusion_reaction_hdg(source_h, reaction_h, exact, space, **kwargs)
    solver = DiffReaSolver(space, **kwargs)
    solver.set_discrete_problem(source_h, reaction_h, exact)
    class_result = solver.solve()

    np.testing.assert_allclose(class_result.trace, function_result.trace, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(class_result.field.coeffs, function_result.field.coeffs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(class_result.local_unknowns, function_result.local_unknowns, rtol=1e-12, atol=1e-12)
    assert class_result.reduction is not None
    assert class_result.local_solver is not None
    assert class_result.element_boundary_mats is not None
    _assert_solver_cache_matches_result(solver, class_result)

    scaled_source = space.field(1.1 * source_h.coeffs, name="scaled_source")
    solver.set_source(scaled_source)
    assert solver.result is None
    assert solver.trace is None
    assert solver.rows is None

    updated_function_result = solve_diffusion_reaction_hdg(
        scaled_source,
        reaction_h,
        exact,
        space,
        **kwargs,
    )
    updated_class_result = solver.solve()

    np.testing.assert_allclose(updated_class_result.trace, updated_function_result.trace, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        updated_class_result.field.coeffs,
        updated_function_result.field.coeffs,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        updated_class_result.flux.as_component_first(),
        updated_function_result.flux.as_component_first(),
        rtol=1e-12,
        atol=1e-12,
    )
    assert not np.allclose(updated_class_result.trace, class_result.trace)
    _assert_solver_cache_matches_result(solver, updated_class_result)


def test_diff_rea_solver_rejects_incomplete_problem_update() -> None:
    solver = DiffReaSolver(_space(order=1), solver="direct", preconditioner=None, verbose=False)
    source, reaction, exact = _callable_problem()

    with pytest.raises(ValueError, match="must be provided together"):
        solver.solve(source=source, reaction=reaction)

    solver.set_problem(source, reaction, exact)
    result = solver.solve()
    assert result.trace is not None


def test_identity_diffusion_argument_preserves_default_solution() -> None:
    space = _space(order=2)
    diffusion, reaction, source, exact = quadratic_poisson_case()
    kwargs = {
        "stabilization": 1.0,
        "solver": "direct",
        "preconditioner": None,
        "boundary_mode": "eliminate",
        "verbose": False,
    }

    default = solve_diffusion_reaction_hdg(source, reaction, exact, space, **kwargs)
    explicit = solve_diffusion_reaction_hdg(source, reaction, exact, space, diffusion=diffusion, **kwargs)

    np.testing.assert_allclose(explicit.trace, default.trace, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(explicit.field.coeffs, default.field.coeffs, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        explicit.flux.as_component_first(),
        default.flux.as_component_first(),
        rtol=0.0,
        atol=0.0,
    )


def test_tensor_diffusion_manufactured_solution_numpy_and_projected_numba_are_accurate() -> None:
    mesh = rectangle_mesh(3, 3)
    space = DGSpace(mesh, 2, basis_type="dub_orth")
    diffusion, reaction, source, exact = tensor_sine_diffusion_reaction_case()
    kwargs = {
        "diffusion": diffusion,
        "stabilization": 4.0,
        "solver": "direct",
        "preconditioner": None,
        "boundary_mode": "eliminate",
        "verbose": False,
    }

    numpy_result = solve_diffusion_reaction_hdg(source, reaction, exact, space, assembly_backend="numpy", **kwargs)
    numba_result = solve_diffusion_reaction_hdg(source, reaction, exact, space, assembly_backend="numba", **kwargs)

    assert numpy_result.field.l2_error(exact) < 2.0e-2
    assert numba_result.field.l2_error(exact) < 2.0e-2
    assert numba_result.local_solver is None
    assert numba_result.element_boundary_mats is None
    np.testing.assert_allclose(numba_result.trace, numpy_result.trace, rtol=0.0, atol=1.0e-3)
    np.testing.assert_allclose(numba_result.field.coeffs, numpy_result.field.coeffs, rtol=0.0, atol=1.0e-3)


def test_tensor_diffusion_flux_uses_conservative_sign() -> None:
    mesh = rectangle_mesh(4, 4)
    space = DGSpace(mesh, 2, basis_type="dub_orth")
    diffusion, reaction, source, exact = tensor_sine_diffusion_reaction_case()
    gradx, grady = tensor_sine_exact_gradients()
    k11, k12, k22 = diffusion

    result = solve_diffusion_reaction_hdg(
        source,
        reaction,
        exact,
        space,
        diffusion=diffusion,
        stabilization=4.0,
        solver="direct",
        preconditioner=None,
        boundary_mode="eliminate",
        verbose=False,
    )

    points = space.mapped_quads()
    x = points[:, :, 0]
    y = points[:, :, 1]
    exact_qx = -(k11(x, y) * gradx(x, y) + k12(x, y) * grady(x, y))
    exact_qy = -(k12(x, y) * gradx(x, y) + k22(x, y) * grady(x, y))
    qx = result.flux.components[0].values()
    qy = result.flux.components[1].values()

    weights = space.quad_data.Krf_w
    jac = space.mesh.aff_jacs
    flux_error = np.sqrt(
        np.einsum("K,Kq,q->", jac, (qx - exact_qx) ** 2 + (qy - exact_qy) ** 2, weights, optimize=True)
    )
    flux_norm = np.sqrt(np.einsum("K,Kq,q->", jac, exact_qx**2 + exact_qy**2, weights, optimize=True))
    assert flux_error / flux_norm < 4.0e-2
