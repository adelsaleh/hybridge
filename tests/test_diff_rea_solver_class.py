from __future__ import annotations

import numpy as np
import pytest

from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGField, DGSpace
from hdgfem.solvers.diff_rea import (
    DiffusionReactionHDGSolver as DiffReaSolver,
    solve_diffusion_reaction_hdg,
    test0 as diff_rea_test0,
)


def _space(order: int = 2) -> DGSpace:
    return DGSpace(rectangle_mesh(1, 1), order, basis_type="dub_orth")


def _callable_problem():
    reaction, source, exact = diff_rea_test0()
    return source, reaction, exact


def _projected_problem(space: DGSpace):
    reaction, source, exact = diff_rea_test0()
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
