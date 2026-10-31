"""Static host/device solves verifying frozen transport operator reuse."""
from dataclasses import replace
from copy import deepcopy
import numpy as np
import pytest
from hybridge import DGSpace, VectorDGField, rectangle_mesh
from hybridge.runtime.precision import REAL_DTYPE
from hybridge.solvers.advection_reaction import AdvectionReactionHDGOptions, AdvectionReactionHDGSolver
from hybridge.core.field_ops import solution_field


def problem(order=2):
    space = DGSpace(rectangle_mesh(2, 1), order, basis_type="dub_orth")
    beta = VectorDGField((space.project_callable(lambda x, y: .03*(1+.2*x)),
                          space.project_callable(lambda x, y: .03*(.3+.1*y))))
    return space, beta


@pytest.mark.parametrize("basis", ["legacy-lagrange", "legendre-modal"])
@pytest.mark.parametrize("scaled", [False, True])
def test_host_source_and_boundary_refresh_reuses_matrix_and_preconditioner(basis, scaled):
    space, beta = problem()
    opts = AdvectionReactionHDGOptions(assembly_backend="numpy", boundary_mode="eliminate",
        solver="BICGSTAB", solver_rtol=1e-12, scale_system=scaled, cache_operator=True,
        trace_basis=basis, verbose=0)
    solver = AdvectionReactionHDGSolver(space, source=space.constant(1), beta=beta,
        reaction=space.constant(1), boundary_condition=1., options=opts)
    first = solver.solve()
    matrix = solver._operator_cache["matrix"]
    local = solver.local_solver
    preconditioner = first.global_solve_result.preconditioner
    for marker in (2., -.5):
        src = space.project_callable(lambda x, y: marker+x-y)
        bc = lambda x, y: marker+x+y
        solver.set_source(src, boundary_condition=bc)
        actual = solver.solve(initial_guess=first.trace)
        fresh = AdvectionReactionHDGSolver(space, source=src, beta=beta,
            reaction=space.constant(1), boundary_condition=bc, options=replace(opts, cache_operator=False)).solve()
        np.testing.assert_allclose(actual.field.coeffs, fresh.field.coeffs, atol=1e-10, rtol=1e-10)
        assert solver._operator_cache["matrix"] is matrix
        assert solver.local_solver is local
        assert actual.global_solve_result.preconditioner is preconditioner
        assert actual.timings.details["operator.reused"] == 1.
    solver.set_reaction(space.constant(2))
    assert not solver._operator_cache
    renewed = solver.solve()
    assert renewed.timings.details["operator.reused"] == 0.
    assert solver._operator_cache["matrix"] is not matrix




@pytest.mark.parametrize("change", ["beta", "reaction", "space", "options", "clear"])
def test_cache_invalidates_for_operator_changes(change):
    s, beta = problem()
    solver = AdvectionReactionHDGSolver(s, source=s.constant(1), beta=beta,
        reaction=s.constant(1), boundary_condition=1., assembly_backend="numpy",
        boundary_mode="eliminate", solver="BICGSTAB", cache_operator=True, verbose=0)
    solver.solve()
    previous = solver._operator_cache["matrix"]
    if change == "beta":
        solver.set_beta(beta)
    elif change == "reaction":
        solver.set_reaction(s.constant(2))
    elif change == "space":
        solver.set_space(s)
    elif change == "options":
        solver.with_options(scale_system=False)
    else:
        solver.clear_cache()
    assert not solver._operator_cache
    result = solver.solve()
    assert solver._operator_cache["matrix"] is not previous
    assert result.timings.details["operator.reused"] == 0


def test_clear_factorization_preserves_operator_but_rebuilds_preconditioner():
    s, beta = problem()
    solver = AdvectionReactionHDGSolver(s, source=s.constant(1), beta=beta,
        reaction=s.constant(1), boundary_condition=1., assembly_backend="numpy",
        boundary_mode="eliminate", solver="BICGSTAB", cache_operator=True, verbose=0)
    first = solver.solve()
    matrix = solver._operator_cache["matrix"]
    preconditioner = first.global_solve_result.preconditioner
    solver.clear_factorization()
    result = solver.solve()
    assert solver._operator_cache["matrix"] is matrix
    assert result.global_solve_result.preconditioner is not preconditioner
