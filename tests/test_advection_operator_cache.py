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


@pytest.mark.parametrize("order,basis,fmt", [(2, "legacy-lagrange", "csr"), (6, "legendre-modal", "bsr")])
@pytest.mark.parametrize("boundary_mode", ["zero-flux", "eliminate"])
def test_device_source_refresh_matches_fresh_and_retains_amgx_setup(order, basis, fmt, boundary_mode):
    cp = pytest.importorskip("cupy")
    pytest.importorskip("pyamgx")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA device unavailable")
    from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
    from scripts.guiding_center.runtime.configuration import _make_transport_options
    space, beta = problem(order)
    base = _make_transport_options(preset_by_key("euler_vortex_gas_p6_50k_dt001_t50_raw_cuda_bsr"), boundary_mode)
    amgx_config = deepcopy(base.amgx_config)
    amgx_config["solver"].update(convergence="ABSOLUTE", tolerance=1e-13)
    opts = replace(base, amgx_config=amgx_config, cache_operator=True, raw_local_assembly="fused", raw_matrix_format=fmt,
                   trace_basis=basis, verbose=0, amgx_retry_attempts=(), solver_rtol=1e-10,
                   materialize_host_solution=False)
    bc = None if boundary_mode == "zero-flux" else 1.
    solver = AdvectionReactionHDGSolver(space, source=space.constant(1), beta=beta,
                                       reaction=space.constant(1), boundary_condition=bc, options=opts)
    try:
        first = solver.solve()
        field0 = solution_field(first, space).coeffs.copy()
        assembly = solver._operator_cache["assembly"]["cuda_assembly"]
        lu = solver._raw_cuda_factor_workspace.local_lu
        amgx = solver._raw_cuda_amgx_retry_solver_cache[("fixed-operator", "primary")]
        assert amgx.setup_count == 1
        for marker in (2., -.5):
            src = space.project_callable(lambda x, y: marker+x-y)
            bc = None if boundary_mode == "zero-flux" else lambda x, y: marker+x+y
            solver.set_source(src, boundary_condition=bc)
            actual = solver.solve(initial_guess=first.trace_reduced_device.copy())
            fresh_solver = AdvectionReactionHDGSolver(space, source=src, beta=beta,
                reaction=space.constant(1), boundary_condition=bc, options=replace(opts, cache_operator=False))
            try:
                fresh = fresh_solver.solve()
                np.testing.assert_allclose(solution_field(actual, space).coeffs,
                                           solution_field(fresh, space).coeffs, atol=1e-8, rtol=1e-8)
            finally:
                fresh_solver.close()
            current = solver._operator_cache["assembly"]["cuda_assembly"]
            assert current.data is assembly.data
            assert current.indptr is assembly.indptr
            assert solver._raw_cuda_factor_workspace.local_lu is lu
            assert amgx.setup_count == 1
            assert actual.global_solve_result.amgx_preconditioner_reused
            assert actual.timings.details["operator.reused"] == 1.
            np.testing.assert_array_equal(solution_field(first, space).coeffs, field0)
        solver.set_reaction(space.constant(2))
        solver.solve()
        assert amgx.setup_count == 2
        assert solver._raw_cuda_factor_workspace.local_lu is lu
    finally:
        solver.close()
        # Flush native AMGX startup text inside this test's capture boundary.
        import ctypes
        ctypes.CDLL(None).fflush(None)


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
