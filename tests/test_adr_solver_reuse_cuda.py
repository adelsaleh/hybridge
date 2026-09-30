"""Cross-solve caching in AdvectionDiffusionReactionHDGSolver on the raw-CUDA path (TODO L394).

Bounded stationary solves on a 12-triangle mesh: the problem data change
between solves (as in a time step) while the space, diffusion and AMGX config
stay fixed. No time integration.
"""
import numpy as np
import pytest

from hdgfem import (AdvectionDiffusionReactionHDGSolver, DGMesh, DGSpace, rectangle_mesh,
                    solve_advection_diffusion_reaction_hdg)

DIFFUSION = (lambda x, y: .05*(1. + .1*x), lambda x, y: .01 + .002*y, lambda x, y: .04*(1. + .05*y))
BOUNDARY = lambda x, y: .2 + x - .3*y


@pytest.fixture(scope='module')
def cp():
    cupy = pytest.importorskip('cupy')
    pytest.importorskip('pyamgx')
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip('No CUDA device')
    return cupy


def space():
    mesh = rectangle_mesh(3, 2)
    mesh = DGMesh.from_arrays(mesh.node_coords @ np.array([[1.2, .3], [-.1, .8]]), mesh.triangles)
    return DGSpace(mesh, 3, basis_type='dub_orth')


def problems(space, count=5):
    """A sequence of problems with changing advection, reaction and source (fixed diffusion)."""
    for k in range(count):
        beta = (space*space).field((space.constant(.7 - .2*k), space.constant(-.2 + .1*k)))
        yield (lambda x, y, k=k: 1. + .1*k*x*y), beta, 20. + k, BOUNDARY


def options(**extra):
    return dict(diffusion=DIFFUSION, assembly_backend='raw-cuda', solver='amgx', solver_rtol=1e-12,
                raw_matrix_format='bsr', hdg_postprocess='none', materialize_host_solution=False,
                trace_basis='legendre-modal', verbose=False, **extra)


def test_amgx_reuse_needs_the_reusable_solver():
    host_space = space()
    with pytest.raises(ValueError, match='reusable'):
        solve_advection_diffusion_reaction_hdg(1., (host_space*host_space).field((host_space.constant(1.),
                                               host_space.constant(0.))), 1., BOUNDARY, host_space,
                                               **options(amgx_reuse='solver'))
    with pytest.raises(ValueError, match='amgx_reuse must'):
        solve_advection_diffusion_reaction_hdg(1., None, 1., BOUNDARY, host_space, **options(amgx_reuse='always'))


@pytest.mark.parametrize('reuse', ['solver', 'preconditioner'])
def test_reused_solves_match_fresh_solves(cp, reuse):
    dg = space()
    solver = AdvectionDiffusionReactionHDGSolver(dg, **options(amgx_reuse=reuse, amgx_refresh_interval=2))
    reused_flags = []
    for index, problem in enumerate(problems(dg)):
        result = solver.set_problem(*problem).solve()
        fresh = solve_advection_diffusion_reaction_hdg(*problem, dg, **options())
        np.testing.assert_allclose(cp.asnumpy(result.local_unknowns), cp.asnumpy(fresh.local_unknowns),
                                   rtol=1e-8, atol=1e-9)
        reused_flags.append(bool(result.global_solve_result.amgx_preconditioner_reused))
        details = result.timings.details or {}
        if index:
            assert details.get('raw.coefficients.diffusion.cached') == 1.
            assert details.get('raw.coefficients.tau_diffusion.cached') == 1.
            assert details.get('raw.mass_factors.cached') == 1.
    states = list(solver._raw_cache['amgx'].values())
    assert len(states) == 1  # one persistent AMGX solver served every solve
    amgx = states[0]['solver']
    if reuse == 'solver':
        assert reused_flags == [False]*5 and amgx.setup_count == 5
    else:
        # interval 2: fresh, stale, stale, fresh (since_refresh reached 2), stale
        assert reused_flags == [False, True, True, False, True]
        assert amgx.setup_count == 2 and amgx.coefficients_replace_count == 3
    solver.clear_cache()
    assert amgx.closed and solver._raw_cache == {}


def test_failed_stale_solve_retries_with_fresh_setup(cp, monkeypatch):
    from hdgfem.backends import advection_cuda
    import hdgfem.linalg.amgx.device_solver as amgx_device_solver
    dg = space()
    solver = AdvectionDiffusionReactionHDGSolver(dg, **options(amgx_reuse='preconditioner'))
    sequence = list(problems(dg, 3))
    solver.set_problem(*sequence[0]).solve()
    original, calls = amgx_device_solver.solve_reduced_system_amgx_device, []

    def flaky(*args, reuse_primary_preconditioner=False, **kwargs):
        calls.append(reuse_primary_preconditioner)
        if reuse_primary_preconditioner:
            raise RuntimeError('stale preconditioner diverged')
        return original(*args, reuse_primary_preconditioner=reuse_primary_preconditioner, **kwargs)
    monkeypatch.setattr(amgx_device_solver, 'solve_reduced_system_amgx_device', flaky)
    result = solver.set_problem(*sequence[1]).solve()
    assert calls == [True, False] and not result.global_solve_result.amgx_preconditioner_reused
    fresh = solve_advection_diffusion_reaction_hdg(*sequence[1], dg, **options())
    np.testing.assert_allclose(cp.asnumpy(result.local_unknowns), cp.asnumpy(fresh.local_unknowns),
                               rtol=1e-8, atol=1e-9)
    state = next(iter(solver._raw_cache['amgx'].values()))
    assert state['since_refresh'] == 0
    solver.close()
