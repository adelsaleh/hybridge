"""Unsteady reusable-class validation for AdvectionDiffusionReactionHDGSolver (TODO L393).

Backward Euler and BDF2 are written as stationary ADR solves: the reaction is
shifted by ``alpha`` (``1/dt`` or ``3/(2 dt)``) and the history enters the
source through an ``ElementCoefficient`` evaluated from the previous DG fields.
The exact solution is quadratic in space, so at p=3 the spatial error is far
below the temporal error and the observed rates measure the time integrator.
Bounded checks only: a 4x4 mesh, T=0.4, at most 16 steps per run.
"""
import numpy as np
import pytest

from hdgfem import (AdvectionDiffusionReactionHDGSolver, DGSpace, ElementCoefficient, evaluate_scalar_error,
                    field_values_at_ref, rectangle_mesh, solve_advection_diffusion_reaction_hdg)

BETA = (.7, -.2)
NU = .1
T_FINAL = .4
STEPS = (4, 8, 16)


def exact(x, y, t):
    return np.exp(-t)*(1. + x*x + .5*x*y) + np.sin(2.*t)*(y*y - .3*x)


def source(x, y, t):
    """u_t + div(beta u - K grad u) + r0 u for the symmetric variable tensor K below."""
    a, b = np.exp(-t), np.sin(2.*t)
    u = exact(x, y, t)
    u_t = -a*(1. + x*x + .5*x*y) + 2.*np.cos(2.*t)*(y*y - .3*x)
    u_x, u_y = a*(2.*x + .5*y) - .3*b, .5*a*x + 2.*b*y
    u_xx, u_xy, u_yy = 2.*a, .5*a, 2.*b
    k00, k01, k11 = NU*(1. + .1*x), NU*(.2 + .05*y), NU*(.8 + .1*y)
    div_flux = (NU*.1*u_x + k00*u_xx + k01*u_xy + NU*.05*u_x + k01*u_xy + NU*.1*u_y + k11*u_yy)
    return u_t + BETA[0]*u_x + BETA[1]*u_y + reaction0(x, y)*u - div_flux


def reaction0(x, y):
    return .3 + .1*x


DIFFUSION = (lambda x, y: NU*(1. + .1*x), lambda x, y: NU*(.2 + .05*y), lambda x, y: NU*(.8 + .1*y))


def history_source(space, t_new, terms):
    """f(t_new) + sum(weight*u_i) as an element-local coefficient (host or device)."""
    def function(points, *, xp, t=None):
        mapped = xp.asarray(space.mesh.map_reference_points(points))
        values = source(mapped[..., 0], mapped[..., 1], t_new)
        for weight, field in terms:
            values = values + weight*field_values_at_ref(field, points, device=xp is not np)
        return values
    return ElementCoefficient(function, space.mesh, name='bdf_source')


def options(backend):
    common = dict(diffusion=DIFFUSION, hdg_postprocess='none', trace_basis='legendre-modal', verbose=False)
    if backend == 'numba':
        return dict(assembly_backend='numba', solver='direct', **common)
    return dict(assembly_backend='raw-cuda', solver='amgx', solver_rtol=1e-12, materialize_host_solution=False,
                **common)


def integrate(space, backend, scheme, steps, *, check_fresh=False):
    """Advance to T_FINAL with one reusable solver; return the final field and per-step residuals."""
    dt = T_FINAL/steps
    alpha = 1./dt if scheme == 'euler' else 1.5/dt
    project = lambda t: space.project_callable(lambda x, y: exact(x, y, t))
    beta = (space*space).field((space.constant(BETA[0]), space.constant(BETA[1])))
    reaction = lambda x, y: reaction0(x, y) + alpha
    if scheme == 'euler':
        history, first = [project(0.)], 1
    else:  # exact two-level history, first computed endpoint 2*dt
        history, first = [project(0.), project(dt)], 2
    solver = AdvectionDiffusionReactionHDGSolver(space, options=None, **options(backend))
    residuals = []
    for step in range(first, steps + 1):
        t_new = step*dt
        if scheme == 'euler':
            terms = [(1./dt, history[-1])]
        else:
            terms = [(2./dt, history[-1]), (-.5/dt, history[-2])]
        problem = (history_source(space, t_new, terms), beta, reaction, lambda x, y, t=t_new: exact(x, y, t))
        if step == first:
            solver.set_problem(*problem)
        else:
            solver.set_source(problem[0]).set_boundary_condition(problem[3])
        result = solver.solve()
        solve = result.global_solve_result
        relative = next(value for value in (solve.physical_relative_residual_norm, solve.relative_residual_norm)
                        if value is not None)
        residuals.append(float(relative))
        if check_fresh and step == first + 1:
            fresh = solve_advection_diffusion_reaction_hdg(*problem, space, **options(backend))
            local, reference = (np.asarray(getattr(v, 'get', lambda: v)()) for v in
                                (result.local_unknowns, fresh.local_unknowns))
            # Host: identical direct solves. Raw CUDA: iterative AMGX (rtol 1e-12) and a reused
            # solver that reloads the cached factored mass (equal to roundoff).
            tight = backend == 'numba'
            np.testing.assert_allclose(local, reference, rtol=1e-12 if tight else 1e-9,
                                       atol=1e-13 if tight else 1e-11)
        history = (history + [result.field])[-2:]
    return history[-1], residuals


@pytest.fixture(scope='module')
def space():
    return DGSpace(rectangle_mesh(4, 4, xlim=(0., 1.), ylim=(0., 1.)), 3, basis_type='dub_orth', volume_degree=10)


def run_orders(space, backend, scheme):
    errors, residuals = [], []
    for steps in STEPS:
        field, step_residuals = integrate(space, backend, scheme, steps, check_fresh=steps == STEPS[0])
        errors.append(evaluate_scalar_error(field, lambda x, y: exact(x, y, T_FINAL)).metrics.l2)
        residuals.extend(step_residuals)
    return np.asarray(errors), np.log2(np.asarray(errors[:-1])/np.asarray(errors[1:])), residuals


@pytest.mark.parametrize('scheme,order', [('euler', 1), ('bdf2', 2)])
def test_unsteady_host_temporal_order(space, scheme, order):
    errors, rates, residuals = run_orders(space, 'numba', scheme)
    assert rates[-1] > order - .15, (errors, rates)
    assert max(residuals) < 1e-10, residuals


@pytest.mark.parametrize('scheme,order', [('euler', 1), ('bdf2', 2)])
def test_unsteady_raw_cuda_temporal_order_matches_host(space, scheme, order):
    cp = pytest.importorskip('cupy')
    pytest.importorskip('pyamgx')
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip('No CUDA device')
    host_field, _ = integrate(space, 'numba', scheme, STEPS[0])
    device_field, _ = integrate(space, 'raw-cuda', scheme, STEPS[0])
    assert device_field.device_coefficients_materialized()
    np.testing.assert_allclose(device_field.coeffs, host_field.coeffs, rtol=1e-9, atol=1e-10)
    errors, rates, residuals = run_orders(space, 'raw-cuda', scheme)
    assert rates[-1] > order - .15, (errors, rates)
    assert max(residuals) < 1e-10, residuals
