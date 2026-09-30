"""NGammaBDF2Stepper contract tests with stub forcing (TODO L98 D1).

Bounded checks on a 12-triangle mesh with at most three steps: no study runs.
Host references use Numba assembly with the direct solver.
"""
import numpy as np
import pytest

from hdgfem import DGSpace, rectangle_mesh
from scripts.n_gamma import coefficients as nc
from scripts.n_gamma.stepper import NGammaBDF2Stepper, StepRejected

N0 = 2.
HOST = dict(assembly_backend='numba', solver='direct', trace_basis='legendre-modal')


def b_poloidal(R, Z):
    x, y = R - 3., Z
    scale = 1./np.sqrt(1. + x*x + y*y)
    return -y*scale, x*scale


@pytest.fixture(scope='module')
def space():
    return DGSpace(rectangle_mesh(3, 2, xlim=(2., 4.), ylim=(-1., 1.)), 2, basis_type='dub_orth')


def zero(t):
    return lambda R, Z: 0.*R


def constant(value):
    return lambda t: (lambda R, Z: value + 0.*R)


def make(space, *, previous=True, options=HOST, density=None, momentum=None, geometry='axisymmetric', **kwargs):
    """Stepper for the exact discrete equilibrium n=N0, Gamma=0 (zero sources)."""
    density = space.constant(N0) if density is None else density
    momentum = space.zeros() if momentum is None else momentum
    history = dict(previous_density=density, previous_momentum=momentum) if previous else {}
    settings = dict(dt=.05, time=0., geometry=geometry, b_poloidal=b_poloidal, diffusion=.02, viscosity=.03,
                    density_floor=1e-8,
                    source_density=zero, source_momentum=zero, boundary_density=constant(N0),
                    boundary_momentum=constant(0.), options=options)
    settings.update(kwargs)
    return NGammaBDF2Stepper(space, density, momentum, **history, **settings)


def test_incomplete_history_is_rejected(space):
    with pytest.raises(ValueError, match='together'):
        NGammaBDF2Stepper(space, space.constant(N0), space.zeros(), previous_density=space.constant(N0),
                          dt=.1, time=0., geometry='cartesian', b_poloidal=b_poloidal, diffusion=.02, viscosity=.03,
                          density_floor=1e-8,
                          source_density=zero, source_momentum=zero, boundary_density=constant(N0),
                          boundary_momentum=constant(0.))
    with pytest.raises(ValueError, match='floor'):
        make(space, density_floor=0.)
    with pytest.raises(ValueError, match='geometry'):
        make(space, geometry='slab')


@pytest.mark.parametrize('geometry', ['cartesian', 'axisymmetric'])
def test_equilibrium_is_preserved_through_euler_startup_and_bdf2(space, geometry):
    """Weighted reaction, history and diffusion are consistent: n=N0, Gamma=0 is a discrete fixed point."""
    stepper = make(space, previous=False, geometry=geometry)
    stages = []
    for _ in range(3):
        result = stepper.advance()
        stages.append((result.diagnostics.stage, result.diagnostics.alpha))
        n, gamma = stepper.current
        np.testing.assert_allclose(n.coeffs, space.constant(N0).coeffs, rtol=0, atol=1e-11)
        np.testing.assert_allclose(gamma.coeffs, 0., atol=1e-11)
    assert stages == [('euler', 20.), ('bdf2', 30.), ('bdf2', 30.)]
    assert stepper.step_count == 3 and stepper.time == pytest.approx(.15)
    diagnostics = result.diagnostics
    assert diagnostics.floor.volume_clamps == 0 and diagnostics.floor.min_extrapolated_density == pytest.approx(N0)
    assert diagnostics.min_density_sampled == pytest.approx(N0)
    assert diagnostics.density_solve.relative_residual < 1e-10


def test_solve_order_frozen_advection_and_new_density_pressure(space, monkeypatch):
    density = space.project_callable(lambda R, Z: 2. + .1*np.sin(2*R)*np.cos(Z))
    momentum = space.project_callable(lambda R, Z: .2 + .1*R*Z)
    stepper = make(space, density=density, momentum=momentum, source_density=lambda t: (lambda R, Z: .1*R),
                   source_momentum=lambda t: (lambda R, Z: np.cos(Z) + t))
    calls = []
    for name, solver in zip(('density', 'momentum'), stepper.solvers):
        original = solver.solve

        def recorded(*args, _name=name, _original=original, _solver=solver, **kwargs):
            calls.append((_name, _solver.beta, _solver.source))
            return _original(*args, **kwargs)
        monkeypatch.setattr(solver, 'solve', recorded)
    result = stepper.advance()
    assert [name for name, _, _ in calls] == ['density', 'momentum']
    assert calls[0][1] is calls[1][1]  # one frozen advection coefficient for both solves
    # BDF2 extrapolated velocity from the exact-history pair (n^{k-1} = n^k here).
    quad = space.quad_data.Krf_quads
    mapped = space.mesh.map_reference_points(quad)
    R, Z = mapped[..., 0], mapped[..., 1]
    b_r, _ = b_poloidal(R, Z)
    np.testing.assert_allclose(calls[0][1].volume_values(space)[..., 0],
                               R*momentum.values_at_ref(quad)/density.values_at_ref(quad)*b_r, rtol=1e-12, atol=1e-13)
    # The momentum source uses the newly solved density and the BDF2 history (4w^k - w^{k-1})/(2dt).
    n_new = result.density.field
    dn_dr, dn_dz = n_new.grad_at_ref(quad)
    b_r, b_z = b_poloidal(R, Z)
    expected = R*(np.cos(Z) + .05 + 1.5/.05*momentum.values_at_ref(quad) - (b_r*dn_dr + b_z*dn_dz))
    np.testing.assert_allclose(calls[1][2].volume_values(space), expected, rtol=1e-11, atol=1e-10)


def test_failed_momentum_solve_rolls_back(space, monkeypatch):
    stepper = make(space)
    before = (stepper.current, stepper.previous, stepper.time, stepper.step_count, stepper.traces)

    def fail(*args, **kwargs):
        raise RuntimeError('injected AMGX failure')
    monkeypatch.setattr(stepper.solvers[1], 'solve', fail)
    with pytest.raises(StepRejected, match='momentum solve failed: injected') as info:
        stepper.advance()
    assert info.value.diagnostics.accepted is False and info.value.diagnostics.density_solve is not None
    after = (stepper.current, stepper.previous, stepper.time, stepper.step_count, stepper.traces)
    assert all(a is b for a, b in zip(before[:2], after[:2])) and before[2:] == after[2:]
    monkeypatch.undo()
    assert stepper.advance().diagnostics.step == 1


def test_nonfinite_extrapolation_is_rejected_not_floored(space):
    bad = space.field(np.full(space.shape, np.nan))
    stepper = make(space, density=bad, previous=False)
    with pytest.raises(StepRejected, match='not finite'):
        stepper.advance()
    assert stepper.step_count == 0


def test_raw_cuda_steps_stay_device_resident(space):
    cp = pytest.importorskip('cupy')
    pytest.importorskip('pyamgx')
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip('No CUDA device')
    density = space.project_callable(lambda R, Z: 2. + .1*np.sin(2*R)*np.cos(Z))
    momentum = space.project_callable(lambda R, Z: .2 + .1*R*Z)
    common = dict(density=density, momentum=momentum, source_density=lambda t: (lambda R, Z: .1*R),
                  source_momentum=lambda t: (lambda R, Z: np.cos(Z) + t))
    host = make(space, previous=False, **common)
    device = make(space, previous=False, options=dict(assembly_backend='raw-cuda', solver='amgx',
                  solver_rtol=1e-12, materialize_host_solution=False, trace_basis='legendre-modal'), **common)
    for _ in range(2):
        host.advance()
        result = device.advance()
    for field in device.current:
        assert field.device_coefficients_materialized() and not field.coefficients_materialized
    assert isinstance(result.density.local_unknowns, cp.ndarray)
    for device_field, host_field in zip(device.current, host.current):
        np.testing.assert_allclose(device_field.coeffs, host_field.coeffs, rtol=1e-8, atol=1e-9)


def test_failed_solve_retries_once_with_fallback_and_restores_options(space, monkeypatch):
    stepper = make(space, fallback_options=dict(solver_rtol=1e-9))
    density_solver = stepper.solvers[0]
    original, calls = density_solver.solve, []

    def flaky(**overrides):
        calls.append(dict(overrides))
        if not overrides:
            raise RuntimeError('primary preconditioner diverged')
        return original(**overrides)
    monkeypatch.setattr(density_solver, 'solve', flaky)
    rtol = density_solver.options.solver_rtol
    result = stepper.advance()
    assert calls == [{}, {'solver_rtol': 1e-9}]
    assert result.diagnostics.density_solve.fallback and not result.diagnostics.momentum_solve.fallback
    assert density_solver.options.solver_rtol == rtol  # the fallback does not persist
    assert stepper.step_count == 1 and result.diagnostics.dt == stepper.dt
