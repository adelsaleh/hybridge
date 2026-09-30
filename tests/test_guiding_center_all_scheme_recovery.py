"""Failure-injection checks with canned fields/solvers; no meshes, kernels or PDE solves."""
from types import SimpleNamespace
import numpy as np
import pytest

from hdgfem.linalg.results import (
    LinearSolveError,
    LinearSolveConvergenceError,
    LinearSolveCapacityError,
)
from hdgfem.transport.diagnostics import UpwindHDGTraceRankError
from scripts.guiding_center.time_schemes import STEPPERS
from scripts.guiding_center.time_schemes import (
    si_euler, si_bdf2, predictor_corrector, hybrid_bdf3, imex_ark3, recovery, stage_support,
)
from scripts.guiding_center.poisson.poisson_recovery import is_transport_solve_failure


class CannedField(np.ndarray):
    """Small ndarray test double with the naming API used by DG fields."""

    __array_priority__ = 1000

    def __new__(cls, values, *, name="canned_h"):
        result = np.asarray(values, dtype=float).view(cls)
        result.name = name
        return result

    def __array_finalize__(self, source):
        self.name = getattr(source, "name", "canned_h")

    def copy(self, order="C", *, name=None):
        result = super().copy(order=order)
        result.name = self.name if name is None else name
        return result


@pytest.fixture(autouse=True)
def canned_field_operations(monkeypatch):
    def perpendicular(flux, scale, space, **kwargs):
        return scale*CannedField([-flux[1], flux[0]])
    for module in (si_euler, si_bdf2, predictor_corrector, hybrid_bdf3, imex_ark3, recovery):
        for name, function in {
            'perpendicular_vector_field': perpendicular,
            'solution_field': lambda result, space, **kw: result.field,
            'solution_trace': lambda result, space, **kw: result.trace,
            'field_l2_norm': lambda field: float(np.linalg.norm(field)),
        }.items():
            if hasattr(module, name):
                monkeypatch.setattr(module, name, function)
    for module in (predictor_corrector, stage_support):
        monkeypatch.setattr(module, 'trace_linear_combination',
                            lambda terms: sum(w*v for w, v in terms))
    monkeypatch.setattr(predictor_corrector, 'solver_result_metrics', lambda *a: {})


class CannedPoisson:
    def __init__(self, tau):
        self.options = SimpleNamespace(stabilization=tau, diffusion=1.)
        self.space = object()
        self.calls, self.invalidations = [], []
    def with_options(self, **kwargs):
        self.options.stabilization = kwargs['stabilization']
        self.invalidations.append(self.options.stabilization)
    def set_source(self, source):
        self.source = source
    def set_boundary_condition(self, time):
        self.boundary_time = time
    def result(self, density):
        return SimpleNamespace(field=density.copy(), trace=density.copy(),
            flux=CannedField([0., -float(density[0])*self.options.stabilization]))
    def solve(self, *, initial_guess, **kwargs):
        self.calls.append(dict(source=self.source.copy(), time=self.boundary_time,
                               guess=initial_guess.copy(), tau=self.options.stabilization))
        return self.result(self.source)


class CannedResidual:
    backend = 'canned'
    def synchronize(self):
        pass
    def project_trace(self, density):
        return density.copy()
    def evaluate(self, density, drift, boundary=None):
        # A tau-dependent marker, not a discretized PDE residual.
        return drift[:1].copy(), density.copy()


def problem(scheme, *, tau=1., history=False, retries=4, fail_call=None, fail_error=None,
            startup='si-euler-extrap3'):
    p = CannedPoisson(tau)
    events, calls = [], []
    density = CannedField([2.])
    initial = p.result(density)
    options = dict(density_boundary=lambda t: None, potential_boundary=lambda t: t,
                   poisson_solver=p, poisson_tau_max_retries=retries, recovery_record=events.append)
    if scheme in ('si-euler', 'si-bdf2', 'predictor-corrector'):
        stepper = STEPPERS[scheme](p.space, .1, density, initial, density.copy(), **options)
    else:
        if scheme != 'imex-ark3':
            options['startup_method'] = startup
        stepper = STEPPERS[scheme](p.space, .1, density, initial, CannedResidual(), **options)
    if history and scheme == 'si-bdf2':
        stepper.time = .2
        stepper.previous_density = CannedField([1.5])
        stepper.previous_flux = p.result(stepper.previous_density).flux
    if history and scheme in ('h1-bdf3', 'h2-bdf3'):
        stepper.time = .2
        stepper.densities = [density, CannedField([1.5]), CannedField([1.])]
        stepper.drifts = [CannedField([d[0]*tau, 0.]) for d in stepper.densities]
        stepper.drift = stepper.drifts[0]
        if scheme == 'h1-bdf3':
            stepper.residuals = [d[:1].copy() for d in stepper.drifts]
            stepper.drifts = stepper.drifts[:1]
    def transport(source, beta, guess, scale, **kwargs):
        calls.append(dict(source=source.copy(), beta=beta.copy(), tau=p.options.stabilization,
                          stage=kwargs.get('stage'), reuse=kwargs.get('reuse_operator', False)))
        if fail_call == 'always' or len(calls) == fail_call:
            raise (fail_error or LinearSolveConvergenceError('canned nonconvergence'))
        # Canned successful stage result: exercise orchestration, not integration.
        return SimpleNamespace(field=CannedField([3.]), trace=CannedField([3.]))
    return stepper, p, transport, calls, events


CASES = [(scheme, False, 1) for scheme in STEPPERS]
CASES += [('predictor-corrector', False, 2), ('si-bdf2', True, 1)]
CASES += [(scheme, True, stage) for scheme, stages in [('h1-bdf3', (1,)), ('h2-bdf3', (1, 2))]
          for stage in stages]
CASES += [(scheme, False, stage) for scheme in ('h1-bdf3', 'h2-bdf3') for stage in (2, 3, 4, 5, 6)]
CASES += [('imex-ark3', False, stage) for stage in (2, 3)]


@pytest.mark.parametrize('scheme,history,fail_call', CASES)
def test_every_transport_stage_replays_with_rebuilt_poisson_history(scheme, history, fail_call):
    stepper, p, solve, calls, events = problem(scheme, history=history, fail_call=fail_call)
    start = stepper.time
    outcome = stepper.advance(p, solve)
    reference, q, reference_solve, reference_calls, _ = problem(scheme, tau=2., history=history)
    expected = reference.advance(q, reference_solve)
    assert p.invalidations == [2.]
    assert len(events) == 1 and events[0]['reason'] == 'transport-solve-failure'
    assert events[0]['tau_before'] == 1. and events[0]['tau_after'] == 2.
    assert stepper.time == pytest.approx(start+.1)
    assert outcome.metrics['poisson_tau_retry_count'] == 1
    assert outcome.metrics['transport_failed_stage_count'] == 1
    np.testing.assert_allclose(outcome.density, expected.density)
    replay = [call for call in calls if call['tau'] == 2.]
    assert len(replay) == len(reference_calls)
    for actual, target in zip(replay, reference_calls):
        np.testing.assert_allclose(actual['beta'], target['beta'])
        np.testing.assert_allclose(actual['source'], target['source'])
    # Recovery first repeats the owning Poisson evaluation at its original time.
    retry_index = next(i for i, call in enumerate(p.calls) if call['tau'] == 2.)
    retry = p.calls[retry_index]
    assert retry['time'] == pytest.approx(events[0]['poisson_time'])
    if retry_index and events[0]['poisson_stage'] != 'accepted start':
        np.testing.assert_allclose(retry['source'], p.calls[retry_index-1]['source'])
    assert outcome.metrics['transport_attempt_count'] == len(calls)
    assert len(outcome.poisson_results) == len(p.calls)


@pytest.mark.parametrize('scheme', list(STEPPERS))
def test_exhaustion_restores_accepted_state_and_later_call_rebuilds(scheme):
    stepper, p, solve, calls, events = problem(scheme, retries=2, fail_call='always',
                                             history=scheme in ('si-bdf2', 'h1-bdf3', 'h2-bdf3'))
    saved = stepper.__dict__.copy()
    with pytest.raises(LinearSolveConvergenceError):
        stepper.advance(p, solve)
    assert p.invalidations == [2., 4.]
    assert events[-1]['status'] == 'exhausted'
    for key in ('density', 'densities', 'drifts', 'residuals', 'rhs', 'time', 'potential_trace',
                'previous_density', 'previous_flux', 'poisson_result'):
        if key in saved:
            assert stepper.__dict__[key] is saved[key]
    def success(*args, **kwargs):
        return SimpleNamespace(field=CannedField([3.]), trace=CannedField([3.]))
    outcome = stepper.advance(p, success)
    assert outcome.metrics['poisson_tau'] == 4.
    assert stepper.time == pytest.approx(saved['time']+.1)


@pytest.mark.parametrize('scheme', list(STEPPERS))
@pytest.mark.parametrize('error', [LinearSolveCapacityError('capacity'), ValueError('bad config'), RuntimeError('bug')])
def test_non_numerical_failures_do_not_increase_tau(scheme, error):
    stepper, p, solve, _, events = problem(scheme, fail_call=1, fail_error=error)
    with pytest.raises(type(error)):
        stepper.advance(p, solve)
    assert p.invalidations == [] and events == [] and stepper.time == 0.


@pytest.mark.parametrize('error', [LinearSolveError('direct solver failed'),
    LinearSolveConvergenceError('not converged'), np.linalg.LinAlgError('singular'),
    FloatingPointError('nonfinite'), RuntimeError('Factor is exactly singular'),
    UpwindHDGTraceRankError([1], [1], 2)])
def test_backend_independent_numerical_failure_classification(error):
    assert is_transport_solve_failure(error)


@pytest.mark.parametrize('scheme', ['h1-bdf3', 'h2-bdf3'])
@pytest.mark.parametrize('startup', ['ssprk3', 'si-euler-extrap3'])
def test_hybrid_initial_residual_rank_failure_rebuilds_poisson(monkeypatch, scheme, startup):
    if scheme == 'h2-bdf3' and startup != 'ssprk3':
        return  # This initializer only projects a trace; it has no residual solve.
    original = CannedResidual.evaluate
    def fail_once(self, density, drift, boundary=None):
        if drift[0] == 2.:
            raise UpwindHDGTraceRankError([1], [1], 2)
        return original(self, density, drift, boundary)
    monkeypatch.setattr(CannedResidual, 'evaluate', fail_once)
    stepper, p, _, _, events = problem(scheme, startup=startup)
    assert p.invalidations == [2.] and stepper.time == 0.
    assert stepper.initial_recovery_metrics['poisson_tau_retry_count'] == 1
    assert stepper.initial_poisson_result.flux[1] == -4.


@pytest.mark.parametrize('scheme', list(STEPPERS))
@pytest.mark.parametrize('error_type', [LinearSolveError, np.linalg.LinAlgError, FloatingPointError])
def test_direct_factorization_and_arithmetic_failures_recover_in_every_scheme(scheme, error_type):
    stepper, p, solve, _, events = problem(scheme, fail_call=1, fail_error=error_type('numerical failure'))
    outcome = stepper.advance(p, solve)
    assert p.invalidations == [2.]
    assert events[0]['error_type'] == error_type.__name__
    assert outcome.metrics['poisson_tau_retry_count'] == 1


@pytest.mark.parametrize('scheme', list(STEPPERS))
def test_disabled_recovery_propagates_original_error(scheme):
    error = LinearSolveConvergenceError('failed')
    stepper, p, solve, _, events = problem(scheme, retries=0, fail_call=1, fail_error=error)
    with pytest.raises(LinearSolveConvergenceError) as caught:
        stepper.advance(p, solve)
    assert caught.value is error
    assert p.invalidations == [] and events[-1]['status'] == 'exhausted'
    assert stepper.time == 0.


@pytest.mark.parametrize('scheme', list(STEPPERS))
def test_poisson_failure_is_not_misclassified_as_transport_failure(scheme):
    stepper, p, solve, _, events = problem(scheme)
    def failed_poisson(**kwargs):
        raise LinearSolveConvergenceError('Poisson itself failed')
    p.solve = failed_poisson
    with pytest.raises(LinearSolveConvergenceError, match='Poisson itself'):
        stepper.advance(p, solve)
    assert p.invalidations == [] and events == [] and stepper.time == 0.


@pytest.mark.parametrize('scheme', ['h1-bdf3', 'h2-bdf3'])
@pytest.mark.parametrize('startup', ['ssprk3', 'si-euler-extrap3'])
def test_hybrid_residual_stage_failure_replays_with_new_tau(scheme, startup):
    stepper, p, solve, _, events = problem(scheme, startup=startup)
    evaluate = stepper.residual.evaluate
    failed = False
    def fail_once(*args):
        nonlocal failed
        if not failed:
            failed = True
            raise UpwindHDGTraceRankError([1], [1], 2)
        return evaluate(*args)
    # H2 extrapolated-Euler startup has no residual evaluations.
    if scheme == 'h2-bdf3' and startup == 'si-euler-extrap3':
        return
    stepper.residual.evaluate = fail_once
    result = stepper.advance(p, solve)
    assert p.invalidations == [2.]
    assert events[0]['reason'] == 'trace-rank-loss'
    assert result.metrics['explicit_residual_failed_count'] == 1
    assert stepper.time == .1


@pytest.mark.parametrize('scheme', list(STEPPERS))
def test_factory_forwards_recovery_policy_for_every_scheme(monkeypatch, scheme):
    from dataclasses import replace
    from scripts.guiding_center.runtime import steppers
    from scripts.guiding_center.cases.guiding_center_presets import PRESETS
    from hdgfem.assembly import advection_residual
    config = replace(PRESETS['euler_vortex_gas_imex_ark3_p6_h008_dt0005_t50_raw_cuda_bsr'],
                     time_scheme=scheme, poisson_tau_retry_factor=2., poisson_tau_max_retries=7,
                     verbosity=0)
    class Capture(STEPPERS[scheme]):
        def __init__(self, *args, **kwargs):
            self.received = kwargs
    monkeypatch.setitem(steppers.STEPPERS, scheme, Capture)
    monkeypatch.setattr(advection_residual, 'HDGTraceWorkspace', lambda *a, **kw: object())
    monkeypatch.setattr(advection_residual, 'UpwindHDGTransportResidual', lambda *a, **kw: object())
    case = SimpleNamespace(potential_boundary_at=lambda t: t, density_boundary_at=lambda t: None)
    record = lambda event: None
    solver = object()
    stepper = steppers.make_stepper(config, case, object(), object(), object(), object(), object(),
                                   transport_boundary_mode='zero-flux', poisson_solver=solver,
                                   recovery_record=record)
    assert stepper.received['poisson_solver'] is solver
    assert stepper.received['poisson_tau_retry_factor'] == 2.
    assert stepper.received['poisson_tau_max_retries'] == 7
    assert stepper.received['recovery_record'] is record
