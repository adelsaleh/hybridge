"""Poisson tau backtracking: provenance, atomic replay, caches and residency."""
from dataclasses import replace
from types import SimpleNamespace
import json
import numpy as np
import pytest

from hdgfem import DGSpace, VectorDGField, rectangle_mesh
from hdgfem.assembly.advection_residual import UpwindHDGTransportResidual
from hdgfem.core.field_ops import field_linear_combination, perpendicular_vector_field
from hdgfem.linalg.transport_diagnostics import (
    UpwindHDGTraceRankError, transport_rank_failure_details,
)
from hdgfem.solvers.diffusion_reaction import DiffusionReactionHDGOptions
from scripts.guiding_center.time_schemes import imex_ark3 as ark
from scripts.guiding_center.poisson.poisson_recovery import PoissonTauRecovery
from tests.test_guiding_center_imex_ark3 import scalar_stepper


def tau_problem(*, tau=1000., fail_call=1, required_tau=2000., max_retries=4):
    """The actual split changes with tau: F(y) = -(tau/1000) y**2."""
    stepper, p, transport, calls, scalar = scalar_stepper(.1)
    p.options = DiffusionReactionHDGOptions(stabilization=tau)
    p.space = stepper.space
    p.sources, p.invalidations = [], []
    original_solve = p.solve
    def solve(**kwargs):
        result = original_solve(**kwargs)
        p.sources.append((scalar(p.source), p.options.stabilization, kwargs['initial_guess'].copy()))
        return SimpleNamespace(field=result.field, trace=result.trace,
            flux=VectorDGField(tuple(field_linear_combination(p.space,
                [(p.options.stabilization/1000, f)]) for f in result.flux.components)))
    def with_options(**kwargs):
        p.options = p.options.with_overrides(**kwargs)
        p.invalidations.append(kwargs['stabilization'])
    p.solve, p.with_options = solve, with_options
    stepper.accepted_poisson_tau = tau
    stepper.drift = VectorDGField(tuple(field_linear_combination(p.space, [(tau/1000, f)])
                                         for f in stepper.drift.components))
    stepper.rhs, stepper.density_trace = stepper.residual.evaluate(stepper.density, stepper.drift)
    original_evaluate = stepper.residual.evaluate
    triggered = False
    def evaluate(*args, **kwargs):
        nonlocal triggered
        if len(p.sources) == fail_call:
            triggered = True
        if triggered and p.options.stabilization < required_tau:
            raise UpwindHDGTraceRankError([156512], [6], 7)
        return original_evaluate(*args, **kwargs)
    stepper.residual.evaluate = evaluate
    events = []
    stepper.recovery_options.update(max_retries=max_retries, record=events.append)
    return stepper, p, transport, calls, scalar, events


@pytest.mark.parametrize('fail_call', [1, 2, 3, 4])
def test_replays_all_ark_stages_at_one_tau_and_warms_exact_poisson_checkpoint(fail_call):
    stepper, p, transport, calls, scalar, events = tau_problem(fail_call=fail_call)
    result = stepper.advance(p, transport)
    reference, q, solve, _, _, _ = tau_problem(tau=2000., fail_call=-1)
    expected = reference.advance(q, solve)
    np.testing.assert_allclose(result.density.coeffs, expected.density.coeffs, rtol=1e-14)
    assert p.invalidations == [2000.]
    # First retry must use the failed stage's source and potential trace, at the
    # same time. Older stages are rebuilt afterward, under the new tau.
    failed, repeated = p.sources[fail_call-1:fail_call+1]
    assert repeated[0] == failed[0] and repeated[1] == 2000.
    np.testing.assert_allclose(repeated[2], failed[0])
    assert events[0]['poisson_stage'] == ('accepted endpoint' if fail_call == 4 else f'ARK stage {fail_call+1}')
    assert result.metrics['poisson_tau_retry_count'] == 1
    assert result.metrics['poisson_tau'] == 2000.
    assert result.metrics['imex_ark3_operator_assemblies'] == 2
    assert result.metrics['explicit_residual_failed_count'] == 1
    assert [c['reuse'] for c in calls[-3:]] == [False, True, True]
    # Replayed stage 2 warms from the discarded stage 2 at the identical time.
    np.testing.assert_allclose(calls[-3]['guess'], calls[0]['value'])
    assert len(result.poisson_results) == fail_call+6
    assert len(result.metrics['poisson_stage_times']) == len(result.poisson_results)
    assert len(result.metrics['transport_stage_wall_times']) == len(calls)
    assert stepper.time == .1
    before = len(p.invalidations)
    second = stepper.advance(p, transport)
    assert second.metrics['poisson_tau_retry_count'] == 0
    assert len(p.invalidations) == before
    assert second.metrics['imex_ark3_operator_assemblies'] == 1


def test_repeated_doubling_is_bounded_and_exhaustion_does_not_commit():
    stepper, p, transport, _, _, events = tau_problem(required_tau=8000., max_retries=2)
    saved = (stepper.density, stepper.drift, stepper.rhs,
             stepper.density_trace, stepper.potential_trace)
    with pytest.raises(UpwindHDGTraceRankError) as raised:
        stepper.advance(p, transport)
    assert p.invalidations == [2000., 4000.]
    assert stepper.time == 0.
    assert all(x is y for x, y in zip(saved, (stepper.density, stepper.drift, stepper.rhs,
                                             stepper.density_trace, stepper.potential_trace)))
    assert events[-1]['status'] == 'exhausted'
    assert 'exhausted after 2 retries' in raised.value.__notes__[0]
    # A later retry must detect the changed solver tau and rebuild the accepted
    # starting field, even though the preceding attempt exhausted its budget.
    stepper.recovery_options['max_retries'] = 2
    result = stepper.advance(p, transport)
    assert result.metrics['poisson_tau'] == 8000.
    assert stepper.time == .1


def test_disabled_recovery_and_unrelated_errors_do_not_change_tau():
    stepper, p, transport, _, _, events = tau_problem(max_retries=0)
    with pytest.raises(UpwindHDGTraceRankError):
        stepper.advance(p, transport)
    assert p.invalidations == [] and stepper.time == 0.
    assert events[-1]['status'] == 'exhausted'
    stepper, p, transport, _, _, events = tau_problem(fail_call=-1)
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError('unrelated factorization error')
    stepper.residual.evaluate = fail
    with pytest.raises(np.linalg.LinAlgError, match='unrelated'):
        stepper.advance(p, transport)
    assert p.invalidations == [] and events == [] and stepper.time == 0.


def test_frozen_transport_rank_failure_rebuilds_start_field_and_operator():
    stepper, p, transport, calls, _, events = tau_problem(fail_call=-1)
    def fail_first(*args, **kwargs):
        if not p.invalidations:
            raise UpwindHDGTraceRankError([9], [6], 7)
        return transport(*args, **kwargs)
    result = stepper.advance(p, fail_first)
    assert events[0]['poisson_time'] == 0.
    assert events[0]['poisson_stage'] == 'accepted start'
    assert result.metrics['transport_failed_stage_count'] == 1
    assert result.metrics['imex_ark3_operator_assemblies'] == 2
    assert [c['reuse'] for c in calls] == [False, True, True]


def test_initial_rank_failure_rebuilds_poisson_before_starting():
    old, p, _, _, _, events = tau_problem(fail_call=0)
    result = SimpleNamespace(flux=VectorDGField((p.space.zeros(), p.space.constant(-1.))),
                             trace=old.potential_trace)
    initialized = ark.IMEXARK3Stepper(p.space, .1, old.density, result, old.residual,
        density_boundary=lambda t: None, potential_boundary=lambda t: None,
        poisson_solver=p, recovery_record=events.append)
    assert initialized.initial_recovery_metrics['poisson_tau_retry_count'] == 1
    assert initialized.accepted_poisson_tau == 2000.
    assert initialized.time == 0.
    assert p.invalidations == [2000.]


@pytest.mark.parametrize('options', [dict(factor=1), dict(factor=float('nan')),
                                    dict(max_retries=-1), dict(max_retries=1.5)])
def test_invalid_retry_policy_rejected(options):
    with pytest.raises(ValueError):
        PoissonTauRecovery(**options)


def test_rank_classification_requires_active_structurally_deficient_face():
    error = RuntimeError('iterative solver failed')
    assert transport_rank_failure_details(error) is None
    error.transport_diagnostics = {'trace_inflow_diagnostics': {
        'trace_dofs_per_face': 7, 'worst_faces': [dict(edge=9, inflow_nodes=0,
                                                    outward_normal_samples=[[0.]*13]*2)]}}
    assert transport_rank_failure_details(error) is None
    face = error.transport_diagnostics['trace_inflow_diagnostics']['worst_faces'][0]
    face.update(inflow_nodes=6, outward_normal_samples=[[1.]*7+[-1.]*6]*2)
    assert transport_rank_failure_details(error) == {'edges': [9], 'inflow_nodes': [6], 'trace_dofs': 7}


def test_device_face_solve_never_calls_numpy_solve(monkeypatch):
    cp = pytest.importorskip('cupy')
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip('CUDA unavailable')
    from hdgfem.backends.cupy import field_from_cupy_coefficients, as_cupy_coefficients, as_cupy_space
    space = DGSpace(rectangle_mesh(2, 1), 6, basis_type='dub_orth')
    density, bx, by = space.constant(2), space.constant(.2), space.constant(.5)
    device_fields = [field_from_cupy_coefficients(space, cp.asarray(f.coeffs), device=0)
                     for f in (density, bx, by)]
    residual = UpwindHDGTransportResidual(space, backend='device')
    def forbidden(*args, **kwargs):
        raise AssertionError('NumPy solve was used on the device path')
    monkeypatch.setattr(np.linalg, 'solve', forbidden)
    result, trace = residual.evaluate(device_fields[0], VectorDGField(tuple(device_fields[1:])))
    residual.synchronize()
    assert isinstance(trace, cp.ndarray)
    assert isinstance(as_cupy_coefficients(result, as_cupy_space(space)), cp.ndarray)
    assert all(not field.coefficients_materialized for field in [*device_fields, result])


@pytest.mark.parametrize('backend', ['host', 'device'])
def test_real_poisson_retry_rebuilds_then_reuses_and_records_all_work(tmp_path, monkeypatch, capfd, backend):
    from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
    from scripts.guiding_center.runtime import runner
    if backend == 'device':
        cp = pytest.importorskip('cupy')
        pytest.importorskip('pyamgx')
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('CUDA unavailable')
    common = dict(nx=4, ny=4, order=3, dt=.002, num_steps=2,
                  time_scheme='imex-ark3', poisson_tau=1000., verbosity=0,
                  plot_every=0, diagnostics_every=1, diagnostics_dir=str(tmp_path))
    config = replace(preset_by_key('rho_helm_wave_host_accuracy' if backend == 'host'
                                  else 'rho_helm_wave_raw_cuda_amgx_accuracy'), **common,
                     diagnostics_prefix='recovery')
    if backend == 'device':
        config = replace(config, poisson_assembly_backend='raw-cuda',
            poisson_cache_local_factors='schur-lu', poisson_raw_matrix_format='csr',
            transport_materialize_host_solution=False)
    evaluate = UpwindHDGTransportResidual.evaluate
    calls = 0
    def inject_stage2(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise UpwindHDGTraceRankError([7], [2], 4)
        return evaluate(self, *args, **kwargs)
    monkeypatch.setattr(UpwindHDGTransportResidual, 'evaluate', inject_stage2)
    actual = runner.run_guiding_center_case(config)
    monkeypatch.setattr(UpwindHDGTransportResidual, 'evaluate', evaluate)
    reference = runner.run_guiding_center_case(replace(config, poisson_tau=2000., diagnostics_prefix='reference'))
    tolerance = 2e-10 if backend == 'device' else 1e-12
    if backend == 'device':
        assert not actual.final_density.coefficients_materialized
    np.testing.assert_allclose(actual.final_density.coeffs, reference.final_density.coeffs,
                               rtol=tolerance, atol=tolerance)
    assert actual.config.poisson_tau == 2000.
    rows = [json.loads(line) for line in actual.timings_jsonl_path.read_text().splitlines()]
    assert rows[1]['poisson_tau_retry_count'] == 1
    assert rows[1]['poisson_stage_count'] == 7 and rows[1]['transport_stage_count'] == 4
    assert rows[1]['explicit_residual_failed_count'] == 1
    assert rows[2]['poisson_tau_retry_count'] == 0
    assert rows[2]['poisson_stage_count'] == 4 and rows[2]['transport_stage_count'] == 3
    assert [rows[1][f'stage{i}_transport_detail_operator_reused'] for i in (1, 2, 3, 4)] == [0, 0, 1, 1]
    if backend == 'device':
        assert [rows[1][f'stage{i}_poisson_detail_raw_assembly_operator_reused']
                for i in range(1, 8)] == [1, 0, 1, 1, 1, 1, 1]
    events = [json.loads(line) for line in (tmp_path/'recovery_poisson_tau_recovery.jsonl').read_text().splitlines()]
    assert len(events) == 1 and events[0]['tau_after'] == 2000.
    assert events[0]['poisson_stage'] == 'ARK stage 2'
    # Flush native output while pytest's FD capture is still active.
    import ctypes
    ctypes.CDLL(None).fflush(None)
    capfd.readouterr()


def test_rejected_result_snapshot_preserves_metrics_without_solution_or_preconditioner():
    from hdgfem.diagnostics import solver_diagnostics_snapshot, solver_result_metrics
    from hdgfem.linalg.system import SolveResult
    global_result = SolveResult(x=np.ones(4), x_device=object(), preconditioner=object(),
                                iteration_count=5, status='converged')
    global_result.amgx_attempts = [dict(iterations=5)]
    result = SimpleNamespace(field=object(), matrix_data=np.eye(4), global_solve_result=global_result,
        timings=SimpleNamespace(total=4., assembly=1., solve=2., reconstruction=1., details={}),
        assembly_backend='raw-cuda', boundary_mode='zero-flux')
    snapshot = solver_diagnostics_snapshot(result)
    assert solver_result_metrics('stage', snapshot) == solver_result_metrics('stage', result)
    assert not hasattr(snapshot, 'field') and not hasattr(snapshot, 'matrix_data')
    assert snapshot.global_solve_result.x is None
    assert snapshot.global_solve_result.x_device is None
    assert snapshot.global_solve_result.preconditioner is None
    assert global_result.x is not None and global_result.preconditioner is not None
