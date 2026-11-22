"""Host tests of calibration, budget, cache isolation, and failure behavior."""
from dataclasses import replace

import numpy as np
import pytest

from hybridge.runtime import cuda_tuning as tuning


@pytest.fixture
def calibration(monkeypatch, tmp_path):
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING', 'auto')
    monkeypatch.setattr(tuning, 'device_signature', lambda cp: dict(name='test', major=9, uuid='one'))
    tuning.clear_tuning_cache()
    active, launches = {}, []
    speeds = {'reference': .01, 'fast': .005, 'wrong': .001, 'nan': .001, 'noise': .0099}
    def run(name, handle):
        launches.append(name)
        active['name'] = name
        return np.array([np.nan if name == 'nan' else 2. if name == 'wrong' else 1.])
    def event(cp, function):
        function()
        return speeds[active['name']]
    monkeypatch.setattr(tuning, 'event_seconds', event)
    kwargs = dict(cupy=np, identity=dict(family='test', source_version='one', precision='float64'),
                  candidates=['reference', 'fast', 'wrong', 'nan'], prepare=lambda name: name,
                  run=run, budget=tuning.TuningBudget(30., 8, 3, 1))
    return kwargs, launches


def test_only_correct_candidates_can_win_and_disk_hit_never_runs_trials(calibration):
    kwargs, launches = calibration
    choice, _, report = tuning.select_variant(**kwargs)
    assert choice == 'fast'
    assert [r.get('rejected') for r in report['candidates']] == [None, None, 'numerical parity', 'numerical parity']
    tuning.clear_tuning_cache()
    launches.clear()
    assert tuning.select_variant(**kwargs)[2]['source'] == 'disk'
    assert launches == []


def test_trial_preparation_only_on_misses_and_rejections_count(calibration):
    kwargs, launches = calibration
    prepared = []
    def trial_prepare(name, handle):
        prepared.append(name)
        if name == 'wrong':
            raise tuning.CandidateUnavailable('optional plan unavailable')
    kwargs['trial_prepare'] = trial_prepare
    choice, _, report = tuning.select_variant(**kwargs)
    assert choice == 'fast'
    assert report['candidates'][2]['rejected'] == 'optional plan unavailable'
    assert 'wrong' not in launches
    prepared.clear()
    tuning.clear_tuning_cache()
    assert tuning.select_variant(**kwargs)[2]['source'] == 'disk'
    assert prepared == []


def test_shared_deadline_stops_further_trials_without_changing_identity(calibration):
    kwargs, launches = calibration
    choice, _, report = tuning.select_variant(**kwargs, calibration_deadline=0.)
    assert choice == 'reference' and launches == ['reference']
    assert len(report['candidates']) == 1
    tuning.clear_tuning_cache()
    launches.clear()
    assert tuning.select_variant(**kwargs, calibration_deadline=float('inf'))[2]['source'] == 'disk'
    assert launches == []


@pytest.mark.parametrize('change', ['source', 'precision', 'device', 'candidates', 'budget'])
def test_relevant_changes_invalidate_cached_winner(calibration, monkeypatch, change):
    kwargs, launches = calibration
    tuning.select_variant(**kwargs)
    launches.clear()
    kwargs['identity'] = dict(kwargs['identity'])
    if change == 'source':
        kwargs['identity']['source_version'] = 'two'
    elif change == 'precision':
        kwargs['identity']['precision'] = 'float32'
    elif change == 'device':
        monkeypatch.setattr(tuning, 'device_signature', lambda cp: dict(name='test', major=9, uuid='two'))
    elif change == 'candidates':
        kwargs['candidates'] = kwargs['candidates'][:2]
    else:
        kwargs['budget'] = replace(kwargs['budget'], candidates=2)
    assert tuning.select_variant(**kwargs)[2]['source'] == 'measured'
    assert launches


def test_count_limit_counts_baseline_and_rejected_compiles(calibration):
    kwargs, launches = calibration
    kwargs['budget'] = tuning.TuningBudget(30., 2, 1, 0)
    kwargs['candidates'] = ['reference', 'unavailable', 'fast']
    def prepare(name):
        if name == 'unavailable':
            raise tuning.CandidateUnavailable('no optional library')
        return name
    kwargs['prepare'] = prepare
    name, _, report = tuning.select_variant(**kwargs)
    assert name == 'reference' and len(report['candidates']) == 2
    assert 'fast' not in launches
    assert report['stop_reason'] == 'candidate limit'


def test_baseline_compile_overrun_keeps_validated_fallback_without_further_trials(calibration, monkeypatch):
    kwargs, launches = calibration
    now = [0.]
    monkeypatch.setattr(tuning.time, 'perf_counter', lambda: now[0])
    def prepare(name):
        now[0] += 31.
        return name
    kwargs['prepare'] = prepare
    name, _, report = tuning.select_variant(**kwargs)
    assert name == 'reference' and launches == ['reference']
    assert report['stop_reason'] == 'time limit'
    assert report['candidates'][0]['samples_seconds'] == []


def test_explicit_and_disabled_modes_never_run_trials(calibration, monkeypatch):
    kwargs, launches = calibration
    assert tuning.select_variant(**kwargs, explicit='fast')[0] == 'fast'
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING', 'off')
    assert tuning.select_variant(**kwargs)[0] == 'reference'
    assert not launches


def test_noise_margin_and_refresh(calibration):
    kwargs, launches = calibration
    kwargs['candidates'] = ['reference', 'noise']
    assert tuning.select_variant(**kwargs)[0] == 'reference'
    launches.clear()
    assert tuning.select_variant(**kwargs, refresh=True)[2]['source'] == 'measured'
    assert launches


def test_independent_residual_gate_and_fatal_launch_errors(calibration):
    kwargs, _ = calibration
    kwargs['candidates'] = ['reference', 'fast']
    state = {}
    original_run = kwargs['run']
    def run(name, handle):
        state['name'] = name
        return original_run(name, handle)
    kwargs.update(run=run, validate=lambda out: state['name'] == 'reference')
    assert tuning.select_variant(**kwargs)[0] == 'reference'
    def broken(name, handle):
        raise RuntimeError('illegal device access')
    kwargs['run'] = broken
    with pytest.raises(RuntimeError, match='illegal device access'):
        tuning.select_variant(**kwargs, refresh=True)
