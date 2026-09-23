"""Archived/current ARK orchestration equivalence with canned solvers only."""
import numpy as np
import pytest
from hdgfem.linalg.transport_diagnostics import UpwindHDGTraceRankError
from scripts.guiding_center.time_schemes import STEPPERS, imex_ark3 as current
from scripts.guiding_center.reference.legacy_ark3 import imex_ark3 as legacy
from tests.test_guiding_center_all_scheme_recovery import canned_field_operations, problem


@pytest.mark.parametrize('dt', [.5, 1.])
@pytest.mark.parametrize('rank_retry', [False, True])
def test_legacy_and_current_ark_build_identical_stage_problems(monkeypatch, dt, rank_retry):
    monkeypatch.setattr(legacy, 'field_linear_combination',
                        lambda space, terms, **kwargs: sum(weight*field for weight, field in terms))
    for name in ('perpendicular_vector_field',
                 'solution_field', 'solution_trace', 'field_l2_norm'):
        monkeypatch.setattr(legacy, name, getattr(current, name))
    outcomes, histories, poisson_histories = [], [], []
    for implementation in (current.IMEXARK3Stepper, legacy.IMEXARK3Stepper):
        monkeypatch.setitem(STEPPERS, 'imex-ark3', implementation)
        stepper, poisson, solve, calls, events = problem(
            'imex-ark3', fail_call=1 if rank_retry else None,
            fail_error=UpwindHDGTraceRankError([1], [1], 2))
        stepper.dt = dt
        result = stepper.advance(poisson, solve)
        assert stepper.time == dt
        assert result.metrics['poisson_tau_retry_count'] == int(rank_retry)
        outcomes.append(result)
        histories.append(calls)
        poisson_histories.append(poisson.calls)
    np.testing.assert_array_equal(outcomes[0].density, outcomes[1].density)
    np.testing.assert_array_equal(outcomes[0].density_trace, outcomes[1].density_trace)
    assert len(histories[0]) == len(histories[1])
    for a, b in zip(*histories):
        assert a.keys() == b.keys()
        for key in ('source', 'beta'):
            np.testing.assert_array_equal(a[key], b[key])
        assert (a['stage'], a['reuse'], a['tau']) == (b['stage'], b['reuse'], b['tau'])
    assert len(poisson_histories[0]) == len(poisson_histories[1])
    for a, b in zip(*poisson_histories):
        np.testing.assert_array_equal(a['source'], b['source'])
        np.testing.assert_array_equal(a['guess'], b['guess'])
        assert (a['time'], a['tau']) == (b['time'], b['tau'])
