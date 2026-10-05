"""Keep positive-case recovery scoped and preserve GPU handoff and acceptance."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from scripts.reports import record_gpu_showcase as recorder
from hybridge.linalg.results import LinearSolveConvergenceError


def test_positive_options_reuse_robust_runner_attempts_and_corrected_upwind():
    """The recorder selects the existing bounded retries rather than inventing them."""
    options = recorder.positive_transport_options()
    assert options['advection_stabilization'] == 'conflict-averaged-upwind'
    assert [a['label'] for a in options['amgx_retry_attempts']] == [
        'pbicgstab-l1-zero', 'pbicgstab-block-jacobi-zero', 'robust-zero-scaled',
        'robust-correction-1', 'robust-correction-2']
    assert options['amgx_config']['solver']['solver'] == 'BICGSTAB'


def test_accepted_gpu_solve_never_constructs_host_solver(monkeypatch):
    """Normal stepping remains on the GPU with no CPU factorization."""
    accepted = object()
    transport = SimpleNamespace(solve=lambda **kw: accepted)
    monkeypatch.setattr(recorder.hdg, 'AdvectionReactionHDGSolver',
                        lambda *a, **kw: pytest.fail('unnecessary host solver'))
    metadata = {}
    assert recorder.solve_positive_transport(transport, None, None, None, None, None, 1, metadata) is accepted
    assert metadata == {}


@pytest.mark.parametrize('handoff', [True, False])
def test_exhausted_gpu_attempts_use_pardiso_and_require_device_handoff(monkeypatch, handoff):
    """Only accepted host results with reconstructed device fields can advance."""
    from hybridge.linalg import pardiso_runtime
    events = []
    error = LinearSolveConvergenceError('exhausted retries')
    error.amgx_attempts = ({'label': 'primary', 'success': False},)

    def reject(**kw):
        raise error

    @contextmanager
    def thread_limit(count):
        assert count == 16
        events.append('threads-enter')
        try:
            yield 16
        finally:
            events.append('threads-exit')

    result = SimpleNamespace(field_device=object() if handoff else None,
                             trace_device=object() if handoff else None,
                             global_solve_result=SimpleNamespace(
                                 physical_residual_norm=1e-13, physical_residual_target=1e-10))

    class Host:
        def __init__(self, space, **kw):
            assert kw['solver'] == 'pypardiso'
            assert kw['assembly_backend'] == 'raw-cuda'
            assert kw['raw_local_assembly'] == 'fused'
            assert kw['raw_matrix_format'] == 'coo'
            assert kw['advection_stabilization'] == 'conflict-averaged-upwind'
            assert not kw['scale_system']
        def set_problem(self, rhs, beta, reaction, boundary):
            assert (rhs, beta, reaction, boundary) == ('rhs', 'beta', 'reaction', None)
        def solve(self, **kw):
            assert kw['initial_guess'] is None
            return result
        def close(self):
            events.append('close')

    monkeypatch.setattr(pardiso_runtime, 'pardiso_thread_limit', thread_limit)
    monkeypatch.setattr(recorder.hdg, 'AdvectionReactionHDGSolver', Host)
    transport = SimpleNamespace(solve=reject, options=SimpleNamespace(
        solver_rtol=1e-9, solver_atol=1e-10, trace_basis='legacy-lagrange'))
    metadata = {}
    if handoff:
        assert recorder.solve_positive_transport(transport, None, 'rhs', 'beta', 'reaction', None, 9, metadata) is result
    else:
        with pytest.raises(RuntimeError, match='device field and trace'):
            recorder.solve_positive_transport(transport, None, 'rhs', 'beta', 'reaction', None, 9, metadata)
    event = metadata['transport_host_recoveries'][0]
    assert event['step'] == 9 and event['mkl_threads'] == 16
    assert event['status'] == ('accepted' if handoff else 'failed')
    assert event['gpu_attempts'] == list(error.amgx_attempts)
    assert events == ['threads-enter', 'close', 'threads-exit']
