"""Tuning resource and persistence contracts independent of a particular GPU."""
import json

import numpy as np
import pytest

from hybridge.runtime import cuda_tuning as tuning
from hybridge.mixed.raw_cuda.poisson import sample_elements, shared_bytes, validate_block_size


def test_selection_says_when_a_workload_is_measured_or_found_and_then_stays_quiet(tmp_path, monkeypatch, capsys):
    from hybridge.runtime import kernel_log, optional
    monkeypatch.setattr(optional, 'require_cupy', lambda: np)
    monkeypatch.setattr(optional, 'cp', None)  # no kernel classes to replace in this test
    monkeypatch.setattr(kernel_log, '_enabled', False)
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING', 'auto')
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(tuning, 'device_signature', lambda cp: dict(maxThreadsPerBlock=1024))
    monkeypatch.setattr(tuning, 'event_seconds', lambda cp, call: (call(), .001)[1])
    def kernel(grid, block, args):
        args[1][:] = args[0]
    kernel.attributes = dict(max_threads_per_block=1024)
    arguments = dict(outputs=(1,), grid=lambda b: (1,), identity={'operation': 'log-test', 'order': 6},
                     source='log-test-v1')
    inputs = np.arange(7., dtype=float)
    tuning.clear_tuning_cache()
    tuning.launch_tuned(kernel, (inputs, np.zeros(7)), **arguments)
    assert capsys.readouterr().out == ""  # silent unless asked
    kernel_log.log_kernel_first_runs()
    tuning.clear_tuning_cache()
    tuning.launch_tuned(kernel, (inputs, np.zeros(7)), **arguments)
    output = capsys.readouterr().out
    assert "tuned on an earlier run: log-test (order=6): 256 threads" in output
    tuning.launch_tuned(kernel, (inputs, np.zeros(7)), **arguments)
    assert capsys.readouterr().out == ""  # the same process already knows it
    for path in tmp_path.iterdir():
        path.unlink()
    tuning.clear_tuning_cache()
    tuning.launch_tuned(kernel, (inputs, np.zeros(7)), **arguments)
    output = capsys.readouterr().out
    assert "first time on this machine: log-test (order=6): measured 5 candidates" in output
    assert "later runs reuse this" in output


def test_tuned_atomic_launch_keeps_production_outputs_out_of_trials(tmp_path, monkeypatch):
    from hybridge.runtime import optional
    monkeypatch.setattr(optional, 'require_cupy', lambda: np)
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING', 'auto')
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(tuning, 'device_signature', lambda cp: dict(maxThreadsPerBlock=1024))
    monkeypatch.setattr(tuning, 'event_seconds', lambda cp, call: (call(), .001)[1])
    tuning.clear_tuning_cache()
    inputs, output = np.arange(7., dtype=float), np.full(7, 5.)
    def kernel(grid, block, args):
        args[1][:] += args[0]
    kernel.attributes = dict(max_threads_per_block=1024)
    arguments = dict(outputs=(1,), grid=lambda b: (1,), identity={'operation': 'atomic-test'},
                     source='atomic-test-v1', zero_outputs=True)
    report = tuning.launch_tuned(kernel, (inputs, output), **arguments)
    np.testing.assert_array_equal(output, 5.+inputs)
    assert not report['reused']
    output.fill(5.)
    report = tuning.launch_tuned(kernel, (inputs, output), **arguments)
    np.testing.assert_array_equal(output, 5.+inputs)
    assert report['reused']


def test_tuned_launch_reads_device_properties_once_per_workload(tmp_path, monkeypatch):
    from hybridge.runtime import optional
    monkeypatch.setattr(optional, 'require_cupy', lambda: np)
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING', 'auto')
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING_CACHE_DIR', str(tmp_path))
    queries = []
    monkeypatch.setattr(tuning, 'device_signature',
                        lambda cp: queries.append(1) or dict(maxThreadsPerBlock=1024))
    monkeypatch.setattr(tuning, 'event_seconds', lambda cp, call: (call(), .001)[1])
    tuning.clear_tuning_cache()
    calls = []
    def kernel(grid, block, args):
        calls.append(block)
        args[1][:] = args[0]
    kernel.attributes = dict(max_threads_per_block=1024)
    arguments = dict(outputs=(1,), grid=lambda b: (1,), identity={'operation': 'memo-test'}, source='memo-test-v1')
    inputs = np.arange(7., dtype=float)
    tuning.launch_tuned(kernel, (inputs, np.zeros(7)), **arguments)
    before, launched = len(queries), len(calls)
    output = np.zeros(7)
    assert tuning.launch_tuned(kernel, (inputs, output), **arguments)['reused']
    np.testing.assert_array_equal(output, inputs)
    assert len(queries) == before and len(calls) == launched+1
    # A new output shape is a different workload and is calibrated separately.
    tuning.launch_tuned(kernel, (inputs[:3], np.zeros(3)), **arguments)
    assert len(queries) > before


def test_device_signature_queries_properties_once_per_device():
    calls = []
    class Runtime:
        def getDevice(self):
            return 0
        def getDeviceProperties(self, ordinal):
            calls.append(ordinal)
            return dict(name=b'stub', uuid=b'\x01', multiProcessorCount=4, maxThreadsPerBlock=1024)
        def driverGetVersion(self):
            return 13010
        def runtimeGetVersion(self):
            return 13010
        def deviceGetAttribute(self, code, ordinal):
            return 2
    class Cuda:
        runtime = Runtime()
    class Cupy:
        cuda = Cuda()
        __version__ = 'stub'
    tuning.clear_tuning_cache()
    first = tuning.device_signature(Cupy)
    assert first['name'] == 'stub' and first['uuid'] == '01' and first['fp32_fp64_ratio'] == 2
    assert tuning.device_signature(Cupy) == first and len(calls) == 1
    first['name'] = 'changed'  # A caller's copy does not reach later callers.
    assert tuning.device_signature(Cupy)['name'] == 'stub'
    tuning.clear_tuning_cache()
    tuning.device_signature(Cupy)
    assert len(calls) == 2


def test_resolved_selection_runs_once_per_key_mode_and_budget(monkeypatch):
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING', 'auto')
    tuning.clear_tuning_cache()
    calls = []
    def select(value):
        def run():
            calls.append(value)
            return value
        return run
    assert tuning.resolved_selection(('a', 1), select('first')) == ('first', False)
    assert tuning.resolved_selection(('a', 1), select('second')) == ('first', True)
    assert tuning.resolved_selection(('a', 2), select('third')) == ('third', False)
    assert calls == ['first', 'third']
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING_CANDIDATES', '3')
    assert tuning.resolved_selection(('a', 1), select('budget')) == ('budget', False)
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING', 'off')
    assert tuning.resolved_selection(('a', 1), select('off')) == ('off', False)
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING', 'refresh')
    assert tuning.resolved_selection(('a', 1), select('r1')) == ('r1', False)
    assert tuning.resolved_selection(('a', 1), select('r2')) == ('r2', False)
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING', 'auto')
    tuning.clear_tuning_cache()
    assert tuning.resolved_selection(('a', 1), select('fresh')) == ('fresh', False)
    tuning.clear_tuning_cache()


@pytest.mark.parametrize('limit',[48*1024,64*1024,99*1024,160*1024])
def test_high_order_conservative_configuration_fits_resource_tiers(limit):
    device={'sharedMemPerBlockOptin':limit,'maxThreadsPerBlock':1024,'regsPerBlock':65536}
    compact=tuning.KernelConfiguration(64,1,False)
    assert tuning.legal_configuration(compact,shared_bytes(55,10,compact),device)
    large=tuning.KernelConfiguration(256,8,True)
    actual=shared_bytes(55,10,large)
    assert tuning.legal_configuration(large,actual,device)==(actual<=limit)
    assert not tuning.legal_configuration(compact,shared_bytes(55,10,compact),device,
        {'max_threads_per_block':32})
    assert not tuning.legal_configuration(large,actual,device,
        {'max_threads_per_block':1024,'num_regs':300})


def test_cache_corruption_and_read_only_directory_are_nonfatal(tmp_path,monkeypatch):
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING_CACHE_DIR',str(tmp_path))
    tuning.clear_tuning_cache()
    (tmp_path/'broken.json').write_text('{not json')
    assert tuning._read('broken')==(None,None)
    (tmp_path/'broken.json').write_text('[]')
    assert tuning._read('broken')==(None,None)
    record=dict(key='valid',version=tuning.CACHE_VERSION,variant='valid')
    tuning._write('valid',record)
    tuning.clear_tuning_cache()
    assert tuning._read('valid')==(record,'disk')
    # A file cannot serve as the cache directory on any permission model.
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING_CACHE_DIR',str(tmp_path/'broken.json'))
    tuning._write('process-only',record)
    assert tuning._read('process-only')==(record,'process')


def test_sampling_is_bounded_and_contains_both_element_classes():
    from types import SimpleNamespace
    mesh=SimpleNamespace(num_tri=6000,loc2glob_edge=np.arange(18000).reshape(6000,3),
                         bnd_edges_inds=np.arange(0,12000,3))
    selected=sample_elements(mesh)
    assert selected.size==2048
    assert np.unique(selected).size==2048
    assert np.any(selected<4000) and np.any(selected>=4000)
    np.testing.assert_array_equal(selected,sample_elements(mesh))


def test_workload_key_changes_with_precision_source_device_and_boundary():
    baseline=dict(source='a',device='gpu0',precision='float64',boundary_nonzero=False)
    original=tuning.fingerprint(baseline)
    for key,value in dict(source='b',device='gpu1',precision='float32',boundary_nonzero=True).items():
        assert tuning.fingerprint(dict(baseline,**{key:value}))!=original


def test_incorrect_candidate_is_rejected_and_explicit_launch_skips_trials(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING_CACHE_DIR',str(tmp_path))
    monkeypatch.setattr(tuning,'device_signature',lambda cp:dict(sharedMemPerBlock=49152,maxThreadsPerBlock=1024))
    monkeypatch.setattr(tuning,'event_seconds',lambda cp,call: (call(),.001)[1])
    tuning.clear_tuning_cache()
    first,wrong=tuning.KernelConfiguration(64),tuning.KernelConfiguration(128)
    kernel=SimpleNamespace(attributes={'num_regs':16})
    kwargs=dict(cupy=np,identity={'test':'reject'},candidates=[first,wrong],shared_bytes=lambda c:1024,
        compile_kernel=lambda c:kernel,trial=lambda c,k:np.array([1. if c==first else 2.]))
    config,_,report=tuning.select_configuration(**kwargs)
    assert config==first
    assert report['candidates'][1]['rejected']=='numerical parity'
    saved = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert saved['candidates'][0]['configuration'] == report['configuration']
    assert saved['candidates'][0]['attributes']['num_regs'] == 16
    def forbidden(*args):
        raise AssertionError('explicit launch must skip tuning scratch')
    kwargs['trial']=forbidden
    config,_,report=tuning.select_configuration(**kwargs,explicit=128)
    assert config==wrong and report['source']=='explicit'


@pytest.mark.parametrize('order',range(10))
def test_auto_and_explicit_high_order_launch_validation(order):
    assert validate_block_size('auto',order)=='auto'
    assert validate_block_size(256,order)==256
    if order>6:
        with pytest.raises(ValueError,match='cooperative'):
            validate_block_size(1,order)
    else:
        assert validate_block_size(1,order)==1


def test_unrepresentative_input_selects_the_baseline_unmeasured_and_uncached(monkeypatch):
    """A singular trial input must not abort tuning; nothing measured on it is cached."""
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING', 'auto')
    written = {}
    monkeypatch.setattr(tuning, '_read', lambda key: (written.get(key), 'disk' if key in written else None))
    monkeypatch.setattr(tuning, '_write', lambda key, result: written.__setitem__(key, result))
    monkeypatch.setattr(tuning, 'device_signature', lambda cp: dict(maxThreadsPerBlock=1024))
    monkeypatch.setattr(tuning, 'event_seconds', lambda cp, call: (call(), .001)[1])
    tuning.clear_tuning_cache()
    runs = []
    singular = {'baseline': True}

    def run(name, handle):
        runs.append(name)
        return np.array([1.])

    def validate(output):
        if singular['baseline']:
            raise tuning.UnrepresentativeInput('local factorization failed')
        return True

    kwargs = dict(cupy=np, identity={'test': 'unrepresentative'}, candidates=['base', 'other'],
                  prepare=lambda name: name, run=run, validate=validate)
    name, handle, report = tuning.select_variant(**kwargs)
    assert (name, handle) == ('base', 'base') and runs == ['base']
    assert report['source'] == 'unrepresentative-input' and not report['reused']
    assert report['candidates'][0]['rejected'].startswith('unrepresentative input')
    assert not written
    # A regular input afterwards is measured and cached as usual.
    singular['baseline'] = False
    runs.clear()
    name, _, report = tuning.select_variant(**kwargs)
    assert report['source'] == 'measured' and 'other' in runs
    assert len(written) == 1
    # Raised on a later candidate only, it rejects that candidate.
    written.clear()

    def validate_other(output):
        if runs[-1] == 'other':
            raise tuning.UnrepresentativeInput('only this candidate fails')
        return True

    name, _, report = tuning.select_variant(**dict(kwargs, validate=validate_other))
    assert name == 'base' and report['source'] == 'measured'
    assert report['candidates'][1]['rejected'] == 'independent validation'
    tuning.clear_tuning_cache()


def test_resolved_selection_does_not_keep_rejected_values(monkeypatch):
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING', 'auto')
    tuning.clear_tuning_cache()
    keep = lambda value: value != 'fallback'
    assert tuning.resolved_selection(('k',), lambda: 'fallback', keep=keep) == ('fallback', False)
    assert tuning.resolved_selection(('k',), lambda: 'measured', keep=keep) == ('measured', False)
    assert tuning.resolved_selection(('k',), lambda: 'again', keep=keep) == ('measured', True)
    tuning.clear_tuning_cache()


def test_adr_tensor_selection_retries_after_an_unrepresentative_input(monkeypatch):
    """The ADR adapter keeps a measured selection for the process, never a fallback."""
    from types import SimpleNamespace
    from hybridge.mixed.raw_cuda import tensor_tuning
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING', 'auto')
    tuning.clear_tuning_cache()
    cupy = SimpleNamespace(cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda: 0)))
    sources = iter(['unrepresentative-input', 'measured', 'unexpected'])
    calls = []

    def measure(**kwargs):
        calls.append(1)
        return 'config', 'module', 'kernel', dict(source=next(sources), reused=False,
                                                  wall_seconds=1., sample_elements=2)

    monkeypatch.setattr(tensor_tuning, '_measure_tensor_execution', measure)
    class Owner:
        """A weakly referenceable space or trace-space stand-in."""

    space, trace = Owner(), Owner()
    kwargs = dict(cupy=cupy, space=space, trace_space=trace, prepared=None,
                  tensor=SimpleNamespace(counts={'constant-isotropic': 2}), args=(),
                  matrix_format='csr', workspace='workspace', block_size=64, shared_limit=49152,
                  cached_factors=True, mass_stride=0, reuse_mass=False)
    assert tensor_tuning.select_tensor_execution(**kwargs)[3]['source'] == 'unrepresentative-input'
    assert tensor_tuning.select_tensor_execution(**kwargs)[3]['source'] == 'measured'
    assert tensor_tuning.select_tensor_execution(**kwargs)[3]['source'] == 'resolved'
    assert len(calls) == 2
    tuning.clear_tuning_cache()
