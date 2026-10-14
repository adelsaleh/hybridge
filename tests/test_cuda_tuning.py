"""Tuning resource and persistence contracts independent of a particular GPU."""
import json

import numpy as np
import pytest

from hybridge.hdg.cuda import tuning
from hybridge.mixed.raw_cuda.poisson import sample_elements, shared_bytes, validate_block_size


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
    record=dict(key='valid',version=tuning.CACHE_VERSION,configuration=dict(threads=64,columns=1,retained_coupling=False))
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
