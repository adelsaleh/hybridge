"""adr_unified_worker.run_numerical up to its dispatch, for every stage, with a stubbed CuPy.

The worker only ever ran on the GPU, so its pre-dispatch logic (path setup, operator identity,
memory guards, device-campaign switches) was tested by lane smokes alone. This drives it on the
CPU: the numerical measures are replaced by stubs and ``cupy`` by a namespace that answers the
few runtime calls the guard makes.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


class _Device:
    pci_bus_id = '0000:00:00.0'

    def __init__(self, *_):
        pass

    def use(self):
        pass

    def synchronize(self):
        pass


def _cupy_stub():
    runtime = types.SimpleNamespace(memGetInfo=lambda: (120 << 30, 140 << 30))
    pool = types.SimpleNamespace(free_all_blocks=lambda: None, used_bytes=lambda: 0, total_bytes=lambda: 0)
    return types.SimpleNamespace(cuda=types.SimpleNamespace(Device=_Device, runtime=runtime),
                                 get_default_memory_pool=lambda: pool)


@pytest.fixture
def worker(monkeypatch, tmp_path):
    # The worker puts the vendored ``scripts`` package first on sys.path and imports from it;
    # restore master's ``scripts`` modules afterwards so later tests can import, for example,
    # ``scripts.guiding_center``.
    saved_modules = {k: v for k, v in sys.modules.items() if k == 'scripts' or k.startswith('scripts.')}
    saved_path = list(sys.path)
    monkeypatch.setitem(sys.modules, 'cupy', _cupy_stub())
    for key in ('HYBRIDGE_ADR_GPU_SAMPLING', 'HYBRIDGE_ADR_SINGLE_GUARD'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('HYBRIDGE_ADR_DEVICE_CHECKS', '1')
    monkeypatch.setenv('HYBRIDGE_ADR_CACHE_IDENTITY', 'metadata')
    from campaigns.adr_unified import adr_unified_worker as module
    common = module.load_file('_test_common', ROOT/'vendor/adr_gmres/scripts/adr_performance_common.py')
    cache = tmp_path/'cache'
    cache.mkdir()
    np.save(cache/'system_blocks.npy', np.ones((4, 5, 2, 2)))
    np.save(cache/'system_rhs.npy', np.ones((4, 2)))
    estimate = dict(assembly_device_bytes=1 << 20, assembly_host_bytes=1 << 20, solver_device_bytes=1 << 20,
                    solver_host_bytes=1 << 20)
    spec = dict(branch_root=str(ROOT/'vendor/adr_gmres'), case='trigonometric', n=4, p=2, triangles=32,
                cache=str(cache), result=str(tmp_path/'result.json'), memory_estimate=estimate,
                memory_fraction=.9, reserve_gpu_gib=1., assembly_backend='cupy', host_limit_gib=None)
    try:
        yield module, common, spec
    finally:
        for key in [k for k in sys.modules if k == 'scripts' or k.startswith('scripts.')]:
            del sys.modules[key]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


def test_system_stage_dispatches_to_the_system_worker(worker, monkeypatch):
    module, common, spec = worker
    import campaigns.adr_h200_tuned.system_worker as system_worker
    seen = {}
    monkeypatch.setattr(system_worker, 'run', lambda s, c, root, log=print: seen.update(spec=s) or dict(status='passed'))
    spec.update(stage='system', candidates=[dict(candidate='a', family='asm_pp', memory_estimate=spec['memory_estimate'])],
                expected_operator_sha256='ignored for the system stage')
    result = module.run_numerical(spec, common)
    assert result['status'] == 'passed' and 'sampled_device_peak_bytes' in result
    assert seen['spec']['stage'] == 'system'


def test_solve_stage_checks_cache_identity_by_metadata(worker, monkeypatch):
    module, common, spec = worker
    import campaigns.adr_h200_tuned.device_measure as device_measure
    monkeypatch.setattr(device_measure, 'measure', lambda s, log=print, context=None: dict(status='passed', seen=s['operator_identity']))
    files = {p.name: dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in sorted(Path(spec['cache']).glob('*.npy'))}
    spec.update(stage='solve', family='asm_pp', expected_operator_sha256='abc', expected_cache_files=files)
    result = module.run_numerical(spec, common)
    assert result['status'] == 'passed' and result['seen'] == 'cache_metadata'
    files['system_rhs.npy'] = dict(bytes=1, mtime_ns=1)
    with pytest.raises(ValueError, match='changed after assembly'):
        module.run_numerical(dict(spec, expected_cache_files=files), common)


def test_device_budget_refuses_oversized_jobs(worker):
    module, common, spec = worker
    spec.update(stage='solve', family='asm_pp')
    spec['memory_estimate'] = dict(spec['memory_estimate'], solver_device_bytes=200 << 30)
    with pytest.raises(module.BudgetExceeded, match='Device estimate'):
        module.run_numerical(spec, common)
    big = dict(spec, stage='system', candidates=[dict(candidate='a', family='asm_pp', memory_estimate=spec['memory_estimate'])])
    with pytest.raises(module.BudgetExceeded):
        module.run_numerical(big, common)
