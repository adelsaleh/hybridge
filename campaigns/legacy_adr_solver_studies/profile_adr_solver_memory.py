#!/usr/bin/env python3
"""Separate allocator/whole-device peak-memory diagnostic for an ADR candidate.

Instrumentation changes timing: these samples must never be ranked. CUDA
memGetInfo samples whole-device use, including libraries and other processes;
its sampled peak is a lower bound, not an exact process allocation peak.
"""
from __future__ import annotations

# Canonical campaign sources and saved outputs have independent locations.
import sys as _campaign_sys
from pathlib import Path
_campaign_repo = Path(__file__).resolve().parents[2]
_campaign_sys.path.insert(0, str(_campaign_repo))
from campaigns.common.locations import (
    workspace_path as _campaign_workspace_path, data_directory as _campaign_data_directory,
    source_path as _campaign_source_path, source_directory as _campaign_source_directory,
)
_campaign_code = Path(__file__).resolve().parent
_campaign_sys.path.insert(0, str(_campaign_code))
_campaign_sys.path.insert(0, str(_campaign_workspace_path('hdgfem-gmres')))

import argparse
from pathlib import Path
import threading
import time
import cupy as cp
from cupy.cuda.memory_hook import MemoryHook
from scripts.adr_performance_common import atomic_json, read_json


class AllocationPeak(MemoryHook):
    """Track the high-water live bytes seen by CuPy's pool hooks."""
    name='ADRAllocationPeak'
    def __init__(self):
        self.live=0
        self.peak=0
        self.allocations={}
        self.requests=0
    def malloc_postprocess(self,**kwargs):
        key=kwargs['pmem_id'];size=kwargs['mem_size']
        if key == 0:return
        self.live-=self.allocations.get(key,0)
        self.allocations[key]=size;self.live+=size
        self.peak=max(self.peak,self.live);self.requests+=1
    def free_postprocess(self,**kwargs):
        self.live-=self.allocations.pop(kwargs['pmem_id'],0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    spec=read_json(a.spec)
    family=spec.get('family','assembly' if spec.get('task')=='assemble' else None)
    if family=='assembly':
        from scripts.adr_performance_worker import assemble as measure
        spec.update(family='assembly',candidate='shared_cupy_assembly',assembly_warmup=0,assembly_repeats=1,
            cache=str(a.output.parent/'assembly_profile_cache'))
    elif family=='native_hp':
        from scripts.adr_native_hp_worker import measure
    else:
        from scripts.adr_solver_comparison_worker import measure
    a.output.parent.mkdir(parents=True,exist_ok=True)
    spec.update(warmup=0,repeats=1,solves_per_setup=2,result=str(a.output.with_suffix('.instrumented.json')))
    cp.cuda.Device(spec['device']).use()
    cp.zeros(1).sum().item()
    baseline_free,total=cp.cuda.runtime.memGetInfo()
    samples=[];stop=threading.Event();errors=[]
    def sample():
        try:
            with cp.cuda.Device(spec['device']):
                while not stop.is_set():
                    free,capacity=cp.cuda.runtime.memGetInfo()
                    samples.append((time.monotonic(),int(capacity-free)))
                    stop.wait(.01)
        except Exception as exc:errors.append(str(exc))
    thread=threading.Thread(target=sample,daemon=True)
    hook=AllocationPeak();thread.start()
    try:
        with hook: result=measure(spec)
        atomic_json(spec['result'],result)
    finally:
        stop.set();thread.join()
    used=int(total-baseline_free)
    peak=max((s[1] for s in samples),default=used)
    report={'status':result['status'],'family':spec['family'],'candidate':spec['candidate'],
        'case':spec['case'],'n':spec['n'],'p':spec['p'],
        'cupy_tracked_peak_live_bytes':hook.peak,'cupy_allocation_requests':hook.requests,
        'whole_device_baseline_used_bytes':used,'whole_device_sampled_peak_used_bytes':peak,
        'whole_device_sampled_peak_increase_bytes':max(0,peak-used),
        'sample_count':len(samples),'sample_interval_target_seconds':.01,'sampling_errors':errors,
        'amgx_managed_memory':result.get('samples',[{}])[0].get('amgx_memory'),
        'peak_process_rss_kib':__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss,
        'note':'Instrumented timings excluded from ranking. CuPy peak is hook-tracked pool allocations only; AMGX peak is allocator-managed. Whole-device sampled peak includes library allocations and other processes, and can miss short-lived peaks.'}
    atomic_json(a.output,report)
    print(report,flush=True)
    if result['status']!='passed' or errors:raise SystemExit(1)


if __name__=='__main__':main()
