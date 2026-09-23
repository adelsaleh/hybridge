#!/usr/bin/env python3
"""Diagnostic AMGX preconditioner timings from existing native CUDA-event timers.

Only the immediate preconditioner enables timing/verbosity. The ordinary worker
still controls the matrix, outer convergence, zero guess and physical checks.
Instrumentation synchronizes each application; exclude these solves from ranking.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import statistics
import pyamgx
from scripts.adr_performance_common import atomic_json,read_json
from scripts.adr_solver_comparison_worker import measure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();spec=read_json(args.spec)
    if spec['family']!='amgx':parser.error('Require an AMGX candidate')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    spec.update(warmup=0,repeats=1,solves_per_setup=2,result=str(args.output.with_suffix('.instrumented.json')))
    pre=spec['amgx_config']['solver']['preconditioner']
    if not isinstance(pre,dict):parser.error('Require a configured immediate preconditioner')
    pre.update(obtain_timings=1,verbosity_level=3,print_solve_stats=0,print_grid_stats=0)
    messages=[]
    original=pyamgx.register_print_callback
    pyamgx.register_print_callback=lambda ignored:original(messages.append)
    try:result=measure(spec)
    finally:pyamgx.register_print_callback=original
    atomic_json(spec['result'],result)
    log=''.join(m.decode() if isinstance(m,bytes) else m for m in messages)
    args.output.with_suffix('.native.log').write_text(log)
    times=[1000*float(t) for t in re.findall(r'^\s+solve: ([\d.eE+-]+) s$',log,re.MULTILINE)]
    solves=[s for row in result.get('samples',[]) for s in row['solves']]
    report={'status':result['status'],'case':spec['case'],'candidate':spec['candidate'],
        'native_event_preconditioner_calls':len(times),'solve_count':len(solves),
        'preconditioner_call_mean_ms':statistics.mean(times) if times else None,
        'preconditioner_call_median_ms':statistics.median(times) if times else None,
        'preconditioner_total_ms':sum(times),'iterations':[s['iterations'] for s in solves],
        'note':'Immediate native preconditioner only; CUDA-event times with synchronization per call. Includes both instrumented zero-guess solves and their first application. No ranking uses these times. Callback records native application count, not internal level SpMV counts.'}
    atomic_json(args.output,report);print(report,flush=True)
    if result['status']!='passed' or not times:raise SystemExit(1)


if __name__=='__main__':main()
