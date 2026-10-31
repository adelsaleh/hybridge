#!/usr/bin/env python3
"""Bounded large-mesh tuning using existing immutable ADR matrix caches."""
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
from dataclasses import asdict
import hashlib
from pathlib import Path
import random
import subprocess
import sys
from scripts.adr_performance_common import Configuration,atomic_json,read_json
from campaigns.legacy_adr_solver_studies.run_adr_solver_comparison import candidates


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--confirm-from',type=Path)
    p.add_argument('--timeout',type=float,default=1200)
    p.add_argument('--rtol',type=float,default=1e-11)
    p.add_argument('--internal-rtol',type=float)
    p.add_argument('--allow-near-threshold',action='store_true',help='Select strict-test near misses for a new, explicitly different common residual contract; never relabel old results')
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(exist_ok=False,parents=True)
    for name in ('specs','jobs','logs'):(a.output/name).mkdir()
    cases=['trigonometric','advection_dominated','anisotropic']
    options=candidates(Path(str(_campaign_workspace_path('hdgfem/configs/amgx'))))
    names=['amgx_bsr_fgmres_block_graph_dense_multicolor_dilu','amgx_bsr_fgmres_block_graph_dense_block_jacobi',
        'amgx_bsr_fgmres_scalar_expand_multicolor_dilu','amgx_bsr_fgmres_scalar_expand_block_jacobi',
        'amgx_fgmres_amg_d2','amgx_bicgstab_classical_l1_aggressive','amgx_bsr_pbicgstab_dilu']
    options=[c for c in options if c['candidate'] in names]
    options += [dict(family='asm_pp',candidate=f'asm_d{d}_raw_fused_fused',
        configuration=asdict(Configuration(polynomial_degree=d))) for d in (12,24,36,48)]
    previous=read_json(a.confirm_from/'summary.json') if a.confirm_from else None
    manifest=dict(source=str(a.source.resolve()),confirm_from=str(a.confirm_from) if a.confirm_from else None,
        candidates=options,worker_sha256=hashlib.sha256(_campaign_source_path(Path('hdgfem-gmres/scripts') / 'adr_solver_comparison_worker.py').read_bytes()).hexdigest(),
        physical_rtol=a.rtol,internal_rtol=a.internal_rtol,allow_near_threshold=a.allow_near_threshold,
        policy='Common physical residual target recorded separately from internal tolerance. FP64, zero guess; matrix unchanged. Qualification: one setup/two solves. Confirmation: warmup1/repeats3/three solves per setup. Near-threshold selection, when requested, only chooses candidates to rerun under the new contract; it never changes old pass/fail status. No instrumented timings ranked.')
    atomic_json(a.output/'manifest.json',manifest)
    results=[]
    for case in cases:
        base=read_json(a.source/'specs'/f'assemble_{case}.json');base.pop('task',None)
        base.update(maxiter=1000,rtol=a.rtol,amgx_internal_rtol=1e-12,
            warmup=1 if previous else 0,repeats=3 if previous else 1,solves_per_setup=3 if previous else 2)
        if a.internal_rtol is not None:base['internal_rtol']=a.internal_rtol
        choices=options
        if previous:
            choices=[]
            for family in ('asm_pp','amgx'):
                def eligible(r):
                    if r['status']=='passed':return True
                    solves=[s for row in r.get('samples',[]) for s in row['solves']]
                    return bool(a.allow_near_threshold and r['status']=='numerical_failure' and solves and
                        all(s['status'] in ('success','converged') and s['true_relative_residual']<=a.rtol for s in solves))
                def metric(r):
                    return r.get('fresh_setup_solve_median_ms',min((row['fresh_setup_solve_ms'] for row in r.get('samples',[])),default=float('inf')))
                good=sorted((r for r in previous if r['case']==case and r['family']==family and eligible(r)),key=metric)
                choices += [next(c for c in options if c['candidate']==r['candidate']) for r in good[:2]]
        choices=list(choices);random.Random(811).shuffle(choices)
        for c in choices:
            key=case+'_'+c['candidate'];spec={**base,**c,'result':str(a.output/'jobs'/f'{key}.json')}
            path=a.output/'specs'/f'{key}.json';atomic_json(path,spec)
            with (a.output/'logs'/f'{key}.log').open('w') as log:
                try:
                    proc=subprocess.run([sys.executable,'-m','scripts.adr_solver_comparison_worker','--spec',str(path)],stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
                    record=read_json(spec['result']) if Path(spec['result']).exists() else {'status':'process_error','returncode':proc.returncode}
                except subprocess.TimeoutExpired:record={'status':'timeout','timeout_seconds':a.timeout}
            record.update(case=case,candidate=c['candidate'],family=c['family']);atomic_json(spec['result'],record)
            results.append(record);atomic_json(a.output/'summary.json',results)
            print(case,c['candidate'],record['status'],record.get('hot_solve_median_ms'),flush=True)
    best=[]
    for case in cases:
        for family in ('asm_pp','amgx'):
            good=[r for r in results if r['case']==case and r['family']==family and r['status']=='passed']
            if good:best.append(min(good,key=lambda r:r['fresh_setup_solve_median_ms']))
    atomic_json(a.output/'best.json',best)
    completion=dict(status='completed' if len(best)==6 else 'incomplete',attempted=len(results),passed=sum(r['status']=='passed' for r in results),
        failures=[{k:r[k] for k in ('case','candidate','status')} for r in results if r['status']!='passed'])
    atomic_json(a.output/'completion.json',completion);print(completion,flush=True)
    if completion['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
