#!/usr/bin/env python3
"""Run separate application profiles and validation after both timing sweeps finish."""
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
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from scripts.adr_performance_common import atomic_json,read_json

CASES=('cellular7_high','cellular7_weak','cellular7_directional')


def main():
    """Keep instrumented checks out of ranked timings and serialize all GPU jobs."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root',type=Path,required=True)
    args=parser.parse_args();study=args.study_root.resolve();campaign=study/'scaling/annulus'
    out=study/'diagnostics';out.mkdir(exist_ok=True)
    completion=campaign/'completion.json'
    active=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv=path.read_bytes().decode().split('\0')
            if 'campaigns.legacy_adr_solver_studies.run_oscillatory_adr_scaling' in argv and str(campaign) in argv:active.append(int(path.parent.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    if not completion.exists():assert len(active)==1,('Expected one live annular runner',active)
    print('Waiting for verified annular runner',active,flush=True)
    while not completion.exists():
        if not Path('/proc').joinpath(str(active[0])).exists():raise RuntimeError('Annular runner stopped without completion')
        time.sleep(5)
    done=read_json(completion)
    assert done['status']=='completed' and done['attempted']==done['scheduled']==96
    rows=read_json(campaign/'summary.json')
    selected=[r for r in rows if r['nominal_n']==223 and r['p']==6 and r['family']=='asm_pp']
    selected += [r for r in rows if r['nominal_n']==223 and r['p']==6 and r['family']=='native_hp']
    for case in CASES[:2]:
        passing=[r for r in rows if r['nominal_n']==223 and r['p']==6 and r['case']==case and r['family']=='amgx' and r['status']=='passed']
        if passing:selected.append(min(passing,key=lambda r:statistics.mean(s['solves'][1]['solve_ms'] for s in r['samples'])))
    scripts=('run_oscillatory_scaling_diagnostics.py','adr_performance_worker.py','adr_native_hp_worker.py',
             'profile_adr_amgx_preconditioner.py','adr_solver_comparison_worker.py',
             'check_oscillatory_adr_quadrature.py','diagnose_oscillatory_adr_matrix.py','oscillatory_adr_cases.py')
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    hashes={}
    for name in scripts:
        source=_campaign_source_path(Path('hdgfem-gmres/scripts') / name);shutil.copy2(source,snapshot/name)
        hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    manifest=dict(annular_completion=done,source_sha256=hashes,profiles=[r['job_key'] for r in selected],
        quadrature_cases=list(CASES[::2]),quadrature_nominal_n=64,p=6,
        note='Only new annular meshes: 99,984 triangles for profiles, 8,039 for quadrature and conditioning. No 22,825-triangle work. Instrumented profiles excluded from rankings. ASM profile worker uses 1e-11 for both internal and physical thresholds; ranked solves use physical 1e-10.')
    atomic_json(out/'manifest.json',manifest)
    reports=[]
    def execute(module,extra,name,timeout=900):
        with (out/(name+'.log')).open('w') as log:
            result=subprocess.run([sys.executable,'-m',module,*map(str,extra)],stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        if result.returncode:raise RuntimeError(f'{name} failed with exit code {result.returncode}; see diagnostic log')
    for row in selected:
        if row['status']!='passed':
            reports.append(dict(job_key=row['job_key'],status='skipped',reason='No accepted uninstrumented solve'))
            continue
        spec=deepcopy(read_json(campaign/'specs'/(row['job_key']+'.json')))
        name=row['case']+'_'+row['candidate'];result_path=out/(name+'.json')
        spec.update(result=str(result_path),warmup=1,repeats=1)
        specpath=out/(name+'.spec.json')
        if row['family']=='asm_pp':
            spec.update(task='profile',rtol=spec['internal_rtol'],component_warmup=3,component_repeats=10)
            module='scripts.adr_performance_worker'
        elif row['family']=='native_hp':
            spec.update(profile=True);module='scripts.adr_native_hp_worker'
        else:module='scripts.profile_adr_amgx_preconditioner'
        atomic_json(specpath,spec)
        extra=['--spec',specpath]
        if row['family']=='amgx':extra+=['--output',result_path]
        execute(module,extra,name)
        result=read_json(result_path)
        assert result['status']=='passed'
        if row['family']=='native_hp':assert result['profile_validation']['passed']
        reports.append(dict(case=row['case'],candidate=row['candidate'],family=row['family'],
                            triangles=row['triangles'],trace_dofs=row['trace_dofs'],result=result))
        atomic_json(out/'profiles.json',reports)
        print('Application profile',name,'passed',flush=True)
    quadrature=[]
    for case in CASES[::2]:
        specpath=campaign/'specs'/f'assemble_annulus_{case}_target8192_p6.json'
        folder=out/('quadrature_'+case)
        execute('scripts.check_oscillatory_adr_quadrature',['--spec',specpath,'--output',folder],'quadrature_'+case)
        result=read_json(folder/'comparison.json');assert result['status']=='passed'
        quadrature.append(result);atomic_json(out/'quadrature.json',quadrature)
        print('Quadrature',case,result['difference_over_original_error'],flush=True)
    specs=[campaign/'specs'/f'assemble_annulus_{case}_target8192_p6.json' for case in CASES]
    execute('scripts.diagnose_oscillatory_adr_matrix',['--specs',*specs,'--output',out/'matrix_diagnostics.json'],'matrix_diagnostics')
    atomic_json(out/'completion.json',dict(status='completed',profiles=len(reports),quadrature=len(quadrature),condition_estimates=3))
    print('Separate scaling diagnostics complete',flush=True)


if __name__=='__main__':main()
