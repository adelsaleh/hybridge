#!/usr/bin/env python3
"""Serial, separate memory and component diagnostics for confirmed ADR winners."""
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
import subprocess
import sys
from scripts.adr_performance_common import atomic_json,read_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison',type=Path,required=True)
    p.add_argument('--assembly-source',type=Path,required=True)
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    jobs=[]
    def run(module,spec,key,extra=None):
        specpath=a.output/(key+'.spec.json');atomic_json(specpath,spec)
        args=[sys.executable,'-m',module,'--spec',str(specpath)]
        if extra:args+=extra
        with (a.output/(key+'.log')).open('w') as log:
            try:
                completed=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
                result={'job':key,'returncode':completed.returncode}
            except subprocess.TimeoutExpired:result={'job':key,'timeout':1200}
        jobs.append(result);atomic_json(a.output/'completion.json',jobs);print(result,flush=True)
    chosen=read_json(a.comparison/'best.json')
    all_rows=read_json(a.comparison/'summary.json')
    for case in ('trigonometric','advection_dominated','anisotropic'):
        good=[r for r in all_rows if r['case']==case and r['family']=='amgx' and r['status']=='passed']
        fastest=min(good,key=lambda r:r['hot_solve_median_ms'])
        if not any(r['case']==case and r['candidate']==fastest['candidate'] for r in chosen):chosen.append(fastest)
    for row in chosen:
        key=row['case']+'_'+row['candidate']
        spec=read_json(a.comparison/'specs'/(key+'.json'))
        run('campaigns.legacy_adr_solver_studies.profile_adr_solver_memory',spec,key+'_memory',['--output',str(a.output/(key+'_memory.json'))])
        if row['family']=='asm_pp':
            spec.update(task='profile',component_warmup=3,component_repeats=10,
                rtol=spec.get('internal_rtol',spec['rtol']),result=str(a.output/(key+'_components.json')))
            # The canonical component worker uses one rtol for its solver and check.
            # Final uninstrumented comparisons retain the common physical contract.
            run('scripts.adr_performance_worker',spec,key+'_components')
        else:
            run('scripts.profile_adr_amgx_preconditioner',spec,key+'_components',['--output',str(a.output/(key+'_components.json'))])
    tolerance=read_json(a.comparison/'specs/advection_dominated_asm_d24_raw_fused_fused.json')
    tolerance.update(internal_rtol=1e-11,candidate='asm_d24_raw_fused_fused_internal1e-11',result=str(a.output/'transport_tolerance_confirmation.json'))
    run('scripts.adr_solver_comparison_worker',tolerance,'transport_tolerance_confirmation')
    tolerance['result']=str(a.output/'transport_tolerance_cycles.json')
    run('campaigns.legacy_adr_solver_studies.diagnose_adr_gmres_tolerance',tolerance,'transport_tolerance_cycles')
    native=read_json(a.native/'standard_spec.json')
    run('campaigns.legacy_adr_solver_studies.profile_adr_solver_memory',native,'native_hp_memory',['--output',str(a.output/'native_hp_memory.json')])
    native.update(profile=True,warmup=1,repeats=1,solves_per_setup=2,result=str(a.output/'native_hp_components.json'))
    run('scripts.adr_native_hp_worker',native,'native_hp_components')
    for case in ('trigonometric','advection_dominated','anisotropic'):
        spec=read_json(a.assembly_source/'specs'/('assemble_'+case+'.json'))
        directory=a.output/(case+'_assembly');directory.mkdir()
        run('campaigns.legacy_adr_solver_studies.profile_adr_solver_memory',spec,case+'_assembly_memory',['--output',str(directory/'memory.json')])
    if any(r.get('returncode')!=0 for r in jobs):raise SystemExit(1)


if __name__=='__main__':main()
