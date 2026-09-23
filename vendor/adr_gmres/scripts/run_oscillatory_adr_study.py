#!/usr/bin/env python3
"""Screen oscillatory ADR cases with the existing matched GPU solver workers."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path
import random
import shutil
import subprocess
import sys
from scripts.adr_performance_common import Configuration,atomic_json,read_json
from scripts.oscillatory_adr_cases import VARIANTS,parameters
from scripts.run_adr_solver_comparison import candidates
from scripts.run_adv_diff_rea_performance import source_hash


def choices(degree=24):
    """Freeze baseline settings, including identical AMG for both outer methods."""
    source=candidates(Path(__file__).resolve().parents[3]/'configs/amgx')
    result=[dict(family='asm_pp',candidate=f'asm_d{degree}',configuration=asdict(Configuration(polynomial_degree=degree)))]
    base=next(c for c in source if c['candidate']=='amgx_bsr_fgmres_block_graph_dense_multicolor_dilu')
    for method in ('FGMRES','PBICGSTAB'):
        c=deepcopy(base);c['candidate']='amg_'+method.lower();c['amgx_config']['solver']['solver']=method
        result.append(c)
    c=deepcopy(next(c for c in source if c['candidate']=='amgx_bsr_pbicgstab_dilu'))
    c['candidate']='dilu_pbicgstab';result.append(c)
    return result


def main():
    """Run isolated serial assembly and solver jobs, retaining every failure."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',default=['advection_dominated',*VARIANTS])
    parser.add_argument('--n',type=int,default=32)
    parser.add_argument('--p',type=int,default=4)
    parser.add_argument('--mesh-path',type=Path,help='Explicit nodes/triangles NPZ; n becomes a conservative memory-planning bound')
    parser.add_argument('--pp-degree',type=int,default=24)
    parser.add_argument('--candidates',nargs='+')
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--quadrature',type=int)
    parser.add_argument('--edge-quadrature',type=int)
    parser.add_argument('--maxiter',type=int,default=1000)
    parser.add_argument('--timeout',type=float,default=180.)
    parser.add_argument('--resume',action='store_true')
    a=parser.parse_args();out=a.output.resolve()
    mesh_info=None
    if a.mesh_path:
        import numpy as np
        a.mesh_path=a.mesh_path.resolve(strict=True)
        with np.load(a.mesh_path,allow_pickle=False) as arrays:
            triangles=len(arrays['triangles'])
        a.n=int(np.ceil(np.sqrt(triangles/2)))
        mesh_info=dict(path=str(a.mesh_path),triangles=triangles,sha256=hashlib.sha256(a.mesh_path.read_bytes()).hexdigest())
    for name in ('jobs','specs','logs','cache','source_snapshot'):(out/name).mkdir(parents=True,exist_ok=True)
    active=choices(a.pp_degree)
    if a.candidates:
        active=[c for c in active if c['candidate'] in a.candidates]
        if len(active)!=len(set(a.candidates)):parser.error('Unknown candidate')
    sources=['run_oscillatory_adr_study.py','oscillatory_adr_cases.py','adv_diff_rea_cases.py',
             'adr_performance_worker.py','adr_solver_comparison_worker.py']
    hashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources}
    manifest=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='resume'},
        cases={name:parameters(name) if name in VARIANTS else 'original smooth transport reference' for name in a.cases},
        mesh=mesh_info,candidates=active,source_sha256=source_hash(),script_sha256=hashes,
        rtol=1e-10,internal_rtol=1e-11,restart=75,solves_per_setup=2,
        policy='Serial synchronized FP64 jobs; same immutable unscaled Bernstein matrix/RHS per case; explicit zero initial guess; one excluded warmup setup; mean hot time from second solve of each measured fresh setup. Failures retained. No per-case solver tuning.')
    path=out/'manifest.json'
    if path.exists() and (not a.resume or read_json(path)!=__import__('json').loads(__import__('json').dumps(manifest))):
        parser.error('Existing/changed study; --resume requires identical settings')
    atomic_json(path,manifest)
    for name in sources:shutil.copy2(Path(__file__).with_name(name),out/'source_snapshot'/name)
    def run(spec,module,key):
        """Execute one bounded worker and preserve partial results on a failure."""
        spec['result']=str(out/'jobs'/(key+'.json'))
        specpath=out/'specs'/(key+'.json')
        if specpath.exists() and read_json(specpath)!=spec:raise RuntimeError('Changed spec: '+key)
        atomic_json(specpath,spec)
        if a.resume and Path(spec['result']).exists():
            previous=read_json(spec['result'])
            if previous['status']!='running':return previous
        with (out/'logs'/(key+'.log')).open('w') as log:
            try:
                process=subprocess.run([sys.executable,'-m',module,'--spec',str(specpath)],stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
                result=read_json(spec['result']) if Path(spec['result']).exists() else dict(status='process_error',returncode=process.returncode)
                if result['status']=='running':result.update(status='process_error',returncode=process.returncode)
            except subprocess.TimeoutExpired:
                result=read_json(spec['result']) if Path(spec['result']).exists() else {}
                result.update(status='timeout',timeout_seconds=a.timeout)
        atomic_json(spec['result'],result)
        return result
    rows=[];assemblies=[]
    for case in a.cases:
        key=f'{case}_n{a.n}_p{a.p}'
        common=dict(case=case,n=a.n,p=a.p,cache=str(out/'cache'/key),engine='gpu',device=0,
            rtol=1e-10,internal_rtol=1e-11,maxiter=a.maxiter,warmup=a.warmup,repeats=a.repeats,solves_per_setup=2,
            assembly_backend='cupy',assembly_warmup=0,assembly_repeats=1,
            reference_max_dofs=100000,memory_fraction=.8,host_limit_gib=None,
            volume_quad_1d=a.quadrature,edge_quad_1d=a.edge_quadrature,
            mesh_path=str(a.mesh_path) if a.mesh_path else None)
        assembly=run({**common,'task':'assemble'},'scripts.adr_performance_worker','assemble_'+key)
        assemblies.append(dict(case=case,n=a.n,p=a.p,**assembly));atomic_json(out/'assemblies.json',assemblies)
        print('assembly',key,assembly['status'],assembly.get('validation'),flush=True)
        if assembly['status']!='passed':raise RuntimeError('Assembly validation failed: '+key)
        order=deepcopy(active);random.Random(2108).shuffle(order)
        for choice in order:
            r=run({**common,**choice},'scripts.adr_solver_comparison_worker',key+'_'+choice['candidate'])
            r.update(case=case,n=a.n,p=a.p,candidate=choice['candidate'],family=choice['family'])
            rows.append(r);atomic_json(out/'summary.json',rows)
            hot=[s for trial in r.get('samples',[]) for s in trial['solves'][1:]]
            partial=[s for trial in r.get('samples',[])+r.get('warmups',[]) for s in trial['solves']]
            print(case,choice['candidate'],r['status'],
                'mean_hot_ms',sum(s['solve_ms'] for s in hot)/len(hot) if hot else None,
                'iterations',sorted({s['iterations'] for s in hot or partial}),
                'residual',max((s['true_relative_residual'] for s in partial),default=None),flush=True)
    completion=dict(status='completed',attempted=len(rows),scheduled=len(a.cases)*len(active),passed=sum(r['status']=='passed' for r in rows),
        failures=[dict(case=r['case'],candidate=r['candidate'],status=r['status']) for r in rows if r['status']!='passed'])
    atomic_json(out/'completion.json',completion);print(completion,flush=True)


if __name__=='__main__':main()
