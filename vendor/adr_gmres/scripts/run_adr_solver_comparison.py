#!/usr/bin/env python3
"""Recorded, serial ASM+PP versus AMGX sweep on shared stationary ADR systems."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from scripts.adr_performance_common import Configuration, atomic_json, read_json
from scripts.run_adv_diff_rea_performance import source_hash


def candidates(config_root):
    result=[]
    for degree in (6,12,18,24):
        for operator,application in (('raw','raw'),('raw_fused','fused'),('matmul','matmul')):
            c=Configuration(operator=operator,application=application,polynomial_degree=degree)
            result.append(dict(family='asm_pp',candidate=f'asm_d{degree}_{operator}_{application}',configuration=asdict(c)))
    files=['fgmres_dilu_abs','fgmres_aggregation_dilu','fgmres_amg_d2','gmres_amg_d2',
        'bicgstab_aggregation_dilu','bicgstab_classical_l1_aggressive','bicgstab_ilu0_amg']
    for name in files:
        config=read_json(config_root/f'adv_rea_gpu4_hdg_{name}.json')
        result.append(dict(family='amgx',candidate='amgx_'+name,matrix_format='csr',amgx_config=config))
    dilu=read_json(config_root/'adv_rea_gpu4_hdg_pbicgstab_dilu_bsr_p1_p3.json')
    jacobi=read_json(config_root/'adv_rea_gpu4_hdg_pbicgstab_block_jacobi_bsr.json')
    agg=read_json(config_root/'adv_rea_gpu4_hdg_fgmres_aggregation_dilu.json')
    for outer in ('FGMRES','PBICGSTAB'):
        for name,base in (('dilu',dilu),('block_jacobi',jacobi),('aggregation_dilu',agg)):
            config=deepcopy(base);config['solver'].update(solver=outer,bsr_spmv_backend='cusparse_generic',use_scalar_norm=1)
            result.append(dict(family='amgx',candidate=f'amgx_bsr_{outer.lower()}_{name}',matrix_format='bsr',amgx_config=config))
    base=read_json(config_root/'diff_rea_gpu4_hdg_pcgf_cheb_block_jacobi_block_graph_dense_bsr.json')
    for hierarchy in ('block_graph_dense','scalar_expand'):
        for smoother in ('MULTICOLOR_DILU','BLOCK_JACOBI'):
            config=deepcopy(base);config['solver']['solver']='FGMRES'
            pre=config['solver']['preconditioner']
            pre.update(classical_bsr_hierarchy=hierarchy,presweeps=1,postsweeps=1)
            pre['smoother']=dict(solver=smoother,max_iters=1,relaxation_factor=1.)
            result.append(dict(family='amgx',candidate=f'amgx_bsr_fgmres_{hierarchy}_{smoother.lower()}',matrix_format='bsr',amgx_config=config))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config-root',type=Path,default=Path(__file__).resolve().parents[3]/'configs/amgx')
    parser.add_argument('--cases',nargs='+',default=['trigonometric','advection_dominated','anisotropic'])
    parser.add_argument('--mesh',type=int,default=32)
    parser.add_argument('--degree',type=int,default=4)
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--solves-per-setup',type=int,default=3)
    parser.add_argument('--rtol',type=float,default=1e-11)
    parser.add_argument('--maxiter',type=int,default=1000)
    parser.add_argument('--timeout',type=float,default=180)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--select-from',type=Path,help='Confirm two best candidates per family/case from an existing sweep')
    parser.add_argument('--smoke',action='store_true',help='One candidate per family for harness qualification')
    a=parser.parse_args();a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=True)
    for sub in ('jobs','specs','logs','cache'):(a.output/sub).mkdir(exist_ok=True)
    choices=candidates(a.config_root)
    if a.smoke:choices=[choices[1],choices[12],next(c for c in choices if c['candidate']=='amgx_bsr_fgmres_dilu')]
    selected=None
    if a.select_from:
        previous=read_json(a.select_from/'summary.json');selected={}
        for case in a.cases:
            selected[case]=[]
            for family in ('asm_pp','amgx'):
                ranked=sorted([r for r in previous if r.get('case')==case and r.get('family')==family and r['status']=='passed'],key=lambda r:r['fresh_setup_solve_median_ms'])
                if not ranked:raise RuntimeError(f'No passing {family} for {case}')
                selected[case].extend(next(c for c in choices if c['candidate']==r['candidate']) for r in ranked[:2])
            # Retain one fully block AMG hierarchy to test mesh-size scaling.
            block_amg=[r for r in previous if r.get('case')==case and r['status']=='passed' and 'block_graph_dense' in r.get('candidate','')]
            if block_amg:
                leader=min(block_amg,key=lambda r:r['fresh_setup_solve_median_ms'])
                choice=next(c for c in choices if c['candidate']==leader['candidate'])
                if choice not in selected[case]:selected[case].append(choice)
    manifest={'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='resume'},
        'source_sha256':source_hash(),'worker_sha256':hashlib.sha256(Path(__file__).with_name('adr_solver_comparison_worker.py').read_bytes()).hexdigest(),
        'candidates':choices,'selected':selected,
        'policy':'Same FP64 Bernstein trace operator and RHS, no scaling, zero guess every solve. AMGX x is explicitly zeroed before every solve, including the reset cost in solve time. Native setup refreshed per repeat; subsequent solves reuse setup. All timings synchronized; independent residual/PDE/reference checks excluded from timings. Failures retained, never ranked.'}
    path=a.output/'manifest.json'
    if path.exists():
        if not a.resume or read_json(path)!=manifest:parser.error('Existing or changed campaign; use --resume with identical arguments or a new output')
    atomic_json(path,manifest)
    common=dict(n=a.mesh,p=a.degree,rtol=a.rtol,maxiter=a.maxiter,device=0,engine='gpu',
        warmup=a.warmup,repeats=a.repeats,solves_per_setup=a.solves_per_setup,
        assembly_backend='cupy',assembly_warmup=1,assembly_repeats=3,reference_max_dofs=100000,
        memory_fraction=.8,host_limit_gib=None)
    records=[]
    def run(spec,module,key,timeout):
        spec['result']=str(a.output/'jobs'/f'{key}.json')
        specpath=a.output/'specs'/f'{key}.json'
        if specpath.exists() and read_json(specpath)!=spec:raise RuntimeError(f'Specification changed: {key}')
        atomic_json(specpath,spec)
        if a.resume and Path(spec['result']).exists():
            old=read_json(spec['result'])
            if old['status'] not in ('running',):return old
        with (a.output/'logs'/f'{key}.log').open('w') as log:
            try:
                proc=subprocess.run([sys.executable,'-m',module,'--spec',str(specpath)],stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
                if Path(spec['result']).exists():return read_json(spec['result'])
                value={'status':'process_error','returncode':proc.returncode}
            except subprocess.TimeoutExpired:value={'status':'timeout','timeout_seconds':timeout}
        atomic_json(spec['result'],value);return value
    for case in a.cases:
        cache=a.output/'cache'/f'{case}_n{a.mesh}_p{a.degree}'
        spec={**common,'case':case,'cache':str(cache),'task':'assemble'}
        assembled=run(spec,'scripts.adr_performance_worker','assemble_'+case,600)
        print('assemble',case,assembled['status'],flush=True)
        if assembled['status']!='passed':raise RuntimeError(f'Assembly failed for {case}')
        active=list(selected[case] if selected is not None else choices)
        random.Random(1729).shuffle(active)
        for candidate in active:
            spec={**common,**candidate,'case':case,'cache':str(cache)}
            record=run(spec,'scripts.adr_solver_comparison_worker',case+'_'+candidate['candidate'],a.timeout)
            record.update(case=case,n=a.mesh,p=a.degree,family=candidate['family'],candidate=candidate['candidate'])
            records.append(record)
            atomic_json(a.output/'summary.json',records)
            print(case,candidate['candidate'],record['status'],record.get('fresh_setup_solve_median_ms'),flush=True)
    best=[]
    for case in a.cases:
        for family in ('asm_pp','amgx'):
            good=[r for r in records if r['case']==case and r['family']==family and r['status']=='passed']
            if good:best.append(min(good,key=lambda r:r['fresh_setup_solve_median_ms']))
    atomic_json(a.output/'best.json',best)
    completion={'status':'completed' if len(best)==2*len(a.cases) else 'incomplete',
        'attempted':len(records),'passed':sum(r['status']=='passed' for r in records),
        'failures':[{'case':r['case'],'candidate':r['candidate'],'status':r['status']} for r in records if r['status']!='passed'],
        'note':'Complete coverage means every scheduled candidate was attempted and both families have a valid candidate for every case; it does not mean all candidates converged.'}
    atomic_json(a.output/'completion.json',completion)
    print(completion,flush=True)
    if completion['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
