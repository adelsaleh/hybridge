#!/usr/bin/env python3
"""ADR GPU performance campaign with explicit path coverage and million-DOF cases.

full: all implementation paths -> parameter tuning of selected paths -> every
path at the large size -> independent CUDA profiles and end-to-end confirmation.
exhaustive: full Cartesian parameter grid on screening AND large problems.
CPU checks are labelled as such and exercise orchestration without CUDA claims.
"""
from __future__ import annotations
from dataclasses import asdict,replace
from pathlib import Path
from itertools import product
import argparse
import json
import os
import random
import subprocess
import sys
from time import perf_counter
from scripts.adr_performance_common import (Configuration,FAMILIES,OPERATORS,INVERSES,ORTHOGONALIZATIONS,
    fingerprint,path_configurations,parameter_grid,trace_dofs,mesh_for_target,estimates,atomic_json,
    read_json,eligible,best_by,write_csv)
from scripts.adv_diff_rea_cases import CASES

ROOT=Path(__file__).resolve().parents[1]


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--level',choices=('smoke','full','exhaustive'),default='full')
    p.add_argument('--output',type=Path,default=Path('results/adr_performance'))
    p.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    p.add_argument('--orders',nargs='+',type=int,default=[3,4])
    p.add_argument('--screen-mesh',type=int,default=16)
    p.add_argument('--target-dofs',type=int,default=1_000_000)
    p.add_argument('--preconditioners',nargs='+',choices=FAMILIES,default=list(FAMILIES))
    p.add_argument('--operators',nargs='+',choices=OPERATORS,default=list(OPERATORS))
    p.add_argument('--local-solvers',nargs='+',choices=INVERSES,default=list(INVERSES))
    p.add_argument('--degrees',nargs='+',type=int,default=[6,12,18,24])
    p.add_argument('--restarts',nargs='+',type=int,default=[50,75,100])
    p.add_argument('--orthogonalizations',nargs='+',choices=ORTHOGONALIZATIONS,default=list(ORTHOGONALIZATIONS))
    p.add_argument('--polynomial-orthogonalizations',nargs='+',choices=ORTHOGONALIZATIONS,default=list(ORTHOGONALIZATIONS))
    p.add_argument('--tuning-paths-per-family',type=int,default=1)
    p.add_argument('--large-scope',choices=('paths','finalists'),default='paths',
                   help='paths remeasures every implementation path, finalists only successful tuned family winners')
    p.add_argument('--rtol',type=float,default=1e-12)
    p.add_argument('--maxiter',type=int,default=5000)
    p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--assembly-warmup',type=int,default=1)
    p.add_argument('--assembly-repeats',type=int,default=3)
    p.add_argument('--component-warmup',type=int,default=5)
    p.add_argument('--component-repeats',type=int,default=50)
    p.add_argument('--profile-scope',choices=('paths','families','none'),default='paths')
    p.add_argument('--assembly-backend',choices=('numpy','cupy'),default='cupy')
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--timeout',type=float,default=1800,help='Maximum seconds per worker, including all repetitions')
    p.add_argument('--assembly-timeout',type=float,default=3600)
    p.add_argument('--reference-max-dofs',type=int,default=20000)
    p.add_argument('--memory-fraction',type=float,default=.8)
    p.add_argument('--host-limit-gib',type=float)
    p.add_argument('--seed',type=int,default=1729)
    p.add_argument('--cpu-check',action='store_true',help='Small CPU-only orchestration validation, never GPU performance')
    p.add_argument('--dry-run',action='store_true',help='Write full plan, resource estimates and grid without a GPU')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--retry-failed',action='store_true')
    return p


def validate_args(a,p):
    if (min(a.orders)<1 or a.screen_mesh<2 or a.target_dofs<1 or min(a.degrees)<1 or min(a.restarts)<1
        or a.repeats<1 or a.assembly_repeats<1 or a.component_repeats<1 or a.maxiter<1
        or min(a.warmup,a.assembly_warmup,a.component_warmup)<0 or a.timeout<=0 or a.assembly_timeout<=0
        or not 0<a.memory_fraction<1 or not 0<a.rtol<1 or a.tuning_paths_per_family<1
        or (a.host_limit_gib is not None and a.host_limit_gib<=0)):
        p.error('invalid size, tolerance, iteration count, repeats, timeout or memory budget')
    if a.cpu_check and a.level!='smoke':p.error('--cpu-check requires --level smoke')
    if a.reference_max_dofs<120:p.error('reference-max-dofs must allow the small independent CPU validation')
    if a.retry_failed and not a.resume:p.error('--retry-failed requires --resume')


def source_hash():
    import hashlib
    h=hashlib.sha256()
    files=list((ROOT/'hdgfem').rglob('*.py'))+[Path(__file__),ROOT/'scripts/adr_performance_worker.py',
        ROOT/'scripts/adr_performance_common.py',ROOT/'scripts/adv_diff_rea_cases.py']
    for path in sorted(set(files)):
        h.update(str(path.relative_to(ROOT)).encode());h.update(path.read_bytes())
    return h.hexdigest()


def base_paths(a):
    if a.cpu_check:
        return [Configuration('none','cpu_face_dense',None,None,None,'scipy_gmres',30,None)]
    if a.level=='smoke':
        paths=path_configurations(a.preconditioners,('raw_fused',),('cublas_inverse',),degree=6,restart=30)
        # One application per family for the smoke gate.
        return list({c.preconditioner:c for c in paths}.values())
    return path_configurations(a.preconditioners,a.operators,a.local_solvers,
        degree=a.degrees[len(a.degrees)//2],restart=a.restarts[len(a.restarts)//2],
        orthogonalization='cgs2')


def grid(a,paths):
    return parameter_grid(paths,degrees=a.degrees,restarts=a.restarts,
        orthogonalizations=a.orthogonalizations,polynomial_orthogonalizations=a.polynomial_orthogonalizations)


def plan(a):
    paths=base_paths(a)
    problems=[]
    if a.level!='smoke':
        for case,p in product(a.cases,sorted(set(a.orders))):
            large=mesh_for_target(p,a.target_dofs)
            problems.append(dict(case=case,p=p,screen_n=a.screen_mesh,large_n=large,
                screening_estimate=estimates(a.screen_mesh,p,max(a.restarts)),
                large_estimate=estimates(large,p,max(a.restarts))))
    cartesian=len(grid(a,paths)) if a.level=='exhaustive' else None
    # Parameter grid ceiling if each family contributes the requested number of paths.
    upper=a.tuning_paths_per_family*sum(len(grid(a,[next(c for c in paths if c.preconditioner==f)]))
        for f in sorted({c.preconditioner for c in paths})) if a.level=='full' else 0
    return dict(level=a.level,engine='cpu_reference_check' if a.cpu_check else 'cuda',
        source_sha256=source_hash(),paths=[asdict(c) for c in paths],path_count=len(paths),
        parameter_grid=dict(degrees=a.degrees,restarts=a.restarts,orthogonalizations=a.orthogonalizations,
                            polynomial_orthogonalizations=a.polynomial_orthogonalizations),
        exhaustive_configurations_per_problem=cartesian,
        tuning_upper_bound_per_screen_problem=upper,large_scope=a.large_scope,
        profile_scope=a.profile_scope,problems=problems,
        validation=[dict(case=c,n=n,p=2,trace_dofs=trace_dofs(n,2)) for c,n in product(a.cases,(4,8))],
        notes=['Full is a staged search; it does not claim the global optimum of the entire Cartesian grid.',
               'Every legal local-solver/application combination is explicit; gpu_solve only supports matmul.',
               'Resource estimates are conservative planning values, not measured peaks or guarantees.',
               'Failed, timed-out and budget-skipped candidates remain in coverage and never rank.',
               'CGS fallback is disabled to compare actual orthogonalization choices.'])


class Campaign:
    def __init__(self,a):
        self.a=a;self.out=a.output.resolve();self.out.mkdir(parents=True,exist_ok=True)
        self.records=[];self.issues=[];self.coverage=[];self.stage_records=[]
        self.rng=random.Random(a.seed)
        for d in ('specs','jobs','logs','cache'):(self.out/d).mkdir(exist_ok=True)
        self.common={key:getattr(a,key) for key in ('rtol','maxiter','warmup','repeats','assembly_warmup',
            'assembly_repeats','component_warmup','component_repeats','device','assembly_backend',
            'reference_max_dofs','memory_fraction','host_limit_gib')}
        self.common['engine']='cpu' if a.cpu_check else 'gpu'
        if a.cpu_check:self.common['assembly_backend']='numpy'

    def job(self,task,case,n,p,config=None,*,l2_bound=None):
        key=f'{task}_{case}_n{n}_p{p}'+('_'+config.key if config else '')
        output=self.out/'jobs'/(key+'.json');specfile=self.out/'specs'/(key+'.json')
        spec={**self.common,'task':task,'case':case,'n':n,'p':p,
            'cache':str(self.out/'cache'/f'{case}_n{n}_p{p}'),'result':str(output)}
        if config:spec['configuration']=asdict(config)
        if l2_bound is not None:spec['l2_bound']=l2_bound
        oldspec=read_json(specfile) if specfile.exists() else None
        if oldspec is not None and oldspec!=spec:
            raise RuntimeError(f'job specification changed: {key}; use a new output directory')
        atomic_json(specfile,spec)
        old=read_json(output) if output.exists() else None
        if old is not None and task=='assemble' and old.get('status')=='passed':
            cache=Path(spec['cache'])
            if not (cache/'metadata.json').exists() or any(
                    not (cache/name).exists() or (cache/name).stat().st_size!=size
                    for name,size in old.get('arrays',{}).items()):
                old=None
        if old is not None and old.get('status')!='running' and not(self.a.retry_failed and old.get('status')!='passed'):
            result=old
        else:
            if output.exists():output.unlink()
            started=perf_counter()
            with (self.out/'logs'/(key+'.log')).open('w') as log:
                try:
                    done=subprocess.run([sys.executable,'-m','scripts.adr_performance_worker','--spec',str(specfile)],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                        timeout=self.a.assembly_timeout if task in ('assemble','pipeline') else self.a.timeout)
                    result=read_json(output) if output.exists() else dict(status='worker_crash',returncode=done.returncode)
                    if done.returncode!=0 and result.get('status')=='passed':
                        result.update(status='worker_crash',returncode=done.returncode)
                except subprocess.TimeoutExpired:
                    result=read_json(output) if output.exists() else {}
                    result.update(status='timeout',error='Worker exceeded wall-time limit; partial samples retained')
            result['worker_wall_seconds']=perf_counter()-started
            atomic_json(output,result)
        row={**result,'job':key,'case':case,'n':n,'p':p,'trace_dofs':trace_dofs(n,p),'task':task}
        if config:row['configuration']=asdict(config)
        self.stage_records.append(row)
        if task=='measure':
            # Reusing a baseline during tuning must not duplicate a ranking row.
            self.records=[r for r in self.records if r['job']!=key]+[row]
        print(f"{key}: {row['status']}"+(f" {row['setup_solve_median_ms']:.3f} ms" if eligible(row) else ''),flush=True)
        with (self.out/'progress.jsonl').open('a') as f:
            f.write(json.dumps({'job':key,'status':row['status']})+'\n')
        return row

    def ensure(self,case,n,p):
        r=self.job('assemble',case,n,p)
        if r['status']!='passed':self.issues.append(dict(stage='assembly',case=case,n=n,p=p,status=r['status']))
        return r if r['status']=='passed' else None

    def measured(self,case,n,p,configs,*,bound=None,stage):
        configs=list(dict.fromkeys(configs));self.rng.shuffle(configs)
        rows=[self.job('measure',case,n,p,c,l2_bound=bound) for c in configs]
        passed=[r for r in rows if eligible(r)]
        families=sorted({c.preconditioner for c in configs})
        execution_issues=[r for r in rows if r['status'] not in ('passed','numerical_failure')]
        for r in execution_issues:
            self.issues.append(dict(stage=stage,job=r['job'],status=r['status']))
        missing=[f for f in families if not any(r['configuration']['preconditioner']==f for r in passed)]
        self.coverage.append(dict(stage=stage,case=case,n=n,p=p,requested=len(configs),
            passed=len(passed),failed=len(rows)-len(passed),execution_issues=len(execution_issues),missing_families=missing,
            paths_requested=len({c.path for c in configs}),
            paths_passed=len({Configuration(**r['configuration']).path for r in passed})))
        if not passed:self.issues.append(dict(stage=stage,case=case,n=n,p=p,status='no_valid_configuration'))
        self.report()
        return rows

    def diagnostics(self,case,n,p,rows,bound):
        if self.a.profile_scope!='none':
            group=(lambda r:Configuration(**r['configuration']).path) if self.a.profile_scope=='paths' else (lambda r:r['configuration']['preconditioner'])
            for r in best_by(rows,group):
                d=self.job('profile',case,n,p,Configuration(**r['configuration']),l2_bound=bound)
                if d['status']!='passed':self.issues.append(dict(stage='profile',job=d['job'],status=d['status']))
        for r in best_by(rows,lambda r:r['configuration']['preconditioner']):
            d=self.job('pipeline',case,n,p,Configuration(**r['configuration']),l2_bound=bound)
            if d['status']!='passed':self.issues.append(dict(stage='pipeline',job=d['job'],status=d['status']))

    def report(self):
        winners=best_by(self.records,lambda r:(r['case'],r['n'],r['p']))
        family=best_by(self.records,lambda r:(r['case'],r['n'],r['p'],r['configuration']['preconditioner']))
        path=best_by(self.records,lambda r:(r['case'],r['n'],r['p'],Configuration(**r['configuration']).path))
        for name,rows in [('candidates',self.records),('best_overall',winners),('best_per_family',family),('best_per_path',path),
                          ('end_to_end',[r for r in self.stage_records if r['task']=='pipeline']),
                          ('profiles',[r for r in self.stage_records if r['task']=='profile']),
                          ('failures',[r for r in self.stage_records if r['status']!='passed'])]:
            write_csv(self.out/(name+'.csv'),rows)
        atomic_json(self.out/'coverage.json',self.coverage)
        atomic_json(self.out/'issues.json',self.issues)
        with (self.out/'report.md').open('w') as f:
            f.write('# ADR performance campaign\n\n')
            f.write('Engine: **'+('CPU orchestration check (no CUDA performance)' if self.a.cpu_check else 'CUDA')+'**.\n\n')
            f.write('Primary rank: median fresh setup + uninstrumented solve, only fully passing repeats.\n'
                    'Assembly is shared for solver comparison. Actual end-to-end samples are in end_to_end.csv.\n\n')
            f.write('| Case | n | p | Dofs | Best family | Setup + solve (ms) |\n|---|---:|---:|---:|---|---:|\n')
            for r in sorted(winners,key=lambda r:(r['case'],r['p'],r['n'])):
                f.write(f"| {r['case']} | {r['n']} | {r['p']} | {r['trace_dofs']} | {r['configuration']['preconditioner']} | {r['setup_solve_median_ms']:.4f} |\n")
            f.write(f'\nRecorded issues: {len(self.issues)}. See coverage.json for failed paths and missing families.\n'
                    'A winning configuration does not establish convergence of other configurations.\n'
                    'CUDA event details and GMRES attribution are in the profile job JSON files.\n')

    def run(self):
        paths=base_paths(self.a)
        canonical=paths[0] if self.a.cpu_check else Configuration(polynomial_degree=6,restart=30)
        # Independent small CPU-reference gate for every manufactured case.
        validation_ok=True
        for case,n in product(self.a.cases,(4,8)):
            if self.ensure(case,n,2) is None:
                validation_ok=False;continue
            selected=paths if self.a.level=='smoke' else [canonical]
            rows=self.measured(case,n,2,selected,stage='validation')
            if not any(eligible(r) for r in rows):validation_ok=False
            if self.a.level=='smoke':self.diagnostics(case,n,2,rows,None)
        if self.a.level=='smoke' or not validation_ok:
            if not validation_ok:self.issues.append({'stage':'gate','status':'validation_failed_large_not_started'})
            self.report();return validation_ok and not self.issues
        for case,p in product(self.a.cases,sorted(set(self.a.orders))):
            n=self.a.screen_mesh
            assembly=self.ensure(case,n,p)
            if assembly is None:continue
            configs=grid(self.a,paths) if self.a.level=='exhaustive' else paths
            screen=self.measured(case,n,p,configs,stage='screen')
            if self.a.level=='full':
                # Tune top K distinct implementation paths per family.
                leaders=best_by(screen,lambda r:Configuration(**r['configuration']).path)
                seeds=[]
                for family in self.a.preconditioners:
                    rank=sorted([r for r in leaders if r['configuration']['preconditioner']==family],key=lambda r:r['setup_solve_median_ms'])
                    seeds.extend(Configuration(**r['configuration']) for r in rank[:self.a.tuning_paths_per_family])
                if seeds:
                    self.measured(case,n,p,grid(self.a,seeds),stage='tune')
                    screen=[r for r in self.records if (r['case'],r['n'],r['p'])==(case,n,p)]
            self.diagnostics(case,n,p,screen,None)
            large=mesh_for_target(p,self.a.target_dofs)
            if large<=n:
                self.issues.append(dict(stage='large',case=case,p=p,status='target_not_larger_than_screen'));continue
            if self.a.level=='exhaustive':large_configs=grid(self.a,paths)
            elif self.a.large_scope=='finalists':
                large_configs=[Configuration(**r['configuration']) for r in best_by(screen,lambda r:r['configuration']['preconditioner'])]
            else:
                bypath={Configuration(**r['configuration']).path:Configuration(**r['configuration']) for r in best_by(screen,lambda r:Configuration(**r['configuration']).path)}
                # A path failing screening remains an explicit large candidate.
                large_configs=[bypath.get(c.path,c) for c in paths]
            good=[r for r in screen if eligible(r)]
            if not good or not large_configs:
                self.issues.append(dict(stage='large',case=case,p=p,status='no_valid_coarse_reference'));continue
            coarse=assembly.get('validation',{}).get('reference_l2',min(r['l2_error'] for r in good))
            # Polynomial roundoff gets an absolute gate; smooth cases must improve.
            bound=2e-8 if case in ('quadratic','variable_velocity') else max(1e-10,.95*coarse)
            if self.ensure(case,large,p) is None:continue
            rows=self.measured(case,large,p,large_configs,bound=bound,stage='large')
            self.diagnostics(case,large,p,rows,bound)
        self.report()
        return not self.issues


def main():
    p=parser();a=p.parse_args();validate_args(a,p)
    a.output=a.output.resolve()
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()
            if k not in ('resume','retry_failed','dry_run','timeout','assembly_timeout')}
    value=plan(a);value['arguments']=config;value['campaign_fingerprint']=fingerprint(value)
    manifest=a.output/'manifest.json'
    if manifest.exists():
        previous=read_json(manifest)
        if previous['campaign_fingerprint']!=value['campaign_fingerprint']:
            p.error('parameters or source changed; choose a new output directory to avoid mixing measurements')
        if not (a.resume or a.dry_run) and any((a.output/'jobs').glob('*.json')):
            p.error('existing results: use --resume or a new output directory')
    elif a.output.exists() and any(a.output.iterdir()):
        p.error('nonempty output directory without a matching manifest')
    atomic_json(manifest,value)
    print(f"Plan: {len(value['paths'])} paths; level={a.level}; {len(value['problems'])} screening/large pairs",flush=True)
    for problem in value['problems']:
        e=problem['large_estimate']
        print(f"{problem['case']} p={problem['p']} n={problem['large_n']}: {e['trace_dofs']} dofs, "
              f"estimated assembly host/GPU {e['assembly_host_bytes']/2**30:.1f}/{e['assembly_device_bytes']/2**30:.1f} GiB",flush=True)
    if a.dry_run:return
    campaign=Campaign(a)
    # Hardware/versions are checked in an isolated worker before resuming timings.
    spec={**campaign.common,'task':'environment','result':str(a.output/'current_environment.json'),
          'case':a.cases[0],'n':4,'p':2}
    envspec=a.output/'environment_spec.json';atomic_json(envspec,spec)
    proc=subprocess.run([sys.executable,'-m','scripts.adr_performance_worker','--spec',str(envspec)],cwd=ROOT,
                        timeout=120)
    current=read_json(spec['result'])
    if proc.returncode or current.get('status')!='passed':
        print(current.get('error','environment check failed'),file=sys.stderr);raise SystemExit(2)
    env=current['environment'];env.pop('pid',None)
    saved=a.output/'environment.json'
    if saved.exists() and read_json(saved)!=env:p.error('hardware or runtime changed: use a new output directory')
    atomic_json(saved,env)
    try:
        success=campaign.run()
    except KeyboardInterrupt:
        campaign.issues.append({'status':'interrupted','note':'use --resume'});campaign.report();raise SystemExit(130)
    atomic_json(a.output/'completion.json',dict(status='completed' if success else 'incomplete',
        candidate_count=len(campaign.records),eligible_count=sum(eligible(r) for r in campaign.records),
        issues=campaign.issues,coverage=campaign.coverage,
        note='completed means all scheduled stages ran with at least one valid solver per problem, not that every path converged'))
    if not success:raise SystemExit(1)


if __name__=='__main__':main()
