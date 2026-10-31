#!/usr/bin/env python3
"""Extend oscillatory ADR to the original h/p ladder, preserving prior solves."""
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
import random
import shutil
import subprocess
import sys
import numpy as np
from scripts.adr_performance_common import atomic_json, read_json
from campaigns.legacy_adr_solver_studies.run_adr_scaling_study import POINTS, configurations as previous_configurations
from campaigns.legacy_adr_solver_studies.run_oscillatory_adr_study import choices
from campaigns.legacy_adr_solver_studies.run_adr_solver_comparison import candidates
from campaigns.legacy_adr_performance.run_adv_diff_rea_performance import source_hash

ROOT=_campaign_workspace_path('hdgfem-gmres')
CASES=('cellular7_high','cellular7_weak','cellular7_directional')


def sha256(path):
    """Fingerprint preserved small artifacts and immutable source files."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configurations(case):
    """Transfer the prior high/transport/anisotropic configurations without retuning."""
    source=candidates(_campaign_workspace_path('.')/'hdgfem/configs/amgx')
    regime={'cellular7_high':'trigonometric','cellular7_weak':'advection_dominated',
            'cellular7_directional':'anisotropic'}[case]
    result=previous_configurations(regime,source)
    aliases={'asm_gmres':'asm_d48' if regime=='anisotropic' else 'asm_d24',
             'amgx_fgmres_amg':'amg_fgmres','amgx_pbicgstab_amg':'amg_pbicgstab',
             'amgx_fgmres_dilu':'dilu_fgmres','amgx_pbicgstab_dilu':'dilu_pbicgstab'}
    for row in result:row['candidate']=aliases.get(row['candidate'],row['candidate'])
    return result


def main():
    """Attempt every fixed configuration serially and record all failures and reuse."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mesh-manifest',type=Path)
    parser.add_argument('--geometries',nargs='+',choices=('square','annulus'),default=['square','annulus'])
    parser.add_argument('--timeout',type=float,default=900.)
    parser.add_argument('--resume',action='store_true')
    a=parser.parse_args();out=a.output.resolve()
    for name in ('specs','jobs','logs','cache','source_snapshot'):(out/name).mkdir(parents=True,exist_ok=True)
    if 'annulus' in a.geometries and not a.mesh_manifest:parser.error('Annulus requires --mesh-manifest')
    meshes={r['target_triangles']:r for r in read_json(a.mesh_manifest)} if a.mesh_manifest else {}
    active={case:configurations(case) for case in CASES}
    sources=['run_oscillatory_adr_scaling.py','run_oscillatory_adr_study.py','run_adr_scaling_study.py',
             'oscillatory_adr_cases.py','adv_diff_rea_cases.py','adr_performance_worker.py',
             'adr_solver_comparison_worker.py','adr_performance_common.py','adr_native_hp_worker.py']
    old=ROOT/'run_logs/adr_oscillatory_square_p6_20260918'
    old_rows=read_json(old/'summary.json')
    forbidden=ROOT/'run_logs/adr_oscillatory_star_h0.02_p6_20260918'
    preserved={str(p):dict(sha256=sha256(p),mtime_ns=p.stat().st_mtime_ns)
               for directory in (forbidden/'jobs',forbidden/'specs') for p in directory.glob('*.json')}
    manifest=dict(cases=CASES,points=POINTS,geometries=a.geometries,meshes=meshes,candidates=active,
        case_parameters={case:__import__('scripts.oscillatory_adr_cases',fromlist=['parameters']).parameters(case) for case in CASES},
        source_sha256=source_hash(),script_sha256={name:sha256(ROOT/'scripts'/name) for name in sources},
        physical_rtol=1e-10,internal_rtol=1e-11,maxiter=1000,restart=75,
        warmup=1,repeats=3,solves_per_setup=2,assembly_warmup=1,assembly_repeats=3,
        previous_square=str(old),preserved_22825_jobs=preserved,
        policy='Original square points exactly; annular counts matched within 3% and below 100k. Same per-class solver coverage as the prior study: high diffusion ASM24/AMG-Jacobi/native hp; transport ASM24/AMG-DILU/direct DILU; anisotropy ASM48/AMG-Jacobi. All AMG outer-method pairs share their preconditioner. Serial synchronized FP64 jobs, explicit zero guesses, one warmup plus three measured fresh setups, two solves/setup. Failures retained. Existing square n64/p6 outcomes reused; no 22,825-triangle job is run.')
    path=out/'manifest.json'
    if path.exists():
        previous=read_json(path)
        revisions=previous.pop('orchestration_revisions',[])
        expected=__import__('json').loads(__import__('json').dumps(manifest))
        before=previous['script_sha256'].pop('run_oscillatory_adr_scaling.py')
        after=expected['script_sha256'].pop('run_oscillatory_adr_scaling.py')
        if not a.resume or previous!=expected:
            parser.error('Existing/changed study; --resume requires identical numerical settings')
        if before!=after:
            snapshot=out/'source_snapshot/run_oscillatory_adr_scaling.py'
            assert sha256(snapshot)==before
            archived=out/'source_snapshot/revisions'/(before+'.py')
            archived.parent.mkdir(exist_ok=True)
            shutil.copy2(snapshot,archived)
            revisions.append(dict(previous_sha256=before,current_sha256=after,
                reason='Restrict prior-square cache reuse to cases actually present in the previous campaign; no numerical worker/configuration changes or repeated completed solves.'))
        if revisions:manifest['orchestration_revisions']=revisions
    atomic_json(path,manifest)
    for name in sources:
        destination=out/'source_snapshot'/name
        if destination.exists() and name!='run_oscillatory_adr_scaling.py':
            assert sha256(destination)==manifest['script_sha256'][name]
        else:shutil.copy2(ROOT/'scripts'/name,destination)
    def run(spec,module,key):
        """Run one isolated job, retaining partial records if execution fails."""
        spec['result']=str(out/'jobs'/(key+'.json'))
        specpath=out/'specs'/(key+'.json')
        if specpath.exists():assert read_json(specpath)==spec
        atomic_json(specpath,spec)
        if a.resume and Path(spec['result']).exists():
            data=read_json(spec['result'])
            if data['status']!='running':return data
        with (out/'logs'/(key+'.log')).open('w') as stream:
            try:
                process=subprocess.run([sys.executable,'-m',module,'--spec',str(specpath)],
                    stdout=stream,stderr=subprocess.STDOUT,timeout=a.timeout)
                data=read_json(spec['result']) if Path(spec['result']).exists() else dict(status='process_error',returncode=process.returncode)
                if data.get('status')=='running':data.update(status='process_error',returncode=process.returncode)
            except subprocess.TimeoutExpired:
                data=read_json(spec['result']) if Path(spec['result']).exists() else {}
                data.update(status='timeout',timeout_seconds=a.timeout)
        atomic_json(spec['result'],data)
        return data
    rows=[];assemblies=[]
    for geometry in a.geometries:
        for nominal_n,p in POINTS:
            target=2*nominal_n*nominal_n
            mesh=meshes[target] if geometry=='annulus' else None
            triangles=mesh['triangles'] if mesh else target
            assert triangles<=100000 and triangles!=22825
            n=int(np.ceil(np.sqrt(triangles/2))) if mesh else nominal_n
            ndofs=(mesh['interior_faces'] if mesh else 3*n*n-2*n)*(p+1)
            for case in CASES:
                key=f'{geometry}_{case}_target{target}_p{p}'
                reused_square=geometry=='square' and nominal_n==64 and p==6 and any(r['case']==case for r in old_rows)
                cache=(old/'cache'/f'{case}_n64_p6') if reused_square else out/'cache'/key
                common=dict(case=case,n=n,p=p,cache=str(cache.resolve()),engine='gpu',device=0,
                    rtol=1e-10,internal_rtol=1e-11,maxiter=1000,warmup=1,repeats=3,solves_per_setup=2,
                    assembly_backend='cupy',assembly_warmup=1,assembly_repeats=3,
                    reference_max_dofs=100000,memory_fraction=.8,host_limit_gib=None,
                    volume_quad_1d=None,edge_quad_1d=None,mesh_path=mesh['path'] if mesh else None)
                if reused_square:
                    assembled=read_json(cache/'metadata.json');origin='reused original square assembly'
                else:
                    assembled=run({**common,'task':'assemble'},'scripts.adr_performance_worker','assemble_'+key)
                    origin='new assembly'
                assembly=dict(case=case,n=n,nominal_n=nominal_n,p=p,geometry=geometry,target_triangles=target,
                    triangles=triangles,trace_dofs=ndofs,cache=str(cache),origin=origin,**assembled)
                assemblies.append(assembly);atomic_json(out/'assemblies.json',assemblies)
                print('assembly',key,assembled['status'],origin,'triangles',triangles,flush=True)
                if assembled['status']!='passed':raise RuntimeError('Assembly failed: '+key)
                order=deepcopy(active[case]);random.Random(8401+nominal_n+p).shuffle(order)
                for choice in order:
                    job=key+'_'+choice['candidate'];spec={**common,**choice}
                    prior=next((r for r in old_rows if r['case']==case and r['candidate']==choice['candidate']),None) if reused_square else None
                    if prior is not None:
                        oldkey=f'{case}_n64_p6_'+choice['candidate']
                        oldspec=read_json(old/'specs'/(oldkey+'.json'))
                        for setting in ('case','n','p','rtol','internal_rtol','maxiter','warmup','repeats','solves_per_setup','mesh_path'):
                            assert oldspec.get(setting)==spec.get(setting),(job,setting)
                        for setting in ('configuration','amgx_config','matrix_format'):
                            assert oldspec.get(setting)==spec.get(setting),(job,setting)
                        assert prior['worker_sha256']==manifest['script_sha256']['adr_solver_comparison_worker.py']
                        data=deepcopy(prior);data['origin']='reused existing square result';data['original_result']=str(old/'jobs'/(oldkey+'.json'))
                        spec['result']=str(out/'jobs'/(job+'.json'));atomic_json(out/'specs'/(job+'.json'),spec)
                        if prior['status']=='passed':shutil.copy2(old/'jobs'/(oldkey+'.solution.npy'),out/'jobs'/(job+'.solution.npy'))
                    else:
                        data=run(spec,'scripts.adr_native_hp_worker' if choice['family']=='native_hp' else 'scripts.adr_solver_comparison_worker',job)
                        data['origin']='new solve'
                    data.update(case=case,n=n,nominal_n=nominal_n,p=p,geometry=geometry,target_triangles=target,
                        triangles=triangles,trace_dofs=ndofs,candidate=choice['candidate'],family=choice['family'],
                        h=2/nominal_n if not mesh else mesh['mesh_size'],job_key=job)
                    atomic_json(out/'jobs'/(job+'.json'),data)
                    rows.append(data);atomic_json(out/'summary.json',rows)
                    hot=[t['solves'][1] for t in data.get('samples',[]) if len(t['solves'])==2]
                    partial=[s for t in data.get('samples',[])+data.get('warmups',[]) for s in t['solves']]
                    print(job,data['status'],data['origin'],'mean_hot_ms',sum(s['solve_ms'] for s in hot)/len(hot) if hot and data['status']=='passed' else None,
                          'iterations',sorted({s['iterations'] for s in hot or partial}),flush=True)
    for path,expected in preserved.items():
        assert sha256(path)==expected['sha256'] and Path(path).stat().st_mtime_ns==expected['mtime_ns']
    completion=dict(status='completed',scheduled=sum(len(v) for v in active.values())*len(POINTS)*len(a.geometries),attempted=len(rows),
        passed=sum(r['status']=='passed' for r in rows),reused=sum(r['origin'].startswith('reused') for r in rows),
        preserved_22825_jobs_unchanged=True,failures=[{k:r[k] for k in ('case','geometry','target_triangles','p','candidate','status')} for r in rows if r['status']!='passed'])
    atomic_json(out/'completion.json',completion);print(completion,flush=True)


if __name__=='__main__':main()
