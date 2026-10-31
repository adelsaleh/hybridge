#!/usr/bin/env python3
"""Audit the complete three-class oscillatory h/p campaign and preserved evidence."""
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
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scripts.adr_performance_common import read_json, atomic_json
from hdgfem.assembly.face_dense import face_dense_matvec
from campaigns.legacy_adr_solver_studies.run_adr_scaling_study import POINTS
from campaigns.legacy_adr_performance.run_adv_diff_rea_performance import source_hash

CASES=('cellular7_high','cellular7_weak','cellular7_directional')


def digest(path):
    """Hash streamed inputs without materializing large cache arrays."""
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def mesh_quality(path):
    """Check actual mesh topology and report affine triangle quality."""
    with np.load(path,allow_pickle=False) as arrays:
        nodes=arrays['node_coords'];tri=arrays['triangles']
    pairs=np.sort(np.concatenate((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1)
    edges,counts=np.unique(pairs,axis=0,return_counts=True)
    assert np.all((counts==1)|(counts==2))
    graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(nodes),len(nodes))).tocsr()
    connected=int(connected_components(graph,directed=False,return_labels=False))
    boundary=edges[counts==1];ids=np.unique(boundary)
    bg=coo_matrix((np.ones(len(boundary)),(boundary[:,0],boundary[:,1])),shape=graph.shape).tocsr()
    bgraph=(bg+bg.T)[ids][:,ids]
    assert np.all(np.diff(bgraph.indptr)==2)
    loops=int(connected_components(bgraph,directed=False,return_labels=False))
    vertices=nodes[tri]
    vectors=np.roll(vertices,-1,axis=1)-vertices
    lengths=np.linalg.norm(vectors,axis=2)
    angles=np.arccos(np.clip((lengths**2+np.roll(lengths,1,axis=1)**2-np.roll(lengths,2,axis=1)**2)/(2*lengths*np.roll(lengths,1,axis=1)),-1,1))
    a=vertices[:,1]-vertices[:,0];b=vertices[:,2]-vertices[:,0]
    area2=a[:,0]*b[:,1]-a[:,1]*b[:,0]
    assert np.all(area2>0) and connected==1 and loops==2
    return dict(triangles=len(tri),vertices=len(nodes),interior_faces=int(np.count_nonzero(counts==2)),
        connected_components=connected,boundary_components=loops,area=float(area2.sum()/2),
        minimum_angle_degrees=float(np.rad2deg(angles.min())),minimum_edge=float(lengths.min()),maximum_edge=float(lengths.max()))


def main():
    """Verify every required class/point/method and audit numerical acceptance."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();study=args.study_root.resolve();branch=_campaign_workspace_path('hdgfem-gmres')
    master=branch.parent/'hdgfem'
    all_rows=[];assemblies=[];systems=[];campaigns=[];mesh_checks=[];preserved={}
    for geometry in ('square','annulus'):
        folder=study/'scaling'/geometry
        manifest=read_json(folder/'manifest.json');completion=read_json(folder/'completion.json')
        rows=read_json(folder/'summary.json');built=read_json(folder/'assemblies.json')
        assert manifest['cases']==list(CASES)
        assert manifest['points']==[list(p) for p in POINTS]
        assert manifest['geometries']==[geometry]
        assert completion['status']=='completed' and completion['scheduled']==completion['attempted']==len(rows)==96
        assert len(built)==24
        assert manifest['source_sha256']==source_hash()
        assert manifest['warmup']==1 and manifest['repeats']==3 and manifest['solves_per_setup']==2
        for name,expected in manifest['script_sha256'].items():
            assert digest(folder/'source_snapshot'/name)==expected
            assert digest(branch/'scripts'/name)==expected
        for revision in manifest.get('orchestration_revisions',[]):
            assert digest(folder/'source_snapshot/revisions'/(revision['previous_sha256']+'.py'))==revision['previous_sha256']
        assert sum(r['status']=='passed' for r in rows)==completion['passed']
        assert sum(r['origin'].startswith('reused') for r in rows)==completion['reused']==(4 if geometry=='square' else 0)
        for path,expected in manifest['preserved_22825_jobs'].items():
            assert digest(path)==expected['sha256']
            assert Path(path).stat().st_mtime_ns==expected['mtime_ns']
            preserved[path]=expected
        for raw in manifest['meshes'].values():
            assert digest(raw['path'])==raw['sha256']==raw['mesh_sha256']
            quality=mesh_quality(raw['path'])
            assert quality['triangles']==raw['triangles']<=100000
            assert quality['interior_faces']==raw['interior_faces']
            assert abs(raw['triangles']/raw['target_triangles']-1)<.03
            mesh_checks.append(dict(**raw,quality=quality))
        for nominal_n,p in POINTS:
            for case in CASES:
                group=[r for r in rows if r['case']==case and r['nominal_n']==nominal_n and r['p']==p]
                expected={c['candidate']:c for c in manifest['candidates'][case]}
                assert {r['candidate'] for r in group}==set(expected) and len(group)==len(expected)
                built_row=next(a for a in built if a['case']==case and a['nominal_n']==nominal_n and a['p']==p)
                assert built_row['status']=='passed'
                cache=Path(built_row['cache'])
                blocks=np.load(cache/'system_blocks.npy',mmap_mode='r')
                rhs=np.load(cache/'system_rhs.npy',mmap_mode='r')
                op_hash=hashlib.sha256(blocks.tobytes()+rhs.reshape(-1).tobytes()).hexdigest()
                reference=None;solutions={};residuals={}
                neighbors=np.load(cache/'system_neighbors.npy',mmap_mode='r')
                for row in group:
                    assert row['status'] in ('passed','numerical_failure'),(row['job_key'],row['status'])
                    assert row['operator_sha256']==op_hash
                    assert row['geometry']==geometry and row['trace_dofs']==rhs.size
                    assert row['triangles']<=100000 and row['triangles']!=22825
                    worker='adr_native_hp_worker.py' if row['family']=='native_hp' else 'adr_solver_comparison_worker.py'
                    assert row['worker_sha256']==manifest['script_sha256'][worker]
                    if row['family']=='native_hp':
                        assert row['adapter_sha256']==digest(master/'hdgfem/linalg/face_hp_krylov.py')
                    spec=read_json(folder/'specs'/(row['job_key']+'.json'))
                    assert Path(spec['cache'])==cache
                    assert spec['rtol']==row['rtol']==1e-10 and spec['internal_rtol']==row['internal_rtol']==1e-11
                    assert spec['maxiter']==1000
                    for key,value in expected[row['candidate']].items():assert spec[key]==value
                    if row['family']=='amgx':
                        assert spec['matrix_format']=='bsr'
                        actual=row['effective_amgx_config']['solver']
                        assert actual['max_iters']==1000 and actual['use_scalar_norm']==1 and actual['norm']=='L2'
                        assert np.isclose(actual['tolerance'],1e-11*np.linalg.norm(rhs),rtol=1e-14,atol=0)
                        if actual['solver']=='FGMRES':assert actual['gmres_n_restart']==75
                    if row['status']=='passed':
                        assert len(row['warmups'])==1 and len(row['samples'])==3
                        for sample in row['warmups']+row['samples']:
                            assert len(sample['solves'])==2
                            assert all(s['passed'] and np.isfinite(s['l2_error']) and s['true_relative_residual']<=1e-10 for s in sample['solves'])
                        solution=np.load(folder/'jobs'/(row['job_key']+'.solution.npy'))
                        assert solution.size==rhs.size and np.isfinite(solution).all()
                        residual=float(np.linalg.norm(face_dense_matvec(blocks,neighbors,solution).ravel()-rhs.ravel())/np.linalg.norm(rhs))
                        assert residual<=1e-10,(row['job_key'],residual)
                        residuals[row['candidate']]=residual
                        solutions[row['candidate']]=solution
                        if row['family']=='asm_pp':reference=solution
                    else:
                        assert any(not s['passed'] for sample in row['warmups']+row['samples'] for s in sample['solves'])
                    if row['origin'].startswith('reused'):
                        original=read_json(row['original_result'])
                        for key in ('samples','warmups','worker_sha256','operator_sha256','status'):
                            assert original[key]==row[key]
                deviations={}
                if reference is not None:
                    deviations={name:float(np.linalg.norm(solution-reference)/np.linalg.norm(reference)) for name,solution in solutions.items()}
                assert all(value<1e-8 for value in deviations.values()),deviations
                systems.append(dict(geometry=geometry,case=case,nominal_n=nominal_n,p=p,
                    triangles=built_row['triangles'],trace_dofs=rhs.size,operator_rhs_sha256=op_hash,
                    cross_solver_trace_relative_difference_to_asm=deviations,
                    recomputed_saved_trace_relative_residuals=residuals,
                    passing_candidates=list(solutions),failed_candidates=[r['candidate'] for r in group if r['status']!='passed']))
        all_rows+=rows;assemblies+=built;campaigns.append(dict(geometry=geometry,completion=completion,manifest=manifest))
    passing=[s for r in all_rows if r['status']=='passed' for t in r['warmups']+r['samples'] for s in t['solves']]
    attempted=[s for r in all_rows for t in r['warmups']+r['samples'] for s in t['solves']]
    references=[a for a in assemblies if 'reference_l2' in a['validation']]
    original=read_json(master/'docs/research/solver_studies/adr_solver_comparison_2026_09_17.data.json')['provenance']
    binaries={}
    for path,record in original['binaries'].items():
        current=digest(path);assert current==record['sha256']
        binaries[path]=dict(sha256=current,size_bytes=Path(path).stat().st_size)
    diagnostics=study/'diagnostics'
    assert read_json(diagnostics/'completion.json')['status']=='completed'
    diagnostic_manifest=read_json(diagnostics/'manifest.json')
    for name,expected in diagnostic_manifest['source_sha256'].items():
        assert digest(diagnostics/'source_snapshot'/name)==expected
        assert digest(branch/'scripts'/name)==expected
    profile_checks=[]
    for profile in read_json(diagnostics/'profiles.json'):
        if profile.get('status')=='skipped':continue
        data=profile['result'];assert data['status']=='passed'
        assert profile['triangles']==99984 and profile['triangles']!=22825
        name=profile['case']+'_'+profile['candidate']
        spec=read_json(diagnostics/(name+'.spec.json'))
        system=next(row for row in systems if row['geometry']=='annulus' and row['case']==profile['case'] and row['nominal_n']==223 and row['p']==6)
        if profile['family']=='amgx':
            instrumented=read_json(diagnostics/(name+'.instrumented.json'))
            assert instrumented['operator_sha256']==system['operator_rhs_sha256']
            assert all(solve['passed'] and solve['true_relative_residual']<=1e-10 for trial in instrumented['samples'] for solve in trial['solves'])
            assert data['native_event_preconditioner_calls']>0 and data['preconditioner_call_mean_ms']>0
        else:
            timing=data['complete_preconditioner']
            assert timing['warmup']==3 and timing['repeats']==len(timing['samples_ms'])==10
            assert 0<timing['minimum_ms']<=timing['mean_ms'] and np.isfinite(timing['samples_ms']).all()
            validation=data['validation' if profile['family']=='asm_pp' else 'profile_validation']
            assert validation['passed'] and validation['true_relative_residual']<=1e-10
            if profile['family']=='native_hp':assert data['operator_sha256']==system['operator_rhs_sha256']
        original=next(r for r in all_rows if r['geometry']=='annulus' and r['case']==profile['case'] and r['nominal_n']==223 and r['p']==6 and r['candidate']==profile['candidate'])
        original_spec=read_json(study/'scaling/annulus/specs'/(original['job_key']+'.json'))
        assert spec['cache']==original_spec['cache']
        for key in ('configuration','amgx_config','policy'):
            if key in original_spec:assert spec[key]==original_spec[key]
        profile_checks.append(dict(case=profile['case'],candidate=profile['candidate'],status='passed'))
    quadrature=read_json(diagnostics/'quadrature.json')
    assert {row['case'] for row in quadrature}=={'cellular7_high','cellular7_directional'}
    for row in quadrature:
        assert row['status']=='passed' and row['volume_points_1d']==row['edge_points_1d']==[14,28]
        assert 0<=row['difference_over_original_error']<.05
    matrices=read_json(diagnostics/'matrix_diagnostics.json')
    assert {row['case'] for row in matrices}==set(CASES)
    for row in matrices:
        assert row['trace_dofs']==11785*7 and np.isfinite(row['condition1_lower_estimate']) and row['condition1_lower_estimate']>=1
        blocks=np.load(Path(row['cache'])/'system_blocks.npy',mmap_mode='r')
        assert hashlib.sha256(blocks.tobytes()).hexdigest()==row['matrix_sha256']
    migration=read_json(study/'raw_migration.json')
    moved_files=0
    for directory in migration['directories']:
        previous=Path(directory['previous_path']);canonical=Path(directory['canonical_path'])
        assert previous.is_symlink() and previous.resolve()==canonical.resolve()
        for row in directory['files']:
            path=canonical/row['path'];stat=path.stat()
            assert stat.st_size==row['size'] and stat.st_mtime_ns==row['mtime_ns'],str(path)
            if 'sha256' in row:assert digest(path)==row['sha256'],str(path)
            moved_files+=1
    test_log=study/'checks/cpu_tests.log'
    assert '23 passed, 9 deselected' in test_log.read_text()
    result=dict(status='passed',systems=len(systems),configurations=len(all_rows),
        passed_configurations=sum(r['status']=='passed' for r in all_rows),failed_configurations=sum(r['status']!='passed' for r in all_rows),
        reused_configurations=4,new_configurations=len(all_rows)-4,
        passing_configuration_solves=len(passing),attempted_solves=len(attempted),
        worst_physical_relative_residual=max(s['true_relative_residual'] for s in passing),
        worst_cross_solver_trace_relative_difference=max((v for s in systems for v in s['cross_solver_trace_relative_difference_to_asm'].values()),default=None),
        independent_cpu_reference_systems=len(references),
        worst_cpu_gpu_blocks_relative_error=max(a['validation']['cpu_gpu_blocks_relative_error'] for a in references),
        worst_cpu_gpu_rhs_relative_error=max(a['validation']['cpu_gpu_rhs_relative_error'] for a in references),
        systems_audit=systems,campaigns=campaigns,geometry=mesh_checks,binaries=binaries,
        separate_diagnostics=dict(profiles=profile_checks,quadrature=quadrature,matrix_diagnostics=matrices),
        moved_raw_files_unchanged=moved_files,
        preserved_22825_jobs_unchanged=True,preserved_22825_jobs=preserved,
        cpu_tests=dict(passed=23,deselected=9,log=str(test_log),log_sha256=digest(test_log),command="python -m pytest -q tests/test_oscillatory_adr_cases.py tests/test_adv_diff_rea.py -k 'not gpu'"),
        native_adapter_sha256=digest(master/'hdgfem/linalg/face_hp_krylov.py'),
        scope='Exact original square h/p points and solver coverage; annular counts within 3%. No 22,825-triangle assembly, solve or profile repeated. Current source snapshots and unchanged prebuilt libraries identify execution. Numerical failures are retained, not timing ranks.')
    args.output.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if isinstance(v,(int,float,bool,str))},indent=2))


if __name__=='__main__':main()
