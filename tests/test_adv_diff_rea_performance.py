from dataclasses import asdict
from pathlib import Path
import math
import subprocess
import numpy as np
import pytest
from scripts.adr_performance_common import (Configuration,path_configurations,parameter_grid,
    mesh_for_target,trace_dofs,eligible,best_by,atomic_json,read_json)
from scripts.run_adv_diff_rea_performance import parser,plan,Campaign


def test_all_legal_paths_are_present_and_gpu_solve_is_not_duplicated():
    paths=path_configurations()
    assert len(paths)==108==len({p.path for p in paths})
    assert len([p for p in paths if p.preconditioner=='block_jacobi'])==21
    assert len([p for p in paths if p.preconditioner=='asm'])==30
    for p in paths:
        if p.local_solver=='gpu_solve':assert p.application=='matmul'
        assert (p.polynomial_degree is not None)==p.preconditioner.endswith('poly')
        if p.preconditioner in ('none','poly'):assert p.local_solver is None and p.application is None


def test_grid_expands_only_relevant_parameters():
    paths=path_configurations(('none','poly'),('raw',))
    grid=parameter_grid(paths,degrees=(6,12),restarts=(30,60),orthogonalizations=('mgs','cgs2'),
                        polynomial_orthogonalizations=('mgs','cgs2'))
    assert len(grid)==4+16
    assert len({c.key for c in grid})==20


@pytest.mark.parametrize('p',[1,2,3,4,6])
def test_target_uses_free_trace_dofs_and_nearest_mesh(p):
    n=mesh_for_target(p)
    assert abs(trace_dofs(n,p)-1_000_000)<=abs(trace_dofs(n-1,p)-1_000_000)
    assert abs(trace_dofs(n,p)-1_000_000)<=abs(trace_dofs(n+1,p)-1_000_000)
    assert abs(trace_dofs(n,p)-1_000_000)<10000


def test_rank_rejects_fast_failures_partial_samples_and_nonfinite_timing():
    good=dict(status='passed',samples=[dict(passed=True)]*2,requested_repeats=2,
              setup_solve_median_ms=10.,family='asm')
    variants=[dict(good,status='timeout',setup_solve_median_ms=1.),
              dict(good,samples=[dict(passed=True)]),
              dict(good,samples=[dict(passed=True),dict(passed=False)]),
              dict(good,setup_solve_median_ms=float('nan')),dict(good,setup_solve_median_ms=None)]
    assert all(not eligible(r) for r in variants)
    assert best_by(variants+[good],lambda r:r['family'])==[good]


def test_manifest_full_includes_all_cases_paths_and_large_targets():
    a=parser().parse_args(['--level','full'])
    result=plan(a)
    assert result['path_count']==108
    assert len(result['problems'])==10
    assert len(result['validation'])==10
    assert result['large_scope']=='paths'
    assert all(990000<r['large_estimate']['trace_dofs']<1010000 for r in result['problems'])


def test_json_retains_failure_record_without_nonstandard_nan(tmp_path):
    path=tmp_path/'failure.json'
    atomic_json(path,dict(status='numerical_failure',residual=float('nan'),flag=np.bool_(False)))
    assert read_json(path)==dict(status='numerical_failure',residual=None,flag=False)


def test_timeout_preserves_partial_repetitions_and_cannot_rank(tmp_path,monkeypatch):
    a=parser().parse_args(['--level','smoke','--cpu-check','--output',str(tmp_path)])
    campaign=Campaign(a)
    def fail(argv,**kwargs):
        spec=read_json(argv[-1]);atomic_json(spec['result'],dict(status='running',samples=[dict(passed=True)],requested_repeats=2))
        raise subprocess.TimeoutExpired(argv,1)
    monkeypatch.setattr(subprocess,'run',fail)
    c=Configuration('none','cpu_face_dense',None,None,None,'scipy_gmres',30,None)
    row=campaign.job('measure','quadratic',4,2,c)
    assert row['status']=='timeout' and len(row['samples'])==1
    assert not eligible(row)


def test_worker_cpu_assembly_measure_profile_and_pipeline(tmp_path):
    from scripts.adr_performance_worker import assemble,measure,profile,pipeline
    c=Configuration('none','cpu_face_dense',None,None,None,'scipy_gmres',30,None)
    spec=dict(case='variable_velocity',n=4,p=2,engine='cpu',assembly_backend='numpy',device=0,
        cache=str(tmp_path/'cache'),result=str(tmp_path/'result.json'),configuration=asdict(c),
        rtol=1e-11,maxiter=1000,warmup=0,repeats=2,assembly_warmup=0,assembly_repeats=1,
        component_warmup=0,component_repeats=2,reference_max_dofs=1000,memory_fraction=.9,host_limit_gib=None)
    a=assemble(spec);assert a['validation']['reference_l2']<1e-10
    result=measure(spec);assert eligible(result)
    assert result['worst_true_relative_residual']<spec['rtol']
    assert profile(spec)['status']=='passed'
    assert pipeline(spec)['status']=='passed'


def test_gpu_worker_measures_and_profiles_actual_adr_system(tmp_path):
    from hdgfem.backends.cupy import require_cupy_device
    try:require_cupy_device()
    except RuntimeError as exc:pytest.skip(str(exc))
    from scripts.adr_performance_worker import assemble,measure,profile
    c=Configuration(polynomial_degree=6,restart=30)
    spec=dict(case='variable_velocity',n=4,p=2,engine='gpu',assembly_backend='cupy',device=0,
        cache=str(tmp_path/'cache'),result=str(tmp_path/'result.json'),configuration=asdict(c),
        rtol=1e-11,maxiter=1000,warmup=1,repeats=2,assembly_warmup=0,assembly_repeats=1,
        component_warmup=1,component_repeats=2,reference_max_dofs=1000,memory_fraction=.9,host_limit_gib=None)
    assert assemble(spec)['status']=='passed'
    assert eligible(measure(spec))
    report=profile(spec)
    assert report['status']=='passed' and 'gmres' in report
    assert report['operator']['total']['median_ms']>=0


def test_unexecuted_candidates_make_coverage_incomplete(tmp_path,monkeypatch):
    a=parser().parse_args(['--level','smoke','--cpu-check','--output',str(tmp_path)])
    campaign=Campaign(a)
    c=Configuration('none','cpu_face_dense',None,None,None,'scipy_gmres',30,None)
    monkeypatch.setattr(campaign,'job',lambda *args,**kwargs:dict(status='skipped_budget',job='missing',configuration=asdict(c)))
    campaign.measured('quadratic',4,2,[c],stage='screen')
    assert campaign.coverage[-1]['execution_issues']==1
    assert any(r['status']=='skipped_budget' for r in campaign.issues)
