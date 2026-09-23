#!/usr/bin/env python3
"""Replay a branch ADR cache with master native hp-BSR inside GPU GMRES.

The nonsymmetric operator stays in Bernstein coordinates. The preconditioner
uses the symmetric part through the native hp hierarchy. All numerical imports
come from master; cached assembly arrays remain the shared branch reference.
"""
from __future__ import annotations
import argparse
from dataclasses import fields
import hashlib
import gc
import json
import resource
from pathlib import Path
import sys
from time import perf_counter
import traceback


def measure(spec, master_root=None):
    master_root = Path(__file__).resolve().parents[3] if master_root is None else Path(master_root)
    # Load the branch's canonical PDE factory before selecting master numerics.
    from scripts.adv_diff_rea_cases import get_case
    _, exact = get_case(spec['case'])
    master_root = master_root.resolve()
    sys.path.insert(0,str(master_root))
    import numpy as np
    import cupy as cp
    from scipy import sparse
    import hdgfem
    assert Path(hdgfem.__file__).is_relative_to(master_root)
    from hdgfem.assembly.face_dense import FaceDenseSystem,face_dense_matvec,expand_eliminated_solution
    from hdgfem.assembly.hdg import reconstruct_local_unknowns
    from hdgfem.core.mesh import rectangle_mesh
    from hdgfem.core.space import DGSpace
    from hdgfem.backends.cupy_face_dense import CuPyFaceDenseOperator
    from hdgfem.backends.cupy_solver import CuPyProductionGMRESSolver,CuPyProductionGMRESOptions
    from hdgfem.linalg.face_hp_krylov import BernsteinHpSymmetricPartPreconditioner
    cp.cuda.Device(spec.get('device',0)).use()
    cache=Path(spec['cache'])
    get=lambda name:np.load(cache/(name+'.npy'),mmap_mode='r',allow_pickle=False)
    system=FaceDenseSystem(**{f.name:('eliminate' if f.name=='mode' else get('system_'+f.name)) for f in fields(FaceDenseSystem)})
    if spec.get('mesh_path'):
        with np.load(spec['mesh_path'],allow_pickle=False) as arrays:
            mesh=(arrays['node_coords'],arrays['triangles'])
    else:
        mesh=rectangle_mesh(spec['n'],spec['n'])
    space=DGSpace(mesh,spec['p'],basis_type='dub_orth',volume_quad_1d=spec.get('volume_quad_1d') or 2*spec['p']+2,
                  edge_quad_1d=spec.get('edge_quad_1d'))
    trace_ref=space.trace_space('bernstein')
    rhs_host=np.array(system.rhs.ravel(),copy=True)
    peer_path = spec.get('peer_solution')
    peer = np.load(peer_path, allow_pickle=False).reshape(-1) if peer_path else None
    if peer is not None and peer.size != rhs_host.size:
        raise ValueError('Archived comparison solution has a different trace dimension')
    def verify(x,converged):
        residual=face_dense_matvec(system.blocks,system.neighbors,x).ravel()-rhs_host
        rel=float(np.linalg.norm(residual)/np.linalg.norm(rhs_host))
        if spec.get('profile_residual_only'):
            return dict(passed=False,true_relative_residual=rel,l2_error=None,
                        trace_reference_error=None,peer_trace_relative_difference=None,
                        profile_note='Short, residual-only replay; convergence and PDE error are not assessed.')
        trace=expand_eliminated_solution(x,system)
        unknown=reconstruct_local_unknowns(trace,get('source_rhs'),get('local_inverse'),get('boundary'),space,trace_space=trace_ref)
        field=space.field(unknown.reshape(space.mesh.num_tri,3,space.el_dof)[:,0])
        l2=field.l2_error(exact)
        ref=cache/'reference_trace.npy'
        error=float(np.linalg.norm(trace-np.load(ref))/np.linalg.norm(np.load(ref))) if ref.exists() else None
        peer_error = float(np.linalg.norm(x.ravel()-peer)/max(np.linalg.norm(peer),1e-300)) if peer is not None else None
        bound = spec.get('l2_bound')
        return dict(passed=bool(converged and np.isfinite(rel) and rel<=spec['rtol'] and np.isfinite(l2) and (error is None or error<1e-8)
                               and (bound is None or l2 <= bound)),
            true_relative_residual=rel,l2_error=float(l2),trace_reference_error=error,
            peer_trace_relative_difference=peer_error)
    output={'status':'running','family':'native_hp','candidate':spec.get('candidate','native_hp_standard'),
        'case':spec['case'],'n':spec['n'],'p':spec['p'],'rtol':spec['rtol'],'internal_rtol':spec.get('internal_rtol',spec['rtol']),'samples':[],'warmups':[],
        'note':'Master native hp-BSR hierarchy on symmetric part, Bernstein coordinate transforms, GMRES on unchanged full ADR operator; not native PCGF on ADR.'}
    output['operator_sha256']=hashlib.sha256(np.asarray(system.blocks).tobytes()+rhs_host.tobytes()).hexdigest()
    if spec.get('expected_operator_sha256') and output['operator_sha256'] != spec['expected_operator_sha256']:
        raise ValueError('Cached operator/RHS differs from the archived comparison')
    from hdgfem.linalg.face_hp_multigrid import face_hp_mg_preconditioner_parameters
    output.update(policy=spec.get('policy','standard'),
        configuration=face_hp_mg_preconditioner_parameters(spec.get('policy','standard'), overrides=spec.get('native_tuning')),
        restart=spec.get('restart',75), maxiter=spec['maxiter'],
        requested_repeats=spec['repeats'], requested_warmups=spec['warmup'],
        solves_per_setup=spec['solves_per_setup'], peer_solution=peer_path,
        trace_dofs=system.num_dofs, triangles=space.mesh.num_tri,
        neighbors_sha256=hashlib.sha256(np.asarray(system.neighbors).tobytes()).hexdigest())
    import platform
    properties=cp.cuda.runtime.getDeviceProperties(spec.get('device',0))
    gpu_name=properties['name']
    output['environment']=dict(python=platform.python_version(),numpy=np.__version__,cupy=cp.__version__,
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),cuda_driver=cp.cuda.runtime.driverGetVersion(),
        gpu=gpu_name.decode() if isinstance(gpu_name,bytes) else gpu_name)
    output['worker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output['matrix_cache']=str(cache)
    output['matrix_format']='bsr'
    output['adapter_sha256']=hashlib.sha256((master_root/'hdgfem/linalg/face_hp_krylov.py').read_bytes()).hexdigest()
    def save():
        from scripts.adr_performance_common import atomic_json
        atomic_json(spec['result'],output)
    sync=cp.cuda.get_current_stream().synchronize
    cp.zeros(1).sum().item()
    phase='setup'
    try:
        for repetition in range(spec['warmup']+spec['repeats']):
            pre=None
            try:
                phase='setup'
                sync();start=perf_counter()
                valid=system.neighbors>=0;q=system.block_size
                matrix=sparse.bsr_matrix((np.ascontiguousarray(system.blocks[valid]),system.neighbors[valid].astype(np.int32),
                    np.r_[0,np.cumsum(valid.sum(axis=1))].astype(np.int32)),shape=(system.num_dofs,system.num_dofs))
                matrix.sort_indices()
                pre=BernsteinHpSymmetricPartPreconditioner(matrix,degree=spec['p'],policy=spec.get('policy','standard'),
                    tuning=spec.get('native_tuning'))
                op=CuPyFaceDenseOperator.from_system(system,implementation='raw_fused',dtype=np.float64,device_id=spec.get('device',0))
                solver=CuPyProductionGMRESSolver(op,preconditioner=pre,options=CuPyProductionGMRESOptions(
                    restart=spec.get('restart',75),max_iterations=spec['maxiter'],rtol=spec.get('internal_rtol',spec['rtol']),atol=0.,orthogonalization='cgs2',
                    raise_on_failure=False,cgs2_fallback_threshold=None))
                rhs=cp.asarray(rhs_host);sync();setup=1000*(perf_counter()-start)
                solves=[]
                phase='solve'
                for _ in range(spec['solves_per_setup']):
                    sync();start=perf_counter()
                    if spec.get('profile_nvtx'):
                        cp.cuda.nvtx.RangePush('hdgfem_solver')
                    try:
                        r=solver.solve(rhs,x0=None,raise_on_failure=False)
                    finally:
                        if spec.get('profile_nvtx'):
                            cp.cuda.nvtx.RangePop()
                    sync()
                    elapsed=1000*(perf_counter()-start)
                    x=cp.asnumpy(r.solution)
                    checked=verify(x,r.converged)
                    if spec.get('record_residual_history'):
                        checked.update(residual_history=r.true_residual_history.tolist(),
                            estimated_preconditioned_residual_history=r.estimated_preconditioned_residual_history.tolist(),
                            residual_history_kind='absolute true residuals at restart boundaries')
                    solves.append(dict(solve_ms=elapsed,iterations=r.iterations,matvec_count=r.matvec_count,
                        preconditioner_applications=r.preconditioner_count,**checked))
                    if not checked['passed']:break
                row=dict(setup_ms=setup,solves=solves,fresh_setup_solve_ms=setup+solves[0]['solve_ms'],
                    amortized_ms=(setup+sum(s['solve_ms'] for s in solves))/len(solves))
                if 'pyamgx' in sys.modules:
                    row['amgx_memory']=sys.modules['pyamgx'].get_device_memory_stats()
                if spec.get('profile') and repetition == spec['warmup']:
                    from hdgfem.backends.cupy_profiling import benchmark_cuda_call,CuPyGMRESProfiler
                    probe=cp.asarray(np.random.default_rng(1729).standard_normal(system.num_dofs));out=cp.empty_like(probe)
                    output['complete_preconditioner']=benchmark_cuda_call(lambda:pre.apply_into(probe,out),warmup=3,repeats=10).to_dict(include_samples=True)
                    profiler=CuPyGMRESProfiler(device_id=spec.get('device',0))
                    diagnostic=solver.solve(rhs,profiler=profiler,raise_on_failure=False)
                    output['gmres_profile']=profiler.finalize().to_dict()
                    output['profile_validation']=verify(cp.asnumpy(diagnostic.solution),diagnostic.converged)
                    output['profile_note']='Separate instrumented diagnostic; never used for ranking.'
                    del probe,out,diagnostic
                (output['warmups'] if repetition<spec['warmup'] else output['samples']).append(row);save()
                if not all(s['passed'] for s in solves):output['status']='numerical_failure';save();raise SystemExit(1)
            finally:
                if pre is not None:pre.close()
                solver=op=rhs=pre=None
                if 'r' in locals():del r
                gc.collect()
        for key in ('setup_ms','fresh_setup_solve_ms','amortized_ms'):
            output[key.replace('_ms','_median_ms')]=float(np.median([r[key] for r in output['samples']]))
        hot=[s['solve_ms'] for r in output['samples'] for s in r['solves'][1:]]
        output['hot_solve_median_ms']=float(np.median(hot))
        output['hot_solve_mean_ms']=float(np.mean(hot))
        output['hot_solve_min_ms']=float(min(hot))
        output['hot_solve_max_ms']=float(max(hot))
        output['worst_true_relative_residual']=max(s['true_relative_residual'] for r in output['samples']+output['warmups'] for s in r['solves'])
        np.save(Path(spec['result']).with_suffix('.solution.npy'),x,allow_pickle=False)
        output['peak_process_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        output['status']='passed'
        output['adapter_sha256']=hashlib.sha256((master_root/'hdgfem/linalg/face_hp_krylov.py').read_bytes()).hexdigest()
        output['worker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except Exception as exc:output.update(status='setup_error' if phase=='setup' else 'solve_error',error=str(exc),traceback=traceback.format_exc())
    save();print(output['status'],output.get('fresh_setup_solve_median_ms'),flush=True)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--master-root',type=Path,default=Path(__file__).resolve().parents[3])
    args=parser.parse_args()
    result=measure(json.loads(args.spec.read_text()),args.master_root)
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
