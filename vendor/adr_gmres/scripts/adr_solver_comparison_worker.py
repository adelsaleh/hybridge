#!/usr/bin/env python3
"""Time ASM+PP GMRES or AMGX against one immutable ADR cache.

Fresh setup and repeated zero-guess solves are separated. Independent CPU
physical-residual, reconstructed PDE-error and direct-reference checks use the
existing ADR worker. Failed candidates never enter the ranking.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import resource
from time import perf_counter
import traceback
import numpy as np
from scripts.adr_performance_common import atomic_json, read_json
from scripts.adr_performance_worker import load_cache, get_space, verify, build_gpu, metadata, guard
from hdgfem.solvers.adv_diff_rea import _csr


def measure(spec):
    import cupy as cp
    guard(spec,'solver')
    system,get=load_cache(spec['cache'])
    space=get_space(spec)
    rhs_host=np.array(system.rhs.reshape(-1),copy=True)
    cp.zeros(1).sum().item()  # Runtime context before timing; common to both engines.
    sync=cp.cuda.get_current_stream().synchronize
    pyamgx=None
    native = spec['family'] != 'amgx'
    if not native:
        import pyamgx
        pyamgx.initialize()
        pyamgx.register_print_callback(lambda message: None)
    result={'status':'running','family':spec['family'],'candidate':spec['candidate'],
        'case':spec['case'],'n':spec['n'],'p':spec['p'],'rtol':spec['rtol'],'internal_rtol':spec.get('internal_rtol',spec['rtol']),
        'environment':metadata({**spec,'engine':'gpu'}),'samples':[],'warmups':[],
        'requested_repeats':spec['repeats'],'solves_per_setup':spec['solves_per_setup'],
        'matrix_cache':str(spec['cache']),'matrix_format':spec.get('matrix_format','face_dense'),
        'worker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result['operator_sha256']=hashlib.sha256(np.asarray(system.blocks).tobytes()+rhs_host.tobytes()).hexdigest()
    if spec.get('expected_operator_sha256') and result['operator_sha256'] != spec['expected_operator_sha256']:
        raise ValueError('Cached operator/RHS differs from the recorded assembly')
    result['configuration']=spec.get('configuration',spec.get('amgx_config'))
    try:
        for repetition in range(spec['warmup']+spec['repeats']):
            objects=[]
            solver=op=base=pre=rhs=solution=None
            sync(); start=perf_counter()
            try:
                if native:
                    solver,op,base,pre,rhs,stages=build_gpu({**spec,'rtol':spec.get('internal_rtol',spec['rtol'])},system,get('element_blocks'),get('loc2glob'))
                else:
                    config=deepcopy(spec['amgx_config'])
                    cfgsolver=config['solver']
                    cfgsolver.update(convergence='ABSOLUTE',tolerance=float(spec.get('internal_rtol',spec.get('amgx_internal_rtol',spec['rtol']))*np.linalg.norm(rhs_host)),
                        max_iters=spec['maxiter'],monitor_residual=1,norm='L2',use_scalar_norm=1,print_solve_stats=0,obtain_timings=0)
                    if cfgsolver['solver'] in ('FGMRES','GMRES'):cfgsolver['gmres_n_restart']=spec.get('restart',75)
                    if spec.get('record_residual_history'):cfgsolver['store_res_history']=1
                    result['effective_amgx_config']=config
                    result['amgx_internal_rtol']=spec.get('internal_rtol',spec.get('amgx_internal_rtol',spec['rtol']))
                    stamp=perf_counter()
                    block_size = system.block_size if spec.get('matrix_format')=='bsr' else 1
                    if block_size > 1:
                        from scipy.sparse import bsr_matrix
                        valid=system.neighbors>=0
                        indptr=np.r_[0,np.cumsum(np.count_nonzero(valid,axis=1))].astype(np.int32)
                        matrix=bsr_matrix((np.ascontiguousarray(system.blocks[valid]),
                            np.asarray(system.neighbors[valid],dtype=np.int32),indptr),
                            shape=(system.num_dofs,system.num_dofs))
                        matrix.sort_indices()
                    else:
                        matrix=_csr(system)
                    matrix.indptr=np.asarray(matrix.indptr,dtype=np.int32)
                    matrix.indices=np.asarray(matrix.indices,dtype=np.int32)
                    conversion=perf_counter()-stamp
                    cfg=pyamgx.Config().create_from_dict(config);objects.append(cfg)
                    resources=pyamgx.Resources().create_simple(cfg);objects.append(resources)
                    A=pyamgx.Matrix().create(resources);objects.append(A)
                    if block_size > 1:
                        A.upload(matrix.indptr,matrix.indices,np.ascontiguousarray(matrix.data),
                            block_dims=[block_size,block_size],shape=[system.num_rows,system.num_rows])
                    else:A.upload_CSR(matrix)
                    rhs=pyamgx.Vector().create(resources);objects.append(rhs);rhs.upload_raw(rhs_host.ctypes.data,rhs_host.size//block_size,block_size)
                    solution=pyamgx.Vector().create(resources);objects.append(solution);solution.set_zero(n=rhs_host.size//block_size,block_dim=block_size)
                    solver=pyamgx.Solver().create(resources,cfg);objects.append(solver)
                    sync();stamp=perf_counter();solver.setup(A);sync()
                    stages={'csr_conversion_ms':1000*conversion,'hierarchy_setup_ms':1000*(perf_counter()-stamp)}
                sync(); setup=1000*(perf_counter()-start)
                solves=[]
                for solve_index in range(spec['solves_per_setup']):
                    if native:
                        before=(pre.matvec_count,pre.base_preconditioner_count)
                    sync();stamp=perf_counter()
                    if not native:
                        # The installed native zero-guess flag alone does not clear a reused x.
                        # Reset storage explicitly and include the reset in solve timing.
                        solution.set_zero(n=rhs_host.size//block_size,block_dim=block_size)
                    if spec.get('profile_nvtx'):
                        cp.cuda.nvtx.RangePush('hdgfem_solver')
                    try:
                        if native:
                            answer=solver.solve(rhs,x0=None,raise_on_failure=False)
                        else:
                            solver.solve(rhs,solution,zero_initial_guess=True)
                    finally:
                        if spec.get('profile_nvtx'):
                            cp.cuda.nvtx.RangePop()
                    sync();elapsed=1000*(perf_counter()-stamp)
                    if native:
                        x=cp.asnumpy(answer.solution)
                        converged=answer.converged
                        base_applications=int(pre.base_preconditioner_count-before[1])
                        counts={'iterations':int(answer.iterations),'outer_matvec_count':int(answer.matvec_count),
                            'polynomial_applications':int(answer.preconditioner_count),
                            'polynomial_inner_matvecs':int(pre.matvec_count-before[0]),
                            'base_preconditioner_applications':base_applications,
                            'total_solve_matvecs':int(answer.matvec_count+pre.matvec_count-before[0]),
                            'status':answer.status}
                        family_counter = {
                            'asm_pp': 'asm_applications',
                            'bj_pp': 'block_jacobi_applications',
                            'pp': 'unbased_polynomial_applications',
                        }.get(spec['family'])
                        if family_counter is not None:
                            counts[family_counter]=base_applications
                    else:
                        x=solution.download(np.empty_like(rhs_host));converged=solver.status=='success'
                        counts={'iterations':int(solver.iterations_number),'status':solver.status,
                            'outer_matvec_count':None,'preconditioner_applications':None,
                            'counter_note':'AMGX public bindings do not expose exact application counts.'}
                    if spec.get('profile_residual_only'):
                        residual=face_dense_matvec(system.blocks,system.neighbors,x).reshape(-1)-rhs_host
                        rn=float(np.linalg.norm(residual))
                        bn=float(np.linalg.norm(rhs_host))
                        checked={'passed':False,'true_residual_norm':rn,
                                 'true_relative_residual':rn/bn if bn else rn,
                                 'l2_error':None,'trace_reference_error':None,
                                 'l2_bound':None,'validation_reconstruction_ms':0.,
                                 'profile_note':'Short, residual-only replay; convergence and PDE error are not assessed.'}
                    else:
                        checked=verify(spec,space,system,get,x,solver_converged=converged)
                    if spec.get('record_residual_history'):
                        if native:
                            counts['residual_history']=answer.true_residual_history.tolist()
                            counts['estimated_preconditioned_residual_history']=answer.estimated_preconditioned_residual_history.tolist()
                            counts['residual_history_kind']='absolute true residuals at restart boundaries'
                        else:
                            counts['residual_history']=[float(solver.get_residual(i)) for i in range(solver.iterations_number+1)]
                            counts['residual_history_kind']='AMGX reported absolute L2 residuals; final residual checked independently'
                    solves.append({'solve_ms':elapsed,**counts,**checked})
                    if not checked['passed']:break
                sample={'setup_ms':setup,'setup_stages':stages,'solves':solves,
                    'fresh_setup_solve_ms':setup+solves[0]['solve_ms'],
                    'amortized_ms':(setup+sum(s['solve_ms'] for s in solves))/len(solves),
                    'passed':len(solves)==spec['solves_per_setup'] and all(s['passed'] for s in solves),
                    'cupy_pool_used_bytes':int(cp.get_default_memory_pool().used_bytes()),
                    'cupy_pool_reserved_bytes':int(cp.get_default_memory_pool().total_bytes())}
                if pyamgx is not None:sample['amgx_memory']=pyamgx.get_device_memory_stats()
                (result['warmups'] if repetition<spec['warmup'] else result['samples']).append(sample)
                atomic_json(spec['result'],result)
                if not sample['passed']:
                    result['status']='numerical_failure'
                    return result
            finally:
                for obj in reversed(objects):obj.destroy()
                solver=op=base=pre=rhs=solution=None
                if 'answer' in locals():del answer
                gc.collect()
        np.save(Path(spec['result']).with_suffix('.solution.npy'),x,allow_pickle=False)
        result['status']='passed'
        for key in ('setup_ms','fresh_setup_solve_ms','amortized_ms'):
            values=[r[key] for r in result['samples']]
            result[key.replace('_ms','_median_ms')]=float(np.median(values))
        hot=[s['solve_ms'] for r in result['samples'] for s in r['solves'][1:]]
        result['hot_solve_median_ms']=float(np.median(hot))
        result['worst_true_relative_residual']=max(s['true_relative_residual'] for r in result['samples'] for s in r['solves'])
        result['l2_error']=max(s['l2_error'] for r in result['samples'] for s in r['solves'])
        result['peak_process_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['memory_note']='CuPy counters are post-solve pool snapshots; AMGX peaks are allocator-managed process peaks. Whole-device peak comparison requires separate profiling.'
        return result
    finally:
        if pyamgx is not None:pyamgx.finalize()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True)
    args=parser.parse_args();spec=read_json(args.spec)
    try:result=measure(spec)
    except Exception as exc:
        result=read_json(spec['result']) if Path(spec['result']).exists() else {}
        result.update(status='error',error=str(exc),traceback=traceback.format_exc())
    atomic_json(spec['result'],result)
    print(result['status'],result.get('fresh_setup_solve_median_ms'),flush=True)
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
