"""Isolated workers for assembly, solver timing, CUDA profiling and end-to-end checks.

Invoked by run_adv_diff_rea_performance.py; worker JSON is explicit, never pickle.
All primary timing samples rebuild setup and solve from x0=0. CUDA event
instrumentation is confined to separate profile jobs.
"""
from __future__ import annotations
from dataclasses import fields
from pathlib import Path
import argparse
import gc
import os
import platform
import sys
from time import perf_counter
import traceback
import numpy as np
from scripts.adr_performance_common import (Configuration,atomic_json,read_json,estimates,solver_workspace_degree)
from scripts.adv_diff_rea_cases import get_case
from hdgfem.core.mesh import rectangle_mesh
from hdgfem.core.space import DGSpace
from hdgfem.assembly import hdg
from hdgfem.assembly.face_dense import FaceDenseSystem,face_dense_matvec,expand_eliminated_solution
from hdgfem.solvers.adv_diff_rea import assemble_advection_diffusion_reaction,_csr


class BudgetExceeded(RuntimeError):
    pass


def memory_available():
    available=None
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):available=int(line.split()[1])*1024
        # Respect a smaller cgroup allocation when present.
        limit=Path('/sys/fs/cgroup/memory.max')
        used=Path('/sys/fs/cgroup/memory.current')
        if limit.exists() and limit.read_text().strip()!='max':
            cgroup_free=max(0,int(limit.read_text())-int(used.read_text()))
            available=min(available,cgroup_free) if available else cgroup_free
    except (OSError,ValueError):pass
    return available


def cupy(spec):
    from hdgfem.backends.cupy import require_cupy_device
    cp=require_cupy_device()
    cp.cuda.Device(spec['device']).use()
    return cp


def guard(spec,phase):
    e=estimates(spec['n'],spec['p'],solver_workspace_degree(spec))
    host=e['assembly_host_bytes'] if phase in ('assembly','pipeline') else e['cache_disk_bytes']+2*e['solver_device_bytes']
    limit=spec.get('host_limit_gib')
    if limit is not None and host>limit*2**30:
        raise BudgetExceeded(f'host planning estimate {host/2**30:.2f} GiB exceeds explicit limit {limit} GiB')
    available=memory_available()
    if available is not None and host>spec['memory_fraction']*available:
        raise BudgetExceeded(f'host planning estimate {host/2**30:.2f} GiB exceeds budget from live available RAM {available/2**30:.2f} GiB')
    if spec['engine']=='gpu':
        cp=cupy(spec)
        free,total=cp.cuda.runtime.memGetInfo()
        need=e['assembly_device_bytes'] if phase=='assembly' and spec['assembly_backend']=='cupy' else e['solver_device_bytes']
        if phase=='assembly' and spec['assembly_backend'] in ('numpy','numba'):need=0
        if phase=='pipeline' and spec['assembly_backend']=='cupy':need=max(e['assembly_device_bytes'],e['solver_device_bytes'])
        if need>spec['memory_fraction']*free:
            raise BudgetExceeded(f'GPU planning estimate {need/2**30:.2f} GiB exceeds budget from live free VRAM {free/2**30:.2f} GiB')
    return e


def metadata(spec):
    import scipy
    value={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
           'platform':platform.platform(),'pid':os.getpid(),
           'thread_env':{key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                                                           'NUMBA_NUM_THREADS','NUMBA_THREADING_LAYER')}}
    if spec['engine']=='gpu':
        cp=cupy(spec)
        prop=cp.cuda.runtime.getDeviceProperties(spec['device'])
        name=prop['name']
        value.update(gpu=name.decode() if isinstance(name,bytes) else name,
            total_vram_bytes=int(prop['totalGlobalMem']),device=spec['device'],cupy=cp.__version__,
            cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),driver=cp.cuda.runtime.driverGetVersion())
    return value


def get_space(spec):
    """Construct the recorded rectangular or explicitly supplied triangular mesh."""
    if spec.get('mesh_path'):
        with np.load(spec['mesh_path'],allow_pickle=False) as arrays:
            mesh=(arrays['node_coords'],arrays['triangles'])
    else:
        mesh=rectangle_mesh(spec['n'],spec['n'])
    return DGSpace(mesh,spec['p'],basis_type='dub_orth',
        volume_quad_1d=spec.get('volume_quad_1d'),edge_quad_1d=spec.get('edge_quad_1d'))


def relative(a,b):
    return float(np.linalg.norm((a-b).ravel())/max(np.linalg.norm(b.ravel()),1e-300))


def reconstruct(space,system,local_inverse,boundary,source_rhs,x,case):
    trace=expand_eliminated_solution(x,system)
    unknowns=hdg.reconstruct_local_unknowns(trace,source_rhs,local_inverse,boundary,space)
    coeff=unknowns.reshape(space.mesh.num_tri,3,space.el_dof)
    _,exact=get_case(case)
    l2=space.field(coeff[:,0]).l2_error(exact)
    return trace,float(l2)


def save_cache(cache,assembly,space):
    cache.mkdir(parents=True,exist_ok=True)
    for f in fields(FaceDenseSystem):
        if f.name!='mode':np.save(cache/('system_'+f.name+'.npy'),getattr(assembly.system,f.name),allow_pickle=False)
    for name,value in [('element_blocks',assembly.element_blocks),('local_inverse',assembly.local_solver),
                       ('boundary',assembly.boundary),('source_rhs',assembly.source_rhs),
                       ('loc2glob',space.mesh.loc2glob_edge)]:
        np.save(cache/(name+'.npy'),value,allow_pickle=False)


def load_cache(cache):
    cache=Path(cache)
    get=lambda n:np.load(cache/(n+'.npy'),mmap_mode='r',allow_pickle=False)
    system=FaceDenseSystem(**{f.name:('eliminate' if f.name=='mode' else get('system_'+f.name)) for f in fields(FaceDenseSystem)})
    return system,get


def assemble(spec):
    estimate=guard(spec,'assembly')
    cache=Path(spec['cache'])
    cache.mkdir(parents=True,exist_ok=True)
    if __import__('shutil').disk_usage(cache).free<1.2*estimate['cache_disk_bytes']:
        raise BudgetExceeded('not enough free disk for uncompressed assembly cache')
    kw,exact=get_case(spec['case'])
    numba_info={}
    if spec['assembly_backend']=='numba':
        import numba
        if numba.config.DISABLE_JIT:
            raise RuntimeError('Numba campaign assembly requires NUMBA_DISABLE_JIT=0; interpreted execution is not a performance run')
        from hdgfem.kernels.adr_local import warmup
        print('Numba CPU assembly: warming cached parallel FP64 kernels (outside assembly timing)',flush=True)
        t=perf_counter();warmup()
        numba_info=dict(numba_warmup_ms=1000*(perf_counter()-t),numba_threads=numba.get_num_threads(),
                        numba_threading_layer=numba.threading_layer(),numba_version=numba.__version__,
                        numba_fastmath=False)
        print(f'Numba CPU assembly ready: {numba_info}',flush=True)
    t=perf_counter();space=get_space(spec);space_ms=1000*(perf_counter()-t)
    samples=[]
    for i in range(spec['assembly_warmup']+spec['assembly_repeats']):
        a=assemble_advection_diffusion_reaction(space=space,backend=spec['assembly_backend'],**kw)
        if i>=spec['assembly_warmup']:samples.append({k:1000*v for k,v in a.timings.items()})
        if i<spec['assembly_warmup']+spec['assembly_repeats']-1:
            del a;gc.collect()
    validation={}
    if a.system.num_dofs<=spec['reference_max_dofs']:
        ref=a
        if spec['assembly_backend'] in ('cupy','numba'):
            ref=assemble_advection_diffusion_reaction(space=space,backend='numpy',**kw)
            prefix='cpu_gpu' if spec['assembly_backend']=='cupy' else 'numpy_numba'
            validation[prefix+'_blocks_relative_error']=relative(a.system.blocks,ref.system.blocks)
            validation[prefix+'_rhs_relative_error']=relative(a.system.rhs,ref.system.rhs)
            if max(validation.values())>1e-9:raise RuntimeError('Assembly disagrees with independent NumPy assembly')
        from scipy.sparse.linalg import spsolve
        x=spsolve(_csr(ref.system),ref.system.rhs.reshape(-1))
        rn=np.linalg.norm(face_dense_matvec(ref.system.blocks,ref.system.neighbors,x).reshape(-1)-ref.system.rhs.reshape(-1))
        if not np.isfinite(rn) or rn>spec['rtol']*np.linalg.norm(ref.system.rhs):
            raise RuntimeError('CPU direct reference failed true-residual check')
        trace,l2=reconstruct(space,ref.system,ref.local_solver,ref.boundary,ref.source_rhs,x,spec['case'])
        np.save(cache/'reference_trace.npy',trace,allow_pickle=False)
        validation.update(reference_l2=l2,reference_residual_norm=float(rn))
        if ref is not a:del ref
    save_cache(cache,a,space)
    result={'status':'passed','kind':'assembly','environment':metadata(spec),'estimates':estimate,
        **numba_info,
        'space_setup_ms':space_ms,'assembly_samples_ms':samples,'validation':validation,
        'assembly_median_ms':float(np.median([s['assembly_total'] for s in samples])),
        'arrays':{p.name:p.stat().st_size for p in cache.glob('*.npy')}}
    atomic_json(cache/'metadata.json',result)
    return result


def build_gpu(spec,system,element_blocks,loc2glob):
    cp=cupy(spec);c=Configuration(**spec['configuration'])
    from hdgfem.backends.cupy_face_dense import CuPyFaceDenseOperator
    from hdgfem.backends.cupy_preconditionners import CuPyFaceBlockJacobiPreconditioner,CuPyFaceAdditiveSchwarzPreconditioner
    from hdgfem.backends.cupy_polynomial import CuPyPolynomialPreconditioner
    from hdgfem.backends.cupy_solver import CuPyProductionGMRESSolver,CuPyProductionGMRESOptions
    times={}
    def timed(name,fn):
        cp.cuda.get_current_stream().synchronize();t=perf_counter();v=fn()
        cp.cuda.get_current_stream().synchronize();times[name]=1000*(perf_counter()-t);return v
    op=timed('operator_setup_ms',lambda:CuPyFaceDenseOperator.from_system(system,implementation=c.operator,dtype=np.float64,device_id=spec['device']))
    base=None;times['base_preconditioner_setup_ms']=0.
    if c.preconditioner.startswith('block_jacobi'):
        base=timed('base_preconditioner_setup_ms',lambda:CuPyFaceBlockJacobiPreconditioner.from_system(system,
            device_id=spec['device'],local_solver=c.local_solver,application=c.application))
    elif c.preconditioner.startswith('asm'):
        base=timed('base_preconditioner_setup_ms',lambda:CuPyFaceAdditiveSchwarzPreconditioner.from_system(system,
            element_blocks,loc2glob,device_id=spec['device'],local_solver=c.local_solver,application=c.application))
    pre=base;times['polynomial_setup_ms']=0.
    if c.polynomial_degree is not None:
        pre=timed('polynomial_setup_ms',lambda:CuPyPolynomialPreconditioner.from_operator(op,
            degree=c.polynomial_degree,base_preconditioner=base,seed=1729,
            setup_orthogonalization=c.polynomial_orthogonalization))
    solver=timed('gmres_workspace_setup_ms',lambda:CuPyProductionGMRESSolver(op,preconditioner=pre,
        options=CuPyProductionGMRESOptions(restart=c.restart,max_iterations=spec['maxiter'],rtol=spec['rtol'],
            atol=0.,orthogonalization=c.orthogonalization,raise_on_failure=False,cgs2_fallback_threshold=None)))
    rhs=timed('rhs_transfer_ms',lambda:cp.asarray(system.rhs.reshape(-1)))
    return solver,op,base,pre,rhs,times


def verify(spec,space,system,get,x,*,solver_converged=True):
    t=perf_counter()
    rhs=system.rhs.reshape(-1)
    residual=face_dense_matvec(system.blocks,system.neighbors,x).reshape(-1)-rhs
    rn,bn=float(np.linalg.norm(residual)),float(np.linalg.norm(rhs))
    trace,l2=reconstruct(space,system,get('local_inverse'),get('boundary'),get('source_rhs'),x,spec['case'])
    passed=solver_converged and np.isfinite(rn) and rn<=spec['rtol']*bn and np.isfinite(l2)
    if spec['case'] in ('quadratic','variable_velocity') and spec['p']>=2:
        passed=passed and l2<2e-8
    ref_file=Path(spec['cache'])/'reference_trace.npy'
    trace_error=None
    if ref_file.exists():
        trace_error=relative(trace,np.load(ref_file,mmap_mode='r',allow_pickle=False))
        passed=passed and trace_error<=max(1e-8,100*spec['rtol'])
    bound=spec.get('l2_bound')
    if bound is not None:passed=passed and l2<=bound
    return dict(passed=bool(passed),true_residual_norm=rn,true_relative_residual=rn/bn if bn else rn,
        l2_error=l2,trace_reference_error=trace_error,l2_bound=bound,
        validation_reconstruction_ms=1000*(perf_counter()-t))


def one_solve(spec,space,system,get):
    c=Configuration(**spec['configuration'])
    if spec['engine']=='cpu':
        from scipy.sparse.linalg import gmres,LinearOperator
        t=perf_counter()
        a=LinearOperator((system.num_dofs,system.num_dofs),dtype=np.float64,
            matvec=lambda x:face_dense_matvec(system.blocks,system.neighbors,x).reshape(-1))
        setup=1000*(perf_counter()-t);iterations=0
        def callback(_):
            nonlocal iterations
            iterations+=1
        t=perf_counter()
        x,info=gmres(a,system.rhs.reshape(-1),rtol=spec['rtol'],atol=0.,restart=c.restart,
                     maxiter=spec['maxiter'],callback=callback,callback_type='legacy')
        elapsed=1000*(perf_counter()-t)
        return {**verify(spec,space,system,get,x,solver_converged=info==0),
            'setup_ms':setup,'solve_ms':elapsed,'setup_solve_ms':setup+elapsed,'iterations':iterations,
            'solver_status':'converged' if info==0 else str(info)}
    cp=cupy(spec)
    cp.cuda.get_current_stream().synchronize();t0=perf_counter()
    solver,op,base,pre,rhs,times=build_gpu(spec,system,get('element_blocks'),get('loc2glob'))
    cp.cuda.get_current_stream().synchronize();setup=1000*(perf_counter()-t0)
    t=perf_counter();r=solver.solve(rhs,x0=None,raise_on_failure=False)
    cp.cuda.get_current_stream().synchronize();elapsed=1000*(perf_counter()-t)
    combined=1000*(perf_counter()-t0)
    t=perf_counter();x=cp.asnumpy(r.solution);transfer=1000*(perf_counter()-t)
    checked=verify(spec,space,system,get,x,solver_converged=r.converged)
    # Validate the operator against the CPU face-dense action, outside timings.
    rng=np.random.default_rng(1729);probe=rng.standard_normal(system.num_dofs)
    probe_d=cp.asarray(probe);out=cp.empty_like(probe_d);op.matvec_into(probe_d,out)
    operator_error=relative(cp.asnumpy(out),face_dense_matvec(system.blocks,system.neighbors,probe).reshape(-1))
    checked['passed']=checked['passed'] and operator_error<=1e-11
    return {**times,**checked,'setup_ms':setup,'solve_ms':elapsed,'setup_solve_ms':combined,
        'solution_to_host_ms':transfer,'operator_cpu_relative_error':operator_error,
        'iterations':int(r.iterations),'solver_status':r.status,'termination_reason':r.termination_reason,
        'actual_orthogonalization':r.orthogonalization,'fallback_count':int(r.fallback_count),
        'matvec_count':int(r.matvec_count),'preconditioner_count':int(r.preconditioner_count),
        'gmres_workspace_bytes':int(solver.workspace_device_bytes),
        'pool_used_bytes_after_solve':int(cp.get_default_memory_pool().used_bytes()),
        'pool_reserved_bytes_after_solve':int(cp.get_default_memory_pool().total_bytes()),
        'base_inverse_residual':getattr(base,'maximum_inverse_residual',None),
        'residual_history':r.true_residual_history.tolist()}


def measure(spec):
    estimate=guard(spec,'solver')
    system,get=load_cache(spec['cache']);space=get_space(spec)
    result={'status':'running','kind':'candidate','configuration':spec['configuration'],
            'requested_repeats':spec['repeats'],'samples':[],'warmups':[],
            'environment':metadata(spec),'estimates':estimate}
    for i in range(spec['warmup']+spec['repeats']):
        sample=one_solve(spec,space,system,get)
        if i<spec['warmup']:result['warmups'].append(sample)
        else:result['samples'].append(sample)
        atomic_json(spec['result'],result)
        gc.collect()
        if not sample['passed']:
            result['status']='numerical_failure';return result
    result['status']='passed'
    for key in ('setup_ms','solve_ms','setup_solve_ms','validation_reconstruction_ms',
                'operator_setup_ms','base_preconditioner_setup_ms','polynomial_setup_ms','rhs_transfer_ms'):
        vals=[s[key] for s in result['samples'] if key in s]
        if vals:
            stem=key[:-3]
            for label,fn in [('median',np.median),('min',np.min),('max',np.max)]:
                result[f'{stem}_{label}_ms']=float(fn(vals))
    result['l2_error']=max(s['l2_error'] for s in result['samples'])
    result['worst_true_relative_residual']=max(s['true_relative_residual'] for s in result['samples'])
    return result


def profile(spec):
    guard(spec,'solver')
    system,get=load_cache(spec['cache'])
    if spec['engine']=='cpu':
        import cProfile,pstats
        prof=cProfile.Profile();prof.runcall(one_solve,spec,get_space(spec),system,get)
        prof.dump_stats(str(Path(spec['result']).with_suffix('.prof')))
        with Path(spec['result']).with_suffix('.txt').open('w') as f:
            pstats.Stats(prof,stream=f).sort_stats('cumulative').print_stats(40)
        return {'status':'passed','kind':'profile','engine':'cpu','note':'CPU orchestration check; no CUDA measurements'}
    from hdgfem.backends.cupy_profiling import (profile_face_dense_operator,profile_additive_schwarz,
        profile_block_jacobi,benchmark_cuda_call,CuPyGMRESProfiler)
    cp=cupy(spec)
    solver,op,base,pre,rhs,_=build_gpu(spec,system,get('element_blocks'),get('loc2glob'))
    # Warm every solver and polynomial path before profiling.
    warm=solver.solve(rhs,raise_on_failure=False)
    if not warm.converged:raise RuntimeError('profiling warmup did not converge')
    c=Configuration(**spec['configuration']);out=cp.empty_like(rhs)
    # Representative deterministic vector independent of the PDE RHS.
    x=cp.asarray(np.random.default_rng(1729).standard_normal(system.num_dofs))
    kw=dict(warmup=spec['component_warmup'],repeats=spec['component_repeats'])
    data={'status':'passed','kind':'profile','configuration':spec['configuration'],
          'environment':metadata(spec),'units':'milliseconds (CUDA events)',
          'operator':profile_face_dense_operator(op,x,out,**kw).to_dict()}
    if base is not None:
        data['base_preconditioner']=(profile_additive_schwarz(base,x,out,**kw) if c.preconditioner.startswith('asm')
                                    else profile_block_jacobi(base,x,out,**kw)).to_dict()
    if pre is not None:
        data['complete_preconditioner']=benchmark_cuda_call(lambda:pre.apply_into(x,out),**kw).to_dict(include_samples=True)
    profiler=CuPyGMRESProfiler(device_id=spec['device'])
    t=perf_counter();r=solver.solve(rhs,profiler=profiler,raise_on_failure=False)
    summary=profiler.finalize();data['instrumented_wall_ms']=1000*(perf_counter()-t)
    data['gmres']=summary.to_dict();data['instrumented_converged']=r.converged
    data['warning']='Instrumented timings are diagnostics and never used for ranking; zero substage times denote fusion.'
    checked=verify(spec,get_space(spec),system,get,cp.asnumpy(r.solution),solver_converged=r.converged)
    data['validation']=checked
    if not checked['passed']:data['status']='numerical_failure'
    return data


def pipeline(spec):
    guard(spec,'pipeline')
    from hdgfem.solvers.adv_diff_rea import solve_advection_diffusion_reaction
    space=get_space(spec);kw,exact=get_case(spec['case']);c=Configuration(**spec['configuration'])
    options={**c.gpu_options(),'device_id':spec['device']}
    samples=[]
    for i in range(spec['warmup']+spec['repeats']):
        t=perf_counter()
        r=solve_advection_diffusion_reaction(space=space,**kw,assembly_backend=spec['assembly_backend'],
            solver='gpu' if spec['engine']=='gpu' else 'gmres',gpu_options=options,rtol=spec['rtol'],
            maxiter=spec['maxiter'],restart=c.restart)
        elapsed=1000*(perf_counter()-t);l2=r.field.l2_error(exact)
        if not np.isfinite(l2) or (spec.get('l2_bound') is not None and l2>spec['l2_bound']):
            raise RuntimeError('end-to-end solution failed L2 validation')
        if spec['case'] in ('quadratic','variable_velocity') and spec['p']>=2 and l2>=2e-8:
            raise RuntimeError('end-to-end solution failed polynomial exactness')
        if i>=spec['warmup']:
            samples.append(dict(end_to_end_ms=elapsed,l2_error=float(l2),true_relative_residual=r.relative_residual,
                timings_ms={k:1000*v for k,v in r.timings.items()}))
        del r;gc.collect()
    return dict(status='passed',kind='pipeline',samples=samples,environment=metadata(spec),
        end_to_end_median_ms=float(np.median([s['end_to_end_ms'] for s in samples])),
        note='Actual repeated end-to-end wall time excluding DGSpace creation and final L2 integration; includes true residual and reconstruction.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True)
    args=parser.parse_args();spec=read_json(args.spec)
    try:
        output={'assemble':assemble,'measure':measure,'profile':profile,'pipeline':pipeline,
                'environment':lambda s:dict(status='passed',environment=metadata(s))}[spec['task']](spec)
    except BudgetExceeded as exc:
        output={'status':'skipped_budget','error':str(exc)}
    except Exception as exc:
        output=read_json(spec['result']) if Path(spec['result']).exists() else {}
        output.update(status='error',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
    atomic_json(spec['result'],output)
    print(f"{spec['task']} {spec['case']} {spec['n']} p={spec['p']}: {output['status']}",flush=True)
    if output['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
