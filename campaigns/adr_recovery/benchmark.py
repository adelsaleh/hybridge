"""One degree per process, with synchronized phase and complete-call timings."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

from campaigns.common.lifecycle import log
from campaigns.adr_recovery.problem import mesh, chunk_mesh, problem
from hybridge.runtime.benchmarking import measure_synchronized
from hybridge.runtime.cuda_tuning import CandidateUnavailable, device_signature
from hybridge.runtime.optional import require_cupy_device
from hybridge.mixed.postprocess.adr_recovery import ADRRecoveryWorkspace, recovery_working_bytes
from hybridge.io.records import append_jsonl_record


class PhaseRecorder:
    """Separate instrumented stream/wall timings from uninstrumented totals."""
    def __init__(self, cp):
        self.cp, self.rows = cp, []

    def __call__(self, name, function):
        cp = self.cp
        cp.cuda.get_current_stream().synchronize()
        first, last = cp.cuda.Event(), cp.cuda.Event()
        start = time.perf_counter()
        first.record()
        value = function()
        last.record()
        last.synchronize()
        self.rows.append(dict(phase=name, wall_seconds=time.perf_counter()-start,
                              stream_seconds=cp.cuda.get_elapsed_time(first, last)/1000.))
        return value


def choose_cells(order, free_bytes, maximum=8192):
    budget = min(24*1024**3, int(free_bytes*.35))
    count = min(maximum, budget//recovery_working_bytes(order, 1))
    return max(0, int(count)//128*128)


def choose_mesh(chunk_cells, seconds_per_chunk, remaining_seconds, *, min_nx=192, max_nx=512, repeats=3):
    # Reserve half the worker budget for qualification/search, setup and report.
    cells = remaining_seconds*.45*chunk_cells/max(seconds_per_chunk*repeats, 1e-6)
    nx = min(max_nx, max(min_nx, int(math.sqrt(cells/2))))
    return max(min_nx, nx//16*16)


def relative_error(cp, actual, reference):
    return float((cp.max(cp.abs(actual-reference))/cp.maximum(cp.max(cp.abs(reference)), 1.)).item())


def backward_error(cp, a, b, x):
    r = cp.max(cp.abs((a @ x[..., None])[..., 0]-b), axis=1)
    scale = cp.max(cp.sum(cp.abs(a), axis=2), axis=1)*cp.max(cp.abs(x), axis=1)+cp.max(cp.abs(b), axis=1)
    return float(cp.max(r/cp.maximum(scale, np.finfo(float).tiny)).item())


class DegreeBenchmark:
    def __init__(self, order, output, *, seconds=1000, repeats=3, max_nx=512, min_nx=192, chunk_limit=8192):
        self.order, self.output = order, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.cp = require_cupy_device()
        self.started, self.deadline = time.monotonic(), time.monotonic()+seconds
        self.repeats, self.max_nx, self.min_nx = repeats, max_nx, min_nx
        free, total = self.cp.cuda.runtime.memGetInfo()
        self.chunk_cells = choose_cells(order, free, chunk_limit)
        if self.chunk_cells < 512:
            raise RuntimeError('Insufficient GPU memory for a representative recovery chunk')
        self.results = dict(order=order, device=device_signature(self.cp), free_bytes=free, total_bytes=total,
            chunk_cells=self.chunk_cells, memory_estimate_bytes=recovery_working_bytes(order,self.chunk_cells),
            repeats=repeats, assembly=[], solvers=[], pipelines=[], chunks=[], status='running',
            input_scope='deterministic mixed fields and trace on distorted affine triangles; no global PDE solve',
            cache_scope='fixed geometry/maps reused or rebuilt; immutable reference basis retained in both; all PDE-dependent matrices refreshed',
            timing_scope='FP64; synchronized wall and stream phases; JIT/warmups separate; full totals without phase barriers')

    def emit(self, kind, **data):
        row = dict(kind=kind, order=self.order, elapsed_seconds=time.monotonic()-self.started, **data)
        append_jsonl_record(self.output/'progress.jsonl', row)
        log(json.dumps(row, sort_keys=True))

    def check_time(self, reserve=5):
        if time.monotonic()+reserve > self.deadline:
            raise TimeoutError('Degree budget exhausted; saved records remain usable, degree is incomplete')

    def measure(self, label, function, *, repeats=None):
        self.check_time()
        return measure_synchronized(function, synchronize=self.cp.cuda.get_current_stream().synchronize,
            repeats=repeats or self.repeats, warmups=1,
            on_sample=lambda i,s: self.emit('sample', phase=label, sample=i, seconds=s))

    def inputs(self, geometry):
        begin = time.perf_counter()
        p = problem(self.order, geometry)
        p['local'] = self.cp.asarray(p['local'])
        p['trace'] = self.cp.asarray(p['trace'])
        self.cp.cuda.get_current_stream().synchronize()
        self.emit('inputs', elements=geometry.num_tri, seconds=time.perf_counter()-begin)
        return p

    def args(self, p):
        return p['local'],p['trace'],p['beta'],p['prepared'],None,p['diffusion']

    def search(self):
        from hybridge.mixed.postprocess.total_flux import _adr_postprocess_samples, _postprocess_total_flux
        from hybridge.mixed.postprocess.primal_raw_cuda import primal_system_raw_cuda, _kernel
        from hybridge.mixed.coefficients import sample_diffusion_tensor, inverse_diffusion_values
        from hybridge.core.device import as_cupy_coefficients, as_cupy_space
        from hybridge.linalg.gpu.tuned_batched_solve import solve_selected
        from hybridge.linalg.gpu.cublas_batched import BatchedLUWorkspace
        from hybridge.linalg.gpu.fused_batched_solve import default_config
        cp = self.cp
        # A real, nonuniform chunk at the same physical cell size as the large mesh.
        self.full_candidate = mesh(self.max_nx)
        wide = chunk_mesh(self.full_candidate, 0, self.chunk_cells)
        p = self.inputs(wide)
        setup = PhaseRecorder(cp)
        workspace = ADRRecoveryWorkspace(p['space'],p['trace_space'],phase=setup)
        workspace.phase = lambda name, fn: fn()
        self.results['setup_pilot'] = setup.rows
        self.results['flux_cache_bytes'] = workspace.flux.bytes
        self.results['flux_constraint_map_defect'] = workspace.flux.constraint_map_defect
        s,t,cache = p['space'],p['trace_space'],workspace.cache
        samples = _adr_postprocess_samples(p['beta'],p['prepared'],s,cache.post_space,None,xp=cp)
        previous = getattr(s,'_adr_recovery_cache',None)
        s._adr_recovery_cache=cache
        try:
            flux = _postprocess_total_flux(p['local'],p['trace'],p['beta'],p['prepared'],s,t,None,
                'l2_closest','cupy',samples=samples,recovery=workspace.flux)
        finally:
            s._adr_recovery_cache=previous
        inverse = inverse_diffusion_values(sample_diffusion_tensor(p['diffusion'],cache.post_space,device=True))
        values = cp.stack([as_cupy_coefficients(f,as_cupy_space(f.space)) for f in flux.components],axis=1)
        reference_a,reference_b = workspace.tensor.assemble(p['local'],values,samples,inverse)
        candidates = [(engine,threads) for engine in ('raw','compact') for threads in (32,64,128,256,512)] + [('tensor',None)]
        # Rotate the first candidate by degree to reduce systematic warm-cache/order bias.
        shift=self.order%len(candidates); candidates=candidates[shift:]+candidates[:shift]
        for engine,threads in candidates:
            self.check_time()
            label=f'{engine}:{threads or "gemm"}'
            try:
                run = (lambda:workspace.tensor.assemble(p['local'],values,samples,inverse)) if engine=='tensor' else (
                    lambda:primal_system_raw_cuda(p['local'],flux,s,cache,samples,inverse,
                                                   threads=threads,compact=engine=='compact'))
                result,timing = self.measure('assembly/'+label,run)
                err_a,err_b=relative_error(cp,result[0],reference_a),relative_error(cp,result[1],reference_b)
                if max(err_a,err_b)>3e-10:
                    raise ArithmeticError(f'Assembly parity failed {label}: {err_a}, {err_b}')
                rec=dict(engine=engine,threads=threads,**timing,matrix_relative_error=err_a,rhs_relative_error=err_b,status='passed')
                if engine!='tensor':
                    q=cache.post_space.quad_data
                    kernel=_kernel(cache.post_space.el_dof,q.edg_dof,len(q.Krf_w),len(q.weights_JGL),s.el_dof,engine=='compact')
                    rec['attributes']=kernel.attributes
                    try:
                        active=cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(kernel.kernel.ptr,threads,0)
                        rec.update(active_blocks_per_sm=active,modeled_occupancy=active*threads/2048.)
                    except Exception as exc:
                        rec['occupancy_unavailable']=str(exc)
                result=None
            except CandidateUnavailable as exc:
                rec=dict(engine=engine,threads=threads,status='resource_rejected',reason=str(exc))
            self.results['assembly'].append(rec);self.emit('assembly_result',**rec)
        n=reference_a.shape[1]
        variants=list(dict.fromkeys(['cublas','magma']+['fused:'+':'.join(map(str,c)) for c in (
            default_config(n),(8,8,2,1),(8,8,2,3),(8,4,4,2))]))
        for variant in variants:
            self.check_time()
            # Restore operands outside the solver-only timing. Packing is measured separately.
            a=cp.ascontiguousarray(reference_a.transpose(0,2,1)); b=reference_b[:,None,:].copy()
            info=cp.empty(s.mesh.num_tri,dtype=cp.int32); lu=BatchedLUWorkspace()
            def one():
                a[...] = reference_a.transpose(0,2,1); b[:,0,:] = reference_b
                timer=PhaseRecorder(cp)
                timer('solve',lambda:solve_selected(variant,a,b,info=info,workspace=lu))
                if bool(cp.any(info).item()) or not bool(cp.all(cp.isfinite(b)).item()):
                    raise ArithmeticError('Dense solver status/nonfinite failure')
                return timer.rows[0]
            try:
                warm=one(); measurements=[one() for _ in range(self.repeats)]
                residual=backward_error(cp,reference_a,reference_b,b[:,0,:])
                if not np.isfinite(residual) or residual>5e-11: raise ArithmeticError(f'Dense backward error {residual}')
                rec=dict(variant=variant,status='passed',backward_error=residual,
                    median_seconds=statistics.median(r['wall_seconds'] for r in measurements),samples=measurements,warmup=warm)
            except (CandidateUnavailable,RuntimeError,ValueError) as exc:
                # Missing/illegal optional candidates are explicit; no silent replacement.
                rec=dict(variant=variant,status='unavailable',reason=str(exc))
            self.results['solvers'].append(rec);self.emit('solver_result',**rec)
            del a,b,info,lu
        if not any(r['variant']=='magma' and r['status']=='passed' for r in self.results['solvers']):
            raise RuntimeError('Required MAGMA comparison did not pass')
        best_solver=min((r for r in self.results['solvers'] if r['status']=='passed'),key=lambda r:r['median_seconds'])['variant']
        best_by_engine=[min((r for r in self.results['assembly'] if r['engine']==engine and r['status']=='passed'),
                            key=lambda r:r['median_seconds']) for engine in ('raw','compact','tensor')]
        # End-to-end comparison includes original and optimized flux paths, both lifecycle policies.
        reference=None
        for policy in ('reuse','rebuild'):
            for candidate in best_by_engine:
                engine,threads=candidate['engine'],candidate['threads']
                for cached in (False,True):
                    self.check_time()
                    config=dict(assembly=engine,threads=threads or 128,solver=best_solver,cached_flux=cached)
                    fixed=ADRRecoveryWorkspace(s,t,**config)
                    def run():
                        work=fixed if policy=='reuse' else ADRRecoveryWorkspace(s,t,**config,reference_cache=fixed.cache)
                        return work.recover(*self.args(p))
                    field,timing=self.measure(f'pipeline/{policy}/{engine}/flux-cache-{cached}',run)
                    coeff=as_cupy_coefficients(field,as_cupy_space(field.space))
                    if reference is None: reference=coeff.copy()
                    err=relative_error(cp,coeff,reference)
                    if err>3e-8: raise ArithmeticError(f'Pipeline parity defect {err}')
                    mean=float(cp.max(cp.abs(coeff @ cp.asarray(cache.mean_post)-p['local'][:,:s.el_dof] @ cp.asarray(cache.mean_base))).item())
                    if mean>2e-9: raise ArithmeticError(f'Mean defect {mean}')
                    rec=dict(policy=policy,config=config,**timing,coefficient_relative_error=err,mean_defect=mean)
                    self.results['pipelines'].append(rec);self.emit('pipeline_result',**rec)
                    del fixed,field,coeff
        winners={policy:min((r for r in self.results['pipelines'] if r['policy']==policy),key=lambda r:r['median_seconds'])
                 for policy in ('reuse','rebuild')}
        original={'assembly':'raw','threads':next(r['threads'] for r in best_by_engine if r['engine']=='raw'),
                  'solver':best_solver,'cached_flux':False,'reuse_samples':False}
        self.results.update(winners=winners,baseline_config=original)
        return sum(r['median_seconds'] for r in winners.values())+2*max(r['median_seconds'] for r in self.results['pipelines'])

    def sweep(self):
        from hybridge.core.device import as_cupy_coefficients,as_cupy_space
        from hybridge.core.transfer import project_same_mesh_field
        estimate=self.search()
        gc.collect();self.cp.get_default_memory_pool().free_all_blocks()
        nx=choose_mesh(self.chunk_cells,estimate,self.deadline-time.monotonic(),min_nx=self.min_nx,max_nx=self.max_nx,repeats=self.repeats)
        full=self.full_candidate if nx==self.max_nx else mesh(nx)
        self.full_candidate=None
        self.results.update(mesh_nx=nx,elements=full.num_tri,
            mesh_sha256=hashlib.sha256(full.node_coords.tobytes()+full.triangles.tobytes()).hexdigest())
        self.emit('mesh_selected',nx=nx,elements=full.num_tri,chunk_cells=self.chunk_cells,estimated_chunk_seconds=estimate)
        totals={f'{variant}/{policy}':[] for variant in ('baseline','optimized') for policy in ('reuse','rebuild')}
        for first in range(0,full.num_tri,self.chunk_cells):
            self.check_time()
            p=self.inputs(chunk_mesh(full,first,min(first+self.chunk_cells,full.num_tri)))
            expected=None
            for variant in ('baseline','optimized'):
                for policy in ('reuse','rebuild'):
                    config=self.results['baseline_config'] if variant=='baseline' else self.results['winners'][policy]['config']
                    setup=PhaseRecorder(self.cp)
                    fixed=ADRRecoveryWorkspace(p['space'],p['trace_space'],**config,phase=setup)
                    fixed.phase=lambda name,fn:fn()
                    def run():
                        workspace=fixed if policy=='reuse' else ADRRecoveryWorkspace(p['space'],p['trace_space'],**config,reference_cache=fixed.cache)
                        field=workspace.recover(*self.args(p))
                        return project_same_mesh_field(field,p['space'])
                    result,timing=self.measure(f'full/{first}/{variant}/{policy}',run)
                    coeff=as_cupy_coefficients(result,as_cupy_space(result.space))
                    if expected is None: expected=coeff.copy()
                    err=relative_error(self.cp,coeff,expected)
                    if err>5e-8: raise ArithmeticError(f'Full-mesh projected-source parity defect {err}')
                    weights=self.cp.asarray(p['space'].quad_data.Krf_w @ p['space'].quad_data.phi)
                    mean_error=relative_error(self.cp,coeff @ weights,p['local'][:,:p['space'].el_dof] @ weights)
                    if mean_error>2e-9: raise ArithmeticError(f'Full-mesh mean defect {mean_error}')
                    # One separate instrumented call, including rebuild setup when applicable.
                    profiler=PhaseRecorder(self.cp)
                    work=fixed if policy=='reuse' else ADRRecoveryWorkspace(p['space'],p['trace_space'],**config,
                                                                            reference_cache=fixed.cache,phase=profiler)
                    work.phase=profiler
                    f=work.recover(*self.args(p))
                    profiler('projection.to_primary',lambda:project_same_mesh_field(f,p['space']))
                    rec=dict(first=first,cells=p['space'].mesh.num_tri,variant=variant,policy=policy,config=config,
                        **timing,setup=setup.rows,profile=profiler.rows,coefficient_relative_error=err,mean_relative_defect=mean_error,
                        device_pool_used=self.cp.get_default_memory_pool().used_bytes(),
                        device_pool_reserved=self.cp.get_default_memory_pool().total_bytes(),
                        cache_bytes=0 if work.flux is None else work.flux.bytes)
                    totals[f'{variant}/{policy}'].append(timing['samples_seconds'])
                    self.results['chunks'].append(rec);self.emit('chunk_result',**rec)
                    del result,coeff,f,work,fixed
            del p,expected
            gc.collect();self.cp.get_default_memory_pool().free_all_blocks()
            self.save()
        self.results['full_mesh_totals']={name:dict(samples_seconds=np.sum(parts,axis=0).tolist(),
            median_seconds=float(np.median(np.sum(parts,axis=0)))) for name,parts in totals.items()}
        self.results.update(status='complete',elapsed_seconds=time.monotonic()-self.started)
        self.save()
        self.emit('degree_complete',elements=full.num_tri,totals=self.results['full_mesh_totals'])

    def save(self):
        path=self.output/'result.json'; temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.results,indent=2,default=str)+'\n');temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--order',type=int,choices=range(2,10),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=1000)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--max-nx',type=int,default=512)
    parser.add_argument('--min-nx',type=int,default=192)
    parser.add_argument('--chunk-limit',type=int,default=8192)
    a=parser.parse_args()
    runner=DegreeBenchmark(a.order,a.output,seconds=a.seconds,repeats=a.repeats,max_nx=a.max_nx,
                           min_nx=a.min_nx,chunk_limit=a.chunk_limit)
    try:
        runner.sweep()
    except BaseException as exc:
        runner.results.update(status='failed',error=repr(exc),elapsed_seconds=time.monotonic()-runner.started)
        runner.save()
        raise


if __name__=='__main__': main()
