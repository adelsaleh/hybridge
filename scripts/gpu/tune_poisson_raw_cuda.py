#!/usr/bin/env python3
"""Tune/benchmark stationary Poisson kernels without a global solve or AMGX build.

Automatic samples time local construction/solves and isolated response writes;
the warmed full-mesh measurements below include direct sparse scatter. Hardware
occupancy values are CUDA resource-model predictions, not profiler counters.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import statistics
import sys
import time

if __package__ in {None, ''}:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import numpy as np

from hybridge import DGSpace, rectangle_mesh
from hybridge.core.device import as_cupy_space
from hybridge.hdg.cuda.tuning import device_signature
from hybridge.mixed.cupy import (build_trace_reference, reference_derivative_mats,
    face_element_mass, source_moments_cupy, boundary_trace_values_cupy)
from hybridge.mixed.raw_cuda import poisson
from hybridge.runtime.optional import require_cupy
from hybridge.runtime.precision import REAL_DTYPE


def _list(text,cast=str):
    return [cast(value.strip()) for value in text.split(',')]


def _measure(function,warmups,repeats,device_time):
    for _ in range(warmups):
        function()
    wall=[]
    device=[]
    result=None
    for _ in range(repeats):
        start=time.perf_counter()
        result=function()
        wall.append(time.perf_counter()-start)
        device.append(float(device_time(result)))
    return result,dict(wall_samples_seconds=wall,kernel_samples_seconds=device,
        wall_median_seconds=statistics.median(wall),kernel_median_seconds=statistics.median(device),
        kernel_min_seconds=min(device),kernel_max_seconds=max(device))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--orders',default='7,8,9')
    parser.add_argument('--mesh-sizes',default='16,32',help='structured square subdivisions, comma-separated')
    parser.add_argument('--trace-bases',default='legacy-lagrange,legendre-modal')
    parser.add_argument('--formats',default='coo,csr,bsr')
    parser.add_argument('--cache-policies',default='none,schur-lu,schur-cholesky')
    parser.add_argument('--warmups',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--refresh',action='store_true',help='remeasure matching persistent records')
    parser.add_argument('--thorough',action='store_true',help='test every legal configuration, with ten samples each')
    parser.add_argument('--include-legacy',action='store_true',help='compare the previous fused assembly where it fits')
    parser.add_argument('--block-size',default='auto',choices=['auto','32','64','128','256'])
    parser.add_argument('--output',type=Path,default=Path('artifacts/poisson_raw_cuda_tuning.json'))
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('-v','--verbosity',type=int,choices=range(4),default=1)
    args=parser.parse_args(argv)
    orders=_list(args.orders,int)
    sizes=_list(args.mesh_sizes,int)
    bases=_list(args.trace_bases)
    formats=_list(args.formats)
    policies=_list(args.cache_policies)
    if any(p<0 or p>9 for p in orders) or any(n<1 for n in sizes):
        parser.error('orders must be 0..9 and mesh sizes must be positive')
    if not set(bases)<={'legacy-lagrange','legendre-modal'} or not set(formats)<={'coo','csr','bsr'}:
        parser.error('unsupported trace basis or sparse format')
    if not set(policies)<={'none','schur-lu','schur-cholesky'} or args.repeats<1 or args.warmups<0:
        parser.error('invalid cache policy or repetition count')
    cases=list(itertools.product(orders,sizes,bases,formats,policies))
    if args.dry_run:
        print(json.dumps(dict(cases=cases,output=str(args.output),thorough=args.thorough),indent=2))
        return 0
    cp=require_cupy()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(device=device_signature(cp),kernel_source=poisson.SOURCE_VERSION,
        precision=np.dtype(REAL_DTYPE).name,arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        tuning_measurement='isolated local responses; full kernel timings below include direct scatter',
        occupancy='CUDA resource model; not measured achieved occupancy',cases=[])
    with args.output.with_suffix('.log').open('w',buffering=1) as log:
        def say(message):
            print(message,file=log,flush=True)
            if args.verbosity:
                print(message,flush=True)
        for index,(p,size,basis,fmt,policy) in enumerate(cases,1):
            say(f'[{index}/{len(cases)}] p={p} triangles={2*size*size} {basis} {fmt} {policy}')
            space=DGSpace(rectangle_mesh(size,size),p,basis_type='dub_orth')
            device=as_cupy_space(space)
            trace_ref=build_trace_reference(device,basis)
            d0,d1=reference_derivative_mats(device)
            source=source_moments_cupy(space.project_callable(lambda x,y:1.+x-.2*y),device)
            boundary=boundary_trace_values_cupy(lambda x,y:.2+x-.5*y,device,trace_ref)
            common=dict(source_rhs=source,boundary_trace=boundary,cspace=device,trace_ref=trace_ref,
                d0_reference=d0,d1_reference=d1,face_element_mass=face_element_mass(trace_ref),tau=1.3)
            options=dict(matrix_format=fmt,block_size=args.block_size,cache_local_factors=policy!='none',
                         local_factor_kind=policy if policy!='none' else 'schur-lu')
            first_call_start=time.perf_counter()
            raw=poisson.assemble(**common,**options,refresh=args.refresh or args.thorough,thorough=args.thorough)
            first_call_seconds=time.perf_counter()-first_call_start
            first_report=raw.poisson_context.reports['assembly']
            raw,assembly_times=_measure(lambda:poisson.assemble(**common,**options),args.warmups,args.repeats,
                                        lambda result:result.timings['raw.kernel.device'])
            context=raw.poisson_context
            row=dict(order=p,triangles=space.mesh.num_tri,trace_basis=basis,format=fmt,policy=policy,
                assembly_and_factor_construction=assembly_times,assembly_tuning=first_report,
                assembly_first_call_wall_seconds=first_call_seconds,
                factor_bytes=context.local_factor_bytes)
            if fmt!='coo':
                def rhs(refresh=False,thorough=False):
                    return poisson.refresh_rhs(**common,csr_pattern=raw.csr_pattern,block_size=args.block_size,
                        cached_factors=raw if policy!='none' else None,refresh=refresh,thorough=thorough)
                first_rhs_start=time.perf_counter()
                refreshed=rhs(args.refresh or args.thorough,args.thorough)
                row['rhs_first_call_wall_seconds']=time.perf_counter()-first_rhs_start
                row['rhs_tuning']=refreshed.poisson_context.reports['rhs']
                refreshed,row['rhs_refresh']=_measure(rhs,args.warmups,args.repeats,
                    lambda result:result.timings['raw.kernel.device'])
                row['rhs_relative_difference']=float((cp.linalg.norm(refreshed.rhs-raw.rhs)/cp.maximum(cp.linalg.norm(raw.rhs),1.)).item())
                if row['rhs_relative_difference']>max(2.e-10,512*np.finfo(REAL_DTYPE).eps):
                    raise ArithmeticError('cached RHS differs from fresh assembly')
            trace=cp.sin(cp.arange(space.mesh.num_edg*device.edg_dof,dtype=REAL_DTYPE))
            reconstruction_args={k:v for k,v in common.items() if k!='boundary_trace'}
            def reconstruct(refresh=False,thorough=False):
                return poisson.reconstruct(**reconstruction_args,trace=trace,context=context,
                    block_size=args.block_size,return_local_unknowns=True,refresh=refresh,thorough=thorough)
            first_reconstruction_start=time.perf_counter()
            reconstruct(args.refresh or args.thorough,args.thorough)
            row['reconstruction_first_call_wall_seconds']=time.perf_counter()-first_reconstruction_start
            row['reconstruction_tuning']=context.reports['reconstruction']
            _,row['reconstruction']=_measure(reconstruct,args.warmups,args.repeats,lambda result:result[-1])
            if args.include_legacy:
                from hybridge.mixed.raw_cuda.identity import (
                    assemble_projected_diffusion_trace_system_eliminated_raw_cuda as legacy,
                    _coop_shared_sizes)
                needed=_coop_shared_sizes(space.el_dof,device.edg_dof,batched_full=True)
                capacity=report['device'].get('sharedMemPerBlockOptin') or report['device']['sharedMemPerBlock']
                if needed<=capacity and policy!='schur-cholesky':
                    def baseline():
                        return legacy(**common,matrix_format=fmt,block_size=128,
                                      cache_local_factors=policy=='schur-lu',_legacy=True)
                    old,row['legacy_assembly']=_measure(baseline,args.warmups,args.repeats,
                                                       lambda result:result.timings['raw.kernel.device'])
                    # Direct COO may have different duplicate placement; compare matvecs.
                    from scipy.sparse import coo_matrix,csr_matrix,bsr_matrix
                    def host_matrix(result):
                        shape=(result.rhs.size,)*2
                        if fmt=='coo':
                            return coo_matrix((cp.asnumpy(result.data),(cp.asnumpy(result.rows),cp.asnumpy(result.cols))),shape=shape).tocsr()
                        constructor=bsr_matrix if fmt=='bsr' else csr_matrix
                        return constructor((cp.asnumpy(result.data),cp.asnumpy(result.indices),cp.asnumpy(result.indptr)),shape=shape).tocsr()
                    left,right=host_matrix(raw),host_matrix(old)
                    v=np.sin(np.arange(raw.rhs.size))
                    error=np.linalg.norm((left-right)@v)/max(np.linalg.norm(right@v),1.)
                    row['legacy_matvec_relative_difference']=float(error)
                    if error>max(2.e-10,512*np.finfo(REAL_DTYPE).eps):
                        raise ArithmeticError('new and legacy assembly matvecs differ')
                else:
                    row['legacy_assembly_unavailable']='shared-memory capacity or combined Cholesky construction'
            report['cases'].append(row)
            args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
            say(f'  assembly {assembly_times["kernel_median_seconds"]*1.e3:.3f} ms; '
                f'reconstruction {row["reconstruction"]["kernel_median_seconds"]*1.e3:.3f} ms')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
