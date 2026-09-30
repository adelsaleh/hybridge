#!/usr/bin/env python3
"""Reproducible FP64 assembly sweeps and paired diffusion regression checks.

No global solve, AMGX build, reconstruction, or time integration is performed.
JSONL retains each repetition and its preparation/upload/graph/JIT/kernel timings.
"""
from __future__ import annotations
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

if __package__ in {None, ''}:
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))

import numpy as np
from hdgfem import DGSpace, rectangle_mesh
from hdgfem.runtime.optional import require_cupy
from hdgfem.assembly.advection_diffusion_reaction import prepare_adr_data
from hdgfem.backends.advection_diffusion_reaction_raw_cuda import assemble_projected_adr_trace_operator_raw_cuda


from scripts.advection_diffusion_reaction.cases.tensor_cases import raw_cuda_coefficient as coefficient


def zero(x,y):
    return 0.*x


def summarize(samples):
    return {key:statistics.median(row[key] for row in samples) for key in samples[0]}


def emit(path, record):
    with path.open('a') as stream:
        stream.write(json.dumps(record)+'\n')
    print(json.dumps({key:value for key,value in record.items() if key not in {'samples','before','after'}}),flush=True)


def sweep(args,cp):
    for nx in args.nx:
        for order in args.orders:
            space=DGSpace(rectangle_mesh(nx,nx),order,basis_type='dub_orth')
            beta=(space*space).field((space.constant(.7),space.constant(-.2)))
            for basis in ('legacy-lagrange','legendre-modal'):
                trace=space.trace_space(basis)
                for name in args.coefficients:
                    diffusion=coefficient(name)
                    prep=prepare_adr_data(space.constant(1.),.3,beta,space,diffusion=diffusion,
                        trace_space=trace,dense_local_matrices=False,
                        diffusion_stabilization=lambda x,y: .7+.05*x+.03*y)
                    for fmt in ('coo','csr','bsr'):
                        for block in (32,64,128):
                            def assemble():
                                return assemble_projected_adr_trace_operator_raw_cuda(
                                    prep,zero,space,diffusion=diffusion,trace_space=trace,
                                    matrix_format=fmt,block_size=block)
                            for _ in range(args.warmup):
                                result=assemble()
                                del result
                            samples=[]
                            for _ in range(args.repeats):
                                result=assemble()
                                samples.append(result.assembly.timings)
                                del result
                            emit(args.output,dict(mode='adr',elements=space.mesh.num_tri,order=order,
                                basis=basis,coefficient=name,matrix_format=fmt,block_size=block,
                                samples=samples,median=summarize(samples)))
                    del prep
            del space,beta,trace,assemble
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()


def diffusion_comparison(args,cp):
    from hdgfem.backends import diffusion_cupy as wrapper, diffusion_raw_cuda as current
    from scripts.gpu.benchmark_fused_raw_assembly_kernels import _make_assembler
    spec=importlib.util.spec_from_file_location('hdgfem.backends._tensor_baseline',args.diffusion_before)
    baseline=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=baseline
    spec.loader.exec_module(baseline)
    assert baseline._RAW_ASSEMBLY_COOP_TEMPLATE==current._RAW_ASSEMBLY_COOP_TEMPLATE
    assert baseline._RAW_TRACE_ORIENTATION_HELPERS==current._RAW_TRACE_ORIENTATION_HELPERS
    original=wrapper.assemble_projected_diffusion_trace_system_eliminated_raw_cuda
    try:
        for nx in args.nx:
            for order in args.orders:
                space=DGSpace(rectangle_mesh(nx,nx),order,basis_type='dub_orth')
                for basis in ('legacy-lagrange','legendre-modal'):
                    for fmt in ('coo','csr','bsr'):
                        options=SimpleNamespace(trace_basis=basis,matrix_format=fmt,block_size='auto',
                            cache_policy='none',phase='assembly',coefficient_case='constant')
                        assemble=_make_assembler('poisson',space,options)
                        versions=[baseline.assemble_projected_diffusion_trace_system_eliminated_raw_cuda,original]
                        samples=[[],[]]
                        for _ in range(args.warmup):
                            for version in versions:
                                wrapper.assemble_projected_diffusion_trace_system_eliminated_raw_cuda=version
                                result=assemble()
                                del result
                        for repeat in range(2*args.repeats):
                            for index in ([0,1] if repeat%2==0 else [1,0]):
                                wrapper.assemble_projected_diffusion_trace_system_eliminated_raw_cuda=versions[index]
                                start=time.perf_counter()
                                result=assemble()
                                cp.cuda.get_current_stream().synchronize()
                                samples[index].append({'kernel':result.timings['raw.kernel.device'],
                                                       'wall':time.perf_counter()-start})
                                del result
                        medians=[summarize(sample) for sample in samples]
                        emit(args.output,dict(mode='diffusion-paired',elements=space.mesh.num_tri,order=order,
                            basis=basis,matrix_format=fmt,before=samples[0],after=samples[1],
                            source_identical=True,
                            ratios={k:medians[1][k]/medians[0][k] for k in medians[0]}))
                del space,assemble
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()
    finally:
        wrapper.assemble_projected_diffusion_trace_system_eliminated_raw_cuda=original


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--nx',type=int,nargs='+',default=[16,128])
    parser.add_argument('--orders',type=int,nargs='+',default=[2,4,6])
    parser.add_argument('--coefficients',nargs='+',choices=['scalar','constant-full','variable-full'],
                        default=['scalar','constant-full','variable-full'])
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--diffusion-before',type=Path)
    args=parser.parse_args()
    if min(args.nx)<1 or min(args.orders)<0 or max(args.orders)>6 or args.warmup<1 or args.repeats<1:
        parser.error('require nx>=1, p=0--6, and positive warmup/repeats')
    cp=require_cupy()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    emit(args.output,dict(mode='environment',gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        cupy=cp.__version__,cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        precision='FP64',warmup=args.warmup,repeats=args.repeats))
    if args.diffusion_before:
        diffusion_comparison(args,cp)
    else:
        sweep(args,cp)


if __name__=='__main__':
    main()
