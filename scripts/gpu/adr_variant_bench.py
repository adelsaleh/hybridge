#!/usr/bin/env python3
"""ADR tensor-kernel variant study driver around benchmark_tensor_raw_cuda.sweep.

Adds, by monkeypatching only inside this process:
  --block-sizes 256/512        (extends RAW_CUDA_EXPLICIT_BLOCK_SIZES)
  --volume-quad-1d 8:10,9:11   (DGSpace(..., volume_quad_1d=n) per order)
  --layout noncompact          (p>=7: keep N-D in shared memory when it fits the opt-in limit)
  --maxrregcount N             (NVRTC register cap, needed for 512 threads at >128 regs)
and records kernel registers/local bytes/occupancy through cuFuncGetAttribute
(record_kernel_resources silently records nothing for RawModule functions).
Run from the worktree with PYTHONPATH=<worktree> (worktree_stage.sh does this).
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--nx', type=int, nargs='+', default=[128])
parser.add_argument('--orders', type=int, nargs='+', default=[9])
parser.add_argument('--coefficients', nargs='+', default=['variable-full'])
parser.add_argument('--block-sizes', type=int, nargs='+', default=[128, 256])
parser.add_argument('--cache-local-factors', default='none')
parser.add_argument('--warmup', type=int, default=1)
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--reconstruction', action='store_true')
parser.add_argument('--volume-quad-1d', default='', help='order:n pairs, e.g. 8:10,9:11')
parser.add_argument('--layout', choices=['auto', 'noncompact'], default='auto')
parser.add_argument('--maxrregcount', type=int, default=0)
parser.add_argument('--gpu-lock', type=Path, default=Path(os.environ.get('HYBRIDGE_SCRATCH', 'run_outputs'))/'asm_h200_tuning'/'gpu.lock')
parser.add_argument('--dry-run', action='store_true')
parser.add_argument('--parity', action='store_true',
                    help='instead of timing: compare BSR data/rhs of each block size and the selected layout '
                         'against block 128 with the stock layout (same quadrature), print max differences')
args = parser.parse_args()
quad = {int(k): int(v) for k, v in (item.split(':') for item in args.volume_quad_1d.split(',') if item)}

import numpy as np
import hybridge.hdg.cuda.launch as launch
import hybridge.mixed.raw_cuda.tensor as T
import scripts.advection_diffusion_reaction.benchmarks.benchmark_tensor_raw_cuda as bench

launch.RAW_CUDA_EXPLICIT_BLOCK_SIZES = (1, 32, 64, 128, 256, 512)
variant = dict(volume_quad_1d=quad, layout=args.layout, maxrregcount=args.maxrregcount,
               mass_env=os.environ.get('HYBRIDGE_RAW_ADR_TENSOR_MASS', 'auto'),
               phase_clock=os.environ.get('HYBRIDGE_RAW_ADR_PHASE_CLOCK', ''))

_DGSpace = bench.DGSpace
def DGSpace(mesh, order, **kw):
    if order in quad:
        kw['volume_quad_1d'] = quad[order]
    return _DGSpace(mesh, order, **kw)
bench.DGSpace = DGSpace

if args.layout == 'noncompact':
    _ws = T.tensor_execution_workspace
    def tensor_execution_workspace(nel, kinds, nq, nfq, *, order, shared_limit):
        if order > 6:
            try:
                return T.TensorExecutionWorkspace(*T.tensor_workspace(
                    nel, kinds, nq, nfq, order=order, compact=False, shared_limit=shared_limit), compact=False)
            except T.TensorWorkspaceError:
                pass
        return _ws(nel, kinds, nq, nfq, order=order, shared_limit=shared_limit)
    T.tensor_execution_workspace = tensor_execution_workspace

if args.maxrregcount:
    _module = T.real_raw_module
    def real_raw_module(*, code, options=(), **kw):
        return _module(code=code, options=tuple(options) + (f'--maxrregcount={args.maxrregcount}',), **kw)
    T.real_raw_module = real_raw_module

def record_kernel_resources(cupy, kernel, block_size, dynamic_shared_bytes, timings, prefix):
    try:
        drv = cupy.cuda.driver
        ptr = getattr(kernel, 'ptr', None) or kernel.kernel.ptr
        get = lambda attribute: int(drv.funcGetAttribute(attribute, ptr))
        blocks = drv.occupancyMaxActiveBlocksPerMultiprocessor(ptr, int(block_size), int(dynamic_shared_bytes))
        timings.update({prefix + 'num_regs': float(get(4)), prefix + 'local_size_bytes': float(get(3)),
                        prefix + 'blocks_per_sm': float(blocks),
                        prefix + 'warps_per_sm': float(blocks * block_size // 32)})
    except Exception as exc:
        timings[prefix + 'resource_error'] = str(exc)
launch.record_kernel_resources = record_kernel_resources

_emit = bench.emit
def emit(path, record):
    record = dict(record, variant=variant)
    _emit(path, record)
bench.emit = emit

ns = argparse.Namespace(output=args.output, nx=args.nx, orders=args.orders, coefficients=args.coefficients,
                        bases=['legacy-lagrange'], matrix_formats=['bsr'], block_sizes=args.block_sizes,
                        cache_local_factors=args.cache_local_factors, warmup=args.warmup, repeats=args.repeats,
                        reconstruction=args.reconstruction)
if args.dry_run:
    from hybridge import rectangle_mesh
    for p in args.orders:
        space = bench.DGSpace(rectangle_mesh(2, 2), p, basis_type='dub_orth')
        trace = space.trace_space('legacy-lagrange')
        ws = T.tensor_execution_workspace(space.el_dof, np.array([6]), space.quad_data.Krf_w.size,
                                          trace.weights.size, order=p, shared_limit=232448)
        print(json.dumps(dict(order=p, nq=int(space.quad_data.Krf_w.size), workspace=ws.__dict__, variant=variant)))
    sys.exit(0)

from hybridge.runtime.optional import require_cupy
from hybridge.runtime.benchmarking import reserve_gpu
with reserve_gpu(args.gpu_lock, notify=lambda message: print(message, flush=True)):
    cp = require_cupy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    emit(args.output, dict(mode='environment', gpu=cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())['name'].decode(),
                           cupy=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                           precision='FP64', warmup=args.warmup, repeats=args.repeats))
    if not args.parity:
        bench.sweep(ns, cp)
    else:
        from hybridge import rectangle_mesh
        from hybridge.mixed.adr_preparation import prepare_adr_data
        from hybridge.mixed.raw_cuda.adr_operator import assemble_projected_adr_trace_operator_raw_cuda as assemble
        from scripts.advection_diffusion_reaction.cases.tensor_cases import raw_cuda_coefficient
        patched = T.tensor_execution_workspace
        for p in args.orders:
            space = bench.DGSpace(rectangle_mesh(args.nx[0], args.nx[0]), p, basis_type='dub_orth')
            beta = (space*space).field((space.constant(.7), space.constant(-.2)))
            trace = space.trace_space('legacy-lagrange')
            for name in args.coefficients:
                diffusion = raw_cuda_coefficient(name)
                prep = prepare_adr_data(space.constant(1.), .3, beta, space, diffusion=diffusion, trace_space=trace,
                                        dense_local_matrices=False, diffusion_stabilization=lambda x, y: .7+.05*x+.03*y)
                run = lambda block: assemble(prep, bench.zero, space, diffusion=diffusion, trace_space=trace,
                                             matrix_format='bsr', block_size=block, cache_local_factors='none').assembly
                if args.layout == 'noncompact':
                    T.tensor_execution_workspace = _ws
                base = run(128)
                T.tensor_execution_workspace = patched
                for block in args.block_sizes:
                    other = run(block)
                    d = float(cp.abs(other.data - base.data).max()); r = float(cp.abs(other.rhs - base.rhs).max())
                    rec = dict(mode='parity', order=p, coefficient=name, elements=space.mesh.num_tri, block_size=block,
                               max_abs_data_diff=d, max_abs_rhs_diff=r, max_abs_data=float(cp.abs(base.data).max()),
                               bitwise=bool(d == 0.0 and r == 0.0),
                               shared_bytes=other.timings.get('raw.shared_bytes'), compact=other.timings.get('raw.compact_coupling'))
                    emit(args.output, rec)
