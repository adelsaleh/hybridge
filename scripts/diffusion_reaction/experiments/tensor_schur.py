#!/usr/bin/env python3
"""Quick fused-WMMA diffusion prototype; FP64 CUDA-core LU/solves stay intact.

No global production solver or time integration is invoked. Tiny dense solves
are used only to verify the original physical residual. Production defaults
and kernel templates are never modified on disk.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hdgfem import DGSpace, rectangle_mesh
from hdgfem.core.mesh import DGMesh
from hdgfem.runtime.optional import require_cupy
from hdgfem.backends import diffusion_raw_cuda as raw
from hdgfem.backends.diffusion_cupy import assemble_projected_diffusion_trace_system_eliminated_raw_cupy
from scripts.diffusion_reaction.experiments.tensor_schur_kernels import specialize_tensor_schur


def zero(x, y):
    """Supply zero boundary trace data in either array namespace."""
    return 0.0 * (x + y)


@contextmanager
def prototype(mode, block, evidence, output):
    """Specialize source in one isolated process; restore hooks on exit."""
    original_source = raw._kernel_source
    original_compile = raw._compile_kernel_timed

    def source(template, **kwargs):
        """Substitute only the two dense Schur products in full assembly."""
        code = original_source(template, **kwargs)
        if mode in {"fp64", "fp32"} or not kwargs.get("batched_full"):
            return code
        modified, metadata = specialize_tensor_schur(
            code, nel=kwargs['nel'], ntr=kwargs['ntr'],
            batch_cols=raw._batched_full_column_count(kwargs['nel'], kwargs['ntr'], kwargs['ncols']),
            block_size=block, mode=mode)
        start = code.index('__device__ __forceinline__ void factor_diffusion_schur_lu_coop_raw')
        stop = code.index('extern "C" __global__ void assemble_diffusion_raw_coop', start)
        untouched = code[start:stop]
        if untouched not in modified:
            raise AssertionError("prototype changed FP64 factorization/triangular-solve source")
        metadata['fp64_factor_and_triangular_source_sha256'] = hashlib.sha256(untouched.encode()).hexdigest()
        evidence.update(metadata)
        return modified

    def compile_kernel(cp, code, name, shared):
        """Capture compiler resources once, outside warmed measurements."""
        kernel, seconds = original_compile(cp, code, name, shared)
        if 'kernel_attributes' not in evidence:
            from hdgfem.runtime.precision import cuda_source
            compiled_source = cuda_source(code)
            evidence['kernel_attributes'] = dict(kernel.attributes)
            evidence['dynamic_shared_bytes'] = int(shared)
            evidence['source_sha256'] = hashlib.sha256(compiled_source.encode()).hexdigest()
            (output / f'{mode}-b{block}.cu').write_text(compiled_source)
        return kernel, seconds

    with patch.object(raw, '_kernel_source', source), patch.object(raw, '_compile_kernel_timed', compile_kernel):
        yield


def make_space(nx, order, sheared, ny=None):
    """Use package meshes; shear the accuracy mesh to expose rounding errors."""
    mesh = rectangle_mesh(nx, nx if ny is None else ny)
    if sheared:
        mesh = DGMesh.from_arrays(mesh.node_coords @ np.array([[1.3, 0.27], [-0.19, 0.83]]), mesh.triangles)
    return DGSpace(mesh, order, basis_type='dub_orth')


def assemble(space, block, basis):
    """Use the unchanged public backend wrapper for fused direct BSR assembly."""
    return assemble_projected_diffusion_trace_system_eliminated_raw_cupy(
        space.project_callable(lambda x, y: 1.0 + x*y), space.zeros(), zero,
        1.0, space, trace_basis=basis, matrix_format='bsr', block_size=block)


def dense_system(cp, result):
    """Materialize only tiny validation matrices, without a sparse factorization."""
    from scipy.sparse import bsr_matrix
    n = result.rhs.size
    return bsr_matrix((cp.asnumpy(result.data), cp.asnumpy(result.indices), cp.asnumpy(result.indptr)),
                      shape=(n, n)).toarray(), cp.asnumpy(result.rhs)


def relative(cp, value, reference):
    """Compute a scalar Frobenius-relative error with a safe zero denominator."""
    return float((cp.linalg.norm(value-reference) / cp.maximum(cp.linalg.norm(reference), 1.e-300)).get())


def main():
    """Record accuracy and warmed fused-assembly cost for each experimental mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--order', type=int, default=6)
    parser.add_argument('--nx', type=int, default=64)
    parser.add_argument('--ny', type=int, help='rectangle subdivisions in y; defaults to nx')
    parser.add_argument('--blocks', nargs='+', type=int, choices=(32,64,128), default=[32,64,128])
    parser.add_argument('--modes', nargs='+', choices=('fp64','fp32','tf32','tf32x3'), default=['fp64','tf32','tf32x3'])
    parser.add_argument('--basis', choices=('legacy-lagrange','legendre-modal'), default='legendre-modal')
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, help='write FP64 references or read them in an FP32 process')
    parser.add_argument('--reference-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.order <= 6 or args.nx < 1 or (args.ny is not None and args.ny < 1) or args.repeats < 1:
        parser.error('require p=1..6, nx>=1, ny>=1, repeats>=1')
    cp = require_cupy()
    from hdgfem.runtime.precision import REAL_ITEMSIZE
    if REAL_ITEMSIZE == 4 and (args.modes != ['fp32'] or args.reference_dir is None):
        raise ValueError('FP32 requires --modes fp32 and an existing --reference-dir from FP64')
    if REAL_ITEMSIZE == 8 and 'fp32' in args.modes:
        raise ValueError('run fp32 in a separate HDGFEM_PRECISION=float32 process')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'results.json').exists():
        parser.error('output already contains results; use another directory')
    validation_space = make_space(2, args.order, True)
    space = make_space(args.nx, args.order, False, args.ny)
    reference_config = {'order': args.order, 'nx': args.nx, 'basis': args.basis}
    if args.ny is not None:
        reference_config['ny'] = args.ny
    if REAL_ITEMSIZE == 8:
        small_reference = assemble(validation_space, 128, args.basis)
        ar, br = dense_system(cp, small_reference)
        reference = assemble(space, 128, args.basis)
        if args.reference_dir:
            args.reference_dir.mkdir(parents=True, exist_ok=True)
            np.savez(args.reference_dir/'small.npz', matrix=ar, rhs=br)
            np.save(args.reference_dir/'data.npy', cp.asnumpy(reference.data))
            np.save(args.reference_dir/'rhs.npy', cp.asnumpy(reference.rhs))
            (args.reference_dir/'config.json').write_text(json.dumps(reference_config))
    else:
        if json.loads((args.reference_dir/'config.json').read_text()) != reference_config:
            raise ValueError('FP64 reference configuration does not match')
        with np.load(args.reference_dir/'small.npz') as saved:
            ar, br = saved['matrix'], saved['rhs']
        reference = SimpleNamespace(data=cp.load(args.reference_dir/'data.npy'), rhs=cp.load(args.reference_dir/'rhs.npy'))
    if args.reference_only:
        if REAL_ITEMSIZE != 8 or args.reference_dir is None:
            raise ValueError('--reference-only requires FP64 and --reference-dir')
        return 0
    xr = np.linalg.solve(ar, br)
    scope = ('Unchanged fused BSR kernel specialized to FP32, including CUDA-core LU and triangular solves; errors use separately generated FP64 references.'
             if REAL_ITEMSIZE == 4 else
             'Only two dense Schur products replaced inside fused BSR assembly; all LU and triangular solves remain original FP64 CUDA-core code.')
    report = {'scope':scope,
              'configuration': {**vars(args), 'output':str(args.output), 'reference_dir':str(args.reference_dir)},
              'process_precision': 'fp64' if REAL_ITEMSIZE == 8 else 'fp32',
              'gpu':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
              'cuda_runtime':cp.cuda.runtime.runtimeGetVersion(), 'cupy':cp.__version__,
              'triangles':int(space.mesh.num_tri), 'runs':[]}
    for block in args.blocks:
        for mode in args.modes:
            print(f'p={args.order} block={block} mode={mode}', flush=True)
            evidence = {}
            with prototype(mode, block, evidence, args.output):
                small = assemble(validation_space, block, args.basis)
                a, b = dense_system(cp, small)
                x = np.linalg.solve(a.astype(np.float64), b.astype(np.float64))
                small_errors = {
                    'matrix_relative':float(np.linalg.norm(a-ar)/np.linalg.norm(ar)),
                    'rhs_relative':float(np.linalg.norm(b-br)/np.linalg.norm(br)),
                    'trace_relative':float(np.linalg.norm(x-xr)/np.linalg.norm(xr)),
                    'original_physical_residual':float(np.linalg.norm(ar@x-br)/np.linalg.norm(br)),
                }
                for _ in range(2):
                    warmed = assemble(space, block, args.basis)
                    del warmed
                device, wall = [], []
                for _ in range(args.repeats):
                    cp.cuda.get_current_stream().synchronize()
                    started=time.perf_counter()
                    result=assemble(space, block, args.basis)
                    cp.cuda.get_current_stream().synchronize()
                    wall.append(1000*(time.perf_counter()-started))
                    device.append(1000*result.timings['raw.kernel.device'])
                errors={'matrix_data_relative':relative(cp,result.data,reference.data),
                        'rhs_relative':relative(cp,result.rhs,reference.rhs)}
            report['runs'].append({'mode':mode,'block':block,'small_sheared_errors':small_errors,
                'large_structured_errors':errors,'kernel_ms':device,'kernel_median_ms':statistics.median(device),
                'kernel_stdev_ms':statistics.stdev(device) if len(device)>1 else 0.0,
                'wrapper_ms':wall,'resources':evidence,
                'accuracy_gate_1e_minus_9': all(v < 1.e-9 for v in (*small_errors.values(),*errors.values()))})
            eligible = [run for run in report['runs'] if run['accuracy_gate_1e_minus_9']]
            best = min(eligible, key=lambda run: run['kernel_median_ms']) if eligible else None
            report['prototype_selection'] = {
                'key': {'gpu': report['gpu'], 'order': args.order, 'basis': args.basis,
                        'phase': 'uncached fused BSR assembly', 'triangles': report['triangles']},
                'selection': None if best is None else {'mode': best['mode'], 'block': best['block']},
                'policy': 'fastest measured candidate passing the small-sheared and structured accuracy gates; experimental evidence only',
            }
            (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report['runs'][-1],indent=2),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
