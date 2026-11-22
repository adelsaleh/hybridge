#!/usr/bin/env python3
"""Measured resource attributes of the raw-CUDA assembly kernels on the allocated GPU (study step 0).

Compiles, through hybridge's own NVRTC path, the advection kernels that production launches
(the fused COO/CSR/BSR entries with their launch bounds, the fused reconstruction, the three
TSLE stages) for p = 1..9, both trace orientations and the three stabilization variants, plus
the tensor ADR kernels, and reads the loaded kernels' attributes (registers, local memory,
static shared memory) and the CUDA occupancy model (resident blocks and warps per SM) at every
legal block size. Nothing is assembled; a minute of GPU time and a few hundred MB of memory.
With --ptxas-table the registers are compared with the offline nvcc/ptxas numbers.

    kernel_attributes.py OUTPUT.jsonl [--ptxas-table ptxas_offline_sm90.json] [--orders 1-9]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
ORIENTATIONS = {'legacy-lagrange': 'nodal', 'legendre-modal': 'modal'}
VARIANTS = ('upwind', 'conflict-averaged-upwind', 'conflict-averaged-upwind+pointwise-beta')
FUSED_BLOCKS = (32, 64, 128)
TSLE_BLOCKS = (32, 64, 128, 256)


def parse_orders(text):
    orders = set()
    for part in text.split(','):
        low, _, high = part.partition('-')
        orders.update(range(int(low), int(high or low) + 1))
    return sorted(orders)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output', type=Path)
    parser.add_argument('--ptxas-table', type=Path)
    parser.add_argument('--orders', default='1-9')
    args = parser.parse_args(argv)
    import cupy as cp
    import numpy as np
    import hybridge as hdg
    from hybridge.core.mesh import rectangle_mesh
    from hybridge.runtime.precision import real_raw_module
    import hybridge.transport.raw_cuda as R
    import hybridge.transport.tsle_bsr as S
    import hybridge.mixed.raw_cuda.tensor as T
    from hybridge.hdg.trace_maps import _trace_orientation_mode
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name = device['name'].decode() if isinstance(device['name'], bytes) else str(device['name'])
    limit = int(device['sharedMemPerBlockOptin'])
    per_sm = dict(shared=int(device['sharedMemPerMultiprocessor']), regs=int(device['regsPerMultiprocessor']),
                  threads=int(device['maxThreadsPerMultiProcessor']), blocks=32)
    ptxas = json.loads(args.ptxas_table.read_text()) if args.ptxas_table else {}
    rows = []

    def record(kind, p, variant, kernel, label, block, dynamic_shared, **extra):
        attributes = {k: int(kernel.attributes[k]) for k in ('num_regs', 'local_size_bytes', 'shared_size_bytes', 'max_threads_per_block')}
        if dynamic_shared > 48 * 1024:
            kernel.max_dynamic_shared_size_bytes = int(dynamic_shared)
        blocks = int(cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(getattr(kernel, "kernel", kernel).ptr, block, int(dynamic_shared)))
        by = dict(regs=per_sm['regs'] // max(attributes['num_regs'] * block, 1),
                  shared=per_sm['shared'] // (int(dynamic_shared) + attributes['shared_size_bytes'] + 1024),
                  threads=per_sm['threads'] // block, blocks=per_sm['blocks'])
        limiter = min(by, key=by.get)
        row = dict(kind=kind, p=p, variant=variant, kernel=label, block=block, dynamic_shared=int(dynamic_shared),
                   blocks_per_sm=blocks, warps_per_sm=blocks * block // 32, limiter=limiter, gpu=name, **attributes, **extra)
        rows.append(row)
        print(f"timings attrs {kind} p={p} {variant} {label} b={block}: regs={attributes['num_regs']} "
              f"local={attributes['local_size_bytes']} dyn={dynamic_shared/1024:.1f}KiB blocks/SM={blocks} "
              f"warps/SM={row['warps_per_sm']} limiter={limiter}", flush=True)
        with args.output.open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        return row

    mesh = rectangle_mesh(4, 4)
    for p in parse_orders(args.orders):
        space = hdg.DGSpace(mesh, p, basis_type='dub_orth')
        nel, ntr = (p + 1) * (p + 2) // 2, p + 1
        q = next(getattr(space, a) for a in dir(space) if hasattr(getattr(space, a, None), 'Krf_w'))
        nqv = int(q.Krf_w.size)
        for basis, orientation in ORIENTATIONS.items():
            trace = space.trace_space(basis)
            nqf = int(trace.weights.size)
            shared = R._fused_shared_sizes(nel, ntr, nqf, lu_mode='coop')
            for variant in VARIANTS:
                pointwise = variant.endswith('+pointwise-beta')
                stabilization = variant.replace('+pointwise-beta', '')
                kw = dict(nel=nel, ntr=ntr, ncols=3 * ntr + 1, nqf=nqf, lu_mode='coop', trace_orientation=orientation,
                          advection_stabilization=stabilization, nqv=nqv if pointwise else None, pointwise_beta=pointwise)
                extra = dict(basis=basis, orientation=orientation, nel=nel, nqf=nqf)
                for fmt, template, entry in (('coo', R._RAW_FUSED_TEMPLATE + "\n" + S._TSLE_KERNEL_TEMPLATE, 'assemble_advection_raw_fused'),
                                             ('csr', R._raw_fused_csr_template(), 'assemble_advection_raw_fused_csr'),
                                             ('bsr', R._raw_fused_bsr_template(), 'assemble_advection_raw_fused_bsr')):
                    source = R._kernel_source(template, **kw)
                    for block in FUSED_BLOCKS:
                        # The p=6/128 launch-bound specialization changes the compiled kernel.
                        bounded = R._apply_fused_assembly_launch_bounds(source, kernel_name=entry, nel=nel, block_size=block)
                        module = real_raw_module(code=bounded, options=('--std=c++11',))
                        row = record('ar-fused', p, variant, module.get_function(entry), f'fused-{fmt}', block, shared[0],
                                     launch_bounds=int(bounded != source), **extra)
                        if fmt == 'coo' and block == FUSED_BLOCKS[-1]:
                            record('ar-fused-reconstruct', p, variant, module.get_function('reconstruct_advection_raw_fused'),
                                   'reconstruct', block, shared[1], **extra)
                            for stage, dyn in (('build', lambda b: S._shared_bytes_build(nel, 3 * ntr + 1, nqf)),
                                               ('solve', lambda b: S._shared_bytes_solve(nel, 3 * ntr + 1, b)),
                                               ('scatter', lambda b: S._shared_bytes_scatter(nel, ntr))):
                                kernel = module.get_function(f'advection_tsle_{stage}' + ('_bsr' if stage == 'scatter' else ''))
                                for tb in TSLE_BLOCKS:
                                    record('ar-tsle', p, variant, kernel, f'tsle-{stage}', tb, dyn(tb), **extra)
                    reference = ptxas.get('fused', {}).get(variant, {}).get(str(p)) if fmt == 'coo' and orientation == 'nodal' else None
                    if reference:
                        delta = row['num_regs'] - reference['regs']
                        print(f"   ptxas check fused p={p} {variant}: nvrtc regs={row['num_regs']} local={row['local_size_bytes']} "
                              f"vs ptxas regs={reference['regs']} spill={reference['spill']} -> delta {delta:+d}"
                              + ('  MISMATCH' if abs(delta) > 8 else ''), flush=True)
        orientation = _trace_orientation_mode(space.trace_space('legacy-lagrange'))
        nqf = int(space.trace_space('legacy-lagrange').weights.size)
        for coefficient, kind in (('scalar', 0), ('variable-full', 6)):
            ws = T.tensor_execution_workspace(nel, np.array([kind]), nqv, nqf, order=p, shared_limit=limit)
            for block in (1, 128):
                code = T.kernel_source(nel, ntr, nqv, nqf, orientation, 'coo', ws.batch, ws.storage, compact=ws.compact,
                                       external_mass=ws.external_mass, cooperative_only=p > 6 and block != 1)
                kernel = real_raw_module(code=code, options=('--std=c++11',)).get_function('assemble_adr_tensor')
                record('adr-tensor', p, coefficient, kernel, 'adr-tensor', block, ws.shared_bytes, batch=ws.batch,
                       compact=int(ws.compact), external_mass=int(ws.external_mass))
        cp.get_default_memory_pool().free_all_blocks()
    print(f'{len(rows)} kernels recorded in {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
