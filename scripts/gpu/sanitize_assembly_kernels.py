#!/usr/bin/env python3
"""Exercise every raw-CUDA advection assembly kernel variant once on a tiny mesh (race-check gate).

Meant to run under ``compute-sanitizer --tool racecheck|synccheck|memcheck``: 24 elements
(``rectangle_mesh(4, 3)``), fused and split3, COO/CSR/BSR, tangent (zero-flux) and eliminate walls,
plain and conflict-averaged upwind and the pointwise-beta path, every block size, then the
reconstruction kernel from the cached local response. Each configuration is also compared with
the first configuration of its (p, boundary, variant, format) group (CSR/BSR only). Any Python
error exits 1; the sanitizer's own verdict comes from its exit status and summary lines.

    sanitize_assembly_kernels.py [--orders 2,6,9] [--fused-blocks 32,64,128] [--split3-blocks 32,64,128,256]
                                 [--env KEY=VALUE ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--orders', default='2,6,9')
    parser.add_argument('--boundary', default='tangent,eliminate')
    parser.add_argument('--variants', default='conflict,upwind,conflict-pointwise')
    parser.add_argument('--formats', default='coo,csr,bsr')
    parser.add_argument('--paths', default='fused,split3')
    parser.add_argument('--fused-blocks', default='32,64,128')
    parser.add_argument('--split3-blocks', default='32,64,128,256')
    parser.add_argument('--parity-tol', type=float, default=1e-10)
    parser.add_argument('--env', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    for item in args.env:
        key, _, value = item.partition('=')
        os.environ[key] = value
    split = lambda text, cast=str: [cast(x) for x in text.split(',') if x]
    configurations = []
    for p in split(args.orders, int):
        for boundary in split(args.boundary):
            for variant in split(args.variants):
                for fmt in split(args.formats):
                    for path in split(args.paths):
                        if path == 'split3' and fmt != 'bsr':
                            continue
                        for block in split(args.fused_blocks if path == 'fused' else args.split3_blocks, int):
                            configurations.append((p, boundary, variant, fmt, path, block))
    print(f'{len(configurations)} configurations', flush=True)
    if args.dry_run:
        return 0
    import cupy as cp
    import numpy as np
    from hybridge.core.device import as_cupy_space, as_cupy_trace_space, as_cupy_vector_coefficients
    from hybridge.core.mesh import rectangle_mesh
    from hybridge.core.quadrature_coefficients import QuadratureVectorField
    from hybridge.hdg.stabilization import resolve_transport_stabilization
    from hybridge.transport.cuda import assemble_reduced_system_cuda, reconstruct_advection_field_cuda
    from scripts.gpu.benchmark_advection_tsle_bsr import _parity
    from scripts.gpu.benchmark_advection_tsle_tensor import _transport_step
    mesh = rectangle_mesh(4, 3)
    rng = np.random.default_rng(7)
    failures = 0
    for p in split(args.orders, int):
        space, source, beta = _transport_step(cp, mesh, p)
        trace = space.trace_space('legacy-lagrange')
        cspace, trace_ref = as_cupy_space(space), as_cupy_trace_space(trace)
        coefficients = as_cupy_vector_coefficients(beta, cspace)
        values = QuadratureVectorField.from_vector_field(beta, space, trace_space=trace, device=True)
        reaction = space.constant(1.0)
        full_trace = cp.asarray(rng.normal(size=space.mesh.num_edg * trace.edg_dof))
        references = {}
        for (order, boundary, variant, fmt, path, block) in configurations:
            if order != p:
                continue
            label = f'p={p} {boundary} {variant} {fmt} {path} b={block}'
            try:
                pointwise = variant == 'conflict-pointwise'
                policy = 'upwind' if variant == 'upwind' else 'conflict-averaged-upwind'
                stabilization = resolve_transport_stabilization(policy, values if pointwise else coefficients)
                tangent = boundary == 'tangent'
                result = assemble_reduced_system_cuda(
                    source, reaction, None if tangent else (lambda x, y: 0.0 * x), None if pointwise else coefficients,
                    cspace, trace_ref, backend='raw-cuda', raw_block_size=block, raw_local_assembly=path, raw_lu_mode='coop',
                    raw_matrix_format=fmt, zero_boundary_flux=tangent, raw_cache_local_response=True,
                    advection_stabilization=stabilization, beta_values=values if pointwise else None)
                field, _ = reconstruct_advection_field_cuda(full_trace, source, reaction, None if pointwise else coefficients, result)
                cp.cuda.Device().synchronize()
                finite = bool(cp.isfinite(result.data).all()) and bool(cp.isfinite(result.rhs).all()) and bool(cp.isfinite(field).all())
                group = (boundary, variant, fmt)
                note = 'reference' if group not in references else ''
                if group not in references:
                    references[group] = result
                elif fmt != 'coo':
                    errors = _parity(cp, references[group], result)
                    note = f"parity {errors['max_relative']:.1e}"
                    if errors['max_relative'] > args.parity_tol:
                        failures += 1
                        note += ' FAIL'
                if not finite:
                    failures += 1
                    note += ' NONFINITE'
                print(f'ok {label}: {note}', flush=True)
            except Exception as error:
                failures += 1
                print(f'ERROR {label}: {type(error).__name__}: {str(error)[:300]}', flush=True)
        del references
        cp.get_default_memory_pool().free_all_blocks()
    print(f'{failures} failures', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
