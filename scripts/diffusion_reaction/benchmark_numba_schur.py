"""Warmed host diffusion phase/thread benchmarks, without a global solve.

Each policy/mesh/order/basis/thread case runs in its own process, so peak RSS
is process-local (including imports and discarded JIT warmup). Timed assembly
includes COO emission; COO-to-CSR conversion is measured separately. CPU/wall
ratios report observed process CPU consumption, not just configured threads.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

from hybridge.io.records import append_jsonl_record
from hybridge.runtime.benchmarking import measure


def worker(args):
    """Measure one isolated case through the package's public backend adapters."""
    import numpy as np
    import numba
    import scipy
    from scipy.sparse import coo_matrix
    from hybridge import DGSpace, rectangle_mesh
    from hybridge.mixed.numba import (
            build_diffusion_schur_cache_numba,
            assemble_projected_diffusion_trace_system_eliminated_numba as assemble,
            assemble_projected_diffusion_trace_rhs_eliminated_numba as assemble_rhs,
            reconstruct_projected_diffusion_local_unknowns_numba as reconstruct,
        )
    if numba.config.DISABLE_JIT:
        raise RuntimeError('Benchmarks require Numba JIT enabled')
    threads = args.threads[0]
    numba.set_num_threads(threads)
    assert numba.get_num_threads() == threads
    n, p, basis, policy = args.meshes[0], args.orders[0], args.bases[0], args.policies[0]
    space = DGSpace(rectangle_mesh(n, n), p)
    source = space.project_callable(lambda x, y: 1. + x * y)
    reaction = space.zeros()
    def boundary(x, y):
        """Nonzero data exercises prescribed trace elimination."""
        return 0.3 + x - 0.5 * y
    def build():
        """Construct only local factors, including input validation."""
        return build_diffusion_schur_cache_numba(reaction, 1., space, factor_kind=policy)
    cache = None if policy == 'none' else build()
    kwargs = dict(trace_space=space.trace_space(basis), cached_factors=cache)
    def assembly():
        """Emit the reduced COO matrix and source/boundary RHS."""
        return assemble(source, reaction, boundary, 1., space, **kwargs)
    assembled = assembly()
    system = assembled.trace_system
    trace = np.linspace(-0.1, 0.2, space.mesh.num_edg * (p + 1))
    def rhs():
        """Rebuild a reduced RHS without emitting the matrix."""
        return assemble_rhs(source, reaction, boundary, 1., space, **kwargs)
    def recovery():
        """Recover primal/flux coefficients using prescribed test trace data."""
        return reconstruct(trace, source, reaction, 1., space, **kwargs)
    def csr():
        """Measure duplicate summation and canonical CSR conversion separately."""
        return coo_matrix((system.data, (system.rows, system.cols)),
                          shape=(system.rhs.size, system.rhs.size)).tocsr()
    phases = dict(assembly=assembly, rhs=rhs, reconstruction=recovery, coo_to_csr=csr)
    if cache is not None:
        phases['factor_build'] = build
    results = {name: measure(fn, args.repeats, args.minimum_seconds) for name, fn in phases.items()}
    # Untimed checks compare all three paths against uncached execution.
    baseline_kwargs = dict(trace_space=space.trace_space(basis))
    baseline = assemble(source, reaction, boundary, 1., space, **baseline_kwargs).trace_system
    np.testing.assert_allclose(system.data, baseline.data, rtol=2e-8, atol=2e-10)
    np.testing.assert_allclose(rhs()[0], baseline.rhs, rtol=2e-8, atol=2e-10)
    expected = reconstruct(trace, source, reaction, 1., space, **baseline_kwargs)
    np.testing.assert_allclose(recovery(), expected, rtol=2e-8, atol=2e-10)
    row = dict(mesh_subdivisions=n, elements=space.mesh.num_tri, order=p,
               trace_basis=basis, policy=policy, threads=numba.get_num_threads(),
               threading_layer=numba.threading_layer(), phases=results,
               trace_dofs=system.rhs.size, factor_bytes=0 if cache is None else cache.local_factor_bytes,
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
               scipy=scipy.__version__, platform=platform.platform(),
               affinity=sorted(os.sched_getaffinity(0)), repeats=args.repeats,
               stabilization=1., reaction=0., diffusion=1., solver='none',
               validation_rtol=2e-8, validation_atol=2e-10, parity='passed')
    print(json.dumps(row), flush=True)


def main():
    """Run isolated cases and incrementally persist reproducible timing records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--meshes', nargs='+', type=int, default=[16, 32])
    parser.add_argument('--orders', nargs='+', type=int, default=[2, 4, 6])
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 4, 16])
    parser.add_argument('--bases', nargs='+', default=['legacy-lagrange', 'legendre-modal'])
    parser.add_argument('--policies', nargs='+', choices=['none', 'schur-lu', 'schur-cholesky'],
                        default=['none', 'schur-lu', 'schur-cholesky'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--minimum-seconds', type=float, default=0.03)
    parser.add_argument('--output', type=Path, default=Path('run_outputs/numba_schur.jsonl'))
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.meshes + args.orders + args.threads + [args.repeats]) < 1 or args.minimum_seconds < 0:
        parser.error('mesh sizes, orders, threads and repeats must be positive')
    if args.worker:
        worker(args)
        return
    if args.output.exists():
        parser.error(f'output already exists: {args.output}')
    for n in args.meshes:
        for p in args.orders:
            for basis in args.bases:
                for policy in args.policies:
                    for threads in args.threads:
                        command = [sys.executable, '-m', 'scripts.diffusion_reaction.benchmark_numba_schur',
                                   '--worker', '--meshes', str(n), '--orders', str(p), '--bases', basis,
                                   '--policies', policy, '--threads', str(threads), '--repeats', str(args.repeats),
                                   '--minimum-seconds', str(args.minimum_seconds)]
                        env = dict(os.environ, NUMBA_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1',
                                   MKL_NUM_THREADS='1', OMP_NUM_THREADS=str(threads), NUMBA_DISABLE_JIT='0')
                        result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
                        row = json.loads(result.stdout)
                        append_jsonl_record(args.output, row)
                        print(f'n={n} p={p} {basis} {policy} threads={threads}: '
                              f'assembly={row["phases"]["assembly"]["median_seconds"]:.6f}s '
                              f'rhs={row["phases"]["rhs"]["median_seconds"]:.6f}s', flush=True)


if __name__ == '__main__':
    main()
