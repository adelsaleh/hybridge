#!/usr/bin/env python3
"""GPU gate for the device paths of the H200 tuned ADR campaign (run before the ladder).

1. Closed-loop stress coefficients on CuPy points equal the NumPy evaluation (all variants).
2. ``BernsteinHpSymmetricPartPreconditioner.from_face_dense`` against the host constructor on
   cached systems: same application to random vectors, to round-off.
3. The NVML sampler reads this GPU.

    run_cuda13.sh python campaigns/adr_h200_tuned/gpu_gate.py --caches CACHE [CACHE ...]
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def stress_parity(cp, np):
    from scripts.advection_diffusion_reaction.cases import closed_loop_stress_cases as stress
    rng = np.random.default_rng(3)
    rho, phi = rng.uniform(0.01, 0.99, (2000, 64)), rng.uniform(0, 2*np.pi, (2000, 64))
    worst = 0.
    for variant in stress.VARIANTS:
        parameters = stress.StressParameters(variant=variant)
        x, y = stress.polar_points(rho, phi, parameters.hole_radius)
        kwargs, exact = stress.make_case(parameters, 1.0 if variant == 'orthogonal' else 37.25)
        functions = [exact, kwargs['source'], *kwargs['beta'], *(k for row in kwargs['diffusion'] for k in row)]
        for function in functions:
            host = function(x, y)
            device = function(cp.asarray(x), cp.asarray(y))
            assert isinstance(device, cp.ndarray), 'stress coefficient left the device'
            error = float(np.max(np.abs(cp.asnumpy(device)-host))/max(np.max(np.abs(host)), 1e-300))
            worst = max(worst, error)
    assert worst <= 1e-12, worst
    print(f'gate stress coefficients: CuPy = NumPy to {worst:.2e} (max relative, {len(stress.VARIANTS)} variants)', flush=True)


def hp_mg_parity(cp, np, cache):
    from scipy import sparse
    from hybridge.linalg.face_dense import FaceDenseSystem
    from hybridge.linalg.multigrid.krylov import BernsteinHpSymmetricPartPreconditioner
    get = lambda name: np.load(Path(cache)/(name+'.npy'), mmap_mode='r', allow_pickle=False)
    system = FaceDenseSystem(**{f.name: ('eliminate' if f.name == 'mode' else get('system_'+f.name)) for f in fields(FaceDenseSystem)})
    degree = system.block_size-1
    valid = system.neighbors >= 0
    matrix = sparse.bsr_matrix((np.ascontiguousarray(system.blocks[valid]), system.neighbors[valid].astype(np.int32),
                                np.r_[0, np.cumsum(valid.sum(axis=1))].astype(np.int32)), shape=(system.num_dofs, system.num_dofs))
    matrix.sort_indices()
    started = time.perf_counter()
    host = BernsteinHpSymmetricPartPreconditioner(matrix, degree=degree, kernels='graph')
    cp.cuda.Device().synchronize()
    host_seconds = time.perf_counter()-started
    blocks, neighbors = cp.asarray(np.asarray(system.blocks)), cp.asarray(np.asarray(system.neighbors))
    cp.cuda.Device().synchronize()
    started = time.perf_counter()
    device = BernsteinHpSymmetricPartPreconditioner.from_face_dense(blocks, neighbors, degree=degree, kernels='graph')
    cp.cuda.Device().synchronize()
    device_seconds = time.perf_counter()-started
    worst = 0.
    for seed in range(3):
        v = cp.asarray(np.random.default_rng(seed).standard_normal(system.num_dofs))
        a, b = cp.empty_like(v), cp.empty_like(v)
        host.apply_into(v, a)
        device.apply_into(v, b)
        worst = max(worst, float(cp.linalg.norm(a-b)/cp.linalg.norm(a)))
    host.close()
    device.close()
    assert worst <= 1e-10, worst
    print(f'gate hp-MG device constructor ({system.num_dofs:,} DOFs, p={degree}): application = host to {worst:.2e}; '
          f'setup host {host_seconds:.2f} s, device {device_seconds:.2f} s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--caches', nargs='*', default=[])
    args = parser.parse_args()
    import numpy as np
    import cupy as cp
    from campaigns.adr_h200_tuned.gpu_sampler import GpuSampler
    sampler = GpuSampler(cp.cuda.Device(0).pci_bus_id, period=0.2).start()
    stress_parity(cp, np)
    for cache in args.caches:
        hp_mg_parity(cp, np, cache)
    sampler.stop()
    usage = sampler.summary()
    assert usage.get('available') and usage.get('samples'), usage
    print(f"gate NVML sampler: {usage['samples']} samples, SM busy mean {usage['sm_busy_mean']:.0f}%", flush=True)
    print('gate passed', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
