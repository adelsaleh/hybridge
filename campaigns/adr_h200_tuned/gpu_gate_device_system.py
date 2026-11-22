#!/usr/bin/env python3
"""GPU gate for the device-resident system path, against systems a v2 (host-output) run cached.

For each system id of a campaign output (its manifest gives the problem, its mesh job the mesh,
its cache the host-output route's arrays):

1. ``assemble_on_device`` (device outputs, no copy back) gives the cached trace system bitwise;
2. the operator, block-Jacobi and ASM built from the device arrays apply like the host-built
   ones (``from_system`` on the cache) to random vectors, to round-off;
3. the verifier from the device assembly agrees with the one built from the cache.

    run_cuda13.sh python campaigns/adr_h200_tuned/gpu_gate_device_system.py --campaign OUT --systems ID [ID ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'vendor/adr_gmres'))
sys.path.append(str(ROOT))


def register_stress(problem, root):
    if 'stress_parameters' not in problem:
        return
    from scripts.adv_diff_rea_cases import register_case
    from campaigns.adr_unified.adr_unified_worker import load_file
    cases = load_file('_adr_stress_cases', root/'scripts/advection_diffusion_reaction/cases/closed_loop_stress_cases.py')
    params = cases.StressParameters(**problem['stress_parameters'])
    try:
        register_case(problem['case'], lambda: cases.make_case(params, problem['velocity_normalization']))
    except ValueError as error:  # a second system of the same stress class in this process
        if 'already exists' not in str(error):
            raise


def check_system(cp, np, campaign, system, log):
    from scripts.adr_performance_worker import get_space, load_cache
    from scripts.adv_diff_rea_cases import get_case
    from hybridge.linalg.gpu.face_dense import CuPyFaceDenseOperator
    from hybridge.linalg.gpu.preconditioners import CuPyFaceAdditiveSchwarzPreconditioner, CuPyFaceBlockJacobiPreconditioner
    from campaigns.adr_h200_tuned.device_system import (
        asm_from_device, assemble_on_device, block_jacobi_from_device, operator_from_device,
    )
    from campaigns.adr_h200_tuned.device_verify import DeviceVerifier
    problem = system['problem']
    register_stress(problem, ROOT)
    spec = dict(problem, case=problem['case'], cache=str(campaign/'cache'/system['id']), rtol=1e-10, mesh_path=None)
    mesh = campaign/'jobs'/f"mesh_{system['id']}"/'result.json'
    if mesh.exists():
        spec.update(mesh_path=json.loads(mesh.read_text())['mesh_path'], n=json.loads(mesh.read_text())['n'])
    space = get_space(spec)
    kw, exact = get_case(spec['case'])
    started = time.perf_counter()
    assembly = assemble_on_device(cp, space, kw, log=log)
    cp.cuda.Device().synchronize()
    seconds = time.perf_counter()-started
    vendored, get = load_cache(spec['cache'])
    # hybridge's constructors check for hybridge's FaceDenseSystem class, not the vendored one.
    from hybridge.linalg.face_dense import FaceDenseSystem as HostSystem
    cached = HostSystem(blocks=np.asarray(vendored.blocks), neighbors=np.asarray(vendored.neighbors),
                        rhs=np.asarray(vendored.rhs), global_faces=np.asarray(vendored.global_faces),
                        global_to_local=np.asarray(vendored.global_to_local),
                        boundary_trace=np.asarray(vendored.boundary_trace), mode='eliminate')
    for name in ('blocks', 'neighbors', 'rhs', 'global_faces', 'global_to_local', 'boundary_trace'):
        device = cp.asnumpy(getattr(assembly.system, name))
        host = np.asarray(getattr(cached, name))
        if not np.array_equal(device, host):
            worst = float(np.max(np.abs(device-host))/max(np.max(np.abs(host)), 1e-300))
            assert worst <= 1e-14, (system['id'], name, worst)
            log(f'  {name}: not bitwise, max relative difference {worst:.1e}')
    assert np.array_equal(cp.asnumpy(assembly.element_blocks), np.asarray(get('element_blocks')))
    log(f"gate device assembly {system['id']} ({problem['case']}, p={problem['p']}, {space.mesh.num_tri:,} triangles): "
        f'trace system and element blocks equal the cache; {seconds:.2f} s')
    rng = np.random.default_rng(5)
    worst = {}
    for implementation in ('raw_fused', 'interleaved'):
        a = CuPyFaceDenseOperator.from_system(cached, implementation=implementation, dtype=np.float64)
        b = operator_from_device(cp, assembly.system, implementation=implementation)
        for seed in range(2):
            v = cp.asarray(rng.standard_normal(cached.num_dofs))
            worst[f'operator {implementation}'] = max(worst.get(f'operator {implementation}', 0.),
                                                      float(cp.linalg.norm(a.matvec(v)-b.matvec(v))/cp.linalg.norm(a.matvec(v))))
    for application in ('raw', 'interleaved'):
        a = CuPyFaceBlockJacobiPreconditioner.from_system(cached, local_solver='cublas_inverse', application=application)
        b = block_jacobi_from_device(cp, assembly.system, local_solver='cublas_inverse', application=application)
        v = cp.asarray(rng.standard_normal(cached.num_dofs))
        x, y = cp.empty_like(v), cp.empty_like(v)
        a.apply_into(v, x)
        b.apply_into(v, y)
        worst[f'bj {application}'] = float(cp.linalg.norm(x-y)/cp.linalg.norm(x))
    for application in ('fused', 'interleaved', 'interleaved_f32'):
        a = CuPyFaceAdditiveSchwarzPreconditioner.from_system(cached, get('element_blocks'), get('loc2glob'),
                                                              local_solver='cublas_inverse', application=application)
        b = asm_from_device(cp, assembly.system, assembly.element_blocks, assembly.loc2glob,
                            local_solver='cublas_inverse', application=application)
        v = cp.asarray(rng.standard_normal(cached.num_dofs))
        x, y = cp.empty_like(v), cp.empty_like(v)
        a.apply_into(v, x)
        b.apply_into(v, y)
        worst[f'asm {application}'] = float(cp.linalg.norm(x-y)/cp.linalg.norm(x))
    for name, value in worst.items():
        assert value <= (1e-6 if 'f32' in name else 1e-12), (system['id'], name, value)
    log('  device-built solver inputs vs host-built: ' + ', '.join(f'{k} {v:.1e}' for k, v in worst.items()))
    verifier_cache = DeviceVerifier.from_cache(cp, spec, cached, get, exact, budget_bytes=1 << 40, log=lambda *_: None)
    verifier_device = DeviceVerifier.from_device(cp, spec, assembly, space, exact, log=lambda *_: None)
    x = cp.asarray(rng.standard_normal(cached.num_dofs))
    a, b = verifier_cache.verify(x), verifier_device.verify(x)
    for key in ('true_relative_residual', 'l2_error'):
        assert abs(a[key]-b[key]) <= 1e-12*max(abs(a[key]), 1e-300), (system['id'], key, a[key], b[key])
    log(f"  verifier from device vs from cache: residual {a['true_relative_residual']:.6e} vs {b['true_relative_residual']:.6e}, "
        f"L2 {a['l2_error']:.6e} vs {b['l2_error']:.6e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--systems', nargs='+', required=True)
    args = parser.parse_args()
    import numpy as np
    import cupy as cp
    plan = json.loads((args.campaign/'manifest.json').read_text())['plan']
    systems = {s['id']: s for s in plan['systems']}
    log = lambda message: print(message, flush=True)
    for sid in args.systems:
        check_system(cp, np, args.campaign, systems[sid], log)
    print('gate passed', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
