#!/usr/bin/env python3
"""Isolated preparation/solve adapter; only invoked by an explicit campaign run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import re
import sys
import threading
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[2]


class BudgetExceeded(RuntimeError):
    pass


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compatibility(device, runtime_version, backend):
    """Reject architecture/library mismatches; never change requested backend."""
    if backend not in ('cusparse_generic', 'legacy'):
        raise ValueError(f'Unsupported AMGX BSR backend: {backend}')
    if runtime_version < 12000:
        raise ValueError('This patched AMGX source requires CUDA 12 or newer; K80 is unsupported')
    capability = 10*device['major']+device['minor']
    if runtime_version >= 13000 and capability < 75:
        raise ValueError('CUDA 13 libraries do not support this GPU; P100/V100 require CUDA 12')
    if runtime_version >= 12000 and capability < 50:
        raise ValueError('K80/Kepler requires a pre-CUDA-12 stack')
    if backend == 'cusparse_generic' and (runtime_version < 13000 or capability < 75):
        raise ValueError('Requested generic BSR requires CUDA 13.0 Update 1+ and a supported GPU; '
                         'AMU K80/P100/V100 cannot run this path. No silent legacy fallback.')


def check_loaded_cuda_runtime(libraries, runtime_version):
    """Catch accidentally loading a CUDA-13 AMGX build beside CuPy CUDA 12."""
    majors = {int(match.group(1)) for path in libraries
              if (match := re.search(r'libcudart\.so\.(\d+)', Path(path).name))}
    expected = runtime_version//1000
    if majors and majors != {expected}:
        raise ValueError(f'Mixed CUDA runtimes loaded: {sorted(majors)}; CuPy uses CUDA {expected}. '
                         'Rebuild AMGX/PyAMGX for the selected toolkit and fix library paths.')


def probe(spec):
    import cupy as cp
    inventory = load_file('_adr_cuda_inventory', ROOT/'hybridge/runtime/devices.py')
    devices = inventory.discover_cuda_devices(cp.cuda.runtime)
    selected = inventory.select_fp64_device(devices, overrides={int(k): v for k, v in spec.get('fp64_overrides', {}).items()})
    runtime = cp.cuda.runtime.runtimeGetVersion()
    compatibility(selected, runtime, spec['amgx_backend'])
    import pyamgx
    # Reject stock libraries lacking the patched telemetry used by pMG-AMG.
    if not hasattr(pyamgx, 'get_device_memory_stats'):
        raise ValueError('Patched PyAMGX with get_device_memory_stats is required')
    libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if any(name in line for name in ('libamgx', 'libcusparse', 'libcudart'))})
    check_loaded_cuda_runtime(libraries, runtime)
    pyamgx.initialize()
    try:
        for config in spec['amgx_configs']:
            handle = pyamgx.Config().create_from_dict(config)
            handle.destroy()
    finally:
        pyamgx.finalize()
    libraries = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                        if any(name in line for name in ('libamgx', 'libcusparse', 'libcudart'))})
    check_loaded_cuda_runtime(libraries, runtime)
    if spec['amgx_backend'] == 'cusparse_generic':
        import ctypes
        sparse = [p for p in libraries if 'libcusparse.so' in p]
        if not sparse or any(not hasattr(ctypes.CDLL(p), 'cusparseCreateBsr') for p in sparse):
            raise ValueError('Loaded cuSPARSE lacks generic cusparseCreateBsr; need CUDA 13 Update 1+')
    library_hashes = {}
    for path in libraries:
        with Path(path).open('rb') as stream:
            library_hashes[Path(path).name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(status='passed', devices=devices, selected=selected, cuda_runtime=runtime,
                amgx_backend=spec['amgx_backend'],
                cuda_driver=cp.cuda.runtime.driverGetVersion(), cupy=cp.__version__,
                pyamgx_file=pyamgx.__file__, libraries=libraries, library_sha256=library_hashes,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                note='Import/configuration preflight only; compiled kernel coverage awaits execution')


def match_triangle_target(generate, target, max_triangles, size, attempts=16, rtol=.03):
    """Mesh size whose ``generate(size).num_tri`` is within ``rtol`` of ``target`` (and at most ``max_triangles``).

    The count scales like size**-2 on average, which gives the step, but it is steep and step-like
    in places: the five-lobed annulus has 893,947 triangles at size 0.002652 and 964,391 at
    0.002633, so that step jumps over a 3% band and then cycles. The sizes already known to give
    too few and too many triangles bracket the answer; a step that leaves the bracket is replaced
    by a bisection in log size, so trajectories that never leave it are unchanged.
    """
    coarse, fine = math.inf, 0.  # sizes known to give too few / too many triangles
    for attempt in range(attempts):
        generated = generate(size)
        count = generated.num_tri
        print(f'mesh attempt={attempt} size={size:.6g} triangles={count}', flush=True)
        if abs(count/target-1) <= rtol and count <= max_triangles:
            return generated, size
        if count < target:
            coarse = min(coarse, size)
        else:
            fine = max(fine, size)
        step = size*math.sqrt(count/target)
        size = step if fine < step < coarse or not (fine > 0 and math.isfinite(coarse)) else math.sqrt(fine*coarse)
    raise ValueError('Could not match L5 annulus triangle target')


def mesh(spec, common):
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from hybridge.core.mesh import gmsh_smooth_star_mesh, as_dg_mesh
    system, directory = spec['system'], Path(spec['mesh_directory'])
    directory.mkdir(parents=True, exist_ok=True)
    definition = system['mesh']
    if definition.get('generator') == 'stress':
        from scripts.advection_diffusion_reaction.meshes.closed_loop_stress_mesh import prepare_mesh
        from scripts.advection_diffusion_reaction.cases.closed_loop_stress_cases import StressParameters
        path, info = prepare_mesh(StressParameters(**system['problem']['stress_parameters']),
                                 definition['target_triangles'], directory,
                                 boundary_points=definition['boundary_points'], neck_elements=definition['neck_elements'],
                                 max_triangles=definition['max_triangles'])
        if not info['neck_size_screen_passed']:
            raise ValueError('L5 stress neck resolution screen failed')
    elif definition.get('generator') == 'five_lobed':
        target = definition['target_triangles']
        generated, size = match_triangle_target(
            lambda size: gmsh_smooth_star_mesh(size, radius=1., amplitude=.35, mode=5, hole_radius=.58,
                                               boundary_points=definition['boundary_points'], cache=False,
                                               num_threads=1, log_cache=False),
            target, definition['max_triangles'], math.sqrt(9.3/target))
        path = directory/'mesh.npz'
        np.savez(path, node_coords=generated.node_coords, triangles=generated.triangles)
        info = dict(target_triangles=target, triangles=int(generated.num_tri), mesh_size=size)
    else:
        path = Path(spec['inventory_directory'])/definition['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != definition['sha256']:
            raise ValueError('Archived mesh asset changed')
        info = dict(triangles=system['triangles'])
    with np.load(path, allow_pickle=False) as data:
        actual = as_dg_mesh((data['node_coords'], data['triangles']))
    return dict(status='passed', mesh_path=str(path.resolve()), mesh_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                triangles=int(actual.num_tri), trace_dofs=len(actual.int_edges_inds)*(system['problem']['p']+1),
                n=math.ceil(math.sqrt(actual.num_tri/2)), diagnostics=info)


def run_numerical(spec, common):
    # Import the branch case registry first; pMG adapter subsequently selects
    # master numerics. Do not import master's hybridge before that selection.
    sys.path.insert(0, spec['branch_root'])
    # Run as a script, so the checkout root is not on the path without PYTHONPATH (frozen
    # copies); append it after the branch so `campaigns` imports and vendored `scripts` win.
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    if 'stress_parameters' in spec:
        from scripts.adv_diff_rea_cases import register_case
        cases = load_file('_adr_stress_cases', ROOT/'scripts/advection_diffusion_reaction/cases/closed_loop_stress_cases.py')
        params = cases.StressParameters(**spec['stress_parameters'])
        register_case(spec['case'], lambda: cases.make_case(params, spec['velocity_normalization']))
    stress_worker = load_file('_adr_stress_worker', ROOT/'campaigns/adr_stress/closed_loop_stress_worker.py')
    operator_hash = stress_worker.operator_hash
    # Device campaigns (adr_h200_tuned): device checks and measures, NVML sampling, per-phase times.
    device_checks = os.environ.get('HYBRIDGE_ADR_DEVICE_CHECKS') == '1'
    worker_phases = {}

    def timed(name, function):
        started = perf_counter()
        try:
            return function()
        finally:
            worker_phases[name] = worker_phases.get(name, 0.)+perf_counter()-started
            if device_checks:
                print(f'[worker] {name}: {worker_phases[name]:.3f} s', flush=True)

    if spec.get('mesh_path'):
        if timed('mesh_identity', lambda: hashlib.sha256(Path(spec['mesh_path']).read_bytes()).hexdigest()) != spec['mesh_sha256']:
            raise ValueError('Mesh changed after preparation')
    phase = spec['stage']
    expected = spec.get('expected_operator_sha256')
    if phase == 'system':
        expected = None  # the system is assembled here; its hash is taken at the one-way export
    if expected:
        files = spec.get('expected_cache_files')
        if os.environ.get('HYBRIDGE_ADR_CACHE_IDENTITY') == 'metadata' and files:
            # The runner's own resume check (size and mtime of every cached array) instead of
            # re-hashing several GB on the host in every job; the hash was taken at assembly.
            current = timed('operator_identity', lambda: {p.name: dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                                                          for p in sorted(Path(spec['cache']).glob('*.npy'))})
            if current != files:
                raise ValueError('Operator/RHS cache changed after assembly (array sizes or mtimes)')
            spec = dict(spec, verified_operator_sha256=expected, operator_identity='cache_metadata')
        else:
            if timed('operator_identity', lambda: operator_hash(spec['cache'])) != expected:
                raise ValueError('Operator/RHS changed after assembly')
            spec = dict(spec, verified_operator_sha256=expected, operator_identity='sha256')
    inventory = load_file('_adr_cuda_inventory', ROOT/'hybridge/runtime/devices.py')
    estimate = spec['memory_estimate']
    host_need = estimate['assembly_host_bytes' if phase in ('assemble', 'system') else 'solver_host_bytes']
    available = inventory.host_available_bytes()
    limits = [spec['memory_fraction']*available] if available is not None else []
    if spec.get('host_limit_gib'):
        limits.append(spec['host_limit_gib']*2**30)
    if limits and host_need > min(limits):
        raise BudgetExceeded(f'Host estimate {host_need/2**30:.2f} GiB exceeds live/explicit budget')
    import cupy as cp
    cp.cuda.Device(0).use()
    free, total = cp.cuda.runtime.memGetInfo()
    budget = min(spec['memory_fraction']*free, free-spec['reserve_gpu_gib']*2**30)
    assembly = spec['assembly_backend']
    if phase == 'assemble' and assembly == 'auto':
        assembly = 'cupy' if estimate['assembly_device_bytes'] <= budget else 'numba'
        spec = dict(spec, assembly_backend=assembly)
    if phase == 'system':
        # Assembly, then every candidate with the trace system, element blocks (until after
        # ASM) and verification data resident on the device.
        from campaigns.adr_h200_tuned.device_system import retained_bytes
        retained = retained_bytes(spec['triangles'], spec['p'])
        solver = max(c['memory_estimate']['solver_device_bytes'] for c in spec['candidates'])
        need = max(estimate['assembly_device_bytes'], retained['during_asm']+solver)
    else:
        need = (estimate['assembly_device_bytes'] if assembly == 'cupy' else 0) if phase == 'assemble' else estimate['solver_device_bytes']
    if need > budget:
        raise BudgetExceeded(f'Device estimate {need/2**30:.2f} GiB exceeds budget {budget/2**30:.2f} GiB')
    minimum_free = [free]
    stop = threading.Event()
    def monitor():
        cp.cuda.Device(0).use()
        while not stop.wait(1):
            try:
                minimum_free[0] = min(minimum_free[0], cp.cuda.runtime.memGetInfo()[0])
            except Exception:
                break
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    sampler = None
    if os.environ.get('HYBRIDGE_ADR_GPU_SAMPLING') == '1' and phase != 'system':  # the system worker samples per candidate
        from campaigns.adr_h200_tuned.gpu_sampler import GpuSampler
        sampler = GpuSampler(cp.cuda.Device(0).pci_bus_id).start()
    if os.environ.get('HYBRIDGE_ADR_SINGLE_GUARD') == '1':
        # The checks above use the measured calibration; the vendored guard repeats them with
        # the planning model only (6x host assembly bytes, cache bytes as resident at solve time)
        # and would exclude systems that fit. Keep its estimates, drop its veto.
        import scripts.adr_performance_worker as performance
        performance.guard = lambda spec, phase: performance.estimates(
            spec['n'], spec['p'], performance.solver_workspace_degree(spec))
    if device_checks and phase == 'assemble':
        # Store the DG geometry the device checks need, from the space the assembly built,
        # so no solver job rebuilds the space on the host.
        import scripts.adr_performance_worker as performance
        from campaigns.adr_h200_tuned.device_verify import save_geometry
        save_cache = performance.save_cache

        def save_cache_with_geometry(cache, assembly, space):
            timed('save_cache', lambda: save_cache(cache, assembly, space))
            timed('save_geometry', lambda: save_geometry(cache, space))
        performance.save_cache = save_cache_with_geometry
    try:
        if phase == 'system':
            from campaigns.adr_h200_tuned.system_worker import run
            result = run(spec, common, ROOT, log=lambda message: print(message, flush=True))
        elif phase == 'assemble':
            from scripts.adr_performance_worker import assemble
            result = timed('assemble', lambda: assemble(spec))
            result['operator_sha256'] = timed('operator_hash', lambda: operator_hash(spec['cache']))
        elif spec['family'] == 'raw_assembly':
            from campaigns.adr_h200_tuned.raw_assembly_timing import measure
            result = timed('measure', lambda: measure(spec, master_root=ROOT))
        elif device_checks and spec['family'] == 'native_hp':
            from campaigns.adr_h200_tuned.device_native_hp import measure
            result = timed('measure', lambda: measure(spec, master_root=ROOT))
        elif device_checks:
            from campaigns.adr_h200_tuned.device_measure import measure
            result = timed('measure', lambda: measure(spec))
        elif spec.get('tuned_face_dense'):
            from campaigns.adr_h200_tuned.tuned_face_dense import measure
            result = measure(spec)
        elif spec['family'] == 'native_hp':
            from campaigns.adr_unified.native_hp_worker import measure
            result = measure(spec, master_root=ROOT)
        else:
            from scripts.adr_solver_comparison_worker import measure
            result = measure(spec)
    finally:
        stop.set()
        watcher.join()
        memory = dict(assembly_backend=assembly, live_memory_before=dict(free_bytes=free, total_bytes=total),
                      sampled_device_peak_bytes=total-minimum_free[0],
                      memory_note='1-second whole-device samples, including context/other processes; not allocator-exact peaks')
        if sampler is not None:
            sampler.stop()
            memory['gpu_utilization'] = usage = sampler.summary()
            if usage.get('samples'):
                print(f"[gpu] {usage['samples']} samples: SM busy mean {usage['sm_busy_mean']:.0f}%, median "
                      f"{usage['sm_busy_median']:.0f}%, p10 {usage['sm_busy_p10']}%, below 10% for "
                      f"{100*usage['idle_share_below_10']:.0f}% of the job", flush=True)
        if worker_phases:
            memory['worker_phases'] = worker_phases
        common.atomic_json(Path(spec['result']).with_suffix('.memory.json'), memory)
    result.update(memory)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    common = load_file('_adr_unified_common', Path(spec['branch_root'])/'scripts/adr_performance_common.py')
    started = perf_counter()
    try:
        if spec['stage'] == 'preflight':
            result = probe(spec)
        elif spec['stage'] == 'mesh':
            result = mesh(spec, common)
        else:
            result = run_numerical(spec, common)
    except Exception as exc:
        text = str(exc)
        status = ('memory_excluded' if type(exc).__name__ == 'BudgetExceeded' else
                  'out_of_memory' if isinstance(exc, MemoryError) or 'out of memory' in text.lower() else 'error')
        try:
            result = common.read_json(spec['result'])
        except (OSError, ValueError):
            result = {}
        result.update(status=status, error=text, traceback=traceback.format_exc())
    memory_path = Path(spec['result']).with_suffix('.memory.json')
    if memory_path.exists():
        result.update(common.read_json(memory_path))
    result.update(worker_wall_seconds=perf_counter()-started,
                  peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if 'amgx_backend' in spec:
        result['amgx_backend'] = spec['amgx_backend']
    common.atomic_json(spec['result'], result)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
