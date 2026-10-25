#!/usr/bin/env python3
"""Bounded CPU checks of archived equilibria using the normal GC runner.

Defaults are a smoke check, not a spatially converged or long-time study.
One process and one run at a time; no GPU use. Full original meshes can be
selected explicitly by --original-mesh and raising --max-triangles.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import json
import os
import resource
from pathlib import Path
import sys
import threading
import time
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hybridge.core.projection import project_callable
from hybridge.linalg.pardiso_runtime import pardiso_thread_limit
from scripts.guiding_center.cases.imported_equilibrium import imported_equilibrium
from scripts.guiding_center.cases.guiding_center_presets import GuidingCenterRunPreset
from scripts.guiding_center.runtime.runner import run_guiding_center_case


class CPUActivity:
    """Observe actual process/thread CPU use during a bounded run."""
    def __enter__(self):
        ticks = os.sysconf('SC_CLK_TCK')
        def thread_cpu():
            values = {}
            for path in Path('/proc/self/task').glob('*/stat'):
                try:
                    fields = path.read_text().rsplit(')', 1)[1].split()
                    values[path.parent.name] = (int(fields[11])+int(fields[12]))/ticks
                except (FileNotFoundError, ProcessLookupError):
                    pass
            return values
        self.stop = threading.Event()
        self.peak_active_threads = 0
        self.peak_cpu_cores = 0.
        self.peak_rss_bytes = 0
        self.started, self.cpu_started = time.perf_counter(), time.process_time()
        def monitor():
            previous = thread_cpu()
            wall = time.perf_counter()
            while not self.stop.wait(.2):
                now = time.perf_counter()
                current = thread_cpu()
                increments = [max(0., value-previous.get(key, value)) for key, value in current.items()]
                self.peak_active_threads = max(self.peak_active_threads, sum(v > .005 for v in increments))
                self.peak_cpu_cores = max(self.peak_cpu_cores, sum(increments)/(now-wall))
                self.peak_rss_bytes = max(self.peak_rss_bytes, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
                previous, wall = current, now
        self.monitor = threading.Thread(target=monitor, daemon=True)
        self.monitor.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.monitor.join()
        self.wall_seconds = time.perf_counter()-self.started
        self.cpu_seconds = time.process_time()-self.cpu_started

    def data(self):
        return {key: getattr(self, key) for key in ("wall_seconds", "cpu_seconds",
            "peak_active_threads", "peak_cpu_cores", "peak_rss_bytes")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--equilibria', default='star_02,uglypacman_04,iter_02,horseshoe_02')
    p.add_argument('--schemes', default='si-euler,si-bdf2')
    p.add_argument('--orders', default='2')
    p.add_argument('--dt', type=float, default=.05)
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--remesh-size', type=float, default=.12)
    p.add_argument('--iter-remesh-size', type=float, default=.6)
    p.add_argument('--original-mesh', action='store_true')
    p.add_argument('--max-triangles', type=int, default=20000)
    p.add_argument('--quadrature-1d', type=int, default=12)
    p.add_argument('--threads', type=int, default=16, help='Numba, host assembly and Pardiso threads (verified)')
    p.add_argument('--poisson-cache-local-factors', choices=['none', 'schur-lu', 'schur-cholesky'],
                   default='none', help='Optional Numba local Schur factor cache for higher orders')
    p.add_argument('--linear-rtol', type=float, default=1e-12,
                   help='Physical relative residual target for both condensed direct solves')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    if args.steps < 2 or args.dt <= 0 or args.max_triangles < 1:
        p.error('positive dt/max-triangles and at least two steps required')
    if not np.isfinite(args.linear_rtol) or not 0 < args.linear_rtol < 1:
        p.error('linear-rtol must be finite and between zero and one')
    orders = [int(v) for v in args.orders.split(',')]
    schemes = args.schemes.split(',')
    if set(schemes)-{'si-euler', 'si-bdf2', 'si-bdf3'} or min(orders) < 1:
        p.error('unsupported scheme or order')
    if args.quadrature_1d < max(orders)+1:
        p.error('quadrature must be at least order+1')
    if args.dry_run:
        print(json.dumps(vars(args), default=str, indent=2))
        return
    import numba
    from hybridge.runtime.threads import set_host_threads
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    requested = min(args.threads, available)
    if requested < 1 or requested > numba.config.NUMBA_NUM_THREADS:
        p.error('threads must fit the Numba pool; launch with NUMBA_NUM_THREADS at least --threads')
    numba.set_num_threads(requested)
    host_count = set_host_threads(requested)
    if numba.get_num_threads() != requested or host_count != requested:
        raise RuntimeError('assembly thread count could not be set')
    thread_settings = dict(numba=numba.get_num_threads(), host_numpy=host_count,
                           pardiso_requested=requested, available_cpus=available)
    # Do not reuse/overwrite a previous result directory.
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(scope='short unperturbed preservation check; no long-time certification',
                    command=sys.argv, settings=vars(args), assembly_threads=thread_settings, runs=[])
    def save():
        (args.output/'summary.json').write_text(json.dumps(manifest, default=str, indent=2)+'\n')
    save()
    for name in args.equilibria.split(','):
        h = None if args.original_mesh else (args.iter_remesh_size if name.startswith('iter') else args.remesh_size)
        case = imported_equilibrium(archive=args.archive, equilibrium=name, remesh_size=h)
        mesh = case.imported_equilibrium.to_mesh()
        print(f'[equilibrium] {name}: {mesh.num_tri} target triangles', flush=True)
        if mesh.num_tri > args.max_triangles:
            raise RuntimeError(f'{mesh.num_tri} triangles exceeds explicit cap {args.max_triangles}')
        for order in orders:
            for scheme in schemes:
                key = f'{name}_p{order}_{scheme}'
                config = GuidingCenterRunPreset(case='imported_equilibrium', description=key,
                    case_params=dict(archive=str(args.archive.resolve()), equilibrium=name, remesh_size=h),
                    mesh_size=h or float(np.max(np.linalg.norm(mesh.element_vertices[:, 1]-mesh.element_vertices[:, 0], axis=1))), order=order, time_scheme=scheme, dt=args.dt, num_steps=args.steps,
                    initial_projection_quad_1d=args.quadrature_1d, volume_quad_1d=args.quadrature_1d,
                    trace_basis='legendre-modal', poisson_assembly_backend='numba',
                    poisson_local_backend='numba', poisson_solver='pypardiso',
                    poisson_cache_local_factors=args.poisson_cache_local_factors,
                    poisson_solver_rtol=args.linear_rtol, transport_solver_rtol=args.linear_rtol,
                    transport_assembly_backend='numba', transport_solver='pypardiso',
                    poisson_scale_system=False, transport_scale_system=False,
                    poisson_tau_max_retries=0, positivity_diagnostics=True,
                    diagnostics_dir=str(args.output), diagnostics_prefix=key, verbosity=1)
                (args.output/f'{key}_config.json').write_text(json.dumps(asdict(config), indent=2)+'\n')
                initial_phi = None
                def observe(snapshot):
                    nonlocal initial_phi
                    if initial_phi is None:
                        initial_phi = snapshot.equilibrium_potential.coeffs.copy()
                with pardiso_thread_limit(args.threads) as threads, CPUActivity() as activity:
                    result = run_guiding_center_case(config, preset_key=key, step_observer=observe, case=case)
                initial = project_callable(case.initial_density, result.space,
                                           volume_quad_1d=args.quadrature_1d)
                np.savez_compressed(args.output/f'{key}_fields.npz', nodes=mesh.node_coords,
                    triangles=mesh.triangles, order=order, basis=config.basis, rho_initial=initial.coeffs,
                    rho_final=result.final_density.coeffs, phi_initial=initial_phi,
                    phi_final=result.final_potential.coeffs, time=args.dt*args.steps)
                first, final = result.diagnostics[0], result.diagnostics[-1]
                metrics = {k: v for k, v in first.items() if k.startswith('equilibrium_')}
                metrics.update({k: final[k] for k in ('diocotron_rho_eq_relative_l2',
                    'diocotron_phi_eq_relative_l2', 'mass_relative_drift', 'energy_relative_drift',
                    'enstrophy_relative_drift', 'rho_min', 'rho_max')})
                metrics['maximum_poisson_physical_relative_residual'] = max(
                    r.get('poisson_physical_rel_residual') or 0. for r in result.diagnostics)
                manifest['runs'].append(dict(name=name, key=key, order=order, scheme=scheme,
                    triangles=mesh.num_tri, dt=args.dt, steps=args.steps, time=args.dt*args.steps,
                    pardiso_threads=threads, assembly_threads=thread_settings, activity=activity.data(), metrics=metrics,
                    diagnostics=f'{key}.jsonl', fields=f'{key}_fields.npz'))
                save()
                print(json.dumps(manifest['runs'][-1], default=str), flush=True)
                del result, initial
                gc.collect()
        del case, mesh
        gc.collect()


if __name__ == '__main__':
    main()
