#!/usr/bin/env python3
"""One CPU ADR HDG pipeline point: Numba assembly, PARDISO, Numba reconstruction and postprocessing.

Called by run.py in a fresh process whose NUMBA/MKL/OpenMP thread counts and CPU affinity
are already set. Records the interpreter and import time, the case/mesh/space construction,
then one cold solve (first in the process: Numba cache load or JIT compile, first-touch
allocations, MKL initialisation) and REPEATS warm solves. Every solve uses a fresh solver
object, so nothing of the previous solve (local columns, PARDISO analysis) is reused: each
is the full pipeline. Every stage and substep the solver times is recorded, together with
process CPU time, RSS and L2 errors of the primal, diffusive-flux and postprocessed fields.

    python campaigns/cpu_adr_scaling/point.py --spec SPEC.json --output DIR
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def process_age_seconds():
    """Seconds since this process was exec'd: interpreter startup plus everything imported so far."""
    with open('/proc/self/stat') as stream:
        start_ticks = int(stream.read().rsplit(')', 1)[1].split()[19])
    with open('/proc/uptime') as stream:
        uptime = float(stream.read().split()[0])
    return uptime - start_ticks/os.sysconf('SC_CLK_TCK')


BEFORE_IMPORTS = process_age_seconds()

from campaigns.common.reporting import event  # noqa: E402


def status_kib(key):
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(key + ':'):
            return int(line.split()[1])
    return 0


def memory():
    return dict(rss_gib=status_kib('VmRSS')/1024**2, hwm_gib=status_kib('VmHWM')/1024**2,
                maxrss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)


def configure_blas():
    """NumPy's OpenBLAS keeps the run's thread count; scipy's OpenBLAS, the one Numba's np.linalg
    and @ call inside prange kernels, runs single-threaded there (nested OpenBLAS threads from
    Numba workers segfault). Returns what threadpoolctl sees, for the record."""
    import scipy.linalg  # noqa: F401 - loads scipy's OpenBLAS so it can be limited
    from threadpoolctl import ThreadpoolController
    controller = ThreadpoolController()
    for library in controller.lib_controllers:
        if library.internal_api == 'openblas' and 'scipy.libs' in library.filepath:
            library.set_num_threads(1)
    return [dict(api=library.internal_api, threads=library.num_threads, version=library.version,
                 library=library.filepath.split('site-packages/')[-1]) for library in ThreadpoolController().lib_controllers]


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=1, sort_keys=True, default=float))
    temporary.replace(path)


def run(spec, output):
    report = dict(status='running', spec=spec, startup=dict(before_imports_seconds=BEFORE_IMPORTS), solves=[])

    def stage(name, message='', **fields):
        snapshot = memory()
        report['stage'] = name
        write_json(output/'result.json', report)
        event(output, name, message=f"{name} {message}RSS={snapshot['rss_gib']:.2f} GiB", **snapshot, **fields)

    try:
        start = time.perf_counter()
        import numpy as np
        import numba
        report['startup']['numpy_numba_seconds'] = time.perf_counter()-start
        start = time.perf_counter()
        from hybridge import DGSpace, AdvectionDiffusionReactionHDGSolver, AdvectionDiffusionReactionHDGOptions
        from hybridge.core.mesh import as_dg_mesh, rectangle_mesh
        import hybridge.solvers.advection_diffusion_reaction as adr
        from hybridge.linalg.direct import clear_pypardiso_cache
        from scripts.advection_diffusion_reaction.cases import catalogue, closed_loop_stress_cases as stress
        from hybridge.cases.closed_loop_pointwise import closed_loop_pointwise_case
        report['startup']['hybridge_seconds'] = time.perf_counter()-start
        report['startup']['total_seconds'] = process_age_seconds()
        blas = configure_blas()
        report['environment'] = dict(blas=blas,
            numba_threads=numba.get_num_threads(), numba_version=numba.__version__,
            affinity=len(os.sched_getaffinity(0)), host=os.uname().nodename,
            threads_env={k: os.environ.get(k) for k in ('NUMBA_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
                                                        'OPENBLAS_NUM_THREADS', 'NUMBA_CACHE_DIR',
                                                        'NUMBA_THREADING_LAYER', 'PYTHONPYCACHEPREFIX')})
        if numba.get_num_threads() != spec['threads']:
            raise RuntimeError(f"Numba threads {numba.get_num_threads()} != requested {spec['threads']}")
        stage('imported', f"in {report['startup']['total_seconds']:.1f} s; ", startup=report['startup'])

        # Every stage (_timed_call) and substep (_timed_substep) the ADR solve times, in order.
        steps = []

        def recorded(kind, original):
            def wrapper(label, *args, **kwargs):
                value, seconds = original(label, *args, **kwargs)
                if not label.startswith(' '):  # _timed_substep re-enters _timed_call with an indented label
                    steps.append(dict(kind=kind, label=label, seconds=seconds))
                return value, seconds
            return wrapper
        adr._timed_call = recorded('stage', adr._timed_call)
        adr._timed_substep = recorded('substep', adr._timed_substep)

        # The device campaign's system: its case (stress parameters and frozen velocity normalization
        # included), mesh file or square size, degree and quadrature, built as in the tuned campaign's
        # raw-CUDA rows (campaigns/adr_h200_tuned/raw_assembly_timing.py:problem_for).
        start = time.perf_counter()
        # Compiled, multithreaded coefficients (hybridge PointwiseLaw) unless the NumPy closures are asked
        # for or the case has none; the NumPy closures sample on one core.
        compiled = spec.get('coefficients', 'numba') == 'numba'
        if 'stress_parameters' in spec:
            if compiled:
                kwargs, exact = closed_loop_pointwise_case(**spec['stress_parameters'],
                                                           normalization=spec['velocity_normalization'])
            else:
                kwargs, exact = stress.make_case(stress.StressParameters(**spec['stress_parameters']),
                                                 spec['velocity_normalization'])
            problem = catalogue.AdvectionDiffusionReactionProblem(**kwargs, exact=exact)
        else:
            problem = catalogue.compiled_case(spec['case']) if compiled else None
            compiled = problem is not None
            problem = problem or catalogue.CASE_DEFINITIONS[spec['case']].build()
        report['coefficients'] = 'pointwise_law' if compiled else 'numpy'
        case_seconds = time.perf_counter()-start
        start = time.perf_counter()
        if spec.get('mesh_path'):
            with np.load(spec['mesh_path'], allow_pickle=False) as data:
                mesh = as_dg_mesh((data['node_coords'], data['triangles']))
        else:
            mesh = rectangle_mesh(spec['n'], spec['n'])
        mesh_seconds = time.perf_counter()-start
        start = time.perf_counter()
        space = DGSpace(mesh, spec['p'], basis_type='dub_orth', volume_quad_1d=spec.get('volume_quad_1d'),
                        edge_quad_1d=spec.get('edge_quad_1d'))
        space_seconds = time.perf_counter()-start
        # Legacy square cases give a constant velocity as numbers; the solver takes callables or fields.
        beta = tuple(value if callable(value) else space.constant(float(value)) for value in problem.beta)
        report['setup'] = dict(case_seconds=case_seconds, mesh_seconds=mesh_seconds, space_seconds=space_seconds,
                               triangles=int(mesh.num_tri), el_dof=int(space.el_dof))
        stage('space_built', f"{mesh.num_tri:,} triangles p={spec['p']} (mesh {mesh_seconds:.2f} s, "
                             f"space {space_seconds:.2f} s); ", setup=report['setup'])
        options = AdvectionDiffusionReactionHDGOptions(
            diffusion=problem.diffusion, assembly_backend='numba', reconstruction_backend='numba',
            postprocessing_backend='numba', hdg_postprocess='both', solver='pypardiso',
            # The device campaign's acceptance tolerance; the 1e-13 default rejects exact LU
            # solutions of the ill-conditioned stress systems (residual ~1e-13).
            solver_rtol=spec.get('rtol', 1e-10),
            pardiso_threads=spec['pardiso_threads'], verbose=spec['verbosity'])

        for index in range(1 + spec['repeats']):
            kind = 'cold' if index == 0 else 'warm'
            steps.clear()
            # PyPardiso's process-wide solver skips factorization when it sees the matrix it factored
            # last; release it so every solve, warm ones included, analyses and factorizes afresh.
            clear_pypardiso_cache()
            gc.collect()
            stage('solve_started', f"{kind} {index}; ", kind=kind, index=index)
            solver = AdvectionDiffusionReactionHDGSolver(
                space, source=problem.source, beta=beta, reaction=problem.reaction,
                boundary_condition=problem.boundary_condition, options=options)
            cpu, wall = time.process_time(), time.perf_counter()
            result = solver.solve()
            wall, cpu = time.perf_counter()-wall, time.process_time()-cpu
            timings = asdict(result.timings)
            details = timings.pop('details') or {}
            solve = dict(kind=kind, index=index, wall_seconds=wall, cpu_seconds=cpu, cpu_wall_ratio=cpu/max(wall, 1e-12),
                         timings_seconds=dict(timings, assembly=result.timings.assembly),
                         details=details, steps=list(steps), memory=memory())
            if index == 0:
                linear = result.global_solve_result
                report['trace_system'] = dict(trace_dofs=int(np.asarray(result.trace).size),
                                        matrix_format=result.matrix_format,
                                        relative_residual=None if linear is None else linear.physical_relative_residual_norm)
                report['errors'] = dict(
                    primal_l2=None if problem.exact is None else float(result.field.l2_error(problem.exact)),
                    diffusive_flux_l2=None if problem.exact_flux is None
                    else float(result.flux.l2_error(problem.exact_flux)),
                    postprocessed_primal_l2=None if result.postprocessed_field is None or problem.exact is None
                    else float(result.postprocessed_field.l2_error(problem.exact)))
                report['environment']['numba_threading_layer'] = numba.threading_layer()
            report['solves'].append(solve)
            t = solve['timings_seconds']
            stage('solve_finished', f"{kind} {index}: wall {wall:.3f} s (cpu/wall {solve['cpu_wall_ratio']:.1f}) = "
                                    f"preparation {t['preparation']:.3f} + local {t['local_solver']:.3f} + "
                                    f"trace assembly {t['trace_assembly']:.3f} + solve {t['solve']:.3f} + "
                                    f"reconstruction {t['reconstruction']:.3f} + postprocessing {t['postprocessing']:.3f}; ",
                  kind=kind, index=index, wall_seconds=wall, timings_seconds=t,
                  steps=[(s['label'], round(s['seconds'], 4)) for s in steps])
            del result, solver
        report['status'] = 'passed'
        stage('finished')
        return 0
    except Exception as error:
        report.update(status='error', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        stage('failed')
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    return run(json.loads(args.spec.read_text()), args.output)


if __name__ == '__main__':
    raise SystemExit(main())
