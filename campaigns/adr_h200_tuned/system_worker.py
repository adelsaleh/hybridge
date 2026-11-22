"""One process per system: assemble on the GPU, export for PyPardiso, run every candidate in place.

The unified campaign ran one process per solver job, each reading the assembled system back
from the cache on scratch (device -> host -> disk -> host -> device). Here the trace system,
the element blocks and the verification data stay on the device from assembly to the last
candidate (``device_system``). The cache receives the trace system once, in one direction, for
the host PyPardiso reference.

Per candidate (asm_pp first, so the element blocks can be released after the ASM setups):
its own wall-clock budget (``candidate_timeout``, SIGALRM), its own NVML utilization window,
the same record fields as the per-job workers, and ``jobs/<system>_<candidate>/result.json``
written as the runner would. Candidates whose final result already exists with a non-retryable
status are skipped, so a resumed system job reruns only what is missing.
"""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import resource
import signal
import threading
from time import perf_counter
import traceback

FAMILY_ORDER = {'asm_pp': 0, 'bj_pp': 1, 'amgx': 2, 'native_hp': 3}
RETRYABLE = {'running', 'interrupted', 'process_error'}


class CandidateTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise CandidateTimeout()


def run(spec, common, master_root, log=print):
    import numpy as np
    import cupy as cp
    from scripts.adr_performance_worker import get_space, metadata
    from scripts.adv_diff_rea_cases import get_case
    from campaigns.adr_unified.adr_unified_plan import fingerprint
    from campaigns.common.reporting import convergence_status
    from campaigns.adr_h200_tuned.device_system import (
        SolveContext, assemble_on_device, export_for_pardiso, retained_bytes,
    )
    from campaigns.adr_h200_tuned.device_verify import DeviceVerifier
    from campaigns.adr_h200_tuned.gpu_sampler import GpuSampler

    started = perf_counter()
    output_dir = Path(spec['output'])
    result = dict(status='running', stage='system', system_id=spec['system_id'], case=spec['case'], n=spec['n'],
                  p=spec['p'], candidates={}, environment=metadata({**spec, 'engine': 'gpu'}),
                  worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def save():
        common.atomic_json(spec['result'], result)

    phases = {}

    def timed(name, function):
        cp.cuda.Device().synchronize()
        t = perf_counter()
        try:
            return function()
        finally:
            cp.cuda.Device().synchronize()
            phases[name] = phases.get(name, 0.)+perf_counter()-t
            log(f'[system] {name}: {phases[name]:.3f} s')

    sampler = GpuSampler(cp.cuda.Device(0).pci_bus_id).start()
    try:
        space = timed('space_setup', lambda: get_space(spec))
        kw, exact = get_case(spec['case'])
        assembly = timed('assembly', lambda: assemble_on_device(cp, space, kw, log=log))
        system = assembly.system
        cache = Path(spec['cache'])
        triangles = int(space.mesh.num_tri)
        result['assembly'] = dict(
            status='exporting', kind='assembly', route='device', triangles=triangles, trace_dofs=system.num_dofs,
            assembly_median_ms=1000*assembly.timings['assembly_total'], assembly_timings_ms={k: 1000*v for k, v in assembly.timings.items()},
            info=assembly.info, retained_device_bytes=retained_bytes(triangles, spec['p']),
            host_callable_fallbacks=assembly.info.get('host_callable_fallbacks', []))
        save()
        # The PyPardiso export only reads the device arrays and writes to scratch, whose
        # write-back stalls for a fixed ~307 s at times: it runs beside the solves and is
        # joined at the end. The candidates' operator_sha256 is filled in then.
        export = dict(started=perf_counter())

        def run_export():
            try:
                export['files'], export['operator_sha256'], export['info'] = export_for_pardiso(cp, cache, system, log=log)
            except Exception as exc:  # recorded on the assembly, never fatal for the solves
                export['error'] = f'{type(exc).__name__}: {exc}'
                export['traceback'] = traceback.format_exc()
            export['seconds'] = perf_counter()-export['started']

        exporter = threading.Thread(target=run_export, name='pardiso-export', daemon=True)
        exporter.start()
        verifier = timed('verifier_setup', lambda: DeviceVerifier.from_device(cp, spec, assembly, space, exact, log=log))
        context = SolveContext(cp=cp, system=system, element_blocks=assembly.element_blocks,
                               loc2glob=assembly.loc2glob, verifier=verifier)
        del assembly
        candidates = sorted(spec['candidates'], key=lambda c: (FAMILY_ORDER.get(c['family'], 9), c['candidate']))
        last_asm = max((i for i, c in enumerate(candidates) if c['family'] == 'asm_pp'), default=-1)
        signal.signal(signal.SIGALRM, _alarm)
        for index, candidate in enumerate(candidates):
            key = f"{spec['system_id']}_{candidate['candidate']}"
            final = output_dir/'jobs'/key/'result.json'
            if final.exists():
                prior = json.loads(final.read_text())
                if prior.get('status') not in RETRYABLE:
                    result['candidates'][key] = prior['status']
                    log(f'[candidate] reuse {key}: {prior["status"]}')
                    continue
            final.parent.mkdir(parents=True, exist_ok=True)
            cspec = dict(spec, **candidate, stage='solve', verified_operator_sha256='pending-export',
                         operator_identity='in_process', result=str(final))
            cspec.pop('candidates', None)
            identity = fingerprint({k: v for k, v in cspec.items() if k != 'result'})
            common.atomic_json(final, dict(status='running', specification_sha256=identity, attempt=spec.get('attempt', 1)))
            result['candidates'][key] = 'running'
            save()
            log(f"[candidate] start {key} ({candidate['family']}), budget {spec['candidate_timeout']:.0f} s")
            window_start = sampler.now()
            t = perf_counter()
            signal.setitimer(signal.ITIMER_REAL, float(spec['candidate_timeout']))
            try:
                if candidate['family'] == 'native_hp':
                    from campaigns.adr_h200_tuned.device_native_hp import measure
                    record = measure(cspec, master_root=master_root, log=log, context=context)
                else:
                    from campaigns.adr_h200_tuned.device_measure import measure
                    record = measure(cspec, log=log, context=context)
            except CandidateTimeout:
                record = _partial(common, final)
                record.update(status='timeout', error=f"candidate budget of {spec['candidate_timeout']:.0f} s exceeded")
            except MemoryError as exc:
                record = _partial(common, final)
                record.update(status='out_of_memory', error=str(exc), traceback=traceback.format_exc())
            except Exception as exc:
                record = _partial(common, final)
                text = str(exc)
                record.update(status='out_of_memory' if 'out of memory' in text.lower() else 'error', error=text,
                              traceback=traceback.format_exc())
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()
            record['status'] = convergence_status(record, cspec)
            record.update(specification_sha256=identity, attempt=spec.get('attempt', 1), worker_wall_seconds=perf_counter()-t,
                          runner_wall_seconds=perf_counter()-t, log=str(Path(spec['result']).with_name('worker.log')),
                          in_process=True,
                          gpu_utilization=sampler.summary(window_start, sampler.now()),
                          peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          amgx_backend=spec.get('amgx_backend'))
            common.atomic_json(final, record)
            result['candidates'][key] = record['status']
            save()
            log(f"[candidate] finish {key}: {record['status']} in {record['worker_wall_seconds']:.1f} s")
            if index == last_asm and context.element_blocks is not None:
                freed = context.element_blocks.nbytes
                context.element_blocks = None
                gc.collect()
                cp.get_default_memory_pool().free_all_blocks()
                log(f'[system] element blocks released after the ASM candidates ({freed/2**30:.2f} GiB)')
        t = perf_counter()
        exporter.join()
        waited = perf_counter()-t
        phases['export_join_wait'] = waited
        log(f"[system] export finished {export.get('seconds', 0):.1f} s after assembly; waited {waited:.1f} s for it after the last candidate")
        if 'error' in export:
            result['assembly'].update(status='error', error=export['error'], traceback=export['traceback'])
        else:
            result['assembly'].update(status='passed', operator_sha256=export['operator_sha256'], cache_files=export['files'],
                                      export_seconds=export['info']['seconds'], export_file_timings=export['info']['file_timings'])
            for key in result['candidates']:
                final = output_dir/'jobs'/key/'result.json'
                try:
                    record = json.loads(final.read_text())
                except (OSError, ValueError):
                    continue
                if record.get('operator_sha256') in (None, 'pending-export'):
                    record['operator_sha256'] = export['operator_sha256']
                    common.atomic_json(final, record)
        result['status'] = 'passed'
    except Exception as exc:
        text = str(exc)
        kind = type(exc).__name__
        result.update(status=('memory_excluded' if kind == 'BudgetExceeded' else
                              'out_of_memory' if isinstance(exc, MemoryError) or 'out of memory' in text.lower() else 'error'),
                      error=text, traceback=traceback.format_exc())
    finally:
        sampler.stop()
        result['phases'] = phases
        result['gpu_utilization'] = sampler.summary(timeline=False)
        result['worker_wall_seconds'] = perf_counter()-started
        save()
    return result


def _partial(common, final):
    try:
        return common.read_json(final)
    except (OSError, ValueError):
        return {}
