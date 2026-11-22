"""One process, one article row: a real worker run, optionally instrumented without editing source.

Modes
-----
real      ``gc_article_h200.worker.run`` untouched; the arm's Hybridge tree and flags decide the code.
spans     real run with module-attribute wrappers around every step of the RHS and reconstruction
          paths, then isolated microbenchmarks of the operations on the captured live objects.
cprofile  real run with cProfile enabled only inside the timed RHS and reconstruction calls.
kernel    real run that captures the production launch arguments, then replays the RHS and
          reconstruction kernels on those arrays for each variant in ``--variants`` (event time,
          compiled resources, modeled occupancy, phase clock and parity against the first variant).

Nothing here edits source. Timing helpers come from ``hybridge.runtime`` (``event_seconds``,
``measure_synchronized``); solver objects come from the live run, so no ABI is duplicated.
"""
from __future__ import annotations

import argparse
import cProfile
import functools
import io
import json
import os
import pstats
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

NVRTC_PHASE_LABELS = {1: 'rhs', 2: 'reconstruction'}
# Index of the arguments of poisson_local that the replay replaces.
ARG_COUNT, ARG_SOURCE, ARG_SOURCE_STRIDE, ARG_FACTORS, ARG_RHS, ARG_LOCAL, ARG_VALIDATION, ARG_FAILURES = 2, 22, 23, 26, 33, 34, 35, 36
CLOCK_SLOTS = 10
CLOCK_LABELS = ('entry', 'factor load', 'pivot load/setup', 'column build (trace lifts)', 'coupling',
                'triangular solve', 'q and outputs', 'flux', 'emit')


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=1, default=str, allow_nan=True) + '\n')


def load_spec(row, steps=None):
    from guiding_center_cg_hdg.campaigns.gc_article_h200.matrix import build_matrix
    spec = next(s for s in build_matrix() if s['id'] == row)
    if steps:
        spec = dict(spec, steps=int(steps))
    return spec


class Spans:
    """Nested wall-clock spans keyed by their path; off outside the measured case."""

    def __init__(self):
        self.enabled = False
        self.stack = []
        self.data = defaultdict(list)

    def timed(self, name, function, /, *args, **kwargs):
        if not self.enabled:
            return function(*args, **kwargs)
        path = '/'.join(self.stack + [name])
        self.stack.append(name)
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            self.stack.pop()
            self.data[path].append(elapsed)

    def wrap(self, name, function):
        @functools.wraps(function)
        def inner(*args, **kwargs):
            return self.timed(name, function, *args, **kwargs)
        return inner

    def summary(self):
        rows = {}
        for path, values in sorted(self.data.items()):
            ordered = sorted(values)
            trimmed = ordered[:max(1, int(.95*len(ordered)))]
            rows[path] = dict(count=len(values), median_ms=1e3*statistics.median(values),
                              trimmed_mean_ms=1e3*statistics.fmean(trimmed),
                              p10_ms=1e3*ordered[len(ordered)//10], p90_ms=1e3*ordered[(9*len(ordered))//10],
                              sum_ms=1e3*sum(values))
        return rows


class CapturedKernel:
    """Forward a compiled kernel; remember its production launch and time the host-side calls."""

    def __init__(self, kernel, phase, store, spans):
        self._kernel, self._phase, self._store, self._spans = kernel, phase, store, spans

    def __call__(self, grid, block, args, **kwargs):
        if int(args[1]) == 0:
            self._store[self._phase] = dict(grid=grid, block=block, args=args, kwargs=kwargs)
        return self._spans.timed('kernel_call', self._kernel, grid, block, args, **kwargs)

    @property
    def attributes(self):
        return self._spans.timed('kernel_attributes', lambda: self._kernel.attributes)

    def __getattr__(self, name):
        return getattr(self._kernel, name)


def install(spans, capture):
    """Wrap module attributes on the RHS and reconstruction path; source stays untouched."""
    import cupy
    import hybridge.core.device as core_device
    import hybridge.core.field_ops as field_ops
    import hybridge.hdg.condensation_device as condensation
    import hybridge.mixed.cupy as mixed
    import hybridge.mixed.raw_cuda.identity as identity
    import hybridge.mixed.raw_cuda.poisson as poisson
    import hybridge.runtime.cuda_tuning as tuning
    import hybridge.runtime.precision as precision
    import hybridge.solvers.diffusion_raw_cuda as driver

    for module, names in (
        (cupy, ('any', 'empty', 'zeros', 'ascontiguousarray', 'isfinite')),
        (field_ops, ('coefficient_field',)),
        (core_device, ('field_from_cupy_coefficients',)),
        (condensation, ('reconstruct_trace_cupy',)),
        (mixed, ('assemble_projected_diffusion_trace_rhs_eliminated_raw_cupy', 'source_moments_cupy',
                 'boundary_trace_values_cupy', 'assemble_projected_diffusion_trace_rhs_eliminated_raw_cuda',
                 '_validate_raw_cuda_source', '_require_same_space_dg_field', '_raw_cuda_reaction_is_zero',
                 'raw_cuda_diffusion_fallback_reason', 'validate_raw_cuda_supported', 'as_cupy_space')),
        (identity, ('reconstruct_projected_diffusion_field_raw_cuda', 'validate_raw_cuda_supported')),
        (poisson, ('refresh_rhs', 'reconstruct', '_launch', '_full_trace', '_context', 'configurations',
                   'device_signature', 'select_configuration', 'event_seconds', 'validate_block_size',
                   '_validate_context')),
        (tuning, ('device_signature', 'select_variant', 'fingerprint', '_read', 'legal_configuration',
                  'tuning_mode', 'tuning_budget')),
        (driver, ('require_finite_device_values', 'audit_arrays')),
    ):
        for name in names:
            setattr(module, name, spans.wrap(name, getattr(module, name)))

    # Keep the live inputs of the last production calls, for isolated replay.
    def recording(module, name, label):
        original = getattr(module, name)

        @functools.wraps(original)
        def inner(*args, **kwargs):
            capture[label] = (args, kwargs)
            return original(*args, **kwargs)
        setattr(module, name, inner)
    recording(poisson, 'refresh_rhs', 'refresh_rhs')
    recording(poisson, 'reconstruct', 'reconstruct')
    recording(mixed, 'assemble_projected_diffusion_trace_rhs_eliminated_raw_cupy', 'rhs_wrapper')
    recording(condensation, 'reconstruct_trace_cupy', 'reconstruct_trace')

    launch = poisson._launch

    @functools.wraps(launch)
    def captured_launch(context, *args, **kwargs):
        capture.setdefault('contexts', {})[kwargs['phase']] = (context, kwargs.get('fmt'))
        return launch(context, *args, **kwargs)
    poisson._launch = captured_launch

    compile_kernel = poisson._compile
    capture['compile'] = compile_kernel

    @functools.wraps(compile_kernel)
    def captured_compile(context, phase, fmt, config):
        return CapturedKernel(compile_kernel(context, phase, fmt, config), phase,
                              capture.setdefault('launches', {}), spans)
    poisson._compile = captured_compile

    stage = driver._timed_call

    @functools.wraps(stage)
    def timed_call(label, verbosity, function, **kwargs):
        name = ('RHS' if 'reduced RHS' in label else 'RECONSTRUCT' if 'reconstructing local' in label else None)
        if name is None:
            return stage(label, verbosity, function, **kwargs)
        return spans.timed(name, stage, label, verbosity, function, **kwargs)
    driver._timed_call = timed_call


def run_worker(spec, out, on_case=None):
    """Run the campaign worker in this process; ``on_case(index)`` precedes warmup (0) and measurement (1)."""
    from guiding_center_cg_hdg.campaigns.gc_article_h200 import worker
    import guiding_center_cg_hdg.guiding_center.runtime.runner as runner
    overrides = json.loads(os.environ.get('HYBRIDGE_POISSON_ELEMENT_COST_OVERRIDES') or '{}')
    if overrides:
        # An existing solver option changed for one arm; the worker's measurement stays as it is.
        from dataclasses import replace
        configuration = worker.configuration

        def configured(run_spec, run_out):
            original, config = configuration(run_spec, run_out)
            return original, replace(config, **overrides)
        worker.configuration = configured
    if on_case is not None:
        original = runner.run_guiding_center_case
        count = {'n': 0}

        @functools.wraps(original)
        def case(*args, **kwargs):
            on_case(count['n'])
            count['n'] += 1
            return original(*args, **kwargs)
        runner.run_guiding_center_case = case
    worker.run(spec, out)


def series(function, *, repeats=200, synchronize=None):
    """Median wall time of isolated calls, with the device idle before each."""
    samples = []
    for _ in range(repeats):
        if synchronize is not None:
            synchronize()
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter()-start)
    samples.sort()
    return dict(median_ms=1e3*statistics.median(samples), p10_ms=1e3*samples[len(samples)//10],
                p90_ms=1e3*samples[(9*len(samples))//10], repeats=repeats)


def microbenchmarks(capture, *, repeats):
    """Host operations of ``_launch`` and its callers, on the run's live arrays."""
    import cupy as cp
    import hybridge.mixed.raw_cuda.poisson as poisson
    import hybridge.runtime.cuda_tuning as tuning
    import numpy as np
    from hybridge.hdg.condensation_device import require_finite_device_values
    from hybridge.runtime.precision import REAL_DTYPE
    runtime = cp.cuda.runtime
    ordinal = runtime.getDevice()
    stream = cp.cuda.get_current_stream()
    sync = stream.synchronize
    results = {}

    def add(name, function, n=repeats):
        results[name] = series(function, repeats=n, synchronize=sync)
    results['stream.synchronize (idle)'] = series(sync, repeats=repeats)
    add('runtime.getDevice', runtime.getDevice)
    add('runtime.getDeviceProperties', lambda: runtime.getDeviceProperties(ordinal), 100)
    add('runtime.driverGetVersion', runtime.driverGetVersion)
    add('runtime.runtimeGetVersion', runtime.runtimeGetVersion)
    add('runtime.deviceGetAttribute(87)', lambda: runtime.deviceGetAttribute(87, ordinal))
    add('device_signature', lambda: tuning.device_signature(cp), 100)
    device = tuning.device_signature(cp)
    add('fingerprint(device signature)', lambda: tuning.fingerprint(device))
    add('cp.empty(1)', lambda: cp.empty(1, dtype=REAL_DTYPE))
    add('event_seconds(no work)', lambda: tuning.event_seconds(cp, lambda: None))

    rhs_args, rhs_kwargs = capture['refresh_rhs']
    context, fmt = capture['contexts'][1]
    launch = capture['launches'][1]
    ne = int(context.cspace.mesh.num_tri)
    add('cp.empty(ne,int32)', lambda: cp.empty(ne, dtype=cp.int32))
    full_trace = poisson._full_trace(rhs_kwargs['boundary_trace'], context.cspace)
    add('_full_trace', lambda: poisson._full_trace(rhs_kwargs['boundary_trace'], context.cspace))
    add('cp.any(trace!=0).item()', lambda: bool(cp.any(full_trace != 0).item()))
    failures = launch['args'][ARG_FAILURES]
    add('cp.any(failures).item()', lambda: bool(cp.any(failures).item()))
    add('configurations(phase=1)', lambda: poisson.configurations(context, 1, device, None))
    kernel = capture['compile'](context, 1, fmt, poisson.KernelConfiguration(int(launch['block'][0]), 1, False))
    add('kernel.attributes', lambda: kernel.attributes)
    empty = list(launch['args'])
    empty[ARG_COUNT] = np.int64(0)

    def zero_launch():
        kernel((1,), launch['block'], tuple(empty), **launch['kwargs'])
    add('raw kernel call, one no-op block', zero_launch)
    add('rhs cp.zeros', lambda: cp.zeros(int(context.cspace.mesh.int_edges_inds.size*context.cspace.edg_dof), dtype=REAL_DTYPE))

    trace_args, _ = capture['reconstruct_trace']
    add('reconstruct_trace_cupy', lambda: __import__('hybridge.hdg.condensation_device', fromlist=['x'])
        .reconstruct_trace_cupy(*trace_args))
    local = capture['launches'][2]['args'][ARG_LOCAL]
    nel = int(context.cspace.el_dof)
    add('require_finite_device_values(local)', lambda: require_finite_device_values(local, 'probe'))
    add('copy local[:,:nel]', lambda: cp.ascontiguousarray(local[:, :nel]))
    add('copy local[:,nel:2nel]', lambda: cp.ascontiguousarray(local[:, nel:2*nel]))
    # The whole calls, without the stage wrappers above them.
    add('poisson.refresh_rhs (whole)', lambda: poisson.refresh_rhs(*rhs_args, **rhs_kwargs), 100)
    rec_args, rec_kwargs = capture['reconstruct']
    add('poisson.reconstruct (whole)', lambda: poisson.reconstruct(*rec_args, **rec_kwargs), 100)
    wrapper_args, wrapper_kwargs = capture['rhs_wrapper']
    import hybridge.mixed.cupy as mixed
    add('RHS wrapper (whole)', lambda: mixed.assemble_projected_diffusion_trace_rhs_eliminated_raw_cupy(
        *wrapper_args, **wrapper_kwargs), 100)
    return results


def profile_stages(profiles):
    """Enable cProfile only inside the two timed solver stages."""
    import hybridge.solvers.diffusion_raw_cuda as driver
    stage = driver._timed_call
    active = {'on': False}

    @functools.wraps(stage)
    def timed_call(label, verbosity, function, **kwargs):
        name = ('RHS' if 'reduced RHS' in label else 'RECONSTRUCT' if 'reconstructing local' in label else None)
        if name is None or not active['on']:
            return stage(label, verbosity, function, **kwargs)
        profile = profiles.setdefault(name, cProfile.Profile())
        profile.enable()
        try:
            return stage(label, verbosity, function, **kwargs)
        finally:
            profile.disable()
    driver._timed_call = timed_call
    return active


def write_profile(profile, path, limit=70):
    for key in ('tottime', 'cumulative'):
        buffer = io.StringIO()
        pstats.Stats(profile, stream=buffer).strip_dirs().sort_stats(key).print_stats(limit)
        Path(f'{path}.{key}.txt').write_text(buffer.getvalue())
    profile.dump_stats(f'{path}.pstats')


def sm_clock_mhz():
    try:
        text = subprocess.run(['nvidia-smi', '--query-gpu=clocks.sm,clocks.mem,clocks.max.sm', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, check=True).stdout
        return [float(x) for x in text.splitlines()[0].split(',')]
    except Exception:
        return None


def relative_max(cp, value, reference):
    scale = float(cp.max(cp.abs(reference)).item())
    return float(cp.max(cp.abs(value-reference)).item())/max(scale, 1e-300)


def kernel_probe(capture, variants, out, repeats=15):
    """Replay production launches for each variant on the captured arrays."""
    import cupy as cp
    import hybridge.mixed.raw_cuda.poisson as poisson
    from hybridge.runtime.cuda_tuning import KernelConfiguration, event_seconds
    from hybridge.runtime.precision import REAL_DTYPE
    rows, references, inverses = [], {}, {}
    compile_kernel = capture['compile']
    device_props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    for variant in variants:
        poisson.KERNEL_FLAGS = dict(variant.get('flags', {}))
        poisson.COMPILE_OPTIONS = tuple(variant.get('options', ()))
        for phase in variant.get('phases', (1, 2)):
            launch = capture['launches'][phase]
            context, fmt = capture['contexts'][phase]
            ne = int(context.cspace.mesh.num_tri)
            tuned = int(launch['block'][0])
            for threads in variant.get('threads', ['tuned']):
                threads = tuned if threads == 'tuned' else int(threads)
                config = KernelConfiguration(threads, 1, bool(variant.get('retain', False)))
                row = dict(variant=variant['name'], phase=NVRTC_PHASE_LABELS[phase], threads=threads, tuned_threads=tuned,
                           flags=variant.get('flags', {}), options=variant.get('options', []), pad_to=variant.get('pad_to'))
                try:
                    start = time.perf_counter()
                    kernel = compile_kernel(context, phase, fmt, config)
                    row['compile_seconds'] = time.perf_counter()-start
                    shared = poisson._workspace_bytes(context, phase, config)
                    if variant.get('pad_to'):
                        shared = max(shared, int(variant['pad_to']))
                        kernel.max_dynamic_shared_size_bytes = int(shared)
                    attributes = dict(kernel.attributes)
                    row.update(shared_bytes=shared, attributes=attributes)
                    function = getattr(kernel, 'kernel', kernel)
                    blocks = cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(function.ptr, threads, shared)
                    row.update(blocks_per_sm=int(blocks), occupancy=blocks*threads/device_props['maxThreadsPerMultiProcessor'])
                except Exception as error:
                    row.update(status='unavailable', error=str(error)[:400])
                    rows.append(row)
                    dump(Path(out)/'kernel_probe.json', rows)
                    continue
                args = list(launch['args'])
                if variant.get('flags', {}).get('INVERSE'):
                    if context.policy not in inverses:
                        start = time.perf_counter()
                        inverses[context.policy] = poisson.local_inverses(context)
                        row['inverse_build_seconds'] = time.perf_counter()-start
                    args[ARG_FACTORS] = inverses[context.policy][0]
                    row['inverse_residual'] = inverses[context.policy][1]
                if variant.get('use_moments'):
                    args[ARG_SOURCE], args[ARG_SOURCE_STRIDE] = capture['launches'][1]['args'][ARG_SOURCE:ARG_SOURCE+2]
                rhs = cp.empty_like(args[ARG_RHS])
                local = cp.empty_like(args[ARG_LOCAL]) if phase == 2 else args[ARG_LOCAL]
                validation = cp.zeros((ne, CLOCK_SLOTS), dtype=REAL_DTYPE)
                failures = cp.empty(ne, dtype=cp.int32)
                args[ARG_RHS], args[ARG_LOCAL] = rhs, local
                args[ARG_VALIDATION], args[ARG_FAILURES] = validation, failures
                args = tuple(args)
                samples = []
                for index in range(repeats):
                    if phase == 1:
                        rhs.fill(0)
                    cp.cuda.get_current_stream().synchronize()
                    samples.append(event_seconds(cp, lambda: kernel((ne,), (threads,), args, shared_mem=shared)))
                retained = samples[3:]
                row.update(status='measured', device_ms_median=1e3*statistics.median(retained),
                           device_ms_min=1e3*min(retained), device_ms_max=1e3*max(retained))
                if bool(cp.any(failures).item()):
                    row['failures'] = int(cp.count_nonzero(failures).item())
                flags = variant.get('flags', {})
                if flags.get('PHASE_CLOCK'):
                    validation.fill(0)
                    if phase == 1:
                        rhs.fill(0)
                    kernel((ne,), (threads,), args, shared_mem=shared)
                    cp.cuda.get_current_stream().synchronize()
                    mean = cp.asnumpy(cp.mean(validation, axis=0)).tolist()
                    row['clock_cycles_per_block'] = dict(zip(CLOCK_LABELS, mean))
                    row['clock_cycles_per_block_total'] = sum(mean)
                    row['sm_clock_mhz'] = sm_clock_mhz()
                if variant.get('save') and phase == 2:
                    nel = int(context.cspace.el_dof)
                    numpy_module = __import__('numpy')
                    numpy_module.savez_compressed(Path(out)/f"local_{variant['name']}.npz", local=cp.asnumpy(local), nel=nel)
                if not flags.get('STOP'):
                    output = rhs if phase == 1 else local
                    if not flags.get('PHASE_CLOCK'):
                        key = (phase,)
                        if key not in references:
                            references[key] = output.copy()
                        row['parity_relative_max'] = relative_max(cp, output, references[key])
                    production = launch['args'][ARG_RHS if phase == 1 else ARG_LOCAL]
                    row['against_last_production_output'] = relative_max(cp, output, production)
                rows.append(row)
                dump(Path(out)/'kernel_probe.json', rows)
                print(f"[probe] {variant['name']:<28} {row['phase']:<15} t={threads:<4} regs={attributes.get('num_regs')} "
                      f"shared={shared} blk/SM={row.get('blocks_per_sm')} device={row['device_ms_median']:.3f} ms "
                      f"parity={row.get('parity_relative_max')}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--row', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('real', 'spans', 'cprofile', 'kernel'), default='real')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--variants', type=Path, help='JSON list of kernel variants (mode kernel)')
    parser.add_argument('--micro-repeats', type=int, default=200)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(args.row, args.steps)
    dump(args.out/'spec.json', spec)
    started = time.time()
    if args.mode == 'real':
        run_worker(spec, args.out)
    elif args.mode == 'spans':
        spans, capture = Spans(), {}
        install(spans, capture)

        def on_case(index):
            spans.enabled = index == 1
        run_worker(spec, args.out, on_case)
        spans.enabled = False
        dump(args.out/'spans.json', spans.summary())
        dump(args.out/'microbenchmarks.json', microbenchmarks(capture, repeats=args.micro_repeats))
    elif args.mode == 'cprofile':
        profiles = {}
        active = profile_stages(profiles)

        def on_case(index):
            active['on'] = index == 1
        run_worker(spec, args.out, on_case)
        for name, profile in profiles.items():
            write_profile(profile, args.out/f'profile_{name}')
    else:
        spans, capture = Spans(), {}
        install(spans, capture)
        run_worker(spec, args.out)
        kernel_probe(capture, json.loads(args.variants.read_text()), args.out)
    import hybridge
    dump(args.out/'harness.json', dict(mode=args.mode, row=args.row, seconds=time.time()-started, hybridge_file=hybridge.__file__,
                                       host_flags=os.environ.get('HYBRIDGE_POISSON_HOST_FLAGS'),
                                       kernel_flags=os.environ.get('HYBRIDGE_POISSON_KERNEL_FLAGS'),
                                       options=os.environ.get('HYBRIDGE_POISSON_KERNEL_OPTIONS')))


if __name__ == '__main__':
    main()
