"""Resource-aware, correctness-checked CUDA tuning with a machine-local cache.

The caller owns representative inputs and isolated trial outputs. No PDE or
solver policy lives here. CUDA imports remain lazy for host-only installations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
import time

from hybridge.runtime.env import getenv
from hybridge.runtime.kernel_log import report_selection


@dataclass(frozen=True)
class KernelConfiguration:
    threads: int = 64
    columns: int = 1
    retained_coupling: bool = False


_RESULTS: dict[str, dict] = {}
_SIGNATURES: dict[int, dict] = {}
_RESOLVED: dict = {}
CACHE_VERSION = 2


@dataclass(frozen=True)
class TuningBudget:
    """First-use limits; a running compilation or CUDA launch is not preempted."""
    seconds: float = 30.
    candidates: int = 8
    repeats: int = 5
    warmups: int = 2

    def __post_init__(self):
        """Reject invalid limits before any compilation or allocation."""
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError('CUDA tuning seconds must be finite and positive')
        if self.candidates < 1 or self.repeats < 1 or self.warmups < 0:
            raise ValueError('CUDA tuning counts must be positive (warmups may be zero)')


def tuning_budget() -> TuningBudget:
    """Read the user-selected first-use calibration bounds."""
    return TuningBudget(seconds=float(getenv('CUDA_TUNING_SECONDS', '30')),
                        candidates=int(getenv('CUDA_TUNING_CANDIDATES', '8')))


def tuning_mode() -> str:
    """Auto calibrates misses, off selects the prior, refresh ignores records."""
    mode = getenv('CUDA_TUNING', 'auto').strip().lower()
    if mode not in ('auto', 'off', 'refresh'):
        raise ValueError('HYBRIDGE_CUDA_TUNING must be auto, off, or refresh')
    return mode


def device_signature(cupy) -> dict:
    """Query allocation-visible hardware once per device and process; never switch the selected device.

    The query costs about 0.7 ms on an H200 and ran on every tuned lookup, where it exceeded the
    kernels being selected on small meshes (campaigns/poisson_element_cost, 2026-11-02). Device
    properties, driver and runtime do not change within a process. Callers receive a copy.
    """
    runtime = cupy.cuda.runtime
    ordinal = runtime.getDevice()
    cached = _SIGNATURES.get(ordinal)
    if cached is not None:
        return dict(cached)
    properties = runtime.getDeviceProperties(ordinal)
    keys = ('name', 'uuid', 'major', 'minor', 'multiProcessorCount',
            'sharedMemPerBlock', 'sharedMemPerBlockOptin',
            'sharedMemPerMultiprocessor', 'totalGlobalMem', 'l2CacheSize', 'memoryBusWidth',
            'clockRate', 'memoryClockRate', 'maxThreadsPerBlock',
            'maxThreadsPerMultiProcessor', 'regsPerBlock', 'regsPerMultiprocessor')
    def plain(value):
        """Convert CUDA property values to JSON-safe scalar data."""
        if isinstance(value, bytes):
            return value.hex()
        if hasattr(value, 'tolist'):
            return value.tolist()
        return value
    result = {key: plain(properties.get(key)) for key in keys}
    if isinstance(properties.get('name'), bytes):
        result['name'] = properties['name'].decode(errors='replace')
    result.update(driver=runtime.driverGetVersion(), runtime=runtime.runtimeGetVersion(),
                  cupy=cupy.__version__)
    try:
        result['fp32_fp64_ratio'] = runtime.deviceGetAttribute(87, ordinal)
    except Exception:
        result['fp32_fp64_ratio'] = None
    _SIGNATURES[ordinal] = dict(result)
    return result


def legal_configuration(config, shared_bytes, device, attributes=None) -> bool:
    """Screen runtime and compiler allocation limits for one launch."""
    limit = device.get('sharedMemPerBlockOptin') or device.get('sharedMemPerBlock') or 49152
    if shared_bytes > limit or config.threads > (device.get('maxThreadsPerBlock') or 1024):
        return False
    if attributes:
        if config.threads > attributes.get('max_threads_per_block', 1024):
            return False
        registers = attributes.get('num_regs', 0) * config.threads
        if registers > (device.get('regsPerBlock') or 65536):
            return False
        if shared_bytes + attributes.get('shared_size_bytes', 0) > limit:
            return False
    return True


def kernel_attributes(cupy, kernel) -> dict:
    """Read resources from CuPy RawKernel or low-level Function handles."""
    attributes = getattr(kernel, 'attributes', None)
    if attributes is not None:
        return dict(attributes)
    function = getattr(kernel, 'kernel', kernel)
    return {name: cupy.cuda.driver.funcGetAttribute(code, function.ptr) for name, code in (
        ('max_threads_per_block', 0), ('shared_size_bytes', 1),
        ('local_size_bytes', 3), ('num_regs', 4))}


def cache_directory() -> Path:
    """Resolve the machine-local cache without creating directories."""
    override = getenv('CUDA_TUNING_CACHE_DIR')
    return Path(override).expanduser() if override else Path(
        os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))
    ) / 'hybridge' / 'cuda-tuning'


def fingerprint(value) -> str:
    """Hash versioned workload/device metadata deterministically."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def clear_tuning_cache() -> None:
    """Clear only in-process records (disk refresh is explicit per workload)."""
    _RESULTS.clear()
    _SIGNATURES.clear()
    _RESOLVED.clear()


def resolved_selection(key, select, keep=None):
    """Run ``select()`` once per hashable structural ``key`` and process; return ``(value, reused)``.

    For launches repeated with unchanged kernels, where rebuilding a selection's identity costs
    more than the decision is worth. ``key`` holds what the kernels and the workspace depend on;
    the current mode and budget are added. ``refresh`` keeps its contract and measures every call.
    A selection must not retain per-call arrays. ``clear_tuning_cache`` forgets the entries.
    When ``keep(value)`` is false the value is returned but not remembered, so the next call
    selects again (for example after a fallback chosen without a measurement).
    """
    mode = tuning_mode()
    if mode == 'refresh':
        return select(), False
    key = (mode, tuning_budget(), key)
    if key in _RESOLVED:
        return _RESOLVED[key], True
    value = select()
    if keep is None or keep(value):
        _RESOLVED[key] = value
    return value, False


def launch_tuned(kernel, args, *, outputs, grid, identity, source, baseline=256, zero_outputs=False):
    """Launch ``kernel`` once for the caller with the block size measured for this workload.

    ``grid(threads)`` owns the kernel's indexing contract. Input arrays are read-only and
    ``outputs`` lists every written argument; atomic accumulations request ``zero_outputs``.
    Calibration writes to private arrays, never to the caller's, including on the first call.
    Reading device properties and rebuilding an identity cost more than these kernels on small
    meshes, so each launch resolves once per process through :func:`resolved_selection`.
    """
    from hybridge.runtime.optional import require_cupy

    cupy = require_cupy()
    outputs = tuple(outputs)
    shapes = [(args[index].shape, args[index].dtype.str) for index in outputs]

    def select():
        """Choose the block size with ``select_variant``, which times the candidates on copies of the output arrays or reuses an earlier record; return ``(threads, report)``."""
        scratch = {}
        device = device_signature(cupy)

        def prepare(name):
            """Return the block size ``int(name)``; raise ``CandidateUnavailable`` if the compiled kernel cannot launch with it on this device."""
            threads = int(name)
            if not legal_configuration(KernelConfiguration(threads), 0, device, kernel_attributes(cupy, kernel)):
                raise CandidateUnavailable('compiled launch resources')
            return threads

        def before(name, threads):
            """On first use, copy ``args`` with new empty arrays in place of the outputs; zero those arrays when ``zero_outputs``."""
            if not scratch:
                trial = list(args)
                for index in outputs:
                    trial[index] = cupy.empty_like(args[index])
                scratch['args'] = tuple(trial)
            if zero_outputs:
                for index in outputs:
                    scratch['args'][index].fill(0)

        def run(name, threads):
            """Launch the kernel with ``threads`` per block on the copied arguments; return the copied output arrays."""
            kernel(grid(threads), (threads,), scratch['args'])
            return tuple(scratch['args'][index] for index in outputs)

        _, threads, report = select_variant(
            cupy=cupy, identity=dict(**identity, launch_source=fingerprint(source), outputs=shapes),
            candidates=[str(block) for block in dict.fromkeys((baseline, 128, 256, 64, 512, 32))],
            prepare=prepare, before=before, run=run, choice_label=lambda name: f"{name} threads")
        return threads, report

    (threads, report), reused = resolved_selection(
        (source, repr(sorted(identity.items())), repr(shapes)), select)
    kernel(grid(threads), (threads,), args)
    return dict(report, reused=True, source='process-launch') if reused else report


def _read(key):
    """Read a compatible process or disk record, ignoring malformed files."""
    if key in _RESULTS:
        return _RESULTS[key], 'process'
    try:
        result = json.loads((cache_directory() / (key + '.json')).read_text())
        if isinstance(result, dict) and result.get('version') == CACHE_VERSION and result.get('key') == key:
            if not isinstance(result.get('variant'), str):
                return None, None
            return result, 'disk'
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None, None


def _write(key, result):
    """Persist an atomic record with a process-only fallback on I/O errors."""
    _RESULTS[key] = result
    temporary = None
    try:
        directory = cache_directory()
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', dir=directory, prefix='.tune-', delete=False) as stream:
            temporary = stream.name
            json.dump(result, stream, indent=2, sort_keys=True)
        os.replace(temporary, directory / (key + '.json'))
    except OSError:
        pass  # read-only caches must not make a solve fail
    finally:
        if temporary:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass


def event_seconds(cupy, function):
    """Time and synchronize a callable's stream interval with CUDA events."""
    begin, end = cupy.cuda.Event(), cupy.cuda.Event()
    begin.record()
    function()
    end.record()
    end.synchronize()
    return float(cupy.cuda.get_elapsed_time(begin, end)) / 1000.


class CandidateUnavailable(ValueError):
    """A candidate cannot run under current resource or optional-library limits."""


class UnrepresentativeInput(ArithmeticError):
    """Raised by ``validate`` when the trial input itself fails, whatever the candidate.

    Example: a local factorization fails because the caller's system is singular. On the
    baseline trial, ``select_variant`` then returns the baseline without measuring the other
    candidates and without writing a cache record; the caller's own checks report the failure.
    On any other candidate it counts as a failed validation of that candidate.
    """


def _arrays(output):
    """Represent multiple independent outputs without concatenating GPU buffers."""
    return tuple(output) if isinstance(output, (tuple, list)) else (output,)


def _compare(cupy, output, reference, rtol):
    """Check every output, including finite values and its own magnitude scale."""
    actual = _arrays(output)
    if len(actual) != len(reference):
        return float('inf')
    worst = 0.
    for value, expected in zip(actual, reference):
        if value.shape != expected.shape or value.dtype != expected.dtype:
            return float('inf')
        if not bool(cupy.all(cupy.isfinite(value)).item()):
            return float('inf')
        if not value.size:
            continue
        scale = max(float(cupy.max(cupy.abs(expected)).item()), 1.)
        error = float(cupy.max(cupy.abs(value - expected)).item()) / scale
        eps = float(cupy.finfo(value.dtype).eps) if value.dtype.kind in 'fc' else 0.
        worst = max(worst, error / max(rtol, 256 * eps))
    return worst


def select_variant(*, cupy, identity, candidates, prepare, run, before=None,
                   validate=None, baseline=None, explicit=None, refresh=False,
                   budget=None, rtol=2.e-11, minimum_gain=.03, describe=None,
                   trial_prepare=None, calibration_deadline=None, choice_label=str):
    """Choose equivalent implementations on bounded, caller-owned trial inputs.

    Candidates are stable string names ordered by measured architecture priors.
    ``prepare(name)`` compiles and resource-checks a candidate, returning a handle.
    ``trial_prepare(name, handle)`` optionally allocates isolated samples and
    warms library plans on cache misses only, within the calibration budget.
    ``calibration_deadline`` can share an absolute monotonic deadline across
    several stages; it is deliberately excluded from the persistent identity.
    ``before(name, handle)`` restores destructive inputs outside event timing;
    ``run(name, handle)`` launches and returns an array or tuple of arrays.
    ``validate(output)`` may additionally check an independent residual/status, and raises
    UnrepresentativeInput when the input itself fails; the baseline is then returned unmeasured
    with ``source='unrepresentative-input'`` and nothing is cached.
    The baseline is checked first, even when its compilation consumes the budget.
    ``choice_label(name)`` is what a high-verbosity log calls a candidate.
    No trial output may alias a production solution or retained factor cache.
    Compile/resource rejection uses CandidateUnavailable; fatal launch errors
    propagate. No reduced precision or altered solver/discretization is implied.
    """
    start = time.perf_counter()
    names = list(dict.fromkeys(candidates))
    if not names or any(not isinstance(name, str) for name in names):
        raise ValueError('CUDA tuning needs named candidates')
    baseline = names[0] if baseline is None else baseline
    if baseline not in names:
        raise ValueError('CUDA tuning baseline must be a candidate')
    mode = tuning_mode()
    if explicit is not None or mode == 'off':
        name = names[0] if explicit is None else explicit
        if name not in names:
            raise ValueError('explicit CUDA candidate is not supported')
        handle = prepare(name)
        source = 'explicit' if explicit is not None else 'disabled'
        report_selection(identity, choice_label(name), source, time.perf_counter()-start)
        return name, handle, dict(variant=name, reused=False, source=source,
            wall_seconds=time.perf_counter()-start)
    budget = budget or tuning_budget()
    device = device_signature(cupy)
    key = fingerprint(dict(version=CACHE_VERSION, device=device, workload=identity,
                           candidates=names, baseline=baseline, budget=asdict(budget),
                           rtol=rtol, minimum_gain=minimum_gain))
    cached, origin = (None, None) if refresh or mode == 'refresh' else _read(key)
    if cached and cached['variant'] in names:
        try:
            handle = prepare(cached['variant'])
        except CandidateUnavailable:
            pass
        else:
            _RESULTS[key] = cached
            report_selection(identity, choice_label(cached['variant']), origin, time.perf_counter()-start)
            return cached['variant'], handle, dict(cached, reused=True, source=origin,
                                                    wall_seconds=time.perf_counter()-start)
    deadline = start + budget.seconds
    if calibration_deadline is not None:
        deadline = min(deadline, calibration_deadline)
    ordered = [baseline] + [name for name in names if name != baseline]
    records, reference, winner, best = [], None, None, float('inf')
    stop_reason = 'all candidates'
    for name in ordered:
        if records and (len(records) >= budget.candidates or time.perf_counter() >= deadline):
            stop_reason = 'candidate limit' if len(records) >= budget.candidates else 'time limit'
            break
        record = dict(variant=name)
        compile_start = time.perf_counter()
        try:
            handle = prepare(name)
            if trial_prepare is not None:
                trial_prepare(name, handle)
        except CandidateUnavailable as exc:
            if name == baseline:
                raise
            record['rejected'] = str(exc)
            records.append(record)
            continue
        record['compile_seconds'] = time.perf_counter()-compile_start
        if describe is not None:
            record.update(describe(name, handle))
        # Always finish baseline validation; do not begin another candidate's
        # trial when its compilation has already consumed the remaining budget.
        if reference is not None and time.perf_counter() >= deadline:
            record['rejected'] = 'time limit after compilation'
            records.append(record)
            stop_reason = 'time limit'
            break
        if before is not None:
            before(name, handle)
        output = run(name, handle)
        try:
            valid = validate is None or validate(output)
        except UnrepresentativeInput as exc:
            if name != baseline:
                valid = False
            else:
                # Nothing measured on this input is worth keeping: no timing, no cache record.
                record['rejected'] = f'unrepresentative input: {exc}'
                records.append(record)
                seconds = time.perf_counter()-start
                report_selection(identity, choice_label(name), 'unrepresentative-input', seconds,
                                 candidates=len(records))
                return name, handle, dict(variant=name, candidates=records, reused=False,
                                          source='unrepresentative-input',
                                          stop_reason='unrepresentative input', wall_seconds=seconds)
        if not valid:
            if name == baseline:
                raise ArithmeticError('Conservative CUDA tuning trial failed validation')
            record['rejected'] = 'independent validation'
            records.append(record)
            continue
        if reference is None:
            if any(not bool(cupy.all(cupy.isfinite(a)).item()) for a in _arrays(output)):
                raise ArithmeticError('Conservative CUDA tuning trial returned nonfinite values')
            reference = tuple(a.copy() for a in _arrays(output))
        else:
            error = _compare(cupy, output, reference, rtol)
            record['parity_error_over_tolerance'] = error if math.isfinite(error) else None
            if error > 1.:
                record['rejected'] = 'numerical parity'
                records.append(record)
                continue
        samples = []
        for repetition in range(budget.warmups + budget.repeats):
            if time.perf_counter() >= deadline:
                stop_reason = 'time limit'
                break
            if before is not None:
                before(name, handle)
            elapsed = event_seconds(cupy, lambda: run(name, handle))
            if repetition >= budget.warmups:
                samples.append(elapsed)
        median = statistics.median(samples) if samples else None
        record.update(samples_seconds=samples, median_seconds=median)
        records.append(record)
        # Keep a validated fallback even when the mandatory baseline JIT took
        # the whole budget. Require a margin to displace an earlier candidate.
        if winner is None or median is not None and median < best * (1. - minimum_gain):
            winner = name, handle
            best = median if median is not None else float('inf')
    if winner is None:
        raise ValueError('No validated CUDA candidate fits this device')
    name, handle = winner
    result = dict(version=CACHE_VERSION, key=key, device=device, workload=identity,
                  variant=name, candidates=records, budget=asdict(budget),
                  stop_reason=stop_reason, wall_seconds=time.perf_counter()-start,
                  reused=False, source='measured')
    _write(key, result)
    report_selection(identity, choice_label(name), 'measured', result['wall_seconds'], candidates=len(records))
    return name, handle, result


def select_configuration(*, cupy, identity, candidates, shared_bytes, compile_kernel,
                         trial, explicit=None, refresh=False, thorough=False, baseline=None,
                         validate=None):
    """Raw-kernel adapter to the shared architecture/workload tuning policy."""
    device = device_signature(cupy)
    legal = [c for c in candidates if legal_configuration(c, shared_bytes(c), device)]
    if not legal:
        raise ValueError('No CUDA kernel configuration fits the device resources')
    if explicit is not None:
        legal = [c for c in legal if c.threads == explicit]
        if not legal:
            raise ValueError(f'No legal CUDA configuration for block_size={explicit}')
    configs = {fingerprint(asdict(c)): c for c in legal}
    def prepare(name):
        """Compile and screen the selected raw-kernel configuration."""
        config = configs[name]
        try:
            kernel = compile_kernel(config)
        except Exception as exc:
            # Compile failure is recoverable; execution failures are not caught.
            raise CandidateUnavailable(f'compilation: {exc}') from exc
        if not legal_configuration(config, shared_bytes(config), device, kernel.attributes):
            raise CandidateUnavailable('compiled resources')
        return kernel
    def describe(name, kernel):
        """Preserve resource/occupancy diagnostics in persistent records."""
        config = configs[name]
        result = dict(configuration=asdict(config), shared_bytes=shared_bytes(config),
                      attributes=kernel_attributes(cupy, kernel))
        try:
            function = getattr(kernel, 'kernel', kernel)
            active = cupy.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
                function.ptr, config.threads, shared_bytes(config))
            result['modeled_resident_blocks_per_sm'] = int(active)
            result['modeled_occupancy'] = active*config.threads / device['maxThreadsPerMultiProcessor']
        except Exception:
            pass  # optional diagnostic; legality is checked independently
        return result
    budget = tuning_budget()
    if thorough:
        budget = TuningBudget(budget.seconds, len(legal), 10, budget.warmups)
    name, kernel, report = select_variant(cupy=cupy, identity=identity, candidates=list(configs),
        prepare=prepare, run=lambda name, kernel: trial(configs[name], kernel),
        baseline=fingerprint(asdict(baseline)) if baseline in legal else None,
        validate=validate, describe=describe,
        explicit=next(iter(configs)) if explicit is not None else None,
        refresh=refresh, budget=budget,
        choice_label=lambda name: f"{configs[name].threads} threads, {configs[name].columns} columns")
    report = dict(report, configuration=asdict(configs[name]))
    report['candidates'] = [dict(record, configuration=asdict(configs[record['variant']]))
                            for record in report.get('candidates', [])]
    return configs[name], kernel, report
