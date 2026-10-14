"""Resource-aware, correctness-checked CUDA tuning with a machine-local cache.

The caller owns representative inputs and isolated trial outputs. No PDE or
solver policy lives here. CUDA imports remain lazy for host-only installations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import statistics
import tempfile
import time

from hybridge.runtime.env import getenv


@dataclass(frozen=True)
class KernelConfiguration:
    threads: int = 64
    columns: int = 1
    retained_coupling: bool = False


_RESULTS: dict[str, dict] = {}
CACHE_VERSION = 1


def device_signature(cupy) -> dict:
    """Query allocation-visible hardware; never switch the selected device."""
    runtime = cupy.cuda.runtime
    ordinal = runtime.getDevice()
    properties = runtime.getDeviceProperties(ordinal)
    keys = ('name', 'uuid', 'major', 'minor', 'multiProcessorCount',
            'sharedMemPerBlock', 'sharedMemPerBlockOptin',
            'sharedMemPerMultiprocessor', 'maxThreadsPerBlock',
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


def _read(key):
    """Read a compatible process or disk record, ignoring malformed files."""
    if key in _RESULTS:
        return _RESULTS[key], 'process'
    try:
        result = json.loads((cache_directory() / (key + '.json')).read_text())
        if isinstance(result, dict) and result.get('version') == CACHE_VERSION and result.get('key') == key:
            KernelConfiguration(**result['configuration'])
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


def select_configuration(*, cupy, identity, candidates, shared_bytes, compile_kernel,
                         trial, explicit=None, refresh=False, thorough=False):
    """Select a legal candidate after checking isolated outputs against a baseline.

    ``trial(config, kernel)`` launches on caller-owned scratch and returns its
    numerical output. Compiling and output comparisons are outside event timing.
    The first candidate is the conservative reference and must succeed.
    """
    start = time.perf_counter()
    device = device_signature(cupy)
    key = fingerprint(dict(version=CACHE_VERSION, device=device, workload=identity))
    legal = [c for c in candidates if legal_configuration(c, shared_bytes(c), device)]
    if not legal:
        raise ValueError('No Poisson kernel configuration fits the CUDA device resources')
    if explicit is not None:
        legal = [c for c in legal if c.threads == explicit]
        if not legal:
            raise ValueError(f'No legal Poisson configuration for block_size={explicit}')
        config = legal[0]
        kernel = compile_kernel(config)
        if not legal_configuration(config, shared_bytes(config), device, kernel.attributes):
            raise ValueError('Explicit Poisson launch exceeds compiled kernel resources')
        return config, kernel, dict(reused=False, source='explicit', wall_seconds=time.perf_counter()-start)
    cached, origin = (None, None) if refresh else _read(key)
    if cached:
        try:
            config = KernelConfiguration(**cached['configuration'])
            if config in legal:
                kernel = compile_kernel(config)
                if legal_configuration(config, shared_bytes(config), device, kernel.attributes):
                    _RESULTS[key] = cached
                    return config, kernel, dict(cached, reused=True, source=origin,
                                               wall_seconds=time.perf_counter()-start)
        except (ValueError, TypeError, KeyError):
            pass
    selected = legal if thorough else legal[:4]
    records, reference, winner, best = [], None, None, float('inf')
    for config in selected:
        record = dict(configuration=asdict(config), shared_bytes=shared_bytes(config))
        jit_start = time.perf_counter()
        try:
            kernel = compile_kernel(config)
        except Exception as exc:
            if reference is None:
                raise
            record['rejected'] = f'compilation: {exc}'
            records.append(record)
            continue
        record['compile_seconds'] = time.perf_counter()-jit_start
        record['attributes'] = dict(kernel.attributes)
        if not legal_configuration(config, shared_bytes(config), device, kernel.attributes):
            record['rejected'] = 'compiled resources'
            records.append(record)
            continue
        try:
            active = cupy.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
                kernel.kernel.ptr, config.threads, shared_bytes(config))
            record['modeled_resident_blocks_per_sm'] = int(active)
            record['modeled_occupancy'] = active*config.threads / device['maxThreadsPerMultiProcessor']
        except (AttributeError, TypeError, KeyError):
            pass
        output = trial(config, kernel)
        if reference is None:
            if not bool(cupy.all(cupy.isfinite(output)).item()):
                raise ArithmeticError('Conservative Poisson tuning trial returned nonfinite values')
            reference = output.copy()
        else:
            eps = float(cupy.finfo(output.dtype).eps)
            scale = cupy.maximum(cupy.max(cupy.abs(reference)), 1.)
            error = float((cupy.max(cupy.abs(output-reference))/scale).item()) if output.size else 0.
            record['relative_error'] = error
            if not error <= max(2.e-11, 256*eps):
                record['rejected'] = 'numerical parity'
                records.append(record)
                continue
        for _ in range(2):
            trial(config, kernel)
        samples = [event_seconds(cupy, lambda: trial(config, kernel))
                   for _ in range(10 if thorough else 5)]
        record.update(samples_seconds=samples, median_seconds=statistics.median(samples),
                      min_seconds=min(samples), max_seconds=max(samples))
        records.append(record)
        if record['median_seconds'] < best:
            best, winner = record['median_seconds'], (config, kernel)
    if winner is None:
        raise ValueError('No validated Poisson kernel configuration fits this device')
    config, kernel = winner
    result = dict(version=CACHE_VERSION, key=key, device=device, workload=identity,
                  configuration=asdict(config), candidates=records,
                  wall_seconds=time.perf_counter()-start, reused=False, source='measured')
    _write(key, result)
    return config, kernel, result
