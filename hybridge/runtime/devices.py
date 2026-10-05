"""Lazy CUDA inventory and non-tensor FP64 peak-throughput selection.

Only devices exposed by the CUDA runtime are considered. In particular, never
override a scheduler's CUDA_VISIBLE_DEVICES mask using physical nvidia-smi IDs.
Importing this module does not import CuPy or initialize CUDA. Discovery queries
properties and free memory but neither compiles nor launches kernels.
"""
from __future__ import annotations

import math
from pathlib import Path


def host_available_bytes():
    """Linux live available RAM, restricted by the current cgroup allocation.

    Extend the ADR worker's MemAvailable/cgroup check without importing its
    numerical assembly modules. Also inspect nested cgroups used by schedulers.
    """
    values = []
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                values.append(int(line.split()[1])*1024)
        root = Path('/sys/fs/cgroup')
        candidates = [root]
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            hierarchy, controllers, suffix = line.split(':', 2)
            if not controllers:
                current = root/suffix.lstrip('/')
                candidates.extend([current, *[p for p in current.parents if p.is_relative_to(root)]])
            elif 'memory' in controllers.split(','):
                current = root/'memory'/suffix.lstrip('/')
                candidates.extend([current, root/'memory'])
        for directory in candidates:
            for limit_name, used_name in (('memory.max', 'memory.current'),
                                          ('memory.limit_in_bytes', 'memory.usage_in_bytes')):
                limit, used = directory/limit_name, directory/used_name
                if limit.exists() and used.exists() and limit.read_text().strip() != 'max':
                    values.append(max(0, int(limit.read_text())-int(used.read_text())))
    except (OSError, ValueError):
        pass
    return min(values) if values else None

# FP32 lanes per SM, following NVIDIA cuda-samples/Common/helper_cuda.h.
# Unknown architectures intentionally require explicit throughput information.
_FP32_LANES = {
    (3, 0): 192, (3, 2): 192, (3, 5): 192, (3, 7): 192,
    (5, 0): 128, (5, 2): 128, (5, 3): 128,
    (6, 0): 64, (6, 1): 128, (6, 2): 128,
    (7, 0): 64, (7, 2): 64, (7, 5): 64,
    (8, 0): 64, (8, 6): 128, (8, 7): 128, (8, 9): 128,
    (9, 0): 128, (10, 0): 128, (10, 1): 128, (10, 3): 128,
    (11, 0): 128, (12, 0): 128, (12, 1): 128,
}


def fp64_peak_flops(major, minor, multiprocessors, clock_khz, fp32_fp64_ratio):
    """Estimated scalar/vector FP64 FMA peak, not tensor-core performance.

    Count FMA as two FLOPs. Return None rather than guess for unknown hardware
    or absent/invalid CUDA attributes. This is not an application speed forecast.
    """
    lanes = _FP32_LANES.get((major, minor))
    values = (multiprocessors, clock_khz, fp32_fp64_ratio)
    if lanes is None or any(v is None or not math.isfinite(v) or v <= 0 for v in values):
        return None
    return 2.0 * lanes * multiprocessors * clock_khz * 1000 / fp32_fp64_ratio


def discover_cuda_devices(runtime=None):
    """Return JSON-compatible records with allocation-local CUDA ordinals.

    ``runtime`` may be a CuPy-compatible fake for CPU-only tests. CUDA 13 removed
    clockRate and singleToDoublePrecisionPerfRatio from device properties, so
    use cudaDeviceGetAttribute (13 and 87) rather than those structure fields.
    Context creation for memGetInfo can consume a small amount of device memory.
    Restore the caller's selected device, including when a query fails.
    """
    if runtime is None:
        from cupy.cuda import runtime
    count = runtime.getDeviceCount()
    if not count:
        return []
    previous = runtime.getDevice()
    records = []
    try:
        for ordinal in range(count):
            row = dict(device=ordinal, usable=False)
            try:
                properties = runtime.getDeviceProperties(ordinal)
                name = properties['name']
                row.update(name=name.decode() if isinstance(name, bytes) else str(name),
                           major=int(properties['major']), minor=int(properties['minor']),
                           multiprocessors=int(properties['multiProcessorCount']),
                           total_bytes=int(properties['totalGlobalMem']))
                attributes = {}
                for key, attribute in (('clock_khz', 13), ('fp32_fp64_ratio', 87)):
                    try:
                        attributes[key] = int(runtime.deviceGetAttribute(attribute, ordinal))
                    except Exception as exc:
                        attributes[key] = None
                        row[key + '_error'] = str(exc)
                row.update(attributes)
                runtime.setDevice(ordinal)
                free, total = runtime.memGetInfo()
                row.update(free_bytes=int(free), total_bytes=int(total), usable=True)
                row['fp64_peak_flops'] = fp64_peak_flops(
                    row['major'], row['minor'], row['multiprocessors'],
                    row['clock_khz'], row['fp32_fp64_ratio'])
            except Exception as exc:
                row['error'] = str(exc)
            records.append(row)
    finally:
        runtime.setDevice(previous)
    return records


def select_fp64_device(records, *, minimum_free_bytes=0, overrides=None):
    """Select the highest estimated FP64 peak among usable visible devices.

    Optional overrides map CUDA-local ordinals to FP64 FLOP/s (not TFLOP/s).
    Fail closed when any eligible GPU has unknown throughput: silently ignoring
    it cannot establish which device is fastest. Tie-break by free memory, then
    ordinal. The returned record includes the ranking score and its provenance.
    """
    if minimum_free_bytes < 0:
        raise ValueError('minimum_free_bytes must be nonnegative')
    overrides = overrides or {}
    visible = {r['device'] for r in records}
    if set(overrides) - visible:
        raise ValueError('FP64 override refers to a device outside CUDA visibility')
    if any(not math.isfinite(v) or v <= 0 for v in overrides.values()):
        raise ValueError('FP64 overrides must be finite positive FLOP/s')
    eligible = []
    for record in records:
        if not record.get('usable') or record['free_bytes'] < minimum_free_bytes:
            continue
        score = overrides.get(record['device'], record.get('fp64_peak_flops'))
        if score is None or not math.isfinite(score) or score <= 0:
            raise ValueError(f"Unknown FP64 throughput for visible device {record['device']}; supply an override")
        eligible.append(dict(record, selection_fp64_flops=score,
                             selection_source='override' if record['device'] in overrides else 'cuda_attributes'))
    if not eligible:
        raise ValueError('No usable CUDA device satisfies the free-memory requirement')
    return max(eligible, key=lambda r: (r['selection_fp64_flops'], r['free_bytes'], -r['device']))
