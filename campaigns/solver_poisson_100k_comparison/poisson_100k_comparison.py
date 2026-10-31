"""One stationary GPU comparison; invoke while holding gpu-campaign.lock."""

# Canonical campaign sources and saved outputs have independent locations.
import sys as _campaign_sys
from pathlib import Path
_campaign_repo = Path(__file__).resolve().parents[2]
_campaign_sys.path.insert(0, str(_campaign_repo))
from campaigns.common.locations import (
    workspace_path as _campaign_workspace_path, data_directory as _campaign_data_directory,
    source_path as _campaign_source_path, source_directory as _campaign_source_directory,
)
_campaign_code = Path(__file__).resolve().parent
_campaign_sys.path.insert(0, str(_campaign_code))
_campaign_output = _campaign_data_directory('hybridge/artifacts')

import gc
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(str(_campaign_workspace_path('hybridge')))
sys.path.insert(0, str(_campaign_workspace_path('hybridge')))
OUT = ROOT / 'artifacts/poisson_100k_comparison.json'
LOG = OUT.with_suffix('.log').open('w', buffering=1)

def say(message):
    print(message, flush=True)
    print(message, file=LOG, flush=True)

# flock is held by the launcher. Also wait for uncoordinated device processes.
while True:
    active = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,process_name',
        '--format=csv,noheader'], text=True).strip()
    if not active:
        break
    say(f'Waiting for existing CUDA compute processes: {active}')
    time.sleep(5)

import cupy as cp
import cupyx.scipy.sparse as sparse
from hybridge import DGSpace, rectangle_mesh
from hybridge.core.device import as_cupy_space
from hybridge.hdg.cuda.tuning import device_signature
from hybridge.linalg.gpu.sparse import (
    _assembly_device_csr_matrix, _device_compressed_matvec,
)
from hybridge.mixed.cupy import (
    assemble_projected_diffusion_trace_system_eliminated_cupy as cupy_assemble,
    assemble_projected_diffusion_trace_system_eliminated_raw_cupy as raw_assemble,
    assemble_projected_diffusion_trace_rhs_cached_cupy as cupy_refresh,
    assemble_projected_diffusion_trace_rhs_eliminated_raw_cupy as raw_refresh,
    build_trace_reference, solve_mixed_from_scalar_cholesky_cupy,
)
from hybridge.mixed.raw_cuda import poisson

stream = cp.cuda.get_current_stream()
report = dict(device=device_signature(cp), triangles=100000,
              subdivisions=[250,200], precision='float64', trace_basis='legendre-modal',
              element_basis='dub_orth', tau=1.3, source=1., dirichlet=0.,
              warmups=1, repeats=3,
              timing='synchronized wall seconds; complete assembly wrappers; mesh/JIT/autotuning excluded',
              scope='stationary assembly/RHS/reconstruction only; no global solve, no NumPy assembly',
              cases=[])

def save():
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')

def measure(label, function):
    result = None
    samples = []
    for i in range(4):
        result = None
        gc.collect()
        stream.synchronize()
        start = time.perf_counter()
        result = function()
        stream.synchronize()
        elapsed = time.perf_counter()-start
        if i:
            samples.append(elapsed)
        say(f'  {label}: {"warmup" if i==0 else i} {elapsed:.6f} s')
    return result, dict(median_seconds=statistics.median(samples), samples_seconds=samples)

def relative(a, b):
    return float((cp.linalg.norm(a-b)/cp.maximum(cp.linalg.norm(b), 1.e-30)).item())

def matvec(assembled, vector):
    matrix = _assembly_device_csr_matrix(assembled, cp, sparse)
    return _device_compressed_matvec(matrix, vector, sparse, cp)

def boundary(x, y):
    return 0.*x

boundary._hybridge_constant_value = 0.
say('Reserved GPU via .cache/gpu-campaign.lock; existing CUDA compute processes have finished.')
save()

def run_order(p):
    say(f'p={p}: prepare exactly 100000 triangles')
    space = DGSpace(rectangle_mesh(250, 200), p, basis_type='dub_orth')
    device = as_cupy_space(space)
    tr = build_trace_reference(device, 'legendre-modal')
    source, reaction = space.constant(1.), space.zeros()
    reduced = int(device.mesh.int_edges_inds.size)*device.edg_dof
    vector = cp.sin(.731*cp.arange(reduced, dtype=cp.float64))
    trace = cp.sin(.273*cp.arange(device.mesh.num_edg*device.edg_dof, dtype=cp.float64))
    common = dict(trace_basis='legendre-modal', trace_ref=tr)
    row = dict(order=p, triangles=space.mesh.num_tri, trace_unknowns=reduced, raw=[])
    report['cases'].append(row)

    oracle, row['cupy_schur_cholesky_assembly'] = measure('CuPy Schur-Cholesky CSR assembly',
        lambda: cupy_assemble(source, reaction, boundary, 1.3, space,
                              **common, use_schur_cholesky=True))
    expected_mv, expected_rhs = matvec(oracle, vector), oracle.rhs.copy()
    row['cupy_factor_bytes'] = oracle.schur_cholesky_cache.local_factor_bytes
    row['cupy_retained_device_bytes'] = cp.get_default_memory_pool().used_bytes()
    refreshed, row['cupy_rhs'] = measure('CuPy cached RHS',
        lambda: cupy_refresh(source, boundary, device, tr, oracle))
    row['cupy_rhs_relative_error'] = relative(refreshed.rhs, expected_rhs)
    del refreshed

    def cupy_reconstruct():
        traces = trace.reshape(device.mesh.num_edg, device.edg_dof)[device.mesh.loc2glob_edge]
        traces = traces.reshape(device.mesh.num_tri, 3*device.edg_dof)
        rhs = oracle.source_rhs[...,None]+oracle.element_boundary_mats@traces[...,None]
        return solve_mixed_from_scalar_cholesky_cupy(oracle.schur_cholesky_cache,rhs).squeeze(-1)

    expected_local, row['cupy_reconstruction'] = measure('CuPy mixed reconstruction', cupy_reconstruct)
    del oracle
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    save()

    for policy, fmt in [('schur-cholesky','coo'), ('schur-cholesky','csr'),
                        ('schur-cholesky','bsr'), ('none','csr'), ('schur-lu','csr')]:
        label = f'raw {policy} {fmt}'
        options = dict(matrix_format=fmt, block_size='auto', cache_local_factors=policy!='none',
                       local_factor_kind=policy if policy!='none' else 'schur-lu')
        assembled, timing = measure(label+' assembly',
            lambda: raw_assemble(source, reaction, boundary, 1.3, space, **common, **options))
        raw = assembled.raw_assembly
        context = raw.poisson_context
        item = dict(policy=policy, format=fmt, assembly=timing,
                    matrix_relative_error=relative(matvec(assembled, vector), expected_mv),
                    rhs_relative_error=relative(assembled.rhs,expected_rhs),
                    factor_bytes=context.local_factor_bytes,
                    assembly_kernel_seconds=raw.timings['raw.kernel.device'],
                    tuning=context.reports)
        if fmt=='csr':
            refreshed, item['rhs'] = measure(label+' RHS',
                lambda: raw_refresh(source,reaction,boundary,1.3,space,cached_raw=raw,
                                    **common,block_size='auto'))
            item['cached_rhs_relative_error'] = relative(refreshed.rhs,expected_rhs)
            del refreshed
            recovered, item['reconstruction'] = measure(label+' mixed reconstruction',
                lambda: poisson.reconstruct(trace=trace,source_rhs=assembled.source_rhs,
                    cspace=device,trace_ref=tr,d0_reference=raw.d0_reference,
                    d1_reference=raw.d1_reference,face_element_mass=raw.face_element_mass,
                    tau=1.3,context=context,block_size='auto',return_local_unknowns=True))
            item['reconstruction_relative_error'] = relative(recovered[1],expected_local)
            del recovered
        for key, error in item.items():
            if key.endswith('relative_error') and error>2.e-10:
                raise AssertionError(f'{p} {label}: {key}={error}')
        row['raw'].append(item)
        save()
        say(f'  {label}: CuPy matrix/RHS parity passed')
        del assembled, raw, context
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    say(f'p={p}: complete')

try:
    for p in (7,8,9):
        run_order(p)
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    report['complete'] = True
    save()
    say(f'Done: {OUT}')
except BaseException as exc:
    report['error'] = repr(exc)
    save()
    raise
