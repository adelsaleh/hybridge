"""One process, one Hybridge tree: repeated ADR tensor assemblies on a fixed mesh, or a latency probe.

The loop follows ``assemble_projected_adr_trace_system_eliminated_raw_cuda``: device coefficient
sampling, retained diffusion, tau, pattern and mass factors, automatic measured selection, cached
reconstruction. Coefficients change every step, within one coefficient class. There is no global
solve: the question is the cost of the selection, and the solver path is covered by the tests.
Spans wrap module attributes; the tree's source is not edited.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import time
from pathlib import Path

import numpy as np
import cupy as cp

import hybridge
from hybridge import DGSpace, rectangle_mesh
from hybridge.mixed.coefficients import prepare_diffusion
from hybridge.mixed.coefficients_device import prepare_adr_data_cupy
from hybridge.mixed.raw_cuda import tensor as raw
from hybridge.mixed.raw_cuda import tensor_tuning
from hybridge.mixed.raw_cuda.adr_operator import (
    assemble_projected_adr_trace_operator_raw_cuda as assemble,
    reconstruct_projected_adr_local_unknowns_raw_cuda as reconstruct,
)
from hybridge.runtime import cuda_tuning


SPANS: dict[str, float] = {}
COUNTS: dict[str, int] = {}


def wrap(owner, name, label):
    """Accumulate wall time per step around one module or class attribute."""
    function = getattr(owner, name)

    def timed(*args, **kwargs):
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            SPANS[label] = SPANS.get(label, 0.) + time.perf_counter() - start
            COUNTS[label] = COUNTS.get(label, 0) + 1
    setattr(owner, name, timed)


def install_spans():
    """Spans of the identity work; ``cv.*`` lie inside ``select_variant``, ``tt.*`` outside it."""
    for name in ('_coefficient_class', 'array_digest', 'candidate_configurations', 'select_variant'):
        wrap(tensor_tuning, name, 'tt.'+name)
    for name in ('device_signature', 'fingerprint'):
        wrap(tensor_tuning, name, 'tt.'+name)
        wrap(cuda_tuning, name, 'cv.'+name)
    for name in ('kernel_attributes', 'legal_configuration'):
        wrap(tensor_tuning, name, 'tt.'+name)
    wrap(tensor_tuning, 'real_raw_module', 'tt.real_raw_module')
    wrap(raw, 'kernel_source', 'tt.kernel_source')
    wrap(pathlib.Path, 'read_text', 'io.read_text')
    # The structural key of the resolved arm, and what remains in the timer around it.
    for name in ('resolved_selection', '_identity_token', '_measure_tensor_execution'):
        if hasattr(tensor_tuning, name):
            wrap(tensor_tuning, name, 'key.'+name)
    for name in ('tensor_mass_placement', 'tensor_mass_inverse_mode', 'tensor_phase_clock'):
        wrap(raw, name, 'key.'+name)
    for name in ('tuning_mode', 'tuning_budget'):
        wrap(cuda_tuning, name, 'key.'+name)
    wrap(cp.cuda.runtime, 'getDevice', 'key.getDevice')


GC_PAUSE = [0., 0.]


def gc_clock(phase, info):
    """Attribute collector pauses to the step in which they happen."""
    if phase == 'start':
        GC_PAUSE[1] = time.perf_counter()
    else:
        GC_PAUSE[0] += time.perf_counter() - GC_PAUSE[1]


def diffusion_case(name):
    """The constant scalar of the turbulence runs, or a variable tensor that retains mass factors."""
    if name == 'scalar':
        return 1.e-3
    return (lambda x, y: 2.e-3+.1e-3*x, lambda x, y: .3e-3+.02e-3*y,
            lambda x, y: -.1e-3+.01e-3*x, lambda x, y: 1.e-3+.05e-3*y)


def loop(args):
    """Run the assemblies and write one JSON line per step."""
    mesh = rectangle_mesh(args.nx, args.nx)
    space = DGSpace(mesh, args.order, basis_type='dub_orth')
    trace = space.trace_space('legendre-modal')
    diffusion = diffusion_case(args.diffusion)
    host_x = space.project_callable(lambda x, y: np.sin(2.*np.pi*x)*np.cos(2.*np.pi*y)).coeffs
    host_y = space.project_callable(lambda x, y: -np.cos(2.*np.pi*x)*np.sin(2.*np.pi*y)).coeffs
    tensor = prepare_diffusion(diffusion, space, device=True)
    static: dict = {}
    trace_values = cp.sin(cp.arange(mesh.num_edg*trace.edg_dof, dtype=cp.float64))
    if args.spans:
        install_spans()
    gc.callbacks.append(gc_clock)
    written = args.out/'steps.jsonl'
    with written.open('w') as stream:
        for step in range(args.steps):
            scale = 1.+.1*np.sin(.37*step)
            beta = (space*space).field((space.field(scale*host_x), space.field(scale*host_y)))
            source = lambda x, y, s=scale: s*cp.sin(2.*np.pi*x)*cp.cos(2.*np.pi*y)
            SPANS.clear()
            COUNTS.clear()
            GC_PAUSE[0] = 0.
            start = time.perf_counter()
            prepared = prepare_adr_data_cupy(source, .4+.01*scale, beta, space, diffusion=diffusion, trace_space=trace,
                                             tau_diffusion=static.get('tau_diffusion'))
            middle = time.perf_counter()
            operator = assemble(prepared, 0., space, diffusion=diffusion, trace_space=trace, matrix_format='bsr',
                                block_size='auto', prepared_diffusion=tensor, csr_pattern=static.get('pattern'),
                                mass_factors=static.get('mass_factors'))
            static.update(tau_diffusion=prepared.tau_diffusion, pattern=operator.csr_pattern,
                          mass_factors=operator.mass_factors)
            finish = time.perf_counter()
            unknowns, reconstruction = reconstruct(operator, trace_values)
            cp.cuda.Stream.null.synchronize()
            end = time.perf_counter()
            timings, record = operator.assembly.timings, operator.reconstruction_data['tuning']
            stream.write(json.dumps(dict(
                step=step, jit=timings['raw.kernel.jit'], tuning_wall=timings['raw.tuning.wall'],
                tuning_reused=timings['raw.tuning.reused'], source=record['source'], variant=record['variant'],
                sample_elements=timings['raw.tuning.sample_elements'], kernel_device=timings['raw.kernel.device'],
                raw_total=timings.get('raw.wall_total'), block=timings.get('raw.kernel.block_size'),
                recon_jit=reconstruction['raw.reconstruction.jit'], recon_device=reconstruction['raw.reconstruction.device'],
                prepare_s=middle-start, assemble_s=finish-middle, step_s=end-start, gc_s=GC_PAUSE[0],
                spans=dict(SPANS), counts=dict(COUNTS)))+'\n')
            stream.flush()
            if args.save and step in (1, args.steps-1):
                np.savez(args.save.parent/f'{args.save.name}-step{step}.npz', data=cp.asnumpy(operator.assembly.data),
                         rhs=cp.asnumpy(operator.assembly.rhs), unknowns=cp.asnumpy(unknowns))
    return dict(elements=mesh.num_tri, order=args.order)


def probe(args):
    """Time candidate blocking operations in turn, to see which of them stall on a shared node."""
    array = cp.ones(1024, dtype=cp.int32)
    text = Path(tensor_tuning.__file__)
    payload = 'x'*45000
    ordinal = cp.cuda.runtime.getDevice()
    operations = dict(
        device_properties=lambda: cp.cuda.runtime.getDeviceProperties(ordinal),
        nfs_read_text=lambda: text.read_text(),
        python_hash=lambda: cuda_tuning.fingerprint(dict(payload=payload)),
        device_sync_any=lambda: bool(cp.any(array)),
        python_idle=lambda: None,
    )
    samples = {name: [] for name in operations}
    events = []
    stop = time.perf_counter() + args.probe_seconds
    while time.perf_counter() < stop:
        for name, operation in operations.items():
            start = time.perf_counter()
            operation()
            elapsed = time.perf_counter() - start
            samples[name].append(elapsed)
            if elapsed > .02:
                events.append(dict(operation=name, at=start, seconds=elapsed))
        time.sleep(.005)
    summary = {name: dict(count=len(v), median_ms=1e3*float(np.median(v)), p99_ms=1e3*float(np.percentile(v, 99)),
                          max_ms=1e3*float(np.max(v)), over_20ms=int(np.sum(np.array(v) > .02)))
               for name, v in samples.items()}
    (args.out/'probe.json').write_text(json.dumps(dict(summary=summary, events=events), indent=1)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tree', type=Path, required=True)
    parser.add_argument('--order', type=int, default=6)
    parser.add_argument('--nx', type=int, default=8)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--diffusion', default='scalar', choices=('scalar', 'variable'))
    parser.add_argument('--spans', action='store_true')
    parser.add_argument('--save', type=Path)
    parser.add_argument('--probe-seconds', type=float, default=0.)
    args = parser.parse_args()
    module = Path(hybridge.__file__).resolve()
    if not module.is_relative_to(args.tree.resolve()):
        raise RuntimeError(f'imported {module}, expected the tree {args.tree}')
    args.out.mkdir(parents=True, exist_ok=True)
    result = probe(args) if args.probe_seconds else loop(args)
    (args.out/'harness.json').write_text(json.dumps(dict(
        hybridge=str(module), node=os.uname().nodename, job=os.environ.get('SLURM_JOB_ID'), result=result), indent=1)+'\n')


if __name__ == '__main__':
    main()
