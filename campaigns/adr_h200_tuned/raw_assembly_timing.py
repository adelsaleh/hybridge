"""Time the tuned raw-CUDA ADR tensor assembly on a ladder system's mesh, degree and case.

This is an assembly cost reference, not the solved system. hybridge's raw path uses its own
trace basis (Lagrange or Legendre, not Bernstein), quadrature and stabilization, so its
matrix differs from the vendored cache that the solvers and PyPardiso use
(docs/research/solver_studies/adr_scaling_2026_09_17/baseline/adr_assembler_comparison_2026_09_17.md).

It runs the stages of hybridge's production raw-CUDA ADR assembly
(``assemble_projected_adr_trace_system_eliminated_raw_cuda``): device coefficient sampling,
the tensor kernel, then reconstruction from a zero trace. The solve is skipped. Launch
settings follow the sm_90 records per degree: ``block_size`` from the plan (32 at p<=4,
otherwise ``auto``), the explicit kind-6 mass inverse at p>=7 (code default), and BSR with
``schur-lu+mass`` factor caching. The first call includes NVRTC compilation; warm calls give
the medians. Registers, local memory and occupancy of the tensor kernel are recorded
(``raw.kernel.*``) to watch the spills listed in docs/algorithms/assembly/h200_sm90/README.md.
"""
from __future__ import annotations

import dataclasses
import gc
import statistics
import sys
from time import perf_counter
import types


def _master_catalogue(master_root):
    """Import master's ADR case catalogue in a worker that has the vendored tree on ``sys.path``.

    Master's ``scripts`` is a namespace package and the vendored one a regular package, which
    Python prefers wherever it sits on the path. Bind ``scripts`` to master's directory for the
    rest of this process: this job runs alone in its worker and needs no vendored script.
    """
    for key in [k for k in sys.modules if k == 'scripts' or k.startswith('scripts.')]:
        del sys.modules[key]
    package = types.ModuleType('scripts')
    package.__path__ = [str(master_root/'scripts')]
    sys.modules['scripts'] = package
    from scripts.advection_diffusion_reaction.cases import catalogue, closed_loop_stress_cases
    return catalogue, closed_loop_stress_cases


def _constant(value):
    return lambda x, y: value+0*x


def problem_for(spec, master_root):
    catalogue, stress = _master_catalogue(master_root)
    if 'stress_parameters' in spec:
        # The campaign's own stress parameters and frozen velocity normalization.
        kwargs, _ = stress.make_case(stress.StressParameters(**spec['stress_parameters']), spec['velocity_normalization'])
        problem = catalogue.AdvectionDiffusionReactionProblem(**kwargs)
    else:
        problem = catalogue.CASE_DEFINITIONS[spec['case']].build()
    # The legacy square cases give a constant velocity as two numbers; the device sampling
    # takes callables, arrays or DG fields.
    beta = tuple(_constant(float(b)) if isinstance(b, (int, float)) else b for b in problem.beta)
    return dataclasses.replace(problem, beta=beta)


def measure(spec, master_root):
    import cupy as cp
    import numpy as np
    from hybridge import DGSpace, rectangle_mesh
    from hybridge.core.mesh import as_dg_mesh
    from hybridge.mixed.coefficients_device import prepare_adr_data_cupy
    from hybridge.mixed.raw_cuda.adr_operator import (
        assemble_projected_adr_trace_operator_raw_cuda, reconstruct_projected_adr_local_unknowns_raw_cuda,
    )
    from hybridge.solvers.advection_diffusion_reaction import AdvectionDiffusionReactionHDGOptions

    from hybridge.mixed.raw_cuda import tensor as raw_tensor
    if not getattr(raw_tensor.with_mass_inverse, 'accepts_device_kinds', False):
        # Frozen copies whose tensor.py predates the branch fix test the kind-6 elements with
        # np.asarray, which rejects the CuPy kinds of device-sampled variable diffusion (p>=7).
        original = raw_tensor.with_mass_inverse

        def with_mass_inverse(workspace, kinds, **options):
            return original(workspace, cp.asnumpy(kinds) if isinstance(kinds, cp.ndarray) else kinds, **options)
        with_mass_inverse.accepts_device_kinds = True
        raw_tensor.with_mass_inverse = with_mass_inverse
    problem = problem_for(spec, master_root)
    if spec.get('mesh_path'):
        with np.load(spec['mesh_path'], allow_pickle=False) as data:
            mesh = as_dg_mesh((data['node_coords'], data['triangles']))
    else:
        mesh = rectangle_mesh(spec['n'], spec['n'])
    p = spec['p']
    space = DGSpace(mesh, p, basis_type='dub_orth')
    trace = space.trace_space(spec['trace_basis'])
    options = AdvectionDiffusionReactionHDGOptions()
    trace_values = cp.zeros(mesh.num_edg*(p+1), dtype=cp.float64)
    sync = cp.cuda.Device().synchronize

    def call():
        coefficient_timings = {}
        sync()
        start = perf_counter()
        prepared = prepare_adr_data_cupy(
            problem.source, problem.reaction, problem.beta, space, diffusion=problem.diffusion,
            advection_stabilization=options.advection_stabilization,
            diffusion_stabilization=options.diffusion_stabilization,
            diffusion_penalty_constant=options.diffusion_penalty_constant,
            trace_space=trace, timings=coefficient_timings)
        operator = assemble_projected_adr_trace_operator_raw_cuda(
            prepared, problem.boundary_condition, space, diffusion=problem.diffusion, trace_space=trace,
            matrix_format=spec['matrix_format'], block_size=spec['block_size'],
            cache_local_factors=spec['cache_local_factors'])
        sync()
        assembled = perf_counter()
        _, recon = reconstruct_projected_adr_local_unknowns_raw_cuda(operator, trace_values, block_size=spec['block_size'])
        sync()
        done = perf_counter()
        timings = dict(operator.assembly.timings, **coefficient_timings)
        row = dict(assembly_ms=1000*(assembled-start), reconstruction_ms=1000*(done-assembled),
                   trace_dofs=int(operator.assembly.rhs.size), stored_values=int(operator.assembly.data.size),
                   timings={k: v for k, v in timings.items() if isinstance(v, (int, float, str))},
                   reconstruction_timings={k: v for k, v in recon.items() if isinstance(v, (int, float))})
        del operator, prepared
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        return row

    calls = []
    for index in range(max(2, spec.get('timed_calls', 3))):
        calls.append(call())
        row = calls[-1]
        stages = {k: v for k, v in row['timings'].items() if k.startswith('raw.coefficients.') and isinstance(v, float)
                  and not k.endswith('host_fallback')}
        print(f"[raw_assembly] call {index+1}: assembly {row['assembly_ms']:.1f} ms, reconstruction "
              f"{row['reconstruction_ms']:.1f} ms; kernel {1000*row['timings'].get('raw.kernel.device', 0):.1f} ms; "
              + ', '.join(f"{k.removeprefix('raw.coefficients.')} {1000*v:.1f} ms" for k, v in stages.items()), flush=True)
    warm = calls[1:]
    median = lambda key: float(statistics.median(c[key] for c in warm))
    first = calls[0]['timings']
    fallbacks = sorted({k.removesuffix('.host_fallback') for c in calls for k in c['timings'] if k.endswith('host_fallback')})
    if fallbacks:
        print(f'[raw_assembly] WARNING host coefficient sampling: {", ".join(fallbacks)}', flush=True)
    return dict(
        status='passed', family='raw_assembly', candidate=spec['candidate'], case=spec['case'], n=spec['n'], p=p,
        triangles=int(mesh.num_tri), trace_dofs=calls[0]['trace_dofs'], trace_basis=spec['trace_basis'],
        matrix_format=spec['matrix_format'], block_size=spec['block_size'],
        cache_local_factors=spec['cache_local_factors'],
        first_assembly_ms=calls[0]['assembly_ms'], assembly_median_ms=median('assembly_ms'),
        reconstruction_median_ms=median('reconstruction_ms'),
        kernel_resources={k: v for k, v in first.items() if k.startswith('raw.kernel.')},
        host_fallbacks=fallbacks,
        calls=calls,
        note='hybridge raw-CUDA discretization (not the vendored system the solvers use); assembly cost reference.')
