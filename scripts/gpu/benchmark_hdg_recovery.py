"""Device-only reconstruction and flux/primal recovery on a prescribed trace.

No global solve or CPU reference benchmark. Finite values and the primal mean
constraint are checked outside timers. This is an algebra/stage-cost benchmark,
not a manufactured-solution accuracy result. Unsupported kernels fail explicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--equation', choices=('poisson', 'adr'), required=True)
    parser.add_argument('--order', type=int, required=True)
    parser.add_argument('--nx', type=int, default=16)
    parser.add_argument('--trace-basis', choices=('legacy-lagrange', 'legendre-modal'), default='legendre-modal')
    parser.add_argument('--coefficient', choices=('scalar', 'constant-full', 'variable-full'), default='scalar')
    parser.add_argument('--cache-policy', choices=('none', 'schur-lu', 'schur-lu+mass', 'schur-cholesky'), default='schur-lu')
    parser.add_argument('--flux-space', choices=('l2_closest', 'RT_projection'), default='l2_closest')
    parser.add_argument('--block-size', type=int, choices=(32, 64, 128), default=128)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nvtx', action='store_true', help='tag the workload for Nsight; profile timings are not ranking samples')
    return parser


def benchmark(args):
    import cupy as cp
    import numpy as np
    from hybridge import DGSpace, rectangle_mesh
    from hybridge.core.device import as_cupy_coefficients, as_cupy_space
    from hybridge.runtime.benchmarking import measure_synchronized
    from hybridge.mixed.postprocess.flux import _build_hdg_postprocess_cache
    from hybridge.mixed.postprocess.flux_cupy import postprocess_diffusion_primal_cupy
    from scripts.advection_diffusion_reaction.cases.tensor_cases import raw_cuda_coefficient

    space = DGSpace(rectangle_mesh(args.nx, args.nx), args.order, basis_type='dub_orth')
    trace_ref = space.trace_space(args.trace_basis)
    trace = cp.sin(cp.arange(space.mesh.num_edg * (args.order + 1), dtype=cp.float64))
    if args.equation == 'poisson':
        from hybridge.core.device import as_cupy_space
        from hybridge.mixed.cupy import (build_trace_reference, reference_derivative_mats,
                                        face_element_mass, source_moments_cupy, boundary_trace_values_cupy)
        from hybridge.mixed.raw_cuda import poisson
        cspace = as_cupy_space(space)
        device_trace = build_trace_reference(cspace, args.trace_basis)
        d0, d1 = reference_derivative_mats(cspace)
        common = dict(source_rhs=source_moments_cupy(space.constant(1.), cspace), cspace=cspace,
                      trace_ref=device_trace, d0_reference=d0, d1_reference=d1,
                      face_element_mass=face_element_mass(device_trace), tau=1.3)
        operator = poisson.assemble(**common, boundary_trace=boundary_trace_values_cupy(lambda x,y:0*x, cspace, device_trace),
            matrix_format='bsr', block_size=args.block_size, cache_local_factors=args.cache_policy != 'none',
            local_factor_kind=args.cache_policy if args.cache_policy != 'none' else 'schur-lu')

        def reconstruct():
            return poisson.reconstruct(**common, trace=trace, context=operator.poisson_context,
                       block_size=args.block_size, return_local_unknowns=True)[1]

        cache = _build_hdg_postprocess_cache(space, trace_ref, want_primal=False, want_flux=False)
        def primal(local):
            return postprocess_diffusion_primal_cupy(local, space, cache, 1.)
        flux_recovery = None
    else:
        from hybridge.mixed.adr_preparation import prepare_adr_data
        from hybridge.mixed.raw_cuda.adr_operator import (assemble_projected_adr_trace_operator_raw_cuda,
                                                        reconstruct_projected_adr_local_unknowns_raw_cuda)
        from hybridge.mixed.postprocess.total_flux import _postprocess_total_flux, _postprocess_primal_from_total_flux
        beta = (space * space).field((space.constant(.7), space.constant(-.2)))
        diffusion = raw_cuda_coefficient(args.coefficient)
        prepared = prepare_adr_data(space.constant(1.), .3, beta, space, diffusion=diffusion,
                                   trace_space=trace_ref, dense_local_matrices=False, diffusion_stabilization=.6)
        operator = assemble_projected_adr_trace_operator_raw_cuda(prepared, lambda x,y:0*x, space,
            diffusion=diffusion, trace_space=trace_ref, matrix_format='bsr', block_size=args.block_size,
            cache_local_factors=args.cache_policy)

        def reconstruct():
            return reconstruct_projected_adr_local_unknowns_raw_cuda(operator, trace, block_size=args.block_size)[0]

        def flux_recovery(local):
            return _postprocess_total_flux(local, trace, beta, prepared, space, trace_ref,
                                           None, args.flux_space, 'cupy')

        def primal(local, flux):
            return _postprocess_primal_from_total_flux(local, flux, beta, prepared, space, trace_ref,
                                                       None, diffusion, 'cupy')
    cp.cuda.get_current_stream().synchronize()
    phases = {}
    local, phases['reconstruction'] = measure_synchronized(reconstruct,
        synchronize=cp.cuda.get_current_stream().synchronize, warmups=2, repeats=args.repeats)
    if not bool(cp.all(cp.isfinite(local))):
        raise ArithmeticError('nonfinite reconstructed fields')
    if flux_recovery:
        flux, phases['flux_postprocessing'] = measure_synchronized(lambda: flux_recovery(local),
            synchronize=cp.cuda.get_current_stream().synchronize, warmups=2, repeats=args.repeats)
        function = lambda: primal(local, flux)
    else:
        function = lambda: primal(local)
    post, phases['primal_postprocessing'] = measure_synchronized(function,
        synchronize=cp.cuda.get_current_stream().synchronize, warmups=2, repeats=args.repeats)
    coefficients = as_cupy_coefficients(post, as_cupy_space(post.space))
    if not bool(cp.all(cp.isfinite(coefficients))):
        raise ArithmeticError('nonfinite primal postprocessing')
    q = post.space.quad_data
    weights = cp.asarray(q.Krf_w)
    post_mean = cp.asarray(q.phi).T @ weights
    base_mean = cp.asarray(space.basis_at(q.Krf_quads)).T @ weights
    observed = coefficients @ post_mean
    expected = local.reshape(-1, 3, space.el_dof)[:, 0] @ base_mean
    error = float((cp.linalg.norm(observed-expected) / cp.maximum(cp.linalg.norm(expected), 1.)).item())
    if not np.isfinite(error) or error > 2e-9:
        raise ArithmeticError(f'primal mean preservation failed: {error}')
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    from scripts.reports.run_provenance import mesh_fingerprint
    return dict(equation=args.equation, order=args.order, triangles=space.mesh.num_tri,
        mesh=mesh_fingerprint(space.mesh), timing='synchronized wall seconds; warmups and checks excluded',
        trace_basis=args.trace_basis, coefficient=args.coefficient, cache_policy=args.cache_policy,
        flux_space=args.flux_space if flux_recovery else None, block_size=args.block_size,
        phases=phases, primal_mean_relative_error=error, validation='prescribed trace: finite values and mean preservation',
        gpu=properties['name'].decode(), cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        qualification='high-order-probe' if args.order > 6 else 'existing-tested-range')


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not 1 <= args.order <= 9 or min(args.nx, args.repeats) < 1:
        parser.error('require p=1..9 and positive nx/repeats')
    if args.equation == 'poisson' and (args.coefficient != 'scalar' or args.cache_policy == 'schur-lu+mass'):
        parser.error('Poisson uses scalar diffusion and none/Schur LU/Schur Cholesky')
    if args.equation == 'adr' and args.cache_policy == 'schur-cholesky':
        parser.error('ADR uses none/Schur LU/Schur LU plus mass')
    started = time.monotonic()
    if args.nvtx:
        import cupy as cp
        cp.cuda.nvtx.RangePush('hybridge_recovery')
    try:
        row = benchmark(args)
    finally:
        if args.nvtx:
            cp.cuda.nvtx.RangePop()
    row['wall_seconds'] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(row, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
