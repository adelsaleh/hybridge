"""Warmed tensor recovery timings and algebra parity; no global solve or time stepping.

Run as a module from the repository root. Both paths sample the same variable
nonsymmetric tensor, assemble the same Neumann problem, and use batched LU.
The reference uses independent CuPy contractions; production uses fused CUDA.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from hdgfem import DGSpace, rectangle_mesh
from hdgfem.runtime.benchmarking import measure
from hdgfem.core.space import VectorDGField
from hdgfem.solvers.diffusion_reaction import _build_hdg_postprocess_cache


def benchmark(order, nx, repeats):
    """Return synchronized wall timings with sampling, allocations and solution included."""
    import cupy as cp
    from hdgfem.core.device import field_from_cupy_coefficients
    from hdgfem.backends.advection_diffusion_reaction_cupy import (
        _primal_system_cupy, postprocess_primal_cupy)
    from scripts.advection_diffusion_reaction.cases.tensor_cases import raw_cuda_coefficient

    space = DGSpace(rectangle_mesh(nx, nx), order, basis_type='dub_orth')
    cache = _build_hdg_postprocess_cache(space, space.trace_space('legendre-modal'),
                                       want_primal=False, want_flux=False)
    post, n = cache.post_space, space.mesh.num_tri
    q = post.quad_data
    flux = VectorDGField(tuple(field_from_cupy_coefficients(post, cp.ones(post.shape)) for _ in range(2)))
    local = cp.ones((n, 3*space.el_dof))
    samples = (cp.full((n, q.Krf_w.size, 2), .2),
               cp.full((n, 3, q.weights_JGL.size, 2), .2),
               cp.full((n, 3, q.weights_JGL.size), .6))
    diffusion = raw_cuda_coefficient('variable-full')

    def reference():
        """Independent contraction-based assembly plus batched solve."""
        matrix, rhs = _primal_system_cupy(local, flux, space, cache, samples, diffusion)
        return cp.linalg.solve(matrix, rhs[..., None])[:, :post.el_dof, 0]

    def production():
        """Production fused assembly plus batched solve, with tensor resampling."""
        return postprocess_primal_cupy(local, flux, space, cache, samples, diffusion)

    expected, actual = reference(), production()
    np.testing.assert_allclose(actual.coeffs, cp.asnumpy(expected), rtol=2e-9, atol=2e-9)
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    device_name = properties['name']
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    result = dict(order=order, elements=n, device=device_name, recovery_dtype='float64')
    for name, function in [('reference', reference), ('production', production)]:
        def synchronized():
            """Include all GPU work in the wall-clock measurement."""
            function()
            cp.cuda.get_current_stream().synchronize()
        result[name] = measure(synchronized, repeats, .1)
    ref = result['reference']['median_seconds']
    new = result['production']['median_seconds']
    result.update(speedup=ref/new, elements_per_second=n/new)
    return result


def main():
    """Print repeatable machine-readable timing evidence for a bounded local workload."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nx', type=int, default=16)
    parser.add_argument('--orders', type=int, nargs='+', default=[1, 3, 4, 6])
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.nx < 1 or args.repeats < 1 or any(p < 0 or p > 6 for p in args.orders):
        parser.error('nx and repeats must be positive, orders must lie in 0..6')
    for order in args.orders:
        print(json.dumps(benchmark(order, args.nx, args.repeats)), flush=True)


if __name__ == '__main__':
    main()
