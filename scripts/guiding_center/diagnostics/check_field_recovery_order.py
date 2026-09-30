"""Bounded stationary matrix diagnostic for electric-field recovery order.

This uses manufactured data on a fixed unit square, at most 512 triangles,
NumPy assembly and SciPy direct solves. It never advances a time scheme.
NUMBA_DISABLE_JIT=1 is mandatory: this diagnostic must not compile kernels.
The degree-p solve is an accuracy reference, not a recovery implementation.
"""
from __future__ import annotations

import argparse
import json
import os


def check_field_recovery_order(density_order=3, subdivisions=(2, 4, 8), jitter=0., stabilization=1., case="trig"):
    """Measure full-domain L2 errors; report evidence without certifying a rate."""
    if os.environ.get('NUMBA_DISABLE_JIT') != '1':
        raise RuntimeError('Run this matrix diagnostic with NUMBA_DISABLE_JIT=1')
    import numba
    import numpy as np

    if not numba.config.DISABLE_JIT:
        raise RuntimeError('Numba was imported before JIT was disabled')
    from hdgfem.core.mesh import DGMesh, rectangle_mesh
    from hdgfem.core.space import DGSpace
    from hdgfem.core.transfer import project_same_mesh_field
    from hdgfem.mixed.postprocess.flux import _postprocess_diffusion_solution
    from hdgfem.solvers.diffusion_reaction import solve_diffusion_reaction_hdg

    p = int(density_order)
    sizes = tuple(int(n) for n in subdivisions)
    if p != density_order or not 2 <= p <= 6:
        raise ValueError('density_order must be an integer between 2 and 6')
    if len(sizes) < 2 or any(n < 1 or n > 16 for n in sizes) or any(a >= b for a, b in zip(sizes, sizes[1:])):
        raise ValueError('subdivisions must be increasing integers between 1 and 16')
    if not 0. <= jitter <= 0.2 or not np.isfinite(stabilization) or stabilization <= 0:
        raise ValueError('jitter must be in [0, 0.2] and stabilization finite and positive')

    # Nonzero boundary data also tests trace elimination and avoids accidental
    # cancellation specific to homogeneous data or a polynomial exact solution.
    if case == 'trig':
        potential = lambda x, y: np.sin(np.pi*(x+.23))*np.sin(np.pi*(y+.17))
        source_exact = lambda x, y: 2.*np.pi**2*potential(x, y)
        field_exact = lambda x, y: -np.pi*np.stack((
            np.cos(np.pi*(x+.23))*np.sin(np.pi*(y+.17)),
            np.sin(np.pi*(x+.23))*np.cos(np.pi*(y+.17))), axis=-1)
    elif case == 'poly':
        potential = lambda x, y: (x+.5*y)**(p+1)
        source_exact = lambda x, y: -1.25*p*(p+1)*(x+.5*y)**(p-1)
        field_exact = lambda x, y: -(p+1)*np.stack(((x+.5*y)**p, .5*(x+.5*y)**p), axis=-1)
    elif case == 'harmonic':
        potential = lambda x, y: np.real((x+1j*y)**(p+1))
        source_exact = lambda x, y: 0.*x
        field_exact = lambda x, y: -(p+1)*np.stack((
            np.real((x+1j*y)**p), -np.imag((x+1j*y)**p)), axis=-1)
    else:
        raise ValueError('case must be trig, poly, or harmonic')
    rows = []
    for n in sizes:
        mesh = rectangle_mesh(n, n, xlim=(0., 1.), ylim=(0., 1.))
        if jitter:
            nodes = mesh.node_coords.copy()
            interior = np.all((nodes > 0.) & (nodes < 1.), axis=1)
            rng = np.random.default_rng(1907+n)
            nodes[interior] += rng.uniform(-jitter/n, jitter/n, (int(interior.sum()), 2))
            mesh = DGMesh.from_arrays(nodes, mesh.triangles)
        nquad = max(12, p+4)
        density_space = DGSpace(mesh, p, basis_type='dub_orth', volume_quad_1d=nquad, edge_quad_1d=nquad)
        space = DGSpace(mesh, p-1, basis_type='dub_orth', volume_quad_1d=nquad, edge_quad_1d=nquad)
        source = density_space.project_callable(source_exact)
        solve_kwargs = dict(stabilization=stabilization, solver='direct', preconditioner=None,
                            assembly_backend='numpy', local_solver_backend='numpy',
                            boundary_mode='eliminate', trace_basis='legendre-modal', verbose=False)
        low = solve_diffusion_reaction_hdg(project_same_mesh_field(source, space), 0., potential, space,
                                         **solve_kwargs)
        high = solve_diffusion_reaction_hdg(source, 0., potential, density_space, **solve_kwargs)
        trace_space = space.trace_space('legendre-modal')
        primal, combined, _ = _postprocess_diffusion_solution(
            low.local_unknowns, low.trace, space, stabilization, 1., 'both', trace_space=trace_space)
        _, direct, _ = _postprocess_diffusion_solution(
            low.local_unknowns, low.trace, space, stabilization, 1., 'flux', trace_space=trace_space)
        _, rt, _ = _postprocess_diffusion_solution(
            low.local_unknowns, low.trace, space, stabilization, 1., 'flux', trace_space=trace_space,
            flux_postprocess_space='RT_projection')

        ref = density_space.quad_data.Krf_quads
        points = density_space.mapped_quads()
        x, y = points[..., 0], points[..., 1]
        exact = field_exact(x, y)
        weights = density_space.quad_data.Krf_w

        def norm(values):
            return float(np.sqrt(np.einsum('kq,q,k->', np.sum(values**2, axis=-1),
                                            weights, mesh.aff_jacs)))

        def error(values):
            return norm(values-exact)

        errors = {name: error(np.stack([c.values_at_ref(ref) for c in field.components], axis=-1))
                  for name, field in (('raw', low.flux), ('RT_projection', rt), ('l2_closest', direct),
                                      ('primal_reference', combined), ('degree_p_solve_reference', high.flux))}
        errors['potential_gradient'] = error(-np.stack(primal.grad_at_ref(ref), axis=-1))

        # A patch fit cannot assume that the raw flux is its exact L2 projection.
        # Measure that premise for the interior moments used by HDG recovery.
        moment_space = DGSpace(mesh, p-2, basis_type='dub_orth', volume_quad_1d=nquad)
        moment_basis = moment_space.basis_at(ref)
        raw_values = np.stack([c.values_at_ref(ref) for c in low.flux.components], axis=-1)
        moment_coeffs = np.einsum('ij,qj,kqc->kic', moment_space.quad_data.MKrf_inv,
                                  weights[:, None]*moment_basis, raw_values-exact)
        potential_mean_errors = (low.field.values_at_ref(ref)-potential(x, y)) @ weights / weights.sum()
        moment_errors = {
            'flux_interior_moments': norm(np.einsum('qi,kic->kqc', moment_basis, moment_coeffs)),
            'potential_cell_means': float(np.sqrt(np.dot(potential_mean_errors**2,
                                                         mesh.aff_jacs*weights.sum()))),
        }
        row = dict(subdivisions=n, h=1./n, triangles=mesh.num_tri,
                   reference_trace_dofs=mesh.num_edg*(p+1), electric_field_l2_error=errors,
                   observed_orders={}, moment_l2_error=moment_errors, observed_moment_orders={})
        if rows:
            previous = rows[-1]
            row['observed_orders'] = {name: float(np.log(previous['electric_field_l2_error'][name]/value)
                                                    / np.log(previous['h']/row['h']))
                                      for name, value in errors.items()}
            row['observed_moment_orders'] = {
                name: float(np.log(previous['moment_l2_error'][name]/value) / np.log(previous['h']/row['h']))
                for name, value in moment_errors.items()
            }
        rows.append(row)
    return dict(density_order=p, poisson_order=p-1, recovered_field_order=p,
                required_electric_field_l2_order=p+1, jitter=jitter, stabilization=stabilization,
                case=case, flux_moment_degree=p-2,
                rows=rows,
                limitation=('Small stationary matrix diagnostic only. Observed rates on these meshes '
                            'are not a theorem or a production/GPU qualification. '
                            'The degree-p solve reference is an additional global solve, not local recovery.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--density-order', type=int, default=3)
    parser.add_argument('--subdivisions', type=int, nargs='+', default=[2, 4, 8])
    parser.add_argument('--jitter', type=float, default=0.)
    parser.add_argument('--stabilization', type=float, default=1.)
    parser.add_argument('--case', choices=('trig', 'poly', 'harmonic'), default='trig')
    args = parser.parse_args()
    print(json.dumps(check_field_recovery_order(args.density_order, args.subdivisions,
                                                args.jitter, args.stabilization, args.case), indent=2))
