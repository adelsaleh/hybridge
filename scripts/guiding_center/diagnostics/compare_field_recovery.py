"""One-element recovery diagnostic; no global PDE solve or time integration.

Use NUMBA_DISABLE_JIT=1 to execute entirely without native compilation. The
prescribed exact trace isolates local recovery error; this is not a mesh
convergence study or a production-backend timing benchmark.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from hdgfem.assembly.hdg import block_source_moments, element_traces
from hdgfem.core.field_ops import project_callable_to_trace
from hdgfem.core.mesh import DGMesh
from hdgfem.core.space import DGSpace
from hdgfem.core.transfer import project_same_mesh_field
from hdgfem.solvers.diffusion_reaction import (
    _postprocess_diffusion_solution,
    diffusion_element_boundary_mats,
    local_solvers_numpy,
    split_diffusion_unknowns,
)


def compare_recovery(density_order=6, element_size=1.):
    """Compare recovered fields on one triangle with prescribed trace moments."""
    if density_order < 1 or not np.isfinite(element_size) or element_size <= 0:
        raise ValueError('positive density order and finite positive element size required')
    p, h = int(density_order), float(element_size)
    mesh = DGMesh.from_arrays(np.array([[0., 0.], [h, 0.], [0., h]]), np.array([[0, 1, 2]]))
    nquad = max(12, p+3)
    density_space = DGSpace(mesh, p, basis_type='dub_orth', volume_quad_1d=nquad, edge_quad_1d=nquad)
    space = DGSpace(mesh, p-1, basis_type='dub_orth', volume_quad_1d=nquad, edge_quad_1d=nquad)
    trace_space = space.trace_space('legendre-modal')
    potential = lambda x, y: np.exp(x+.5*y)
    density = density_space.project_callable(lambda x, y: -1.25*potential(x, y))
    source = project_same_mesh_field(density, space)
    trace = project_callable_to_trace(space, potential, trace_basis='legendre-modal', reduced=False)
    rhs = block_source_moments(source, space, num_blocks=3)
    rhs += np.einsum('kij,kj->ki', diffusion_element_boundary_mats(1., space, trace_space=trace_space),
                     element_traces(trace, space, trace_space=trace_space))
    local = np.einsum('kij,kj->ki', local_solvers_numpy(0., 1., space), rhs)
    _, raw = split_diffusion_unknowns(local, space)
    primal, combined, _ = _postprocess_diffusion_solution(local, trace, space, 1., 1., 'both',
                                                         trace_space=trace_space)
    _, direct, _ = _postprocess_diffusion_solution(local, trace, space, 1., 1., 'flux',
                                                   trace_space=trace_space)
    _, rt, _ = _postprocess_diffusion_solution(local, trace, space, 1., 1., 'flux',
        trace_space=trace_space, flux_postprocess_space='RT_projection')
    points = density_space.mapped_quads()
    exact = np.stack((-potential(points[..., 0], points[..., 1]),
                      -.5*potential(points[..., 0], points[..., 1])), axis=-1)

    def error(values):
        squared = np.sum((values-exact)**2, axis=-1)
        return float(np.sqrt(np.einsum('kq,q,k->', squared, density_space.quad_data.Krf_w, mesh.aff_jacs)))

    errors = {}
    for name, field in (('raw', raw), ('RT_projection', rt), ('l2_closest', direct), ('primal_reference', combined)):
        values = np.stack([c.values_at_ref(density_space.quad_data.Krf_quads) for c in field.components], axis=-1)
        errors[name] = error(values)
    errors['potential_gradient'] = error(-np.stack(primal.grad_at_ref(density_space.quad_data.Krf_quads), axis=-1))
    return dict(density_order=p, poisson_order=p-1, element_size=h,
        recovered_rt_order=rt.components[0].space.order,
        potential_gradient_order=primal.space.order-1,
        electric_field_l2_error=errors,
        limitation='Prescribed exact trace on one triangle; no global convergence or backend timing claim.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--density-order', type=int, default=6)
    parser.add_argument('--element-size', type=float, default=1.)
    arguments = parser.parse_args()
    print(json.dumps(compare_recovery(arguments.density_order, arguments.element_size), indent=2))
