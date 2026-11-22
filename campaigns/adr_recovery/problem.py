"""Deterministic recovery inputs on distorted affine triangles.

These are nontrivial local mixed fields, not a manufactured PDE accuracy run.
The independent Numba recovery is the numerical oracle. Both cache policies
receive identical reconstructed primal/flux coefficients and trace values.
"""
import numpy as np

from hybridge import DGSpace, rectangle_mesh
from hybridge.core.mesh import DGMesh
from hybridge.mixed.adr_preparation import prepare_adr_data


def mesh(nx, ny=None):
    original = rectangle_mesh(nx, nx if ny is None else ny)
    xy = original.node_coords.copy()
    x, y = xy.T.copy()
    scale = .15/max(nx, ny or nx)
    xy[:, 0] += scale*np.sin(3*x+2*y)*(1-x*x)*(1-y*y)
    xy[:, 1] += scale*np.cos(2*x-4*y)*(1-x*x)*(1-y*y)
    return DGMesh(xy, original.triangles.copy())


def chunk_mesh(full, first, stop):
    triangles = full.triangles[first:stop]
    ids, inverse = np.unique(triangles, return_inverse=True)
    return DGMesh(full.node_coords[ids].copy(), inverse.reshape(-1, 3))


def problem(order, geometry, *, variable_tensor=False, change=0.):
    space = DGSpace(geometry, order, basis_type='dub_orth')
    trace_space = space.trace_space('legendre-modal')
    coefficient_space = DGSpace(geometry, 1, basis_type='dub_orth')
    beta = (coefficient_space*coefficient_space).field((
        coefficient_space.project_callable(lambda x,y: .2+.1*y+.03*change*x),
        coefficient_space.project_callable(lambda x,y: -.1+.07*x-.02*change*y)))
    diffusion = tuple(coefficient_space.project_callable(f) for f in (
        lambda x,y: 1.2+.1*x, lambda x,y: .13+.02*y,
        lambda x,y: -.07+.01*x, lambda x,y: .9+.1*y)) if variable_tensor else .01/30.
    prepared = prepare_adr_data(space.constant(1.), space.constant(.3), beta, space,
        diffusion=diffusion, diffusion_stabilization='global_length', trace_space=trace_space)
    centers = geometry.node_coords[geometry.triangles].mean(axis=1)
    modes = np.arange(3*space.el_dof)[None, :]
    local = .07*np.sin(3*centers[:, 0, None]+5*centers[:, 1, None]+1.37*modes+change)/(1+modes%space.el_dof)
    local[:, 0] += 1.5
    midpoints = geometry.node_coords[geometry.edges].mean(axis=1)
    edge_modes = np.arange(trace_space.edg_dof)[None, :]
    trace = .03*np.cos(2*midpoints[:, 0, None]-3*midpoints[:, 1, None]+edge_modes+change)/(1+edge_modes)
    trace[:, 0] += 1.
    return dict(space=space, trace_space=trace_space, beta=beta, diffusion=diffusion,
                prepared=prepared, local=np.ascontiguousarray(local), trace=trace.ravel())
