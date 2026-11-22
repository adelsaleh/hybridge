"""Bounded actual-element fixtures shared by transport CUDA tuners.

The prefix keeps neighboring elements together. Cut interior edges become
sample boundaries with zero prescribed trace; original boundary values and
all reference quadrature/basis data are retained. This is a calibration
fixture, not a restriction of the complete production trace operator.
"""
from __future__ import annotations

import numpy as np

from hybridge.core.mesh import DGMesh
from hybridge.core.space import DGSpace


def sample_space(space, *, limit=2048):
    """Return a compact host space, element IDs and original boundary-row IDs.

    Boundary row IDs are -1 on artificial cut boundaries. Only selected
    connectivity and vertices are copied; no full-mesh device or host mask is
    created. Ascending element IDs preserve interior trace orientation.
    """
    if int(limit) < 1:
        raise ValueError("transport tuning sample limit must be positive")
    mesh = space.mesh
    indices = np.arange(min(mesh.num_tri, int(limit)), dtype=np.int64)
    vertices, triangles = np.unique(mesh.triangles[indices], return_inverse=True)
    sample_mesh = DGMesh(mesh.node_coords[vertices], triangles.reshape(-1, 3))
    reference = space.reference
    sample = DGSpace(
        sample_mesh, space.order, basis_type=reference.basis_type,
        volume_quadrature=reference.volume_quadrature,
        volume_quad_1d=reference.volume_quad_1d,
        volume_degree=reference.volume_degree,
        edge_quad_1d=reference.edge_quad_1d,
    )
    original_edges = np.empty(sample_mesh.num_edg, dtype=np.int64)
    original_edges[sample_mesh.loc2glob_edge.ravel()] = mesh.loc2glob_edge[indices].ravel()
    boundary_edges = original_edges[sample_mesh.bnd_edges_inds]
    rows = np.searchsorted(mesh.bnd_edges_inds, boundary_edges)
    valid = rows < mesh.bnd_edges_inds.size
    valid[valid] &= mesh.bnd_edges_inds[rows[valid]] == boundary_edges[valid]
    rows[~valid] = -1
    return sample, indices, rows


def sample_raw_options(cupy, common_options, *, limit=2048):
    """Return ``(isolated_options, metadata)`` for the shared raw AR adapters.

    Scalar/projected/cross-space and quadrature coefficient representations
    remain unchanged. Reference tensors are read-only shared inputs; sampled
    coefficient and boundary arrays own their storage. Workspace/cache arguments
    are removed so a trial cannot overwrite retained production state.
    """
    from hybridge.core.device import as_cupy_space, as_cupy_trace_space

    options = dict(common_options)
    original = options["cspace"]
    host, indices, boundary_rows = sample_space(original.host, limit=limit)
    selected = cupy.asarray(indices)
    options["cspace"] = as_cupy_space(host, device=original.device_id)
    options["trace_ref"] = as_cupy_trace_space(
        host.trace_space(options["trace_ref"].kind), device=original.device_id)
    if options.get("beta_face_basis") is common_options["trace_ref"].bas_of_bd_quads:
        options["beta_face_basis"] = options["trace_ref"].bas_of_bd_quads
    for name in ("source_coeffs", "reaction_coeffs", "beta_volume_values",
                 "beta_normal_flux", "reaction_values"):
        value = options.get(name)
        if value is not None and getattr(value, "ndim", 0) >= 2:
            options[name] = cupy.ascontiguousarray(value[selected]).copy()
    beta = options.get("beta_coeffs")
    if beta is not None and getattr(beta, "ndim", 0) == 3:
        options["beta_coeffs"] = cupy.ascontiguousarray(beta[:, selected, :]).copy()
    original_boundary = options["boundary_trace"]
    boundary = cupy.zeros((boundary_rows.size, original_boundary.shape[1]),
                         dtype=original_boundary.dtype)
    valid = np.flatnonzero(boundary_rows >= 0)
    if valid.size:
        boundary[cupy.asarray(valid)] = original_boundary[cupy.asarray(boundary_rows[valid])]
    options["boundary_trace"] = boundary
    for name in ("workspace", "local_response", "factor_workspace"):
        options.pop(name, None)
    metadata = dict(elements=int(indices.size), original_elements=int(original.host.mesh.num_tri),
                    element_indices=indices.tolist(), original_boundary_rows=boundary_rows.tolist(),
                    artificial_boundary_edges=int(np.count_nonzero(boundary_rows < 0)),
                    policy="element-prefix-zero-cut-boundary-v1")
    return options, metadata
