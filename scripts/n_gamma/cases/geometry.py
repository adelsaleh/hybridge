"""Polygonal manufactured domains and reproducible mesh records.

Two domains, B (rectangle) and H (five-lobe star with an offset hole), are
meshed in one of two coordinate frames:

* ``geometry="cartesian"``: directly in the poloidal plane ``(x, y)``, B is
  ``(-1,1)^2`` and H is centred at the origin with the hole at ``(0.28, 0.10)``;
* ``geometry="axisymmetric"``: in ``(R, Z)`` with ``x = R - 3``, B is
  ``(2,4) x (-1,1)`` and H is centred at ``(3, 0)`` with the hole at
  ``(3.28, 0.10)``; every node must have ``R > 0``.
"""
from dataclasses import dataclass

import numpy as np

from hybridge.core.mesh import DGMesh, gmsh_rectangle_mesh, gmsh_smooth_star_mesh

BASELINE_SIZES = (0.20, 0.10, 0.05)
STRESS_POLYGONIZATIONS = ((80, 20), (160, 40), (320, 80))
GEOMETRIES = ("cartesian", "axisymmetric")
AXISYMMETRIC_SHIFT = 3.0  # x = R - 3


def frame_shift(geometry) -> float:
    """Return the first-coordinate offset ``first = x + shift`` of ``geometry``."""
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry must be one of {GEOMETRIES}; got {geometry!r}")
    return AXISYMMETRIC_SHIFT if geometry == "axisymmetric" else 0.0


@dataclass(frozen=True)
class CaseMesh:
    """Mesh together with serializable requested and actual geometry metadata."""

    mesh: DGMesh
    metadata: dict


def build_case_mesh(domain, h=0.20, *, geometry, outer_vertices=80, hole_vertices=20,
                    cache=True, cache_dir=None, log_cache=False, num_threads=None):
    """Mesh B or H in the ``geometry`` frame, with exact data on its straight boundaries.

    Vertex counts specify CAD polygon vertices; meshing may subdivide edges.
    ``h`` is the Gmsh target and ``metadata['actual_h']`` is the largest edge.
    """
    shift = frame_shift(geometry)
    h = float(h)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be finite and positive")
    options = dict(cache=cache, cache_dir=cache_dir, log_cache=log_cache,
                   num_threads=num_threads)
    if domain == "B":
        mesh = gmsh_rectangle_mesh(h, xlim=(shift - 1., shift + 1.), ylim=(-1., 1.), **options)
        polygonization = {"outer_vertices": 4, "hole_vertices": 0}
    elif domain == "H":
        star_center, hole_center = (shift, 0.), (shift + .28, .10)
        mesh = gmsh_smooth_star_mesh(
            h, center=star_center, radius=.70, amplitude=.224, mode=5,
            boundary_points=outer_vertices, hole_center=hole_center,
            hole_radius=.12, hole_boundary_points=hole_vertices, **options)
        polygonization = {"outer_vertices": int(outer_vertices),
                          "hole_vertices": int(hole_vertices),
                          "star_center": list(star_center), "star_radius": .70,
                          "star_amplitude": .224, "star_mode": 5,
                          "hole_center": list(hole_center), "hole_radius": .12}
    else:
        raise ValueError(f"unknown manufactured domain {domain!r}; expected B or H")
    if geometry == "axisymmetric" and not np.all(mesh.node_coords[:, 0] > 0):
        raise ValueError("axisymmetric manufactured meshes require R > 0")
    metadata = dict(domain=domain, geometry=geometry, requested_h=h, actual_h=float(mesh.h),
                    elements=mesh.num_tri, nodes=len(mesh.node_coords),
                    boundary_edges=len(mesh.bnd_edges_inds),
                    polygonal=True, **polygonization)
    return CaseMesh(mesh, metadata)
