"""Original polygonal cosine-star mesh for the standalone Newton experiment.

This preserves that experiment's geometry and Gmsh options independently of
HDGFEM. It deliberately differs from the canonical spline/sine-star geometry.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_sampled_star_mesh(
    mesh_size: float, *, boundary_points: int, radius: float, amplitude: float,
    mode: int, verbosity: int = 0, algorithm: int | None = None,
    write_path: str | Path, msh_file_version: float = 2.2,
) -> None:
    """Write the original first-order, OCC polygon mesh without DG objects."""
    import gmsh

    if mesh_size <= 0:
        raise ValueError("mesh_size must be positive")
    if boundary_points < max(8, 4 * mode):
        raise ValueError("boundary_points is too small for the requested star mode")
    if radius <= abs(amplitude):
        raise ValueError("radius must be larger than abs(amplitude)")
    theta = np.linspace(0.0, 2.0 * np.pi, boundary_points, endpoint=False)
    rr = radius + amplitude * np.cos(mode * theta)
    vertices = np.column_stack((rr * np.cos(theta), rr * np.sin(theta)))
    gmsh.initialize()
    try:
        gmsh.model.add("smooth_star")
        for key, value in {
            "General.Verbosity": verbosity,
            "Mesh.ElementOrder": 1,
            "Mesh.MeshSizeMin": mesh_size,
            "Mesh.MeshSizeMax": mesh_size,
            "Mesh.CharacteristicLengthMin": mesh_size,
            "Mesh.CharacteristicLengthMax": mesh_size,
            "Mesh.MshFileVersion": msh_file_version,
        }.items():
            gmsh.option.setNumber(key, value)
        if algorithm is not None:
            gmsh.option.setNumber("Mesh.Algorithm", algorithm)
        occ = gmsh.model.occ
        points = [occ.addPoint(float(x), float(y), 0.0, mesh_size) for x, y in vertices]
        lines = [occ.addLine(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
        surface = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        occ.synchronize()
        gmsh.model.addPhysicalGroup(1, lines, tag=1, name="smooth_star_boundary")
        gmsh.model.addPhysicalGroup(2, [surface], tag=1, name="smooth_star")
        gmsh.model.mesh.generate(2)
        gmsh.write(str(write_path))
    finally:
        gmsh.finalize()
