#!/usr/bin/env python3
"""Canonical geometries and deterministic Gmsh meshes for torsion studies.

The public helpers in this module deliberately avoid importing Gmsh at module
import time.  Study planning, alias validation, rank selection, and unit tests
therefore remain usable on machines that do not have the DOLFINx/Gmsh stack.
"""

from __future__ import annotations

# Support direct execution as well as python -m from a checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
ITER_GEO_PATH = REPO_ROOT / "projects/diocotron/freefem" / "msh" / "iter.geo"
MESH_METADATA_VERSION = 1


@dataclass(frozen=True)
class GeometryDefinition:
    slug: str
    title: str
    default_size: float
    parameters: dict[str, float | int | str]
    source_path: Path | None = None


GEOMETRIES: dict[str, GeometryDefinition] = {
    "disk": GeometryDefinition(
        "disk",
        "Disk",
        0.10,
        {"radius": 1.0},
    ),
    "ellipse": GeometryDefinition(
        "ellipse",
        "Ellipse",
        0.10,
        {"radius": 1.0, "ellipse_ratio": 0.7},
    ),
    "smooth_star": GeometryDefinition(
        "smooth_star",
        "Smooth sinusoidal star",
        0.075,
        {"boundary_points": 140, "radius": 1.5, "amplitude": 0.32, "mode": 5},
    ),
    "pacman": GeometryDefinition(
        "pacman",
        "Pacman",
        0.075,
        {"radius": 1.0, "mouth_half_angle": 0.38, "tip_x": -0.78},
    ),
    "horseshoe": GeometryDefinition(
        "horseshoe",
        "Horseshoe",
        0.075,
        {
            "gap_half_angle": 0.48,
            "outer_radius": 1.16,
            "inner_radius": 0.46,
            "outer_points": 170,
            "inner_points": 130,
            "cap_points": 28,
            "x_shift": 0.10,
            "y_scale": 0.92,
        },
    ),
    "iter": GeometryDefinition(
        "iter",
        "ITER wall",
        0.15,
        {"source": "projects/diocotron/freefem/msh/iter.geo"},
        ITER_GEO_PATH,
    ),
}

GEOMETRY_ALIASES = {
    "star": "smooth_star",
    "smooth-star": "smooth_star",
    "smoothstar": "smooth_star",
    "sinusoidal_star": "smooth_star",
    "pac-man": "pacman",
    "horse_shoe": "horseshoe",
    "iter_wall": "iter",
    "iter-wall": "iter",
}


def canonical_geometry_name(name: str) -> str:
    """Return the stable geometry slug for a user-facing name or alias."""
    normalized = str(name).strip().lower().replace(" ", "_")
    normalized = GEOMETRY_ALIASES.get(normalized, normalized)
    if normalized not in GEOMETRIES:
        choices = ", ".join(sorted(GEOMETRIES))
        raise ValueError(f"unknown geometry {name!r}; choose one of {choices}")
    return normalized


def geometry_definition(name: str) -> GeometryDefinition:
    return GEOMETRIES[canonical_geometry_name(name)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometry_source_hash(name: str, parameters: dict[str, Any] | None = None) -> str:
    """Hash the builder implementation, parameters, and external CAD source.

    Including this module's bytes makes a canonical mesh cache conservative:
    changing any Python geometry builder selects a new cache key even when its
    user-facing parameters happen to be unchanged.  ITER additionally hashes
    the imported ``.geo`` file because that file defines its boundary.
    """
    definition = geometry_definition(name)
    payload = {
        "version": MESH_METADATA_VERSION,
        "slug": definition.slug,
        "parameters": definition.parameters if parameters is None else parameters,
        "generator_source_hash": sha256_file(Path(__file__).resolve()),
        "source_hash": None,
    }
    if definition.source_path is not None:
        if not definition.source_path.is_file():
            raise FileNotFoundError(f"canonical ITER source not found: {definition.source_path}")
        payload["source_hash"] = sha256_file(definition.source_path)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _add_polygon(gmsh, points: np.ndarray, h: float) -> list[int]:
    point_tags = [gmsh.model.geo.addPoint(float(x), float(y), 0.0, h) for x, y in points]
    return [
        gmsh.model.geo.addLine(start, point_tags[(index + 1) % len(point_tags)])
        for index, start in enumerate(point_tags)
    ]


def _add_smooth_loop(gmsh, points: np.ndarray, h: float) -> int:
    point_tags = [gmsh.model.geo.addPoint(float(x), float(y), 0.0, h) for x, y in points]
    return gmsh.model.geo.addSpline([*point_tags, point_tags[0]])


def _build_smooth_star(gmsh, h: float, params: dict[str, Any]) -> tuple[int, list[int]]:
    count = int(params["boundary_points"])
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    radius = float(params["radius"]) + float(params["amplitude"]) * np.sin(
        int(params["mode"]) * theta
    )
    boundary = _add_smooth_loop(
        gmsh,
        np.column_stack((radius * np.cos(theta), radius * np.sin(theta))),
        h,
    )
    loop = gmsh.model.geo.addCurveLoop([boundary])
    return gmsh.model.geo.addPlaneSurface([loop]), [boundary]


def _build_pacman(gmsh, h: float, params: dict[str, Any]) -> tuple[int, list[int]]:
    radius = float(params["radius"])
    angle = float(params["mouth_half_angle"])
    center = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, h)
    upper = gmsh.model.geo.addPoint(radius * math.cos(angle), radius * math.sin(angle), 0.0, h)
    back = gmsh.model.geo.addPoint(-radius, 0.0, 0.0, h)
    lower = gmsh.model.geo.addPoint(radius * math.cos(angle), -radius * math.sin(angle), 0.0, h)
    tip = gmsh.model.geo.addPoint(float(params["tip_x"]), 0.0, 0.0, h)
    boundary = [
        gmsh.model.geo.addCircleArc(upper, center, back),
        gmsh.model.geo.addCircleArc(back, center, lower),
        gmsh.model.geo.addLine(lower, tip),
        gmsh.model.geo.addLine(tip, upper),
    ]
    loop = gmsh.model.geo.addCurveLoop(boundary)
    return gmsh.model.geo.addPlaneSurface([loop]), boundary


def _build_horseshoe(gmsh, h: float, params: dict[str, Any]) -> tuple[int, list[int]]:
    angle = float(params["gap_half_angle"])
    outer_radius = float(params["outer_radius"])
    inner_radius = float(params["inner_radius"])
    outer_theta = np.linspace(angle, 2.0 * np.pi - angle, int(params["outer_points"]))
    inner_theta = np.linspace(2.0 * np.pi - angle, angle, int(params["inner_points"]))
    lower_r = np.linspace(outer_radius, inner_radius, int(params["cap_points"]))
    upper_r = np.linspace(inner_radius, outer_radius, int(params["cap_points"]))
    outer = np.column_stack((outer_radius * np.cos(outer_theta), outer_radius * np.sin(outer_theta)))
    lower = np.column_stack((lower_r * np.cos(2.0 * np.pi - angle), lower_r * np.sin(2.0 * np.pi - angle)))
    inner = np.column_stack((inner_radius * np.cos(inner_theta), inner_radius * np.sin(inner_theta)))
    upper = np.column_stack((upper_r * np.cos(angle), upper_r * np.sin(angle)))
    points = np.vstack((outer, lower[1:], inner[1:], upper[1:-1]))
    points[:, 0] += float(params["x_shift"])
    points[:, 1] *= float(params["y_scale"])
    points = np.column_stack((points[:, 1], -points[:, 0]))
    boundary = _add_smooth_loop(gmsh, points, h)
    loop = gmsh.model.geo.addCurveLoop([boundary])
    return gmsh.model.geo.addPlaneSurface([loop]), [boundary]


def configure_gmsh(gmsh, *, h: float, verbosity: int, algorithm: int) -> None:
    """Set deterministic first-order triangular-mesh options."""
    gmsh.option.setNumber("General.Verbosity", int(verbosity))
    gmsh.option.setNumber("Mesh.Algorithm", int(algorithm))
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.45 * float(h))
    gmsh.option.setNumber("Mesh.MeshSizeMax", float(h))
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.RandomFactor", 0.0)
    gmsh.option.setNumber("Mesh.SaveAll", 0)


def build_gmsh_model(
    name: str,
    h: float,
    *,
    parameters: dict[str, Any] | None = None,
    verbosity: int = 0,
    algorithm: int = 6,
) -> dict[str, Any]:
    """Populate the active Gmsh model for one canonical geometry.

    Gmsh must already be initialized.  The returned dictionary is the exact
    parameter set used and can be embedded in mesh metadata.
    """
    import gmsh

    definition = geometry_definition(name)
    params = dict(definition.parameters)
    if parameters:
        params.update(parameters)
    configure_gmsh(gmsh, h=h, verbosity=verbosity, algorithm=algorithm)
    if definition.slug == "iter":
        source = definition.source_path
        if source is None or not source.is_file():
            raise FileNotFoundError(f"canonical ITER source not found: {source}")
        gmsh.open(str(source))
        configure_gmsh(gmsh, h=h, verbosity=verbosity, algorithm=algorithm)
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), float(h))
        return params

    gmsh.model.add(definition.slug)
    if definition.slug in {"disk", "ellipse"}:
        radius = float(params["radius"])
        ratio = float(params.get("ellipse_ratio", 1.0))
        surface = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, radius, radius * ratio)
        gmsh.model.occ.synchronize()
        boundary = gmsh.model.getBoundary([(2, surface)], oriented=False)
        gmsh.model.addPhysicalGroup(2, [surface], 1, "Omega")
        gmsh.model.addPhysicalGroup(1, [entity[1] for entity in boundary], 2, "Dirichlet")
        return params
    if definition.slug == "smooth_star":
        surface, boundary = _build_smooth_star(gmsh, h, params)
    elif definition.slug == "pacman":
        surface, boundary = _build_pacman(gmsh, h, params)
    elif definition.slug == "horseshoe":
        surface, boundary = _build_horseshoe(gmsh, h, params)
    else:  # pragma: no cover - protected by the registry
        raise ValueError(definition.slug)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [surface], 1, "Omega")
    gmsh.model.addPhysicalGroup(1, boundary, 2, "Dirichlet")
    return params


def triangle_mesh_statistics(points: np.ndarray, triangles: np.ndarray) -> dict[str, float | int]:
    """Return mesh/domain measurements used by calibration and provenance."""
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.size == 0:
        raise ValueError("expected a nonempty (n,3) triangle array")
    xy = points[:, :2]
    tri_xy = xy[triangles]
    edge01 = tri_xy[:, 1] - tri_xy[:, 0]
    edge12 = tri_xy[:, 2] - tri_xy[:, 1]
    edge20 = tri_xy[:, 0] - tri_xy[:, 2]
    lengths = np.concatenate(
        (np.linalg.norm(edge01, axis=1), np.linalg.norm(edge12, axis=1), np.linalg.norm(edge20, axis=1))
    )
    signed_twice_area = edge01[:, 0] * (tri_xy[:, 2, 1] - tri_xy[:, 0, 1]) - edge01[:, 1] * (
        tri_xy[:, 2, 0] - tri_xy[:, 0, 0]
    )
    areas = 0.5 * np.abs(signed_twice_area)
    if np.any(areas <= 0.0):
        raise ValueError("mesh contains a non-positive-area triangle")
    edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    edges.sort(axis=1)
    unique_edges = np.unique(edges, axis=0)
    diameter = float(np.linalg.norm(np.max(xy, axis=0) - np.min(xy, axis=0)))
    return {
        "vertices": int(points.shape[0]),
        "edges": int(unique_edges.shape[0]),
        "cells": int(triangles.shape[0]),
        "area": float(np.sum(areas)),
        "h_max": float(np.max(lengths)),
        "domain_diameter": diameter,
    }


def lagrange_dofs_from_metadata(metadata: dict[str, Any], order: int) -> int:
    """Exact scalar continuous-Lagrange DOF count on a triangular mesh."""
    p = int(order)
    if p < 1:
        raise ValueError("polynomial order must be positive")
    return int(
        int(metadata["vertices"])
        + (p - 1) * int(metadata["edges"])
        + ((p - 1) * (p - 2) // 2) * int(metadata["cells"])
    )


def generate_mesh(
    name: str,
    mesh_size: float,
    output_path: Path,
    *,
    metadata_path: Path | None = None,
    parameters: dict[str, Any] | None = None,
    verbosity: int = 0,
    algorithm: int = 6,
    optimize: bool = True,
    geometry_degree: int = 1,
) -> dict[str, Any]:
    """Generate tagged triangles, optionally with curved high-order geometry.

    ``geometry_degree`` controls the coordinate map, not the solution degree.
    The default retains the affine meshes required by equiband's production
    ray evaluator. Higher-order nodes are placed on the original CAD curves;
    the torsion-center audit can import these without linearizing the boundary.
    Reported mesh-size/area statistics use the corner triangles for comparison;
    the FE integral, not that chordal area, measures a curved domain's area.
    """
    import gmsh

    if isinstance(geometry_degree, bool) or not isinstance(geometry_degree, (int, np.integer)) or geometry_degree not in (1, 2, 3):
        raise ValueError("geometry_degree must be 1, 2 or 3 (supported DOLFINx Gmsh triangle import)")

    output_path = Path(output_path).expanduser().resolve()
    metadata_path = (
        output_path.with_suffix(output_path.suffix + ".json")
        if metadata_path is None
        else Path(metadata_path).expanduser().resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh_version = str(getattr(gmsh, "__version__", "unknown"))
        used_parameters = build_gmsh_model(
            name,
            mesh_size,
            parameters=parameters,
            verbosity=verbosity,
            algorithm=algorithm,
        )
        gmsh.model.mesh.generate(2)
        if optimize:
            gmsh.model.mesh.optimize("Netgen")
        if geometry_degree > 1:
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
            gmsh.model.mesh.setOrder(geometry_degree)
            gmsh.model.mesh.optimize("HighOrder")
        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        tag_to_index = {int(tag): index for index, tag in enumerate(node_tags)}
        points = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
        element_type = gmsh.model.mesh.getElementType("Triangle", geometry_degree)
        nodes_per_cell = gmsh.model.mesh.getElementProperties(element_type)[3]
        triangle_tags, triangle_nodes = gmsh.model.mesh.getElementsByType(element_type)
        if len(triangle_tags) == 0:
            raise RuntimeError(f"Gmsh produced no triangles for {name}")
        triangles = np.asarray(
            [tag_to_index[int(tag)] for tag in triangle_nodes], dtype=np.int64
        ).reshape(-1, nodes_per_cell)[:, :3]
        # Count only nodes referenced by triangle corners.  High-order geometry
        # has additional coordinate nodes, and spline CAD models can retain
        # unused control points even for first-order meshes.  Including either
        # group would overpredict continuous-Lagrange FE degrees of freedom.
        corners, compact = np.unique(triangles, return_inverse=True)
        stats = triangle_mesh_statistics(points[corners], compact.reshape(-1, 3))
        gmsh.write(str(output_path))
    finally:
        gmsh.finalize()

    definition = geometry_definition(name)
    metadata: dict[str, Any] = {
        "format": "hybridge_canonical_gmsh_v1",
        "geometry": definition.slug,
        "title": definition.title,
        "geometry_parameters": used_parameters,
        "geometry_source": None if definition.source_path is None else str(definition.source_path.relative_to(REPO_ROOT)),
        "source_hash": geometry_source_hash(definition.slug, used_parameters),
        "requested_size": float(mesh_size),
        "gmsh_algorithm": int(algorithm),
        "gmsh_version": gmsh_version,
        "msh_file_version": 2.2,
        "geometry_degree": geometry_degree,
        "statistics_geometry": "corner_triangles",
        **stats,
    }
    metadata["mesh_sha256"] = sha256_file(output_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def _parse_cli() -> tuple[str, float, Path, Path | None, int, int, dict[str, Any] | None, int]:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry", choices=sorted({*GEOMETRIES, *GEOMETRY_ALIASES}))
    parser.add_argument("--mesh-size", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--algorithm", type=int, default=6)
    parser.add_argument("--parameters-json", type=json.loads, default=None)
    parser.add_argument("--geometry-degree", type=int, choices=(1, 2, 3), default=1,
                        help="coordinate-map degree; >1 places curved high-order nodes on the CAD boundary")
    args = parser.parse_args()
    return (
        args.geometry,
        args.mesh_size,
        args.output,
        args.metadata,
        args.verbosity,
        args.algorithm,
        args.parameters_json,
        args.geometry_degree,
    )


def main() -> int:
    name, h, output, metadata, verbosity, algorithm, parameters, geometry_degree = _parse_cli()
    result = generate_mesh(
        name,
        h,
        output,
        metadata_path=metadata,
        parameters=parameters,
        verbosity=verbosity,
        algorithm=algorithm,
        geometry_degree=geometry_degree,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
