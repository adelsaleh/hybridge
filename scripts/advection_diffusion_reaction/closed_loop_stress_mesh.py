"""Count-matched annular or square meshes for the ADR stress runner (explicit use only)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from scripts.advection_diffusion_reaction.closed_loop_stress_cases import VARIANTS, unscaled_velocity


def background_sizes(parameters, bulk_size, neck_elements):
    """Resolve narrow angular sectors while leaving lobe interiors coarser."""
    extent = 1.36
    count = int(np.ceil(2*extent/(parameters.neck_width/4)))+1
    axis = np.linspace(-extent, extent, count)
    phi = np.arctan2(axis[None, :], axis[:, None])
    gap = 1+0.35*np.cos(9*phi)-parameters.hole_radius
    sizes = np.minimum(bulk_size, gap/neck_elements)
    return (-extent, -extent), (axis[1]-axis[0],)*2, sizes


def mesh_diagnostics(mesh, parameters):
    """Report mesh shape and topology, plus geometry-specific boundary diagnostics."""
    from hdgfem.core.mesh import mesh_edge_min_max

    vertices = mesh.node_coords[mesh.triangles]
    lengths = np.linalg.norm(vertices-np.roll(vertices, 1, axis=1), axis=2)
    centres = vertices.mean(axis=1)
    square = parameters.geometry == "square"
    neck_ratio = None
    if not square:
        phi = np.arctan2(centres[:, 1], centres[:, 0])
        gap = 1+0.35*np.cos(9*phi)-parameters.hole_radius
        near = gap <= 1.5*parameters.neck_width
        neck_ratio = float(np.min(gap[near]/lengths[near].max(axis=1))) if np.any(near) else 0.0
    boundary = mesh.edges[mesh.bnd_edges_inds]
    neighbors = {}
    for first, second in boundary:
        neighbors.setdefault(int(first), []).append(int(second))
        neighbors.setdefault(int(second), []).append(int(first))
    if any(len(values) != 2 for values in neighbors.values()):
        raise ValueError("Boundary edges do not form closed manifold loops")
    remaining, components = set(neighbors), 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            for node in neighbors[stack.pop()]:
                if node in remaining:
                    remaining.remove(node)
                    stack.append(node)
    expected_components = 1 if square else 2
    if components != expected_components:
        raise ValueError(f"Expected {expected_components} boundary components, got {components}")
    mid = mesh.node_coords[boundary].mean(axis=1)
    if square:
        wall_error = np.min(abs(abs(mid)-1), axis=1)
        if np.any(abs(mesh.node_coords) > 1+1e-12) or np.max(wall_error) > 1e-12:
            raise ValueError("Mesh does not follow the square [-1,1]^2")
    else:
        radius = np.linalg.norm(mid, axis=1)
        theta = np.arctan2(mid[:, 1], mid[:, 0])
        wall_error = np.minimum(abs(radius-parameters.hole_radius), abs(radius-1-0.35*np.cos(9*theta)))
    wall = mesh.node_coords[boundary]
    tangent = wall[:, 1]-wall[:, 0]
    normal = np.column_stack((tangent[:, 1], -tangent[:, 0]))/np.linalg.norm(tangent, axis=1)[:, None]
    fraction = np.linspace(0, 1, 5)
    samples = wall[:, 0, None, :]+fraction[None, :, None]*tangent[:, None, :]
    wall_transport = {}
    for variant in VARIANTS:
        vx, vy = unscaled_velocity(samples[..., 0], samples[..., 1], replace(parameters, variant=variant))
        wall_transport[variant] = float(np.max(abs(vx*normal[:, 0, None]+vy*normal[:, 1, None])))
    a, b = vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0]
    twice_area = abs(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])
    quality = 2*np.sqrt(3)*twice_area/(lengths**2).sum(axis=1)
    if np.any(quality <= 0) or not np.all(np.isfinite(quality)):
        raise ValueError("Mesh contains degenerate elements")
    return dict(geometry=parameters.geometry, triangles=int(mesh.num_tri), vertices=len(mesh.node_coords),
                boundary_components=components, edge_lengths=mesh_edge_min_max(mesh),
                minimum_shape_quality=float(quality.min()),
                minimum_neck_gap_over_element_diameter=neck_ratio,
                neck_size_screen_passed=None if square else neck_ratio >= 6,
                **{("boundary_midpoint_distance_max" if square else "boundary_midpoint_radial_error_max"): float(wall_error.max())},
                sampled_unscaled_wall_normal_speed_max=wall_transport,
                note=("Straight square walls; exact Dirichlet data on every edge. No neck screen applies."
                      if square else "Affine polygonal walls; exact Dirichlet data on every edge. "
                      "Neck size is a screen, not an h/p or quadrature convergence certificate."))


def prepare_mesh(parameters, target, directory, *, neck_elements=8, boundary_points=1800,
                 max_triangles=100000, count_rtol=0.03, max_attempts=14):
    """Reuse package mesh generators; preserve annular neck sizing when applicable."""
    if target < 1 or target > max_triangles:
        raise ValueError("Triangle target must be positive and within the selected-mesh ceiling")
    square = parameters.geometry == "square"
    settings = (dict(target_triangles=target, geometry="square", xlim=[-1.0, 1.0], ylim=[-1.0, 1.0])
                if square else dict(target_triangles=target, hole_radius=parameters.hole_radius,
                                    neck_elements=neck_elements, outer_boundary_points=boundary_points))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory/"mesh.json"
    if manifest.exists():
        record = json.loads(manifest.read_text())
        if (record.get("geometry", "annulus") != parameters.geometry or
                any(record.get(key) != value for key, value in settings.items())):
            raise ValueError(f"Changed mesh preparation settings: {manifest}")
        if record["triangles"] > max_triangles:
            raise ValueError(f"Cached mesh exceeds --max-triangles: {manifest}")
        path = directory/record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Changed mesh: {path}")
        return path, record
    from hdgfem.core.mesh import gmsh_smooth_star_mesh_with_background_sizes, gmsh_rectangle_mesh
    search_target = min(target, 0.99*max_triangles)
    minimum_size = 0.0 if square else parameters.neck_width/neck_elements
    bulk_size = max(1.01*minimum_size, np.sqrt((9.24 if square else 9)/search_target))
    trialpath = directory/"trials.json"
    trials = json.loads(trialpath.read_text()) if trialpath.exists() else []
    fine, coarse = None, None
    for record in trials:
        if (record.get("geometry", "annulus") != parameters.geometry or
                any(record.get(key) != value for key, value in settings.items())):
            raise ValueError(f"Changed trial mesh settings: {trialpath}")
        if abs(record["triangles"]/target-1) <= count_rtol and record["triangles"] <= max_triangles:
            path = directory/record["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Changed trial mesh: {path}")
            temporary = directory/"mesh.json.tmp"
            temporary.write_text(json.dumps(record, indent=2)+"\n")
            temporary.replace(manifest)
            return path, record
        if record["triangles"] > search_target:
            fine = record["bulk_size"]
        else:
            coarse = record["bulk_size"]
        bulk_size = ((fine+coarse)/2 if fine and coarse else
                     record["bulk_size"]*np.sqrt(record["triangles"]/search_target))
    bulk_size = max(minimum_size, float(bulk_size))
    for attempt in range(len(trials), max_attempts):
        if square:
            mesh = gmsh_rectangle_mesh(bulk_size, xlim=(-1, 1), ylim=(-1, 1),
                                       cache=False, log_cache=False, num_threads=1, verbosity=0)
        else:
            origin, spacing, sizes = background_sizes(parameters, bulk_size, neck_elements)
            mesh = gmsh_smooth_star_mesh_with_background_sizes(
                boundary_points=boundary_points, radius=1, amplitude=0.35, mode=9,
                hole_radius=parameters.hole_radius, hmin=float(sizes.min()), hmax=bulk_size,
                background_origin=origin, background_spacing=spacing, background_values=sizes,
                num_threads=1, verbosity=0)
        record = dict(mesh_diagnostics(mesh, parameters), attempt=attempt, bulk_size=bulk_size)
        record.update(settings)
        if not square:
            record["nominal_neck_width"] = parameters.neck_width
        path = directory/f"trial_{attempt:02d}.npz"
        if path.exists():
            # Preserve a mesh written just before an interrupted metadata write.
            index = 0
            saved = path.with_suffix(f".interrupted{index}.npz")
            while saved.exists():
                index += 1
                saved = path.with_suffix(f".interrupted{index}.npz")
            path.rename(saved)
        np.savez(path, node_coords=mesh.node_coords, triangles=mesh.triangles)
        record.update(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        trials.append(record)
        temporary = directory/"trials.json.tmp"
        temporary.write_text(json.dumps(trials, indent=2)+"\n")
        temporary.replace(trialpath)
        print(f"mesh target={target} attempt={attempt} triangles={mesh.num_tri}", flush=True)
        relative = abs(mesh.num_tri/target-1)
        if relative <= count_rtol and mesh.num_tri <= max_triangles:
            temporary = directory/"mesh.json.tmp"
            temporary.write_text(json.dumps(record, indent=2)+"\n")
            temporary.replace(manifest)
            return path, record
        if mesh.num_tri > search_target:
            fine = bulk_size
        else:
            coarse = bulk_size
        next_size = (fine+coarse)/2 if fine and coarse else bulk_size*np.sqrt(mesh.num_tri/search_target)
        bulk_size = max(minimum_size, float(next_size))
    raise RuntimeError(f"Could not match {target} triangles within {count_rtol:.1%}; "
                       "Trials are retained; annular neck refinement, if applicable, was preserved.")
