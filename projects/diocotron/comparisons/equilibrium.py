#!/usr/bin/env python3
"""Compare two optimizer checkpoints on reference-mesh quadrature points."""

from __future__ import annotations

# Support direct execution as well as python -m from a checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from projects.diocotron.paths import resolve_archive_path


def _checkpoint(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(resolve_archive_path(path), allow_pickle=False) as data:
        return (
            np.asarray(data["coordinates"], dtype=np.float64).copy(),
            np.asarray(data["phi"], dtype=np.float64).copy(),
            np.asarray(data["rho"], dtype=np.float64).copy(),
            json.loads(str(data["metadata"].item())),
        )


def _mesh_from_checkpoint(metadata: dict[str, Any]):
    from mpi4py import MPI
    from dolfinx.io import gmsh as gmshio

    mesh_path = resolve_archive_path(metadata["mesh_path"])
    if not mesh_path.is_file():
        raise FileNotFoundError(f"checkpoint mesh is missing: {mesh_path}")
    mesh_data = gmshio.read_from_msh(mesh_path, MPI.COMM_SELF, rank=0, gdim=2)
    return mesh_data.mesh if hasattr(mesh_data, "mesh") else mesh_data[0]


def _load_fields(path: Path):
    from dolfinx import fem
    from scipy.spatial import cKDTree

    coordinates, phi_values, rho_values, metadata = _checkpoint(path)
    domain = _mesh_from_checkpoint(metadata)
    space = fem.functionspace(domain, ("Lagrange", int(metadata["order"])))
    local_coordinates = np.asarray(space.tabulate_dof_coordinates()[:, :2], dtype=np.float64)
    distances, indices = cKDTree(coordinates[:, :2]).query(local_coordinates, k=1)
    if float(np.max(distances, initial=0.0)) > 1.0e-10:
        raise ValueError(f"checkpoint coordinate mismatch for {path}: {np.max(distances):.3e}")
    phi = fem.Function(space, name="phi")
    rho = fem.Function(space, name="rho")
    phi.x.array[:] = phi_values[indices]
    rho.x.array[:] = rho_values[indices]
    phi.x.scatter_forward()
    rho.x.scatter_forward()
    return domain, phi, rho, metadata


def _gradient_function(function):
    from dolfinx import fem
    import ufl

    degree = int(function.function_space.element.basix_element.degree)
    value_size = int(function.function_space.mesh.geometry.dim)
    space = fem.functionspace(
        function.function_space.mesh,
        ("DG", max(degree - 1, 0), (value_size,)),
    )
    result = fem.Function(space)
    points = space.element.interpolation_points
    if callable(points):
        points = points()
    result.interpolate(fem.Expression(ufl.grad(function), points))
    result.x.scatter_forward()
    return result


def _locate_cells(domain, points_xy: np.ndarray, *, tree=None) -> tuple[np.ndarray, np.ndarray]:
    from dolfinx import geometry

    points = np.zeros((len(points_xy), 3), dtype=np.float64)
    points[:, :2] = points_xy
    if tree is None:
        tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    collisions = geometry.compute_colliding_cells(domain, candidates, points)
    cells = np.full(len(points), -1, dtype=np.int32)
    for index in range(len(points)):
        links = collisions.links(index)
        if len(links):
            cells[index] = int(links[0])
    return points, cells


def _evaluate_in_cells(function, points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    valid = cells >= 0
    values = np.full((len(points), function.function_space.value_size), np.nan)
    if np.any(valid):
        raw = function.eval(points[valid], cells[valid])
        values[valid] = np.asarray(raw, dtype=np.float64).reshape(
            -1, function.function_space.value_size
        )
    return values


def _evaluate(function, points_xy: np.ndarray, *, tree=None) -> tuple[np.ndarray, np.ndarray]:
    points, cells = _locate_cells(function.function_space.mesh, points_xy, tree=tree)
    return _evaluate_in_cells(function, points, cells), cells


def _reference_samples(domain, expressions, cells, qweights: np.ndarray):
    """Evaluate one bounded cell chunk of reference quadrature samples."""
    coordinate_expression, phi_expression, rho_expression, gradient_expression = expressions
    qcount = len(qweights)
    points = np.asarray(coordinate_expression.eval(domain, cells), dtype=np.float64)[..., :2]
    phi_values = np.asarray(phi_expression.eval(domain, cells), dtype=np.float64).reshape(
        len(cells), qcount
    )
    rho_values = np.asarray(rho_expression.eval(domain, cells), dtype=np.float64).reshape(
        len(cells), qcount
    )
    phi_grad = np.asarray(
        gradient_expression.eval(domain, cells), dtype=np.float64
    ).reshape(len(cells), qcount, -1)[..., :2]
    geometry_dofs = np.asarray(domain.geometry.dofmaps[0][cells], dtype=np.int32)
    vertices = np.asarray(domain.geometry.x[geometry_dofs[:, :3], :2], dtype=np.float64)
    jacobian0 = vertices[:, 1] - vertices[:, 0]
    jacobian1 = vertices[:, 2] - vertices[:, 0]
    determinants = np.abs(jacobian0[:, 0] * jacobian1[:, 1] - jacobian0[:, 1] * jacobian1[:, 0])
    weights = determinants[:, None] * np.asarray(qweights, dtype=np.float64)[None, :]
    edge_lengths = np.stack(
        (
            np.linalg.norm(vertices[:, 1] - vertices[:, 0], axis=1),
            np.linalg.norm(vertices[:, 2] - vertices[:, 1], axis=1),
            np.linalg.norm(vertices[:, 0] - vertices[:, 2], axis=1),
        ),
        axis=1,
    )
    h_values = np.max(edge_lengths, axis=1)[:, None] * np.ones((1, qcount))
    return (
        points.reshape(-1, 2),
        weights.reshape(-1),
        h_values.reshape(-1),
        phi_values.reshape(-1),
        rho_values.reshape(-1),
        phi_grad.reshape(-1, 2),
    )


def _contour_points(points: np.ndarray, triangles: np.ndarray, values: np.ndarray, levels: tuple[float, float]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    fig, ax = plt.subplots()
    contour = ax.tricontour(triangulation, values, levels=list(levels))
    output: list[np.ndarray] = []
    for segments in contour.allsegs:
        usable = [np.asarray(segment, dtype=np.float64) for segment in segments if len(segment)]
        output.append(np.vstack(usable) if usable else np.empty((0, 2), dtype=np.float64))
    plt.close(fig)
    return output


def compare_equilibria(
    reference_checkpoint: Path,
    current_checkpoint: Path,
    *,
    quadrature_degree: int | None = None,
    kappa: float = 2.0,
    cell_chunk_size: int = 1024,
) -> dict[str, float]:
    """Return field, set, contour, and transition metrics with bounded memory."""
    import basix
    import basix.ufl
    from dolfinx import fem
    from mpi4py import MPI
    import ufl
    from scipy.spatial import ConvexHull, distance
    try:
        from projects.diocotron.studies.torsion_optimizer.metrics import normalized_hausdorff
    except ImportError:
        from projects.diocotron.studies.torsion_optimizer.metrics import normalized_hausdorff  # type: ignore

    reference_checkpoint = Path(reference_checkpoint).resolve()
    current_checkpoint = Path(current_checkpoint).resolve()
    comm = MPI.COMM_WORLD
    ref_domain, ref_phi, ref_rho, ref_meta = _load_fields(reference_checkpoint)
    if reference_checkpoint == current_checkpoint:
        cur_phi, cur_rho, cur_meta = ref_phi, ref_rho, ref_meta
    else:
        _, cur_phi, cur_rho, cur_meta = _load_fields(current_checkpoint)
    cur_gradient_function = _gradient_function(cur_phi)
    degree = quadrature_degree or max(2 * int(ref_meta["order"]) + 8, 12)
    qpoints, qweights = basix.make_quadrature(basix.CellType.triangle, int(degree))
    expressions = (
        fem.Expression(ufl.SpatialCoordinate(ref_domain), qpoints),
        fem.Expression(ref_phi, qpoints),
        fem.Expression(ref_rho, qpoints),
        fem.Expression(ufl.grad(ref_phi), qpoints),
    )
    ref_c1, ref_c2, ref_eps = (
        float(ref_meta[key]) for key in ("c1_phi", "c2_phi", "eps_phi")
    )
    cur_c1, cur_c2, cur_eps = (
        float(cur_meta[key]) for key in ("c1_phi", "c2_phi", "eps_phi")
    )
    cell_count = int(
        ref_domain.topology.index_map(ref_domain.topology.dim).size_local
    )
    owned_chunks = [
        np.arange(start, min(start + int(cell_chunk_size), cell_count), dtype=np.int32)
        for chunk_index, start in enumerate(range(0, cell_count, int(cell_chunk_size)))
        if chunk_index % comm.size == comm.rank
    ]
    owned_cells = (
        np.concatenate(owned_chunks)
        if owned_chunks
        else np.empty(0, dtype=np.int32)
    )

    scalar_quadrature_element = basix.ufl.quadrature_element(
        ref_domain.basix_cell(), points=qpoints, weights=qweights
    )
    scalar_quadrature_space = fem.functionspace(
        ref_domain, scalar_quadrature_element
    )
    current_phi_quadrature = fem.Function(scalar_quadrature_space)
    current_rho_quadrature = fem.Function(scalar_quadrature_space)
    if len(owned_cells):
        scalar_interpolation_data = fem.create_interpolation_data(
            scalar_quadrature_space,
            cur_phi.function_space,
            owned_cells,
            padding=1.0e-10,
        )
        current_phi_quadrature.interpolate_nonmatching(
            cur_phi, owned_cells, scalar_interpolation_data
        )
        current_rho_quadrature.interpolate_nonmatching(
            cur_rho, owned_cells, scalar_interpolation_data
        )

    gradient_quadrature_element = basix.ufl.quadrature_element(
        ref_domain.basix_cell(), value_shape=(2,), points=qpoints, weights=qweights
    )
    gradient_quadrature_space = fem.functionspace(
        ref_domain, gradient_quadrature_element
    )
    current_gradient_quadrature = fem.Function(gradient_quadrature_space)
    if len(owned_cells):
        gradient_interpolation_data = fem.create_interpolation_data(
            gradient_quadrature_space,
            cur_gradient_function.function_space,
            owned_cells,
            padding=1.0e-10,
        )
        current_gradient_quadrature.interpolate_nonmatching(
            cur_gradient_function, owned_cells, gradient_interpolation_data
        )
    current_gradient_array = current_gradient_quadrature.x.array.reshape(-1, 2)

    totals = {
        "domain_area": 0.0,
        "phi_difference_sq": 0.0,
        "phi_reference_sq": 0.0,
        "rho_difference_sq": 0.0,
        "rho_reference_sq": 0.0,
        "gradient_difference_sq": 0.0,
        "gradient_reference_sq": 0.0,
        "active_symmetric": 0.0,
        "active_intersection": 0.0,
        "active_union": 0.0,
        "certified_symmetric": 0.0,
        "certified_intersection": 0.0,
        "certified_union": 0.0,
    }
    transition_parts: list[np.ndarray] = []
    for cells in owned_chunks:
        points, weights, h_values, ref_phi_values, ref_rho_values, ref_grad = (
            _reference_samples(ref_domain, expressions, cells, qweights)
        )
        scalar_dofs = np.vstack([
            scalar_quadrature_space.dofmap.cell_dofs(int(cell)) for cell in cells
        ])
        gradient_dofs = np.vstack([
            gradient_quadrature_space.dofmap.cell_dofs(int(cell)) for cell in cells
        ])
        cur_phi_values = current_phi_quadrature.x.array[scalar_dofs].reshape(-1)
        cur_rho_values = current_rho_quadrature.x.array[scalar_dofs].reshape(-1)
        cur_gradient = current_gradient_array[gradient_dofs].reshape(-1, 2)
        missing_mask = (
            ~np.isfinite(cur_phi_values)
            | ~np.isfinite(cur_rho_values)
            | ~np.all(np.isfinite(cur_gradient), axis=1)
        )
        if np.any(missing_mask):
            raise RuntimeError(
                f"current mesh did not contain {np.count_nonzero(missing_mask)} "
                "reference quadrature points"
            )
        totals["domain_area"] += float(np.sum(weights))
        totals["phi_difference_sq"] += float(
            np.sum(weights * (cur_phi_values - ref_phi_values) ** 2)
        )
        totals["phi_reference_sq"] += float(np.sum(weights * ref_phi_values**2))
        totals["rho_difference_sq"] += float(
            np.sum(weights * (cur_rho_values - ref_rho_values) ** 2)
        )
        totals["rho_reference_sq"] += float(np.sum(weights * ref_rho_values**2))
        totals["gradient_difference_sq"] += float(
            np.sum(weights * np.sum((cur_gradient - ref_grad) ** 2, axis=1))
        )
        totals["gradient_reference_sq"] += float(
            np.sum(weights * np.sum(ref_grad**2, axis=1))
        )

        ref_active = (ref_phi_values >= ref_c1) & (ref_phi_values <= ref_c2)
        cur_active = (cur_phi_values >= cur_c1) & (cur_phi_values <= cur_c2)
        ref_certified = (
            (ref_phi_values >= ref_c1 + kappa * ref_eps)
            & (ref_phi_values <= ref_c2 - kappa * ref_eps)
        )
        cur_certified = (
            (cur_phi_values >= cur_c1 + kappa * cur_eps)
            & (cur_phi_values <= cur_c2 - kappa * cur_eps)
        )
        for prefix, reference_mask, current_mask in (
            ("active", ref_active, cur_active),
            ("certified", ref_certified, cur_certified),
        ):
            totals[f"{prefix}_symmetric"] += float(
                np.sum(weights[np.logical_xor(reference_mask, current_mask)])
            )
            totals[f"{prefix}_intersection"] += float(
                np.sum(weights[np.logical_and(reference_mask, current_mask)])
            )
            totals[f"{prefix}_union"] += float(
                np.sum(weights[np.logical_or(reference_mask, current_mask)])
            )
        transition_mask = (
            (np.abs(cur_phi_values - cur_c1) <= cur_eps)
            | (np.abs(cur_phi_values - cur_c2) <= cur_eps)
        )
        gradient_magnitude = np.linalg.norm(cur_gradient, axis=1)
        valid_transition = transition_mask & (gradient_magnitude > 1.0e-14)
        if np.any(valid_transition):
            transition_parts.append(
                cur_eps
                / (
                    h_values[valid_transition]
                    * gradient_magnitude[valid_transition]
                )
            )

    totals = {
        key: comm.allreduce(value, op=MPI.SUM) for key, value in totals.items()
    }
    local_resolution = (
        np.concatenate(transition_parts)
        if transition_parts
        else np.asarray([], dtype=np.float64)
    )
    gathered_resolution = comm.gather(local_resolution, root=0)
    contour_distances: list[float] = []
    resolution_metrics: tuple[float, float, float]
    if comm.rank == 0:
        import meshio
        from scipy.spatial import cKDTree

        mesh_data = meshio.read(Path(ref_meta["mesh_path"]))
        triangle_blocks = [
            block.data for block in mesh_data.cells if block.type == "triangle"
        ]
        base_points = np.asarray(mesh_data.points[:, :2], dtype=np.float64)
        base_triangles = np.vstack(triangle_blocks).astype(np.int64)
        current_mesh_data = meshio.read(Path(cur_meta["mesh_path"]))
        current_triangle_blocks = [
            block.data for block in current_mesh_data.cells if block.type == "triangle"
        ]
        current_base_points = np.asarray(
            current_mesh_data.points[:, :2], dtype=np.float64
        )
        current_base_triangles = np.vstack(current_triangle_blocks).astype(np.int64)

        ref_coordinates, ref_checkpoint_phi, _, _ = _checkpoint(reference_checkpoint)
        cur_coordinates, cur_checkpoint_phi, _, _ = _checkpoint(current_checkpoint)
        ref_distances, ref_indices = cKDTree(ref_coordinates[:, :2]).query(base_points)
        cur_distances, cur_indices = cKDTree(cur_coordinates[:, :2]).query(
            current_base_points
        )
        if (
            float(np.max(ref_distances, initial=0.0)) > 1.0e-10
            or float(np.max(cur_distances, initial=0.0)) > 1.0e-10
        ):
            raise RuntimeError("checkpoint coordinates did not contain their mesh vertices")
        ref_vertex_values = ref_checkpoint_phi[ref_indices]
        cur_vertex_values = cur_checkpoint_phi[cur_indices]
        ref_contours = _contour_points(
            base_points, base_triangles, ref_vertex_values, (ref_c1, ref_c2)
        )
        cur_contours = _contour_points(
            current_base_points,
            current_base_triangles,
            cur_vertex_values,
            (cur_c1, cur_c2),
        )
        hull_points = base_points[ConvexHull(base_points).vertices]
        diameter = float(np.max(distance.pdist(hull_points)))
        contour_distances = [
            normalized_hausdorff(reference, current, diameter)
            for reference, current in zip(ref_contours, cur_contours, strict=True)
            if len(reference) and len(current)
        ]
        resolution = np.concatenate(
            [part for part in gathered_resolution if len(part)]
        ) if any(len(part) for part in gathered_resolution) else np.asarray([], dtype=np.float64)
        resolution_metrics = (
            float(np.min(resolution)) if len(resolution) else math.nan,
            float(np.quantile(resolution, 0.1)) if len(resolution) else math.nan,
            float(np.median(resolution)) if len(resolution) else math.nan,
        )
    else:
        resolution_metrics = (math.nan, math.nan, math.nan)
    contour_distances = comm.bcast(contour_distances, root=0)
    resolution_metrics = comm.bcast(resolution_metrics, root=0)
    phi_l2 = math.sqrt(totals["phi_difference_sq"]) / max(
        math.sqrt(totals["phi_reference_sq"]), 1.0e-300
    )
    phi_h1 = math.sqrt(totals["gradient_difference_sq"]) / max(
        math.sqrt(totals["gradient_reference_sq"]), 1.0e-300
    )
    rho_l2 = math.sqrt(totals["rho_difference_sq"]) / max(
        math.sqrt(totals["rho_reference_sq"]), 1.0e-300
    )
    return {
        "relative_l2": phi_l2,
        "relative_h1": phi_h1,
        "phi_relative_l2": phi_l2,
        "phi_relative_h1": phi_h1,
        "rho_relative_l2": rho_l2,
        "active_symmetric_difference": totals["active_symmetric"]
        / max(totals["domain_area"], 1.0e-300),
        "active_jaccard": totals["active_intersection"]
        / max(totals["active_union"], 1.0e-300),
        "certified_symmetric_difference": totals["certified_symmetric"]
        / max(totals["domain_area"], 1.0e-300),
        "certified_jaccard": totals["certified_intersection"]
        / max(totals["certified_union"], 1.0e-300),
        "normalized_contour_hausdorff": max(contour_distances)
        if contour_distances
        else math.nan,
        "transition_resolution_min": resolution_metrics[0],
        "transition_resolution_p10": resolution_metrics[1],
        "transition_resolution_median": resolution_metrics[2],
        "epsilon_ratio_observed": cur_eps / (cur_c2 - cur_c1),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--quadrature-degree", type=int)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare_equilibria(
        args.reference,
        args.current,
        quadrature_degree=args.quadrature_degree,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    from mpi4py import MPI
    if MPI.COMM_WORLD.rank == 0:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
