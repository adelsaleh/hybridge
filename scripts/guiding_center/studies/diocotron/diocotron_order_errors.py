"""Cross-mesh L2 errors of saved diocotron fields against a reference run (D8); no solves.

The error of a run field u (mesh T, degree p) against the reference field u_ref (mesh T_ref, degree
p_ref) at a saved time is

    e_abs = ||u - u_ref||_{L2(Omega_ref)},   e_rel = e_abs / ||u_ref||_{L2(Omega_ref)},

integrated with the reference space's volume quadrature, which is exact for u_ref^2 on every reference
element. The run field is evaluated at those points (point location in T with Matplotlib's TriFinder);
where the run's element boundaries cross a reference element the cross term is integrated only
approximately, which is negligible when T_ref is much finer than T (D8: references at L4, runs at
L0..L3). Reference quadrature points outside the run's polygonal domain (both walls are polygons of
different resolution) take the polynomial of the run element with the nearest centroid; the fields are
negligible there (rings at r <= 0.8, phi = 0 on the wall).

Point location and the run basis at the reference points are computed once per (run, reference) pair,
in chunks of reference elements, and serve every saved time and field in one pass.

Snapshot files are the ``*_snapshots.npz`` of ``run_diocotron_benchmark.py`` (``times``, ``density``
(n_t, K, n), ``potential`` (n_t, K, n), ``flux`` (n_t, 2, K, n), ``node_coords``, ``triangles``,
``order``, ``basis``). Times are matched to within half a step.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FIELDS = ("density", "potential", "flux")


def _space(npz, *, volume_quad_1d=None):
    from hybridge.core.mesh import DGMesh
    from hybridge.core.space import DGSpace

    mesh = DGMesh.from_arrays(np.asarray(npz["node_coords"]), np.asarray(npz["triangles"]))
    return DGSpace(mesh, int(npz["order"]), basis_type=str(npz["basis"]), volume_quad_1d=volume_quad_1d,
                   cache=False)


def _time_index(times, wanted, tolerance):
    index = int(np.argmin(np.abs(np.asarray(times) - wanted)))
    if abs(float(times[index]) - wanted) > tolerance:
        raise ValueError(f"no saved state within {tolerance:g} of t = {wanted:g} (saved: {list(np.round(times, 6))})")
    return index


def _stack(npz, field, index):
    """Coefficients of one saved state as (components, K, n)."""
    data = np.asarray(npz[field][index])
    return data[None] if field != "flux" else data


def cross_mesh_errors(run_path, reference_path, times, *, fields=FIELDS, tolerance=1e-6, chunk_elements=20000,
                      reference_quad_1d=None) -> list[dict]:
    """Relative and absolute L2 errors of every field at every time of the run against the reference."""
    from matplotlib.tri import Triangulation
    from scipy.spatial import cKDTree

    with np.load(run_path) as run_npz, np.load(reference_path) as reference_npz:
        run_space = _space(run_npz)
        reference_space = _space(reference_npz, volume_quad_1d=reference_quad_1d)
        fields = [field for field in fields if field in run_npz.files and field in reference_npz.files]
        selected = []
        for wanted in times:
            selected.append((float(wanted), _time_index(run_npz["times"], wanted, tolerance),
                             _time_index(reference_npz["times"], wanted, tolerance)))
        run_states = {(field, i): _stack(run_npz, field, i) for field in fields for _, i, _ in selected}
        reference_states = {(field, j): _stack(reference_npz, field, j) for field in fields for _, _, j in selected}
    run_mesh, reference_mesh = run_space.mesh, reference_space.mesh
    nodes = np.asarray(run_mesh.node_coords)
    finder = Triangulation(nodes[:, 0], nodes[:, 1], np.asarray(run_mesh.triangles)).get_trifinder()
    centroids = nodes[np.asarray(run_mesh.triangles)].mean(axis=1)
    nearest = cKDTree(centroids)
    quad = reference_space.quad_data
    reference_basis = np.asarray(quad.bas_of_quads).T  # (q, n_ref)
    weights_ref = np.asarray(quad.Krf_w)
    jacobians = np.abs(np.asarray(reference_mesh.aff_jacs))
    keys = [(wanted, field) for wanted, _, _ in selected for field in fields]
    error_sq = dict.fromkeys(keys, 0.0)
    norm_sq = dict.fromkeys(keys, 0.0)
    outside = 0
    for start in range(0, reference_mesh.num_tri, int(chunk_elements)):
        stop = min(start + int(chunk_elements), reference_mesh.num_tri)
        points = _map(reference_mesh, np.asarray(quad.Krf_quads), start, stop)
        flat = points.reshape(-1, 2)
        owners = np.asarray(finder(flat[:, 0], flat[:, 1]), dtype=np.int64)
        missing = owners < 0
        if np.any(missing):
            outside += int(missing.sum())
            owners[missing] = nearest.query(flat[missing])[1]
        local = np.einsum("nij,nj->ni", np.asarray(run_mesh.inv_aff_mats)[owners],
                          flat - np.asarray(run_mesh.aff_vecs)[owners])
        run_basis = run_space.reference.basis_at(local)  # (points, n_run)
        weights = (jacobians[start:stop, None] * weights_ref[None, :]).reshape(-1)
        for wanted, i, j in selected:
            for field in fields:
                run_coefficients = run_states[(field, i)]
                reference_coefficients = reference_states[(field, j)]
                for component in range(run_coefficients.shape[0]):
                    u = np.einsum("pn,pn->p", run_basis, run_coefficients[component][owners])
                    u_ref = (reference_coefficients[component][start:stop] @ reference_basis.T).reshape(-1)
                    error_sq[(wanted, field)] += float(np.dot(weights, (u - u_ref) ** 2))
                    norm_sq[(wanted, field)] += float(np.dot(weights, u_ref ** 2))
    return [dict(time=wanted, field=field, absolute=float(np.sqrt(error_sq[(wanted, field)])),
                 relative=float(np.sqrt(error_sq[(wanted, field)] / norm_sq[(wanted, field)]))
                 if norm_sq[(wanted, field)] > 0 else None,
                 reference_norm=float(np.sqrt(norm_sq[(wanted, field)])), points_outside=outside)
            for wanted, field in keys]


def _map(mesh, reference_points, start, stop):
    """Physical coordinates (stop-start, q, 2) of reference points on elements start:stop."""
    matrices, vectors = np.asarray(mesh.aff_mats[start:stop]), np.asarray(mesh.aff_vecs[start:stop])
    return np.einsum("kij,qj->kqi", matrices, reference_points) + vectors[:, None, :]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path, help="run snapshots (.npz)")
    parser.add_argument("reference", type=Path, help="reference snapshots (.npz)")
    parser.add_argument("--times", type=lambda text: [float(v) for v in text.split(",")], required=True)
    parser.add_argument("--fields", default=",".join(FIELDS))
    parser.add_argument("--output", type=Path, default=None, help="append JSON lines here")
    args = parser.parse_args()
    rows = cross_mesh_errors(args.run, args.reference, args.times, fields=tuple(args.fields.split(",")))
    for row in rows:
        row.update(run=str(args.run), reference=str(args.reference))
        print(json.dumps(row))
        if args.output is not None:
            with args.output.open("a") as stream:
                stream.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
