"""Transport constraint checks and host inspection of failed systems.

Array reductions accept an explicit NumPy/CuPy namespace; reports and snapshot
inspection remain host-only. This module does not invoke solvers or JIT itself.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np



class UpwindHDGTraceRankError(np.linalg.LinAlgError):
    """A proven loss of inflow support, independent of the solve backend.

    Only small diagnostic samples are stored on the host. In particular, this
    exception does not indicate a NumPy solve or authorize CPU fallback.
    """

    def __init__(self, edges, inflow_nodes, trace_dofs):
        self.edges = list(edges)
        self.inflow_nodes = list(inflow_nodes)
        self.trace_dofs = int(trace_dofs)
        super().__init__(
            "Upwind HDG residual has a rank-deficient active trace constraint: "
            f"edges={self.edges}, inflow_nodes={self.inflow_nodes}, "
            f"trace_dofs={self.trace_dofs}. No residual/history was accepted."
        )


def transport_rank_failure_details(error):
    """Recognize proven active-face rank loss, never generic solve failure.

    Global transport failures may carry the existing saved-system diagnostics.
    Exclude completely inactive faces, which need not constrain a trace.
    """
    if isinstance(error, UpwindHDGTraceRankError):
        return {"edges": error.edges, "inflow_nodes": error.inflow_nodes,
                "trace_dofs": error.trace_dofs}
    report = getattr(error, "transport_diagnostics", {})
    if report.get("advection_stabilization", "upwind") != "upwind":
        return None  # Inflow support alone is not a nullspace proof for other fluxes.
    inflow = report.get("trace_inflow_diagnostics", {})
    dofs = inflow.get("trace_dofs_per_face", 0)
    deficient = [face for face in inflow.get("worst_faces", ())
                 if face.get("inflow_nodes", dofs) < dofs
                 and any(value != 0 for side in face.get("outward_normal_samples", ())
                         for value in side)]
    if deficient:
        return {"edges": [face["edge"] for face in deficient],
                "inflow_nodes": [face["inflow_nodes"] for face in deficient],
                "trace_dofs": dofs}
    return None


def trace_matrix_diagnostics(arrays):
    """Measure rows AND columns without expanding face BSR to scalar CSR.

    Zero columns prove singularity. Nonzero columns and row/column scales do
    not prove nonsingularity or estimate the condition number.
    """
    data = np.asarray(arrays["data"])
    n = int(np.asarray(arrays["rhs"]).size)
    fmt = str(np.asarray(arrays["matrix_format"]).item())
    if not np.isfinite(data).all():
        return {"matrix_size": n, "matrix_finite": False}
    stored_entries = int(data.size)
    if fmt == "coo":
        # Assembly COO contains repeated element contributions. Sum signed
        # duplicates first: abs() before summation can hide cancelled columns.
        from scipy.sparse import coo_matrix

        matrix = coo_matrix((data, (arrays["rows"], arrays["cols"])), shape=(n, n)).tocsr()
        data, indptr, indices = matrix.data, matrix.indptr, matrix.indices
        fmt = "csr"
        if not np.isfinite(data).all():
            return {"matrix_size": n, "matrix_finite": False}
    elif fmt in {"bsr", "csr"}:
        indptr, indices = arrays["indptr"], arrays["indices"]
    row_l1 = np.zeros(n, dtype=np.float64)
    col_l1 = np.zeros(n, dtype=np.float64)
    chunk = 8192
    if fmt in {"bsr", "csr"}:
        block = data.shape[1] if fmt == "bsr" else 1
        rows = row_l1.reshape(-1, block)
        cols = col_l1.reshape(-1, block)
        for start in range(0, len(data), chunk):
            end = min(start + chunk, len(data))
            row_ids = np.searchsorted(indptr, np.arange(start, end), side="right") - 1
            values = np.abs(data[start:end]).reshape(-1, block, block)
            np.add.at(rows, row_ids, values.sum(axis=2, dtype=np.float64))
            np.add.at(cols, indices[start:end], values.sum(axis=1, dtype=np.float64))
    else:
        raise ValueError(f"Unsupported matrix format {fmt!r}")
    zero_rows, zero_cols = np.flatnonzero(row_l1 == 0), np.flatnonzero(col_l1 == 0)
    return {
        "matrix_size": n,
        "matrix_scalar_stored_entries": stored_entries,
        "matrix_finite": True,
        "matrix_row_l1_min": float(row_l1.min()) if n else 0.0,
        "matrix_row_l1_max": float(row_l1.max()) if n else 0.0,
        "matrix_column_l1_min": float(col_l1.min()) if n else 0.0,
        "matrix_column_l1_max": float(col_l1.max()) if n else 0.0,
        "matrix_zero_rows": int(zero_rows.size),
        "matrix_zero_columns": int(zero_cols.size),
        "matrix_zero_row_samples": zero_rows[:28].tolist(),
        "matrix_zero_column_samples": zero_cols[:28].tolist(),
    }


def trace_inflow_node_counts(aligned_normal_pairs, *, xp=np):
    """Count distinct inflow nodes per face from globally aligned side normals.

    Input has shape (faces, 2, quadrature nodes). For the standard upwind
    stabilization, fewer inflow nodes than trace DOFs proves rank deficiency.
    ``xp`` preserves host/device residency; this is not a condition estimate.
    """
    return xp.count_nonzero(xp.any(aligned_normal_pairs < 0, axis=1), axis=1)


def trace_inflow_diagnostics(normal, edge_ids, orientations, interior_edges,
                             trace_basis, weights, *, sample_count=12, stabilization=None):
    """Audit gamma=|beta.n|-beta.n at the actual assembly quadrature nodes.

    The two outward normal traces are aligned in global edge orientation.
    A trace polynomial vanishing at all inflow nodes has zero coupling to
    both elements. Fewer than p+1 distinct inflow nodes therefore proves a
    nullspace in this quadrature-assembled trace system. SVD additionally
    measures near dependence of the weighted trace samples, not of the global
    matrix. Supports standard upwind and conflict-averaged upwind; reported
    support and rank use effective velocities, alongside the raw samples.
    """
    from ..solvers.stabilization import is_conflict_averaged_upwind, conflict_averaged_normal_pair

    normal = np.asarray(normal)
    edge_ids, orientations = np.asarray(edge_ids), np.asarray(orientations)
    interior_edges = np.asarray(interior_edges)
    basis, weights = np.asarray(trace_basis), np.asarray(weights)
    if normal.shape[:2] != edge_ids.shape or orientations.shape != edge_ids.shape:
        raise ValueError("Normal samples and mesh sides must have matching shapes")
    if basis.shape[1] != normal.shape[-1] or weights.shape != (normal.shape[-1],):
        raise ValueError("Normal samples must use the trace assembly quadrature")
    if not np.isfinite(normal).all() or np.any(weights <= 0):
        raise ValueError("Finite normal samples and positive quadrature weights required")
    aligned = np.where(orientations[..., None], normal, normal[..., ::-1]).reshape(-1, normal.shape[-1])
    order = np.argsort(edge_ids.ravel(), kind="stable")
    sorted_edges = edge_ids.ravel()[order]
    starts = np.searchsorted(sorted_edges, interior_edges, side="left")
    ends = np.searchsorted(sorted_edges, interior_edges, side="right")
    if np.any(ends - starts != 2):
        raise ValueError("Every interior edge must have exactly two adjacent sides")
    ntr, nq = basis.shape
    counts = {"interior_faces": int(interior_edges.size), "trace_dofs_per_face": ntr,
              "quadrature_nodes_per_face": nq, "double_outflow_faces": 0,
              "no_inflow_faces": 0, "insufficient_inflow_node_faces": 0,
              "nullity_lower_bound_from_inflow_nodes": 0,
              "numerically_rank_deficient_faces": 0, "inactive_faces": 0,
              "raw_conflict_nodes": 0}
    worst = []
    tol = max(nq, ntr) * np.finfo(normal.dtype).eps
    for start in range(0, len(interior_edges), 2048):
        stop = start + 2048
        sides = np.stack((order[starts[start:stop]], order[starts[start:stop] + 1]), axis=1)
        pair = aligned[sides]
        raw_pair = pair
        if is_conflict_averaged_upwind(stabilization):
            a, b = conflict_averaged_normal_pair(pair[:, 0], pair[:, 1])
            pair = np.stack((a, b), axis=1)
        counts["raw_conflict_nodes"] += int(np.count_nonzero(
            (raw_pair[:, 0] >= 0) & (raw_pair[:, 1] >= 0) & (raw_pair.sum(axis=1) > 0)))
        counts["inactive_faces"] += int(np.count_nonzero(np.all(pair == 0, axis=(1, 2))))
        gamma = (np.abs(pair) - pair).sum(axis=1)
        support = trace_inflow_node_counts(pair)
        maximum = gamma.max(axis=1)
        normalized = gamma / np.where(maximum > 0, maximum, 1)[:, None]
        weighted = np.sqrt(normalized * weights)[:, :, None] * basis.T[None, :, :]
        singular = np.linalg.svd(weighted, compute_uv=False)
        rank = np.count_nonzero(singular > tol * singular[:, :1], axis=1)
        smallest = singular[:, -1] if nq >= ntr else np.zeros(len(pair))
        rcond = smallest / np.where(singular[:, 0] > 0, singular[:, 0], 1)
        counts["double_outflow_faces"] += int(np.count_nonzero(np.all(pair > 0, axis=(1, 2))))
        counts["no_inflow_faces"] += int(np.count_nonzero(support == 0))
        counts["insufficient_inflow_node_faces"] += int(np.count_nonzero(support < ntr))
        counts["nullity_lower_bound_from_inflow_nodes"] += int(np.maximum(ntr-support, 0).sum())
        counts["numerically_rank_deficient_faces"] += int(np.count_nonzero(rank < ntr))
        for j in np.argsort(rcond, kind="stable")[:sample_count]:
            worst.append({
                "edge": int(interior_edges[start+j]),
                "elements": (sides[j] // edge_ids.shape[1]).tolist(),
                "local_faces": (sides[j] % edge_ids.shape[1]).tolist(),
                "inflow_nodes": int(support[j]),
                "sampling_rank": int(rank[j]),
                "sampling_rcond": float(rcond[j]),
                "outward_normal_samples": raw_pair[j].tolist(),
                "effective_outward_normal_samples": pair[j].tolist(),
                "effective_trace_support": int(support[j]),
                "inactive_trace": bool(np.all(pair[j] == 0)),
            })
        worst = sorted(worst, key=lambda item: item["sampling_rcond"])[:sample_count]
    counts["sampling_rank_relative_tolerance"] = float(tol)
    counts["worst_faces"] = worst
    return counts


def trace_face_column_diagnostics(data, indptr, indices, face_index):
    """SVD of all BSR blocks touching one face's columns, including neighbors.

    A null vector of this small panel extends by zero to a null vector of the
    complete matrix. A full-rank panel does not prove global nonsingularity.
    """
    positions = np.flatnonzero(np.asarray(indices) == face_index)
    block = data.shape[-1]
    panel = data[positions].reshape(-1, block)
    if len(positions):
        _, singular, vh = np.linalg.svd(panel, full_matrices=False)
        mode = vh[-1]
    else:
        singular = np.zeros(block)
        mode = np.eye(block)[0]
    scale = float(singular[0])
    tolerance = max(panel.shape) * np.finfo(data.dtype).eps
    residual = float(np.linalg.norm(panel @ mode))
    return {
        "reduced_face_index": int(face_index),
        "column_panel_shape": list(panel.shape),
        "touched_block_rows": (np.searchsorted(indptr, positions, side="right") - 1).tolist(),
        "column_panel_singular_values": singular.tolist(),
        "column_panel_rank": int(np.count_nonzero(singular > tolerance * scale)),
        "column_panel_rcond": float(singular[-1] / scale) if scale else 0.0,
        "column_panel_rank_relative_tolerance": float(tolerance),
        "weakest_trace_mode": mode.tolist(),
        "matrix_mode_residual_norm": residual,
        "matrix_mode_relative_residual": residual / scale if scale else 0.0,
    }


def analyze_transport_snapshot(arrays):
    """Analyze a loaded failure archive without invoking assembly or a solver."""
    report = {"matrix_diagnostics": trace_matrix_diagnostics(arrays)}
    policy = str(np.asarray(arrays.get("advection_stabilization", "upwind")).item())
    report["advection_stabilization"] = policy
    if "normal_flux" in arrays and policy in {"upwind", "conflict-averaged-upwind"}:
        report["trace_inflow_diagnostics"] = trace_inflow_diagnostics(
            arrays["normal_flux"], arrays["edge_ids"], arrays["orientations"],
            arrays["interior_edges"], arrays["trace_basis"], arrays["trace_weights"],
            stabilization=policy,
        )
        inflow = report["trace_inflow_diagnostics"]
        suspect = [face for face in inflow["worst_faces"]
                   if face["sampling_rank"] < inflow["trace_dofs_per_face"] and not face["inactive_trace"]]
        if (suspect and str(np.asarray(arrays["matrix_format"]).item()) == "bsr"
                and report["matrix_diagnostics"]["matrix_finite"]):
            # Read the large data member only once when arrays is an NPZ file.
            data = np.asarray(arrays["data"])
            indptr, indices = arrays["indptr"], arrays["indices"]
            edges = arrays["interior_edges"]
            report["deficient_face_matrix_checks"] = [
                {"edge": face["edge"], **trace_face_column_diagnostics(
                    data, indptr, indices, int(np.searchsorted(edges, face["edge"])))}
                for face in suspect
            ]
    return report


def save_transport_failure_snapshot(path, assembly, *, initial_guess=None, best_solution=None):
    """Save the restored, unscaled failed system and raw assembly inputs.

    Called only after failure. Transfers arrays to host without evaluating any
    GPU kernel. The archive permits independent inspection without replaying
    time integration. It contains no pickle objects. Scaling/restoration may
    have changed physical matrix entries by roundoff during the attempts.
    """
    def host(array):
        return np.asarray(array.get() if hasattr(array, "get") else array)

    arrays = {"format_version": np.array(1), "matrix_format": np.array(assembly.matrix_format),
              "data": host(assembly.data), "rhs": host(assembly.rhs)}
    for name in ("indptr", "indices", "rows", "cols", "boundary_trace"):
        value = getattr(assembly, name, None)
        if value is not None:
            arrays[name] = host(value)
    for name, value in (("initial_guess", initial_guess), ("best_solution", best_solution)):
        if value is not None:
            arrays[name] = host(value)
    raw = getattr(assembly, "raw", None)
    if raw is not None and raw.beta_coeffs is not None:
        space, trace = assembly.cspace.host, assembly.trace_ref.host
        mesh = space.mesh
        for name in ("source_coeffs", "beta_coeffs", "reaction_coeffs"):
            value = getattr(raw, name, None)
            if value is not None:
                arrays[name] = host(value)
        arrays.update(
            advection_stabilization=np.array(str(getattr(raw, "advection_stabilization", None) or "upwind")),
            reaction_scalar=np.array(raw.reaction_scalar),
            reaction_is_scalar=np.array(raw.reaction_is_scalar),
            zero_boundary_flux=np.array(raw.zero_boundary_flux),
            node_coords=mesh.node_coords, triangles=mesh.triangles,
            edge_ids=mesh.loc2glob_edge, orientations=mesh.orientations,
            interior_edges=mesh.int_edges_inds, edge_side_indices=mesh.edge_side_indices,
            trace_basis=trace.bas1d_of_ref_edg_qds, trace_weights=trace.weights,
            trace_quads=trace.quads,
            order=np.array(space.order), basis_type=np.array(space.reference.basis_type),
            trace_kind=np.array(trace.kind),
        )
        # Raw fused assembly uses this exact reference face basis and each
        # element's own outward normal (before boundary-flux suppression).
        arrays["normal_flux"] = np.einsum(
            "dki,kfd,fiq->kfq", arrays["beta_coeffs"], mesh.normals,
            trace.bas_of_bd_quads, optimize=True,
        )
        from ..solvers.stabilization import effective_advection_normal_flux
        arrays["effective_normal_flux"] = effective_advection_normal_flux(
            arrays["normal_flux"], mesh, getattr(raw, "advection_stabilization", None)).copy()
        if raw.zero_boundary_flux:
            arrays["effective_normal_flux"][~mesh.interior_face_mask] = 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, **arrays)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    # Save first, so an analysis failure never loses the numerical evidence.
    report = {"system_snapshot": str(path), "system_snapshot_bytes": path.stat().st_size}
    try:
        report.update(analyze_transport_snapshot(arrays))
    except Exception as error:
        report["snapshot_analysis_error"] = f"{type(error).__name__}: {error}"
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="Failed transport system .npz archive")
    args = parser.parse_args()
    with np.load(args.snapshot, allow_pickle=False) as snapshot:
        print(json.dumps(analyze_transport_snapshot(snapshot), indent=2, allow_nan=False))
