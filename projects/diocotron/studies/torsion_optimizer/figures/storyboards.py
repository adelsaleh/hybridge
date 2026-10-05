"""Paired design/trajectory figures for the optimizer numerical tests.

The publication figures deliberately show the fixed design fields once and the
evolving equilibrium fields in a separate, immediately following storyboard.
All archive data are global, rank-zero products of the MPI optimizer; this
module validates that global layout before attempting contour reconstruction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from projects.diocotron.paths import resolve_archive_path


GEOMETRY_ORDER = ("smooth_star", "pacman", "horseshoe", "iter")
FIXED_STORYBOARD_BANDS: Mapping[str, tuple[float, float]] = {
    "smooth_star": (0.45, 0.50),
    "pacman": (0.60, 0.695),
    "horseshoe": (0.20, 0.295),
    "iter": (0.25, 0.345),
}
PACMAN_STORYBOARD_WIDTH = 0.095
PACMAN_LOWER_LEVELS = (0.60,)
PDE_CONVERGED_CLASSES = {
    "strict_convergence",
    "certified_subband_convergence",
    "geometrically_unsuccessful_pde_converged",
    "pde_converged_with_capped_trials",
}


def _equilibrium(row: Mapping[str, Any]) -> Path | None:
    """Return an existing final-equilibrium checkpoint named by a result row."""
    candidates = (
        row.get("equilibrium"),
        Path(str(row.get("run_dir"))) / "out" / "equilibrium.npz"
        if row.get("run_dir") else None,
    )
    for candidate in candidates:
        if candidate and str(candidate) != "None" and resolve_archive_path(str(candidate)).is_file():
            return resolve_archive_path(str(candidate))
    return None


def _failure_label(row: Mapping[str, Any]) -> str | None:
    """Describe a non-successful attempt without hiding partial PDE evidence."""
    if (
            row.get("state") == "completed"
            and (
                row.get("classification") in PDE_CONVERGED_CLASSES
                or row.get("classification") in (None, "")
            )
    ):
        return None
    fields = [
        f"state={row.get('state', 'unknown')}",
        f"classification={row.get('classification', 'unknown')}",
    ]
    for key in (
            "failure_reason", "validation_error", "error", "terminal_status",
            "exit_status", "exit_code",
    ):
        value = row.get(key)
        if value not in (None, ""):
            fields.append(f"{key}={value}")
            break
    return "; ".join(fields)


def _row_band(row: Mapping[str, Any]) -> tuple[float, float]:
    band = _mapping_band(row)
    if band is not None:
        return band
    geometry = str(row.get("geometry", ""))
    return tuple(FIXED_STORYBOARD_BANDS.get(geometry, (0.25, 0.35)))


def _archive(row: dict[str, Any]) -> Path | None:
    candidates = (
        row.get("trajectoryArchive"),
        row.get("trajectory"),
        Path(str(row.get("run_dir"))) / "out" / "trajectory.npz"
        if row.get("run_dir") else None,
    )
    for candidate in candidates:
        if candidate and str(candidate) != "None" and resolve_archive_path(str(candidate)).is_file():
            return resolve_archive_path(str(candidate))
    return None


def _validate_archive_layout(data, archive: Path) -> None:
    """Reject incomplete or rank-local archives before plotting."""
    required = {
        "mesh_points", "mesh_cells", "dof_coordinates", "fixed_torsion",
        "fixed_target_density", "fixed_target_potential", "states_phi",
        "states_rho", "states_mismatch", "metadata",
    }
    missing = sorted(required.difference(data.files))
    if missing:
        raise ValueError(f"trajectory archive {archive} lacks {', '.join(missing)}")
    coordinates = np.asarray(data["dof_coordinates"], dtype=float)
    cells = np.asarray(data["mesh_cells"], dtype=np.int64)
    points = np.asarray(data["mesh_points"], dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError(f"invalid global DOF coordinates in {archive}")
    if cells.ndim != 2 or cells.shape[1] != 3 or not cells.size:
        raise ValueError(f"invalid global triangular connectivity in {archive}")
    if np.min(cells) < 0 or np.max(cells) >= points.shape[0]:
        raise ValueError(f"out-of-range global mesh connectivity in {archive}")
    rounded = np.round(coordinates[:, :2], decimals=13)
    if np.unique(rounded, axis=0).shape[0] != rounded.shape[0]:
        raise ValueError(f"duplicate owned/global DOF coordinates in {archive}")
    ndofs = coordinates.shape[0]
    for key in ("fixed_torsion", "fixed_target_density", "fixed_target_potential"):
        if np.asarray(data[key]).shape != (ndofs,):
            raise ValueError(f"{key} is not a complete rank-zero global field in {archive}")
    state_shape = np.asarray(data["states_phi"]).shape
    if len(state_shape) != 2 or state_shape[1] != ndofs:
        raise ValueError(f"states_phi is not a complete rank-zero global history in {archive}")
    for key in ("states_rho", "states_mismatch"):
        if np.asarray(data[key]).shape != state_shape:
            raise ValueError(f"{key} does not match states_phi in {archive}")



def _recoverable_storyboard_history(
        data,
        archive: Path,
) -> tuple[
        dict[str, Any],
        list[dict[str, Any]],
        np.ndarray,
        np.ndarray,
        np.ndarray,
]:
    """Return every finite, globally shaped state recoverable from an archive.

    Failed runs can terminate while metadata or the derived mismatch history is
    being written. The primary fields remain useful numerical evidence, so a
    missing mismatch is reconstructed from phi-phi_T and missing state labels
    are synthesized. Mesh/global-field validation stays strict.
    """
    fixed_required = {
        "mesh_points", "mesh_cells", "dof_coordinates", "fixed_torsion",
        "fixed_target_density", "fixed_target_potential", "states_phi",
        "states_rho",
    }
    missing = sorted(fixed_required.difference(data.files))
    if missing:
        raise ValueError(f"trajectory archive {archive} lacks {', '.join(missing)}")

    coordinates = np.asarray(data["dof_coordinates"], dtype=float)
    points = np.asarray(data["mesh_points"], dtype=float)
    cells = np.asarray(data["mesh_cells"], dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError(f"invalid global DOF coordinates in {archive}")
    if cells.ndim != 2 or cells.shape[1] != 3 or not cells.size:
        raise ValueError(f"invalid global triangular connectivity in {archive}")
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"invalid global mesh points in {archive}")
    if np.min(cells) < 0 or np.max(cells) >= points.shape[0]:
        raise ValueError(f"out-of-range global mesh connectivity in {archive}")
    rounded = np.round(coordinates[:, :2], decimals=13)
    if np.unique(rounded, axis=0).shape[0] != rounded.shape[0]:
        raise ValueError(f"duplicate owned/global DOF coordinates in {archive}")

    ndofs = coordinates.shape[0]
    for key in ("fixed_torsion", "fixed_target_density", "fixed_target_potential"):
        if np.asarray(data[key]).shape != (ndofs,):
            raise ValueError(f"{key} is not a complete rank-zero global field in {archive}")

    phi = np.asarray(data["states_phi"], dtype=float)
    rho = np.asarray(data["states_rho"], dtype=float)
    if phi.ndim != 2 or phi.shape[1] != ndofs:
        raise ValueError(f"states_phi has no recoverable global states in {archive}")
    if rho.ndim != 2 or rho.shape[1] != ndofs:
        raise ValueError(f"states_rho has no recoverable global states in {archive}")
    if "states_mismatch" in data.files:
        mismatch = np.asarray(data["states_mismatch"], dtype=float)
        if mismatch.ndim != 2 or mismatch.shape[1] != ndofs:
            mismatch = phi - np.asarray(data["fixed_target_potential"], dtype=float)[None, :]
    else:
        mismatch = phi - np.asarray(data["fixed_target_potential"], dtype=float)[None, :]

    metadata: dict[str, Any] = {}
    if "metadata" in data.files:
        try:
            decoded = json.loads(str(data["metadata"].item()))
            if isinstance(decoded, Mapping):
                metadata = dict(decoded)
        except (json.JSONDecodeError, TypeError, ValueError):
            metadata = {}

    raw_states = metadata.get("states", [])
    if not isinstance(raw_states, Sequence) or isinstance(raw_states, (str, bytes)):
        raw_states = []
    count = min(phi.shape[0], rho.shape[0], mismatch.shape[0])
    if count <= 0:
        raise ValueError(f"trajectory archive has no usable state fields: {archive}")

    terminal_status = str(metadata.get("terminal_status", "")).upper()
    terminal_stage = str(metadata.get("terminal_stage", "")).strip()
    states: list[dict[str, Any]] = []
    for index in range(count):
        raw_state = raw_states[index] if index < len(raw_states) else None
        state = dict(raw_state) if isinstance(raw_state, Mapping) else {
            "stage": "recovered_state",
            "outer_iteration": -1,
            "homotopy_lambda": math.nan,
        }
        if index == count - 1 and not isinstance(raw_state, Mapping):
            if terminal_stage:
                stage_name = terminal_stage
                if "FAIL" in terminal_status and "failure" not in stage_name.lower():
                    stage_name = f"{stage_name}_terminal_failure"
                state["stage"] = stage_name
            else:
                state["stage"] = "recovered_terminal"
        states.append(state)

    finite = (
        np.all(np.isfinite(phi[:count]), axis=1)
        & np.all(np.isfinite(rho[:count]), axis=1)
        & np.all(np.isfinite(mismatch[:count]), axis=1)
    )
    usable = np.flatnonzero(finite)
    if not usable.size:
        raise ValueError(f"trajectory archive has no finite usable states: {archive}")
    return (
        metadata,
        [states[int(index)] for index in usable],
        np.ascontiguousarray(phi[usable]),
        np.ascontiguousarray(rho[usable]),
        np.ascontiguousarray(mismatch[usable]),
    )

def _triangulation(data):
    import matplotlib.tri as mtri

    coordinates = np.asarray(data["dof_coordinates"], dtype=float)
    triangulation = mtri.Triangulation(coordinates[:, 0], coordinates[:, 1])
    physical = mtri.Triangulation(
        np.asarray(data["mesh_points"])[:, 0],
        np.asarray(data["mesh_points"])[:, 1],
        np.asarray(data["mesh_cells"], dtype=np.int64),
    )
    centroids = coordinates[triangulation.triangles].mean(axis=1)
    triangulation.set_mask(
        physical.get_trifinder()(centroids[:, 0], centroids[:, 1]) < 0
    )
    return triangulation


def _physical_boundary(data) -> tuple[np.ndarray, np.ndarray, float, float]:
    points = np.asarray(data["mesh_points"], dtype=float)[:, :2]
    cells = np.asarray(data["mesh_cells"], dtype=np.int64)
    edges = np.sort(
        np.vstack((cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]])),
        axis=1,
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if not len(boundary_edges):
        raise ValueError("physical mesh has no boundary edges")
    starts, ends = points[boundary_edges[:, 0]], points[boundary_edges[:, 1]]
    boundary_length = float(np.linalg.norm(ends - starts, axis=1).sum())
    triangles = points[cells]
    twice_area = (
        (triangles[:, 1, 0] - triangles[:, 0, 0])
        * (triangles[:, 2, 1] - triangles[:, 0, 1])
        - (triangles[:, 1, 1] - triangles[:, 0, 1])
        * (triangles[:, 2, 0] - triangles[:, 0, 0])
    )
    domain_area = float(0.5 * np.abs(twice_area).sum())
    return starts, ends, boundary_length, domain_area


def _distance_to_segments(
        points: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
) -> np.ndarray:
    """Compute exact point-to-segment distances without SciPy."""
    result = np.empty(len(points), dtype=float)
    direction = ends - starts
    denominator = np.einsum("ij,ij->i", direction, direction)
    denominator = np.maximum(denominator, np.finfo(float).tiny)
    for first in range(0, len(points), 128):
        chunk = points[first:first + 128]
        relative = chunk[:, None, :] - starts[None, :, :]
        fraction = np.einsum("csi,si->cs", relative, direction) / denominator[None, :]
        fraction = np.clip(fraction, 0.0, 1.0)
        projected = starts[None, :, :] + fraction[:, :, None] * direction[None, :, :]
        result[first:first + len(chunk)] = np.min(
            np.linalg.norm(chunk[:, None, :] - projected, axis=2), axis=1,
        )
    return result


def _longest_contour(triangulation, values: np.ndarray, level: float) -> np.ndarray:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        contour = ax.tricontour(triangulation, values, levels=[level])
        segments = [
            np.asarray(segment, dtype=float)
            for segment in contour.allsegs[0]
            if len(segment) >= 3
        ]
    finally:
        plt.close(fig)
    if not segments:
        raise ValueError(f"torsion level {level:.6g} has no valid contour")
    return max(
        segments,
        key=lambda segment: float(np.linalg.norm(np.diff(segment, axis=0), axis=1).sum()),
    )


def _contour_features(
        data,
        alpha: float,
        *,
        triangulation=None,
        boundary: tuple[np.ndarray, np.ndarray, float, float] | None = None,
) -> np.ndarray:
    torsion = np.asarray(data["fixed_torsion"], dtype=float)
    coordinates = np.asarray(data["dof_coordinates"], dtype=float)[:, :2]
    contour = _longest_contour(
        triangulation if triangulation is not None else _triangulation(data),
        torsion,
        alpha * float(np.max(torsion)),
    )
    starts, ends, boundary_length, domain_area = (
        boundary if boundary is not None else _physical_boundary(data)
    )
    boundary_distance = _distance_to_segments(contour, starts, ends)
    core = coordinates[int(np.argmax(torsion))]
    core_distance = np.linalg.norm(contour - core[None, :], axis=1)
    proximity = float(np.median(boundary_distance / np.maximum(
        boundary_distance + core_distance, np.finfo(float).tiny,
    )))
    contour_length = float(np.linalg.norm(np.diff(contour, axis=0), axis=1).sum())
    closed = np.vstack((contour, contour[0]))
    contour_area = float(0.5 * abs(np.sum(
        closed[:-1, 0] * closed[1:, 1] - closed[1:, 0] * closed[:-1, 1]
    )))
    return np.asarray((
        proximity,
        contour_length / max(boundary_length, np.finfo(float).tiny),
        contour_area / max(domain_area, np.finfo(float).tiny),
    ))


def _overview_archives(rows: Sequence[dict[str, Any]]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for geometry in GEOMETRY_ORDER:
        candidates = [
            row for row in rows
            if row.get("geometry") == geometry
            and row.get("kind") == "geometry_overview"
            and _archive(row) is not None
        ]
        candidates.sort(key=lambda row: (
            0 if row.get("state") == "completed" else 1,
            -int(row.get("dof_target", 0) or 0),
            str(row.get("id", "")),
        ))
        if candidates:
            archive = _archive(candidates[0])
            if archive is not None:
                result[geometry] = archive
    return result


def select_pacman_storyboard_band(
        rows: Sequence[dict[str, Any]],
) -> tuple[float, float]:
    """Return the explicitly requested revised Pacman storyboard band."""
    del rows
    return tuple(FIXED_STORYBOARD_BANDS["pacman"])


def select_storyboard_bands(
        rows: Sequence[dict[str, Any]],
) -> dict[str, tuple[float, float]]:
    """Return all four publication storyboard bands."""
    del rows
    return {geometry: tuple(FIXED_STORYBOARD_BANDS[geometry]) for geometry in GEOMETRY_ORDER}


def _quantile_indices(
        indices: Sequence[int],
        slots: int,
) -> list[int]:
    """Select ordered quantiles, repeating endpoints for short histories."""
    if slots <= 0:
        return []
    if not indices:
        raise ValueError("cannot select quantiles from an empty history")
    positions = np.linspace(0, len(indices) - 1, slots, dtype=int)
    return [int(indices[int(position)]) for position in positions]


def _frame_indices(states: Sequence[dict[str, Any]]) -> list[int]:
    """Select six chronological evolution columns with the terminal state last.

    The last slot is reserved before any sampling. When an archive has no
    homotopy states, the four interior slots are ordered quantiles of accepted
    outer iterations; repeated states are intentional for short histories.
    """
    count = len(states)
    if count == 0:
        return []
    homotopy = [
        i for i, state in enumerate(states) if state.get("stage") == "homotopy"
    ]
    accepted = [
        i for i, state in enumerate(states) if state.get("stage") == "accepted_outer"
    ]

    def accepted_order(index: int) -> tuple[float, int]:
        try:
            iteration = float(states[index].get("outer_iteration", math.inf))
        except (TypeError, ValueError):
            iteration = math.inf
        return iteration, index

    accepted.sort(key=accepted_order)
    restored = [
        i for i, state in enumerate(states) if state.get("stage") == "restored_best"
    ]
    final = [
        i for i, state in enumerate(states)
        if (
            str(state.get("stage", "")) == "final"
            or str(state.get("stage", "")).endswith("terminal_failure")
            or str(state.get("stage", "")) == "recovered_terminal"
        )
    ]
    seed = next(
        (i for i, state in enumerate(states) if state.get("stage") == "selected_seed"),
        0,
    )
    terminal = final[-1] if final else restored[-1] if restored else count - 1

    if accepted and not homotopy:
        return [seed, *_quantile_indices(accepted, 4), terminal]

    if accepted:
        def homotopy_distance(index: int) -> float:
            try:
                value = float(states[index].get("homotopy_lambda", 0.0))
            except (TypeError, ValueError):
                value = 0.0
            return abs(value - 0.5) if math.isfinite(value) else math.inf

        mid_homotopy = min(
            homotopy,
            key=homotopy_distance,
        )
        penultimate = (
            restored[-1]
            if restored and restored[-1] != terminal
            else accepted[-1]
        )
        return [
            seed,
            mid_homotopy,
            accepted[0],
            accepted[len(accepted) // 2],
            penultimate,
            terminal,
        ]

    # A failed homotopy can contain a primary failure, a fallback seed, and a
    # second partial continuation. Preserve their archive order instead of
    # sorting by lambda, because lambda legitimately resets between attempts.
    chronological_end = terminal if terminal > seed else count
    progress = [
        index for index in range(seed, chronological_end)
        if index != terminal
    ] or [seed]
    return [*_quantile_indices(progress, 5), terminal]


def _title(state: dict[str, Any]) -> str:
    stage = str(state.get("stage", "state")).replace("_", " ")
    if stage == "homotopy" or "failure" in stage:
        try:
            homotopy_lambda = float(state.get("homotopy_lambda", math.nan))
        except (TypeError, ValueError):
            homotopy_lambda = math.nan
        suffix = f"$\\lambda={homotopy_lambda:.3g}$" if math.isfinite(homotopy_lambda) else "partial state"
        return f"{stage}\n{suffix}"
    try:
        iteration = int(state.get("outer_iteration", -1))
    except (TypeError, ValueError):
        iteration = -1
    return f"{stage}\n$k={iteration}$" if iteration >= 0 else stage


def _mapping_band(mapping: Mapping[str, Any]) -> tuple[float, float] | None:
    """Extract a finite increasing torsion-level pair from a row or metadata."""
    try:
        band = (float(mapping["alpha_t1"]), float(mapping["alpha_t2"]))
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in band) or band[1] <= band[0]:
        return None
    return band


def _finite_levels(
        values: Sequence[Any],
        *,
        minimum: float,
        maximum: float,
) -> list[float]:
    """Return sorted distinct contour levels strictly inside a field range."""
    result: list[float] = []
    for value in values:
        try:
            level = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(level) and minimum < level < maximum:
            result.append(level)
    return sorted(set(result))


def _level_text(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _band_slug(geometry: str, band: tuple[float, float]) -> str:
    levels = "_".join(_level_text(value).replace(".", "p") for value in band)
    return f"{geometry}_{levels}"


def _matches_band(row: dict[str, Any], band: tuple[float, float]) -> bool:
    try:
        return math.isclose(float(row["alpha_t1"]), band[0], abs_tol=5.0e-7) and math.isclose(
            float(row["alpha_t2"]), band[1], abs_tol=5.0e-7
        )
    except (KeyError, TypeError, ValueError):
        return False


def _archive_matches_band(archive: Path, band: tuple[float, float]) -> bool:
    """Require the selected archive itself to encode the requested band."""
    try:
        with np.load(archive, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return _matches_band(metadata, band)




def _render_pending(
        destination: Path,
        geometry: str,
        band: tuple[float, float],
        kind: str,
        reason: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    width, height = (15.5, 4.2) if kind == "design" else (15.5, 8.0)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    ax.axis("off")
    ax.text(
        0.5, 0.62,
        f"{geometry.replace('_', ' ')} "
        f"[{_level_text(band[0])}, {_level_text(band[1])}]",
        ha="center", va="center", fontsize=18, fontweight="bold",
    )
    if kind == "design":
        needed = "fixed-field overview pending"
    elif kind == "failed_storyboard":
        needed = "FAILED/PARTIAL TRAJECTORY — no recoverable storyboard"
    else:
        needed = "PDE-converged trajectory pending"
    message = needed if not reason else f"{needed}\n{reason[:500]}"
    ax.text(
        0.5, 0.42, message, ha="center", va="center", fontsize=13,
        color="#9b1c1c" if reason else "black", wrap=True,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=300)
    plt.close(fig)


def _render_design(
        row: dict[str, Any],
        archive: Path,
        destination: Path,
) -> None:
    import matplotlib.pyplot as plt

    with np.load(archive, allow_pickle=False) as data:
        _validate_archive_layout(data, archive)
        metadata = json.loads(str(data["metadata"].item()))
        tri = _triangulation(data)
        torsion = np.asarray(data["fixed_torsion"], dtype=float)
        levels = [
            float(value) for value in (metadata.get("c1_t"), metadata.get("c2_t"))
            if value is not None and float(np.min(torsion)) < float(value) < float(np.max(torsion))
        ]
        fields = (
            (torsion, r"$T$"),
            (np.asarray(data["fixed_target_density"], dtype=float), r"$\rho_T$"),
            (np.asarray(data["fixed_target_potential"], dtype=float), r"$\phi_T$"),
        )
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2), constrained_layout=True)
        for ax, (values, title) in zip(axes, fields, strict=True):
            image = ax.tricontourf(
                tri, values, levels=48, cmap="viridis", antialiased=False,
            )
            image.set_rasterized(True)
            if levels:
                ax.tricontour(
                    tri, torsion, levels=sorted(levels), colors="#e68613",
                    linestyles="--", linewidths=0.7,
                )
            ax.set_aspect("equal")
            ax.set_axis_off()
            ax.set_title(title, fontsize=13)
            fig.colorbar(image, ax=ax, shrink=0.78, pad=0.015)
        fig.suptitle(
            f"{str(row.get('geometry', '')).replace('_', ' ')} design — "
            f"$[\\alpha_{{T1}},\\alpha_{{T2}}]="
            f"[{_level_text(float(row.get('alpha_t1', 0)))},"
            f"{_level_text(float(row.get('alpha_t2', 0)))}]$",
            fontsize=12,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=300)
        plt.close(fig)


def _render_storyboard(
        row: dict[str, Any],
        archive: Path,
        destination: Path,
        *,
        failure_label: str | None = None,
        include_design: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    with np.load(archive, allow_pickle=False) as data:
        metadata, states, phi_all, rho_all, mismatch_all = (
            _recoverable_storyboard_history(data, archive)
        )
        indices = _frame_indices(states)
        if not indices:
            raise ValueError(f"trajectory archive has no states: {archive}")
        tri = _triangulation(data)
        torsion = np.asarray(data["fixed_torsion"], dtype=float)
        torsion_min = float(np.min(torsion))
        torsion_max = float(np.max(torsion))
        band = _mapping_band(row) or _mapping_band(metadata)
        fixed_levels = _finite_levels(
            (metadata.get("c1_t"), metadata.get("c2_t")),
            minimum=torsion_min,
            maximum=torsion_max,
        )
        if len(fixed_levels) < 2 and band is not None:
            fixed_levels = _finite_levels(
                (band[0] * torsion_max, band[1] * torsion_max),
                minimum=torsion_min,
                maximum=torsion_max,
            )

        rho_values = rho_all[indices]
        phi_values = phi_all[indices]
        mismatch_values = mismatch_all[indices]
        mismatch_limit = max(float(np.max(np.abs(mismatch_values))), 1.0e-16)
        ranges = (
            (float(np.min(rho_values)), float(np.max(rho_values))),
            (float(np.min(phi_values)), float(np.max(phi_values))),
            (-mismatch_limit, mismatch_limit),
        )
        cmaps = ("viridis", "viridis", "coolwarm")
        arrays = (rho_values, phi_values, mismatch_values)

        if include_design:
            fig = plt.figure(figsize=(15.5, 10.8), constrained_layout=True)
            grid = fig.add_gridspec(4, 6, height_ratios=(1.0, 1.0, 1.0, 1.0))
            design_axes = (
                fig.add_subplot(grid[0, 0:2]),
                fig.add_subplot(grid[0, 2:4]),
                fig.add_subplot(grid[0, 4:6]),
            )
            axes = np.asarray([
                [fig.add_subplot(grid[row_index, column]) for column in range(6)]
                for row_index in range(1, 4)
            ])
            design_fields = (
                (torsion, r"$T$"),
                (np.asarray(data["fixed_target_density"], dtype=float), r"$\rho_T$"),
                (np.asarray(data["fixed_target_potential"], dtype=float), r"$\phi_T$"),
            )
            for design_axis, (design_values, design_title) in zip(
                    design_axes, design_fields, strict=True,
            ):
                design_image = design_axis.tricontourf(
                    tri, design_values, levels=48, cmap="viridis", antialiased=False,
                )
                design_image.set_rasterized(True)
                if fixed_levels:
                    design_axis.tricontour(
                        tri, torsion, levels=fixed_levels, colors="#e68613",
                        linestyles="--", linewidths=0.7,
                    )
                design_axis.set_aspect("equal")
                design_axis.set_axis_off()
                design_axis.set_title(f"design: {design_title}", fontsize=10)
                fig.colorbar(design_image, ax=design_axis, shrink=0.72, pad=0.015)
        else:
            fig, axes = plt.subplots(3, 6, figsize=(15.5, 8.0), constrained_layout=True)
        row_images = []
        for column, index in enumerate(indices):
            state = states[index]
            for row_index, (values, value_range, cmap) in enumerate(
                    zip(arrays, ranges, cmaps, strict=True)):
                ax = axes[row_index, column]
                vmin, vmax = value_range
                if math.isclose(vmin, vmax):
                    vmax = vmin + 1.0e-15
                image = ax.tricontourf(
                    tri, values[column], levels=48, cmap=cmap, vmin=vmin, vmax=vmax,
                    antialiased=False,
                )
                image.set_rasterized(True)
                if column == 0:
                    row_images.append(image)
                moving_levels = _finite_levels(
                    (state.get("c1"), state.get("c2")),
                    minimum=float(np.min(phi_all[index])),
                    maximum=float(np.max(phi_all[index])),
                )
                if fixed_levels:
                    ax.tricontour(
                        tri, torsion, levels=sorted(fixed_levels), colors="#e68613",
                        linestyles="--", linewidths=0.7,
                    )
                if moving_levels:
                    ax.tricontour(
                        tri, phi_all[index], levels=sorted(moving_levels), colors="#1769aa",
                        linestyles="-", linewidths=0.7,
                    )
                ax.set_aspect("equal")
                ax.set_axis_off()
                if row_index == 0:
                    ax.set_title(_title(state), fontsize=9)
        for ax, label in zip(
                axes[:, 0], (r"$\rho$", r"$\phi$", r"$\phi-\phi_T$"), strict=True,
        ):
            ax.text(
                -0.08, 0.5, label, transform=ax.transAxes, rotation=90,
                ha="center", va="center", fontsize=11, clip_on=False,
            )
        for row_index, image in enumerate(row_images):
            fig.colorbar(image, ax=axes[row_index, :], shrink=0.76, pad=0.01)
        band_text = (
            f"[{_level_text(band[0])}, {_level_text(band[1])}]"
            if band is not None else "[band metadata unavailable]"
        )
        terminal_status = str(
            metadata.get("terminal_status", row.get("classification", row.get("state", "")))
        )
        heading = (
            f"{str(row.get('geometry', '')).replace('_', ' ')} {band_text} — {terminal_status}"
        )
        if failure_label:
            heading = f"FAILED/PARTIAL CASE — {heading}\n{failure_label[:400]}"
        fig.suptitle(heading, fontsize=12, color="#9b1c1c" if failure_label else "black")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=300)
        plt.close(fig)


def _load_equilibrium_fields(
        checkpoint: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load the portable v1/v2 nodal fields needed for a fallback panel."""
    with np.load(checkpoint, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        format_name = str(metadata.get("format", "hybridge_equilibrium_v1"))
        coordinates = np.asarray(data["coordinates"], dtype=float)
        if format_name == "hybridge_equilibrium_v2":
            names = [str(value) for value in np.asarray(data["field_names"]).tolist()]
            nodal_values = np.asarray(data["nodal_values"], dtype=float)
            phi = np.asarray(nodal_values[names.index("phi")], dtype=float)
            rho = np.asarray(nodal_values[names.index("rho")], dtype=float)
        else:
            phi = np.asarray(data["phi"], dtype=float)
            rho = np.asarray(data["rho"], dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("equilibrium checkpoint has invalid coordinates")
    if phi.shape != (coordinates.shape[0],) or rho.shape != phi.shape:
        raise ValueError("equilibrium checkpoint has incompatible nodal fields")
    if not (
            np.all(np.isfinite(coordinates))
            and np.all(np.isfinite(phi))
            and np.all(np.isfinite(rho))
    ):
        raise ValueError("equilibrium checkpoint contains non-finite fields")
    return coordinates, phi, rho, metadata


def _equilibrium_triangulation(
        coordinates: np.ndarray,
        metadata: Mapping[str, Any],
):
    """Triangulate checkpoint DOFs and mask exterior/hole simplices from its mesh."""
    mesh_path = resolve_archive_path(str(metadata.get("mesh_path", "")))
    if not mesh_path.is_file():
        raise ValueError(f"equilibrium mesh is unavailable: {mesh_path}")
    try:
        import meshio
    except ImportError as exc:  # pragma: no cover - project environment provides meshio
        raise RuntimeError("meshio is required to reconstruct equilibrium contours") from exc
    mesh = meshio.read(mesh_path)
    triangles = [
        np.asarray(block.data[:, :3], dtype=np.int64)
        for block in mesh.cells
        if str(block.type).startswith("triangle") and len(block.data)
    ]
    if not triangles:
        raise ValueError(f"equilibrium mesh contains no triangular cells: {mesh_path}")
    payload = {
        "dof_coordinates": coordinates,
        "mesh_points": np.asarray(mesh.points, dtype=float),
        "mesh_cells": np.vstack(triangles),
    }
    return _triangulation(payload)


def _render_equilibrium_attempt(
        row: Mapping[str, Any],
        checkpoint: Path,
        destination: Path,
        *,
        design_archive: Path | None,
        failure_label: str | None,
) -> None:
    """Render a prior run that retained only its final equilibrium checkpoint."""
    import matplotlib.pyplot as plt

    coordinates, phi, rho, metadata = _load_equilibrium_fields(checkpoint)
    final_tri = _equilibrium_triangulation(coordinates, metadata)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.0), constrained_layout=True)
    fixed_overlay = None
    design_error = None
    if design_archive is not None:
        try:
            with np.load(design_archive, allow_pickle=False) as design:
                _validate_archive_layout(design, design_archive)
                design_metadata = json.loads(str(design["metadata"].item()))
                design_tri = _triangulation(design)
                torsion = np.asarray(design["fixed_torsion"], dtype=float)
                fixed_levels = _finite_levels(
                    (design_metadata.get("c1_t"), design_metadata.get("c2_t")),
                    minimum=float(np.min(torsion)), maximum=float(np.max(torsion)),
                )
                fixed_overlay = (design_tri, torsion, fixed_levels)
                design_fields = (
                    (torsion, r"design: $T$"),
                    (np.asarray(design["fixed_target_density"], dtype=float),
                     r"design: $\rho_T$"),
                    (np.asarray(design["fixed_target_potential"], dtype=float),
                     r"design: $\phi_T$"),
                )
                for axis, (values, title) in zip(axes[0], design_fields, strict=True):
                    image = axis.tricontourf(
                        design_tri, values, levels=48, cmap="viridis", antialiased=False,
                    )
                    image.set_rasterized(True)
                    if fixed_levels:
                        axis.tricontour(
                            design_tri, torsion, levels=fixed_levels, colors="#e68613",
                            linestyles="--", linewidths=0.7,
                        )
                    axis.set_aspect("equal")
                    axis.set_axis_off()
                    axis.set_title(title, fontsize=10)
                    fig.colorbar(image, ax=axis, shrink=0.72, pad=0.015)
        except Exception as exc:
            design_error = f"{type(exc).__name__}: {exc}"
    if fixed_overlay is None:
        band = _row_band(row)
        design_lines = (
            f"design geometry\n{str(row.get('geometry', 'unknown')).replace('_', ' ')}",
            f"torsion target band\n[{_level_text(band[0])}, {_level_text(band[1])}]",
            r"target construction" "\n" r"$\rho_T=\mathbf{1}_{\alpha_{T1}<T/T_{\max}<\alpha_{T2}}$"
            "\n" r"$-\Delta\phi_T=\rho_T$",
        )
        for axis, text in zip(axes[0], design_lines, strict=True):
            axis.axis("off")
            axis.text(0.5, 0.5, text, ha="center", va="center", fontsize=13)

    final_fields = ((rho, r"final checkpoint: $\rho$"), (phi, r"final checkpoint: $\phi$"))
    for axis, (values, title) in zip(axes[1, :2], final_fields, strict=True):
        image = axis.tricontourf(
            final_tri, values, levels=48, cmap="viridis", antialiased=False,
        )
        image.set_rasterized(True)
        if fixed_overlay is not None:
            design_tri, torsion, fixed_levels = fixed_overlay
            if fixed_levels:
                axis.tricontour(
                    design_tri, torsion, levels=fixed_levels, colors="#e68613",
                    linestyles="--", linewidths=0.7,
                )
        if title.endswith("$\\phi$"):
            moving_levels = _finite_levels(
                (metadata.get("c1_phi"), metadata.get("c2_phi")),
                minimum=float(np.min(phi)), maximum=float(np.max(phi)),
            )
            if moving_levels:
                axis.tricontour(
                    final_tri, phi, levels=moving_levels, colors="#1769aa",
                    linestyles="-", linewidths=0.7,
                )
        axis.set_aspect("equal")
        axis.set_axis_off()
        axis.set_title(title, fontsize=10)
        fig.colorbar(image, ax=axis, shrink=0.72, pad=0.015)

    diagnostic = axes[1, 2]
    diagnostic.axis("off")
    lines = [
        "trajectory archive unavailable; final checkpoint recovered",
        f"status: {metadata.get('final_status', row.get('state', 'unknown'))}",
        f"residual: {metadata.get('final_residual', 'unknown')}",
        f"order: {metadata.get('order', row.get('order', 'unknown'))}",
    ]
    if design_error:
        lines.append(f"visual design fallback: {design_error}")
    diagnostic.text(0.02, 0.96, "\n".join(lines), va="top", fontsize=10, wrap=True)
    heading = f"{row.get('id', 'attempt')} — design and recovered final equilibrium"
    if failure_label:
        heading = f"FAILED/PARTIAL ATTEMPT — {heading}\n{failure_label[:400]}"
    fig.suptitle(heading, fontsize=12, color="#9b1c1c" if failure_label else "black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=300)
    plt.close(fig)


def render_attempt_storyboard(
        row: Mapping[str, Any],
        destination: Path,
        *,
        trajectory: Path | None = None,
        equilibrium: Path | None = None,
        design_archive: Path | None = None,
) -> dict[str, Any]:
    """Always emit one PNG evidence artifact for one concrete run attempt.

    Recoverable trajectory states take priority, followed by a final
    equilibrium checkpoint. Infrastructure failures and archives with no
    usable fields produce a diagnostic PNG rather than silently disappearing.
    """
    row_dict = dict(row)
    destination = Path(destination)
    if destination.suffix.lower() != ".png":
        destination = destination.with_suffix(".png")
    trajectory = resolve_archive_path(trajectory) if trajectory is not None else _archive(row_dict)
    equilibrium = resolve_archive_path(equilibrium) if equilibrium is not None else _equilibrium(row_dict)
    design_archive = resolve_archive_path(design_archive) if design_archive is not None else None
    label = _failure_label(row_dict)
    errors: list[str] = []

    if trajectory is not None and trajectory.is_file():
        try:
            if row_dict.get("kind") == "geometry_overview":
                _render_design(row_dict, trajectory, destination)
                source = "design_archive"
            else:
                _render_storyboard(
                    row_dict, trajectory, destination,
                    failure_label=label, include_design=True,
                )
                source = "trajectory"
            return {
                "file": str(destination), "status": "actual", "source": source,
                "failure_label": label,
            }
        except Exception as exc:
            errors.append(f"trajectory {trajectory}: {type(exc).__name__}: {exc}")

    if equilibrium is not None and equilibrium.is_file():
        try:
            _render_equilibrium_attempt(
                row_dict, equilibrium, destination,
                design_archive=design_archive, failure_label=label,
            )
            return {
                "file": str(destination), "status": "checkpoint_only",
                "source": "equilibrium", "failure_label": label,
                "warnings": errors,
            }
        except Exception as exc:
            errors.append(f"equilibrium {equilibrium}: {type(exc).__name__}: {exc}")

    provenance = []
    for key in ("id", "attempt_id", "kind", "run_dir", "stdout", "stderr", "exit_code"):
        value = row_dict.get(key)
        if value not in (None, ""):
            provenance.append(f"{key}={value}")
    reason = "; ".join([*(errors or ["no trajectory or equilibrium fields were written"]), *provenance])
    _render_pending(
        destination,
        str(row_dict.get("geometry", "unknown")),
        _row_band(row_dict),
        "failed_storyboard",
        reason=reason,
    )
    return {
        "file": str(destination), "status": "diagnostic_only",
        "source": "placeholder", "failure_label": label, "warnings": errors,
    }


def create_trajectory_storyboards(
        bundle: Path,
        rows: Sequence[dict[str, Any]],
        stage: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[Path]]:
    """Render four exact-band design/storyboard pairs.

    PDE-converged exact-band trajectories remain preferred. If none exists,
    an exact-band failed/partial archive fills the canonical evidence panel.
    Every failed trajectory archive is also retained as a clearly labelled,
    PNG-only supplement, including an explicit reason panel when unusable.
    """
    try:
        bands = select_storyboard_bands(rows)
        pacman_selection_pending = False
    except ValueError:
        # Figure generation must remain usable before the overview campaign.
        # This placeholder is explicitly pending and cannot select a run.
        bands = dict(FIXED_STORYBOARD_BANDS)
        bands["pacman"] = (PACMAN_LOWER_LEVELS[0],
                           PACMAN_LOWER_LEVELS[0] + PACMAN_STORYBOARD_WIDTH)
        pacman_selection_pending = True

    class_priority = {
        "strict_convergence": 0,
        "certified_subband_convergence": 1,
        "geometrically_unsuccessful_pde_converged": 2,
        "pde_converged_with_capped_trials": 3,
    }


    geometric_success_classes = {
        "strict_convergence",
        "certified_subband_convergence",
    }

    def is_working_trajectory(row: Mapping[str, Any]) -> bool:
        return bool(
            row.get("state") == "completed"
            and row.get("classification") in PDE_CONVERGED_CLASSES
        )

    def is_failed_evidence(row: Mapping[str, Any]) -> bool:
        return not (
            row.get("state") == "completed"
            and row.get("classification") in geometric_success_classes
        )

    def failure_label(row: Mapping[str, Any]) -> str:
        fields = [
            f"state={row.get('state', 'unknown')}",
            f"classification={row.get('classification', 'unknown')}",
        ]
        for key in (
                "failure_reason", "validation_error", "error",
                "terminal_status", "exit_status",
        ):
            value = row.get(key)
            if value not in (None, ""):
                fields.append(f"{key}={value}")
                break
        return "; ".join(fields)

    def evidence_band(
            row: Mapping[str, Any],
            archive: Path | None,
    ) -> tuple[float, float]:
        band = _mapping_band(row)
        if band is not None:
            return band
        if archive is not None:
            try:
                with np.load(archive, allow_pickle=False) as data:
                    decoded = json.loads(str(data["metadata"].item()))
                if isinstance(decoded, Mapping):
                    band = _mapping_band(decoded)
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                band = None
        geometry = str(row.get("geometry", ""))
        return band or tuple(FIXED_STORYBOARD_BANDS.get(geometry, (0.25, 0.35)))

    def filename_token(case_id: Any) -> str:
        raw = str(case_id or "unnamed")
        token = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in raw
        ).strip("_")
        return (token or "unnamed")[:100]


    def render_case(
            row: dict[str, Any],
            archive: Path,
            destination: Path,
            geometry: str,
            band: tuple[float, float],
            *,
            label_failure: bool,
    ) -> tuple[str, str | None]:
        label = failure_label(row) if label_failure else None
        try:
            _render_storyboard(
                row,
                archive,
                destination,
                failure_label=label,
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            _render_pending(
                destination,
                geometry,
                band,
                "failed_storyboard" if label_failure else "storyboard",
                reason=reason,
            )
            return "failed_archive", reason
        return ("failed_evidence" if label_failure else "actual"), None

    def exact_candidates(geometry: str, band: tuple[float, float]):
        return [
            row for row in rows
            if row.get("geometry") == geometry
            and _matches_band(row, band)
            and (archive := _archive(row)) is not None
            and _archive_matches_band(archive, band)
        ]

    def design_row(geometry: str, band: tuple[float, float]):
        candidates = [
            row for row in exact_candidates(geometry, band)
            if row.get("kind") == "geometry_overview"
            and row.get("state") == "completed"
        ]
        candidates.sort(key=lambda row: (
            -int(row.get("dof_target", 0) or 0),
            str(row.get("id", "")),
        ))
        return candidates[0] if candidates else None

    def trajectory_row(geometry: str, band: tuple[float, float]):
        candidates = [
            row for row in exact_candidates(geometry, band)
            if str(row.get("kind", "")).startswith("trajectory")
        ]
        working = [row for row in candidates if is_working_trajectory(row)]
        working.sort(key=lambda row: (
            class_priority.get(str(row.get("classification")), 9),
            -int(row.get("dof_target", 0) or 0),
            str(row.get("id", "")),
        ))
        if working:
            return working[0]
        # Do not hide an exact-band failed/partial trajectory. Prefer a
        # completed geometric failure over an interrupted PDE solve, then the
        # highest-resolution deterministic case.
        candidates.sort(key=lambda row: (
            0 if row.get("state") == "completed" else 1,
            class_priority.get(str(row.get("classification")), 9),
            -int(row.get("dof_target", 0) or 0),
            str(row.get("id", "")),
        ))
        return candidates[0] if candidates else None

    output = bundle / "figures"
    primary: dict[str, Any] | None = None
    supplements: list[dict[str, Any]] = []
    written: list[Path] = []
    for geometry in GEOMETRY_ORDER:
        band = bands[geometry]
        slug = _band_slug(geometry, band)
        fixed_row = None if geometry == "pacman" and pacman_selection_pending else design_row(
            geometry, band
        )
        evolution_row = None if geometry == "pacman" and pacman_selection_pending else trajectory_row(
            geometry, band
        )

        design_filename = f"design_fields_{slug}.png"
        design_destination = output / design_filename
        design_archive = _archive(fixed_row) if fixed_row else None
        if fixed_row is not None and design_archive is not None:
            _render_design(fixed_row, design_archive, design_destination)
            design_status = "actual"
            design_cases = [str(fixed_row["id"])]
        else:
            _render_pending(design_destination, geometry, band, "design")
            design_status = "pending"
            design_cases = []
        supplements.append({
            "id": f"design_fields_{geometry}",
            "file": design_filename,
            "caption": (
                f"{geometry.replace('_', ' ').title()} design fields for torsion levels "
                f"$[{_level_text(band[0])},{_level_text(band[1])}]$: torsion, "
                "target density, and target potential.  The design is shown once; "
                "orange dashed curves mark its fixed band boundaries."
            ),
            "case_ids": design_cases,
            "stage": stage,
            "status": design_status,
            "group": "storyboard_pair",
            "geometry": geometry,
            "pair_order": 0,
        })
        written.append(design_destination)

        storyboard_filename = f"band_evolution_{slug}.png"
        storyboard_destination = output / storyboard_filename
        evolution_archive = _archive(evolution_row) if evolution_row else None
        if evolution_row is not None and evolution_archive is not None:
            working_case = is_working_trajectory(evolution_row)
            storyboard_status, storyboard_error = render_case(
                evolution_row,
                evolution_archive,
                storyboard_destination,
                geometry,
                band,
                label_failure=not working_case,
            )
            storyboard_cases = [str(evolution_row["id"])]
            if working_case:
                outcome = {
                    "strict_convergence": "strictly converged",
                    "certified_subband_convergence": "certified-subband converged",
                    "geometrically_unsuccessful_pde_converged": (
                        "PDE-converged with a retained geometric plateau"
                    ),
                    "pde_converged_with_capped_trials": (
                        "PDE-converged with retained capped trials"
                    ),
                }.get(str(evolution_row.get("classification")), "PDE-converged")
            else:
                outcome = f"failed/partial exact-band evidence ({failure_label(evolution_row)})"
            if storyboard_error:
                outcome = f"{outcome}; archive reason: {storyboard_error}"
        else:
            _render_pending(storyboard_destination, geometry, band, "storyboard")
            storyboard_status = "pending"
            storyboard_cases = []
            outcome = "awaiting an exact-band trajectory archive"
        entry = {
            "id": "band_evolution" if geometry == "smooth_star"
                  else f"band_evolution_{geometry}",
            "file": storyboard_filename,
            "caption": (
                f"{geometry.replace('_', ' ').title()} equilibrium formation for torsion "
                f"levels $[{_level_text(band[0])},{_level_text(band[1])}]$: selected "
                "seed, intermediate homotopy, early and middle accepted states, late or "
                f"restored-best state, and final projection; {outcome}.  Orange dashed curves are "
                "the fixed target band and blue solid curves are the moving equilibrium band."
            ),
            "case_ids": storyboard_cases,
            "stage": stage,
            "status": storyboard_status,
            "group": "storyboard_pair",
            "geometry": geometry,
            "pair_order": 1,
        }
        if geometry == "smooth_star":
            primary = entry
        else:
            supplements.append(entry)
        written.append(storyboard_destination)
    geometry_rank = {
        geometry: index for index, geometry in enumerate(GEOMETRY_ORDER)
    }
    failed_rows = sorted(
        (
            row for row in rows
            if str(row.get("kind", "")).startswith("trajectory")
            and is_failed_evidence(row)
        ),
        key=lambda row: (
            geometry_rank.get(str(row.get("geometry", "")), len(GEOMETRY_ORDER)),
            str(row.get("id", "")),
        ),
    )
    for evidence_index, evidence_row in enumerate(failed_rows):
        archive = _archive(evidence_row)
        geometry = str(evidence_row.get("geometry", "unknown"))
        band = evidence_band(evidence_row, archive)
        case_id = str(evidence_row.get("id", f"case-{evidence_index}"))
        filename = (
            f"failed_band_evolution_{evidence_index:03d}_"
            f"{_band_slug(geometry, band)}_{filename_token(case_id)}.png"
        )
        destination = output / filename
        attempt_result = render_attempt_storyboard(
            evidence_row,
            destination,
            trajectory=archive,
            equilibrium=_equilibrium(evidence_row),
            design_archive=_overview_archives(rows).get(geometry),
        )
        label = failure_label(evidence_row)
        warnings = attempt_result.get("warnings", [])
        if warnings:
            label = f"{label}; archive reason={warnings[0]}"
        supplements.append({
            "id": f"failed_storyboard_{filename_token(case_id)}",
            "file": filename,
            "caption": (
                f"Supplemental failed/partial trajectory evidence for "
                f"{geometry.replace('_', ' ')} at torsion levels "
                f"$[{_level_text(band[0])},{_level_text(band[1])}]$ "
                f"(case {case_id}; {label}). The case is retained regardless "
                "of PDE or geometric success. Orange dashed curves are the fixed "
                "target band and blue solid curves are the moving equilibrium band; "
                "when state recovery is impossible, the PNG records the reason."
            ),
            "case_ids": [case_id],
            "stage": stage,
            "status": attempt_result["status"],
            "group": "failed_storyboard",
            "geometry": geometry,
            "pair_order": 2,
            "failure_reason": label,
        })
        written.append(destination)

    return primary, supplements, written
