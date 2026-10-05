from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import projects.diocotron.studies.torsion_optimizer.figures.storyboards as storyboards


def _archive(path: Path, alpha_t1: float = 0.15, alpha_t2: float = 0.20) -> Path:
    axis = np.linspace(0.0, 1.0, 7)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    coordinates = np.column_stack((xx.ravel(), yy.ravel()))
    cells = []
    width = len(axis)
    for j in range(width - 1):
        for i in range(width - 1):
            lower = j * width + i
            cells.extend((
                (lower, lower + 1, lower + width + 1),
                (lower, lower + width + 1, lower + width),
            ))
    cells = np.asarray(cells, dtype=np.int64)
    torsion = 16.0 * coordinates[:, 0] * (1.0 - coordinates[:, 0])
    torsion *= coordinates[:, 1] * (1.0 - coordinates[:, 1])
    c1_t, c2_t = alpha_t1 * np.max(torsion), alpha_t2 * np.max(torsion)
    target_density = ((torsion > c1_t) & (torsion < c2_t)).astype(float)
    target_potential = torsion * 0.8
    states = [
        {"stage": "selected_seed", "c1": 0.1, "c2": 0.3,
         "homotopy_lambda": 0.0, "outer_iteration": -1},
        {"stage": "homotopy", "c1": 0.1, "c2": 0.3,
         "homotopy_lambda": 0.5, "outer_iteration": -1},
        {"stage": "accepted_outer", "c1": 0.11, "c2": 0.29,
         "homotopy_lambda": 1.0, "outer_iteration": 0},
        {"stage": "accepted_outer", "c1": 0.12, "c2": 0.28,
         "homotopy_lambda": 1.0, "outer_iteration": 2},
        {"stage": "restored_best", "c1": 0.12, "c2": 0.28,
         "homotopy_lambda": 1.0, "outer_iteration": 2},
        {"stage": "final", "c1": 0.12, "c2": 0.28,
         "homotopy_lambda": 1.0, "outer_iteration": -1},
    ]
    phi = np.vstack([target_potential * factor for factor in (0.8, 0.9, 0.95, 1, 1, 1)])
    rho = np.vstack([target_density * factor for factor in (0.8, 0.9, 0.95, 1, 1, 1)])
    metadata = {
        "alpha_t1": alpha_t1,
        "alpha_t2": alpha_t2,
        "c1_t": c1_t,
        "c2_t": c2_t,
        "states": states,
        "terminal_status": "CONVERGED",
    }
    np.savez_compressed(
        path,
        mesh_points=coordinates,
        mesh_cells=cells,
        dof_coordinates=coordinates,
        fixed_torsion=torsion,
        fixed_target_band=target_density,
        fixed_target_density=target_density,
        fixed_target_potential=target_potential,
        states_phi=phi,
        states_rho=rho,
        states_mismatch=phi - target_potential[None, :],
        metadata=np.asarray(json.dumps(metadata)),
    )
    return path


def test_pacman_band_selection_is_deterministic_and_strict(tmp_path):
    expected = {
        "smooth_star": (0.45, 0.50),
        "pacman": (0.60, 0.695),
        "horseshoe": (0.20, 0.295),
        "iter": (0.25, 0.345),
    }
    assert storyboards.select_storyboard_bands([]) == expected

    rows = []
    for geometry in storyboards.GEOMETRY_ORDER:
        archive = _archive(tmp_path / f"{geometry}.npz")
        rows.append({
            "id": f"overview-{geometry}",
            "kind": "geometry_overview",
            "geometry": geometry,
            "state": "completed",
            "dof_target": 200_000,
            "trajectoryArchive": str(archive),
        })
    first = storyboards.select_pacman_storyboard_band(rows)
    second = storyboards.select_pacman_storyboard_band(list(reversed(rows)))
    assert first == second
    assert first == expected["pacman"]
    assert any(
        first[0] == pytest.approx(candidate)
        for candidate in storyboards.PACMAN_LOWER_LEVELS
    )
    assert first[1] - first[0] == pytest.approx(0.095)
    assert first[1] - first[0] < 0.1


def test_publication_pairs_require_exact_band_pde_convergence(
        tmp_path, monkeypatch,
):
    bands = storyboards.select_storyboard_bands([])
    rows = []
    for geometry in storyboards.GEOMETRY_ORDER:
        archive = _archive(tmp_path / f"overview-{geometry}.npz", *bands[geometry])
        rows.append({
            "id": f"overview-{geometry}",
            "kind": "geometry_overview",
            "geometry": geometry,
            "state": "completed",
            "dof_target": 200_000,
            "alpha_t1": bands[geometry][0],
            "alpha_t2": bands[geometry][1],
            "trajectoryArchive": str(archive),
        })
    bands = storyboards.select_storyboard_bands(rows)
    for geometry in storyboards.GEOMETRY_ORDER:
        archive = _archive(tmp_path / f"trajectory-{geometry}.npz", *bands[geometry])
        rows.append({
            "id": f"trajectory-{geometry}",
            "kind": "trajectory_v3",
            "geometry": geometry,
            "state": "completed",
            "classification": "certified_subband_convergence",
            "dof_target": 200_000,
            "alpha_t1": bands[geometry][0],
            "alpha_t2": bands[geometry][1],
            "trajectoryArchive": str(archive),
        })
    wrong_archive = _archive(tmp_path / "horse-wrong.npz", 0.60, 0.70)
    rows.append({
        "id": "horse-wrong",
        "kind": "trajectory",
        "geometry": "horseshoe",
        "state": "completed",
        "classification": "certified_subband_convergence",
        "dof_target": 200_000,
        "alpha_t1": bands["horseshoe"][0],
        "alpha_t2": bands["horseshoe"][1],
        "trajectoryArchive": str(wrong_archive),
    })

    def fake_render(*args, **kwargs):
        destination = next(
            value for value in args
            if isinstance(value, Path) and value.suffix == ".png"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"png")

    monkeypatch.setattr(storyboards, "_render_design", fake_render)
    monkeypatch.setattr(storyboards, "_render_storyboard", fake_render)
    monkeypatch.setattr(storyboards, "_render_pending", fake_render)
    primary, supplements, assets = storyboards.create_trajectory_storyboards(
        tmp_path / "bundle", rows, "preliminary",
    )
    assert primary is not None
    assert primary["geometry"] == "smooth_star"
    assert primary["status"] == "actual"
    assert len(supplements) == 7
    paired = [*supplements, primary]
    assert {item["geometry"] for item in paired} == set(storyboards.GEOMETRY_ORDER)
    assert sum(item["status"] == "actual" for item in paired) == 8
    for geometry in storyboards.GEOMETRY_ORDER:
        design = next(item for item in paired if item["geometry"] == geometry
                      and item["pair_order"] == 0)
        assert design["status"] == "actual"
        assert design["case_ids"] == [f"overview-{geometry}"]
        evolution = next(item for item in paired if item["geometry"] == geometry
                         and item["pair_order"] == 1)
        assert evolution["status"] == "actual"
        assert evolution["case_ids"] == [f"trajectory-{geometry}"]
    assert len(assets) == 8
    assert all(asset.suffix == ".png" for asset in assets)


def test_renderers_separate_design_and_hide_mesh_edges(tmp_path, monkeypatch):
    band = storyboards.FIXED_STORYBOARD_BANDS["smooth_star"]
    archive = _archive(tmp_path / "render.npz", *band)
    row = {
        "id": "render",
        "kind": "trajectory_v3",
        "geometry": "smooth_star",
        "state": "completed",
        "classification": "strict_convergence",
        "alpha_t1": band[0],
        "alpha_t2": band[1],
    }

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    contours = []
    fills = []
    saved_dpis = []
    original_contour = Axes.tricontour
    original_fill = Axes.tricontourf

    def record_contour(self, *args, **kwargs):
        contours.append(dict(kwargs))
        return original_contour(self, *args, **kwargs)

    def record_fill(self, *args, **kwargs):
        fills.append(dict(kwargs))
        return original_fill(self, *args, **kwargs)

    def forbid_triplot(*args, **kwargs):
        raise AssertionError("storyboards must not draw mesh edges")

    def fake_savefig(self, destination, **kwargs):
        saved_dpis.append(kwargs.get("dpi"))
        Path(destination).write_bytes(b"png")

    monkeypatch.setattr(Axes, "tricontour", record_contour)
    monkeypatch.setattr(Axes, "tricontourf", record_fill)
    monkeypatch.setattr(Axes, "triplot", forbid_triplot)
    monkeypatch.setattr(Figure, "savefig", fake_savefig)

    design = tmp_path / "design.png"
    evolution = tmp_path / "evolution.png"
    storyboards._render_design(row, archive, design)
    design_contours = list(contours)
    assert len(fills) == 3
    fills.clear()
    contours.clear()
    storyboards._render_storyboard(row, archive, evolution)

    assert len(fills) == 18
    fills.clear()
    contours.clear()
    combined = tmp_path / "attempt.png"
    storyboards._render_storyboard(
        row, archive, combined, include_design=True,
    )

    assert len(fills) == 21
    assert saved_dpis == [300, 300, 300]
    assert any(call.get("colors") == "#e68613"
               and call.get("linestyles") == "--" for call in design_contours)
    assert not any(call.get("colors") == "#1769aa" for call in design_contours)
    assert any(call.get("colors") == "#e68613"
               and call.get("linestyles") == "--" for call in contours)


@pytest.mark.parametrize("terminal_stage", ["final", "restored_best"])
def test_frame_indices_without_homotopy_are_chronological_and_terminal_last(
        terminal_stage,
):
    accepted_iterations = [9, 2, 29, 4, 14, 3, 25, 6, 20]
    states = [
        {
            "stage": "checkpoint_transfer",
            "homotopy_lambda": 1.0,
            "outer_iteration": -1,
        },
        *[
            {
                "stage": "accepted_outer",
                "homotopy_lambda": 1.0,
                "outer_iteration": iteration,
            }
            for iteration in accepted_iterations
        ],
        {
            "stage": terminal_stage,
            "homotopy_lambda": 1.0,
            "outer_iteration": -1,
        },
    ]

    indices = storyboards._frame_indices(states)
    selected = [states[index] for index in indices]
    selected_accepted_iterations = [
        int(state["outer_iteration"])
        for state in selected
        if state["stage"] == "accepted_outer"
    ]

    assert len(indices) == 6
    assert selected[0]["stage"] == "checkpoint_transfer"
    assert selected_accepted_iterations == [2, 4, 14, 29]
    assert selected_accepted_iterations == sorted(selected_accepted_iterations)
    assert indices[-1] == len(states) - 1
    assert selected[-1]["stage"] == terminal_stage


def test_failed_homotopy_frames_preserve_attempt_order_and_terminal_last():
    states = [
        {"stage": "selected_seed", "homotopy_lambda": 0.0},
        {"stage": "homotopy", "homotopy_lambda": 0.10},
        {"stage": "homotopy", "homotopy_lambda": 0.19},
        {"stage": "primary_failure", "homotopy_lambda": 0.19},
        {"stage": "window_fit_seed", "homotopy_lambda": 0.0},
        {"stage": "homotopy", "homotopy_lambda": 0.10},
        {"stage": "homotopy", "homotopy_lambda": 0.16},
        {"stage": "fallback_terminal_failure", "homotopy_lambda": 0.16},
    ]

    indices = storyboards._frame_indices(states)

    assert indices == sorted(indices)
    assert indices[-1] == len(states) - 1
    assert states[indices[-1]]["stage"] == "fallback_terminal_failure"
    assert any(states[index]["stage"] == "window_fit_seed" for index in indices)


def test_attempt_storyboard_always_writes_diagnostic_png(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    def fake_savefig(self, destination, **kwargs):
        assert kwargs["dpi"] == 300
        Path(destination).write_bytes(b"png")

    monkeypatch.setattr(Figure, "savefig", fake_savefig)
    destination = tmp_path / "attempt.pdf"
    result = storyboards.render_attempt_storyboard(
        {
            "id": "infrastructure-failure",
            "attempt_id": "attempt_002",
            "kind": "robustness",
            "geometry": "iter",
            "state": "failed",
            "error": "MPI launcher failed before output creation",
            "exit_code": None,
        },
        destination,
    )

    assert result["status"] == "diagnostic_only"
    assert result["source"] == "placeholder"
    assert Path(result["file"]) == destination.with_suffix(".png")
    assert destination.with_suffix(".png").read_bytes() == b"png"
    assert not destination.exists()


def test_attempt_storyboard_prefers_recoverable_failed_trajectory(
        tmp_path, monkeypatch,
):
    archive = _archive(tmp_path / "partial.npz", 0.45, 0.50)
    destination = tmp_path / "storyboard.png"
    calls = []

    def fake_storyboard(row, source, target, **kwargs):
        calls.append((source, kwargs))
        target.write_bytes(b"trajectory")

    monkeypatch.setattr(storyboards, "_render_storyboard", fake_storyboard)
    result = storyboards.render_attempt_storyboard(
        {
            "id": "partial",
            "kind": "trajectory_v3",
            "geometry": "smooth_star",
            "state": "failed",
            "classification": "pde_failure",
            "trajectory": str(archive),
        },
        destination,
    )

    assert result["status"] == "actual"
    assert result["source"] == "trajectory"
    assert calls == [(archive, {
        "failure_label": "state=failed; classification=pde_failure",
        "include_design": True,
    })]
    assert destination.read_bytes() == b"trajectory"


def test_attempt_storyboard_falls_back_to_final_equilibrium(
        tmp_path, monkeypatch,
):
    invalid_trajectory = tmp_path / "trajectory.npz"
    invalid_trajectory.write_bytes(b"not-an-npz")
    equilibrium = tmp_path / "equilibrium.npz"
    equilibrium.write_bytes(b"checkpoint")
    destination = tmp_path / "storyboard.png"
    calls = []

    def fail_storyboard(*args, **kwargs):
        raise ValueError("partial archive has no usable states")

    def fake_equilibrium(row, source, target, **kwargs):
        calls.append((source, kwargs))
        target.write_bytes(b"equilibrium")

    monkeypatch.setattr(storyboards, "_render_storyboard", fail_storyboard)
    monkeypatch.setattr(storyboards, "_render_equilibrium_attempt", fake_equilibrium)
    result = storyboards.render_attempt_storyboard(
        {
            "id": "checkpoint-only",
            "kind": "robustness",
            "geometry": "horseshoe",
            "state": "failed",
            "classification": "geometric_failure",
        },
        destination,
        trajectory=invalid_trajectory,
        equilibrium=equilibrium,
    )

    assert result["status"] == "checkpoint_only"
    assert result["source"] == "equilibrium"
    assert calls[0][0] == equilibrium
    assert calls[0][1]["failure_label"] == (
        "state=failed; classification=geometric_failure"
    )
    assert "partial archive has no usable states" in result["warnings"][0]
    assert destination.read_bytes() == b"equilibrium"
