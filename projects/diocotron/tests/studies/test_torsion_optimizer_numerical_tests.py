from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import projects.diocotron.studies.torsion_optimizer.run as study_driver

from projects.diocotron.dolfinx.geometry.canonical import (
    ITER_GEO_PATH,
    canonical_geometry_name,
    generate_mesh,
    lagrange_dofs_from_metadata,
    triangle_mesh_statistics,
)
from projects.diocotron.dolfinx.runtime.mpi_rank_policy import (
    smaller_rank_within_ten_percent,
)
from projects.diocotron.studies.torsion_optimizer.metrics import (
    normalized_hausdorff,
    relative_field_errors,
    set_comparison,
    transition_resolution,
)
from projects.diocotron.studies.torsion_optimizer.schedule import (
    additional_refinement_parameters,
    fine_confirmation_parameters,
)
from projects.diocotron.studies.torsion_optimizer.cases_v3 import (
    STRICT_STORYBOARD_BANDS,
    detect_threshold_plateau,
    gradient_angle_degrees,
    mumps_expansion_is_cost_effective,
    summarize_inexact_policy,
)

from projects.diocotron.studies.torsion_optimizer.run import (
    EPSILON_RATIO,
    INEXACT_REPLAY_CSV_FIELDS,
    _best_measured_rank,
    _mpi_argv,
    aggregate_manifest,
    attach_reference_comparisons,
    append_adaptive_cases,
    build_parser,
    build_study_cases,
    calibrate_mesh_size,
    classify_status,
    completed_case_outputs_valid,
    create_trajectory_storyboard,
    deterministic_case_id,
    epsilon_invariant,
    hash_outputs,
    inexact_replay_csv_valid,
    new_manifest,
    observed_rates,
    robustness_matrix,
    select_mumps_ranks,
    select_reference,
    storyboard_confirmation_parameters,
    trajectory_frames,
    trajectory_storyboard_indices,
)


def _write_inexact_replay(
    path: Path,
    *,
    snapshots: int,
    requests: int,
    policies=("raw", "one-correction", "reassembled"),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INEXACT_REPLAY_CSV_FIELDS)
        writer.writeheader()
        for snapshot_index in range(snapshots):
            for request_index in range(requests):
                for policy in policies:
                    row = {field: "0" for field in INEXACT_REPLAY_CSV_FIELDS}
                    row.update({
                        "snapshot_id": f"snapshot-{snapshot_index}",
                        "policy": policy,
                        "requested_tolerance": 10.0 ** (-2 - request_index),
                        "outer_iteration": snapshot_index,
                        "c1": 0.1,
                        "c2": 0.2,
                        "state_residual": 1.0e-8,
                        "reference_state_residual": 1.0e-12,
                        "acceptance_decision_agrees": "True",
                    })
                    writer.writerow(row)


def test_robustness_matrix_has_exactly_100_unique_cases():
    cases = robustness_matrix()
    assert len(cases) == 100
    assert len({case["id"] for case in cases}) == 100
    assert {case["geometry"] for case in cases} == {"smooth_star", "pacman", "horseshoe", "iter"}
    for geometry in {case["geometry"] for case in cases}:
        subset = [case for case in cases if case["geometry"] == geometry]
        assert len(subset) == 25
        assert sum(case.get("anchor", False) for case in subset) == 1


def test_baseline_robustness_argv_uses_automatic_window_fit_fallback(tmp_path):
    case = robustness_matrix()[0]
    assert case["id"] == "robustness-smooth_star-f6b6828e6995"
    argv = study_driver.optimizer_argv(
        case,
        tmp_path / "mesh.msh",
        tmp_path / "run",
    )
    assert argv.count("--init-mode") == 1
    assert argv[argv.index("--init-mode") + 1] == "homotopy"
    assert "--no-include-fit-init" in argv
    assert argv.count("--init-fallback") == 1
    assert argv[argv.index("--init-fallback") + 1] == "window-fit"
    assert argv.count("--save-trajectory") == 1
    assert argv.count("--trajectory-every") == 1
    assert argv[argv.index("--trajectory-every") + 1] == "1"


def test_homotopy_tolerance_case_maps_to_optimizer_cli(tmp_path):
    case = {
        **robustness_matrix()[0],
        "kind": "trajectory_homotopy_tolerance_v3",
        "homotopy_tol_res": 1.0e-7,
    }
    argv = study_driver.optimizer_argv(
        case,
        tmp_path / "mesh.msh",
        tmp_path / "run",
    )
    assert argv.count("--homotopy-tol-res") == 1
    assert float(argv[argv.index("--homotopy-tol-res") + 1]) == pytest.approx(1.0e-7)


def test_historical_attempt_storyboard_backfill_is_png_and_nonrerendering(tmp_path):
    run_root = tmp_path / "runs"
    attempt_dir = run_root / "cases" / "failed-case" / "attempt_001"
    run_dir = attempt_dir / "run"
    run_dir.mkdir(parents=True)
    attempt = {
        "exit_code": 1,
        "error": "RuntimeError: homotopy failed",
        "run_dir": str(run_dir),
    }
    current = dict(attempt)
    case = {
        "id": "failed-case",
        "kind": "robustness",
        "geometry": "horseshoe",
        "order": 4,
        "alpha_t1": 0.4,
        "alpha_t2": 0.5,
        "state": "failed",
        "attempts": [attempt],
        "result": current,
    }
    manifest = {"version": 3, "cases": [case]}

    rendered, existing = study_driver.backfill_attempt_storyboards(
        manifest, run_root,
    )
    storyboard = attempt_dir / "storyboard.png"
    assert (rendered, existing) == (1, 0)
    assert storyboard.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert attempt["storyboard_status"] == "diagnostic_only"
    assert attempt["storyboard_source"] == "placeholder"
    assert attempt["output_hashes"]["storyboard.png"] == study_driver.sha256_file(
        storyboard
    )
    assert current["storyboard"] == str(storyboard)
    first_mtime = storyboard.stat().st_mtime_ns

    rendered, existing = study_driver.backfill_attempt_storyboards(
        manifest, run_root,
    )
    assert (rendered, existing) == (0, 1)
    assert storyboard.stat().st_mtime_ns == first_mtime


def test_initialization_policy_can_explicitly_disable_baseline_fallback(tmp_path):
    case = {
        **robustness_matrix()[0],
        "kind": "initialization_policy_v3",
        "init_fallback": "none",
    }
    argv = study_driver.optimizer_argv(
        case,
        tmp_path / "mesh.msh",
        tmp_path / "run",
    )
    fallbacks = [
        argv[index + 1]
        for index, token in enumerate(argv[:-1])
        if token == "--init-fallback"
    ]
    assert fallbacks == ["window-fit", "none"]


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("star", "smooth_star"),
        ("smooth-star", "smooth_star"),
        ("pac-man", "pacman"),
        ("horse_shoe", "horseshoe"),
        ("iter_wall", "iter"),
    ],
)
def test_geometry_aliases(alias, canonical):
    assert canonical_geometry_name(alias) == canonical


def test_iter_source_is_canonical_ff_msh_path():
    assert ITER_GEO_PATH == Path(__file__).resolve().parents[4] / "projects/diocotron/freefem/msh" / "iter.geo"
    assert ITER_GEO_PATH.is_file()


def test_triangle_metadata_and_lagrange_dof_count():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    metadata = triangle_mesh_statistics(points, triangles)
    assert metadata["area"] == pytest.approx(1.0)
    assert metadata["cells"] == 2
    assert metadata["edges"] == 5
    assert metadata["h_max"] == pytest.approx(math.sqrt(2.0))
    assert metadata["domain_diameter"] == pytest.approx(math.sqrt(2.0))
    assert lagrange_dofs_from_metadata(metadata, 1) == 4
    assert lagrange_dofs_from_metadata(metadata, 2) == 9
    assert lagrange_dofs_from_metadata(metadata, 4) == 25


def test_dof_target_calibration_reaches_tolerance_and_records_history():
    def sample(h: float) -> int:
        return round(10_000 / h**2)

    h, dofs, history = calibrate_mesh_size(40_000, sample, 1.0, tolerance=0.01)
    assert h == pytest.approx(0.5)
    assert dofs == 40_000
    assert len(history) == 2


@pytest.mark.parametrize(
    ("mesh_size", "order", "dofs", "expected"),
    [
        (0.30, 6, None, 1),
        (0.15, 5, None, 2),
        (0.075, 6, None, 4),
        (0.05, 4, None, 8),
        (0.03, 6, None, 16),
        (0.061, 2, 19_999, 1),
        (0.061, 6, 25_000, 4),
        (0.061, 4, 100_000, 4),
        (0.061, 4, 200_000, 8),
        (0.061, 4, 700_000, 8),
    ],
)
def test_mumps_rank_policy(mesh_size, order, dofs, expected):
    assert select_mumps_ranks(mesh_size, order, dofs) == expected


def test_smaller_rank_is_selected_when_timings_are_within_ten_percent():
    assert smaller_rank_within_ten_percent({4: 11.1, 8: 10.0, 12: 10.3}) == 8
    assert smaller_rank_within_ten_percent({4: 10.9, 8: 10.0, 12: 9.95}) == 4
    with pytest.raises(ValueError):
        smaller_rank_within_ten_percent({})


def test_every_case_command_is_an_mpi_launch():
    argv = _mpi_argv(4, ["python", "optimizer.py"])
    assert argv == [
        "mpirun", "--bind-to", "core", "--map-by", "core",
        "-n", "4", "python", "optimizer.py",
    ]


def test_best_measured_rank_uses_completed_matching_repeats_only():
    baseline = {
        "geometry": "smooth_star", "order": 4, "dof_target": 700_000,
        "alpha_t1": 0.6, "alpha_t2": 0.7,
    }
    cases = []
    for ranks, times in {4: (10.9, 11.0), 8: (10.0, 10.1), 12: (9.95, 10.0)}.items():
        for elapsed in times:
            cases.append({
                **baseline, "kind": "strong_scaling", "ranks": ranks,
                "state": "completed", "result": {"elapsed": elapsed},
            })
    cases.append({
        **baseline, "kind": "strong_scaling", "ranks": 20,
        "state": "failed", "result": {"elapsed": 1.0},
    })
    assert _best_measured_rank({"cases": cases}, baseline, fallback=8) == 4
    assert _best_measured_rank({"cases": []}, baseline, fallback=8) == 8


def test_parent_resolver_accepts_failed_but_pde_qualified_active_checkpoint(tmp_path):
    attempt = tmp_path / "attempt_001"
    output = attempt / "run" / "out"
    output.mkdir(parents=True)
    equilibrium = output / "equilibrium.npz"
    equilibrium.write_bytes(b"qualified checkpoint")
    summary = output / "summary.txt"
    summary.write_text(
        "bestResidual 1.5726403753760868e-13\n"
        "bestActivityArea 0.1444194772175608\n",
        encoding="utf-8",
    )
    parent = {
        "id": "screen-parent",
        "state": "failed",
        "result": {
            "run_dir": str(attempt / "run"),
            "equilibrium": str(equilibrium),
            "summary": str(summary),
            "output_hashes": hash_outputs(attempt),
        },
    }
    child = {"id": "trajectory-child", "parent_case_id": parent["id"]}
    assert study_driver._parent_equilibrium({"cases": [parent]}, child) == equilibrium


def test_completed_case_resumption_requires_intact_hashed_outputs(tmp_path):
    attempt = tmp_path / "attempt_001"
    output = attempt / "run" / "out"
    output.mkdir(parents=True)
    summary = output / "summary.txt"
    equilibrium = output / "equilibrium.npz"
    summary.write_text("status CONVERGED\n", encoding="utf-8")
    equilibrium.write_bytes(b"checkpoint")
    case = {
        "kind": "robustness",
        "state": "completed",
        "result": {
            "exit_code": 0,
            "run_dir": str(attempt / "run"),
            "summary": str(summary),
            "equilibrium": str(equilibrium),
            "output_hashes": hash_outputs(attempt),
        },
    }
    assert completed_case_outputs_valid(case)
    summary.write_text("tampered\n", encoding="utf-8")
    assert not completed_case_outputs_valid(case)


def test_inexact_replay_completion_requires_full_data_coverage(tmp_path):
    attempt = tmp_path / "attempt_001"
    run_dir = attempt / "run"
    output = run_dir / "out"
    logs = run_dir / "logs"
    output.mkdir(parents=True)
    logs.mkdir()
    summary = output / "summary.txt"
    equilibrium = output / "equilibrium.npz"
    replay = logs / "inexact_newton.csv"
    summary.write_text("status CONVERGED\n", encoding="utf-8")
    equilibrium.write_bytes(b"checkpoint")

    def completed_case():
        return {
            "kind": "inner_newton_policy_v3",
            "inner_accuracy": "adaptive",
            "state": "completed",
            "result": {
                "exit_code": 0,
                "run_dir": str(run_dir),
                "summary": str(summary),
                "equilibrium": str(equilibrium),
                "output_hashes": hash_outputs(attempt),
            },
        }

    _write_inexact_replay(replay, snapshots=0, requests=0)
    assert not inexact_replay_csv_valid(replay)
    assert not completed_case_outputs_valid(completed_case())

    _write_inexact_replay(
        replay, snapshots=4, requests=5, policies=("raw", "one-correction"),
    )
    assert not inexact_replay_csv_valid(replay)
    assert not completed_case_outputs_valid(completed_case())

    _write_inexact_replay(replay, snapshots=4, requests=4)
    assert not inexact_replay_csv_valid(replay)
    assert not completed_case_outputs_valid(completed_case())

    _write_inexact_replay(replay, snapshots=3, requests=5)
    assert not inexact_replay_csv_valid(replay)
    assert not completed_case_outputs_valid(completed_case())

    _write_inexact_replay(replay, snapshots=4, requests=5)
    assert inexact_replay_csv_valid(replay)
    assert completed_case_outputs_valid(completed_case())


def test_run_cases_immediately_rejects_header_only_adaptive_replay(
    tmp_path, monkeypatch,
):
    case = {
        "id": "adaptive-replay",
        "kind": "inner_newton_policy_v3",
        "geometry": "smooth_star",
        "order": 4,
        "dof_target": 50_000,
        "inner_accuracy": "adaptive",
        "campaigns": [],
        "state": "planned",
        "attempts": [],
    }
    manifest = {"version": 3, "cases": [case]}
    manifest_path = tmp_path / "manifest.json"
    run_root = tmp_path / "runs"
    mesh = tmp_path / "mesh.msh"
    mesh.write_bytes(b"mesh")

    monkeypatch.setattr(study_driver, "load_manifest", lambda path: manifest)
    monkeypatch.setattr(
        study_driver,
        "ensure_calibrated_mesh",
        lambda case, root, calibration: (
            mesh,
            {"requested_size": 0.3, "dofs": 1_000},
        ),
    )
    monkeypatch.setattr(
        study_driver,
        "optimizer_argv",
        lambda case, mesh_path, run_dir, initial_equilibrium=None: [
            "fake-optimizer", "--run-dir", str(run_dir),
        ],
    )
    monkeypatch.setattr(study_driver, "_git_provenance", lambda: {})
    monkeypatch.setattr(study_driver, "_package_versions", lambda: {})

    expected_run = run_root / "cases" / case["id"] / "attempt_001" / "run"

    def fake_run(argv, *, cwd, env, stdout, stderr):
        output = expected_run / "out"
        output.mkdir(parents=True)
        (output / "summary.txt").write_text(
            "finalStatus CONVERGED\nbestResidual 1e-13\n", encoding="utf-8",
        )
        (output / "equilibrium.npz").write_bytes(b"checkpoint")
        _write_inexact_replay(
            expected_run / "logs" / "inexact_newton.csv",
            snapshots=0,
            requests=0,
        )
        return 0

    monkeypatch.setattr(study_driver, "_run_interruptible", fake_run)
    args = build_parser().parse_args([
        "--manifest", str(manifest_path),
        "--run-root", str(run_root),
        "run", "--case-id", case["id"],
    ])

    assert study_driver.run_cases(args) == 0
    assert case["state"] == "failed"
    assert case["result"]["exit_code"] == 0
    assert case["result"]["validation_error"] == (
        "process exited successfully but required outputs failed semantic validation"
    )
    assert case["attempts"] == [case["result"]]
    storyboard = Path(case["result"]["storyboard"])
    assert storyboard.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert case["result"]["storyboard_status"] == "diagnostic_only"
    assert case["result"]["output_hashes"]["storyboard.png"] == study_driver.sha256_file(
        storyboard
    )


def test_run_case_prelaunch_exception_still_records_storyboard(tmp_path, monkeypatch):
    case = {
        **robustness_matrix()[0],
        "id": "prelaunch-failure",
        "state": "planned",
        "attempts": [],
        "campaigns": [],
    }
    manifest = {"version": 3, "cases": [case]}
    manifest_path = tmp_path / "manifest.json"
    run_root = tmp_path / "runs"

    monkeypatch.setattr(study_driver, "load_manifest", lambda path: manifest)

    def fail_mesh(*args, **kwargs):
        raise RuntimeError("mesh calibration failed")

    monkeypatch.setattr(study_driver, "ensure_calibrated_mesh", fail_mesh)
    args = build_parser().parse_args([
        "--manifest", str(manifest_path),
        "--run-root", str(run_root),
        "run", "--case-id", case["id"],
    ])

    assert study_driver.run_cases(args) == 0
    assert case["state"] == "failed"
    assert len(case["attempts"]) == 1
    result = case["result"]
    assert "mesh calibration failed" in result["error"]
    storyboard = Path(result["storyboard"])
    assert storyboard.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert result["storyboard_status"] == "diagnostic_only"
    assert result["output_hashes"]["storyboard.png"] == study_driver.sha256_file(storyboard)


@pytest.mark.parametrize(
    ("snapshots", "expected_state"),
    [(3, "failed"), (4, "completed")],
)
def test_load_manifest_revalidates_adaptive_replay_without_losing_evidence(
    tmp_path, monkeypatch, snapshots, expected_state,
):
    run_dir = tmp_path / f"run-{snapshots}"
    replay = run_dir / "logs" / "inexact_newton.csv"
    _write_inexact_replay(replay, snapshots=snapshots, requests=5)
    attempts = [{"attempt_id": "attempt_001", "marker": "retain-attempt"}]
    result = {
        "run_dir": str(run_dir),
        "inexact_newton": str(replay),
        "marker": "retain-result",
    }
    case = {
        "id": f"adaptive-replay-{snapshots}",
        "kind": "inner_newton_policy_v3",
        "inner_accuracy": "adaptive",
        "campaigns": [],
        "state": "completed",
        "attempts": attempts,
        "result": result,
    }
    manifest_path = tmp_path / f"manifest-{snapshots}.json"
    manifest_path.write_text(
        json.dumps({"version": 3, "cases": [case]}), encoding="utf-8",
    )
    monkeypatch.setattr(
        study_driver, "build_study_cases", lambda: [{"id": case["id"]}],
    )

    loaded = study_driver.load_manifest(manifest_path)
    loaded_case = loaded["cases"][0]
    assert loaded_case["state"] == expected_state
    assert loaded_case["attempts"] == attempts
    assert loaded_case["result"]["marker"] == "retain-result"
    assert loaded_case["result"]["run_dir"] == str(run_dir)
    if expected_state == "failed":
        assert loaded_case["result"]["validation_error"] == (
            "incomplete inexact-Newton replay coverage"
        )
    else:
        assert "validation_error" not in loaded_case["result"]


def test_status_classification_keeps_geometric_and_pde_failures_distinct():
    assert classify_status("CONVERGED", 0, 1e-13) == "strict_convergence"
    assert classify_status("CONVERGED_CERTIFIED_SUBBAND", 0, 1e-13) == "certified_subband_convergence"
    assert classify_status("MAX_OPT_IT", 0, 1e-13) == "geometrically_unsuccessful_pde_converged"
    assert classify_status("MAX_OPT_IT", 0, 1e-13, capped_trials=1) == "pde_converged_with_capped_trials"
    assert classify_status("CONVERGED", 0, 2e-12) == "geometrically_unsuccessful_pde_converged"
    assert classify_status("NEWTON_NOT_CONVERGED", 3, 6.65e-12) == "geometrically_unsuccessful_pde_converged"
    assert classify_status("NEWTON_NOT_CONVERGED", 3, 2e-11) == "pde_failure"
    assert classify_status("CONVERGED", 3, 1e-13) == "pde_failure"


def test_epsilon_invariant_and_certified_width_fraction():
    assert epsilon_invariant(0.2, 0.7, EPSILON_RATIO * 0.5)
    assert not epsilon_invariant(0.2, 0.7, 0.05)
    assert 1.0 - 2.0 * 2.0 * EPSILON_RATIO == pytest.approx(0.68)


def test_reference_selection_and_observed_rates():
    rows = [
        {"classification": "strict_convergence", "ndof": 100, "two_grid_change": 0.08},
        {"classification": "strict_convergence", "ndof": 400, "two_grid_change": 0.02},
        {"classification": "certified_subband_convergence", "ndof": 1600, "two_grid_change": 0.005},
    ]
    reference, unresolved = select_reference(rows)
    assert reference is rows[-1]
    assert not unresolved
    rates = observed_rates([0.4, 0.2, 0.1], [0.16, 0.04, 0.01])
    assert math.isnan(rates[0])
    assert rates[1:] == pytest.approx([2.0, 2.0])


def test_reference_comparisons_attach_two_grid_metrics_and_choose_best_endpoint(tmp_path):
    rows = []
    for order in (2, 4):
        for label, target, ndof in (("coarse", 100_000, 98_000), ("fine", 200_000, 201_000)):
            checkpoint = tmp_path / f"p{order}_{label}.npz"
            checkpoint.touch()
            rows.append({
                "id": f"p{order}-{label}",
                "kind": "mesh_order",
                "state": "completed",
                "classification": "strict_convergence",
                "geometry": "smooth_star",
                "order": order,
                "alpha_t1": 0.6,
                "alpha_t2": 0.7,
                "dof_target": target,
                "ndof": ndof,
                "equilibrium": str(checkpoint),
            })

    def compare(reference, current):
        if reference.stem == "p2_fine" and current.stem == "p2_coarse":
            error = 0.02
        elif reference.stem == "p4_fine" and current.stem == "p4_coarse":
            error = 0.005
        else:
            error = 0.0 if reference == current else 0.01
        return {
            "relative_l2": error,
            "relative_h1": 2.0 * error,
            "active_jaccard": 1.0 - error,
        }

    assert attach_reference_comparisons(
        rows, tmp_path / "bundle", comparison_runner=compare
    ) == 6
    p2_fine = next(row for row in rows if row["id"] == "p2-fine")
    p4_fine = next(row for row in rows if row["id"] == "p4-fine")
    assert p2_fine["two_grid_change"] == pytest.approx(0.04)
    assert p4_fine["two_grid_change"] == pytest.approx(0.01)
    assert p4_fine["is_numerical_reference"]
    assert all(row["reference_id"] == "p4-fine" for row in rows)
    assert p4_fine["relative_l2"] == 0.0


def test_field_set_contour_and_transition_metrics():
    reference = np.array([1.0, 2.0, 3.0])
    values = np.array([1.0, 2.0, 2.0])
    weights = np.array([1.0, 1.0, 2.0])
    gradients_ref = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    gradients = gradients_ref.copy()
    errors = relative_field_errors(
        reference, values, weights,
        reference_gradients=gradients_ref,
        gradients=gradients,
    )
    assert errors["relative_l2"] > 0.0
    assert errors["relative_h1"] == 0.0
    sets = set_comparison(np.array([1, 1, 0], bool), np.array([1, 0, 1], bool), weights)
    assert sets["symmetric_difference"] == pytest.approx(3.0)
    assert sets["jaccard"] == pytest.approx(0.25)
    scipy = pytest.importorskip("scipy")
    assert scipy is not None
    contour = np.array([[0.0, 0.0], [1.0, 0.0]])
    displaced = contour + np.array([0.0, 0.1])
    assert normalized_hausdorff(contour, displaced, 2.0) == pytest.approx(0.05)
    assert transition_resolution(0.08, np.array([0.1]), np.array([2.0])) == pytest.approx([0.4])


def test_manifest_ids_are_deterministic_unique_and_parser_has_all_subcommands():
    cases = build_study_cases()
    assert len(cases) == 352
    assert len({case["id"] for case in cases}) == len(cases)
    assert sum(case["kind"] == "geometry_overview" for case in cases) == 8
    assert sum(case["kind"] == "trajectory_rescue" for case in cases) == 4
    assert sum(case["kind"] == "trajectory_fit_rescue" for case in cases) == 3
    assert sum(case["kind"] == "reference_fit_rescue" for case in cases) == 3
    assert sum(case["kind"] == "trajectory" for case in cases) == 5
    assert sum(case["kind"] == "trajectory_candidate" for case in cases) == 13
    assert sum(case["kind"] == "homotopy_robustness" for case in cases) == 24
    assert sum(case["kind"] == "mumps_parent" for case in cases) == 11
    assert sum(case["kind"] == "warm_started_child" for case in cases) == 11
    assert sum(case["kind"] == "trajectory_v3" for case in cases) == 4
    assert sum(case["kind"] == "trajectory_homotopy_tolerance_v3" for case in cases) == 9
    assert sum(case["kind"] == "fixed_mesh_p" for case in cases) == 3
    assert sum(case["kind"] == "initialization_policy_v3" for case in cases) == 20
    assert sum(case["kind"] == "inner_newton_accuracy" for case in cases) == 16
    assert all(case["campaigns"] for case in cases)
    copy = dict(cases[0])
    copy["state"] = "failed"
    copy["attempts"] = [{"exit_code": 1}]
    copy["requires_non_success_of"] = "prerequisite-id"
    assert deterministic_case_id(copy) == cases[0]["id"]
    manifest = new_manifest()
    assert manifest["certified_potential_width_fraction"] == pytest.approx(0.68)
    for command in ("plan", "run", "aggregate", "figures", "movie", "validate", "report"):
        argv = [command]
        if command == "movie":
            argv.append("trajectory.npz")
        assert build_parser().parse_args(argv).command == command
    filtered = build_parser().parse_args(["run", "--geometry", "horseshoe"])
    assert filtered.geometry == ["horseshoe"]


def test_adaptive_followups_include_worst_and_both_boundary_sides_without_duplicates():
    rows = [
        {"kind": "robustness", "state": "completed", "geometry": "smooth_star",
         "classification": "strict_convergence", "alpha_t1": 0.55, "alpha_t2": 0.65,
         "bestLeakageRel": 0.01, "bestMissingRel": 0.02},
        {"kind": "robustness", "state": "completed", "geometry": "smooth_star",
         "classification": "strict_convergence", "alpha_t1": 0.45, "alpha_t2": 0.55,
         "bestLeakageRel": 0.03, "bestMissingRel": 0.04},
        {"kind": "robustness", "state": "failed", "geometry": "smooth_star",
         "classification": "pde_failure", "alpha_t1": 0.35, "alpha_t2": 0.45},
    ]
    followups = fine_confirmation_parameters(rows, fine_targets={"smooth_star": 700_000})
    bands = {(row["alpha_t1"], row["alpha_t2"]): row for row in followups}
    assert (0.45, 0.55) in bands
    assert "worst_success" in bands[(0.45, 0.55)]["adaptive_reasons"]
    assert "boundary_success_side" in bands[(0.45, 0.55)]["adaptive_reasons"]
    assert "boundary_failure_side" in bands[(0.35, 0.45)]["adaptive_reasons"]
    manifest = {"cases": []}
    assert append_adaptive_cases(manifest, rows) == len(manifest["cases"])
    assert append_adaptive_cases(manifest, rows) == 0


def test_unresolved_reference_schedules_exactly_one_extra_level():
    rows = [
        {"kind": "mesh_order", "state": "completed", "classification": "strict_convergence",
         "geometry": "pacman", "order": 4, "alpha_t1": 0.6, "alpha_t2": 0.7,
         "dof_target": 75_000, "ndof": 73_000, "two_grid_change": 0.04},
        {"kind": "mesh_order", "state": "completed", "classification": "strict_convergence",
         "geometry": "pacman", "order": 4, "alpha_t1": 0.6, "alpha_t2": 0.7,
         "dof_target": 200_000, "ndof": 198_000, "two_grid_change": 0.05},
    ]
    followups = additional_refinement_parameters(rows)
    assert len(followups) == 1
    assert followups[0]["dof_target"] == 320_000


def test_unresolved_reference_waits_for_an_already_planned_finer_level():
    rows = [
        {"kind": "mesh_order", "state": "completed", "classification": "strict_convergence",
         "geometry": "smooth_star", "order": 2, "alpha_t1": 0.6, "alpha_t2": 0.7,
         "dof_target": 50_000, "ndof": 49_000, "two_grid_change": 0.04},
        {"kind": "mesh_order", "state": "completed", "classification": "strict_convergence",
         "geometry": "smooth_star", "order": 2, "alpha_t1": 0.6, "alpha_t2": 0.7,
         "dof_target": 100_000, "ndof": 96_000, "two_grid_change": 0.05},
        {"kind": "mesh_order", "state": "planned", "classification": "incomplete",
         "geometry": "smooth_star", "order": 2, "alpha_t1": 0.6, "alpha_t2": 0.7,
         "dof_target": 200_000},
    ]
    assert additional_refinement_parameters(rows) == []


def test_storyboard_confirmation_uses_best_robustness_band_only_when_needed(tmp_path):
    archive = tmp_path / "pacman.npz"
    archive.write_bytes(b"trajectory")
    rows = [
        {"id": "horse-plateau", "kind": "robustness", "geometry": "horseshoe",
         "classification": "geometrically_unsuccessful_pde_converged", "order": 4,
         "dof_target": 200_000, "alpha_t1": 0.2, "alpha_t2": 0.3,
         "bestLeakageRel": 0.02, "bestMissingRel": 0.20, "bestResidual": 1e-13},
        {"id": "horse-certified", "kind": "robustness", "geometry": "horseshoe",
         "classification": "certified_subband_convergence", "order": 4,
         "dof_target": 200_000, "alpha_t1": 0.3, "alpha_t2": 0.4,
         "bestLeakageRel": 0.03, "bestMissingRel": 0.04, "bestResidual": 5e-13},
        {"id": "pacman-working", "kind": "trajectory_candidate", "geometry": "pacman",
         "classification": "certified_subband_convergence",
         "trajectoryArchive": str(archive)},
    ]
    followups = storyboard_confirmation_parameters(rows)
    assert len(followups) == 1
    assert followups[0]["geometry"] == "horseshoe"
    assert followups[0]["alpha_t1"] == pytest.approx(0.3)
    assert followups[0]["alpha_t2"] == pytest.approx(0.4)
    assert followups[0]["source_case_id"] == "horse-certified"


def test_unattempted_stationarity_is_incomplete_and_completed_summary_is_flattened(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps({
        "ranks": 2,
        "final_diagnostics": {
            "rho_change_l2_rel": 1.2e-5,
            "mass_rel_drift": 3.0e-13,
        },
        "equilibrium_handoff": {
            "relative_l2": 4.0e-14,
            "relative_h1": 5.0e-13,
        },
        "equilibrium_poisson_consistency": {
            "relative_l2": 2.0e-12,
            "relative_h1": 3.0e-11,
        },
    }), encoding="utf-8")
    planned = {
        "id": "planned", "kind": "stationarity", "geometry": "smooth_star",
        "order": 4, "state": "planned", "attempts": [],
    }
    completed = {
        "id": "completed", "kind": "stationarity", "geometry": "smooth_star",
        "order": 4, "state": "completed", "attempts": [],
        "result": {
            "exit_code": 0,
            "summary": str(summary_path),
            "run_dir": str(tmp_path),
            "elapsed": 2.5,
        },
    }
    rows = aggregate_manifest({"cases": [planned, completed]})
    assert rows[0]["classification"] == "incomplete"
    assert rows[1]["classification"] == "stationarity_completed"
    assert rows[1]["stationarity_rho_change_l2_rel"] == pytest.approx(1.2e-5)
    assert rows[1]["handoff_relative_h1"] == pytest.approx(5.0e-13)
    assert rows[1]["poisson_consistency_relative_h1"] == pytest.approx(3.0e-11)


def test_strict_final_projection_near_miss_is_parsed_as_pde_converged(tmp_path):
    run_dir = tmp_path / "run"
    output = run_dir / "out"
    output.mkdir(parents=True)
    summary = output / "summary.txt"
    summary.write_text(
        "finalStatus NEWTON_NOT_CONVERGED\nbestResidual 6.5e-12\n",
        encoding="utf-8",
    )
    case = {
        "id": "failed-pde", "kind": "robustness", "geometry": "pacman",
        "order": 4, "state": "failed", "attempts": [],
        "result": {
            "exit_code": 3, "summary": str(summary), "run_dir": str(run_dir),
        },
    }
    row = aggregate_manifest({"cases": [case]})[0]
    assert row["classification"] == "geometrically_unsuccessful_pde_converged"
    assert row["bestResidual"] == "6.5e-12"


def test_trajectory_storyboard_selection_and_target_once_rendering(tmp_path):
    states = [
        {"stage": "selected_seed", "c1": 0.2, "c2": 0.6, "eps_phi": 0.032,
         "homotopy_lambda": 0.0, "outer_iteration": -1},
        {"stage": "homotopy", "c1": 0.2, "c2": 0.6, "eps_phi": 0.032,
         "homotopy_lambda": 0.5, "outer_iteration": -1},
        {"stage": "accepted_outer", "c1": 0.22, "c2": 0.58, "eps_phi": 0.0288,
         "homotopy_lambda": 1.0, "outer_iteration": 0},
        {"stage": "final", "c1": 0.24, "c2": 0.56, "eps_phi": 0.0256,
         "homotopy_lambda": 1.0, "outer_iteration": -1},
    ]
    assert trajectory_storyboard_indices(states) == [0, 1, 2, 3]
    coordinates = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        [0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5], [0.5, 0.5],
    ])
    torsion = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0])
    target_phi = 0.8 * torsion
    target_rho = np.exp(-((torsion - 0.5) / 0.2) ** 2)
    states_phi = np.vstack([
        target_phi,
        target_phi * 0.98,
        target_phi * 1.01,
        target_phi,
    ])
    states_rho = np.vstack([
        target_rho,
        target_rho * 0.98,
        target_rho * 1.01,
        target_rho,
    ])
    metadata = {
        "format": "hdgfem_torsion_optimizer_trajectory_v1",
        "c1_t": 0.2,
        "c2_t": 0.7,
        "states": states,
    }
    archive = tmp_path / "trajectory.npz"
    np.savez_compressed(
        archive,
        mesh_points=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        mesh_cells=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        dof_coordinates=coordinates,
        fixed_torsion=torsion,
        fixed_target_band=np.ones_like(torsion),
        fixed_target_density=target_rho,
        fixed_target_potential=target_phi,
        states_phi=states_phi,
        states_rho=states_rho,
        states_mismatch=states_phi - target_phi,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    frames = trajectory_frames(archive, tmp_path / "frames")
    assert len(frames) == len(states)
    assert all(frame.stat().st_size > 10_000 for frame in frames)
    png = create_trajectory_storyboard(archive, tmp_path / "storyboard.png")
    assert png.stat().st_size > 10_000
    assert png.suffix == ".png"
    assert not png.with_suffix(".svg").exists()


@pytest.mark.parametrize("geometry", ["smooth_star", "pacman", "horseshoe", "iter"])
def test_canonical_geometry_gmsh_smoke(tmp_path, geometry):
    pytest.importorskip("gmsh")
    mesh_path = tmp_path / f"{geometry}.msh"
    metadata = generate_mesh(geometry, 0.6, mesh_path)
    assert mesh_path.is_file()
    assert metadata["format"] == "hdgfem_canonical_gmsh_v1"
    assert metadata["geometry"] == geometry
    assert metadata["area"] > 0.0
    assert metadata["cells"] > 0
    assert metadata["h_max"] > 0.0
    assert metadata["domain_diameter"] > 0.0
    if geometry == "iter":
        assert metadata["geometry_source"] == "projects/diocotron/freefem/msh/iter.geo"


def test_observed_findings_are_evidence_gated_and_report_uses_conda(tmp_path):
    pending = study_driver.build_observed_findings_tex([])
    assert r"\subsection{Observed behavior}" in pending
    assert "no numerical inference is made" in pending
    assert "63 accepted" not in pending

    manifest = {"cases": []}
    study_driver.write_numerical_tests_section(tmp_path, manifest)
    assert (tmp_path / "generated" / "observed_findings.tex").read_text(
        encoding="utf-8"
    ) == pending
    assert r"\input{generated/observed_findings.tex}" in (
        tmp_path / "numerical_tests_section.tex"
    ).read_text(encoding="utf-8")
    wrapper = (tmp_path / "numerical_tests.tex").read_text(encoding="utf-8")
    assert r"\usepackage{subcaption}" in wrapper
    assert r"\input{numerical_tests_section.tex}" in wrapper
    assert r"\input{successful_numerical_tests.tex}" in wrapper
    assert (tmp_path / "successful_numerical_tests.tex").is_file()
    reproduce = (tmp_path / "REPRODUCE.md").read_text(encoding="utf-8")
    assert "python -m projects.diocotron.studies.torsion_optimizer.build_report" in reproduce
    assert "latexmk -cd" not in reproduce


def test_report_escapes_figure_captions_and_flushes_large_float_sets(tmp_path):
    generated = tmp_path / "generated"
    figures = tmp_path / "figures"
    generated.mkdir()
    figures.mkdir()
    entries = []
    for index in range(9):
        filename = f"figure_{index}.png"
        (figures / filename).write_bytes(b"test")
        entries.append({
            "id": f"figure_{index}",
            "file": filename,
            "caption": f"case_with_underscore_{index}",
            "status": "failed_evidence",
        })
    (generated / "figure_registry.json").write_text(
        json.dumps({
            "format": "hdgfem_torsion_optimizer_figure_registry_v1",
            "stage": "preliminary",
            "asset_format": "png",
            "figures": entries,
        }),
        encoding="utf-8",
    )

    study_driver.write_numerical_tests_section(tmp_path, {"cases": []})
    section = (tmp_path / "numerical_tests_section.tex").read_text(encoding="utf-8")
    assert r"case\_with\_underscore\_0" in section
    assert section.count(r"\clearpage") == 1


    run_dir = tmp_path / "replay"
    replay = run_dir / "logs" / "inexact_newton.csv"
    replay.parent.mkdir(parents=True)
    with replay.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INEXACT_REPLAY_CSV_FIELDS)
        writer.writeheader()
        for policy, tolerance, state, sensitivity, gradient, step in (
            ("raw", 1.0e-5, 0.02, 0.03, 0.04, 0.05),
            ("raw", 1.0e-3, 0.20, 0.30, 0.40, 0.50),
            ("one-correction", 1.0e-3, 0.10, 0.12, 0.14, 0.16),
        ):
            row = {field: "0" for field in INEXACT_REPLAY_CSV_FIELDS}
            row.update({
                "snapshot_id": "snapshot-0",
                "policy": policy,
                "requested_tolerance": tolerance,
                "state_l2_relative_error": state / 2.0,
                "state_h1_relative_error": state,
                "sensitivity1_l2_relative_error": sensitivity,
                "sensitivity1_h1_relative_error": sensitivity,
                "sensitivity2_l2_relative_error": sensitivity,
                "sensitivity2_h1_relative_error": sensitivity,
                "reduced_gradient_relative_error": gradient,
                "threshold_step_relative_error": step,
                "acceptance_decision_agrees": "True",
            })
            writer.writerow(row)

    common = {
        "geometry": "smooth_star",
        "order": 4,
        "dof_target": 50_000,
        "alpha_t1": 0.45,
        "alpha_t2": 0.50,
    }
    iterative = {
        **common,
        "id": "iterative",
        "state": "failed",
        "difficulty": "difficult",
        "init_search": "full",
        "init_fallback": "window-fit",
        "homotopy_lambda_max_logged": 0.42,
        "homotopy_logged_accepted": 7,
        "homotopy_logged_rejected": 2,
        "homotopy_logged_newton_iterations": 31,
        "homotopy_logged_seconds": 9.5,
    }
    mumps = {
        **common,
        "id": "mumps",
        "kind": "mumps_parent",
        "state": "completed",
        "init_search": "full",
        "init_fallback": "window-fit",
        "homotopyLambdaFinal": 1.0,
        "homotopy_logged_accepted": 3,
        "homotopy_logged_rejected": 0,
        "homotopy_logged_newton_iterations": 12,
        "homotopy_logged_seconds": 1.25,
        **{
            field: "mumps"
            for field in (
                "stiffnessLinearSolver", "homotopyLinearSolver",
                "nonlinearLinearSolver", "sensitivityLinearSolver",
                "finalLinearSolver",
            )
        },
    }
    fixed = [
        {
            "id": f"fixed-p{order}",
            "kind": "fixed_mesh_p",
            "state": "completed",
            "order": order,
            "meshFile": "shared-linear-triangulation.msh",
            "ndof": dofs,
            "bestC1Phi": c1,
            "bestC2Phi": c2,
            "phase_total_seconds": seconds,
            "relative_l2": l2,
            "relative_h1": h1,
        }
        for order, dofs, c1, c2, seconds, l2, h1 in (
            (2, 100, 0.10, 0.20, 1.0, 0.01, 0.02),
            (4, 300, 0.11, 0.21, 2.0, 0.03, 0.04),
            (6, 600, 0.1101, 0.2102, 3.0, 0.025, 0.035),
        )
    ]
    storyboard = {
        "id": "pacman-storyboard",
        "kind": "trajectory_v3",
        "state": "completed",
        "algorithm_variant": "strict_storyboard_fast_fallback",
        "geometry": "pacman",
        "classification": "geometrically_unsuccessful_pde_converged",
        "bestResidual": 5.0e-13,
        "finalNewtonTolRes": 1.0e-12,
        "finalStatus": "MAX_OPT_IT",
        "bestLeakageRel": 0.2,
        "bestMissingRel": 0.3,
    }
    replay_row = {
        "id": "replay",
        "state": "completed",
        "run_dir": str(run_dir),
    }

    observed = study_driver.build_observed_findings_tex(
        [iterative, mumps, *fixed, storyboard, replay_row]
    )
    assert r"\lambda=0.42" in observed
    assert "7 accepted and 2 rejected" in observed
    assert "all-MUMPS full-search" in observed
    assert r"raw \(10^{-5}\)" in observed
    assert r"one-correction \(10^{-3}\)" in observed
    assert r"\emph{model-feasibility proxy}" in observed
    assert "identical linear triangulation" in observed
    assert "non-monotonicity" in observed
    assert "does not show that higher \\(p\\) is intrinsically worse" in observed
    assert "Non-star storyboard outcomes" in observed
    assert r"MAX\_OPT\_IT" in observed


@pytest.mark.skipif(
    os.environ.get("HDGFEM_RUN_DOLFINX_INTEGRATION") != "1",
    reason="set HDGFEM_RUN_DOLFINX_INTEGRATION=1 for MPI/DOLFINx integration tests",
)
def test_two_rank_optimizer_trajectory_and_supg_checkpoint_handoff(tmp_path):
    pytest.importorskip("dolfinx")
    pytest.importorskip("gmsh")
    if shutil.which("mpirun") is None:
        pytest.skip("mpirun is unavailable")
    repo = Path(__file__).resolve().parents[4]
    mesh_path = tmp_path / "star.msh"
    generate_mesh("smooth_star", 0.8, mesh_path)
    optimizer_run = tmp_path / "optimizer"
    optimizer = repo / "projects/diocotron/dolfinx/torsion/optimization/homotopy.py"
    command = [
        "mpirun", "--bind-to", "core", "--map-by", "core", "-n", "2",
        sys.executable, str(optimizer), "--run-dir", str(optimizer_run),
        "--mesh", str(mesh_path), "--order", "2", "--quad-degree", "12",
        "--init-hminus1-grid", "2", "--init-hminus1-refine-grid", "2",
        "--init-hminus1-refine-passes", "0", "--max-opt-it", "0",
        "--tol-res", "1e-7", "--final-newton-tol-res", "1e-7",
        "--run-inexact-newton-study", "--inexact-newton-tolerances", "1e-3",
        "--inexact-newton-reference-tol", "1e-7",
        "--inexact-newton-max-snapshots", "1",
        "--save-trajectory", "--save-frames", "--plot-off-screen",
        "--no-plot-optimization",
    ]
    optimizer_completed = subprocess.run(
        command, cwd=repo, check=True, timeout=300, text=True, capture_output=True,
    )
    assert "PLOT_MPI_GRID ranks=2" in optimizer_completed.stdout
    assert "complete=1" in optimizer_completed.stdout
    assert (optimizer_run / "logs" / "newton.csv").is_file()
    assert (optimizer_run / "logs" / "phases.csv").is_file()
    inexact_path = optimizer_run / "logs" / "inexact_newton.csv"
    assert inexact_path.is_file()
    with inexact_path.open(newline="", encoding="utf-8") as handle:
        inexact_rows = list(csv.DictReader(handle))
    assert [row["policy"] for row in inexact_rows] == [
        "raw", "one-correction", "reassembled",
    ]
    assert all(float(row["state_solve_time"]) >= 0.0 for row in inexact_rows)
    assert all(float(row["sensitivity_solve_time"]) >= 0.0 for row in inexact_rows)
    assert "INEXACT_REPLAY status=DONE" in optimizer_completed.stdout
    frames = sorted((optimizer_run / "frames").glob("*.png"))
    assert len(frames) == 2
    for frame in frames:
        png = frame.read_bytes()
        assert len(png) > 10_000
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        assert int.from_bytes(png[16:20], "big") == 1800
        assert int.from_bytes(png[20:24], "big") == 700
    trajectory = optimizer_run / "out" / "trajectory.npz"
    equilibrium = optimizer_run / "out" / "equilibrium.npz"
    assert trajectory.is_file() and equilibrium.is_file()
    with np.load(trajectory, allow_pickle=False) as archive:
        assert archive["mesh_cells"].shape[1] == 3
        assert archive["states_phi"].shape == archive["states_mismatch"].shape
        state_metadata = json.loads(str(archive["metadata"].item()))
        assert state_metadata["states"][0]["stage"] == "selected_seed"
        assert state_metadata["states"][-1]["stage"] == "final"

    dynamics_run = tmp_path / "dynamics"
    dynamics = repo / "projects/diocotron/dolfinx/guiding_center/supg.py"
    subprocess.run([
        "mpirun", "--bind-to", "core", "--map-by", "core", "-n", "2",
        sys.executable, str(dynamics), "--run-dir", str(dynamics_run),
        "--equilibrium", str(equilibrium), "--order", "2", "--dt", "0.025",
        "--allow-nonconverged-equilibrium",
        "--num-steps", "1", "--supg-scale", "0.1", "--supg-tau-mode", "transient",
        "--flux-stabilization", "0", "--no-plot",
    ], cwd=repo, check=True, timeout=300)
    assert (dynamics_run / "diagnostics.csv").is_file()
    summary = json.loads((dynamics_run / "summary.json").read_text(encoding="utf-8"))
    assert summary["equilibrium"] == str(equilibrium)
    assert summary["equilibrium_handoff"]["relative_l2"] < 1.0e-10
    assert summary["equilibrium_handoff"]["relative_h1"] < 1.0e-9
    assert summary["equilibrium_poisson_consistency"]["relative_h1"] < 1.0e-5
