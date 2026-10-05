from __future__ import annotations

import math
from pathlib import Path

import pytest

from projects.diocotron.studies.torsion_optimizer.cases_v3 import (
    HORSESHOE_BRANCH_CONTINUATION_BANDS,
    HORSESHOE_BRANCH_CONTINUATION_PARENT,
    STRICT_STORYBOARD_BANDS,
    detect_threshold_plateau,
    gradient_angle_degrees,
    mumps_expansion_is_cost_effective,
    summarize_inexact_policy,
)
from projects.diocotron.studies.torsion_optimizer.run import (
    build_parser,
    build_study_cases,
    optimizer_argv,
    select_mumps_ranks,
)


def test_strict_storyboard_bands_and_parent_links_are_cost_gated():
    assert all(0.0 < a2 - a1 < 0.1 for a1, a2 in STRICT_STORYBOARD_BANDS.values())
    assert STRICT_STORYBOARD_BANDS["pacman"] == pytest.approx((0.60, 0.695))
    cases = build_study_cases()
    parents = {case["id"]: case for case in cases if case["kind"] == "mumps_parent"}
    children = [case for case in cases if case["kind"] == "warm_started_child"]
    assert len(parents) == len(children) == 11
    assert all(child["parent_case_id"] in parents for child in children)
    assert all(child["init_search"] == "fast" for child in children)
    assert all(child["init_fallback"] == "window-fit" for child in children)


def test_fixed_mesh_p_cases_share_one_p2_calibrated_triangulation():
    cases = [case for case in build_study_cases() if case["kind"] == "fixed_mesh_p"]
    assert {case["order"] for case in cases} == {2, 4, 6}
    assert {case["mesh_reference_order"] for case in cases} == {2}
    assert {case["dof_target"] for case in cases} == {50_000}


def test_horseshoe_band_screen_is_isolated_deterministic_mumps_pilot(tmp_path):
    all_cases = build_study_cases()
    screens = [
        case for case in all_cases
        if case["kind"] == "horseshoe_band_screen_v3"
    ]
    assert len(screens) == 1
    case = screens[0]
    assert case["id"] == "horseshoe_band_screen_v3-horseshoe-141f7407f3cd"
    assert case["geometry"] == "horseshoe"
    assert case["order"] == 4
    assert case["dof_target"] == 75_000
    assert (case["alpha_t1"], case["alpha_t2"]) == pytest.approx((0.40, 0.50))
    assert case["campaigns"] == ["v3_pilot"]
    assert case["max_opt_it"] == 10
    assert sum(item["kind"] == "robustness" for item in all_cases) == 100
    assert sum(item["kind"] == "trajectory_v3" for item in all_cases) == 4

    argv = optimizer_argv(case, Path(tmp_path / "mesh.msh"), tmp_path / "run")
    assert argv[argv.index("--linear-solver") + 1] == "mumps"
    assert argv[argv.index("--init-search") + 1] == "full"
    fallback_indices = [
        index for index, token in enumerate(argv[:-1])
        if token == "--init-fallback"
    ]
    assert argv[fallback_indices[-1] + 1] == "window-fit"
    assert argv[argv.index("--max-opt-it") + 1] == "10"
    assert "--no-include-fit-init" in argv
    assert "--save-trajectory" not in argv
    assert select_mumps_ranks(None, case["order"], case["dof_target"]) == 4


def test_horseshoe_band_continuation_is_parented_noncanonical_trajectory(tmp_path):
    all_cases = build_study_cases()
    candidates = [
        case for case in all_cases
        if case["kind"] == "trajectory_horseshoe_band_candidate_v3"
    ]
    assert len(candidates) == 1
    case = candidates[0]
    assert case["id"] == "trajectory_horseshoe_band_candidate_v3-horseshoe-dfba417fa450"
    assert case["parent_case_id"] == "horseshoe_band_screen_v3-horseshoe-141f7407f3cd"
    assert (case["alpha_t1"], case["alpha_t2"]) == pytest.approx((0.40, 0.50))
    assert case["campaigns"] == ["v3_pilot"]
    assert STRICT_STORYBOARD_BANDS["horseshoe"] == pytest.approx((0.20, 0.295))

    checkpoint = tmp_path / "parent-equilibrium.npz"
    argv = optimizer_argv(
        case,
        Path(tmp_path / "mesh.msh"),
        tmp_path / "run",
        initial_equilibrium=checkpoint,
    )
    assert "--save-trajectory" in argv
    assert argv[argv.index("--trajectory-every") + 1] == "1"
    assert argv[argv.index("--linear-solver") + 1] == "mumps"
    assert argv[argv.index("--max-opt-it") + 1] == "30"
    assert argv[argv.index("--init-search") + 1] == "fast"
    fallback_indices = [
        index for index, token in enumerate(argv[:-1])
        if token == "--init-fallback"
    ]
    assert argv[fallback_indices[-1] + 1] == "window-fit"
    assert argv[argv.index("--initial-equilibrium") + 1] == str(checkpoint)
    assert select_mumps_ranks(None, case["order"], case["dof_target"]) == 4


def test_horseshoe_cross_target_branch_screens_are_ordered_mumps_diagnostics(tmp_path):
    all_cases = build_study_cases()
    screens = [
        case for case in all_cases
        if case["kind"] == "trajectory_horseshoe_branch_continuation_v3"
    ]
    assert [
        (case["alpha_t1"], case["alpha_t2"])
        for case in screens
    ] == list(HORSESHOE_BRANCH_CONTINUATION_BANDS)
    assert len({case["id"] for case in screens}) == len(screens) == 5
    assert all(case["parent_case_id"] == HORSESHOE_BRANCH_CONTINUATION_PARENT
               for case in screens)
    assert all(case["algorithm_variant"]
               == "pde_exact_cross_target_branch_continuation_diagnostic"
               for case in screens)
    assert all(case["campaigns"] == ["v3_pilot"] for case in screens)
    assert all(case["order"] == 4 and case["dof_target"] == 75_000
               for case in screens)
    assert all(case["linear_solver"] == "mumps" for case in screens)
    assert all(case["max_opt_it"] == 20 and case["trajectory_every"] == 1
               for case in screens)
    assert STRICT_STORYBOARD_BANDS["horseshoe"] == pytest.approx((0.20, 0.295))

    checkpoint = tmp_path / "parent-equilibrium.npz"
    for case in screens:
        argv = optimizer_argv(
            case,
            Path(tmp_path / "mesh.msh"),
            tmp_path / case["id"],
            initial_equilibrium=checkpoint,
        )
        assert "--save-trajectory" in argv
        assert argv[argv.index("--trajectory-every") + 1] == "1"
        assert argv[argv.index("--linear-solver") + 1] == "mumps"
        assert argv[argv.index("--max-opt-it") + 1] == "20"
        assert argv[argv.index("--initial-equilibrium") + 1] == str(checkpoint)
        assert select_mumps_ranks(None, case["order"], case["dof_target"]) == 4


def test_iter_window_fit_high_newton_retry_is_low_dof_mumps_and_trajectory(tmp_path):
    matches = [
        case for case in build_study_cases()
        if case["kind"] == "trajectory_iter_window_fit_high_newton_v3"
    ]
    assert len(matches) == 1
    case = matches[0]
    assert case["geometry"] == "iter"
    assert (case["alpha_t1"], case["alpha_t2"]) == pytest.approx((0.60, 0.70))
    assert case["order"] == 4
    assert case["dof_target"] == 75_000
    assert case["linear_solver"] == "mumps"
    assert case["initialization_method"] == "legacy_fit_window_newton"
    assert case["max_newton_it"] == 160
    assert case["final_newton_max_it"] == 400
    assert case["trajectory_every"] == 1
    argv = optimizer_argv(
        case,
        Path(tmp_path / "iter.msh"),
        tmp_path / "run",
    )
    assert argv[argv.index("--linear-solver") + 1] == "mumps"
    assert argv[argv.index("--max-newton-it") + 1] == "160"
    assert argv[argv.index("--final-newton-max-it") + 1] == "400"
    init_mode_indices = [
        index for index, token in enumerate(argv[:-1])
        if token == "--init-mode"
    ]
    assert init_mode_indices
    assert argv[init_mode_indices[-1] + 1] == "legacy"
    assert "--include-fit-init" in argv
    assert "--legacy-project-preselected-only" in argv
    assert "--save-trajectory" in argv
    assert argv[argv.index("--trajectory-every") + 1] == "1"
    assert select_mumps_ranks(None, case["order"], case["dof_target"]) == 4


def test_v3_campaigns_are_exposed_by_the_cli():
    for campaign in ("v3_pilot", "v3_full"):
        args = build_parser().parse_args(["run", "--campaign", campaign, "--dry-run"])
        assert args.campaign == campaign


def test_plateau_detector_uses_accepted_states_and_requires_nonzero_gradient():
    records = []
    for k in range(7):
        records.append({
            "k": k,
            "accepted": 0 if k == 1 else 1,
            "c1Phi": 0.2 + min(k, 2) * 1.0e-8,
            "c2Phi": 0.4 + min(k, 2) * 1.0e-8,
            "leakageRel": 0.05,
            "missingRel": 0.05,
            "projectedGradNorm": 1.0e-3,
        })
    result = detect_threshold_plateau(records, gradient_tolerance=1.0e-8)
    assert result["threshold_plateau"]
    assert result["plateau_type"] == "stalled"
    assert result["plateau_gradient"] == pytest.approx(1.0e-3)


def test_inexact_policy_gate_checks_gradient_step_and_decision_errors():
    records = [{
        "reduced_gradient_relative_error": 0.04,
        "reduced_gradient_angle_degrees": 2.0,
        "trust_step_relative_error": 0.06,
        "decision_agrees": True,
        "runtime": 0.5,
    } for _ in range(20)]
    assert summarize_inexact_policy(records)["admissible"]
    records[-1] = {**records[-1], "reduced_gradient_angle_degrees": 8.0}
    assert not summarize_inexact_policy(records)["admissible"]
    assert gradient_angle_degrees((1.0, 0.0), (0.0, 1.0)) == pytest.approx(90.0)
    assert gradient_angle_degrees((1.0, 0.0), (1.0, 0.0)) == pytest.approx(0.0)


def test_mumps_parent_expansion_rule_accepts_cost_or_rescued_success():
    assert mumps_expansion_is_cost_effective(
        parent_seconds=10.0,
        child_seconds=88.0,
        standalone_seconds=100.0,
        child_success=True,
        standalone_success=True,
    )
    assert not mumps_expansion_is_cost_effective(
        parent_seconds=30.0,
        child_seconds=100.0,
        standalone_seconds=100.0,
        child_success=True,
        standalone_success=True,
    )
    assert mumps_expansion_is_cost_effective(
        parent_seconds=30.0,
        child_seconds=100.0,
        standalone_seconds=100.0,
        child_success=True,
        standalone_success=False,
    )
    with pytest.raises(ValueError):
        mumps_expansion_is_cost_effective(
            parent_seconds=1.0,
            child_seconds=1.0,
            standalone_seconds=0.0,
            child_success=False,
            standalone_success=False,
        )


def test_gradient_angle_rejects_mismatched_vectors():
    with pytest.raises(ValueError):
        gradient_angle_degrees((1.0,), (1.0, 2.0))
    assert math.isfinite(gradient_angle_degrees((1.0, 1.0), (1.0, 1.0)))
