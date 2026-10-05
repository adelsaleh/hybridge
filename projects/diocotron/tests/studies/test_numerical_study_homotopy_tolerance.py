from __future__ import annotations

from pathlib import Path

import pytest

from projects.diocotron.studies.torsion_optimizer.initialization_cases import (
    HOMOTOPY_TOLERANCE_LEVELS,
    homotopy_tolerance_cases,
)
from projects.diocotron.studies.torsion_optimizer.cases_v3 import STRICT_STORYBOARD_BANDS
from projects.diocotron.studies.torsion_optimizer.run import (
    build_study_cases,
    make_case,
    optimizer_argv,
    select_mumps_ranks,
)


EXPECTED_IDS = {
    "trajectory_homotopy_tolerance_v3-smooth_star-6befe58e8b70",
    "trajectory_homotopy_tolerance_v3-smooth_star-90988afeef54",
    "trajectory_homotopy_tolerance_v3-smooth_star-f029111e143f",
    "trajectory_homotopy_tolerance_v3-smooth_star-667b42e5844b",
    "trajectory_homotopy_tolerance_v3-smooth_star-776e06d5fc23",
    "trajectory_homotopy_tolerance_v3-smooth_star-3d948ec9d961",
    "trajectory_homotopy_tolerance_v3-horseshoe-d029b6ba1cb4",
    "trajectory_homotopy_tolerance_v3-horseshoe-1fe114c7e89b",
    "trajectory_homotopy_tolerance_v3-horseshoe-82108791db49",
}


def test_homotopy_tolerance_matrix_is_nine_controlled_low_cost_cases():
    cases = homotopy_tolerance_cases(make_case, STRICT_STORYBOARD_BANDS)

    assert len(cases) == 9
    assert {case["id"] for case in cases} == EXPECTED_IDS
    assert {case["homotopy_tol_res"] for case in cases} == {
        tolerance for _, tolerance in HOMOTOPY_TOLERANCE_LEVELS
    }
    assert len({case["diagnostic_label"] for case in cases}) == 9
    assert all(case["kind"] == "trajectory_homotopy_tolerance_v3" for case in cases)
    assert all(case["linear_solver"] == "mumps" for case in cases)
    assert all(case["max_opt_it"] == 0 for case in cases)
    assert all(case["final_newton_tol_res"] == pytest.approx(1.0e-12)
               for case in cases)
    assert {
        (case["geometry"], case["alpha_t1"], case["alpha_t2"], case["dof_target"])
        for case in cases
    } == {
        ("smooth_star", 0.60, 0.70, 50_000),
        ("smooth_star", 0.45, 0.50, 50_000),
        ("horseshoe", 0.20, 0.295, 75_000),
    }


def test_homotopy_tolerance_cases_are_integrated_and_map_to_optimizer_argv(tmp_path):
    cases = [case for case in build_study_cases()
             if case["kind"] == "trajectory_homotopy_tolerance_v3"]
    assert len(cases) == 9
    for case in cases:
        argv = optimizer_argv(case, Path(tmp_path / "mesh.msh"), tmp_path / case["id"])
        assert "--save-trajectory" in argv
        assert argv[argv.index("--trajectory-every") + 1] == "1"
        assert float(argv[argv.index("--homotopy-tol-res") + 1]) == pytest.approx(
            case["homotopy_tol_res"]
        )
        assert argv[argv.index("--max-opt-it") + 1] == "0"
        assert argv[argv.index("--final-newton-tol-res") + 1] == "1e-12"
        assert argv[argv.index("--linear-solver") + 1] == "mumps"
        assert argv[argv.index("--init-search") + 1] == "full"
        fallback_indices = [
            index for index, token in enumerate(argv[:-1]) if token == "--init-fallback"
        ]
        assert argv[fallback_indices[-1] + 1] == "window-fit"
        assert select_mumps_ranks(None, case["order"], case["dof_target"]) == 4
