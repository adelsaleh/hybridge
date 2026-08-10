from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts.diff_rea_cases import CASE_DEFINITIONS
from scripts.run_diff_rea_gpu_hard_campaign import build_commands
from scripts.analyze_diff_rea_gpu_hard_results import _robust_rankings
from scripts.validate_diff_rea_gpu_hard_cases import (
    _aggregate_rows,
    _best_rows,
    _uncovered_case_orders,
    _validation_exit_code,
    build_solver_configurations,
)


def _configuration_args() -> Namespace:
    return Namespace(
        preconditioners=[
            "none",
            "polynomial",
            "bj",
            "bj-polynomial",
            "asm",
            "asm-polynomial",
        ],
        polynomial_degrees=[8, 18],
        asm_applications=["raw", "fused"],
        bj_applications=["raw", "matmul"],
        operators=["raw", "raw_fused"],
        orthogonalizations=["cgs"],
        max_configurations=None,
    )


def test_configuration_grid_avoids_irrelevant_cross_products() -> None:
    configurations = build_solver_configurations(_configuration_args())
    # none 2; polynomial 4; BJ 4; BJ-poly 8; ASM 4; ASM-poly 8
    assert len(configurations) == 30
    for item in configurations:
        assert (item.polynomial_degree is not None) == (
            item.preconditioner in {"polynomial", "bj-polynomial", "asm-polynomial"}
        )
        assert (item.asm_application is not None) == item.preconditioner.startswith("asm")
        assert (item.block_jacobi_application is not None) == item.preconditioner.startswith("bj")


def test_all_registered_diffusion_reaction_cases_are_covered() -> None:
    assert tuple(case.key for case in CASE_DEFINITIONS) == (
        "quadratic-poisson",
        "exponential-bubble",
        "trigonometric-poisson",
        "quadratic-variable-reaction",
        "lshape-singular",
        "tensor-sine",
        "rotated-anisotropic-sine",
    )


def test_summary_ranks_only_fully_passing_configurations() -> None:
    common = {
        "case": "tensor-sine",
        "order": 6,
        "operator": "raw",
        "orthogonalization": "cgs",
        "polynomial_degree": None,
        "asm_application": None,
        "block_jacobi_application": None,
        "solver_relative_residual": 1.0e-12,
        "physical_relative_residual": 2.0e-12,
        "trace_relative_difference": 3.0e-10,
        "primal_l2_error": 1.0e-7,
        "flux_l2_error": 2.0e-6,
        "preconditioner_setup_ms": 1.0,
    }
    rows = [
        {
            **common,
            "configuration": "asm",
            "preconditioner": "asm",
            "repeat": 0,
            "passed": True,
            "iterations": 30,
            "solve_ms": 4.0,
            "hot_time_to_solution_ms": 5.0,
            "cold_time_to_solution_ms": 8.0,
            "end_to_end_ms": 20.0,
        },
        {
            **common,
            "configuration": "asm",
            "preconditioner": "asm",
            "repeat": 1,
            "passed": True,
            "iterations": 30,
            "solve_ms": 6.0,
            "hot_time_to_solution_ms": 7.0,
            "cold_time_to_solution_ms": 10.0,
            "end_to_end_ms": 22.0,
        },
        {
            **common,
            "configuration": "none",
            "preconditioner": "none",
            "repeat": 0,
            "passed": False,
            "iterations": 5000,
            "solve_ms": 1.0,
            "hot_time_to_solution_ms": 1.0,
            "cold_time_to_solution_ms": 1.0,
            "end_to_end_ms": 2.0,
        },
    ]
    summaries = _aggregate_rows(rows)
    asm = next(item for item in summaries if item["configuration"] == "asm")
    none = next(item for item in summaries if item["configuration"] == "none")
    assert asm["all_passed"]
    assert asm["median_solve_ms"] == 5.0
    assert asm["solve_rank"] == 1
    assert not none["all_passed"]
    assert none["solve_rank"] is None
    best = _best_rows(summaries)
    assert best[0]["best_solve_configuration"] == "asm"


def test_retuned_cold_sample_controls_cold_rank() -> None:
    base = {
        "case": "tensor-sine",
        "order": 6,
        "operator": "auto",
        "orthogonalization": "cgs",
        "polynomial_degree": 18,
        "asm_application": "auto",
        "block_jacobi_application": None,
        "configuration": "asm-polynomial",
        "preconditioner": "asm-polynomial",
        "passed": True,
        "iterations": 20,
        "solver_relative_residual": 1.0e-12,
        "physical_relative_residual": 2.0e-12,
        "trace_relative_difference": 3.0e-10,
        "primal_l2_error": 1.0e-7,
        "flux_l2_error": 2.0e-6,
        "preconditioner_setup_ms": 3.0,
        "operator_setup_ms": 1.0,
        "face_assembly_ms": 2.0,
        "workspace_device_bytes": 1024,
        "solve_ms": 5.0,
        "hot_time_to_solution_ms": 9.0,
        "end_to_end_ms": 30.0,
    }
    summaries = _aggregate_rows(
        [
            {
                **base,
                "repeat": 0,
                "timing_sample": "retuned-cold",
                "autotune_ms": 100.0,
                "cold_time_to_solution_ms": 109.0,
            },
            {
                **base,
                "repeat": 1,
                "timing_sample": "cached-or-explicit",
                "autotune_ms": 0.1,
                "cold_time_to_solution_ms": 9.1,
            },
        ]
    )
    assert summaries[0]["representative_cold_time_to_solution_ms"] == 109.0
    assert summaries[0]["retuned_autotune_ms"] == 100.0


def test_coverage_allows_failed_candidates_but_requires_one_robust_choice() -> None:
    summaries = [
        {"case": "a", "order": 4, "configuration": "none", "all_passed": False},
        {"case": "a", "order": 4, "configuration": "asm", "all_passed": True},
        {"case": "b", "order": 4, "configuration": "none", "all_passed": False},
    ]
    assert _uncovered_case_orders(summaries) == [("b", 4)]
    summaries.append(
        {"case": "b", "order": 4, "configuration": "asm", "all_passed": True}
    )
    assert _uncovered_case_orders(summaries) == []


def test_coverage_exit_policy_does_not_hide_uncovered_or_exceptional_runs() -> None:
    candidate_failure = {
        "any_candidate_failed": True,
        "uncovered_count": 0,
        "exception_count": 0,
    }
    assert _validation_exit_code(
        strict=False, require_coverage=True, **candidate_failure
    ) == 0
    assert _validation_exit_code(
        strict=True, require_coverage=False, **candidate_failure
    ) == 1
    assert _validation_exit_code(
        strict=False,
        require_coverage=True,
        any_candidate_failed=True,
        uncovered_count=1,
        exception_count=0,
    ) == 1
    assert _validation_exit_code(
        strict=False,
        require_coverage=True,
        any_candidate_failed=False,
        uncovered_count=0,
        exception_count=1,
    ) == 1


def test_robust_ranking_requires_every_case_order() -> None:
    rows = [
        {
            "case": case,
            "order": order,
            "configuration": configuration,
            "preconditioner": configuration,
            "all_passed": passed,
            "median_solve_ms": solve,
            "worst_solver_relative_residual": 1.0e-12,
            "worst_physical_relative_residual": 2.0e-12,
        }
        for case, order, configuration, passed, solve in (
            ("a", "4", "robust", True, 2.0),
            ("b", "4", "robust", True, 3.0),
            ("a", "4", "fragile", True, 1.0),
            ("b", "4", "fragile", False, 1.0),
        )
    ]
    ranked = _robust_rankings(rows)
    assert ranked[0]["configuration"] == "robust"
    assert ranked[0]["all_case_orders_passed"]
    assert not ranked[1]["all_case_orders_passed"]


def test_campaign_levels_are_deterministic(tmp_path: Path) -> None:
    smoke_with_tests = build_commands(
        "smoke",
        output_dir=tmp_path,
        python="python",
        skip_tests=False,
        with_nsys=False,
        with_ncu=False,
    )
    smoke = build_commands(
        "smoke",
        output_dir=tmp_path,
        python="python",
        skip_tests=True,
        with_nsys=False,
        with_ncu=False,
    )
    validation = build_commands(
        "validation",
        output_dir=tmp_path,
        python="python",
        skip_tests=True,
        with_nsys=False,
        with_ncu=False,
    )
    full = build_commands(
        "full",
        output_dir=tmp_path,
        python="python",
        skip_tests=True,
        with_nsys=True,
        with_ncu=True,
    )
    regression = smoke_with_tests[0]
    assert regression.name == "gpu_regression_tests"
    assert "tests/test_diff_rea_gpu_hard_campaign.py" in regression.argv
    assert [item.name for item in smoke] == ["smoke_hard_cases"]
    assert [item.name for item in validation] == [
        "smoke_hard_cases",
        "all_cases_p4_p6_validation",
        "final_analysis",
    ]
    validation_command = next(
        item for item in validation if item.name == "all_cases_p4_p6_validation"
    )
    assert "--require-coverage" in validation_command.argv
    assert "--strict" not in validation_command.argv
    cases_start = validation_command.argv.index("--cases") + 1
    assert validation_command.argv[cases_start] == "all"
    component = next(item for item in full if item.name == "component_profile_p4_p6")
    assert component.argv[component.argv.index("--cases") + 1 : component.argv.index("--orders")] == (
        "trigonometric-poisson",
        "tensor-sine",
    )
    assert full[-3].name == "nsight_systems_tensor_p6"
    assert full[-2].name == "nsight_compute_tensor_p6"
    assert full[-1].name == "final_analysis"
