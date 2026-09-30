"""Low-cost initialization and homotopy policy cases for the v3 study."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

HOMOTOPY_TOLERANCE_LEVELS: tuple[tuple[str, float], ...] = (
    ("tol_1e-7", 1.0e-7),
    ("tol_1e-9", 1.0e-9),
    ("tol_1e-11", 1.0e-11),
)


def homotopy_tolerance_cases(
    make_case: Callable[..., dict[str, Any]],
    strict_bands: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    """Return a small controlled homotopy nonlinear-tolerance experiment.

    The three tolerances share the same deterministic full H-minus-one seed
    search at each anchor. ``max_opt_it=0`` prevents reduced-space motion from
    contaminating the comparison, while the strict final projection separates
    inexact intermediate continuation from reported-equilibrium accuracy.
    MUMPS and modest targets keep the nine cases inexpensive enough to pilot.
    """
    anchors = (
        ("smooth_star", 0.60, 0.70, "easy", 50_000),
        ("smooth_star", 0.45, 0.50, "difficult", 50_000),
        ("horseshoe", *strict_bands["horseshoe"],
         "difficult_storyboard_anchor", 75_000),
    )
    cases: list[dict[str, Any]] = []
    for geometry, alpha_t1, alpha_t2, difficulty, dof_target in anchors:
        for tolerance_label, tolerance in HOMOTOPY_TOLERANCE_LEVELS:
            cases.append(make_case(
                "trajectory_homotopy_tolerance_v3",
                geometry=geometry,
                order=4,
                dof_target=dof_target,
                alpha_t1=alpha_t1,
                alpha_t2=alpha_t2,
                difficulty=difficulty,
                diagnostic="homotopy_nonlinear_tolerance_cost_accuracy",
                diagnostic_label=f"{geometry}_{difficulty}_{tolerance_label}",
                algorithm_variant="low_dof_mumps_fixed_homotopy_tolerance",
                linear_solver="mumps",
                initialization_seed_policy="automatic_full_hminus1_identical_grid",
                init_search="full",
                init_fallback="window-fit",
                homotopy_tol_res=tolerance,
                homotopy_tolerance_label=tolerance_label,
                max_opt_it=0,
                final_newton_tol_res=1.0e-12,
                trajectory_every=1,
                campaigns=["v3_pilot", "v3_full"],
            ))
    return cases



def initialization_policy_cases(
    make_case: Callable[..., dict[str, Any]],
    strict_bands: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    anchors = (
        ("smooth_star", 0.60, 0.70, "easy"),
        ("smooth_star", 0.45, 0.50, "difficult"),
        ("pacman", *strict_bands["pacman"], "storyboard"),
        ("horseshoe", *strict_bands["horseshoe"], "storyboard"),
        ("iter", *strict_bands["iter"], "storyboard"),
    )
    variants = (
        ("full_hminus1", "full", "none", None),
        ("fast_hminus1", "fast", "none", None),
        ("fast_window_fit_fallback", "fast", "window-fit", None),
        ("window_fit_only", "fast", "none", "legacy_fit_window_newton"),
    )
    cases: list[dict[str, Any]] = []
    for geometry, a1, a2, difficulty in anchors:
        for variant, init_search, init_fallback, method in variants:
            parameters: dict[str, Any] = {
                "geometry": geometry,
                "order": 4,
                "dof_target": 50_000 if geometry == "smooth_star" else 75_000,
                "alpha_t1": a1,
                "alpha_t2": a2,
                "difficulty": difficulty,
                "algorithm_variant": variant,
                "init_search": init_search,
                "init_fallback": init_fallback,
                "max_opt_it": 0,
                "final_newton_tol_res": 1.0e-11,
                "campaigns": ["v3_pilot", "v3_full"],
            }
            if method is not None:
                parameters["initialization_method"] = method
            cases.append(make_case("initialization_policy_v3", **parameters))
    cases.extend(homotopy_tolerance_cases(make_case, strict_bands))
    return cases
