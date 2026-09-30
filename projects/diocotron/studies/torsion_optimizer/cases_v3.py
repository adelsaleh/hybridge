"""Decision rules and case families added for the v3 numerical-test campaign.

This module is intentionally dependency-light.  It keeps the expensive-case
policy testable without importing DOLFINx and leaves all v1/v2 manifest entries
unchanged.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any
try:
    from projects.diocotron.studies.torsion_optimizer.initialization_cases import initialization_policy_cases
except ImportError:
    from projects.diocotron.studies.torsion_optimizer.initialization_cases import initialization_policy_cases  # type: ignore


STRICT_STORYBOARD_BANDS: dict[str, tuple[float, float]] = {
    "smooth_star": (0.45, 0.50),
    "horseshoe": (0.20, 0.295),
    "iter": (0.25, 0.345),
    # A 0.095-wide perturbation of the PDE-converged [0.60, 0.70] anchor; the
    # automatically selected near-wall band did not admit a converged nonzero
    # branch under either homotopy or fit-Newton continuation.
    "pacman": (0.60, 0.695),
}

STORYBOARD_RESCUE_PARENTS: dict[str, str] = {
    "pacman": "trajectory-pacman-f64373773ee3",
    "horseshoe": "trajectory_candidate-horseshoe-9e9172e39927",
    "iter": "trajectory_candidate-iter-9aecf30783ce",
}

ANCHOR_BANDS: tuple[tuple[float, float], ...] = ((0.60, 0.70), (0.45, 0.50))
HORSESHOE_BRANCH_CONTINUATION_BANDS: tuple[tuple[float, float], ...] = (
    (0.415, 0.490),
    (0.420, 0.480),
    (0.425, 0.475),
    (0.425, 0.500),
    (0.400, 0.475),
)
HORSESHOE_BRANCH_CONTINUATION_PARENT = (
    "trajectory_horseshoe_band_candidate_v3-horseshoe-dfba417fa450"
)
INEXACT_GRADIENT_ERROR_LIMIT = 0.10
INEXACT_ANGLE_LIMIT_DEGREES = 5.0
INEXACT_DECISION_AGREEMENT_LIMIT = 0.95


def _unique_bands(*bands: tuple[float, float]) -> list[tuple[float, float]]:
    return sorted({(round(float(a1), 12), round(float(a2), 12)) for a1, a2 in bands})


def v3_cases(
    make_case: Callable[..., dict[str, Any]],
    geometries: Sequence[str],
) -> list[dict[str, Any]]:
    """Build the cost-gated v3 cases without modifying legacy case identities."""
    cases: list[dict[str, Any]] = []
    parents: dict[tuple[str, float, float], dict[str, Any]] = {}

    for geometry in geometries:
        strict = STRICT_STORYBOARD_BANDS[geometry]
        for a1, a2 in _unique_bands(strict, *ANCHOR_BANDS):
            parent = make_case(
                "mumps_parent",
                geometry=geometry,
                order=4,
                dof_target=50_000 if geometry == "smooth_star" else 75_000,
                alpha_t1=a1,
                alpha_t2=a2,
                algorithm_variant="low_dof_mumps_full_hminus1_window_fit_fallback",
                init_search="full",
                init_fallback="window-fit",
                campaigns=["v3_pilot", "v3_full"],
            )
            parents[(geometry, a1, a2)] = parent
            cases.append(parent)
            cases.append(make_case(
                "warm_started_child",
                geometry=geometry,
                order=4,
                dof_target=200_000,
                alpha_t1=a1,
                alpha_t2=a2,
                algorithm_variant="transferred_mumps_parent_fast_fallback",
                init_search="fast",
                init_fallback="window-fit",
                parent_case_id=parent["id"],
                campaigns=["v3_full"],
            ))

        cases.append(make_case(
            "geometry_overview",
            geometry=geometry,
            order=4,
            dof_target=200_000,
            alpha_t1=strict[0],
            alpha_t2=strict[1],
            algorithm_variant="strict_storyboard_design_v3",
            campaigns=["v3_pilot", "v3_full"],
        ))
        strict_parent = parents[(geometry, *strict)]
        cases.append(make_case(
            "trajectory_v3",
            geometry=geometry,
            order=4,
            dof_target=200_000,
            alpha_t1=strict[0],
            alpha_t2=strict[1],
            trajectory_every=1,
            algorithm_variant="strict_storyboard_fast_fallback",
            init_search="fast",
            init_fallback="window-fit",
            parent_case_id=strict_parent["id"],
            campaigns=["v3_pilot", "v3_full"],
            **({
                "fallback_parent_case_ids": [STORYBOARD_RESCUE_PARENTS[geometry]],
                "iterative_fallback_solver": "none",
                **({"inner_newton_tol": 1.0e-10}
                   if geometry in {"pacman", "horseshoe"} else {}),
            } if geometry in STORYBOARD_RESCUE_PARENTS else {}),
        ))

    cases.extend(initialization_policy_cases(make_case, STRICT_STORYBOARD_BANDS))

    # A deliberately separate, low-cost screen for the user-requested
    # horseshoe band. It is not part of the canonical 100-case robustness
    # matrix and carries no trajectory payload, so it cannot affect storyboard
    # selection or inflate the pilot artifact set.
    horseshoe_screen = make_case(
        "horseshoe_band_screen_v3",
        geometry="horseshoe",
        order=4,
        dof_target=75_000,
        alpha_t1=0.40,
        alpha_t2=0.50,
        algorithm_variant="low_dof_mumps_band_screen",
        linear_solver="mumps",
        init_search="full",
        init_fallback="window-fit",
        max_opt_it=10,
        campaigns=["v3_pilot"],
    )
    cases.append(horseshoe_screen)
    cases.append(make_case(
        "trajectory_horseshoe_band_candidate_v3",
        geometry="horseshoe",
        order=4,
        dof_target=75_000,
        alpha_t1=0.40,
        alpha_t2=0.50,
        trajectory_every=1,
        algorithm_variant="continued_low_dof_mumps_band_candidate",
        linear_solver="mumps",
        init_search="fast",
        init_fallback="window-fit",
        max_opt_it=30,
        parent_case_id=horseshoe_screen["id"],
        campaigns=["v3_pilot"],
    ))

    # Target-space continuation from the PDE-exact, nonempty [0.40, 0.50]
    # Horseshoe branch.  These deliberately remain a diagnostic family: the
    # checkpoint supplies an equilibrium branch but the new torsion target is
    # still specified independently by each alpha pair.  Keeping a distinct
    # kind and algorithm variant prevents a merely PDE-converged screen from
    # masquerading as the canonical storyboard.
    for alpha_t1, alpha_t2 in HORSESHOE_BRANCH_CONTINUATION_BANDS:
        cases.append(make_case(
            "trajectory_horseshoe_branch_continuation_v3",
            geometry="horseshoe",
            order=4,
            dof_target=75_000,
            alpha_t1=alpha_t1,
            alpha_t2=alpha_t2,
            trajectory_every=1,
            algorithm_variant="pde_exact_cross_target_branch_continuation_diagnostic",
            linear_solver="mumps",
            init_search="fast",
            init_fallback="window-fit",
            max_opt_it=20,
            parent_case_id=HORSESHOE_BRANCH_CONTINUATION_PARENT,
            campaigns=["v3_pilot"],
        ))

    # Low-resolution retry requested for the ITER [0.60, 0.70] band.  This is
    # deliberately a window-fit-only initialization: no H^{-1} search or
    # source homotopy is used.  Four-rank MUMPS is appropriate for the
    # approximately 75k-DOF P4 problem, and the enlarged global Newton budget
    # is combined with the residual-trend soft-cap policy.
    cases.append(make_case(
        "trajectory_iter_window_fit_high_newton_v3",
        geometry="iter",
        order=4,
        dof_target=75_000,
        alpha_t1=0.60,
        alpha_t2=0.70,
        trajectory_every=1,
        algorithm_variant="window_fit_only_mumps_high_newton_low_dof",
        initialization_method="legacy_fit_window_newton",
        linear_solver="mumps",
        max_opt_it=80,
        max_newton_it=160,
        final_newton_max_it=400,
        campaigns=["v3_pilot", "v3_full"],
    ))

    # A fixed P2-calibrated triangulation isolates p-enrichment from the
    # coarser-geometry effect in the matched-global-DOF campaign.
    for order in (2, 4, 6):
        cases.append(make_case(
            "fixed_mesh_p",
            geometry="smooth_star",
            order=order,
            mesh_reference_order=2,
            dof_target=50_000,
            alpha_t1=0.60,
            alpha_t2=0.70,
            algorithm_variant="fixed_triangulation_p_enrichment",
            init_search="fast",
            init_fallback="window-fit",
            campaigns=["v3_pilot", "v3_full"],
        ))

    # Low-cost end-to-end forcing-policy screen.  Snapshot replay diagnostics
    # are appended adaptively once a source trajectory has completed.
    for difficulty, (a1, a2) in (
        ("easy", (0.60, 0.70)),
        ("difficult", (0.45, 0.50)),
    ):
        for label, tolerance in (
            ("adaptive", None),
            ("fixed_1e-3", 1.0e-3),
            ("fixed_1e-5", 1.0e-5),
            ("fixed_1e-7", 1.0e-7),
        ):
            parameters: dict[str, Any] = {
                "geometry": "smooth_star",
                "order": 4,
                "dof_target": 50_000,
                "alpha_t1": a1,
                "alpha_t2": a2,
                "difficulty": difficulty,
                "inner_accuracy": label,
                "algorithm_variant": "low_dof_inexact_newton_screen",
                "init_search": "fast",
                "init_fallback": "window-fit",
                "campaigns": ["v3_pilot", "v3_full"],
            }
            # The difficult iterative continuation can exhaust its homotopy
            # budget before producing replay states.  Use the measured MUMPS
            # path as a controlled source of accepted non-optimal pairs.
            if difficulty == "difficult":
                parameters["algorithm_variant"] = "low_dof_mumps_inexact_newton_screen"
                parameters["linear_solver"] = "mumps"
                parameters["init_search"] = "full"
            if tolerance is not None:
                parameters["inner_newton_tol"] = tolerance
            cases.append(make_case("inner_newton_policy_v3", **parameters))
    return cases


def detect_threshold_plateau(
    records: Sequence[Mapping[str, Any]],
    *,
    window: int = 5,
    relative_limit: float = 1.0e-3,
    gradient_tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Detect a nonstationary reduced-space plateau over accepted states only."""
    accepted = []
    for record in records:
        accepted_value = str(record.get("accepted", "1")).strip().lower()
        if accepted_value not in {"1", "true", "yes"}:
            continue
        try:
            accepted.append({
                "k": int(float(record["k"])),
                "c1": float(record["c1Phi"]),
                "c2": float(record["c2Phi"]),
                "objective": float(record.get(
                    "objective", float(record["leakageRel"]) + float(record["missingRel"])
                )),
                "gradient": abs(float(record["projectedGradNorm"])),
            })
        except (KeyError, TypeError, ValueError):
            continue
    if len(accepted) < window:
        return {
            "threshold_plateau": False,
            "plateau_onset_iteration": None,
            "plateau_type": "insufficient_history",
        }
    for start in range(len(accepted) - window + 1):
        sample = accepted[start:start + window]
        width = max(abs(sample[-1]["c2"] - sample[-1]["c1"]), 1.0e-30)
        movement = max(
            math.hypot(item["c1"] - sample[0]["c1"], item["c2"] - sample[0]["c2"])
            for item in sample
        ) / width
        objective_change = abs(sample[-1]["objective"] - sample[0]["objective"]) / max(
            abs(sample[0]["objective"]), 1.0e-30
        )
        gradient = sample[-1]["gradient"]
        if movement <= relative_limit and objective_change <= relative_limit:
            stationary = gradient <= gradient_tolerance
            return {
                "threshold_plateau": True,
                "plateau_onset_iteration": sample[0]["k"],
                "plateau_type": "stationary" if stationary else "stalled",
                "plateau_threshold_movement": movement,
                "plateau_objective_change": objective_change,
                "plateau_gradient": gradient,
            }
    return {
        "threshold_plateau": False,
        "plateau_onset_iteration": None,
        "plateau_type": "none",
    }


def gradient_angle_degrees(reference: Sequence[float], candidate: Sequence[float]) -> float:
    """Return a stable angle between two reduced gradients in degrees."""
    if len(reference) != len(candidate) or not reference:
        raise ValueError("nonempty gradients with equal length are required")
    dot = sum(float(a) * float(b) for a, b in zip(reference, candidate, strict=True))
    norm_a = math.sqrt(sum(float(a) ** 2 for a in reference))
    norm_b = math.sqrt(sum(float(b) ** 2 for b in candidate))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0 if norm_a == norm_b else 90.0
    cosine = min(1.0, max(-1.0, dot / (norm_a * norm_b)))
    return math.degrees(math.acos(cosine))


def summarize_inexact_policy(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate snapshot-replay errors and apply the study admissibility gate."""
    usable = []
    for record in records:
        try:
            usable.append({
                "gradient_error": float(record["reduced_gradient_relative_error"]),
                "gradient_angle": float(record["reduced_gradient_angle_degrees"]),
                "step_error": float(
                    record["threshold_step_relative_error"]
                    if "threshold_step_relative_error" in record
                    else record["trust_step_relative_error"]
                ),
                "decision_agrees": str(record.get(
                    "acceptance_decision_agrees", record.get("decision_agrees", "false")
                )).lower() in {"1", "true", "yes"},
                "runtime": float(record.get("elapsed", record.get("runtime", 0.0))),
            })
        except (KeyError, TypeError, ValueError):
            continue
    if not usable:
        return {"admissible": False, "reason": "no_snapshot_records"}
    agreement = sum(item["decision_agrees"] for item in usable) / len(usable)
    result = {
        "snapshot_count": len(usable),
        "gradient_error_max": max(item["gradient_error"] for item in usable),
        "gradient_angle_max_degrees": max(item["gradient_angle"] for item in usable),
        "trust_step_error_max": max(item["step_error"] for item in usable),
        "decision_agreement": agreement,
        "runtime_total": sum(item["runtime"] for item in usable),
    }
    result["admissible"] = (
        result["gradient_error_max"] <= INEXACT_GRADIENT_ERROR_LIMIT
        and result["gradient_angle_max_degrees"] <= INEXACT_ANGLE_LIMIT_DEGREES
        and result["trust_step_error_max"] <= INEXACT_GRADIENT_ERROR_LIMIT
        and agreement >= INEXACT_DECISION_AGREEMENT_LIMIT
    )
    result["reason"] = "passed" if result["admissible"] else "snapshot_gate_failed"
    return result


def mumps_expansion_is_cost_effective(
    *,
    parent_seconds: float,
    child_seconds: float,
    standalone_seconds: float,
    child_success: bool,
    standalone_success: bool,
    overhead_limit: float = 0.20,
) -> bool:
    """Apply the anchor-first rule for expanding low-DOF MUMPS parents."""
    if child_success and not standalone_success:
        return True
    if min(parent_seconds, child_seconds, standalone_seconds) < 0.0 or standalone_seconds == 0.0:
        raise ValueError("timings must be nonnegative and standalone time positive")
    return parent_seconds + child_seconds <= (1.0 + overhead_limit) * standalone_seconds


def group_inexact_snapshot_rows(
    records: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Summarize records by forcing policy and Jacobian reuse strategy."""
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record.get("policy", "unknown")), str(record.get("matrix_policy", "unknown")))].append(record)
    return {key: summarize_inexact_policy(group) for key, group in groups.items()}
