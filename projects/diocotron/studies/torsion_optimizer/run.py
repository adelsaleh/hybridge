#!/usr/bin/env python3
"""Plan, run, aggregate, plot, and report torsion-optimizer numerical tests.

The generated LaTeX is an includable numerical-tests section, not a complete
article.  Expensive MPI cases are executed serially by this driver so timing
measurements are not contaminated by concurrent study jobs.
"""

from __future__ import annotations

# Support direct execution as well as python -m from a checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[4]))

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from projects.diocotron.paths import rebase_archive_paths, resolve_archive_path

try:
    from projects.diocotron.dolfinx.geometry.canonical import (
        GEOMETRIES,
        canonical_geometry_name,
        generate_mesh,
        geometry_definition,
        lagrange_dofs_from_metadata,
        sha256_file,
    )
except ImportError:  # direct script execution
    from projects.diocotron.dolfinx.geometry.canonical import (  # type: ignore
        GEOMETRIES,
        canonical_geometry_name,
        generate_mesh,
        geometry_definition,
        lagrange_dofs_from_metadata,
        sha256_file,
    )

try:
    from projects.diocotron.studies.torsion_optimizer.schedule import (
        additional_refinement_parameters, fine_confirmation_parameters,
    )
except ImportError:
    from projects.diocotron.studies.torsion_optimizer.schedule import (  # type: ignore
        additional_refinement_parameters, fine_confirmation_parameters,
    )

try:
    from projects.diocotron.dolfinx.runtime.mpi_rank_policy import (
        select_mpi_ranks as _select_mpi_ranks,
        smaller_rank_within_ten_percent,
    )
except ImportError:
    from projects.diocotron.dolfinx.runtime.mpi_rank_policy import (  # type: ignore
        select_mpi_ranks as _select_mpi_ranks,
        smaller_rank_within_ten_percent,
    )

try:
    from projects.diocotron.studies.torsion_optimizer.extended import (
        FIGURE_CONTRACT, augment_aggregate_row, create_contract_figures,
        extend_optimizer_argv, extended_cases, inferred_campaigns,
        validate_figure_contract,
    )
except ImportError:
    from projects.diocotron.studies.torsion_optimizer.extended import (  # type: ignore
        FIGURE_CONTRACT, augment_aggregate_row, create_contract_figures,
        extend_optimizer_argv, extended_cases, inferred_campaigns,
        validate_figure_contract,
    )

try:
    from projects.diocotron.studies.torsion_optimizer.cases_v3 import (
        detect_threshold_plateau as detect_threshold_plateau_v3,
        group_inexact_snapshot_rows,
        v3_cases,
    )
except ImportError:
    from projects.diocotron.studies.torsion_optimizer.cases_v3 import (  # type: ignore
        detect_threshold_plateau as detect_threshold_plateau_v3,
        group_inexact_snapshot_rows,
        v3_cases,
    )

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_BUNDLE = REPO_ROOT / "projects/diocotron/studies/torsion_optimizer/report"
DEFAULT_RUN_ROOT = REPO_ROOT / "projects/diocotron/runs" / "torsion_optimizer"
MANIFEST_VERSION = 3
EPSILON_RATIO = 0.08
KAPPA = 2.0
SUCCESS_STATUSES = {"CONVERGED", "CONVERGED_CERTIFIED_SUBBAND"}
SUCCESS_CLASSIFICATIONS = {
    "strict_convergence",
    "certified_subband_convergence",
}
TERMINAL_CASE_STATES = {"completed", "failed", "deferred"}
GEOMETRY_ORDER = ("smooth_star", "pacman", "horseshoe", "iter")

INEXACT_REPLAY_POLICIES = frozenset({"raw", "one-correction", "reassembled"})
INEXACT_REPLAY_MIN_SNAPSHOTS = 4
INEXACT_REPLAY_MIN_REQUESTS_PER_POLICY = 5
INEXACT_REPLAY_CSV_FIELDS = (
    "snapshot_id", "policy", "requested_tolerance", "outer_iteration",
    "c1", "c2", "state_residual", "reference_state_residual",
    "state_l2_relative_error", "state_h1_relative_error",
    "sensitivity1_l2_relative_error", "sensitivity1_h1_relative_error",
    "sensitivity2_l2_relative_error", "sensitivity2_h1_relative_error",
    "sensitivity1_defect", "sensitivity2_defect",
    "reduced_gradient_relative_error", "reduced_gradient_angle_degrees",
    "threshold_step_relative_error", "threshold_step_angle_degrees",
    "predicted_reduction_relative_error", "acceptance_decision_agrees",
    "state_solve_time", "sensitivity_solve_time", "elapsed",
)

OPTIMIZER = REPO_ROOT / "projects/diocotron/dolfinx/torsion/optimization/homotopy.py"
DYNAMICS = REPO_ROOT / "projects/diocotron/dolfinx/guiding_center/supg.py"
EQUILIBRIUM_COMPARATOR = REPO_ROOT / "projects/diocotron/comparisons/equilibrium.py"

GEOMETRY_OVERVIEW = REPO_ROOT / "projects/diocotron/studies/torsion_optimizer/figures/geometry_standalone.py"
BASELINE_FLAGS = (
    "--solver-preset", "optimized",
    "--init-mode", "homotopy",
    "--no-include-fit-init",
    # Every production homotopy run gets one automatic fit-Newton rescue if
    # source continuation cannot reach lambda=1. Explicit policy-study cases
    # can still append ``--init-fallback none`` to retain a no-rescue control.
    "--init-fallback", "window-fit",
    "--init-hminus1-grid", "20",
    "--init-hminus1-refine-grid", "11",
    "--init-hminus1-refine-passes", "2",
    "--homotopy-initial-step", "0.1",
    "--homotopy-min-step", "0.00025",
    "--homotopy-max-step", "0.35",
    "--tol-res", "1e-11",
    "--inner-tol-max", "1e-7",
    "--inner-tol-gamma", "1e-8",
    "--final-newton-tol-res", "1e-12",
    "--trust-radius", "0.005",
    "--trial-linear-max-it", "300",
    "--eps-mode", "relative",
    "--eps-ratio", "0.08",
    "--no-plot-design",
    "--no-plot-optimization",
    "--no-plot-final",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def deterministic_case_id(case: dict[str, Any]) -> str:
    identity = {
        key: value for key, value in case.items()
        if key not in {
            "id", "state", "attempts", "result", "campaigns",
            "requires_failure_of", "requires_non_success_of", "deferred_reason",
        }
    }
    digest = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()[:12]
    geometry = str(case.get("geometry", "global"))
    kind = str(case["kind"])
    return f"{kind}-{geometry}-{digest}"


def make_case(kind: str, **parameters: Any) -> dict[str, Any]:
    case = {"kind": kind, **parameters, "state": "planned", "attempts": []}
    case["id"] = deterministic_case_id(case)
    return case


def robustness_matrix() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    centers = [0.15 + 0.10 * index for index in range(8)]
    for geometry in GEOMETRY_ORDER:
        for center in centers:
            for width in (0.05, 0.10, 0.20):
                cases.append(make_case(
                    "robustness",
                    geometry=geometry,
                    order=4,
                    dof_target=200_000,
                    alpha_t1=round(center - 0.5 * width, 12),
                    alpha_t2=round(center + 0.5 * width, 12),
                ))
        cases.append(make_case(
            "robustness",
            geometry=geometry,
            order=4,
            dof_target=200_000,
            alpha_t1=0.45,
            alpha_t2=0.50,
            anchor=True,
        ))
    return cases


def mesh_order_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    targets = {
        "smooth_star": (50_000, 100_000, 200_000, 400_000, 700_000),
        "pacman": (75_000, 200_000, 500_000),
        "horseshoe": (75_000, 200_000, 500_000),
        "iter": (75_000, 200_000, 500_000),
    }
    for geometry, geometry_targets in targets.items():
        for order in (2, 4, 6):
            for target in geometry_targets:
                cases.append(make_case(
                    "mesh_order",
                    geometry=geometry,
                    order=order,
                    dof_target=target,
                    alpha_t1=0.60,
                    alpha_t2=0.70,
                ))
                if geometry == "smooth_star":
                    cases.append(make_case(
                        "mesh_order_difficult",
                        geometry=geometry,
                        order=order,
                        dof_target=target,
                        alpha_t1=0.45,
                        alpha_t2=0.50,
                    ))
    return cases


def scaling_cases() -> list[dict[str, Any]]:
    cases = []
    for alpha_t1, alpha_t2, difficulty in ((0.60, 0.70, "easy"), (0.45, 0.50, "difficult")):
        for ranks in (4, 8, 12, 16, 20):
            for repeat in range(1, 4):
                cases.append(make_case(
                    "strong_scaling",
                    geometry="smooth_star",
                    order=4,
                    dof_target=700_000,
                    alpha_t1=alpha_t1,
                    alpha_t2=alpha_t2,
                    difficulty=difficulty,
                    ranks=ranks,
                    repeat=repeat,
                ))
    for difficulty, alpha_t1, alpha_t2 in (("easy", 0.60, 0.70), ("difficult", 0.45, 0.50)):
        cases.append(make_case(
            "all_mumps_baseline",
            geometry="smooth_star",
            order=4,
            dof_target=700_000,
            alpha_t1=alpha_t1,
            alpha_t2=alpha_t2,
            difficulty=difficulty,
            rank_selector="best_measured",
        ))
    return cases


def trajectory_cases() -> list[dict[str, Any]]:
    return [
        make_case("trajectory", geometry="smooth_star", order=4, dof_target=200_000,
                  alpha_t1=0.60, alpha_t2=0.70, trajectory_every=1),
        make_case("trajectory", geometry="smooth_star", order=4, dof_target=200_000,
                  alpha_t1=0.45, alpha_t2=0.50, trajectory_every=1),
        make_case("trajectory", geometry="iter", order=4, dof_target=200_000,
                  alpha_t1=0.60, alpha_t2=0.70, trajectory_every=1),
    ]


def diagnostic_cases() -> list[dict[str, Any]]:
    return [
        make_case("coercivity", geometry="smooth_star", order=4, dof_target=200_000,
                  alpha_t1=0.45, alpha_t2=0.50),
        make_case("coercivity", geometry="iter", order=4, dof_target=200_000,
                  alpha_t1=0.05, alpha_t2=0.25, nonstar_proxy=True,
                  source_case_id="trajectory_candidate-iter-9aecf30783ce"),
    ]


def dynamics_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for geometry in GEOMETRY_ORDER:
        for dt in (0.025, 0.0125):
            cases.append(make_case(
                "stationarity",
                geometry=geometry,
                order=4,
                equilibrium_selector="finest_reference",
                alpha_t1=0.60,
                alpha_t2=0.70,
                dt=dt,
                final_time=0.1,
            ))
    for dt in (0.025, 0.0125):
        cases.append(make_case(
            "stationarity",
            geometry="smooth_star",
            order=4,
            equilibrium_selector="finest_reference_difficult",
            alpha_t1=0.45,
            alpha_t2=0.50,
            dt=dt,
            final_time=0.1,
        ))
    cases.append(make_case(
        "stationarity_flagship",
        geometry="smooth_star",
        order=4,
        equilibrium_selector="finest_reference",
        alpha_t1=0.60,
        alpha_t2=0.70,
        dt=0.025,
        final_time=0.5,
    ))
    return cases


def build_study_cases() -> list[dict[str, Any]]:
    cases = [
        *robustness_matrix(),
        *mesh_order_cases(),
        *scaling_cases(),
        *trajectory_cases(),
        *diagnostic_cases(),
        *extended_cases(make_case, GEOMETRY_ORDER),
        *v3_cases(make_case, GEOMETRY_ORDER),
        *dynamics_cases(),
    ]
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        duplicates = sorted(case_id for case_id in set(ids) if ids.count(case_id) > 1)
        raise ValueError(f"duplicate deterministic case IDs: {duplicates}")
    for case in cases:
        case.setdefault("campaigns", inferred_campaigns(case))
    return cases


def select_mumps_ranks(mesh_size: float | None, order: int, dofs: int | None = None) -> int:
    """Apply the workstation rank map from ``AGENTS.md``."""
    return _select_mpi_ranks(mesh_size, order, dofs)


def calibrate_mesh_size(
    target_dofs: int,
    sample: Callable[[float], int],
    initial_size: float,
    *,
    tolerance: float = 0.15,
    max_iterations: int = 8,
) -> tuple[float, int, list[dict[str, float | int]]]:
    """Calibrate size by the 2-D relation DOFs proportional to ``h^-2``."""
    if target_dofs <= 0 or initial_size <= 0.0:
        raise ValueError("target_dofs and initial_size must be positive")
    h = float(initial_size)
    history: list[dict[str, float | int]] = []
    for _ in range(max_iterations):
        dofs = int(sample(h))
        if dofs <= 0:
            raise ValueError("mesh calibration sample returned a non-positive DOF count")
        relative_error = abs(dofs - target_dofs) / target_dofs
        history.append({"mesh_size": h, "dofs": dofs, "relative_error": relative_error})
        if relative_error <= tolerance:
            return h, dofs, history
        factor = math.sqrt(dofs / target_dofs)
        h *= min(2.0, max(0.5, factor))
    best = min(history, key=lambda item: float(item["relative_error"]))
    return float(best["mesh_size"]), int(best["dofs"]), history


def epsilon_invariant(c1: float, c2: float, eps_phi: float, *, ratio: float = EPSILON_RATIO) -> bool:
    return c2 > c1 and math.isclose(eps_phi / (c2 - c1), ratio, rel_tol=1.0e-10, abs_tol=1.0e-13)


def classify_status(final_status: str | None, exit_code: int | None, residual: float | None,
                    requested_residual: float = 1.0e-12, capped_trials: int = 0,
                    pde_residual: float = 1.0e-11) -> str:
    if exit_code is None:
        return "incomplete"
    if residual is None or not math.isfinite(residual):
        return "pde_failure"
    if residual > pde_residual:
        return "pde_failure"
    # A nonzero exit is intentional when only the strict final projection
    # misses requested_residual.  Preserve the distinct baseline-PDE outcome.
    if final_status == "NEWTON_NOT_CONVERGED":
        return "geometrically_unsuccessful_pde_converged"
    if exit_code != 0:
        return "pde_failure"
    if residual > requested_residual:
        return "geometrically_unsuccessful_pde_converged"
    if final_status == "CONVERGED":
        return "strict_convergence"
    if final_status == "CONVERGED_CERTIFIED_SUBBAND":
        return "certified_subband_convergence"
    if capped_trials:
        return "pde_converged_with_capped_trials"
    return "geometrically_unsuccessful_pde_converged"


def observed_rates(h_values: Sequence[float], errors: Sequence[float]) -> list[float]:
    if len(h_values) != len(errors):
        raise ValueError("h_values and errors must have equal length")
    rates = [math.nan]
    for old_h, new_h, old_e, new_e in zip(h_values, h_values[1:], errors, errors[1:]):
        if min(old_h, new_h, old_e, new_e) <= 0.0 or math.isclose(old_h, new_h):
            rates.append(math.nan)
        else:
            rates.append(math.log(new_e / old_e) / math.log(new_h / old_h))
    return rates


def select_reference(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any] | None, bool]:
    """Choose the finest converged row and flag unresolved final-grid changes."""
    converged = [row for row in rows if row.get("classification") in SUCCESS_CLASSIFICATIONS]
    if not converged:
        return None, True
    ordered = sorted(converged, key=lambda row: int(row.get("ndof", row.get("dof_target", 0))))
    reference = ordered[-1]
    if len(ordered) < 2:
        return reference, True
    changes = [float(row["two_grid_change"]) for row in ordered[-2:] if row.get("two_grid_change") not in (None, "")]
    unresolved = len(changes) < 2 or changes[-1] >= changes[-2] or changes[-1] > 0.02
    return reference, unresolved


def hash_outputs(root: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    if not root.exists():
        return output
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        output[str(path.relative_to(root))] = sha256_file(path)
    return output


def inexact_replay_csv_valid(path: Path) -> bool:
    """Validate that an inexact-Newton replay contains complete real samples."""
    if not path.is_file():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not set(INEXACT_REPLAY_CSV_FIELDS).issubset(reader.fieldnames or ()):
                return False
            rows = list(reader)
    except (OSError, csv.Error, UnicodeError):
        return False
    if not rows:
        return False

    numeric_fields = set(INEXACT_REPLAY_CSV_FIELDS) - {
        "snapshot_id", "policy", "acceptance_decision_agrees",
    }
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        snapshot = str(row.get("snapshot_id", "")).strip()
        policy = str(row.get("policy", "")).strip()
        if not snapshot or policy not in INEXACT_REPLAY_POLICIES:
            return False
        if str(row.get("acceptance_decision_agrees", "")).strip().lower() not in {
            "0", "1", "false", "true", "no", "yes",
        }:
            return False
        try:
            values = {field: float(row[field]) for field in numeric_fields}
        except (KeyError, TypeError, ValueError):
            return False
        if not all(math.isfinite(value) for value in values.values()):
            return False
        if values["requested_tolerance"] <= 0.0 or values["c2"] <= values["c1"]:
            return False
        if any(
            values[field] < 0.0
            for field in numeric_fields - {"outer_iteration", "c1", "c2"}
        ):
            return False
        if not all(
            0.0 <= values[field] <= 180.0
            for field in (
                "reduced_gradient_angle_degrees",
                "threshold_step_angle_degrees",
            )
        ):
            return False
        grouped[snapshot][policy].append(values["requested_tolerance"])

    if len(grouped) < INEXACT_REPLAY_MIN_SNAPSHOTS:
        return False
    for policies in grouped.values():
        if set(policies) != INEXACT_REPLAY_POLICIES:
            return False
        tolerance_sets = [sorted(policies[policy]) for policy in INEXACT_REPLAY_POLICIES]
        if any(
            len(tolerances) < INEXACT_REPLAY_MIN_REQUESTS_PER_POLICY
            for tolerances in tolerance_sets
        ):
            return False
        reference = tolerance_sets[0]
        if any(
            len(tolerances) != len(reference)
            or not np.allclose(tolerances, reference, rtol=1.0e-13, atol=0.0)
            for tolerances in tolerance_sets[1:]
        ):
            return False
    return True


def completed_case_outputs_valid(case: dict[str, Any]) -> bool:
    """Return whether a completed manifest entry is intact and hash-valid."""
    result = case.get("result", {})
    if case.get("state") != "completed" or result.get("exit_code") != 0:
        return False
    output_hashes = result.get("output_hashes", {})
    run_dir = result.get("run_dir")
    if not output_hashes or not run_dir:
        return False
    attempt_dir = Path(run_dir).parent
    if any(
        not (attempt_dir / relative).is_file()
        or sha256_file(attempt_dir / relative) != expected
        for relative, expected in output_hashes.items()
    ):
        return False
    if case["kind"].startswith("stationarity"):
        required = ("summary",)
    elif case["kind"] == "geometry_overview":
        required = ("summary", "trajectory")
    elif case["kind"].startswith("trajectory"):
        required = ("summary", "equilibrium")
        trajectory = (
            Path(str(result["trajectory"]))
            if result.get("trajectory")
            else Path(str(run_dir)) / "out" / "trajectory.npz"
        )
        if not trajectory.is_file():
            return False
    else:
        required = ("summary", "equilibrium")
    if not all(result.get(key) and Path(result[key]).is_file() for key in required):
        return False
    if (
        case.get("kind") == "inner_newton_policy_v3"
        and case.get("inner_accuracy") == "adaptive"
    ):
        inexact_path = Path(str(run_dir)) / "logs" / "inexact_newton.csv"
        if not inexact_replay_csv_valid(inexact_path):
            return False
    return True


def parse_key_value_summary(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if not path.is_file():
        return parsed
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition(" ")
        if separator:
            parsed[key] = value.strip()
    return parsed


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _git_provenance() -> dict[str, Any]:
    def run(*argv: str) -> str:
        completed = subprocess.run(argv, cwd=REPO_ROOT, check=False, text=True, capture_output=True)
        return completed.stdout.strip()
    return {
        "revision": run("git", "rev-parse", "HEAD"),
        "status": run("git", "status", "--short"),
        "diff": run("git", "diff", "--no-ext-diff"),
    }


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in ("numpy", "scipy", "matplotlib", "meshio", "gmsh", "mpi4py", "petsc4py", "fenics-dolfinx"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def new_manifest() -> dict[str, Any]:
    return {
        "format": "hdgfem_torsion_optimizer_numerical_tests",
        "version": MANIFEST_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "epsilon_ratio": EPSILON_RATIO,
        "kappa": KAPPA,
        "certified_potential_width_fraction": 1.0 - 2.0 * KAPPA * EPSILON_RATIO,
        "cases": build_study_cases(),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = rebase_archive_paths(json.loads(path.read_text(encoding="utf-8")))
    version = int(manifest.get("version", 0))
    if version not in {1, 2, MANIFEST_VERSION}:
        raise ValueError(f"unsupported manifest version {manifest.get('version')!r}")
    desired = build_study_cases()
    desired_ids = {case["id"] for case in desired}
    existing_ids = {case["id"] for case in manifest["cases"]}
    appended = 0
    for case in desired:
        if case["id"] not in existing_ids:
            manifest["cases"].append(case)
            existing_ids.add(case["id"])
            appended += 1
    for case in manifest["cases"]:
        case.setdefault("campaigns", inferred_campaigns(case))
        # Protocol revisions retain terminal evidence but must not leave stale
        # dependency graphs silently planned forever. Only v3 families whose
        # deterministic identity is no longer generated are superseded here.
        if (
            case["id"] not in desired_ids
            and case.get("kind") in {
                "initialization_policy_v3", "mumps_parent",
                "warm_started_child", "trajectory_v3",
                "inner_newton_policy_v3",
            }
            and case.get("state") == "planned"
        ):
            case["state"] = "deferred"
        if (
            case.get("state") == "completed"
            and case.get("kind") == "inner_newton_policy_v3"
            and case.get("inner_accuracy") == "adaptive"
        ):
            result = case.get("result", {})
            run_dir = result.get("run_dir")
            replay_path = (
                Path(str(result.get("inexact_newton")))
                if result.get("inexact_newton")
                else Path(str(run_dir)) / "logs" / "inexact_newton.csv"
            )
            if not run_dir or not inexact_replay_csv_valid(replay_path):
                case["state"] = "failed"
                result["validation_error"] = "incomplete inexact-Newton replay coverage"
            case["deferred_reason"] = "superseded_by_revised_v3_protocol"
    if version < MANIFEST_VERSION:
        manifest["version"] = MANIFEST_VERSION
        manifest.setdefault("migrations", []).append({
            "from": version, "to": MANIFEST_VERSION,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "appended_cases": appended,
        })
        write_json(path, manifest)
    ids = [case["id"] for case in manifest["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest contains duplicate case IDs")
    return manifest


def _calibration_key(geometry: str, order: int, target: int) -> str:
    return f"{canonical_geometry_name(geometry)}-p{int(order)}-dof{int(target)}"


def ensure_calibrated_mesh(case: dict[str, Any], run_root: Path, cache: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    geometry = canonical_geometry_name(case["geometry"])
    order = int(case["order"])
    calibration_order = int(case.get("mesh_reference_order", order))
    target = int(case["dof_target"])
    key = _calibration_key(geometry, calibration_order, target)
    mesh_dir = run_root / "meshes"
    mesh_path = mesh_dir / f"{key}.msh"
    metadata_path = mesh_path.with_suffix(".msh.json")
    cached = cache.get(key)
    if cached and mesh_path.is_file() and metadata_path.is_file() and sha256_file(mesh_path) == cached.get("mesh_sha256"):
        runtime_metadata = dict(cached)
        runtime_metadata["dofs"] = lagrange_dofs_from_metadata(runtime_metadata, order)
        runtime_metadata["runtime_order"] = order
        return mesh_path, runtime_metadata

    generated: dict[float, dict[str, Any]] = {}
    def sample(h: float) -> int:
        candidate_path = mesh_dir / f".{key}-{len(generated):02d}.msh"
        metadata = generate_mesh(geometry, h, candidate_path)
        generated[h] = {**metadata, "candidate_path": str(candidate_path)}
        return lagrange_dofs_from_metadata(metadata, calibration_order)

    h, dofs, history = calibrate_mesh_size(target, sample, geometry_definition(geometry).default_size)
    chosen = generated[h]
    chosen_path = Path(chosen.pop("candidate_path"))
    mesh_dir.mkdir(parents=True, exist_ok=True)
    chosen_path.replace(mesh_path)
    chosen_sidecar = chosen_path.with_suffix(chosen_path.suffix + ".json")
    if chosen_sidecar.exists():
        chosen_sidecar.replace(metadata_path)
    chosen.update({
        "order": calibration_order,
        "calibration_order": calibration_order,
        "dofs": dofs,
        "dofs_by_order": {
            str(candidate_order): lagrange_dofs_from_metadata(chosen, candidate_order)
            for candidate_order in range(1, 7)
        },
        "target_dofs": target,
        "calibration_history": history,
    })
    chosen["mesh_sha256"] = sha256_file(mesh_path)
    write_json(metadata_path, chosen)
    cache[key] = chosen
    runtime_metadata = dict(chosen)
    runtime_metadata["dofs"] = lagrange_dofs_from_metadata(runtime_metadata, order)
    runtime_metadata["runtime_order"] = order
    return mesh_path, runtime_metadata


def optimizer_argv(
    case: dict[str, Any],
    mesh_path: Path,
    run_dir: Path,
    *,
    initial_equilibrium: Path | None = None,
) -> list[str]:
    argv = [
        sys.executable, str(OPTIMIZER),
        "--run-dir", str(run_dir),
        "--run-tag", case["id"],
        "--mesh", str(mesh_path),
        "--order", str(case["order"]),
        "--alphaT1", str(case["alpha_t1"]),
        "--alphaT2", str(case["alpha_t2"]),
        *BASELINE_FLAGS,
    ]
    if (
        case["kind"] in {"all_mumps_baseline", "mumps_parent"}
        or case.get("linear_solver") == "mumps"
    ):
        argv.extend(("--linear-solver", "mumps"))
    if case["kind"] == "coercivity":
        argv.append("--diagnose-capped-trial-coercivity")
    argv = extend_optimizer_argv(argv, case)
    # Every campaign attempt must retain enough globally gathered state to
    # render its PNG storyboard, including partial and failed continuations.
    # Explicit case-level trajectory options remain authoritative, while the
    # default records every selected state.
    if "--save-trajectory" not in argv:
        argv.append("--save-trajectory")
    if "--trajectory-every" not in argv:
        argv.extend(("--trajectory-every", str(case.get("trajectory_every", 1))))
    if case.get("max_newton_it") is not None:
        argv.extend(("--max-newton-it", str(case["max_newton_it"])))
    if case.get("final_newton_max_it") is not None:
        argv.extend(("--final-newton-max-it", str(case["final_newton_max_it"])))
    if (case.get("initialization_method") == "legacy_fit_window_newton"
            and case["kind"] not in {"trajectory_fit_rescue", "reference_fit_rescue"}):
        argv.extend(("--init-mode", "legacy", "--include-fit-init",
                     "--legacy-project-preselected-only"))
    if case.get("init_search"):
        argv.extend(("--init-search", str(case["init_search"])))
    if case.get("init_fallback"):
        argv.extend(("--init-fallback", str(case["init_fallback"])))
    if case.get("homotopy_tol_res") is not None:
        argv.extend(("--homotopy-tol-res", str(case["homotopy_tol_res"])))
    if case.get("iterative_fallback_solver") is not None:
        argv.extend(("--iterative-fallback-solver", str(case["iterative_fallback_solver"])))
    if case.get("inner_newton_tol") is not None and case["kind"] != "inner_newton_accuracy":
        argv.extend(("--inner-newton-tol", str(case["inner_newton_tol"])))
    if case.get("trust_radius") is not None:
        argv.extend(("--trust-radius", str(case["trust_radius"])))
    # The adaptive low-DOF cases are the source trajectories for the dedicated
    # identical-threshold inexact-Newton replay. Keeping this derived from the
    # existing v3 case identity avoids invalidating or duplicating manifests.
    if (
        case.get("kind") == "inner_newton_policy_v3"
        and case.get("inner_accuracy") == "adaptive"
    ):
        argv.extend((
            "--run-inexact-newton-study",
            "--inexact-newton-tolerances", "1e-2,1e-3,1e-5,1e-7,adaptive",
            "--inexact-newton-reference-tol", "1e-12",
            "--inexact-newton-max-snapshots", "4",
        ))
    if initial_equilibrium is not None:
        argv.extend(("--initial-equilibrium", str(initial_equilibrium)))
    return argv


def _parent_equilibrium(manifest: dict[str, Any], case: dict[str, Any]) -> Path | None:
    """Resolve a hash-valid, tightly converged, nontrivial parent checkpoint."""
    primary = case.get("parent_case_id")
    fallback_ids = [
        str(value) for value in case.get("fallback_parent_case_ids", [])
    ]
    parent_ids: list[str] = []
    if primary:
        parent_ids.append(str(primary))
    parent_ids.extend(fallback_ids)
    if not parent_ids:
        return None

    by_id = {candidate["id"]: candidate for candidate in manifest["cases"]}
    reasons: list[str] = []
    for parent_id in parent_ids:
        parent = by_id.get(parent_id)
        if parent is None:
            reasons.append(f"{parent_id}: unknown")
            continue
        result = parent.get("result", {})
        run_dir_value = result.get("run_dir")
        equilibrium_value = result.get("equilibrium")
        summary_value = result.get("summary")
        output_hashes = result.get("output_hashes", {})
        if not run_dir_value or not equilibrium_value or not summary_value or not output_hashes:
            reasons.append(f"{parent_id}: no recorded outputs")
            continue
        run_dir = Path(str(run_dir_value))
        equilibrium = Path(str(equilibrium_value))
        summary_path = Path(str(summary_value))
        attempt_dir = run_dir.parent
        if any(
            not (attempt_dir / relative).is_file()
            or sha256_file(attempt_dir / relative) != expected
            for relative, expected in output_hashes.items()
        ):
            reasons.append(f"{parent_id}: output hash mismatch")
            continue
        if not equilibrium.is_file() or not summary_path.is_file():
            reasons.append(f"{parent_id}: checkpoint or summary missing")
            continue
        summary = parse_key_value_summary(summary_path)
        residual = _number(summary.get("bestResidual"))
        activity = _number(summary.get("bestActivityArea"))
        if residual is None or residual > 1.0e-11:
            reasons.append(f"{parent_id}: residual {residual!r} exceeds 1e-11")
            continue
        if activity is None or activity <= 0.0:
            reasons.append(f"{parent_id}: zero or unreported activity")
            continue
        return equilibrium

    if all(by_id.get(parent_id, {}).get("state") == "planned" for parent_id in parent_ids):
        raise LookupError(f"parent cases for {case['id']} have not run")
    raise RuntimeError(
        f"no usable parent checkpoint for {case['id']}: " + "; ".join(reasons)
    )


def overview_argv(case: dict[str, Any], mesh_path: Path, run_dir: Path) -> list[str]:
    return [
        sys.executable, str(GEOMETRY_OVERVIEW),
        "--run-dir", str(run_dir),
        "--run-tag", case["id"],
        "--mesh", str(mesh_path),
        "--order", str(case["order"]),
        "--alphaT1", str(case["alpha_t1"]),
        "--alphaT2", str(case["alpha_t2"]),
    ]


def _latest_successful_equilibrium(manifest: dict[str, Any], case: dict[str, Any]) -> Path:
    candidates = []
    for candidate in manifest["cases"]:
        if candidate.get("state") != "completed" or candidate.get("geometry") != case.get("geometry"):
            continue
        if candidate.get("order") != case.get("order"):
            continue
        if candidate.get("alpha_t1") != case.get("alpha_t1") or candidate.get("alpha_t2") != case.get("alpha_t2"):
            continue
        path = candidate.get("result", {}).get("equilibrium")
        if path and Path(path).is_file():
            candidates.append((int(candidate.get("dof_target", 0)), Path(path)))
    if not candidates:
        raise RuntimeError(f"no completed equilibrium satisfies selector for {case['id']}")
    return max(candidates)[1]


def dynamics_argv(case: dict[str, Any], equilibrium: Path, run_dir: Path) -> list[str]:
    steps = int(round(float(case["final_time"]) / float(case["dt"])))
    return [
        sys.executable, str(DYNAMICS),
        "--run-dir", str(run_dir),
        "--run-tag", case["id"],
        "--equilibrium", str(equilibrium),
        "--order", str(case["order"]),
        "--dt", str(case["dt"]),
        "--num-steps", str(steps),
        "--supg-scale", "0.1",
        "--supg-tau-mode", "transient",
        "--flux-stabilization", "0",
    ]


def _equilibrium_discretization(equilibrium: Path) -> tuple[float | None, int, int]:
    """Read mesh size, polynomial order, and global DOFs from a checkpoint."""
    with np.load(equilibrium, allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint["metadata"].item()))
    order = int(metadata["order"])
    dofs = int(metadata["num_dofs"])
    mesh_size = None
    mesh_path = resolve_archive_path(str(metadata.get("mesh_path", "")))
    sidecar = mesh_path.with_suffix(mesh_path.suffix + ".json")
    if sidecar.is_file():
        mesh_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if mesh_metadata.get("requested_size") is not None:
            mesh_size = float(mesh_metadata["requested_size"])
    return mesh_size, order, dofs


def _best_measured_rank(
    manifest: dict[str, Any],
    case: dict[str, Any],
    fallback: int,
) -> int:
    """Apply the smaller-rank-within-10% rule to completed scaling repeats."""
    timings: dict[int, list[float]] = defaultdict(list)
    for candidate in manifest["cases"]:
        if candidate.get("kind") != "strong_scaling":
            continue
        if any(candidate.get(key) != case.get(key) for key in (
            "geometry", "order", "dof_target", "alpha_t1", "alpha_t2",
        )):
            continue
        result = candidate.get("result", {})
        elapsed = result.get("elapsed")
        if candidate.get("state") == "completed" and elapsed is not None:
            timings[int(candidate["ranks"])].append(float(elapsed))
    medians = {
        ranks: statistics.median(samples)
        for ranks, samples in timings.items()
        if samples
    }
    if not medians:
        return fallback
    return smaller_rank_within_ten_percent(medians)


def _mpi_argv(ranks: int, program_argv: Sequence[str]) -> list[str]:
    """Launch every numerical case through MPI, including one-rank cases."""
    return [
        "mpirun", "--bind-to", "core", "--map-by", "core",
        "-n", str(int(ranks)), *program_argv,
    ]


def _case_matches(case: dict[str, Any], kinds: set[str], ids: set[str],
                  campaign: str | None, geometries: set[str]) -> bool:
    return (
        (not kinds or case["kind"] in kinds)
        and (not ids or case["id"] in ids)
        and (not geometries or canonical_geometry_name(case["geometry"]) in geometries)
        and (campaign is None or campaign in case.get("campaigns", ()))
    )

def _terminal_optimizer_classification(case: dict[str, Any]) -> str:
    result = case.get("result", {})
    summary_value = result.get("summary")
    if not summary_value or not Path(str(summary_value)).is_file():
        return "pde_failure" if case.get("state") == "failed" else "incomplete"
    summary = parse_key_value_summary(Path(str(summary_value)))
    return classify_status(
        summary.get("finalStatus"), result.get("exit_code"),
        _number(summary.get("bestResidual")), 1.0e-12, 0,
    )


def _run_interruptible(
        argv: Sequence[str], *, cwd: Path, env: dict[str, str], stdout: Any, stderr: Any,
) -> int:
    process = subprocess.Popen(
        list(argv), cwd=cwd, env=env, stdout=stdout, stderr=stderr,
        start_new_session=True,
    )
    try:
        return process.wait()
    except KeyboardInterrupt:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise


def _render_attempt_storyboard_artifact(
        case: dict[str, Any],
        result: dict[str, Any],
        attempt_dir: Path,
        *,
        state: str,
) -> dict[str, Any]:
    """Render and record one failure-safe PNG for a concrete attempt.

    The plotting module is imported only after the child MPI process exits, so
    Matplotlib cannot interfere with PETSc/MPI initialization or timed work.
    """
    try:
        from projects.diocotron.studies.torsion_optimizer.figures.storyboards import render_attempt_storyboard
    except ImportError:
        from projects.diocotron.studies.torsion_optimizer.figures.storyboards import render_attempt_storyboard  # type: ignore

    row = {
        key: value
        for key, value in case.items()
        if key not in {"attempts", "result"}
    }
    row.update(result)
    row["state"] = str(state)
    destination = attempt_dir / "storyboard.png"
    trajectory_value = result.get("trajectory")
    equilibrium_value = result.get("equilibrium")
    trajectory = Path(str(trajectory_value)) if trajectory_value else None
    equilibrium = Path(str(equilibrium_value)) if equilibrium_value else None
    design_archive = trajectory if case.get("kind") == "geometry_overview" else None
    try:
        rendered = render_attempt_storyboard(
            row,
            destination,
            trajectory=trajectory,
            equilibrium=equilibrium,
            design_archive=design_archive,
        )
    except Exception as error:
        # The public renderer already recovers corrupt/missing scientific
        # artifacts. This final plotting-only fallback keeps the per-attempt
        # evidence contract even if an unexpected renderer bug is encountered.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        destination.parent.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
        axis.axis("off")
        axis.text(
            0.02,
            0.96,
            "\n".join((
                "FAILED/PARTIAL NUMERICAL ATTEMPT",
                f"case: {case.get('id', 'unknown')}",
                f"geometry: {case.get('geometry', 'unknown')}",
                f"state: {state}",
                f"renderer error: {type(error).__name__}: {error}",
            )),
            va="top",
            wrap=True,
        )
        figure.savefig(destination, dpi=180)
        plt.close(figure)
        rendered = {
            "file": str(destination),
            "status": "diagnostic_only",
            "source": "driver_placeholder",
            "failure_label": f"{type(error).__name__}: {error}",
        }
    result["storyboard"] = str(rendered["file"])
    result["storyboard_status"] = str(rendered.get("status", "unknown"))
    result["storyboard_source"] = str(rendered.get("source", "unknown"))
    warnings = rendered.get("warnings")
    if warnings:
        result["storyboard_warnings"] = [str(value) for value in warnings]
    return rendered


def _recover_orphan_attempts(case: dict[str, Any], run_root: Path) -> int:
    """Record attempt directories left behind by an interrupted driver."""
    case_root = run_root / "cases" / case["id"]
    if not case_root.is_dir():
        return 0
    known = {
        Path(str(attempt["run_dir"])).parent.resolve()
        for attempt in case.get("attempts", [])
        if attempt.get("run_dir")
    }
    recovered = 0
    for attempt_dir in sorted(case_root.glob("attempt_*")):
        if not attempt_dir.is_dir() or attempt_dir.resolve() in known:
            continue
        run_dir = attempt_dir / "run"
        result = {
            "exit_code": None,
            "error": "interrupted_before_manifest_commit",
            "run_dir": str(run_dir),
        }
        summary = run_dir / (
            "out/summary.json" if case["kind"] == "geometry_overview"
            else "out/summary.txt"
        )
        equilibrium = run_dir / "out/equilibrium.npz"
        trajectory = run_dir / "out/trajectory.npz"
        if summary.is_file():
            result["summary"] = str(summary)
        if equilibrium.is_file():
            result["equilibrium"] = str(equilibrium)
        if trajectory.is_file():
            result["trajectory"] = str(trajectory)
        _render_attempt_storyboard_artifact(
            case, result, attempt_dir, state="failed",
        )
        result["output_hashes"] = hash_outputs(attempt_dir)
        case.setdefault("attempts", []).append(result)
        case["result"] = result
        case["state"] = "failed"
        recovered += 1
    return recovered



def _valid_png(path: Path) -> bool:
    """Return whether a pre-existing attempt artifact is a nonempty PNG."""
    try:
        with path.open("rb") as handle:
            signature = handle.read(8)
        return path.stat().st_size > 100 and signature == b"\x89PNG\r\n\x1a\n"
    except OSError:
        return False


def backfill_attempt_storyboards(
        manifest: dict[str, Any],
        run_root: Path,
) -> tuple[int, int]:
    """Create missing PNG evidence for every recorded historical attempt.

    Existing valid PNGs are never rerendered. Output hashes are refreshed only
    after the derived artifact is present, preserving resumability while keeping
    this post-processing outside all numerical timing measurements.
    """
    rendered = 0
    recorded_existing = 0
    for case in manifest.get("cases", []):
        attempts = case.get("attempts", [])
        for attempt_index, attempt in enumerate(attempts, start=1):
            run_dir_value = attempt.get("run_dir")
            if run_dir_value:
                run_dir = Path(str(run_dir_value))
                attempt_dir = run_dir.parent if run_dir.name == "run" else run_dir
            else:
                attempt_dir = (
                    Path(run_root) / "cases" / str(case["id"])
                    / f"attempt_{attempt_index:03d}"
                )
                run_dir = attempt_dir / "run"
                attempt["run_dir"] = str(run_dir)
            if not attempt_dir.is_dir():
                continue

            destination = attempt_dir / "storyboard.png"
            if _valid_png(destination):
                attempt["storyboard"] = str(destination)
                attempt.setdefault("storyboard_status", "existing")
                attempt.setdefault("storyboard_source", "existing_png")
                recorded_existing += 1
            else:
                if not attempt.get("trajectory"):
                    candidate = run_dir / "out" / (
                        "overview.npz"
                        if case.get("kind") == "geometry_overview"
                        else "trajectory.npz"
                    )
                    if candidate.is_file():
                        attempt["trajectory"] = str(candidate)
                if not attempt.get("equilibrium"):
                    candidate = run_dir / "out" / "equilibrium.npz"
                    if candidate.is_file():
                        attempt["equilibrium"] = str(candidate)
                attempt_state = (
                    "completed"
                    if attempt.get("exit_code") == 0 and not attempt.get("error")
                    else "failed"
                )
                _render_attempt_storyboard_artifact(
                    case, attempt, attempt_dir, state=attempt_state,
                )
                rendered += 1
            attempt["output_hashes"] = hash_outputs(attempt_dir)

            current = case.get("result", {})
            if (
                current is attempt
                or (
                    current.get("run_dir")
                    and str(current.get("run_dir")) == str(attempt.get("run_dir"))
                )
            ):
                for key in (
                    "storyboard", "storyboard_status", "storyboard_source",
                    "storyboard_warnings", "output_hashes", "trajectory",
                    "equilibrium",
                ):
                    if key in attempt:
                        current[key] = attempt[key]
    return rendered, recorded_existing


def run_cases(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    args.run_root.mkdir(parents=True, exist_ok=True)
    calibration_path = args.run_root / "mesh_calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else {}
    kinds = set(args.kind or ())
    ids = set(args.case_id or ())
    geometries = {canonical_geometry_name(name) for name in (args.geometry or ())}
    selected = [case for case in manifest["cases"]
                if _case_matches(case, kinds, ids, args.campaign, geometries)]
    attempted = 0
    for case in selected:
        if _recover_orphan_attempts(case, args.run_root):
            write_json(args.manifest, manifest)
        required_failure = case.get("requires_failure_of")
        if required_failure:
            prerequisite = next(
                (candidate for candidate in manifest["cases"] if candidate["id"] == required_failure),
                None,
            )
            if prerequisite is None or prerequisite.get("state") == "planned":
                continue
            if prerequisite.get("state") == "completed":
                case["state"] = "deferred"
                case["deferred_reason"] = f"rescue prerequisite {required_failure} succeeded"
                write_json(args.manifest, manifest)
                continue
        required_non_success = case.get("requires_non_success_of")
        if required_non_success:
            prerequisite = next(
                (candidate for candidate in manifest["cases"]
                 if candidate["id"] == required_non_success),
                None,
            )
            if prerequisite is None or prerequisite.get("state") == "planned":
                continue
            prerequisite_class = _terminal_optimizer_classification(prerequisite)
            if prerequisite.get("state") == "deferred" or prerequisite_class in SUCCESS_CLASSIFICATIONS:
                case["state"] = "deferred"
                case["deferred_reason"] = (
                    f"working-storyboard prerequisite {required_non_success} "
                    f"ended as {prerequisite_class}"
                )
                write_json(args.manifest, manifest)
                continue

        parent_id = case.get("parent_case_id")
        if parent_id:
            try:
                _parent_equilibrium(manifest, case)
            except LookupError:
                continue
            except RuntimeError as error:
                case["state"] = "deferred"
                case["deferred_reason"] = str(error)
                write_json(args.manifest, manifest)
                continue

        if case["state"] == "completed":
            valid = completed_case_outputs_valid(case)
            if valid and not args.rerun_completed:
                continue
            if not valid and not (args.retry or args.rerun_completed):
                continue
        if case["state"] in {"failed", "deferred"} and not args.retry:
            continue
        if args.limit is not None and attempted >= args.limit:
            break
        attempted += 1
        attempt_number = len(case["attempts"]) + 1
        attempt_dir = args.run_root / "cases" / case["id"] / f"attempt_{attempt_number:03d}"
        while attempt_dir.exists():
            attempt_number += 1
            attempt_dir = args.run_root / "cases" / case["id"] / f"attempt_{attempt_number:03d}"
        try:
            if case["kind"].startswith("stationarity"):
                equilibrium = _latest_successful_equilibrium(manifest, case)
                mesh_size, checkpoint_order, checkpoint_dofs = _equilibrium_discretization(equilibrium)
                ranks = int(case.get("ranks") or select_mumps_ranks(
                    mesh_size, checkpoint_order, checkpoint_dofs,
                ))
                program_argv = dynamics_argv(case, equilibrium, attempt_dir / "run")
                inputs = {"equilibrium": sha256_file(equilibrium)}
            else:
                initial_equilibrium = _parent_equilibrium(manifest, case)
                mesh_path, mesh_metadata = ensure_calibrated_mesh(case, args.run_root, calibration)
                write_json(calibration_path, calibration)
                policy_ranks = select_mumps_ranks(
                    float(mesh_metadata["requested_size"]), int(case["order"]), int(mesh_metadata["dofs"])
                )
                ranks = int(case.get("ranks") or policy_ranks)
                if case.get("rank_selector") == "best_measured":
                    ranks = _best_measured_rank(manifest, case, policy_ranks)
                program_argv = (
                    overview_argv(case, mesh_path, attempt_dir / "run")
                    if case["kind"] == "geometry_overview"
                    else optimizer_argv(
                        case,
                        mesh_path,
                        attempt_dir / "run",
                        initial_equilibrium=initial_equilibrium,
                    )
                )
                inputs = {"mesh": sha256_file(mesh_path)}
                if initial_equilibrium is not None:
                    inputs["parent_equilibrium"] = sha256_file(initial_equilibrium)
            argv = _mpi_argv(ranks, program_argv)
            environment = os.environ.copy()
            environment.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            provenance = {
                "argv": argv,
                "environment": {key: environment.get(key) for key in (
                    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PATH", "PYTHONPATH"
                )},
                "mpi_ranks": ranks,
                "cpu": platform.processor(),
                "platform": platform.platform(),
                "python": sys.version,
                "packages": _package_versions(),
                "git": _git_provenance(),
                "input_hashes": inputs,
            }
            if args.dry_run:
                print(" ".join(argv))
                continue
            attempt_dir.mkdir(parents=True, exist_ok=False)
            write_json(attempt_dir / "provenance.json", provenance)
            started = time.time()
            with (attempt_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (
                attempt_dir / "stderr.log"
            ).open("w", encoding="utf-8") as stderr:
                return_code = _run_interruptible(argv, cwd=REPO_ROOT, env=environment, stdout=stdout, stderr=stderr)
            run_dir = attempt_dir / "run"
            result = {
                "exit_code": return_code,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed": time.time() - started,
                "run_dir": str(run_dir),
            }
            if case["kind"].startswith("stationarity"):
                result["summary"] = str(run_dir / "summary.json")
            elif case["kind"] == "geometry_overview":
                result["summary"] = str(run_dir / "out" / "summary.json")
                result["trajectory"] = str(run_dir / "out" / "overview.npz")
            else:
                summary = run_dir / "out" / "summary.txt"
                equilibrium = run_dir / "out" / "equilibrium.npz"
                result.update({
                    "summary": str(summary),
                    "equilibrium": str(equilibrium),
                    "trajectory": str(run_dir / "out" / "trajectory.npz"),
                })
                if (
                    case.get("kind") == "inner_newton_policy_v3"
                    and case.get("inner_accuracy") == "adaptive"
                ):
                    result["inexact_newton"] = str(run_dir / "logs" / "inexact_newton.csv")
            # The first hash snapshot permits semantic validation; the final
            # snapshot below also covers the derived storyboard artifact.
            result["output_hashes"] = hash_outputs(attempt_dir)
            case["attempts"].append(result)
            case["result"] = result
            case["state"] = "completed" if return_code == 0 else "failed"
            if case["state"] == "completed" and not completed_case_outputs_valid(case):
                case["state"] = "failed"
                result["validation_error"] = (
                    "process exited successfully but required outputs failed semantic validation"
                )
            _render_attempt_storyboard_artifact(
                case, result, attempt_dir, state=case["state"],
            )
            result["output_hashes"] = hash_outputs(attempt_dir)
        except Exception as error:
            if args.dry_run:
                print(f"SKIP {case['id']}: {type(error).__name__}: {error}")
                continue
            attempt_dir.mkdir(parents=True, exist_ok=True)
            run_dir = attempt_dir / "run"
            result = {
                "exit_code": None,
                "error": f"{type(error).__name__}: {error}",
                "run_dir": str(run_dir),
            }
            summary = run_dir / (
                "out/summary.json" if case["kind"] == "geometry_overview"
                else (
                    "summary.json"
                    if case["kind"].startswith("stationarity")
                    else "out/summary.txt"
                )
            )
            equilibrium = run_dir / "out/equilibrium.npz"
            trajectory = run_dir / "out/trajectory.npz"
            if summary.is_file():
                result["summary"] = str(summary)
            if equilibrium.is_file():
                result["equilibrium"] = str(equilibrium)
            if trajectory.is_file():
                result["trajectory"] = str(trajectory)
            case["state"] = "failed"
            _render_attempt_storyboard_artifact(
                case, result, attempt_dir, state="failed",
            )
            result["output_hashes"] = hash_outputs(attempt_dir)
            case["attempts"].append(result)
            case["result"] = result
        write_json(args.manifest, manifest)
    return 0


def aggregate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten every case and its structured diagnostics into analysis rows."""
    rows: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        result = case.get("result", {})
        row = {key: value for key, value in case.items() if key not in {"attempts", "result"}}
        row["exit_code"] = result.get("exit_code")
        row["run_dir"] = result.get("run_dir")
        row["equilibrium"] = result.get("equilibrium")
        row["wall_seconds"] = result.get("elapsed")
        if case.get("state") not in {"completed", "failed"}:
            row["classification"] = "incomplete" if case.get("state") == "planned" else case.get("state")
            rows.append(row)
            continue

        if case["kind"] == "geometry_overview":
            summary_path = Path(result["summary"]) if result.get("summary") else None
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path and summary_path.is_file() else {}
            row.update(summary)
            row["trajectoryArchive"] = result.get("trajectory") or summary.get("trajectory")
            row["classification"] = (
                "overview_completed" if case.get("state") == "completed" and summary else "pde_failure"
            )
        elif case["kind"].startswith("stationarity"):
            summary_path = Path(result["summary"]) if result.get("summary") else None
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path and summary_path.is_file() else {}
            for key, value in summary.items():
                if key not in {
                    "final_diagnostics", "equilibrium_handoff",
                    "equilibrium_poisson_consistency",
                }:
                    row[key] = value
            for key, value in summary.get("final_diagnostics", {}).items():
                row[f"stationarity_{key}"] = value
            for key, value in (summary.get("equilibrium_handoff") or {}).items():
                row[f"handoff_{key}"] = value
            for key, value in (summary.get("equilibrium_poisson_consistency") or {}).items():
                row[f"poisson_consistency_{key}"] = value
            row["classification"] = (
                "stationarity_completed"
                if case.get("state") == "completed" and result.get("exit_code") == 0 and summary
                else "pde_failure"
            )
        else:
            summary_path = Path(result["summary"]) if result.get("summary") else None
            summary = parse_key_value_summary(summary_path) if summary_path else {}
            row.update(summary)
            residual = _number(summary.get("bestResidual"))
            run_dir = Path(result.get("run_dir", ""))
            log_dir = run_dir / "logs"
            newton_csv = log_dir / "newton.csv"
            augment_aggregate_row(row, log_dir)
            newton_records: list[dict[str, str]] = []
            if newton_csv.is_file():
                with newton_csv.open(newline="", encoding="utf-8") as handle:
                    newton_records = list(csv.DictReader(handle))
            capped = sum(int(record.get("linear_cap", "0") or 0) for record in newton_records)
            row["capped_trials"] = capped
            row["newton_records"] = len(newton_records)
            for source, output, reducer in (
                ("ksp_iterations", "ksp_iterations_total", sum),
                ("ksp_time", "ksp_seconds_total", sum),
                ("assembly_time", "assembly_seconds_total", sum),
                ("line_search_time", "line_search_seconds_total", sum),
                ("backtracks", "backtracks_total", sum),
                ("damping", "damping_min", min),
                ("residual", "newton_residual_min", min),
            ):
                values = [
                    number
                    for record in newton_records
                    if (number := _number(record.get(source))) is not None
                ]
                row[output] = reducer(values) if values else None

            phases_csv = log_dir / "phases.csv"
            if phases_csv.is_file():
                with phases_csv.open(newline="", encoding="utf-8") as handle:
                    for record in csv.DictReader(handle):
                        phase = str(record["phase"])
                        row[f"phase_{phase}_seconds"] = _number(record.get("elapsed"))
                        row[f"phase_{phase}_calls"] = int(record.get("calls", "0") or 0)
                row["timeTotal"] = row.get("phase_total_seconds")

            optimization_csv = log_dir / "optimization.csv"
            if optimization_csv.is_file():
                with optimization_csv.open(newline="", encoding="utf-8") as handle:
                    optimization_records = list(csv.DictReader(handle))
                row.update(detect_threshold_plateau_v3(optimization_records))
                rejected = [
                    record for record in optimization_records
                    if str(record.get("accepted", "")).strip().lower() in {"0", "false"}
                ]
                row["accepted_outer_states"] = sum(
                    str(record.get("accepted", "")).strip().lower() in {"1", "true"}
                    for record in optimization_records
                )
                row["rejected_trials"] = len(rejected)
                row["rejected_trial_seconds"] = sum(
                    _number(record.get("stepTime")) or 0.0 for record in rejected
                )
                for source, output in (
                    ("branchOverlap", "branch_overlap_min"),
                    ("rhoRatio", "acceptance_ratio_min"),
                    ("trustRadius", "trust_radius_min"),
                ):
                    values = [
                        value for record in optimization_records
                        if (value := _number(record.get(source))) is not None
                    ]
                    row[output] = min(values) if values else None

            coercivity_csv = log_dir / "newton_spd.csv"
            if coercivity_csv.is_file():
                with coercivity_csv.open(newline="", encoding="utf-8") as handle:
                    coercivity_records = list(csv.DictReader(handle))
                margins = [
                    value for record in coercivity_records
                    if (value := _number(record.get("muMin"))) is not None
                ]
                row["coercivity_margin_min"] = min(margins) if margins else None
                row["coercivity_diagnostic_seconds"] = sum(
                    _number(record.get("elapsed")) or 0.0
                    for record in coercivity_records
                )

            inexact_csv = log_dir / "inexact_newton.csv"
            if inexact_csv.is_file():
                with inexact_csv.open(newline="", encoding="utf-8") as handle:
                    inexact_records = list(csv.DictReader(handle))
                summaries = group_inexact_snapshot_rows(inexact_records)
                row["inexact_snapshot_records"] = len(inexact_records)
                row["inexact_policy_summaries"] = {
                    f"{policy}|{matrix_policy}": summary
                    for (policy, matrix_policy), summary in summaries.items()
                }
                row["inexact_any_admissible"] = any(item["admissible"] for item in summaries.values())
            mesh_file = summary.get("meshFile")
            if mesh_file:
                mesh_path = Path(mesh_file)
                metadata_path = mesh_path.with_suffix(mesh_path.suffix + ".json")
                if metadata_path.is_file():
                    mesh_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    for source, output in (
                        ("requested_size", "mesh_size"),
                        ("h_max", "h_max"),
                        ("domain_diameter", "domain_diameter"),
                        ("cells", "mesh_cells"),
                        ("dofs", "mesh_dofs"),
                    ):
                        if source in mesh_metadata:
                            row[output] = mesh_metadata[source]
                    if _number(row.get("h_max")) and _number(row.get("domain_diameter")):
                        row["normalized_h"] = float(row["h_max"]) / float(row["domain_diameter"])

            c1 = _number(summary.get("bestC1Phi"))
            c2 = _number(summary.get("bestC2Phi"))
            eps = _number(summary.get("bestEpsPhi"))
            if c1 is not None and c2 is not None and c2 > c1:
                row["bestWidth"] = c2 - c1
                row["epsilon_ratio_observed"] = None if eps is None else eps / (c2 - c1)
            row["classification"] = classify_status(
                summary.get("finalStatus"),
                result.get("exit_code") if result.get("exit_code") is not None else 1,
                residual, 1.0e-12, capped,
            )
        rows.append(row)
    return rows


def _cached_reference_comparison(
    reference: Path,
    current: Path,
    cache_dir: Path,
) -> dict[str, float]:
    """Run a hash-keyed comparison under the repository MPI rank policy."""
    reference = reference.resolve()
    current = current.resolve()
    identity = hashlib.sha256(
        canonical_json({
            "format": 1,
            "comparator": sha256_file(EQUILIBRIUM_COMPARATOR),
            "reference": sha256_file(reference),
            "current": sha256_file(current),
        }).encode("utf-8")
    ).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / f"{identity}.json"
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))

    mesh_size, order, dofs = _equilibrium_discretization(reference)
    ranks = select_mumps_ranks(mesh_size, order, dofs)
    python = os.environ.get("HDGFEM_DOLFINX_PYTHON", sys.executable)
    argv = _mpi_argv(ranks, (
        python,
        str(EQUILIBRIUM_COMPARATOR),
        str(reference),
        str(current),
        "--output",
        str(output),
    ))
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    completed = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 or not output.is_file():
        output.unlink(missing_ok=True)
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"reference comparison failed with exit code {completed.returncode}: {detail[-2000:]}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def attach_reference_comparisons(
    rows: list[dict[str, Any]],
    bundle: Path,
    *,
    comparison_runner: Callable[[Path, Path], dict[str, float]] | None = None,
) -> int:
    """Attach adjacent-grid and selected-reference field/set comparisons."""
    eligible = [
        row for row in rows
        if row.get("kind") in {"mesh_order", "mesh_order_difficult", "mesh_order_refinement"}
        and row.get("classification") in SUCCESS_CLASSIFICATIONS
        and row.get("equilibrium")
        and Path(str(row["equilibrium"])).is_file()
    ]
    if not eligible:
        return 0
    cache_dir = bundle / "data" / "reference_comparisons"
    runner = comparison_runner or (
        lambda reference, current: _cached_reference_comparison(
            reference, current, cache_dir
        )
    )
    completed = 0

    per_order: dict[tuple[str, int, float, float], list[dict[str, Any]]] = defaultdict(list)
    per_band: dict[tuple[str, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        band = (
            str(row["geometry"]),
            round(float(row["alpha_t1"]), 12),
            round(float(row["alpha_t2"]), 12),
        )
        per_order[(band[0], int(row["order"]), band[1], band[2])].append(row)
        per_band[band].append(row)

    for group in per_order.values():
        ordered = sorted(group, key=lambda row: int(float(row.get("ndof", row["dof_target"]))))
        for coarse, fine in zip(ordered, ordered[1:]):
            try:
                metrics = runner(Path(fine["equilibrium"]), Path(coarse["equilibrium"]))
            except Exception as error:
                fine["comparison_error"] = f"{type(error).__name__}: {error}"
                continue
            fine.update({f"two_grid_{key}": value for key, value in metrics.items()})
            fine["two_grid_change"] = max(
                float(metrics["relative_l2"]), float(metrics["relative_h1"])
            )
            completed += 1

    for group in per_band.values():
        endpoints: list[dict[str, Any]] = []
        for order in sorted({int(row["order"]) for row in group}):
            order_rows = [row for row in group if int(row["order"]) == order]
            endpoints.append(max(
                order_rows,
                key=lambda row: int(float(row.get("ndof", row["dof_target"]))),
            ))
        highest_target = max(int(row["dof_target"]) for row in endpoints)
        candidates = [row for row in endpoints if int(row["dof_target"]) == highest_target]
        reference = min(
            candidates,
            key=lambda row: (
                float(row.get("two_grid_change", math.inf)),
                -int(float(row.get("ndof", row["dof_target"]))),
            ),
        )
        reference["is_numerical_reference"] = True
        for row in group:
            row["reference_id"] = reference["id"]
            try:
                metrics = runner(Path(reference["equilibrium"]), Path(row["equilibrium"]))
            except Exception as error:
                row["comparison_error"] = f"{type(error).__name__}: {error}"
                continue
            row.update(metrics)
            completed += 1
    references_by_band = {
        (str(row["geometry"]), round(float(row["alpha_t1"]), 12),
         round(float(row["alpha_t2"]), 12)): row
        for row in rows
        if row.get("is_numerical_reference") and row.get("equilibrium")
    }
    for row in rows:
        if row.get("kind") != "fixed_mesh_p":
            continue
        if row.get("classification") not in SUCCESS_CLASSIFICATIONS:
            continue
        if not row.get("equilibrium") or not Path(str(row["equilibrium"])).is_file():
            continue
        key = (
            str(row["geometry"]),
            round(float(row["alpha_t1"]), 12),
            round(float(row["alpha_t2"]), 12),
        )
        reference = references_by_band.get(key)
        if reference is None:
            continue
        row["reference_id"] = reference["id"]
        try:
            metrics = runner(Path(reference["equilibrium"]), Path(row["equilibrium"]))
        except Exception as error:
            row["comparison_error"] = f"{type(error).__name__}: {error}"
            continue
        row.update(metrics)
        completed += 1
    return completed


def append_adaptive_cases(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> int:
    """Append data-driven confirmations, storyboards, and extra mesh levels."""
    parameters = [
        *fine_confirmation_parameters(
            rows,
            fine_targets={
                "smooth_star": 700_000,
                "pacman": 500_000,
                "horseshoe": 500_000,
                "iter": 500_000,
            },
        ),
        *additional_refinement_parameters(rows),
        *storyboard_confirmation_parameters(rows),
    ]
    existing = {case["id"] for case in manifest["cases"]}
    appended = 0
    for values in parameters:
        values = dict(values)
        kind = str(values.pop("kind"))
        case = make_case(kind, **values)
        if case["id"] not in existing:
            manifest["cases"].append(case)
            existing.add(case["id"])
            appended += 1
    return appended


def storyboard_confirmation_parameters(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select a usable robustness result when a geometry lacks a PDE storyboard.

    A confirmation rerun is needed because robustness cases intentionally do
    not write the large trajectory archive. Strict and certified solutions
    are preferred; a PDE-converged geometric plateau remains useful as an
    explicitly labelled formation path when no geometrically successful band
    is available.
    """
    usable_classes = {
        "strict_convergence",
        "certified_subband_convergence",
        "geometrically_unsuccessful_pde_converged",
        "pde_converged_with_capped_trials",
    }
    priority = {
        "strict_convergence": 0,
        "certified_subband_convergence": 1,
        "geometrically_unsuccessful_pde_converged": 2,
        "pde_converged_with_capped_trials": 3,
    }
    parameters: list[dict[str, Any]] = []
    for geometry in GEOMETRY_ORDER:
        existing = [
            row for row in rows
            if str(row.get("geometry")) == geometry
            and str(row.get("kind", "")).startswith("trajectory")
            and row.get("classification") in usable_classes
            and row.get("trajectoryArchive")
            and Path(str(row["trajectoryArchive"])).is_file()
        ]
        if existing:
            continue
        candidates = [
            row for row in rows
            if row.get("kind") == "robustness"
            and str(row.get("geometry")) == geometry
            and row.get("classification") in usable_classes
        ]
        if not candidates:
            continue

        def score(row: dict[str, Any]) -> tuple[float, float, float]:
            discrepancy = sum(
                abs(float(value))
                for value in (row.get("bestLeakageRel"), row.get("bestMissingRel"))
                if _number(value) is not None
            )
            residual = _number(row.get("bestResidual"))
            return (
                float(priority[str(row["classification"])]),
                discrepancy,
                math.inf if residual is None else float(residual),
            )

        selected = min(candidates, key=score)
        source_id = selected.get("id") or (
            f"robustness-{geometry}-{float(selected['alpha_t1']):.12g}-"
            f"{float(selected['alpha_t2']):.12g}"
        )
        parameters.append({
            "kind": "trajectory_confirmation",
            "geometry": geometry,
            "order": int(selected.get("order", 4)),
            "dof_target": int(selected.get("dof_target", 200_000)),
            "alpha_t1": float(selected["alpha_t1"]),
            "alpha_t2": float(selected["alpha_t2"]),
            "trajectory_every": 1,
            "source_case_id": str(source_id),
            "source_classification": str(selected["classification"]),
        })
    return parameters


def write_aggregate(rows: list[dict[str, Any]], bundle: Path) -> None:
    data_dir = bundle / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    write_json(data_dir / "aggregated.json", rows)
    fields = sorted({key for row in rows for key in row})
    with (data_dir / "aggregated.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: canonical_json(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _load_rows(bundle: Path) -> list[dict[str, Any]]:
    path = bundle / "data" / "aggregated.json"
    if not path.is_file():
        raise FileNotFoundError(f"run aggregate first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def create_figures(bundle: Path, *, stage: str = "preliminary") -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load_rows(bundle)
    output = bundle / "figures"
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    colors = {
        "strict_convergence": "#2b8cbe",
        "certified_subband_convergence": "#41ab5d",
        "geometrically_unsuccessful_pde_converged": "#fdae6b",
        "pde_converged_with_capped_trials": "#756bb1",
        "pde_failure": "#de2d26",
        "incomplete": "#bdbdbd",
    }
    for geometry in GEOMETRY_ORDER:
        subset = [row for row in rows if row["kind"] == "robustness" and row["geometry"] == geometry]
        if not subset:
            continue
        fig, ax = plt.subplots(figsize=(5.2, 3.9), constrained_layout=True)
        for row in subset:
            center = 0.5 * (float(row["alpha_t1"]) + float(row["alpha_t2"]))
            width = float(row["alpha_t2"]) - float(row["alpha_t1"])
            ax.scatter(center, width, s=48, color=colors.get(row["classification"], "black"), edgecolor="black", linewidth=0.35)
        ax.set(xlabel="torsion-band center", ylabel="torsion-band width", title=geometry.replace("_", " "))
        path = output / f"robustness_{geometry}.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        written.append(path)

    scaling = [row for row in rows if row["kind"] == "strong_scaling" and _number(row.get("timeTotal"))]
    if scaling:
        fig, ax = plt.subplots(figsize=(5.2, 3.9), constrained_layout=True)
        groups: dict[tuple[str, int], list[float]] = defaultdict(list)
        for row in scaling:
            groups[(row["difficulty"], int(row["ranks"]))].append(float(row["timeTotal"]))
        for difficulty in ("easy", "difficult"):
            ranks = sorted(rank for label, rank in groups if label == difficulty)
            medians = [statistics.median(groups[(difficulty, rank)]) for rank in ranks]
            q1 = [float(np.quantile(groups[(difficulty, rank)], 0.25)) for rank in ranks]
            q3 = [float(np.quantile(groups[(difficulty, rank)], 0.75)) for rank in ranks]
            ax.errorbar(ranks, medians, yerr=(np.subtract(medians, q1), np.subtract(q3, medians)), marker="o", label=difficulty)
        ax.set(xlabel="MPI ranks", ylabel="total time [s]")
        ax.legend()
        path = output / "strong_scaling_time.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        written.append(path)

    successful = [
        row for row in rows
        if row.get("classification") in SUCCESS_CLASSIFICATIONS
    ]
    for geometry in GEOMETRY_ORDER:
        subset = [
            row for row in successful
            if row.get("kind") == "robustness" and row.get("geometry") == geometry
        ]
        metric_specs = (
            ("bestCertifiedAreaRel", "certified area fraction"),
            ("bestCertifiedLeakageRel", "certified leakage"),
            ("bestWidth", r"final $\Delta c_\phi$"),
        )
        if subset and all(all(_number(row.get(key)) is not None for row in subset) for key, _ in metric_specs):
            fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.8), constrained_layout=True)
            centers = np.asarray([
                0.5 * (float(row["alpha_t1"]) + float(row["alpha_t2"])) for row in subset
            ])
            widths = np.asarray([
                float(row["alpha_t2"]) - float(row["alpha_t1"]) for row in subset
            ])
            for ax, (key, title) in zip(axes, metric_specs, strict=True):
                values = np.asarray([float(row[key]) for row in subset])
                image = ax.scatter(centers, widths, c=values, cmap="viridis", s=65, edgecolor="black", linewidth=0.3)
                fig.colorbar(image, ax=ax)
                ax.set(xlabel="torsion-band center", ylabel="torsion-band width", title=title)
            path = output / f"robustness_metrics_{geometry}.png"
            fig.savefig(path, dpi=300)
            plt.close(fig)
            written.append(path)

    for geometry in GEOMETRY_ORDER:
        subset = [
            row for row in successful
            if str(row.get("kind", "")).startswith("mesh_order")
            and row.get("geometry") == geometry
            and _number(row.get("ndof")) is not None
        ]
        if len(subset) < 2:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.8), constrained_layout=True)
        for order in (2, 4, 6):
            ordered = sorted(
                (row for row in subset if int(row["order"]) == order),
                key=lambda row: int(float(row["ndof"])),
            )
            if not ordered:
                continue
            dofs = np.asarray([float(row["ndof"]) for row in ordered])
            c1 = np.asarray([float(row["bestC1Phi"]) for row in ordered])
            c2 = np.asarray([float(row["bestC2Phi"]) for row in ordered])
            residual = np.asarray([float(row["bestResidual"]) for row in ordered])
            axes[0].plot(dofs, c1, marker="o", label=f"P{order} lower")
            axes[0].plot(dofs, c2, marker="s", linestyle="--", label=f"P{order} upper")
            axes[1].loglog(dofs, residual, marker="o", label=f"P{order}")
            errors = [
                _number(row.get("relative_l2")) for row in ordered
            ]
            if any(value is not None and value > 0.0 for value in errors):
                axes[2].loglog(
                    dofs,
                    [np.nan if value is None else value for value in errors],
                    marker="o",
                    label=f"P{order}",
                )
        axes[0].set(xscale="log", xlabel="global DOFs", ylabel=r"$c_\phi$", title="threshold convergence")
        axes[1].set(xlabel="global DOFs", ylabel="final PDE residual", title="PDE residual")
        axes[2].set(xlabel="global DOFs", ylabel=r"relative $L^2$ error", title="reference error")
        for ax in axes:
            if ax.lines:
                ax.legend(fontsize=7)
        path = output / f"mesh_order_convergence_{geometry}.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        written.append(path)

    epsilon_rows = [
        row for row in successful
        if _number(row.get("bestWidth")) is not None
        and _number(row.get("bestEpsPhi")) is not None
    ]
    if epsilon_rows:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
        widths = np.asarray([float(row["bestWidth"]) for row in epsilon_rows])
        epsilons = np.asarray([float(row["bestEpsPhi"]) for row in epsilon_rows])
        geometry_colors = {name: index for index, name in enumerate(GEOMETRY_ORDER)}
        color_values = [geometry_colors[row["geometry"]] for row in epsilon_rows]
        axes[0].scatter(widths, epsilons, c=color_values, cmap="tab10", edgecolor="black", linewidth=0.3)
        interval = np.linspace(0.0, 1.05 * max(widths), 100)
        axes[0].plot(interval, EPSILON_RATIO * interval, color="black", linestyle="--", label=r"$0.08\Delta c_\phi$")
        axes[0].set(xlabel=r"$\Delta c_\phi$", ylabel=r"$\varepsilon_\phi$", title="epsilon-width invariant")
        axes[0].legend()
        centers = np.asarray([
            0.5 * (float(row["alpha_t1"]) + float(row["alpha_t2"])) for row in epsilon_rows
        ])
        axes[1].scatter(epsilons, centers, c=color_values, cmap="tab10", edgecolor="black", linewidth=0.3)
        axes[1].set(xlabel=r"final $\varepsilon_\phi$", ylabel="torsion-band center", title="outcome smoothing scale")
        path = output / "epsilon_width_coupling.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        written.append(path)

    if scaling:
        phases = ("hminus1_search", "homotopy", "sensitivities", "reduced_gradients", "final_projection")
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), constrained_layout=True)
        for difficulty in ("easy", "difficult"):
            groups = defaultdict(list)
            for row in scaling:
                if row["difficulty"] == difficulty:
                    groups[int(row["ranks"])].append(row)
            if not groups:
                continue
            ranks = sorted(groups)
            medians = np.asarray([
                statistics.median(float(row["timeTotal"]) for row in groups[rank]) for rank in ranks
            ])
            baseline_time = medians[0]
            speedup = baseline_time / medians
            efficiency = speedup / (np.asarray(ranks) / ranks[0])
            axes[0].plot(ranks, speedup, marker="o", label=difficulty)
            axes[1].plot(ranks, efficiency, marker="o", label=difficulty)
            if difficulty == "easy":
                bottoms = np.zeros(len(ranks))
                for phase in phases:
                    values = np.asarray([
                        statistics.median(
                            float(row.get(f"phase_{phase}_seconds") or 0.0)
                            for row in groups[rank]
                        )
                        for rank in ranks
                    ])
                    axes[2].bar(ranks, values, bottom=bottoms, label=phase.replace("_", " "))
                    bottoms += values
        axes[0].set(xlabel="MPI ranks", ylabel="speedup", title="strong-scaling speedup")
        axes[1].set(xlabel="MPI ranks", ylabel="parallel efficiency", title="efficiency")
        axes[2].set(xlabel="MPI ranks", ylabel="median time [s]", title="easy-case phase breakdown")
        axes[0].legend()
        axes[1].legend()
        axes[2].legend(fontsize=6)
        path = output / "strong_scaling_summary.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        written.append(path)

    stationarity = [
        row for row in rows
        if row.get("classification") == "stationarity_completed"
    ]
    if stationarity:
        fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.5), constrained_layout=True)
        specs = (
            ("stationarity_rho_change_l2_rel", "density relative $L^2$ drift"),
            ("stationarity_rho_change_linf", "density $L^\\infty$ drift"),
            ("stationarity_mass_rel_drift", "relative mass drift"),
            ("stationarity_energy_rel_drift", "relative energy drift"),
            ("stationarity_advective_defect_rel", "relative advective defect"),
        )
        grouped = defaultdict(list)
        for row in stationarity:
            if float(row["final_time"]) <= 0.1:
                grouped[(str(row["geometry"]), float(row["alpha_t1"]), float(row["alpha_t2"]))].append(row)
        consistency_labels = []
        handoff_values = []
        poisson_values = []
        for (geometry, alpha_t1, alpha_t2), group in sorted(grouped.items()):
            subset = sorted(
                group,
                key=lambda row: float(row["dt"]),
            )
            if not subset:
                continue
            label = f"{geometry} [{alpha_t1:.2f},{alpha_t2:.2f}]"
            for ax, (key, title) in zip(axes.flat[:5], specs, strict=True):
                values = [_number(row.get(key)) for row in subset]
                if all(value is not None for value in values):
                    ax.loglog([float(row["dt"]) for row in subset], np.maximum(np.abs(values), 1e-18),
                              marker="o", label=label)
                    ax.set(xlabel=r"$\Delta t$", ylabel=title)
            consistency_labels.append(label)
            handoff_values.append(max(float(subset[0].get("handoff_relative_h1") or 0.0), 1e-18))
            poisson_values.append(max(float(subset[0].get("poisson_consistency_relative_h1") or 0.0), 1e-18))
        timestep_ticks = sorted({float(row["dt"]) for group in grouped.values() for row in group})
        for ax in axes.flat[:5]:
            ax.set_xticks(timestep_ticks, [f"{value:g}" for value in timestep_ticks])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        consistency_ax = axes.flat[5]
        positions = np.arange(len(consistency_labels), dtype=float)
        consistency_ax.scatter(positions - 0.08, handoff_values, marker="o", label="checkpoint handoff")
        consistency_ax.scatter(positions + 0.08, poisson_values, marker="s", label="Poisson consistency")
        consistency_ax.set_yscale("log")
        consistency_ax.set_xticks(positions, consistency_labels, rotation=20, ha="right")
        consistency_ax.set(ylabel="relative $H^1$ error", title="exact handoff and consistency")
        consistency_ax.legend(fontsize=7)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            axes.flat[0].legend(handles, labels, loc="best", fontsize=7)
        path = output / "guiding_center_stationarity.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        written.append(path)

    initialization_done: set[str] = set()
    for row in rows:
        geometry = str(row["geometry"])
        if geometry in initialization_done or not row.get("run_dir"):
            continue
        initialization_csv = Path(row["run_dir"]) / "logs" / "initialization.csv"
        if not initialization_csv.is_file():
            continue
        with initialization_csv.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        samples = [
            record for record in records
            if _number(record.get("c1")) is not None
            and _number(record.get("c2")) is not None
            and _number(record.get("psiHminus1")) is not None
        ]
        if not samples:
            continue
        fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
        image = ax.scatter(
            [float(record["c1"]) for record in samples],
            [float(record["c2"]) for record in samples],
            c=[float(record["psiHminus1"]) for record in samples],
            cmap="viridis",
            s=22,
        )
        accepted = [record for record in samples if str(record.get("accepted", "")).lower() in {"1", "true"}]
        if accepted:
            ax.scatter(
                [float(record["c1"]) for record in accepted],
                [float(record["c2"]) for record in accepted],
                marker="*", s=110, color="#d7301f", label="selected/accepted",
            )
            ax.legend()
        fig.colorbar(image, ax=ax, label=r"$H^{-1}$ objective")
        ax.set(xlabel=r"$c_{1,\phi}$", ylabel=r"$c_{2,\phi}$", title=f"{geometry}: initializer landscape")
        path = output / f"hminus1_landscape_{geometry}.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        written.append(path)
        initialization_done.add(geometry)

        homotopy = [
            record for record in records
            if _number(record.get("lambdaTrial")) is not None
            and _number(record.get("elapsed")) is not None
        ]
        if homotopy:
            fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), constrained_layout=True)
            lambdas = np.asarray([float(record["lambdaTrial"]) for record in homotopy])
            elapsed = np.asarray([float(record["elapsed"]) for record in homotopy])
            iterations = np.asarray([
                _number(record.get("newtonIterations")) or 0.0 for record in homotopy
            ])
            accepted_mask = np.asarray([
                str(record.get("accepted", "")).lower() in {"1", "true"}
                for record in homotopy
            ])
            axes[0].scatter(lambdas, elapsed, c=np.where(accepted_mask, 1, 0), cmap="coolwarm", edgecolor="black")
            axes[1].scatter(lambdas, iterations, c=np.where(accepted_mask, 1, 0), cmap="coolwarm", edgecolor="black")
            axes[0].set(xlabel=r"homotopy $\lambda$", ylabel="stage time [s]", title="work per continuation stage")
            axes[1].set(xlabel=r"homotopy $\lambda$", ylabel="Newton iterations", title="nonlinear work")
            path = output / f"homotopy_work_{geometry}.png"
            fig.savefig(path, dpi=300)
            plt.close(fig)
            written.append(path)

    for row in rows:
        if row.get("kind") != "coercivity" or row.get("state") != "completed" or not row.get("run_dir"):
            continue
        log_dir = Path(row["run_dir"]) / "logs"

        def records(name: str) -> list[dict[str, str]]:
            path = log_dir / name
            if not path.is_file():
                return []
            with path.open(newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))

        newton = records("newton.csv")
        optimization = records("optimization.csv")
        coercivity = records("newton_spd.csv")
        if not (newton or optimization or coercivity):
            continue
        fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.0), constrained_layout=True)
        flat = axes.flat

        def series(source, key):
            return np.asarray([
                value for record in source if (value := _number(record.get(key))) is not None
            ], dtype=float)

        residual = series(newton, "residual")
        ksp_iterations = series(newton, "ksp_iterations")
        ksp_time = series(newton, "ksp_time")
        damping = series(newton, "damping")
        for ax, values, ylabel, logarithmic in (
            (flat[0], residual, "nonlinear residual", True),
            (flat[1], ksp_iterations, "KSP iterations", False),
            (flat[2], ksp_time, "KSP time [s]", False),
            (flat[3], damping, "damping", False),
        ):
            if len(values):
                ax.plot(np.arange(len(values)), values, marker=".", linewidth=0.8)
                if logarithmic and np.all(values > 0.0):
                    ax.set_yscale("log")
            ax.set(xlabel="logged Newton record", ylabel=ylabel)
        margin = series(coercivity, "muMin")
        if len(margin):
            flat[4].plot(np.arange(len(margin)), margin, marker=".")
            flat[4].axhline(0.0, color="black", linewidth=0.7)
        flat[4].set(xlabel="coercivity check", ylabel=r"minimum $\mu$")

        trust = series(optimization, "trustRadius")
        ratio = series(optimization, "rhoRatio")
        overlap = series(optimization, "branchOverlap")
        trial_cost = series(optimization, "stepTime")
        if len(trust):
            flat[5].plot(np.arange(len(trust)), trust, marker=".", label="trust radius")
        if len(ratio):
            flat[5].plot(np.arange(len(ratio)), ratio, marker=".", label="acceptance ratio")
        if flat[5].lines:
            flat[5].legend(fontsize=7)
        flat[5].set(xlabel="outer trial", ylabel="trust/acceptance")
        if len(overlap):
            flat[6].plot(np.arange(len(overlap)), overlap, marker=".")
        flat[6].set(xlabel="outer trial", ylabel="branch overlap")
        if len(trial_cost):
            rejected = np.asarray([
                str(record.get("accepted", "")).strip().lower() in {"0", "false"}
                for record in optimization
                if _number(record.get("stepTime")) is not None
            ])
            flat[7].bar(
                np.arange(len(trial_cost)), trial_cost,
                color=np.where(rejected, "#de2d26", "#41ab5d"),
            )
        flat[7].set(xlabel="outer trial", ylabel="trial cost [s]")
        fig.suptitle(f"{row['geometry']} capped-trial and coercivity diagnostics")
        path = output / f"solver_diagnostics_{row['id']}.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        written.append(path)

    # Publication storyboards are selected and rendered once by the figure
    # contract below. The former all-archive pass duplicated large contour
    # figures, rendered unclassified failure directories, and could overwrite
    # distinct runs sharing the same geometry/band filename.
    written.extend(create_contract_figures(bundle, rows, stage=stage))
    return written


def _trajectory_triangulation(data):
    """Mask a DOF triangulation to the archived physical mesh."""
    import matplotlib.tri as mtri

    coordinates = np.asarray(data["dof_coordinates"], dtype=float)
    triangulation = mtri.Triangulation(coordinates[:, 0], coordinates[:, 1])
    mesh_points = np.asarray(data["mesh_points"], dtype=float)
    mesh_cells = np.asarray(data["mesh_cells"], dtype=np.int64)
    physical = mtri.Triangulation(mesh_points[:, 0], mesh_points[:, 1], mesh_cells)
    centroids = coordinates[triangulation.triangles].mean(axis=1)
    outside = physical.get_trifinder()(centroids[:, 0], centroids[:, 1]) < 0
    triangulation.set_mask(outside)
    return triangulation


def _add_trajectory_overlays(ax, triangulation, torsion, phi, metadata, state) -> None:
    """Draw fixed torsion and moving equilibrium band boundaries."""
    fixed_levels = (float(metadata["c1_t"]), float(metadata["c2_t"]))
    moving_levels = (float(state["c1"]), float(state["c2"]))
    for values, levels, color, style in (
        (torsion, fixed_levels, "#e68613", "--"),
        (phi, moving_levels, "#1769aa", "-"),
    ):
        finite = np.asarray(values, dtype=float)
        valid_levels = [level for level in levels if np.nanmin(finite) < level < np.nanmax(finite)]
        if valid_levels:
            ax.tricontour(
                triangulation,
                finite,
                levels=valid_levels,
                colors=color,
                linestyles=style,
                linewidths=0.7,
            )


def _trajectory_panel_fields(data, index: int) -> list[tuple[np.ndarray, str]]:
    phi = np.asarray(data["states_phi"][index], dtype=float)
    rho = np.asarray(data["states_rho"][index], dtype=float)
    return [
        (np.asarray(data["fixed_torsion"], dtype=float), r"$T$"),
        (np.asarray(data["fixed_target_density"], dtype=float), r"$\rho_{\rm target}$"),
        (np.asarray(data["fixed_target_potential"], dtype=float), r"$\phi_{\rm target}$"),
        (rho, r"$\rho$"),
        (phi, r"$\phi$"),
        (phi - np.asarray(data["fixed_target_potential"], dtype=float), r"$\phi-\phi_{\rm target}$"),
    ]


def _draw_trajectory_row(fig, axes, data, triangulation, metadata, index: int) -> None:
    state = metadata["states"][index]
    phi = np.asarray(data["states_phi"][index], dtype=float)
    torsion = np.asarray(data["fixed_torsion"], dtype=float)
    for ax, (values, title) in zip(axes, _trajectory_panel_fields(data, index), strict=True):
        image = ax.tricontourf(triangulation, values, levels=40, cmap="viridis")
        _add_trajectory_overlays(ax, triangulation, torsion, phi, metadata, state)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9)
        fig.colorbar(image, ax=ax, shrink=0.72, pad=0.01)
    axes[0].set_ylabel(
        f"{state['stage']}\n"
        + rf"$\lambda={state['homotopy_lambda']:.3g}$"
        + "\n"
        + rf"$c=({state['c1']:.3g},{state['c2']:.3g})$",
        fontsize=8,
    )


def trajectory_storyboard_indices(states: Sequence[dict[str, Any]]) -> list[int]:
    """Select seed, homotopy, early/middle accepted, restored, and final states."""
    if not states:
        return []
    selected: list[int] = []

    def add(index: int | None) -> None:
        if index is not None and index not in selected:
            selected.append(index)

    seed = next((i for i, state in enumerate(states) if state["stage"] == "selected_seed"), None)
    homotopy = [i for i, state in enumerate(states) if state["stage"] == "homotopy"]
    accepted = [i for i, state in enumerate(states) if state["stage"] == "accepted_outer"]
    restored = next((i for i, state in enumerate(states) if state["stage"] == "restored_best"), None)
    final = next((i for i in range(len(states) - 1, -1, -1) if states[i]["stage"] == "final"), len(states) - 1)
    add(seed)
    add(homotopy[len(homotopy) // 2] if homotopy else None)
    add(accepted[0] if accepted else None)
    add(accepted[len(accepted) // 2] if len(accepted) > 1 else None)
    add(restored)
    add(final)
    return selected


def create_trajectory_storyboard(archive: Path, output: Path) -> Path:
    """Render one target-design strip followed only by evolving states."""
    try:
        from projects.diocotron.studies.torsion_optimizer.figures.storyboards import (
            _render_storyboard as render_storyboard,
        )
    except ImportError:
        from projects.diocotron.studies.torsion_optimizer.figures.storyboards import (  # type: ignore
            _render_storyboard as render_storyboard,
        )

    archive = Path(archive)
    output = Path(output)
    if output.suffix.lower() != ".png":
        raise ValueError(f"trajectory storyboard output must be PNG: {output}")
    with np.load(archive, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
    row = dict(metadata)
    row.setdefault("state", "completed")
    render_storyboard(row, archive, output, include_design=True)
    return output


def trajectory_frames(archive: Path, frame_dir: Path) -> list[Path]:
    """Render lossless six-panel movie frames without rerunning the optimizer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    with np.load(archive, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        triangulation = _trajectory_triangulation(data)
        for index, state in enumerate(metadata["states"]):
            fig, axes = plt.subplots(1, 6, figsize=(18.0, 3.2), constrained_layout=True)
            _draw_trajectory_row(fig, axes, data, triangulation, metadata, index)
            fig.suptitle(
                f"{state['stage']} — orange dashed: torsion band; blue solid: equilibrium band"
            )
            path = frame_dir / f"frame_{index:05d}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            frames.append(path)
    return frames


def create_movie(archive: Path, output: Path, fps: int) -> tuple[Path | None, list[Path]]:
    frames = trajectory_frames(archive, output.parent / f"{output.stem}_frames")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return None, frames
    command = [ffmpeg, "-y", "-framerate", str(fps), "-i", str(frames[0].parent / "frame_%05d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)]
    completed = subprocess.run(command, check=False)
    return (output if completed.returncode == 0 else None), frames


def validate_study(manifest: dict[str, Any], bundle: Path, *, require_terminal: bool) -> list[str]:
    errors: list[str] = []
    ids = [case["id"] for case in manifest["cases"]]
    if len(ids) != len(set(ids)):
        errors.append("duplicate case IDs")
    for case in manifest["cases"]:
        if require_terminal and case["state"] not in TERMINAL_CASE_STATES:
            errors.append(f"{case['id']}: state is {case['state']!r}, not terminal")
        if case["state"] == "completed":
            result = case.get("result", {})
            if result.get("exit_code") != 0:
                errors.append(f"{case['id']}: completed case has nonzero exit status")
            if not completed_case_outputs_valid(case):
                errors.append(f"{case['id']}: completed case has missing or modified outputs")
            for relative, expected in result.get("output_hashes", {}).items():
                path = Path(result["run_dir"]).parent / relative
                if not path.is_file() or sha256_file(path) != expected:
                    errors.append(f"{case['id']}: output hash mismatch for {relative}")
    aggregate = bundle / "data" / "aggregated.json"
    if aggregate.is_file():
        rows = json.loads(aggregate.read_text(encoding="utf-8"))
        for row in rows:
            c1 = _number(row.get("bestC1Phi"))
            c2 = _number(row.get("bestC2Phi"))
            eps = _number(row.get("bestEpsPhi"))
            if c1 is not None and c2 is not None and eps is not None and not epsilon_invariant(c1, c2, eps):
                errors.append(f"{row['id']}: epsilon/width invariant violated")
            if row.get("state") == "completed" and row.get("classification") in SUCCESS_CLASSIFICATIONS:
                residual = _number(row.get("bestResidual"))
                if residual is None or residual > 1.0e-12:
                    errors.append(f"{row['id']}: successful case missed final residual tolerance")
    registry_path = bundle / "generated" / "figure_registry.json"
    registry_final = registry_path.is_file() and (
        json.loads(registry_path.read_text(encoding="utf-8")).get("stage") == "final"
    )
    errors.extend(validate_figure_contract(bundle, final=registry_final))
    return errors


def latex_escape(value: str) -> str:
    replacements = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#"}
    return "".join(replacements.get(character, character) for character in value)

def _finite_number(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and math.isfinite(number) else None


def _first_finite(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        number = _finite_number(row.get(key))
        if number is not None:
            return number
    return None


def _tex_observed_number(value: float, digits: int = 3) -> str:
    formatted = f"{float(value):.{digits}g}"
    if "e" not in formatted.lower():
        return formatted
    mantissa, exponent = formatted.lower().split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def _homotopy_measurement(row: dict[str, Any]) -> dict[str, float] | None:
    values = {
        "lambda": _first_finite(
            row, "homotopyLambdaFinal", "homotopy_lambda_max_logged"
        ),
        "accepted": _first_finite(
            row, "homotopy_logged_accepted", "homotopyStages"
        ),
        "rejected": _first_finite(
            row, "homotopy_logged_rejected", "homotopyRejectedSteps"
        ),
        "newton": _first_finite(
            row, "homotopy_logged_newton_iterations", "homotopyNewtonIterations"
        ),
        "seconds": _first_finite(
            row, "homotopy_logged_seconds", "phase_homotopy_seconds"
        ),
    }
    return None if any(value is None for value in values.values()) else {
        key: float(value) for key, value in values.items() if value is not None
    }


def _uses_all_mumps(row: dict[str, Any]) -> bool:
    solver_fields = (
        "stiffnessLinearSolver", "homotopyLinearSolver",
        "nonlinearLinearSolver", "sensitivityLinearSolver", "finalLinearSolver",
    )
    solvers = [
        str(row[field]).strip().lower()
        for field in solver_fields
        if row.get(field) not in (None, "")
    ]
    return len(solvers) >= 3 and all(solver == "mumps" for solver in solvers)


def _same_homotopy_problem(
        left: dict[str, Any],
        right: dict[str, Any],
) -> bool:
    for key in ("geometry", "order", "dof_target"):
        if str(left.get(key)) != str(right.get(key)):
            return False
    for key in ("alpha_t1", "alpha_t2"):
        left_value = _finite_number(left.get(key))
        right_value = _finite_number(right.get(key))
        if (
            left_value is None
            or right_value is None
            or not math.isclose(left_value, right_value, abs_tol=1.0e-12)
        ):
            return False
    return True


def _completed_inexact_records(
        rows: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[Path] = set()
    for row in rows:
        if row.get("state") != "completed" or not row.get("run_dir"):
            continue
        path = Path(str(row["run_dir"])) / "logs" / "inexact_newton.csv"
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                records.extend(csv.DictReader(handle))
        except OSError:
            continue
    return records


def _inexact_metrics(
        records: Sequence[dict[str, str]],
        policy: str,
        tolerance: float,
) -> dict[str, float] | None:
    sample = [
        record for record in records
        if str(record.get("policy", "")) == policy
        and (requested := _finite_number(record.get("requested_tolerance"))) is not None
        and math.isclose(requested, tolerance, rel_tol=1.0e-12, abs_tol=1.0e-15)
    ]
    if not sample:
        return None

    def maximum(*fields: str) -> float | None:
        values = [
            number
            for record in sample
            for field in fields
            if (number := _finite_number(record.get(field))) is not None
        ]
        return max(values) if values else None

    agreement_values = [
        str(record.get("acceptance_decision_agrees", "")).strip().lower()
        for record in sample
    ]
    if not all(value in {"0", "1", "false", "true", "no", "yes"} for value in agreement_values):
        return None
    metrics = {
        "state": maximum("state_l2_relative_error", "state_h1_relative_error"),
        "sensitivity": maximum(
            "sensitivity1_l2_relative_error", "sensitivity1_h1_relative_error",
            "sensitivity2_l2_relative_error", "sensitivity2_h1_relative_error",
        ),
        "gradient": maximum("reduced_gradient_relative_error"),
        "step": maximum("threshold_step_relative_error"),
    }
    if any(value is None for value in metrics.values()):
        return None
    return {
        **{key: float(value) for key, value in metrics.items() if value is not None},
        "agreement": sum(
            value in {"1", "true", "yes"} for value in agreement_values
        ) / len(agreement_values),
        "count": float(len(sample)),
    }


def build_observed_findings_tex(rows: Sequence[dict[str, Any]]) -> str:
    """Render only observations backed by terminal aggregate/CSV evidence."""
    findings: list[str] = []

    iterative_candidates = []
    for row in rows:
        measurement = _homotopy_measurement(row)
        if (
            row.get("state") == "failed"
            and row.get("difficulty") == "difficult"
            and str(row.get("init_fallback", row.get("initFallback", ""))) == "window-fit"
            and measurement is not None
            and measurement["lambda"] < 1.0 - 1.0e-10
            and not _uses_all_mumps(row)
        ):
            iterative_candidates.append((row, measurement))
    iterative_candidates.sort(
        key=lambda item: (-item[1]["accepted"], str(item[0].get("id", "")))
    )
    if iterative_candidates:
        iterative_row, iterative = iterative_candidates[0]
        mumps_candidates = []
        for row in rows:
            measurement = _homotopy_measurement(row)
            if (
                row.get("state") == "completed"
                and measurement is not None
                and measurement["lambda"] >= 1.0 - 1.0e-10
                and _uses_all_mumps(row)
                and _same_homotopy_problem(iterative_row, row)
            ):
                mumps_candidates.append((row, measurement))
        mumps_candidates.sort(key=lambda item: (
            0 if item[0].get("kind") == "mumps_parent" else 1,
            str(item[0].get("id", "")),
        ))
        if mumps_candidates:
            mumps_row, mumps = mumps_candidates[0]
            geometry = latex_escape(
                str(iterative_row.get("geometry", "domain")).replace("_", " ")
            )
            alpha_t1 = _finite_number(iterative_row.get("alpha_t1"))
            alpha_t2 = _finite_number(iterative_row.get("alpha_t2"))
            band = (
                rf" at band \(({_tex_observed_number(alpha_t1)},"
                rf"{_tex_observed_number(alpha_t2)})\)"
                if alpha_t1 is not None and alpha_t2 is not None else ""
            )
            iterative_search = latex_escape(str(
                iterative_row.get("init_search", iterative_row.get("initSearch", "recorded"))
            ))
            mumps_search = latex_escape(str(
                mumps_row.get("init_search", mumps_row.get("initSearch", "recorded"))
            ))
            findings.append(
                "\\paragraph{Homotopy robustness.} "
                f"For the difficult low-DOF {geometry} case{band}, the measured "
                f"optimized-iterative {iterative_search}-search path with window-fit "
                rf"fallback stopped near \(\lambda={_tex_observed_number(iterative['lambda'])}\) "
                f"after {round(iterative['accepted'])} accepted and "
                f"{round(iterative['rejected'])} rejected continuation steps, "
                f"{round(iterative['newton'])} Newton iterations, and "
                rf"\({_tex_observed_number(iterative['seconds'])}\,\mathrm{{s}}\) of "
                f"logged homotopy work.  The matched all-MUMPS {mumps_search}-search "
                rf"path reached \(\lambda={_tex_observed_number(mumps['lambda'])}\) in "
                f"{round(mumps['accepted'])} accepted and {round(mumps['rejected'])} "
                f"rejected steps, {round(mumps['newton'])} Newton iterations, and "
                rf"\({_tex_observed_number(mumps['seconds'])}\,\mathrm{{s}}\)."
            )

    inexact_records = _completed_inexact_records(rows)
    requested_metrics = [
        ("raw", 1.0e-5, r"raw \(10^{-5}\)"),
        ("raw", 1.0e-3, r"raw \(10^{-3}\)"),
        ("one-correction", 1.0e-3, r"one-correction \(10^{-3}\)"),
    ]
    inexact_sentences = []
    agreements = []
    for policy, tolerance, label in requested_metrics:
        metrics = _inexact_metrics(inexact_records, policy, tolerance)
        if metrics is None:
            continue
        inexact_sentences.append(
            f"For {label}, the maxima over {round(metrics['count'])} completed "
            "snapshot-policy records were "
            rf"\({_tex_observed_number(metrics['state'])}\) for the state, "
            rf"\({_tex_observed_number(metrics['sensitivity'])}\) for the "
            "sensitivities, "
            rf"\({_tex_observed_number(metrics['gradient'])}\) for the reduced "
            "gradient, and "
            rf"\({_tex_observed_number(metrics['step'])}\) for the threshold step."
        )
        agreements.append(
            f"{label}: {_tex_observed_number(100.0 * metrics['agreement'])}\\%"
        )
    if inexact_sentences:
        findings.append(
            "\\paragraph{Inexact intermediate Newton solves.} "
            + "  ".join(inexact_sentences)
            + "  The acceptance-agreement rates were "
            + ", ".join(agreements)
            + ".  The reported agreement refers only to the replay's "
            "\\emph{model-feasibility proxy}; it is not agreement with a fully "
            "corrected nonlinear trial and is not a proof of optimizer acceptance."
        )

    fixed_rows: dict[int, dict[str, Any]] = {}
    for order in (2, 4, 6):
        candidates = sorted(
            (
                row for row in rows
                if row.get("state") == "completed"
                and row.get("kind") == "fixed_mesh_p"
                and int(float(row.get("order", -1))) == order
                and _first_finite(row, "bestC1Phi") is not None
                and _first_finite(row, "bestC2Phi") is not None
                and _first_finite(
                    row, "phase_total_seconds", "timeTotal", "total_seconds", "wall_seconds"
                ) is not None
            ),
            key=lambda row: str(row.get("id", "")),
        )
        if candidates:
            fixed_rows[order] = candidates[0]
    mesh_files = {
        str(row.get("meshFile", "")).strip() for row in fixed_rows.values()
    }
    if set(fixed_rows) == {2, 4, 6} and len(mesh_files) == 1 and "" not in mesh_files:
        run_descriptions = []
        for order in (2, 4, 6):
            row = fixed_rows[order]
            c1 = _first_finite(row, "bestC1Phi")
            c2 = _first_finite(row, "bestC2Phi")
            seconds = _first_finite(
                row, "phase_total_seconds", "timeTotal", "total_seconds", "wall_seconds"
            )
            ndof = _first_finite(row, "ndof", "mesh_dofs")
            dof_text = f", {round(ndof)} DOFs" if ndof is not None else ""
            run_descriptions.append(
                f"P{order}{dof_text}: "
                rf"\((c_{{1,\phi}},c_{{2,\phi}})="
                rf"({_tex_observed_number(c1)},{_tex_observed_number(c2)})\), "
                rf"\({_tex_observed_number(seconds)}\,\mathrm{{s}}\)"
            )
        p4_p6_difference = max(
            abs(
                float(_first_finite(fixed_rows[4], key))
                - float(_first_finite(fixed_rows[6], key))
            )
            for key in ("bestC1Phi", "bestC2Phi")
        )
        paragraph = (
            "\\paragraph{Fixed-triangulation polynomial enrichment.} "
            "On the identical linear triangulation, " + "; ".join(run_descriptions)
            + ".  The P4 and P6 thresholds agree to within "
            + rf"\({_tex_observed_number(p4_p6_difference)}\)."
        )
        external_errors = [
            (
                _first_finite(fixed_rows[order], "relative_l2"),
                _first_finite(fixed_rows[order], "relative_h1"),
            )
            for order in (2, 4, 6)
        ]
        if all(l2 is not None and h1 is not None for l2, h1 in external_errors):
            error_text = "; ".join(
                f"P{order}: "
                rf"\((L^2,H^1)=({_tex_observed_number(float(l2))},"
                rf"{_tex_observed_number(float(h1))})\)"
                for order, (l2, h1) in zip((2, 4, 6), external_errors, strict=True)
            )
            nonmonotone = any(
                float(external_errors[index + 1][component])
                > float(external_errors[index][component])
                for index in range(2)
                for component in range(2)
            )
            paragraph += "  The external-reference errors are " + error_text + "."
            if nonmonotone:
                paragraph += (
                    "  Their non-monotonicity mixes discretization error with "
                    "independent nonlinear-branch selection and stopping; it does "
                    "not show that higher \\(p\\) is intrinsically worse."
                )
        findings.append(paragraph)

    storyboard_rows = []
    for geometry in ("pacman", "horseshoe", "iter"):
        candidates = sorted(
            (
                row for row in rows
                if row.get("state") == "completed"
                and row.get("kind") == "trajectory_v3"
                and row.get("algorithm_variant") == "strict_storyboard_fast_fallback"
                and row.get("geometry") == geometry
                and row.get("classification") not in SUCCESS_CLASSIFICATIONS
                and (residual := _finite_number(row.get("bestResidual"))) is not None
                and (tolerance := _finite_number(row.get("finalNewtonTolRes"))) is not None
                and residual <= tolerance
            ),
            key=lambda row: str(row.get("id", "")),
        )
        if candidates:
            storyboard_rows.append(candidates[0])
    if storyboard_rows:
        details = []
        for row in storyboard_rows:
            geometry = {
                "pacman": "Pacman", "horseshoe": "horseshoe", "iter": "ITER",
            }.get(str(row.get("geometry")), str(row.get("geometry")))
            detail = (
                f"{geometry}: residual "
                rf"\({_tex_observed_number(float(_finite_number(row['bestResidual'])))}\), "
                f"terminal status {latex_escape(str(row.get('finalStatus', 'recorded')))}"
            )
            leakage = _first_finite(row, "bestLeakageRel")
            missing = _first_finite(row, "bestMissingRel")
            if leakage is not None and missing is not None:
                detail += (
                    rf", relative leakage/missing "
                    rf"\(({_tex_observed_number(leakage)},"
                    rf"{_tex_observed_number(missing)})\)"
                )
            details.append(r"\item " + detail + ".")
        findings.append(
            "\\paragraph{Non-star storyboard outcomes.} "
            "The completed non-star storyboard runs reached their recorded strict "
            "final PDE tolerances:\n"
            "\\begin{itemize}\\setlength{\\itemsep}{0pt}"
            "\\setlength{\\parskip}{0pt}\n"
            + "\n".join(details)
            + "\n\\end{itemize}\n"
            "These terminal classifications are geometric iteration-limit/plateau "
            "or capped-trial failures rather than strict band matches.  PDE "
            "residual convergence must therefore be reported separately from "
            "geometric success."
        )

    lines = [r"\subsection{Observed behavior}"]
    if findings:
        lines.extend(findings)
    else:
        lines.append(
            "No terminal aggregate or completed replay evidence currently supports "
            "an observed-behavior comparison; no numerical inference is made."
        )
    return "\n\n".join(lines) + "\n"


def write_numerical_tests_section(bundle: Path, manifest: dict[str, Any]) -> None:
    rows = _load_rows(bundle) if (bundle / "data" / "aggregated.json").is_file() else []
    generated = bundle / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    for row in rows:
        counts[row.get("classification", "unclassified")] += 1
    macros = [
        "% generated; do not edit",
        rf"\newcommand{{\NumericalTestsPlannedCases}}{{{len(manifest['cases'])}}}",
        rf"\newcommand{{\NumericalTestsCompletedCases}}{{{sum(case['state'] == 'completed' for case in manifest['cases'])}}}",
        rf"\newcommand{{\NumericalTestsStrictCases}}{{{counts['strict_convergence']}}}",
        rf"\newcommand{{\NumericalTestsCertifiedCases}}{{{counts['certified_subband_convergence']}}}",
        rf"\newcommand{{\NumericalTestsEpsilonRatio}}{{{EPSILON_RATIO:.2f}}}",
        rf"\newcommand{{\NumericalTestsCertifiedWidthFraction}}{{{1 - 2 * KAPPA * EPSILON_RATIO:.2f}}}",
    ]
    (generated / "macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    statuses = sorted(counts.items())
    table_lines = [r"\begin{tabular}{lr}", r"\toprule", r"Classification & Cases\\", r"\midrule"]
    table_lines.extend(rf"{latex_escape(name)} & {count}\\" for name, count in statuses)
    table_lines.extend((r"\bottomrule", r"\end{tabular}"))
    (generated / "status_table.tex").write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    def tex_number(value: Any, digits: int = 3) -> str:
        number = _number(value)
        return "--" if number is None or not math.isfinite(number) else f"{number:.{digits}g}"

    references = [
        row for row in rows
        if row.get("is_numerical_reference")
        and row.get("classification") in SUCCESS_CLASSIFICATIONS
    ]
    discretization = [
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"Geometry & $p$ & DOFs & $c_{1,\phi}$ & $c_{2,\phi}$ & $L^2$ error & $H^1$ error & Jaccard\\",
        r"\midrule",
    ]
    if references:
        for row in sorted(references, key=lambda item: (str(item["geometry"]), float(item["alpha_t1"]))):
            discretization.append(
                f"{latex_escape(str(row['geometry']))} & {int(row['order'])} & "
                f"{int(float(row.get('ndof', row['dof_target'])))} & "
                f"{tex_number(row.get('bestC1Phi'))} & {tex_number(row.get('bestC2Phi'))} & "
                f"{tex_number(row.get('relative_l2'))} & {tex_number(row.get('relative_h1'))} & "
                f"{tex_number(row.get('active_jaccard'))}\\\\"
            )
    else:
        discretization.append(r"\multicolumn{8}{c}{Reference solutions pending.}\\")
    discretization.extend((r"\bottomrule", r"\end{tabular}"))
    (generated / "discretization_table.tex").write_text(
        "\n".join(discretization) + "\n", encoding="utf-8"
    )

    stationarity_rows = [row for row in rows if row.get("classification") == "stationarity_completed"]
    stationarity = [
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Geometry & $\Delta t$ & $T$ & density $L^2$ drift & mass drift & handoff $H^1$ & Poisson $H^1$\\",
        r"\midrule",
    ]
    if stationarity_rows:
        for row in sorted(stationarity_rows, key=lambda item: (str(item["geometry"]), float(item["dt"]))):
            stationarity.append(
                f"{latex_escape(str(row['geometry']))} & {tex_number(row.get('dt'))} & "
                f"{tex_number(row.get('final_time'))} & {tex_number(row.get('stationarity_rho_change_l2_rel'))} & "
                f"{tex_number(row.get('stationarity_mass_rel_drift'))} & "
                f"{tex_number(row.get('handoff_relative_h1'))} & "
                f"{tex_number(row.get('poisson_consistency_relative_h1'))}\\\\"
            )
    else:
        stationarity.append(r"\multicolumn{7}{c}{Stationarity runs pending.}\\")
    stationarity.extend((r"\bottomrule", r"\end{tabular}"))
    (generated / "stationarity_table.tex").write_text(
        "\n".join(stationarity) + "\n", encoding="utf-8"
    )
    (generated / "observed_findings.tex").write_text(
        build_observed_findings_tex(rows), encoding="utf-8"
    )

    registry_path = generated / "figure_registry.json"
    if registry_path.is_file():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        registry_items = list(registry.get("figures", []))
        paired = {
            (str(item.get("geometry")), int(item.get("pair_order", -1))): item
            for item in registry_items
            if item.get("group") == "storyboard_pair"
        }
        paired_ids = {str(item.get("id")) for item in paired.values()}
        figure_items = []
        for item in registry_items:
            identifier = str(item.get("id"))
            if identifier in paired_ids and identifier != "band_evolution":
                continue
            if identifier == "band_evolution" and paired:
                for geometry in ("smooth_star", "pacman", "horseshoe", "iter"):
                    for pair_order in (0, 1):
                        pair = paired.get((geometry, pair_order))
                        if pair is not None:
                            figure_items.append(pair)
                continue
            figure_items.append(item)
    else:
        figure_items = [
            {**item, "status": "registry pending"} for item in FIGURE_CONTRACT
        ]
    early_lambda_filename = "iter_0p6_0p7_early_threshold_lambda_0p025_storyboard.png"
    if (
        (bundle / "figures" / early_lambda_filename).is_file()
        and all(str(item.get("file")) != early_lambda_filename for item in figure_items)
    ):
        figure_items.append({
            "id": "iter_early_threshold_lambda_0025",
            "file": early_lambda_filename,
            "status": "actual; partial-homotopy PDE-converged, geometrically unsuccessful",
            "caption": (
                "ITER [0.6,0.7] early threshold release at fixed source-homotopy lambda=0.025: "
                "two continuation Newton iterations reached 1.67e-14, while 30 reduced steps "
                "ended at residual 8.86e-14, relative leakage 2.002e-2, relative missing area "
                "9.757e-1, and zero certified area."
            ),
        })
    figure_blocks = []
    brute_force_items = [
        {
            "id": "iter_bruteforce_grid_search",
            "file": "iter_0p6_0p7_bruteforce_grid_search.png",
            "width": "0.97",
            "status": "actual; PDE-feasible initialization, geometrically unsuccessful",
            "caption": (
                "MPI-split brute-force threshold search for sharp-target ITER [0.6,0.7] "
                "with epsPhi/(c2Phi-c1Phi)=0.04. Five four-rank MUMPS groups evaluated "
                "95 threshold pairs at Newton tolerance 1e-4: 3 of 46 broad-grid pairs "
                "and all 49 continuation-refined pairs converged. The selected pair was "
                "(0.1700,0.29167), with residual 2.42e-7, relative leakage 0.1978, "
                "and relative activity area 0.1978."
            ),
        },
        {
            "id": "iter_bruteforce_sharp_storyboard",
            "file": "iter_0p6_0p7_bruteforce_sharp_eps004_storyboard.png",
            "width": "0.97",
            "status": "actual; final PDE-converged, threshold-space step stagnation",
            "caption": (
                "Separated design and final fields for the sharp-target ITER brute-force "
                "initialization. The upper row contains the target design once; the lower "
                "row contains the final reduced-optimization state. Rank zero gathered all "
                "8888 cells and 71661 P4 degrees of freedom, and element edges are hidden. "
                "The final exact residual is 2.07e-15, but relative leakage and missing area "
                "are 0.0752 and 0.9981, with zero certified area."
            ),
        },
        {
            "id": "iter_torsion_cap_grid_search",
            "file": "iter_0p6_0p7_torsion_cap_fine_grid_search.png",
            "width": "0.97",
            "landscape": True,
            "status": "actual; 555 PDE-feasible pairs, boundary-limited initializer",
            "caption": (
                "Expanded physical-cap search for sharp-target ITER [0.6,0.7]. "
                "The broad triangle and two 15-by-15 refinements evaluated 555 "
                "threshold pairs and 1660 independently seeded Newton branches "
                "at tolerance 1e-6. The selected pair (0.3827000,1.7642896) "
                "reached residual 9.82e-11, but lies on both upper boundaries "
                "of the final refinement."
            ),
        },
        {
            "id": "iter_torsion_cap_grid_level_0",
            "file": "iter_0p6_0p7_torsion_cap_fine_grid_level_0.png",
            "width": "0.97",
            "landscape": True,
            "status": "actual; full physical search triangle",
            "caption": (
                "Broad-grid objective, Newton residual and work, leakage, missing "
                "area, and activity-area mismatch over 0 <= c1Phi < c2Phi <= max(T). "
                "All 105 pairs possess at least one converged, bound-respecting branch."
            ),
        },
        {
            "id": "iter_torsion_cap_grid_level_2",
            "file": "iter_0p6_0p7_torsion_cap_fine_grid_level_2.png",
            "width": "0.97",
            "landscape": True,
            "status": "actual; fine-grid boundary trend unresolved",
            "caption": (
                "Fine refinement diagnostics. The objective decreases toward the "
                "upper c1Phi and c2Phi corner, so this level supplies a robust PDE "
                "initializer but does not certify an interior threshold minimizer."
            ),
        },
        {
            "id": "iter_torsion_cap_branch_residuals",
            "file": "iter_0p6_0p7_torsion_cap_fine_branch_residuals.png",
            "width": "0.97",
            "landscape": True,
            "status": "actual; all 1660 branch trials retained",
            "caption": (
                "Terminal Newton residual by seed family. Target-potential, torsion, "
                "and within-group continuation seeds are solved independently; red "
                "rings identify the branch selected for each threshold pair."
            ),
        },
        {
            "id": "iter_torsion_cap_branch_timing",
            "file": "iter_0p6_0p7_torsion_cap_fine_branch_timing.png",
            "width": "0.97",
            "landscape": True,
            "status": "actual; five concurrent four-rank MUMPS groups",
            "caption": (
                "Per-seed Newton wall time across the expanded grid. Continuation "
                "has the smallest median cost (2.69 s), while isolated alternate "
                "branches produce the long tail retained in the robustness record."
            ),
        },
        {
            "id": "iter_torsion_cap_maximum_principle",
            "file": "iter_0p6_0p7_torsion_cap_fine_maximum_principle.png",
            "width": "0.88",
            "status": "actual; discrete physical-bound audit",
            "caption": (
                "Lower and upper nodal violations against terminal residual for every "
                "seed attempt. Candidate selection requires the effective bound check "
                "0 <= phi <= T, with tolerance max(1e-8,10 times the grid residual tolerance)."
            ),
        },
        {
            "id": "iter_torsion_cap_storyboard",
            "file": "iter_0p6_0p7_torsion_cap_fine_storyboard.png",
            "width": "0.70",
            "landscape": True,
            "status": "actual; PDE-converged, geometrically unsuccessful plateau",
            "caption": (
                "ITER design fields shown once above selected early, middle, late, "
                "and final threshold states. Rank zero gathered all 8888 cells and "
                "71661 P4 degrees of freedom; mesh edges are suppressed. The final "
                "residual is 9.57e-15, but the certified area is zero."
            ),
        },
        {
            "id": "iter_torsion_cap_optimization_trace",
            "file": "iter_0p6_0p7_torsion_cap_fine_optimization_trace.png",
            "width": "0.86",
            "status": "actual; trust-limited threshold plateau",
            "caption": (
                "Reduced-optimization trace separating threshold motion, inexact PDE "
                "residual, projected gradient, geometric discrepancies, nonlinear "
                "work, acceptance ratio, and time. Four steps are accepted before "
                "fourteen rejections drive the trust radius to the step floor."
            ),
        },
        {
            "id": "iter_torsion_cap_contact_1",
            "file": "iter_0p6_0p7_torsion_cap_fine_frames_contact_01.png",
            "width": "0.97",
            "landscape": True,
            "status": "actual; native frames 0 through 2",
            "caption": (
                "First contact sheet from the complete high-resolution frame archive: "
                "design, the grid-selected state, and the first accepted threshold "
                "updates. The source PNGs remain available at 2800 by 1200 pixels."
            ),
        },
        {
            "id": "iter_torsion_cap_contact_2",
            "file": "iter_0p6_0p7_torsion_cap_fine_frames_contact_02.png",
            "width": "0.97",
            "landscape": True,
            "status": "actual; rejected-step plateau sequence",
            "caption": (
                "Second contact sheet from the ITER frame archive, showing the onset "
                "of repeated rejected threshold steps while the PDE state remains "
                "converged."
            ),
        },
        {
            "id": "iter_torsion_cap_contact_3",
            "file": "iter_0p6_0p7_torsion_cap_fine_frames_contact_03.png",
            "width": "0.97",
            "landscape": True,
            "status": "actual; late plateau and final projection",
            "caption": (
                "Late rejected-step states and the exact final MUMPS projection. "
                "These failure frames are retained rather than omitted from the "
                "qualitative convergence record."
            ),
        },
    ]
    for item in brute_force_items:
        if (
            (bundle / "figures" / str(item["file"])).is_file()
            and all(str(existing.get("file")) != str(item["file"]) for existing in figure_items)
        ):
            figure_items.append(item)
    rendered_figure_count = 0
    for item in figure_items:
        filename = str(item["file"])
        caption = latex_escape(str(item["caption"]))
        status = str(item.get("status", "unknown"))
        identifier = str(item.get("id", Path(filename).stem)).replace("_", "-")
        if (bundle / "figures" / filename).is_file():
            is_storyboard_pair = (
                item.get("group") == "storyboard_pair"
                or bool(item.get("landscape", False))
            )
            width = str(item.get("width", "0.97" if is_storyboard_pair else "0.88"))
            block = (
                "\\begin{figure}[tbp]\n"
                "\\centering\n"
                f"\\includegraphics[width={width}\\linewidth]{{figures/{filename}}}\n"
                f"\\caption{{{caption} Data status: {latex_escape(str(status))}.}}\n"
                f"\\label{{fig:{identifier}}}\n"
                "\\end{figure}"
            )
            if is_storyboard_pair:
                block = "\\begin{landscape}\n" + block + "\n\\end{landscape}"
            figure_blocks.append(block)
            rendered_figure_count += 1
            if rendered_figure_count % 8 == 0:
                figure_blocks.append(r"\clearpage")
    generated_figures = "\n\n".join(figure_blocks) or (
        r"\paragraph{Figures.} Publication figures will be inserted after completed cases are aggregated."
    )
    section = r"""\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{pdflscape}
\usepackage[margin=1in]{geometry}

% Resolve generated inputs when compiled either from the repository root or
% from this file's directory.
\makeatletter
\def\input@path{{./}{projects/diocotron/studies/torsion_optimizer/report/}}
\makeatother
\graphicspath{{./}{projects/diocotron/studies/torsion_optimizer/report/}}
\setlength{\emergencystretch}{1em}

\begin{document}
\section{Numerical tests}\label{sec:numerical-tests}
\input{generated/macros.tex}

We test robustness, mesh and polynomial-order sensitivity, computational
performance, initialization and continuation diagnostics, band evolution, and
short-time stationarity of the optimized equilibria.  The four domains are a
smooth sinusoidal star, a pacman, a horseshoe, and the ITER wall.  Every
optimization initializes its potential thresholds automatically by the
$H^{-1}$ search followed by source homotopy; no optimized threshold is supplied
manually.

\subsection{Protocol and reproducibility}
The production configuration uses the optimized solver preset, a $20\times20$
coarse $H^{-1}$ scan followed by two $11\times11$ refinements, homotopy steps
$0.1$, $2.5\times10^{-4}$, and $0.35$, a final residual tolerance of
$10^{-12}$, and a trial Krylov cap of 300.  The quadrature degree is
$\max(2p+8,12)$.  The relative equilibrium smoothing is fixed:
\[
  \varepsilon_\phi=\NumericalTestsEpsilonRatio
  (c_{2,\phi}-c_{1,\phi}).
\]
For $\kappa=2$, the certified potential-width fraction is therefore
$\NumericalTestsCertifiedWidthFraction$.  MPI cases run one at a time with one
numerical-library thread per rank.  The rank count follows the measured
$(h,p)$ workstation map and its global-DOF fallback; after strong scaling, the
smallest rank count within ten percent of the fastest median is retained.
Rank zero renders only after collectively gathering all owned cells and the
global nodal values from every partition, with explicit completeness checks.
Each case record contains the command, environment, package versions, Git
revision and dirty diff, inputs, exit status, terminal output, and output
hashes.

\subsection{Robustness and discretization studies}
The robustness matrix contains 25 bands per geometry: eight centers, three
widths, and the difficult $(0.45,0.50)$ anchor.  Failed and geometrically
unsuccessful cases remain in every status plot.  Mesh studies use P2, P4, and
P6 at matched global-DOF targets.  A numerical reference is accepted only when
the finest converged level also exhibits a decreasing final two-grid change;
otherwise the result is labelled unresolved and another refinement is
scheduled.

The selected numerical references and their comparison errors are reported in
Table~\ref{tab:numerical-reference}.  Relative field errors are evaluated at
quadrature points of the reference mesh.  Active and certified symmetric
differences use the same area weights, while contour Hausdorff distances are
normalized by the domain diameter.
\begin{table}[tb]
  \centering
  \resizebox{\linewidth}{!}{\input{generated/discretization_table.tex}}
  \caption{Selected numerical references.  A dash denotes an unavailable
  quantity, and pending rows are retained until their MPI cases terminate.}
  \label{tab:numerical-reference}
\end{table}

\subsection{Solver diagnostics and dynamics}
Phase and Newton/KSP logs separate initialization, homotopy, reduced
sensitivities and gradients, rejected trials, final projection, and
checkpointing.  Strong-scaling results report medians and interquartile ranges
over three repetitions.  Trajectory archives contain the mesh and fixed fields
once and record the selected seed, homotopy stages, accepted outer states,
restored-best state, and final state.  Short guiding-center tests use transient
SUPG with scale $0.1$, no CIP/flux stabilization, and both requested time
steps.

The continuation study varies the production and conservative step schedules
with and without the tangent predictor, and reports terminal $\lambda$,
accepted and rejected stages, Newton work, and time.  Diagnostic and rescue
cases remain separate from baseline robustness.  To quantify how accurately
non-optimal threshold iterates must be solved, we compare the adaptive forcing
rule with fixed intermediate Newton tolerances $10^{-3}$, $10^{-5}$, and
$10^{-7}$ while retaining the same strict final projection.
Threshold-space plateaus are identified from five consecutive outer records
with less than $0.1\%$ normalized threshold motion and discrepancy change.
The projected-gradient reduction then distinguishes a stationary plateau from
trust- or branch-limited stagnation.

\paragraph{Sharp-target brute-force diagnostic.}
For ITER $[0.6,0.7]$, the target density was made discontinuous
($\varepsilon_T=0$) and the equilibrium smoothing ratio was reduced to
$\varepsilon_\phi/\Delta c_\phi=0.04$.  A 20-rank search split the communicator
into five independent four-rank MUMPS groups.  The broad and refined grids
evaluated 95 pairs with a Newton tolerance of $10^{-4}$; 3 of 46 broad-grid
pairs and all 49 refined pairs converged.  The selected pair
$(0.1700,0.29167)$ had residual $2.42\times10^{-7}$, relative leakage $0.1978$,
and relative activity area $0.1978$.  Its distributed state seeded the normal
four-rank optimizer, which accepted eight threshold steps before the trust
region reached the step floor.  The final MUMPS projection reached
$2.07\times10^{-15}$, but relative leakage and missing area were $0.0752$ and
$0.9981$, respectively, and the certified area was zero.  Thus the brute-force
search removed PDE infeasibility as the immediate initializer failure, but it
selected and retained a collapsed off-target branch rather than recovering the
desired ITER band.

\paragraph{Expanded physical-cap and multi-seed diagnostic.}
We then repeated the sharp-target experiment over the full maximum-principle
triangle $0\leq c_{1,\phi}<c_{2,\phi}\leq\max T=1.934224$, using two
$15\times15$ refinements, Newton tolerance $10^{-6}$, and three branch
families.  Five concurrent four-rank MUMPS groups evaluated 555 distinct
threshold pairs and retained all 1660 target-potential, torsion, and
within-group continuation attempts.  Every pair possessed at least one
converged branch satisfying the discrete $0\leq\phi\leq T$ audit.  Considered
separately, 404 of 555 target-potential seeds, 513 of 555 torsion seeds, and
489 of 550 continuation seeds were both converged and bound-respecting; the
selected branches were supplied by these families in 151, 86, and 318 cases,
respectively.  Thus multi-seeding removed candidate-level PDE failure even
though no individual seed was uniformly robust.  The grid objective decreased
from 1.4722 to 1.2982 and 1.2746 over the three levels.  Median continuation
work was two Newton iterations and 2.69 seconds per attempt.

The selected pair $(0.3827000,1.7642896)$ reached residual
$9.82\times10^{-11}$, but it lies at the upper--upper corner of the last
refinement.  More importantly, its equilibrium has
$\max\phi=0.48139<c_{2,\phi}$: the upper threshold is inactive.  Although its
activity area is $0.7100$ times the target area, its relative leakage and
missing area are $0.6947$ and $0.9846$.  This is a direct counterexample to
using leakage plus unsigned total-area balance as a sufficient overlap
criterion: off-target activity compensates for missing target activity in the
scalar area term.  The normal reduced optimizer accepted four steps and then
rejected fourteen; its trust radius reached $1.934\times10^{-5}$ while the
projected gradient remained 9.917.  The exact final projection converged to
$9.57\times10^{-15}$, but relative leakage and missing area were 0.4344 and
0.9569 and the certified area remained zero.  We therefore classify this as
PDE-converged, geometrically unsuccessful, and trust/branch limited rather
than stationary.  The experiment motivates an overlap-aware missing-area
term and an active-threshold constraint in any subsequent brute-force
scalarization; the present unsuccessful run and every intermediate PNG remain
part of the reported robustness evidence.

\begin{table}[tb]
  \centering
  \input{generated/stationarity_table.tex}
  \caption{Short guiding-center stationarity and exact checkpoint-handoff
  errors.}
  \label{tab:stationarity}
\end{table}

\input{generated/observed_findings.tex}

\subsection{Numerical results}
%%GENERATED_FIGURES%%

\subsection{Completion status}
\input{generated/status_table.tex}

\subsection{Limitations and unresolved cases}
The finest levels and extrapolated MUMPS rank choices must be measured before
claims of mesh independence or optimal scaling are made.  Cases lacking a
terminal state, a requested final PDE residual, or a decreasing two-grid
change are reported explicitly and are never silently omitted.
\end{document}
"""
    section = section.replace("%%GENERATED_FIGURES%%", generated_figures)
    body_start = section.index(r"\section{Numerical tests}")
    body_end = section.rindex(r"\end{document}")
    body = section[body_start:body_end].rstrip() + "\n"
    (bundle / "numerical_tests_section.tex").write_text(body, encoding="utf-8")
    successful_cases = bundle / "successful_numerical_tests.tex"
    if not successful_cases.exists():
        successful_cases.write_text(
            "% Hand-audited successful case studies belong in this input fragment.\n",
            encoding="utf-8",
        )
    wrapper = (
        section[:body_start]
        + r"\input{numerical_tests_section.tex}"
        + "\n"
        + r"\input{successful_numerical_tests.tex}"
        + "\n"
        + r"\end{document}"
        + "\n"
    )
    (bundle / "numerical_tests.tex").write_text(wrapper, encoding="utf-8")
    commands = f"""# Reproduction commands

Figure generation emits only high-resolution PNG publication assets. The
portable bundle and current generators contain, regenerate, and register PNG
figures only; SVG assets are not part of the current study output contract.

```bash
conda run -n fenicsx-dgfem python projects/diocotron/studies/torsion_optimizer/run.py plan
conda run -n fenicsx-dgfem python projects/diocotron/studies/torsion_optimizer/run.py run --campaign draft
conda run -n fenicsx-dgfem python projects/diocotron/studies/torsion_optimizer/run.py aggregate
conda run -n fenicsx-dgfem python projects/diocotron/studies/torsion_optimizer/run.py figures --stage preliminary
conda run -n fenicsx-dgfem python projects/diocotron/studies/torsion_optimizer/run.py validate --require-terminal
conda run -n fenicsx-dgfem python projects/diocotron/studies/torsion_optimizer/run.py report
python -m projects.diocotron.studies.torsion_optimizer.build_report
```
"""
    (bundle / "REPRODUCE.md").write_text(commands, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_BUNDLE / "manifest.json")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    run = subparsers.add_parser("run")
    run.add_argument("--kind", action="append")
    run.add_argument("--case-id", action="append")
    run.add_argument("--geometry", action="append", choices=("smooth_star", "pacman", "horseshoe", "iter"))
    run.add_argument("--campaign", choices=("draft", "full", "v3_pilot", "v3_full"))
    run.add_argument("--limit", type=int)
    run.add_argument("--retry", action="store_true", help="retry failed or incomplete cases")
    run.add_argument("--rerun-completed", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    subparsers.add_parser("aggregate")
    figures = subparsers.add_parser("figures")
    figures.add_argument("--stage", choices=("preliminary", "final"), default="preliminary")
    movie = subparsers.add_parser("movie")
    movie.add_argument("archive", type=Path)
    movie.add_argument("--output", type=Path, default=None)
    movie.add_argument("--fps", type=int, default=6)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--require-terminal", action="store_true")
    subparsers.add_parser("report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.manifest = args.manifest.expanduser().resolve()
    args.bundle = args.bundle.expanduser().resolve()
    args.run_root = args.run_root.expanduser().resolve()
    if args.command == "plan":
        if args.manifest.exists():
            before = len(json.loads(args.manifest.read_text(encoding="utf-8"))["cases"])
            existing = load_manifest(args.manifest)
            write_json(args.manifest, existing)
            appended = len(existing["cases"]) - before
            print(
                f"refreshed existing manifest with {len(existing['cases'])} cases "
                f"({appended} appended): {args.manifest}"
            )
        else:
            manifest = new_manifest()
            write_json(args.manifest, manifest)
            print(f"planned {len(manifest['cases'])} cases: {args.manifest}")
        return 0
    if not args.manifest.is_file():
        raise FileNotFoundError(f"manifest not found; run plan first: {args.manifest}")
    manifest = load_manifest(args.manifest)
    if args.command == "run":
        return run_cases(args)
    if args.command == "aggregate":
        rows = aggregate_manifest(manifest)
        comparisons = attach_reference_comparisons(rows, args.bundle)
        appended = append_adaptive_cases(manifest, rows)
        if appended:
            write_json(args.manifest, manifest)
            rows = aggregate_manifest(manifest)
            comparisons += attach_reference_comparisons(rows, args.bundle)
        write_aggregate(rows, args.bundle)
        print(
            f"aggregated {len(rows)} cases, computed {comparisons} reference comparisons, "
            f"and scheduled {appended} follow-ups under {args.bundle / 'data'}"
        )
        return 0
    if args.command == "figures":
        backfilled, existing = backfill_attempt_storyboards(manifest, args.run_root)
        if backfilled or existing:
            write_json(args.manifest, manifest)
        figures = create_figures(args.bundle, stage=args.stage)
        print(
            f"wrote {len(figures)} PNG figure assets; "
            f"backfilled {backfilled} attempt storyboards "
            f"({existing} valid existing)"
        )
        return 0
    if args.command == "movie":
        output = args.output or args.bundle / "movies" / f"{args.archive.stem}.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        movie, frames = create_movie(args.archive, output, args.fps)
        print(f"frames retained: {frames[0].parent if frames else 'none'}")
        print(f"movie: {movie if movie else 'ffmpeg unavailable or encoder failed'}")
        return 0
    if args.command == "validate":
        errors = validate_study(manifest, args.bundle, require_terminal=args.require_terminal)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print("study validation passed")
        return 0
    if args.command == "report":
        write_numerical_tests_section(args.bundle, manifest)
        print(f"wrote standalone numerical-tests document: {args.bundle / 'numerical_tests.tex'}")
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
