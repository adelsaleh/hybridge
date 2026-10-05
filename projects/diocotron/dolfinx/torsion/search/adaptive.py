#!/usr/bin/env python3
"""Adaptive MPI threshold search followed by reduced optimization.

This is the successor to the Cartesian brute-force driver.  It searches the
complete maximum-principle triangle

    0 <= c1 < c2 <= max(T),

fits the two equilibrium level curves to the two torsion level curves, and
uses a conservative row/column thickness heuristic after a candidate passes
strict witness gates.  Candidate groups persist for the whole search and draw
work from an MPI-RMA queue.  Every *solved* candidate state is stored; PNG
rendering is deliberately deferred until all nonlinear work has finished, and
is then distributed across candidate subcommunicators.  Each plot is still
assembled collectively from a complete global field.

The module keeps the output and reduced-optimizer handoff conventions of
``dolfinx_torsion_bruteforce_window_reduced_optimization.py`` while adding
``geometry.csv`` diagnostics and ``pruned.csv`` provenance.
"""

from __future__ import annotations

# Support direct execution as well as python -m from a checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[5]))

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
for _import_path in (REPO_ROOT, SCRIPT_DIR):
    if str(_import_path) not in sys.path:
        sys.path.insert(0, str(_import_path))

import projects.diocotron.dolfinx.torsion.search.brute_force as brute
import projects.diocotron.dolfinx.torsion.optimization.reduced as reduced
from projects.diocotron.dolfinx.geometry.canonical import lagrange_dofs_from_metadata
from projects.diocotron.dolfinx.runtime.mpi_rank_policy import select_mpi_ranks
from projects.diocotron.dolfinx.torsion.search.metrics import DistributedBandMetricWorkspace
import projects.diocotron.dolfinx.torsion.search.helpers as adaptive_helpers


@dataclass(frozen=True)
class AdaptiveCandidate:
    """One dimensioned point in the normalized threshold triangle."""

    generation: int
    candidate: int
    uid: str
    c1_hat: float
    c2_hat: float
    c1: float
    c2: float
    parent_uid: str = ""

    def brute_candidate(self) -> brute.GridCandidate:
        return brute.GridCandidate(
            stage=self.generation,
            candidate=self.candidate,
            c1=self.c1,
            c2=self.c2,
        )


ADAPTIVE_BASE_FIELDS = (
    *brute.GRID_FIELDS,
    "uid",
    "parentUid",
    "parentStatePath",
    "c1Hat",
    "c2Hat",
    "statePath",
    "continuationTier",
    "pdeSelectionEligible",
    "explorationEligible",
    "handoffGeometryEligible",
    "handoffGeometryRejectionReason",
    "strictSelectionEligible",
    "geometryEligible",
    "strictGeometryEligible",
    "geometryRejectionReason",
    "strictGeometryRejectionReason",
    "tooThick",
    "dominanceWitness",
    "monotoneForBisection",
    "pairFitScore",
    "pairContainmentScore",
    "normalizedCurveError",
    "minimumHandoffContainment",
    "minimumHandoffJaccard",
    "maximumHandoffNormalizedCurveError",
    "robustTauSpan",
)

PRUNED_FIELDS = (
    "generation",
    "candidate",
    "uid",
    "c1",
    "c2",
    "c1Hat",
    "c2Hat",
    "reason",
    "dominatingUid",
)


def _finite(value: Any, default: float = math.inf) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "nan"}
    return bool(value)


def _lookup(mapping: dict[str, Any], names: Iterable[str], default: Any) -> Any:
    normalized = {
        str(key).replace("_", "").lower(): value for key, value in mapping.items()
    }
    for name in names:
        key = str(name).replace("_", "").lower()
        if key in normalized:
            return normalized[key]
    return default


def automatic_initial_node_count(group_count: int) -> int:
    """Supply at least two queue waves, without geometry-specific knowledge."""
    return adaptive_helpers.auto_initial_node_count(
        int(group_count), target_waves=2.0, minimum_nodes=7, maximum_nodes=13
    )


def candidate_uid(c1_hat: float, c2_hat: float) -> str:
    return adaptive_helpers.deterministic_candidate_id(
        "torsion-adaptive-bisection-v1", float(c1_hat), float(c2_hat)
    )


def _pair_key(c1_hat: float, c2_hat: float) -> tuple[float, float]:
    return round(float(c1_hat), 14), round(float(c2_hat), 14)


def initial_pairs(nodes: Sequence[float], min_width_hat: float) -> list[tuple[float, float]]:
    pairs = {
        _pair_key(c1, c2)
        for c1 in nodes
        for c2 in nodes
        if float(c2) - float(c1) >= float(min_width_hat)
    }
    return sorted(pairs, key=lambda pair: (pair[1] - pair[0], pair[0], pair[1]))


def make_candidates(
    pairs: Iterable[tuple[float, float]],
    *,
    generation: int,
    first_id: int,
    tmax: float,
    parent_by_pair: dict[tuple[float, float], str] | None = None,
) -> list[AdaptiveCandidate]:
    result: list[AdaptiveCandidate] = []
    parents = parent_by_pair or {}
    for offset, pair in enumerate(sorted({_pair_key(*pair) for pair in pairs})):
        c1_hat, c2_hat = pair
        result.append(
            AdaptiveCandidate(
                generation=int(generation),
                candidate=int(first_id) + offset,
                uid=candidate_uid(c1_hat, c2_hat),
                c1_hat=c1_hat,
                c2_hat=c2_hat,
                c1=float(tmax) * c1_hat,
                c2=float(tmax) * c2_hat,
                parent_uid=str(parents.get(pair, "")),
            )
        )
    return result


def adaptive_result_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Select only a PDE- and geometry-eligible seed as the true winner."""

    eligible = _bool(row.get("selectionEligible")) and _bool(
        row.get("geometryEligible")
    )
    pde = _bool(row.get("converged")) and _bool(row.get("boundSatisfied", 1))
    fit = _finite(row.get("pairFitScore"), default=-math.inf)
    containment = _finite(row.get("pairContainmentScore"), default=-math.inf)
    discrepancy = _finite(row.get("leakageRel")) + _finite(row.get("missingRel"))
    residual = _finite(row.get("residual"))
    physically_active = _bool(row.get("bothThresholdsActive")) or all(
        _bool(row.get(key)) for key in ("lowerCurveActive", "upperCurveActive")
    )
    too_thick = _bool(row.get("tooThick"))
    if eligible:
        return (0, discrepancy, -containment, -fit, residual, int(row["candidate"]))
    if physically_active:
        return (
            1,
            int(too_thick),
            0 if pde else 1,
            -fit,
            -containment,
            discrepancy,
            residual,
            int(row["candidate"]),
        )
    if pde:
        return (2, -fit, -containment, discrepancy, residual, int(row["candidate"]))
    return (3, residual, discrepancy, int(row["candidate"]))


def _geometry_dict(
    workspace: DistributedBandMetricWorkspace,
    c1: float,
    c2: float,
    *,
    minimum_practical_containment: float,
    minimum_handoff_jaccard: float,
    maximum_handoff_normalized_curve_error: float,
) -> dict[str, Any]:
    result = workspace.evaluate(c1=float(c1), c2=float(c2))
    if hasattr(result, "as_dict"):
        flat = dict(result.as_dict())
    elif isinstance(result, dict):
        flat = dict(result)
    else:
        flat = dict(vars(result))
    strict_eligible = _bool(
        _lookup(
            flat,
            ("geometryEligible", "geometricallyEligible"),
            getattr(result, "geometrically_eligible", False),
        )
    )
    strict_reason = str(
        _lookup(
            flat,
            ("geometryRejectionReason", "rejectionReason"),
            getattr(result, "rejection_reason", ""),
        )
    )
    # Strict eligibility is the certification tier and is retained verbatim.
    # Exploration may continue from active/resolved non-final states. Reduced
    # handoff has its own explicit, looser but still quantitative safeguards.
    lower_active = _bool(_lookup(flat, ("lowerCurveActive",), False))
    upper_active = _bool(_lookup(flat, ("upperCurveActive",), False))
    lower_resolved = _bool(_lookup(flat, ("lowerCurveResolved",), False))
    upper_resolved = _bool(_lookup(flat, ("upperCurveResolved",), False))
    lower_containment = _finite(_lookup(flat, ("lowerContainment",), 0.0), 0.0)
    upper_containment = _finite(_lookup(flat, ("upperContainment",), 0.0), 0.0)
    too_thick = _bool(
        _lookup(flat, ("tooThick",), getattr(result, "too_thick", False))
    )
    hard_jaccard = _finite(_lookup(flat, ("hardJaccard",), 0.0), 0.0)
    lower_tolerance = _finite(
        _lookup(flat, ("lowerPositionTolerance",), math.nan)
    )
    upper_tolerance = _finite(
        _lookup(flat, ("upperPositionTolerance",), math.nan)
    )
    lower_residual = abs(_curve_residual(flat, True))
    upper_residual = abs(_curve_residual(flat, False))
    normalized_curve_error = max(
        lower_residual / lower_tolerance if lower_tolerance > 0.0 else math.inf,
        upper_residual / upper_tolerance if upper_tolerance > 0.0 else math.inf,
    )
    exploration_eligible = bool(
        lower_active and upper_active and lower_resolved and upper_resolved
    )
    handoff_reasons: list[str] = []
    if not lower_active or not upper_active:
        handoff_reasons.append("INACTIVE_CURVE")
    if not lower_resolved or not upper_resolved:
        handoff_reasons.append("UNRESOLVED_CURVE")
    if min(lower_containment, upper_containment) < float(minimum_practical_containment):
        handoff_reasons.append("HANDOFF_CONTAINMENT_INSUFFICIENT")
    if too_thick:
        handoff_reasons.append("THICK_BAND_REJECTED")
    if hard_jaccard < float(minimum_handoff_jaccard):
        handoff_reasons.append("HANDOFF_JACCARD_INSUFFICIENT")
    if normalized_curve_error > float(maximum_handoff_normalized_curve_error):
        handoff_reasons.append("HANDOFF_CURVE_ERROR_EXCESSIVE")
    flat["strictGeometryEligible"] = int(strict_eligible)
    flat["strictGeometryRejectionReason"] = strict_reason
    flat["explorationEligible"] = int(exploration_eligible)
    flat["handoffGeometryEligible"] = int(not handoff_reasons)
    flat["handoffGeometryRejectionReason"] = (
        ";".join(handoff_reasons) if handoff_reasons else "HANDOFF_GEOMETRY_ELIGIBLE"
    )
    # Compatibility aliases used by the existing winner and plotting code.
    flat["geometryEligible"] = flat["handoffGeometryEligible"]
    flat["geometryRejectionReason"] = flat["handoffGeometryRejectionReason"]
    flat["normalizedCurveError"] = normalized_curve_error
    flat["minimumHandoffContainment"] = float(minimum_practical_containment)
    flat["minimumHandoffJaccard"] = float(minimum_handoff_jaccard)
    flat["maximumHandoffNormalizedCurveError"] = float(
        maximum_handoff_normalized_curve_error
    )
    flat["tooThick"] = int(too_thick)
    flat["monotoneForBisection"] = int(
        _bool(
            _lookup(
                flat,
                ("monotoneForBisection",),
                getattr(result, "monotone_for_bisection", False),
            )
        )
    )
    flat["pairFitScore"] = _finite(
        _lookup(flat, ("pairFitScore",), getattr(result, "pair_fit_score", math.nan)),
        default=math.nan,
    )
    flat["pairContainmentScore"] = _finite(
        _lookup(
            flat,
            ("pairContainmentScore",),
            getattr(result, "pair_containment_score", math.nan),
        ),
        default=math.nan,
    )
    flat["robustTauSpan"] = _finite(
        _lookup(flat, ("robustTauSpan",), getattr(result, "robust_tau_span", math.nan)),
        default=math.nan,
    )
    return flat


def _curve_residual(row: dict[str, Any], lower: bool) -> float:
    names = (
        ("r1", "lowerPositionResidual", "lowerCurveResidual", "curveLowerResidual", "lowerMedianError")
        if lower
        else ("r2", "upperPositionResidual", "upperCurveResidual", "curveUpperResidual", "upperMedianError")
    )
    return _finite(_lookup(row, names, math.inf))


def _observation_from_row(row: dict[str, Any]) -> adaptive_helpers.CandidateObservation:
    pair = adaptive_helpers.make_threshold_pair(
        namespace="torsion-adaptive-bisection-v1",
        generation=int(row["stage"]),
        c1_hat=float(row["c1Hat"]),
        c2_hat=float(row["c2Hat"]),
        source="evaluated",
        parent_ids=(() if not row.get("parentUid") else (str(row["parentUid"]),)),
        preferred_group=int(row["group"]),
    )
    return adaptive_helpers.CandidateObservation(
        pair=pair,
        pde_converged=_bool(row.get("converged")),
        bounds_satisfied=_bool(row.get("boundSatisfied", 1)),
        contours_active=all(
            _bool(row.get(key))
            for key in (
                "lowerCurveActive",
                "upperCurveActive",
                "lowerCurveResolved",
                "upperCurveResolved",
            )
        ),
        too_thick=_bool(row.get("tooThick")),
        curve1_residual=_curve_residual(row, True),
        curve2_residual=_curve_residual(row, False),
        curve1_containment=_finite(row.get("lowerContainment"), 0.0),
        curve2_containment=_finite(row.get("upperContainment"), 0.0),
        curve1_spread=_finite(row.get("lowerTauSpread")),
        curve2_spread=_finite(row.get("upperTauSpread")),
        leakage_rel=_finite(row.get("leakageRel")),
        missing_rel=_finite(row.get("missingRel")),
        nonlinear_residual=_finite(row.get("residual")),
        claimed_group=int(row["group"]),
        thresholds_active=_bool(row.get("bothThresholdsActive")),
        nontrivial_field=bool(
            _bool(row.get("lowerThresholdActive"))
            or _bool(row.get("upperThresholdActive"))
            or _finite(row.get("activityAreaRel"), 0.0) > 1.0e-8
        ),
    )


def _continuation_key(
    row: dict[str, Any],
    criteria: adaptive_helpers.RankingCriteria,
    child_pair: adaptive_helpers.ThresholdPair | None = None,
) -> tuple[Any, ...]:
    return adaptive_helpers.continuation_rank(
        _observation_from_row(row),
        criteria,
        child_pair,
    )


def _usable_parent_attempt(row: dict[str, Any]) -> bool:
    observation = _observation_from_row(row)
    return bool(
        observation.physically_active
        and not observation.too_thick
        and str(row.get("newtonStatus", ""))
        in {"CONVERGED_RESIDUAL", "MAX_NEWTON"}
        and math.isfinite(_finite(row.get("residual")))
    )


def dominance_reason(
    c1: float,
    c2: float,
    thick_rows: Sequence[dict[str, Any]],
    *,
    tolerance: float,
) -> tuple[str, str] | None:
    """Apply the conservative fixed-row/fixed-column thickness heuristic."""
    for row in thick_rows:
        # Keep this defensive gate even though the scheduler only broadcasts
        # vetted rows.  An incomplete or stale gossip payload must never prune.
        if not adaptive_helpers.is_dominance_witness(row):
            continue
        tc1 = float(row["c1"])
        tc2 = float(row["c2"])
        if abs(float(c1) - tc1) <= tolerance and float(c2) > tc2 + tolerance:
            return "THICK_ROW_C2_DOMINANCE", str(row.get("uid", ""))
        if abs(float(c2) - tc2) <= tolerance and float(c1) < tc1 - tolerance:
            return "THICK_COLUMN_C1_DOMINANCE", str(row.get("uid", ""))
    return None


def adaptive_pairs(
    rows: Sequence[dict[str, Any]],
    *,
    seen: set[tuple[float, float]],
    min_width_hat: float,
    beam: int,
    criteria: adaptive_helpers.RankingCriteria,
    generation: int,
) -> tuple[list[tuple[float, float]], dict[tuple[float, float], str]]:
    """Bisect coordinate brackets around a parallel beam of promising roots."""

    observations = [_observation_from_row(row) for row in rows]
    shared = adaptive_helpers.propose_frontier_midpoints(
        observations,
        criteria=criteria,
        namespace="torsion-adaptive-bisection-v1",
        generation=int(generation),
        maximum_candidates=max(1, int(beam)),
        evaluated_ids=(str(row["uid"]) for row in rows),
    )
    shared_pairs = [
        _pair_key(pair.c1_hat, pair.c2_hat)
        for pair in shared
        if pair.width_hat >= float(min_width_hat)
        and _pair_key(pair.c1_hat, pair.c2_hat) not in seen
    ]
    if shared_pairs:
        shared_by_key = {
            _pair_key(pair.c1_hat, pair.c2_hat): pair for pair in shared
        }
        parents: dict[tuple[float, float], str] = {}
        for key in shared_pairs:
            proposal = shared_by_key[key]
            selected = adaptive_helpers.select_continuation_parent(
                proposal.parent_ids,
                observations,
                criteria=criteria,
                child_pair=proposal,
            )
            parents[key] = (
                selected.pair.candidate_id if selected is not None else ""
            )
        ordered_shared = list(dict.fromkeys(shared_pairs))[: max(0, int(beam))]
        return ordered_shared, {key: parents[key] for key in ordered_shared}

    # A coarse generation can have no basic-eligible point yet. Retain a
    # deterministic local-midpoint fallback so the search can approach the
    # active-curve region instead of terminating prematurely.
    usable = [
        row for row in rows if _observation_from_row(row).physically_active
    ]
    if not usable:
        usable = [row for row in rows if _bool(row.get("converged"))]
    if not usable:
        usable = list(rows)
    anchors = sorted(
        usable,
        key=lambda row: _continuation_key(row, criteria),
    )[: max(1, int(beam))]
    c1_nodes = sorted({0.0, 1.0, *(float(row["c1Hat"]) for row in rows)})
    c2_nodes = sorted({0.0, 1.0, *(float(row["c2Hat"]) for row in rows)})

    def local_midpoints(value: float, nodes: Sequence[float]) -> list[float]:
        index = min(range(len(nodes)), key=lambda k: abs(nodes[k] - value))
        values = {value}
        if index:
            values.add(0.5 * (nodes[index - 1] + value))
        if index + 1 < len(nodes):
            values.add(0.5 * (value + nodes[index + 1]))
        return sorted(values)

    pairs: set[tuple[float, float]] = set()
    parents: dict[tuple[float, float], str] = {}
    priorities: dict[tuple[float, float], tuple[Any, ...]] = {}
    for anchor_index, row in enumerate(anchors):
        c1_values = local_midpoints(float(row["c1Hat"]), c1_nodes)
        c2_values = local_midpoints(float(row["c2Hat"]), c2_nodes)
        for c1_hat in c1_values:
            for c2_hat in c2_values:
                pair = _pair_key(c1_hat, c2_hat)
                if pair in seen or pair[1] - pair[0] < float(min_width_hat):
                    continue
                pairs.add(pair)
                priority = (
                    anchor_index,
                    (pair[0] - float(row["c1Hat"])) ** 2
                    + (pair[1] - float(row["c2Hat"])) ** 2,
                    pair,
                )
                if pair not in priorities or priority < priorities[pair]:
                    priorities[pair] = priority
                    parents[pair] = str(row.get("uid", ""))
    ordered = sorted(pairs, key=lambda pair: priorities[pair])[: max(0, int(beam))]
    return ordered, {pair: parents[pair] for pair in ordered}


def _all_csv_fields(rows: Sequence[dict[str, Any]], preferred: Sequence[str]) -> list[str]:
    present = {str(key) for row in rows for key in row}
    result = [name for name in preferred if name in present]
    result.extend(sorted(present.difference(result)))
    return result


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], preferred: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = _all_csv_fields(rows, preferred)
    if not fields:
        fields = list(preferred)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_candidate_state(
    group: MPI.Comm,
    problem: dict[str, Any],
    candidate: AdaptiveCandidate,
    state: np.ndarray,
    output_dir: Path,
) -> str:
    values = brute.gather_global_state(group, problem["u"].function_space, state)
    relative = (
        Path("states")
        / f"generation_{candidate.generation:03d}"
        / f"candidate_{candidate.candidate:06d}_{candidate.uid}.npz"
    )
    if group.rank == 0:
        path = output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            phi=values,
            c1=np.asarray([candidate.c1], dtype=np.float64),
            c2=np.asarray([candidate.c2], dtype=np.float64),
            order=np.asarray([problem["args"].order], dtype=np.int32),
        )
    return str(relative)


def _evaluate_one(
    candidate: AdaptiveCandidate,
    *,
    group_id: int,
    problem: dict[str, Any],
    workspace: DistributedBandMetricWorkspace,
    seeds: Sequence[tuple[str, np.ndarray]],
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempts: list[tuple[dict[str, Any], np.ndarray]] = []
    criteria = problem["ranking_criteria"]
    candidate_comm = problem["domain"].comm
    verbose_candidates = int(problem["args"].verbosity) >= 2
    for seed_index, (label, seed) in enumerate(seeds):
        if verbose_candidates and candidate_comm.rank == 0:
            print(
                "ADAPTIVE_ATTEMPT_START "
                f"g={candidate.generation} id={candidate.candidate} group={group_id} "
                f"seedIndex={seed_index} seed={label} "
                f"c1={candidate.c1:.10e} c2={candidate.c2:.10e} "
                f"width={candidate.c2-candidate.c1:.10e}",
                flush=True,
            )
        base_row, attempt_state = brute.evaluate_candidate(
            candidate=candidate.brute_candidate(),
            group_id=group_id,
            seed=seed,
            seed_label=label,
            problem=problem,
        )
        problem["u"].x.array[:] = attempt_state
        problem["u"].x.scatter_forward()
        attempt = dict(base_row)
        attempt.update(
            _geometry_dict(
                workspace,
                candidate.c1,
                candidate.c2,
                minimum_practical_containment=float(
                    problem["minimum_practical_containment"]
                ),
                minimum_handoff_jaccard=float(problem["minimum_handoff_jaccard"]),
                maximum_handoff_normalized_curve_error=float(
                    problem["maximum_handoff_normalized_curve_error"]
                ),
            )
        )
        attempt.update(
            {
                "uid": candidate.uid,
                "parentUid": candidate.parent_uid,
                "c1Hat": candidate.c1_hat,
                "c2Hat": candidate.c2_hat,
                "pdeSelectionEligible": int(_bool(attempt.get("selectionEligible"))),
            }
        )
        attempt["selectionEligible"] = int(
            _bool(attempt["pdeSelectionEligible"])
            and _bool(attempt["handoffGeometryEligible"])
        )
        attempt["strictSelectionEligible"] = int(
            _bool(attempt["pdeSelectionEligible"])
            and _bool(attempt["strictGeometryEligible"])
        )
        attempt["continuationTier"] = int(
            _continuation_key(attempt, criteria)[0]
        )
        attempt["dominanceWitness"] = int(
            adaptive_helpers.is_dominance_witness(attempt)
        )
        if verbose_candidates and candidate_comm.rank == 0:
            print(
                "ADAPTIVE_ATTEMPT_DONE "
                f"g={candidate.generation} id={candidate.candidate} group={group_id} seed={label} "
                f"status={attempt['newtonStatus']} converged={int(_bool(attempt['converged']))} "
                f"iterations={int(attempt['newtonIterations'])} residual={float(attempt['residual']):.10e} "
                f"leakRel={float(attempt['leakageRel']):.10e} missingRel={float(attempt['missingRel']):.10e} "
                f"jaccard={_finite(attempt.get('hardJaccard'), math.nan):.10e} "
                f"finalBudget={attempt.get('newtonFinalBudget', '')} wall={float(attempt['wallTime']):.6f}s",
                flush=True,
            )
        attempts.append((attempt, attempt_state))
        if (
            seed_index == 0
            and str(label).startswith("parent:")
            and _usable_parent_attempt(attempt)
        ):
            break
    selected_index = min(
        range(len(attempts)),
        key=lambda index: (
            adaptive_result_key(attempts[index][0]),
            str(attempts[index][0].get("seed", "")),
        ),
    )
    row = dict(attempts[selected_index][0])
    state = attempts[selected_index][1]
    row["seedAttempts"] = len(attempts)
    row["wallTime"] = sum(float(attempt[0]["wallTime"]) for attempt in attempts)
    problem["u"].x.array[:] = state
    problem["u"].x.scatter_forward()
    row["statePath"] = _save_candidate_state(
        problem["domain"].comm, problem, candidate, state, output_dir
    )
    trials: list[dict[str, Any]] = []
    for index, (attempt_row, _) in enumerate(attempts):
        trial = dict(attempt_row)
        trial.update(
            {
                "uid": candidate.uid,
                "parentUid": candidate.parent_uid,
                "c1Hat": candidate.c1_hat,
                "c2Hat": candidate.c2_hat,
                "selected": int(index == selected_index),
            }
        )
        trials.append(trial)
    return row, trials


def _rma_claim(win: MPI.Win) -> int:
    one = np.asarray([1], dtype=np.int64)
    old = np.asarray([-1], dtype=np.int64)
    win.Lock(0)
    win.Fetch_and_op([one, MPI.INT64_T], [old, MPI.INT64_T], 0, 0, MPI.SUM)
    win.Flush(0)
    win.Unlock(0)
    return int(old[0])


_THICK_GOSSIP_TAG = 27184


def _drain_thick_gossip(leaders: MPI.Comm, known: list[dict[str, Any]]) -> None:
    """Receive completed thick witnesses without synchronizing worker groups."""

    known_uids = {str(row.get("uid", "")) for row in known}
    while leaders.Iprobe(source=MPI.ANY_SOURCE, tag=_THICK_GOSSIP_TAG):
        row = leaders.recv(source=MPI.ANY_SOURCE, tag=_THICK_GOSSIP_TAG)
        uid = str(row.get("uid", ""))
        if uid not in known_uids:
            known.append(row)
            known_uids.add(uid)


def _publish_thick_gossip(leaders: MPI.Comm, row: dict[str, Any]) -> None:
    """Make a new dominance witness visible to every still-working group."""

    if not adaptive_helpers.is_dominance_witness(row):
        raise ValueError("refusing to publish an unvetted thickness witness")
    fields = (
        "c1",
        "c2",
        "uid",
        "tooThick",
        "converged",
        "boundSatisfied",
        "lowerCurveActive",
        "upperCurveActive",
        "lowerCurveResolved",
        "upperCurveResolved",
        "monotoneForBisection",
        "bothThresholdsActive",
        "activityAreaRel",
        "targetUnderresolved",
    )
    payload = {field: row.get(field, "") for field in fields}
    for destination in range(leaders.size):
        if destination != leaders.rank:
            request = leaders.isend(payload, dest=destination, tag=_THICK_GOSSIP_TAG)
            request.Free()


def _render_all_candidates(
    *,
    world: MPI.Comm,
    group: MPI.Comm,
    group_id: int,
    group_count: int,
    problem: dict[str, Any],
    rows: list[dict[str, Any]] | None,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]] | None:
    """Render deterministic row partitions on complete candidate submeshes."""

    if not bool(args.save_candidate_pngs):
        return rows
    rows = world.bcast(rows if world.rank == 0 else None, root=0)
    render_group_count = (
        int(group_count) if int(args.render_groups) == 0 else int(args.render_groups)
    )
    if not 1 <= render_group_count <= int(group_count):
        raise ValueError(
            f"render-groups must lie in [1,{group_count}] or be zero for all groups"
        )
    ordered_rows = sorted(
        rows,
        key=lambda item: (int(item["stage"]), int(item["candidate"]), str(item["uid"])),
    )
    row_by_id = {int(row["candidate"]): row for row in ordered_rows}
    if len(row_by_id) != len(ordered_rows):
        raise RuntimeError("candidate IDs are not unique before deferred rendering")
    partitions = adaptive_helpers.partition_candidate_ids(
        tuple(row_by_id),
        render_group_count,
    )
    assigned = (
        [row_by_id[identifier] for identifier in partitions[group_id]]
        if group_id < render_group_count
        else []
    )
    local_mapping: dict[int, str] = {}
    if assigned:
        plotter = reduced.MPIPyVistaTorsionPlotter(
            problem["args"],
            run_tag=f"adaptive_candidates_group_{group_id:03d}",
            run_dir=output_dir,
            frame_writer=None,
            comm=group,
        )
        for row in assigned:
            reduced.load_global_initial_state(
                output_dir / str(row["statePath"]), problem["u"], group
            )
            c1 = float(row["c1"])
            c2 = float(row["c2"])
            eps_phi = float(row["epsPhi"])
            problem["c1_const"].value = PETSc.ScalarType(c1)
            problem["c2_const"].value = PETSc.ScalarType(c2)
            problem["eps_const"].value = PETSc.ScalarType(eps_phi)
            reduced.update_interpolated(
                problem["rho"],
                reduced.window_density_const_ufl(
                    problem["u"],
                    problem["c1_const"],
                    problem["c2_const"],
                    problem["eps_const"],
                    problem["params"].rho_amp,
                ),
            )
            difference = problem["candidate_difference"]
            difference.x.array[:] = problem["u"].x.array - problem["phi_target"].x.array
            difference.x.scatter_forward()
            status = "eligible" if _bool(row["selectionEligible"]) else "rejected"
            relative = (
                Path("candidate_pairs")
                / f"generation_{int(row['stage']):03d}"
                / (
                    f"candidate_{int(row['candidate']):06d}_{row['uid']}_"
                    f"{status}.png"
                )
            )
            summary = (
                f"generation={row['stage']} candidate={row['candidate']} uid={row['uid']}\n"
                f"c1={c1:.8e} c2={c2:.8e} width={c2-c1:.4e}\n"
                f"Newton={row['newtonStatus']} residual={float(row['residual']):.4e}\n"
                f"leak={float(row['leakageRel']):.4e} missing={float(row['missingRel']):.4e}\n"
                f"curve fit={_finite(row.get('pairFitScore'), math.nan):.4e} "
                f"containment={_finite(row.get('pairContainmentScore'), math.nan):.4e}\n"
                f"thick={row['tooThick']} geometry={row['geometryEligible']} "
                f"reason={row['geometryRejectionReason']}"
            )
            plotter.save_candidate_fields(
                [
                    problem["T"],
                    problem["tau_band"],
                    problem["rho_design"],
                    problem["phi_target"],
                    problem["u"],
                    problem["rho"],
                    difference,
                ],
                [
                    "torsion T",
                    "target band",
                    "target density",
                    "target potential",
                    "candidate potential",
                    "candidate density",
                    "potential mismatch",
                ],
                save_path=output_dir / relative,
                window_size=(int(args.candidate_png_width), int(args.candidate_png_height)),
                nt=problem["nt"],
                ndof=problem["ndof"],
                target_field_index=0,
                target_levels=(problem["c1_t"], problem["c2_t"]),
                equilibrium_field_index=4,
                equilibrium_levels=(c1, c2),
                summary=summary,
                stage=f"adaptive_g{int(row['stage'])}_c{int(row['candidate'])}",
            )
            if group.rank == 0:
                local_mapping[int(row["candidate"])] = str(relative)
    gathered = world.gather(local_mapping if group.rank == 0 else None, root=0)
    if world.rank != 0:
        return None

    merged: dict[int, str] = {}
    for mapping in gathered:
        if mapping is None:
            continue
        for identifier, relative in mapping.items():
            if identifier in merged:
                raise RuntimeError(
                    f"candidate {identifier} was rendered by more than one subgroup"
                )
            merged[int(identifier)] = str(relative)
    expected = set(row_by_id)
    received = set(merged)
    if received != expected:
        missing = sorted(expected - received)
        extra = sorted(received - expected)
        raise RuntimeError(
            "candidate PNG coverage mismatch: "
            f"missing={missing[:12]} extra={extra[:12]}"
        )
    for row in ordered_rows:
        relative = merged[int(row["candidate"])]
        path = output_dir / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(
                f"candidate {row['candidate']} PNG is missing or empty: {path}"
            )
        row["candidatePng"] = relative
    print(
        f"CANDIDATE_PNG_COVERAGE solved={len(ordered_rows)} rendered={len(merged)} "
        f"groups={render_group_count} complete=1",
        flush=True,
    )
    return ordered_rows


def _show_interactive_candidates(
    *,
    group: MPI.Comm,
    problem: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    plotter: reduced.MPIPyVistaTorsionPlotter,
) -> None:
    """Show every fully converged candidate on one MPI-complete live mesh."""

    ordered = sorted(
        rows,
        key=lambda row: (int(row["stage"]), int(row["candidate"])),
    )
    shown = 0
    skipped = 0
    for row in ordered:
        converged = candidate_live_plot_eligible(row)
        status = str(row.get("newtonStatus", ""))
        if not converged:
            skipped += 1
            if group.rank == 0:
                print(
                    "CANDIDATE_LIVE_SKIP "
                    f"g={row['stage']} id={row['candidate']} "
                    f"status={status} residual={float(row['residual']):.6e} "
                    "reason=NEWTON_NOT_FULLY_CONVERGED",
                    flush=True,
                )
            continue

        reduced.load_global_initial_state(
            output_dir / str(row["statePath"]),
            problem["u"],
            group,
        )
        c1 = float(row["c1"])
        c2 = float(row["c2"])
        eps_phi = float(row["epsPhi"])
        problem["c1_const"].value = PETSc.ScalarType(c1)
        problem["c2_const"].value = PETSc.ScalarType(c2)
        problem["eps_const"].value = PETSc.ScalarType(eps_phi)
        reduced.update_interpolated(
            problem["rho"],
            reduced.window_density_const_ufl(
                problem["u"],
                problem["c1_const"],
                problem["c2_const"],
                problem["eps_const"],
                problem["params"].rho_amp,
            ),
        )
        problem["candidate_difference"].x.array[:] = (
            problem["u"].x.array - problem["phi_target"].x.array
        )
        problem["candidate_difference"].x.scatter_forward()

        title = (
            f"candidate {int(row['candidate'])}, g={int(row['stage'])}; "
            f"c=({c1:.4e},{c2:.4e}); "
            f"R={float(row['residual']):.3e}; "
            f"J={_finite(row.get('hardJaccard'), math.nan):.3f}"
        )
        if problem["args"].plot_fields == "density":
            fields = [problem["rho"], problem["T"]]
            titles = [title]
            target_contour_index = 1
        else:
            fields = [
                problem["T"],
                problem["tau_band"],
                problem["rho_design"],
                problem["phi_target"],
                problem["u"],
                problem["rho"],
                problem["candidate_difference"],
            ]
            titles = [
                "torsion T",
                "target band",
                "target density",
                "target potential",
                "candidate potential",
                title,
                "potential mismatch",
            ]
            target_contour_index = 0
        plotter.emit(
            fields,
            titles,
            stage=f"ADAPTIVE_G{int(row['stage'])}",
            ieps=int(row["stage"]),
            k=int(row["candidate"]),
            eps_phi=eps_phi,
            residual=float(row["residual"]),
            metrics={},
            token=f"candidate_{int(row['candidate']):06d}",
            save=False,
            show=True,
            nt=int(problem["nt"]),
            ndof=int(problem["ndof"]),
            contour_field_index=target_contour_index,
            contour_levels=(float(problem["c1_t"]), float(problem["c2_t"])),
        )
        shown += 1
        if group.rank == 0:
            print(
                "CANDIDATE_LIVE_SHOW "
                f"g={row['stage']} id={row['candidate']} status={status} "
                f"c1={c1:.10e} c2={c2:.10e} eps={eps_phi:.10e} "
                f"residual={float(row['residual']):.10e}",
                flush=True,
            )

    if group.rank == 0:
        print(
            f"CANDIDATE_LIVE_GENERATION solved={len(ordered)} "
            f"shownConverged={shown} skippedUnresolved={skipped}",
            flush=True,
        )


def candidate_live_plot_eligible(row: dict[str, Any]) -> bool:
    """Return whether an adaptive candidate may open an interactive plot.

    Only a state that reached the requested candidate Newton tolerance is
    displayable.  In particular, line-search failures, stagnated states, and
    iteration-cap states remain terminal diagnostics and are never shown.
    """

    return _bool(row.get("converged"))


def _compat_reduced_args(args: argparse.Namespace) -> tuple[list[str], argparse.Namespace]:
    values = brute.forwarded_reduced_args(args)
    managed = {"--threshold-cap-mode", "--c-lower-fraction", "--c-upper-fraction"}
    cleaned: list[str] = []
    skip = False
    for token in values:
        if skip:
            skip = False
            continue
        key = token.split("=", 1)[0]
        if key in managed:
            skip = "=" not in token
            continue
        cleaned.append(token)
    cleaned.extend(
        [
            "--threshold-cap-mode",
            "torsion",
            "--c-lower-fraction",
            "0",
            "--c-upper-fraction",
            "1",
        ]
    )
    parsed = reduced.parse_args(cleaned)
    reduced.validate_args(parsed)
    parsed.grid_guess_c1 = 0.0
    parsed.grid_guess_c2 = 1.0
    parsed.grid_newton_tol = float(args.grid_newton_tol)
    parsed.grid_newton_max_it = int(args.grid_newton_max_it)
    parsed.grid_area_weight = float(args.grid_area_weight)
    parsed.grid_missing_weight = float(args.grid_missing_weight)
    parsed.grid_require_active_thresholds = bool(args.grid_require_active_thresholds)
    parsed.grid_cap_mode = "torsion"
    parsed.grid_seed = str(args.grid_seed)
    parsed.grid_bound_tol = float(args.grid_bound_tol)
    parsed.plot = bool(args.interactive_candidate_plots)
    parsed.save_frames = False
    parsed.plot_off_screen = not bool(args.interactive_candidate_plots)
    parsed.plot_mesh_edges = False
    return cleaned, parsed


def run_search(args: argparse.Namespace) -> int:
    world = MPI.COMM_WORLD
    if int(args.ranks_per_candidate) < 1 or world.size % int(args.ranks_per_candidate):
        raise ValueError("MPI world must be divisible by positive ranks-per-candidate")
    reduced_argv, reduced_args = _compat_reduced_args(args)
    group_id = world.rank // int(args.ranks_per_candidate)
    group_count = world.size // int(args.ranks_per_candidate)
    if int(args.render_groups) > group_count:
        raise ValueError(
            f"render-groups={args.render_groups} exceeds candidate group count {group_count}"
        )
    group = world.Split(group_id, world.rank)
    peer = world.Split(group.rank, group_id)
    leaders = world.Split(0 if group.rank == 0 else MPI.UNDEFINED, group_id)
    output_dir = args.output_dir.resolve()
    if world.rank == 0:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"search output already exists: {output_dir}")
        output_dir.mkdir(parents=True)
    world.barrier()
    group_dir = output_dir / "scratch" / f"group_{group_id:03d}"
    if group.rank == 0:
        group_dir.mkdir(parents=True, exist_ok=True)
    group.barrier()
    problem = brute.prepare_group_problem(
        args=reduced_args, group=group, group_dir=group_dir, group_id=group_id
    )
    nonlinear_args = problem["nonlinear_args"]
    nonlinear_args.verbosity = int(reduced_args.verbosity)
    nonlinear_args.newton_soft_cap = bool(args.grid_newton_soft_cap)
    nonlinear_args.newton_soft_cap_chunk = int(args.grid_newton_soft_cap_chunk)
    nonlinear_args.newton_soft_cap_factor = float(args.grid_newton_soft_cap_factor)
    nonlinear_args.newton_soft_cap_window = int(args.grid_newton_soft_cap_window)
    nonlinear_args.newton_soft_cap_contraction = float(
        args.grid_newton_soft_cap_contraction
    )
    problem["minimum_practical_containment"] = float(
        args.minimum_practical_containment
    )
    problem["minimum_handoff_jaccard"] = float(args.minimum_handoff_jaccard)
    problem["maximum_handoff_normalized_curve_error"] = float(
        args.maximum_handoff_normalized_curve_error
    )
    workspace = DistributedBandMetricWorkspace(
        u=problem["u"],
        torsion=problem["T"],
        tmax=problem["tmax"],
        alpha_t1=problem["params"].alpha_t1,
        alpha_t2=problem["params"].alpha_t2,
        quadrature_degree=problem["qdeg"],
        order=int(reduced_args.order),
        comm=group,
        histogram_bins=int(args.contour_histogram_bins),
        kernel_scale=float(args.contour_kernel_scale),
    )
    ranking_criteria = adaptive_helpers.RankingCriteria(
        curve1_tolerance=float(workspace.target.lower_tolerance),
        curve2_tolerance=float(workspace.target.upper_tolerance),
        minimum_containment=float(args.minimum_practical_containment),
        curve1_spread_limit=2.0 * float(workspace.target.lower_tolerance),
        curve2_spread_limit=2.0 * float(workspace.target.upper_tolerance),
    )
    problem["ranking_criteria"] = ranking_criteria
    candidate_live_plotter = (
        reduced.MPIPyVistaTorsionPlotter(
            reduced_args,
            run_tag="adaptive_candidates_live",
            run_dir=output_dir,
            frame_writer=None,
            comm=group,
        )
        if bool(args.interactive_candidate_plots) and group_id == 0
        else None
    )
    phi_target_seed = problem["phi_target"].x.array.copy()
    torsion_seed = problem["T"].x.array.copy()
    predicted_pair: tuple[float, float] | None = None
    predictor = getattr(workspace, "predict_threshold_pair", None)
    if callable(predictor):
        try:
            predicted_pair = tuple(map(float, predictor(problem["phi_target"])))
        except (RuntimeError, ValueError):
            predicted_pair = None

    counter_storage = (
        np.zeros(1, dtype=np.int64)
        if group.rank == 0 and leaders.rank == 0
        else np.empty(0, dtype=np.int64)
    )
    counter_win = (
        MPI.Win.Create(counter_storage, disp_unit=8, comm=leaders)
        if group.rank == 0
        else None
    )
    all_rows: list[dict[str, Any]] = []
    all_trials: list[dict[str, Any]] = []
    all_pruned: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    next_id = 0
    incumbent_state: np.ndarray | None = None
    previous_best = math.inf
    plateau_count = 0
    candidates: list[AdaptiveCandidate] = []
    generation_budgets: list[dict[str, Any]] = []

    for generation in range(int(args.adaptive_generations)):
        if world.rank == 0:
            if generation == 0:
                node_count = (
                    int(args.initial_nodes)
                    if int(args.initial_nodes) > 0
                    else automatic_initial_node_count(group_count)
                )
                nodes = list(adaptive_helpers.chebyshev_lobatto_nodes(node_count))
                if predicted_pair is not None:
                    predicted_hats = [
                        min(max(value / problem["tmax"], 0.0), 1.0)
                        for value in predicted_pair
                    ]
                    if predicted_hats[0] < predicted_hats[1]:
                        nodes.extend(predicted_hats)
                nodes = sorted(set(nodes))
                pairs = initial_pairs(nodes, problem["min_width"] / problem["tmax"])
                parents: dict[tuple[float, float], str] = {}
            else:
                desired = int(args.beam_candidates) or max(2 * group_count, 12)
                beam = adaptive_helpers.adaptive_generation_limit(
                    total_cap=max(0, int(args.max_candidates)),
                    completed=len(all_rows),
                    remaining_generations=int(args.adaptive_generations) - generation,
                    desired=desired,
                )
                pairs, parents = adaptive_pairs(
                    all_rows,
                    seen=seen,
                    min_width_hat=problem["min_width"] / problem["tmax"],
                    beam=beam,
                    criteria=ranking_criteria,
                    generation=generation,
                )
            candidates = make_candidates(
                pairs,
                generation=generation,
                first_id=next_id,
                tmax=problem["tmax"],
                parent_by_pair=parents,
            )
            if int(args.max_candidates) > 0:
                remaining = max(0, int(args.max_candidates) - len(all_rows))
                candidates = candidates[:remaining]
            generation_budgets.append(
                {
                    "generation": generation,
                    "candidateLimit": (
                        len(candidates) if generation == 0 else int(beam)
                    ),
                    "proposed": len(candidates),
                    "completedBefore": len(all_rows),
                }
            )
            for candidate in candidates:
                seen.add(_pair_key(candidate.c1_hat, candidate.c2_hat))
            next_id += len(candidates)
            parent_state_paths = {
                str(row["uid"]): str(row["statePath"])
                for row in all_rows
                if row.get("uid") and row.get("statePath")
            }
            prior_dominance_witnesses = [
                row
                for row in all_rows
                if adaptive_helpers.is_dominance_witness(row)
            ]
        candidates = world.bcast(candidates if world.rank == 0 else None, root=0)
        if not candidates:
            break
        parent_state_paths = world.bcast(
            parent_state_paths if world.rank == 0 else None,
            root=0,
        )
        prior_dominance_witnesses = world.bcast(
            prior_dominance_witnesses if world.rank == 0 else None,
            root=0,
        )

        # Narrow bands are claimed first so a discovered thick point can prune
        # still-unclaimed larger c2 (or smaller c1) values in the same row/column.
        queue = sorted(candidates, key=lambda c: (c.c2 - c.c1, c.c1, c.c2, c.candidate))
        if group.rank == 0:
            if leaders.rank == 0:
                counter_storage[0] = 0
                counter_win.Sync()
            leaders.Barrier()
        local_rows: list[dict[str, Any]] = []
        local_trials: list[dict[str, Any]] = []
        local_pruned: list[dict[str, Any]] = []
        # All leaders receive the previous generations' vetted witnesses.
        # In-generation discoveries are propagated asynchronously below.
        known_thick = list(prior_dominance_witnesses)
        while True:
            claimed = _rma_claim(counter_win) if group.rank == 0 else None
            claimed = group.bcast(claimed, root=0)
            if int(claimed) >= len(queue):
                break
            candidate = queue[int(claimed)]
            if group.rank == 0:
                _drain_thick_gossip(leaders, known_thick)
                dominance = dominance_reason(
                    candidate.c1,
                    candidate.c2,
                    known_thick,
                    tolerance=max(1.0e-13 * problem["tmax"], 1.0e-15),
                )
            else:
                dominance = None
            dominance = group.bcast(dominance, root=0)
            skip = dominance is not None
            if skip:
                if group.rank == 0:
                    local_pruned.append(
                        {
                            "generation": generation,
                            "candidate": candidate.candidate,
                            "uid": candidate.uid,
                            "c1": candidate.c1,
                            "c2": candidate.c2,
                            "c1Hat": candidate.c1_hat,
                            "c2Hat": candidate.c2_hat,
                            "reason": dominance[0],
                            "dominatingUid": dominance[1],
                        }
                    )
                continue
            seeds: list[tuple[str, np.ndarray]] = []
            parent_state_path = ""
            if candidate.parent_uid:
                parent_state_path = str(
                    parent_state_paths.get(candidate.parent_uid, "")
                )
                if not parent_state_path:
                    raise RuntimeError(
                        f"candidate {candidate.uid} is missing parent state "
                        f"{candidate.parent_uid}"
                    )
                reduced.load_global_initial_state(
                    output_dir / parent_state_path,
                    problem["u"],
                    group,
                )
                seeds.append(
                    (
                        f"parent:{candidate.parent_uid}",
                        problem["u"].x.array.copy(),
                    )
                )
            elif incumbent_state is not None:
                seeds.append(("generation_incumbent", incumbent_state))
            if args.grid_seed in {"phi-target", "both"}:
                seeds.append(("phi_target", phi_target_seed))
            if args.grid_seed in {"torsion", "both"}:
                seeds.append(("torsion", torsion_seed))
            row, trials = _evaluate_one(
                candidate,
                group_id=group_id,
                problem=problem,
                workspace=workspace,
                seeds=seeds,
                output_dir=output_dir,
            )
            row["parentStatePath"] = parent_state_path
            for trial in trials:
                trial["parentStatePath"] = parent_state_path
            if adaptive_helpers.is_dominance_witness(row):
                if group.rank == 0:
                    known_thick.append(row)
                    _publish_thick_gossip(leaders, row)
            if group.rank == 0:
                local_rows.append(row)
                local_trials.extend(trials)
                print(
                    "ADAPTIVE_POINT "
                    f"g={generation} id={candidate.candidate} group={group_id} "
                    f"c1={candidate.c1:.8e} c2={candidate.c2:.8e} "
                    f"res={float(row['residual']):.3e} fit={_finite(row.get('pairFitScore'), math.nan):.3e} "
                    f"contain={_finite(row.get('pairContainmentScore'), math.nan):.3e} "
                    f"thick={row['tooThick']} eligible={row['selectionEligible']}",
                    flush=True,
                )
        if group.rank == 0:
            _drain_thick_gossip(leaders, known_thick)
            leaders.Barrier()
            _drain_thick_gossip(leaders, known_thick)
            gathered = leaders.gather((local_rows, local_trials, local_pruned), root=0)
        else:
            gathered = None
        if world.rank == 0:
            generation_rows = [row for part in gathered for row in part[0]]
            generation_trials = [row for part in gathered for row in part[1]]
            generation_pruned = [row for part in gathered for row in part[2]]
            generation_rows.sort(key=lambda row: int(row["candidate"]))
            all_rows.extend(generation_rows)
            all_trials.extend(generation_trials)
            all_pruned.extend(generation_pruned)
            incumbent = min(
                all_rows,
                key=lambda row: _continuation_key(row, ranking_criteria),
            )
            eligible_rows = [row for row in all_rows if _bool(row["selectionEligible"])]
            best_value = (
                min(float(row["leakageRel"]) + float(row["missingRel"]) for row in eligible_rows)
                if eligible_rows
                else math.inf
            )
            if math.isfinite(previous_best) and math.isfinite(best_value):
                relative = abs(previous_best - best_value) / max(abs(previous_best), 1.0e-30)
                plateau_count = plateau_count + 1 if relative <= float(args.plateau_rtol) else 0
            previous_best = best_value
            incumbent_path = output_dir / str(incumbent["statePath"])
        else:
            generation_rows = None
            incumbent = None
            incumbent_path = None
        if args.interactive_candidate_plots:
            generation_rows = world.bcast(generation_rows, root=0)
            if group_id == 0:
                if candidate_live_plotter is None:
                    raise RuntimeError("rank group zero did not initialize the live candidate plotter")
                _show_interactive_candidates(
                    group=group,
                    problem=problem,
                    rows=generation_rows,
                    output_dir=output_dir,
                    plotter=candidate_live_plotter,
                )
            world.barrier()
        incumbent = world.bcast(incumbent, root=0)
        incumbent_path = world.bcast(incumbent_path, root=0)
        reduced.load_global_initial_state(Path(incumbent_path), problem["u"], group)
        incumbent_state = problem["u"].x.array.copy()
        stop = world.bcast(
            bool(world.rank == 0 and plateau_count >= int(args.plateau_generations)),
            root=0,
        )
        if stop:
            break

    if group.rank == 0:
        counter_win.Free()
        leaders.Free()
    rows_for_render = all_rows if world.rank == 0 else None
    rows_for_render = _render_all_candidates(
        world=world,
        group=group,
        group_id=group_id,
        group_count=group_count,
        problem=problem,
        rows=rows_for_render,
        output_dir=output_dir,
        args=args,
    )
    if world.rank == 0:
        all_rows = rows_for_render or all_rows
        eligible = [row for row in all_rows if _bool(row.get("selectionEligible"))]
        strict_eligible = [
            row for row in all_rows if _bool(row.get("strictSelectionEligible"))
        ]
        diagnostic = min(all_rows, key=adaptive_result_key) if all_rows else None
        winner = min(eligible, key=adaptive_result_key) if eligible else None
        strict_winner = (
            min(strict_eligible, key=adaptive_result_key) if strict_eligible else None
        )
        if winner is not None:
            shutil.copy2(output_dir / str(winner["statePath"]), output_dir / "winner_state.npz")
        _write_csv(output_dir / "grid.csv", all_rows, ADAPTIVE_BASE_FIELDS)
        _write_csv(
            output_dir / "geometry.csv",
            all_rows,
            (
                "stage",
                "candidate",
                "uid",
                "c1",
                "c2",
                "c1Hat",
                "c2Hat",
                "converged",
                "residual",
                "selectionEligible",
                "strictSelectionEligible",
                "explorationEligible",
                "handoffGeometryEligible",
                "handoffGeometryRejectionReason",
                "strictGeometryEligible",
                "geometryEligible",
                "geometryRejectionReason",
                "r1",
                "r2",
                "lowerContainment",
                "upperContainment",
                "pairFitScore",
                "pairContainmentScore",
                "normalizedCurveError",
                "minimumHandoffContainment",
                "minimumHandoffJaccard",
                "maximumHandoffNormalizedCurveError",
                "robustTauSpan",
                "robustSpanLimit",
                "robustTooThick",
                "hardSpanTooThick",
                "robustSpanWarning",
                "tooThick",
                "physicalTooThick",
                "dominanceWitness",
                "monotoneForBisection",
                "targetUnderresolved",
                "hardLeakage",
                "hardMissing",
                "hardJaccard",
            ),
        )
        _write_csv(output_dir / "branch_trials.csv", all_trials, (*ADAPTIVE_BASE_FIELDS, "selected"))
        _write_csv(output_dir / "pruned.csv", all_pruned, PRUNED_FIELDS)
        payload = {
            "version": 1,
            "status": "HANDOFF_ELIGIBLE_WINNER" if winner else "NO_HANDOFF_ELIGIBLE_SEED",
            "winner": winner,
            "strictWinner": strict_winner,
            "diagnosticBest": diagnostic,
            "candidateCount": len(all_rows),
            "prunedCount": len(all_pruned),
            "search": {
                "thresholdMinimum": 0.0,
                "thresholdMaximum": problem["tmax"],
                "normalizedDomain": "0 <= c1/Tmax < c2/Tmax <= 1",
                "adaptiveGenerations": args.adaptive_generations,
                "initialNodes": args.initial_nodes or automatic_initial_node_count(group_count),
                "gridNewtonTolerance": args.grid_newton_tol,
                "gridNewtonMaxIterations": args.grid_newton_max_it,
                "gridNewtonSoftCap": args.grid_newton_soft_cap,
                "gridNewtonSoftCapChunk": args.grid_newton_soft_cap_chunk,
                "gridNewtonSoftCapFactor": args.grid_newton_soft_cap_factor,
                "gridNewtonSoftCapWindow": args.grid_newton_soft_cap_window,
                "gridNewtonSoftCapContraction": args.grid_newton_soft_cap_contraction,
                "plateauRelativeTolerance": args.plateau_rtol,
                "plateauGenerations": args.plateau_generations,
                "minimumHandoffContainment": args.minimum_practical_containment,
                "minimumHandoffJaccard": args.minimum_handoff_jaccard,
                "maximumHandoffNormalizedCurveError": args.maximum_handoff_normalized_curve_error,
                "generationBudgets": generation_budgets,
                "pruning": {
                    "method": (
                        "conservative same-row/same-column branch-affinity "
                        "heuristic; not a proof"
                    ),
                    "branchAffinityAvailable": False,
                    "thicknessDecision": (
                        "robustTooThick AND "
                        "(hardSpanTooThick OR physicalTooThick)"
                    ),
                    "witnessRequirements": [
                        "converged",
                        "boundSatisfied",
                        "lowerCurveActive",
                        "upperCurveActive",
                        "lowerCurveResolved",
                        "upperCurveResolved",
                        "monotoneForBisection",
                        "bothThresholdsActive",
                        "activityAreaRel > 1e-8",
                        "not targetUnderresolved",
                    ],
                },
            },
            "problem": {
                "mesh": problem["mesh_path"],
                "geometry": problem["geometry_mode"],
                "cells": problem["nt"],
                "dofs": problem["ndof"],
                "order": reduced_args.order,
                "torsionMaximum": problem["tmax"],
                "alphaT1": problem["params"].alpha_t1,
                "alphaT2": problem["params"].alpha_t2,
            },
            "mpi": {
                "worldRanks": world.size,
                "candidateGroups": group_count,
                "ranksPerCandidate": args.ranks_per_candidate,
                "scheduler": "MPI-RMA fetch-and-add",
                "renderGroups": (
                    group_count if int(args.render_groups) == 0 else int(args.render_groups)
                ),
            },
            "reducedArgv": reduced_argv,
            "initialState": str(output_dir / "winner_state.npz") if winner else None,
        }
        (output_dir / "winner.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"ADAPTIVE_SEARCH status={payload['status']} solved={len(all_rows)} "
            f"pruned={len(all_pruned)}",
            flush=True,
        )
    else:
        winner = None
    winner = world.bcast(winner, root=0)
    return 0 if winner is not None else 2


def _metadata_for_mesh(mesh: Path | None) -> dict[str, Any] | None:
    if mesh is None:
        return None
    candidates = (mesh.with_suffix(mesh.suffix + ".json"), mesh.with_suffix(".json"))
    for path in candidates:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if all(key in payload for key in ("vertices", "edges", "cells")):
                return payload
    return None


def _automatic_ranks(args: argparse.Namespace, reduced_argv: Sequence[str]) -> tuple[int, int, int]:
    parsed = reduced.parse_args(list(reduced_argv))
    metadata = _metadata_for_mesh(parsed.mesh)
    dofs = lagrange_dofs_from_metadata(metadata, parsed.order) if metadata else None
    try:
        candidate_ranks = select_mpi_ranks(parsed.mesh_size, parsed.order, dofs)
    except ValueError:
        candidate_ranks = 1
    if int(args.ranks_per_candidate) > 0:
        candidate_ranks = int(args.ranks_per_candidate)
    available = max(1, int(args.available_cores))
    search_ranks = int(args.search_ranks)
    if search_ranks <= 0:
        search_ranks = max(candidate_ranks, (available // candidate_ranks) * candidate_ranks)
    optimization_ranks = int(args.optimization_ranks)
    if optimization_ranks <= 0:
        optimization_ranks = candidate_ranks
    return candidate_ranks, search_ranks, optimization_ranks


def _forward_search_options(args: argparse.Namespace, output_dir: Path, ranks: int) -> list[str]:
    return [
        "--output-dir", str(output_dir),
        "--initial-nodes", str(args.initial_nodes),
        "--adaptive-generations", str(args.adaptive_generations),
        "--max-candidates", str(args.max_candidates),
        "--beam-candidates", str(args.beam_candidates),
        "--grid-newton-tol", str(args.grid_newton_tol),
        "--grid-newton-max-it", str(args.grid_newton_max_it),
        "--grid-newton-soft-cap-chunk", str(args.grid_newton_soft_cap_chunk),
        "--grid-newton-soft-cap-factor", str(args.grid_newton_soft_cap_factor),
        "--grid-newton-soft-cap-window", str(args.grid_newton_soft_cap_window),
        "--grid-newton-soft-cap-contraction", str(args.grid_newton_soft_cap_contraction),
        "--grid-area-weight", str(args.grid_area_weight),
        "--grid-missing-weight", str(args.grid_missing_weight),
        "--grid-seed", str(args.grid_seed),
        "--grid-bound-tol", str(args.grid_bound_tol),
        "--ranks-per-candidate", str(ranks),
        "--contour-histogram-bins", str(args.contour_histogram_bins),
        "--contour-kernel-scale", str(args.contour_kernel_scale),
        "--minimum-practical-containment", str(args.minimum_practical_containment),
        "--minimum-handoff-jaccard", str(args.minimum_handoff_jaccard),
        "--maximum-handoff-normalized-curve-error", str(args.maximum_handoff_normalized_curve_error),
        "--plateau-rtol", str(args.plateau_rtol),
        "--plateau-generations", str(args.plateau_generations),
        "--candidate-png-width", str(args.candidate_png_width),
        "--candidate-png-height", str(args.candidate_png_height),
        "--render-groups", str(args.render_groups),
        "--grid-newton-soft-cap" if args.grid_newton_soft_cap else "--no-grid-newton-soft-cap",
        "--save-candidate-pngs" if args.save_candidate_pngs else "--no-save-candidate-pngs",
        "--interactive-candidate-plots" if args.interactive_candidate_plots else "--no-interactive-candidate-plots",
        "--grid-require-active-thresholds" if args.grid_require_active_thresholds else "--no-grid-require-active-thresholds",
    ]


def _handoff_epsilon_args(args: argparse.Namespace) -> list[str]:
    """Build optimizer-only smoothing overrides after adaptive selection."""

    value = getattr(args, "handoff_eps_phi", None)
    if value is None:
        return []
    return ["--eps-mode", "fixed", "--eps-phi", f"{float(value):.17g}"]


def _run_workflow_in_directory(args: argparse.Namespace, output_dir: Path) -> int:
    reduced_argv, _ = _compat_reduced_args(args)
    candidate_ranks, search_ranks, optimization_ranks = _automatic_ranks(args, reduced_argv)
    if search_ranks % candidate_ranks:
        raise ValueError("search ranks must be divisible by ranks per candidate")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"workflow output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    env["PATH"] = str(Path(args.python).resolve().parent) + os.pathsep + env.get("PATH", "")
    module = Path(__file__).resolve()
    search_dir = output_dir / "grid_search"
    search_command = [
        *brute.mpirun_prefix(args.mpirun, search_ranks),
        args.python,
        str(module),
        "search",
        *_forward_search_options(args, search_dir, candidate_ranks),
        "--",
        *reduced_argv,
    ]
    search_code = brute.run_logged(search_command, output_dir / "search_stdout.txt", env)
    winner_file = search_dir / "winner.json"
    if not winner_file.is_file():
        raise RuntimeError("adaptive search did not write winner.json")
    document = json.loads(winner_file.read_text(encoding="utf-8"))
    winner = document.get("winner")
    optimizer_command = None
    optimizer_code = None
    if winner is not None:
        optimizer_command = [
            *brute.mpirun_prefix(args.mpirun, optimization_ranks),
            args.python,
            str(Path(reduced.__file__).resolve()),
            "--run-dir", str(output_dir / "optimization"),
            "--c1-phi", f"{float(winner['c1']):.17g}",
            "--c2-phi", f"{float(winner['c2']):.17g}",
            "--initial-state", str(search_dir / "winner_state.npz"),
            *reduced_argv,
            *_handoff_epsilon_args(args),
        ]
        optimizer_code = brute.run_logged(
            optimizer_command, output_dir / "optimizer_stdout.txt", env
        )
    workflow = {
        "searchCommand": search_command,
        "searchExitCode": search_code,
        "optimizerCommand": optimizer_command,
        "optimizerExitCode": optimizer_code,
        "winner": winner,
        "autoRanks": {
            "candidate": candidate_ranks,
            "search": search_ranks,
            "optimization": optimization_ranks,
        },
    }
    (output_dir / "workflow.json").write_text(
        json.dumps(workflow, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return int(optimizer_code) if optimizer_code is not None else 2


def run_workflow(args: argparse.Namespace) -> int:
    """Run search and optimization, optionally deleting all transient files."""

    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("run must be launched without mpirun")
    transient_root: Path | None = None
    if bool(args.transient_output):
        transient_root = Path(
            tempfile.mkdtemp(prefix="torsion_adaptive_interactive_")
        ).resolve()
        output_dir = transient_root / "workflow"
        print(f"TRANSIENT_OUTPUT path={output_dir} retained=0", flush=True)
    else:
        output_dir = args.output_dir.expanduser().resolve()

    try:
        return _run_workflow_in_directory(args, output_dir)
    finally:
        if transient_root is not None:
            shutil.rmtree(transient_root, ignore_errors=True)
            print(f"TRANSIENT_OUTPUT_REMOVED path={transient_root}", flush=True)


def render_figures(workflow_dir: Path, figure_dir: Path, prefix: str | None) -> int:
    """Render search evidence and a target-once handoff storyboard.

    Target fields occupy one design strip; all remaining tiles are candidates
    or reduced-optimization iterates with the target-band contour retained.
    """

    from projects.diocotron.studies.torsion_optimizer.figures.adaptive import render_adaptive_search_figures

    workflow_dir = Path(workflow_dir).resolve()
    figure_dir = Path(figure_dir).resolve()
    figure_prefix = prefix or "adaptive_threshold"
    bundle = render_adaptive_search_figures(
        workflow_dir,
        figure_dir,
        figure_prefix,
        contacts_per_sheet=4,
    )
    for output in bundle.output_paths():
        print(f"FIGURE {output}")

    # Optimization-specific traces require an eligible handoff. They are an
    # optional supplement; the complete search record above never depends on
    # their existence and remains valid for a no-winner result. The shared
    # compositor strips fixed target panels from every OPT/FINAL tile while
    # preserving the complete native diagnostic frames in the frame bundle.
    frames_path = workflow_dir / "optimization" / "logs" / "frames.csv"
    if bundle.has_eligible_winner and frames_path.is_file():
        return brute.render_workflow_figures(workflow_dir, figure_dir, figure_prefix)
    return 0


def add_search_options(parser: argparse.ArgumentParser, *, orchestrated: bool) -> None:
    parser.add_argument("--output-dir", type=Path, required=not orchestrated)
    parser.add_argument("--initial-nodes", type=int, default=0, help="zero chooses from MPI group count")
    parser.add_argument("--adaptive-generations", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=256)
    parser.add_argument("--beam-candidates", type=int, default=0)
    parser.add_argument("--grid-newton-tol", type=float, default=1.0e-6)
    parser.add_argument(
        "--grid-newton-max-it",
        type=int,
        default=40,
        help="initial candidate Newton budget; the soft cap may extend it",
    )
    parser.add_argument(
        "--grid-newton-soft-cap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="extend a spent candidate budget only while recent residuals clearly contract",
    )
    parser.add_argument("--grid-newton-soft-cap-chunk", type=int, default=20)
    parser.add_argument("--grid-newton-soft-cap-factor", type=float, default=4.0)
    parser.add_argument("--grid-newton-soft-cap-window", type=int, default=4)
    parser.add_argument("--grid-newton-soft-cap-contraction", type=float, default=0.98)
    parser.add_argument("--grid-area-weight", type=float, default=0.0)
    parser.add_argument("--grid-missing-weight", type=float, default=1.0)
    parser.add_argument("--grid-require-active-thresholds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grid-seed", choices=("phi-target", "torsion", "both"), default="both")
    parser.add_argument("--grid-bound-tol", type=float, default=1.0e-8)
    parser.add_argument("--ranks-per-candidate", type=int, default=0 if orchestrated else 1)
    parser.add_argument("--contour-histogram-bins", type=int, default=64)
    parser.add_argument("--contour-kernel-scale", type=float, default=1.5)
    parser.add_argument("--minimum-practical-containment", type=float, default=0.75)
    parser.add_argument("--minimum-handoff-jaccard", type=float, default=0.40)
    parser.add_argument(
        "--maximum-handoff-normalized-curve-error",
        type=float,
        default=2.0,
    )
    parser.add_argument("--plateau-rtol", type=float, default=1.0e-3)
    parser.add_argument("--plateau-generations", type=int, default=2)
    parser.add_argument("--save-candidate-pngs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--interactive-candidate-plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show every fully converged candidate after each parallel search generation",
    )
    parser.add_argument("--candidate-png-width", type=int, default=2400)
    parser.add_argument("--candidate-png-height", type=int, default=1250)
    parser.add_argument(
        "--render-groups",
        type=int,
        default=0,
        help="candidate subgroups used for deferred PNG rendering; zero uses all",
    )
    if orchestrated:
        parser.add_argument(
            "--transient-output",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="use a temporary workflow directory and delete it when the run exits",
        )
        parser.add_argument(
            "--handoff-eps-phi",
            type=float,
            default=None,
            help=(
                "optional fixed epsilon used only by the post-search optimizer; "
                "the adaptive search retains the reduced-argument smoothing"
            ),
        )
        parser.add_argument("--search-ranks", type=int, default=0)
        parser.add_argument("--optimization-ranks", type=int, default=0)
        parser.add_argument("--available-cores", type=int, default=20)
        parser.add_argument("--mpirun", default="mpirun")
        parser.add_argument("--python", default=sys.executable)
    parser.add_argument("reduced_args", nargs=argparse.REMAINDER)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    add_search_options(sub.add_parser("search"), orchestrated=False)
    add_search_options(sub.add_parser("run"), orchestrated=True)
    figures = sub.add_parser("figures")
    figures.add_argument("--workflow-dir", type=Path, required=True)
    figures.add_argument("--figure-dir", type=Path, required=True)
    figures.add_argument("--figure-prefix", default=None)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.command == "figures":
        return
    if args.command == "run":
        if args.transient_output and args.output_dir is not None:
            raise ValueError("--transient-output and --output-dir are mutually exclusive")
        if not args.transient_output and args.output_dir is None:
            raise ValueError("run requires --output-dir unless --transient-output is used")
        if args.handoff_eps_phi is not None and (
            not math.isfinite(args.handoff_eps_phi) or args.handoff_eps_phi <= 0.0
        ):
            raise ValueError("--handoff-eps-phi must be finite and positive")
    if args.initial_nodes not in {0} and args.initial_nodes < 3:
        raise ValueError("initial-nodes must be zero or at least three")
    if args.adaptive_generations < 1 or args.grid_newton_max_it < 1:
        raise ValueError("generation and Newton iteration counts must be positive")
    if args.max_candidates < 0 or args.beam_candidates < 0:
        raise ValueError("candidate budgets must be nonnegative")
    if args.grid_newton_tol <= 0.0 or args.contour_histogram_bins < 16:
        raise ValueError("Newton tolerance must be positive and histogram needs >=16 bins")
    if args.grid_newton_soft_cap_chunk < 1:
        raise ValueError("Newton soft-cap chunk must be positive")
    if (
        not math.isfinite(args.grid_newton_soft_cap_factor)
        or args.grid_newton_soft_cap_factor < 1.0
    ):
        raise ValueError("Newton soft-cap factor must be finite and at least one")
    if args.grid_newton_soft_cap_window < 2:
        raise ValueError("Newton soft-cap window must contain at least two contractions")
    if not 0.0 < args.grid_newton_soft_cap_contraction < 1.0:
        raise ValueError("Newton soft-cap contraction must lie strictly in (0,1)")
    if args.plateau_generations < 1 or args.plateau_rtol < 0.0:
        raise ValueError("invalid plateau controls")
    if not 0.0 <= args.minimum_practical_containment <= 1.0:
        raise ValueError("minimum practical containment must lie in [0,1]")
    if not 0.0 <= args.minimum_handoff_jaccard <= 1.0:
        raise ValueError("minimum handoff Jaccard must lie in [0,1]")
    if (
        not math.isfinite(args.maximum_handoff_normalized_curve_error)
        or args.maximum_handoff_normalized_curve_error <= 0.0
    ):
        raise ValueError("maximum handoff normalized curve error must be positive")
    if args.candidate_png_width < 800 or args.candidate_png_height < 500:
        raise ValueError("candidate PNG dimensions are too small")
    if args.render_groups < 0:
        raise ValueError("render-groups must be nonnegative")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    if args.command == "search":
        return run_search(args)
    if args.command == "run":
        return run_workflow(args)
    return render_figures(args.workflow_dir, args.figure_dir, args.figure_prefix)


if __name__ == "__main__":
    raise SystemExit(main())
