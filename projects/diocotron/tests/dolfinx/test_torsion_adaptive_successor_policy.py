"""Focused policy tests for the adaptive threshold-search successor."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[4] / "projects/diocotron/dolfinx"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import projects.diocotron.dolfinx.torsion.search.adaptive as adaptive  # noqa: E402


class StaticWorkspace:
    def __init__(self, values):
        self.values = dict(values)

    def evaluate(self, *, c1, c2):
        assert c1 < c2
        return dict(self.values)


def geometry_values(**updates):
    values = {
        "geometryEligible": 0,
        "geometryRejectionReason": "STRICT_ONLY_FAILURE",
        "lowerCurveActive": 1,
        "upperCurveActive": 1,
        "lowerCurveResolved": 1,
        "upperCurveResolved": 1,
        "lowerContainment": 0.082,
        "upperContainment": 0.091,
        "tooThick": 0,
        "hardJaccard": 0.507,
        "r1": 0.0184,
        "r2": -0.011,
        "lowerPositionTolerance": 0.01,
        "upperPositionTolerance": 0.01,
        "pairFitScore": 0.082,
        "pairContainmentScore": 0.082,
        "robustTauSpan": 0.171,
        "robustSpanLimit": 0.1445,
        "robustTooThick": 1,
        "hardSpanTooThick": 0,
        "robustSpanWarning": 1,
    }
    values.update(updates)
    return values


def classify(**updates):
    return adaptive._geometry_dict(
        StaticWorkspace(geometry_values(**updates)),
        0.2,
        0.3,
        minimum_practical_containment=0.05,
        minimum_handoff_jaccard=0.40,
        maximum_handoff_normalized_curve_error=2.0,
    )


def test_exploration_handoff_and_strict_certification_are_separate():
    result = classify()

    assert result["explorationEligible"] == 1
    assert result["handoffGeometryEligible"] == 1
    assert result["geometryEligible"] == 1  # compatibility alias
    assert result["strictGeometryEligible"] == 0
    assert result["normalizedCurveError"] == pytest.approx(1.84)
    assert result["robustSpanWarning"] == 1


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"hardJaccard": 0.39}, "HANDOFF_JACCARD_INSUFFICIENT"),
        ({"r1": 0.021}, "HANDOFF_CURVE_ERROR_EXCESSIVE"),
        ({"lowerContainment": 0.049}, "HANDOFF_CONTAINMENT_INSUFFICIENT"),
        ({"tooThick": 1}, "THICK_BAND_REJECTED"),
    ],
)
def test_handoff_guards_reject_without_changing_exploration(updates, reason):
    result = classify(**updates)

    assert result["explorationEligible"] == 1
    assert result["handoffGeometryEligible"] == 0
    assert reason in result["handoffGeometryRejectionReason"]


def test_active_partial_attempt_precedes_converged_inactive_zero():
    common = {
        "candidate": 7,
        "selectionEligible": 0,
        "geometryEligible": 0,
        "boundSatisfied": 1,
        "pairFitScore": 0.0,
        "pairContainmentScore": 0.0,
        "leakageRel": 1.0,
        "missingRel": 1.0,
        "tooThick": 0,
    }
    partial = {
        **common,
        "converged": 0,
        "bothThresholdsActive": 1,
        "lowerCurveActive": 1,
        "upperCurveActive": 1,
        "residual": 1.0e-5,
    }
    zero = {
        **common,
        "converged": 1,
        "bothThresholdsActive": 0,
        "lowerCurveActive": 0,
        "upperCurveActive": 0,
        "residual": 1.0e-15,
    }

    assert adaptive.adaptive_result_key(partial) < adaptive.adaptive_result_key(zero)


def test_driver_dominance_rule_defensively_rejects_unvetted_thickness_rows():
    witness = {
        "c1": 0.2,
        "c2": 0.4,
        "uid": "vetted",
        "tooThick": 1,
        "converged": 1,
        "boundSatisfied": 1,
        "lowerCurveActive": 1,
        "upperCurveActive": 1,
        "lowerCurveResolved": 1,
        "upperCurveResolved": 1,
        "monotoneForBisection": 1,
        "bothThresholdsActive": 1,
        "activityAreaRel": 0.8,
        "targetUnderresolved": 0,
    }
    assert adaptive.dominance_reason(0.2, 0.5, [witness], tolerance=1.0e-12) == (
        "THICK_ROW_C2_DOMINANCE",
        "vetted",
    )

    # A robust/corroborated thickness flag from a nonconverged branch cannot
    # eliminate another candidate, even when the coordinate relation matches.
    witness["converged"] = 0
    assert adaptive.dominance_reason(0.2, 0.5, [witness], tolerance=1.0e-12) is None


def test_successor_cli_exposes_candidate_budget_handoff_and_parallel_render_defaults(tmp_path):
    args = adaptive.parse_args(["run", "--output-dir", str(tmp_path / "run")])

    assert args.grid_newton_max_it == 40
    assert args.grid_newton_soft_cap is True
    assert args.grid_newton_soft_cap_factor == pytest.approx(4.0)
    assert args.interactive_candidate_plots is False
    assert args.transient_output is False
    assert args.minimum_handoff_jaccard == pytest.approx(0.40)
    assert args.maximum_handoff_normalized_curve_error == pytest.approx(2.0)
    assert args.render_groups == 0


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"converged": 1, "newtonStatus": "CONVERGED_RESIDUAL"}, True),
        ({"converged": 0, "newtonStatus": "FAIL_LS"}, False),
        ({"converged": 0, "newtonStatus": "MAX_NEWTON"}, False),
        ({"converged": 0, "newtonStatus": "FAIL_STEP_STAGNATION"}, False),
    ],
)
def test_live_candidate_plot_requires_full_newton_convergence(row, expected):
    assert adaptive.candidate_live_plot_eligible(row) is expected


def test_transient_interactive_cli_forwards_density_and_verbose_plotting(tmp_path):
    args = adaptive.parse_args(
        [
            "run",
            "--transient-output",
            "--interactive-candidate-plots",
            "--no-save-candidate-pngs",
            "--handoff-eps-phi",
            "0.003",
            "--search-ranks",
            "20",
            "--ranks-per-candidate",
            "1",
            "--optimization-ranks",
            "1",
            "--",
            "--plot",
            "--plot-fields",
            "density",
            "--plot-accepted-states",
            "--verbosity",
            "2",
        ]
    )
    adaptive.validate_args(args)
    forwarded, reduced_args = adaptive._compat_reduced_args(args)

    assert args.output_dir is None
    assert args.transient_output is True
    assert args.interactive_candidate_plots is True
    assert args.save_candidate_pngs is False
    assert args.handoff_eps_phi == pytest.approx(0.003)
    handoff_args = adaptive._handoff_epsilon_args(args)
    assert handoff_args[:3] == ["--eps-mode", "fixed", "--eps-phi"]
    assert float(handoff_args[3]) == pytest.approx(0.003)
    assert reduced_args.plot is True
    assert reduced_args.plot_fields == "density"
    assert reduced_args.plot_accepted_states is True
    assert reduced_args.plot_off_screen is False
    assert reduced_args.verbosity == 2
    assert "--plot-fields" in forwarded


def test_transient_output_rejects_a_persistent_output_directory(tmp_path):
    args = adaptive.parse_args(
        [
            "run",
            "--transient-output",
            "--output-dir",
            str(tmp_path / "persistent"),
        ]
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        adaptive.validate_args(args)
