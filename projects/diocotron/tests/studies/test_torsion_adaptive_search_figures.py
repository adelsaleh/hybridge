"""Synthetic tests for unconditional adaptive threshold-search figures."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

from PIL import Image, ImageDraw
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[4] / "projects/diocotron/dolfinx"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from projects.diocotron.studies.torsion_optimizer.figures.adaptive import (  # noqa: E402
    CandidateArchiveError,
    _uses_corroborated_thickness,
    render_adaptive_search_figures,
    representative_candidate_rows,
    validate_candidate_png_archive,
)


FIELDS = (
    "stage",
    "candidate",
    "uid",
    "c1Hat",
    "c2Hat",
    "converged",
    "boundSatisfied",
    "newtonStatus",
    "residual",
    "newtonIterations",
    "wallTime",
    "selectionEligible",
    "geometryEligible",
    "strictGeometryEligible",
    "pairFitScore",
    "pairContainmentScore",
    "robustTauSpan",
    "robustSpanLimit",
    "tooThick",
    "r1",
    "r2",
    "lowerPositionTolerance",
    "upperPositionTolerance",
    "lowerBoundaryFit",
    "upperBoundaryFit",
    "hardPrecision",
    "hardRecall",
    "hardJaccard",
    "activeJaccard",
    "leakageRel",
    "missingRel",
    "candidatePng",
)


def _candidate_image(path: Path, index: int) -> None:
    """Create a recognizable synthetic version of the 4-by-2 field layout."""

    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 800, 440
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    colors = (
        "#440154",
        "#3b528b",
        "#21918c",
        "#5ec962",
        "#fde725",
        "#e69f00",
        "#0072b2",
        "#eeeeee",
    )
    for panel, color in enumerate(colors):
        column = panel % 4
        row = panel // 4
        bounds = (
            column * width // 4,
            row * height // 2,
            (column + 1) * width // 4 - 2,
            (row + 1) * height // 2 - 2,
        )
        draw.rectangle(bounds, fill=color)
        draw.text((bounds[0] + 8, bounds[1] + 8), f"panel {panel}; c{index}", fill="black")
    image.save(path, format="PNG")


def _fixture(tmp_path: Path, *, count: int = 5, winner: bool = False):
    workflow = tmp_path / "workflow"
    search = workflow / "grid_search"
    search.mkdir(parents=True)
    rows = []
    for index in range(count):
        generation = 0 if index < 3 else 1
        c1_hat = 0.04 + 0.10 * index
        c2_hat = min(c1_hat + 0.18 + 0.02 * (index % 2), 0.96)
        relative = Path("candidate_pairs") / f"candidate_{index:04d}.png"
        _candidate_image(search / relative, index)
        is_winner = winner and index == count - 1
        rows.append(
            {
                "stage": generation,
                "candidate": index,
                "uid": f"pair-{index:04d}",
                "c1Hat": c1_hat,
                "c2Hat": c2_hat,
                "converged": int(index != 1),
                "boundSatisfied": 1,
                "newtonStatus": "CONVERGED" if index != 1 else "MAX_IT",
                "residual": 10.0 ** (-5 - index),
                "newtonIterations": 8 + 3 * index,
                "wallTime": 0.4 + 0.2 * index,
                "selectionEligible": int(is_winner),
                "geometryEligible": int(is_winner),
                "strictGeometryEligible": 0,
                "pairFitScore": 0.1 + 0.15 * index,
                "pairContainmentScore": 0.2 + 0.12 * index,
                "robustTauSpan": 0.08 + 0.025 * index,
                "robustSpanLimit": 0.16,
                "tooThick": int(index == 1),
                "r1": 0.03 - 0.008 * index,
                "r2": -0.025 + 0.006 * index,
                "lowerPositionTolerance": 0.01,
                "upperPositionTolerance": 0.012,
                "lowerBoundaryFit": 0.20 + 0.13 * index,
                "upperBoundaryFit": 0.25 + 0.11 * index,
                "hardPrecision": 0.25 + 0.12 * index,
                "hardRecall": 0.18 + 0.14 * index,
                "hardJaccard": 0.10 + 0.12 * index,
                "activeJaccard": 0.10 + 0.12 * index,
                "leakageRel": 0.8 - 0.1 * index,
                "missingRel": 0.9 - 0.08 * index,
                "candidatePng": str(relative),
            }
        )
    with (search / "grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (search / "pruned.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("generation", "candidate", "uid", "c1Hat", "c2Hat", "reason"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "generation": 1,
                "candidate": 99,
                "uid": "pair-pruned",
                "c1Hat": 0.12,
                "c2Hat": 0.54,
                "reason": "THICK_ROW_C2_DOMINANCE",
            }
        )
    document = {
        "status": "GEOMETRICALLY_ELIGIBLE_WINNER" if winner else "NO_GEOMETRICALLY_ELIGIBLE_SEED",
        "winner": rows[-1] if winner else None,
        "diagnosticBest": rows[-1],
    }
    (search / "winner.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    return workflow, search, rows


def test_unconditional_figures_and_all_candidate_contacts_without_winner(tmp_path: Path) -> None:
    workflow, _, rows = _fixture(tmp_path, count=5, winner=False)
    output = tmp_path / "figures"

    bundle = render_adaptive_search_figures(
        workflow,
        output,
        "iter_adaptive_synthetic",
        contacts_per_sheet=4,
    )

    assert bundle.candidate_count == 5
    assert bundle.pruned_count == 1
    assert bundle.has_eligible_winner is False
    assert len(bundle.candidate_contacts) == math.ceil(len(rows) / 4)
    assert len(bundle.output_paths()) == 6
    assert not list(output.glob("*.svg"))
    for path in bundle.output_paths():
        assert path.suffix == ".png"
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width >= 1000
            assert image.height >= 400


def test_candidate_archive_rejects_any_missing_solved_png(tmp_path: Path) -> None:
    _, search, rows = _fixture(tmp_path, count=3)
    missing = search / rows[1]["candidatePng"]
    missing.unlink()

    with pytest.raises(CandidateArchiveError, match="missing or empty"):
        validate_candidate_png_archive(rows, search)


def test_candidate_archive_rejects_nonportable_escape(tmp_path: Path) -> None:
    _, search, rows = _fixture(tmp_path, count=2)
    escaped = tmp_path / "outside.png"
    _candidate_image(escaped, 7)
    rows[1]["candidatePng"] = str(escaped)

    with pytest.raises(CandidateArchiveError, match="escapes the search archive"):
        validate_candidate_png_archive(rows, search)


def test_representative_storyboard_is_unique_and_retains_diagnostic_best(tmp_path: Path) -> None:
    _, _, rows = _fixture(tmp_path, count=9, winner=False)
    selected = rows[-1]

    representatives = representative_candidate_rows(rows, selected, False)

    identities = [row["uid"] for _, row in representatives]
    assert len(representatives) <= 6
    assert len(identities) == len(set(identities))
    assert selected["uid"] in identities
    selected_roles = [role for role, row in representatives if row["uid"] == selected["uid"]]
    assert any("no handoff" in role for role in selected_roles)


def test_corroborated_thickness_policy_is_detected_from_manifest() -> None:
    assert not _uses_corroborated_thickness({})
    assert not _uses_corroborated_thickness(
        {"search": {"pruning": {"thicknessDecision": "robustTooThick"}}}
    )
    assert _uses_corroborated_thickness(
        {
            "search": {
                "pruning": {
                    "thicknessDecision": (
                        "robustTooThick AND "
                        "(hardSpanTooThick OR physicalTooThick)"
                    )
                }
            }
        }
    )
