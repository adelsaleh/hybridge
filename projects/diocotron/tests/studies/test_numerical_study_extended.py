from __future__ import annotations

import json
from pathlib import Path

from projects.diocotron.studies.torsion_optimizer.extended import (
    FIGURE_CONTRACT,
    STRICT_STORYBOARD_BANDS,
    select_contract_representatives,
    validate_figure_contract,
)


def _representative_row(
    root: Path,
    *,
    case_id: str,
    geometry: str,
    band: tuple[float, float],
    kind: str = "trajectory_v3",
    state: str = "completed",
    dof_target: int = 200_000,
    missing: tuple[str, ...] = (),
) -> dict[str, object]:
    run_dir = root / case_id
    output = run_dir / "out"
    logs = run_dir / "logs"
    output.mkdir(parents=True)
    logs.mkdir()
    equilibrium = output / "equilibrium.npz"
    if "equilibrium" not in missing:
        equilibrium.write_bytes(b"real-checkpoint")
    if "optimization" not in missing:
        (logs / "optimization.csv").write_text(
            "k,c1Phi,c2Phi\n0,0.1,0.2\n", encoding="utf-8",
        )
    if "newton" not in missing:
        (logs / "newton.csv").write_text(
            "residual,ksp_iterations\n1e-9,3\n", encoding="utf-8",
        )
    if "phases" not in missing:
        (logs / "phases.csv").write_text(
            "phase,elapsed,calls\nmesh,0.25,1\n", encoding="utf-8",
        )
    return {
        "id": case_id,
        "kind": kind,
        "geometry": geometry,
        "state": state,
        "dof_target": dof_target,
        "alpha_t1": band[0],
        "alpha_t2": band[1],
        "run_dir": str(run_dir),
        "equilibrium": str(equilibrium),
    }


def test_contract_representatives_prefer_complete_exact_storyboard_rows(tmp_path):
    rows = []
    expected = {}
    for geometry, band in STRICT_STORYBOARD_BANDS.items():
        legacy = _representative_row(
            tmp_path,
            case_id=f"legacy-{geometry}",
            geometry=geometry,
            band=(0.60, 0.70),
            kind="trajectory",
            dof_target=500_000,
        )
        exact = _representative_row(
            tmp_path,
            case_id=f"exact-{geometry}",
            geometry=geometry,
            band=band,
            dof_target=200_000,
        )
        rows.extend((legacy, exact))
        expected[geometry] = exact["id"]

    selected = select_contract_representatives(rows)
    reversed_selected = select_contract_representatives(list(reversed(rows)))
    assert {geometry: row["id"] for geometry, row in selected.items()} == expected
    assert {
        geometry: row["id"] for geometry, row in reversed_selected.items()
    } == expected


def test_contract_representative_falls_back_to_artifact_complete_trajectory(
    tmp_path,
):
    exact_incomplete = _representative_row(
        tmp_path,
        case_id="exact-iter-incomplete",
        geometry="iter",
        band=STRICT_STORYBOARD_BANDS["iter"],
        dof_target=500_000,
        missing=("phases",),
    )
    legacy_complete = _representative_row(
        tmp_path,
        case_id="legacy-iter-complete",
        geometry="iter",
        band=(0.60, 0.70),
        kind="trajectory",
        dof_target=200_000,
    )

    selected = select_contract_representatives([
        exact_incomplete, legacy_complete,
    ])
    assert selected["iter"]["id"] == "legacy-iter-complete"


def test_figure_registry_contract_is_png_only(tmp_path):
    bundle = tmp_path / "bundle"
    figure_dir = bundle / "figures"
    generated_dir = bundle / "generated"
    figure_dir.mkdir(parents=True)
    generated_dir.mkdir()
    entries = []
    for spec in FIGURE_CONTRACT:
        (figure_dir / spec["file"]).write_bytes(b"png")
        entries.append({**spec, "status": "actual"})
    entries.append({
        "id": "legacy-vector",
        "file": "legacy-vector.svg",
        "status": "actual",
    })
    (generated_dir / "figure_registry.json").write_text(
        json.dumps({
            "format": "hdgfem_torsion_optimizer_figure_registry_v1",
            "asset_format": "png",
            "stage": "preliminary",
            "figures": entries,
        }),
        encoding="utf-8",
    )
    errors = validate_figure_contract(bundle)
    assert "non-PNG figure asset registered: legacy-vector.svg" in errors
