"""Stationary ADR diffusion-stabilization qualification-study tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.advection_diffusion_reaction.study_diffusion_stabilization import (
    ERROR_KEYS,
    add_pairwise_rates,
    parse_choices,
    parse_float_list,
    parse_int_list,
    run_stabilization_study,
    stabilization_cases,
)


def _rate_row(mesh_h: float, error: float) -> dict[str, object]:
    """Build one synthetic convergence row with every required error key."""
    row: dict[str, object] = {
        "order": 2,
        "stabilization_mode": "global-length",
        "gamma_d": 1.0,
        "flux_postprocess_space": "RT_projection",
        "mesh_h": mesh_h,
    }
    row.update({key: error for key in ERROR_KEYS})
    return row


def test_list_parsers_and_case_expansion() -> None:
    """Parse ordered unique values and expand gamma-dependent cases."""
    assert parse_float_list("0.5,1,2", label="gamma") == (0.5, 1.0, 2.0)
    assert parse_int_list("1,3", label="order") == (1, 3)
    assert parse_choices(
        "inverse-h,global-length",
        label="mode",
        choices=("global-length", "inverse-h"),
    ) == ("inverse-h", "global-length")
    assert stabilization_cases(
        ("global-length", "inverse-h"),
        (0.5, 2.0),
    ) == (
        ("global-length", 0.5),
        ("global-length", 2.0),
        ("inverse-h", None),
    )
    with pytest.raises(ValueError, match="duplicates"):
        parse_float_list("1,1", label="gamma")
    with pytest.raises(ValueError, match="unsupported"):
        parse_choices("local-spectral", label="mode", choices=("inverse-h",))


def test_pairwise_rates_keep_sampled_errors_and_use_actual_h() -> None:
    """Compute rates without replacing sampled-Linf error columns."""
    coarse = _rate_row(0.4, 0.16)
    fine = _rate_row(0.2, 0.04)
    rows = [fine, coarse]
    add_pairwise_rates(rows)

    assert coarse["primal_linf_sampled"] == 0.16
    assert fine["primal_linf_sampled"] == 0.04
    assert coarse["primal_linf_sampled_rate"] is None
    assert fine["primal_linf_sampled_rate"] == pytest.approx(2.0)
    assert fine["post_primal_l2_rate"] == pytest.approx(2.0)
    assert fine["post_total_flux_l2_rate"] == pytest.approx(2.0)


def test_tiny_host_study_writes_machine_and_human_readable_outputs(tmp_path) -> None:
    """Exercise one manufactured solve through diagnostics and serialization."""
    pytest.importorskip("gmsh")
    rows, csv_path, json_path, markdown_path = run_stabilization_study(
        peclet=10.0,
        mesh_sizes=(0.55,),
        orders=(1,),
        gammas=(1.0,),
        modes=("global-length",),
        flux_spaces=("l2_closest",),
        domain_length=1.0,
        assembly_backend="numpy",
        reconstruction_backend="numba",
        postprocessing_backend="numba",
        solver="direct",
        sample_resolution=8,
        error_volume_quad_1d=8,
        condition_max_dofs=300,
        output_dir=tmp_path,
        prefix="smoke",
        verbosity=0,
        solver_verbosity=0,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["tau_diff_min"] == pytest.approx(0.1)
    assert row["tau_diff_max"] == pytest.approx(0.1)
    assert row["linf_kind"] == "triangular-grid-sampled-euclidean"
    assert np.isfinite(row["post_primal_linf_sampled"])
    assert np.isfinite(row["post_total_flux_linf_sampled"])
    assert row["post_primal_linf_sampled_rate"] is None
    assert row["condition_method"] in {"dense-2norm", "skipped"}
    assert csv_path.is_file()
    assert json_path.is_file()
    assert markdown_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload[0]["domain_length"] == 1.0
    assert "not certified continuum norm bounds" in markdown_path.read_text(
        encoding="utf-8"
    )
