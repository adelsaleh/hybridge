from __future__ import annotations

import json
from pathlib import Path

import pytest

from hdgfem.linalg.amgx.config import (
    describe_amgx_preconditioner,
    describe_amgx_solver,
    format_amgx_configuration,
    load_amgx_config,
)


def test_load_amgx_config_applies_standard_overrides(tmp_path) -> None:
    path = tmp_path / "amgx.json"
    path.write_text(json.dumps({
        "solver": {
            "solver": "CG",
            "preconditioner": {"solver": "AMG", "algorithm": "CLASSICAL", "selector": "PMIS"},
        }
    }), encoding="utf-8")

    config, selected = load_amgx_config(
        path, solver="BICGSTAB", tolerance=1.0e-9, maxiter=75, verbose=2
    )

    assert selected == path
    assert describe_amgx_solver(config) == "BICGSTAB"
    assert config["solver"]["tolerance"] == 1.0e-9
    assert config["solver"]["max_iters"] == 75
    assert config["solver"]["print_solve_stats"] == 0
    assert config["solver"]["obtain_timings"] == 1

    fully_verbose, _ = load_amgx_config(path, verbose=3)
    assert fully_verbose["solver"]["print_solve_stats"] == 1
    assert fully_verbose["solver"]["print_solve_stats_interval"] == 1
    assert fully_verbose["solver"]["obtain_timings"] == 0
    assert describe_amgx_preconditioner(config) == "AMG / CLASSICAL / PMIS"


def test_format_amgx_configuration_names_solver_preconditioner_and_controls() -> None:
    config = {
        "solver": {
            "solver": "BICGSTAB",
            "preconditioner": {"solver": "AMG", "algorithm": "AGGREGATION", "smoother": "DILU"},
            "convergence": "ABSOLUTE",
            "tolerance": 1.0e-12,
            "max_iters": 250,
        }
    }

    text = format_amgx_configuration(config)
    assert "iterative solver: BICGSTAB" in text
    assert "preconditioner: AMG / AGGREGATION / DILU" in text
    assert "convergence: ABSOLUTE" in text
    assert "tolerance: 1e-12" in text
    assert "max iterations: 250" in text


def test_load_amgx_config_uses_an_independent_embedded_default(tmp_path) -> None:
    default = {"solver": {"solver": "CG"}}

    config, selected = load_amgx_config(
        None,
        default_path=tmp_path / "missing.json",
        default_config=default,
        solver="GMRES",
    )

    assert selected is None
    assert config["solver"]["solver"] == "GMRES"
    assert default["solver"]["solver"] == "CG"


def test_load_amgx_config_rejects_an_explicit_missing_path(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_amgx_config(tmp_path / "missing.json", default_config={"solver": {}})


def test_diffusion_relative_amgx_config_uses_aggregate_block_l2_norm() -> None:
    config, _ = load_amgx_config(
        Path("configs/amgx/diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive.json")
    )

    solver = config["solver"]
    assert solver["convergence"] == "RELATIVE_INI"
    assert solver["norm"] == "L2"
    assert solver["use_scalar_norm"] == 1
