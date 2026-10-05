"""Transport guard configuration and result handling; no GPU or compilation."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from hybridge.linalg.amgx.device_solver import _amgx_config_for_solve
from hybridge.linalg.results import SolveResult, finalize_solve_result
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime import configuration as runner


@pytest.mark.parametrize("policy", ["none", "amgx-robust"])
def test_transport_guard_covers_primary_and_every_amgx_retry(policy):
    preset = replace(
        preset_by_key("euler_vortex_gas_p6_50k_dt001_t50_raw_cuda_bsr"),
        transport_retry_policy=policy, transport_direct_fallback="cusolver-qr",
    )
    options = runner._make_transport_options(preset, "zero-flux")
    configs = [options.amgx_config]
    configs.extend(attempt["config"] for attempt in options.amgx_retry_attempts
                   if attempt.get("backend", "amgx") == "amgx")
    assert len(configs) == (6 if policy == "amgx-robust" else 1)
    for config in configs:
        solver = _amgx_config_for_solve(config=config, verbose=3)["solver"]
        assert solver["rel_div_tolerance"] == 1000
        assert solver["divergence_patience"] == 5
        assert solver["divergence_grace_iters"] == 10
        assert solver["print_solve_stats_interval"] == 10
        assert solver["monitor_residual"] == solver["store_res_history"] == 1
    assert "config" not in options.amgx_retry_attempts[-1]  # Direct QR is separate.
    poisson = runner._make_poisson_options(preset)
    assert "rel_div_tolerance" not in poisson.amgx_config["solver"]


@pytest.mark.parametrize("factor", [-1, 200])
def test_transport_guard_preserves_explicit_overrides_and_input(factor):
    config = {"solver": {
        "solver": "PBICGSTAB", "rel_div_tolerance": factor,
        "divergence_patience": 3, "divergence_grace_iters": 20,
        "print_solve_stats_interval": 1,
        "preconditioner": {"solver": "BLOCK_JACOBI"},
    }}
    original = deepcopy(config)
    guarded = runner._transport_amgx_divergence_config(config, tolerance=1e-11)
    actual = _amgx_config_for_solve(config=guarded, verbose=3)["solver"]
    for key in ("rel_div_tolerance", "divergence_patience", "divergence_grace_iters",
                "print_solve_stats_interval"):
        assert actual[key] == original["solver"][key]
    guarded["solver"]["preconditioner"]["solver"] = "NOSOLVER"
    assert config == original


def test_transport_guard_supplies_default_solver_when_no_json_is_given():
    guarded = runner._transport_amgx_divergence_config(None, tolerance=1e-9)
    assert guarded["solver"]["solver"] == "BICGSTAB"
    assert guarded["solver"]["tolerance"] == 1e-9
    assert guarded["solver"]["rel_div_tolerance"] == 1000


@pytest.mark.parametrize("residual", [0.0, 1e6])
def test_native_divergence_cannot_be_accepted_or_relabelled_stagnation(residual):
    result = SolveResult(
        x=np.ones(2), info=0, rtol=1e-11, atol=0,
        solver_residual_norm=residual, solver_rhs_norm=1.0,
        solver_relative_residual_norm=residual, solver_residual_target=1e-11,
        physical_residual_norm=residual, physical_rhs_norm=1.0,
        physical_relative_residual_norm=residual, physical_residual_target=1e-11,
    )
    finalize_solve_result(
        result, backend="pyamgx-device", backend_info="diverged",
        backend_success=True, residual_history=[residual] * 64,
    )
    assert not result.converged
    assert result.info != 0
    assert result.backend_info == "diverged"
    assert result.status == "diverged"
    assert result.failure_reason == "backend-divergence"
