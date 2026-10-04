from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import maximum_filter, minimum_filter

from scripts.guiding_center.cases.guiding_center_cases import euler_vortex_gas
from scripts.guiding_center.cases.guiding_center_presets import PRESETS, preset_by_key
from scripts.guiding_center.runtime.arguments import build_parser
from scripts.guiding_center.runtime.configuration import _make_poisson_options, _runtime_config


PRESET = "euler_vortex_gas_p6_50k_dt001_t50_raw_cuda_bsr"


@pytest.mark.parametrize("preset", sorted(
    key for key, config in PRESETS.items() if config.case == "euler_vortex_gas"
))
@pytest.mark.parametrize("override_atol", [None, 2.0e-10])
def test_disk_vortex_gas_keeps_previous_fast_poisson_settings(preset, override_atol) -> None:
    from hdgfem.linalg.multigrid.policy import face_hp_mg_preconditioner_parameters

    config = preset_by_key(preset)
    assert config.poisson_solver_atol == 1.0e-12
    assert config.poisson_retry_policy == "none"
    assert Path(config.poisson_amgx_config_path).name == "diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive_abs.json"
    arguments = ["--preset", preset]
    if override_atol is not None:
        arguments += ["--poisson-solver-atol", str(override_atol)]
    runtime = _runtime_config(config, build_parser().parse_args(arguments))
    options = _make_poisson_options(runtime)

    assert options.solver == "fb-hp-mg-pcg"
    assert options.solver_rtol == 1.0e-11
    assert options.solver_atol == (1.0e-12 if override_atol is None else override_atol)
    assert options.maxiter is None  # standard policy supplies the old 500 cap
    assert options.amgx_retry_attempts is None
    assert options.fb_hp_mg_preconditioner_policy == "standard"
    policy = face_hp_mg_preconditioner_parameters(options.fb_hp_mg_preconditioner_policy)
    assert policy["schedule"] == "direct-to-zero"
    assert policy["chebyshev_order"] == 2
    assert policy["presweeps"] == policy["postsweeps"] == 1
    coarse = policy["coarse_config"]["solver"]
    assert coarse["presweeps"] == coarse["postsweeps"] == 1
    assert coarse["coarsest_sweeps"] == 2

    hybrid = options.amgx_config["solver"]
    assert hybrid["solver"] == "PCGF"
    assert hybrid["max_iters"] == 500
    assert hybrid["tolerance"] == options.solver_atol
    assert hybrid["preconditioner"]["smoother"]["chebyshev_polynomial_order"] == 2
    # Returning to the disk Poisson tuning must not change transport recovery.
    assert runtime.transport_retry_policy == "amgx-robust"
    assert runtime.transport_solver == "amgx"
    assert runtime.transport_raw_matrix_format == "bsr"


def test_vortex_gas_is_signed_reproducible_and_domain_filling() -> None:
    case = euler_vortex_gas()
    grid = np.linspace(-1, 1, 501)
    x, y = np.meshgrid(grid, grid)
    values = case.initial_density(x, y)
    np.testing.assert_array_equal(values, euler_vortex_gas().initial_density(x, y))
    np.testing.assert_allclose(euler_vortex_gas(amplitude=8).initial_density(x, y), 2*values)
    assert not np.allclose(euler_vortex_gas(seed=18).initial_density(x, y), values)
    inside = x*x+y*y < 1
    extrema = inside & (
        ((values == maximum_filter(values, size=5)) & (values > 1)) |
        ((values == minimum_filter(values, size=5)) & (values < -1))
    )
    # Overlapping blobs are not necessarily distinct coherent vortices; count
    # strong resolved extrema to verify that the default really populates the disk.
    assert np.count_nonzero(extrema) > 250
    theta = np.arctan2(y, x)
    for sector in range(8):
        mask = inside & (theta >= -np.pi+sector*np.pi/4) & (theta < -np.pi+(sector+1)*np.pi/4)
        assert np.count_nonzero(extrema & mask) > 15
        assert values[mask].min() < -2 and values[mask].max() > 2
    assert case.density_is_vorticity
    assert case.density_transport_boundary_mode == "zero-flux"
    assert case.density_boundary_at(1) is None
    np.testing.assert_array_equal(case.potential_boundary_at(1)(x, y), 0)
    assert case.initial_density(np.zeros((3, 1)), np.zeros((1, 4))).shape == (3, 4)
    assert case.initial_density(0, 0).shape == ()


@pytest.mark.parametrize("center_radius", [0.96, 0.5])
def test_vortex_gas_circulation_cancels_on_the_disk(center_radius) -> None:
    # Independent polar quadrature checks the physical disk, not the plane.
    nodes, weights = np.polynomial.legendre.leggauss(320)
    radius = 0.5*(nodes+1)
    theta = np.linspace(0, 2*np.pi, 1024, endpoint=False)
    values = euler_vortex_gas(center_radius=center_radius).initial_density(radius[:, None]*np.cos(theta), radius[:, None]*np.sin(theta))
    circulation = np.sum(values * (0.5*weights*radius)[:, None]) * 2*np.pi/len(theta)
    assert abs(circulation) < 1.e-10


@pytest.mark.parametrize("params", [
    {"counts": [3], "sigmas": [0.02]},
    {"counts": [4], "sigmas": []},
    {"sigmas": [0, 0.016, 0.032, 0.064]},
    {"amplitude": -1}, {"center_radius": 1.1}, {"seed": -1},
])
def test_vortex_gas_rejects_invalid_parameters(params) -> None:
    with pytest.raises(ValueError):
        euler_vortex_gas(**params)


def test_vortex_gas_response_file() -> None:
    config = preset_by_key(PRESET)
    assert config.minimum_triangles == 50_000 and config.order == 6
    assert config.dt == 0.01 and config.num_steps == 5000
    assert config.time_scheme == "si-euler"
    assert config.transport_retry_policy == "amgx-robust"
    assert sum(config.case_params["counts"]) == 360
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.guiding_center.run_guiding_center_cases",
         f"@run_configs/guiding_center/{PRESET}.args", "--num-steps", "500", "--dry-run"],
        cwd=root, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "minimum_triangles: 50000" in completed.stdout
    assert "num_steps: 500" in completed.stdout


def test_localized_preset_has_small_initial_wall_and_outer_rim_values() -> None:
    config = preset_by_key("euler_vortex_gas_localized_p6_50k_dt001_t50_raw_cuda_bsr")
    original = preset_by_key(PRESET)
    assert config.case_params == {**original.case_params, "center_radius": 0.5}
    assert config.diagnostics_prefix != original.diagnostics_prefix
    assert config.dt == original.dt and config.order == original.order
    assert config.transport_direct_fallback == "none"
    case = euler_vortex_gas(**config.case_params)
    theta = np.linspace(0, 2*np.pi, 2048, endpoint=False)
    assert np.max(np.abs(case.initial_density(np.cos(theta), np.sin(theta)))) < 1.e-10
    radius = np.linspace(0.85, 1, 20)[:, None]
    rim = case.initial_density(radius*np.cos(theta), radius*np.sin(theta))
    assert np.max(np.abs(rim)) < 1.e-4
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.guiding_center.run_guiding_center_cases",
         "@run_configs/guiding_center/euler_vortex_gas_localized_p6_50k_dt001_t50_raw_cuda_bsr.args",
         "--transport-direct-fallback", "cusolver-qr", "--dry-run"],
        cwd=root, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "'center_radius': 0.5" in completed.stdout
    assert "transport_direct_fallback: 'cusolver-qr'" in completed.stdout
