from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hdgfem.core.geometry import DiskDomain
from hdgfem.cases.profiles import GaussianBlobField, sample_gaussian_blob_field
from scripts.guiding_center.cases.guiding_center_cases import (
    case_definition_by_key,
    positive_turbulence,
)
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime.arguments import build_parser
from scripts.guiding_center.runtime.configuration import _make_poisson_options, _runtime_config
from scripts.guiding_center.runtime.labels import run_label


PRESET = "positive_turbulence_imex_ark3_p6_h008_dt0005_t50_raw_cuda_bsr"
BDF2_PRESETS = {
    "positive_turbulence_si_bdf2_p6_h0068_dt0005_t50_raw_cuda_bsr": {
        "geometry": "disc", "mesh_size": 0.0068, "minimum_triangles": 150_000,
        "count": 360,
    },
    "positive_turbulence_iter_si_bdf2_p6_h014_dt0005_t50_raw_cuda_bsr": {
        "geometry": "iter", "mesh_size": 0.014, "minimum_triangles": 300_000,
        "count": 11_520,
    },
}


def test_positive_turbulence_is_reproducible_nonnegative_and_wall_separated() -> None:
    case = positive_turbulence()
    field = case.initial_density

    assert isinstance(field, GaussianBlobField)
    assert len(field.centers) == 360
    assert np.all(field.strengths > 0.0)
    np.testing.assert_array_equal(field.centers, positive_turbulence().initial_density.centers)
    np.testing.assert_array_equal(field.strengths, positive_turbulence().initial_density.strengths)
    assert not np.array_equal(field.centers, positive_turbulence(seed=18).initial_density.centers)

    doubled = positive_turbulence(amplitude=8.0).initial_density
    np.testing.assert_array_equal(doubled.centers, field.centers)
    np.testing.assert_allclose(doubled.strengths, 2.0 * field.strengths)

    wall_gap = case.parameters["wall_gap"]
    support_outer_radius = np.linalg.norm(field.centers, axis=1) + field.cutoff * field.sigmas
    assert np.all(support_outer_radius < 1.0 - wall_gap)

    theta = np.linspace(0.0, 2.0 * np.pi, 2048, endpoint=False)
    radius = np.linspace(1.0 - wall_gap / 2.0, 1.0, 8)[:, None]
    rim = field(radius * np.cos(theta), radius * np.sin(theta))
    np.testing.assert_array_equal(rim, 0.0)

    rng = np.random.default_rng(4)
    points = rng.uniform(-1.0, 1.0, size=(20_000, 2))
    points = points[np.sum(points * points, axis=1) <= 1.0]
    values = field(points[:, 0], points[:, 1])
    assert np.all(values >= 0.0)
    center_values = field(field.centers[:, 0], field.centers[:, 1])
    assert np.max(center_values) >= np.max(field.strengths) > 4.0

    assert not case.density_is_vorticity
    assert case.density_transport_boundary_mode == "zero-flux"
    assert case.density_boundary_at(1.0) is None
    np.testing.assert_array_equal(case.potential_boundary_at(1.0)(points[:, 0], points[:, 1]), 0.0)
    assert field(0.0, 0.0).shape == ()


def test_shared_blob_sampler_supports_positive_odd_populations_and_disk_clearance() -> None:
    domain = DiskDomain(radius=2.0, center=(1.0, -1.0))
    field = sample_gaussian_blob_field(
        domain,
        counts=(3,),
        sigmas=(0.1,),
        amplitude=2.0,
        seed=9,
        cutoff=5.0,
        wall_clearance=0.25,
        strength_mode="positive",
    )

    assert domain.area == pytest.approx(4.0 * np.pi)
    assert np.all(field.strengths > 0.0)
    assert np.all(domain.contains(field.centers))
    assert np.all(domain.boundary_distance(field.centers) > 0.75)
    wall = np.column_stack((1.0 + 2.0 * np.cos(np.arange(32) * np.pi / 16.0),
                            -1.0 + 2.0 * np.sin(np.arange(32) * np.pi / 16.0)))
    np.testing.assert_array_equal(field(wall[:, 0], wall[:, 1]), 0.0)


def test_positive_turbulence_samples_the_requested_shaped_domain(monkeypatch) -> None:
    domain = DiskDomain(radius=2.0, center=(1.0, -1.0))
    calls = []

    def shaped_domain(kind, **params):
        calls.append((kind, params))
        return domain

    monkeypatch.setattr("hdgfem.core.geometry.shaped_domain", shaped_domain)
    case = positive_turbulence(
        geometry="iter",
        geometry_params={"boundary_points": 512},
        counts=(5,),
        sigmas=(0.1,),
        cutoff=3.0,
        wall_gap=0.2,
    )
    field = case.initial_density

    assert calls == [("iter", {"boundary_points": 512})]
    assert case.default_domain == "iter"
    assert case.parameters["geometry_name"] == "iter"
    assert case.parameters["geometry"] == {"boundary_points": 512}
    assert np.all(domain.contains(field.centers))
    assert np.all(domain.boundary_distance(field.centers) > 0.5)
    assert np.all(field.strengths > 0.0)


@pytest.mark.parametrize("params", [
    {"counts": [0], "sigmas": [0.02]},
    {"counts": [4], "sigmas": []},
    {"counts": [4], "sigmas": [0.0]},
    {"amplitude": -1.0},
    {"seed": -1},
    {"cutoff": 0.0},
    {"wall_gap": -0.1},
    {"wall_gap": 0.5},
    {"geometry": "triangle"},
    {"geometry_params": {"boundary_points": 64}},
])
def test_positive_turbulence_rejects_invalid_parameters(params) -> None:
    with pytest.raises(ValueError):
        positive_turbulence(**params)


def test_positive_turbulence_preset_enables_every_stage_positivity_audit() -> None:
    config = preset_by_key(PRESET)
    assert config.case == "positive_turbulence"
    assert config.time_scheme == "imex-ark3"
    assert config.order == 6 and config.mesh_size == pytest.approx(0.008)
    assert config.dt == pytest.approx(0.005) and config.num_steps == 10_000
    assert config.initial_projection_quad_1d == 16
    assert config.positivity_diagnostics
    assert config.case_params["wall_gap"] == pytest.approx(0.04)
    assert sum(config.case_params["counts"]) == 360
    assert run_label(config) == "Positive guiding-center turbulence | IMEX-ARK3"

    case = case_definition_by_key(config.case).build(**config.case_params)
    assert case.key == config.case and not case.density_is_vorticity

    root = Path(__file__).resolve().parents[1]
    response = root / "run_configs" / "guiding_center" / f"{PRESET}.args"
    assert response.is_file()
    args = build_parser().parse_args([f"@{response}", "--dry-run"])
    assert args.preset == PRESET and args.dry_run


@pytest.mark.parametrize("preset, expected", BDF2_PRESETS.items())
def test_positive_turbulence_bdf2_presets_and_response_files(preset, expected) -> None:
    config = preset_by_key(preset)

    assert config.case == "positive_turbulence"
    assert config.time_scheme == "si-bdf2"
    assert config.order == 6
    assert config.mesh_size == pytest.approx(expected["mesh_size"])
    assert config.minimum_triangles == expected["minimum_triangles"]
    assert config.dt == pytest.approx(0.005) and config.num_steps == 10_000
    assert config.positivity_diagnostics and config.diagnostics_every == 10
    assert config.plot_every == 100 and config.initial_projection_quad_1d == 16
    assert config.poisson_retry_policy == "amgx-robust"
    assert config.poisson_fb_hp_mg_preconditioner_policy == (
        "fast" if expected["geometry"] == "iter" else "robust"
    )
    assert sum(config.case_params["counts"]) == expected["count"]
    assert config.case_params.get("geometry", "disc") == expected["geometry"]

    suffix = "" if expected["geometry"] == "disc" else " (ITER)"
    assert run_label(config) == f"Positive guiding-center turbulence{suffix} | SI BDF2"

    root = Path(__file__).resolve().parents[1]
    response = root / "run_configs" / "guiding_center" / f"{preset}.args"
    assert response.is_file()
    args = build_parser().parse_args([f"@{response}", "--dry-run"])
    assert args.preset == preset and args.dry_run


@pytest.mark.parametrize("preset", tuple(BDF2_PRESETS))
def test_positive_turbulence_poisson_retry_order_and_conditioning(preset) -> None:
    options = _make_poisson_options(preset_by_key(preset))

    assert options.fb_hp_mg_preconditioner_policy == (
        "fast" if BDF2_PRESETS[preset]["geometry"] == "iter" else "robust"
    )
    primary = options.amgx_config["solver"]
    primary_amg = primary["preconditioner"]
    assert primary["solver"] == "PCGF"
    assert primary_amg["classical_bsr_hierarchy"] == "scalar_expand"
    assert primary_amg["presweeps"] == primary_amg["postsweeps"] == 2
    smoother = primary_amg["smoother"]
    # CHEBYSHEV_POLY is scalar-only and does not use its nested L1 config.
    # Hybrid AMG still smooths the original p+1 face blocks on its fine level.
    assert smoother["solver"] == "CHEBYSHEV"
    assert smoother["chebyshev_polynomial_order"] == 4
    assert smoother["chebyshev_lambda_estimate_mode"] == 2
    assert smoother["preconditioner"]["solver"] == "JACOBI_L1"
    assert smoother["preconditioner"]["jacobi_l1_scalar_rows_for_blocks"] == 1

    attempts = options.amgx_retry_attempts
    assert tuple(attempt["label"] for attempt in attempts) == (
        "hybrid-pcgf-zero",
        "pure-csr-pcgf-zero",
        "pure-csr-pcgf-correction-1",
        "pure-csr-pcgf-correction-2",
        "pure-csr-fgmres-dilu-last",
    )
    assert attempts[0]["reuse_primary_solver"]
    assert all(
        attempt["config"]["solver"]["solver"] == "PCGF"
        for attempt in attempts[:-1]
    )
    assert all(attempt.get("scalarize_bsr", False) for attempt in attempts[1:])
    scalar_smoother = attempts[1]["config"]["solver"]["preconditioner"]["smoother"]
    assert scalar_smoother["solver"] == "MULTICOLOR_GS"
    assert scalar_smoother["symmetric_GS"] == 1
    assert all(
        attempts[index].get("residual_correction", False)
        for index in (2, 3)
    )
    terminal = attempts[-1]["config"]["solver"]
    assert terminal["solver"] == "FGMRES"
    assert terminal["preconditioner"]["solver"] == "MULTICOLOR_DILU"


def test_robust_native_poisson_preconditioner_remains_symmetric() -> None:
    from hdgfem.linalg.multigrid.policy import face_hp_mg_preconditioner_parameters

    policy = face_hp_mg_preconditioner_parameters("robust")
    assert policy["schedule"] == "halve"
    assert policy["chebyshev_order"] == 4
    assert policy["presweeps"] == policy["postsweeps"] == 2
    coarse = policy["coarse_config"]["solver"]
    assert coarse["presweeps"] == coarse["postsweeps"] == 2
    assert coarse["coarsest_sweeps"] == 4
    assert coarse["error_scaling"] == 0


@pytest.mark.parametrize("preset", [
    "euler_horseshoe_gas_imex_ark3_p6_150k_t50",
    "euler_iter_gas_imex_ark3_p6_300k_t50",
    "euler_iter_gas_si_bdf2_p6_300k_t50",
    "euler_pacman_gas_imex_ark3_p6_150k_t50",
    "euler_star_hole_si_bdf2_p6_h008_dt005_t50_raw_cuda_bsr",
])
def test_non_disk_vortex_gas_retains_robust_poisson_fallback_ladder(preset) -> None:
    config = preset_by_key(preset)
    options = _make_poisson_options(config)

    assert config.poisson_retry_policy == "amgx-robust"
    assert options.solver_atol == 1.0e-10
    assert options.fb_hp_mg_preconditioner_policy == "robust"
    assert options.amgx_config["solver"]["solver"] == "PCGF"
    assert tuple(
        attempt["label"] for attempt in options.amgx_retry_attempts
    ) == (
        "hybrid-pcgf-zero",
        "pure-csr-pcgf-zero",
        "pure-csr-pcgf-correction-1",
        "pure-csr-pcgf-correction-2",
        "pure-csr-fgmres-dilu-last",
    )


@pytest.mark.parametrize('preset', [
    *BDF2_PRESETS,
    PRESET,
])
@pytest.mark.parametrize('override', [None, 2.0e-10])
def test_turbulence_poisson_absolute_floor_reaches_all_solver_attempts(preset, override):
    arguments = ['--preset', preset]
    if override is not None:
        arguments += ['--poisson-solver-atol', str(override)]
    runtime = _runtime_config(preset_by_key(preset), build_parser().parse_args(arguments))
    options = _make_poisson_options(runtime)
    expected = 1.0e-10 if override is None else override

    assert options.solver_atol == expected
    assert options.solver_rtol == 1.0e-11
    assert max(options.solver_atol, options.solver_rtol * 0.4097018) == expected
    configs = [options.amgx_config] + [a['config'] for a in options.amgx_retry_attempts]
    for config in configs:
        assert config['solver']['convergence'] == 'ABSOLUTE'
        assert config['solver']['tolerance'] == expected
