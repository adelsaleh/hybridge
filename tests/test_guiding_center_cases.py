from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.guiding_center.cases.guiding_center_cases import (
    CASE_DEFINITIONS,
    case_definition_by_key,
    rho_eq_annular_band,
    rho_eq_gaussian_annulus,
    rho_eq_super_gaussian_annulus,
)
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from hdgfem.core.field_ops import project_callable_to_trace
from scripts.guiding_center.time_schemes.stage_support import _fixed_operator_trace_predictor
from scripts.guiding_center.runtime.configuration import _make_transport_options, _poisson_postprocess_overrides, _solver_verbosity, _validate_config
from scripts.guiding_center.runtime.reporting import _print_linear_step_summary, _print_step_summary
from scripts.guiding_center.runtime.runner import run_guiding_center_case


def test_terminal_log_tee_captures_python_and_native_streams(
        tmp_path: Path,
) -> None:
    log_path = tmp_path / "guiding_center.log"
    code = "\n".join(
        [
            "import os",
            "import sys",
            "from scripts.guiding_center.runtime.terminal_log import _TerminalLogTee",
            "with _TerminalLogTee(sys.argv[1]):",
            "    print('python stdout', flush=True)",
            "    print('python stderr', file=sys.stderr, flush=True)",
            "    os.write(1, b'native stdout\\n')",
            "    os.write(2, b'native stderr\\n')",
        ]
    )

    completed = subprocess.run(
        [sys.executable, "-c", code, str(log_path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "python stdout" in completed.stdout
    assert "native stdout" in completed.stdout
    assert "python stderr" in completed.stderr
    assert "native stderr" in completed.stderr
    log_text = log_path.read_text(encoding="utf-8")
    assert "python stdout" in log_text
    assert "native stdout" in log_text
    assert "python stderr" in log_text
    assert "native stderr" in log_text


def test_00_legacy_gaussian_annulus_cli_smoke(tmp_path: Path) -> None:
    pytest.importorskip("gmsh")
    script = Path("scripts/guiding_center/run_guiding_center_cases.py")
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--preset",
            "diocotron_gaussian_annulus_host_smoke",
            "--num-steps",
            "1",
            "--mesh-size",
            "0.3",
            "--diagnostics-dir",
            str(tmp_path),
            "--diagnostics-prefix",
            "diocotron_gaussian_smoke_test",
            "--quiet",
            "--plot-every",
            "0",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    csv_path = tmp_path / "diocotron_gaussian_smoke_test.csv"
    jsonl_path = tmp_path / "diocotron_gaussian_smoke_test.jsonl"
    timings_csv_path = tmp_path / "diocotron_gaussian_smoke_test_timings.csv"
    timings_jsonl_path = tmp_path / "diocotron_gaussian_smoke_test_timings.jsonl"
    terminal_log_path = tmp_path / "diocotron_gaussian_smoke_test.log"
    assert csv_path.exists()
    assert jsonl_path.exists()
    assert timings_csv_path.exists()
    assert timings_jsonl_path.exists()
    assert terminal_log_path.exists()
    timing_rows = [json.loads(line) for line in timings_jsonl_path.read_text().splitlines()]
    assert [row["step"] for row in timing_rows] == [0, 1]
    assert timing_rows[0]["phase"] == "initial"
    assert timing_rows[0]["first_poisson_wall_time"] > 0.0
    assert timing_rows[0]["initial_poisson_wall_time"] > 0.0
    assert timing_rows[0]["poisson_time_total"] >= timing_rows[0]["initial_poisson_wall_time"]
    assert "poisson_time_total" in timing_rows[-1]
    assert "transport_time_total" in timing_rows[-1]

    final = json.loads(jsonl_path.read_text().strip().splitlines()[-1])
    for key in ("mass_relative_drift", "q_l2_relative_drift", "rho_min", "rho_max", "transport_solver_residual"):
        assert final[key] is not None
        assert math.isfinite(final[key])
    assert abs(final["mass_relative_drift"]) < 1.0e-3
    assert final["transport_solver_residual"] < 1.0e-4


def test_guiding_center_case_registry_has_legacy_gaussian_new_diocotron_and_rho_helm() -> None:
    assert tuple(CASE_DEFINITIONS) == (
        "diocotron_gaussian_annulus", "diocotron_k", "euler_vortex_gas",
        "positive_turbulence", "euler_star_vortex_gas", "euler_shaped_vortex_gas",
        "spiral_sheet", "rho_helm_wave",
    )


def test_fixed_operator_trace_predictor_uses_constant_linear_then_quadratic_history() -> None:
    current = np.array([3.0, -1.0])
    previous = np.array([2.0, -2.0])
    older = np.array([1.5, -4.0])

    constant, order = _fixed_operator_trace_predictor(current)
    assert constant is current
    assert order == 0

    linear, order = _fixed_operator_trace_predictor(current, previous)
    np.testing.assert_allclose(linear, 2.0 * current - previous)
    assert order == 1

    quadratic, order = _fixed_operator_trace_predictor(current, previous, older)
    np.testing.assert_allclose(quadratic, 3.0 * current - 3.0 * previous + older)
    assert order == 2


def test_guiding_center_case_factories_vectorize_on_arrays() -> None:
    x = np.array([[0.1, 0.2, -0.3], [0.4, -0.5, 0.0]])
    y = np.array([[0.0, 0.3, -0.2], [0.1, 0.2, -0.4]])

    for key in CASE_DEFINITIONS:
        case = case_definition_by_key(key).build()
        rho0 = case.initial_density(x, y)
        phi_boundary = case.potential_boundary_at(0.125)(x, y)
        assert rho0.shape == x.shape
        assert phi_boundary.shape == x.shape
        if case.density_boundary is not None:
            rho_boundary = case.density_boundary_at(0.125)(x, y)
            assert rho_boundary.shape == x.shape


def test_legacy_gaussian_annulus_diocotron_defaults() -> None:
    case = case_definition_by_key("diocotron_gaussian_annulus").build()
    assert case.default_domain == "disc"
    assert case.density_transport_boundary_mode == "zero-flux"
    assert case.parameters["r0"] == pytest.approx(0.45)
    assert case.parameters["sigma"] == pytest.approx(0.03)

    rho_at_peak = case.initial_density(np.array([0.45]), np.array([0.0]))
    rho_far = case.initial_density(np.array([0.2]), np.array([0.0]))
    wrapper_value = rho_eq_gaussian_annulus(np.array([0.45]), np.array([0.0]))
    assert rho_at_peak[0] == pytest.approx(1.0 + case.parameters["eps"])
    assert wrapper_value[0] == pytest.approx(1.0)
    assert rho_far[0] < 1.0e-10


def test_sharp_annular_band_diocotron_k_defaults() -> None:
    case = case_definition_by_key("diocotron_k").build()
    assert case.default_domain == "disc"
    assert case.density_transport_boundary_mode == "zero-flux"
    assert case.parameters["s_minus"] == pytest.approx(0.79)
    assert case.parameters["s_plus"] == pytest.approx(0.80)
    assert case.parameters["edge_width"] == pytest.approx(0.0)

    rho_on_band = case.initial_density(np.array([0.795]), np.array([0.0]))
    rho_inside_hole = case.initial_density(np.array([0.2]), np.array([0.0]))
    wrapper_value = rho_eq_annular_band(np.array([0.795]), np.array([0.0]))
    assert rho_on_band[0] == pytest.approx(1.0 + case.parameters["epsilon"])
    assert wrapper_value[0] == pytest.approx(1.0)
    assert rho_inside_hole[0] == pytest.approx(0.0)


def test_annular_band_supports_smooth_edges() -> None:
    case = case_definition_by_key("diocotron_k").build(
        s_minus=0.78,
        s_plus=0.82,
        edge_width=0.005,
    )
    center = case.equilibrium_density(np.array([0.80]), np.array([0.0]))
    inner_edge = case.equilibrium_density(np.array([0.78]), np.array([0.0]))
    outside = case.equilibrium_density(np.array([0.74]), np.array([0.0]))

    assert center[0] > 0.99
    assert inner_edge[0] == pytest.approx(0.5, abs=1.0e-3)
    assert outside[0] < 1.0e-6


def test_diocotron_k_supports_super_gaussian_radial_power() -> None:
    case = case_definition_by_key("diocotron_k").build(
        s_bar=0.80,
        s_d=0.02,
        p=4,
    )
    center = case.equilibrium_density(np.array([0.80]), np.array([0.0]))
    scale_radius = case.equilibrium_density(np.array([0.82]), np.array([0.0]))
    outside = case.equilibrium_density(np.array([0.84]), np.array([0.0]))
    wrapper_value = rho_eq_super_gaussian_annulus(
        np.array([0.82]),
        np.array([0.0]),
        s_bar=0.80,
        s_d=0.02,
        p=4,
    )

    assert case.parameters["p"] == pytest.approx(4.0)
    assert center[0] == pytest.approx(1.0)
    assert scale_radius[0] == pytest.approx(np.exp(-1.0))
    assert wrapper_value[0] == pytest.approx(np.exp(-1.0))
    assert outside[0] == pytest.approx(np.exp(-16.0))



def test_gaussian_annulus_full_raw_cuda_t50_preset_uses_legacy_case() -> None:
    config = preset_by_key("diocotron_gaussian_annulus_k3_p6_dt01_t50_full_raw_cuda_amgx")

    assert config.case == "diocotron_gaussian_annulus"
    assert config.case_params == {"k": 3, "eps": 0.05, "r0": 0.45, "sigma": 0.03}
    assert config.order == 6
    assert config.dt == pytest.approx(0.1)
    assert config.num_steps == 500
    assert config.poisson_assembly_backend == "raw-cuda"
    assert config.transport_assembly_backend == "raw-cuda"
    assert config.transport_materialize_host_system is False
    assert config.transport_materialize_host_solution is False


def test_gaussian_annulus_k3_p6_numba_ilu_upwind_preset() -> None:
    config = preset_by_key("diocotron_gaussian_annulus_k3_p6_30k_numba_ilu_upwind")

    assert config.case == "diocotron_gaussian_annulus"
    assert config.case_params == {"k": 3, "eps": 0.05, "r0": 0.45, "sigma": 0.03}
    assert config.order == 6
    assert config.minimum_triangles == 30_000
    assert config.poisson_assembly_backend == "numba"
    assert config.poisson_local_backend == "numba"
    assert config.poisson_solver == "BICGSTAB"
    assert config.poisson_solver_rtol == pytest.approx(1.0e-11)
    assert config.poisson_preconditioner == "ilu"
    assert config.poisson_ilu_fill_factor == pytest.approx(35.0)
    assert config.poisson_ilu_permc_spec == "NATURAL"
    assert config.poisson_hdg_postprocess == "none"
    assert config.transport_assembly_backend == "numba"
    assert config.transport_solver == "BICGSTAB"
    assert config.transport_preconditioner == "ilu"
    assert config.transport_trace_ordering == "upwind-scc"
    assert config.transport_ilu_permc_spec == "COLAMD"
    assert config.transport_boundary_mode == "zero-flux"
    assert config.transport_initial_guess == "initial-density-trace"


def test_gaussian_annulus_k3_p6_numba_pypardiso_lu_upwind_preset() -> None:
    config = preset_by_key(
        "diocotron_gaussian_annulus_k3_p6_30k_numba_pypardiso_lu_upwind"
    )

    assert config.case == "diocotron_gaussian_annulus"
    assert config.case_params == {"k": 3, "eps": 0.05, "r0": 0.45, "sigma": 0.03}
    assert config.order == 6
    assert config.minimum_triangles == 30_000
    assert config.poisson_assembly_backend == "numba"
    assert config.poisson_local_backend == "numba"
    assert config.poisson_solver == "pypardiso"
    assert config.poisson_preconditioner is None
    assert config.poisson_scale_system is False
    assert config.poisson_hdg_postprocess == "none"
    assert config.transport_assembly_backend == "numba"
    assert config.transport_solver == "BICGSTAB"
    assert config.transport_preconditioner == "ilu"
    assert config.transport_trace_ordering == "upwind-scc"
    assert config.transport_ilu_permc_spec == "COLAMD"
    assert config.transport_boundary_mode == "zero-flux"
    assert config.transport_initial_guess == "initial-density-trace"


@pytest.mark.parametrize(
    ("preset_key", "poisson_solver", "poisson_preconditioner"),
    (
        (
            "diocotron_gaussian_annulus_k3_p6_50k_numba_medium_ilu_upwind",
            "BICGSTAB",
            "ilu",
        ),
        (
            "diocotron_gaussian_annulus_k3_p6_50k_numba_pypardiso_medium_ilu_upwind",
            "pypardiso",
            None,
        ),
    ),
)
def test_gaussian_annulus_k3_p6_50k_matched_comparison_presets(
    preset_key: str,
    poisson_solver: str,
    poisson_preconditioner: str | None,
) -> None:
    config = preset_by_key(preset_key)

    assert config.case_params == {"k": 3, "eps": 0.05, "r0": 0.45, "sigma": 0.03}
    assert config.mesh_size == pytest.approx(0.012)
    assert config.minimum_triangles == 50_000
    assert config.order == 6
    assert config.poisson_assembly_backend == "numba"
    assert config.poisson_local_backend == "numba"
    assert config.poisson_solver == poisson_solver
    assert config.poisson_preconditioner == poisson_preconditioner
    assert config.poisson_hdg_postprocess == "none"
    assert config.transport_assembly_backend == "numba"
    assert config.transport_solver == "BICGSTAB"
    assert config.transport_preconditioner == "ilu"
    assert config.transport_ilu_drop_tol == pytest.approx(1.0e-5)
    assert config.transport_ilu_fill_factor == pytest.approx(5.0)
    assert config.transport_trace_ordering == "upwind-scc"
    assert config.transport_ilu_permc_spec == "COLAMD"
    assert config.transport_reuse_first_preconditioner is False
    assert config.transport_initial_guess == "initial-density-trace"


@pytest.mark.parametrize(
    ("preset_key", "ordering", "permc_spec", "reuse"),
    (
        (
            "diocotron_gaussian_annulus_k3_p6_30k_pypardiso_upwind_colamd",
            "upwind-scc",
            "COLAMD",
            False,
        ),
        (
            "diocotron_gaussian_annulus_k3_p6_30k_pypardiso_cached_adv_natural",
            "none",
            "NATURAL",
            True,
        ),
        (
            "diocotron_gaussian_annulus_k3_p6_30k_pypardiso_cached_adv_colamd",
            "none",
            "COLAMD",
            True,
        ),
        (
            "diocotron_gaussian_annulus_k3_p6_30k_pypardiso_unordered_colamd",
            "none",
            "COLAMD",
            False,
        ),
        (
            "diocotron_gaussian_annulus_k3_p6_30k_pypardiso_upwind_colamd_weak_ilu",
            "upwind-scc",
            "COLAMD",
            False,
        ),
    ),
)
def test_pypardiso_transport_cache_comparison_presets(
    preset_key: str,
    ordering: str,
    permc_spec: str,
    reuse: bool,
) -> None:
    config = preset_by_key(preset_key)

    assert config.poisson_solver == "pypardiso"
    assert config.transport_trace_ordering == ordering
    assert config.transport_ilu_permc_spec == permc_spec
    assert config.transport_reuse_first_preconditioner is reuse
    assert config.transport_initial_guess == "initial-density-trace"


def test_diocotron_full_raw_cuda_t50_preset_is_device_csr_long_run() -> None:
    config = preset_by_key("diocotron_k3_p6_dt01_t50_full_raw_cuda_amgx")

    assert config.case == "diocotron_k"
    assert config.case_params["k"] == 3
    assert config.order == 6
    assert config.mesh_size == pytest.approx(0.02)
    assert config.dt == pytest.approx(0.1)
    assert config.num_steps * config.dt == pytest.approx(50.0)
    assert config.poisson_assembly_backend == "raw-cuda"
    assert config.poisson_solver == "fb-hp-mg-pcg"
    assert config.poisson_trace_basis == "legendre-modal"
    assert config.poisson_raw_matrix_format == "bsr"
    assert config.poisson_cache_local_factors == "schur-cholesky"
    assert config.transport_assembly_backend == "raw-cuda"
    assert config.transport_solver == "amgx"
    assert config.transport_trace_basis == "legacy-lagrange"
    assert config.transport_raw_matrix_format == "bsr"
    assert config.transport_initial_guess == "initial-density-trace"
    assert config.transport_materialize_host_system is False
    assert config.transport_materialize_host_solution is False


def test_guiding_center_bsr_preset_preserves_current_amgx_configuration() -> None:
    csr = preset_by_key("diocotron_k3_raw_cuda_amgx")
    bsr = preset_by_key("diocotron_k3_raw_cuda_amgx_bsr")
    amgx_config = json.loads(Path(bsr.transport_amgx_config_path).read_text())

    assert bsr.transport_raw_matrix_format == "bsr"
    assert bsr.transport_amgx_config_path == csr.transport_amgx_config_path
    assert bsr.transport_solver == csr.transport_solver == "amgx"
    assert bsr.transport_scale_system == csr.transport_scale_system
    assert bsr.diagnostics_prefix == f"{csr.diagnostics_prefix}_bsr"
    assert amgx_config["solver"]["solver"] == "BICGSTAB"
    assert amgx_config["solver"]["bsr_spmv_backend"] == "cusparse_generic"

def test_diocotron_k100_stress_preset_uses_resolved_single_mode_band() -> None:
    config = preset_by_key("diocotron_k100_p6_dt01_t50_full_raw_cuda_amgx")

    assert config.case == "diocotron_k"
    assert config.case_params["k"] == 100
    assert config.case_params["eps"] == pytest.approx(1.0)
    assert config.case_params["s_bar"] == pytest.approx(0.80)
    assert config.case_params["s_d"] == pytest.approx(0.04)
    assert config.case_params["p"] == pytest.approx(6)
    assert config.order == 6
    assert config.mesh_size == pytest.approx(0.006)
    assert config.dt == pytest.approx(0.1)
    assert config.num_steps == 500
    assert config.poisson_assembly_backend == "raw-cuda"
    assert config.transport_assembly_backend == "raw-cuda"
    assert config.transport_solver_rtol == pytest.approx(1.0e-11)
    assert config.transport_initial_guess == "initial-density-trace"
    assert config.poisson_solver == "fb-hp-mg-pcg"
    assert config.poisson_raw_matrix_format == "bsr"
    assert config.poisson_cache_local_factors == "schur-cholesky"
    assert config.transport_raw_matrix_format == "bsr"


def test_guiding_center_accepts_cupy_schur_cholesky_poisson_configuration() -> None:
    base = preset_by_key("diocotron_k100_p6_dt01_t50_full_raw_cuda_amgx")
    config = replace(
        base,
        poisson_assembly_backend="cupy",
        poisson_solver="amgx",
        poisson_cache_local_factors="schur-cholesky",
        poisson_hdg_postprocess="none",
    )

    _validate_config(config)

    with pytest.raises(ValueError, match="requires 'schur-cholesky'"):
        _validate_config(replace(config, poisson_cache_local_factors="schur-lu"))





def test_diocotron_k100_has_one_hundred_angular_maxima_and_only_mode_100() -> None:
    case = case_definition_by_key("diocotron_k").build(
        k=100,
        eps=1.0,
        s_bar=0.8,
        s_d=0.04,
        p=6,
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    x = 0.8 * np.cos(theta)
    y = 0.8 * np.sin(theta)
    rho = case.initial_density(x, y)
    rho_eq = case.equilibrium_density(x, y)
    angular_factor = rho / rho_eq
    spectrum = np.fft.rfft(angular_factor) / theta.size
    maxima = np.count_nonzero((angular_factor > np.roll(angular_factor, 1)) & (angular_factor > np.roll(angular_factor, -1)))

    assert maxima == 100
    assert abs(spectrum[0]) == pytest.approx(1.0, abs=1.0e-13)
    assert abs(spectrum[100]) == pytest.approx(0.5, abs=1.0e-13)
    outside = np.delete(spectrum, (0, 100))
    assert np.max(np.abs(outside)) < 1.0e-12
    assert np.min(angular_factor) == pytest.approx(0.0, abs=1.0e-13)
    assert np.max(angular_factor) == pytest.approx(2.0, abs=1.0e-13)

def test_initial_density_trace_guess_projects_constant_to_reduced_skeleton() -> None:
    from hdgfem.core.mesh import rectangle_mesh
    from hdgfem.core.space import DGSpace

    space = DGSpace(rectangle_mesh(2, 2), 2)
    guess = project_callable_to_trace(
        space,
        lambda x, y: 2.5 + 0.0 * x + 0.0 * y,
        trace_basis="legacy-lagrange",
        reduced=True,
        backend="host",
    )

    assert guess.shape == (space.mesh.int_edges_inds.size * space.trace_space("legacy-lagrange").edg_dof,)
    np.testing.assert_allclose(guess, 2.5, rtol=1.0e-13, atol=1.0e-13)


def test_rho_helm_wave_satisfies_negative_laplacian_phi_equals_rho() -> None:
    case = case_definition_by_key("rho_helm_wave").build(U=1.25, kx=1.4, ky=0.7)
    assert case.negative_laplacian_potential is not None
    assert case.exact_density is not None
    assert case.exact_potential is not None

    x = np.array([-0.7, -0.2, 0.35, 0.8])
    y = np.array([-0.4, 0.25, 0.6, -0.1])
    t = 0.37
    h = 1.0e-5
    phi = case.exact_potential
    negative_laplacian_fd = -(
        phi(x + h, y, t)
        + phi(x - h, y, t)
        + phi(x, y + h, t)
        + phi(x, y - h, t)
        - 4.0 * phi(x, y, t)
    ) / (h * h)
    np.testing.assert_allclose(
        negative_laplacian_fd,
        case.exact_density(x, y, t),
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    np.testing.assert_allclose(
        case.negative_laplacian_potential(x, y, t),
        case.exact_density(x, y, t),
        rtol=1.0e-14,
        atol=1.0e-14,
    )


def test_rho_helm_wave_host_accuracy_preset_runs_few_steps(tmp_path: Path) -> None:
    config = replace(
        preset_by_key("rho_helm_wave_host_accuracy"),
        diagnostics_dir=str(tmp_path),
        diagnostics_prefix="rho_helm_accuracy_test",
        verbosity=0,
        plot_every=0,
    )
    result = run_guiding_center_case(config, preset_key="rho_helm_wave_host_accuracy")
    final = result.diagnostics[-1]

    assert result.csv_path.exists()
    assert result.jsonl_path.exists()
    assert result.timings_csv_path.exists()
    assert result.timings_jsonl_path.exists()
    timing_rows = [json.loads(line) for line in result.timings_jsonl_path.read_text().splitlines()]
    assert len(timing_rows) == config.num_steps + 1
    assert len(result.diagnostics) == config.num_steps + 1
    assert final["rho_l2_error"] is not None
    assert final["phi_l2_error"] is not None
    assert final["rho_l2_error"] < 2.0e-2
    assert final["phi_l2_error"] < 2.0e-2


def test_guiding_center_verbose_logging_reports_post_poisson_work(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = replace(
        preset_by_key("rho_helm_wave_host_accuracy"),
        nx=2,
        ny=2,
        order=1,
        num_steps=1,
        diagnostics_dir=str(tmp_path),
        diagnostics_prefix="post_poisson_logging_test",
        verbosity=3,
        plot_every=0,
    )

    result = run_guiding_center_case(config, preset_key="rho_helm_wave_host_accuracy")
    output = capsys.readouterr().out
    final = result.diagnostics[-1]

    assert "[gc] updating accepted potential trace" in output
    assert "[gc] computing accepted-step diagnostics" in output
    assert "[gc] writing diagnostics JSONL" in output
    assert "GUIDING-CENTER ACCEPTED-STATE DIAGNOSTICS" in output
    assert "Conservation" in output
    assert "relative mass drift" in output
    assert "relative energy drift" in output
    assert "Linear-solver checks" in output
    assert "Phase timings" in output
    assert "first Poisson wall" in output
    assert "first Poisson operator assembly" in output
    assert "accepted-state diagnostics" in output
    assert "post-Poisson application work" in output
    assert final["energy_relative_drift"] == pytest.approx(
        (final["q_l2"] ** 2 - result.diagnostics[0]["q_l2"] ** 2)
        / result.diagnostics[0]["q_l2"] ** 2
    )
    for key in (
        "potential_trace_update_time",
        "diagnostics_core_time",
        "diagnostics_equilibrium_potential_time",
        "diagnostics_equilibrium_density_time",
        "diagnostics_azimuthal_mode_time",
        "diagnostics_wall_time",
        "post_poisson_application_time",
    ):
        assert final[key] >= 0.0


def test_diagnostics_block_clearly_labels_instability_and_modes(capsys) -> None:
    config = replace(
        preset_by_key("diocotron_k100_p6_dt01_t50_full_raw_cuda_amgx"),
        verbosity=3,
    )
    row = {
        "phase": "step",
        "step": 12,
        "time": 1.2,
        "mass": 0.25,
        "mass_relative_drift": 2.0e-14,
        "q_l2": 0.125,
        "energy_from_q_l2": 0.0078125,
        "energy_relative_drift": 4.0e-6,
        "rho_min": -1.0e-8,
        "rho_max": 1.04,
        "phi_min": 2.0e-6,
        "phi_max": 1.0e-2,
        "diocotron_phi_eq_l2": 3.0e-5,
        "diocotron_phi_eq_relative_l2": 7.0e-4,
        "diocotron_phi_eq_linf": 8.0e-5,
        "diocotron_rho_eq_l2": 9.0e-4,
        "diocotron_rho_eq_relative_l2": 1.0e-3,
        "diocotron_mode_base": 50.0,
        "diocotron_mode_1k_amplitude": 0.05,
        "diocotron_mode_2k_amplitude": 0.002,
        "diocotron_mode_3k_amplitude": 0.0001,
        "diocotron_harmonic_ratio": 0.04,
        "poisson_solver_rel_residual": 1.0e-9,
        "transport_solver_rel_residual": 2.0e-14,
        "poisson_time": 0.9,
        "transport_time": 0.4,
        "diagnostics_time": 0.2,
        "diagnostics_core_time": 0.03,
        "diagnostics_equilibrium_potential_time": 0.02,
        "diagnostics_equilibrium_density_time": 0.01,
        "diagnostics_azimuthal_mode_time": 0.14,
    }

    _print_step_summary(config, row)
    output = capsys.readouterr().out

    assert output.startswith("\n" + "=" * 78)
    assert "Instability relative to equilibrium" in output
    assert "potential amplitude ||phi-phi_eq|| L2" in output
    assert "density amplitude ||rho-rho_eq|| L2" in output
    assert "normalized mode k=50 amplitude" in output
    assert "harmonic ratio (2k/k)" in output
    assert output.endswith("=" * 78 + "\n\n")


def test_guiding_center_cli_accepts_response_file(tmp_path: Path) -> None:
    script = Path("scripts/guiding_center/run_guiding_center_cases.py")
    args_path = tmp_path / "guiding_center.args"
    args_path.write_text(
        "# comments and blank lines are allowed\n"
        "--preset diocotron_gaussian_annulus_host_smoke\n"
        "--num-steps 0\n"
        "--plot-every 0\n"
        "--diagnostics-prefix response_file_smoke\n"
        "--dry-run\n"
    )

    completed = subprocess.run(
        [sys.executable, str(script), f"@{args_path}"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Preset: diocotron_gaussian_annulus_host_smoke" in completed.stdout
    assert "num_steps: 0" in completed.stdout
def test_guiding_center_runner_uses_only_public_solver_classes_for_gpu_paths() -> None:
    path = Path("scripts/guiding_center/runtime/runner.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    source = path.read_text(encoding="utf-8")

    assert "hdgfem.backends" not in "\n".join(sorted(imported_modules))
    assert not {"cupy", "pyamgx"} & imported_names
    assert "_device_coefficients_for" not in source
    assert "DiffusionReactionHDGSolver" in source
    assert "AdvectionReactionHDGSolver" in source


def test_fb_hp_mg_benchmark_presets_are_matched_six_step_runs() -> None:
    native = preset_by_key(
        "diocotron_gaussian_annulus_k3_p6_150k_fb_hp_mg_6step"
    )
    hybrid = preset_by_key(
        "diocotron_gaussian_annulus_k3_p6_150k_hybrid_amgx_6step"
    )

    assert native.mesh_size == hybrid.mesh_size == pytest.approx(0.0068)
    assert native.minimum_triangles == hybrid.minimum_triangles == 150_000
    assert native.num_steps == hybrid.num_steps == 6
    assert native.time_scheme == hybrid.time_scheme == "si-euler"
    assert native.plot_every == hybrid.plot_every == 0
    assert native.poisson_solver == "fb-hp-mg-pcg"
    assert hybrid.poisson_solver == "amgx"
    assert native.poisson_trace_basis == hybrid.poisson_trace_basis == "legendre-modal"
    assert native.poisson_raw_matrix_format == hybrid.poisson_raw_matrix_format == "bsr"


def test_diocotron_k50_150k_plot30_preset_uses_bsr_for_both_systems() -> None:
    config = preset_by_key("diocotron_k50_p6_150k_raw_cuda_bsr_plot30")

    assert config.case == "diocotron_k"
    assert config.case_params["k"] == 50
    assert config.case_params["eps"] == pytest.approx(0.2)
    assert config.case_params["s_bar"] == pytest.approx(0.45)
    assert config.case_params["s_d"] == pytest.approx(0.016)
    assert config.order == 6
    assert config.mesh_size == pytest.approx(0.0068)
    assert config.minimum_triangles == 150_000
    assert config.num_steps == 500
    assert config.plot_every == 30
    assert config.diagnostics_every == 30
    assert config.poisson_assembly_backend == "raw-cuda"
    assert config.poisson_solver == "fb-hp-mg-pcg"
    assert config.poisson_trace_basis == "legendre-modal"
    assert config.poisson_raw_matrix_format == "bsr"
    assert config.transport_assembly_backend == "raw-cuda"
    assert config.transport_solver == "amgx"
    assert config.transport_trace_basis == "legacy-lagrange"
    assert config.transport_raw_matrix_format == "bsr"
    assert config.transport_initial_guess == "initial-density-trace"


def test_level_three_uses_compact_native_solver_logging() -> None:
    config = replace(
        preset_by_key("diocotron_k50_p6_150k_raw_cuda_bsr_plot30"),
        verbosity=3,
    )
    assert _solver_verbosity(config) == 3
    assert _solver_verbosity(replace(config, verbosity=2)) == 1


def test_linear_step_summary_is_balanced_and_compact(capsys) -> None:
    config = replace(
        preset_by_key("diocotron_k50_p6_150k_raw_cuda_bsr_plot30"),
        verbosity=3,
        num_steps=12,
    )
    row = {
        "phase": "step",
        "step": 3,
        "time": 0.125,
        "linear_step_wall_time": 0.8123,
        "beta_build_time": 0.0012,
        "potential_trace_update_time": 0.0001,
        "transport_time": 0.3642,
        "transport_step_wall_time": 0.3654,
        "transport_step_time_assembly": 0.2408,
        "transport_step_time_solve": 0.1237,
        "transport_step_time_reconstruction": 0.0009,
        "transport_solver_iterations": 32,
        "transport_physical_rel_residual": 3.264e-13,
        "poisson_time": 0.4438,
        "poisson_step_wall_time": 0.4457,
        "poisson_step_time_assembly": 0.1212,
        "poisson_time_rhs_assembly": 0.1212,
        "poisson_step_time_solve": 0.1908,
        "poisson_step_time_reconstruction": 0.1337,
        "poisson_solver_iterations": 9,
        "poisson_physical_rel_residual": 1.392e-10,
        "poisson_detail_raw_assembly_operator_reused": 1.0,
        "poisson_detail_solve_fb_hp_mg_hierarchy_reused": 1.0,
    }

    _print_linear_step_summary(config, row)
    output = capsys.readouterr().out

    assert "[gc:linear] step 00003/00012" in output
    assert "coupled wall=0.8123s" in output
    assert "transport HDG=0.3642s | stage wall=0.3654s | asm=0.2408s" in output
    assert "it=32 | true_rel=3.264e-13" in output
    assert "poisson   HDG=0.4438s | stage wall=0.4457s | rhs=0.1212s" in output
    assert "it=9 | true_rel=1.392e-10" in output
    assert "reuse=operator+hierarchy" in output
    assert len(output.rstrip().splitlines()) == 4


@pytest.mark.parametrize("preset,atol,primary_tolerance", (
    ("diocotron_k50_p6_150k_raw_cuda_bsr_plot30", 5.0e-9, 1.0e-8),
    ("euler_vortex_gas_p6_50k_dt001_t50_raw_cuda_bsr", 1.0e-12, 1.0e-11),
))
def test_robust_transport_uses_bsr_preconditioners_before_scaled_fgmres(
    preset, atol, primary_tolerance
) -> None:
    config = preset_by_key(preset)
    options = _make_transport_options(config, boundary_mode="zero-flux")

    assert options.scale_system is True
    assert options.solver_rtol == pytest.approx(1.0e-11)
    assert options.solver_atol == pytest.approx(atol)
    assert options.amgx_config["solver"]["solver"] == "BICGSTAB"
    assert options.amgx_config["solver"]["tolerance"] == pytest.approx(primary_tolerance)
    assert "preconditioner" not in options.amgx_config["solver"]
    assert options.amgx_retry_attempts is not None
    for attempt, preconditioner in zip(
        options.amgx_retry_attempts[:2], ("JACOBI_L1", "BLOCK_JACOBI"), strict=True
    ):
        solver = attempt["config"]["solver"]
        assert solver["solver"] == "PBICGSTAB"
        assert solver["preconditioner"]["solver"] == preconditioner
        assert solver["preconditioner"]["max_iters"] == 1
        assert solver["preconditioner"]["relaxation_factor"] == 1.0
        assert solver["bsr_spmv_backend"] == "cusparse_generic"
        assert solver["tolerance"] == config.transport_solver_rtol
        assert attempt["scale_system"] is config.transport_scale_system
        assert attempt["scalarize_bsr"] is False
        assert attempt["reuse_preconditioner"] is False
        assert attempt["use_initial_guess"] is False
    l1 = options.amgx_retry_attempts[0]["config"]["solver"]["preconditioner"]
    assert l1["jacobi_l1_scalar_rows_for_blocks"] == 1
    assert options.amgx_retry_attempts[2]["label"] == "robust-zero-scaled"
    for attempt in options.amgx_retry_attempts[2:]:
        assert attempt["scale_system"] is True
        solver = attempt["config"]["solver"]
        assert attempt["scalarize_bsr"] is True
        assert attempt["reuse_preconditioner"] is True
        assert attempt["solver_cache_key"] == "transport-fgmres-dilu"
        assert solver["solver"] == "FGMRES"
        assert solver["preconditioner"]["solver"] == "MULTICOLOR_DILU"


def test_poisson_flux_postprocess_cadence_defaults_to_disabled() -> None:
    config = preset_by_key("diocotron_gaussian_annulus_host_smoke")
    assert config.poisson_flux_postprocess_every == 0
    assert config.poisson_flux_postprocess_space == "RT_projection"


def test_poisson_flux_postprocess_cadence_targets_only_accepted_steps() -> None:
    config = replace(
        preset_by_key("diocotron_gaussian_annulus_host_smoke"),
        poisson_flux_postprocess_every=3,
        poisson_postprocessing_backend="raw-cuda",
    )

    assert _poisson_postprocess_overrides(config, 0) == {}
    assert _poisson_postprocess_overrides(config, 2) == {}
    assert _poisson_postprocess_overrides(config, 3) == {
        "postprocess_overrides": {
            "hdg_postprocess": "flux",
            "flux_postprocess_space": "RT_projection",
            "postprocessing_backend": "raw-cuda",
        }
    }
