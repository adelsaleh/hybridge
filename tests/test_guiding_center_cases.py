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

from scripts.guiding_center.guiding_center_cases import (
    CASE_DEFINITIONS,
    case_definition_by_key,
    rho_eq_annular_band,
    rho_eq_gaussian_annulus,
    rho_eq_super_gaussian_annulus,
)
from scripts.guiding_center.guiding_center_presets import preset_by_key
from hdgfem.core.field_ops import project_callable_to_trace
from scripts.guiding_center.run_guiding_center_cases import run_guiding_center_case


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
    assert csv_path.exists()
    assert jsonl_path.exists()

    final = json.loads(jsonl_path.read_text().strip().splitlines()[-1])
    for key in ("mass_relative_drift", "q_l2_relative_drift", "rho_min", "rho_max", "transport_solver_residual"):
        assert final[key] is not None
        assert math.isfinite(final[key])
    assert abs(final["mass_relative_drift"]) < 1.0e-3
    assert final["transport_solver_residual"] < 1.0e-4


def test_guiding_center_case_registry_has_legacy_gaussian_new_diocotron_and_rho_helm() -> None:
    assert tuple(CASE_DEFINITIONS) == ("diocotron_gaussian_annulus", "diocotron_k", "rho_helm_wave")


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
    assert config.poisson_reuse_equilibrium_solver is True
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
    assert config.poisson_reuse_equilibrium_solver is True
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
    assert config.poisson_reuse_equilibrium_solver is True
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
    assert config.poisson_reuse_equilibrium_solver is True
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
    assert config.poisson_solver == "amgx"
    assert config.poisson_raw_matrix_format == "csr"
    assert config.transport_assembly_backend == "raw-cuda"
    assert config.transport_solver == "amgx"
    assert config.transport_raw_matrix_format == "csr"
    assert config.transport_materialize_host_system is False
    assert config.transport_materialize_host_solution is False

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
    assert config.poisson_raw_matrix_format == "csr"
    assert config.transport_raw_matrix_format == "csr"




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
    assert len(result.diagnostics) == config.num_steps + 1
    assert final["rho_l2_error"] is not None
    assert final["phi_l2_error"] is not None
    assert final["rho_l2_error"] < 2.0e-2
    assert final["phi_l2_error"] < 2.0e-2


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
    path = Path("scripts/guiding_center/run_guiding_center_cases.py")
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
