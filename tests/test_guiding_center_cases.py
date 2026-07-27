from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.guiding_center.guiding_center_cases import case_definition_by_key
from scripts.guiding_center.guiding_center_presets import preset_by_key
from scripts.guiding_center.run_guiding_center_cases import run_guiding_center_case


def test_guiding_center_case_factories_vectorize_on_arrays() -> None:
    x = np.array([[0.1, 0.2, -0.3], [0.4, -0.5, 0.0]])
    y = np.array([[0.0, 0.3, -0.2], [0.1, 0.2, -0.4]])

    for key in ("diocotron_k", "rho_helm_wave"):
        case = case_definition_by_key(key).build()
        rho0 = case.initial_density(x, y)
        phi_boundary = case.potential_boundary_at(0.125)(x, y)
        assert rho0.shape == x.shape
        assert phi_boundary.shape == x.shape
        if case.density_boundary is not None:
            rho_boundary = case.density_boundary_at(0.125)(x, y)
            assert rho_boundary.shape == x.shape


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


def test_guiding_center_diocotron_cli_smoke(tmp_path: Path) -> None:
    pytest.importorskip("gmsh")
    script = Path("scripts/guiding_center/run_guiding_center_cases.py")
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--preset",
            "diocotron_k3_host_smoke",
            "--num-steps",
            "1",
            "--mesh-size",
            "0.9",
            "--diagnostics-dir",
            str(tmp_path),
            "--diagnostics-prefix",
            "diocotron_smoke_test",
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
    assert (tmp_path / "diocotron_smoke_test.csv").exists()
    assert (tmp_path / "diocotron_smoke_test.jsonl").exists()
