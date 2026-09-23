from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.guiding_center.cases.guiding_center_cases import case_definition_by_key
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key


PRESET = "diocotron_multimode_p6_50k_dt01_t100_raw_cuda_bsr"


def test_multimode_annulus_has_requested_spectrum_and_radial_profile() -> None:
    config = preset_by_key(PRESET)
    shift = 0.17
    case = case_definition_by_key(config.case).build(
        **config.case_params, theta_shift=shift,
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=False)
    radius = 0.4162
    values = case.initial_density(radius * np.cos(theta), radius * np.sin(theta))
    equilibrium = case.equilibrium_density(radius, 0.0)
    spectrum = np.fft.rfft(values / equilibrium) / len(theta)
    expected = np.zeros_like(spectrum)
    expected[0] = 1.0
    for mode in range(3, 8):
        expected[mode] = 0.075 * np.exp(1j * (0.37 * mode**2 - mode * shift))
    np.testing.assert_allclose(spectrum, expected, atol=1.0e-13)
    assert 0.25 * equilibrium <= np.min(values) < 0.5 * equilibrium
    assert 1.4 * equilibrium < np.max(values) <= 1.75 * equilibrium

    radii = np.array([0.0, 0.3, 0.3724, radius, 0.46, 0.6, 1.0])
    profile = case.equilibrium_density(radii, 0.0)
    np.testing.assert_allclose(profile, [0, 0, 0.5, 1, 0.5, 0, 0], atol=1.0e-12)
    assert case.initial_density(radii[:, None], np.zeros((1, 3))).shape == (7, 3)
    np.testing.assert_array_equal(case.potential_boundary_at(2.0)(radii, 0.0), 0.0)
    assert case.default_domain == "disc"
    assert case.density_boundary_at(2.0) is None
    assert case.density_transport_boundary_mode == "zero-flux"
    assert case.parameters["modes"] == (3, 4, 5, 6, 7)
    assert case.parameters["k"] == 5


@pytest.mark.parametrize("modes", [[], [0, 3], [-2, 4], [3.5, 4], [3, 3]])
def test_multimode_annulus_rejects_invalid_modes(modes) -> None:
    with pytest.raises(ValueError, match="modes"):
        case_definition_by_key("diocotron_k").build(modes=modes)


def test_multimode_preset_is_a_50k_long_run() -> None:
    config = preset_by_key(PRESET)
    assert config.minimum_triangles == 50_000
    assert config.mesh_size == pytest.approx(0.012)
    assert config.order == 6
    assert config.dt == pytest.approx(0.1)
    assert config.num_steps == 1000
    assert config.poisson_solver == "fb-hp-mg-pcg"
    assert config.transport_solver == "amgx"
    assert config.poisson_raw_matrix_format == config.transport_raw_matrix_format == "bsr"
    assert config.plot_every == config.diagnostics_every == 20


def test_multimode_response_file_accepts_parameter_overrides() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m", "scripts.guiding_center.run_guiding_center_cases",
            f"@run_configs/guiding_center/{PRESET}.args",
            "--case-param", "modes=[4, 5, 6]",
            "--num-steps", "500",
            "--dry-run",
        ],
        cwd=root, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"Preset: {PRESET}" in completed.stdout
    assert "minimum_triangles: 50000" in completed.stdout
    assert "num_steps: 500" in completed.stdout
    fields = dict(line.split(": ", 1) for line in completed.stdout.splitlines() if ": " in line)
    params = ast.literal_eval(fields["case_params"])
    case = case_definition_by_key(ast.literal_eval(fields["case"])).build(**params)
    assert case.parameters["modes"] == (4, 5, 6)
