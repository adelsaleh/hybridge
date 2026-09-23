from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from scripts.guiding_center.cases.guiding_center_cases import case_definition_by_key, spiral_sheet
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key


PRESET = "spiral_sheet_p6_50k_dt002_t100_raw_cuda_bsr"


@pytest.mark.parametrize("params", [{}, {"turns": 2, "sigma": 0.003, "theta_shift": 0.31}])
def test_spiral_matches_independent_full_curve_integral(params) -> None:
    case = spiral_sheet(**params)
    p = case.parameters
    end = 2 * np.pi * p["turns"]
    pitch = (p["r_outer"] - p["r_inner"]) / end
    angles = np.array([0, 0.03, 1.7, 2*np.pi, end-0.03, end])
    r = p["r_inner"] + pitch * angles
    theta = angles + p["theta_shift"]
    x, y = r * np.cos(theta), r * np.sin(theta)
    # Include normal offsets, points on either side of the polar seam, and vacuum.
    x = np.r_[x, x[2] + p["sigma"], p["r_outer"], p["r_outer"], 0.0]
    y = np.r_[y, y[2] - p["sigma"], -1.e-8, 1.e-8, 0.0]
    reference = []
    for px, py in zip(x, y):
        def integrand(a):
            radius = p["r_inner"] + pitch * a
            angle = a + p["theta_shift"]
            distance_sq = (px - radius*np.cos(angle))**2 + (py - radius*np.sin(angle))**2
            return np.exp(-distance_sq / (2*p["sigma"]**2)) * np.hypot(radius, pitch)
        # Panel breakpoints prevent the adaptive integrator missing narrow peaks.
        value, _ = quad(
            integrand, 0, end, points=np.linspace(0, end, 601),
            limit=1200, epsabs=1.e-13, epsrel=1.e-12,
        )
        reference.append(value * p["rho_bar"] / (np.sqrt(2*np.pi)*p["sigma"]))
    np.testing.assert_allclose(case.initial_density(x, y), reference, atol=3.e-11, rtol=3.e-11)


def test_spiral_width_turns_and_boundary_conditions() -> None:
    case = spiral_sheet()
    pitch = 0.63 / (10*np.pi)
    # Away from the tips, every winding has unit peak and Gaussian normal width.
    angles = np.pi + 2*np.pi*np.arange(5)
    radii = 0.12 + pitch * angles
    centers = np.stack((radii*np.cos(angles), radii*np.sin(angles)))
    tangent = np.stack((pitch*np.cos(angles)-radii*np.sin(angles),
                        pitch*np.sin(angles)+radii*np.cos(angles)))
    normal = np.stack((-tangent[1], tangent[0])) / np.linalg.norm(tangent, axis=0)
    peaks = case.initial_density(*centers)
    half_width = np.sqrt(2*np.log(2)) * 0.005
    half_values = case.initial_density(*(centers + half_width*normal))
    np.testing.assert_allclose(peaks, 1.0, atol=2.e-4)
    np.testing.assert_allclose(half_values/peaks, 0.5, atol=0.01)
    gaps = radii[:-1] + 0.063
    assert np.max(case.initial_density(-gaps, 0.0)) < 1.e-20
    theta = np.linspace(0, 2*np.pi, 100)
    np.testing.assert_array_equal(case.initial_density(np.cos(theta), np.sin(theta)), 0.0)
    np.testing.assert_array_equal(case.potential_boundary_at(7.0)(np.cos(theta), np.sin(theta)), 0.0)
    assert case.density_boundary_at(7.0) is None
    assert case.density_transport_boundary_mode == "zero-flux"
    assert case.default_domain == "disc"
    assert case.equilibrium_density is None
    assert case.initial_density(0.0, 0.0).shape == ()
    assert case.initial_density(np.zeros((3, 1)), np.zeros((1, 4))).shape == (3, 4)


@pytest.mark.parametrize("params", [
    {"turns": 0}, {"turns": 2.5}, {"sigma": 0}, {"sigma": np.nan},
    {"r_inner": 0.8}, {"r_outer": 1.1},
])
def test_spiral_rejects_invalid_parameters(params) -> None:
    with pytest.raises(ValueError):
        spiral_sheet(**params)


def test_spiral_response_file_and_overrides() -> None:
    config = preset_by_key(PRESET)
    assert config.case == "spiral_sheet"
    assert config.minimum_triangles == 50_000
    assert config.order == 6
    assert config.dt * config.num_steps == pytest.approx(100.0)
    assert config.time_scheme == "predictor-corrector"
    assert config.poisson_solver == "fb-hp-mg-pcg"
    assert config.transport_solver == "amgx"
    assert config.poisson_raw_matrix_format == config.transport_raw_matrix_format == "bsr"
    assert config.plot_every == config.diagnostics_every == 50
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.guiding_center.run_guiding_center_cases",
         f"@run_configs/guiding_center/{PRESET}.args",
         "--case-param", "sigma=0.004", "--num-steps", "2500", "--dry-run"],
        cwd=root, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "minimum_triangles: 50000" in completed.stdout
    assert "num_steps: 2500" in completed.stdout
    fields = dict(line.split(": ", 1) for line in completed.stdout.splitlines() if ": " in line)
    case = case_definition_by_key(ast.literal_eval(fields["case"])).build(
        **ast.literal_eval(fields["case_params"]),
    )
    assert case.parameters["sigma"] == 0.004
