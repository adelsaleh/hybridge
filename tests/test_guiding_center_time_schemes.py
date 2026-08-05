from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.guiding_center.guiding_center_cases import case_definition_by_key
from scripts.guiding_center.guiding_center_presets import preset_by_key
from scripts.guiding_center.run_guiding_center_cases import (
    _average_boundary_data,
    _make_transport_options,
    _validate_config,
    run_guiding_center_case,
)
from scripts.guiding_center.run_guiding_center_temporal_convergence import (
    _pairwise_rates,
    _selected_schemes,
    plot_convergence,
)


def test_rho_helm_wave_has_non_tangent_boundary_velocity_and_midpoint_data() -> None:
    case = case_definition_by_key("rho_helm_wave").build(U=1.0, kx=1.0, ky=1.0)
    assert case.exact_flux is not None
    y = np.linspace(-1.0, 1.0, 17)
    x = np.ones_like(y)
    qx, qy = case.exact_flux(x, y, 0.13)
    normal_velocity = -np.asarray(qy)
    assert np.max(np.abs(normal_velocity)) > 0.1

    left = case.density_boundary_at(0.1)
    right = case.density_boundary_at(0.2)
    midpoint = _average_boundary_data(left, right)
    np.testing.assert_allclose(midpoint(x, y), 0.5 * (left(x, y) + right(x, y)))


def test_rho_helm_wave_rejects_zero_flux_transport() -> None:
    config = replace(
        preset_by_key("rho_helm_wave_host_accuracy"),
        transport_boundary_mode="zero-flux",
    )
    with pytest.raises(ValueError, match="zero-flux"):
        _validate_config(config)


def test_predictor_corrector_host_manufactured_smoke(tmp_path: Path) -> None:
    config = replace(
        preset_by_key("rho_helm_wave_host_accuracy"),
        time_scheme="predictor-corrector",
        dt=0.01,
        num_steps=2,
        diagnostics_dir=str(tmp_path),
        diagnostics_prefix="predictor_corrector_host",
        plot_every=0,
        verbosity=0,
    )
    result = run_guiding_center_case(config, preset_key="predictor_corrector_host")
    final = result.diagnostics[-1]

    assert len(result.diagnostics) == 3
    assert final["time_scheme"] == "predictor-corrector"
    assert final["predictor_transport_time_total"] > 0.0
    assert final["predictor_poisson_time_total"] > 0.0
    assert final["corrector_transport_time_total"] > 0.0
    assert final["rho_l2_error"] < 2.0e-3
    assert final["phi_l2_error"] < 2.0e-3


def test_solver_initial_guesses_are_per_call_and_not_stored() -> None:
    from hdgfem.core.mesh import rectangle_mesh
    from hdgfem.core.space import DGSpace, VectorDGField
    from hdgfem.solvers.advection_reaction import AdvectionReactionHDGOptions, AdvectionReactionHDGSolver
    from hdgfem.solvers.diffusion_reaction import DiffusionReactionHDGOptions, DiffusionReactionHDGSolver

    space = DGSpace(rectangle_mesh(2, 2), 1)
    source = space.constant(1.0)
    reaction = space.constant(1.0)
    zero_reaction = space.zeros()
    beta = VectorDGField((space.constant(0.2), space.constant(-0.1)))
    boundary = lambda x, y: 0.0 * x + 0.0 * y

    adv = AdvectionReactionHDGSolver(
        space,
        source=source,
        beta=beta,
        reaction=reaction,
        boundary_condition=boundary,
        options=AdvectionReactionHDGOptions(
            solver="direct",
            preconditioner=None,
            assembly_backend="numpy",
            boundary_mode="eliminate",
            verbose=0,
        ),
    )
    first_adv = adv.solve()
    adv.solve(initial_guess=np.asarray(first_adv.trace).reshape(-1))
    assert adv.options.initial_guess is None

    diffusion = DiffusionReactionHDGSolver(
        space,
        source=source,
        reaction=zero_reaction,
        boundary_condition=boundary,
        options=DiffusionReactionHDGOptions(
            solver="direct",
            preconditioner=None,
            assembly_backend="numpy",
            boundary_mode="eliminate",
            verbose=0,
        ),
    )
    first_diffusion = diffusion.solve()
    diffusion.solve(initial_guess=np.asarray(first_diffusion.trace).reshape(-1))
    assert diffusion.options.initial_guess is None


def test_robust_transport_policy_builds_defect_correction_retries() -> None:
    config = preset_by_key("diocotron_k100_p6_dt01_t50_full_raw_cuda_amgx")
    options = _make_transport_options(config, "zero-flux")
    retries = options.amgx_retry_attempts

    assert retries is not None
    assert [retry["label"] for retry in retries] == [
        "primary-zero",
        "robust-zero-unscaled",
        "robust-correction-1",
        "robust-correction-2",
    ]
    assert retries[1]["scale_system"] is False
    assert retries[1]["use_initial_guess"] is False
    assert retries[1]["config"]["solver"]["convergence"] == "ABSOLUTE"
    assert retries[1]["config"]["solver"]["tolerance"] == config.transport_solver_atol
    assert retries[2]["scale_system"] is False
    assert retries[2]["use_initial_guess"] is False
    assert retries[2]["use_best_solution"] is True
    assert retries[2]["residual_correction"] is True
    assert retries[3]["use_best_solution"] is True
    assert retries[3]["residual_correction"] is True


def test_convergence_scheme_selection_rates_and_plot(tmp_path: Path, monkeypatch) -> None:
    assert _selected_schemes("both") == ("si-euler", "predictor-corrector")
    assert _selected_schemes("si-euler") == ("si-euler",)
    rows = []
    for scheme, power in (("si-euler", 1), ("predictor-corrector", 2)):
        for dt in (0.04, 0.02, 0.01):
            rows.append(
                {
                    "scheme": scheme,
                    "dt": dt,
                    "rho_l2_error": dt**power,
                    "rho_linf_error": 2.0 * dt**power,
                    "phi_l2_error": 0.5 * dt**power,
                    "phi_linf_error": 0.75 * dt**power,
                }
            )
    _pairwise_rates(rows)
    for row in rows:
        if row["dt"] == 0.04:
            continue
        expected = 1.0 if row["scheme"] == "si-euler" else 2.0
        assert row["rho_l2_rate"] == pytest.approx(expected)
        assert row["phi_linf_rate"] == pytest.approx(expected)

    monkeypatch.setenv("MPLBACKEND", "Agg")
    output = plot_convergence(rows, tmp_path / "convergence.png", show=False)
    assert output.exists()
    assert output.stat().st_size > 0
