"""Guiding-center diagnostics helpers."""

from __future__ import annotations
import time
from typing import Any
from hdgfem.core.field_ops import perpendicular_vector_field
from hdgfem.diagnostics import (
    evaluate_scalar_error,
    guiding_center_field_diagnostics,
    transport_velocity_diagnostics,
    relative_drift,
)


def _electric_flux(poisson_result):
    """Return the accepted higher-order electric field when available."""
    return poisson_result.postprocessed_flux or poisson_result.flux


def _compute_diagnostics(
        *,
        case,
        rho_field,
        poisson_result,
        step: int,
        time_value: float,
        baseline_mass: float | None,
        baseline_q_l2: float | None,
        baseline_enstrophy: float | None = None,
        equilibrium_potential=None,
        equilibrium_density=None,
        equilibrium_potential_l2: float | None = None,
        equilibrium_density_l2: float | None = None,
        extra: dict[str, Any] | None = None,
        transport_electric_field: str = "raw",
) -> dict[str, Any]:
    start = time.perf_counter()
    core_start = time.perf_counter()
    phi_field = poisson_result.field
    field_metrics = guiding_center_field_diagnostics(
        rho_field,
        phi_field,
        poisson_result.flux,
        postprocessed_flux=poisson_result.postprocessed_flux,
        equilibrium_potential=equilibrium_potential,
        equilibrium_density=equilibrium_density,
        mode=int(case.parameters.get("k", 0)),
        backend="auto",
    )
    drift_flux = poisson_result.flux
    if transport_electric_field == "postprocessed":
        drift_flux = poisson_result.postprocessed_flux
        if drift_flux is None:
            raise ValueError("recovered transport diagnostics require a postprocessed flux")
    # Measure the accepted electric field actually supplying the scheme's drift.
    velocity_metrics = transport_velocity_diagnostics(
        perpendicular_vector_field(drift_flux, 1.0, drift_flux.components[0].space),
    )
    standard_q_l2 = float(field_metrics["q_l2_standard"])
    postprocessed_q_l2 = field_metrics.get("q_l2_postprocessed")
    if postprocessed_q_l2 is not None:
        postprocessed_q_l2 = float(postprocessed_q_l2)
    q_l2 = postprocessed_q_l2 if transport_electric_field == "postprocessed" else standard_q_l2
    mass = float(field_metrics["mass"])
    effective_baseline_mass = mass if baseline_mass is None else float(baseline_mass)
    effective_baseline_q_l2 = q_l2 if baseline_q_l2 is None else float(baseline_q_l2)
    exact_density = case.exact_density_at(time_value)
    exact_potential = case.exact_potential_at(time_value)
    rho_report = (
        None
        if exact_density is None
        else evaluate_scalar_error(rho_field, exact_density, backend="auto", include_samples=False)
    )
    phi_report = (
        None
        if exact_potential is None
        else evaluate_scalar_error(phi_field, exact_potential, backend="auto", include_samples=False)
    )
    rho_l2_error = None if rho_report is None else rho_report.metrics.l2
    rho_linf_error = None if rho_report is None else rho_report.metrics.linf
    phi_l2_error = None if phi_report is None else phi_report.metrics.l2
    phi_linf_error = None if phi_report is None else phi_report.metrics.linf
    core_time = time.perf_counter() - core_start
    diagnostic_backend = str(field_metrics["diagnostics_backend"])
    row: dict[str, Any] = {
        "step": int(step),
        "time": float(time_value),
        "mass": mass,
        "mass_drift": mass - effective_baseline_mass,
        "mass_relative_drift": None if case.density_is_vorticity else relative_drift(mass, effective_baseline_mass),
        "q_l2": q_l2,
        "q_l2_standard": standard_q_l2,
        "q_l2_postprocessed": postprocessed_q_l2,
        "electric_flux_postprocessed": poisson_result.postprocessed_flux is not None,
        "transport_electric_field": transport_electric_field,
        "density_order": rho_field.space.order,
        "poisson_order": phi_field.space.order,
        "transport_electric_field_order": drift_flux.components[0].space.order,
        "q_l2_drift": q_l2 - effective_baseline_q_l2,
        "q_l2_relative_drift": relative_drift(q_l2, effective_baseline_q_l2),
        "energy_from_q_l2": 0.5 * q_l2 * q_l2,
        "energy_drift": 0.5 * (q_l2 * q_l2 - effective_baseline_q_l2 * effective_baseline_q_l2),
        "energy_relative_drift": relative_drift(
            0.5 * q_l2 * q_l2,
            0.5 * effective_baseline_q_l2 * effective_baseline_q_l2,
        ),
        "rho_min": float(field_metrics["rho_min"]),
        "rho_max": float(field_metrics["rho_max"]),
        "phi_min": float(field_metrics["phi_min"]),
        "phi_max": float(field_metrics["phi_max"]),
        "rho_l2_error": rho_l2_error,
        "rho_linf_error": rho_linf_error,
        "phi_l2_error": phi_l2_error,
        "phi_linf_error": phi_linf_error,
        "diagnostics_backend": diagnostic_backend,
        "diagnostics_core_time": core_time,
        "diagnostics_device_reduction_time": core_time if diagnostic_backend == "cuda" else 0.0,
        "diagnostics_equilibrium_potential_time": 0.0,
        "diagnostics_equilibrium_density_time": 0.0,
        "diagnostics_azimuthal_mode_time": 0.0,
    }
    row.update(velocity_metrics)
    enstrophy = 0.5 * float(field_metrics["rho_l2_squared"])
    z0 = enstrophy if baseline_enstrophy is None else float(baseline_enstrophy)
    row.update(enstrophy=enstrophy, enstrophy_drift=enstrophy-z0,
               enstrophy_relative_drift=relative_drift(enstrophy, z0))
    if case.density_is_vorticity:
        row["circulation"] = mass
        row["circulation_drift"] = mass - effective_baseline_mass
    if equilibrium_potential is not None:
        phi_eq_l2 = float(field_metrics["diocotron_phi_eq_l2"])
        eq_norm = max(
            float(field_metrics["diocotron_phi_eq_reference_l2"])
            if equilibrium_potential_l2 is None
            else float(equilibrium_potential_l2),
            1.0e-300,
        )
        row["diocotron_phi_eq_l2"] = phi_eq_l2
        row["diocotron_phi_eq_relative_l2"] = phi_eq_l2 / eq_norm
        row["diocotron_phi_eq_linf"] = float(field_metrics["diocotron_phi_eq_linf"])
        row["diocotron_phi_eq_reference_l2"] = eq_norm
    if equilibrium_density is not None:
        rho_eq_l2 = float(field_metrics["diocotron_rho_eq_l2"])
        eq_norm = max(
            float(field_metrics["diocotron_rho_eq_reference_l2"])
            if equilibrium_density_l2 is None
            else float(equilibrium_density_l2),
            1.0e-300,
        )
        row["diocotron_rho_eq_l2"] = rho_eq_l2
        row["diocotron_rho_eq_relative_l2"] = rho_eq_l2 / eq_norm
        row["diocotron_rho_eq_reference_l2"] = eq_norm
        for key in (
            "diocotron_mode_base",
            "diocotron_mode_1k_amplitude",
            "diocotron_mode_2k_amplitude",
            "diocotron_mode_3k_amplitude",
            "diocotron_harmonic_ratio",
        ):
            if key in field_metrics:
                row[key] = float(field_metrics[key])
    if extra:
        row.update(extra)
    row["diagnostics_time"] = time.perf_counter() - start
    return row
