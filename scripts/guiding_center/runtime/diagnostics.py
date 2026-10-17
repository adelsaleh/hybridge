"""Guiding-center diagnostics helpers."""

from __future__ import annotations
import math
import time
from typing import Any
from hybridge.core.field_ops import perpendicular_vector_field
from hybridge.diagnostics.errors import evaluate_scalar_error
from hybridge.diagnostics.guiding_center import (
    guiding_center_field_diagnostics,
    transport_velocity_diagnostics,
)
from hybridge.diagnostics.solver import relative_drift


def _electric_flux(poisson_result):
    """Return the accepted higher-order electric field when available."""
    return poisson_result.postprocessed_flux or poisson_result.flux


def _forced_mass_balance(case, mass: float, baseline_mass: float, time_value: float) -> dict[str, Any]:
    """Exact mass of a forced density equation and the relative error of the computed one.

    dM/dt = int S + alpha*M: M0 + int_0^t int S without reaction, M0*exp(alpha*t)
    without source; nothing is reported for unforced cases, when both act or for a
    space-dependent reaction (whose balance uses the recorded ``reaction_production``).
    """
    alpha = float(getattr(case, "density_reaction", 0.0))
    source_mass = getattr(case, "density_source_mass", None)
    if getattr(case, "density_reaction_profile", None) is not None:
        return {}
    if source_mass is not None and alpha == 0.0:
        expected = baseline_mass + float(source_mass(float(time_value)))
    elif getattr(case, "density_source", None) is None and alpha != 0.0:
        expected = baseline_mass * math.exp(alpha * float(time_value))
    else:
        return {}
    return {"mass_expected": expected,
            "mass_balance_relative_error": (mass - expected) / expected if expected else None}


class WallChargeBalance:
    """Wall outflux through an open, zero-inflow boundary and the discrete charge balance.

    With one transport solve per step (SI-Euler, SI-BDF2, SI-BDF3), velocity beta = s v
    and the forcing taken at t^{n+1}, testing the solve with 1 gives

        M^{n+1} - H^n = s (int S_h + int alpha_h rho^{n+1} - I^{n+1}),

    H^n = M^n (Euler and the BDF2 start-up), (4 M^n - M^{n-1})/3 (BDF2) or
    (18 M^n - 9 M^{n-1} + 2 M^{n-2})/11 (BDF3, after its SI-BDF2 second step; its
    Richardson SI-Euler first step combines three solves and has no such identity), and
    I = oint |v.n| rho_h the wall outflux of the eliminated upwind traces
    (:func:`hybridge.diagnostics.guiding_center.boundary_upwind_outflux`).
    ``charge_balance_defect`` is the left side minus the right side; it is
    round-off when every step is recorded. On an annulus the outflux is also
    split between the inner and the outer wall.
    """

    def __init__(self, case, scheme: str, trace_basis: str):
        geometry = case.parameters.get("geometry") or {}
        inner, outer = geometry.get("inner_radius"), geometry.get("outer_radius")
        self.groups = None
        if inner and outer:
            middle = (0.5 * (float(inner) + float(outer))) ** 2
            self.groups = {"inner": lambda x, y: x * x + y * y < middle,
                           "outer": lambda x, y: x * x + y * y >= middle}
        self.scheme = str(scheme)
        self.exact = self.scheme in {"si-euler", "si-bdf2", "si-bdf3"}
        self.trace_basis = trace_basis
        self.masses: dict[int, float] = {}

    def flux(self, beta, density, scale: float) -> dict[str, float]:
        """Physical wall outflux (per unit time) of the step's transport solve."""
        from hybridge.diagnostics.guiding_center import boundary_upwind_outflux

        values = boundary_upwind_outflux(beta, density, trace_basis=self.trace_basis, groups=self.groups)
        return {f"wall_{key}": value / float(scale) for key, value in values.items()}

    def balance(self, step: int, row: dict[str, Any]) -> dict[str, Any]:
        """Record M^{step}; return the defect of the discrete charge balance when its history is complete."""
        step, mass = int(step), float(row["mass"])
        self.masses[step] = mass
        for old in [key for key in self.masses if key < step - 3]:
            self.masses.pop(old)
        if step == 0 or not self.exact or step - 1 not in self.masses:
            return {}
        masses = [self.masses.get(step - k) for k in (1, 2, 3)]
        if self.scheme == "si-bdf3":
            method = row.get("bdf3_startup_method")
            if method == "si-euler-extrap2":
                return {}
            order = 2 if method == "si-bdf2" else 3
        else:
            order = 1 if self.scheme == "si-euler" or step == 1 or row.get("bdf2_startup", False) else 2
        weights = {1: (1.0,), 2: (4.0 / 3.0, -1.0 / 3.0), 3: (18.0 / 11.0, -9.0 / 11.0, 2.0 / 11.0)}[order]
        if any(mass is None for mass in masses[:order]):
            return {}
        history = sum(weight * mass for weight, mass in zip(weights, masses))
        scale = float(row["transport_beta_scale"])
        production = (float(row.get("source_production") or 0.0) + float(row.get("reaction_production") or 0.0)
                      - float(row["wall_outflux"]))
        defect = mass - history - scale * production
        return {"charge_balance_defect": defect,
                "charge_balance_relative_defect": defect / max(abs(mass), 1.0e-300)}


class EnstrophyBudget:
    """Enstrophy budget of each transport solve and the defect of the exact L2 identity.

    Records ``enstrophy_dissipation`` D_h and ``enstrophy_defect`` G_h (with its parts)
    of :func:`hybridge.diagnostics.hdg_identities.transport_enstrophy_terms`, for the
    unscaled drift b = beta/scale of the step and the new density and interior trace,
    and ``l2_budget_relative_defect``:

        SI-Euler (and the SI-BDF2 start-up):  |r1|^2 + |r1 - r0|^2 + 2 dt D + dt G - |r0|^2,
        SI-BDF2:  E^{n+1} + |delta^2 r|^2 + 4 dt D + 2 dt G - E^n,  E^n = |r^n|^2 + |2 r^n - r^{n-1}|^2,

    relative to the enstrophy-like norm on the right; round-off when every step is recorded.
    Also records ``rho_l1`` (volume quadrature).
    """

    def __init__(self, scheme: str, trace_basis: str, dt: float):
        if scheme not in {"si-euler", "si-bdf2"}:
            raise ValueError("the enstrophy budget is implemented for si-euler and si-bdf2")
        self.scheme, self.trace_basis, self.dt = str(scheme), trace_basis, float(dt)
        self.history: list = []  # accepted densities (copies), newest last

    def start(self, density) -> None:
        self.history = [density.copy(name="rho_budget_0")]

    def step(self, density, trace_reduced, beta, scale: float, startup: bool) -> dict[str, Any]:
        from hybridge.core.field_ops import field_linear_combination
        from hybridge.diagnostics.hdg_identities import (
            field_l1_quadrature, field_l2_squared, transport_enstrophy_terms)

        terms = transport_enstrophy_terms(beta, density, trace_reduced, trace_basis=self.trace_basis)
        terms = {key: value / float(scale) for key, value in terms.items()}  # D_h, G_h are linear in b
        row = {f"enstrophy_{key}": value for key, value in terms.items()}
        row["rho_l1"] = field_l1_quadrature(density)
        space, dt = density.space, self.dt
        if self.history:
            previous = self.history[-1]
            if self.scheme == "si-euler" or startup or len(self.history) < 2:
                difference = field_linear_combination(space, [(1.0, density), (-1.0, previous)])
                right = field_l2_squared(previous)
                left = (field_l2_squared(density) + field_l2_squared(difference)
                        + 2.0 * dt * terms["dissipation"] + dt * terms["defect"])
            else:
                older = self.history[-2]
                extrapolated_old = field_linear_combination(space, [(2.0, previous), (-1.0, older)])
                extrapolated_new = field_linear_combination(space, [(2.0, density), (-1.0, previous)])
                second = field_linear_combination(space, [(1.0, density), (-2.0, previous), (1.0, older)])
                right = field_l2_squared(previous) + field_l2_squared(extrapolated_old)
                left = (field_l2_squared(density) + field_l2_squared(extrapolated_new) + field_l2_squared(second)
                        + 4.0 * dt * terms["dissipation"] + 2.0 * dt * terms["defect"])
            row["l2_budget_relative_defect"] = (left - right) / max(abs(right), 1.0e-300)
        self.history = (self.history + [density.copy(name="rho_budget")])[-2:]
        return row


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
        **_forced_mass_balance(case, mass, effective_baseline_mass, time_value),
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
