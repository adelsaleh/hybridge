"""Guiding-center reporting helpers."""

from __future__ import annotations
import math
from typing import Any
from scripts.guiding_center.runtime.labels import run_label
from hybridge.runtime.precision import PRECISION
from scripts.guiding_center.cases.guiding_center_presets import GuidingCenterRunPreset
from scripts.guiding_center.runtime.configuration import _hybrid_startup_method, _verbosity_level
from scripts.guiding_center.runtime.models import GuidingCenterRunResult


def _format_metric(value: Any, fmt: str = ".3e") -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return format(number, fmt)


def _first_metric(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _print_linear_step_summary(config: GuidingCenterRunPreset, row: dict[str, Any]) -> None:
    """Print one balanced, machine-readable summary for the accepted linear stages."""
    if _verbosity_level(config) < 3 or str(row.get("phase", "step")) != "step":
        return

    step = int(row.get("step", 0))
    step_label = f"{step:05d}/{int(config.num_steps):05d}"
    coupled_wall = _format_metric(row.get("linear_step_wall_time"), ".4f")
    beta_wall = _format_metric(row.get("beta_build_time"), ".4f")
    trace_wall = _format_metric(row.get("potential_trace_update_time"), ".4f")
    lines = [
        "",
        (
            f"[gc:linear] step {step_label} | t={_format_metric(row.get('time'), '.6f')} "
            f"| coupled wall={coupled_wall}s | beta={beta_wall}s | trace={trace_wall}s"
        ),
    ]

    for prefix, label in (("transport", "transport"), ("poisson", "poisson")):
        wall = _format_metric(row.get(f"{prefix}_step_wall_time"), ".4f")
        hdg_total = _format_metric(row.get(f"{prefix}_time"), ".4f")
        assembly = row.get(f"{prefix}_step_time_assembly")
        phase_label = "asm"
        if prefix == "poisson" and float(row.get("poisson_time_rhs_assembly", 0.0) or 0.0) > 0.0:
            phase_label = "rhs"
        solve = _format_metric(row.get(f"{prefix}_step_time_solve"), ".4f")
        reconstruction = _format_metric(row.get(f"{prefix}_step_time_reconstruction"), ".4f")
        iterations = _first_metric(
            row,
            f"{prefix}_step_iterations",
            f"{prefix}_solver_iterations",
        )
        iteration_text = "n/a" if iterations is None or int(iterations) < 0 else str(int(iterations))
        relative = _first_metric(
            row,
            f"{prefix}_physical_rel_residual",
            f"{prefix}_solver_rel_residual",
        )
        parts = [
            f"  {label:<9} HDG={hdg_total}s",
            f"stage wall={wall}s",
            f"{phase_label}={_format_metric(assembly, '.4f')}s",
            f"solve={solve}s",
            f"recon={reconstruction}s",
            f"it={iteration_text}",
            f"true_rel={_format_metric(relative)}",
        ]
        stage_count = int(row.get(f"{prefix}_stage_count", 1))
        if stage_count != 1:
            parts.append(f"stages={stage_count}")
        attempts = row.get(f"{prefix}_amgx_attempt_count")
        if attempts is not None and int(attempts) > 1:
            parts.append(f"attempts={int(attempts)}")
        if prefix == "poisson":
            reuse = []
            if bool(row.get("poisson_detail_raw_assembly_operator_reused", 0.0)):
                reuse.append("operator")
            hierarchy_reused = bool(
                row.get("poisson_detail_solve_fb_hp_mg_hierarchy_reused", 0.0)
            ) or bool(row.get("poisson_detail_solve_amgx_hierarchy_reused", 0.0))
            if hierarchy_reused:
                reuse.append("hierarchy")
            if reuse:
                parts.append("reuse=" + "+".join(reuse))
        lines.append(" | ".join(parts))

    scheme_key = str(row.get("time_scheme")).replace("-", "_")
    if row.get("time_scheme") in {"h1-bdf3", "h2-bdf3"}:
        mode = (str(row.get(f"{scheme_key}_startup_method")) + " startup") if row.get(f"{scheme_key}_startup") else "BDF3"
        lines.append(
            f"  {row['time_scheme']:<9} mode={mode} | residuals={row.get('explicit_residual_count', 0)}"
            f" | residual={_format_metric(row.get('explicit_residual_time'), '.4f')}s"
            f" | predictor={_format_metric(row.get('density_predictor_time'), '.4f')}s"
            f" | trace projection={_format_metric(row.get('transport_trace_projection_time'), '.4f')}s"
        )

    if row.get("time_scheme") == "imex-ark3":
        lines.append(
            "  imex-ark3 embedded relative error="
            f"{_format_metric(row.get('imex_ark3_embedded_error_relative'))}"
            f" | operator assemblies/reuses={row.get('imex_ark3_operator_assemblies', 0)}/{row.get('imex_ark3_operator_reuses', 0)}"
            f" | residual={_format_metric(row.get('explicit_residual_time'), '.4f')}s"
            f" | Poisson tau={_format_metric(row.get('poisson_tau'))}"
            f" | tau retries={row.get('poisson_tau_retry_count', 0)}"
        )

    print("\n".join(lines), flush=True)


def _print_step_summary(config: GuidingCenterRunPreset, row: dict[str, Any]) -> None:
    if _verbosity_level(config) < 1:
        return
    step = int(row.get("step", 0))
    phase = str(row.get("phase", "step"))
    step_label = "initial" if phase == "initial" else f"{step:05d}/{int(config.num_steps):05d}"
    if _verbosity_level(config) >= 2:
        _print_diagnostics_block(config, row, step_label=step_label, phase=phase)
        return

    pieces = [
        f"[gc] step={step_label}",
        f"t={_format_metric(row.get('time'), '.6f')}",
        (f"circulation_drift={_format_metric(row['circulation_drift'])}"
         if "circulation_drift" in row else
         f"mass_rel_drift={_format_metric(row.get('mass_relative_drift'))}"),
        f"energy_rel_drift={_format_metric(row.get('energy_relative_drift'))}",
        (
            "rho=["
            f"{_format_metric(row.get('rho_min'))},"
            f"{_format_metric(row.get('rho_max'))}]"
        ),
        (
            "phi=["
            f"{_format_metric(row.get('phi_min'))},"
            f"{_format_metric(row.get('phi_max'))}]"
        ),
    ]
    beta_time = row.get("beta_build_time")
    if beta_time is not None and phase != "initial":
        pieces.append(f"beta={_format_metric(beta_time, '.3f')}s")
    poisson_time = _first_metric(row, "poisson_time", "poisson_time_total")
    if poisson_time is not None:
        pieces.append(f"poisson={_format_metric(poisson_time, '.3f')}s")
    transport_time = _first_metric(row, "transport_time", "transport_time_total")
    if transport_time is not None and phase != "initial":
        pieces.append(f"transport={_format_metric(transport_time, '.3f')}s")
    scheme_key = str(row.get("time_scheme")).replace("-", "_")
    if row.get("time_scheme") in {"h1-bdf3", "h2-bdf3"} and phase != "initial":
        if row.get(f"{scheme_key}_startup"):
            pieces.append(f"startup={row.get(f'{scheme_key}_startup_method')}")
        pieces.append(f"rhs={_format_metric(row.get('explicit_residual_time'), '.3f')}s")
    if row.get("time_scheme") == "imex-ark3" and phase != "initial":
        pieces.append(f"embedded_rel={_format_metric(row.get('imex_ark3_embedded_error_relative'))}")
        pieces.append(f"poisson_tau={_format_metric(row.get('poisson_tau'))}")
        pieces.append(f"tau_retries={row.get('poisson_tau_retry_count', 0)}")
        pieces.append(f"rhs={_format_metric(row.get('explicit_residual_time'), '.3f')}s")
    plot_time = row.get("plot_time")
    if plot_time:
        pieces.append(f"plot={_format_metric(plot_time, '.3f')}s")
    poisson_rel = row.get("poisson_solver_rel_residual")
    if poisson_rel is not None:
        pieces.append(f"p_rel={_format_metric(poisson_rel)}")
    transport_rel = row.get("transport_solver_rel_residual")
    if transport_rel is not None:
        pieces.append(f"t_rel={_format_metric(transport_rel)}")
    if row.get("rho_l2_error") is not None:
        pieces.append(f"rho_l2={_format_metric(row.get('rho_l2_error'))}")
    if row.get("phi_l2_error") is not None:
        pieces.append(f"phi_l2={_format_metric(row.get('phi_l2_error'))}")
    if row.get("diocotron_phi_eq_relative_l2") is not None:
        pieces.append(f"instability_l2={_format_metric(row.get('diocotron_phi_eq_l2'))}")
    print(" ".join(pieces), flush=True)


def _print_diagnostics_block(
        config: GuidingCenterRunPreset,
        row: dict[str, Any],
        *,
        step_label: str,
        phase: str,
) -> None:
    """Print accepted-state physics separately from backend solver logs."""
    lines = [
        "",
        "=" * 78,
        (
            "GUIDING-CENTER ACCEPTED-STATE DIAGNOSTICS"
            f" | step {step_label} | t={_format_metric(row.get('time'), '.6f')}"
        ),
        run_label(config),
        "-" * 78,
        "Conservation",
        f"  mass                                  {_format_metric(row.get('mass'), '.10e')}",
        f"  relative mass drift                   {_format_metric(row.get('mass_relative_drift'))}",
        f"  electrostatic energy (1/2 ||q||²)     {_format_metric(row.get('energy_from_q_l2'), '.10e')}",
        f"  relative energy drift                 {_format_metric(row.get('energy_relative_drift'))}",
        f"  electric-field norm ||q|| L2          {_format_metric(row.get('q_l2'), '.10e')}",
    ]
    lines.extend([
        f"  enstrophy (1/2 ||rho||²)              {_format_metric(row.get('enstrophy'), '.10e')}",
        f"  relative enstrophy drift              {_format_metric(row.get('enstrophy_relative_drift'))}",
    ])
    if row.get("rho_min_checked") is not None:
        lines.extend(["", "Positivity (no limiter)",
            f"  initial/endpoint status               {row.get('positivity_status')}",
            f"  checked minimum / cell-average min    {_format_metric(row.get('rho_min_checked'))} / {_format_metric(row.get('rho_cell_average_min'))}",
            f"  negative mass (volume quadrature)     {_format_metric(row.get('rho_negative_mass_quadrature'))}",
            f"  all-stage minimum this step           {_format_metric(row.get('positivity_stage_min'))}"])
    if row.get("diocotron_phi_mode_target_l2") is not None:
        lines.extend(["", "Polar Fourier potential diagnostics",
            f"  target mode amplitude L2              {_format_metric(row.get('diocotron_phi_mode_target_l2'))}",
            f"  axisymmetric departure L2             {_format_metric(row.get('diocotron_phi_axisymmetric_l2'))}",
            f"  potential harmonic ratio (2k/k)       {_format_metric(row.get('diocotron_phi_harmonic_ratio'))}"])
    if row.get("q_l2_postprocessed") is not None:
        lines.append(
            "  postprocessed electric-field norm L2      "
            f"{_format_metric(row.get('q_l2_postprocessed'), '.10e')}"
        )

    if row.get("diocotron_phi_eq_l2") is not None:
        lines.extend(
            [
                "",
                "Instability relative to equilibrium",
                f"  potential amplitude ||phi-phi_eq|| L2 {_format_metric(row.get('diocotron_phi_eq_l2'))}",
                f"  relative potential amplitude          {_format_metric(row.get('diocotron_phi_eq_relative_l2'))}",
                f"  potential difference Linf             {_format_metric(row.get('diocotron_phi_eq_linf'))}",
            ]
        )
    if row.get("diocotron_rho_eq_l2") is not None:
        lines.extend(
            [
                f"  density amplitude ||rho-rho_eq|| L2   {_format_metric(row.get('diocotron_rho_eq_l2'))}",
                f"  relative density amplitude            {_format_metric(row.get('diocotron_rho_eq_relative_l2'))}",
            ]
        )
    if row.get("diocotron_mode_1k_amplitude") is not None:
        mode = _format_metric(row.get("diocotron_mode_base"), ".0f")
        lines.extend(
            [
                (
                    f"  {f'normalized mode k={mode} amplitude':<38}"
                    f"{_format_metric(row.get('diocotron_mode_1k_amplitude'))}"
                ),
                f"  normalized mode 2k amplitude           {_format_metric(row.get('diocotron_mode_2k_amplitude'))}",
                f"  normalized mode 3k amplitude           {_format_metric(row.get('diocotron_mode_3k_amplitude'))}",
                f"  harmonic ratio (2k/k)                  {_format_metric(row.get('diocotron_harmonic_ratio'))}",
            ]
        )

    lines.extend(
        [
            "",
            "Field ranges",
            (
                "  density rho                           "
                f"[{_format_metric(row.get('rho_min'))}, {_format_metric(row.get('rho_max'))}]"
            ),
            (
                "  potential phi                        "
                f"[{_format_metric(row.get('phi_min'))}, {_format_metric(row.get('phi_max'))}]"
            ),
        ]
    )
    if row.get("rho_l2_error") is not None or row.get("phi_l2_error") is not None:
        lines.extend(
            [
                "",
                "Manufactured-solution errors",
                f"  density L2 / Linf                    {_format_metric(row.get('rho_l2_error'))} / {_format_metric(row.get('rho_linf_error'))}",
                f"  potential L2 / Linf                  {_format_metric(row.get('phi_l2_error'))} / {_format_metric(row.get('phi_linf_error'))}",
            ]
        )

    lines.extend(["", "Linear-solver checks"])
    if row.get("poisson_solver_rel_residual") is not None:
        lines.append(
            "  Poisson independently checked residual "
            f"{_format_metric(row.get('poisson_solver_rel_residual'))}"
        )
    if phase != "initial" and row.get("transport_solver_rel_residual") is not None:
        lines.append(
            "  transport independently checked residual "
            f"{_format_metric(row.get('transport_solver_rel_residual'))}"
        )

    scheme_key = str(row.get("time_scheme")).replace("-", "_")
    if row.get("time_scheme") in {"h1-bdf3", "h2-bdf3"} and phase != "initial":
        mode = (str(row.get(f"{scheme_key}_startup_method")) + " startup") if row.get(f"{scheme_key}_startup") else "BDF3"
        lines.extend([
            "", f"{row['time_scheme'].upper()} stages ({mode})",
            f"  transport / Poisson / residual       {row.get('transport_stage_count', 0)} / "
            f"{row.get('poisson_stage_count', 0)} / {row.get('explicit_residual_count', 0)}",
        ])
    if row.get("time_scheme") == "imex-ark3" and phase != "initial":
        lines.extend([
            "", "IMEX-ARK3 stages (ARK3(2)4L[2]SA)",
            f"  transport / Poisson / residual       {row.get('transport_stage_count', 0)} / "
            f"{row.get('poisson_stage_count', 0)} / {row.get('explicit_residual_count', 0)}",
            f"  embedded second-order difference    L2={_format_metric(row.get('imex_ark3_embedded_error_l2'))} "
            f"relative={_format_metric(row.get('imex_ark3_embedded_error_relative'))}",
            f"  transport assemblies / reuses       {row.get('imex_ark3_operator_assemblies', 0)} / "
            f"{row.get('imex_ark3_operator_reuses', 0)}",
            f"  Poisson tau / recovery retries      {_format_metric(row.get('poisson_tau'))} / "
            f"{row.get('poisson_tau_retry_count', 0)}",
            f"  rejected transport / Poisson stages {row.get('transport_rejected_stage_count', 0)} / "
            f"{row.get('poisson_rejected_stage_count', 0)}",
        ])
    lines.extend(["", "Phase timings"])
    timing_rows = [
        ("complete coupled linear step", row.get("linear_step_wall_time") if phase != "initial" else None),
        ("complete transport stage wall", row.get("transport_step_wall_time") if phase != "initial" else None),
        ("complete Poisson stage wall", row.get("poisson_step_wall_time") if phase != "initial" else None),
        ("beta construction", row.get("beta_build_time") if phase != "initial" else None),
        ("transport HDG solve", _first_metric(row, "transport_time", "transport_time_total") if phase != "initial" else None),
        ("Poisson HDG solve", _first_metric(row, "poisson_time", "poisson_time_total")),
    ]
    if phase == "initial":
        timing_rows.extend([
            ("first Poisson wall", row.get("first_poisson_wall_time")),
            ("first Poisson operator assembly", row.get("first_poisson_time_operator_assembly")),
            ("first Poisson native hierarchy", row.get("first_poisson_detail_solve_fb_hp_mg_setup_outer")),
            ("first Poisson Krylov", row.get("first_poisson_krylov_time")),
            ("first Poisson reconstruction", row.get("first_poisson_time_reconstruction")),
            ("reused initial-state Poisson wall", row.get("initial_poisson_wall_time")),
        ])
    elif row.get("poisson_time_rhs_assembly"):
        timing_rows.append(("Poisson cached RHS-only assembly", row.get("poisson_time_rhs_assembly")))
    timing_rows.extend([
        ("time scheme state and cache priming", row.get("time_scheme_setup_time", row.get("hybrid_bdf3_setup_time"))),
        ("explicit HDG residuals", row.get("explicit_residual_time")),
        ("density prediction / extrapolation", row.get("density_predictor_time")),
        ("transport guess trace projection", row.get("transport_trace_projection_time")),
        ("accepted potential trace", row.get("potential_trace_update_time")),
        ("plot update", row.get("plot_time") if row.get("plot_time") else None),
        ("accepted-state diagnostics", row.get("diagnostics_time")),
        ("post-Poisson application work", row.get("post_poisson_application_time")),
    ])
    for label, value in timing_rows:
        if value is not None:
            lines.append(f"  {label:<38} {_format_metric(value, '.5f')} s")
    if _verbosity_level(config) >= 3:
        lines.extend(
            [
                "  diagnostics: core                     "
                f"{_format_metric(row.get('diagnostics_core_time'), '.5f')} s",
                "  diagnostics: equilibrium potential    "
                f"{_format_metric(row.get('diagnostics_equilibrium_potential_time'), '.5f')} s",
                "  diagnostics: equilibrium density      "
                f"{_format_metric(row.get('diagnostics_equilibrium_density_time'), '.5f')} s",
                "  diagnostics: azimuthal modes           "
                f"{_format_metric(row.get('diagnostics_azimuthal_mode_time'), '.5f')} s",
            ]
        )
    lines.extend(["=" * 78, ""])
    print("\n".join(lines), flush=True)


def _print_run_summary(result: GuidingCenterRunResult) -> None:
    if _verbosity_level(result.config) < 1:
        return
    from hybridge.io.output import pretty_print_sections

    if not result.diagnostics:
        print(f"[gc] completed {result.config.num_steps:,} steps to "
              f"T={result.config.num_steps * result.config.dt:g}; field diagnostics disabled", flush=True)
        return
    final = result.diagnostics[-1]
    run_rows = [
        ("model / scheme", run_label(result.config), "s"),
        ("preset", result.preset_key, "s"),
        ("case", result.case_key, "s"),
        ("time scheme", result.config.time_scheme, "s"),
        ("precision", PRECISION, "s"),
        ("DG order", result.space.order, ",d"),
        ("triangles", result.mesh.num_tri, ",d"),
        ("steps", result.config.num_steps, ",d"),
        ("dt", result.config.dt, ".4e"),
    ]
    if result.config.time_scheme in {"h1-bdf3", "h2-bdf3"}:
        run_rows.append(("startup", _hybrid_startup_method(result.config), "s"))
    radial_power = result.config.case_params.get("p")
    if radial_power is not None:
        run_rows.insert(3, ("radial p", radial_power, ".4g"))
    final_rows = [
        ("time", final["time"], ".4e"),
        (("circulation drift", final["circulation_drift"], ".4e") if "circulation_drift" in final else
         ("relative mass drift", final["mass_relative_drift"], ".4e")),
        ("electrostatic energy", final["energy_from_q_l2"], ".4e"),
        ("relative energy drift", final["energy_relative_drift"], ".4e"),
    ]
    if final.get("diocotron_phi_eq_l2") is not None:
        final_rows.extend(
            [
                ("||phi - phi_eq|| L2", final["diocotron_phi_eq_l2"], ".4e"),
                ("relative equilibrium departure", final["diocotron_phi_eq_relative_l2"], ".4e"),
            ]
        )
    if final.get("diocotron_mode_1k_amplitude") is not None:
        final_rows.extend(
            [
                ("normalized k-mode amplitude", final["diocotron_mode_1k_amplitude"], ".4e"),
                ("2k/k harmonic ratio", final["diocotron_harmonic_ratio"], ".4e"),
            ]
        )
    if final.get("rho_l2_error") is not None:
        final_rows.append(("rho L2 error", final["rho_l2_error"], ".4e"))
    if final.get("phi_l2_error") is not None:
        final_rows.append(("phi L2 error", final["phi_l2_error"], ".4e"))
    output_rows = [
        ("Diagnostics CSV", str(result.csv_path), "s"),
        ("Diagnostics JSONL", str(result.jsonl_path), "s"),
        ("Every-step timings CSV", str(result.timings_csv_path), "s"),
        ("Every-step timings JSONL", str(result.timings_jsonl_path), "s"),
    ]
    output_rows = [entry for entry in output_rows if entry[1] != "None"]
    if result.terminal_log_path is not None:
        output_rows.append(("Terminal log", str(result.terminal_log_path), "s"))
    pretty_print_sections(
        [
            (
                "Run / mesh",
                run_rows,
            ),
            (
                "Final diagnostics",
                final_rows,
            ),
            (
                "Outputs",
                output_rows,
            ),
        ],
        title="Guiding-Center Run Summary",
    )

