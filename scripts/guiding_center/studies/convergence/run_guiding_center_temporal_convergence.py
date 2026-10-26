#!/usr/bin/env python3
"""Manufactured temporal convergence and fixed-space vortex-gas sensitivity."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from argparse import ArgumentParser
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.guiding_center.paths import AMGX_CONFIGS
from scripts.guiding_center.runtime.device import release_device_memory
from hybridge.diagnostics.errors import evaluate_hdg_scalar_error, evaluate_trace_error
from hybridge.hdg.condensation import expand_interior_trace
from scripts.guiding_center.cases.guiding_center_cases import case_definition_by_key
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime.runner import run_guiding_center_case

SCHEMES = ("si-euler", "predictor-corrector", "si-bdf2", "si-bdf3", "h1-bdf3", "h2-bdf3", "imex-ark3")
BASE_ERROR_KEYS = ("rho_l2_error", "rho_linf_error", "phi_l2_error", "phi_linf_error")
INTERIOR_ERROR_KEYS = ("rho_l2_interior_error", "phi_l2_interior_error")
TRACE_ERROR_KEYS = (tuple(f"{field}_trace_{kind}_error" for field in ("rho", "phi") for kind in ("l2", "projection"))
                    + ("rho_trace_flux_l2_error", "rho_trace_flux_projection_error"))
HDG_ERROR_KEYS = tuple(f"{field}_{norm}_error" for field in ("rho", "phi")
                       for norm in ("gradient_l2", "trace_mismatch", "hdg_h1"))
# Differences from a same-mesh reference run (temporal study with --reference-dt): the
# spatial error cancels, leaving the temporal error of the fully discrete scheme.
REFERENCE_ERROR_KEYS = ("rho_ref_l2_error", "phi_ref_l2_error", "rho_trace_ref_l2_error",
                        "rho_trace_flux_ref_l2_error", "phi_trace_ref_l2_error")
ERROR_KEYS = (BASE_ERROR_KEYS + HDG_ERROR_KEYS + INTERIOR_ERROR_KEYS + TRACE_ERROR_KEYS
              + REFERENCE_ERROR_KEYS)


def _parse_dts(raw: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("dts must be a comma-separated list of finite positive values")
    if len(set(values)) != len(values):
        raise ValueError("dts must not contain duplicates")
    return tuple(sorted(values, reverse=True))


def _selected_schemes(selection: str) -> tuple[str, ...]:
    if selection == "all":
        return SCHEMES
    if "," in selection:  # explicit list, e.g. "si-euler,si-bdf2,imex-ark3"
        chosen = tuple(name.strip() for name in selection.split(","))
        unknown = [name for name in chosen if name not in SCHEMES]
        if unknown:
            raise ValueError(f"unknown schemes {unknown}; choose from {SCHEMES}")
        return chosen
    # Preserve the original two-scheme comparison for existing commands.
    return SCHEMES[:2] if selection == "both" else (selection,)


def _pairwise_rates(rows: list[dict[str, Any]]) -> None:
    by_scheme: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scheme.setdefault(str(row["scheme"]), []).append(row)
    for scheme_rows in by_scheme.values():
        scheme_rows.sort(key=lambda row: float(row["dt"]), reverse=True)
        for index, row in enumerate(scheme_rows):
            for error_key in ERROR_KEYS:
                rate_key = error_key.replace("_error", "_rate")
                if index == 0 or row.get(error_key) is None or scheme_rows[index - 1].get(error_key) is None:
                    row[rate_key] = None
                    continue
                coarse = scheme_rows[index - 1]
                coarse_error = float(coarse[error_key])
                fine_error = float(row[error_key])
                if (coarse_error <= 0.0 or fine_error <= 0.0
                        or not math.isfinite(coarse_error) or not math.isfinite(fine_error)):
                    row[rate_key] = None
                else:
                    row[rate_key] = math.log(coarse_error / fine_error) / math.log(
                        float(coarse["dt"]) / float(row["dt"])
                    )


def _write_outputs(rows: list[dict[str, Any]], output_dir: Path, prefix: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{prefix}.csv"
    json_path = output_dir / f"{prefix}.json"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, json_path


def _print_table(rows: list[dict[str, Any]]) -> None:
    print("\nGuiding-center temporal convergence")
    print("scheme                 dt        steps   rho L2       rate    phi L2       rate")
    print("-" * 86)
    for row in rows:
        rho_rate = row.get("rho_l2_rate")
        phi_rate = row.get("phi_l2_rate")
        print(
            f"{row['scheme']:<22} {row['dt']:<9.3g} {row['num_steps']:<7d} "
            f"{row['rho_l2_error']:<12.4e} "
            f"{'-' if rho_rate is None else f'{rho_rate:.3f}':<7} "
            f"{row['phi_l2_error']:<12.4e} "
            f"{'-' if phi_rate is None else f'{phi_rate:.3f}':<7}"
        )

    if all("rho_hdg_h1_error" in row for row in rows):
        print("\nHDG H1 errors (L2 + physical gradient + 1/h_K face mismatch, in quadrature)")
        print("scheme                 dt        rho HDG H1   rate    phi HDG H1   rate")
        for row in rows:
            rates = [row.get(f"{field}_hdg_h1_rate") for field in ("rho", "phi")]
            rate_text = ["-" if value is None else f"{value:.3f}" for value in rates]
            print(f"{row['scheme']:<22} {row['dt']:<9.3g} "
                  f"{row['rho_hdg_h1_error']:<12.4e} {rate_text[0]:<7} "
                  f"{row['phi_hdg_h1_error']:<12.4e} {rate_text[1]:<7}")


def plot_convergence(rows: list[dict[str, Any]], output: Path, *, show: bool = False) -> Path:
    """Plot available L2/Linf and HDG error components, including older results."""
    import matplotlib.pyplot as plt

    norm_labels = {
        "l2": r"$L^2$ error", "linf": r"$L^\infty$ error",
        "gradient_l2": r"$\|\nabla_h(u_h-u)\|$",
        "trace_mismatch": r"$\sqrt{J_h}$ (error face term)",
        "hdg_h1": r"HDG $H^1$ error",
    }
    labels = {f"{field}_{norm}_error": f"${symbol}$: {label}"
              for norm, label in norm_labels.items() for field, symbol in (("rho", r"\rho"), ("phi", r"\phi"))}
    keys = [key for key in labels if any(row.get(key) is not None for row in rows)]
    if not keys:
        raise ValueError("no error metrics to plot")
    styles = {"si-euler": "o-", "predictor-corrector": "s-", "si-bdf2": "^-", "si-bdf3": "<-", "h1-bdf3": "d-", "h2-bdf3": "v-", "imex-ark3": "x-"}
    nrows = (len(keys) + 1) // 2
    figure, axes = plt.subplots(nrows, 2, figsize=(11, 3.5*nrows), squeeze=False, constrained_layout=True)
    for axis in axes.flat[len(keys):]:
        axis.set_visible(False)
    for axis, error_key in zip(axes.flat, keys):
        all_dts = []
        reference_scale = None
        for scheme in SCHEMES:
            selected = sorted(
                (row for row in rows if row["scheme"] == scheme and row.get(error_key) is not None
                 and math.isfinite(float(row[error_key])) and float(row[error_key]) > 0),
                key=lambda row: float(row["dt"]),
                reverse=True,
            )
            if not selected:
                continue
            dts = np.asarray([row["dt"] for row in selected], dtype=np.float64)
            errors = np.asarray([row[error_key] for row in selected], dtype=np.float64)
            axis.loglog(dts, errors, styles[scheme], label=scheme)
            all_dts.extend(dts.tolist())
            if reference_scale is None:
                reference_scale = float(errors[0])
        if all_dts and reference_scale is not None:
            refs = np.asarray(sorted(set(all_dts), reverse=True), dtype=np.float64)
            coarse_dt = float(refs[0])
            axis.loglog(refs, reference_scale * refs / coarse_dt, "k--", alpha=0.55, label=r"$O(\Delta t)$")
            axis.loglog(refs, reference_scale * (refs / coarse_dt) ** 2, "k:", alpha=0.65, label=r"$O(\Delta t^2)$")
            if any(row["scheme"] in {"si-bdf3", "h1-bdf3", "h2-bdf3", "imex-ark3"} for row in rows):
                axis.loglog(refs, reference_scale * (refs / coarse_dt) ** 3, "k-.", alpha=0.65, label=r"$O(\Delta t^3)$")
        axis.set_xlabel(r"$\Delta t$")
        axis.set_ylabel(labels[error_key])
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    if show:
        plt.show()
    plt.close(figure)
    return output


def _full_traces(result) -> dict[str, tuple[Any, Any, str]]:
    """Accepted final traces with the prescribed boundary values filled in: {name: (field, trace, basis)}."""
    config = result.config
    case = case_definition_by_key(config.case).build(**config.case_params)
    time_value = float(result.diagnostics[-1]["time"])
    traces = {}
    for name, field, reduced, basis, boundary in (
        ("rho", result.final_density, result.final_density_trace_reduced,
         config.transport_trace_basis or config.trace_basis, case.density_boundary_at(time_value)),
        ("phi", result.final_potential, result.final_potential_trace_reduced,
         config.poisson_trace_basis or config.trace_basis, case.potential_boundary_at(time_value)),
    ):
        traces[name] = (field, expand_interior_trace(field.space, reduced, boundary, trace_basis=basis).ravel(),
                        basis)
    return traces


def _manufactured_hdg_errors(result, error_quad_1d: int | None = None) -> dict[str, float | str]:
    """Compare accepted fields/traces with the analytic solution at final time.

    ``error_quad_1d``: collapsed Gauss rule (volume) and Gauss-Legendre rule (edges) with that
    many points per direction for the L2 and trace norms, for exact solutions of limited
    smoothness; ``None`` keeps the spaces' own rules.
    """
    config = result.config
    case = case_definition_by_key(config.case).build(**config.case_params)
    time_value = float(result.diagnostics[-1]["time"])
    errors = {"hdg_jump_weight": "1/h_K; h_K=element diameter",
              "hdg_trace_faces": "all element sides, including prescribed boundaries"}
    if error_quad_1d is not None:
        from hybridge.diagnostics.errors import evaluate_scalar_error

        errors["error_quad_1d"] = int(error_quad_1d)
        for name, field, exact in (("rho", result.final_density, case.exact_density_at(time_value)),
                                   ("phi", result.final_potential, case.exact_potential_at(time_value))):
            errors[f"{name}_l2_error"] = float(evaluate_scalar_error(
                field, exact, volume_quad_1d=int(error_quad_1d)).metrics.l2)
    for field_name, field, reduced, basis, exact, exact_gradient, boundary in (
        ("rho", result.final_density, result.final_density_trace_reduced,
         config.transport_trace_basis or config.trace_basis, case.exact_density_at(time_value),
         case.exact_density_gradient_at(time_value), case.density_boundary_at(time_value)),
        ("phi", result.final_potential, result.final_potential_trace_reduced,
         config.poisson_trace_basis or config.trace_basis, case.exact_potential_at(time_value),
         case.exact_potential_gradient_at(time_value), case.potential_boundary_at(time_value)),
    ):
        if exact is None or exact_gradient is None or boundary is None or reduced is None:
            raise ValueError("manufactured HDG error requires analytic gradients and accepted numerical traces")
        # Expand only prescribed boundaries, keeping accepted interior traces
        # (including the PC endpoint extrapolation) on their resident backend.
        # The potential lives in its own space when the Poisson degree differs (offset -1).
        full = expand_interior_trace(field.space, reduced, boundary, trace_basis=basis)
        metrics = evaluate_hdg_scalar_error(field, full.ravel(), exact, exact_gradient, trace_basis=basis)
        # Numerical trace on the computed (interior) edges vs the exact trace and its
        # edgewise L2 projection, h-weighted: (sum_F h_F ||uhat - u||_F^2)^(1/2).
        trace_metrics = evaluate_trace_error(field.space, full.ravel(), exact, trace_basis=basis,
                                             edge_quad_1d=error_quad_1d)
        if field_name == "rho" and getattr(case, "exact_flux", None) is not None:
            # The transport trace only enters the flux through |u.n|: also report the
            # flux-weighted norm (sum_F h_F int_F |u.n| (rhohat - rho)^2)^(1/2), exact drift u
            # (skipped for cases without an analytic flux).
            def drift_normal(x, y, nx, ny):
                qx, qy = case.exact_flux(x, y, time_value)  # q = -grad(phi)
                return abs(-qy * nx + qx * ny)  # u = q^perp = (-q_y, q_x)

            weighted = evaluate_trace_error(field.space, full.ravel(), exact, trace_basis=basis,
                                            face_weight=drift_normal, edge_quad_1d=error_quad_1d)
            errors.update({"rho_trace_flux_l2_error": weighted.l2,
                           "rho_trace_flux_projection_error": weighted.projection_l2,
                           "rho_trace_flux_exact_norm": weighted.exact_l2})
        errors.update({f"{field_name}_gradient_l2_error": metrics.gradient_l2,
                       f"{field_name}_trace_mismatch_error": metrics.trace_mismatch,
                       f"{field_name}_hdg_h1_error": metrics.hdg_h1,
                       f"{field_name}_hdg_diagnostics_backend": metrics.backend,
                       f"{field_name}_trace_l2_error": trace_metrics.l2,
                       f"{field_name}_trace_projection_error": trace_metrics.projection_l2,
                       f"{field_name}_trace_exact_norm": trace_metrics.exact_l2})
    return errors


def _final_state(result) -> dict[str, Any]:
    """Host copies of the final fields and full traces (reference for the temporal study)."""
    traces = _full_traces(result)
    return {name: (np.array(field.coeffs, dtype=np.float64, copy=True),
                   np.asarray(trace.get() if hasattr(trace, "get") else trace, dtype=np.float64).copy())
            for name, (field, trace, _) in traces.items()}


def save_final_fields(result, path: Path, **metadata) -> Path | None:
    """Write the final state of one run to ``path`` (npz) and return it; ``None`` if it could not be saved.

    Holds what any later error evaluation needs: the DG coefficients of rho_h and phi_h (element-local, in
    the run's basis), their full traces (prescribed boundaries expanded) with the trace bases, the mesh
    (node coordinates, triangles, edges, interior edges), the final time, degrees and case parameters, and
    ``metadata`` (scheme, dt, mesh size and family, drift). A failure is reported, never raised.
    """
    try:
        def host(array):
            array = array.get() if hasattr(array, "get") else array
            return np.asarray(array)

        config = result.config
        mesh = result.mesh
        traces = _full_traces(result)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            time=float(result.diagnostics[-1]["time"]), case=str(config.case),
            case_params=json.dumps({key: float(value) for key, value in dict(config.case_params).items()}),
            order=int(result.final_density.space.order), poisson_order=int(result.final_potential.space.order),
            basis_type=str(result.final_density.space.quad_data.basis_type),
            node_coords=host(mesh.node_coords), triangles=host(mesh.triangles), edges=host(mesh.edges),
            interior_edges=host(mesh.int_edges_inds),
            density=host(result.final_density.coeffs), potential=host(result.final_potential.coeffs),
            **{f"{name}_trace": host(trace) for name, (_, trace, _) in traces.items()},
            **{f"{name}_trace_basis": str(basis) for name, (_, _, basis) in traces.items()},
            **{key: (json.dumps(value) if isinstance(value, (dict, list, tuple)) else value)
               for key, value in metadata.items() if value is not None},
        )
        return path
    except Exception as exc:  # noqa: BLE001 - never lose a finished run to its archive
        print(f"[gc] final fields not saved: {exc!r}", flush=True)
        return None


def _reference_errors(result, reference: dict[str, Any]) -> dict[str, float]:
    """Differences from the same-mesh reference: fields in L2, traces in the skeleton norms."""
    from hybridge.core.space import DGField
    from hybridge.diagnostics.errors import evaluate_scalar_error

    config = result.config
    case = case_definition_by_key(config.case).build(**config.case_params)
    time_value = float(result.diagnostics[-1]["time"])
    zero = lambda x, y: 0.0 * x
    errors = {}
    for name, (field, trace, basis) in _full_traces(result).items():
        coeffs, ref_trace = reference[name]
        difference = DGField(field.space, np.asarray(field.coeffs, dtype=np.float64) - coeffs)
        errors[f"{name}_ref_l2_error"] = float(evaluate_scalar_error(difference, zero, backend="host").metrics.l2)
        host_trace = np.asarray(trace.get() if hasattr(trace, "get") else trace, dtype=np.float64)
        errors[f"{name}_trace_ref_l2_error"] = evaluate_trace_error(
            field.space, host_trace - ref_trace, zero, trace_basis=basis).l2
        if name == "rho":
            def drift_normal(x, y, nx, ny):
                qx, qy = case.exact_flux(x, y, time_value)
                return abs(-qy * nx + qx * ny)

            errors["rho_trace_flux_ref_l2_error"] = evaluate_trace_error(
                field.space, host_trace - ref_trace, zero, trace_basis=basis, face_weight=drift_normal).l2
    return errors


def _max_drift_speed(case_params: dict[str, float], case_key: str = "rho_helm_wave",
                     final_time: float = 0.0) -> float:
    """Return max |u| of the exact drift over the domain and [0, final_time].

    Helmholtz wave: max_s |(A kx s, A ky s - U)| (time independent). Other cases: sampled on a
    401 x 401 grid of the case's polygon (mesh independent) at 17 instants.
    """
    if case_key != "rho_helm_wave":
        from hybridge.core.geometry import PolygonDomain

        case = case_definition_by_key(case_key).build(**case_params)
        domain = PolygonDomain(case.parameters["geometry"]["vertices"])
        (x0, y0), (x1, y1) = domain.bounds
        x, y = np.meshgrid(np.linspace(x0, x1, 401), np.linspace(y0, y1, 401))
        points = np.column_stack((x.ravel(), y.ravel()))
        points = points[domain.contains(points)]
        speed = 0.0
        for t in np.linspace(0.0, float(final_time), 17):
            qx, qy = case.exact_flux(points[:, 0], points[:, 1], float(t))
            speed = max(speed, float(np.max(np.hypot(qx, qy))))
        return speed
    kx = float(case_params.get("kx", 1.0))
    ky = float(case_params.get("ky", 1.0))
    velocity = float(case_params.get("U", 1.0))
    amplitude = float(case_params.get("A", 1.0))
    s = np.linspace(-1.0, 1.0, 20001)
    return float(np.max(np.hypot(amplitude * kx * s, amplitude * ky * s - velocity)))


def _min_edge_length(mesh) -> float:
    """Return the shortest triangle edge of a DG mesh."""
    xy = np.asarray(mesh.node_coords, dtype=np.float64)
    edges = np.asarray(mesh.edges)
    return float(np.min(np.linalg.norm(xy[edges[:, 0]] - xy[edges[:, 1]], axis=1)))


# Spatial-study mesh families (``--mesh-family``): "gmsh" is Gmsh's Frontal-Delaunay (the default; on the
# rectangle a lattice of equilateral triangles whose orientation changes with the target size),
# "structured" the uniform nx x nx squares of (-1,1)^2 split along their (-1,1) diagonal (rectangle_mesh,
# nx = round(2 / h); rectangle cases only), and "unstructured" Gmsh's Delaunay algorithm (Mesh.Algorithm 5),
# which is no lattice (edge-lattice coherence below 0.05 in (-0.6,0.6)^2, equilateral share about 0.1,
# about 14% more triangles than "gmsh" at the same target size).
MESH_FAMILIES = ("gmsh", "structured", "unstructured")
UNSTRUCTURED_GMSH_ALGORITHM = 5


def mesh_family_gmsh_algorithm(mesh_family: str) -> int | None:
    """Gmsh ``Mesh.Algorithm`` of a Gmsh mesh family (``None``: Gmsh's default, Frontal-Delaunay)."""
    if mesh_family not in MESH_FAMILIES:
        raise ValueError(f"mesh_family must be one of {MESH_FAMILIES}")
    return UNSTRUCTURED_GMSH_ALGORITHM if mesh_family == "unstructured" else None


def _mesh_family_options(mesh_family: str, mesh_size: float, case: str) -> dict:
    """Run-preset fields selecting the mesh of one family (``MESH_FAMILIES``)."""
    if mesh_family == "structured":
        cells = int(round(2.0 / float(mesh_size)))
        return {"domain": "structured-rectangle", "nx": cells, "ny": cells}
    options = {"domain": getattr(case_definition_by_key(case), "default_domain", "rectangle")}
    algorithm = mesh_family_gmsh_algorithm(mesh_family)
    if algorithm is not None:
        options["gmsh_algorithm"] = algorithm
    return options


def mesh_family_of(config) -> str:
    """Mesh family of a run preset (recorded in the result rows)."""
    if config.domain == "structured-rectangle":
        return "structured"
    return "unstructured" if config.gmsh_algorithm == UNSTRUCTURED_GMSH_ALGORITHM else "gmsh"


def _manufactured_config(base, *, scheme: str, dt: float, num_steps: int, mesh_size: float, order: int,
                         case_params: dict[str, float], poisson_tau: float, output_path: Path,
                         run_prefix: str, verbosity: int, backend: str = "device",
                         electric_field: str = "raw", poisson_order_offset: int = 0,
                         device_profile: str = "csr", case: str = "rho_helm_wave",
                         mesh_family: str = "gmsh", diagnostics_every: int | None = None,
                         solver_rtol: float | None = None, exact_drift_order_offset: int | None = None,
                         exact_drift_mode: str | None = None):
    """Configuration of one manufactured run (fixed Poisson tau, conflict-averaged upwind).

    ``diagnostics_every``: steps between recorded diagnostics (and verbosity-1 step summaries);
    default: the final step only. ``solver_rtol``: relative tolerance of the device Poisson and
    transport AMGX solves (default 1e-11), with an absolute tolerance of a tenth of it.
    ``exact_drift_mode`` (``electric_field="exact"``): ``"pointwise"`` (the preset's default: the exact
    drift at the transport's quadrature points) or ``"projected"`` (the L2 projection of the exact flux
    onto DG(p+s)); ``exact_drift_order_offset``: that degree offset s (default: the preset's, 2), which
    also sets the projected flux the diagnostics read in pointwise mode.

    ``backend="device"``: raw-CUDA assembly and AMGX (FP64, rtol 1e-11).
    ``backend="host"``: Numba assembly and PyPardiso direct solves on every available core.
    """
    backend = _resolve_backend(backend, mesh_size, scheme)
    if backend not in {"device", "host"}:
        raise ValueError("backend must be 'auto', 'device' or 'host'")
    if electric_field not in ELECTRIC_FIELDS:
        raise ValueError(f"electric_field must be one of {ELECTRIC_FIELDS}")
    compatible = electric_field in COMPATIBLE_FIELDS
    recovered = electric_field not in {"raw", "exact", *COMPATIBLE_FIELDS}
    field_options = dict(
        poisson_order_offset=-1 if recovered else int(poisson_order_offset),
        # "exact": the case's exact drift, pointwise or projected onto DG(p+s) (runtime/exact_drift.py).
        transport_electric_field=("compatible" if compatible else "postprocessed" if recovered
                                  else "exact" if electric_field == "exact" else "raw"),
        compatible_potential=COMPATIBLE_FIELDS.get(electric_field, "auto"),
        poisson_hdg_postprocess="flux" if recovered else "none",
        poisson_flux_postprocess_space=electric_field if recovered else "RT_projection",
        poisson_flux_postprocess_every=0,
    )
    common = dict(
        **field_options,
        case=case,
        case_params=dict(case_params),
        # Domain and Gmsh algorithm of the mesh family (MESH_FAMILIES).
        **_mesh_family_options(mesh_family, mesh_size, case),
        mesh_size=float(mesh_size),
        order=int(order),
        dt=float(dt),
        num_steps=num_steps,
        time_scheme=scheme,
        poisson_tau=float(poisson_tau),
        **({} if exact_drift_order_offset is None else {"exact_drift_order_offset": int(exact_drift_order_offset)}),
        **({} if exact_drift_mode is None else {"exact_drift_mode": str(exact_drift_mode)}),
        poisson_tau_max_retries=0,
        # Explicit: never let a face DOF be both inflow and outflow.
        transport_advection_stabilization="conflict-averaged-upwind",
        transport_boundary_mode="eliminate",
        plot_every=0,
        diagnostics_every=num_steps if diagnostics_every is None else max(1, int(diagnostics_every)),
        diagnostics_dir=str(output_path / "runs"),
        diagnostics_prefix=run_prefix,
        verbosity=int(verbosity),
    )
    if backend == "host":
        if scheme == "imex-ark3":
            # Host assembly is Numba only; ARK3 operator reuse has no Numba path.
            raise ValueError("IMEX-ARK3 runs on the device only (no Numba transport path for operator reuse)")
        return replace(
            base, **common,
            poisson_assembly_backend="numba",
            poisson_local_backend="numba",
            # Schur caches win from p=4 and lose at p=2
            # (docs/research/solver_studies/numba_diffusion_schur_2026_09_25.md).
            poisson_cache_local_factors="schur-lu" if int(order) >= 4 else "none",
            poisson_solver="pypardiso",
            poisson_preconditioner=None,
            poisson_scale_system=False,
            transport_assembly_backend="numba",
            transport_solver="pypardiso",
            transport_preconditioner=None,
            transport_scale_system=False,
            transport_materialize_host_system=True,
            transport_materialize_host_solution=True,
            transport_retry_policy="none",
        )
    rtol = 1.0e-11 if solver_rtol is None else float(solver_rtol)
    config = replace(
        base,
        **common,
        poisson_assembly_backend="raw-cuda",
        # This option selects the CPU fallback only (Numba, never NumPy). Raw-CUDA
        # uses device assembly and reconstruction independently.
        poisson_local_backend="numba",
        # The manufactured boundary is nonzero and time dependent;
        # the raw LU cache supports its eliminated trace contribution.
        poisson_cache_local_factors="schur-lu",
        poisson_solver="amgx",
        poisson_solver_rtol=rtol,
        poisson_solver_atol=0.1 * rtol,
        poisson_raw_matrix_format="csr",
        transport_assembly_backend="raw-cuda",
        transport_solver="amgx",
        transport_solver_rtol=rtol,
        transport_solver_atol=0.1 * rtol,
        transport_raw_local_assembly="fused",
        transport_raw_lu_mode="coop",
        transport_raw_matrix_format="csr",
        transport_materialize_host_system=False,
        transport_materialize_host_solution=False,
        transport_initial_guess="initial-density-trace",
        transport_retry_policy="amgx-robust",
    )
    if mesh_family not in MESH_FAMILIES:
        raise ValueError(f"mesh_family must be one of {MESH_FAMILIES}")
    if device_profile == "auto":
        device_profile = _auto_device_profile(order, mesh_size)
    if device_profile not in DEVICE_PROFILES:
        raise ValueError(f"device_profile must be one of {DEVICE_PROFILES} or 'auto'")
    if device_profile in {"bsr", "bsr-modal", "fbhpmg", "bsr-bsrpoisson"}:
        # Native face-block BSR transport matrix, same AMGX solver.
        config = replace(config, transport_raw_matrix_format="bsr")
    if device_profile in {"csr-bsrpoisson", "bsr-bsrpoisson"}:
        # Hybrid face-BSR Poisson: nodal trace, coefficient-exact scalar AMG hierarchy
        # (classical_bsr_hierarchy=scalar_expand), cuSPARSE generic BSR SpMV, scalar norm
        # (docs/research/solver_studies/classical_amg_bsr_2026_08.md).
        config = replace(config, poisson_raw_matrix_format="bsr", poisson_trace_basis="legacy-lagrange",
                         poisson_amgx_config_path=str(AMGX_CONFIGS
                                                      / "diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive_abs_scalar_expand_bsr.json"))
    if device_profile == "bsr-modal":
        config = replace(config, poisson_trace_basis="legendre-modal")
    if device_profile == "fbhpmg":
        # Native reusable face-block hp-multigrid PCG for Poisson (production disk presets).
        config = replace(config, poisson_solver="fb-hp-mg-pcg", poisson_scale_system=False,
                         poisson_trace_basis="legendre-modal", poisson_raw_matrix_format="bsr",
                         poisson_cache_local_factors="schur-cholesky")
    return config


def _manufactured_row(result, *, scheme: str, dt: float, final_time: float, num_steps: int,
                      mesh_size: float, order: int, case_params: dict[str, float],
                      poisson_tau: float, wall_time: float, backend: str = "device",
                      error_quad_1d: int | None = None) -> dict[str, Any]:
    """Collect final-time errors, invariants and the Courant number of one run."""
    config = result.config
    final = result.diagnostics[-1]
    min_edge = _min_edge_length(result.mesh)
    step_times = _step_wall_times(result.config)
    steady = step_times[2:] if len(step_times) > 3 else step_times
    speed = _max_drift_speed(case_params, config.case, final_time)
    return {
        "case": config.case,
        "scheme": scheme,
        "dt": float(dt),
        "final_time": float(final_time),
        "num_steps": num_steps,
        "mesh_size": float(mesh_size),
        "order": int(order),
        "triangles": int(result.mesh.num_tri),
        # Reference triangle (-1,-1),(1,-1),(-1,1) has area 2.
        "domain_area": float(2.0 * np.sum(np.abs(np.asarray(result.mesh.aff_jacs)))),
        "min_edge": min_edge,
        "max_speed": speed,
        "poisson_tau": float(poisson_tau),
        "backend": _resolve_backend(backend, mesh_size, scheme),
        "device_profile": _device_profile_label(config) if _resolve_backend(backend, mesh_size, scheme) != "host" else "-",
        "mesh_family": mesh_family_of(config),
        "poisson_matrix_format": config.poisson_raw_matrix_format,
        "transport_matrix_format": config.transport_raw_matrix_format,
        "poisson_trace_basis": config.poisson_trace_basis or config.trace_basis,
        "transport_trace_basis": config.transport_trace_basis or config.trace_basis,
        "solver_rtol": (float(config.poisson_solver_rtol)
                        if _resolve_backend(backend, mesh_size, scheme) != "host" else "-"),
        "transport_stabilization": str(result.config.transport_advection_stabilization),
        "electric_field": (config.poisson_flux_postprocess_space
                           if config.transport_electric_field == "postprocessed"
                           else _compatible_field_label(config)
                           if config.transport_electric_field == "compatible"
                           else _exact_field_label(config)
                           if config.transport_electric_field == "exact" else "raw"),
        "exact_drift_mode": (config.exact_drift_mode if config.transport_electric_field == "exact" else "-"),
        "poisson_order": int(order) + int(config.poisson_order_offset),
        "courant": float(dt) * speed / min_edge,
        "courant_explicit_dg": float(dt) * speed / min_edge * (2 * int(order) + 1),
        "wall_time_s": float(wall_time),
        # Steady per-step cost: median linear-step wall time, first two steps
        # (start-up scheme, solver setup, JIT) excluded.
        "step_time_median_s": float(np.median(steady)) if steady else None,
        "step_time_mean_s": float(np.mean(steady)) if steady else None,
        "steady_time_s": float(np.median(steady)) * num_steps if steady else None,
        "element_unknowns": int(result.mesh.num_tri) * (int(order) + 1) * (int(order) + 2) // 2,
        "trace_unknowns": int(result.mesh.num_edg) * (int(order) + 1),
        "poisson_trace_unknowns": int(result.mesh.num_edg) * (int(order) + int(config.poisson_order_offset) + 1),
        **{f"case_{key}": float(value) for key, value in case_params.items()},
        **{key: float(final[key]) for key in BASE_ERROR_KEYS},
        # Overrides the L2 errors with the enriched rule when error_quad_1d is set.
        **_manufactured_hdg_errors(result, error_quad_1d),
        **(_interior_errors(result) if config.case == "rho_helm_wave" else {}),
        **_accurate_errors(result),
        "mass_relative_drift": None if final.get("mass_relative_drift") is None else float(final["mass_relative_drift"]),
        "q_l2_relative_drift": None if final.get("q_l2_relative_drift") is None else float(final["q_l2_relative_drift"]),
    }


INTERIOR_HALF_WIDTH = 0.6


def _interior_errors(result) -> dict[str, float]:
    """L2 errors restricted to [-a, a]^2 (a = INTERIOR_HALF_WIDTH), away from the inflow boundary."""
    from hybridge.diagnostics.errors import evaluate_scalar_error

    config = result.config
    case = case_definition_by_key(config.case).build(**config.case_params)
    time_value = float(result.diagnostics[-1]["time"])
    half = INTERIOR_HALF_WIDTH

    def inside(x, y):
        xp = np if isinstance(x, np.ndarray) else __import__("cupy")
        return ((xp.abs(x) <= half) & (xp.abs(y) <= half)).astype(x.dtype)

    errors = {}
    for name, field, exact in (("rho", result.final_density, case.exact_density_at(time_value)),
                               ("phi", result.final_potential, case.exact_potential_at(time_value))):
        errors[f"{name}_l2_interior_error"] = float(evaluate_scalar_error(field, exact, weight=inside).metrics.l2)
    return errors


def accurate_error_points(order: int) -> int:
    """Collapsed-Gauss points per direction of the accurate error rule for a degree-``order`` field.

    The spaces' own volume rule is exact for degree 2p only. The squared error (u_h - u)^2 has leading
    degree 2p + 2, and that rule underestimates its integral: by a factor of about 3 in the norm of an L2
    projection error at p = 2, 1.7 at p = 4, 2 at p = 6 (one lattice triangle, k = 8 pi). With 2p + 2 points
    per direction (exact for degree 4p + 2) the relative error of that norm is below 1e-7 for p <= 5.
    """
    return 2 * int(order) + 2


def _accurate_errors(result) -> dict[str, Any]:
    """Final-time L2 errors (Omega and, for the travelling wave, the interior square) with the accurate rule.

    Recorded next to the spaces'-rule errors (``*_accurate`` keys); a failure is reported, never raised, so
    that a finished run is not lost to a diagnostic.
    """
    from hybridge.diagnostics.errors import evaluate_scalar_error

    config = result.config
    out: dict[str, Any] = {}
    try:
        case = case_definition_by_key(config.case).build(**config.case_params)
        time_value = float(result.diagnostics[-1]["time"])
        half = INTERIOR_HALF_WIDTH

        def inside(x, y):
            xp = np if isinstance(x, np.ndarray) else __import__("cupy")
            return ((xp.abs(x) <= half) & (xp.abs(y) <= half)).astype(x.dtype)

        for name, field, exact in (("rho", result.final_density, case.exact_density_at(time_value)),
                                   ("phi", result.final_potential, case.exact_potential_at(time_value))):
            if exact is None:
                continue
            points = accurate_error_points(field.space.order)
            out[f"{name}_accurate_quad_1d"] = points
            out[f"{name}_l2_error_accurate"] = float(
                evaluate_scalar_error(field, exact, volume_quad_1d=points).metrics.l2)
            if config.case == "rho_helm_wave":
                out[f"{name}_l2_interior_error_accurate"] = float(
                    evaluate_scalar_error(field, exact, weight=inside, volume_quad_1d=points).metrics.l2)
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        print(f"[gc] accurate error rule skipped: {exc!r}", flush=True)
        out["accurate_error_failure"] = repr(exc)
    return out


def _step_wall_times(config) -> list[float]:
    """Per-step linear wall times written by the runner's timing JSONL."""
    path = Path(config.diagnostics_dir) / f"{config.diagnostics_prefix}_timings.jsonl"
    if not path.exists():
        return []
    times = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record.get("linear_step_wall_time") is not None:
            times.append(float(record["linear_step_wall_time"]))
    return times


# Host/device crossover measured for this case (SI-BDF2, p=6, 10 steps, warm caches):
# 944 triangles host 2.1 s vs device 2.3 s; 3,704 triangles host 7.3 s vs device 2.9 s
# (run_outputs/article/convergence/backend_timing/, 2026-10-07).
HOST_MIN_MESH_SIZE = 0.1
# raw: rotated LDG-H flux of the degree-(p + offset) Poisson solve;
# RT_projection / l2_closest: Poisson in DG(p-1), flux recovered into DG(p).
# compatible: -grad of the Oswald-averaged potential (H(div), divergence-free, wall-tangent drift);
# "compatible" averages the p+1 postprocessed potential on every backend (V2; raw-CUDA Poisson
# solves postprocess on the device), "compatible-primal" always averages the HDG potential.
COMPATIBLE_FIELDS = {"compatible": "postprocessed", "compatible-primal": "primal"}
ELECTRIC_FIELDS = ("raw", "RT_projection", "l2_closest", "exact", *COMPATIBLE_FIELDS)


def _compatible_field_label(config) -> str:
    """Report the resolved compatible-drift potential of a configuration."""
    from scripts.guiding_center.runtime.compatible_drift import compatible_potential_source

    source = compatible_potential_source(config.compatible_potential, config.poisson_assembly_backend)
    return "compatible" if source == "postprocessed" else "compatible-primal"


def _exact_field_label(config) -> str:
    """Report the exact drift of a configuration: ``exact`` (pointwise) or ``exact_p+<s>`` (projected)."""
    if getattr(config, "exact_drift_mode", "pointwise") == "pointwise":
        return "exact"
    return f"exact_p+{config.exact_drift_order_offset}"
# Device kernels/solvers: csr = raw-CUDA CSR + AMGX for both solves; bsr = face-BSR
# transport; bsr-modal = bsr with a Legendre-modal (instead of nodal Lagrange) Poisson trace;
# fbhpmg = face-BSR transport + native FB-HP-MG-PCG Poisson (zero Dirichlet data only).
# csr-bsrpoisson / bsr-bsrpoisson = CSR / face-BSR transport + hybrid face-BSR Poisson (nodal trace,
# scalar_expand AMG hierarchy).
DEVICE_PROFILES = ("csr", "bsr", "bsr-modal", "fbhpmg", "csr-bsrpoisson", "bsr-bsrpoisson")


def _auto_device_profile(order: int, mesh_size: float) -> str:
    """Face-BSR transport (recorded evidence; equivalent to CSR within noise at this case's sizes).

    Transport AMGX solve times per stage (IMEX-ARK3, RTX PRO 5000, 2026-10-09), CSR vs BSR:
    4.56 vs 4.10 ms (p=3, 14,776 triangles), 7.64 vs 7.56 ms (p=2, 59,334), 4.82 vs 4.81 ms
    (p=4, 14,776); the transport stages are 6-25% of a step, so storage changes a step by at most
    ~1-2 ms (run_outputs/article/convergence/profile_timing/). Step-level CSR/BSR differences seen
    in non-interleaved passes on the shared machine were load drift. Larger meshes favour BSR
    except at p=2 (docs/backends/advection_bsr_benchmark_20260824.md). Poisson stays scalar CSR +
    AMGX: the hybrid face-BSR Poisson path fails with nonzero Dirichlet data in this runner.
    """
    return "bsr"


def _device_profile_label(config) -> str:
    """Name the device profile a resolved configuration corresponds to."""
    if config.poisson_solver == "fb-hp-mg-pcg":
        return "fbhpmg"
    if config.poisson_trace_basis == "legendre-modal":
        return "bsr-modal"
    transport = "bsr" if config.transport_raw_matrix_format == "bsr" else "csr"
    return f"{transport}-bsrpoisson" if config.poisson_raw_matrix_format == "bsr" else transport


def _resolve_backend(backend: str, mesh_size: float, scheme: str = "") -> str:
    """Pick the faster path: host below ~1000 triangles, device above (measured, p=6).

    IMEX-ARK3 operator reuse has no Numba transport path, and host assembly is
    Numba only, so ARK3 always runs on the device.
    """
    if backend == "auto":
        if scheme == "imex-ark3":
            return "device"
        return "host" if float(mesh_size) >= HOST_MIN_MESH_SIZE else "device"
    return backend


def _run_case(config, run_prefix: str, backend: str):
    """Run one case with Numba and host NumPy on every available core, verified.

    Host runs also use PyPardiso (MKL) on every available core, verified.
    """
    from hybridge.runtime.threads import use_all_cpu_threads

    threads = use_all_cpu_threads()
    if config.verbosity >= 1:
        print(f"[threads] cpus={threads['cpus']} numba={threads['numba_threads']} "
              f"({threads['numba_threading_layer']}) host={threads['host_threads']}", flush=True)
    if backend != "host":
        return run_guiding_center_case(config, preset_key=run_prefix)
    from hybridge.linalg.pardiso_runtime import pardiso_thread_limit

    with pardiso_thread_limit("all") as pardiso:
        if pardiso != threads["cpus"]:
            raise RuntimeError(f"PyPardiso runs with {pardiso} threads, expected {threads['cpus']}")
        if config.verbosity >= 1:
            print(f"[threads] pardiso={pardiso}", flush=True)
        return run_guiding_center_case(config, preset_key=run_prefix)


def _num_steps(final_time: float, dt: float) -> int:
    num_steps = int(round(final_time / dt))
    if num_steps < 1 or not math.isclose(num_steps * dt, final_time, rel_tol=1.0e-12, abs_tol=1.0e-14):
        raise ValueError(f"final_time={final_time} must be an integer multiple of dt={dt}")
    return num_steps


def run_temporal_convergence(
    *,
    scheme: str = "both",
    final_time: float = 0.2,
    dts: tuple[float, ...] = (0.04, 0.02, 0.01, 0.005),
    mesh_size: float = 0.025,
    order: int = 6,
    output_dir: str | Path = "run_outputs/guiding_center/convergence",
    prefix: str = "rho_helm_wave_temporal_convergence",
    verbosity: int = 1,
    plot: bool = False,
    plot_output: str | Path | None = None,
    show_plot: bool = False,
    case_params: dict[str, float] | None = None,
    poisson_tau: float = 1.0,
    backend: str = "auto",
    device_profile: str = "auto",
    case: str = "rho_helm_wave",
    error_quad_1d: int | None = None,
    reference_dt: float | None = None,
    reference_scheme: str = "imex-ark3",
    electric_field: str = "raw",
) -> tuple[list[dict[str, Any]], Path, Path]:
    """Dt ladder on one mesh; errors against the exact solution and, with ``reference_dt``,
    against a same-mesh ``reference_scheme`` run at that step (spatial error cancels)."""
    case_params = dict(case_params or {})
    if scheme not in {*SCHEMES, "both", "all"} and "," not in scheme:
        raise ValueError("scheme must be 'si-euler', 'predictor-corrector', 'si-bdf2', 'si-bdf3', 'h1-bdf3', 'h2-bdf3', 'imex-ark3', 'both', or 'all'")
    if final_time <= 0.0:
        raise ValueError("final_time must be positive")
    output_path = Path(output_dir)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    base = preset_by_key("rho_helm_wave_raw_cuda_amgx_accuracy")
    reference = None
    if reference_dt is not None:
        num_steps = _num_steps(final_time, reference_dt)
        run_prefix = f"{prefix}_reference_{reference_scheme.replace('-', '_')}_dt{reference_dt:.8g}".replace(".", "p")
        config = _manufactured_config(
            base, scheme=reference_scheme, dt=reference_dt, num_steps=num_steps, mesh_size=mesh_size,
            order=order, case_params=case_params, poisson_tau=poisson_tau, output_path=output_path,
            run_prefix=run_prefix, verbosity=verbosity, backend=backend, device_profile=device_profile,
            case=case, electric_field=electric_field,
        )
        started = time.perf_counter()
        result = _run_case(config, run_prefix, _resolve_backend(backend, mesh_size, config.time_scheme))
        reference = _final_state(result)
        reference_row = _manufactured_row(
            result, scheme=reference_scheme, dt=reference_dt, final_time=final_time, num_steps=num_steps,
            mesh_size=mesh_size, order=order, case_params=case_params, poisson_tau=poisson_tau,
            wall_time=time.perf_counter() - started, backend=backend, error_quad_1d=error_quad_1d,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / f"{prefix}_reference.json").write_text(json.dumps(reference_row, indent=2) + "\n")
        print(f"reference {reference_scheme} dt={reference_dt:g}: rho L2={reference_row['rho_l2_error']:.4e} "
              f"phi L2={reference_row['phi_l2_error']:.4e}", flush=True)
        del result
    for selected_scheme in _selected_schemes(scheme):
        for dt in sorted(dts, reverse=True):
            num_steps = _num_steps(final_time, dt)
            run_prefix = f"{prefix}_{selected_scheme.replace('-', '_')}_dt{dt:.8g}".replace(".", "p")
            config = _manufactured_config(
                base, scheme=selected_scheme, dt=dt, num_steps=num_steps, mesh_size=mesh_size,
                order=order, case_params=case_params, poisson_tau=poisson_tau,
                output_path=output_path, run_prefix=run_prefix, verbosity=verbosity, backend=backend,
                device_profile=device_profile, case=case, electric_field=electric_field,
            )
            started = time.perf_counter()
            try:
                result = _run_case(config, run_prefix, _resolve_backend(backend, mesh_size, config.time_scheme))
            except Exception as exc:  # a failed step at large dt is a result, not a crash
                print(f"{selected_scheme} dt={dt:g}: FAILED ({type(exc).__name__}: {exc})", flush=True)
                failures.append({"scheme": selected_scheme, "dt": float(dt), "mesh_size": float(mesh_size),
                                 "order": int(order), "poisson_tau": float(poisson_tau),
                                 "error": f"{type(exc).__name__}: {exc}"})
                continue
            row = _manufactured_row(
                result, scheme=selected_scheme, dt=dt, final_time=final_time, num_steps=num_steps,
                mesh_size=mesh_size, order=order, case_params=case_params,
                poisson_tau=poisson_tau, wall_time=time.perf_counter() - started, backend=backend,
                error_quad_1d=error_quad_1d,
            )
            if reference is not None:
                row.update(reference_dt=float(reference_dt), reference_scheme=reference_scheme,
                           **_reference_errors(result, reference))
            fields = save_final_fields(result, output_path / "fields" / f"{run_prefix}_final.npz",
                                       scheme=selected_scheme, dt=float(dt), mesh_size=float(mesh_size),
                                       electric_field=electric_field)
            row["final_fields"] = None if fields is None else str(fields)
            rows.append(row)
            del result
            release_device_memory()
            print(f"{selected_scheme} dt={dt:g} C={row['courant']:.3g} rho L2={row['rho_l2_error']:.4e}"
                  + (f" rho-ref={row['rho_ref_l2_error']:.4e} phi-ref={row['phi_ref_l2_error']:.4e}"
                     if reference is not None else ""), flush=True)
    _pairwise_rates(rows)
    rows.sort(key=lambda row: (SCHEMES.index(row["scheme"]), -float(row["dt"])))
    csv_path, json_path = _write_outputs(rows, output_path, prefix)
    if failures:
        (output_path / f"{prefix}_failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    _print_table(rows)
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    if plot:
        figure_path = Path(plot_output) if plot_output is not None else output_path / f"{prefix}.png"
        print(f"plot: {plot_convergence(rows, figure_path, show=show_plot)}")
    return rows, csv_path, json_path


def _spatial_rates(rows: list[dict[str, Any]]) -> None:
    """Observed orders against h = sqrt(|Omega|/N_K) between consecutive meshes of one (scheme, p)."""
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        row["h_eff"] = math.sqrt(float(row.get("domain_area", 4.0)) / float(row["triangles"]))
        groups.setdefault((str(row["scheme"]), int(row["order"])), []).append(row)
    for group in groups.values():
        group.sort(key=lambda row: float(row["h_eff"]), reverse=True)
        for index, row in enumerate(group):
            for error_key in ERROR_KEYS:
                rate_key = error_key.replace("_error", "_rate")
                coarse = group[index - 1] if index else None
                if coarse is None or row.get(error_key) is None or coarse.get(error_key) is None:
                    row[rate_key] = None
                    continue
                e0, e1 = float(coarse[error_key]), float(row[error_key])
                row[rate_key] = (math.log(e0 / e1) / math.log(float(coarse["h_eff"]) / float(row["h_eff"]))
                                 if e0 > 0.0 and e1 > 0.0 else None)


def run_spatial_convergence(
    *,
    scheme: str = "imex-ark3",
    final_time: float = 0.5,
    orders: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    mesh_sizes: tuple[float, ...] = (0.4, 0.2, 0.1, 0.05),
    dt_for: dict[tuple[int, float], float] | None = None,
    default_dt: float = 0.005,
    output_dir: str | Path = "run_outputs/guiding_center/convergence",
    prefix: str = "rho_helm_wave_spatial_convergence",
    verbosity: int = 1,
    case_params: dict[str, float] | None = None,
    poisson_tau: float = 1.0,
    backend: str = "auto",
    electric_field: str = "raw",
    poisson_order_offset: int = 0,
    device_profile: str = "auto",
    mesh_family: str = "gmsh",
    case: str = "rho_helm_wave",
    error_quad_1d: int | None = None,
    diagnostics_every: int | None = None,
    solver_rtol: float | None = None,
    exact_drift_order_offset: int | None = None,
    exact_drift_mode: str | None = None,
) -> tuple[list[dict[str, Any]], Path, Path]:
    """Fixed-dt h/p refinement; dt must make the temporal error negligible (checked by the caller)."""
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}")
    case_params = dict(case_params or {})
    output_path = Path(output_dir)
    base = preset_by_key("rho_helm_wave_raw_cuda_amgx_accuracy")
    rows: list[dict[str, Any]] = []
    for order in orders:
        for mesh_size in sorted(mesh_sizes, reverse=True):
            dt = float((dt_for or {}).get((int(order), float(mesh_size)), default_dt))
            num_steps = _num_steps(final_time, dt)
            run_prefix = (f"{prefix}_{scheme.replace('-', '_')}_p{order}_h{mesh_size:.6g}_dt{dt:.8g}"
                          .replace(".", "p"))
            config = _manufactured_config(
                base, scheme=scheme, dt=dt, num_steps=num_steps, mesh_size=mesh_size, order=order,
                case_params=case_params, poisson_tau=poisson_tau, output_path=output_path,
                run_prefix=run_prefix, verbosity=verbosity, backend=backend,
                electric_field=electric_field, poisson_order_offset=poisson_order_offset,
                device_profile=device_profile, mesh_family=mesh_family, case=case,
                diagnostics_every=diagnostics_every, solver_rtol=solver_rtol,
                exact_drift_order_offset=exact_drift_order_offset, exact_drift_mode=exact_drift_mode,
            )
            started = time.perf_counter()
            result = _run_case(config, run_prefix, _resolve_backend(backend, mesh_size, config.time_scheme))
            row = _manufactured_row(
                result, scheme=scheme, dt=dt, final_time=final_time, num_steps=num_steps,
                mesh_size=mesh_size, order=order, case_params=case_params,
                poisson_tau=poisson_tau, wall_time=time.perf_counter() - started, backend=backend,
                error_quad_1d=error_quad_1d,
            )
            fields = save_final_fields(result, output_path / "fields" / f"{run_prefix}_final.npz",
                                       scheme=scheme, dt=float(dt), mesh_size=float(mesh_size),
                                       mesh_family=mesh_family, electric_field=electric_field,
                                       poisson_order_offset=int(poisson_order_offset), solver_rtol=solver_rtol,
                                       exact_drift_mode=(config.exact_drift_mode
                                                         if config.transport_electric_field == "exact" else None),
                                       exact_drift_order_offset=(config.exact_drift_order_offset
                                                                 if config.transport_electric_field == "exact"
                                                                 else None))
            row["final_fields"] = None if fields is None else str(fields)
            rows.append(row)
            print(f"p={order} h={mesh_size:g} K={row['triangles']} dt={dt:g} C={row['courant']:.3g} "
                  f"rho L2={row['rho_l2_error']:.4e} phi L2={row['phi_l2_error']:.4e} "
                  f"({row['wall_time_s']:.1f} s)", flush=True)
            del result
            release_device_memory()
    _spatial_rates(rows)
    rows.sort(key=lambda row: (int(row["order"]), -float(row["h_eff"])))
    csv_path, json_path = _write_outputs(rows, output_path, prefix)
    print("\nGuiding-center spatial convergence")
    print("p  K        h_eff     dt        rho L2       rate    phi L2       rate    phi HDG-H1   rate")
    for row in rows:
        def fmt(key):
            value = row.get(key)
            return "-" if value is None else f"{value:.3f}"
        print(f"{row['order']:<2d} {row['triangles']:<8d} {row['h_eff']:<9.4f} {row['dt']:<9.3g} "
              f"{row['rho_l2_error']:<12.4e} {fmt('rho_l2_rate'):<7} {row['phi_l2_error']:<12.4e} "
              f"{fmt('phi_l2_rate'):<7} {row['phi_hdg_h1_error']:<12.4e} {fmt('phi_hdg_h1_rate'):<7}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    return rows, csv_path, json_path


def _parse_case_params(items: list[str] | None) -> dict[str, float]:
    params: dict[str, float] = {}
    for item in items or ():
        key, _, value = item.partition("=")
        if not key or not value:
            raise ValueError(f"--case-param expects key=value, got {item!r}")
        expression = value.strip().replace("pi", repr(math.pi))
        params[key.strip()] = float(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307 - arithmetic only
    return params


def _main() -> None:
    parser = ArgumentParser(description="Manufactured temporal convergence or matched-time vortex-gas comparison.")
    parser.add_argument("--study", choices=("manufactured", "spatial", "vortex-gas"), default="manufactured")
    parser.add_argument("--case", choices=("rho_helm_wave", "orbiting_vortex"), default="rho_helm_wave",
                        help="Exact solution: Helmholtz wave on the square, or the compact C^{m-1} "
                             "vortex orbiting in the cross-shaped polygon.")
    parser.add_argument("--case-param", action="append", default=None,
                        help="Manufactured case parameter key=value (wave: U, kx, ky; vortex: rho0, A, R, "
                             "m, d, theta0); 'pi' allowed, e.g. kx=pi.")
    parser.add_argument("--error-quad-1d", type=int, default=None,
                        help="Points per direction of the error rules (L2: collapsed Gauss; traces: "
                             "Gauss-Legendre); default: the spaces' own rules.")
    parser.add_argument("--reference-dt", type=float, default=None,
                        help="Temporal study: also measure differences from a same-mesh run at this dt.")
    parser.add_argument("--reference-scheme", choices=SCHEMES, default="imex-ark3")
    parser.add_argument("--poisson-tau", type=float, default=1.0, help="Fixed Poisson HDG stabilization.")
    parser.add_argument("--backend", choices=("auto", "device", "host"), default="auto",
                        help="device: raw-CUDA + AMGX; host: Numba + PyPardiso (every available core); "
                             "auto: host for mesh_size >= 0.1 (<~1000 triangles), device otherwise "
                             "(measured break-even on the RTX PRO 5000 host).")
    parser.add_argument("--orders", default="1,2,3,4,5,6", help="Spatial study: polynomial degrees.")
    parser.add_argument("--mesh-family", choices=MESH_FAMILIES, default="gmsh",
                        help="Spatial study: gmsh = Gmsh Frontal-Delaunay (a lattice of equilateral triangles on "
                             "the rectangle), structured = uniform squares split along the (-1,1) diagonal, "
                             "unstructured = Gmsh Delaunay (no lattice).")
    parser.add_argument("--device-profile", choices=("auto", *DEVICE_PROFILES), default="auto",
                        help="Device kernels/solvers (see DEVICE_PROFILES).")
    parser.add_argument("--electric-field", choices=ELECTRIC_FIELDS, default="raw",
                        help="Transport drift: raw LDG-H flux, recovered RT/l2_closest flux (si-bdf2/si-bdf3 "
                             "only), the compatible Oswald drift (every scheme), or the exact drift (reference; "
                             "see --exact-drift-mode).")
    parser.add_argument("--exact-drift-mode", choices=("pointwise", "projected"), default=None,
                        help="Spatial study, --electric-field exact: the exact drift at the transport's "
                             "quadrature points (pointwise, the default) or its flux projected onto DG(p+s).")
    parser.add_argument("--exact-drift-order-offset", type=int, default=None,
                        help="Spatial study, --electric-field exact: degree offset s of the projected exact "
                             "flux (default 2).")
    parser.add_argument("--poisson-order-offset", type=int, default=0,
                        help="Spatial study, raw field: Poisson degree p + offset (-1, 0, or raised > 0; "
                             "a raised degree drives the transport with the degree-(p+offset) flux).")
    parser.add_argument("--mesh-sizes", default="0.4,0.2,0.1,0.05", help="Spatial study: Gmsh mesh sizes.")
    parser.add_argument("--dt-table", default=None,
                        help="Spatial study: JSON {\"p,h\": dt} overriding --dts' single value per run.")
    parser.add_argument("--scheme", default=None,
                        help=f"One of {SCHEMES}, 'both', 'all', or a comma-separated list (temporal study).")
    parser.add_argument("--final-time", type=float, default=None)
    parser.add_argument("--dts", default=None)
    parser.add_argument("--mesh-size", type=float, default=None)
    parser.add_argument("--order", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("run_outputs/guiding_center/convergence"))
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--verbosity", type=int, choices=(0, 1, 2, 3), default=None)
    parser.add_argument("--plot-convergence", action="store_true", help="Manufactured error panels.")
    parser.add_argument("--plot-output", type=Path, default=None)
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--preset", default=None, help="Vortex study: inherit spatial settings, seed and tolerances.")
    parser.add_argument("--sample-interval", type=float, default=0.5, help="Vortex study: physical time between field samples.")
    parser.add_argument("--plot-comparison", action="store_true", help="Save vortex histories and matched field panels.")
    parser.add_argument("--resolution", type=int, default=1024, help="Vortex raster width/height; norms use DG quadrature.")
    parser.add_argument("--cached-kernels-only", action="store_true", help="Reject new Numba/NVRTC/NVCC compilation.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print vortex run configurations without solving.")
    args = parser.parse_args()
    from scripts.guiding_center.studies.convergence.guiding_center_temporal_comparison import (
        DEFAULT_VORTEX_PRESET, kernel_cache_only, prepare_vortex_comparison, run_vortex_comparison,
    )
    if args.study == "vortex-gas":
        if args.plot_convergence or args.plot_output or args.show_plot:
            parser.error("Use --plot-comparison for vortex-gas; two dts do not measure formal order")
        options = dict(
            preset=args.preset or DEFAULT_VORTEX_PRESET, scheme=args.scheme or "si-bdf2",
            final_time=5.0 if args.final_time is None else args.final_time,
            dts=_parse_dts(args.dts or "0.01,0.005"), sample_interval=args.sample_interval,
            mesh_size=args.mesh_size, order=args.order, output_dir=args.output_dir,
            prefix=args.prefix or "vortex_temporal_comparison",
            verbosity=0 if args.verbosity is None else args.verbosity,
        )
        if args.dry_run:
            from dataclasses import asdict
            print(json.dumps([asdict(c) for c in prepare_vortex_comparison(**options)], indent=2))
            return
        run_vortex_comparison(**options, resolution=args.resolution,
                              plot=args.plot_comparison, cached_kernels_only=args.cached_kernels_only)
        return
    if args.study == "spatial":
        dts = _parse_dts(args.dts or "0.005")
        if len(dts) != 1:
            parser.error("--study spatial takes one --dts value (or --dt-table)")
        table = {}
        if args.dt_table:
            for key, value in json.loads(Path(args.dt_table).read_text() if Path(args.dt_table).exists()
                                         else args.dt_table).items():
                order_text, size_text = key.split(",")
                table[(int(order_text), float(size_text))] = float(value)
        with kernel_cache_only(args.cached_kernels_only):
            run_spatial_convergence(
                scheme=args.scheme or "imex-ark3",
                final_time=0.5 if args.final_time is None else args.final_time,
                orders=tuple(int(v) for v in args.orders.split(",")),
                mesh_sizes=tuple(float(v) for v in args.mesh_sizes.split(",")),
                dt_for=table, default_dt=dts[0], output_dir=args.output_dir,
                prefix=args.prefix or f"{args.case}_spatial_convergence",
                verbosity=0 if args.verbosity is None else args.verbosity,
                case_params=_parse_case_params(args.case_param), poisson_tau=args.poisson_tau,
                backend=args.backend, electric_field=args.electric_field,
                poisson_order_offset=args.poisson_order_offset, device_profile=args.device_profile,
                mesh_family=args.mesh_family,
                case=args.case, error_quad_1d=args.error_quad_1d,
                exact_drift_order_offset=args.exact_drift_order_offset, exact_drift_mode=args.exact_drift_mode,
            )
        return
    if args.preset or args.plot_comparison or args.dry_run:
        parser.error("--preset, --plot-comparison and --dry-run require --study vortex-gas")
    with kernel_cache_only(args.cached_kernels_only):
        run_temporal_convergence(
            scheme=args.scheme or "both",
            final_time=0.2 if args.final_time is None else args.final_time,
            dts=_parse_dts(args.dts or "0.04,0.02,0.01,0.005"),
            mesh_size=0.025 if args.mesh_size is None else args.mesh_size,
            order=6 if args.order is None else args.order,
            output_dir=args.output_dir, prefix=args.prefix or f"{args.case}_temporal_convergence",
            verbosity=1 if args.verbosity is None else args.verbosity,
            plot=args.plot_convergence, plot_output=args.plot_output, show_plot=args.show_plot,
            case_params=_parse_case_params(args.case_param), poisson_tau=args.poisson_tau,
            backend=args.backend, device_profile=args.device_profile, case=args.case,
            error_quad_1d=args.error_quad_1d, reference_dt=args.reference_dt,
            reference_scheme=args.reference_scheme, electric_field=args.electric_field,
        )


if __name__ == "__main__":
    _main()
