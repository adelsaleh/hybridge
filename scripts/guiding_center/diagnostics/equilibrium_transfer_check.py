"""Residuals of the projected and of the consistent transfer of an E7 imported equilibrium (no time stepping).

Builds the control of ``run_band_evolution`` exactly as the runner does (case, mesh, space, projected density
rho_h = Pi_h W(phi_CG), Poisson solver with the run's fastest device options and compatible drift), then
  1. measures the projected state with ``runtime/equilibrium_transfer.transfer_residuals``: window-law mismatch
     for the compatible and the HDG potential (L2), HDG semilinear residual in the flux-jump Gram dual norm (with
     ``--gram``), steady upwind-HDG transport residual, and the potential rate dphi/dt(0) it drives (energy norm,
     band modes in the run's normalization);
  2. solves rho = Pi_h W(phi_c(rho)) by Newton-GMRES (``ConsistentWindowTransfer``);
  3. measures the consistent state the same way and the change rho* - rho_h (L2, max, band modes).
Writes <prefix>_transfer.json and <prefix>_consistent_density.npz (host coefficients of rho*, with the mesh and
space description) in --output-dir. The driver options are those of run_band_evolution (one geometry).
--gram-jump-weight sets the face weight of ||u - u_hat||^2_{dK} in the Gram: 'unit' (1, the flux-jump Gram),
'element' (1/h_K, h_K the longest edge: the discrete H^1_0 x L^2 norm, uniform in h) or 'scaled' (p^2/h_F).
--consistent-npz reuses the consistent density of an earlier check instead of solving Newton again (the mesh and
the degree must match; the difference between its stored and the new projected density is reported).

Example (resolved Ugly Pac-Man, p = 6):
  .venv/bin/python -m scripts.guiding_center.diagnostics.equilibrium_transfer_check \
      --stage-root ../gc-hdg-article/run_outputs/equilibrium_dt080_20261016/cg6 --stage-name "" \
      --geometries uglypacman --band-half-width-factor 1 --order 6 --gram --output-dir OUT --prefix tr
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gram", action="store_true", help="host Gram dual norms (minutes, several GB of memory)")
    parser.add_argument("--potential", choices=("compatible", "primal"), default="compatible")
    parser.add_argument("--newton-rtol", type=float, default=1e-11)
    parser.add_argument("--max-newton", type=int, default=6)
    parser.add_argument("--gmres-rtol", type=float, default=1e-6)
    parser.add_argument("--gmres-restart", type=int, default=150)
    parser.add_argument("--max-products", type=int, default=900)
    parser.add_argument("--host-threads", type=int, default=4)
    parser.add_argument("--gram-jump-weight", choices=("unit", "element", "scaled"), default="unit")
    parser.add_argument("--consistent-npz", type=Path, default=None,
                        help="<prefix>_consistent_density.npz of an earlier check: skip Newton")
    own, rest = parser.parse_known_args(argv)
    from scripts.guiding_center.benchmarks.run_band_evolution import parse_args as driver_args
    driver = driver_args(rest)
    if len(driver.geometries) != 1:
        parser.error("one geometry per check")
    return own, driver


def main(argv=None) -> int:
    own, args = parse_args(argv)
    import numba
    from hybridge.core.field_ops import solution_field
    from hybridge.core.space import DGSpace
    from hybridge.diagnostics.band_modes import BandModeSpectrum
    from hybridge.runtime.threads import set_host_threads
    from hybridge.solvers.diffusion_reaction import DiffusionReactionHDGSolver
    from hybridge.transport.residual import UpwindHDGTransportResidual
    from scripts.guiding_center.benchmarks.run_band_evolution import build_case, build_config, prefix_of
    from scripts.guiding_center.runtime.compatible_drift import CompatibleFluxPoissonSolver, compatible_potential_source
    from scripts.guiding_center.runtime.configuration import _make_poisson_options, _transport_trace_basis
    from scripts.guiding_center.runtime.equilibrium_transfer import (
        ConsistentWindowTransfer, WindowLaw, transfer_residuals)
    from scripts.guiding_center.runtime.runner import _project_case_density

    numba.set_num_threads(min(own.host_threads, numba.config.NUMBA_NUM_THREADS))
    set_host_threads(own.host_threads)
    started = time.perf_counter()

    def log(message):
        print(f"[{time.perf_counter() - started:8.1f}s] {message}", flush=True)

    geometry = args.geometries[0]
    prefix = prefix_of(args, geometry, "control").replace("_control", "")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case = build_case(args, geometry, "control")
    equilibrium = case.imported_equilibrium
    mesh = equilibrium.to_mesh()
    config = build_config(args, geometry, "control", mesh.num_tri)
    log(f"{geometry}: {mesh.num_tri} triangles, p = {config.order}, projection quadrature "
        f"{config.initial_projection_quad_1d}, Poisson {config.poisson_assembly_backend}/{config.poisson_solver}")
    space = DGSpace(mesh, config.order, basis_type=config.basis, volume_quadrature=config.volume_quadrature,
                    volume_quad_1d=config.volume_quad_1d, edge_quad_1d=config.edge_quad_1d)
    density = _project_case_density(config, space, case, equilibrium=True, name="rho_eq_h")
    log("projected density rho_h = Pi_h W(phi_CG)")
    solver = DiffusionReactionHDGSolver(space, source=density, reaction=space.zeros(name="zero_reaction_h"),
                                        boundary_condition=case.potential_boundary_at(0.0),
                                        options=_make_poisson_options(config))
    poisson = CompatibleFluxPoissonSolver(
        solver, space, potential=compatible_potential_source(config.compatible_potential,
                                                             config.poisson_assembly_backend))
    reference = poisson.solve()
    phi_reference = solution_field(reference, space)
    log(f"reference Poisson solve (trace basis {solver.options.trace_basis}, tau {solver.options.stabilization})")
    law = WindowLaw(equilibrium.window, equilibrium.midpoint)
    transfer = ConsistentWindowTransfer(space, poisson, law, quadrature_1d=config.initial_projection_quad_1d,
                                        potential=own.potential, device=True, log=log)
    transport = UpwindHDGTransportResidual(space, trace_basis=_transport_trace_basis(config),
                                           boundary_mode="zero-flux", backend="device",
                                           advection_stabilization=config.transport_advection_stabilization)
    band = equilibrium.band
    spectrum = BandModeSpectrum(space, band.theta_and_weight, max_mode=band.max_mode, density_reference=density,
                                potential_reference=phi_reference, potential_space=space, backend="device")
    gram = None
    if own.gram:
        from hybridge.hdg.gram import build_condensed_hdg_gram_inverse, build_flux_jump_gram_inverse
        from scripts.guiding_center.runtime.equilibrium_transfer import FactoredGramInverse
        condensed = (build_flux_jump_gram_inverse(space) if own.gram_jump_weight == "unit" else
                     build_condensed_hdg_gram_inverse(space, sigma=1.0, jump_weight=own.gram_jump_weight))
        gram = FactoredGramInverse(condensed, threads=own.host_threads)
        log(f"{own.gram_jump_weight}-weight Gram inverse: {gram.local_dofs} local + {gram.trace_dofs} trace dofs, "
            f"trace Schur complement with {gram.nnz} upper nonzeros factored by PARDISO in {gram.setup_seconds:.1f}s")
    common = dict(transport_residual=transport, band_spectrum=spectrum, gram=gram, gram_space=space,
                  cg_potential=equilibrium.potential)
    report = dict(geometry=geometry, triangles=int(mesh.num_tri), order=config.order, prefix=prefix,
                  potential=own.potential, projection_quadrature_1d=config.initial_projection_quad_1d,
                  gram_jump_weight=own.gram_jump_weight if own.gram else None,
                  poisson=dict(backend=config.poisson_assembly_backend, solver=config.poisson_solver,
                               trace_basis=solver.options.trace_basis, rtol=config.poisson_solver_rtol,
                               atol=config.poisson_solver_atol, tau=float(solver.options.stabilization)),
                  source=equilibrium.provenance, band=band.parameters)
    # torsion energy E_T = ||q(T_h)||^2 of -div q = 1: the scale of the source's DOLFINx residual
    # sqrt(r^T K^{-1} r / E_T), so that sqrt(R^T G^{-1} R / E_T) can be compared with it
    poisson.set_source(space.constant(1.0, name="one_h"))
    torsion = poisson.solve()
    torsion_energy = float(getattr(torsion, "raw_flux", torsion.flux).l2_norm() ** 2)
    report["torsion_energy"] = torsion_energy
    log(f"torsion energy E_T = {torsion_energy:.6e}")
    report["projected"] = transfer_residuals(transfer, density, **common)
    log("projected state: " + ", ".join(f"{k} {v:.3e}" for k, v in report["projected"].items()
                                         if isinstance(v, float)))
    if own.consistent_npz is None:
        consistent, _, records = transfer.solve(
            density, rtol=own.newton_rtol, max_newton=own.max_newton, gmres_rtol=own.gmres_rtol,
            restart=own.gmres_restart, max_products=own.max_products)
    else:
        saved = np.load(own.consistent_npz)
        if int(saved["order"]) != config.order or not np.array_equal(saved["triangles"], mesh.triangles):
            raise SystemExit(f"{own.consistent_npz}: other mesh or degree than this check")
        stored = transfer.xp.asarray(saved["projected"])
        drift = float(abs(stored - transfer.coefficients(density)).max())
        consistent = transfer.field(transfer.xp.asarray(saved["coefficients"]), name="rho_star")
        records = dict(loaded=str(own.consistent_npz), projected_max_difference=drift)
        log(f"consistent density loaded from {own.consistent_npz} (stored projection differs by {drift:.2e})")
    report["newton"] = records
    report["poisson_solves"], report["poisson_solve_seconds"] = transfer.solves, transfer.solve_seconds
    report["consistent"] = transfer_residuals(transfer, consistent, **common)
    log("consistent state: " + ", ".join(f"{k} {v:.3e}" for k, v in report["consistent"].items()
                                          if isinstance(v, float)))
    for state in ("projected", "consistent"):
        values = report[state]
        for key in [k for k in values if k.endswith("_gram_dual")]:
            values[key + "_over_torsion"] = values[key] / torsion_energy ** 0.5
    rho0, rho1 = transfer.coefficients(density), transfer.coefficients(consistent)
    change = transfer.field(rho1 - rho0, name="delta_rho")
    tables = spectrum.density_tables
    change_modes = spectrum._spectrum(tables, tables.values(change), 0.0, spectrum.density_norm)
    change_modes = change_modes.get() if hasattr(change_modes, "get") else np.asarray(change_modes)
    report["change"] = dict(relative_l2=transfer.norm(rho1 - rho0) / transfer.norm(rho0),
                            max_coefficient=float(abs(rho1 - rho0).max()),
                            band_modes=[float(v) for v in change_modes])
    report["seconds"] = time.perf_counter() - started
    out = args.output_dir / f"{prefix}_transfer.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    if own.consistent_npz is None:
        np.savez(args.output_dir / f"{prefix}_consistent_density.npz", coefficients=rho1.get(),
                 projected=rho0.get(), node_coords=mesh.node_coords, triangles=mesh.triangles, order=config.order,
                 basis=config.basis, metadata=json.dumps(dict(geometry=geometry, prefix=prefix,
                                                              potential=own.potential,
                                                              source=equilibrium.provenance), default=str))
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
