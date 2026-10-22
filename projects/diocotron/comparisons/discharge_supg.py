"""D7 comparison: the collisional parallel-vane magnetron with CG Poisson and CG-SUPG density (DOLFINx).

Same mesh (HYBRIDGE's parallel-vane Gmsh mesh), same case (``magnetron(...)`` with the D7 options:
electrodes, pulsed emission, Hall drift and diffusion) and the same SI-BDF2 splitting as the HYBRIDGE
runs. Per step:

1. density with the drift of the extrapolated potential ``2 phi^n - phi^{n-1}`` (SI-Euler start-up):
   ``(rho - f)/s - alpha rho + div(u rho) - div(D grad rho) = S``, ``u = (d_y phi, -d_x phi) + q/h``,
   ``q = -grad phi``, the case's density data on every wall (absorbing anode, cathode emitting at rho_c
   times the emission pulse), CG P_k with SUPG
   (residual-based, test ``tau u . grad v``);
2. endpoint Poisson ``-lap phi = rho`` with the electrode data at ``t^{n+1}``, CG P_k (LU factored once).

Per step it records the charge, the wall outflux from the variationally consistent boundary flux
(minus the integrated-by-parts Galerkin residual tested with the boundary hat functions), split into
cathode and anode, and the charge-balance defect ``M^{n+1} - H^n - s (int S + int alpha rho - I)``.
For CG-SUPG that defect is the sum of the SUPG terms at the interior nodes, so it is not round-off:
that is the comparison with HDG, whose defect is round-off. No positivity limiter.

Run in the FEniCSx environment (imports HYBRIDGE from the source tree; serial)::

    cd ~/src/hybridge && ~/src/.envs/diocotron-fenicsx/bin/python -m \
        projects.diocotron.comparisons.discharge_supg --output-dir <dir> [options]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np


def build_case_and_mesh(args):
    """The HYBRIDGE D7 case and its parallel-vane mesh (the HYBRIDGE runs use the same arguments)."""
    from hybridge.core.mesh import gmsh_vane_anode_mesh
    from scripts.guiding_center.cases.guiding_center_cases import magnetron

    case = magnetron(
        anode_geometry="vanes", wave_mode=6, wave_amplitude=args.wave_amplitude, wave_speed=args.wave_speed,
        hub_edge=args.hub_edge, emission_top=args.emission_top, vane_width=args.vane_width,
        band_mesh_size=args.band_mesh_size, refine_band=tuple(args.refine_band),
        corner_mesh_size=args.corner_mesh_size, hall_parameter=args.hall_parameter,
        electron_temperature=args.electron_temperature, emission_period=args.emission_period,
        emission_duty=args.emission_duty, emission_phase=args.emission_phase,
        cathode_mesh_size=args.cathode_mesh_size, cathode_grading=args.cathode_grading,
        curvature_points=args.curvature_points, anode_mesh_size=args.anode_mesh_size,
        anode_grading=args.anode_grading)
    g = case.parameters["geometry"]
    mesh = gmsh_vane_anode_mesh(
        args.mesh_size, cathode_radius=g["inner_radius"], anode_radius=g["anode_radius"],
        back_radius=g["back_radius"], vanes=g["vanes"], vane_fraction=g["vane_fraction"],
        vane_width=g.get("vane_width"), tip_radius=g.get("tip_radius"), band=g["band"],
        cathode_mesh_size=g.get("cathode_mesh_size"), cathode_grading=float(g.get("cathode_grading", 0.3)),
        band_mesh_size=g["band_mesh_size"],
        band_grading=g["band_grading"], corner_mesh_size=g["corner_mesh_size"],
        corner_grading=g["corner_grading"], curvature_points=g.get("curvature_points"),
        anode_mesh_size=g.get("anode_mesh_size"), anode_grading=float(g.get("anode_grading", 0.3)), log_cache=False)
    return case, mesh


def _lu(comm, matrix, name):
    from petsc4py import PETSc

    ksp = PETSc.KSP().create(comm)
    ksp.setOptionsPrefix(name)
    ksp.setOperators(matrix)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setFromOptions()
    return ksp


def run(args) -> int:
    import basix.ufl
    import ufl
    from dolfinx import fem
    from dolfinx import mesh as dmesh
    from dolfinx.fem import petsc as fem_petsc
    from mpi4py import MPI
    from petsc4py import PETSc

    case, hmesh = build_case_and_mesh(args)
    collisional = case.parameters["collisional"]
    hall, diffusion = float(collisional["hall_parameter"]), float(collisional["diffusion"])
    comm = MPI.COMM_SELF
    domain = dmesh.create_mesh(comm, np.asarray(hmesh.triangles)[:, :3].astype(np.int64),
                               basix.ufl.element("Lagrange", "triangle", 1, shape=(2,)),
                               np.asarray(hmesh.node_coords, dtype=np.float64)[:, :2])
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    space = fem.functionspace(domain, ("Lagrange", args.order))
    walls = fem.locate_dofs_topological(space, tdim - 1, dmesh.exterior_facet_indices(domain.topology))
    wall_xy = space.tabulate_dof_coordinates()[walls, :2]
    middle = 0.5 * (case.parameters["cathode"] + case.parameters["anode"])
    on_cathode = np.hypot(wall_xy[:, 0], wall_xy[:, 1]) < middle

    def function(callable_xy=None, name="f"):
        out = fem.Function(space, name=name)
        if callable_xy is not None:
            out.interpolate(lambda X: np.asarray(callable_xy(X[0], X[1]), dtype=np.float64) + 0.0 * X[0])
        return out

    rho, rho_prev, rho_new = (function(case.initial_density_at(), "rho") for _ in range(3))
    phi, phi_prev, phi_data = function(name="phi"), function(name="phi_prev"), function(name="phi_D")
    source_profile = function(case.density_source_profile, "S_profile")  # zero when the case has none
    reaction_profile = function(case.density_reaction_profile, "nu_profile")
    zero = fem.Constant(domain, PETSc.ScalarType(0.0))

    # Poisson: constant operator, LU once; data and source change per solve.
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    poisson_bc = fem.dirichletbc(phi_data, walls)
    a_poisson = fem.form(ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx)
    poisson_source = function(name="poisson_source")
    L_poisson = fem.form(poisson_source * test * ufl.dx)
    A_poisson = fem_petsc.assemble_matrix(a_poisson, bcs=[poisson_bc])
    A_poisson.assemble()
    ksp_poisson = _lu(comm, A_poisson, "poisson_")
    b_poisson = fem_petsc.create_vector(space)

    def solve_poisson(density, time_value, out):
        phi_data.interpolate(lambda X: np.asarray(case.wall_potential_at(time_value)(X[0], X[1]),
                                                  dtype=np.float64) + 0.0 * X[0])
        poisson_source.x.array[:] = density.x.array
        with b_poisson.localForm() as local:
            local.set(0.0)
        fem_petsc.assemble_vector(b_poisson, L_poisson)
        fem_petsc.apply_lifting(b_poisson, [a_poisson], bcs=[[poisson_bc]])
        b_poisson.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(b_poisson, [poisson_bc])
        ksp_poisson.solve(b_poisson, out.x.petsc_vec)
        out.x.scatter_forward()

    solve_poisson(rho, 0.0, phi)
    phi_prev.x.array[:] = phi.x.array

    # Density: SUPG ADR with scale s, history f, drift from the extrapolated potential.
    s_const, rate_source, rate_reaction = (fem.Constant(domain, PETSc.ScalarType(1.0)) for _ in range(3))
    weight_now, weight_prev = (fem.Constant(domain, PETSc.ScalarType(1.0)) for _ in range(2))
    history = function(name="f")
    phi_star = weight_now * phi - weight_prev * phi_prev
    q = -ufl.grad(phi_star)
    u = ufl.as_vector((phi_star.dx(1), -phi_star.dx(0))) + q / hall
    alpha = rate_reaction * reaction_profile
    source = rate_source * source_profile
    h_cell = ufl.CellDiameter(domain)
    k = float(args.order)
    speed = ufl.sqrt(ufl.dot(u, u) + 1e-12)
    tau = float(args.supg_scale) / ufl.sqrt((2.0 / s_const) ** 2 + (2.0 * k * speed / h_cell) ** 2
                                            + (4.0 * k * k * diffusion / h_cell ** 2) ** 2)

    def strong(r):
        return r / s_const - alpha * r + ufl.div(u * r) - diffusion * ufl.div(ufl.grad(r))

    # Density data of the case on every wall: the anode absorbs (0), the cathode emits rho_c times the pulse.
    density_data = function(name="rho_D")

    def update_density_data(time_value):
        density_data.interpolate(lambda X: np.asarray(case.density_boundary(X[0], X[1], time_value),
                                                      dtype=np.float64) + 0.0 * X[0])

    density_bc = fem.dirichletbc(density_data, walls)
    # Fixed quadrature: UFL's estimate for the SUPG weight (a square root) would be needlessly high.
    dx = ufl.dx(metadata={"quadrature_degree": 3 * args.order})
    a_density = fem.form((trial / s_const - alpha * trial) * test * dx + ufl.div(u * trial) * test * dx
                         + diffusion * ufl.inner(ufl.grad(trial), ufl.grad(test)) * dx
                         + tau * strong(trial) * ufl.dot(u, ufl.grad(test)) * dx)
    L_density = fem.form((history / s_const + source) * test * dx
                         + tau * (history / s_const + source) * ufl.dot(u, ufl.grad(test)) * dx)
    # Integrated-by-parts Galerkin residual of the solved density: its wall nodes give the consistent outflux.
    residual = fem.form(((rho_new - history) / s_const - alpha * rho_new - source) * test * dx
                        - ufl.inner(u * rho_new - diffusion * ufl.grad(rho_new), ufl.grad(test)) * dx)
    mass = fem.form(rho_new * ufl.dx)
    source_integral = fem.form(source * ufl.dx)
    reaction_integral = fem.form(alpha * rho_new * ufl.dx)
    A_density = fem_petsc.create_matrix(a_density)
    b_density = fem_petsc.create_vector(space)
    ksp_density = _lu(comm, A_density, "density_")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    columns = ("step", "time", "scale", "mass", "outflux", "outflux_cathode", "outflux_anode", "source_production",
               "reaction_production", "charge_balance_defect", "charge_balance_relative_defect", "rho_min", "rho_max",
               "density_assembly_s", "density_solve_s", "poisson_solve_s")
    stream = (output / "discharge_supg.csv").open("w", newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    pulse = case.density_source_rate or (lambda t: 1.0)
    reaction_rate = case.density_reaction_rate or (lambda t: 1.0)
    dt, steps = float(args.dt), int(round(args.final_time / args.dt))
    mass_now = mass_prev = fem.assemble_scalar(fem.form(rho * ufl.dx))
    start = time.perf_counter()
    for step in range(1, steps + 1):
        t_new = step * dt
        bdf2 = step > 1
        scale = 2.0 * dt / 3.0 if bdf2 else dt
        s_const.value, weight_now.value, weight_prev.value = scale, (2.0 if bdf2 else 1.0), (1.0 if bdf2 else 0.0)
        rate_source.value, rate_reaction.value = float(pulse(t_new)), float(reaction_rate(t_new))
        history.x.array[:] = ((4.0 * rho.x.array - rho_prev.x.array) / 3.0) if bdf2 else rho.x.array
        update_density_data(t_new)
        clock = time.perf_counter()
        A_density.zeroEntries()
        fem_petsc.assemble_matrix(A_density, a_density, bcs=[density_bc])
        A_density.assemble()
        with b_density.localForm() as local:
            local.set(0.0)
        fem_petsc.assemble_vector(b_density, L_density)
        fem_petsc.apply_lifting(b_density, [a_density], bcs=[[density_bc]])
        b_density.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(b_density, [density_bc])
        assembly_clock = time.perf_counter()
        ksp_density.setOperators(A_density)
        ksp_density.solve(b_density, rho_new.x.petsc_vec)
        rho_new.x.scatter_forward()
        density_seconds = time.perf_counter() - clock
        assembly_seconds = assembly_clock - clock
        # Diagnostics of the solved density (before the potential update; s and f of this stage).
        r = fem_petsc.assemble_vector(residual)
        r.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        values = r.array[walls]
        outflux_cathode, outflux_anode = -float(values[on_cathode].sum()), -float(values[~on_cathode].sum())
        outflux = outflux_cathode + outflux_anode
        mass_new = fem.assemble_scalar(mass)
        history_mass = (4.0 * mass_now - mass_prev) / 3.0 if bdf2 else mass_now
        produced_source = fem.assemble_scalar(source_integral)
        produced_reaction = fem.assemble_scalar(reaction_integral)
        defect = mass_new - history_mass - scale * (produced_source + produced_reaction - outflux)
        r.destroy()
        clock = time.perf_counter()
        phi_prev.x.array[:] = phi.x.array
        solve_poisson(rho_new, t_new, phi)
        poisson_seconds = time.perf_counter() - clock
        rho_prev.x.array[:] = rho.x.array
        rho.x.array[:] = rho_new.x.array
        mass_prev, mass_now = mass_now, mass_new
        writer.writerow({"step": step, "time": t_new, "scale": scale, "mass": mass_new, "outflux": outflux,
                         "outflux_cathode": outflux_cathode, "outflux_anode": outflux_anode,
                         "source_production": produced_source, "reaction_production": produced_reaction,
                         "charge_balance_defect": defect,
                         "charge_balance_relative_defect": defect / max(abs(mass_new), 1e-300),
                         "rho_min": float(rho.x.array.min()), "rho_max": float(rho.x.array.max()),
                         "density_assembly_s": assembly_seconds, "density_solve_s": density_seconds - assembly_seconds,
                         "poisson_solve_s": poisson_seconds})
        if step % args.log_every == 0 or step == steps:
            stream.flush()
            print(f"[supg] step {step}/{steps} t={t_new:.3f} M={mass_new:.6f} I={outflux:.4e} "
                  f"defect/M={defect / max(abs(mass_new), 1e-300):.2e} rho_min={rho.x.array.min():.3e} "
                  f"({time.perf_counter() - start:.0f}s)", flush=True)
    stream.close()
    np.savez_compressed(output / "final_density.npz", coordinates=space.tabulate_dof_coordinates()[:, :2],
                        density=rho.x.array.copy(), potential=phi.x.array.copy())
    meta = {key: (list(value) if isinstance(value, tuple) else value) for key, value in vars(args).items()}
    meta.update(triangles=int(np.asarray(hmesh.triangles).shape[0]), dofs=int(space.dofmap.index_map.size_global),
                hall_parameter=hall, diffusion=diffusion, wall_time_s=time.perf_counter() - start)
    (output / "discharge_supg.json").write_text(json.dumps(meta, indent=2, default=float) + "\n")
    (output / "all-done").write_text(time.strftime("%F %T") + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--final-time", type=float, default=40.0)
    parser.add_argument("--supg-scale", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--vane-width", type=float, default=0.26)
    parser.add_argument("--hall-parameter", type=float, default=30.0)
    parser.add_argument("--electron-temperature", type=float, default=0.01)
    parser.add_argument("--emission-period", type=float, default=20.0)
    parser.add_argument("--emission-duty", type=float, default=0.5)
    parser.add_argument("--wave-amplitude", type=float, default=0.05)
    parser.add_argument("--wave-speed", type=float, default=0.25)
    parser.add_argument("--hub-edge", type=float, default=0.025)
    parser.add_argument("--emission-top", type=float, default=0.6)
    parser.add_argument("--mesh-size", type=float, default=0.03)
    parser.add_argument("--band-mesh-size", type=float, default=0.015)
    parser.add_argument("--refine-band", type=float, nargs=2, default=(0.55, 1.35))
    parser.add_argument("--corner-mesh-size", type=float, default=0.005)
    parser.add_argument("--cathode-mesh-size", type=float, default=None, help="as run_magnetron.py")
    parser.add_argument("--cathode-grading", type=float, default=0.3)
    parser.add_argument("--curvature-points", type=int, default=None, help="as run_magnetron.py")
    parser.add_argument("--anode-mesh-size", type=float, default=None, help="as run_magnetron.py")
    parser.add_argument("--anode-grading", type=float, default=0.3)
    parser.add_argument("--emission-phase", type=float, default=0.0, help="as run_magnetron.py")
    return parser


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
