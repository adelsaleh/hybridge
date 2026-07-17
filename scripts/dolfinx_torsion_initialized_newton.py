#!/usr/bin/env python3
"""DOLFINx fixed-mesh torsion-initialized Newton comparison runner.

This script mirrors the legacy scalar CG/P2 FreeFEM runner on a fixed mesh.  It uses
continuous Lagrange elements of user-selected order, solves the torsion
initializer, builds the torsion-designed density band, solves the Poisson
initializer, and then applies the same epsilon-continuation Newton loop for

    -Delta phi = f_epsilon(phi),  phi|_boundary = 0.

The intended comparison workflow is to pass the exact ``initial_mesh.msh``
saved by ``hdg_torsion_initialized_newton.py`` via ``--mesh``.

Verbosity levels are intentionally coarse:

``-v 0``
    Only essential run status is printed.
``-v 1``
    Default progress output at accepted/equilibrium stages.
``-v 2``
    Line-search diagnostics and band-overlap metrics for tracking where the
    designed band and final density band separate.
``-v 3``
    Level 2 plus PETSc KSP residual monitors for iterative linear solvers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import meshio
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh, plot as dolfinx_plot
from dolfinx.fem import petsc as fem_petsc
from dolfinx.io import XDMFFile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RUN_LOG_ROOT = REPO_ROOT / "run_logs" / "dolfinx_torsion_initialized_newton"


@dataclass
class TorsionParameters:
    alpha_t1: float = 0.60
    alpha_t2: float = 0.70
    eps_t_ratio: float = 0.06
    beta_phi1: float = 0.60
    beta_phi2: float = 0.70
    eps_phi_ratios: tuple[float, ...] = (0.11, 0.08, 0.06)
    rho_amp: float = 1.0
    max_it: int = 70
    tol_res: float = 1.0e-10
    tol_newton: float = 1.0e-10
    beta_ls: float = 0.5
    armijo_c: float = 1.0e-6
    alpha_min: float = 1.0e-7
    max_backtrack: int = 30
    mass_floor_fraction: float = 0.01
    rho_max_floor: float = 0.02
    max_stagnation: int = 5
    stagnation_tol: float = 1.0e-5
    active_threshold: float = 0.05
    plateau_threshold: float = 0.90


def root_print(comm: MPI.Comm, message: str) -> None:
    if comm.rank == 0:
        print(message, flush=True)


def slug_for_path(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text.strip())
    return safe.strip("_") or "run"


def logistic_ufl(z, eps: float):
    zz = z / float(eps)
    return ufl.conditional(
        ufl.gt(zz, 50.0),
        1.0,
        ufl.conditional(ufl.lt(zz, -50.0), 0.0, 1.0 / (1.0 + ufl.exp(-zz))),
    )


def window_ufl(values, c1: float, c2: float, eps: float, amp: float):
    return float(amp) * (logistic_ufl(values - float(c1), eps) - logistic_ufl(values - float(c2), eps))


def window_derivative_ufl(values, c1: float, c2: float, eps: float, amp: float):
    s1 = logistic_ufl(values - float(c1), eps)
    s2 = logistic_ufl(values - float(c2), eps)
    return float(amp) * (s1 * (1.0 - s1) - s2 * (1.0 - s2)) / float(eps)


def allreduce_scalar(comm: MPI.Comm, value: float, op=MPI.SUM) -> float:
    return float(comm.allreduce(float(value), op=op))


def local_minmax(function: fem.Function) -> tuple[float, float]:
    arr = function.x.array
    if arr.size == 0:
        return math.inf, -math.inf
    return float(np.min(arr)), float(np.max(arr))


def global_minmax(comm: MPI.Comm, function: fem.Function) -> tuple[float, float]:
    local_min, local_max = local_minmax(function)
    return (
        float(comm.allreduce(local_min, op=MPI.MIN)),
        float(comm.allreduce(local_max, op=MPI.MAX)),
    )


def assemble_scalar(comm: MPI.Comm, form) -> float:
    return allreduce_scalar(comm, fem.assemble_scalar(fem.form(form)), op=MPI.SUM)


def read_mesh_with_meshio(path: Path, comm: MPI.Comm):
    """Read a triangular Gmsh mesh through meshio/XDMF."""
    if comm.rank == 0:
        msh = meshio.read(path)
        triangles = None
        for cell_block in msh.cells:
            if cell_block.type == "triangle":
                triangles = np.asarray(cell_block.data, dtype=np.int64)
                break
        if triangles is None:
            raise ValueError(f"mesh {path} does not contain triangle cells")
        points = np.asarray(msh.points[:, :2], dtype=np.float64)
    else:
        points = np.empty((0, 2), dtype=np.float64)
        triangles = np.empty((0, 3), dtype=np.int64)

    with tempfile.TemporaryDirectory() as tmpdir:
        xdmf_path = Path(tmpdir) / "mesh.xdmf"
        if comm.rank == 0:
            meshio.write(xdmf_path, meshio.Mesh(points=points, cells=[("triangle", triangles)]))
        comm.barrier()
        with XDMFFile(comm, str(xdmf_path), "r") as xdmf:
            domain = xdmf.read_mesh(name="Grid")
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)
    return domain


def load_or_generate_mesh(args: argparse.Namespace, run_dir: Path, comm: MPI.Comm):
    """Load a fixed mesh, or generate the smooth-star mesh on rank zero.

    Mesh generation is intentionally delegated to a short subprocess.  The
    DOLFINx process already has PETSc/MPI loaded, while the Python Gmsh module
    may bring in a different MPI stack on some machines.  Keeping Gmsh in a
    separate process avoids mixed-MPI runtime warnings during the actual solve.
    """
    if args.mesh is not None:
        mesh_path = args.mesh.resolve()
        domain = read_mesh_with_meshio(mesh_path, comm)
        return domain, mesh_path, "file"

    mesh_path = run_dir / "initial_mesh.msh"
    if comm.rank == 0:
        mesh_config = {
            "mesh_size": float(args.mesh_size),
            "boundary_points": int(args.star_n),
            "radius": float(args.star_r0),
            "amplitude": float(args.star_amp),
            "mode": int(args.star_mode),
            "verbosity": int(args.gmsh_verbosity),
            "algorithm": None if args.gmsh_algorithm is None else int(args.gmsh_algorithm),
            "write_path": str(mesh_path),
            "msh_file_version": 2.2,
        }
        code = (
            "import json, sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from hdgfem.core.mesh import gmsh_smooth_star_mesh\n"
            "cfg = json.loads(sys.argv[2])\n"
            "mesh_size = cfg.pop('mesh_size')\n"
            "gmsh_smooth_star_mesh(mesh_size, **cfg)\n"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(REPO_ROOT)
            if not env.get("PYTHONPATH")
            else f"{REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}"
        )
        subprocess.run(
            [sys.executable, "-c", code, str(REPO_ROOT), json.dumps(mesh_config, separators=(",", ":"))],
            check=True,
            env=env,
        )
    comm.barrier()
    domain = read_mesh_with_meshio(mesh_path, comm)
    return domain, mesh_path, "smooth_star"


def make_run_dir(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if args.run_dir is None:
        prefix = f"{slug_for_path(args.run_tag)}_" if args.run_tag else ""
        run_dir = DEFAULT_RUN_LOG_ROOT / f"{prefix}{timestamp}"
    else:
        requested = args.run_dir
        run_dir = requested if not requested.exists() else requested.parent / f"{requested.name}_{timestamp}"
    candidate = run_dir
    suffix = 1
    while candidate.exists():
        candidate = run_dir.parent / f"{run_dir.name}_{suffix:03d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def boundary_bc(V: fem.FunctionSpace):
    domain = V.mesh
    fdim = domain.topology.dim - 1
    facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    return fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)


def solver_options(kind: str, *, ksp_type: str | None = None) -> dict[str, object]:
    if kind == "mumps":
        return {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
    if kind == "lu":
        return {"ksp_type": "preonly", "pc_type": "lu"}
    if kind == "hypre":
        return {"ksp_type": ksp_type or "gmres", "pc_type": "hypre", "pc_hypre_type": "boomeramg"}
    if kind == "gamg":
        return {"ksp_type": ksp_type or "gmres", "pc_type": "gamg"}
    raise ValueError(f"unknown solver kind {kind!r}")


def solve_linear_form(
        a,
        L,
        u: fem.Function,
        bcs: list,
        *,
        prefix: str,
        solver: str,
        ksp_type: str | None,
        rtol: float,
        atol: float,
        max_it: int | None,
        verbosity: int = 1,
) -> tuple[int, float, float]:
    """Assemble and solve a linear variational problem into ``u``.

    When an iterative PETSc solver is used, ``verbosity >= 3`` attaches a KSP
    monitor that prints the iteration number and residual norm.  Direct solvers
    intentionally skip this monitor because PETSc reports no useful Krylov
    history for LU/MUMPS.
    """
    start = time.perf_counter()
    a_form = fem.form(a)
    L_form = fem.form(L)
    A = fem_petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    b = fem_petsc.assemble_vector(L_form)
    fem_petsc.apply_lifting(b, [a_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, bcs)

    ksp = PETSc.KSP().create(u.function_space.mesh.comm)
    ksp.setOptionsPrefix(prefix)
    opts = PETSc.Options()
    for key, value in solver_options(solver, ksp_type=ksp_type).items():
        opts[f"{prefix}{key}"] = value
    if solver not in {"mumps", "lu"}:
        opts[f"{prefix}ksp_rtol"] = rtol
        opts[f"{prefix}ksp_atol"] = atol
        if max_it is not None:
            opts[f"{prefix}ksp_max_it"] = max_it
    ksp.setFromOptions()
    if verbosity >= 3 and solver not in {"mumps", "lu"}:
        comm = u.function_space.mesh.comm

        def monitor(_, iteration: int, residual_norm: float) -> None:
            root_print(comm, f"KSP prefix={prefix} it={iteration} rnorm={float(residual_norm):.6e}")

        ksp.setMonitor(monitor)
    ksp.setOperators(A)
    ksp.solve(b, u.x.petsc_vec)
    u.x.scatter_forward()
    elapsed = time.perf_counter() - start
    its = int(ksp.getIterationNumber())
    reason = ksp.getConvergedReason()
    residual = float(ksp.getResidualNorm())
    ksp.destroy()
    A.destroy()
    b.destroy()
    if reason < 0:
        raise RuntimeError(f"linear solve {prefix!r} failed with PETSc reason {reason}")
    return its, residual, elapsed


def update_interpolated(target: fem.Function, expression) -> None:
    interpolation_points = target.function_space.element.interpolation_points
    if callable(interpolation_points):
        interpolation_points = interpolation_points()
    expr = fem.Expression(expression, interpolation_points)
    target.interpolate(expr)
    target.x.scatter_forward()


def residual_vector_norm(R_form, bc, comm: MPI.Comm) -> float:
    r = fem_petsc.assemble_vector(fem.form(R_form))
    fem_petsc.set_bc(r, [bc])
    norm = float(r.norm())
    r.destroy()
    # PETSc Vec.norm is already collective/global.
    return norm


def compute_metrics(
        *,
        u: fem.Function,
        rho: fem.Function,
        rho_design: fem.Function,
        rho_design_l2: float,
        residual_form,
        bc,
        dx,
        c2_phi: float,
        active_threshold: float,
        plateau_threshold: float,
        rho_amp: float,
) -> dict[str, float]:
    """Evaluate nonlinear residual, density, and band-comparison diagnostics.

    The active and plateau Jaccard values compare thresholded final-density
    sets against the torsion-designed density sets.  These diagnostics are
    designed for parameter sweeps: they expose whether a run only converged as
    a nonlinear solve, or whether its converged band is geometrically close to
    the intended design band.
    """
    comm = u.function_space.mesh.comm
    min_u, max_u = global_minmax(comm, u)
    min_rho, max_rho = global_minmax(comm, rho)
    mass_rho = assemble_scalar(comm, rho * dx)
    rho_l2 = math.sqrt(max(assemble_scalar(comm, rho * rho * dx), 0.0))
    energy_phi = math.sqrt(max(assemble_scalar(comm, ufl.inner(ufl.grad(u), ufl.grad(u)) * dx), 0.0))
    active_area = assemble_scalar(comm, ufl.conditional(ufl.gt(rho, active_threshold * rho_amp), 1.0, 0.0) * dx)
    plateau_area = assemble_scalar(comm, ufl.conditional(ufl.gt(rho, plateau_threshold * rho_amp), 1.0, 0.0) * dx)
    active_design = ufl.conditional(ufl.gt(rho_design, active_threshold * rho_amp), 1.0, 0.0)
    active_final = ufl.conditional(ufl.gt(rho, active_threshold * rho_amp), 1.0, 0.0)
    plateau_design = ufl.conditional(ufl.gt(rho_design, plateau_threshold * rho_amp), 1.0, 0.0)
    plateau_final = ufl.conditional(ufl.gt(rho, plateau_threshold * rho_amp), 1.0, 0.0)
    active_design_area = assemble_scalar(comm, active_design * dx)
    active_overlap = assemble_scalar(comm, active_design * active_final * dx)
    active_union = max(active_design_area + active_area - active_overlap, 1.0e-30)
    plateau_design_area = assemble_scalar(comm, plateau_design * dx)
    plateau_overlap = assemble_scalar(comm, plateau_design * plateau_final * dx)
    plateau_union = max(plateau_design_area + plateau_area - plateau_overlap, 1.0e-30)
    rho_design_diff_l2 = math.sqrt(max(assemble_scalar(comm, (rho - rho_design) ** 2 * dx), 0.0))
    rel_design = rho_design_diff_l2 / max(rho_design_l2, 1.0e-30)
    return {
        "resEuclid": residual_vector_norm(residual_form, bc, comm),
        "minU": min_u,
        "maxU": max_u,
        "minRho": min_rho,
        "maxRho": max_rho,
        "massRho": mass_rho,
        "rhoL2": rho_l2,
        "energyPhi": energy_phi,
        "activeArea": active_area,
        "plateauArea": plateau_area,
        "plateauFrac": plateau_area / max(active_area, 1.0e-30),
        "activeDesignArea": active_design_area,
        "activeOverlapArea": active_overlap,
        "activeJaccard": active_overlap / active_union,
        "plateauDesignArea": plateau_design_area,
        "plateauOverlapArea": plateau_overlap,
        "plateauJaccard": plateau_overlap / plateau_union,
        "rhoDesignDiffL2": rho_design_diff_l2,
        "relRhoDesign": rel_design,
        "massRhoMinusDesign": mass_rho - assemble_scalar(comm, rho_design * dx),
        "annularPhiMinusC2": max_u - c2_phi,
    }


def write_row(writer: csv.DictWriter, **row) -> None:
    writer.writerow(row)


class PyVistaTorsionPlotter:
    """PyVista plotting and frame writer for DOLFINx torsion-initialized runs.

    The layout and command-line behavior mirror
    ``hdg_torsion_initialized_newton.py``: interactive plots stay open
    until Enter is pressed in the terminal, and saved frames are written through
    the faster PyVista path.
    """

    def __init__(
            self,
            args: argparse.Namespace,
            *,
            run_tag: str,
            run_dir: Path,
            frame_writer: csv.DictWriter | None,
            comm: MPI.Comm,
    ) -> None:
        self.args = args
        self.run_tag = run_tag
        self.run_dir = run_dir
        self.frame_writer = frame_writer
        self.comm = comm
        self.frame_counter = 0
        self.frame_dir = args.frame_dir or (run_dir / "frames")
        if comm.rank == 0 and args.save_frames:
            self.frame_dir.mkdir(parents=True, exist_ok=True)

    @property
    def active(self) -> bool:
        return self.comm.rank == 0 and bool(self.args.plot or self.args.save_frames)

    @staticmethod
    def _wait_for_enter(plotter) -> None:
        print("PyVista plot is interactive. Press Enter here to continue...", flush=True)
        entered = threading.Event()

        def read_enter() -> None:
            try:
                sys.stdin.readline()
            finally:
                entered.set()

        threading.Thread(target=read_enter, daemon=True).start()
        while True:
            try:
                plotter.update()
            except Exception:
                break
            if entered.wait(0.05):
                break

    @staticmethod
    def _function_grid(function: fem.Function, *, scalar_name: str):
        import pyvista as pv

        topology, cell_types, geometry = dolfinx_plot.vtk_mesh(function.function_space)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        values = np.asarray(function.x.array, dtype=np.float64)
        if values.size != grid.n_points:
            # DOLFINx may include block data for vector spaces; this runner only
            # plots scalar CG functions, so mismatch means the plot is unsafe.
            raise ValueError(
                f"function values have length {values.size}, but VTK grid has {grid.n_points} points"
            )
        grid.point_data[scalar_name] = values
        return grid

    @staticmethod
    def _safe_clim(values: np.ndarray) -> tuple[float, float]:
        finite = np.asarray(values, dtype=np.float64)
        minimum = float(np.nanmin(finite))
        maximum = float(np.nanmax(finite))
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            return 0.0, 1.0
        if minimum == maximum:
            return minimum, minimum + 1.0
        return minimum, maximum

    def _plot(
            self,
            fields: list[fem.Function],
            titles: list[str],
            *,
            save_path: Path | None,
            show: bool,
            window_size: tuple[int, int],
            nt: int,
            ndof: int,
    ) -> None:
        if not fields or self.comm.rank != 0:
            return
        try:
            import pyvista as pv

            plotter = pv.Plotter(
                shape=(1, len(fields)),
                window_size=list(window_size),
                off_screen=save_path is not None or self.args.plot_off_screen,
            )
            for index, (field, title) in enumerate(zip(fields, titles)):
                plotter.subplot(0, index)
                scalar_name = field.name or f"field_{index}"
                grid = self._function_grid(field, scalar_name=scalar_name)
                values = grid.point_data[scalar_name]
                plotter.add_mesh(
                    grid,
                    scalars=scalar_name,
                    cmap="viridis",
                    clim=self._safe_clim(values),
                    show_edges=False,
                    scalar_bar_args={
                        "vertical": False,
                        "width": 0.55,
                        "height": 0.08,
                        "position_x": 0.225,
                        "position_y": 0.02,
                    },
                )
                plotter.add_mesh(
                    grid.extract_all_edges(),
                    color="black",
                    line_width=1.0,
                    opacity=0.45,
                )
                plotter.add_text(
                    f"{title}\nnt={nt} ndof={ndof}",
                    position="upper_edge",
                    font_size=11,
                    shadow=False,
                )
                plotter.enable_parallel_projection()
                plotter.view_xy()
                plotter.show_grid(color=(100, 100, 100, 0.15))
            if len(fields) > 1:
                plotter.link_views()
            if save_path is not None:
                plotter.screenshot(str(save_path))
            if show and not self.args.plot_off_screen:
                plotter.show(interactive_update=True, auto_close=False)
                self._wait_for_enter(plotter)
            plotter.close()
        except Exception as exc:
            print(f"PLOT_SKIP stage={titles[0] if titles else 'unknown'} error={type(exc).__name__}: {exc}", flush=True)

    def emit(
            self,
            fields: list[fem.Function],
            titles: list[str],
            *,
            stage: str,
            ieps,
            k,
            eps_phi: float,
            residual: float,
            metrics: dict[str, float],
            token: str,
            save: bool,
            show: bool,
            nt: int,
            ndof: int,
    ) -> None:
        if not self.active:
            return
        save_path = None
        if self.args.save_frames and save:
            save_path = self.frame_dir / f"{self.run_tag}_frame_{self.frame_counter:04d}_{token}.png"
        if save_path is not None:
            self._plot(
                fields,
                titles,
                save_path=save_path,
                show=False,
                window_size=(self.args.frame_window_width, self.args.frame_window_height),
                nt=nt,
                ndof=ndof,
            )
            if self.frame_writer is not None:
                self.frame_writer.writerow({
                    "frame": self.frame_counter,
                    "runTag": self.run_tag,
                    "stage": stage,
                    "ieps": ieps,
                    "k": k,
                    "nt": nt,
                    "ndof": ndof,
                    "epsPhi": eps_phi,
                    "resEuclid": residual,
                    "massRho": metrics.get("massRho", ""),
                    "maxRho": metrics.get("maxRho", ""),
                    "activeArea": metrics.get("activeArea", ""),
                    "plateauArea": metrics.get("plateauArea", ""),
                    "relRhoDesign": metrics.get("relRhoDesign", ""),
                    "filename": save_path,
                })
            self.frame_counter += 1
        if self.args.plot and show:
            self._plot(
                fields,
                titles,
                save_path=None,
                show=not self.args.plot_off_screen,
                window_size=(self.args.plot_window_width, self.args.plot_window_height),
                nt=nt,
                ndof=ndof,
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--mesh", type=Path, default=None, help="Gmsh .msh file; use saved Python initial_mesh.msh for exact comparison")
    parser.add_argument("--mesh-size", type=float, default=0.08)
    parser.add_argument("--star-n", type=int, default=260)
    parser.add_argument("--star-r0", type=float, default=1.5)
    parser.add_argument("--star-amp", type=float, default=0.32)
    parser.add_argument("--star-mode", type=int, default=5)
    parser.add_argument("--gmsh-verbosity", type=int, default=0)
    parser.add_argument("--gmsh-algorithm", type=int, default=None)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--quad-degree", type=int, default=None)
    parser.add_argument("--alphaT1", dest="alpha_t1", type=float, default=None)
    parser.add_argument("--alphaT2", dest="alpha_t2", type=float, default=None)
    parser.add_argument("--betaPhi1", dest="beta_phi1", type=float, default=None)
    parser.add_argument("--betaPhi2", dest="beta_phi2", type=float, default=None)
    parser.add_argument(
        "--phi-window-source",
        choices=("phi-design", "torsion"),
        default="phi-design",
        help=(
            "scale the nonlinear phi window by max(phiDesign), or reuse the "
            "absolute torsion thresholds alphaT1*Tmax and alphaT2*Tmax"
        ),
    )
    parser.add_argument(
        "--phi-window-torsion-width-scale",
        type=float,
        default=1.0,
        help=(
            "when --phi-window-source=torsion, use "
            "c2Phi-c1Phi = scale * (alphaT2-alphaT1) * Tmax"
        ),
    )
    parser.add_argument(
        "--phi-window-torsion-shift-scale",
        type=float,
        default=0.0,
        help=(
            "when --phi-window-source=torsion, shift both nonlinear-window "
            "edges by scale * (alphaT2-alphaT1) * Tmax before applying the "
            "torsion width scale"
        ),
    )
    parser.add_argument(
        "--eps-t-ratio",
        dest="eps_t_ratio",
        type=float,
        default=None,
        help="torsion design smoothing ratio: epsT = ratio * (alphaT2-alphaT1) * Tmax",
    )
    parser.add_argument(
        "--eps-phi-ratios",
        "--eps-ratios",
        dest="eps_phi_ratios",
        default=None,
        help=(
            "comma-separated nonlinear epsilon continuation ratios; each "
            "epsPhi = ratio * (c2Phi-c1Phi). Example: 0.11,0.08,0.06"
        ),
    )
    parser.add_argument("--rho-amp", type=float, default=None)
    parser.add_argument("--max-it", type=int, default=None)
    parser.add_argument("--tol-res", type=float, default=None)
    parser.add_argument("--tol-newton", type=float, default=None)
    parser.add_argument("--linear-solver", choices=("mumps", "lu", "hypre", "gamg"), default="mumps")
    parser.add_argument("--ksp-type", default=None)
    parser.add_argument("--linear-rtol", type=float, default=1.0e-10)
    parser.add_argument("--linear-atol", type=float, default=1.0e-12)
    parser.add_argument("--linear-max-it", type=int, default=None)
    parser.add_argument("--use-mu-shift", action="store_true")
    parser.add_argument("--mu-shift", type=float, default=2.0)
    parser.add_argument("--mu-min", type=float, default=0.20)
    parser.add_argument("--mu-max", type=float, default=30.0)
    parser.add_argument("--mu-accept-factor", type=float, default=0.85)
    parser.add_argument("--mu-hard-backtrack-factor", type=float, default=1.5)
    parser.add_argument("--mu-fail-factor", type=float, default=2.0)
    parser.add_argument("--mu-stagnation-factor", type=float, default=1.25)
    parser.add_argument("--terminal-every", type=int, default=1)
    parser.add_argument("--verbosity", "-v", type=int, choices=(0, 1, 2, 3), default=1)
    parser.add_argument("--verbose-ls", action="store_true")
    parser.add_argument("--plot", action="store_true", help="show PyVista plot windows at enabled stages")
    parser.add_argument("--plot-off-screen", action="store_true", help="render plot windows off-screen")
    parser.add_argument("--plot-window-width", type=int, default=1600)
    parser.add_argument("--plot-window-height", type=int, default=700)
    parser.add_argument("--plot-initial", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-design", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-newton", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-newton-every", type=int, default=5)
    parser.add_argument("--plot-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-frames", action="store_true", help="save PyVista PNG frames at enabled stages")
    parser.add_argument("--frame-dir", type=Path, default=None)
    parser.add_argument("--frame-every", type=int, default=None)
    parser.add_argument("--frame-design", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frame-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frame-window-width", type=int, default=1600)
    parser.add_argument("--frame-window-height", type=int, default=700)
    return parser.parse_args(argv)


def params_from_args(args: argparse.Namespace) -> TorsionParameters:
    params = TorsionParameters()
    for attr in (
            "alpha_t1", "alpha_t2", "eps_t_ratio", "beta_phi1", "beta_phi2",
            "rho_amp", "max_it", "tol_res", "tol_newton",
    ):
        value = getattr(args, attr)
        if value is not None:
            setattr(params, attr, value)
    if args.eps_phi_ratios:
        params.eps_phi_ratios = tuple(float(x.strip()) for x in args.eps_phi_ratios.split(",") if x.strip())
    if not (params.alpha_t2 > params.alpha_t1):
        raise ValueError("require alphaT2 > alphaT1")
    if not (params.beta_phi2 > params.beta_phi1):
        raise ValueError("require betaPhi2 > betaPhi1")
    if not (params.eps_t_ratio > 0.0):
        raise ValueError("require eps_t_ratio > 0")
    if not params.eps_phi_ratios or any(ratio <= 0.0 for ratio in params.eps_phi_ratios):
        raise ValueError("require positive eps phi continuation ratios")
    return params


def run_strategy(args: argparse.Namespace) -> int:
    comm = MPI.COMM_WORLD
    params = params_from_args(args)
    run_dir = make_run_dir(args) if comm.rank == 0 else None
    run_dir = Path(comm.bcast(str(run_dir), root=0))
    run_tag = run_dir.name
    log_dir = run_dir / "logs"
    out_dir = run_dir / "out"
    if comm.rank == 0:
        log_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
    comm.barrier()

    newton_csv = log_dir / "newton.csv"
    frame_csv = log_dir / "frames.csv"
    summary_path = out_dir / "summary.txt"
    frame_every = args.frame_every if args.frame_every is not None else args.plot_newton_every
    total_start = time.perf_counter()

    root_print(comm, "========== START STRATEGY A DOLFINX NOADAPT TORSION NEWTON ==========")
    root_print(comm, f"RUN_TAG {run_tag}")
    root_print(comm, f"RUN_DIR {run_dir}")
    root_print(comm, f"NEWTON_CSV {newton_csv}")
    root_print(comm, f"FRAME_CSV {frame_csv}")
    root_print(comm, f"SUMMARY {summary_path}")

    mesh_start = time.perf_counter()
    domain, mesh_path, geometry_mode = load_or_generate_mesh(args, run_dir, comm)
    tdim = domain.topology.dim
    nt = int(domain.topology.index_map(tdim).size_global)
    mesh_time = time.perf_counter() - mesh_start
    root_print(comm, f"GEOMETRY {geometry_mode} meshFile={mesh_path}")
    root_print(comm, f"MESH nt={nt} loadTime={mesh_time:.6f}")

    V = fem.functionspace(domain, ("Lagrange", int(args.order)))
    ndof = int(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)
    bc = boundary_bc(V)
    root_print(comm, f"SPACE order={args.order} ndof={ndof}")

    frame_fields = [
        "frame", "runTag", "stage", "ieps", "k", "nt", "ndof", "epsPhi", "resEuclid",
        "massRho", "maxRho", "activeArea", "plateauArea", "relRhoDesign", "filename",
    ]
    frame_handle = frame_csv.open("w", newline="", encoding="utf-8") if comm.rank == 0 else None
    frame_writer = csv.DictWriter(frame_handle, fieldnames=frame_fields) if comm.rank == 0 else None
    if frame_writer is not None:
        frame_writer.writeheader()
    plotter = PyVistaTorsionPlotter(args, run_tag=run_tag, run_dir=run_dir, frame_writer=frame_writer, comm=comm)

    if args.plot_initial:
        mesh_field = fem.Function(V, name="mesh")
        plotter.emit(
            [mesh_field],
            ["Initial mesh"],
            stage="INITIAL_MESH",
            ieps=-1,
            k=-1,
            eps_phi=0.0,
            residual=0.0,
            metrics={},
            token="initial_mesh",
            save=False,
            show=True,
            nt=nt,
            ndof=ndof,
        )

    qdeg = args.quad_degree if args.quad_degree is not None else max(2 * int(args.order) + 8, 12)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": qdeg})
    u_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    T = fem.Function(V, name="T")
    phi_design = fem.Function(V, name="phiDesign")
    rho_design = fem.Function(V, name="rhoDesign")
    u = fem.Function(V, name="phi")
    du = fem.Function(V, name="du")
    rho = fem.Function(V, name="rho")

    its, rel, solve_time = solve_linear_form(
        ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * dx,
        1.0 * v * dx,
        T,
        [bc],
        prefix="torsion_",
        solver=args.linear_solver,
        ksp_type=args.ksp_type,
        rtol=args.linear_rtol,
        atol=args.linear_atol,
        max_it=args.linear_max_it,
        verbosity=args.verbosity,
    )
    _, tmax = global_minmax(comm, T)
    c1_t = params.alpha_t1 * tmax
    c2_t = params.alpha_t2 * tmax
    eps_t = params.eps_t_ratio * (c2_t - c1_t)
    root_print(comm, f"SOLVER_OK problem=torsion iters={its} rel={rel:.3e} time={solve_time:.3f}")
    root_print(comm, f"TORSION Tmax={tmax:.6e} c1T={c1_t:.6e} c2T={c2_t:.6e} epsT={eps_t:.6e}")
    root_print(
        comm,
        "EPS_CONTINUATION "
        f"epsTRatio={params.eps_t_ratio:.6g} "
        f"epsPhiRatios={','.join(f'{ratio:.6g}' for ratio in params.eps_phi_ratios)}",
    )

    update_interpolated(rho_design, window_ufl(T, c1_t, c2_t, eps_t, params.rho_amp))
    rho_design_l2 = math.sqrt(max(assemble_scalar(comm, rho_design * rho_design * dx), 0.0))
    rho_design_mass = assemble_scalar(comm, rho_design * dx)
    _, rho_design_max = global_minmax(comm, rho_design)

    its, rel, solve_time = solve_linear_form(
        ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * dx,
        rho_design * v * dx,
        phi_design,
        [bc],
        prefix="phi_design_",
        solver=args.linear_solver,
        ksp_type=args.ksp_type,
        rtol=args.linear_rtol,
        atol=args.linear_atol,
        max_it=args.linear_max_it,
        verbosity=args.verbosity,
    )
    _, phi_design_max = global_minmax(comm, phi_design)
    if args.phi_window_source == "torsion":
        torsion_width = c2_t - c1_t
        c1_phi = c1_t + float(args.phi_window_torsion_shift_scale) * torsion_width
        c2_phi = c1_phi + float(args.phi_window_torsion_width_scale) * torsion_width
    else:
        c1_phi = params.beta_phi1 * phi_design_max
        c2_phi = params.beta_phi2 * phi_design_max
    width_phi = c2_phi - c1_phi
    u.x.array[:] = phi_design.x.array
    u.x.scatter_forward()
    root_print(comm, f"SOLVER_OK problem=phi_design iters={its} rel={rel:.3e} time={solve_time:.3f}")
    root_print(comm, f"PHI_DESIGN max={phi_design_max:.6e} rhoDesignMass={rho_design_mass:.6e} rhoDesignMax={rho_design_max:.6e}")
    root_print(
        comm,
        f"PHI_WINDOW source={args.phi_window_source} "
        f"torsionShiftScale={args.phi_window_torsion_shift_scale:.6e} "
        f"torsionWidthScale={args.phi_window_torsion_width_scale:.6e} "
        f"c1Phi={c1_phi:.6e} c2Phi={c2_phi:.6e} widthPhi={width_phi:.6e}",
    )

    fieldnames = [
        "record", "runTag", "ieps", "epsPhiRatio", "epsPhi", "k", "nt", "ndof",
        "resEuclid", "stepH1", "alpha", "bt", "muShift",
        "minU", "maxU", "minRho", "maxRho", "massRho", "rhoL2", "energyPhi",
        "activeArea", "plateauArea", "plateauFrac", "relRhoDesign", "annularPhiMinusC2",
        "activeDesignArea", "activeOverlapArea", "activeJaccard",
        "plateauDesignArea", "plateauOverlapArea", "plateauJaccard",
        "rhoDesignDiffL2", "massRhoMinusDesign",
        "solveTime", "metricTime", "stepTime", "linearIterations", "linearResidual", "status",
    ]
    final_metrics: dict[str, float] | None = None
    final_eps_phi = params.eps_phi_ratios[-1] * width_phi
    mu_shift = float(args.mu_shift if args.use_mu_shift else 0.0)
    stop_reasons: list[str] = []

    newton_handle = newton_csv.open("w", newline="", encoding="utf-8") if comm.rank == 0 else None
    writer = csv.DictWriter(newton_handle, fieldnames=fieldnames) if comm.rank == 0 else None
    if writer is not None:
        writer.writeheader()

    try:
        if args.plot_design:
            plotter.emit(
                [T, rho_design],
                ["Torsion T", "Torsion-designed rho"],
                stage="TORSION_DESIGN",
                ieps=-1,
                k=-1,
                eps_phi=eps_t,
                residual=0.0,
                metrics={"massRho": rho_design_mass, "maxRho": rho_design_max},
                token="torsion_design",
                save=bool(args.frame_design),
                show=True,
                nt=nt,
                ndof=ndof,
            )
            plotter.emit(
                [phi_design, rho_design],
                ["Poisson initializer phiDesign", "rhoDesign"],
                stage="PHI_DESIGN",
                ieps=-1,
                k=-1,
                eps_phi=eps_t,
                residual=0.0,
                metrics={"massRho": rho_design_mass, "maxRho": rho_design_max},
                token="phi_design",
                save=bool(args.frame_design),
                show=True,
                nt=nt,
                ndof=ndof,
            )

        for ieps, eps_ratio in enumerate(params.eps_phi_ratios):
            eps_phi = eps_ratio * width_phi
            final_eps_phi = eps_phi
            if args.use_mu_shift:
                mu_shift = max(mu_shift, float(args.mu_shift))
            else:
                mu_shift = 0.0

            update_interpolated(rho, window_ufl(u, c1_phi, c2_phi, eps_phi, params.rho_amp))
            residual_expr = (ufl.inner(ufl.grad(u), ufl.grad(v)) - window_ufl(u, c1_phi, c2_phi, eps_phi, params.rho_amp) * v) * dx
            metric_start = time.perf_counter()
            metrics = compute_metrics(
                u=u,
                rho=rho,
                rho_design=rho_design,
                rho_design_l2=rho_design_l2,
                residual_form=residual_expr,
                bc=bc,
                dx=dx,
                c2_phi=c2_phi,
                active_threshold=params.active_threshold,
                plateau_threshold=params.plateau_threshold,
                rho_amp=params.rho_amp,
            )
            final_metrics = metrics
            metric_time = time.perf_counter() - metric_start
            mass_floor = params.mass_floor_fraction * max(metrics["massRho"], rho_design_mass)
            root_print(
                comm,
                f"EPS_START ieps={ieps} epsRatio={eps_ratio:.6g} epsPhi={eps_phi:.6e} "
                f"nt={nt} ndof={ndof} resE={metrics['resEuclid']:.6e} "
                f"mass={metrics['massRho']:.6e} metricT={metric_time:.3f}",
            )
            if writer is not None:
                write_row(writer, record="EPS_START", runTag=run_tag, ieps=ieps, epsPhiRatio=eps_ratio,
                          epsPhi=eps_phi, k="NA", nt=nt, ndof=ndof, stepH1="NA", alpha="NA", bt="NA",
                          muShift=mu_shift, solveTime="NA", metricTime=metric_time, stepTime="NA",
                          linearIterations="NA", linearResidual="NA", status="EPS_START", **metrics)
                newton_handle.flush()

            reject_count = 0
            stagnation_count = 0
            eps_status = "MAX_IT"

            for k in range(params.max_it):
                step_start = time.perf_counter()
                update_interpolated(rho, window_ufl(u, c1_phi, c2_phi, eps_phi, params.rho_amp))
                residual_expr = (ufl.inner(ufl.grad(u), ufl.grad(v)) - window_ufl(u, c1_phi, c2_phi, eps_phi, params.rho_amp) * v) * dx
                metric_start = time.perf_counter()
                old_metrics = compute_metrics(
                    u=u,
                    rho=rho,
                    rho_design=rho_design,
                    rho_design_l2=rho_design_l2,
                    residual_form=residual_expr,
                    bc=bc,
                    dx=dx,
                    c2_phi=c2_phi,
                    active_threshold=params.active_threshold,
                    plateau_threshold=params.plateau_threshold,
                    rho_amp=params.rho_amp,
                )
                metric_time = time.perf_counter() - metric_start
                res_old = old_metrics["resEuclid"]

                if res_old < params.tol_res:
                    eps_status = "CONVERGED"
                    final_metrics = old_metrics
                    if writer is not None:
                        write_row(writer, record="NEWTON", runTag=run_tag, ieps=ieps, epsPhiRatio=eps_ratio,
                                  epsPhi=eps_phi, k=k, nt=nt, ndof=ndof, stepH1="NA", alpha=0.0, bt=0,
                                  muShift=mu_shift, solveTime=0.0, metricTime=metric_time,
                                  stepTime=time.perf_counter() - step_start, linearIterations="NA",
                                  linearResidual="NA", status="CONVERGED_RESIDUAL", **old_metrics)
                        newton_handle.flush()
                    root_print(
                        comm,
                        f"STEP ieps={ieps} k={k} resE={res_old:.6e} alpha=0.000e+00 bt=0 "
                        f"mu={mu_shift:.6e} maxU={old_metrics['maxU']:.6e} "
                        f"maxRho={old_metrics['maxRho']:.6e} mass={old_metrics['massRho']:.6e} "
                        f"solveT=0.000 metricT={metric_time:.3f} status=CONVERGED_RESIDUAL",
                    )
                    break

                jac_expr = (
                    (1.0 + mu_shift) * ufl.inner(ufl.grad(u_trial), ufl.grad(v))
                    - window_derivative_ufl(u, c1_phi, c2_phi, eps_phi, params.rho_amp) * u_trial * v
                ) * dx
                solve_start = time.perf_counter()
                its, lin_res, solve_time = solve_linear_form(
                    jac_expr,
                    -residual_expr,
                    du,
                    [bc],
                    prefix=f"newton_{ieps}_{k}_",
                    solver=args.linear_solver,
                    ksp_type=args.ksp_type,
                    rtol=args.linear_rtol,
                    atol=args.linear_atol,
                    max_it=args.linear_max_it,
                    verbosity=args.verbosity,
                )
                solve_time = time.perf_counter() - solve_start
                step_h1 = math.sqrt(max(assemble_scalar(comm, ufl.inner(ufl.grad(du), ufl.grad(du)) * dx), 0.0))

                if step_h1 < params.tol_newton:
                    eps_status = "CONVERGED"
                    final_metrics = old_metrics
                    if writer is not None:
                        write_row(writer, record="NEWTON", runTag=run_tag, ieps=ieps, epsPhiRatio=eps_ratio,
                                  epsPhi=eps_phi, k=k, nt=nt, ndof=ndof, stepH1=step_h1, alpha=0.0, bt=0,
                                  muShift=mu_shift, solveTime=solve_time, metricTime=metric_time,
                                  stepTime=time.perf_counter() - step_start, linearIterations=its,
                                  linearResidual=lin_res, status="CONVERGED_STEP", **old_metrics)
                        newton_handle.flush()
                    root_print(
                        comm,
                        f"STEP ieps={ieps} k={k} resE={res_old:.6e} alpha=0.000e+00 bt=0 "
                        f"mu={mu_shift:.6e} maxU={old_metrics['maxU']:.6e} "
                        f"maxRho={old_metrics['maxRho']:.6e} mass={old_metrics['massRho']:.6e} "
                        f"solveT={solve_time:.3f} metricT={metric_time:.3f} status=CONVERGED_STEP",
                    )
                    break

                if old_metrics["massRho"] < mass_floor or old_metrics["maxRho"] < params.rho_max_floor:
                    u.x.array[:] = phi_design.x.array
                    u.x.scatter_forward()
                    reject_count += 1
                    if args.use_mu_shift:
                        mu_shift = min(float(args.mu_max), float(args.mu_hard_backtrack_factor) * mu_shift)
                    eps_status = "RESET_TO_DESIGN"
                    if reject_count >= 5:
                        stop_reasons.append(f"ieps={ieps}:too_many_branch_rejections")
                        break
                    continue

                alpha = 1.0
                n_backtrack = 0
                u_old = u.x.array.copy()
                accepted = False
                trial_metrics = old_metrics
                while alpha >= params.alpha_min and n_backtrack <= params.max_backtrack:
                    u.x.array[:] = u_old + alpha * du.x.array
                    u.x.scatter_forward()
                    update_interpolated(rho, window_ufl(u, c1_phi, c2_phi, eps_phi, params.rho_amp))
                    trial_residual_expr = (
                        ufl.inner(ufl.grad(u), ufl.grad(v))
                        - window_ufl(u, c1_phi, c2_phi, eps_phi, params.rho_amp) * v
                    ) * dx
                    metric_start = time.perf_counter()
                    trial_metrics = compute_metrics(
                        u=u,
                        rho=rho,
                        rho_design=rho_design,
                        rho_design_l2=rho_design_l2,
                        residual_form=trial_residual_expr,
                        bc=bc,
                        dx=dx,
                        c2_phi=c2_phi,
                        active_threshold=params.active_threshold,
                        plateau_threshold=params.plateau_threshold,
                        rho_amp=params.rho_amp,
                    )
                    metric_time = time.perf_counter() - metric_start
                    branch_ok = trial_metrics["massRho"] >= mass_floor and trial_metrics["maxRho"] >= params.rho_max_floor
                    armijo = branch_ok and trial_metrics["resEuclid"] <= (1.0 - params.armijo_c * alpha) * res_old
                    if args.verbose_ls or args.verbosity >= 2:
                        root_print(
                            comm,
                            f"LS ieps={ieps} k={k} alpha={alpha:.6e} resNew={trial_metrics['resEuclid']:.6e} "
                            f"resOld={res_old:.6e} mass={trial_metrics['massRho']:.6e} "
                            f"maxRho={trial_metrics['maxRho']:.6e} relDesign={trial_metrics['relRhoDesign']:.6e} "
                            f"activeJ={trial_metrics['activeJaccard']:.6e} plateauJ={trial_metrics['plateauJaccard']:.6e} "
                            f"branchOK={branch_ok} armijo={armijo}",
                        )
                    if armijo:
                        accepted = True
                        break
                    alpha *= params.beta_ls
                    n_backtrack += 1

                if not accepted:
                    u.x.array[:] = u_old
                    u.x.scatter_forward()
                    reject_count += 1
                    if args.use_mu_shift:
                        mu_shift = min(float(args.mu_max), float(args.mu_fail_factor) * mu_shift)
                    eps_status = "FAIL_LS"
                    if writer is not None:
                        write_row(writer, record="NEWTON", runTag=run_tag, ieps=ieps, epsPhiRatio=eps_ratio,
                                  epsPhi=eps_phi, k=k, nt=nt, ndof=ndof, stepH1=step_h1, alpha=alpha,
                                  bt=n_backtrack, muShift=mu_shift, solveTime=solve_time,
                                  metricTime=metric_time, stepTime=time.perf_counter() - step_start,
                                  linearIterations=its, linearResidual=lin_res, status="FAIL_LS", **old_metrics)
                        newton_handle.flush()
                    if reject_count >= 5 or mu_shift >= args.mu_max:
                        stop_reasons.append(f"ieps={ieps}:too_many_failed_steps")
                        break
                    continue

                reject_count = 0
                final_metrics = trial_metrics
                step_time = time.perf_counter() - step_start
                if writer is not None:
                    write_row(writer, record="NEWTON", runTag=run_tag, ieps=ieps, epsPhiRatio=eps_ratio,
                              epsPhi=eps_phi, k=k, nt=nt, ndof=ndof, stepH1=step_h1, alpha=alpha,
                              bt=n_backtrack, muShift=mu_shift, solveTime=solve_time,
                              metricTime=metric_time, stepTime=step_time, linearIterations=its,
                              linearResidual=lin_res, status="ACCEPT", **trial_metrics)
                    newton_handle.flush()
                if args.terminal_every > 0 and (k % args.terminal_every == 0 or trial_metrics["resEuclid"] < params.tol_res):
                    detail = ""
                    if args.verbosity >= 2:
                        detail = (
                            f" activeArea={trial_metrics['activeArea']:.6e}"
                            f" activeJ={trial_metrics['activeJaccard']:.6e}"
                            f" plateauJ={trial_metrics['plateauJaccard']:.6e}"
                            f" rhoDiffL2={trial_metrics['rhoDesignDiffL2']:.6e}"
                            f" massDiff={trial_metrics['massRhoMinusDesign']:.6e}"
                        )
                    root_print(
                        comm,
                        f"STEP ieps={ieps} k={k} resE={trial_metrics['resEuclid']:.6e} "
                        f"alpha={alpha:.3e} bt={n_backtrack} mu={mu_shift:.6e} "
                        f"maxU={trial_metrics['maxU']:.6e} maxRho={trial_metrics['maxRho']:.6e} "
                        f"mass={trial_metrics['massRho']:.6e} relDesign={trial_metrics['relRhoDesign']:.6e} "
                        f"solveT={solve_time:.3f} metricT={metric_time:.3f} stepT={step_time:.3f}"
                        f"{detail} status=ACCEPT",
                    )
                if args.plot_newton and args.plot_newton_every > 0 and k % args.plot_newton_every == 0:
                    plotter.emit(
                        [u, rho],
                        [f"Newton phi ieps={ieps} k={k}", "rho=f(phi)"],
                        stage="ACCEPT",
                        ieps=ieps,
                        k=k,
                        eps_phi=eps_phi,
                        residual=trial_metrics["resEuclid"],
                        metrics=trial_metrics,
                        token=f"ieps_{ieps}_k_{k}_accept",
                        save=bool(args.save_frames and frame_every is not None and frame_every > 0 and k % frame_every == 0),
                        show=True,
                        nt=nt,
                        ndof=ndof,
                    )

                if n_backtrack <= 1:
                    if args.use_mu_shift:
                        mu_shift = max(float(args.mu_min), float(args.mu_accept_factor) * mu_shift)
                elif n_backtrack >= 8:
                    if args.use_mu_shift:
                        mu_shift = min(float(args.mu_max), float(args.mu_hard_backtrack_factor) * mu_shift)

                rel_drop = abs(res_old - trial_metrics["resEuclid"]) / max(res_old, 1.0e-30)
                if rel_drop < params.stagnation_tol:
                    stagnation_count += 1
                    if args.use_mu_shift:
                        mu_shift = min(float(args.mu_max), float(args.mu_stagnation_factor) * mu_shift)
                    if stagnation_count >= params.max_stagnation:
                        stop_reasons.append(f"ieps={ieps}:repeated_stagnation")
                        break
                else:
                    stagnation_count = 0

            if final_metrics is None:
                final_metrics = metrics
            if eps_status == "MAX_IT":
                stop_reasons.append(f"ieps={ieps}:max_it")
            root_print(
                comm,
                f"EPS_END ieps={ieps} resE={final_metrics['resEuclid']:.6e} "
                f"mass={final_metrics['massRho']:.6e} maxRho={final_metrics['maxRho']:.6e} status={eps_status}",
            )
            if writer is not None:
                write_row(writer, record="EPS_END", runTag=run_tag, ieps=ieps, epsPhiRatio=eps_ratio,
                          epsPhi=eps_phi, k="NA", nt=nt, ndof=ndof, stepH1="NA", alpha="NA", bt="NA",
                          muShift=mu_shift, solveTime="NA", metricTime="NA", stepTime="NA",
                          linearIterations="NA", linearResidual="NA", status=eps_status, **final_metrics)
                newton_handle.flush()
            if args.plot_newton and frame_every is not None and frame_every > 0:
                plotter.emit(
                    [u, rho, rho_design],
                    [f"Epsilon end phi ieps={ieps}", "rho=f(phi)", "rhoDesign"],
                    stage="EPS_END",
                    ieps=ieps,
                    k=-1,
                    eps_phi=eps_phi,
                    residual=final_metrics["resEuclid"],
                    metrics=final_metrics,
                    token=f"ieps_{ieps}_eps_end",
                    save=bool(args.save_frames),
                    show=False,
                    nt=nt,
                    ndof=ndof,
                )
    finally:
        if newton_handle is not None:
            newton_handle.close()

    if final_metrics is None:
        raise RuntimeError("no final metrics were computed")
    if args.plot_final:
        plotter.emit(
            [T, rho_design, phi_design, u, rho],
            ["Torsion T", "rhoDesign", "phiDesign", "Final phi", "Final rho=f(phi)"],
            stage="FINAL",
            ieps=len(params.eps_phi_ratios),
            k=-1,
            eps_phi=final_eps_phi,
            residual=final_metrics["resEuclid"],
            metrics=final_metrics,
            token="final",
            save=bool(args.frame_final),
            show=True,
            nt=nt,
            ndof=ndof,
        )
    if frame_handle is not None:
        frame_handle.close()

    elapsed = time.perf_counter() - total_start
    final_status = "OK" if not stop_reasons else "NONCONVERGED"
    root_print(
        comm,
        f"FINAL resE={final_metrics['resEuclid']:.6e} maxPhi={final_metrics['maxU']:.6e} "
        f"maxRho={final_metrics['maxRho']:.6e} mass={final_metrics['massRho']:.6e} "
        f"relDesign={final_metrics['relRhoDesign']:.6e}",
    )
    root_print(comm, f"FINAL_STATUS {final_status}")
    root_print(comm, f"TIME_TOTAL {elapsed:.3f}")
    root_print(comm, "========== END STRATEGY A DOLFINX NOADAPT TORSION NEWTON ==========")

    if comm.rank == 0:
        with summary_path.open("w", encoding="utf-8") as handle:
            handle.write(f"runTag {run_tag}\n")
            handle.write(f"runDir {run_dir}\n")
            handle.write(f"meshFile {mesh_path}\n")
            handle.write(f"geometryMode {geometry_mode}\n")
            handle.write(f"nt {nt}\n")
            handle.write(f"ndof {ndof}\n")
            handle.write(f"order {args.order}\n")
            handle.write(f"quadDegree {qdeg}\n")
            handle.write(f"linearSolver {args.linear_solver}\n")
            handle.write(f"useMuShift {int(args.use_mu_shift)}\n")
            handle.write(f"muShiftFinal {mu_shift}\n")
            handle.write(f"muMin {args.mu_min}\n")
            handle.write(f"muAcceptFactor {args.mu_accept_factor}\n")
            handle.write(f"alphaT1 {params.alpha_t1}\n")
            handle.write(f"alphaT2 {params.alpha_t2}\n")
            handle.write(f"c1T {c1_t}\n")
            handle.write(f"c2T {c2_t}\n")
            handle.write(f"epsTRatio {params.eps_t_ratio}\n")
            handle.write(f"epsT {eps_t}\n")
            handle.write(f"betaPhi1 {params.beta_phi1}\n")
            handle.write(f"betaPhi2 {params.beta_phi2}\n")
            handle.write(f"phiWindowSource {args.phi_window_source}\n")
            handle.write(f"phiWindowTorsionShiftScale {args.phi_window_torsion_shift_scale}\n")
            handle.write(f"phiWindowTorsionWidthScale {args.phi_window_torsion_width_scale}\n")
            handle.write(f"c1Phi {c1_phi}\n")
            handle.write(f"c2Phi {c2_phi}\n")
            handle.write(f"epsPhiRatios {','.join(str(ratio) for ratio in params.eps_phi_ratios)}\n")
            handle.write(f"epsPhi {final_eps_phi}\n")
            handle.write(f"resEuclid {final_metrics['resEuclid']}\n")
            handle.write(f"massRho {final_metrics['massRho']}\n")
            handle.write(f"maxRho {final_metrics['maxRho']}\n")
            handle.write(f"minPhi {final_metrics['minU']}\n")
            handle.write(f"maxPhi {final_metrics['maxU']}\n")
            handle.write(f"activeArea {final_metrics['activeArea']}\n")
            handle.write(f"plateauArea {final_metrics['plateauArea']}\n")
            handle.write(f"plateauFrac {final_metrics['plateauFrac']}\n")
            handle.write(f"activeDesignArea {final_metrics['activeDesignArea']}\n")
            handle.write(f"activeOverlapArea {final_metrics['activeOverlapArea']}\n")
            handle.write(f"activeJaccard {final_metrics['activeJaccard']}\n")
            handle.write(f"plateauDesignArea {final_metrics['plateauDesignArea']}\n")
            handle.write(f"plateauOverlapArea {final_metrics['plateauOverlapArea']}\n")
            handle.write(f"plateauJaccard {final_metrics['plateauJaccard']}\n")
            handle.write(f"rhoDesignDiffL2 {final_metrics['rhoDesignDiffL2']}\n")
            handle.write(f"relRhoDesign {final_metrics['relRhoDesign']}\n")
            handle.write(f"massRhoMinusDesign {final_metrics['massRhoMinusDesign']}\n")
            handle.write(f"annularPhiMinusC2 {final_metrics['annularPhiMinusC2']}\n")
            handle.write(f"finalStatus {final_status}\n")
            handle.write(f"stopReasons {';'.join(stop_reasons)}\n")
            handle.write(f"timeTotal {elapsed}\n")
    return 0 if final_status == "OK" else 2


def main(argv: list[str] | None = None) -> int:
    return run_strategy(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
