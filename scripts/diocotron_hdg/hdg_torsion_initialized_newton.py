"""HDG torsion-initialized Newton solve on the native smooth star domain.

This script mirrors the FreeFEM torsion/Newton run at the algorithmic level
while keeping the HDG unknowns and residuals explicit:

* torsion-designed initializer on the smooth star;
* epsilon continuation for the semilinear window;
* HDG residual assembled from the element and trace equations.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RUN_LOG_ROOT = REPO_ROOT / "run_logs" / "hdg_torsion_initialized_newton"

from hdgfem.assembly import hdg as hdg_assembly
from hdgfem.assembly.hdg_gram import CondensedHDGGramInverse, build_flux_jump_gram_inverse
from hdgfem.assembly.matrices_numpy import scalar_volume_residual
from hdgfem.assembly.projection import (
    field_from_moments,
    l2_from_values,
    mass_from_values,
    project_quadrature_values,
)
from hdgfem.core.mesh import (
    gmsh_smooth_star_mesh,
    mesh_edge_min_max,
)
from hdgfem.core.space import DGField, DGSpace
from hdgfem.io.plot import add_field_to_plotter, reference_plot_points
from hdgfem.solvers.diff_rea import (
    DiffusionReactionHDGOptions,
    DiffusionReactionHDGSolver,
    flux_coefficients,
    hdg_residual,
)


@dataclass
class TorsionParameters:
    """FreeFEM defaults for the native star torsion-initialized Newton run."""

    alpha_t1: float = 0.60
    alpha_t2: float = 0.70
    eps_t_ratio: float = 0.06
    rho_amp: float = 1.0
    beta_phi1: float = 0.60
    beta_phi2: float = 0.70
    eps_phi_ratios: tuple[float, ...] = (0.11, 0.08, 0.06)
    rho_extrema_resolution: int = 32
    rho_extrema_chunk_elements: int = 2048
    max_it: int = 70
    tol_res: float = 1.0e-10
    tol_newton: float = 1.0e-10
    beta_ls: float = 0.5
    armijo_c: float = 1.0e-6
    alpha_min: float = 1.0e-7
    max_backtrack: int = 30
    mu_shift: float = 2.0
    mu_min: float = 0.20
    mu_max: float = 30.0
    mass_floor_fraction: float = 0.01
    rho_max_floor: float = 0.02
    max_stagnation: int = 5
    stagnation_tol: float = 1.0e-5
    active_threshold: float = 0.05
    plateau_threshold: float = 0.90


@dataclass
class State:
    """Current HDG nonlinear state."""

    u: DGField
    flux: np.ndarray
    trace: np.ndarray
    rho: DGField
    residual: np.ndarray
    residual_volume: np.ndarray
    residual_coeff_l2: float
    residual_volume_coeff_l2: float
    residual_primal_coeff_l2: float
    residual_flux_coeff_l2: float
    residual_trace_coeff_l2: float
    residual_hdg: float
    residual_hdg_squared: float
    merit: float
    merit_squared: float


def residual_summary(state: State) -> str:
    return (
        f"resVolume={state.residual_volume_coeff_l2:.6e} "
        f"resPrimal={state.residual_primal_coeff_l2:.6e} "
        f"resFlux={state.residual_flux_coeff_l2:.6e} "
        f"resTrace={state.residual_trace_coeff_l2:.6e} "
        f"resFull={state.residual_coeff_l2:.6e}"
    )


def metrics_summary(metrics: dict[str, float]) -> str:
    return (
        f"minRho={metrics['min_rho']:.6e} maxRho={metrics['max_rho']:.6e} "
        f"mass={metrics['mass_rho']:.6e} "
        f"activeArea={metrics['active_area']:.6e} plateauArea={metrics['plateau_area']:.6e} "
        f"plateauFrac={metrics['plateau_frac']:.6e} "
        f"minPhi={metrics['min_u']:.6e} maxPhi={metrics['max_u']:.6e} "
        f"annularPhiMinusC2={metrics['annular_phi_minus_c2']:.6e}"
    )


def active_mu(args: argparse.Namespace, mu_shift: float) -> float:
    return float(mu_shift) if args.newton_shift_mode == "freefem" else 0.0


def damping_summary(args: argparse.Namespace, mu_shift: float) -> str:
    mu_eff = active_mu(args, mu_shift)
    return f"muEff={mu_eff:.3e} diffusion={1.0 + mu_eff:.3e}"


def log3(args: argparse.Namespace, message: str) -> None:
    """Print detailed strategy diagnostics at verbosity level 3."""
    if int(args.verbosity) >= 3:
        print(message, flush=True)


def log2(args: argparse.Namespace, message: str) -> None:
    """Print timing and line-search diagnostics at verbosity level 2."""
    if int(args.verbosity) >= 2:
        print(message, flush=True)


@dataclass
class DesignState:
    """Torsion-designed fields used by the semilinear initializer."""

    torsion: DGField
    rho_design: DGField
    phi_design: DGField
    c1_t: float
    c2_t: float
    eps_t: float


def zero_boundary(x, y):
    return np.zeros_like(x, dtype=np.float64)


def one_source(x, y):
    return np.ones_like(x, dtype=np.float64)


def logistic(z: np.ndarray, eps: float) -> np.ndarray:
    zz = np.asarray(z, dtype=np.float64) / float(eps)
    out = np.empty_like(zz)
    out[zz > 50.0] = 1.0
    out[zz < -50.0] = 0.0
    mask = (zz >= -50.0) & (zz <= 50.0)
    out[mask] = 1.0 / (1.0 + np.exp(-zz[mask]))
    return out


def softplus(z: np.ndarray, eps: float) -> np.ndarray:
    zz = np.asarray(z, dtype=np.float64) / float(eps)
    out = np.empty_like(zz)
    out[zz > 50.0] = z[zz > 50.0]
    out[zz < -50.0] = eps * np.exp(zz[zz < -50.0])
    mask = (zz >= -50.0) & (zz <= 50.0)
    out[mask] = eps * np.log1p(np.exp(zz[mask]))
    return out


def window_values(values: np.ndarray, c1: float, c2: float, eps: float, amp: float) -> np.ndarray:
    return amp * (logistic(values - c1, eps) - logistic(values - c2, eps))


def window_derivative(values: np.ndarray, c1: float, c2: float, eps: float, amp: float) -> np.ndarray:
    s1 = logistic(values - c1, eps)
    s2 = logistic(values - c2, eps)
    return amp * (s1 * (1.0 - s1) - s2 * (1.0 - s2)) / eps


def window_primitive(values: np.ndarray, c1: float, c2: float, eps: float, amp: float) -> np.ndarray:
    return amp * (softplus(values - c1, eps) - softplus(values - c2, eps))


def validate_band(c1: float, c2: float, *, name: str) -> None:
    if not (math.isfinite(c1) and math.isfinite(c2)):
        raise ValueError(f"{name} band endpoints must be finite; got c1={c1}, c2={c2}")
    if c2 <= c1:
        raise ValueError(f"{name} band requires c2 > c1; got c1={c1}, c2={c2}")


def resolve_scaled_band(
        *,
        scale: float,
        lower_ratio: float,
        upper_ratio: float,
        name: str,
) -> tuple[float, float]:
    c1 = float(lower_ratio) * float(scale)
    c2 = float(upper_ratio) * float(scale)
    validate_band(c1, c2, name=name)
    return c1, c2


def rho_extrema_from_field(
        field: DGField,
        *,
        c1: float,
        c2: float,
        eps: float,
        amp: float,
        resolution: int,
        chunk_elements: int,
) -> tuple[float, float, float, float]:
    """Return refined extrema of ``rho=f_eps(field)`` and ``field``.

    The assembly quadrature is intentionally left untouched.  For diagnostics
    we sample ``field`` on a dense reference lattice and then compute extrema
    of the one-dimensional window on each sampled ``field`` range.  Since the
    window is unimodal for ``c2 > c1``, this captures the analytic peak when
    the sampled field range crosses the band midpoint.
    """
    validate_band(c1, c2, name="rho extrema")
    if eps <= 0.0 or not math.isfinite(eps):
        raise ValueError(f"rho extrema require positive finite eps; got {eps}")
    resolution = max(2, int(resolution))
    chunk_elements = max(1, int(chunk_elements))
    reference_points = reference_plot_points(resolution)
    basis = field.space.basis_at(reference_points)
    peak_location = 0.5 * (float(c1) + float(c2))
    peak_value = float(window_values(np.array([peak_location]), c1, c2, eps, amp)[0])

    min_phi = math.inf
    max_phi = -math.inf
    min_rho = math.inf
    max_rho = -math.inf
    coeffs = field.coeffs
    for start in range(0, field.space.mesh.num_tri, chunk_elements):
        stop = min(start + chunk_elements, field.space.mesh.num_tri)
        phi_values = coeffs[start:stop] @ basis.T
        element_min_phi = np.min(phi_values, axis=1)
        element_max_phi = np.max(phi_values, axis=1)
        endpoint_rho = window_values(
            np.column_stack((element_min_phi, element_max_phi)),
            c1,
            c2,
            eps,
            amp,
        )
        element_min_rho = np.min(endpoint_rho, axis=1)
        element_max_rho = np.max(endpoint_rho, axis=1)
        crosses_peak = (element_min_phi <= peak_location) & (peak_location <= element_max_phi)
        if np.any(crosses_peak):
            element_max_rho = np.where(crosses_peak, np.maximum(element_max_rho, peak_value), element_max_rho)

        min_phi = min(min_phi, float(np.min(element_min_phi)))
        max_phi = max(max_phi, float(np.max(element_max_phi)))
        min_rho = min(min_rho, float(np.min(element_min_rho)))
        max_rho = max(max_rho, float(np.max(element_max_rho)))

    return min_rho, max_rho, min_phi, max_phi


def solve_hdg_with_fallback(
        source,
        reaction,
        boundary_condition,
        space: DGSpace,
        args: argparse.Namespace,
        *,
        diffusion: float = 1.0,
        stabilization: float = 1.0,
        initial_guess=None,
        problem_label: str,
):
    attempts = []
    if not args.skip_petsc:
        petsc_preset = str(args.hdg_petsc_preset)
        attempts.append((f"petsc_{petsc_preset}", {
            "solver": "petsc",
            "petsc_preset": petsc_preset,
            "preconditioner": None,
            "solver_rtol": args.linear_rtol,
            "solver_atol": args.linear_atol,
            "maxiter": args.linear_maxiter,
        }))
    attempts.extend([
        ("scipy_ilu_bicgstab", {
            "solver": "BICGSTAB",
            "preconditioner": "ilu",
            "solver_rtol": args.linear_rtol,
            "solver_atol": args.linear_atol,
            "maxiter": args.linear_maxiter,
            "ilu_drop_tol": args.ilu_drop_tol,
            "ilu_fill_factor": args.ilu_fill_factor,
        }),
        ("scipy_direct", {
            "solver": "direct",
            "preconditioner": None,
            "solver_rtol": args.linear_rtol,
            "solver_atol": args.linear_atol,
        }),
    ])

    last_error = None
    for label, kwargs in attempts:
        print(f"SOLVER_TRY problem={problem_label} label={label}", flush=True)
        solve_start = time.perf_counter()
        try:
            solver_verbose = max(int(args.hdg_solver_verbose), 3 if int(args.verbosity) >= 3 else 0)
            petsc_preset = kwargs.get("petsc_preset")
            is_direct_petsc = kwargs.get("solver") == "petsc" and petsc_preset in {"lu", "mumps_lu"}
            options = DiffusionReactionHDGOptions(
                diffusion=diffusion,
                stabilization=stabilization,
                boundary_mode="eliminate",
                assembly_backend=args.hdg_assembly_backend,
                local_solver_backend=args.hdg_local_solver_backend,
                initial_guess=None if is_direct_petsc else initial_guess,
                verbose=solver_verbose,
                scale_system=True,
                petsc_monitor=bool(int(args.verbosity) >= 3),
                **kwargs,
            )
            result = DiffusionReactionHDGSolver(space, options=options).solve(
                source=source,
                reaction=reaction,
                boundary_condition=boundary_condition,
            )
            solve = result.global_solve_result
            print(
                f"SOLVER_OK problem={problem_label} label={label} "
                f"iters={None if solve is None else solve.iteration_count} "
                f"rel={np.nan if solve is None else solve.relative_residual_norm:.3e} "
                f"time={time.perf_counter() - solve_start:.3f}",
                flush=True,
            )
            return result, label
        except Exception as exc:
            last_error = exc
            print(
                f"SOLVER_FAIL problem={problem_label} label={label} "
                f"time={time.perf_counter() - solve_start:.3f} error={type(exc).__name__}: {exc}",
                flush=True,
            )
            if "petsc4py is not importable" in str(exc):
                continue
    raise RuntimeError(f"all HDG solver attempts failed for {problem_label}") from last_error


def build_state(
        space: DGSpace,
        u: DGField,
        flux: np.ndarray,
        trace: np.ndarray,
        *,
        c1_phi: float,
        c2_phi: float,
        eps_phi: float,
        rho_amp: float,
        tau: float,
        residual_norm: str,
        gram_inverse: CondensedHDGGramInverse | None,
        gram_rtol: float,
        gram_atol: float,
        gram_maxiter: int | None,
) -> State:
    rho_values = window_values(u.values(), c1_phi, c2_phi, eps_phi, rho_amp)
    rho = project_quadrature_values(space, rho_values, name="rho")
    residual = hdg_residual(u, flux, trace, source_values=rho_values, stabilization=tau)
    volume_residual = scalar_volume_residual(u, rho_values)
    volume_l2 = float(np.linalg.norm(volume_residual))
    coeff_l2 = float(np.linalg.norm(residual))
    local_size = space.mesh.num_tri * 3 * space.el_dof
    local_residual = residual[:local_size].reshape(space.mesh.num_tri, 3 * space.el_dof)
    primal_l2 = float(np.linalg.norm(local_residual[:, :space.el_dof]))
    flux_l2 = float(np.linalg.norm(local_residual[:, space.el_dof:]))
    trace_l2 = float(np.linalg.norm(residual[local_size:]))
    if residual_norm == "euclid":
        residual_hdg2 = coeff_l2 * coeff_l2
        merit = coeff_l2
        merit_squared = residual_hdg2
    elif residual_norm == "hdg-local":
        residual_hdg2 = primal_l2 * primal_l2
        merit = primal_l2
        merit_squared = residual_hdg2
    elif residual_norm == "edp-volume":
        residual_hdg2 = volume_l2 * volume_l2
        merit = volume_l2
        merit_squared = residual_hdg2
    elif residual_norm == "hdg":
        if gram_inverse is None:
            raise ValueError("gram_inverse is required when residual_norm='hdg'")
        residual_hdg2, diag = gram_inverse.dual_norm_squared(
            residual,
            rtol=gram_rtol,
            atol=gram_atol,
            maxiter=gram_maxiter,
        )
        if diag.info != 0:
            print(
                f"GRAM_WARNING info={diag.info} iterations={diag.iterations} rel={diag.relative_residual:.3e}",
                flush=True,
            )
        merit = float(np.sqrt(max(residual_hdg2, 0.0)))
        merit_squared = float(residual_hdg2)
    else:
        raise ValueError("residual_norm must be 'euclid', 'hdg-local', 'edp-volume', or 'hdg'")
    return State(
        u=u,
        flux=flux,
        trace=trace,
        rho=rho,
        residual=residual,
        residual_volume=volume_residual,
        residual_coeff_l2=coeff_l2,
        residual_volume_coeff_l2=volume_l2,
        residual_primal_coeff_l2=primal_l2,
        residual_flux_coeff_l2=flux_l2,
        residual_trace_coeff_l2=trace_l2,
        residual_hdg=float(np.sqrt(max(residual_hdg2, 0.0))),
        residual_hdg_squared=float(residual_hdg2),
        merit=merit,
        merit_squared=merit_squared,
    )


def compute_metrics(
        state: State,
        design: DesignState,
        *,
        c1_phi: float,
        c2_phi: float,
        eps_phi: float,
        params: TorsionParameters,
) -> dict[str, float]:
    space = state.u.space
    u_values = state.u.values()
    rho_values = window_values(u_values, c1_phi, c2_phi, eps_phi, params.rho_amp)
    min_rho, max_rho, min_u, max_u = rho_extrema_from_field(
        state.u,
        c1=c1_phi,
        c2=c2_phi,
        eps=eps_phi,
        amp=params.rho_amp,
        resolution=params.rho_extrema_resolution,
        chunk_elements=params.rho_extrema_chunk_elements,
    )
    rho_design_values = design.rho_design.values()
    rho_design_l2 = max(l2_from_values(space, rho_design_values), 1.0e-30)
    dx_u, dy_u = state.u.grad_values()
    active = rho_values > params.active_threshold * params.rho_amp
    plateau = rho_values > params.plateau_threshold * params.rho_amp
    active_area = mass_from_values(space, active.astype(np.float64))
    plateau_area = mass_from_values(space, plateau.astype(np.float64))
    rel_design = l2_from_values(space, rho_values - rho_design_values) / rho_design_l2
    return {
        "min_u": min_u,
        "max_u": max_u,
        "min_rho": min_rho,
        "max_rho": max_rho,
        "mass_rho": mass_from_values(space, rho_values),
        "rho_l2": l2_from_values(space, rho_values),
        "energy_phi": float(np.sqrt(np.einsum(
            "K,Kq,q->",
            space.mesh.aff_jacs,
            dx_u * dx_u + dy_u * dy_u,
            space.quad_data.Krf_w,
            optimize=True,
        ))),
        "active_area": active_area,
        "plateau_area": plateau_area,
        "plateau_frac": plateau_area / max(active_area, 1.0e-30),
        "rel_rho_design": rel_design,
        "annular_phi_minus_c2": max_u - c2_phi,
    }


def write_newton_row(writer: csv.DictWriter, **row) -> None:
    writer.writerow(row)


def residual_row(state: State) -> dict[str, float]:
    return {
        "resHdg": state.residual_hdg,
        "resHdg2": state.residual_hdg_squared,
        "resCoeffL2": state.residual_coeff_l2,
        "resVolumeL2": state.residual_volume_coeff_l2,
        "resPrimalL2": state.residual_primal_coeff_l2,
        "resFluxL2": state.residual_flux_coeff_l2,
        "resTraceL2": state.residual_trace_coeff_l2,
    }


class PyVistaTorsionPlotter:
    """PyVista plotting and frame writer using ``hdgfem.io.plot`` helpers."""

    def __init__(
            self,
            args: argparse.Namespace,
            *,
            run_tag: str,
            run_dir: Path,
            frame_writer: csv.DictWriter,
    ) -> None:
        self.args = args
        self.run_tag = run_tag
        self.run_dir = run_dir
        self.frame_writer = frame_writer
        self.frame_counter = 0
        self.frame_dir = args.frame_dir or (run_dir / "frames")
        if args.save_frames:
            self.frame_dir.mkdir(parents=True, exist_ok=True)

    @property
    def active(self) -> bool:
        return bool(self.args.plot or self.args.save_frames)

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

    def _plot(
            self,
            fields: list[DGField],
            titles: list[str],
            *,
            save_path: Path | None,
            show: bool,
            window_size: tuple[int, int],
    ) -> None:
        if not fields:
            return
        try:
            import pyvista as pv

            plotter = pv.Plotter(
                shape=(1, len(fields)),
                window_size=list(window_size),
                off_screen=save_path is not None or self.args.plot_off_screen,
            )
            reference_points = reference_plot_points(self.args.plot_resolution)
            for index, (field, title) in enumerate(zip(fields, titles)):
                mesh_title = f"{title}\nnt={field.space.mesh.num_tri} ndof={field.space.ndof}"
                add_field_to_plotter(
                    plotter,
                    field,
                    reference_points=reference_points,
                    title=mesh_title,
                    subplot=(0, index),
                    show_mesh=True,
                    cmap="viridis",
                )
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
            fields: list[DGField],
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
            )
            self.frame_writer.writerow({
                "frame": self.frame_counter,
                "runTag": self.run_tag,
                "stage": stage,
                "ieps": ieps,
                "k": k,
                "nt": fields[0].space.mesh.num_tri if fields else "NA",
                "ndof": fields[0].space.ndof if fields else "NA",
                "epsPhi": eps_phi,
                "resHdg": residual,
                "massRho": metrics.get("mass_rho", ""),
                "maxRho": metrics.get("max_rho", ""),
                "activeArea": metrics.get("active_area", ""),
                "plateauArea": metrics.get("plateau_area", ""),
                "relRhoDesign": metrics.get("rel_rho_design", ""),
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
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--verbosity", "-v", type=int, default=1, help="script logging level")
    parser.add_argument("--mesh-size", type=float, default=0.08)
    parser.add_argument("--star-n", type=int, default=260)
    parser.add_argument("--star-r0", type=float, default=1.5)
    parser.add_argument("--star-amp", type=float, default=0.32)
    parser.add_argument("--star-mode", type=int, default=5)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--basis", default="dub_orth", choices=("bernstein", "hier_C0", "dub_orth"))
    parser.add_argument("--hdg-tau", type=float, default=1.0)
    band_group = parser.add_argument_group("FreeFEM band parameters")
    band_group.add_argument(
        "--alphaT1",
        dest="alpha_t1",
        type=float,
        default=None,
        help="torsion design lower fraction: c1T = alphaT1*Tmax (FreeFEM default 0.6)",
    )
    band_group.add_argument(
        "--alphaT2",
        dest="alpha_t2",
        type=float,
        default=None,
        help="torsion design upper fraction: c2T = alphaT2*Tmax (FreeFEM default 0.7)",
    )
    band_group.add_argument(
        "--betaPhi1",
        dest="beta_phi1",
        type=float,
        default=None,
        help="semilinear lower fraction: c1Phi = betaPhi1*max(phiDesign) (FreeFEM default 0.6)",
    )
    band_group.add_argument(
        "--betaPhi2",
        dest="beta_phi2",
        type=float,
        default=None,
        help="semilinear upper fraction: c2Phi = betaPhi2*max(phiDesign) (FreeFEM default 0.7)",
    )
    parser.add_argument("--alpha-t1", dest="alpha_t1", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--alpha-t2", dest="alpha_t2", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--beta-phi1", dest="beta_phi1", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--beta-phi2", dest="beta_phi2", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rho-extrema-resolution", type=int, default=32,
                        help="reference-lattice resolution for refined min/max diagnostics of rho=f_eps(phi_h)")
    parser.add_argument("--rho-extrema-chunk-elements", type=int, default=2048)
    parser.add_argument("--hdg-assembly-backend", choices=("numpy", "numba", "auto"), default="numba")
    parser.add_argument("--hdg-local-solver-backend", choices=("numpy", "numba"), default="numba")
    parser.add_argument(
        "--hdg-petsc-preset",
        choices=(
            "cg_ilu",
            "cg_icc",
            "cg_hypre",
            "cg_gamg",
            "bicgstab_ilu",
            "bicgstab_asm_ilu",
            "bicgstab_gamg",
            "gmres_ilu",
            "gmres_asm_ilu",
            "gmres_gamg",
            "lu",
            "mumps_lu",
        ),
        default="gmres_gamg",
        help="PETSc preset used for HDG global trace solves before SciPy fallbacks",
    )
    parser.add_argument("--newton-initial-guess", choices=("zero", "previous-correction"), default="zero",
                        help="initial trace guess for each Newton correction linear solve; nonlinear Newton starts from phiDesign")
    parser.add_argument("--hdg-solver-verbose", type=int, default=0)
    parser.add_argument("--gmsh-verbosity", type=int, default=0)
    parser.add_argument("--gmsh-algorithm", type=int, default=None)
    parser.add_argument("--skip-petsc", action="store_true")
    parser.add_argument("--linear-rtol", type=float, default=1.0e-10)
    parser.add_argument("--linear-atol", type=float, default=1.0e-12)
    parser.add_argument("--linear-maxiter", type=int, default=None)
    parser.add_argument("--ilu-drop-tol", type=float, default=1.0e-10)
    parser.add_argument("--ilu-fill-factor", type=float, default=50.0)
    parser.add_argument("--gram-cg-rtol", type=float, default=1.0e-9)
    parser.add_argument("--gram-cg-atol", type=float, default=1.0e-12)
    parser.add_argument("--gram-cg-maxiter", type=int, default=250)
    parser.add_argument("--gram-cg-verbose-every", type=int, default=0)
    parser.add_argument("--gram-no-verify", action="store_true")
    parser.add_argument(
        "--residual-norm",
        choices=("euclid", "hdg-local", "edp-volume", "hdg"),
        default="euclid",
        help="Newton merit norm; euclid is the full mixed HDG coefficient residual",
    )
    parser.add_argument(
        "--newton-shift-mode",
        choices=("none", "freefem"),
        default="none",
        help="none uses the convergent HDG Newton correction; freefem uses diffusion 1+mu as in the scalar EDP solver",
    )
    parser.add_argument("--eps-ratios", default=None, help="comma-separated override for epsilon continuation")
    parser.add_argument("--max-it", type=int, default=None)
    parser.add_argument("--tol-res", type=float, default=None)
    parser.add_argument("--tol-newton", type=float, default=None)
    parser.add_argument("--plot", action="store_true", help="show PyVista plot windows at enabled stages")
    parser.add_argument("--plot-off-screen", action="store_true", help="render plot windows off-screen")
    parser.add_argument("--plot-resolution", type=int, default=10)
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


def configure_params(args: argparse.Namespace) -> TorsionParameters:
    params = TorsionParameters()
    if args.alpha_t1 is not None:
        params.alpha_t1 = float(args.alpha_t1)
    if args.alpha_t2 is not None:
        params.alpha_t2 = float(args.alpha_t2)
    if args.beta_phi1 is not None:
        params.beta_phi1 = float(args.beta_phi1)
    if args.beta_phi2 is not None:
        params.beta_phi2 = float(args.beta_phi2)
    if args.eps_ratios:
        params.eps_phi_ratios = tuple(float(x.strip()) for x in args.eps_ratios.split(",") if x.strip())
    if args.rho_extrema_resolution is not None:
        params.rho_extrema_resolution = int(args.rho_extrema_resolution)
    if args.rho_extrema_chunk_elements is not None:
        params.rho_extrema_chunk_elements = int(args.rho_extrema_chunk_elements)
    if args.max_it is not None:
        params.max_it = int(args.max_it)
    if args.tol_res is not None:
        params.tol_res = float(args.tol_res)
    if args.tol_newton is not None:
        params.tol_newton = float(args.tol_newton)
    if params.rho_extrema_resolution < 2:
        raise ValueError("--rho-extrema-resolution must be at least 2")
    if params.rho_extrema_chunk_elements < 1:
        raise ValueError("--rho-extrema-chunk-elements must be positive")
    validate_band(params.alpha_t1, params.alpha_t2, name="torsion ratio")
    validate_band(params.beta_phi1, params.beta_phi2, name="phi ratio")
    return params


def _slug_for_path(value: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
    return slug.strip("_") or "run"


def csv_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if args.run_dir is None:
        prefix = f"{_slug_for_path(args.run_tag)}_" if args.run_tag else ""
        run_dir = DEFAULT_RUN_LOG_ROOT / f"{prefix}{timestamp}"
    else:
        requested = args.run_dir
        if requested.exists():
            run_dir = requested.parent / f"{requested.name}_{timestamp}"
        else:
            run_dir = requested
    suffix = 0
    unique_run_dir = run_dir
    while unique_run_dir.exists():
        suffix += 1
        unique_run_dir = run_dir.parent / f"{run_dir.name}_{suffix:03d}"
    run_dir = unique_run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    return (
        run_dir / "newton.csv",
        run_dir / "frames.csv",
        run_dir / "summary.txt",
    )


def run_strategy(args: argparse.Namespace) -> State:
    params = configure_params(args)

    newton_csv, frame_csv, summary_path = csv_paths(args)
    run_dir = newton_csv.parent
    run_tag = run_dir.name
    initial_mesh_path = run_dir / "initial_mesh.msh"
    frame_every = args.frame_every if args.frame_every is not None else args.plot_newton_every
    total_start = time.perf_counter()
    print("========== START STRATEGY A HDG TORSION NEWTON ==========")
    print(f"RUN_TAG {run_tag}")
    print(f"RUN_DIR {run_dir}")
    print(f"NEWTON_CSV {newton_csv}")
    print(f"FRAME_CSV {frame_csv}")
    print(f"INITIAL_MESH_FILE {initial_mesh_path}")
    print(f"RESIDUAL_NORM {args.residual_norm}")
    print(f"NEWTON_SHIFT_MODE {args.newton_shift_mode}")
    print(f"VERBOSITY {args.verbosity}")
    print(
        f"BAND_PARAMETERS alphaT1={params.alpha_t1} alphaT2={params.alpha_t2} "
        f"betaPhi1={params.beta_phi1} betaPhi2={params.beta_phi2}"
    )
    print(
        f"RHO_EXTREMA resolution={params.rho_extrema_resolution} "
        f"chunkElements={params.rho_extrema_chunk_elements}"
    )
    print(f"HDG_PETSC_PRESET {args.hdg_petsc_preset}")
    print("NONLINEAR_INITIAL_STATE phi_design_from_torsion_band")
    print(f"NEWTON_CORRECTION_INITIAL_GUESS {args.newton_initial_guess}")

    newton_fields = [
        "record", "runTag", "ieps", "epsPhiRatio", "epsPhi", "k", "nt", "ndof",
        "resHdg", "resHdg2", "resCoeffL2", "resVolumeL2", "resPrimalL2", "resFluxL2", "resTraceL2",
        "stepHdg", "stepFluxL2", "stepJumpL2",
        "alpha", "bt", "muEff", "minU", "maxU", "minRho", "maxRho", "massRho", "rhoL2",
        "energyPhi", "activeArea", "plateauArea", "plateauFrac", "relRhoDesign",
        "annularPhiMinusC2", "solveTime", "metricTime", "stepTime", "solver", "status",
    ]
    frame_fields = [
        "frame", "runTag", "stage", "ieps", "k", "nt", "ndof", "epsPhi", "resHdg",
        "massRho", "maxRho", "activeArea", "plateauArea", "relRhoDesign", "filename",
    ]

    with (
        newton_csv.open("w", newline="", encoding="utf-8") as newton_handle,
        frame_csv.open("w", newline="", encoding="utf-8") as frame_handle,
    ):
        newton_writer = csv.DictWriter(newton_handle, fieldnames=newton_fields)
        frame_writer = csv.DictWriter(frame_handle, fieldnames=frame_fields)
        newton_writer.writeheader()
        frame_writer.writeheader()
        plotter = PyVistaTorsionPlotter(args, run_tag=run_tag, run_dir=run_dir, frame_writer=frame_writer)

        mesh = gmsh_smooth_star_mesh(
            args.mesh_size,
            boundary_points=args.star_n,
            radius=args.star_r0,
            amplitude=args.star_amp,
            mode=args.star_mode,
            verbosity=args.gmsh_verbosity,
            algorithm=args.gmsh_algorithm,
            write_path=str(initial_mesh_path),
            msh_file_version=2.2,
        )
        initial_hmin, initial_hmax = mesh_edge_min_max(mesh)
        space = DGSpace(mesh, args.order, basis_type=args.basis)
        print(
            f"GEOMETRY smooth_star starN={args.star_n} r0={args.star_r0} "
            f"amp={args.star_amp} mode={args.star_mode} meshSize={args.mesh_size}"
        )
        print(
            f"INITIAL_MESH nt={mesh.num_tri} ndof={space.ndof} hMin={initial_hmin:.6e} "
            f"hMax={initial_hmax:.6e}"
        )
        if args.plot_initial:
            mesh_field = space.field(np.zeros(space.shape, dtype=np.float64), name="mesh")
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
            )

        torsion_result, torsion_solver = solve_hdg_with_fallback(
            one_source,
            0.0,
            zero_boundary,
            space,
            args,
            diffusion=1.0,
            stabilization=args.hdg_tau,
            problem_label="torsion",
        )
        torsion = torsion_result.field
        t_values = torsion.values()
        t_max = float(np.max(t_values))
        if t_max <= 1.0e-14:
            raise RuntimeError("torsion maximum is too small")
        c1_t, c2_t = resolve_scaled_band(
            scale=t_max,
            lower_ratio=params.alpha_t1,
            upper_ratio=params.alpha_t2,
            name="torsion",
        )
        eps_t = params.eps_t_ratio * (c2_t - c1_t)
        rho_design_values = window_values(t_values, c1_t, c2_t, eps_t, params.rho_amp)
        rho_design = project_quadrature_values(space, rho_design_values, name="rhoDesign")
        print(
            f"TORSION solver={torsion_solver} Tmax={t_max:.6e} c1T={c1_t:.6e} "
            f"c2T={c2_t:.6e} epsT={eps_t:.6e} alphaT1={params.alpha_t1:.6e} "
            f"alphaT2={params.alpha_t2:.6e}"
        )
        if args.plot_design:
            plotter.emit(
                [torsion, rho_design],
                ["Torsion T", "Torsion-designed rho"],
                stage="TORSION_DESIGN",
                ieps=-1,
                k=-1,
                eps_phi=eps_t,
                residual=0.0,
                metrics={"mass_rho": mass_from_values(space, rho_design.values()), "max_rho": float(np.max(rho_design.values()))},
                token="torsion_design",
                save=bool(args.frame_design),
                show=True,
            )

        phi_result, phi_solver = solve_hdg_with_fallback(
            rho_design,
            0.0,
            zero_boundary,
            space,
            args,
            diffusion=1.0,
            stabilization=args.hdg_tau,
            initial_guess=torsion_result.trace,
            problem_label="phi_design",
        )
        design = DesignState(
            torsion=torsion,
            rho_design=rho_design,
            phi_design=phi_result.field,
            c1_t=c1_t,
            c2_t=c2_t,
            eps_t=eps_t,
        )
        print(
            f"PHI_DESIGN solver={phi_solver} max={np.max(design.phi_design.values()):.6e} "
            f"rhoDesignMass={mass_from_values(space, rho_design.values()):.6e}"
        )
        if args.plot_design:
            plotter.emit(
                [design.phi_design, design.rho_design],
                ["Poisson initializer phiDesign", "rhoDesign"],
                stage="PHI_DESIGN",
                ieps=-1,
                k=-1,
                eps_phi=eps_t,
                residual=0.0,
                metrics={"mass_rho": mass_from_values(space, design.rho_design.values()), "max_rho": float(np.max(design.rho_design.values()))},
                token="phi_design",
                save=bool(args.frame_design),
                show=True,
            )

        phi_design_values = design.phi_design.values()
        phi_max = float(np.max(phi_design_values))
        if phi_max <= 1.0e-14:
            raise RuntimeError("phiDesign maximum is too small")
        c1_phi, c2_phi = resolve_scaled_band(
            scale=phi_max,
            lower_ratio=params.beta_phi1,
            upper_ratio=params.beta_phi2,
            name="phi",
        )
        width_phi = c2_phi - c1_phi
        eps_phi = params.eps_phi_ratios[0] * width_phi
        print(
            f"PHI_BAND phiDesignMax={phi_max:.6e} c1Phi={c1_phi:.6e} "
            f"c2Phi={c2_phi:.6e} widthPhi={width_phi:.6e} "
            f"betaPhi1={params.beta_phi1:.6e} betaPhi2={params.beta_phi2:.6e}"
        )
        u = design.phi_design
        flux = flux_coefficients(phi_result)
        if phi_result.field.space is not space:
            flux = np.zeros((2, space.mesh.num_tri, space.el_dof), dtype=np.float64)
        trace = phi_result.trace.copy()
        if trace.size != space.mesh.num_edg * space.quad_data.edg_dof:
            trace = hdg_assembly.trace_from_field_faces(u)

        gram_args = {
            "cg_rtol": args.gram_cg_rtol,
            "cg_atol": args.gram_cg_atol,
            "cg_maxiter": args.gram_cg_maxiter,
            "verbose_every": args.gram_cg_verbose_every,
            "verify_residual": not args.gram_no_verify,
        }
        gram_inverse = build_flux_jump_gram_inverse(space, **gram_args) if args.residual_norm == "hdg" else None
        state = build_state(
            space,
            u,
            flux,
            trace,
            c1_phi=c1_phi,
            c2_phi=c2_phi,
            eps_phi=eps_phi,
            rho_amp=params.rho_amp,
            tau=args.hdg_tau,
            residual_norm=args.residual_norm,
            gram_inverse=gram_inverse,
            gram_rtol=args.gram_cg_rtol,
            gram_atol=args.gram_cg_atol,
            gram_maxiter=args.gram_cg_maxiter,
        )
        setup_metrics = compute_metrics(state, design, c1_phi=c1_phi, c2_phi=c2_phi, eps_phi=eps_phi, params=params)
        write_newton_row(
            newton_writer,
            record="SETUP",
            runTag=run_tag,
            ieps="NA",
            epsPhiRatio="NA",
            epsPhi=eps_phi,
            k="NA",
            nt=space.mesh.num_tri,
            ndof=space.ndof,
            **residual_row(state),
            stepHdg="NA",
            stepFluxL2="NA",
            stepJumpL2="NA",
            alpha="NA",
            bt="NA",
            muEff=active_mu(args, params.mu_shift),
            minU=setup_metrics["min_u"],
            maxU=setup_metrics["max_u"],
            minRho=setup_metrics["min_rho"],
            maxRho=setup_metrics["max_rho"],
            massRho=setup_metrics["mass_rho"],
            rhoL2=setup_metrics["rho_l2"],
            energyPhi=setup_metrics["energy_phi"],
            activeArea=setup_metrics["active_area"],
            plateauArea=setup_metrics["plateau_area"],
            plateauFrac=setup_metrics["plateau_frac"],
            relRhoDesign=setup_metrics["rel_rho_design"],
            annularPhiMinusC2=setup_metrics["annular_phi_minus_c2"],
            solveTime="NA",
            metricTime="NA",
            stepTime="NA",
            solver="NA",
            status="INITIAL",
        )
        if args.plot_design:
            plotter.emit(
                [state.u, state.rho, design.rho_design, design.phi_design],
                ["Initial phi", "Initial rho=f(phi)", "rhoDesign", "phiDesign"],
                stage="INIT",
                ieps=-1,
                k=-1,
                eps_phi=eps_phi,
                residual=state.residual_hdg,
                metrics=setup_metrics,
                token="init",
                save=bool(args.frame_design),
                show=True,
            )

        mu_shift = params.mu_shift
        previous_correction_trace: np.ndarray | None = None
        run_status = "OK"
        stop_reasons: list[str] = []
        for ieps, eps_ratio in enumerate(params.eps_phi_ratios):
            eps_phi = eps_ratio * width_phi
            mu_shift = max(mu_shift, 2.0)
            state = build_state(
                space,
                state.u,
                state.flux,
                state.trace,
                c1_phi=c1_phi,
                c2_phi=c2_phi,
                eps_phi=eps_phi,
                rho_amp=params.rho_amp,
                tau=args.hdg_tau,
                residual_norm=args.residual_norm,
                gram_inverse=gram_inverse,
                gram_rtol=args.gram_cg_rtol,
                gram_atol=args.gram_cg_atol,
                gram_maxiter=args.gram_cg_maxiter,
            )
            metrics = compute_metrics(state, design, c1_phi=c1_phi, c2_phi=c2_phi, eps_phi=eps_phi, params=params)
            mass_initial = metrics["mass_rho"]
            rho_design_mass = mass_from_values(space, design.rho_design.values())
            mass_floor = params.mass_floor_fraction * max(mass_initial, rho_design_mass)
            print(
                f"EPS_START ieps={ieps} epsRatio={eps_ratio} epsPhi={eps_phi:.6e} "
                f"nt={space.mesh.num_tri} ndof={space.ndof} merit={state.merit:.6e} "
                f"{residual_summary(state)} {damping_summary(args, mu_shift)} {metrics_summary(metrics)}"
            )
            write_newton_row(
                newton_writer,
                record="EPS_START",
                runTag=run_tag,
                ieps=ieps,
                epsPhiRatio=eps_ratio,
                epsPhi=eps_phi,
                k="NA",
                nt=space.mesh.num_tri,
                ndof=space.ndof,
                **residual_row(state),
                stepHdg="NA",
                stepFluxL2="NA",
                stepJumpL2="NA",
                alpha="NA",
                bt="NA",
                muEff=active_mu(args, mu_shift),
                minU=metrics["min_u"],
                maxU=metrics["max_u"],
                minRho=metrics["min_rho"],
                maxRho=metrics["max_rho"],
                massRho=metrics["mass_rho"],
                rhoL2=metrics["rho_l2"],
                energyPhi=metrics["energy_phi"],
                activeArea=metrics["active_area"],
                plateauArea=metrics["plateau_area"],
                plateauFrac=metrics["plateau_frac"],
                relRhoDesign=metrics["rel_rho_design"],
                annularPhiMinusC2=metrics["annular_phi_minus_c2"],
                solveTime="NA",
                metricTime="NA",
                stepTime="NA",
                solver="NA",
                status="EPS_START",
            )

            reject_count = 0
            stagnation_count = 0
            eps_status = "MAX_IT"
            eps_stop_reason = ""

            for k in range(params.max_it):
                step_start = time.perf_counter()
                if state.merit < params.tol_res:
                    metric_start = time.perf_counter()
                    log3(args, f"NEWTON_METRICS_START ieps={ieps} k={k} status=CONVERGED_RESIDUAL")
                    metrics = compute_metrics(state, design, c1_phi=c1_phi, c2_phi=c2_phi, eps_phi=eps_phi, params=params)
                    metric_time = time.perf_counter() - metric_start
                    log3(args, f"NEWTON_METRICS_DONE ieps={ieps} k={k} time={metric_time:.6f}")
                    mu_used = active_mu(args, mu_shift)
                    write_newton_row(
                        newton_writer,
                        record="NEWTON",
                        runTag=run_tag,
                        ieps=ieps,
                        epsPhiRatio=eps_ratio,
                        epsPhi=eps_phi,
                        k=k,
                        nt=space.mesh.num_tri,
                        ndof=space.ndof,
                        **residual_row(state),
                        stepHdg="NA",
                        stepFluxL2="NA",
                        stepJumpL2="NA",
                        alpha=0.0,
                        bt=0,
                        muEff=mu_used,
                        minU=metrics["min_u"],
                        maxU=metrics["max_u"],
                        minRho=metrics["min_rho"],
                        maxRho=metrics["max_rho"],
                        massRho=metrics["mass_rho"],
                        rhoL2=metrics["rho_l2"],
                        energyPhi=metrics["energy_phi"],
                        activeArea=metrics["active_area"],
                        plateauArea=metrics["plateau_area"],
                        plateauFrac=metrics["plateau_frac"],
                        relRhoDesign=metrics["rel_rho_design"],
                        annularPhiMinusC2=metrics["annular_phi_minus_c2"],
                        solveTime=0.0,
                        metricTime=metric_time,
                        stepTime=time.perf_counter() - step_start,
                        solver="NA",
                        status="CONVERGED_RESIDUAL",
                    )
                    print(
                        f"STEP ieps={ieps} k={k} merit={state.merit:.6e} "
                        f"{residual_summary(state)} {damping_summary(args, mu_used)} "
                        f"{metrics_summary(metrics)} status=CONVERGED_RESIDUAL"
                    )
                    eps_status = "CONVERGED_RESIDUAL"
                    break

                df_values = window_derivative(state.u.values(), c1_phi, c2_phi, eps_phi, params.rho_amp)
                source_moments = hdg_assembly.mixed_u_block_rhs_from_residual(state.residual, space)
                reaction_values = -df_values
                source_h = field_from_moments(space, source_moments, name=f"newton_source_{ieps}_{k}")
                reaction_h = project_quadrature_values(space, reaction_values, name=f"newton_reaction_{ieps}_{k}")
                mu_used = active_mu(args, mu_shift)
                correction_diffusion = 1.0 + mu_used
                correction_initial_guess = None
                if (
                    args.newton_initial_guess == "previous-correction"
                    and previous_correction_trace is not None
                    and previous_correction_trace.shape == state.trace.shape
                ):
                    correction_initial_guess = previous_correction_trace

                solve_start = time.perf_counter()
                correction, solver_label = solve_hdg_with_fallback(
                    source_h,
                    reaction_h,
                    zero_boundary,
                    space,
                    args,
                    diffusion=correction_diffusion,
                    stabilization=args.hdg_tau,
                    initial_guess=correction_initial_guess,
                    problem_label=f"newton_ieps{ieps}_k{k}",
                )
                solve_time = time.perf_counter() - solve_start
                post_solve_start = time.perf_counter()
                log3(args, f"NEWTON_POST_SOLVE_START ieps={ieps} k={k}")
                du = correction.field
                flux_extract_start = time.perf_counter()
                flux_du = flux_coefficients(correction)
                flux_extract_time = time.perf_counter() - flux_extract_start
                trace_copy_start = time.perf_counter()
                trace_du = correction.trace
                previous_correction_trace = trace_du.copy()
                trace_copy_time = time.perf_counter() - trace_copy_start
                step_norm_start = time.perf_counter()
                step_norm, step_flux_l2, step_jump_l2 = hdg_assembly.h1_flux_jump_norm(du, flux_du, trace_du)
                step_norm_time = time.perf_counter() - step_norm_start
                log3(
                    args,
                    f"NEWTON_POST_SOLVE_DONE ieps={ieps} k={k} "
                    f"fluxExtract={flux_extract_time:.6f} traceCopy={trace_copy_time:.6f} "
                    f"stepNorm={step_norm_time:.6f} total={time.perf_counter() - post_solve_start:.6f} "
                    f"stepHdg={step_norm:.6e} stepFlux={step_flux_l2:.6e} stepJump={step_jump_l2:.6e}",
                )

                if step_norm < params.tol_newton:
                    metric_start = time.perf_counter()
                    log3(args, f"NEWTON_METRICS_START ieps={ieps} k={k} status=CONVERGED_STEP")
                    metrics = compute_metrics(state, design, c1_phi=c1_phi, c2_phi=c2_phi, eps_phi=eps_phi, params=params)
                    metric_time = time.perf_counter() - metric_start
                    log3(args, f"NEWTON_METRICS_DONE ieps={ieps} k={k} time={metric_time:.6f}")
                    write_newton_row(
                        newton_writer,
                        record="NEWTON",
                        runTag=run_tag,
                        ieps=ieps,
                        epsPhiRatio=eps_ratio,
                        epsPhi=eps_phi,
                        k=k,
                        nt=space.mesh.num_tri,
                        ndof=space.ndof,
                        **residual_row(state),
                        stepHdg=step_norm,
                        stepFluxL2=step_flux_l2,
                        stepJumpL2=step_jump_l2,
                        alpha=0.0,
                        bt=0,
                        muEff=mu_used,
                        minU=metrics["min_u"],
                        maxU=metrics["max_u"],
                        minRho=metrics["min_rho"],
                        maxRho=metrics["max_rho"],
                        massRho=metrics["mass_rho"],
                        rhoL2=metrics["rho_l2"],
                        energyPhi=metrics["energy_phi"],
                        activeArea=metrics["active_area"],
                        plateauArea=metrics["plateau_area"],
                        plateauFrac=metrics["plateau_frac"],
                        relRhoDesign=metrics["rel_rho_design"],
                        annularPhiMinusC2=metrics["annular_phi_minus_c2"],
                        solveTime=solve_time,
                        metricTime=metric_time,
                        stepTime=time.perf_counter() - step_start,
                        solver=solver_label,
                        status="CONVERGED_STEP",
                    )
                    print(
                        f"STEP ieps={ieps} k={k} merit={state.merit:.6e} "
                        f"{residual_summary(state)} {damping_summary(args, mu_used)} "
                        f"{metrics_summary(metrics)} status=CONVERGED_STEP"
                    )
                    eps_status = "CONVERGED_STEP"
                    break

                old_state = state
                old_merit = old_state.merit
                alpha = 1.0
                n_backtrack = 0
                accepted = False
                best_trial = None

                line_search_start = time.perf_counter()
                log2(args, f"LINE_SEARCH_START ieps={ieps} k={k} oldMerit={old_merit:.6e}")
                while alpha >= params.alpha_min and n_backtrack <= params.max_backtrack:
                    trial_start = time.perf_counter()
                    trial_u = space.field(old_state.u.coeffs + alpha * du.coeffs, name="phi")
                    trial_flux = old_state.flux + alpha * flux_du
                    trial_trace = old_state.trace + alpha * trace_du
                    trial_values_start = time.perf_counter()
                    trial_rho_values = window_values(trial_u.values(), c1_phi, c2_phi, eps_phi, params.rho_amp)
                    trial_mass = mass_from_values(space, trial_rho_values)
                    trial_max_rho = float(np.max(trial_rho_values))
                    trial_values_time = time.perf_counter() - trial_values_start
                    branch_ok = trial_mass >= mass_floor and trial_max_rho >= params.rho_max_floor
                    build_state_time = 0.0
                    if branch_ok:
                        trial_build_start = time.perf_counter()
                        trial_state = build_state(
                            space,
                            trial_u,
                            trial_flux,
                            trial_trace,
                            c1_phi=c1_phi,
                            c2_phi=c2_phi,
                            eps_phi=eps_phi,
                            rho_amp=params.rho_amp,
                            tau=args.hdg_tau,
                            residual_norm=args.residual_norm,
                            gram_inverse=gram_inverse,
                            gram_rtol=args.gram_cg_rtol,
                            gram_atol=args.gram_cg_atol,
                            gram_maxiter=args.gram_cg_maxiter,
                        )
                        build_state_time = time.perf_counter() - trial_build_start
                        merit = trial_state.merit
                    else:
                        trial_state = None
                        merit = math.inf
                    armijo = branch_ok and merit <= (1.0 - params.armijo_c * alpha) * old_merit
                    log2(
                        args,
                        f"LINE_SEARCH_TRIAL ieps={ieps} k={k} bt={n_backtrack} alpha={alpha:.6e} "
                        f"branchOk={branch_ok} merit={merit:.6e} armijo={armijo} "
                        f"rhoEvalMass={trial_values_time:.6f} buildState={build_state_time:.6f} "
                        f"total={time.perf_counter() - trial_start:.6f} mass={trial_mass:.6e} "
                        f"maxRho={trial_max_rho:.6e}",
                    )
                    if armijo:
                        best_trial = trial_state
                        accepted = True
                        break
                    alpha *= params.beta_ls
                    n_backtrack += 1
                log2(
                    args,
                    f"LINE_SEARCH_DONE ieps={ieps} k={k} accepted={accepted} bt={n_backtrack} "
                    f"alpha={alpha:.6e} time={time.perf_counter() - line_search_start:.6f}",
                )

                if not accepted:
                    reject_count += 1
                    if args.newton_shift_mode == "freefem":
                        mu_shift = min(params.mu_max, 2.0 * mu_shift)
                    metric_start = time.perf_counter()
                    log3(args, f"NEWTON_METRICS_START ieps={ieps} k={k} status=FAIL_LS")
                    metrics = compute_metrics(old_state, design, c1_phi=c1_phi, c2_phi=c2_phi, eps_phi=eps_phi, params=params)
                    metric_time = time.perf_counter() - metric_start
                    log3(args, f"NEWTON_METRICS_DONE ieps={ieps} k={k} time={metric_time:.6f}")
                    write_newton_row(
                        newton_writer,
                        record="NEWTON",
                        runTag=run_tag,
                        ieps=ieps,
                        epsPhiRatio=eps_ratio,
                        epsPhi=eps_phi,
                        k=k,
                        nt=space.mesh.num_tri,
                        ndof=space.ndof,
                        **residual_row(old_state),
                        stepHdg=step_norm,
                        stepFluxL2=step_flux_l2,
                        stepJumpL2=step_jump_l2,
                        alpha=alpha,
                        bt=n_backtrack,
                        muEff=mu_used,
                        minU=metrics["min_u"],
                        maxU=metrics["max_u"],
                        minRho=metrics["min_rho"],
                        maxRho=metrics["max_rho"],
                        massRho=metrics["mass_rho"],
                        rhoL2=metrics["rho_l2"],
                        energyPhi=metrics["energy_phi"],
                        activeArea=metrics["active_area"],
                        plateauArea=metrics["plateau_area"],
                        plateauFrac=metrics["plateau_frac"],
                        relRhoDesign=metrics["rel_rho_design"],
                        annularPhiMinusC2=metrics["annular_phi_minus_c2"],
                        solveTime=solve_time,
                        metricTime=metric_time,
                        stepTime=time.perf_counter() - step_start,
                        solver=solver_label,
                        status="FAIL_LS",
                    )
                    print(
                        f"STEP ieps={ieps} k={k} merit={old_state.merit:.6e} "
                        f"{residual_summary(old_state)} {metrics_summary(metrics)} "
                        f"alpha={alpha:.3e} bt={n_backtrack} {damping_summary(args, mu_used)} "
                        f"status=FAIL_LS"
                    )
                    if reject_count >= 5 or (args.newton_shift_mode == "freefem" and mu_shift >= params.mu_max):
                        eps_status = "STOPPED"
                        eps_stop_reason = "too_many_failed_steps"
                        run_status = "NONCONVERGED"
                        stop_reasons.append(f"ieps={ieps}:{eps_stop_reason}")
                        print(f"EPS_STOP ieps={ieps} reason={eps_stop_reason}")
                        break
                    continue

                reject_count = 0
                state = best_trial
                metrics_start = time.perf_counter()
                log3(args, f"NEWTON_METRICS_START ieps={ieps} k={k} status=ACCEPT")
                metrics = compute_metrics(state, design, c1_phi=c1_phi, c2_phi=c2_phi, eps_phi=eps_phi, params=params)
                metric_time = time.perf_counter() - metrics_start
                log3(args, f"NEWTON_METRICS_DONE ieps={ieps} k={k} time={metric_time:.6f}")
                write_newton_row(
                    newton_writer,
                    record="NEWTON",
                    runTag=run_tag,
                    ieps=ieps,
                    epsPhiRatio=eps_ratio,
                    epsPhi=eps_phi,
                    k=k,
                    nt=space.mesh.num_tri,
                    ndof=space.ndof,
                    **residual_row(state),
                    stepHdg=step_norm,
                    stepFluxL2=step_flux_l2,
                    stepJumpL2=step_jump_l2,
                    alpha=alpha,
                    bt=n_backtrack,
                    muEff=mu_used,
                    minU=metrics["min_u"],
                    maxU=metrics["max_u"],
                    minRho=metrics["min_rho"],
                    maxRho=metrics["max_rho"],
                    massRho=metrics["mass_rho"],
                    rhoL2=metrics["rho_l2"],
                    energyPhi=metrics["energy_phi"],
                    activeArea=metrics["active_area"],
                    plateauArea=metrics["plateau_area"],
                    plateauFrac=metrics["plateau_frac"],
                    relRhoDesign=metrics["rel_rho_design"],
                    annularPhiMinusC2=metrics["annular_phi_minus_c2"],
                    solveTime=solve_time,
                    metricTime=metric_time,
                    stepTime=time.perf_counter() - step_start,
                    solver=solver_label,
                    status="ACCEPT",
                )
                print(
                    f"STEP ieps={ieps} k={k} merit={state.merit:.6e} "
                    f"{residual_summary(state)} alpha={alpha:.3e} bt={n_backtrack} "
                    f"{damping_summary(args, mu_used)} {metrics_summary(metrics)} status=ACCEPT"
                )
                if args.plot_newton and args.plot_newton_every > 0 and k % args.plot_newton_every == 0:
                    plotter.emit(
                        [state.u, state.rho],
                        [f"Newton phi ieps={ieps} k={k}", "rho=f(phi)"],
                        stage="ACCEPT",
                        ieps=ieps,
                        k=k,
                        eps_phi=eps_phi,
                        residual=state.residual_hdg,
                        metrics=metrics,
                        token=f"ieps_{ieps}_k_{k}_accept",
                        save=bool(args.save_frames and frame_every is not None and frame_every > 0 and k % frame_every == 0),
                        show=True,
                    )

                if args.newton_shift_mode == "freefem":
                    if n_backtrack <= 1:
                        mu_shift = max(params.mu_min, 0.85 * mu_shift)
                    elif n_backtrack >= 8:
                        mu_shift = min(params.mu_max, 1.5 * mu_shift)

                rel_drop = abs(old_merit - state.merit) / max(old_merit, 1.0e-30)
                if rel_drop < params.stagnation_tol:
                    stagnation_count += 1
                    if args.newton_shift_mode == "freefem":
                        mu_shift = min(params.mu_max, 1.25 * mu_shift)
                    if stagnation_count >= params.max_stagnation:
                        eps_status = "STOPPED"
                        eps_stop_reason = "repeated_stagnation"
                        run_status = "NONCONVERGED"
                        stop_reasons.append(f"ieps={ieps}:{eps_stop_reason}")
                        print(f"EPS_STOP ieps={ieps} reason={eps_stop_reason}")
                        break
                else:
                    stagnation_count = 0

            if eps_status == "MAX_IT":
                run_status = "NONCONVERGED"
                eps_stop_reason = "max_it_reached"
                stop_reasons.append(f"ieps={ieps}:{eps_stop_reason}")
            metrics = compute_metrics(state, design, c1_phi=c1_phi, c2_phi=c2_phi, eps_phi=eps_phi, params=params)
            write_newton_row(
                newton_writer,
                record="EPS_END",
                runTag=run_tag,
                ieps=ieps,
                epsPhiRatio=eps_ratio,
                epsPhi=eps_phi,
                k="NA",
                nt=space.mesh.num_tri,
                ndof=space.ndof,
                **residual_row(state),
                stepHdg="NA",
                stepFluxL2="NA",
                stepJumpL2="NA",
                alpha="NA",
                bt="NA",
                muEff=active_mu(args, mu_shift),
                minU=metrics["min_u"],
                maxU=metrics["max_u"],
                minRho=metrics["min_rho"],
                maxRho=metrics["max_rho"],
                massRho=metrics["mass_rho"],
                rhoL2=metrics["rho_l2"],
                energyPhi=metrics["energy_phi"],
                activeArea=metrics["active_area"],
                plateauArea=metrics["plateau_area"],
                plateauFrac=metrics["plateau_frac"],
                relRhoDesign=metrics["rel_rho_design"],
                annularPhiMinusC2=metrics["annular_phi_minus_c2"],
                solveTime="NA",
                metricTime="NA",
                stepTime="NA",
                solver="NA",
                status=f"EPS_END_{eps_status}",
            )
            print(
                f"EPS_END ieps={ieps} merit={state.merit:.6e} {residual_summary(state)} "
                f"{damping_summary(args, mu_shift)} {metrics_summary(metrics)} "
                f"status={eps_status}{(' reason=' + eps_stop_reason) if eps_stop_reason else ''}"
            )
            if args.plot_newton and frame_every is not None and frame_every > 0:
                plotter.emit(
                    [state.u, state.rho, design.rho_design],
                    [f"Epsilon end phi ieps={ieps}", "rho=f(phi)", "rhoDesign"],
                    stage="EPS_END",
                    ieps=ieps,
                    k=-1,
                    eps_phi=eps_phi,
                    residual=state.residual_hdg,
                    metrics=metrics,
                    token=f"ieps_{ieps}_eps_end",
                    save=bool(args.save_frames),
                    show=False,
                )

        metrics = compute_metrics(state, design, c1_phi=c1_phi, c2_phi=c2_phi, eps_phi=eps_phi, params=params)
        write_newton_row(
            newton_writer,
            record="FINAL",
            runTag=run_tag,
            ieps="NA",
            epsPhiRatio="NA",
            epsPhi=eps_phi,
            k="NA",
            nt=space.mesh.num_tri,
            ndof=space.ndof,
            **residual_row(state),
            stepHdg="NA",
            stepFluxL2="NA",
            stepJumpL2="NA",
            alpha="NA",
            bt="NA",
            muEff=active_mu(args, mu_shift),
            minU=metrics["min_u"],
            maxU=metrics["max_u"],
            minRho=metrics["min_rho"],
            maxRho=metrics["max_rho"],
            massRho=metrics["mass_rho"],
            rhoL2=metrics["rho_l2"],
            energyPhi=metrics["energy_phi"],
            activeArea=metrics["active_area"],
            plateauArea=metrics["plateau_area"],
            plateauFrac=metrics["plateau_frac"],
            relRhoDesign=metrics["rel_rho_design"],
            annularPhiMinusC2=metrics["annular_phi_minus_c2"],
            solveTime="NA",
            metricTime="NA",
            stepTime="NA",
            solver="NA",
            status="FINAL",
        )
        if args.plot_final:
            plotter.emit(
                [design.torsion, design.rho_design, design.phi_design, state.u, state.rho],
                ["Torsion T", "rhoDesign", "phiDesign", "Final phi", "Final rho=f(phi)"],
                stage="FINAL",
                ieps=len(params.eps_phi_ratios),
                k=-1,
                eps_phi=eps_phi,
                residual=state.residual_hdg,
                metrics=metrics,
                token="final",
                save=bool(args.frame_final),
                show=True,
            )

    elapsed = time.perf_counter() - total_start
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"runTag {run_tag}\n")
        handle.write(f"initialMesh {initial_mesh_path}\n")
        handle.write(f"nt {space.mesh.num_tri}\n")
        handle.write(f"ndof {space.ndof}\n")
        handle.write(f"order {args.order}\n")
        handle.write(f"alphaT1 {params.alpha_t1}\n")
        handle.write(f"alphaT2 {params.alpha_t2}\n")
        handle.write(f"c1T {design.c1_t}\n")
        handle.write(f"c2T {design.c2_t}\n")
        handle.write(f"epsTRatio {params.eps_t_ratio}\n")
        handle.write(f"epsT {design.eps_t}\n")
        handle.write(f"betaPhi1 {params.beta_phi1}\n")
        handle.write(f"betaPhi2 {params.beta_phi2}\n")
        handle.write(f"c1Phi {c1_phi}\n")
        handle.write(f"c2Phi {c2_phi}\n")
        handle.write(f"rhoExtremaResolution {params.rho_extrema_resolution}\n")
        handle.write(f"rhoExtremaChunkElements {params.rho_extrema_chunk_elements}\n")
        handle.write("nonlinearInitialState phi_design_from_torsion_band\n")
        handle.write(f"newtonCorrectionInitialGuess {args.newton_initial_guess}\n")
        handle.write(f"hdgPetscPreset {args.hdg_petsc_preset}\n")
        handle.write(f"resHdg {state.residual_hdg}\n")
        handle.write(f"resCoeffL2 {state.residual_coeff_l2}\n")
        handle.write(f"resVolumeL2 {state.residual_volume_coeff_l2}\n")
        handle.write(f"resPrimalL2 {state.residual_primal_coeff_l2}\n")
        handle.write(f"resFluxL2 {state.residual_flux_coeff_l2}\n")
        handle.write(f"resTraceL2 {state.residual_trace_coeff_l2}\n")
        handle.write(f"residualNorm {args.residual_norm}\n")
        handle.write(f"newtonShiftMode {args.newton_shift_mode}\n")
        handle.write(f"merit {state.merit}\n")
        handle.write(f"minRho {metrics['min_rho']}\n")
        handle.write(f"maxRho {metrics['max_rho']}\n")
        handle.write(f"minPhi {metrics['min_u']}\n")
        handle.write(f"maxPhi {metrics['max_u']}\n")
        handle.write(f"massRho {metrics['mass_rho']}\n")
        handle.write(f"rhoL2 {metrics['rho_l2']}\n")
        handle.write(f"energyPhi {metrics['energy_phi']}\n")
        handle.write(f"activeArea {metrics['active_area']}\n")
        handle.write(f"plateauArea {metrics['plateau_area']}\n")
        handle.write(f"plateauFrac {metrics['plateau_frac']}\n")
        handle.write(f"relRhoDesign {metrics['rel_rho_design']}\n")
        handle.write(f"annularPhiMinusC2 {metrics['annular_phi_minus_c2']}\n")
        handle.write(f"finalStatus {run_status}\n")
        handle.write(f"stopReasons {';'.join(stop_reasons)}\n")
        handle.write(f"timeTotal {elapsed}\n")
    print(
        f"FINAL merit={state.merit:.6e} {residual_summary(state)} "
        f"{damping_summary(args, mu_shift)} {metrics_summary(metrics)}"
    )
    print(f"FINAL_STATUS {run_status}{(' reasons=' + ';'.join(stop_reasons)) if stop_reasons else ''}")
    print(f"SUMMARY {summary_path}")
    print(f"TIME_TOTAL {elapsed:.3f}")
    print("========== END STRATEGY A HDG TORSION NEWTON ==========")
    return state


def main() -> State:
    return run_strategy(parse_args())


if __name__ == "__main__":
    main()
