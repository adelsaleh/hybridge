#!/usr/bin/env python3
"""Closed-loop torsion-initialized window refit with Newton state projection.

This script implements the reduced closed-loop algorithm only:

1. Polish the semilinear state with Newton for the current thresholds
   ``c1,c2``.
2. Measure the actual torsion-density error
   ``||rho(phi; c1,c2) - rho_T||_L2`` on the converged state.
3. If the torsion error is still too large, use a local contraction/expansion
   map to define a pushed active-band target, then fit a new ``c1,c2`` pair on
   the current unpushed Newton state against that target and the torsion band.
4. Repeat from Newton projection for the new thresholds.

The algorithm declares success only after Newton projection, never on the
cheap pushed fit alone.  The pushed fit is a threshold proposal mechanism, and
only nontrivial threshold updates are accepted into the outer loop.

The threshold refit is deliberately local.  In center-width variables,

    m = (c1 + c2) / 2,    w = c2 - c1,

it searches only over a trust region

    m_new = m + shift,        |shift| <= center_fraction * w,
    w_new = scale * w,        scale in [1-width_fraction, 1+width_fraction].

The spatial push is ray-wise and boundary-aware.  For the generated smooth-star
domain it uses the exact boundary ray

    R(theta) = star_r0 + star_amp cos(star_mode theta),

so the boundary intersection of ``x = x_center + r e(theta)`` is
``x_boundary = x_center + R(theta)e(theta)``.  With
``d_boundary = R(theta)-r``, the intended forward band motion is

    r_new = r_old + beta d_boundary.

The moved scalar field is evaluated by pullback, so the code samples the old
``phi`` at the inverse source radius

    r_old = (r_new - beta R(theta)) / (1 - beta).

Positive ``beta`` expands the displayed band toward the boundary; negative
``beta`` contracts it toward the origin.  The expensive Newton solve is still
used as a state projector, while the threshold update itself is only a small
boundary-aware push/threshold fit on quadrature samples.
"""

from __future__ import annotations

# Support direct execution as well as python -m from a checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[5]))

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import ufl
import basix
from mpi4py import MPI

from dolfinx import fem, geometry, plot as dolfinx_plot

REPO_ROOT = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from projects.diocotron.dolfinx.torsion.equilibrium.window_fit import (  # noqa: E402
    PyVistaTorsionPlotter as BasePyVistaTorsionPlotter,
    assemble_scalar,
    boundary_bc,
    compute_metrics,
    fit_phi_window_to_torsion_design,
    global_minmax,
    load_or_generate_mesh,
    root_print,
    slug_for_path,
    solve_linear_form,
    update_interpolated,
    window_derivative_ufl,
    window_numpy,
    window_ufl,
)


DEFAULT_RUN_LOG_ROOT = REPO_ROOT / "projects/diocotron/runs" / "dolfinx_torsion_initialized_closed_loop_refit"


@dataclass
class TorsionParameters:
    """Parameters defining the fixed torsion target."""

    alpha_t1: float = 0.60
    alpha_t2: float = 0.70
    eps_t_ratio: float = 0.06
    rho_amp: float = 1.0
    active_threshold: float = 0.05
    plateau_threshold: float = 0.90


@dataclass
class NewtonPolishResult:
    """Outcome of one fixed-threshold semilinear Newton projection.

    The closed-loop refit algorithm treats threshold fitting as a cheap
    proposal step.  Each proposal is only meaningful after projection onto the
    fixed-window semilinear branch, and this record captures the projection
    status used by the outer-loop acceptance logic and CSV diagnostics.
    """

    status: str
    iterations: int
    residual_euclid: float
    step_h1: float
    alpha: float
    backtracks: int


def project_center_width(
        m: float,
        w: float,
        *,
        c_max: float,
        min_width: float,
) -> tuple[float, float]:
    """Project center-width coordinates onto admissible thresholds.

    Thresholds are represented locally by center ``m=(c1+c2)/2`` and width
    ``w=c2-c1`` because the refit trust region naturally separates band motion
    from band thickening/thinning.  This helper enforces ``0 <= c1 < c2 <=
    c_max`` and ``w >= min_width`` in those coordinates.
    """
    c_max = max(float(c_max), float(min_width))
    w = min(max(float(w), float(min_width)), c_max)
    lo = 0.5 * w
    hi = c_max - 0.5 * w
    if hi < lo:
        w = c_max
        lo = hi = 0.5 * c_max
    m = min(max(float(m), lo), hi)
    return m, w


def center_width_to_c(m: float, w: float) -> tuple[float, float]:
    """Convert center-width variables back to ordered thresholds."""
    return float(m) - 0.5 * float(w), float(m) + 0.5 * float(w)


def _flush_optional(handle) -> None:
    """Flush a CSV file handle when one is available."""
    if handle is not None:
        handle.flush()


def run_newton_polish(
        *,
        u: fem.Function,
        du: fem.Function,
        rho: fem.Function,
        phi_target: fem.Function,
        rho_design: fem.Function,
        rho_design_l2: float,
        z,
        v,
        dx,
        bc,
        c1: float,
        c2: float,
        eps_phi: float,
        rho_amp: float,
        active_threshold: float,
        plateau_threshold: float,
        start: str,
        max_it: int,
        tol_res: float,
        tol_step: float,
        armijo_c: float,
        beta_ls: float,
        alpha_min: float,
        max_backtrack: int,
        mu_shift: float,
        linear_solver: str,
        ksp_type: str | None,
        linear_rtol: float,
        linear_atol: float,
        linear_max_it: int | None,
        verbosity: int,
        terminal_every: int,
        writer: csv.DictWriter | None,
        handle,
        comm: MPI.Comm,
        nt: int,
        ndof: int,
) -> tuple[NewtonPolishResult, dict[str, float]]:
    """Project ``u`` onto the fixed-window semilinear solution branch.

    The solve holds ``c1``, ``c2``, and ``eps_phi`` fixed and applies a damped
    Newton method to ``-Delta u = W(u;c1,c2,eps_phi)``.  It updates ``rho`` to
    the corresponding windowed density after every accepted trial, reports the
    same geometric diagnostics as the fixed-mesh DOLFINx runner, and returns
    both the final Newton status and the final metrics dictionary.  The
    ``start`` argument controls whether the current state is reused
    (``"penalty"``), reset to the Poisson target (``"target"``), or reset to
    zero (``"zero"``).
    """
    if start == "target":
        u.x.array[:] = phi_target.x.array
        u.x.scatter_forward()
    elif start == "zero":
        u.x.array[:] = 0.0
        u.x.scatter_forward()
    elif start != "penalty":
        raise ValueError(f"unknown Newton polish start {start!r}")

    final_metrics: dict[str, float] | None = None
    last_step_h1 = math.inf
    last_alpha = 0.0
    last_bt = 0
    status = "MAX_IT"

    for k in range(int(max_it) + 1):
        step_start = time.perf_counter()
        update_interpolated(rho, window_ufl(u, c1, c2, eps_phi, rho_amp))
        residual_expr = (
            ufl.inner(ufl.grad(u), ufl.grad(v))
            - window_ufl(u, c1, c2, eps_phi, rho_amp) * v
        ) * dx
        metric_start = time.perf_counter()
        old_metrics = compute_metrics(
            u=u,
            rho=rho,
            rho_design=rho_design,
            rho_design_l2=rho_design_l2,
            residual_form=residual_expr,
            bc=bc,
            dx=dx,
            c2_phi=c2,
            active_threshold=active_threshold,
            plateau_threshold=plateau_threshold,
            rho_amp=rho_amp,
        )
        metric_time = time.perf_counter() - metric_start
        final_metrics = old_metrics
        res_old = float(old_metrics["resEuclid"])

        if res_old < float(tol_res):
            status = "CONVERGED_RESIDUAL"
            if writer is not None:
                writer.writerow({
                    "record": "NEWTON_POLISH",
                    "k": k,
                    "nt": nt,
                    "ndof": ndof,
                    "resEuclid": res_old,
                    "stepH1": "",
                    "alpha": 0.0,
                    "bt": 0,
                    "muShift": mu_shift,
                    "solveTime": 0.0,
                    "metricTime": metric_time,
                    "stepTime": time.perf_counter() - step_start,
                    "linearIterations": "",
                    "linearResidual": "",
                    "maxPhi": old_metrics["maxU"],
                    "maxRho": old_metrics["maxRho"],
                    "massRho": old_metrics["massRho"],
                    "activeJaccard": old_metrics["activeJaccard"],
                    "plateauJaccard": old_metrics["plateauJaccard"],
                    "relRhoDesign": old_metrics["relRhoDesign"],
                    "status": status,
                })
                _flush_optional(handle)
            root_print(comm, f"NEWTON_POLISH k={k} resE={res_old:.6e} status={status}")
            return NewtonPolishResult(status, k, res_old, last_step_h1, last_alpha, last_bt), old_metrics

        if k == int(max_it):
            break

        jac_expr = (
            (1.0 + float(mu_shift)) * ufl.inner(ufl.grad(z), ufl.grad(v))
            - window_derivative_ufl(u, c1, c2, eps_phi, rho_amp) * z * v
        ) * dx
        solve_start = time.perf_counter()
        its, lin_res, solve_time = solve_linear_form(
            jac_expr,
            -residual_expr,
            du,
            [bc],
            prefix=f"newton_polish_{k}_",
            solver=linear_solver,
            ksp_type=ksp_type,
            rtol=linear_rtol,
            atol=linear_atol,
            max_it=linear_max_it,
            verbosity=verbosity,
        )
        solve_time = time.perf_counter() - solve_start
        step_h1 = math.sqrt(max(assemble_scalar(comm, ufl.inner(ufl.grad(du), ufl.grad(du)) * dx), 0.0))
        last_step_h1 = step_h1

        if step_h1 < float(tol_step):
            status = "CONVERGED_STEP"
            if writer is not None:
                writer.writerow({
                    "record": "NEWTON_POLISH",
                    "k": k,
                    "nt": nt,
                    "ndof": ndof,
                    "resEuclid": res_old,
                    "stepH1": step_h1,
                    "alpha": 0.0,
                    "bt": 0,
                    "muShift": mu_shift,
                    "solveTime": solve_time,
                    "metricTime": metric_time,
                    "stepTime": time.perf_counter() - step_start,
                    "linearIterations": its,
                    "linearResidual": lin_res,
                    "maxPhi": old_metrics["maxU"],
                    "maxRho": old_metrics["maxRho"],
                    "massRho": old_metrics["massRho"],
                    "activeJaccard": old_metrics["activeJaccard"],
                    "plateauJaccard": old_metrics["plateauJaccard"],
                    "relRhoDesign": old_metrics["relRhoDesign"],
                    "status": status,
                })
                _flush_optional(handle)
            root_print(comm, f"NEWTON_POLISH k={k} resE={res_old:.6e} stepH1={step_h1:.6e} status={status}")
            return NewtonPolishResult(status, k, res_old, step_h1, 0.0, 0), old_metrics

        u_old = u.x.array.copy()
        alpha = 1.0
        bt = 0
        accepted = False
        trial_metrics = old_metrics
        trial_metric_time = metric_time
        while alpha >= float(alpha_min) and bt <= int(max_backtrack):
            u.x.array[:] = u_old + alpha * du.x.array
            u.x.scatter_forward()
            update_interpolated(rho, window_ufl(u, c1, c2, eps_phi, rho_amp))
            trial_residual_expr = (
                ufl.inner(ufl.grad(u), ufl.grad(v))
                - window_ufl(u, c1, c2, eps_phi, rho_amp) * v
            ) * dx
            trial_metric_start = time.perf_counter()
            trial_metrics = compute_metrics(
                u=u,
                rho=rho,
                rho_design=rho_design,
                rho_design_l2=rho_design_l2,
                residual_form=trial_residual_expr,
                bc=bc,
                dx=dx,
                c2_phi=c2,
                active_threshold=active_threshold,
                plateau_threshold=plateau_threshold,
                rho_amp=rho_amp,
            )
            trial_metric_time = time.perf_counter() - trial_metric_start
            res_trial = float(trial_metrics["resEuclid"])
            armijo = math.isfinite(res_trial) and res_trial <= (1.0 - float(armijo_c) * alpha) * res_old
            if verbosity >= 2:
                root_print(
                    comm,
                    f"NEWTON_POLISH_LS k={k} alpha={alpha:.6e} resTrial={res_trial:.6e} "
                    f"resOld={res_old:.6e} armijo={armijo}",
                )
            if armijo:
                accepted = True
                break
            alpha *= float(beta_ls)
            bt += 1

        if not accepted:
            u.x.array[:] = u_old
            u.x.scatter_forward()
            status = "FAIL_LS"
            final_metrics = old_metrics
            root_print(comm, f"NEWTON_POLISH_STOP reason=FAIL_LS k={k} alpha={alpha:.3e} bt={bt}")
            if writer is not None:
                writer.writerow({
                    "record": "NEWTON_POLISH",
                    "k": k,
                    "nt": nt,
                    "ndof": ndof,
                    "resEuclid": res_old,
                    "stepH1": step_h1,
                    "alpha": alpha,
                    "bt": bt,
                    "muShift": mu_shift,
                    "solveTime": solve_time,
                    "metricTime": metric_time,
                    "stepTime": time.perf_counter() - step_start,
                    "linearIterations": its,
                    "linearResidual": lin_res,
                    "maxPhi": old_metrics["maxU"],
                    "maxRho": old_metrics["maxRho"],
                    "massRho": old_metrics["massRho"],
                    "activeJaccard": old_metrics["activeJaccard"],
                    "plateauJaccard": old_metrics["plateauJaccard"],
                    "relRhoDesign": old_metrics["relRhoDesign"],
                    "status": status,
                })
                _flush_optional(handle)
            return NewtonPolishResult(status, k, res_old, step_h1, alpha, bt), old_metrics

        final_metrics = trial_metrics
        last_alpha = alpha
        last_bt = bt
        step_time = time.perf_counter() - step_start
        if writer is not None:
            writer.writerow({
                "record": "NEWTON_POLISH",
                "k": k,
                "nt": nt,
                "ndof": ndof,
                "resEuclid": trial_metrics["resEuclid"],
                "stepH1": step_h1,
                "alpha": alpha,
                "bt": bt,
                "muShift": mu_shift,
                "solveTime": solve_time,
                "metricTime": trial_metric_time,
                "stepTime": step_time,
                "linearIterations": its,
                "linearResidual": lin_res,
                "maxPhi": trial_metrics["maxU"],
                "maxRho": trial_metrics["maxRho"],
                "massRho": trial_metrics["massRho"],
                "activeJaccard": trial_metrics["activeJaccard"],
                "plateauJaccard": trial_metrics["plateauJaccard"],
                "relRhoDesign": trial_metrics["relRhoDesign"],
                "status": "ACCEPT",
            })
            _flush_optional(handle)
        if terminal_every > 0 and k % int(terminal_every) == 0:
            root_print(
                comm,
                f"NEWTON_POLISH k={k} resE={trial_metrics['resEuclid']:.6e} "
                f"alpha={alpha:.3e} bt={bt} stepH1={step_h1:.6e} "
                f"maxPhi={trial_metrics['maxU']:.6e} maxRho={trial_metrics['maxRho']:.6e} "
                f"activeJ={trial_metrics['activeJaccard']:.6e}",
            )

    assert final_metrics is not None
    return NewtonPolishResult(
        status,
        int(max_it),
        float(final_metrics["resEuclid"]),
        last_step_h1,
        last_alpha,
        last_bt,
    ), final_metrics


@dataclass
class LocalRefitResult:
    """Best local threshold update from the push/threshold search."""

    c1: float
    c2: float
    center: float
    width: float
    shift: float
    scale: float
    push_scale: float
    objective_l2: float
    objective_rel: float
    accepted: bool
    changed_fraction: float = 0.0
    fit_score: float = 0.0
    band_jaccard: float = 0.0
    band_recall: float = 0.0
    band_precision: float = 0.0
    band_dice: float = 0.0
    band_miss_fraction: float = 0.0
    band_spill_fraction: float = 0.0
    band_area_ratio: float = 0.0
    band_area_mismatch: float = 0.0
    radial_eta: float = 0.0
    radial_eta_error: float = 0.0
    radial_eta_step: float = 0.0
    width_growth_penalty: float = 0.0
    anchor_penalty: float = 0.0


@dataclass
class LocalRefitScore:
    """Dimensionless score used to rank cheap pushed refit candidates."""

    objective: float
    density_sq: float
    density_l2: float
    density_rel: float
    band_loss: float
    band_jaccard: float
    band_recall: float
    band_precision: float
    band_dice: float
    band_miss_fraction: float
    band_spill_fraction: float
    band_area_ratio: float
    band_area_mismatch: float
    width_growth_penalty: float
    width_change_penalty: float
    anchor_penalty: float


@dataclass
class BoundaryRadiusTable:
    """Periodic ray-length table for a star-shaped mesh around a center."""

    theta: np.ndarray
    radius: np.ndarray


@dataclass
class SmoothStarBoundary:
    """Exact smooth-star ray geometry used by ``gmsh_smooth_star_mesh``."""

    radius: float
    amplitude: float
    mode: int
    rotation: float = 0.0


class InPlacePyVistaTorsionPlotter(BasePyVistaTorsionPlotter):
    """Use one live PyVista window for nonblocking DOLFINx iterate plots.

    Saved frames and blocking plots still use the base one-shot renderer.  For
    ``--plot-mode nonblocking`` this class keeps the current plotter, VTK grids,
    mesh actors, and scalar actors alive, then only updates point scalar arrays
    and panel text on each emit.  That is the PyVista/DOLFINx analogue of the
    reusable plotting style in ``hdgfem.io.plot``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._live_key: tuple[int, ...] | None = None
        self._live_grids: list | None = None
        self._live_scalar_names: list[str] = []
        self._live_actors: list | None = None
        self._live_text_actors: list = []

    @staticmethod
    def _function_values(function: fem.Function, expected_size: int) -> np.ndarray:
        values = np.asarray(function.x.array, dtype=np.float64)
        if values.size != expected_size:
            raise ValueError(
                f"function values have length {values.size}, but VTK grid has {expected_size} points"
            )
        return values

    @staticmethod
    def _new_grid(function: fem.Function, *, scalar_name: str):
        import pyvista as pv

        topology, cell_types, geometry = dolfinx_plot.vtk_mesh(function.function_space)
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        grid.point_data[scalar_name] = np.asarray(function.x.array, dtype=np.float64).copy()
        return grid

    def _live_layout_key(self, fields: list[fem.Function]) -> tuple[int, ...]:
        return tuple(int(field.x.array.size) for field in fields)

    def _reset_live_cache(self) -> None:
        self._close_live_plotter()
        self._live_key = None
        self._live_grids = None
        self._live_scalar_names = []
        self._live_actors = None
        self._live_text_actors = []

    def _update_text_actor(self, plotter, index: int, title: str, nt: int, ndof: int):
        text = f"{title}\nnt={nt} ndof={ndof}"
        if index < len(self._live_text_actors):
            actor = self._live_text_actors[index]
            try:
                plotter.remove_actor(actor)
            except Exception:
                pass
            self._live_text_actors[index] = plotter.add_text(
                text,
                position="upper_edge",
                font_size=11,
                shadow=False,
            )
        else:
            self._live_text_actors.append(plotter.add_text(
                text,
                position="upper_edge",
                font_size=11,
                shadow=False,
            ))

    def _build_live_plotter(
            self,
            fields: list[fem.Function],
            titles: list[str],
            *,
            window_size: tuple[int, int],
            nt: int,
            ndof: int,
    ) -> None:
        import pyvista as pv

        self._reset_live_cache()
        plotter = pv.Plotter(
            shape=(1, len(fields)),
            window_size=list(window_size),
            off_screen=False,
        )
        self._live_grids = []
        self._live_actors = []
        self._live_scalar_names = []
        self._live_text_actors = []
        for index, (field, title) in enumerate(zip(fields, titles)):
            plotter.subplot(0, index)
            scalar_name = f"panel_{index}"
            grid = self._new_grid(field, scalar_name=scalar_name)
            values = grid.point_data[scalar_name]
            actor = plotter.add_mesh(
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
            self._update_text_actor(plotter, index, title, nt, ndof)
            plotter.enable_parallel_projection()
            plotter.view_xy()
            plotter.show_grid(color=(100, 100, 100, 0.15))
            self._live_grids.append(grid)
            self._live_actors.append(actor)
            self._live_scalar_names.append(scalar_name)
        if len(fields) > 1:
            plotter.link_views()
        plotter.show(interactive_update=True, auto_close=False)
        self._live_plotter = plotter
        self._live_key = self._live_layout_key(fields)

    def _update_live_plotter(
            self,
            fields: list[fem.Function],
            titles: list[str],
            *,
            window_size: tuple[int, int],
            nt: int,
            ndof: int,
    ) -> None:
        key = self._live_layout_key(fields)
        if (
                self._live_plotter is None
                or self._live_grids is None
                or self._live_actors is None
                or self._live_key != key
        ):
            self._build_live_plotter(fields, titles, window_size=window_size, nt=nt, ndof=ndof)
            return

        plotter = self._live_plotter
        for index, (field, title) in enumerate(zip(fields, titles)):
            plotter.subplot(0, index)
            grid = self._live_grids[index]
            scalar_name = self._live_scalar_names[index]
            values = self._function_values(field, grid.n_points)
            grid.point_data[scalar_name][:] = values
            grid.Modified()
            actor = self._live_actors[index]
            clim = self._safe_clim(values)
            try:
                actor.mapper.scalar_range = clim
            except Exception:
                try:
                    actor.mapper.SetScalarRange(*clim)
                except Exception:
                    pass
            self._update_text_actor(plotter, index, title, nt, ndof)
        plotter.update()

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
        plot_mode = getattr(self.args, "plot_mode", "blocking")
        use_live_update = (
            save_path is None
            and show
            and self.comm.rank == 0
            and not self.args.plot_off_screen
            and plot_mode == "nonblocking"
        )
        if not use_live_update:
            if show and plot_mode != "nonblocking":
                self._reset_live_cache()
            super()._plot(
                fields,
                titles,
                save_path=save_path,
                show=show,
                window_size=window_size,
                nt=nt,
                ndof=ndof,
            )
            return
        if not fields:
            return
        try:
            self._update_live_plotter(fields, titles, window_size=window_size, nt=nt, ndof=ndof)
        except Exception as exc:
            self._reset_live_cache()
            print(f"PLOT_SKIP stage={titles[0] if titles else 'unknown'} error={type(exc).__name__}: {exc}", flush=True)


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


def params_from_args(args: argparse.Namespace) -> TorsionParameters:
    params = TorsionParameters()
    if args.alpha_t1 is not None:
        params.alpha_t1 = float(args.alpha_t1)
    if args.alpha_t2 is not None:
        params.alpha_t2 = float(args.alpha_t2)
    if args.eps_t_ratio is not None:
        params.eps_t_ratio = float(args.eps_t_ratio)
    if args.rho_amp is not None:
        params.rho_amp = float(args.rho_amp)
    if not params.alpha_t2 > params.alpha_t1:
        raise ValueError("require alphaT2 > alphaT1")
    if params.eps_t_ratio <= 0.0:
        raise ValueError("require positive eps_t_ratio")
    return params


def fit_objective_global(
        comm: MPI.Comm,
        phi_values: np.ndarray,
        rho_values: np.ndarray,
        weights: np.ndarray,
        *,
        c1: float,
        c2: float,
        eps_ratio: float,
        rho_amp: float,
) -> float:
    """Evaluate the exact quadrature L2 density mismatch for one window."""
    if c2 <= c1:
        return math.inf
    eps = float(eps_ratio) * (float(c2) - float(c1))
    predicted = window_numpy(phi_values, c1, c2, eps, rho_amp)
    local = float(np.dot(weights, (predicted - rho_values) ** 2))
    return float(comm.allreduce(local, op=MPI.SUM))


def refit_band_mask_numpy(
        phi_values: np.ndarray,
        *,
        c1: float,
        c2: float,
        eps_ratio: float,
        rho_amp: float,
        band_threshold: float,
        current_band_mode: str,
) -> np.ndarray:
    """Return the candidate active/band mask used by the cheap refit."""
    if current_band_mode == "active-density":
        eps = float(eps_ratio) * (float(c2) - float(c1))
        predicted = window_numpy(phi_values, c1, c2, eps, rho_amp)
        return predicted > float(band_threshold) * float(rho_amp)
    return (phi_values >= float(c1)) & (phi_values <= float(c2))


def design_band_mask_numpy(rho_values: np.ndarray, *, rho_amp: float, band_threshold: float) -> np.ndarray:
    """Return the torsion active-region mask used by the cheap refit."""
    return np.asarray(rho_values, dtype=np.float64) > float(band_threshold) * float(rho_amp)


def radial_mask_centroid_global(
        comm: MPI.Comm,
        points_xy: np.ndarray,
        weights: np.ndarray,
        mask: np.ndarray,
        *,
        push_center: np.ndarray,
        radius_table: BoundaryRadiusTable | SmoothStarBoundary,
) -> tuple[float, float]:
    """Return weighted mean normalized radius and area for one mask."""
    if points_xy.size == 0:
        local = np.zeros(2, dtype=np.float64)
    else:
        vec = np.asarray(points_xy, dtype=np.float64) - push_center[None, :]
        r = np.linalg.norm(vec, axis=1)
        theta = np.arctan2(vec[:, 1], vec[:, 0])
        R = np.maximum(ray_boundary_radius(radius_table, theta), 1.0e-300)
        eta = np.clip(r / R, 0.0, 1.0)
        mask_f = np.asarray(mask, dtype=np.float64)
        local = np.array([
            float(np.dot(weights, mask_f * eta)),
            float(np.dot(weights, mask_f)),
        ], dtype=np.float64)
    global_values = np.empty_like(local)
    comm.Allreduce(local, global_values, op=MPI.SUM)
    area = float(global_values[1])
    return float(global_values[0] / max(area, 1.0e-300)), area


def auto_push_direction(
        *,
        current_eta: float,
        design_eta: float,
        current_area: float,
        design_area: float,
        tol: float,
) -> str:
    """Choose a geometric direction from radial drift of active regions."""
    if current_area <= 1.0e-300 or design_area <= 1.0e-300:
        return "both"
    if current_eta < design_eta - float(tol):
        return "expand"
    if current_eta > design_eta + float(tol):
        return "contract"
    return "both"


def push_scales_for_direction(*, push_radius: float, push_scale_grid: int, direction: str) -> np.ndarray:
    """Build beta candidates for a requested visible band motion direction."""
    radius = min(max(float(push_radius), 0.0), 0.95)
    grid = max(1, int(push_scale_grid))
    if grid == 1 or radius == 0.0:
        return np.array([0.0], dtype=np.float64)
    if direction == "expand":
        return np.linspace(0.0, radius, grid, dtype=np.float64)
    if direction == "contract":
        return np.linspace(-radius, 0.0, grid, dtype=np.float64)
    return np.linspace(-radius, radius, grid, dtype=np.float64)


def refit_score_global(
        comm: MPI.Comm,
        phi_values: np.ndarray,
        rho_values: np.ndarray,
        weights: np.ndarray,
        *,
        c1: float,
        c2: float,
        eps_ratio: float,
        rho_amp: float,
        rho_design_l2: float,
        objective_mode: str,
        current_band_mode: str,
        band_threshold: float,
        band_miss_weight: float,
        band_spill_weight: float,
        band_jaccard_weight: float,
        band_area_weight: float,
        target_area_ratio: float,
        density_weight: float,
        width_growth_weight: float,
        width_change_weight: float,
        anchor_mask: np.ndarray | None = None,
        anchor_weight: float = 0.0,
        old_width: float,
) -> LocalRefitScore:
    """Score one pushed-window candidate by band containment/matching.

    The density L2 term is kept as a diagnostic and optional tie-breaker.  The
    default ranking is based on the thresholded band geometry: recall measures
    how much of the torsion band is covered, while precision measures how much
    of the semilinear band is contained in the torsion band.
    """
    width = float(c2) - float(c1)
    if width <= 0.0:
        return LocalRefitScore(
            objective=math.inf,
            density_sq=math.inf,
            density_l2=math.inf,
            density_rel=math.inf,
            band_loss=math.inf,
            band_jaccard=0.0,
            band_recall=0.0,
            band_precision=0.0,
            band_dice=0.0,
            band_miss_fraction=1.0,
            band_spill_fraction=1.0,
            band_area_ratio=0.0,
            band_area_mismatch=1.0,
            width_growth_penalty=math.inf,
            width_change_penalty=math.inf,
            anchor_penalty=math.inf,
        )

    eps = float(eps_ratio) * width
    predicted = window_numpy(phi_values, c1, c2, eps, rho_amp)
    density_local = float(np.dot(weights, (predicted - rho_values) ** 2))

    design_mask = design_band_mask_numpy(rho_values, rho_amp=rho_amp, band_threshold=band_threshold)
    if current_band_mode == "active-density":
        current_mask = predicted > float(band_threshold) * float(rho_amp)
    else:
        current_mask = (phi_values >= float(c1)) & (phi_values <= float(c2))

    current_mask_f = current_mask.astype(np.float64)
    design_mask_f = design_mask.astype(np.float64)
    overlap_mask_f = (current_mask & design_mask).astype(np.float64)
    if anchor_mask is None or float(anchor_weight) == 0.0:
        anchor_area_local = 0.0
        anchor_diff_local = 0.0
    else:
        anchor_bool = np.asarray(anchor_mask, dtype=bool)
        anchor_area_local = float(np.dot(weights, anchor_bool.astype(np.float64)))
        anchor_diff_local = float(np.dot(weights, np.logical_xor(current_mask, anchor_bool).astype(np.float64)))

    local_metrics = np.array([
        density_local,
        float(np.dot(weights, current_mask_f)),
        float(np.dot(weights, design_mask_f)),
        float(np.dot(weights, overlap_mask_f)),
        anchor_area_local,
        anchor_diff_local,
    ], dtype=np.float64)
    global_metrics = np.empty_like(local_metrics)
    comm.Allreduce(local_metrics, global_metrics, op=MPI.SUM)

    density_sq = float(global_metrics[0])
    active_area = float(global_metrics[1])
    design_area = float(global_metrics[2])
    overlap_area = float(global_metrics[3])
    anchor_area = float(global_metrics[4])
    anchor_diff_area = float(global_metrics[5])
    union_area = max(design_area + active_area - overlap_area, 1.0e-300)
    sum_area = max(design_area + active_area, 1.0e-300)
    band_jaccard = overlap_area / union_area
    band_recall = overlap_area / max(design_area, 1.0e-300)
    band_precision = overlap_area / max(active_area, 1.0e-300)
    band_dice = 2.0 * overlap_area / sum_area
    band_miss_fraction = (design_area - overlap_area) / max(design_area, 1.0e-300)
    band_spill_fraction = (active_area - overlap_area) / max(active_area, 1.0e-300)
    band_area_ratio = active_area / max(design_area, 1.0e-300)
    target_area = max(float(target_area_ratio), 0.0) * design_area
    band_area_mismatch = abs(active_area - target_area) / max(design_area, 1.0e-300)
    anchor_penalty = float(anchor_weight) * anchor_diff_area / max(anchor_area, design_area, 1.0e-300)
    band_loss = (
        float(band_miss_weight) * band_miss_fraction
        + float(band_spill_weight) * band_spill_fraction
        + float(band_jaccard_weight) * (1.0 - band_jaccard)
        + float(band_area_weight) * band_area_mismatch
    )

    density_l2 = math.sqrt(max(density_sq, 0.0))
    density_rel = density_l2 / max(float(rho_design_l2), 1.0e-300)
    width_scale = width / max(float(old_width), 1.0e-300)
    width_growth = max(width_scale - 1.0, 0.0)
    width_change = abs(width_scale - 1.0)
    width_growth_penalty = float(width_growth_weight) * width_growth * width_growth
    width_change_penalty = float(width_change_weight) * width_change * width_change

    if objective_mode == "density":
        objective = density_rel * density_rel + width_growth_penalty + width_change_penalty
    else:
        effective_density_weight = float(density_weight)
        if objective_mode == "band-containment":
            effective_density_weight = float(density_weight)
        objective = (
            band_loss
            + effective_density_weight * density_rel * density_rel
            + width_growth_penalty
            + width_change_penalty
            + anchor_penalty
        )

    return LocalRefitScore(
        objective=float(objective),
        density_sq=density_sq,
        density_l2=density_l2,
        density_rel=density_rel,
        band_loss=band_loss,
        band_jaccard=band_jaccard,
        band_recall=band_recall,
        band_precision=band_precision,
        band_dice=band_dice,
        band_miss_fraction=band_miss_fraction,
        band_spill_fraction=band_spill_fraction,
        band_area_ratio=band_area_ratio,
        band_area_mismatch=band_area_mismatch,
        width_growth_penalty=width_growth_penalty,
        width_change_penalty=width_change_penalty,
        anchor_penalty=anchor_penalty,
    )


def global_bbox_center(domain) -> np.ndarray:
    """Return the MPI-global center of the mesh geometry bounding box."""
    x = np.asarray(domain.geometry.x[:, :2], dtype=np.float64)
    local_min = np.min(x, axis=0) if x.size else np.array([math.inf, math.inf])
    local_max = np.max(x, axis=0) if x.size else np.array([-math.inf, -math.inf])
    global_min = np.empty(2, dtype=np.float64)
    global_max = np.empty(2, dtype=np.float64)
    domain.comm.Allreduce(local_min, global_min, op=MPI.MIN)
    domain.comm.Allreduce(local_max, global_max, op=MPI.MAX)
    return 0.5 * (global_min + global_max)


def build_boundary_radius_table(domain, center: np.ndarray, *, bins: int) -> BoundaryRadiusTable:
    """Approximate ray-to-boundary distances from mesh geometry nodes.

    The closed-loop push assumes the domain is star-shaped with respect to the
    chosen center.  For each angular bin we take the maximum observed geometry
    radius; this is a cheap mesh-based estimate of the boundary radius on that
    ray.  Empty bins are filled by periodic linear interpolation.
    """
    comm = domain.comm
    bins = max(32, int(bins))
    coords = np.asarray(domain.geometry.x[:, :2], dtype=np.float64)
    local_radius = np.zeros(bins, dtype=np.float64)
    if coords.size:
        vec = coords - center[None, :]
        radii = np.linalg.norm(vec, axis=1)
        theta = np.mod(np.arctan2(vec[:, 1], vec[:, 0]), 2.0 * math.pi)
        indices = np.minimum((theta / (2.0 * math.pi) * bins).astype(np.int64), bins - 1)
        np.maximum.at(local_radius, indices, radii)

    global_radius = np.empty_like(local_radius)
    comm.Allreduce(local_radius, global_radius, op=MPI.MAX)
    bin_theta = (np.arange(bins, dtype=np.float64) + 0.5) * (2.0 * math.pi / bins)

    valid = global_radius > 0.0
    if not np.any(valid):
        raise ValueError("cannot build boundary radius table from an empty mesh geometry")
    if not np.all(valid):
        valid_theta = bin_theta[valid]
        valid_radius = global_radius[valid]
        extended_theta = np.concatenate((valid_theta - 2.0 * math.pi, valid_theta, valid_theta + 2.0 * math.pi))
        extended_radius = np.concatenate((valid_radius, valid_radius, valid_radius))
        global_radius = np.interp(bin_theta, extended_theta, extended_radius)

    # A tiny inflation avoids numerical underestimation of R(theta), which would
    # otherwise clip points that are already on the boundary.
    return BoundaryRadiusTable(theta=bin_theta, radius=1.001 * global_radius)


def smooth_star_boundary_radius(star: SmoothStarBoundary, theta: np.ndarray) -> np.ndarray:
    """Evaluate the exact smooth-star boundary radius at ray angles ``theta``."""
    return (
        float(star.radius)
        + float(star.amplitude) * np.cos(int(star.mode) * (np.asarray(theta, dtype=np.float64) - float(star.rotation)))
    )


def ray_boundary_radius(table: BoundaryRadiusTable | SmoothStarBoundary, theta: np.ndarray) -> np.ndarray:
    """Return boundary ray lengths at angles ``theta``."""
    if isinstance(table, SmoothStarBoundary):
        return smooth_star_boundary_radius(table, theta)

    theta_mod = np.mod(theta, 2.0 * math.pi)
    theta_ext = np.concatenate((
        table.theta[-1:] - 2.0 * math.pi,
        table.theta,
        table.theta[:1] + 2.0 * math.pi,
    ))
    radius_ext = np.concatenate((table.radius[-1:], table.radius, table.radius[:1]))
    return np.interp(theta_mod, theta_ext, radius_ext)


def boundary_aware_push_points(
        points_xy: np.ndarray,
        *,
        push_center: np.ndarray,
        push_beta: float,
        radius_table: BoundaryRadiusTable | SmoothStarBoundary,
) -> np.ndarray:
    """Return pullback sample points for a boundary-aware band motion.

    The intended forward motion of a band point is
    ``r_new = r_old + beta*(R(theta)-r_old)``.  Since the moved scalar field is
    evaluated on fixed mesh coordinates, this helper returns the inverse source
    coordinate ``r_old = (r_new - beta*R(theta))/(1-beta)``.  Thus positive
    ``beta`` expands the displayed band toward the boundary and negative
    ``beta`` contracts it toward the origin.
    """
    if points_xy.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    vec = np.asarray(points_xy, dtype=np.float64) - push_center[None, :]
    r = np.linalg.norm(vec, axis=1)
    theta = np.arctan2(vec[:, 1], vec[:, 0])
    R = np.maximum(ray_boundary_radius(radius_table, theta), 1.0e-300)
    beta = float(np.clip(push_beta, -0.95, 0.95))
    r_clipped = np.clip(r, 0.0, R)
    r_source = np.clip((r_clipped - beta * R) / max(1.0 - beta, 1.0e-300), 0.0, R)
    scale = np.ones_like(r)
    mask = r > 1.0e-300
    scale[mask] = r_source[mask] / r[mask]
    mapped = push_center[None, :] + scale[:, None] * vec
    return np.ascontiguousarray(mapped)


def quadrature_geometry_samples(
        phi: fem.Function,
        rho_design: fem.Function,
        *,
        quadrature_degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample quadrature coordinates, ``phi``, ``rho_design``, and weights.

    The returned coordinates are physical points for this rank's owned cells.
    They are used both directly and after a spatial push/pull map.  Keeping the
    coordinates lets the refit test windows of the form
    ``W(phi(x_center + s*(x-x_center)); c1,c2)`` without assembling a new form
    for every candidate.
    """
    V = phi.function_space
    if rho_design.function_space is not V:
        raise ValueError("phi and rho_design must use the same function space")
    domain = V.mesh
    if domain.topology.dim != 2:
        raise ValueError("closed-loop refit currently expects a 2D triangle mesh")

    q_points, q_weights = basix.make_quadrature(basix.CellType.triangle, int(quadrature_degree))
    basis = V.element.basix_element.tabulate(0, q_points)[0, :, :, 0]
    cell_dofs = np.asarray(V.dofmap.list, dtype=np.int64)
    geometry_dofs = np.asarray(domain.geometry.dofmaps[0], dtype=np.int64)
    x = np.asarray(domain.geometry.x, dtype=np.float64)

    phi_coeffs = np.asarray(phi.x.array, dtype=np.float64)
    rho_coeffs = np.asarray(rho_design.x.array, dtype=np.float64)
    point_chunks: list[np.ndarray] = []
    phi_chunks: list[np.ndarray] = []
    rho_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []

    for start in range(0, cell_dofs.shape[0], 2048):
        stop = min(start + 2048, cell_dofs.shape[0])
        cdofs = cell_dofs[start:stop]
        gdofs = geometry_dofs[start:stop]

        phi_local = phi_coeffs[cdofs] @ basis.T
        rho_local = rho_coeffs[cdofs] @ basis.T

        coords = x[gdofs, :2]
        j0 = coords[:, 1, :] - coords[:, 0, :]
        j1 = coords[:, 2, :] - coords[:, 0, :]
        physical = (
            coords[:, None, 0, :]
            + q_points[None, :, 0, None] * j0[:, None, :]
            + q_points[None, :, 1, None] * j1[:, None, :]
        )
        det_j = np.abs(j0[:, 0] * j1[:, 1] - j0[:, 1] * j1[:, 0])
        weights = det_j[:, None] * q_weights[None, :]

        point_chunks.append(np.reshape(physical, (-1, 2)))
        phi_chunks.append(np.ravel(phi_local))
        rho_chunks.append(np.ravel(rho_local))
        weight_chunks.append(np.ravel(weights))

    if not point_chunks:
        empty = np.empty(0, dtype=np.float64)
        return np.empty((0, 2), dtype=np.float64), empty, empty, empty
    return (
        np.ascontiguousarray(np.concatenate(point_chunks)),
        np.ascontiguousarray(np.concatenate(phi_chunks)),
        np.ascontiguousarray(np.concatenate(rho_chunks)),
        np.ascontiguousarray(np.concatenate(weight_chunks)),
    )


def sample_function_at_points(function: fem.Function, points_xy: np.ndarray) -> np.ndarray:
    """Evaluate a CG function at physical points, using zero outside the mesh.

    The zero outside value is consistent with the homogeneous Dirichlet
    potential used here and prevents a spatial expansion from creating invalid
    samples.  This helper is intended for serial/local diagnostic refits; in a
    distributed run, pushed points that move to another rank are treated as
    outside on the current rank.
    """
    if points_xy.size == 0:
        return np.empty(0, dtype=np.float64)
    domain = function.function_space.mesh
    points = np.zeros((points_xy.shape[0], 3), dtype=np.float64)
    points[:, :2] = points_xy
    tree = geometry.bb_tree(domain, domain.topology.dim)
    candidates = geometry.compute_collisions_points(tree, points)
    colliding = geometry.compute_colliding_cells(domain, candidates, points)
    cells = np.full(points.shape[0], -1, dtype=np.int32)
    valid: list[int] = []
    for i in range(points.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            cells[i] = int(links[0])
            valid.append(i)
    values = np.zeros(points.shape[0], dtype=np.float64)
    if valid:
        valid_idx = np.asarray(valid, dtype=np.int32)
        evaluated = function.eval(points[valid_idx], cells[valid_idx])
        values[valid_idx] = np.asarray(evaluated, dtype=np.float64).reshape(len(valid_idx), -1)[:, 0]
    return values


def update_pushed_function(
        target: fem.Function,
        source: fem.Function,
        *,
        push_center: np.ndarray,
        push_beta: float,
        radius_table: BoundaryRadiusTable | SmoothStarBoundary,
) -> None:
    """Approximate the boundary-aware pushed source field at dofs.

    This is a visualization helper for the refit stage.  The optimization fit
    itself uses quadrature samples, not this dof-sampled function.
    """
    if target.function_space is not source.function_space:
        raise ValueError("target and source must use the same function space")
    dof_coords = np.asarray(source.function_space.tabulate_dof_coordinates(), dtype=np.float64)
    if dof_coords.shape[0] != source.x.array.size:
        raise ValueError(
            "cannot build pushed plot field because dof coordinate count "
            f"{dof_coords.shape[0]} != local array size {source.x.array.size}"
        )
    points_xy = dof_coords[:, :2]
    mapped = boundary_aware_push_points(
        points_xy,
        push_center=push_center,
        push_beta=push_beta,
        radius_table=radius_table,
    )
    target.x.array[:] = sample_function_at_points(source, mapped)
    target.x.scatter_forward()


def update_active_mask(target: fem.Function, source: fem.Function, *, threshold: float) -> None:
    """Store a 0/1 active-band mask for clearer plotting of band motion."""
    target.x.array[:] = (np.asarray(source.x.array, dtype=np.float64) > float(threshold)).astype(np.float64)
    target.x.scatter_forward()


def update_window_band_mask(target: fem.Function, source: fem.Function, *, c1: float, c2: float) -> None:
    """Store the hard scalar-window band ``c1 <= source <= c2``."""
    values = np.asarray(source.x.array, dtype=np.float64)
    target.x.array[:] = ((values >= float(c1)) & (values <= float(c2))).astype(np.float64)
    target.x.scatter_forward()


def update_difference(target: fem.Function, lhs: fem.Function, rhs: fem.Function) -> None:
    """Store ``lhs-rhs`` in ``target`` for plotting signed band changes."""
    target.x.array[:] = np.asarray(lhs.x.array, dtype=np.float64) - np.asarray(rhs.x.array, dtype=np.float64)
    target.x.scatter_forward()


def push_direction_label(push_beta: float) -> str:
    """Name the visible band motion induced by ``push_beta``."""
    beta = float(push_beta)
    if beta > 1.0e-14:
        return "expand"
    if beta < -1.0e-14:
        return "contract"
    return "none"


def torsion_error_reached(*, rho_l2: float, rho_rel: float, tol_l2: float | None, tol_rel: float) -> bool:
    """Return true when the projected state is close enough to torsion density."""
    rel_ok = float(rho_rel) <= float(tol_rel)
    abs_ok = tol_l2 is not None and float(rho_l2) <= float(tol_l2)
    return rel_ok or abs_ok


def local_center_width_refit(
        *,
        phi: fem.Function,
        rho_design: fem.Function,
        rho_design_l2: float,
        m_old: float,
        w_old: float,
        c_max: float,
        min_width: float,
        eps_ratio: float,
        rho_amp: float,
        objective_mode: str,
        current_band_mode: str,
        band_threshold: float,
        band_miss_weight: float,
        band_spill_weight: float,
        band_jaccard_weight: float,
        band_area_weight: float,
        target_area_ratio: float,
        density_weight: float,
        width_growth_weight: float,
        width_change_weight: float,
        anchor_weight: float,
        directional_threshold_guard: bool,
        push_direction: str,
        center_fraction: float,
        width_fraction: float,
        center_grid: int,
        width_grid: int,
        refine_passes: int,
        push_center: np.ndarray,
        radius_table: BoundaryRadiusTable | SmoothStarBoundary,
        push_scale_fraction: float,
        push_scale_grid: int,
        quadrature_degree: int,
        min_threshold_step: float,
        verbosity: int = 0,
        push_callback: Callable[[int, int, int, float, float, float, float], None] | None = None,
) -> LocalRefitResult:
    """Refit thresholds by a local spatial-push and threshold search.

    The refit first chooses a boundary-aware push parameter ``beta`` from the
    active-region radial drift and containment score.  That push is only a
    target generator: after beta is fixed, threshold candidates are scored on
    the current, unpushed Newton state against both the torsion band and the
    pushed active mask.  The returned proposal is therefore an actionable
    ``c1,c2`` update for the next Newton projection, not a pure pushed-field
    diagnostic.
    """
    comm = phi.function_space.mesh.comm
    t_start = time.perf_counter()
    if verbosity >= 1:
        root_print(
            comm,
            "REFIT_START "
            f"centerGrid={int(center_grid)} widthGrid={int(width_grid)} "
            f"pushGrid={int(push_scale_grid)} refinePasses={int(refine_passes)} "
            f"quadDegree={int(quadrature_degree)}",
        )
    points_xy, phi_values_unpushed, rho_values, weights = quadrature_geometry_samples(
        phi,
        rho_design,
        quadrature_degree=int(quadrature_degree),
    )
    center_grid = max(3, int(center_grid))
    width_grid = max(3, int(width_grid))
    refine_passes = max(0, int(refine_passes))
    push_scale_grid = max(1, int(push_scale_grid))
    center_radius = max(float(center_fraction) * float(w_old), 0.0)
    width_radius = max(float(width_fraction), 0.0)
    push_radius = max(float(push_scale_fraction), 0.0)
    local_samples = int(points_xy.shape[0])
    global_samples = int(comm.allreduce(local_samples, op=MPI.SUM))
    if verbosity >= 1:
        root_print(
            comm,
            "REFIT_SAMPLES "
            f"n={global_samples} local={local_samples} "
            f"sampleTime={time.perf_counter() - t_start:.3f}",
        )

    best_m = float(m_old)
    best_w = float(w_old)
    best_shift = 0.0
    best_scale = 1.0
    best_push_scale = 0.0
    c1_old, c2_old = center_width_to_c(m_old, w_old)
    best_score = refit_score_global(
        comm,
        phi_values_unpushed,
        rho_values,
        weights,
        c1=c1_old,
        c2=c2_old,
        eps_ratio=eps_ratio,
        rho_amp=rho_amp,
        rho_design_l2=rho_design_l2,
        objective_mode=objective_mode,
        current_band_mode=current_band_mode,
        band_threshold=band_threshold,
        band_miss_weight=band_miss_weight,
        band_spill_weight=band_spill_weight,
        band_jaccard_weight=band_jaccard_weight,
        band_area_weight=band_area_weight,
        target_area_ratio=target_area_ratio,
        density_weight=density_weight,
        width_growth_weight=width_growth_weight,
        width_change_weight=width_change_weight,
        anchor_mask=None,
        anchor_weight=0.0,
        old_width=w_old,
    )

    cur_center_radius = center_radius
    cur_width_radius = width_radius
    pushed_cache: dict[float, np.ndarray] = {0.0: phi_values_unpushed}

    def pushed_values(push_beta: float) -> np.ndarray:
        key = round(float(push_beta), 14)
        if key not in pushed_cache:
            mapped = boundary_aware_push_points(
                points_xy,
                push_center=push_center,
                push_beta=float(push_beta),
                radius_table=radius_table,
            )
            pushed_cache[key] = sample_function_at_points(phi, mapped)
        return pushed_cache[key]

    design_mask_unpushed = design_band_mask_numpy(rho_values, rho_amp=rho_amp, band_threshold=band_threshold)
    current_mask_unpushed = refit_band_mask_numpy(
        phi_values_unpushed,
        c1=c1_old,
        c2=c2_old,
        eps_ratio=eps_ratio,
        rho_amp=rho_amp,
        band_threshold=band_threshold,
        current_band_mode=current_band_mode,
    )
    current_eta, current_active_area = radial_mask_centroid_global(
        comm,
        points_xy,
        weights,
        current_mask_unpushed,
        push_center=push_center,
        radius_table=radius_table,
    )
    design_eta, design_active_area = radial_mask_centroid_global(
        comm,
        points_xy,
        weights,
        design_mask_unpushed,
        push_center=push_center,
        radius_table=radius_table,
    )
    current_eta_error = abs(current_eta - design_eta)
    detected_direction = auto_push_direction(
        current_eta=current_eta,
        design_eta=design_eta,
        current_area=current_active_area,
        design_area=design_active_area,
        tol=2.0e-3,
    )
    effective_push_direction = detected_direction if push_direction == "auto" else push_direction
    push_scales = push_scales_for_direction(
        push_radius=push_radius,
        push_scale_grid=push_scale_grid,
        direction=effective_push_direction,
    )
    if verbosity >= 1:
        root_print(
            comm,
            "REFIT_DRIFT "
            f"currentEta={current_eta:.6e} targetEta={design_eta:.6e} "
            f"currentArea={current_active_area:.6e} targetArea={design_active_area:.6e} "
            f"detected={detected_direction} search={effective_push_direction} "
            f"pushRange=[{float(push_scales[0]):.6e},{float(push_scales[-1]):.6e}]",
        )

    old_design_score = best_score
    push_select_score = old_design_score
    best_push_score = old_design_score
    for push_index, push_scale in enumerate(push_scales):
        push_t0 = time.perf_counter()
        phi_values = pushed_values(float(push_scale))
        push_score = refit_score_global(
            comm,
            phi_values,
            rho_values,
            weights,
            c1=c1_old,
            c2=c2_old,
            eps_ratio=eps_ratio,
            rho_amp=rho_amp,
            rho_design_l2=rho_design_l2,
            objective_mode=objective_mode,
            current_band_mode=current_band_mode,
            band_threshold=band_threshold,
            band_miss_weight=band_miss_weight,
            band_spill_weight=band_spill_weight,
            band_jaccard_weight=band_jaccard_weight,
            band_area_weight=band_area_weight,
            target_area_ratio=target_area_ratio,
            density_weight=density_weight,
            width_growth_weight=width_growth_weight,
            width_change_weight=width_change_weight,
            anchor_mask=None,
            anchor_weight=0.0,
            old_width=w_old,
        )
        if push_score.objective < push_select_score.objective:
            push_select_score = push_score
            best_push_score = push_score
            best_push_scale = float(push_scale)
        if push_callback is not None:
            push_callback(
                -1,
                push_index,
                len(push_scales),
                float(push_scale),
                best_m,
                best_w,
                push_select_score.objective,
            )
        if verbosity >= 2:
            root_print(
                comm,
                "REFIT_PUSH_SELECT "
                f"index={push_index + 1}/{len(push_scales)} "
                f"pushBeta={float(push_scale):.6e} pushMode={push_direction_label(float(push_scale))} "
                f"sampleTime={time.perf_counter() - push_t0:.3f} "
                f"score={push_score.objective:.6e} recall={push_score.band_recall:.6e} "
                f"precision={push_score.band_precision:.6e} areaRatio={push_score.band_area_ratio:.6e}",
            )

    pushed_phi_for_threshold = pushed_values(best_push_scale)
    anchor_mask = refit_band_mask_numpy(
        pushed_phi_for_threshold,
        c1=c1_old,
        c2=c2_old,
        eps_ratio=eps_ratio,
        rho_amp=rho_amp,
        band_threshold=band_threshold,
        current_band_mode=current_band_mode,
    )
    actionable_baseline_score = refit_score_global(
        comm,
        phi_values_unpushed,
        rho_values,
        weights,
        c1=c1_old,
        c2=c2_old,
        eps_ratio=eps_ratio,
        rho_amp=rho_amp,
        rho_design_l2=rho_design_l2,
        objective_mode=objective_mode,
        current_band_mode=current_band_mode,
        band_threshold=band_threshold,
        band_miss_weight=band_miss_weight,
        band_spill_weight=band_spill_weight,
        band_jaccard_weight=band_jaccard_weight,
        band_area_weight=band_area_weight,
        target_area_ratio=target_area_ratio,
        density_weight=density_weight,
        width_growth_weight=width_growth_weight,
        width_change_weight=width_change_weight,
        anchor_mask=anchor_mask,
        anchor_weight=anchor_weight,
        old_width=w_old,
    )
    best_score = actionable_baseline_score
    best_eta = current_eta
    best_eta_error = current_eta_error
    threshold_phi_values = phi_values_unpushed
    eta_tol = 2.0e-3
    selected_push_mode = push_direction_label(best_push_scale)
    directional_tol = max(float(min_threshold_step), 1.0e-4 * max(float(w_old), 1.0e-300))
    directional_skip_count = 0

    def threshold_radial_eta(phi_values: np.ndarray, c1: float, c2: float) -> tuple[float, float]:
        mask = refit_band_mask_numpy(
            phi_values,
            c1=c1,
            c2=c2,
            eps_ratio=eps_ratio,
            rho_amp=rho_amp,
            band_threshold=band_threshold,
            current_band_mode=current_band_mode,
        )
        return radial_mask_centroid_global(
            comm,
            points_xy,
            weights,
            mask,
            push_center=push_center,
            radius_table=radius_table,
        )

    def radial_direction_ok(trial_eta: float) -> bool:
        trial_error = abs(float(trial_eta) - design_eta)
        if effective_push_direction == "expand":
            return trial_eta >= current_eta - eta_tol and trial_error <= current_eta_error + eta_tol
        if effective_push_direction == "contract":
            return trial_eta <= current_eta + eta_tol and trial_error <= current_eta_error + eta_tol
        return trial_error <= current_eta_error + eta_tol

    def directional_threshold_ok(trial_c1: float, trial_c2: float, trial_w: float) -> bool:
        if not directional_threshold_guard:
            return True
        if selected_push_mode == "expand":
            return (
                trial_c1 <= c1_old + directional_tol
                and trial_c2 <= c2_old + directional_tol
                and trial_w >= float(w_old) - directional_tol
            )
        if selected_push_mode == "contract":
            return (
                trial_c1 >= c1_old - directional_tol
                and trial_c2 >= c2_old - directional_tol
                and trial_w <= float(w_old) + directional_tol
            )
        return True

    if verbosity >= 1:
        root_print(
            comm,
            "REFIT_THRESHOLD_BASE "
            f"pushBeta={best_push_scale:.6e} pushMode={push_direction_label(best_push_scale)} "
            f"pushScore={best_push_score.objective:.6e} "
            f"currentScore={old_design_score.objective:.6e} "
            f"actionableScore={actionable_baseline_score.objective:.6e} "
            f"actionableJ={actionable_baseline_score.band_jaccard:.6e} "
            f"currentEta={current_eta:.6e} targetEta={design_eta:.6e} "
            f"anchorPenalty={actionable_baseline_score.anchor_penalty:.6e}",
        )

    for pass_index in range(refine_passes + 1):
        # Freeze the search center for this grid pass.  Updating best_m/best_w
        # while enumerating a pass would let the grid ratchet repeatedly and
        # violate the requested local trust region.
        pass_m = best_m
        pass_w = best_w
        pass_start_obj = best_score.objective
        shifts = np.linspace(-cur_center_radius, cur_center_radius, center_grid, dtype=np.float64)
        scales = np.linspace(max(0.05, 1.0 - cur_width_radius), 1.0 + cur_width_radius, width_grid, dtype=np.float64)
        phi_values = threshold_phi_values
        if verbosity >= 1:
            root_print(
                comm,
                "REFIT_PASS "
                f"pass={pass_index} center={pass_m:.6e} width={pass_w:.6e} "
                f"bestPushBeta={best_push_scale:.6e} bestPushMode={push_direction_label(best_push_scale)} "
                f"centerRadius={cur_center_radius:.6e} widthRadius={cur_width_radius:.6e} "
                f"candidates={len(shifts) * len(scales)}",
            )
        for shift in shifts:
            for scale in scales:
                trial_m, trial_w = project_center_width(
                    pass_m + float(shift),
                    pass_w * float(scale),
                    c_max=c_max,
                    min_width=min_width,
                )
                trial_c1, trial_c2 = center_width_to_c(trial_m, trial_w)
                if not directional_threshold_ok(trial_c1, trial_c2, trial_w):
                    directional_skip_count += 1
                    continue
                trial_score = refit_score_global(
                    comm,
                    phi_values,
                    rho_values,
                    weights,
                    c1=trial_c1,
                    c2=trial_c2,
                    eps_ratio=eps_ratio,
                    rho_amp=rho_amp,
                    rho_design_l2=rho_design_l2,
                    objective_mode=objective_mode,
                    current_band_mode=current_band_mode,
                    band_threshold=band_threshold,
                    band_miss_weight=band_miss_weight,
                    band_spill_weight=band_spill_weight,
                    band_jaccard_weight=band_jaccard_weight,
                    band_area_weight=band_area_weight,
                    target_area_ratio=target_area_ratio,
                    density_weight=density_weight,
                    width_growth_weight=width_growth_weight,
                    width_change_weight=width_change_weight,
                    anchor_mask=anchor_mask,
                    anchor_weight=anchor_weight,
                    old_width=w_old,
                )
                trial_eta, _ = threshold_radial_eta(phi_values, trial_c1, trial_c2)
                trial_eta_error = abs(trial_eta - design_eta)
                if not radial_direction_ok(trial_eta):
                    continue
                if trial_score.objective < best_score.objective:
                    best_score = trial_score
                    best_m = trial_m
                    best_w = trial_w
                    best_shift = trial_m - float(m_old)
                    best_scale = trial_w / max(float(w_old), 1.0e-300)
                    best_eta = trial_eta
                    best_eta_error = trial_eta_error
        if push_callback is not None:
            push_callback(
                pass_index,
                0,
                1,
                best_push_scale,
                best_m,
                best_w,
                best_score.objective,
            )
        if verbosity >= 2:
            root_print(
                comm,
                "REFIT_THRESHOLD "
                f"pass={pass_index} pushBeta={best_push_scale:.6e} "
                f"pushMode={push_direction_label(best_push_scale)} "
                f"bestScore={best_score.objective:.6e} "
                f"recall={best_score.band_recall:.6e} precision={best_score.band_precision:.6e} "
                f"areaRatio={best_score.band_area_ratio:.6e} "
                f"eta={best_eta:.6e} etaErr={best_eta_error:.6e} fitRel={best_score.density_rel:.6e} "
                f"dirGuard={selected_push_mode if directional_threshold_guard else 'off'} "
                f"guardSkips={directional_skip_count}",
            )
        if verbosity >= 1:
            root_print(
                comm,
                "REFIT_PASS_DONE "
                f"pass={pass_index} improved={best_score.objective < pass_start_obj} "
                f"bestC1={center_width_to_c(best_m, best_w)[0]:.6e} "
                f"bestC2={center_width_to_c(best_m, best_w)[1]:.6e} "
                f"bestPushBeta={best_push_scale:.6e} bestPushMode={push_direction_label(best_push_scale)} "
                f"bestScore={best_score.objective:.6e} "
                f"recall={best_score.band_recall:.6e} precision={best_score.band_precision:.6e} "
                f"areaRatio={best_score.band_area_ratio:.6e} "
                f"eta={best_eta:.6e} etaErr={best_eta_error:.6e} "
                f"anchorPenalty={best_score.anchor_penalty:.6e} fitRel={best_score.density_rel:.6e} "
                f"dirGuard={selected_push_mode if directional_threshold_guard else 'off'} "
                f"guardSkips={directional_skip_count}",
            )
        cur_center_radius /= max(center_grid - 1, 2)
        cur_width_radius /= max(width_grid - 1, 2)

    c1, c2 = center_width_to_c(best_m, best_w)
    old_eps = float(eps_ratio) * (float(c2_old) - float(c1_old))
    new_eps = float(eps_ratio) * (float(c2) - float(c1))
    old_window = window_numpy(phi_values_unpushed, c1_old, c2_old, old_eps, rho_amp)
    new_window = window_numpy(phi_values_unpushed, c1, c2, new_eps, rho_amp)
    local_changed = float(np.dot(weights, np.abs(new_window - old_window) > 0.10 * float(rho_amp)))
    local_area = float(np.sum(weights))
    changed_area = float(comm.allreduce(local_changed, op=MPI.SUM))
    total_area = float(comm.allreduce(local_area, op=MPI.SUM))
    result = LocalRefitResult(
        c1=c1,
        c2=c2,
        center=best_m,
        width=best_w,
        shift=best_shift,
        scale=best_scale,
        push_scale=best_push_scale,
        objective_l2=best_score.density_l2,
        objective_rel=best_score.density_rel,
        accepted=(
            best_score.objective < actionable_baseline_score.objective
            and max(abs(best_m - float(m_old)), abs(best_w - float(w_old))) > float(min_threshold_step)
        ),
        changed_fraction=changed_area / max(total_area, 1.0e-300),
        fit_score=best_score.objective,
        band_jaccard=best_score.band_jaccard,
        band_recall=best_score.band_recall,
        band_precision=best_score.band_precision,
        band_dice=best_score.band_dice,
        band_miss_fraction=best_score.band_miss_fraction,
        band_spill_fraction=best_score.band_spill_fraction,
        band_area_ratio=best_score.band_area_ratio,
        band_area_mismatch=best_score.band_area_mismatch,
        radial_eta=best_eta,
        radial_eta_error=best_eta_error,
        radial_eta_step=best_eta - current_eta,
        width_growth_penalty=best_score.width_growth_penalty,
        anchor_penalty=best_score.anchor_penalty,
    )
    if verbosity >= 1:
        root_print(
            comm,
            "REFIT_DONE "
            f"c1={result.c1:.6e} c2={result.c2:.6e} "
            f"shift={result.shift:.6e} scale={result.scale:.6e} "
            f"pushBeta={result.push_scale:.6e} fitScore={result.fit_score:.6e} "
            f"bandJ={result.band_jaccard:.6e} recall={result.band_recall:.6e} "
            f"precision={result.band_precision:.6e} areaRatio={result.band_area_ratio:.6e} "
            f"eta={result.radial_eta:.6e} etaStep={result.radial_eta_step:.6e} "
            f"areaMismatch={result.band_area_mismatch:.6e} anchorPenalty={result.anchor_penalty:.6e} "
            f"fitRel={result.objective_rel:.6e} "
            f"changedFrac={result.changed_fraction:.6e} "
            f"thresholdStep={max(abs(result.center - float(m_old)), abs(result.width - float(w_old))):.6e} "
            f"minThresholdStep={float(min_threshold_step):.6e} "
            f"accepted={result.accepted} time={time.perf_counter() - t_start:.3f}",
        )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--mesh-size", type=float, default=0.18)
    parser.add_argument("--star-n", type=int, default=140)
    parser.add_argument("--star-r0", type=float, default=1.5)
    parser.add_argument("--star-amp", type=float, default=0.32)
    parser.add_argument("--star-mode", type=int, default=5)
    parser.add_argument("--gmsh-verbosity", type=int, default=0)
    parser.add_argument("--gmsh-algorithm", type=int, default=None)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--quad-degree", type=int, default=None)
    parser.add_argument("--alphaT1", dest="alpha_t1", type=float, default=None)
    parser.add_argument("--alphaT2", dest="alpha_t2", type=float, default=None)
    parser.add_argument("--eps-t-ratio", dest="eps_t_ratio", type=float, default=None)
    parser.add_argument("--eps-ratio", "--eps-phi-ratio", dest="eps_ratio", type=float, default=0.08)
    parser.add_argument("--rho-amp", type=float, default=None)
    parser.add_argument("--c1-phi", dest="c1_phi", type=float, default=None)
    parser.add_argument("--c2-phi", dest="c2_phi", type=float, default=None)
    parser.add_argument("--cmax-factor", type=float, default=1.25)
    parser.add_argument("--min-width-fraction", type=float, default=1.0e-3)
    parser.add_argument("--outer-it", type=int, default=5)
    parser.add_argument("--refit-center-fraction", type=float, default=0.25)
    parser.add_argument("--refit-width-fraction", type=float, default=0.25)
    parser.add_argument("--refit-center-grid", type=int, default=17)
    parser.add_argument("--refit-width-grid", type=int, default=17)
    parser.add_argument("--refit-refine-passes", type=int, default=1)
    parser.add_argument(
        "--refit-objective",
        choices=("band-containment", "hybrid", "density"),
        default="band-containment",
        help="cheap refit objective used to rank pushed threshold candidates",
    )
    parser.add_argument(
        "--refit-current-band-mode",
        choices=("hard-window", "active-density"),
        default="active-density",
        help="represent the semilinear band by c1<=phi<=c2 or by thresholding the logistic density",
    )
    parser.add_argument(
        "--refit-band-threshold",
        type=float,
        default=None,
        help="torsion-band threshold as a fraction of rho_amp; default uses the Newton active threshold",
    )
    parser.add_argument("--refit-band-miss-weight", type=float, default=1.0)
    parser.add_argument("--refit-band-spill-weight", type=float, default=2.0)
    parser.add_argument("--refit-band-jaccard-weight", type=float, default=0.25)
    parser.add_argument(
        "--refit-band-area-weight",
        type=float,
        default=1.0,
        help="penalty weight for mismatch between candidate active area and torsion active area",
    )
    parser.add_argument(
        "--refit-target-area-ratio",
        type=float,
        default=1.0,
        help=(
            "target candidate active-area/design-area ratio for the cheap refit; "
            "values below 1 can compensate for later Newton active-band inflation"
        ),
    )
    parser.add_argument(
        "--refit-density-weight",
        type=float,
        default=0.02,
        help="optional density-L2 tie-break weight in band-containment or hybrid mode",
    )
    parser.add_argument(
        "--refit-width-growth-weight",
        type=float,
        default=5.0,
        help="penalty on candidate width growth relative to the current width",
    )
    parser.add_argument("--refit-width-change-weight", type=float, default=0.0)
    parser.add_argument(
        "--refit-anchor-weight",
        type=float,
        default=1.0,
        help="penalty that keeps c1/c2 fitting from undoing the selected pushed active band",
    )
    parser.add_argument(
        "--refit-directional-threshold-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "keep c1/c2 updates directionally consistent with the selected push: "
            "expand lowers both thresholds without shrinking width; contract raises both thresholds without growing width"
        ),
    )
    parser.add_argument(
        "--refit-push-direction",
        choices=("auto", "both", "expand", "contract"),
        default="auto",
        help="which boundary-aware beta signs to search; auto chooses from radial drift after Newton",
    )
    parser.add_argument(
        "--push-scale-fraction",
        "--contraction-expansion-fraction",
        dest="push_scale_fraction",
        type=float,
        default=0.15,
        help="search contraction/expansion deformation parameters beta in [-f, f] during the cheap refit",
    )
    parser.add_argument("--push-scale-grid", type=int, default=9)
    parser.add_argument(
        "--push-ray-bins",
        type=int,
        default=720,
        help="angular bins used to estimate boundary ray lengths for boundary-aware push",
    )
    parser.add_argument(
        "--push-boundary",
        choices=("auto", "exact-star", "mesh-table"),
        default="auto",
        help=(
            "boundary model for the radial push; auto uses exact-star for generated "
            "smooth_star meshes and mesh-table for loaded meshes"
        ),
    )
    parser.add_argument(
        "--tol-rho-rel",
        type=float,
        default=5.0e-2,
        help="stop successfully once the Newton-projected relative torsion-density L2 error is below this value",
    )
    parser.add_argument(
        "--tol-rho-l2",
        type=float,
        default=None,
        help="optional absolute L2 torsion-density error tolerance",
    )
    parser.add_argument(
        "--tol-c-step",
        type=float,
        default=1.0e-8,
        help="stagnation safeguard for the proposed threshold center/width update; this is not a success criterion",
    )
    parser.add_argument("--fit-window-grid", type=int, default=64)
    parser.add_argument("--fit-window-refine-grid", type=int, default=25)
    parser.add_argument("--fit-window-refine-passes", type=int, default=2)
    parser.add_argument("--fit-window-bins", type=int, default=4096)
    parser.add_argument("--fit-window-quad-degree", type=int, default=None)
    parser.add_argument("--newton-max-it", type=int, default=30)
    parser.add_argument("--newton-tol-res", type=float, default=1.0e-8)
    parser.add_argument("--newton-tol-step", type=float, default=1.0e-10)
    parser.add_argument("--newton-mu-shift", type=float, default=0.0)
    parser.add_argument("--linear-solver", choices=("mumps", "lu", "hypre", "gamg"), default="mumps")
    parser.add_argument("--ksp-type", default=None)
    parser.add_argument("--linear-rtol", type=float, default=1.0e-10)
    parser.add_argument("--linear-atol", type=float, default=1.0e-12)
    parser.add_argument("--linear-max-it", type=int, default=None)
    parser.add_argument("--beta-ls", type=float, default=0.5)
    parser.add_argument("--alpha-min", type=float, default=1.0e-8)
    parser.add_argument("--max-backtrack", type=int, default=30)
    parser.add_argument("--terminal-every", type=int, default=1)
    parser.add_argument("--verbosity", "-v", type=int, choices=(0, 1, 2, 3), default=1)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-mode", choices=("blocking", "nonblocking"), default="blocking")
    parser.add_argument("--plot-off-screen", action="store_true")
    parser.add_argument("--plot-window-width", type=int, default=1800)
    parser.add_argument("--plot-window-height", type=int, default=700)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--frame-dir", type=Path, default=None)
    parser.add_argument("--frame-design", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frame-iterates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frame-refit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frame-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frame-window-width", type=int, default=1800)
    parser.add_argument("--frame-window-height", type=int, default=700)
    parser.add_argument("--fail-on-nonconvergence", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.eps_ratio <= 0.0:
        raise ValueError("require positive --eps-ratio")
    if args.outer_it < 0:
        raise ValueError("require nonnegative --outer-it")
    if args.refit_center_fraction < 0.0 or args.refit_width_fraction < 0.0:
        raise ValueError("refit fractions must be nonnegative")
    if args.refit_center_grid < 3 or args.refit_width_grid < 3:
        raise ValueError("refit grids must be at least 3")
    if args.refit_band_threshold is not None and not (0.0 <= args.refit_band_threshold <= 1.0):
        raise ValueError("require 0 <= --refit-band-threshold <= 1")
    if args.refit_band_miss_weight < 0.0 or args.refit_band_spill_weight < 0.0:
        raise ValueError("refit band weights must be nonnegative")
    if args.refit_band_jaccard_weight < 0.0 or args.refit_band_area_weight < 0.0:
        raise ValueError("refit band weights must be nonnegative")
    if args.refit_target_area_ratio <= 0.0:
        raise ValueError("require positive --refit-target-area-ratio")
    if args.refit_density_weight < 0.0 or args.refit_anchor_weight < 0.0:
        raise ValueError("refit objective weights must be nonnegative")
    if args.refit_width_growth_weight < 0.0 or args.refit_width_change_weight < 0.0:
        raise ValueError("refit width penalties must be nonnegative")
    if args.push_scale_fraction < 0.0:
        raise ValueError("require nonnegative --push-scale-fraction")
    if args.push_scale_grid < 1:
        raise ValueError("require --push-scale-grid >= 1")
    if args.push_ray_bins < 32:
        raise ValueError("require --push-ray-bins >= 32")
    if args.push_boundary == "exact-star" and float(args.star_r0) <= abs(float(args.star_amp)):
        raise ValueError("exact-star push requires --star-r0 > abs(--star-amp)")
    if args.tol_rho_rel <= 0.0:
        raise ValueError("require positive --tol-rho-rel")
    if args.tol_rho_l2 is not None and args.tol_rho_l2 <= 0.0:
        raise ValueError("require positive --tol-rho-l2 when supplied")
    if args.tol_c_step < 0.0:
        raise ValueError("require nonnegative --tol-c-step")
    if args.newton_max_it < 0:
        raise ValueError("require nonnegative --newton-max-it")
    if args.newton_tol_res <= 0.0 or args.newton_tol_step <= 0.0:
        raise ValueError("require positive Newton tolerances")
    if args.c1_phi is None and args.c2_phi is not None:
        raise ValueError("--c1-phi and --c2-phi must be supplied together")
    if args.c1_phi is not None and args.c2_phi is None:
        raise ValueError("--c1-phi and --c2-phi must be supplied together")
    if args.c1_phi is not None and not (args.c2_phi > args.c1_phi >= 0.0):
        raise ValueError("require 0 <= c1_phi < c2_phi")


def run_strategy(args: argparse.Namespace) -> int:
    validate_args(args)
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

    loop_csv = log_dir / "closed_loop_refit.csv"
    newton_csv = log_dir / "newton.csv"
    frame_csv = log_dir / "frames.csv"
    summary_path = out_dir / "summary.txt"
    total_start = time.perf_counter()

    root_print(comm, "========== START STRATEGY A CLOSED-LOOP REFIT ==========")
    root_print(comm, f"RUN_TAG {run_tag}")
    root_print(comm, f"RUN_DIR {run_dir}")
    root_print(comm, f"LOOP_CSV {loop_csv}")
    root_print(comm, f"NEWTON_CSV {newton_csv}")
    root_print(comm, f"SUMMARY {summary_path}")

    domain, mesh_path, geometry_mode = load_or_generate_mesh(args, run_dir, comm)
    tdim = domain.topology.dim
    nt = int(domain.topology.index_map(tdim).size_global)
    root_print(comm, f"GEOMETRY {geometry_mode} meshFile={mesh_path}")
    root_print(comm, f"MESH nt={nt}")

    V = fem.functionspace(domain, ("Lagrange", int(args.order)))
    ndof = int(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)
    bc = boundary_bc(V)
    qdeg = args.quad_degree if args.quad_degree is not None else max(2 * int(args.order) + 8, 12)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": qdeg})
    z = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    root_print(comm, f"SPACE order={args.order} ndof={ndof} quadDegree={qdeg}")

    T = fem.Function(V, name="T")
    rho_design = fem.Function(V, name="rhoDesign")
    phi_target = fem.Function(V, name="phiT")
    u = fem.Function(V, name="phi")
    rho = fem.Function(V, name="rho")
    du = fem.Function(V, name="newtonStep")
    phi_diff = fem.Function(V, name="phiMinusPhiT")
    rho_proposal = fem.Function(V, name="rhoProposal")
    phi_push_plot = fem.Function(V, name="phiPush")
    rho_push_plot = fem.Function(V, name="rhoPushFit")
    rho_error_plot = fem.Function(V, name="rhoMinusDesign")
    design_mask_plot = fem.Function(V, name="designBand")
    current_mask_plot = fem.Function(V, name="currentBand")
    proposal_mask_plot = fem.Function(V, name="proposalBand")
    pushed_mask_plot = fem.Function(V, name="pushedBand")
    proposal_move_plot = fem.Function(V, name="proposalMove")
    pushed_move_plot = fem.Function(V, name="pushedMove")

    frame_fields = [
        "frame", "runTag", "stage", "ieps", "k", "nt", "ndof", "epsPhi", "resEuclid",
        "massRho", "maxRho", "activeArea", "plateauArea", "relRhoDesign", "filename",
    ]
    frame_handle = frame_csv.open("w", newline="", encoding="utf-8") if comm.rank == 0 else None
    frame_writer = csv.DictWriter(frame_handle, fieldnames=frame_fields) if comm.rank == 0 else None
    if frame_writer is not None:
        frame_writer.writeheader()
    plotter = InPlacePyVistaTorsionPlotter(args, run_tag=run_tag, run_dir=run_dir, frame_writer=frame_writer, comm=comm)

    if args.plot:
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

    solve_linear_form(
        ufl.inner(ufl.grad(z), ufl.grad(v)) * dx,
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
    update_interpolated(rho_design, window_ufl(T, c1_t, c2_t, eps_t, params.rho_amp))
    rho_design_l2 = math.sqrt(max(assemble_scalar(comm, rho_design * rho_design * dx), 0.0))
    rho_design_mass = assemble_scalar(comm, rho_design * dx)
    _, rho_design_max = global_minmax(comm, rho_design)
    if args.plot or args.save_frames:
        plotter.emit(
            [T, rho_design],
            ["Torsion T", "rho_T"],
            stage="TORSION_DESIGN",
            ieps=0,
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

    solve_linear_form(
        ufl.inner(ufl.grad(z), ufl.grad(v)) * dx,
        rho_design * v * dx,
        phi_target,
        [bc],
        prefix="phi_target_",
        solver=args.linear_solver,
        ksp_type=args.ksp_type,
        rtol=args.linear_rtol,
        atol=args.linear_atol,
        max_it=args.linear_max_it,
        verbosity=args.verbosity,
    )
    _, phi_target_max = global_minmax(comm, phi_target)
    c_max = max(float(args.cmax_factor) * phi_target_max, 1.0e-12)
    min_width = max(float(args.min_width_fraction) * c_max, 1.0e-14)
    use_exact_star_boundary = args.push_boundary == "exact-star" or (
        args.push_boundary == "auto" and geometry_mode == "smooth_star"
    )
    if use_exact_star_boundary:
        push_center = np.array([0.0, 0.0], dtype=np.float64)
        radius_table: BoundaryRadiusTable | SmoothStarBoundary = SmoothStarBoundary(
            radius=float(args.star_r0),
            amplitude=float(args.star_amp),
            mode=int(args.star_mode),
        )
        push_boundary_mode = "exact_star"
    else:
        push_center = global_bbox_center(domain)
        radius_table = build_boundary_radius_table(domain, push_center, bins=args.push_ray_bins)
        push_boundary_mode = "mesh_table"
    root_print(
        comm,
        f"PUSH_CENTER x={push_center[0]:.6e} y={push_center[1]:.6e} "
        f"betaFraction={args.push_scale_fraction:.6e} boundary={push_boundary_mode} "
        f"rayBins={args.push_ray_bins}",
    )
    if use_exact_star_boundary:
        root_print(
            comm,
            "PUSH_BOUNDARY_EXACT_STAR "
            f"R(theta)={args.star_r0:.6e}+{args.star_amp:.6e}*cos({int(args.star_mode)}*theta)",
        )
    root_print(
        comm,
        "ALGORITHM closed_loop_refit "
        f"tolRhoRel={args.tol_rho_rel:.6e} "
        f"tolRhoL2={args.tol_rho_l2 if args.tol_rho_l2 is not None else 'None'} "
        f"tolCStep={args.tol_c_step:.6e}",
    )
    refit_band_threshold = (
        float(args.refit_band_threshold)
        if args.refit_band_threshold is not None
        else float(params.active_threshold)
    )
    root_print(
        comm,
        "REFIT_OBJECTIVE "
        f"mode={args.refit_objective} currentBand={args.refit_current_band_mode} "
        f"bandThreshold={refit_band_threshold:.6e} missW={args.refit_band_miss_weight:.6e} "
        f"spillW={args.refit_band_spill_weight:.6e} jaccardW={args.refit_band_jaccard_weight:.6e} "
        f"areaW={args.refit_band_area_weight:.6e} targetAreaRatio={args.refit_target_area_ratio:.6e} "
        f"densityW={args.refit_density_weight:.6e} "
        f"widthGrowthW={args.refit_width_growth_weight:.6e} anchorW={args.refit_anchor_weight:.6e} "
        f"pushDirection={args.refit_push_direction} "
        f"directionalThresholdGuard={args.refit_directional_threshold_guard}",
    )

    if args.c1_phi is not None:
        c1_phi = float(args.c1_phi)
        c2_phi = float(args.c2_phi)
    else:
        fit_quad_degree = args.fit_window_quad_degree if args.fit_window_quad_degree is not None else qdeg
        fit_result = fit_phi_window_to_torsion_design(
            phi_target,
            rho_design,
            rho_design_l2=rho_design_l2,
            phi_design_max=phi_target_max,
            eps_ratio=args.eps_ratio,
            rho_amp=params.rho_amp,
            quadrature_degree=fit_quad_degree,
            grid_points=args.fit_window_grid,
            refine_points=args.fit_window_refine_grid,
            refine_passes=args.fit_window_refine_passes,
            histogram_bins=args.fit_window_bins,
        )
        c1_phi = fit_result.c1
        c2_phi = fit_result.c2
    m, w = project_center_width(0.5 * (c1_phi + c2_phi), c2_phi - c1_phi, c_max=c_max, min_width=min_width)
    c1_phi, c2_phi = center_width_to_c(m, w)
    u.x.array[:] = phi_target.x.array
    u.x.scatter_forward()
    root_print(
        comm,
        f"INIT c1={c1_phi:.6e} c2={c2_phi:.6e} width={w:.6e} epsPhi={args.eps_ratio * w:.6e}",
    )

    loop_fields = [
        "record", "outer", "nt", "ndof", "c1Phi", "c2Phi", "center", "width", "epsPhi",
        "newC1Phi", "newC2Phi", "newCenter", "newWidth", "shift", "scale", "pushScale", "pushMode",
        "fitL2", "fitRel", "fitScore", "bandJaccard", "bandRecall", "bandPrecision",
        "bandDice", "bandMiss", "bandSpill", "bandAreaRatio", "bandAreaMismatch",
        "radialEta", "radialEtaError", "radialEtaStep", "widthGrowthPenalty", "anchorPenalty",
        "changedFraction", "thresholdStep", "newtonStatus", "newtonIterations",
        "resEuclid", "maxPhi", "maxRho", "massRho", "activeArea", "activeDesignArea",
        "activeOverlapArea", "phiDiffL2", "rhoDiffL2",
        "massDiff", "activeJaccard", "plateauJaccard", "relRhoDesign", "status",
    ]
    newton_fields = [
        "record", "outer", "k", "nt", "ndof", "resEuclid", "stepH1", "alpha", "bt", "muShift",
        "solveTime", "metricTime", "stepTime", "linearIterations", "linearResidual",
        "maxPhi", "maxRho", "massRho", "activeJaccard", "plateauJaccard", "relRhoDesign", "status",
    ]
    loop_handle = loop_csv.open("w", newline="", encoding="utf-8") if comm.rank == 0 else None
    loop_writer = csv.DictWriter(loop_handle, fieldnames=loop_fields) if comm.rank == 0 else None
    if loop_writer is not None:
        loop_writer.writeheader()
    newton_handle = newton_csv.open("w", newline="", encoding="utf-8") if comm.rank == 0 else None
    newton_writer_base = csv.DictWriter(newton_handle, fieldnames=newton_fields) if comm.rank == 0 else None
    if newton_writer_base is not None:
        newton_writer_base.writeheader()

    class _NewtonWriter:
        """Attach the current outer index to rows emitted by run_newton_polish."""

        def __init__(
                self,
                writer: csv.DictWriter | None,
                outer: int,
                c1_phi: float,
                c2_phi: float,
                eps_phi: float,
        ):
            self.writer = writer
            self.outer = outer
            self.c1_phi = c1_phi
            self.c2_phi = c2_phi
            self.eps_phi = eps_phi

        def writerow(self, row: dict) -> None:
            if self.writer is not None:
                row = dict(row)
                row["outer"] = self.outer
                self.writer.writerow(row)
            if args.plot:
                emit_newton_polish_plot(
                    outer=self.outer,
                    row=row,
                    c1_phi=self.c1_phi,
                    c2_phi=self.c2_phi,
                    eps_phi=self.eps_phi,
                )

    def write_projected_state(
            *,
            outer: int,
            c1_phi: float,
            c2_phi: float,
            center: float,
            width: float,
            eps_phi: float,
            newton_result: NewtonPolishResult,
            metrics: dict[str, float],
            phi_l2: float,
            rho_l2: float,
            mass_diff: float,
            status: str,
    ) -> None:
        """Write a row for a Newton-projected state with no threshold proposal."""
        if loop_writer is None:
            return
        loop_writer.writerow({
            "record": "STATE",
            "outer": outer,
            "nt": nt,
            "ndof": ndof,
            "c1Phi": c1_phi,
            "c2Phi": c2_phi,
            "center": center,
            "width": width,
            "epsPhi": eps_phi,
            "newC1Phi": "",
            "newC2Phi": "",
            "newCenter": "",
            "newWidth": "",
            "shift": "",
            "scale": "",
            "pushScale": "",
            "pushMode": "",
            "fitL2": "",
            "fitRel": "",
            "fitScore": "",
            "bandJaccard": "",
            "bandRecall": "",
            "bandPrecision": "",
            "bandDice": "",
            "bandMiss": "",
            "bandSpill": "",
            "bandAreaRatio": "",
            "bandAreaMismatch": "",
            "radialEta": "",
            "radialEtaError": "",
            "radialEtaStep": "",
            "widthGrowthPenalty": "",
            "anchorPenalty": "",
            "changedFraction": "",
            "thresholdStep": "",
            "newtonStatus": newton_result.status,
            "newtonIterations": newton_result.iterations,
            "resEuclid": metrics["resEuclid"],
            "maxPhi": metrics["maxU"],
            "maxRho": metrics["maxRho"],
            "massRho": metrics["massRho"],
            "activeArea": metrics["activeArea"],
            "activeDesignArea": metrics["activeDesignArea"],
            "activeOverlapArea": metrics["activeOverlapArea"],
            "phiDiffL2": phi_l2,
            "rhoDiffL2": rho_l2,
            "massDiff": mass_diff,
            "activeJaccard": metrics["activeJaccard"],
            "plateauJaccard": metrics["plateauJaccard"],
            "relRhoDesign": metrics["relRhoDesign"],
            "status": status,
        })
        loop_handle.flush()

    def emit_newton_polish_plot(
            *,
            outer: int,
            row: dict,
            c1_phi: float,
            c2_phi: float,
            eps_phi: float,
    ) -> None:
        """Render every Newton polish state emitted by ``run_newton_polish``."""
        rho_error_plot.x.array[:] = rho.x.array - rho_design.x.array
        rho_error_plot.x.scatter_forward()
        k = row.get("k", -1)
        status = row.get("status", "")
        rel = row.get("relRhoDesign", "")
        res = float(row.get("resEuclid", 0.0) or 0.0)
        rel_text = f"{float(rel):.3e}" if rel != "" else "NA"
        try:
            k_token = int(k)
        except (TypeError, ValueError):
            k_token = -1
        plotter.emit(
            [rho_design, rho, rho_error_plot, u],
            [
                "rho_T",
                f"Newton rho outer={outer} k={k}\nc1={c1_phi:.3e} c2={c2_phi:.3e}",
                f"rho-rho_T\nrel={rel_text} status={status}",
                "Newton phi",
            ],
            stage="NEWTON_POLISH",
            ieps=0,
            k=k,
            eps_phi=eps_phi,
            residual=res,
            metrics=row,
            token=f"newton_{outer:03d}_{k_token:03d}",
            save=bool(args.frame_iterates),
            show=True,
            nt=nt,
            ndof=ndof,
        )

    def emit_projected_state_plot(
            *,
            outer: int,
            c1_phi: float,
            c2_phi: float,
            eps_phi: float,
            metrics: dict[str, float],
            rho_l2: float,
            rho_rel: float,
    ) -> None:
        """Plot the Newton-projected state for the current thresholds."""
        rho_error_plot.x.array[:] = rho.x.array - rho_design.x.array
        rho_error_plot.x.scatter_forward()
        plotter.emit(
            [rho_design, rho, rho_error_plot, u],
            [
                "rho_T",
                f"rho(phi;c) k={outer}\nc1={c1_phi:.3e} c2={c2_phi:.3e}",
                f"rho-rho_T\nrel={rho_rel:.3e} L2={rho_l2:.3e}",
                "Newton phi",
            ],
            stage="LOOP_STATE",
            ieps=0,
            k=outer,
            eps_phi=eps_phi,
            residual=float(metrics["resEuclid"]),
            metrics=metrics,
            token=f"loop_state_{outer:03d}",
            save=bool(args.frame_iterates),
            show=True,
            nt=nt,
            ndof=ndof,
        )

    def update_refit_design_mask(target: fem.Function) -> None:
        """Update the plotted torsion band using the same mode as the refit score."""
        if args.refit_current_band_mode == "active-density":
            update_active_mask(
                target,
                rho_design,
                threshold=refit_band_threshold * params.rho_amp,
            )
        else:
            update_window_band_mask(target, T, c1=c1_t, c2=c2_t)

    def update_refit_candidate_mask(
            target: fem.Function,
            source_phi: fem.Function,
            *,
            c1: float,
            c2: float,
            work_density: fem.Function,
    ) -> None:
        """Update a plotted semilinear band using the same mode as the refit score."""
        if args.refit_current_band_mode == "active-density":
            update_interpolated(
                work_density,
                window_ufl(source_phi, c1, c2, args.eps_ratio * (float(c2) - float(c1)), params.rho_amp),
            )
            update_active_mask(
                target,
                work_density,
                threshold=refit_band_threshold * params.rho_amp,
            )
        else:
            update_window_band_mask(target, source_phi, c1=c1, c2=c2)

    def emit_refit_plot(
            *,
            outer: int,
            refit: LocalRefitResult,
            current_c1: float,
            current_c2: float,
            current_eps: float,
            metrics: dict[str, float],
    ) -> None:
        """Plot the c1/c2 proposal, including the pushed scalar-coordinate fit."""
        band_label = "active band" if args.refit_current_band_mode == "active-density" else "phi-window"
        update_refit_design_mask(design_mask_plot)
        update_refit_candidate_mask(
            current_mask_plot,
            u,
            c1=current_c1,
            c2=current_c2,
            work_density=rho_proposal,
        )
        update_refit_candidate_mask(
            proposal_mask_plot,
            u,
            c1=refit.c1,
            c2=refit.c2,
            work_density=rho_proposal,
        )
        update_difference(rho_error_plot, current_mask_plot, design_mask_plot)
        update_difference(proposal_move_plot, proposal_mask_plot, design_mask_plot)
        fields = [design_mask_plot, current_mask_plot, rho_error_plot, proposal_mask_plot, proposal_move_plot]
        titles = [
            f"target {band_label}",
            f"current {band_label}\nc1={current_c1:.3e} c2={current_c2:.3e}",
            "current-target\nbright=spill dark=miss",
            f"proposal {band_label}\nc1={refit.c1:.3e} c2={refit.c2:.3e}",
            "proposal-target\nbright=spill dark=miss",
        ]
        try:
            update_pushed_function(
                phi_push_plot,
                u,
                push_center=push_center,
                push_beta=refit.push_scale,
                radius_table=radius_table,
            )
            update_refit_candidate_mask(
                pushed_mask_plot,
                phi_push_plot,
                c1=refit.c1,
                c2=refit.c2,
                work_density=rho_push_plot,
            )
            update_difference(pushed_move_plot, pushed_mask_plot, design_mask_plot)
            fields.extend([pushed_mask_plot, pushed_move_plot])
            titles.extend([
                f"pushed reference {band_label}\nbeta={refit.push_scale:.3e} {push_direction_label(refit.push_scale)}",
                "pushed reference-target\nbright=spill dark=miss",
            ])
        except Exception as exc:
            root_print(comm, f"PLOT_REFIT_PUSH_SKIP k={outer} error={type(exc).__name__}: {exc}")
        plotter.emit(
            fields,
            titles,
            stage="REFIT_PROPOSAL",
            ieps=0,
            k=outer,
            eps_phi=current_eps,
            residual=float(metrics["resEuclid"]),
            metrics=metrics,
            token=f"refit_{outer:03d}",
            save=bool(args.frame_refit),
            show=True,
            nt=nt,
            ndof=ndof,
        )

    def emit_refit_push_plot(
            *,
            outer: int,
            pass_index: int,
            push_index: int,
            push_count: int,
            push_beta: float,
            best_m: float,
            best_w: float,
            best_score: float,
            current_c1: float,
            current_c2: float,
            current_eps: float,
            metrics: dict[str, float],
    ) -> None:
        """Render one boundary-aware push scale during the c1/c2 refit scan."""
        if not args.plot:
            return
        best_c1, best_c2 = center_width_to_c(best_m, best_w)
        band_label = "active band" if args.refit_current_band_mode == "active-density" else "phi-window"
        update_pushed_function(
            phi_push_plot,
            u,
            push_center=push_center,
            push_beta=push_beta,
            radius_table=radius_table,
        )
        update_refit_design_mask(design_mask_plot)
        update_refit_candidate_mask(
            current_mask_plot,
            u,
            c1=current_c1,
            c2=current_c2,
            work_density=rho_proposal,
        )
        update_refit_candidate_mask(
            pushed_mask_plot,
            phi_push_plot,
            c1=best_c1,
            c2=best_c2,
            work_density=rho_push_plot,
        )
        update_difference(proposal_move_plot, current_mask_plot, design_mask_plot)
        update_difference(pushed_move_plot, pushed_mask_plot, design_mask_plot)
        plotter.emit(
            [design_mask_plot, current_mask_plot, proposal_move_plot, pushed_mask_plot, pushed_move_plot],
            [
                f"target {band_label}",
                f"current {band_label}\nc1={current_c1:.3e} c2={current_c2:.3e}",
                "current-target\nbright=spill dark=miss",
                f"push {push_index + 1}/{push_count} pass={pass_index}\nbeta={push_beta:.3e} {push_direction_label(push_beta)}",
                f"pushed-target\nbestScore={best_score:.3e}",
            ],
            stage="REFIT_PUSH",
            ieps=pass_index,
            k=push_index,
            eps_phi=current_eps,
            residual=float(metrics["resEuclid"]),
            metrics=metrics,
            token=f"refit_push_{outer:03d}_{pass_index:02d}_{push_index:03d}",
            save=bool(args.frame_refit),
            show=True,
            nt=nt,
            ndof=ndof,
        )

    final_metrics: dict[str, float] | None = None
    final_newton: NewtonPolishResult | None = None
    final_status = "MAX_OUTER"
    converged_newton_statuses = {"CONVERGED_RESIDUAL", "CONVERGED_STEP"}
    pending_projected_refit: dict | None = None

    try:
        for outer in range(int(args.outer_it) + 1):
            c1_phi, c2_phi = center_width_to_c(m, w)
            eps_phi = args.eps_ratio * w
            newton_result, metrics = run_newton_polish(
                u=u,
                du=du,
                rho=rho,
                phi_target=phi_target,
                rho_design=rho_design,
                rho_design_l2=rho_design_l2,
                z=z,
                v=v,
                dx=dx,
                bc=bc,
                c1=c1_phi,
                c2=c2_phi,
                eps_phi=eps_phi,
                rho_amp=params.rho_amp,
                active_threshold=params.active_threshold,
                plateau_threshold=params.plateau_threshold,
                start="penalty",
                max_it=args.newton_max_it,
                tol_res=args.newton_tol_res,
                tol_step=args.newton_tol_step,
                armijo_c=1.0e-4,
                beta_ls=args.beta_ls,
                alpha_min=args.alpha_min,
                max_backtrack=args.max_backtrack,
                mu_shift=args.newton_mu_shift,
                linear_solver=args.linear_solver,
                ksp_type=args.ksp_type,
                linear_rtol=args.linear_rtol,
                linear_atol=args.linear_atol,
                linear_max_it=args.linear_max_it,
                verbosity=args.verbosity,
                terminal_every=args.terminal_every,
                writer=_NewtonWriter(newton_writer_base, outer, c1_phi, c2_phi, eps_phi),
                handle=newton_handle,
                comm=comm,
                nt=nt,
                ndof=ndof,
            )
            final_metrics = metrics
            final_newton = newton_result

            phi_l2 = math.sqrt(max(assemble_scalar(comm, (u - phi_target) ** 2 * dx), 0.0))
            rho_l2 = float(metrics["rhoDesignDiffL2"])
            mass_diff = float(metrics["massRhoMinusDesign"])
            rho_rel = float(metrics["relRhoDesign"])
            root_print(
                comm,
                f"LOOP_STATE k={outer} newton={newton_result.status} resE={metrics['resEuclid']:.6e} "
                f"rhoL2={rho_l2:.6e} rhoRel={rho_rel:.6e} "
                f"activeArea={metrics['activeArea']:.6e} activeJ={metrics['activeJaccard']:.6e} "
                f"c1={c1_phi:.6e} c2={c2_phi:.6e}",
            )
            emit_projected_state_plot(
                outer=outer,
                c1_phi=c1_phi,
                c2_phi=c2_phi,
                eps_phi=eps_phi,
                metrics=metrics,
                rho_l2=rho_l2,
                rho_rel=rho_rel,
            )

            if newton_result.status not in converged_newton_statuses:
                final_status = f"NEWTON_{newton_result.status}"
                write_projected_state(
                    outer=outer,
                    c1_phi=c1_phi,
                    c2_phi=c2_phi,
                    center=m,
                    width=w,
                    eps_phi=eps_phi,
                    newton_result=newton_result,
                    metrics=metrics,
                    phi_l2=phi_l2,
                    rho_l2=rho_l2,
                    mass_diff=mass_diff,
                    status=final_status,
                )
                root_print(comm, f"LOOP_STOP reason={final_status} rhoRel={rho_rel:.6e} rhoL2={rho_l2:.6e}")
                break

            if pending_projected_refit is not None:
                previous_rho_rel = float(pending_projected_refit["rho_rel"])
                previous_active_j = float(pending_projected_refit["active_jaccard"])
                projected_worse = (
                    rho_rel > previous_rho_rel
                    and float(metrics["activeJaccard"]) < previous_active_j
                )
                if projected_worse:
                    u.x.array[:] = pending_projected_refit["u_values"]
                    u.x.scatter_forward()
                    rho.x.array[:] = pending_projected_refit["rho_values"]
                    rho.x.scatter_forward()
                    m = float(pending_projected_refit["m"])
                    w = float(pending_projected_refit["w"])
                    c1_restore, c2_restore = center_width_to_c(m, w)
                    eps_restore = args.eps_ratio * w
                    final_metrics = dict(pending_projected_refit["metrics"])
                    final_newton = pending_projected_refit["newton_result"]
                    final_status = "REJECTED_PROJECTED_REFIT"
                    write_projected_state(
                        outer=outer,
                        c1_phi=c1_restore,
                        c2_phi=c2_restore,
                        center=m,
                        width=w,
                        eps_phi=eps_restore,
                        newton_result=final_newton,
                        metrics=final_metrics,
                        phi_l2=float(pending_projected_refit["phi_l2"]),
                        rho_l2=float(pending_projected_refit["rho_l2"]),
                        mass_diff=float(pending_projected_refit["mass_diff"]),
                        status=final_status,
                    )
                    root_print(
                        comm,
                        "LOOP_STOP "
                        f"reason={final_status} rejectedOuter={outer} "
                        f"rhoRel={rho_rel:.6e}>{previous_rho_rel:.6e} "
                        f"activeJ={metrics['activeJaccard']:.6e}<{previous_active_j:.6e}",
                    )
                    pending_projected_refit = None
                    break
                pending_projected_refit = None

            if torsion_error_reached(
                    rho_l2=rho_l2,
                    rho_rel=rho_rel,
                    tol_l2=args.tol_rho_l2,
                    tol_rel=args.tol_rho_rel,
            ):
                final_status = "CONVERGED_TORSION"
                write_projected_state(
                    outer=outer,
                    c1_phi=c1_phi,
                    c2_phi=c2_phi,
                    center=m,
                    width=w,
                    eps_phi=eps_phi,
                    newton_result=newton_result,
                    metrics=metrics,
                    phi_l2=phi_l2,
                    rho_l2=rho_l2,
                    mass_diff=mass_diff,
                    status=final_status,
                )
                root_print(comm, f"LOOP_STOP reason={final_status} rhoRel={rho_rel:.6e} rhoL2={rho_l2:.6e}")
                break

            if outer == int(args.outer_it):
                final_status = "MAX_OUTER"
                write_projected_state(
                    outer=outer,
                    c1_phi=c1_phi,
                    c2_phi=c2_phi,
                    center=m,
                    width=w,
                    eps_phi=eps_phi,
                    newton_result=newton_result,
                    metrics=metrics,
                    phi_l2=phi_l2,
                    rho_l2=rho_l2,
                    mass_diff=mass_diff,
                    status=final_status,
                )
                root_print(comm, f"LOOP_STOP reason={final_status} rhoRel={rho_rel:.6e} rhoL2={rho_l2:.6e}")
                break

            refit = local_center_width_refit(
                phi=u,
                rho_design=rho_design,
                rho_design_l2=rho_design_l2,
                m_old=m,
                w_old=w,
                c_max=c_max,
                min_width=min_width,
                eps_ratio=args.eps_ratio,
                rho_amp=params.rho_amp,
                objective_mode=args.refit_objective,
                current_band_mode=args.refit_current_band_mode,
                band_threshold=refit_band_threshold,
                band_miss_weight=args.refit_band_miss_weight,
                band_spill_weight=args.refit_band_spill_weight,
                band_jaccard_weight=args.refit_band_jaccard_weight,
                band_area_weight=args.refit_band_area_weight,
                target_area_ratio=args.refit_target_area_ratio,
                density_weight=args.refit_density_weight,
                width_growth_weight=args.refit_width_growth_weight,
                width_change_weight=args.refit_width_change_weight,
                anchor_weight=args.refit_anchor_weight,
                directional_threshold_guard=args.refit_directional_threshold_guard,
                push_direction=args.refit_push_direction,
                center_fraction=args.refit_center_fraction,
                width_fraction=args.refit_width_fraction,
                center_grid=args.refit_center_grid,
                width_grid=args.refit_width_grid,
                refine_passes=args.refit_refine_passes,
                push_center=push_center,
                radius_table=radius_table,
                push_scale_fraction=args.push_scale_fraction,
                push_scale_grid=args.push_scale_grid,
                quadrature_degree=qdeg,
                min_threshold_step=args.tol_c_step,
                verbosity=args.verbosity,
                push_callback=lambda pass_i, push_i, push_n, beta, best_m, best_w, best_score: emit_refit_push_plot(
                    outer=outer,
                    pass_index=pass_i,
                    push_index=push_i,
                    push_count=push_n,
                    push_beta=beta,
                    best_m=best_m,
                    best_w=best_w,
                    best_score=best_score,
                    current_c1=c1_phi,
                    current_c2=c2_phi,
                    current_eps=eps_phi,
                    metrics=metrics,
                ),
            )
            emit_refit_plot(
                outer=outer,
                refit=refit,
                current_c1=c1_phi,
                current_c2=c2_phi,
                current_eps=eps_phi,
                metrics=metrics,
            )
            threshold_step = max(abs(refit.center - m), abs(refit.width - w))
            step_stagnated = threshold_step <= float(args.tol_c_step)
            if refit.accepted:
                status = "ACCEPT_REFIT"
            elif step_stagnated:
                status = "NO_ACTIONABLE_C_STEP"
            else:
                status = "NO_REFIT_IMPROVEMENT"
            if loop_writer is not None:
                loop_writer.writerow({
                    "record": "OUTER",
                    "outer": outer,
                    "nt": nt,
                    "ndof": ndof,
                    "c1Phi": c1_phi,
                    "c2Phi": c2_phi,
                    "center": m,
                    "width": w,
                    "epsPhi": eps_phi,
                    "newC1Phi": refit.c1,
                    "newC2Phi": refit.c2,
                    "newCenter": refit.center,
                    "newWidth": refit.width,
                    "shift": refit.shift,
                    "scale": refit.scale,
                    "pushScale": refit.push_scale,
                    "pushMode": push_direction_label(refit.push_scale),
                    "fitL2": refit.objective_l2,
                    "fitRel": refit.objective_rel,
                    "fitScore": refit.fit_score,
                    "bandJaccard": refit.band_jaccard,
                    "bandRecall": refit.band_recall,
                    "bandPrecision": refit.band_precision,
                    "bandDice": refit.band_dice,
                    "bandMiss": refit.band_miss_fraction,
                    "bandSpill": refit.band_spill_fraction,
                    "bandAreaRatio": refit.band_area_ratio,
                    "bandAreaMismatch": refit.band_area_mismatch,
                    "radialEta": refit.radial_eta,
                    "radialEtaError": refit.radial_eta_error,
                    "radialEtaStep": refit.radial_eta_step,
                    "widthGrowthPenalty": refit.width_growth_penalty,
                    "anchorPenalty": refit.anchor_penalty,
                    "changedFraction": refit.changed_fraction,
                    "thresholdStep": threshold_step,
                    "newtonStatus": newton_result.status,
                    "newtonIterations": newton_result.iterations,
                    "resEuclid": metrics["resEuclid"],
                    "maxPhi": metrics["maxU"],
                    "maxRho": metrics["maxRho"],
                    "massRho": metrics["massRho"],
                    "activeArea": metrics["activeArea"],
                    "activeDesignArea": metrics["activeDesignArea"],
                    "activeOverlapArea": metrics["activeOverlapArea"],
                    "phiDiffL2": phi_l2,
                    "rhoDiffL2": rho_l2,
                    "massDiff": mass_diff,
                    "activeJaccard": metrics["activeJaccard"],
                    "plateauJaccard": metrics["plateauJaccard"],
                    "relRhoDesign": metrics["relRhoDesign"],
                    "status": status,
                })
                loop_handle.flush()
            root_print(
                comm,
                f"LOOP_REFIT k={outer} c1={c1_phi:.6e} c2={c2_phi:.6e} "
                f"-> c1New={refit.c1:.6e} c2New={refit.c2:.6e} "
                f"step={threshold_step:.6e} pushBeta={refit.push_scale:.6e} "
                f"pushMode={push_direction_label(refit.push_scale)} "
                f"fitScore={refit.fit_score:.6e} bandJ={refit.band_jaccard:.6e} "
                f"recall={refit.band_recall:.6e} precision={refit.band_precision:.6e} "
                f"areaRatio={refit.band_area_ratio:.6e} areaMismatch={refit.band_area_mismatch:.6e} "
                f"eta={refit.radial_eta:.6e} etaStep={refit.radial_eta_step:.6e} "
                f"activeArea={metrics['activeArea']:.6e} "
                f"anchorPenalty={refit.anchor_penalty:.6e} fitRel={refit.objective_rel:.6e} "
                f"changedFrac={refit.changed_fraction:.6e} status={status}",
            )
            if not refit.accepted:
                final_status = "STAGNATED_C_STEP" if step_stagnated else "STAGNATED_REFIT"
                root_print(comm, f"LOOP_STOP reason={final_status} rhoRel={rho_rel:.6e} rhoL2={rho_l2:.6e}")
                break
            if step_stagnated:
                final_status = "STAGNATED_C_STEP"
                root_print(
                    comm,
                    f"LOOP_STOP reason={final_status} step={threshold_step:.6e} "
                    f"rhoRel={rho_rel:.6e} rhoL2={rho_l2:.6e}",
                )
                break
            pending_projected_refit = {
                "m": float(m),
                "w": float(w),
                "u_values": np.asarray(u.x.array, dtype=np.float64).copy(),
                "rho_values": np.asarray(rho.x.array, dtype=np.float64).copy(),
                "metrics": dict(metrics),
                "newton_result": newton_result,
                "phi_l2": float(phi_l2),
                "rho_l2": float(rho_l2),
                "mass_diff": float(mass_diff),
                "rho_rel": float(rho_rel),
                "active_jaccard": float(metrics["activeJaccard"]),
            }
            m = refit.center
            w = refit.width
    finally:
        if loop_handle is not None:
            loop_handle.close()
        if newton_handle is not None:
            newton_handle.close()

    c1_phi, c2_phi = center_width_to_c(m, w)
    eps_phi = args.eps_ratio * w
    update_interpolated(rho, window_ufl(u, c1_phi, c2_phi, eps_phi, params.rho_amp))
    if final_metrics is None:
        raise RuntimeError("closed-loop refit did not produce final metrics")
    phi_l2 = math.sqrt(max(assemble_scalar(comm, (u - phi_target) ** 2 * dx), 0.0))

    if args.plot or args.save_frames:
        phi_diff.x.array[:] = u.x.array - phi_target.x.array
        phi_diff.x.scatter_forward()
        plotter.emit(
            [T, rho_design, phi_target, u, rho, phi_diff],
            ["Torsion T", "rhoDesign", "phiT", "Closed-loop phi", "rho(phi;c)", "phi-phiT"],
            stage="FINAL",
            ieps=0,
            k=-1,
            eps_phi=eps_phi,
            residual=float(final_metrics["resEuclid"]),
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
    root_print(
        comm,
        f"FINAL status={final_status} newtonStatus={final_newton.status if final_newton else 'NA'} "
        f"resE={final_metrics['resEuclid']:.6e} phiL2={phi_l2:.6e} "
        f"rhoL2={final_metrics['rhoDesignDiffL2']:.6e} relRho={final_metrics['relRhoDesign']:.6e} "
        f"activeArea={final_metrics['activeArea']:.6e} "
        f"activeJ={final_metrics['activeJaccard']:.6e} "
        f"plateauJ={final_metrics['plateauJaccard']:.6e} c1={c1_phi:.6e} c2={c2_phi:.6e}",
    )
    root_print(comm, f"TIME_TOTAL {elapsed:.3f}")
    root_print(comm, "========== END STRATEGY A CLOSED-LOOP REFIT ==========")

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
            handle.write(f"alphaT1 {params.alpha_t1}\n")
            handle.write(f"alphaT2 {params.alpha_t2}\n")
            handle.write(f"epsTRatio {params.eps_t_ratio}\n")
            handle.write(f"epsPhiRatio {args.eps_ratio}\n")
            handle.write(f"rhoDesignMass {rho_design_mass}\n")
            handle.write(f"rhoDesignMax {rho_design_max}\n")
            handle.write(f"rhoDesignL2 {rho_design_l2}\n")
            handle.write(f"phiTargetMax {phi_target_max}\n")
            handle.write(f"cMax {c_max}\n")
            handle.write(f"minWidth {min_width}\n")
            handle.write(f"pushCenterX {push_center[0]}\n")
            handle.write(f"pushCenterY {push_center[1]}\n")
            handle.write(f"pushScaleFraction {args.push_scale_fraction}\n")
            handle.write(f"pushScaleGrid {args.push_scale_grid}\n")
            handle.write(f"pushRayBins {args.push_ray_bins}\n")
            handle.write(f"refitObjective {args.refit_objective}\n")
            handle.write(f"refitCurrentBandMode {args.refit_current_band_mode}\n")
            handle.write(f"refitBandThreshold {refit_band_threshold}\n")
            handle.write(f"refitBandMissWeight {args.refit_band_miss_weight}\n")
            handle.write(f"refitBandSpillWeight {args.refit_band_spill_weight}\n")
            handle.write(f"refitBandJaccardWeight {args.refit_band_jaccard_weight}\n")
            handle.write(f"refitBandAreaWeight {args.refit_band_area_weight}\n")
            handle.write(f"refitTargetAreaRatio {args.refit_target_area_ratio}\n")
            handle.write(f"refitDensityWeight {args.refit_density_weight}\n")
            handle.write(f"refitWidthGrowthWeight {args.refit_width_growth_weight}\n")
            handle.write(f"refitWidthChangeWeight {args.refit_width_change_weight}\n")
            handle.write(f"refitAnchorWeight {args.refit_anchor_weight}\n")
            handle.write(f"refitDirectionalThresholdGuard {args.refit_directional_threshold_guard}\n")
            handle.write(f"refitPushDirection {args.refit_push_direction}\n")
            handle.write(f"tolRhoRel {args.tol_rho_rel}\n")
            handle.write(f"tolRhoL2 {args.tol_rho_l2}\n")
            handle.write(f"tolCStep {args.tol_c_step}\n")
            handle.write(f"c1Phi {c1_phi}\n")
            handle.write(f"c2Phi {c2_phi}\n")
            handle.write(f"center {m}\n")
            handle.write(f"width {w}\n")
            handle.write(f"epsPhi {eps_phi}\n")
            handle.write(f"resEuclid {final_metrics['resEuclid']}\n")
            handle.write(f"phiDiffL2 {phi_l2}\n")
            handle.write(f"rhoDiffL2 {final_metrics['rhoDesignDiffL2']}\n")
            handle.write(f"massDiff {final_metrics['massRhoMinusDesign']}\n")
            handle.write(f"massRho {final_metrics['massRho']}\n")
            handle.write(f"activeArea {final_metrics['activeArea']}\n")
            handle.write(f"activeDesignArea {final_metrics['activeDesignArea']}\n")
            handle.write(f"activeOverlapArea {final_metrics['activeOverlapArea']}\n")
            handle.write(f"activeJaccard {final_metrics['activeJaccard']}\n")
            handle.write(f"plateauJaccard {final_metrics['plateauJaccard']}\n")
            handle.write(f"relRhoDesign {final_metrics['relRhoDesign']}\n")
            handle.write(f"newtonStatus {final_newton.status if final_newton else 'NA'}\n")
            handle.write(f"newtonIterations {final_newton.iterations if final_newton else 'NA'}\n")
            handle.write(f"finalStatus {final_status}\n")
            handle.write(f"timeTotal {elapsed}\n")
    success = final_status == "CONVERGED_TORSION"
    if args.fail_on_nonconvergence and not success:
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_strategy(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
