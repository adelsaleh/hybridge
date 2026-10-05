#!/usr/bin/env python3
"""Reduced-space optimizer for torsion-initialized logistic-window thresholds.

This runner implements the reduced optimization algorithm described in
``projects/diocotron/docs/torsion``.  It keeps the
torsion target fixed, solves the semilinear state equation for each current
``(c1Phi, c2Phi)``, computes reduced gradients through two sensitivity solves,
and updates the two thresholds with a constrained trust-region step.

The optimized geometric quantities are the soft leakage outside the crisp
torsion band and the soft missing area inside that band:

    L = int_{Omega \\ B_tau} W(phi; c1, c2, eps) dx
    M = int_{B_tau} (1 - W(phi; c1, c2, eps)) dx

where ``W`` is the unscaled logistic activity.  The semilinear PDE still uses
``rho_amp * W`` as its density.

The implementation also reports a practical certified-subband success status.
When the whole torsion band cannot be matched on the selected semilinear
branch, a run can still be useful if the final certified plateau is contained
in the torsion band and has enough area.  In that case the final status is
``CONVERGED_CERTIFIED_SUBBAND`` instead of ``CONVERGED``.
"""

from __future__ import annotations

# Support direct execution as well as python -m from a checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[5]))

import atexit
import argparse
import csv
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, plot as dolfinx_plot
from dolfinx.fem import petsc as fem_petsc

REPO_ROOT = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from projects.diocotron.dolfinx.torsion.equilibrium.window_fit import (  # noqa: E402
    assemble_scalar,
    boundary_bc,
    compute_metrics,
    fit_phi_window_to_torsion_design,
    global_minmax,
    load_or_generate_mesh,
    quadrature_samples_for_fit,
    read_mesh_with_meshio,
    root_print,
    slug_for_path,
    solve_linear_form,
    solver_options,
    update_interpolated,
    window_numpy,
    window_ufl,
)
from projects.diocotron.dolfinx.torsion.equilibrium.closed_loop import (  # noqa: E402
    InPlacePyVistaTorsionPlotter as _LocalPyVistaTorsionPlotter,
)
from projects.diocotron.dolfinx.plotting.mpi_pyvista import MPIPyVistaTorsionPlotter  # noqa: E402
from projects.diocotron.dolfinx.torsion.equilibrium.newton_budget import (  # noqa: E402
    NewtonBudgetDecision,
    decide_newton_budget_extension,
    newton_hard_ceiling,
)
from projects.diocotron.dolfinx.torsion.initialization.source_homotopy import (  # noqa: E402
    AdaptiveSourceHomotopy,
    SourceHomotopySchedule,
)
from projects.diocotron.dolfinx.runtime.terminal_log_capture import TerminalLogCapture  # noqa: E402
from projects.diocotron.paths import resolve_archive_path


DEFAULT_RUN_LOG_ROOT = REPO_ROOT / "projects/diocotron/runs" / "torsion_reduced_optimization_homotopy"

INITIALIZATION_CSV_FIELDS = [
    "record", "runTag", "method", "eval_id", "c1", "c2", "width", "eps",
    "psiHminus1", "residualHminus1", "Lrel", "Mrel", "activityAreaRel",
    "gradPsi1", "gradPsi2", "gradientEvaluated", "feasibleGeometry", "acceptedThresholdStep",
    "lambdaOld", "lambdaTrial", "dlambda", "tangentH1", "predictedResidual",
    "correctedResidual", "newtonIterations", "damping", "backtracks",
    "accepted", "elapsed",
]

NEWTON_CSV_FIELDS = [
    "solve", "phase", "outer_iteration", "homotopy_lambda", "nonlinear_iteration",
    "status", "residual", "damping", "backtracks", "step_norm",
    "ksp_iterations", "ksp_residual", "ksp_time", "assembly_time",
    "line_search_time", "linear_cap", "fallback_used", "elapsed",
    "newton_initial_budget", "newton_current_budget", "newton_hard_ceiling",
    "newton_cap_extension", "newton_cap_reason", "newton_contraction",
]

INEXACT_NEWTON_CSV_FIELDS = [
    "snapshot_id", "policy", "requested_tolerance", "outer_iteration",
    "c1", "c2", "state_residual", "reference_state_residual",
    "state_l2_relative_error", "state_h1_relative_error",
    "sensitivity1_l2_relative_error", "sensitivity1_h1_relative_error",
    "sensitivity2_l2_relative_error", "sensitivity2_h1_relative_error",
    "sensitivity1_defect", "sensitivity2_defect",
    "reduced_gradient_relative_error", "reduced_gradient_angle_degrees",
    "threshold_step_relative_error", "threshold_step_angle_degrees",
    "predicted_reduction_relative_error", "acceptance_decision_agrees",
    "state_solve_time", "sensitivity_solve_time", "elapsed",
]

PHASE_CSV_FIELDS = ["phase", "elapsed", "calls", "detail"]
PHASE_NAMES = (
    "mesh", "torsion", "target_solve", "hminus1_search", "homotopy",
    "window_fit", "mesh_transfer",
    "sensitivities", "reduced_gradients", "trial_corrections",
    "final_projection", "inexact_newton", "checkpointing", "total",
)


class InPlacePyVistaTorsionPlotter(_LocalPyVistaTorsionPlotter):
    """Render complete distributed fields in one rank-zero PyVista window.

    DOLFINx partitions both cells and degrees of freedom under MPI.  The base
    plotter renders only rank zero's local partition because its ``active``
    property excludes every other rank.  This specialization enters plotting
    collectively, gathers owned-cell topology once, merges interface degrees
    of freedom by coordinate, and then gathers only point values on later
    frames.  Interactive and saved plots therefore show the complete domain.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._global_grid = None
        self._global_point_maps: list[np.ndarray] | None = None
        self._global_layout_ready = False
        self._global_space_signature: tuple[int, int, int] | None = None
        self._mpi_live_key: tuple[int, int, int] | None = None
        self._mpi_live_actors: list | None = None
        self._mpi_live_text_actors: list = []

    @property
    def active(self) -> bool:
        """Make plotting collective while retaining the base CLI semantics."""
        return bool(self.args.plot or self.args.save_frames)

    @staticmethod
    def _space_signature(function_space) -> tuple[int, int, int]:
        index_map = function_space.dofmap.index_map
        return (
            int(index_map.size_global),
            int(function_space.dofmap.index_map_bs),
            int(function_space.mesh.topology.index_map(function_space.mesh.topology.dim).size_global),
        )

    def _initialize_global_layout(self, function_space) -> None:
        """Gather owned cells and build a duplicate-free global VTK grid."""
        signature = self._space_signature(function_space)
        if self._global_layout_ready:
            if signature != self._global_space_signature:
                raise ValueError("MPI plot fields changed function-space layout during the run")
            return

        domain = function_space.mesh
        tdim = domain.topology.dim
        owned_cells = np.arange(domain.topology.index_map(tdim).size_local, dtype=np.int32)
        topology, cell_types, geometry = dolfinx_plot.vtk_mesh(
            function_space,
            entities=owned_cells,
        )
        index_map = function_space.dofmap.index_map
        if int(function_space.dofmap.index_map_bs) != 1:
            raise ValueError("MPI plotting currently supports scalar finite-element fields only")
        local_dofs = np.arange(index_map.size_local + index_map.num_ghosts, dtype=np.int32)
        global_dofs = np.asarray(index_map.local_to_global(local_dofs), dtype=np.int64)
        local_payload = (
            np.asarray(topology, dtype=np.int64),
            np.asarray(cell_types, dtype=np.uint8),
            np.asarray(geometry, dtype=np.float64),
            int(function_space.dofmap.dof_layout.num_dofs),
            global_dofs,
        )
        gathered = self.comm.gather(local_payload, root=0)

        if self.comm.rank == 0:
            import pyvista as pv

            remapped_topologies: list[np.ndarray] = []
            type_parts: list[np.ndarray] = []
            point_maps: list[np.ndarray] = []
            global_geometry = np.full(
                (int(function_space.dofmap.index_map.size_global), gathered[0][2].shape[1]),
                np.nan,
                dtype=np.float64,
            )
            for rank_topology, rank_types, rank_geometry, nodes_per_cell, rank_global_dofs in gathered:
                point_maps.append(rank_global_dofs)
                global_geometry[rank_global_dofs, :] = rank_geometry
                if rank_types.size == 0:
                    continue
                rows = rank_topology.reshape(rank_types.size, nodes_per_cell + 1).copy()
                rows[:, 1:] = rank_global_dofs[rows[:, 1:]]
                remapped_topologies.append(rows.reshape(-1))
                type_parts.append(rank_types)

            if not remapped_topologies:
                raise RuntimeError("cannot plot an MPI mesh with no owned cells")
            if not np.all(np.isfinite(global_geometry)):
                raise RuntimeError("MPI plot gather did not receive coordinates for every global degree of freedom")
            global_topology = np.ascontiguousarray(np.concatenate(remapped_topologies)).astype(
                np.int64,
                copy=False,
            )
            global_types = np.ascontiguousarray(np.concatenate(type_parts))
            self._global_grid = pv.UnstructuredGrid(
                global_topology,
                global_types,
                global_geometry,
            )
            self._global_point_maps = point_maps
            if self.comm.size > 1 and int(self.args.verbosity) >= 1:
                root_print(
                    self.comm,
                    f"PLOT_MPI_GRID ranks={self.comm.size} cells={self._global_grid.n_cells} "
                    f"points={self._global_grid.n_points}",
                )

        self._global_layout_ready = True
        self._global_space_signature = signature

    def _gather_global_field_grid(self, fields: list[fem.Function]):
        """Gather scalar point values and update the cached global grid."""
        if not fields:
            return None
        function_space = fields[0].function_space
        signature = self._space_signature(function_space)
        if any(self._space_signature(field.function_space) != signature for field in fields):
            raise ValueError("all MPI plot panels must use the same scalar function-space layout")
        self._initialize_global_layout(function_space)

        local_point_count = int(function_space.tabulate_dof_coordinates().shape[0])
        local_values = np.empty((len(fields), local_point_count), dtype=np.float64)
        for index, field in enumerate(fields):
            values = np.asarray(field.x.array, dtype=np.float64)
            if values.size != local_point_count:
                raise ValueError(
                    f"MPI plotting requires scalar fields; got {values.size} values for "
                    f"{local_point_count} plot points"
                )
            local_values[index, :] = values
        gathered_values = self.comm.gather(local_values, root=0)

        if self.comm.rank != 0:
            return None
        if self._global_grid is None or self._global_point_maps is None:
            raise RuntimeError("rank zero did not initialize the global MPI plot layout")
        global_count = int(self._global_grid.n_points)
        for field_index in range(len(fields)):
            global_values = np.empty(global_count, dtype=np.float64)
            for rank_values, point_map in zip(gathered_values, self._global_point_maps, strict=True):
                global_values[point_map] = rank_values[field_index]
            self._global_grid.point_data[f"panel_{field_index}"] = global_values
        return self._global_grid

    @staticmethod
    def _configure_panel(plotter, grid, scalar_name: str, title: str, nt: int, ndof: int):
        values = grid.point_data[scalar_name]
        actor = plotter.add_mesh(
            grid,
            scalars=scalar_name,
            cmap="viridis",
            clim=InPlacePyVistaTorsionPlotter._safe_clim(values),
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
        text_actor = plotter.add_text(
            f"{title}\nnt={nt} ndof={ndof}",
            position="upper_edge",
            font_size=11,
            shadow=False,
        )
        plotter.enable_parallel_projection()
        plotter.view_xy()
        plotter.show_grid(color=(100, 100, 100, 0.15))
        return actor, text_actor

    def _render_once(
            self,
            grid,
            titles: list[str],
            *,
            save_path: Path | None,
            show: bool,
            window_size: tuple[int, int],
            nt: int,
            ndof: int,
    ) -> None:
        import pyvista as pv

        plotter = pv.Plotter(
            shape=(1, len(titles)),
            window_size=list(window_size),
            off_screen=save_path is not None or self.args.plot_off_screen,
        )
        for index, title in enumerate(titles):
            plotter.subplot(0, index)
            self._configure_panel(plotter, grid, f"panel_{index}", title, nt, ndof)
        if len(titles) > 1:
            plotter.link_views()
        if save_path is not None:
            plotter.screenshot(str(save_path))
        if show and not self.args.plot_off_screen:
            self._close_live_plotter()
            plotter.show(interactive_update=True, auto_close=False)
            self._wait_for_enter(plotter)
        plotter.close()

    def _reset_mpi_live_plotter(self) -> None:
        self._close_live_plotter()
        self._mpi_live_key = None
        self._mpi_live_actors = None
        self._mpi_live_text_actors = []

    def _update_mpi_live_plotter(
            self,
            grid,
            titles: list[str],
            *,
            window_size: tuple[int, int],
            nt: int,
            ndof: int,
    ) -> None:
        import pyvista as pv

        key = (int(grid.n_points), int(grid.n_cells), len(titles))
        if (
                self._live_plotter is None
                or self._mpi_live_actors is None
                or self._mpi_live_key != key
        ):
            self._reset_mpi_live_plotter()
            plotter = pv.Plotter(
                shape=(1, len(titles)),
                window_size=list(window_size),
                off_screen=False,
            )
            self._mpi_live_actors = []
            self._mpi_live_text_actors = []
            for index, title in enumerate(titles):
                plotter.subplot(0, index)
                actor, text_actor = self._configure_panel(
                    plotter,
                    grid,
                    f"panel_{index}",
                    title,
                    nt,
                    ndof,
                )
                self._mpi_live_actors.append(actor)
                self._mpi_live_text_actors.append(text_actor)
            if len(titles) > 1:
                plotter.link_views()
            plotter.show(interactive_update=True, auto_close=False)
            self._live_plotter = plotter
            self._mpi_live_key = key
            return

        plotter = self._live_plotter
        grid.Modified()
        for index, title in enumerate(titles):
            plotter.subplot(0, index)
            actor = self._mpi_live_actors[index]
            values = grid.point_data[f"panel_{index}"]
            clim = self._safe_clim(values)
            try:
                actor.mapper.scalar_range = clim
            except Exception:
                actor.mapper.SetScalarRange(*clim)
            try:
                plotter.remove_actor(self._mpi_live_text_actors[index])
            except Exception:
                pass
            self._mpi_live_text_actors[index] = plotter.add_text(
                f"{title}\nnt={nt} ndof={ndof}",
                position="upper_edge",
                font_size=11,
                shadow=False,
            )
        plotter.update()

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
        """Collect distributed fields once, then save and/or display on rank zero."""
        if not self.active:
            return
        save_frame = bool(self.args.save_frames and save)
        show_plot = bool(self.args.plot and show and not self.args.plot_off_screen)
        if not save_frame and not show_plot:
            return

        grid = self._gather_global_field_grid(fields)
        save_path = self.frame_dir / f"{self.run_tag}_frame_{self.frame_counter:04d}_{token}.png"
        try:
            if self.comm.rank == 0:
                if save_frame:
                    self._render_once(
                        grid,
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
                if show_plot:
                    if getattr(self.args, "plot_mode", "blocking") == "nonblocking":
                        self._update_mpi_live_plotter(
                            grid,
                            titles,
                            window_size=(self.args.plot_window_width, self.args.plot_window_height),
                            nt=nt,
                            ndof=ndof,
                        )
                    else:
                        self._reset_mpi_live_plotter()
                        self._render_once(
                            grid,
                            titles,
                            save_path=None,
                            show=True,
                            window_size=(self.args.plot_window_width, self.args.plot_window_height),
                            nt=nt,
                            ndof=ndof,
                        )
        except Exception as exc:
            if self.comm.rank == 0:
                self._reset_mpi_live_plotter()
                print(
                    f"PLOT_SKIP stage={titles[0] if titles else 'unknown'} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
        finally:
            if save_frame:
                self.frame_counter += 1
            self.comm.barrier()


@dataclass
class TorsionParameters:
    """Fixed parameters defining the torsion-designed target band.

    These values are not optimized by this reduced-space runner.  They define
    the reference torsion solve and the sharp target density
    ``rho_amp * 1_{a1<T<a2}`` used to build ``phi_target``.  The outer
    optimization changes only the semilinear potential thresholds
    ``(c1_phi, c2_phi)``.

    Attributes:
        alpha_t1: Lower torsion threshold as a fraction of ``max(T)``.
        alpha_t2: Upper torsion threshold as a fraction of ``max(T)``.
        eps_t_ratio: Logistic smoothing ratio retained solely for the legacy
            initializer.  Homotopy mode constructs its target from the sharp
            torsion indicator and never uses this value.
        rho_amp: Density amplitude multiplying both the torsion target and the
            semilinear logistic activity window.
        active_threshold: Relative density cutoff used only for diagnostic
            active-set metrics.
        plateau_threshold: Relative density cutoff used only for diagnostic
            plateau-set metrics.
    """

    alpha_t1: float = 0.60
    alpha_t2: float = 0.70
    eps_t_ratio: float = 0.06
    rho_amp: float = 1.0
    active_threshold: float = 0.05
    plateau_threshold: float = 0.90


@dataclass
class NewtonResult:
    """Outcome of one damped Newton solve at fixed thresholds.

    The reduced algorithm uses Newton in two places: to project the current
    iterate onto the selected semilinear branch and to correct the sensitivity
    predictor at a trial threshold pair.  This record stores enough state to
    decide whether that projection is acceptable and to report where time was
    spent.

    Attributes:
        status: Human-readable termination reason.  Only
            ``CONVERGED_RESIDUAL`` means the requested residual tolerance was
            achieved.
        converged: True exactly when ``status`` represents residual
            convergence.
        iterations: Number of Newton iterations performed or attempted.
        residual: Final residual norm in the requested residual norm.
        step_h1: H1 seminorm of the most recent Newton correction.
        alpha: Last accepted line-search damping factor.
        backtracks: Number of backtracking reductions for the last accepted
            step.
        solve_time: Accumulated time spent in linear Newton solves, excluding
            residual/diagnostic assembly.
    """

    status: str
    converged: bool
    iterations: int
    residual: float
    step_h1: float
    alpha: float
    backtracks: int
    solve_time: float


@dataclass
class HomotopyResult:
    """Outcome of the torsion-to-semilinear initialization continuation.

    The homotopy keeps ``(c1,c2)`` fixed and continues the source from the
    sharp torsion target at ``lambda=0`` to the nonlinear logistic source at
    ``lambda=1``.  A first-order tangent predictor is followed by damped
    Newton correction at each accepted continuation value.

    Attributes:
        status: Human-readable continuation termination reason.
        converged: True exactly when ``lambda=1`` was reached with a
            residual-converged Newton correction.
        lambda_final: Last accepted continuation parameter.
        stages: Number of accepted positive continuation steps.
        rejected_steps: Number of failed trial continuation steps.
        total_newton_iterations: Sum of Newton iterations over accepted and
            rejected continuation trials.
        tangent_solve_time: Accumulated time in branch-tangent linear solves.
        newton_solve_time: Accumulated time in Newton linear solves.
        elapsed: Total wall time for the continuation.
        last_newton: Newton result at the last attempted/accepted stage.
    """

    status: str
    converged: bool
    lambda_final: float
    stages: int
    rejected_steps: int
    total_newton_iterations: int
    tangent_solve_time: float
    newton_solve_time: float
    elapsed: float
    last_newton: NewtonResult


@dataclass(frozen=True)
class HomotopyInitializationExceptionReport:
    """Collective description of an exception from source homotopy."""

    status: str
    recoverable: bool
    ranks: tuple[int, ...]
    exception_types: tuple[str, ...]
    messages: tuple[str, ...]

    @property
    def summary(self) -> str:
        details = "; ".join(
            f"rank={rank} {exception_type}: {message}"
            for rank, exception_type, message in zip(
                self.ranks,
                self.exception_types,
                self.messages,
                strict=True,
            )
        )
        return details or "no exception detail"


def is_recoverable_homotopy_solver_exception(error: Exception) -> bool:
    """Recognize numerical PETSc/KSP failures without hiding code defects."""
    if not isinstance(error, RuntimeError):
        return False
    message = " ".join(str(error).split()).lower()
    solver_context = any(token in message for token in (
        "linear solve",
        "ksp",
        "preconditioner",
        "pc setup",
        "petsc reason",
        "diverged_",
    ))
    failure_signal = any(token in message for token in (
        "failed",
        "diverged",
        "reason -",
        "breakdown",
    ))
    return solver_context and failure_signal


def synchronize_homotopy_initialization_exception(
        comm: MPI.Comm,
        local_error: Exception | None,
) -> HomotopyInitializationExceptionReport | None:
    """Make homotopy exception handling a collective rank-consistent decision."""
    local_payload = None
    if local_error is not None:
        local_payload = {
            "rank": int(comm.rank),
            "exception_type": (
                f"{type(local_error).__module__}.{type(local_error).__qualname__}"
            ),
            "message": " ".join(str(local_error).split())[:1000],
            "recoverable": is_recoverable_homotopy_solver_exception(local_error),
        }
    gathered = comm.allgather(local_payload)
    failures = tuple(item for item in gathered if item is not None)
    if not failures:
        return None
    recoverable = all(bool(item["recoverable"]) for item in failures)
    return HomotopyInitializationExceptionReport(
        status=(
            "FAIL_SOLVER_EXCEPTION"
            if recoverable
            else "FAIL_UNRECOVERABLE_EXCEPTION"
        ),
        recoverable=recoverable,
        ranks=tuple(int(item["rank"]) for item in failures),
        exception_types=tuple(str(item["exception_type"]) for item in failures),
        messages=tuple(str(item["message"]) for item in failures),
    )


def require_window_fit_homotopy_exception_fallback(
        report: HomotopyInitializationExceptionReport,
        *,
        init_fallback: str,
        local_error: Exception | None,
) -> None:
    """Return only for an explicitly enabled, collectively recoverable rescue."""
    if report.recoverable and str(init_fallback) == "window-fit":
        return
    if local_error is not None:
        raise local_error
    reason = (
        "window-fit fallback is disabled"
        if report.recoverable
        else "exception is not a recoverable PETSc/solver failure"
    )
    raise RuntimeError(
        f"homotopy initialization exception on peer rank ({reason}): "
        f"{report.summary}"
    )


@dataclass
class FrozenThresholdEvaluation:
    """Frozen-state threshold objective and geometric guardrails.

    ``psi`` is assembled as ``0.5*r.T*K^{-1}*r`` at the fixed sharp-target
    potential.  The two gradient entries are exact total derivatives,
    including the epsilon-width chain rule in relative-epsilon mode.
    """

    c1: float
    c2: float
    eps_phi: float
    psi: float
    residual_dual: float
    leakage: float
    missing: float
    leakage_rel: float
    missing_rel: float
    activity_area: float
    activity_area_rel: float
    grad_psi: np.ndarray
    projected_grad_norm: float | None = None
    evaluation_time: float | None = None
    gradient_evaluated: bool = True


@dataclass(frozen=True)
class InexactNewtonSnapshotDiagnostics:
    """Errors induced by using an off-manifold Newton state.

    The record intentionally contains only scalar, serialization-friendly
    values.  A numerical-study replay can compute field norms with its chosen
    quadrature and use :func:`compare_inexact_newton_snapshot` for consistent
    reduced-gradient, threshold-step, and acceptance comparisons.
    """

    state_l2_relative_error: float
    state_h1_relative_error: float
    sensitivity1_l2_relative_error: float
    sensitivity1_h1_relative_error: float
    sensitivity2_l2_relative_error: float
    sensitivity2_h1_relative_error: float
    sensitivity1_defect: float
    sensitivity2_defect: float
    reduced_gradient_relative_error: float
    reduced_gradient_angle_degrees: float
    threshold_step_relative_error: float
    threshold_step_angle_degrees: float
    predicted_reduction_relative_error: float
    acceptance_decision_agrees: bool

    def as_dict(self) -> dict[str, float | bool]:
        """Return a stable mapping suitable for CSV or JSON output."""
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


@dataclass
class InexactNewtonReplaySnapshot:
    """Predictor and accepted-branch reference for stopping-policy replay.

    ``predictor_state`` is deliberately off the nonlinear solution manifold and
    is the common starting point for every inexact policy.  For an accepted
    threshold trial, ``reference_seed_state`` is the corrected state that the
    production run accepted at the *same* threshold pair.  Starting the tight
    reference solve there avoids changing branches (or failing line search)
    merely because a deliberately crude sensitivity predictor is far from the
    solution manifold.
    """

    snapshot_id: str
    predictor_state: np.ndarray
    reference_seed_state: np.ndarray
    c1: float
    c2: float
    trust_radius: float
    discrepancy_rel: float
    outer_iteration: int


@dataclass
class BandMetrics:
    """Soft and certified geometric discrepancy values.

    The optimization objective is expressed with the smooth logistic activity
    ``W(phi; c1, c2, eps)`` so that gradients exist.  The certified values use
    the stricter interior band
    ``c1 + kappa*eps <= phi <= c2 - kappa*eps`` and are intended for final
    geometric checks rather than differentiation.

    Attributes:
        leakage: Soft activity area outside the crisp torsion band.
        missing: Soft unfilled area inside the crisp torsion band.
        leakage_rel: ``leakage / target_area``.
        missing_rel: ``missing / target_area``.
        target_area: Area of the crisp torsion band ``B_tau``.
        activity_area: Integral of the soft activity over the whole domain.
        certified_area: Area of the interior certified potential band.
        certified_leakage: Certified active area outside ``B_tau``.
        certified_missing: Missing certified area inside ``B_tau``.
    """

    leakage: float
    missing: float
    leakage_rel: float
    missing_rel: float
    target_area: float
    activity_area: float
    certified_area: float
    certified_leakage: float
    certified_missing: float


@dataclass
class ReducedGradient:
    """Reduced gradients and sensitivity solve diagnostics.

    The reduced derivatives use the chain rule
    ``d D_hat / dc = D_c + S.T D_U``.  Here ``S`` is obtained by solving the
    two sensitivity equations with the Newton matrix.  This record separates
    the direct terms from the final reduced gradients so the output can expose
    whether a search direction is dominated by threshold motion itself or by
    the equilibrium response.

    Attributes:
        grad_l: Reduced gradient of soft leakage with respect to ``(c1,c2)``.
        grad_m: Reduced gradient of missing area with respect to ``(c1,c2)``.
        direct_l: Explicit leakage derivative at fixed state.
        direct_m: Explicit missing-area derivative at fixed state.
        solve_iterations: PETSc iteration counts for the two sensitivity
            solves.
        solve_residuals: PETSc residual norms for the two sensitivity solves.
        solve_time: Total matrix-assembly plus RHS-assembly plus solve time for
            the two sensitivity equations.
        matrix_assembly_time: Time to assemble the shared Newton matrix.
        rhs_assembly_time: Time to assemble the two parameter-derivative RHS
            vectors.
        linear_solve_time: Time spent in the two PETSc solves.
        gradient_assembly_time: Time to assemble functional derivative vectors
            and combine direct/sensitivity terms.
        total_time: Wall time for the full reduced-gradient computation.
    """

    grad_l: np.ndarray
    grad_m: np.ndarray
    direct_l: np.ndarray
    direct_m: np.ndarray
    solve_iterations: tuple[int, int]
    solve_residuals: tuple[float, float]
    solve_time: float
    matrix_assembly_time: float
    rhs_assembly_time: float
    linear_solve_time: float
    gradient_assembly_time: float
    total_time: float


@dataclass
class ParameterStep:
    """Trust-region step in the two threshold variables.

    The outer update solves a small convex model problem in ``(dc1, dc2)``.
    When leakage is infeasible, the model prioritizes leakage reduction.  Once
    leakage is within the admissible envelope, the model minimizes missing area
    while enforcing the linearized leakage constraint.

    Attributes:
        dc: Proposed threshold increment ``[dc1, dc2]``.
        objective_name: ``"leakage"`` or ``"missing"``, depending on the
            filter state at the current iterate.
        predicted_reduction: Positive model-predicted decrease in the active
            objective.
        model_change: Raw quadratic model value at ``dc``.
        hessian_scale: Scalar positive Hessian approximation used in the
            two-dimensional model.
        step_norm: Euclidean norm of ``dc``.
        hit_boundary: True when the trust-region radius is active.
        status: Solver status for the tiny QP enumerator.
    """

    dc: np.ndarray
    objective_name: str
    predicted_reduction: float
    model_change: float
    hessian_scale: float
    step_norm: float
    hit_boundary: bool
    status: str


@dataclass
class InitialWindowCandidate:
    """One automatically generated initial potential-window candidate.

    The reduced optimizer is sensitive to the branch selected by the initial
    thresholds.  The existing L2 density fit is retained, but this record lets
    the script compare it against geometric candidates before starting the
    nonlinear closed-loop phase.

    Attributes:
        name: Stable label identifying how the candidate was generated.
        c1: Lower potential threshold.
        c2: Upper potential threshold.
        eps_phi: Smoothing width associated with ``(c1,c2)``.
        leakage_rel: Sampled soft leakage divided by the torsion target area.
        missing_rel: Sampled missing target area divided by the torsion target
            area.
        activity_area: Sampled total soft activity area.
        area_rel: ``activity_area / target_area``.
        active_jaccard: Sampled soft overlap Jaccard score with the torsion
            design activity.
        l2_rel: Relative L2 density mismatch
            ``||rho_amp*W(phi_target)-rho_design||/||rho_design||``.
        score: Fixed initializer score used to rank candidates.  It emphasizes
            target coverage, leakage, and total area mismatch.
    """

    name: str
    c1: float
    c2: float
    eps_phi: float
    leakage_rel: float
    missing_rel: float
    activity_area: float
    area_rel: float
    active_jaccard: float
    l2_rel: float
    score: float


@dataclass
class ProjectedInitialCandidate:
    """Initial window candidate after fixed-threshold Newton projection.

    The algebraic fit on ``phi_target`` is only a branch-selection hint.  The
    optimizer actually starts from a state that satisfies the semilinear PDE at
    fixed ``(c1,c2)``.  This record stores the post-projection state and its
    geometric diagnostics so the initializer can choose the branch that remains
    closest to the torsion-designed band after Newton, instead of choosing only
    from the pre-projection density fit.

    Attributes:
        base: Pre-projection candidate and its sampled ``phi_target`` metrics.
        newton: Newton projection result for this fixed threshold pair.
        metrics: Soft leakage/missing metrics measured on the projected state.
        diagnostics: Legacy density and active-set diagnostics measured on the
            projected state.
        score: Projected initializer score.  Lower is better.
        state: Copy of the projected finite-element state coefficients.
        density: Copy of the projected window-density coefficients.
        homotopy: Continuation diagnostics when the homotopy initializer is
            used; ``None`` for the legacy direct-Newton path.
    """

    base: InitialWindowCandidate
    newton: NewtonResult
    metrics: BandMetrics
    diagnostics: dict[str, float]
    score: float
    state: np.ndarray
    density: np.ndarray
    homotopy: HomotopyResult | None = None


def _relative_vector_error(approximate: np.ndarray, reference: np.ndarray) -> float:
    """Return a scale-safe Euclidean relative error."""
    approximate = np.asarray(approximate, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    if approximate.shape != reference.shape:
        raise ValueError("approximate and reference vectors must have the same shape")
    denominator = max(float(np.linalg.norm(reference)), 1.0e-30)
    return float(np.linalg.norm(approximate - reference)) / denominator


def _vector_angle_degrees(approximate: np.ndarray, reference: np.ndarray) -> float:
    """Return the unsigned angle between two vectors in degrees."""
    approximate = np.asarray(approximate, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    if approximate.shape != reference.shape:
        raise ValueError("approximate and reference vectors must have the same shape")
    norm_approximate = float(np.linalg.norm(approximate))
    norm_reference = float(np.linalg.norm(reference))
    if norm_approximate <= 1.0e-30 and norm_reference <= 1.0e-30:
        return 0.0
    if norm_approximate <= 1.0e-30 or norm_reference <= 1.0e-30:
        return 90.0
    cosine = float(np.dot(approximate, reference)) / (norm_approximate * norm_reference)
    angular_roundoff = 16.0 * np.finfo(np.float64).eps
    if cosine >= 1.0 - angular_roundoff:
        return 0.0
    if cosine <= -1.0 + angular_roundoff:
        return 180.0
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def model_reduction_is_sufficient(
        predicted_reduction: float,
        *,
        target_area: float,
        sufficient_decrease_fraction: float,
) -> bool:
    """Return whether a model decrease clears the area-scaled filter floor."""
    predicted_reduction = float(predicted_reduction)
    target_area = float(target_area)
    sufficient_decrease_fraction = float(sufficient_decrease_fraction)
    return bool(
        math.isfinite(predicted_reduction)
        and predicted_reduction > sufficient_decrease_fraction * target_area
    )


def compare_inexact_newton_snapshot(
        *,
        state_l2_relative_error: float,
        state_h1_relative_error: float,
        sensitivity1_l2_relative_error: float,
        sensitivity1_h1_relative_error: float,
        sensitivity2_l2_relative_error: float,
        sensitivity2_h1_relative_error: float,
        sensitivity1_defect: float,
        sensitivity2_defect: float,
        approximate_reduced_gradient: np.ndarray,
        reference_reduced_gradient: np.ndarray,
        approximate_threshold_step: np.ndarray,
        reference_threshold_step: np.ndarray,
        approximate_predicted_reduction: float,
        reference_predicted_reduction: float,
        approximate_accepted: bool,
        reference_accepted: bool,
) -> InexactNewtonSnapshotDiagnostics:
    """Build the common off-manifold Newton/sensitivity error record.

    The state and sensitivity norms, as well as ``||J s_i + F_c_i||``, are
    accepted as scalars because their assembly depends on the replay mesh and
    quadrature.  The algebraic comparisons are performed here so every study
    variant uses identical zero handling and angle conventions.
    """
    scalar_values = (
        state_l2_relative_error,
        state_h1_relative_error,
        sensitivity1_l2_relative_error,
        sensitivity1_h1_relative_error,
        sensitivity2_l2_relative_error,
        sensitivity2_h1_relative_error,
        sensitivity1_defect,
        sensitivity2_defect,
        approximate_predicted_reduction,
        reference_predicted_reduction,
    )
    if not all(math.isfinite(float(value)) for value in scalar_values):
        raise ValueError("inexact-Newton snapshot inputs must be finite")
    predicted_reduction_relative_error = abs(
        float(approximate_predicted_reduction) - float(reference_predicted_reduction)
    ) / max(abs(float(reference_predicted_reduction)), 1.0e-30)
    return InexactNewtonSnapshotDiagnostics(
        state_l2_relative_error=float(state_l2_relative_error),
        state_h1_relative_error=float(state_h1_relative_error),
        sensitivity1_l2_relative_error=float(sensitivity1_l2_relative_error),
        sensitivity1_h1_relative_error=float(sensitivity1_h1_relative_error),
        sensitivity2_l2_relative_error=float(sensitivity2_l2_relative_error),
        sensitivity2_h1_relative_error=float(sensitivity2_h1_relative_error),
        sensitivity1_defect=float(sensitivity1_defect),
        sensitivity2_defect=float(sensitivity2_defect),
        reduced_gradient_relative_error=_relative_vector_error(
            approximate_reduced_gradient, reference_reduced_gradient
        ),
        reduced_gradient_angle_degrees=_vector_angle_degrees(
            approximate_reduced_gradient, reference_reduced_gradient
        ),
        threshold_step_relative_error=_relative_vector_error(
            approximate_threshold_step, reference_threshold_step
        ),
        threshold_step_angle_degrees=_vector_angle_degrees(
            approximate_threshold_step, reference_threshold_step
        ),
        predicted_reduction_relative_error=predicted_reduction_relative_error,
        acceptance_decision_agrees=bool(approximate_accepted) == bool(reference_accepted),
    )


def write_inexact_newton_snapshot_row(
        writer: csv.DictWriter,
        diagnostics: InexactNewtonSnapshotDiagnostics,
        *,
        snapshot_id: str,
        policy: str,
        requested_tolerance: float,
        outer_iteration: int,
        c1: float,
        c2: float,
        state_residual: float,
        reference_state_residual: float,
        state_solve_time: float,
        sensitivity_solve_time: float,
        elapsed: float,
) -> None:
    """Write one standardized row for the inexact-Newton replay study."""
    row: dict[str, object] = {
        "snapshot_id": str(snapshot_id),
        "policy": str(policy),
        "requested_tolerance": float(requested_tolerance),
        "outer_iteration": int(outer_iteration),
        "c1": float(c1),
        "c2": float(c2),
        "state_residual": float(state_residual),
        "reference_state_residual": float(reference_state_residual),
        "state_solve_time": float(state_solve_time),
        "sensitivity_solve_time": float(sensitivity_solve_time),
        "elapsed": float(elapsed),
    }
    row.update(diagnostics.as_dict())
    writer.writerow(row)


def make_run_dir(args: argparse.Namespace) -> Path:
    """Create a unique run directory for logs, plots, and summaries.

    If the user supplies ``--run-dir`` and that path already exists, the
    function appends a timestamp/suffix rather than overwriting prior results.
    This mirrors the other torsion-initialized Newton runners and keeps every
    optimization run reproducible from its generated CSV and summary files.

    Args:
        args: Parsed command-line namespace containing ``run_dir`` and
            ``run_tag``.

    Returns:
        Path to a newly created directory that is safe to write into.
    """
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


def write_equilibrium_checkpoint(
        path: Path,
        *,
        phi: fem.Function,
        rho: fem.Function,
        metadata: dict,
) -> None:
    """Write a rank-count-independent scalar Lagrange equilibrium checkpoint.

    PETSc global degree-of-freedom numbering changes when the same mesh is
    repartitioned with a different MPI size.  Saving owned nodal coordinates
    alongside the field values makes the checkpoint portable across those
    partitions while retaining the finite-element nodal state exactly.
    """
    V = phi.function_space
    comm = V.mesh.comm
    if rho.function_space is not V:
        raise ValueError("equilibrium checkpoint fields must share one function space")
    if V.dofmap.index_map_bs != 1:
        raise ValueError("equilibrium checkpoint currently supports scalar spaces only")
    owned = int(V.dofmap.index_map.size_local)
    coordinates = np.asarray(V.tabulate_dof_coordinates()[:owned], dtype=np.float64)
    local_payload = (
        coordinates,
        np.asarray(np.real(phi.x.array[:owned]), dtype=np.float64).copy(),
        np.asarray(np.real(rho.x.array[:owned]), dtype=np.float64).copy(),
    )
    gathered = comm.gather(local_payload, root=0)
    if comm.rank == 0:
        all_coordinates = np.concatenate([part[0] for part in gathered], axis=0)
        all_phi = np.concatenate([part[1] for part in gathered])
        all_rho = np.concatenate([part[2] for part in gathered])
        rounded = np.round(all_coordinates, decimals=13)
        if np.unique(rounded, axis=0).shape[0] != rounded.shape[0]:
            raise RuntimeError("duplicate nodal coordinates in equilibrium checkpoint")
        keys = tuple(all_coordinates[:, axis] for axis in range(all_coordinates.shape[1] - 1, -1, -1))
        order = np.lexsort(keys)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            coordinates=all_coordinates[order],
            phi=all_phi[order],
            rho=all_rho[order],
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    comm.barrier()


INITIAL_EQUILIBRIUM_SUCCESS_STATUSES = frozenset(
    {"", "OK", "CONVERGED", "CONVERGED_CERTIFIED_SUBBAND"}
)


def initial_equilibrium_status_is_acceptable(
        final_status: object,
        final_residual: object,
        *,
        max_residual: float | None,
) -> bool:
    """Return whether checkpoint termination metadata is safe to replay.

    Successful outer statuses remain acceptable independently of residual
    metadata. Other statuses require an explicitly requested residual bound
    and a finite saved residual satisfying that bound.
    """
    status = str(final_status).strip().upper()
    if status in INITIAL_EQUILIBRIUM_SUCCESS_STATUSES:
        return True
    try:
        saved_residual = float(final_residual)
    except (TypeError, ValueError):
        return False
    return (
        max_residual is not None
        and math.isfinite(saved_residual)
        and saved_residual <= float(max_residual)
    )


def load_initial_equilibrium_state(
        path: Path,
        *,
        target: fem.Function,
        alpha_t1: float,
        alpha_t2: float,
        rho_amp: float,
        max_residual: float | None = None,
) -> tuple[dict, str, float]:
    """Load a portable equilibrium into ``target`` across MPI partitions.

    Same-space checkpoints use a direct coordinate lookup.  When the target
    mesh/order differs, every rank reconstructs the complete source function
    on ``MPI.COMM_SELF`` and DOLFINx interpolates it nonmatching onto that
    rank's owned target cells.  The caller must still run a target-mesh Newton
    correction before treating the state as an equilibrium.
    """
    comm = target.function_space.mesh.comm
    payload = None
    if comm.rank == 0:
        try:
            with np.load(resolve_archive_path(path), allow_pickle=False) as checkpoint:
                metadata = json.loads(str(checkpoint["metadata"].item()))
                format_name = str(metadata.get("format", ""))
                if format_name == "hybridge_equilibrium_v1":
                    coordinates = np.asarray(checkpoint["coordinates"], dtype=np.float64).copy()
                    phi_values = np.asarray(checkpoint["phi"], dtype=np.float64).copy()
                elif format_name == "hybridge_equilibrium_v2":
                    names = [str(value) for value in np.asarray(checkpoint["field_names"]).tolist()]
                    if "phi" not in names:
                        raise ValueError("v2 equilibrium checkpoint has no 'phi' field")
                    coordinates = np.asarray(checkpoint["coordinates"], dtype=np.float64).copy()
                    nodal_values = np.asarray(checkpoint["nodal_values"], dtype=np.float64)
                    phi_values = np.asarray(nodal_values[names.index("phi")], dtype=np.float64).copy()
                else:
                    raise ValueError(f"unsupported equilibrium checkpoint format {format_name!r}")
            if coordinates.ndim != 2 or coordinates.shape[0] != phi_values.size:
                raise ValueError("equilibrium coordinate/value arrays have incompatible shapes")
            if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(phi_values)):
                raise ValueError("equilibrium checkpoint contains non-finite values")
            payload = ("", coordinates, phi_values, metadata)
        except Exception as exc:
            payload = (f"{type(exc).__name__}: {exc}", None, None, None)
    error, coordinates, phi_values, metadata = comm.bcast(payload, root=0)
    if error:
        raise RuntimeError(f"failed to read initial equilibrium {path}: {error}")

    status = str(metadata.get("final_status", "")).strip().upper()
    if not initial_equilibrium_status_is_acceptable(
            status,
            metadata.get("final_residual", math.inf),
            max_residual=max_residual,
    ):
        raise ValueError(f"initial equilibrium has unsuccessful final_status={status!r}")
    for key, expected in (
            ("alpha_t1", alpha_t1),
            ("alpha_t2", alpha_t2),
            ("rho_amp", rho_amp),
    ):
        if key in metadata and not math.isclose(
                float(metadata[key]), float(expected), rel_tol=1.0e-12, abs_tol=1.0e-14
        ):
            raise ValueError(
                f"initial equilibrium {key}={metadata[key]} does not match current {expected}"
            )
    if "c1_phi" not in metadata or "c2_phi" not in metadata:
        raise ValueError("initial equilibrium metadata is missing c1_phi/c2_phi")

    target_coordinates = np.asarray(
        target.function_space.tabulate_dof_coordinates(), dtype=np.float64
    )[:, :2]
    saved_coordinates = np.asarray(coordinates, dtype=np.float64)[:, :2]
    transfer_mode = "coordinate_match"
    maximum_error = math.inf
    if int(metadata.get("num_dofs", len(phi_values))) == int(
            target.function_space.dofmap.index_map.size_global
    ):
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(saved_coordinates).query(target_coordinates, k=1)
        maximum_error = float(np.max(distances, initial=0.0))
        if maximum_error <= 1.0e-9:
            target.x.array[:] = np.asarray(phi_values, dtype=np.float64)[indices]
            target.x.scatter_forward()
            return dict(metadata), transfer_mode, maximum_error

    source_mesh_path = resolve_archive_path(str(metadata.get("mesh_path", "")))
    if not source_mesh_path.is_absolute():
        source_mesh_path = path.parent / source_mesh_path
    source_mesh_path = source_mesh_path.resolve()
    if not source_mesh_path.is_file():
        raise FileNotFoundError(
            f"cross-mesh equilibrium transfer requires recorded source mesh {source_mesh_path}"
        )

    from scipy.spatial import cKDTree

    source_domain = read_mesh_with_meshio(source_mesh_path, MPI.COMM_SELF)
    source_order = int(metadata.get("order", metadata.get("source_order", 0)))
    if source_order < 1:
        raise ValueError("initial equilibrium metadata has no valid finite-element order")
    source_space = fem.functionspace(source_domain, ("Lagrange", source_order))
    source = fem.Function(source_space, name="initial_equilibrium_phi")
    source_coordinates = np.asarray(source_space.tabulate_dof_coordinates(), dtype=np.float64)[:, :2]
    source_distances, source_indices = cKDTree(saved_coordinates).query(source_coordinates, k=1)
    source_error = float(np.max(source_distances, initial=0.0))
    if source_error > 1.0e-9:
        raise ValueError(
            f"source mesh does not reproduce checkpoint nodes (maximum error {source_error:.3e})"
        )
    source.x.array[:] = np.asarray(phi_values, dtype=np.float64)[source_indices]
    source.x.scatter_forward()

    target_domain = target.function_space.mesh
    target_min_local = np.min(target_domain.geometry.x[:, :2], axis=0)
    target_max_local = np.max(target_domain.geometry.x[:, :2], axis=0)
    target_min = np.empty_like(target_min_local)
    target_max = np.empty_like(target_max_local)
    comm.Allreduce(target_min_local, target_min, op=MPI.MIN)
    comm.Allreduce(target_max_local, target_max, op=MPI.MAX)
    target_bounds = np.vstack((target_min, target_max))
    source_bounds = np.vstack((
        np.min(source_domain.geometry.x[:, :2], axis=0),
        np.max(source_domain.geometry.x[:, :2], axis=0),
    ))
    bounds_error = float(np.max(np.abs(target_bounds - source_bounds)))
    bounds_scale = max(float(np.max(np.ptp(source_bounds, axis=0))), 1.0)
    if bounds_error > 2.0e-2 * bounds_scale:
        raise ValueError(
            "initial equilibrium and target meshes have incompatible bounding boxes "
            f"(maximum difference {bounds_error:.3e})"
        )

    tdim = target_domain.topology.dim
    target_owned_cells = np.arange(
        target_domain.topology.index_map(tdim).size_local, dtype=np.int32
    )
    target_geometry_dofs = np.asarray(target_domain.geometry.dofmaps[0], dtype=np.int64)[
        target_owned_cells, :3
    ]
    target_vertices = np.asarray(target_domain.geometry.x, dtype=np.float64)[
        target_geometry_dofs, :2
    ]
    target_area_local = 0.5 * float(np.sum(np.abs(
        (target_vertices[:, 1, 0] - target_vertices[:, 0, 0])
        * (target_vertices[:, 2, 1] - target_vertices[:, 0, 1])
        - (target_vertices[:, 1, 1] - target_vertices[:, 0, 1])
        * (target_vertices[:, 2, 0] - target_vertices[:, 0, 0])
    )))
    target_area = float(comm.allreduce(target_area_local, op=MPI.SUM))
    source_cells = np.arange(
        source_domain.topology.index_map(source_domain.topology.dim).size_local,
        dtype=np.int32,
    )
    source_geometry_dofs = np.asarray(source_domain.geometry.dofmaps[0], dtype=np.int64)[
        source_cells, :3
    ]
    source_vertices = np.asarray(source_domain.geometry.x, dtype=np.float64)[
        source_geometry_dofs, :2
    ]
    source_area = 0.5 * float(np.sum(np.abs(
        (source_vertices[:, 1, 0] - source_vertices[:, 0, 0])
        * (source_vertices[:, 2, 1] - source_vertices[:, 0, 1])
        - (source_vertices[:, 1, 1] - source_vertices[:, 0, 1])
        * (source_vertices[:, 2, 0] - source_vertices[:, 0, 0])
    )))
    area_relative_difference = abs(target_area - source_area) / max(source_area, 1.0e-30)
    if area_relative_difference > 5.0e-2:
        raise ValueError(
            "initial equilibrium and target meshes have incompatible areas "
            f"(relative difference {area_relative_difference:.3e})"
        )

    # Independently generated linear boundary meshes can differ in a very
    # thin shell even when they approximate the same CAD curve.  Evaluate the
    # complete source field at every target node and use the homogeneous
    # Dirichlet value for the small set lying just outside the source mesh.
    from dolfinx import geometry as dolfinx_geometry

    evaluation_points = np.zeros((len(target_coordinates), 3), dtype=np.float64)
    evaluation_points[:, :2] = target_coordinates
    tree = dolfinx_geometry.bb_tree(source_domain, source_domain.topology.dim)
    candidates = dolfinx_geometry.compute_collisions_points(tree, evaluation_points)
    collisions = dolfinx_geometry.compute_colliding_cells(
        source_domain, candidates, evaluation_points
    )
    evaluation_cells = np.full(len(evaluation_points), -1, dtype=np.int32)
    for point_index in range(len(evaluation_points)):
        links = collisions.links(point_index)
        if len(links):
            evaluation_cells[point_index] = int(links[0])
    valid = evaluation_cells >= 0
    unmatched_local = int(np.count_nonzero(~valid))
    point_count_local = int(len(valid))
    unmatched = int(comm.allreduce(unmatched_local, op=MPI.SUM))
    point_count = int(comm.allreduce(point_count_local, op=MPI.SUM))
    unmatched_fraction = unmatched / max(point_count, 1)
    if unmatched_fraction > 2.5e-1:
        raise ValueError(
            "cross-mesh equilibrium transfer left too many target nodes outside "
            f"the source mesh ({unmatched_fraction:.3%})"
        )
    target.x.array[:] = 0.0
    if np.any(valid):
        evaluated = np.asarray(
            source.eval(evaluation_points[valid], evaluation_cells[valid]),
            dtype=np.float64,
        ).reshape(-1)
        target.x.array[valid] = evaluated
    target.x.scatter_forward()
    if not np.all(np.isfinite(target.x.array)):
        raise RuntimeError("cross-mesh equilibrium interpolation produced non-finite values")
    transfer_mode = "nonmatching_interpolation"
    maximum_error = source_error
    metadata = dict(metadata)
    metadata["transfer_unmatched_fraction"] = float(unmatched_fraction)
    metadata["transfer_area_relative_difference"] = float(area_relative_difference)
    metadata["transfer_bounds_error"] = float(bounds_error)
    return metadata, transfer_mode, maximum_error


def gather_scalar_fields(fields: list[fem.Function]) -> tuple[np.ndarray, np.ndarray] | None:
    """Gather scalar nodal fields in deterministic coordinate order."""
    if not fields:
        raise ValueError("at least one field is required")
    V = fields[0].function_space
    if any(field.function_space is not V for field in fields):
        raise ValueError("trajectory fields must share one function space")
    comm = V.mesh.comm
    owned = int(V.dofmap.index_map.size_local)
    coordinates = np.asarray(V.tabulate_dof_coordinates()[:owned], dtype=np.float64)
    values = np.vstack([
        np.asarray(np.real(field.x.array[:owned]), dtype=np.float64)
        for field in fields
    ])
    gathered = comm.gather((coordinates, values), root=0)
    if comm.rank != 0:
        return None
    all_coordinates = np.concatenate([part[0] for part in gathered], axis=0)
    all_values = np.concatenate([part[1] for part in gathered], axis=1)
    keys = tuple(all_coordinates[:, axis] for axis in range(all_coordinates.shape[1] - 1, -1, -1))
    order = np.lexsort(keys)
    return all_coordinates[order], all_values[:, order]


class TrajectoryRecorder:
    """Collect selected optimizer states and write one portable NPZ archive.

    Every selected state is gathered while MPI is known to be healthy. Rank
    zero therefore owns enough information to write a useful partial archive
    from an ``atexit`` hook if a later solver or checkpoint operation raises.
    The emergency hook deliberately performs no MPI communication: collective
    gathering at interpreter shutdown can deadlock after only one rank fails.
    """

    def __init__(
            self,
            *,
            enabled: bool,
            every: int,
            output: Path,
            mesh_path: Path,
            comm: MPI.Comm,
    ) -> None:
        self.enabled = bool(enabled)
        self.every = int(every)
        self.output = Path(output)
        self.mesh_path = Path(mesh_path)
        self.comm = comm
        self.coordinates: np.ndarray | None = None
        self.fixed: dict[str, np.ndarray] = {}
        self.states_phi: list[np.ndarray] = []
        self.states_rho: list[np.ndarray] = []
        self.states: list[dict[str, object]] = []
        self.context: dict[str, object] = {}
        self._finalized = False
        if self.enabled and self.comm.rank == 0:
            atexit.register(self._flush_at_exit)

    def set_context(self, metadata: dict[str, object]) -> None:
        """Store run metadata that is also available to emergency flushes."""
        if self.enabled:
            self.context.update(metadata)

    def set_fixed(self, *, torsion: fem.Function, target_band: fem.Function,
                  target_density: fem.Function, target_potential: fem.Function) -> None:
        if not self.enabled:
            return
        gathered = gather_scalar_fields([torsion, target_band, target_density, target_potential])
        if self.comm.rank == 0:
            if gathered is None:
                raise RuntimeError("rank zero did not receive fixed trajectory fields")
            self.coordinates, values = gathered
            self.fixed = {
                "torsion": values[0],
                "target_band": values[1],
                "target_density": values[2],
                "target_potential": values[3],
            }

    def add(
            self,
            *,
            stage: str,
            phi: fem.Function,
            rho: fem.Function,
            c1: float,
            c2: float,
            eps_phi: float,
            homotopy_lambda: float = 1.0,
            outer_iteration: int = -1,
            sequence: int = 0,
            force: bool = False,
    ) -> None:
        if not self.enabled or (not force and sequence % self.every):
            return
        gathered = gather_scalar_fields([phi, rho])
        if self.comm.rank == 0:
            if gathered is None:
                raise RuntimeError("rank zero did not receive trajectory state")
            coordinates, values = gathered
            if self.coordinates is None:
                self.coordinates = coordinates
            elif not np.allclose(coordinates, self.coordinates, rtol=0.0, atol=1.0e-13):
                raise RuntimeError("trajectory coordinate ordering changed during the run")
            self.states_phi.append(values[0].copy())
            self.states_rho.append(values[1].copy())
            self.states.append({
                "stage": str(stage),
                "c1": float(c1),
                "c2": float(c2),
                "width": float(c2 - c1),
                "eps_phi": float(eps_phi),
                "homotopy_lambda": float(homotopy_lambda),
                "outer_iteration": int(outer_iteration),
            })

    def _ensure_design_only_state(self) -> None:
        """Create a renderable state when failure predates the first iterate."""
        if self.comm.rank != 0 or self.states:
            return
        required = {"target_potential", "target_density"}
        if self.coordinates is None or not required.issubset(self.fixed):
            return
        target_potential = np.asarray(self.fixed["target_potential"], dtype=np.float64)
        target_density = np.asarray(self.fixed["target_density"], dtype=np.float64)
        self.states_phi.append(target_potential.copy())
        self.states_rho.append(target_density.copy())
        self.states.append({
            "stage": "design_only_failure",
            "c1": float("nan"),
            "c2": float("nan"),
            "width": float("nan"),
            "eps_phi": float("nan"),
            "homotopy_lambda": 0.0,
            "outer_iteration": -1,
        })

    def _write_rank_zero(self, metadata: dict[str, object]) -> None:
        """Write already-gathered data atomically; callable without MPI."""
        if self.comm.rank != 0:
            return
        import meshio

        self._ensure_design_only_state()
        if self.coordinates is None or not self.states:
            raise RuntimeError("cannot write a trajectory before fixed fields are gathered")
        required = {"torsion", "target_band", "target_density", "target_potential"}
        missing = sorted(required.difference(self.fixed))
        if missing:
            raise RuntimeError(f"trajectory archive is missing fixed fields: {missing}")
        mesh_data = meshio.read(self.mesh_path)
        triangle_blocks = [
            np.asarray(block.data, dtype=np.int64)
            for block in mesh_data.cells
            if block.type == "triangle"
        ]
        if not triangle_blocks:
            raise ValueError(f"trajectory mesh {self.mesh_path} has no triangles")
        archive_metadata = dict(self.context)
        archive_metadata.update(metadata)
        archive_metadata.update({
            "format": "hybridge_torsion_optimizer_trajectory_v2",
            "mesh_path": str(self.mesh_path.resolve()),
            "states": self.states,
            "complete": bool(metadata.get("complete", False)),
        })
        state_phi = np.vstack(self.states_phi)
        state_rho = np.vstack(self.states_rho)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_name(f".{self.output.name}.tmp.npz")
        np.savez_compressed(
            temporary,
            mesh_points=np.asarray(mesh_data.points[:, :2], dtype=np.float64),
            mesh_cells=np.vstack(triangle_blocks),
            dof_coordinates=self.coordinates,
            fixed_torsion=self.fixed["torsion"],
            fixed_target_band=self.fixed["target_band"],
            fixed_target_density=self.fixed["target_density"],
            fixed_target_potential=self.fixed["target_potential"],
            states_phi=state_phi,
            states_rho=state_rho,
            states_mismatch=state_phi - self.fixed["target_potential"][None, :],
            metadata=np.asarray(json.dumps(archive_metadata, sort_keys=True)),
        )
        temporary.replace(self.output)

    def _collective_write(self, metadata: dict[str, object]) -> None:
        """Write on rank zero and make any I/O error rank-consistent."""
        local_error = ""
        if self.comm.rank == 0:
            try:
                self._write_rank_zero(metadata)
            except Exception as error:
                local_error = f"{type(error).__name__}: {error}"
        error_message = self.comm.bcast(local_error, root=0)
        if error_message:
            raise RuntimeError(f"failed to write trajectory archive: {error_message}")

    def write(self, metadata: dict[str, object]) -> None:
        """Collectively finalize a complete or explicitly failed archive."""
        if not self.enabled:
            return
        self._collective_write(metadata)
        self._finalized = True

    def flush_partial(self, metadata: dict[str, object]) -> None:
        """Collectively finalize a partial archive on a handled failure path."""
        if not self.enabled or self._finalized:
            return
        partial = {
            "complete": False,
            "terminal_stage": "failure",
            "terminal_status": "PARTIAL_FAILURE",
            **metadata,
        }
        self._collective_write(partial)
        self._finalized = True

    def _flush_at_exit(self) -> None:
        """Best-effort rank-zero-only archive for unhandled failures."""
        if not self.enabled or self._finalized or self.comm.rank != 0:
            return
        try:
            self._write_rank_zero({
                "complete": False,
                "terminal_stage": "unhandled_exception_or_early_exit",
                "terminal_status": "PARTIAL_UNHANDLED_EXIT",
                "failure_message": "optimizer exited before collective trajectory finalization",
            })
        except Exception as error:
            # Never mask the original exception. The campaign driver writes a
            # diagnostic storyboard PNG if no scientific archive survives.
            try:
                print(f"TRAJECTORY_EMERGENCY_FLUSH_FAILED {type(error).__name__}: {error}")
            except Exception:
                pass
        finally:
            self._finalized = True

def params_from_args(args: argparse.Namespace) -> TorsionParameters:
    """Build validated torsion-target parameters from command-line overrides.

    The dataclass carries the defaults used in the torsion-initialized Newton
    parameter studies.  This function applies optional CLI overrides and
    validates the mathematical precondition for the torsion band: the upper
    fractional threshold must exceed the lower one.

    Args:
        args: Parsed command-line namespace.

    Returns:
        ``TorsionParameters`` with user overrides applied.

    Raises:
        ValueError: If the torsion thresholds are not ordered.
    """
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
    if args.init_mode == "legacy" and params.eps_t_ratio <= 0.0:
        raise ValueError("require positive --eps-t-ratio in legacy initialization mode")
    return params


def log_algorithm_step(
        comm: MPI.Comm,
        args: argparse.Namespace,
        *,
        iteration: int,
        step: int,
        label: str,
        elapsed: float | None = None,
        detail: str = "",
) -> None:
    """Print one detailed algorithm-step trace line at verbosity level 2.

    The step numbers intentionally match the numbered algorithm in
    ``torsion_initialized_window_reduced_optimization.tex``.  The output is therefore
    grep-friendly and can be used to identify which expensive phase dominates
    wall time in a run.

    Args:
        comm: MPI communicator.  Only rank zero prints through ``root_print``.
        args: Parsed command-line namespace containing ``verbosity``.
        iteration: Outer optimization iteration.  The final exact Newton
            projection uses ``-1``.
        step: Algorithm step number, normally 1 through 12.
        label: Short stable name for the step.
        elapsed: Optional wall time in seconds for the step.
        detail: Optional preformatted diagnostic payload.
    """
    if int(args.verbosity) < 2:
        return
    elapsed_text = "" if elapsed is None else f" time={elapsed:.6f}s"
    detail_text = "" if not detail else f" {detail}"
    root_print(comm, f"ALGO_STEP k={iteration} step={step:02d} {label}{elapsed_text}{detail_text}")


def logistic_const_ufl(z, eps):
    """Return a numerically clipped logistic expression in UFL.

    UFL expressions are evaluated by generated quadrature kernels, so extremely
    large positive or negative exponents should be avoided in the symbolic
    expression itself.  This helper clips the argument ``z / eps`` at
    ``[-50, 50]`` using ``ufl.conditional`` while preserving the smooth
    logistic formula in the transition region.

    Args:
        z: UFL scalar expression representing ``s - c``.
        eps: UFL or scalar smoothing width.

    Returns:
        UFL expression approximating ``1 / (1 + exp(-z/eps))``.
    """
    zz = z / eps
    return ufl.conditional(
        ufl.gt(zz, 50.0),
        1.0,
        ufl.conditional(ufl.lt(zz, -50.0), 0.0, 1.0 / (1.0 + ufl.exp(-zz))),
    )


def logistic_q_ufl(values, c, eps):
    """Return the logistic derivative factor ``sigma(1-sigma)``.

    The factor is reused in all derivatives of the logistic window.  Keeping it
    centralized reduces the risk of sign mistakes between state, threshold, and
    epsilon derivatives.

    Args:
        values: UFL expression for the current potential/state value.
        c: UFL or scalar threshold.
        eps: UFL or scalar logistic smoothing width.

    Returns:
        UFL expression for ``sigma((values-c)/eps) * (1-sigma(...))``.
    """
    s = logistic_const_ufl(values - c, eps)
    return s * (1.0 - s)


def window_activity_const_ufl(values, c1, c2, eps):
    """Return the unscaled logistic activity window.

    The activity is the differentiable surrogate for the active band used in
    the outer objective.  It is intentionally unscaled: geometric leakage and
    missing-area functionals measure area, while the PDE right-hand side uses
    ``rho_amp`` times this expression.

    Args:
        values: UFL expression for the potential ``phi``.
        c1: Lower potential threshold.
        c2: Upper potential threshold.
        eps: Logistic smoothing width.

    Returns:
        UFL expression for ``sigma((phi-c1)/eps) - sigma((phi-c2)/eps)``.
    """
    return logistic_const_ufl(values - c1, eps) - logistic_const_ufl(values - c2, eps)


def window_density_const_ufl(values, c1, c2, eps, amp: float):
    """Return the semilinear density ``rho_amp * activity``.

    This is the source term in the semilinear equilibrium
    ``-Delta phi = rho_amp * W(phi;c1,c2,eps)``.  It is separated from
    ``window_activity_const_ufl`` because the optimization functionals use
    activity area, not density mass.

    Args:
        values: UFL expression for the potential.
        c1: Lower potential threshold.
        c2: Upper potential threshold.
        eps: Logistic smoothing width.
        amp: Density amplitude.

    Returns:
        UFL expression for the scaled density window.
    """
    return float(amp) * window_activity_const_ufl(values, c1, c2, eps)


def window_s_derivative_activity_ufl(values, c1, c2, eps):
    """Differentiate the unscaled activity window with respect to state.

    This derivative appears in the Newton matrix and in functional gradients
    with respect to the finite-element coefficients.  It is
    ``(q1 - q2) / eps`` where ``qi = sigma_i(1-sigma_i)``.

    Args:
        values: UFL expression for the potential.
        c1: Lower potential threshold.
        c2: Upper potential threshold.
        eps: Logistic smoothing width.

    Returns:
        UFL expression for ``dW/dphi``.
    """
    q1 = logistic_q_ufl(values, c1, eps)
    q2 = logistic_q_ufl(values, c2, eps)
    return (q1 - q2) / eps


def window_eps_derivative_activity_ufl(values, c1, c2, eps):
    """Differentiate the unscaled activity window with respect to epsilon.

    The reduced optimizer defaults to ``eps = eps_ratio * (c2-c1)``.  In that
    relative-epsilon mode, total threshold derivatives must include the chain
    rule contribution from this epsilon derivative.

    Args:
        values: UFL expression for the potential.
        c1: Lower potential threshold.
        c2: Upper potential threshold.
        eps: Logistic smoothing width.

    Returns:
        UFL expression for ``dW/deps`` at fixed state and thresholds.
    """
    q1 = logistic_q_ufl(values, c1, eps)
    q2 = logistic_q_ufl(values, c2, eps)
    eps2 = eps * eps
    return -((values - c1) / eps2) * q1 + ((values - c2) / eps2) * q2


def window_c_derivatives_activity_ufl(values, c1, c2, eps, *, eps_mode: str, eps_ratio: float):
    """Return total threshold derivatives of the unscaled activity window.

    For a fixed epsilon, the derivatives are the formulas from the algorithm
    note: ``dW/dc1 = -q1/eps`` and ``dW/dc2 = q2/eps``.  In relative-epsilon
    mode this script uses ``eps = eps_ratio * (c2-c1)`` to match the other
    torsion-initialized Newton runners.  The total derivatives then include
    ``dW/deps * deps/dci`` with ``deps/dc1 = -eps_ratio`` and
    ``deps/dc2 = eps_ratio``.

    Args:
        values: UFL expression for the potential.
        c1: Lower potential threshold.
        c2: Upper potential threshold.
        eps: Logistic smoothing width.
        eps_mode: ``"fixed"`` for fixed epsilon or ``"relative"`` for
            ``eps_ratio * (c2-c1)``.
        eps_ratio: Relative smoothing ratio used only in relative mode.

    Returns:
        Pair of UFL expressions ``(dW/dc1, dW/dc2)`` including epsilon-chain
        terms when applicable.
    """
    q1 = logistic_q_ufl(values, c1, eps)
    q2 = logistic_q_ufl(values, c2, eps)
    dc1 = -q1 / eps
    dc2 = q2 / eps
    if eps_mode == "relative":
        deps = window_eps_derivative_activity_ufl(values, c1, c2, eps)
        dc1 = dc1 - float(eps_ratio) * deps
        dc2 = dc2 + float(eps_ratio) * deps
    return dc1, dc2


def window_center_width_derivatives_activity_ufl(
        values,
        c1,
        c2,
        eps,
        *,
        eps_ratio: float,
):
    """Return explicit ``(partial_m W, partial_d W)`` for ``eps=r_eps*d``.

    This is the center--width counterpart of
    :func:`window_c_derivatives_activity_ufl`.  The latter already includes
    the total epsilon dependence for ``eps=r_eps*(c2-c1)``.  Applying
    ``c1=m-d/2`` and ``c2=m+d/2`` therefore gives

    ``W_m = W_c1 + W_c2`` and ``W_d = (W_c2-W_c1)/2``.

    Centralizing this transformation prevents the width sensitivity from
    accidentally omitting the ``d -> eps`` chain-rule term.
    """
    dc1, dc2 = window_c_derivatives_activity_ufl(
        values,
        c1,
        c2,
        eps,
        eps_mode="relative",
        eps_ratio=eps_ratio,
    )
    return dc1 + dc2, 0.5 * (dc2 - dc1)


def epsilon_from_thresholds(args: argparse.Namespace, c1: float, c2: float) -> float:
    """Compute the semilinear smoothing width for one threshold pair.

    The mathematical note is written for a fixed ``epsilon``.  The numerical
    torsion-initialized Newton runners usually use a width-relative value so
    the interface thickness follows the band width.  This function is the
    single place where that convention is selected.

    Args:
        args: Parsed command-line namespace containing ``eps_mode``,
            ``eps_phi``, and ``eps_ratio``.
        c1: Lower potential threshold.
        c2: Upper potential threshold.

    Returns:
        Positive smoothing width.

    Raises:
        ValueError: If fixed-epsilon mode is selected without ``--eps-phi``.
    """
    if args.eps_mode == "fixed":
        if args.eps_phi is None:
            raise ValueError("--eps-phi is required when --eps-mode fixed")
        return float(args.eps_phi)
    return float(args.eps_ratio) * (float(c2) - float(c1))


def certified_min_width(args: argparse.Namespace, c_scale: float) -> float:
    """Return the minimum admissible threshold width.

    The certified active band removes ``kappa * eps`` from both logistic
    transition layers.  For fixed epsilon the width condition is simply
    ``c2-c1 >= 2*kappa*eps + delta_c``.  For relative epsilon,
    ``eps = eps_ratio * (c2-c1)``, so the nonempty-plateau condition becomes a
    constraint on the relative ratio: ``1 - 2*kappa*eps_ratio > 0``.

    Args:
        args: Parsed command-line namespace.
        c_scale: Search interval scale used to convert
            ``min_width_fraction`` into an absolute fallback width.

    Returns:
        Absolute minimum width for ``c2-c1``.

    Raises:
        ValueError: If fixed-epsilon mode lacks ``eps_phi`` or relative mode
            cannot produce a nonempty certified band.
    """
    delta_c = max(float(args.delta_c), float(args.min_width_fraction) * float(c_scale), 1.0e-14)
    if args.eps_mode == "fixed":
        if args.eps_phi is None:
            raise ValueError("--eps-phi is required when --eps-mode fixed")
        return max(delta_c, 2.0 * float(args.kappa) * float(args.eps_phi) + float(args.delta_c))
    denominator = 1.0 - 2.0 * float(args.kappa) * float(args.eps_ratio)
    if denominator <= 0.0:
        raise ValueError("relative epsilon requires 2*kappa*eps_ratio < 1 for a nonempty certified band")
    return max(delta_c, float(args.delta_c) / denominator if args.delta_c > 0.0 else delta_c)


def project_thresholds(
        c1: float,
        c2: float,
        *,
        c_min: float,
        c_max: float,
        min_width: float,
) -> tuple[float, float]:
    """Project thresholds onto the box and minimum-width constraints.

    Projection is done in center-width coordinates to preserve the proposed
    band center whenever possible.  Width is clamped first, then the center is
    clamped so the interval lies inside ``[c_min, c_max]``.

    Args:
        c1: Proposed lower threshold.
        c2: Proposed upper threshold.
        c_min: Lower search bound.
        c_max: Upper search bound.
        min_width: Minimum admissible ``c2-c1``.

    Returns:
        Projected ``(c1, c2)`` satisfying the simple admissible set.
    """
    width = min(max(float(c2) - float(c1), float(min_width)), float(c_max) - float(c_min))
    center = 0.5 * (float(c1) + float(c2))
    lo = float(c_min) + 0.5 * width
    hi = float(c_max) - 0.5 * width
    if hi < lo:
        width = float(c_max) - float(c_min)
        center = 0.5 * (float(c_min) + float(c_max))
    else:
        center = min(max(center, lo), hi)
    return center - 0.5 * width, center + 0.5 * width


def global_weighted_quantile(
        comm: MPI.Comm,
        values: np.ndarray,
        weights: np.ndarray,
        quantile: float,
) -> float:
    """Compute a weighted quantile from distributed NumPy samples.

    The initializer uses this only once for modest quadrature sample arrays, so
    gathering to rank zero is simpler and less error-prone than implementing a
    distributed selection algorithm.  Nonpositive and nonfinite weights are
    ignored.

    Args:
        comm: MPI communicator.
        values: Local sample values.
        weights: Local nonnegative sample weights.
        quantile: Desired quantile in ``[0,1]``.

    Returns:
        Weighted quantile value, broadcast to every rank.  Returns ``nan`` if
        no positive-weight samples exist.
    """
    return float(global_weighted_quantiles(comm, values, weights, [quantile])[0])


def global_weighted_quantiles(
        comm: MPI.Comm,
        values: np.ndarray,
        weights: np.ndarray,
        quantiles,
) -> np.ndarray:
    """Compute several distributed weighted quantiles with one gather/sort.

    Initializer quantiles all use the same samples.  Gathering and sorting
    once avoids repeating the dominant work for every requested probability.
    """
    requested = np.clip(np.atleast_1d(np.asarray(quantiles, dtype=np.float64)), 0.0, 1.0)
    local_values = np.asarray(values, dtype=np.float64)
    local_weights = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(local_values) & np.isfinite(local_weights) & (local_weights > 0.0)
    gathered = comm.gather((local_values[mask], local_weights[mask]), root=0)
    result = np.full(requested.shape, math.nan, dtype=np.float64)
    if comm.rank == 0:
        value_parts = [part_values for part_values, part_weights in gathered if part_values.size and part_weights.size]
        weight_parts = [part_weights for part_values, part_weights in gathered if part_values.size and part_weights.size]
        if value_parts:
            all_values = np.concatenate(value_parts)
            all_weights = np.concatenate(weight_parts)
            order = np.argsort(all_values)
            sorted_values = all_values[order]
            sorted_weights = all_weights[order]
            cumulative = np.cumsum(sorted_weights)
            total = float(cumulative[-1])
            if total > 0.0:
                indices = np.searchsorted(cumulative, requested * total, side="left")
                np.clip(indices, 0, sorted_values.size - 1, out=indices)
                result[:] = sorted_values[indices]
    comm.Bcast(result, root=0)
    return result


def initial_candidate_from_thresholds(
        *,
        comm: MPI.Comm,
        name: str,
        c1: float,
        c2: float,
        phi_values: np.ndarray,
        rho_values: np.ndarray,
        weights: np.ndarray,
        rho_design_l2: float,
        rho_amp: float,
        target_area: float,
        c_min: float,
        c_max: float,
        min_width: float,
        args: argparse.Namespace,
) -> InitialWindowCandidate:
    """Project, evaluate, and score one initializer threshold pair.

    Candidate metrics are sampled on ``phi_target`` before the nonlinear state
    solve.  ``rho_design`` is now an interpolated visualization/diagnostic
    representation of the sharp target ``rho_amp*1_{B_T}``; the Poisson design
    potential itself is assembled directly from the sharp UFL indicator.

    Args:
        comm: MPI communicator.
        name: Candidate label.
        c1: Proposed lower threshold.
        c2: Proposed upper threshold.
        phi_values: Local quadrature samples of ``phi_target``.
        rho_values: Local quadrature samples of ``rho_design``.
        weights: Local physical quadrature weights.
        rho_design_l2: Global L2 norm of ``rho_design``.
        rho_amp: Density amplitude.
        target_area: Crisp torsion target area used for relative metrics.
        c_min: Lower admissible threshold.
        c_max: Upper admissible threshold.
        min_width: Minimum admissible threshold width.
        args: Parsed CLI namespace.

    Returns:
        Scored ``InitialWindowCandidate``.
    """
    c1, c2 = project_thresholds(c1, c2, c_min=c_min, c_max=c_max, min_width=min_width)
    eps_phi = epsilon_from_thresholds(args, c1, c2)
    activity = window_numpy(np.asarray(phi_values, dtype=np.float64), c1, c2, eps_phi, 1.0)
    density = float(rho_amp) * activity
    target = np.clip(np.asarray(rho_values, dtype=np.float64) / max(float(rho_amp), 1.0e-30), 0.0, 1.0)
    weights = np.asarray(weights, dtype=np.float64)
    local_leakage = float(np.dot(weights, (1.0 - target) * activity))
    local_missing = float(np.dot(weights, target * (1.0 - activity)))
    local_activity_area = float(np.dot(weights, activity))
    local_overlap = float(np.dot(weights, target * activity))
    local_l2_sq = float(np.dot(weights, (density - rho_values) ** 2))
    global_values = np.array(
        [local_leakage, local_missing, local_activity_area, local_overlap, local_l2_sq],
        dtype=np.float64,
    )
    reduced = np.empty_like(global_values)
    comm.Allreduce(global_values, reduced, op=MPI.SUM)
    leakage, missing, activity_area, overlap, l2_sq = map(float, reduced)
    target_scale = max(float(target_area), 1.0e-30)
    leakage_rel = leakage / target_scale
    missing_rel = missing / target_scale
    area_rel = activity_area / target_scale
    union = max(float(target_area) + activity_area - overlap, 1.0e-30)
    active_jaccard = overlap / union
    l2_rel = math.sqrt(max(l2_sq, 0.0)) / max(float(rho_design_l2), 1.0e-30)
    score = 2.0 * missing_rel + leakage_rel + abs(area_rel - 1.0)
    return InitialWindowCandidate(
        name=name,
        c1=c1,
        c2=c2,
        eps_phi=eps_phi,
        leakage_rel=leakage_rel,
        missing_rel=missing_rel,
        activity_area=activity_area,
        area_rel=area_rel,
        active_jaccard=active_jaccard,
        l2_rel=l2_rel,
        score=score,
    )


def area_matched_initial_thresholds(
        *,
        comm: MPI.Comm,
        center: float,
        phi_values: np.ndarray,
        weights: np.ndarray,
        target_area: float,
        c_min: float,
        c_max: float,
        min_width: float,
        args: argparse.Namespace,
) -> tuple[float, float]:
    """Choose a centered window width whose sampled activity area matches target area.

    The center is usually the target-weighted median of ``phi_target``.  Width
    is found by bisection because the sampled activity area is monotone in the
    window width for a fixed center and positive smoothing ratio.  Projection
    keeps the candidate admissible near search-boundary centers.

    Args:
        comm: MPI communicator.
        center: Desired threshold center.
        phi_values: Local quadrature samples of ``phi_target``.
        weights: Local physical quadrature weights.
        target_area: Desired activity area.
        c_min: Lower admissible threshold.
        c_max: Upper admissible threshold.
        min_width: Minimum admissible threshold width.
        args: Parsed CLI namespace.

    Returns:
        Area-matched admissible ``(c1,c2)``.
    """

    def projected_from_width(width: float) -> tuple[float, float]:
        """Return projected thresholds with the requested center/width."""
        return project_thresholds(
            float(center) - 0.5 * float(width),
            float(center) + 0.5 * float(width),
            c_min=c_min,
            c_max=c_max,
            min_width=min_width,
        )

    def activity_area_for(c1: float, c2: float) -> float:
        """Evaluate sampled activity area for one threshold interval."""
        eps_phi = epsilon_from_thresholds(args, c1, c2)
        activity = window_numpy(np.asarray(phi_values, dtype=np.float64), c1, c2, eps_phi, 1.0)
        local_area = float(np.dot(np.asarray(weights, dtype=np.float64), activity))
        return float(comm.allreduce(local_area, op=MPI.SUM))

    lo = float(min_width)
    hi = max(float(c_max) - float(c_min), lo)
    best = projected_from_width(lo)
    best_error = math.inf
    for _ in range(48):
        width = 0.5 * (lo + hi)
        c1, c2 = projected_from_width(width)
        area = activity_area_for(c1, c2)
        error = abs(area - float(target_area))
        if error < best_error:
            best = (c1, c2)
            best_error = error
        if area < float(target_area):
            lo = width
        else:
            hi = width
    return best


def build_initial_window_candidates(
        *,
        phi_target: fem.Function,
        rho_design: fem.Function,
        fit_candidate: tuple[str, float, float] | None,
        rho_design_l2: float,
        rho_amp: float,
        target_area: float,
        quadrature_degree: int,
        c_min: float,
        c_max: float,
        min_width: float,
        args: argparse.Namespace,
        quadrature_samples: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> list[InitialWindowCandidate]:
    """Generate L2, target-quantile, and area-matched initial windows.

    No additional user parameters are exposed.  The quantile candidates use
    fixed central target-weighted ranges of ``phi_target`` under the smoothed
    target-band weights.  The area-matched candidate uses the same
    target-weighted median as its center and chooses a width whose sampled
    activity area is close to the torsion target area.

    Args:
        phi_target: Poisson target potential ``-Delta^{-1} rho_design``.
        rho_design: Interpolated diagnostic representation of the sharp
            torsion-designed density.
        fit_candidate: Optional ``(name,c1,c2)`` from the existing L2 fit.
        rho_design_l2: Global L2 norm of ``rho_design``.
        rho_amp: Density amplitude.
        target_area: Crisp torsion target area.
        quadrature_degree: Degree used for initialization samples.
        c_min: Lower admissible threshold.
        c_max: Upper admissible threshold.
        min_width: Minimum admissible threshold width.
        args: Parsed CLI namespace.

    Returns:
        List of scored candidates.  The caller should choose the minimum score.
    """
    comm = phi_target.function_space.mesh.comm
    if quadrature_samples is None:
        phi_values, rho_values, weights = quadrature_samples_for_fit(
            phi_target,
            rho_design,
            quadrature_degree=int(quadrature_degree),
        )
    else:
        phi_values, rho_values, weights = quadrature_samples
    # Candidate construction is lightweight compared with the nonlinear
    # projections, and these arrays were already small enough for the former
    # quantile gather.  Gather once so quantiles, area matching, and all
    # candidate scores run on rank zero without dozens of synchronized scalar
    # reductions.  The resulting dataclasses are then broadcast to every rank.
    gathered = comm.gather((phi_values, rho_values, weights), root=0)
    if comm.rank != 0:
        return comm.bcast(None, root=0)
    phi_values = np.ascontiguousarray(np.concatenate([part[0] for part in gathered if part[0].size]))
    rho_values = np.ascontiguousarray(np.concatenate([part[1] for part in gathered if part[1].size]))
    weights = np.ascontiguousarray(np.concatenate([part[2] for part in gathered if part[2].size]))
    candidate_comm = MPI.COMM_SELF
    candidates: list[InitialWindowCandidate] = []

    def append_candidate(name: str, c1: float, c2: float) -> None:
        """Add one projected/scored candidate if it is finite."""
        candidate = initial_candidate_from_thresholds(
            comm=candidate_comm,
            name=name,
            c1=c1,
            c2=c2,
            phi_values=phi_values,
            rho_values=rho_values,
            weights=weights,
            rho_design_l2=rho_design_l2,
            rho_amp=rho_amp,
            target_area=target_area,
            c_min=c_min,
            c_max=c_max,
            min_width=min_width,
            args=args,
        )
        if math.isfinite(candidate.score):
            candidates.append(candidate)

    if fit_candidate is not None:
        name, c1_fit, c2_fit = fit_candidate
        append_candidate(name, c1_fit, c2_fit)

    target_weights = np.asarray(weights, dtype=np.float64) * np.clip(
        np.asarray(rho_values, dtype=np.float64) / max(float(rho_amp), 1.0e-30),
        0.0,
        1.0,
    )
    q05, q10, q50, q90, q95 = global_weighted_quantiles(
        candidate_comm,
        phi_values,
        target_weights,
        [0.05, 0.10, 0.50, 0.90, 0.95],
    )
    if math.isfinite(q05) and math.isfinite(q95) and q95 > q05:
        append_candidate("target_quantile_05_95", q05, q95)
    if math.isfinite(q10) and math.isfinite(q90) and q90 > q10:
        append_candidate("target_quantile_10_90", q10, q90)
    if math.isfinite(q50):
        c1_area, c2_area = area_matched_initial_thresholds(
            comm=candidate_comm,
            center=q50,
            phi_values=phi_values,
            weights=weights,
            target_area=target_area,
            c_min=c_min,
            c_max=c_max,
            min_width=min_width,
            args=args,
        )
        append_candidate("target_median_area", c1_area, c2_area)
    return comm.bcast(candidates, root=0)


def projected_initial_candidate_score(
        *,
        metrics: BandMetrics,
        diagnostics: dict[str, float],
        target_area: float,
        converged: bool,
) -> float:
    """Score an initializer candidate after semilinear Newton projection.

    The pre-projection density L2 fit can look acceptable on ``phi_target`` and
    still jump to a branch whose active set misses the torsion band once the
    fixed-threshold PDE is solved.  This score therefore emphasizes quantities
    measured on the projected semilinear state:

    * missing target area, with the largest weight, because a branch that does
      not cover the torsion band leaves the reduced optimizer minimizing
      leakage while the missing-area gradient can be uninformative;
    * leakage and total activity-area mismatch as secondary controls, to avoid
      simply selecting an overly thick global band;
    * active-set recall/Jaccard and relative density mismatch, to keep the
      selected state geometrically close to the torsion-designed density.

    Args:
        metrics: Soft leakage/missing metrics for the projected state.
        diagnostics: Density diagnostics returned by ``compute_metrics`` for
            the same projected state.
        target_area: Crisp torsion target area used to normalize area
            mismatch.
        converged: Whether the initializer Newton projection reached its
            requested residual tolerance.

    Returns:
        Nonnegative scalar score.  Lower is better; a failed projection receives
        a finite penalty so it can still be reported if every candidate fails.
    """
    area_rel = metrics.activity_area / max(float(target_area), 1.0e-30)
    active_overlap = float(diagnostics.get("activeOverlapArea", 0.0))
    active_design_area = max(float(diagnostics.get("activeDesignArea", 0.0)), 1.0e-30)
    active_recall = active_overlap / active_design_area
    active_jaccard = float(diagnostics.get("activeJaccard", 0.0))
    rho_rel = float(diagnostics.get("relRhoDesign", 0.0))
    projection_penalty = 5.0 if not converged else 0.0
    return (
        projection_penalty
        + 10.0 * max(metrics.missing_rel, 0.0)
        + 0.5 * max(metrics.leakage_rel, 0.0)
        + 0.5 * abs(area_rel - 1.0)
        + max(1.0 - active_recall, 0.0)
        + 0.5 * max(1.0 - active_jaccard, 0.0)
        + 0.25 * max(rho_rel, 0.0)
    )


def assemble_vector_form(linear_form, bc) -> PETSc.Vec:
    """Assemble a linear UFL form into a PETSc vector with Dirichlet data.

    The helper mirrors the residual-vector assembly pattern used elsewhere in
    the torsion-initialized Newton scripts: assemble local contributions,
    accumulate ghosts back to owners, then impose the homogeneous boundary
    condition entries.

    Args:
        linear_form: UFL linear form.
        bc: DOLFINx Dirichlet boundary condition.

    Returns:
        PETSc vector owned by the caller, who must destroy it when finished.
    """
    vec = fem_petsc.assemble_vector(fem.form(linear_form))
    vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(vec, [bc])
    return vec


def assemble_residual_vector(residual_form, bc) -> PETSc.Vec:
    """Assemble a residual form into a PETSc vector.

    This is a semantic wrapper around ``assemble_vector_form``.  It makes call
    sites involving residual norms easier to read and keeps residual assembly
    consistent with generic linear-form assembly.

    Args:
        residual_form: UFL residual form tested against the FE basis.
        bc: Homogeneous Dirichlet boundary condition.

    Returns:
        PETSc residual vector owned by the caller.
    """
    return assemble_vector_form(residual_form, bc)


def assemble_matrix_form(bilinear_form, bcs: list) -> PETSc.Mat:
    """Assemble a bilinear UFL form into a PETSc matrix.

    Args:
        bilinear_form: UFL bilinear form.
        bcs: Boundary conditions applied during matrix assembly.

    Returns:
        Assembled PETSc matrix owned by the caller.
    """
    mat = fem_petsc.assemble_matrix(fem.form(bilinear_form), bcs=bcs)
    mat.assemble()
    return mat


def diagnose_newton_spd(
        jacobian_expr,
        bcs: list,
        stiffness_matrix: PETSc.Mat,
        *,
        args: argparse.Namespace,
        prefix: str,
        stage: str,
        newton_iteration: int,
        residual: float,
        homotopy_lambda: float,
        c1: float,
        c2: float,
        eps_phi: float,
) -> None:
    """Measure the stiffness-relative coercivity margin of one Newton matrix.

    For J = K - R, the smallest generalized eigenvalue of J x = mu K x is
    mu_min = 1 - theta_max, where theta_max is the largest eigenvalue of
    R x = theta K x. The latter is an extremal Hermitian generalized
    eigenproblem and is substantially easier to resolve than searching for an
    eigenvalue of J nearest zero.
    """
    from slepc4py import SLEPc

    comm = stiffness_matrix.getComm()
    started = time.perf_counter()
    jacobian = assemble_matrix_form(jacobian_expr, bcs)
    symmetry_tolerance = max(1.0e-13, 0.1 * float(args.newton_spd_eig_tol))
    symmetric = bool(jacobian.isSymmetric(tol=symmetry_tolerance))
    theta_max = math.nan
    mu_min = math.nan
    eigen_error = math.nan
    eig_iterations = 0
    eig_converged = 0
    negative_modes = -1
    zero_modes = -1
    positive_modes = -1
    inertia_ksp = None
    status = "NONSYMMETRIC"

    reaction = None
    eps = None
    try:
        if symmetric:
            reaction = stiffness_matrix.copy()
            reaction.axpy(
                PETSc.ScalarType(-1.0),
                jacobian,
                structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN,
            )
            reaction.assemble()
            reaction.setOption(PETSc.Mat.Option.SYMMETRIC, True)

            eps = SLEPc.EPS().create(comm)
            eps.setOperators(reaction, stiffness_matrix)
            eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
            eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
            eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
            eps.setDimensions(1)
            eps.setTolerances(
                tol=float(args.newton_spd_eig_tol),
                max_it=int(args.newton_spd_eig_max_it),
            )
            eps.solve()
            eig_iterations = int(eps.getIterationNumber())
            eig_converged = int(eps.getConverged())
            if eig_converged:
                theta_max = float(np.real(eps.getEigenvalue(0)))
                eigen_error = float(eps.computeError(0, SLEPc.EPS.ErrorType.ABSOLUTE))
                mu_min = 1.0 - theta_max
                lower = mu_min - eigen_error
                upper = mu_min + eigen_error
                zero_tolerance = float(args.newton_spd_zero_tol)
                if lower > zero_tolerance:
                    status = "SPD"
                elif upper < -zero_tolerance:
                    status = "INDEFINITE"
                else:
                    status = "NEAR_BOUNDARY"
            else:
                status = "EIGEN_NOT_CONVERGED"

            if bool(args.newton_spd_inertia):
                jacobian.setOption(PETSc.Mat.Option.SYMMETRIC, True)
                inertia_ksp = PETSc.KSP().create(comm)
                inertia_prefix = "newton_spd_inertia_"
                inertia_ksp.setOptionsPrefix(inertia_prefix)
                options = PETSc.Options()
                options[f"{inertia_prefix}ksp_type"] = "preonly"
                options[f"{inertia_prefix}pc_type"] = "cholesky"
                options[f"{inertia_prefix}pc_factor_mat_solver_type"] = "mumps"
                options[f"{inertia_prefix}mat_mumps_icntl_24"] = 1
                inertia_ksp.setFromOptions()
                inertia_ksp.setOperators(jacobian)
                inertia_ksp.setUp()
                negative_modes, zero_modes, positive_modes = (
                    int(value)
                    for value in inertia_ksp.getPC().getFactorMatrix().getInertia()
                )
    finally:
        if inertia_ksp is not None:
            inertia_ksp.destroy()
        if eps is not None:
            eps.destroy()
        if reaction is not None:
            reaction.destroy()
        jacobian.destroy()

    record = {
        "prefix": prefix,
        "stage": stage,
        "newtonIteration": int(newton_iteration),
        "lambda": float(homotopy_lambda),
        "residual": float(residual),
        "c1": float(c1),
        "c2": float(c2),
        "eps": float(eps_phi),
        "symmetric": int(symmetric),
        "status": status,
        "thetaMax": theta_max,
        "muMin": mu_min,
        "shiftToSpd": max(0.0, -mu_min) if math.isfinite(mu_min) else math.nan,
        "eigenError": eigen_error,
        "eigenIterations": eig_iterations,
        "eigenConverged": eig_converged,
        "negativeModes": negative_modes,
        "zeroModes": zero_modes,
        "positiveModes": positive_modes,
        "inertiaAgrees": int((status == "INDEFINITE") == (negative_modes > 0)) if negative_modes >= 0 else -1,
        "elapsed": time.perf_counter() - started,
    }
    args.newton_spd_records.append(record)
    root_print(
        comm,
        f"NEWTON_SPD prefix={prefix} stage={stage} k={newton_iteration} "
        f"lambda={homotopy_lambda:.6e} symmetric={int(symmetric)} status={status} "
        f"muMin={mu_min:.12e} shiftToSpd={record['shiftToSpd']:.12e} "
        f"eigError={eigen_error:.3e} eigIts={eig_iterations} "
        f"inertia=({negative_modes},{zero_modes},{positive_modes})",
    )

def zero_vector(vec: PETSc.Vec) -> None:
    """Zero owned and ghost entries of a DOLFINx PETSc vector in place."""
    with vec.localForm() as local:
        local.set(0.0)


class FixedStiffnessSolver:
    """Run-scoped stiffness matrix, factorization/preconditioner, and vectors.

    The stiffness operator is invariant throughout the run.  Besides the two
    Poisson solves, it defines the discrete dual residual norm and Newton-step
    H1 seminorm, so retaining it removes a matrix assembly and KSP setup from
    every residual and line-search evaluation.
    """

    def __init__(
            self,
            stiffness_form,
            V,
            bcs: list,
            *,
            prefix: str,
            solver: str,
            ksp_type: str | None,
            rtol: float,
            atol: float,
            max_it: int | None,
    ) -> None:
        setup_start = time.perf_counter()
        self.form = fem.form(stiffness_form)
        self.bcs = bcs
        self.matrix = fem_petsc.assemble_matrix(self.form, bcs=bcs)
        self.matrix.assemble()
        self.rhs = fem_petsc.create_vector(V)
        self.solution = self.rhs.duplicate()
        self.matvec = self.rhs.duplicate()
        self.ksp = PETSc.KSP().create(V.mesh.comm)
        self.ksp.setOptionsPrefix(prefix)
        opts = PETSc.Options()
        for key, value in solver_options(solver, ksp_type=ksp_type).items():
            opts[f"{prefix}{key}"] = value
        if solver not in {"mumps", "lu"}:
            opts[f"{prefix}ksp_rtol"] = rtol
            opts[f"{prefix}ksp_atol"] = atol
            if max_it is not None:
                opts[f"{prefix}ksp_max_it"] = max_it
        self.ksp.setFromOptions()
        self.ksp.setOperators(self.matrix)
        self.setup_time = time.perf_counter() - setup_start
        self._report_setup = True
        self._closed = False

    def _assemble_rhs(self, linear_form, *, lifting: bool) -> None:
        zero_vector(self.rhs)
        fem_petsc.assemble_vector(self.rhs, linear_form)
        if lifting:
            fem_petsc.apply_lifting(self.rhs, [self.form], [self.bcs])
        self.rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(self.rhs, self.bcs)

    def solve_form(self, rhs_form, target: fem.Function) -> tuple[int, float, float]:
        """Solve the fixed stiffness system for a variational RHS."""
        start = time.perf_counter()
        self._assemble_rhs(fem.form(rhs_form), lifting=True)
        self.ksp.solve(self.rhs, target.x.petsc_vec)
        target.x.scatter_forward()
        reason = self.ksp.getConvergedReason()
        elapsed = time.perf_counter() - start
        if self._report_setup:
            elapsed += self.setup_time
            self._report_setup = False
        if reason < 0:
            raise RuntimeError(f"fixed stiffness solve failed with PETSc reason {reason}")
        return int(self.ksp.getIterationNumber()), float(self.ksp.getResidualNorm()), elapsed

    def residual_norm(self, residual_form, mode: str) -> float:
        """Evaluate a compiled residual form in Euclidean or dual norm."""
        self._assemble_rhs(residual_form, lifting=False)
        if mode == "euclidean":
            return float(self.rhs.norm())
        zero_vector(self.solution)
        self.ksp.solve(self.rhs, self.solution)
        reason = self.ksp.getConvergedReason()
        if reason < 0:
            raise RuntimeError(f"dual residual solve failed with PETSc reason {reason}")
        return math.sqrt(max(float(self.rhs.dot(self.solution)), 0.0))

    def solve_residual_form(
            self,
            residual_form,
            target: fem.Function,
    ) -> tuple[float, int, float, float]:
        """Assemble ``r``, solve ``K target=r``, and return ``r.T*K^-1*r``.

        The stiffness matrix and KSP are the run-scoped objects owned by this
        class.  Consequently frozen-threshold evaluations assemble only their
        changing residual vector and reuse the existing factorization or
        preconditioner hierarchy.
        """
        start = time.perf_counter()
        self._assemble_rhs(residual_form, lifting=False)
        target.x.petsc_vec.set(0.0)
        self.ksp.solve(self.rhs, target.x.petsc_vec)
        target.x.scatter_forward()
        reason = self.ksp.getConvergedReason()
        if reason < 0:
            raise RuntimeError(f"frozen H^-1 solve failed with PETSc reason {reason}")
        dual_sq = max(float(self.rhs.dot(target.x.petsc_vec)), 0.0)
        return (
            dual_sq,
            int(self.ksp.getIterationNumber()),
            float(self.ksp.getResidualNorm()),
            time.perf_counter() - start,
        )

    def h1_seminorm(self, function: fem.Function) -> float:
        """Compute ``sqrt(x.T K x)`` without assembling a scalar form."""
        self.matrix.mult(function.x.petsc_vec, self.matvec)
        return math.sqrt(max(float(function.x.petsc_vec.dot(self.matvec)), 0.0))

    @property
    def residual_vector(self) -> PETSc.Vec:
        """Most recently assembled residual vector (borrowed, not owned)."""
        return self.rhs

    def close(self) -> None:
        if self._closed:
            return
        self.ksp.destroy()
        self.matrix.destroy()
        self.rhs.destroy()
        self.solution.destroy()
        self.matvec.destroy()
        self._closed = True


class FrozenThresholdObjective:
    """Compiled frozen-state forms backed by the shared stiffness solver."""

    def __init__(
            self,
            *,
            phi_target: fem.Function,
            riesz: fem.Function,
            tau_mask,
            test,
            dx,
            bc,
            stiffness_solver: FixedStiffnessSolver,
            c1_const: fem.Constant,
            c2_const: fem.Constant,
            eps_const: fem.Constant,
            rho_amp: float,
            target_area: float,
            c_min: float,
            c_max: float,
            min_width: float,
            args: argparse.Namespace,
    ) -> None:
        self.comm = phi_target.function_space.mesh.comm
        self.stiffness_solver = stiffness_solver
        self.riesz = riesz
        self.c1_const = c1_const
        self.c2_const = c2_const
        self.eps_const = eps_const
        self.rho_amp = float(rho_amp)
        self.target_area = float(target_area)
        self.c_min = float(c_min)
        self.c_max = float(c_max)
        self.min_width = float(min_width)
        self.args = args

        activity = window_activity_const_ufl(phi_target, c1_const, c2_const, eps_const)
        d1w, d2w = window_c_derivatives_activity_ufl(
            phi_target,
            c1_const,
            c2_const,
            eps_const,
            eps_mode=args.eps_mode,
            eps_ratio=args.eps_ratio,
        )
        self.residual_form = fem.form(
            (
                ufl.inner(ufl.grad(phi_target), ufl.grad(test))
                - self.rho_amp * activity * test
            ) * dx
        )
        self.gradient_forms = (
            fem.form(-self.rho_amp * d1w * riesz * dx),
            fem.form(-self.rho_amp * d2w * riesz * dx),
        )
        self.leakage_form = fem.form((1.0 - tau_mask) * activity * dx)
        self.missing_form = fem.form(tau_mask * (1.0 - activity) * dx)
        self.records: list[tuple[int, str, FrozenThresholdEvaluation]] = []
        self._cache: dict[tuple[str, str, bool], FrozenThresholdEvaluation] = {}
        self._evaluation_ids: dict[int, int] = {}

    def _assemble_scalars(self, forms: tuple) -> np.ndarray:
        """Assemble several scalar forms with one collective reduction."""
        local = np.asarray(
            [float(fem.assemble_scalar(form)) for form in forms],
            dtype=np.float64,
        )
        global_values = np.empty_like(local)
        self.comm.Allreduce(local, global_values, op=MPI.SUM)
        return global_values

    def evaluation_id(self, evaluation: FrozenThresholdEvaluation) -> int:
        return self._evaluation_ids[id(evaluation)]

    def evaluate(
            self,
            c1: float,
            c2: float,
            *,
            method: str,
            with_gradient: bool = True,
    ) -> FrozenThresholdEvaluation:
        """Evaluate the assembled frozen objective, optionally with gradient.

        Coarse/refinement screening needs only ``Psi``, leakage, and missing
        area.  Skipping the two gradient forms there avoids needless work and
        two collective reductions per candidate.  The selected point is
        reevaluated once with ``with_gradient=True`` before homotopy.
        """
        c1, c2 = project_thresholds(
            c1,
            c2,
            c_min=self.c_min,
            c_max=self.c_max,
            min_width=self.min_width,
        )
        key = (float(c1).hex(), float(c2).hex(), bool(with_gradient))
        cached = self._cache.get(key)
        if cached is None and not with_gradient:
            cached = self._cache.get((key[0], key[1], True))
        if cached is not None:
            return cached

        start = time.perf_counter()
        eps_phi = epsilon_from_thresholds(self.args, c1, c2)
        self.c1_const.value = PETSc.ScalarType(c1)
        self.c2_const.value = PETSc.ScalarType(c2)
        self.eps_const.value = PETSc.ScalarType(eps_phi)

        dual_sq, _, _, _ = self.stiffness_solver.solve_residual_form(
            self.residual_form,
            self.riesz,
        )
        psi = 0.5 * dual_sq
        scalar_forms = (
            (*self.gradient_forms, self.leakage_form, self.missing_form)
            if with_gradient
            else (self.leakage_form, self.missing_form)
        )
        scalar_values = self._assemble_scalars(scalar_forms)
        if with_gradient:
            grad_psi = np.asarray(scalar_values[:2], dtype=np.float64)
            leakage, missing = map(float, scalar_values[2:])
        else:
            grad_psi = np.full(2, np.nan, dtype=np.float64)
            leakage, missing = map(float, scalar_values)
        activity_area = self.target_area + leakage - missing
        target_scale = max(self.target_area, 1.0e-30)

        projected_grad_norm = None
        if with_gradient:
            projected = project_thresholds(
                c1 - float(grad_psi[0]),
                c2 - float(grad_psi[1]),
                c_min=self.c_min,
                c_max=self.c_max,
                min_width=self.min_width,
            )
            projected_grad_norm = float(
                np.linalg.norm(np.array([c1 - projected[0], c2 - projected[1]], dtype=np.float64))
            )
        evaluation = FrozenThresholdEvaluation(
            c1=c1,
            c2=c2,
            eps_phi=eps_phi,
            psi=psi,
            residual_dual=math.sqrt(max(dual_sq, 0.0)),
            leakage=leakage,
            missing=missing,
            leakage_rel=leakage / target_scale,
            missing_rel=missing / target_scale,
            activity_area=activity_area,
            activity_area_rel=activity_area / target_scale,
            grad_psi=grad_psi,
            projected_grad_norm=projected_grad_norm,
            evaluation_time=time.perf_counter() - start,
            gradient_evaluated=bool(with_gradient),
        )
        eval_id = len(self.records) + 1
        self.records.append((eval_id, method, evaluation))
        self._evaluation_ids[id(evaluation)] = eval_id
        self._cache[key] = evaluation
        return evaluation


def evaluate_frozen_threshold_objective(
        objective: FrozenThresholdObjective,
        c1: float,
        c2: float,
        *,
        method: str = "frozen",
        with_gradient: bool = True,
) -> FrozenThresholdEvaluation:
    """Evaluate ``Psi_T(c)`` using a precompiled, matrix-reusing workspace."""
    return objective.evaluate(c1, c2, method=method, with_gradient=with_gradient)


def frozen_center_width_grid(
        *,
        center_min: float,
        center_max: float,
        width_min: float,
        width_max: float,
        center_points: int,
        width_points: int,
        c_min: float,
        c_max: float,
) -> np.ndarray:
    """Generate all admissible center/width grid pairs without nested loops."""
    centers = np.linspace(float(center_min), float(center_max), int(center_points))
    widths = np.linspace(float(width_min), float(width_max), int(width_points))
    center_grid, width_grid = np.meshgrid(centers, widths, indexing="ij")
    c1 = center_grid.ravel() - 0.5 * width_grid.ravel()
    c2 = center_grid.ravel() + 0.5 * width_grid.ravel()
    tolerance = 64.0 * np.finfo(np.float64).eps * max(abs(c_min), abs(c_max), 1.0)
    valid = (c1 >= float(c_min) - tolerance) & (c2 <= float(c_max) + tolerance)
    return np.column_stack((c1[valid], c2[valid]))


def optimize_frozen_hminus1_thresholds(
        objective: FrozenThresholdObjective,
        *,
        args: argparse.Namespace,
        seed_pairs: np.ndarray | None = None,
) -> tuple[FrozenThresholdEvaluation, float, float]:
    """Minimize the assembled frozen H^-1 objective under geometric caps.

    A coarse center/width scan identifies a geometrically meaningful region.
    If necessary, the leakage and missing caps are relaxed lexicographically,
    without mixing them into the objective.  Small local grids then refine the
    lowest-Psi feasible point.
    """
    comm = objective.comm
    search_started = time.perf_counter()
    fast_search = str(args.init_search) == "fast"
    grid_size = min(int(args.init_hminus1_grid), 7) if fast_search else int(args.init_hminus1_grid)
    refine_size = min(int(args.init_hminus1_refine_grid), 7) if fast_search else int(args.init_hminus1_refine_grid)
    refine_passes = min(int(args.init_hminus1_refine_passes), int(args.init_hminus1_max_it))
    screen_with_gradient = not fast_search
    progress_every = int(args.init_hminus1_progress_every)
    if args.verbosity >= 1:
        root_print(
            comm,
            f"HMINUS1_SEARCH status=START coarseGrid={grid_size}x{grid_size} "
            f"refineGrid={refine_size}x{refine_size} refinePasses={refine_passes} "
            f"progressEvery={progress_every}",
        )

    def evaluate_pairs(pairs: np.ndarray, *, method: str) -> tuple[list[FrozenThresholdEvaluation], float]:
        """Evaluate one grid with periodic wall-time, rate, and ETA reports."""
        pass_started = time.perf_counter()
        total = int(len(pairs))
        records_before = len(objective.records)
        evaluations: list[FrozenThresholdEvaluation] = []
        if args.verbosity >= 1:
            root_print(
                comm,
                f"HMINUS1_SCAN pass={method} status=START points={total}",
            )
        for completed, pair in enumerate(pairs, start=1):
            evaluation = evaluate_frozen_threshold_objective(
                objective,
                pair[0],
                pair[1],
                method=method,
                with_gradient=screen_with_gradient,
            )
            evaluations.append(evaluation)
            if args.verbosity >= 2 and (
                    completed == 1
                    or completed % progress_every == 0
                    or completed == total
            ):
                elapsed = time.perf_counter() - pass_started
                rate = completed / max(elapsed, 1.0e-30)
                eta = (total - completed) / max(rate, 1.0e-30)
                unique = len(objective.records) - records_before
                root_print(
                    comm,
                    f"HMINUS1_PROGRESS pass={method} completed={completed}/{total} "
                    f"unique={unique} cacheHits={completed - unique} "
                    f"elapsed={elapsed:.3f}s rate={rate:.3f}/s eta={eta:.3f}s "
                    f"lastEval={evaluation.evaluation_time:.3f}s",
                )
        elapsed = time.perf_counter() - pass_started
        if args.verbosity >= 1:
            root_print(
                comm,
                f"HMINUS1_SCAN pass={method} status=EVALUATED points={total} "
                f"unique={len(objective.records) - records_before} time={elapsed:.3f}s",
            )
        return evaluations, elapsed

    half_min_width = 0.5 * objective.min_width
    coarse_pairs = frozen_center_width_grid(
        center_min=objective.c_min + half_min_width,
        center_max=objective.c_max - half_min_width,
        width_min=objective.min_width,
        width_max=objective.c_max - objective.c_min,
        center_points=grid_size,
        width_points=grid_size,
        c_min=objective.c_min,
        c_max=objective.c_max,
    )
    if seed_pairs is not None and np.asarray(seed_pairs).size:
        seed_array = np.asarray(seed_pairs, dtype=np.float64).reshape(-1, 2)
        coarse_pairs = np.unique(np.vstack((seed_array, coarse_pairs)), axis=0)
    coarse, coarse_elapsed = evaluate_pairs(coarse_pairs, method="coarse")
    if not coarse:
        raise RuntimeError("frozen H^-1 coarse grid contains no admissible threshold pair")

    leakage_cap = max(float(args.eta_out), float(args.init_leakage_cap))
    missing_cap = max(float(args.tol_area), float(args.init_missing_cap))

    def feasible(evaluation: FrozenThresholdEvaluation) -> bool:
        return (
            evaluation.leakage_rel <= leakage_cap + 1.0e-14
            and evaluation.missing_rel <= missing_cap + 1.0e-14
        )

    feasible_coarse = [evaluation for evaluation in coarse if feasible(evaluation)]
    while not feasible_coarse and (leakage_cap < 1.0 or missing_cap < 1.0):
        leakage_cap = min(1.0, leakage_cap * float(args.init_geom_relax_factor))
        missing_cap = min(1.0, missing_cap * float(args.init_geom_relax_factor))
        feasible_coarse = [evaluation for evaluation in coarse if feasible(evaluation)]
    if not feasible_coarse:
        raise RuntimeError(
            "no frozen H^-1 coarse point satisfies geometric caps even after relaxation to "
            f"Lrel<={leakage_cap:.3e}, Mrel<={missing_cap:.3e}"
        )

    best = min(feasible_coarse, key=lambda evaluation: evaluation.psi)
    c_range = objective.c_max - objective.c_min
    center_span = c_range / max(grid_size - 1, 1)
    width_span = max(c_range - objective.min_width, center_span) / max(grid_size - 1, 1)
    if args.verbosity >= 1:
        root_print(
            comm,
            f"HMINUS1_SCAN pass=coarse points={len(coarse)} feasible={len(feasible_coarse)} "
            f"Lcap={leakage_cap:.3e} Mcap={missing_cap:.3e} bestPsi={best.psi:.6e} "
            f"time={coarse_elapsed:.3f}s",
        )

    for refinement in range(refine_passes):
        center = 0.5 * (best.c1 + best.c2)
        width = best.c2 - best.c1
        pairs = frozen_center_width_grid(
            center_min=max(objective.c_min + half_min_width, center - center_span),
            center_max=min(objective.c_max - half_min_width, center + center_span),
            width_min=max(objective.min_width, width - width_span),
            width_max=min(c_range, width + width_span),
            center_points=refine_size,
            width_points=refine_size,
            c_min=objective.c_min,
            c_max=objective.c_max,
        )
        pass_name = f"refine_{refinement + 1}"
        refined, refine_elapsed = evaluate_pairs(pairs, method=pass_name)
        feasible_refined = [evaluation for evaluation in refined if feasible(evaluation)]
        if feasible_refined:
            best = min([best, *feasible_refined], key=lambda evaluation: evaluation.psi)
        if args.verbosity >= 1:
            root_print(
                comm,
                f"HMINUS1_SCAN pass=refine_{refinement + 1} points={len(refined)} "
                f"feasible={len(feasible_refined)} bestPsi={best.psi:.6e} "
                f"time={refine_elapsed:.3f}s",
            )
        center_span *= 2.0 / max(refine_size - 1, 1)
        width_span *= 2.0 / max(refine_size - 1, 1)

    if not best.gradient_evaluated:
        best = evaluate_frozen_threshold_objective(
            objective,
            best.c1,
            best.c2,
            method="selected_gradient",
            with_gradient=True,
        )

    if args.verbosity >= 1:
        root_print(
            comm,
            f"HMINUS1_SEARCH status=DONE time={time.perf_counter() - search_started:.3f}s "
            f"mode={args.init_search} objectiveEvaluations={len(objective.records)} "
            f"bestPsi={best.psi:.6e}",
        )
    return best, leakage_cap, missing_cap


def verify_frozen_hminus1_gradient(
        objective: FrozenThresholdObjective,
        evaluation: FrozenThresholdEvaluation,
) -> None:
    """Compare both exact frozen gradients with centered finite differences."""
    c_range = objective.c_max - objective.c_min
    base = np.array([evaluation.c1, evaluation.c2], dtype=np.float64)
    width_margin = evaluation.c2 - evaluation.c1 - objective.min_width
    margins = (
        (evaluation.c1 - objective.c_min, width_margin),
        (width_margin, objective.c_max - evaluation.c2),
    )
    for component, (minus_margin, plus_margin) in enumerate(margins):
        h = min(1.0e-5 * c_range, 0.25 * minus_margin, 0.25 * plus_margin)
        if h <= 1.0e-12 * max(c_range, 1.0):
            root_print(
                objective.comm,
                f"HMINUS1_GRAD_CHECK i={component + 1} skipped=active_threshold_constraint",
            )
            continue
        c_minus = base.copy()
        c_plus = base.copy()
        c_minus[component] -= h
        c_plus[component] += h
        minus = evaluate_frozen_threshold_objective(
            objective,
            c_minus[0],
            c_minus[1],
            method=f"gradient_check_minus_{component + 1}",
        )
        plus = evaluate_frozen_threshold_objective(
            objective,
            c_plus[0],
            c_plus[1],
            method=f"gradient_check_plus_{component + 1}",
        )
        finite_difference = (plus.psi - minus.psi) / (2.0 * h)
        exact = float(evaluation.grad_psi[component])
        relative_error = abs(finite_difference - exact) / max(abs(finite_difference), abs(exact), 1.0e-30)
        root_print(
            objective.comm,
            f"HMINUS1_GRAD_CHECK i={component + 1} h={h:.6e} exact={exact:.12e} "
            f"finiteDifference={finite_difference:.12e} relativeError={relative_error:.6e}",
        )


def write_frozen_initialization_records(
        writer: csv.DictWriter | None,
        *,
        run_tag: str,
        objective: FrozenThresholdObjective,
        selected: FrozenThresholdEvaluation,
        leakage_cap: float,
        missing_cap: float,
) -> None:
    """Write every unique frozen objective solve to ``initialization.csv``."""
    if writer is None:
        return
    selected_id = objective.evaluation_id(selected)
    for eval_id, method, evaluation in objective.records:
        writer.writerow({
            "record": "frozen_threshold",
            "runTag": run_tag,
            "method": method,
            "eval_id": eval_id,
            "c1": evaluation.c1,
            "c2": evaluation.c2,
            "width": evaluation.c2 - evaluation.c1,
            "eps": evaluation.eps_phi,
            "psiHminus1": evaluation.psi,
            "residualHminus1": evaluation.residual_dual,
            "Lrel": evaluation.leakage_rel,
            "Mrel": evaluation.missing_rel,
            "activityAreaRel": evaluation.activity_area_rel,
            "gradPsi1": evaluation.grad_psi[0],
            "gradPsi2": evaluation.grad_psi[1],
            "gradientEvaluated": int(evaluation.gradient_evaluated),
            "feasibleGeometry": int(
                evaluation.leakage_rel <= leakage_cap + 1.0e-14
                and evaluation.missing_rel <= missing_cap + 1.0e-14
            ),
            "acceptedThresholdStep": int(eval_id == selected_id),
            "elapsed": evaluation.evaluation_time,
        })


class ReusableLinearSolver:
    """Reassemble a changing matrix into fixed PETSc storage and reuse KSP."""

    def __init__(
            self,
            bilinear_form,
            V,
            bcs: list,
            *,
            prefix: str,
            solver: str,
            ksp_type: str | None,
            rtol: float,
            atol: float,
            max_it: int | None,
            verbosity: int,
    ) -> None:
        self.form = fem.form(bilinear_form)
        self.bcs = bcs
        self.matrix = fem_petsc.create_matrix(self.form)
        self.rhs = fem_petsc.create_vector(V)
        self.ksp = PETSc.KSP().create(V.mesh.comm)
        self.ksp.setOptionsPrefix(prefix)
        opts = PETSc.Options()
        for key, value in solver_options(solver, ksp_type=ksp_type).items():
            opts[f"{prefix}{key}"] = value
        if solver not in {"mumps", "lu"}:
            opts[f"{prefix}ksp_rtol"] = rtol
            opts[f"{prefix}ksp_atol"] = atol
            if max_it is not None:
                opts[f"{prefix}ksp_max_it"] = max_it
        self.ksp.setFromOptions()
        self._closed = False

    def _assemble_matrix(self) -> None:
        self.matrix.zeroEntries()
        fem_petsc.assemble_matrix(self.matrix, self.form, bcs=self.bcs)
        self.matrix.assemble()
        self.ksp.setOperators(self.matrix)

    def _solve(self, target: fem.Function, start: float) -> tuple[int, float, float]:
        self.ksp.solve(self.rhs, target.x.petsc_vec)
        target.x.scatter_forward()
        reason = self.ksp.getConvergedReason()
        if reason < 0:
            raise RuntimeError(f"reusable linear solve failed with PETSc reason {reason}")
        return (
            int(self.ksp.getIterationNumber()),
            float(self.ksp.getResidualNorm()),
            time.perf_counter() - start,
        )

    def solve_form(self, rhs_form, target: fem.Function) -> tuple[int, float, float]:
        """Reassemble the matrix and a variational RHS, then solve."""
        start = time.perf_counter()
        self._assemble_matrix()
        zero_vector(self.rhs)
        fem_petsc.assemble_vector(self.rhs, rhs_form)
        fem_petsc.apply_lifting(self.rhs, [self.form], [self.bcs])
        self.rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(self.rhs, self.bcs)
        return self._solve(target, start)

    def solve_vector(
            self,
            source: PETSc.Vec,
            target: fem.Function,
            *,
            scale: float = 1.0,
    ) -> tuple[int, float, float]:
        """Reassemble the matrix and solve from an existing assembled RHS."""
        start = time.perf_counter()
        self._assemble_matrix()
        source.copy(self.rhs)
        if scale != 1.0:
            self.rhs.scale(scale)
        fem_petsc.set_bc(self.rhs, self.bcs)
        return self._solve(target, start)

    def close(self) -> None:
        if self._closed:
            return
        self.ksp.destroy()
        self.matrix.destroy()
        self.rhs.destroy()
        self._closed = True


def solve_vector_from_vector(
        metric_form,
        rhs: PETSc.Vec,
        bcs: list,
        *,
        comm: MPI.Comm,
        prefix: str,
        solver: str,
        ksp_type: str | None,
        rtol: float,
        atol: float,
        max_it: int | None,
) -> tuple[PETSc.Vec, int, float, float]:
    """Solve a variational metric system for an already assembled RHS vector.

    This helper is used for dual residual norms.  Given a metric form ``A`` and
    a residual vector ``r``, it returns ``x = A^{-1} r`` so the caller can
    compute ``sqrt(r dot x)``.  The solver options are intentionally identical
    to the scalar variational solves used elsewhere in the script.

    Args:
        metric_form: UFL bilinear form defining the metric matrix.
        rhs: PETSc RHS vector.  It is copied before boundary conditions are
            imposed, so the input vector remains valid for the caller.
        bcs: Boundary conditions for the metric solve.
        comm: MPI communicator for PETSc object creation.
        prefix: PETSc options prefix.
        solver: Solver family name accepted by ``solver_options``.
        ksp_type: Optional PETSc KSP type override.
        rtol: Relative tolerance for iterative solvers.
        atol: Absolute tolerance for iterative solvers.
        max_it: Optional iteration cap for iterative solvers.

    Returns:
        Tuple ``(solution_vec, iterations, residual_norm, elapsed_seconds)``.
        The returned vector is owned by the caller.

    Raises:
        RuntimeError: If PETSc reports a negative convergence reason.
    """
    start = time.perf_counter()
    mat = assemble_matrix_form(metric_form, bcs)
    b = rhs.copy()
    fem_petsc.set_bc(b, bcs)
    x = b.duplicate()
    x.set(0.0)
    ksp = PETSc.KSP().create(comm)
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
    ksp.setOperators(mat)
    ksp.solve(b, x)
    reason = ksp.getConvergedReason()
    its = int(ksp.getIterationNumber())
    residual = float(ksp.getResidualNorm())
    elapsed = time.perf_counter() - start
    ksp.destroy()
    mat.destroy()
    b.destroy()
    if reason < 0:
        x.destroy()
        raise RuntimeError(f"vector solve {prefix!r} failed with PETSc reason {reason}")
    return x, its, residual, elapsed


def solve_same_matrix_forms(
        bilinear_form,
        rhs_forms: list,
        targets: list[fem.Function],
        bcs: list,
        *,
        prefix: str,
        solver: str,
        ksp_type: str | None,
        rtol: float,
        atol: float,
        max_it: int | None,
        verbosity: int,
) -> tuple[tuple[int, ...], tuple[float, ...], float, float, float, float]:
    """Solve several linear variational problems using one shared matrix.

    The sensitivity equations for ``c1`` and ``c2`` have the same Newton
    matrix and different right-hand sides.  Reusing the matrix is both faster
    and closer to the reduced algorithm, which says one Newton factorization
    plus two sensitivity solves.  This helper also returns timing components
    so verbosity level 2 can identify matrix assembly, RHS assembly, and solve
    costs separately.

    Args:
        bilinear_form: Shared UFL bilinear form, usually the Newton matrix.
        rhs_forms: Linear RHS forms, one per target function.
        targets: Functions receiving the solved coefficient vectors.
        bcs: Boundary conditions for all solves.
        prefix: PETSc options prefix shared by this matrix/solver.
        solver: Solver family name accepted by ``solver_options``.
        ksp_type: Optional PETSc KSP type override.
        rtol: Relative tolerance for iterative solvers.
        atol: Absolute tolerance for iterative solvers.
        max_it: Optional iteration cap for iterative solvers.
        verbosity: Current script verbosity.  Kept for API symmetry even
            though detailed KSP monitors are not enabled in this runner.

    Returns:
        Tuple containing per-RHS iteration counts, per-RHS solver residuals,
        total elapsed time, matrix assembly time, RHS assembly time, and
        accumulated linear solve time.

    Raises:
        RuntimeError: If any PETSc solve fails.
    """
    start = time.perf_counter()
    a_form = fem.form(bilinear_form)
    matrix_start = time.perf_counter()
    mat = fem_petsc.assemble_matrix(a_form, bcs=bcs)
    mat.assemble()
    matrix_assembly_time = time.perf_counter() - matrix_start
    ksp = PETSc.KSP().create(targets[0].function_space.mesh.comm)
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
    ksp.setOperators(mat)

    iterations: list[int] = []
    residuals: list[float] = []
    rhs_assembly_time = 0.0
    linear_solve_time = 0.0
    for i, (rhs_form, target) in enumerate(zip(rhs_forms, targets, strict=True)):
        rhs_start = time.perf_counter()
        b = fem_petsc.assemble_vector(fem.form(rhs_form))
        fem_petsc.apply_lifting(b, [a_form], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(b, bcs)
        rhs_assembly_time += time.perf_counter() - rhs_start
        solve_start = time.perf_counter()
        ksp.solve(b, target.x.petsc_vec)
        target.x.scatter_forward()
        linear_solve_time += time.perf_counter() - solve_start
        reason = ksp.getConvergedReason()
        iterations.append(int(ksp.getIterationNumber()))
        residuals.append(float(ksp.getResidualNorm()))
        b.destroy()
        if reason < 0:
            ksp.destroy()
            mat.destroy()
            raise RuntimeError(f"linear solve {prefix}{i} failed with PETSc reason {reason}")

    elapsed = time.perf_counter() - start
    ksp.destroy()
    mat.destroy()
    return (
        tuple(iterations),
        tuple(residuals),
        elapsed,
        matrix_assembly_time,
        rhs_assembly_time,
        linear_solve_time,
    )


def residual_norm(
        residual_form,
        bc,
        *,
        comm: MPI.Comm,
        mode: str,
        metric_form,
        solver: str,
        ksp_type: str | None,
        rtol: float,
        atol: float,
        max_it: int | None,
        prefix: str,
        stiffness_solver: FixedStiffnessSolver | None = None,
) -> float:
    """Compute the nonlinear residual norm requested by the CLI.

    The cheap Euclidean norm is useful while iterating quickly.  The dual norm
    ``sqrt(r^T K^{-1} r)`` matches the residual merit function in the
    algorithm note and is available through ``--residual-norm dual``.

    Args:
        residual_form: UFL residual form.
        bc: Homogeneous Dirichlet boundary condition.
        comm: MPI communicator.
        mode: ``"euclidean"`` or ``"dual"``.
        metric_form: Bilinear form for the dual metric, usually stiffness.
        solver: Solver family used for the metric solve in dual mode.
        ksp_type: Optional PETSc KSP type override.
        rtol: Relative tolerance for the metric solve.
        atol: Absolute tolerance for the metric solve.
        max_it: Optional iteration cap for the metric solve.
        prefix: PETSc options prefix.

    Returns:
        Scalar residual norm.
    """
    if stiffness_solver is not None:
        return stiffness_solver.residual_norm(residual_form, mode)
    vec = assemble_residual_vector(residual_form, bc)
    if mode == "euclidean":
        norm = float(vec.norm())
        vec.destroy()
        return norm
    dual_vec, _, _, _ = solve_vector_from_vector(
        metric_form,
        vec,
        [bc],
        comm=comm,
        prefix=prefix,
        solver=solver,
        ksp_type=ksp_type,
        rtol=rtol,
        atol=atol,
        max_it=max_it,
    )
    value = max(float(vec.dot(dual_vec)), 0.0)
    vec.destroy()
    dual_vec.destroy()
    return math.sqrt(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the reduced optimizer.

    The CLI intentionally follows the naming conventions of the existing
    DOLFINx torsion-initialized Newton scripts.  Defaults match the star-shaped
    mesh generation and torsion initializer used in the current experiments,
    while options are exposed for the reduced optimizer controls, Newton
    globalization, plotting, and logging.

    Args:
        argv: Optional argument list.  ``None`` means use ``sys.argv``.

    Returns:
        Parsed ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--equilibrium-output",
        type=Path,
        default=None,
        help="portable final phi/rho checkpoint; defaults to RUN_DIR/out/equilibrium.npz",
    )
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
    parser.add_argument(
        "--eps-t-ratio",
        dest="eps_t_ratio",
        type=float,
        default=None,
        help="torsion-window smoothing ratio used only by --init-mode legacy",
    )
    parser.add_argument("--rho-amp", type=float, default=None)
    parser.add_argument("--eps-mode", choices=("relative", "fixed"), default="relative")
    parser.add_argument("--eps-ratio", "--eps-phi-ratio", dest="eps_ratio", type=float, default=0.08)
    parser.add_argument("--eps-phi", type=float, default=None)
    parser.add_argument("--c1-phi", dest="c1_phi", type=float, default=None)
    parser.add_argument("--c2-phi", dest="c2_phi", type=float, default=None)
    parser.add_argument(
        "--initial-equilibrium",
        type=Path,
        default=None,
        help=(
            "portable optimizer equilibrium used as an automatic initial state; "
            "its stored thresholds are corrected on the current mesh before optimization"
        ),
    )
    parser.add_argument(
        "--init-mode",
        choices=("homotopy", "legacy"),
        default="homotopy",
        help=(
            "'homotopy' minimizes the frozen H^-1 residual and continues the sharp torsion source "
            "at fixed thresholds; 'legacy' preserves candidate generation/direct projection/ranking"
        ),
    )
    parser.add_argument(
        "--init-search",
        choices=("full", "fast"),
        default="full",
        help=(
            "full preserves the 20x20 plus refined frozen H^-1 search; fast uses "
            "window-fit/quantile seeds, 7x7 guarded scans, and value-only screening"
        ),
    )
    parser.add_argument(
        "--init-fallback",
        choices=("none", "window-fit"),
        default="none",
        help="after failed H^-1-seeded homotopy, retry once from the fitted window in the same run",
    )
    parser.add_argument("--init-hminus1-grid", type=int, default=20)
    parser.add_argument("--init-hminus1-refine-grid", type=int, default=11)
    parser.add_argument("--init-hminus1-refine-passes", type=int, default=2)
    parser.add_argument(
        "--init-hminus1-progress-every",
        type=int,
        default=10,
        help="at verbosity 2, report frozen H^-1 grid-search progress every N candidates",
    )
    parser.add_argument(
        "--init-hminus1-max-it",
        type=int,
        default=20,
        help="upper bound on local frozen-H^-1 refinement passes",
    )
    parser.add_argument("--init-leakage-cap", type=float, default=0.05)
    parser.add_argument("--init-missing-cap", type=float, default=0.10)
    parser.add_argument("--init-geom-relax-factor", type=float, default=1.5)
    parser.add_argument(
        "--verify-homotopy-init",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run finite-difference frozen-gradient and predictor-order checks",
    )
    parser.add_argument("--homotopy-initial-step", type=float, default=0.20)
    parser.add_argument("--homotopy-min-step", type=float, default=1.0e-3)
    parser.add_argument("--homotopy-max-step", type=float, default=0.50)
    parser.add_argument("--homotopy-step-grow", type=float, default=1.5)
    parser.add_argument("--homotopy-step-shrink", type=float, default=0.5)
    parser.add_argument("--homotopy-max-stages", type=int, default=64)
    parser.add_argument("--homotopy-tol-res", type=float, default=None)
    parser.add_argument("--homotopy-predictor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--threshold-optimization-start-lambda",
        type=float,
        default=1.0,
        help=(
            "diagnostic source-homotopy value at which reduced threshold optimization begins; "
            "values below one hold lambda fixed during optimization and do not certify the "
            "original lambda=1 equilibrium"
        ),
    )
    parser.add_argument("--include-fit-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--legacy-project-preselected-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="project only the automatically preselected legacy window candidate",
    )
    parser.add_argument("--fit-window-grid", type=int, default=64)
    parser.add_argument("--fit-window-refine-grid", type=int, default=25)
    parser.add_argument("--fit-window-refine-passes", type=int, default=2)
    parser.add_argument("--fit-window-bins", type=int, default=4096)
    parser.add_argument("--fit-window-quad-degree", type=int, default=None)
    parser.add_argument("--cmax-factor", type=float, default=1.25)
    parser.add_argument("--c-lower-fraction", type=float, default=0.0)
    parser.add_argument("--c-upper-fraction", type=float, default=1.0)
    parser.add_argument("--min-width-fraction", type=float, default=1.0e-3)
    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument("--delta-c", type=float, default=0.0)
    parser.add_argument("--max-opt-it", type=int, default=30)
    parser.add_argument("--eta-out", type=float, default=2.0e-2)
    parser.add_argument("--tol-area", type=float, default=5.0e-2)
    parser.add_argument(
        "--threshold-objective-mode",
        choices=("filter", "weighted"),
        default="filter",
        help=(
            "filter retains the leakage-first lexicographic policy; weighted "
            "minimizes a finite normalized leakage/missing merit"
        ),
    )
    parser.add_argument(
        "--threshold-leakage-weight",
        type=float,
        default=1.25,
        help="positive leakage weight in --threshold-objective-mode weighted",
    )
    parser.add_argument(
        "--threshold-missing-weight",
        type=float,
        default=1.0,
        help="positive missing-area weight in --threshold-objective-mode weighted",
    )
    parser.add_argument("--tol-grad", type=float, default=1.0e-8)
    parser.add_argument("--tol-res", type=float, default=1.0e-10)
    parser.add_argument("--inner-newton-tol", type=float, default=None)
    parser.add_argument("--inner-tol-max", type=float, default=1.0e-5)
    parser.add_argument("--inner-tol-gamma", type=float, default=1.0e-6)
    parser.add_argument(
        "--run-inexact-newton-study",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="replay selected non-final predictor states and write logs/inexact_newton.csv",
    )
    parser.add_argument(
        "--inexact-newton-tolerances",
        default="1e-2,1e-3,1e-5,1e-7,adaptive",
        help="comma-separated replay residual tolerances; 'adaptive' uses the production rule",
    )
    parser.add_argument("--inexact-newton-reference-tol", type=float, default=1.0e-12)
    parser.add_argument("--inexact-newton-max-snapshots", type=int, default=4)
    parser.add_argument(
        "--inexact-newton-output",
        type=Path,
        default=None,
        help="replay CSV path; defaults to RUN_DIR/logs/inexact_newton.csv",
    )
    parser.add_argument("--max-newton-it", type=int, default=40)
    parser.add_argument(
        "--newton-soft-cap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "treat --max-newton-it as an initial budget and extend it in bounded "
            "chunks while recent finite residuals clearly contract"
        ),
    )
    parser.add_argument(
        "--newton-soft-cap-chunk",
        type=int,
        default=20,
        help="additional iterations granted at each contracting-residual cap",
    )
    parser.add_argument(
        "--newton-soft-cap-factor",
        type=float,
        default=4.0,
        help="hard iteration ceiling as a multiple of the initial budget",
    )
    parser.add_argument(
        "--newton-soft-cap-window",
        type=int,
        default=4,
        help="number of recent residual contractions used at a budget boundary",
    )
    parser.add_argument(
        "--newton-soft-cap-contraction",
        type=float,
        default=0.98,
        help="largest accepted geometric-mean residual ratio (strictly below one)",
    )
    parser.add_argument("--final-newton-tol-res", type=float, default=None)
    parser.add_argument("--final-newton-max-it", type=int, default=200)
    parser.add_argument("--min-certified-area-fraction", type=float, default=0.50)
    parser.add_argument("--tol-step", type=float, default=1.0e-10)
    parser.add_argument("--max-backtrack", type=int, default=24)
    parser.add_argument("--alpha-min", type=float, default=1.0e-8)
    parser.add_argument("--beta-ls", type=float, default=0.5)
    parser.add_argument("--armijo-c", type=float, default=1.0e-6)
    parser.add_argument("--residual-norm", choices=("euclidean", "dual"), default="euclidean")
    parser.add_argument("--trust-radius", type=float, default=8.0e-2, help="initial trust radius as a fraction of c-scale")
    parser.add_argument("--trust-radius-min", type=float, default=1.0e-5)
    parser.add_argument("--trust-radius-max", type=float, default=0.30)
    parser.add_argument("--trust-shrink", type=float, default=0.5)
    parser.add_argument("--trust-grow", type=float, default=1.8)
    parser.add_argument("--qp-hessian-floor", type=float, default=1.0e-14)
    parser.add_argument("--leakage-filter-slack", type=float, default=1.0e-4)
    parser.add_argument("--accept-sufficient-decrease", type=float, default=1.0e-10)
    parser.add_argument("--eta-overlap", type=float, default=0.10)
    parser.add_argument("--disable-branch-check", action="store_true")
    parser.add_argument("--min-activity-fraction", type=float, default=1.0e-2)
    parser.add_argument(
        "--solver-preset",
        choices=("optimized", "legacy"),
        default="optimized",
        help=(
            "optimized uses CG/Hypre for fixed SPD systems and accepted-state "
            "sensitivities, GMRES/Hypre for changing Newton systems, and MUMPS "
            "for final certification; legacy uses --linear-solver for every phase"
        ),
    )
    parser.add_argument(
        "--linear-solver",
        choices=("mumps", "lu", "hypre", "gamg"),
        default=None,
        help="global solver override; supplying it disables phase-specific optimized defaults",
    )
    parser.add_argument("--stiffness-linear-solver", choices=("mumps", "lu", "hypre", "gamg"), default=None)
    parser.add_argument("--homotopy-linear-solver", choices=("mumps", "lu", "hypre", "gamg"), default=None)
    parser.add_argument("--nonlinear-linear-solver", choices=("mumps", "lu", "hypre", "gamg"), default=None)
    parser.add_argument("--sensitivity-linear-solver", choices=("mumps", "lu", "hypre", "gamg"), default=None)
    parser.add_argument("--final-linear-solver", choices=("mumps", "lu", "hypre", "gamg"), default=None)
    parser.add_argument(
        "--sensitivity-iterative-fallback",
        choices=("none", "gmres"),
        default="gmres",
        help="KSP fallback when CG detects a non-SPD accepted-state sensitivity Jacobian",
    )
    parser.add_argument(
        "--iterative-fallback-solver",
        choices=("none", "mumps", "lu"),
        default="mumps",
        help="final direct fallback for failed changing-Jacobian iterative solves",
    )
    parser.add_argument("--ksp-type", default=None)
    parser.add_argument("--linear-rtol", type=float, default=1.0e-10)
    parser.add_argument("--linear-atol", type=float, default=1.0e-12)
    parser.add_argument("--linear-max-it", type=int, default=None)
    parser.add_argument(
        "--trial-linear-max-it",
        type=int,
        default=300,
        help=(
            "maximum Krylov iterations for each outer trial Newton correction; "
            "hitting the cap rejects the trial without invoking the direct fallback"
        ),
    )
    parser.add_argument(
        "--diagnose-capped-trial-coercivity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "on a trial Krylov iteration-cap failure, run one stiffness-relative "
            "Jacobian coercivity eigensolve and record it in newton_spd.csv"
        ),
    )
    parser.add_argument(
        "--check-newton-spd",
        action="store_true",
        help="measure the stiffness-relative smallest eigenvalue of every Newton Jacobian",
    )
    parser.add_argument(
        "--newton-spd-inertia",
        action="store_true",
        help="also factor each checked Jacobian with symmetric MUMPS and record its inertia",
    )
    parser.add_argument("--newton-spd-every", type=int, default=1)
    parser.add_argument(
        "--newton-spd-prefix-filter",
        default=None,
        help="only diagnose Newton solves whose prefix contains this text",
    )
    parser.add_argument("--newton-spd-eig-tol", type=float, default=1.0e-8)
    parser.add_argument("--newton-spd-eig-max-it", type=int, default=500)
    parser.add_argument("--newton-spd-zero-tol", type=float, default=1.0e-8)

    parser.add_argument(
        "--certified-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop after a certified subband ceases to improve materially",
    )
    parser.add_argument("--certified-stop-min-accepted", type=int, default=8)
    parser.add_argument("--certified-stop-patience", type=int, default=5)
    parser.add_argument(
        "--certified-stop-rtol",
        type=float,
        default=1.0e-3,
        help="relative improvement in leakage+missing needed to reset certified-stop patience",
    )
    parser.add_argument(
        "--retain-best-certified",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="finalize from the certified state with minimum leakage+missing rather than the last iterate",
    )
    parser.add_argument("--terminal-every", type=int, default=1)
    parser.add_argument(
        "--save-terminal-log",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "mirror stdout and stderr from every MPI rank to "
            "RUN_DIR/out/terminal.log, preserving partial output on "
            "Ctrl+C, SIGTERM, and SIGHUP"
        ),
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="0=essential output, 1=iteration summaries, 2=algorithm step/timing detail",
    )
    parser.add_argument("--fail-on-nonconvergence", action="store_true")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="show full-domain PyVista plots on rank zero; MPI partitions are gathered collectively",
    )
    parser.add_argument(
        "--plot-mode",
        choices=("blocking", "nonblocking"),
        default="blocking",
        help="blocking waits on rank zero while other MPI ranks synchronize; nonblocking updates one live window",
    )
    parser.add_argument("--plot-off-screen", action="store_true")
    parser.add_argument("--plot-window-width", type=int, default=1800)
    parser.add_argument("--plot-window-height", type=int, default=700)
    parser.add_argument("--plot-design", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-optimization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-severe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--plot-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--frame-dir", type=Path, default=None)
    parser.add_argument("--frame-every", type=int, default=None)
    parser.add_argument("--frame-design", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frame-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frame-window-width", type=int, default=1800)
    parser.add_argument("--frame-window-height", type=int, default=700)
    parser.add_argument(
        "--save-trajectory",
        action="store_true",
        help="archive selected optimizer states for publication figures and movies",
    )
    parser.add_argument(
        "--trajectory-every",
        type=int,
        default=1,
        help="record every Nth accepted homotopy/outer state",
    )
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        default=None,
        help="trajectory NPZ path; defaults to RUN_DIR/out/trajectory.npz",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Validate cross-argument constraints after parsing.

    ``argparse`` checks types and simple choices, but the optimizer has
    additional mathematical constraints: positive smoothing widths, ordered
    torsion thresholds, a nonempty certified band, positive trust radii, and a
    valid generated star shape.  Failing fast here avoids long DOLFINx setup
    before detecting an impossible run configuration.

    Args:
        args: Parsed command-line namespace.

    Raises:
        ValueError: If a supplied option combination is inconsistent.
    """
    if args.eps_mode == "relative" and args.eps_ratio <= 0.0:
        raise ValueError("require positive --eps-ratio")
    if args.eps_mode == "fixed" and (args.eps_phi is None or args.eps_phi <= 0.0):
        raise ValueError("require positive --eps-phi when --eps-mode fixed")
    if args.homotopy_initial_step <= 0.0 or args.homotopy_min_step <= 0.0 or args.homotopy_max_step <= 0.0:
        raise ValueError("require positive homotopy step sizes")
    if args.homotopy_min_step > args.homotopy_max_step:
        raise ValueError("require homotopy-min-step <= homotopy-max-step")
    if not (0.0 < args.homotopy_step_shrink < 1.0):
        raise ValueError("require 0 < --homotopy-step-shrink < 1")
    if args.homotopy_step_grow <= 1.0:
        raise ValueError("require --homotopy-step-grow > 1")
    if args.homotopy_max_stages < 1:
        raise ValueError("require positive --homotopy-max-stages")
    if args.homotopy_tol_res is not None and args.homotopy_tol_res <= 0.0:
        raise ValueError("require positive --homotopy-tol-res")
    if not (0.0 < args.threshold_optimization_start_lambda <= 1.0):
        raise ValueError("require 0 < --threshold-optimization-start-lambda <= 1")
    if args.threshold_optimization_start_lambda < 1.0 and args.init_mode != "homotopy":
        raise ValueError("partial-lambda threshold optimization requires --init-mode homotopy")
    if args.init_hminus1_grid < 2 or args.init_hminus1_refine_grid < 2:
        raise ValueError("require --init-hminus1-grid and --init-hminus1-refine-grid >= 2")
    if args.init_hminus1_refine_passes < 0 or args.init_hminus1_max_it < 1:
        raise ValueError("require nonnegative refinement passes and positive --init-hminus1-max-it")
    if args.init_hminus1_progress_every < 1:
        raise ValueError("require positive --init-hminus1-progress-every")
    if args.init_leakage_cap < 0.0 or args.init_missing_cap < 0.0:
        raise ValueError("require nonnegative frozen geometric caps")
    if args.init_geom_relax_factor <= 1.0:
        raise ValueError("require --init-geom-relax-factor > 1")
    if args.cmax_factor <= 0.0:
        raise ValueError("require positive --cmax-factor")
    if float(args.star_r0) <= abs(float(args.star_amp)):
        raise ValueError("star-shaped mesh generation requires --star-r0 > abs(--star-amp)")
    if args.c_lower_fraction < 0.0 or args.c_upper_fraction <= args.c_lower_fraction:
        raise ValueError("require 0 <= c-lower-fraction < c-upper-fraction")
    if args.min_width_fraction <= 0.0:
        raise ValueError("require positive --min-width-fraction")
    if args.kappa < 0.0:
        raise ValueError("require nonnegative --kappa")
    if args.delta_c < 0.0:
        raise ValueError("require nonnegative --delta-c")
    if args.max_opt_it < 0:
        raise ValueError("require nonnegative --max-opt-it")
    if args.eta_out < 0.0:
        raise ValueError("require nonnegative --eta-out")
    if args.tol_area < 0.0:
        raise ValueError("require nonnegative --tol-area")
    if (
        not math.isfinite(args.threshold_leakage_weight)
        or args.threshold_leakage_weight <= 0.0
        or not math.isfinite(args.threshold_missing_weight)
        or args.threshold_missing_weight <= 0.0
    ):
        raise ValueError("threshold objective weights must be positive finite values")
    if args.tol_res <= 0.0 or args.inner_tol_max <= 0.0 or args.inner_tol_gamma <= 0.0:
        raise ValueError("require positive residual tolerances")
    if args.inner_newton_tol is not None and args.inner_newton_tol <= 0.0:
        raise ValueError("require positive --inner-newton-tol")
    if args.inexact_newton_reference_tol <= 0.0:
        raise ValueError("require positive --inexact-newton-reference-tol")
    if args.inexact_newton_max_snapshots < 1:
        raise ValueError("require positive --inexact-newton-max-snapshots")
    inexact_tokens = [
        token.strip().lower()
        for token in str(args.inexact_newton_tolerances).split(",")
        if token.strip()
    ]
    if not inexact_tokens:
        raise ValueError("--inexact-newton-tolerances cannot be empty")
    for token in inexact_tokens:
        if token == "adaptive":
            continue
        try:
            tolerance = float(token)
        except ValueError as exc:
            raise ValueError(
                f"invalid inexact Newton tolerance {token!r}"
            ) from exc
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("inexact Newton tolerances must be finite and positive")
    if args.max_newton_it < 1:
        raise ValueError("require positive --max-newton-it")
    if args.newton_soft_cap_chunk < 1:
        raise ValueError("require positive --newton-soft-cap-chunk")
    if not math.isfinite(args.newton_soft_cap_factor) or args.newton_soft_cap_factor < 1.0:
        raise ValueError("require finite --newton-soft-cap-factor >= 1")
    if args.newton_soft_cap_window < 2:
        raise ValueError("require --newton-soft-cap-window >= 2")
    if not math.isfinite(args.newton_soft_cap_contraction) or not (0.0 < args.newton_soft_cap_contraction < 1.0):
        raise ValueError("require 0 < --newton-soft-cap-contraction < 1")
    if args.linear_max_it is not None and args.linear_max_it < 1:
        raise ValueError("require positive --linear-max-it")
    if args.trial_linear_max_it < 1:
        raise ValueError("require positive --trial-linear-max-it")
    if args.final_newton_tol_res is not None and args.final_newton_tol_res <= 0.0:
        raise ValueError("require positive --final-newton-tol-res")
    if args.final_newton_max_it < 1:
        raise ValueError("require positive --final-newton-max-it")
    if not (0.0 <= args.min_certified_area_fraction <= 1.0):
        raise ValueError("require 0 <= --min-certified-area-fraction <= 1")
    if args.max_backtrack < 0:
        raise ValueError("require nonnegative --max-backtrack")
    if not (0.0 < args.beta_ls < 1.0):
        raise ValueError("require 0 < --beta-ls < 1")
    if args.trust_radius <= 0.0 or args.trust_radius_min <= 0.0 or args.trust_radius_max <= 0.0:
        raise ValueError("require positive trust radii")
    if args.trust_radius_min > args.trust_radius_max:
        raise ValueError("require trust-radius-min <= trust-radius-max")
    if args.trust_shrink <= 0.0 or args.trust_shrink >= 1.0:
        raise ValueError("require 0 < --trust-shrink < 1")
    if args.trust_grow <= 1.0:
        raise ValueError("require --trust-grow > 1")
    if args.eta_overlap < 0.0:
        raise ValueError("require nonnegative --eta-overlap")
    if args.min_activity_fraction < 0.0:
        raise ValueError("require nonnegative --min-activity-fraction")
    if args.newton_spd_every < 1:
        raise ValueError("require positive --newton-spd-every")
    if args.newton_spd_eig_tol <= 0.0:
        raise ValueError("require positive --newton-spd-eig-tol")
    if args.newton_spd_eig_max_it < 1:
        raise ValueError("require positive --newton-spd-eig-max-it")
    if args.newton_spd_zero_tol < 0.0:
        raise ValueError("require nonnegative --newton-spd-zero-tol")
    if args.certified_stop_min_accepted < 0:
        raise ValueError("require nonnegative --certified-stop-min-accepted")
    if args.certified_stop_patience < 1:
        raise ValueError("require positive --certified-stop-patience")
    if not (0.0 <= args.certified_stop_rtol < 1.0):
        raise ValueError("require 0 <= --certified-stop-rtol < 1")
    if args.trajectory_every < 1:
        raise ValueError("require positive --trajectory-every")
    if args.solver_preset == "legacy" and args.linear_solver is None:
        args.linear_solver = "mumps"
    if (args.c1_phi is None) != (args.c2_phi is None):
        raise ValueError("--c1-phi and --c2-phi must be supplied together")
    if args.c1_phi is not None and not (args.c2_phi > args.c1_phi >= 0.0):
        raise ValueError("require 0 <= c1_phi < c2_phi")
    if args.initial_equilibrium is not None:
        args.initial_equilibrium = resolve_archive_path(args.initial_equilibrium).resolve()
        if not args.initial_equilibrium.is_file():
            raise ValueError(f"--initial-equilibrium does not exist: {args.initial_equilibrium}")
        if args.c1_phi is not None:
            raise ValueError("--initial-equilibrium cannot be combined with manual --c1-phi/--c2-phi")
        if args.init_mode != "homotopy":
            raise ValueError("--initial-equilibrium currently requires --init-mode homotopy")


def phase_solver_args(args: argparse.Namespace, phase: str) -> argparse.Namespace:
    """Return a copy of the run arguments with a phase-specific solver.

    An explicit ``--linear-solver`` remains a global override for reproducible
    legacy runs. Otherwise the optimized preset keeps fixed coercive systems
    and accepted-state sensitivities on CG/BoomerAMG, uses GMRES/BoomerAMG for
    changing Jacobians that can become indefinite away from the accepted
    branch, and reserves MUMPS for final certification.
    """
    if phase not in {"stiffness", "homotopy", "nonlinear", "sensitivity", "final"}:
        raise ValueError(f"unknown solver phase {phase!r}")
    configured = argparse.Namespace(**vars(args))
    if args.linear_solver is not None:
        solver = str(args.linear_solver)
    else:
        defaults = {
            "stiffness": "hypre",
            "homotopy": "hypre",
            "nonlinear": "hypre",
            "sensitivity": "hypre",
            "final": "mumps",
        }
        override = getattr(args, f"{phase}_linear_solver")
        solver = str(override or defaults[phase])
    configured.linear_solver = solver
    if args.ksp_type is not None:
        configured.ksp_type = args.ksp_type
    elif solver in {"mumps", "lu"}:
        configured.ksp_type = None
    elif phase in {"stiffness", "sensitivity"}:
        configured.ksp_type = "cg"
    else:
        configured.ksp_type = "gmres"
    return configured


def fallback_solver_name(args: argparse.Namespace) -> str | None:
    """Return the configured direct fallback for an iterative phase."""
    if args.linear_solver in {"mumps", "lu"}:
        return None
    fallback = str(args.iterative_fallback_solver)
    return None if fallback == "none" else fallback


def petsc_failure_reason(error: RuntimeError) -> int | None:
    """Extract the integer PETSc convergence reason from a solve error."""
    marker = "PETSc reason "
    message = str(error)
    if marker not in message:
        return None
    try:
        return int(message.rsplit(marker, 1)[1].split()[0])
    except (IndexError, ValueError):
        return None


def sensitivity_iterative_fallback_ksp(args: argparse.Namespace) -> str | None:
    """Return the guarded iterative fallback for a sensitivity CG solve."""
    if args.linear_solver in {"mumps", "lu"} or args.ksp_type != "cg":
        return None
    fallback = str(args.sensitivity_iterative_fallback)
    return None if fallback == "none" else fallback


def inner_tolerance(args: argparse.Namespace, discrepancy_rel: float) -> float:
    """Choose the adaptive Newton tolerance for outer-loop state projections.

    By default the outer loop uses relaxed state solves while the geometric
    discrepancy is far above the requested leakage/missing-area envelope.  The
    tolerance is scaled by
    ``discrepancy_rel / (eta_out + tol_area)``, clipped by
    ``--inner-tol-max``, and never allowed below ``--tol-res``.  This keeps the
    first few reduced iterations from oversolving poor threshold pairs while
    still tightening the solve as the band becomes competitive.  Supplying
    ``--inner-newton-tol`` disables the adaptive rule and forces every inner
    projection and trial correction to that fixed residual tolerance; this is
    mainly a testing and verification knob.

    Args:
        args: Parsed command-line namespace.
        discrepancy_rel: Current ``leakage_rel + missing_rel``.

    Returns:
        Residual tolerance for the next fixed-threshold Newton projection.
    """
    if args.inner_newton_tol is not None:
        return float(args.inner_newton_tol)
    target_discrepancy = max(float(args.eta_out) + float(args.tol_area), 1.0e-14)
    discrepancy_scale = max(float(discrepancy_rel) / target_discrepancy, 1.0)
    loose = min(float(args.inner_tol_max), float(args.inner_tol_gamma) * discrepancy_scale)
    return max(float(args.tol_res), loose)


def final_newton_tolerance(args: argparse.Namespace) -> float:
    """Return the residual target for the final exact Newton projection.

    ``--tol-res`` remains the baseline residual tolerance used by the adaptive
    inner policy.  ``--final-newton-tol-res`` can be set independently when the
    outer loop should run with a looser baseline but the final projected state
    must be certified to a tighter residual.  If the final-specific option is
    omitted, the final projection uses ``--tol-res`` for backward-compatible
    behavior.

    Args:
        args: Parsed command-line namespace.

    Returns:
        Positive final Newton residual tolerance.
    """
    if args.final_newton_tol_res is not None:
        return float(args.final_newton_tol_res)
    return float(args.tol_res)


def certified_subband_success(
        *,
        metrics: BandMetrics,
        target_area: float,
        eta_out: float,
        min_area_fraction: float,
) -> tuple[bool, float, float]:
    """Check the practical certified-subband success criterion.

    The strict reduced objective tries to match the whole torsion-designed band:
    small soft leakage and small soft missing area.  Some semilinear branches
    cannot match that target well, but still produce a useful equilibrium band
    that is safely contained in the torsion band and has nontrivial area.  This
    helper separates that practical outcome from a true failure.

    The certified region is the interior plateau
    ``c1 + kappa*eps <= phi <= c2 - kappa*eps``.  It is intentionally stricter
    than the soft logistic activity used by the optimizer, so passing this
    check means the final state contains a genuine plateau sub-band rather than
    only transition-layer mass.

    Args:
        metrics: Final soft/certified band metrics.
        target_area: Crisp torsion target area.
        eta_out: Allowed certified leakage as a fraction of ``target_area``.
        min_area_fraction: Required certified area as a fraction of
            ``target_area``.

    Returns:
        ``(ok, certified_leakage_rel, certified_area_rel)``.
    """
    area_scale = max(float(target_area), 1.0e-30)
    certified_leakage_rel = metrics.certified_leakage / area_scale
    certified_area_rel = metrics.certified_area / area_scale
    ok = (
        certified_leakage_rel <= float(eta_out)
        and certified_area_rel >= float(min_area_fraction)
    )
    return ok, certified_leakage_rel, certified_area_rel


class HomotopySolveWorkspace:
    """Compiled forms and reusable nonlinear solver for one continuation."""

    def __init__(
            self,
            *,
            u: fem.Function,
            trial,
            test,
            dx,
            bc,
            target_density,
            c1_const: fem.Constant,
            c2_const: fem.Constant,
            eps_const: fem.Constant,
            rho_amp: float,
            args: argparse.Namespace,
    ) -> None:
        domain = u.function_space.mesh
        self.lambda_const = fem.Constant(domain, PETSc.ScalarType(0.0))
        self.c1_const = c1_const
        self.c2_const = c2_const
        self.eps_const = eps_const
        self.nonlinear_density = window_density_const_ufl(
            u,
            c1_const,
            c2_const,
            eps_const,
            rho_amp,
        )
        self.source_density = (
            (1.0 - self.lambda_const) * target_density
            + self.lambda_const * self.nonlinear_density
        )
        self.residual_expr = (
            ufl.inner(ufl.grad(u), ufl.grad(test)) - self.source_density * test
        ) * dx
        self.residual_form = fem.form(self.residual_expr)
        ws = window_s_derivative_activity_ufl(u, c1_const, c2_const, eps_const)
        self.jac_expr = (
            ufl.inner(ufl.grad(trial), ufl.grad(test))
            - self.lambda_const * float(rho_amp) * ws * trial * test
        ) * dx
        self.tangent_rhs_form = fem.form((self.nonlinear_density - target_density) * test * dx)
        interpolation_points = u.function_space.element.interpolation_points
        if callable(interpolation_points):
            interpolation_points = interpolation_points()
        self.density_expression = fem.Expression(self.source_density, interpolation_points)
        self.nonlinear_density_expression = fem.Expression(
            self.nonlinear_density,
            interpolation_points,
        )
        self.linear_solver = ReusableLinearSolver(
            self.jac_expr,
            u.function_space,
            [bc],
            prefix="init_homotopy_shared_",
            solver=args.linear_solver,
            ksp_type=args.ksp_type,
            rtol=args.linear_rtol,
            atol=args.linear_atol,
            max_it=args.linear_max_it,
            verbosity=args.verbosity,
        )

    def set_parameters(self, *, lam: float, c1: float, c2: float, eps_phi: float) -> None:
        self.lambda_const.value = PETSc.ScalarType(lam)
        self.c1_const.value = PETSc.ScalarType(c1)
        self.c2_const.value = PETSc.ScalarType(c2)
        self.eps_const.value = PETSc.ScalarType(eps_phi)

    def update_density(self, rho: fem.Function) -> None:
        rho.interpolate(self.density_expression)
        rho.x.scatter_forward()

    def update_nonlinear_density(self, rho: fem.Function) -> None:
        """Interpolate the ordinary lambda=1 density for candidate scoring."""
        rho.interpolate(self.nonlinear_density_expression)
        rho.x.scatter_forward()

    def close(self) -> None:
        self.linear_solver.close()


def write_newton_record(
        args: argparse.Namespace,
        comm: MPI.Comm,
        *,
        solve: str,
        phase: str,
        outer_iteration: int,
        homotopy_lambda: float,
        nonlinear_iteration: int,
        status: str,
        residual: float,
        damping: float = 0.0,
        backtracks: int = 0,
        step_norm: float = math.nan,
        ksp_iterations: int = 0,
        ksp_residual: float = math.nan,
        ksp_time: float = 0.0,
        assembly_time: float = 0.0,
        line_search_time: float = 0.0,
        linear_cap: bool = False,
        fallback_used: bool = False,
        newton_cap_extension: int = 0,
        newton_cap_reason: str = "",
        newton_contraction: float = math.nan,
        elapsed: float = 0.0,
) -> None:
    writer = getattr(args, "newton_writer", None)
    if comm.rank != 0 or writer is None:
        return
    writer.writerow({
        "solve": solve, "phase": phase, "outer_iteration": outer_iteration,
        "homotopy_lambda": homotopy_lambda, "nonlinear_iteration": nonlinear_iteration,
        "status": status, "residual": residual, "damping": damping,
        "backtracks": backtracks, "step_norm": step_norm,
        "ksp_iterations": ksp_iterations, "ksp_residual": ksp_residual,
        "ksp_time": ksp_time, "assembly_time": assembly_time,
        "line_search_time": line_search_time, "linear_cap": int(linear_cap),
        "fallback_used": int(fallback_used), "elapsed": elapsed,
        "newton_initial_budget": getattr(args, "_newton_initial_budget", ""),
        "newton_current_budget": getattr(args, "_newton_current_budget", ""),
        "newton_hard_ceiling": getattr(args, "_newton_hard_ceiling", ""),
        "newton_cap_extension": int(newton_cap_extension),
        "newton_cap_reason": newton_cap_reason,
        "newton_contraction": (
            newton_contraction
            if math.isfinite(float(newton_contraction))
            else ""
        ),
    })
    handle = getattr(args, "newton_handle", None)
    if handle is not None:
        handle.flush()


def solve_equilibrium(
        *,
        u: fem.Function,
        du: fem.Function,
        rho: fem.Function,
        trial,
        test,
        dx,
        bc,
        stiffness_form,
        c1_const: fem.Constant,
        c2_const: fem.Constant,
        eps_const: fem.Constant,
        c1: float,
        c2: float,
        eps_phi: float,
        rho_amp: float,
        tol_res: float,
        args: argparse.Namespace,
        prefix: str,
        homotopy_lambda: float = 1.0,
        homotopy_target_density=None,
        stiffness_solver: FixedStiffnessSolver | None = None,
        homotopy_workspace: HomotopySolveWorkspace | None = None,
        sync_density_on_return: bool = True,
        initial_residual: float | None = None,
        plot_callback: Callable[[str, int, float, float, int, float, float, float], None] | None = None,
        phase: str = "nonlinear",
        outer_iteration: int = -1,
) -> NewtonResult:
    """Project the current state onto the fixed-threshold semilinear branch.

    For fixed ``(c1,c2,eps)``, this solves the nonlinear finite-element
    residual.  In the standard case ``homotopy_lambda=1`` the source is
    ``rho_amp*W(u;c1,c2,eps)``.  During initialization continuation the source
    is

        (1-lambda)*rho_target + lambda*rho_amp*W(u;c1,c2,eps),

    where ``rho_target = rho_amp*1_{B_T}``.  The Jacobian is the exact
    derivative of this residual
    with respect to the finite-element state.  The line search uses residual
    decrease as a merit condition.  A small Newton step is treated as
    stagnation unless the residual tolerance has already been reached; this is
    deliberate because the algorithm requires true residual convergence for
    the final state.

    Args:
        u: State function updated in place.
        du: Work function receiving each Newton correction.
        rho: Work/output density function updated at interpolation points.
        trial: UFL trial function for the Newton matrix.
        test: UFL test function for residual/Jacobian forms.
        dx: UFL measure with the selected quadrature degree.
        bc: Homogeneous Dirichlet boundary condition.
        stiffness_form: Stiffness bilinear form used for dual residual norms.
        c1_const: Mutable DOLFINx constant for the lower threshold.
        c2_const: Mutable DOLFINx constant for the upper threshold.
        eps_const: Mutable DOLFINx constant for the smoothing width.
        c1: Lower threshold value for this solve.
        c2: Upper threshold value for this solve.
        eps_phi: Smoothing width for this solve.
        rho_amp: Density amplitude in the PDE right-hand side.
        tol_res: Required residual tolerance for this projection.
        args: Parsed command-line namespace controlling Newton and linear
            solver tolerances.
        prefix: PETSc options prefix stem for all linear solves and residual
            norm solves from this projection.
        homotopy_lambda: Source-continuation parameter in ``[0,1]``.  The
            default ``1`` recovers the original semilinear state equation.
        homotopy_target_density: UFL expression for the sharp target density.
            Required when ``homotopy_lambda < 1``.
        stiffness_solver: Optional run-scoped stiffness operator used for
            residual and H1 norms.
        homotopy_workspace: Optional compiled continuation forms and reusable
            Jacobian solver.
        sync_density_on_return: Whether to interpolate ``rho`` before
            returning.  Homotopy stages defer this until candidate scoring.
        initial_residual: Optional norm of an already assembled residual at
            the current state.  The homotopy predictor uses this to avoid
            assembling the same residual again before its Newton corrector.
        plot_callback: Optional hook called after each accepted Newton update.
            The callback receives ``(prefix, newton_iteration, residual,
            alpha, backtracks, c1, c2, eps_phi)``.  It is used only for severe
            plotting and must not change the numerical state.

    Returns:
        ``NewtonResult`` describing convergence, residual, damping, and solve
        time.
    """
    comm = u.function_space.mesh.comm
    solve_started = time.perf_counter()
    lam = float(homotopy_lambda)
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"homotopy_lambda must lie in [0,1], got {lam}")
    if lam < 1.0 and homotopy_target_density is None:
        raise ValueError("homotopy_target_density is required when homotopy_lambda < 1")
    if homotopy_workspace is not None:
        if stiffness_solver is None:
            raise ValueError("homotopy workspace requires the shared stiffness solver")
        homotopy_workspace.set_parameters(lam=lam, c1=c1, c2=c2, eps_phi=eps_phi)
        source_density = homotopy_workspace.source_density
        residual_expr = homotopy_workspace.residual_expr
        residual_form = homotopy_workspace.residual_form
        jac_expr = homotopy_workspace.jac_expr

        def sync_density() -> None:
            homotopy_workspace.update_density(rho)

    else:
        c1_const.value = PETSc.ScalarType(c1)
        c2_const.value = PETSc.ScalarType(c2)
        eps_const.value = PETSc.ScalarType(eps_phi)
        nonlinear_density = window_density_const_ufl(u, c1_const, c2_const, eps_const, rho_amp)
        source_density = (
            nonlinear_density
            if lam == 1.0
            else (1.0 - lam) * homotopy_target_density + lam * nonlinear_density
        )
        residual_expr = (
            ufl.inner(ufl.grad(u), ufl.grad(test)) - source_density * test
        ) * dx
        residual_form = fem.form(residual_expr)
        jac_expr = (
            ufl.inner(ufl.grad(trial), ufl.grad(test))
            - lam * float(rho_amp)
            * window_s_derivative_activity_ufl(u, c1_const, c2_const, eps_const)
            * trial * test
        ) * dx
        interpolation_points = rho.function_space.element.interpolation_points
        if callable(interpolation_points):
            interpolation_points = interpolation_points()
        density_expression = fem.Expression(source_density, interpolation_points)

        def sync_density() -> None:
            rho.interpolate(density_expression)
            rho.x.scatter_forward()

    def finish(result: NewtonResult) -> NewtonResult:
        if sync_density_on_return:
            sync_density()
        return result

    def check_spd(stage: str, newton_iteration: int, residual: float, *, force: bool = False) -> None:
        if stiffness_solver is None:
            return
        if not force and not bool(args.check_newton_spd):
            return
        if not force and args.newton_spd_prefix_filter and args.newton_spd_prefix_filter not in prefix:
            return
        if not force and stage == "iterate" and newton_iteration % int(args.newton_spd_every):
            return
        diagnose_newton_spd(
            jac_expr,
            [bc],
            stiffness_solver.matrix,
            args=args,
            prefix=prefix,
            stage=stage,
            newton_iteration=newton_iteration,
            residual=residual,
            homotopy_lambda=lam,
            c1=c1,
            c2=c2,
            eps_phi=eps_phi,
        )

    status = "MAX_NEWTON"
    converged = False
    initial_budget = int(args.max_newton_it)
    soft_cap_enabled = bool(getattr(args, "newton_soft_cap", False))
    hard_cap_reason = str(getattr(args, "newton_hard_cap_reason", "")).strip()
    hard_ceiling = (
        newton_hard_ceiling(initial_budget, float(args.newton_soft_cap_factor))
        if soft_cap_enabled and not hard_cap_reason
        else initial_budget
    )
    iteration_budget = initial_budget
    args._newton_initial_budget = initial_budget
    args._newton_current_budget = iteration_budget
    args._newton_hard_ceiling = hard_ceiling
    last_step_h1 = math.inf
    last_alpha = 0.0
    last_bt = 0
    solve_time_total = 0.0
    final_residual = math.inf

    initial_assembly_started = time.perf_counter()
    residual_old = (
        float(initial_residual)
        if initial_residual is not None
        else residual_norm(
            residual_form,
            bc,
            comm=comm,
            mode=args.residual_norm,
            metric_form=stiffness_form,
            solver=args.linear_solver,
            ksp_type=args.ksp_type,
            rtol=args.linear_rtol,
            atol=args.linear_atol,
            max_it=args.linear_max_it,
            prefix=f"{prefix}_resnorm_0_",
            stiffness_solver=stiffness_solver,
        )
    )
    initial_assembly_time = 0.0 if initial_residual is not None else time.perf_counter() - initial_assembly_started
    final_residual = residual_old
    residual_history = [residual_old]
    if residual_old <= float(tol_res):
        write_newton_record(
            args, comm, solve=prefix, phase=phase, outer_iteration=outer_iteration,
            homotopy_lambda=lam, nonlinear_iteration=0,
            status="CONVERGED_INITIAL", residual=residual_old,
            assembly_time=initial_assembly_time,
            elapsed=time.perf_counter() - solve_started,
        )
        check_spd("initial_converged", 0, residual_old)
        return finish(NewtonResult(
            "CONVERGED_RESIDUAL", True, 0, residual_old,
            last_step_h1, last_alpha, last_bt, solve_time_total,
        ))

    k = 0
    while k < iteration_budget:
        iteration_started = time.perf_counter()
        linear_started = time.perf_counter()
        fallback_used = False
        check_spd("iterate", k, residual_old)
        if homotopy_workspace is None:
            try:
                its, lin_res, solve_time = solve_linear_form(
                    jac_expr,
                    -residual_expr,
                    du,
                    [bc],
                    prefix=f"{prefix}_newton_{k}_",
                    solver=args.linear_solver,
                    ksp_type=args.ksp_type,
                    rtol=args.linear_rtol,
                    atol=args.linear_atol,
                    max_it=args.linear_max_it,
                    verbosity=args.verbosity,
                )
            except RuntimeError as primary_error:
                failure_reason = petsc_failure_reason(primary_error)
                iteration_cap_hit = failure_reason == int(PETSc.KSP.ConvergedReason.DIVERGED_MAX_IT)
                if iteration_cap_hit and bool(getattr(args, "reject_on_linear_max_it", False)):
                    solve_time = time.perf_counter() - linear_started
                    solve_time_total += solve_time
                    its = int(args.linear_max_it)
                    root_print(
                        comm,
                        f"INNER_NEWTON_LINEAR_CAP prefix={prefix} k={k} res={residual_old:.6e} "
                        f"linIts={its} linTime={solve_time:.6f}s action=REJECT_TRIAL",
                    )
                    if bool(getattr(args, "diagnose_capped_trial_coercivity", False)):
                        check_spd("linear_cap", k, residual_old, force=True)
                    write_newton_record(
                        args, comm, solve=prefix, phase=phase, outer_iteration=outer_iteration,
                        homotopy_lambda=lam, nonlinear_iteration=k,
                        status="LINEAR_CAP", residual=residual_old,
                        ksp_iterations=its, ksp_time=solve_time,
                        linear_cap=True, elapsed=time.perf_counter() - iteration_started,
                    )
                    return finish(NewtonResult(
                        "FAIL_LINEAR_MAX_IT",
                        False,
                        k,
                        residual_old,
                        last_step_h1,
                        last_alpha,
                        last_bt,
                        solve_time_total,
                    ))
                fallback = fallback_solver_name(args)
                if fallback is None:
                    if bool(getattr(args, "reject_on_linear_failure", False)):
                        solve_time = time.perf_counter() - linear_started
                        solve_time_total += solve_time
                        root_print(
                            comm,
                            f"INNER_NEWTON_LINEAR_FAILURE prefix={prefix} k={k} "
                            f"res={residual_old:.6e} linTime={solve_time:.6f}s "
                            f"reason={failure_reason} action=REJECT_TRIAL",
                        )
                        write_newton_record(
                            args, comm, solve=prefix, phase=phase,
                            outer_iteration=outer_iteration,
                            homotopy_lambda=lam, nonlinear_iteration=k,
                            status="LINEAR_FAILURE", residual=residual_old,
                            ksp_time=solve_time,
                            elapsed=time.perf_counter() - iteration_started,
                        )
                        return finish(NewtonResult(
                            "FAIL_LINEAR", False, k, residual_old,
                            last_step_h1, last_alpha, last_bt,
                            solve_time_total,
                        ))
                    raise
                fallback_used = True
                root_print(
                    comm,
                    f"LINEAR_FALLBACK prefix={prefix}_newton_{k}_ "
                    f"primary={args.linear_solver}/{args.ksp_type} fallback={fallback} error={primary_error}",
                )
                try:
                    its, lin_res, solve_time = solve_linear_form(
                        jac_expr,
                        -residual_expr,
                        du,
                        [bc],
                        prefix=f"{prefix}_newton_{k}_fallback_",
                        solver=fallback,
                        ksp_type=None,
                        rtol=args.linear_rtol,
                        atol=args.linear_atol,
                        max_it=args.linear_max_it,
                        verbosity=args.verbosity,
                    )
                except RuntimeError as fallback_error:
                    solve_time = time.perf_counter() - linear_started
                    solve_time_total += solve_time
                    root_print(
                        comm,
                        f"LINEAR_FALLBACK_FAILED prefix={prefix}_newton_{k}_ "
                        f"primary={args.linear_solver}/{args.ksp_type} fallback={fallback} "
                        f"primaryError={primary_error} fallbackError={fallback_error}",
                    )
                    write_newton_record(
                        args, comm, solve=prefix, phase=phase,
                        outer_iteration=outer_iteration, homotopy_lambda=lam,
                        nonlinear_iteration=k, status="LINEAR_FALLBACK_FAILED",
                        residual=residual_old, ksp_time=solve_time,
                        fallback_used=True,
                        elapsed=time.perf_counter() - iteration_started,
                    )
                    return finish(NewtonResult(
                        "FAIL_LINEAR_FALLBACK", False, k, residual_old,
                        last_step_h1, last_alpha, last_bt, solve_time_total,
                    ))
        else:
            its, lin_res, solve_time = homotopy_workspace.linear_solver.solve_vector(
                stiffness_solver.residual_vector,
                du,
                scale=-1.0,
            )
        linear_elapsed = time.perf_counter() - linear_started
        assembly_time = max(0.0, linear_elapsed - solve_time)
        solve_time_total += solve_time
        step_h1 = (
            stiffness_solver.h1_seminorm(du)
            if stiffness_solver is not None
            else math.sqrt(max(assemble_scalar(comm, ufl.inner(ufl.grad(du), ufl.grad(du)) * dx), 0.0))
        )
        last_step_h1 = step_h1
        if step_h1 < float(args.tol_step):
            status = "FAIL_STEP_STAGNATION"
            write_newton_record(
                args, comm, solve=prefix, phase=phase, outer_iteration=outer_iteration,
                homotopy_lambda=lam, nonlinear_iteration=k, status=status,
                residual=residual_old, step_norm=step_h1,
                ksp_iterations=its, ksp_residual=lin_res, ksp_time=solve_time,
                assembly_time=assembly_time, fallback_used=fallback_used,
                elapsed=time.perf_counter() - iteration_started,
            )
            return finish(NewtonResult(status, False, k, residual_old, step_h1, 0.0, 0, solve_time_total))

        old_u = u.x.array.copy()
        alpha = 1.0
        accepted = False
        bt = 0
        line_search_started = time.perf_counter()
        while alpha >= float(args.alpha_min) and bt <= int(args.max_backtrack):
            np.multiply(du.x.array, alpha, out=u.x.array)
            np.add(u.x.array, old_u, out=u.x.array)
            u.x.scatter_forward()
            residual_trial = residual_norm(
                residual_form,
                bc,
                comm=comm,
                mode=args.residual_norm,
                metric_form=stiffness_form,
                solver=args.linear_solver,
                ksp_type=args.ksp_type,
                rtol=args.linear_rtol,
                atol=args.linear_atol,
                max_it=args.linear_max_it,
                prefix=f"{prefix}_ls_resnorm_{k}_{bt}_",
                stiffness_solver=stiffness_solver,
            )
            if math.isfinite(residual_trial) and residual_trial <= (1.0 - float(args.armijo_c) * alpha) * residual_old:
                accepted = True
                final_residual = residual_trial
                break
            alpha *= float(args.beta_ls)
            bt += 1

        line_search_time = time.perf_counter() - line_search_started
        if not accepted:
            u.x.array[:] = old_u
            u.x.scatter_forward()
            write_newton_record(
                args, comm, solve=prefix, phase=phase, outer_iteration=outer_iteration,
                homotopy_lambda=lam, nonlinear_iteration=k, status="LINE_SEARCH_FAILED",
                residual=residual_old, damping=alpha, backtracks=bt, step_norm=step_h1,
                ksp_iterations=its, ksp_residual=lin_res, ksp_time=solve_time,
                assembly_time=assembly_time, line_search_time=line_search_time,
                fallback_used=fallback_used, elapsed=time.perf_counter() - iteration_started,
            )
            return finish(NewtonResult(
                "FAIL_LS", False, k, residual_old, step_h1, alpha, bt, solve_time_total,
            ))

        last_alpha = alpha
        last_bt = bt
        if plot_callback is not None:
            sync_density()
            plot_callback(prefix, k, final_residual, alpha, bt, c1, c2, eps_phi)
        if args.verbosity >= 2:
            root_print(
                comm,
                f"INNER_NEWTON prefix={prefix} k={k} res={final_residual:.6e} "
                f"alpha={alpha:.3e} bt={bt} stepH1={step_h1:.6e} "
                f"linIts={its} linRes={lin_res:.3e} linTime={solve_time:.6f}s",
            )
        iteration_status = "CONVERGED" if final_residual <= float(tol_res) else "ACCEPTED_STEP"
        write_newton_record(
            args, comm, solve=prefix, phase=phase, outer_iteration=outer_iteration,
            homotopy_lambda=lam, nonlinear_iteration=k, status=iteration_status,
            residual=final_residual, damping=alpha, backtracks=bt, step_norm=step_h1,
            ksp_iterations=its, ksp_residual=lin_res, ksp_time=solve_time,
            assembly_time=assembly_time, line_search_time=line_search_time,
            fallback_used=fallback_used, elapsed=time.perf_counter() - iteration_started,
        )
        if final_residual <= float(tol_res):
            check_spd("converged", k + 1, final_residual)
            return finish(NewtonResult(
                "CONVERGED_RESIDUAL",
                True,
                k + 1,
                final_residual,
                step_h1,
                last_alpha,
                last_bt,
                solve_time_total,
            ))
        # The accepted line-search residual and its assembled vector are the
        # residual at the next Newton iterate; do not assemble/solve it again.
        residual_old = final_residual
        residual_history.append(final_residual)
        completed_iterations = k + 1
        if completed_iterations >= iteration_budget:
            local_decision = (
                decide_newton_budget_extension(
                    residual_history,
                    initial_budget=initial_budget,
                    current_budget=iteration_budget,
                    hard_ceiling=hard_ceiling,
                    chunk=int(args.newton_soft_cap_chunk),
                    trend_window=int(args.newton_soft_cap_window),
                    maximum_contraction=float(args.newton_soft_cap_contraction),
                    soft_cap_enabled=soft_cap_enabled,
                    hard_cap_reason=hard_cap_reason,
                )
                if comm.rank == 0
                else None
            )
            decision: NewtonBudgetDecision = comm.bcast(local_decision, root=0)
            contraction_text = (
                f"{decision.geometric_contraction:.6e}"
                if math.isfinite(decision.geometric_contraction)
                else "NA"
            )
            if decision.extend:
                extension = decision.next_budget - iteration_budget
                iteration_budget = decision.next_budget
                args._newton_current_budget = iteration_budget
                root_print(
                    comm,
                    f"NEWTON_CAP_EXTEND prefix={prefix} phase={phase} "
                    f"iterations={completed_iterations} residual={final_residual:.6e} "
                    f"contraction={contraction_text} extension={extension} "
                    f"newBudget={iteration_budget} hardCeiling={hard_ceiling}",
                )
                write_newton_record(
                    args, comm, solve=prefix, phase=phase,
                    outer_iteration=outer_iteration, homotopy_lambda=lam,
                    nonlinear_iteration=completed_iterations, status="CAP_EXTENDED",
                    residual=final_residual, newton_cap_extension=extension,
                    newton_cap_reason=decision.reason,
                    newton_contraction=decision.geometric_contraction,
                    elapsed=time.perf_counter() - solve_started,
                )
            else:
                root_print(
                    comm,
                    f"NEWTON_CAP_STOP prefix={prefix} phase={phase} "
                    f"iterations={completed_iterations} residual={final_residual:.6e} "
                    f"contraction={contraction_text} reason={decision.reason} "
                    f"hardCeiling={hard_ceiling}",
                )
                write_newton_record(
                    args, comm, solve=prefix, phase=phase,
                    outer_iteration=outer_iteration, homotopy_lambda=lam,
                    nonlinear_iteration=completed_iterations, status="CAP_STOP",
                    residual=final_residual, newton_cap_reason=decision.reason,
                    newton_contraction=decision.geometric_contraction,
                    elapsed=time.perf_counter() - solve_started,
                )
                return finish(NewtonResult(
                    status, False, completed_iterations, final_residual,
                    last_step_h1, last_alpha, last_bt, solve_time_total,
                ))
        k += 1

    return finish(NewtonResult(
        status,
        converged,
        k,
        final_residual,
        last_step_h1,
        last_alpha,
        last_bt,
        solve_time_total,
    ))


def homotopy_tolerance(args: argparse.Namespace) -> float:
    """Return the nonlinear residual tolerance used during initialization continuation."""
    if args.homotopy_tol_res is not None:
        return float(args.homotopy_tol_res)
    return float(args.tol_res)


def solve_homotopy_initialization(
        *,
        u: fem.Function,
        du: fem.Function,
        tangent: fem.Function,
        rho: fem.Function,
        trial,
        test,
        dx,
        bc,
        stiffness_form,
        stiffness_solver: FixedStiffnessSolver,
        homotopy_workspace: HomotopySolveWorkspace,
        target_density,
        c1_const: fem.Constant,
        c2_const: fem.Constant,
        eps_const: fem.Constant,
        c1: float,
        c2: float,
        eps_phi: float,
        rho_amp: float,
        args: argparse.Namespace,
        prefix: str,
        initialization_writer: csv.DictWriter | None = None,
        initialization_handle=None,
        run_tag: str = "",
        initialization_method: str = "source_homotopy",
        frozen_residual_dual: float | None = None,
        trajectory_callback: Callable[[int, float], None] | None = None,
        trial_acceptance_callback: Callable[[int, float, float], str | None] | None = None,
        accepted_stage_control_callback: Callable[
            [int, float], tuple[float, float, float] | None
        ] | None = None,
        rejected_stage_control_callback: Callable[
            [int, float, float], tuple[float, float, float] | None
        ] | None = None,
) -> HomotopyResult:
    """Continue the sharp torsion source to the semilinear source.

    Thresholds remain fixed unless one of the optional control callbacks is
    supplied.  Those callbacks let a specialized caller strictly correct a
    bounded auxiliary-control update at the current accepted ``lambda``;
    callers that omit them retain the original fixed-threshold behavior.  At
    an accepted continuation state ``(u, lambda)``, the branch tangent
    ``s = du/dlambda``
    solves

        J_lambda s = rho_amp * (W(u;c1,c2,eps) - 1_{B_T}),

    with ``J_lambda = K - lambda*rho_amp*W_s``.  A first-order predictor
    ``u + dlambda*s`` is then Newton-corrected at the trial lambda.  Failed
    corrector steps are rolled back.  A specialized trial-acceptance callback
    may also reject an exactly corrected source step using caller-owned branch
    diagnostics before the controller commits it.  A specialized rejection
    callback may then strictly update the controls and retry the same increment
    once; otherwise the continuation increment is shortened.  Reaching
    ``lambda=1`` leaves ``u`` on the ordinary semilinear equilibrium used by
    the reduced optimizer.
    """
    comm = u.function_space.mesh.comm
    start = time.perf_counter()
    tol = homotopy_tolerance(args)
    target_lambda = float(getattr(args, "threshold_optimization_start_lambda", 1.0))
    target_status = "CONVERGED_LAMBDA1" if target_lambda >= 1.0 else "CONVERGED_LAMBDA_TARGET"
    controller = AdaptiveSourceHomotopy(SourceHomotopySchedule(
        initial_step=float(args.homotopy_initial_step),
        min_step=float(args.homotopy_min_step),
        max_step=float(args.homotopy_max_step),
        step_grow=float(args.homotopy_step_grow),
        step_shrink=float(args.homotopy_step_shrink),
        max_attempts=int(args.homotopy_max_stages),
        target_lambda=target_lambda,
        easy_newton_iterations=int(
            getattr(args, "homotopy_easy_newton_iterations", 2)
        ),
    ))
    lam = float(controller.lambda_value)
    step = float(controller.step)
    accepted_stages = 0
    rejected_steps = 0
    total_newton_iterations = 0
    tangent_solve_time = 0.0
    newton_solve_time = 0.0
    last_newton = NewtonResult(
        status="EXACT_LAMBDA0",
        converged=True,
        iterations=0,
        residual=0.0,
        step_h1=0.0,
        alpha=1.0,
        backtracks=0,
        solve_time=0.0,
    )

    def install_control_update(
            update: tuple[float, float, float],
            *,
            callback_name: str,
    ) -> tuple[float, float, float]:
        """Validate callback thresholds and synchronize all mutable constants."""
        updated_c1, updated_c2, updated_eps = map(float, update)
        if not all(math.isfinite(value) for value in (updated_c1, updated_c2, updated_eps)):
            raise RuntimeError(f"{callback_name} returned non-finite thresholds")
        if updated_c2 <= updated_c1 or updated_eps <= 0.0:
            raise RuntimeError(
                f"{callback_name} returned an invalid band: "
                f"c1={updated_c1:.16e}, c2={updated_c2:.16e}, eps={updated_eps:.16e}"
            )
        c1_const.value = PETSc.ScalarType(updated_c1)
        c2_const.value = PETSc.ScalarType(updated_c2)
        eps_const.value = PETSc.ScalarType(updated_eps)
        homotopy_workspace.set_parameters(
            lam=lam,
            c1=updated_c1,
            c2=updated_c2,
            eps_phi=updated_eps,
        )
        callback_residual = residual_norm(
            homotopy_workspace.residual_form,
            bc,
            comm=comm,
            mode=args.residual_norm,
            metric_form=stiffness_form,
            solver=args.linear_solver,
            ksp_type=args.ksp_type,
            rtol=args.linear_rtol,
            atol=args.linear_atol,
            max_it=args.linear_max_it,
            prefix=f"{prefix}_{callback_name}_residual_",
            stiffness_solver=stiffness_solver,
        )
        if not math.isfinite(callback_residual) or callback_residual > tol:
            raise RuntimeError(
                f"{callback_name} returned a state with non-strict dual residual "
                f"{callback_residual:.16e} > {tol:.16e}"
            )
        return updated_c1, updated_c2, updated_eps

    c1_const.value = PETSc.ScalarType(c1)
    c2_const.value = PETSc.ScalarType(c2)
    eps_const.value = PETSc.ScalarType(eps_phi)
    homotopy_workspace.set_parameters(lam=0.0, c1=c1, c2=c2, eps_phi=eps_phi)
    lambda0_residual = residual_norm(
        homotopy_workspace.residual_form,
        bc,
        comm=comm,
        mode=args.residual_norm,
        metric_form=stiffness_form,
        solver=args.linear_solver,
        ksp_type=args.ksp_type,
        rtol=args.linear_rtol,
        atol=args.linear_atol,
        max_it=args.linear_max_it,
        prefix=f"{prefix}_lambda0_check_",
        stiffness_solver=stiffness_solver,
    )
    root_print(
        comm,
        f"HOMOTOPY_CHECK lambda=0 residual={lambda0_residual:.12e} "
        f"norm={args.residual_norm}",
    )

    for stage in range(int(args.homotopy_max_stages)):
        if lam >= target_lambda - 1.0e-14:
            return HomotopyResult(
                status=target_status,
                converged=True,
                lambda_final=target_lambda,
                stages=accepted_stages,
                rejected_steps=rejected_steps,
                total_newton_iterations=total_newton_iterations,
                tangent_solve_time=tangent_solve_time,
                newton_solve_time=newton_solve_time,
                elapsed=time.perf_counter() - start,
                last_newton=last_newton,
            )

        proposal = controller.next_trial()
        dlambda = proposal.delta_lambda
        old_u = u.x.array.copy()
        stage_start = time.perf_counter()
        tangent_h1 = 0.0

        if bool(args.homotopy_predictor):
            homotopy_workspace.set_parameters(lam=lam, c1=c1, c2=c2, eps_phi=eps_phi)
            tangent_start = time.perf_counter()
            try:
                _, _, tangent_time = homotopy_workspace.linear_solver.solve_form(
                    homotopy_workspace.tangent_rhs_form,
                    tangent,
                )
            except RuntimeError:
                u.x.array[:] = old_u
                u.x.scatter_forward()
                return HomotopyResult(
                    status="FAIL_TANGENT_SOLVE",
                    converged=False,
                    lambda_final=lam,
                    stages=accepted_stages,
                    rejected_steps=rejected_steps,
                    total_newton_iterations=total_newton_iterations,
                    tangent_solve_time=tangent_solve_time + (time.perf_counter() - tangent_start),
                    newton_solve_time=newton_solve_time,
                    elapsed=time.perf_counter() - start,
                    last_newton=last_newton,
                )
            tangent_solve_time += tangent_time
            tangent_h1 = stiffness_solver.h1_seminorm(tangent)
            if lam <= 1.0e-14 and frozen_residual_dual is not None:
                relative_difference = abs(tangent_h1 - float(frozen_residual_dual)) / max(
                    tangent_h1,
                    float(frozen_residual_dual),
                    1.0e-30,
                )
                root_print(
                    comm,
                    f"HOMOTOPY_CHECK lambda=0 tangentH1={tangent_h1:.12e} "
                    f"frozenResidualHminus1={float(frozen_residual_dual):.12e} "
                    f"relativeDifference={relative_difference:.6e}",
                )
        else:
            tangent.x.array[:] = 0.0
            tangent.x.scatter_forward()

        trial_lambda = proposal.lambda_trial
        predictor_half_residual = math.nan
        predictor_check_residual = math.nan
        predictor_check_step = math.nan
        if bool(args.homotopy_predictor) and bool(args.verify_homotopy_init) and lam <= 1.0e-14:
            index_map = tangent.function_space.dofmap.index_map
            owned_size = index_map.size_local * tangent.function_space.dofmap.index_map_bs
            local_tangent_max = float(np.max(np.abs(tangent.x.array[:owned_size]))) if owned_size else 0.0
            tangent_max = float(comm.allreduce(local_tangent_max, op=MPI.MAX))
            predictor_check_step = min(
                dlambda,
                max(1.0e-8, 0.25 * eps_phi / max(tangent_max, 1.0e-30)),
            )
            np.multiply(tangent.x.array, 0.5 * predictor_check_step, out=u.x.array)
            np.add(u.x.array, old_u, out=u.x.array)
            u.x.scatter_forward()
            homotopy_workspace.set_parameters(
                lam=lam + 0.5 * predictor_check_step,
                c1=c1,
                c2=c2,
                eps_phi=eps_phi,
            )
            predictor_half_residual = residual_norm(
                homotopy_workspace.residual_form,
                bc,
                comm=comm,
                mode=args.residual_norm,
                metric_form=stiffness_form,
                solver=args.linear_solver,
                ksp_type=args.ksp_type,
                rtol=args.linear_rtol,
                atol=args.linear_atol,
                max_it=args.linear_max_it,
                prefix=f"{prefix}_predictor_half_",
                stiffness_solver=stiffness_solver,
            )
            np.multiply(tangent.x.array, predictor_check_step, out=u.x.array)
            np.add(u.x.array, old_u, out=u.x.array)
            u.x.scatter_forward()
            homotopy_workspace.set_parameters(
                lam=lam + predictor_check_step,
                c1=c1,
                c2=c2,
                eps_phi=eps_phi,
            )
            predictor_check_residual = residual_norm(
                homotopy_workspace.residual_form,
                bc,
                comm=comm,
                mode=args.residual_norm,
                metric_form=stiffness_form,
                solver=args.linear_solver,
                ksp_type=args.ksp_type,
                rtol=args.linear_rtol,
                atol=args.linear_atol,
                max_it=args.linear_max_it,
                prefix=f"{prefix}_predictor_check_",
                stiffness_solver=stiffness_solver,
            )
        np.multiply(tangent.x.array, dlambda, out=u.x.array)
        np.add(u.x.array, old_u, out=u.x.array)
        u.x.scatter_forward()
        homotopy_workspace.set_parameters(lam=trial_lambda, c1=c1, c2=c2, eps_phi=eps_phi)
        predicted_residual = residual_norm(
            homotopy_workspace.residual_form,
            bc,
            comm=comm,
            mode=args.residual_norm,
            metric_form=stiffness_form,
            solver=args.linear_solver,
            ksp_type=args.ksp_type,
            rtol=args.linear_rtol,
            atol=args.linear_atol,
            max_it=args.linear_max_it,
            prefix=f"{prefix}_predictor_{stage}_",
            stiffness_solver=stiffness_solver,
        )
        if math.isfinite(predictor_half_residual):
            observed_order = math.log(
                max(predictor_check_residual, 1.0e-300) / max(predictor_half_residual, 1.0e-300),
                2.0,
            )
            root_print(
                comm,
                f"HOMOTOPY_PREDICTOR_ORDER dlambda={predictor_check_step:.6e} "
                f"residualFull={predictor_check_residual:.12e} "
                f"residualHalf={predictor_half_residual:.12e} observedOrder={observed_order:.6f}",
            )
        trial_newton = solve_equilibrium(
            u=u,
            du=du,
            rho=rho,
            trial=trial,
            test=test,
            dx=dx,
            bc=bc,
            stiffness_form=stiffness_form,
            c1_const=c1_const,
            c2_const=c2_const,
            eps_const=eps_const,
            c1=c1,
            c2=c2,
            eps_phi=eps_phi,
            rho_amp=rho_amp,
            tol_res=tol,
            args=args,
            prefix=f"{prefix}_lambda_{stage}_{trial_lambda:.6f}".replace(".", "p"),
            homotopy_lambda=trial_lambda,
            homotopy_target_density=target_density,
            stiffness_solver=stiffness_solver,
            homotopy_workspace=homotopy_workspace,
            sync_density_on_return=False,
            initial_residual=predicted_residual,
            phase="homotopy",
            outer_iteration=-1,
        )
        total_newton_iterations += int(trial_newton.iterations)
        newton_solve_time += float(trial_newton.solve_time)
        last_newton = trial_newton

        trial_rejection_reason = ""
        if trial_newton.converged and trial_acceptance_callback is not None:
            callback_reason = trial_acceptance_callback(stage, lam, trial_lambda)
            trial_rejection_reason = str(callback_reason or "")
        stage_accepted = bool(trial_newton.converged and not trial_rejection_reason)

        if initialization_writer is not None:
            initialization_writer.writerow({
                "record": "homotopy_stage",
                "runTag": run_tag,
                "method": initialization_method,
                "c1": c1,
                "c2": c2,
                "width": c2 - c1,
                "eps": eps_phi,
                "lambdaOld": lam,
                "lambdaTrial": trial_lambda,
                "dlambda": dlambda,
                "tangentH1": tangent_h1,
                "predictedResidual": predicted_residual,
                "correctedResidual": trial_newton.residual,
                "newtonIterations": trial_newton.iterations,
                "damping": trial_newton.alpha,
                "backtracks": trial_newton.backtracks,
                "accepted": int(stage_accepted),
                "elapsed": time.perf_counter() - stage_start,
            })
            if initialization_handle is not None:
                initialization_handle.flush()

        if stage_accepted:
            controller.accept(
                proposal,
                newton_iterations=int(trial_newton.iterations),
                backtracks=int(trial_newton.backtracks),
            )
            lam = float(controller.lambda_value)
            step = float(controller.step)
            accepted_stages = int(controller.accepted_steps)
            if accepted_stage_control_callback is not None:
                control_update = accepted_stage_control_callback(accepted_stages, lam)
                if control_update is not None:
                    c1, c2, eps_phi = install_control_update(
                        control_update,
                        callback_name=f"accepted_control_{accepted_stages}",
                    )
            if trajectory_callback is not None:
                trajectory_callback(accepted_stages, lam)
            if args.verbosity >= 1:
                root_print(
                    comm,
                    f"HOMOTOPY_STAGE prefix={prefix} stage={stage} accepted=1 "
                    f"lambda={lam:.6e} dlambda={dlambda:.6e} newtonIts={trial_newton.iterations} "
                    f"residual={trial_newton.residual:.6e} alpha={trial_newton.alpha:.3e} "
                    f"backtracks={trial_newton.backtracks}",
                )
            if lam >= target_lambda - 1.0e-14:
                return HomotopyResult(
                    status=target_status,
                    converged=True,
                    lambda_final=target_lambda,
                    stages=accepted_stages,
                    rejected_steps=rejected_steps,
                    total_newton_iterations=total_newton_iterations,
                    tangent_solve_time=tangent_solve_time,
                    newton_solve_time=newton_solve_time,
                    elapsed=time.perf_counter() - start,
                    last_newton=last_newton,
                )
            continue

        u.x.array[:] = old_u
        u.x.scatter_forward()
        homotopy_workspace.set_parameters(lam=lam, c1=c1, c2=c2, eps_phi=eps_phi)
        if rejected_stage_control_callback is not None:
            control_update = rejected_stage_control_callback(stage, lam, trial_lambda)
            if control_update is not None:
                c1, c2, eps_phi = install_control_update(
                    control_update,
                    callback_name=f"rejected_control_{stage}",
                )
                controller.retry_after_external_update(proposal)
                rejected_steps = int(controller.rejected_steps)
                step = float(controller.step)
                if args.verbosity >= 1:
                    root_print(
                        comm,
                        f"HOMOTOPY_STAGE prefix={prefix} stage={stage} accepted=0 "
                        f"lambda={lam:.6e} trialLambda={trial_lambda:.6e} "
                        f"dlambda={dlambda:.6e} newtonStatus={trial_newton.status} "
                        f"postcheck={trial_rejection_reason or 'PASSED'} "
                        f"residual={trial_newton.residual:.6e} "
                        "controlRestoration=ACCEPTED retrySameStep=1",
                    )
                continue
        controller.reject(proposal)
        rejected_steps = int(controller.rejected_steps)
        step = float(controller.step)
        if args.verbosity >= 1:
            root_print(
                comm,
                f"HOMOTOPY_STAGE prefix={prefix} stage={stage} accepted=0 "
                f"lambda={lam:.6e} trialLambda={trial_lambda:.6e} dlambda={dlambda:.6e} "
                f"newtonStatus={trial_newton.status} "
                f"postcheck={trial_rejection_reason or 'PASSED'} "
                f"residual={trial_newton.residual:.6e} "
                f"nextStep={step:.6e}",
            )
        if controller.below_minimum_step:
            return HomotopyResult(
                status="FAIL_MIN_STEP",
                converged=False,
                lambda_final=lam,
                stages=accepted_stages,
                rejected_steps=rejected_steps,
                total_newton_iterations=total_newton_iterations,
                tangent_solve_time=tangent_solve_time,
                newton_solve_time=newton_solve_time,
                elapsed=time.perf_counter() - start,
                last_newton=last_newton,
            )

    return HomotopyResult(
        status="FAIL_MAX_STAGES",
        converged=False,
        lambda_final=lam,
        stages=accepted_stages,
        rejected_steps=rejected_steps,
        total_newton_iterations=total_newton_iterations,
        tangent_solve_time=tangent_solve_time,
        newton_solve_time=newton_solve_time,
        elapsed=time.perf_counter() - start,
        last_newton=last_newton,
    )


def evaluate_band_metrics(
        *,
        comm: MPI.Comm,
        u: fem.Function,
        tau_mask,
        dx,
        c1_const: fem.Constant,
        c2_const: fem.Constant,
        eps_const: fem.Constant,
        c1: float,
        c2: float,
        eps_phi: float,
        kappa: float,
        target_area: float,
) -> BandMetrics:
    """Evaluate soft discrepancy and certified active-band metrics.

    The soft metrics are differentiable and match the reduced optimization
    problem.  The certified metrics use a hard interior potential band that
    removes ``kappa*eps`` from both logistic transition layers; these metrics
    are not used to build gradients but expose whether the optimized smooth
    activity corresponds to an actual plateau region.

    Args:
        comm: MPI communicator for scalar reductions.
        u: Current semilinear state.
        tau_mask: UFL expression for the crisp torsion band indicator.
        dx: UFL measure.
        c1_const: Mutable lower-threshold constant.
        c2_const: Mutable upper-threshold constant.
        eps_const: Mutable smoothing-width constant.
        c1: Lower threshold value.
        c2: Upper threshold value.
        eps_phi: Smoothing width.
        kappa: Number of smoothing widths trimmed from each transition for
            certified band measurement.
        target_area: Area of the crisp torsion band.

    Returns:
        ``BandMetrics`` with soft leakage/missing and certified geometry.
    """
    c1_const.value = PETSc.ScalarType(c1)
    c2_const.value = PETSc.ScalarType(c2)
    eps_const.value = PETSc.ScalarType(eps_phi)
    activity = window_activity_const_ufl(u, c1_const, c2_const, eps_const)
    outside = 1.0 - tau_mask
    leakage = assemble_scalar(comm, outside * activity * dx)
    missing = assemble_scalar(comm, tau_mask * (1.0 - activity) * dx)
    activity_area = assemble_scalar(comm, activity * dx)
    certified = ufl.conditional(
        ufl.gt(u, c1 + float(kappa) * eps_phi),
        ufl.conditional(ufl.lt(u, c2 - float(kappa) * eps_phi), 1.0, 0.0),
        0.0,
    )
    certified_area = assemble_scalar(comm, certified * dx)
    certified_leakage = assemble_scalar(comm, outside * certified * dx)
    certified_missing = assemble_scalar(comm, tau_mask * (1.0 - certified) * dx)
    return BandMetrics(
        leakage=leakage,
        missing=missing,
        leakage_rel=leakage / max(target_area, 1.0e-30),
        missing_rel=missing / max(target_area, 1.0e-30),
        target_area=target_area,
        activity_area=activity_area,
        certified_area=certified_area,
        certified_leakage=certified_leakage,
        certified_missing=certified_missing,
    )


class InitializationDiagnostics:
    """Compiled, batched diagnostics used to rank projected initial states.

    Only fields consumed by the initializer score are evaluated.  All local
    integrals are reduced together, replacing the many scalar collectives and
    the unused residual/energy diagnostics in the general ``compute_metrics``
    helper.
    """

    def __init__(
            self,
            *,
            u: fem.Function,
            rho: fem.Function,
            rho_design: fem.Function,
            rho_design_l2: float,
            tau_mask,
            dx,
            c1_const: fem.Constant,
            c2_const: fem.Constant,
            eps_const: fem.Constant,
            kappa: float,
            active_threshold: float,
            rho_amp: float,
            target_area: float,
    ) -> None:
        self.comm = u.function_space.mesh.comm
        self.rho_design_l2 = float(rho_design_l2)
        self.target_area = float(target_area)
        self.c1_const = c1_const
        self.c2_const = c2_const
        self.eps_const = eps_const
        activity = window_activity_const_ufl(u, c1_const, c2_const, eps_const)
        outside = 1.0 - tau_mask
        certified = ufl.conditional(
            ufl.gt(u, c1_const + float(kappa) * eps_const),
            ufl.conditional(
                ufl.lt(u, c2_const - float(kappa) * eps_const),
                1.0,
                0.0,
            ),
            0.0,
        )
        active_design = ufl.conditional(
            ufl.gt(rho_design, float(active_threshold) * float(rho_amp)), 1.0, 0.0,
        )
        active_final = ufl.conditional(
            ufl.gt(rho, float(active_threshold) * float(rho_amp)), 1.0, 0.0,
        )
        self.active_design_area = assemble_scalar(self.comm, active_design * dx)
        integrands = (
            outside * activity,
            tau_mask * (1.0 - activity),
            activity,
            certified,
            outside * certified,
            tau_mask * (1.0 - certified),
            active_final,
            active_design * active_final,
            (rho - rho_design) ** 2,
        )
        self.forms = tuple(fem.form(integrand * dx) for integrand in integrands)

    def evaluate(
            self,
            *,
            c1: float,
            c2: float,
            eps_phi: float,
    ) -> tuple[BandMetrics, dict[str, float]]:
        """Assemble all candidate metrics with one MPI all-reduction."""
        self.c1_const.value = PETSc.ScalarType(c1)
        self.c2_const.value = PETSc.ScalarType(c2)
        self.eps_const.value = PETSc.ScalarType(eps_phi)
        local = np.fromiter(
            (float(fem.assemble_scalar(form)) for form in self.forms),
            dtype=np.float64,
            count=len(self.forms),
        )
        reduced = np.empty_like(local)
        self.comm.Allreduce(local, reduced, op=MPI.SUM)
        (
            leakage,
            missing,
            activity_area,
            certified_area,
            certified_leakage,
            certified_missing,
            active_area,
            active_overlap,
            rho_diff_sq,
        ) = map(float, reduced)
        area_scale = max(self.target_area, 1.0e-30)
        metrics = BandMetrics(
            leakage=leakage,
            missing=missing,
            leakage_rel=leakage / area_scale,
            missing_rel=missing / area_scale,
            target_area=self.target_area,
            activity_area=activity_area,
            certified_area=certified_area,
            certified_leakage=certified_leakage,
            certified_missing=certified_missing,
        )
        active_union = max(self.active_design_area + active_area - active_overlap, 1.0e-30)
        diagnostics = {
            "activeArea": active_area,
            "activeDesignArea": self.active_design_area,
            "activeOverlapArea": active_overlap,
            "activeJaccard": active_overlap / active_union,
            "rhoDesignDiffL2": math.sqrt(max(rho_diff_sq, 0.0)),
            "relRhoDesign": (
                math.sqrt(max(rho_diff_sq, 0.0))
                / max(self.rho_design_l2, 1.0e-30)
            ),
        }
        return metrics, diagnostics


def compute_reduced_gradient(
        *,
        u: fem.Function,
        s1: fem.Function,
        s2: fem.Function,
        trial,
        test,
        dx,
        bc,
        tau_mask,
        c1_const: fem.Constant,
        c2_const: fem.Constant,
        eps_const: fem.Constant,
        rho_amp: float,
        eps_mode: str,
        eps_ratio: float,
        homotopy_lambda: float = 1.0,
        args: argparse.Namespace,
        iteration: int,
) -> ReducedGradient:
    """Compute leakage and missing-area reduced gradients.

    This implements steps 3 through 7 of the algorithm note.  It assembles the
    Newton matrix, assembles the two residual parameter derivatives, solves the
    two sensitivity equations

        J_U s_i = -d r / d c_i,

    and then forms

        grad D_hat = D_c + S^T D_U

    for both the leakage functional ``L`` and missing-area functional ``M``.
    The signs in the RHS forms follow the residual definition
    ``r = K*u - rho_amp*W(u,c)`` and the sensitivity equation
    ``J_U*S = -r_c``.

    Args:
        u: Current converged or approximately converged state.
        s1: Work/output function for ``du/dc1``.
        s2: Work/output function for ``du/dc2``.
        trial: UFL trial function for the Newton matrix.
        test: UFL test function.
        dx: UFL integration measure.
        bc: Homogeneous Dirichlet boundary condition.
        tau_mask: UFL expression for the crisp torsion band indicator.
        c1_const: Current lower-threshold constant.
        c2_const: Current upper-threshold constant.
        eps_const: Current smoothing-width constant.
        rho_amp: Density amplitude.
        eps_mode: Fixed or relative epsilon mode.
        eps_ratio: Relative epsilon ratio used when ``eps_mode`` is
            ``"relative"``.
        args: Parsed command-line namespace for linear solver controls.
        iteration: Outer iteration index, used in PETSc prefixes.

    Returns:
        ``ReducedGradient`` including gradients and timing diagnostics.
    """
    total_start = time.perf_counter()
    comm = u.function_space.mesh.comm
    lam = float(homotopy_lambda)
    ws = window_s_derivative_activity_ufl(u, c1_const, c2_const, eps_const)
    dc1_activity, dc2_activity = window_c_derivatives_activity_ufl(
        u,
        c1_const,
        c2_const,
        eps_const,
        eps_mode=eps_mode,
        eps_ratio=eps_ratio,
    )
    jac_expr = (
        ufl.inner(ufl.grad(trial), ufl.grad(test))
        - lam * float(rho_amp) * ws * trial * test
    ) * dx
    rhs_s1 = lam * float(rho_amp) * dc1_activity * test * dx
    rhs_s2 = lam * float(rho_amp) * dc2_activity * test * dx

    def solve_sensitivity_systems(*, prefix: str, solver: str, ksp_type: str | None):
        return solve_same_matrix_forms(
            jac_expr,
            [rhs_s1, rhs_s2],
            [s1, s2],
            [bc],
            prefix=prefix,
            solver=solver,
            ksp_type=ksp_type,
            rtol=args.linear_rtol,
            atol=args.linear_atol,
            max_it=args.linear_max_it,
            verbosity=args.verbosity,
        )

    primary_prefix = f"sensitivity_{iteration}_"
    try:
        sensitivity_solve = solve_sensitivity_systems(
            prefix=primary_prefix,
            solver=args.linear_solver,
            ksp_type=args.ksp_type,
        )
    except RuntimeError as primary_error:
        iterative_fallback = sensitivity_iterative_fallback_ksp(args)
        iterative_error: RuntimeError | None = None
        if iterative_fallback is not None:
            root_print(
                comm,
                f"LINEAR_FALLBACK prefix={primary_prefix} primary={args.linear_solver}/{args.ksp_type} "
                f"fallback={args.linear_solver}/{iterative_fallback} error={primary_error}",
            )
            try:
                sensitivity_solve = solve_sensitivity_systems(
                    prefix=f"{primary_prefix}{iterative_fallback}_",
                    solver=args.linear_solver,
                    ksp_type=iterative_fallback,
                )
            except RuntimeError as error:
                iterative_error = error
        else:
            iterative_error = primary_error

        if iterative_error is not None:
            direct_fallback = fallback_solver_name(args)
            if direct_fallback is None:
                raise iterative_error
            root_print(
                comm,
                f"LINEAR_FALLBACK prefix={primary_prefix} "
                f"primary={args.linear_solver}/{iterative_fallback or args.ksp_type} "
                f"fallback={direct_fallback} error={iterative_error}",
            )
            sensitivity_solve = solve_sensitivity_systems(
                prefix=f"{primary_prefix}direct_fallback_",
                solver=direct_fallback,
                ksp_type=None,
            )

    (
        its,
        lin_res,
        solve_time,
        matrix_assembly_time,
        rhs_assembly_time,
        linear_solve_time,
    ) = sensitivity_solve

    gradient_assembly_start = time.perf_counter()
    outside = 1.0 - tau_mask
    grad_u_l_vec = assemble_vector_form(outside * ws * test * dx, bc)
    grad_u_m_vec = assemble_vector_form(-tau_mask * ws * test * dx, bc)
    direct_l = np.array([
        assemble_scalar(comm, outside * dc1_activity * dx),
        assemble_scalar(comm, outside * dc2_activity * dx),
    ], dtype=np.float64)
    direct_m = np.array([
        -assemble_scalar(comm, tau_mask * dc1_activity * dx),
        -assemble_scalar(comm, tau_mask * dc2_activity * dx),
    ], dtype=np.float64)
    sens_l = np.array([
        float(grad_u_l_vec.dot(s1.x.petsc_vec)),
        float(grad_u_l_vec.dot(s2.x.petsc_vec)),
    ], dtype=np.float64)
    sens_m = np.array([
        float(grad_u_m_vec.dot(s1.x.petsc_vec)),
        float(grad_u_m_vec.dot(s2.x.petsc_vec)),
    ], dtype=np.float64)
    grad_u_l_vec.destroy()
    grad_u_m_vec.destroy()
    gradient_assembly_time = time.perf_counter() - gradient_assembly_start
    return ReducedGradient(
        grad_l=direct_l + sens_l,
        grad_m=direct_m + sens_m,
        direct_l=direct_l,
        direct_m=direct_m,
        solve_iterations=(int(its[0]), int(its[1])),
        solve_residuals=(float(lin_res[0]), float(lin_res[1])),
        solve_time=solve_time,
        matrix_assembly_time=matrix_assembly_time,
        rhs_assembly_time=rhs_assembly_time,
        linear_solve_time=linear_solve_time,
        gradient_assembly_time=gradient_assembly_time,
        total_time=time.perf_counter() - total_start,
    )


def constraint_rows(
        *,
        c1: float,
        c2: float,
        c_min: float,
        c_max: float,
        min_width: float,
        include_leakage: bool,
        leakage: float,
        leakage_limit: float,
        grad_l: np.ndarray,
) -> list[tuple[np.ndarray, float]]:
    """Build linearized inequality rows for the two-variable QP.

    Each row represents ``a.dot(dc) <= b`` in threshold-increment coordinates.
    The rows enforce lower/upper box bounds for each threshold, the minimum
    width constraint, and optionally the linearized leakage filter constraint.

    Args:
        c1: Current lower threshold.
        c2: Current upper threshold.
        c_min: Lower search bound.
        c_max: Upper search bound.
        min_width: Minimum admissible threshold width.
        include_leakage: Whether to add the linearized leakage constraint.
        leakage: Current soft leakage value.
        leakage_limit: Admissible leakage limit.
        grad_l: Reduced leakage gradient.

    Returns:
        List of ``(a,b)`` pairs defining ``a.dot(dc) <= b``.
    """
    rows: list[tuple[np.ndarray, float]] = [
        (np.array([-1.0, 0.0]), float(c1) - float(c_min)),
        (np.array([1.0, 0.0]), float(c_max) - float(min_width) - float(c1)),
        (np.array([0.0, -1.0]), float(c2) - (float(c_min) + float(min_width))),
        (np.array([0.0, 1.0]), float(c_max) - float(c2)),
        (np.array([1.0, -1.0]), float(c2) - float(c1) - float(min_width)),
    ]
    if include_leakage:
        rows.append((np.asarray(grad_l, dtype=np.float64), float(leakage_limit) - float(leakage)))
    return rows


def feasible_qp_point(d: np.ndarray, rows: list[tuple[np.ndarray, float]], radius: float, tol: float = 1.0e-10) -> bool:
    """Check trust-region and linear inequality feasibility for one point.

    Args:
        d: Candidate two-vector ``(dc1, dc2)``.
        rows: Linear constraints represented as ``a.dot(d) <= b``.
        radius: Euclidean trust-region radius.
        tol: Numerical tolerance for boundary comparisons.

    Returns:
        True if ``d`` satisfies all constraints within ``tol``.
    """
    if float(np.linalg.norm(d)) > float(radius) + tol:
        return False
    return all(float(a.dot(d)) <= float(b) + tol for a, b in rows)


def solve_trust_region_qp(
        *,
        g: np.ndarray,
        rows: list[tuple[np.ndarray, float]],
        radius: float,
        hessian_scale: float,
) -> tuple[np.ndarray, float, str, bool]:
    """Solve the two-dimensional convex trust-region QP by enumeration.

    The model is

        min g.dot(d) + 0.5*hessian_scale*||d||^2

    subject to linear inequalities and ``||d|| <= radius``.  Because there are
    only two variables, a robust dependency-free solution is to enumerate all
    plausible active-set candidates: unconstrained minimizer, trust-circle
    Cauchy point, projections onto each active line, intersections of active
    lines, and intersections of active lines with the trust circle.

    Args:
        g: Objective gradient for the current QP objective.
        rows: Linear constraints ``a.dot(d) <= b``.
        radius: Trust-region radius.
        hessian_scale: Positive scalar Hessian approximation.

    Returns:
        Tuple ``(d, model_value, status, hit_boundary)``.  ``model_value`` is
        the objective value of the local quadratic model at ``d``.
    """
    g = np.asarray(g, dtype=np.float64)
    radius = float(radius)
    hessian_scale = max(float(hessian_scale), 1.0e-30)

    def model(d: np.ndarray) -> float:
        """Evaluate the scalar quadratic trust-region model at ``d``.

        Args:
            d: Candidate two-component threshold increment.

        Returns:
            Model value ``g.dot(d) + 0.5*hessian_scale*||d||^2``.
        """
        return float(g.dot(d) + 0.5 * hessian_scale * d.dot(d))

    candidates: list[np.ndarray] = []

    def add_candidate(d: np.ndarray) -> None:
        """Append a candidate only if it satisfies every QP constraint.

        The enumerator generates many algebraic candidates, including points
        from inactive constraints.  Filtering at insertion keeps the final
        minimization simple and prevents invalid active-set points from
        influencing the model minimum.
        """
        if feasible_qp_point(d, rows, radius):
            candidates.append(np.asarray(d, dtype=np.float64))

    origin = np.zeros(2, dtype=np.float64)
    add_candidate(origin)
    d0 = -g / hessian_scale
    add_candidate(d0)
    norm_g = float(np.linalg.norm(g))
    if norm_g > 0.0:
        add_candidate(-radius * g / norm_g)

    for a, b in rows:
        aa = float(a.dot(a))
        if aa <= 0.0:
            continue
        projected = d0 - a * ((float(a.dot(d0)) - float(b)) / aa)
        add_candidate(projected)
        tangent = np.array([-a[1], a[0]], dtype=np.float64)
        tt = float(tangent.dot(tangent))
        particular = a * (float(b) / aa)
        remaining = radius * radius - float(particular.dot(particular))
        if tt > 0.0 and remaining >= -1.0e-12:
            root = math.sqrt(max(remaining / tt, 0.0))
            add_candidate(particular + root * tangent)
            add_candidate(particular - root * tangent)

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a1, b1 = rows[i]
            a2, b2 = rows[j]
            mat = np.vstack([a1, a2])
            det = float(np.linalg.det(mat))
            if abs(det) <= 1.0e-14:
                continue
            rhs = np.array([b1, b2], dtype=np.float64)
            add_candidate(np.linalg.solve(mat, rhs))

    if not candidates:
        return origin, 0.0, "NO_FEASIBLE_CANDIDATE", False

    best = min(candidates, key=model)
    best_model = model(best)
    hit_boundary = abs(float(np.linalg.norm(best)) - radius) <= 1.0e-8 * max(radius, 1.0)
    return best, best_model, "OK", hit_boundary


def threshold_objective_weights(
        args: argparse.Namespace,
) -> tuple[float, float]:
    """Return normalized finite weights for the coupled threshold merit."""
    leakage = float(getattr(args, "threshold_leakage_weight", 1.25))
    missing = float(getattr(args, "threshold_missing_weight", 1.0))
    total = leakage + missing
    return leakage / total, missing / total


def threshold_merit_rel(
        metrics: BandMetrics,
        args: argparse.Namespace,
) -> float:
    """Return the active normalized outer merit used for progress bookkeeping."""
    if getattr(args, "threshold_objective_mode", "filter") == "weighted":
        leakage_weight, missing_weight = threshold_objective_weights(args)
        return (
            leakage_weight * float(metrics.leakage_rel)
            + missing_weight * float(metrics.missing_rel)
        )
    return float(metrics.leakage_rel) + float(metrics.missing_rel)


def threshold_objective_gradient(
        metrics: BandMetrics,
        gradient: ReducedGradient,
        args: argparse.Namespace,
) -> np.ndarray:
    """Return the reduced gradient for the selected outer-objective policy."""
    if getattr(args, "threshold_objective_mode", "filter") == "weighted":
        leakage_weight, missing_weight = threshold_objective_weights(args)
        return (
            leakage_weight * np.asarray(gradient.grad_l, dtype=np.float64)
            + missing_weight * np.asarray(gradient.grad_m, dtype=np.float64)
        )
    if metrics.leakage_rel <= float(args.eta_out):
        return np.asarray(gradient.grad_m, dtype=np.float64)
    return np.asarray(gradient.grad_l, dtype=np.float64)


def choose_parameter_step(
        *,
        c1: float,
        c2: float,
        c_min: float,
        c_max: float,
        min_width: float,
        metrics: BandMetrics,
        gradient: ReducedGradient,
        trust_radius_abs: float,
        args: argparse.Namespace,
) -> ParameterStep:
    """Choose the reduced threshold update for the current filter state.

    If leakage is currently infeasible, containment has priority and the QP
    objective is leakage reduction.  If leakage is feasible, the QP objective
    is missing-area reduction and the leakage limit is included as a linearized
    constraint.  This matches the filter interpretation in the algorithm note:
    first stop spill outside the torsion band, then fill as much of the target
    band as possible.

    Args:
        c1: Current lower threshold.
        c2: Current upper threshold.
        c_min: Lower search bound.
        c_max: Upper search bound.
        min_width: Minimum admissible threshold width.
        metrics: Current soft and certified band metrics.
        gradient: Current reduced gradients of leakage and missing area.
        trust_radius_abs: Current absolute trust-region radius.
        args: Parsed command-line namespace for filter and QP parameters.

    Returns:
        ``ParameterStep`` describing the proposed increment and model quality.
    """
    leakage_limit = float(args.eta_out) * metrics.target_area
    feasible_leakage = metrics.leakage <= leakage_limit
    if getattr(args, "threshold_objective_mode", "filter") == "weighted":
        objective_name = "weighted"
        g = threshold_objective_gradient(metrics, gradient, args)
        include_leakage = False
    elif feasible_leakage:
        objective_name = "missing"
        g = gradient.grad_m
        include_leakage = True
    else:
        objective_name = "leakage"
        g = gradient.grad_l
        include_leakage = False
    rows = constraint_rows(
        c1=c1,
        c2=c2,
        c_min=c_min,
        c_max=c_max,
        min_width=min_width,
        include_leakage=include_leakage,
        leakage=metrics.leakage,
        leakage_limit=leakage_limit,
        grad_l=gradient.grad_l,
    )
    grad_norm = float(np.linalg.norm(g))
    if grad_norm <= 0.0:
        return ParameterStep(
            dc=np.zeros(2, dtype=np.float64),
            objective_name=objective_name,
            predicted_reduction=0.0,
            model_change=0.0,
            hessian_scale=0.0,
            step_norm=0.0,
            hit_boundary=False,
            status="ZERO_GRADIENT",
        )
    hessian_scale = max(grad_norm / max(float(trust_radius_abs), 1.0e-30), float(args.qp_hessian_floor))
    dc, model_change, status, hit_boundary = solve_trust_region_qp(
        g=g,
        rows=rows,
        radius=trust_radius_abs,
        hessian_scale=hessian_scale,
    )
    return ParameterStep(
        dc=dc,
        objective_name=objective_name,
        predicted_reduction=max(-float(model_change), 0.0),
        model_change=float(model_change),
        hessian_scale=hessian_scale,
        step_norm=float(np.linalg.norm(dc)),
        hit_boundary=hit_boundary,
        status=status,
    )


def normalized_branch_retention(
        numerator: float,
        reference_self_overlap: float,
) -> float:
    """Normalize a trial/reference overlap so the unchanged state scores one."""
    if reference_self_overlap <= 1.0e-30:
        return 1.0
    return float(numerator) / max(float(reference_self_overlap), 1.0e-30)


def normalized_branch_dice(
        cross_overlap: float,
        reference_self_overlap: float,
        trial_self_overlap: float,
) -> float:
    """Return a symmetric soft-Dice similarity for two activity fields.

    Unlike the historical retention ratio, this score cannot make a large
    trial activity look harmless merely because it contains the reference
    activity.  An unchanged diffuse field scores one, disjoint fields score
    zero, and every finite nonnegative pair lies in ``[0, 1]`` up to roundoff.
    """
    cross = max(float(cross_overlap), 0.0)
    reference_sq = max(float(reference_self_overlap), 0.0)
    trial_sq = max(float(trial_self_overlap), 0.0)
    denominator = reference_sq + trial_sq
    if denominator <= 1.0e-30:
        return 1.0
    return float(np.clip(2.0 * cross / denominator, 0.0, 1.0))


def branch_overlap_ratio(
        *,
        comm: MPI.Comm,
        activity_ref: fem.Function,
        u_trial: fem.Function,
        dx,
        c1_const: fem.Constant,
        c2_const: fem.Constant,
        eps_const: fem.Constant,
        c1_trial: float,
        c2_trial: float,
        eps_trial: float,
) -> float:
    """Measure soft overlap between current and trial activity fields.

    Nonlinear semilinear problems may have multiple solution branches.  The
    optimizer is intended to follow the localized branch selected by warm-start
    Newton and the sensitivity predictor, not jump to a disconnected or
    collapsed activity region.  This ratio approximates the branch-preservation
    test from the algorithm note.

    Args:
        comm: MPI communicator for scalar reductions.
        activity_ref: Interpolated soft activity at the current accepted state.
        u_trial: Trial corrected semilinear state.
        dx: UFL integration measure.
        c1_const: Mutable lower-threshold constant.
        c2_const: Mutable upper-threshold constant.
        eps_const: Mutable smoothing-width constant.
        c1_trial: Trial lower threshold.
        c2_trial: Trial upper threshold.
        eps_trial: Trial smoothing width.

    Returns:
        Ratio ``int W_trial*W_current / int W_current**2``.  The
        self-overlap normalization makes an unchanged diffuse state score
        exactly one; values near zero indicate collapse or a substantial
        branch jump.
    """
    c1_const.value = PETSc.ScalarType(c1_trial)
    c2_const.value = PETSc.ScalarType(c2_trial)
    eps_const.value = PETSc.ScalarType(eps_trial)
    denominator = assemble_scalar(comm, activity_ref * activity_ref * dx)
    if denominator <= 1.0e-30:
        return 1.0
    numerator = assemble_scalar(comm, activity_ref * window_activity_const_ufl(u_trial, c1_const, c2_const, eps_const) * dx)
    return normalized_branch_retention(numerator, denominator)


def activity_dice_ratio(
        *,
        comm: MPI.Comm,
        activity_ref: fem.Function,
        activity_trial: fem.Function,
        dx,
) -> float:
    """Measure symmetric soft-Dice similarity in one common FE space.

    Both nonlinear windows must be interpolated by the caller.  Comparing two
    like representations makes the identity score exactly one and avoids a
    mesh-dependent mismatch between an interpolated reference and an exact
    non-polynomial UFL trial expression.
    """
    reference_sq = assemble_scalar(comm, activity_ref * activity_ref * dx)
    trial_sq = assemble_scalar(comm, activity_trial * activity_trial * dx)
    cross = assemble_scalar(comm, activity_ref * activity_trial * dx)
    return normalized_branch_dice(cross, reference_sq, trial_sq)


def run_strategy(args: argparse.Namespace) -> int:
    """Execute the complete reduced-space optimization workflow.

    The function owns the end-to-end run:

    1. Validate arguments and create dated run directories under ``tmp``.
    2. Generate or load the star-shaped mesh.
    3. Build the finite-element space and homogeneous Dirichlet boundary
       condition.
    4. Solve the torsion problem and construct the crisp torsion band.
    5. Build the sharp torsion-designed density ``rho_amp*1_{B_T}`` and solve
       its Poisson design potential without projecting the source into the
       conforming space.
    6. In homotopy mode, minimize the frozen assembled H^-1 residual subject
       to geometric caps, then continue the single selected pair through an
       adaptive source homotopy ``lambda:0->1``.  ``--init-mode legacy``
       retains candidate generation, direct Newton projection, and projected
       score selection.
    7. Iterate the reduced algorithm: adaptive Newton projection, soft
       discrepancy evaluation, sensitivity solves, reduced-gradient formation,
       two-variable trust-region update, sensitivity prediction, Newton
       correction, and accept/reject filtering.
    8. Always run a final exact Newton projection to
       ``--final-newton-tol-res`` when supplied, otherwise ``--tol-res``,
       before reporting final metrics.

    The implementation deliberately keeps the outer geometric objective, final
    Newton feasibility, and practical certified-subband status separate.  A
    strict full-band match reports ``CONVERGED``.  A contained nontrivial
    certified plateau reports ``CONVERGED_CERTIFIED_SUBBAND``.  The script
    exits with a nonzero code if the final exact Newton projection cannot reach
    the requested tolerance, even if the geometric metrics improved.

    Args:
        args: Parsed and already syntactically valid command-line namespace.

    Returns:
        Process-style return code.  ``0`` means the final Newton projection
        reached the requested final residual tolerance; ``2`` means the optional
        ``--fail-on-nonconvergence`` policy rejected the final geometric
        status; ``3`` means the final Newton projection itself did not
        converge.
    """
    validate_args(args)
    args.newton_spd_records = []
    args.newton_writer = None
    args.newton_handle = None
    stiffness_args = phase_solver_args(args, "stiffness")
    homotopy_args = phase_solver_args(args, "homotopy")
    nonlinear_args = phase_solver_args(args, "nonlinear")
    nonlinear_args.reject_on_linear_max_it = True
    nonlinear_args.reject_on_linear_failure = True
    sensitivity_args = phase_solver_args(args, "sensitivity")
    final_solver_args = phase_solver_args(args, "final")
    trial_nonlinear_args = argparse.Namespace(**vars(nonlinear_args))
    trial_nonlinear_args.linear_max_it = int(args.trial_linear_max_it)
    trial_nonlinear_args.iterative_fallback_solver = "none"
    trial_nonlinear_args.reject_on_linear_max_it = True
    trial_nonlinear_args.reject_on_linear_failure = True
    trial_nonlinear_args.newton_hard_cap_reason = "outer_trial_projection"
    if args.plot_severe:
        args.plot = True
    comm = MPI.COMM_WORLD
    params = params_from_args(args)
    final_tol_res = final_newton_tolerance(args)
    optimization_homotopy_lambda = float(args.threshold_optimization_start_lambda)
    run_dir = make_run_dir(args) if comm.rank == 0 else None
    run_dir = Path(comm.bcast(str(run_dir), root=0))
    run_tag = run_dir.name
    log_dir = run_dir / "logs"
    out_dir = run_dir / "out"
    if comm.rank == 0:
        log_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
    comm.barrier()

    terminal_log_path = out_dir / "terminal.log"
    terminal_log_capture: TerminalLogCapture | None = None
    if args.save_terminal_log:
        if comm.rank == 0:
            terminal_log_path.write_bytes(b"")
        comm.barrier()
        local_capture_error: str | None = None
        try:
            terminal_log_capture = TerminalLogCapture(
                terminal_log_path,
                rank=int(comm.rank),
            )
            terminal_log_capture.start()
        except Exception as error:  # pragma: no cover - platform/descriptor failure
            local_capture_error = (
                f"rank={comm.rank} {type(error).__name__}: {error}"
            )
        capture_errors = tuple(
            error for error in comm.allgather(local_capture_error) if error is not None
        )
        if capture_errors:
            if terminal_log_capture is not None:
                terminal_log_capture.close()
            raise RuntimeError(
                "failed to start terminal log capture: " + "; ".join(capture_errors)
            )

    opt_csv = log_dir / "optimization.csv"
    frame_csv = log_dir / "frames.csv"
    initialization_csv = log_dir / "initialization.csv"
    newton_csv = log_dir / "newton.csv"
    phases_csv = log_dir / "phases.csv"
    newton_spd_csv = log_dir / "newton_spd.csv"
    inexact_newton_csv = (
        Path(args.inexact_newton_output).expanduser().resolve()
        if args.inexact_newton_output is not None
        else log_dir / "inexact_newton.csv"
    )
    summary_path = out_dir / "summary.txt"
    trajectory_path = Path(args.trajectory_output).expanduser().resolve() if args.trajectory_output else out_dir / "trajectory.npz"
    equilibrium_path = (
        Path(args.equilibrium_output).expanduser().resolve()
        if args.equilibrium_output is not None
        else out_dir / "equilibrium.npz"
    )
    initialization_handle = initialization_csv.open("w", newline="", encoding="utf-8") if comm.rank == 0 else None
    initialization_writer = (
        csv.DictWriter(initialization_handle, fieldnames=INITIALIZATION_CSV_FIELDS)
        if initialization_handle is not None
        else None
    )
    if initialization_writer is not None:
        initialization_writer.writeheader()
    newton_handle = newton_csv.open("w", newline="", encoding="utf-8") if comm.rank == 0 else None
    newton_writer = csv.DictWriter(newton_handle, fieldnames=NEWTON_CSV_FIELDS) if newton_handle is not None else None
    if newton_writer is not None:
        newton_writer.writeheader()
        newton_handle.flush()
    for phase_args in (
            args, stiffness_args, homotopy_args, nonlinear_args,
            sensitivity_args, final_solver_args, trial_nonlinear_args,
    ):
        phase_args.newton_writer = newton_writer
        phase_args.newton_handle = newton_handle
    if comm.rank == 0:
        with phases_csv.open("w", newline="", encoding="utf-8") as phase_handle:
            csv.DictWriter(phase_handle, fieldnames=PHASE_CSV_FIELDS).writeheader()
    phase_times = {name: 0.0 for name in PHASE_NAMES}
    phase_calls = {name: 0 for name in PHASE_NAMES}

    def add_phase(name: str, elapsed: float) -> None:
        if name not in phase_times:
            raise KeyError(name)
        phase_times[name] += float(elapsed)
        phase_calls[name] += 1

    def write_phases(terminal_detail: str = "") -> None:
        """Atomically replace the phase table with the current run ledger."""
        if comm.rank != 0:
            return
        with phases_csv.open("w", newline="", encoding="utf-8") as phase_handle:
            phase_writer = csv.DictWriter(phase_handle, fieldnames=PHASE_CSV_FIELDS)
            phase_writer.writeheader()
            for phase_name in PHASE_NAMES:
                detail = "not used" if phase_calls[phase_name] == 0 else ""
                if terminal_detail and phase_name == "total":
                    detail = terminal_detail
                phase_writer.writerow({
                    "phase": phase_name,
                    "elapsed": phase_times[phase_name],
                    "calls": phase_calls[phase_name],
                    "detail": detail,
                })

    total_start = time.perf_counter()

    root_print(comm, "========== START STRATEGY A DOLFINX WINDOW REDUCED OPTIMIZATION ==========")
    root_print(comm, f"RUN_TAG {run_tag}")
    root_print(comm, f"RUN_DIR {run_dir}")
    root_print(comm, f"OPT_CSV {opt_csv}")
    root_print(comm, f"FRAME_CSV {frame_csv}")
    root_print(comm, f"INITIALIZATION_CSV {initialization_csv}")
    root_print(comm, f"NEWTON_CSV {newton_csv}")
    root_print(comm, f"PHASES_CSV {phases_csv}")
    root_print(
        comm,
        f"INEXACT_NEWTON_CSV {inexact_newton_csv if args.run_inexact_newton_study else 'disabled'}",
    )
    root_print(comm, f"TRAJECTORY {trajectory_path if args.save_trajectory else 'disabled'}")
    root_print(comm, f"EQUILIBRIUM_CHECKPOINT {equilibrium_path}")
    root_print(comm, f"SUMMARY {summary_path}")
    root_print(
        comm,
        f"TERMINAL_LOG {terminal_log_path if args.save_terminal_log else 'disabled'}",
    )
    root_print(
        comm,
        "NEWTON_BUDGET "
        f"initial={args.max_newton_it} softCap={int(args.newton_soft_cap)} "
        f"chunk={args.newton_soft_cap_chunk} factor={args.newton_soft_cap_factor:.3g} "
        f"window={args.newton_soft_cap_window} "
        f"maxContraction={args.newton_soft_cap_contraction:.6g}",
    )
    root_print(
        comm,
        "REDUCED_OBJECTIVE "
        f"etaOut={args.eta_out:.6e} tolArea={args.tol_area:.6e} "
        f"epsMode={args.eps_mode} epsRatio={args.eps_ratio:.6e} epsPhi={args.eps_phi} "
        f"kappa={args.kappa:.6e} deltaC={args.delta_c:.6e} "
        f"tolRes={args.tol_res:.6e} finalNewtonTolRes={final_tol_res:.6e} "
        f"innerNewtonTol={args.inner_newton_tol} innerTolMax={args.inner_tol_max:.6e} "
        f"innerTolGamma={args.inner_tol_gamma:.6e} "
        f"initMode={args.init_mode} initSearch={args.init_search} "
        f"initFallback={args.init_fallback} initialEquilibrium={args.initial_equilibrium} "
        f"homotopyInitialStep={args.homotopy_initial_step:.3e} "
        f"homotopyMinStep={args.homotopy_min_step:.3e} homotopyMaxStep={args.homotopy_max_step:.3e} "
        f"thresholdOptimizationLambda={optimization_homotopy_lambda:.6e} "
        f"homotopyTol={homotopy_tolerance(args):.3e} predictor={int(args.homotopy_predictor)} "
        f"minCertifiedAreaFraction={args.min_certified_area_fraction:.6e} "
        f"plotSevere={int(args.plot_severe)}",
    )

    root_print(
        comm,
        "SOLVER_PHASES "
        f"preset={args.solver_preset} stiffness={stiffness_args.linear_solver}/{stiffness_args.ksp_type} "
        f"homotopy={homotopy_args.linear_solver}/{homotopy_args.ksp_type} "
        f"nonlinear={nonlinear_args.linear_solver}/{nonlinear_args.ksp_type} "
        f"sensitivity={sensitivity_args.linear_solver}/{sensitivity_args.ksp_type}"
        f"->{args.sensitivity_iterative_fallback} "
        f"final={final_solver_args.linear_solver}/{final_solver_args.ksp_type} "
        f"fallback={args.iterative_fallback_solver} trialLinearMaxIt={args.trial_linear_max_it} "
        f"diagnoseCappedTrialCoercivity={int(args.diagnose_capped_trial_coercivity)}",
    )
    mesh_start = time.perf_counter()
    domain, mesh_path, geometry_mode = load_or_generate_mesh(args, run_dir, comm)
    tdim = domain.topology.dim
    nt = int(domain.topology.index_map(tdim).size_global)
    mesh_time = time.perf_counter() - mesh_start
    add_phase("mesh", mesh_time)
    root_print(comm, f"GEOMETRY {geometry_mode} meshFile={mesh_path}")
    root_print(comm, f"MESH nt={nt} loadTime={mesh_time:.6f}")

    V = fem.functionspace(domain, ("Lagrange", int(args.order)))
    ndof = int(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)
    bc = boundary_bc(V)
    root_print(comm, f"SPACE order={args.order} ndof={ndof}")

    qdeg = args.quad_degree if args.quad_degree is not None else max(2 * int(args.order) + 8, 12)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": qdeg})
    trial = ufl.TrialFunction(V)
    test = ufl.TestFunction(V)
    stiffness_form = ufl.inner(ufl.grad(trial), ufl.grad(test)) * dx

    T = fem.Function(V, name="T")
    tau_band = fem.Function(V, name="tauBand")
    rho_design = fem.Function(V, name="rhoDesign")
    phi_target = fem.Function(V, name="phiT")
    u = fem.Function(V, name="phi")
    du = fem.Function(V, name="du")
    rho = fem.Function(V, name="rho")
    s1 = fem.Function(V, name="sensitivityC1")
    s2 = fem.Function(V, name="sensitivityC2")
    frozen_riesz = fem.Function(V, name="frozenRiesz")
    homotopy_tangent = fem.Function(V, name="homotopyTangent")
    phi_diff = fem.Function(V, name="phiMinusPhiT")
    activity_ref = fem.Function(V, name="activityRef")

    frame_fields = [
        "frame", "runTag", "stage", "ieps", "k", "nt", "ndof", "epsPhi", "resEuclid",
        "massRho", "maxRho", "activeArea", "plateauArea", "relRhoDesign", "filename",
    ]
    frame_handle = frame_csv.open("w", newline="", encoding="utf-8") if comm.rank == 0 else None
    frame_writer = csv.DictWriter(frame_handle, fieldnames=frame_fields) if comm.rank == 0 else None
    if frame_writer is not None:
        frame_writer.writeheader()
    plotter = MPIPyVistaTorsionPlotter(
        args,
        run_tag=run_tag,
        run_dir=run_dir,
        frame_writer=frame_writer,
        comm=comm,
    )

    torsion_started = time.perf_counter()
    stiffness_solver = FixedStiffnessSolver(
        stiffness_form,
        V,
        [bc],
        prefix="shared_stiffness_",
        solver=stiffness_args.linear_solver,
        ksp_type=stiffness_args.ksp_type,
        rtol=stiffness_args.linear_rtol,
        atol=stiffness_args.linear_atol,
        max_it=stiffness_args.linear_max_it,
    )
    its, rel, solve_time = stiffness_solver.solve_form(1.0 * test * dx, T)
    _, tmax = global_minmax(comm, T)
    c1_t = params.alpha_t1 * tmax
    c2_t = params.alpha_t2 * tmax
    eps_t = params.eps_t_ratio * (c2_t - c1_t)
    tau_mask = ufl.conditional(ufl.gt(T, c1_t), ufl.conditional(ufl.lt(T, c2_t), 1.0, 0.0), 0.0)
    target_density = float(params.rho_amp) * tau_mask

    def optimization_source_density():
        nonlinear_density = window_density_const_ufl(
            u, c1_const, c2_const, eps_const, params.rho_amp
        )
        return (
            nonlinear_density
            if optimization_homotopy_lambda >= 1.0
            else (1.0 - optimization_homotopy_lambda) * target_density
            + optimization_homotopy_lambda * nonlinear_density
        )

    update_interpolated(tau_band, tau_mask)
    target_area = assemble_scalar(comm, tau_mask * dx)
    root_print(comm, f"SOLVER_OK problem=torsion iters={its} rel={rel:.3e} time={solve_time:.3f}")
    root_print(
        comm,
        f"TORSION Tmax={tmax:.6e} c1T={c1_t:.6e} c2T={c2_t:.6e} "
        f"epsT={eps_t:.6e} targetArea={target_area:.6e}",
    )
    add_phase("torsion", time.perf_counter() - torsion_started)
    if target_area <= 0.0:
        raise RuntimeError("torsion target band has zero area")

    target_solve_started = time.perf_counter()
    if args.init_mode == "homotopy":
        # In homotopy mode this interpolation is strictly diagnostic.  The
        # Poisson RHS below is the original discontinuous UFL indicator.
        update_interpolated(rho_design, target_density)
        phi_target_rhs = target_density * test * dx
        target_source_name = "sharp_indicator"
    else:
        # Preserve the legacy target construction exactly: its smoothed
        # torsion density is interpolated before entering the Poisson RHS.
        update_interpolated(
            rho_design,
            window_ufl(T, c1_t, c2_t, eps_t, params.rho_amp),
        )
        phi_target_rhs = rho_design * test * dx
        target_source_name = "legacy_smoothed_interpolant"
    rho_design_l2 = math.sqrt(max(assemble_scalar(comm, rho_design * rho_design * dx), 0.0))
    rho_design_mass = assemble_scalar(comm, rho_design * dx)
    _, rho_design_max = global_minmax(comm, rho_design)

    its, rel, solve_time = stiffness_solver.solve_form(phi_target_rhs, phi_target)
    _, phi_target_max = global_minmax(comm, phi_target)
    phi_target_l2 = math.sqrt(max(assemble_scalar(comm, phi_target * phi_target * dx), 0.0))
    c_search_max = max(float(args.cmax_factor) * phi_target_max, 1.0e-12)
    c_min = float(args.c_lower_fraction) * c_search_max
    c_upper = float(args.c_upper_fraction) * c_search_max
    c_scale = max(c_upper - c_min, 1.0e-14)
    required_min_width = certified_min_width(args, c_scale)
    if required_min_width > c_scale:
        raise ValueError(
            "empty threshold search interval: certified minimum width "
            f"{required_min_width:.6e} exceeds c-range {c_scale:.6e}"
        )
    min_width = required_min_width
    root_print(comm, f"SOLVER_OK problem=phi_target iters={its} rel={rel:.3e} time={solve_time:.3f}")
    root_print(
        comm,
        f"PHI_TARGET max={phi_target_max:.6e} l2={phi_target_l2:.6e} "
        f"rhoDesignMass={rho_design_mass:.6e} rhoDesignMax={rho_design_max:.6e} "
        f"targetSource={target_source_name}",
    )
    root_print(comm, f"SEARCH_DOMAIN cMin={c_min:.6e} cMax={c_upper:.6e} minWidth={min_width:.6e}")
    add_phase("target_solve", time.perf_counter() - target_solve_started)

    trajectory = TrajectoryRecorder(
        enabled=bool(args.save_trajectory),
        every=int(args.trajectory_every),
        output=trajectory_path,
        mesh_path=mesh_path,
        comm=comm,
    )
    trajectory.set_context({
        "run_tag": run_tag,
        "order": int(args.order),
        "quadrature_degree": int(qdeg),
        "alpha_t1": float(params.alpha_t1),
        "alpha_t2": float(params.alpha_t2),
        "c1_t": float(c1_t),
        "c2_t": float(c2_t),
        "rho_amp": float(params.rho_amp),
        "complete": False,
    })
    trajectory.set_fixed(
        torsion=T, target_band=tau_band,
        target_density=rho_design, target_potential=phi_target,
    )

    if args.plot_design:
        plotter.emit(
            [T, rho_design, phi_target],
            ["Torsion T", "rho_design", "phi_design"],
            stage="DESIGN",
            ieps=-1,
            k=-1,
            eps_phi=0.0,
            residual=0.0,
            metrics={"massRho": rho_design_mass, "maxRho": rho_design_max},
            token="design",
            save=bool(args.frame_design),
            show=True,
            nt=nt,
            ndof=ndof,
            contour_field_index=0,
            contour_levels=(c1_t, c2_t),
        )

    init_candidate_name = "manual"
    init_candidate_score = math.nan
    init_candidates: list[InitialWindowCandidate] = []
    fit_result = None
    fit_candidates_cache: list[InitialWindowCandidate] | None = None

    def ensure_window_fit_candidates() -> list[InitialWindowCandidate]:
        """Compute the cheap fitted/quantile seeds at most once per run."""
        nonlocal fit_result, fit_candidates_cache
        if fit_candidates_cache is not None:
            return fit_candidates_cache
        fit_started = time.perf_counter()
        fit_quad_degree = (
            args.fit_window_quad_degree
            if args.fit_window_quad_degree is not None
            else qdeg
        )
        fit_samples = quadrature_samples_for_fit(
            phi_target,
            rho_design,
            quadrature_degree=int(fit_quad_degree),
        )
        fit_result = fit_phi_window_to_torsion_design(
            phi_target,
            rho_design,
            rho_design_l2=rho_design_l2,
            phi_design_max=phi_target_max,
            eps_ratio=(
                args.eps_ratio
                if args.eps_mode == "relative"
                else max(args.eps_phi / max(c_scale, 1.0e-30), 1.0e-6)
            ),
            rho_amp=params.rho_amp,
            quadrature_degree=fit_quad_degree,
            grid_points=args.fit_window_grid,
            refine_points=args.fit_window_refine_grid,
            refine_passes=args.fit_window_refine_passes,
            histogram_bins=args.fit_window_bins,
            quadrature_samples=fit_samples,
        )
        fit_candidates_cache = build_initial_window_candidates(
            phi_target=phi_target,
            rho_design=rho_design,
            fit_candidate=("density_l2", fit_result.c1, fit_result.c2),
            rho_design_l2=rho_design_l2,
            rho_amp=params.rho_amp,
            target_area=target_area,
            quadrature_degree=fit_quad_degree,
            c_min=c_min,
            c_max=c_upper,
            min_width=min_width,
            args=args,
            quadrature_samples=fit_samples,
        )
        if not fit_candidates_cache:
            raise RuntimeError("failed to generate any automatic fitted-window candidates")
        fit_elapsed = time.perf_counter() - fit_started
        add_phase("window_fit", fit_elapsed)
        root_print(
            comm,
            f"FIT_INIT c1={fit_result.c1:.6e} c2={fit_result.c2:.6e} "
            f"objectiveRel={fit_result.objective_rel:.6e} fitTime={fit_result.elapsed:.3f} "
            f"totalTime={fit_elapsed:.3f}",
        )
        for candidate in fit_candidates_cache:
            root_print(
                comm,
                f"INIT_CANDIDATE name={candidate.name} c1={candidate.c1:.6e} "
                f"c2={candidate.c2:.6e} width={candidate.c2 - candidate.c1:.6e} "
                f"epsPhi={candidate.eps_phi:.6e} score={candidate.score:.6e} "
                f"Lrel={candidate.leakage_rel:.6e} Mrel={candidate.missing_rel:.6e} "
                f"areaRel={candidate.area_rel:.6e} activeJ={candidate.active_jaccard:.6e} "
                f"l2Rel={candidate.l2_rel:.6e}",
            )
        return fit_candidates_cache

    initial_equilibrium_metadata: dict | None = None
    initial_transfer_mode = ""
    initial_transfer_coordinate_error = math.nan
    if args.initial_equilibrium is not None:
        transfer_load_started = time.perf_counter()
        (
            initial_equilibrium_metadata,
            initial_transfer_mode,
            initial_transfer_coordinate_error,
        ) = load_initial_equilibrium_state(
            args.initial_equilibrium,
            target=u,
            alpha_t1=params.alpha_t1,
            alpha_t2=params.alpha_t2,
            rho_amp=params.rho_amp,
            max_residual=float(args.tol_res),
        )
        add_phase("mesh_transfer", time.perf_counter() - transfer_load_started)
        c1_phi = float(initial_equilibrium_metadata["c1_phi"])
        c2_phi = float(initial_equilibrium_metadata["c2_phi"])
        init_candidate_name = "checkpoint_transfer"
        root_print(
            comm,
            f"INITIAL_EQUILIBRIUM status=LOADED path={args.initial_equilibrium} "
            f"mode={initial_transfer_mode} coordinateError={initial_transfer_coordinate_error:.3e} "
            f"unmatchedFraction={float(initial_equilibrium_metadata.get('transfer_unmatched_fraction', 0.0)):.3e} "
            f"areaRelativeDifference={float(initial_equilibrium_metadata.get('transfer_area_relative_difference', 0.0)):.3e} "
            f"c1={c1_phi:.6e} c2={c2_phi:.6e}",
        )
    elif args.c1_phi is not None:
        c1_phi = float(args.c1_phi)
        c2_phi = float(args.c2_phi)
    elif args.init_mode == "legacy" and args.include_fit_init:
        init_candidates = ensure_window_fit_candidates()
        selected_init = min(init_candidates, key=lambda candidate: candidate.score)
        c1_phi = selected_init.c1
        c2_phi = selected_init.c2
        init_candidate_name = selected_init.name
        init_candidate_score = selected_init.score
        root_print(
            comm,
            f"INIT_PRESELECT name={selected_init.name} c1={c1_phi:.6e} c2={c2_phi:.6e} "
            f"score={selected_init.score:.6e}",
        )
    elif args.init_mode == "legacy":
        center = c_min + 0.65 * c_scale
        width = max(0.15 * c_scale, min_width)
        c1_phi = center - 0.5 * width
        c2_phi = center + 0.5 * width
        init_candidate_name = "fallback_center_width"
    else:
        # Constants need finite initial values before the compiled frozen
        # forms are built; this placeholder is immediately replaced by the
        # H^-1 minimizer below and is never continued through homotopy.
        center = c_min + 0.5 * c_scale
        width = max(0.25 * c_scale, min_width)
        c1_phi = center - 0.5 * width
        c2_phi = center + 0.5 * width
        init_candidate_name = "pending_hminus1"

    c1_phi, c2_phi = project_thresholds(c1_phi, c2_phi, c_min=c_min, c_max=c_upper, min_width=min_width)
    eps_phi = epsilon_from_thresholds(args, c1_phi, c2_phi)

    if initial_equilibrium_metadata is None:
        u.x.array[:] = phi_target.x.array
        u.x.scatter_forward()
    c1_const = fem.Constant(domain, PETSc.ScalarType(c1_phi))
    c2_const = fem.Constant(domain, PETSc.ScalarType(c2_phi))
    eps_const = fem.Constant(domain, PETSc.ScalarType(eps_phi))
    initial_transfer_succeeded = False
    initial_transfer_newton: NewtonResult | None = None
    if initial_equilibrium_metadata is not None:
        transfer_correction_started = time.perf_counter()
        update_interpolated(
            rho,
            window_density_const_ufl(
                u, c1_const, c2_const, eps_const, params.rho_amp
            ),
        )
        initial_transfer_newton = solve_equilibrium(
            u=u,
            du=du,
            rho=rho,
            trial=trial,
            test=test,
            dx=dx,
            bc=bc,
            stiffness_form=stiffness_form,
            c1_const=c1_const,
            c2_const=c2_const,
            eps_const=eps_const,
            c1=c1_phi,
            c2=c2_phi,
            eps_phi=eps_phi,
            rho_amp=params.rho_amp,
            tol_res=min(float(args.tol_res), float(homotopy_tolerance(args))),
            args=nonlinear_args,
            prefix="initial_equilibrium_correction",
            stiffness_solver=stiffness_solver,
            phase="initialization",
            outer_iteration=-1,
        )
        correction_elapsed = time.perf_counter() - transfer_correction_started
        add_phase("mesh_transfer", correction_elapsed)
        initial_transfer_succeeded = bool(initial_transfer_newton.converged)
        if initialization_writer is not None:
            initialization_writer.writerow({
                "record": "checkpoint_transfer",
                "runTag": run_tag,
                "method": initial_transfer_mode,
                "c1": c1_phi,
                "c2": c2_phi,
                "width": c2_phi - c1_phi,
                "eps": eps_phi,
                "correctedResidual": initial_transfer_newton.residual,
                "newtonIterations": initial_transfer_newton.iterations,
                "damping": initial_transfer_newton.alpha,
                "backtracks": initial_transfer_newton.backtracks,
                "accepted": int(initial_transfer_succeeded),
                "elapsed": correction_elapsed,
            })
            initialization_handle.flush()
        root_print(
            comm,
            f"INITIAL_EQUILIBRIUM status={'ACCEPTED' if initial_transfer_succeeded else 'REJECTED'} "
            f"correctionStatus={initial_transfer_newton.status} "
            f"iterations={initial_transfer_newton.iterations} "
            f"residual={initial_transfer_newton.residual:.6e} time={correction_elapsed:.3f}s",
        )
        if not initial_transfer_succeeded:
            u.x.array[:] = phi_target.x.array
            u.x.scatter_forward()
            rho.x.array[:] = rho_design.x.array
            rho.x.scatter_forward()
            init_candidate_name = "pending_hminus1_after_transfer_failure"

    selected_frozen_evaluation: FrozenThresholdEvaluation | None = None
    if args.init_mode == "homotopy" and not initial_transfer_succeeded:
        hminus1_phase_started = time.perf_counter()
        root_print(
            comm,
            f"HMINUS1_PHASE status=START selection={'manual' if args.c1_phi is not None else 'automatic'}",
        )
        workspace_started = time.perf_counter()
        root_print(comm, "HMINUS1_PHASE phase=workspace_build_start")
        frozen_objective = FrozenThresholdObjective(
            phi_target=phi_target,
            riesz=frozen_riesz,
            tau_mask=tau_mask,
            test=test,
            dx=dx,
            bc=bc,
            stiffness_solver=stiffness_solver,
            c1_const=c1_const,
            c2_const=c2_const,
            eps_const=eps_const,
            rho_amp=params.rho_amp,
            target_area=target_area,
            c_min=c_min,
            c_max=c_upper,
            min_width=min_width,
            args=args,
        )
        root_print(
            comm,
            f"HMINUS1_PHASE phase=workspace_build_done "
            f"time={time.perf_counter() - workspace_started:.3f}s",
        )
        selection_started = time.perf_counter()
        if args.c1_phi is not None:
            leakage_cap = max(float(args.eta_out), float(args.init_leakage_cap))
            missing_cap = max(float(args.tol_area), float(args.init_missing_cap))
            selected_frozen_evaluation = evaluate_frozen_threshold_objective(
                frozen_objective,
                c1_phi,
                c2_phi,
                method="manual",
            )
            init_candidate_name = "manual"
        else:
            fast_seed_candidates = (
                ensure_window_fit_candidates()
                if args.init_search == "fast"
                else []
            )
            fast_seed_pairs = np.asarray(
                [(candidate.c1, candidate.c2) for candidate in fast_seed_candidates],
                dtype=np.float64,
            ).reshape(-1, 2)
            selected_frozen_evaluation, leakage_cap, missing_cap = optimize_frozen_hminus1_thresholds(
                frozen_objective,
                args=args,
                seed_pairs=fast_seed_pairs,
            )
            init_candidate_name = f"hminus1_{args.init_search}_refined"
        root_print(
            comm,
            f"HMINUS1_PHASE phase=selection_done "
            f"time={time.perf_counter() - selection_started:.3f}s "
            f"evaluations={len(frozen_objective.records)}",
        )
        if args.verify_homotopy_init:
            verification_started = time.perf_counter()
            root_print(comm, "HMINUS1_PHASE phase=gradient_verification_start")
            verify_frozen_hminus1_gradient(frozen_objective, selected_frozen_evaluation)
            root_print(
                comm,
                f"HMINUS1_PHASE phase=gradient_verification_done "
                f"time={time.perf_counter() - verification_started:.3f}s",
            )
        c1_phi = selected_frozen_evaluation.c1
        c2_phi = selected_frozen_evaluation.c2
        eps_phi = selected_frozen_evaluation.eps_phi
        init_candidate_score = selected_frozen_evaluation.psi
        c1_const.value = PETSc.ScalarType(c1_phi)
        c2_const.value = PETSc.ScalarType(c2_phi)
        eps_const.value = PETSc.ScalarType(eps_phi)
        records_started = time.perf_counter()
        write_frozen_initialization_records(
            initialization_writer,
            run_tag=run_tag,
            objective=frozen_objective,
            selected=selected_frozen_evaluation,
            leakage_cap=leakage_cap,
            missing_cap=missing_cap,
        )
        if initialization_handle is not None:
            initialization_handle.flush()
        root_print(
            comm,
            f"HMINUS1_PHASE phase=records_written "
            f"time={time.perf_counter() - records_started:.3f}s "
            f"records={len(frozen_objective.records)}",
        )
        hminus1_elapsed = time.perf_counter() - hminus1_phase_started
        add_phase("hminus1_search", hminus1_elapsed)
        root_print(
            comm,
            f"HMINUS1_INIT c1={c1_phi:.6e} c2={c2_phi:.6e} "
            f"width={c2_phi - c1_phi:.6e} eps={eps_phi:.6e} "
            f"dualResidual={selected_frozen_evaluation.residual_dual:.6e} "
            f"Psi={selected_frozen_evaluation.psi:.6e} "
            f"frozenLrel={selected_frozen_evaluation.leakage_rel:.6e} "
            f"frozenMrel={selected_frozen_evaluation.missing_rel:.6e} "
            f"activityAreaRel={selected_frozen_evaluation.activity_area_rel:.6e} "
            f"projectedGradNorm={selected_frozen_evaluation.projected_grad_norm:.6e} "
            f"numberObjectiveEvaluations={len(frozen_objective.records)} "
            f"Lcap={leakage_cap:.6e} Mcap={missing_cap:.6e} "
            f"phaseTime={hminus1_elapsed:.3f}s",
        )
    homotopy_setup_started = time.perf_counter()
    use_homotopy_initialization = args.init_mode == "homotopy" and not initial_transfer_succeeded
    if use_homotopy_initialization:
        root_print(comm, "HOMOTOPY_SETUP status=START")
    update_interpolated(rho, window_density_const_ufl(u, c1_const, c2_const, eps_const, params.rho_amp))
    homotopy_workspace = (
        HomotopySolveWorkspace(
            u=u,
            trial=trial,
            test=test,
            dx=dx,
            bc=bc,
            target_density=target_density,
            c1_const=c1_const,
            c2_const=c2_const,
            eps_const=eps_const,
            rho_amp=params.rho_amp,
            args=homotopy_args,
        )
        if use_homotopy_initialization
        else None
    )
    if use_homotopy_initialization:
        root_print(
            comm,
            f"HOMOTOPY_SETUP status=DONE time={time.perf_counter() - homotopy_setup_started:.3f}s",
        )
    initialization_diagnostics = (
        InitializationDiagnostics(
            u=u,
            rho=rho,
            rho_design=rho_design,
            rho_design_l2=rho_design_l2,
            tau_mask=tau_mask,
            dx=dx,
            c1_const=c1_const,
            c2_const=c2_const,
            eps_const=eps_const,
            kappa=args.kappa,
            active_threshold=params.active_threshold,
            rho_amp=params.rho_amp,
            target_area=target_area,
        )
        if init_candidates
        else None
    )

    selected_homotopy_result: HomotopyResult | None = None
    homotopy_attempts: list[tuple[str, HomotopyResult]] = []

    def project_initial_candidate(
            candidate: InitialWindowCandidate,
            index: int,
    ) -> ProjectedInitialCandidate:
        """Run the unchanged legacy direct-Newton candidate projection."""
        if args.init_mode != "legacy":
            raise RuntimeError("legacy candidate projection entered outside legacy mode")
        c1_candidate, c2_candidate = project_thresholds(
            candidate.c1,
            candidate.c2,
            c_min=c_min,
            c_max=c_upper,
            min_width=min_width,
        )
        eps_candidate = epsilon_from_thresholds(args, c1_candidate, c2_candidate)
        u.x.array[:] = phi_target.x.array
        u.x.scatter_forward()
        rho.x.array[:] = rho_design.x.array
        rho.x.scatter_forward()
        projection_start = time.perf_counter()
        projection_tol = inner_tolerance(args, candidate.leakage_rel + candidate.missing_rel)
        newton = solve_equilibrium(
            u=u,
            du=du,
            rho=rho,
            trial=trial,
            test=test,
            dx=dx,
            bc=bc,
            stiffness_form=stiffness_form,
            c1_const=c1_const,
            c2_const=c2_const,
            eps_const=eps_const,
            c1=c1_candidate,
            c2=c2_candidate,
            eps_phi=eps_candidate,
            rho_amp=params.rho_amp,
            tol_res=projection_tol,
            args=nonlinear_args,
            prefix=f"init_{index}_{slug_for_path(candidate.name)}",
            stiffness_solver=stiffness_solver,
            phase="initialization",
            outer_iteration=-1,
        )
        projection_converged = newton.converged

        projection_time = time.perf_counter() - projection_start
        if initialization_diagnostics is None:
            raise RuntimeError("automatic candidate diagnostics were not initialized")
        metrics, diagnostics = initialization_diagnostics.evaluate(
            c1=c1_candidate,
            c2=c2_candidate,
            eps_phi=eps_candidate,
        )
        projected_score = projected_initial_candidate_score(
            metrics=metrics,
            diagnostics=diagnostics,
            target_area=target_area,
            converged=projection_converged,
        )
        area_rel = metrics.activity_area / max(target_area, 1.0e-30)
        projection_detail = (
            f"mode=legacy tol={projection_tol:.3e} status={newton.status} "
            f"converged={int(newton.converged)} iters={newton.iterations} "
            f"residual={newton.residual:.6e}"
        )
        root_print(
            comm,
            f"INIT_PROJECT name={candidate.name} c1={c1_candidate:.6e} c2={c2_candidate:.6e} "
            f"width={c2_candidate - c1_candidate:.6e} epsPhi={eps_candidate:.6e} "
            f"{projection_detail} time={projection_time:.3f} score={projected_score:.6e} "
            f"Lrel={metrics.leakage_rel:.6e} Mrel={metrics.missing_rel:.6e} "
            f"areaRel={area_rel:.6e} activeJ={diagnostics['activeJaccard']:.6e} "
            f"rhoRel={diagnostics['relRhoDesign']:.6e}",
        )
        return ProjectedInitialCandidate(
            base=candidate,
            newton=newton,
            metrics=metrics,
            diagnostics=diagnostics,
            score=projected_score,
            state=u.x.array.copy(),
            density=rho.x.array.copy(),
        )

    if initial_transfer_succeeded:
        trajectory.add(
            stage="checkpoint_transfer",
            phi=u,
            rho=rho,
            c1=c1_phi,
            c2=c2_phi,
            eps_phi=eps_phi,
            homotopy_lambda=1.0,
            force=True,
        )
    elif init_candidates:
        projection_candidates = init_candidates
        if args.legacy_project_preselected_only:
            projection_candidates = [min(init_candidates, key=lambda candidate: candidate.score)]
        root_print(
            comm,
            f"INIT_PROJECT_POLICY candidates={len(projection_candidates)} "
            f"preselectedOnly={int(args.legacy_project_preselected_only)}",
        )
        projected_candidates = [
            project_initial_candidate(candidate, index)
            for index, candidate in enumerate(projection_candidates)
        ]
        # Preserve the legacy selection rule exactly: failed direct Newton
        # projections retain the old finite score penalty and remain
        # reportable when all candidates fail.
        selected_projected = min(projected_candidates, key=lambda candidate: candidate.score)
        c1_phi = selected_projected.base.c1
        c2_phi = selected_projected.base.c2
        c1_phi, c2_phi = project_thresholds(c1_phi, c2_phi, c_min=c_min, c_max=c_upper, min_width=min_width)
        eps_phi = epsilon_from_thresholds(args, c1_phi, c2_phi)
        c1_const.value = PETSc.ScalarType(c1_phi)
        c2_const.value = PETSc.ScalarType(c2_phi)
        eps_const.value = PETSc.ScalarType(eps_phi)
        u.x.array[:] = selected_projected.state
        u.x.scatter_forward()
        rho.x.array[:] = selected_projected.density
        rho.x.scatter_forward()
        init_candidate_name = selected_projected.base.name
        init_candidate_score = selected_projected.score
        selected_homotopy_result = selected_projected.homotopy
        root_print(
            comm,
            f"INIT_PROJECT_SELECT mode={args.init_mode} name={init_candidate_name} "
            f"c1={c1_phi:.6e} c2={c2_phi:.6e} score={init_candidate_score:.6e} "
            f"residual={selected_projected.newton.residual:.6e}",
        )
        trajectory.add(
            stage="selected_fit_projection",
            phi=u,
            rho=rho,
            c1=c1_phi,
            c2=c2_phi,
            eps_phi=eps_phi,
            homotopy_lambda=1.0,
            force=True,
        )
    elif use_homotopy_initialization:
        # The manual pair or the single frozen-H^-1 minimizer is continued
        # exactly once before the reduced optimization begins.
        u.x.array[:] = phi_target.x.array
        u.x.scatter_forward()
        rho.x.array[:] = rho_design.x.array
        rho.x.scatter_forward()
        trajectory.add(
            stage="selected_seed",
            phi=u,
            rho=rho,
            c1=c1_phi,
            c2=c2_phi,
            eps_phi=eps_phi,
            homotopy_lambda=0.0,
            force=True,
        )

        def record_homotopy_state(stage_index: int, lambda_value: float) -> None:
            if homotopy_workspace is None:
                raise RuntimeError("homotopy trajectory callback has no workspace")
            homotopy_workspace.update_density(rho)
            trajectory.add(
                stage="homotopy", phi=u, rho=rho, c1=c1_phi, c2=c2_phi,
                eps_phi=eps_phi, homotopy_lambda=lambda_value,
                sequence=stage_index,
            )

        primary_homotopy_error: Exception | None = None
        primary_homotopy_started = time.perf_counter()
        try:
            selected_homotopy_result = solve_homotopy_initialization(
                u=u,
                du=du,
                tangent=homotopy_tangent,
                rho=rho,
                trial=trial,
                test=test,
                dx=dx,
                bc=bc,
                stiffness_form=stiffness_form,
                stiffness_solver=stiffness_solver,
                homotopy_workspace=homotopy_workspace,
                target_density=target_density,
                c1_const=c1_const,
                c2_const=c2_const,
                eps_const=eps_const,
                c1=c1_phi,
                c2=c2_phi,
                eps_phi=eps_phi,
                rho_amp=params.rho_amp,
                args=homotopy_args,
                prefix=f"init_homotopy_{slug_for_path(init_candidate_name)}",
                initialization_writer=initialization_writer,
                initialization_handle=initialization_handle,
                run_tag=run_tag,
                frozen_residual_dual=(
                    selected_frozen_evaluation.residual_dual
                    if selected_frozen_evaluation is not None
                    else None
                ),
                trajectory_callback=record_homotopy_state,
            )
        except Exception as error:
            primary_homotopy_error = error

        primary_exception_report = synchronize_homotopy_initialization_exception(
            comm,
            primary_homotopy_error,
        )
        if primary_exception_report is not None:
            primary_elapsed = time.perf_counter() - primary_homotopy_started
            root_print(
                comm,
                f"INIT_HOMOTOPY_EXCEPTION status={primary_exception_report.status} "
                f"recoverable={int(primary_exception_report.recoverable)} "
                f"fallback={args.init_fallback} "
                f"details={primary_exception_report.summary}",
            )
            if initialization_writer is not None:
                initialization_writer.writerow({
                    "record": "homotopy_exception",
                    "runTag": run_tag,
                    "method": "source_homotopy_exception",
                    "c1": c1_phi,
                    "c2": c2_phi,
                    "width": c2_phi - c1_phi,
                    "eps": eps_phi,
                    "accepted": 0,
                    "elapsed": primary_elapsed,
                })
                if initialization_handle is not None:
                    initialization_handle.flush()
            write_newton_record(
                homotopy_args,
                comm,
                solve=f"init_homotopy_{slug_for_path(init_candidate_name)}",
                phase="homotopy",
                outer_iteration=-1,
                homotopy_lambda=math.nan,
                nonlinear_iteration=-1,
                status=primary_exception_report.status,
                residual=math.nan,
                fallback_used=(
                    primary_exception_report.recoverable
                    and args.init_fallback == "window-fit"
                ),
                elapsed=primary_elapsed,
            )
            require_window_fit_homotopy_exception_fallback(
                primary_exception_report,
                init_fallback=args.init_fallback,
                local_error=primary_homotopy_error,
            )

            # The failed KSP and every partially corrected vector are discarded
            # collectively before constructing the independent fitted-window
            # rescue. Rebuilding the workspace avoids reusing a diverged KSP.
            homotopy_workspace.close()
            u.x.array[:] = phi_target.x.array
            u.x.scatter_forward()
            rho.x.array[:] = rho_design.x.array
            rho.x.scatter_forward()
            du.x.array[:] = 0.0
            du.x.scatter_forward()
            homotopy_tangent.x.array[:] = 0.0
            homotopy_tangent.x.scatter_forward()
            c1_const.value = PETSc.ScalarType(c1_phi)
            c2_const.value = PETSc.ScalarType(c2_phi)
            eps_const.value = PETSc.ScalarType(eps_phi)
            homotopy_workspace = HomotopySolveWorkspace(
                u=u,
                trial=trial,
                test=test,
                dx=dx,
                bc=bc,
                target_density=target_density,
                c1_const=c1_const,
                c2_const=c2_const,
                eps_const=eps_const,
                rho_amp=params.rho_amp,
                args=homotopy_args,
            )
            selected_homotopy_result = HomotopyResult(
                status=primary_exception_report.status,
                converged=False,
                lambda_final=math.nan,
                stages=0,
                rejected_steps=1,
                total_newton_iterations=0,
                tangent_solve_time=0.0,
                newton_solve_time=0.0,
                elapsed=primary_elapsed,
                last_newton=NewtonResult(
                    status=primary_exception_report.status,
                    converged=False,
                    iterations=0,
                    residual=math.nan,
                    step_h1=math.nan,
                    alpha=0.0,
                    backtracks=0,
                    solve_time=0.0,
                ),
            )
        else:
            if selected_homotopy_result is None:
                raise RuntimeError(
                    "homotopy initialization returned no result and no exception"
                )
            homotopy_workspace.update_nonlinear_density(rho)

        homotopy_attempts.append((init_candidate_name, selected_homotopy_result))
        add_phase("homotopy", selected_homotopy_result.elapsed)
        if (
                not selected_homotopy_result.converged
                and args.init_fallback == "window-fit"
        ):
            primary_result = selected_homotopy_result
            trajectory.add(
                stage=(
                    "homotopy_primary_exception_reset"
                    if primary_exception_report is not None
                    else "homotopy_primary_failure"
                ),
                phi=u,
                rho=rho,
                c1=c1_phi,
                c2=c2_phi,
                eps_phi=eps_phi,
                homotopy_lambda=primary_result.lambda_final,
                sequence=primary_result.stages,
                force=True,
            )
            fitted_candidates = ensure_window_fit_candidates()
            fitted_by_preference = sorted(
                fitted_candidates,
                key=lambda candidate: (
                    candidate.name != "density_l2",
                    candidate.score,
                    candidate.name,
                ),
            )
            threshold_separation = max(1.0e-6 * c_scale, 1.0e-14)
            fallback_candidate = next(
                (
                    candidate
                    for candidate in fitted_by_preference
                    if np.linalg.norm(
                        np.array(
                            [candidate.c1 - c1_phi, candidate.c2 - c2_phi],
                            dtype=np.float64,
                        )
                    ) > threshold_separation
                ),
                None,
            )
            if fallback_candidate is None:
                root_print(
                    comm,
                    "INIT_FALLBACK status=SKIPPED method=window_fit "
                    "reason=no_distinct_threshold_pair",
                )
            else:
                c1_phi, c2_phi = project_thresholds(
                    fallback_candidate.c1,
                    fallback_candidate.c2,
                    c_min=c_min,
                    c_max=c_upper,
                    min_width=min_width,
                )
            if fallback_candidate is not None:
                eps_phi = epsilon_from_thresholds(args, c1_phi, c2_phi)
                c1_const.value = PETSc.ScalarType(c1_phi)
                c2_const.value = PETSc.ScalarType(c2_phi)
                eps_const.value = PETSc.ScalarType(eps_phi)
                u.x.array[:] = phi_target.x.array
                u.x.scatter_forward()
                rho.x.array[:] = rho_design.x.array
                rho.x.scatter_forward()
                init_candidate_name = "window_fit_fallback"
                init_candidate_score = fallback_candidate.score
                trajectory.add(
                    stage="window_fit_fallback_seed",
                    phi=u,
                    rho=rho,
                    c1=c1_phi,
                    c2=c2_phi,
                    eps_phi=eps_phi,
                    homotopy_lambda=0.0,
                    force=True,
                )
                root_print(
                    comm,
                    f"INIT_FALLBACK status=START method=window_fit "
                    f"primaryStatus={primary_result.status} "
                    f"primaryLambda={primary_result.lambda_final:.6e} "
                    f"c1={c1_phi:.6e} c2={c2_phi:.6e}",
                )
                selected_homotopy_result = solve_homotopy_initialization(
                    u=u,
                    du=du,
                    tangent=homotopy_tangent,
                    rho=rho,
                    trial=trial,
                    test=test,
                    dx=dx,
                    bc=bc,
                    stiffness_form=stiffness_form,
                    stiffness_solver=stiffness_solver,
                    homotopy_workspace=homotopy_workspace,
                    target_density=target_density,
                    c1_const=c1_const,
                    c2_const=c2_const,
                    eps_const=eps_const,
                    c1=c1_phi,
                    c2=c2_phi,
                    eps_phi=eps_phi,
                    rho_amp=params.rho_amp,
                    args=homotopy_args,
                    prefix="init_homotopy_window_fit_fallback",
                    initialization_writer=initialization_writer,
                    initialization_handle=initialization_handle,
                    run_tag=run_tag,
                    initialization_method="window_fit_source_homotopy",
                    frozen_residual_dual=None,
                    trajectory_callback=record_homotopy_state,
                )
                homotopy_attempts.append((init_candidate_name, selected_homotopy_result))
                add_phase("homotopy", selected_homotopy_result.elapsed)
                homotopy_workspace.update_nonlinear_density(rho)
                root_print(
                    comm,
                    f"INIT_FALLBACK status={'CONVERGED' if selected_homotopy_result.converged else 'FAILED'} "
                    f"method=window_fit homotopyStatus={selected_homotopy_result.status} "
                    f"lambda={selected_homotopy_result.lambda_final:.6e} "
                    f"iterations={selected_homotopy_result.total_newton_iterations}",
                )
        if not selected_homotopy_result.converged:
            failure_message = (
                f"homotopy initialization failed at lambda={selected_homotopy_result.lambda_final:.6e} "
                f"with status {selected_homotopy_result.status} after "
                f"{len(homotopy_attempts)} attempt(s)"
            )
            trajectory.add(
                stage="homotopy_terminal_failure", phi=u, rho=rho,
                c1=c1_phi, c2=c2_phi, eps_phi=eps_phi,
                homotopy_lambda=selected_homotopy_result.lambda_final,
                sequence=selected_homotopy_result.stages, force=True,
            )
            trajectory.write({
                "run_tag": run_tag,
                "order": int(args.order),
                "quadrature_degree": int(qdeg),
                "alpha_t1": float(params.alpha_t1),
                "alpha_t2": float(params.alpha_t2),
                "c1_t": float(c1_t),
                "c2_t": float(c2_t),
                "rho_amp": float(params.rho_amp),
                "complete": False,
                "terminal_stage": "homotopy",
                "terminal_status": selected_homotopy_result.status,
                "failure_message": failure_message,
                "lambda_final": float(selected_homotopy_result.lambda_final),
            })
            phase_times["total"] = time.perf_counter() - total_start
            phase_calls["total"] = 1
            write_phases(f"failed during homotopy: {selected_homotopy_result.status}")
            if newton_handle is not None:
                newton_handle.close()
            homotopy_workspace.close()
            if initialization_handle is not None:
                initialization_handle.close()
            raise RuntimeError(failure_message)
        root_print(
            comm,
            f"INIT_HOMOTOPY_SELECT name={init_candidate_name} c1={c1_phi:.6e} c2={c2_phi:.6e} "
            f"stages={selected_homotopy_result.stages} rejected={selected_homotopy_result.rejected_steps} "
            f"newtonIts={selected_homotopy_result.total_newton_iterations} "
            f"residual={selected_homotopy_result.last_newton.residual:.6e}",
        )

    if homotopy_workspace is not None:
        homotopy_workspace.close()
    if initialization_handle is not None:
        initialization_handle.close()

    root_print(
        comm,
        f"WINDOW_INIT c1Phi={c1_phi:.6e} c2Phi={c2_phi:.6e} "
        f"width={c2_phi - c1_phi:.6e} epsPhi={eps_phi:.6e} "
        f"initMode={args.init_mode} init={init_candidate_name} initScore={init_candidate_score:.6e}",
    )

    def emit_severe_plot(
            *,
            stage: str,
            outer_k: int,
            token: str,
            eps_phi_value: float,
            residual: float,
            title_suffix: str,
    ) -> None:
        """Emit a high-frequency diagnostic plot for Newton/refit internals.

        Severe plotting is intentionally isolated from the normal
        ``--plot-every`` cadence.  It is meant for debugging branch following:
        every accepted Newton update can be visualized, and the predicted or
        accepted/rejected threshold-refit state can be inspected immediately.
        """
        if not args.plot_severe:
            return
        phi_diff.x.array[:] = u.x.array - phi_target.x.array
        phi_diff.x.scatter_forward()
        plotter.emit(
            [T, tau_band, phi_target, u, rho, phi_diff],
            [
                "Torsion T",
                "tau band",
                "phiT",
                f"phi {title_suffix}",
                f"rho(phi) {title_suffix}",
                "phi-phiT",
            ],
            stage=stage,
            ieps=0,
            k=outer_k,
            eps_phi=eps_phi_value,
            residual=residual,
            metrics={},
            token=token,
            save=bool(args.save_frames),
            show=True,
            nt=nt,
            ndof=ndof,
        )

    def make_newton_plot_callback(
            *,
            outer_k: int,
            stage: str,
    ) -> Callable[[str, int, float, float, int, float, float, float], None] | None:
        """Build the severe plotting hook for one fixed-threshold Newton solve."""
        if not args.plot_severe:
            return None

        def callback(
                prefix: str,
                newton_k: int,
                residual: float,
                alpha: float,
                backtracks: int,
                c1_value: float,
                c2_value: float,
                eps_phi_value: float,
        ) -> None:
            """Emit the current accepted Newton state for severe plotting."""
            token = f"{slug_for_path(prefix)}_newton_{newton_k:03d}"
            suffix = (
                f"{stage} n={newton_k} "
                f"c=({c1_value:.3e},{c2_value:.3e}) "
                f"a={alpha:.2e} bt={backtracks}"
            )
            emit_severe_plot(
                stage=stage,
                outer_k=outer_k,
                token=token,
                eps_phi_value=eps_phi_value,
                residual=residual,
                title_suffix=suffix,
            )

        return callback

    fieldnames = [
        "record", "runTag", "k", "nt", "ndof", "status", "accepted",
        "c1Phi", "c2Phi", "width", "epsPhi", "homotopyLambda", "trustRadius",
        "innerTol", "newtonStatus", "newtonSteps", "residual", "resEuclid",
        "leakage", "missing", "leakageRel", "missingRel", "activityArea",
        "certifiedArea", "certifiedLeakage", "certifiedMissing",
        "gradL1", "gradL2", "gradM1", "gradM2", "directL1", "directL2", "directM1", "directM2",
        "projectedGradNorm", "sensIts1", "sensIts2", "sensRes1", "sensRes2", "sensSolveTime",
        "stepObjective", "dc1", "dc2", "predictedReduction", "actualReduction", "rhoRatio",
        "branchOverlap", "activeJaccard", "activeRecall", "activePrecision", "activeDice",
        "plateauJaccard", "rhoRel", "massRel", "phiRel", "solveTime", "stepTime",
    ]
    opt_handle = opt_csv.open("w", newline="", encoding="utf-8") if comm.rank == 0 else None
    writer = csv.DictWriter(opt_handle, fieldnames=fieldnames) if comm.rank == 0 else None
    if writer is not None:
        writer.writeheader()

    final_status = "MAX_OPT_IT"
    final_metrics: BandMetrics | None = None
    final_newton: NewtonResult | None = None
    final_compute_metrics: dict[str, float] | None = None
    accepted_steps = 0
    certified_stale_steps = 0
    best_certified_score = math.inf
    best_certified_state: np.ndarray | None = None
    best_certified_c1 = math.nan
    best_certified_c2 = math.nan
    best_certified_iteration = -1
    cached_reduced_gradient: ReducedGradient | None = None
    trust_radius_abs = min(max(float(args.trust_radius) * c_scale, float(args.trust_radius_min) * c_scale), float(args.trust_radius_max) * c_scale)
    previous_discrepancy_rel = 2.0
    inexact_replay_snapshots: list[InexactNewtonReplaySnapshot] = []
    if args.run_inexact_newton_study:
        inexact_replay_snapshots.append(InexactNewtonReplaySnapshot(
            snapshot_id="post_homotopy_seed",
            predictor_state=u.x.array.copy(),
            reference_seed_state=u.x.array.copy(),
            c1=float(c1_phi),
            c2=float(c2_phi),
            trust_radius=float(trust_radius_abs),
            discrepancy_rel=float(previous_discrepancy_rel),
            outer_iteration=-1,
        ))
    frame_every = args.frame_every if args.frame_every is not None else args.plot_every

    try:
        for k in range(int(args.max_opt_it) + 1):
            step_start = time.perf_counter()
            eps_phi = epsilon_from_thresholds(args, c1_phi, c2_phi)
            inner_tol = inner_tolerance(args, previous_discrepancy_rel)

            # Algorithm step 1: at fixed thresholds, project the current state
            # onto the selected semilinear branch.  The tolerance can be
            # inexact here, but final reporting later uses an exact projection.
            inner_start = time.perf_counter()
            newton = solve_equilibrium(
                u=u,
                du=du,
                rho=rho,
                trial=trial,
                test=test,
                dx=dx,
                bc=bc,
                stiffness_form=stiffness_form,
                c1_const=c1_const,
                c2_const=c2_const,
                eps_const=eps_const,
                c1=c1_phi,
                c2=c2_phi,
                eps_phi=eps_phi,
                rho_amp=params.rho_amp,
                tol_res=inner_tol,
                args=nonlinear_args,
                prefix=f"outer_{k}",
                homotopy_lambda=optimization_homotopy_lambda,
                homotopy_target_density=target_density,
                stiffness_solver=stiffness_solver,
                plot_callback=make_newton_plot_callback(outer_k=k, stage="NEWTON_OUTER"),
                phase="outer_projection",
                outer_iteration=k,
            )
            inner_time = time.perf_counter() - inner_start
            log_algorithm_step(
                comm,
                args,
                iteration=k,
                step=1,
                label="inner_damped_newton",
                elapsed=inner_time,
                detail=(
                    f"tol={inner_tol:.3e} status={newton.status} converged={int(newton.converged)} "
                    f"iters={newton.iterations} residual={newton.residual:.6e} "
                    f"linearSolveTime={newton.solve_time:.6f}s"
                ),
            )
            final_newton = newton

            # Algorithm step 2: evaluate the soft activity window and both
            # geometric discrepancies.  The separate diagnostic block computes
            # legacy density/active-set metrics for comparison with the other
            # torsion-initialized Newton runners.
            band_start = time.perf_counter()
            metrics = evaluate_band_metrics(
                comm=comm,
                u=u,
                tau_mask=tau_mask,
                dx=dx,
                c1_const=c1_const,
                c2_const=c2_const,
                eps_const=eps_const,
                c1=c1_phi,
                c2=c2_phi,
                eps_phi=eps_phi,
                kappa=args.kappa,
                target_area=target_area,
            )
            band_time = time.perf_counter() - band_start
            previous_discrepancy_rel = threshold_merit_rel(metrics, args)
            final_metrics = metrics
            residual_form_for_metrics = (
                ufl.inner(ufl.grad(u), ufl.grad(test))
                - optimization_source_density() * test
            ) * dx
            diagnostic_start = time.perf_counter()
            update_interpolated(rho, window_density_const_ufl(u, c1_const, c2_const, eps_const, params.rho_amp))
            diagnostic_metrics = compute_metrics(
                u=u,
                rho=rho,
                rho_design=rho_design,
                rho_design_l2=rho_design_l2,
                residual_form=residual_form_for_metrics,
                bc=bc,
                dx=dx,
                c2_phi=c2_phi,
                active_threshold=params.active_threshold,
                plateau_threshold=params.plateau_threshold,
                rho_amp=params.rho_amp,
            )
            diagnostic_time = time.perf_counter() - diagnostic_start
            log_algorithm_step(
                comm,
                args,
                iteration=k,
                step=2,
                label="evaluate_window_and_discrepancy",
                elapsed=band_time + diagnostic_time,
                detail=(
                    f"L={metrics.leakage:.6e} M={metrics.missing:.6e} "
                    f"Lrel={metrics.leakage_rel:.6e} Mrel={metrics.missing_rel:.6e} "
                    f"activityArea={metrics.activity_area:.6e} bandEvalTime={band_time:.6f}s "
                    f"diagnosticTime={diagnostic_time:.6f}s"
                ),
            )
            final_compute_metrics = diagnostic_metrics
            active_overlap_area = diagnostic_metrics["activeOverlapArea"]
            active_recall = active_overlap_area / max(diagnostic_metrics["activeDesignArea"], 1.0e-30)
            active_precision = active_overlap_area / max(diagnostic_metrics["activeArea"], 1.0e-30)
            active_dice = 2.0 * active_overlap_area / max(
                diagnostic_metrics["activeDesignArea"] + diagnostic_metrics["activeArea"],
                1.0e-30,
            )
            phi_l2 = math.sqrt(max(assemble_scalar(comm, (u - phi_target) ** 2 * dx), 0.0))
            phi_rel = phi_l2 / max(phi_target_l2, 1.0e-30)
            mass_rel = diagnostic_metrics["massRhoMinusDesign"] / max(abs(rho_design_mass), 1.0e-30)

            gradient: ReducedGradient | None = None
            gradient_cached = False
            step: ParameterStep | None = None
            status = "ITERATE"
            accepted = False
            trial_predictor_state: np.ndarray | None = None
            trial_predictor_trust_radius = float(trust_radius_abs)
            trial_predictor_discrepancy = threshold_merit_rel(metrics, args)
            actual_reduction = 0.0
            rho_ratio = math.nan
            branch_overlap = 1.0
            projected_grad_norm = math.nan
            inner_projection_ok = (
                newton.converged
                or newton.residual <= max(10.0 * float(inner_tol), float(args.tol_res))
            )
            if not inner_projection_ok:
                status = "PDE_FAILURE"
                final_status = status
            stop_ready = (
                inner_projection_ok
                and metrics.leakage_rel <= float(args.eta_out)
                and metrics.missing_rel <= float(args.tol_area)
            )

            if k < int(args.max_opt_it) and inner_projection_ok:
                # Algorithm steps 3-7: assemble the Newton matrix, assemble
                # parameter RHS vectors, solve the two sensitivity equations,
                # assemble functional derivatives, and combine them into
                # reduced leakage/missing-area gradients.
                if cached_reduced_gradient is not None:
                    gradient = cached_reduced_gradient
                    gradient_cached = True
                    gradient_time = 0.0
                else:
                    gradient_start = time.perf_counter()
                    gradient = compute_reduced_gradient(
                        u=u,
                        s1=s1,
                        s2=s2,
                        trial=trial,
                        test=test,
                        dx=dx,
                        bc=bc,
                        tau_mask=tau_mask,
                        c1_const=c1_const,
                        c2_const=c2_const,
                        eps_const=eps_const,
                        rho_amp=params.rho_amp,
                        eps_mode=args.eps_mode,
                        eps_ratio=args.eps_ratio,
                        homotopy_lambda=optimization_homotopy_lambda,
                        args=sensitivity_args,
                        iteration=k,
                    )
                    gradient_time = time.perf_counter() - gradient_start
                    add_phase("sensitivities", gradient.solve_time)
                    add_phase(
                        "reduced_gradients", gradient.gradient_assembly_time,
                    )
                log_algorithm_step(
                    comm,
                    args,
                    iteration=k,
                    step=3,
                    label="assemble_newton_matrix",
                    elapsed=0.0 if gradient_cached else gradient.matrix_assembly_time,
                    detail=f"cached={int(gradient_cached)} matrix=J_U reused for sensitivity solves",
                )
                log_algorithm_step(
                    comm,
                    args,
                    iteration=k,
                    step=4,
                    label="assemble_residual_parameter_derivatives",
                    elapsed=0.0 if gradient_cached else gradient.rhs_assembly_time,
                    detail="columns=dr/dc1,dr/dc2",
                )
                log_algorithm_step(
                    comm,
                    args,
                    iteration=k,
                    step=5,
                    label="solve_equilibrium_sensitivities",
                    elapsed=0.0 if gradient_cached else gradient.linear_solve_time,
                    detail=(
                        f"cached={int(gradient_cached)} total={0.0 if gradient_cached else gradient.solve_time:.6f}s "
                        f"its=({gradient.solve_iterations[0]},{gradient.solve_iterations[1]}) "
                        f"linRes=({gradient.solve_residuals[0]:.3e},{gradient.solve_residuals[1]:.3e})"
                    ),
                )
                log_algorithm_step(
                    comm,
                    args,
                    iteration=k,
                    step=6,
                    label="assemble_functional_gradients",
                    elapsed=0.0 if gradient_cached else gradient.gradient_assembly_time,
                    detail=(
                        f"directL=({gradient.direct_l[0]:.6e},{gradient.direct_l[1]:.6e}) "
                        f"directM=({gradient.direct_m[0]:.6e},{gradient.direct_m[1]:.6e})"
                    ),
                )
                log_algorithm_step(
                    comm,
                    args,
                    iteration=k,
                    step=7,
                    label="form_reduced_gradients",
                    elapsed=gradient_time,
                    detail=(
                        f"gradL=({gradient.grad_l[0]:.6e},{gradient.grad_l[1]:.6e}) "
                        f"gradM=({gradient.grad_m[0]:.6e},{gradient.grad_m[1]:.6e})"
                    ),
                )
                projected_grad_norm = float(np.linalg.norm(threshold_objective_gradient(metrics, gradient, args)))
                stop_ready = stop_ready and projected_grad_norm <= float(args.tol_grad)
                if stop_ready:
                    status = "CONVERGED"
                    final_status = status
                else:
                    # Algorithm step 8: solve the two-dimensional constrained
                    # trust-region model in threshold space.
                    trust_start = time.perf_counter()
                    step = choose_parameter_step(
                        c1=c1_phi,
                        c2=c2_phi,
                        c_min=c_min,
                        c_max=c_upper,
                        min_width=min_width,
                        metrics=metrics,
                        gradient=gradient,
                        trust_radius_abs=trust_radius_abs,
                        args=args,
                    )
                    trust_time = time.perf_counter() - trust_start
                    log_algorithm_step(
                        comm,
                        args,
                        iteration=k,
                        step=8,
                        label="solve_constrained_trust_region_step",
                        elapsed=trust_time,
                        detail=(
                            f"status={step.status} objective={step.objective_name} "
                            f"dc=({step.dc[0]:.6e},{step.dc[1]:.6e}) "
                            f"norm={step.step_norm:.6e} predictedReduction={step.predicted_reduction:.6e} "
                            f"radius={trust_radius_abs:.6e}"
                        ),
                    )
                    if step.step_norm <= max(float(args.trust_radius_min) * c_scale, 1.0e-14):
                        status = "STEP_TOO_SMALL"
                        final_status = status
                    else:
                        # Algorithm step 9: use the sensitivity predictor
                        # U_trial = U + S*dc before correcting with Newton.
                        predictor_start = time.perf_counter()
                        update_interpolated(activity_ref, window_activity_const_ufl(u, c1_const, c2_const, eps_const))
                        old_u = u.x.array.copy()
                        old_c1 = c1_phi
                        old_c2 = c2_phi
                        old_eps = eps_phi
                        trial_c1 = c1_phi + float(step.dc[0])
                        trial_c2 = c2_phi + float(step.dc[1])
                        trial_eps = epsilon_from_thresholds(args, trial_c1, trial_c2)
                        u.x.array[:] = old_u + float(step.dc[0]) * s1.x.array + float(step.dc[1]) * s2.x.array
                        u.x.scatter_forward()
                        if args.run_inexact_newton_study:
                            trial_predictor_state = u.x.array.copy()
                        c1_const.value = PETSc.ScalarType(trial_c1)
                        c2_const.value = PETSc.ScalarType(trial_c2)
                        eps_const.value = PETSc.ScalarType(trial_eps)
                        update_interpolated(rho, window_density_const_ufl(u, c1_const, c2_const, eps_const, params.rho_amp))
                        trial_inner_tol = inner_tolerance(args, previous_discrepancy_rel)
                        emit_severe_plot(
                            stage="REFIT_PREDICT",
                            outer_k=k,
                            token=f"refit_predict_{k:03d}",
                            eps_phi_value=trial_eps,
                            residual=newton.residual,
                            title_suffix=(
                                f"predict k={k} "
                                f"c=({trial_c1:.3e},{trial_c2:.3e}) "
                                f"dc=({step.dc[0]:.2e},{step.dc[1]:.2e})"
                            ),
                        )
                        predictor_time = time.perf_counter() - predictor_start
                        log_algorithm_step(
                            comm,
                            args,
                            iteration=k,
                            step=9,
                            label="predict_trial_equilibrium",
                            elapsed=predictor_time,
                            detail=(
                                f"trialC=({trial_c1:.6e},{trial_c2:.6e}) trialEps={trial_eps:.6e} "
                                f"predictor=U+Sdc"
                            ),
                        )
                        # Algorithm step 10: correct the predicted state at
                        # the trial thresholds by damped Newton.
                        correction_start = time.perf_counter()
                        trial_newton = solve_equilibrium(
                            u=u,
                            du=du,
                            rho=rho,
                            trial=trial,
                            test=test,
                            dx=dx,
                            bc=bc,
                            stiffness_form=stiffness_form,
                            c1_const=c1_const,
                            c2_const=c2_const,
                            eps_const=eps_const,
                            c1=trial_c1,
                            c2=trial_c2,
                            eps_phi=trial_eps,
                            rho_amp=params.rho_amp,
                            tol_res=trial_inner_tol,
                            args=trial_nonlinear_args,
                            prefix=f"outer_{k}_trial",
                            homotopy_lambda=optimization_homotopy_lambda,
                            homotopy_target_density=target_density,
                            stiffness_solver=stiffness_solver,
                            plot_callback=make_newton_plot_callback(outer_k=k, stage="NEWTON_TRIAL"),
                            phase="trial_correction",
                            outer_iteration=k,
                        )
                        correction_time = time.perf_counter() - correction_start
                        add_phase("trial_corrections", correction_time)
                        log_algorithm_step(
                            comm,
                            args,
                            iteration=k,
                            step=10,
                            label="correct_predictor_with_newton",
                            elapsed=correction_time,
                            detail=(
                                f"tol={trial_inner_tol:.3e} status={trial_newton.status} "
                                f"converged={int(trial_newton.converged)} iters={trial_newton.iterations} "
                                f"residual={trial_newton.residual:.6e} "
                                f"linearSolveTime={trial_newton.solve_time:.6f}s"
                            ),
                        )
                        # Algorithm step 11: evaluate the corrected trial,
                        # apply residual/geometric/branch/collapse filters,
                        # and either keep the trial or roll back in place.
                        acceptance_start = time.perf_counter()
                        trial_band = evaluate_band_metrics(
                            comm=comm,
                            u=u,
                            tau_mask=tau_mask,
                            dx=dx,
                            c1_const=c1_const,
                            c2_const=c2_const,
                            eps_const=eps_const,
                            c1=trial_c1,
                            c2=trial_c2,
                            eps_phi=trial_eps,
                            kappa=args.kappa,
                            target_area=target_area,
                        )
                        branch_overlap = branch_overlap_ratio(
                            comm=comm,
                            activity_ref=activity_ref,
                            u_trial=u,
                            dx=dx,
                            c1_const=c1_const,
                            c2_const=c2_const,
                            eps_const=eps_const,
                            c1_trial=trial_c1,
                            c2_trial=trial_c2,
                            eps_trial=trial_eps,
                        )
                        leakage_limit = float(args.eta_out) * target_area
                        leakage_envelope = leakage_limit + float(args.leakage_filter_slack) * target_area
                        current_feasible = metrics.leakage <= leakage_limit
                        if args.threshold_objective_mode == "weighted":
                            leakage_weight, missing_weight = threshold_objective_weights(args)
                            actual_reduction = (
                                leakage_weight * (metrics.leakage - trial_band.leakage)
                                + missing_weight * (metrics.missing - trial_band.missing)
                            )
                            discrepancy_ok = (
                                actual_reduction
                                >= float(args.accept_sufficient_decrease) * target_area
                            )
                        elif current_feasible:
                            actual_reduction = metrics.missing - trial_band.missing
                            discrepancy_ok = (
                                trial_band.leakage <= leakage_envelope
                                and (
                                    trial_band.missing <= metrics.missing - float(args.accept_sufficient_decrease) * target_area
                                    or trial_band.leakage <= metrics.leakage - float(args.accept_sufficient_decrease) * target_area
                                )
                            )
                        else:
                            actual_reduction = metrics.leakage - trial_band.leakage
                            discrepancy_ok = trial_band.leakage <= metrics.leakage - float(args.accept_sufficient_decrease) * target_area
                        residual_ok = trial_newton.converged or trial_newton.residual <= max(10.0 * trial_inner_tol, float(args.tol_res))
                        branch_ok = args.disable_branch_check or branch_overlap >= float(args.eta_overlap)
                        activity_ok = trial_band.activity_area >= float(args.min_activity_fraction) * target_area
                        accepted = bool(residual_ok and discrepancy_ok and branch_ok and activity_ok)
                        rho_ratio = actual_reduction / max(step.predicted_reduction, 1.0e-30)
                        if accepted:
                            c1_phi = trial_c1
                            c2_phi = trial_c2
                            eps_phi = trial_eps
                            metrics = trial_band
                            final_metrics = trial_band
                            final_newton = trial_newton
                            residual_form_for_metrics = (
                                ufl.inner(ufl.grad(u), ufl.grad(test))
                                - optimization_source_density() * test
                            ) * dx
                            update_interpolated(
                                rho,
                                window_density_const_ufl(u, c1_const, c2_const, eps_const, params.rho_amp),
                            )
                            diagnostic_metrics = compute_metrics(
                                u=u,
                                rho=rho,
                                rho_design=rho_design,
                                rho_design_l2=rho_design_l2,
                                residual_form=residual_form_for_metrics,
                                bc=bc,
                                dx=dx,
                                c2_phi=c2_phi,
                                active_threshold=params.active_threshold,
                                plateau_threshold=params.plateau_threshold,
                                rho_amp=params.rho_amp,
                            )
                            final_compute_metrics = diagnostic_metrics
                            active_overlap_area = diagnostic_metrics["activeOverlapArea"]
                            active_recall = active_overlap_area / max(diagnostic_metrics["activeDesignArea"], 1.0e-30)
                            active_precision = active_overlap_area / max(diagnostic_metrics["activeArea"], 1.0e-30)
                            active_dice = 2.0 * active_overlap_area / max(
                                diagnostic_metrics["activeDesignArea"] + diagnostic_metrics["activeArea"],
                                1.0e-30,
                            )
                            phi_l2 = math.sqrt(max(assemble_scalar(comm, (u - phi_target) ** 2 * dx), 0.0))
                            phi_rel = phi_l2 / max(phi_target_l2, 1.0e-30)
                            mass_rel = diagnostic_metrics["massRhoMinusDesign"] / max(abs(rho_design_mass), 1.0e-30)
                            previous_discrepancy_rel = threshold_merit_rel(metrics, args)
                            newton = trial_newton
                            inner_tol = trial_inner_tol
                            if rho_ratio < 0.25:
                                trust_radius_abs *= float(args.trust_shrink)
                            elif rho_ratio > 0.75 and step.hit_boundary:
                                trust_radius_abs *= float(args.trust_grow)
                            trust_radius_abs = min(
                                max(trust_radius_abs, float(args.trust_radius_min) * c_scale),
                                float(args.trust_radius_max) * c_scale,
                            )
                            status = "ACCEPT"
                        else:
                            u.x.array[:] = old_u
                            u.x.scatter_forward()
                            c1_phi = old_c1
                            c2_phi = old_c2
                            eps_phi = old_eps
                            c1_const.value = PETSc.ScalarType(c1_phi)
                            c2_const.value = PETSc.ScalarType(c2_phi)
                            eps_const.value = PETSc.ScalarType(eps_phi)
                            update_interpolated(rho, window_density_const_ufl(u, c1_const, c2_const, eps_const, params.rho_amp))
                            trust_radius_abs = max(float(args.trust_radius_min) * c_scale, trust_radius_abs * float(args.trust_shrink))
                            status = "REJECT"
                        emit_severe_plot(
                            stage=f"REFIT_{status}",
                            outer_k=k,
                            token=f"refit_{status.lower()}_{k:03d}",
                            eps_phi_value=eps_phi,
                            residual=trial_newton.residual if accepted else newton.residual,
                            title_suffix=(
                                f"{status.lower()} k={k} "
                                f"c=({c1_phi:.3e},{c2_phi:.3e}) "
                                f"rho={rho_ratio:.2e}"
                            ),
                        )
                        acceptance_time = time.perf_counter() - acceptance_start
                        log_algorithm_step(
                            comm,
                            args,
                            iteration=k,
                            step=11,
                            label="accept_or_reject_trial",
                            elapsed=acceptance_time,
                            detail=(
                                f"accepted={int(accepted)} residualOk={int(residual_ok)} "
                                f"discrepancyOk={int(discrepancy_ok)} branchOk={int(branch_ok)} "
                                f"activityOk={int(activity_ok)} branchOverlap={branch_overlap:.6e} "
                                f"actualReduction={actual_reduction:.6e} rhoRatio={rho_ratio:.6e} "
                                f"trialLrel={trial_band.leakage_rel:.6e} trialMrel={trial_band.missing_rel:.6e}"
                            ),
                        )
                        # Algorithm step 12: update the trust radius and let
                        # the next outer iteration restart from the accepted
                        # state, or from the rolled-back state after rejection.
                        log_algorithm_step(
                            comm,
                            args,
                            iteration=k,
                            step=12,
                            label="update_trust_radius_and_repeat",
                            detail=f"nextTrustRadius={trust_radius_abs:.6e} nextStatus={status}",
                        )
            elif stop_ready:
                status = "CONVERGED"
                final_status = status
            else:
                status = "MAX_OPT_IT"

            if status == "REJECT" and gradient is not None:
                cached_reduced_gradient = gradient
            else:
                cached_reduced_gradient = None

            if accepted:
                accepted_steps += 1
                if args.run_inexact_newton_study and trial_predictor_state is not None:
                    inexact_replay_snapshots.append(InexactNewtonReplaySnapshot(
                        snapshot_id=f"accepted_outer_{k}",
                        predictor_state=trial_predictor_state,
                        reference_seed_state=u.x.array.copy(),
                        c1=float(c1_phi),
                        c2=float(c2_phi),
                        trust_radius=trial_predictor_trust_radius,
                        discrepancy_rel=trial_predictor_discrepancy,
                        outer_iteration=int(k),
                    ))
                trajectory.add(
                    stage="accepted_outer", phi=u, rho=rho,
                    c1=c1_phi, c2=c2_phi, eps_phi=eps_phi,
                    homotopy_lambda=optimization_homotopy_lambda,
                    outer_iteration=k, sequence=accepted_steps,
                )
            certified_ok, certified_leakage_rel, certified_area_rel = certified_subband_success(
                metrics=metrics,
                target_area=target_area,
                eta_out=args.eta_out,
                min_area_fraction=args.min_certified_area_fraction,
            )
            if certified_ok:
                certified_score = threshold_merit_rel(metrics, args)
                previous_best_score = best_certified_score
                meaningful_improvement = (
                    not math.isfinite(previous_best_score)
                    or certified_score <= previous_best_score * (1.0 - float(args.certified_stop_rtol))
                )
                if certified_score < best_certified_score:
                    best_certified_score = certified_score
                    best_certified_state = u.x.array.copy()
                    best_certified_c1 = c1_phi
                    best_certified_c2 = c2_phi
                    best_certified_iteration = k
                if accepted:
                    if meaningful_improvement:
                        certified_stale_steps = 0
                    else:
                        certified_stale_steps += 1
                if (
                    bool(args.certified_stop)
                    and accepted
                    and accepted_steps >= int(args.certified_stop_min_accepted)
                    and certified_stale_steps >= int(args.certified_stop_patience)
                ):
                    status = "STOPPED_CERTIFIED_STAGNATION"
                    final_status = status
                    root_print(
                        comm,
                        f"CERTIFIED_STOP k={k} accepted={accepted_steps} stale={certified_stale_steps} "
                        f"bestK={best_certified_iteration} bestScore={best_certified_score:.6e} "
                        f"certifiedLeakRel={certified_leakage_rel:.6e} certifiedAreaRel={certified_area_rel:.6e}",
                    )

            if final_status == "MAX_OPT_IT" and status in {"STEP_TOO_SMALL"}:
                final_status = status
            if args.verbosity >= 1 and args.terminal_every > 0 and (
                    k % int(args.terminal_every) == 0
                    or status in {"CONVERGED", "STOPPED_CERTIFIED_STAGNATION",
                                  "PDE_FAILURE", "REJECT", "STEP_TOO_SMALL"}
            ):
                root_print(
                    comm,
                    f"REDUCED_OPT k={k} status={status} c1={c1_phi:.6e} c2={c2_phi:.6e} "
                    f"eps={eps_phi:.6e} res={newton.residual:.6e} Lrel={metrics.leakage_rel:.6e} "
                    f"Mrel={metrics.missing_rel:.6e} trust={trust_radius_abs:.6e} "
                    f"grad={projected_grad_norm:.6e} "
                    f"activeJ={diagnostic_metrics['activeJaccard']:.6e} "
                    f"stepObj={'' if step is None else step.objective_name} "
                    f"dc={'' if step is None else f'({step.dc[0]:.3e},{step.dc[1]:.3e})'} "
                    f"accepted={int(accepted)}",
                )

            if writer is not None:
                writer.writerow({
                    "record": "OPT",
                    "runTag": run_tag,
                    "k": k,
                    "nt": nt,
                    "ndof": ndof,
                    "status": status,
                    "accepted": int(accepted),
                    "c1Phi": c1_phi,
                    "c2Phi": c2_phi,
                    "width": c2_phi - c1_phi,
                    "epsPhi": eps_phi,
                    "homotopyLambda": optimization_homotopy_lambda,
                    "trustRadius": trust_radius_abs,
                    "innerTol": inner_tol,
                    "newtonStatus": newton.status,
                    "newtonSteps": newton.iterations,
                    "residual": newton.residual,
                    "resEuclid": diagnostic_metrics["resEuclid"],
                    "leakage": metrics.leakage,
                    "missing": metrics.missing,
                    "leakageRel": metrics.leakage_rel,
                    "missingRel": metrics.missing_rel,
                    "activityArea": metrics.activity_area,
                    "certifiedArea": metrics.certified_area,
                    "certifiedLeakage": metrics.certified_leakage,
                    "certifiedMissing": metrics.certified_missing,
                    "gradL1": "" if gradient is None else gradient.grad_l[0],
                    "gradL2": "" if gradient is None else gradient.grad_l[1],
                    "gradM1": "" if gradient is None else gradient.grad_m[0],
                    "gradM2": "" if gradient is None else gradient.grad_m[1],
                    "directL1": "" if gradient is None else gradient.direct_l[0],
                    "directL2": "" if gradient is None else gradient.direct_l[1],
                    "directM1": "" if gradient is None else gradient.direct_m[0],
                    "directM2": "" if gradient is None else gradient.direct_m[1],
                    "projectedGradNorm": projected_grad_norm,
                    "sensIts1": "" if gradient is None else (0 if gradient_cached else gradient.solve_iterations[0]),
                    "sensIts2": "" if gradient is None else (0 if gradient_cached else gradient.solve_iterations[1]),
                    "sensRes1": "" if gradient is None else gradient.solve_residuals[0],
                    "sensRes2": "" if gradient is None else gradient.solve_residuals[1],
                    "sensSolveTime": "" if gradient is None else (0.0 if gradient_cached else gradient.solve_time),
                    "stepObjective": "" if step is None else step.objective_name,
                    "dc1": "" if step is None else step.dc[0],
                    "dc2": "" if step is None else step.dc[1],
                    "predictedReduction": "" if step is None else step.predicted_reduction,
                    "actualReduction": actual_reduction,
                    "rhoRatio": rho_ratio,
                    "branchOverlap": branch_overlap,
                    "activeJaccard": diagnostic_metrics["activeJaccard"],
                    "activeRecall": active_recall,
                    "activePrecision": active_precision,
                    "activeDice": active_dice,
                    "plateauJaccard": diagnostic_metrics["plateauJaccard"],
                    "rhoRel": diagnostic_metrics["relRhoDesign"],
                    "massRel": mass_rel,
                    "phiRel": phi_rel,
                    "solveTime": newton.solve_time,
                    "stepTime": time.perf_counter() - step_start,
                })
                opt_handle.flush()

            if args.plot_optimization and frame_every > 0 and k % int(frame_every) == 0:
                phi_diff.x.array[:] = u.x.array - phi_target.x.array
                phi_diff.x.scatter_forward()
                plotter.emit(
                    [T, tau_band, phi_target, u, rho, phi_diff],
                    ["Torsion T", "tau band", "phiT", "phi", "rho(phi)", "phi-phiT"],
                    stage="OPT",
                    ieps=0,
                    k=k,
                    eps_phi=eps_phi,
                    residual=newton.residual,
                    metrics={
                        "massRho": diagnostic_metrics["massRho"],
                        "maxRho": diagnostic_metrics["maxRho"],
                        "activeArea": diagnostic_metrics["activeArea"],
                        "plateauArea": diagnostic_metrics["plateauArea"],
                        "relRhoDesign": diagnostic_metrics["relRhoDesign"],
                    },
                    token=f"opt_{k}",
                    save=bool(args.save_frames and frame_every > 0 and k % int(frame_every) == 0),
                    show=True,
                    nt=nt,
                    ndof=ndof,
                    contour_field_index=0,
                    contour_levels=(c1_t, c2_t),
                )

            if status in {"CONVERGED", "STOPPED_CERTIFIED_STAGNATION",
                          "PDE_FAILURE", "STEP_TOO_SMALL"}:
                break
    finally:
        if opt_handle is not None:
            opt_handle.close()

    if final_metrics is None or final_newton is None or final_compute_metrics is None:
        raise RuntimeError("optimization did not produce a final iterate")

    if (
        bool(args.retain_best_certified)
        and best_certified_state is not None
        and not (
            final_metrics.leakage_rel <= float(args.eta_out)
            and final_metrics.missing_rel <= float(args.tol_area)
        )
    ):
        c1_phi = best_certified_c1
        c2_phi = best_certified_c2
        eps_phi = epsilon_from_thresholds(args, c1_phi, c2_phi)
        c1_const.value = PETSc.ScalarType(c1_phi)
        c2_const.value = PETSc.ScalarType(c2_phi)
        eps_const.value = PETSc.ScalarType(eps_phi)
        u.x.array[:] = best_certified_state
        u.x.scatter_forward()
        update_interpolated(
            rho,
            window_density_const_ufl(u, c1_const, c2_const, eps_const, params.rho_amp),
        )
        trajectory.add(
            stage="restored_best", phi=u, rho=rho,
            c1=c1_phi, c2=c2_phi, eps_phi=eps_phi,
            homotopy_lambda=optimization_homotopy_lambda,
            outer_iteration=best_certified_iteration, force=True,
        )
        root_print(
            comm,
            f"RESTORE_BEST_CERTIFIED k={best_certified_iteration} score={best_certified_score:.6e} "
            f"c1={c1_phi:.6e} c2={c2_phi:.6e}",
        )

    # Final feasibility policy: no matter how loose the adaptive outer Newton
    # solves were, the reported final state must satisfy the requested final
    # residual tolerance at the final thresholds.
    eps_phi = epsilon_from_thresholds(args, c1_phi, c2_phi)
    final_newton_args = argparse.Namespace(**vars(final_solver_args))
    final_newton_args.max_newton_it = int(args.final_newton_max_it)
    # The generic nonlinear step cutoff is deliberately loose enough to stop
    # unproductive intermediate solves.  Reusing it for final certification can
    # reject a useful correction before it is applied (for example, an O(1e-11)
    # H1 correction for an O(1e-12) residual target).  Scale only the final
    # cutoff with its requested residual, while preserving a tighter user value.
    final_newton_args.tol_step = min(float(args.tol_step), 0.1 * final_tol_res)
    final_exact_start = time.perf_counter()
    final_newton = solve_equilibrium(
        u=u,
        du=du,
        rho=rho,
        trial=trial,
        test=test,
        dx=dx,
        bc=bc,
        stiffness_form=stiffness_form,
        c1_const=c1_const,
        c2_const=c2_const,
        eps_const=eps_const,
        c1=c1_phi,
        c2=c2_phi,
        eps_phi=eps_phi,
        rho_amp=params.rho_amp,
        tol_res=final_tol_res,
        args=final_newton_args,
        prefix="final_exact",
        homotopy_lambda=optimization_homotopy_lambda,
        homotopy_target_density=target_density,
        stiffness_solver=stiffness_solver,
        plot_callback=make_newton_plot_callback(outer_k=-1, stage="NEWTON_FINAL"),
        phase="final_projection",
        outer_iteration=-1,
    )
    final_exact_time = time.perf_counter() - final_exact_start
    add_phase("final_projection", final_exact_time)
    log_algorithm_step(
        comm,
        args,
        iteration=-1,
        step=1,
        label="final_exact_newton_projection",
        elapsed=final_exact_time,
        detail=(
            f"tol={final_tol_res:.3e} maxIt={args.final_newton_max_it} status={final_newton.status} "
            f"converged={int(final_newton.converged)} iters={final_newton.iterations} "
            f"residual={final_newton.residual:.6e} linearSolveTime={final_newton.solve_time:.6f}s"
        ),
    )
    final_metrics = evaluate_band_metrics(
        comm=comm,
        u=u,
        tau_mask=tau_mask,
        dx=dx,
        c1_const=c1_const,
        c2_const=c2_const,
        eps_const=eps_const,
        c1=c1_phi,
        c2=c2_phi,
        eps_phi=eps_phi,
        kappa=args.kappa,
        target_area=target_area,
    )
    update_interpolated(rho, window_density_const_ufl(u, c1_const, c2_const, eps_const, params.rho_amp))
    final_residual_form = (
        ufl.inner(ufl.grad(u), ufl.grad(test))
        - optimization_source_density() * test
    ) * dx
    final_compute_metrics = compute_metrics(
        u=u,
        rho=rho,
        rho_design=rho_design,
        rho_design_l2=rho_design_l2,
        residual_form=final_residual_form,
        bc=bc,
        dx=dx,
        c2_phi=c2_phi,
        active_threshold=params.active_threshold,
        plateau_threshold=params.plateau_threshold,
        rho_amp=params.rho_amp,
    )
    final_geometry_ok = (
        final_metrics.leakage_rel <= float(args.eta_out)
        and final_metrics.missing_rel <= float(args.tol_area)
    )
    certified_subband_ok, final_certified_leakage_rel, final_certified_area_rel = certified_subband_success(
        metrics=final_metrics,
        target_area=target_area,
        eta_out=args.eta_out,
        min_area_fraction=args.min_certified_area_fraction,
    )
    if not final_newton.converged:
        final_status = "NEWTON_NOT_CONVERGED"
    elif final_geometry_ok:
        final_status = "CONVERGED"
    elif certified_subband_ok:
        final_status = "CONVERGED_CERTIFIED_SUBBAND"
    elif final_status == "CONVERGED":
        final_status = "FINAL_GEOMETRY_NOT_CONVERGED"

    inexact_newton_rows = 0
    if args.run_inexact_newton_study:
        replay_started = time.perf_counter()
        final_state_before_replay = u.x.array.copy()
        final_density_before_replay = rho.x.array.copy()
        final_c1_before_replay = float(c1_phi)
        final_c2_before_replay = float(c2_phi)
        final_eps_before_replay = float(eps_phi)
        if comm.rank == 0:
            inexact_newton_csv.parent.mkdir(parents=True, exist_ok=True)
        inexact_handle = (
            inexact_newton_csv.open("w", newline="", encoding="utf-8")
            if comm.rank == 0
            else None
        )
        inexact_writer = (
            csv.DictWriter(inexact_handle, fieldnames=INEXACT_NEWTON_CSV_FIELDS)
            if inexact_handle is not None
            else None
        )
        if inexact_writer is not None:
            inexact_writer.writeheader()
            inexact_handle.flush()

        snapshot_count = min(
            int(args.inexact_newton_max_snapshots), len(inexact_replay_snapshots)
        )
        if snapshot_count == len(inexact_replay_snapshots):
            replay_snapshots = list(inexact_replay_snapshots)
        else:
            replay_indices = np.unique(np.rint(np.linspace(
                0,
                len(inexact_replay_snapshots) - 1,
                snapshot_count,
            )).astype(np.int64))
            replay_snapshots = [
                inexact_replay_snapshots[int(index)] for index in replay_indices
            ]

        tolerance_tokens = [
            token.strip().lower()
            for token in str(args.inexact_newton_tolerances).split(",")
            if token.strip()
        ]
        reference_u = fem.Function(V, name="inexactReferenceState")
        reference_s1 = fem.Function(V, name="inexactReferenceSensitivityC1")
        reference_s2 = fem.Function(V, name="inexactReferenceSensitivityC2")

        def replay_max_time(local_elapsed: float) -> float:
            """Return the collective wall time of one distributed replay phase."""
            return float(comm.allreduce(float(local_elapsed), op=MPI.MAX))

        def set_replay_state(state: np.ndarray, replay_c1: float, replay_c2: float) -> float:
            """Restore one replay state and threshold pair in all live forms."""
            replay_eps = epsilon_from_thresholds(args, replay_c1, replay_c2)
            c1_const.value = PETSc.ScalarType(replay_c1)
            c2_const.value = PETSc.ScalarType(replay_c2)
            eps_const.value = PETSc.ScalarType(replay_eps)
            u.x.array[:] = state
            u.x.scatter_forward()
            update_interpolated(
                rho,
                window_density_const_ufl(
                    u, c1_const, c2_const, eps_const, params.rho_amp
                ),
            )
            return replay_eps

        def relative_field_errors(
                approximate: fem.Function,
                reference: fem.Function,
        ) -> tuple[float, float]:
            """Return relative L2 and H1-seminorm errors on the replay mesh."""
            difference_l2 = assemble_scalar(comm, (approximate - reference) ** 2 * dx)
            reference_l2 = assemble_scalar(comm, reference ** 2 * dx)
            difference_h1 = assemble_scalar(
                comm,
                ufl.inner(
                    ufl.grad(approximate - reference),
                    ufl.grad(approximate - reference),
                ) * dx,
            )
            reference_h1 = assemble_scalar(
                comm, ufl.inner(ufl.grad(reference), ufl.grad(reference)) * dx
            )
            return (
                math.sqrt(max(difference_l2, 0.0))
                / max(math.sqrt(max(reference_l2, 0.0)), 1.0e-30),
                math.sqrt(max(difference_h1, 0.0))
                / max(math.sqrt(max(reference_h1, 0.0)), 1.0e-30),
            )

        def sensitivity_defect(sensitivity: fem.Function, component: int) -> float:
            """Return ``||J s_i + F_c_i||_2`` at the current replay state."""
            ws = window_s_derivative_activity_ufl(
                u, c1_const, c2_const, eps_const
            )
            dc_activity = window_c_derivatives_activity_ufl(
                u,
                c1_const,
                c2_const,
                eps_const,
                eps_mode=args.eps_mode,
                eps_ratio=args.eps_ratio,
            )[component]
            defect_form = (
                ufl.inner(ufl.grad(sensitivity), ufl.grad(test))
                - float(params.rho_amp) * ws * sensitivity * test
                - float(params.rho_amp) * dc_activity * test
            ) * dx
            return residual_norm(
                fem.form(defect_form),
                bc,
                comm=comm,
                mode="euclidean",
                metric_form=stiffness_form,
                solver=stiffness_args.linear_solver,
                ksp_type=stiffness_args.ksp_type,
                rtol=stiffness_args.linear_rtol,
                atol=stiffness_args.linear_atol,
                max_it=stiffness_args.linear_max_it,
                prefix="inexact_sensitivity_defect_",
                stiffness_solver=stiffness_solver,
            )

        reference_newton_args = argparse.Namespace(**vars(final_solver_args))
        reference_newton_args.max_newton_it = int(args.final_newton_max_it)
        reference_newton_args.tol_step = min(
            float(args.tol_step), 0.1 * float(args.inexact_newton_reference_tol)
        )
        one_correction_args = argparse.Namespace(**vars(final_solver_args))
        one_correction_args.max_newton_it = 1
        one_correction_args.newton_hard_cap_reason = "one_correction_replay"
        one_correction_args.tol_step = 0.0

        try:
            for snapshot_index, snapshot in enumerate(replay_snapshots):
                replay_eps = set_replay_state(
                    snapshot.reference_seed_state, snapshot.c1, snapshot.c2
                )
                reference_state_started = time.perf_counter()
                reference_newton = solve_equilibrium(
                    u=u,
                    du=du,
                    rho=rho,
                    trial=trial,
                    test=test,
                    dx=dx,
                    bc=bc,
                    stiffness_form=stiffness_form,
                    c1_const=c1_const,
                    c2_const=c2_const,
                    eps_const=eps_const,
                    c1=snapshot.c1,
                    c2=snapshot.c2,
                    eps_phi=replay_eps,
                    rho_amp=params.rho_amp,
                    tol_res=float(args.inexact_newton_reference_tol),
                    args=reference_newton_args,
                    prefix=f"inexact_reference_{snapshot_index}",
                    stiffness_solver=stiffness_solver,
                    phase="inexact_replay_reference",
                    outer_iteration=snapshot.outer_iteration,
                )
                reference_state_time = replay_max_time(
                    time.perf_counter() - reference_state_started
                )
                if not reference_newton.converged:
                    root_print(
                        comm,
                        f"INEXACT_REPLAY snapshot={snapshot.snapshot_id} status=SKIP "
                        f"reason=reference_{reference_newton.status} "
                        f"residual={reference_newton.residual:.6e}",
                    )
                    continue
                reference_u.x.array[:] = u.x.array
                reference_u.x.scatter_forward()
                reference_gradient = compute_reduced_gradient(
                    u=u,
                    s1=s1,
                    s2=s2,
                    trial=trial,
                    test=test,
                    dx=dx,
                    bc=bc,
                    tau_mask=tau_mask,
                    c1_const=c1_const,
                    c2_const=c2_const,
                    eps_const=eps_const,
                    rho_amp=params.rho_amp,
                    eps_mode=args.eps_mode,
                    eps_ratio=args.eps_ratio,
                    args=reference_newton_args,
                    iteration=-1000 - snapshot_index,
                )
                reference_sensitivity_time = replay_max_time(
                    reference_gradient.total_time
                )
                reference_s1.x.array[:] = s1.x.array
                reference_s1.x.scatter_forward()
                reference_s2.x.array[:] = s2.x.array
                reference_s2.x.scatter_forward()
                reference_metrics = evaluate_band_metrics(
                    comm=comm,
                    u=u,
                    tau_mask=tau_mask,
                    dx=dx,
                    c1_const=c1_const,
                    c2_const=c2_const,
                    eps_const=eps_const,
                    c1=snapshot.c1,
                    c2=snapshot.c2,
                    eps_phi=replay_eps,
                    kappa=args.kappa,
                    target_area=target_area,
                )
                reference_step = choose_parameter_step(
                    c1=snapshot.c1,
                    c2=snapshot.c2,
                    c_min=c_min,
                    c_max=c_upper,
                    min_width=min_width,
                    metrics=reference_metrics,
                    gradient=reference_gradient,
                    trust_radius_abs=snapshot.trust_radius,
                    args=args,
                )
                objective_attribute = (
                    "grad_m"
                    if reference_metrics.leakage_rel <= float(args.eta_out)
                    else "grad_l"
                )
                reference_reduced_vector = np.asarray(
                    getattr(reference_gradient, objective_attribute), dtype=np.float64
                )
                reference_model_accepts = model_reduction_is_sufficient(
                    reference_step.predicted_reduction,
                    target_area=target_area,
                    sufficient_decrease_fraction=args.accept_sufficient_decrease,
                )
                reference_defects = (
                    sensitivity_defect(reference_s1, 0),
                    sensitivity_defect(reference_s2, 1),
                )

                for tolerance_index, tolerance_token in enumerate(tolerance_tokens):
                    requested_tolerance = (
                        inner_tolerance(args, snapshot.discrepancy_rel)
                        if tolerance_token == "adaptive"
                        else float(tolerance_token)
                    )
                    raw_started = time.perf_counter()
                    set_replay_state(snapshot.predictor_state, snapshot.c1, snapshot.c2)
                    raw_newton = solve_equilibrium(
                        u=u,
                        du=du,
                        rho=rho,
                        trial=trial,
                        test=test,
                        dx=dx,
                        bc=bc,
                        stiffness_form=stiffness_form,
                        c1_const=c1_const,
                        c2_const=c2_const,
                        eps_const=eps_const,
                        c1=snapshot.c1,
                        c2=snapshot.c2,
                        eps_phi=replay_eps,
                        rho_amp=params.rho_amp,
                        tol_res=requested_tolerance,
                        args=nonlinear_args,
                        prefix=f"inexact_raw_{snapshot_index}_{tolerance_index}",
                        stiffness_solver=stiffness_solver,
                        phase="inexact_replay_raw",
                        outer_iteration=snapshot.outer_iteration,
                    )
                    raw_state_time = replay_max_time(
                        time.perf_counter() - raw_started
                    )
                    raw_state = u.x.array.copy()
                    raw_gradient = compute_reduced_gradient(
                        u=u,
                        s1=s1,
                        s2=s2,
                        trial=trial,
                        test=test,
                        dx=dx,
                        bc=bc,
                        tau_mask=tau_mask,
                        c1_const=c1_const,
                        c2_const=c2_const,
                        eps_const=eps_const,
                        rho_amp=params.rho_amp,
                        eps_mode=args.eps_mode,
                        eps_ratio=args.eps_ratio,
                        args=sensitivity_args,
                        iteration=-2000 - 100 * snapshot_index - tolerance_index,
                    )
                    raw_sensitivity_time = replay_max_time(raw_gradient.total_time)
                    raw_state_errors = relative_field_errors(u, reference_u)
                    raw_s1_errors = relative_field_errors(s1, reference_s1)
                    raw_s2_errors = relative_field_errors(s2, reference_s2)
                    raw_defects = (
                        sensitivity_defect(s1, 0), sensitivity_defect(s2, 1)
                    )
                    raw_metrics = evaluate_band_metrics(
                        comm=comm,
                        u=u,
                        tau_mask=tau_mask,
                        dx=dx,
                        c1_const=c1_const,
                        c2_const=c2_const,
                        eps_const=eps_const,
                        c1=snapshot.c1,
                        c2=snapshot.c2,
                        eps_phi=replay_eps,
                        kappa=args.kappa,
                        target_area=target_area,
                    )
                    raw_step = choose_parameter_step(
                        c1=snapshot.c1,
                        c2=snapshot.c2,
                        c_min=c_min,
                        c_max=c_upper,
                        min_width=min_width,
                        metrics=raw_metrics,
                        gradient=raw_gradient,
                        trust_radius_abs=snapshot.trust_radius,
                        args=args,
                    )
                    raw_model_accepts = model_reduction_is_sufficient(
                        raw_step.predicted_reduction,
                        target_area=target_area,
                        sufficient_decrease_fraction=args.accept_sufficient_decrease,
                    )
                    raw_diagnostics = compare_inexact_newton_snapshot(
                        state_l2_relative_error=raw_state_errors[0],
                        state_h1_relative_error=raw_state_errors[1],
                        sensitivity1_l2_relative_error=raw_s1_errors[0],
                        sensitivity1_h1_relative_error=raw_s1_errors[1],
                        sensitivity2_l2_relative_error=raw_s2_errors[0],
                        sensitivity2_h1_relative_error=raw_s2_errors[1],
                        sensitivity1_defect=raw_defects[0],
                        sensitivity2_defect=raw_defects[1],
                        approximate_reduced_gradient=np.asarray(
                            getattr(raw_gradient, objective_attribute), dtype=np.float64
                        ),
                        reference_reduced_gradient=reference_reduced_vector,
                        approximate_threshold_step=raw_step.dc,
                        reference_threshold_step=reference_step.dc,
                        approximate_predicted_reduction=raw_step.predicted_reduction,
                        reference_predicted_reduction=reference_step.predicted_reduction,
                        approximate_accepted=raw_model_accepts,
                        reference_accepted=reference_model_accepts,
                    )
                    if inexact_writer is not None:
                        write_inexact_newton_snapshot_row(
                            inexact_writer,
                            raw_diagnostics,
                            snapshot_id=snapshot.snapshot_id,
                            policy="raw",
                            requested_tolerance=requested_tolerance,
                            outer_iteration=snapshot.outer_iteration,
                            c1=snapshot.c1,
                            c2=snapshot.c2,
                            state_residual=raw_newton.residual,
                            reference_state_residual=reference_newton.residual,
                            state_solve_time=raw_state_time,
                            sensitivity_solve_time=raw_sensitivity_time,
                            elapsed=raw_state_time + raw_sensitivity_time,
                        )
                        inexact_handle.flush()
                    inexact_newton_rows += 1

                    set_replay_state(raw_state, snapshot.c1, snapshot.c2)
                    correction_started = time.perf_counter()
                    one_newton = solve_equilibrium(
                        u=u,
                        du=du,
                        rho=rho,
                        trial=trial,
                        test=test,
                        dx=dx,
                        bc=bc,
                        stiffness_form=stiffness_form,
                        c1_const=c1_const,
                        c2_const=c2_const,
                        eps_const=eps_const,
                        c1=snapshot.c1,
                        c2=snapshot.c2,
                        eps_phi=replay_eps,
                        rho_amp=params.rho_amp,
                        tol_res=float(args.inexact_newton_reference_tol),
                        args=one_correction_args,
                        prefix=f"inexact_one_correction_{snapshot_index}_{tolerance_index}",
                        stiffness_solver=stiffness_solver,
                        phase="inexact_replay_one_correction",
                        outer_iteration=snapshot.outer_iteration,
                    )
                    correction_time = replay_max_time(
                        time.perf_counter() - correction_started
                    )
                    corrected_gradient = compute_reduced_gradient(
                        u=u,
                        s1=s1,
                        s2=s2,
                        trial=trial,
                        test=test,
                        dx=dx,
                        bc=bc,
                        tau_mask=tau_mask,
                        c1_const=c1_const,
                        c2_const=c2_const,
                        eps_const=eps_const,
                        rho_amp=params.rho_amp,
                        eps_mode=args.eps_mode,
                        eps_ratio=args.eps_ratio,
                        args=sensitivity_args,
                        iteration=-3000 - 100 * snapshot_index - tolerance_index,
                    )
                    corrected_sensitivity_time = replay_max_time(
                        corrected_gradient.total_time
                    )
                    corrected_state_errors = relative_field_errors(u, reference_u)
                    corrected_s1_errors = relative_field_errors(s1, reference_s1)
                    corrected_s2_errors = relative_field_errors(s2, reference_s2)
                    corrected_defects = (
                        sensitivity_defect(s1, 0), sensitivity_defect(s2, 1)
                    )
                    corrected_metrics = evaluate_band_metrics(
                        comm=comm,
                        u=u,
                        tau_mask=tau_mask,
                        dx=dx,
                        c1_const=c1_const,
                        c2_const=c2_const,
                        eps_const=eps_const,
                        c1=snapshot.c1,
                        c2=snapshot.c2,
                        eps_phi=replay_eps,
                        kappa=args.kappa,
                        target_area=target_area,
                    )
                    corrected_step = choose_parameter_step(
                        c1=snapshot.c1,
                        c2=snapshot.c2,
                        c_min=c_min,
                        c_max=c_upper,
                        min_width=min_width,
                        metrics=corrected_metrics,
                        gradient=corrected_gradient,
                        trust_radius_abs=snapshot.trust_radius,
                        args=args,
                    )
                    corrected_model_accepts = model_reduction_is_sufficient(
                        corrected_step.predicted_reduction,
                        target_area=target_area,
                        sufficient_decrease_fraction=args.accept_sufficient_decrease,
                    )
                    corrected_diagnostics = compare_inexact_newton_snapshot(
                        state_l2_relative_error=corrected_state_errors[0],
                        state_h1_relative_error=corrected_state_errors[1],
                        sensitivity1_l2_relative_error=corrected_s1_errors[0],
                        sensitivity1_h1_relative_error=corrected_s1_errors[1],
                        sensitivity2_l2_relative_error=corrected_s2_errors[0],
                        sensitivity2_h1_relative_error=corrected_s2_errors[1],
                        sensitivity1_defect=corrected_defects[0],
                        sensitivity2_defect=corrected_defects[1],
                        approximate_reduced_gradient=np.asarray(
                            getattr(corrected_gradient, objective_attribute),
                            dtype=np.float64,
                        ),
                        reference_reduced_gradient=reference_reduced_vector,
                        approximate_threshold_step=corrected_step.dc,
                        reference_threshold_step=reference_step.dc,
                        approximate_predicted_reduction=corrected_step.predicted_reduction,
                        reference_predicted_reduction=reference_step.predicted_reduction,
                        approximate_accepted=corrected_model_accepts,
                        reference_accepted=reference_model_accepts,
                    )
                    if inexact_writer is not None:
                        write_inexact_newton_snapshot_row(
                            inexact_writer,
                            corrected_diagnostics,
                            snapshot_id=snapshot.snapshot_id,
                            policy="one-correction",
                            requested_tolerance=requested_tolerance,
                            outer_iteration=snapshot.outer_iteration,
                            c1=snapshot.c1,
                            c2=snapshot.c2,
                            state_residual=one_newton.residual,
                            reference_state_residual=reference_newton.residual,
                            state_solve_time=raw_state_time + correction_time,
                            sensitivity_solve_time=corrected_sensitivity_time,
                            elapsed=(
                                raw_state_time
                                + correction_time
                                + corrected_sensitivity_time
                            ),
                        )
                        inexact_handle.flush()
                    inexact_newton_rows += 1

                    set_replay_state(
                        reference_u.x.array, snapshot.c1, snapshot.c2
                    )
                    s1.x.array[:] = reference_s1.x.array
                    s1.x.scatter_forward()
                    s2.x.array[:] = reference_s2.x.array
                    s2.x.scatter_forward()
                    reference_diagnostics = compare_inexact_newton_snapshot(
                        state_l2_relative_error=0.0,
                        state_h1_relative_error=0.0,
                        sensitivity1_l2_relative_error=0.0,
                        sensitivity1_h1_relative_error=0.0,
                        sensitivity2_l2_relative_error=0.0,
                        sensitivity2_h1_relative_error=0.0,
                        sensitivity1_defect=reference_defects[0],
                        sensitivity2_defect=reference_defects[1],
                        approximate_reduced_gradient=reference_reduced_vector,
                        reference_reduced_gradient=reference_reduced_vector,
                        approximate_threshold_step=reference_step.dc,
                        reference_threshold_step=reference_step.dc,
                        approximate_predicted_reduction=reference_step.predicted_reduction,
                        reference_predicted_reduction=reference_step.predicted_reduction,
                        approximate_accepted=reference_model_accepts,
                        reference_accepted=reference_model_accepts,
                    )
                    if inexact_writer is not None:
                        write_inexact_newton_snapshot_row(
                            inexact_writer,
                            reference_diagnostics,
                            snapshot_id=snapshot.snapshot_id,
                            policy="reassembled",
                            requested_tolerance=requested_tolerance,
                            outer_iteration=snapshot.outer_iteration,
                            c1=snapshot.c1,
                            c2=snapshot.c2,
                            state_residual=reference_newton.residual,
                            reference_state_residual=reference_newton.residual,
                            state_solve_time=reference_state_time,
                            sensitivity_solve_time=reference_sensitivity_time,
                            elapsed=reference_state_time + reference_sensitivity_time,
                        )
                        inexact_handle.flush()
                    inexact_newton_rows += 1
        finally:
            if inexact_handle is not None:
                inexact_handle.close()
            c1_phi = final_c1_before_replay
            c2_phi = final_c2_before_replay
            eps_phi = final_eps_before_replay
            c1_const.value = PETSc.ScalarType(c1_phi)
            c2_const.value = PETSc.ScalarType(c2_phi)
            eps_const.value = PETSc.ScalarType(eps_phi)
            u.x.array[:] = final_state_before_replay
            u.x.scatter_forward()
            rho.x.array[:] = final_density_before_replay
            rho.x.scatter_forward()
        replay_elapsed = replay_max_time(time.perf_counter() - replay_started)
        add_phase("inexact_newton", replay_elapsed)
        root_print(
            comm,
            f"INEXACT_REPLAY status=DONE snapshots={len(replay_snapshots)} "
            f"rows={inexact_newton_rows} output={inexact_newton_csv} "
            f"acceptance=model_feasibility_proxy time={replay_elapsed:.3f}s",
        )

    checkpoint_started = time.perf_counter()
    # The nonlinear residual integrates the smooth source at quadrature points.
    # Nodal interpolation of that source does not generally produce the same
    # load vector, especially when epsilon is narrow relative to the mesh.  A
    # guiding-center restart would then change phi at t=0 even though both
    # programs use the same mesh and polynomial order.  Store the L2 projection
    # whose mass action is exactly the nonlinear source load instead.
    rho_checkpoint = fem.Function(V, name="rho_equilibrium_load_equivalent")
    mass_solver = ReusableLinearSolver(
        trial * test * dx,
        V,
        [],
        prefix="checkpoint_density_projection_",
        solver=final_solver_args.linear_solver,
        ksp_type=final_solver_args.ksp_type,
        rtol=min(float(final_solver_args.linear_rtol), 1.0e-13),
        atol=min(float(final_solver_args.linear_atol), 1.0e-14),
        max_it=final_solver_args.linear_max_it,
        verbosity=args.verbosity,
    )
    try:
        projection_iterations, projection_residual, projection_time = mass_solver.solve_form(
            fem.form(
                window_density_const_ufl(
                    u, c1_const, c2_const, eps_const, params.rho_amp
                ) * test * dx
            ),
            rho_checkpoint,
        )
    finally:
        mass_solver.close()
    root_print(
        comm,
        "CHECKPOINT_DENSITY_PROJECTION "
        f"iterations={projection_iterations} residual={projection_residual:.3e} "
        f"time={projection_time:.6f}s representation=l2_load_equivalent",
    )
    write_equilibrium_checkpoint(
        equilibrium_path,
        phi=u,
        rho=rho_checkpoint,
        metadata={
            "format": "hybridge_equilibrium_v1",
            "run_tag": run_tag,
            "mesh_path": str(Path(mesh_path).resolve()),
            "order": int(args.order),
            "quadrature_degree": int(qdeg),
            "alpha_t1": float(params.alpha_t1),
            "alpha_t2": float(params.alpha_t2),
            "rho_amp": float(params.rho_amp),
            "c1_phi": float(c1_phi),
            "c2_phi": float(c2_phi),
            "eps_phi": float(eps_phi),
            "final_residual": float(final_newton.residual),
            "homotopy_lambda": optimization_homotopy_lambda,
            "final_status": final_status,
            "initialization_mode": str(args.init_mode),
            "initialization_search": str(args.init_search),
            "initialization_fallback": str(args.init_fallback),
            "initial_equilibrium": (
                str(args.initial_equilibrium) if args.initial_equilibrium is not None else None
            ),
            "rho_representation": "l2_load_equivalent_projection",
            "rho_projection_iterations": int(projection_iterations),
            "rho_projection_residual": float(projection_residual),
            "num_cells": int(nt),
            "num_dofs": int(ndof),
        },
    )
    add_phase("checkpointing", time.perf_counter() - checkpoint_started)

    trajectory.add(
        stage="final", phi=u, rho=rho,
        c1=c1_phi, c2=c2_phi, eps_phi=eps_phi,
        homotopy_lambda=optimization_homotopy_lambda,
        outer_iteration=-1, force=True,
    )
    trajectory_started = time.perf_counter()
    trajectory.write({
        "run_tag": run_tag,
        "order": int(args.order),
        "quadrature_degree": int(qdeg),
        "alpha_t1": float(params.alpha_t1),
        "alpha_t2": float(params.alpha_t2),
        "c1_t": float(c1_t),
        "c2_t": float(c2_t),
        "rho_amp": float(params.rho_amp),
        "final_status": final_status,
        "final_residual": float(final_newton.residual),
        "homotopy_lambda": optimization_homotopy_lambda,
        "complete": True,
        "terminal_stage": "final",
        "terminal_status": final_status,
    })
    if args.save_trajectory:
        add_phase("checkpointing", time.perf_counter() - trajectory_started)

    phi_diff.x.array[:] = u.x.array - phi_target.x.array
    phi_diff.x.scatter_forward()
    if args.plot_final:
        # Intermediate optimization plots may use the live nonblocking updater,
        # but the final state is the inspection point for the run.  Force this
        # one emission through the blocking plot path and then restore the user
        # selected mode in case future cleanup/extension emits more frames.
        original_plot_mode = args.plot_mode
        args.plot_mode = "blocking"
        try:
            plotter.emit(
                [T, tau_band, rho_design, phi_target, u, rho, phi_diff],
                ["Torsion T", "tau band", "rhoDesign", "phiT", "phi", "rho(phi)", "phi-phiT"],
                stage="FINAL",
                ieps=0,
                k=-1,
                eps_phi=eps_phi,
                residual=final_newton.residual,
                metrics={
                    "massRho": final_compute_metrics["massRho"],
                    "maxRho": final_compute_metrics["maxRho"],
                    "activeArea": final_compute_metrics["activeArea"],
                    "plateauArea": final_compute_metrics["plateauArea"],
                    "relRhoDesign": final_compute_metrics["relRhoDesign"],
                },
                token="final",
                save=bool(args.frame_final),
                show=True,
                nt=nt,
                ndof=ndof,
                contour_field_index=0,
                contour_levels=(c1_t, c2_t),
            )
        finally:
            args.plot_mode = original_plot_mode
    if frame_handle is not None:
        frame_handle.close()

    elapsed = time.perf_counter() - total_start
    phase_times["total"] = elapsed
    phase_calls["total"] = 1
    write_phases()
    if newton_handle is not None:
        newton_handle.close()
    root_print(
        comm,
        f"FINAL status={final_status} c1={c1_phi:.6e} c2={c2_phi:.6e} epsPhi={eps_phi:.6e} "
        f"res={final_newton.residual:.6e} Lrel={final_metrics.leakage_rel:.6e} "
        f"Mrel={final_metrics.missing_rel:.6e} certifiedArea={final_metrics.certified_area:.6e} "
        f"certifiedAreaRel={final_certified_area_rel:.6e} "
        f"certifiedLeakRel={final_certified_leakage_rel:.6e} "
        f"activeJ={final_compute_metrics['activeJaccard']:.6e} rhoRel={final_compute_metrics['relRhoDesign']:.6e}",
    )
    root_print(comm, f"TIME_TOTAL {elapsed:.3f}")
    root_print(comm, "========== END STRATEGY A DOLFINX WINDOW REDUCED OPTIMIZATION ==========")

    if comm.rank == 0 and (args.check_newton_spd or args.diagnose_capped_trial_coercivity):
        spd_fields = [
            "prefix",
            "stage",
            "newtonIteration",
            "lambda",
            "residual",
            "c1",
            "c2",
            "eps",
            "symmetric",
            "status",
            "thetaMax",
            "muMin",
            "shiftToSpd",
            "eigenError",
            "eigenIterations",
            "eigenConverged",
            "negativeModes",
            "zeroModes",
            "positiveModes",
            "inertiaAgrees",
            "elapsed",
        ]
        with newton_spd_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=spd_fields)
            writer.writeheader()
            writer.writerows(args.newton_spd_records)

    if comm.rank == 0:
        with summary_path.open("w", encoding="utf-8") as handle:
            handle.write(f"runTag {run_tag}\n")
            handle.write(f"runDir {run_dir}\n")
            handle.write(f"initializationCsv {initialization_csv}\n")
            handle.write(f"newtonCsv {newton_csv}\n")
            handle.write(f"phasesCsv {phases_csv}\n")
            handle.write(
                f"inexactNewtonCsv {inexact_newton_csv if args.run_inexact_newton_study else None}\n"
            )
            handle.write(f"trajectoryArchive {trajectory_path if args.save_trajectory else None}\n")
            handle.write(f"equilibriumCheckpoint {equilibrium_path}\n")
            handle.write(
                f"terminalLog {terminal_log_path if args.save_terminal_log else None}\n"
            )
            handle.write(f"meshFile {mesh_path}\n")
            handle.write(f"geometryMode {geometry_mode}\n")
            handle.write(f"nt {nt}\n")
            handle.write(f"ndof {ndof}\n")
            handle.write(f"order {args.order}\n")
            handle.write(f"quadDegree {qdeg}\n")
            handle.write(f"alphaT1 {params.alpha_t1}\n")
            handle.write(f"newtonSpdCsv {newton_spd_csv if (args.check_newton_spd or args.diagnose_capped_trial_coercivity) else None}\n")
            handle.write(f"alphaT2 {params.alpha_t2}\n")
            handle.write(f"c1T {c1_t}\n")
            handle.write(f"c2T {c2_t}\n")
            handle.write(f"targetSource {target_source_name}\n")
            handle.write(f"epsTRatio {params.eps_t_ratio}\n")
            handle.write(f"epsT {eps_t}\n")
            handle.write(f"targetArea {target_area}\n")
            handle.write(f"epsMode {args.eps_mode}\n")
            handle.write(f"epsPhiRatio {args.eps_ratio}\n")
            handle.write(f"epsPhiFixed {args.eps_phi}\n")
            handle.write(f"kappa {args.kappa}\n")
            handle.write(f"deltaC {args.delta_c}\n")
            handle.write(f"rhoDesignMass {rho_design_mass}\n")
            handle.write(f"rhoDesignMax {rho_design_max}\n")
            handle.write(f"rhoDesignL2 {rho_design_l2}\n")
            handle.write(f"phiTargetMax {phi_target_max}\n")
            handle.write(f"phiTargetL2 {phi_target_l2}\n")
            handle.write(f"tolRes {args.tol_res}\n")
            handle.write(f"maxNewtonInitialBudget {args.max_newton_it}\n")
            handle.write(f"newtonSoftCap {int(args.newton_soft_cap)}\n")
            handle.write(f"newtonSoftCapChunk {args.newton_soft_cap_chunk}\n")
            handle.write(f"newtonSoftCapFactor {args.newton_soft_cap_factor}\n")
            handle.write(f"newtonSoftCapWindow {args.newton_soft_cap_window}\n")
            handle.write(f"newtonSoftCapContraction {args.newton_soft_cap_contraction}\n")
            handle.write(f"innerNewtonTol {args.inner_newton_tol}\n")
            handle.write(f"innerTolMax {args.inner_tol_max}\n")
            handle.write(f"innerTolGamma {args.inner_tol_gamma}\n")
            handle.write(f"runInexactNewtonStudy {int(args.run_inexact_newton_study)}\n")
            handle.write(f"inexactNewtonTolerances {args.inexact_newton_tolerances}\n")
            handle.write(f"inexactNewtonReferenceTol {args.inexact_newton_reference_tol}\n")
            handle.write(f"inexactNewtonMaxSnapshots {args.inexact_newton_max_snapshots}\n")
            handle.write(f"inexactNewtonRows {inexact_newton_rows}\n")
            handle.write("inexactNewtonAcceptance model_feasibility_proxy\n")
            handle.write("inexactOneCorrectionReassemblesSensitivityJacobian 1\n")
            handle.write(f"minCertifiedAreaFraction {args.min_certified_area_fraction}\n")
            handle.write(f"solverPreset {args.solver_preset}\n")
            handle.write(f"stiffnessLinearSolver {stiffness_args.linear_solver}\n")
            handle.write(f"stiffnessKspType {stiffness_args.ksp_type}\n")
            handle.write(f"homotopyLinearSolver {homotopy_args.linear_solver}\n")
            handle.write(f"homotopyKspType {homotopy_args.ksp_type}\n")
            handle.write(f"nonlinearLinearSolver {nonlinear_args.linear_solver}\n")
            handle.write(f"nonlinearKspType {nonlinear_args.ksp_type}\n")
            handle.write(f"sensitivityLinearSolver {sensitivity_args.linear_solver}\n")
            handle.write(f"sensitivityKspType {sensitivity_args.ksp_type}\n")
            handle.write(f"sensitivityIterativeFallback {args.sensitivity_iterative_fallback}\n")
            handle.write(f"finalLinearSolver {final_solver_args.linear_solver}\n")
            handle.write(f"finalKspType {final_solver_args.ksp_type}\n")
            handle.write(f"iterativeFallbackSolver {args.iterative_fallback_solver}\n")
            handle.write(f"certifiedStop {int(args.certified_stop)}\n")
            handle.write(f"certifiedStopMinAccepted {args.certified_stop_min_accepted}\n")
            handle.write(f"certifiedStopPatience {args.certified_stop_patience}\n")
            handle.write(f"certifiedStopRtol {args.certified_stop_rtol}\n")
            handle.write(f"retainBestCertified {int(args.retain_best_certified)}\n")
            handle.write(f"acceptedOptimizationSteps {accepted_steps}\n")
            handle.write(f"trialLinearMaxIt {args.trial_linear_max_it}\n")
            handle.write(f"diagnoseCappedTrialCoercivity {int(args.diagnose_capped_trial_coercivity)}\n")
            handle.write(f"certifiedStaleSteps {certified_stale_steps}\n")
            handle.write(f"checkNewtonSpd {int(args.check_newton_spd)}\n")
            handle.write(f"newtonSpdEvery {args.newton_spd_every}\n")
            handle.write(f"newtonSpdPrefixFilter {args.newton_spd_prefix_filter}\n")
            handle.write(f"newtonSpdInertia {int(args.newton_spd_inertia)}\n")
            handle.write(f"newtonSpdEigTol {args.newton_spd_eig_tol}\n")
            handle.write(f"newtonSpdZeroTol {args.newton_spd_zero_tol}\n")
            handle.write(f"newtonSpdRecords {len(args.newton_spd_records)}\n")
            handle.write(f"bestCertifiedIteration {best_certified_iteration}\n")
            handle.write(f"bestCertifiedScore {best_certified_score}\n")
            handle.write(f"plotSevere {int(args.plot_severe)}\n")
            handle.write(f"searchCMin {c_min}\n")
            handle.write(f"searchCMax {c_upper}\n")
            handle.write(f"minWidth {min_width}\n")
            handle.write(f"initMode {args.init_mode}\n")
            handle.write(f"initSearch {args.init_search}\n")
            handle.write(f"initFallback {args.init_fallback}\n")
            handle.write(f"initialEquilibrium {args.initial_equilibrium}\n")
            handle.write(f"initialTransferMode {initial_transfer_mode or None}\n")
            handle.write(f"initialTransferCoordinateError {initial_transfer_coordinate_error}\n")
            handle.write(
                "initialTransferUnmatchedFraction "
                f"{float((initial_equilibrium_metadata or {}).get('transfer_unmatched_fraction', 0.0))}\n"
            )
            handle.write(
                "initialTransferAreaRelativeDifference "
                f"{float((initial_equilibrium_metadata or {}).get('transfer_area_relative_difference', 0.0))}\n"
            )
            handle.write(f"initialTransferAccepted {int(initial_transfer_succeeded)}\n")
            if initial_transfer_newton is not None:
                handle.write(f"initialTransferNewtonStatus {initial_transfer_newton.status}\n")
                handle.write(f"initialTransferNewtonResidual {initial_transfer_newton.residual}\n")
                handle.write(f"initialTransferNewtonIterations {initial_transfer_newton.iterations}\n")
            handle.write(f"initCandidate {init_candidate_name}\n")
            handle.write(f"initCandidateScore {init_candidate_score}\n")
            if selected_frozen_evaluation is not None:
                handle.write(f"initPsiHminus1 {selected_frozen_evaluation.psi}\n")
                handle.write(f"initResidualHminus1 {selected_frozen_evaluation.residual_dual}\n")
                handle.write(f"initFrozenLeakageRel {selected_frozen_evaluation.leakage_rel}\n")
                handle.write(f"initFrozenMissingRel {selected_frozen_evaluation.missing_rel}\n")
                handle.write(f"initFrozenActivityAreaRel {selected_frozen_evaluation.activity_area_rel}\n")
                handle.write(f"initFrozenGradPsi1 {selected_frozen_evaluation.grad_psi[0]}\n")
                handle.write(f"initFrozenGradPsi2 {selected_frozen_evaluation.grad_psi[1]}\n")
            handle.write(f"thresholdObjectiveMode {args.threshold_objective_mode}\n")
            handle.write(f"thresholdLeakageWeight {args.threshold_leakage_weight}\n")
            handle.write(f"thresholdMissingWeight {args.threshold_missing_weight}\n")
            handle.write(f"thresholdMeritRel {threshold_merit_rel(final_metrics, args)}\n")
            handle.write(f"homotopyTolRes {homotopy_tolerance(args)}\n")
            handle.write(f"homotopyInitialStep {args.homotopy_initial_step}\n")
            handle.write(f"homotopyMinStep {args.homotopy_min_step}\n")
            handle.write(f"homotopyMaxStep {args.homotopy_max_step}\n")
            handle.write(f"thresholdOptimizationLambda {optimization_homotopy_lambda}\n")
            handle.write(f"homotopyPredictor {int(args.homotopy_predictor)}\n")
            handle.write(f"homotopyAttempts {len(homotopy_attempts)}\n")
            for attempt_index, (attempt_name, attempt_result) in enumerate(homotopy_attempts, start=1):
                handle.write(f"homotopyAttempt{attempt_index}Name {attempt_name}\n")
                handle.write(f"homotopyAttempt{attempt_index}Status {attempt_result.status}\n")
                handle.write(f"homotopyAttempt{attempt_index}LambdaFinal {attempt_result.lambda_final}\n")
                handle.write(f"homotopyAttempt{attempt_index}Stages {attempt_result.stages}\n")
                handle.write(f"homotopyAttempt{attempt_index}RejectedSteps {attempt_result.rejected_steps}\n")
                handle.write(
                    f"homotopyAttempt{attempt_index}NewtonIterations "
                    f"{attempt_result.total_newton_iterations}\n"
                )
            if selected_homotopy_result is not None:
                handle.write(f"homotopyStatus {selected_homotopy_result.status}\n")
                handle.write(f"homotopyConverged {int(selected_homotopy_result.converged)}\n")
                handle.write(f"homotopyLambdaFinal {selected_homotopy_result.lambda_final}\n")
                handle.write(f"homotopyStages {selected_homotopy_result.stages}\n")
                handle.write(f"homotopyRejectedSteps {selected_homotopy_result.rejected_steps}\n")
                handle.write(f"homotopyNewtonIterations {selected_homotopy_result.total_newton_iterations}\n")
                handle.write(f"homotopyTangentSolveTime {selected_homotopy_result.tangent_solve_time}\n")
                handle.write(f"homotopyNewtonSolveTime {selected_homotopy_result.newton_solve_time}\n")
            handle.write(f"bestC1Phi {c1_phi}\n")
            handle.write(f"bestC2Phi {c2_phi}\n")
            handle.write(f"bestEpsPhi {eps_phi}\n")
            handle.write(f"finalNewtonTolRes {final_tol_res}\n")
            handle.write(f"finalNewtonTolStep {final_newton_args.tol_step}\n")
            handle.write(f"finalNewtonMaxIt {args.final_newton_max_it}\n")
            handle.write(f"finalNewtonStatus {final_newton.status}\n")
            handle.write(f"finalNewtonConverged {int(final_newton.converged)}\n")
            handle.write(f"finalNewtonIterations {final_newton.iterations}\n")
            handle.write(f"finalNewtonSolveTime {final_newton.solve_time}\n")
            handle.write(f"bestResidual {final_newton.residual}\n")
            handle.write(f"bestLeakage {final_metrics.leakage}\n")
            handle.write(f"bestMissing {final_metrics.missing}\n")
            handle.write(f"bestLeakageRel {final_metrics.leakage_rel}\n")
            handle.write(f"bestMissingRel {final_metrics.missing_rel}\n")
            handle.write(f"bestActivityArea {final_metrics.activity_area}\n")
            handle.write(f"bestCertifiedArea {final_metrics.certified_area}\n")
            handle.write(f"bestCertifiedAreaRel {final_certified_area_rel}\n")
            handle.write(f"bestCertifiedLeakage {final_metrics.certified_leakage}\n")
            handle.write(f"bestCertifiedLeakageRel {final_certified_leakage_rel}\n")
            handle.write(f"bestCertifiedMissing {final_metrics.certified_missing}\n")
            handle.write(f"bestActiveJaccard {final_compute_metrics['activeJaccard']}\n")
            handle.write(f"bestPlateauJaccard {final_compute_metrics['plateauJaccard']}\n")
            handle.write(f"bestRhoRel {final_compute_metrics['relRhoDesign']}\n")
            handle.write(f"finalStatus {final_status}\n")
            handle.write(f"timeTotal {elapsed}\n")

    stiffness_solver.close()
    successful_statuses = {"CONVERGED", "CONVERGED_CERTIFIED_SUBBAND"}
    if not final_newton.converged:
        exit_code = 3
    elif args.fail_on_nonconvergence and final_status not in successful_statuses:
        exit_code = 2
    else:
        exit_code = 0

    if terminal_log_capture is not None:
        root_print(
            comm,
            f"TERMINAL_LOG_COMPLETE path={terminal_log_path} exitCode={exit_code}",
        )
        comm.barrier()
        terminal_log_capture.close()
    return exit_code


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the reduced optimizer.

    Args:
        argv: Optional command-line argument list.  Passing ``None`` delegates
            to ``argparse``/``sys.argv``.

    Returns:
        Integer process return code from ``run_strategy``.
    """
    return run_strategy(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
