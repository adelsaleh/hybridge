#!/usr/bin/env python3
"""Reduced-space optimizer for torsion-initialized logistic-window thresholds.

This runner implements the reduced optimization algorithm described in
``docs/research/torsion_initialized_equilibrium``.  It keeps the
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

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dolfinx_torsion_initialized_window_fit_newton import (  # noqa: E402
    assemble_scalar,
    boundary_bc,
    compute_metrics,
    fit_phi_window_to_torsion_design,
    global_minmax,
    load_or_generate_mesh,
    quadrature_samples_for_fit,
    root_print,
    slug_for_path,
    solve_linear_form,
    solver_options,
    update_interpolated,
    window_numpy,
    window_ufl,
)
from dolfinx_torsion_initialized_closed_loop_refit import (  # noqa: E402
    InPlacePyVistaTorsionPlotter as _LocalPyVistaTorsionPlotter,
)


DEFAULT_RUN_LOG_ROOT = REPO_ROOT / "tmp" / "torsion_reduced_optimization_homotopy"

INITIALIZATION_CSV_FIELDS = [
    "record", "runTag", "method", "eval_id", "c1", "c2", "width", "eps",
    "psiHminus1", "residualHminus1", "Lrel", "Mrel", "activityAreaRel",
    "gradPsi1", "gradPsi2", "feasibleGeometry", "acceptedThresholdStep",
    "lambdaOld", "lambdaTrial", "dlambda", "tangentH1", "predictedResidual",
    "correctedResidual", "newtonIterations", "damping", "backtracks",
    "accepted", "elapsed",
]


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
    phi_values, rho_values, weights = quadrature_samples_for_fit(
        phi_target,
        rho_design,
        quadrature_degree=int(quadrature_degree),
    )
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
        self._cache: dict[tuple[str, str], FrozenThresholdEvaluation] = {}
        self._evaluation_ids: dict[int, int] = {}

    def _assemble_scalar(self, form) -> float:
        local = float(fem.assemble_scalar(form))
        return float(self.comm.allreduce(local, op=MPI.SUM))

    def evaluation_id(self, evaluation: FrozenThresholdEvaluation) -> int:
        return self._evaluation_ids[id(evaluation)]

    def evaluate(
            self,
            c1: float,
            c2: float,
            *,
            method: str,
    ) -> FrozenThresholdEvaluation:
        """Evaluate the exact assembled frozen H^-1 objective and gradient."""
        c1, c2 = project_thresholds(
            c1,
            c2,
            c_min=self.c_min,
            c_max=self.c_max,
            min_width=self.min_width,
        )
        key = (float(c1).hex(), float(c2).hex())
        cached = self._cache.get(key)
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
        grad_psi = np.array(
            [self._assemble_scalar(form) for form in self.gradient_forms],
            dtype=np.float64,
        )
        leakage = self._assemble_scalar(self.leakage_form)
        missing = self._assemble_scalar(self.missing_form)
        activity_area = self.target_area + leakage - missing
        target_scale = max(self.target_area, 1.0e-30)

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
) -> FrozenThresholdEvaluation:
    """Evaluate ``Psi_T(c)`` using a precompiled, matrix-reusing workspace."""
    return objective.evaluate(c1, c2, method=method)


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
) -> tuple[FrozenThresholdEvaluation, float, float]:
    """Minimize the assembled frozen H^-1 objective under geometric caps.

    A coarse center/width scan identifies a geometrically meaningful region.
    If necessary, the leakage and missing caps are relaxed lexicographically,
    without mixing them into the objective.  Small local grids then refine the
    lowest-Psi feasible point.
    """
    comm = objective.comm
    grid_size = int(args.init_hminus1_grid)
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
    coarse = [
        evaluate_frozen_threshold_objective(objective, pair[0], pair[1], method="coarse")
        for pair in coarse_pairs
    ]
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
    refine_size = int(args.init_hminus1_refine_grid)
    refine_passes = min(int(args.init_hminus1_refine_passes), int(args.init_hminus1_max_it))
    if args.verbosity >= 1:
        root_print(
            comm,
            f"HMINUS1_SCAN pass=coarse points={len(coarse)} feasible={len(feasible_coarse)} "
            f"Lcap={leakage_cap:.3e} Mcap={missing_cap:.3e} bestPsi={best.psi:.6e}",
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
        refined = [
            evaluate_frozen_threshold_objective(
                objective,
                pair[0],
                pair[1],
                method=f"refine_{refinement + 1}",
            )
            for pair in pairs
        ]
        feasible_refined = [evaluation for evaluation in refined if feasible(evaluation)]
        if feasible_refined:
            best = min([best, *feasible_refined], key=lambda evaluation: evaluation.psi)
        if args.verbosity >= 1:
            root_print(
                comm,
                f"HMINUS1_SCAN pass=refine_{refinement + 1} points={len(refined)} "
                f"feasible={len(feasible_refined)} bestPsi={best.psi:.6e}",
            )
        center_span *= 2.0 / max(refine_size - 1, 1)
        width_span *= 2.0 / max(refine_size - 1, 1)

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
        "--init-mode",
        choices=("homotopy", "legacy"),
        default="homotopy",
        help=(
            "'homotopy' minimizes the frozen H^-1 residual and continues the sharp torsion source "
            "at fixed thresholds; 'legacy' preserves candidate generation/direct projection/ranking"
        ),
    )
    parser.add_argument("--init-hminus1-grid", type=int, default=20)
    parser.add_argument("--init-hminus1-refine-grid", type=int, default=11)
    parser.add_argument("--init-hminus1-refine-passes", type=int, default=2)
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
    parser.add_argument("--include-fit-init", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--tol-grad", type=float, default=1.0e-8)
    parser.add_argument("--tol-res", type=float, default=1.0e-10)
    parser.add_argument("--inner-newton-tol", type=float, default=None)
    parser.add_argument("--inner-tol-max", type=float, default=1.0e-5)
    parser.add_argument("--inner-tol-gamma", type=float, default=1.0e-6)
    parser.add_argument("--max-newton-it", type=int, default=40)
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
    parser.add_argument("--linear-solver", choices=("mumps", "lu", "hypre", "gamg"), default="mumps")
    parser.add_argument("--ksp-type", default=None)
    parser.add_argument("--linear-rtol", type=float, default=1.0e-10)
    parser.add_argument("--linear-atol", type=float, default=1.0e-12)
    parser.add_argument("--linear-max-it", type=int, default=None)
    parser.add_argument("--terminal-every", type=int, default=1)
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
    parser.add_argument("--plot-initial", action=argparse.BooleanOptionalAction, default=True)
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
    if args.init_hminus1_grid < 2 or args.init_hminus1_refine_grid < 2:
        raise ValueError("require --init-hminus1-grid and --init-hminus1-refine-grid >= 2")
    if args.init_hminus1_refine_passes < 0 or args.init_hminus1_max_it < 1:
        raise ValueError("require nonnegative refinement passes and positive --init-hminus1-max-it")
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
    if args.tol_res <= 0.0 or args.inner_tol_max <= 0.0 or args.inner_tol_gamma <= 0.0:
        raise ValueError("require positive residual tolerances")
    if args.inner_newton_tol is not None and args.inner_newton_tol <= 0.0:
        raise ValueError("require positive --inner-newton-tol")
    if args.max_newton_it < 1:
        raise ValueError("require positive --max-newton-it")
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
    if (args.c1_phi is None) != (args.c2_phi is None):
        raise ValueError("--c1-phi and --c2-phi must be supplied together")
    if args.c1_phi is not None and not (args.c2_phi > args.c1_phi >= 0.0):
        raise ValueError("require 0 <= c1_phi < c2_phi")


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

    status = "MAX_NEWTON"
    converged = False
    last_step_h1 = math.inf
    last_alpha = 0.0
    last_bt = 0
    solve_time_total = 0.0
    final_residual = math.inf

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
    final_residual = residual_old
    if residual_old <= float(tol_res):
        return finish(NewtonResult(
            "CONVERGED_RESIDUAL", True, 0, residual_old,
            last_step_h1, last_alpha, last_bt, solve_time_total,
        ))

    for k in range(int(args.max_newton_it)):
        if homotopy_workspace is None:
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
        else:
            its, lin_res, solve_time = homotopy_workspace.linear_solver.solve_vector(
                stiffness_solver.residual_vector,
                du,
                scale=-1.0,
            )
        solve_time_total += solve_time
        step_h1 = (
            stiffness_solver.h1_seminorm(du)
            if stiffness_solver is not None
            else math.sqrt(max(assemble_scalar(comm, ufl.inner(ufl.grad(du), ufl.grad(du)) * dx), 0.0))
        )
        last_step_h1 = step_h1
        if step_h1 < float(args.tol_step):
            status = "FAIL_STEP_STAGNATION"
            return finish(NewtonResult(status, False, k, residual_old, step_h1, 0.0, 0, solve_time_total))

        old_u = u.x.array.copy()
        alpha = 1.0
        accepted = False
        bt = 0
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

        if not accepted:
            u.x.array[:] = old_u
            u.x.scatter_forward()
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
                f"alpha={alpha:.3e} bt={bt} stepH1={step_h1:.6e} linIts={its} linRes={lin_res:.3e}",
            )
        if final_residual <= float(tol_res):
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

    return finish(NewtonResult(
        status,
        converged,
        int(args.max_newton_it),
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
        frozen_residual_dual: float | None = None,
) -> HomotopyResult:
    """Continue the sharp torsion source to the semilinear source.

    The thresholds remain fixed throughout the continuation.  At an accepted
    continuation state ``(u, lambda)``, the branch tangent ``s = du/dlambda``
    solves

        J_lambda s = rho_amp * (W(u;c1,c2,eps) - 1_{B_T}),

    with ``J_lambda = K - lambda*rho_amp*W_s``.  A first-order predictor
    ``u + dlambda*s`` is then Newton-corrected at the trial lambda.  Failed
    corrector steps are rolled back and retried with a smaller continuation
    step.  Reaching ``lambda=1`` leaves ``u`` on the ordinary semilinear
    equilibrium used by the reduced optimizer.
    """
    comm = u.function_space.mesh.comm
    start = time.perf_counter()
    tol = homotopy_tolerance(args)
    lam = 0.0
    step = min(
        max(float(args.homotopy_initial_step), float(args.homotopy_min_step)),
        float(args.homotopy_max_step),
    )
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
        if lam >= 1.0 - 1.0e-14:
            return HomotopyResult(
                status="CONVERGED_LAMBDA1",
                converged=True,
                lambda_final=1.0,
                stages=accepted_stages,
                rejected_steps=rejected_steps,
                total_newton_iterations=total_newton_iterations,
                tangent_solve_time=tangent_solve_time,
                newton_solve_time=newton_solve_time,
                elapsed=time.perf_counter() - start,
                last_newton=last_newton,
            )

        dlambda = min(step, 1.0 - lam)
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

        trial_lambda = lam + dlambda
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
        )
        total_newton_iterations += int(trial_newton.iterations)
        newton_solve_time += float(trial_newton.solve_time)
        last_newton = trial_newton

        if initialization_writer is not None:
            initialization_writer.writerow({
                "record": "homotopy_stage",
                "runTag": run_tag,
                "method": "source_homotopy",
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
                "accepted": int(trial_newton.converged),
                "elapsed": time.perf_counter() - stage_start,
            })
            if initialization_handle is not None:
                initialization_handle.flush()

        if trial_newton.converged:
            lam = trial_lambda
            accepted_stages += 1
            if args.verbosity >= 1:
                root_print(
                    comm,
                    f"HOMOTOPY_STAGE prefix={prefix} stage={stage} accepted=1 "
                    f"lambda={lam:.6e} dlambda={dlambda:.6e} newtonIts={trial_newton.iterations} "
                    f"residual={trial_newton.residual:.6e} alpha={trial_newton.alpha:.3e} "
                    f"backtracks={trial_newton.backtracks}",
                )
            if lam >= 1.0 - 1.0e-14:
                return HomotopyResult(
                    status="CONVERGED_LAMBDA1",
                    converged=True,
                    lambda_final=1.0,
                    stages=accepted_stages,
                    rejected_steps=rejected_steps,
                    total_newton_iterations=total_newton_iterations,
                    tangent_solve_time=tangent_solve_time,
                    newton_solve_time=newton_solve_time,
                    elapsed=time.perf_counter() - start,
                    last_newton=last_newton,
                )
            if trial_newton.iterations <= 2 and trial_newton.backtracks == 0:
                step = min(float(args.homotopy_max_step), step * float(args.homotopy_step_grow))
            continue

        rejected_steps += 1
        u.x.array[:] = old_u
        u.x.scatter_forward()
        step *= float(args.homotopy_step_shrink)
        if args.verbosity >= 1:
            root_print(
                comm,
                f"HOMOTOPY_STAGE prefix={prefix} stage={stage} accepted=0 "
                f"lambda={lam:.6e} trialLambda={trial_lambda:.6e} dlambda={dlambda:.6e} "
                f"newtonStatus={trial_newton.status} residual={trial_newton.residual:.6e} "
                f"nextStep={step:.6e}",
            )
        if step < float(args.homotopy_min_step) * (1.0 - 1.0e-12):
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
        - float(rho_amp) * ws * trial * test
    ) * dx
    rhs_s1 = float(rho_amp) * dc1_activity * test * dx
    rhs_s2 = float(rho_amp) * dc2_activity * test * dx
    its, lin_res, solve_time, matrix_assembly_time, rhs_assembly_time, linear_solve_time = solve_same_matrix_forms(
        jac_expr,
        [rhs_s1, rhs_s2],
        [s1, s2],
        [bc],
        prefix=f"sensitivity_{iteration}_",
        solver=args.linear_solver,
        ksp_type=args.ksp_type,
        rtol=args.linear_rtol,
        atol=args.linear_atol,
        max_it=args.linear_max_it,
        verbosity=args.verbosity,
    )

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
    if feasible_leakage:
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
        Ratio ``int W_trial*W_current / int W_current``.  Values near zero
        indicate collapse or a substantial branch jump.
    """
    c1_const.value = PETSc.ScalarType(c1_trial)
    c2_const.value = PETSc.ScalarType(c2_trial)
    eps_const.value = PETSc.ScalarType(eps_trial)
    denominator = assemble_scalar(comm, activity_ref * dx)
    if denominator <= 1.0e-30:
        return 1.0
    numerator = assemble_scalar(comm, activity_ref * window_activity_const_ufl(u_trial, c1_const, c2_const, eps_const) * dx)
    return numerator / max(denominator, 1.0e-30)


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
    if args.plot_severe:
        args.plot = True
    comm = MPI.COMM_WORLD
    params = params_from_args(args)
    final_tol_res = final_newton_tolerance(args)
    run_dir = make_run_dir(args) if comm.rank == 0 else None
    run_dir = Path(comm.bcast(str(run_dir), root=0))
    run_tag = run_dir.name
    log_dir = run_dir / "logs"
    out_dir = run_dir / "out"
    if comm.rank == 0:
        log_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
    comm.barrier()

    opt_csv = log_dir / "optimization.csv"
    frame_csv = log_dir / "frames.csv"
    initialization_csv = log_dir / "initialization.csv"
    summary_path = out_dir / "summary.txt"
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
    total_start = time.perf_counter()

    root_print(comm, "========== START STRATEGY A DOLFINX WINDOW REDUCED OPTIMIZATION ==========")
    root_print(comm, f"RUN_TAG {run_tag}")
    root_print(comm, f"RUN_DIR {run_dir}")
    root_print(comm, f"OPT_CSV {opt_csv}")
    root_print(comm, f"FRAME_CSV {frame_csv}")
    root_print(comm, f"INITIALIZATION_CSV {initialization_csv}")
    root_print(comm, f"EQUILIBRIUM_CHECKPOINT {equilibrium_path}")
    root_print(comm, f"SUMMARY {summary_path}")
    root_print(
        comm,
        "REDUCED_OBJECTIVE "
        f"etaOut={args.eta_out:.6e} tolArea={args.tol_area:.6e} "
        f"epsMode={args.eps_mode} epsRatio={args.eps_ratio:.6e} epsPhi={args.eps_phi} "
        f"kappa={args.kappa:.6e} deltaC={args.delta_c:.6e} "
        f"tolRes={args.tol_res:.6e} finalNewtonTolRes={final_tol_res:.6e} "
        f"innerNewtonTol={args.inner_newton_tol} innerTolMax={args.inner_tol_max:.6e} "
        f"innerTolGamma={args.inner_tol_gamma:.6e} "
        f"initMode={args.init_mode} homotopyInitialStep={args.homotopy_initial_step:.3e} "
        f"homotopyMinStep={args.homotopy_min_step:.3e} homotopyMaxStep={args.homotopy_max_step:.3e} "
        f"homotopyTol={homotopy_tolerance(args):.3e} predictor={int(args.homotopy_predictor)} "
        f"minCertifiedAreaFraction={args.min_certified_area_fraction:.6e} "
        f"plotSevere={int(args.plot_severe)}",
    )

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
    plotter = InPlacePyVistaTorsionPlotter(args, run_tag=run_tag, run_dir=run_dir, frame_writer=frame_writer, comm=comm)

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

    stiffness_solver = FixedStiffnessSolver(
        stiffness_form,
        V,
        [bc],
        prefix="shared_stiffness_",
        solver=args.linear_solver,
        ksp_type=args.ksp_type,
        rtol=args.linear_rtol,
        atol=args.linear_atol,
        max_it=args.linear_max_it,
    )
    its, rel, solve_time = stiffness_solver.solve_form(1.0 * test * dx, T)
    _, tmax = global_minmax(comm, T)
    c1_t = params.alpha_t1 * tmax
    c2_t = params.alpha_t2 * tmax
    eps_t = params.eps_t_ratio * (c2_t - c1_t)
    tau_mask = ufl.conditional(ufl.gt(T, c1_t), ufl.conditional(ufl.lt(T, c2_t), 1.0, 0.0), 0.0)
    target_density = float(params.rho_amp) * tau_mask
    update_interpolated(tau_band, tau_mask)
    target_area = assemble_scalar(comm, tau_mask * dx)
    root_print(comm, f"SOLVER_OK problem=torsion iters={its} rel={rel:.3e} time={solve_time:.3f}")
    root_print(
        comm,
        f"TORSION Tmax={tmax:.6e} c1T={c1_t:.6e} c2T={c2_t:.6e} "
        f"epsT={eps_t:.6e} targetArea={target_area:.6e}",
    )
    if target_area <= 0.0:
        raise RuntimeError("torsion target band has zero area")

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

    init_candidate_name = "manual"
    init_candidate_score = math.nan
    init_candidates: list[InitialWindowCandidate] = []
    if args.c1_phi is not None:
        c1_phi = float(args.c1_phi)
        c2_phi = float(args.c2_phi)
        fit_result = None
    elif args.init_mode == "legacy" and args.include_fit_init:
        fit_quad_degree = args.fit_window_quad_degree if args.fit_window_quad_degree is not None else qdeg
        fit_result = fit_phi_window_to_torsion_design(
            phi_target,
            rho_design,
            rho_design_l2=rho_design_l2,
            phi_design_max=phi_target_max,
            eps_ratio=args.eps_ratio if args.eps_mode == "relative" else max(args.eps_phi / max(c_scale, 1.0e-30), 1.0e-6),
            rho_amp=params.rho_amp,
            quadrature_degree=fit_quad_degree,
            grid_points=args.fit_window_grid,
            refine_points=args.fit_window_refine_grid,
            refine_passes=args.fit_window_refine_passes,
            histogram_bins=args.fit_window_bins,
        )
        root_print(
            comm,
            f"FIT_INIT c1={fit_result.c1:.6e} c2={fit_result.c2:.6e} "
            f"objectiveRel={fit_result.objective_rel:.6e} time={fit_result.elapsed:.3f}",
        )
        init_candidates = build_initial_window_candidates(
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
        )
        if not init_candidates:
            raise RuntimeError("failed to generate any automatic initial window candidates")
        for candidate in init_candidates:
            root_print(
                comm,
                f"INIT_CANDIDATE name={candidate.name} c1={candidate.c1:.6e} "
                f"c2={candidate.c2:.6e} width={candidate.c2 - candidate.c1:.6e} "
                f"epsPhi={candidate.eps_phi:.6e} score={candidate.score:.6e} "
                f"Lrel={candidate.leakage_rel:.6e} Mrel={candidate.missing_rel:.6e} "
                f"areaRel={candidate.area_rel:.6e} activeJ={candidate.active_jaccard:.6e} "
                f"l2Rel={candidate.l2_rel:.6e}",
            )
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

    u.x.array[:] = phi_target.x.array
    u.x.scatter_forward()
    c1_const = fem.Constant(domain, PETSc.ScalarType(c1_phi))
    c2_const = fem.Constant(domain, PETSc.ScalarType(c2_phi))
    eps_const = fem.Constant(domain, PETSc.ScalarType(eps_phi))
    selected_frozen_evaluation: FrozenThresholdEvaluation | None = None
    if args.init_mode == "homotopy":
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
            selected_frozen_evaluation, leakage_cap, missing_cap = optimize_frozen_hminus1_thresholds(
                frozen_objective,
                args=args,
            )
            init_candidate_name = "hminus1_grid_refined"
        if args.verify_homotopy_init:
            verify_frozen_hminus1_gradient(frozen_objective, selected_frozen_evaluation)
        c1_phi = selected_frozen_evaluation.c1
        c2_phi = selected_frozen_evaluation.c2
        eps_phi = selected_frozen_evaluation.eps_phi
        init_candidate_score = selected_frozen_evaluation.psi
        c1_const.value = PETSc.ScalarType(c1_phi)
        c2_const.value = PETSc.ScalarType(c2_phi)
        eps_const.value = PETSc.ScalarType(eps_phi)
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
            f"HMINUS1_INIT c1={c1_phi:.6e} c2={c2_phi:.6e} "
            f"width={c2_phi - c1_phi:.6e} eps={eps_phi:.6e} "
            f"dualResidual={selected_frozen_evaluation.residual_dual:.6e} "
            f"Psi={selected_frozen_evaluation.psi:.6e} "
            f"frozenLrel={selected_frozen_evaluation.leakage_rel:.6e} "
            f"frozenMrel={selected_frozen_evaluation.missing_rel:.6e} "
            f"activityAreaRel={selected_frozen_evaluation.activity_area_rel:.6e} "
            f"projectedGradNorm={selected_frozen_evaluation.projected_grad_norm:.6e} "
            f"numberObjectiveEvaluations={len(frozen_objective.records)} "
            f"Lcap={leakage_cap:.6e} Mcap={missing_cap:.6e}",
        )
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
            args=args,
        )
        if args.init_mode == "homotopy"
        else None
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
            args=args,
            prefix=f"init_{index}_{slug_for_path(candidate.name)}",
            stiffness_solver=stiffness_solver,
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

    if init_candidates:
        projected_candidates = [
            project_initial_candidate(candidate, index)
            for index, candidate in enumerate(init_candidates)
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
    elif args.init_mode == "homotopy":
        # The manual pair or the single frozen-H^-1 minimizer is continued
        # exactly once before the reduced optimization begins.
        u.x.array[:] = phi_target.x.array
        u.x.scatter_forward()
        rho.x.array[:] = rho_design.x.array
        rho.x.scatter_forward()
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
            args=args,
            prefix=f"init_homotopy_{slug_for_path(init_candidate_name)}",
            initialization_writer=initialization_writer,
            initialization_handle=initialization_handle,
            run_tag=run_tag,
            frozen_residual_dual=(
                selected_frozen_evaluation.residual_dual
                if selected_frozen_evaluation is not None
                else None
            ),
        )
        homotopy_workspace.update_nonlinear_density(rho)
        if not selected_homotopy_result.converged:
            raise RuntimeError(
                f"homotopy initialization failed at lambda={selected_homotopy_result.lambda_final:.6e} "
                f"with status {selected_homotopy_result.status}; use --init-mode legacy to recover the old path"
            )
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

    if args.plot_design:
        selected_u = u.x.array.copy()
        selected_rho = rho.x.array.copy()
        u.x.array[:] = phi_target.x.array
        u.x.scatter_forward()
        update_interpolated(rho, window_density_const_ufl(u, c1_const, c2_const, eps_const, params.rho_amp))
        plotter.emit(
            [T, tau_band, rho_design, phi_target, rho],
            ["Torsion T", "tau band", "rhoDesign", "phiT", "rho(phiT; c)"],
            stage="DESIGN",
            ieps=-1,
            k=-1,
            eps_phi=eps_phi,
            residual=0.0,
            metrics={"massRho": rho_design_mass, "maxRho": rho_design_max},
            token="design",
            save=bool(args.frame_design),
            show=True,
            nt=nt,
            ndof=ndof,
        )
        u.x.array[:] = selected_u
        u.x.scatter_forward()
        rho.x.array[:] = selected_rho
        rho.x.scatter_forward()

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
        "c1Phi", "c2Phi", "width", "epsPhi", "trustRadius",
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
    trust_radius_abs = min(max(float(args.trust_radius) * c_scale, float(args.trust_radius_min) * c_scale), float(args.trust_radius_max) * c_scale)
    previous_discrepancy_rel = 2.0
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
                args=args,
                prefix=f"outer_{k}",
                stiffness_solver=stiffness_solver,
                plot_callback=make_newton_plot_callback(outer_k=k, stage="NEWTON_OUTER"),
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
            previous_discrepancy_rel = metrics.leakage_rel + metrics.missing_rel
            final_metrics = metrics
            residual_form_for_metrics = (
                ufl.inner(ufl.grad(u), ufl.grad(test))
                - window_density_const_ufl(u, c1_const, c2_const, eps_const, params.rho_amp) * test
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
            step: ParameterStep | None = None
            status = "ITERATE"
            accepted = False
            actual_reduction = 0.0
            rho_ratio = math.nan
            branch_overlap = 1.0
            projected_grad_norm = math.nan
            inner_projection_ok = (
                newton.converged
                or newton.residual <= max(10.0 * float(inner_tol), float(args.tol_res))
            )
            stop_ready = (
                inner_projection_ok
                and metrics.leakage_rel <= float(args.eta_out)
                and metrics.missing_rel <= float(args.tol_area)
            )

            if k < int(args.max_opt_it):
                # Algorithm steps 3-7: assemble the Newton matrix, assemble
                # parameter RHS vectors, solve the two sensitivity equations,
                # assemble functional derivatives, and combine them into
                # reduced leakage/missing-area gradients.
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
                    args=args,
                    iteration=k,
                )
                gradient_time = time.perf_counter() - gradient_start
                log_algorithm_step(
                    comm,
                    args,
                    iteration=k,
                    step=3,
                    label="assemble_newton_matrix",
                    elapsed=gradient.matrix_assembly_time,
                    detail="matrix=J_U reused for sensitivity solves",
                )
                log_algorithm_step(
                    comm,
                    args,
                    iteration=k,
                    step=4,
                    label="assemble_residual_parameter_derivatives",
                    elapsed=gradient.rhs_assembly_time,
                    detail="columns=dr/dc1,dr/dc2",
                )
                log_algorithm_step(
                    comm,
                    args,
                    iteration=k,
                    step=5,
                    label="solve_equilibrium_sensitivities",
                    elapsed=gradient.linear_solve_time,
                    detail=(
                        f"total={gradient.solve_time:.6f}s "
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
                    elapsed=gradient.gradient_assembly_time,
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
                projected_grad_norm = float(np.linalg.norm(gradient.grad_m if metrics.leakage_rel <= args.eta_out else gradient.grad_l))
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
                            args=args,
                            prefix=f"outer_{k}_trial",
                            stiffness_solver=stiffness_solver,
                            plot_callback=make_newton_plot_callback(outer_k=k, stage="NEWTON_TRIAL"),
                        )
                        correction_time = time.perf_counter() - correction_start
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
                        if current_feasible:
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
                                - window_density_const_ufl(
                                    u,
                                    c1_const,
                                    c2_const,
                                    eps_const,
                                    params.rho_amp,
                                ) * test
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
                            previous_discrepancy_rel = metrics.leakage_rel + metrics.missing_rel
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

            if final_status == "MAX_OPT_IT" and status in {"STEP_TOO_SMALL"}:
                final_status = status
            if args.verbosity >= 1 and args.terminal_every > 0 and (
                    k % int(args.terminal_every) == 0
                    or status in {"CONVERGED", "REJECT", "STEP_TOO_SMALL"}
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
                    "sensIts1": "" if gradient is None else gradient.solve_iterations[0],
                    "sensIts2": "" if gradient is None else gradient.solve_iterations[1],
                    "sensRes1": "" if gradient is None else gradient.solve_residuals[0],
                    "sensRes2": "" if gradient is None else gradient.solve_residuals[1],
                    "sensSolveTime": "" if gradient is None else gradient.solve_time,
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
                )

            if status in {"CONVERGED", "STEP_TOO_SMALL"}:
                break
    finally:
        if opt_handle is not None:
            opt_handle.close()

    if final_metrics is None or final_newton is None or final_compute_metrics is None:
        raise RuntimeError("optimization did not produce a final iterate")

    # Final feasibility policy: no matter how loose the adaptive outer Newton
    # solves were, the reported final state must satisfy the requested final
    # residual tolerance at the final thresholds.
    eps_phi = epsilon_from_thresholds(args, c1_phi, c2_phi)
    final_newton_args = argparse.Namespace(**vars(args))
    final_newton_args.max_newton_it = int(args.final_newton_max_it)
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
        stiffness_solver=stiffness_solver,
        plot_callback=make_newton_plot_callback(outer_k=-1, stage="NEWTON_FINAL"),
    )
    final_exact_time = time.perf_counter() - final_exact_start
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
        - window_density_const_ufl(u, c1_const, c2_const, eps_const, params.rho_amp) * test
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
    write_equilibrium_checkpoint(
        equilibrium_path,
        phi=u,
        rho=rho,
        metadata={
            "format": "hdgfem_equilibrium_v1",
            "run_tag": run_tag,
            "mesh_path": str(Path(mesh_path).resolve()),
            "order": int(args.order),
            "quadrature_degree": int(qdeg),
            "rho_amp": float(params.rho_amp),
            "c1_phi": float(c1_phi),
            "c2_phi": float(c2_phi),
            "eps_phi": float(eps_phi),
            "final_residual": float(final_newton.residual),
            "num_cells": int(nt),
            "num_dofs": int(ndof),
        },
    )
    final_geometry_ok = (
        final_metrics.leakage_rel <= float(args.eta_out)
        and final_metrics.missing_rel <= float(args.tol_area)
    )
    # The optimizer's strict objective is a full soft-band match.  The
    # certified-subband status is a weaker but useful success mode: a genuine
    # plateau with small certified leakage and enough certified area.
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
            )
        finally:
            args.plot_mode = original_plot_mode
    if frame_handle is not None:
        frame_handle.close()

    elapsed = time.perf_counter() - total_start
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

    if comm.rank == 0:
        with summary_path.open("w", encoding="utf-8") as handle:
            handle.write(f"runTag {run_tag}\n")
            handle.write(f"runDir {run_dir}\n")
            handle.write(f"initializationCsv {initialization_csv}\n")
            handle.write(f"equilibriumCheckpoint {equilibrium_path}\n")
            handle.write(f"meshFile {mesh_path}\n")
            handle.write(f"geometryMode {geometry_mode}\n")
            handle.write(f"nt {nt}\n")
            handle.write(f"ndof {ndof}\n")
            handle.write(f"order {args.order}\n")
            handle.write(f"quadDegree {qdeg}\n")
            handle.write(f"alphaT1 {params.alpha_t1}\n")
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
            handle.write(f"innerNewtonTol {args.inner_newton_tol}\n")
            handle.write(f"innerTolMax {args.inner_tol_max}\n")
            handle.write(f"innerTolGamma {args.inner_tol_gamma}\n")
            handle.write(f"minCertifiedAreaFraction {args.min_certified_area_fraction}\n")
            handle.write(f"plotSevere {int(args.plot_severe)}\n")
            handle.write(f"searchCMin {c_min}\n")
            handle.write(f"searchCMax {c_upper}\n")
            handle.write(f"minWidth {min_width}\n")
            handle.write(f"initMode {args.init_mode}\n")
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
            handle.write(f"homotopyTolRes {homotopy_tolerance(args)}\n")
            handle.write(f"homotopyInitialStep {args.homotopy_initial_step}\n")
            handle.write(f"homotopyMinStep {args.homotopy_min_step}\n")
            handle.write(f"homotopyMaxStep {args.homotopy_max_step}\n")
            handle.write(f"homotopyPredictor {int(args.homotopy_predictor)}\n")
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
    if not final_newton.converged:
        return 3
    successful_statuses = {"CONVERGED", "CONVERGED_CERTIFIED_SUBBAND"}
    if args.fail_on_nonconvergence and final_status not in successful_statuses:
        return 2
    return 0


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
