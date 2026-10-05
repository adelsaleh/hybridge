#!/usr/bin/env python3
"""Metric projection of a torsion design onto a local equilibrium branch.

This fixed-mesh DOLFINx driver minimizes the normalized H0-1 distance between
the semilinear equilibrium and the sharp torsion-design potential.  Leakage
and missing area are inequality safeguards, not objective penalties.  Every
state used by the optimizer is projected with the same requested strict
stiffness-dual residual tolerance; this mode contains no inexact-Newton path.

The distributed PDE work reuses the torsion optimizer's stiffness cache,
damped Newton implementation, window assembly, branch-overlap diagnostic, and
shared-Jacobian sensitivity solver.  Only the two-dimensional center--width
algebra is implemented locally through ``torsion_h1_projection_controls``.
"""

from __future__ import annotations

# Support direct execution as well as python -m from a checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[5]))

import sys
from pathlib import Path

_BOOTSTRAP_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_BOOTSTRAP_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_SCRIPT_DIR))

from projects.diocotron.dolfinx.runtime.terminal_log_capture import (  # noqa: E402
    TerminalLogCapture,
    get_bootstrap_terminal_log_capture,
    start_bootstrap_terminal_log_capture,
)

start_bootstrap_terminal_log_capture(sys.argv[1:])

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
import time
from typing import Any, Callable, Sequence

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from scipy.optimize import minimize

from dolfinx import fem, io


REPO_ROOT = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import projects.diocotron.dolfinx.torsion.optimization.h1_controls as controls  # noqa: E402
import projects.diocotron.dolfinx.torsion.optimization.homotopy as shared  # noqa: E402
from projects.diocotron.dolfinx.plotting.mpi_pyvista import MPIPyVistaTorsionPlotter  # noqa: E402


DEFAULT_RUN_ROOT = REPO_ROOT / "projects/diocotron/runs" / "dolfinx_torsion_h1_projection"


INITIALIZATION_FIELDS = [
    "candidate",
    "center",
    "width",
    "c1",
    "c2",
    "epsilon",
    "frozen_leakage",
    "frozen_missing",
    "frozen_coverage",
    "newton_success",
    "newton_status",
    "newton_iterations",
    "damping_history",
    "initial_dual_residual",
    "final_dual_residual",
    "first_correction_h1",
    "projected_objective",
    "projected_leakage",
    "projected_missing",
    "activity_area_ratio",
    "overlap_area",
    "overlap_area_ratio",
    "recall",
    "precision",
    "jaccard",
    "linear_iterations",
    "linear_convergence_reason",
    "coercivity_margin",
    "coercivity_error",
    "topology_components",
    "topology_raw_components",
    "topology_largest_fraction",
    "topology_second_fraction",
    "selected",
    "reason",
]


OUTER_FIELDS = [
    "event",
    "iteration",
    "trial",
    "accepted",
    "center",
    "width",
    "c1",
    "c2",
    "epsilon",
    "objective",
    "relative_h1_distance",
    "leakage",
    "missing",
    "leakage_slack",
    "missing_slack",
    "coverage",
    "activity_area_ratio",
    "overlap_area",
    "overlap_area_ratio",
    "recall",
    "precision",
    "jaccard",
    "grad_objective_center",
    "grad_objective_width",
    "grad_leakage_center",
    "grad_leakage_width",
    "grad_missing_center",
    "grad_missing_width",
    "gn_00",
    "gn_01",
    "gn_11",
    "gn_condition",
    "step_center",
    "step_width",
    "scaled_step_inf",
    "trust_radius",
    "predicted_objective_reduction",
    "actual_objective_reduction",
    "predicted_leakage",
    "predicted_missing",
    "actual_leakage",
    "actual_missing",
    "acceptance_ratio",
    "newton_status",
    "newton_iterations",
    "newton_initial_residual",
    "newton_final_residual",
    "minimum_damping",
    "damping_history",
    "predictor_corrector_h1",
    "predictor_corrector_relative",
    "predicted_state_change_relative",
    "branch_overlap",
    "topology_components",
    "topology_raw_components",
    "topology_largest_fraction",
    "topology_second_fraction",
    "linear_iterations",
    "linear_convergence_reason",
    "sensitivity_iterations_center",
    "sensitivity_iterations_width",
    "sensitivity_residual_center",
    "sensitivity_residual_width",
    "sensitivity_reason_center",
    "sensitivity_reason_width",
    "sensitivity_solve_time",
    "coercivity_margin",
    "coercivity_error",
    "geometric_violation",
    "predicted_geometric_violation",
    "primary_merit_mode",
    "old_primary_merit",
    "new_primary_merit",
    "primary_merit_improvement",
    "primary_merit_required_improvement",
    "functional_stagnation_count",
    "active_constraints",
    "kkt_residual",
    "kkt_stationarity",
    "kkt_primal",
    "kkt_complementarity",
    "reason",
]


HOMOTOPY_THRESHOLD_FIELDS = [
    "trigger",
    "stage",
    "lambda",
    "source_trial_lambda",
    "threshold_step",
    "trial",
    "accepted",
    "objective_scale",
    "old_center",
    "old_width",
    "new_center",
    "new_width",
    "new_c1",
    "new_c2",
    "new_epsilon",
    "old_objective",
    "new_objective",
    "old_leakage",
    "new_leakage",
    "old_missing",
    "new_missing",
    "old_violation",
    "new_violation",
    "grad_objective",
    "grad_leakage",
    "grad_missing",
    "gauss_newton",
    "gauss_newton_condition",
    "step_center",
    "step_width",
    "step_c1",
    "step_c2",
    "edge_step_fraction",
    "scaled_step_inf",
    "trust_radius",
    "predicted_objective_reduction",
    "actual_objective_reduction",
    "predicted_leakage",
    "predicted_missing",
    "acceptance_ratio",
    "newton_status",
    "newton_iterations",
    "newton_initial_residual",
    "newton_final_residual",
    "minimum_damping",
    "predictor_corrector_h1",
    "source_state_step_h1",
    "branch_overlap",
    "topology_components",
    "topology_raw_components",
    "topology_largest_fraction",
    "topology_second_fraction",
    "reason",
]


@dataclass
class NewtonTrace:
    """Aggregated per-projection diagnostics captured from the shared solver."""

    damping_history: tuple[float, ...]
    minimum_damping: float
    first_direction_h1: float
    first_correction_h1: float
    linear_iterations: int
    linear_reason: str


class NewtonTraceWriter:
    """Forward shared Newton rows to CSV while retaining per-solve histories."""

    def __init__(self, writer: csv.DictWriter | None, handle) -> None:
        """Initialize optional CSV forwarding and the in-memory trace map."""
        self.writer = writer
        self.handle = handle
        self.records: dict[str, list[dict[str, Any]]] = {}

    def writerow(self, row: dict[str, Any]) -> None:
        """Record one row using the interface expected by the shared solver."""
        copied = dict(row)
        self.records.setdefault(str(copied.get("solve", "")), []).append(copied)
        if self.writer is not None:
            self.writer.writerow(copied)
            self.handle.flush()

    def clear(self, solve: str) -> None:
        """Discard stale rows for a solve prefix before a new projection."""
        for key in tuple(self.records):
            if key == solve or key.startswith(f"{solve}_"):
                self.records.pop(key, None)

    def summarize(self, solve: str) -> NewtonTrace:
        """Aggregate damping, correction, and linear-iteration diagnostics."""
        rows = [
            row
            for key, keyed_rows in self.records.items()
            if key == solve or key.startswith(f"{solve}_")
            for row in keyed_rows
        ]
        step_rows = [
            row
            for row in rows
            if _finite(row.get("step_norm")) and float(row.get("step_norm", 0.0)) > 0.0
        ]
        damping = tuple(
            float(row["damping"])
            for row in step_rows
            if _finite(row.get("damping")) and float(row.get("damping", 0.0)) > 0.0
        )
        first_direction = float(step_rows[0]["step_norm"]) if step_rows else 0.0
        first_alpha = float(step_rows[0].get("damping", 0.0)) if step_rows else 0.0
        failure_statuses = {
            str(row.get("status", ""))
            for row in rows
            if "FAIL" in str(row.get("status", "")) or "CAP" in str(row.get("status", ""))
        }
        reason = ";".join(sorted(failure_statuses)) if failure_statuses else "PETSC_NONNEGATIVE_REASON"
        return NewtonTrace(
            damping_history=damping,
            minimum_damping=min(damping) if damping else 1.0,
            first_direction_h1=first_direction,
            first_correction_h1=first_alpha * first_direction,
            linear_iterations=sum(int(row.get("ksp_iterations", 0) or 0) for row in rows),
            linear_reason=reason,
        )


@dataclass
class ProjectionMetrics:
    """Reduced objective, geometric safeguards, and branch diagnostics."""

    objective: float
    leakage: float
    missing: float
    activity_area_ratio: float
    overlap_area: float
    overlap_area_ratio: float
    recall: float
    precision: float
    jaccard: float


@dataclass
class ProjectionResult:
    """Strict state-projection result with diagnostics."""

    newton: shared.NewtonResult
    trace: NewtonTrace
    initial_residual: float
    metrics: ProjectionMetrics | None
    predictor_corrector_h1: float
    predicted_state_change_h1: float
    branch_overlap: float
    topology: controls.ActivityTopology | None
    coercivity_margin: float
    coercivity_error: float
    rejection_reason: str


@dataclass
class ReducedData:
    """Sensitivities, total gradients, and Gauss--Newton model at one state."""

    gradient_objective: np.ndarray
    gradient_leakage: np.ndarray
    gradient_missing: np.ndarray
    hessian: np.ndarray
    condition_number: float
    iterations: tuple[int, int]
    residuals: tuple[float, float]
    reasons: tuple[str, str]
    solve_time: float


class DistributedActivityTopology:
    """Collectively diagnose activity connectivity on the fixed cell graph.

    The expensive graph layout is gathered once.  Each subsequent call sends
    only one peak activity value per owned cell to rank zero, where the pure
    NumPy hysteretic component counter is evaluated and broadcast.  Cell
    peaks use every scalar Lagrange interpolation value in the cell, avoiding
    missed narrow bands without introducing another finite-element space.
    """

    def __init__(self, function_space) -> None:
        self.V = function_space
        self.domain = function_space.mesh
        self.comm = self.domain.comm
        if int(function_space.dofmap.index_map_bs) != 1:
            raise ValueError("activity topology requires a scalar function space")
        topology = self.domain.topology
        tdim = topology.dim
        topology.create_connectivity(tdim, tdim - 1)
        cell_to_facet = topology.connectivity(tdim, tdim - 1)
        if cell_to_facet is None:
            raise RuntimeError("failed to construct cell-to-facet connectivity")
        facet_map = topology.index_map(tdim - 1)
        local_facets = np.arange(
            facet_map.size_local + facet_map.num_ghosts, dtype=np.int32
        )
        global_facets = np.asarray(
            facet_map.local_to_global(local_facets), dtype=np.int64
        )
        self.owned_cell_count = int(topology.index_map(tdim).size_local)
        local_rows = [
            global_facets[np.asarray(cell_to_facet.links(cell), dtype=np.int32)]
            for cell in range(self.owned_cell_count)
        ]
        local_facet_count = len(local_rows[0]) if local_rows else 0
        facet_counts = self.comm.allgather(local_facet_count)
        facets_per_cell = max(facet_counts, default=0)
        if facets_per_cell < 1:
            raise RuntimeError("activity topology received a mesh with no cell facets")
        if any(count not in (0, facets_per_cell) for count in facet_counts):
            raise ValueError("mixed cell types are not supported by the topology guard")
        local_array = (
            np.asarray(local_rows, dtype=np.int64).reshape(-1, facets_per_cell)
            if local_rows
            else np.empty((0, facets_per_cell), dtype=np.int64)
        )
        gathered = self.comm.gather(local_array, root=0)
        self.global_cell_facets = (
            np.concatenate(gathered, axis=0)
            if self.comm.rank == 0
            else None
        )
        self.global_cell_count = int(
            self.comm.allreduce(self.owned_cell_count, op=MPI.SUM)
        )
        if self.comm.rank == 0 and self.global_cell_facets.shape[0] != self.global_cell_count:
            raise RuntimeError("activity topology gather lost owned mesh cells")

    def analyze(
        self,
        activity: fem.Function,
        *,
        core_level: float,
        bridge_level: float,
        min_component_fraction: float,
    ) -> controls.ActivityTopology:
        """Return a collective hysteretic component summary for ``activity``."""
        if activity.function_space is not self.V:
            raise ValueError("activity topology function-space mismatch")
        activity.x.scatter_forward()
        coefficients = np.asarray(activity.x.array, dtype=np.float64)
        local_peaks = np.empty(self.owned_cell_count, dtype=np.float64)
        for cell in range(self.owned_cell_count):
            dofs = self.V.dofmap.cell_dofs(cell)
            local_peaks[cell] = float(np.max(coefficients[dofs]))
        gathered = self.comm.gather(local_peaks, root=0)
        if self.comm.rank == 0:
            peaks = np.concatenate(gathered)
            result = controls.hysteretic_activity_components(
                self.global_cell_facets,
                peaks,
                core_level=core_level,
                bridge_level=bridge_level,
                min_component_fraction=min_component_fraction,
            )
        else:
            result = None
        return self.comm.bcast(result, root=0)


def _finite(value: Any) -> bool:
    """Return whether a value can be interpreted as a finite float."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _json_array(values: Sequence[float]) -> str:
    """Serialize numeric diagnostics compactly for CSV cells."""
    return json.dumps([float(value) for value in values], separators=(",", ":"))


def _finite_or_none(value: Any) -> float | None:
    """Convert a finite numeric diagnostic to float and JSON null otherwise."""
    return float(value) if _finite(value) else None


def _copy_function(source: fem.Function, target: fem.Function) -> None:
    """Copy all local/ghost coefficients between matching scalar spaces."""
    target.x.array[:] = source.x.array
    target.x.scatter_forward()


def _make_run_dir(args: argparse.Namespace, comm: MPI.Comm) -> Path:
    """Create a collision-free run directory on rank zero and broadcast it."""
    if comm.rank == 0:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        if args.run_dir is None:
            tag = f"{shared.slug_for_path(args.run_tag)}_" if args.run_tag else ""
            base = DEFAULT_RUN_ROOT / f"{tag}{stamp}"
        else:
            base = args.run_dir.expanduser().resolve()
            if base.exists():
                base = base.parent / f"{base.name}_{stamp}"
        candidate = base
        suffix = 1
        while candidate.exists():
            candidate = base.parent / f"{base.name}_{suffix:03d}"
            suffix += 1
        candidate.mkdir(parents=True)
        for child in ("logs", "out", "fields", "plots"):
            (candidate / child).mkdir()
        encoded = str(candidate)
    else:
        encoded = None
    result = Path(comm.bcast(encoded, root=0))
    comm.barrier()
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for the fixed-mesh projection mode."""
    parser = argparse.ArgumentParser(
        description=(
            "Project the sharp torsion-design potential onto one locally tracked "
            "semilinear equilibrium branch in the normalized H0-1 metric."
        )
    )
    scientific = parser.add_argument_group("scientific bounds (required)")
    scientific.add_argument("--center-min", type=float, required=True, help="lower bound m_min")
    scientific.add_argument("--center-max", type=float, required=True, help="upper bound m_max")
    scientific.add_argument("--width-min", type=float, required=True, help="positive lower bound d_min")
    scientific.add_argument("--width-max", type=float, required=True, help="upper bound d_max")
    scientific.add_argument("--leakage-max", type=float, required=True, help="maximum normalized leakage L")
    scientific.add_argument("--missing-max", type=float, required=True, help="maximum normalized missing area M")

    mesh = parser.add_argument_group("mesh and discretization")
    mesh.add_argument("--mesh", type=Path, default=None, help="existing Gmsh .msh file; otherwise generate a smooth star")
    mesh.add_argument("--mesh-size", type=float, default=0.30)
    mesh.add_argument("--order", type=int, default=2)
    mesh.add_argument("--quad-degree", type=int, default=None)
    mesh.add_argument("--star-n", type=int, default=240)
    mesh.add_argument("--star-r0", type=float, default=1.0)
    mesh.add_argument("--star-amp", type=float, default=0.18)
    mesh.add_argument("--star-mode", type=int, default=5)
    mesh.add_argument("--gmsh-verbosity", type=int, default=0)
    mesh.add_argument("--gmsh-algorithm", type=int, default=None)

    design = parser.add_argument_group("torsion design and activity window")
    design.add_argument("--alpha-t1", type=float, default=0.60)
    design.add_argument("--alpha-t2", type=float, default=0.70)
    design.add_argument("--rho-amp", type=float, default=1.0)
    design.add_argument("--eps-ratio", type=float, default=0.08, help="r_eps in epsilon=r_eps*d")

    newton = parser.add_argument_group("strict damped Newton projection")
    newton.add_argument("--newton-tol", type=float, default=1.0e-10, help="single dual-residual tolerance used by every production projection")
    newton.add_argument("--max-newton-it", type=int, default=70)
    newton.add_argument(
        "--trial-max-newton-it",
        type=int,
        default=60,
        help=(
            "hard outer-trial iteration cap; capped trials are rejected unless "
            "they already meet newton-tol"
        ),
    )
    newton.add_argument("--tol-step", type=float, default=1.0e-14)
    newton.add_argument("--beta-ls", type=float, default=0.5)
    newton.add_argument("--armijo-c", type=float, default=1.0e-6)
    newton.add_argument("--alpha-min", type=float, default=1.0e-8)
    newton.add_argument("--max-backtrack", type=int, default=35)

    linear = parser.add_argument_group("PETSc linear solvers")
    linear.add_argument("--linear-solver", choices=("mumps", "lu", "hypre", "gamg"), default="mumps")
    linear.add_argument("--ksp-type", default=None)
    linear.add_argument("--linear-rtol", type=float, default=1.0e-11)
    linear.add_argument("--linear-atol", type=float, default=1.0e-13)
    linear.add_argument("--linear-max-it", type=int, default=1000)
    linear.add_argument("--iterative-fallback-solver", choices=("none", "mumps", "lu"), default="none")

    initializer = parser.add_argument_group("three-stage frozen initializer")
    initializer.add_argument("--init-leakage", type=float, default=None, help="frozen leakage budget; default 0.8*leakage-max")
    initializer.add_argument("--init-refine-max-it", type=int, default=80)
    initializer.add_argument("--init-refine-ftol", type=float, default=1.0e-12)
    initializer.add_argument(
        "--init-topology-repair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "search the frozen center-width box for a geometrically viable "
            "interval matching the discrete target component count"
        ),
    )
    initializer.add_argument(
        "--init-topology-repair-samples",
        type=int,
        default=24,
        help="samples per axis used by the frozen topology-repair scans",
    )
    initializer.add_argument("--init-shortlist", type=int, choices=range(1, 6), default=3)
    initializer.add_argument(
        "--init-candidate-max-newton-it",
        type=int,
        default=12,
        help=(
            "hard iteration cap for each direct shortlist projection; a capped "
            "candidate is rejected unless it already meets newton-tol"
        ),
    )
    initializer.add_argument("--init-width-perturbation", type=float, default=0.08)
    initializer.add_argument("--init-center-perturbation", type=float, default=0.03)
    initializer.add_argument("--init-shrink-factor", type=float, default=0.70)
    initializer.add_argument("--init-fallback-stages", type=int, default=12)
    initializer.add_argument(
        "--init-min-coverage",
        type=float,
        default=1.0e-3,
        help="minimum projected target coverage for a branch seed to be usable",
    )
    initializer.add_argument(
        "--init-homotopy-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use strict source continuation after direct shortlist projections fail",
    )
    initializer.add_argument("--init-homotopy-initial-step", type=float, default=0.10)
    initializer.add_argument("--init-homotopy-min-step", type=float, default=1.0e-3)
    initializer.add_argument("--init-homotopy-max-step", type=float, default=0.20)
    initializer.add_argument("--init-homotopy-step-shrink", type=float, default=0.5)
    initializer.add_argument("--init-homotopy-step-grow", type=float, default=1.5)
    initializer.add_argument(
        "--init-homotopy-easy-newton-it",
        type=int,
        default=6,
        help=(
            "largest strict corrector iteration count that permits growth of "
            "the next source-homotopy step; this never relaxes newton-tol"
        ),
    )
    initializer.add_argument("--init-homotopy-max-stages", type=int, default=64)
    initializer.add_argument(
        "--init-homotopy-predictor",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    initializer.add_argument(
        "--init-homotopy-threshold-steps-per-stage",
        type=int,
        default=3,
        help=(
            "maximum accepted center-width micro-SQP steps after every strict "
            "nonzero source-homotopy stage; the first is always attempted and "
            "later steps are used only while geometry is infeasible"
        ),
    )
    initializer.add_argument(
        "--init-homotopy-threshold-max-trials",
        type=int,
        default=4,
        help="strict trial projections allowed for each homotopy threshold micro-step",
    )
    initializer.add_argument(
        "--init-homotopy-threshold-trust-radius",
        type=float,
        default=0.05,
        help="initial scaled infinity trust radius for homotopy threshold updates",
    )
    initializer.add_argument(
        "--init-homotopy-threshold-trust-min",
        type=float,
        default=1.0e-5,
    )
    initializer.add_argument(
        "--init-homotopy-threshold-trust-max",
        type=float,
        default=0.25,
    )
    initializer.add_argument(
        "--init-homotopy-threshold-edge-step-fraction",
        type=float,
        default=0.20,
        help=(
            "maximum motion of either c1 or c2 in one homotopy threshold "
            "micro-step, expressed as a fraction of the current width"
        ),
    )
    initializer.add_argument(
        "--init-homotopy-threshold-rescue-attempts",
        type=int,
        default=1,
        help=(
            "threshold restorations attempted at the last exact state before "
            "a failed source increment is shortened"
        ),
    )
    initializer.add_argument(
        "--init-homotopy-threshold-stagnation-atol",
        type=float,
        default=1.0e-6,
        help=(
            "absolute minimum meaningful decrease of the active homotopy "
            "threshold merit (geometric violation while infeasible, scaled "
            "projection objective once feasible)"
        ),
    )
    initializer.add_argument(
        "--init-homotopy-threshold-stagnation-rtol",
        type=float,
        default=1.0e-2,
        help=(
            "relative minimum meaningful decrease of the active homotopy "
            "threshold merit; stagnation freezes thresholds at that lambda"
        ),
    )

    outer = parser.add_argument_group("two-dimensional trust-region SQP/filter")
    outer.add_argument("--max-opt-it", type=int, default=35)
    outer.add_argument("--max-trials-per-iteration", type=int, default=10)
    outer.add_argument("--trust-radius", type=float, default=0.25, help="initial scaled infinity radius")
    outer.add_argument("--trust-radius-min", type=float, default=1.0e-4)
    outer.add_argument("--trust-radius-max", type=float, default=1.0)
    outer.add_argument("--trust-shrink", type=float, default=0.5)
    outer.add_argument("--trust-grow", type=float, default=2.0)
    outer.add_argument("--acceptance-eta", type=float, default=0.10)
    outer.add_argument("--acceptance-grow-eta", type=float, default=0.75)
    outer.add_argument("--gn-regularization", type=float, default=1.0e-10)
    outer.add_argument("--kkt-tol", type=float, default=1.0e-6)
    outer.add_argument("--active-tol", type=float, default=1.0e-7)
    outer.add_argument("--constraint-tol", type=float, default=1.0e-9)
    outer.add_argument("--restoration-fraction", type=float, default=1.0e-3)
    outer.add_argument("--filter-objective-margin", type=float, default=1.0e-4)
    outer.add_argument("--filter-violation-margin", type=float, default=1.0e-2)
    outer.add_argument(
        "--functional-stagnation-atol",
        type=float,
        default=1.0e-6,
        help=(
            "absolute minimum meaningful decrease of the active outer merit "
            "(geometric violation while infeasible, H01 projection objective "
            "once feasible)"
        ),
    )
    outer.add_argument(
        "--functional-stagnation-rtol",
        type=float,
        default=1.0e-2,
        help="relative minimum meaningful decrease of the active outer merit",
    )
    outer.add_argument(
        "--functional-stagnation-patience",
        type=int,
        default=2,
        help="consecutive accepted stagnant outer steps required for termination",
    )

    branch = parser.add_argument_group("local branch continuity")
    branch.add_argument(
        "--branch-overlap-min",
        type=float,
        default=0.80,
        help=(
            "minimum symmetric soft-Dice similarity between consecutive "
            "nonlinear activity fields"
        ),
    )
    branch.add_argument(
        "--branch-topology-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "require the significant connected-component count of W to match "
            "the discrete torsion target"
        ),
    )
    branch.add_argument(
        "--branch-topology-expected-components",
        type=int,
        default=None,
        help="override the component count inferred from the discrete target",
    )
    branch.add_argument(
        "--branch-topology-core-level",
        type=float,
        default=0.50,
        help="activity level defining component cores",
    )
    branch.add_argument(
        "--branch-topology-bridge-level",
        type=float,
        default=0.25,
        help="minimum activity allowed along a connection between core regions",
    )
    branch.add_argument(
        "--branch-topology-min-component-fraction",
        type=float,
        default=0.02,
        help="ignore components carrying less than this fraction of core cells",
    )
    branch.add_argument("--predictor-correction-absolute", type=float, default=0.05, help="absolute H1/sqrt(E_T) correction allowance")
    branch.add_argument("--predictor-correction-factor", type=float, default=2.0)
    branch.add_argument("--predictor-max-change", type=float, default=5.0)

    spectral = parser.add_argument_group("optional Jacobian coercivity guard")
    spectral.add_argument("--coercivity-min", type=float, default=None, help="reject unless mu_min-eigen_error meets this bound")
    spectral.add_argument("--report-coercivity", action=argparse.BooleanOptionalAction, default=False)
    spectral.add_argument("--coercivity-eig-tol", type=float, default=1.0e-8)
    spectral.add_argument("--coercivity-eig-max-it", type=int, default=200)
    spectral.add_argument("--coercivity-zero-tol", type=float, default=1.0e-8)
    spectral.add_argument("--coercivity-inertia", action=argparse.BooleanOptionalAction, default=False)

    verification = parser.add_argument_group("verification diagnostics")
    verification.add_argument("--verify-window-derivatives", action=argparse.BooleanOptionalAction, default=True)
    verification.add_argument("--verify-reduced-derivatives", action=argparse.BooleanOptionalAction, default=False)
    verification.add_argument("--fd-relative-step", type=float, default=2.0e-6)
    verification.add_argument("--fd-signal-factor", type=float, default=20.0)

    plotting = parser.add_argument_group("interactive MPI PyVista plotting")
    plotting.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show full-domain interactive PyVista plots gathered on rank zero",
    )
    plotting.add_argument(
        "--plot-mode",
        choices=("blocking", "nonblocking"),
        default="blocking",
        help="use one live nonblocking window or pause for each plotted state",
    )
    plotting.add_argument("--plot-off-screen", action=argparse.BooleanOptionalAction, default=False)
    plotting.add_argument("--plot-window-width", type=int, default=1800)
    plotting.add_argument("--plot-window-height", type=int, default=900)
    plotting.add_argument("--plot-mesh-edges", action=argparse.BooleanOptionalAction, default=True)
    plotting.add_argument("--plot-design", action=argparse.BooleanOptionalAction, default=True)
    plotting.add_argument("--plot-design-hold-seconds", type=float, default=2.0)
    plotting.add_argument(
        "--plot-fields",
        choices=("full", "state", "density"),
        default="state",
        help="select the visible live-window panel set",
    )
    plotting.add_argument(
        "--plot-initial-candidates",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    plotting.add_argument(
        "--plot-homotopy-stages",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="update the live window after every accepted source-homotopy stage",
    )
    plotting.add_argument(
        "--plot-accepted-states",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    plotting.add_argument("--plot-final", action=argparse.BooleanOptionalAction, default=True)
    plotting.add_argument(
        "--plot-final-blocking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pause at the final plot for interactive inspection even in nonblocking mode",
    )

    output = parser.add_argument_group("output and policy")
    output.add_argument("--run-dir", type=Path, default=None)
    output.add_argument("--run-tag", default="")
    output.add_argument("--write-xdmf", action=argparse.BooleanOptionalAction, default=True)
    output.add_argument("--make-plots", action=argparse.BooleanOptionalAction, default=True)
    output.add_argument(
        "--save-terminal-log",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="mirror live stdout/stderr from every MPI rank to out/terminal.log",
    )
    output.add_argument("--fail-on-nonconvergence", action=argparse.BooleanOptionalAction, default=False)
    output.add_argument("-v", "--verbosity", action="count", default=1)
    return parser


def validate_args(args: argparse.Namespace) -> controls.CenterWidthBounds:
    """Validate mathematical, solver, and globalization options."""
    bounds = controls.CenterWidthBounds(
        args.center_min, args.center_max, args.width_min, args.width_max
    )
    positive = {
        "mesh-size": args.mesh_size,
        "rho-amp": args.rho_amp,
        "eps-ratio": args.eps_ratio,
        "newton-tol": args.newton_tol,
        "max-newton-it": args.max_newton_it,
        "linear-rtol": args.linear_rtol,
        "linear-atol": args.linear_atol,
        "linear-max-it": args.linear_max_it,
        "trust-radius": args.trust_radius,
        "trust-radius-min": args.trust_radius_min,
        "trust-radius-max": args.trust_radius_max,
        "gn-regularization": args.gn_regularization,
        "kkt-tol": args.kkt_tol,
    }
    for name, value in positive.items():
        if float(value) <= 0.0 or not math.isfinite(float(value)):
            raise ValueError(f"--{name} must be finite and positive")
    if not (0.0 < args.alpha_t1 < args.alpha_t2 < 1.0):
        raise ValueError("require 0 < alpha-t1 < alpha-t2 < 1")
    if args.leakage_max < 0.0 or args.missing_max < 0.0:
        raise ValueError("geometric bounds must be nonnegative")
    if not math.isfinite(args.leakage_max) or not math.isfinite(args.missing_max):
        raise ValueError("geometric bounds must be finite")
    if args.init_leakage is None:
        args.init_leakage = 0.8 * float(args.leakage_max)
    if not (
        0.0 <= args.init_leakage < args.leakage_max
        or args.init_leakage == args.leakage_max == 0.0
    ):
        raise ValueError("--init-leakage must satisfy 0 <= value < leakage-max")
    if not (0.0 < args.trust_shrink < 1.0) or args.trust_grow <= 1.0:
        raise ValueError("require 0<trust-shrink<1 and trust-grow>1")
    if not (
        0.0 <= args.acceptance_eta < args.acceptance_grow_eta <= 1.0
    ):
        raise ValueError("require 0 <= acceptance-eta < acceptance-grow-eta <= 1")
    if not (args.trust_radius_min <= args.trust_radius <= args.trust_radius_max):
        raise ValueError("initial trust radius must lie between its minimum and maximum")
    if not (0.0 < args.beta_ls < 1.0) or args.max_backtrack < 0:
        raise ValueError("invalid Newton line-search controls")
    if args.tol_step <= 0.0 or not (0.0 < args.alpha_min <= 1.0):
        raise ValueError("require positive tol-step and 0 < alpha-min <= 1")
    if not (0.0 < args.armijo_c < 1.0):
        raise ValueError("--armijo-c must lie in (0,1)")
    if not (0.0 < args.init_shrink_factor < 1.0):
        raise ValueError("--init-shrink-factor must lie in (0,1)")
    if args.init_refine_max_it < 1 or args.init_refine_ftol <= 0.0:
        raise ValueError("invalid frozen-refinement tolerances")
    if args.init_topology_repair_samples < 2:
        raise ValueError("--init-topology-repair-samples must be at least two")
    if args.init_candidate_max_newton_it < 1:
        raise ValueError("--init-candidate-max-newton-it must be positive")
    if args.trial_max_newton_it < 1:
        raise ValueError("--trial-max-newton-it must be positive")
    if not (0.0 <= args.init_width_perturbation < 1.0):
        raise ValueError("--init-width-perturbation must lie in [0,1)")
    if args.init_center_perturbation < 0.0 or args.init_fallback_stages < 1:
        raise ValueError("invalid initialization perturbation or fallback stage count")
    if not (0.0 < args.init_min_coverage <= 1.0):
        raise ValueError("--init-min-coverage must lie in (0,1]")
    if args.init_homotopy_max_stages < 1:
        raise ValueError("--init-homotopy-max-stages must be positive")
    if args.init_homotopy_threshold_steps_per_stage < 1:
        raise ValueError("--init-homotopy-threshold-steps-per-stage must be positive")
    if args.init_homotopy_threshold_max_trials < 1:
        raise ValueError("--init-homotopy-threshold-max-trials must be positive")
    if args.init_homotopy_threshold_rescue_attempts < 0:
        raise ValueError("--init-homotopy-threshold-rescue-attempts must be nonnegative")
    if (
        args.init_homotopy_threshold_stagnation_atol < 0.0
        or args.init_homotopy_threshold_stagnation_rtol < 0.0
        or not math.isfinite(args.init_homotopy_threshold_stagnation_atol)
        or not math.isfinite(args.init_homotopy_threshold_stagnation_rtol)
        or (
            args.init_homotopy_threshold_stagnation_atol == 0.0
            and args.init_homotopy_threshold_stagnation_rtol == 0.0
        )
    ):
        raise ValueError(
            "homotopy threshold stagnation tolerances must be finite, "
            "nonnegative, and not both zero"
        )
    if not (0.0 < args.init_homotopy_threshold_edge_step_fraction <= 1.0):
        raise ValueError(
            "--init-homotopy-threshold-edge-step-fraction must lie in (0,1]"
        )
    if not (
        0.0 < args.init_homotopy_threshold_trust_min
        <= args.init_homotopy_threshold_trust_radius
        <= args.init_homotopy_threshold_trust_max
    ):
        raise ValueError(
            "require 0 < init-homotopy-threshold-trust-min <= "
            "init-homotopy-threshold-trust-radius <= "
            "init-homotopy-threshold-trust-max"
        )
    if any(
        value <= 0.0 or not math.isfinite(float(value))
        for value in (
            args.init_homotopy_initial_step,
            args.init_homotopy_min_step,
            args.init_homotopy_max_step,
        )
    ):
        raise ValueError("homotopy fallback step sizes must be finite and positive")
    if args.init_homotopy_min_step > args.init_homotopy_max_step:
        raise ValueError("homotopy minimum step must not exceed its maximum")
    if not (0.0 < args.init_homotopy_step_shrink < 1.0):
        raise ValueError("--init-homotopy-step-shrink must lie in (0,1)")
    if args.init_homotopy_step_grow <= 1.0:
        raise ValueError("--init-homotopy-step-grow must exceed one")
    if args.init_homotopy_easy_newton_it < 0:
        raise ValueError("--init-homotopy-easy-newton-it must be nonnegative")
    if args.order < 1 or args.max_opt_it < 0 or args.max_trials_per_iteration < 1:
        raise ValueError("invalid discretization or iteration count")
    if args.quad_degree is not None and args.quad_degree < 1:
        raise ValueError("--quad-degree must be positive")
    if args.constraint_tol < 0.0 or args.active_tol < 0.0:
        raise ValueError("constraint and active-set tolerances must be nonnegative")
    if not (0.0 < args.restoration_fraction <= 1.0):
        raise ValueError("--restoration-fraction must lie in (0,1]")
    if args.filter_objective_margin < 0.0 or not (0.0 < args.filter_violation_margin < 1.0):
        raise ValueError("invalid filter margins")
    if (
        args.functional_stagnation_atol < 0.0
        or args.functional_stagnation_rtol < 0.0
        or not math.isfinite(args.functional_stagnation_atol)
        or not math.isfinite(args.functional_stagnation_rtol)
        or (
            args.functional_stagnation_atol == 0.0
            and args.functional_stagnation_rtol == 0.0
        )
        or args.functional_stagnation_patience < 1
    ):
        raise ValueError(
            "outer functional stagnation tolerances must be finite, "
            "nonnegative, not both zero, and patience must be positive"
        )
    if not (0.0 <= args.branch_overlap_min <= 1.0):
        raise ValueError("--branch-overlap-min must lie in [0,1]")
    if (
        args.branch_topology_expected_components is not None
        and args.branch_topology_expected_components < 1
    ):
        raise ValueError("--branch-topology-expected-components must be positive")
    if not (
        0.0
        <= args.branch_topology_bridge_level
        <= args.branch_topology_core_level
        <= 1.0
    ):
        raise ValueError(
            "require 0 <= branch-topology-bridge-level <= "
            "branch-topology-core-level <= 1"
        )
    if not (0.0 <= args.branch_topology_min_component_fraction < 1.0):
        raise ValueError(
            "--branch-topology-min-component-fraction must lie in [0,1)"
        )
    if args.predictor_correction_absolute < 0.0 or args.predictor_correction_factor < 0.0:
        raise ValueError("predictor-correction guards must be nonnegative")
    if args.predictor_max_change <= 0.0:
        raise ValueError("--predictor-max-change must be positive")
    if args.fd_relative_step <= 0.0 or args.fd_signal_factor <= 0.0:
        raise ValueError("finite-difference controls must be positive")
    if args.plot_window_width < 1 or args.plot_window_height < 1:
        raise ValueError("interactive plot dimensions must be positive")
    if (
        not math.isfinite(float(args.plot_design_hold_seconds))
        or args.plot_design_hold_seconds < 0.0
    ):
        raise ValueError("--plot-design-hold-seconds must be finite and nonnegative")
    if args.coercivity_min is not None:
        args.report_coercivity = True
    if args.coercivity_eig_tol <= 0.0 or args.coercivity_eig_max_it < 1:
        raise ValueError("invalid coercivity eigensolver controls")
    if args.mesh is not None:
        args.mesh = args.mesh.expanduser().resolve()
        if not args.mesh.is_file():
            raise ValueError(f"mesh file does not exist: {args.mesh}")
    if args.star_r0 <= abs(args.star_amp):
        raise ValueError("smooth-star generation requires star-r0 > abs(star-amp)")
    return bounds


def configure_shared_newton_args(args: argparse.Namespace, trace_writer: NewtonTraceWriter) -> argparse.Namespace:
    """Create the strict fixed-tolerance namespace required by shared Newton."""
    configured = argparse.Namespace(**vars(args))
    configured.residual_norm = "dual"
    configured.tol_res = float(args.newton_tol)
    configured.newton_soft_cap = False
    configured.newton_soft_cap_factor = 1.0
    configured.newton_soft_cap_chunk = 1
    configured.newton_soft_cap_window = 2
    configured.newton_soft_cap_contraction = 0.95
    configured.newton_hard_cap_reason = "strict_h1_projection"
    configured.reject_on_linear_max_it = True
    configured.reject_on_linear_failure = True
    configured.diagnose_capped_trial_coercivity = False
    configured.check_newton_spd = False
    configured.newton_spd_prefix_filter = ""
    configured.newton_spd_every = 1
    configured.newton_spd_eig_tol = float(args.coercivity_eig_tol)
    configured.newton_spd_eig_max_it = int(args.coercivity_eig_max_it)
    configured.newton_spd_zero_tol = float(args.coercivity_zero_tol)
    configured.newton_spd_inertia = bool(args.coercivity_inertia)
    configured.newton_spd_records = []
    configured.newton_writer = trace_writer
    configured.newton_handle = trace_writer.handle
    # Source continuation is an initialization globalization only.  Every
    # accepted continuation state uses the same strict residual tolerance as
    # production projections; there is no adaptive/inexact Newton tolerance.
    configured.homotopy_tol_res = float(args.newton_tol)
    configured.threshold_optimization_start_lambda = 1.0
    configured.homotopy_initial_step = float(args.init_homotopy_initial_step)
    configured.homotopy_min_step = float(args.init_homotopy_min_step)
    configured.homotopy_max_step = float(args.init_homotopy_max_step)
    configured.homotopy_step_shrink = float(args.init_homotopy_step_shrink)
    configured.homotopy_step_grow = float(args.init_homotopy_step_grow)
    configured.homotopy_easy_newton_iterations = int(
        args.init_homotopy_easy_newton_it
    )
    configured.homotopy_max_stages = int(args.init_homotopy_max_stages)
    configured.homotopy_predictor = bool(args.init_homotopy_predictor)
    configured.verify_homotopy_init = False
    return configured


class ProjectionProblem:
    """Fixed-mesh distributed PDE and reduced-functional workspace."""

    def __init__(
        self,
        *,
        domain,
        function_space,
        bc,
        trial,
        test,
        dx,
        stiffness_form,
        stiffness_solver: shared.FixedStiffnessSolver,
        phi_target: fem.Function,
        target_mask,
        target_area: float,
        target_energy: float,
        topology_analyzer: DistributedActivityTopology,
        expected_topology_components: int,
        rho_amp: float,
        eps_ratio: float,
        args: argparse.Namespace,
        newton_args: argparse.Namespace,
        trace_writer: NewtonTraceWriter,
    ) -> None:
        """Bind fixed finite-element data and allocate reusable work objects."""
        self.domain = domain
        self.comm = domain.comm
        self.V = function_space
        self.bc = bc
        self.trial = trial
        self.test = test
        self.dx = dx
        self.stiffness_form = stiffness_form
        self.stiffness = stiffness_solver
        self.phi_target = phi_target
        self.target_mask = target_mask
        self.target_area = float(target_area)
        self.target_energy = float(target_energy)
        self.topology_analyzer = topology_analyzer
        self.expected_topology_components = int(expected_topology_components)
        self.rho_amp = float(rho_amp)
        self.eps_ratio = float(eps_ratio)
        self.args = args
        self.newton_args = newton_args
        self.trace_writer = trace_writer
        self.c1_const = fem.Constant(domain, PETSc.ScalarType(0.0))
        self.c2_const = fem.Constant(domain, PETSc.ScalarType(1.0))
        self.eps_const = fem.Constant(domain, PETSc.ScalarType(1.0))
        self.work = fem.Function(function_space, name="h1Work")
        self.branch_trial_activity = fem.Function(
            function_space, name="branchTrialActivity"
        )
        self.matvec = self.work.x.petsc_vec.duplicate()

    def close(self) -> None:
        """Release the additional PETSc work vector owned by this workspace."""
        self.matvec.destroy()

    def set_control(self, point: Sequence[float]) -> tuple[float, float, float]:
        """Update UFL constants and return ``(c1,c2,epsilon)``."""
        center, width = map(float, point)
        c1, c2 = controls.thresholds_from_center_width(center, width)
        epsilon = self.eps_ratio * width
        self.c1_const.value = PETSc.ScalarType(c1)
        self.c2_const.value = PETSc.ScalarType(c2)
        self.eps_const.value = PETSc.ScalarType(epsilon)
        return c1, c2, epsilon

    def activity(self, state: fem.Function):
        """Return the unscaled activity UFL expression at current constants."""
        return shared.window_activity_const_ufl(
            state, self.c1_const, self.c2_const, self.eps_const
        )

    def residual_expression(self, state: fem.Function, homotopy_lambda: float = 1.0):
        """Return the source-homotopy residual at fixed controls."""
        lam = float(homotopy_lambda)
        source = self.rho_amp * (
            (1.0 - lam) * self.target_mask + lam * self.activity(state)
        )
        return (
            ufl.inner(ufl.grad(state), ufl.grad(self.test))
            - source * self.test
        ) * self.dx

    def dual_residual(self, state: fem.Function, homotopy_lambda: float = 1.0) -> float:
        """Evaluate ``sqrt(R^T K^-1 R)`` with the cached stiffness solve."""
        return self.stiffness.residual_norm(
            fem.form(self.residual_expression(state, homotopy_lambda)), "dual"
        )

    def h1_inner(self, first: fem.Function, second: fem.Function) -> float:
        """Compute the stiffness inner product of two finite-element fields."""
        self.stiffness.matrix.mult(second.x.petsc_vec, self.matvec)
        return float(first.x.petsc_vec.dot(self.matvec))

    def h1_distance(self, first: fem.Function, second: fem.Function) -> float:
        """Compute the basis-independent H0-1 distance between two fields."""
        self.work.x.array[:] = first.x.array - second.x.array
        self.work.x.scatter_forward()
        return self.stiffness.h1_seminorm(self.work)

    def set_predictor(
        self,
        target: fem.Function,
        accepted: fem.Function,
        sensitivity_center: fem.Function,
        sensitivity_width: fem.Function,
        step: np.ndarray,
    ) -> tuple[bool, float]:
        """Form the first-order state predictor, falling back on invalid data."""
        target.x.array[:] = (
            accepted.x.array
            + float(step[0]) * sensitivity_center.x.array
            + float(step[1]) * sensitivity_width.x.array
        )
        local_finite = int(np.all(np.isfinite(target.x.array)))
        globally_finite = bool(self.comm.allreduce(local_finite, op=MPI.MIN))
        if globally_finite:
            target.x.scatter_forward()
            predicted_change = self.h1_distance(target, accepted)
            normalized = predicted_change / math.sqrt(self.target_energy)
            globally_finite = math.isfinite(normalized) and normalized <= float(
                self.args.predictor_max_change
            )
        else:
            predicted_change = math.inf
        if not globally_finite:
            _copy_function(accepted, target)
            return False, 0.0
        return True, predicted_change

    def evaluate_metrics(self, state: fem.Function) -> ProjectionMetrics:
        """Assemble the H1 objective and normalized soft geometry."""
        activity = self.activity(state)
        leakage = shared.assemble_scalar(
            self.comm, (1.0 - self.target_mask) * activity * self.dx
        ) / self.target_area
        missing = shared.assemble_scalar(
            self.comm, self.target_mask * (1.0 - activity) * self.dx
        ) / self.target_area
        activity_area = shared.assemble_scalar(self.comm, activity * self.dx) / self.target_area
        distance = self.h1_distance(state, self.phi_target)
        objective = 0.5 * distance * distance / self.target_energy
        overlap = controls.soft_overlap_diagnostics(leakage, missing)
        # Assemble A_W explicitly and keep the identity A_W/A_T=1-M+L as a
        # consistency check rather than silently substituting it.
        identity_area = overlap["activity_area_ratio"]
        if abs(activity_area - identity_area) > 1.0e-8 * max(1.0, abs(activity_area)):
            raise RuntimeError(
                "activity-area identity failed: "
                f"assembled={activity_area:.16e}, identity={identity_area:.16e}"
            )
        return ProjectionMetrics(
            objective=objective,
            leakage=leakage,
            missing=missing,
            activity_area_ratio=activity_area,
            overlap_area=self.target_area * overlap["overlap_area_ratio"],
            overlap_area_ratio=overlap["overlap_area_ratio"],
            recall=overlap["recall"],
            precision=overlap["precision"],
            jaccard=overlap["jaccard"],
        )

    def update_density(self, state: fem.Function, density: fem.Function) -> None:
        """Interpolate the scaled semilinear density into an output function."""
        shared.update_interpolated(density, self.rho_amp * self.activity(state))

    def update_activity(self, state: fem.Function, activity: fem.Function) -> None:
        """Interpolate the unscaled activity for branch-overlap comparisons."""
        shared.update_interpolated(activity, self.activity(state))

    def activity_topology(self, activity: fem.Function) -> controls.ActivityTopology:
        """Evaluate the configured hysteretic component diagnostic."""
        return self.topology_analyzer.analyze(
            activity,
            core_level=float(self.args.branch_topology_core_level),
            bridge_level=float(self.args.branch_topology_bridge_level),
            min_component_fraction=float(
                self.args.branch_topology_min_component_fraction
            ),
        )

    def topology_rejection_reason(
        self, topology: controls.ActivityTopology | None
    ) -> str:
        """Return the topology safeguard reason, or an empty string."""
        if not bool(self.args.branch_topology_guard):
            return ""
        if topology is None:
            return "TOPOLOGY_UNAVAILABLE"
        if topology.component_count != self.expected_topology_components:
            return (
                f"TOPOLOGY_COMPONENTS:{topology.component_count}"
                f"!={self.expected_topology_components}"
            )
        return ""

    def coercivity(
        self,
        state: fem.Function,
        prefix: str,
        homotopy_lambda: float = 1.0,
    ) -> tuple[float, float, str]:
        """Return conservative generalized-Jacobian coercivity diagnostics."""
        if not bool(self.args.report_coercivity):
            return math.nan, math.nan, "DISABLED"
        ws = shared.window_s_derivative_activity_ufl(
            state, self.c1_const, self.c2_const, self.eps_const
        )
        jacobian = (
            ufl.inner(ufl.grad(self.trial), ufl.grad(self.test))
            - float(homotopy_lambda) * self.rho_amp * ws * self.trial * self.test
        ) * self.dx
        before = len(self.newton_args.newton_spd_records)
        try:
            shared.diagnose_newton_spd(
                jacobian,
                [self.bc],
                self.stiffness.matrix,
                args=self.newton_args,
                prefix=prefix,
                stage="strict_projected_state",
                newton_iteration=-1,
                residual=self.dual_residual(state, homotopy_lambda),
                homotopy_lambda=float(homotopy_lambda),
                c1=float(self.c1_const.value),
                c2=float(self.c2_const.value),
                eps_phi=float(self.eps_const.value),
            )
        except Exception as error:  # SLEPc is an optional diagnostic dependency
            return math.nan, math.nan, f"ERROR:{type(error).__name__}:{error}"
        if len(self.newton_args.newton_spd_records) == before:
            return math.nan, math.nan, "NO_RECORD"
        record = self.newton_args.newton_spd_records[-1]
        mu = float(record.get("muMin", math.nan))
        error = float(record.get("eigenError", math.nan))
        margin = mu - error if math.isfinite(mu) and math.isfinite(error) else math.nan
        return margin, error, str(record.get("status", "UNKNOWN"))

    def project(
        self,
        *,
        state: fem.Function,
        correction: fem.Function,
        density: fem.Function,
        point: Sequence[float],
        prefix: str,
        predictor: fem.Function | None = None,
        accepted_state: fem.Function | None = None,
        reference_activity: fem.Function | None = None,
        phase: str,
        outer_iteration: int,
        max_newton_it: int | None = None,
        homotopy_lambda: float = 1.0,
        homotopy_workspace: shared.HomotopySolveWorkspace | None = None,
    ) -> ProjectionResult:
        """Run one strict damped-Newton projection and collect all diagnostics."""
        c1, c2, epsilon = self.set_control(point)
        if homotopy_workspace is not None:
            homotopy_workspace.set_parameters(
                lam=float(homotopy_lambda), c1=c1, c2=c2, eps_phi=epsilon
            )
            initial_residual = self.stiffness.residual_norm(
                homotopy_workspace.residual_form, "dual"
            )
        else:
            initial_residual = self.dual_residual(state, homotopy_lambda)
        self.trace_writer.clear(prefix)
        full_newton_limit = int(self.newton_args.max_newton_it)
        if max_newton_it is not None:
            self.newton_args.max_newton_it = min(full_newton_limit, int(max_newton_it))
        try:
            newton = shared.solve_equilibrium(
                u=state,
                du=correction,
                rho=density,
                trial=self.trial,
                test=self.test,
                dx=self.dx,
                bc=self.bc,
                stiffness_form=self.stiffness_form,
                c1_const=self.c1_const,
                c2_const=self.c2_const,
                eps_const=self.eps_const,
                c1=c1,
                c2=c2,
                eps_phi=epsilon,
                rho_amp=self.rho_amp,
                tol_res=float(self.args.newton_tol),
                args=self.newton_args,
                prefix=prefix,
                stiffness_solver=self.stiffness,
                initial_residual=initial_residual,
                phase=phase,
                outer_iteration=outer_iteration,
                homotopy_lambda=float(homotopy_lambda),
                homotopy_target_density=self.rho_amp * self.target_mask,
                homotopy_workspace=homotopy_workspace,
            )
        finally:
            self.newton_args.max_newton_it = full_newton_limit
        trace = self.trace_writer.summarize(prefix)
        metrics: ProjectionMetrics | None = None
        predictor_corrector = math.nan
        predicted_change = math.nan
        branch_overlap = math.nan
        topology = None
        margin = math.nan
        eigen_error = math.nan
        reason = ""
        strict_success = bool(
            newton.converged
            and math.isfinite(newton.residual)
            and newton.residual <= float(self.args.newton_tol)
        )
        if not strict_success:
            reason = f"STRICT_NEWTON:{newton.status}"
        else:
            self.set_control(point)
            metrics = self.evaluate_metrics(state)
            if predictor is not None:
                predictor_corrector = self.h1_distance(state, predictor)
            if accepted_state is not None and predictor is not None:
                predicted_change = self.h1_distance(predictor, accepted_state)
            self.update_activity(state, self.branch_trial_activity)
            if reference_activity is not None:
                branch_overlap = shared.activity_dice_ratio(
                    comm=self.comm,
                    activity_ref=reference_activity,
                    activity_trial=self.branch_trial_activity,
                    dx=self.dx,
                )
            topology = self.activity_topology(self.branch_trial_activity)
            topology_reason = self.topology_rejection_reason(topology)
            if topology_reason:
                reason = topology_reason
            margin, eigen_error, spectral_status = self.coercivity(
                state,
                f"{prefix}_coercivity",
                homotopy_lambda,
            )
            if self.args.coercivity_min is not None and not reason:
                if not math.isfinite(margin):
                    reason = f"COERCIVITY_UNAVAILABLE:{spectral_status}"
                elif margin < float(self.args.coercivity_min):
                    reason = f"COERCIVITY_MARGIN:{margin:.6e}"
        if int(self.args.verbosity) >= 3:
            topology_guard = (
                "NOT_EVALUATED"
                if topology is None
                else (
                    "PASS"
                    if topology.component_count == self.expected_topology_components
                    else "REJECT"
                )
            )
            if self.args.coercivity_min is None:
                coercivity_guard = "DISABLED"
            elif not math.isfinite(margin):
                coercivity_guard = "UNAVAILABLE"
            elif margin >= float(self.args.coercivity_min):
                coercivity_guard = "PASS"
            else:
                coercivity_guard = "REJECT"
            shared.root_print(
                self.comm,
                "STRICT_PROJECTION_GUARD "
                f"prefix={prefix} phase={phase} "
                f"lambda={float(homotopy_lambda):.6e} "
                f"m={float(point[0]):.12e} d={float(point[1]):.12e} "
                f"c1={c1:.12e} c2={c2:.12e} epsilon={epsilon:.12e} "
                f"newtonConvergenceGuard="
                f"{'PASS' if newton.converged else 'REJECT'} "
                f"dualResidualGuard="
                f"{'PASS' if math.isfinite(newton.residual) and newton.residual <= float(self.args.newton_tol) else 'REJECT'} "
                f"residual={newton.residual:.12e} "
                f"tolerance={float(self.args.newton_tol):.12e} "
                f"topologyGuard={topology_guard} "
                f"coercivityGuard={coercivity_guard} "
                f"reason={reason or 'NONE'} "
                f"action={'RETURN_REJECTED_PROJECTION' if reason else 'RETURN_STRICT_STATE_FOR_CALLER_GUARDS'}",
            )
        return ProjectionResult(
            newton=newton,
            trace=trace,
            initial_residual=initial_residual,
            metrics=metrics,
            predictor_corrector_h1=predictor_corrector,
            predicted_state_change_h1=predicted_change,
            branch_overlap=branch_overlap,
            topology=topology,
            coercivity_margin=margin,
            coercivity_error=eigen_error,
            rejection_reason=reason,
        )

    def sensitivities(
        self,
        *,
        state: fem.Function,
        sensitivity_center: fem.Function,
        sensitivity_width: fem.Function,
        point: Sequence[float],
        iteration: int,
        homotopy_lambda: float = 1.0,
        objective_scale: float = 1.0,
        homotopy_workspace: shared.HomotopySolveWorkspace | None = None,
    ) -> ReducedData:
        """Solve exact partial-homotopy sensitivities with one Jacobian.

        At source parameter ``lambda``, both explicit threshold right-hand
        sides carry the required factor ``lambda``.  ``objective_scale`` is
        used only by the alternating initializer to model ``J/lambda^2``;
        all stored physical metrics remain the unscaled H1 objective.
        """
        self.set_control(point)
        lam = float(homotopy_lambda)
        model_scale = float(objective_scale)
        if homotopy_workspace is not None:
            c1, c2, epsilon = self.set_control(point)
            homotopy_workspace.set_parameters(
                lam=lam, c1=c1, c2=c2, eps_phi=epsilon
            )
            lam_coefficient = homotopy_workspace.lambda_const
        else:
            lam_coefficient = lam
        ws = shared.window_s_derivative_activity_ufl(
            state, self.c1_const, self.c2_const, self.eps_const
        )
        dm_activity, dd_activity = shared.window_center_width_derivatives_activity_ufl(
            state,
            self.c1_const,
            self.c2_const,
            self.eps_const,
            eps_ratio=self.eps_ratio,
        )
        jacobian = (
            ufl.inner(ufl.grad(self.trial), ufl.grad(self.test))
            - lam_coefficient * self.rho_amp * ws * self.trial * self.test
        ) * self.dx
        rhs_center = lam_coefficient * self.rho_amp * dm_activity * self.test * self.dx
        rhs_width = lam_coefficient * self.rho_amp * dd_activity * self.test * self.dx
        solve = shared.solve_same_matrix_forms(
            jacobian,
            [rhs_center, rhs_width],
            [sensitivity_center, sensitivity_width],
            [self.bc],
            prefix=f"h1proj_sensitivity_{iteration}_",
            solver=self.args.linear_solver,
            ksp_type=self.args.ksp_type,
            rtol=self.args.linear_rtol,
            atol=self.args.linear_atol,
            max_it=self.args.linear_max_it,
            verbosity=self.args.verbosity,
        )
        iterations, residuals, elapsed = solve[0], solve[1], float(solve[2])
        error = state - self.phi_target
        sensitivities = (sensitivity_center, sensitivity_width)
        explicit = (dm_activity, dd_activity)
        grad_objective = model_scale * np.asarray(
            [
                shared.assemble_scalar(
                    self.comm,
                    ufl.inner(ufl.grad(error), ufl.grad(sensitivity)) * self.dx,
                )
                / self.target_energy
                for sensitivity in sensitivities
            ],
            dtype=np.float64,
        )
        grad_leakage = np.asarray(
            [
                shared.assemble_scalar(
                    self.comm,
                    (1.0 - self.target_mask)
                    * (ws * sensitivity + direct)
                    * self.dx,
                )
                / self.target_area
                for sensitivity, direct in zip(sensitivities, explicit, strict=True)
            ],
            dtype=np.float64,
        )
        grad_missing = -np.asarray(
            [
                shared.assemble_scalar(
                    self.comm,
                    self.target_mask * (ws * sensitivity + direct) * self.dx,
                )
                / self.target_area
                for sensitivity, direct in zip(sensitivities, explicit, strict=True)
            ],
            dtype=np.float64,
        )
        hessian = np.empty((2, 2), dtype=np.float64)
        for first in range(2):
            for second in range(first, 2):
                value = (
                    model_scale
                    * self.h1_inner(sensitivities[first], sensitivities[second])
                    / self.target_energy
                )
                hessian[first, second] = value
                hessian[second, first] = value
        hessian += float(self.args.gn_regularization) * np.eye(2)
        condition = float(np.linalg.cond(hessian))
        return ReducedData(
            gradient_objective=grad_objective,
            gradient_leakage=grad_leakage,
            gradient_missing=grad_missing,
            hessian=hessian,
            condition_number=condition,
            iterations=(int(iterations[0]), int(iterations[1])),
            residuals=(float(residuals[0]), float(residuals[1])),
            reasons=("PETSC_NONNEGATIVE_REASON", "PETSC_NONNEGATIVE_REASON"),
            solve_time=elapsed,
        )

    def kkt(
        self,
        point: np.ndarray,
        metrics: ProjectionMetrics,
        reduced: ReducedData,
        bounds: controls.CenterWidthBounds,
    ) -> tuple[controls.KKTResult, tuple[str, ...]]:
        """Compute scaled reduced KKT residual including geometry and boxes."""
        scale = bounds.scale
        signed = np.asarray(
            [
                metrics.leakage - float(self.args.leakage_max),
                metrics.missing - float(self.args.missing_max),
                (point[0] - bounds.center_max) / scale[0],
                (bounds.center_min - point[0]) / scale[0],
                (point[1] - bounds.width_max) / scale[1],
                (bounds.width_min - point[1]) / scale[1],
            ],
            dtype=np.float64,
        )
        gradients = np.asarray(
            [
                reduced.gradient_leakage,
                reduced.gradient_missing,
                [1.0 / scale[0], 0.0],
                [-1.0 / scale[0], 0.0],
                [0.0, 1.0 / scale[1]],
                [0.0, -1.0 / scale[1]],
            ],
            dtype=np.float64,
        )
        # Express stationarity in scaled controls q=z/scale so the KKT number
        # is dimensionless and not dominated by unlike m/d units.
        result = controls.reduced_kkt_residual(
            reduced.gradient_objective * scale,
            signed,
            gradients * scale[None, :],
            active_tolerance=float(self.args.active_tol),
        )
        names = ("leakage", "missing", "center_upper", "center_lower", "width_upper", "width_lower")
        return result, tuple(names[index] for index in result.active_rows)


def frozen_refinement(
    values: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    initial: controls.HardWindowSelection,
    bounds: controls.CenterWidthBounds,
    args: argparse.Namespace,
) -> tuple[np.ndarray, controls.FrozenWindowMetrics, dict[str, Any]]:
    """Run the small smooth frozen constrained refinement on rank zero."""
    cache: dict[tuple[float, float], controls.FrozenWindowMetrics] = {}
    feasible_evaluations: list[tuple[float, np.ndarray, controls.FrozenWindowMetrics]] = []

    def evaluate(point: Sequence[float]) -> controls.FrozenWindowMetrics:
        """Return a cached frozen metric and retain feasible evaluations."""
        clipped = bounds.clip(float(point[0]), float(point[1]))
        key = (float(clipped[0]).hex(), float(clipped[1]).hex())
        if key not in cache:
            metric = controls.frozen_window_metrics(
                values,
                target,
                weights,
                clipped[0],
                clipped[1],
                float(args.eps_ratio),
            )
            cache[key] = metric
            if metric.leakage <= float(args.init_leakage) + 1.0e-10:
                feasible_evaluations.append((metric.missing, np.asarray(clipped), metric))
        return cache[key]

    def objective(point: np.ndarray) -> float:
        """Return frozen normalized missing area."""
        return evaluate(point).missing

    def objective_jac(point: np.ndarray) -> np.ndarray:
        """Return the explicit frozen missing-area gradient."""
        return evaluate(point).gradient_missing

    def leakage_constraint(point: np.ndarray) -> float:
        """Return positive slack for the frozen leakage inequality."""
        return float(args.init_leakage) - evaluate(point).leakage

    def leakage_jac(point: np.ndarray) -> np.ndarray:
        """Return the gradient of the frozen leakage slack."""
        return -evaluate(point).gradient_leakage

    initial_point = np.asarray([initial.center, initial.width], dtype=np.float64)
    evaluate(initial_point)
    result = minimize(
        objective,
        initial_point,
        jac=objective_jac,
        method="SLSQP",
        bounds=list(zip(bounds.lower, bounds.upper, strict=True)),
        constraints=[{"type": "ineq", "fun": leakage_constraint, "jac": leakage_jac}],
        options={
            "maxiter": int(args.init_refine_max_it),
            "ftol": float(args.init_refine_ftol),
            "disp": False,
        },
    )
    evaluate(result.x)
    if not feasible_evaluations:
        raise RuntimeError(
            "smooth frozen refinement found no point satisfying the strict initialization leakage budget"
        )
    _, point, metric = min(
        feasible_evaluations,
        key=lambda item: (item[0], item[2].leakage, item[1][1], item[1][0]),
    )
    diagnostics = {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "evaluations": len(cache),
        "returned_best_feasible": not np.allclose(point, result.x, rtol=0.0, atol=1.0e-13),
    }
    return point, metric, diagnostics


def pointwise_window_derivative_check(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Write a centered-difference check of all pointwise window derivatives."""
    center = 0.5 * (float(args.center_min) + float(args.center_max))
    width = 0.5 * (float(args.width_min) + float(args.width_max))
    c1, c2 = controls.thresholds_from_center_width(center, width)
    values = np.asarray(
        [
            c1 - 1.5 * args.eps_ratio * width,
            c1,
            center,
            c2,
            c2 + 1.5 * args.eps_ratio * width,
        ],
        dtype=np.float64,
    )
    exact = controls.window_center_width(values, center, width, args.eps_ratio)
    scales = np.asarray([max(1.0, abs(center)), max(1.0, width), max(1.0, np.max(np.abs(values)))])
    h_phi = 2.0e-7 * scales[2]
    h_center = 2.0e-7 * scales[0]
    h_width = min(2.0e-7 * scales[1], 0.1 * width)

    def activity(sample_values, m, d):
        """Evaluate only the activity for centered differences."""
        return controls.window_center_width(sample_values, m, d, args.eps_ratio).activity

    finite_difference = {
        "state": (activity(values + h_phi, center, width) - activity(values - h_phi, center, width)) / (2.0 * h_phi),
        "center": (activity(values, center + h_center, width) - activity(values, center - h_center, width)) / (2.0 * h_center),
        "width": (activity(values, center, width + h_width) - activity(values, center, width - h_width)) / (2.0 * h_width),
    }
    exact_arrays = {
        "state": exact.state_derivative,
        "center": exact.center_derivative,
        "width": exact.width_derivative,
    }
    rows = []
    maximum = 0.0
    for derivative_name in ("state", "center", "width"):
        for index, value in enumerate(values):
            analytic = float(exact_arrays[derivative_name][index])
            fd = float(finite_difference[derivative_name][index])
            # Near symmetry points both derivatives are analytically zero and
            # centered subtraction leaves roundoff of order 1e-10.  A unit
            # floor makes this a standard relative-or-absolute scaled error
            # instead of reporting a meaningless O(1) relative error to zero.
            relative = abs(analytic - fd) / max(abs(analytic), abs(fd), 1.0)
            maximum = max(maximum, relative)
            rows.append(
                {
                    "derivative": derivative_name,
                    "sample": float(value),
                    "analytic": analytic,
                    "centered_difference": fd,
                    "relative_error": relative,
                }
            )
    payload = {
        "center": center,
        "width": width,
        "eps_ratio": float(args.eps_ratio),
        "maximum_relative_error": maximum,
        "rows": rows,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def append_outer_row(
    writer: csv.DictWriter | None,
    records: list[dict[str, Any]],
    **updates: Any,
) -> None:
    """Append a complete outer-log row, filling unavailable diagnostics."""
    row = {field: "" for field in OUTER_FIELDS}
    row.update(updates)
    if writer is not None:
        writer.writerow(row)
    records.append(row)


def metric_log_values(metrics: ProjectionMetrics | None, args: argparse.Namespace) -> dict[str, Any]:
    """Flatten projection metrics into the common outer-log schema."""
    if metrics is None:
        return {}
    return {
        "objective": metrics.objective,
        "relative_h1_distance": math.sqrt(max(2.0 * metrics.objective, 0.0)),
        "leakage": metrics.leakage,
        "missing": metrics.missing,
        "leakage_slack": float(args.leakage_max) - metrics.leakage,
        "missing_slack": float(args.missing_max) - metrics.missing,
        "coverage": 1.0 - metrics.missing,
        "activity_area_ratio": metrics.activity_area_ratio,
        "overlap_area": metrics.overlap_area,
        "overlap_area_ratio": metrics.overlap_area_ratio,
        "recall": metrics.recall,
        "precision": metrics.precision,
        "jaccard": metrics.jaccard,
        "geometric_violation": controls.normalized_violation(
            np.asarray(
                [metrics.leakage - float(args.leakage_max), metrics.missing - float(args.missing_max)]
            )
        ),
    }


def control_log_values(point: Sequence[float], eps_ratio: float) -> dict[str, float]:
    """Flatten ``m,d,c1,c2,epsilon`` into log columns."""
    center, width = map(float, point)
    c1, c2 = controls.thresholds_from_center_width(center, width)
    return {
        "center": center,
        "width": width,
        "c1": c1,
        "c2": c2,
        "epsilon": float(eps_ratio) * width,
    }


def reduced_log_values(reduced: ReducedData | None) -> dict[str, Any]:
    """Flatten reduced derivatives and Gauss--Newton data for CSV output."""
    if reduced is None:
        return {}
    return {
        "grad_objective_center": reduced.gradient_objective[0],
        "grad_objective_width": reduced.gradient_objective[1],
        "grad_leakage_center": reduced.gradient_leakage[0],
        "grad_leakage_width": reduced.gradient_leakage[1],
        "grad_missing_center": reduced.gradient_missing[0],
        "grad_missing_width": reduced.gradient_missing[1],
        "gn_00": reduced.hessian[0, 0],
        "gn_01": reduced.hessian[0, 1],
        "gn_11": reduced.hessian[1, 1],
        "gn_condition": reduced.condition_number,
        "sensitivity_iterations_center": reduced.iterations[0],
        "sensitivity_iterations_width": reduced.iterations[1],
        "sensitivity_residual_center": reduced.residuals[0],
        "sensitivity_residual_width": reduced.residuals[1],
        "sensitivity_reason_center": reduced.reasons[0],
        "sensitivity_reason_width": reduced.reasons[1],
        "sensitivity_solve_time": reduced.solve_time,
    }


def projection_log_values(result: ProjectionResult | None, target_energy: float) -> dict[str, Any]:
    """Flatten Newton, predictor, branch, and spectral projection diagnostics."""
    if result is None:
        return {}
    scale = math.sqrt(target_energy)
    topology = result.topology
    return {
        "newton_status": result.newton.status,
        "newton_iterations": result.newton.iterations,
        "newton_initial_residual": result.initial_residual,
        "newton_final_residual": result.newton.residual,
        "minimum_damping": result.trace.minimum_damping,
        "damping_history": _json_array(result.trace.damping_history),
        "predictor_corrector_h1": result.predictor_corrector_h1,
        "predictor_corrector_relative": result.predictor_corrector_h1 / scale,
        "predicted_state_change_relative": result.predicted_state_change_h1 / scale,
        "branch_overlap": result.branch_overlap,
        "topology_components": topology.component_count if topology else "",
        "topology_raw_components": topology.raw_component_count if topology else "",
        "topology_largest_fraction": topology.largest_fraction if topology else "",
        "topology_second_fraction": topology.second_fraction if topology else "",
        "linear_iterations": result.trace.linear_iterations,
        "linear_convergence_reason": result.trace.linear_reason,
        "coercivity_margin": result.coercivity_margin,
        "coercivity_error": result.coercivity_error,
    }


def branch_acceptance_reason(
    result: ProjectionResult,
    target_energy: float,
    args: argparse.Namespace,
) -> str:
    """Return an empty string exactly when branch-continuity gates pass."""
    if result.rejection_reason:
        return result.rejection_reason
    scale = math.sqrt(target_energy)
    correction = result.predictor_corrector_h1 / scale
    predicted = result.predicted_state_change_h1 / scale
    allowed = max(
        float(args.predictor_correction_absolute),
        float(args.predictor_correction_factor) * max(predicted, 1.0e-14),
    )
    if not math.isfinite(correction) or correction > allowed:
        return f"PREDICTOR_CORRECTION:{correction:.6e}>{allowed:.6e}"
    if not math.isfinite(result.branch_overlap) or result.branch_overlap < float(args.branch_overlap_min):
        return f"BRANCH_OVERLAP:{result.branch_overlap:.6e}"
    return ""


def verify_reduced_derivatives(
    *,
    problem: ProjectionProblem,
    point: np.ndarray,
    state: fem.Function,
    sensitivity_center: fem.Function,
    sensitivity_width: fem.Function,
    reduced: ReducedData,
    bounds: controls.CenterWidthBounds,
    path: Path,
) -> list[dict[str, Any]]:
    """Check fully corrected reduced derivatives on the tracked local branch."""
    args = problem.args
    current_activity = fem.Function(problem.V, name="fdReferenceActivity")
    problem.set_control(point)
    problem.update_activity(state, current_activity)
    sensitivities = (sensitivity_center, sensitivity_width)
    exact_gradients = {
        "objective": reduced.gradient_objective,
        "leakage": reduced.gradient_leakage,
        "missing": reduced.gradient_missing,
    }
    rows: list[dict[str, Any]] = []
    for component, sensitivity in enumerate(sensitivities):
        minus_margin = point[component] - bounds.lower[component]
        plus_margin = bounds.upper[component] - point[component]
        step = min(
            float(args.fd_relative_step) * bounds.scale[component],
            0.25 * minus_margin,
            0.25 * plus_margin,
        )
        if step <= 100.0 * np.finfo(float).eps * max(1.0, abs(point[component])):
            for quantity in exact_gradients:
                rows.append(
                    {
                        "component": ("center", "width")[component],
                        "quantity": quantity,
                        "status": "SKIPPED_ACTIVE_BOX_BOUND",
                        "step": step,
                    }
                )
            continue
        corrected_metrics: dict[int, ProjectionMetrics] = {}
        projection_status: dict[int, str] = {}
        for sign in (-1, 1):
            trial_point = point.copy()
            trial_point[component] += sign * step
            trial_state = fem.Function(problem.V, name=f"fdState{component}{sign}")
            trial_correction = fem.Function(problem.V, name=f"fdCorrection{component}{sign}")
            trial_density = fem.Function(problem.V, name=f"fdDensity{component}{sign}")
            predictor = fem.Function(problem.V, name=f"fdPredictor{component}{sign}")
            predictor.x.array[:] = state.x.array + sign * step * sensitivity.x.array
            predictor.x.scatter_forward()
            _copy_function(predictor, trial_state)
            result = problem.project(
                state=trial_state,
                correction=trial_correction,
                density=trial_density,
                point=trial_point,
                prefix=f"reduced_fd_{component}_{'plus' if sign > 0 else 'minus'}",
                predictor=predictor,
                accepted_state=state,
                reference_activity=current_activity,
                phase="reduced_derivative_verification",
                outer_iteration=-20 - component,
            )
            gate = branch_acceptance_reason(result, problem.target_energy, args)
            if result.metrics is None or gate:
                projection_status[sign] = gate or result.rejection_reason or result.newton.status
            else:
                corrected_metrics[sign] = result.metrics
                projection_status[sign] = "OK"
        for quantity, exact in exact_gradients.items():
            row: dict[str, Any] = {
                "component": ("center", "width")[component],
                "quantity": quantity,
                "step": step,
                "minus_projection": projection_status.get(-1, "NOT_RUN"),
                "plus_projection": projection_status.get(1, "NOT_RUN"),
                "analytic": float(exact[component]),
            }
            if len(corrected_metrics) == 2:
                minus_value = float(getattr(corrected_metrics[-1], quantity))
                plus_value = float(getattr(corrected_metrics[1], quantity))
                finite_difference = (plus_value - minus_value) / (2.0 * step)
                relative_error = abs(finite_difference - exact[component]) / max(
                    abs(finite_difference), abs(exact[component]), 1.0e-14
                )
                signal = abs(plus_value - minus_value)
                row.update(
                    {
                        "status": "OK",
                        "minus_value": minus_value,
                        "plus_value": plus_value,
                        "finite_difference": finite_difference,
                        "relative_error": relative_error,
                        "finite_difference_signal": signal,
                        "signal_above_newton_tolerance": int(
                            signal >= float(args.fd_signal_factor) * float(args.newton_tol)
                        ),
                    }
                )
            else:
                row["status"] = "FAILED_BRANCH_PROJECTION"
            rows.append(row)
    problem.set_control(point)
    if problem.comm.rank == 0:
        path.write_text(
            json.dumps(rows, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return rows


def write_visualization(
    *,
    problem: ProjectionProblem,
    torsion: fem.Function,
    target_mask,
    final_state: fem.Function,
    final_density: fem.Function,
    run_dir: Path,
) -> None:
    """Write XDMF-compatible CG1 fields and DG0 projected densities."""
    V1 = fem.functionspace(problem.domain, ("Lagrange", 1))
    continuous_sources = (
        ("T_h", torsion),
        ("phi_T", problem.phi_target),
        ("phi_h", final_state),
    )
    continuous: list[fem.Function] = []
    for name, source in continuous_sources:
        target = fem.Function(V1, name=name)
        target.interpolate(source)
        target.x.scatter_forward()
        continuous.append(target)
    error_high = fem.Function(problem.V, name="potential_error_high")
    error_high.x.array[:] = final_state.x.array - problem.phi_target.x.array
    error_high.x.scatter_forward()
    error = fem.Function(V1, name="potential_error")
    error.interpolate(error_high)
    error.x.scatter_forward()
    continuous.append(error)

    V0 = fem.functionspace(problem.domain, ("DG", 0))
    q0 = ufl.TrialFunction(V0)
    v0 = ufl.TestFunction(V0)
    dx0 = ufl.Measure(
        "dx",
        domain=problem.domain,
        metadata={"quadrature_degree": int(problem.args.quad_degree)},
    )
    m_target = fem.Function(V0, name="m_T_h")
    activity = fem.Function(V0, name="W_phi_h")
    mismatch = fem.Function(V0, name="density_mismatch")
    final_activity = problem.activity(final_state)
    shared.solve_same_matrix_forms(
        q0 * v0 * dx0,
        [
            target_mask * v0 * dx0,
            final_activity * v0 * dx0,
            (final_activity - target_mask) * v0 * dx0,
        ],
        [m_target, activity, mismatch],
        [],
        prefix="h1proj_dg0_output_",
        solver=problem.args.linear_solver,
        ksp_type=problem.args.ksp_type,
        rtol=problem.args.linear_rtol,
        atol=problem.args.linear_atol,
        max_it=problem.args.linear_max_it,
        verbosity=problem.args.verbosity,
    )
    fields_dir = run_dir / "fields"
    with io.XDMFFile(problem.comm, str(fields_dir / "continuous_fields.xdmf"), "w") as xdmf:
        xdmf.write_mesh(problem.domain)
        for function in continuous:
            xdmf.write_function(function, 0.0)
    with io.XDMFFile(problem.comm, str(fields_dir / "density_fields.xdmf"), "w") as xdmf:
        xdmf.write_mesh(problem.domain)
        for function in (m_target, activity, mismatch):
            xdmf.write_function(function, 0.0)


def write_summary_plot(records: list[dict[str, Any]], args: argparse.Namespace, path: Path) -> None:
    """Create a compact multi-panel optimizer history plot on rank zero."""
    strict_events = {"accepted_initial", "accepted_iteration", "final_strict"}
    by_iteration: dict[int, dict[str, Any]] = {}
    for row in records:
        if row.get("event") in strict_events and int(row.get("accepted", 0) or 0) == 1:
            by_iteration[int(row["iteration"])] = row
    accepted = [by_iteration[index] for index in sorted(by_iteration)]
    if not accepted:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    def series(field: str) -> np.ndarray:
        """Extract one numeric field from strict accepted records."""
        return np.asarray([float(row[field]) for row in accepted], dtype=np.float64)

    iteration = series("iteration")
    objective = series("objective")
    figure, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    axes[0, 0].semilogy(iteration, np.maximum(objective, 1.0e-18), marker="o", label="J")
    axes[0, 0].semilogy(iteration, np.maximum(np.sqrt(2.0 * objective), 1.0e-18), marker="s", label="sqrt(2J)")
    axes[0, 0].set_title("H0-1 projection")
    axes[0, 0].legend()
    axes[0, 1].plot(iteration, series("leakage"), marker="o", label="L")
    axes[0, 1].plot(iteration, series("missing"), marker="s", label="M")
    axes[0, 1].axhline(args.leakage_max, color="C0", linestyle="--")
    axes[0, 1].axhline(args.missing_max, color="C1", linestyle="--")
    axes[0, 1].set_title("Geometric safeguards")
    axes[0, 1].legend()
    for field in ("center", "width", "c1", "c2"):
        axes[1, 0].plot(iteration, series(field), marker="o", label=field)
    axes[1, 0].set_title("Threshold controls")
    axes[1, 0].legend()
    axes[1, 1].semilogy(iteration, np.maximum(series("newton_final_residual"), 1.0e-18), marker="o")
    axes[1, 1].axhline(args.newton_tol, color="black", linestyle="--")
    axes[1, 1].set_title("Strict dual Newton residual")
    axes[2, 0].semilogy(
        iteration,
        np.maximum(series("predictor_corrector_relative"), 1.0e-18),
        marker="o",
    )
    axes[2, 0].set_title("Predictor-corrector H0-1 distance")
    coercivity = np.asarray(
        [float(row["coercivity_margin"]) if _finite(row.get("coercivity_margin")) else np.nan for row in accepted]
    )
    if np.any(np.isfinite(coercivity)):
        spectral_axis = axes[2, 0].twinx()
        spectral_axis.plot(iteration, coercivity, color="C3", marker="x", label="coercivity")
        spectral_axis.set_ylabel("coercivity margin")
    axes[2, 1].plot(iteration, series("jaccard"), marker="o", label="Jaccard")
    axes[2, 1].plot(iteration, series("coverage"), marker="s", label="coverage")
    axes[2, 1].set_title("Density overlap")
    axes[2, 1].legend()
    for axis in axes.ravel():
        axis.set_xlabel("accepted iteration")
        axis.grid(True, alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> int:
    """Execute initialization, local-branch SQP/filter iterations, and output."""
    bounds = validate_args(args)
    comm = MPI.COMM_WORLD
    run_dir = _make_run_dir(args, comm)
    logs_dir = run_dir / "logs"
    out_dir = run_dir / "out"
    terminal_log_path = out_dir / "terminal.log"
    terminal_log_capture = get_bootstrap_terminal_log_capture()
    if args.save_terminal_log:
        if comm.rank == 0:
            terminal_log_path.write_bytes(b"")
        comm.barrier()
        local_capture_error: str | None = None
        try:
            if terminal_log_capture is None:
                terminal_log_capture = TerminalLogCapture(
                    terminal_log_path,
                    rank=int(comm.rank),
                )
                terminal_log_capture.start()
            else:
                terminal_log_capture.retarget(
                    terminal_log_path,
                    rank=int(comm.rank),
                )
        except Exception as error:  # pragma: no cover - descriptor/platform failure
            local_capture_error = f"rank={comm.rank} {type(error).__name__}: {error}"
        capture_errors = tuple(
            error for error in comm.allgather(local_capture_error) if error is not None
        )
        if capture_errors:
            if terminal_log_capture is not None:
                terminal_log_capture.close()
            raise RuntimeError(
                "failed to start terminal log capture: " + "; ".join(capture_errors)
            )
    if comm.rank == 0:
        shared.root_print(comm, "========== TORSION H1 BRANCH PROJECTION ==========")
        shared.root_print(comm, f"RUN_DIR {run_dir}")
        shared.root_print(
            comm,
            f"TERMINAL_LOG {terminal_log_path if args.save_terminal_log else 'disabled'}",
        )
        shared.root_print(
            comm,
            "STRICT_NEWTON "
            f"dualTolerance={args.newton_tol:.6e} maxIterations={args.max_newton_it} "
            "inexactNewton=DISABLED",
        )
        shared.root_print(
            comm,
            "HOMOTOPY_POLICY "
            f"easyStrictNewtonIterations={args.init_homotopy_easy_newton_it} "
            "topologyRejectedSourceRestoration=SKIP_AND_SHRINK "
            "thresholdStopping=FUNCTIONAL_STAGNATION "
            f"thresholdStagnationAtol={args.init_homotopy_threshold_stagnation_atol:.3e} "
            f"thresholdStagnationRtol={args.init_homotopy_threshold_stagnation_rtol:.3e} "
            "thresholdTrustRestart=PERSISTENT",
        )
        shared.root_print(
            comm,
            "INITIALIZATION_ANCHOR_POLICY "
            "primaryPairRanking=GEOMETRIC_VIOLATION "
            "primaryDirectShortlist=UNCHANGED "
            "homotopyFallbackActivation=NO_USABLE_DIRECT_PROJECTION "
            "fallbackTiers=GEOMETRIC,OUTWARD_CONTINUATION,"
            "TOPOLOGY_CONTINUATION,FROZEN_JACCARD,TARGET_COVERAGE "
            f"fallbackCapacity={int(args.init_shortlist)} "
            f"guardTrace={'EVERY_EVALUATION' if int(args.verbosity) >= 3 else 'SUMMARY'}",
        )
        shared.root_print(
            comm,
            "OUTER_STOPPING_POLICY primaryMerit=VIOLATION_IF_INFEASIBLE_ELSE_H1_OBJECTIVE "
            f"stagnationAtol={args.functional_stagnation_atol:.3e} "
            f"stagnationRtol={args.functional_stagnation_rtol:.3e} "
            f"stagnationPatience={args.functional_stagnation_patience} "
            "finalStrictProjection=REQUIRED",
        )

    initialization_handle = (
        (logs_dir / "initialization.csv").open("w", newline="", encoding="utf-8")
        if comm.rank == 0
        else None
    )
    initialization_writer = (
        csv.DictWriter(initialization_handle, fieldnames=INITIALIZATION_FIELDS)
        if initialization_handle is not None
        else None
    )
    if initialization_writer is not None:
        initialization_writer.writeheader()
    outer_handle = (
        (logs_dir / "outer_iterations.csv").open("w", newline="", encoding="utf-8")
        if comm.rank == 0
        else None
    )
    outer_writer = (
        csv.DictWriter(outer_handle, fieldnames=OUTER_FIELDS)
        if outer_handle is not None
        else None
    )
    if outer_writer is not None:
        outer_writer.writeheader()
    homotopy_threshold_handle = (
        (logs_dir / "homotopy_threshold.csv").open("w", newline="", encoding="utf-8")
        if comm.rank == 0
        else None
    )
    homotopy_threshold_writer = (
        csv.DictWriter(
            homotopy_threshold_handle,
            fieldnames=HOMOTOPY_THRESHOLD_FIELDS,
        )
        if homotopy_threshold_handle is not None
        else None
    )
    if homotopy_threshold_writer is not None:
        homotopy_threshold_writer.writeheader()
        shared.root_print(
            comm,
            f"HOMOTOPY_THRESHOLD_CSV {logs_dir / 'homotopy_threshold.csv'}",
        )
    newton_handle = (
        (logs_dir / "newton.csv").open("w", newline="", encoding="utf-8")
        if comm.rank == 0
        else None
    )
    raw_newton_writer = (
        csv.DictWriter(newton_handle, fieldnames=shared.NEWTON_CSV_FIELDS)
        if newton_handle is not None
        else None
    )
    if raw_newton_writer is not None:
        raw_newton_writer.writeheader()
    trace_writer = NewtonTraceWriter(raw_newton_writer, newton_handle)
    newton_args = configure_shared_newton_args(args, trace_writer)
    outer_records: list[dict[str, Any]] = []
    initialization_rows: list[dict[str, Any]] = []
    stiffness_solver: shared.FixedStiffnessSolver | None = None
    problem: ProjectionProblem | None = None
    plotter: MPIPyVistaTorsionPlotter | None = None
    homotopy_workspace: shared.HomotopySolveWorkspace | None = None
    # The shared plotter supports saved frames too, but this mode keeps its
    # durable output in XDMF and the summary PNG.  These compatibility fields
    # configure only the requested live interactive window.
    args.save_frames = False
    args.frame_dir = None
    args.frame_window_width = int(args.plot_window_width)
    args.frame_window_height = int(args.plot_window_height)
    total_started = time.perf_counter()
    return_code = 0

    try:
        domain, mesh_path, geometry_mode = shared.load_or_generate_mesh(args, run_dir, comm)
        V = fem.functionspace(domain, ("Lagrange", int(args.order)))
        bc = shared.boundary_bc(V)
        qdeg = int(args.quad_degree) if args.quad_degree is not None else max(2 * int(args.order) + 8, 12)
        args.quad_degree = qdeg
        dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": qdeg})
        trial = ufl.TrialFunction(V)
        test = ufl.TestFunction(V)
        stiffness_form = ufl.inner(ufl.grad(trial), ufl.grad(test)) * dx
        stiffness_solver = shared.FixedStiffnessSolver(
            stiffness_form,
            V,
            [bc],
            prefix="h1proj_stiffness_",
            solver=args.linear_solver,
            ksp_type=args.ksp_type,
            rtol=args.linear_rtol,
            atol=args.linear_atol,
            max_it=args.linear_max_it,
        )
        torsion = fem.Function(V, name="T_h")
        target_interpolant = fem.Function(V, name="m_T_interpolant")
        phi_target = fem.Function(V, name="phi_T")
        torsion_info = stiffness_solver.solve_form(1.0 * test * dx, torsion)
        _, torsion_max = shared.global_minmax(comm, torsion)
        c1_t = float(args.alpha_t1) * torsion_max
        c2_t = float(args.alpha_t2) * torsion_max
        target_mask = ufl.conditional(
            ufl.gt(torsion, c1_t),
            ufl.conditional(ufl.lt(torsion, c2_t), 1.0, 0.0),
            0.0,
        )
        shared.update_interpolated(target_interpolant, target_mask)
        target_area = shared.assemble_scalar(comm, target_mask * dx)
        if target_area <= 0.0:
            raise RuntimeError("the sharp torsion target band has zero area")
        target_info = stiffness_solver.solve_form(
            float(args.rho_amp) * target_mask * test * dx, phi_target
        )
        target_energy = stiffness_solver.h1_seminorm(phi_target) ** 2
        if target_energy <= 0.0:
            raise RuntimeError("the torsion-design potential has zero H0-1 energy")
        nt = int(domain.topology.index_map(domain.topology.dim).size_global)
        ndof = int(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)
        shared.root_print(
            comm,
            f"MESH mode={geometry_mode} file={mesh_path} cells={nt} order={args.order} dofs={ndof} qdeg={qdeg}",
        )
        shared.root_print(
            comm,
            f"DESIGN Tmax={torsion_max:.12e} c1T={c1_t:.12e} c2T={c2_t:.12e} "
            f"AT={target_area:.12e} ET={target_energy:.12e} "
            f"torsionLinearIts={torsion_info[0]} targetLinearIts={target_info[0]}",
        )

        topology_analyzer = DistributedActivityTopology(V)
        target_topology = topology_analyzer.analyze(
            target_interpolant,
            core_level=0.5,
            bridge_level=0.5,
            min_component_fraction=float(
                args.branch_topology_min_component_fraction
            ),
        )
        expected_topology_components = (
            int(args.branch_topology_expected_components)
            if args.branch_topology_expected_components is not None
            else int(target_topology.component_count)
        )
        if expected_topology_components < 1:
            raise RuntimeError(
                "the discrete torsion target has no significant topology component"
            )
        shared.root_print(
            comm,
            "BRANCH_TOPOLOGY "
            f"guard={int(bool(args.branch_topology_guard))} "
            f"targetComponents={target_topology.component_count} "
            f"targetRawComponents={target_topology.raw_component_count} "
            f"expectedComponents={expected_topology_components} "
            f"coreLevel={float(args.branch_topology_core_level):.3f} "
            f"bridgeLevel={float(args.branch_topology_bridge_level):.3f} "
            f"minimumFraction={float(args.branch_topology_min_component_fraction):.3e}",
        )

        problem = ProjectionProblem(
            domain=domain,
            function_space=V,
            bc=bc,
            trial=trial,
            test=test,
            dx=dx,
            stiffness_form=stiffness_form,
            stiffness_solver=stiffness_solver,
            phi_target=phi_target,
            target_mask=target_mask,
            target_area=target_area,
            target_energy=target_energy,
            topology_analyzer=topology_analyzer,
            expected_topology_components=expected_topology_components,
            rho_amp=args.rho_amp,
            eps_ratio=args.eps_ratio,
            args=args,
            newton_args=newton_args,
            trace_writer=trace_writer,
        )
        plot_error = fem.Function(V, name="plot_phi_minus_phi_T")
        plotter = MPIPyVistaTorsionPlotter(
            args,
            run_tag=run_dir.name,
            run_dir=run_dir,
            frame_writer=None,
            comm=comm,
        )

        def emit_interactive_plot(
            *,
            state: fem.Function,
            density: fem.Function,
            stage: str,
            iteration: int,
            residual: float,
            metrics_for_plot: ProjectionMetrics | None,
            state_title: str = "equilibrium phi_h",
            density_title: str = "density W(phi_h)",
        ) -> None:
            """Collectively update the rank-zero PyVista inspection window."""
            if plotter is None or not args.plot:
                return
            plot_error.x.array[:] = state.x.array - phi_target.x.array
            plot_error.x.scatter_forward()
            if args.plot_fields == "full":
                fields = [
                    torsion,
                    target_interpolant,
                    phi_target,
                    state,
                    density,
                    plot_error,
                ]
                titles = [
                    "torsion T_h",
                    "target m_T,h",
                    "target potential phi_T",
                    state_title,
                    density_title,
                    "phi_h - phi_T",
                ]
                contour_index = 0
            elif args.plot_fields == "density":
                # The hidden torsion field supplies the target-band contours.
                fields = [density, torsion]
                titles = [density_title]
                contour_index = 1
            else:
                fields = [state, density, torsion]
                titles = [state_title, density_title]
                contour_index = 2
            plot_metrics = {
                "massRho": (
                    target_area
                    if metrics_for_plot is None
                    else metrics_for_plot.activity_area_ratio * target_area
                ),
                "activeArea": (
                    target_area
                    if metrics_for_plot is None
                    else metrics_for_plot.activity_area_ratio * target_area
                ),
                "relRhoDesign": (
                    0.0 if metrics_for_plot is None else 1.0 - metrics_for_plot.jaccard
                ),
            }
            plotter.emit(
                fields,
                titles,
                stage=stage,
                ieps=0,
                k=iteration,
                eps_phi=float(problem.eps_const.value),
                residual=float(residual),
                metrics=plot_metrics,
                token=f"{stage.lower()}_{iteration}",
                save=False,
                show=True,
                nt=nt,
                ndof=ndof,
                contour_field_index=contour_index,
                contour_levels=(c1_t, c2_t),
            )

        if args.plot_design:
            problem.set_control(
                np.asarray(
                    [0.5 * (bounds.center_min + bounds.center_max), bounds.width_min],
                    dtype=np.float64,
                )
            )
            emit_interactive_plot(
                state=phi_target,
                density=target_interpolant,
                stage="DESIGN",
                iteration=-1,
                residual=0.0,
                metrics_for_plot=None,
                state_title="target potential phi_T",
                density_title="target density m_T,h",
            )
            plotter.hold_interactive(
                float(args.plot_design_hold_seconds),
                stage="DESIGN",
            )

        if args.verify_window_derivatives and comm.rank == 0:
            check = pointwise_window_derivative_check(
                logs_dir / "window_derivative_verification.json", args
            )
            shared.root_print(
                comm,
                f"WINDOW_DERIVATIVE_CHECK maxRelativeError={check['maximum_relative_error']:.6e}",
            )

        # Stage A: gather the fixed design-potential quadrature samples and run
        # the exact discrete sorted hard-window scan on rank zero.
        initialization_phase_start = time.perf_counter()
        if int(args.verbosity) >= 2:
            shared.root_print(
                comm,
                "INITIALIZATION_PHASE phase=quadrature_sampling_start "
                f"qdeg={qdeg} ranks={comm.size}",
            )
        local_phi, local_torsion, local_weights = shared.quadrature_samples_for_fit(
            phi_target, torsion, quadrature_degree=qdeg
        )
        local_indicator = (
            (local_torsion > c1_t) & (local_torsion < c2_t)
        ).astype(np.float64)
        gathered = comm.gather((local_phi, local_indicator, local_weights), root=0)
        if comm.rank == 0:
            global_phi = np.concatenate([part[0] for part in gathered])
            global_indicator = np.concatenate([part[1] for part in gathered])
            global_weights = np.concatenate([part[2] for part in gathered])
            if int(args.verbosity) >= 2:
                shared.root_print(
                    comm,
                    "INITIALIZATION_PHASE phase=quadrature_sampling_done "
                    f"samples={global_phi.size} "
                    f"rankSampleCounts="
                    f"{[int(part[0].size) for part in gathered]} "
                    f"phiRange=({float(np.min(global_phi)):.12e},"
                    f"{float(np.max(global_phi)):.12e}) "
                    f"sampledTargetArea="
                    f"{float(np.dot(global_weights, global_indicator)):.12e} "
                    f"elapsed={time.perf_counter() - initialization_phase_start:.6f}s",
                )
                shared.root_print(
                    comm,
                    "INITIALIZATION_PHASE phase=hard_window_scan_start "
                    f"samples={global_phi.size} "
                    f"leakageBudget={float(args.init_leakage):.6e} "
                    f"widthBounds=({bounds.width_min:.6e},{bounds.width_max:.6e})",
                )
            hard_scan_start = time.perf_counter()
            hard = controls.sorted_hard_window_scan(
                global_phi,
                global_indicator,
                global_weights,
                bounds,
                float(args.init_leakage),
            )
            reachability = controls.frozen_window_reachability_bound(
                global_phi,
                global_indicator,
                global_weights,
                bounds,
                float(args.eps_ratio),
            )
            reachability_passed = bool(
                reachability.smooth_coverage_upper_bound + 1.0e-12
                >= float(args.init_min_coverage)
            )
            if int(args.verbosity) >= 2:
                shared.root_print(
                    comm,
                    "INITIALIZATION_PHASE phase=hard_window_scan_done "
                    f"m={hard.center:.12e} d={hard.width:.12e} "
                    f"c1={hard.c1:.12e} c2={hard.c2:.12e} "
                    f"L={hard.leakage:.12e} M={hard.missing:.12e} "
                    f"coverage={hard.target_coverage:.12e} "
                    f"elapsed={time.perf_counter() - hard_scan_start:.6f}s",
                )
                shared.root_print(
                    comm,
                    "INITIALIZATION_REACHABILITY_GUARD "
                    f"result={'PASS' if reachability_passed else 'REJECT'} "
                    f"reachableEdgeEnvelope=("
                    f"{reachability.lower_edge_min:.12e},"
                    f"{reachability.upper_edge_max:.12e}) "
                    f"epsilonMax={reachability.epsilon_max:.12e} "
                    f"targetPhiRange=("
                    f"{reachability.target_potential_min:.12e},"
                    f"{reachability.target_potential_max:.12e}) "
                    f"hardCoverageUpperBound="
                    f"{reachability.hard_coverage_upper_bound:.12e} "
                    f"smoothCoverageUpperBound="
                    f"{reachability.smooth_coverage_upper_bound:.12e} "
                    f"minimumCoverage={float(args.init_min_coverage):.12e} "
                    f"action={'CONTINUE_INITIALIZATION' if reachability_passed else 'ABORT_BEFORE_TOPOLOGY_REPAIR_SCAN'}",
                )
                shared.root_print(
                    comm,
                    "INITIALIZATION_PHASE phase=smooth_frozen_refinement_start "
                    f"seedM={hard.center:.12e} seedD={hard.width:.12e} "
                    f"maxIterations={int(args.init_refine_max_it)} "
                    f"ftol={float(args.init_refine_ftol):.3e}",
                )
            frozen_refinement_start = time.perf_counter()
            refined_point, frozen_refined, refinement_info = frozen_refinement(
                global_phi,
                global_indicator,
                global_weights,
                hard,
                bounds,
                args,
            )
            if int(args.verbosity) >= 2:
                shared.root_print(
                    comm,
                    "INITIALIZATION_PHASE phase=smooth_frozen_refinement_done "
                    f"m={refined_point[0]:.12e} d={refined_point[1]:.12e} "
                    f"L={frozen_refined.leakage:.12e} "
                    f"M={frozen_refined.missing:.12e} "
                    f"coverage={frozen_refined.coverage:.12e} "
                    f"success={int(bool(refinement_info['success']))} "
                    f"iterations={int(refinement_info['iterations'])} "
                    f"evaluations={int(refinement_info['evaluations'])} "
                    f"returnedBestFeasible="
                    f"{int(bool(refinement_info['returned_best_feasible']))} "
                    f"message={json.dumps(str(refinement_info['message']))} "
                    f"elapsed={time.perf_counter() - frozen_refinement_start:.6f}s",
                )
            stage_payload = {
                "quadrature_samples": int(global_phi.size),
                "assembled_target_area": target_area,
                "sampled_target_area": float(np.dot(global_weights, global_indicator)),
                "hard_scan": {
                    "center": hard.center,
                    "width": hard.width,
                    "c1": hard.c1,
                    "c2": hard.c2,
                    "leakage": hard.leakage,
                    "missing": hard.missing,
                },
                "reachability_guard": {
                    "passed": reachability_passed,
                    "lower_edge_min": reachability.lower_edge_min,
                    "upper_edge_max": reachability.upper_edge_max,
                    "epsilon_max": reachability.epsilon_max,
                    "target_potential_min": reachability.target_potential_min,
                    "target_potential_max": reachability.target_potential_max,
                    "hard_coverage_upper_bound": (
                        reachability.hard_coverage_upper_bound
                    ),
                    "smooth_coverage_upper_bound": (
                        reachability.smooth_coverage_upper_bound
                    ),
                    "minimum_coverage": float(args.init_min_coverage),
                },
                "smooth_refinement": {
                    "center": float(refined_point[0]),
                    "width": float(refined_point[1]),
                    "leakage": frozen_refined.leakage,
                    "missing": frozen_refined.missing,
                    **refinement_info,
                },
            }
            (logs_dir / "initialization_stages.json").write_text(
                json.dumps(stage_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        else:
            global_phi = global_indicator = global_weights = None
            hard = refined_point = frozen_refined = refinement_info = None
            reachability = None
        hard = comm.bcast(hard, root=0)
        refined_point = np.asarray(comm.bcast(refined_point, root=0), dtype=np.float64)
        reachability = comm.bcast(reachability, root=0)
        if (
            reachability.smooth_coverage_upper_bound + 1.0e-12
            < float(args.init_min_coverage)
        ):
            raise RuntimeError(
                "the explicit center-width bounds cannot capture the configured "
                "minimum target coverage on the frozen design potential: "
                f"smooth coverage upper bound "
                f"{reachability.smooth_coverage_upper_bound:.6e} < "
                f"{float(args.init_min_coverage):.6e}; enlarge --center-min, "
                "--center-max, or --width-max"
            )
        if int(args.verbosity) >= 2:
            shared.root_print(
                comm,
                "INITIALIZATION_PHASE phase=frozen_topology_check_start "
                f"m={refined_point[0]:.12e} d={refined_point[1]:.12e}",
            )
        frozen_topology_start = time.perf_counter()
        problem.set_control(refined_point)
        problem.update_activity(phi_target, problem.branch_trial_activity)
        frozen_topology = problem.activity_topology(problem.branch_trial_activity)
        smooth_refinement_topology = frozen_topology
        if int(args.verbosity) >= 2:
            shared.root_print(
                comm,
                "INITIALIZATION_PHASE phase=frozen_topology_check_done "
                f"components={frozen_topology.component_count}/"
                f"{problem.expected_topology_components} "
                f"rawComponents={frozen_topology.raw_component_count} "
                f"coreCells={frozen_topology.core_cell_count} "
                f"bridgeCells={frozen_topology.bridge_cell_count} "
                f"largestFraction={frozen_topology.largest_fraction:.6e} "
                f"secondFraction={frozen_topology.second_fraction:.6e} "
                f"elapsed={time.perf_counter() - frozen_topology_start:.6f}s",
            )
        topology_repair_info: dict[str, Any] = {
            "attempted": False,
            "repaired": False,
            "original_center": float(refined_point[0]),
            "original_width": float(refined_point[1]),
            "original_components": int(frozen_topology.component_count),
            "original_raw_components": int(frozen_topology.raw_component_count),
        }
        topology_repair_seed_points = [refined_point.copy()]
        topology_repair_seed_metadata: list[dict[str, Any]] = [
            {
                "selection_tier": "smooth_refinement",
                "strategy": "smooth_refinement",
                "geometric_rank": 1,
                "continuation_rank": 1,
            }
        ]
        topology_repair_ranked_diagnostics: list[dict[str, Any]] = []
        topology_repair_required = bool(
            bool(args.branch_topology_guard)
            and bool(args.init_topology_repair)
            and frozen_topology.component_count
            != problem.expected_topology_components
        )
        if int(args.verbosity) >= 3:
            shared.root_print(
                comm,
                "INITIALIZATION_TOPOLOGY_REPAIR_GUARD "
                f"branchTopologyGuard="
                f"{'PASS' if bool(args.branch_topology_guard) else 'DISABLED'} "
                f"repairEnabledGuard="
                f"{'PASS' if bool(args.init_topology_repair) else 'DISABLED'} "
                f"componentMatchGuard="
                f"{'PASS_NO_REPAIR' if frozen_topology.component_count == problem.expected_topology_components else 'REJECT_REPAIR_REQUIRED'} "
                f"components={frozen_topology.component_count}/"
                f"{problem.expected_topology_components} "
                f"action={'START_FROZEN_TOPOLOGY_REPAIR_SCAN' if topology_repair_required else 'KEEP_SMOOTH_REFINED_PAIR'}",
            )
        if topology_repair_required:
            topology_repair_info["attempted"] = True
            original_point = refined_point.copy()
            original_c1, original_c2 = controls.thresholds_from_center_width(
                *original_point
            )
            usable_matches: list[
                tuple[
                    np.ndarray,
                    controls.FrozenWindowMetrics,
                    controls.ActivityTopology,
                    float,
                    str,
                ]
            ] = []
            checked_candidates = 0
            topology_matches = 0
            seen_repair_points: set[tuple[float, float]] = set()
            topology_scan_start = time.perf_counter()

            outward_repair_candidates = controls.outward_interval_repair_candidates(
                original_point[0],
                original_point[1],
                bounds,
                samples=int(args.init_topology_repair_samples),
            )
            box_repair_candidates = controls.box_interval_repair_candidates(
                original_point[0],
                original_point[1],
                bounds,
                samples=int(args.init_topology_repair_samples),
            )

            def repair_point_key(point: np.ndarray) -> tuple[float, float]:
                return (
                    round(float(point[0]), 15),
                    round(float(point[1]), 15),
                )

            total_unique_repair_candidates = len(
                {
                    repair_point_key(point)
                    for point in outward_repair_candidates + box_repair_candidates
                }
            )
            topology_progress_interval = max(
                1, total_unique_repair_candidates // 20
            )

            def repair_entry_key(
                item,
            ) -> tuple[float, float, float, float, float]:
                _, metric_value, _, radius_value, _ = item
                return controls.frozen_topology_repair_rank_key(
                    metric_value,
                    leakage_max=float(args.leakage_max),
                    missing_max=float(args.missing_max),
                    edge_motion=float(radius_value),
                )

            def repair_continuation_key(item) -> tuple[Any, ...]:
                """Rank a frozen match by branch-anchor cleanliness."""
                _, metric_value, topology_value, radius_value, _ = item
                return controls.frozen_topology_continuation_rank_key(
                    metric_value,
                    topology_value,
                    expected_components=problem.expected_topology_components,
                    leakage_max=float(args.leakage_max),
                    missing_max=float(args.missing_max),
                    edge_motion=float(radius_value),
                )

            def repair_jaccard_key(item) -> tuple[Any, ...]:
                """Use the frozen-frontier overlap criterion as one tier."""
                overlap = controls.soft_overlap_diagnostics(
                    item[1].leakage, item[1].missing
                )
                return (-float(overlap["jaccard"]), *repair_continuation_key(item))

            def repair_outward_key(item) -> tuple[Any, ...]:
                """Prefer the nearest clean outward enclosure."""
                key = repair_continuation_key(item)
                return (*key[:4], key[6], key[4], key[5], key[7])

            def repair_coverage_key(item) -> tuple[Any, ...]:
                """Retain one noncollapsed high-coverage continuation option."""
                return (-float(item[1].coverage), *repair_continuation_key(item))

            def report_topology_scan_progress(
                strategy: str,
                *,
                current_point: np.ndarray | None = None,
                current_topology: controls.ActivityTopology | None = None,
                force: bool = False,
            ) -> None:
                """Emit bounded rank-zero progress for the full frozen scan."""
                if int(args.verbosity) < 2 or comm.rank != 0:
                    return
                if (
                    not force
                    and checked_candidates % topology_progress_interval != 0
                    and checked_candidates != total_unique_repair_candidates
                ):
                    return
                if usable_matches:
                    best = min(usable_matches, key=repair_entry_key)
                    best_text = (
                        f"bestM={best[0][0]:.12e} bestD={best[0][1]:.12e} "
                        f"bestL={best[1].leakage:.6e} "
                        f"bestMissing={best[1].missing:.6e} "
                        f"bestCoverage={best[1].coverage:.6e} "
                        f"bestViolation={repair_entry_key(best)[0]:.6e}"
                    )
                else:
                    best_text = "best=unavailable"
                if current_point is not None and current_topology is not None:
                    current_text = (
                        f"currentM={current_point[0]:.12e} "
                        f"currentD={current_point[1]:.12e} "
                        f"currentComponents={current_topology.component_count} "
                        f"currentRawComponents={current_topology.raw_component_count}"
                    )
                else:
                    current_text = "current=complete"
                shared.root_print(
                    comm,
                    "INITIALIZATION_TOPOLOGY_SCAN "
                    f"strategy={strategy} "
                    f"checked={checked_candidates}/"
                    f"{total_unique_repair_candidates} "
                    f"topologyMatches={topology_matches} "
                    f"usableMatches={len(usable_matches)} "
                    f"{current_text} "
                    f"{best_text} "
                    f"elapsed={time.perf_counter() - topology_scan_start:.6f}s",
                )

            if int(args.verbosity) >= 2:
                shared.root_print(
                    comm,
                    "INITIALIZATION_PHASE phase=topology_repair_scan_start "
                    f"outwardCandidates={len(outward_repair_candidates)} "
                    f"boxCandidates={len(box_repair_candidates)} "
                    f"uniqueCandidates={total_unique_repair_candidates} "
                    f"requiredComponents={problem.expected_topology_components} "
                    f"minimumCoverage={float(args.init_min_coverage):.6e}",
                )

            def scan_topology_repair_candidates(
                candidates: list[np.ndarray], strategy: str
            ) -> None:
                """Collect every unique topology/coverage-compatible point."""
                nonlocal checked_candidates, topology_matches
                for repair_point in candidates:
                    point_key = repair_point_key(repair_point)
                    if point_key in seen_repair_points:
                        continue
                    seen_repair_points.add(point_key)
                    repair_c1, repair_c2 = controls.thresholds_from_center_width(
                        *repair_point
                    )
                    repair_radius = max(
                        abs(repair_c1 - original_c1),
                        abs(repair_c2 - original_c2),
                    ) / float(original_point[1])
                    problem.set_control(repair_point)
                    problem.update_activity(
                        phi_target, problem.branch_trial_activity
                    )
                    repair_topology = problem.activity_topology(
                        problem.branch_trial_activity
                    )
                    checked_candidates += 1
                    topology_passed = bool(
                        repair_topology.component_count
                        == problem.expected_topology_components
                    )
                    if not topology_passed:
                        if int(args.verbosity) >= 3:
                            shared.root_print(
                                comm,
                                "INITIALIZATION_TOPOLOGY_GUARD "
                                f"candidate={checked_candidates}/"
                                f"{total_unique_repair_candidates} "
                                f"strategy={strategy} "
                                f"m={repair_point[0]:.12e} "
                                f"d={repair_point[1]:.12e} "
                                "boundsGuard=PASS "
                                "componentGuard=REJECT "
                                f"components={repair_topology.component_count}/"
                                f"{problem.expected_topology_components} "
                                f"rawComponents={repair_topology.raw_component_count} "
                                f"secondFraction={repair_topology.second_fraction:.6e} "
                                "coverageGuard=NOT_EVALUATED "
                                "geometryRole=NOT_EVALUATED "
                                "action=SKIP_CANDIDATE",
                            )
                        report_topology_scan_progress(
                            strategy,
                            current_point=repair_point,
                            current_topology=repair_topology,
                        )
                        continue
                    topology_matches += 1
                    repair_metrics = comm.bcast(
                        (
                            controls.frozen_window_metrics(
                                global_phi,
                                global_indicator,
                                global_weights,
                                repair_point[0],
                                repair_point[1],
                                args.eps_ratio,
                            )
                            if comm.rank == 0
                            else None
                        ),
                        root=0,
                    )
                    entry = (
                        repair_point.copy(),
                        repair_metrics,
                        repair_topology,
                        repair_radius,
                        strategy,
                    )
                    coverage_passed = bool(
                        repair_metrics.coverage + 1.0e-12
                        >= float(args.init_min_coverage)
                    )
                    if coverage_passed:
                        usable_matches.append(entry)
                    if int(args.verbosity) >= 3:
                        continuation_key = repair_continuation_key(entry)
                        geometric_key = repair_entry_key(entry)
                        overlap = controls.soft_overlap_diagnostics(
                            repair_metrics.leakage, repair_metrics.missing
                        )
                        shared.root_print(
                            comm,
                            "INITIALIZATION_TOPOLOGY_GUARD "
                            f"candidate={checked_candidates}/"
                            f"{total_unique_repair_candidates} "
                            f"strategy={strategy} "
                            f"m={repair_point[0]:.12e} "
                            f"d={repair_point[1]:.12e} "
                            "boundsGuard=PASS componentGuard=PASS "
                            f"components={repair_topology.component_count}/"
                            f"{problem.expected_topology_components} "
                            f"rawComponents={repair_topology.raw_component_count} "
                            f"rawExcess={continuation_key[3]} "
                            f"unexpectedCoreFraction={continuation_key[2]:.6e} "
                            f"secondFraction={repair_topology.second_fraction:.6e} "
                            f"coverageGuard={'PASS' if coverage_passed else 'REJECT'} "
                            f"coverage={repair_metrics.coverage:.6e} "
                            f"minimumCoverage={float(args.init_min_coverage):.6e} "
                            "geometryRole=RANK_ONLY "
                            f"L={repair_metrics.leakage:.6e} "
                            f"M={repair_metrics.missing:.6e} "
                            f"violation={geometric_key[0]:.6e} "
                            f"jaccard={overlap['jaccard']:.6e} "
                            f"action={'RETAIN_FOR_RANKING' if coverage_passed else 'SKIP_CANDIDATE'}",
                        )
                    report_topology_scan_progress(
                        strategy,
                        current_point=repair_point,
                        current_topology=repair_topology,
                    )

            scan_topology_repair_candidates(
                outward_repair_candidates,
                "outward",
            )
            scan_topology_repair_candidates(
                box_repair_candidates,
                "box",
            )
            report_topology_scan_progress("complete", force=True)

            if comm.rank == 0:
                ranked_repair_matches = sorted(
                    usable_matches, key=repair_entry_key
                )
                ranked_continuation_matches = sorted(
                    usable_matches, key=repair_continuation_key
                )
                selected_repair = (
                    ranked_repair_matches[0] if ranked_repair_matches else None
                )
                geometric_rank_by_point = {
                    repair_point_key(item[0]): rank
                    for rank, item in enumerate(ranked_repair_matches, start=1)
                }
                continuation_rank_by_point = {
                    repair_point_key(item[0]): rank
                    for rank, item in enumerate(ranked_continuation_matches, start=1)
                }

                # Preserve the primary geometric winner, then diversify only
                # the conditional homotopy fallback.  This ensemble borrows
                # the older initializers' principles: outward continuation,
                # a topology-clean branch seed, frozen-frontier Jaccard, and
                # target coverage.  No additional nonlinear solve is used.
                tiered_matches = [
                    ("geometric_violation", ranked_repair_matches),
                    (
                        "outward_continuation",
                        sorted(
                            [item for item in usable_matches if item[4] == "outward"],
                            key=repair_outward_key,
                        ),
                    ),
                    ("topology_continuation", ranked_continuation_matches),
                    (
                        "frozen_jaccard",
                        sorted(usable_matches, key=repair_jaccard_key),
                    ),
                    (
                        "target_coverage",
                        sorted(usable_matches, key=repair_coverage_key),
                    ),
                ]
                selected_anchor_items: list[tuple[str, Any]] = []
                selected_anchor_points: set[tuple[float, float]] = set()
                shortlist_capacity = int(args.init_shortlist)
                for tier_name, tier_matches in tiered_matches:
                    if len(selected_anchor_items) >= shortlist_capacity:
                        if int(args.verbosity) >= 3:
                            shared.root_print(
                                comm,
                                "INITIALIZATION_ANCHOR_TIER_GUARD "
                                f"tier={tier_name} candidates={len(tier_matches)} "
                                "capacityGuard=REJECT result=NOT_EVALUATED "
                                "action=SKIP_TIER_SHORTLIST_FULL",
                            )
                        continue
                    tier_candidate = next(
                        (
                            item
                            for item in tier_matches
                            if repair_point_key(item[0])
                            not in selected_anchor_points
                        ),
                        None,
                    )
                    duplicates = sum(
                        repair_point_key(item[0]) in selected_anchor_points
                        for item in tier_matches
                    )
                    if tier_candidate is None:
                        if int(args.verbosity) >= 3:
                            shared.root_print(
                                comm,
                                "INITIALIZATION_ANCHOR_TIER_GUARD "
                                f"tier={tier_name} candidates={len(tier_matches)} "
                                f"duplicatesSkipped={duplicates} "
                                "capacityGuard=PASS diversityGuard=REJECT "
                                "action=SKIP_TIER_NO_UNIQUE_CANDIDATE",
                            )
                        continue
                    selected_anchor_items.append((tier_name, tier_candidate))
                    selected_anchor_points.add(repair_point_key(tier_candidate[0]))
                    if int(args.verbosity) >= 3:
                        shared.root_print(
                            comm,
                            "INITIALIZATION_ANCHOR_TIER_GUARD "
                            f"tier={tier_name} candidates={len(tier_matches)} "
                            f"duplicatesSkipped={duplicates} "
                            "capacityGuard=PASS diversityGuard=PASS "
                            f"m={tier_candidate[0][0]:.12e} "
                            f"d={tier_candidate[0][1]:.12e} "
                            "action=ADD_TO_CONDITIONAL_HOMOTOPY_SHORTLIST",
                        )

                # Defensive fill: a requested capacity larger than the number
                # of distinct tiers still receives the next geometric points.
                for item in ranked_repair_matches:
                    if len(selected_anchor_items) >= shortlist_capacity:
                        break
                    point_key = repair_point_key(item[0])
                    if point_key in selected_anchor_points:
                        continue
                    selected_anchor_items.append(("geometric_fill", item))
                    selected_anchor_points.add(point_key)

                topology_repair_seed_points_payload = []
                for tier_name, item in selected_anchor_items:
                    point_value, metric_value, topology_value, radius_value, strategy_value = item
                    point_key = repair_point_key(point_value)
                    overlap = controls.soft_overlap_diagnostics(
                        metric_value.leakage, metric_value.missing
                    )
                    continuation_key = repair_continuation_key(item)
                    topology_repair_seed_points_payload.append(
                        {
                            "point": point_value.tolist(),
                            "selection_tier": tier_name,
                            "strategy": str(strategy_value),
                            "geometric_rank": int(
                                geometric_rank_by_point[point_key]
                            ),
                            "continuation_rank": int(
                                continuation_rank_by_point[point_key]
                            ),
                            "leakage": float(metric_value.leakage),
                            "missing": float(metric_value.missing),
                            "coverage": float(metric_value.coverage),
                            "jaccard": float(overlap["jaccard"]),
                            "edge_motion_in_original_widths": float(radius_value),
                            "components": int(topology_value.component_count),
                            "raw_components": int(
                                topology_value.raw_component_count
                            ),
                            "second_fraction": float(
                                topology_value.second_fraction
                            ),
                            "raw_component_excess": int(continuation_key[3]),
                            "unexpected_core_fraction": float(
                                continuation_key[2]
                            ),
                            "geometric_violation": float(
                                repair_entry_key(item)[0]
                            ),
                        }
                    )
                topology_repair_ranked_diagnostics = [
                    {
                        "rank": rank,
                        "center": float(item[0][0]),
                        "width": float(item[0][1]),
                        "leakage": float(item[1].leakage),
                        "missing": float(item[1].missing),
                        "coverage": float(item[1].coverage),
                        "geometric_violation": float(repair_entry_key(item)[0]),
                        "edge_motion_in_original_widths": float(item[3]),
                        "strategy": str(item[4]),
                    }
                    for rank, item in enumerate(
                        ranked_repair_matches[: int(args.init_shortlist)], start=1
                    )
                ]
            else:
                selected_repair = None
                topology_repair_seed_points_payload = None
            selected_repair = comm.bcast(selected_repair, root=0)
            topology_repair_seed_payload = comm.bcast(
                topology_repair_seed_points_payload, root=0
            )
            topology_repair_seed_points = [
                np.asarray(record["point"], dtype=np.float64)
                for record in topology_repair_seed_payload
            ]
            topology_repair_seed_metadata = [
                {key: value for key, value in record.items() if key != "point"}
                for record in topology_repair_seed_payload
            ]
            if selected_repair is None:
                raise RuntimeError(
                    "the frozen design potential admits no usable topology-"
                    "compatible window inside the explicit center-width bounds"
                )
            refined_point = np.asarray(selected_repair[0], dtype=np.float64)
            frozen_refined = selected_repair[1]
            frozen_topology = selected_repair[2]
            repair_radius = float(selected_repair[3])
            repair_strategy = str(selected_repair[4])
            problem.set_control(refined_point)
            problem.update_activity(phi_target, problem.branch_trial_activity)
            topology_repair_info.update(
                {
                    "repaired": True,
                    "checked_candidates": int(checked_candidates),
                    "strategy": repair_strategy,
                    "minimum_required_coverage": float(args.init_min_coverage),
                    "edge_motion_in_original_widths": repair_radius,
                    "center": float(refined_point[0]),
                    "width": float(refined_point[1]),
                    "leakage": float(frozen_refined.leakage),
                    "missing": float(frozen_refined.missing),
                    "coverage": float(frozen_refined.coverage),
                    "geometric_violation": controls.normalized_violation(
                        np.asarray(
                            [
                                frozen_refined.leakage - float(args.leakage_max),
                                frozen_refined.missing - float(args.missing_max),
                            ],
                            dtype=np.float64,
                        )
                    ),
                    "components": int(frozen_topology.component_count),
                    "raw_components": int(frozen_topology.raw_component_count),
                    "ranked_shortlist": topology_repair_ranked_diagnostics,
                    "continuation_shortlist": topology_repair_seed_payload,
                }
            )
            shared.root_print(
                comm,
                "INITIALIZATION_TOPOLOGY_REPAIR "
                f"old=({original_point[0]:.12e},{original_point[1]:.12e}) "
                f"new=({refined_point[0]:.12e},{refined_point[1]:.12e}) "
                f"strategy={repair_strategy} "
                f"edgeMotionWidths={repair_radius:.6e} "
                f"components={frozen_topology.component_count}/"
                f"{problem.expected_topology_components} "
                f"frozenL={frozen_refined.leakage:.6e} "
                f"frozenM={frozen_refined.missing:.6e} "
                f"coverage={frozen_refined.coverage:.6e} "
                f"violation={topology_repair_info['geometric_violation']:.6e}",
            )
            if comm.rank == 0 and int(args.verbosity) >= 2:
                for ranked in topology_repair_ranked_diagnostics:
                    shared.root_print(
                        comm,
                        "INITIALIZATION_TOPOLOGY_RANK "
                        f"rank={ranked['rank']} "
                        f"m={ranked['center']:.12e} "
                        f"d={ranked['width']:.12e} "
                        f"L={ranked['leakage']:.6e} "
                        f"M={ranked['missing']:.6e} "
                        f"coverage={ranked['coverage']:.6e} "
                        f"violation={ranked['geometric_violation']:.6e} "
                        f"edgeMotionWidths="
                        f"{ranked['edge_motion_in_original_widths']:.6e} "
                        f"strategy={ranked['strategy']}",
                    )
                for anchor_rank, anchor in enumerate(
                    topology_repair_seed_payload, start=1
                ):
                    shared.root_print(
                        comm,
                        "INITIALIZATION_CONTINUATION_ANCHOR "
                        f"rank={anchor_rank}/"
                        f"{len(topology_repair_seed_payload)} "
                        f"tier={anchor['selection_tier']} "
                        f"strategy={anchor['strategy']} "
                        f"geometricRank={anchor['geometric_rank']} "
                        f"continuationRank={anchor['continuation_rank']} "
                        f"m={anchor['point'][0]:.12e} "
                        f"d={anchor['point'][1]:.12e} "
                        f"L={anchor['leakage']:.6e} "
                        f"M={anchor['missing']:.6e} "
                        f"coverage={anchor['coverage']:.6e} "
                        f"jaccard={anchor['jaccard']:.6e} "
                        f"components={anchor['components']}/"
                        f"{problem.expected_topology_components} "
                        f"rawComponents={anchor['raw_components']} "
                        f"rawExcess={anchor['raw_component_excess']} "
                        f"unexpectedCoreFraction="
                        f"{anchor['unexpected_core_fraction']:.6e} "
                        f"edgeMotionWidths="
                        f"{anchor['edge_motion_in_original_widths']:.6e} "
                        "activation=ONLY_IF_DIRECT_SHORTLIST_HAS_NO_USABLE_STATE",
                    )
        shared.root_print(
            comm,
            f"INITIALIZATION hard=({hard.center:.12e},{hard.width:.12e}) "
            f"refined=({refined_point[0]:.12e},{refined_point[1]:.12e}) "
            f"Linit={args.init_leakage:.6e} "
            f"components={frozen_topology.component_count}/"
            f"{problem.expected_topology_components} "
            f"rawComponents={frozen_topology.raw_component_count} "
            f"secondFraction={frozen_topology.second_fraction:.6e}",
        )
        if comm.rank == 0:
            stage_payload["smooth_refinement"]["topology_components"] = int(
                smooth_refinement_topology.component_count
            )
            stage_payload["smooth_refinement"]["topology_raw_components"] = int(
                smooth_refinement_topology.raw_component_count
            )
            stage_payload["smooth_refinement"]["topology_largest_fraction"] = float(
                smooth_refinement_topology.largest_fraction
            )
            stage_payload["smooth_refinement"]["topology_second_fraction"] = float(
                smooth_refinement_topology.second_fraction
            )
            stage_payload["topology_repair"] = topology_repair_info
            (logs_dir / "initialization_stages.json").write_text(
                json.dumps(stage_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )

        # Stage C: only the requested small deterministic shortlist receives
        # full strict nonlinear projections from phi_T.
        shortlist = controls.initialization_shortlist(
            refined_point[0],
            refined_point[1],
            bounds,
            count=int(args.init_shortlist),
            width_fraction=float(args.init_width_perturbation),
            center_fraction=float(args.init_center_perturbation),
        )
        if int(args.verbosity) >= 2:
            shared.root_print(
                comm,
                "INITIALIZATION_PHASE phase=strict_shortlist_start "
                f"candidates={len(shortlist)} "
                f"newtonIterationCap={int(args.init_candidate_max_newton_it)}",
            )
            if comm.rank == 0:
                for shortlist_rank, (name, center, width) in enumerate(
                    shortlist, start=1
                ):
                    shared.root_print(
                        comm,
                        "INITIALIZATION_SHORTLIST "
                        f"rank={shortlist_rank} name={name} "
                        f"m={center:.12e} d={width:.12e}",
                    )
        init_state = fem.Function(V, name="initialCandidateState")
        init_correction = fem.Function(V, name="initialCandidateCorrection")
        init_density = fem.Function(V, name="initialCandidateDensity")
        candidate_data: list[dict[str, Any]] = []

        def evaluate_initial_candidate(name: str, candidate_point: np.ndarray) -> dict[str, Any]:
            """Strictly project and record one initialization candidate."""
            _copy_function(phi_target, init_state)
            prefix = f"initial_{shared.slug_for_path(name)}_{len(candidate_data)}"
            projection = problem.project(
                state=init_state,
                correction=init_correction,
                density=init_density,
                point=candidate_point,
                prefix=prefix,
                predictor=phi_target,
                accepted_state=phi_target,
                phase="initialization_candidate",
                outer_iteration=-1,
                max_newton_it=int(args.init_candidate_max_newton_it),
            )
            if comm.rank == 0:
                frozen = controls.frozen_window_metrics(
                    global_phi,
                    global_indicator,
                    global_weights,
                    candidate_point[0],
                    candidate_point[1],
                    args.eps_ratio,
                )
            else:
                frozen = None
            frozen = comm.bcast(frozen, root=0)
            strict_success = projection.metrics is not None and not projection.rejection_reason
            feasible = bool(
                strict_success
                and projection.metrics.leakage <= args.leakage_max + args.constraint_tol
                and projection.metrics.missing <= args.missing_max + args.constraint_tol
            )
            metrics = projection.metrics
            coverage = 1.0 - metrics.missing if metrics is not None else 0.0
            usable = controls.initialization_candidate_is_usable(
                strict_success=strict_success,
                missing=metrics.missing if metrics is not None else math.inf,
                minimum_coverage=float(args.init_min_coverage),
            )
            overlap = (
                {
                    "activity_area_ratio": metrics.activity_area_ratio,
                    "overlap_area": metrics.overlap_area,
                    "overlap_area_ratio": metrics.overlap_area_ratio,
                    "recall": metrics.recall,
                    "precision": metrics.precision,
                    "jaccard": metrics.jaccard,
                }
                if metrics is not None
                else {}
            )
            reason = projection.rejection_reason
            if strict_success and not usable:
                reason = (
                    f"COLLAPSED_BRANCH:COVERAGE={coverage:.6e}"
                    f"<{float(args.init_min_coverage):.6e}"
                )
            elif strict_success and not feasible:
                reason = "GEOMETRIC_SAFEGUARD"
            row = {
                "candidate": name,
                **control_log_values(candidate_point, args.eps_ratio),
                "frozen_leakage": frozen.leakage,
                "frozen_missing": frozen.missing,
                "frozen_coverage": frozen.coverage,
                "newton_success": int(strict_success),
                "newton_status": projection.newton.status,
                "newton_iterations": projection.newton.iterations,
                "damping_history": _json_array(projection.trace.damping_history),
                "initial_dual_residual": projection.initial_residual,
                "final_dual_residual": projection.newton.residual,
                "first_correction_h1": projection.trace.first_correction_h1,
                "projected_objective": metrics.objective if metrics else "",
                "projected_leakage": metrics.leakage if metrics else "",
                "projected_missing": metrics.missing if metrics else "",
                **overlap,
                "linear_iterations": projection.trace.linear_iterations,
                "linear_convergence_reason": projection.trace.linear_reason,
                "coercivity_margin": projection.coercivity_margin,
                "coercivity_error": projection.coercivity_error,
                "topology_components": (
                    projection.topology.component_count if projection.topology else ""
                ),
                "topology_raw_components": (
                    projection.topology.raw_component_count if projection.topology else ""
                ),
                "topology_largest_fraction": (
                    projection.topology.largest_fraction if projection.topology else ""
                ),
                "topology_second_fraction": (
                    projection.topology.second_fraction if projection.topology else ""
                ),
                "selected": 0,
                "reason": reason or "ELIGIBLE",
            }
            data = {
                "name": name,
                "point": candidate_point.copy(),
                "projection": projection,
                "state": init_state.x.array.copy(),
                "row": row,
                "strict_success": strict_success,
                "usable": usable,
                "feasible": feasible,
            }
            candidate_data.append(data)
            if int(args.verbosity) >= 3:
                newton_guard = bool(
                    projection.newton.converged
                    and math.isfinite(projection.newton.residual)
                    and projection.newton.residual <= float(args.newton_tol)
                )
                topology_guard = (
                    "NOT_EVALUATED"
                    if projection.topology is None
                    else (
                        "PASS"
                        if projection.topology.component_count
                        == problem.expected_topology_components
                        else "REJECT"
                    )
                )
                coverage_guard = (
                    "NOT_EVALUATED"
                    if metrics is None
                    else (
                        "PASS"
                        if coverage + 1.0e-12 >= float(args.init_min_coverage)
                        else "REJECT"
                    )
                )
                leakage_guard = (
                    "NOT_EVALUATED"
                    if metrics is None
                    else (
                        "PASS"
                        if metrics.leakage
                        <= float(args.leakage_max) + float(args.constraint_tol)
                        else "REJECT"
                    )
                )
                missing_guard = (
                    "NOT_EVALUATED"
                    if metrics is None
                    else (
                        "PASS"
                        if metrics.missing
                        <= float(args.missing_max) + float(args.constraint_tol)
                        else "REJECT"
                    )
                )
                shared.root_print(
                    comm,
                    "INITIAL_CANDIDATE_GUARD "
                    f"name={name} strictNewtonGuard="
                    f"{'PASS' if newton_guard else 'REJECT'} "
                    f"residual={projection.newton.residual:.12e} "
                    f"requestedTolerance={float(args.newton_tol):.12e} "
                    f"projectionReason={projection.rejection_reason or 'NONE'} "
                    f"topologyGuard={topology_guard} "
                    f"coverageGuard={coverage_guard} "
                    f"leakageGuard={leakage_guard} "
                    f"missingGuard={missing_guard} "
                    f"action={'RETAIN_USABLE_CANDIDATE' if usable else 'REJECT_CANDIDATE'}",
                )
            if args.plot_initial_candidates and metrics is not None:
                emit_interactive_plot(
                    state=init_state,
                    density=init_density,
                    stage="INITIAL_CANDIDATE",
                    iteration=len(candidate_data) - 1,
                    residual=projection.newton.residual,
                    metrics_for_plot=metrics,
                    state_title=f"candidate phi_h: {name}",
                    density_title=f"candidate W(phi_h): {name}",
                )
            if int(args.verbosity) >= 2:
                projected = (
                    f"J={metrics.objective:.12e} L={metrics.leakage:.12e} "
                    f"M={metrics.missing:.12e} Jaccard={metrics.jaccard:.6e}"
                    if metrics is not None
                    else "J=unavailable L=unavailable M=unavailable Jaccard=unavailable"
                )
                shared.root_print(
                    comm,
                    f"INITIAL_CANDIDATE name={name} m={candidate_point[0]:.12e} "
                    f"d={candidate_point[1]:.12e} frozenL={frozen.leakage:.12e} "
                    f"frozenM={frozen.missing:.12e} {projected} "
                    f"Newton={projection.newton.status} iterations={projection.newton.iterations} "
                    f"residual={projection.newton.residual:.12e} reason={reason or 'ELIGIBLE'}",
                )
            return data

        for name, center, width in shortlist:
            evaluate_initial_candidate(name, np.asarray([center, width], dtype=np.float64))

        homotopy_threshold_reference = fem.Function(V, name="homotopyThresholdReference")
        homotopy_threshold_predictor = fem.Function(V, name="homotopyThresholdPredictor")
        homotopy_threshold_activity = fem.Function(V, name="homotopyThresholdActivity")
        homotopy_source_reference = fem.Function(V, name="homotopySourceReference")
        homotopy_source_activity = fem.Function(V, name="homotopySourceActivity")
        homotopy_sensitivity_center = fem.Function(V, name="homotopySensitivityCenter")
        homotopy_sensitivity_width = fem.Function(V, name="homotopySensitivityWidth")
        homotopy_threshold_trust = float(args.init_homotopy_threshold_trust_radius)
        homotopy_threshold_event = 0
        homotopy_rescue_counts: dict[tuple[float, float], int] = {}
        homotopy_source_metrics: ProjectionMetrics | None = None
        homotopy_last_source_rejection_reason = ""

        def append_homotopy_threshold_row(**updates: Any) -> None:
            """Write one complete alternating-continuation diagnostic row."""
            row = {field: "" for field in HOMOTOPY_THRESHOLD_FIELDS}
            row.update(updates)
            if homotopy_threshold_writer is not None:
                homotopy_threshold_writer.writerow(row)
                homotopy_threshold_handle.flush()

        def homotopy_threshold_micro_step(
            *,
            point_at_lambda: np.ndarray,
            lam: float,
            source_trial_lambda: float | None,
            stage: int,
            threshold_step: int,
            trigger: str,
            workspace: shared.HomotopySolveWorkspace,
            prefix: str,
        ) -> tuple[bool, np.ndarray]:
            """Attempt one bounded, strictly corrected threshold SQP step."""
            nonlocal homotopy_threshold_trust, homotopy_threshold_event
            current_point = np.asarray(point_at_lambda, dtype=np.float64).copy()
            problem.set_control(current_point)
            workspace.set_parameters(
                lam=lam,
                c1=float(problem.c1_const.value),
                c2=float(problem.c2_const.value),
                eps_phi=float(problem.eps_const.value),
            )
            current_metrics = problem.evaluate_metrics(init_state)
            current_violation = controls.normalized_violation(
                np.asarray(
                    [
                        current_metrics.leakage - float(args.leakage_max),
                        current_metrics.missing - float(args.missing_max),
                    ]
                )
            )
            currently_feasible = current_violation <= float(args.constraint_tol)
            # Along the regular branch phi(lambda,z)-phi_T is O(lambda).
            # Scaling by lambda^-2 avoids a vanishing objective model without
            # changing the minimizer or any reported physical metric.
            objective_scale = 1.0 / max(float(lam) * float(lam), 1.0e-12)
            reduced = problem.sensitivities(
                state=init_state,
                sensitivity_center=homotopy_sensitivity_center,
                sensitivity_width=homotopy_sensitivity_width,
                point=current_point,
                iteration=-(100000 + homotopy_threshold_event),
                homotopy_lambda=lam,
                objective_scale=objective_scale,
                homotopy_workspace=workspace,
            )
            if int(args.verbosity) >= 3:
                shared.root_print(
                    comm,
                    "HOMOTOPY_THRESHOLD_POLICY "
                    f"trigger={trigger} stage={stage} lambda={lam:.6e} "
                    f"sourceTrialLambda="
                    f"{source_trial_lambda if source_trial_lambda is not None else 'NONE'} "
                    f"thresholdStep={threshold_step} "
                    f"mode={'FEASIBLE_OBJECTIVE_SQP' if currently_feasible else 'GEOMETRIC_RESTORATION'} "
                    f"J={current_metrics.objective:.12e} "
                    f"L={current_metrics.leakage:.12e} "
                    f"M={current_metrics.missing:.12e} "
                    f"violation={current_violation:.12e} "
                    f"objectiveScale={objective_scale:.12e} "
                    f"trust={homotopy_threshold_trust:.6e} "
                    "action=ASSEMBLE_REDUCED_MODEL_AND_SOLVE_2D_SUBPROBLEM",
                )
            _copy_function(init_state, homotopy_threshold_reference)
            problem.update_activity(init_state, homotopy_threshold_activity)

            for trial_number in range(1, int(args.init_homotopy_threshold_max_trials) + 1):
                radius = float(homotopy_threshold_trust)
                base_rows, base_rhs, _ = controls.linear_step_constraints(
                    current_point, bounds, radius
                )
                edge_rows, edge_rhs, _ = controls.threshold_edge_step_constraints(
                    current_point[1],
                    float(args.init_homotopy_threshold_edge_step_fraction),
                )
                base_rows = np.vstack((base_rows, edge_rows))
                base_rhs = np.concatenate((base_rhs, edge_rhs))
                if comm.rank == 0:
                    if currently_feasible:
                        rows = np.vstack(
                            (
                                base_rows,
                                reduced.gradient_leakage,
                                reduced.gradient_missing,
                            )
                        )
                        rhs = np.concatenate(
                            (
                                base_rhs,
                                [
                                    args.leakage_max - current_metrics.leakage,
                                    args.missing_max - current_metrics.missing,
                                ],
                            )
                        )
                        algebraic = controls.solve_convex_qp_2d(
                            reduced.gradient_objective,
                            reduced.hessian,
                            rows,
                            rhs,
                        )
                    else:
                        algebraic = controls.solve_restoration_step_2d(
                            reduced.gradient_objective,
                            reduced.hessian,
                            np.asarray(
                                [
                                    current_metrics.leakage - args.leakage_max,
                                    current_metrics.missing - args.missing_max,
                                ]
                            ),
                            np.vstack(
                                (reduced.gradient_leakage, reduced.gradient_missing)
                            ),
                            base_rows,
                            base_rhs,
                        )
                    proposal_payload = (
                        bool(algebraic.success),
                        np.asarray(algebraic.step, dtype=np.float64),
                        str(algebraic.reason),
                    )
                else:
                    proposal_payload = None
                subproblem_success, step, subproblem_reason = comm.bcast(
                    proposal_payload, root=0
                )
                step = np.asarray(step, dtype=np.float64)
                trial_point = np.asarray(
                    bounds.clip(*(current_point + step)), dtype=np.float64
                )
                step = trial_point - current_point
                scaled_step = float(np.linalg.norm(step / bounds.scale, ord=np.inf))
                predicted_model_change = float(
                    reduced.gradient_objective @ step
                    + 0.5 * step @ reduced.hessian @ step
                )
                predicted_reduction = -predicted_model_change
                predicted_leakage = current_metrics.leakage + float(
                    reduced.gradient_leakage @ step
                )
                predicted_missing = current_metrics.missing + float(
                    reduced.gradient_missing @ step
                )
                if not subproblem_success:
                    preprojection_reason = f"SUBPROBLEM:{subproblem_reason}"
                elif scaled_step <= 10.0 * np.finfo(float).eps:
                    preprojection_reason = "ZERO_CONTROL_STEP"
                elif currently_feasible and (
                    predicted_leakage > args.leakage_max + args.constraint_tol
                    or predicted_missing > args.missing_max + args.constraint_tol
                ):
                    preprojection_reason = "PREDICTED_GEOMETRIC_CROSSING"
                else:
                    preprojection_reason = ""

                if int(args.verbosity) >= 3:
                    predicted_geometry_enforced = currently_feasible
                    shared.root_print(
                        comm,
                        "HOMOTOPY_THRESHOLD_PREPROJECTION_GUARD "
                        f"trigger={trigger} stage={stage} lambda={lam:.6e} "
                        f"thresholdStep={threshold_step} trial={trial_number} "
                        f"subproblemGuard={'PASS' if subproblem_success else 'REJECT'} "
                        f"subproblemReason={subproblem_reason} "
                        f"nonzeroStepGuard={'PASS' if scaled_step > 10.0 * np.finfo(float).eps else 'REJECT'} "
                        f"predictedLeakage={predicted_leakage:.12e} "
                        f"predictedMissing={predicted_missing:.12e} "
                        f"predictedLeakageGuard="
                        f"{('PASS' if predicted_leakage <= args.leakage_max + args.constraint_tol else 'REJECT') if predicted_geometry_enforced else 'DIAGNOSTIC_ONLY'} "
                        f"predictedMissingGuard="
                        f"{('PASS' if predicted_missing <= args.missing_max + args.constraint_tol else 'REJECT') if predicted_geometry_enforced else 'DIAGNOSTIC_ONLY'} "
                        f"reason={preprojection_reason or 'NONE'} "
                        f"action={'REJECT_WITHOUT_PDE_AND_SHRINK_TRUST' if preprojection_reason else 'RUN_STRICT_NEWTON_PROJECTION'}",
                    )

                common_log = {
                    "trigger": trigger,
                    "stage": stage,
                    "lambda": lam,
                    "source_trial_lambda": (
                        "" if source_trial_lambda is None else source_trial_lambda
                    ),
                    "threshold_step": threshold_step,
                    "trial": trial_number,
                    "objective_scale": objective_scale,
                    "old_center": current_point[0],
                    "old_width": current_point[1],
                    "new_center": trial_point[0],
                    "new_width": trial_point[1],
                    "new_c1": trial_point[0] - 0.5 * trial_point[1],
                    "new_c2": trial_point[0] + 0.5 * trial_point[1],
                    "new_epsilon": args.eps_ratio * trial_point[1],
                    "old_objective": current_metrics.objective,
                    "old_leakage": current_metrics.leakage,
                    "old_missing": current_metrics.missing,
                    "old_violation": current_violation,
                    "grad_objective": _json_array(reduced.gradient_objective),
                    "grad_leakage": _json_array(reduced.gradient_leakage),
                    "grad_missing": _json_array(reduced.gradient_missing),
                    "gauss_newton": json.dumps(
                        reduced.hessian.tolist(), separators=(",", ":")
                    ),
                    "gauss_newton_condition": reduced.condition_number,
                    "step_center": step[0],
                    "step_width": step[1],
                    "step_c1": step[0] - 0.5 * step[1],
                    "step_c2": step[0] + 0.5 * step[1],
                    "edge_step_fraction": max(
                        abs(step[0] - 0.5 * step[1]),
                        abs(step[0] + 0.5 * step[1]),
                    ) / max(float(current_point[1]), 1.0e-30),
                    "scaled_step_inf": scaled_step,
                    "trust_radius": radius,
                    "predicted_objective_reduction": predicted_reduction,
                    "predicted_leakage": predicted_leakage,
                    "predicted_missing": predicted_missing,
                }
                homotopy_threshold_event += 1
                if preprojection_reason:
                    append_homotopy_threshold_row(
                        **common_log,
                        accepted=0,
                        reason=preprojection_reason,
                    )
                    if int(args.verbosity) >= 1:
                        shared.root_print(
                            comm,
                            f"HOMOTOPY_THRESHOLD_REJECT trigger={trigger} stage={stage} "
                            f"lambda={lam:.6e} trial={trial_number} "
                            f"dm={step[0]:.6e} dd={step[1]:.6e} "
                            f"trust={radius:.6e} reason={preprojection_reason}",
                        )
                    if preprojection_reason.startswith("SUBPROBLEM") or preprojection_reason == "ZERO_CONTROL_STEP":
                        return False, current_point
                    homotopy_threshold_trust = max(
                        float(args.init_homotopy_threshold_trust_min),
                        min(radius * float(args.trust_shrink), scaled_step * float(args.trust_shrink)),
                    )
                    if radius <= float(args.init_homotopy_threshold_trust_min) * (
                        1.0 + 1.0e-12
                    ):
                        return False, current_point
                    continue

                predictor_usable, _ = problem.set_predictor(
                    homotopy_threshold_predictor,
                    homotopy_threshold_reference,
                    homotopy_sensitivity_center,
                    homotopy_sensitivity_width,
                    step,
                )
                _copy_function(homotopy_threshold_predictor, init_state)
                projection = problem.project(
                    state=init_state,
                    correction=init_correction,
                    density=init_density,
                    point=trial_point,
                    prefix=(
                        f"{prefix}_threshold_{trigger}_{stage}_{threshold_step}_"
                        f"trial_{trial_number}"
                    ),
                    predictor=homotopy_threshold_predictor,
                    accepted_state=homotopy_threshold_reference,
                    reference_activity=homotopy_threshold_activity,
                    phase="initialization_homotopy_threshold_strict",
                    outer_iteration=-1,
                    max_newton_it=int(args.trial_max_newton_it),
                    homotopy_lambda=lam,
                    homotopy_workspace=workspace,
                )
                trial_metrics = projection.metrics
                branch_reason = branch_acceptance_reason(projection, target_energy, args)
                actual_violation = (
                    controls.normalized_violation(
                        np.asarray(
                            [
                                trial_metrics.leakage - args.leakage_max,
                                trial_metrics.missing - args.missing_max,
                            ]
                        )
                    )
                    if trial_metrics is not None
                    else math.inf
                )
                actual_reduction = (
                    objective_scale
                    * (current_metrics.objective - trial_metrics.objective)
                    if trial_metrics is not None
                    else math.nan
                )
                ratio = (
                    actual_reduction / predicted_reduction
                    if predicted_reduction > 1.0e-16 and math.isfinite(actual_reduction)
                    else math.nan
                )
                if branch_reason:
                    accepted = False
                    acceptance_reason = branch_reason
                elif currently_feasible:
                    actual_feasible = bool(
                        trial_metrics.leakage <= args.leakage_max + args.constraint_tol
                        and trial_metrics.missing <= args.missing_max + args.constraint_tol
                    )
                    if not actual_feasible:
                        accepted = False
                        acceptance_reason = "ACTUAL_GEOMETRIC_CROSSING"
                    elif not math.isfinite(ratio) or ratio < float(args.acceptance_eta):
                        accepted = False
                        acceptance_reason = f"OBJECTIVE_RATIO:{ratio:.6e}"
                    else:
                        accepted = True
                        acceptance_reason = "PARTIAL_OBJECTIVE_MODEL_ACCEPTED"
                else:
                    required = float(args.restoration_fraction) * max(
                        current_violation, 1.0e-14
                    )
                    accepted = bool(actual_violation <= current_violation - required)
                    acceptance_reason = (
                        "PARTIAL_GEOMETRY_RESTORATION_ACCEPTED"
                        if accepted
                        else "INSUFFICIENT_PARTIAL_GEOMETRY_RESTORATION"
                    )
                if not predictor_usable:
                    acceptance_reason += ";PREDICTOR_FALLBACK_ACCEPTED_STATE"
                if int(args.verbosity) >= 3:
                    scale = math.sqrt(target_energy)
                    correction_relative = (
                        projection.predictor_corrector_h1 / scale
                        if math.isfinite(projection.predictor_corrector_h1)
                        else math.nan
                    )
                    predicted_relative = (
                        projection.predicted_state_change_h1 / scale
                        if math.isfinite(projection.predicted_state_change_h1)
                        else math.nan
                    )
                    allowed_correction = max(
                        float(args.predictor_correction_absolute),
                        float(args.predictor_correction_factor)
                        * max(predicted_relative, 1.0e-14),
                    )
                    topology_guard = (
                        "NOT_EVALUATED"
                        if projection.topology is None
                        else (
                            "PASS"
                            if projection.topology.component_count
                            == problem.expected_topology_components
                            else "REJECT"
                        )
                    )
                    overlap_guard = (
                        "PASS"
                        if math.isfinite(projection.branch_overlap)
                        and projection.branch_overlap
                        >= float(args.branch_overlap_min)
                        else "REJECT"
                    )
                    predictor_guard = (
                        "PASS"
                        if math.isfinite(correction_relative)
                        and correction_relative <= allowed_correction
                        else "REJECT"
                    )
                    actual_geometry_guard = (
                        "NOT_EVALUATED"
                        if trial_metrics is None
                        else (
                            "PASS"
                            if trial_metrics.leakage
                            <= args.leakage_max + args.constraint_tol
                            and trial_metrics.missing
                            <= args.missing_max + args.constraint_tol
                            else "REJECT"
                        )
                    )
                    model_guard = (
                        (
                            "PASS"
                            if math.isfinite(ratio)
                            and ratio >= float(args.acceptance_eta)
                            else "REJECT"
                        )
                        if currently_feasible
                        else (
                            "PASS"
                            if actual_violation
                            <= current_violation
                            - float(args.restoration_fraction)
                            * max(current_violation, 1.0e-14)
                            else "REJECT"
                        )
                    )
                    shared.root_print(
                        comm,
                        "HOMOTOPY_THRESHOLD_POSTPROJECTION_GUARD "
                        f"trigger={trigger} stage={stage} lambda={lam:.6e} "
                        f"thresholdStep={threshold_step} trial={trial_number} "
                        f"strictNewtonGuard="
                        f"{'PASS' if projection.newton.converged and projection.newton.residual <= float(args.newton_tol) else 'REJECT'} "
                        f"residual={projection.newton.residual:.12e} "
                        f"predictorGuard={predictor_guard} "
                        f"predictorCorrection={correction_relative:.6e} "
                        f"predictorAllowed={allowed_correction:.6e} "
                        f"overlapGuard={overlap_guard} "
                        f"overlap={projection.branch_overlap:.6e} "
                        f"topologyGuard={topology_guard} "
                        f"actualGeometryGuard={actual_geometry_guard} "
                        f"modelOrRestorationGuard={model_guard} "
                        f"ratio={ratio:.6e} "
                        f"oldViolation={current_violation:.6e} "
                        f"newViolation={actual_violation:.6e} "
                        f"reason={acceptance_reason} "
                        f"action={'ACCEPT_THRESHOLD_STEP' if accepted else 'RESTORE_ACCEPTED_STATE_AND_SHRINK_TRUST'}",
                    )
                append_homotopy_threshold_row(
                    **common_log,
                    accepted=int(accepted),
                    new_objective=(trial_metrics.objective if trial_metrics else ""),
                    new_leakage=(trial_metrics.leakage if trial_metrics else ""),
                    new_missing=(trial_metrics.missing if trial_metrics else ""),
                    new_violation=(actual_violation if trial_metrics else ""),
                    actual_objective_reduction=actual_reduction,
                    acceptance_ratio=ratio,
                    newton_status=projection.newton.status,
                    newton_iterations=projection.newton.iterations,
                    newton_initial_residual=projection.initial_residual,
                    newton_final_residual=projection.newton.residual,
                    minimum_damping=projection.trace.minimum_damping,
                    predictor_corrector_h1=projection.predictor_corrector_h1,
                    branch_overlap=projection.branch_overlap,
                    topology_components=(
                        projection.topology.component_count
                        if projection.topology
                        else ""
                    ),
                    topology_raw_components=(
                        projection.topology.raw_component_count
                        if projection.topology
                        else ""
                    ),
                    topology_largest_fraction=(
                        projection.topology.largest_fraction
                        if projection.topology
                        else ""
                    ),
                    topology_second_fraction=(
                        projection.topology.second_fraction
                        if projection.topology
                        else ""
                    ),
                    reason=acceptance_reason,
                )
                if accepted:
                    if (
                        (math.isfinite(ratio) and ratio >= float(args.acceptance_grow_eta))
                        or actual_violation <= 0.5 * current_violation
                    ) and scaled_step >= 0.8 * radius:
                        homotopy_threshold_trust = min(
                            float(args.init_homotopy_threshold_trust_max),
                            radius * float(args.trust_grow),
                        )
                    if int(args.verbosity) >= 1:
                        shared.root_print(
                            comm,
                            f"HOMOTOPY_THRESHOLD_ACCEPT trigger={trigger} stage={stage} "
                            f"lambda={lam:.6e} m={trial_point[0]:.12e} "
                            f"d={trial_point[1]:.12e} J={trial_metrics.objective:.12e} "
                            f"L={trial_metrics.leakage:.12e} M={trial_metrics.missing:.12e} "
                            f"violation={actual_violation:.6e} ratio={ratio:.6e} "
                            f"components={projection.topology.component_count}/"
                            f"{problem.expected_topology_components} "
                            f"residual={projection.newton.residual:.6e}",
                        )
                    return True, trial_point

                _copy_function(homotopy_threshold_reference, init_state)
                problem.set_control(current_point)
                c1_old, c2_old = controls.thresholds_from_center_width(*current_point)
                workspace.set_parameters(
                    lam=lam,
                    c1=c1_old,
                    c2=c2_old,
                    eps_phi=args.eps_ratio * current_point[1],
                )
                if int(args.verbosity) >= 1:
                    shared.root_print(
                        comm,
                        f"HOMOTOPY_THRESHOLD_REJECT trigger={trigger} stage={stage} "
                        f"lambda={lam:.6e} trial={trial_number} "
                        f"m={trial_point[0]:.12e} d={trial_point[1]:.12e} "
                        f"trust={radius:.6e} reason={acceptance_reason}",
                    )
                homotopy_threshold_trust = max(
                    float(args.init_homotopy_threshold_trust_min),
                    min(radius * float(args.trust_shrink), scaled_step * float(args.trust_shrink)),
                )
                if radius <= float(args.init_homotopy_threshold_trust_min) * (
                    1.0 + 1.0e-12
                ):
                    return False, current_point
            return False, current_point

        def evaluate_homotopy_candidate(
            name: str,
            candidate_point: np.ndarray,
        ) -> dict[str, Any]:
            """Track a branch with alternating strict source/control corrections."""
            nonlocal homotopy_workspace, homotopy_source_metrics
            nonlocal homotopy_threshold_trust, homotopy_rescue_counts
            nonlocal homotopy_last_source_rejection_reason
            homotopy_threshold_trust = float(
                args.init_homotopy_threshold_trust_radius
            )
            homotopy_rescue_counts = {}
            homotopy_last_source_rejection_reason = ""
            current_point = np.asarray(candidate_point, dtype=np.float64).copy()
            threshold_stagnated_lambda: float | None = None
            _copy_function(phi_target, init_state)
            problem.set_control(current_point)
            initial_residual = problem.dual_residual(init_state)
            prefix = f"initial_{shared.slug_for_path(name)}_{len(candidate_data)}"
            trace_writer.clear(prefix)
            if homotopy_workspace is None:
                homotopy_workspace = shared.HomotopySolveWorkspace(
                    u=init_state,
                    trial=trial,
                    test=test,
                    dx=dx,
                    bc=bc,
                    target_density=float(args.rho_amp) * target_mask,
                    c1_const=problem.c1_const,
                    c2_const=problem.c2_const,
                    eps_const=problem.eps_const,
                    rho_amp=float(args.rho_amp),
                    args=newton_args,
                )
            homotopy_tangent = fem.Function(V, name=f"{shared.slug_for_path(name)}_tangent")
            c1, c2, epsilon = problem.set_control(current_point)

            def refresh_source_reference() -> None:
                """Snapshot the complete last accepted source/control state."""
                nonlocal homotopy_source_metrics
                _copy_function(init_state, homotopy_source_reference)
                problem.update_activity(init_state, homotopy_source_activity)
                homotopy_source_metrics = problem.evaluate_metrics(init_state)

            refresh_source_reference()

            def source_trial_acceptance(
                stage: int,
                lambda_old: float,
                lambda_trial: float,
            ) -> str | None:
                """Reject an exact source step that jumps between activity branches."""
                nonlocal homotopy_last_source_rejection_reason
                if homotopy_source_metrics is None:
                    raise RuntimeError("homotopy source reference was not initialized")
                c1_trial, c2_trial = controls.thresholds_from_center_width(*current_point)
                epsilon_trial = float(args.eps_ratio) * float(current_point[1])
                problem.update_activity(init_state, problem.branch_trial_activity)
                similarity = shared.activity_dice_ratio(
                    comm=comm,
                    activity_ref=homotopy_source_activity,
                    activity_trial=problem.branch_trial_activity,
                    dx=dx,
                )
                topology = problem.activity_topology(problem.branch_trial_activity)
                topology_reason = problem.topology_rejection_reason(topology)
                trial_metrics = problem.evaluate_metrics(init_state)
                old_violation = controls.normalized_violation(
                    np.asarray(
                        [
                            homotopy_source_metrics.leakage - float(args.leakage_max),
                            homotopy_source_metrics.missing - float(args.missing_max),
                        ]
                    )
                )
                trial_violation = controls.normalized_violation(
                    np.asarray(
                        [
                            trial_metrics.leakage - float(args.leakage_max),
                            trial_metrics.missing - float(args.missing_max),
                        ]
                    )
                )
                state_step = problem.h1_distance(
                    init_state, homotopy_source_reference
                ) / math.sqrt(target_energy)
                accepted = bool(
                    math.isfinite(similarity)
                    and similarity >= float(args.branch_overlap_min)
                    and not topology_reason
                )
                if accepted:
                    reason = "SOURCE_BRANCH_GUARDS_ACCEPTED"
                elif topology_reason:
                    reason = topology_reason
                else:
                    reason = (
                        f"SOURCE_DENSITY_DICE:{similarity:.6e}"
                        f"<{float(args.branch_overlap_min):.6e}"
                    )
                homotopy_last_source_rejection_reason = "" if accepted else reason
                append_homotopy_threshold_row(
                    trigger="source_stage_guard",
                    stage=stage,
                    source_trial_lambda=lambda_trial,
                    accepted=int(accepted),
                    old_center=current_point[0],
                    old_width=current_point[1],
                    new_center=current_point[0],
                    new_width=current_point[1],
                    new_c1=c1_trial,
                    new_c2=c2_trial,
                    new_epsilon=epsilon_trial,
                    old_objective=homotopy_source_metrics.objective,
                    new_objective=trial_metrics.objective,
                    old_leakage=homotopy_source_metrics.leakage,
                    new_leakage=trial_metrics.leakage,
                    old_missing=homotopy_source_metrics.missing,
                    new_missing=trial_metrics.missing,
                    old_violation=old_violation,
                    new_violation=trial_violation,
                    source_state_step_h1=state_step,
                    branch_overlap=similarity,
                    topology_components=topology.component_count,
                    topology_raw_components=topology.raw_component_count,
                    topology_largest_fraction=topology.largest_fraction,
                    topology_second_fraction=topology.second_fraction,
                    reason=reason,
                    **{"lambda": lambda_trial},
                )
                if int(args.verbosity) >= 1:
                    verb = "ACCEPT" if accepted else "REJECT"
                    shared.root_print(
                        comm,
                        f"HOMOTOPY_SOURCE_GUARD_{verb} stage={stage} "
                        f"lambdaOld={lambda_old:.6e} lambdaTrial={lambda_trial:.6e} "
                        f"densityDice={similarity:.6e} "
                        f"minimum={float(args.branch_overlap_min):.6e} "
                        f"components={topology.component_count}/"
                        f"{problem.expected_topology_components} "
                        f"secondFraction={topology.second_fraction:.6e} "
                        f"stateStep={state_step:.6e} L={trial_metrics.leakage:.6e} "
                        f"M={trial_metrics.missing:.6e}",
                    )
                if int(args.verbosity) >= 3:
                    shared.root_print(
                        comm,
                        "HOMOTOPY_SOURCE_GUARD_DETAIL "
                        f"stage={stage} lambdaOld={lambda_old:.6e} "
                        f"lambdaTrial={lambda_trial:.6e} "
                        "strictNewtonGuard=PASS "
                        f"finiteDiceGuard="
                        f"{'PASS' if math.isfinite(similarity) else 'REJECT'} "
                        f"overlapGuard="
                        f"{'PASS' if math.isfinite(similarity) and similarity >= float(args.branch_overlap_min) else 'REJECT'} "
                        f"topologyGuard={'PASS' if not topology_reason else 'REJECT'} "
                        f"components={topology.component_count}/"
                        f"{problem.expected_topology_components} "
                        f"rawComponents={topology.raw_component_count} "
                        f"secondFraction={topology.second_fraction:.6e} "
                        "geometryGuard=DIAGNOSTIC_ONLY "
                        f"oldViolation={old_violation:.6e} "
                        f"newViolation={trial_violation:.6e} "
                        f"reason={reason} "
                        f"action={'ACCEPT_STAGE_THEN_APPLY_THRESHOLD_OPTIMIZER' if accepted else ('REJECT_STAGE_SKIP_THRESHOLD_RESCUE_AND_SHRINK_SOURCE_STEP' if topology_reason else 'REJECT_STAGE_CONSIDER_THRESHOLD_RESCUE_THEN_SHRINK_SOURCE_STEP')}",
                    )
                return None if accepted else reason

            def apply_threshold_updates(
                *,
                trigger: str,
                stage: int,
                lam: float,
                source_trial_lambda: float | None,
            ) -> tuple[float, float, float] | None:
                """Take bounded micro-steps until their primary merit stagnates."""
                nonlocal current_point, threshold_stagnated_lambda
                changed = False
                for threshold_step in range(
                    1, int(args.init_homotopy_threshold_steps_per_stage) + 1
                ):
                    problem.set_control(current_point)
                    old_metrics = problem.evaluate_metrics(init_state)
                    accepted, updated_point = homotopy_threshold_micro_step(
                        point_at_lambda=current_point,
                        lam=lam,
                        source_trial_lambda=source_trial_lambda,
                        stage=stage,
                        threshold_step=threshold_step,
                        trigger=trigger,
                        workspace=homotopy_workspace,
                        prefix=prefix,
                    )
                    if not accepted:
                        if homotopy_threshold_trust <= float(
                            args.init_homotopy_threshold_trust_min
                        ) * (1.0 + 1.0e-12):
                            threshold_stagnated_lambda = float(lam)
                            if int(args.verbosity) >= 1:
                                shared.root_print(
                                    comm,
                                    "HOMOTOPY_THRESHOLD_STAGNATION "
                                    f"trigger={trigger} stage={stage} lambda={lam:.6e} "
                                    f"thresholdStep={threshold_step} "
                                    "mode=no_acceptable_step_at_min_trust "
                                    f"trust={homotopy_threshold_trust:.6e} "
                                    f"J={old_metrics.objective:.12e} "
                                    f"L={old_metrics.leakage:.12e} "
                                    f"M={old_metrics.missing:.12e}",
                                )
                        break
                    current_point = updated_point
                    changed = True
                    problem.set_control(current_point)
                    new_metrics = problem.evaluate_metrics(init_state)
                    progress = controls.threshold_functional_progress(
                        old_objective=old_metrics.objective,
                        new_objective=new_metrics.objective,
                        old_leakage=old_metrics.leakage,
                        new_leakage=new_metrics.leakage,
                        old_missing=old_metrics.missing,
                        new_missing=new_metrics.missing,
                        leakage_max=float(args.leakage_max),
                        missing_max=float(args.missing_max),
                        feasibility_tolerance=float(args.constraint_tol),
                        objective_scale=1.0 / max(float(lam) * float(lam), 1.0e-12),
                        absolute_tolerance=float(
                            args.init_homotopy_threshold_stagnation_atol
                        ),
                        relative_tolerance=float(
                            args.init_homotopy_threshold_stagnation_rtol
                        ),
                    )
                    if progress.stagnated:
                        threshold_stagnated_lambda = float(lam)
                        if int(args.verbosity) >= 1:
                            shared.root_print(
                                comm,
                                "HOMOTOPY_THRESHOLD_STAGNATION "
                                f"trigger={trigger} stage={stage} lambda={lam:.6e} "
                                f"thresholdStep={threshold_step} mode={progress.mode} "
                                f"oldMerit={progress.old_merit:.12e} "
                                f"newMerit={progress.new_merit:.12e} "
                                f"improvement={progress.improvement:.12e} "
                                f"required={progress.required_improvement:.12e} "
                                f"Lold={old_metrics.leakage:.12e} "
                                f"Lnew={new_metrics.leakage:.12e} "
                                f"Mold={old_metrics.missing:.12e} "
                                f"Mnew={new_metrics.missing:.12e}",
                            )
                        break
                    threshold_stagnated_lambda = None
                return problem.set_control(current_point) if changed else None

            def accepted_control_update(
                accepted_stage: int, lam: float
            ) -> tuple[float, float, float] | None:
                """Optimize thresholds after every accepted nonzero stage."""
                nonzero_lambda = bool(lam > 1.0e-14)
                if int(args.verbosity) >= 3:
                    shared.root_print(
                        comm,
                        "HOMOTOPY_THRESHOLD_TRIGGER_GUARD "
                        f"trigger=accepted_stage stage={accepted_stage} "
                        f"lambda={lam:.6e} nonzeroLambdaGuard="
                        f"{'PASS' if nonzero_lambda else 'REJECT'} "
                        f"action={'APPLY_AT_LEAST_ONE_STRICT_THRESHOLD_MICRO_STEP' if nonzero_lambda else 'SKIP_ZERO_LAMBDA_THRESHOLD_UPDATE'}",
                    )
                if not nonzero_lambda:
                    return None
                return apply_threshold_updates(
                    trigger="accepted_stage",
                    stage=accepted_stage,
                    lam=lam,
                    source_trial_lambda=None,
                )

            def rejected_control_restoration(
                stage: int, lam: float, trial_lambda: float
            ) -> tuple[float, float, float] | None:
                """Try one bounded restoration before shortening delta-lambda."""
                if lam <= 1.0e-14:
                    if int(args.verbosity) >= 3:
                        shared.root_print(
                            comm,
                            "HOMOTOPY_THRESHOLD_RESCUE_GUARD "
                            f"stage={stage} lambda={lam:.6e} "
                            f"trialLambda={trial_lambda:.6e} "
                            "positiveLambdaGuard=REJECT "
                            "action=SKIP_RESCUE_AND_SHRINK_SOURCE_STEP",
                        )
                    return None
                if homotopy_last_source_rejection_reason.startswith("TOPOLOGY_"):
                    if int(args.verbosity) >= 1:
                        shared.root_print(
                            comm,
                            "HOMOTOPY_THRESHOLD_RESCUE_SKIP "
                            f"stage={stage} lambda={lam:.6e} "
                            f"trialLambda={trial_lambda:.6e} "
                            "reason=SOURCE_TOPOLOGY_REJECTION",
                        )
                    if int(args.verbosity) >= 3:
                        shared.root_print(
                            comm,
                            "HOMOTOPY_THRESHOLD_RESCUE_GUARD "
                            f"stage={stage} lambda={lam:.6e} "
                            f"trialLambda={trial_lambda:.6e} "
                            "positiveLambdaGuard=PASS topologyGuard=REJECT "
                            "stagnationGuard=NOT_EVALUATED "
                            "attemptBudgetGuard=NOT_EVALUATED "
                            "action=SKIP_RESCUE_AND_SHRINK_SOURCE_STEP",
                        )
                    return None
                if (
                    threshold_stagnated_lambda is not None
                    and math.isclose(
                        float(lam),
                        threshold_stagnated_lambda,
                        rel_tol=0.0,
                        abs_tol=1.0e-14,
                    )
                ):
                    if int(args.verbosity) >= 1:
                        shared.root_print(
                            comm,
                            "HOMOTOPY_THRESHOLD_RESCUE_SKIP "
                            f"stage={stage} lambda={lam:.6e} "
                            f"trialLambda={trial_lambda:.6e} "
                            "reason=THRESHOLD_FUNCTIONALS_STAGNATED",
                        )
                    if int(args.verbosity) >= 3:
                        shared.root_print(
                            comm,
                            "HOMOTOPY_THRESHOLD_RESCUE_GUARD "
                            f"stage={stage} lambda={lam:.6e} "
                            f"trialLambda={trial_lambda:.6e} "
                            "positiveLambdaGuard=PASS topologyGuard=PASS "
                            "stagnationGuard=REJECT "
                            "attemptBudgetGuard=NOT_EVALUATED "
                            "action=SKIP_RESCUE_AND_SHRINK_SOURCE_STEP",
                        )
                    return None
                key = (round(float(lam), 14), round(float(trial_lambda), 14))
                used = homotopy_rescue_counts.get(key, 0)
                if used >= int(args.init_homotopy_threshold_rescue_attempts):
                    if int(args.verbosity) >= 3:
                        shared.root_print(
                            comm,
                            "HOMOTOPY_THRESHOLD_RESCUE_GUARD "
                            f"stage={stage} lambda={lam:.6e} "
                            f"trialLambda={trial_lambda:.6e} "
                            f"attemptsUsed={used} "
                            f"attemptLimit="
                            f"{int(args.init_homotopy_threshold_rescue_attempts)} "
                            "attemptBudgetGuard=REJECT "
                            "action=SKIP_RESCUE_AND_SHRINK_SOURCE_STEP",
                        )
                    return None
                homotopy_rescue_counts[key] = used + 1
                if int(args.verbosity) >= 3:
                    shared.root_print(
                        comm,
                        "HOMOTOPY_THRESHOLD_RESCUE_GUARD "
                        f"stage={stage} lambda={lam:.6e} "
                        f"trialLambda={trial_lambda:.6e} "
                        f"attempt={used + 1}/"
                        f"{int(args.init_homotopy_threshold_rescue_attempts)} "
                        "positiveLambdaGuard=PASS topologyGuard=PASS "
                        "stagnationGuard=PASS attemptBudgetGuard=PASS "
                        "action=APPLY_STRICT_THRESHOLD_RESTORATION",
                    )
                update = apply_threshold_updates(
                    trigger="failed_source_step",
                    stage=stage,
                    lam=lam,
                    source_trial_lambda=trial_lambda,
                )
                if update is not None:
                    refresh_source_reference()
                return update

            def plot_homotopy_stage(stage: int, lam: float) -> None:
                """Commit the reference snapshot and refresh the live plot."""
                refresh_source_reference()
                if not (args.plot and args.plot_homotopy_stages):
                    return
                homotopy_workspace.update_nonlinear_density(init_density)
                stage_metrics = problem.evaluate_metrics(init_state)
                stage_residual = stiffness_solver.residual_norm(
                    homotopy_workspace.residual_form, "dual"
                )
                emit_interactive_plot(
                    state=init_state,
                    density=init_density,
                    stage="INITIAL_HOMOTOPY",
                    iteration=stage,
                    residual=stage_residual,
                    metrics_for_plot=stage_metrics,
                    state_title=f"homotopy phi_h: {name}, lambda={lam:.3f}",
                    density_title=f"nonlinear W(phi_h): lambda={lam:.3f}",
                )

            result = shared.solve_homotopy_initialization(
                u=init_state,
                du=init_correction,
                tangent=homotopy_tangent,
                rho=init_density,
                trial=trial,
                test=test,
                dx=dx,
                bc=bc,
                stiffness_form=stiffness_form,
                stiffness_solver=stiffness_solver,
                homotopy_workspace=homotopy_workspace,
                target_density=float(args.rho_amp) * target_mask,
                c1_const=problem.c1_const,
                c2_const=problem.c2_const,
                eps_const=problem.eps_const,
                c1=c1,
                c2=c2,
                eps_phi=epsilon,
                rho_amp=float(args.rho_amp),
                args=newton_args,
                prefix=prefix,
                initialization_writer=None,
                initialization_handle=None,
                run_tag=run_dir.name,
                initialization_method="strict_source_homotopy",
                trajectory_callback=plot_homotopy_stage,
                trial_acceptance_callback=source_trial_acceptance,
                accepted_stage_control_callback=accepted_control_update,
                rejected_stage_control_callback=rejected_control_restoration,
            )
            trace = trace_writer.summarize(prefix)
            problem.set_control(current_point)
            final_residual = (
                problem.dual_residual(init_state) if result.converged else result.last_newton.residual
            )
            strict_success = bool(
                result.converged
                and result.lambda_final >= 1.0 - 1.0e-14
                and math.isfinite(final_residual)
                and final_residual <= float(args.newton_tol)
            )
            metrics: ProjectionMetrics | None = None
            topology = None
            topology_reason = ""
            margin = math.nan
            eigen_error = math.nan
            if strict_success:
                homotopy_workspace.update_nonlinear_density(init_density)
                metrics = problem.evaluate_metrics(init_state)
                problem.update_activity(init_state, problem.branch_trial_activity)
                topology = problem.activity_topology(problem.branch_trial_activity)
                topology_reason = problem.topology_rejection_reason(topology)
                margin, eigen_error, _ = problem.coercivity(
                    init_state, f"{prefix}_coercivity"
                )
            aggregate_newton = shared.NewtonResult(
                status=(
                    "HOMOTOPY_CONVERGED_LAMBDA1"
                    if strict_success
                    else f"HOMOTOPY_{result.status}"
                ),
                converged=strict_success,
                iterations=int(result.total_newton_iterations),
                residual=float(final_residual),
                step_h1=float(result.last_newton.step_h1),
                alpha=float(result.last_newton.alpha),
                backtracks=int(result.last_newton.backtracks),
                solve_time=float(result.newton_solve_time),
            )
            predictor_corrector = (
                problem.h1_distance(init_state, phi_target) if strict_success else math.nan
            )
            projection = ProjectionResult(
                newton=aggregate_newton,
                trace=trace,
                initial_residual=initial_residual,
                metrics=metrics,
                predictor_corrector_h1=predictor_corrector,
                predicted_state_change_h1=0.0,
                branch_overlap=math.nan,
                topology=topology,
                coercivity_margin=margin,
                coercivity_error=eigen_error,
                rejection_reason=(
                    topology_reason
                    if strict_success and topology_reason
                    else ("" if strict_success else f"STRICT_HOMOTOPY:{result.status}")
                ),
            )
            if comm.rank == 0:
                frozen = controls.frozen_window_metrics(
                    global_phi,
                    global_indicator,
                    global_weights,
                    current_point[0],
                    current_point[1],
                    args.eps_ratio,
                )
            else:
                frozen = None
            frozen = comm.bcast(frozen, root=0)
            coverage = 1.0 - metrics.missing if metrics is not None else 0.0
            candidate_success = bool(strict_success and not topology_reason)
            usable = controls.initialization_candidate_is_usable(
                strict_success=candidate_success,
                missing=metrics.missing if metrics is not None else math.inf,
                minimum_coverage=float(args.init_min_coverage),
            )
            feasible = bool(
                usable
                and metrics.leakage <= args.leakage_max + args.constraint_tol
                and metrics.missing <= args.missing_max + args.constraint_tol
            )
            if not strict_success:
                reason = f"STRICT_HOMOTOPY:{result.status}"
            elif topology_reason:
                reason = topology_reason
            elif not usable:
                reason = (
                    f"COLLAPSED_BRANCH:COVERAGE={coverage:.6e}"
                    f"<{float(args.init_min_coverage):.6e}"
                )
            elif not feasible:
                reason = "GEOMETRIC_SAFEGUARD"
            else:
                reason = "ELIGIBLE"
            overlap = (
                {
                    "activity_area_ratio": metrics.activity_area_ratio,
                    "overlap_area": metrics.overlap_area,
                    "overlap_area_ratio": metrics.overlap_area_ratio,
                    "recall": metrics.recall,
                    "precision": metrics.precision,
                    "jaccard": metrics.jaccard,
                }
                if metrics is not None
                else {}
            )
            row = {
                "candidate": name,
                **control_log_values(current_point, args.eps_ratio),
                "frozen_leakage": frozen.leakage,
                "frozen_missing": frozen.missing,
                "frozen_coverage": frozen.coverage,
                "newton_success": int(strict_success),
                "newton_status": aggregate_newton.status,
                "newton_iterations": aggregate_newton.iterations,
                "damping_history": _json_array(trace.damping_history),
                "initial_dual_residual": initial_residual,
                "final_dual_residual": final_residual,
                "first_correction_h1": trace.first_correction_h1,
                "projected_objective": metrics.objective if metrics else "",
                "projected_leakage": metrics.leakage if metrics else "",
                "projected_missing": metrics.missing if metrics else "",
                **overlap,
                "linear_iterations": trace.linear_iterations,
                "linear_convergence_reason": trace.linear_reason,
                "coercivity_margin": margin,
                "coercivity_error": eigen_error,
                "topology_components": topology.component_count if topology else "",
                "topology_raw_components": (
                    topology.raw_component_count if topology else ""
                ),
                "topology_largest_fraction": (
                    topology.largest_fraction if topology else ""
                ),
                "topology_second_fraction": (
                    topology.second_fraction if topology else ""
                ),
                "selected": 0,
                "reason": reason,
            }
            data = {
                "name": name,
                "point": current_point.copy(),
                "projection": projection,
                "state": init_state.x.array.copy(),
                "row": row,
                "strict_success": candidate_success,
                "usable": usable,
                "feasible": feasible,
            }
            candidate_data.append(data)
            if args.plot_initial_candidates and metrics is not None:
                emit_interactive_plot(
                    state=init_state,
                    density=init_density,
                    stage="INITIAL_HOMOTOPY",
                    iteration=len(candidate_data) - 1,
                    residual=final_residual,
                    metrics_for_plot=metrics,
                    state_title=f"homotopy phi_h: {name}",
                    density_title=f"homotopy W(phi_h): {name}",
                )
            if int(args.verbosity) >= 1:
                projected = (
                    f"J={metrics.objective:.12e} L={metrics.leakage:.12e} "
                    f"M={metrics.missing:.12e}"
                    if metrics is not None
                    else "J=unavailable L=unavailable M=unavailable"
                )
                shared.root_print(
                    comm,
                    f"INITIAL_HOMOTOPY name={name} m={current_point[0]:.12e} "
                    f"d={current_point[1]:.12e} stages={result.stages} "
                    f"rejectedStages={result.rejected_steps} {projected} "
                    f"residual={final_residual:.12e} reason={reason}",
                )
            return data

        direct_usable_count = sum(
            bool(candidate["usable"]) for candidate in candidate_data
        )
        activate_homotopy_fallback = bool(
            direct_usable_count == 0 and args.init_homotopy_fallback
        )
        if int(args.verbosity) >= 2:
            shared.root_print(
                comm,
                "INITIALIZATION_HOMOTOPY_FALLBACK_GUARD "
                f"directCandidates={len(candidate_data)} "
                f"directUsable={direct_usable_count} "
                f"fallbackEnabled={int(bool(args.init_homotopy_fallback))} "
                f"topologyRepairApplied={int(bool(topology_repair_info['repaired']))} "
                f"anchors={len(topology_repair_seed_points)} "
                f"result={'ACTIVATE' if activate_homotopy_fallback else 'SKIP'} "
                f"action={'TRY_DIVERSIFIED_STRICT_SOURCE_HOMOTOPY' if activate_homotopy_fallback else 'KEEP_DIRECT_INITIALIZER_RESULT'}",
            )
        if activate_homotopy_fallback:
            # When frozen topology repair was needed, its full-box scan
            # supplies a small geometrically ranked set of connected anchors.
            # Trying these is safer than repeatedly shrinking the width at one
            # center, which can drive a continuation directly into a split or
            # collapsed activity branch.
            for seed_rank, (homotopy_seed, seed_metadata) in enumerate(
                zip(
                    topology_repair_seed_points,
                    topology_repair_seed_metadata,
                    strict=True,
                ),
                start=1,
            ):
                if any(candidate["usable"] for candidate in candidate_data):
                    if int(args.verbosity) >= 2:
                        shared.root_print(
                            comm,
                            "INITIALIZATION_HOMOTOPY_ANCHOR_GUARD "
                            f"rank={seed_rank}/"
                            f"{len(topology_repair_seed_points)} "
                            f"tier={seed_metadata['selection_tier']} "
                            "previousAnchorUsable=PASS "
                            "action=STOP_FALLBACK_SKIP_REMAINING_ANCHORS",
                        )
                    break
                seed_name = (
                    "homotopy_coupled_thresholds"
                    if seed_rank == 1
                    else (
                        f"homotopy_{seed_metadata['selection_tier']}_"
                        f"{seed_rank}"
                    )
                )
                if int(args.verbosity) >= 2:
                    shared.root_print(
                        comm,
                        "INITIALIZATION_PHASE phase=homotopy_anchor_start "
                        f"rank={seed_rank}/{len(topology_repair_seed_points)} "
                        f"name={seed_name} m={homotopy_seed[0]:.12e} "
                        f"d={homotopy_seed[1]:.12e} "
                        f"tier={seed_metadata['selection_tier']} "
                        f"strategy={seed_metadata['strategy']} "
                        f"geometricRank={seed_metadata['geometric_rank']} "
                        f"continuationRank={seed_metadata['continuation_rank']} "
                        "previousAnchorUsable=REJECT "
                        "action=START_STRICT_SOURCE_HOMOTOPY",
                    )
                homotopy_data = evaluate_homotopy_candidate(
                    seed_name, homotopy_seed
                )
                if int(args.verbosity) >= 2:
                    shared.root_print(
                        comm,
                        "INITIALIZATION_HOMOTOPY_ANCHOR_RESULT "
                        f"rank={seed_rank}/"
                        f"{len(topology_repair_seed_points)} "
                        f"tier={seed_metadata['selection_tier']} "
                        f"usable={int(bool(homotopy_data['usable']))} "
                        f"strictSuccess="
                        f"{int(bool(homotopy_data['strict_success']))} "
                        f"feasible={int(bool(homotopy_data['feasible']))} "
                        f"reason={homotopy_data['row']['reason']} "
                        f"action={'ACCEPT_AND_STOP_FALLBACK' if homotopy_data['usable'] else 'ADVANCE_TO_NEXT_DIVERSE_ANCHOR'}",
                    )

        if (
            not any(candidate["usable"] for candidate in candidate_data)
            and not bool(topology_repair_info["repaired"])
        ):
            # Requested fallback: hold the selected center, shrink d until a
            # regular strict equilibrium is found, then strictly continue d
            # back toward the frozen point.  Every intermediate state is a
            # fully converged equilibrium, never a one-correction surrogate.
            fallback_state: np.ndarray | None = None
            fallback_width = float(refined_point[1])
            fallback_data: dict[str, Any] | None = None
            tried_widths: set[float] = set()
            for stage in range(int(args.init_fallback_stages)):
                fallback_width = max(
                    bounds.width_min,
                    float(refined_point[1]) * float(args.init_shrink_factor) ** (stage + 1),
                )
                key = round(fallback_width, 15)
                if key in tried_widths:
                    break
                tried_widths.add(key)
                fallback_data = evaluate_initial_candidate(
                    f"fallback_shrink_{stage + 1}",
                    np.asarray([refined_point[0], fallback_width]),
                )
                if fallback_data["usable"]:
                    fallback_state = fallback_data["state"].copy()
                    break
            if fallback_state is not None and fallback_data is not None:
                current_width = fallback_width
                for stage in range(int(args.init_fallback_stages)):
                    if current_width >= refined_point[1] - 1.0e-14 * bounds.scale[1]:
                        break
                    next_width = min(
                        float(refined_point[1]),
                        current_width / float(args.init_shrink_factor),
                    )
                    init_state.x.array[:] = fallback_state
                    init_state.x.scatter_forward()
                    # This continuation starts from the last strict state,
                    # unlike ordinary shortlist candidates which start at phi_T.
                    point_continue = np.asarray([refined_point[0], next_width])
                    prefix = f"initial_fallback_continue_{stage + 1}"
                    projection = problem.project(
                        state=init_state,
                        correction=init_correction,
                        density=init_density,
                        point=point_continue,
                        prefix=prefix,
                        phase="initialization_width_continuation",
                        outer_iteration=-2,
                    )
                    if projection.metrics is None or projection.rejection_reason:
                        break
                    if comm.rank == 0:
                        frozen = controls.frozen_window_metrics(
                            global_phi,
                            global_indicator,
                            global_weights,
                            point_continue[0],
                            point_continue[1],
                            args.eps_ratio,
                        )
                    else:
                        frozen = None
                    frozen = comm.bcast(frozen, root=0)
                    feasible = bool(
                        projection.metrics.leakage <= args.leakage_max + args.constraint_tol
                        and projection.metrics.missing <= args.missing_max + args.constraint_tol
                    )
                    coverage = 1.0 - projection.metrics.missing
                    usable = controls.initialization_candidate_is_usable(
                        strict_success=True,
                        missing=projection.metrics.missing,
                        minimum_coverage=float(args.init_min_coverage),
                    )
                    if not usable:
                        continuation_reason = (
                            f"COLLAPSED_BRANCH:COVERAGE={coverage:.6e}"
                            f"<{float(args.init_min_coverage):.6e}"
                        )
                    else:
                        continuation_reason = "ELIGIBLE" if feasible else "GEOMETRIC_SAFEGUARD"
                    row = {
                        "candidate": f"fallback_continue_{stage + 1}",
                        **control_log_values(point_continue, args.eps_ratio),
                        "frozen_leakage": frozen.leakage,
                        "frozen_missing": frozen.missing,
                        "frozen_coverage": frozen.coverage,
                        "newton_success": 1,
                        "newton_status": projection.newton.status,
                        "newton_iterations": projection.newton.iterations,
                        "damping_history": _json_array(projection.trace.damping_history),
                        "initial_dual_residual": projection.initial_residual,
                        "final_dual_residual": projection.newton.residual,
                        "first_correction_h1": projection.trace.first_correction_h1,
                        "projected_objective": projection.metrics.objective,
                        "projected_leakage": projection.metrics.leakage,
                        "projected_missing": projection.metrics.missing,
                        "activity_area_ratio": projection.metrics.activity_area_ratio,
                        "overlap_area": projection.metrics.overlap_area,
                        "overlap_area_ratio": projection.metrics.overlap_area_ratio,
                        "recall": projection.metrics.recall,
                        "precision": projection.metrics.precision,
                        "jaccard": projection.metrics.jaccard,
                        "linear_iterations": projection.trace.linear_iterations,
                        "linear_convergence_reason": projection.trace.linear_reason,
                        "coercivity_margin": projection.coercivity_margin,
                        "coercivity_error": projection.coercivity_error,
                        "selected": 0,
                        "reason": continuation_reason,
                    }
                    fallback_data = {
                        "name": row["candidate"],
                        "point": point_continue.copy(),
                        "projection": projection,
                        "state": init_state.x.array.copy(),
                        "row": row,
                        "strict_success": True,
                        "usable": usable,
                        "feasible": feasible,
                    }
                    candidate_data.append(fallback_data)
                    if not usable:
                        break
                    fallback_state = init_state.x.array.copy()
                    current_width = next_width

        successful_indices = [
            index for index, candidate in enumerate(candidate_data) if candidate["usable"]
        ]
        if int(args.verbosity) >= 3:
            shared.root_print(
                comm,
                "INITIALIZATION_FINAL_SELECTION_GUARD "
                f"evaluated={len(candidate_data)} "
                f"strictNoncollapsed={len(successful_indices)} "
                f"usableCandidateGuard="
                f"{'PASS' if successful_indices else 'REJECT'} "
                f"action={'RANK_USABLE_CANDIDATES' if successful_indices else 'WRITE_FAILURE_TABLE_AND_ABORT'}",
            )
        if not successful_indices:
            for candidate in candidate_data:
                initialization_rows.append(candidate["row"])
                if initialization_writer is not None:
                    initialization_writer.writerow(candidate["row"])
            if initialization_handle is not None:
                initialization_handle.flush()
            raise RuntimeError(
                "no initialization candidate reached a strict noncollapsed equilibrium; "
                "the missing-area safeguard forbids accepting W approximately zero"
            )

        def initialization_key(index: int) -> tuple[float, ...]:
            """Return the requested lexicographic candidate ranking key."""
            candidate = candidate_data[index]
            projection: ProjectionResult = candidate["projection"]
            metrics = projection.metrics
            margin_key = (
                -projection.coercivity_margin
                if math.isfinite(projection.coercivity_margin)
                else math.inf
            )
            geometric_violation = controls.normalized_violation(
                np.asarray(
                    [
                        metrics.leakage - args.leakage_max,
                        metrics.missing - args.missing_max,
                    ]
                )
            )
            return (
                0.0 if candidate["feasible"] else 1.0,
                geometric_violation,
                metrics.objective,
                projection.newton.residual,
                projection.predictor_corrector_h1,
                margin_key,
                float(index),
            )

        winning_index = min(successful_indices, key=initialization_key)
        if int(args.verbosity) >= 3:
            for index in successful_indices:
                candidate = candidate_data[index]
                key = initialization_key(index)
                shared.root_print(
                    comm,
                    "INITIALIZATION_FINAL_RANK_GUARD "
                    f"candidate={candidate['name']} "
                    f"feasiblePriority={key[0]:.0f} "
                    f"geometricViolation={key[1]:.12e} "
                    f"J={key[2]:.12e} residual={key[3]:.12e} "
                    f"predictorCorrection={key[4]:.12e} "
                    f"coercivityRank={key[5]:.12e} "
                    f"result={'SELECTED' if index == winning_index else 'LOWER_RANK'} "
                    f"action={'COMMIT_INITIAL_ACCEPTED_STATE' if index == winning_index else 'RETAIN_DIAGNOSTIC_ONLY'}",
                )
        candidate_data[winning_index]["row"]["selected"] = 1
        candidate_data[winning_index]["row"]["reason"] = "SELECTED"
        for index, candidate in enumerate(candidate_data):
            if index != winning_index and candidate["row"]["reason"] == "ELIGIBLE":
                candidate["row"]["reason"] = "LOWER_LEXICOGRAPHIC_RANK"
            initialization_rows.append(candidate["row"])
            if initialization_writer is not None:
                initialization_writer.writerow(candidate["row"])
        if initialization_handle is not None:
            initialization_handle.flush()

        winning = candidate_data[winning_index]
        point = np.asarray(winning["point"], dtype=np.float64)
        accepted_state = fem.Function(V, name="phi_h")
        accepted_density = fem.Function(V, name="rho_h")
        accepted_correction = fem.Function(V, name="acceptedCorrection")
        accepted_state.x.array[:] = winning["state"]
        accepted_state.x.scatter_forward()
        problem.set_control(point)
        problem.update_density(accepted_state, accepted_density)
        metrics = problem.evaluate_metrics(accepted_state)
        accepted_activity = fem.Function(V, name="acceptedActivity")
        problem.update_activity(accepted_state, accepted_activity)
        sensitivity_center = fem.Function(V, name="sensitivity_m")
        sensitivity_width = fem.Function(V, name="sensitivity_d")
        reduced = problem.sensitivities(
            state=accepted_state,
            sensitivity_center=sensitivity_center,
            sensitivity_width=sensitivity_width,
            point=point,
            iteration=0,
        )
        kkt, active_names = problem.kkt(point, metrics, reduced, bounds)
        selected_projection: ProjectionResult = winning["projection"]
        append_outer_row(
            outer_writer,
            outer_records,
            event="accepted_initial",
            iteration=0,
            trial=0,
            accepted=1,
            **control_log_values(point, args.eps_ratio),
            **metric_log_values(metrics, args),
            **reduced_log_values(reduced),
            **projection_log_values(selected_projection, target_energy),
            trust_radius=args.trust_radius,
            active_constraints=";".join(active_names),
            kkt_residual=kkt.residual,
            kkt_stationarity=kkt.stationarity,
            kkt_primal=kkt.primal_infeasibility,
            kkt_complementarity=kkt.complementarity,
            reason="INITIALIZER_WINNER",
        )
        if outer_handle is not None:
            outer_handle.flush()
        shared.root_print(
            comm,
            f"INITIAL_ACCEPTED candidate={winning['name']} m={point[0]:.12e} d={point[1]:.12e} "
            f"J={metrics.objective:.12e} L={metrics.leakage:.12e} M={metrics.missing:.12e} "
            f"residual={selected_projection.newton.residual:.12e} KKT={kkt.residual:.12e}",
        )
        if args.plot_accepted_states:
            emit_interactive_plot(
                state=accepted_state,
                density=accepted_density,
                stage="ACCEPTED",
                iteration=0,
                residual=selected_projection.newton.residual,
                metrics_for_plot=metrics,
            )

        if args.verify_reduced_derivatives:
            fd_rows = verify_reduced_derivatives(
                problem=problem,
                point=point,
                state=accepted_state,
                sensitivity_center=sensitivity_center,
                sensitivity_width=sensitivity_width,
                reduced=reduced,
                bounds=bounds,
                path=logs_dir / "reduced_derivative_verification.json",
            )
            if comm.rank == 0:
                checked = [row for row in fd_rows if row.get("status") == "OK"]
                maximum = max((float(row["relative_error"]) for row in checked), default=math.nan)
                shared.root_print(comm, f"REDUCED_DERIVATIVE_CHECK maxRelativeError={maximum:.6e}")

        trial_state = fem.Function(V, name="trialPhi")
        trial_density = fem.Function(V, name="trialDensity")
        trial_correction = fem.Function(V, name="trialCorrection")
        predictor = fem.Function(V, name="trialPredictor")
        trust_radius = float(args.trust_radius)
        filter_entries: list[tuple[float, float]] = [
            (
                metrics.objective,
                controls.normalized_violation(
                    np.asarray([metrics.leakage - args.leakage_max, metrics.missing - args.missing_max])
                ),
            )
        ]
        final_status = "MAX_OPT_IT"
        final_stagnation_reason = ""
        functional_stagnation_count = 0
        accepted_iteration = 0

        def stagnation_status(feasible: bool) -> str:
            """Name a valid stationary terminal state without overstating feasibility."""
            return "STAGNATED_FEASIBLE" if feasible else "STAGNATED_INFEASIBLE"

        for _outer_attempt in range(int(args.max_opt_it)):
            current_violation = controls.normalized_violation(
                np.asarray([metrics.leakage - args.leakage_max, metrics.missing - args.missing_max])
            )
            currently_feasible = current_violation <= float(args.constraint_tol)
            kkt, active_names = problem.kkt(point, metrics, reduced, bounds)
            if int(args.verbosity) >= 3:
                shared.root_print(
                    comm,
                    "OUTER_ITERATION_POLICY "
                    f"attempt={_outer_attempt + 1}/{int(args.max_opt_it)} "
                    f"acceptedIteration={accepted_iteration} "
                    f"mode={'FEASIBLE_OBJECTIVE_SQP' if currently_feasible else 'GEOMETRIC_RESTORATION'} "
                    f"m={point[0]:.12e} d={point[1]:.12e} "
                    f"J={metrics.objective:.12e} "
                    f"L={metrics.leakage:.12e} M={metrics.missing:.12e} "
                    f"violation={current_violation:.12e} "
                    f"feasibilityGuard="
                    f"{'PASS' if currently_feasible else 'REJECT_RESTORE'} "
                    f"kktGuard="
                    f"{'PASS_STOP' if currently_feasible and kkt.residual <= float(args.kkt_tol) else 'CONTINUE'} "
                    f"KKT={kkt.residual:.12e} "
                    f"trust={trust_radius:.6e} "
                    f"action={'TERMINATE_REDUCED_KKT' if currently_feasible and kkt.residual <= float(args.kkt_tol) else 'SOLVE_NEXT_2D_SUBPROBLEM'}",
                )
            if currently_feasible and kkt.residual <= float(args.kkt_tol):
                final_status = "CONVERGED_REDUCED_KKT"
                break
            accepted_this_iteration = False
            for trial_number in range(1, int(args.max_trials_per_iteration) + 1):
                base_rows, base_rhs, _ = controls.linear_step_constraints(
                    point, bounds, trust_radius
                )
                if comm.rank == 0:
                    if currently_feasible:
                        rows = np.vstack(
                            (base_rows, reduced.gradient_leakage, reduced.gradient_missing)
                        )
                        rhs = np.concatenate(
                            (
                                base_rhs,
                                [
                                    args.leakage_max - metrics.leakage,
                                    args.missing_max - metrics.missing,
                                ],
                            )
                        )
                        subproblem = controls.solve_convex_qp_2d(
                            reduced.gradient_objective,
                            reduced.hessian,
                            rows,
                            rhs,
                        )
                        step = subproblem.step
                        subproblem_reason = subproblem.reason
                        subproblem_success = subproblem.success
                    else:
                        signed_geom = np.asarray(
                            [metrics.leakage - args.leakage_max, metrics.missing - args.missing_max]
                        )
                        geom_grad = np.vstack(
                            (reduced.gradient_leakage, reduced.gradient_missing)
                        )
                        restoration = controls.solve_restoration_step_2d(
                            reduced.gradient_objective,
                            reduced.hessian,
                            signed_geom,
                            geom_grad,
                            base_rows,
                            base_rhs,
                        )
                        step = restoration.step
                        subproblem_reason = restoration.reason
                        subproblem_success = restoration.success
                    proposal = (
                        bool(subproblem_success),
                        np.asarray(step, dtype=np.float64),
                        str(subproblem_reason),
                    )
                else:
                    proposal = None
                subproblem_success, step, subproblem_reason = comm.bcast(proposal, root=0)
                step = np.asarray(step, dtype=np.float64)
                trial_point = point + step
                predicted_objective_change = float(
                    reduced.gradient_objective @ step
                    + 0.5 * step @ reduced.hessian @ step
                )
                predicted_reduction = -predicted_objective_change
                predicted_leakage = metrics.leakage + float(reduced.gradient_leakage @ step)
                predicted_missing = metrics.missing + float(reduced.gradient_missing @ step)
                predicted_violation = controls.normalized_violation(
                    np.asarray(
                        [predicted_leakage - args.leakage_max, predicted_missing - args.missing_max]
                    )
                )
                scaled_step = float(np.linalg.norm(step / bounds.scale, ord=np.inf))
                if not subproblem_success:
                    rejection = f"SUBPROBLEM:{subproblem_reason}"
                elif scaled_step <= 10.0 * np.finfo(float).eps:
                    rejection = "ZERO_CONTROL_STEP_WITH_NONZERO_KKT"
                elif currently_feasible and (
                    predicted_leakage > args.leakage_max + args.constraint_tol
                    or predicted_missing > args.missing_max + args.constraint_tol
                ):
                    rejection = "PREDICTED_GEOMETRIC_CROSSING"
                else:
                    rejection = ""
                if int(args.verbosity) >= 3:
                    predicted_geometry_enforced = currently_feasible
                    shared.root_print(
                        comm,
                        "OUTER_PREPROJECTION_GUARD "
                        f"k={accepted_iteration} trial={trial_number} "
                        f"mode={'FEASIBLE_OBJECTIVE_SQP' if currently_feasible else 'GEOMETRIC_RESTORATION'} "
                        f"subproblemGuard="
                        f"{'PASS' if subproblem_success else 'REJECT'} "
                        f"subproblemReason={subproblem_reason} "
                        f"boxGuard="
                        f"{'PASS' if bounds.contains(*trial_point, tolerance=1.0e-12) else 'REJECT'} "
                        f"trustGuard="
                        f"{'PASS' if scaled_step <= trust_radius + 1.0e-12 else 'REJECT'} "
                        f"nonzeroStepGuard="
                        f"{'PASS' if scaled_step > 10.0 * np.finfo(float).eps else 'REJECT'} "
                        f"predictedL={predicted_leakage:.12e} "
                        f"predictedM={predicted_missing:.12e} "
                        f"predictedLeakageGuard="
                        f"{('PASS' if predicted_leakage <= args.leakage_max + args.constraint_tol else 'REJECT') if predicted_geometry_enforced else 'DIAGNOSTIC_ONLY'} "
                        f"predictedMissingGuard="
                        f"{('PASS' if predicted_missing <= args.missing_max + args.constraint_tol else 'REJECT') if predicted_geometry_enforced else 'DIAGNOSTIC_ONLY'} "
                        f"reason={rejection or 'NONE'} "
                        f"action={'REJECT_WITHOUT_PDE_AND_SHRINK_TRUST' if rejection else 'RUN_STRICT_NEWTON_PROJECTION'}",
                    )
                if rejection:
                    append_outer_row(
                        outer_writer,
                        outer_records,
                        event="rejected_preprojection",
                        iteration=accepted_iteration,
                        trial=trial_number,
                        accepted=0,
                        **control_log_values(trial_point, args.eps_ratio),
                        **metric_log_values(metrics, args),
                        **reduced_log_values(reduced),
                        step_center=step[0],
                        step_width=step[1],
                        scaled_step_inf=scaled_step,
                        trust_radius=trust_radius,
                        predicted_objective_reduction=predicted_reduction,
                        predicted_leakage=predicted_leakage,
                        predicted_missing=predicted_missing,
                        predicted_geometric_violation=predicted_violation,
                        active_constraints=";".join(active_names),
                        kkt_residual=kkt.residual,
                        kkt_stationarity=kkt.stationarity,
                        kkt_primal=kkt.primal_infeasibility,
                        kkt_complementarity=kkt.complementarity,
                        reason=rejection,
                    )
                    if rejection == "ZERO_CONTROL_STEP_WITH_NONZERO_KKT":
                        final_status = stagnation_status(currently_feasible)
                        final_stagnation_reason = "ZERO_REDUCED_CONTROL_STEP"
                        if int(args.verbosity) >= 1:
                            shared.root_print(
                                comm,
                                "OUTER_FUNCTIONAL_STAGNATION "
                                f"k={accepted_iteration} mode="
                                f"{'scaled_objective' if currently_feasible else 'geometric_violation'} "
                                "reason=ZERO_REDUCED_CONTROL_STEP "
                                f"J={metrics.objective:.12e} "
                                f"L={metrics.leakage:.12e} M={metrics.missing:.12e} "
                                f"violation={current_violation:.12e}",
                            )
                        if outer_handle is not None:
                            outer_handle.flush()
                        break
                    trust_radius = min(
                        trust_radius * float(args.trust_shrink),
                        (
                            scaled_step * float(args.trust_shrink)
                            if scaled_step > 0.0
                            else math.inf
                        ),
                    )
                    if outer_handle is not None:
                        outer_handle.flush()
                    if int(args.verbosity) >= 2:
                        shared.root_print(
                            comm,
                            f"OUTER_REJECT_PREPROJECTION k={accepted_iteration} "
                            f"trial={trial_number} dm={step[0]:.12e} dd={step[1]:.12e} "
                            f"predictedDeltaJ={predicted_objective_change:.12e} "
                            f"predictedL={predicted_leakage:.12e} "
                            f"predictedM={predicted_missing:.12e} "
                            f"trust={trust_radius:.6e} reason={rejection}",
                        )
                    if trust_radius < float(args.trust_radius_min):
                        final_status = stagnation_status(currently_feasible)
                        final_stagnation_reason = (
                            f"TRUST_RADIUS_MINIMUM_AFTER_{rejection}"
                        )
                        break
                    continue

                predictor_usable, _ = problem.set_predictor(
                    predictor,
                    accepted_state,
                    sensitivity_center,
                    sensitivity_width,
                    step,
                )
                _copy_function(predictor, trial_state)
                problem.set_control(point)
                problem.update_activity(accepted_state, accepted_activity)
                projection = problem.project(
                    state=trial_state,
                    correction=trial_correction,
                    density=trial_density,
                    point=trial_point,
                    prefix=f"outer_{accepted_iteration}_trial_{trial_number}",
                    predictor=predictor,
                    accepted_state=accepted_state,
                    reference_activity=accepted_activity,
                    phase="outer_trial_strict",
                    outer_iteration=accepted_iteration,
                    max_newton_it=int(args.trial_max_newton_it),
                )
                branch_reason = branch_acceptance_reason(projection, target_energy, args)
                trial_metrics = projection.metrics
                actual_reduction = (
                    metrics.objective - trial_metrics.objective
                    if trial_metrics is not None
                    else math.nan
                )
                ratio = (
                    actual_reduction / predicted_reduction
                    if trial_metrics is not None and predicted_reduction > 1.0e-16
                    else math.nan
                )
                actual_violation = (
                    controls.normalized_violation(
                        np.asarray(
                            [
                                trial_metrics.leakage - args.leakage_max,
                                trial_metrics.missing - args.missing_max,
                            ]
                        )
                    )
                    if trial_metrics is not None
                    else math.inf
                )
                filter_ok: bool | None = None
                if branch_reason:
                    acceptance_reason = branch_reason
                    accepted = False
                elif currently_feasible:
                    actual_feasible = bool(
                        trial_metrics.leakage <= args.leakage_max + args.constraint_tol
                        and trial_metrics.missing <= args.missing_max + args.constraint_tol
                    )
                    if not actual_feasible:
                        accepted = False
                        acceptance_reason = "ACTUAL_GEOMETRIC_CROSSING"
                    elif not math.isfinite(ratio) or ratio < float(args.acceptance_eta):
                        accepted = False
                        acceptance_reason = f"OBJECTIVE_RATIO:{ratio:.6e}"
                    else:
                        accepted = True
                        acceptance_reason = "OBJECTIVE_MODEL_ACCEPTED"
                else:
                    required = float(args.restoration_fraction) * max(current_violation, 1.0e-14)
                    filter_ok = controls.filter_accepts(
                        trial_metrics.objective,
                        actual_violation,
                        filter_entries,
                        objective_margin=float(args.filter_objective_margin),
                        violation_margin=float(args.filter_violation_margin),
                    )
                    accepted = bool(
                        actual_violation <= current_violation - required and filter_ok
                    )
                    acceptance_reason = (
                        "RESTORATION_FILTER_ACCEPTED"
                        if accepted
                        else "INSUFFICIENT_RESTORATION_OR_FILTER"
                    )
                logged_reason = (
                    acceptance_reason
                    if predictor_usable
                    else f"{acceptance_reason};PREDICTOR_FALLBACK_ACCEPTED_STATE"
                )
                if int(args.verbosity) >= 3:
                    scale = math.sqrt(target_energy)
                    correction_relative = (
                        projection.predictor_corrector_h1 / scale
                        if math.isfinite(projection.predictor_corrector_h1)
                        else math.nan
                    )
                    predicted_relative = (
                        projection.predicted_state_change_h1 / scale
                        if math.isfinite(projection.predicted_state_change_h1)
                        else math.nan
                    )
                    allowed_correction = max(
                        float(args.predictor_correction_absolute),
                        float(args.predictor_correction_factor)
                        * max(predicted_relative, 1.0e-14),
                    )
                    topology_guard = (
                        "NOT_EVALUATED"
                        if projection.topology is None
                        else (
                            "PASS"
                            if projection.topology.component_count
                            == problem.expected_topology_components
                            else "REJECT"
                        )
                    )
                    actual_geometry_guard = (
                        "NOT_EVALUATED"
                        if trial_metrics is None
                        else (
                            "PASS"
                            if trial_metrics.leakage
                            <= args.leakage_max + args.constraint_tol
                            and trial_metrics.missing
                            <= args.missing_max + args.constraint_tol
                            else "REJECT"
                        )
                    )
                    if currently_feasible:
                        acceptance_merit_guard = (
                            "PASS"
                            if math.isfinite(ratio)
                            and ratio >= float(args.acceptance_eta)
                            else "REJECT"
                        )
                        filter_guard = "NOT_APPLICABLE"
                    else:
                        required = float(args.restoration_fraction) * max(
                            current_violation, 1.0e-14
                        )
                        acceptance_merit_guard = (
                            "PASS"
                            if actual_violation <= current_violation - required
                            else "REJECT"
                        )
                        filter_guard = "PASS" if filter_ok else "REJECT"
                    shared.root_print(
                        comm,
                        "OUTER_POSTPROJECTION_GUARD "
                        f"k={accepted_iteration} trial={trial_number} "
                        f"strictNewtonGuard="
                        f"{'PASS' if projection.newton.converged and projection.newton.residual <= float(args.newton_tol) else 'REJECT'} "
                        f"residual={projection.newton.residual:.12e} "
                        f"predictorUsableGuard="
                        f"{'PASS' if predictor_usable else 'FALLBACK'} "
                        f"predictorCorrectionGuard="
                        f"{'PASS' if math.isfinite(correction_relative) and correction_relative <= allowed_correction else 'REJECT'} "
                        f"predictorCorrection={correction_relative:.6e} "
                        f"predictorAllowed={allowed_correction:.6e} "
                        f"overlapGuard="
                        f"{'PASS' if math.isfinite(projection.branch_overlap) and projection.branch_overlap >= float(args.branch_overlap_min) else 'REJECT'} "
                        f"overlap={projection.branch_overlap:.6e} "
                        f"topologyGuard={topology_guard} "
                        f"actualGeometryGuard={actual_geometry_guard} "
                        f"modelOrRestorationGuard={acceptance_merit_guard} "
                        f"filterGuard={filter_guard} "
                        f"ratio={ratio:.6e} "
                        f"oldViolation={current_violation:.6e} "
                        f"newViolation={actual_violation:.6e} "
                        f"reason={logged_reason} "
                        f"action={'COMMIT_TRIAL_AS_ACCEPTED_STATE' if accepted else 'PRESERVE_ACCEPTED_STATE_AND_SHRINK_TRUST'}",
                    )
                if not accepted:
                    append_outer_row(
                        outer_writer,
                        outer_records,
                        event="rejected_trial",
                        iteration=accepted_iteration,
                        trial=trial_number,
                        accepted=0,
                        **control_log_values(trial_point, args.eps_ratio),
                        **metric_log_values(trial_metrics, args),
                        **reduced_log_values(reduced),
                        **projection_log_values(projection, target_energy),
                        step_center=step[0],
                        step_width=step[1],
                        scaled_step_inf=scaled_step,
                        trust_radius=trust_radius,
                        predicted_objective_reduction=predicted_reduction,
                        actual_objective_reduction=actual_reduction,
                        predicted_leakage=predicted_leakage,
                        predicted_missing=predicted_missing,
                        actual_leakage=trial_metrics.leakage if trial_metrics else "",
                        actual_missing=trial_metrics.missing if trial_metrics else "",
                        acceptance_ratio=ratio,
                        predicted_geometric_violation=predicted_violation,
                        active_constraints=";".join(active_names),
                        kkt_residual=kkt.residual,
                        kkt_stationarity=kkt.stationarity,
                        kkt_primal=kkt.primal_infeasibility,
                        kkt_complementarity=kkt.complementarity,
                        reason=logged_reason,
                    )
                    if outer_handle is not None:
                        outer_handle.flush()
                    if int(args.verbosity) >= 2:
                        actual_text = (
                            f"J={trial_metrics.objective:.12e} "
                            f"L={trial_metrics.leakage:.12e} M={trial_metrics.missing:.12e}"
                            if trial_metrics is not None
                            else "J=unavailable L=unavailable M=unavailable"
                        )
                        shared.root_print(
                            comm,
                            f"OUTER_REJECT k={accepted_iteration} trial={trial_number} "
                            f"m={trial_point[0]:.12e} d={trial_point[1]:.12e} "
                            f"{actual_text} predictedReduction={predicted_reduction:.12e} "
                            f"actualReduction={actual_reduction:.12e} ratio={ratio:.6e} "
                            f"trust={trust_radius:.6e} reason={logged_reason}",
                        )
                    # If the QP minimizer was strictly inside a much larger
                    # radius, shrinking only the old radius would propose the
                    # identical rejected step many times.  Standard trust-
                    # region contraction is tied to the rejected step norm.
                    trust_radius = min(
                        trust_radius * float(args.trust_shrink),
                        scaled_step * float(args.trust_shrink),
                    )
                    if trust_radius < float(args.trust_radius_min):
                        final_status = stagnation_status(currently_feasible)
                        final_stagnation_reason = (
                            "TRUST_RADIUS_MINIMUM_WITHOUT_ACCEPTABLE_STRICT_TRIAL"
                        )
                        break
                    continue

                # The only state mutation of the accepted branch occurs here,
                # after strict Newton, actual constraints, filter/model, branch,
                # predictor, and optional coercivity checks all passed.
                functional_progress = controls.threshold_functional_progress(
                    old_objective=metrics.objective,
                    new_objective=trial_metrics.objective,
                    old_leakage=metrics.leakage,
                    new_leakage=trial_metrics.leakage,
                    old_missing=metrics.missing,
                    new_missing=trial_metrics.missing,
                    leakage_max=float(args.leakage_max),
                    missing_max=float(args.missing_max),
                    feasibility_tolerance=float(args.constraint_tol),
                    absolute_tolerance=float(args.functional_stagnation_atol),
                    relative_tolerance=float(args.functional_stagnation_rtol),
                )
                if functional_progress.stagnated:
                    functional_stagnation_count += 1
                else:
                    functional_stagnation_count = 0
                if int(args.verbosity) >= 3:
                    shared.root_print(
                        comm,
                        "OUTER_ACCEPTED_STATE_COMMIT "
                        f"oldIteration={accepted_iteration} "
                        f"newIteration={accepted_iteration + 1} "
                        f"primaryMeritMode={functional_progress.mode} "
                        f"oldMerit={functional_progress.old_merit:.12e} "
                        f"newMerit={functional_progress.new_merit:.12e} "
                        f"improvement={functional_progress.improvement:.12e} "
                        f"required={functional_progress.required_improvement:.12e} "
                        f"stagnationGuard="
                        f"{'TRIGGER_COUNT_INCREMENT' if functional_progress.stagnated else 'PASS_RESET_COUNT'} "
                        "stateMutationGuard=ALL_TRIAL_GUARDS_PASSED "
                        "action=OVERWRITE_ACCEPTED_STATE_AND_RECOMPUTE_SENSITIVITIES",
                    )
                _copy_function(trial_state, accepted_state)
                _copy_function(trial_density, accepted_density)
                point = trial_point
                metrics = trial_metrics
                accepted_iteration += 1
                filter_entries = controls.update_filter(
                    filter_entries, metrics.objective, actual_violation
                )
                proposal_trust_radius = trust_radius
                if (
                    math.isfinite(ratio)
                    and ratio >= float(args.acceptance_grow_eta)
                    and scaled_step >= 0.8 * trust_radius
                ) or (not currently_feasible and actual_violation <= 0.5 * current_violation):
                    trust_radius = min(
                        float(args.trust_radius_max), trust_radius * float(args.trust_grow)
                    )
                problem.set_control(point)
                problem.update_activity(accepted_state, accepted_activity)
                reduced = problem.sensitivities(
                    state=accepted_state,
                    sensitivity_center=sensitivity_center,
                    sensitivity_width=sensitivity_width,
                    point=point,
                    iteration=accepted_iteration,
                )
                kkt, active_names = problem.kkt(point, metrics, reduced, bounds)
                append_outer_row(
                    outer_writer,
                    outer_records,
                    event="accepted_iteration",
                    iteration=accepted_iteration,
                    trial=trial_number,
                    accepted=1,
                    **control_log_values(point, args.eps_ratio),
                    **metric_log_values(metrics, args),
                    **reduced_log_values(reduced),
                    **projection_log_values(projection, target_energy),
                    step_center=step[0],
                    step_width=step[1],
                    scaled_step_inf=scaled_step,
                    trust_radius=proposal_trust_radius,
                    predicted_objective_reduction=predicted_reduction,
                    actual_objective_reduction=actual_reduction,
                    predicted_leakage=predicted_leakage,
                    predicted_missing=predicted_missing,
                    actual_leakage=metrics.leakage,
                    actual_missing=metrics.missing,
                    acceptance_ratio=ratio,
                    predicted_geometric_violation=predicted_violation,
                    primary_merit_mode=functional_progress.mode,
                    old_primary_merit=functional_progress.old_merit,
                    new_primary_merit=functional_progress.new_merit,
                    primary_merit_improvement=functional_progress.improvement,
                    primary_merit_required_improvement=(
                        functional_progress.required_improvement
                    ),
                    functional_stagnation_count=functional_stagnation_count,
                    active_constraints=";".join(active_names),
                    kkt_residual=kkt.residual,
                    kkt_stationarity=kkt.stationarity,
                    kkt_primal=kkt.primal_infeasibility,
                    kkt_complementarity=kkt.complementarity,
                    reason=logged_reason,
                )
                if outer_handle is not None:
                    outer_handle.flush()
                shared.root_print(
                    comm,
                    f"OUTER_ACCEPT k={accepted_iteration} trial={trial_number} "
                    f"m={point[0]:.12e} d={point[1]:.12e} J={metrics.objective:.12e} "
                    f"L={metrics.leakage:.12e} M={metrics.missing:.12e} "
                    f"ratio={ratio:.6e} trust={trust_radius:.6e} KKT={kkt.residual:.6e}",
                )
                if int(args.verbosity) >= 1:
                    shared.root_print(
                        comm,
                        "OUTER_FUNCTIONAL_PROGRESS "
                        f"k={accepted_iteration} mode={functional_progress.mode} "
                        f"oldMerit={functional_progress.old_merit:.12e} "
                        f"newMerit={functional_progress.new_merit:.12e} "
                        f"improvement={functional_progress.improvement:.12e} "
                        f"required={functional_progress.required_improvement:.12e} "
                        f"stagnated={int(functional_progress.stagnated)} "
                        f"count={functional_stagnation_count}/"
                        f"{int(args.functional_stagnation_patience)}",
                    )
                if args.plot_accepted_states:
                    emit_interactive_plot(
                        state=accepted_state,
                        density=accepted_density,
                        stage="ACCEPTED",
                        iteration=accepted_iteration,
                        residual=projection.newton.residual,
                        metrics_for_plot=metrics,
                    )
                if functional_stagnation_count >= int(
                    args.functional_stagnation_patience
                ):
                    final_status = stagnation_status(
                        actual_violation <= float(args.constraint_tol)
                    )
                    final_stagnation_reason = "ACCEPTED_PRIMARY_MERIT_STAGNATION"
                    shared.root_print(
                        comm,
                        "OUTER_FUNCTIONAL_STAGNATION "
                        f"k={accepted_iteration} mode={functional_progress.mode} "
                        f"count={functional_stagnation_count} "
                        f"J={metrics.objective:.12e} L={metrics.leakage:.12e} "
                        f"M={metrics.missing:.12e} "
                        f"violation={actual_violation:.12e}",
                    )
                accepted_this_iteration = True
                break
            if final_status.startswith("STAGNATED_"):
                if final_stagnation_reason.startswith("TRUST_RADIUS_MINIMUM"):
                    shared.root_print(
                        comm,
                        "OUTER_FUNCTIONAL_STAGNATION "
                        f"k={accepted_iteration} mode="
                        f"{'scaled_objective' if currently_feasible else 'geometric_violation'} "
                        f"reason={final_stagnation_reason} "
                        f"J={metrics.objective:.12e} L={metrics.leakage:.12e} "
                        f"M={metrics.missing:.12e} violation={current_violation:.12e} "
                        f"trust={trust_radius:.12e}",
                    )
                break
            if not accepted_this_iteration:
                final_status = stagnation_status(currently_feasible)
                final_stagnation_reason = "NO_ACCEPTABLE_STRICT_TRIAL"
                shared.root_print(
                    comm,
                    "OUTER_FUNCTIONAL_STAGNATION "
                    f"k={accepted_iteration} mode="
                    f"{'scaled_objective' if currently_feasible else 'geometric_violation'} "
                    "reason=NO_ACCEPTABLE_STRICT_TRIAL "
                    f"J={metrics.objective:.12e} L={metrics.leakage:.12e} "
                    f"M={metrics.missing:.12e} violation={current_violation:.12e}",
                )
                break
        else:
            final_status = "MAX_OPT_IT"

        # Mandatory final strict projection and derivative/KKT recomputation.
        final_seed = accepted_state.x.array.copy()
        _copy_function(accepted_state, predictor)
        problem.set_control(point)
        problem.update_activity(accepted_state, accepted_activity)
        final_projection = problem.project(
            state=accepted_state,
            correction=accepted_correction,
            density=accepted_density,
            point=point,
            prefix="final_strict_projection",
            predictor=predictor,
            accepted_state=predictor,
            reference_activity=accepted_activity,
            phase="final_strict_projection",
            outer_iteration=accepted_iteration,
        )
        final_branch_reason = branch_acceptance_reason(
            final_projection, target_energy, args
        )
        if final_projection.metrics is None or final_branch_reason:
            accepted_state.x.array[:] = final_seed
            accepted_state.x.scatter_forward()
            problem.set_control(point)
            problem.update_density(accepted_state, accepted_density)
            metrics = problem.evaluate_metrics(accepted_state)
            final_status = f"FINAL_PROJECTION_FAILED:{final_branch_reason or final_projection.newton.status}"
            return_code = 3
        else:
            metrics = final_projection.metrics
            reduced = problem.sensitivities(
                state=accepted_state,
                sensitivity_center=sensitivity_center,
                sensitivity_width=sensitivity_width,
                point=point,
                iteration=-1,
            )
            kkt, active_names = problem.kkt(point, metrics, reduced, bounds)
            feasible_final = bool(
                metrics.leakage <= args.leakage_max + args.constraint_tol
                and metrics.missing <= args.missing_max + args.constraint_tol
            )
            if final_status.startswith("STAGNATED_"):
                final_status = stagnation_status(feasible_final)
            if feasible_final and kkt.residual <= args.kkt_tol:
                final_status = "CONVERGED_REDUCED_KKT"
            elif final_status == "CONVERGED_REDUCED_KKT":
                final_status = "FINAL_KKT_NOT_CONVERGED"
            append_outer_row(
                outer_writer,
                outer_records,
                event="final_strict",
                iteration=accepted_iteration,
                trial=0,
                accepted=1,
                **control_log_values(point, args.eps_ratio),
                **metric_log_values(metrics, args),
                **reduced_log_values(reduced),
                **projection_log_values(final_projection, target_energy),
                trust_radius=trust_radius,
                active_constraints=";".join(active_names),
                kkt_residual=kkt.residual,
                kkt_stationarity=kkt.stationarity,
                kkt_primal=kkt.primal_infeasibility,
                kkt_complementarity=kkt.complementarity,
                reason=(
                    final_status
                    if not final_stagnation_reason
                    else f"{final_status}:{final_stagnation_reason}"
                ),
            )
            valid_terminal_statuses = {
                "CONVERGED_REDUCED_KKT",
                "STAGNATED_FEASIBLE",
                "STAGNATED_INFEASIBLE",
            }
            if args.fail_on_nonconvergence and final_status not in valid_terminal_statuses:
                return_code = 2
        if outer_handle is not None:
            outer_handle.flush()

        problem.set_control(point)
        problem.update_density(accepted_state, accepted_density)
        problem.update_activity(accepted_state, accepted_activity)
        feasible_final = bool(
            metrics.leakage <= args.leakage_max + args.constraint_tol
            and metrics.missing <= args.missing_max + args.constraint_tol
        )
        final_topology = problem.activity_topology(accepted_activity)
        checkpoint_metadata = {
            "mode": "torsion_h1_projection_local_branch",
            "status": final_status,
            "center": float(point[0]),
            "width": float(point[1]),
            "c1": controls.thresholds_from_center_width(*point)[0],
            "c2": controls.thresholds_from_center_width(*point)[1],
            "epsilon": float(args.eps_ratio * point[1]),
            "newton_tolerance": float(args.newton_tol),
            "residual_norm": "stiffness_dual",
            "final_dual_residual": _finite_or_none(final_projection.newton.residual),
            "final_newton_iterations": int(final_projection.newton.iterations),
            "objective": metrics.objective,
            "relative_h1_distance": math.sqrt(max(2.0 * metrics.objective, 0.0)),
            "target_energy": target_energy,
            "target_area": target_area,
            "leakage": metrics.leakage,
            "missing": metrics.missing,
            "coverage": 1.0 - metrics.missing,
            "activity_area_ratio": metrics.activity_area_ratio,
            "overlap_area": metrics.overlap_area,
            "precision": metrics.precision,
            "jaccard": metrics.jaccard,
            "topology_expected_components": int(
                problem.expected_topology_components
            ),
            "topology_components": int(final_topology.component_count),
            "topology_raw_components": int(final_topology.raw_component_count),
            "topology_largest_fraction": float(final_topology.largest_fraction),
            "topology_second_fraction": float(final_topology.second_fraction),
            "coercivity_margin": _finite_or_none(final_projection.coercivity_margin),
            "coercivity_error": _finite_or_none(final_projection.coercivity_error),
            "kkt_residual": kkt.residual,
            "kkt_stationarity": kkt.stationarity,
            "kkt_primal_infeasibility": kkt.primal_infeasibility,
            "kkt_complementarity": kkt.complementarity,
            "active_constraints": list(active_names),
            "geometric_feasible": feasible_final,
            "functional_stagnation_count": int(functional_stagnation_count),
            "functional_stagnation_reason": final_stagnation_reason or None,
            "functional_stagnation_atol": float(args.functional_stagnation_atol),
            "functional_stagnation_rtol": float(args.functional_stagnation_rtol),
            "functional_stagnation_patience": int(
                args.functional_stagnation_patience
            ),
            "local_branch_only": True,
            "global_convergence_claim": False,
        }
        shared.write_equilibrium_checkpoint(
            out_dir / "final_equilibrium.npz",
            phi=accepted_state,
            rho=accepted_density,
            metadata=checkpoint_metadata,
        )
        if args.write_xdmf:
            write_visualization(
                problem=problem,
                torsion=torsion,
                target_mask=target_mask,
                final_state=accepted_state,
                final_density=accepted_density,
                run_dir=run_dir,
            )
        if args.plot_final:
            original_plot_mode = args.plot_mode
            if args.plot_final_blocking:
                args.plot_mode = "blocking"
            try:
                emit_interactive_plot(
                    state=accepted_state,
                    density=accepted_density,
                    stage="FINAL",
                    iteration=accepted_iteration,
                    residual=final_projection.newton.residual,
                    metrics_for_plot=metrics,
                    state_title="final equilibrium phi_h",
                    density_title="final density W(phi_h)",
                )
            finally:
                args.plot_mode = original_plot_mode
        elapsed_local = time.perf_counter() - total_started
        elapsed = float(comm.allreduce(elapsed_local, op=MPI.MAX))
        summary = {
            **checkpoint_metadata,
            "run_directory": str(run_dir),
            "mesh": str(mesh_path),
            "geometry_mode": geometry_mode,
            "cells": nt,
            "degrees_of_freedom": ndof,
            "order": int(args.order),
            "quadrature_degree": qdeg,
            "mpi_ranks": int(comm.size),
            "accepted_iterations": accepted_iteration,
            "trust_radius_final": trust_radius,
            "elapsed_seconds": elapsed,
            "assumptions": [
                "fixed mesh throughout optimization",
                "one locally selected equilibrium branch tracked by sensitivity prediction",
                "no assertion of global closest-equilibrium uniqueness or global convergence",
            ],
            "unavailable_diagnostics": (
                ["generalized Jacobian coercivity (disabled)"]
                if not args.report_coercivity
                else (
                    []
                    if math.isfinite(final_projection.coercivity_margin)
                    else ["generalized Jacobian coercivity (eigensolve unavailable or failed)"]
                )
            ),
        }
        if comm.rank == 0:
            (out_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            if args.make_plots:
                write_summary_plot(
                    outer_records,
                    args,
                    run_dir / "plots" / "optimization_summary.png",
                )
        shared.root_print(
            comm,
            f"FINAL status={final_status} m={point[0]:.12e} d={point[1]:.12e} "
            f"J={metrics.objective:.12e} sqrt2J={math.sqrt(2.0 * metrics.objective):.12e} "
            f"L={metrics.leakage:.12e}/{args.leakage_max:.12e} "
            f"M={metrics.missing:.12e}/{args.missing_max:.12e} "
            f"feasible={int(feasible_final)} "
            f"components={final_topology.component_count}/"
            f"{problem.expected_topology_components} "
            f"residual={final_projection.newton.residual:.12e} KKT={kkt.residual:.12e} "
            f"stoppingReason={final_stagnation_reason or 'REDUCED_KKT_OR_ITERATION_STATUS'}",
        )
        shared.root_print(
            comm,
            "LOCAL_BRANCH_SCOPE successful optimization does not prove global convergence, "
            "global uniqueness, or global closest-equilibrium optimality",
        )
        if terminal_log_capture is not None:
            shared.root_print(
                comm,
                f"TERMINAL_LOG_COMPLETE path={terminal_log_path} exitCode={return_code}",
            )
            comm.barrier()
        return return_code
    finally:
        if plotter is not None:
            plotter._reset_mpi_live_plotter()
        if homotopy_workspace is not None:
            homotopy_workspace.close()
        if problem is not None:
            problem.close()
        if stiffness_solver is not None:
            stiffness_solver.close()
        for handle in (
            initialization_handle,
            outer_handle,
            homotopy_threshold_handle,
            newton_handle,
        ):
            if handle is not None:
                handle.close()
        if terminal_log_capture is not None and sys.exc_info()[0] is None:
            terminal_log_capture.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the H0-1 projection optimizer."""
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
