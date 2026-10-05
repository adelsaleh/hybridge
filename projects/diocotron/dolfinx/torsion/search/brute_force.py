#!/usr/bin/env python3
"""MPI-parallel brute-force initializer for torsion window optimization.

The search evaluates a Cartesian threshold grid with several independent MPI
subcommunicators.  Every subcommunicator owns a full distributed mesh and
projects one threshold pair at a time with damped Newton.  Different groups
work concurrently, while neighboring candidates within a group reuse the last
converged state.  A refined grid is centered on the best result after each
level.

The grid objective is

    leakage / target_area
    + missing_weight * missing / target_area
    + area_weight * abs(activity_area / target_area - 1).

Only candidates reaching the requested grid Newton tolerance are preferred.
The run subcommand first launches the parallel search, then invokes the normal
reduced optimizer with the selected pair at a separately configurable MPI
rank count.
"""

from __future__ import annotations

# Support direct execution as well as python -m from a checkout.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[5]))

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
for import_path in (REPO_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import projects.diocotron.dolfinx.torsion.optimization.reduced as reduced


@dataclass(frozen=True)
class GridCandidate:
    """One deterministic threshold-grid point."""

    stage: int
    candidate: int
    c1: float
    c2: float


GRID_FIELDS = (
    "stage",
    "candidate",
    "group",
    "c1",
    "c2",
    "width",
    "epsPhi",
    "converged",
    "newtonStatus",
    "newtonIterations",
    "newtonInitialBudget",
    "newtonFinalBudget",
    "newtonHardCeiling",
    "newtonCapExtensions",
    "newtonCapStopReason",
    "newtonContraction",
    "residual",
    "leakage",
    "leakageRel",
    "missing",
    "missingRel",
    "activityArea",
    "activityAreaRel",
    "areaMismatchRel",
    "objective",
    "solveTime",
    "wallTime",
    "seed",
    "seedAttempts",
    "minPhi",
    "maxPhi",
    "maxPhiMinusT",
    "lowerBoundViolation",
    "upperBoundViolation",
    "boundSatisfied",
    "lowerThresholdActive",
    "upperThresholdActive",
    "bothThresholdsActive",
    "selectionEligible",
    "candidatePng",
)

BRANCH_TRIAL_FIELDS = (*GRID_FIELDS, "selected")


def strip_remainder(values: Sequence[str]) -> list[str]:
    """Remove argparse's optional remainder separator."""

    result = list(values)
    if result and result[0] == "--":
        result.pop(0)
    return result


def threshold_token(value: float) -> str:
    """Encode a threshold reproducibly without filesystem punctuation ambiguity."""
    return f"{float(value):.8e}".replace("-", "m").replace("+", "p").replace(".", "d")


def threshold_grid(
    *,
    stage: int,
    center_c1: float,
    center_c2: float,
    radius_c1: float,
    radius_c2: float,
    points: int,
    c_min: float,
    c_max: float,
    min_width: float,
) -> list[GridCandidate]:
    """Build a clipped, duplicate-free grid ordered from its center outward."""

    if points < 2:
        raise ValueError("grid points must be at least two")
    if radius_c1 <= 0.0 or radius_c2 <= 0.0:
        raise ValueError("grid radii must be positive")
    c1_values = np.linspace(center_c1 - radius_c1, center_c1 + radius_c1, points)
    c2_values = np.linspace(center_c2 - radius_c2, center_c2 + radius_c2, points)
    pairs: set[tuple[float, float]] = set()
    for c1 in c1_values:
        for c2 in c2_values:
            clipped_c1 = max(float(c_min), min(float(c1), float(c_max)))
            clipped_c2 = max(float(c_min), min(float(c2), float(c_max)))
            if clipped_c2 - clipped_c1 < float(min_width):
                continue
            pairs.add((round(clipped_c1, 15), round(clipped_c2, 15)))
    scale1 = max(float(radius_c1), 1.0e-30)
    scale2 = max(float(radius_c2), 1.0e-30)
    ordered = sorted(
        pairs,
        key=lambda pair: (
            ((pair[0] - center_c1) / scale1) ** 2
            + ((pair[1] - center_c2) / scale2) ** 2,
            pair[0],
            pair[1],
        ),
    )
    return [
        GridCandidate(stage=stage, candidate=index, c1=pair[0], c2=pair[1])
        for index, pair in enumerate(ordered)
    ]


def objective_value(
    *,
    leakage_rel: float,
    activity_area: float,
    target_area: float,
    area_weight: float,
    missing_rel: float = 0.0,
    missing_weight: float = 0.0,
) -> tuple[float, float, float]:
    """Return objective, normalized activity area, and area mismatch."""

    activity_rel = float(activity_area) / max(float(target_area), 1.0e-30)
    area_mismatch = abs(activity_rel - 1.0)
    objective = (
        float(leakage_rel)
        + float(missing_weight) * float(missing_rel)
        + float(area_weight) * area_mismatch
    )
    return objective, activity_rel, area_mismatch


def result_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
    """Prefer physical, active-threshold, converged candidates deterministically."""

    if bool(row["converged"]):
        bound_satisfied = bool(row.get("boundSatisfied", True))
        selection_eligible = bool(row.get("selectionEligible", bound_satisfied))
        if bound_satisfied and selection_eligible:
            tier = 0.0
        elif bound_satisfied:
            tier = 0.25
        else:
            tier = 0.5
        return (
            tier,
            float(row["objective"]),
            float(row["residual"]),
            int(row["candidate"]),
        )
    residual = float(row["residual"])
    if not math.isfinite(residual):
        residual = math.inf
    return (1.0, residual, float(row["objective"]), int(row["candidate"]))


def effective_bound_tolerance(args: argparse.Namespace) -> float:
    """Scale the discrete maximum-principle check to the inexact PDE solve."""

    return max(float(args.grid_bound_tol), 10.0 * float(args.grid_newton_tol))


def choose_candidate_attempt(
    attempts: Sequence[tuple[dict[str, Any], np.ndarray]],
) -> tuple[dict[str, Any], np.ndarray]:
    """Choose the best independently seeded branch for one threshold pair."""

    if not attempts:
        raise ValueError("candidate branch list must not be empty")
    selected_row, selected_state = min(
        attempts,
        key=lambda attempt: result_key(attempt[0]),
    )
    row = dict(selected_row)
    row["seedAttempts"] = len(attempts)
    row["wallTime"] = sum(float(attempt[0]["wallTime"]) for attempt in attempts)
    return row, selected_state


def add_grid_options(parser: argparse.ArgumentParser, *, orchestrated: bool) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-guess-c1", type=float, required=True)
    parser.add_argument("--grid-guess-c2", type=float, required=True)
    parser.add_argument("--grid-radius-c1", type=float, default=0.025)
    parser.add_argument("--grid-radius-c2", type=float, default=0.035)
    parser.add_argument("--grid-points", type=int, default=7)
    parser.add_argument("--grid-levels", type=int, default=2)
    parser.add_argument("--grid-refine-factor", type=float, default=0.25)
    parser.add_argument("--grid-newton-tol", type=float, default=1.0e-6)
    parser.add_argument("--grid-newton-max-it", type=int, default=120)
    parser.add_argument("--grid-area-weight", type=float, default=1.0)
    parser.add_argument(
        "--grid-missing-weight",
        type=float,
        default=1.0,
        help="explicit weight for target area missed by the candidate band",
    )
    parser.add_argument(
        "--grid-require-active-thresholds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="prefer candidates whose two threshold contours both exist",
    )
    parser.add_argument(
        "--grid-cap-mode",
        choices=("target", "torsion"),
        default="target",
        help=(
            "cap candidates using cmax-factor*max(phi_target) or the "
            "maximum-principle bound max(T)"
        ),
    )
    parser.add_argument(
        "--grid-seed",
        choices=("phi-target", "torsion", "both"),
        default="phi-target",
        help="Newton seed(s) tested independently for every threshold pair",
    )
    parser.add_argument(
        "--grid-bound-tol",
        type=float,
        default=1.0e-8,
        help="nodal tolerance used to verify 0 <= phi <= T",
    )
    parser.add_argument("--ranks-per-candidate", type=int, default=4)
    parser.add_argument(
        "--save-candidate-pngs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="collect every selected candidate branch and save one complete PNG per c1,c2 pair",
    )
    parser.add_argument("--candidate-png-width", type=int, default=2400)
    parser.add_argument("--candidate-png-height", type=int, default=1250)
    if orchestrated:
        parser.add_argument("--search-ranks", type=int, default=20)
        parser.add_argument("--optimization-ranks", type=int, default=4)
        parser.add_argument("--mpirun", default="mpirun")
        parser.add_argument("--python", default=sys.executable)
    parser.add_argument("reduced_args", nargs=argparse.REMAINDER)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    search = subparsers.add_parser("search", help="run the MPI-split grid search")
    add_grid_options(search, orchestrated=False)
    run = subparsers.add_parser(
        "run",
        help="orchestrate the MPI search and normal reduced optimization",
    )
    add_grid_options(run, orchestrated=True)
    figures = subparsers.add_parser("figures", help="render PNG grid diagnostics and storyboard")
    figures.add_argument("--workflow-dir", type=Path, required=True)
    figures.add_argument("--figure-dir", type=Path, required=True)
    figures.add_argument(
        "--figure-prefix",
        default=None,
        help="optional filename prefix preserving figures from earlier workflows",
    )
    return parser.parse_args(argv)


def validate_grid_args(args: argparse.Namespace, *, world_size: int | None = None) -> None:
    if not args.grid_guess_c2 > args.grid_guess_c1 >= 0.0:
        raise ValueError("require 0 <= grid-guess-c1 < grid-guess-c2")
    if args.grid_points < 2:
        raise ValueError("require grid-points >= 2")
    if args.grid_levels < 1:
        raise ValueError("require grid-levels >= 1")
    if not (0.0 < args.grid_refine_factor < 1.0):
        raise ValueError("require 0 < grid-refine-factor < 1")
    if args.grid_newton_tol <= 0.0:
        raise ValueError("require positive grid-newton-tol")
    if args.grid_newton_max_it < 1:
        raise ValueError("require positive grid-newton-max-it")
    if args.grid_area_weight < 0.0:
        raise ValueError("require nonnegative grid-area-weight")
    if args.grid_missing_weight < 0.0:
        raise ValueError("require nonnegative grid-missing-weight")
    if args.grid_bound_tol < 0.0:
        raise ValueError("require nonnegative grid-bound-tol")
    if args.ranks_per_candidate < 1:
        raise ValueError("require positive ranks-per-candidate")
    if args.candidate_png_width < 800 or args.candidate_png_height < 500:
        raise ValueError("candidate PNG dimensions are too small for seven diagnostic panels")
    if world_size is not None:
        if world_size < args.ranks_per_candidate:
            raise ValueError("MPI world is smaller than ranks-per-candidate")
        if world_size % args.ranks_per_candidate:
            raise ValueError("MPI world size must be divisible by ranks-per-candidate")


def forwarded_reduced_args(args: argparse.Namespace) -> list[str]:
    values = strip_remainder(args.reduced_args)
    forbidden = {
        "--run-dir",
        "--run-tag",
        "--c1-phi",
        "--c2-phi",
        "--initial-state",
    }
    for token in values:
        if token.split("=", 1)[0] in forbidden:
            raise ValueError(
                f"{token.split('=', 1)[0]} is managed by the brute-force workflow"
            )
    parsed = reduced.parse_args(values)
    reduced.validate_args(parsed)
    return values


def prepare_group_problem(
    *,
    args: argparse.Namespace,
    group: MPI.Comm,
    group_dir: Path,
    group_id: int,
) -> dict[str, Any]:
    """Build one complete target/state workspace per MPI candidate group."""

    params = reduced.params_from_args(args)
    if args.grid_cap_mode == "torsion" and float(params.rho_amp) > 1.0 + 1.0e-12:
        raise ValueError("torsion cap 0 <= phi <= T requires rho-amp <= 1")
    stiffness_args = reduced.phase_solver_args(args, "stiffness")
    nonlinear_args = reduced.phase_solver_args(args, "nonlinear")
    domain, mesh_path, geometry_mode = reduced.load_or_generate_mesh(
        args,
        group_dir,
        group,
    )
    tdim = domain.topology.dim
    nt = int(domain.topology.index_map(tdim).size_global)
    V = fem.functionspace(domain, ("Lagrange", int(args.order)))
    ndof = int(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)
    bc = reduced.boundary_bc(V)
    trial = ufl.TrialFunction(V)
    test = ufl.TestFunction(V)
    qdeg = (
        int(args.quad_degree)
        if args.quad_degree is not None
        else max(2 * int(args.order) + 8, 12)
    )
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": qdeg})
    stiffness_form = ufl.inner(ufl.grad(trial), ufl.grad(test)) * dx

    T = fem.Function(V, name="torsion")
    tau_band = fem.Function(V, name="torsionBand")
    rho_design = fem.Function(V, name="rhoDesign")
    phi_target = fem.Function(V, name="phiTarget")
    u = fem.Function(V, name="phi")
    du = fem.Function(V, name="newtonUpdate")
    rho = fem.Function(V, name="rho")
    candidate_difference = fem.Function(V, name="phiMinusTarget")

    reduced.solve_linear_form(
        stiffness_form,
        1.0 * test * dx,
        T,
        [bc],
        prefix=f"grid_g{group_id}_torsion_",
        solver=stiffness_args.linear_solver,
        ksp_type=stiffness_args.ksp_type,
        rtol=stiffness_args.linear_rtol,
        atol=stiffness_args.linear_atol,
        max_it=stiffness_args.linear_max_it,
        verbosity=0,
    )
    _, tmax = reduced.global_minmax(group, T)
    c1_t = params.alpha_t1 * tmax
    c2_t = params.alpha_t2 * tmax
    eps_t = params.eps_t_ratio * (c2_t - c1_t)
    tau_mask = ufl.conditional(
        ufl.gt(T, c1_t),
        ufl.conditional(ufl.lt(T, c2_t), 1.0, 0.0),
        0.0,
    )
    reduced.update_interpolated(tau_band, tau_mask)
    target_area = reduced.assemble_scalar(group, tau_mask * dx)
    if params.eps_t_ratio == 0.0:
        target_density_mode = "sharp_indicator"
        target_density = float(params.rho_amp) * tau_mask
    else:
        target_density_mode = "smoothed_window"
        target_density = reduced.window_ufl(
            T,
            c1_t,
            c2_t,
            eps_t,
            params.rho_amp,
        )
    reduced.update_interpolated(rho_design, target_density)
    reduced.solve_linear_form(
        stiffness_form,
        rho_design * test * dx,
        phi_target,
        [bc],
        prefix=f"grid_g{group_id}_target_",
        solver=stiffness_args.linear_solver,
        ksp_type=stiffness_args.ksp_type,
        rtol=stiffness_args.linear_rtol,
        atol=stiffness_args.linear_atol,
        max_it=stiffness_args.linear_max_it,
        verbosity=0,
    )
    _, phi_target_max = reduced.global_minmax(group, phi_target)
    target_cap = max(float(args.cmax_factor) * phi_target_max, 1.0e-12)
    if args.grid_cap_mode == "torsion":
        # Since 0 <= rho <= 1, comparison with -Delta T = 1 gives the
        # maximum-principle box 0 <= phi <= T. Search that complete range;
        # the reduced optimizer retains its own step bounds afterwards.
        c_min = 0.0
        c_upper = max(float(tmax), 1.0e-12)
    else:
        c_min = float(args.c_lower_fraction) * target_cap
        c_upper = float(args.c_upper_fraction) * target_cap
    c_scale = max(c_upper - c_min, 1.0e-14)
    min_width = reduced.certified_min_width(args, c_scale)

    c1_const = fem.Constant(domain, PETSc.ScalarType(args.grid_guess_c1))
    c2_const = fem.Constant(domain, PETSc.ScalarType(args.grid_guess_c2))
    eps_const = fem.Constant(
        domain,
        PETSc.ScalarType(
            reduced.epsilon_from_thresholds(
                args,
                args.grid_guess_c1,
                args.grid_guess_c2,
            )
        ),
    )
    nonlinear_args.max_newton_it = int(args.grid_newton_max_it)
    nonlinear_args.verbosity = 0
    return {
        "params": params,
        "args": args,
        "nonlinear_args": nonlinear_args,
        "domain": domain,
        "mesh_path": str(mesh_path),
        "geometry_mode": geometry_mode,
        "nt": nt,
        "ndof": ndof,
        "qdeg": qdeg,
        "trial": trial,
        "test": test,
        "dx": dx,
        "bc": bc,
        "stiffness_form": stiffness_form,
        "tau_mask": tau_mask,
        "T": T,
        "tau_band": tau_band,
        "rho_design": rho_design,
        "c1_t": c1_t,
        "c2_t": c2_t,
        "tmax": tmax,
        "target_area": target_area,
        "target_density_mode": target_density_mode,
        "phi_target_max": phi_target_max,
        "target_cap": target_cap,
        "phi_target": phi_target,
        "u": u,
        "du": du,
        "rho": rho,
        "candidate_difference": candidate_difference,
        "c1_const": c1_const,
        "c2_const": c2_const,
        "eps_const": eps_const,
        "c_min": c_min,
        "c_max": c_upper,
        "min_width": min_width,
    }


def evaluate_candidate(
    *,
    candidate: GridCandidate,
    group_id: int,
    seed: np.ndarray,
    seed_label: str,
    problem: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    """Project and score one threshold pair on its candidate communicator."""

    args = problem["args"]
    u = problem["u"]
    u.x.array[:] = seed
    u.x.scatter_forward()
    eps_phi = reduced.epsilon_from_thresholds(args, candidate.c1, candidate.c2)
    started = time.perf_counter()
    newton = reduced.solve_equilibrium(
        u=u,
        du=problem["du"],
        rho=problem["rho"],
        trial=problem["trial"],
        test=problem["test"],
        dx=problem["dx"],
        bc=problem["bc"],
        stiffness_form=problem["stiffness_form"],
        c1_const=problem["c1_const"],
        c2_const=problem["c2_const"],
        eps_const=problem["eps_const"],
        c1=candidate.c1,
        c2=candidate.c2,
        eps_phi=eps_phi,
        rho_amp=problem["params"].rho_amp,
        tol_res=float(args.grid_newton_tol),
        args=problem["nonlinear_args"],
        prefix=(
            f"grid_s{candidate.stage}_c{candidate.candidate}_g{group_id}_"
            f"{seed_label.replace(chr(45), chr(95))}_"
        ),
    )
    metrics = reduced.evaluate_band_metrics(
        comm=u.function_space.mesh.comm,
        u=u,
        tau_mask=problem["tau_mask"],
        dx=problem["dx"],
        c1_const=problem["c1_const"],
        c2_const=problem["c2_const"],
        eps_const=problem["eps_const"],
        c1=candidate.c1,
        c2=candidate.c2,
        eps_phi=eps_phi,
        kappa=args.kappa,
        target_area=problem["target_area"],
    )
    objective, area_rel, area_mismatch = objective_value(
        leakage_rel=metrics.leakage_rel,
        activity_area=metrics.activity_area,
        target_area=problem["target_area"],
        area_weight=args.grid_area_weight,
        missing_rel=metrics.missing_rel,
        missing_weight=args.grid_missing_weight,
    )
    comm = u.function_space.mesh.comm
    index_map = u.function_space.dofmap.index_map
    owned = int(index_map.size_local * u.function_space.dofmap.index_map_bs)
    phi_owned = np.real(np.asarray(u.x.array[:owned]))
    torsion_owned = np.real(np.asarray(problem["T"].x.array[:owned]))
    local_min_phi = float(np.min(phi_owned)) if owned else math.inf
    local_max_phi = float(np.max(phi_owned)) if owned else -math.inf
    local_max_phi_minus_t = (
        float(np.max(phi_owned - torsion_owned)) if owned else -math.inf
    )
    min_phi = float(comm.allreduce(local_min_phi, op=MPI.MIN))
    max_phi = float(comm.allreduce(local_max_phi, op=MPI.MAX))
    max_phi_minus_t = float(comm.allreduce(local_max_phi_minus_t, op=MPI.MAX))
    lower_bound_violation = max(0.0, -min_phi)
    upper_bound_violation = max(0.0, max_phi_minus_t)
    bound_tolerance = effective_bound_tolerance(args)
    lower_threshold_active = (
        min_phi - bound_tolerance <= candidate.c1 <= max_phi + bound_tolerance
    )
    upper_threshold_active = (
        min_phi - bound_tolerance <= candidate.c2 <= max_phi + bound_tolerance
    )
    both_thresholds_active = lower_threshold_active and upper_threshold_active
    bound_satisfied = (
        lower_bound_violation <= bound_tolerance
        and upper_bound_violation <= bound_tolerance
    )
    selection_eligible = (
        bool(newton.converged)
        and bound_satisfied
        and (not args.grid_require_active_thresholds or both_thresholds_active)
    )
    row = {
        "stage": candidate.stage,
        "candidate": candidate.candidate,
        "group": group_id,
        "c1": candidate.c1,
        "c2": candidate.c2,
        "width": candidate.c2 - candidate.c1,
        "epsPhi": eps_phi,
        "converged": int(newton.converged),
        "newtonStatus": newton.status,
        "newtonIterations": newton.iterations,
        "newtonInitialBudget": getattr(
            problem["nonlinear_args"], "_newton_initial_budget", ""
        ),
        "newtonFinalBudget": getattr(
            problem["nonlinear_args"], "_newton_current_budget", ""
        ),
        "newtonHardCeiling": getattr(
            problem["nonlinear_args"], "_newton_hard_ceiling", ""
        ),
        "newtonCapExtensions": getattr(
            problem["nonlinear_args"], "_newton_cap_extensions", 0
        ),
        "newtonCapStopReason": getattr(
            problem["nonlinear_args"], "_newton_cap_reason", ""
        ),
        "newtonContraction": getattr(
            problem["nonlinear_args"], "_newton_contraction", math.nan
        ),
        "residual": newton.residual,
        "leakage": metrics.leakage,
        "leakageRel": metrics.leakage_rel,
        "missing": metrics.missing,
        "missingRel": metrics.missing_rel,
        "activityArea": metrics.activity_area,
        "activityAreaRel": area_rel,
        "areaMismatchRel": area_mismatch,
        "objective": objective,
        "solveTime": newton.solve_time,
        "wallTime": time.perf_counter() - started,
        "seed": seed_label,
        "seedAttempts": 1,
        "minPhi": min_phi,
        "maxPhi": max_phi,
        "maxPhiMinusT": max_phi_minus_t,
        "lowerBoundViolation": lower_bound_violation,
        "upperBoundViolation": upper_bound_violation,
        "boundSatisfied": int(bound_satisfied),
        "lowerThresholdActive": int(lower_threshold_active),
        "upperThresholdActive": int(upper_threshold_active),
        "bothThresholdsActive": int(both_thresholds_active),
        "selectionEligible": int(selection_eligible),
        "candidatePng": "",
    }
    return row, u.x.array.copy()


def gather_global_state(
    comm: MPI.Comm,
    function_space,
    local_values: np.ndarray,
) -> np.ndarray | None:
    """Gather owned scalar coefficients by stable global DOF index."""

    index_map = function_space.dofmap.index_map
    owned = int(index_map.size_local)
    local_dofs = np.arange(owned, dtype=np.int32)
    global_dofs = np.asarray(index_map.local_to_global(local_dofs), dtype=np.int64)
    payload = (
        global_dofs,
        np.asarray(local_values[:owned], dtype=np.float64).copy(),
    )
    gathered = comm.gather(payload, root=0)
    if comm.rank != 0:
        return None
    values = np.full(int(index_map.size_global), np.nan, dtype=np.float64)
    for indices, coefficients in gathered:
        values[indices] = coefficients
    if not np.all(np.isfinite(values)):
        missing = int(np.count_nonzero(~np.isfinite(values)))
        raise RuntimeError(f"winner-state gather left {missing} global DOFs unset")
    return values

def write_search_outputs(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    branch_trials: list[dict[str, Any]],
    winner: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "grid.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GRID_FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (int(item["stage"]), int(item["candidate"]))):
            writer.writerow(row)
    with (output_dir / "branch_trials.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=BRANCH_TRIAL_FIELDS)
        writer.writeheader()
        for row in sorted(
            branch_trials,
            key=lambda item: (
                int(item["stage"]),
                int(item["candidate"]),
                str(item["seed"]),
            ),
        ):
            writer.writerow(row)
    payload = dict(metadata)
    payload["winner"] = winner
    payload["candidateCount"] = len(rows)
    payload["branchTrialCount"] = len(branch_trials)
    payload["convergedCount"] = sum(int(row["converged"]) for row in rows)
    with (output_dir / "winner.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_search(args: argparse.Namespace) -> int:
    world = MPI.COMM_WORLD
    validate_grid_args(args, world_size=world.size)
    reduced_argv = forwarded_reduced_args(args)
    reduced_args = reduced.parse_args(reduced_argv)
    reduced_args.grid_guess_c1 = float(args.grid_guess_c1)
    reduced_args.grid_guess_c2 = float(args.grid_guess_c2)
    reduced_args.grid_newton_tol = float(args.grid_newton_tol)
    reduced_args.grid_newton_max_it = int(args.grid_newton_max_it)
    reduced_args.grid_area_weight = float(args.grid_area_weight)
    reduced_args.grid_missing_weight = float(args.grid_missing_weight)
    reduced_args.grid_require_active_thresholds = bool(args.grid_require_active_thresholds)
    reduced_args.grid_cap_mode = str(args.grid_cap_mode)
    reduced_args.grid_seed = str(args.grid_seed)
    reduced_args.grid_bound_tol = float(args.grid_bound_tol)
    reduced_args.plot = False
    reduced_args.save_frames = False
    reduced_args.plot_off_screen = True
    reduced_args.plot_mesh_edges = False

    group_id = world.rank // int(args.ranks_per_candidate)
    group_count = world.size // int(args.ranks_per_candidate)
    group = world.Split(color=group_id, key=world.rank)
    peer = world.Split(color=group.rank, key=group_id)
    output_dir = args.output_dir.resolve()
    if world.rank == 0:
        if (output_dir / "winner.json").exists() or (output_dir / "grid.csv").exists():
            raise FileExistsError(f"search output already exists: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    world.barrier()
    group_dir = output_dir / "scratch" / f"group_{group_id:03d}"
    if group.rank == 0:
        group_dir.mkdir(parents=True, exist_ok=True)
    group.barrier()

    problem = prepare_group_problem(
        args=reduced_args,
        group=group,
        group_dir=group_dir,
        group_id=group_id,
    )
    candidate_plotter = None
    if args.save_candidate_pngs:
        candidate_plotter = reduced.MPIPyVistaTorsionPlotter(
            reduced_args,
            run_tag=f"grid_group_{group_id:03d}",
            run_dir=output_dir,
            frame_writer=None,
            comm=group,
        )
    phi_target_state = problem["phi_target"].x.array.copy()
    torsion_state = problem["T"].x.array.copy()
    stage_seed = (
        torsion_state.copy()
        if args.grid_seed == "torsion"
        else phi_target_state.copy()
    )
    center_c1 = float(args.grid_guess_c1)
    center_c2 = float(args.grid_guess_c2)
    radius_c1 = float(args.grid_radius_c1)
    radius_c2 = float(args.grid_radius_c2)
    all_rows: list[dict[str, Any]] = []
    all_branch_trials: list[dict[str, Any]] = []
    latest_winner: dict[str, Any] | None = None

    if world.rank == 0:
        print(
            "GRID_SEARCH "
            f"worldRanks={world.size} groups={group_count} "
            f"ranksPerCandidate={args.ranks_per_candidate} "
            f"targetMode={problem['target_density_mode']} "
            f"capMode={args.grid_cap_mode} cap={problem['c_max']:.6e} "
            f"seedMode={args.grid_seed} targetArea={problem['target_area']:.6e} "
            f"ndof={problem['ndof']}",
            flush=True,
        )

    for stage in range(int(args.grid_levels)):
        candidates = threshold_grid(
            stage=stage,
            center_c1=center_c1,
            center_c2=center_c2,
            radius_c1=radius_c1,
            radius_c2=radius_c2,
            points=int(args.grid_points),
            c_min=problem["c_min"],
            c_max=problem["c_max"],
            min_width=problem["min_width"],
        )
        assigned = [
            candidate
            for index, candidate in enumerate(candidates)
            if index % group_count == group_id
        ]
        candidate_stage_dir = output_dir / "candidate_pairs" / f"stage_{stage:03d}"
        if args.save_candidate_pngs and group.rank == 0:
            candidate_stage_dir.mkdir(parents=True, exist_ok=True)
        group.barrier()
        local_rows: list[dict[str, Any]] = []
        local_branch_trials: list[dict[str, Any]] = []
        local_best_row: dict[str, Any] | None = None
        local_best_state: np.ndarray | None = None
        continuation_state = stage_seed.copy()
        continuation_label = (
            "torsion"
            if args.grid_seed == "torsion"
            else ("phi_target" if stage == 0 else "previous_stage_winner")
        )

        for candidate in assigned:
            if args.grid_seed == "torsion":
                seed_specs = [("torsion", torsion_state)]
            elif args.grid_seed == "both":
                seed_specs = [
                    ("phi_target", phi_target_state),
                    ("torsion", torsion_state),
                ]
                if continuation_label not in {"phi_target", "torsion"}:
                    seed_specs.insert(0, (continuation_label, continuation_state))
            else:
                seed_specs = [(continuation_label, continuation_state)]
            attempts = [
                evaluate_candidate(
                    candidate=candidate,
                    group_id=group_id,
                    seed=seed_values,
                    seed_label=seed_name,
                    problem=problem,
                )
                for seed_name, seed_values in seed_specs
            ]
            row, state = choose_candidate_attempt(attempts)
            if candidate_plotter is not None:
                problem["u"].x.array[:] = state
                problem["u"].x.scatter_forward()
                problem["c1_const"].value = PETSc.ScalarType(candidate.c1)
                problem["c2_const"].value = PETSc.ScalarType(candidate.c2)
                problem["eps_const"].value = PETSc.ScalarType(row["epsPhi"])
                reduced.update_interpolated(
                    problem["rho"],
                    reduced.window_density_const_ufl(
                        problem["u"],
                        problem["c1_const"],
                        problem["c2_const"],
                        problem["eps_const"],
                        problem["params"].rho_amp,
                    ),
                )
                difference = problem["candidate_difference"]
                difference.x.array[:] = (
                    problem["u"].x.array - problem["phi_target"].x.array
                )
                difference.x.scatter_forward()
                status_token = (
                    "converged" if bool(row["converged"]) else "partial"
                )
                png_name = (
                    f"candidate_{candidate.candidate:05d}_"
                    f"c1_{threshold_token(candidate.c1)}_"
                    f"c2_{threshold_token(candidate.c2)}_{status_token}.png"
                )
                png_path = candidate_stage_dir / png_name
                summary = (
                    f"stage={stage} candidate={candidate.candidate} group={group_id}\n"
                    f"c1={candidate.c1:.8e}  c2={candidate.c2:.8e}\n"
                    f"width={candidate.c2 - candidate.c1:.4e}  eps={row['epsPhi']:.4e}\n"
                    f"Newton={row['newtonStatus']}  iterations={row['newtonIterations']}\n"
                    f"residual={row['residual']:.4e}  seed={row['seed']}\n"
                    f"leak={row['leakageRel']:.4e}  missing={row['missingRel']:.4e}\n"
                    f"area ratio={row['activityAreaRel']:.4e}  J={row['objective']:.4e}\n"
                    f"active contours={row['bothThresholdsActive']}  "
                    f"bounds={row['boundSatisfied']}"
                )
                candidate_plotter.save_candidate_fields(
                    [
                        problem["T"],
                        problem["tau_band"],
                        problem["rho_design"],
                        problem["phi_target"],
                        problem["u"],
                        problem["rho"],
                        difference,
                    ],
                    [
                        "torsion T",
                        "target band",
                        "target density",
                        "target potential",
                        "candidate potential",
                        "candidate density",
                        "potential mismatch",
                    ],
                    save_path=png_path,
                    window_size=(
                        int(args.candidate_png_width),
                        int(args.candidate_png_height),
                    ),
                    nt=problem["nt"],
                    ndof=problem["ndof"],
                    target_field_index=0,
                    target_levels=(problem["c1_t"], problem["c2_t"]),
                    equilibrium_field_index=4,
                    equilibrium_levels=(candidate.c1, candidate.c2),
                    summary=summary,
                    stage=f"grid_s{stage}_c{candidate.candidate}",
                )
                row["candidatePng"] = str(png_path.relative_to(output_dir))
            local_rows.append(row)
            for attempt_row, _ in attempts:
                trial_row = dict(attempt_row)
                trial_row["seedAttempts"] = len(attempts)
                trial_row["selected"] = int(attempt_row["seed"] == row["seed"])
                if trial_row["selected"]:
                    trial_row["candidatePng"] = row["candidatePng"]
                local_branch_trials.append(trial_row)
            if group.rank == 0:
                print(
                    "GRID_POINT "
                    f"stage={stage} id={candidate.candidate} group={group_id} "
                    f"c1={candidate.c1:.8e} c2={candidate.c2:.8e} "
                    f"conv={row['converged']} res={row['residual']:.3e} "
                    f"leakRel={row['leakageRel']:.3e} "
                    f"areaRel={row['activityAreaRel']:.3e} "
                    f"J={row['objective']:.3e} seed={row['seed']} "
                    f"bound={row['boundSatisfied']} active={row['bothThresholdsActive']} "
                    f"eligible={row['selectionEligible']} wall={row['wallTime']:.2f}s",
                    flush=True,
                )
            if bool(row["converged"]):
                continuation_state = state
                continuation_label = f"stage_{stage}_candidate_{candidate.candidate}"
            if local_best_row is None or result_key(row) < result_key(local_best_row):
                local_best_row = row
                local_best_state = state

        payload = (local_rows, local_branch_trials) if group.rank == 0 else None
        gathered = world.gather(payload, root=0)
        if world.rank == 0:
            stage_rows: list[dict[str, Any]] = []
            stage_branch_trials: list[dict[str, Any]] = []
            for item in gathered:
                if item:
                    stage_rows.extend(item[0])
                    stage_branch_trials.extend(item[1])
            if len(stage_rows) != len(candidates):
                raise RuntimeError(
                    f"grid stage {stage} returned {len(stage_rows)} of {len(candidates)} candidates"
                )
            if args.save_candidate_pngs:
                missing_pngs = [
                    row["candidatePng"]
                    for row in stage_rows
                    if not row["candidatePng"]
                    or not (output_dir / str(row["candidatePng"])).is_file()
                ]
                if missing_pngs:
                    raise RuntimeError(
                        f"grid stage {stage} is missing {len(missing_pngs)} candidate PNGs"
                    )
            all_rows.extend(stage_rows)
            all_branch_trials.extend(stage_branch_trials)
            latest_winner = min(stage_rows, key=result_key)
            print(
                "GRID_STAGE_WINNER "
                f"stage={stage} group={latest_winner['group']} "
                f"candidate={latest_winner['candidate']} "
                f"c1={latest_winner['c1']:.8e} c2={latest_winner['c2']:.8e} "
                f"conv={latest_winner['converged']} "
                f"res={latest_winner['residual']:.3e} "
                f"leakRel={latest_winner['leakageRel']:.3e} "
                f"areaRel={latest_winner['activityAreaRel']:.3e} "
                f"J={latest_winner['objective']:.3e}",
                flush=True,
            )
        latest_winner = world.bcast(latest_winner, root=0)
        winning_group = int(latest_winner["group"])
        if group_id == winning_group:
            if local_best_row is None or int(local_best_row["candidate"]) != int(
                latest_winner["candidate"]
            ):
                raise RuntimeError("winning group did not retain its selected state")
            winning_local_state = local_best_state
        else:
            winning_local_state = None
        stage_seed = peer.bcast(winning_local_state, root=winning_group)
        center_c1 = float(latest_winner["c1"])
        center_c2 = float(latest_winner["c2"])
        radius_c1 *= float(args.grid_refine_factor)
        radius_c2 *= float(args.grid_refine_factor)
        world.barrier()

    if latest_winner is None:
        raise RuntimeError("grid search did not select a final-stage winner")
    if group_id == 0:
        global_phi = gather_global_state(
            group,
            problem["u"].function_space,
            stage_seed,
        )
    else:
        global_phi = None

    if world.rank == 0:
        winner = dict(latest_winner)
        state_path = output_dir / "winner_state.npz"
        np.savez_compressed(
            state_path,
            phi=global_phi,
            c1=np.asarray([winner["c1"]], dtype=np.float64),
            c2=np.asarray([winner["c2"]], dtype=np.float64),
            order=np.asarray([reduced_args.order], dtype=np.int32),
        )
        metadata = {
            "version": 3,
            "objective": (
                "leakageRel + missingWeight*missingRel + "
                "areaWeight*abs(activityAreaRel-1)"
            ),
            "grid": {
                "guessC1": args.grid_guess_c1,
                "guessC2": args.grid_guess_c2,
                "radiusC1": args.grid_radius_c1,
                "radiusC2": args.grid_radius_c2,
                "points": args.grid_points,
                "levels": args.grid_levels,
                "refineFactor": args.grid_refine_factor,
                "newtonTolerance": args.grid_newton_tol,
                "newtonMaxIterations": args.grid_newton_max_it,
                "areaWeight": args.grid_area_weight,
                "missingWeight": args.grid_missing_weight,
                "requireActiveThresholds": args.grid_require_active_thresholds,
                "saveCandidatePngs": args.save_candidate_pngs,
                "candidatePngWidth": args.candidate_png_width,
                "candidatePngHeight": args.candidate_png_height,
                "capMode": args.grid_cap_mode,
                "seedMode": args.grid_seed,
                "boundTolerance": args.grid_bound_tol,
                "effectiveBoundTolerance": effective_bound_tolerance(args),
                "thresholdMinimum": problem["c_min"],
                "thresholdMaximum": problem["c_max"],
            },
            "mpi": {
                "worldRanks": world.size,
                "groups": group_count,
                "ranksPerCandidate": args.ranks_per_candidate,
            },
            "problem": {
                "mesh": problem["mesh_path"],
                "geometry": problem["geometry_mode"],
                "cells": problem["nt"],
                "dofs": problem["ndof"],
                "order": reduced_args.order,
                "quadratureDegree": problem["qdeg"],
                "targetDensityMode": problem["target_density_mode"],
                "targetArea": problem["target_area"],
                "torsionMaximum": problem["tmax"],
                "targetPotentialMaximum": problem["phi_target_max"],
                "legacyTargetCap": problem["target_cap"],
                "maximumPrinciple": "0 <= phi <= T (rhoAmp <= 1)",
                "epsTRatio": reduced_args.eps_t_ratio,
                "epsPhiRatio": reduced_args.eps_ratio,
            },
            "reducedArgv": reduced_argv,
            "initialState": str(state_path),
        }
        write_search_outputs(
            output_dir=output_dir,
            rows=all_rows,
            branch_trials=all_branch_trials,
            winner=winner,
            metadata=metadata,
        )
        print(
            "GRID_WINNER "
            f"c1={winner['c1']:.16e} c2={winner['c2']:.16e} "
            f"conv={winner['converged']} residual={winner['residual']:.6e} "
            f"objective={winner['objective']:.6e} "
            f"output={output_dir / 'winner.json'}",
            flush=True,
        )
    else:
        winner = None
    winner = world.bcast(winner, root=0)
    return 0 if bool(winner["converged"]) else 2


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            handle.write(line)
            handle.flush()
        return int(process.wait())


def mpirun_prefix(executable: str, ranks: int) -> list[str]:
    return [
        executable,
        "--bind-to",
        "core",
        "--map-by",
        "core",
        "-n",
        str(ranks),
    ]


def grid_forward_options(args: argparse.Namespace, output_dir: Path) -> list[str]:
    return [
        "--output-dir",
        str(output_dir),
        "--grid-guess-c1",
        str(args.grid_guess_c1),
        "--grid-guess-c2",
        str(args.grid_guess_c2),
        "--grid-radius-c1",
        str(args.grid_radius_c1),
        "--grid-radius-c2",
        str(args.grid_radius_c2),
        "--grid-points",
        str(args.grid_points),
        "--grid-levels",
        str(args.grid_levels),
        "--grid-refine-factor",
        str(args.grid_refine_factor),
        "--grid-newton-tol",
        str(args.grid_newton_tol),
        "--grid-newton-max-it",
        str(args.grid_newton_max_it),
        "--grid-area-weight",
        str(args.grid_area_weight),
        "--grid-missing-weight",
        str(args.grid_missing_weight),
        (
            "--grid-require-active-thresholds"
            if args.grid_require_active_thresholds
            else "--no-grid-require-active-thresholds"
        ),
        "--grid-cap-mode",
        str(args.grid_cap_mode),
        "--grid-seed",
        str(args.grid_seed),
        "--grid-bound-tol",
        str(args.grid_bound_tol),
        (
            "--save-candidate-pngs"
            if args.save_candidate_pngs
            else "--no-save-candidate-pngs"
        ),
        "--candidate-png-width",
        str(args.candidate_png_width),
        "--candidate-png-height",
        str(args.candidate_png_height),
        "--ranks-per-candidate",
        str(args.ranks_per_candidate),
    ]


def run_workflow(args: argparse.Namespace) -> int:
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("run subcommand must be launched without mpirun")
    validate_grid_args(args)
    reduced_argv = forwarded_reduced_args(args)
    if args.search_ranks % args.ranks_per_candidate:
        raise ValueError("search-ranks must be divisible by ranks-per-candidate")
    if args.optimization_ranks < 1:
        raise ValueError("optimization-ranks must be positive")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"workflow output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    search_dir = output_dir / "grid_search"
    optimization_dir = output_dir / "optimization"
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    python_dir = str(Path(args.python).resolve().parent)
    env["PATH"] = python_dir + os.pathsep + env.get("PATH", "")

    module_path = Path(__file__).resolve()
    search_command = [
        *mpirun_prefix(args.mpirun, int(args.search_ranks)),
        args.python,
        str(module_path),
        "search",
        *grid_forward_options(args, search_dir),
        "--",
        *reduced_argv,
    ]
    search_code = run_logged(
        search_command,
        output_dir / "search_stdout.txt",
        env,
    )
    winner_path = search_dir / "winner.json"
    if not winner_path.is_file():
        raise RuntimeError(f"grid search did not produce {winner_path}")
    winner_payload = json.loads(winner_path.read_text(encoding="utf-8"))
    winner = winner_payload["winner"]
    if not bool(winner["converged"]):
        workflow = {
            "searchCommand": search_command,
            "searchExitCode": search_code,
            "optimizerCommand": None,
            "optimizerExitCode": None,
            "winner": winner,
        }
        (output_dir / "workflow.json").write_text(
            json.dumps(workflow, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 2

    optimizer_command = [
        *mpirun_prefix(args.mpirun, int(args.optimization_ranks)),
        args.python,
        str(Path(reduced.__file__).resolve()),
        "--run-dir",
        str(optimization_dir),
        "--c1-phi",
        f"{float(winner['c1']):.17g}",
        "--c2-phi",
        f"{float(winner['c2']):.17g}",
        "--initial-state",
        str(search_dir / "winner_state.npz"),
        *reduced_argv,
    ]
    optimizer_code = run_logged(
        optimizer_command,
        output_dir / "optimizer_stdout.txt",
        env,
    )
    workflow = {
        "searchCommand": search_command,
        "searchExitCode": search_code,
        "optimizerCommand": optimizer_command,
        "optimizerExitCode": optimizer_code,
        "winner": winner,
        "optimizationDir": str(optimization_dir),
    }
    (output_dir / "workflow.json").write_text(
        json.dumps(workflow, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return optimizer_code



def render_workflow_figures(
    workflow_dir: Path,
    figure_dir: Path,
    figure_prefix: str | None = None,
) -> int:
    """Render diagnostics and a target-once design/evolution storyboard."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from projects.diocotron.studies.torsion_optimizer.figures.frames import (
        render as render_frame_storyboard,
    )

    workflow_dir = workflow_dir.resolve()
    figure_dir = figure_dir.resolve()
    grid_path = workflow_dir / "grid_search" / "grid.csv"
    winner_path = workflow_dir / "grid_search" / "winner.json"
    frames_path = workflow_dir / "optimization" / "logs" / "frames.csv"
    if not grid_path.is_file() or not winner_path.is_file() or not frames_path.is_file():
        raise FileNotFoundError("workflow lacks grid.csv, winner.json, or frames.csv")
    figure_dir.mkdir(parents=True, exist_ok=True)

    with grid_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    winner_document = json.loads(winner_path.read_text(encoding="utf-8"))
    winner = winner_document["winner"]
    numeric = {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in (
            "stage",
            "c1",
            "c2",
            "converged",
            "residual",
            "objective",
            "activityAreaRel",
            "leakageRel",
            "bothThresholdsActive",
            "selectionEligible",
        )
    }

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), constrained_layout=True)
    stage_zero = numeric["stage"] == 0
    stage_one = numeric["stage"] == np.max(numeric["stage"])
    residual_color = np.log10(np.maximum(numeric["residual"][stage_zero], 1.0e-16))
    scatter = axes[0].scatter(
        numeric["c1"][stage_zero],
        numeric["c2"][stage_zero],
        c=residual_color,
        cmap="magma_r",
        s=62,
        edgecolors="black",
        linewidths=0.35,
    )
    converged_zero = stage_zero & (numeric["converged"] > 0.5)
    axes[0].scatter(
        numeric["c1"][converged_zero],
        numeric["c2"][converged_zero],
        s=125,
        facecolors="none",
        edgecolors="#00a65a",
        linewidths=1.6,
        label="Newton converged",
    )
    eligible_zero = stage_zero & (numeric["selectionEligible"] > 0.5)
    axes[0].scatter(
        numeric["c1"][eligible_zero],
        numeric["c2"][eligible_zero],
        marker="o",
        s=34,
        color="#00a65a",
        linewidths=0.0,
        label="PDE/bounds/two contours eligible",
    )
    inactive_zero = converged_zero & (numeric["bothThresholdsActive"] < 0.5)
    axes[0].scatter(
        numeric["c1"][inactive_zero],
        numeric["c2"][inactive_zero],
        marker="x",
        s=68,
        color="#6b6b6b",
        linewidths=1.2,
        label="inactive threshold contour",
    )
    fig.colorbar(scatter, ax=axes[0], label=r"$\log_{10}\|R\|_2$")
    axes[0].legend(loc="best", frameon=True)
    axes[0].set_title("Broad grid: Newton feasibility")

    scatter = axes[1].scatter(
        numeric["c1"][stage_one],
        numeric["c2"][stage_one],
        c=numeric["objective"][stage_one],
        cmap="viridis_r",
        s=62,
        edgecolors="black",
        linewidths=0.35,
    )
    axes[1].scatter(
        [float(winner["c1"])],
        [float(winner["c2"])],
        marker="*",
        s=230,
        facecolor="#ffcc00",
        edgecolor="black",
        linewidth=0.8,
        label="selected pair",
        zorder=5,
    )
    ineligible_final = stage_one & (numeric["selectionEligible"] < 0.5)
    axes[1].scatter(
        numeric["c1"][ineligible_final],
        numeric["c2"][ineligible_final],
        marker="x",
        s=62,
        color="#6b6b6b",
        linewidths=1.1,
        label="not selection-eligible",
    )
    fig.colorbar(scatter, ax=axes[1], label=r"$J_{\rm grid}$")
    axes[1].legend(loc="best", frameon=True)
    axes[1].set_title("Refined grid: direct objective")

    converged = numeric["converged"] > 0.5
    scatter = axes[2].scatter(
        numeric["activityAreaRel"][converged],
        numeric["leakageRel"][converged],
        c=numeric["stage"][converged],
        cmap="coolwarm",
        s=48,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.25,
    )
    axes[2].scatter([1.0], [0.0], marker="*", s=230, color="#00a65a", label="ideal")
    axes[2].scatter(
        [float(winner["activityAreaRel"])],
        [float(winner["leakageRel"])],
        marker="*",
        s=190,
        color="#ffcc00",
        edgecolor="black",
        linewidth=0.8,
        label="selected pair",
    )
    fig.colorbar(scatter, ax=axes[2], label="grid level")
    axes[2].legend(loc="best", frameon=True)
    axes[2].set_xlabel(r"activity area $/|B_T|$")
    axes[2].set_ylabel(r"leakage $/|B_T|$")
    axes[2].set_title("Feasible-pair geometry")
    for axis in axes[:2]:
        axis.set_xlabel(r"$c_{1,\phi}$")
        axis.set_ylabel(r"$c_{2,\phi}$")
    for axis in axes:
        axis.grid(alpha=0.22)
    diagnostic_name = (
        f"{figure_prefix}_grid_search.png"
        if figure_prefix
        else "iter_0p6_0p7_bruteforce_grid_search.png"
    )
    diagnostic_path = figure_dir / diagnostic_name
    fig.savefig(diagnostic_path, dpi=240, facecolor="white")
    plt.close(fig)

    with frames_path.open(newline="", encoding="utf-8") as handle:
        frame_rows = list(csv.DictReader(handle))
    design_rows = [row for row in frame_rows if row["stage"] == "DESIGN"]
    final_rows = [row for row in frame_rows if row["stage"] == "FINAL"]
    if not design_rows or not final_rows:
        raise RuntimeError("workflow frames.csv lacks DESIGN or FINAL PNG")
    storyboard_name = (
        f"{figure_prefix}_storyboard.png"
        if figure_prefix
        else "iter_0p6_0p7_bruteforce_sharp_eps004_storyboard.png"
    )
    storyboard_path = figure_dir / storyboard_name
    render_frame_storyboard(
        workflow_dir / "optimization",
        [storyboard_path],
    )

    # Preserve the complete rank-zero-rendered optimizer sequence at native
    # resolution so the portable figure bundle supports qualitative review.
    prefix = figure_prefix or "iter_0p6_0p7_bruteforce"
    frame_bundle = figure_dir / f"{prefix}_frames"
    frame_bundle.mkdir(parents=True, exist_ok=True)
    copied_frames: list[Path] = []
    for row in frame_rows:
        source = Path(row["filename"])
        if not source.is_file():
            continue
        destination = frame_bundle / source.name
        shutil.copy2(source, destination)
        copied_frames.append(destination)

    # Four frames per sheet keeps narrow bands and panel annotations legible.
    contact_paths: list[Path] = []
    for page, offset in enumerate(range(0, len(copied_frames), 4), start=1):
        page_frames = [
            Image.open(path).convert("RGB")
            for path in copied_frames[offset : offset + 4]
        ]
        if not page_frames:
            continue
        cell_width = 1400
        cell_height = max(
            1,
            round(page_frames[0].height * cell_width / page_frames[0].width),
        )
        sheet = Image.new("RGB", (2 * cell_width, 2 * cell_height), "white")
        for index, frame_image in enumerate(page_frames):
            frame_image.thumbnail(
                (cell_width, cell_height),
                Image.Resampling.LANCZOS,
            )
            x = (index % 2) * cell_width + (cell_width - frame_image.width) // 2
            y = (index // 2) * cell_height + (cell_height - frame_image.height) // 2
            sheet.paste(frame_image, (x, y))
        contact_path = figure_dir / f"{prefix}_frames_contact_{page:02d}.png"
        sheet.save(contact_path, format="PNG", compress_level=6)
        contact_paths.append(contact_path)

    def save_grid_scatter(
        axis: Any,
        mask: np.ndarray,
        values: np.ndarray,
        *,
        title: str,
        label: str,
        logarithmic: bool = False,
    ) -> None:
        plotted = np.asarray(values[mask], dtype=np.float64)
        if logarithmic:
            plotted = np.log10(np.maximum(plotted, 1.0e-16))
            label = rf"$\log_{{10}}$({label})"
        artist = axis.scatter(
            numeric["c1"][mask],
            numeric["c2"][mask],
            c=plotted,
            cmap="viridis_r",
            s=54,
            edgecolors="black",
            linewidths=0.25,
        )
        axis.figure.colorbar(artist, ax=axis, label=label)
        axis.set_title(title)
        axis.set_xlabel(r"$c_{1,\phi}$")
        axis.set_ylabel(r"$c_{2,\phi}$")
        axis.grid(alpha=0.2)

    # Render each refinement level separately: the final search region can be
    # orders of magnitude narrower than the broad physical search triangle.
    stage_paths: list[Path] = []
    extra_numeric = {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in (
            "newtonIterations",
            "wallTime",
            "missingRel",
            "areaMismatchRel",
        )
    }
    for stage_value in sorted(set(numeric["stage"].astype(int).tolist())):
        stage_mask = numeric["stage"].astype(int) == stage_value
        stage_fig, stage_axes = plt.subplots(
            2,
            3,
            figsize=(15.6, 9.2),
            constrained_layout=True,
        )
        panels = (
            (stage_axes[0, 0], numeric["objective"], "Direct grid objective", r"$J_{\rm grid}$", False),
            (stage_axes[0, 1], numeric["residual"], "Terminal Newton residual", r"$\|R\|_2$", True),
            (stage_axes[0, 2], extra_numeric["newtonIterations"], "Newton work", "iterations", False),
            (stage_axes[1, 0], numeric["leakageRel"], "Relative leakage", r"$L/|B_T|$", False),
            (stage_axes[1, 1], extra_numeric["missingRel"], "Relative missing area", r"$M/|B_T|$", False),
            (stage_axes[1, 2], extra_numeric["areaMismatchRel"], "Activity-area mismatch", r"$||B_\rho|/|B_T|-1|$", False),
        )
        for axis, values, title, label, logarithmic in panels:
            save_grid_scatter(
                axis,
                stage_mask,
                values,
                title=title,
                label=label,
                logarithmic=logarithmic,
            )
        stage_fig.suptitle(
            f"Threshold-grid refinement level {stage_value}",
            fontsize=15,
        )
        stage_path = figure_dir / f"{prefix}_grid_level_{stage_value}.png"
        stage_fig.savefig(stage_path, dpi=260, facecolor="white")
        plt.close(stage_fig)
        stage_paths.append(stage_path)

    # Every independently seeded Newton attempt is retained.  Plotting it by
    # seed family exposes branch dependence that a one-row-per-pair CSV hides.
    branch_csv = workflow_dir / "grid_search" / "branch_trials.csv"
    branch_paths: list[Path] = []
    if branch_csv.is_file():
        with branch_csv.open(newline="", encoding="utf-8") as handle:
            branch_rows = list(csv.DictReader(handle))

        def branch_family(seed: str) -> str:
            if seed == "phi_target":
                return "target-potential seed"
            if seed == "torsion":
                return "torsion seed"
            return "continuation seed"

        families = (
            "target-potential seed",
            "torsion seed",
            "continuation seed",
        )
        branch_c1 = np.asarray([float(row["c1"]) for row in branch_rows])
        branch_c2 = np.asarray([float(row["c2"]) for row in branch_rows])
        branch_residual = np.asarray([float(row["residual"]) for row in branch_rows])
        branch_iterations = np.asarray(
            [float(row["newtonIterations"]) for row in branch_rows]
        )
        branch_time = np.asarray([float(row["wallTime"]) for row in branch_rows])
        branch_objective = np.asarray([float(row["objective"]) for row in branch_rows])
        branch_selected = np.asarray(
            [float(row["selected"]) > 0.5 for row in branch_rows]
        )
        branch_names = np.asarray(
            [branch_family(row["seed"]) for row in branch_rows]
        )
        branch_specs = (
            ("Terminal residual", branch_residual, r"$\|R\|_2$", True, "branch_residuals"),
            ("Newton iterations", branch_iterations, "iterations", False, "branch_newton_iterations"),
            ("Per-seed wall time", branch_time, "seconds", False, "branch_timing"),
            ("Per-seed objective", branch_objective, r"$J_{\rm grid}$", False, "branch_objective"),
        )
        for title, source_values, label, logarithmic, suffix in branch_specs:
            branch_fig, branch_axes = plt.subplots(
                1,
                3,
                figsize=(15.6, 4.8),
                constrained_layout=True,
                sharex=True,
                sharey=True,
            )
            for axis, family in zip(branch_axes, families):
                family_mask = branch_names == family
                values = source_values[family_mask]
                plot_label = label
                if logarithmic:
                    values = np.log10(np.maximum(values, 1.0e-16))
                    plot_label = rf"$\log_{{10}}$({label})"
                artist = axis.scatter(
                    branch_c1[family_mask],
                    branch_c2[family_mask],
                    c=values,
                    cmap="viridis_r",
                    s=38,
                    edgecolors="black",
                    linewidths=0.2,
                )
                selected_mask = family_mask & branch_selected
                axis.scatter(
                    branch_c1[selected_mask],
                    branch_c2[selected_mask],
                    s=82,
                    facecolors="none",
                    edgecolors="#ef3b2c",
                    linewidths=0.9,
                    label="selected branch",
                )
                branch_fig.colorbar(artist, ax=axis, label=plot_label)
                axis.set_title(family)
                axis.set_xlabel(r"$c_{1,\phi}$")
                axis.grid(alpha=0.2)
            branch_axes[0].set_ylabel(r"$c_{2,\phi}$")
            branch_axes[-1].legend(loc="best", frameon=True)
            branch_fig.suptitle(title, fontsize=15)
            output_path = figure_dir / f"{prefix}_{suffix}.png"
            branch_fig.savefig(output_path, dpi=260, facecolor="white")
            plt.close(branch_fig)
            branch_paths.append(output_path)

        lower_violation = np.asarray(
            [float(row["lowerBoundViolation"]) for row in branch_rows]
        )
        upper_violation = np.asarray(
            [float(row["upperBoundViolation"]) for row in branch_rows]
        )
        bound_fig, bound_axes = plt.subplots(
            1,
            2,
            figsize=(11.2, 4.8),
            constrained_layout=True,
        )
        for axis, violation, title in (
            (bound_axes[0], lower_violation, r"Lower violation $\max(0,-\phi)$"),
            (bound_axes[1], upper_violation, r"Upper violation $\max(0,\phi-T)$"),
        ):
            artist = axis.scatter(
                np.maximum(branch_residual, 1.0e-16),
                np.maximum(violation, 1.0e-18),
                c=branch_iterations,
                cmap="plasma",
                s=32,
                alpha=0.8,
                edgecolors="black",
                linewidths=0.18,
            )
            axis.figure.colorbar(artist, ax=axis, label="Newton iterations")
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.axvline(
                1.0e-6,
                color="#d62728",
                linestyle="--",
                linewidth=0.9,
                label=r"$10^{-6}$ grid tolerance",
            )
            axis.set_xlabel(r"terminal $\|R\|_2$")
            axis.set_ylabel("nodal violation")
            axis.set_title(title)
            axis.grid(alpha=0.2, which="both")
            axis.legend(loc="best")
        bound_output = figure_dir / f"{prefix}_maximum_principle.png"
        bound_fig.savefig(bound_output, dpi=260, facecolor="white")
        plt.close(bound_fig)
        branch_paths.append(bound_output)

    optimization_csv = workflow_dir / "optimization" / "logs" / "optimization.csv"
    optimization_paths: list[Path] = []
    if optimization_csv.is_file():
        with optimization_csv.open(newline="", encoding="utf-8") as handle:
            optimization_rows = list(csv.DictReader(handle))
        if optimization_rows:
            def opt_values(name: str, default: float = math.nan) -> np.ndarray:
                return np.asarray(
                    [float(row.get(name, default) or default) for row in optimization_rows],
                    dtype=np.float64,
                )

            opt_k = opt_values("k")
            opt_series = {
                "c1": opt_values("c1Phi"),
                "c2": opt_values("c2Phi"),
                "width": opt_values("width"),
                "eps": opt_values("epsPhi"),
                "residual": opt_values("residual"),
                "inner": opt_values("innerTol"),
                "gradient": opt_values("projectedGradNorm"),
                "trust": opt_values("trustRadius"),
                "leakage": opt_values("leakageRel"),
                "missing": opt_values("missingRel"),
                "newton": opt_values("newtonSteps"),
                "step_time": opt_values("stepTime"),
                "solve_time": opt_values("solveTime"),
                "ratio": opt_values("rhoRatio"),
                "accepted": opt_values("accepted", 0.0),
                "branch_overlap": opt_values("branchOverlap"),
                "trial_residual": opt_values("trialResidual"),
                "base_residual": opt_values("baseResidual"),
                "residual_ok": opt_values("residualOk"),
                "branch_ok": opt_values("branchOk"),
            }
            trace_fig, trace_axes = plt.subplots(
                3,
                2,
                figsize=(13.2, 12.2),
                constrained_layout=True,
            )
            trace_axes[0, 0].plot(opt_k, opt_series["c1"], "o-", label=r"$c_{1,\phi}$")
            trace_axes[0, 0].plot(opt_k, opt_series["c2"], "o-", label=r"$c_{2,\phi}$")
            trace_axes[0, 0].plot(opt_k, opt_series["width"], "o-", label=r"$\Delta c_\phi$")
            trace_axes[0, 0].plot(opt_k, opt_series["eps"], "o-", label=r"$\varepsilon_\phi$")
            trace_axes[0, 0].set_title("Threshold and smoothing evolution")
            trace_axes[0, 0].legend(ncol=2)
            trace_axes[0, 1].semilogy(opt_k, np.maximum(opt_series["residual"], 1.0e-16), "o-", label="PDE residual")
            trace_axes[0, 1].semilogy(opt_k, np.maximum(opt_series["inner"], 1.0e-16), "o--", label="requested inner tolerance")
            trace_axes[0, 1].set_title("Inexact intermediate solves")
            trace_axes[0, 1].legend()
            trace_axes[1, 0].semilogy(opt_k, np.maximum(opt_series["gradient"], 1.0e-16), "o-", label="projected gradient")
            trace_axes[1, 0].semilogy(opt_k, np.maximum(opt_series["trust"], 1.0e-16), "o-", label="trust radius")
            trace_axes[1, 0].set_title("Threshold-space convergence")
            trace_axes[1, 0].legend()
            trace_axes[1, 1].plot(opt_k, opt_series["leakage"], "o-", label="relative leakage")
            trace_axes[1, 1].plot(opt_k, opt_series["missing"], "o-", label="relative missing area")
            trace_axes[1, 1].set_title("Geometric mismatch")
            trace_axes[1, 1].legend()
            trace_axes[2, 0].plot(opt_k, opt_series["newton"], "o-", label="Newton iterations")
            trace_axes[2, 0].plot(opt_k, opt_series["ratio"], "o-", label="acceptance ratio")
            trace_axes[2, 0].set_title("Nonlinear work and acceptance")
            trace_axes[2, 0].legend()
            trace_axes[2, 1].plot(opt_k, opt_series["solve_time"], "o-", label="PDE solve")
            trace_axes[2, 1].plot(opt_k, opt_series["step_time"], "o-", label="outer step")
            trace_axes[2, 1].set_title("Per-iteration wall time")
            trace_axes[2, 1].legend()
            for axis in trace_axes.flat:
                axis.set_xlabel("outer iteration")
                axis.grid(alpha=0.24, which="both")
            trace_output = figure_dir / f"{prefix}_optimization_trace.png"
            trace_fig.savefig(trace_output, dpi=260, facecolor="white")
            plt.close(trace_fig)
            optimization_paths.append(trace_output)

            small_specs = (
                ("threshold_convergence", ("c1", "c2", "width"), (r"$c_{1,\phi}$", r"$c_{2,\phi}$", r"$\Delta c_\phi$"), "threshold value", False),
                ("pde_residual_convergence", ("residual", "inner"), ("PDE residual", "requested inner tolerance"), "residual / tolerance", True),
                ("geometric_convergence", ("leakage", "missing"), ("relative leakage", "relative missing area"), "relative discrepancy", False),
                ("newton_work", ("newton",), ("Newton iterations",), "Newton iterations", False),
                ("iteration_timing", ("solve_time", "step_time"), ("PDE solve", "outer step"), "seconds", False),
            )
            for name, keys, labels, ylabel, logarithmic in small_specs:
                small_fig, small_axis = plt.subplots(
                    figsize=(8.2, 5.2),
                    constrained_layout=True,
                )
                for key, label in zip(keys, labels):
                    values = opt_series[key]
                    if logarithmic:
                        small_axis.semilogy(
                            opt_k,
                            np.maximum(values, 1.0e-16),
                            "o-",
                            label=label,
                        )
                    else:
                        small_axis.plot(opt_k, values, "o-", label=label)
                small_axis.set_xlabel("outer iteration")
                small_axis.set_ylabel(ylabel)
                small_axis.grid(alpha=0.24, which="both")
                small_axis.legend()
                small_output = figure_dir / f"{prefix}_{name}.png"
                small_fig.savefig(small_output, dpi=280, facecolor="white")
                plt.close(small_fig)
                optimization_paths.append(small_output)

            acceptance_fig, acceptance_axes = plt.subplots(
                2,
                2,
                figsize=(12.4, 9.2),
                constrained_layout=True,
            )
            acceptance_axes[0, 0].semilogy(
                opt_k, np.maximum(opt_series["trust"], 1.0e-18), "o-"
            )
            acceptance_axes[0, 0].set_title("Trust-radius contraction")
            acceptance_axes[0, 0].set_ylabel("trust radius")
            acceptance_axes[0, 1].semilogy(
                opt_k, np.maximum(opt_series["branch_overlap"], 1.0e-18), "o-", label="branch overlap"
            )
            acceptance_axes[0, 1].axhline(
                0.10, color="#d62728", linestyle="--", linewidth=1.0, label="acceptance floor"
            )
            acceptance_axes[0, 1].set_title("Corrected branch-retention test")
            acceptance_axes[0, 1].legend()
            acceptance_axes[1, 0].semilogy(
                opt_k, np.maximum(opt_series["base_residual"], 1.0e-18), "o-", label="base residual"
            )
            acceptance_axes[1, 0].semilogy(
                opt_k, np.maximum(opt_series["trial_residual"], 1.0e-18), "o-", label="trial residual"
            )
            acceptance_axes[1, 0].semilogy(
                opt_k, np.maximum(opt_series["inner"], 1.0e-18), "--", label="inner tolerance"
            )
            acceptance_axes[1, 0].set_title("Base and trial PDE qualification")
            acceptance_axes[1, 0].legend()
            acceptance_axes[1, 1].step(
                opt_k, opt_series["residual_ok"], where="mid", label="residual filter"
            )
            acceptance_axes[1, 1].step(
                opt_k, opt_series["branch_ok"], where="mid", label="branch filter"
            )
            acceptance_axes[1, 1].scatter(
                opt_k[opt_series["accepted"] > 0.5],
                np.ones(np.count_nonzero(opt_series["accepted"] > 0.5)),
                marker="*",
                s=150,
                color="#00a65a",
                edgecolor="black",
                linewidth=0.5,
                label="accepted threshold step",
                zorder=5,
            )
            acceptance_axes[1, 1].set_ylim(-0.1, 1.2)
            acceptance_axes[1, 1].set_title("Acceptance-filter decisions")
            acceptance_axes[1, 1].legend()
            for axis in acceptance_axes.flat:
                axis.set_xlabel("outer iteration")
                axis.grid(alpha=0.24, which="both")
            acceptance_output = figure_dir / f"{prefix}_trial_acceptance.png"
            acceptance_fig.savefig(acceptance_output, dpi=280, facecolor="white")
            plt.close(acceptance_fig)
            optimization_paths.append(acceptance_output)

    sensitivity_paths: list[Path] = []
    sensitivity_candidates = (
        workflow_dir / "sensitivity_check_local_width_fine" / "logs" / "sensitivity_check.csv",
        workflow_dir / "sensitivity_check_local_width" / "logs" / "sensitivity_check.csv",
        workflow_dir / "optimization" / "logs" / "sensitivity_check.csv",
    )
    sensitivity_csv = next((path for path in sensitivity_candidates if path.is_file()), None)
    if sensitivity_csv is not None:
        with sensitivity_csv.open(newline="", encoding="utf-8") as handle:
            sensitivity_rows = list(csv.DictReader(handle))
        if sensitivity_rows:
            sensitivity_fig, sensitivity_axes = plt.subplots(
                1,
                2,
                figsize=(12.8, 5.3),
                constrained_layout=True,
            )
            colors = {1: "#0072b2", 2: "#d55e00"}
            for component in (1, 2):
                component_rows = sorted(
                    (row for row in sensitivity_rows if int(row["component"]) == component),
                    key=lambda row: float(row["step"]),
                )
                if not component_rows:
                    continue
                h_values = np.asarray([float(row["step"]) for row in component_rows])
                l2_errors = np.asarray([float(row["stateSensitivityL2RelativeError"]) for row in component_rows])
                h1_errors = np.asarray([float(row["stateSensitivityH1RelativeError"]) for row in component_rows])
                leakage_errors = np.asarray([float(row["leakageDerivativeRelativeError"]) for row in component_rows])
                missing_errors = np.asarray([float(row["missingDerivativeRelativeError"]) for row in component_rows])
                valid = np.asarray([int(row["valid"]) > 0 for row in component_rows])
                color = colors[component]
                sensitivity_axes[0].loglog(h_values, l2_errors, "o-", color=color, label=rf"$s_{component}$ $L^2$")
                sensitivity_axes[0].loglog(h_values, h1_errors, "s--", color=color, label=rf"$s_{component}$ $H^1$")
                sensitivity_axes[1].loglog(h_values, leakage_errors, "o-", color=color, label=rf"$\widehat L_{{,c_{component}}}$")
                sensitivity_axes[1].loglog(h_values, missing_errors, "s--", color=color, label=rf"$\widehat M_{{,c_{component}}}$")
                for axis, primary in ((sensitivity_axes[0], l2_errors), (sensitivity_axes[1], leakage_errors)):
                    if np.any(~valid):
                        axis.scatter(
                            h_values[~valid],
                            primary[~valid],
                            marker="x",
                            s=65,
                            color=color,
                            linewidths=1.3,
                            zorder=5,
                        )
            sensitivity_axes[0].set_title("State-sensitivity finite differences")
            sensitivity_axes[0].set_ylabel("relative error")
            sensitivity_axes[1].set_title("Reduced-gradient finite differences")
            for axis in sensitivity_axes:
                axis.set_xlabel(r"centered perturbation $h$")
                axis.grid(alpha=0.24, which="both")
                axis.legend()
            base_residual = float(sensitivity_rows[0]["baseResidual"])
            sensitivity_fig.suptitle(
                rf"ITER sensitivity verification; base residual ${base_residual:.2e}$; crosses mark an unconverged perturbed solve",
                fontsize=13,
            )
            sensitivity_output = figure_dir / f"{prefix}_sensitivity_finite_difference.png"
            sensitivity_fig.savefig(sensitivity_output, dpi=280, facecolor="white")
            plt.close(sensitivity_fig)
            sensitivity_paths.append(sensitivity_output)

    epsilon_paths: list[Path] = []
    epsilon_sources = (
        (0.04, workflow_dir / "epsilon_pair_control_eps004"),
        (0.02, workflow_dir / "optimization"),
        (0.01, workflow_dir / "epsilon_pair_control_eps001"),
    )
    epsilon_records: list[tuple[float, dict[str, str], Path]] = []
    for ratio, source_dir in epsilon_sources:
        csv_path = source_dir / "logs" / "optimization.csv"
        frames_csv_path = source_dir / "logs" / "frames.csv"
        if not csv_path.is_file() or not frames_csv_path.is_file():
            continue
        with csv_path.open(newline="", encoding="utf-8") as handle:
            control_rows = list(csv.DictReader(handle))
        with frames_csv_path.open(newline="", encoding="utf-8") as handle:
            control_frames = list(csv.DictReader(handle))
        if not control_rows or not control_frames:
            continue
        if ratio == 0.02:
            selected_row = control_rows[0]
            selected_frames = [
                row for row in control_frames
                if row["stage"] == "OPT" and int(row["k"]) == 0
            ]
        else:
            selected_row = control_rows[-1]
            selected_frames = [row for row in control_frames if row["stage"] == "FINAL"]
        if selected_frames and Path(selected_frames[-1]["filename"]).is_file():
            epsilon_records.append(
                (ratio, selected_row, Path(selected_frames[-1]["filename"]))
            )
    if len(epsilon_records) == len(epsilon_sources):
        epsilon_records.sort(key=lambda item: item[0])
        ratios = np.asarray([item[0] for item in epsilon_records])
        leakage = np.asarray([float(item[1]["leakageRel"]) for item in epsilon_records])
        missing = np.asarray([float(item[1]["missingRel"]) for item in epsilon_records])
        active_jaccard = np.asarray([float(item[1]["activeJaccard"]) for item in epsilon_records])
        rho_relative = np.asarray([float(item[1]["rhoRel"]) for item in epsilon_records])
        residual = np.asarray([float(item[1]["residual"]) for item in epsilon_records])
        epsilon_fig, epsilon_axes = plt.subplots(
            1,
            3,
            figsize=(15.0, 4.8),
            constrained_layout=True,
        )
        epsilon_axes[0].plot(ratios, leakage, "o-", label="relative leakage")
        epsilon_axes[0].plot(ratios, missing, "s--", label="relative missing area")
        epsilon_axes[0].set_title("Fixed-pair geometric discrepancy")
        epsilon_axes[0].legend()
        epsilon_axes[1].plot(ratios, active_jaccard, "o-", label="active Jaccard")
        epsilon_axes[1].plot(ratios, rho_relative, "s--", label="relative density mismatch")
        epsilon_axes[1].set_title("Overlap and density mismatch")
        epsilon_axes[1].legend()
        epsilon_axes[2].semilogy(ratios, np.maximum(residual, 1.0e-18), "o-")
        epsilon_axes[2].axhline(1.0e-11, color="#d62728", linestyle="--", linewidth=1.0)
        epsilon_axes[2].set_title("PDE residual qualification")
        epsilon_axes[2].set_ylabel(r"$\|R\|_2$")
        for axis in epsilon_axes:
            axis.set_xlabel(r"$\varepsilon_\phi/(c_{2,\phi}-c_{1,\phi})$")
            axis.grid(alpha=0.24, which="both")
        epsilon_output = figure_dir / f"{prefix}_epsilon_fixed_pair.png"
        epsilon_fig.savefig(epsilon_output, dpi=280, facecolor="white")
        plt.close(epsilon_fig)
        epsilon_paths.append(epsilon_output)

        epsilon_images = [Image.open(item[2]).convert("RGB") for item in epsilon_records]
        comparison_width = max(image.width for image in epsilon_images)
        comparison_gutter = 20
        resized_images = []
        for image_value in epsilon_images:
            if image_value.width != comparison_width:
                image_value = image_value.resize(
                    (
                        comparison_width,
                        round(image_value.height * comparison_width / image_value.width),
                    ),
                    Image.Resampling.LANCZOS,
                )
            resized_images.append(image_value)
        comparison_height = (
            sum(image.height for image in resized_images)
            + comparison_gutter * (len(resized_images) - 1)
        )
        comparison = Image.new("RGB", (comparison_width, comparison_height), "white")
        y_offset = 0
        for image_value in resized_images:
            comparison.paste(image_value, (0, y_offset))
            y_offset += image_value.height + comparison_gutter
        epsilon_fields_output = figure_dir / f"{prefix}_epsilon_fixed_pair_fields.png"
        comparison.save(epsilon_fields_output, format="PNG", compress_level=6)
        epsilon_paths.append(epsilon_fields_output)

    print(f"FIGURE {diagnostic_path}")
    print(f"FIGURE {storyboard_path}")
    for output_path in [
        *stage_paths,
        *branch_paths,
        *optimization_paths,
        *sensitivity_paths,
        *epsilon_paths,
        *contact_paths,
    ]:
        print(f"FIGURE {output_path}")
    print(f"FRAME_BUNDLE {frame_bundle} count={len(copied_frames)}")
    return 0
def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "search":
        return run_search(args)
    if args.command == "figures":
        return render_workflow_figures(
            args.workflow_dir,
            args.figure_dir,
            args.figure_prefix,
        )
    return run_workflow(args)


if __name__ == "__main__":
    raise SystemExit(main())
