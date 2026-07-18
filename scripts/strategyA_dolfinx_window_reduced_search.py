#!/usr/bin/env python3
"""Coarse reduced search for Strategy A semilinear window thresholds.

For fixed torsion parameters ``alphaT1, alphaT2`` and a fixed semilinear
epsilon ratio, this script searches over admissible ``(c1Phi, c2Phi)`` pairs.
Each candidate pair is evaluated by solving

    -Delta phi = W(phi; c1Phi, c2Phi, epsPhi),   phi|boundary = 0,

then scoring the converged state against the torsion-designed active band.
This is expensive but directly measures the closed-loop map

    (c1Phi, c2Phi) -> phi(c1Phi, c2Phi).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from strategyA_dolfinx_noadapt_torsion_newton_v2 import (  # noqa: E402
    PyVistaStrategyPlotter,
    assemble_scalar,
    boundary_bc,
    compute_metrics,
    fit_phi_window_to_torsion_design,
    global_minmax,
    load_or_generate_mesh,
    logistic_ufl,
    root_print,
    slug_for_path,
    solve_linear_form,
    update_interpolated,
    window_ufl,
)


DEFAULT_RUN_LOG_ROOT = REPO_ROOT / "run_logs" / "dolfinx_window_reduced_search"


@dataclass
class StrategyParameters:
    """Fixed Strategy A target parameters."""

    alpha_t1: float = 0.60
    alpha_t2: float = 0.70
    eps_t_ratio: float = 0.06
    rho_amp: float = 1.0
    active_threshold: float = 0.05
    plateau_threshold: float = 0.90


@dataclass
class CandidateResult:
    """Result of one closed-loop candidate solve."""

    pass_index: int
    candidate_index: int
    c1: float
    c2: float
    eps_phi: float
    score: float
    status: str
    converged: bool
    newton_steps: int
    residual: float
    phi_l2: float
    phi_rel: float
    rho_l2: float
    rho_rel: float
    mass_diff: float
    mass_rel: float
    active_jaccard: float
    active_overlap_area: float
    active_recall: float
    active_precision: float
    active_dice: float
    plateau_jaccard: float
    active_area: float
    active_design_area: float
    plateau_area: float
    solve_time: float
    best_coeffs: np.ndarray


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


def params_from_args(args: argparse.Namespace) -> StrategyParameters:
    params = StrategyParameters()
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
        raise ValueError("require positive --eps-t-ratio")
    return params


def logistic_const_ufl(z, eps):
    zz = z / eps
    return ufl.conditional(
        ufl.gt(zz, 50.0),
        1.0,
        ufl.conditional(ufl.lt(zz, -50.0), 0.0, 1.0 / (1.0 + ufl.exp(-zz))),
    )


def window_const_ufl(values, c1, c2, eps, amp: float):
    return float(amp) * (logistic_const_ufl(values - c1, eps) - logistic_const_ufl(values - c2, eps))


def window_const_derivative_ufl(values, c1, c2, eps, amp: float):
    s1 = logistic_const_ufl(values - c1, eps)
    s2 = logistic_const_ufl(values - c2, eps)
    return float(amp) * (s1 * (1.0 - s1) - s2 * (1.0 - s2)) / eps


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--mesh", type=Path, default=None)
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
    parser.add_argument("--eps-t-ratio", dest="eps_t_ratio", type=float, default=None)
    parser.add_argument("--eps-ratio", "--eps-phi-ratio", dest="eps_ratio", type=float, default=0.08)
    parser.add_argument("--rho-amp", type=float, default=None)
    parser.add_argument("--cmax-factor", type=float, default=1.25)
    parser.add_argument("--c-lower-fraction", type=float, default=0.0)
    parser.add_argument("--c-upper-fraction", type=float, default=1.0)
    parser.add_argument("--min-width-fraction", type=float, default=1.0e-3)
    parser.add_argument("--candidate-mode", choices=("smart", "triangular", "hybrid"), default="smart")
    parser.add_argument("--candidate-width-grid", type=int, default=None)
    parser.add_argument("--candidate-center-lower-fraction", type=float, default=0.45)
    parser.add_argument("--candidate-center-upper-fraction", type=float, default=0.90)
    parser.add_argument("--candidate-width-min-factor", type=float, default=0.50)
    parser.add_argument("--candidate-width-max-factor", type=float, default=2.50)
    parser.add_argument("--grid", type=int, default=8, help="initial grid points per c-axis")
    parser.add_argument("--refine-passes", type=int, default=2)
    parser.add_argument("--refine-grid", type=int, default=5)
    parser.add_argument("--include-fit-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fit-window-grid", type=int, default=64)
    parser.add_argument("--fit-window-refine-grid", type=int, default=25)
    parser.add_argument("--fit-window-refine-passes", type=int, default=2)
    parser.add_argument("--fit-window-bins", type=int, default=4096)
    parser.add_argument("--fit-window-quad-degree", type=int, default=None)
    parser.add_argument("--initial-state", choices=("phi-target", "previous-best"), default="phi-target")
    parser.add_argument("--max-newton-it", type=int, default=40)
    parser.add_argument("--tol-res", type=float, default=1.0e-10)
    parser.add_argument("--tol-step", type=float, default=1.0e-10)
    parser.add_argument("--max-backtrack", type=int, default=24)
    parser.add_argument("--alpha-min", type=float, default=1.0e-8)
    parser.add_argument("--beta-ls", type=float, default=0.5)
    parser.add_argument("--armijo-c", type=float, default=1.0e-6)
    parser.add_argument("--score-mode", choices=("overlap", "weighted"), default="overlap")
    parser.add_argument("--score-overlap-metric", choices=("jaccard", "recall", "dice"), default="jaccard")
    parser.add_argument("--score-potential-weight", type=float, default=0.0)
    parser.add_argument("--score-rho-weight", type=float, default=0.0)
    parser.add_argument("--score-mass-weight", type=float, default=0.0)
    parser.add_argument("--score-active-weight", type=float, default=1.0)
    parser.add_argument("--score-plateau-weight", type=float, default=0.0)
    parser.add_argument("--score-residual-weight", type=float, default=1000.0)
    parser.add_argument("--score-active-recall-weight", type=float, default=0.0)
    parser.add_argument("--score-active-precision-weight", type=float, default=0.0)
    parser.add_argument("--failed-score-penalty", type=float, default=1.0e3)
    parser.add_argument("--linear-solver", choices=("mumps", "lu", "hypre", "gamg"), default="mumps")
    parser.add_argument("--ksp-type", default=None)
    parser.add_argument("--linear-rtol", type=float, default=1.0e-10)
    parser.add_argument("--linear-atol", type=float, default=1.0e-12)
    parser.add_argument("--linear-max-it", type=int, default=None)
    parser.add_argument("--terminal-every", type=int, default=1)
    parser.add_argument("--verbosity", "-v", type=int, choices=(0, 1, 2, 3), default=1)
    parser.add_argument("--fail-on-nonconvergence", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-mode", choices=("blocking", "nonblocking"), default="blocking")
    parser.add_argument("--plot-off-screen", action="store_true")
    parser.add_argument("--plot-window-width", type=int, default=1800)
    parser.add_argument("--plot-window-height", type=int, default=700)
    parser.add_argument("--plot-initial", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-design", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-candidate-every", type=int, default=1)
    parser.add_argument("--plot-best-each-pass", action=argparse.BooleanOptionalAction, default=False)
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
    if args.eps_ratio <= 0.0:
        raise ValueError("require positive --eps-ratio")
    if args.cmax_factor <= 0.0:
        raise ValueError("require positive --cmax-factor")
    if args.c_lower_fraction < 0.0 or args.c_upper_fraction <= args.c_lower_fraction:
        raise ValueError("require 0 <= c-lower-fraction < c-upper-fraction")
    if args.min_width_fraction <= 0.0:
        raise ValueError("require positive --min-width-fraction")
    if args.candidate_center_lower_fraction < 0.0:
        raise ValueError("require nonnegative --candidate-center-lower-fraction")
    if args.candidate_center_upper_fraction <= args.candidate_center_lower_fraction:
        raise ValueError("require candidate center upper fraction > lower fraction")
    if args.candidate_width_min_factor <= 0.0 or args.candidate_width_max_factor < args.candidate_width_min_factor:
        raise ValueError("require 0 < candidate width min factor <= max factor")
    if args.candidate_width_grid is not None and args.candidate_width_grid < 1:
        raise ValueError("require positive --candidate-width-grid")
    if args.grid < 3:
        raise ValueError("require --grid >= 3")
    if args.refine_grid < 3:
        raise ValueError("require --refine-grid >= 3")
    if args.refine_passes < 0:
        raise ValueError("require nonnegative --refine-passes")
    if args.max_newton_it < 1:
        raise ValueError("require positive --max-newton-it")
    if args.plot_candidate_every < 1:
        raise ValueError("require positive --plot-candidate-every")


def unique_candidates(candidates: list[tuple[float, float]], *, tol: float) -> list[tuple[float, float]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[float, float]] = []
    scale = max(float(tol), 1.0e-16)
    for c1, c2 in candidates:
        key = (int(round(c1 / scale)), int(round(c2 / scale)))
        if key in seen:
            continue
        seen.add(key)
        out.append((float(c1), float(c2)))
    return out


def initial_candidates(
        *,
        c_min: float,
        c_max: float,
        min_width: float,
        grid: int,
        width_grid: int,
        mode: str,
        center_lower_fraction: float,
        center_upper_fraction: float,
        width_min_factor: float,
        width_max_factor: float,
        include_pair: tuple[float, float] | None,
) -> list[tuple[float, float]]:
    candidates: list[tuple[float, float]] = []
    if mode in ("triangular", "hybrid"):
        values = np.linspace(float(c_min), float(c_max), int(grid), dtype=np.float64)
        for i, c1 in enumerate(values[:-1]):
            for c2 in values[i + 1:]:
                if c2 - c1 >= min_width:
                    candidates.append((float(c1), float(c2)))

    if mode in ("smart", "hybrid"):
        domain_width = float(c_max) - float(c_min)
        if include_pair is not None and include_pair[1] > include_pair[0]:
            base_width = float(include_pair[1] - include_pair[0])
        else:
            base_width = domain_width / max(int(grid), 1)
        width_lo = max(float(min_width), width_min_factor * base_width)
        width_hi = min(domain_width, max(width_lo, width_max_factor * base_width))
        widths = np.linspace(width_lo, width_hi, int(width_grid), dtype=np.float64)
        center_lo = float(c_min) + center_lower_fraction * domain_width
        center_hi = float(c_min) + center_upper_fraction * domain_width
        centers = np.linspace(center_lo, center_hi, int(grid), dtype=np.float64)
        for width in widths:
            half_width = 0.5 * float(width)
            for center in centers:
                c1 = float(center) - half_width
                c2 = float(center) + half_width
                if c1 < c_min or c2 > c_max or c2 - c1 < min_width:
                    continue
                candidates.append((float(c1), float(c2)))
    if include_pair is not None and include_pair[1] - include_pair[0] >= min_width:
        candidates.append(include_pair)
    return unique_candidates(candidates, tol=1.0e-12 * max(c_max, 1.0))


def refined_candidates(
        *,
        best_c1: float,
        best_c2: float,
        c_min: float,
        c_max: float,
        min_width: float,
        radius: float,
        grid: int,
        width_grid: int,
        mode: str,
) -> list[tuple[float, float]]:
    candidates: list[tuple[float, float]] = []
    if mode in ("triangular", "hybrid"):
        lo1 = max(float(c_min), float(best_c1) - float(radius))
        hi1 = min(float(c_max) - min_width, float(best_c1) + float(radius))
        lo2 = max(float(c_min) + min_width, float(best_c2) - float(radius))
        hi2 = min(float(c_max), float(best_c2) + float(radius))
        if hi1 > lo1 and hi2 > lo2:
            c1_values = np.linspace(lo1, hi1, int(grid), dtype=np.float64)
            c2_values = np.linspace(lo2, hi2, int(grid), dtype=np.float64)
            candidates.extend(
                (float(c1), float(c2))
                for c1 in c1_values
                for c2 in c2_values
                if c2 - c1 >= min_width
            )

    if mode in ("smart", "hybrid"):
        best_center = 0.5 * (float(best_c1) + float(best_c2))
        best_width = float(best_c2) - float(best_c1)
        center_lo = max(float(c_min) + 0.5 * min_width, best_center - float(radius))
        center_hi = min(float(c_max) - 0.5 * min_width, best_center + float(radius))
        width_lo = max(float(min_width), best_width - float(radius))
        width_hi = min(float(c_max) - float(c_min), best_width + float(radius))
        if center_hi > center_lo and width_hi >= width_lo:
            centers = np.linspace(center_lo, center_hi, int(grid), dtype=np.float64)
            widths = np.linspace(width_lo, width_hi, int(width_grid), dtype=np.float64)
            for width in widths:
                half_width = 0.5 * float(width)
                for center in centers:
                    c1 = float(center) - half_width
                    c2 = float(center) + half_width
                    if c1 < c_min or c2 > c_max or c2 - c1 < min_width:
                        continue
                    candidates.append((float(c1), float(c2)))
    candidates.append((float(best_c1), float(best_c2)))
    return unique_candidates(candidates, tol=1.0e-12 * max(c_max, 1.0))


def score_candidate(
        *,
        phi_l2: float,
        phi_target_l2: float,
        rho_l2: float,
        rho_design_l2: float,
        mass_diff: float,
        rho_design_mass: float,
        active_jaccard: float,
        active_recall: float,
        active_precision: float,
        active_dice: float,
        plateau_jaccard: float,
        residual: float,
        converged: bool,
        args: argparse.Namespace,
) -> float:
    phi_rel = phi_l2 / max(phi_target_l2, 1.0e-30)
    rho_rel = rho_l2 / max(rho_design_l2, 1.0e-30)
    mass_rel = mass_diff / max(abs(rho_design_mass), 1.0e-30)
    if args.score_mode == "overlap":
        if args.score_overlap_metric == "recall":
            overlap_miss = 1.0 - active_recall
        elif args.score_overlap_metric == "dice":
            overlap_miss = 1.0 - active_dice
        else:
            overlap_miss = 1.0 - active_jaccard
        score = (
            args.score_active_weight * overlap_miss
            + args.score_active_recall_weight * (1.0 - active_recall)
            + args.score_active_precision_weight * (1.0 - active_precision)
            + 0.5 * args.score_residual_weight * residual * residual
            + 0.5 * args.score_mass_weight * mass_rel * mass_rel
            + 0.5 * args.score_rho_weight * rho_rel * rho_rel
            + 0.5 * args.score_potential_weight * phi_rel * phi_rel
            + args.score_plateau_weight * (1.0 - plateau_jaccard)
        )
    else:
        score = (
            0.5 * args.score_potential_weight * phi_rel * phi_rel
            + 0.5 * args.score_rho_weight * rho_rel * rho_rel
            + 0.5 * args.score_mass_weight * mass_rel * mass_rel
            + args.score_active_weight * (1.0 - active_jaccard)
            + args.score_active_recall_weight * (1.0 - active_recall)
            + args.score_active_precision_weight * (1.0 - active_precision)
            + args.score_plateau_weight * (1.0 - plateau_jaccard)
            + 0.5 * args.score_residual_weight * residual * residual
        )
    if not converged:
        score += args.failed_score_penalty
    return float(score)


def solve_candidate(
        *,
        pass_index: int,
        candidate_index: int,
        c1: float,
        c2: float,
        initial_coeffs: np.ndarray,
        u: fem.Function,
        du: fem.Function,
        rho: fem.Function,
        phi_target: fem.Function,
        rho_design: fem.Function,
        c1_const: fem.Constant,
        c2_const: fem.Constant,
        eps_const: fem.Constant,
        residual_form,
        jac_form,
        residual_form_for_metrics,
        dx,
        bc,
        rho_design_l2: float,
        phi_target_l2: float,
        rho_design_mass: float,
        args: argparse.Namespace,
        params: StrategyParameters,
) -> CandidateResult:
    start = time.perf_counter()
    eps_phi = args.eps_ratio * (c2 - c1)
    c1_const.value = PETSc.ScalarType(c1)
    c2_const.value = PETSc.ScalarType(c2)
    eps_const.value = PETSc.ScalarType(eps_phi)
    u.x.array[:] = initial_coeffs
    u.x.scatter_forward()
    status = "MAX_NEWTON"
    converged = False
    final_metrics: dict[str, float] | None = None
    final_phi_l2 = math.inf
    final_rho_l2 = math.inf
    final_steps = 0

    for k in range(args.max_newton_it):
        update_interpolated(rho, window_const_ufl(u, c1_const, c2_const, eps_const, params.rho_amp))
        metrics = compute_metrics(
            u=u,
            rho=rho,
            rho_design=rho_design,
            rho_design_l2=rho_design_l2,
            residual_form=residual_form_for_metrics,
            bc=bc,
            dx=dx,
            c2_phi=c2,
            active_threshold=params.active_threshold,
            plateau_threshold=params.plateau_threshold,
            rho_amp=params.rho_amp,
        )
        final_metrics = metrics
        residual_old = metrics["resEuclid"]
        final_steps = k
        if residual_old < args.tol_res:
            status = "CONVERGED_RESIDUAL"
            converged = True
            break

        solve_linear_form(
            jac_form,
            -residual_form,
            du,
            [bc],
            prefix=f"candidate_{pass_index}_{candidate_index}_{k}_",
            solver=args.linear_solver,
            ksp_type=args.ksp_type,
            rtol=args.linear_rtol,
            atol=args.linear_atol,
            max_it=args.linear_max_it,
            verbosity=args.verbosity,
        )
        step_h1 = math.sqrt(max(assemble_scalar(u.function_space.mesh.comm, ufl.inner(ufl.grad(du), ufl.grad(du)) * dx), 0.0))
        if step_h1 < args.tol_step:
            status = "CONVERGED_STEP"
            converged = residual_old < max(10.0 * args.tol_res, 1.0e-8)
            break

        old_u = u.x.array.copy()
        alpha = 1.0
        accepted = False
        for _ in range(args.max_backtrack + 1):
            u.x.array[:] = old_u + alpha * du.x.array
            u.x.scatter_forward()
            update_interpolated(rho, window_const_ufl(u, c1_const, c2_const, eps_const, params.rho_amp))
            trial_metrics = compute_metrics(
                u=u,
                rho=rho,
                rho_design=rho_design,
                rho_design_l2=rho_design_l2,
                residual_form=residual_form_for_metrics,
                bc=bc,
                dx=dx,
                c2_phi=c2,
                active_threshold=params.active_threshold,
                plateau_threshold=params.plateau_threshold,
                rho_amp=params.rho_amp,
            )
            if trial_metrics["resEuclid"] <= (1.0 - args.armijo_c * alpha) * residual_old:
                final_metrics = trial_metrics
                accepted = True
                break
            alpha *= args.beta_ls
            if alpha < args.alpha_min:
                break
        if not accepted:
            u.x.array[:] = old_u
            u.x.scatter_forward()
            status = "FAIL_LS"
            break
    else:
        final_steps = args.max_newton_it

    update_interpolated(rho, window_const_ufl(u, c1_const, c2_const, eps_const, params.rho_amp))
    if final_metrics is None:
        final_metrics = compute_metrics(
            u=u,
            rho=rho,
            rho_design=rho_design,
            rho_design_l2=rho_design_l2,
            residual_form=residual_form_for_metrics,
            bc=bc,
            dx=dx,
            c2_phi=c2,
            active_threshold=params.active_threshold,
            plateau_threshold=params.plateau_threshold,
            rho_amp=params.rho_amp,
        )
    final_phi_l2 = math.sqrt(max(assemble_scalar(u.function_space.mesh.comm, (u - phi_target) ** 2 * dx), 0.0))
    final_rho_l2 = final_metrics["rhoDesignDiffL2"]
    phi_rel = final_phi_l2 / max(phi_target_l2, 1.0e-30)
    rho_rel = final_rho_l2 / max(rho_design_l2, 1.0e-30)
    mass_rel = final_metrics["massRhoMinusDesign"] / max(abs(rho_design_mass), 1.0e-30)
    active_overlap_area = final_metrics["activeOverlapArea"]
    active_recall = active_overlap_area / max(final_metrics["activeDesignArea"], 1.0e-30)
    active_precision = active_overlap_area / max(final_metrics["activeArea"], 1.0e-30)
    active_dice = 2.0 * active_overlap_area / max(final_metrics["activeDesignArea"] + final_metrics["activeArea"], 1.0e-30)
    score = score_candidate(
        phi_l2=final_phi_l2,
        phi_target_l2=phi_target_l2,
        rho_l2=final_rho_l2,
        rho_design_l2=rho_design_l2,
        mass_diff=final_metrics["massRhoMinusDesign"],
        rho_design_mass=rho_design_mass,
        active_jaccard=final_metrics["activeJaccard"],
        active_recall=active_recall,
        active_precision=active_precision,
        active_dice=active_dice,
        plateau_jaccard=final_metrics["plateauJaccard"],
        residual=final_metrics["resEuclid"],
        converged=converged,
        args=args,
    )
    return CandidateResult(
        pass_index=pass_index,
        candidate_index=candidate_index,
        c1=float(c1),
        c2=float(c2),
        eps_phi=float(eps_phi),
        score=score,
        status=status,
        converged=converged,
        newton_steps=final_steps,
        residual=final_metrics["resEuclid"],
        phi_l2=final_phi_l2,
        phi_rel=phi_rel,
        rho_l2=final_rho_l2,
        rho_rel=rho_rel,
        mass_diff=final_metrics["massRhoMinusDesign"],
        mass_rel=mass_rel,
        active_jaccard=final_metrics["activeJaccard"],
        active_overlap_area=active_overlap_area,
        active_recall=active_recall,
        active_precision=active_precision,
        active_dice=active_dice,
        plateau_jaccard=final_metrics["plateauJaccard"],
        active_area=final_metrics["activeArea"],
        active_design_area=final_metrics["activeDesignArea"],
        plateau_area=final_metrics["plateauArea"],
        solve_time=time.perf_counter() - start,
        best_coeffs=u.x.array.copy(),
    )


def write_candidate(writer: csv.DictWriter, result: CandidateResult, run_tag: str) -> None:
    writer.writerow({
        "record": "CANDIDATE",
        "runTag": run_tag,
        "pass": result.pass_index,
        "candidate": result.candidate_index,
        "c1Phi": result.c1,
        "c2Phi": result.c2,
        "width": result.c2 - result.c1,
        "epsPhi": result.eps_phi,
        "score": result.score,
        "status": result.status,
        "converged": int(result.converged),
        "newtonSteps": result.newton_steps,
        "residual": result.residual,
        "phiL2": result.phi_l2,
        "phiRel": result.phi_rel,
        "rhoL2": result.rho_l2,
        "rhoRel": result.rho_rel,
        "massDiff": result.mass_diff,
        "massRel": result.mass_rel,
        "activeJaccard": result.active_jaccard,
        "activeOverlapArea": result.active_overlap_area,
        "activeRecall": result.active_recall,
        "activePrecision": result.active_precision,
        "activeDice": result.active_dice,
        "plateauJaccard": result.plateau_jaccard,
        "activeArea": result.active_area,
        "activeDesignArea": result.active_design_area,
        "plateauArea": result.plateau_area,
        "solveTime": result.solve_time,
    })


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

    candidates_csv = log_dir / "candidates.csv"
    frame_csv = log_dir / "frames.csv"
    summary_path = out_dir / "summary.txt"
    total_start = time.perf_counter()

    root_print(comm, "========== START STRATEGY A DOLFINX WINDOW REDUCED SEARCH ==========")
    root_print(comm, f"RUN_TAG {run_tag}")
    root_print(comm, f"RUN_DIR {run_dir}")
    root_print(comm, f"CANDIDATES_CSV {candidates_csv}")
    root_print(comm, f"FRAME_CSV {frame_csv}")
    root_print(comm, f"SUMMARY {summary_path}")
    root_print(
        comm,
        "SCORE_WEIGHTS "
        f"mode={args.score_mode} "
        f"overlapMetric={args.score_overlap_metric} "
        f"phi={args.score_potential_weight:.6e} "
        f"rho={args.score_rho_weight:.6e} "
        f"mass={args.score_mass_weight:.6e} "
        f"active={args.score_active_weight:.6e} "
        f"activeRecall={args.score_active_recall_weight:.6e} "
        f"activePrecision={args.score_active_precision_weight:.6e} "
        f"plateau={args.score_plateau_weight:.6e} "
        f"residual={args.score_residual_weight:.6e}",
    )
    root_print(
        comm,
        "CANDIDATE_GENERATION "
        f"mode={args.candidate_mode} "
        f"centerFrac=[{args.candidate_center_lower_fraction:.6e},{args.candidate_center_upper_fraction:.6e}] "
        f"widthFactor=[{args.candidate_width_min_factor:.6e},{args.candidate_width_max_factor:.6e}]",
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

    frame_fields = [
        "frame", "runTag", "stage", "ieps", "k", "nt", "ndof", "epsPhi", "resEuclid",
        "massRho", "maxRho", "activeArea", "plateauArea", "relRhoDesign", "filename",
    ]
    frame_handle = frame_csv.open("w", newline="", encoding="utf-8") if comm.rank == 0 else None
    frame_writer = csv.DictWriter(frame_handle, fieldnames=frame_fields) if comm.rank == 0 else None
    if frame_writer is not None:
        frame_writer.writeheader()
    plotter = PyVistaStrategyPlotter(args, run_tag=run_tag, run_dir=run_dir, frame_writer=frame_writer, comm=comm)

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
    trial = ufl.TrialFunction(V)
    test = ufl.TestFunction(V)

    T = fem.Function(V, name="T")
    rho_design = fem.Function(V, name="rhoDesign")
    phi_target = fem.Function(V, name="phiT")
    u = fem.Function(V, name="phi")
    du = fem.Function(V, name="du")
    rho = fem.Function(V, name="rho")
    best_phi = fem.Function(V, name="bestPhi")
    best_rho = fem.Function(V, name="bestRho")
    phi_diff = fem.Function(V, name="bestPhiMinusPhiT")

    its, rel, solve_time = solve_linear_form(
        ufl.inner(ufl.grad(trial), ufl.grad(test)) * dx,
        1.0 * test * dx,
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

    update_interpolated(rho_design, window_ufl(T, c1_t, c2_t, eps_t, params.rho_amp))
    rho_design_l2 = math.sqrt(max(assemble_scalar(comm, rho_design * rho_design * dx), 0.0))
    rho_design_mass = assemble_scalar(comm, rho_design * dx)
    _, rho_design_max = global_minmax(comm, rho_design)

    its, rel, solve_time = solve_linear_form(
        ufl.inner(ufl.grad(trial), ufl.grad(test)) * dx,
        rho_design * test * dx,
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
    phi_target_l2 = math.sqrt(max(assemble_scalar(comm, phi_target * phi_target * dx), 0.0))
    c_max = max(float(args.cmax_factor) * phi_target_max, 1.0e-12)
    c_min = float(args.c_lower_fraction) * c_max
    c_upper = float(args.c_upper_fraction) * c_max
    min_width = max(float(args.min_width_fraction) * c_max, 1.0e-14)
    root_print(comm, f"SOLVER_OK problem=phi_target iters={its} rel={rel:.3e} time={solve_time:.3f}")
    root_print(comm, f"PHI_TARGET max={phi_target_max:.6e} rhoDesignMass={rho_design_mass:.6e} rhoDesignMax={rho_design_max:.6e}")
    root_print(comm, f"SEARCH_DOMAIN cMin={c_min:.6e} cMax={c_upper:.6e} minWidth={min_width:.6e} epsRatio={args.eps_ratio:.6e}")

    fit_pair = None
    fit_result = None
    if args.include_fit_init:
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
        fit_pair = (fit_result.c1, fit_result.c2)
        root_print(
            comm,
            f"FIT_INIT c1={fit_result.c1:.6e} c2={fit_result.c2:.6e} "
            f"objectiveRel={fit_result.objective_rel:.6e} time={fit_result.elapsed:.3f}",
        )

    if args.plot_design:
        plotter.emit(
            [T, rho_design, phi_target],
            ["Torsion T", "rhoDesign", "phiT"],
            stage="DESIGN",
            ieps=-1,
            k=-1,
            eps_phi=eps_t,
            residual=0.0,
            metrics={"massRho": rho_design_mass, "maxRho": rho_design_max},
            token="design",
            save=bool(args.frame_design),
            show=True,
            nt=nt,
            ndof=ndof,
        )

    c1_const = fem.Constant(domain, PETSc.ScalarType(0.0))
    c2_const = fem.Constant(domain, PETSc.ScalarType(1.0))
    eps_const = fem.Constant(domain, PETSc.ScalarType(1.0))
    rho_expr = window_const_ufl(u, c1_const, c2_const, eps_const, params.rho_amp)
    residual_form = (ufl.inner(ufl.grad(u), ufl.grad(test)) - rho_expr * test) * dx
    jac_form = (
        ufl.inner(ufl.grad(trial), ufl.grad(test))
        - window_const_derivative_ufl(u, c1_const, c2_const, eps_const, params.rho_amp) * trial * test
    ) * dx

    candidate_fields = [
        "record", "runTag", "pass", "candidate", "c1Phi", "c2Phi", "width", "epsPhi",
        "score", "status", "converged", "newtonSteps", "residual",
        "phiL2", "phiRel", "rhoL2", "rhoRel", "massDiff", "massRel",
        "activeJaccard", "activeOverlapArea", "activeRecall", "activePrecision",
        "activeDice", "plateauJaccard", "activeArea", "activeDesignArea",
        "plateauArea", "solveTime",
    ]
    candidate_handle = candidates_csv.open("w", newline="", encoding="utf-8") if comm.rank == 0 else None
    writer = csv.DictWriter(candidate_handle, fieldnames=candidate_fields) if comm.rank == 0 else None
    if writer is not None:
        writer.writeheader()

    all_best: CandidateResult | None = None
    pass_best: CandidateResult | None = None
    best_coeffs = phi_target.x.array.copy()
    initial_coeffs = phi_target.x.array.copy()
    width_grid = args.candidate_width_grid if args.candidate_width_grid is not None else max(3, args.grid // 2 + 1)
    radius = (c_upper - c_min) / max(args.grid - 1, 1)
    candidates = initial_candidates(
        c_min=c_min,
        c_max=c_upper,
        min_width=min_width,
        grid=args.grid,
        width_grid=width_grid,
        mode=args.candidate_mode,
        center_lower_fraction=args.candidate_center_lower_fraction,
        center_upper_fraction=args.candidate_center_upper_fraction,
        width_min_factor=args.candidate_width_min_factor,
        width_max_factor=args.candidate_width_max_factor,
        include_pair=fit_pair,
    )

    try:
        for pass_index in range(args.refine_passes + 1):
            root_print(comm, f"SEARCH_PASS_START pass={pass_index} candidates={len(candidates)} radius={radius:.6e}")
            pass_best = None
            for candidate_index, (c1_phi, c2_phi) in enumerate(candidates):
                if args.initial_state == "previous-best" and all_best is not None:
                    initial_coeffs = best_coeffs
                else:
                    initial_coeffs = phi_target.x.array
                result = solve_candidate(
                    pass_index=pass_index,
                    candidate_index=candidate_index,
                    c1=c1_phi,
                    c2=c2_phi,
                    initial_coeffs=initial_coeffs,
                    u=u,
                    du=du,
                    rho=rho,
                    phi_target=phi_target,
                    rho_design=rho_design,
                    c1_const=c1_const,
                    c2_const=c2_const,
                    eps_const=eps_const,
                    residual_form=residual_form,
                    jac_form=jac_form,
                    residual_form_for_metrics=residual_form,
                    dx=dx,
                    bc=bc,
                    rho_design_l2=rho_design_l2,
                    phi_target_l2=phi_target_l2,
                    rho_design_mass=rho_design_mass,
                    args=args,
                    params=params,
                )
                if writer is not None:
                    write_candidate(writer, result, run_tag)
                    candidate_handle.flush()
                if pass_best is None or result.score < pass_best.score:
                    pass_best = result
                if all_best is None or result.score < all_best.score:
                    all_best = result
                    best_coeffs = result.best_coeffs.copy()
                if args.terminal_every > 0 and (
                        candidate_index % args.terminal_every == 0
                        or result is pass_best
                ):
                    root_print(
                        comm,
                        f"CANDIDATE pass={pass_index} i={candidate_index} score={result.score:.6e} "
                        f"c1={result.c1:.6e} c2={result.c2:.6e} res={result.residual:.6e} "
                        f"rhoRel={result.rho_rel:.6e} massRel={result.mass_rel:.6e} "
                        f"activeJ={result.active_jaccard:.6e} activeRecall={result.active_recall:.6e} "
                        f"activePrecision={result.active_precision:.6e} activeDice={result.active_dice:.6e} "
                        f"status={result.status} "
                        f"time={result.solve_time:.3f}",
                    )
                if args.plot_candidates and candidate_index % args.plot_candidate_every == 0:
                    best_phi.x.array[:] = result.best_coeffs
                    best_phi.x.scatter_forward()
                    update_interpolated(
                        best_rho,
                        window_ufl(best_phi, result.c1, result.c2, result.eps_phi, params.rho_amp),
                    )
                    phi_diff.x.array[:] = best_phi.x.array - phi_target.x.array
                    phi_diff.x.scatter_forward()
                    plotter.emit(
                        [T, rho_design, phi_target, best_phi, best_rho, phi_diff],
                        [
                            "Torsion T",
                            "rhoDesign",
                            "phiT",
                            f"candidate phi pass={pass_index} i={candidate_index}",
                            "rho(candidate)",
                            "phi-phiT",
                        ],
                        stage="CANDIDATE",
                        ieps=pass_index,
                        k=candidate_index,
                        eps_phi=result.eps_phi,
                        residual=result.residual,
                        metrics={
                            "massRho": result.mass_diff + rho_design_mass,
                            "maxRho": 0.0,
                            "activeArea": result.active_area,
                            "plateauArea": result.plateau_area,
                            "relRhoDesign": result.rho_rel,
                        },
                        token=f"pass_{pass_index}_candidate_{candidate_index}",
                        save=bool(
                            args.save_frames
                            and args.frame_every is not None
                            and args.frame_every > 0
                            and candidate_index % args.frame_every == 0
                        ),
                        show=True,
                        nt=nt,
                        ndof=ndof,
                    )
            if pass_best is None:
                break
            root_print(
                comm,
                f"SEARCH_PASS_BEST pass={pass_index} score={pass_best.score:.6e} "
                f"c1={pass_best.c1:.6e} c2={pass_best.c2:.6e} res={pass_best.residual:.6e} "
                f"rhoRel={pass_best.rho_rel:.6e} activeJ={pass_best.active_jaccard:.6e} "
                f"activeRecall={pass_best.active_recall:.6e} activePrecision={pass_best.active_precision:.6e} "
                f"activeDice={pass_best.active_dice:.6e}",
            )
            if args.plot_best_each_pass:
                best_phi.x.array[:] = pass_best.best_coeffs
                best_phi.x.scatter_forward()
                update_interpolated(best_rho, window_ufl(best_phi, pass_best.c1, pass_best.c2, pass_best.eps_phi, params.rho_amp))
                phi_diff.x.array[:] = best_phi.x.array - phi_target.x.array
                phi_diff.x.scatter_forward()
                plotter.emit(
                    [T, rho_design, phi_target, best_phi, best_rho, phi_diff],
                    [
                        "Torsion T",
                        "rhoDesign",
                        "phiT",
                        f"Best phi pass={pass_index}",
                        "rho(best)",
                        "phi-phiT",
                    ],
                    stage="PASS_BEST",
                    ieps=0,
                    k=pass_index,
                    eps_phi=pass_best.eps_phi,
                    residual=pass_best.residual,
                    metrics={
                        "massRho": pass_best.mass_diff + rho_design_mass,
                        "maxRho": 0.0,
                        "activeArea": pass_best.active_area,
                        "plateauArea": pass_best.plateau_area,
                        "relRhoDesign": pass_best.rho_rel,
                    },
                    token=f"pass_{pass_index}_best",
                    save=bool(args.save_frames),
                    show=True,
                    nt=nt,
                    ndof=ndof,
                )
            if pass_index == args.refine_passes:
                break
            radius = radius / max(args.refine_grid - 1, 2)
            candidates = refined_candidates(
                best_c1=pass_best.c1,
                best_c2=pass_best.c2,
                c_min=c_min,
                c_max=c_upper,
                min_width=min_width,
                radius=radius * max(args.refine_grid - 1, 2),
                grid=args.refine_grid,
                width_grid=args.candidate_width_grid if args.candidate_width_grid is not None else args.refine_grid,
                mode=args.candidate_mode,
            )
    finally:
        if candidate_handle is not None:
            candidate_handle.close()

    if all_best is None:
        raise RuntimeError("no candidates were evaluated")

    best_phi.x.array[:] = all_best.best_coeffs
    best_phi.x.scatter_forward()
    update_interpolated(best_rho, window_ufl(best_phi, all_best.c1, all_best.c2, all_best.eps_phi, params.rho_amp))
    phi_diff.x.array[:] = best_phi.x.array - phi_target.x.array
    phi_diff.x.scatter_forward()

    if args.plot_final:
        plotter.emit(
            [T, rho_design, phi_target, best_phi, best_rho, phi_diff],
            ["Torsion T", "rhoDesign", "phiT", "Best phi", "rho(best)", "phi-phiT"],
            stage="FINAL",
            ieps=0,
            k=-1,
            eps_phi=all_best.eps_phi,
            residual=all_best.residual,
            metrics={
                "massRho": all_best.mass_diff + rho_design_mass,
                "maxRho": 0.0,
                "activeArea": all_best.active_area,
                "plateauArea": all_best.plateau_area,
                "relRhoDesign": all_best.rho_rel,
            },
            token="final_best",
            save=bool(args.frame_final),
            show=True,
            nt=nt,
            ndof=ndof,
        )
    if frame_handle is not None:
        frame_handle.close()

    elapsed = time.perf_counter() - total_start
    final_status = "OK" if all_best.converged else "NONCONVERGED"
    root_print(
        comm,
        f"FINAL score={all_best.score:.6e} c1={all_best.c1:.6e} c2={all_best.c2:.6e} "
        f"epsPhi={all_best.eps_phi:.6e} res={all_best.residual:.6e} "
        f"rhoRel={all_best.rho_rel:.6e} massRel={all_best.mass_rel:.6e} "
        f"activeJ={all_best.active_jaccard:.6e} activeRecall={all_best.active_recall:.6e} "
        f"activePrecision={all_best.active_precision:.6e} activeDice={all_best.active_dice:.6e} "
        f"plateauJ={all_best.plateau_jaccard:.6e}",
    )
    root_print(comm, f"FINAL_STATUS {final_status}")
    root_print(comm, f"TIME_TOTAL {elapsed:.3f}")
    root_print(comm, "========== END STRATEGY A DOLFINX WINDOW REDUCED SEARCH ==========")

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
            handle.write(f"c1T {c1_t}\n")
            handle.write(f"c2T {c2_t}\n")
            handle.write(f"epsTRatio {params.eps_t_ratio}\n")
            handle.write(f"epsT {eps_t}\n")
            handle.write(f"epsPhiRatio {args.eps_ratio}\n")
            handle.write(f"scoreMode {args.score_mode}\n")
            handle.write(f"scoreOverlapMetric {args.score_overlap_metric}\n")
            handle.write(f"candidateMode {args.candidate_mode}\n")
            handle.write(f"rhoDesignMass {rho_design_mass}\n")
            handle.write(f"rhoDesignMax {rho_design_max}\n")
            handle.write(f"rhoDesignL2 {rho_design_l2}\n")
            handle.write(f"phiTargetMax {phi_target_max}\n")
            handle.write(f"phiTargetL2 {phi_target_l2}\n")
            handle.write(f"searchCMin {c_min}\n")
            handle.write(f"searchCMax {c_upper}\n")
            handle.write(f"minWidth {min_width}\n")
            handle.write(f"bestScore {all_best.score}\n")
            handle.write(f"bestC1Phi {all_best.c1}\n")
            handle.write(f"bestC2Phi {all_best.c2}\n")
            handle.write(f"bestEpsPhi {all_best.eps_phi}\n")
            handle.write(f"bestResidual {all_best.residual}\n")
            handle.write(f"bestPhiL2 {all_best.phi_l2}\n")
            handle.write(f"bestPhiRel {all_best.phi_rel}\n")
            handle.write(f"bestRhoL2 {all_best.rho_l2}\n")
            handle.write(f"bestRhoRel {all_best.rho_rel}\n")
            handle.write(f"bestMassDiff {all_best.mass_diff}\n")
            handle.write(f"bestMassRel {all_best.mass_rel}\n")
            handle.write(f"bestActiveJaccard {all_best.active_jaccard}\n")
            handle.write(f"bestActiveOverlapArea {all_best.active_overlap_area}\n")
            handle.write(f"bestActiveRecall {all_best.active_recall}\n")
            handle.write(f"bestActivePrecision {all_best.active_precision}\n")
            handle.write(f"bestActiveDice {all_best.active_dice}\n")
            handle.write(f"bestPlateauJaccard {all_best.plateau_jaccard}\n")
            handle.write(f"bestStatus {all_best.status}\n")
            handle.write(f"finalStatus {final_status}\n")
            handle.write(f"timeTotal {elapsed}\n")
    if args.fail_on_nonconvergence and final_status != "OK":
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_strategy(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
