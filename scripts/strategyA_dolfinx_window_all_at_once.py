#!/usr/bin/env python3
"""All-at-once residual-penalty optimizer for Strategy A window thresholds.

This prototype optimizes over the semilinear state ``phi`` and the two
window thresholds simultaneously.  It is deliberately separate from the
fixed-window Newton runners:

    scripts/strategyA_dolfinx_noadapt_torsion_newton.py
    scripts/strategyA_dolfinx_noadapt_torsion_newton_v2.py

The objective is a smooth penalty problem

    0.5*w_phi*||phi-phi_T||^2
  + 0.5*w_rho*||W(phi;c1,c2,eps)-rho_T||^2
  + 0.5*w_mass*(int(W-rho_T))^2
  + 0.5*gamma*||F(phi,c1,c2)||_2^2,

where F is the discrete weak semilinear residual.  This is not the exact
constrained closed-loop problem unless the residual term is driven close to
zero.  The script always reports both target mismatch and residual feasibility.
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
from dolfinx.fem import petsc as fem_petsc

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from strategyA_dolfinx_noadapt_torsion_newton_v2 import (  # noqa: E402
    PyVistaStrategyPlotter,
    allreduce_scalar,
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
    window_derivative_ufl,
    window_ufl,
)


DEFAULT_RUN_LOG_ROOT = REPO_ROOT / "run_logs" / "dolfinx_window_all_at_once"


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
class ObjectiveParts:
    """Scalar objective components for the all-at-once penalty problem."""

    total: float
    data: float
    penalty: float
    active_loss: float
    active_jaccard: float
    active_recall: float
    active_precision: float
    active_dice: float
    phi_l2: float
    rho_l2: float
    mass_diff: float
    residual: float


@dataclass
class GradientInfo:
    """Gradient, descent direction, and diagnostic scalars."""

    gradient_vector: PETSc.Vec
    direction: fem.Function
    grad_m: float
    grad_w: float
    dir_m: float
    dir_w: float
    grad_norm: float
    directional_derivative: float
    solve_time: float
    ksp_iterations: int
    ksp_residual: float


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


def project_center_width(
        m: float,
        w: float,
        *,
        c_max: float,
        min_width: float,
) -> tuple[float, float]:
    """Project center-width coordinates onto the admissible threshold set."""
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
    return float(m) - 0.5 * float(w), float(m) + 0.5 * float(w)


def window_epsilon_derivative_ufl(values, c1: float, c2: float, eps: float, amp: float):
    z1 = values - float(c1)
    z2 = values - float(c2)
    s1 = logistic_ufl(z1, eps)
    s2 = logistic_ufl(z2, eps)
    sp1 = s1 * (1.0 - s1)
    sp2 = s2 * (1.0 - s2)
    eps2 = float(eps) * float(eps)
    return float(amp) * (-(z1 / eps2) * sp1 + (z2 / eps2) * sp2)


def window_c1_derivative_ufl(values, c1: float, eps: float, amp: float):
    s1 = logistic_ufl(values - float(c1), eps)
    return -float(amp) * s1 * (1.0 - s1) / float(eps)


def window_c2_derivative_ufl(values, c2: float, eps: float, amp: float):
    s2 = logistic_ufl(values - float(c2), eps)
    return float(amp) * s2 * (1.0 - s2) / float(eps)


def window_m_derivative_ufl(values, c1: float, c2: float, eps: float, amp: float):
    return (
        window_c1_derivative_ufl(values, c1, eps, amp)
        + window_c2_derivative_ufl(values, c2, eps, amp)
    )


def window_w_derivative_ufl(values, c1: float, c2: float, eps: float, eta_phi: float, amp: float):
    return (
        -0.5 * window_c1_derivative_ufl(values, c1, eps, amp)
        + 0.5 * window_c2_derivative_ufl(values, c2, eps, amp)
        + float(eta_phi) * window_epsilon_derivative_ufl(values, c1, c2, eps, amp)
    )


def assemble_vector_form(linear_form, bc) -> PETSc.Vec:
    vec = fem_petsc.assemble_vector(fem.form(linear_form))
    vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(vec, [bc])
    return vec


def assemble_residual_vector(residual_form, bc) -> PETSc.Vec:
    return assemble_vector_form(residual_form, bc)


def assemble_matrix_form(bilinear_form, bcs: list) -> PETSc.Mat:
    mat = fem_petsc.assemble_matrix(fem.form(bilinear_form), bcs=bcs)
    mat.assemble()
    return mat


def solve_direction_from_vector(
        metric_form,
        rhs: PETSc.Vec,
        target: fem.Function,
        bcs: list,
        *,
        prefix: str,
        solver: str,
        ksp_type: str | None,
        rtol: float,
        atol: float,
        max_it: int | None,
) -> tuple[int, float, float]:
    """Solve the Riesz/preconditioner system into ``target`` from a PETSc RHS."""
    start = time.perf_counter()
    mat = assemble_matrix_form(metric_form, bcs)
    b = rhs.copy()
    fem_petsc.set_bc(b, bcs)
    ksp = PETSc.KSP().create(target.function_space.mesh.comm)
    ksp.setOptionsPrefix(prefix)
    opts = PETSc.Options()
    if solver == "mumps":
        opts[f"{prefix}ksp_type"] = "preonly"
        opts[f"{prefix}pc_type"] = "lu"
        opts[f"{prefix}pc_factor_mat_solver_type"] = "mumps"
    elif solver == "lu":
        opts[f"{prefix}ksp_type"] = "preonly"
        opts[f"{prefix}pc_type"] = "lu"
    elif solver == "hypre":
        opts[f"{prefix}ksp_type"] = ksp_type or "cg"
        opts[f"{prefix}pc_type"] = "hypre"
        opts[f"{prefix}pc_hypre_type"] = "boomeramg"
        opts[f"{prefix}ksp_rtol"] = rtol
        opts[f"{prefix}ksp_atol"] = atol
    elif solver == "gamg":
        opts[f"{prefix}ksp_type"] = ksp_type or "cg"
        opts[f"{prefix}pc_type"] = "gamg"
        opts[f"{prefix}ksp_rtol"] = rtol
        opts[f"{prefix}ksp_atol"] = atol
    else:
        raise ValueError(f"unknown solver {solver!r}")
    if max_it is not None and solver not in {"mumps", "lu"}:
        opts[f"{prefix}ksp_max_it"] = max_it
    ksp.setFromOptions()
    ksp.setOperators(mat)
    ksp.solve(b, target.x.petsc_vec)
    target.x.scatter_forward()
    reason = ksp.getConvergedReason()
    its = int(ksp.getIterationNumber())
    residual = float(ksp.getResidualNorm())
    elapsed = time.perf_counter() - start
    ksp.destroy()
    mat.destroy()
    b.destroy()
    if reason < 0:
        raise RuntimeError(f"direction solve {prefix!r} failed with PETSc reason {reason}")
    return its, residual, elapsed


def scale_function_in_place(function: fem.Function, scale: float) -> None:
    function.x.array[:] *= float(scale)
    function.x.scatter_forward()


def global_absmax_array(comm: MPI.Comm, values: np.ndarray) -> float:
    local = float(np.max(np.abs(values))) if values.size else 0.0
    return float(comm.allreduce(local, op=MPI.MAX))


def smooth_active_mask_ufl(rho_values, *, threshold: float, smooth_eps: float):
    """Return a differentiable active-set indicator for density values."""
    return logistic_ufl(rho_values - float(threshold), float(smooth_eps))


def smooth_active_metrics(
        *,
        comm: MPI.Comm,
        rho_expr,
        rho_design: fem.Function,
        dx,
        threshold: float,
        smooth_eps: float,
) -> dict[str, float | object]:
    """Evaluate soft active-set overlap metrics and keep UFL masks."""
    active = smooth_active_mask_ufl(rho_expr, threshold=threshold, smooth_eps=smooth_eps)
    active_design = smooth_active_mask_ufl(rho_design, threshold=threshold, smooth_eps=smooth_eps)
    overlap = assemble_scalar(comm, active * active_design * dx)
    active_area = assemble_scalar(comm, active * dx)
    design_area = assemble_scalar(comm, active_design * dx)
    union = max(design_area + active_area - overlap, 1.0e-30)
    sum_area = max(design_area + active_area, 1.0e-30)
    return {
        "active": active,
        "activeDesign": active_design,
        "overlap": overlap,
        "activeArea": active_area,
        "designArea": design_area,
        "jaccard": overlap / union,
        "recall": overlap / max(design_area, 1.0e-30),
        "precision": overlap / max(active_area, 1.0e-30),
        "dice": 2.0 * overlap / sum_area,
    }


def smooth_active_loss(
        *,
        metrics: dict[str, float | object],
        metric: str,
        overlap_weight: float,
        miss_weight: float,
        spill_weight: float,
) -> float:
    """Return the scalar soft active-set loss used by the objective."""
    if metric == "recall":
        overlap_metric = float(metrics["recall"])
    elif metric == "dice":
        overlap_metric = float(metrics["dice"])
    else:
        overlap_metric = float(metrics["jaccard"])
    return float(
        float(overlap_weight) * (1.0 - overlap_metric)
        + float(miss_weight) * (1.0 - float(metrics["recall"]))
        + float(spill_weight) * (1.0 - float(metrics["precision"]))
    )


def smooth_active_loss_derivative_wrt_rho(
        *,
        metrics: dict[str, float | object],
        metric: str,
        threshold: float,
        smooth_eps: float,
        overlap_weight: float,
        miss_weight: float,
        spill_weight: float,
):
    """Differentiate the soft active loss with respect to ``rho_expr``."""
    active = metrics["active"]
    active_design = metrics["activeDesign"]
    overlap = float(metrics["overlap"])
    active_area = float(metrics["activeArea"])
    design_area = float(metrics["designArea"])
    union = max(design_area + active_area - overlap, 1.0e-30)
    sum_area = max(design_area + active_area, 1.0e-30)
    dloss_dactive = 0.0
    if float(overlap_weight) != 0.0:
        if metric == "recall":
            dmetric = active_design / max(design_area, 1.0e-30)
        elif metric == "dice":
            dmetric = (2.0 * active_design * sum_area - 2.0 * overlap) / (sum_area * sum_area)
        else:
            dmetric = (active_design * union - overlap * (1.0 - active_design)) / (union * union)
        dloss_dactive = dloss_dactive - float(overlap_weight) * dmetric
    if float(miss_weight) != 0.0:
        drecall = active_design / max(design_area, 1.0e-30)
        dloss_dactive = dloss_dactive - float(miss_weight) * drecall
    if float(spill_weight) != 0.0:
        dprecision = (active_design * active_area - overlap) / max(active_area * active_area, 1.0e-30)
        dloss_dactive = dloss_dactive - float(spill_weight) * dprecision
    dactive_drho = active * (1.0 - active) / float(smooth_eps)
    return dloss_dactive * dactive_drho


def evaluate_objective(
        *,
        u: fem.Function,
        phi_target: fem.Function,
        rho_design: fem.Function,
        v,
        dx,
        bc,
        c1: float,
        c2: float,
        eps_phi: float,
        rho_amp: float,
        phi_weight: float,
        rho_weight: float,
        mass_weight: float,
        active_overlap_metric: str,
        active_overlap_weight: float,
        active_miss_weight: float,
        active_spill_weight: float,
        active_threshold: float,
        active_smooth_eps: float,
        residual_penalty: float,
) -> tuple[ObjectiveParts, PETSc.Vec]:
    comm = u.function_space.mesh.comm
    rho_expr = window_ufl(u, c1, c2, eps_phi, rho_amp)
    residual_form = (ufl.inner(ufl.grad(u), ufl.grad(v)) - rho_expr * v) * dx
    residual_vec = assemble_residual_vector(residual_form, bc)
    residual_norm = float(residual_vec.norm())
    phi_sq = assemble_scalar(comm, (u - phi_target) ** 2 * dx)
    rho_sq = assemble_scalar(comm, (rho_expr - rho_design) ** 2 * dx)
    mass_diff = assemble_scalar(comm, (rho_expr - rho_design) * dx)
    phi_l2 = math.sqrt(max(phi_sq, 0.0))
    rho_l2 = math.sqrt(max(rho_sq, 0.0))
    active_metrics = smooth_active_metrics(
        comm=comm,
        rho_expr=rho_expr,
        rho_design=rho_design,
        dx=dx,
        threshold=active_threshold,
        smooth_eps=active_smooth_eps,
    )
    active_loss = smooth_active_loss(
        metrics=active_metrics,
        metric=active_overlap_metric,
        overlap_weight=active_overlap_weight,
        miss_weight=active_miss_weight,
        spill_weight=active_spill_weight,
    )
    data = (
        0.5 * float(phi_weight) * phi_sq
        + 0.5 * float(rho_weight) * rho_sq
        + 0.5 * float(mass_weight) * mass_diff * mass_diff
        + active_loss
    )
    penalty = 0.5 * float(residual_penalty) * residual_norm * residual_norm
    return ObjectiveParts(
        total=data + penalty,
        data=data,
        penalty=penalty,
        active_loss=active_loss,
        active_jaccard=float(active_metrics["jaccard"]),
        active_recall=float(active_metrics["recall"]),
        active_precision=float(active_metrics["precision"]),
        active_dice=float(active_metrics["dice"]),
        phi_l2=phi_l2,
        rho_l2=rho_l2,
        mass_diff=mass_diff,
        residual=residual_norm,
    ), residual_vec


def compute_gradient_and_direction(
        *,
        u: fem.Function,
        direction: fem.Function,
        phi_target: fem.Function,
        rho_design: fem.Function,
        v,
        z,
        dx,
        bc,
        parts: ObjectiveParts,
        residual_vec: PETSc.Vec,
        m: float,
        w: float,
        c_max: float,
        eps_ratio: float,
        rho_amp: float,
        phi_weight: float,
        rho_weight: float,
        mass_weight: float,
        active_overlap_metric: str,
        active_overlap_weight: float,
        active_miss_weight: float,
        active_spill_weight: float,
        active_threshold: float,
        active_smooth_eps: float,
        residual_penalty: float,
        metric_mass: float,
        max_c_step_fraction: float,
        max_u_step_fraction: float,
        direction_solver: str,
        direction_ksp_type: str | None,
        direction_rtol: float,
        direction_atol: float,
        direction_max_it: int | None,
        iteration: int,
) -> GradientInfo:
    comm = u.function_space.mesh.comm
    c1, c2 = center_width_to_c(m, w)
    eps_phi = float(eps_ratio) * float(w)
    rho_expr = window_ufl(u, c1, c2, eps_phi, rho_amp)
    ws = window_derivative_ufl(u, c1, c2, eps_phi, rho_amp)
    wm = window_m_derivative_ufl(u, c1, c2, eps_phi, rho_amp)
    ww = window_w_derivative_ufl(u, c1, c2, eps_phi, eps_ratio, rho_amp)
    jac_form = (ufl.inner(ufl.grad(z), ufl.grad(v)) - ws * z * v) * dx
    jac = assemble_matrix_form(jac_form, [bc])
    active_metrics = smooth_active_metrics(
        comm=comm,
        rho_expr=rho_expr,
        rho_design=rho_design,
        dx=dx,
        threshold=active_threshold,
        smooth_eps=active_smooth_eps,
    )
    active_dloss_drho = smooth_active_loss_derivative_wrt_rho(
        metrics=active_metrics,
        metric=active_overlap_metric,
        threshold=active_threshold,
        smooth_eps=active_smooth_eps,
        overlap_weight=active_overlap_weight,
        miss_weight=active_miss_weight,
        spill_weight=active_spill_weight,
    )

    data_grad_form = (
        float(phi_weight) * (u - phi_target) * v
        + float(rho_weight) * (rho_expr - rho_design) * ws * v
        + float(mass_weight) * parts.mass_diff * ws * v
        + active_dloss_drho * ws * v
    ) * dx
    grad_vec = assemble_vector_form(data_grad_form, bc)
    penalty_phi = grad_vec.duplicate()
    jac.multTranspose(residual_vec, penalty_phi)
    grad_vec.axpy(float(residual_penalty), penalty_phi)

    rm_vec = assemble_vector_form((-wm * v) * dx, bc)
    rw_vec = assemble_vector_form((-ww * v) * dx, bc)
    grad_m_data = assemble_scalar(
        comm,
        (
            float(rho_weight) * (rho_expr - rho_design) * wm
            + float(mass_weight) * parts.mass_diff * wm
            + active_dloss_drho * wm
        ) * dx,
    )
    grad_w_data = assemble_scalar(
        comm,
        (
            float(rho_weight) * (rho_expr - rho_design) * ww
            + float(mass_weight) * parts.mass_diff * ww
            + active_dloss_drho * ww
        ) * dx,
    )
    grad_m = grad_m_data + float(residual_penalty) * float(residual_vec.dot(rm_vec))
    grad_w = grad_w_data + float(residual_penalty) * float(residual_vec.dot(rw_vec))

    rhs = grad_vec.copy()
    rhs.scale(-1.0)
    metric_form = (ufl.inner(ufl.grad(z), ufl.grad(v)) + float(metric_mass) * z * v) * dx
    its, lin_res, solve_time = solve_direction_from_vector(
        metric_form,
        rhs,
        direction,
        [bc],
        prefix=f"opt_direction_{iteration}_",
        solver=direction_solver,
        ksp_type=direction_ksp_type,
        rtol=direction_rtol,
        atol=direction_atol,
        max_it=direction_max_it,
    )

    max_u_ref = max(global_absmax_array(comm, u.x.array), global_absmax_array(comm, phi_target.x.array), 1.0e-14)
    max_du = global_absmax_array(comm, direction.x.array)
    u_step_limit = max(float(max_u_step_fraction) * max_u_ref, 1.0e-14)
    if max_du > u_step_limit:
        scale_function_in_place(direction, u_step_limit / max_du)

    dir_m = -float(grad_m)
    dir_w = -float(grad_w)
    max_c_step = max(float(max_c_step_fraction) * float(c_max), 1.0e-14)
    raw_c_step = max(abs(dir_m), abs(dir_w))
    if raw_c_step > max_c_step:
        scale = max_c_step / raw_c_step
        dir_m *= scale
        dir_w *= scale

    directional_derivative = float(grad_vec.dot(direction.x.petsc_vec) + grad_m * dir_m + grad_w * dir_w)
    grad_norm = math.sqrt(max(float(grad_vec.norm()) ** 2 + grad_m * grad_m + grad_w * grad_w, 0.0))

    jac.destroy()
    penalty_phi.destroy()
    rm_vec.destroy()
    rw_vec.destroy()
    rhs.destroy()

    return GradientInfo(
        gradient_vector=grad_vec,
        direction=direction,
        grad_m=float(grad_m),
        grad_w=float(grad_w),
        dir_m=float(dir_m),
        dir_w=float(dir_w),
        grad_norm=grad_norm,
        directional_derivative=directional_derivative,
        solve_time=solve_time,
        ksp_iterations=its,
        ksp_residual=lin_res,
    )


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
    parser.add_argument("--eps-ratio", "--eps-phi-ratio", dest="eps_ratio", type=float, default=0.06)
    parser.add_argument("--rho-amp", type=float, default=None)
    parser.add_argument("--c1-phi", dest="c1_phi", type=float, default=None)
    parser.add_argument("--c2-phi", dest="c2_phi", type=float, default=None)
    parser.add_argument("--cmax-factor", type=float, default=1.25)
    parser.add_argument("--min-width-fraction", type=float, default=1.0e-3)
    parser.add_argument("--phi-target-weight", type=float, default=0.0)
    parser.add_argument("--rho-target-weight", type=float, default=0.0)
    parser.add_argument("--mass-target-weight", type=float, default=0.0)
    parser.add_argument("--active-overlap-metric", choices=("jaccard", "recall", "dice"), default="jaccard")
    parser.add_argument("--active-overlap-weight", type=float, default=1.0)
    parser.add_argument("--active-miss-weight", type=float, default=0.0)
    parser.add_argument("--active-spill-weight", type=float, default=0.25)
    parser.add_argument("--active-smooth-ratio", type=float, default=0.02)
    parser.add_argument("--residual-penalty", type=float, default=1.0e4)
    parser.add_argument("--metric-mass", type=float, default=1.0)
    parser.add_argument("--max-opt-it", type=int, default=40)
    parser.add_argument("--tol-grad", type=float, default=1.0e-8)
    parser.add_argument("--tol-step", type=float, default=1.0e-10)
    parser.add_argument("--accept-residual", type=float, default=1.0e-8)
    parser.add_argument("--armijo-c", type=float, default=1.0e-4)
    parser.add_argument("--beta-ls", type=float, default=0.5)
    parser.add_argument("--max-backtrack", type=int, default=20)
    parser.add_argument("--alpha-min", type=float, default=1.0e-8)
    parser.add_argument("--max-c-step-fraction", type=float, default=0.05)
    parser.add_argument("--max-u-step-fraction", type=float, default=0.25)
    parser.add_argument("--fit-window-grid", type=int, default=64)
    parser.add_argument("--fit-window-refine-grid", type=int, default=25)
    parser.add_argument("--fit-window-refine-passes", type=int, default=2)
    parser.add_argument("--fit-window-bins", type=int, default=4096)
    parser.add_argument("--fit-window-quad-degree", type=int, default=None)
    parser.add_argument("--linear-solver", choices=("mumps", "lu", "hypre", "gamg"), default="mumps")
    parser.add_argument("--ksp-type", default=None)
    parser.add_argument("--linear-rtol", type=float, default=1.0e-10)
    parser.add_argument("--linear-atol", type=float, default=1.0e-12)
    parser.add_argument("--linear-max-it", type=int, default=None)
    parser.add_argument("--direction-solver", choices=("mumps", "lu", "hypre", "gamg"), default=None)
    parser.add_argument("--direction-ksp-type", default=None)
    parser.add_argument("--direction-rtol", type=float, default=1.0e-8)
    parser.add_argument("--direction-atol", type=float, default=1.0e-11)
    parser.add_argument("--direction-max-it", type=int, default=None)
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
    parser.add_argument("--plot-optimization", action=argparse.BooleanOptionalAction, default=True)
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
    if args.eps_ratio <= 0.0:
        raise ValueError("require positive --eps-ratio")
    if args.cmax_factor <= 0.0:
        raise ValueError("require positive --cmax-factor")
    if args.min_width_fraction <= 0.0:
        raise ValueError("require positive --min-width-fraction")
    if args.residual_penalty <= 0.0:
        raise ValueError("require positive --residual-penalty")
    if args.active_overlap_weight < 0.0 or args.active_miss_weight < 0.0 or args.active_spill_weight < 0.0:
        raise ValueError("active overlap/miss/spill weights must be nonnegative")
    if args.active_smooth_ratio <= 0.0:
        raise ValueError("require positive --active-smooth-ratio")
    if args.metric_mass < 0.0:
        raise ValueError("require nonnegative --metric-mass")
    if args.max_opt_it < 0:
        raise ValueError("require nonnegative --max-opt-it")
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

    opt_csv = log_dir / "optimization.csv"
    frame_csv = log_dir / "frames.csv"
    summary_path = out_dir / "summary.txt"
    total_start = time.perf_counter()

    root_print(comm, "========== START STRATEGY A DOLFINX WINDOW ALL-AT-ONCE ==========")
    root_print(comm, f"RUN_TAG {run_tag}")
    root_print(comm, f"RUN_DIR {run_dir}")
    root_print(comm, f"OPT_CSV {opt_csv}")
    root_print(comm, f"FRAME_CSV {frame_csv}")
    root_print(comm, f"SUMMARY {summary_path}")
    root_print(
        comm,
        "OBJECTIVE "
        f"phiWeight={args.phi_target_weight:.6e} "
        f"rhoWeight={args.rho_target_weight:.6e} "
        f"massWeight={args.mass_target_weight:.6e} "
        f"activeMetric={args.active_overlap_metric} "
        f"activeWeight={args.active_overlap_weight:.6e} "
        f"activeMissWeight={args.active_miss_weight:.6e} "
        f"activeSpillWeight={args.active_spill_weight:.6e} "
        f"activeSmoothRatio={args.active_smooth_ratio:.6e} "
        f"residualPenalty={args.residual_penalty:.6e}",
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
    z = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    T = fem.Function(V, name="T")
    rho_design = fem.Function(V, name="rhoDesign")
    phi_target = fem.Function(V, name="phiT")
    u = fem.Function(V, name="phi")
    rho = fem.Function(V, name="rho")
    direction = fem.Function(V, name="descent")
    phi_diff = fem.Function(V, name="phiMinusPhiT")

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

    its, rel, solve_time = solve_linear_form(
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
    root_print(comm, f"SOLVER_OK problem=torsion iters={its} rel={rel:.3e} time={solve_time:.3f}")
    root_print(comm, f"TORSION Tmax={tmax:.6e} c1T={c1_t:.6e} c2T={c2_t:.6e} epsT={eps_t:.6e}")

    update_interpolated(rho_design, window_ufl(T, c1_t, c2_t, eps_t, params.rho_amp))
    rho_design_l2 = math.sqrt(max(assemble_scalar(comm, rho_design * rho_design * dx), 0.0))
    rho_design_mass = assemble_scalar(comm, rho_design * dx)
    _, rho_design_max = global_minmax(comm, rho_design)

    its, rel, solve_time = solve_linear_form(
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

    if args.c1_phi is not None:
        c1_phi = float(args.c1_phi)
        c2_phi = float(args.c2_phi)
        fit_result = None
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
    eps_phi = args.eps_ratio * w

    u.x.array[:] = phi_target.x.array
    u.x.scatter_forward()
    update_interpolated(rho, window_ufl(u, c1_phi, c2_phi, eps_phi, params.rho_amp))
    root_print(comm, f"SOLVER_OK problem=phi_target iters={its} rel={rel:.3e} time={solve_time:.3f}")
    root_print(comm, f"PHI_TARGET max={phi_target_max:.6e} rhoDesignMass={rho_design_mass:.6e} rhoDesignMax={rho_design_max:.6e}")
    root_print(
        comm,
        f"WINDOW_INIT c1Phi={c1_phi:.6e} c2Phi={c2_phi:.6e} "
        f"width={w:.6e} epsPhi={eps_phi:.6e} cMax={c_max:.6e}",
    )
    if fit_result is not None:
        root_print(
            comm,
            f"WINDOW_INIT_FIT objectiveL2={fit_result.objective_l2:.6e} "
            f"objectiveRel={fit_result.objective_rel:.6e} time={fit_result.elapsed:.3f}",
        )

    if args.plot_design:
        plotter.emit(
            [T, rho_design, phi_target, rho],
            ["Torsion T", "rhoDesign", "phiT", "rho(phiT; c)"],
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

    fieldnames = [
        "record", "runTag", "k", "nt", "ndof", "c1Phi", "c2Phi", "center", "width", "epsPhi",
        "objective", "dataObjective", "penaltyObjective", "activeLoss",
        "softActiveJaccard", "softActiveRecall", "softActivePrecision", "softActiveDice",
        "phiDiffL2", "rhoDiffL2",
        "massDiff", "residual", "gradNorm", "gradM", "gradW", "dirDeriv", "alpha", "bt",
        "directionSolveTime", "directionIterations", "directionResidual",
        "maxPhi", "maxRho", "massRho", "activeArea", "plateauArea", "activeJaccard",
        "plateauJaccard", "relRhoDesign", "stepTime", "status",
    ]
    opt_handle = opt_csv.open("w", newline="", encoding="utf-8") if comm.rank == 0 else None
    writer = csv.DictWriter(opt_handle, fieldnames=fieldnames) if comm.rank == 0 else None
    if writer is not None:
        writer.writeheader()

    final_parts: ObjectiveParts | None = None
    final_metrics: dict[str, float] | None = None
    final_status = "MAX_OPT_IT"
    frame_every = args.frame_every if args.frame_every is not None else args.plot_every
    direction_solver = args.direction_solver or args.linear_solver
    active_threshold = params.active_threshold * params.rho_amp
    active_smooth_eps = float(args.active_smooth_ratio) * params.rho_amp

    try:
        for k in range(args.max_opt_it + 1):
            step_start = time.perf_counter()
            c1_phi, c2_phi = center_width_to_c(m, w)
            eps_phi = args.eps_ratio * w
            update_interpolated(rho, window_ufl(u, c1_phi, c2_phi, eps_phi, params.rho_amp))
            residual_form = (
                ufl.inner(ufl.grad(u), ufl.grad(v))
                - window_ufl(u, c1_phi, c2_phi, eps_phi, params.rho_amp) * v
            ) * dx
            parts, residual_vec = evaluate_objective(
                u=u,
                phi_target=phi_target,
                rho_design=rho_design,
                v=v,
                dx=dx,
                bc=bc,
                c1=c1_phi,
                c2=c2_phi,
                eps_phi=eps_phi,
                rho_amp=params.rho_amp,
                phi_weight=args.phi_target_weight,
                rho_weight=args.rho_target_weight,
                mass_weight=args.mass_target_weight,
                active_overlap_metric=args.active_overlap_metric,
                active_overlap_weight=args.active_overlap_weight,
                active_miss_weight=args.active_miss_weight,
                active_spill_weight=args.active_spill_weight,
                active_threshold=active_threshold,
                active_smooth_eps=active_smooth_eps,
                residual_penalty=args.residual_penalty,
            )
            metrics = compute_metrics(
                u=u,
                rho=rho,
                rho_design=rho_design,
                rho_design_l2=rho_design_l2,
                residual_form=residual_form,
                bc=bc,
                dx=dx,
                c2_phi=c2_phi,
                active_threshold=params.active_threshold,
                plateau_threshold=params.plateau_threshold,
                rho_amp=params.rho_amp,
            )
            final_parts = parts
            final_metrics = metrics

            if k == args.max_opt_it:
                grad_info = None
                status = "MAX_OPT_IT"
            else:
                grad_info = compute_gradient_and_direction(
                    u=u,
                    direction=direction,
                    phi_target=phi_target,
                    rho_design=rho_design,
                    v=v,
                    z=z,
                    dx=dx,
                    bc=bc,
                    parts=parts,
                    residual_vec=residual_vec,
                    m=m,
                    w=w,
                    c_max=c_max,
                    eps_ratio=args.eps_ratio,
                    rho_amp=params.rho_amp,
                    phi_weight=args.phi_target_weight,
                    rho_weight=args.rho_target_weight,
                    mass_weight=args.mass_target_weight,
                    active_overlap_metric=args.active_overlap_metric,
                    active_overlap_weight=args.active_overlap_weight,
                    active_miss_weight=args.active_miss_weight,
                    active_spill_weight=args.active_spill_weight,
                    active_threshold=active_threshold,
                    active_smooth_eps=active_smooth_eps,
                    residual_penalty=args.residual_penalty,
                    metric_mass=args.metric_mass,
                    max_c_step_fraction=args.max_c_step_fraction,
                    max_u_step_fraction=args.max_u_step_fraction,
                    direction_solver=direction_solver,
                    direction_ksp_type=args.direction_ksp_type,
                    direction_rtol=args.direction_rtol,
                    direction_atol=args.direction_atol,
                    direction_max_it=args.direction_max_it,
                    iteration=k,
                )
                if grad_info.grad_norm < args.tol_grad and parts.residual < args.accept_residual:
                    final_status = "CONVERGED"
                    status = "CONVERGED"
                else:
                    status = "ITERATE"

            if args.terminal_every > 0 and (k % args.terminal_every == 0 or status == "CONVERGED"):
                root_print(
                    comm,
                    f"OPT_STEP k={k} J={parts.total:.6e} data={parts.data:.6e} "
                    f"penalty={parts.penalty:.6e} activeLoss={parts.active_loss:.6e} "
                    f"softActiveJ={parts.active_jaccard:.6e} softRecall={parts.active_recall:.6e} "
                    f"softPrecision={parts.active_precision:.6e} res={parts.residual:.6e} "
                    f"phiL2={parts.phi_l2:.6e} rhoL2={parts.rho_l2:.6e} "
                    f"massDiff={parts.mass_diff:.6e} c1={c1_phi:.6e} c2={c2_phi:.6e} "
                    f"activeJ={metrics['activeJaccard']:.6e} plateauJ={metrics['plateauJaccard']:.6e} "
                    f"status={status}",
                )

            if writer is not None:
                writer.writerow({
                    "record": "OPT",
                    "runTag": run_tag,
                    "k": k,
                    "nt": nt,
                    "ndof": ndof,
                    "c1Phi": c1_phi,
                    "c2Phi": c2_phi,
                    "center": m,
                    "width": w,
                    "epsPhi": eps_phi,
                    "objective": parts.total,
                    "dataObjective": parts.data,
                    "penaltyObjective": parts.penalty,
                    "activeLoss": parts.active_loss,
                    "softActiveJaccard": parts.active_jaccard,
                    "softActiveRecall": parts.active_recall,
                    "softActivePrecision": parts.active_precision,
                    "softActiveDice": parts.active_dice,
                    "phiDiffL2": parts.phi_l2,
                    "rhoDiffL2": parts.rho_l2,
                    "massDiff": parts.mass_diff,
                    "residual": parts.residual,
                    "gradNorm": "" if grad_info is None else grad_info.grad_norm,
                    "gradM": "" if grad_info is None else grad_info.grad_m,
                    "gradW": "" if grad_info is None else grad_info.grad_w,
                    "dirDeriv": "" if grad_info is None else grad_info.directional_derivative,
                    "alpha": "",
                    "bt": "",
                    "directionSolveTime": "" if grad_info is None else grad_info.solve_time,
                    "directionIterations": "" if grad_info is None else grad_info.ksp_iterations,
                    "directionResidual": "" if grad_info is None else grad_info.ksp_residual,
                    "maxPhi": metrics["maxU"],
                    "maxRho": metrics["maxRho"],
                    "massRho": metrics["massRho"],
                    "activeArea": metrics["activeArea"],
                    "plateauArea": metrics["plateauArea"],
                    "activeJaccard": metrics["activeJaccard"],
                    "plateauJaccard": metrics["plateauJaccard"],
                    "relRhoDesign": metrics["relRhoDesign"],
                    "stepTime": time.perf_counter() - step_start,
                    "status": status,
                })
                opt_handle.flush()

            if status == "CONVERGED" or k == args.max_opt_it:
                residual_vec.destroy()
                if grad_info is not None:
                    grad_info.gradient_vector.destroy()
                break

            assert grad_info is not None
            old_u = u.x.array.copy()
            old_m = m
            old_w = w
            alpha = 1.0
            accepted = False
            bt = 0
            trial_parts = parts
            while alpha >= args.alpha_min and bt <= args.max_backtrack:
                u.x.array[:] = old_u + alpha * direction.x.array
                u.x.scatter_forward()
                trial_m, trial_w = project_center_width(
                    old_m + alpha * grad_info.dir_m,
                    old_w + alpha * grad_info.dir_w,
                    c_max=c_max,
                    min_width=min_width,
                )
                trial_c1, trial_c2 = center_width_to_c(trial_m, trial_w)
                trial_eps = args.eps_ratio * trial_w
                trial_parts, trial_residual_vec = evaluate_objective(
                    u=u,
                    phi_target=phi_target,
                    rho_design=rho_design,
                    v=v,
                    dx=dx,
                    bc=bc,
                    c1=trial_c1,
                    c2=trial_c2,
                    eps_phi=trial_eps,
                    rho_amp=params.rho_amp,
                    phi_weight=args.phi_target_weight,
                    rho_weight=args.rho_target_weight,
                    mass_weight=args.mass_target_weight,
                    active_overlap_metric=args.active_overlap_metric,
                    active_overlap_weight=args.active_overlap_weight,
                    active_miss_weight=args.active_miss_weight,
                    active_spill_weight=args.active_spill_weight,
                    active_threshold=active_threshold,
                    active_smooth_eps=active_smooth_eps,
                    residual_penalty=args.residual_penalty,
                )
                trial_residual_vec.destroy()
                armijo_rhs = parts.total + args.armijo_c * alpha * min(grad_info.directional_derivative, -1.0e-30)
                if math.isfinite(trial_parts.total) and trial_parts.total <= armijo_rhs:
                    accepted = True
                    m = trial_m
                    w = trial_w
                    break
                if args.verbosity >= 2:
                    root_print(
                        comm,
                        f"OPT_LS k={k} alpha={alpha:.6e} Jtrial={trial_parts.total:.6e} "
                        f"Jold={parts.total:.6e} armijo={armijo_rhs:.6e} resTrial={trial_parts.residual:.6e}",
                    )
                alpha *= args.beta_ls
                bt += 1

            if not accepted:
                u.x.array[:] = old_u
                u.x.scatter_forward()
                m = old_m
                w = old_w
                final_status = "FAIL_LS"
                root_print(comm, f"OPT_STOP reason=FAIL_LS k={k} alpha={alpha:.3e} bt={bt}")
                residual_vec.destroy()
                grad_info.gradient_vector.destroy()
                break

            if writer is not None:
                writer.writerow({
                    "record": "STEP",
                    "runTag": run_tag,
                    "k": k,
                    "nt": nt,
                    "ndof": ndof,
                    "c1Phi": center_width_to_c(m, w)[0],
                    "c2Phi": center_width_to_c(m, w)[1],
                    "center": m,
                    "width": w,
                    "epsPhi": args.eps_ratio * w,
                    "objective": trial_parts.total,
                    "dataObjective": trial_parts.data,
                    "penaltyObjective": trial_parts.penalty,
                    "activeLoss": trial_parts.active_loss,
                    "softActiveJaccard": trial_parts.active_jaccard,
                    "softActiveRecall": trial_parts.active_recall,
                    "softActivePrecision": trial_parts.active_precision,
                    "softActiveDice": trial_parts.active_dice,
                    "phiDiffL2": trial_parts.phi_l2,
                    "rhoDiffL2": trial_parts.rho_l2,
                    "massDiff": trial_parts.mass_diff,
                    "residual": trial_parts.residual,
                    "gradNorm": grad_info.grad_norm,
                    "gradM": grad_info.grad_m,
                    "gradW": grad_info.grad_w,
                    "dirDeriv": grad_info.directional_derivative,
                    "alpha": alpha,
                    "bt": bt,
                    "directionSolveTime": grad_info.solve_time,
                    "directionIterations": grad_info.ksp_iterations,
                    "directionResidual": grad_info.ksp_residual,
                    "maxPhi": "",
                    "maxRho": "",
                    "massRho": "",
                    "activeArea": "",
                    "plateauArea": "",
                    "activeJaccard": "",
                    "plateauJaccard": "",
                    "relRhoDesign": "",
                    "stepTime": time.perf_counter() - step_start,
                    "status": "ACCEPT",
                })
                opt_handle.flush()

            step_size = max(
                global_absmax_array(comm, alpha * direction.x.array),
                abs(alpha * grad_info.dir_m),
                abs(alpha * grad_info.dir_w),
            )
            if args.terminal_every > 0 and k % args.terminal_every == 0:
                root_print(
                    comm,
                    f"OPT_ACCEPT k={k} alpha={alpha:.3e} bt={bt} Jnew={trial_parts.total:.6e} "
                    f"resNew={trial_parts.residual:.6e} step={step_size:.6e}",
                )
            if args.plot_optimization and args.plot_every > 0 and k % args.plot_every == 0:
                c1_plot, c2_plot = center_width_to_c(m, w)
                eps_plot = args.eps_ratio * w
                update_interpolated(rho, window_ufl(u, c1_plot, c2_plot, eps_plot, params.rho_amp))
                plot_residual_form = (
                    ufl.inner(ufl.grad(u), ufl.grad(v))
                    - window_ufl(u, c1_plot, c2_plot, eps_plot, params.rho_amp) * v
                ) * dx
                plot_metrics = compute_metrics(
                    u=u,
                    rho=rho,
                    rho_design=rho_design,
                    rho_design_l2=rho_design_l2,
                    residual_form=plot_residual_form,
                    bc=bc,
                    dx=dx,
                    c2_phi=c2_plot,
                    active_threshold=params.active_threshold,
                    plateau_threshold=params.plateau_threshold,
                    rho_amp=params.rho_amp,
                )
                phi_diff.x.array[:] = u.x.array - phi_target.x.array
                phi_diff.x.scatter_forward()
                plotter.emit(
                    [T, rho_design, phi_target, u, rho, phi_diff],
                    [
                        "Torsion T",
                        "rhoDesign",
                        "phiT",
                        f"phi opt k={k}",
                        "rho(phi;c)",
                        "phi-phiT",
                    ],
                    stage="OPT",
                    ieps=0,
                    k=k,
                    eps_phi=eps_plot,
                    residual=trial_parts.residual,
                    metrics=plot_metrics,
                    token=f"opt_k_{k}",
                    save=bool(args.save_frames and frame_every is not None and frame_every > 0 and k % frame_every == 0),
                    show=True,
                    nt=nt,
                    ndof=ndof,
                )
            residual_vec.destroy()
            grad_info.gradient_vector.destroy()

            if step_size < args.tol_step:
                final_status = "CONVERGED_STEP"
                root_print(comm, f"OPT_STOP reason=STEP k={k} step={step_size:.6e}")
                break
    finally:
        if opt_handle is not None:
            opt_handle.close()

    c1_phi, c2_phi = center_width_to_c(m, w)
    eps_phi = args.eps_ratio * w
    update_interpolated(rho, window_ufl(u, c1_phi, c2_phi, eps_phi, params.rho_amp))
    residual_form = (
        ufl.inner(ufl.grad(u), ufl.grad(v))
        - window_ufl(u, c1_phi, c2_phi, eps_phi, params.rho_amp) * v
    ) * dx
    final_parts, final_residual_vec = evaluate_objective(
        u=u,
        phi_target=phi_target,
        rho_design=rho_design,
        v=v,
        dx=dx,
        bc=bc,
        c1=c1_phi,
        c2=c2_phi,
        eps_phi=eps_phi,
        rho_amp=params.rho_amp,
        phi_weight=args.phi_target_weight,
        rho_weight=args.rho_target_weight,
        mass_weight=args.mass_target_weight,
        active_overlap_metric=args.active_overlap_metric,
        active_overlap_weight=args.active_overlap_weight,
        active_miss_weight=args.active_miss_weight,
        active_spill_weight=args.active_spill_weight,
        active_threshold=active_threshold,
        active_smooth_eps=active_smooth_eps,
        residual_penalty=args.residual_penalty,
    )
    final_residual_vec.destroy()
    final_metrics = compute_metrics(
        u=u,
        rho=rho,
        rho_design=rho_design,
        rho_design_l2=rho_design_l2,
        residual_form=residual_form,
        bc=bc,
        dx=dx,
        c2_phi=c2_phi,
        active_threshold=params.active_threshold,
        plateau_threshold=params.plateau_threshold,
        rho_amp=params.rho_amp,
    )
    if final_status == "MAX_OPT_IT" and final_parts.residual < args.accept_residual:
        final_status = "OK_RESIDUAL"
    elif final_status in {"CONVERGED", "CONVERGED_STEP"} and final_parts.residual < args.accept_residual:
        final_status = "OK"
    elif final_status in {"CONVERGED", "CONVERGED_STEP"}:
        final_status = f"{final_status}_RESIDUAL_HIGH"
    elif final_status == "MAX_OPT_IT":
        final_status = "NONCONVERGED"

    if args.plot_final:
        phi_diff.x.array[:] = u.x.array - phi_target.x.array
        phi_diff.x.scatter_forward()
        plotter.emit(
            [T, rho_design, phi_target, u, rho, phi_diff],
            ["Torsion T", "rhoDesign", "phiT", "Optimized phi", "rho(phi;c)", "phi-phiT"],
            stage="FINAL",
            ieps=0,
            k=-1,
            eps_phi=eps_phi,
            residual=final_parts.residual,
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
        f"FINAL J={final_parts.total:.6e} data={final_parts.data:.6e} penalty={final_parts.penalty:.6e} "
        f"activeLoss={final_parts.active_loss:.6e} softActiveJ={final_parts.active_jaccard:.6e} "
        f"softRecall={final_parts.active_recall:.6e} softPrecision={final_parts.active_precision:.6e} "
        f"res={final_parts.residual:.6e} phiL2={final_parts.phi_l2:.6e} "
        f"rhoL2={final_parts.rho_l2:.6e} massDiff={final_parts.mass_diff:.6e} "
        f"c1={c1_phi:.6e} c2={c2_phi:.6e} maxPhi={final_metrics['maxU']:.6e} "
        f"maxRho={final_metrics['maxRho']:.6e} activeJ={final_metrics['activeJaccard']:.6e}",
    )
    root_print(comm, f"FINAL_STATUS {final_status}")
    root_print(comm, f"TIME_TOTAL {elapsed:.3f}")
    root_print(comm, "========== END STRATEGY A DOLFINX WINDOW ALL-AT-ONCE ==========")

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
            handle.write(f"directionSolver {direction_solver}\n")
            handle.write(f"alphaT1 {params.alpha_t1}\n")
            handle.write(f"alphaT2 {params.alpha_t2}\n")
            handle.write(f"c1T {c1_t}\n")
            handle.write(f"c2T {c2_t}\n")
            handle.write(f"epsTRatio {params.eps_t_ratio}\n")
            handle.write(f"epsT {eps_t}\n")
            handle.write(f"epsPhiRatio {args.eps_ratio}\n")
            handle.write(f"epsPhi {eps_phi}\n")
            handle.write(f"rhoDesignMass {rho_design_mass}\n")
            handle.write(f"rhoDesignMax {rho_design_max}\n")
            handle.write(f"rhoDesignL2 {rho_design_l2}\n")
            handle.write(f"phiTargetMax {phi_target_max}\n")
            handle.write(f"cMax {c_max}\n")
            handle.write(f"minWidth {min_width}\n")
            handle.write(f"c1Phi {c1_phi}\n")
            handle.write(f"c2Phi {c2_phi}\n")
            handle.write(f"center {m}\n")
            handle.write(f"width {w}\n")
            handle.write(f"objective {final_parts.total}\n")
            handle.write(f"dataObjective {final_parts.data}\n")
            handle.write(f"penaltyObjective {final_parts.penalty}\n")
            handle.write(f"activeLoss {final_parts.active_loss}\n")
            handle.write(f"softActiveJaccard {final_parts.active_jaccard}\n")
            handle.write(f"softActiveRecall {final_parts.active_recall}\n")
            handle.write(f"softActivePrecision {final_parts.active_precision}\n")
            handle.write(f"softActiveDice {final_parts.active_dice}\n")
            handle.write(f"phiDiffL2 {final_parts.phi_l2}\n")
            handle.write(f"rhoDiffL2 {final_parts.rho_l2}\n")
            handle.write(f"massDiff {final_parts.mass_diff}\n")
            handle.write(f"residual {final_parts.residual}\n")
            handle.write(f"phiTargetWeight {args.phi_target_weight}\n")
            handle.write(f"rhoTargetWeight {args.rho_target_weight}\n")
            handle.write(f"massTargetWeight {args.mass_target_weight}\n")
            handle.write(f"activeOverlapMetric {args.active_overlap_metric}\n")
            handle.write(f"activeOverlapWeight {args.active_overlap_weight}\n")
            handle.write(f"activeMissWeight {args.active_miss_weight}\n")
            handle.write(f"activeSpillWeight {args.active_spill_weight}\n")
            handle.write(f"activeSmoothRatio {args.active_smooth_ratio}\n")
            handle.write(f"residualPenalty {args.residual_penalty}\n")
            handle.write(f"maxRho {final_metrics['maxRho']}\n")
            handle.write(f"massRho {final_metrics['massRho']}\n")
            handle.write(f"activeArea {final_metrics['activeArea']}\n")
            handle.write(f"plateauArea {final_metrics['plateauArea']}\n")
            handle.write(f"activeJaccard {final_metrics['activeJaccard']}\n")
            handle.write(f"plateauJaccard {final_metrics['plateauJaccard']}\n")
            handle.write(f"relRhoDesign {final_metrics['relRhoDesign']}\n")
            handle.write(f"finalStatus {final_status}\n")
            handle.write(f"timeTotal {elapsed}\n")
    if args.fail_on_nonconvergence and final_status not in {"OK", "OK_RESIDUAL"}:
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_strategy(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
