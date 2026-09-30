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

The reduced objective is the target-free soft Jaccard loss

    1 - J_soft = (L + M) / (A_tau + L),

where ``A_tau`` is the fixed torsion-band area.  The optimizer never assumes
that either sensitivity functional has an attainable zero minimum and never
classifies an iterate by a requested geometric objective tolerance.  It stops
only for an algorithmic reason (for example Jaccard stagnation, a small
metric-dual gradient, a small step, or the iteration limit) and reports Newton
equilibrium convergence separately.
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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc

REPO_ROOT = Path(__file__).resolve().parents[5]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from projects.diocotron.dolfinx.checkpoint import write_equilibrium_checkpoint_v2  # noqa: E402
from projects.diocotron.dolfinx.torsion.equilibrium.window_fit import (  # noqa: E402
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
from projects.diocotron.dolfinx.plotting.mpi_pyvista import MPIPyVistaTorsionPlotter  # noqa: E402
from projects.diocotron.dolfinx.torsion.equilibrium.newton_budget import (  # noqa: E402
    NewtonBudgetDecision,
    decide_newton_budget_extension,
    forecast_newton_stall,
    newton_hard_ceiling,
)
from projects.diocotron.dolfinx.torsion.initialization.energy_primer import (  # noqa: E402
    LOGISTIC_CLIP,
    classify_picard_spectrum,
    residual_progress_is_material,
)
from projects.diocotron.dolfinx.torsion.initialization.source_homotopy import (  # noqa: E402
    AdaptiveSourceHomotopy,
    SourceHomotopySchedule,
)
from projects.diocotron.dolfinx.runtime.terminal_log_capture import (  # noqa: E402
    TerminalLogCapture,
    get_bootstrap_terminal_log_capture,
)
from projects.diocotron.dolfinx.torsion.initialization.frozen_frontier import (  # noqa: E402
    FrozenFrontier,
    FrozenFrontierPoint,
    LogisticWindowMetrics,
    PushforwardHistogram,
    hard_window_pareto_frontier,
    mass_roundoff_tolerance,
    sampled_logistic_window_metrics,
    select_frozen_frontier_point,
    weighted_pushforward_histogram,
)


DEFAULT_RUN_LOG_ROOT = REPO_ROOT / "projects/diocotron/runs" / "dolfinx_torsion_initialized_window_reduced_optimization"
SENSITIVITY_CHECK_FIELDS = (
    "component",
    "stepFraction",
    "stepScale",
    "step",
    "baseC1",
    "baseC2",
    "baseEps",
    "baseStatus",
    "baseResidual",
    "minusC1",
    "minusC2",
    "plusC1",
    "plusC2",
    "minusStatus",
    "plusStatus",
    "minusIterations",
    "plusIterations",
    "minusResidual",
    "plusResidual",
    "analyticLeakageDerivative",
    "finiteDifferenceLeakageDerivative",
    "leakageDerivativeAbsoluteError",
    "leakageDerivativeRelativeError",
    "analyticMissingDerivative",
    "finiteDifferenceMissingDerivative",
    "missingDerivativeAbsoluteError",
    "missingDerivativeRelativeError",
    "stateDerivativeL2Norm",
    "sensitivityL2Norm",
    "stateSensitivityL2AbsoluteError",
    "stateSensitivityL2RelativeError",
    "stateDerivativeH1Seminorm",
    "sensitivityH1Seminorm",
    "stateSensitivityH1AbsoluteError",
    "stateSensitivityH1RelativeError",
    "valid",
)
SEARCH_SPACE_FIELDS = (
    "record",
    "runTag",
    "k",
    "nt",
    "ndof",
    "status",
    "accepted",
    "cMin",
    "cMax",
    "minWidth",
    "simplexScale",
    "trustRadius",
    "dikinMetric11",
    "dikinMetric12",
    "dikinMetric22",
    "relativePullbackMetric11",
    "relativePullbackMetric12",
    "relativePullbackMetric22",
    "trustMetric11",
    "trustMetric12",
    "trustMetric22",
    "trustMetricEigMin",
    "trustMetricEigMax",
    "trustMetricCondition",
    "pullbackAvailable",
    "baseC1",
    "baseC2",
    "baseCenter",
    "baseWidth",
    "baseGapLeft",
    "baseGapWidth",
    "baseGapRight",
    "baseSimplexLeft",
    "baseSimplexWidth",
    "baseSimplexRight",
    "baseSimplexMin",
    "baseLogLeftToRight",
    "baseLogWidthToRight",
    "dc1",
    "dc2",
    "deltaCenter",
    "deltaWidth",
    "stepNorm",
    "stepMetricNorm",
    "stepDikinNorm",
    "stepPullbackNorm",
    "trialC1",
    "trialC2",
    "trialCenter",
    "trialWidth",
    "trialGapLeft",
    "trialGapWidth",
    "trialGapRight",
    "trialSimplexLeft",
    "trialSimplexWidth",
    "trialSimplexRight",
    "trialSimplexMin",
    "trialLogLeftToRight",
    "trialLogWidthToRight",
    "baseStateH1",
    "sensitivityMetric11",
    "sensitivityMetric12",
    "sensitivityMetric22",
    "sensitivityMetricEigMin",
    "sensitivityMetricEigMax",
    "sensitivityMetricCondition",
    "sensitivityMetricScaledEigMin",
    "sensitivityMetricScaledEigMax",
    "predictedStateH1",
    "predictedStateH1Relative",
    "predictorDefectHminus1",
    "predictorDefectRelative",
    "predictorDefectToPredictedStep",
    "predictorMetricSolveTime",
    "initialNewtonStatus",
    "initialNewtonIterations",
    "initialNewtonFirstDirectionH1",
    "initialNewtonFirstUpdateH1",
    "initialNewtonFirstAlpha",
    "initialNewtonFirstBacktracks",
    "initialCorrectionToPredictedStep",
    "retryNewtonStatus",
    "retryNewtonIterations",
    "retryNewtonFirstDirectionH1",
    "retryNewtonFirstUpdateH1",
    "trialResidual",
    "trialBranchOverlap",
    "trialActivityArea",
    "trialLeakageRel",
    "trialMissingRel",
    "predictedReduction",
    "actualReduction",
    "rhoRatio",
    "baseActiveJaccard",
    "resultActiveJaccard",
)


@dataclass
class TorsionParameters:
    """Fixed parameters defining the torsion-designed target band.

    These values are not optimized by this reduced-space runner.  They define
    the reference torsion solve, the torsion threshold band, and the smoothed
    density used to build ``rho_design`` and ``phi_target``.  The outer
    optimization changes only the semilinear potential thresholds
    ``(c1_phi, c2_phi)``.

    Attributes:
        alpha_t1: Lower torsion threshold as a fraction of ``max(T)``.
        alpha_t2: Upper torsion threshold as a fraction of ``max(T)``.
        eps_t_ratio: Logistic smoothing width for the torsion window, scaled
            by ``c2_t - c1_t``.
        rho_amp: Density amplitude multiplying the logistic activity window in
            the PDE right-hand side.
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
        first_direction_h1: H0^1 seminorm of the first undamped Newton
            direction, or ``nan`` when no Newton solve was needed.
        first_update_h1: H0^1 seminorm of the first accepted damped update.
        first_alpha: Damping factor accepted for the first Newton update.
        first_backtracks: Backtracking reductions before that first update.
    """

    status: str
    converged: bool
    iterations: int
    residual: float
    step_h1: float
    alpha: float
    backtracks: int
    solve_time: float
    contraction: float = math.nan
    predicted_remaining: float = math.inf
    stall_windows: int = 0
    forecast_remaining_budget: int = 0
    first_direction_h1: float = math.nan
    first_update_h1: float = math.nan
    first_alpha: float = math.nan
    first_backtracks: int = 0


@dataclass
class EnergyPrimerRecord:
    """One accepted descent step or terminal energy-primer record."""

    record: str
    context: str
    prefix: str
    outer_iteration: int
    homotopy_lambda: float
    primer_iteration: int
    energy_before: float
    energy_after: float
    gradient_norm: float
    alpha: float
    backtracks: int
    status: str
    elapsed: float
    gradient_norm_after: float = math.nan
    residual_ratio: float = math.nan
    branch_overlap: float = math.nan
    activity_area: float = math.nan
    plot_time: float = 0.0


@dataclass
class EnergyPrimerResult:
    """Outcome of the optional H0^1 Sobolev-gradient globalization phase."""

    status: str
    accepted_steps: int
    initial_energy: float
    final_energy: float
    gradient_norm: float
    alpha: float
    backtracks: int
    solve_time: float
    elapsed: float
    restored_initial_state: bool = False
    initial_gradient_norm: float = math.nan
    residual_ratio: float = math.nan
    branch_overlap: float = math.nan
    activity_area: float = math.nan
    plot_time: float = 0.0


@dataclass(frozen=True)
class EnergyPrimerGuardResult:
    """Branch and activity acceptance decision for one candidate primer step."""

    accepted: bool
    branch_overlap: float
    activity_area: float
    reason: str


@dataclass
class EnergyPrimerStatistics:
    """Run-level aggregate included in the summary file."""

    calls: int = 0
    accepted_steps: int = 0
    solve_time: float = 0.0
    plot_time: float = 0.0
    elapsed: float = 0.0
    restored_calls: int = 0

    def record(self, result: EnergyPrimerResult) -> None:
        self.calls += 1
        self.accepted_steps += int(result.accepted_steps)
        self.solve_time += float(result.solve_time)
        self.plot_time += float(result.plot_time)
        self.elapsed += float(result.elapsed)
        self.restored_calls += int(result.restored_initial_state)


@dataclass
class EquilibriumCorrectionResult:
    """Primary Newton attempt plus an optional primer-assisted retry."""

    newton: NewtonResult
    initial_newton: NewtonResult
    retry_newton: NewtonResult | None
    primer: EnergyPrimerResult | None
    rescue_triggered: bool

    @property
    def total_newton_iterations(self) -> int:
        retry_iterations = (
            0 if self.retry_newton is None else int(self.retry_newton.iterations)
        )
        return int(self.initial_newton.iterations) + retry_iterations

    @property
    def total_newton_solve_time(self) -> float:
        retry_time = (
            0.0 if self.retry_newton is None else float(self.retry_newton.solve_time)
        )
        return float(self.initial_newton.solve_time) + retry_time


@dataclass
class PicardSpectrumRecord:
    """Stiffness-relative extremal spectrum of the local Picard derivative."""

    stage: str
    outer_iteration: int
    c1: float
    c2: float
    eps_phi: float
    residual: float
    mu_min: float
    mu_max: float
    error_min: float
    error_max: float
    iterations_min: int
    iterations_max: int
    converged_min: int
    converged_max: int
    spectral_radius_bound: float
    energy_minimum: int
    picard_contracting: int
    status: str
    elapsed: float


@dataclass
class HomotopyResult:
    """Outcome of fixed-threshold source-homotopy initialization."""

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
        sensitivity_metric: H0^1 Gram matrix ``S.T K S`` of the two state
            sensitivities, used by the intrinsic threshold trust metric.
        sensitivity_metric_eigenvalues: Ascending eigenvalues of that Gram
            matrix.
        sensitivity_metric_condition: Ratio of its largest to smallest
            eigenvalue, or infinity for a singular direction.
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
    sensitivity_metric: np.ndarray
    sensitivity_metric_eigenvalues: tuple[float, float]
    sensitivity_metric_condition: float


@dataclass
class ParameterStep:
    """Trust-region step in the two threshold variables.

    The outer update solves a small convex model problem in ``(dc1, dc2)``
    using the sum of the simplex-Dikin metric and the relative equilibrium
    pullback metric.  Thus the trust radius is dimensionless even though
    ``dc`` retains physical threshold units.

    Attributes:
        dc: Proposed threshold increment ``[dc1, dc2]``.
        objective_name: Name of the reduced scalar objective.
        predicted_reduction: Positive model-predicted decrease in the active
            objective.
        model_change: Raw quadratic model value at ``dc``.
        hessian_scale: Scalar positive Hessian approximation used in the
            two-dimensional model.
        step_norm: Euclidean norm of ``dc``.
        metric_norm: Norm in the combined trust metric.
        dikin_norm: Component of ``metric_norm`` caused by relative changes
            in the three threshold-simplex slacks.
        pullback_norm: Component caused by the predicted relative H0^1 state
            change.
        hit_boundary: True when the trust-region radius is active.
        status: Solver status for the tiny QP enumerator.
    """

    dc: np.ndarray
    objective_name: str
    predicted_reduction: float
    model_change: float
    hessian_scale: float
    step_norm: float
    metric_norm: float
    dikin_norm: float
    pullback_norm: float
    hit_boundary: bool
    status: str


@dataclass(frozen=True)
class ThresholdSimplexPoint:
    """One threshold pair represented by the three admissible-set slacks.

    The feasible threshold triangle is the simplex of the nonnegative left
    margin, excess window width, and right margin.  Fractions and log-ratios
    are dimensionless, so records can be compared across potential scales and
    geometries without choosing a preferred threshold direction.
    """

    c1: float
    c2: float
    center: float
    width: float
    left_gap: float
    width_gap: float
    right_gap: float
    simplex_scale: float
    left_fraction: float
    width_fraction: float
    right_fraction: float
    minimum_fraction: float
    log_left_to_right: float
    log_width_to_right: float


@dataclass(frozen=True)
class ThresholdTrustMetric:
    """Intrinsic local metric used by the threshold trust region.

    ``dikin`` is the Hessian of the logarithmic barrier for the three
    threshold-simplex slacks.  ``pullback`` is the H0^1 sensitivity Gram
    matrix divided by the squared H0^1 norm of the current equilibrium.
    Their sum needs no cross-geometry weight because both terms are already
    dimensionless squared relative changes.
    """

    dikin: np.ndarray
    pullback: np.ndarray
    combined: np.ndarray
    eigenvalues: tuple[float, float]
    condition: float
    pullback_available: bool


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
    """

    base: InitialWindowCandidate
    newton: NewtonResult
    metrics: BandMetrics
    diagnostics: dict[str, float]
    score: float
    state: np.ndarray
    density: np.ndarray
    homotopy: HomotopyResult | None = None


@dataclass(frozen=True)
class FrozenFrontierInitialization:
    """Auditable result of hard-frontier selection and smooth refinement."""

    hard_point: FrozenFrontierPoint
    smooth_metrics: LogisticWindowMetrics
    selection_mode: str
    leakage_cap_rel: float | None
    leakage_cap: float | None
    bins: int
    histogram_lower: float
    histogram_upper: float
    frontier_points: int
    evaluated_intervals: int
    refinement_iterations: int
    refinement_message: str
    target_area_sampled: float
    target_area_assembled: float


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


def params_from_args(args: argparse.Namespace) -> TorsionParameters:
    """Build validated torsion-target parameters from command-line overrides.

    The dataclass carries the defaults used in the torsion-initialized Newton
    parameter studies.  This function applies optional CLI overrides and
    validates the mathematical preconditions for the torsion band: the upper
    fractional threshold must exceed the lower one, and the torsion smoothing
    ratio must be nonnegative.  A zero ratio requests the sharp torsion-band
    indicator instead of the logistic target-density smoothing.

    Args:
        args: Parsed command-line namespace.

    Returns:
        ``TorsionParameters`` with user overrides applied.

    Raises:
        ValueError: If the torsion thresholds are not ordered or the torsion
            smoothing ratio is negative.
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
    if params.eps_t_ratio < 0.0:
        raise ValueError("require nonnegative --eps-t-ratio")
    return params

def load_global_initial_state(
        path: Path,
        u: fem.Function,
        comm: MPI.Comm,
) -> int:
    """Load a scalar FE state stored by global degree-of-freedom index."""

    if comm.rank == 0:
        with np.load(path, allow_pickle=False) as archive:
            if "phi" not in archive:
                raise ValueError(f"initial-state archive lacks phi array: {path}")
            values = np.asarray(archive["phi"], dtype=np.float64)
        expected = int(u.function_space.dofmap.index_map.size_global)
        if values.ndim != 1 or values.size != expected:
            raise ValueError(
                f"initial-state archive has {values.size} values, expected {expected}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("initial-state archive contains non-finite values")
    else:
        values = None
    values = comm.bcast(values, root=0)
    index_map = u.function_space.dofmap.index_map
    local_dofs = np.arange(
        index_map.size_local + index_map.num_ghosts,
        dtype=np.int32,
    )
    global_dofs = np.asarray(index_map.local_to_global(local_dofs), dtype=np.int64)
    u.x.array[:] = values[global_dofs]
    u.x.scatter_forward()
    return int(values.size)


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
        ufl.gt(zz, LOGISTIC_CLIP),
        1.0,
        ufl.conditional(
            ufl.lt(zz, -LOGISTIC_CLIP),
            0.0,
            1.0 / (1.0 + ufl.exp(-zz)),
        ),
    )


def clipped_softplus_ufl(z):
    """Return a continuous primitive of the exactly clipped logistic."""
    low_value = math.log1p(math.exp(-LOGISTIC_CLIP))
    high_value = math.log1p(math.exp(LOGISTIC_CLIP))
    return ufl.conditional(
        ufl.gt(z, LOGISTIC_CLIP),
        high_value + (z - LOGISTIC_CLIP),
        ufl.conditional(
            ufl.lt(z, -LOGISTIC_CLIP),
            low_value,
            ufl.ln(1.0 + ufl.exp(z)),
        ),
    )


def window_primitive_const_ufl(values, c1, c2, eps):
    """Return ``P`` with ``dP/dvalues`` equal to the clipped activity."""
    return eps * (
        clipped_softplus_ufl((values - c1) / eps)
        - clipped_softplus_ufl((values - c2) / eps)
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


def threshold_simplex_point(
        c1: float,
        c2: float,
        *,
        c_min: float,
        c_max: float,
        min_width: float,
) -> ThresholdSimplexPoint:
    """Return intrinsic simplex coordinates for one threshold pair.

    The three gaps are retained without clipping.  Consequently, a negative
    fraction records a genuinely infeasible point instead of hiding it.  Log
    ratios are defined only in the simplex interior and are ``nan`` on a face.
    """
    c1_value = float(c1)
    c2_value = float(c2)
    left_gap = c1_value - float(c_min)
    width = c2_value - c1_value
    width_gap = width - float(min_width)
    right_gap = float(c_max) - c2_value
    simplex_scale = float(c_max) - float(c_min) - float(min_width)
    if simplex_scale > 0.0:
        left_fraction = left_gap / simplex_scale
        width_fraction = width_gap / simplex_scale
        right_fraction = right_gap / simplex_scale
    else:
        left_fraction = math.nan
        width_fraction = math.nan
        right_fraction = math.nan

    def interior_log_ratio(numerator: float, denominator: float) -> float:
        if numerator <= 0.0 or denominator <= 0.0:
            return math.nan
        return math.log(numerator / denominator)

    fractions = (left_fraction, width_fraction, right_fraction)
    minimum_fraction = (
        min(fractions) if all(math.isfinite(value) for value in fractions) else math.nan
    )
    return ThresholdSimplexPoint(
        c1=c1_value,
        c2=c2_value,
        center=0.5 * (c1_value + c2_value),
        width=width,
        left_gap=left_gap,
        width_gap=width_gap,
        right_gap=right_gap,
        simplex_scale=simplex_scale,
        left_fraction=left_fraction,
        width_fraction=width_fraction,
        right_fraction=right_fraction,
        minimum_fraction=minimum_fraction,
        log_left_to_right=interior_log_ratio(left_gap, right_gap),
        log_width_to_right=interior_log_ratio(width_gap, right_gap),
    )


def threshold_dikin_metric(simplex: ThresholdSimplexPoint) -> np.ndarray:
    """Return the log-barrier Hessian for the threshold-simplex slacks.

    If ``d = (dc1, dc2)``, the three slack changes are

    ``(dc1, dc2-dc1, -dc2)``.

    Consequently ``d.T @ metric @ d`` is the sum of their squared relative
    changes.  The metric is defined only in the strict simplex interior; this
    is intentional because a Dikin trust region is an interior-point model.
    """
    slacks = np.array(
        [simplex.left_gap, simplex.width_gap, simplex.right_gap],
        dtype=np.float64,
    )
    if simplex.simplex_scale <= 0.0 or not np.all(np.isfinite(slacks)):
        raise ValueError("threshold simplex has no finite interior")
    if np.any(slacks <= 0.0):
        raise ValueError(
            "Dikin trust metric requires strictly positive left, width, and right slacks"
        )
    slack_jacobian = np.array(
        [[1.0, 0.0], [-1.0, 1.0], [0.0, -1.0]],
        dtype=np.float64,
    )
    metric = slack_jacobian.T @ np.diag(1.0 / (slacks * slacks)) @ slack_jacobian
    if not np.all(np.isfinite(metric)):
        raise ValueError("nonfinite Dikin trust metric")
    return metric


def threshold_trust_metric(
        *,
        simplex: ThresholdSimplexPoint,
        sensitivity_metric: np.ndarray,
        state_h1: float,
) -> ThresholdTrustMetric:
    """Combine simplex feasibility and equilibrium-response geometry.

    The pullback term is ``S.T K S / ||u||_H1**2``.  Roundoff-sized negative
    eigenvalues of the assembled Gram matrix are clipped to zero; a materially
    indefinite Gram matrix is rejected as an assembly error.  When the state
    norm is unavailable or zero, the Dikin term remains active and the record
    explicitly marks the pullback as unavailable.
    """
    dikin = threshold_dikin_metric(simplex)
    raw_pullback = np.asarray(sensitivity_metric, dtype=np.float64)
    pullback_available = (
        raw_pullback.shape == (2, 2)
        and np.all(np.isfinite(raw_pullback))
        and math.isfinite(float(state_h1))
        and float(state_h1) > 0.0
    )
    if pullback_available:
        symmetric = 0.5 * (raw_pullback + raw_pullback.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        spectral_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        negative_tolerance = 256.0 * np.finfo(np.float64).eps * spectral_scale
        if float(eigenvalues[0]) < -negative_tolerance:
            raise ValueError("assembled sensitivity pullback metric is not positive semidefinite")
        clipped = np.maximum(eigenvalues, 0.0)
        positive_semidefinite = (eigenvectors * clipped) @ eigenvectors.T
        pullback = positive_semidefinite / (float(state_h1) ** 2)
    else:
        pullback = np.zeros((2, 2), dtype=np.float64)
    combined = 0.5 * (dikin + pullback + (dikin + pullback).T)
    combined_eigenvalues = np.linalg.eigvalsh(combined)
    eigen_min = float(combined_eigenvalues[0])
    eigen_max = float(combined_eigenvalues[1])
    if not math.isfinite(eigen_min) or eigen_min <= 0.0:
        raise ValueError("combined threshold trust metric is not positive definite")
    return ThresholdTrustMetric(
        dikin=dikin,
        pullback=pullback,
        combined=combined,
        eigenvalues=(eigen_min, eigen_max),
        condition=eigen_max / eigen_min,
        pullback_available=pullback_available,
    )


def quadratic_metric_norm(increment: np.ndarray, metric: np.ndarray) -> float:
    """Return ``sqrt(increment.T @ metric @ increment)`` robustly."""
    vector = np.asarray(increment, dtype=np.float64)
    matrix = np.asarray(metric, dtype=np.float64)
    squared = float(vector.dot(matrix).dot(vector))
    roundoff_scale = max(
        float(np.linalg.norm(vector)) ** 2 * float(np.linalg.norm(matrix, ord=2)),
        1.0,
    )
    if squared < 0.0 and abs(squared) <= 64.0 * np.finfo(np.float64).eps * roundoff_scale:
        squared = 0.0
    return math.sqrt(squared) if squared >= 0.0 else math.nan


def quadratic_metric_dual_norm(covector: np.ndarray, metric: np.ndarray) -> float:
    """Return the dual norm induced by a positive-definite metric."""
    gradient = np.asarray(covector, dtype=np.float64)
    matrix = np.asarray(metric, dtype=np.float64)
    return math.sqrt(max(float(gradient.dot(np.linalg.solve(matrix, gradient))), 0.0))


def h1_seminorm(*, comm: MPI.Comm, function: fem.Function, dx) -> float:
    """Return the global H0^1 seminorm of a scalar finite-element field."""
    squared = assemble_scalar(
        comm,
        ufl.inner(ufl.grad(function), ufl.grad(function)) * dx,
    )
    return math.sqrt(max(float(squared), 0.0))


def sensitivity_pullback_metric(
        *,
        comm: MPI.Comm,
        s1: fem.Function,
        s2: fem.Function,
        dx,
) -> tuple[np.ndarray, tuple[float, float], float]:
    """Assemble the H0^1 pullback metric of the equilibrium sensitivities."""
    metric = np.array(
        [
            [
                assemble_scalar(comm, ufl.inner(ufl.grad(s1), ufl.grad(s1)) * dx),
                assemble_scalar(comm, ufl.inner(ufl.grad(s1), ufl.grad(s2)) * dx),
            ],
            [0.0, assemble_scalar(comm, ufl.inner(ufl.grad(s2), ufl.grad(s2)) * dx)],
        ],
        dtype=np.float64,
    )
    metric[1, 0] = metric[0, 1]
    eigenvalues = np.linalg.eigvalsh(metric)
    eigen_min = float(eigenvalues[0])
    eigen_max = float(eigenvalues[1])
    condition = (
        eigen_max / eigen_min
        if eigen_min > 0.0 and math.isfinite(eigen_max)
        else math.inf
    )
    return metric, (eigen_min, eigen_max), condition


def predicted_state_h1(
        dc: np.ndarray,
        sensitivity_metric: np.ndarray,
) -> float:
    """Return the H0^1 size predicted by the sensitivity pullback metric."""
    return quadratic_metric_norm(dc, sensitivity_metric)


def torsion_fraction_phi_target_thresholds(
        *,
        phi_target_max: float,
        alpha1: float,
        alpha2: float,
        c_min: float,
        c_max: float,
        min_width: float,
) -> tuple[float, float]:
    """Build the direct geometry-local threshold seed.

    The seed deliberately uses no density fit, quantile, area match, or prior
    equilibrium information. It applies two dimensionless fractions to the
    maximum target Poisson potential computed on the current geometry, then
    projects only onto the physical box and minimum-width constraint. A
    positive lower fraction keeps the zero boundary outside the window.

    Args:
        phi_target_max: Global maximum of the current target potential.
        alpha1: Lower dimensionless fraction.
        alpha2: Upper dimensionless fraction.
        c_min: Lower physical threshold bound.
        c_max: Upper physical threshold bound.
        min_width: Minimum admissible threshold separation.

    Returns:
        Admissible absolute potential thresholds ``(c1,c2)``.
    """
    if not math.isfinite(phi_target_max) or phi_target_max <= 0.0:
        raise ValueError("direct initialization requires positive finite max(phi_target)")
    if not (0.0 < float(alpha1) < float(alpha2) < 1.0):
        raise ValueError("direct initialization requires 0 < alpha1 < alpha2 < 1")
    return project_thresholds(
        float(alpha1) * float(phi_target_max),
        float(alpha2) * float(phi_target_max),
        c_min=float(c_min),
        c_max=float(c_max),
        min_width=float(min_width),
    )


def threshold_constant_mismatch(
        c1_const,
        c2_const,
        eps_const,
        *,
        c1: float,
        c2: float,
        eps_phi: float,
) -> float:
    """Return the maximum absolute mismatch between constants and a pair.

    This guard makes the accepted-state contract observable: the residual,
    Newton Jacobian, sensitivities, and reduced gradients must all be assembled
    with constants belonging to the same threshold pair.
    """

    def scalar_value(constant) -> float:
        values = np.asarray(constant.value)
        return float(np.real(values.reshape(-1)[0]))

    return max(
        abs(scalar_value(c1_const) - float(c1)),
        abs(scalar_value(c2_const) - float(c2)),
        abs(scalar_value(eps_const) - float(eps_phi)),
    )


def synchronize_threshold_constants(
        c1_const,
        c2_const,
        eps_const,
        *,
        c1: float,
        c2: float,
        eps_phi: float,
) -> float:
    """Assign mutable threshold constants and verify their shared state.

    Returns:
        Maximum post-assignment mismatch, normally roundoff zero.

    Raises:
        RuntimeError: If the constants do not retain the accepted values.
    """
    c1_const.value = PETSc.ScalarType(c1)
    c2_const.value = PETSc.ScalarType(c2)
    eps_const.value = PETSc.ScalarType(eps_phi)
    mismatch = threshold_constant_mismatch(
        c1_const,
        c2_const,
        eps_const,
        c1=c1,
        c2=c2,
        eps_phi=eps_phi,
    )
    scale = max(abs(float(c1)), abs(float(c2)), abs(float(eps_phi)), 1.0)
    if mismatch > 64.0 * np.finfo(np.float64).eps * scale:
        raise RuntimeError(
            "threshold constants are inconsistent with the accepted pair: "
            f"mismatch={mismatch:.6e}"
        )
    return mismatch


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
    q = min(max(float(quantile), 0.0), 1.0)
    local_values = np.asarray(values, dtype=np.float64)
    local_weights = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(local_values) & np.isfinite(local_weights) & (local_weights > 0.0)
    gathered = comm.gather((local_values[mask], local_weights[mask]), root=0)
    result = math.nan
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
                target = q * total
                index = int(np.searchsorted(cumulative, target, side="left"))
                index = min(max(index, 0), sorted_values.size - 1)
                result = float(sorted_values[index])
    return float(comm.bcast(result, root=0))


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
    solve.  The target indicator is the smoothed torsion design
    ``rho_design/rho_amp`` clipped to ``[0,1]``; this is deliberate because the
    initializer should be robust to the same smoothing used to create the
    Poisson target.

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
        rho_design: Smoothed torsion-designed density.
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
    candidates: list[InitialWindowCandidate] = []

    def append_candidate(name: str, c1: float, c2: float) -> None:
        """Add one projected/scored candidate if it is finite."""
        candidate = initial_candidate_from_thresholds(
            comm=comm,
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
    q05 = global_weighted_quantile(comm, phi_values, target_weights, 0.05)
    q10 = global_weighted_quantile(comm, phi_values, target_weights, 0.10)
    q50 = global_weighted_quantile(comm, phi_values, target_weights, 0.50)
    q90 = global_weighted_quantile(comm, phi_values, target_weights, 0.90)
    q95 = global_weighted_quantile(comm, phi_values, target_weights, 0.95)
    if math.isfinite(q05) and math.isfinite(q95) and q95 > q05:
        append_candidate("target_quantile_05_95", q05, q95)
    if math.isfinite(q10) and math.isfinite(q90) and q90 > q10:
        append_candidate("target_quantile_10_90", q10, q90)
    if math.isfinite(q50):
        c1_area, c2_area = area_matched_initial_thresholds(
            comm=comm,
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
    return candidates


def globally_sampled_logistic_window_metrics(
        *,
        comm: MPI.Comm,
        phi_values: np.ndarray,
        target_values: np.ndarray,
        weights: np.ndarray,
        c1: float,
        c2: float,
        target_area: float,
        args: argparse.Namespace,
) -> LogisticWindowMetrics:
    """Evaluate frozen smooth geometry on distributed quadrature samples."""

    local = sampled_logistic_window_metrics(
        phi_values,
        target_values,
        weights,
        c1=float(c1),
        c2=float(c2),
        eps_mode=str(args.eps_mode),
        eps_ratio=float(args.eps_ratio),
        eps_fixed=args.eps_phi,
    )
    local_values = np.array(
        [
            local.leakage,
            local.overlap,
            local.grad_leakage[0],
            local.grad_leakage[1],
            local.grad_overlap[0],
            local.grad_overlap[1],
        ],
        dtype=np.float64,
    )
    global_values = np.empty_like(local_values)
    comm.Allreduce(local_values, global_values, op=MPI.SUM)
    leakage = float(global_values[0])
    overlap = float(global_values[1])
    grad_leakage = np.asarray(global_values[2:4], dtype=np.float64)
    grad_overlap = np.asarray(global_values[4:6], dtype=np.float64)
    missing = float(target_area) - overlap
    denominator = max(float(target_area) + leakage, np.finfo(np.float64).tiny)
    grad_missing = -grad_overlap
    grad_jaccard = (
        denominator * grad_overlap - overlap * grad_leakage
    ) / (denominator * denominator)
    return LogisticWindowMetrics(
        leakage=leakage,
        missing=missing,
        overlap=overlap,
        activity_area=overlap + leakage,
        jaccard=overlap / denominator,
        grad_leakage=grad_leakage,
        grad_missing=grad_missing,
        grad_overlap=grad_overlap,
        grad_jaccard=grad_jaccard,
    )


def refine_frozen_logistic_thresholds(
        *,
        comm: MPI.Comm,
        phi_values: np.ndarray,
        target_values: np.ndarray,
        weights: np.ndarray,
        initial_c1: float,
        initial_c2: float,
        c_min: float,
        c_max: float,
        min_width: float,
        target_area: float,
        leakage_cap: float | None,
        args: argparse.Namespace,
) -> tuple[float, float, LogisticWindowMetrics, int, str]:
    """Refine one hard-frontier pair with the actual logistic window.

    Every rank executes the same SLSQP iterations.  Objective and derivative
    callbacks use collective reductions, so the optimizer sees exact global
    quadrature metrics without gathering the high-order sample arrays.
    """

    from scipy.optimize import minimize

    search_span = float(c_max) - float(c_min)
    if not math.isfinite(search_span) or search_span <= 0.0:
        raise ValueError("logistic refinement requires a nonempty threshold range")
    initial_c1, initial_c2 = project_thresholds(
        initial_c1,
        initial_c2,
        c_min=c_min,
        c_max=c_max,
        min_width=min_width,
    )
    free_span = search_span - float(min_width)
    if free_span < 0.0:
        raise ValueError("minimum threshold width exceeds the search range")

    # Coordinates are two simplex slacks, not raw thresholds:
    #
    #   x_left  = (c1-cmin)/free_span,
    #   x_width = (c2-c1-min_width)/free_span.
    #
    # Bounds x>=0 preserve c1>=cmin and c2-c1>=min_width even while SLSQP
    # probes an infeasible right-slack trial.  The sole geometric constraint
    # x_left+x_width<=1 enforces c2<=cmax.  This avoids evaluating the
    # logistic window at reversed or zero-width thresholds.
    if free_span <= mass_roundoff_tolerance(search_span, min_width):
        metrics = globally_sampled_logistic_window_metrics(
            comm=comm,
            phi_values=phi_values,
            target_values=target_values,
            weights=weights,
            c1=float(c_min),
            c2=float(c_max),
            target_area=target_area,
            args=args,
        )
        numerical_tolerance = mass_roundoff_tolerance(
            target_area,
            metrics.leakage,
            0.0 if leakage_cap is None else leakage_cap,
        )
        if metrics.overlap <= numerical_tolerance:
            raise RuntimeError(
                "the only admissible frozen logistic window has zero overlap"
            )
        if (
            leakage_cap is not None
            and metrics.leakage > float(leakage_cap)
        ):
            raise RuntimeError(
                "the only admissible frozen logistic window violates the "
                "unchanged leakage cap"
            )
        return float(c_min), float(c_max), metrics, 0, "DEGENERATE_SIMPLEX"

    x0 = np.array(
        [
            (initial_c1 - float(c_min)) / free_span,
            (initial_c2 - initial_c1 - float(min_width)) / free_span,
        ],
        dtype=np.float64,
    )
    x0 = np.clip(x0, 0.0, 1.0)
    if float(np.sum(x0)) > 1.0:
        x0 /= float(np.sum(x0))
    cache_key: tuple[str, str] | None = None
    cache_metrics: LogisticWindowMetrics | None = None

    def thresholds(x: np.ndarray) -> tuple[float, float]:
        c1 = float(c_min) + free_span * float(x[0])
        c2 = c1 + float(min_width) + free_span * float(x[1])
        return c1, c2

    def pull_back_gradient(gradient: np.ndarray) -> np.ndarray:
        gradient = np.asarray(gradient, dtype=np.float64)
        return free_span * np.array(
            [gradient[0] + gradient[1], gradient[1]],
            dtype=np.float64,
        )

    def evaluate(x: np.ndarray) -> LogisticWindowMetrics:
        nonlocal cache_key, cache_metrics
        key = (float(x[0]).hex(), float(x[1]).hex())
        if cache_key != key or cache_metrics is None:
            c1, c2 = thresholds(x)
            cache_metrics = globally_sampled_logistic_window_metrics(
                comm=comm,
                phi_values=phi_values,
                target_values=target_values,
                weights=weights,
                c1=c1,
                c2=c2,
                target_area=target_area,
                args=args,
            )
            cache_key = key
        return cache_metrics

    area_scale = max(float(target_area), np.finfo(np.float64).tiny)

    def objective(x: np.ndarray) -> float:
        metrics = evaluate(x)
        if leakage_cap is None:
            return 1.0 - float(metrics.jaccard)
        return float(metrics.missing) / area_scale

    def objective_jacobian(x: np.ndarray) -> np.ndarray:
        metrics = evaluate(x)
        gradient = (
            -np.asarray(metrics.grad_jaccard, dtype=np.float64)
            if leakage_cap is None
            else np.asarray(metrics.grad_missing, dtype=np.float64) / area_scale
        )
        return pull_back_gradient(gradient)

    constraints: list[dict[str, object]] = [
        {
            "type": "ineq",
            "fun": lambda x: float(1.0 - x[0] - x[1]),
            "jac": lambda x: np.array([-1.0, -1.0], dtype=np.float64),
        }
    ]
    if leakage_cap is not None:
        cap = float(leakage_cap)
        # SLSQP terminates with a finite feasibility tolerance.  Solve against
        # an infinitesimally tighter numerical cap so the independently
        # re-evaluated result remains on the scientifically prescribed side of
        # the original cap.  This is a roundoff-scale inward margin, never an
        # enlargement of the admissible set.
        enforced_cap = max(
            0.0,
            cap - 16.0 * mass_roundoff_tolerance(target_area, cap),
        )

        def cap_constraint(x: np.ndarray) -> float:
            return (enforced_cap - float(evaluate(x).leakage)) / area_scale

        def cap_constraint_jacobian(x: np.ndarray) -> np.ndarray:
            return -pull_back_gradient(
                evaluate(x).grad_leakage,
            ) / area_scale

        constraints.append(
            {
                "type": "ineq",
                "fun": cap_constraint,
                "jac": cap_constraint_jacobian,
            }
        )

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        jac=objective_jacobian,
        bounds=((0.0, 1.0), (0.0, 1.0)),
        constraints=tuple(constraints),
        options={"ftol": 1.0e-12, "maxiter": 100, "disp": False},
    )
    if not bool(result.success):
        raise RuntimeError(
            "frozen logistic refinement failed without fallback: "
            f"status={result.status} message={result.message}"
        )
    refined_c1, refined_c2 = thresholds(np.asarray(result.x, dtype=np.float64))
    refined_c1, refined_c2 = project_thresholds(
        refined_c1,
        refined_c2,
        c_min=c_min,
        c_max=c_max,
        min_width=min_width,
    )
    metrics = globally_sampled_logistic_window_metrics(
        comm=comm,
        phi_values=phi_values,
        target_values=target_values,
        weights=weights,
        c1=refined_c1,
        c2=refined_c2,
        target_area=target_area,
        args=args,
    )
    numerical_tolerance = mass_roundoff_tolerance(
        target_area,
        metrics.leakage,
        0.0 if leakage_cap is None else leakage_cap,
    )
    if metrics.overlap <= numerical_tolerance:
        raise RuntimeError(
            "frozen logistic refinement produced a trivial zero-overlap window"
        )
    if leakage_cap is not None and metrics.leakage > float(leakage_cap):
        raise RuntimeError(
            "frozen logistic refinement violated the unchanged leakage cap: "
            f"L={metrics.leakage:.12e} Lmax={float(leakage_cap):.12e}"
        )
    return (
        refined_c1,
        refined_c2,
        metrics,
        int(getattr(result, "nit", 0)),
        str(result.message),
    )


def write_frozen_frontier_records(
        path: Path,
        *,
        run_tag: str,
        frontier: FrozenFrontier,
        selected: FrozenFrontierPoint | None,
        selection_mode: str,
        leakage_cap_rel: float | None,
        leakage_cap: float | None,
        refined: tuple[float, float, LogisticWindowMetrics] | None = None,
) -> None:
    """Write the complete binned frontier and optional refined point."""

    fields = (
        "record", "runTag", "selectionMode", "selected", "lowerBin",
        "upperBin", "c1", "c2", "width", "leakage", "missing",
        "leakageRel", "missingRel", "overlap", "jaccard",
        "targetArea", "frontierPoints", "evaluatedIntervals",
        "numericalTolerance", "leakageCapRel", "leakageCap",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in frontier.points:
            writer.writerow(
                {
                    "record": "HARD_FRONTIER",
                    "runTag": run_tag,
                    "selectionMode": selection_mode,
                    "selected": int(selected is not None and point == selected),
                    "lowerBin": point.lower_bin,
                    "upperBin": point.upper_bin,
                    "c1": point.c1,
                    "c2": point.c2,
                    "width": point.width,
                    "leakage": point.leakage,
                    "missing": point.missing,
                    "leakageRel": point.leakage / frontier.target_area,
                    "missingRel": point.missing / frontier.target_area,
                    "overlap": point.overlap,
                    "jaccard": point.jaccard,
                    "targetArea": frontier.target_area,
                    "frontierPoints": len(frontier.points),
                    "evaluatedIntervals": frontier.evaluated_intervals,
                    "numericalTolerance": frontier.numerical_tolerance,
                    "leakageCapRel": leakage_cap_rel,
                    "leakageCap": leakage_cap,
                }
            )
        if refined is not None:
            c1, c2, metrics = refined
            writer.writerow(
                {
                    "record": "LOGISTIC_REFINED",
                    "runTag": run_tag,
                    "selectionMode": selection_mode,
                    "selected": 1,
                    "c1": c1,
                    "c2": c2,
                    "width": c2 - c1,
                    "leakage": metrics.leakage,
                    "missing": metrics.missing,
                    "leakageRel": metrics.leakage / frontier.target_area,
                    "missingRel": metrics.missing / frontier.target_area,
                    "overlap": metrics.overlap,
                    "jaccard": metrics.jaccard,
                    "targetArea": frontier.target_area,
                    "frontierPoints": len(frontier.points),
                    "evaluatedIntervals": frontier.evaluated_intervals,
                    "numericalTolerance": frontier.numerical_tolerance,
                    "leakageCapRel": leakage_cap_rel,
                    "leakageCap": leakage_cap,
                }
            )


def build_frozen_frontier_initial_candidate(
        *,
        phi_target: fem.Function,
        torsion: fem.Function,
        c1_t: float,
        c2_t: float,
        target_area: float,
        quadrature_degree: int,
        c_min: float,
        c_max: float,
        min_width: float,
        frontier_csv: Path,
        run_tag: str,
        args: argparse.Namespace,
) -> tuple[InitialWindowCandidate, FrozenFrontierInitialization]:
    """Select and smoothly refine the single frozen-frontier initializer."""

    comm = phi_target.function_space.mesh.comm
    phi_values, torsion_values, weights = quadrature_samples_for_fit(
        phi_target,
        torsion,
        quadrature_degree=int(quadrature_degree),
    )
    target_values = (
        (np.asarray(torsion_values) > float(c1_t))
        & (np.asarray(torsion_values) < float(c2_t))
    ).astype(np.float64)
    local_phi_max = float(np.max(phi_values)) if phi_values.size else -math.inf
    global_phi_max = float(comm.allreduce(local_phi_max, op=MPI.MAX))
    histogram_upper = min(float(c_max), global_phi_max)
    if not math.isfinite(histogram_upper) or histogram_upper <= float(c_min):
        histogram_upper = float(c_max)
    if histogram_upper - float(c_min) < float(min_width):
        histogram_upper = min(float(c_max), float(c_min) + float(min_width))
    local_histogram = weighted_pushforward_histogram(
        phi_values,
        target_values,
        weights,
        lower=float(c_min),
        upper=histogram_upper,
        bins=int(args.frozen_frontier_bins),
    )
    global_target_bins = np.empty_like(local_histogram.target_mass)
    global_outside_bins = np.empty_like(local_histogram.outside_mass)
    comm.Allreduce(local_histogram.target_mass, global_target_bins, op=MPI.SUM)
    comm.Allreduce(local_histogram.outside_mass, global_outside_bins, op=MPI.SUM)
    global_scalars = np.empty(5, dtype=np.float64)
    comm.Allreduce(
        np.array(
            [
                local_histogram.target_mass_total,
                local_histogram.outside_mass_total,
                local_histogram.target_mass_in_range,
                local_histogram.outside_mass_in_range,
                float(local_histogram.sample_count),
            ],
            dtype=np.float64,
        ),
        global_scalars,
        op=MPI.SUM,
    )
    histogram = PushforwardHistogram(
        edges=local_histogram.edges,
        target_mass=global_target_bins,
        outside_mass=global_outside_bins,
        target_mass_total=float(global_scalars[0]),
        outside_mass_total=float(global_scalars[1]),
        target_mass_in_range=float(global_scalars[2]),
        outside_mass_in_range=float(global_scalars[3]),
        sample_count=int(round(global_scalars[4])),
    )
    root_print(
        comm,
        "FROZEN_PUSHFORWARD "
        f"bins={args.frozen_frontier_bins} samples={histogram.sample_count} "
        f"phiRange=({float(c_min):.12e},{histogram_upper:.12e}) "
        f"targetMassSampled={histogram.target_mass_total:.12e} "
        f"targetMassAssembled={float(target_area):.12e} "
        f"targetMassInRange={histogram.target_mass_in_range:.12e} "
        f"outsideMassInRange={histogram.outside_mass_in_range:.12e}",
    )
    frontier: FrozenFrontier | None = None
    selection: FrozenFrontierPoint | None = None
    selection_error: str | None = None
    selection_mode = (
        "strict_leakage_cap"
        if args.frozen_leakage_cap_rel is not None
        else "hard_jaccard"
    )
    leakage_cap = (
        None
        if args.frozen_leakage_cap_rel is None
        else float(args.frozen_leakage_cap_rel) * float(target_area)
    )
    if comm.rank == 0:
        try:
            frontier = hard_window_pareto_frontier(
                histogram,
                target_area=float(target_area),
                min_width=float(min_width),
            )
            selection = select_frozen_frontier_point(
                frontier,
                leakage_cap=leakage_cap,
            )
        except (ValueError, RuntimeError) as error:
            selection_error = str(error)
        if frontier is not None:
            write_frozen_frontier_records(
                frontier_csv,
                run_tag=run_tag,
                frontier=frontier,
                selected=selection,
                selection_mode=selection_mode,
                leakage_cap_rel=args.frozen_leakage_cap_rel,
                leakage_cap=leakage_cap,
            )
    frontier = comm.bcast(frontier, root=0)
    selection = comm.bcast(selection, root=0)
    selection_error = comm.bcast(selection_error, root=0)
    if selection_error is not None or selection is None or frontier is None:
        root_print(
            comm,
            "FROZEN_FRONTIER_INFEASIBLE "
            f"mode={selection_mode} capRel={args.frozen_leakage_cap_rel} "
            f"cap={leakage_cap} reason={selection_error}",
        )
        raise RuntimeError(
            "frozen frontier initialization is infeasible: "
            f"{selection_error}"
        )

    root_print(
        comm,
        "FROZEN_FRONTIER_SELECTED "
        f"mode={selection_mode} bins={args.frozen_frontier_bins} "
        f"points={len(frontier.points)} evaluated={frontier.evaluated_intervals} "
        f"c1={selection.c1:.12e} c2={selection.c2:.12e} "
        f"Lrel={selection.leakage / target_area:.12e} "
        f"Mrel={selection.missing / target_area:.12e} "
        f"J={selection.jaccard:.12e} capRel={args.frozen_leakage_cap_rel}",
    )
    refined_c1, refined_c2, smooth_metrics, refinement_iterations, refinement_message = (
        refine_frozen_logistic_thresholds(
            comm=comm,
            phi_values=phi_values,
            target_values=target_values,
            weights=weights,
            initial_c1=selection.c1,
            initial_c2=selection.c2,
            c_min=c_min,
            c_max=c_max,
            min_width=min_width,
            target_area=target_area,
            leakage_cap=leakage_cap,
            args=args,
        )
    )
    if comm.rank == 0:
        write_frozen_frontier_records(
            frontier_csv,
            run_tag=run_tag,
            frontier=frontier,
            selected=selection,
            selection_mode=selection_mode,
            leakage_cap_rel=args.frozen_leakage_cap_rel,
            leakage_cap=leakage_cap,
            refined=(refined_c1, refined_c2, smooth_metrics),
        )
    eps_phi = epsilon_from_thresholds(args, refined_c1, refined_c2)
    root_print(
        comm,
        "FROZEN_LOGISTIC_REFINED "
        f"status=SELECTED iterations={refinement_iterations} "
        f"c1={refined_c1:.12e} c2={refined_c2:.12e} eps={eps_phi:.12e} "
        f"Lrel={smooth_metrics.leakage / target_area:.12e} "
        f"Mrel={smooth_metrics.missing / target_area:.12e} "
        f"J={smooth_metrics.jaccard:.12e} capRel={args.frozen_leakage_cap_rel}",
    )
    candidate = InitialWindowCandidate(
        name=f"frozen_frontier_{selection_mode}",
        c1=refined_c1,
        c2=refined_c2,
        eps_phi=eps_phi,
        leakage_rel=smooth_metrics.leakage / float(target_area),
        missing_rel=smooth_metrics.missing / float(target_area),
        activity_area=smooth_metrics.activity_area,
        area_rel=smooth_metrics.activity_area / float(target_area),
        active_jaccard=smooth_metrics.jaccard,
        l2_rel=math.nan,
        score=1.0 - smooth_metrics.jaccard,
    )
    return candidate, FrozenFrontierInitialization(
        hard_point=selection,
        smooth_metrics=smooth_metrics,
        selection_mode=selection_mode,
        leakage_cap_rel=args.frozen_leakage_cap_rel,
        leakage_cap=leakage_cap,
        bins=int(args.frozen_frontier_bins),
        histogram_lower=float(c_min),
        histogram_upper=histogram_upper,
        frontier_points=len(frontier.points),
        evaluated_intervals=frontier.evaluated_intervals,
        refinement_iterations=refinement_iterations,
        refinement_message=refinement_message,
        target_area_sampled=histogram.target_mass_total,
        target_area_assembled=float(target_area),
    )


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


def select_projected_initial_candidate(
        candidates: list[ProjectedInitialCandidate],
        *,
        require_converged: bool,
) -> ProjectedInitialCandidate:
    """Select the best projected initializer under an optional strict PDE gate.

    The historical finite failure penalty is useful for diagnostics, but it is
    not a convergence guarantee: minimum-score selection can still choose an
    unresolved state. A run requesting converged inner Newton solves must
    therefore filter the candidate set explicitly.
    """

    pool = list(candidates)
    if not pool:
        raise RuntimeError("no projected initial-window candidates were produced")
    if require_converged:
        pool = [candidate for candidate in pool if candidate.newton.converged]
        if not pool:
            details = ", ".join(
                f"{candidate.base.name}:{candidate.newton.status}:"
                f"{candidate.newton.residual:.3e}"
                for candidate in candidates
            )
            raise RuntimeError(
                "no automatic initial-window candidate reached the required "
                f"Newton tolerance ({details})"
            )
    return min(pool, key=lambda candidate: candidate.score)


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


class PersistentStiffnessSolver:
    """Reuse one assembled H0^1 stiffness matrix and PETSc solver.

    The energy primer may be called before many Newton correctors.  Its
    Sobolev metric never changes, so retaining the matrix and KSP avoids a
    fresh MUMPS analysis/factorization for every gradient step.
    """

    def __init__(
            self,
            stiffness_form,
            bc,
            template: fem.Function,
            *,
            prefix: str,
            solver: str,
            ksp_type: str | None,
            rtol: float,
            atol: float,
            max_it: int | None,
    ) -> None:
        self.a_form = fem.form(stiffness_form)
        self.bcs = [bc]
        self.matrix = fem_petsc.assemble_matrix(self.a_form, bcs=self.bcs)
        self.matrix.assemble()
        self.ksp = PETSc.KSP().create(template.function_space.mesh.comm)
        self.ksp.setOptionsPrefix(prefix)
        options = PETSc.Options()
        for key, value in solver_options(solver, ksp_type=ksp_type).items():
            options[f"{prefix}{key}"] = value
        if solver not in {"mumps", "lu"}:
            options[f"{prefix}ksp_rtol"] = rtol
            options[f"{prefix}ksp_atol"] = atol
            if max_it is not None:
                options[f"{prefix}ksp_max_it"] = max_it
        self.ksp.setFromOptions()
        self.ksp.setOperators(self.matrix)
        self.closed = False

    def solve(self, rhs_form, target: fem.Function) -> tuple[int, float, float]:
        """Assemble one residual RHS and solve into ``target``."""
        if self.closed:
            raise RuntimeError("persistent stiffness solver is closed")
        started = time.perf_counter()
        rhs = fem_petsc.assemble_vector(rhs_form)
        fem_petsc.apply_lifting(rhs, [self.a_form], [self.bcs])
        rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(rhs, self.bcs)
        self.ksp.solve(rhs, target.x.petsc_vec)
        target.x.scatter_forward()
        reason = self.ksp.getConvergedReason()
        iterations = int(self.ksp.getIterationNumber())
        residual = float(self.ksp.getResidualNorm())
        elapsed = time.perf_counter() - started
        rhs.destroy()
        if reason < 0:
            raise RuntimeError(
                f"persistent stiffness solve failed with PETSc reason {reason}"
            )
        return iterations, residual, elapsed

    def close(self) -> None:
        """Destroy the retained PETSc objects exactly once."""
        if self.closed:
            return
        self.ksp.destroy()
        self.matrix.destroy()
        self.closed = True


def prime_equilibrium_with_energy_descent(
        *,
        u: fem.Function,
        gradient: fem.Function,
        rho: fem.Function,
        test,
        dx,
        c1_const: fem.Constant,
        c2_const: fem.Constant,
        eps_const: fem.Constant,
        c1: float,
        c2: float,
        eps_phi: float,
        rho_amp: float,
        args: argparse.Namespace,
        solver: PersistentStiffnessSolver,
        prefix: str,
        context: str,
        outer_iteration: int = -1,
        homotopy_lambda: float = 1.0,
        homotopy_target_density=None,
        record_callback: Callable[[EnergyPrimerRecord], None] | None = None,
        plot_callback: Callable[[EnergyPrimerRecord], None] | None = None,
        statistics: EnergyPrimerStatistics | None = None,
        guard_callback: Callable[[], EnergyPrimerGuardResult] | None = None,
) -> EnergyPrimerResult:
    """Take a bounded number of guarded, residual-improving rescue steps.

    The H0^1 negative-gradient direction is the Picard displacement. Each
    candidate must decrease the exact equilibrium energy, preserve the
    selected activity branch, and materially decrease the H^-1 residual.
    Failure restores the last safe state. Newton remains the exact corrector.
    """
    comm = u.function_space.mesh.comm
    started = time.perf_counter()
    lambda_value = float(homotopy_lambda)
    if not bool(getattr(args, "energy_primer", False)) or lambda_value == 0.0:
        return EnergyPrimerResult(
            "DISABLED" if lambda_value != 0.0 else "SKIP_LAMBDA0",
            0,
            math.nan,
            math.nan,
            math.nan,
            0.0,
            0,
            0.0,
            time.perf_counter() - started,
        )
    if not 0.0 < lambda_value <= 1.0:
        raise ValueError("energy-primer homotopy lambda must lie in (0,1]")
    if lambda_value < 1.0 and homotopy_target_density is None:
        raise ValueError("energy primer below lambda=1 requires target density")

    synchronize_threshold_constants(
        c1_const,
        c2_const,
        eps_const,
        c1=c1,
        c2=c2,
        eps_phi=eps_phi,
    )
    initial_u = u.x.array.copy()
    nonlinear_density = window_density_const_ufl(
        u, c1_const, c2_const, eps_const, rho_amp
    )
    source_density = lambda_value * nonlinear_density
    if homotopy_target_density is not None:
        source_density += (1.0 - lambda_value) * homotopy_target_density
    residual_expr = (
        ufl.inner(ufl.grad(u), ufl.grad(test)) - source_density * test
    ) * dx
    primitive = window_primitive_const_ufl(u, c1_const, c2_const, eps_const)
    energy_expr = (
        0.5 * ufl.inner(ufl.grad(u), ufl.grad(u))
        - lambda_value * float(rho_amp) * primitive
    ) * dx
    if homotopy_target_density is not None:
        energy_expr += -(1.0 - lambda_value) * homotopy_target_density * u * dx
    residual_form = fem.form(residual_expr)
    energy_form = fem.form(energy_expr)
    gradient_norm_form = fem.form(
        ufl.inner(ufl.grad(gradient), ufl.grad(gradient)) * dx
    )

    def energy_value() -> float:
        local_value = fem.assemble_scalar(energy_form)
        return float(comm.allreduce(local_value, op=MPI.SUM))

    def state_is_finite() -> bool:
        local_ok = bool(np.all(np.isfinite(u.x.array)))
        return bool(comm.allreduce(local_ok, op=MPI.LAND))

    def update_density() -> None:
        if float(rho_amp) == 0.0:
            rho.x.array.fill(0.0)
            rho.x.scatter_forward()
        else:
            update_interpolated(
                rho,
                window_density_const_ufl(
                    u, c1_const, c2_const, eps_const, rho_amp
                ),
            )

    initial_energy = energy_value()

    def solve_gradient_norm() -> float:
        nonlocal solve_time
        _, _, linear_time = solver.solve(residual_form, gradient)
        solve_time += float(linear_time)
        gradient_squared = max(float(comm.allreduce(
            fem.assemble_scalar(gradient_norm_form), op=MPI.SUM
        )), 0.0)
        return math.sqrt(gradient_squared)
    current_energy = initial_energy
    accepted_steps = 0
    last_gradient_norm = math.inf
    last_alpha = 0.0
    last_backtracks = 0
    solve_time = 0.0
    restored = False

    initial_gradient_norm = math.nan
    last_residual_ratio = math.nan
    last_branch_overlap = math.nan
    last_activity_area = math.nan
    plot_time = 0.0

    def finish(status: str) -> EnergyPrimerResult:
        nonlocal restored
        if not state_is_finite():
            u.x.array[:] = initial_u
            u.x.scatter_forward()
            update_density()
            restored = True
            status = "INVALID_STATE_RESTORED"
        elapsed = time.perf_counter() - started
        result = EnergyPrimerResult(
            status=status,
            accepted_steps=accepted_steps,
            initial_energy=initial_energy,
            final_energy=current_energy,
            gradient_norm=last_gradient_norm,
            alpha=last_alpha,
            backtracks=last_backtracks,
            solve_time=solve_time,
            elapsed=elapsed,
            restored_initial_state=restored,
            initial_gradient_norm=initial_gradient_norm,
            residual_ratio=last_residual_ratio,
            branch_overlap=last_branch_overlap,
            activity_area=last_activity_area,
            plot_time=plot_time,
        )
        if record_callback is not None:
            record_callback(EnergyPrimerRecord(
                record="result",
                context=context,
                prefix=prefix,
                outer_iteration=outer_iteration,
                homotopy_lambda=lambda_value,
                primer_iteration=accepted_steps,
                energy_before=initial_energy,
                energy_after=current_energy,
                gradient_norm=last_gradient_norm,
                gradient_norm_after=last_gradient_norm,
                residual_ratio=last_residual_ratio,
                branch_overlap=last_branch_overlap,
                activity_area=last_activity_area,
                plot_time=plot_time,
                alpha=last_alpha,
                backtracks=last_backtracks,
                status=status,
                elapsed=elapsed,
            ))
        if statistics is not None:
            statistics.record(result)
        root_print(
            comm,
            f"ENERGY_PRIMER_RESULT context={context} prefix={prefix} "
            f"lambda={lambda_value:.6e} status={status} "
            f"steps={accepted_steps} energy0={initial_energy:.12e} "
            f"energy={current_energy:.12e} gradHminus1={last_gradient_norm:.6e} "
            f"residualRatio={last_residual_ratio:.6e} "
            f"branchOverlap={last_branch_overlap:.6e} "
            f"activityArea={last_activity_area:.6e} "
            f"computeTime={max(elapsed - plot_time, 0.0):.6f}s "
            f"plotTime={plot_time:.6f}s time={elapsed:.6f}s",
        )
        return result

    if not state_is_finite() or not math.isfinite(initial_energy):
        return finish("INVALID_INITIAL_STATE")

    try:
        current_gradient_norm = solve_gradient_norm()
    except RuntimeError as error:
        root_print(
            comm,
            f"ENERGY_PRIMER_LINEAR_FAIL context={context} prefix={prefix} "
            f"iteration=initial error={error}",
        )
        return finish("FAIL_LINEAR_RETAINED")
    initial_gradient_norm = current_gradient_norm
    last_gradient_norm = current_gradient_norm
    if not math.isfinite(current_gradient_norm):
        return finish("FAIL_NONFINITE_GRADIENT_RETAINED")
    if current_gradient_norm <= float(args.energy_primer_tol):
        return finish("CONVERGED_GRADIENT")

    for primer_k in range(int(args.energy_primer_max_it)):
        gradient_before = current_gradient_norm
        gradient_squared = gradient_before * gradient_before
        old_u = u.x.array.copy()
        energy_before = current_energy
        alpha = 1.0
        backtracks = 0
        accepted = False
        trial_energy = math.inf
        while (
                alpha >= float(args.alpha_min)
                and backtracks <= int(args.max_backtrack)
        ):
            u.x.array[:] = old_u - alpha * gradient.x.array
            u.x.scatter_forward()
            trial_energy = energy_value() if state_is_finite() else math.inf
            armijo_bound = (
                energy_before
                - float(args.armijo_c) * alpha * gradient_squared
            )
            if math.isfinite(trial_energy) and trial_energy <= armijo_bound:
                accepted = True
                break
            alpha *= float(args.beta_ls)
            backtracks += 1

        if not accepted:
            u.x.array[:] = old_u
            u.x.scatter_forward()
            update_density()
            restored = True
            last_alpha = alpha
            last_backtracks = backtracks
            return finish("FAIL_LS_RETAINED")

        if guard_callback is not None:
            guard = guard_callback()
            last_branch_overlap = float(guard.branch_overlap)
            last_activity_area = float(guard.activity_area)
            if not guard.accepted:
                u.x.array[:] = old_u
                u.x.scatter_forward()
                update_density()
                restored = True
                last_alpha = alpha
                last_backtracks = backtracks
                return finish(f"FAIL_GUARD_{guard.reason}_RETAINED")

        try:
            candidate_gradient_norm = solve_gradient_norm()
        except RuntimeError as error:
            u.x.array[:] = old_u
            u.x.scatter_forward()
            update_density()
            restored = True
            root_print(
                comm,
                f"ENERGY_PRIMER_LINEAR_FAIL context={context} prefix={prefix} "
                f"iteration={primer_k}_candidate error={error}",
            )
            return finish("FAIL_LINEAR_RESTORED")
        if not math.isfinite(candidate_gradient_norm):
            u.x.array[:] = old_u
            u.x.scatter_forward()
            update_density()
            restored = True
            return finish("FAIL_NONFINITE_GRADIENT_RESTORED")

        last_residual_ratio = (
            candidate_gradient_norm / gradient_before
            if gradient_before > 0.0
            else 0.0
        )
        if not residual_progress_is_material(
            gradient_before,
            candidate_gradient_norm,
            minimum_relative_reduction=float(
                args.energy_primer_min_residual_reduction
            ),
            tolerance=float(args.energy_primer_tol),
        ):
            u.x.array[:] = old_u
            u.x.scatter_forward()
            update_density()
            restored = True
            last_gradient_norm = gradient_before
            last_alpha = alpha
            last_backtracks = backtracks
            return finish("FAIL_RESIDUAL_PROGRESS_RETAINED")

        accepted_steps += 1
        current_energy = trial_energy
        current_gradient_norm = candidate_gradient_norm
        last_gradient_norm = candidate_gradient_norm
        last_alpha = alpha
        last_backtracks = backtracks
        update_density()
        record = EnergyPrimerRecord(
            record="step",
            context=context,
            prefix=prefix,
            outer_iteration=outer_iteration,
            homotopy_lambda=lambda_value,
            primer_iteration=primer_k,
            energy_before=energy_before,
            energy_after=current_energy,
            gradient_norm=gradient_before,
            alpha=alpha,
            backtracks=backtracks,
            status="ACCEPT",
            elapsed=time.perf_counter() - started,
            gradient_norm_after=candidate_gradient_norm,
            residual_ratio=last_residual_ratio,
            branch_overlap=last_branch_overlap,
            activity_area=last_activity_area,
        )
        step_plot_time = 0.0
        if plot_callback is not None:
            plot_started = time.perf_counter()
            plot_callback(record)
            local_plot_time = time.perf_counter() - plot_started
            step_plot_time = float(
                comm.allreduce(local_plot_time, op=MPI.MAX)
            )
            plot_time += step_plot_time
            record.plot_time = plot_time
            record.elapsed = time.perf_counter() - started
        if record_callback is not None:
            record_callback(record)
        if int(args.verbosity) >= 1:
            root_print(
                comm,
                f"ENERGY_PRIMER context={context} prefix={prefix} k={primer_k} "
                f"lambda={lambda_value:.6e} energy={current_energy:.12e} "
                f"gradHminus1Before={gradient_before:.6e} "
                f"gradHminus1After={candidate_gradient_norm:.6e} "
                f"residualRatio={last_residual_ratio:.6e} "
                f"alpha={alpha:.3e} backtracks={backtracks} "
                f"branchOverlap={last_branch_overlap:.6e} "
                f"activityArea={last_activity_area:.6e} "
                f"stepPlotTime={step_plot_time:.6f}s "
                f"plotTime={plot_time:.6f}s",
            )
        if candidate_gradient_norm <= float(args.energy_primer_tol):
            return finish("CONVERGED_GRADIENT")

    return finish("MAX_STEPS_RETAINED")


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
        help="Portable v2 equilibrium checkpoint (default: RUN_DIR/out/equilibrium.npz).",
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
        help="torsion-target smoothing divided by c2T-c1T; zero uses the sharp band indicator",
    )
    parser.add_argument("--rho-amp", type=float, default=None)
    parser.add_argument("--eps-mode", choices=("relative", "fixed"), default="relative")
    parser.add_argument("--eps-ratio", "--eps-phi-ratio", dest="eps_ratio", type=float, default=0.08)
    parser.add_argument("--eps-phi", type=float, default=None)
    parser.add_argument("--c1-phi", dest="c1_phi", type=float, default=None)
    parser.add_argument("--c2-phi", dest="c2_phi", type=float, default=None)
    parser.add_argument("--initial-state", type=Path, default=None)
    parser.add_argument(
        "--initial-threshold-mode",
        choices=("ensemble", "torsion-fraction-phi-target", "frozen-frontier"),
        default="ensemble",
        help=(
            "ensemble compares L2, target-quantile, and area-matched seeds; "
            "torsion-fraction-phi-target uses exactly one seed "
            "(alpha1,alpha2)*max(phi_target); frozen-frontier computes the "
            "complete binned hard-window leakage/missing Pareto frontier, "
            "selects by a strict supplied leakage cap or otherwise Jaccard, "
            "and refines exactly with the logistic window"
        ),
    )
    parser.add_argument(
        "--initial-projection-mode",
        choices=("newton", "homotopy"),
        default="newton",
        help=(
            "project automatic threshold candidates directly at lambda=1 or "
            "continue rho_design to the nonlinear source with fixed thresholds"
        ),
    )
    parser.add_argument("--homotopy-initial-step", type=float, default=0.5)
    parser.add_argument("--homotopy-min-step", type=float, default=1.0e-3)
    parser.add_argument("--homotopy-max-step", type=float, default=0.5)
    parser.add_argument("--homotopy-step-grow", type=float, default=1.5)
    parser.add_argument("--homotopy-step-shrink", type=float, default=0.5)
    parser.add_argument("--homotopy-max-stages", type=int, default=64)
    parser.add_argument(
        "--homotopy-tol-res",
        type=float,
        default=None,
        help="Newton residual tolerance at every homotopy stage (default: inner/tol-res)",
    )
    parser.add_argument(
        "--homotopy-predictor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the branch tangent as the initial guess for each homotopy corrector",
    )
    parser.add_argument(
        "--initial-alpha1",
        type=float,
        default=None,
        help=(
            "lower fraction of max(phi_target) in torsion-fraction mode "
            "(default: --alphaT1)"
        ),
    )
    parser.add_argument(
        "--initial-alpha2",
        type=float,
        default=None,
        help=(
            "upper fraction of max(phi_target) in torsion-fraction mode "
            "(default: --alphaT2)"
        ),
    )
    parser.add_argument(
        "--require-direct-seed-interior-level-curves",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "require a converged torsion-fraction-phi-target seed to satisfy "
            "0 < c1 < c2 < max(phi); disable only to continue from a "
            "boundary-touching direct seed"
        ),
    )
    parser.add_argument("--include-fit-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fit-window-grid", type=int, default=64)
    parser.add_argument("--fit-window-refine-grid", type=int, default=25)
    parser.add_argument("--fit-window-refine-passes", type=int, default=2)
    parser.add_argument("--fit-window-bins", type=int, default=4096)
    parser.add_argument("--fit-window-quad-degree", type=int, default=None)
    parser.add_argument(
        "--frozen-frontier-bins",
        type=int,
        default=4096,
        help="number of deterministic potential bins used for the hard-window Pareto frontier",
    )
    parser.add_argument(
        "--frozen-leakage-cap-rel",
        type=float,
        default=None,
        help=(
            "scientifically prescribed frozen leakage cap divided by the "
            "crisp target-band area; omission selects by frozen Jaccard and "
            "the cap is never invented or relaxed"
        ),
    )
    parser.add_argument("--cmax-factor", type=float, default=1.25)
    parser.add_argument(
        "--threshold-cap-mode",
        choices=("target", "torsion"),
        default="target",
        help="bound thresholds by cmax-factor*max(phi_target) or max(T)",
    )
    parser.add_argument("--c-lower-fraction", type=float, default=0.0)
    parser.add_argument("--c-upper-fraction", type=float, default=1.0)
    parser.add_argument("--min-width-fraction", type=float, default=1.0e-3)
    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument("--delta-c", type=float, default=0.0)
    parser.add_argument("--max-opt-it", type=int, default=30)
    parser.add_argument("--tol-grad", type=float, default=1.0e-8)
    parser.add_argument("--tol-res", type=float, default=1.0e-10)
    parser.add_argument(
        "--verify-sensitivities",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="at the first threshold pair, compare sensitivities and reduced gradients with centered finite differences",
    )
    parser.add_argument(
        "--sensitivity-check-steps",
        default="1e-3,3e-4,1e-4",
        help="comma-separated perturbations as fractions of the local threshold-band width",
    )
    parser.add_argument("--sensitivity-check-newton-tol", type=float, default=1.0e-11)
    parser.add_argument("--sensitivity-check-max-newton-it", type=int, default=160)
    parser.add_argument("--inner-newton-tol", type=float, default=None)
    parser.add_argument(
        "--require-inner-newton-convergence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="require every base and trial threshold state to reach its requested Newton tolerance",
    )
    parser.add_argument("--inner-tol-max", type=float, default=1.0e-5)
    parser.add_argument("--inner-tol-gamma", type=float, default=1.0e-6)
    parser.add_argument("--max-newton-it", type=int, default=40)
    parser.add_argument(
        "--energy-primer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "after a recoverable Newton stall, restore the original predictor "
            "and take bounded, guarded H0^1 Sobolev-gradient steps before one "
            "Newton retry; every accepted step is plotted when plotting is active"
        ),
    )
    parser.add_argument(
        "--energy-primer-tol",
        type=float,
        default=1.0e-5,
        help="H^-1 residual/Sobolev-gradient norm that ends the primer",
    )
    parser.add_argument(
        "--energy-primer-max-it",
        type=int,
        default=3,
        help="maximum accepted rescue steps; must lie between one and three",
    )
    parser.add_argument(
        "--energy-primer-min-residual-reduction",
        type=float,
        default=5.0e-2,
        help="minimum relative H^-1 residual decrease required from every rescue step",
    )
    parser.add_argument(
        "--newton-stall-forecast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "abort recoverable damped-Newton correctors after repeated "
            "scale-free forecasts that cannot reach tolerance within budget"
        ),
    )
    parser.add_argument(
        "--newton-stall-window",
        type=int,
        default=4,
        help="number of consecutive damped accepted updates in each forecast",
    )
    parser.add_argument(
        "--newton-stall-patience",
        type=int,
        default=3,
        help="consecutive over-budget forecasts required for early rollback",
    )
    parser.add_argument("--final-newton-tol-res", type=float, default=None)
    parser.add_argument("--final-newton-max-it", type=int, default=200)
    parser.add_argument(
        "--picard-spectrum",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "measure the stiffness-relative extremal eigenvalues of the local "
            "Picard derivative at the selected initial and final equilibria"
        ),
    )
    parser.add_argument(
        "--picard-spectrum-eig-tol",
        type=float,
        default=1.0e-6,
        help="SLEPc tolerance for representative Picard-spectrum eigensolves",
    )
    parser.add_argument(
        "--picard-spectrum-eig-max-it",
        type=int,
        default=300,
        help="maximum SLEPc iterations for each extremal Picard eigenvalue",
    )
    parser.add_argument("--tol-step", type=float, default=1.0e-10)
    parser.add_argument("--max-backtrack", type=int, default=24)
    parser.add_argument("--alpha-min", type=float, default=1.0e-8)
    parser.add_argument("--beta-ls", type=float, default=0.5)
    parser.add_argument("--armijo-c", type=float, default=1.0e-6)
    parser.add_argument("--residual-norm", choices=("euclidean", "dual"), default="euclidean")
    parser.add_argument(
        "--search-space-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "record scale-free threshold-simplex coordinates, the H0^1 "
            "sensitivity pullback metric, and the H^-1 predictor defect for "
            "every proposed threshold step; the Dikin/pullback trust metric "
            "itself is always active"
        ),
    )
    parser.add_argument(
        "--trust-radius",
        type=float,
        default=0.5,
        help="initial dimensionless radius in the combined Dikin/pullback metric",
    )
    parser.add_argument("--trust-radius-min", type=float, default=1.0e-5)
    parser.add_argument(
        "--trust-radius-max",
        type=float,
        default=0.95,
        help="maximum metric radius; must remain below one to stay inside the simplex",
    )
    parser.add_argument("--trust-shrink", type=float, default=0.5)
    parser.add_argument("--trust-grow", type=float, default=1.8)
    parser.add_argument("--qp-hessian-floor", type=float, default=1.0e-14)
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
        "--jaccard-oscillation-stop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "stop after repeated active-Jaccard direction reversals without a "
            "new best value, restore the maximum-Jaccard state, "
            "and run the final exact Newton projection"
        ),
    )
    parser.add_argument(
        "--jaccard-stagnation-stop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "stop after the configured number of accepted threshold updates "
            "without a new best-Jaccard value, restore the "
            "maximum-Jaccard state, and run the final exact Newton projection"
        ),
    )
    parser.add_argument(
        "--jaccard-oscillation-min-accepted",
        type=int,
        default=6,
        help="minimum accepted threshold updates before Jaccard oscillation can stop the loop",
    )
    parser.add_argument(
        "--jaccard-oscillation-patience",
        type=int,
        default=4,
        help="accepted updates without a new best Jaccard value used as the oscillation window",
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
    parser.add_argument(
        "--fail-on-nonconvergence",
        action="store_true",
        help=(
            "deprecated compatibility flag; failure of the final exact Newton "
            "projection is always reported with a nonzero exit status"
        ),
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-mode", choices=("blocking", "nonblocking"), default="blocking")
    parser.add_argument("--plot-off-screen", action="store_true")
    parser.add_argument("--plot-window-width", type=int, default=1800)
    parser.add_argument("--plot-window-height", type=int, default=700)
    parser.add_argument(
        "--plot-design-hold-seconds",
        type=float,
        default=2.0,
        help=(
            "seconds to keep the initial window density rho(phi_design) "
            "visible before Newton starts in nonblocking interactive mode; "
            "zero disables"
        ),
    )
    parser.add_argument(
        "--plot-fields",
        choices=("full", "state", "density"),
        default="full",
        help=(
            "show the complete diagnostic panel set, current rho and phi "
            "side by side, or only density; compact views retain the "
            "target-band outline"
        ),
    )
    parser.add_argument(
        "--plot-initial-candidates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show each fully converged automatic initializer after its Newton projection",
    )
    parser.add_argument("--plot-optimization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-severe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--plot-accepted-states",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show every accepted Newton update and every accepted threshold-pair update",
    )
    parser.add_argument(
        "--plot-mesh-edges",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draw element edges; use --no-plot-mesh-edges for publication-resolution fields",
    )
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--plot-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--frame-dir", type=Path, default=None)
    parser.add_argument("--frame-every", type=int, default=None)
    parser.add_argument("--frame-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frame-window-width", type=int, default=1800)
    parser.add_argument("--frame-window-height", type=int, default=700)
    return parser.parse_args(argv)


def design_output_enabled(args: argparse.Namespace) -> bool:
    """Return whether the mandatory initial window density must be emitted."""
    return bool(args.plot or args.save_frames)


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
    if (args.initial_alpha1 is None) != (args.initial_alpha2 is None):
        raise ValueError("--initial-alpha1 and --initial-alpha2 must be supplied together")
    if args.initial_alpha1 is not None and not (
        0.0 < float(args.initial_alpha1) < float(args.initial_alpha2) < 1.0
    ):
        raise ValueError("require 0 < initial-alpha1 < initial-alpha2 < 1")
    if (
        args.initial_threshold_mode != "torsion-fraction-phi-target"
        and args.initial_alpha1 is not None
    ):
        raise ValueError("--initial-alpha1/2 require torsion-fraction-phi-target mode")
    if args.initial_threshold_mode == "torsion-fraction-phi-target":
        if args.c1_phi is not None or args.c2_phi is not None:
            raise ValueError("direct fractional initialization does not accept manual c1/c2")
        if args.initial_state is not None:
            raise ValueError("direct fractional initialization does not accept --initial-state")
    if args.initial_threshold_mode == "frozen-frontier":
        if args.c1_phi is not None or args.c2_phi is not None:
            raise ValueError("frozen-frontier initialization does not accept manual c1/c2")
        if args.initial_state is not None:
            raise ValueError("frozen-frontier initialization does not accept --initial-state")
    elif args.frozen_leakage_cap_rel is not None:
        raise ValueError("--frozen-leakage-cap-rel requires frozen-frontier mode")
    if args.frozen_frontier_bins < 64:
        raise ValueError("require --frozen-frontier-bins >= 64")
    if args.frozen_leakage_cap_rel is not None and (
        not math.isfinite(float(args.frozen_leakage_cap_rel))
        or float(args.frozen_leakage_cap_rel) < 0.0
    ):
        raise ValueError("require nonnegative finite --frozen-leakage-cap-rel")
    if args.initial_projection_mode == "homotopy" and args.initial_state is not None:
        raise ValueError("homotopy initialization starts from phi_target, not --initial-state")
    if args.initial_projection_mode == "homotopy" and args.c1_phi is not None:
        raise ValueError("homotopy initialization requires an automatic threshold candidate")
    if (
        args.initial_projection_mode == "homotopy"
        and args.initial_threshold_mode == "ensemble"
        and not args.include_fit_init
    ):
        raise ValueError("homotopy initialization requires automatic candidates")
    SourceHomotopySchedule(
        initial_step=float(args.homotopy_initial_step),
        min_step=float(args.homotopy_min_step),
        max_step=float(args.homotopy_max_step),
        step_grow=float(args.homotopy_step_grow),
        step_shrink=float(args.homotopy_step_shrink),
        max_attempts=int(args.homotopy_max_stages),
    )
    if args.homotopy_tol_res is not None and args.homotopy_tol_res <= 0.0:
        raise ValueError("require positive --homotopy-tol-res")

    if args.eps_mode == "relative" and args.eps_ratio <= 0.0:
        raise ValueError("require positive --eps-ratio")
    sensitivity_steps = [
        float(value.strip())
        for value in str(args.sensitivity_check_steps).split(",")
        if value.strip()
    ]
    if not sensitivity_steps or any(
            not math.isfinite(value) or value <= 0.0 for value in sensitivity_steps
    ):
        raise ValueError("--sensitivity-check-steps must contain positive finite values")
    if args.sensitivity_check_newton_tol <= 0.0:
        raise ValueError("require positive --sensitivity-check-newton-tol")
    if args.sensitivity_check_max_newton_it < 1:
        raise ValueError("require positive --sensitivity-check-max-newton-it")
    if args.eps_mode == "fixed" and (args.eps_phi is None or args.eps_phi <= 0.0):
        raise ValueError("require positive --eps-phi when --eps-mode fixed")
    if args.cmax_factor <= 0.0:
        raise ValueError("require positive --cmax-factor")
    if float(args.star_r0) <= abs(float(args.star_amp)):
        raise ValueError("star-shaped mesh generation requires --star-r0 > abs(--star-amp)")
    if args.c_lower_fraction < 0.0 or args.c_upper_fraction <= args.c_lower_fraction:
        raise ValueError("require 0 <= c-lower-fraction < c-upper-fraction")
    if args.min_width_fraction <= 0.0:
        raise ValueError("require positive --min-width-fraction")
    if (
        not math.isfinite(float(args.plot_design_hold_seconds))
        or float(args.plot_design_hold_seconds) < 0.0
    ):
        raise ValueError("require nonnegative finite --plot-design-hold-seconds")
    if args.kappa < 0.0:
        raise ValueError("require nonnegative --kappa")
    if args.delta_c < 0.0:
        raise ValueError("require nonnegative --delta-c")
    if args.max_opt_it < 0:
        raise ValueError("require nonnegative --max-opt-it")
    if args.tol_res <= 0.0 or args.inner_tol_max <= 0.0 or args.inner_tol_gamma <= 0.0:
        raise ValueError("require positive residual tolerances")
    if args.inner_newton_tol is not None and args.inner_newton_tol <= 0.0:
        raise ValueError("require positive --inner-newton-tol")
    if args.max_newton_it < 1:
        raise ValueError("require positive --max-newton-it")
    if not math.isfinite(args.energy_primer_tol) or args.energy_primer_tol <= 0.0:
        raise ValueError("require positive finite --energy-primer-tol")
    if not 1 <= args.energy_primer_max_it <= 3:
        raise ValueError("require 1 <= --energy-primer-max-it <= 3")
    if (
        not math.isfinite(args.energy_primer_min_residual_reduction)
        or not 0.0 <= args.energy_primer_min_residual_reduction < 1.0
    ):
        raise ValueError(
            "require 0 <= --energy-primer-min-residual-reduction < 1"
        )
    if args.energy_primer and not args.newton_stall_forecast:
        raise ValueError("--energy-primer requires --newton-stall-forecast")
    if (
        not math.isfinite(args.picard_spectrum_eig_tol)
        or args.picard_spectrum_eig_tol <= 0.0
    ):
        raise ValueError("require positive finite --picard-spectrum-eig-tol")
    if args.picard_spectrum_eig_max_it < 1:
        raise ValueError("require positive --picard-spectrum-eig-max-it")
    if args.newton_stall_window < 2:
        raise ValueError("require --newton-stall-window >= 2")
    if args.newton_stall_patience < 1:
        raise ValueError("require positive --newton-stall-patience")
    if args.final_newton_tol_res is not None and args.final_newton_tol_res <= 0.0:
        raise ValueError("require positive --final-newton-tol-res")
    if args.initial_state is not None and args.c1_phi is None:
        raise ValueError("--initial-state requires manual --c1-phi and --c2-phi")
    if args.final_newton_max_it < 1:
        raise ValueError("require positive --final-newton-max-it")
    if args.max_backtrack < 0:
        raise ValueError("require nonnegative --max-backtrack")
    if not (0.0 < args.beta_ls < 1.0):
        raise ValueError("require 0 < --beta-ls < 1")
    if args.trust_radius <= 0.0 or args.trust_radius_min <= 0.0 or args.trust_radius_max <= 0.0:
        raise ValueError("require positive trust radii")
    if args.trust_radius_min > args.trust_radius_max:
        raise ValueError("require trust-radius-min <= trust-radius-max")
    if args.trust_radius > args.trust_radius_max:
        raise ValueError("require trust-radius <= trust-radius-max")
    if args.trust_radius_max >= 1.0:
        raise ValueError("require --trust-radius-max < 1 for the Dikin interior guard")
    if args.trust_shrink <= 0.0 or args.trust_shrink >= 1.0:
        raise ValueError("require 0 < --trust-shrink < 1")
    if args.trust_grow <= 1.0:
        raise ValueError("require --trust-grow > 1")
    if args.eta_overlap < 0.0:
        raise ValueError("require nonnegative --eta-overlap")
    if args.min_activity_fraction < 0.0:
        raise ValueError("require nonnegative --min-activity-fraction")
    if args.jaccard_oscillation_min_accepted < 0:
        raise ValueError("require nonnegative --jaccard-oscillation-min-accepted")
    if args.jaccard_oscillation_patience < 3:
        raise ValueError("require --jaccard-oscillation-patience >= 3")
    if args.solver_preset == "legacy" and args.linear_solver is None:
        args.linear_solver = "mumps"
    if (args.c1_phi is None) != (args.c2_phi is None):
        raise ValueError("--c1-phi and --c2-phi must be supplied together")
    if args.c1_phi is not None and not (args.c2_phi > args.c1_phi >= 0.0):
        raise ValueError("require 0 <= c1_phi < c2_phi")


def phase_solver_args(args: argparse.Namespace, phase: str) -> argparse.Namespace:
    """Return a copy of the run arguments with a phase-specific solver.

    An explicit ``--linear-solver`` remains a global override for reproducible
    legacy runs. Otherwise the optimized preset uses CG/BoomerAMG for fixed
    coercive systems and accepted-state sensitivities, GMRES/BoomerAMG for
    changing Newton Jacobians, and MUMPS for the final exact projection.
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


def sensitivity_iterative_fallback_ksp(args: argparse.Namespace) -> str | None:
    """Return the guarded iterative fallback for a sensitivity CG solve."""
    if args.linear_solver in {"mumps", "lu"} or args.ksp_type != "cg":
        return None
    fallback = str(args.sensitivity_iterative_fallback)
    return None if fallback == "none" else fallback


def inner_tolerance(args: argparse.Namespace, merit: float) -> float:
    """Choose the adaptive Newton tolerance for outer-loop state projections.

    By default the outer tolerance scales directly with the dimensionless soft
    Jaccard loss.  It is clipped by ``--inner-tol-max`` and never allowed below
    ``--tol-res``.  This rule does not divide by an assumed attainable target
    value.  Supplying ``--inner-newton-tol`` disables the adaptive rule and
    forces every inner projection and trial correction to that fixed residual
    tolerance; this is mainly a testing and verification knob.

    Args:
        args: Parsed command-line namespace.
        merit: Current soft Jaccard loss.

    Returns:
        Residual tolerance for the next fixed-threshold Newton projection.
    """
    if args.inner_newton_tol is not None:
        return float(args.inner_newton_tol)
    merit_scale = max(float(merit), 1.0e-14)
    loose = min(float(args.inner_tol_max), float(args.inner_tol_gamma) * merit_scale)
    return max(float(args.tol_res), loose)


def homotopy_tolerance(args: argparse.Namespace) -> float:
    """Return the fully converged Newton tolerance for every homotopy stage."""
    if args.homotopy_tol_res is not None:
        return float(args.homotopy_tol_res)
    if args.inner_newton_tol is not None:
        return float(args.inner_newton_tol)
    return float(args.tol_res)


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


def jaccard_direction_changes(
        values: Sequence[float],
        *,
        patience: int,
) -> int:
    """Count strict direction reversals in recent Jaccard values.

    Only the last ``patience + 1`` accepted states are considered. Changes
    that are exactly zero are ignored.  No requested Jaccard tolerance enters
    the decision.
    """
    if patience < 3:
        raise ValueError("patience must be at least three")
    if len(values) < patience + 1:
        return 0
    window = np.asarray(values[-(patience + 1):], dtype=np.float64)
    if not np.all(np.isfinite(window)):
        return 0
    directions = [
        1 if delta > 0.0 else -1
        for delta in np.diff(window)
        if float(delta) != 0.0
    ]
    return sum(
        int(current != previous)
        for previous, current in zip(directions, directions[1:], strict=False)
    )


def jaccard_oscillation_detected(
        values: Sequence[float],
        *,
        patience: int,
) -> bool:
    """Return true after at least two strict recent reversals.

    The optimizer separately requires ``patience`` accepted updates without
    a new best value before acting on this signal.
    """
    return jaccard_direction_changes(
        values,
        patience=patience,
    ) >= 2


def jaccard_stagnation_detected(
        *,
        accepted_steps: int,
        stale_steps: int,
        minimum_accepted: int,
        patience: int,
) -> bool:
    """Return whether accepted updates have plateaued in best Jaccard."""
    return (
        int(accepted_steps) >= int(minimum_accepted)
        and int(stale_steps) >= int(patience)
    )


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
        plot_callback: Callable[[str, int, float, float, int, float, float, float], None] | None = None,
        homotopy_lambda: float = 1.0,
        homotopy_target_density=None,
        enable_stall_forecast: bool = False,
) -> NewtonResult:
    """Project the current state onto the fixed-threshold semilinear branch.

    For fixed ``(c1,c2,eps)``, this solves the nonlinear finite-element
    residual

        int grad(u).grad(v) dx - int rho_amp*W(u;c1,c2,eps)*v dx = 0

    with damped Newton.  When ``homotopy_target_density`` is supplied, the
    source is instead

        (1-lambda)*rho_design + lambda*rho_amp*W(u;c1,c2,eps),

    and the exact Jacobian scales the nonlinear derivative by ``lambda``.
    The density work function still stores ``rho_amp*W(u)`` so plots track the
    nonlinear density approaching the endpoint rather than the mixed source.
    The line search uses residual decrease as a merit condition.  A small
    Newton step is treated as stagnation unless the residual tolerance has
    already been reached; this is deliberate because the algorithm requires
    true residual convergence for every accepted homotopy and optimizer state.

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
        plot_callback: Optional hook called after each accepted Newton update.
            The callback receives ``(prefix, newton_iteration, residual,
            alpha, backtracks, c1, c2, eps_phi)``.  It is used only for severe
            plotting and must not change the numerical state.
        homotopy_lambda: Source-homotopy parameter.  Ordinary semilinear
            solves use the default value one.
        homotopy_target_density: Target density used by the linear endpoint at
            lambda zero.  It is required whenever ``homotopy_lambda != 1``.
        enable_stall_forecast: Enable early failure after repeated rolling
            forecasts exceed the budget available to this recoverable
            corrector.  Base-state and final exact projections leave this off.

    Returns:
        ``NewtonResult`` describing convergence, residual, damping, and solve
        time.
    """
    comm = u.function_space.mesh.comm
    c1_const.value = PETSc.ScalarType(c1)
    c2_const.value = PETSc.ScalarType(c2)
    eps_const.value = PETSc.ScalarType(eps_phi)
    lambda_value = float(homotopy_lambda)
    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError("homotopy_lambda must lie in [0,1]")
    nonlinear_density = window_density_const_ufl(
        u,
        c1_const,
        c2_const,
        eps_const,
        rho_amp,
    )

    def update_nonlinear_density_output() -> None:
        # UFL simplifies an exactly zero amplitude to a domain-free Zero,
        # which cannot be wrapped in a DOLFINx Expression.  Keep the
        # diagnostic field well-defined for that (useful testing) limit.
        if float(rho_amp) == 0.0:
            rho.x.array[:] = 0.0
            rho.x.scatter_forward()
        else:
            update_interpolated(rho, nonlinear_density)

    if homotopy_target_density is None:
        if abs(lambda_value - 1.0) > 1.0e-14:
            raise ValueError("homotopy_target_density is required below lambda=1")
        source_density = nonlinear_density
    else:
        source_density = (
            (1.0 - lambda_value) * homotopy_target_density
            + lambda_value * nonlinear_density
        )
    residual_expr = (
        ufl.inner(ufl.grad(u), ufl.grad(test))
        - source_density * test
    ) * dx
    jac_expr = (
        ufl.inner(ufl.grad(trial), ufl.grad(test))
        - lambda_value * float(rho_amp)
        * window_s_derivative_activity_ufl(u, c1_const, c2_const, eps_const)
        * trial * test
    ) * dx
    status = "MAX_NEWTON"
    converged = False
    initial_budget = int(args.max_newton_it)
    soft_cap_enabled = bool(getattr(args, "newton_soft_cap", False))
    hard_cap_reason = str(getattr(args, "newton_hard_cap_reason", "")).strip()
    hard_ceiling = (
        newton_hard_ceiling(
            initial_budget,
            float(getattr(args, "newton_soft_cap_factor", 1.0)),
        )
        if soft_cap_enabled and not hard_cap_reason
        else initial_budget
    )
    iteration_budget = initial_budget
    args._newton_initial_budget = initial_budget
    args._newton_current_budget = iteration_budget
    args._newton_hard_ceiling = hard_ceiling
    args._newton_cap_extensions = 0
    args._newton_cap_reason = ""
    args._newton_contraction = math.nan
    args._newton_predicted_remaining = math.inf
    args._newton_stall_windows = 0
    args._newton_forecast_remaining_budget = 0
    last_step_h1 = math.inf
    last_alpha = 0.0
    last_bt = 0
    solve_time_total = 0.0
    final_residual = math.inf
    last_contraction = math.nan
    last_predicted_remaining = math.inf
    stall_windows = 0
    forecast_remaining_budget = 0
    first_direction_h1 = math.nan
    first_update_h1 = math.nan
    first_alpha = math.nan
    first_backtracks = 0

    def result(
            result_status: str,
            result_converged: bool,
            iterations: int,
            residual: float,
            step_h1: float,
            alpha: float,
            backtracks: int,
    ) -> NewtonResult:
        return NewtonResult(
            status=result_status,
            converged=result_converged,
            iterations=iterations,
            residual=residual,
            step_h1=step_h1,
            alpha=alpha,
            backtracks=backtracks,
            solve_time=solve_time_total,
            contraction=last_contraction,
            predicted_remaining=last_predicted_remaining,
            stall_windows=stall_windows,
            forecast_remaining_budget=forecast_remaining_budget,
            first_direction_h1=first_direction_h1,
            first_update_h1=first_update_h1,
            first_alpha=first_alpha,
            first_backtracks=first_backtracks,
        )

    residual_history: list[float] = []
    damping_history: list[float] = []
    k = 0
    while k < iteration_budget:
        update_nonlinear_density_output()
        residual_old = residual_norm(
            residual_expr,
            bc,
            comm=comm,
            mode=args.residual_norm,
            metric_form=stiffness_form,
            solver=args.linear_solver,
            ksp_type=args.ksp_type,
            rtol=args.linear_rtol,
            atol=args.linear_atol,
            max_it=args.linear_max_it,
            prefix=f"{prefix}_resnorm_{k}_",
        )
        final_residual = residual_old
        if not residual_history:
            residual_history.append(residual_old)
        if residual_old <= float(tol_res):
            status = "CONVERGED_RESIDUAL"
            converged = True
            return result(
                status, converged, k, residual_old,
                last_step_h1, last_alpha, last_bt,
            )

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
            fallback = fallback_solver_name(args)
            if fallback is None:
                raise
            root_print(
                comm,
                f"LINEAR_FALLBACK prefix={prefix}_newton_{k}_ "
                f"primary={args.linear_solver}/{args.ksp_type} fallback={fallback} error={primary_error}",
            )
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
        solve_time_total += solve_time
        step_h1 = math.sqrt(max(assemble_scalar(comm, ufl.inner(ufl.grad(du), ufl.grad(du)) * dx), 0.0))
        last_step_h1 = step_h1
        if k == 0:
            first_direction_h1 = step_h1
        if step_h1 < float(args.tol_step):
            status = "FAIL_STEP_STAGNATION"
            return result(status, False, k, residual_old, step_h1, 0.0, 0)

        old_u = u.x.array.copy()
        alpha = 1.0
        accepted = False
        bt = 0
        while alpha >= float(args.alpha_min) and bt <= int(args.max_backtrack):
            u.x.array[:] = old_u + alpha * du.x.array
            u.x.scatter_forward()
            update_nonlinear_density_output()
            residual_trial = residual_norm(
                residual_expr,
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
            update_nonlinear_density_output()
            return result("FAIL_LS", False, k, residual_old, step_h1, alpha, bt)

        last_alpha = alpha
        last_bt = bt
        if k == 0:
            first_update_h1 = alpha * step_h1
            first_alpha = alpha
            first_backtracks = bt
        if plot_callback is not None:
            plot_callback(prefix, k, final_residual, alpha, bt, c1, c2, eps_phi)
        if args.verbosity >= 2:
            root_print(
                comm,
                f"INNER_NEWTON prefix={prefix} k={k} res={final_residual:.6e} "
                f"alpha={alpha:.3e} bt={bt} stepH1={step_h1:.6e} "
                f"linIts={its} linRes={lin_res:.3e} linTime={solve_time:.6f}s",
            )
        if final_residual <= float(tol_res):
            return result(
                "CONVERGED_RESIDUAL",
                True,
                k + 1,
                final_residual,
                step_h1,
                last_alpha,
                last_bt,
            )
        residual_history.append(final_residual)
        damping_history.append(last_alpha)
        completed_iterations = k + 1
        if bool(enable_stall_forecast) and bool(
                getattr(args, "newton_stall_forecast", False)
        ):
            local_extension = (
                decide_newton_budget_extension(
                    residual_history,
                    initial_budget=initial_budget,
                    current_budget=iteration_budget,
                    hard_ceiling=hard_ceiling,
                    chunk=int(getattr(args, "newton_soft_cap_chunk", 1)),
                    trend_window=int(getattr(args, "newton_soft_cap_window", 2)),
                    maximum_contraction=float(
                        getattr(args, "newton_soft_cap_contraction", 0.98)
                    ),
                    soft_cap_enabled=soft_cap_enabled,
                    hard_cap_reason=hard_cap_reason,
                )
                if comm.rank == 0
                else None
            )
            extension_forecast: NewtonBudgetDecision = comm.bcast(
                local_extension, root=0
            )
            available_budget = (
                hard_ceiling if extension_forecast.extend else iteration_budget
            )
            local_forecast = (
                forecast_newton_stall(
                    residual_history,
                    damping_history,
                    tolerance=float(tol_res),
                    completed_iterations=completed_iterations,
                    available_budget=available_budget,
                    window=int(args.newton_stall_window),
                    patience=int(args.newton_stall_patience),
                    consecutive_exceedances=stall_windows,
                )
                if comm.rank == 0
                else None
            )
            forecast = comm.bcast(local_forecast, root=0)
            stall_windows = int(forecast.consecutive_exceedances)
            forecast_remaining_budget = int(forecast.remaining_budget)
            if forecast.eligible:
                last_contraction = float(forecast.geometric_contraction)
                last_predicted_remaining = float(forecast.predicted_remaining)
            else:
                last_contraction = math.nan
                last_predicted_remaining = math.inf
            args._newton_contraction = last_contraction
            args._newton_predicted_remaining = last_predicted_remaining
            args._newton_stall_windows = stall_windows
            args._newton_forecast_remaining_budget = forecast_remaining_budget
            if forecast.eligible and (
                    forecast.reason == "forecast_exceeds_budget"
                    or int(args.verbosity) >= 2
            ):
                predicted_text = (
                    "inf"
                    if not math.isfinite(forecast.predicted_remaining)
                    else str(int(forecast.predicted_remaining))
                )
                root_print(
                    comm,
                    f"NEWTON_STALL_FORECAST prefix={prefix} "
                    f"iterations={completed_iterations} residual={final_residual:.6e} "
                    f"contraction={forecast.geometric_contraction:.6e} "
                    f"predictedRemaining={predicted_text} "
                    f"remainingBudget={forecast.remaining_budget} "
                    f"stallWindows={stall_windows}/{args.newton_stall_patience} "
                    f"softCapTrend={extension_forecast.reason}",
                )
            if forecast.stalled:
                return result(
                    "FAIL_PREDICTED_STALL",
                    False,
                    completed_iterations,
                    final_residual,
                    last_step_h1,
                    last_alpha,
                    last_bt,
                )
        if completed_iterations >= iteration_budget:
            local_decision = (
                decide_newton_budget_extension(
                    residual_history,
                    initial_budget=initial_budget,
                    current_budget=iteration_budget,
                    hard_ceiling=hard_ceiling,
                    chunk=int(getattr(args, "newton_soft_cap_chunk", 1)),
                    trend_window=int(getattr(args, "newton_soft_cap_window", 2)),
                    maximum_contraction=float(
                        getattr(args, "newton_soft_cap_contraction", 0.98)
                    ),
                    soft_cap_enabled=soft_cap_enabled,
                    hard_cap_reason=hard_cap_reason,
                )
                if comm.rank == 0
                else None
            )
            decision: NewtonBudgetDecision = comm.bcast(local_decision, root=0)
            args._newton_cap_reason = decision.reason
            args._newton_contraction = decision.geometric_contraction
            if decision.extend:
                extension = decision.next_budget - iteration_budget
                iteration_budget = decision.next_budget
                args._newton_current_budget = iteration_budget
                args._newton_cap_extensions += 1
                if int(args.verbosity) >= 1:
                    root_print(
                        comm,
                        f"NEWTON_CAP_EXTEND prefix={prefix} "
                        f"iterations={completed_iterations} residual={final_residual:.6e} "
                        f"extension={extension} newBudget={iteration_budget} "
                        f"hardCeiling={hard_ceiling} "
                        f"contraction={decision.geometric_contraction:.6e}",
                    )
            else:
                return result(
                    status,
                    converged,
                    completed_iterations,
                    final_residual,
                    last_step_h1,
                    last_alpha,
                    last_bt,
                )
        k = completed_iterations

    return result(
        status, converged, k, final_residual,
        last_step_h1, last_alpha, last_bt,
    )




def solve_equilibrium_with_primer_rescue(
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
        context: str,
        outer_iteration: int,
        plot_callback: Callable[
            [str, int, float, float, int, float, float, float], None
        ] | None = None,
        homotopy_lambda: float = 1.0,
        homotopy_target_density=None,
        energy_gradient: fem.Function | None = None,
        energy_solver: PersistentStiffnessSolver | None = None,
        energy_record_callback: Callable[[EnergyPrimerRecord], None] | None = None,
        energy_plot_callback: Callable[[EnergyPrimerRecord], None] | None = None,
        energy_statistics: EnergyPrimerStatistics | None = None,
        energy_guard_callback: Callable[
            [], EnergyPrimerGuardResult
        ] | None = None,
) -> EquilibriumCorrectionResult:
    """Try Newton first, then restore, prime, and retry once after forecast stall."""
    comm = u.function_space.mesh.comm
    predictor_u = u.x.array.copy()

    def restore_predictor() -> None:
        u.x.array[:] = predictor_u
        u.x.scatter_forward()
        if float(rho_amp) == 0.0:
            rho.x.array.fill(0.0)
            rho.x.scatter_forward()
        else:
            update_interpolated(
                rho,
                window_density_const_ufl(
                    u, c1_const, c2_const, eps_const, rho_amp
                ),
            )

    initial_newton = solve_equilibrium(
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
        tol_res=tol_res,
        args=args,
        prefix=prefix,
        plot_callback=plot_callback,
        homotopy_lambda=homotopy_lambda,
        homotopy_target_density=homotopy_target_density,
        enable_stall_forecast=True,
    )
    rescue_available = (
        bool(getattr(args, "energy_primer", False))
        and energy_gradient is not None
        and energy_solver is not None
    )
    if (
        initial_newton.status != "FAIL_PREDICTED_STALL"
        or not rescue_available
    ):
        return EquilibriumCorrectionResult(
            newton=initial_newton,
            initial_newton=initial_newton,
            retry_newton=None,
            primer=None,
            rescue_triggered=False,
        )

    root_print(
        comm,
        f"ENERGY_PRIMER_TRIGGER context={context} prefix={prefix} "
        f"newtonStatus={initial_newton.status} "
        f"newtonIts={initial_newton.iterations} "
        f"residual={initial_newton.residual:.6e} restorePredictor=1",
    )
    restore_predictor()
    primer = prime_equilibrium_with_energy_descent(
        u=u,
        gradient=energy_gradient,
        rho=rho,
        test=test,
        dx=dx,
        c1_const=c1_const,
        c2_const=c2_const,
        eps_const=eps_const,
        c1=c1,
        c2=c2,
        eps_phi=eps_phi,
        rho_amp=rho_amp,
        args=args,
        solver=energy_solver,
        prefix=f"{prefix}_primer",
        context=context,
        outer_iteration=outer_iteration,
        homotopy_lambda=homotopy_lambda,
        homotopy_target_density=homotopy_target_density,
        record_callback=energy_record_callback,
        plot_callback=energy_plot_callback,
        statistics=energy_statistics,
        guard_callback=energy_guard_callback,
    )
    if primer.accepted_steps < 1:
        root_print(
            comm,
            f"ENERGY_PRIMER_RETRY_SKIP context={context} prefix={prefix} "
            f"primerStatus={primer.status} primerSteps=0",
        )
        return EquilibriumCorrectionResult(
            newton=initial_newton,
            initial_newton=initial_newton,
            retry_newton=None,
            primer=primer,
            rescue_triggered=True,
        )

    retry_newton = solve_equilibrium(
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
        tol_res=tol_res,
        args=args,
        prefix=f"{prefix}_rescue",
        plot_callback=plot_callback,
        homotopy_lambda=homotopy_lambda,
        homotopy_target_density=homotopy_target_density,
        enable_stall_forecast=True,
    )
    root_print(
        comm,
        f"ENERGY_PRIMER_RETRY context={context} prefix={prefix} "
        f"primerStatus={primer.status} primerSteps={primer.accepted_steps} "
        f"retryStatus={retry_newton.status} retryIts={retry_newton.iterations} "
        f"retryResidual={retry_newton.residual:.6e}",
    )
    return EquilibriumCorrectionResult(
        newton=retry_newton,
        initial_newton=initial_newton,
        retry_newton=retry_newton,
        primer=primer,
        rescue_triggered=True,
    )


def solve_source_homotopy_initialization(
        *,
        u: fem.Function,
        du: fem.Function,
        tangent: fem.Function,
        rho: fem.Function,
        target_density: fem.Function,
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
        args: argparse.Namespace,
        prefix: str,
        run_tag: str,
        candidate_name: str,
        stage_callback: Callable[[int, float, NewtonResult], None] | None = None,
        stage_writer: csv.DictWriter | None = None,
        stage_handle=None,
        energy_gradient: fem.Function | None = None,
        energy_solver: PersistentStiffnessSolver | None = None,
        energy_record_callback: Callable[[EnergyPrimerRecord], None] | None = None,
        energy_plot_callback: Callable[[EnergyPrimerRecord], None] | None = None,
        energy_statistics: EnergyPrimerStatistics | None = None,
        target_area: float | None = None,
        energy_activity_reference: fem.Function | None = None,
) -> HomotopyResult:
    """Continue the design source to the nonlinear source at fixed thresholds.

    Every proposed continuation value receives a tangent predictor followed by
    the same fully converged damped Newton correction used by the reduced
    optimizer.  A failed correction is never accepted: ``u`` is restored to
    the last accepted state, and the attempted ``delta_lambda`` is reduced by
    ``--homotopy-step-shrink`` before retrying.
    """
    comm = u.function_space.mesh.comm
    started = time.perf_counter()
    tolerance = homotopy_tolerance(args)
    schedule = SourceHomotopySchedule(
        initial_step=float(args.homotopy_initial_step),
        min_step=float(args.homotopy_min_step),
        max_step=float(args.homotopy_max_step),
        step_grow=float(args.homotopy_step_grow),
        step_shrink=float(args.homotopy_step_shrink),
        max_attempts=int(args.homotopy_max_stages),
    )
    controller = AdaptiveSourceHomotopy(schedule)
    tangent_solve_time = 0.0
    newton_solve_time = 0.0
    total_newton_iterations = 0

    def finish(status: str, converged: bool, last_newton: NewtonResult) -> HomotopyResult:
        return HomotopyResult(
            status=status,
            converged=converged,
            lambda_final=float(controller.lambda_value),
            stages=int(controller.accepted_steps),
            rejected_steps=int(controller.rejected_steps),
            total_newton_iterations=int(total_newton_iterations),
            tangent_solve_time=float(tangent_solve_time),
            newton_solve_time=float(newton_solve_time),
            elapsed=time.perf_counter() - started,
            last_newton=last_newton,
        )

    def write_stage(
            *,
            proposal,
            tangent_h1: float,
            predicted_residual: float,
            correction: EquilibriumCorrectionResult,
            accepted: bool,
            elapsed: float,
    ) -> None:
        if stage_writer is None:
            return
        corrected = correction.newton
        primer = correction.primer
        stage_writer.writerow({
            "record": "homotopy_stage",
            "runTag": run_tag,
            "candidate": candidate_name,
            "c1": c1,
            "c2": c2,
            "epsPhi": eps_phi,
            "attempt": proposal.attempt,
            "lambdaOld": proposal.lambda_old,
            "lambdaTrial": proposal.lambda_trial,
            "deltaLambda": proposal.delta_lambda,
            "tangentH1": tangent_h1,
            "predictedResidual": predicted_residual,
            "primerStatus": primer.status if primer is not None else "DISABLED",
            "primerSteps": primer.accepted_steps if primer is not None else 0,
            "primerGradientNorm": primer.gradient_norm if primer is not None else "",
            "primerEnergyInitial": primer.initial_energy if primer is not None else "",
            "primerEnergyFinal": primer.final_energy if primer is not None else "",
            "rescueTriggered": int(correction.rescue_triggered),
            "initialNewtonStatus": correction.initial_newton.status,
            "initialNewtonIterations": correction.initial_newton.iterations,
            "retryNewtonStatus": (
                correction.retry_newton.status
                if correction.retry_newton is not None
                else ""
            ),
            "retryNewtonIterations": (
                correction.retry_newton.iterations
                if correction.retry_newton is not None
                else ""
            ),
            "totalNewtonIterations": correction.total_newton_iterations,
            "correctedResidual": corrected.residual,
            "newtonStatus": corrected.status,
            "newtonIterations": corrected.iterations,
            "damping": corrected.alpha,
            "backtracks": corrected.backtracks,
            "accepted": int(accepted),
            "elapsed": elapsed,
        })
        if stage_handle is not None:
            stage_handle.flush()

    synchronize_threshold_constants(
        c1_const,
        c2_const,
        eps_const,
        c1=c1,
        c2=c2,
        eps_phi=eps_phi,
    )

    # The linear target solve normally already satisfies lambda=0.  Running it
    # through the Newton gate makes that invariant explicit even with an
    # iterative stiffness solve or a separately configured strict tolerance.
    lambda0_newton = solve_equilibrium(
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
        tol_res=tolerance,
        args=args,
        prefix=f"{prefix}_lambda0",
        homotopy_lambda=0.0,
        homotopy_target_density=target_density,
    )
    total_newton_iterations += int(lambda0_newton.iterations)
    newton_solve_time += float(lambda0_newton.solve_time)
    root_print(
        comm,
        f"HOMOTOPY_CHECK prefix={prefix} lambda=0 "
        f"status={lambda0_newton.status} residual={lambda0_newton.residual:.12e} "
        f"tol={tolerance:.3e}",
    )
    if not lambda0_newton.converged:
        return finish("FAIL_LAMBDA0", False, lambda0_newton)

    last_newton = lambda0_newton
    while not controller.complete and not controller.exhausted:
        proposal = controller.next_trial()
        stage_started = time.perf_counter()
        old_u = u.x.array.copy()
        tangent_h1 = 0.0
        if energy_activity_reference is not None:
            update_interpolated(
                energy_activity_reference,
                window_activity_const_ufl(
                    u, c1_const, c2_const, eps_const
                ),
            )

        if bool(args.homotopy_predictor):
            nonlinear_density = window_density_const_ufl(
                u,
                c1_const,
                c2_const,
                eps_const,
                rho_amp,
            )
            tangent_jacobian = (
                ufl.inner(ufl.grad(trial), ufl.grad(test))
                - proposal.lambda_old * float(rho_amp)
                * window_s_derivative_activity_ufl(
                    u,
                    c1_const,
                    c2_const,
                    eps_const,
                )
                * trial * test
            ) * dx
            tangent_rhs = (nonlinear_density - target_density) * test * dx
            try:
                _, _, tangent_time = solve_linear_form(
                    tangent_jacobian,
                    tangent_rhs,
                    tangent,
                    [bc],
                    prefix=f"{prefix}_tangent_{proposal.attempt}_",
                    solver=args.linear_solver,
                    ksp_type=args.ksp_type,
                    rtol=args.linear_rtol,
                    atol=args.linear_atol,
                    max_it=args.linear_max_it,
                    verbosity=args.verbosity,
                )
            except RuntimeError as primary_error:
                fallback = fallback_solver_name(args)
                if fallback is None:
                    root_print(
                        comm,
                        f"HOMOTOPY_TANGENT_FAIL prefix={prefix} "
                        f"lambda={proposal.lambda_old:.6e} error={primary_error}",
                    )
                    return finish("FAIL_TANGENT_SOLVE", False, last_newton)
                root_print(
                    comm,
                    f"LINEAR_FALLBACK prefix={prefix}_tangent_{proposal.attempt}_ "
                    f"primary={args.linear_solver}/{args.ksp_type} "
                    f"fallback={fallback} error={primary_error}",
                )
                try:
                    _, _, tangent_time = solve_linear_form(
                        tangent_jacobian,
                        tangent_rhs,
                        tangent,
                        [bc],
                        prefix=f"{prefix}_tangent_{proposal.attempt}_fallback_",
                        solver=fallback,
                        ksp_type=None,
                        rtol=args.linear_rtol,
                        atol=args.linear_atol,
                        max_it=args.linear_max_it,
                        verbosity=args.verbosity,
                    )
                except RuntimeError as fallback_error:
                    root_print(
                        comm,
                        f"HOMOTOPY_TANGENT_FAIL prefix={prefix} "
                        f"lambda={proposal.lambda_old:.6e} error={fallback_error}",
                    )
                    return finish("FAIL_TANGENT_SOLVE", False, last_newton)
            tangent_solve_time += float(tangent_time)
            tangent_h1 = math.sqrt(max(assemble_scalar(
                comm,
                ufl.inner(ufl.grad(tangent), ufl.grad(tangent)) * dx,
            ), 0.0))
        else:
            tangent.x.array[:] = 0.0
            tangent.x.scatter_forward()

        u.x.array[:] = old_u + proposal.delta_lambda * tangent.x.array
        u.x.scatter_forward()
        nonlinear_density = window_density_const_ufl(
            u,
            c1_const,
            c2_const,
            eps_const,
            rho_amp,
        )
        trial_source = (
            (1.0 - proposal.lambda_trial) * target_density
            + proposal.lambda_trial * nonlinear_density
        )
        predictor_residual_form = (
            ufl.inner(ufl.grad(u), ufl.grad(test)) - trial_source * test
        ) * dx
        predicted_residual = residual_norm(
            predictor_residual_form,
            bc,
            comm=comm,
            mode=args.residual_norm,
            metric_form=stiffness_form,
            solver=args.linear_solver,
            ksp_type=args.ksp_type,
            rtol=args.linear_rtol,
            atol=args.linear_atol,
            max_it=args.linear_max_it,
            prefix=f"{prefix}_predictor_{proposal.attempt}_",
        )
        guard_callback = None
        if energy_activity_reference is not None and target_area is not None:
            guard_callback = lambda: evaluate_energy_primer_guard(
                comm=comm,
                activity_ref=energy_activity_reference,
                u_trial=u,
                dx=dx,
                c1_const=c1_const,
                c2_const=c2_const,
                eps_const=eps_const,
                c1_trial=c1,
                c2_trial=c2,
                eps_trial=eps_phi,
                minimum_branch_overlap=(
                    0.0 if args.disable_branch_check else float(args.eta_overlap)
                ),
                minimum_activity_area=(
                    float(args.min_activity_fraction) * float(target_area)
                ),
            )
        stage_prefix = (
            f"{prefix}_lambda_{proposal.attempt}_"
            f"{proposal.lambda_trial:.6f}"
        ).replace(".", "p")
        correction = solve_equilibrium_with_primer_rescue(
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
            tol_res=tolerance,
            args=args,
            prefix=stage_prefix,
            context=f"homotopy:{candidate_name}",
            outer_iteration=-1,
            homotopy_lambda=proposal.lambda_trial,
            homotopy_target_density=target_density,
            energy_gradient=energy_gradient,
            energy_solver=energy_solver,
            energy_record_callback=energy_record_callback,
            energy_plot_callback=energy_plot_callback,
            energy_statistics=energy_statistics,
            energy_guard_callback=guard_callback,
        )
        trial_newton = correction.newton
        stage_primer = correction.primer
        total_newton_iterations += int(correction.total_newton_iterations)
        newton_solve_time += float(correction.total_newton_solve_time)
        last_newton = trial_newton
        stage_elapsed = time.perf_counter() - stage_started
        write_stage(
            proposal=proposal,
            tangent_h1=tangent_h1,
            predicted_residual=predicted_residual,
            correction=correction,
            accepted=trial_newton.converged,
            elapsed=stage_elapsed,
        )

        if trial_newton.converged:
            controller.accept(
                proposal,
                newton_iterations=int(correction.total_newton_iterations),
                backtracks=int(trial_newton.backtracks),
            )
            if stage_callback is not None:
                stage_callback(
                    int(controller.accepted_steps),
                    float(controller.lambda_value),
                    trial_newton,
                )
            if args.verbosity >= 1:
                root_print(
                    comm,
                    f"HOMOTOPY_STAGE prefix={prefix} attempt={proposal.attempt} "
                    f"accepted=1 lambda={controller.lambda_value:.6e} "
                    f"dlambda={proposal.delta_lambda:.6e} "
                    f"newtonIts={correction.total_newton_iterations} "
                    f"rescue={int(correction.rescue_triggered)} "
                    f"primerSteps={stage_primer.accepted_steps if stage_primer is not None else 0} "
                    f"residual={trial_newton.residual:.6e} "
                    f"alpha={trial_newton.alpha:.3e} "
                    f"backtracks={trial_newton.backtracks} "
                    f"nextStep={controller.step:.6e}",
                )
            continue

        u.x.array[:] = old_u
        u.x.scatter_forward()
        update_interpolated(
            rho,
            window_density_const_ufl(
                u,
                c1_const,
                c2_const,
                eps_const,
                rho_amp,
            ),
        )
        controller.reject(proposal)
        if args.verbosity >= 1:
            root_print(
                comm,
                f"HOMOTOPY_STAGE prefix={prefix} attempt={proposal.attempt} "
                f"accepted=0 lambda={controller.lambda_value:.6e} "
                f"trialLambda={proposal.lambda_trial:.6e} "
                f"dlambda={proposal.delta_lambda:.6e} "
                f"newtonStatus={trial_newton.status} "
                f"newtonIts={correction.total_newton_iterations} "
                f"rescue={int(correction.rescue_triggered)} "
                f"residual={trial_newton.residual:.6e} "
                f"nextStep={controller.step:.6e}",
            )
        if controller.below_minimum_step:
            return finish("FAIL_MIN_STEP", False, last_newton)

    if controller.complete:
        return finish("CONVERGED_LAMBDA1", True, last_newton)
    return finish("FAIL_MAX_STAGES", False, last_newton)


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
    pullback_metric, pullback_eigenvalues, pullback_condition = (
        sensitivity_pullback_metric(
            comm=comm,
            s1=s1,
            s2=s2,
            dx=dx,
        )
    )
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
        sensitivity_metric=pullback_metric,
        sensitivity_metric_eigenvalues=pullback_eigenvalues,
        sensitivity_metric_condition=pullback_condition,
    )


def relative_scalar_error(reference: float, approximation: float) -> float:
    """Return a symmetric relative error for a scalar derivative check."""
    return abs(float(reference) - float(approximation)) / max(
        abs(float(reference)), abs(float(approximation)), 1.0e-30
    )


def verify_reduced_gradient_finite_differences(
        *,
        output_path: Path,
        u: fem.Function,
        du: fem.Function,
        rho: fem.Function,
        s1: fem.Function,
        s2: fem.Function,
        trial,
        test,
        dx,
        bc,
        stiffness_form,
        tau_mask,
        c1_const: fem.Constant,
        c2_const: fem.Constant,
        eps_const: fem.Constant,
        rho_amp: float,
        c1: float,
        c2: float,
        c_min: float,
        c_max: float,
        min_width: float,
        c_scale: float,
        target_area: float,
        gradient: ReducedGradient,
        base_newton: NewtonResult,
        args: argparse.Namespace,
        nonlinear_args: argparse.Namespace,
        iteration: int,
) -> list[dict[str, Any]]:
    """Verify state sensitivities and reduced derivatives by centered solves.

    Each perturbed state starts from the first-order sensitivity predictor and
    is Newton-corrected to ``--sensitivity-check-newton-tol``.  The routine
    restores the exact base state and all mutable threshold constants before
    returning, so enabling the audit cannot change the optimization path.
    """
    comm = u.function_space.mesh.comm
    base_state = u.x.array.copy()
    base_eps = epsilon_from_thresholds(args, c1, c2)
    check_args = argparse.Namespace(**vars(nonlinear_args))
    check_args.max_newton_it = int(args.sensitivity_check_max_newton_it)
    step_fractions = [
        float(value.strip())
        for value in str(args.sensitivity_check_steps).split(",")
        if value.strip()
    ]
    step_scale = min(float(c_scale), float(c2) - float(c1))
    if step_scale <= 0.0:
        raise ValueError("finite-difference sensitivity check requires positive band width")
    rows: list[dict[str, Any]] = []
    component_data = (
        (1, s1, float(gradient.grad_l[0]), float(gradient.grad_m[0])),
        (2, s2, float(gradient.grad_l[1]), float(gradient.grad_m[1])),
    )

    try:
        for component, sensitivity, analytic_l, analytic_m in component_data:
            for step_index, step_fraction in enumerate(step_fractions):
                h = float(step_fraction) * step_scale
                minus_c1 = c1 - h if component == 1 else c1
                plus_c1 = c1 + h if component == 1 else c1
                minus_c2 = c2 - h if component == 2 else c2
                plus_c2 = c2 + h if component == 2 else c2
                feasible = (
                    c_min <= minus_c1
                    and plus_c2 <= c_max
                    and minus_c2 - plus_c1 >= min_width
                )
                if not feasible:
                    root_print(
                        comm,
                        f"SENSITIVITY_CHECK_SKIP component={component} "
                        f"stepFraction={step_fraction:.6e} reason=infeasible",
                    )
                    continue

                perturbed_states: dict[int, np.ndarray] = {}
                perturbed_metrics: dict[int, BandMetrics] = {}
                perturbed_newton: dict[int, NewtonResult] = {}
                for sign, label in ((-1, "minus"), (1, "plus")):
                    trial_c1 = c1 + (sign * h if component == 1 else 0.0)
                    trial_c2 = c2 + (sign * h if component == 2 else 0.0)
                    trial_eps = epsilon_from_thresholds(args, trial_c1, trial_c2)
                    u.x.array[:] = base_state + sign * h * sensitivity.x.array
                    u.x.scatter_forward()
                    check = solve_equilibrium(
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
                        rho_amp=rho_amp,
                        tol_res=float(args.sensitivity_check_newton_tol),
                        args=check_args,
                        prefix=(
                            f"sensitivity_check_{iteration}_c{component}_"
                            f"h{step_index}_{label}"
                        ),
                    )
                    check_metrics = evaluate_band_metrics(
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
                    perturbed_states[sign] = u.x.array.copy()
                    perturbed_metrics[sign] = check_metrics
                    perturbed_newton[sign] = check

                state_fd = fem.Function(u.function_space, name=f"stateFDc{component}")
                state_fd.x.array[:] = (
                    perturbed_states[1] - perturbed_states[-1]
                ) / (2.0 * h)
                state_fd.x.scatter_forward()
                state_error = fem.Function(
                    u.function_space, name=f"stateSensitivityErrorc{component}"
                )
                state_error.x.array[:] = state_fd.x.array - sensitivity.x.array
                state_error.x.scatter_forward()
                state_fd_l2 = math.sqrt(max(assemble_scalar(comm, state_fd * state_fd * dx), 0.0))
                sensitivity_l2 = math.sqrt(
                    max(assemble_scalar(comm, sensitivity * sensitivity * dx), 0.0)
                )
                state_error_l2 = math.sqrt(
                    max(assemble_scalar(comm, state_error * state_error * dx), 0.0)
                )
                state_fd_h1 = math.sqrt(max(
                    assemble_scalar(comm, ufl.inner(ufl.grad(state_fd), ufl.grad(state_fd)) * dx),
                    0.0,
                ))
                sensitivity_h1 = math.sqrt(max(
                    assemble_scalar(
                        comm,
                        ufl.inner(ufl.grad(sensitivity), ufl.grad(sensitivity)) * dx,
                    ),
                    0.0,
                ))
                state_error_h1 = math.sqrt(max(
                    assemble_scalar(
                        comm,
                        ufl.inner(ufl.grad(state_error), ufl.grad(state_error)) * dx,
                    ),
                    0.0,
                ))
                fd_l = (
                    perturbed_metrics[1].leakage - perturbed_metrics[-1].leakage
                ) / (2.0 * h)
                fd_m = (
                    perturbed_metrics[1].missing - perturbed_metrics[-1].missing
                ) / (2.0 * h)
                minus_newton = perturbed_newton[-1]
                plus_newton = perturbed_newton[1]
                row = {
                    "component": component,
                    "stepFraction": step_fraction,
                    "stepScale": step_scale,
                    "step": h,
                    "baseC1": c1,
                    "baseC2": c2,
                    "baseEps": base_eps,
                    "baseStatus": base_newton.status,
                    "baseResidual": base_newton.residual,
                    "minusC1": minus_c1,
                    "minusC2": minus_c2,
                    "plusC1": plus_c1,
                    "plusC2": plus_c2,
                    "minusStatus": minus_newton.status,
                    "plusStatus": plus_newton.status,
                    "minusIterations": minus_newton.iterations,
                    "plusIterations": plus_newton.iterations,
                    "minusResidual": minus_newton.residual,
                    "plusResidual": plus_newton.residual,
                    "analyticLeakageDerivative": analytic_l,
                    "finiteDifferenceLeakageDerivative": fd_l,
                    "leakageDerivativeAbsoluteError": abs(analytic_l - fd_l),
                    "leakageDerivativeRelativeError": relative_scalar_error(analytic_l, fd_l),
                    "analyticMissingDerivative": analytic_m,
                    "finiteDifferenceMissingDerivative": fd_m,
                    "missingDerivativeAbsoluteError": abs(analytic_m - fd_m),
                    "missingDerivativeRelativeError": relative_scalar_error(analytic_m, fd_m),
                    "stateDerivativeL2Norm": state_fd_l2,
                    "sensitivityL2Norm": sensitivity_l2,
                    "stateSensitivityL2AbsoluteError": state_error_l2,
                    "stateSensitivityL2RelativeError": state_error_l2 / max(
                        state_fd_l2, sensitivity_l2, 1.0e-30
                    ),
                    "stateDerivativeH1Seminorm": state_fd_h1,
                    "sensitivityH1Seminorm": sensitivity_h1,
                    "stateSensitivityH1AbsoluteError": state_error_h1,
                    "stateSensitivityH1RelativeError": state_error_h1 / max(
                        state_fd_h1, sensitivity_h1, 1.0e-30
                    ),
                    "valid": int(
                        base_newton.converged
                        and minus_newton.converged
                        and plus_newton.converged
                    ),
                }
                rows.append(row)
                root_print(
                    comm,
                    f"SENSITIVITY_CHECK component={component} h={h:.6e} "
                    f"valid={row['valid']} stateL2Rel={row['stateSensitivityL2RelativeError']:.6e} "
                    f"stateH1Rel={row['stateSensitivityH1RelativeError']:.6e} "
                    f"leakGradRel={row['leakageDerivativeRelativeError']:.6e} "
                    f"missingGradRel={row['missingDerivativeRelativeError']:.6e}",
                )
    finally:
        u.x.array[:] = base_state
        u.x.scatter_forward()
        c1_const.value = PETSc.ScalarType(c1)
        c2_const.value = PETSc.ScalarType(c2)
        eps_const.value = PETSc.ScalarType(base_eps)
        update_interpolated(
            rho,
            window_density_const_ufl(
                u, c1_const, c2_const, eps_const, rho_amp
            ),
        )

    if comm.rank == 0:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SENSITIVITY_CHECK_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    comm.barrier()
    return rows


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
        metric: np.ndarray | None = None,
) -> tuple[np.ndarray, float, str, bool]:
    """Solve a two-dimensional metric trust-region QP by enumeration.

    The model is

        min g.dot(d) + 0.5*hessian_scale*d.T*metric*d

    subject to linear inequalities and
    ``sqrt(d.T*metric*d) <= radius``.  A Cholesky change of coordinates turns
    the ellipsoid into a Euclidean circle.  Because there are only two
    variables, the transformed problem is solved by enumerating all plausible
    active-set candidates.

    Args:
        g: Objective gradient for the current QP objective.
        rows: Linear constraints ``a.dot(d) <= b``.
        radius: Dimensionless metric trust-region radius.
        hessian_scale: Positive scalar Hessian approximation.
        metric: Symmetric positive-definite trust metric.  ``None`` uses the
            identity and preserves the Euclidean helper behavior.

    Returns:
        Tuple ``(d, model_value, status, hit_boundary)``.  ``model_value`` is
        the objective value of the local quadratic model at ``d``.
    """
    g = np.asarray(g, dtype=np.float64)
    radius = float(radius)
    hessian_scale = max(float(hessian_scale), 1.0e-30)
    trust_metric = (
        np.eye(2, dtype=np.float64)
        if metric is None
        else np.asarray(metric, dtype=np.float64)
    )
    if trust_metric.shape != (2, 2) or not np.all(np.isfinite(trust_metric)):
        raise ValueError("trust metric must be a finite 2x2 matrix")
    trust_metric = 0.5 * (trust_metric + trust_metric.T)
    try:
        # numpy returns L with H=L L^T; R=L^T gives ||R d||_2^2=d^T H d.
        coordinate_map = np.linalg.cholesky(trust_metric).T
    except np.linalg.LinAlgError as error:
        raise ValueError("trust metric must be positive definite") from error
    coordinate_gradient = np.linalg.solve(coordinate_map.T, g)
    coordinate_rows = [
        (np.linalg.solve(coordinate_map.T, np.asarray(a, dtype=np.float64)), float(b))
        for a, b in rows
    ]

    def model(y: np.ndarray) -> float:
        """Evaluate the scalar quadratic trust-region model at ``d``.

        Args:
            d: Candidate two-component threshold increment.

        Returns:
            Model value ``g.dot(d) + 0.5*hessian_scale*||d||^2``.
        """
        return float(
            coordinate_gradient.dot(y) + 0.5 * hessian_scale * y.dot(y)
        )

    candidates: list[np.ndarray] = []

    def add_candidate(y: np.ndarray) -> None:
        """Append a candidate only if it satisfies every QP constraint.

        The enumerator generates many algebraic candidates, including points
        from inactive constraints.  Filtering at insertion keeps the final
        minimization simple and prevents invalid active-set points from
        influencing the model minimum.
        """
        if feasible_qp_point(y, coordinate_rows, radius):
            candidates.append(np.asarray(y, dtype=np.float64))

    origin = np.zeros(2, dtype=np.float64)
    add_candidate(origin)
    y0 = -coordinate_gradient / hessian_scale
    add_candidate(y0)
    norm_g = float(np.linalg.norm(coordinate_gradient))
    if norm_g > 0.0:
        add_candidate(-radius * coordinate_gradient / norm_g)

    for a, b in coordinate_rows:
        aa = float(a.dot(a))
        if aa <= 0.0:
            continue
        projected = y0 - a * ((float(a.dot(y0)) - float(b)) / aa)
        add_candidate(projected)
        tangent = np.array([-a[1], a[0]], dtype=np.float64)
        tt = float(tangent.dot(tangent))
        particular = a * (float(b) / aa)
        remaining = radius * radius - float(particular.dot(particular))
        if tt > 0.0 and remaining >= -1.0e-12:
            root = math.sqrt(max(remaining / tt, 0.0))
            add_candidate(particular + root * tangent)
            add_candidate(particular - root * tangent)

    for i in range(len(coordinate_rows)):
        for j in range(i + 1, len(coordinate_rows)):
            a1, b1 = coordinate_rows[i]
            a2, b2 = coordinate_rows[j]
            mat = np.vstack([a1, a2])
            det = float(np.linalg.det(mat))
            if abs(det) <= 1.0e-14:
                continue
            rhs = np.array([b1, b2], dtype=np.float64)
            add_candidate(np.linalg.solve(mat, rhs))

    if not candidates:
        return origin, 0.0, "NO_FEASIBLE_CANDIDATE", False

    best_coordinate = min(candidates, key=model)
    best_model = model(best_coordinate)
    hit_boundary = (
        abs(float(np.linalg.norm(best_coordinate)) - radius)
        <= 1.0e-8 * max(radius, 1.0)
    )
    best_increment = np.linalg.solve(coordinate_map, best_coordinate)
    return best_increment, best_model, "OK", hit_boundary


def threshold_merit_rel(
        metrics: BandMetrics,
        _args: argparse.Namespace | None = None,
) -> float:
    """Return the target-free soft Jaccard loss.

    For fixed target area ``A``, soft intersection is ``A-M`` and soft union
    is ``A+L``.  Hence minimizing ``(L+M)/(A+L)`` is exactly equivalent to
    maximizing the corresponding soft Jaccard score.  No unattainable target
    value or relative weighting is introduced.
    """
    denominator = max(float(metrics.target_area) + float(metrics.leakage), 1.0e-30)
    return (float(metrics.leakage) + float(metrics.missing)) / denominator


def threshold_objective_gradient(
        metrics: BandMetrics,
        gradient: ReducedGradient,
        _args: argparse.Namespace | None = None,
) -> np.ndarray:
    """Return the exact reduced gradient of the soft Jaccard loss."""
    target_area = float(metrics.target_area)
    leakage = float(metrics.leakage)
    missing = float(metrics.missing)
    denominator = max(target_area + leakage, 1.0e-30)
    return (
        (target_area - missing) * np.asarray(gradient.grad_l, dtype=np.float64)
        + (target_area + leakage) * np.asarray(gradient.grad_m, dtype=np.float64)
    ) / (denominator * denominator)


def choose_parameter_step(
        *,
        c1: float,
        c2: float,
        c_min: float,
        c_max: float,
        min_width: float,
        metrics: BandMetrics,
        gradient: ReducedGradient,
        trust_metric: ThresholdTrustMetric,
        trust_radius: float,
        args: argparse.Namespace,
) -> ParameterStep:
    """Choose a constrained trust-region step for the soft Jaccard loss.

    Args:
        c1: Current lower threshold.
        c2: Current upper threshold.
        c_min: Lower search bound.
        c_max: Upper search bound.
        min_width: Minimum admissible threshold width.
        metrics: Current soft and certified band metrics.
        gradient: Current reduced gradients of leakage and missing area.
        trust_metric: Combined Dikin and relative state-pullback metric.
        trust_radius: Current dimensionless metric trust-region radius.
        args: Parsed command-line namespace for QP parameters.

    Returns:
        ``ParameterStep`` describing the proposed increment and model quality.
    """
    objective_name = "soft_jaccard_loss"
    g = threshold_objective_gradient(metrics, gradient, args)
    rows = constraint_rows(
        c1=c1,
        c2=c2,
        c_min=c_min,
        c_max=c_max,
        min_width=min_width,
        include_leakage=False,
        leakage=metrics.leakage,
        leakage_limit=0.0,
        grad_l=gradient.grad_l,
    )
    grad_norm = quadratic_metric_dual_norm(g, trust_metric.combined)
    if grad_norm <= 0.0:
        return ParameterStep(
            dc=np.zeros(2, dtype=np.float64),
            objective_name=objective_name,
            predicted_reduction=0.0,
            model_change=0.0,
            hessian_scale=0.0,
            step_norm=0.0,
            metric_norm=0.0,
            dikin_norm=0.0,
            pullback_norm=0.0,
            hit_boundary=False,
            status="ZERO_GRADIENT",
        )
    hessian_scale = max(
        grad_norm / max(float(trust_radius), 1.0e-30),
        float(args.qp_hessian_floor),
    )
    dc, model_change, status, hit_boundary = solve_trust_region_qp(
        g=g,
        rows=rows,
        radius=trust_radius,
        hessian_scale=hessian_scale,
        metric=trust_metric.combined,
    )
    return ParameterStep(
        dc=dc,
        objective_name=objective_name,
        predicted_reduction=max(-float(model_change), 0.0),
        model_change=float(model_change),
        hessian_scale=hessian_scale,
        step_norm=float(np.linalg.norm(dc)),
        metric_norm=quadratic_metric_norm(dc, trust_metric.combined),
        dikin_norm=quadratic_metric_norm(dc, trust_metric.dikin),
        pullback_norm=quadratic_metric_norm(dc, trust_metric.pullback),
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


def evaluate_energy_primer_guard(
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
        minimum_branch_overlap: float,
        minimum_activity_area: float,
) -> EnergyPrimerGuardResult:
    """Reject an energy-decreasing primer step that leaves the selected branch."""
    branch_overlap = branch_overlap_ratio(
        comm=comm,
        activity_ref=activity_ref,
        u_trial=u_trial,
        dx=dx,
        c1_const=c1_const,
        c2_const=c2_const,
        eps_const=eps_const,
        c1_trial=c1_trial,
        c2_trial=c2_trial,
        eps_trial=eps_trial,
    )
    activity_area = assemble_scalar(
        comm,
        window_activity_const_ufl(
            u_trial, c1_const, c2_const, eps_const
        ) * dx,
    )
    if activity_area < float(minimum_activity_area):
        return EnergyPrimerGuardResult(
            False, branch_overlap, activity_area, "ACTIVITY_COLLAPSE"
        )
    if branch_overlap < float(minimum_branch_overlap):
        return EnergyPrimerGuardResult(
            False, branch_overlap, activity_area, "BRANCH_LOSS"
        )
    return EnergyPrimerGuardResult(
        True, branch_overlap, activity_area, "OK"
    )




def measure_picard_spectrum(
        *,
        u: fem.Function,
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
        residual: float,
        args: argparse.Namespace,
        stage: str,
        outer_iteration: int,
        record_callback: Callable[[PicardSpectrumRecord], None] | None = None,
) -> PicardSpectrumRecord:
    """Measure the two extremal stiffness-relative Picard eigenvalues."""
    comm = u.function_space.mesh.comm
    started = time.perf_counter()

    def unavailable(status: str) -> PicardSpectrumRecord:
        record = PicardSpectrumRecord(
            stage=stage,
            outer_iteration=outer_iteration,
            c1=c1,
            c2=c2,
            eps_phi=eps_phi,
            residual=residual,
            mu_min=math.nan,
            mu_max=math.nan,
            error_min=math.nan,
            error_max=math.nan,
            iterations_min=0,
            iterations_max=0,
            converged_min=0,
            converged_max=0,
            spectral_radius_bound=math.nan,
            energy_minimum=-1,
            picard_contracting=-1,
            status=status,
            elapsed=time.perf_counter() - started,
        )
        if record_callback is not None:
            record_callback(record)
        root_print(
            comm,
            f"PICARD_SPECTRUM stage={stage} k={outer_iteration} "
            f"status={status}",
        )
        return record

    try:
        from slepc4py import SLEPc
    except ImportError:
        return unavailable("SLEPC_UNAVAILABLE")

    synchronize_threshold_constants(
        c1_const,
        c2_const,
        eps_const,
        c1=c1,
        c2=c2,
        eps_phi=eps_phi,
    )
    stiffness_matrix = None
    reaction = None
    free_index_set = None
    stiffness_reduced = None
    reaction_reduced = None
    try:
        stiffness_matrix = fem_petsc.assemble_matrix(
            fem.form(stiffness_form), bcs=[bc]
        )
        stiffness_matrix.assemble()
        reaction_expr = (
            float(rho_amp)
            * window_s_derivative_activity_ufl(
                u, c1_const, c2_const, eps_const
            )
            * trial * test
        ) * dx
        reaction = fem_petsc.assemble_matrix(
            fem.form(reaction_expr), bcs=[]
        )
        reaction.assemble()
        symmetry_tolerance = max(
            1.0e-13, 0.1 * float(args.picard_spectrum_eig_tol)
        )
        if not bool(reaction.isSymmetric(tol=symmetry_tolerance)):
            return unavailable("NONSYMMETRIC_REACTION")
        index_map = u.function_space.dofmap.index_map
        if int(u.function_space.dofmap.index_map_bs) != 1:
            return unavailable("UNSUPPORTED_BLOCK_SIZE")
        boundary_dofs, _ = bc.dof_indices()
        owned_boundary = np.asarray(boundary_dofs, dtype=np.int32)
        owned_boundary = owned_boundary[
            owned_boundary < int(index_map.size_local)
        ]
        free_mask = np.ones(int(index_map.size_local), dtype=bool)
        free_mask[owned_boundary] = False
        free_local = np.flatnonzero(free_mask).astype(np.int32)
        free_count = int(comm.allreduce(free_local.size, op=MPI.SUM))
        if free_count < 1:
            return unavailable("NO_FREE_DOFS")
        free_global = np.asarray(
            index_map.local_to_global(free_local), dtype=PETSc.IntType
        )
        free_index_set = PETSc.IS().createGeneral(free_global, comm=comm)
        stiffness_reduced = stiffness_matrix.createSubMatrix(
            free_index_set, free_index_set
        )
        stiffness_matrix.destroy()
        stiffness_matrix = None
        reaction_reduced = reaction.createSubMatrix(
            free_index_set, free_index_set
        )
        reaction.destroy()
        reaction = None
        free_index_set.destroy()
        free_index_set = None
        stiffness_reduced.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        reaction_reduced.setOption(PETSc.Mat.Option.SYMMETRIC, True)

        def solve_extreme(which, label: str) -> tuple[float, float, int, int]:
            eps = SLEPc.EPS().create(comm)
            try:
                options_prefix = (
                    f"picard_spectrum_{stage.lower()}_"
                    f"{abs(int(outer_iteration))}_{label}_"
                )
                eps.setOptionsPrefix(options_prefix)
                eps.setOperators(reaction_reduced, stiffness_reduced)
                eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
                eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
                eps.setWhichEigenpairs(which)
                eps.setDimensions(1)
                eps.setTolerances(
                    tol=float(args.picard_spectrum_eig_tol),
                    max_it=int(args.picard_spectrum_eig_max_it),
                )
                options = PETSc.Options()
                for key, value in solver_options(
                    args.linear_solver,
                    ksp_type=getattr(args, "ksp_type", None),
                ).items():
                    options[f"{options_prefix}st_{key}"] = value
                eps.setFromOptions()
                try:
                    eps.solve()
                except (PETSc.Error, RuntimeError, SystemError) as error:
                    root_print(
                        comm,
                        f"PICARD_SPECTRUM_EIGEN_FAIL stage={stage} "
                        f"k={outer_iteration} which={which} error={error}",
                    )
                    return math.nan, math.nan, 0, 0
                iterations = int(eps.getIterationNumber())
                converged = int(eps.getConverged())
                if converged < 1:
                    return math.nan, math.nan, iterations, converged
                eigenvalue = float(np.real(eps.getEigenvalue(0)))
                error = float(
                    eps.computeError(0, SLEPc.EPS.ErrorType.ABSOLUTE)
                )
                return eigenvalue, error, iterations, converged
            finally:
                eps.destroy()

        mu_min, error_min, iterations_min, converged_min = solve_extreme(
            SLEPc.EPS.Which.SMALLEST_REAL, "min"
        )
        mu_max, error_max, iterations_max, converged_max = solve_extreme(
            SLEPc.EPS.Which.LARGEST_REAL, "max"
        )
        (
            status,
            energy_minimum,
            picard_contracting,
            spectral_radius_bound,
        ) = classify_picard_spectrum(
            mu_min,
            mu_max,
            error_min=error_min,
            error_max=error_max,
        )
        record = PicardSpectrumRecord(
            stage=stage,
            outer_iteration=outer_iteration,
            c1=c1,
            c2=c2,
            eps_phi=eps_phi,
            residual=residual,
            mu_min=mu_min,
            mu_max=mu_max,
            error_min=error_min,
            error_max=error_max,
            iterations_min=iterations_min,
            iterations_max=iterations_max,
            converged_min=converged_min,
            converged_max=converged_max,
            spectral_radius_bound=spectral_radius_bound,
            energy_minimum=(
                -1 if energy_minimum is None else int(energy_minimum)
            ),
            picard_contracting=(
                -1 if picard_contracting is None else int(picard_contracting)
            ),
            status=status,
            elapsed=time.perf_counter() - started,
        )
        if record_callback is not None:
            record_callback(record)
        root_print(
            comm,
            f"PICARD_SPECTRUM stage={stage} k={outer_iteration} "
            f"status={status} muMin={mu_min:.12e} muMax={mu_max:.12e} "
            f"radiusBound={spectral_radius_bound:.12e} "
            f"energyMinimum={record.energy_minimum} "
            f"contracting={record.picard_contracting} "
            f"eigError=({error_min:.3e},{error_max:.3e}) "
            f"eigIts=({iterations_min},{iterations_max}) "
            f"time={record.elapsed:.6f}s",
        )
        return record
    finally:
        if reaction_reduced is not None:
            reaction_reduced.destroy()
        if stiffness_reduced is not None:
            stiffness_reduced.destroy()
        if free_index_set is not None:
            free_index_set.destroy()
        if reaction is not None:
            reaction.destroy()
        if stiffness_matrix is not None:
            stiffness_matrix.destroy()


def run_strategy(args: argparse.Namespace) -> int:
    """Execute the complete reduced-space optimization workflow.

    The function owns the end-to-end run:

    1. Validate arguments and create run-mpi2.log directories.
    2. Generate or load the star-shaped mesh.
    3. Build the finite-element space and homogeneous Dirichlet boundary
       condition.
    4. Solve the torsion problem and construct the crisp torsion band.
    5. Build the smoothed torsion-designed density and Poisson target
       potential.
    6. Initialize potential thresholds from either the fitted/quantile/area
       ensemble or the single direct fractional target-potential seed, then
       Newton-project the selected branch.
    7. Iterate the reduced algorithm: adaptive Newton projection, soft-Jaccard
       loss evaluation, sensitivity solves, reduced-gradient formation,
       two-variable trust-region update, sensitivity prediction, Newton
       correction, and accept/reject filtering.
    8. Always run a final exact Newton projection to
       ``--final-newton-tol-res`` when supplied, otherwise ``--tol-res``,
       before reporting final metrics.

    The implementation deliberately keeps the outer stopping reason and final
    Newton feasibility separate.  It does not declare geometric convergence
    from requested leakage, missing-area, or Jaccard values.  The script exits
    with a nonzero code only if the final exact Newton projection cannot reach
    its requested residual tolerance.

    Args:
        args: Parsed and already syntactically valid command-line namespace.

    Returns:
        Process-style return code.  ``0`` means the final Newton projection
        reached the requested final residual tolerance; ``3`` means it did
        not converge.
    """
    validate_args(args)
    stiffness_args = phase_solver_args(args, "stiffness")
    homotopy_args = phase_solver_args(args, "homotopy")
    nonlinear_args = phase_solver_args(args, "nonlinear")
    sensitivity_args = phase_solver_args(args, "sensitivity")
    final_solver_args = phase_solver_args(args, "final")
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
    homotopy_csv = log_dir / "homotopy.csv"
    frozen_frontier_csv = log_dir / "frozen_frontier.csv"
    energy_primer_csv = log_dir / "energy_primer.csv"
    picard_spectrum_csv = log_dir / "picard_spectrum.csv"
    search_space_csv = log_dir / "search_space.csv"
    sensitivity_check_csv = log_dir / "sensitivity_check.csv"
    summary_path = out_dir / "summary.txt"
    equilibrium_path = (
        args.equilibrium_output.expanduser().resolve()
        if args.equilibrium_output is not None
        else (out_dir / "equilibrium.npz").resolve()
    )
    total_start = time.perf_counter()

    root_print(comm, "========== START STRATEGY A DOLFINX WINDOW REDUCED OPTIMIZATION ==========")
    root_print(comm, f"RUN_TAG {run_tag}")
    root_print(comm, f"RUN_DIR {run_dir}")
    root_print(comm, f"OPT_CSV {opt_csv}")
    root_print(comm, f"FRAME_CSV {frame_csv}")
    root_print(
        comm,
        f"HOMOTOPY_CSV "
        f"{homotopy_csv if args.initial_projection_mode == 'homotopy' else 'disabled'}",
    )
    root_print(
        comm,
        "FROZEN_FRONTIER_CSV "
        f"{frozen_frontier_csv if args.initial_threshold_mode == 'frozen-frontier' else 'disabled'}",
    )
    root_print(
        comm,
        f"ENERGY_PRIMER_CSV {energy_primer_csv if args.energy_primer else 'disabled'}",
    )
    root_print(
        comm,
        f"PICARD_SPECTRUM_CSV {picard_spectrum_csv if args.picard_spectrum else 'disabled'}",
    )
    root_print(
        comm,
        f"SEARCH_SPACE_CSV "
        f"{search_space_csv if args.search_space_diagnostics else 'disabled'}",
    )
    root_print(
        comm,
        f"SENSITIVITY_CHECK_CSV "
        f"{sensitivity_check_csv if args.verify_sensitivities else 'disabled'}",
    )
    root_print(comm, f"SUMMARY {summary_path}")
    root_print(
        comm,
        f"TERMINAL_LOG {terminal_log_path if args.save_terminal_log else 'disabled'}",
    )
    root_print(
        comm,
        "INITIALIZATION_POLICY "
        f"mode={args.initial_threshold_mode} includeFit={int(args.include_fit_init)} "
        f"initialAlpha1={args.initial_alpha1} initialAlpha2={args.initial_alpha2} "
        f"projection={args.initial_projection_mode} "
        "requireDirectSeedInteriorLevelCurves="
        f"{int(args.require_direct_seed_interior_level_curves)} "
        f"homotopyTol={homotopy_tolerance(args):.3e} "
        f"homotopyInitialStep={args.homotopy_initial_step:.3e} "
        f"homotopyMinStep={args.homotopy_min_step:.3e} "
        f"homotopyMaxStep={args.homotopy_max_step:.3e} "
        f"homotopyShrink={args.homotopy_step_shrink:.3e} "
        f"homotopyGrow={args.homotopy_step_grow:.3e} "
        f"homotopyPredictor={int(args.homotopy_predictor)} "
        f"frozenFrontierBins={args.frozen_frontier_bins} "
        f"frozenLeakageCapRel={args.frozen_leakage_cap_rel}",
    )
    root_print(
        comm,
        "REDUCED_OBJECTIVE "
        "objective=soft_jaccard_loss targetTolerances=none "
        f"epsMode={args.eps_mode} epsRatio={args.eps_ratio:.6e} epsPhi={args.eps_phi} "
        f"kappa={args.kappa:.6e} deltaC={args.delta_c:.6e} "
        f"tolRes={args.tol_res:.6e} finalNewtonTolRes={final_tol_res:.6e} "
        f"innerNewtonTol={args.inner_newton_tol} "
        f"requireInnerNewton={int(args.require_inner_newton_convergence)} "
        f"innerTolMax={args.inner_tol_max:.6e} "
        f"innerTolGamma={args.inner_tol_gamma:.6e} "
        f"plotSevere={int(args.plot_severe)} "
        f"plotInitialCandidates={int(args.plot_initial_candidates)} "
        f"plotAcceptedStates={int(args.plot_accepted_states)} "
        f"plotDesignHoldSeconds={args.plot_design_hold_seconds:.3f}",
    )
    root_print(
        comm,
        "JACCARD_OSCILLATION_STOP "
        f"enabled={int(args.jaccard_oscillation_stop)} "
        f"minAccepted={args.jaccard_oscillation_min_accepted} "
        f"patience={args.jaccard_oscillation_patience} "
        f"stagnationEnabled={int(args.jaccard_stagnation_stop)} "
        "comparison=strict_new_best",
    )
    root_print(
        comm,
        "NONLINEAR_GLOBALIZATION "
        f"energyPrimer={int(args.energy_primer)} "
        f"primerTol={args.energy_primer_tol:.3e} "
        f"primerMaxSteps={args.energy_primer_max_it} "
        f"primerMinResidualReduction={args.energy_primer_min_residual_reduction:.3e} "
        "primerRescueOnly=1 "
        f"branchGuard={int(not args.disable_branch_check)} "
        f"activityGuardFraction={args.min_activity_fraction:.3e} "
        f"plotPrimerSteps={int(design_output_enabled(args))} "
        f"stallForecast={int(args.newton_stall_forecast)} "
        f"stallWindow={args.newton_stall_window} "
        f"stallPatience={args.newton_stall_patience}",
    )
    root_print(
        comm,
        "PICARD_SPECTRUM_POLICY "
        f"enabled={int(args.picard_spectrum)} eigTol={args.picard_spectrum_eig_tol:.3e} "
        f"eigMaxIt={args.picard_spectrum_eig_max_it} stages=INITIAL,FINAL",
    )
    root_print(
        comm,
        "THRESHOLD_TRUST_METRIC "
        "mode=simplex_dikin_plus_relative_H1_pullback "
        f"radius={args.trust_radius:.6e} "
        f"radiusMin={args.trust_radius_min:.6e} "
        f"radiusMax={args.trust_radius_max:.6e} "
        "simplexInteriorGuard=1 fittedGeometryParameters=0",
    )
    root_print(
        comm,
        "SEARCH_SPACE_DIAGNOSTICS "
        f"recordingEnabled={int(args.search_space_diagnostics)} "
        "coordinates=normalized_threshold_simplex "
        "stateMetric=H1_sensitivity_pullback predictorDefect=Hminus1 "
        "trustMetricAlwaysActive=1",
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
        f"fallback={args.iterative_fallback_solver}",
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
    energy_gradient = fem.Function(V, name="energyPrimerGradient")
    predictor_dual = fem.Function(V, name="thresholdPredictorDualResidual")
    homotopy_tangent = fem.Function(V, name="homotopyTangent")
    rho = fem.Function(V, name="rho")
    s1 = fem.Function(V, name="sensitivityC1")
    s2 = fem.Function(V, name="sensitivityC2")
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
    homotopy_fields = [
        "record", "runTag", "candidate", "c1", "c2", "epsPhi", "attempt",
        "lambdaOld", "lambdaTrial",
        "deltaLambda", "tangentH1", "predictedResidual",
        "primerStatus", "primerSteps", "primerGradientNorm",
        "primerEnergyInitial", "primerEnergyFinal",
        "rescueTriggered", "initialNewtonStatus", "initialNewtonIterations",
        "retryNewtonStatus", "retryNewtonIterations", "totalNewtonIterations",
        "correctedResidual", "newtonStatus", "newtonIterations", "damping",
        "backtracks", "accepted", "elapsed",
    ]
    homotopy_handle = (
        homotopy_csv.open("w", newline="", encoding="utf-8")
        if comm.rank == 0 and args.initial_projection_mode == "homotopy"
        else None
    )
    homotopy_writer = (
        csv.DictWriter(homotopy_handle, fieldnames=homotopy_fields)
        if homotopy_handle is not None
        else None
    )
    if homotopy_writer is not None:
        homotopy_writer.writeheader()
    energy_primer_fields = [
        "record", "runTag", "context", "prefix", "outerIteration",
        "homotopyLambda", "primerIteration", "energyBefore", "energyAfter",
        "gradientNormHminus1Before", "gradientNormHminus1After",
        "residualRatio", "branchOverlap", "activityArea",
        "alpha", "backtracks", "status", "computeTime", "plotTime", "elapsed",
    ]
    energy_primer_handle = (
        energy_primer_csv.open("w", newline="", encoding="utf-8")
        if comm.rank == 0 and args.energy_primer
        else None
    )
    energy_primer_writer = (
        csv.DictWriter(energy_primer_handle, fieldnames=energy_primer_fields)
        if energy_primer_handle is not None
        else None
    )
    if energy_primer_writer is not None:
        energy_primer_writer.writeheader()

    picard_spectrum_fields = [
        "record", "runTag", "stage", "outerIteration", "c1", "c2",
        "epsPhi", "residual", "muMin", "muMax", "errorMin", "errorMax",
        "iterationsMin", "iterationsMax", "convergedMin", "convergedMax",
        "spectralRadiusBound", "energyMinimum", "picardContracting",
        "status", "elapsed",
    ]
    picard_spectrum_handle = (
        picard_spectrum_csv.open("w", newline="", encoding="utf-8")
        if comm.rank == 0 and args.picard_spectrum
        else None
    )
    picard_spectrum_writer = (
        csv.DictWriter(
            picard_spectrum_handle, fieldnames=picard_spectrum_fields
        )
        if picard_spectrum_handle is not None
        else None
    )
    if picard_spectrum_writer is not None:
        picard_spectrum_writer.writeheader()
    picard_spectrum_records: list[PicardSpectrumRecord] = []

    search_space_handle = (
        search_space_csv.open("w", newline="", encoding="utf-8")
        if comm.rank == 0 and args.search_space_diagnostics
        else None
    )
    search_space_writer = (
        csv.DictWriter(search_space_handle, fieldnames=SEARCH_SPACE_FIELDS)
        if search_space_handle is not None
        else None
    )
    if search_space_writer is not None:
        search_space_writer.writeheader()

    def record_picard_spectrum(record: PicardSpectrumRecord) -> None:
        """Retain and persist one synchronized extremal-spectrum audit."""
        picard_spectrum_records.append(record)
        if picard_spectrum_writer is None:
            return
        picard_spectrum_writer.writerow({
            "record": "spectrum",
            "runTag": run_tag,
            "stage": record.stage,
            "outerIteration": record.outer_iteration,
            "c1": record.c1,
            "c2": record.c2,
            "epsPhi": record.eps_phi,
            "residual": record.residual,
            "muMin": record.mu_min,
            "muMax": record.mu_max,
            "errorMin": record.error_min,
            "errorMax": record.error_max,
            "iterationsMin": record.iterations_min,
            "iterationsMax": record.iterations_max,
            "convergedMin": record.converged_min,
            "convergedMax": record.converged_max,
            "spectralRadiusBound": record.spectral_radius_bound,
            "energyMinimum": record.energy_minimum,
            "picardContracting": record.picard_contracting,
            "status": record.status,
            "elapsed": record.elapsed,
        })
        picard_spectrum_handle.flush()

    def record_energy_primer(record: EnergyPrimerRecord) -> None:
        """Append one synchronized primer event on rank zero."""
        if energy_primer_writer is None:
            return
        energy_primer_writer.writerow({
            "record": record.record,
            "runTag": run_tag,
            "context": record.context,
            "prefix": record.prefix,
            "outerIteration": record.outer_iteration,
            "homotopyLambda": record.homotopy_lambda,
            "primerIteration": record.primer_iteration,
            "energyBefore": record.energy_before,
            "energyAfter": record.energy_after,
            "gradientNormHminus1Before": record.gradient_norm,
            "gradientNormHminus1After": record.gradient_norm_after,
            "residualRatio": record.residual_ratio,
            "branchOverlap": record.branch_overlap,
            "activityArea": record.activity_area,
            "alpha": record.alpha,
            "backtracks": record.backtracks,
            "status": record.status,
            "elapsed": record.elapsed,
            "computeTime": max(record.elapsed - record.plot_time, 0.0),
            "plotTime": record.plot_time,
        })
        energy_primer_handle.flush()

    energy_primer_statistics = EnergyPrimerStatistics()
    stiffness_metric_solver = (
        PersistentStiffnessSolver(
            stiffness_form,
            bc,
            energy_gradient,
            prefix="energy_primer_stiffness_",
            solver=stiffness_args.linear_solver,
            ksp_type=stiffness_args.ksp_type,
            rtol=stiffness_args.linear_rtol,
            atol=stiffness_args.linear_atol,
            max_it=stiffness_args.linear_max_it,
        )
        if args.energy_primer or args.search_space_diagnostics
        else None
    )
    energy_primer_solver = stiffness_metric_solver if args.energy_primer else None
    predictor_metric_solver = (
        stiffness_metric_solver if args.search_space_diagnostics else None
    )
    plotter = MPIPyVistaTorsionPlotter(
        args,
        run_tag=run_tag,
        run_dir=run_dir,
        frame_writer=frame_writer,
        comm=comm,
    )

    its, rel, solve_time = solve_linear_form(
        stiffness_form,
        1.0 * test * dx,
        T,
        [bc],
        prefix="torsion_",
        solver=stiffness_args.linear_solver,
        ksp_type=stiffness_args.ksp_type,
        rtol=stiffness_args.linear_rtol,
        atol=stiffness_args.linear_atol,
        max_it=stiffness_args.linear_max_it,
        verbosity=args.verbosity,
    )
    _, tmax = global_minmax(comm, T)
    c1_t = params.alpha_t1 * tmax
    c2_t = params.alpha_t2 * tmax
    eps_t = params.eps_t_ratio * (c2_t - c1_t)
    tau_mask = ufl.conditional(ufl.gt(T, c1_t), ufl.conditional(ufl.lt(T, c2_t), 1.0, 0.0), 0.0)
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

    if params.eps_t_ratio == 0.0:
        target_density_mode = "sharp_indicator"
        target_density_expr = float(params.rho_amp) * tau_mask
    else:
        target_density_mode = "smoothed_window"
        target_density_expr = window_ufl(T, c1_t, c2_t, eps_t, params.rho_amp)
    update_interpolated(rho_design, target_density_expr)
    rho_design_l2 = math.sqrt(max(assemble_scalar(comm, rho_design * rho_design * dx), 0.0))
    rho_design_mass = assemble_scalar(comm, rho_design * dx)
    _, rho_design_max = global_minmax(comm, rho_design)

    its, rel, solve_time = solve_linear_form(
        stiffness_form,
        rho_design * test * dx,
        phi_target,
        [bc],
        prefix="phi_target_",
        solver=stiffness_args.linear_solver,
        ksp_type=stiffness_args.ksp_type,
        rtol=stiffness_args.linear_rtol,
        atol=stiffness_args.linear_atol,
        max_it=stiffness_args.linear_max_it,
        verbosity=args.verbosity,
    )
    _, phi_target_max = global_minmax(comm, phi_target)
    phi_target_l2 = math.sqrt(max(assemble_scalar(comm, phi_target * phi_target * dx), 0.0))
    if args.threshold_cap_mode == "torsion":
        c_search_max = max(float(tmax), 1.0e-12)
    else:
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
        f"targetDensityMode={target_density_mode}",
    )
    root_print(
        comm,
        f"SEARCH_DOMAIN capMode={args.threshold_cap_mode} "
        f"cMin={c_min:.6e} cMax={c_upper:.6e} minWidth={min_width:.6e}",
    )

    init_candidate_name = "manual"
    init_candidate_score = math.nan
    init_candidates: list[InitialWindowCandidate] = []
    frozen_frontier_initialization: FrozenFrontierInitialization | None = None
    if args.c1_phi is not None:
        c1_phi = float(args.c1_phi)
        c2_phi = float(args.c2_phi)
        fit_result = None
    elif args.initial_threshold_mode == "frozen-frontier":
        fit_result = None
        frontier_candidate, frozen_frontier_initialization = (
            build_frozen_frontier_initial_candidate(
                phi_target=phi_target,
                torsion=T,
                c1_t=c1_t,
                c2_t=c2_t,
                target_area=target_area,
                quadrature_degree=int(qdeg),
                c_min=c_min,
                c_max=c_upper,
                min_width=min_width,
                frontier_csv=frozen_frontier_csv,
                run_tag=run_tag,
                args=args,
            )
        )
        init_candidates = [frontier_candidate]
        c1_phi = frontier_candidate.c1
        c2_phi = frontier_candidate.c2
        init_candidate_name = frontier_candidate.name
        init_candidate_score = frontier_candidate.score
    elif args.initial_threshold_mode == "torsion-fraction-phi-target":
        fit_result = None
        direct_alpha1 = (
            float(params.alpha_t1)
            if args.initial_alpha1 is None
            else float(args.initial_alpha1)
        )
        direct_alpha2 = (
            float(params.alpha_t2)
            if args.initial_alpha2 is None
            else float(args.initial_alpha2)
        )
        c1_direct, c2_direct = torsion_fraction_phi_target_thresholds(
            phi_target_max=phi_target_max,
            alpha1=direct_alpha1,
            alpha2=direct_alpha2,
            c_min=c_min,
            c_max=c_upper,
            min_width=min_width,
        )
        direct_quad_degree = (
            args.fit_window_quad_degree
            if args.fit_window_quad_degree is not None
            else qdeg
        )
        phi_values, rho_values, weights = quadrature_samples_for_fit(
            phi_target,
            rho_design,
            quadrature_degree=int(direct_quad_degree),
        )
        direct_candidate = initial_candidate_from_thresholds(
            comm=comm,
            name="torsion_fraction_phi_target",
            c1=c1_direct,
            c2=c2_direct,
            phi_values=phi_values,
            rho_values=rho_values,
            weights=weights,
            rho_design_l2=rho_design_l2,
            rho_amp=params.rho_amp,
            target_area=target_area,
            c_min=c_min,
            c_max=c_upper,
            min_width=min_width,
            args=args,
        )
        init_candidates = [direct_candidate]
        c1_phi = direct_candidate.c1
        c2_phi = direct_candidate.c2
        init_candidate_name = direct_candidate.name
        init_candidate_score = direct_candidate.score
        root_print(
            comm,
            f"DIRECT_INIT alpha1={direct_alpha1:.6e} alpha2={direct_alpha2:.6e} "
            f"phiTargetMax={phi_target_max:.6e} c1={c1_phi:.6e} c2={c2_phi:.6e} "
            f"epsPhi={direct_candidate.eps_phi:.6e} "
            f"Lrel={direct_candidate.leakage_rel:.6e} "
            f"Mrel={direct_candidate.missing_rel:.6e} "
            f"activeJ={direct_candidate.active_jaccard:.6e}",
        )
    elif args.include_fit_init:
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
    else:
        center = c_min + 0.65 * c_scale
        width = max(0.15 * c_scale, min_width)
        c1_phi = center - 0.5 * width
        c2_phi = center + 0.5 * width
        init_candidate_name = "fallback_center_width"

    c1_phi, c2_phi = project_thresholds(c1_phi, c2_phi, c_min=c_min, c_max=c_upper, min_width=min_width)
    eps_phi = epsilon_from_thresholds(args, c1_phi, c2_phi)

    u.x.array[:] = phi_target.x.array
    u.x.scatter_forward()
    if args.initial_state is not None:
        loaded_dofs = load_global_initial_state(args.initial_state, u, comm)
        init_candidate_name = "manual_bruteforce_state"
        root_print(
            comm,
            f"INITIAL_STATE file={args.initial_state} globalDofs={loaded_dofs}",
        )
    c1_const = fem.Constant(domain, PETSc.ScalarType(c1_phi))
    c2_const = fem.Constant(domain, PETSc.ScalarType(c2_phi))
    eps_const = fem.Constant(domain, PETSc.ScalarType(eps_phi))
    predictor_residual_form = (
        fem.form(
            (
                ufl.inner(ufl.grad(u), ufl.grad(test))
                - window_density_const_ufl(
                    u,
                    c1_const,
                    c2_const,
                    eps_const,
                    params.rho_amp,
                ) * test
            ) * dx
        )
        if args.search_space_diagnostics
        else None
    )
    update_interpolated(rho, window_density_const_ufl(u, c1_const, c2_const, eps_const, params.rho_amp))

    if design_output_enabled(args):
        # This is the first density induced by the selected initial threshold
        # pair. At this point u is phi_target (the design potential), unless an
        # explicit initial state was loaded. Keep T hidden behind the visible
        # compact panels so the target-band contours remain overlaid.
        design_phi_name = (
            "phi_design" if args.initial_state is None else "phi_initial"
        )
        if args.plot_fields == "state":
            design_fields = [rho, u, T]
            design_titles = [
                f"initial rho({design_phi_name})",
                f"initial {design_phi_name}",
            ]
            design_contour_index = 2
        else:
            design_fields = [rho, T]
            design_titles = [f"initial rho({design_phi_name})"]
            design_contour_index = 1
        plotter.emit(
            design_fields,
            design_titles,
            stage="DESIGN",
            ieps=-1,
            k=-1,
            eps_phi=eps_phi,
            residual=0.0,
            metrics={"c1Phi": c1_phi, "c2Phi": c2_phi},
            token="design",
            save=True,
            show=True,
            nt=nt,
            ndof=ndof,
            contour_field_index=design_contour_index,
            contour_levels=(c1_t, c2_t),
        )
        plotter.hold_interactive(
            float(args.plot_design_hold_seconds),
            stage="DESIGN",
        )

    def project_initial_candidate(
            candidate: InitialWindowCandidate,
            index: int,
    ) -> ProjectedInitialCandidate:
        """Newton-project and score one automatic initial-window candidate.

        Each candidate is projected from the same target potential ``phi_target``
        so the comparison reflects the branch induced by its thresholds rather
        than any previous candidate's terminal state.  The returned state arrays
        are copies because later candidate projections reuse the same work
        functions.
        """
        c1_candidate, c2_candidate = project_thresholds(
            candidate.c1,
            candidate.c2,
            c_min=c_min,
            c_max=c_upper,
            min_width=min_width,
        )
        eps_candidate = epsilon_from_thresholds(args, c1_candidate, c2_candidate)


        def emit_initializer_state(
                *,
                stage: str,
                token: str,
                title: str,
                residual: float,
                eps_value: float,
        ) -> None:
            """Collectively show one initializer state with target-band contours."""
            if args.plot_fields == "density":
                display_fields = [rho, T]
                display_titles = [title]
                target_contour_index = 1
            elif args.plot_fields == "state":
                display_fields = [rho, u, T]
                display_titles = [f"rho(phi), {title}", f"phi, {title}"]
                target_contour_index = 2
            else:
                phi_diff.x.array[:] = u.x.array - phi_target.x.array
                phi_diff.x.scatter_forward()
                display_fields = [
                    T,
                    tau_band,
                    rho_design,
                    phi_target,
                    u,
                    rho,
                    phi_diff,
                ]
                display_titles = [
                    "torsion T",
                    "target band",
                    "target density",
                    "target potential",
                    "candidate potential",
                    title,
                    "potential mismatch",
                ]
                target_contour_index = 0
            plotter.emit(
                display_fields,
                display_titles,
                stage=stage,
                ieps=-1,
                k=index,
                eps_phi=eps_value,
                residual=residual,
                metrics={},
                token=token,
                save=bool(args.save_frames),
                show=True,
                nt=nt,
                ndof=ndof,
                contour_field_index=target_contour_index,
                contour_levels=(c1_t, c2_t),
            )

        def initial_newton_callback(
                prefix: str,
                newton_k: int,
                residual: float,
                alpha: float,
                backtracks: int,
                c1_value: float,
                c2_value: float,
                eps_value: float,
        ) -> None:
            """Show each line-search-accepted initializer Newton update."""
            emit_initializer_state(
                stage="NEWTON_INIT",
                token=(
                    f"init_{index}_{slug_for_path(candidate.name)}_"
                    f"newton_{newton_k:03d}"
                ),
                title=(
                    f"{candidate.name} Newton n={newton_k}; "
                    f"c=({c1_value:.3e},{c2_value:.3e}); "
                    f"a={alpha:.2e}, bt={backtracks}"
                ),
                residual=residual,
                eps_value=eps_value,
            )

        def initial_energy_plot_callback(record: EnergyPrimerRecord) -> None:
            """Plot every accepted initializer/homotopy descent step."""
            emit_initializer_state(
                stage="ENERGY_PRIMER",
                token=(
                    f"init_{index}_{slug_for_path(candidate.name)}_"
                    f"energy_lambda_{record.homotopy_lambda:.6f}_"
                    f"{record.primer_iteration:03d}"
                ).replace(".", "p"),
                title=(
                    f"ENERGY_PRIMER {candidate.name} "
                    f"lambda={record.homotopy_lambda:.3f} "
                    f"m={record.primer_iteration}; "
                    f"E={record.energy_after:.6e}, "
                    f"|g|H1={record.gradient_norm:.3e}, "
                    f"a={record.alpha:.2e}, bt={record.backtracks}"
                ),
                residual=record.gradient_norm,
                eps_value=eps_candidate,
            )

        def homotopy_stage_callback(
                accepted_stage: int,
                lambda_value: float,
                stage_newton: NewtonResult,
        ) -> None:
            """Show every fully corrected, accepted continuation stage."""
            if not design_output_enabled(args):
                return
            emit_initializer_state(
                stage="HOMOTOPY",
                token=(
                    f"init_{index}_{slug_for_path(candidate.name)}_"
                    f"homotopy_{accepted_stage:03d}_"
                    f"lambda_{lambda_value:.6f}"
                ).replace(".", "p"),
                title=(
                    f"homotopy stage {accepted_stage}, "
                    f"lambda={lambda_value:.3f}; "
                    f"c=({c1_candidate:.3e},{c2_candidate:.3e})"
                ),
                residual=stage_newton.residual,
                eps_value=eps_candidate,
            )

        u.x.array[:] = phi_target.x.array
        u.x.scatter_forward()
        projection_tol = (
            homotopy_tolerance(args)
            if args.initial_projection_mode == "homotopy"
            else inner_tolerance(
                args,
                (candidate.leakage_rel + candidate.missing_rel)
                / max(1.0 + candidate.leakage_rel, 1.0e-30),
            )
        )
        projection_start = time.perf_counter()
        homotopy_result: HomotopyResult | None = None
        if args.initial_projection_mode == "homotopy":
            homotopy_result = solve_source_homotopy_initialization(
                u=u,
                du=du,
                tangent=homotopy_tangent,
                rho=rho,
                target_density=rho_design,
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
                args=homotopy_args,
                prefix=f"init_{index}_{slug_for_path(candidate.name)}_homotopy",
                run_tag=run_tag,
                candidate_name=candidate.name,
                stage_callback=homotopy_stage_callback,
                stage_writer=homotopy_writer,
                stage_handle=homotopy_handle,
                energy_gradient=energy_gradient,
                energy_solver=energy_primer_solver,
                energy_record_callback=record_energy_primer,
                energy_plot_callback=(
                    initial_energy_plot_callback
                    if design_output_enabled(args)
                    else None
                ),
                energy_statistics=energy_primer_statistics,
                target_area=target_area,
                energy_activity_reference=activity_ref,
            )
            last_newton = homotopy_result.last_newton
            if homotopy_result.converged:
                newton = last_newton
            else:
                newton = NewtonResult(
                    status=f"HOMOTOPY_{homotopy_result.status}",
                    converged=False,
                    iterations=last_newton.iterations,
                    residual=last_newton.residual,
                    step_h1=last_newton.step_h1,
                    alpha=last_newton.alpha,
                    backtracks=last_newton.backtracks,
                    solve_time=last_newton.solve_time,
                    contraction=last_newton.contraction,
                    predicted_remaining=last_newton.predicted_remaining,
                    stall_windows=last_newton.stall_windows,
                    forecast_remaining_budget=(
                        last_newton.forecast_remaining_budget
                    ),
                )
            root_print(
                comm,
                f"HOMOTOPY_RESULT name={candidate.name} "
                f"status={homotopy_result.status} "
                f"converged={int(homotopy_result.converged)} "
                f"lambda={homotopy_result.lambda_final:.6e} "
                f"stages={homotopy_result.stages} "
                f"rejected={homotopy_result.rejected_steps} "
                f"newtonIts={homotopy_result.total_newton_iterations} "
                f"time={homotopy_result.elapsed:.6f}s",
            )
        else:
            synchronize_threshold_constants(
                c1_const,
                c2_const,
                eps_const,
                c1=c1_candidate,
                c2=c2_candidate,
                eps_phi=eps_candidate,
            )
            update_interpolated(
                activity_ref,
                window_activity_const_ufl(
                    u, c1_const, c2_const, eps_const
                ),
            )
            guard_callback = lambda: evaluate_energy_primer_guard(
                comm=comm,
                activity_ref=activity_ref,
                u_trial=u,
                dx=dx,
                c1_const=c1_const,
                c2_const=c2_const,
                eps_const=eps_const,
                c1_trial=c1_candidate,
                c2_trial=c2_candidate,
                eps_trial=eps_candidate,
                minimum_branch_overlap=(
                    0.0 if args.disable_branch_check else float(args.eta_overlap)
                ),
                minimum_activity_area=(
                    float(args.min_activity_fraction) * target_area
                ),
            )
            correction = solve_equilibrium_with_primer_rescue(
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
                context=f"initial:{candidate.name}",
                outer_iteration=-1,
                plot_callback=(
                    initial_newton_callback if args.plot_accepted_states else None
                ),
                energy_gradient=energy_gradient,
                energy_solver=energy_primer_solver,
                energy_record_callback=record_energy_primer,
                energy_plot_callback=(
                    initial_energy_plot_callback
                    if design_output_enabled(args)
                    else None
                ),
                energy_statistics=energy_primer_statistics,
                energy_guard_callback=guard_callback,
            )
            newton = correction.newton
        _, projected_phi_max = global_minmax(comm, u)
        interior_level_curves = bool(
            c1_candidate > 0.0 and c2_candidate < projected_phi_max
        )
        if (
            candidate.name == "torsion_fraction_phi_target"
            and newton.converged
            and not interior_level_curves
            and args.require_direct_seed_interior_level_curves
        ):
            raise RuntimeError(
                "direct seed converged but does not define two strictly "
                "interior equilibrium level curves: "
                f"c1={c1_candidate:.6e} c2={c2_candidate:.6e} "
                f"maxPhi={projected_phi_max:.6e}"
            )
        if (
            candidate.name == "torsion_fraction_phi_target"
            and newton.converged
            and not interior_level_curves
            and not args.require_direct_seed_interior_level_curves
        ):
            root_print(
                comm,
                "DIRECT_SEED_INTERIOR_CURVE_CHECK_BYPASSED "
                f"c1={c1_candidate:.6e} c2={c2_candidate:.6e} "
                f"maxPhi={projected_phi_max:.6e}",
            )
        projection_time = time.perf_counter() - projection_start
        metrics = evaluate_band_metrics(
            comm=comm,
            u=u,
            tau_mask=tau_mask,
            dx=dx,
            c1_const=c1_const,
            c2_const=c2_const,
            eps_const=eps_const,
            c1=c1_candidate,
            c2=c2_candidate,
            eps_phi=eps_candidate,
            kappa=args.kappa,
            target_area=target_area,
        )
        residual_form_for_projection = (
            ufl.inner(ufl.grad(u), ufl.grad(test))
            - window_density_const_ufl(u, c1_const, c2_const, eps_const, params.rho_amp) * test
        ) * dx
        update_interpolated(rho, window_density_const_ufl(u, c1_const, c2_const, eps_const, params.rho_amp))
        diagnostics = compute_metrics(
            u=u,
            rho=rho,
            rho_design=rho_design,
            rho_design_l2=rho_design_l2,
            residual_form=residual_form_for_projection,
            bc=bc,
            dx=dx,
            c2_phi=c2_candidate,
            active_threshold=params.active_threshold,
            plateau_threshold=params.plateau_threshold,
            rho_amp=params.rho_amp,
        )
        projected_score = projected_initial_candidate_score(
            metrics=metrics,
            diagnostics=diagnostics,
            target_area=target_area,
            converged=newton.converged,
        )
        area_rel = metrics.activity_area / max(target_area, 1.0e-30)
        root_print(
            comm,
            f"INIT_PROJECT name={candidate.name} c1={c1_candidate:.6e} c2={c2_candidate:.6e} "
            f"width={c2_candidate - c1_candidate:.6e} epsPhi={eps_candidate:.6e} "
            f"projection={args.initial_projection_mode} "
            f"tol={projection_tol:.3e} status={newton.status} converged={int(newton.converged)} "
            f"iters={newton.iterations} residual={newton.residual:.6e} time={projection_time:.3f} "
            f"score={projected_score:.6e} Lrel={metrics.leakage_rel:.6e} "
            f"Mrel={metrics.missing_rel:.6e} areaRel={area_rel:.6e} "
            f"activeJ={diagnostics['activeJaccard']:.6e} rhoRel={diagnostics['relRhoDesign']:.6e} "
            f"phiMax={projected_phi_max:.6e} interiorCurves={int(interior_level_curves)}",
        )
        if args.plot_initial_candidates:
            if newton.converged:
                emit_initializer_state(
                    stage="INIT_CANDIDATE",
                    token=f"init_{index}_{slug_for_path(candidate.name)}_projected",
                    title=(
                        f"{candidate.name} projected; "
                        f"c=({c1_candidate:.3e},{c2_candidate:.3e}); "
                        f"J={diagnostics['activeJaccard']:.3f}"
                    ),
                    residual=newton.residual,
                    eps_value=eps_candidate,
                )
                root_print(
                    comm,
                    f"INIT_CANDIDATE_PLOT name={candidate.name} "
                    f"status={newton.status} residual={newton.residual:.6e}",
                )
            else:
                root_print(
                    comm,
                    f"INIT_CANDIDATE_PLOT_SKIP name={candidate.name} "
                    f"status={newton.status} residual={newton.residual:.6e} "
                    "reason=NEWTON_NOT_FULLY_CONVERGED",
                )
        return ProjectedInitialCandidate(
            base=candidate,
            newton=newton,
            metrics=metrics,
            diagnostics=diagnostics,
            score=projected_score,
            state=u.x.array.copy(),
            density=rho.x.array.copy(),
            homotopy=homotopy_result,
        )

    selected_homotopy_result: HomotopyResult | None = None
    if init_candidates:
        projected_candidates = [
            project_initial_candidate(candidate, index)
            for index, candidate in enumerate(init_candidates)
        ]
        selected_projected = select_projected_initial_candidate(
            projected_candidates,
            require_converged=bool(args.require_inner_newton_convergence),
        )
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
            f"INIT_PROJECT_SELECT name={init_candidate_name} c1={c1_phi:.6e} c2={c2_phi:.6e} "
            f"score={init_candidate_score:.6e} residual={selected_projected.newton.residual:.6e}",
        )

    root_print(
        comm,
        f"WINDOW_INIT c1Phi={c1_phi:.6e} c2Phi={c2_phi:.6e} "
        f"width={c2_phi - c1_phi:.6e} epsPhi={eps_phi:.6e} "
        f"init={init_candidate_name} initScore={init_candidate_score:.6e} "
        f"projection={args.initial_projection_mode} "
        f"homotopyStatus="
        f"{selected_homotopy_result.status if selected_homotopy_result is not None else 'None'}",
    )

    def emit_severe_plot(
            *,
            stage: str,
            outer_k: int,
            token: str,
            eps_phi_value: float,
            residual: float,
            title_suffix: str,
            accepted_event: bool = False,
            force: bool = False,
    ) -> None:
        """Emit a live diagnostic plot for selected Newton/refit internals.

        The severe mode retains the complete predictor/accept/reject stream.
        The accepted-states mode is narrower: it emits only line-search
        accepted Newton states and accepted threshold-pair states.
        """
        if not force and not args.plot_severe and not (
            args.plot_accepted_states and accepted_event
        ):
            return
        phi_diff.x.array[:] = u.x.array - phi_target.x.array
        phi_diff.x.scatter_forward()
        if args.plot_fields == "density":
            display_fields = [rho, T]
            display_titles = [f"rho(phi) {title_suffix}"]
            target_contour_index = 1
        elif args.plot_fields == "state":
            display_fields = [rho, u, T]
            display_titles = [
                f"rho(phi) {title_suffix}",
                f"phi {title_suffix}",
            ]
            target_contour_index = 2
        else:
            display_fields = [T, tau_band, phi_target, u, rho, phi_diff]
            display_titles = [
                "Torsion T",
                "tau band",
                "phiT",
                f"phi {title_suffix}",
                f"rho(phi) {title_suffix}",
                "phi-phiT",
            ]
            target_contour_index = 0
        plotter.emit(
            display_fields,
            display_titles,
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
            contour_field_index=target_contour_index,
            contour_levels=(c1_t, c2_t),
        )

    def make_newton_plot_callback(
            *,
            outer_k: int,
            stage: str,
    ) -> Callable[[str, int, float, float, int, float, float, float], None] | None:
        """Build the plotting hook called after each accepted Newton update."""
        if not args.plot_severe and not args.plot_accepted_states:
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
            """Emit the current line-search-accepted Newton state."""
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
                accepted_event=True,
            )

        return callback

    def make_energy_primer_plot_callback(
            *,
            outer_k: int,
            c1_value: float,
            c2_value: float,
            eps_phi_value: float,
    ) -> Callable[[EnergyPrimerRecord], None] | None:
        """Build the mandatory accepted energy-descent plot hook."""
        if not design_output_enabled(args):
            return None

        def callback(record: EnergyPrimerRecord) -> None:
            emit_severe_plot(
                stage="ENERGY_PRIMER",
                outer_k=outer_k,
                token=(
                    f"energy_primer_outer_{outer_k:03d}_"
                    f"{record.primer_iteration:03d}"
                ),
                eps_phi_value=eps_phi_value,
                residual=record.gradient_norm_after,
                title_suffix=(
                    f"ENERGY_PRIMER m={record.primer_iteration} "
                    f"c=({c1_value:.3e},{c2_value:.3e}) "
                    f"E={record.energy_after:.6e} "
                    f"|R|H^-1={record.gradient_norm_after:.3e} "
                    f"a={record.alpha:.2e} bt={record.backtracks}"
                ),
                accepted_event=True,
                force=True,
            )

        return callback

    fieldnames = [
        "record", "runTag", "k", "nt", "ndof", "status", "accepted",
        "baseC1Phi", "baseC2Phi", "baseEpsPhi", "baseResidual",
        "baseLeakage", "baseMissing", "baseLeakageRel", "baseMissingRel",
        "trialC1Phi", "trialC2Phi", "trialEpsPhi", "trialResidual",
        "trialLeakage", "trialMissing", "trialLeakageRel", "trialMissingRel",
        "trialActivityArea", "residualOk", "discrepancyOk", "branchOk", "activityOk",
        "trialPrimerStatus", "trialPrimerSteps", "trialPrimerEnergyInitial",
        "trialPrimerEnergyFinal", "trialPrimerInitialGradientNorm",
        "trialPrimerGradientNorm", "trialPrimerResidualRatio",
        "trialPrimerBranchOverlap", "trialPrimerActivityArea",
        "trialPrimerComputeTime", "trialPrimerPlotTime", "trialPrimerTime",
        "trialNewtonRescueTriggered", "trialNewtonInitialStatus",
        "trialNewtonInitialSteps", "trialNewtonRetryStatus",
        "trialNewtonRetrySteps", "trialNewtonTotalSteps",
        "trialNewtonTotalSolveTime",
        "trialNewtonStatus", "trialNewtonSteps", "trialNewtonContraction",
        "trialNewtonPredictedRemaining", "trialNewtonStallWindows",
        "trialNewtonForecastRemainingBudget",
        "c1Phi", "c2Phi", "width", "epsPhi", "trustRadius",
        "innerTol", "newtonStatus", "newtonSteps", "residual", "resEuclid",
        "leakage", "missing", "leakageRel", "missingRel", "activityArea",
        "certifiedArea", "certifiedLeakage", "certifiedMissing",
        "gradL1", "gradL2", "gradM1", "gradM2", "directL1", "directL2", "directM1", "directM2",
        "projectedGradNorm", "metricDualGradNorm", "sensIts1", "sensIts2", "sensRes1", "sensRes2", "sensSolveTime",
        "stepObjective", "dc1", "dc2", "stepMetricNorm", "stepDikinNorm",
        "stepPullbackNorm", "predictedReduction", "actualReduction", "rhoRatio",
        "branchOverlap", "activeJaccard", "activeRecall", "activePrecision", "activeDice",
        "bestActiveJaccardSeen", "jaccardStaleSteps", "jaccardDirectionChanges",
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
    best_jaccard = -math.inf
    best_jaccard_state: np.ndarray | None = None
    best_jaccard_c1 = math.nan
    best_jaccard_c2 = math.nan
    best_jaccard_iteration = -1
    jaccard_history: list[float] = []
    jaccard_stale_steps = 0
    jaccard_direction_change_count = 0
    jaccard_stop_triggered = False
    restored_best_jaccard = False
    initial_spectrum_measured = False
    cached_reduced_gradient: ReducedGradient | None = None
    sensitivity_check_completed = False
    trust_radius = min(
        max(float(args.trust_radius), float(args.trust_radius_min)),
        float(args.trust_radius_max),
    )
    previous_merit = 1.0
    frame_every = args.frame_every if args.frame_every is not None else args.plot_every

    try:
        for k in range(int(args.max_opt_it) + 1):
            step_start = time.perf_counter()
            eps_phi = epsilon_from_thresholds(args, c1_phi, c2_phi)
            inner_tol = inner_tolerance(args, previous_merit)
            if args.verify_sensitivities and not sensitivity_check_completed:
                inner_tol = min(
                    float(inner_tol), float(args.sensitivity_check_newton_tol)
                )

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
                plot_callback=make_newton_plot_callback(outer_k=k, stage="NEWTON_OUTER"),
            )
            state_pair_mismatch = threshold_constant_mismatch(
                c1_const,
                c2_const,
                eps_const,
                c1=c1_phi,
                c2=c2_phi,
                eps_phi=eps_phi,
            )
            state_pair_scale = max(abs(c1_phi), abs(c2_phi), abs(eps_phi), 1.0)
            if state_pair_mismatch > 64.0 * np.finfo(np.float64).eps * state_pair_scale:
                raise RuntimeError(
                    "Newton state was projected with constants from a different "
                    f"threshold pair (mismatch={state_pair_mismatch:.6e})"
                )
            if args.verbosity >= 2:
                root_print(
                    comm,
                    f"BASE_PAIR_STATE k={k} c1={c1_phi:.6e} c2={c2_phi:.6e} "
                    f"eps={eps_phi:.6e} residual={newton.residual:.6e} "
                    f"constantMismatch={state_pair_mismatch:.3e}",
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
            previous_merit = threshold_merit_rel(metrics, args)
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
            base_c1_phi = c1_phi
            base_c2_phi = c2_phi
            base_eps_phi = eps_phi
            base_residual = newton.residual
            base_leakage = metrics.leakage
            base_missing = metrics.missing
            base_leakage_rel = metrics.leakage_rel
            base_missing_rel = metrics.missing_rel
            base_active_jaccard = float(diagnostic_metrics["activeJaccard"])
            proposal_trust_radius = trust_radius
            base_simplex = threshold_simplex_point(
                c1_phi,
                c2_phi,
                c_min=c_min,
                c_max=c_upper,
                min_width=min_width,
            )
            base_state_h1 = h1_seminorm(comm=comm, function=u, dx=dx)
            if base_active_jaccard > best_jaccard:
                best_jaccard = base_active_jaccard
                best_jaccard_state = u.x.array.copy()
                best_jaccard_c1 = c1_phi
                best_jaccard_c2 = c2_phi
                best_jaccard_iteration = k
            if not jaccard_history:
                jaccard_history.append(base_active_jaccard)

            gradient: ReducedGradient | None = None
            trust_geometry: ThresholdTrustMetric | None = None
            gradient_cached = False
            step: ParameterStep | None = None
            status = "ITERATE"
            accepted = False
            actual_reduction = 0.0
            rho_ratio = math.nan
            branch_overlap = 1.0
            metric_dual_grad_norm = math.nan
            trial_log_c1: float | str = ""
            trial_log_c2: float | str = ""
            trial_log_eps: float | str = ""
            trial_log_residual: float | str = ""
            trial_log_leakage: float | str = ""
            trial_log_missing: float | str = ""
            trial_log_leakage_rel: float | str = ""
            trial_log_missing_rel: float | str = ""
            trial_log_activity_area: float | str = ""
            residual_ok_log: int | str = ""
            discrepancy_ok_log: int | str = ""
            branch_ok_log: int | str = ""
            activity_ok_log: int | str = ""
            trial_primer: EnergyPrimerResult | None = None
            trial_newton_log: NewtonResult | None = None
            trial_correction: EquilibriumCorrectionResult | None = None
            trial_simplex: ThresholdSimplexPoint | None = None
            delta_center = math.nan
            delta_width = math.nan
            sensitivity_metric_scaled_eig_min = math.nan
            sensitivity_metric_scaled_eig_max = math.nan
            predicted_state_step_h1 = math.nan
            predicted_state_step_h1_relative = math.nan
            predictor_defect_hminus1 = math.nan
            predictor_defect_relative = math.nan
            predictor_defect_to_step = math.nan
            predictor_metric_solve_time = 0.0
            initial_correction_to_step = math.nan
            inner_projection_ok = (
                newton.converged
                if args.require_inner_newton_convergence
                else (
                    newton.converged
                    or newton.residual <= max(10.0 * float(inner_tol), float(args.tol_res))
                )
            )
            if (
                args.picard_spectrum
                and not initial_spectrum_measured
                and newton.converged
            ):
                measure_picard_spectrum(
                    u=u,
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
                    residual=newton.residual,
                    args=args,
                    stage="INITIAL",
                    outer_iteration=k,
                    record_callback=record_picard_spectrum,
                )
                initial_spectrum_measured = True
            if k < int(args.max_opt_it) and not inner_projection_ok:
                status = "INNER_NEWTON_UNRESOLVED"
                final_status = status
                root_print(
                    comm,
                    f"THRESHOLD_STEP_SKIP k={k} reason=unresolved_base_projection "
                    f"newtonStatus={newton.status} residual={newton.residual:.6e} "
                    f"required={max(10.0 * float(inner_tol), float(args.tol_res)):.6e}",
                )
            elif k < int(args.max_opt_it):
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
                        args=sensitivity_args,
                        iteration=k,
                    )
                    gradient_time = time.perf_counter() - gradient_start
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
                try:
                    trust_geometry = threshold_trust_metric(
                        simplex=base_simplex,
                        sensitivity_metric=gradient.sensitivity_metric,
                        state_h1=base_state_h1,
                    )
                except ValueError as error:
                    raise RuntimeError(
                        f"cannot build intrinsic threshold trust metric at outer iteration {k}: {error}"
                    ) from error
                objective_gradient = threshold_objective_gradient(
                    metrics, gradient, args
                )
                metric_dual_grad_norm = quadratic_metric_dual_norm(
                    objective_gradient,
                    trust_geometry.combined,
                )
                if args.search_space_diagnostics:
                    metric_eig_min, metric_eig_max = (
                        gradient.sensitivity_metric_eigenvalues
                    )
                    if math.isfinite(base_state_h1) and base_state_h1 > 0.0:
                        dimensionless_metric_scale = (
                            c_scale / base_state_h1
                        ) ** 2
                        sensitivity_metric_scaled_eig_min = (
                            dimensionless_metric_scale * metric_eig_min
                        )
                        sensitivity_metric_scaled_eig_max = (
                            dimensionless_metric_scale * metric_eig_max
                        )
                    root_print(
                        comm,
                        "SEARCH_SPACE_BASE "
                        f"k={k} c=({base_c1_phi:.6e},{base_c2_phi:.6e}) "
                        f"center={base_simplex.center:.6e} "
                        f"width={base_simplex.width:.6e} "
                        "slack="
                        f"({base_simplex.left_gap:.6e},"
                        f"{base_simplex.width_gap:.6e},"
                        f"{base_simplex.right_gap:.6e}) "
                        "simplex="
                        f"({base_simplex.left_fraction:.6e},"
                        f"{base_simplex.width_fraction:.6e},"
                        f"{base_simplex.right_fraction:.6e}) "
                        f"simplexMin={base_simplex.minimum_fraction:.6e} "
                        f"trustRadius={proposal_trust_radius:.6e} "
                        f"stateH1={base_state_h1:.6e} "
                        "metric="
                        f"({gradient.sensitivity_metric[0, 0]:.6e},"
                        f"{gradient.sensitivity_metric[0, 1]:.6e},"
                        f"{gradient.sensitivity_metric[1, 1]:.6e}) "
                        f"metricEig=({metric_eig_min:.6e},{metric_eig_max:.6e}) "
                        "metricEigScaled="
                        f"({sensitivity_metric_scaled_eig_min:.6e},"
                        f"{sensitivity_metric_scaled_eig_max:.6e}) "
                        f"metricCond={gradient.sensitivity_metric_condition:.6e} "
                        "dikinMetric="
                        f"({trust_geometry.dikin[0, 0]:.6e},"
                        f"{trust_geometry.dikin[0, 1]:.6e},"
                        f"{trust_geometry.dikin[1, 1]:.6e}) "
                        "trustMetric="
                        f"({trust_geometry.combined[0, 0]:.6e},"
                        f"{trust_geometry.combined[0, 1]:.6e},"
                        f"{trust_geometry.combined[1, 1]:.6e}) "
                        f"trustMetricEig=({trust_geometry.eigenvalues[0]:.6e},"
                        f"{trust_geometry.eigenvalues[1]:.6e}) "
                        f"trustMetricCond={trust_geometry.condition:.6e} "
                        f"pullbackAvailable={int(trust_geometry.pullback_available)} "
                        f"metricDualGrad={metric_dual_grad_norm:.6e}",
                    )
                if args.verify_sensitivities and not sensitivity_check_completed:
                    verify_reduced_gradient_finite_differences(
                        output_path=sensitivity_check_csv,
                        u=u,
                        du=du,
                        rho=rho,
                        s1=s1,
                        s2=s2,
                        trial=trial,
                        test=test,
                        dx=dx,
                        bc=bc,
                        stiffness_form=stiffness_form,
                        tau_mask=tau_mask,
                        c1_const=c1_const,
                        c2_const=c2_const,
                        eps_const=eps_const,
                        rho_amp=params.rho_amp,
                        c1=c1_phi,
                        c2=c2_phi,
                        c_min=c_min,
                        c_max=c_upper,
                        min_width=min_width,
                        c_scale=c_scale,
                        target_area=target_area,
                        gradient=gradient,
                        base_newton=newton,
                        args=args,
                        nonlinear_args=nonlinear_args,
                        iteration=k,
                    )
                    sensitivity_check_completed = True
                stationary = metric_dual_grad_norm <= float(args.tol_grad)
                if stationary:
                    status = "STOPPED_METRIC_GRADIENT"
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
                        trust_metric=trust_geometry,
                        trust_radius=trust_radius,
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
                            f"euclideanNorm={step.step_norm:.6e} "
                            f"metricNorm={step.metric_norm:.6e} "
                            f"dikinNorm={step.dikin_norm:.6e} "
                            f"pullbackNorm={step.pullback_norm:.6e} "
                            f"predictedReduction={step.predicted_reduction:.6e} "
                            f"radius={trust_radius:.6e}"
                        ),
                    )
                    if step.metric_norm <= max(float(args.trust_radius_min), 1.0e-14):
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
                        trial_inner_tol = inner_tolerance(args, previous_merit)
                        if args.search_space_diagnostics:
                            if base_simplex is None:
                                raise RuntimeError(
                                    "search-space diagnostics require a base simplex point"
                                )
                            trial_simplex = threshold_simplex_point(
                                trial_c1,
                                trial_c2,
                                c_min=c_min,
                                c_max=c_upper,
                                min_width=min_width,
                            )
                            delta_center = (
                                trial_simplex.center - base_simplex.center
                            )
                            delta_width = (
                                trial_simplex.width - base_simplex.width
                            )
                            predicted_state_step_h1 = predicted_state_h1(
                                step.dc,
                                gradient.sensitivity_metric,
                            )
                            if base_state_h1 > 0.0:
                                predicted_state_step_h1_relative = (
                                    predicted_state_step_h1 / base_state_h1
                                )
                            if (
                                predictor_metric_solver is None
                                or predictor_residual_form is None
                            ):
                                raise RuntimeError(
                                    "predictor metric solver is unavailable while "
                                    "search-space diagnostics are enabled"
                                )
                            (
                                _,
                                _,
                                predictor_metric_solve_time,
                            ) = predictor_metric_solver.solve(
                                predictor_residual_form,
                                predictor_dual,
                            )
                            predictor_defect_hminus1 = h1_seminorm(
                                comm=comm,
                                function=predictor_dual,
                                dx=dx,
                            )
                            if base_state_h1 > 0.0:
                                predictor_defect_relative = (
                                    predictor_defect_hminus1 / base_state_h1
                                )
                            if predicted_state_step_h1 > 0.0:
                                predictor_defect_to_step = (
                                    predictor_defect_hminus1
                                    / predicted_state_step_h1
                                )
                            root_print(
                                comm,
                                "SEARCH_SPACE_TRIAL "
                                f"k={k} dc=({step.dc[0]:.6e},{step.dc[1]:.6e}) "
                                f"dCenter={delta_center:.6e} "
                                f"dWidth={delta_width:.6e} "
                                f"trialC=({trial_c1:.6e},{trial_c2:.6e}) "
                                f"trialCenter={trial_simplex.center:.6e} "
                                f"trialWidth={trial_simplex.width:.6e} "
                                "trialSimplex="
                                f"({trial_simplex.left_fraction:.6e},"
                                f"{trial_simplex.width_fraction:.6e},"
                                f"{trial_simplex.right_fraction:.6e}) "
                                f"trialSimplexMin={trial_simplex.minimum_fraction:.6e} "
                                f"stepMetricNorm={step.metric_norm:.6e} "
                                f"stepDikinNorm={step.dikin_norm:.6e} "
                                f"stepPullbackNorm={step.pullback_norm:.6e} "
                                f"predictedStateH1={predicted_state_step_h1:.6e} "
                                "predictedStateH1Rel="
                                f"{predicted_state_step_h1_relative:.6e} "
                                "predictorDefectHminus1="
                                f"{predictor_defect_hminus1:.6e} "
                                f"predictorDefectRel={predictor_defect_relative:.6e} "
                                f"defectToStep={predictor_defect_to_step:.6e} "
                                f"metricSolveTime={predictor_metric_solve_time:.6f}s",
                            )
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
                                f"predictor=U+Sdc dCenter={delta_center:.6e} "
                                f"dWidth={delta_width:.6e} "
                                f"predictedStateH1={predicted_state_step_h1:.6e} "
                                f"predictorDefectHminus1={predictor_defect_hminus1:.6e}"
                            ),
                        )
                        # Algorithm step 10: correct the predicted state at
                        # the trial thresholds. Newton runs first. Only a
                        # forecast stall restores the original predictor,
                        # applies the guarded short primer, and retries Newton.
                        correction_start = time.perf_counter()
                        guard_callback = lambda: evaluate_energy_primer_guard(
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
                            minimum_branch_overlap=(
                                0.0 if args.disable_branch_check else float(args.eta_overlap)
                            ),
                            minimum_activity_area=(
                                float(args.min_activity_fraction) * target_area
                            ),
                        )
                        trial_correction = solve_equilibrium_with_primer_rescue(
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
                            args=nonlinear_args,
                            prefix=f"outer_{k}_trial",
                            context="outer_trial",
                            outer_iteration=k,
                            plot_callback=make_newton_plot_callback(outer_k=k, stage="NEWTON_TRIAL"),
                            energy_gradient=energy_gradient,
                            energy_solver=energy_primer_solver,
                            energy_record_callback=record_energy_primer,
                            energy_plot_callback=make_energy_primer_plot_callback(
                                outer_k=k,
                                c1_value=trial_c1,
                                c2_value=trial_c2,
                                eps_phi_value=trial_eps,
                            ),
                            energy_statistics=energy_primer_statistics,
                            energy_guard_callback=guard_callback,
                        )
                        trial_newton = trial_correction.newton
                        trial_primer = trial_correction.primer
                        trial_newton_log = trial_newton
                        if (
                            predicted_state_step_h1 > 0.0
                            and math.isfinite(
                                trial_correction.initial_newton.first_update_h1
                            )
                        ):
                            initial_correction_to_step = (
                                trial_correction.initial_newton.first_update_h1
                                / predicted_state_step_h1
                            )
                        if args.search_space_diagnostics:
                            retry_first_direction_h1 = (
                                math.nan
                                if trial_correction.retry_newton is None
                                else trial_correction.retry_newton.first_direction_h1
                            )
                            retry_first_update_h1 = (
                                math.nan
                                if trial_correction.retry_newton is None
                                else trial_correction.retry_newton.first_update_h1
                            )
                            root_print(
                                comm,
                                "SEARCH_SPACE_CORRECTOR "
                                f"k={k} initialStatus="
                                f"{trial_correction.initial_newton.status} "
                                f"initialIts={trial_correction.initial_newton.iterations} "
                                "initialFirstDirectionH1="
                                f"{trial_correction.initial_newton.first_direction_h1:.6e} "
                                "initialFirstUpdateH1="
                                f"{trial_correction.initial_newton.first_update_h1:.6e} "
                                "initialFirstAlpha="
                                f"{trial_correction.initial_newton.first_alpha:.6e} "
                                "initialFirstBacktracks="
                                f"{trial_correction.initial_newton.first_backtracks} "
                                f"initialCorrectionToPredictedStep="
                                f"{initial_correction_to_step:.6e} "
                                f"rescue={int(trial_correction.rescue_triggered)} "
                                "retryStatus="
                                f"{trial_correction.retry_newton.status if trial_correction.retry_newton is not None else 'NONE'} "
                                f"retryFirstDirectionH1={retry_first_direction_h1:.6e} "
                                f"retryFirstUpdateH1={retry_first_update_h1:.6e}",
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
                                f"converged={int(trial_newton.converged)} "
                                f"initialStatus={trial_correction.initial_newton.status} "
                                f"initialIts={trial_correction.initial_newton.iterations} "
                                "initialFirstUpdateH1="
                                f"{trial_correction.initial_newton.first_update_h1:.6e} "
                                f"initialCorrectionToPredictedStep={initial_correction_to_step:.6e} "
                                f"rescue={int(trial_correction.rescue_triggered)} "
                                f"retryStatus={trial_correction.retry_newton.status if trial_correction.retry_newton is not None else 'NONE'} "
                                f"totalNewtonIts={trial_correction.total_newton_iterations} "
                                f"residual={trial_newton.residual:.6e} "
                                f"contraction={trial_newton.contraction:.6e} "
                                f"predictedRemaining={trial_newton.predicted_remaining} "
                                f"stallWindows={trial_newton.stall_windows} "
                                f"primerStatus={trial_primer.status if trial_primer is not None else 'DISABLED'} "
                                f"primerSteps={trial_primer.accepted_steps if trial_primer is not None else 0} "
                                f"linearSolveTime={trial_correction.total_newton_solve_time:.6f}s"
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
                        actual_reduction = (
                            threshold_merit_rel(metrics)
                            - threshold_merit_rel(trial_band)
                        )
                        discrepancy_ok = (
                            actual_reduction
                            >= float(args.accept_sufficient_decrease)
                        )
                        residual_ok = (
                            trial_newton.converged
                            if args.require_inner_newton_convergence
                            else (
                                trial_newton.converged
                                or trial_newton.residual <= max(
                                    10.0 * trial_inner_tol,
                                    float(args.tol_res),
                                )
                            )
                        )
                        branch_ok = args.disable_branch_check or branch_overlap >= float(args.eta_overlap)
                        activity_ok = trial_band.activity_area >= float(args.min_activity_fraction) * target_area
                        trial_log_c1 = trial_c1
                        trial_log_c2 = trial_c2
                        trial_log_eps = trial_eps
                        trial_log_residual = trial_newton.residual
                        trial_log_leakage = trial_band.leakage
                        trial_log_missing = trial_band.missing
                        trial_log_leakage_rel = trial_band.leakage_rel
                        trial_log_missing_rel = trial_band.missing_rel
                        trial_log_activity_area = trial_band.activity_area
                        residual_ok_log = int(residual_ok)
                        discrepancy_ok_log = int(discrepancy_ok)
                        branch_ok_log = int(branch_ok)
                        activity_ok_log = int(activity_ok)
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
                            previous_merit = threshold_merit_rel(metrics, args)
                            newton = trial_newton
                            inner_tol = trial_inner_tol
                            if rho_ratio < 0.25:
                                trust_radius *= float(args.trust_shrink)
                            elif rho_ratio > 0.75 and step.hit_boundary:
                                trust_radius *= float(args.trust_grow)
                            trust_radius = min(
                                max(trust_radius, float(args.trust_radius_min)),
                                float(args.trust_radius_max),
                            )
                            status = "ACCEPT"
                            threshold_sync_error = synchronize_threshold_constants(
                                c1_const,
                                c2_const,
                                eps_const,
                                c1=c1_phi,
                                c2=c2_phi,
                                eps_phi=eps_phi,
                            )
                            for stale_work in (du, s1, s2):
                                stale_work.x.array.fill(0.0)
                                stale_work.x.scatter_forward()
                            cached_reduced_gradient = None
                            root_print(
                                comm,
                                f"ACCEPTED_PAIR_STATE_SYNC k={k} "
                                f"c1={c1_phi:.6e} c2={c2_phi:.6e} eps={eps_phi:.6e} "
                                f"residual={trial_newton.residual:.6e} "
                                f"constantMismatch={threshold_sync_error:.3e} "
                                "derivativeCache=invalidated nextNewton=required",
                            )
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
                            trust_radius = max(
                                float(args.trust_radius_min),
                                trust_radius * float(args.trust_shrink),
                            )
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
                            accepted_event=accepted,
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
                            detail=f"nextTrustRadius={trust_radius:.6e} nextStatus={status}",
                        )
            else:
                status = "MAX_OPT_IT"

            if status == "REJECT" and gradient is not None:
                cached_reduced_gradient = gradient
            else:
                cached_reduced_gradient = None

            if accepted:
                accepted_steps += 1
                current_jaccard = float(diagnostic_metrics["activeJaccard"])
                previous_best_jaccard = best_jaccard
                new_best_jaccard = (
                    not math.isfinite(previous_best_jaccard)
                    or current_jaccard > previous_best_jaccard
                )
                if current_jaccard > best_jaccard:
                    best_jaccard = current_jaccard
                    best_jaccard_state = u.x.array.copy()
                    best_jaccard_c1 = c1_phi
                    best_jaccard_c2 = c2_phi
                    best_jaccard_iteration = k
                if new_best_jaccard:
                    jaccard_stale_steps = 0
                else:
                    jaccard_stale_steps += 1
                jaccard_history.append(current_jaccard)
                jaccard_direction_change_count = jaccard_direction_changes(
                    jaccard_history,
                    patience=int(args.jaccard_oscillation_patience),
                )
                if (
                    bool(args.jaccard_stagnation_stop)
                    and jaccard_stagnation_detected(
                        accepted_steps=accepted_steps,
                        stale_steps=jaccard_stale_steps,
                        minimum_accepted=int(args.jaccard_oscillation_min_accepted),
                        patience=int(args.jaccard_oscillation_patience),
                    )
                ):
                    status = "STOPPED_JACCARD_STAGNATION"
                    final_status = status
                    jaccard_stop_triggered = True
                    root_print(
                        comm,
                        f"JACCARD_STAGNATION_STOP k={k} accepted={accepted_steps} "
                        f"stale={jaccard_stale_steps} "
                        f"current={current_jaccard:.6e} best={best_jaccard:.6e} "
                        f"bestK={best_jaccard_iteration}",
                    )
                elif (
                    bool(args.jaccard_oscillation_stop)
                    and accepted_steps >= int(args.jaccard_oscillation_min_accepted)
                    and jaccard_stale_steps >= int(args.jaccard_oscillation_patience)
                    and jaccard_oscillation_detected(
                        jaccard_history,
                        patience=int(args.jaccard_oscillation_patience),
                    )
                ):
                    status = "STOPPED_JACCARD_OSCILLATION"
                    final_status = status
                    jaccard_stop_triggered = True
                    root_print(
                        comm,
                        f"JACCARD_STOP k={k} accepted={accepted_steps} "
                        f"stale={jaccard_stale_steps} reversals={jaccard_direction_change_count} "
                        f"current={current_jaccard:.6e} best={best_jaccard:.6e} "
                        f"bestK={best_jaccard_iteration}",
                    )
            if final_status == "MAX_OPT_IT" and status in {"STEP_TOO_SMALL"}:
                final_status = status
            if args.verbosity >= 1 and args.terminal_every > 0 and (
                    k % int(args.terminal_every) == 0
                    or status in {
                        "STOPPED_METRIC_GRADIENT",
                        "STOPPED_JACCARD_STAGNATION",
                        "STOPPED_JACCARD_OSCILLATION",
                        "REJECT",
                        "STEP_TOO_SMALL",
                    }
            ):
                root_print(
                    comm,
                    f"REDUCED_OPT k={k} status={status} c1={c1_phi:.6e} c2={c2_phi:.6e} "
                    f"eps={eps_phi:.6e} res={newton.residual:.6e} Lrel={metrics.leakage_rel:.6e} "
                    f"Mrel={metrics.missing_rel:.6e} trust={trust_radius:.6e} "
                    f"metricDualGrad={metric_dual_grad_norm:.6e} "
                    f"activeJ={diagnostic_metrics['activeJaccard']:.6e} "
                    f"stepObj={'' if step is None else step.objective_name} "
                    f"dc={'' if step is None else f'({step.dc[0]:.3e},{step.dc[1]:.3e})'} "
                    f"stepMetric={'' if step is None else f'{step.metric_norm:.3e}'} "
                    f"accepted={int(accepted)}",
                )

            if search_space_writer is not None:
                if base_simplex is None:
                    raise RuntimeError(
                        "search-space CSV requires a base simplex point"
                    )
                initial_newton_diagnostic = (
                    None
                    if trial_correction is None
                    else trial_correction.initial_newton
                )
                retry_newton_diagnostic = (
                    None
                    if trial_correction is None
                    else trial_correction.retry_newton
                )
                search_space_writer.writerow({
                    "record": "SEARCH_SPACE",
                    "runTag": run_tag,
                    "k": k,
                    "nt": nt,
                    "ndof": ndof,
                    "status": status,
                    "accepted": int(accepted),
                    "cMin": c_min,
                    "cMax": c_upper,
                    "minWidth": min_width,
                    "simplexScale": base_simplex.simplex_scale,
                    "trustRadius": proposal_trust_radius,
                    "dikinMetric11": "" if trust_geometry is None else trust_geometry.dikin[0, 0],
                    "dikinMetric12": "" if trust_geometry is None else trust_geometry.dikin[0, 1],
                    "dikinMetric22": "" if trust_geometry is None else trust_geometry.dikin[1, 1],
                    "relativePullbackMetric11": "" if trust_geometry is None else trust_geometry.pullback[0, 0],
                    "relativePullbackMetric12": "" if trust_geometry is None else trust_geometry.pullback[0, 1],
                    "relativePullbackMetric22": "" if trust_geometry is None else trust_geometry.pullback[1, 1],
                    "trustMetric11": "" if trust_geometry is None else trust_geometry.combined[0, 0],
                    "trustMetric12": "" if trust_geometry is None else trust_geometry.combined[0, 1],
                    "trustMetric22": "" if trust_geometry is None else trust_geometry.combined[1, 1],
                    "trustMetricEigMin": "" if trust_geometry is None else trust_geometry.eigenvalues[0],
                    "trustMetricEigMax": "" if trust_geometry is None else trust_geometry.eigenvalues[1],
                    "trustMetricCondition": "" if trust_geometry is None else trust_geometry.condition,
                    "pullbackAvailable": "" if trust_geometry is None else int(trust_geometry.pullback_available),
                    "baseC1": base_simplex.c1,
                    "baseC2": base_simplex.c2,
                    "baseCenter": base_simplex.center,
                    "baseWidth": base_simplex.width,
                    "baseGapLeft": base_simplex.left_gap,
                    "baseGapWidth": base_simplex.width_gap,
                    "baseGapRight": base_simplex.right_gap,
                    "baseSimplexLeft": base_simplex.left_fraction,
                    "baseSimplexWidth": base_simplex.width_fraction,
                    "baseSimplexRight": base_simplex.right_fraction,
                    "baseSimplexMin": base_simplex.minimum_fraction,
                    "baseLogLeftToRight": base_simplex.log_left_to_right,
                    "baseLogWidthToRight": base_simplex.log_width_to_right,
                    "dc1": "" if step is None else step.dc[0],
                    "dc2": "" if step is None else step.dc[1],
                    "deltaCenter": "" if step is None else delta_center,
                    "deltaWidth": "" if step is None else delta_width,
                    "stepNorm": "" if step is None else step.step_norm,
                    "stepMetricNorm": "" if step is None else step.metric_norm,
                    "stepDikinNorm": "" if step is None else step.dikin_norm,
                    "stepPullbackNorm": "" if step is None else step.pullback_norm,
                    "trialC1": "" if trial_simplex is None else trial_simplex.c1,
                    "trialC2": "" if trial_simplex is None else trial_simplex.c2,
                    "trialCenter": "" if trial_simplex is None else trial_simplex.center,
                    "trialWidth": "" if trial_simplex is None else trial_simplex.width,
                    "trialGapLeft": "" if trial_simplex is None else trial_simplex.left_gap,
                    "trialGapWidth": "" if trial_simplex is None else trial_simplex.width_gap,
                    "trialGapRight": "" if trial_simplex is None else trial_simplex.right_gap,
                    "trialSimplexLeft": "" if trial_simplex is None else trial_simplex.left_fraction,
                    "trialSimplexWidth": "" if trial_simplex is None else trial_simplex.width_fraction,
                    "trialSimplexRight": "" if trial_simplex is None else trial_simplex.right_fraction,
                    "trialSimplexMin": "" if trial_simplex is None else trial_simplex.minimum_fraction,
                    "trialLogLeftToRight": "" if trial_simplex is None else trial_simplex.log_left_to_right,
                    "trialLogWidthToRight": "" if trial_simplex is None else trial_simplex.log_width_to_right,
                    "baseStateH1": base_state_h1,
                    "sensitivityMetric11": "" if gradient is None else gradient.sensitivity_metric[0, 0],
                    "sensitivityMetric12": "" if gradient is None else gradient.sensitivity_metric[0, 1],
                    "sensitivityMetric22": "" if gradient is None else gradient.sensitivity_metric[1, 1],
                    "sensitivityMetricEigMin": "" if gradient is None else gradient.sensitivity_metric_eigenvalues[0],
                    "sensitivityMetricEigMax": "" if gradient is None else gradient.sensitivity_metric_eigenvalues[1],
                    "sensitivityMetricCondition": "" if gradient is None else gradient.sensitivity_metric_condition,
                    "sensitivityMetricScaledEigMin": "" if gradient is None else sensitivity_metric_scaled_eig_min,
                    "sensitivityMetricScaledEigMax": "" if gradient is None else sensitivity_metric_scaled_eig_max,
                    "predictedStateH1": "" if trial_simplex is None else predicted_state_step_h1,
                    "predictedStateH1Relative": "" if trial_simplex is None else predicted_state_step_h1_relative,
                    "predictorDefectHminus1": "" if trial_simplex is None else predictor_defect_hminus1,
                    "predictorDefectRelative": "" if trial_simplex is None else predictor_defect_relative,
                    "predictorDefectToPredictedStep": "" if trial_simplex is None else predictor_defect_to_step,
                    "predictorMetricSolveTime": "" if trial_simplex is None else predictor_metric_solve_time,
                    "initialNewtonStatus": "" if initial_newton_diagnostic is None else initial_newton_diagnostic.status,
                    "initialNewtonIterations": "" if initial_newton_diagnostic is None else initial_newton_diagnostic.iterations,
                    "initialNewtonFirstDirectionH1": "" if initial_newton_diagnostic is None else initial_newton_diagnostic.first_direction_h1,
                    "initialNewtonFirstUpdateH1": "" if initial_newton_diagnostic is None else initial_newton_diagnostic.first_update_h1,
                    "initialNewtonFirstAlpha": "" if initial_newton_diagnostic is None else initial_newton_diagnostic.first_alpha,
                    "initialNewtonFirstBacktracks": "" if initial_newton_diagnostic is None else initial_newton_diagnostic.first_backtracks,
                    "initialCorrectionToPredictedStep": "" if initial_newton_diagnostic is None else initial_correction_to_step,
                    "retryNewtonStatus": "" if retry_newton_diagnostic is None else retry_newton_diagnostic.status,
                    "retryNewtonIterations": "" if retry_newton_diagnostic is None else retry_newton_diagnostic.iterations,
                    "retryNewtonFirstDirectionH1": "" if retry_newton_diagnostic is None else retry_newton_diagnostic.first_direction_h1,
                    "retryNewtonFirstUpdateH1": "" if retry_newton_diagnostic is None else retry_newton_diagnostic.first_update_h1,
                    "trialResidual": trial_log_residual,
                    "trialBranchOverlap": "" if trial_correction is None else branch_overlap,
                    "trialActivityArea": trial_log_activity_area,
                    "trialLeakageRel": trial_log_leakage_rel,
                    "trialMissingRel": trial_log_missing_rel,
                    "predictedReduction": "" if step is None else step.predicted_reduction,
                    "actualReduction": "" if trial_correction is None else actual_reduction,
                    "rhoRatio": "" if trial_correction is None else rho_ratio,
                    "baseActiveJaccard": base_active_jaccard,
                    "resultActiveJaccard": diagnostic_metrics["activeJaccard"],
                })
                search_space_handle.flush()

            if writer is not None:
                writer.writerow({
                    "record": "OPT",
                    "runTag": run_tag,
                    "k": k,
                    "nt": nt,
                    "ndof": ndof,
                    "status": status,
                    "accepted": int(accepted),
                    "baseC1Phi": base_c1_phi,
                    "baseC2Phi": base_c2_phi,
                    "baseEpsPhi": base_eps_phi,
                    "baseResidual": base_residual,
                    "baseLeakage": base_leakage,
                    "baseMissing": base_missing,
                    "baseLeakageRel": base_leakage_rel,
                    "baseMissingRel": base_missing_rel,
                    "trialC1Phi": trial_log_c1,
                    "trialC2Phi": trial_log_c2,
                    "trialEpsPhi": trial_log_eps,
                    "trialResidual": trial_log_residual,
                    "trialLeakage": trial_log_leakage,
                    "trialMissing": trial_log_missing,
                    "trialLeakageRel": trial_log_leakage_rel,
                    "trialMissingRel": trial_log_missing_rel,
                    "trialActivityArea": trial_log_activity_area,
                    "residualOk": residual_ok_log,
                    "discrepancyOk": discrepancy_ok_log,
                    "branchOk": branch_ok_log,
                    "activityOk": activity_ok_log,
                    "trialPrimerStatus": "" if trial_primer is None else trial_primer.status,
                    "trialPrimerSteps": "" if trial_primer is None else trial_primer.accepted_steps,
                    "trialPrimerEnergyInitial": "" if trial_primer is None else trial_primer.initial_energy,
                    "trialPrimerEnergyFinal": "" if trial_primer is None else trial_primer.final_energy,
                    "trialPrimerInitialGradientNorm": "" if trial_primer is None else trial_primer.initial_gradient_norm,
                    "trialPrimerGradientNorm": "" if trial_primer is None else trial_primer.gradient_norm,
                    "trialPrimerResidualRatio": "" if trial_primer is None else trial_primer.residual_ratio,
                    "trialPrimerBranchOverlap": "" if trial_primer is None else trial_primer.branch_overlap,
                    "trialPrimerActivityArea": "" if trial_primer is None else trial_primer.activity_area,
                    "trialPrimerComputeTime": "" if trial_primer is None else max(trial_primer.elapsed - trial_primer.plot_time, 0.0),
                    "trialPrimerPlotTime": "" if trial_primer is None else trial_primer.plot_time,
                    "trialPrimerTime": "" if trial_primer is None else trial_primer.elapsed,
                    "trialNewtonRescueTriggered": "" if trial_correction is None else int(trial_correction.rescue_triggered),
                    "trialNewtonInitialStatus": "" if trial_correction is None else trial_correction.initial_newton.status,
                    "trialNewtonInitialSteps": "" if trial_correction is None else trial_correction.initial_newton.iterations,
                    "trialNewtonRetryStatus": "" if trial_correction is None or trial_correction.retry_newton is None else trial_correction.retry_newton.status,
                    "trialNewtonRetrySteps": "" if trial_correction is None or trial_correction.retry_newton is None else trial_correction.retry_newton.iterations,
                    "trialNewtonTotalSteps": "" if trial_correction is None else trial_correction.total_newton_iterations,
                    "trialNewtonTotalSolveTime": "" if trial_correction is None else trial_correction.total_newton_solve_time,
                    "trialNewtonStatus": "" if trial_newton_log is None else trial_newton_log.status,
                    "trialNewtonSteps": "" if trial_newton_log is None else trial_newton_log.iterations,
                    "trialNewtonContraction": "" if trial_newton_log is None else trial_newton_log.contraction,
                    "trialNewtonPredictedRemaining": "" if trial_newton_log is None else trial_newton_log.predicted_remaining,
                    "trialNewtonStallWindows": "" if trial_newton_log is None else trial_newton_log.stall_windows,
                    "trialNewtonForecastRemainingBudget": "" if trial_newton_log is None else trial_newton_log.forecast_remaining_budget,
                    "c1Phi": c1_phi,
                    "c2Phi": c2_phi,
                    "width": c2_phi - c1_phi,
                    "epsPhi": eps_phi,
                    "trustRadius": trust_radius,
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
                    # Retain the historical CSV column as an alias so existing
                    # readers do not fail when consuming new runs.
                    "projectedGradNorm": metric_dual_grad_norm,
                    "metricDualGradNorm": metric_dual_grad_norm,
                    "sensIts1": "" if gradient is None else (0 if gradient_cached else gradient.solve_iterations[0]),
                    "sensIts2": "" if gradient is None else (0 if gradient_cached else gradient.solve_iterations[1]),
                    "sensRes1": "" if gradient is None else gradient.solve_residuals[0],
                    "sensRes2": "" if gradient is None else gradient.solve_residuals[1],
                    "sensSolveTime": "" if gradient is None else (0.0 if gradient_cached else gradient.solve_time),
                    "stepObjective": "" if step is None else step.objective_name,
                    "dc1": "" if step is None else step.dc[0],
                    "dc2": "" if step is None else step.dc[1],
                    "stepMetricNorm": "" if step is None else step.metric_norm,
                    "stepDikinNorm": "" if step is None else step.dikin_norm,
                    "stepPullbackNorm": "" if step is None else step.pullback_norm,
                    "predictedReduction": "" if step is None else step.predicted_reduction,
                    "actualReduction": actual_reduction,
                    "rhoRatio": rho_ratio,
                    "branchOverlap": branch_overlap,
                    "activeJaccard": diagnostic_metrics["activeJaccard"],
                    "activeRecall": active_recall,
                    "activePrecision": active_precision,
                    "activeDice": active_dice,
                    "bestActiveJaccardSeen": best_jaccard,
                    "jaccardStaleSteps": jaccard_stale_steps,
                    "jaccardDirectionChanges": jaccard_direction_change_count,
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
                if args.plot_fields == "density":
                    display_fields = [rho, T]
                    display_titles = ["rho(phi)"]
                    target_contour_index = 1
                elif args.plot_fields == "state":
                    display_fields = [rho, u, T]
                    display_titles = ["rho(phi)", "phi"]
                    target_contour_index = 2
                else:
                    display_fields = [T, tau_band, phi_target, u, rho, phi_diff]
                    display_titles = [
                        "Torsion T", "tau band", "phiT", "phi", "rho(phi)", "phi-phiT"
                    ]
                    target_contour_index = 0
                plotter.emit(
                    display_fields,
                    display_titles,
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
                    contour_field_index=target_contour_index,
                    contour_levels=(c1_t, c2_t),
                )

            if status in {
                "STOPPED_METRIC_GRADIENT",
                "STOPPED_JACCARD_STAGNATION",
                "STOPPED_JACCARD_OSCILLATION",
                "STEP_TOO_SMALL",
                "INNER_NEWTON_UNRESOLVED",
            }:
                break
    finally:
        if opt_handle is not None:
            opt_handle.close()
        if search_space_handle is not None:
            search_space_handle.close()
        if energy_primer_handle is not None:
            energy_primer_handle.close()
        if stiffness_metric_solver is not None:
            stiffness_metric_solver.close()

    if final_metrics is None or final_newton is None or final_compute_metrics is None:
        raise RuntimeError("optimization did not produce a final iterate")

    if jaccard_stop_triggered and best_jaccard_state is not None:
        c1_phi = best_jaccard_c1
        c2_phi = best_jaccard_c2
        eps_phi = epsilon_from_thresholds(args, c1_phi, c2_phi)
        c1_const.value = PETSc.ScalarType(c1_phi)
        c2_const.value = PETSc.ScalarType(c2_phi)
        eps_const.value = PETSc.ScalarType(eps_phi)
        u.x.array[:] = best_jaccard_state
        u.x.scatter_forward()
        update_interpolated(
            rho,
            window_density_const_ufl(u, c1_const, c2_const, eps_const, params.rho_amp),
        )
        restored_best_jaccard = True
        root_print(
            comm,
            f"RESTORE_BEST_JACCARD k={best_jaccard_iteration} "
            f"activeJ={best_jaccard:.6e} c1={c1_phi:.6e} c2={c2_phi:.6e}",
        )

    # Final feasibility policy: no matter how loose the adaptive outer Newton
    # solves were, the reported final state must satisfy the requested final
    # residual tolerance at the final thresholds.
    eps_phi = epsilon_from_thresholds(args, c1_phi, c2_phi)
    final_newton_args = argparse.Namespace(**vars(final_solver_args))
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
    if args.picard_spectrum and final_newton.converged:
        measure_picard_spectrum(
            u=u,
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
            residual=final_newton.residual,
            args=args,
            stage="FINAL",
            outer_iteration=-1,
            record_callback=record_picard_spectrum,
        )
    elif args.picard_spectrum:
        root_print(
            comm,
            "PICARD_SPECTRUM_SKIP stage=FINAL reason=unconverged_equilibrium "
            f"newtonStatus={final_newton.status} residual={final_newton.residual:.6e}",
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
    area_scale = max(float(target_area), 1.0e-30)
    final_certified_leakage_rel = final_metrics.certified_leakage / area_scale
    final_certified_area_rel = final_metrics.certified_area / area_scale
    if not final_newton.converged:
        final_status = "NEWTON_NOT_CONVERGED"

    checkpoint_start = time.perf_counter()
    # Store the L2 projection whose mass action equals the nonlinear source
    # load. Nodal interpolation of a narrow logistic window can otherwise
    # introduce an artificial Poisson mismatch at transient handoff.
    rho_checkpoint = fem.Function(V, name="rho_equilibrium_load_equivalent")
    projection_iterations, projection_residual, projection_time = solve_linear_form(
        trial * test * dx,
        window_density_const_ufl(
            u, c1_const, c2_const, eps_const, params.rho_amp
        ) * test * dx,
        rho_checkpoint,
        [],
        prefix="checkpoint_density_projection_",
        solver=final_solver_args.linear_solver,
        ksp_type=final_solver_args.ksp_type,
        rtol=min(float(final_solver_args.linear_rtol), 1.0e-13),
        atol=min(float(final_solver_args.linear_atol), 1.0e-14),
        max_it=final_solver_args.linear_max_it,
        verbosity=args.verbosity,
    )
    write_equilibrium_checkpoint_v2(
        equilibrium_path,
        rho=rho_checkpoint,
        phi=u,
        metadata={
            "final_status": final_status,
            "final_residual": float(final_newton.residual),
            "pde_converged": bool(final_newton.converged),
            "run_tag": run_tag,
            "run_dir": run_dir,
            "mesh_path": Path(mesh_path).resolve(),
            "geometry_mode": geometry_mode,
            "quadrature_degree": int(qdeg),
            "alpha_t1": float(params.alpha_t1),
            "alpha_t2": float(params.alpha_t2),
            "rho_amp": float(params.rho_amp),
            "c1_phi": float(c1_phi),
            "c2_phi": float(c2_phi),
            "eps_phi": float(eps_phi),
            "eps_phi_ratio": float(eps_phi / (c2_phi - c1_phi)),
            "threshold_objective_mode": "soft_jaccard_loss",
            "initial_threshold_mode": str(args.initial_threshold_mode),
            "initial_projection_mode": str(args.initial_projection_mode),
            "frozen_frontier_selection_mode": (
                frozen_frontier_initialization.selection_mode
                if frozen_frontier_initialization is not None
                else "not_used"
            ),
            "frozen_frontier_bins": int(args.frozen_frontier_bins),
            "frozen_leakage_cap_rel": args.frozen_leakage_cap_rel,
            "homotopy_status": (
                selected_homotopy_result.status
                if selected_homotopy_result is not None
                else "not_used"
            ),
            "homotopy_lambda_final": (
                float(selected_homotopy_result.lambda_final)
                if selected_homotopy_result is not None
                else 1.0
            ),
            "rho_definition": "rho_amp*window(phi;c1,c2,eps)",
            "rho_representation": "l2_load_equivalent_projection",
            "rho_projection_iterations": int(projection_iterations),
            "rho_projection_residual": float(projection_residual),
        },
    )
    checkpoint_time = time.perf_counter() - checkpoint_start
    root_print(
        comm,
        "EQUILIBRIUM_WRITTEN "
        f"path={equilibrium_path} status={final_status} "
        f"projectionResidual={projection_residual:.3e} "
        f"time={checkpoint_time:.6f}s",
    )

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
            if args.plot_fields == "density":
                display_fields = [rho, T]
                display_titles = ["rho(phi), final"]
                target_contour_index = 1
            elif args.plot_fields == "state":
                display_fields = [rho, u, T]
                display_titles = ["rho(phi), final", "phi, final"]
                target_contour_index = 2
            else:
                display_fields = [T, tau_band, rho_design, phi_target, u, rho, phi_diff]
                display_titles = [
                    "Torsion T", "tau band", "rhoDesign", "phiT",
                    "phi", "rho(phi)", "phi-phiT",
                ]
                target_contour_index = 0
            plotter.emit(
                display_fields,
                display_titles,
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
                contour_field_index=target_contour_index,
                contour_levels=(c1_t, c2_t),
            )
        finally:
            args.plot_mode = original_plot_mode
    if frame_handle is not None:
        frame_handle.close()
    if homotopy_handle is not None:
        homotopy_handle.close()
    if picard_spectrum_handle is not None:
        picard_spectrum_handle.close()
    elapsed = time.perf_counter() - total_start
    final_soft_jaccard_loss = threshold_merit_rel(final_metrics, args)
    final_soft_jaccard = 1.0 - final_soft_jaccard_loss
    root_print(
        comm,
        f"FINAL status={final_status} c1={c1_phi:.6e} c2={c2_phi:.6e} epsPhi={eps_phi:.6e} "
        f"res={final_newton.residual:.6e} Lrel={final_metrics.leakage_rel:.6e} "
        f"Mrel={final_metrics.missing_rel:.6e} softJ={final_soft_jaccard:.6e} "
        f"softJLoss={final_soft_jaccard_loss:.6e} "
        f"certifiedArea={final_metrics.certified_area:.6e} "
        f"certifiedAreaRel={final_certified_area_rel:.6e} "
        f"certifiedLeakRel={final_certified_leakage_rel:.6e} "
        f"activeJ={final_compute_metrics['activeJaccard']:.6e} rhoRel={final_compute_metrics['relRhoDesign']:.6e}",
    )
    root_print(
        comm,
        "ENERGY_PRIMER_TOTAL "
        f"calls={energy_primer_statistics.calls} "
        f"acceptedSteps={energy_primer_statistics.accepted_steps} "
        f"solveTime={energy_primer_statistics.solve_time:.6f}s "
        f"computeTime={max(energy_primer_statistics.elapsed - energy_primer_statistics.plot_time, 0.0):.6f}s "
        f"plotTime={energy_primer_statistics.plot_time:.6f}s "
        f"elapsed={energy_primer_statistics.elapsed:.6f}s "
        f"restored={energy_primer_statistics.restored_calls}",
    )
    root_print(comm, f"TIME_TOTAL {elapsed:.3f}")
    root_print(comm, "========== END STRATEGY A DOLFINX WINDOW REDUCED OPTIMIZATION ==========")

    if comm.rank == 0:
        with summary_path.open("w", encoding="utf-8") as handle:
            handle.write(f"runTag {run_tag}\n")
            handle.write(f"runDir {run_dir}\n")
            handle.write(
                f"terminalLog {terminal_log_path if args.save_terminal_log else None}\n"
            )
            handle.write(f"meshFile {mesh_path}\n")
            handle.write(f"geometryMode {geometry_mode}\n")
            handle.write(f"initialState {args.initial_state}\n")
            handle.write(f"initialThresholdMode {args.initial_threshold_mode}\n")
            handle.write(f"initialProjectionMode {args.initial_projection_mode}\n")
            handle.write(f"includeFitInit {int(args.include_fit_init)}\n")
            handle.write(f"initialAlpha1 {args.initial_alpha1}\n")
            handle.write(f"initialAlpha2 {args.initial_alpha2}\n")
            handle.write(f"frozenFrontierBins {args.frozen_frontier_bins}\n")
            handle.write(
                f"frozenLeakageCapRel {args.frozen_leakage_cap_rel}\n"
            )
            handle.write(
                "frozenFrontierCsv "
                f"{frozen_frontier_csv if frozen_frontier_initialization is not None else None}\n"
            )
            if frozen_frontier_initialization is not None:
                frozen_info = frozen_frontier_initialization
                handle.write(
                    f"frozenSelectionMode {frozen_info.selection_mode}\n"
                )
                handle.write(
                    f"frozenLeakageCap {frozen_info.leakage_cap}\n"
                )
                handle.write(
                    f"frozenHistogramLower {frozen_info.histogram_lower}\n"
                )
                handle.write(
                    f"frozenHistogramUpper {frozen_info.histogram_upper}\n"
                )
                handle.write(
                    f"frozenFrontierPoints {frozen_info.frontier_points}\n"
                )
                handle.write(
                    "frozenEvaluatedIntervals "
                    f"{frozen_info.evaluated_intervals}\n"
                )
                handle.write(
                    f"frozenHardC1 {frozen_info.hard_point.c1}\n"
                )
                handle.write(
                    f"frozenHardC2 {frozen_info.hard_point.c2}\n"
                )
                handle.write(
                    f"frozenHardLeakage {frozen_info.hard_point.leakage}\n"
                )
                handle.write(
                    f"frozenHardMissing {frozen_info.hard_point.missing}\n"
                )
                handle.write(
                    f"frozenHardJaccard {frozen_info.hard_point.jaccard}\n"
                )
                handle.write(
                    "frozenSmoothLeakage "
                    f"{frozen_info.smooth_metrics.leakage}\n"
                )
                handle.write(
                    "frozenSmoothMissing "
                    f"{frozen_info.smooth_metrics.missing}\n"
                )
                handle.write(
                    "frozenSmoothJaccard "
                    f"{frozen_info.smooth_metrics.jaccard}\n"
                )
                handle.write(
                    "frozenRefinementIterations "
                    f"{frozen_info.refinement_iterations}\n"
                )
                handle.write(
                    "frozenRefinementMessage "
                    f"{frozen_info.refinement_message}\n"
                )
                handle.write(
                    "frozenTargetAreaSampled "
                    f"{frozen_info.target_area_sampled}\n"
                )
                handle.write(
                    "frozenTargetAreaAssembled "
                    f"{frozen_info.target_area_assembled}\n"
                )
            handle.write(
                "requireDirectSeedInteriorLevelCurves "
                f"{int(args.require_direct_seed_interior_level_curves)}\n"
            )
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
            handle.write(f"targetArea {target_area}\n")
            handle.write(f"epsMode {args.eps_mode}\n")
            handle.write(f"targetDensityMode {target_density_mode}\n")
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
            handle.write(f"requireInnerNewtonConvergence {int(args.require_inner_newton_convergence)}\n")
            handle.write(f"innerTolMax {args.inner_tol_max}\n")
            handle.write(f"innerTolGamma {args.inner_tol_gamma}\n")
            handle.write(
                f"searchSpaceDiagnostics {int(args.search_space_diagnostics)}\n"
            )
            handle.write(
                "searchSpaceCsv "
                f"{search_space_csv if args.search_space_diagnostics else None}\n"
            )
            handle.write(
                "searchSpaceCoordinates normalized_threshold_simplex\n"
            )
            handle.write("searchSpaceStateMetric H1_sensitivity_pullback\n")
            handle.write("searchSpacePredictorDefect Hminus1\n")
            handle.write("thresholdTrustMetric simplex_dikin_plus_relative_H1_pullback\n")
            handle.write("thresholdTrustMetricAlwaysActive 1\n")
            handle.write(f"thresholdTrustRadiusInitial {args.trust_radius}\n")
            handle.write(f"thresholdTrustRadiusMin {args.trust_radius_min}\n")
            handle.write(f"thresholdTrustRadiusMax {args.trust_radius_max}\n")
            handle.write(f"energyPrimer {int(args.energy_primer)}\n")
            handle.write(f"energyPrimerTol {args.energy_primer_tol}\n")
            handle.write(f"energyPrimerMaxIt {args.energy_primer_max_it}\n")
            handle.write(
                "energyPrimerMinResidualReduction "
                f"{args.energy_primer_min_residual_reduction}\n"
            )
            handle.write("energyPrimerRescueOnly 1\n")
            handle.write(
                f"energyPrimerCsv {energy_primer_csv if args.energy_primer else None}\n"
            )
            handle.write(f"energyPrimerCalls {energy_primer_statistics.calls}\n")
            handle.write(
                f"energyPrimerAcceptedSteps {energy_primer_statistics.accepted_steps}\n"
            )
            handle.write(
                f"energyPrimerSolveTime {energy_primer_statistics.solve_time}\n"
            )
            handle.write(
                "energyPrimerComputeTime "
                f"{max(energy_primer_statistics.elapsed - energy_primer_statistics.plot_time, 0.0)}\n"
            )
            handle.write(
                f"energyPrimerPlotTime {energy_primer_statistics.plot_time}\n"
            )
            handle.write(f"energyPrimerTime {energy_primer_statistics.elapsed}\n")
            handle.write(
                f"energyPrimerRestoredCalls {energy_primer_statistics.restored_calls}\n"
            )
            handle.write(f"newtonStallForecast {int(args.newton_stall_forecast)}\n")
            handle.write(f"newtonStallWindow {args.newton_stall_window}\n")
            handle.write(f"newtonStallPatience {args.newton_stall_patience}\n")
            handle.write(f"picardSpectrum {int(args.picard_spectrum)}\n")
            handle.write(
                f"picardSpectrumCsv {picard_spectrum_csv if args.picard_spectrum else None}\n"
            )
            handle.write(
                f"picardSpectrumEigTol {args.picard_spectrum_eig_tol}\n"
            )
            handle.write(
                f"picardSpectrumEigMaxIt {args.picard_spectrum_eig_max_it}\n"
            )
            handle.write(
                f"picardSpectrumRecordCount {len(picard_spectrum_records)}\n"
            )
            for spectrum in picard_spectrum_records:
                label = spectrum.stage.lower().capitalize()
                handle.write(
                    f"picardSpectrum{label}Status {spectrum.status}\n"
                )
                handle.write(
                    f"picardSpectrum{label}MuMin {spectrum.mu_min}\n"
                )
                handle.write(
                    f"picardSpectrum{label}MuMax {spectrum.mu_max}\n"
                )
                handle.write(
                    "picardSpectrum"
                    f"{label}SpectralRadiusBound {spectrum.spectral_radius_bound}\n"
                )
                handle.write(
                    "picardSpectrum"
                    f"{label}EnergyMinimum {spectrum.energy_minimum}\n"
                )
                handle.write(
                    "picardSpectrum"
                    f"{label}Contracting {spectrum.picard_contracting}\n"
                )
            handle.write("thresholdObjectiveMode soft_jaccard_loss\n")
            handle.write("thresholdTargetTolerances none\n")
            handle.write(f"thresholdMeritRel {final_soft_jaccard_loss}\n")
            handle.write(f"bestSoftJaccard {final_soft_jaccard}\n")
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
            handle.write(f"jaccardOscillationStop {int(args.jaccard_oscillation_stop)}\n")
            handle.write(f"jaccardStagnationStop {int(args.jaccard_stagnation_stop)}\n")
            handle.write(f"jaccardOscillationMinAccepted {args.jaccard_oscillation_min_accepted}\n")
            handle.write(f"jaccardOscillationPatience {args.jaccard_oscillation_patience}\n")
            handle.write("jaccardBestComparison strict\n")
            handle.write(f"jaccardOscillationTriggered {int(jaccard_stop_triggered)}\n")
            handle.write(f"jaccardDirectionChanges {jaccard_direction_change_count}\n")
            handle.write(f"jaccardStaleSteps {jaccard_stale_steps}\n")
            handle.write(f"bestJaccardObserved {best_jaccard}\n")
            handle.write(f"bestJaccardIteration {best_jaccard_iteration}\n")
            handle.write(f"restoredBestJaccard {int(restored_best_jaccard)}\n")
            handle.write("jaccardHistory " + ",".join(f"{value:.17g}" for value in jaccard_history) + "\n")
            handle.write(f"acceptedOptimizationSteps {accepted_steps}\n")
            handle.write(f"plotSevere {int(args.plot_severe)}\n")
            handle.write(f"plotInitialCandidates {int(args.plot_initial_candidates)}\n")
            handle.write(f"plotAcceptedStates {int(args.plot_accepted_states)}\n")
            handle.write(f"plotDesignHoldSeconds {args.plot_design_hold_seconds}\n")
            handle.write(f"thresholdCapMode {args.threshold_cap_mode}\n")
            handle.write(f"searchCMin {c_min}\n")
            handle.write(f"searchCMax {c_upper}\n")
            handle.write(f"minWidth {min_width}\n")
            handle.write(f"initCandidate {init_candidate_name}\n")
            handle.write(f"initCandidateScore {init_candidate_score}\n")
            handle.write(f"homotopyCsv {homotopy_csv if selected_homotopy_result is not None else None}\n")
            handle.write(f"homotopyTolRes {homotopy_tolerance(args)}\n")
            handle.write(f"homotopyInitialStep {args.homotopy_initial_step}\n")
            handle.write(f"homotopyMinStep {args.homotopy_min_step}\n")
            handle.write(f"homotopyMaxStep {args.homotopy_max_step}\n")
            handle.write(f"homotopyStepGrow {args.homotopy_step_grow}\n")
            handle.write(f"homotopyStepShrink {args.homotopy_step_shrink}\n")
            handle.write(f"homotopyMaxStages {args.homotopy_max_stages}\n")
            handle.write(f"homotopyPredictor {int(args.homotopy_predictor)}\n")
            if selected_homotopy_result is not None:
                handle.write(f"homotopyStatus {selected_homotopy_result.status}\n")
                handle.write(f"homotopyConverged {int(selected_homotopy_result.converged)}\n")
                handle.write(f"homotopyLambdaFinal {selected_homotopy_result.lambda_final}\n")
                handle.write(f"homotopyStages {selected_homotopy_result.stages}\n")
                handle.write(f"homotopyRejectedSteps {selected_homotopy_result.rejected_steps}\n")
                handle.write(
                    f"homotopyNewtonIterations "
                    f"{selected_homotopy_result.total_newton_iterations}\n"
                )
                handle.write(f"homotopyTangentSolveTime {selected_homotopy_result.tangent_solve_time}\n")
                handle.write(f"homotopyNewtonSolveTime {selected_homotopy_result.newton_solve_time}\n")
                handle.write(f"homotopyTime {selected_homotopy_result.elapsed}\n")
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
            handle.write(f"equilibriumCheckpoint {equilibrium_path}\n")
            handle.write(f"equilibriumCheckpointTime {checkpoint_time}\n")
            handle.write(f"rhoProjectionIterations {projection_iterations}\n")
            handle.write(f"rhoProjectionResidual {projection_residual}\n")
            handle.write(f"rhoProjectionSolveTime {projection_time}\n")
            handle.write(f"timeTotal {elapsed}\n")

    if not final_newton.converged:
        exit_code = 3
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
