"""Non-adaptive HDG Strategy A Newton solve on a five-corner star domain."""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hdgfem.assembly import hdg as hdg_assembly
from hdgfem.assembly.hdg_gram import (
    assemble_hdg_gram,
    build_condensed_hdg_gram_inverse,
    build_ilu_bicgstab_inverse,
)
from hdgfem.core.mesh import gmsh_smooth_star_mesh, gmsh_star_mesh
from hdgfem.core.space import DGField, DGSpace
from hdgfem.io.plot import add_field_to_plotter, reference_plot_points
from hdgfem.solvers.diff_rea import (
    DiffusionReactionHDGOptions,
    DiffusionReactionHDGSolver,
    _local_solver_pre_mats,
    _normalize_tau,
    diffusion_element_boundary_mats,
)


@dataclass
class StrategyParameters:
    """Parameters matching the star-domain FreeFEM Strategy A run."""

    alpha_t1: float = 0.60
    alpha_t2: float = 0.65
    eps_t_ratio: float = 0.06
    rho_amp: float = 1.0
    beta_phi1: float = 0.60
    beta_phi2: float = 0.65
    eps_phi_ratio: float = 0.10
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


@dataclass
class NewtonLogEntry:
    """One accepted, rejected, or terminal Newton iteration record."""

    iteration: int
    status: str
    euclidean_residual_squared: float
    hminus_residual_squared: float
    hminus_residual: float
    step_norm: float
    alpha: float
    backtracks: int
    mu_shift: float
    solver: str


@dataclass
class StrategyResult:
    """Programmatic result returned by :func:`run_strategy`."""

    field: DGField
    rho: DGField
    flux: np.ndarray
    trace: np.ndarray
    mesh: object
    space: DGSpace
    torsion: DGField
    rho_design: DGField
    phi_design: DGField
    residual: np.ndarray
    euclidean_residual_squared: float
    hminus_residual_squared: float
    hminus_residual: float
    history: list[NewtonLogEntry]
    converged: bool
    stop_reason: str


class TeeStream:
    """Write text to multiple stream-like objects."""

    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def zero_boundary(x, y):
    """Homogeneous Dirichlet data."""
    return np.zeros_like(x, dtype=np.float64)


def one_source(x, y):
    """Unit source."""
    return np.ones_like(x, dtype=np.float64)


def logistic(z: np.ndarray, eps: float) -> np.ndarray:
    """Stable logistic window primitive."""
    zz = np.asarray(z, dtype=np.float64) / float(eps)
    out = np.empty_like(zz)
    out[zz > 50.0] = 1.0
    out[zz < -50.0] = 0.0
    mask = (zz >= -50.0) & (zz <= 50.0)
    out[mask] = 1.0 / (1.0 + np.exp(-zz[mask]))
    return out


def window_values(values: np.ndarray, c1: float, c2: float, eps: float, amp: float) -> np.ndarray:
    """Return ``amp * (sigma(values-c1)-sigma(values-c2))``."""
    return amp * (logistic(values - c1, eps) - logistic(values - c2, eps))


def window_derivative(values: np.ndarray, c1: float, c2: float, eps: float, amp: float) -> np.ndarray:
    """Return derivative of the smooth double logistic window."""
    s1 = logistic(values - c1, eps)
    s2 = logistic(values - c2, eps)
    return amp * (s1 * (1.0 - s1) - s2 * (1.0 - s2)) / eps


def project_quadrature_values(space: DGSpace, values: np.ndarray, *, name: str) -> DGField:
    """Project element quadrature values into ``space``."""
    rhs = np.asarray(values, dtype=np.float64) @ space.quad_data.weighted_phi
    coeffs = rhs @ space.quad_data.MKrf_inv
    return space.field(coeffs, name=name)


def field_from_moments(space: DGSpace, moments: np.ndarray, *, name: str) -> DGField:
    """Return DG coefficients whose physical mass moments are ``moments``."""
    moments = np.asarray(moments, dtype=np.float64)
    expected = (space.mesh.num_tri, space.el_dof)
    if moments.shape != expected:
        raise ValueError(f"moments must have shape {expected}; got {moments.shape}")
    coeffs = (moments / space.mesh.aff_jacs[:, None]) @ space.quad_data.MKrf_inv
    return space.field(np.ascontiguousarray(coeffs), name=name)


def scalar_moments_from_values(space: DGSpace, values: np.ndarray) -> np.ndarray:
    """Return element moments ``int_K values * phi_i dx``."""
    rhs = np.asarray(values, dtype=np.float64) @ space.quad_data.weighted_phi
    rhs *= space.mesh.aff_jacs[:, None]
    return np.ascontiguousarray(rhs)


def block_source_from_scalar_moments(moments: np.ndarray, space: DGSpace) -> np.ndarray:
    """Embed scalar equation moments in a mixed diffusion HDG RHS."""
    out = np.zeros((space.mesh.num_tri, 3 * space.el_dof), dtype=np.float64)
    out[:, :space.el_dof] = moments
    return out


def residual_merit(euclidean_squared: float, hminus_squared: float, norm_name: str) -> float:
    """Return the scalar line-search merit from the requested residual norm."""
    if norm_name == "euclid":
        return float(np.sqrt(max(euclidean_squared, 0.0)))
    if norm_name == "hminus":
        return float(np.sqrt(max(hminus_squared, 0.0)))
    raise ValueError(f"unknown residual norm {norm_name!r}")


def dual_norm_squared_logged(
        gram_inverse,
        residual: np.ndarray,
        *,
        label: str,
        rtol: float | None = None,
        atol: float | None = None,
        maxiter: int | None = None,
):
    """Apply the Gram inverse with timing and solver diagnostics."""
    start = time.perf_counter()
    solve_options = {}
    if rtol is not None:
        solve_options["rtol"] = rtol
    if atol is not None:
        solve_options["atol"] = atol
    if maxiter is not None:
        solve_options["maxiter"] = maxiter
    option_text = " ".join(f"{key}={value}" for key, value in solve_options.items()) or "defaults"
    print(
        f"GRAM_NORM_START label={label} rhsNorm={np.linalg.norm(residual):.6e} "
        f"solveOptions={option_text}",
        flush=True,
    )
    value, diagnostics = gram_inverse.dual_norm_squared(residual, **solve_options)
    print(
        f"GRAM_NORM_DONE label={label} hminus2={value:.6e} "
        f"info={diagnostics.info} iterations={diagnostics.iterations} "
        f"invRel={diagnostics.relative_residual:.3e} solveElapsed={diagnostics.elapsed:.3f} "
        f"totalElapsed={time.perf_counter() - start:.3f}",
        flush=True,
    )
    return value, diagnostics


def build_hdg_gram_inverse_logged(
        space: DGSpace,
        args: argparse.Namespace,
        *,
        label: str,
        cg_rtol: float | None = None,
        cg_atol: float | None = None,
        cg_maxiter: int | None = None,
        verbose_every: int | None = None,
        verify_residual: bool | None = None,
):
    """Build the configured HDG Gram inverse with consistent diagnostics."""
    effective_cg_rtol = args.gram_cg_rtol if cg_rtol is None else cg_rtol
    effective_cg_atol = args.gram_cg_atol if cg_atol is None else cg_atol
    effective_cg_maxiter = args.gram_cg_maxiter if cg_maxiter is None else cg_maxiter
    effective_verbose_every = args.gram_cg_verbose_every if verbose_every is None else verbose_every
    effective_verify_residual = args.gram_verify_residual if verify_residual is None else verify_residual
    print(f"GRAM_BUILD_START label={label} method={args.gram_inverse}", flush=True)
    if args.gram_inverse == "condensed":
        gram_inverse = build_condensed_hdg_gram_inverse(
            space,
            sigma=args.gram_sigma,
            jump_weight=args.gram_jump_weight,
            cg_rtol=effective_cg_rtol,
            cg_atol=effective_cg_atol,
            cg_maxiter=effective_cg_maxiter,
            verbose_every=effective_verbose_every,
            verify_residual=effective_verify_residual,
        )
        print(
            f"GRAM_DONE label={label} method=condensed ndof={gram_inverse.ndof} "
            f"localDofs={gram_inverse.local_dofs} traceDofs={gram_inverse.trace_dofs} "
            f"storageRatio={gram_inverse.fill_ratio:.3f} setup={gram_inverse.setup_seconds:.4f} "
            f"numba={int(gram_inverse.numba_enabled)} jumpWeight={args.gram_jump_weight} "
            f"cgRtol={effective_cg_rtol:.3e} cgAtol={effective_cg_atol:.3e} "
            f"cgMaxiter={effective_cg_maxiter} cgVerboseEvery={effective_verbose_every} "
            f"verifyResidual={int(effective_verify_residual)}",
            flush=True,
        )
        return gram_inverse

    gram = assemble_hdg_gram(space, sigma=args.gram_sigma, jump_weight=args.gram_jump_weight)
    gram_inverse = build_ilu_bicgstab_inverse(
        gram,
        drop_tol=args.ilu_drop_tol,
        fill_factor=args.ilu_fill_factor,
    )
    print(
        f"GRAM_DONE label={label} method=spilu ndof={gram.ndof} nnz={gram.matrix.nnz} "
        f"iluFillRatio={gram_inverse.fill_ratio:.3f} setup={gram_inverse.setup_seconds:.4f}",
        flush=True,
    )
    return gram_inverse


def print_strategy_summary(history: list[NewtonLogEntry], *, final_euclid2: float, final_hminus2: float) -> None:
    """Print compact, grep-friendly Newton iteration and final residual summaries."""
    print("RUN_SUMMARY_BEGIN")
    print("RUN_SUMMARY_TABLE columns=k status solver euclid hminus step alpha bt mu")
    for entry in history:
        euclid = float(np.sqrt(max(entry.euclidean_residual_squared, 0.0)))
        if np.isfinite(entry.hminus_residual_squared):
            hminus_value = float(np.sqrt(max(entry.hminus_residual_squared, 0.0)))
        else:
            hminus_value = float("nan")
        print(
            f"RUN_SUMMARY_ROW {entry.iteration} {entry.status} {entry.solver} "
            f"{euclid:.6e} {hminus_value:.6e} {entry.step_norm:.6e} "
            f"{entry.alpha:.6e} {entry.backtracks} {entry.mu_shift:.6e}"
        )

    final_euclid = float(np.sqrt(max(final_euclid2, 0.0)))
    final_hminus = float(np.sqrt(max(final_hminus2, 0.0))) if np.isfinite(final_hminus2) else float("nan")
    hminus_over_euclid = final_hminus / max(final_euclid, 1.0e-300) if np.isfinite(final_hminus) else float("nan")
    hminus2_over_euclid2 = final_hminus2 / max(final_euclid2, 1.0e-300) if np.isfinite(final_hminus2) else float("nan")
    print(
        f"RUN_SUMMARY_FINAL euclid={final_euclid:.6e} hminus={final_hminus:.6e} "
        f"hminusOverEuclid={hminus_over_euclid:.6e} euclid2={final_euclid2:.6e} "
        f"hminus2={final_hminus2:.6e} hminus2OverEuclid2={hminus2_over_euclid2:.6e}"
    )
    print("RUN_SUMMARY_END")


def face_values(coeffs: np.ndarray, space: DGSpace) -> np.ndarray:
    """Evaluate scalar coefficients on element-face quadrature points."""
    return np.einsum("Ki,fiq->Kfq", coeffs, space.quad_data.bas_of_bd_quads, optimize=True)


def flux_coefficients(result) -> np.ndarray:
    """Return HDG flux coefficients with shape ``(2, elements, el_dof)``."""
    coeffs = np.asarray(result.flux.as_component_first(), dtype=np.float64)
    if coeffs.shape[0] != 2:
        raise ValueError(f"expected two flux components; got shape {coeffs.shape}")
    return np.ascontiguousarray(coeffs)


def finite_or_nan(value) -> float:
    """Return ``value`` as float, mapping missing diagnostics to NaN."""
    if value is None:
        return float("nan")
    return float(value)


def parse_csv_presets(value: str) -> tuple[str, ...]:
    """Parse a comma-separated PETSc preset list."""
    return tuple(preset.strip() for preset in value.split(",") if preset.strip())


def parse_petsc_options(entries: list[str] | None) -> dict[str, str] | None:
    """Parse repeated ``KEY=VALUE`` PETSc option overrides."""
    if not entries:
        return None
    options: dict[str, str] = {}
    for entry in entries:
        text = str(entry).strip()
        if not text:
            continue
        if "=" in text:
            key, value = text.split("=", 1)
        else:
            key, value = text, "1"
        key = key.strip().lstrip("-")
        value = value.strip()
        if not key:
            raise ValueError(f"invalid empty PETSc option key in {entry!r}")
        options[key] = value
    return options or None


def stiffness_moments(field: DGField) -> np.ndarray:
    """Return element moments ``int_K grad(field).grad(phi_i) dx``."""
    space = field.space
    mesh = space.mesh
    q = space.quad_data
    dx_values, dy_values = field.grad_values()
    gradients = np.einsum("KcD,Diq->Kciq", mesh.inv_aff_mats_t, q.dbas_of_quads, optimize=True)
    out = np.einsum(
        "K,Kq,q,Kiq->Ki",
        mesh.aff_jacs,
        dx_values,
        q.Krf_w,
        gradients[:, 0],
        optimize=True,
    )
    out += np.einsum(
        "K,Kq,q,Kiq->Ki",
        mesh.aff_jacs,
        dy_values,
        q.Krf_w,
        gradients[:, 1],
        optimize=True,
    )
    return np.ascontiguousarray(out)


def h1_seminorm(field: DGField) -> float:
    """Return ``sqrt(int |grad field|^2 dx)``."""
    dx_values, dy_values = field.grad_values()
    value = np.einsum(
        "K,Kq,q->",
        field.space.mesh.aff_jacs,
        dx_values * dx_values + dy_values * dy_values,
        field.space.quad_data.Krf_w,
        optimize=True,
    )
    return float(np.sqrt(max(float(value), 0.0)))


def scalar_weak_residual(field: DGField, *, source_values: np.ndarray) -> np.ndarray:
    """Return scalar weak residual moments ``int grad(u).grad(v)-source*v``."""
    return (stiffness_moments(field) - scalar_moments_from_values(field.space, source_values)).reshape(-1)


def scalar_diffusion_reaction_residual(
        field: DGField,
        *,
        diffusion: float,
        reaction_values: np.ndarray,
        source_moments: np.ndarray,
) -> np.ndarray:
    """Return weak residual moments for ``-div(diffusion grad u)+reaction*u=source``."""
    space = field.space
    values = field.values()
    reaction_moments = scalar_moments_from_values(space, reaction_values * values)
    return (float(diffusion) * stiffness_moments(field) + reaction_moments - source_moments).reshape(-1)


def hdg_residual(
        field: DGField,
        flux_coeffs: np.ndarray,
        trace: np.ndarray,
        *,
        source_values: np.ndarray,
        stabilization: float,
) -> np.ndarray:
    """Assemble the full mixed HDG residual for ``-Delta u = source``.

    The local residual follows the diffusion-reaction solver's unknown order
    ``[u_h, q_{x,h}, q_{y,h}]``.  The flux is the HDG mixed unknown returned by
    the solver, not a postprocessed projection of ``-grad u_h``.
    """
    space = field.space
    mesh = space.mesh
    q = space.quad_data
    tau = _normalize_tau(stabilization, space)
    d0, d1, m_tau, m_n0, m_n1, _ = _local_solver_pre_mats(0.0, tau, space)
    element_boundary = diffusion_element_boundary_mats(tau, space)
    source = block_source_from_scalar_moments(scalar_moments_from_values(space, source_values), space)
    local_trace = hdg_assembly.element_traces(trace, space)

    flux_coeffs = np.asarray(flux_coeffs, dtype=np.float64)
    expected_flux_shape = (2, mesh.num_tri, q.el_dof)
    if flux_coeffs.shape != expected_flux_shape:
        raise ValueError(f"flux_coeffs must have shape {expected_flux_shape}; got {flux_coeffs.shape}")
    qx_coeffs = np.ascontiguousarray(flux_coeffs[0])
    qy_coeffs = np.ascontiguousarray(flux_coeffs[1])
    local = np.zeros((mesh.num_tri, 3 * q.el_dof), dtype=np.float64)
    local[:, :q.el_dof] = (
        np.einsum("Kij,Kj->Ki", m_tau, field.coeffs, optimize=True)
        + np.einsum("Kij,Kj->Ki", m_n0 - d0, qx_coeffs, optimize=True)
        + np.einsum("Kij,Kj->Ki", m_n1 - d1, qy_coeffs, optimize=True)
    )
    mass = mesh.aff_jacs[:, None, None] * q.MKrf[None, :, :]
    local[:, q.el_dof:2 * q.el_dof] = (
        np.einsum("Kij,Kj->Ki", d0, field.coeffs, optimize=True)
        - np.einsum("Kij,Kj->Ki", mass, qx_coeffs, optimize=True)
    )
    local[:, 2 * q.el_dof:] = (
        np.einsum("Kij,Kj->Ki", d1, field.coeffs, optimize=True)
        - np.einsum("Kij,Kj->Ki", mass, qy_coeffs, optimize=True)
    )
    local -= np.einsum("Kij,Kj->Ki", element_boundary, local_trace, optimize=True)
    local -= source

    trace_residual_full = np.zeros((mesh.num_edg, q.edg_dof), dtype=np.float64)
    oriented = q.face_trace_test_element_trial_oriented[mesh.loc2oriented_face_coupling]
    u_lift = np.einsum("Kfai,Ki->Kfa", oriented, field.coeffs, optimize=True)
    qx_lift = np.einsum("Kfai,Ki->Kfa", oriented, qx_coeffs, optimize=True)
    qy_lift = np.einsum("Kfai,Ki->Kfa", oriented, qy_coeffs, optimize=True)
    trace_by_edge = trace.reshape(mesh.num_edg, q.edg_dof)
    for local_face in range(3):
        edges = mesh.loc2glob_edge[:, local_face]
        face_contrib = mesh.jacs_el_fc[:, local_face, None] * (
            mesh.normals[:, local_face, 0, None] * qx_lift[:, local_face]
            + mesh.normals[:, local_face, 1, None] * qy_lift[:, local_face]
            + tau[:, local_face, None] * u_lift[:, local_face]
            - tau[:, local_face, None] * (trace_by_edge[edges] @ q.M_rf_fc.T)
        )
        np.add.at(trace_residual_full, edges, face_contrib)

    interior_trace = trace_residual_full[mesh.int_edges_inds].reshape(-1)
    return np.concatenate((local.reshape(-1), interior_trace))


def strategy_residual(
        residual_kind: str,
        field: DGField,
        flux_coeffs: np.ndarray,
        trace: np.ndarray,
        *,
        source_values: np.ndarray,
        stabilization: float,
) -> np.ndarray:
    """Return the selected nonlinear residual vector."""
    if residual_kind == "mixed":
        return hdg_residual(
            field,
            flux_coeffs,
            trace,
            source_values=source_values,
            stabilization=stabilization,
        )
    if residual_kind == "scalar":
        return scalar_weak_residual(field, source_values=source_values)
    raise ValueError("residual_kind must be 'mixed' or 'scalar'")


def mixed_residual_block_norms(residual: np.ndarray, space: DGSpace) -> tuple[float, float, float, float]:
    """Return Euclidean norms of ``[u, qx, qy, trace]`` residual blocks."""
    local_size = space.mesh.num_tri * 3 * space.el_dof
    local = np.asarray(residual[:local_size], dtype=np.float64).reshape(space.mesh.num_tri, 3 * space.el_dof)
    trace = np.asarray(residual[local_size:], dtype=np.float64)
    return (
        float(np.linalg.norm(local[:, :space.el_dof])),
        float(np.linalg.norm(local[:, space.el_dof:2 * space.el_dof])),
        float(np.linalg.norm(local[:, 2 * space.el_dof:])),
        float(np.linalg.norm(trace)),
    )


def mixed_u_block_rhs_from_residual(residual: np.ndarray, space: DGSpace) -> np.ndarray:
    """Return ``-R_u`` moments from the current mixed HDG residual."""
    local_size = space.mesh.num_tri * 3 * space.el_dof
    local = np.asarray(residual[:local_size], dtype=np.float64).reshape(space.mesh.num_tri, 3 * space.el_dof)
    return np.ascontiguousarray(-local[:, :space.el_dof])


def solve_hdg_with_fallback(
        source,
        reaction,
        boundary_condition,
        space: DGSpace,
        *,
        diffusion=1.0,
        stabilization=1.0,
        initial_guess=None,
        try_petsc: bool = True,
        problem_label: str = "hdg_solve",
        verbose: bool | int = False,
        assembly_backend: str = "numpy",
        local_solver_backend: str = "numpy",
):
    """Solve with PETSc cg_gamg, then SciPy ILU/BiCGSTAB, then sparse direct."""
    attempts = [
        {
            "label": "petsc_cg_gamg",
            "kwargs": {
                "solver": "petsc",
                "petsc_preset": "cg_gamg",
                "preconditioner": None,
                "solver_rtol": 1.0e-11,
            },
        },
        {
            "label": "scipy_ilu_bicgstab",
            "kwargs": {
                "solver": "BICGSTAB",
                "preconditioner": "ilu",
                "solver_rtol": 1.0e-11,
                "ilu_drop_tol": 1.0e-10,
                "ilu_fill_factor": 50.0,
            },
        },
        {
            "label": "scipy_direct",
            "kwargs": {
                "solver": "direct",
                "preconditioner": None,
                "solver_rtol": 1.0e-12,
            },
        },
    ]
    if not try_petsc:
        attempts = attempts[1:]
    last_error = None
    for attempt in attempts:
        print(f"SOLVER_TRY problem={problem_label} label={attempt['label']}", flush=True)
        try:
            options = DiffusionReactionHDGOptions(
                diffusion=diffusion,
                stabilization=stabilization,
                boundary_mode="eliminate",
                assembly_backend=assembly_backend,
                local_solver_backend=local_solver_backend,
                initial_guess=initial_guess,
                verbose=verbose,
                **attempt["kwargs"],
            )
            result = DiffusionReactionHDGSolver(space, options=options).solve(
                source=source,
                reaction=reaction,
                boundary_condition=boundary_condition,
            )
            print(f"SOLVER_OK problem={problem_label} label={attempt['label']}", flush=True)
            return result, attempt["label"]
        except Exception as exc:  # pragma: no cover - exercised by optional solver availability.
            last_error = exc
            print(f"SOLVER_FAIL label={attempt['label']} error={type(exc).__name__}: {exc}", flush=True)
    raise RuntimeError("all HDG solver fallback attempts failed") from last_error


class NewtonCorrectionSolver:
    """Class-based Newton HDG correction solves with optional ILU reuse."""

    def __init__(
            self,
            space: DGSpace,
            *,
            assembly_backend: str,
            local_solver_backend: str,
            stabilization: float,
            solver_rtol: float,
            solver_atol: float,
            maxiter: int | None,
            ilu_drop_tol: float,
            ilu_fill_factor: float,
            ilu_reuse: int,
            reused_ilu_maxiter: int | None,
            use_petsc: bool,
            petsc_presets: tuple[str, ...],
            initial_petsc_presets: tuple[str, ...],
            petsc_switch_iteration: int,
            petsc_levels: int | None,
            petsc_divtol: float,
            petsc_monitor: bool,
            petsc_options: dict[str, str] | None,
            verbose: bool | int,
    ) -> None:
        self.space = space
        self.assembly_backend = assembly_backend
        self.local_solver_backend = local_solver_backend
        self.stabilization = stabilization
        self.solver_rtol = solver_rtol
        self.solver_atol = solver_atol
        self.maxiter = maxiter
        self.ilu_drop_tol = ilu_drop_tol
        self.ilu_fill_factor = ilu_fill_factor
        self.ilu_reuse = max(0, int(ilu_reuse))
        self.reused_ilu_maxiter = reused_ilu_maxiter
        self.use_petsc = bool(use_petsc)
        self.petsc_presets = petsc_presets
        self.initial_petsc_presets = initial_petsc_presets
        self.petsc_switch_iteration = max(0, int(petsc_switch_iteration))
        self.petsc_levels = petsc_levels
        self.petsc_divtol = petsc_divtol
        self.petsc_monitor = petsc_monitor
        self.petsc_options = petsc_options
        self.verbose = verbose
        self.cached_preconditioner = None
        self.reuse_remaining = 0
        self.cached_from_iteration: int | None = None

    def petsc_presets_for_iteration(self, iteration: int) -> tuple[str, ...]:
        """Return the PETSc preset list for the current Newton iteration."""
        if self.initial_petsc_presets and iteration < self.petsc_switch_iteration:
            return self.initial_petsc_presets
        return self.petsc_presets

    def _solve_once(
            self,
            *,
            source_h: DGField,
            reaction_h: DGField,
            boundary_condition,
            diffusion: float,
            initial_guess: np.ndarray,
            preconditioner,
            problem_label: str,
            label: str,
            maxiter: int | None,
            solver: str,
            petsc_preset: str = "cg_gamg",
    ):
        print(
            f"SOLVER_TRY problem={problem_label} label={label} "
            f"assembly={self.assembly_backend} preconditioner="
            f"{'reused_ilu' if preconditioner is not None and preconditioner != 'ilu' else preconditioner}",
            flush=True,
        )
        options = DiffusionReactionHDGOptions(
            diffusion=diffusion,
            stabilization=self.stabilization,
            solver=solver,
            preconditioner=preconditioner,
            solver_rtol=self.solver_rtol,
            solver_atol=self.solver_atol,
            maxiter=maxiter,
            ilu_drop_tol=self.ilu_drop_tol,
            ilu_fill_factor=self.ilu_fill_factor,
            petsc_preset=petsc_preset,
            petsc_levels=self.petsc_levels,
            petsc_options=self.petsc_options,
            petsc_divtol=self.petsc_divtol,
            petsc_monitor=self.petsc_monitor,
            boundary_mode="eliminate",
            assembly_backend=self.assembly_backend,
            local_solver_backend=self.local_solver_backend,
            initial_guess=initial_guess,
            verbose=self.verbose,
        )
        result = DiffusionReactionHDGSolver(self.space, options=options).solve(
            source=source_h,
            reaction=reaction_h,
            boundary_condition=boundary_condition,
        )
        solve = result.global_solve_result
        print(
            f"SOLVER_OK problem={problem_label} label={label} "
            f"iters={solve.iteration_count if solve is not None else None} "
            f"rel={finite_or_nan(None if solve is None else solve.relative_residual_norm):.3e} "
            f"tSolve={finite_or_nan(None if solve is None else solve.solve_elapsed_seconds):.3f} "
            f"tPrec={finite_or_nan(None if solve is None else solve.preconditioner_elapsed_seconds):.3f}",
            flush=True,
        )
        return result

    def _try_petsc_presets(
            self,
            *,
            source_h: DGField,
            reaction_h: DGField,
            boundary_condition,
            diffusion: float,
            initial_guess: np.ndarray,
            problem_label: str,
            iteration: int,
    ):
        """Try configured PETSc nonsymmetric Newton presets in order."""
        if not self.use_petsc:
            return None, None
        presets = self.petsc_presets_for_iteration(iteration)
        print(
            f"NEWTON_PETSC_PHASE k={iteration} switch={self.petsc_switch_iteration} "
            f"presets={','.join(presets)}",
            flush=True,
        )
        for preset in presets:
            label = f"petsc_{preset}"
            try:
                result = self._solve_once(
                    source_h=source_h,
                    reaction_h=reaction_h,
                    boundary_condition=boundary_condition,
                    diffusion=diffusion,
                    initial_guess=initial_guess,
                    preconditioner=None,
                    problem_label=problem_label,
                    label=label,
                    maxiter=self.maxiter,
                    solver="petsc",
                    petsc_preset=preset,
                )
                return result, label
            except Exception as exc:
                print(f"SOLVER_FAIL label={label} error={type(exc).__name__}: {exc}", flush=True)
                if "petsc4py is not importable" in str(exc):
                    print("SOLVER_PETSC_DISABLE reason=petsc4py_not_importable", flush=True)
                    self.use_petsc = False
                    break
        return None, None

    def solve(
            self,
            *,
            source_moments: np.ndarray,
            reaction_values: np.ndarray,
            boundary_condition,
            diffusion: float,
            initial_guess: np.ndarray,
            iteration: int,
    ):
        """Solve one Newton correction and update the reusable ILU cache."""
        source_h = field_from_moments(self.space, source_moments, name=f"newton_source_{iteration}")
        reaction_h = project_quadrature_values(
            self.space,
            reaction_values,
            name=f"newton_reaction_{iteration}",
        )
        problem_label = f"newton_{iteration}"
        petsc_result, petsc_label = self._try_petsc_presets(
            source_h=source_h,
            reaction_h=reaction_h,
            boundary_condition=boundary_condition,
            diffusion=diffusion,
            initial_guess=initial_guess,
            problem_label=problem_label,
            iteration=iteration,
        )
        if petsc_result is not None:
            return petsc_result, petsc_label

        if self.cached_preconditioner is not None and self.reuse_remaining > 0:
            try:
                result = self._solve_once(
                    source_h=source_h,
                    reaction_h=reaction_h,
                    boundary_condition=boundary_condition,
                    diffusion=diffusion,
                    initial_guess=initial_guess,
                    preconditioner=self.cached_preconditioner,
                    problem_label=problem_label,
                    label=f"scipy_bicgstab_reused_ilu_from_{self.cached_from_iteration}",
                    maxiter=self.reused_ilu_maxiter,
                    solver="BICGSTAB",
                )
                self.reuse_remaining -= 1
                return result, "scipy_bicgstab_reused_ilu"
            except Exception as exc:
                print(
                    f"SOLVER_FAIL label=scipy_bicgstab_reused_ilu error={type(exc).__name__}: {exc}; rebuilding ILU",
                    flush=True,
                )
                self.cached_preconditioner = None
                self.reuse_remaining = 0

        try:
            result = self._solve_once(
                source_h=source_h,
                reaction_h=reaction_h,
                boundary_condition=boundary_condition,
                diffusion=diffusion,
                initial_guess=initial_guess,
                preconditioner="ilu",
                problem_label=problem_label,
                label="scipy_ilu_bicgstab",
                maxiter=self.maxiter,
                solver="BICGSTAB",
            )
        except Exception as exc:
            print(f"SOLVER_FAIL label=scipy_ilu_bicgstab error={type(exc).__name__}: {exc}", flush=True)
            print(f"SOLVER_TRY problem={problem_label} label=scipy_direct", flush=True)
            options = DiffusionReactionHDGOptions(
                diffusion=diffusion,
                stabilization=self.stabilization,
                solver="direct",
                preconditioner=None,
                solver_rtol=self.solver_rtol,
                solver_atol=self.solver_atol,
                boundary_mode="eliminate",
                assembly_backend=self.assembly_backend,
                local_solver_backend=self.local_solver_backend,
                initial_guess=initial_guess,
                verbose=self.verbose,
            )
            result = DiffusionReactionHDGSolver(self.space, options=options).solve(
                source=source_h,
                reaction=reaction_h,
                boundary_condition=boundary_condition,
            )
            print(f"SOLVER_OK problem={problem_label} label=scipy_direct", flush=True)
            self.cached_preconditioner = None
            self.reuse_remaining = 0
            return result, "scipy_direct"

        solve = result.global_solve_result
        self.cached_preconditioner = None if solve is None else solve.preconditioner
        self.cached_from_iteration = iteration
        self.reuse_remaining = self.ilu_reuse
        print(
            f"NEWTON_ILU_CACHE iteration={iteration} reuseRemaining={self.reuse_remaining} "
            f"available={int(self.cached_preconditioner is not None)}",
            flush=True,
        )
        return result, "scipy_ilu_bicgstab"


class SideBySidePlotter:
    """Small PyVista updater for ``phi`` and ``rho`` panels."""

    def __init__(self, *, enabled: bool, resolution: int = 10):
        self.enabled = enabled
        self.resolution = resolution
        self.plotter = None
        self.reference_points = reference_plot_points(resolution)
        self._shown = False

    def _ensure_plotter(self) -> bool:
        if not self.enabled:
            return False
        try:
            import pyvista as pv
        except ImportError:
            print("PLOT_SKIP pyvista_not_available", flush=True)
            self.enabled = False
            return False
        if self.plotter is None:
            self.plotter = pv.Plotter(shape=(1, 2), window_size=(1500, 650))
        return True

    def show_initial(self, phi: DGField, rho: DGField, *, title: str) -> None:
        if not self.enabled:
            return
        self.update(phi, rho, title=title)
        input("Initial plot is open. Press Enter to start Newton; close the window to disable further plot updates...")

    def update(self, phi: DGField, rho: DGField, *, title: str) -> None:
        if not self._ensure_plotter():
            return
        try:
            self.plotter.clear()
            add_field_to_plotter(
                self.plotter,
                phi,
                reference_points=self.reference_points,
                subplot=(0, 0),
                title=f"{title}: phi",
                show_mesh=True,
            )
            add_field_to_plotter(
                self.plotter,
                rho,
                reference_points=self.reference_points,
                subplot=(0, 1),
                title=f"{title}: rho",
                show_mesh=True,
            )
            self.plotter.link_views()
            if not self._shown:
                self.plotter.show(interactive_update=True, auto_close=False)
                self._shown = True
            self.plotter.update()
        except Exception as exc:  # pragma: no cover - depends on GUI backend/window lifecycle.
            print(f"PLOT_DISABLE error={type(exc).__name__}: {exc}", flush=True)
            self.enabled = False

    def close(self) -> None:
        """Close the live plotter if it is still open."""
        if self.plotter is not None:
            try:
                self.plotter.close()
            except Exception:
                pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-size", type=float, default=0.02)
    parser.add_argument(
        "--star-kind",
        choices=("smooth", "polygonal"),
        default="smooth",
        help="smooth matches the FreeFEM GammaStar parameterization; polygonal uses alternating radii",
    )
    parser.add_argument("--star-n", type=int, default=260, help="number of boundary samples for --star-kind smooth")
    parser.add_argument("--star-r0", type=float, default=1.5, help="base radius for the smooth FreeFEM star")
    parser.add_argument("--star-amp", type=float, default=0.32, help="cosine amplitude for the smooth FreeFEM star")
    parser.add_argument("--star-mode", type=int, default=5, help="cosine mode/corner count for the smooth FreeFEM star")
    parser.add_argument("--star-inner-radius", type=float, default=0.72, help="inner radius for --star-kind polygonal")
    parser.add_argument("--star-outer-radius", type=float, default=1.0, help="outer radius for --star-kind polygonal")
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--basis", default="dub_orth", choices=("bernstein", "hier_C0", "dub_orth"))
    parser.add_argument("--hdg-tau", type=float, default=1.0)
    parser.add_argument("--gram-sigma", type=float, default=10.0)
    parser.add_argument(
        "--gram-jump-weight",
        choices=("unit", "scaled"),
        default="unit",
        help="unit uses ||u-uhat||^2_L2(face); scaled uses sigma*p^2/h times that term",
    )
    parser.add_argument(
        "--gram-inverse",
        choices=("condensed", "spilu"),
        default="condensed",
        help="method for applying the HDG Gram inverse in residual dual norms",
    )
    parser.add_argument("--gram-cg-rtol", type=float, default=1.0e-11)
    parser.add_argument("--gram-cg-atol", type=float, default=0.0)
    parser.add_argument("--gram-cg-maxiter", type=int, default=None)
    parser.add_argument(
        "--gram-cg-verbose-every",
        type=int,
        default=25,
        help="print condensed Gram CG progress every N iterations; use 0 to disable",
    )
    parser.add_argument(
        "--gram-verify-residual",
        action="store_true",
        help="explicitly compute ||rhs-Gz|| after each condensed Gram solve; expensive on large runs",
    )
    parser.add_argument(
        "--skip-final-hminus-check",
        action="store_true",
        help="skip the final diagnostic HDG Gram inverse application",
    )
    parser.add_argument(
        "--final-gram-cg-rtol",
        type=float,
        default=1.0e-6,
        help="relative tolerance for the final diagnostic Gram solve",
    )
    parser.add_argument(
        "--final-gram-cg-atol",
        type=float,
        default=1.0e-12,
        help="absolute tolerance for the final diagnostic Gram solve",
    )
    parser.add_argument(
        "--final-gram-cg-maxiter",
        type=int,
        default=120,
        help="iteration cap for the final diagnostic Gram solve",
    )
    parser.add_argument(
        "--final-gram-cg-verbose-every",
        type=int,
        default=0,
        help="print final Gram CG progress every N iterations; use 0 to disable",
    )
    parser.add_argument("--ilu-drop-tol", type=float, default=1.0e-12)
    parser.add_argument("--ilu-fill-factor", type=float, default=50.0)
    parser.add_argument(
        "--hdg-assembly-backend",
        choices=("numpy", "numba", "auto"),
        default="numba",
        help="diffusion-reaction HDG assembly backend; numba uses projected DGField coefficients",
    )
    parser.add_argument(
        "--hdg-local-solver-backend",
        choices=("numpy", "numba"),
        default="numba",
        help="local mixed inverse assembly backend for non-projected paths",
    )
    parser.add_argument("--hdg-solver-verbose", type=int, default=0, help="verbosity passed to HDG linear solves")
    parser.add_argument("--newton-solver-rtol", type=float, default=1.0e-10)
    parser.add_argument("--newton-solver-atol", type=float, default=1.0e-12)
    parser.add_argument("--newton-maxiter", type=int, default=None)
    parser.add_argument("--newton-ilu-drop-tol", type=float, default=1.0e-10)
    parser.add_argument("--newton-ilu-fill-factor", type=float, default=50.0)
    parser.add_argument(
        "--newton-petsc-presets",
        default="bicgstab_gamg,gmres_gamg,bicgstab_asm_ilu,gmres_asm_ilu",
        help="comma-separated PETSc presets tried before SciPy for Newton corrections after the switch iteration",
    )
    parser.add_argument(
        "--newton-petsc-initial-presets",
        default="",
        help="comma-separated PETSc presets for early Newton corrections; empty uses --newton-petsc-presets",
    )
    parser.add_argument(
        "--newton-petsc-switch-iteration",
        type=int,
        default=1,
        help="first Newton iteration using --newton-petsc-presets when initial presets are set",
    )
    parser.add_argument("--newton-petsc-levels", type=int, default=None)
    parser.add_argument("--newton-petsc-divtol", type=float, default=1.0e4)
    parser.add_argument("--newton-petsc-monitor", action="store_true")
    parser.add_argument(
        "--newton-petsc-option",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="extra PETSc option for Newton solves; repeatable, without leading dash",
    )
    parser.add_argument(
        "--skip-newton-petsc",
        action="store_true",
        help="skip PETSc only for Newton correction solves",
    )
    parser.add_argument(
        "--newton-ilu-reuse",
        type=int,
        default=1,
        help="number of following Newton corrections that reuse the last ILU preconditioner",
    )
    parser.add_argument(
        "--newton-reused-ilu-maxiter",
        type=int,
        default=25,
        help="iteration cap for stale-ILU reuse attempts before rebuilding a fresh ILU",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-resolution", type=int, default=10)
    parser.add_argument(
        "--plot-line-search",
        choices=("none", "candidate", "accepted", "final", "all"),
        default="candidate",
        help="which Newton trial states to send to the live phi/rho plotter",
    )
    parser.add_argument(
        "--line-search-norm",
        choices=("hminus", "euclid"),
        default="euclid",
        help="Armijo merit norm; euclid avoids Gram inverse applications, hminus enables HDG dual norms",
    )
    parser.add_argument(
        "--residual-kind",
        choices=("mixed", "scalar"),
        default="mixed",
        help="mixed uses the full HDG residual; scalar uses FreeFEM-like weak residual moments",
    )
    parser.add_argument(
        "--newton-shift-mode",
        choices=("none", "freefem"),
        default="none",
        help="none keeps physical HDG diffusion in the correction; freefem uses diffusion 1+mu as in scalar P2 FreeFEM",
    )
    parser.add_argument(
        "--compute-hminus",
        action="store_true",
        help="also compute and print r^T G^{-1} r diagnostics when using Euclidean line search",
    )
    parser.add_argument(
        "--plot-newton-trials",
        action="store_true",
        help="shortcut for --plot --plot-line-search all; plots every Armijo trial",
    )
    parser.add_argument("--max-it", type=int, default=None)
    parser.add_argument(
        "--tol-res",
        type=float,
        default=None,
        help="outer Newton residual-merit tolerance; defaults to StrategyParameters.tol_res",
    )
    parser.add_argument(
        "--tol-newton",
        type=float,
        default=None,
        help="outer Newton step tolerance; defaults to StrategyParameters.tol_newton",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="directory for a timestamped benchmark log; stdout/stderr are also kept on the terminal",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="explicit benchmark log path; stdout/stderr are also kept on the terminal",
    )
    parser.add_argument(
        "--check-linear-residuals",
        action="store_true",
        help="compute HDG residuals for the linear design solves to validate signs/order",
    )
    parser.add_argument("--skip-petsc", action="store_true", help="skip PETSc and start with SciPy ILU/BiCGSTAB")
    args = parser.parse_args(argv)
    if args.plot_newton_trials:
        args.plot = True
        args.plot_line_search = "all"
    return args


def default_log_file(args: argparse.Namespace) -> Path:
    """Return the default timestamped benchmark log path for a CLI run."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    case = (
        f"starN{args.star_n}_p{args.order}_tau{args.hdg_tau:g}_"
        f"{args.hdg_assembly_backend}_{args.newton_petsc_presets.replace(',', '-')}"
    )
    return args.log_dir / timestamp / f"strategyA_hdg_newton_{case}.log"


def run_strategy(args: argparse.Namespace, params: StrategyParameters | None = None) -> StrategyResult:
    """Run the non-adaptive HDG Strategy A solve and return final data."""
    if params is None:
        params = StrategyParameters()
    if args.max_it is not None:
        params.max_it = int(args.max_it)
    if args.tol_res is not None:
        params.tol_res = float(args.tol_res)
    if args.tol_newton is not None:
        params.tol_newton = float(args.tol_newton)
    history: list[NewtonLogEntry] = []
    converged = False
    stop_reason = "max_iterations"

    total_start = time.perf_counter()
    print("========== START STRATEGY A HDG NEWTON ==========")
    if args.star_kind == "smooth":
        mesh = gmsh_smooth_star_mesh(
            args.mesh_size,
            boundary_points=args.star_n,
            radius=args.star_r0,
            amplitude=args.star_amp,
            mode=args.star_mode,
            verbosity=0,
        )
        print(
            f"GEOMETRY smooth_star starN={args.star_n} r0={args.star_r0:.6e} "
            f"amp={args.star_amp:.6e} mode={args.star_mode} meshSize={args.mesh_size:.6e}"
        )
    else:
        mesh = gmsh_star_mesh(
            args.mesh_size,
            corners=args.star_mode,
            inner_radius=args.star_inner_radius,
            outer_radius=args.star_outer_radius,
            verbosity=0,
        )
        print(
            f"GEOMETRY polygonal_star corners={args.star_mode} inner={args.star_inner_radius:.6e} "
            f"outer={args.star_outer_radius:.6e} meshSize={args.mesh_size:.6e}"
        )
    space = DGSpace(mesh, args.order, basis_type=args.basis)
    print(f"MESH elements={mesh.num_tri} edges={mesh.num_edg} hmax={mesh.h:.6e}")
    print(f"SPACE order={space.order} basis={args.basis} el_dof={space.el_dof} ndof={space.ndof}")

    need_hminus = args.line_search_norm == "hminus" or args.compute_hminus
    gram_inverse = None
    if need_hminus:
        gram_inverse = build_hdg_gram_inverse_logged(space, args, label="initial")
    else:
        print("GRAM_SKIP label=initial reason=euclidean_line_search computeHminus=0")

    torsion, torsion_solver = solve_hdg_with_fallback(
        one_source,
        0.0,
        zero_boundary,
        space,
        diffusion=1.0,
        stabilization=args.hdg_tau,
        try_petsc=not args.skip_petsc,
        problem_label="torsion",
        verbose=args.hdg_solver_verbose,
        assembly_backend=args.hdg_assembly_backend,
        local_solver_backend=args.hdg_local_solver_backend,
    )
    T = torsion.field
    t_values = T.values()
    t_max = float(np.max(t_values))
    if t_max <= 1.0e-14:
        raise RuntimeError("torsion maximum is too small")
    c1_t = params.alpha_t1 * t_max
    c2_t = params.alpha_t2 * t_max
    eps_t = params.eps_t_ratio * (c2_t - c1_t)
    rho_design_values = window_values(t_values, c1_t, c2_t, eps_t, params.rho_amp)
    rho_design = project_quadrature_values(space, rho_design_values, name="rhoDesign")
    print(
        f"TORSION solver={torsion_solver} Tmin={np.min(t_values):.6e} Tmax={t_max:.6e} "
        f"c1T={c1_t:.6e} c2T={c2_t:.6e} epsT={eps_t:.6e}"
    )

    phi_design, phi_solver = solve_hdg_with_fallback(
        rho_design,
        0.0,
        zero_boundary,
        space,
        diffusion=1.0,
        stabilization=args.hdg_tau,
        initial_guess=torsion.trace,
        try_petsc=not args.skip_petsc,
        problem_label="phi_design",
        verbose=args.hdg_solver_verbose,
        assembly_backend=args.hdg_assembly_backend,
        local_solver_backend=args.hdg_local_solver_backend,
    )
    u = phi_design.field
    flux = flux_coefficients(phi_design)
    trace = phi_design.trace.copy()
    if args.check_linear_residuals:
        phi_design_residual = hdg_residual(
            u,
            flux,
            trace,
            source_values=rho_design.values(),
            stabilization=args.hdg_tau,
        )
        print(
            f"LINEAR_CHECK problem=phi_design residualEuclid2={float(phi_design_residual @ phi_design_residual):.6e} "
            f"residualEuclid={np.linalg.norm(phi_design_residual):.6e}",
            flush=True,
        )
    phi_values = u.values()
    phi_max = float(np.max(phi_values))
    if phi_max <= 1.0e-14:
        raise RuntimeError("phiDesign maximum is too small")
    c1_phi = params.beta_phi1 * phi_max
    c2_phi = params.beta_phi2 * phi_max
    eps_phi = params.eps_phi_ratio * (c2_phi - c1_phi)
    grad_phi = u.grad_values()
    max_grad_phi = float(np.max(np.sqrt(grad_phi[0] ** 2 + grad_phi[1] ** 2)))
    estimated_band_width = eps_phi / max(max_grad_phi, 1.0e-30)
    print(
        f"PHI_DESIGN solver={phi_solver} phiMin={np.min(phi_values):.6e} phiMax={phi_max:.6e} "
        f"c1Phi={c1_phi:.6e} c2Phi={c2_phi:.6e} epsPhi={eps_phi:.6e} "
        f"bandWidthEstimate={estimated_band_width:.6e} hmax={mesh.h:.6e}"
    )
    if mesh.h >= estimated_band_width:
        print("MESH_WARNING hmax_not_smaller_than_estimated_band_width")

    plotter = SideBySidePlotter(enabled=args.plot, resolution=args.plot_resolution)
    rho = project_quadrature_values(
        space,
        window_values(u.values(), c1_phi, c2_phi, eps_phi, params.rho_amp),
        name="rho",
    )
    plotter.show_initial(u, rho, title="initializer")

    rho_design_l2 = rho_design.l2_norm()
    mass_initial = float(np.einsum("K,Kq,q->", mesh.aff_jacs, rho.values(), space.quad_data.Krf_w, optimize=True))
    rho_design_mass = float(np.einsum("K,Kq,q->", mesh.aff_jacs, rho_design.values(), space.quad_data.Krf_w, optimize=True))
    mass_floor = params.mass_floor_fraction * max(mass_initial, rho_design_mass, 1.0e-30)
    print(
        f"BRANCH_GUARD massInitial={mass_initial:.6e} rhoDesignMass={rho_design_mass:.6e} "
        f"massFloor={mass_floor:.6e} rhoMaxFloor={params.rho_max_floor:.6e}"
    )
    mu_shift = params.mu_shift
    stagnation_count = 0
    reject_count = 0

    print(
        "HDG_HEADER "
        "k euclid2 hminus2 hminus nd alpha bt mu solver minU maxU maxRho massRho relRhoDesign status"
    )
    print(
        "HDG_MEANING euclid2=r^T r on full (q,u,uhat) residual; "
        "hminus2=r^T G^{-1} r using the HDG Gram inverse; "
        f"lineSearchNorm={args.line_search_norm} residualKind={args.residual_kind} "
        f"newtonShiftMode={args.newton_shift_mode} computeHminus={int(need_hminus)}"
    )
    residual_start = time.perf_counter()
    print("RESIDUAL_BUILD_START label=initial", flush=True)
    residual = strategy_residual(
        args.residual_kind,
        u,
        flux,
        trace,
        source_values=rho.values(),
        stabilization=args.hdg_tau,
    )
    euclid2 = float(residual @ residual)
    print(
        f"RESIDUAL_BUILD_DONE label=initial ndof={residual.size} euclid2={euclid2:.6e} "
        f"elapsed={time.perf_counter() - residual_start:.3f}",
        flush=True,
    )
    if need_hminus:
        hminus2, inv_diag = dual_norm_squared_logged(gram_inverse, residual, label="initial")
        hminus = float(np.sqrt(hminus2))
        inv_rel = inv_diag.relative_residual
    else:
        hminus2 = float("nan")
        hminus = float("nan")
        inv_rel = float("nan")
    print(f"RES_START euclid2={euclid2:.6e} hminus2={hminus2:.6e} hminus={hminus:.6e} invRel={inv_rel:.3e}")

    newton_petsc_presets = parse_csv_presets(args.newton_petsc_presets)
    initial_newton_petsc_presets = parse_csv_presets(args.newton_petsc_initial_presets)
    newton_petsc_options = parse_petsc_options(args.newton_petsc_option)
    newton_uses_petsc = not (args.skip_petsc or args.skip_newton_petsc)
    if newton_uses_petsc and not newton_petsc_presets:
        raise ValueError("--newton-petsc-presets must contain at least one preset unless PETSc is skipped")
    if newton_uses_petsc and args.newton_petsc_switch_iteration < 0:
        raise ValueError("--newton-petsc-switch-iteration must be non-negative")
    newton_solver = NewtonCorrectionSolver(
        space,
        assembly_backend=args.hdg_assembly_backend,
        local_solver_backend=args.hdg_local_solver_backend,
        stabilization=args.hdg_tau,
        solver_rtol=args.newton_solver_rtol,
        solver_atol=args.newton_solver_atol,
        maxiter=args.newton_maxiter,
        ilu_drop_tol=args.newton_ilu_drop_tol,
        ilu_fill_factor=args.newton_ilu_fill_factor,
        ilu_reuse=args.newton_ilu_reuse,
        reused_ilu_maxiter=args.newton_reused_ilu_maxiter,
        use_petsc=newton_uses_petsc,
        petsc_presets=newton_petsc_presets,
        initial_petsc_presets=initial_newton_petsc_presets,
        petsc_switch_iteration=args.newton_petsc_switch_iteration,
        petsc_levels=args.newton_petsc_levels,
        petsc_divtol=args.newton_petsc_divtol,
        petsc_monitor=args.newton_petsc_monitor,
        petsc_options=newton_petsc_options,
        verbose=args.hdg_solver_verbose,
    )
    print(
        f"NEWTON_SOLVER_CONFIG assembly={args.hdg_assembly_backend} "
        f"localBackend={args.hdg_local_solver_backend} "
        f"petsc={int(newton_uses_petsc)} "
        f"petscPresets={','.join(newton_petsc_presets)} "
        f"petscInitialPresets={','.join(initial_newton_petsc_presets) if initial_newton_petsc_presets else 'none'} "
        f"petscSwitchIteration={args.newton_petsc_switch_iteration} "
        f"petscLevels={args.newton_petsc_levels} "
        f"petscDivtol={args.newton_petsc_divtol:.3e} "
        f"petscOptions={newton_petsc_options if newton_petsc_options else 'none'} "
        f"solverFallback=scipy_bicgstab "
        f"rtol={args.newton_solver_rtol:.3e} atol={args.newton_solver_atol:.3e} "
        f"outerTolRes={params.tol_res:.3e} outerTolNewton={params.tol_newton:.3e} "
        f"maxiter={args.newton_maxiter} iluDrop={args.newton_ilu_drop_tol:.3e} "
        f"iluFill={args.newton_ilu_fill_factor:.3e} iluReuse={args.newton_ilu_reuse} "
        f"reusedIluMaxiter={args.newton_reused_ilu_maxiter}",
        flush=True,
    )

    for k in range(params.max_it):
        u_values = u.values()
        f_values = window_values(u_values, c1_phi, c2_phi, eps_phi, params.rho_amp)
        df_values = window_derivative(u_values, c1_phi, c2_phi, eps_phi, params.rho_amp)
        current_mass = float(np.einsum("K,Kq,q->", mesh.aff_jacs, f_values, space.quad_data.Krf_w, optimize=True))
        current_merit = residual_merit(euclid2, hminus2, args.line_search_norm)
        print(
            f"\nNEWTON_STEP_START k={k} currentEuclid2={euclid2:.6e} currentHminus2={hminus2:.6e} "
            f"currentHminus={hminus:.6e} mu={mu_shift:.6e} "
            f"minPhi={np.min(u_values):.6e} maxPhi={np.max(u_values):.6e} "
            f"minRho={np.min(f_values):.6e} maxRho={np.max(f_values):.6e} massRho={current_mass:.6e}"
        )
        if current_merit < params.tol_res:
            print(
                f"NEWTON_STOP reason=residual_tolerance_before_linear_solve k={k} "
                f"merit={current_merit:.6e} tol={params.tol_res:.6e}",
                flush=True,
            )
            print(
                f"HDG_LOG {k} {euclid2:.6e} {hminus2:.6e} {hminus:.6e} 0 "
                f"0 0 {mu_shift:.6e} none {np.min(u_values):.6e} {np.max(u_values):.6e} "
                f"{np.max(f_values):.6e} {current_mass:.6e} 0 CONVERGED"
            )
            history.append(NewtonLogEntry(
                iteration=k,
                status="CONVERGED",
                euclidean_residual_squared=euclid2,
                hminus_residual_squared=hminus2,
                hminus_residual=hminus,
                step_norm=0.0,
                alpha=0.0,
                backtracks=0,
                mu_shift=mu_shift,
                solver="none",
            ))
            converged = True
            stop_reason = "converged"
            mass_initial = current_mass
            break
        if args.residual_kind == "mixed":
            res_u, res_qx, res_qy, res_trace = mixed_residual_block_norms(residual, space)
            source_moments = mixed_u_block_rhs_from_residual(residual, space)
            print(
                f"MIXED_RES_BLOCKS k={k} u={res_u:.6e} qx={res_qx:.6e} "
                f"qy={res_qy:.6e} trace={res_trace:.6e}",
                flush=True,
            )
        else:
            source_moments = scalar_moments_from_values(space, f_values) - stiffness_moments(u)
        reaction = -df_values
        effective_mu = mu_shift if args.newton_shift_mode == "freefem" else 0.0
        correction_diffusion = 1.0 + effective_mu
        print(
            f"NEWTON_LINEARIZATION k={k} formulation=du shiftMode={args.newton_shift_mode} "
            f"diffusion={correction_diffusion:.6e} "
            f"reactionMin={np.min(reaction):.6e} reactionMax={np.max(reaction):.6e} "
            f"rhsMomentL2={np.linalg.norm(source_moments):.6e}"
        )

        correction, solver_label = newton_solver.solve(
            source_moments=source_moments,
            reaction_values=reaction,
            boundary_condition=zero_boundary,
            diffusion=correction_diffusion,
            initial_guess=np.zeros_like(trace),
            iteration=k,
        )
        du = correction.field
        flux_du = flux_coefficients(correction)
        trace_du = correction.trace
        correction_linear_residual = None
        if args.residual_kind == "scalar":
            correction_linear_residual = scalar_diffusion_reaction_residual(
                du,
                diffusion=correction_diffusion,
                reaction_values=reaction,
                source_moments=source_moments,
            )
        u_candidate = space.field(u.coeffs + du.coeffs, name="phiCandidate")
        flux_candidate = flux + flux_du
        trace_candidate = trace + trace_du
        if need_hminus:
            du_residual = strategy_residual(
                args.residual_kind,
                du,
                flux_du,
                trace_du,
                source_values=np.zeros_like(f_values),
                stabilization=args.hdg_tau,
            )
            nd2, _ = dual_norm_squared_logged(gram_inverse, du_residual, label=f"newton_{k}_step")
            nd = float(np.sqrt(nd2))
        else:
            nd = h1_seminorm(du)
        candidate_rho_values = window_values(u_candidate.values(), c1_phi, c2_phi, eps_phi, params.rho_amp)
        print(
            f"NEWTON_STEP_SOLVED k={k} solver={solver_label} stepH1={nd:.6e} "
            f"linearResidual={np.linalg.norm(correction_linear_residual) if correction_linear_residual is not None else np.nan:.6e} "
            f"candidateMinPhi={np.min(u_candidate.values()):.6e} candidateMaxPhi={np.max(u_candidate.values()):.6e} "
            f"candidateMinRho={np.min(candidate_rho_values):.6e} candidateMaxRho={np.max(candidate_rho_values):.6e}"
        )
        if args.plot and args.plot_line_search in {"candidate", "all"}:
            candidate_rho = project_quadrature_values(space, candidate_rho_values, name="rhoCandidate")
            plotter.update(u_candidate, candidate_rho, title=f"Newton {k} candidate")

        if nd < params.tol_newton:
            print(
                f"HDG_LOG {k} {euclid2:.6e} {hminus2:.6e} {hminus:.6e} {nd:.6e} "
                f"0 0 {mu_shift:.6e} {solver_label} {np.min(u.values()):.6e} {np.max(u.values()):.6e} "
                f"{np.max(rho.values()):.6e} {mass_initial:.6e} 0 CONVERGED"
            )
            history.append(NewtonLogEntry(
                iteration=k,
                status="CONVERGED",
                euclidean_residual_squared=euclid2,
                hminus_residual_squared=hminus2,
                hminus_residual=hminus,
                step_norm=nd,
                alpha=0.0,
                backtracks=0,
                mu_shift=mu_shift,
                solver=solver_label,
            ))
            converged = True
            stop_reason = "converged"
            break

        alpha = 1.0
        accepted = False
        old_hminus = hminus
        old_euclid = float(np.sqrt(max(euclid2, 0.0)))
        old_merit = residual_merit(euclid2, hminus2, args.line_search_norm)
        old_u = u
        old_flux = flux
        old_trace = trace
        old_rho = rho
        n_backtrack = 0
        last_trial_u = None
        last_trial_rho = None

        print(
            "LINESEARCH_HEADER "
            "k bt alpha trialEuclid2 trialHminus2 trialHminus trialMerit armijoTarget "
            "branchOK armijoOK massRho maxRho invRel decision"
        )

        while alpha >= params.alpha_min and n_backtrack <= params.max_backtrack:
            trial_coeffs = old_u.coeffs + alpha * (u_candidate.coeffs - old_u.coeffs)
            trial_flux = old_flux + alpha * (flux_candidate - old_flux)
            trial_trace = old_trace + alpha * (trace_candidate - old_trace)
            trial_u = space.field(trial_coeffs, name="phi")
            trial_rho_values = window_values(trial_u.values(), c1_phi, c2_phi, eps_phi, params.rho_amp)
            trial_mass = float(np.einsum("K,Kq,q->", mesh.aff_jacs, trial_rho_values, space.quad_data.Krf_w, optimize=True))
            trial_max_rho = float(np.max(trial_rho_values))
            branch_ok = trial_mass >= mass_floor and trial_max_rho >= params.rho_max_floor
            if branch_ok:
                trial_residual = strategy_residual(
                    args.residual_kind,
                    trial_u,
                    trial_flux,
                    trial_trace,
                    source_values=trial_rho_values,
                    stabilization=args.hdg_tau,
                )
                trial_euclid2 = float(trial_residual @ trial_residual)
                if need_hminus:
                    trial_hminus2, trial_inv_diag = dual_norm_squared_logged(
                        gram_inverse,
                        trial_residual,
                        label=f"newton_{k}_trial_{n_backtrack}",
                    )
                    trial_hminus = float(np.sqrt(trial_hminus2))
                    trial_inv_rel = trial_inv_diag.relative_residual
                else:
                    trial_hminus2 = float("nan")
                    trial_hminus = float("nan")
                    trial_inv_rel = float("nan")
                trial_merit = residual_merit(trial_euclid2, trial_hminus2, args.line_search_norm)
            else:
                trial_residual = None
                trial_euclid2 = float("nan")
                trial_hminus2 = float("nan")
                trial_hminus = float("nan")
                trial_inv_rel = float("nan")
                trial_merit = float("nan")
            armijo_target = (1.0 - params.armijo_c * alpha) * old_merit
            armijo_ok = branch_ok and trial_merit <= armijo_target
            if not branch_ok:
                decision = "REJECT_BRANCH"
            elif not armijo_ok:
                decision = "REJECT_ARMIJO"
            else:
                decision = "ACCEPT"
            print(
                f"LINESEARCH_TRIAL {k} {n_backtrack} {alpha:.6e} "
                f"{trial_euclid2:.6e} {trial_hminus2:.6e} {trial_hminus:.6e} "
                f"{trial_merit:.6e} {armijo_target:.6e} "
                f"{int(branch_ok)} {int(armijo_ok)} {trial_mass:.6e} {trial_max_rho:.6e} "
                f"{trial_inv_rel:.3e} {decision}",
                flush=True,
            )
            if args.plot and args.plot_line_search == "all":
                trial_rho = project_quadrature_values(space, trial_rho_values, name="rho")
                plotter.update(
                    trial_u,
                    trial_rho,
                    title=(
                        f"Newton {k} trial bt={n_backtrack} alpha={alpha:.2e}\n"
                        f"{args.line_search_norm}={trial_merit:.3e}, target={armijo_target:.3e}, {decision}"
                    ),
                )
            last_trial_u = trial_u
            last_trial_rho = None
            if args.plot and args.plot_line_search in {"final", "all"}:
                last_trial_rho = project_quadrature_values(space, trial_rho_values, name="rho")
            if branch_ok and armijo_ok:
                u = trial_u
                flux = trial_flux
                trace = trial_trace
                rho = project_quadrature_values(space, trial_rho_values, name="rho")
                if last_trial_rho is None:
                    last_trial_rho = rho
                residual = trial_residual
                hminus2 = trial_hminus2
                hminus = trial_hminus
                euclid2 = float(residual @ residual)
                mass_initial = trial_mass
                accepted = True
                rel_rho_design = np.nan
                if rho_design_l2 > 0.0:
                    diff_values = rho.values() - rho_design.values()
                    rel_rho_design = float(
                        np.sqrt(np.einsum("K,Kq,q->", mesh.aff_jacs, diff_values * diff_values, space.quad_data.Krf_w, optimize=True))
                        / rho_design_l2
                    )
                print(
                    f"HDG_LOG {k} {euclid2:.6e} {hminus2:.6e} {hminus:.6e} {nd:.6e} "
                    f"{alpha:.6e} {n_backtrack} {mu_shift:.6e} {solver_label} "
                    f"{np.min(u.values()):.6e} {np.max(u.values()):.6e} {trial_max_rho:.6e} "
                    f"{trial_mass:.6e} {rel_rho_design:.6e} ACCEPT invRel={trial_inv_rel:.3e}"
                )
                history.append(NewtonLogEntry(
                    iteration=k,
                    status="ACCEPT",
                    euclidean_residual_squared=euclid2,
                    hminus_residual_squared=hminus2,
                    hminus_residual=hminus,
                    step_norm=nd,
                    alpha=alpha,
                    backtracks=n_backtrack,
                    mu_shift=mu_shift,
                    solver=solver_label,
                ))
                if args.plot and args.plot_line_search in {"accepted", "final", "all"}:
                    plotter.update(u, rho, title=f"Newton {k} accepted")
                break
            alpha *= params.beta_ls
            n_backtrack += 1

        if not accepted:
            u = old_u
            flux = old_flux
            trace = old_trace
            rho = old_rho
            if args.newton_shift_mode == "freefem":
                mu_shift = min(params.mu_max, 2.0 * mu_shift)
            reject_count += 1
            print(f"HDG_LOG {k} {euclid2:.6e} {hminus2:.6e} {hminus:.6e} {nd:.6e} {alpha:.6e} {n_backtrack} {mu_shift:.6e} {solver_label} 0 0 0 0 0 FAIL_LS")
            history.append(NewtonLogEntry(
                iteration=k,
                status="FAIL_LS",
                euclidean_residual_squared=euclid2,
                hminus_residual_squared=hminus2,
                hminus_residual=hminus,
                step_norm=nd,
                alpha=alpha,
                backtracks=n_backtrack,
                mu_shift=mu_shift,
                solver=solver_label,
            ))
            if args.plot and args.plot_line_search in {"final", "all"} and last_trial_u is not None:
                plotter.update(last_trial_u, last_trial_rho, title=f"Newton {k} final rejected trial\nFAIL_LS")
            if reject_count >= 5 or mu_shift >= params.mu_max:
                print(f"NEWTON_STOP reason=too_many_failed_steps k={k}")
                stop_reason = "too_many_failed_steps"
                break
            continue

        reject_count = 0
        if args.newton_shift_mode == "freefem":
            if n_backtrack <= 1:
                mu_shift = max(params.mu_min, 0.85 * mu_shift)
            elif n_backtrack >= 8:
                mu_shift = min(params.mu_max, 1.5 * mu_shift)
        rel_drop = abs(old_hminus - hminus) / max(old_hminus, 1.0e-30)
        if args.line_search_norm == "euclid":
            rel_drop = abs(old_euclid - float(np.sqrt(max(euclid2, 0.0)))) / max(old_euclid, 1.0e-30)
        if rel_drop < params.stagnation_tol:
            stagnation_count += 1
            if args.newton_shift_mode == "freefem":
                mu_shift = min(params.mu_max, 1.25 * mu_shift)
            print(f"SHIFT {k} newMu={mu_shift:.6e} reason=stagnation")
            if stagnation_count >= params.max_stagnation:
                print(f"NEWTON_STOP reason=repeated_stagnation k={k}")
                stop_reason = "repeated_stagnation"
                break
        else:
            stagnation_count = 0

    euclid2 = float(residual @ residual)
    final_hminus_status = "ok"
    final_hminus_inv_rel = float("nan")
    if args.skip_final_hminus_check:
        if np.isfinite(hminus2):
            final_hminus_status = "reused"
            hminus = float(np.sqrt(max(hminus2, 0.0)))
        else:
            final_hminus_status = "skipped"
            hminus2 = float("nan")
            hminus = float("nan")
    elif np.isfinite(hminus2) and gram_inverse is not None:
        final_hminus_status = "reused"
        hminus = float(np.sqrt(max(hminus2, 0.0)))
    else:
        try:
            if gram_inverse is None:
                gram_inverse = build_hdg_gram_inverse_logged(
                    space,
                    args,
                    label="final",
                    cg_rtol=args.final_gram_cg_rtol,
                    cg_atol=args.final_gram_cg_atol,
                    cg_maxiter=args.final_gram_cg_maxiter,
                    verbose_every=args.final_gram_cg_verbose_every,
                    verify_residual=False,
                )
            hminus2, final_inv_diag = dual_norm_squared_logged(
                gram_inverse,
                residual,
                label="final",
                rtol=args.final_gram_cg_rtol,
                atol=args.final_gram_cg_atol,
                maxiter=args.final_gram_cg_maxiter,
            )
            hminus = float(np.sqrt(max(hminus2, 0.0)))
            final_hminus_inv_rel = final_inv_diag.relative_residual
            if final_inv_diag.info != 0:
                final_hminus_status = f"approx_info{final_inv_diag.info}"
        except Exception as exc:  # pragma: no cover - keeps expensive runs from losing final Euclidean data.
            final_hminus_status = f"failed:{type(exc).__name__}"
            hminus2 = float("nan")
            hminus = float("nan")
            print(f"FINAL_HMINUS_FAIL error={type(exc).__name__}: {exc}", flush=True)

    final_euclid = float(np.sqrt(max(euclid2, 0.0)))
    hminus_over_euclid = hminus / max(final_euclid, 1.0e-300) if np.isfinite(hminus) else float("nan")
    hminus2_over_euclid2 = hminus2 / max(euclid2, 1.0e-300) if np.isfinite(hminus2) else float("nan")
    print(
        f"FINAL_NORM_CHECK status={final_hminus_status} euclid={final_euclid:.6e} "
        f"hminus={hminus:.6e} hminusOverEuclid={hminus_over_euclid:.6e} "
        f"euclid2={euclid2:.6e} hminus2={hminus2:.6e} "
        f"hminus2OverEuclid2={hminus2_over_euclid2:.6e} invRel={final_hminus_inv_rel:.3e}"
    )
    print_strategy_summary(history, final_euclid2=euclid2, final_hminus2=hminus2)

    print(
        f"FINAL euclid2={euclid2:.6e} hminus2={hminus2:.6e} hminus={hminus:.6e} "
        f"minPhi={np.min(u.values()):.6e} maxPhi={np.max(u.values()):.6e} "
        f"maxRho={np.max(window_values(u.values(), c1_phi, c2_phi, eps_phi, params.rho_amp)):.6e} "
        f"massRho={mass_initial:.6e} converged={int(converged)} stopReason={stop_reason}"
    )
    print(f"TIME_TOTAL {time.perf_counter() - total_start:.3f}")
    print("========== END STRATEGY A HDG NEWTON ==========")
    return StrategyResult(
        field=u,
        rho=rho,
        flux=flux,
        trace=trace,
        mesh=mesh,
        space=space,
        torsion=T,
        rho_design=rho_design,
        phi_design=phi_design.field,
        residual=residual,
        euclidean_residual_squared=euclid2,
        hminus_residual_squared=hminus2,
        hminus_residual=hminus,
        history=history,
        converged=converged,
        stop_reason=stop_reason,
    )


def main() -> StrategyResult:
    """CLI entry point returning the final strategy result."""
    args = parse_args()
    if args.log_dir is not None and args.log_file is not None:
        raise ValueError("--log-dir and --log-file are mutually exclusive")
    log_file = args.log_file
    if args.log_dir is not None:
        log_file = default_log_file(args)
    if log_file is None:
        return run_strategy(args)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as handle:
        stdout = TeeStream(sys.stdout, handle)
        stderr = TeeStream(sys.stderr, handle)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print(f"LOG_FILE path={log_file.resolve()}", flush=True)
            print(f"COMMAND {' '.join(sys.argv)}", flush=True)
            return run_strategy(args)


if __name__ == "__main__":
    main()
