"""Stationary conservative advection-diffusion-reaction HDG solver."""

from __future__ import annotations

import time
from dataclasses import dataclass, fields, replace
from typing import Any, Literal

import numpy as np

from hdgfem.hdg import condensation as hdg
from hdgfem.hdg import matrices
from hdgfem.assembly.advection_diffusion_reaction import (
    ADRPreparedData,
    assemble_numpy,
    local_solvers_numpy,
    prepare_adr_data,
)
from hdgfem.core.element_coefficients import ElementCoefficient
from hdgfem.core.space import DGField, DGSpace, VectorDGField
from hdgfem.linalg.reduction import KnownDofReduction, expand_known_dofs
from hdgfem.linalg.results import SolveResult
from hdgfem.linalg.system import solve_global_system
from hdgfem.solvers.diffusion_reaction import (
    FluxPostprocessSpace,
    _build_hdg_postprocess_cache,
    _format_seconds,
    _normalize_flux_postprocess_space,
    _normalize_hdg_postprocess_mode,
    _postprocess_rt_flux_from_samples,
    _timed_call,
    _verbosity_level,
    split_diffusion_unknowns,
)


AssemblyBackend = Literal["numpy", "numba", "raw-cuda"]
ReconstructionBackend = Literal["auto", "numpy", "numba", "raw-cuda"]
PostprocessingBackend = Literal["auto", "numba", "cupy"]
PostprocessMode = Literal["none", "primal", "flux", "both"]
@dataclass(frozen=True)
class AdvectionDiffusionReactionTimings:
    """Wall-time breakdown for one stationary ADR solve."""

    preparation: float = 0.0
    local_solver: float = 0.0
    trace_assembly: float = 0.0
    solve: float = 0.0
    reconstruction: float = 0.0
    postprocessing: float = 0.0
    total: float = 0.0
    details: dict[str, float] | None = None

    @property
    def assembly(self) -> float:
        """Return setup and assembly time before the global sparse solve."""
        return self.preparation + self.local_solver + self.trace_assembly


@dataclass(frozen=True)
class AdvectionDiffusionReactionResult:
    """Fields, trace system, diagnostics, and optional post-processing.

    Raw-CUDA results retain CuPy ``trace`` and ``local_unknowns`` arrays and
    lazy device-backed fields when ``materialize_host_solution=False``.
    """

    field: DGField
    flux: VectorDGField
    trace: Any
    timings: AdvectionDiffusionReactionTimings
    total_flux: VectorDGField | None = None
    postprocessed_field: DGField | None = None
    postprocessed_flux: VectorDGField | None = None
    local_unknowns: Any = None
    matrix_rows: Any = None
    matrix_cols: Any = None
    matrix_data: Any = None
    matrix_indptr: Any = None
    matrix_indices: Any = None
    matrix_format: str = "coo"
    rhs: Any = None
    boundary_trace: Any = None
    reduction: KnownDofReduction | None = None
    local_solver: np.ndarray | None = None
    element_boundary_mats: np.ndarray | None = None
    tau_advection: np.ndarray | None = None
    tau_diffusion: np.ndarray | None = None
    beta_dot_normal: np.ndarray | None = None
    diffusion_structure: dict[str, int] | None = None
    assembly_backend: AssemblyBackend = "numpy"
    reconstruction_backend: str = "numpy"
    postprocessing_backend: str = "numba"
    global_solve_result: SolveResult | None = None


@dataclass(frozen=True)
class AdvectionDiffusionReactionHDGOptions:
    """Configuration for :class:`AdvectionDiffusionReactionHDGSolver`.

    ``advection_stabilization=None`` selects sidewise upwinding
    ``abs(beta_h.n)``. ``diffusion_stabilization="global_length"`` is the
    production default and selects ``gamma_d*kappa/L_Omega``. The legacy
    inverse-h rule remains available explicitly as ``"inverse-h"``; ``None``
    is retained as its compatibility alias. Host
    NumPy/Numba assembly and reconstruction stages may be selected independently.
    Raw CUDA assembly requires Raw CUDA reconstruction. Total-flux
    postprocessing uses public choices ``l2_closest`` and ``RT_projection``;
    legacy spellings remain accepted. CuPy performs both recoveries on device.
    ``materialize_host_solution=False`` retains raw-CUDA results on device;
    accessing a returned field's ``coeffs`` explicitly downloads that field.
    ``cache_local_factors`` (raw CUDA only) keeps local factors from assembly
    for reconstruction: ``"schur-lu"`` the Schur LU (``NEL*NEL`` reals and
    ``NEL`` ints per element), ``"schur-lu+mass"`` (default) also the factored
    variable-tensor mass (up to ``4*NEL*NEL`` reals per element for general
    kappa, none for constant kappa); ``"none"`` refactors during reconstruction.

    Repeated raw-CUDA solves through :class:`AdvectionDiffusionReactionHDGSolver`
    can reuse state across calls. ``amgx_reuse="solver"`` keeps the AMGX
    objects and redoes only the setup for each new matrix (same result, no
    per-call object creation); ``"preconditioner"`` also keeps the previous
    setup and replaces the coefficients (a stale preconditioner), refreshing it
    every ``amgx_refresh_interval`` solves, when iterations exceed
    ``amgx_refresh_iteration_growth`` times the count after the last refresh,
    and after a failed solve. ``reuse_static_coefficients`` keeps the prepared
    diffusion tensor, the diffusion stabilization and the sparsity pattern
    while ``diffusion`` and ``diffusion_stabilization`` stay the same objects.
    The one-shot function has no cache, so ``amgx_reuse`` needs the solver class.

    Host (Numba) counterparts: ``numba_reuse_local_columns`` keeps every
    element's local solution columns from assembly so reconstruction is a
    contraction with the trace (``3*el_dof*(3*edg_dof+1)`` reals per element;
    the solver class also keeps that buffer between solves).
    ``pardiso_reuse_analysis`` (solver class, ``solver="pypardiso"``) keeps one
    PARDISO instance whose reordering and symbolic analysis are reused while
    the reduced sparsity pattern is unchanged; only the numerical
    factorization and solve are repeated. ``pardiso_threads`` scopes the MKL
    thread count of the PARDISO calls (an integer, ``"all"``, or ``None`` to
    keep the current setting).
    """

    diffusion: Any = 1.0
    advection_stabilization: Any = None
    diffusion_stabilization: Any = "global_length"
    diffusion_penalty_constant: float = 1.0
    solver: str | None = "pypardiso"
    preconditioner: Any = None
    solver_rtol: float = 1e-13
    solver_atol: float = 0.0
    maxiter: int | None = None
    scale_system: bool | Literal["none", "left", "symmetric"] = True
    ilu_drop_tol: float = 1e-10
    ilu_fill_factor: float = 35.0
    ilu_failure: Literal["raise", "none"] = "raise"
    ilu_permc_spec: str = "COLAMD"
    cupyx_solver: str = "bicgstab"
    amgx_config: dict | None = None
    initial_guess: np.ndarray | None = None
    assembly_backend: AssemblyBackend = "numba"
    reconstruction_backend: ReconstructionBackend = "auto"
    postprocessing_backend: PostprocessingBackend = "auto"
    boundary_mode: Literal["eliminate"] = "eliminate"
    trace_basis: Literal["legacy-lagrange", "legendre-modal"] = "legacy-lagrange"
    raw_matrix_format: Literal["coo", "csr", "bsr"] = "csr"
    raw_block_size: int | Literal["auto"] = "auto"
    cache_local_factors: Literal["none", "schur-lu", "schur-lu+mass"] = "schur-lu+mass"
    hdg_postprocess: PostprocessMode = "both"
    flux_postprocess_space: FluxPostprocessSpace = "l2_closest"
    materialize_host_solution: bool = True
    amgx_reuse: Literal["none", "solver", "preconditioner"] = "none"
    amgx_refresh_interval: int = 20
    amgx_refresh_iteration_growth: float = 2.0
    reuse_static_coefficients: bool = True
    numba_reuse_local_columns: bool = False
    pardiso_reuse_analysis: bool = False
    pardiso_threads: int | str | None = None
    verbose: bool | int = True

    def with_overrides(self, **overrides) -> "AdvectionDiffusionReactionHDGOptions":
        """Return an immutable options copy with validated overrides."""
        if not overrides:
            return self
        valid = {field.name for field in fields(type(self))}
        unknown = sorted(set(overrides) - valid)
        if unknown:
            raise TypeError(
                "unknown advection-diffusion-reaction solver option(s): " + ", ".join(unknown)
            )
        return replace(self, **overrides)


def _resolve_stage_backends(
        assembly_backend: str,
        reconstruction_backend: str,
        postprocessing_backend: str,
        postprocess_mode: str,
        flux_postprocess_space: str,
) -> tuple[str, str]:
    """Resolve and validate independently selectable ADR execution stages."""
    reconstruction = str(reconstruction_backend).lower().replace("_", "-")
    postprocessing = str(postprocessing_backend).lower().replace("_", "-")
    if reconstruction == "auto":
        reconstruction = "raw-cuda" if assembly_backend == "raw-cuda" else assembly_backend
    if reconstruction not in {"numpy", "numba", "raw-cuda"}:
        raise ValueError("reconstruction_backend must be 'auto', 'numpy', 'numba', or 'raw-cuda'")
    if assembly_backend == "raw-cuda" and reconstruction != "raw-cuda":
        raise NotImplementedError("raw-cuda ADR assembly currently requires reconstruction_backend='raw-cuda'")
    if assembly_backend != "raw-cuda" and reconstruction == "raw-cuda":
        raise NotImplementedError("raw-cuda ADR reconstruction currently requires assembly_backend='raw-cuda'")
    if postprocessing == "auto":
        postprocessing = (
            "cupy"
            if assembly_backend == "raw-cuda"
            else "numba"
        )
    if postprocessing not in {"numba", "cupy"}:
        raise NotImplementedError(
            "ADR supports postprocessing_backend='numba' or 'cupy' "
            "('auto' selects one of them)"
        )
    if postprocess_mode == "none":
        postprocessing = "none"
    return reconstruction, postprocessing


def _detailed_logging(verbose: bool | int) -> bool:
    """Return whether ADR backend micro-timings are printed (levels 2 and 3)."""
    return _verbosity_level(verbose) >= 2


def _solver_verbosity(verbose: bool | int) -> int:
    """Map ADR levels to linear-solver levels.

    Level 3 prints everything: the solver layers print detailed timings only at
    their level 2 or 4, and iteration tables only from level 3.
    """
    level = _verbosity_level(verbose)
    return 4 if level >= 3 else level


def _timed_substep(label: str, verbose: bool | int, function):
    """Time one indented level-2 substep of a multiline ADR stage."""
    return _timed_call(f"  {label}", int(_detailed_logging(verbose)), function)


_NON_TIME_SUFFIXES = ("block_size", "shared_bytes", "batch_columns", "factor_bytes")


def _print_timing_details(title: str, timings: dict[str, float] | None, verbose: bool | int) -> None:
    """Print a backend timing breakdown and its launch settings at level 2 and above."""
    if not _detailed_logging(verbose) or not timings:
        return
    seconds, settings = [], []
    for key, value in timings.items():
        if key.endswith(".elements"):
            continue
        if key.endswith(_NON_TIME_SUFFIXES):
            settings.append(f"{key.rsplit('.', 1)[-1]}={int(value):,}")
        else:
            seconds.append(f"{key}={_format_seconds(float(value))}")
    if seconds:
        print(f"  {title}: " + ", ".join(seconds), flush=True)
    if settings:
        print("  launch settings: " + ", ".join(settings), flush=True)


def _print_diffusion_structure(counts: dict[str, int] | None, verbose: bool | int) -> None:
    """Print the per-element diffusion fast-path classification at level 2."""
    if _detailed_logging(verbose) and counts:
        used = ", ".join(f"{name}={int(count):,}" for name, count in counts.items() if count)
        print(f"  diffusion structure: {used}", flush=True)


def _reported_postprocessing_backend(backend: str, mode: str) -> str:
    """Report the execution backend shared by both recovery stages."""
    return backend


def _positive_scalar_diffusion(diffusion) -> bool:
    """Return whether diffusion is an exactly isotropic positive constant."""
    from hdgfem.hdg.stabilization import constant_isotropic_diffusivity
    try:
        value = constant_isotropic_diffusivity(diffusion)
    except (NotImplementedError, TypeError, ValueError):
        return False
    return bool(np.isfinite(value) and value > 0.0)


def _beta_field(beta, space: DGSpace) -> VectorDGField:
    """Normalize beta to a two-component DG field for postprocessing."""
    if isinstance(beta, VectorDGField):
        return beta
    return hdg.as_vector_field(beta, space)


def _adr_postprocess_samples(
        beta,
        prepared: ADRPreparedData,
        space: DGSpace,
        post_space: DGSpace,
        advection_stabilization,
        *, xp=np,
) -> tuple[Any, Any, Any]:
    """Sample beta and total stabilization on degree-p+1 quadrature rules."""
    if xp is np:
        coefficients = lambda field: field.coeffs
    else:
        from hdgfem.core.device import as_cupy_coefficients, as_cupy_space
        coefficients = lambda field: as_cupy_coefficients(field, as_cupy_space(field.space))
    qpost = post_space.quad_data
    nqf = qpost.weights_JGL.size
    face_points = qpost.pts_fc.reshape(-1, 2)
    if isinstance(beta, ElementCoefficient):
        # Element-local beta is evaluated directly on the recovery quadrature.
        beta_volume = beta.values_at_ref(qpost.Krf_quads, xp=xp, t=prepared.sample_time)
        beta_face = beta.face_values_at_ref(qpost.pts_fc, xp=xp, t=prepared.sample_time)
    else:
        beta_h = _beta_field(beta, space)
        beta_volume = xp.empty((space.mesh.num_tri, qpost.Krf_w.size, 2), dtype=xp.float64)
        beta_face = xp.empty((space.mesh.num_tri, 3, nqf, 2), dtype=xp.float64)
        for component, field in enumerate(beta_h.components):
            volume_basis = xp.asarray(field.space.basis_at(qpost.Krf_quads))
            face_basis = xp.asarray(field.space.basis_at(face_points)).reshape(
                nqf, 3, field.space.el_dof
            ).transpose(1, 2, 0)
            beta_volume[..., component] = coefficients(field) @ volume_basis.T
            beta_face[..., component] = xp.einsum(
                "Ki,fiq->Kfq", coefficients(field), face_basis, optimize=True
            )

    normals = xp.asarray(space.mesh.normals)
    beta_n = (
        beta_face[..., 0] * normals[..., 0, None]
        + beta_face[..., 1] * normals[..., 1, None]
    )
    # Upwind-family policies (None, ScaledUpwind, "lax-friedrichs",
    # "conflict-averaged-upwind") use the same rule as ADR assembly:
    # factor * |beta.n|, with the conflict-averaged interior repair.
    from hdgfem.hdg.stabilization import effective_advection_normal_flux, upwind_factor

    factor = upwind_factor(advection_stabilization)
    if factor is not None:
        if xp is np:
            mesh = space.mesh
        else:
            from hdgfem.core.device import as_cupy_space
            mesh = as_cupy_space(space).mesh
        tau_adv = factor * xp.abs(
            effective_advection_normal_flux(beta_n, mesh, advection_stabilization, xp=xp)
        )
    elif np.isscalar(advection_stabilization) and not isinstance(advection_stabilization, str):
        tau_adv = xp.full_like(beta_n, float(advection_stabilization))
    elif isinstance(advection_stabilization, DGField):
        tau_basis = xp.asarray(advection_stabilization.space.basis_at(face_points)).reshape(
            nqf, 3, advection_stabilization.space.el_dof
        ).transpose(1, 2, 0)
        tau_adv = xp.einsum(
            "Ki,fiq->Kfq", coefficients(advection_stabilization), tau_basis, optimize=True
        )
    elif callable(advection_stabilization):
        mapped = xp.asarray(space.mesh.map_reference_points(face_points))
        raw = xp.asarray(
            advection_stabilization(mapped[..., 0], mapped[..., 1]), dtype=xp.float64
        )
        if raw.ndim == 0:
            tau_adv = xp.broadcast_to(raw, beta_n.shape)
        else:
            tau_adv = xp.broadcast_to(raw, (space.mesh.num_tri, 3 * nqf)).reshape(
                space.mesh.num_tri, nqf, 3
            ).transpose(0, 2, 1).copy()
    else:
        raw = xp.asarray(advection_stabilization, dtype=xp.float64)
        if raw.shape == space.shape:
            tau_basis = xp.asarray(space.basis_at(face_points)).reshape(
                nqf, 3, space.el_dof
            ).transpose(1, 2, 0)
            tau_adv = xp.einsum("Ki,fiq->Kfq", raw, tau_basis, optimize=True)
        elif raw.shape == (space.mesh.num_tri, 3):
            tau_adv = xp.broadcast_to(raw[:, :, None], beta_n.shape)
        elif raw.shape == beta_n.shape:
            tau_adv = raw
        else:
            raise ValueError(
                "ADR postprocessing needs advection stabilization as None, scalar, "
                "callable, DGField, DG coefficients, per-face constants, or values "
                "on its degree-p+1 face quadrature"
            )
    from hdgfem.assembly.advection_diffusion_reaction import diffusion_stabilization_on_trace
    tau_diff = diffusion_stabilization_on_trace(
        prepared, space, post_space.trace_space("bernstein"), device=xp is not np)
    tau_total = tau_adv + xp.asarray(tau_diff)
    if xp.any(~xp.isfinite(tau_total)):
        raise ValueError("ADR postprocessing stabilization must be finite")
    return (
        xp.ascontiguousarray(beta_volume),
        xp.ascontiguousarray(beta_face),
        xp.ascontiguousarray(tau_total),
    )


def _project_total_flux(
        local_unknowns: np.ndarray,
        prepared: ADRPreparedData,
        space: DGSpace,
) -> VectorDGField:
    """Project q_h plus beta_h u_h into the degree-p vector DG space."""
    xp = np
    if hasattr(local_unknowns, "__cuda_array_interface__"):
        from hdgfem.runtime.optional import require_cupy
        from hdgfem.core.device import field_from_cupy_coefficients
        xp = require_cupy()
    blocks = local_unknowns.reshape(space.mesh.num_tri, 3, space.el_dof)
    basis = xp.asarray(space.quad_data.bas_of_quads)
    beta_values = xp.asarray(prepared.beta_values)
    weighted_phi = xp.asarray(space.quad_data.weighted_phi)
    mass_inverse = xp.asarray(space.quad_data.MKrf_inv)
    u = blocks[:, 0] @ basis
    qx = blocks[:, 1] @ basis
    qy = blocks[:, 2] @ basis
    values_x = qx + beta_values[..., 0] * u
    values_y = qy + beta_values[..., 1] * u
    coeffs_x = (values_x @ weighted_phi) @ mass_inverse
    coeffs_y = (values_y @ weighted_phi) @ mass_inverse
    if xp is not np:
        return VectorDGField(tuple(field_from_cupy_coefficients(space, c, name="q_h_plus_beta_u_h")
                                   for c in (coeffs_x, coeffs_y)), name="q_h_plus_beta_u_h")
    return (space * space).field((coeffs_x, coeffs_y), name="q_h_plus_beta_u_h")


def _adr_recovery_cache(space, trace_ref, *, want_flux=False):
    """Reuse geometry/reference recovery tables, never PDE coefficient factors."""
    cache = _build_hdg_postprocess_cache(
        space, trace_ref, want_primal=False, want_flux=want_flux,
        cache=getattr(space, '_adr_recovery_cache', None))
    space._adr_recovery_cache = cache
    return cache


def _postprocess_total_flux(
        local_unknowns: np.ndarray,
        trace: np.ndarray,
        beta,
        prepared: ADRPreparedData,
        space: DGSpace,
        trace_ref,
        advection_stabilization,
        flux_postprocess_space="l2_closest",
        postprocessing_backend="numba",
) -> VectorDGField:
    """Recover the total flux by full-space or RT_p normal-moment matching."""
    from hdgfem.kernels.advection_diffusion_reaction_fused import (
        solve_adr_total_flux_postprocess_kernel,
    )
    from hdgfem.solvers.diffusion_reaction import _trace_basis_at

    xp = np
    if postprocessing_backend == "cupy":
        from hdgfem.runtime.optional import require_cupy
        xp = require_cupy()
    local_unknowns = xp.asarray(local_unknowns)
    flux_space = _normalize_flux_postprocess_space(flux_postprocess_space)
    cache = _adr_recovery_cache(
        space,
        trace_ref,
        want_flux=flux_space == "l2_closest" and xp is np,
    )
    if flux_space == "l2_closest" and xp is np and (
        cache.flux_ainv_constraint_t is None or cache.flux_schur_lu is None
    ):
        raise RuntimeError("ADR total-flux postprocess factorization is unavailable")
    post = cache.post_space
    qpost = post.quad_data
    nqf = qpost.weights_JGL.size
    face_points = qpost.pts_fc.reshape(-1, 2)
    base_face = xp.asarray(space.basis_at(face_points)).reshape(
        nqf, 3, space.el_dof
    ).transpose(1, 2, 0)
    post_face = xp.asarray(qpost.bas_of_bd_quads)
    trace_basis = xp.asarray(_trace_basis_at(trace_ref, qpost.quads_JGL))
    if xp is np:
        local_trace = trace_ref.element_coefficients(trace).reshape(
            space.mesh.num_tri, 3, trace_ref.edg_dof)
    else:
        from hdgfem.backends.cupy import element_traces_cupy
        local_trace = element_traces_cupy(trace, space, trace_space=trace_ref).reshape(
            space.mesh.num_tri, 3, trace_ref.edg_dof)
    blocks = local_unknowns.reshape(space.mesh.num_tri, 3, space.el_dof)
    u_face = xp.einsum("Ki,fiq->Kfq", blocks[:, 0], base_face, optimize=True)
    qx_face = xp.einsum("Ki,fiq->Kfq", blocks[:, 1], base_face, optimize=True)
    qy_face = xp.einsum("Ki,fiq->Kfq", blocks[:, 2], base_face, optimize=True)
    hat_face = xp.einsum("Kfa,aq->Kfq", local_trace, trace_basis, optimize=True)

    beta_volume, beta_face, tau_total = _adr_postprocess_samples(
        beta, prepared, space, post, advection_stabilization, xp=xp
    )
    base_volume = xp.asarray(space.basis_at(qpost.Krf_quads))
    u_volume = blocks[:, 0] @ base_volume.T
    qx_volume = blocks[:, 1] @ base_volume.T
    qy_volume = blocks[:, 2] @ base_volume.T
    total_volume_values = xp.ascontiguousarray(
        xp.stack(
            (
                qx_volume + beta_volume[..., 0] * u_volume,
                qy_volume + beta_volume[..., 1] * u_volume,
            ),
            axis=0,
        )
    )
    normals = xp.asarray(space.mesh.normals)
    beta_n = (
        beta_face[..., 0] * normals[..., 0, None]
        + beta_face[..., 1] * normals[..., 1, None]
    )
    numerical = (
        normals[..., 0, None] * qx_face
        + normals[..., 1, None] * qy_face
        + beta_n * hat_face
        + tau_total * (u_face - hat_face)
    )

    if flux_space == "RT_projection":
        return _postprocess_rt_flux_from_samples(
            total_volume_values,
            numerical,
            space,
            post,
            backend=postprocessing_backend,
            name="total_flux_h_star_rt_p",
            materialize_host=postprocessing_backend != "cupy",
        )

    if xp is not np:
        from hdgfem.backends.advection_diffusion_reaction_cupy import postprocess_total_flux_l2_cupy
        return postprocess_total_flux_l2_cupy(total_volume_values, numerical, space, cache)

    base_weighted = qpost.Krf_w[:, None] * base_volume
    base_coeffs = np.empty(
        (2, space.mesh.num_tri, space.el_dof), dtype=np.float64
    )
    base_coeffs[0] = (
        total_volume_values[0] @ base_weighted @ space.quad_data.MKrf_inv
    )
    base_coeffs[1] = (
        total_volume_values[1] @ base_weighted @ space.quad_data.MKrf_inv
    )

    q0x = base_coeffs[0] @ cache.base_to_post_mass.T @ qpost.MKrf_inv
    q0y = base_coeffs[1] @ cache.base_to_post_mass.T @ qpost.MKrf_inv
    q0x_face = xp.einsum("Ki,fiq->Kfq", q0x, post_face, optimize=True)
    q0y_face = xp.einsum("Ki,fiq->Kfq", q0y, post_face, optimize=True)
    current = normals[..., 0, None] * q0x_face + normals[..., 1, None] * q0y_face
    gap = xp.einsum(
        "Kf,Kfq,aq,q->Kfa",
        space.mesh.jacs_el_fc,
        numerical - current,
        qpost.bas1d_of_ref_edg_qds,
        qpost.weights_JGL,
        optimize=True,
    ).reshape(space.mesh.num_tri, -1)
    coeffs = np.empty((2, space.mesh.num_tri, post.el_dof), dtype=np.float64)
    solve_adr_total_flux_postprocess_kernel(
        coeffs,
        xp.ascontiguousarray(base_coeffs),
        xp.ascontiguousarray(gap),
        xp.ascontiguousarray(space.mesh.aff_jacs),
        xp.ascontiguousarray(qpost.MKrf_inv),
        cache.base_to_post_mass,
        cache.interior_low_to_base,
        cache.interior_low_to_post,
        cache.flux_ainv_constraint_t,
        cache.flux_schur_lu,
        cache.flux_schur_pivots,
    )
    return (post * post).field((coeffs[0], coeffs[1]), name="total_flux_h_star")


def _postprocess_primal_from_total_flux(
        local_unknowns: np.ndarray,
        total_flux_star: VectorDGField,
        beta,
        prepared: ADRPreparedData,
        space: DGSpace,
        trace_ref,
        advection_stabilization,
        diffusion,
        postprocessing_backend="numba",
) -> DGField:
    """Recover u_h^* through the coupled ADR local Neumann HDG problem."""
    from hdgfem.kernels.advection_diffusion_reaction_fused import (
        solve_adr_primal_from_total_flux_postprocess_kernel,
    )

    from hdgfem.assembly.diffusion_coefficients import sample_diffusion_tensor, inverse_diffusion_values
    cache = _adr_recovery_cache(space, trace_ref)
    post = cache.post_space
    qpost = post.quad_data
    if postprocessing_backend == "cupy":
        from hdgfem.backends.advection_diffusion_reaction_cupy import postprocess_primal_cupy
        from hdgfem.runtime.optional import require_cupy
        samples = _adr_postprocess_samples(
            beta, prepared, space, post, advection_stabilization, xp=require_cupy())
        return postprocess_primal_cupy(local_unknowns, total_flux_star, space, cache,
                                      samples, diffusion)
    beta_volume, beta_face, tau_total = _adr_postprocess_samples(
        beta, prepared, space, post, advection_stabilization
    )
    total_coeffs = np.ascontiguousarray(
        np.stack(
            (
                total_flux_star.components[0].coeffs,
                total_flux_star.components[1].coeffs,
            ),
            axis=0,
        )
    )
    base_primal = np.ascontiguousarray(
        local_unknowns.reshape(space.mesh.num_tri, 3, space.el_dof)[:, 0]
    )
    coeffs = np.empty(post.shape, dtype=np.float64)
    solve_adr_primal_from_total_flux_postprocess_kernel(
        coeffs,
        base_primal,
        total_coeffs,
        np.ascontiguousarray(space.mesh.aff_jacs),
        np.ascontiguousarray(space.mesh.inv_aff_mats_t),
        np.ascontiguousarray(space.mesh.jacs_el_fc),
        np.ascontiguousarray(space.mesh.normals),
        np.ascontiguousarray(qpost.Krf_w),
        np.ascontiguousarray(qpost.phi),
        np.ascontiguousarray(qpost.gphi),
        np.ascontiguousarray(qpost.weights_JGL),
        np.ascontiguousarray(qpost.bas_of_bd_quads),
        np.ascontiguousarray(qpost.bas1d_of_ref_edg_qds),
        beta_volume,
        beta_face,
        tau_total,
        cache.mean_base,
        inverse_diffusion_values(sample_diffusion_tensor(diffusion, post)),
    )
    return post.field(coeffs, name="u_h_star")


def solve_advection_diffusion_reaction_hdg(
        source,
        beta,
        reaction,
        boundary_condition,
        space: DGSpace,
        *,
        options: AdvectionDiffusionReactionHDGOptions | None = None,
        cache: dict | None = None,
        **option_overrides,
) -> AdvectionDiffusionReactionResult:
    r"""Solve ``div(beta*u + q) + r*u=f``, ``q=-kappa*grad(u)`` by HDG.

    ``cache`` is the cross-solve state owned by
    :class:`AdvectionDiffusionReactionHDGSolver` (raw-CUDA static data and
    persistent AMGX solvers); one-shot calls leave it ``None``.
    """
    opts = (options or AdvectionDiffusionReactionHDGOptions()).with_overrides(**option_overrides)
    backend = str(opts.assembly_backend).lower().replace("_", "-")
    if backend not in {"numpy", "numba", "raw-cuda"}:
        raise ValueError("assembly_backend must be 'numpy', 'numba', or 'raw-cuda'")
    if opts.amgx_reuse not in {"none", "solver", "preconditioner"}:
        raise ValueError("amgx_reuse must be 'none', 'solver', or 'preconditioner'")
    if opts.amgx_reuse != "none" and (backend != "raw-cuda" or cache is None):
        raise ValueError("amgx_reuse needs assembly_backend='raw-cuda' and the reusable "
                         "AdvectionDiffusionReactionHDGSolver, which owns the cache")
    if opts.pardiso_reuse_analysis and (backend == "raw-cuda" or cache is None
                                        or str(opts.solver).lower() != "pypardiso"):
        raise ValueError("pardiso_reuse_analysis needs a host assembly backend, solver='pypardiso' and the "
                         "reusable AdvectionDiffusionReactionHDGSolver, which owns the cache")
    if int(opts.amgx_refresh_interval) < 1 or not float(opts.amgx_refresh_iteration_growth) >= 1.:
        raise ValueError("amgx_refresh_interval must be >= 1 and amgx_refresh_iteration_growth >= 1")
    if opts.boundary_mode != "eliminate":
        raise ValueError("stationary ADR currently requires boundary_mode='eliminate'")
    if opts.trace_basis not in {"legacy-lagrange", "legendre-modal"}:
        raise ValueError("trace_basis must be 'legacy-lagrange' or 'legendre-modal'")
    post_mode = _normalize_hdg_postprocess_mode(opts.hdg_postprocess)
    flux_postprocess_space = _normalize_flux_postprocess_space(
        opts.flux_postprocess_space
    )
    reconstruction_backend, postprocessing_backend = _resolve_stage_backends(
        backend,
        opts.reconstruction_backend,
        opts.postprocessing_backend,
        post_mode,
        flux_postprocess_space,
    )
    if (backend == "raw-cuda" and not opts.materialize_host_solution
            and postprocessing_backend == "numba" and post_mode != "none"):
        raise ValueError("materialize_host_solution=False requires CuPy ADR postprocessing; "
                         "select postprocessing_backend='auto' or 'cupy'")

    scalar_diffusion = _positive_scalar_diffusion(opts.diffusion)
    if scalar_diffusion and not np.isscalar(opts.diffusion):
        from hdgfem.hdg.stabilization import constant_isotropic_diffusivity
        opts = opts.with_overrides(diffusion=constant_isotropic_diffusivity(opts.diffusion))
    from hdgfem.backends.capabilities import validate_advection_diffusion_backend_configuration

    validate_advection_diffusion_backend_configuration(
        operation="solve",
        assembly_backend=backend,
        solver=opts.solver,
        cupyx_solver=opts.cupyx_solver,
        boundary_mode=opts.boundary_mode,
        trace_basis=opts.trace_basis,
        postprocess_mode=post_mode,
        scalar_diffusion=scalar_diffusion,
    )
    trace_ref = space.trace_space(opts.trace_basis)
    verbosity = _verbosity_level(opts.verbose)
    detailed = _detailed_logging(verbosity)
    if verbosity:
        print("\n----- DG FEM Advection-Diffusion-Reaction HDG Solve -----", flush=True)
    if detailed:
        print(
            f"  backends: assembly={backend}, reconstruction={reconstruction_backend}, "
            f"postprocessing={postprocessing_backend}; trace basis={opts.trace_basis}; "
            f"triangles={space.mesh.num_tri:,}, p={space.order}",
            flush=True,
        )
    total_start = time.perf_counter()
    if backend == "raw-cuda":
        # Device assembly samples its own coefficients on the GPU, timed as the
        # first assembly sub-stage; there is no host preparation stage.
        from hdgfem.backends.advection_diffusion_reaction_raw_cuda import (
            assemble_projected_adr_trace_system_eliminated_raw_cuda,
        )
        return assemble_projected_adr_trace_system_eliminated_raw_cuda(
            source,
            beta,
            reaction,
            boundary_condition,
            space,
            options=opts.with_overrides(
                postprocessing_backend=postprocessing_backend,
                flux_postprocess_space=flux_postprocess_space,
            ),
            trace_space=trace_ref,
            total_start=total_start,
            cache=cache,
        )

    host_static = None
    host_static_reused = False
    if cache is not None and opts.reuse_static_coefficients:
        from hdgfem.backends.advection_diffusion_reaction_raw_cuda import _static_cache_key
        key = ("host",) + _static_cache_key(space, trace_ref, opts)
        host_static = cache.get("host_static")
        if host_static is None or host_static.get("key") != key:
            host_static = cache["host_static"] = {"key": key}
        host_static_reused = len(host_static) > 1

    def prepare():
        """Sample coefficients, stabilization and the diffusion classification."""
        prepared_data, _ = _timed_substep("sampling coefficients and stabilization", verbosity, lambda: prepare_adr_data(
            source,
            reaction,
            beta,
            space,
            diffusion=opts.diffusion,
            advection_stabilization=opts.advection_stabilization,
            diffusion_stabilization=opts.diffusion_stabilization,
            diffusion_penalty_constant=opts.diffusion_penalty_constant,
            trace_space=trace_ref,
            # The Numba kernels build the face tables per element from tau/gamma samples.
            dense_local_matrices=backend == "numpy" or reconstruction_backend == "numpy",
            static=host_static,
        ))
        tensor_data = None
        if backend == "numba" or reconstruction_backend == "numba":
            from hdgfem.assembly.diffusion_coefficients import prepare_diffusion
            tensor_data = None if host_static is None else host_static.get("diffusion")
            if tensor_data is None:
                tensor_data, _ = _timed_substep(
                    "classifying diffusion tensor", verbosity, lambda: prepare_diffusion(opts.diffusion, space))
                if host_static is not None:
                    host_static["diffusion"] = tensor_data
        return prepared_data, tensor_data

    (prepared, diffusion_data), preparation = _timed_call(
        "preparing ADR coefficient data", verbosity, prepare, multiline=detailed)
    _print_diffusion_structure(None if diffusion_data is None else diffusion_data.counts, verbosity)

    local_solver = None
    numba_columns = None
    details: dict[str, float] = {"host.static_reused": float(host_static_reused)}
    start = time.perf_counter()
    if backend == "numpy":
        assembled, _ = _timed_call("assembling reduced global trace system (numpy)", verbosity, lambda: assemble_numpy(
            prepared,
            boundary_condition,
            space,
            diffusion=opts.diffusion,
            trace_space=trace_ref,
        ))
        trace_system = assembled.trace_system
        reduction = assembled.reduction
        local_solver = assembled.local_solver
    elif backend == "numba":
        from hdgfem.backends.advection_diffusion_reaction_numba import (
            assemble_projected_adr_trace_system_eliminated_numba,
        )
        column_buffer = None
        if opts.numba_reuse_local_columns and reconstruction_backend == "numba":
            shape = (space.mesh.num_tri, 3 * space.el_dof, 3 * trace_ref.edg_dof + 1)
            column_buffer = None if cache is None else cache.get("numba_columns")
            if column_buffer is None or column_buffer.shape != shape:
                column_buffer = np.empty(shape, dtype=np.float64)
                if cache is not None:
                    cache["numba_columns"] = column_buffer
        assembled, _ = _timed_call(
            "assembling reduced global trace system (numba)",
            verbosity,
            lambda: assemble_projected_adr_trace_system_eliminated_numba(
                prepared,
                boundary_condition,
                space,
                trace_space=trace_ref,
                diffusion=opts.diffusion,
                diffusion_data=diffusion_data,
                local_columns=column_buffer,
            ),
        )
        numba_columns = assembled.local_columns
        trace_system = assembled.trace_system
        reduction = assembled.reduction
        details.update({f"numba.{key}": value for key, value in assembled.timings.items()})
        _print_timing_details("numba assembly timings", assembled.timings, verbosity)
    trace_assembly = time.perf_counter() - start
    if detailed:
        print(
            f"  reduced trace system: {trace_system.rhs.size:,} free trace dofs, "
            f"{trace_system.data.size:,} COO entries",
            flush=True,
        )

    use_pardiso_cache = opts.pardiso_reuse_analysis and str(opts.solver).lower() == "pypardiso"
    pardiso_solver = None
    if use_pardiso_cache:
        from hdgfem.linalg.pardiso_runtime import ReusablePardisoSolver
        pardiso_solver = cache.get("pardiso")
        if pardiso_solver is None:
            pardiso_solver = cache["pardiso"] = ReusablePardisoSolver()
        pardiso_solver.threads = opts.pardiso_threads

    def global_solve():
        """Solve the reduced system, through the cached PARDISO instance when enabled."""
        if pardiso_solver is not None:
            return pardiso_solver.solve_coo(trace_system.rows, trace_system.cols, trace_system.data,
                                            trace_system.rhs, trace_system.rhs.size, rtol=opts.solver_rtol,
                                            atol=opts.solver_atol, raise_on_nonconvergence=True)
        from contextlib import nullcontext
        from hdgfem.linalg.pardiso_runtime import pardiso_thread_limit
        context = (pardiso_thread_limit(opts.pardiso_threads)
                   if opts.pardiso_threads is not None and str(opts.solver).lower() == "pypardiso" else nullcontext())
        with context:
            return solve_global_system(
                trace_system.rows,
                trace_system.cols,
                trace_system.data,
                trace_system.rhs,
                trace_system.rhs.size,
                solver=opts.solver,
                preconditioner=opts.preconditioner,
                initial_guess=opts.initial_guess,
                rtol=opts.solver_rtol,
                atol=opts.solver_atol,
                maxiter=opts.maxiter,
                ilu_drop_tol=opts.ilu_drop_tol,
                ilu_fill_factor=opts.ilu_fill_factor,
                ilu_failure=opts.ilu_failure,
                ilu_permc_spec=opts.ilu_permc_spec,
                cupyx_solver=opts.cupyx_solver,
                amgx_config=opts.amgx_config,
                scale_system=opts.scale_system,
                raise_on_nonconvergence=True,
                verbose=_solver_verbosity(verbosity),
            )

    solve_result, solve_seconds = _timed_call(
        "solving global system", verbosity, global_solve, multiline=verbosity >= 1)
    if pardiso_solver is not None:
        details["host.pardiso_analysis_reused"] = float(solve_result.pardiso_analysis_reused)
    details["numba.local_columns_reused"] = float(numba_columns is not None)
    trace = expand_known_dofs(solve_result.x, reduction)

    def reconstruct():
        """Recover mixed local fields and project the total flux."""
        nonlocal local_solver
        if reconstruction_backend == "numba":
            from hdgfem.backends.advection_diffusion_reaction_numba import (
                reconstruct_projected_adr_local_unknowns_numba,
            )
            unknowns = reconstruct_projected_adr_local_unknowns_numba(
                trace,
                prepared,
                space,
                trace_space=trace_ref,
                diffusion=opts.diffusion,
                diffusion_data=diffusion_data,
                local_columns=numba_columns,
            )
        else:
            if local_solver is None:
                local_solver, _ = _timed_substep(
                    "building dense local solvers (numpy)", verbosity,
                    lambda: local_solvers_numpy(prepared, space, diffusion=opts.diffusion))
            source_block = np.zeros((space.mesh.num_tri, 3 * space.el_dof), dtype=np.float64)
            source_block[:, :space.el_dof] = prepared.source_rhs
            unknowns = hdg.reconstruct_local_unknowns(
                trace,
                source_block,
                local_solver,
                prepared.element_boundary,
                space,
                trace_space=trace_ref,
            )
        primal, diffusive = split_diffusion_unknowns(unknowns, space)
        total, _ = _timed_substep(
            "projecting total flux q_h + beta u_h", verbosity,
            lambda: _project_total_flux(unknowns, prepared, space))
        return unknowns, primal, diffusive, total

    (local_unknowns, field, flux, total_flux), reconstruction = _timed_call(
        f"reconstructing local fields ({reconstruction_backend})", verbosity, reconstruct, multiline=detailed)

    def postprocess():
        """Recover the requested total flux and primal postprocessed fields."""
        recovered_flux = recovered_field = None
        if post_mode != "none":
            recovered_flux, _ = _timed_substep(
                f"recovering total flux ({flux_postprocess_space})",
                verbosity,
                lambda: _postprocess_total_flux(
                    local_unknowns,
                    trace,
                    beta,
                    prepared,
                    space,
                    trace_ref,
                    opts.advection_stabilization,
                    flux_postprocess_space,
                    postprocessing_backend,
                ),
            )
        if post_mode in {"primal", "both"}:
            recovered_field, _ = _timed_substep(
                "recovering primal field",
                verbosity,
                lambda: _postprocess_primal_from_total_flux(
                    local_unknowns,
                    recovered_flux,
                    beta,
                    prepared,
                    space,
                    trace_ref,
                    opts.advection_stabilization,
                    opts.diffusion,
                    postprocessing_backend,
                ),
            )
        if postprocessing_backend == "cupy":
            from hdgfem.runtime.optional import require_cupy

            def synchronize():
                """Materialize requested host outputs and drain the device stream."""
                if opts.materialize_host_solution:
                    outputs = (recovered_field, *(recovered_flux.components if post_mode in {"flux", "both"} else ()))
                    for output in outputs:
                        if output is not None:
                            _ = output.coeffs
                require_cupy().cuda.get_current_stream().synchronize()

            _timed_substep("synchronizing device outputs", verbosity, synchronize)
        return recovered_flux, recovered_field

    (total_flux_star, post_field), postprocessing = _timed_call(
        f"postprocessing ({post_mode}, {postprocessing_backend})",
        verbosity if post_mode != "none" else 0,
        postprocess,
        multiline=detailed,
    )
    post_flux = total_flux_star if post_mode in {"flux", "both"} else None
    timings = AdvectionDiffusionReactionTimings(
        preparation=preparation,
        trace_assembly=trace_assembly,
        solve=solve_seconds,
        reconstruction=reconstruction,
        postprocessing=postprocessing,
        total=time.perf_counter() - total_start,
        details=details,
    )
    return AdvectionDiffusionReactionResult(
        field=field,
        flux=flux,
        total_flux=total_flux,
        trace=trace,
        timings=timings,
        postprocessed_field=post_field,
        postprocessed_flux=post_flux,
        local_unknowns=local_unknowns,
        matrix_rows=trace_system.rows,
        matrix_cols=trace_system.cols,
        matrix_data=trace_system.data,
        rhs=trace_system.rhs,
        boundary_trace=trace_system.boundary_trace,
        reduction=reduction,
        local_solver=local_solver,
        element_boundary_mats=prepared.element_boundary,
        tau_advection=prepared.tau_advection,
        tau_diffusion=prepared.tau_diffusion,
        diffusion_structure=None if diffusion_data is None else diffusion_data.counts,
        beta_dot_normal=prepared.beta_dot_normal,
        assembly_backend=backend,
        reconstruction_backend=reconstruction_backend,
        postprocessing_backend=_reported_postprocessing_backend(
            postprocessing_backend, post_mode
        ),
        global_solve_result=solve_result,
    )


_UNSET = object()


class AdvectionDiffusionReactionHDGSolver:
    """Reusable public facade for stationary ADR solves.

    Problem data can change between solves on the same space. On the raw-CUDA
    path the instance keeps time-independent diffusion data, the sparsity
    pattern and (with ``amgx_reuse``) persistent AMGX solvers across solves;
    ``clear_cache``/``close`` release them.
    """

    def __init__(
            self,
            space: DGSpace,
            source: Any = _UNSET,
            beta: Any = _UNSET,
            reaction: Any = _UNSET,
            boundary_condition: Any = _UNSET,
            *,
            options: AdvectionDiffusionReactionHDGOptions | None = None,
            **option_overrides,
    ):
        """Initialize the reusable solver and optional complete problem bundle."""
        self.space = space
        self.options = (options or AdvectionDiffusionReactionHDGOptions()).with_overrides(
            **option_overrides
        )
        provided = (source is not _UNSET, beta is not _UNSET, reaction is not _UNSET, boundary_condition is not _UNSET)
        if any(provided) and not all(provided):
            raise ValueError("source, beta, reaction, and boundary_condition must be provided together")
        self.source = self.beta = self.reaction = self.boundary_condition = _UNSET
        self.result: AdvectionDiffusionReactionResult | None = None
        self._raw_cache: dict = {}
        if all(provided):
            self.set_problem(source, beta, reaction, boundary_condition)

    def set_problem(self, source, beta, reaction, boundary_condition):
        """Replace the complete stationary ADR problem data."""
        self.source = source
        self.beta = beta
        self.reaction = reaction
        self.boundary_condition = boundary_condition
        self.result = None
        return self

    def set_source(self, source):
        """Replace the source and invalidate the stored result."""
        self.source = source
        self.result = None
        return self

    def set_beta(self, beta):
        """Replace the advection field and invalidate the stored result."""
        self.beta = beta
        self.result = None
        return self

    def set_reaction(self, reaction):
        """Replace the reaction coefficient and invalidate the stored result."""
        self.reaction = reaction
        self.result = None
        return self

    def set_boundary_condition(self, boundary_condition):
        """Replace the all-Dirichlet data and invalidate the stored result."""
        self.boundary_condition = boundary_condition
        self.result = None
        return self

    def with_options(self, **overrides):
        """Persist validated option overrides and invalidate the stored result."""
        self.options = self.options.with_overrides(**overrides)
        self.result = None
        return self

    def clear_cache(self):
        """Discard the stored result and release cached raw-CUDA data and AMGX solvers."""
        from hdgfem.backends.advection_diffusion_reaction_raw_cuda import close_raw_adr_cache
        close_raw_adr_cache(getattr(self, "_raw_cache", None))
        self.result = None
        return self

    def close(self) -> None:
        """Release persistent device solver state owned by this instance."""
        self.clear_cache()

    def __del__(self):
        """Best-effort release of persistent AMGX solvers."""
        try:
            self.close()
        except Exception:
            pass

    def solve(self, **option_overrides) -> AdvectionDiffusionReactionResult:
        """Solve the currently stored stationary ADR problem."""
        if any(value is _UNSET for value in (self.source, self.beta, self.reaction, self.boundary_condition)):
            raise RuntimeError("no complete advection-diffusion-reaction problem is set")
        if option_overrides:
            self.with_options(**option_overrides)
        self.result = solve_advection_diffusion_reaction_hdg(
            self.source,
            self.beta,
            self.reaction,
            self.boundary_condition,
            self.space,
            options=self.options,
            cache=self._raw_cache,
        )
        return self.result


__all__ = [
    "AdvectionDiffusionReactionHDGOptions",
    "AdvectionDiffusionReactionHDGSolver",
    "AdvectionDiffusionReactionResult",
    "AdvectionDiffusionReactionTimings",
    "solve_advection_diffusion_reaction_hdg",
]
