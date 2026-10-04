r"""HDG solver for scalar diffusion-reaction problems.

This module is the :mod:`hdgfem` rewrite of the legacy ``diff_rea3.py`` path.
It solves

.. math::

    -\nabla\cdot(\kappa\nabla u) + r u = f

with the mixed local unknown vector ``[u_h, q_{x,h}, q_{y,h}]`` and a global
HDG trace unknown.  Identity diffusion uses the scalar fast path; tensor
diffusion uses dense local mixed inverses in NumPy and a projected-coefficient
fused trace assembly path in Numba.
"""

from __future__ import annotations

from hdgfem.runtime.precision import REAL_DTYPE

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields, replace
from numbers import Real
from typing import Any, Literal

import numpy as np

from hdgfem.hdg import condensation as hdg_assembly
from hdgfem.solvers.capabilities import (
    normalize_assembly_backend,
    normalize_solver_backend,
    normalize_trace_basis,
    validate_diffusion_backend_configuration,
)
from hdgfem.hdg.cuda.launch import RawCudaBlockSize, resolve_raw_cuda_block_size
from hdgfem.linalg.reduction import (
    KnownDofReduction,
    eliminate_known_dofs,
    expand_known_dofs,
)
from hdgfem.linalg.results import SolveResult, diagonal_scale_system
from hdgfem.linalg.system import assemble_global_matrix, solve_global_system
from hdgfem.core.space import DGField, DGSpace, VectorDGField
from hdgfem.mixed.stabilization import resolve_diffusion_stabilization

from hdgfem.runtime.logging import _detailed_logging, _timed_call, _verbosity_level
from hdgfem.mixed.coefficients import (
    _project_inverse_diffusion_for_numba,
    normalize_diffusion_stabilization,
)
from hdgfem.mixed.postprocess.flux import (
    FluxPostprocessSpace,
    HDGPostprocessMode,
    _HDGPostprocessCache,
    _normalize_flux_postprocess_space,
    _normalize_hdg_postprocess_mode,
    _postprocess_diffusion_solution,
)
from hdgfem.mixed.coefficients import _diffusion_is_identity
from hdgfem.mixed.local_numpy import (
    _build_res_numba,
    LocalSolverBackend,
    _local_solver_blocks_numpy,
    _local_solver_pre_mats,
    _local_solver_scalar_inverse,
    assemble_diffusion_trace_system,
    diffusion_element_boundary_mats,
    impose_boundary_trace_on_guess,
    local_solvers,
    split_diffusion_unknowns,
)
from hdgfem.hdg.coefficients import _require_same_space_dg_field_for_backend


AssemblyBackend = Literal["numpy", "numba", "auto"]
TraceAssemblyBackend = Literal["numpy", "numba", "cupy", "raw-cuda", "auto"]
PostprocessingBackend = Literal["auto", "numba", "cupy", "raw-cuda"]
LocalFactorCachePolicy = Literal["none", "schur-lu", "schur-cholesky"]
ReturnKey = Literal[
    "result",
    "trace",
    "trace_coeffs",
    "flux",
    "postprocessed_field",
    "postprocessed_flux",
    "local_unknowns",
    "matrix_rows",
    "matrix_cols",
    "matrix_data",
    "local_solver",
    "element_boundary_mats",
    "rhs",
    "solve_matrix_rows",
    "solve_matrix_cols",
    "solve_matrix_data",
    "solve_rhs",
    "boundary_trace",
    "reduction",
    "global_solve_result",
    "timings",
]


_UNSET = object()


@dataclass(frozen=True)
class DiffusionReactionTimings:
    """Wall-clock timings for the main diffusion-reaction solve phases."""

    preparation: float
    local_solver: float
    element_boundary: float
    trace_assembly: float
    solve: float
    reconstruction: float
    total: float
    initial_guess: float = 0.0
    boundary_elimination: float = 0.0
    postprocessing: float = 0.0
    details: dict[str, float] | None = None

    @property
    def assembly(self) -> float:
        """Total setup time excluding the global sparse solve."""
        return (
            self.preparation
            + self.local_solver
            + self.element_boundary
            + self.trace_assembly
            + self.initial_guess
            + self.boundary_elimination
        )


@dataclass(frozen=True)
class DiffusionReactionResult:
    """Container returned by :func:`solve_diffusion_reaction_hdg`."""

    field: DGField
    flux: VectorDGField
    trace: np.ndarray
    timings: DiffusionReactionTimings
    trace_reduced_device: Any = None
    postprocessed_field: DGField | None = None
    postprocessed_flux: VectorDGField | None = None
    flux_postprocess_space: str = "none"
    postprocessing_backend: str = "none"
    local_unknowns: np.ndarray | None = None
    matrix_rows: np.ndarray | None = None
    matrix_cols: np.ndarray | None = None
    matrix_data: np.ndarray | None = None
    rhs: np.ndarray | None = None
    solve_matrix_rows: np.ndarray | None = None
    solve_matrix_cols: np.ndarray | None = None
    solve_matrix_data: np.ndarray | None = None
    solve_rhs: np.ndarray | None = None
    boundary_trace: np.ndarray | None = None
    reduction: Any = None
    local_solver: np.ndarray | None = None
    element_boundary_mats: np.ndarray | None = None
    initial_guess: np.ndarray | None = None
    boundary_mode: Literal["penalty", "eliminate"] = "penalty"
    scale_system: bool | Literal["none", "left", "symmetric"] = True
    assembly_backend: AssemblyBackend = "numpy"
    global_solve_result: SolveResult | None = None
    trace_device: Any = None
    local_unknowns_device: Any = None


@dataclass(frozen=True)
class DiffusionReactionAssemblyResult:
    """Reduced diffusion-reaction trace system assembled without solving it."""

    rows: np.ndarray | None
    cols: np.ndarray | None
    data: np.ndarray
    rhs: np.ndarray
    boundary_trace: np.ndarray
    reduction: Any
    assembly_backend: TraceAssemblyBackend
    matrix_format: Literal["coo", "csr", "bsr"] = "coo"
    indptr: np.ndarray | None = None
    indices: np.ndarray | None = None
    timings: dict[str, float] | None = None


def flux_coefficients(result: DiffusionReactionResult) -> np.ndarray:
    """Return result flux coefficients with shape ``(2, num_elements, el_dof)``.

    The diffusion-reaction solver stores flux as a :class:`VectorDGField`.
    This helper gives assembly code a contiguous component-first array in the
    mixed-HDG ordering ``(q_x, q_y)``.
    """
    coeffs = np.asarray(result.flux.as_component_first(), dtype=REAL_DTYPE)
    if coeffs.shape[0] != 2:
        raise ValueError(f"expected two flux components; got shape {coeffs.shape}")
    return np.ascontiguousarray(coeffs)


@dataclass(frozen=True)
class DiffusionReactionHDGOptions:
    """Configuration for :class:`DiffusionReactionHDGSolver`.

    The dataclass owns numerical coefficients, trace assembly controls, global
    sparse-solver controls, reconstruction/postprocessing choices, and optional
    device-cache policy. It mirrors the keyword-only portion of
    :func:`solve_diffusion_reaction_hdg`; problem data (source, reaction, and
    boundary condition) remain on the solver instance.

    ``assembly_backend`` selects where the trace operator is built, while
    ``solver`` independently selects its sparse inversion backend. Their valid
    combinations are checked against :mod:`hdgfem.solvers.capabilities`
    before coefficient sampling or optional-runtime setup. The
    ``stabilization="global_length"`` is the production default. It resolves
    :class:`GlobalLengthDiffusion` from constant isotropic diffusion and the
    current physical mesh before backend dispatch. Explicit positive scalar or
    incidence-wise stabilization inputs remain supported.
    """

    diffusion: Any = 1.0
    stabilization: Any = "global_length"
    solver: str | None = "BICGSTAB"
    preconditioner: Any = "ilu"
    solver_rtol: float = 1e-13
    solver_atol: float = 0.0
    maxiter: int | None = None
    scale_system: bool | Literal["none", "left", "symmetric"] = True
    petsc_preset: str = "cg_gamg"
    petsc_levels: int | None = None
    petsc_options: dict | None = None
    petsc_divtol: float = 1e4
    petsc_monitor: bool = False
    cupyx_solver: str = "bicgstab"
    amgx_config: dict | None = None
    amgx_retry_attempts: tuple[dict[str, Any], ...] | None = None
    fb_hp_mg_true_residual_every: int = 10
    fb_hp_mg_residual_history: bool = True
    fb_hp_mg_preconditioner_policy: Literal["standard", "fast", "robust"] = "standard"
    cache_device_matrix: bool = True
    cache_local_factors: LocalFactorCachePolicy = "none"
    ilu_drop_tol: float = 1e-10
    ilu_fill_factor: float = 35
    ilu_failure: Literal["raise", "none"] = "raise"
    ilu_permc_spec: str = "COLAMD"
    initial_guess: np.ndarray | None = None
    local_solver_backend: LocalSolverBackend = "numpy"
    assembly_backend: TraceAssemblyBackend = "numpy"
    trace_basis: Literal["legacy-lagrange", "legendre-modal", "bernstein"] = "legacy-lagrange"
    raw_matrix_format: Literal["auto", "coo", "csr", "bsr"] = "auto"
    raw_block_size: RawCudaBlockSize = "auto"
    boundary_penalty: float = 1e20
    boundary_mode: Literal["penalty", "eliminate"] = "penalty"
    hdg_postprocess: HDGPostprocessMode = "none"
    flux_postprocess_space: FluxPostprocessSpace = "l2_closest"
    postprocessing_backend: PostprocessingBackend = "auto"
    verbose: bool | int = True

    def with_overrides(self, **overrides) -> "DiffusionReactionHDGOptions":
        """Return a copy with selected option values replaced."""
        if not overrides:
            return self
        valid = {field.name for field in fields(type(self))}
        unknown = sorted(set(overrides) - valid)
        if unknown:
            raise TypeError(f"unknown diffusion-reaction solver option(s): {', '.join(unknown)}")
        return replace(self, **overrides)

    def as_solve_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for :func:`solve_diffusion_reaction_hdg`."""
        return {field.name: getattr(self, field.name) for field in fields(type(self))}


def _normalize_local_factor_cache_policy(value: str) -> LocalFactorCachePolicy:
    """Normalize the optional persistent element-factor cache policy."""
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized not in {"none", "schur-lu", "schur-cholesky"}:
        raise ValueError("cache_local_factors must be 'none', 'schur-lu', or 'schur-cholesky'")
    return normalized


def _validate_local_factor_cache_configuration(options, backend: str, *, stateful: bool) -> LocalFactorCachePolicy:
    """Validate that persistent Schur factors are used only by their owning path."""
    policy = _normalize_local_factor_cache_policy(options.cache_local_factors)
    if policy == "none":
        return policy
    if not stateful:
        raise ValueError(f"cache_local_factors='{policy}' requires DiffusionReactionHDGSolver")
    if policy == "schur-lu" and backend not in {"numba", "raw-cuda"}:
        raise ValueError("cache_local_factors='schur-lu' requires assembly_backend='numba' or 'raw-cuda'")
    if policy == "schur-cholesky" and backend not in {"numba", "cupy", "raw-cuda"}:
        raise ValueError(
            "cache_local_factors='schur-cholesky' requires assembly_backend='numba', 'cupy' or 'raw-cuda'"
        )
    if backend == "numba" and not _diffusion_is_identity(options.diffusion):
        raise ValueError("Numba local Schur caching currently requires identity diffusion")
    if not options.cache_device_matrix:
        raise ValueError(f"cache_local_factors='{policy}' requires cache_device_matrix=True")
    if options.boundary_mode != "eliminate":
        raise ValueError(f"cache_local_factors='{policy}' requires boundary_mode='eliminate'")
    if backend == "raw-cuda" and str(options.raw_matrix_format).lower() not in {"auto", "csr", "bsr"}:
        raise ValueError(f"cache_local_factors='{policy}' requires raw_matrix_format='csr' or 'bsr'")
    if policy == "schur-cholesky" and (not np.isscalar(options.stabilization) or float(options.stabilization) <= 0.0):
        raise ValueError("cache_local_factors='schur-cholesky' requires strictly positive scalar stabilization")
    return policy


def _resolve_diffusion_postprocessing_backend(
        assembly_backend: str,
        mode: HDGPostprocessMode,
        flux_space: FluxPostprocessSpace,
        requested: str,
) -> str:
    """Resolve the host/CuPy diffusion postprocessing execution path."""
    backend = str(requested).strip().lower().replace("_", "-")
    want_flux = mode in {"flux", "both"}
    if mode == "none":
        return "none"
    if backend == "auto":
        backend = (
            assembly_backend
            if assembly_backend in {"cupy", "raw-cuda"}
            and want_flux
            and (flux_space == "RT_projection" or (assembly_backend == "raw-cuda" and mode == "flux"))
            else "numba"
        )
    if backend not in {"numba", "cupy", "raw-cuda"}:
        raise ValueError(
            "postprocessing_backend must be 'auto', 'numba', 'cupy', or 'raw-cuda'"
        )
    if backend in {"cupy", "raw-cuda"} and not (
        want_flux and (flux_space == "RT_projection" or (backend == "raw-cuda" and mode == "flux"))
    ):
        raise NotImplementedError(
            f"postprocessing_backend={backend!r} supports only flux "
            "postprocessing with flux_postprocess_space='RT_projection', "
            "or flux-only l2_closest with raw-cuda"
        )
    return backend


def _reported_diffusion_postprocessing_backend(
        backend: str,
        mode: HDGPostprocessMode,
) -> str:
    """Report mixed CuPy flux and host-Numba primal recovery explicitly."""
    if backend == "cupy" and mode == "both":
        return "cupy+numba"
    return backend


def _result_with_hdg_postprocessing(
        result: DiffusionReactionResult,
        *,
        postprocessed_field: DGField | None,
        postprocessed_flux: VectorDGField | None,
        flux_postprocess_space: str,
        postprocessing_backend: str,
        elapsed: float,
) -> DiffusionReactionResult:
    """Return ``result`` with optional post-processed fields attached."""
    if postprocessed_field is None and postprocessed_flux is None:
        return result
    timings = replace(
        result.timings,
        postprocessing=result.timings.postprocessing + elapsed,
        total=result.timings.total + elapsed,
    )
    result_values = {field.name: getattr(result, field.name) for field in fields(DiffusionReactionResult)}
    result_values["timings"] = timings
    result_values["postprocessed_field"] = postprocessed_field
    result_values["postprocessed_flux"] = postprocessed_flux
    result_values["flux_postprocess_space"] = flux_postprocess_space
    result_values["postprocessing_backend"] = postprocessing_backend
    if postprocessing_backend == "raw-cuda":
        # Recovery has consumed these transient reconstruction buffers. Keep
        # owning raw/recovered fields and the reduced trace, not duplicate
        # full mixed tables in every accepted BDF history result.
        result_values["local_unknowns_device"] = None
        result_values["trace_device"] = None
    return DiffusionReactionResult(**result_values)


def _host_array(value, *, dtype=None) -> np.ndarray | None:
    """Return a contiguous host NumPy array from a NumPy/CuPy-like input."""
    if value is None:
        return None
    try:
        from hdgfem.runtime.optional import require_cupy

        cupy = require_cupy()
        if isinstance(value, cupy.ndarray):
            value = cupy.asnumpy(value)
    except Exception:
        pass
    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


def _full_boundary_trace_from_compact(boundary_trace, space: DGSpace) -> np.ndarray:
    """Normalize full-edge or compact boundary-edge trace data to full-edge shape."""
    trace = _host_array(boundary_trace, dtype=REAL_DTYPE)
    mesh = space.mesh
    edg_dof = space.quad_data.edg_dof
    full_shape = (mesh.num_edg, edg_dof)
    if trace.shape == full_shape:
        return trace
    compact_shape = (mesh.bnd_edges_inds.size, edg_dof)
    if trace.shape != compact_shape:
        raise ValueError(f"boundary_trace must have shape {full_shape} or {compact_shape}; got {trace.shape}")
    full = np.zeros(full_shape, dtype=REAL_DTYPE)
    full[mesh.bnd_edges_inds] = trace
    return np.ascontiguousarray(full)


def _reduction_from_reduced_trace_system(
        rows: np.ndarray | None,
        cols: np.ndarray | None,
        data: np.ndarray,
        rhs: np.ndarray,
        boundary_trace: np.ndarray,
        space: DGSpace,
) -> KnownDofReduction:
    """Build reduction metadata for an already reduced trace system."""
    mesh = space.mesh
    edg_dof = space.quad_data.edg_dof
    edge_is_free = np.ones(mesh.num_edg, dtype=bool)
    edge_is_free[mesh.bnd_edges_inds] = False
    free_mask = np.repeat(edge_is_free, edg_dof)
    known_mask = ~free_mask
    old_to_new = np.full(mesh.num_edg * edg_dof, -1, dtype=np.int64)
    old_to_new[free_mask] = np.arange(np.count_nonzero(free_mask), dtype=np.int64)
    empty_i = np.empty(0, dtype=np.int64)
    empty_f = np.empty(0, dtype=REAL_DTYPE)
    return KnownDofReduction(
        rows=empty_i if rows is None else np.ascontiguousarray(rows, dtype=np.int64),
        cols=empty_i if cols is None else np.ascontiguousarray(cols, dtype=np.int64),
        data=empty_f if rows is None else np.ascontiguousarray(data, dtype=REAL_DTYPE),
        rhs=np.ascontiguousarray(rhs, dtype=REAL_DTYPE),
        free_mask=np.ascontiguousarray(free_mask),
        known_mask=np.ascontiguousarray(known_mask),
        known_values=np.ascontiguousarray(boundary_trace.ravel(), dtype=REAL_DTYPE),
        old_to_new=np.ascontiguousarray(old_to_new),
    )


def _reduced_result_from_full_trace_system(trace_system, space: DGSpace) -> tuple[hdg_assembly.TraceSystem, KnownDofReduction]:
    """Eliminate boundary dofs from a full trace system."""
    boundary_trace = np.asarray(trace_system.boundary_trace, dtype=REAL_DTYPE)
    known_mask = ~hdg_assembly.free_trace_dofs(space)
    reduction = eliminate_known_dofs(
        trace_system.rows,
        trace_system.cols,
        trace_system.data,
        trace_system.rhs,
        known_mask,
        boundary_trace.ravel(),
    )
    reduced = hdg_assembly.TraceSystem(
        rows=reduction.rows,
        cols=reduction.cols,
        data=reduction.data,
        rhs=reduction.rhs,
        boundary_trace=boundary_trace,
    )
    return reduced, reduction


def _with_implied_options(options: DiffusionReactionHDGOptions, explicit) -> DiffusionReactionHDGOptions:
    """Fill options that the selected solver or assembly admits only one value for.

    ``fb-hp-mg-pcg`` runs on legendre-modal traces without system scaling, and
    raw-CUDA, CuPy and Numba assembly eliminate boundary traces. Explicitly
    passed values are kept, so an incompatible request still fails validation.
    """
    implied = {}
    if normalize_solver_backend(options.solver) == "fb-hp-mg-pcg":
        implied.update(trace_basis="legendre-modal", scale_system=False)
    if normalize_assembly_backend(options.assembly_backend) in {"raw-cuda", "cupy", "numba"}:
        implied["boundary_mode"] = "eliminate"
    implied = {key: value for key, value in implied.items() if key not in explicit}
    return options.with_overrides(**implied) if implied else options


class DiffusionReactionHDGSolver:
    r"""Stateful HDG solver/cache for scalar diffusion-reaction problems.

    The class mirrors :func:`solve_diffusion_reaction_hdg` but stores the space,
    problem data, solver options, and most recent assembled artifacts on one
    object.  With ``assembly_backend="numba"`` the trace system is assembled
    with strongly imposed boundary trace dofs, matching the advection-reaction
    backend's eliminated-boundary convention.

    GPU global solves are selected through ``solver``, independently of the
    assembly backend.  NumPy/CuPy assembly may sample analytic source/reaction
    callables directly; Numba and raw-CUDA assembly require same-space
    :class:`DGField` source/reaction inputs, using ``space.zeros`` or
    ``space.constant`` for exact zero/constant coefficients.  For example,
    ``assembly_backend="numba", solver="amgx"`` or ``solver="cupyx"``
    assembles the reduced trace operator on the host and
    then copies the global sparse matrix to the GPU for inversion.  When
    ``cache_device_matrix=True`` and only the RHS/boundary data changes, the
    cached Numba operator path also reuses the Cupyx device CSR matrix across
    solves.

    Given ``source`` and ``boundary_condition`` without ``reaction``, the
    reaction is zero (pure diffusion, e.g. Poisson). Without an ``options``
    object, options that the chosen solver or assembly admits only one value
    for are implied unless passed: ``solver="fb-hp-mg-pcg"`` uses
    ``trace_basis="legendre-modal"`` and ``scale_system=False``, and raw-CUDA,
    CuPy or Numba assembly uses ``boundary_mode="eliminate"``.
    """

    def __init__(
            self,
            space: DGSpace,
            *,
            source: Any = _UNSET,
            reaction: Any = _UNSET,
            boundary_condition: Callable | object = _UNSET,
            options: DiffusionReactionHDGOptions | None = None,
            **option_overrides,
    ) -> None:
        """Initialize a reusable diffusion-reaction solver for one DG space."""
        self.space = space
        self.options = (options or DiffusionReactionHDGOptions()).with_overrides(**option_overrides)
        if options is None:
            self.options = _with_implied_options(self.options, option_overrides.keys())

        self.source = None
        self.reaction = None
        self.boundary_condition = None
        self._problem_is_set = False
        self._raw_cuda_amgx_retry_solver_cache: dict[Any, Any] = {}

        self.clear_cache()

        if reaction is _UNSET and source is not _UNSET and boundary_condition is not _UNSET:
            reaction = space.zeros()
        provided = (
            source is not _UNSET,
            reaction is not _UNSET,
            boundary_condition is not _UNSET,
        )
        if any(provided):
            if not all(provided):
                raise ValueError("source and boundary_condition must be provided together, "
                                 "with an optional reaction")
            self.set_problem(source, reaction, boundary_condition)

    @property
    def mesh(self):
        """Mesh owned by the current DG space."""
        return self.space.mesh

    @property
    def degree(self) -> int:
        """Polynomial degree of the current solution space."""
        return self.space.order

    @property
    def edg_dof(self) -> int:
        """Number of trace degrees of freedom per mesh edge."""
        return self.space.quad_data.edg_dof

    def _resolved_options(self) -> DiffusionReactionHDGOptions:
        """Return options with built-in stabilization policies lowered for this mesh."""
        stabilization = resolve_diffusion_stabilization(
            self.options.stabilization,
            self.options.diffusion,
            self.space,
        )
        if stabilization is self.options.stabilization:
            return self.options
        return self.options.with_overrides(stabilization=stabilization)

    def with_options(self, **overrides) -> "DiffusionReactionHDGSolver":
        """Update options, retaining compatible raw recovery data on scalar tau retries.

        Only a finite scalar stabilization-only update can preserve recovery
        references and geometry factors. Operators, local diffusion factors,
        solver hierarchies, and accepted solution state are always cleared.
        """
        options = self.options.with_overrides(**overrides)
        recovery_cache = None
        if (overrides.keys() == {"stabilization"}
                and isinstance(options.stabilization, Real)
                and np.isfinite(options.stabilization)):
            recovery_cache = self._raw_flux_cache_for_tau_retry()
        self.options = options
        self.clear_cache()
        self._hdg_postprocess_cache = recovery_cache
        return self

    def _raw_flux_cache_for_tau_retry(self) -> _HDGPostprocessCache | None:
        """Retain only compatible raw flux references, device tables, and factors."""
        cache = self._hdg_postprocess_cache
        if cache is None or cache.raw_flux_cache is None or cache.base_space is not self.space:
            return None
        trace_space = self.space.trace_space(self.options.trace_basis)
        if cache.trace_space is not trace_space:
            return None
        mode = _normalize_hdg_postprocess_mode(self.options.hdg_postprocess)
        flux_space = _normalize_flux_postprocess_space(self.options.flux_postprocess_space)
        backend = _resolve_diffusion_postprocessing_backend(
            normalize_assembly_backend(self.options.assembly_backend), mode,
            flux_space, self.options.postprocessing_backend,
        )
        if mode != "flux" or backend != "raw-cuda":
            return None
        if not cache.raw_flux_cache.is_compatible(self.space, trace_space, flux_space):
            return None
        return replace(cache, flux_ainv_constraint_t=None, flux_schur_lu=None,
                       flux_schur_pivots=None, primal_lu=None, primal_pivots=None)

    def set_space(self, space: DGSpace, *, keep_problem: bool = True) -> "DiffusionReactionHDGSolver":
        """Replace the DG space and invalidate all computed artifacts."""
        self.space = space
        if not keep_problem:
            self.clear_problem()
        self.clear_cache()
        return self

    def set_mesh(
            self,
            mesh,
            *,
            order: int | None = None,
            basis_type: str | None = None,
            name: str | None = None,
            keep_problem: bool = True,
            **space_kwargs,
    ) -> "DiffusionReactionHDGSolver":
        """Build and install a new :class:`DGSpace` from ``mesh``."""
        new_space = DGSpace(
            mesh,
            self.space.order if order is None else order,
            basis_type=self.space.reference.basis_type if basis_type is None else basis_type,
            name=self.space.name if name is None else name,
            **space_kwargs,
        )
        return self.set_space(new_space, keep_problem=keep_problem)

    def clear_problem(self) -> "DiffusionReactionHDGSolver":
        """Remove stored PDE inputs and invalidate computed artifacts."""
        self.source = None
        self.reaction = None
        self.boundary_condition = None
        self._problem_is_set = False
        self.clear_cache()
        return self

    def set_problem(self, source, reaction, boundary_condition: Callable | float) -> "DiffusionReactionHDGSolver":
        """Set source, reaction, and Dirichlet boundary data."""
        self.source = source
        self.reaction = reaction
        self.boundary_condition = hdg_assembly.normalize_boundary_condition(boundary_condition)
        self._problem_is_set = True
        self.clear_cache()
        return self

    def set_discrete_problem(self, source_h, reaction_h, boundary_condition: Callable | float) -> "DiffusionReactionHDGSolver":
        """Set already-discretized source/reaction data."""
        return self.set_problem(source_h, reaction_h, boundary_condition)

    def _can_preserve_operator_on_rhs_update(self) -> bool:
        """Return whether a source update can reuse the assembled trace operator."""
        backend = "numpy" if self.options.assembly_backend == "auto" else str(self.options.assembly_backend)
        return (
            bool(self.options.cache_device_matrix)
            and self.options.boundary_mode == "eliminate"
            and backend in {"numpy", "numba", "cupy", "raw-cuda"}
            and (backend != "raw-cuda" or _diffusion_is_identity(self.options.diffusion))
        )

    def set_source(self, source) -> "DiffusionReactionHDGSolver":
        """Replace only the source input and invalidate cached artifacts."""
        self._require_problem_or_partial_update()
        self.source = source
        self._problem_is_set = self.reaction is not None and self.boundary_condition is not None
        if self._can_preserve_operator_on_rhs_update():
            self.clear_rhs_and_solution()
        else:
            self.clear_cache()
        return self

    def set_reaction(self, reaction) -> "DiffusionReactionHDGSolver":
        """Replace only the reaction input and invalidate cached artifacts."""
        self._require_problem_or_partial_update()
        self.reaction = reaction
        self._problem_is_set = self.source is not None and self.boundary_condition is not None
        self.clear_cache()
        return self

    def set_boundary_condition(self, boundary_condition: Callable | float) -> "DiffusionReactionHDGSolver":
        """Replace Dirichlet trace data and invalidate cached artifacts."""
        self._require_problem_or_partial_update()
        self.boundary_condition = hdg_assembly.normalize_boundary_condition(boundary_condition)
        self._problem_is_set = self.source is not None and self.reaction is not None
        if self._can_preserve_operator_on_rhs_update():
            self.clear_rhs_and_solution()
        else:
            self.clear_cache()
        return self

    def _close_raw_cuda_amgx_retry_solvers(self) -> None:
        """Release stateful AMGX fallback solvers owned by this instance."""
        solvers = getattr(self, "_raw_cuda_amgx_retry_solver_cache", {})
        for solver in set(solvers.values()):
            solver.close(suppress_errors=True)
        solvers.clear()

    def close(self) -> None:
        """Release persistent device solver state owned by this instance."""
        self.clear_cache()

    def __enter__(self):
        """Return the solver; :meth:`close` runs when the ``with`` block exits."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Release device state without suppressing a caller exception."""
        self.close()
        return False

    def __del__(self):
        """Best-effort release of persistent AMGX retry state."""
        try:
            self._close_raw_cuda_amgx_retry_solvers()
        except Exception:
            pass

    def clear_cache(self) -> "DiffusionReactionHDGSolver":
        """Clear assembled matrices, local solvers, and latest solution."""
        self._close_raw_cuda_amgx_retry_solvers()
        native_solver = getattr(self, "_raw_cuda_fb_hp_mg_solver", None)
        if native_solver is not None:
            close = getattr(native_solver, "close", None)
            if close is not None:
                close()
        self._raw_cuda_fb_hp_mg_solver = None
        self._raw_cuda_fb_hp_mg_solver_key = None
        self._raw_cuda_fb_hp_mg_failed_key = None
        self._raw_cuda_fb_hp_mg_failure_reason = None
        raw_amgx_solver = getattr(self, "_raw_cuda_amgx_solver", None)
        if raw_amgx_solver is not None:
            close = getattr(raw_amgx_solver, "close", None)
            if close is not None:
                close()
        self._raw_cuda_amgx_solver = None
        self._raw_cuda_amgx_solver_key = None
        self._raw_cuda_last_trace_reduced = None
        self._raw_cuda_assembly_cache = None
        self._raw_cuda_operator_key = None
        self._raw_cuda_rhs_valid = False
        cupy_amgx_solver = getattr(self, "_cupy_amgx_solver", None)
        if cupy_amgx_solver is not None:
            close = getattr(cupy_amgx_solver, "close", None)
            if close is not None:
                close()
        self._cupy_amgx_solver = None
        self._cupy_amgx_solver_key = None
        self._cupy_last_trace_reduced = None
        self._cupy_assembly_cache = None
        self._cupy_operator_key = None
        self._cupy_rhs_valid = False
        self._numba_local_factors = None
        self._host_cached_rhs_valid = False
        self.result: DiffusionReactionResult | None = None
        self.field: DGField | None = None
        self.flux: VectorDGField | None = None
        self.postprocessed_field: DGField | None = None
        self.postprocessed_flux: VectorDGField | None = None
        self.trace: np.ndarray | None = None

        self.rows: np.ndarray | None = None
        self.cols: np.ndarray | None = None
        self.data: np.ndarray | None = None
        self.rhs: np.ndarray | None = None
        self.solve_rows: np.ndarray | None = None
        self.solve_cols: np.ndarray | None = None
        self.solve_data: np.ndarray | None = None
        self.solve_rhs: np.ndarray | None = None
        self.boundary_trace: np.ndarray | None = None
        self.reduction = None
        self._device_solve_matrix = None
        self._device_solve_matrix_scale_system: bool | None = None
        self._device_solve_matrix_shape: tuple[int, int] | None = None
        self._host_solve_matrix = None
        self._host_scaled_solve_matrix = None
        self._host_inverse_diagonal: np.ndarray | None = None
        self._host_scaled_solve_matrix_shape: tuple[int, int] | None = None

        self.local_solver: np.ndarray | None = None
        self.element_boundary_mats: np.ndarray | None = None
        self.local_unknowns: np.ndarray | None = None
        self._hdg_postprocess_cache: _HDGPostprocessCache | None = None
        self.global_solve_result: SolveResult | None = None
        self.timings: DiffusionReactionTimings | None = None
        return self

    def clear_solution(self) -> "DiffusionReactionHDGSolver":
        """Drop only the latest trace, reconstructed fields, and diagnostics."""
        self.result = None
        self.field = None
        self.flux = None
        self.postprocessed_field = None
        self.postprocessed_flux = None
        self.trace = None
        self.local_unknowns = None
        self.global_solve_result = None
        self.timings = None
        return self

    def clear_rhs_and_solution(self) -> "DiffusionReactionHDGSolver":
        """Drop source/boundary-dependent data while keeping a cached operator."""
        self.clear_solution()
        self.rhs = None
        self.solve_rhs = None
        self.boundary_trace = None
        self._raw_cuda_rhs_valid = False
        self._cupy_rhs_valid = False
        self._host_cached_rhs_valid = False
        return self

    def assemble_global_matrix(
            self,
            *,
            source: Any = _UNSET,
            reaction: Any = _UNSET,
            boundary_condition: Callable | object = _UNSET,
            **option_overrides,
    ) -> DiffusionReactionAssemblyResult:
        """Assemble the reduced HDG trace system without solving it."""
        provided = (
            source is not _UNSET,
            reaction is not _UNSET,
            boundary_condition is not _UNSET,
        )
        if any(provided):
            if not all(provided):
                raise ValueError("source, reaction, and boundary_condition must be provided together")
            self.set_problem(source, reaction, boundary_condition)

        if option_overrides:
            self.with_options(**option_overrides)

        self._require_problem()
        options = self._resolved_options()
        backend = normalize_assembly_backend(options.assembly_backend)
        factor_policy = _validate_local_factor_cache_configuration(options, backend, stateful=True)
        trace_basis = normalize_trace_basis(options.trace_basis)
        validate_diffusion_backend_configuration(
            operation="assemble",
            assembly_backend=backend,
            solver=options.solver,
            cupyx_solver=options.cupyx_solver,
            boundary_mode=options.boundary_mode,
            trace_basis=trace_basis,
            local_solver_backend=options.local_solver_backend,
            raw_matrix_format=options.raw_matrix_format,
            postprocess_mode="none",
            identity_diffusion=_diffusion_is_identity(options.diffusion),
            scalar_stabilization=np.isscalar(options.stabilization),
        )
        raw_block_size = (
            resolve_raw_cuda_block_size(
                options.raw_block_size,
                equation="diffusion-reaction",
                order=self.space.order,
            )
            if backend == "raw-cuda"
            else None
        )
        trace_space = self.space.trace_space(trace_basis)

        start_total = time.perf_counter()
        local_solver = None
        element_boundary_mats = None
        trace_system = None
        reduction = None
        indptr = None
        indices = None
        matrix_format = (
            "coo"
            if backend != "raw-cuda" or str(options.raw_matrix_format).lower() == "auto"
            else str(options.raw_matrix_format).lower()
        )
        timings: dict[str, float] = {}

        if backend == "numba":
            source_input = _require_same_space_dg_field_for_backend(self.source, self.space, label="source", backend="numba")
            reaction_input = _require_same_space_dg_field_for_backend(self.reaction, self.space, label="reaction", backend="numba")
            if _diffusion_is_identity(options.diffusion):
                from hdgfem.mixed.numba import (
                                    assemble_projected_diffusion_trace_system_eliminated_numba,
                                    build_diffusion_schur_cache_numba,
                                )
                if factor_policy != "none" and self._numba_local_factors is None:
                    self._numba_local_factors = build_diffusion_schur_cache_numba(
                        reaction_input, options.stabilization, self.space,
                        factor_kind=factor_policy,
                    )
                    timings["numba.local_factors.build"] = self._numba_local_factors.construction_seconds

                assembled = assemble_projected_diffusion_trace_system_eliminated_numba(
                    source_input,
                    reaction_input,
                    self.boundary_condition,
                    options.stabilization,
                    self.space,
                    trace_space=trace_space,
                    cached_factors=self._numba_local_factors,
                )
            else:
                from hdgfem.mixed.numba import (
                                    assemble_projected_tensor_diffusion_trace_system_eliminated_numba,
                                )

                assembled = assemble_projected_tensor_diffusion_trace_system_eliminated_numba(
                    source_input,
                    reaction_input,
                    _project_inverse_diffusion_for_numba(options.diffusion, self.space),
                    self.boundary_condition,
                    options.stabilization,
                    self.space,
                    trace_space=trace_space,
                )
            trace_system = assembled.trace_system
            reduction = assembled.reduction
            timings.update(assembled.timings)

        elif backend == "numpy":
            tau = normalize_diffusion_stabilization(options.stabilization, self.space)
            source_rhs = hdg_assembly.block_source_moments(self.source, self.space, num_blocks=3, source_block=0)
            local_solver = local_solvers(
                self.reaction,
                tau,
                self.space,
                backend=options.local_solver_backend,
                diffusion=options.diffusion,
            )
            element_boundary_mats = diffusion_element_boundary_mats(tau, self.space, trace_space=trace_space)
            full_trace_system = assemble_diffusion_trace_system(
                local_solver,
                element_boundary_mats,
                source_rhs,
                self.boundary_condition,
                tau,
                self.space,
                boundary_penalty=options.boundary_penalty,
                verbosity=options.verbose,
                trace_space=trace_space,
            )
            trace_system, reduction = _reduced_result_from_full_trace_system(full_trace_system, self.space)

        elif backend in {"cupy", "raw-cuda"}:
            if not _diffusion_is_identity(options.diffusion):
                raise NotImplementedError(f"{backend} diffusion assembly currently supports identity diffusion only")
            if not np.isscalar(options.stabilization):
                raise NotImplementedError(f"{backend} diffusion assembly currently supports scalar stabilization only")
            if backend == "cupy":
                from hdgfem.mixed.cupy import (
                                    assemble_projected_diffusion_trace_system_eliminated_cupy,
                                )

                gpu = assemble_projected_diffusion_trace_system_eliminated_cupy(
                    self.source,
                    self.reaction,
                    self.boundary_condition,
                    float(options.stabilization),
                    self.space,
                    trace_basis=trace_basis,
                )
            else:
                from hdgfem.mixed.cupy import (
                                    assemble_projected_diffusion_trace_system_eliminated_raw_cupy,
                                )

                source_input = _require_same_space_dg_field_for_backend(self.source, self.space, label="source", backend="raw-cuda")
                reaction_input = _require_same_space_dg_field_for_backend(self.reaction, self.space, label="reaction", backend="raw-cuda")
                gpu = assemble_projected_diffusion_trace_system_eliminated_raw_cupy(
                    source_input,
                    reaction_input,
                    self.boundary_condition,
                    float(options.stabilization),
                    self.space,
                    trace_basis=trace_basis,
                    matrix_format=matrix_format,
                    block_size=raw_block_size,
                )
            rows = _host_array(gpu.rows, dtype=np.int64)
            cols = _host_array(gpu.cols, dtype=np.int64)
            data = _host_array(gpu.data, dtype=REAL_DTYPE)
            rhs = _host_array(gpu.rhs, dtype=REAL_DTYPE)
            boundary_trace = _full_boundary_trace_from_compact(gpu.boundary_trace, self.space)
            indptr = _host_array(gpu.indptr, dtype=np.int32)
            indices = _host_array(gpu.indices, dtype=np.int32)
            matrix_format = str(gpu.matrix_format)
            trace_system = hdg_assembly.TraceSystem(
                rows=rows,
                cols=cols,
                data=data,
                rhs=rhs,
                boundary_trace=boundary_trace,
            )
            reduction = _reduction_from_reduced_trace_system(rows, cols, data, rhs, boundary_trace, self.space)
            timings.update(gpu.timings)

        assert trace_system is not None
        assert reduction is not None
        timings.setdefault("total", time.perf_counter() - start_total)
        result = DiffusionReactionAssemblyResult(
            rows=None if trace_system.rows is None else np.ascontiguousarray(trace_system.rows, dtype=np.int64),
            cols=None if trace_system.cols is None else np.ascontiguousarray(trace_system.cols, dtype=np.int64),
            data=np.ascontiguousarray(trace_system.data, dtype=REAL_DTYPE),
            rhs=np.ascontiguousarray(trace_system.rhs, dtype=REAL_DTYPE),
            boundary_trace=np.ascontiguousarray(trace_system.boundary_trace, dtype=REAL_DTYPE),
            reduction=reduction,
            assembly_backend=backend,
            matrix_format=matrix_format,
            indptr=indptr,
            indices=indices,
            timings=timings,
        )

        self.clear_solution()
        # NumPy RHS updates eliminate prescribed values from the full operator;
        # the reduced operator has already lost those boundary couplings.
        cached_system = full_trace_system if backend == "numpy" else trace_system
        self.rows = cached_system.rows
        self.cols = cached_system.cols
        self.data = cached_system.data
        self.rhs = cached_system.rhs
        self.solve_rows = result.rows
        self.solve_cols = result.cols
        self.solve_data = result.data
        self.solve_rhs = result.rhs
        self.boundary_trace = result.boundary_trace
        self.reduction = result.reduction
        self._host_cached_rhs_valid = backend in {"numpy", "numba"}
        self.local_solver = local_solver
        self.element_boundary_mats = element_boundary_mats
        return result

    def solve(
            self,
            *,
            source: Any = _UNSET,
            reaction: Any = _UNSET,
            boundary_condition: Callable | object = _UNSET,
            initial_guess: Any = _UNSET,
            postprocess_overrides: dict[str, Any] | None = None,
            **option_overrides,
    ) -> DiffusionReactionResult:
        """Assemble, solve, reconstruct, cache, and return the HDG result."""
        provided = (
            source is not _UNSET,
            reaction is not _UNSET,
            boundary_condition is not _UNSET,
        )
        if any(provided):
            if not all(provided):
                raise ValueError("source, reaction, and boundary_condition must be provided together")
            self.set_problem(source, reaction, boundary_condition)

        if option_overrides:
            self.with_options(**option_overrides)
        per_call_overrides = dict(postprocess_overrides or {})
        allowed_per_call = {
            "hdg_postprocess", "flux_postprocess_space", "postprocessing_backend"
        }
        unknown_per_call = set(per_call_overrides) - allowed_per_call
        if unknown_per_call:
            names = ", ".join(sorted(unknown_per_call))
            raise ValueError(f"unknown per-call postprocessing options: {names}")

        self._require_problem()
        stored_options = self.options
        active_options = stored_options.with_overrides(**per_call_overrides)
        if initial_guess is not _UNSET:
            active_options = active_options.with_overrides(initial_guess=initial_guess)
        self.options = active_options
        self.options = self._resolved_options()
        try:
            options = self.options
            backend = normalize_assembly_backend(options.assembly_backend)
            _validate_local_factor_cache_configuration(options, backend, stateful=True)
            postprocess_mode = _normalize_hdg_postprocess_mode(options.hdg_postprocess)
            validate_diffusion_backend_configuration(
                operation="solve",
                assembly_backend=backend,
                solver=options.solver,
                cupyx_solver=options.cupyx_solver,
                boundary_mode=options.boundary_mode,
                trace_basis=options.trace_basis,
                local_solver_backend=options.local_solver_backend,
                raw_matrix_format=options.raw_matrix_format,
                postprocess_mode=postprocess_mode,
                identity_diffusion=_diffusion_is_identity(options.diffusion),
                scalar_stabilization=np.isscalar(options.stabilization),
                allow_raw_device_solve=True,
            )
            initial_assembly_seconds = 0.0
            if backend == "numba" and options.cache_local_factors != "none" and self.solve_data is None:
                assembly_start = time.perf_counter()
                self.assemble_global_matrix()
                initial_assembly_seconds = time.perf_counter() - assembly_start
            if backend == "raw-cuda":
                result = self._solve_raw_cuda_device_amgx()
            elif backend == "cupy":
                from hdgfem.solvers.diffusion_device import solve_cupy_device_amgx

                result = solve_cupy_device_amgx(self)
            elif self._can_solve_with_cached_numpy_operator():
                result = self._solve_numpy_with_cached_operator()
            elif (
                backend == "numba"
                and _diffusion_is_identity(self.options.diffusion)
                and self.local_solver is None
                and self.solve_rows is not None
                and self.solve_cols is not None
                and self.solve_data is not None
                and self.reduction is not None
            ):
                result = self._solve_numba_with_cached_operator()
            else:
                solve_kwargs = self.options.as_solve_kwargs()
                solve_kwargs["hdg_postprocess"] = "none"
                result = solve_diffusion_reaction_hdg(
                    self.source,
                    self.reaction,
                    self.boundary_condition,
                    self.space,
                    return_=("result",),
                    **solve_kwargs,
                )
            if initial_assembly_seconds:
                result = replace(result, timings=replace(
                    result.timings,
                    trace_assembly=result.timings.trace_assembly + initial_assembly_seconds,
                    total=result.timings.total + initial_assembly_seconds,
                ))
            result = self._postprocess_result(result)
        finally:
            self.options = stored_options
        self._store_result(result)
        return result

    def _solve_raw_cuda_device_amgx(self) -> DiffusionReactionResult:
        """Solve a raw-CUDA diffusion trace system directly with device CSR/BSR AMGX."""
        from hdgfem.solvers.diffusion_raw_cuda import solve_raw_cuda_device_amgx

        return solve_raw_cuda_device_amgx(self)

    def _cupyx_solver_selected(self) -> bool:
        """Return True when the configured global solve uses Cupyx."""
        solver = self.options.solver
        normalized = "" if solver is None else str(solver).lower()
        return normalized == "cupyx" or normalized.startswith(("cupyx_", "cupyx-"))

    def _pypardiso_solver_selected(self) -> bool:
        """Return True when the global solve uses a PyPardiso direct backend."""
        solver = self.options.solver
        normalized = "" if solver is None else str(solver).lower().replace("_", "-")
        return normalized in {"pypardiso", "pardiso", "pypardiso-spd", "pardiso-spd"}

    def _scipy_iterative_solver_selected(self) -> bool:
        """Return True when the global solve uses a SciPy Krylov backend."""
        solver = self.options.solver
        normalized = "" if solver is None else str(solver).upper()
        return normalized in {"BICG", "BICGSTAB", "CG", "CGS", "GMRES", "LGMRES", "MINRES"}

    def _prepared_cupyx_operator(self, *, scale_system: bool):
        """Return cached host/device matrices for repeated backend solves.

        The Numba diffusion path stores the reduced COO operator on the solver.
        For repeated solves where only the RHS changes, this method builds the
        host CSR matrix once for PyPardiso or SciPy, or converts it once to
        CuPy CSR for Cupyx. For a scaled SciPy Krylov solve, the scaled CSR and
        inverse diagonal are cached as well. The device matrix represents the
        same scaled or unscaled operator that :func:`solve_global_system` will
        use for the solve.
        """
        use_cupyx = self._cupyx_solver_selected()
        use_pypardiso = self._pypardiso_solver_selected()
        use_scipy = self._scipy_iterative_solver_selected()
        if not self.options.cache_device_matrix or not (use_cupyx or use_pypardiso or use_scipy):
            return None, None
        if self.solve_rows is None or self.solve_cols is None or self.solve_data is None or self.solve_rhs is None:
            return None, None

        shape = (self.solve_rhs.size, self.solve_rhs.size)
        cache_valid = self._host_solve_matrix is not None and (
            use_pypardiso
            or (
                use_scipy
                and (
                    not scale_system
                    or (
                        self._host_scaled_solve_matrix is not None
                        and self._host_inverse_diagonal is not None
                        and self._host_scaled_solve_matrix_shape == shape
                    )
                )
            )
            or (
                self._device_solve_matrix is not None
                and self._device_solve_matrix_scale_system == bool(scale_system)
                and self._device_solve_matrix_shape == shape
            )
        )
        if cache_valid:
            return self._host_solve_matrix, self._device_solve_matrix

        host_matrix = assemble_global_matrix(
            self.solve_rows,
            self.solve_cols,
            self.solve_data,
            self.solve_rhs.size,
        )
        self._host_solve_matrix = host_matrix
        if use_pypardiso:
            self._device_solve_matrix = None
            self._device_solve_matrix_scale_system = None
            self._device_solve_matrix_shape = shape
            return self._host_solve_matrix, None

        if use_scipy:
            if scale_system:
                scaled_matrix, inverse_diagonal = diagonal_scale_system(
                    host_matrix,
                    np.ones(self.solve_rhs.size, dtype=REAL_DTYPE),
                    copy_matrix=True,
                )
                self._host_scaled_solve_matrix = scaled_matrix
                self._host_inverse_diagonal = inverse_diagonal
                self._host_scaled_solve_matrix_shape = shape
            return self._host_solve_matrix, None

        from hdgfem.linalg.gpu.sparse import scipy_csr_to_cupy

        if scale_system:
            device_host_matrix, _ = diagonal_scale_system(
                host_matrix,
                np.zeros(self.solve_rhs.size, dtype=REAL_DTYPE),
                copy_matrix=True,
            )
        else:
            device_host_matrix = host_matrix
        self._device_solve_matrix = scipy_csr_to_cupy(device_host_matrix)
        self._device_solve_matrix_scale_system = bool(scale_system)
        self._device_solve_matrix_shape = shape
        return self._host_solve_matrix, self._device_solve_matrix

    def _can_solve_with_cached_numpy_operator(self) -> bool:
        """Return whether the cached host trace operator can serve this solve."""
        options = self.options
        backend = "numpy" if options.assembly_backend == "auto" else str(options.assembly_backend)
        return (
            options.cache_device_matrix
            and backend == "numpy"
            and options.boundary_mode == "eliminate"
            and self.local_solver is not None
            and self.element_boundary_mats is not None
            and self.rows is not None
            and self.cols is not None
            and self.data is not None
            and self.solve_rows is not None
            and self.solve_cols is not None
            and self.solve_data is not None
            and self.reduction is not None
        )

    def _reduced_rhs_from_cached_numpy_operator(self, rhs_full: np.ndarray, boundary_trace: np.ndarray):
        """Eliminate known trace values using the shared fixed-operator helper."""
        from hdgfem.linalg.reduction import update_known_dof_rhs
        if self.reduction is None:
            raise RuntimeError("cached reduced RHS requires a KnownDofReduction")
        reduction = update_known_dof_rhs(self.rows, self.cols, self.data, rhs_full,
                                         boundary_trace, self.reduction)
        return reduction.rhs, reduction

    def _solve_numpy_with_cached_operator(self) -> DiffusionReactionResult:
        """Solve with cached host local solvers and reduced trace matrix."""
        from hdgfem.solvers.diffusion_host import solve_numpy_with_cached_operator

        return solve_numpy_with_cached_operator(self)

    def _solve_numba_with_cached_operator(self) -> DiffusionReactionResult:
        """Solve with cached numba local solvers and reduced trace matrix."""
        from hdgfem.solvers.diffusion_host import solve_numba_with_cached_operator

        return solve_numba_with_cached_operator(self)

    def _require_problem_or_partial_update(self) -> None:
        """Allow partial coefficient updates until a complete PDE problem is set."""
        if self.source is None and self.reaction is None and self.boundary_condition is None:
            return

    def _require_problem(self) -> None:
        """Raise when the reusable solver does not hold a complete PDE problem."""
        if not self._problem_is_set:
            raise RuntimeError(
                "no complete diffusion-reaction problem is set; call set_problem(...) "
                "or pass source, reaction, and boundary_condition to solve(...)"
            )

    def _postprocess_result(self, result: DiffusionReactionResult) -> DiffusionReactionResult:
        """Attach requested HDG post-processed fields using the solver cache."""
        mode = _normalize_hdg_postprocess_mode(self.options.hdg_postprocess)
        if mode == "none":
            return result
        flux_space = _normalize_flux_postprocess_space(
            self.options.flux_postprocess_space
        )
        postprocessing_backend = _resolve_diffusion_postprocessing_backend(
            normalize_assembly_backend(self.options.assembly_backend),
            mode,
            flux_space,
            self.options.postprocessing_backend,
        )
        verbosity = _verbosity_level(self.options.verbose)

        def postprocess():
            """Compute requested primal and conservative flux postprocessing fields."""
            postprocessed_field, postprocessed_flux, cache = _postprocess_diffusion_solution(
                result.local_unknowns_device if result.local_unknowns_device is not None else result.local_unknowns,
                result.trace_device if result.trace_device is not None else result.trace,
                self.space,
                self.options.stabilization,
                self.options.diffusion,
                mode,
                trace_space=self.space.trace_space(self.options.trace_basis),
                cache=self._hdg_postprocess_cache,
                flux_postprocess_space=flux_space,
                postprocessing_backend=postprocessing_backend,
            )
            self._hdg_postprocess_cache = cache
            return postprocessed_field, postprocessed_flux

        (postprocessed_field, postprocessed_flux), elapsed = _timed_call(
            "post-processing HDG fields",
            verbosity,
            postprocess,
        )
        return _result_with_hdg_postprocessing(
            result,
            postprocessed_field=postprocessed_field,
            postprocessed_flux=postprocessed_flux,
            flux_postprocess_space=flux_space,
            postprocessing_backend=_reported_diffusion_postprocessing_backend(
                postprocessing_backend,
                mode,
            ),
            elapsed=elapsed,
        )

    def _store_result(self, result: DiffusionReactionResult) -> None:
        """Copy result artifacts into named cache attributes."""
        self.result = result
        self.field = result.field
        self.flux = result.flux
        self.postprocessed_field = result.postprocessed_field
        self.postprocessed_flux = result.postprocessed_flux
        self.trace = result.trace
        self.timings = result.timings
        self.local_unknowns = result.local_unknowns

        self.rows = result.matrix_rows
        self.cols = result.matrix_cols
        self.data = result.matrix_data
        self.rhs = result.rhs
        self.solve_rows = result.solve_matrix_rows
        self.solve_cols = result.solve_matrix_cols
        self.solve_data = result.solve_matrix_data
        self.solve_rhs = result.solve_rhs
        self.boundary_trace = result.boundary_trace
        self.reduction = result.reduction

        self.local_solver = result.local_solver
        self.element_boundary_mats = result.element_boundary_mats
        self.global_solve_result = result.global_solve_result


def solve_diffusion_reaction_hdg(
        source,
        reaction,
        boundary_condition: Callable | float,
        space: DGSpace,
        *,
        diffusion=1.0,
        stabilization="global_length",
        solver: str | None = "BICGSTAB",
        preconditioner="ilu",
        solver_rtol: float = 1e-13,
        solver_atol: float = 0.0,
        maxiter: int | None = None,
        scale_system: bool = True,
        petsc_preset: str = "cg_gamg",
        petsc_levels: int | None = None,
        petsc_options: dict | None = None,
        petsc_divtol: float = 1e4,
        petsc_monitor: bool = False,
        cupyx_solver: str = "bicgstab",
        amgx_config: dict | None = None,
        amgx_retry_attempts: tuple[dict[str, Any], ...] | None = None,
        fb_hp_mg_preconditioner_policy: Literal["standard", "fast", "robust"] = "standard",
        fb_hp_mg_true_residual_every: int = 10,
        fb_hp_mg_residual_history: bool = True,
        cache_device_matrix: bool = False,
        cache_local_factors: LocalFactorCachePolicy = "none",
        ilu_drop_tol: float = 1e-10,
        ilu_fill_factor: float = 35,
        ilu_failure: Literal["raise", "none"] = "raise",
        ilu_permc_spec: str = "COLAMD",
        initial_guess: np.ndarray | None = None,
        local_solver_backend: LocalSolverBackend = "numpy",
        assembly_backend: TraceAssemblyBackend = "numpy",
        trace_basis: Literal["legacy-lagrange", "legendre-modal", "bernstein"] = "legacy-lagrange",
        raw_matrix_format: Literal["coo", "csr", "bsr"] = "coo",
        raw_block_size: RawCudaBlockSize = "auto",
        boundary_penalty: float = 1e20,
        boundary_mode: Literal["penalty", "eliminate"] = "penalty",
        hdg_postprocess: HDGPostprocessMode = "none",
        flux_postprocess_space: FluxPostprocessSpace = "l2_closest",
        postprocessing_backend: PostprocessingBackend = "auto",
        verbose: bool | int = True,
        return_: Iterable[ReturnKey] = ("result",),
):
    r"""Solve :math:`-\nabla\cdot(\kappa\nabla u) + r u=f` with HDG static condensation.

    ``assembly_backend`` controls how the HDG trace operator is assembled.
    NumPy assembly accepts analytic source/reaction callables; Numba assembly
    requires same-space :class:`DGField` source/reaction inputs, using
    ``space.zeros`` or ``space.constant`` for exact zero/constant coefficients.
    ``solver`` controls where the global trace system is inverted.  In
    particular, ``assembly_backend="numba"`` with ``solver="amgx"`` or
    ``solver="cupyx"`` means host Numba assembly followed by a GPU sparse
    solve.  ``cupyx_solver`` selects the Cupyx Krylov method when
    ``solver="cupyx"``; aliases such as ``solver="cupyx_bicgstab"`` are
    also accepted.  ``cache_device_matrix`` is used by the stateful solver
    class for repeated RHS-only solves and has no effect in this one-shot
    function.

    ``stabilization="global_length"`` is the production default and resolves
    ``gamma_d*kappa/L_Omega`` before backend dispatch. Explicit positive
    stabilization inputs override it.

    ``boundary_condition`` accepts a callable ``g(x, y)`` or a real scalar
    constant. Discrete field boundary inputs are rejected.
    """
    boundary_condition = hdg_assembly.normalize_boundary_condition(boundary_condition)
    cache_policy = _normalize_local_factor_cache_policy(cache_local_factors)
    if cache_policy != "none":
        raise ValueError("cache_local_factors='schur-lu' requires the stateful DiffusionReactionHDGSolver")
    postprocess_mode = _normalize_hdg_postprocess_mode(hdg_postprocess)
    effective_backend = normalize_assembly_backend(assembly_backend)
    flux_postprocess_space = _normalize_flux_postprocess_space(
        flux_postprocess_space
    )
    postprocessing_backend = _resolve_diffusion_postprocessing_backend(
        effective_backend,
        postprocess_mode,
        flux_postprocess_space,
        postprocessing_backend,
    )
    trace_basis = normalize_trace_basis(trace_basis)
    effective_stabilization = resolve_diffusion_stabilization(
        stabilization,
        diffusion,
        space,
    )
    validate_diffusion_backend_configuration(
        operation="solve",
        assembly_backend=effective_backend,
        solver=solver,
        cupyx_solver=cupyx_solver,
        boundary_mode=boundary_mode,
        trace_basis=trace_basis,
        local_solver_backend=local_solver_backend,
        raw_matrix_format=raw_matrix_format,
        postprocess_mode=postprocess_mode,
        identity_diffusion=_diffusion_is_identity(diffusion),
        scalar_stabilization=np.isscalar(effective_stabilization),
        allow_raw_device_solve=False,
    )
    total_start = time.perf_counter()
    verbosity = _verbosity_level(verbose)
    if verbosity:
        print("\n----- DG FEM Diffusion-Reaction HDG Solve -----")
    trace_space = space.trace_space(trace_basis)
    effective_boundary_mode = "eliminate" if effective_backend == "numba" else boundary_mode
    effective_scale_system = False if solver is not None and str(solver).lower() == "petsc" else scale_system
    projected_numba_identity_diffusion = effective_backend == "numba" and _diffusion_is_identity(diffusion)
    projected_numba_tensor_diffusion = effective_backend == "numba" and not _diffusion_is_identity(diffusion)
    projected_numba_diffusion = projected_numba_identity_diffusion or projected_numba_tensor_diffusion

    def prepare_data():
        """Normalize coefficients, stabilization, and backend assembly inputs."""
        tau, _ = _timed_call(
            "normalizing stabilization",
            verbosity,
            lambda: normalize_diffusion_stabilization(effective_stabilization, space),
            level=2,
        )
        source_input = source
        reaction_input = reaction
        diffusion_inverse_input = None
        if projected_numba_diffusion:
            source_input = _require_same_space_dg_field_for_backend(source, space, label="source", backend="numba")
            reaction_input = _require_same_space_dg_field_for_backend(reaction, space, label="reaction", backend="numba")
            if projected_numba_tensor_diffusion:
                diffusion_inverse_input = _project_inverse_diffusion_for_numba(diffusion, space)
            return tau, None, source_input, reaction_input, diffusion_inverse_input
        source_rhs, _ = _timed_call(
            "assembling block source moments",
            verbosity,
            lambda: hdg_assembly.block_source_moments(source_input, space, num_blocks=3, source_block=0),
            level=2,
        )
        return tau, source_rhs, source_input, reaction_input, diffusion_inverse_input

    (tau, source_rhs, source_for_backend, reaction_for_local, diffusion_inverse_for_backend), preparation = _timed_call(
        "preparing source and stabilization",
        verbosity,
        prepare_data,
        multiline=_detailed_logging(verbosity),
    )
    effective_local_solver_backend = "numba" if projected_numba_diffusion else local_solver_backend

    def build_local_solver():
        """Build the selected element-local diffusion solver data."""
        if effective_local_solver_backend not in {"numpy", "numba"}:
            raise ValueError("local_solver_backend must be 'numpy' or 'numba'")
        if effective_local_solver_backend == "numba" and _build_res_numba is None:
            raise RuntimeError("local_solver_backend='numba' requires numba")

        if not _diffusion_is_identity(diffusion):
            return local_solvers(
                reaction_for_local,
                tau,
                space,
                backend=effective_local_solver_backend,
                diffusion=diffusion,
            )

        d0, d1, m_tau, m_n0, m_n1, jacs_inv = _local_solver_pre_mats(
            reaction_for_local,
            tau,
            space,
            verbosity=verbosity,
        )
        e, _ = _timed_call(
            "inverting condensed scalar blocks",
            verbosity,
            lambda: _local_solver_scalar_inverse(d0, d1, m_tau, m_n0, m_n1, jacs_inv, space),
            level=2,
        )
        if effective_local_solver_backend == "numpy":
            local_solver, _ = _timed_call(
                "assembling full mixed inverse blocks (numpy)",
                verbosity,
                lambda: _local_solver_blocks_numpy(e, d0, d1, m_n0, m_n1, jacs_inv, space),
                level=2,
            )
        else:
            local_solver, _ = _timed_call(
                "assembling full mixed inverse blocks (numba)",
                verbosity,
                lambda: np.ascontiguousarray(
                    _build_res_numba(e, d0, d1, m_n0, m_n1, space.quad_data.MKrf_inv, jacs_inv)
                ),
                level=2,
            )
        return local_solver

    if projected_numba_diffusion:
        local_solver = None
        local_solver_time = 0.0
        element_boundary_mats = None
        boundary_time = 0.0
    else:
        local_solver, local_solver_time = _timed_call(
            "building local mixed solvers",
            verbosity,
            build_local_solver,
            multiline=_detailed_logging(verbosity),
        )
        element_boundary_mats, boundary_time = _timed_call(
            "assembling element boundary coupling",
            verbosity,
            lambda: diffusion_element_boundary_mats(tau, space, trace_space=trace_space),
        )

    def assemble_trace():
        """Assemble the condensed global diffusion trace system."""
        if projected_numba_identity_diffusion:
            from hdgfem.mixed.numba import (
                            assemble_projected_diffusion_trace_system_eliminated_numba,
                        )

            return assemble_projected_diffusion_trace_system_eliminated_numba(
                source_for_backend,
                reaction_for_local,
                boundary_condition,
                tau,
                space,
                trace_space=trace_space,
            )
        if projected_numba_tensor_diffusion:
            from hdgfem.mixed.numba import (
                            assemble_projected_tensor_diffusion_trace_system_eliminated_numba,
                        )

            return assemble_projected_tensor_diffusion_trace_system_eliminated_numba(
                source_for_backend,
                reaction_for_local,
                diffusion_inverse_for_backend,
                boundary_condition,
                tau,
                space,
                trace_space=trace_space,
            )
        return assemble_diffusion_trace_system(
            local_solver,
            element_boundary_mats,
            source_rhs,
            boundary_condition,
            tau,
            space,
            boundary_penalty=boundary_penalty,
            verbosity=verbosity,
            trace_space=trace_space,
        )

    trace_assembly_label = (
        "assembling reduced global trace system (numba)"
        if projected_numba_identity_diffusion
        else "assembling reduced tensor trace system (numba)"
        if projected_numba_tensor_diffusion
        else "assembling global trace system"
    )
    trace_out, trace_assembly = _timed_call(
        trace_assembly_label,
        verbosity,
        assemble_trace,
        multiline=_detailed_logging(verbosity),
    )
    if effective_backend == "numba":
        numba_trace = trace_out
        trace_system = numba_trace.trace_system
        reduction = numba_trace.reduction
        boundary_elimination = 0.0
        if _detailed_logging(verbosity):
            timings = numba_trace.timings
            print(
                "  numba diffusion trace assembly timings: "
                f"coefficients={timings.get('coefficient_validation', timings.get('input_validation', 0.0)):.5f}s, "
                f"reduction={timings.get('reduction_map', 0.0):.5f}s, "
                f"kernel={timings.get('kernel', 0.0):.5f}s, "
                f"rhs={timings.get('rhs_finalization', 0.0):.5f}s",
                flush=True,
            )
    else:
        trace_system = trace_out
        reduction = None
        boundary_elimination = 0.0

    initial_guess_time = 0.0
    if initial_guess is not None:
        initial_guess = np.asarray(initial_guess, dtype=REAL_DTYPE)

    if initial_guess is not None:
        initial_guess = impose_boundary_trace_on_guess(initial_guess, trace_system.boundary_trace, space)

    solve_rows = trace_system.rows
    solve_cols = trace_system.cols
    solve_data = trace_system.data
    solve_rhs = trace_system.rhs
    solve_initial_guess = initial_guess
    diagnostic_rows = hdg_assembly.free_trace_dofs(space, trace_space=trace_space)
    if effective_boundary_mode == "eliminate" and reduction is None:
        def eliminate_boundary_trace():
            """Eliminate prescribed boundary trace degrees of freedom."""
            known_mask = ~hdg_assembly.free_trace_dofs(space, trace_space=trace_space)
            known_values = trace_system.boundary_trace.ravel()
            return eliminate_known_dofs(
                trace_system.rows,
                trace_system.cols,
                trace_system.data,
                trace_system.rhs,
                known_mask,
                known_values,
            )

        reduction, boundary_elimination = _timed_call(
            "eliminating boundary trace dofs",
            verbosity,
            eliminate_boundary_trace,
        )
        solve_rows = reduction.rows
        solve_cols = reduction.cols
        solve_data = reduction.data
        solve_rhs = reduction.rhs
        solve_initial_guess = None if initial_guess is None else initial_guess[reduction.free_mask]
        diagnostic_rows = None
    elif effective_boundary_mode == "eliminate":
        solve_rows = reduction.rows
        solve_cols = reduction.cols
        solve_data = reduction.data
        solve_rhs = reduction.rhs
        solve_initial_guess = None if initial_guess is None else initial_guess[reduction.free_mask]
        diagnostic_rows = None

    global_solve_result, solve_time = _timed_call(
        "solving global system",
        verbosity,
        lambda: solve_global_system(
            solve_rows,
            solve_cols,
            solve_data,
            solve_rhs,
            solve_rhs.size,
            solver=solver,
            preconditioner=preconditioner,
            initial_guess=solve_initial_guess,
            rtol=solver_rtol,
            atol=solver_atol,
            maxiter=maxiter,
            ilu_drop_tol=ilu_drop_tol,
            ilu_fill_factor=ilu_fill_factor,
            ilu_failure=ilu_failure,
            ilu_permc_spec=ilu_permc_spec,
            petsc_preset=petsc_preset,
            petsc_levels=petsc_levels,
            petsc_options=petsc_options,
            petsc_divtol=petsc_divtol,
            petsc_monitor=petsc_monitor,
            cupyx_solver=cupyx_solver,
            amgx_config=amgx_config,
            scale_system=effective_scale_system,
            scale_matrix_in_place=effective_scale_system,
            raise_on_nonconvergence=True,
            verbose=verbosity,
            diagnostic_rows=diagnostic_rows,
            diagnostic_label="free trace" if diagnostic_rows is not None else None,
        ),
        multiline=verbosity >= 1,
    )
    if reduction is None:
        trace = np.asarray(global_solve_result.x, dtype=REAL_DTYPE)
    else:
        trace = expand_known_dofs(global_solve_result.x, reduction)

    def reconstruct():
        """Recover local mixed fields from the solved trace coefficients."""
        if projected_numba_identity_diffusion:
            from hdgfem.mixed.numba import (
                            reconstruct_projected_diffusion_local_unknowns_numba,
                        )

            unknowns = reconstruct_projected_diffusion_local_unknowns_numba(
                trace,
                source_for_backend,
                reaction_for_local,
                tau,
                space,
                trace_space=trace_space,
            )
        elif projected_numba_tensor_diffusion:
            from hdgfem.mixed.numba import (
                            reconstruct_projected_tensor_diffusion_local_unknowns_numba,
                        )

            unknowns = reconstruct_projected_tensor_diffusion_local_unknowns_numba(
                trace,
                source_for_backend,
                reaction_for_local,
                diffusion_inverse_for_backend,
                tau,
                space,
                trace_space=trace_space,
            )
        else:
            unknowns = hdg_assembly.reconstruct_local_unknowns(
                trace,
                source_rhs,
                local_solver,
                element_boundary_mats,
                space,
                trace_space=trace_space,
            )
        field, flux = split_diffusion_unknowns(unknowns, space)
        return unknowns, field, flux

    (local_unknowns, field, flux), reconstruction = _timed_call("reconstructing local fields", verbosity, reconstruct)

    postprocessed_field = None
    postprocessed_flux = None
    postprocessing = 0.0
    if postprocess_mode != "none":
        def postprocess():
            """Compute requested superconvergent primal and flux postprocessing fields."""
            post_field, post_flux, _ = _postprocess_diffusion_solution(
                local_unknowns,
                trace,
                space,
                tau,
                diffusion,
                postprocess_mode,
                trace_space=trace_space,
                flux_postprocess_space=flux_postprocess_space,
                postprocessing_backend=postprocessing_backend,
            )
            return post_field, post_flux

        (postprocessed_field, postprocessed_flux), postprocessing = _timed_call(
            "post-processing HDG fields",
            verbosity,
            postprocess,
        )

    timings = DiffusionReactionTimings(
        preparation=preparation,
        local_solver=local_solver_time,
        element_boundary=boundary_time,
        trace_assembly=trace_assembly,
        initial_guess=initial_guess_time,
        boundary_elimination=boundary_elimination,
        solve=solve_time,
        reconstruction=reconstruction,
        postprocessing=postprocessing,
        total=time.perf_counter() - total_start,
    )
    result = DiffusionReactionResult(
        field=field,
        flux=flux,
        trace=trace,
        timings=timings,
        postprocessed_field=postprocessed_field,
        postprocessed_flux=postprocessed_flux,
        flux_postprocess_space=(
            flux_postprocess_space if postprocess_mode in {"flux", "both"} else "none"
        ),
        postprocessing_backend=_reported_diffusion_postprocessing_backend(
            postprocessing_backend,
            postprocess_mode,
        ),
        local_unknowns=local_unknowns,
        matrix_rows=trace_system.rows,
        matrix_cols=trace_system.cols,
        matrix_data=trace_system.data,
        rhs=trace_system.rhs,
        solve_matrix_rows=solve_rows,
        solve_matrix_cols=solve_cols,
        solve_matrix_data=solve_data,
        solve_rhs=solve_rhs,
        boundary_trace=trace_system.boundary_trace,
        reduction=reduction,
        local_solver=local_solver,
        element_boundary_mats=element_boundary_mats,
        initial_guess=initial_guess,
        boundary_mode=effective_boundary_mode,
        scale_system=effective_scale_system,
        assembly_backend=effective_backend,
        global_solve_result=global_solve_result,
    )

    want = tuple(return_)
    if want == ("result",):
        return result

    output = []
    for key in want:
        if key == "result":
            output.append(result)
        elif key in {"trace", "trace_coeffs"}:
            output.append(trace)
        elif key == "flux":
            output.append(flux)
        elif key == "postprocessed_field":
            output.append(postprocessed_field)
        elif key == "postprocessed_flux":
            output.append(postprocessed_flux)
        elif key == "local_unknowns":
            output.append(local_unknowns)
        elif key == "matrix_rows":
            output.append(trace_system.rows)
        elif key == "matrix_cols":
            output.append(trace_system.cols)
        elif key == "matrix_data":
            output.append(trace_system.data)
        elif key == "local_solver":
            output.append(local_solver)
        elif key == "element_boundary_mats":
            output.append(element_boundary_mats)
        elif key == "rhs":
            output.append(trace_system.rhs)
        elif key == "solve_matrix_rows":
            output.append(solve_rows)
        elif key == "solve_matrix_cols":
            output.append(solve_cols)
        elif key == "solve_matrix_data":
            output.append(solve_data)
        elif key == "solve_rhs":
            output.append(solve_rhs)
        elif key == "boundary_trace":
            output.append(trace_system.boundary_trace)
        elif key == "reduction":
            output.append(reduction)
        elif key == "global_solve_result":
            output.append(global_solve_result)
        elif key == "timings":
            output.append(timings)
        else:
            raise ValueError(f"unknown return key {key!r}")
    return tuple(output)


diff_rea_hdg_solve = solve_diffusion_reaction_hdg


__all__ = [
    "DiffusionReactionAssemblyResult",
    "DiffusionReactionHDGOptions",
    "DiffusionReactionHDGSolver",
    "DiffusionReactionResult",
    "DiffusionReactionTimings",
    "diff_rea_hdg_solve",
    "flux_coefficients",
    "solve_diffusion_reaction_hdg",
]
