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

from hdgfem.runtime.precision import audit_arrays, REAL_DTYPE

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, fields, replace
from numbers import Real
from typing import Any, Literal

import numpy as np

from hdgfem.hdg import condensation as hdg_assembly
from hdgfem.solvers.capabilities import (
    normalize_assembly_backend,
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

from hdgfem.hdg.condensation_device import require_finite_device_values
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
    diffusion_trace_lift,
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

        self.source = None
        self.reaction = None
        self.boundary_condition = None
        self._problem_is_set = False
        self._raw_cuda_amgx_retry_solver_cache: dict[Any, Any] = {}

        self.clear_cache()

        provided = (
            source is not _UNSET,
            reaction is not _UNSET,
            boundary_condition is not _UNSET,
        )
        if any(provided):
            if not all(provided):
                raise ValueError("source, reaction, and boundary_condition must be provided together")
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
        options = self.options
        raw_block_size = resolve_raw_cuda_block_size(
            options.raw_block_size,
            equation="diffusion-reaction",
            order=self.space.order,
        )
        normalized_solver = (
            "" if options.solver is None
            else str(options.solver).replace("_", "-").lower()
        )
        if normalized_solver not in {"amgx", "pyamgx", "fb-hp-mg-pcg"}:
            raise ValueError(
                "assembly_backend='raw-cuda' requires solver='amgx' or "
                "solver='fb-hp-mg-pcg' for direct device solves"
            )
        matrix_format = str(options.raw_matrix_format).lower()
        if matrix_format == "auto":
            matrix_format = "bsr"
        if matrix_format not in {"csr", "bsr"}:
            raise ValueError(
                "assembly_backend='raw-cuda' with DiffusionReactionHDGSolver.solve "
                "requires raw_matrix_format='auto', 'csr', or 'bsr'"
            )
        native_requested = normalized_solver == "fb-hp-mg-pcg"
        native_policy = str(options.fb_hp_mg_preconditioner_policy).replace(
            "_", "-"
        ).lower()
        if native_policy not in {"standard", "fast", "robust"}:
            raise ValueError(
                "fb_hp_mg_preconditioner_policy must be 'standard', 'fast' or 'robust'"
            )
        if native_policy != "standard" and not native_requested:
            raise ValueError(
                f"fb_hp_mg_preconditioner_policy={native_policy!r} requires "
                "solver='fb-hp-mg-pcg'"
            )
        if native_requested and matrix_format != "bsr":
            raise ValueError("solver='fb-hp-mg-pcg' requires raw_matrix_format='auto' or 'bsr'")
        if native_requested and str(options.trace_basis).replace("_", "-").lower() != "legendre-modal":
            raise ValueError("solver='fb-hp-mg-pcg' requires trace_basis='legendre-modal'")
        if native_requested and options.scale_system not in {False, "none", "off", "false"}:
            raise ValueError("solver='fb-hp-mg-pcg' requires scale_system=False")
        if options.boundary_mode != "eliminate":
            raise ValueError("assembly_backend='raw-cuda' requires boundary_mode='eliminate'")
        if not _diffusion_is_identity(options.diffusion):
            raise NotImplementedError("raw-CUDA diffusion solve currently supports identity diffusion only")
        if not np.isscalar(options.stabilization):
            raise NotImplementedError("raw-CUDA diffusion solve currently supports scalar stabilization only")
        postprocess_mode = _normalize_hdg_postprocess_mode(options.hdg_postprocess)
        if postprocess_mode not in {"none", "flux"}:
            raise NotImplementedError(
                "raw-CUDA diffusion device solves support hdg_postprocess='none' or 'flux'"
            )

        from hdgfem.core.device import field_from_cupy_coefficients
        from hdgfem.runtime.optional import require_cupy
        from hdgfem.linalg.amgx.device_solver import (
                    PyAMGXCsrDeviceSolver,
                    solve_reduced_system_amgx_device,
                )
        from hdgfem.hdg.condensation_device import reconstruct_trace_cupy
        from hdgfem.core.device import as_cupy_space
        from hdgfem.mixed.cupy import (
                    assemble_projected_diffusion_trace_rhs_cached_cupy,
                    assemble_projected_diffusion_trace_rhs_eliminated_raw_cupy,
                    assemble_projected_diffusion_trace_system_eliminated_raw_cupy,
                    attach_schur_cholesky_cache_cupy,
                    build_trace_reference,
                    reconstruct_compact_diffusion_field_cupy,
                    solve_mixed_from_scalar_cholesky_cupy,
                )
        from hdgfem.mixed.raw_cuda.identity import (
                    reconstruct_projected_diffusion_field_raw_cuda,
                )

        cp = require_cupy()
        total_start = time.perf_counter()
        verbosity = _verbosity_level(options.verbose)
        if verbosity:
            print("\n----- DG FEM Diffusion-Reaction HDG Solve -----")

        trace_basis = str(options.trace_basis).replace("_", "-").lower()
        cspace = as_cupy_space(self.space)
        trace_ref = build_trace_reference(cspace, trace_basis)
        local_factor_policy = _normalize_local_factor_cache_policy(options.cache_local_factors)
        cache_local_factors = local_factor_policy == "schur-lu"
        use_hybrid_cholesky = local_factor_policy == "schur-cholesky"
        local_factor_key = (
            id(self.space),
            "identity-diffusion",
            id(self.reaction),
            float(options.stabilization),
            trace_basis,
        )

        operator_key = (
            id(self.space),
            trace_basis,
            matrix_format,
            int(raw_block_size),
            float(options.stabilization),
            id(self.reaction),
            local_factor_policy,
        )
        operator_cache_valid = (
            options.cache_device_matrix
            and self._raw_cuda_assembly_cache is not None
            and self._raw_cuda_operator_key == operator_key
        )

        def assemble_raw_full():
            """Assemble the complete reduced raw CUDA trace system."""
            from hdgfem.core.field_ops import coefficient_field

            source_input = coefficient_field(self.space, self.source, name="source_h")
            reaction_input = coefficient_field(self.space, self.reaction, name="reaction_h")
            assembled = assemble_projected_diffusion_trace_system_eliminated_raw_cupy(
                source_input,
                reaction_input,
                self.boundary_condition,
                float(options.stabilization),
                self.space,
                trace_basis=trace_basis,
                matrix_format=options.raw_matrix_format,
                block_size=raw_block_size,
                trace_ref=trace_ref,
                cache_local_factors=cache_local_factors,
                local_factor_kind=local_factor_policy if cache_local_factors else "schur-lu",
                local_factor_key=local_factor_key,
            )
            if use_hybrid_cholesky:
                assembled = attach_schur_cholesky_cache_cupy(
                    assembled,
                    reaction_input,
                    cspace,
                    trace_ref,
                    float(options.stabilization),
                )
            return assembled

        def assemble_raw_rhs():
            """Assemble only the reduced raw CUDA RHS for a cached operator."""
            if use_hybrid_cholesky:
                return assemble_projected_diffusion_trace_rhs_cached_cupy(
                    self.source,
                    self.boundary_condition,
                    cspace,
                    trace_ref,
                    self._raw_cuda_assembly_cache,
                )

            from hdgfem.core.field_ops import coefficient_field

            source_input = coefficient_field(self.space, self.source, name="source_h")
            reaction_input = coefficient_field(self.space, self.reaction, name="reaction_h")
            return assemble_projected_diffusion_trace_rhs_eliminated_raw_cupy(
                source_input,
                reaction_input,
                self.boundary_condition,
                float(options.stabilization),
                self.space,
                cached_raw=self._raw_cuda_assembly_cache.raw_assembly,
                trace_basis=trace_basis,
                block_size=raw_block_size,
                trace_ref=trace_ref,
                local_factor_key=local_factor_key,
            )

        if operator_cache_valid and self._raw_cuda_rhs_valid:
            assembly_result = self._raw_cuda_assembly_cache
            trace_assembly = 0.0
            if verbosity:
                print("  reusing cached raw-CUDA diffusion trace operator/RHS", flush=True)
        elif operator_cache_valid:
            assembly_result, trace_assembly = _timed_call(
                (
                    "assembling reduced RHS (cupy cached Schur-Cholesky operator)"
                    if use_hybrid_cholesky
                    else "assembling reduced RHS (raw-cuda cached operator)"
                ),
                verbosity,
                assemble_raw_rhs,
                multiline=_detailed_logging(verbosity),
            )
            self._raw_cuda_assembly_cache = assembly_result
            self._raw_cuda_rhs_valid = True
        else:
            assembly_result, trace_assembly = _timed_call(
                f"assembling reduced global trace system (raw-cuda {matrix_format})",
                verbosity,
                assemble_raw_full,
                multiline=_detailed_logging(verbosity),
            )
            if options.cache_device_matrix:
                self._raw_cuda_assembly_cache = assembly_result
                self._raw_cuda_operator_key = operator_key
                self._raw_cuda_rhs_valid = True

        if trace_assembly > 0.0:
            assembly_result.timings['solver.headline.wall'] = float(trace_assembly)
            assembly_result.timings['solver.headline.unaccounted'] = max(
                0.0, float(trace_assembly) - float(assembly_result.timings.get('total', 0.0))
            )

        if _detailed_logging(verbosity):
            raw_cache = assembly_result.raw_assembly
            cholesky_cache = assembly_result.schur_cholesky_cache
            if use_hybrid_cholesky and cholesky_cache is not None:
                factor_status = "reused" if operator_cache_valid else "created"
                factor_gib = int(cholesky_cache.local_factor_bytes) / (1024 ** 3)
                print(
                    f"  local Schur Cholesky cache: {factor_status}; "
                    f"{self.space.mesh.num_tri} elements, {factor_gib:.3f} GiB",
                    flush=True,
                )
                print("  local reconstruction backend: CuPy/cuBLAS", flush=True)
            elif cache_local_factors and raw_cache is not None:
                factor_status = "reused" if operator_cache_valid else "created"
                factor_gib = int(raw_cache.local_factor_bytes) / (1024 ** 3)
                print(
                    f"  local {local_factor_policy} cache: {factor_status}; "
                    f"{self.space.mesh.num_tri} elements, {factor_gib:.3f} GiB",
                    flush=True,
                )
            else:
                print("  local Schur factor cache: disabled", flush=True)
            assembly_timings = assembly_result.timings or {}
            compressed_format = matrix_format.upper()
            compressed_key = matrix_format
            timing_rows = (
                ("source moments", "wrapper.source_moments"),
                ("boundary trace", "wrapper.boundary_trace"),
                ("reference data", "wrapper.reference_data"),
                ("raw reference precompute", "raw.reference_precompute"),
                ("raw setup", "raw.setup"),
                (f"{compressed_format} pattern", f"raw.{compressed_key}_pattern.wrapper"),
                (f"{compressed_format} zero/allocation", f"raw.{compressed_key}_zero"),
                ("raw kernel host preparation", "raw.kernel.prepare"),
                ("raw kernel NVRTC JIT/load", "raw.kernel.jit"),
                ("raw kernel device execution", "raw.kernel.device"),
                ("raw kernel launch wall", "raw.kernel.wall"),
                ("raw accounted total", "raw.total"),
                ("raw unaccounted", "raw.unaccounted"),
                ("raw wall total", "raw.wall_total"),
                ("wrapper raw call", "wrapper.raw_call"),
                ("wrapper finalization", "wrapper.finalize"),
                ("wrapper accounted total", "wrapper.accounted"),
                ("wrapper unaccounted", "wrapper.unaccounted"),
                ("solver headline wall", "solver.headline.wall"),
                ("solver headline unaccounted", "solver.headline.unaccounted"),
                ("CuPy cache preparation/JIT", "cupy.local_cache.prepare_and_jit"),
                ("coupling-adjoint validation", "cupy.local_cache.coupling_validation"),
                ("dense scalar Schur construction", "cupy.local_cache.schur_build"),
                ("Schur symmetry validation", "cupy.local_cache.symmetry_validation"),
                ("batched Cholesky factorization", "cupy.local_cache.cholesky"),
                ("Cholesky cache finalization", "cupy.local_cache.finalize"),
                ("complete Cholesky attachment", "cupy.local_cache.total"),
            )
            visible_timings = [(label, key) for label, key in timing_rows if key in assembly_timings]
            if visible_timings:
                print("  diffusion device-assembly timings:", flush=True)
                for label, key in visible_timings:
                    print(f"    {label:<36} {float(assembly_timings[key]):.5f}s", flush=True)

        def reduced_initial_guess():
            """Normalize an initial trace guess for the reduced solve system."""
            guess = options.initial_guess if options.initial_guess is not None else self._raw_cuda_last_trace_reduced
            if guess is None:
                return None
            guess_cp = cp.asarray(guess, dtype=REAL_DTYPE)
            reduced_size = int(assembly_result.rhs.size)
            if guess_cp.size == reduced_size:
                return cp.ascontiguousarray(guess_cp.reshape((reduced_size,)))
            full_size = int(self.space.mesh.num_edg * self.edg_dof)
            if guess_cp.size == full_size:
                full = guess_cp.reshape((self.space.mesh.num_edg, self.edg_dof))
                return cp.ascontiguousarray(full[cspace.mesh.int_edges_inds].ravel())
            raise ValueError(
                f"initial_guess must have reduced trace size {reduced_size} or full trace size {full_size}; "
                f"got {guess_cp.size}"
            )

        audit_arrays('poisson-assembly', assembly_result, cspace)
        effective_scale_system = options.scale_system
        scale_mode = (
            "left" if effective_scale_system is True
            else "none" if effective_scale_system is False
            else str(effective_scale_system).lower()
        )
        solve_initial_guess = reduced_initial_guess()
        native_hierarchy_reused = False
        native_setup_wall = 0.0
        native_fallback_reason = None
        native_result = None
        native_fallback_guess = None
        native_physical_stats = None
        native_metrics: dict[str, float] = {}
        native_key = (
            operator_key, int(assembly_result.rhs.size), native_policy,
        )
        if native_requested and self._raw_cuda_fb_hp_mg_failed_key != native_key:
            try:
                from hdgfem.linalg.gpu.legendre_face_bsr import diagonal_block_positions
                from hdgfem.linalg.multigrid.face_hp import FaceBlockHpMgPcgSolver
                from hdgfem.linalg.gpu.sparse import (
                                    _assembly_device_csr_matrix,
                                    _device_compressed_matvec,
                                )
                from hdgfem.linalg.amgx.device_solver import _residual_stats_cp
                from hdgfem.runtime.optional import require_cupyx_sparse

                native_hierarchy_reused = (
                    self._raw_cuda_fb_hp_mg_solver is not None
                    and self._raw_cuda_fb_hp_mg_solver_key == native_key
                )
                if not native_hierarchy_reused:
                    setup_started = time.perf_counter()
                    if self._raw_cuda_fb_hp_mg_solver is not None:
                        self._raw_cuda_fb_hp_mg_solver.close()
                    raw = assembly_result.raw_assembly
                    if raw is None or raw.csr_pattern is None:
                        raise RuntimeError("face-BSR assembly did not retain diagonal metadata")
                    diagonal_positions = diagonal_block_positions(
                        assembly_result.indptr, assembly_result.indices,
                        raw.csr_pattern.mass_csr_block_pos,
                    )
                    self._raw_cuda_fb_hp_mg_solver = FaceBlockHpMgPcgSolver(
                        indptr=assembly_result.indptr,
                        indices=assembly_result.indices,
                        data=assembly_result.data,
                        degree=self.space.order,
                        diagonal_positions=diagonal_positions,
                        preconditioner_policy=native_policy,
                        verbose=verbosity,
                    )
                    cp.cuda.get_current_stream().synchronize()
                    native_setup_wall = time.perf_counter() - setup_started
                    self._raw_cuda_fb_hp_mg_solver_key = native_key
                # Use the same original coefficients for native true refreshes
                # and the final acceptance gate. A transformed-matrix residual
                # can fall just below tolerance while the original one does not;
                # PCGF must continue from that point, not enter a cold fallback.
                checked_at = time.perf_counter()
                sparse = require_cupyx_sparse()
                physical_matrix = _assembly_device_csr_matrix(assembly_result, cp, sparse)
                native_metrics["solve.fb_hp_mg.physical_check"] = time.perf_counter() - checked_at

                def assembly_matvec(vector):
                    return _device_compressed_matvec(physical_matrix, vector, sparse, cp)

                native_result = self._raw_cuda_fb_hp_mg_solver.solve(
                    assembly_result.rhs,
                    initial_guess=solve_initial_guess,
                    assembly_matvec=assembly_matvec,
                    true_residual_every=options.fb_hp_mg_true_residual_every,
                    store_residual_history=options.fb_hp_mg_residual_history,
                    rtol=options.solver_rtol,
                    atol=options.solver_atol,
                    maxiter=(
                        1000
                        if options.maxiter is None and native_policy == "robust"
                        else 500 if options.maxiter is None else options.maxiter
                    ),
                )
                native_metrics.update({
                    "solve.fb_hp_mg.setup": (
                        0.0 if native_hierarchy_reused
                        else float(self._raw_cuda_fb_hp_mg_solver.setup_seconds)
                    ),
                    "solve.fb_hp_mg.setup_outer": float(native_setup_wall),
                    "solve.fb_hp_mg.setup_outer_overhead": max(
                        0.0,
                        float(native_setup_wall) - (
                            0.0 if native_hierarchy_reused
                            else float(self._raw_cuda_fb_hp_mg_solver.setup_seconds)
                        ),
                    ),
                    "solve.fb_hp_mg.krylov": float(native_result.elapsed_seconds),
                    "solve.fb_hp_mg.workspace_bytes": float(
                        self._raw_cuda_fb_hp_mg_solver.workspace_bytes
                    ),
                    "solve.fb_hp_mg.symmetry_defect": float(
                        self._raw_cuda_fb_hp_mg_solver.symmetry_defect
                    ),
                    "solve.fb_hp_mg.positive_curvature": float(
                        self._raw_cuda_fb_hp_mg_solver.positive_curvature
                    ),
                    "solve.fb_hp_mg.best_iteration": float(
                        native_result.best_iteration or 0
                    ),
                    "solve.fb_hp_mg.best_residual": float(
                        native_result.residual_norm
                    ),
                    "solve.fb_hp_mg.terminal_residual": float(
                        native_result.terminal_residual_norm
                        if native_result.terminal_residual_norm is not None
                        else native_result.residual_norm
                    ),
                    "solve.fb_hp_mg.true_residual_checks": float(
                        native_result.true_residual_check_count
                    ),
                    "solve.fb_hp_mg.residual_restarts": float(
                        native_result.residual_restart_count
                    ),
                    "solve.fb_hp_mg.returned_best_iterate": float(
                        native_result.returned_best_iterate
                    ),
                })
                if not native_result.converged:
                    raise RuntimeError(
                        "FB-HP-MG-PCG failed the true-residual convergence gate: "
                        f"{native_result.relative_residual:.3e}"
                        + (f" ({native_result.breakdown_reason})"
                           if native_result.breakdown_reason else "")
                    )
                # Independently validate the returned assembly-basis vector,
                # even though native true checks now use this same action.
                checked_at = time.perf_counter()
                physical_residual = assembly_result.rhs - assembly_matvec(native_result.solution)
                native_physical_stats = _residual_stats_cp(
                    physical_residual, assembly_result.rhs,
                    rtol=options.solver_rtol, atol=options.solver_atol,
                )
                native_metrics["solve.fb_hp_mg.physical_check"] += time.perf_counter() - checked_at
                physical_norm, _, physical_relative, physical_target = native_physical_stats
                native_metrics["solve.fb_hp_mg.physical_residual"] = physical_norm
                if not np.isfinite(physical_norm) or physical_norm > physical_target:
                    raise RuntimeError(
                        "FB-HP-MG-PCG failed the original-matrix residual gate: "
                        f"{physical_relative:.3e} relative; "
                        f"{physical_norm:.3e} > {physical_target:.3e}"
                    )
                if verbosity:
                    print(
                        "  FB-HP-MG-PCG: "
                        f"hierarchy={'reused' if native_hierarchy_reused else 'created'} "
                        f"setup={native_setup_wall:.5f}s "
                        f"solve={native_result.elapsed_seconds:.5f}s "
                        f"iterations={native_result.iterations} "
                        f"true_rel={physical_relative:.3e}",
                        flush=True,
                    )
            except SyntaxError:
                # Import/programming errors are not numerical failures.
                raise
            except Exception as exc:
                if REAL_DTYPE == np.float32:
                    raise RuntimeError(
                        f"FP32 native Poisson solve failed: {exc}. "
                        "No alternate solver or higher-precision fallback was used."
                    ) from exc
                native_fallback_reason = f"{type(exc).__name__}: {exc}"
                if native_result is not None:
                    native_fallback_guess = native_result.solution
                    # A failed native result is a retry seed, never an accepted
                    # Poisson solution.  Clear this sentinel before dispatch.
                    native_result = None
                self._raw_cuda_fb_hp_mg_failed_key = native_key
                self._raw_cuda_fb_hp_mg_failure_reason = native_fallback_reason
                if self._raw_cuda_fb_hp_mg_solver is not None:
                    self._raw_cuda_fb_hp_mg_solver.close()
                self._raw_cuda_fb_hp_mg_solver = None
                self._raw_cuda_fb_hp_mg_solver_key = None
                if verbosity:
                    print(
                        "  FB-HP-MG-PCG gate failed; using cached hybrid AMGX "
                        f"fallback: {native_fallback_reason}",
                        flush=True,
                    )
        elif native_requested:
            native_fallback_reason = self._raw_cuda_fb_hp_mg_failure_reason

        amgx_hierarchy_reused = False
        if native_result is not None:
            from hdgfem.linalg.results import SolveResult

            trace_reduced_cp = native_result.solution
            solve_time = (
                native_setup_wall + native_result.elapsed_seconds
                + native_metrics.get("solve.fb_hp_mg.physical_check", 0.0)
            )
            global_solve_result = SolveResult(
                x=None, x_device=trace_reduced_cp,
                residual_norm=native_result.residual_norm, info=0,
                total_elapsed_seconds=solve_time,
                solve_elapsed_seconds=native_result.elapsed_seconds,
                iteration_count=native_result.iterations,
                initial_residual_norm=(
                    native_result.history[0] if native_result.history else None
                ),
                rhs_norm=native_result.rhs_norm,
                relative_residual_norm=native_result.relative_residual,
                residual_target=native_result.target,
                solver_residual_norm=native_result.residual_norm,
                solver_rhs_norm=native_result.rhs_norm,
                solver_relative_residual_norm=native_result.relative_residual,
                solver_residual_target=native_result.target,
                physical_residual_norm=native_physical_stats[0],
                physical_rhs_norm=native_physical_stats[1],
                physical_relative_residual_norm=native_physical_stats[2],
                physical_residual_target=native_physical_stats[3],
                rtol=options.solver_rtol, atol=options.solver_atol,
                backend="fb-hp-mg-pcg", backend_info=0, status="converged",
                converged=True, solution_is_finite=True,
                solver_residual_is_finite=True, physical_residual_is_finite=True,
                solver_residual_target_met=True, physical_residual_target_met=True,
                residual_history=native_result.history,
            )
            global_solve_result.cupyx_solver = "fb-hp-mg-pcg-device"
        else:
            reusable_solver = None
            if options.cache_device_matrix and scale_mode in {"none", "off", "false"}:
                solver_key = (
                    id(self.space), trace_basis, matrix_format, int(raw_block_size),
                    float(options.stabilization), int(assembly_result.rhs.size),
                    id(options.amgx_config), float(options.solver_rtol),
                    None if options.maxiter is None else int(options.maxiter),
                )
                amgx_hierarchy_reused = (
                    self._raw_cuda_amgx_solver is not None
                    and self._raw_cuda_amgx_solver_key == solver_key
                    and not getattr(self._raw_cuda_amgx_solver, "closed", False)
                )
                if not amgx_hierarchy_reused:
                    if self._raw_cuda_amgx_solver is not None:
                        self._raw_cuda_amgx_solver.close()
                    self._raw_cuda_amgx_solver = PyAMGXCsrDeviceSolver(
                        config=options.amgx_config, tolerance=options.solver_rtol,
                        maxiter=options.maxiter, verbose=verbosity, reusable=True,
                    )
                    self._raw_cuda_amgx_solver_key = solver_key
                reusable_solver = self._raw_cuda_amgx_solver
            if _detailed_logging(verbosity):
                print(
                    "  AMGX hierarchy/setup: "
                    + ("reused" if amgx_hierarchy_reused else "created on this solve"),
                    flush=True,
                )
            amgx_initial_guess = (
                native_fallback_guess
                if native_fallback_guess is not None
                else solve_initial_guess
            )
            retry_seed_guess = native_fallback_guess
            retry_seed_label = "native-fb-hp-mg-best"
            if (
                retry_seed_guess is None
                and options.amgx_retry_attempts
                and solve_initial_guess is not None
            ):
                retry_seed_guess = solve_initial_guess
                retry_seed_label = "poisson-initial-guess"
            (global_solve_result, trace_reduced_cp), amgx_solve_time = _timed_call(
                "solving global system (raw-cuda device hybrid AMGX)",
                verbosity,
                lambda: solve_reduced_system_amgx_device(
                    assembly_result, config=options.amgx_config,
                    retry_attempts=options.amgx_retry_attempts,
                    tolerance=options.solver_rtol, check_rtol=options.solver_rtol,
                    atol=options.solver_atol, maxiter=options.maxiter,
                    initial_guess=amgx_initial_guess,
                    reusable_solver=reusable_solver,
                    retry_solver_cache=self._raw_cuda_amgx_retry_solver_cache,
                    retry_seed_solution=retry_seed_guess,
                    retry_seed_label=retry_seed_label,
                    scale_system=effective_scale_system, raise_on_nonconvergence=True,
                    materialize_host_solution=False, verbose=verbosity,
                ),
                multiline=verbosity >= 1,
            )
            solve_time = (
                amgx_solve_time
                + native_setup_wall
                + native_metrics.get("solve.fb_hp_mg.krylov", 0.0)
                + native_metrics.get("solve.fb_hp_mg.physical_check", 0.0)
            )

        self._raw_cuda_last_trace_reduced = trace_reduced_cp

        def reconstruct_raw():
            """Recover trace and local fields through the selected local backend."""
            trace_cp = reconstruct_trace_cupy(trace_reduced_cp, assembly_result.boundary_trace, cspace)
            if use_hybrid_cholesky:
                cache = assembly_result.schur_cholesky_cache
                if cache is None:
                    raise RuntimeError("hybrid CuPy reconstruction cache is incomplete")
                if cache.compact:
                    uh_cp, local_unknowns_cp, _kernel_elapsed = (
                        reconstruct_compact_diffusion_field_cupy(trace_cp, cache, cspace)
                    )
                else:
                    element_boundary = assembly_result.element_boundary_mats
                    source_rhs = assembly_result.source_rhs
                    if element_boundary is None or source_rhs is None:
                        raise RuntimeError("hybrid CuPy reconstruction cache is incomplete")
                    trace_by_edge = trace_cp.reshape((cspace.mesh.num_edg, cspace.edg_dof))
                    element_traces = trace_by_edge[cspace.mesh.loc2glob_edge].reshape(
                        (cspace.mesh.num_tri, 3 * cspace.edg_dof),
                    )
                    rhs = source_rhs[..., None] + element_boundary @ element_traces[..., None]
                    local_unknowns_cp = solve_mixed_from_scalar_cholesky_cupy(cache, rhs).squeeze(-1)
                    local_unknowns_cp = cp.ascontiguousarray(
                        local_unknowns_cp.reshape((cspace.mesh.num_tri, 3 * cspace.el_dof))
                    )
                    uh_cp = cp.ascontiguousarray(local_unknowns_cp[:, : cspace.el_dof])
            else:
                raw = assembly_result.raw_assembly
                if raw is None:
                    raise RuntimeError("raw-CUDA diffusion assembly did not return raw reconstruction data")
                uh_cp, local_unknowns_cp, _kernel_elapsed = reconstruct_projected_diffusion_field_raw_cuda(
                    trace=trace_cp,
                    source_rhs=assembly_result.source_rhs,
                    cspace=cspace,
                    trace_ref=trace_ref,
                    d0_reference=raw.d0_reference,
                    d1_reference=raw.d1_reference,
                    face_element_mass=raw.face_element_mass,
                    tau=float(options.stabilization),
                    block_size=raw_block_size,
                    return_local_unknowns=True,
                    cached_factors=raw if cache_local_factors else None,
                    local_factor_key=local_factor_key,
                )
            cp.cuda.get_current_stream().synchronize()
            require_finite_device_values(local_unknowns_cp, "raw-CUDA diffusion reconstruction")
            nel = int(self.space.el_dof)
            field = field_from_cupy_coefficients(self.space, uh_cp, device=cspace.device_id, name="u_h")
            qx = field_from_cupy_coefficients(
                self.space,
                cp.ascontiguousarray(local_unknowns_cp[:, nel:2 * nel]),
                device=cspace.device_id,
                name="q_h_x",
            )
            qy = field_from_cupy_coefficients(
                self.space,
                cp.ascontiguousarray(local_unknowns_cp[:, 2 * nel:3 * nel]),
                device=cspace.device_id,
                name="q_h_y",
            )
            flux = VectorDGField((qx, qy), name="q_h")
            return field, flux, trace_cp, local_unknowns_cp

        (field, flux, trace_cp, local_unknowns_cp), reconstruction = _timed_call(
            (
                "reconstructing local fields (cupy/cuBLAS Cholesky)"
                if use_hybrid_cholesky
                else "reconstructing local fields (raw-cuda)"
            ),
            verbosity,
            reconstruct_raw,
        )
        audit_arrays("poisson-reconstruction", field, flux, trace_cp, local_unknowns_cp)

        details = {
            f"raw.assembly.{key}": float(value)
            for key, value in (assembly_result.timings or {}).items()
            if isinstance(value, (int, float))
        }
        details["raw.assembly.operator_reused"] = float(operator_cache_valid)
        details["raw.assembly.rhs_only"] = float(operator_cache_valid and trace_assembly > 0.0)
        details["raw.assembly.operator_and_rhs_reused"] = float(
            operator_cache_valid and trace_assembly == 0.0
        )
        details["raw.reconstruction.local_factors.reused"] = float(cache_local_factors)
        if use_hybrid_cholesky:
            cache = assembly_result.schur_cholesky_cache
            details["cupy.local_factors.bytes"] = float(cache.local_factor_bytes)
            details["cupy.local_factors.symmetry_error"] = float(cache.symmetry_error)
            details["cupy.local_factors.coupling_adjoint_error"] = float(cache.coupling_adjoint_error)
            details["cupy.local_factors.compact"] = float(cache.compact)
            details["cupy.local_factors.trace_response_bytes"] = float(
                cache.trace_response.nbytes if cache.trace_response is not None else 0
            )
            details["cupy.reconstruction.compact"] = float(cache.compact)
            details["cupy.reconstruction.local_factors.reused"] = 1.0
        details["solve.amgx.hierarchy_reused"] = float(amgx_hierarchy_reused)
        details["solve.fb_hp_mg.hierarchy_reused"] = float(native_hierarchy_reused)
        details["solve.fb_hp_mg.fallback"] = float(native_requested and native_result is None)
        details.update(native_metrics)
        if global_solve_result is not None:
            for detail_key, attr in (
                ("solve.amgx.csr", "amgx_csr_elapsed_seconds"),
                ("solve.amgx.matrix_unscale", "amgx_matrix_unscale_elapsed_seconds"),
                ("solve.scale", "scale_elapsed_seconds"),
                ("solve.amgx.setup", "amgx_setup_elapsed_seconds"),
                ("solve.amgx.matrix_upload", "amgx_matrix_upload_elapsed_seconds"),
                ("solve.amgx.solver_setup", "amgx_solver_setup_elapsed_seconds"),
                ("solve.amgx.solve", "amgx_solve_elapsed_seconds"),
                ("solve.amgx.total", "amgx_call_elapsed_seconds"),
                ("solve.validation.total", "solve_validation_elapsed_seconds"),
                ("solve.retry.matrix_backup_to_host", "amgx_retry_matrix_backup_elapsed_seconds"),
                ("solve.retry.matrix_restore_to_device", "amgx_retry_matrix_restore_elapsed_seconds"),
                ("solve.retry.wrapper", "amgx_retry_wrapper_elapsed_seconds"),
                ("solve.retry.outer_overhead", "amgx_retry_outer_overhead_elapsed_seconds"),
            ):
                value = getattr(global_solve_result, attr, None)
                if value is not None:
                    details[detail_key] = float(value)

        timings = DiffusionReactionTimings(
            preparation=0.0,
            local_solver=0.0,
            element_boundary=0.0,
            trace_assembly=trace_assembly,
            initial_guess=0.0,
            boundary_elimination=0.0,
            solve=solve_time,
            reconstruction=reconstruction,
            total=time.perf_counter() - total_start,
            details=dict(details),
        )
        device_postprocess = postprocess_mode == "flux" and _resolve_diffusion_postprocessing_backend(
            "raw-cuda", postprocess_mode,
            _normalize_flux_postprocess_space(options.flux_postprocess_space),
            options.postprocessing_backend,
        ) == "raw-cuda"
        host_postprocess = postprocess_mode == "flux" and not device_postprocess
        host_trace = cp.asnumpy(trace_cp) if host_postprocess else None
        host_local_unknowns = cp.asnumpy(local_unknowns_cp) if host_postprocess else None
        result = DiffusionReactionResult(
            field=field,
            flux=flux,
            trace=host_trace,
            timings=timings,
            trace_reduced_device=trace_reduced_cp,
            trace_device=trace_cp if device_postprocess else None,
            local_unknowns_device=local_unknowns_cp if device_postprocess else None,
            local_unknowns=host_local_unknowns,
            matrix_rows=None,
            matrix_cols=None,
            matrix_data=None,
            rhs=None,
            solve_matrix_rows=None,
            solve_matrix_cols=None,
            solve_matrix_data=None,
            solve_rhs=None,
            boundary_trace=None,
            reduction=None,
            local_solver=None,
            element_boundary_mats=None,
            initial_guess=None,
            boundary_mode="eliminate",
            scale_system=effective_scale_system,
            assembly_backend="raw-cuda",
            global_solve_result=global_solve_result,
        )
        return result

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
        options = self.options
        total_start = time.perf_counter()
        verbosity = _verbosity_level(options.verbose)
        if verbosity:
            print("\n----- DG FEM Diffusion-Reaction HDG Solve -----")
            print("  reusing cached reduced diffusion trace operator (numpy)", flush=True)

        effective_scale_system = (
            False if options.solver is not None and str(options.solver).lower() == "petsc" else options.scale_system
        )
        trace_basis = str(options.trace_basis).replace("_", "-").lower()
        trace_space = self.space.trace_space(trace_basis)

        def assemble_rhs():
            """Assemble the reduced RHS for the cached trace operator."""
            tau = normalize_diffusion_stabilization(options.stabilization, self.space)
            source_rhs = hdg_assembly.block_source_moments(self.source, self.space, num_blocks=3, source_block=0)
            trace_lift = diffusion_trace_lift(tau, self.space, trace_space=trace_space)
            rhs_full, boundary_trace = hdg_assembly.trace_rhs_from_lift(
                trace_lift,
                source_rhs,
                self.local_solver,
                self.boundary_condition,
                self.space,
                options.boundary_penalty,
                trace_space=trace_space,
            )
            solve_rhs, reduction = self._reduced_rhs_from_cached_numpy_operator(rhs_full, boundary_trace)
            return tau, source_rhs, boundary_trace, solve_rhs, reduction

        (tau, source_rhs, boundary_trace, solve_rhs, reduction), trace_assembly = _timed_call(
            "assembling reduced RHS (numpy cached operator)",
            verbosity,
            assemble_rhs,
            multiline=_detailed_logging(verbosity),
        )
        self.solve_rhs = solve_rhs
        self.boundary_trace = boundary_trace
        self.reduction = reduction
        self._host_cached_rhs_valid = True

        initial_guess = None
        if options.initial_guess is not None:
            initial_guess = impose_boundary_trace_on_guess(options.initial_guess, boundary_trace, self.space)
        solve_initial_guess = None if initial_guess is None else initial_guess[reduction.free_mask]

        global_solve_result, solve_time = _timed_call(
            "solving global system",
            verbosity,
            lambda: solve_global_system(
                self.solve_rows,
                self.solve_cols,
                self.solve_data,
                solve_rhs,
                solve_rhs.size,
                solver=options.solver,
                preconditioner=options.preconditioner,
                initial_guess=solve_initial_guess,
                rtol=options.solver_rtol,
                atol=options.solver_atol,
                maxiter=options.maxiter,
                ilu_drop_tol=options.ilu_drop_tol,
                ilu_fill_factor=options.ilu_fill_factor,
                ilu_failure=options.ilu_failure,
                ilu_permc_spec=options.ilu_permc_spec,
                petsc_preset=options.petsc_preset,
                petsc_levels=options.petsc_levels,
                petsc_options=options.petsc_options,
                petsc_divtol=options.petsc_divtol,
                petsc_monitor=options.petsc_monitor,
                cupyx_solver=options.cupyx_solver,
                amgx_config=options.amgx_config,
                scale_system=effective_scale_system,
                scale_matrix_in_place=effective_scale_system,
                raise_on_nonconvergence=True,
                verbose=verbosity,
            ),
            multiline=verbosity >= 1,
        )
        trace = expand_known_dofs(global_solve_result.x, reduction)

        def reconstruct():
            """Recover local mixed fields from the solved trace coefficients."""
            unknowns = hdg_assembly.reconstruct_local_unknowns(
                trace,
                source_rhs,
                self.local_solver,
                self.element_boundary_mats,
                self.space,
                trace_space=trace_space,
            )
            field, flux = split_diffusion_unknowns(unknowns, self.space)
            return unknowns, field, flux

        (local_unknowns, field, flux), reconstruction = _timed_call("reconstructing local fields", verbosity, reconstruct)
        timings = DiffusionReactionTimings(
            preparation=0.0,
            local_solver=0.0,
            element_boundary=0.0,
            trace_assembly=trace_assembly,
            initial_guess=0.0,
            boundary_elimination=0.0,
            solve=solve_time,
            reconstruction=reconstruction,
            total=time.perf_counter() - total_start,
        )
        return DiffusionReactionResult(
            field=field,
            flux=flux,
            trace=trace,
            timings=timings,
            local_unknowns=local_unknowns,
            matrix_rows=self.rows,
            matrix_cols=self.cols,
            matrix_data=self.data,
            rhs=None,
            solve_matrix_rows=self.solve_rows,
            solve_matrix_cols=self.solve_cols,
            solve_matrix_data=self.solve_data,
            solve_rhs=solve_rhs,
            boundary_trace=boundary_trace,
            reduction=reduction,
            local_solver=self.local_solver,
            element_boundary_mats=self.element_boundary_mats,
            initial_guess=initial_guess,
            boundary_mode="eliminate",
            scale_system=effective_scale_system,
            assembly_backend="numpy",
            global_solve_result=global_solve_result,
        )

    def _solve_numba_with_cached_operator(self) -> DiffusionReactionResult:
        """Solve with cached numba local solvers and reduced trace matrix."""
        from hdgfem.mixed.numba import (
                    assemble_projected_diffusion_trace_rhs_eliminated_numba,
                    reconstruct_projected_diffusion_local_unknowns_numba,
                )

        options = self.options
        trace_space = self.space.trace_space(options.trace_basis)
        total_start = time.perf_counter()
        verbosity = _verbosity_level(options.verbose)
        if verbosity:
            print("\n----- DG FEM Diffusion-Reaction HDG Solve -----")
            print("  reusing cached reduced diffusion trace operator (numba)", flush=True)

        effective_scale_system = (
            False if options.solver is not None and str(options.solver).lower() == "petsc" else options.scale_system
        )

        def prepare_source():
            """Validate source and reaction fields for projected Numba assembly."""
            source_input = _require_same_space_dg_field_for_backend(self.source, self.space, label="source", backend="numba")
            reaction_input = _require_same_space_dg_field_for_backend(self.reaction, self.space, label="reaction", backend="numba")
            return source_input, reaction_input

        (source_input, reaction_input), preparation = _timed_call(
            "preparing projected coefficient data",
            verbosity,
            prepare_source,
        )

        def assemble_rhs():
            """Assemble the reduced RHS for the cached trace operator."""
            return assemble_projected_diffusion_trace_rhs_eliminated_numba(
                source_input,
                reaction_input,
                self.boundary_condition,
                options.stabilization,
                self.space,
                trace_space=trace_space,
                cached_factors=self._numba_local_factors,
            )

        if self._host_cached_rhs_valid and self.solve_rhs is not None:
            solve_rhs, boundary_trace, reduction = self.solve_rhs, self.boundary_trace, self.reduction
            rhs_timings, trace_assembly = {"reused": 1.0}, 0.0
        else:
            (solve_rhs, boundary_trace, reduction, rhs_timings), trace_assembly = _timed_call(
                "assembling reduced RHS (numba cached operator)",
                verbosity,
                assemble_rhs,
            )
        reduction = type(reduction)(
            rows=self.solve_rows,
            cols=self.solve_cols,
            data=self.solve_data,
            rhs=solve_rhs,
            free_mask=reduction.free_mask,
            known_mask=reduction.known_mask,
            known_values=reduction.known_values,
            old_to_new=reduction.old_to_new,
        )
        self.solve_rhs = solve_rhs
        self.boundary_trace = boundary_trace
        self.reduction = reduction
        self._host_cached_rhs_valid = True
        if _detailed_logging(verbosity):
            print(
                "  numba cached RHS timings: "
                f"prep={rhs_timings.get('preparation', 0.0):.5f}s, "
                f"kernel={rhs_timings.get('kernel', 0.0):.5f}s, "
                f"rhs={rhs_timings.get('rhs_finalization', 0.0):.5f}s",
                flush=True,
            )

        initial_guess = None
        if options.initial_guess is not None:
            initial_guess = impose_boundary_trace_on_guess(options.initial_guess, boundary_trace, self.space)
        solve_initial_guess = None if initial_guess is None else initial_guess[reduction.free_mask]

        assembled_matrix, prepared_device_matrix = self._prepared_cupyx_operator(
            scale_system=effective_scale_system,
        )

        global_solve_result, solve_time = _timed_call(
            "solving global system",
            verbosity,
            lambda: solve_global_system(
                self.solve_rows,
                self.solve_cols,
                self.solve_data,
                solve_rhs,
                solve_rhs.size,
                solver=options.solver,
                preconditioner=options.preconditioner,
                initial_guess=solve_initial_guess,
                rtol=options.solver_rtol,
                atol=options.solver_atol,
                maxiter=options.maxiter,
                ilu_drop_tol=options.ilu_drop_tol,
                ilu_fill_factor=options.ilu_fill_factor,
                ilu_failure=options.ilu_failure,
                ilu_permc_spec=options.ilu_permc_spec,
                petsc_preset=options.petsc_preset,
                petsc_levels=options.petsc_levels,
                petsc_options=options.petsc_options,
                petsc_divtol=options.petsc_divtol,
                petsc_monitor=options.petsc_monitor,
                cupyx_solver=options.cupyx_solver,
                amgx_config=options.amgx_config,
                scale_system=effective_scale_system,
                scale_matrix_in_place=effective_scale_system and prepared_device_matrix is None,
                assembled_matrix=assembled_matrix,
                prepared_scaled_matrix=self._host_scaled_solve_matrix,
                prepared_inverse_diagonal=self._host_inverse_diagonal,
                prepared_device_matrix=prepared_device_matrix,
                raise_on_nonconvergence=True,
                verbose=verbosity,
            ),
            multiline=verbosity >= 1,
        )
        trace = expand_known_dofs(global_solve_result.x, reduction)

        def reconstruct():
            """Recover local mixed fields from the solved trace coefficients."""
            unknowns = reconstruct_projected_diffusion_local_unknowns_numba(
                trace,
                source_input,
                reaction_input,
                options.stabilization,
                self.space,
                trace_space=trace_space,
                cached_factors=self._numba_local_factors,
            )
            field, flux = split_diffusion_unknowns(unknowns, self.space)
            return unknowns, field, flux

        (local_unknowns, field, flux), reconstruction = _timed_call("reconstructing local fields", verbosity, reconstruct)

        timings = DiffusionReactionTimings(
            preparation=preparation,
            local_solver=0.0,
            element_boundary=0.0,
            trace_assembly=trace_assembly,
            initial_guess=0.0,
            boundary_elimination=0.0,
            solve=solve_time,
            reconstruction=reconstruction,
            total=time.perf_counter() - total_start,
            details={
                "numba.local_factors.reused": float(self._numba_local_factors is not None),
                "numba.local_factors.bytes": float(
                    self._numba_local_factors.local_factor_bytes if self._numba_local_factors else 0),
                **{f"numba.rhs.{key}": value for key, value in rhs_timings.items()},
            },
        )
        return DiffusionReactionResult(
            field=field,
            flux=flux,
            trace=trace,
            timings=timings,
            local_unknowns=local_unknowns,
            matrix_rows=self.rows if self.rows is not None else self.solve_rows,
            matrix_cols=self.cols if self.cols is not None else self.solve_cols,
            matrix_data=self.data if self.data is not None else self.solve_data,
            rhs=solve_rhs,
            solve_matrix_rows=self.solve_rows,
            solve_matrix_cols=self.solve_cols,
            solve_matrix_data=self.solve_data,
            solve_rhs=solve_rhs,
            boundary_trace=boundary_trace,
            reduction=reduction,
            local_solver=None,
            element_boundary_mats=None,
            initial_guess=initial_guess,
            boundary_mode="eliminate",
            scale_system=effective_scale_system,
            assembly_backend="numba",
            global_solve_result=global_solve_result,
        )

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
