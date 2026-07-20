"""Compact HDG solver for linear advection-reaction problems.

This module is the :mod:`hdgfem` rewrite of the legacy
``adv_rea_vec_msh4.py`` solver.  The numerical structure is the same HDG
trace formulation, but the public API works with :class:`DGSpace`,
:class:`DGField`, and :class:`VectorDGField` objects instead of raw mesh and
quadrature tuples.
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import numpy as np

from ..assembly import hdg as hdg_assembly
from ..assembly import matrices_numpy as hdg_mats
from ..linalg.system import (
    KnownDofReduction,
    SolveResult,
    assemble_global_matrix,
    eliminate_known_dofs,
    expand_known_dofs,
    solve_global_system,
)
from ..linalg.ordering import (
    GraphOrderingResult,
    SparsePatternPlotResult,
    save_upwind_reordered_matrix_patterns,
    upwind_scc_trace_ordering,
)
from ..core.space import DGField, DGSpace, VectorDGField


ReturnKey = Literal[
    "trace",
    "trace_coeffs",
    "matrix_rows",
    "matrix_cols",
    "matrix_data",
    "local_solver",
    "element_boundary_mats",
    "ordering_result",
    "global_solve_result",
    "timings",
    "result",
]
AssemblyBackend = Literal["numpy", "numba", "cupy", "auto"]


_UNSET = object()


@dataclass(frozen=True)
class AdvectionReactionTimings:
    """Wall-clock timings for the main HDG solve phases."""

    preparation: float
    assembly: float
    solve: float
    reconstruction: float
    total: float
    boundary_elimination: float = 0.0
    trace_ordering: float = 0.0


@dataclass(frozen=True)
class AdvectionReactionResult:
    """Container returned by :func:`solve_advection_reaction_hdg`."""

    field: DGField | None
    trace: np.ndarray | None
    timings: AdvectionReactionTimings
    matrix_rows: np.ndarray | None = None
    matrix_cols: np.ndarray | None = None
    matrix_data: np.ndarray | None = None
    rhs: np.ndarray | None = None
    solve_matrix_rows: np.ndarray | None = None
    solve_matrix_cols: np.ndarray | None = None
    solve_matrix_data: np.ndarray | None = None
    solve_rhs: np.ndarray | None = None
    boundary_trace: np.ndarray | None = None
    reduction: KnownDofReduction | None = None
    local_solver: np.ndarray | None = None
    element_boundary_mats: np.ndarray | None = None
    boundary_mode: Literal["penalty", "eliminate"] = "penalty"
    trace_ordering: Literal["none", "upwind-scc"] = "none"
    assembly_backend: AssemblyBackend = "numpy"
    ordering_result: GraphOrderingResult | None = None
    global_solve_result: SolveResult | None = None
    matrix_pattern_plots: SparsePatternPlotResult | None = None


@dataclass(frozen=True)
class AdvectionReactionHDGOptions:
    """Configuration for :class:`AdvectionReactionHDGSolver`.

    The options mirror the keyword arguments accepted by
    :func:`solve_advection_reaction_hdg`, excluding the problem data and
    ``return_``.  Keeping the configuration in a dataclass gives repeated-solve
    workflows a stable object that can be copied with small overrides instead
    of rebuilding long keyword dictionaries by hand.
    """

    solver: str | None = "BICGSTAB"
    preconditioner: Any = "ilu"
    solver_rtol: float = 1e-13
    solver_atol: float = 0.0
    maxiter: int | None = None
    petsc_preset: str = "gmres_ilu"
    petsc_levels: int | None = None
    petsc_options: dict | None = None
    petsc_divtol: float = 1e4
    petsc_monitor: bool = False
    cupyx_solver: str = "bicgstab"
    amgx_config: dict | None = None
    ilu_drop_tol: float | None = None
    ilu_fill_factor: float | None = None
    ilu_failure: Literal["raise", "none"] = "raise"
    scale_system: bool | None = None
    boundary_penalty: float = 1e20
    boundary_mode: Literal["penalty", "eliminate"] = "penalty"
    trace_ordering: Literal["none", "upwind-scc"] = "none"
    trace_ordering_flux_tolerance: float = 0.0
    ilu_permc_spec: str | None = None
    matrix_pattern_dir: str | None = None
    matrix_pattern_prefix: str = "adv_rea_trace_matrix"
    matrix_pattern_max_points: int = 2_000_000
    matrix_pattern_dpi: int = 250
    matrix_pattern_only: bool = False
    assembly_backend: AssemblyBackend = "numpy"
    advection_stabilization: Any = None
    cache_local_solvers: bool = False
    verbose: bool | int = True

    def with_overrides(self, **overrides) -> "AdvectionReactionHDGOptions":
        """Return a copy with selected option values replaced.

        Unknown option names raise ``TypeError`` so misspelled solver settings
        fail early instead of being silently ignored in a repeated-solve loop.
        """
        if not overrides:
            return self
        valid = {field.name for field in fields(type(self))}
        unknown = sorted(set(overrides) - valid)
        if unknown:
            raise TypeError(f"unknown advection-reaction solver option(s): {', '.join(unknown)}")
        return replace(self, **overrides)

    def as_solve_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments for :func:`solve_advection_reaction_hdg`."""
        return {field.name: getattr(self, field.name) for field in fields(type(self))}


def _format_seconds(seconds: float) -> str:
    """Format elapsed wall time for concise solver logging."""
    if seconds >= 100.0:
        return f"{seconds:.1f}s"
    if seconds >= 1.0:
        return f"{seconds:.3f}s"
    return f"{seconds:.4f}s"


def _verbosity_level(verbose: bool | int) -> int:
    """Normalize bool/int verbosity flags to an integer level."""
    if isinstance(verbose, bool):
        return 1 if verbose else 0
    return max(0, int(verbose))


def _timed_call(label: str, verbosity: bool | int, function, *, level: int = 1, multiline: bool = False):
    """Run ``function`` with legacy-style one-line timing output."""
    should_print = _verbosity_level(verbosity) >= level
    if should_print:
        indent = "  " * (level - 1)
        label = f"{indent}{label}"
        if multiline:
            print(f"{label} ...", flush=True)
        else:
            print(f"{label} ... ", end="", flush=True)
    start = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - start
    if should_print:
        if multiline:
            print(f"{label} ... done in {_format_seconds(elapsed)}")
        else:
            print(f"done in {_format_seconds(elapsed)}")
    return result, elapsed


def _normalize_coefficient_values(values, space: DGSpace, num_points: int, label: str) -> np.ndarray:
    """Normalize scalar coefficient samples to ``(num_elements, num_points)``."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape == (space.mesh.num_tri, num_points):
        return values
    if values.shape == (num_points,):
        return np.broadcast_to(values[None, :], (space.mesh.num_tri, num_points))
    if values.ndim == 0:
        return np.full((space.mesh.num_tri, num_points), float(values), dtype=np.float64)
    raise ValueError(
        f"{label} must return a scalar, shape ({num_points},), or shape "
        f"({space.mesh.num_tri}, {num_points}); got {values.shape}"
    )


def _callable_beta_values_on_volume(beta: tuple[Callable, Callable], space: DGSpace) -> np.ndarray:
    """Evaluate callable advection coefficients on solution volume quadrature."""
    points = space.mapped_quads()
    num_points = space.quad_data.Krf_w.shape[0]
    values = np.empty((space.mesh.num_tri, num_points, 2), dtype=np.float64)
    values[..., 0] = _normalize_coefficient_values(
        beta[0](points[:, :, 0], points[:, :, 1]),
        space,
        num_points,
        "beta[0]",
    )
    values[..., 1] = _normalize_coefficient_values(
        beta[1](points[:, :, 0], points[:, :, 1]),
        space,
        num_points,
        "beta[1]",
    )
    return values


def _callable_beta_normal_flux(beta: tuple[Callable, Callable], space: DGSpace) -> np.ndarray:
    r"""Evaluate :math:`\beta\cdot n` on element-face quadrature."""
    face_points = space.quad_data.pts_fc.reshape(-1, 2)
    mapped_points = space.mesh.map_reference_points(face_points)
    num_face_quads = space.quad_data.weights_JGL.size
    num_flat_points = face_points.shape[0]
    beta_values = np.empty((space.mesh.num_tri, num_flat_points, 2), dtype=np.float64)
    beta_values[..., 0] = _normalize_coefficient_values(
        beta[0](mapped_points[:, :, 0], mapped_points[:, :, 1]),
        space,
        num_flat_points,
        "beta[0]",
    )
    beta_values[..., 1] = _normalize_coefficient_values(
        beta[1](mapped_points[:, :, 0], mapped_points[:, :, 1]),
        space,
        num_flat_points,
        "beta[1]",
    )
    beta_values = beta_values.reshape(space.mesh.num_tri, num_face_quads, 3, 2).transpose(0, 2, 1, 3)
    return np.einsum("Kfqd,Kfd->Kfq", beta_values, space.mesh.normals, optimize=True)


def _callable_advection_mats(space: DGSpace, beta: tuple[Callable, Callable]) -> np.ndarray:
    r"""Assemble advection matrices from callable coefficients without projection."""
    beta_values = _callable_beta_values_on_volume(beta, space)
    scaled_inv_t = space.mesh.aff_jacs[:, None, None] * space.mesh.inv_aff_mats_t
    return np.einsum(
        "Kqd,KdD,jq,Diq,q->Kij",
        beta_values,
        scaled_inv_t,
        space.quad_data.bas_of_quads,
        space.quad_data.dbas_of_quads,
        space.quad_data.Krf_w,
        optimize=["einsum_path", (0, 1), (0, 2), (0, 2), (0, 1)],
    )


def _is_callable_beta(beta) -> bool:
    """Return ``True`` for a two-component callable advection coefficient."""
    return (
        isinstance(beta, (tuple, list))
        and len(beta) == 2
        and not any(isinstance(component, DGField) for component in beta)
        and all(callable(component) for component in beta)
    )


def _as_beta_field(beta, space: DGSpace) -> VectorDGField:
    """Normalize DG advection coefficients without projecting callables."""
    if isinstance(beta, VectorDGField):
        if beta.dim != 2:
            raise ValueError("advection field must have two components")
        beta.components[0].space.assert_same_mesh(space)
        beta.components[1].space.assert_same_mesh(space)
        return beta
    if _is_callable_beta(beta):
        raise TypeError("callable beta is evaluated directly and should not be converted to a DG field")
    try:
        return (space * space).field(beta, name="beta_h")
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "beta must be a tuple of two callables, a two-component VectorDGField, "
            "a tuple/list of two DGField or coefficient arrays, or a compatible coefficient array"
        ) from exc


def _prepare_beta_data(beta, space: DGSpace) -> tuple[VectorDGField | None, np.ndarray, tuple[Callable, Callable] | None]:
    """Return DG beta data, normal fluxes, and callable beta data for assembly."""
    if _is_callable_beta(beta):
        beta_callables = (beta[0], beta[1])
        return None, _callable_beta_normal_flux(beta_callables, space), beta_callables

    beta_field = _as_beta_field(beta, space)
    beta_normal_flux = hdg_mats.advective_boundary_normal(beta_field, space)
    return beta_field, beta_normal_flux, None


class AdvectionReactionHDGSolver:
    r"""Stateful HDG solver/cache for linear advection-reaction problems.

    This class is the reusable counterpart to
    :func:`solve_advection_reaction_hdg`.  It keeps the mesh/space, solver
    options, problem data, and the most recent assembled/solved artifacts on one
    object so driver code can run repeated solves without manually threading a
    large collection of arrays through the application.

    The first implementation is deliberately conservative about reuse: changing
    the mesh, space, or any PDE coefficient clears assembled matrices,
    reductions, orderings, preconditioners, and solution fields.  The class
    still exposes those cached artifacts after each solve, which gives adaptive
    and continuation workflows a stable place to inspect or reuse data at a
    higher level.  More fine-grained reuse, for example preserving a matrix when
    only the source changes, can be added behind this API without changing user
    code.

    Parameters
    ----------
    space
        Scalar DG solution space.
    source, beta, reaction, boundary_condition
        Optional initial problem data.  Either provide all four here or call
        :meth:`set_problem` before :meth:`solve`.
    options
        Optional :class:`AdvectionReactionHDGOptions` instance.  Keyword
        arguments matching option names may also be supplied and are applied as
        overrides.

    Examples
    --------
    One-shot use with explicit setup::

        solver = AdvectionReactionHDGSolver(
            space,
            assembly_backend="numba",
            boundary_mode="eliminate",
            trace_ordering="upwind-scc",
        )
        solver.set_problem(source_h, beta_h, reaction_h, exact)
        result = solver.solve()

    Repeated source updates::

        solver.set_source(next_source_h)
        next_result = solver.solve()

    Mesh adaptivity::

        solver.set_space(new_space)
        solver.set_problem(new_source_h, new_beta_h, new_reaction_h, boundary)
        result = solver.solve()

    Notes
    -----
    ``set_discrete_problem`` is an alias for ``set_problem`` intended for code
    that already projects coefficients into DG fields before solving.  It keeps
    performance-oriented drivers explicit without adding a separate coefficient
    ownership model.
    """

    def __init__(
            self,
            space: DGSpace,
            *,
            source: Any = _UNSET,
            beta: Any = _UNSET,
            reaction: Any = _UNSET,
            boundary_condition: Callable | object = _UNSET,
            options: AdvectionReactionHDGOptions | None = None,
            **option_overrides,
    ) -> None:
        self.space = space
        self.options = (options or AdvectionReactionHDGOptions()).with_overrides(**option_overrides)

        self.source = None
        self.beta = None
        self.reaction = None
        self.boundary_condition = None
        self._problem_is_set = False

        self.clear_cache()

        provided = (
            source is not _UNSET,
            beta is not _UNSET,
            reaction is not _UNSET,
            boundary_condition is not _UNSET,
        )
        if any(provided):
            if not all(provided):
                raise ValueError(
                    "source, beta, reaction, and boundary_condition must be provided together"
                )
            self.set_problem(source, beta, reaction, boundary_condition)

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

    def with_options(self, **overrides) -> "AdvectionReactionHDGSolver":
        """Update solver options in place and clear stale solve artifacts.

        This is intended for repeated experiments where the discretized problem
        is unchanged but solver controls such as tolerances, boundary mode, or
        backend are changed.  The method conservatively clears all assembled
        data because several options affect matrix layout and boundary
        treatment.
        """
        self.options = self.options.with_overrides(**overrides)
        self.clear_cache()
        return self

    def set_space(self, space: DGSpace, *, keep_problem: bool = True) -> "AdvectionReactionHDGSolver":
        """Replace the DG space and invalidate all computed artifacts.

        Parameters
        ----------
        space
            New scalar solution space, typically built on an adapted mesh.
        keep_problem
            If ``True`` the stored problem inputs are kept.  This is convenient
            when they are callables or externally refreshed DG fields.  If the
            stored inputs are DG fields tied to the old space, call
            :meth:`set_problem` or :meth:`set_discrete_problem` before solving.
            If ``False``, problem inputs are also cleared.
        """
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
    ) -> "AdvectionReactionHDGSolver":
        """Build and install a new :class:`DGSpace` from ``mesh``.

        This is a convenience for adaptive workflows.  By default it preserves
        the current polynomial degree, basis family, and space name unless
        explicit replacements are supplied.
        """
        new_space = DGSpace(
            mesh,
            self.space.order if order is None else order,
            basis_type=self.space.reference.basis_type if basis_type is None else basis_type,
            name=self.space.name if name is None else name,
            **space_kwargs,
        )
        return self.set_space(new_space, keep_problem=keep_problem)

    def clear_problem(self) -> "AdvectionReactionHDGSolver":
        """Remove stored PDE inputs and invalidate all computed data."""
        self.source = None
        self.beta = None
        self.reaction = None
        self.boundary_condition = None
        self._problem_is_set = False
        self.clear_cache()
        return self

    def set_problem(self, source, beta, reaction, boundary_condition: Callable) -> "AdvectionReactionHDGSolver":
        """Set PDE data and invalidate assembled/solved artifacts.

        The inputs are the same objects accepted by
        :func:`solve_advection_reaction_hdg`: callables, DG fields, coefficient
        arrays, or scalars depending on the coefficient.  The class stores the
        objects by reference; if a callable or array is mutated externally, call
        one of the ``set_*`` methods or :meth:`clear_cache` before solving.
        """
        self.source = source
        self.beta = beta
        self.reaction = reaction
        self.boundary_condition = boundary_condition
        self._problem_is_set = True
        self.clear_cache()
        return self

    def set_discrete_problem(
            self,
            source_h,
            beta_h: VectorDGField,
            reaction_h,
            boundary_condition: Callable,
    ) -> "AdvectionReactionHDGSolver":
        """Set already-discretized coefficient data.

        ``source_h`` and ``reaction_h`` may be :class:`DGField` objects or
        coefficient arrays accepted by the solver.  ``beta_h`` should be a
        projected two-component :class:`VectorDGField`.  This method is an
        explicit alias for :meth:`set_problem`; it documents the intended use in
        high-performance loops where projection is managed outside the solver.
        """
        return self.set_problem(source_h, beta_h, reaction_h, boundary_condition)

    def set_source(self, source) -> "AdvectionReactionHDGSolver":
        """Replace only the source input and invalidate cached artifacts."""
        self._require_problem_or_partial_update()
        self.source = source
        self._problem_is_set = self.beta is not None and self.reaction is not None and self.boundary_condition is not None
        self.clear_cache()
        return self

    def set_beta(self, beta) -> "AdvectionReactionHDGSolver":
        """Replace the advection coefficient and invalidate cached artifacts."""
        self._require_problem_or_partial_update()
        self.beta = beta
        self._problem_is_set = self.source is not None and self.reaction is not None and self.boundary_condition is not None
        self.clear_cache()
        return self

    def set_reaction(self, reaction) -> "AdvectionReactionHDGSolver":
        """Replace the reaction coefficient and invalidate cached artifacts."""
        self._require_problem_or_partial_update()
        self.reaction = reaction
        self._problem_is_set = self.source is not None and self.beta is not None and self.boundary_condition is not None
        self.clear_cache()
        return self

    def set_boundary_condition(self, boundary_condition: Callable) -> "AdvectionReactionHDGSolver":
        """Replace the Dirichlet trace data and invalidate cached artifacts."""
        self._require_problem_or_partial_update()
        self.boundary_condition = boundary_condition
        self._problem_is_set = self.source is not None and self.beta is not None and self.reaction is not None
        self.clear_cache()
        return self

    def clear_cache(self) -> "AdvectionReactionHDGSolver":
        """Clear assembled matrices, factorization/preconditioner, and solution.

        Problem inputs and solver options are preserved.  This method is useful
        when external mutable arrays/callables have changed but the Python object
        identities stored on the solver are the same.
        """
        self.result: AdvectionReactionResult | None = None
        self.field: DGField | None = None
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
        self.reduction: KnownDofReduction | None = None

        self.local_solver: np.ndarray | None = None
        self.element_boundary_mats: np.ndarray | None = None
        self.ordering_result: GraphOrderingResult | None = None
        self.matrix_pattern_plots: SparsePatternPlotResult | None = None

        self.global_solve_result: SolveResult | None = None
        self.preconditioner = None
        self.timings: AdvectionReactionTimings | None = None
        return self

    def clear_factorization(self) -> "AdvectionReactionHDGSolver":
        """Drop the cached preconditioner/factorization from the last solve."""
        self.preconditioner = None
        if self.global_solve_result is not None:
            self.global_solve_result.preconditioner = None
        return self

    def clear_solution(self) -> "AdvectionReactionHDGSolver":
        """Drop only the latest trace, field, and solve diagnostics."""
        self.field = None
        self.trace = None
        self.global_solve_result = None
        self.preconditioner = None
        self.timings = None
        self.result = None
        return self

    def assemble_trace_system(self, **option_overrides) -> AdvectionReactionResult:
        """Assemble/order the trace system and return before the global solve.

        This calls :meth:`solve` with ``matrix_pattern_only=True`` and stores the
        assembled matrix data on the solver object.  It is useful for diagnostic
        runs and for workflows that want to inspect the trace sparsity before
        choosing a linear solver.
        """
        previous_options = self.options
        try:
            return self.solve(matrix_pattern_only=True, **option_overrides)
        finally:
            self.options = previous_options

    def solve(
            self,
            *,
            source: Any = _UNSET,
            beta: Any = _UNSET,
            reaction: Any = _UNSET,
            boundary_condition: Callable | object = _UNSET,
            **option_overrides,
    ) -> AdvectionReactionResult:
        """Assemble, solve, reconstruct, cache, and return the HDG result.

        Optional problem arguments update the stored problem before the solve.
        Optional keyword arguments matching :class:`AdvectionReactionHDGOptions`
        override options for this call and become the solver's stored options.
        """
        provided = (
            source is not _UNSET,
            beta is not _UNSET,
            reaction is not _UNSET,
            boundary_condition is not _UNSET,
        )
        if any(provided):
            if not all(provided):
                raise ValueError(
                    "source, beta, reaction, and boundary_condition must be provided together"
                )
            self.set_problem(source, beta, reaction, boundary_condition)

        if option_overrides:
            self.with_options(**option_overrides)

        self._require_problem()
        result = solve_advection_reaction_hdg(
            self.source,
            self.beta,
            self.reaction,
            self.boundary_condition,
            self.space,
            return_=("result",),
            **self.options.as_solve_kwargs(),
        )
        self._store_result(result)
        return result

    def _require_problem_or_partial_update(self) -> None:
        """Allow coefficient setters before a complete problem is available."""
        if self.source is None and self.beta is None and self.reaction is None and self.boundary_condition is None:
            return

    def _require_problem(self) -> None:
        if not self._problem_is_set:
            raise RuntimeError(
                "no complete advection-reaction problem is set; call set_problem(...) "
                "or pass source, beta, reaction, and boundary_condition to solve(...)"
            )

    def _store_result(self, result: AdvectionReactionResult) -> None:
        """Copy result artifacts into named cache attributes."""
        self.result = result
        self.field = result.field
        self.trace = result.trace
        self.timings = result.timings

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
        self.ordering_result = result.ordering_result
        self.matrix_pattern_plots = result.matrix_pattern_plots
        self.global_solve_result = result.global_solve_result
        self.preconditioner = None if result.global_solve_result is None else result.global_solve_result.preconditioner


def solve_advection_reaction_hdg(
        source,
        beta,
        reaction,
        boundary_condition: Callable,
        space: DGSpace,
        *,
        solver: str | None = "BICGSTAB",
        preconditioner="ilu",
        solver_rtol: float = 1e-13,
        solver_atol: float = 0.0,
        maxiter: int | None = None,
        petsc_preset: str = "gmres_ilu",
        petsc_levels: int | None = None,
        petsc_options: dict | None = None,
        petsc_divtol: float = 1e4,
        petsc_monitor: bool = False,
        cupyx_solver: str = "bicgstab",
        amgx_config: dict | None = None,
        ilu_drop_tol: float | None = None,
        ilu_fill_factor: float | None = None,
        ilu_failure: Literal["raise", "none"] = "raise",
        scale_system: bool | None = None,
        boundary_penalty: float = 1e20,
        boundary_mode: Literal["penalty", "eliminate"] = "penalty",
        trace_ordering: Literal["none", "upwind-scc"] = "none",
        trace_ordering_flux_tolerance: float = 0.0,
        ilu_permc_spec: str | None = None,
        matrix_pattern_dir: str | None = None,
        matrix_pattern_prefix: str = "adv_rea_trace_matrix",
        matrix_pattern_max_points: int = 2_000_000,
        matrix_pattern_dpi: int = 250,
        matrix_pattern_only: bool = False,
        assembly_backend: AssemblyBackend = "numpy",
        advection_stabilization=None,
        cache_local_solvers: bool = False,
        verbose: bool | int = True,
        return_: Iterable[ReturnKey] = ("result",),
):
    r"""Solve :math:`\beta\cdot\nabla u + r u = f` with an HDG trace system.

    Parameters
    ----------
    source
        Callable, :class:`DGField`, source moment array, or source values on
        ``space`` volume quadrature points.  Callables are evaluated on the
        quadrature rule; they are not projected by this solver.
    beta
        Tuple of two callables, two-component :class:`VectorDGField`, tuple of
        two :class:`DGField` objects, or compatible DG coefficient array.  A
        callable beta is evaluated directly on volume and face quadrature
        points.  A DG beta uses DG basis contractions, with the fastest path
        when both components live in ``space``.
    reaction
        Scalar constant, callable, :class:`DGField`, DG coefficient array, or
        reaction values on volume quadrature points.  A callable reaction is
        evaluated on quadrature points; a same-space :class:`DGField` uses the
        cached triple-product mass path.
    boundary_condition
        Dirichlet trace callable ``g(x, y)``.
    space
        Scalar solution DG space.
    solver
        Global trace solver name.  The default is ``"BICGSTAB"`` with ILU
        preconditioning.  Use ``"direct"`` or ``None`` for sparse direct solve.
        ``"amgx"``/``"pyamgx"`` and ``"cupyx"`` select GPU global solves
        after whichever assembly backend produced the host trace system.
    cupyx_solver
        Cupyx sparse Krylov method used when ``solver="cupyx"``.  Aliases such
        as ``solver="cupyx_bicgstab"`` select the method inline.
    preconditioner
        Preconditioner passed to :func:`hdgfem.linalg.system.solve_global_system`.
        The default ``"ilu"`` builds a SciPy ILU preconditioner.
    scale_system
        Controls left Jacobi row scaling before a Krylov solve.  ``None`` keeps
        the historical default: SciPy Krylov solves are scaled and PETSc solves
        are unscaled.  Set ``True`` or ``False`` to force the same policy when
        comparing solver backends.
    boundary_penalty
        Penalty used to impose boundary trace coefficients in the full trace
        system.
    boundary_mode
        ``"penalty"`` keeps the legacy full trace system with large boundary
        diagonal entries.  ``"eliminate"`` removes prescribed boundary trace
        dofs, solves only for free trace dofs, then reconstructs the full trace.
    trace_ordering
        ``"upwind-scc"`` builds an experimental edge-block ordering from the
        directed upwind graph and applies it as a symmetric matrix permutation
        before the global solve.  The default ``"none"`` keeps the assembled
        ordering.
    trace_ordering_flux_tolerance
        Mean face-normal flux tolerance used to classify inflow/outflow faces
        for ``trace_ordering="upwind-scc"``.
    ilu_permc_spec
        SuperLU ``spilu`` column permutation.  If omitted, the default is
        ``"COLAMD"`` without trace ordering and ``"NATURAL"`` with
        ``trace_ordering="upwind-scc"`` so SuperLU does not discard the
        experimental ordering.
    matrix_pattern_dir
        Optional directory where sparsity-pattern plots of the solve matrix are
        written before and after upwind SCC reordering.  When this is set, the
        upwind ordering is computed even if ``trace_ordering="none"``.
    matrix_pattern_only
        If ``True``, assemble/order/plot the matrix and return before the
        global solve.  This is useful for diagnosing large cases where ILU is
        expensive.
    assembly_backend
        Assembly backend.  ``"numpy"`` is the established vectorized reference
        path.  ``"numba"`` uses the experimental fused projected-coefficient
        trace assembly backend and requires projected source/beta coefficients.
        ``"auto"`` currently keeps the stable NumPy path.
        ``"cupy"`` assembles the dense local inverses and full trace system on
        the GPU, then returns host arrays for the existing solve/reconstruction
        pipeline.
    advection_stabilization
        Optional HDG advection stabilization :math:`\tau` on element faces.
        ``None`` selects the upwind value ``abs(beta_h . n)``.  The NumPy
        backend accepts scalars, callables, :class:`DGField` objects, same-space
        coefficient arrays, per-face constants, or face-quadrature values.  The
        Numba fused backend accepts ``None``, scalars, :class:`DGField` objects,
        or same-space coefficient arrays; project callable stabilizations before
        requesting ``assembly_backend="numba"``.
    cache_local_solvers
        If ``True``, retain the dense local inverse blocks i0.n the returned
        result.  The Numba backend computes these blocks for reconstruction but
        does not cache them in the result unless this flag or ``return_`` asks
        for them explicitly.
    verbose
        Verbosity level.  ``False`` disables logs, ``True``/``1`` prints one
        line per major solve phase, and ``2`` also prints assembly substeps.
    return_
        By default returns an :class:`AdvectionReactionResult`.  For legacy-like
        tuple output, request keys such as ``"trace"`` or ``"timings"``.
    """
    total_start = time.perf_counter()
    verbosity = _verbosity_level(verbose)
    if verbosity:
        print("\n----- DG FEM Advection-Reaction HDG Solve -----")
    if boundary_mode not in {"penalty", "eliminate"}:
        raise ValueError("boundary_mode must be 'penalty' or 'eliminate'")
    if trace_ordering not in {"none", "upwind-scc"}:
        raise ValueError("trace_ordering must be 'none' or 'upwind-scc'")
    if assembly_backend not in {"numpy", "numba", "cupy", "auto"}:
        raise ValueError("assembly_backend must be 'numpy', 'numba', 'cupy', or 'auto'")
    if ilu_permc_spec is None:
        ilu_permc_spec = "NATURAL" if trace_ordering == "upwind-scc" else "COLAMD"
    want = tuple(return_)

    effective_backend = assembly_backend
    if effective_backend == "auto":
        effective_backend = "numpy"
    solver_is_petsc = solver is not None and str(solver).lower() == "petsc"
    effective_scale_system = not solver_is_petsc if scale_system is None else bool(scale_system)
    effective_ilu_drop_tol = ilu_drop_tol
    if effective_ilu_drop_tol is None:
        effective_ilu_drop_tol = 1e-8 if boundary_mode == "eliminate" else 1e-10
    effective_ilu_fill_factor = ilu_fill_factor
    if effective_ilu_fill_factor is None:
        effective_ilu_fill_factor = 20 if boundary_mode == "eliminate" else 35

    def prepare_data():
        if effective_backend == "numba":
            if _is_callable_beta(beta):
                raise TypeError(
                    "assembly_backend='numba' requires projected beta. "
                    "Use VectorDGField((beta_x, beta_y), space) or --project-beta."
                )
            beta_field = _as_beta_field(beta, space)
            return beta_field, None, None, None, reaction

        beta_field, beta_normal_flux, beta_callables = _prepare_beta_data(beta, space)
        source_rhs = hdg_assembly.source_moments(source, space)
        return beta_field, beta_normal_flux, beta_callables, source_rhs, reaction

    (beta_h, beta_dot_normal, beta_callables, source_moments, reaction_h), preparation = _timed_call(
        "preparing coefficient data",
        verbosity,
        prepare_data,
    )

    trace_permutation = None
    plot_permutation = None
    ordering_result = None
    trace_ordering_time = 0.0
    preordered_trace_permutation = None
    numba_edge_order = None

    def trace_ordering_active_edges():
        if boundary_mode != "eliminate":
            return None
        active_edge_mask = np.ones(space.mesh.num_edg, dtype=bool)
        active_edge_mask[space.mesh.bnd_edges_inds] = False
        return np.flatnonzero(active_edge_mask).astype(np.int64)

    def build_trace_ordering():
        return upwind_scc_trace_ordering(
            space.mesh,
            beta_dot_normal,
            space.quad_data.edg_dof,
            active_edges=trace_ordering_active_edges(),
            flux_tolerance=trace_ordering_flux_tolerance,
        )

    def print_trace_ordering_diagnostics(ordering: GraphOrderingResult) -> None:
        diagnostics = ordering.diagnostics
        timings = diagnostics.timings
        levels = diagnostics.level_widths
        print(
            "  upwind SCC graph: "
            f"nodes={diagnostics.num_nodes:,}, edges={diagnostics.num_directed_edges:,}, "
            f"components={diagnostics.num_components:,}, "
            f"largest={diagnostics.largest_component_size:,}, "
            f"cyclic_nodes={diagnostics.cyclic_nodes:,}",
            flush=True,
        )
        print(
            "  upwind SCC timings: "
            f"pairs={timings.graph_pairs:.5f}s, csr={timings.csr:.5f}s, "
            f"scc={timings.scc:.5f}s, dag={timings.dag:.5f}s, "
            f"topo={timings.topological_order:.5f}s, "
            f"dof_perm={timings.dof_permutation:.5f}s, total={timings.total:.5f}s",
            flush=True,
        )
        print(
            "  upwind level widths: "
            f"levels={levels.num_levels:,}, max={levels.max_width:,}, "
            f"median={levels.median_width:.1f}, mean={levels.mean_width:.1f}, "
            f"top10_fraction={levels.top10_width_fraction:.3f}",
            flush=True,
        )

    def save_trace_ordering_diagnostics(ordering: GraphOrderingResult) -> Path | None:
        if matrix_pattern_dir is None:
            return None
        diagnostics = ordering.diagnostics
        timings = diagnostics.timings
        levels = diagnostics.level_widths
        output_dir = Path(matrix_pattern_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{matrix_pattern_prefix}_upwind_diagnostics.json"
        payload = {
            "num_nodes": diagnostics.num_nodes,
            "num_directed_edges": diagnostics.num_directed_edges,
            "num_components": diagnostics.num_components,
            "largest_component_size": diagnostics.largest_component_size,
            "cyclic_components": diagnostics.cyclic_components,
            "cyclic_nodes": diagnostics.cyclic_nodes,
            "level_widths": {
                "num_levels": levels.num_levels,
                "max_width": levels.max_width,
                "median_width": levels.median_width,
                "mean_width": levels.mean_width,
                "top10_width_fraction": levels.top10_width_fraction,
                "widths": list(levels.widths),
            },
            "timings": {
                "graph_pairs": timings.graph_pairs,
                "csr": timings.csr,
                "scc": timings.scc,
                "dag": timings.dag,
                "topological_order": timings.topological_order,
                "dof_permutation": timings.dof_permutation,
                "total": timings.total,
            },
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    preorder_numba_trace = (
        effective_backend == "numba"
        and trace_ordering == "upwind-scc"
        and matrix_pattern_dir is None
    )
    if preorder_numba_trace:
        if beta_dot_normal is None:
            beta_dot_normal, beta_flux_time = _timed_call(
                "assembling normal flux for trace ordering",
                verbosity,
                lambda: hdg_mats.advective_boundary_normal(beta_h, space),
                level=2,
            )
            preparation += beta_flux_time

        ordering_result, trace_ordering_time = _timed_call(
            "computing upwind SCC trace ordering",
            verbosity,
            build_trace_ordering,
            multiline=verbosity >= 2,
        )
        preordered_trace_permutation = ordering_result.dof_permutation
        plot_permutation = ordering_result.dof_permutation
        numba_edge_order = ordering_result.edge_order
        if verbosity >= 2:
            print_trace_ordering_diagnostics(ordering_result)

    reduction = None
    boundary_elimination = 0.0
    if effective_backend == "numba":
        from ..backends.numba import (
            assemble_local_advection_reaction_numba,
            assemble_projected_trace_system_eliminated_numba,
            assemble_projected_trace_system_numba,
        )

        if boundary_mode == "eliminate":
            trace_assembler = assemble_projected_trace_system_eliminated_numba
            trace_assembly_label = "assembling reduced projected trace system (numba)"
            trace_assembly_kwargs = {}
        else:
            trace_assembler = assemble_projected_trace_system_numba
            trace_assembly_label = "assembling projected trace system (numba)"
            trace_assembly_kwargs = {"boundary_penalty": boundary_penalty}
        trace_assembly_kwargs.update(
            {
                "edge_order": numba_edge_order,
                "beta_dot_normal": beta_dot_normal,
                "advection_stabilization": advection_stabilization,
            }
        )

        numba_trace, trace_assembly = _timed_call(
            trace_assembly_label,
            verbosity,
            lambda: trace_assembler(
                source,
                beta_h,
                reaction_h,
                boundary_condition,
                space,
                **trace_assembly_kwargs,
            ),
            multiline=verbosity >= 2,
        )
        trace_system = numba_trace.trace_system
        rows = trace_system.rows
        cols = trace_system.cols
        data = trace_system.data
        rhs = trace_system.rhs
        boundary_trace = trace_system.boundary_trace
        beta_dot_normal = numba_trace.beta_dot_normal
        reduction = numba_trace.reduction
        local_assembly = 0.0
        local_inverse = 0.0
        boundary_assembly = 0.0
        local_solver = None
        element_boundary_mats = None
        if verbosity >= 2:
            timings = numba_trace.timings
            timing_parts = [
                f"coefficients={timings.get('coefficient_validation', 0.0):.5f}s",
                f"boundary/flux={timings.get('boundary_trace_and_flux', 0.0):.5f}s",
            ]
            if "reduction_map" in timings:
                timing_parts.append(f"reduction={timings['reduction_map']:.5f}s")
            if "trace_weights" in timings:
                timing_parts.append(f"weights={timings['trace_weights']:.5f}s")
            timing_parts.extend(
                [
                    f"kernel={timings.get('kernel', 0.0):.5f}s",
                    f"rhs={timings.get('rhs_finalization', 0.0):.5f}s",
                ]
            )
            print("  numba trace assembly timings: " + ", ".join(timing_parts), flush=True)

        if cache_local_solvers or "local_solver" in want or "element_boundary_mats" in want:
            numba_local, local_assembly = _timed_call(
                "materializing local solver cache (numba)",
                verbosity,
                lambda: assemble_local_advection_reaction_numba(
                    space,
                    beta_field=beta_h,
                    beta_callables=None,
                    beta_dot_normal=beta_dot_normal,
                    reaction=reaction_h,
                    advection_stabilization=advection_stabilization,
                ),
                multiline=verbosity >= 2,
            )
            local_solver, local_inverse = _timed_call(
                "inverting cached local element matrices",
                verbosity,
                lambda: np.linalg.inv(numba_local.local_mats),
            )
            element_boundary_mats = numba_local.element_boundary_mats
    elif effective_backend == "cupy":
        from ..backends.cupy import (
            assemble_advection_reaction_trace_system_cupy,
            assemble_advection_reaction_trace_system_eliminated_cupy,
        )

        cupy_assembler = assemble_advection_reaction_trace_system_cupy
        cupy_label = "assembling global trace system (cupy)"
        cupy_kwargs = {"boundary_penalty": boundary_penalty}
        if boundary_mode == "eliminate":
            cupy_assembler = assemble_advection_reaction_trace_system_eliminated_cupy
            cupy_label = "assembling reduced trace system (cupy)"
            cupy_kwargs = {
                "transfer_local_solver": cache_local_solvers
                or "local_solver" in want
                or "element_boundary_mats" in want
            }

        cupy_trace, trace_assembly = _timed_call(
            cupy_label,
            verbosity,
            lambda: cupy_assembler(
                source_moments,
                beta_h,
                beta_callables,
                beta_dot_normal,
                reaction_h,
                boundary_condition,
                space,
                **cupy_kwargs,
            ),
            multiline=verbosity >= 2,
        )
        trace_system = cupy_trace.trace_system
        rows = trace_system.rows
        cols = trace_system.cols
        data = trace_system.data
        rhs = trace_system.rhs
        boundary_trace = trace_system.boundary_trace
        beta_dot_normal = cupy_trace.beta_dot_normal
        reduction = cupy_trace.reduction
        local_solver = cupy_trace.local_solver
        element_boundary_mats = cupy_trace.element_boundary_mats
        local_assembly = cupy_trace.timings.get("local_assembly", 0.0)
        local_inverse = cupy_trace.timings.get("local_inverse", 0.0)
        boundary_assembly = 0.0
        boundary_elimination = cupy_trace.timings.get("boundary_elimination", 0.0)
        trace_assembly = (
            cupy_trace.timings.get("trace_assembly", 0.0)
            + cupy_trace.timings.get("host_transfer", 0.0)
        )
        if verbosity >= 2:
            timings = cupy_trace.timings
            print(
                "  cupy trace assembly timings: "
                f"local={timings.get('local_assembly', 0.0):.5f}s, "
                f"inverse={timings.get('local_inverse', 0.0):.5f}s, "
                f"trace={timings.get('trace_assembly', 0.0):.5f}s, "
                f"elim={timings.get('boundary_elimination', 0.0):.5f}s, "
                f"host={timings.get('host_transfer', 0.0):.5f}s",
                flush=True,
            )
    else:
        tau_face, gamma_face = hdg_mats.advection_trace_weights_from_normal_flux(
            space,
            beta_dot_normal,
            advection_stabilization,
        )

        def assemble_local_mats():
            local_blocks, _ = _timed_call(
                "assembling boundary mass matrices",
                verbosity,
                lambda: np.ascontiguousarray(hdg_mats.boundary_mass_from_trace_stabilization(space, tau_face)),
                level=2,
            )
            scratch_blocks = np.empty_like(local_blocks)
            _timed_call(
                "accumulating reaction mass matrices",
                verbosity,
                lambda: hdg_mats.add_reaction_mass(
                    local_blocks,
                    reaction_h,
                    space,
                    scratch=scratch_blocks,
                ),
                level=2,
            )
            _timed_call(
                "assembling advection matrices",
                verbosity,
                lambda: (
                    hdg_mats.add_advection_mats(local_blocks, space, beta_h, scale=-1.0)
                    if beta_h is not None
                    else np.subtract(local_blocks, _callable_advection_mats(space, beta_callables), out=local_blocks)
                ),
                level=2,
            )
            return local_blocks

        local_mats, local_assembly = _timed_call(
            "assembling local element matrices",
            verbosity,
            assemble_local_mats,
            multiline=verbosity >= 2,
        )
        element_boundary_mats, boundary_assembly = _timed_call(
            "assembling element boundary coupling",
            verbosity,
            lambda: hdg_mats.element_boundary_mats_from_trace_weight(space, gamma_face),
        )

        local_solver, local_inverse = _timed_call(
            "inverting local element matrices",
            verbosity,
            lambda: np.linalg.inv(local_mats),
        )

        def assemble_global_trace_system():
            trace_lift, _ = _timed_call(
                "building weighted advection trace lift",
                verbosity,
                lambda: hdg_mats.advection_trace_lift_from_stabilization(space, tau_face),
                level=2,
            )
            trace_blocks, _ = _timed_call(
                "forming element trace Schur blocks",
                verbosity,
                lambda: hdg_assembly.element_to_trace_matrix_from_lift(
                    trace_lift,
                    local_solver,
                    element_boundary_mats,
                    space,
                ),
                level=2,
            )
            (matrix_rows, matrix_cols), _ = _timed_call(
                "building global COO index arrays",
                verbosity,
                lambda: hdg_assembly.trace_matrix_indices(space, interior_mass_mode="face"),
                level=2,
            )
            interior_mass_blocks, _ = _timed_call(
                "assembling weighted interior trace masses",
                verbosity,
                lambda: hdg_mats.advection_interior_trace_mass_blocks_from_weight(space, gamma_face),
                level=2,
            )
            matrix_data, _ = _timed_call(
                "assembling global COO data",
                verbosity,
                lambda: hdg_assembly.trace_matrix_data(
                    trace_blocks,
                    space,
                    boundary_penalty,
                    interior_mass_mode="face",
                    interior_mass_blocks=interior_mass_blocks,
                ),
                level=2,
            )
            (matrix_rhs, boundary_trace), _ = _timed_call(
                "assembling global RHS",
                verbosity,
                lambda: hdg_assembly.trace_rhs_from_lift(
                    trace_lift,
                    source_moments,
                    local_solver,
                    boundary_condition,
                    space,
                    boundary_penalty,
                ),
                level=2,
            )
            return matrix_rows, matrix_cols, matrix_data, matrix_rhs, boundary_trace

        (rows, cols, data, rhs, boundary_trace), trace_assembly = _timed_call(
            "assembling global trace system",
            verbosity,
            assemble_global_trace_system,
            multiline=verbosity >= 2,
        )

    solve_rows, solve_cols, solve_data, solve_rhs = rows, cols, data, rhs
    diagnostic_rows = hdg_assembly.free_trace_dofs(space)
    if boundary_mode == "eliminate" and reduction is None:
        def eliminate_boundary_trace():
            known_mask = ~hdg_assembly.free_trace_dofs(space)
            known_values = boundary_trace.ravel()
            return eliminate_known_dofs(rows, cols, data, rhs, known_mask, known_values)

        reduction, boundary_elimination = _timed_call(
            "eliminating boundary trace dofs",
            verbosity,
            eliminate_boundary_trace,
        )
        solve_rows, solve_cols, solve_data, solve_rhs = (
            reduction.rows,
            reduction.cols,
            reduction.data,
            reduction.rhs,
        )
        diagnostic_rows = None
    elif boundary_mode == "eliminate":
        solve_rows, solve_cols, solve_data, solve_rhs = (
            reduction.rows,
            reduction.cols,
            reduction.data,
            reduction.rhs,
        )
        diagnostic_rows = None
    if preordered_trace_permutation is not None and preordered_trace_permutation.shape != solve_rhs.shape:
        raise RuntimeError(
            "preordered trace assembly produced a permutation of shape "
            f"{preordered_trace_permutation.shape}, but the solve RHS has shape {solve_rhs.shape}"
        )
    should_build_upwind_ordering = trace_ordering == "upwind-scc" or matrix_pattern_dir is not None

    if should_build_upwind_ordering and ordering_result is None:
        ordering_result, trace_ordering_time = _timed_call(
            "computing upwind SCC trace ordering",
            verbosity,
            build_trace_ordering,
            multiline=verbosity >= 2,
        )
        trace_permutation = ordering_result.dof_permutation
        if trace_permutation.shape != solve_rhs.shape:
            raise RuntimeError(
                "trace ordering produced a permutation of shape "
                f"{trace_permutation.shape}, but the solve RHS has shape {solve_rhs.shape}"
            )
        plot_permutation = ordering_result.dof_permutation
        if trace_ordering != "upwind-scc":
            trace_permutation = None
        if verbosity >= 2:
            print_trace_ordering_diagnostics(ordering_result)

    matrix_pattern_plots = None
    if matrix_pattern_dir is not None:
        if plot_permutation is None:
            raise RuntimeError("matrix pattern plotting requires an upwind SCC permutation")
        diagnostics_path = None
        if ordering_result is not None:
            diagnostics_path = save_trace_ordering_diagnostics(ordering_result)

        def save_matrix_patterns():
            matrix = assemble_global_matrix(solve_rows, solve_cols, solve_data, solve_rhs.size)
            return save_upwind_reordered_matrix_patterns(
                matrix,
                plot_permutation,
                matrix_pattern_dir,
                prefix=matrix_pattern_prefix,
                max_plot_points=matrix_pattern_max_points,
                dpi=matrix_pattern_dpi,
            )

        matrix_pattern_plots, _ = _timed_call(
            "saving matrix sparsity pattern plots",
            verbosity,
            save_matrix_patterns,
        )
        if verbosity:
            print(f"  matrix pattern before: {matrix_pattern_plots.before_path}", flush=True)
            print(f"  matrix pattern after : {matrix_pattern_plots.after_path}", flush=True)
            if diagnostics_path is not None:
                print(f"  upwind diagnostics : {diagnostics_path}", flush=True)

    assembly = local_assembly + local_inverse + boundary_assembly + trace_assembly
    keep_local_solver = effective_backend == "numpy" or cache_local_solvers or "local_solver" in want

    if matrix_pattern_only:
        timings = AdvectionReactionTimings(
            preparation=preparation,
            assembly=assembly + boundary_elimination + trace_ordering_time,
            solve=0.0,
            reconstruction=0.0,
            total=time.perf_counter() - total_start,
            boundary_elimination=boundary_elimination,
            trace_ordering=trace_ordering_time,
        )
        return AdvectionReactionResult(
            field=None,
            trace=None,
            timings=timings,
            matrix_rows=solve_rows,
            matrix_cols=solve_cols,
            matrix_data=solve_data,
            rhs=rhs,
            solve_matrix_rows=solve_rows,
            solve_matrix_cols=solve_cols,
            solve_matrix_data=solve_data,
            solve_rhs=solve_rhs,
            boundary_trace=boundary_trace,
            reduction=reduction,
            local_solver=local_solver if keep_local_solver else None,
            element_boundary_mats=element_boundary_mats,
            boundary_mode=boundary_mode,
            trace_ordering=trace_ordering,
            assembly_backend=effective_backend,
            ordering_result=ordering_result,
            matrix_pattern_plots=matrix_pattern_plots,
        )

    upwind_level_widths = None
    if ordering_result is not None and ordering_result.diagnostics.largest_component_size == 1:
        upwind_level_widths = ordering_result.diagnostics.level_widths

    if boundary_mode == "penalty":
        solve_lambda = lambda: solve_global_system(
            solve_rows,
            solve_cols,
            solve_data,
            solve_rhs,
            solve_rhs.size,
            solver=solver,
            preconditioner=preconditioner,
            rtol=solver_rtol,
            atol=solver_atol,
            maxiter=maxiter,
            petsc_preset=petsc_preset,
            petsc_levels=petsc_levels,
            petsc_options=petsc_options,
            petsc_divtol=petsc_divtol,
            petsc_monitor=petsc_monitor,
            cupyx_solver=cupyx_solver,
            amgx_config=amgx_config,
            ilu_drop_tol=effective_ilu_drop_tol,
            ilu_fill_factor=effective_ilu_fill_factor,
            ilu_failure=ilu_failure,
            ilu_permc_spec=ilu_permc_spec,
            upwind_block_size=space.quad_data.edg_dof,
            upwind_level_widths=upwind_level_widths,
            scale_system=effective_scale_system,
            scale_matrix_in_place=effective_scale_system,
            permutation=trace_permutation,
            raise_on_nonconvergence=True,
            verbose=verbosity,
            diagnostic_rows=diagnostic_rows,
            diagnostic_label="free trace",
        )
    else:
        solve_lambda = lambda: solve_global_system(
            solve_rows,
            solve_cols,
            solve_data,
            solve_rhs,
            solve_rhs.size,
            solver=solver,
            preconditioner=preconditioner,
            rtol=solver_rtol,
            atol=solver_atol,
            maxiter=maxiter,
            petsc_preset=petsc_preset,
            petsc_levels=petsc_levels,
            petsc_options=petsc_options,
            petsc_divtol=petsc_divtol,
            petsc_monitor=petsc_monitor,
            cupyx_solver=cupyx_solver,
            amgx_config=amgx_config,
            ilu_drop_tol=effective_ilu_drop_tol,
            ilu_fill_factor=effective_ilu_fill_factor,
            ilu_failure=ilu_failure,
            ilu_permc_spec=ilu_permc_spec,
            upwind_block_size=space.quad_data.edg_dof,
            upwind_level_widths=upwind_level_widths,
            scale_system=effective_scale_system,
            scale_matrix_in_place=effective_scale_system,
            permutation=trace_permutation,
            raise_on_nonconvergence=True,
            verbose=verbosity,
        )

    global_solve_result, solve_time = _timed_call(
        "solving global system",
        verbosity,
        solve_lambda,
        multiline=verbosity >= 1,
    )
    if preordered_trace_permutation is not None:
        unpermuted = np.empty_like(global_solve_result.x)
        unpermuted[preordered_trace_permutation] = global_solve_result.x
        global_solve_result.x = unpermuted
        global_solve_result.permutation_elapsed_seconds = 0.0
        global_solve_result.permutation_size = int(preordered_trace_permutation.size)
    if reduction is None:
        trace = np.asarray(global_solve_result.x, dtype=np.float64)
    else:
        trace = expand_known_dofs(global_solve_result.x, reduction)

    can_use_projected_reconstruction = (
        effective_backend == "numba"
        or (
            effective_backend == "cupy"
            and isinstance(source, DGField)
            and isinstance(beta_h, VectorDGField)
            and (np.isscalar(reaction_h) or isinstance(reaction_h, DGField))
        )
    )
    if can_use_projected_reconstruction:
        from ..backends.numba import reconstruct_projected_field_numba

        label = "reconstructing element field (numba)"
        if effective_backend == "cupy":
            label = "reconstructing element field (numba after cupy assembly)"
        field, reconstruction = _timed_call(
            label,
            verbosity,
            lambda: reconstruct_projected_field_numba(
                trace,
                source,
                beta_h,
                reaction_h,
                space,
                advection_stabilization=advection_stabilization,
            ),
        )
    else:
        def reconstruct_from_local_solver():
            nonlocal local_solver, element_boundary_mats
            if local_solver is None or element_boundary_mats is None:
                raise RuntimeError(
                    "local solver cache is required for reconstruction when projected Numba reconstruction is unavailable"
                )
            if not isinstance(local_solver, np.ndarray) or not isinstance(element_boundary_mats, np.ndarray):
                from ..backends.cupy import asnumpy

                local_solver = np.ascontiguousarray(asnumpy(local_solver))
                element_boundary_mats = np.ascontiguousarray(asnumpy(element_boundary_mats))
            return hdg_assembly.reconstruct_field(trace, source_moments, local_solver, element_boundary_mats, space)

        field, reconstruction = _timed_call(
            "reconstructing element field",
            verbosity,
            reconstruct_from_local_solver,
        )

    timings = AdvectionReactionTimings(
        preparation=preparation,
        assembly=assembly + boundary_elimination + trace_ordering_time,
        solve=solve_time,
        reconstruction=reconstruction,
        total=time.perf_counter() - total_start,
        boundary_elimination=boundary_elimination,
        trace_ordering=trace_ordering_time,
    )
    result = AdvectionReactionResult(
        field=field,
        trace=trace,
        timings=timings,
        matrix_rows=rows,
        matrix_cols=cols,
        matrix_data=data,
        rhs=rhs,
        solve_matrix_rows=solve_rows,
        solve_matrix_cols=solve_cols,
        solve_matrix_data=solve_data,
        solve_rhs=solve_rhs,
        boundary_trace=boundary_trace,
        reduction=reduction,
        local_solver=local_solver if keep_local_solver else None,
        element_boundary_mats=element_boundary_mats,
        boundary_mode=boundary_mode,
        trace_ordering=trace_ordering,
        assembly_backend=effective_backend,
        ordering_result=ordering_result,
        global_solve_result=global_solve_result,
        matrix_pattern_plots=matrix_pattern_plots,
    )

    if want == ("result",):
        return result
    output = []
    for key in want:
        if key == "result":
            output.append(result)
        elif key in {"trace", "trace_coeffs"}:
            output.append(trace)
        elif key == "matrix_rows":
            output.append(rows)
        elif key == "matrix_cols":
            output.append(cols)
        elif key == "matrix_data":
            output.append(data)
        elif key == "local_solver":
            output.append(local_solver)
        elif key == "element_boundary_mats":
            output.append(element_boundary_mats)
        elif key == "ordering_result":
            output.append(ordering_result)
        elif key == "global_solve_result":
            output.append(global_solve_result)
        elif key == "timings":
            output.append(timings)
        else:
            raise ValueError(f"unknown return key {key!r}")
    return tuple(output)


adv_rea_hdg_solv = solve_advection_reaction_hdg


__all__ = [
    "AdvectionReactionHDGOptions",
    "AdvectionReactionHDGSolver",
    "AdvectionReactionResult",
    "AdvectionReactionTimings",
    "adv_rea_hdg_solv",
    "solve_advection_reaction_hdg",
]
