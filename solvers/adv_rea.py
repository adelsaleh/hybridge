"""Compact HDG solver for linear advection-reaction problems.

This module is the :mod:`dgfem` rewrite of the legacy
``adv_rea_vec_msh4.py`` solver.  The numerical structure is the same HDG
trace formulation, but the public API works with :class:`DGSpace`,
:class:`DGField`, and :class:`VectorDGField` objects instead of raw mesh and
quadrature tuples.
"""

from __future__ import annotations

if __name__ == "__main__" and __package__ in {None, ""}:
    import runpy
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    runpy.run_module("dgfem.solvers.adv_rea", run_name="__main__")
    raise SystemExit

import time
from argparse import ArgumentParser
from dataclasses import dataclass, fields, replace
from math import pi
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
AssemblyBackend = Literal["numpy", "numba", "auto"]


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
    preconditioner
        Preconditioner passed to :func:`dgfem.linalg.system.solve_global_system`.
        The default ``"ilu"`` builds a SciPy ILU preconditioner.
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
    if assembly_backend not in {"numpy", "numba", "auto"}:
        raise ValueError("assembly_backend must be 'numpy', 'numba', or 'auto'")
    if ilu_permc_spec is None:
        ilu_permc_spec = "NATURAL" if trace_ordering == "upwind-scc" else "COLAMD"
    want = tuple(return_)

    effective_backend = assembly_backend
    if effective_backend == "auto":
        effective_backend = "numpy"

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
                ),
                multiline=verbosity >= 2,
            )
            local_solver, local_inverse = _timed_call(
                "inverting cached local element matrices",
                verbosity,
                lambda: np.linalg.inv(numba_local.local_mats),
            )
            element_boundary_mats = numba_local.element_boundary_mats
    else:
        def assemble_local_mats():
            local_blocks, _ = _timed_call(
                "assembling boundary mass matrices",
                verbosity,
                lambda: np.ascontiguousarray(hdg_mats.boundary_mass_from_normal_flux(space, beta_dot_normal)),
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
            lambda: hdg_mats.element_boundary_mats_from_normal_flux(space, beta_dot_normal),
        )

        local_solver, local_inverse = _timed_call(
            "inverting local element matrices",
            verbosity,
            lambda: np.linalg.inv(local_mats),
        )

        def assemble_global_trace_system():
            trace_blocks, _ = _timed_call(
                "forming element trace Schur blocks",
                verbosity,
                lambda: hdg_assembly.element_to_trace_matrix(local_solver, element_boundary_mats, space),
                level=2,
            )
            (matrix_rows, matrix_cols), _ = _timed_call(
                "building global COO index arrays",
                verbosity,
                lambda: hdg_assembly.trace_matrix_indices(space),
                level=2,
            )
            matrix_data, _ = _timed_call(
                "assembling global COO data",
                verbosity,
                lambda: hdg_assembly.trace_matrix_data(trace_blocks, space, boundary_penalty),
                level=2,
            )
            (matrix_rhs, boundary_trace), _ = _timed_call(
                "assembling global RHS",
                verbosity,
                lambda: hdg_assembly.global_rhs(source_moments, local_solver, boundary_condition, space, boundary_penalty),
                level=2,
            )
            return matrix_rows, matrix_cols, matrix_data, matrix_rhs, boundary_trace

        (rows, cols, data, rhs, boundary_trace), trace_assembly = _timed_call(
            "assembling global trace system",
            verbosity,
            assemble_global_trace_system,
            multiline=verbosity >= 2,
        )

    boundary_elimination = 0.0
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
            ilu_permc_spec=ilu_permc_spec,
            scale_system=True,
            scale_matrix_in_place=True,
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
            ilu_permc_spec=ilu_permc_spec,
            scale_system=True,
            ilu_drop_tol=1e-8,
            ilu_fill_factor=20,
            scale_matrix_in_place=True,
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

    if effective_backend == "numba":
        from ..backends.numba import reconstruct_projected_field_numba

        field, reconstruction = _timed_call(
            "reconstructing element field (numba)",
            verbosity,
            lambda: reconstruct_projected_field_numba(trace, source, beta_h, reaction_h, space),
        )
    else:
        field, reconstruction = _timed_call(
            "reconstructing element field",
            verbosity,
            lambda: hdg_assembly.reconstruct_field(trace, source_moments, local_solver, element_boundary_mats, space),
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


def test2(m: float = 10, n: float = 15, a: float = 2, b: float = 2):
    """Manufactured legacy advection-reaction test used by ``adv_rea_vec_msh4``."""

    def f2(t):
        return a * np.cos(m * pi * t) + b * np.sin(n * pi * t)

    return (
        lambda x, y: x + 0 * y,
        lambda x, y: -y + 0 * x,
        lambda x, y: y**2 + 0 * x,
        lambda x, y: y**2 + 0 * x,
        lambda x, y: f2(x * y) * np.exp(y**2 / 2.0) + 1.0,
    )


import numpy as np


def test3(
    r0=1.0,
    A=1.0,
    N=8,
    M=20.0,
    delta=0.35,
    u0=1.0,
    B=0.35,
    C=0.15,
    sigma=0.6,
    xc=0.0,
    yc=0.0,
    P=13.0 * np.pi,
    Q=17.0 * np.pi,
):
    """
    Build a manufactured solution for the conservative transport-reaction equation

        r(x,y) u(x,y) + div( u(x,y) beta(x,y) ) = f(x,y)

    on the square [-1,1]^2.

    The velocity field beta is generated from a streamfunction

        psi(x,y) = sin(a(x+1)) sin(a(y+1)),
        a = N*pi/2,

    through

        beta = A * grad^perp(psi)
             = A * (psi_y, -psi_x).

    Hence beta is exactly divergence-free:

        div(beta) = 0,

    and the source term is computed as

        f = r u + beta . grad(u).

    The manufactured exact solution has the form

        u(x,y)
        =
        u0
        + B sin(M psi(x,y) + delta x)
        + C exp(-sigma((x-xc)^2 + (y-yc)^2)) cos(Px + Qy).

    Parameters
    ----------
    r0 : float, default=1.0
        Constant reaction coefficient. The returned reaction function is
        reaction(x,y) = r0. Larger r0 makes the reaction term dominate the
        transport term.

    A : float, default=1.0
        Amplitude of the divergence-free velocity field beta. Increasing A
        strengthens advection and increases the magnitude of beta . grad(u).

    N : int or float, default=8
        Number of vortex cells per coordinate direction. The streamfunction
        creates approximately an N-by-N array of counter-rotating vortices on
        [-1,1]^2. Larger N gives smaller, more numerous vortices.

    M : float, default=20.0
        Oscillation frequency of the exact solution along the streamfunction
        psi. Larger M creates more oscillations inside and across vortex cells.

    delta : float, default=0.35
        Linear phase shift in the exact solution, appearing as delta*x inside
        sin(M*psi + delta*x). This prevents the exact solution from being purely
        a function of psi, which would make beta . grad(u) vanish for that part
        because beta is tangent to the level curves of psi.

    u0 : float, default=1.0
        Constant background level of the exact solution. Use this to keep u
        away from zero if desired.

    B : float, default=0.35
        Amplitude of the streamfunction-driven oscillatory part
        sin(M*psi + delta*x). Larger B increases the main vortex-aligned
        oscillations in u.

    C : float, default=0.15
        Amplitude of the localized Gaussian-trigonometric perturbation.
        Setting C=0 removes this extra localized high-frequency component.

    sigma : float, default=0.6
        Localization strength of the Gaussian factor

            exp(-sigma((x-xc)^2 + (y-yc)^2)).

        Larger sigma makes the perturbation more concentrated near (xc,yc).
        Smaller sigma spreads it over more of the square.

    xc : float, default=0.0
        x-coordinate of the center of the Gaussian perturbation.

    yc : float, default=0.0
        y-coordinate of the center of the Gaussian perturbation.

    P : float, default=13*pi
        x-frequency of the localized oscillatory perturbation cos(P*x + Q*y).
        Larger P gives faster oscillations in the x direction.

    Q : float, default=17*pi
        y-frequency of the localized oscillatory perturbation cos(P*x + Q*y).
        Larger Q gives faster oscillations in the y direction.

    Returns
    -------
    betax : callable
        Function betax(x,y) returning the first component beta_x(x,y).

    betay : callable
        Function betay(x,y) returning the second component beta_y(x,y).

    reaction : callable
        Function reaction(x,y) returning the reaction coefficient r(x,y).
        Here this is the constant function r0.

    source : callable
        Function source(x,y) returning the manufactured right-hand side f(x,y)
        such that the returned exact solution satisfies

            r u + div(u beta) = f.

    exact : callable
        Function exact(x,y) returning the manufactured exact solution u(x,y).

    Notes
    -----
    The returned functions are NumPy-vectorized lambdas: x and y may be scalars
    or NumPy arrays of matching shape.

    The boundary-normal velocity vanishes on the boundary of [-1,1]^2 because
    the streamfunction psi vanishes there. Thus beta . n = 0 on the boundary.
    """

    a = 0.5 * N * np.pi

    psi = lambda x, y: (
        np.sin(a * (x + 1.0)) * np.sin(a * (y + 1.0))
    )

    psix = lambda x, y: (
        a * np.cos(a * (x + 1.0)) * np.sin(a * (y + 1.0))
    )

    psiy = lambda x, y: (
        a * np.sin(a * (x + 1.0)) * np.cos(a * (y + 1.0))
    )

    betax = lambda x, y: A * psiy(x, y)
    betay = lambda x, y: -A * psix(x, y)

    reaction = lambda x, y: r0 + 0.0 * x * y

    theta = lambda x, y: M * psi(x, y) + delta * x

    G = lambda x, y: np.exp(
        -sigma * ((x - xc) ** 2 + (y - yc) ** 2)
    )

    Phi = lambda x, y: P * x + Q * y

    exact = lambda x, y: (
        u0
        + B * np.sin(theta(x, y))
        + C * G(x, y) * np.cos(Phi(x, y))
    )

    ux = lambda x, y: (
        B * np.cos(theta(x, y)) * (M * psix(x, y) + delta)
        + C
        * G(x, y)
        * (
            -2.0 * sigma * (x - xc) * np.cos(Phi(x, y))
            - P * np.sin(Phi(x, y))
        )
    )

    uy = lambda x, y: (
        B * np.cos(theta(x, y)) * M * psiy(x, y)
        + C
        * G(x, y)
        * (
            -2.0 * sigma * (y - yc) * np.cos(Phi(x, y))
            - Q * np.sin(Phi(x, y))
        )
    )

    # Since beta = A grad^perp(psi), div(beta) = 0.
    #
    # Therefore:
    #
    #   source = r exact + div(exact beta)
    #          = r exact + beta . grad(exact)
    #
    source = lambda x, y: (
        reaction(x, y) * exact(x, y)
        + betax(x, y) * ux(x, y)
        + betay(x, y) * uy(x, y)
    )

    return betax, betay, reaction, source, exact


__all__ = [
    "AdvectionReactionHDGOptions",
    "AdvectionReactionHDGSolver",
    "AdvectionReactionResult",
    "AdvectionReactionTimings",
    "adv_rea_hdg_solv",
    "solve_advection_reaction_hdg",
    "test2",
]


def _main() -> None:
    """Run the legacy manufactured advection-reaction test."""
    from ..core.mesh import gmsh_disc_mesh, gmsh_rectangle_mesh, gmsh_triangle_mesh, rectangle_mesh
    from ..io.plot import plot_solution_comparison
    from ..core.space import DGSpace
    from ..io.output import pretty_print_ncol

    parser = ArgumentParser(description="Run the dgfem advection-reaction HDG test2 problem.")
    parser.add_argument("--order", "-p", type=int, default=2, help="uniform DG polynomial order")
    parser.add_argument(
        "--domain",
        default="rectangle",
        choices=("rectangle", "disc", "triangle", "structured-rectangle"),
        help="mesh domain; rectangle/disc/triangle use Gmsh",
    )
    parser.add_argument("--mesh-size", "--lc", type=float, default=0.35, help="Gmsh target mesh size")
    parser.add_argument("--nx", type=int, default=8, help="structured rectangle cells in x")
    parser.add_argument("--ny", type=int, default=None, help="structured rectangle cells in y; defaults to nx")
    parser.add_argument("--gmsh-verbosity", type=int, default=0, help="Gmsh verbosity level")
    parser.add_argument("--gmsh-algorithm", type=int, default=None, help="optional Gmsh 2D meshing algorithm")
    parser.add_argument("--basis", default="dub_orth", choices=("bernstein", "hier_C0", "dub_orth"))
    parser.add_argument("--solver", default="BICGSTAB", help="global trace solver; use 'direct' for sparse direct solve")
    parser.add_argument("--preconditioner", default="ilu", choices=("ilu", "none"), help="global trace preconditioner")
    parser.add_argument("--solver-rtol", type=float, default=1e-13, help="relative tolerance for iterative solves")
    parser.add_argument("--solver-atol", type=float, default=0.0, help="absolute tolerance for iterative solves")
    parser.add_argument("--maxiter", type=int, default=None, help="maximum Krylov iterations")
    parser.add_argument(
        "--ilu-permc-spec",
        default=None,
        choices=("NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"),
        help="SuperLU spilu column permutation; defaults to NATURAL when trace ordering is enabled, else COLAMD",
    )
    parser.add_argument(
        "--project-reaction",
        action="store_true",
        help="project callable reaction into Vh before calling the solver",
    )
    parser.add_argument(
        "--project-source",
        action="store_true",
        help="project callable source into Vh before calling the solver",
    )
    parser.add_argument(
        "--project-beta",
        action="store_true",
        help="project callable advection field into Vh x Vh before calling the solver",
    )
    parser.add_argument(
        "--assembly-backend",
        default="numpy",
        choices=("numpy", "numba", "auto"),
        help="assembly backend; 'numba' is an explicit experimental projected-coefficient fused path",
    )
    parser.add_argument(
        "--cache-local-solvers",
        action="store_true",
        help="retain dense local inverse blocks in the returned result",
    )
    parser.add_argument(
        "--boundary-mode",
        default="penalty",
        choices=("penalty", "eliminate"),
        help="Dirichlet trace treatment: legacy penalty rows or reduced known-dof elimination",
    )
    parser.add_argument(
        "--trace-ordering",
        default="none",
        choices=("none", "upwind-scc"),
        help="optional trace-DOF ordering before the global solve",
    )
    parser.add_argument(
        "--trace-ordering-flux-tol",
        type=float,
        default=0.0,
        help="mean face-normal flux tolerance for upwind SCC trace ordering",
    )
    parser.add_argument(
        "--plot-matrix-pattern",
        action="store_true",
        help="write sparse matrix pattern plots before and after upwind SCC ordering",
    )
    parser.add_argument(
        "--matrix-pattern-dir",
        default="matrix_patterns",
        help="output directory for --plot-matrix-pattern",
    )
    parser.add_argument(
        "--matrix-pattern-max-points",
        type=int,
        default=2_000_000,
        help="maximum plotted nonzeros per matrix-pattern figure",
    )
    parser.add_argument(
        "--matrix-pattern-dpi",
        type=int,
        default=250,
        help="DPI for matrix-pattern PNG files",
    )
    parser.add_argument(
        "--matrix-pattern-only",
        action="store_true",
        help="assemble and plot matrix patterns, then stop before the global solve",
    )
    parser.add_argument("--verbosity", "-v", type=int, default=1, help="logging verbosity: 0 quiet, 1 phases, 2 substeps")
    parser.add_argument("--quiet", action="store_true", help="suppress phase timing output")
    parser.add_argument("--plot", action="store_true", help="plot numerical, exact, and absolute-error fields")
    parser.add_argument("--plot-resolution", type=int, default=20, help="samples per reference axis for plotting")
    parser.add_argument(
        "--exact-plot-resolution",
        type=int,
        default=None,
        help="exact-solution panel resolution; default uses an automatic dense reference sampling",
    )
    parser.add_argument("--hide-mesh", action="store_true", help="do not overlay the coarse mesh on plots")
    args = parser.parse_args()

    verbosity = 0 if args.quiet else max(0, int(args.verbosity))

    def build_mesh():
        if args.domain == "structured-rectangle":
            return rectangle_mesh(args.nx, args.ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
        if args.domain == "rectangle":
            return gmsh_rectangle_mesh(
                args.mesh_size,
                xlim=(-1.0, 1.0),
                ylim=(-1.0, 1.0),
                verbosity=args.gmsh_verbosity,
                algorithm=args.gmsh_algorithm,
            )
        if args.domain == "disc":
            return gmsh_disc_mesh(
                args.mesh_size,
                center=(0.0, 0.0),
                radius=1.0,
                verbosity=args.gmsh_verbosity,
                algorithm=args.gmsh_algorithm,
            )
        return gmsh_triangle_mesh(
            args.mesh_size,
            vertices=((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)),
            verbosity=args.gmsh_verbosity,
            algorithm=args.gmsh_algorithm,
        )

    mesh, _ = _timed_call(f"generating {args.domain} mesh", verbosity, build_mesh)
    space = DGSpace(mesh, args.order, basis_type=args.basis)
    beta_x, beta_y, reaction, source, exact = test2()
    # beta_x, beta_y, reaction, source, exact = test3(N=2, sigma=0.0, P=0, Q=0)
    bd_cond = exact
    source_input = DGField(source, space, name="source_h") if args.project_source else source
    reaction_input = DGField(reaction, space, name="reaction_h") if args.project_reaction else reaction
    beta_input = VectorDGField((beta_x, beta_y), space, name="beta_h") if args.project_beta else (beta_x, beta_y)
    plot_matrix_pattern = args.plot_matrix_pattern or args.matrix_pattern_only
    result = solve_advection_reaction_hdg(
        source_input,
        beta_input,
        reaction_input,
        bd_cond,
        space,
        solver=args.solver,
        preconditioner=None if args.preconditioner == "none" else args.preconditioner,
        solver_rtol=args.solver_rtol,
        solver_atol=args.solver_atol,
        maxiter=args.maxiter,
        boundary_mode=args.boundary_mode,
        trace_ordering=args.trace_ordering,
        trace_ordering_flux_tolerance=args.trace_ordering_flux_tol,
        ilu_permc_spec=args.ilu_permc_spec,
        matrix_pattern_dir=args.matrix_pattern_dir if plot_matrix_pattern else None,
        matrix_pattern_prefix=(
            f"adv_rea_p{space.order}_ne{mesh.num_tri}_bd-{args.boundary_mode}"
        ),
        matrix_pattern_max_points=args.matrix_pattern_max_points,
        matrix_pattern_dpi=args.matrix_pattern_dpi,
        matrix_pattern_only=args.matrix_pattern_only,
        assembly_backend=args.assembly_backend,
        cache_local_solvers=args.cache_local_solvers,
        verbose=verbosity,
    )
    if args.matrix_pattern_only:
        if result.matrix_pattern_plots is not None:
            print(f"matrix pattern before: {result.matrix_pattern_plots.before_path}")
            print(f"matrix pattern after : {result.matrix_pattern_plots.after_path}")
        return

    l2_error = result.field.l2_error(exact)
    numerical_values = result.field.values()
    points = space.mapped_quads()
    exact_values = exact(points[:, :, 0], points[:, :, 1])
    abs_error = np.abs(numerical_values - exact_values)
    linfty_error = float(np.max(abs_error))
    element_max_error = np.max(abs_error, axis=1)
    avg_error = float(np.average(element_max_error))
    max_error_element = int(np.argmax(element_max_error))

    global_solve = result.global_solve_result
    items = [
        ("p", space.order, ",d"),
        ("#triangles", mesh.num_tri, ",d"),
        ("# edges", mesh.num_edg, ",d"),
        ("#global_dof", result.trace.size, ",d"),
        ("ℓ_c (Gmsh)", args.mesh_size, ".3f"),
        ("h^p", mesh.h ** (space.order + 1), ".4e"),
        ("L₂ error", l2_error, ".4e"),
        ("L∞ error", linfty_error, ".4e"),
        ("avg error", avg_error, ".4e"),
        ("max_err at el", max_error_element, "d"),
        ("prep time(s)", result.timings.preparation, "1.1f"),
        ("setup time(s)", result.timings.assembly, "1.1f"),
        ("glb_solve time(s)", result.timings.solve, "1.1f"),
        ("recons time(s)", result.timings.reconstruction, "1.1f"),
        ("tot time(s)", result.timings.total, "1.1f"),
        ("solver", args.solver, "s"),
        ("source", "projected" if args.project_source else "exact", "s"),
        ("beta", "projected" if args.project_beta else "exact", "s"),
        ("reaction", "projected" if args.project_reaction else "exact", "s"),
        ("assembly", result.assembly_backend, "s"),
        ("boundary mode", result.boundary_mode, "s"),
        ("trace ordering", result.trace_ordering, "s"),
    ]
    if result.ordering_result is not None:
        diagnostics = result.ordering_result.diagnostics
        items.extend(
            [
                ("ordering time(s)", result.timings.trace_ordering, ".3f"),
                ("SCC components", diagnostics.num_components, ",d"),
                ("largest SCC", diagnostics.largest_component_size, ",d"),
                ("cyclic trace edges", diagnostics.cyclic_nodes, ",d"),
            ]
        )
    if global_solve is not None:
        free_trace_relative_residual = global_solve.diagnostic_relative_residual_norm
        if free_trace_relative_residual is None and result.boundary_mode == "eliminate":
            free_trace_relative_residual = global_solve.solver_relative_residual_norm
        items.extend(
            [
                ("iterations", -1 if global_solve.iteration_count is None else global_solve.iteration_count, ",d"),
                ("solver rel res", np.nan if global_solve.solver_relative_residual_norm is None else global_solve.solver_relative_residual_norm, ".3e"),
                ("free trace rel res", np.nan if free_trace_relative_residual is None else free_trace_relative_residual, ".3e"),
                ("ILU time(s)", 0.0 if global_solve.preconditioner_elapsed_seconds is None else global_solve.preconditioner_elapsed_seconds, ".3f"),
                ("perm time(s)", 0.0 if global_solve.permutation_elapsed_seconds is None else global_solve.permutation_elapsed_seconds, ".3f"),
                ("ILU permc", "-" if global_solve.ilu_permc_spec is None else global_solve.ilu_permc_spec, "s"),
                ("Krylov time(s)", 0.0 if global_solve.solve_elapsed_seconds is None else global_solve.solve_elapsed_seconds, ".3f"),
            ]
        )
    pretty_print_ncol(items, ncols=5, title="Solve Summary")

    if args.plot:
        title = f"test2, p={space.order}, elements={mesh.num_tri}, L2={l2_error:.2e}"
        plot_solution_comparison(
            result.field,
            exact,
            resolution=args.plot_resolution,
            exact_resolution="auto" if args.exact_plot_resolution is None else args.exact_plot_resolution,
            title=title,
            show_mesh=not args.hide_mesh,
        )


if __name__ == "__main__":
    _main()
