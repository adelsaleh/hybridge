r"""Statically condensed continuous Galerkin Poisson solver (CG against HDG, article study E9).

:class:`CGPoissonSolver` solves :math:`-\Delta\phi = \rho_h` with continuous Lagrange elements of
degree ``k`` (default ``p + 1`` for a DG(p) source, so that :math:`q = -\nabla\phi` lies exactly in
DG(p)^2, the transport velocity space) and Dirichlet data ``g``. The operator, its static
condensation and the factorization or AMG setup are built once (:meth:`CGPoissonSolver.setup`);
each :meth:`CGPoissonSolver.solve` costs the load, the condensed right-hand side, one global
solve, the bubble back-substitution and the flux.

Solvers: ``"pypardiso"`` (host, PARDISO SPD factorized once, then only the solve phase, with a
verified thread count), ``"direct"`` (host SciPy factorization, for small checks) and ``"amgx"``
(device, persistent PyAMGX CSR solver set up once).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from hybridge.conforming.poisson import CondensedCGPoissonOperator
from hybridge.core.conforming import ContinuousField
from hybridge.core.space import DGField, DGSpace, VectorDGField

__all__ = ["CGPoissonOptions", "CGPoissonResult", "CGPoissonSolver", "DEFAULT_CG_AMGX_CONFIG"]

# The guiding-center presets' CSR Poisson configuration (PCG-flexible, classical AMG, L1 Chebyshev).
DEFAULT_CG_AMGX_CONFIG = (Path(__file__).resolve().parents[2] / "configs" / "amgx"
                          / "diff_rea_gpu4_hdg_pcgf_cheb_l1_aggressive_abs.json")


@dataclass(frozen=True)
class CGPoissonOptions:
    """Options of :class:`CGPoissonSolver`."""

    degree: int | None = None
    solver: Literal["pypardiso", "direct", "amgx"] = "pypardiso"
    amgx_config: dict | str | Path | None = None
    rtol: float = 1.0e-11
    maxiter: int | None = None
    pardiso_threads: int = 16
    verbose: int = 0


@dataclass(frozen=True)
class CGPoissonResult:
    """Continuous potential, its flux ``q = -grad phi`` in DG(k-1)^2, solver data and stage timings."""

    potential: ContinuousField
    flux: VectorDGField
    iterations: int | None
    timings: dict[str, float] = field(default_factory=dict)


class CGPoissonSolver:
    """Reusable condensed CG Poisson solve on the mesh of ``space`` (see the module docstring)."""

    def __init__(self, space: DGSpace, *, boundary_condition: Callable | None = None,
                 options: CGPoissonOptions | None = None, flux_space: DGSpace | None = None) -> None:
        self.space = space
        self.options = options or CGPoissonOptions()
        self.degree = int(self.options.degree or space.order + 1)
        if self.options.solver not in {"pypardiso", "direct", "amgx"}:
            raise ValueError("solver must be 'pypardiso', 'direct' or 'amgx'")
        if flux_space is None:
            flux_space = space if space.order >= self.degree - 1 else DGSpace(space.mesh, self.degree - 1)
        if flux_space.order < self.degree - 1:
            raise ValueError("flux_space must have degree at least k - 1")
        self.flux_space = flux_space
        self.boundary_condition = boundary_condition
        self.device = self.options.solver == "amgx"
        self.operator: CondensedCGPoissonOperator | None = None
        self._factor = None
        self.setup_seconds = 0.0

    @property
    def xp(self):
        if not self.device:
            return np
        from hybridge.runtime.optional import require_cupy

        return require_cupy()

    def setup(self) -> float:
        """Build the operator and the factorization or AMG setup once; return the wall time."""
        if self.operator is not None:
            return self.setup_seconds
        start = time.perf_counter()
        self.operator = CondensedCGPoissonOperator(self.space, self.degree, device=self.device)
        matrix = self.operator.matrix_free
        solver = self.options.solver
        if solver == "pypardiso":
            from hybridge.linalg.direct import _import_pypardiso, prepare_pypardiso_spd_matrix
            from hybridge.linalg.pardiso_runtime import pardiso_thread_limit

            upper = prepare_pypardiso_spd_matrix(matrix)
            pardiso = _import_pypardiso().PyPardisoSolver(mtype=2)
            with pardiso_thread_limit(self.options.pardiso_threads):
                pardiso.factorize(upper)
            self._factor = (pardiso, upper)
        elif solver == "direct":
            import scipy.sparse.linalg

            self._factor = scipy.sparse.linalg.factorized(matrix.tocsc())
        else:
            from hybridge.linalg.amgx.device_solver import PyAMGXCsrDeviceSolver

            config = self.options.amgx_config or DEFAULT_CG_AMGX_CONFIG
            if isinstance(config, (str, Path)):
                import json

                with open(config) as stream:
                    config = json.load(stream)
            amgx = PyAMGXCsrDeviceSolver(config=config, tolerance=self.options.rtol, maxiter=self.options.maxiter,
                                         verbose=self.options.verbose, reusable=True)
            amgx.setup(self.operator._tables(self.xp)["matrix_free"])
            self._factor = amgx
            self.xp.cuda.Device().synchronize()
        self.setup_seconds = time.perf_counter() - start
        return self.setup_seconds

    def _sync(self) -> None:
        if self.device:
            self.xp.cuda.Device().synchronize()

    def solve(self, source: DGField, *, boundary_condition: Any = ..., initial_guess=None) -> CGPoissonResult:
        """Solve for one source (and optionally new Dirichlet data)."""
        self.setup()
        xp, operator = self.xp, self.operator
        data = self.boundary_condition if boundary_condition is Ellipsis else boundary_condition
        timings: dict[str, float] = {}
        start = time.perf_counter()
        boundary = operator.boundary_values(data, xp)
        system, loads = operator.condensed_system(source, boundary, xp=xp)
        self._sync()
        timings["rhs"] = time.perf_counter() - start
        start = time.perf_counter()
        iterations = None
        if self.options.solver == "pypardiso":
            from hybridge.linalg.pardiso_runtime import pardiso_thread_limit

            pardiso, upper = self._factor
            with pardiso_thread_limit(self.options.pardiso_threads):
                pardiso.set_phase(33)
                free = pardiso._call_pardiso(upper, np.ascontiguousarray(system.rhs))
        elif self.options.solver == "direct":
            free = self._factor(np.asarray(system.rhs))
        else:
            free, info = self._factor.solve(system.rhs, initial_guess=initial_guess)
            iterations = info.get("amgx_iterations")
        self._sync()
        timings["solve"] = time.perf_counter() - start
        start = time.perf_counter()
        potential = operator.reconstruct(free, boundary, loads, xp=xp)
        flux = potential.flux_field(self.flux_space)
        self._sync()
        timings["reconstruction"] = time.perf_counter() - start
        timings["total"] = timings["rhs"] + timings["solve"] + timings["reconstruction"]
        return CGPoissonResult(potential, flux, iterations, timings)

    def close(self) -> None:
        """Release the factorization or the AMGX solver."""
        if self._factor is None:
            return
        if self.options.solver == "amgx":
            self._factor.close(suppress_errors=True)
        elif self.options.solver == "pypardiso":
            self._factor[0].free_memory(everything=True)
        self._factor = None
