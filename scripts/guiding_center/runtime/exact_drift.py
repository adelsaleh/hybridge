"""Exact guiding-center drift for reference runs: the exact flux instead of the HDG flux.

For manufactured cases with an analytic flux ``q = -grad(phi)`` (``GuidingCenterCase.exact_flux``),
:class:`ExactFluxPoissonSolver` hands the transport the exact drift ``u = (-q_y, q_x)`` at the solve
time, in one of two modes (``exact_drift_mode``):

* ``pointwise`` (default): ``result.exact_drift`` is a
  :class:`~hybridge.core.quadrature_coefficients.QuadratureVectorField` holding the exact drift at the
  transport's volume and face quadrature points, with a single-valued normal flux. The transport
  assembly and the explicit residual read these values directly, so the drift is exact wherever the
  scheme evaluates it (``time_schemes.stage_support.drift_flux``).
* ``projected``: the drift is the L2 projection of the exact flux onto DG(p + s) (``s`` =
  ``exact_drift_order_offset``, default 2), with error O(h^{p+s+1}). A flux of degree above p drives
  the DG(p) transport through the cross-space drift tables, without projection into DG(p)
  (``time_schemes.stage_support.transport_drift``).

In both modes ``result.flux`` is the projected exact flux, which the diagnostics read, and
``result.raw_flux`` keeps the HDG flux. The Poisson problem is still solved, so the potential, its
traces and the diagnostics are those of the coupled scheme; only the drift handed to the transport
changes, with the same scheme, time steps and stage structure as the coupled runs: the "exact drift"
reference of the convergence tables. The solve time is read from the boundary callable that the
steppers set before every solve (``GuidingCenterCase.potential_boundary_at`` tags it with
``_hybridge_time``); the first solve, before any boundary update, is at t = 0.
"""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter

from hybridge.core.projection import project_callable
from hybridge.core.quadrature_coefficients import QuadratureVectorField
from hybridge.core.space import DGSpace, VectorDGField

__all__ = ["EXACT_DRIFT_MODES", "ExactFluxPoissonSolver"]

EXACT_DRIFT_MODES = ("pointwise", "projected")


class ExactFluxPoissonSolver:
    """Proxy a reusable Poisson solver and attach the exact flux (and drift) to its results.

    ``mode`` is ``"pointwise"`` (``trace_space``, the transport's trace space, is then required: it
    fixes the face quadrature points of the drift) or ``"projected"``. Attribute reads and writes
    other than :meth:`solve` and :meth:`set_boundary_condition` are forwarded to the wrapped solver,
    as for ``CompatibleFluxPoissonSolver``.
    """

    def __init__(self, solver, case, density_space, *, degree_offset: int = 2, device: bool = True,
                 mode: str = "pointwise", trace_space=None) -> None:
        if getattr(case, "exact_flux", None) is None:
            raise ValueError(f"case {getattr(case, 'key', case)!r} has no exact flux; the exact drift needs one")
        if int(degree_offset) < 0:
            raise ValueError("exact_drift_order_offset must be nonnegative")
        if mode not in EXACT_DRIFT_MODES:
            raise ValueError(f"exact_drift_mode must be one of {', '.join(EXACT_DRIFT_MODES)}; got {mode!r}")
        if mode == "pointwise" and trace_space is None:
            raise ValueError("the pointwise exact drift needs the transport trace space")
        flux_space = DGSpace(density_space.mesh, density_space.order + int(degree_offset),
                             basis_type=density_space.reference.basis_type, name="exact_flux_h")
        object.__setattr__(self, "_solver", solver)
        object.__setattr__(self, "_case", case)
        object.__setattr__(self, "_flux_space", flux_space)
        object.__setattr__(self, "_density_space", density_space)
        object.__setattr__(self, "_trace_space", trace_space)
        object.__setattr__(self, "_mode", mode)
        object.__setattr__(self, "_backend", "device" if device else "host")
        object.__setattr__(self, "_time", 0.0)
        object.__setattr__(self, "last_exact_wall_time", 0.0)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_solver"), name)

    def __setattr__(self, name, value) -> None:
        if name in {"last_exact_wall_time"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._solver, name, value)

    @property
    def wrapped_solver(self):
        """The underlying reusable Poisson solver."""
        return self._solver

    @property
    def flux_space(self) -> DGSpace:
        """DG(p + s) space of the projected exact flux."""
        return self._flux_space

    @property
    def mode(self) -> str:
        """``"pointwise"`` or ``"projected"``: what the transport receives."""
        return self._mode

    @property
    def time(self) -> float:
        """Time of the next solve (from the last boundary update)."""
        return self._time

    def set_boundary_condition(self, boundary) -> None:
        """Forward the Dirichlet data and record its time (``_hybridge_time``) for the exact flux."""
        time = getattr(boundary, "_hybridge_time", None)
        if time is not None:
            object.__setattr__(self, "_time", float(time))
        self._solver.set_boundary_condition(boundary)

    def solve(self, *args, **kwargs):
        """Solve, then attach the exact flux (and, pointwise, the exact drift) at the solve time."""
        return self.attach(self._solver.solve(*args, **kwargs))

    def attach(self, result):
        """Return ``result`` with the exact data of ``q = -grad(phi)`` at :attr:`time`.

        ``flux`` becomes the projection of ``q`` onto DG(p + s) (``raw_flux`` keeps the HDG flux);
        in pointwise mode ``exact_drift`` is the exact drift ``(-q_y, q_x)`` at the quadrature points.
        """
        start = perf_counter()
        time, flux = self._time, self._case.exact_flux
        components = tuple(
            project_callable(lambda x, y, index=index: flux(x, y, time)[index], self._flux_space,
                             backend=self._backend, name=f"q_exact_{axis}")
            for index, axis in enumerate("xy"))
        exact = replace(result, flux=VectorDGField(components, name="q_exact"))
        object.__setattr__(exact, "raw_flux", result.flux)
        object.__setattr__(exact, "exact_flux_time", time)
        object.__setattr__(exact, "exact_drift_mode", self._mode)
        if self._mode == "pointwise":
            def drift(x, y):
                """Exact drift ``(-q_y, q_x)`` at the solve time."""
                qx, qy = flux(x, y, time)
                return -qy, qx

            object.__setattr__(exact, "exact_drift", QuadratureVectorField.from_callable(
                drift, self._density_space, trace_space=self._trace_space, device=self._backend == "device",
                name="u_exact"))
        object.__setattr__(self, "last_exact_wall_time", perf_counter() - start)
        return exact
