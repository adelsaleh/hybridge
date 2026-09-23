"""Semi-implicit Euler and shared accepted-state handling for low-order schemes."""
from __future__ import annotations

import math
from time import perf_counter

from hdgfem.core.field_ops import (
    perpendicular_vector_field, solution_field, solution_trace,
)
from .recovery import with_poisson_recovery, recovery_options, poisson_tau
from .stage_support import GuidingCenterStep, _fixed_operator_trace_predictor


class SIEulerStepper:
    """Freeze the accepted drift, then solve transport and endpoint Poisson.

    ``solve_transport`` accepts source, beta, trace guess and drift scale, plus
    stage metadata. Only successful complete steps enter the accepted history.
    Field and trace operations use the package's host/device-aware helpers.
    """

    scheme = "si-euler"

    def __init__(self, space, dt, density, poisson_result, density_trace, *,
                 density_boundary, potential_boundary, potential_trace=None,
                 phase_verbosity=0, detail_verbosity=0, poisson_solver=None,
                 poisson_tau_retry_factor=2., poisson_tau_max_retries=4,
                 recovery_verbosity=0, recovery_record=None):
        """Initialize the accepted state and its fixed-operator trace history."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError(f"{self.scheme} requires a finite positive constant dt")
        self.space, self.dt, self.time = space, float(dt), 0.0
        self.density = density.copy(name="rho_h")
        self.density_trace = density_trace.copy()
        self.recovery_options = recovery_options(poisson_tau_retry_factor, poisson_tau_max_retries,
                                                 recovery_verbosity, recovery_record)
        self.accepted_poisson_tau = poisson_tau(poisson_solver)
        self.poisson_result = poisson_result
        self.potential_trace = (solution_trace(poisson_result, space, reduced=False)
                                if potential_trace is None else potential_trace).copy()
        self.previous_potential_trace = self.older_potential_trace = None
        self.density_boundary, self.potential_boundary = density_boundary, potential_boundary
        self.phase_verbosity, self.detail_verbosity = phase_verbosity, detail_verbosity

    def _drift_flux(self, result):
        """Select the electric field used by this scheme's transport history."""
        return result.flux

    def _transport_data(self):
        """Return the frozen-drift Euler equation and its stage metadata."""
        return (self.density, perpendicular_vector_field(self.poisson_result.flux, self.dt, self.space),
                self.dt, "predictor", {})

    def _transport(self, solve_transport, source, beta, guess, scale, stage_time, stage,
                   *, boundary_condition=...):
        """Solve and reconstruct one transport stage, recording elapsed work."""
        start = perf_counter()
        keywords = dict(stage_time=stage_time, stage=stage)
        if boundary_condition is not Ellipsis:
            keywords["boundary_condition"] = boundary_condition
        result = solve_transport(source, beta, guess, scale, **keywords)
        density = solution_field(result, self.space).copy(
            name="rho_" + stage + "_h")
        trace = solution_trace(result, self.space, reduced=True).copy()
        self._transport_wall += perf_counter() - start
        self._transport_results.append(result)
        return result, density, trace

    def _poisson(self, solver, density, guess, endpoint_postprocess, *, final):
        """Solve an endpoint RHS, applying optional postprocessing only if final."""
        start = perf_counter()
        result, trace = self._recovery_work.poisson(
            density, self.time + self.dt, "accepted endpoint" if final else "predictor",
            guess=guess, final=final)
        if final:
            self._post_poisson_start = self._recovery_work.last_post_poisson_start
            self._potential_trace_time = self._recovery_work.last_trace_wall
        self._poisson_wall += perf_counter() - start
        self._poisson_results.append(result)
        return result, trace

    def _advance(self, poisson_solver, solve_transport, guess, endpoint_postprocess):
        """Evaluate the Euler transport equation followed by endpoint Poisson."""
        start = perf_counter()
        source, beta, scale, stage, metrics = self._transport_data()
        beta_time = perf_counter() - start
        result, density, trace = self._transport(
            solve_transport, source, beta, self.density_trace, scale, self.time + self.dt, stage)
        poisson, potential_trace = self._poisson(
            poisson_solver, density, guess, endpoint_postprocess, final=True)
        return GuidingCenterStep(
            density, trace, poisson, potential_trace, result, source, beta,
            self.density_trace, guess, self._transport_results, self._poisson_results,
            self._transport_wall, self._poisson_wall, beta_time, metrics,
            transport_boundary=self.density_boundary(self.time + self.dt),
        )

    def _commit(self, result):
        """Commit accepted state and fixed-operator Poisson trace history."""
        self.older_potential_trace = self.previous_potential_trace
        self.previous_potential_trace = self.potential_trace
        self.density, self.density_trace = result.density, result.density_trace
        self.poisson_result, self.potential_trace = result.poisson_result, result.potential_trace
        self.time += self.dt

    def _rebuild_poisson_history(self, work):
        """Recompute cached fluxes at fixed accepted densities under the new tau."""
        self.previous_potential_trace = self.older_potential_trace = None
        previous = getattr(self, "previous_density", None)
        if previous is not None:
            result, _ = work.poisson(previous, self.time-self.dt, "previous accepted state",
                                     guess=self.potential_trace)
            self.previous_flux = self._drift_flux(result).copy(name="q_previous_h")
        self.poisson_result, self.potential_trace = work.poisson(
            self.density, self.time, "restart at accepted state", guess=self.potential_trace)

    @with_poisson_recovery
    def advance(self, poisson_solver, solve_transport, *, endpoint_postprocess=None):
        """Advance one step, committing history only after both solves succeed."""
        self._transport_results, self._poisson_results = [], []
        self._transport_wall = self._poisson_wall = self._potential_trace_time = 0.0
        self._post_poisson_start = None
        guess, order = _fixed_operator_trace_predictor(
            self.potential_trace, self.previous_potential_trace, self.older_potential_trace)
        result = self._advance(poisson_solver, solve_transport, guess, endpoint_postprocess)
        result.metrics.update(poisson_predictor_order=order)
        result.potential_trace_time = self._potential_trace_time
        result.post_poisson_start = self._post_poisson_start
        self._commit(result)
        return result
