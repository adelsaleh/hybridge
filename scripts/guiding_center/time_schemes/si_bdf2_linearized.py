"""SI-BDF2 of the guiding-center equations linearized about an equilibrium.

See :mod:`scripts.guiding_center.runtime.linearized`. The accepted state is the perturbation
``q`` and its Poisson result ``psi``; every transport solve reads

    q^{n+1} + s A(u*) q^{n+1} = h^n + s (2 S^n - S^{n-1}),

with the frozen equilibrium drift ``u*``, the BDF2 history ``h^n = (4 q^n - q^{n-1})/3`` and
``s = 2 dt/3`` (SI-Euler first step: ``h^0 = q^0``, ``s = dt``, source ``S^0``), and
``S^n = Pi_h(-u[psi^n] . grad rho*)``. Only the drift and the explicit coupling differ from
:class:`SIBDF2Stepper`; the Poisson solve, its compatible flux and the accepted-history handling
are shared.

``startup="richardson"`` replaces the first-order SI-Euler first step by its Richardson extrapolation
``2 E_{dt/2} E_{dt/2} q^0 - E_dt q^0`` (the half steps re-evaluate the coupling from the Poisson
solution at ``dt/2``), a second-order start that separates the startup error from the BDF2 error.
"""
from time import perf_counter

from hybridge.core.field_ops import field_linear_combination, trace_linear_combination
from hybridge.core.time_integration import bdf2_transport_data

from .si_bdf2 import SIBDF2Stepper
from .stage_support import GuidingCenterStep, transport_drift


class LinearizedSIBDF2Stepper(SIBDF2Stepper):
    """Frozen equilibrium drift and explicit, BDF2-extrapolated equilibrium coupling."""

    scheme = "si-bdf2-linearized"

    def __init__(self, *args, equilibrium_result, coupling, startup="euler", **kwargs):
        """``equilibrium_result``: Poisson result of the equilibrium density (its drift is frozen);
        ``coupling``: :class:`~scripts.guiding_center.runtime.linearized.EquilibriumCoupling`;
        ``startup``: ``euler`` (SI-Euler first step, as the nonlinear SI-BDF2) or ``richardson``."""
        if startup not in ("euler", "richardson"):
            raise ValueError("startup must be 'euler' or 'richardson'")
        super().__init__(*args, **kwargs)
        self.startup = startup
        self.equilibrium_flux = self._drift_flux(equilibrium_result)
        self.equilibrium_drift = transport_drift(self.equilibrium_flux, 1.0, self.space, name="beta_eq_h")
        self.coupling = coupling
        self.coupling_source = coupling.source(self._drift_flux(self.poisson_result))
        self.previous_coupling_source = None

    def _transport_data(self):
        startup = self.previous_density is None
        source, beta, scale = bdf2_transport_data(
            self.density, self.equilibrium_drift, self.dt,
            previous_field=self.previous_density,
            previous_velocity=None if startup else self.equilibrium_drift,
        )
        if startup:  # keep the accepted density itself as the Euler history, as SIBDF2Stepper does
            source = self.density
        coupling = self.coupling.extrapolated(self.coupling_source, self.previous_coupling_source)
        source = field_linear_combination(self.space, [(1.0, source), (float(scale), coupling)],
                                          name="q_linear_source_h")
        beta.name = "beta_eq_h"
        return source, beta, scale, "linear-bdf2-startup" if startup else "linear-bdf2", dict(
            bdf2_startup=startup, transport_time_order=1 if startup else 2,
            transport_beta_scale=scale, linearized=True,
        )

    def _euler(self, solve_transport, density, coupling_source, step, trace, stage_time, stage):
        """One linearized SI-Euler stage of length ``step``: (I + step A) q = density + step S."""
        source = field_linear_combination(self.space, [(1.0, density), (float(step), coupling_source)],
                                          name="q_linear_source_h")
        beta = transport_drift(self.equilibrium_flux, float(step), self.space, name="beta_eq_h")
        result, solved, solved_trace = self._transport(solve_transport, source, beta, trace, float(step),
                                                       stage_time, stage)
        return result, solved, solved_trace, source, beta

    def _advance(self, poisson_solver, solve_transport, guess, endpoint_postprocess):
        if self.startup != "richardson" or self.previous_density is not None:
            return super()._advance(poisson_solver, solve_transport, guess, endpoint_postprocess)
        start, dt = perf_counter(), self.dt
        _, full, full_trace, _, _ = self._euler(solve_transport, self.density, self.coupling_source, dt,
                                                self.density_trace, self.time + dt, "linear-richardson-full")
        _, half, half_trace, _, _ = self._euler(solve_transport, self.density, self.coupling_source, .5*dt,
                                                self.density_trace, self.time + .5*dt, "linear-richardson-half")
        poisson_half, _ = self._poisson(poisson_solver, half, guess, endpoint_postprocess, final=False)
        half_source = self.coupling.source(self._drift_flux(poisson_half))
        result, second, second_trace, source, beta = self._euler(
            solve_transport, half, half_source, .5*dt, half_trace, self.time + dt, "linear-richardson-second")
        density = field_linear_combination(self.space, [(2.0, second), (-1.0, full)], name="q_h")
        trace = trace_linear_combination([(2.0, second_trace), (-1.0, full_trace)])
        poisson, potential_trace = self._poisson(poisson_solver, density, guess, endpoint_postprocess, final=True)
        metrics = dict(bdf2_startup=True, transport_time_order=2, transport_beta_scale=.5*dt, linearized=True,
                       linear_startup="richardson")
        return GuidingCenterStep(
            density, trace, poisson, potential_trace, result, source, beta, self.density_trace, guess,
            self._transport_results, self._poisson_results, self._transport_wall, self._poisson_wall,
            perf_counter() - start, metrics, transport_boundary=self.density_boundary(self.time + dt))

    def _commit(self, result):
        current = self.coupling_source
        super()._commit(result)
        self.previous_coupling_source = current
        self.coupling_source = self.coupling.source(self._drift_flux(self.poisson_result))

    def _rebuild_poisson_history(self, work):
        """After a tau change the fluxes, hence both coupling sources, are recomputed."""
        super()._rebuild_poisson_history(work)
        self.coupling_source = self.coupling.source(self._drift_flux(self.poisson_result))
        if self.previous_flux is not None:
            self.previous_coupling_source = self.coupling.source(self.previous_flux)
