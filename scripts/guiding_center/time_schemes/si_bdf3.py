"""Constant-step semi-implicit BDF3 with a third-order-compatible startup.

Regular steps solve one linear transport equation and one endpoint Poisson:

    (I + 6*dt/11 A(3*v_n - 3*v_previous + v_older)) rho_next
        = (18*rho_n - 9*rho_previous + 2*rho_older)/11.

The first step is Richardson-extrapolated SI Euler: ``2*E(dt/2) - E(dt)``,
one full step and two half steps, so its local error is O(dt^3). The second
step is SI-BDF2, also O(dt^3) locally. Neither startup step then limits the
global third order, unlike a plain Euler/BDF2 ramp. The startup costs three
transport and two Poisson solves, once. The extrapolation is not positivity
preserving.
"""
from time import perf_counter

from hdgfem.core.field_ops import perpendicular_vector_field, trace_linear_combination
from hdgfem.core.time_integration import bdf3_transport_data
from .si_bdf2 import SIBDF2Stepper
from .stage_support import GuidingCenterStep

STARTUP_METHOD = "si-euler-extrap2"


def _bdf3_transport_data(space, density, flux, dt, *, previous_density=None, previous_flux=None,
                         older_density=None, older_flux=None):
    """Return source and beta for constant-step BDF3, ramping through BDF2.

    Each history level must pair the density and flux of one accepted endpoint.
    All combinations preserve device residency and leave their inputs intact.
    """
    if (previous_density is None) != (previous_flux is None):
        raise ValueError("BDF3 requires both previous density and previous flux, or neither")
    if (older_density is None) != (older_flux is None):
        raise ValueError("BDF3 requires both older density and older flux, or neither")
    perpendicular = lambda value: None if value is None else perpendicular_vector_field(value, 1.0, space)
    source, beta, scale = bdf3_transport_data(
        density, perpendicular(flux), dt,
        previous_field=previous_density, previous_velocity=perpendicular(previous_flux),
        older_field=older_density, older_velocity=perpendicular(older_flux),
    )
    source.name = "rho_bdf3_source_h" if older_density is not None else "rho_bdf2_source_h"
    beta.name = "beta_h"
    for component, label in zip(beta.components, ("beta_h_x", "beta_h_y")):
        component.name = label
    return source, beta, scale


class SIBDF3Stepper(SIBDF2Stepper):
    """Use three accepted densities and extrapolated accepted Poisson fluxes."""

    scheme = "si-bdf3"

    def __init__(self, *args, **kwargs):
        """Initialize the shared state with no older BDF history."""
        super().__init__(*args, **kwargs)
        self.older_density = self.older_flux = None

    def _transport_data(self):
        """Build BDF3 once two histories exist; the second step uses BDF2."""
        if self.previous_density is None:
            raise RuntimeError("the first SI-BDF3 step uses the extrapolated SI-Euler startup")
        startup = self.older_density is None
        source, beta, scale = _bdf3_transport_data(
            self.space, self.density, self._drift_flux(self.poisson_result), self.dt,
            previous_density=self.previous_density, previous_flux=self.previous_flux,
            older_density=self.older_density, older_flux=self.older_flux,
        )
        return source, beta, scale, "bdf3-startup-bdf2" if startup else "bdf3", dict(
            bdf3_startup=startup, bdf3_startup_method="si-bdf2" if startup else None,
            transport_time_order=2 if startup else 3, transport_beta_scale=scale,
        )

    def _stage_poisson(self, density, stage_time, label, guess):
        """Solve a non-final startup Poisson stage and record its cost."""
        start = perf_counter()
        result, trace = self._recovery_work.poisson(density, stage_time, label, guess=guess)
        self._poisson_wall += perf_counter() - start
        self._poisson_results.append(result)
        return result, trace

    def _advance(self, poisson_solver, solve_transport, guess, endpoint_postprocess):
        """Run the Richardson SI-Euler startup, otherwise one BDF2/BDF3 solve."""
        if self.previous_density is not None:
            return super()._advance(poisson_solver, solve_transport, guess, endpoint_postprocess)
        space, dt = self.space, self.dt
        half, endpoint = self.time + 0.5*dt, self.time + dt
        start = perf_counter()
        flux = self._drift_flux(self.poisson_result)
        full_beta = perpendicular_vector_field(flux, dt, space)
        half_beta = perpendicular_vector_field(flux, 0.5*dt, space)
        beta_time = perf_counter() - start
        # Both paths start from the accepted state, so a failure in either
        # first stage belongs to the accepted Poisson checkpoint.
        _, full_density, full_trace = self._transport(
            solve_transport, self.density, full_beta, self.density_trace, dt, endpoint,
            "extrap2-full")
        _, half_density, half_trace = self._transport(
            solve_transport, self.density, half_beta, self.density_trace, 0.5*dt, half,
            "extrap2-half-1")
        half_poisson, half_potential_trace = self._stage_poisson(
            half_density, half, "SI-Euler extrap2 half step", self.potential_trace)
        start = perf_counter()
        second_beta = perpendicular_vector_field(self._drift_flux(half_poisson), 0.5*dt, space)
        beta_time += perf_counter() - start
        result, path_density, path_trace = self._transport(
            solve_transport, half_density, second_beta, half_trace, 0.5*dt, endpoint,
            "extrap2-half-2")
        start = perf_counter()
        density = 2.0 * path_density - full_density
        density.name = "rho_h"
        trace = trace_linear_combination([(2.0, path_trace), (-1.0, full_trace)])
        self._transport_wall += perf_counter() - start
        endpoint_guess = trace_linear_combination([(2.0, half_potential_trace), (-1.0, self.potential_trace)])
        poisson, potential_trace = self._poisson(
            poisson_solver, density, endpoint_guess, endpoint_postprocess, final=True)
        return GuidingCenterStep(
            density, trace, poisson, potential_trace, result, half_density, second_beta,
            half_trace, endpoint_guess, self._transport_results, self._poisson_results,
            self._transport_wall, self._poisson_wall, beta_time,
            dict(bdf3_startup=True, bdf3_startup_method=STARTUP_METHOD,
                 transport_time_order=2, transport_beta_scale=0.5*dt),
            transport_boundary=self.density_boundary(endpoint),
        )

    def _commit(self, result):
        """Shift both accepted history levels only after complete success."""
        self._drift_flux(result.poisson_result)
        older = self.previous_density, self.previous_flux
        super()._commit(result)
        self.older_density, self.older_flux = older

    def _rebuild_poisson_history(self, work):
        """Recompute the older cached flux too, then the BDF2 levels."""
        if self.older_density is not None:
            result, _ = work.poisson(self.older_density, self.time - 2*self.dt, "older accepted state",
                                     guess=self.potential_trace)
            self.older_flux = self._drift_flux(result).copy(name="q_older_h")
        super()._rebuild_poisson_history(work)
