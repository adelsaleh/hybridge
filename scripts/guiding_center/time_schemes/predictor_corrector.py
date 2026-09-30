"""Midpoint/Crank-Nicolson predictor-corrector for guiding-center transport."""
from time import perf_counter

from hdgfem.core.field_ops import perpendicular_vector_field, trace_linear_combination
from hdgfem.diagnostics.solver import solver_result_metrics
from .si_euler import SIEulerStepper
from .stage_support import GuidingCenterStep


def _average_boundary_data(left, right):
    """Average two time-level boundary callables for the midpoint solve."""
    if left is None or right is None:
        return None
    return lambda x, y: 0.5 * (left(x, y) + right(x, y))



def _build_beta_from_flux_pair(left_flux, right_flux, dt: float, space):
    """Build ``(dt/2) * v_mid = (dt/4) * (q_left + q_right)^perp``."""
    midpoint_flux = 0.5 * (left_flux + right_flux)
    midpoint_flux.name = "q_mid_h"
    return perpendicular_vector_field(midpoint_flux, 0.5 * float(dt), space, name="beta_h")



class PredictorCorrectorStepper(SIEulerStepper):
    """Predict the endpoint drift, solve at its mean, and extrapolate density."""

    scheme = "predictor-corrector"

    def _advance(self, poisson_solver, solve_transport, guess, endpoint_postprocess):
        """Preserve midpoint boundary averaging and accepted trace extrapolation."""
        endpoint = self.time + self.dt
        start = perf_counter()
        source, beta, scale, stage, _ = self._transport_data()
        beta_time = perf_counter() - start
        predictor, predicted_density, predicted_trace = self._transport(
            solve_transport, source, beta, self.density_trace, scale, endpoint, stage)
        predicted_poisson, predicted_potential_trace = self._poisson(
            poisson_solver, predicted_density, guess, endpoint_postprocess, final=False)

        start = perf_counter()
        midpoint_beta = _build_beta_from_flux_pair(
            self.poisson_result.flux, predicted_poisson.flux, self.dt, self.space)
        beta_time += perf_counter() - start
        midpoint_boundary = _average_boundary_data(
            self.density_boundary(self.time), self.density_boundary(endpoint))
        midpoint_guess = trace_linear_combination([(0.5, self.density_trace), (0.5, predicted_trace)])
        corrector, midpoint_density, midpoint_trace = self._transport(
            solve_transport, self.density, midpoint_beta, midpoint_guess, 0.5*self.dt,
            self.time + 0.5*self.dt, "corrector", boundary_condition=midpoint_boundary)
        start = perf_counter()
        density = 2.0 * midpoint_density - self.density
        density.name = "rho_h"
        trace = trace_linear_combination([(2.0, midpoint_trace), (-1.0, self.density_trace)])
        self._transport_wall += perf_counter() - start
        poisson, potential_trace = self._poisson(
            poisson_solver, density, predicted_potential_trace, endpoint_postprocess, final=True)
        metrics = solver_result_metrics("predictor_transport", predictor)
        metrics.update(solver_result_metrics("predictor_poisson", predicted_poisson))
        metrics.update(solver_result_metrics("corrector_transport", corrector))
        metrics.update(solver_result_metrics("final_poisson", poisson))
        return GuidingCenterStep(
            density, trace, poisson, potential_trace, corrector, self.density, midpoint_beta,
            midpoint_guess, predicted_potential_trace, self._transport_results, self._poisson_results,
            self._transport_wall, self._poisson_wall, beta_time, metrics,
            transport_boundary=midpoint_boundary,
        )
