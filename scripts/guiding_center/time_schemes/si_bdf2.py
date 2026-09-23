"""Constant-step semi-implicit BDF2 with an SI-Euler first step."""
from hdgfem.core.field_ops import perpendicular_vector_field
from .si_euler import SIEulerStepper


def _bdf2_transport_data(space, density, flux, dt, *, previous_density=None, previous_flux=None):
    """Return source and beta for constant-step BDF2, with SI-Euler startup.

    (I + 2*dt/3 A(2*v_n - v_previous)) rho_next
        = (4*rho_n - rho_previous)/3.

    Both history fields must come from the same previous accepted endpoint.
    All combinations preserve device residency and leave their inputs intact.
    """
    if (previous_density is None) != (previous_flux is None):
        raise ValueError("BDF2 requires both previous density and previous flux, or neither")
    if previous_density is None:
        return density, perpendicular_vector_field(flux, dt, space), float(dt)
    source = (4.0 * density - previous_density) / 3.0
    source.name = "rho_bdf2_source_h"
    extrapolated_flux = 2.0 * flux - previous_flux
    extrapolated_flux.name = "q_bdf2_extrapolated_h"
    beta_scale = 2.0 * float(dt) / 3.0
    return source, perpendicular_vector_field(extrapolated_flux, beta_scale, space), beta_scale



class SIBDF2Stepper(SIEulerStepper):
    """Use two accepted densities and extrapolated accepted Poisson fluxes."""

    scheme = "si-bdf2"

    def __init__(self, *args, use_postprocessed_flux=False, **kwargs):
        """Initialize the shared state with no older BDF history."""
        super().__init__(*args, **kwargs)
        self.use_postprocessed_flux = bool(use_postprocessed_flux)
        self._drift_flux(self.poisson_result)
        self.previous_density = self.previous_flux = None

    def _drift_flux(self, result):
        """Require recovery on every solve when it supplies the BDF2 drift."""
        if not self.use_postprocessed_flux:
            return result.flux
        flux = getattr(result, "postprocessed_flux", None)
        if flux is None:
            raise ValueError("BDF2 recovered drift requires a postprocessed flux on every Poisson solve")
        return flux

    def _transport_data(self):
        """Build the tested BDF2 equation, falling back to Euler for startup."""
        startup = self.previous_density is None
        source, beta, scale = _bdf2_transport_data(
            self.space, self.density, self._drift_flux(self.poisson_result), self.dt,
            previous_density=self.previous_density, previous_flux=self.previous_flux,
        )
        return source, beta, scale, "bdf2-startup" if startup else "bdf2", dict(
            bdf2_startup=startup, transport_time_order=1 if startup else 2,
            transport_beta_scale=scale,
        )

    def _commit(self, result):
        """Retain the previous accepted endpoint only after complete success."""
        # Validate the endpoint before changing any accepted history.
        self._drift_flux(result.poisson_result)
        self.previous_density, self.previous_flux = self.density, self._drift_flux(self.poisson_result)
        super()._commit(result)
