"""Constant-step semi-implicit BDF2 with an SI-Euler first step."""
from hybridge.core.field_ops import perpendicular_vector_field
from hybridge.core.time_integration import bdf2_transport_data
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
    source, beta, scale = bdf2_transport_data(
        density, perpendicular_vector_field(flux, 1.0, space), dt,
        previous_field=previous_density,
        previous_velocity=(None if previous_flux is None
                           else perpendicular_vector_field(previous_flux, 1.0, space)),
    )
    # Preserve the existing private wrapper's startup identity and labels.
    source = density if previous_density is None else source
    if previous_density is not None:
        source.name = "rho_bdf2_source_h"
    beta.name = "beta_h"
    for component, label in zip(beta.components, ("beta_h_x", "beta_h_y")):
        component.name = label
    return source, beta, scale



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
