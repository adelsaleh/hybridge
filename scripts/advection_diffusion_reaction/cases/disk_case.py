"""Analytic stationary conservative ADR data on the unit disk (NumPy only)."""
from __future__ import annotations

from typing import Callable
import numpy as np

DEFAULT_PECLET = 10.0


def manufactured_adr_disk(peclet: float = DEFAULT_PECLET) -> dict[str, Callable]:
    r"""Create a steady conservative ADR problem on the unit disk.

    The PDE and mixed flux convention are

    .. math::

       \nabla\cdot(\boldsymbol\beta u+\boldsymbol q)+r u=f,
       \qquad \boldsymbol q=-\kappa\nabla u,
       \qquad \kappa=1/\mathrm{Pe}.

    The velocity is spatially variable and not divergence-free.  The source
    therefore contains the complete conservative term
    ``beta_x*u_x + beta_y*u_y + div(beta)*u``.  The exact solution is imposed
    as Dirichlet data on the complete discrete disk boundary; no normal-flux
    boundary property is used.
    """
    if not np.isfinite(peclet) or peclet <= 0.0:
        raise ValueError("peclet must be finite and strictly positive")

    pi = np.pi
    kappa = 1.0 / float(peclet)

    def _advection_data(x, y):
        """Return beta_x, beta_y, and the analytical velocity divergence."""
        radius_squared = x**2 + y**2
        q = 1.0 - radius_squared
        geometry_factor = 1.0 + 0.25 * x - 0.2 * y
        speed_factor = 1.0 + 0.2 * x + y / 6.0
        psi_x = -2.0 * x * geometry_factor + 0.25 * q
        psi_y = -2.0 * y * geometry_factor - 0.2 * q
        velocity_scale = 1.0 / 3.0
        velocity_x = velocity_scale * speed_factor * psi_y
        velocity_y = -velocity_scale * speed_factor * psi_x
        base_divergence = (
            geometry_factor * (x / 3.0 - 0.4 * y)
            - (49.0 / 600.0) * q
        )
        return velocity_x, velocity_y, velocity_scale * base_divergence

    def beta_x(x, y):
        """Return the x-component of the stationary velocity."""
        velocity_x, _, _ = _advection_data(x, y)
        return velocity_x

    def beta_y(x, y):
        """Return the y-component of the stationary velocity."""
        _, velocity_y, _ = _advection_data(x, y)
        return velocity_y

    def div_beta(x, y):
        """Return the analytical divergence of the stationary velocity."""
        _, _, velocity_divergence = _advection_data(x, y)
        return velocity_divergence

    def diffusivity(x, y):
        """Return the constant scalar diffusivity ``1/peclet``."""
        return kappa + 0.0 * (np.asarray(x) + np.asarray(y))

    def reaction(x, y):
        """Return the stationary positive reaction coefficient."""
        return 1.5 + 0.2 * x**2 + 0.25 * y**2 + 0.1 * x * y

    def _exact_data(x, y):
        """Return ``u``, its gradient, and its Laplacian."""
        radius_squared = x**2 + y**2
        q = 1.0 - radius_squared
        boundary_factor = q**2
        boundary_factor_x = -4.0 * x * q
        boundary_factor_y = -4.0 * y * q
        boundary_factor_laplacian = -8.0 + 16.0 * radius_squared

        sin_pi_x = np.sin(pi * x)
        cos_pi_x = np.cos(pi * x)
        sin_2pi_y = np.sin(2.0 * pi * y)
        cos_2pi_y = np.cos(2.0 * pi * y)
        shape = 1.0 + 0.3 * sin_pi_x * cos_2pi_y + 0.2 * x * y
        shape_x = 0.3 * pi * cos_pi_x * cos_2pi_y + 0.2 * y
        shape_y = -0.6 * pi * sin_pi_x * sin_2pi_y + 0.2 * x
        shape_laplacian = -1.5 * pi**2 * sin_pi_x * cos_2pi_y

        solution = 2.0 + boundary_factor * shape
        solution_x = boundary_factor_x * shape + boundary_factor * shape_x
        solution_y = boundary_factor_y * shape + boundary_factor * shape_y
        solution_laplacian = (
            boundary_factor_laplacian * shape
            + 2.0 * (
                boundary_factor_x * shape_x
                + boundary_factor_y * shape_y
            )
            + boundary_factor * shape_laplacian
        )
        return solution, solution_x, solution_y, solution_laplacian

    def exact(x, y):
        """Return the exact steady solution."""
        solution, _, _, _ = _exact_data(x, y)
        return solution

    def exact_gradient(x, y):
        """Return the exact spatial gradient."""
        _, solution_x, solution_y, _ = _exact_data(x, y)
        return solution_x, solution_y

    def exact_laplacian(x, y):
        """Return the exact spatial Laplacian."""
        _, _, _, solution_laplacian = _exact_data(x, y)
        return solution_laplacian

    def exact_diffusive_flux(x, y):
        """Return ``q=-kappa*grad(u)``."""
        solution_x, solution_y = exact_gradient(x, y)
        return -kappa * solution_x, -kappa * solution_y

    def exact_total_flux(x, y):
        """Return the conservative total flux ``beta*u-kappa*grad(u)``."""
        solution, solution_x, solution_y, _ = _exact_data(x, y)
        velocity_x, velocity_y, _ = _advection_data(x, y)
        return (
            velocity_x * solution - kappa * solution_x,
            velocity_y * solution - kappa * solution_y,
        )

    def source(x, y):
        r"""Return ``div(beta*u)-kappa*laplacian(u)+reaction*u``."""
        solution, solution_x, solution_y, solution_laplacian = _exact_data(x, y)
        velocity_x, velocity_y, velocity_divergence = _advection_data(x, y)
        conservative_advection = (
            velocity_x * solution_x
            + velocity_y * solution_y
            + velocity_divergence * solution
        )
        return (
            conservative_advection
            - kappa * solution_laplacian
            + reaction(x, y) * solution
        )

    return {
        "beta_x": beta_x,
        "beta_y": beta_y,
        "div_beta": div_beta,
        "diffusivity": diffusivity,
        "reaction": reaction,
        "source": source,
        "exact": exact,
        "exact_gradient": exact_gradient,
        "exact_laplacian": exact_laplacian,
        "exact_diffusive_flux": exact_diffusive_flux,
        "exact_total_flux": exact_total_flux,
    }
