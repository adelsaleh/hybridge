"""Linearized guiding-center evolution about a computed equilibrium (preset ``linearized``).

For an equilibrium ``(rho*, phi*)`` with ``-Lap phi* = rho*`` and zero wall potential, a
perturbation ``q = rho - rho*`` with potential ``psi`` (``-Lap psi = q``, ``psi = 0`` on the walls)
obeys, to first order,

    d_t q + u* . grad q + u[psi] . grad rho* = 0,        u* = grad^perp phi*,  u[psi] = grad^perp psi.

The discretization is that of the nonlinear runs with two changes
(:class:`scripts.guiding_center.time_schemes.si_bdf2_linearized.LinearizedSIBDF2Stepper`): the drift
is frozen at the equilibrium's (compatible) drift, and the coupling ``S = -u[psi] . grad rho*`` enters
every transport solve as an explicit source, extrapolated with the BDF2 weights like the nonlinear
drift (``2 S^n - S^{n-1}``; ``S^0`` on the SI-Euler first step). ``u[psi]`` is the drift that the
nonlinear scheme would build from the Poisson result of ``q`` (same Poisson solver, same compatible
flux), and ``grad rho*`` is sampled once from the case: ``equilibrium_density_gradient`` (imported
bands: ``W'(phi*) grad phi*`` of the CG source) or the radial derivative of a disk profile.

The linear problem is homogeneous: a run with initial perturbation ``q_0`` gives the response per
unit of whatever amplitude ``q_0`` carries, and it has no positivity, so density diagnostics that
assume ``rho >= 0`` are switched off by the runner.
"""
from __future__ import annotations

import numpy as np

from hybridge.core.field_ops import field_linear_combination

__all__ = ["EquilibriumCoupling", "equilibrium_gradient", "linear_initial_density"]


def equilibrium_gradient(case):
    """``(x, y) -> (d_x rho*, d_y rho*)`` of the case, or ``None`` when the case provides none.

    ``case.equilibrium_density_gradient`` when set, else ``rho*'(r) (x, y)/r`` from
    ``case.equilibrium_radial_derivative`` (disk and annulus profiles centered at the origin).
    """
    gradient = getattr(case, "equilibrium_density_gradient", None)
    if gradient is not None:
        return gradient
    radial = getattr(case, "equilibrium_radial_derivative", None)
    if radial is None:
        return None

    def from_radial(x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        r = np.hypot(x, y)
        slope = np.asarray(radial(r), dtype=float)
        safe = np.where(r > 0.0, r, 1.0)
        factor = np.where(r > 0.0, slope/safe, 0.0)
        return factor*x, factor*y

    return from_radial


def linear_initial_density(case):
    """Initial perturbation ``q_0(x, y)`` of a linearized run.

    ``case.linear_perturbation`` (the infinitesimal seed) when set, else
    ``initial_density - equilibrium_density``, which is exactly linear for multiplicative seeds
    ``rho*(r) (1 + eps cos(m theta))``.
    """
    perturbation = getattr(case, "linear_perturbation", None)
    if perturbation is not None:
        return perturbation
    if case.equilibrium_density is None:
        raise ValueError("a linearized run needs the case's equilibrium density")
    initial, equilibrium = case.initial_density, case.equilibrium_density

    def difference(x, y):
        return initial(x, y) - equilibrium(x, y)

    return difference


class EquilibriumCoupling:
    """Projected coupling ``S_h = Pi_h(-u[psi] . grad rho*)`` of a linearized step.

    ``grad rho*`` is sampled once at the density space's volume quadrature points; ``u[psi]`` is the
    rotated Poisson flux ``(-q_y, q_x)`` of the perturbation (the transport drift of
    ``stage_support.transport_drift``), evaluated at the same points. The product is projected with
    the reference mass matrix, as :meth:`DGSpace.project_callable` does. Device-resident fluxes give a
    device-resident source (CuPy); host fluxes a host source.
    """

    def __init__(self, space, gradient, *, name: str = "S_lin_h"):
        if gradient is None:
            raise ValueError("a linearized run needs grad rho*: set the case's equilibrium_density_gradient "
                             "or equilibrium_radial_derivative")
        self.space, self.name = space, name
        points = space.mapped_quads()
        gx, gy = gradient(points[:, :, 0], points[:, :, 1])
        shape = points.shape[:2]
        gx = np.broadcast_to(np.asarray(gx, dtype=float), shape)
        gy = np.broadcast_to(np.asarray(gy, dtype=float), shape)
        if not (np.all(np.isfinite(gx)) and np.all(np.isfinite(gy))):
            raise ValueError("grad rho* is not finite at every quadrature point")
        self.gradient = np.ascontiguousarray(np.stack([gx, gy]))
        quad = space.quad_data
        self.projector = np.ascontiguousarray(quad.weighted_phi @ quad.MKrf_inv)
        self._tables: dict = {}
        self._device: dict = {}

    def _basis(self, component_space):
        """Values at this space's quadrature of the basis of the flux component space."""
        key = id(component_space)
        table = self._tables.get(key)
        if table is None:
            table = np.ascontiguousarray(component_space.basis_at(self.space.quad_data.Krf_quads))
            self._tables[key] = table
        return table

    def source(self, flux):
        """``Pi_h(-u . grad rho*)`` with ``u = (-q_y, q_x)`` for the Poisson flux ``q`` of the perturbation."""
        qx, qy = flux.components
        if qx.device_coefficients_materialized() or qy.device_coefficients_materialized():
            return self._device_source(qx, qy)
        basis_x, basis_y = self._basis(qx.space), self._basis(qy.space)
        values_x = np.asarray(qx.coeffs) @ basis_x.T
        values_y = np.asarray(qy.coeffs) @ basis_y.T
        # -u . g = q_y g_x - q_x g_y
        values = values_y*self.gradient[0] - values_x*self.gradient[1]
        return self.space.field(values @ self.projector, name=self.name, _coefficient_kind="projected")

    def _device_source(self, qx, qy):
        from hybridge.core.device import as_cupy_coefficients, as_cupy_space, field_from_cupy_coefficients
        from hybridge.runtime.optional import require_cupy

        cp = require_cupy()
        device_id = int(cp.cuda.runtime.getDevice())
        for component in (qx, qy):
            if not component.device_coefficients_materialized(device_id):
                for other in range(cp.cuda.runtime.getDeviceCount()):
                    if component.device_coefficients_materialized(other):
                        device_id = other
                        break
        with cp.cuda.Device(device_id):
            tables = self._device.get(device_id)
            if tables is None:
                tables = dict(gradient=cp.asarray(self.gradient), projector=cp.asarray(self.projector))
                self._device[device_id] = tables
            values = []
            for component in (qx, qy):
                key = ("basis", id(component.space))
                if key not in tables:
                    tables[key] = cp.asarray(self._basis(component.space))
                coefficients = as_cupy_coefficients(component, as_cupy_space(component.space, device=device_id))
                values.append(coefficients @ tables[key].T)
            source = values[1]*tables["gradient"][0] - values[0]*tables["gradient"][1]
            return field_from_cupy_coefficients(self.space, cp.ascontiguousarray(source @ tables["projector"]),
                                                device=device_id, name=self.name, _coefficient_kind="projected")

    def extrapolated(self, current, previous):
        """``2 S^n - S^{n-1}`` (``S^n`` alone without a previous value)."""
        if previous is None:
            return current
        return field_linear_combination(self.space, [(2.0, current), (-1.0, previous)], name=self.name)
