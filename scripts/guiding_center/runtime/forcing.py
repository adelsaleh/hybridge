"""Source and reaction terms of the guiding-center density equation.

The forced equation is d_t rho + v.grad rho = S(x, t) + alpha(x, t)*rho, with an
explicit source S independent of rho and a reaction coefficient
alpha = alpha_0 + r(t)*nu(x): a constant and an optional space-dependent part,
e.g. the electron-impact ionization rate of a prescribed neutral gas. Every
implicit transport solve of the supported schemes reads rho + scale*A(v) rho =
source; the forcing is treated implicitly at the stage time t,

    (1 - scale*alpha_h(t)) rho + scale*A(v) rho = source + scale*S_h(t),

so it changes the reaction coefficient and the right-hand side of the same HDG
solve, at no extra cost. IMEX-ARK3 also adds S_h(t) + Pi(alpha_h rho) to its
explicit stage residuals, which keeps the implicit/explicit splitting
consistent. S_h and nu_h are L2 projections, so int S_h = int S, and testing
with 1 gives the discrete mass balance d_t int rho = int S + int alpha_h rho
exactly (:meth:`DensityForcing.production` is the reaction part).
"""
from __future__ import annotations

# Each transport solve of these schemes has the implicit-stage form below (SI-BDF3:
# scale 6 dt/11, its Richardson SI-Euler start-up and SI-BDF2 second step included).
SUPPORTED_SCHEMES = ("si-euler", "si-bdf2", "si-bdf3", "imex-ark3")


class DensityForcing:
    """Projected source S_h(t) and reaction alpha_h(t) of one guiding-center case."""

    def __init__(self, space, case, *, cache_size: int = 8):
        """Read the case's forcing; projections are cached per stage time."""
        self.space = space
        self.source_function = case.density_source
        self.alpha = float(case.density_reaction)
        self.window = case.density_source_window
        separable = case.density_source_rate is not None and case.density_source_profile is not None
        self.rate, self.profile = ((case.density_source_rate, case.density_source_profile) if separable
                                   else (None, None))
        self.reaction_profile_function = getattr(case, "density_reaction_profile", None)
        self.reaction_rate = getattr(case, "density_reaction_rate", None)
        self.cache_size = int(cache_size)
        self._cache: dict[float, object] = {}
        self._profile_field = None
        self._nu_field = None
        self._reaction_cache: dict[tuple[float, float], object] = {}
        self._device_tables: dict[int, tuple[object, object, object]] = {}

    @property
    def active(self) -> bool:
        """Return whether the case forces the density equation."""
        return self.source_function is not None or self.has_reaction

    @property
    def has_reaction(self) -> bool:
        """Return whether the density equation has a reaction term alpha*rho."""
        return self.alpha != 0.0 or self.reaction_profile_function is not None

    @property
    def time_dependent_reaction(self) -> bool:
        """Return whether alpha_h changes in time (then no transport operator may be reused across times)."""
        return self.reaction_profile_function is not None and self.reaction_rate is not None

    def check_scheme(self, scheme: str) -> None:
        """Reject schemes whose transport solves are not of the implicit-stage form above."""
        if self.active and scheme not in SUPPORTED_SCHEMES:
            raise ValueError(f"density source/reaction terms support {SUPPORTED_SCHEMES}, not {scheme!r}")

    def source(self, time: float, like=None):
        """Projected S_h(t) on the devices holding ``like``; None when S vanishes at ``time``."""
        if self.source_function is None:
            return None
        t = float(time)
        if self.window is not None and not self.window[0] <= t <= self.window[1]:
            return None
        if self.profile is not None:  # S = rate(t) * profile(x, y): project the profile once
            if self._profile_field is None:
                self._profile_field = self.space.project_callable(self.profile, name="S_profile_h")
            field, weight = self._profile_field, float(self.rate(t))
        else:
            key = round(t, 12)
            field = self._cache.get(key)
            if field is None:
                field = self.space.project_callable(lambda x, y: self.source_function(x, y, t), name="S_h")
                self._cache[key] = field
                while len(self._cache) > self.cache_size:
                    self._cache.pop(next(iter(self._cache)))
            weight = 1.0
        if like is not None:
            from hybridge.core.device import match_device_residency

            match_device_residency(field, like)
        if weight == 1.0:
            return field
        from hybridge.core.field_ops import field_linear_combination

        return field_linear_combination(self.space, [(weight, field)], name="S_h")

    def reaction_weight(self, time) -> float:
        """Weight r(t) of the space-dependent reaction part (0 without one)."""
        if self.reaction_profile_function is None:
            return 0.0
        return 1.0 if self.reaction_rate is None else float(self.reaction_rate(float(time)))

    def nu(self, like=None):
        """Projected space-dependent reaction part nu_h, on the devices holding ``like``."""
        if self._nu_field is None:
            self._nu_field = self.space.project_callable(self.reaction_profile_function, name="nu_h")
        if like is not None:
            from hybridge.core.device import match_device_residency

            match_device_residency(self._nu_field, like)
        return self._nu_field

    def reaction(self, scale: float, one, time: float | None = None, like=None):
        """Reaction coefficient 1 - scale*alpha_h(t) of the implicit solve (``one`` without reaction)."""
        if self.time_dependent_reaction and time is None:
            raise ValueError("a time-dependent reaction needs the stage time")
        weight = self.reaction_weight(time)
        if weight == 0.0:
            if self.alpha == 0.0:
                return one
            return self.space.constant(1.0 - self.alpha * float(scale), name="reaction_h")
        key = (round(float(scale), 15), round(weight, 15))
        field = self._reaction_cache.get(key)
        if field is None:
            from hybridge.core.field_ops import field_linear_combination

            field = field_linear_combination(
                self.space, [(1.0 - self.alpha * float(scale), one),
                             (-float(scale) * weight, self.nu(like=like))], name="reaction_h")
            self._reaction_cache[key] = field
            while len(self._reaction_cache) > 4:
                self._reaction_cache.pop(next(iter(self._reaction_cache)))
        return field

    def stage_source(self, source, scale: float, time: float):
        """Right-hand side source + scale*S_h(t) of the implicit solve at the stage time."""
        term = self.source(time, like=source)
        if term is None:
            return source
        from hybridge.core.field_ops import field_linear_combination

        return field_linear_combination(self.space, [(1.0, source), (float(scale), term)], name=source.name)

    def rhs(self, density, time: float):
        """Forcing S_h(t) + Pi(alpha_h(t) rho) of an explicit residual, or None when it vanishes."""
        terms = []
        term = self.source(time, like=density)
        if term is not None:
            terms.append((1.0, term))
        if self.alpha != 0.0:
            terms.append((self.alpha, density))
        weight = self.reaction_weight(time)
        if weight != 0.0:
            terms.append((weight, self._nu_product(density)))
        if not terms:
            return None
        from hybridge.core.field_ops import field_linear_combination

        return field_linear_combination(self.space, terms, name="forcing_h")

    def production(self, density, time: float) -> float | None:
        """Density produced per unit time by the reaction, int alpha_h(t) rho_h (None without reaction).

        Exactly the reaction term of the discrete mass balance: testing the
        implicit solve with 1 gives int (alpha_0 + r(t) nu_h) rho_h, an integral of
        two degree-p polynomials evaluated here from the coefficients.
        """
        if not self.has_reaction:
            return None
        weight = self.reaction_weight(time)
        ones = self.space._constant_reference_coeffs(1.0)
        devices = sorted(density._device_coeffs or {})
        if devices:
            cp, jac, mass, _ = self._tables(devices[0])
            with cp.cuda.Device(devices[0]):
                rho = density._device_coefficients_for(devices[0])
                coefficient = cp.broadcast_to(self.alpha * cp.asarray(ones)[None, :], rho.shape)
                if weight != 0.0:
                    coefficient = coefficient + weight * self.nu(like=density)._device_coefficients_for(devices[0])
                return float(cp.einsum("K,Ki,ij,Kj->", jac, coefficient, mass, rho))
        import numpy as np

        coefficient = np.broadcast_to(self.alpha * ones[None, :], density.coeffs.shape)
        if weight != 0.0:
            coefficient = coefficient + weight * self.nu().coeffs
        q = self.space.quad_data
        return float(np.einsum("K,Ki,ij,Kj->", self.space.mesh.aff_jacs, coefficient, q.MKrf, density.coeffs))

    def source_production(self, time: float, like=None) -> float | None:
        """Density injected per unit time by the source, int S_h(t) (None without a source).

        The source term of the discrete mass balance, at the stage time of the solve.
        """
        if self.source_function is None:
            return None
        field = self.source(time, like=like)
        if field is None:
            return 0.0
        ones = self.space._constant_reference_coeffs(1.0)
        devices = sorted(field._device_coeffs or {})
        if devices:
            cp, jac, mass, _ = self._tables(devices[0])
            with cp.cuda.Device(devices[0]):
                values = field._device_coefficients_for(devices[0])
                return float(cp.einsum("K,i,ij,Kj->", jac, cp.asarray(ones), mass, values))
        import numpy as np

        q = self.space.quad_data
        return float(np.einsum("K,i,ij,Kj->", self.space.mesh.aff_jacs, ones, q.MKrf, field.coeffs))

    def _tables(self, device_id: int):
        """CuPy module, element Jacobians, reference mass and triple products on one device."""
        from hybridge.runtime.optional import require_cupy

        cp = require_cupy()
        tables = self._device_tables.get(device_id)
        if tables is None:
            q = self.space.quad_data
            with cp.cuda.Device(device_id):
                tables = (cp.asarray(self.space.mesh.aff_jacs), cp.asarray(q.MKrf),
                          cp.asarray(q.weighted_triple_phi_flat))
            self._device_tables[device_id] = tables
        return (cp, *tables)

    def _nu_product(self, density):
        """L2 projection Pi(nu_h rho_h) onto the density space, on the density's device."""
        devices = sorted(density._device_coeffs or {})
        if not devices:
            return self.nu().project_product(density, name="nu_rho_h")
        from hybridge.core.device import field_from_cupy_coefficients

        device_id = devices[0]
        cp, _, _, triple = self._tables(device_id)
        q, el = self.space.quad_data, self.space.el_dof
        with cp.cuda.Device(device_id):
            nu = self.nu(like=density)._device_coefficients_for(device_id)
            rho = density._device_coefficients_for(device_id)
            weighted_mass = (nu @ triple).reshape(-1, el, el)  # int nu_h phi_j phi_i on the reference
            coeffs = cp.einsum("Kj,Kji->Ki", rho, weighted_mass) @ cp.asarray(q.MKrf_inv)
            return field_from_cupy_coefficients(self.space, cp.ascontiguousarray(coeffs), device=device_id,
                                                name="nu_rho_h")
