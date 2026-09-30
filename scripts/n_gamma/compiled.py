"""Numba-compiled n-Gamma coefficients for the host path (``pointwise_coefficient``/``pointwise_law``).

:class:`CompiledCoefficients` builds, for one space and manufactured case, the
same coefficients as :mod:`scripts.n_gamma.coefficients` from the scalar
evaluators of ``cases/forcing_numba.py``, with the same arithmetic order. The
coefficient functions below are compiled once as Numba ``cfunc`` (cached on
disk) and evaluated in parallel; each step only rebinds fields and time.
They are module-level functions reading their data from ``v``:

* ``diffusion_*``:      ``v = [coefficient, axisymmetric, shift]``
* ``advection_*``:      ``v = [n*, Gamma*, floor, axisymmetric, shift]``
* ``density_source``:   ``v = [h_n, axisymmetric, shift, variant]``
* ``momentum_source``:  ``v = [h_Gamma, dn/dx, dn/dy, c_s^2, axisymmetric, shift, variant]``

``axisymmetric`` is 1 for ``W = R`` (else ``W = 1``), ``shift`` the case-frame
shift (``x = first - shift``) and ``variant`` the ``forcing_numba.VARIANTS``
index of the case's geometry and stationarity. The device path keeps the
NumPy/CuPy coefficients.
"""
from __future__ import annotations

from numba import njit

from hdgfem import pointwise_coefficient, pointwise_law

from .cases import forcing_numba
from .cases.geometry import frame_shift
from .coefficients import check_geometry


@njit(cache=True)
def _weight(first, axisymmetric):
    """Measure weight ``W``: ``R`` (the first coordinate) or 1."""
    return first if axisymmetric != 0. else 1.


@njit(cache=True)
def _b_poloidal(first, second, shift):
    """Poloidal field ``(b_1, b_2)`` at mesh coordinates."""
    return forcing_numba.b_p(first - shift, second, 0.)


def diffusion_00(x, y, t, v):
    b_1, b_2 = _b_poloidal(x, y, v[2])
    return v[0]*_weight(x, v[1])*(1. - b_1*b_1)


def diffusion_01(x, y, t, v):
    b_1, b_2 = _b_poloidal(x, y, v[2])
    return v[0]*_weight(x, v[1])*(-b_1*b_2)


def diffusion_11(x, y, t, v):
    b_1, b_2 = _b_poloidal(x, y, v[2])
    return v[0]*_weight(x, v[1])*(1. - b_2*b_2)


def advection_0(x, y, t, v):
    velocity = v[1]/max(v[0], v[2])*_weight(x, v[3])
    b_1, b_2 = _b_poloidal(x, y, v[4])
    return velocity*b_1


def advection_1(x, y, t, v):
    velocity = v[1]/max(v[0], v[2])*_weight(x, v[3])
    b_1, b_2 = _b_poloidal(x, y, v[4])
    return velocity*b_2


def density_source(x, y, t, v):
    return _weight(x, v[1])*(forcing_numba.S_n(x - v[2], y, t, int(v[3])) + v[0])


def momentum_source(x, y, t, v):
    b_1, b_2 = _b_poloidal(x, y, v[5])
    values = forcing_numba.S_Gamma(x - v[5], y, t, int(v[6])) + v[0] - v[3]*(b_1*v[1] + b_2*v[2])
    return _weight(x, v[4])*values


class CompiledCoefficients:
    """Compiled n-Gamma coefficients of one space and manufactured case (host Numba path).

    Mirrors :func:`~scripts.n_gamma.coefficients.diffusion_tensor`,
    :func:`~scripts.n_gamma.coefficients.advection`,
    :func:`~scripts.n_gamma.coefficients.density_source` and
    :func:`~scripts.n_gamma.coefficients.momentum_source` for ``case``'s
    geometry, frame and stationarity. The first call of each builder compiles
    (or loads from the cache); later calls rebind fields and time.
    """

    def __init__(self, space, case, *, density_floor: float, sound_speed: float = 1.):
        self.space = space
        self.geometry = check_geometry(case.geometry)
        self.floor = float(density_floor)
        if not self.floor > 0.:
            raise ValueError("density floor must be positive")
        self.cs2 = float(sound_speed)**2
        self.frame = (1. if self.geometry == "axisymmetric" else 0., float(frame_shift(self.geometry)))
        self.variant = float(forcing_numba.VARIANTS.index((self.geometry, bool(case.stationary))))
        self._templates = {}

    def diffusion_tensor(self, coefficient: float):
        """Symmetric ``W*coefficient*P`` as three compiled ``(x, y)`` laws ``(k00, k01, k11)``."""
        value = float(coefficient)
        if not value > 0.:
            raise ValueError("diffusion coefficient must be positive")
        return tuple(pointwise_law(function, params=(value, *self.frame), name=f"diffusion{suffix}")
                     for function, suffix in ((diffusion_00, "00"), (diffusion_01, "01"), (diffusion_11, "11")))

    def _bound(self, key, build, **data):
        template = self._templates.get(key)
        if template is None:
            template = self._templates[key] = build()
            return template
        return template.bind(**data)

    def advection(self, n_star, gamma_star):
        """``beta = W*(Gamma*/max(n*, floor))*b_p``."""
        return self._bound("advection", lambda: pointwise_coefficient(
            (advection_0, advection_1), self.space, fields=(n_star, gamma_star),
            params=(self.floor, *self.frame), name="W_u_star_b_p"), fields=(n_star, gamma_star))

    def density_source(self, history, t: float):
        """``W*(S_n + h_n)`` at time ``t``."""
        return self._bound("density_source", lambda: pointwise_coefficient(
            density_source, self.space, fields=(history,), params=(*self.frame, self.variant), time=t,
            name="density_source"), fields=(history,), time=t)

    def momentum_source(self, history, density, t: float):
        """``W*(S_Gamma + h_Gamma - c_s^2 b_p . grad n_h)`` with the new density ``n_h``, at time ``t``."""
        return self._bound("momentum_source", lambda: pointwise_coefficient(
            momentum_source, self.space, fields=(history,), gradients=(density,),
            params=(self.cs2, *self.frame, self.variant), time=t, name="momentum_source"),
            fields=(history,), gradients=(density,), time=t)
