"""The manufactured n–Gamma cases: four problems in each of two geometries.

Every case is selected with an explicit ``geometry`` (no default):
``"cartesian"`` cases take poloidal-plane mesh coordinates ``(x, y)`` and use
the plain divergence; ``"axisymmetric"`` cases take ``(R, Z)`` (``x = R - 3``)
and use the axisymmetric divergence. Sources are continuous and unweighted;
the ADR coefficient builder applies the measure weight (1 or R). Host
evaluations run in element chunks on the ``hdgfem.core.host_threads`` pool
(NumPy ufuncs are single-threaded); device arrays are evaluated directly.
"""
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType

import numpy as np

from hdgfem.runtime.threads import elementwise

from . import forcing
from .geometry import GEOMETRIES, build_case_mesh, frame_shift

CASE_NAMES = ("stationary_baseline", "stationary_stress", "transient_baseline", "transient_stress")


def _pointwise(function, first, second, xp=None):
    """Evaluate ``function(first, second)``, host arrays in parallel element chunks."""
    if (xp is not None and xp is not np) or hasattr(first, "__cuda_array_interface__"):
        return function(first, second)
    return elementwise(function, first, second)


@dataclass(frozen=True)
class ManufacturedCase:
    """Exact Dirichlet data on every boundary; no physical Bohm condition.

    Evaluators take mesh coordinates ``(first, second)``: ``(x, y)`` for
    Cartesian cases, ``(R, Z)`` for axisymmetric ones.
    """

    name: str
    domain: str
    stationary: bool
    geometry: str
    boundary_mode: str = "eliminate"
    bohm_conditions: bool = False

    def _xy(self, first, second):
        return first - frame_shift(self.geometry), second

    def density(self, first, second, t=0., *, xp=None):
        """Evaluate the exact density at mesh coordinates."""
        return _pointwise(lambda a, b: forcing.n_e(*self._xy(a, b), t, stationary=self.stationary, xp=xp),
                          first, second, xp)

    def momentum(self, first, second, t=0., *, xp=None):
        """Evaluate the exact parallel momentum at mesh coordinates."""
        return _pointwise(lambda a, b: forcing.Gamma_e(*self._xy(a, b), t, stationary=self.stationary, xp=xp),
                          first, second, xp)

    def density_source(self, first, second, t=0., *, xp=None):
        """Evaluate the continuous, unweighted density forcing of this geometry."""
        return _pointwise(lambda a, b: forcing.S_n(*self._xy(a, b), t, geometry=self.geometry,
                                                   stationary=self.stationary, xp=xp), first, second, xp)

    def momentum_source(self, first, second, t=0., *, xp=None):
        """Evaluate the continuous momentum forcing, including the exact pressure term."""
        return _pointwise(lambda a, b: forcing.S_Gamma(*self._xy(a, b), t, geometry=self.geometry,
                                                       stationary=self.stationary, xp=xp), first, second, xp)

    def b_poloidal(self, first, second):
        """Return the non-normalized poloidal field as a component pair ``(b_1, b_2)``."""
        field = _pointwise(lambda a, b: forcing.b_p(*self._xy(a, b)), first, second)
        return field[..., 0], field[..., 1]

    def density_boundary_at(self, t):
        """Return exact density Dirichlet data for every exterior edge."""
        return partial(self.density, t=t)

    def momentum_boundary_at(self, t):
        """Return exact momentum Dirichlet data, including on the hole."""
        return partial(self.momentum, t=t)

    def density_source_at(self, t):
        """Return the density source bound to time ``t`` (a stepper source factory)."""
        return partial(self.density_source, t=t)

    def momentum_source_at(self, t):
        """Return the momentum source bound to time ``t`` (a stepper source factory)."""
        return partial(self.momentum_source, t=t)

    def build_mesh(self, h=.20, **options):
        """Build this case's domain in its frame, with the requested/actual mesh record."""
        return build_case_mesh(self.domain, h, geometry=self.geometry, **options)


CASES = MappingProxyType({
    (geometry, name): ManufacturedCase(name, "B" if name.endswith("baseline") else "H",
                                       name.startswith("stationary"), geometry)
    for geometry in GEOMETRIES for name in CASE_NAMES
})


def get_case(name, *, geometry):
    """Look up a manufactured case in an explicit geometry with a clear error."""
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry must be one of {GEOMETRIES}; got {geometry!r}")
    try:
        return CASES[(geometry, name)]
    except KeyError:
        raise ValueError(f"unknown n–Gamma case {name!r}; choose {', '.join(CASE_NAMES)}") from None
