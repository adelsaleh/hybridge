"""Catalogue of stationary ADR problems; importing it never builds or solves a mesh."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

import numpy as np

from . import closed_loop_stress_cases as stress
from . import oscillatory_cases as oscillatory
from .disk_case import manufactured_adr_disk
from .tensor_cases import diffusion_cases, manufactured_raw_tensor, manufactured_tensor, sine_data


@dataclass(frozen=True)
class AdvectionDiffusionReactionProblem:
    """Coefficients, boundary values and optional analytic error references."""

    source: Any
    beta: tuple
    reaction: Any
    diffusion: Any
    boundary_condition: Callable
    exact: Callable | None = None
    exact_flux: Callable | None = None
    domain: str = "square"
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AdvectionDiffusionReactionCaseDefinition:
    """Named factory and its default parameters, matching the other case runners."""

    key: str
    description: str
    factory: Callable
    default_domain: str = "square"
    default_params: dict = field(default_factory=dict)

    def build(self, **params):
        """Build analytic data, applying explicit parameters over case defaults."""
        return self.factory(**(self.default_params | params))


def _constant_flux(diffusion, gradient):
    tensor = np.asarray(diffusion)
    if tensor.ndim == 0:
        tensor = np.eye(2)*tensor

    def flux(x, y):
        ux, uy = gradient(x, y)
        return -tensor[0, 0]*ux-tensor[0, 1]*uy, -tensor[1, 0]*ux-tensor[1, 1]*uy
    return flux


def legacy_case(name):
    """Promoted baseline definitions from the frozen vendor ADR GMRES study."""
    if name in {"quadratic", "variable_velocity"}:
        exact = lambda x, y: 1+x*x+2*y*y
        gradient = lambda x, y: (2*x, 4*y)
        if name == "quadratic":
            beta, diffusion, reaction = (1.2, -.7), np.array([[2., .3], [.3, 1.]]), .8
            source = lambda x, y: -8+2.4*x-2.8*y+.8*exact(x, y)
        else:
            beta, diffusion, reaction = (lambda x, y: 1+x, lambda x, y: -.5+y), 1., .5
            source = lambda x, y: -6+(1+x)*2*x+(-.5+y)*4*y+2.5*exact(x, y)
    else:
        diffusion = {"trigonometric": 1., "advection_dominated": 1e-3,
                     "anisotropic": np.diag([1., .01])}[name]
        beta, reaction = (1., .5), 1.
        exact = lambda x, y: sine_data(x, y)[0]
        gradient = lambda x, y: sine_data(x, y)[1:3]
        tr = 2*diffusion if np.ndim(diffusion) == 0 else np.trace(diffusion)
        source = lambda x, y: (np.pi**2*tr+reaction)*exact(x, y)+np.pi*(
            np.cos(np.pi*x)*np.sin(np.pi*y)+.5*np.sin(np.pi*x)*np.cos(np.pi*y))
    return AdvectionDiffusionReactionProblem(
        source, beta, reaction, diffusion, exact, exact, _constant_flux(diffusion, gradient))


def oscillatory_case(name):
    """Adapt one of the eight archived cellular/Fourier problems."""
    kwargs, exact = oscillatory.make_case(name)
    return AdvectionDiffusionReactionProblem(
        **kwargs, exact=exact,
        exact_flux=_constant_flux(kwargs["diffusion"], lambda x, y: oscillatory.exact_data(x, y)[1:3]),
        metadata=oscillatory.parameters(name))


def disk_case(peclet=10.):
    """Expose the established disk case through the common problem interface."""
    data = manufactured_adr_disk(peclet)
    return AdvectionDiffusionReactionProblem(
        source=data["source"], beta=(data["beta_x"], data["beta_y"]),
        reaction=data["reaction"], diffusion=1./peclet, boundary_condition=data["exact"],
        exact=data["exact"], exact_flux=data["exact_diffusive_flux"], domain="disk",
        metadata=dict(peclet=peclet))


def stress_case(geometry="annulus", variant="trap", level="main", *,
                normalization=None, epsilon=None, speed=None, neck_width=None, reaction=1e-3):
    """Reuse the six stress families and their existing velocity normalization."""
    if level not in stress.LEVELS:
        raise ValueError(f"unknown stress level: {level}")
    values = dict(stress.LEVELS[level])
    values.update({key: value for key, value in dict(
        epsilon=epsilon, speed=speed, neck_width=neck_width).items() if value is not None})
    parameters = stress.StressParameters(geometry=geometry, variant=variant, reaction=reaction, **values)
    if normalization is None:
        evidence = stress.estimate_normalization(parameters)
        if not evidence["converged"]:
            raise ValueError("stress normalization did not converge; supply a qualified normalization")
        normalization = evidence["value"]
    else:
        evidence = {"value": normalization, "method": "provided"}
    kwargs, exact = stress.make_case(parameters, normalization)

    def flux(x, y):
        _, ux, uy, _, _, _ = stress.case_exact_data(x, y, parameters)
        a, b, d, _, _ = stress.diffusion_data(x, y, parameters)
        return -a*ux-b*uy, -b*ux-d*uy

    return AdvectionDiffusionReactionProblem(
        **kwargs, exact=exact, exact_flux=flux, domain=geometry,
        metadata=dict(stress_parameters=parameters.to_dict(), normalization=evidence,
                      hole_radius=parameters.hole_radius))


def tensor_case(kind="general"):
    """Smooth tensor convergence tests on the unit square."""
    kwargs, exact, flux = manufactured_tensor(kind)
    return AdvectionDiffusionReactionProblem(**kwargs, exact=exact, exact_flux=flux, domain="unit-square")


def raw_tensor_case(solution="affine"):
    """General variable tensor used by CUDA reconstruction and convergence checks."""
    kwargs, exact, flux = manufactured_raw_tensor(solution)
    return AdvectionDiffusionReactionProblem(
        **kwargs, exact=exact, exact_flux=flux,
        domain="unit-square" if solution == "sine" else "square")


def scalar_case(solution="affine"):
    """Manufactured scalar tests from test_advection_diffusion_reaction.py."""
    diffusion = .1
    if solution == "affine":
        exact = lambda x, y: 1.+x+y
        gradient = lambda x, y: (np.ones_like(x), np.ones_like(y))
        beta, reaction, domain = (.7, -.2), .3, "square"
        source = lambda x, y: .5+.3*exact(x, y)
    elif solution == "sine":
        exact = lambda x, y: sine_data(x, y)[0]
        gradient = lambda x, y: sine_data(x, y)[1:3]
        beta, reaction, domain = (1., .3), .5, "unit-square"
        source = lambda x, y: (gradient(x, y)[0]+.3*gradient(x, y)[1]
                               +(2*diffusion*np.pi**2+reaction)*exact(x, y))
    else:
        raise ValueError(f"unknown scalar solution: {solution}")
    return AdvectionDiffusionReactionProblem(
        source, beta, reaction, diffusion, exact, exact,
        _constant_flux(diffusion, gradient), domain=domain)


def coefficient_case(name):
    """Benchmark forcing and boundary data; there is no manufactured exact solution."""
    return AdvectionDiffusionReactionProblem(
        source=lambda x, y: np.ones_like(x+y), beta=(.7, -.2), reaction=.4,
        diffusion=diffusion_cases()[name], boundary_condition=lambda x, y: .2+x-.3*y)


CASE_DEFINITIONS = {}


def _register(key, description, factory, domain="square", **params):
    CASE_DEFINITIONS[key] = AdvectionDiffusionReactionCaseDefinition(
        key, description, factory, domain, params)


for _name in ("quadratic", "variable_velocity", "trigonometric", "advection_dominated", "anisotropic"):
    _register(_name, f"Legacy ADR study: {_name.replace('_', ' ')}.", partial(legacy_case, _name))
for _name in oscillatory.VARIANTS:
    _register(_name, f"Fourier/cellular study: {_name.replace('_', ' ')}.", partial(oscillatory_case, _name))
_register("disk", "Unit disk with variable, non-solenoidal velocity.", disk_case, "disk", peclet=10.)
for _geometry in stress.GEOMETRIES:
    for _variant in stress.VARIANTS:
        _register(f"stress_{_geometry}_{_variant}", f"Closed-loop {_geometry} stress: {_variant}.",
                  partial(stress_case, _geometry, _variant), _geometry, level="main")
for _kind in ("scalar", "diagonal", "symmetric", "general"):
    _register(f"tensor_{_kind}", f"Variable {_kind} tensor with a sine solution.",
              partial(tensor_case, _kind), "unit-square")
for _solution in ("affine", "sine"):
    _domain = "square" if _solution == "affine" else "unit-square"
    _register(_solution, f"Scalar diffusion with a manufactured {_solution} solution.", partial(scalar_case, _solution), _domain)
    _register(f"raw_tensor_{_solution}", f"CUDA general tensor with a manufactured {_solution} solution.",
              partial(raw_tensor_case, _solution), _domain)
for _name in diffusion_cases():
    _register(f"coefficient_{_name}", f"Tensor assembly benchmark: {_name}; no exact solution.",
              partial(coefficient_case, _name))
