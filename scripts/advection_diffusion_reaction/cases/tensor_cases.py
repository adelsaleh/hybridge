"""Shared tensor coefficients and manufactured data for ADR tests and benchmarks."""
from __future__ import annotations

import numpy as np


def diffusion_cases():
    """Return the seven exact structural representatives used by the Numba benchmark."""
    return dict(
        constant_isotropic=2., constant_diagonal=(2., 0., 1.),
        constant_full=(2., .2, 1.),
        variable_isotropic=lambda x, y: 2.+.1*x,
        variable_diagonal=(lambda x, y: 2.+.1*x, 0., lambda x, y: 1.+.1*y),
        variable_symmetric=(lambda x, y: 2.+.1*x, lambda x, y: .2+.03*y,
                            lambda x, y: 1.+.1*y),
        variable_full=(lambda x, y: 2.+.1*x, lambda x, y: .3+.02*y,
                       lambda x, y: -.1+.01*x, 1.))


def raw_cuda_coefficient(name):
    """Return the original raw-CUDA benchmark coefficients without changing values."""
    if name == "scalar":
        return 1.
    if name == "constant-full":
        return (2., .3, -.1, 1.)
    if name == "variable-full":
        return (lambda x, y: 2.+.1*x, lambda x, y: .3+.02*y,
                lambda x, y: -.1+.01*x, lambda x, y: 1.+.05*y)
    raise ValueError(f"unknown raw-CUDA coefficient: {name}")


def sine_data(x, y):
    """Return u, ux, uy, uxx=uyy and uxy for the single Fourier mode."""
    pi = np.pi
    u = np.sin(pi*x)*np.sin(pi*y)
    return (u, pi*np.cos(pi*x)*np.sin(pi*y), pi*np.sin(pi*x)*np.cos(pi*y),
            -pi*pi*u, pi*pi*np.cos(pi*x)*np.cos(pi*y))


def manufactured_tensor(kind="general"):
    """Return the variable-tensor convergence problem from the Numba tests.

    The result is (solver coefficients, exact solution, exact diffusive flux).
    All sources use conservative advection and the full divergence of K grad(u).
    """
    if kind not in {"scalar", "diagonal", "symmetric", "general"}:
        raise ValueError(f"unknown tensor kind: {kind}")

    def components(x, y):
        a = 1.+.2*x
        if kind == "scalar":
            return a, 0.*x, 0.*x, a
        d = 2.+.1*y
        if kind == "diagonal":
            return a, 0.*x, 0.*x, d
        b = .15+.03*x
        return a, b, b if kind == "symmetric" else -.05+.02*y, d

    def exact(x, y):
        return sine_data(x, y)[0]

    def source(x, y):
        u, ux, uy, uxx, uxy = sine_data(x, y)
        a, b, c, d = components(x, y)
        divx = .22 if kind == "general" else .2
        divy = 0. if kind == "scalar" else (.1 if kind == "diagonal" else .13)
        return (.7*ux-.2*uy-(a+d)*uxx-(b+c)*uxy-divx*ux-divy*uy+.5*u)

    def flux(x, y):
        _, ux, uy, _, _ = sine_data(x, y)
        a, b, c, d = components(x, y)
        return -a*ux-b*uy, -c*ux-d*uy

    diffusion = tuple(lambda x, y, j=j: components(x, y)[j] for j in range(4))
    return dict(source=source, beta=(.7, -.2), reaction=.5, diffusion=diffusion,
                boundary_condition=exact), exact, flux


def manufactured_raw_tensor(solution="affine"):
    """Return the general-tensor affine/sine problems used by CUDA solver tests."""
    if solution not in {"affine", "sine"}:
        raise ValueError(f"unknown exact solution: {solution}")

    def data(x, y):
        if solution == "affine":
            return 1.+x-.3*y, 1.+0.*x, -.3+0.*y, 0.*x, 0.*x
        return sine_data(x, y)

    def exact(x, y):
        return data(x, y)[0]

    def source(x, y):
        u, ux, uy, uxx, uxy = data(x, y)
        return .6*ux-.25*uy-(3.+.1*x+.05*y)*uxx-(.2+.02*y+.01*x)*uxy+.3*u

    def flux(x, y):
        _, ux, uy, _, _ = data(x, y)
        return -(2.+.1*x)*ux-(.3+.02*y)*uy, -(-.1+.01*x)*ux-(1.+.05*y)*uy

    return dict(source=source, beta=(.7, -.2), reaction=.3,
                diffusion=raw_cuda_coefficient("variable-full"),
                boundary_condition=exact), exact, flux
