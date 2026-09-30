"""Continuous PDE data checks; no solve or time integration."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.n_gamma.cases import forcing as f


def _exact(x, y, t, stationary):
    s = 0.0 if stationary else t
    n = 2+.2*np.sin(np.pi*x-s)*np.cos(np.pi*y+2*s)+.1*np.cos(2*np.pi*x+s)*np.sin(np.pi*y-s)
    u = .2+.4*np.cos(np.pi*x+2*s)*np.sin(np.pi*y-s)+.1*np.sin(2*np.pi*x-s)*np.cos(np.pi*y+3*s)
    return n, u, n*u


def _derivative(fun, coordinates, axis, h=5e-4):
    def at(offset):
        shifted = list(coordinates)
        shifted[axis] = shifted[axis]+offset*h
        return fun(*shifted)
    return (at(-2)-8*at(-1)+8*at(1)-at(2))/(12*h)


def _independent_sources(x, y, t, stationary, geometry):
    def field(index):
        return lambda a, b, c: _exact(a, b, c, stationary)[index]

    def gradient(index, a, b, c):
        return np.stack([_derivative(field(index), (a,b,c), i) for i in (0,1)], axis=-1)

    def flux(index, a, b, c):
        n, u, gamma = _exact(a, b, c, stationary)
        magnetic = np.stack([-b, a], axis=-1)/np.sqrt(1+a*a+b*b)[...,None]
        projector = np.eye(2)-magnetic[..., :, None]*magnetic[..., None, :]
        advected = gamma if index == 0 else n*u*u
        diffusion = .02 if index == 0 else .03
        return advected[..., None]*magnetic-diffusion*np.einsum("...ij,...j->...i", projector, gradient(index,a,b,c))

    sources = []
    for index in (0,2):
        # Axisymmetric: (1/R) d_R(R F_R) with R = x + 3; Cartesian: plain d_x F_x.
        weight = (lambda a: a+3) if geometry == "axisymmetric" else (lambda a: 1.+0*a)
        radial = lambda a,b,c: weight(a)*flux(index,a,b,c)[...,0]
        vertical = lambda a,b,c: flux(index,a,b,c)[...,1]
        source = (_derivative(radial,(x,y,t),0)/weight(x)
                  + _derivative(vertical,(x,y,t),1)
                  + _derivative(field(index),(x,y,t),2))
        if index == 2:
            grad_n = gradient(0,x,y,t)
            source += (-y*grad_n[...,0]+x*grad_n[...,1])/np.sqrt(1+x*x+y*y)
        sources.append(source)
    return sources


@pytest.mark.parametrize("geometry", ["cartesian", "axisymmetric"])
@pytest.mark.parametrize("stationary", [False, True])
def test_continuous_forcing_matches_independent_finite_difference_pde(stationary, geometry):
    rng = np.random.default_rng(891)
    x, y = rng.uniform(-1, 1, (2, 79))
    t = rng.uniform(0, 1, 79)
    sn, sg = _independent_sources(x,y,t,stationary,geometry)
    kwargs = dict(stationary=stationary, geometry=geometry)
    np.testing.assert_allclose(f.S_n(x,y,t,**kwargs), sn, atol=3e-7, rtol=2e-7)
    np.testing.assert_allclose(f.S_Gamma(x,y,t,**kwargs), sg, atol=3e-7, rtol=2e-7)


def test_geometry_is_required_and_changes_only_the_divergence():
    x, y = np.array([.3, -.5]), np.array([.2, .7])
    with pytest.raises(TypeError):
        f.S_n(x, y, .1)
    with pytest.raises(ValueError, match="geometry"):
        f.S_Gamma(x, y, .1, geometry="slab")
    # Curvature term: axisymmetric - Cartesian = F_R/R for the density flux F.
    n, u, gamma = _exact(x, y, .1, False)
    b = np.stack([-y, x], axis=-1)/np.sqrt(1+x*x+y*y)[..., None]
    grad_n = np.stack([_derivative(lambda a,c,d: _exact(a,c,d,False)[0], (x,y,.1), i) for i in (0,1)], axis=-1)
    projector = np.eye(2)-b[..., :, None]*b[..., None, :]
    radial_flux = gamma*b[..., 0]-.02*np.einsum("...j,...j->...", projector[..., 0, :], grad_n)
    difference = f.S_n(x, y, .1, geometry="axisymmetric")-f.S_n(x, y, .1, geometry="cartesian")
    np.testing.assert_allclose(difference, radial_flux/(x+3), atol=1e-8)


def test_stationary_sources_freeze_and_remove_time_derivatives():
    x = np.linspace(-.9,.9,40)
    y = .31+0*x
    for geometry in ("cartesian", "axisymmetric"):
        for source,index in ((f.S_n,0),(f.S_Gamma,2)):
            stationary = source(x,y,.83,stationary=True,geometry=geometry)
            np.testing.assert_array_equal(stationary, source(x,y,0,stationary=True,geometry=geometry))
            derivative = _derivative(lambda a,b,c: _exact(a,b,c,False)[index], (x,y,0), 2)
            np.testing.assert_allclose(source(x,y,0,geometry=geometry)-stationary, derivative, atol=2e-11)
            assert np.max(np.abs(derivative)) > .1


@pytest.mark.parametrize("stationary", [False, True])
def test_exact_fields_density_bounds_and_broadcasting(stationary):
    rng = np.random.default_rng(531)
    x,y,t = rng.uniform(-5,5,(3,10000))
    exact = _exact(x,y,t,stationary)
    for evaluator,expected in zip((f.n_e,f.u_e,f.Gamma_e),exact):
        np.testing.assert_allclose(evaluator(x,y,t,stationary=stationary), expected, atol=8e-15)
    n = f.n_e(x,y,t,stationary=stationary)
    assert np.min(n) >= 1.7 and np.max(n) <= 2.3
    for evaluator in (f.n_e,f.u_e,f.Gamma_e,f.S_n,f.S_Gamma):
        kwargs = {"geometry": "cartesian"} if evaluator in (f.S_n, f.S_Gamma) else {}
        assert evaluator(x[:3,None], y[:4], .2, stationary=stationary, **kwargs).shape == (3,4)
        assert np.ndim(evaluator(.1,.2,.3,stationary=stationary, **kwargs)) == 0


def test_magnetic_projector_eigenvalues_and_no_renormalization():
    x = np.array([0.,.3,-.9,1.2])
    y = np.array([0.,-.7,.2,1.])
    b,p = f.b_p(x,y),f.P(x,y)
    q = 1+x*x+y*y
    np.testing.assert_allclose(np.sum(b*b,axis=-1), 1-1/q, atol=2e-16)
    np.testing.assert_allclose(p,np.eye(2)-b[..., :,None]*b[...,None,:], atol=2e-16)
    np.testing.assert_allclose(np.linalg.eigvalsh(p),np.stack([1/q,np.ones_like(q)],axis=-1))
    assert f.b_p(.2,.1).shape == (2,)
    assert f.P(.2,.1).shape == (2,2)


def test_generated_evaluators_match_committed_module():
    pytest.importorskip("sympy")
    from scripts.n_gamma.manufactured import render_module, render_numba_module
    assert Path(f.__file__).read_text() == render_module()
    assert Path(f.__file__).with_name("forcing_numba.py").read_text() == render_numba_module()


def test_runtime_evaluators_do_not_import_sympy():
    subprocess.run([sys.executable,"-c", "from scripts.n_gamma.cases.forcing import S_Gamma; "
                    "S_Gamma(.2,.3,.4,geometry='cartesian'); import sys; assert 'sympy' not in sys.modules"], check=True)


@pytest.mark.parametrize("stationary", [False, True])
def test_cupy_evaluators_match_numpy_without_download(monkeypatch,stationary):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("No CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    x = np.linspace(-.9,.9,23)[:,None]
    y = np.linspace(-.8,.8,11)
    cx,cy = cp.asarray(x),cp.asarray(y)
    outputs = []
    def forbid(*args,**kwargs):
        raise AssertionError("Unexpected host download in manufactured evaluator")
    with monkeypatch.context() as context:
        context.setattr(cp,"asnumpy",forbid)
        for evaluator in (f.n_e,f.u_e,f.Gamma_e,f.S_n,f.S_Gamma,f.b_p,f.P):
            for geometry in ("cartesian", "axisymmetric"):
                kwargs = {} if evaluator in (f.b_p,f.P) else {"stationary":stationary}
                if evaluator in (f.S_n, f.S_Gamma):
                    kwargs["geometry"] = geometry
                value = evaluator(cx,cy,.37,**kwargs)
                assert isinstance(value,cp.ndarray)
                outputs.append((evaluator,kwargs,value))
    for evaluator,kwargs,value in outputs:
        np.testing.assert_allclose(cp.asnumpy(value), evaluator(x,y,.37,**kwargs), atol=2e-14,rtol=2e-13)
