"""Manufactured ADR stress cases using a published cellular-flow family.

The velocity follows the sinusoidal cellular streamfunction used by Haynes and
Vanneste, arXiv:1401.6666, with an added constant drift. The Fourier solution,
reaction and diffusion parameters below are our manufactured extensions, not
benchmarks claimed to occur in that paper. Promoted from the frozen
``vendor/adr_gmres/scripts/oscillatory_adr_cases.py`` study definitions; the
archived study remains untouched for reproducibility.
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np

MODES = ((1.0, 6, 5), (0.35, 11, 9))
VARIANTS = {
    'oscillatory_rhs': dict(frequency=0, amplitude=0., diffusion=1e-3, reaction=1.),
    'cellular4': dict(frequency=4, amplitude=4., diffusion=1e-3, reaction=1.),
    'cellular7': dict(frequency=7, amplitude=4., diffusion=1e-3, reaction=1.),
    'cellular7_low': dict(frequency=7, amplitude=4., diffusion=1e-5, reaction=1.),
    'cellular7_high': dict(frequency=7, amplitude=4., diffusion=1., reaction=.01),
    'cellular7_directional': dict(frequency=7, amplitude=4., diffusion=(1.,1e-3), reaction=.01),
    'cellular7_weak': dict(frequency=7, amplitude=4., diffusion=1e-5, reaction=.01),
    'cellular7_anisotropic': dict(frequency=7, amplitude=4., diffusion=(1e-4,1e-6), reaction=.01),
}


def exact_data(x, y):
    """Return the two-mode solution, gradient and Hessian analytically."""
    x,y=np.broadcast_arrays(x,y)
    data=[np.zeros_like(x,dtype=np.result_type(x,y,float)) for _ in range(6)]
    for amplitude,m,n in MODES:
        a,b=np.pi*m,np.pi*n
        sx,cx,sy,cy=np.sin(a*x),np.cos(a*x),np.sin(b*y),np.cos(b*y)
        u=amplitude*sx*sy
        for target,value in zip(data,(u,amplitude*a*cx*sy,amplitude*b*sx*cy,
                                     -a*a*u,amplitude*a*b*cx*cy,-b*b*u)):
            target += value
    return tuple(data)


def parameters(name):
    """Return serializable parameters with the explicit SPD diffusion tensor."""
    values=deepcopy(VARIANTS[name])
    diffusivity=values['diffusion']
    if np.ndim(diffusivity):
        c,s=np.cos(np.pi/6),np.sin(np.pi/6)
        rotation=np.array([[c,-s],[s,c]])
        tensor=rotation@np.diag(diffusivity)@rotation.T
    else:tensor=np.eye(2)*diffusivity
    values.update(diffusion_tensor=tensor.tolist(),drift=[1.,.5],solution_modes=MODES)
    return values


def make_case(name):
    """Construct conservative source, divergence-free velocity and exact boundary data."""
    values=parameters(name)
    tensor=np.asarray(values['diffusion_tensor'])
    amplitude=values['amplitude']; frequency=values['frequency']; reaction=values['reaction']
    wave=np.pi*frequency
    def exact(x,y):
        """Return the prescribed oscillatory primal solution."""
        return exact_data(x,y)[0]
    def beta_x(x,y):
        """Return the x-component of the drift plus cellular flow."""
        return 1.+amplitude*np.sin(wave*x)*np.cos(wave*y)
    def beta_y(x,y):
        """Return the y-component; its derivative cancels d(beta_x)/dx."""
        return .5-amplitude*np.cos(wave*x)*np.sin(wave*y)
    def source(x,y):
        """Evaluate -div(K grad u) + div(beta u) + reaction*u, with div(beta)=0."""
        u,ux,uy,uxx,uxy,uyy=exact_data(x,y)
        return -tensor[0,0]*uxx-2*tensor[0,1]*uxy-tensor[1,1]*uyy+beta_x(x,y)*ux+beta_y(x,y)*uy+reaction*u
    beta=(1.,.5) if amplitude==0 else (beta_x,beta_y)
    diffusion=values['diffusion'] if np.ndim(values['diffusion'])==0 else tensor
    return dict(source=source,beta=beta,reaction=reaction,boundary_condition=exact,
                diffusion=diffusion),exact
