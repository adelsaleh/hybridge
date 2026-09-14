"""Manufactured sources include u*div(beta) for conservative transport."""
import numpy as np


def get_case(name):
    if name == 'quadratic':
        exact = lambda x,y: 1+x*x+2*y*y
        beta = (1.2,-.7)
        diffusion = np.array([[2.,.3],[.3,1.]])
        reaction = .8
        source = lambda x,y: -8+2.4*x-2.8*y+.8*exact(x,y)
    elif name == 'variable_velocity':
        exact = lambda x,y: 1+x*x+2*y*y
        beta = (lambda x,y: 1+x,lambda x,y: -.5+y)
        diffusion, reaction = 1., .5
        source = lambda x,y: -6+(1+x)*2*x+(-.5+y)*4*y+2.5*exact(x,y)
    elif name in ('trigonometric','advection_dominated','anisotropic'):
        exact = lambda x,y: np.sin(np.pi*x)*np.sin(np.pi*y)
        beta, reaction = (1.,.5),1.
        diffusion = {'trigonometric':1.,'advection_dominated':1e-3,
                     'anisotropic':np.diag([1.,.01])}[name]
        tr = 2*diffusion if np.ndim(diffusion)==0 else np.trace(diffusion)
        source = lambda x,y: (np.pi**2*tr+reaction)*exact(x,y)+np.pi*(
            np.cos(np.pi*x)*np.sin(np.pi*y)+.5*np.sin(np.pi*x)*np.cos(np.pi*y))
    else:
        raise ValueError(f'unknown case {name}')
    return dict(source=source,beta=beta,reaction=reaction,boundary_condition=exact,
                diffusion=diffusion), exact


CASES = ('quadratic','variable_velocity','trigonometric','advection_dominated','anisotropic')
