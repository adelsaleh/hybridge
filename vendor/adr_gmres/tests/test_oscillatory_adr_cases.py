"""Independent derivative and operator controls for oscillatory ADR cases."""
import numpy as np
import pytest
from scripts.oscillatory_adr_cases import VARIANTS,exact_data,make_case,parameters


@pytest.mark.parametrize('name',list(VARIANTS))
def test_manufactured_flux_divergence_and_boundary(name):
    """Check source by differentiating total flux, plus derivatives, divergence and SPD."""
    kw,u=make_case(name)
    rng=np.random.default_rng(841)
    x,y=rng.uniform(-1,1,(2,61)); h=1e-25
    data=exact_data(x,y)
    np.testing.assert_allclose(np.imag(u(x+1j*h,y))/h,data[1],rtol=2e-14,atol=2e-13)
    np.testing.assert_allclose(np.imag(u(x,y+1j*h))/h,data[2],rtol=2e-14,atol=2e-13)
    beta=tuple(v if callable(v) else (lambda x,y,v=v:v+0*x) for v in kw['beta'])
    div=np.imag(beta[0](x+1j*h,y)+beta[1](x,y+1j*h))/h
    np.testing.assert_allclose(div,0.,atol=4e-14)
    tensor=np.asarray(parameters(name)['diffusion_tensor'])
    assert np.linalg.eigvalsh(tensor).min()>0
    def flux(x,y,component):
        """Evaluate total conservative flux without using the source Hessian."""
        val,ux,uy,*_=exact_data(x,y)
        return -tensor[component,0]*ux-tensor[component,1]*uy+beta[component](x,y)*val
    reference=(np.imag(flux(x+1j*h,y,0)+flux(x,y+1j*h,1))/h+kw['reaction']*u(x,y))
    np.testing.assert_allclose(kw['source'](x,y),reference,rtol=3e-13,atol=2e-12)
    for edge in (-1.,1.):
        np.testing.assert_allclose(u(np.full_like(x,edge),y),0.,atol=2e-14)
        np.testing.assert_allclose(u(x,np.full_like(y,edge)),0.,atol=2e-14)


def test_solution_oscillations_change_rhs_only():
    """Hold coefficients fixed and verify the assembled trace matrix is identical."""
    from scripts.adv_diff_rea_cases import get_case
    from hdgfem.core.mesh import rectangle_mesh
    from hdgfem.core.space import DGSpace
    from hdgfem.solvers.adv_diff_rea import assemble_advection_diffusion_reaction
    space=DGSpace(rectangle_mesh(3,3),2,basis_type='dub_orth')
    baseline=assemble_advection_diffusion_reaction(space=space,**get_case('advection_dominated')[0])
    oscillatory=assemble_advection_diffusion_reaction(space=space,**get_case('oscillatory_rhs')[0])
    np.testing.assert_array_equal(baseline.system.blocks,oscillatory.system.blocks)
    assert not np.allclose(baseline.system.rhs,oscillatory.system.rhs)
