"""Spatial diffusion stabilization contracts shared by all ADR backends."""
import numpy as np
import pytest
from hybridge import DGSpace, rectangle_mesh
from hybridge.hdg import matrices
import hybridge.hdg.coefficients as hdg_coefficients
from hybridge.mixed.adr_preparation import (
    prepare_adr_data,
    normalize_diffusion_stabilization,
    diffusion_stabilization_on_trace,
)
from hybridge.mixed.postprocess.total_flux import _adr_postprocess_samples


def setup():
    space=DGSpace(rectangle_mesh(2,1),2,basis_type='dub_orth')
    beta=(space*space).field((space.constant(.7),space.constant(-.2)))
    return space,beta


@pytest.mark.parametrize('kind',['scalar','element','incidence','field','geometry','legacy','context'])
def test_supported_face_laws_and_resampling(kind):
    space,beta=setup()
    mesh=space.mesh
    def context(x,y,*,element,local_face,normal,t=None):
        return 2.+.1*x+.05*y+.2*element+.03*local_face+.02*normal[...,0]+t
    laws={'scalar':2.,'element':np.arange(mesh.num_tri)+1.,
          'incidence':np.arange(mesh.num_tri*3).reshape(-1,3)+1.,
          'field':DGSpace(mesh,1).project_callable(lambda x,y: 2.+.1*x),
          'geometry':lambda x,y: 2.+.1*x+.05*y,
          'legacy':lambda x,y,K,e: 2.+.1*x+.05*y+.2*K+.03*e,
          'context':context}
    law=laws[kind]
    trace=space.trace_space('legendre-modal')
    prep=prepare_adr_data(space.constant(1.),.2,beta,space,diffusion_stabilization=law,trace_space=trace,t=.3)
    expected=hdg_coefficients._face_quadrature_values_from_scalar_input(law,space,'tau',trace_space=trace,t=.3)
    np.testing.assert_allclose(prep.tau_total-prep.tau_advection,expected)
    post=DGSpace(mesh,space.order+1,basis_type='dub_orth')
    post_trace=post.trace_space('bernstein')
    resampled=diffusion_stabilization_on_trace(prep,space,post_trace)
    expected_post=hdg_coefficients._face_quadrature_values_from_scalar_input(law,space,'tau',trace_space=post_trace,t=.3)
    np.testing.assert_allclose(resampled,expected_post)
    _,beta_face,total=_adr_postprocess_samples(beta,prep,space,post,None)
    normal=np.einsum('Kfqd,Kfd->Kfq',beta_face,mesh.normals)
    np.testing.assert_allclose(total,abs(normal)+expected_post)


def test_quadrature_table_recovery_requires_identical_nodes():
    space,beta=setup()
    trace=space.trace_space('legendre-modal')
    table=np.ones((space.mesh.num_tri,3,trace.weights.size))
    prep=prepare_adr_data(space.constant(1.),.2,beta,space,diffusion_stabilization=table,trace_space=trace)
    np.testing.assert_array_equal(diffusion_stabilization_on_trace(prep,space,trace),table)
    other=DGSpace(space.mesh,3).trace_space('legacy-lagrange')
    with pytest.raises(ValueError,match='incompatible.*quadrature'):
        diffusion_stabilization_on_trace(prep,space,other)


@pytest.mark.parametrize('law',[0.,-1.,np.inf,lambda x,y: np.nan+x*0.,lambda x,y: -x*0.])
def test_invalid_tau_samples(law):
    space,beta=setup()
    with pytest.raises(ValueError,match='finite and strictly positive'):
        prepare_adr_data(space.constant(1.),.2,beta,space,diffusion_stabilization=law)


def test_other_mesh_field_rejected():
    space,beta=setup()
    field=DGSpace(rectangle_mesh(1,1),1).constant(1.)
    with pytest.raises(ValueError,match='mesh'):
        prepare_adr_data(space.constant(1.),.2,beta,space,diffusion_stabilization=field)


def test_callable_internal_typeerror_is_not_retried():
    space,_=setup()
    calls=[]
    def law(x,y,*,element,local_face,normal,t=None):
        calls.append(1)
        raise TypeError('user law failure')
    with pytest.raises(TypeError,match='user law failure'):
        normalize_diffusion_stabilization(law,space)
    assert calls==[1]


def test_light_preparation_never_builds_dense_local_operators(monkeypatch):
    space,beta=setup()
    def forbidden(*args,**kwargs):
        raise AssertionError('dense local operator preparation')
    monkeypatch.setattr(matrices,'boundary_mass_from_trace_stabilization',forbidden)
    result=prepare_adr_data(space.constant(1.),.3,beta,space,dense_local_matrices=False)
    assert result.element_boundary is None and result.trace_lift is None
    assert result.source_rhs.shape==space.shape


@pytest.mark.parametrize(
    ('policy', 'factor'),
    [(None, 1.), ('conflict-averaged-upwind', 1.), ('lax-friedrichs', 2.), ('scaled', 3.)],
)
def test_postprocess_samples_accept_every_upwind_policy(policy, factor):
    """Postprocessing tau_adv follows the same upwind-family rule as ADR assembly."""
    from hybridge.hdg.stabilization import ScaledUpwind

    space, beta = setup()
    if policy == 'scaled':
        policy = ScaledUpwind(factor)
    prep = prepare_adr_data(space.constant(1.), .2, beta, space, diffusion_stabilization=.5,
                            advection_stabilization=policy, trace_space=space.trace_space('legendre-modal'))
    post = DGSpace(space.mesh, space.order + 1, basis_type='dub_orth')
    _, beta_face, total = _adr_postprocess_samples(beta, prep, space, post, policy)
    normal = np.einsum('Kfqd,Kfd->Kfq', beta_face, space.mesh.normals)
    # Constant beta has no interior double outflow, so the conflict repair is inactive.
    np.testing.assert_allclose(total, factor * abs(normal) + .5)


def test_lax_friedrichs_flux_postprocess_runs_end_to_end():
    """An ADR solve with Lax-Friedrichs stabilization recovers the p+1 flux."""
    from hybridge import solve_advection_diffusion_reaction_hdg
    from hybridge.hdg.stabilization import ScaledUpwind

    space, beta = setup()
    common = dict(diffusion=.3, assembly_backend='numba', solver='direct', preconditioner=None,
                  hdg_postprocess='flux', flux_postprocess_space='RT_projection',
                  postprocessing_backend='numba', verbose=False)
    lf = solve_advection_diffusion_reaction_hdg(space.constant(1.), beta, .2, 0., space,
                                                advection_stabilization='lax-friedrichs', **common)
    scaled = solve_advection_diffusion_reaction_hdg(space.constant(1.), beta, .2, 0., space,
                                                    advection_stabilization=ScaledUpwind(2.), **common)
    for a, b in zip(lf.postprocessed_flux.components, scaled.postprocessed_flux.components):
        np.testing.assert_allclose(a.coeffs, b.coeffs, rtol=1e-12, atol=1e-13)
