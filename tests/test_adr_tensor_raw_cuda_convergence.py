"""Raw-CUDA tensor ADR manufactured convergence at the n-Gamma orders (p=3, 4).

Bounded stationary solves on 2x2/4x4/8x8 meshes with device assembly,
reconstruction, AMGX and CuPy postprocessing; no time integration. Complements
the p=1,2 variable-full checks in ``test_adr_tensor_solver_cuda.py``.
"""
import numpy as np
import pytest

from hybridge import DGSpace, rectangle_mesh, solve_advection_diffusion_reaction_hdg
from scripts.advection_diffusion_reaction.cases.tensor_cases import manufactured_tensor


@pytest.fixture(scope='module')
def cp():
    cupy = pytest.importorskip('cupy')
    pytest.importorskip('pyamgx')
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip('No CUDA device')
    return cupy


@pytest.mark.parametrize('kind,structure', [('symmetric', 'variable-symmetric'), ('general', 'variable-full')])
@pytest.mark.parametrize('order', [3, 4])
@pytest.mark.parametrize('basis', ['legacy-lagrange', 'legendre-modal'])
@pytest.mark.parametrize('variant', ['l2_closest', 'RT_projection'])
def test_raw_cuda_tensor_postprocessed_convergence(cp, kind, structure, order, basis, variant):
    problem, exact, flux = manufactured_tensor(kind)

    def total_flux(x, y):
        qx, qy = flux(x, y)
        return qx + .7*exact(x, y), qy - .2*exact(x, y)

    errors = []
    for n in (2, 4, 8):
        space = DGSpace(rectangle_mesh(n, n, xlim=(0., 1.), ylim=(0., 1.)), order, basis_type='dub_orth',
                        volume_quad_1d=order+5)
        beta = (space*space).field((space.constant(.7), space.constant(-.2)))
        result = solve_advection_diffusion_reaction_hdg(
            problem['source'], beta, .5, exact, space, diffusion=problem['diffusion'],
            assembly_backend='raw-cuda', solver='amgx', solver_rtol=1e-11, raw_matrix_format='bsr',
            hdg_postprocess='both', flux_postprocess_space=variant, trace_basis=basis,
            materialize_host_solution=False, verbose=False)
        assert isinstance(result.local_unknowns, cp.ndarray)
        assert result.diffusion_structure[structure] == space.mesh.num_tri
        errors.append((result.field.l2_error(exact), result.postprocessed_field.l2_error(exact),
                       result.postprocessed_flux.l2_error(total_flux)))
    errors = np.asarray(errors)
    rates = np.log2(errors[:-1]/errors[1:])
    assert rates[-1, 0] > order+.7, (errors, rates)
    assert rates[-1, 1] > order+1.5, (errors, rates)
    assert rates[-1, 2] > order+.65, (errors, rates)
