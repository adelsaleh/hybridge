"""Select corrected DG upwinding without sampling, including raw device assembly."""
from contextlib import closing
import os

import numpy as np
import pytest

from hdgfem import DGSpace, rectangle_mesh
from hdgfem.hdg.stabilization import resolve_transport_stabilization, ScaledUpwind
from hdgfem.solvers import advection_reaction as ar


def problem(order=2):
    """Build discontinuous velocities with conflicting outward normal signs."""
    space = DGSpace(rectangle_mesh(1, 1), order, basis_type='dub_orth')
    coefficients = np.zeros((2, *space.shape))
    sides = space.mesh.edge_side_indices[space.mesh.int_edges_inds[0]]
    for side, speed in zip(sides, (.1, .2)):
        element, face = divmod(side, 3)
        coefficients[:, element, 0] = speed*space.mesh.normals[element, face]
    beta = space.vector_field([space.field(c) for c in coefficients])
    return space, beta, coefficients


def test_default_covers_vector_fields_components_and_coefficient_arrays():
    """Every supported DG representation selects the same corrected policy."""
    _, beta, coefficients = problem()
    for supplied in (beta, beta.components, list(beta.components), coefficients,
                     (coefficients[0], coefficients[1])):
        assert resolve_transport_stabilization(None, supplied) == 'conflict-averaged-upwind'
    continuous = (lambda x, y: x, lambda x, y: y)
    assert resolve_transport_stabilization(None, continuous) is None
    assert resolve_transport_stabilization(None, list(continuous)) is None
    assert resolve_transport_stabilization('upwind', beta) == ScaledUpwind(1.)
    tau = np.ones(3)
    assert resolve_transport_stabilization(tau, beta) is tau


@pytest.mark.parametrize('backend', ['numpy', 'numba', 'cupy', 'raw-cuda'])
@pytest.mark.parametrize('policy,corrected', [(None, True), ('upwind', False),
                                             ('conflict-averaged-upwind', True)])
def test_public_solver_resolves_policy_before_backend_dispatch(monkeypatch, backend, policy, corrected):
    """All public backends receive the effective choice before any assembly."""
    space, beta, _ = problem()
    class Captured(Exception):
        pass
    def validate(**kwargs):
        assert kwargs['advection_stabilization_is_conflict_averaged'] is corrected
        if policy == 'upwind':
            assert kwargs['advection_stabilization_is_scaled_upwind']
        raise Captured
    monkeypatch.setattr(ar, 'validate_advection_backend_configuration', validate)
    with pytest.raises(Captured):
        ar.solve_advection_reaction_hdg(space.constant(1.), beta, space.constant(3.),
            None, space, boundary_mode='zero-flux', assembly_backend=backend,
            advection_stabilization=policy)


@pytest.mark.skipif(os.environ.get('HDGFEM_RUN_CUDA_TRANSPORT_TESTS') != '1',
                   reason='requires an available CUDA device')
@pytest.mark.parametrize('entry', ['functional', 'tangent-bsr'])
def test_default_raw_p6_matrix_matches_explicit_corrected_policy(entry):
    """Both public raw assembly paths use the corrected kernel at p = 6."""
    cp = pytest.importorskip('cupy')
    space, beta, _ = problem(order=6)
    options = dict(solver='amgx', assembly_backend='raw-cuda', raw_local_assembly='fused',
                   boundary_mode='zero-flux', raw_matrix_format='coo', verbose=False)
    matrices = []
    for policy in (None, 'conflict-averaged-upwind'):
        with closing(ar.AdvectionReactionHDGSolver(space, source=space.constant(1.),
                     beta=beta, reaction=space.constant(3.), boundary_condition=None,
                     advection_stabilization=policy, **options)) as solver:
            if entry == 'functional':
                result = solver.assemble_trace_system()
                matrices.append((result.matrix_rows.copy(), result.matrix_cols.copy(),
                                 result.matrix_data.copy(), result.rhs.copy()))
            else:
                assembly = solver.assemble_tangent_boundary_raw_cuda_bsr()
                matrices.append(tuple(cp.asnumpy(a) for a in
                    (assembly.indptr, assembly.indices, assembly.data, assembly.rhs)))
    for actual, reference in zip(*matrices):
        np.testing.assert_array_equal(actual, reference)
