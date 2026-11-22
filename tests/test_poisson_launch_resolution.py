"""Poisson raw-CUDA launch resolution and warp solves: workspace arithmetic on the host, parity on the device."""
from types import SimpleNamespace

import numpy as np
import pytest

from hybridge.mixed.raw_cuda.poisson import _workspace_bytes, shared_bytes
from hybridge.runtime.cuda_tuning import KernelConfiguration
from hybridge.runtime.precision import REAL_ITEMSIZE


def _context(policy, nel=15, ntr=5):
    return SimpleNamespace(cspace=SimpleNamespace(el_dof=nel, edg_dof=ntr), policy=policy)


@pytest.mark.parametrize('config', [KernelConfiguration(64, 1, False), KernelConfiguration(256, 16, True)])
def test_workspace_drops_the_factor_copy_and_adds_the_warp_tables_for_cached_lu(config):
    nel, ntr = 15, 5
    full = shared_bytes(nel, ntr, config)
    without_factor = shared_bytes(nel, ntr, config, factor=False)
    assert _workspace_bytes(_context('schur-lu'), 0, config) == full
    for phase in (1, 2):
        assert _workspace_bytes(_context('schur-lu'), phase, config) == without_factor + nel*REAL_ITEMSIZE + nel*4
        assert _workspace_bytes(_context('none'), phase, config) == full
    assert _workspace_bytes(_context('schur-cholesky'), 1, config) == full
    assert _workspace_bytes(_context('schur-cholesky'), 2, config) == without_factor


@pytest.fixture(scope='module')
def cp():
    cupy = pytest.importorskip('cupy')
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('CUDA device unavailable')
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA runtime unavailable')
    return cupy


@pytest.mark.parametrize('order', [4, 8])
def test_resolved_launches_reproduce_the_selected_launch_and_sub_warp_blocks_agree(cp, order):
    from hybridge import DGMesh, DGSpace
    from hybridge.core.device import as_cupy_space
    from hybridge.hdg.condensation_device import reconstruct_trace_cupy
    from hybridge.mixed.cupy import assemble_projected_diffusion_trace_system_eliminated_raw_cupy, build_trace_reference
    from hybridge.mixed.raw_cuda.poisson import reconstruct, refresh_rhs
    mesh = DGMesh.from_arrays(np.array([[-1., -.9], [1.2, -.7], [-.7, 1.1], [.05, -.03]]),
                              np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64))
    space = DGSpace(mesh, order, basis_type='dub_orth')
    device = as_cupy_space(space)
    tr = build_trace_reference(device, 'legacy-lagrange')
    source = space.project_callable(lambda x, y: 1.+.2*x*y)
    assembled = assemble_projected_diffusion_trace_system_eliminated_raw_cupy(source, space.zeros(),
        lambda x, y: 1.+x-.3*y, 1.3, space, trace_basis='legacy-lagrange', matrix_format='csr', block_size=64,
        trace_ref=tr, cache_local_factors=True, local_factor_kind='schur-lu')
    r = assembled.raw_assembly
    common = dict(cspace=device, trace_ref=tr, d0_reference=r.d0_reference, d1_reference=r.d1_reference,
                  face_element_mass=r.face_element_mass, tau=1.3)
    first = refresh_rhs(source_rhs=r.source_rhs, boundary_trace=r.boundary_trace, csr_pattern=r.csr_pattern,
                        block_size=64, cached_factors=r, **common)
    second = refresh_rhs(source_rhs=r.source_rhs, boundary_trace=r.boundary_trace, csr_pattern=r.csr_pattern,
                         block_size=64, cached_factors=r, **common)
    assert first.timings['raw.poisson.autotune.wall'] > 0. or first.timings['raw.poisson.autotune.reused'] == 1.
    assert second.timings['raw.poisson.autotune.wall'] == 0. and second.timings['raw.kernel.jit'] == 0.
    cp.testing.assert_array_equal(first.rhs, second.rhs)
    import scipy.sparse as sparse
    size = int(assembled.rhs.size)
    matrix = sparse.csr_matrix((cp.asnumpy(assembled.data), cp.asnumpy(assembled.indices), cp.asnumpy(assembled.indptr)),
                               shape=(size, size)).toarray()
    trace = reconstruct_trace_cupy(cp.asarray(np.linalg.solve(matrix, cp.asnumpy(first.rhs))), r.boundary_trace, device)
    runs = [reconstruct(trace=trace, source_rhs=r.source_rhs, block_size=64, return_local_unknowns=True,
                        cached_factors=r, **common) for _ in range(2)]
    cp.testing.assert_array_equal(runs[0][1], runs[1][1])
    assert runs[1][2] >= 0. and r.poisson_context.resolved
    if order <= 6:
        # Block size 1 takes the sub-warp column solve of the same factors.
        _, serial, _ = reconstruct(trace=trace, source_rhs=r.source_rhs, block_size=1, return_local_unknowns=True,
                                   cached_factors=r, **common)
        cp.testing.assert_allclose(serial, runs[0][1], rtol=1e-11, atol=1e-12)


def test_workspace_of_lu_responses_keeps_the_rhs_solve_and_drops_the_reconstruction_tables():
    config, nel, ntr = KernelConfiguration(64, 1, False), 15, 5
    without_factor = shared_bytes(nel, ntr, config, factor=False)
    assert _workspace_bytes(_context('schur-lu+responses'), 0, config) == shared_bytes(nel, ntr, config)
    assert _workspace_bytes(_context('schur-lu+responses'), 1, config) == without_factor + nel*REAL_ITEMSIZE + nel*4
    assert _workspace_bytes(_context('schur-lu+responses'), 2, config) == without_factor


@pytest.mark.parametrize('order', [2, 4, 8, 9])
def test_lu_responses_reproduce_cached_lu_after_assembly_and_after_a_refreshed_source(cp, order):
    """Reconstruction uses the source solution of the last RHS: both must see the refreshed source."""
    import scipy.sparse as sparse
    from hybridge import DGMesh, DGSpace
    from hybridge.core.device import as_cupy_space
    from hybridge.hdg.condensation_device import reconstruct_trace_cupy
    from hybridge.mixed.cupy import assemble_projected_diffusion_trace_system_eliminated_raw_cupy, build_trace_reference
    from hybridge.mixed.raw_cuda.poisson import reconstruct, refresh_rhs
    mesh = DGMesh.from_arrays(np.array([[-1., -.9], [1.2, -.7], [-.7, 1.1], [.05, -.03]]),
                              np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64))
    space = DGSpace(mesh, order, basis_type='dub_orth')
    device = as_cupy_space(space)
    tr = build_trace_reference(device, 'legacy-lagrange')
    source = space.project_callable(lambda x, y: 1.+.2*x*y)

    def run(policy):
        assembled = assemble_projected_diffusion_trace_system_eliminated_raw_cupy(source, space.zeros(),
            lambda x, y: 1.+x-.3*y, 1.3, space, trace_basis='legacy-lagrange', matrix_format='csr', block_size=64,
            trace_ref=tr, cache_local_factors=True, local_factor_kind=policy)
        r = assembled.raw_assembly
        common = dict(cspace=device, trace_ref=tr, d0_reference=r.d0_reference, d1_reference=r.d1_reference,
                      face_element_mass=r.face_element_mass, tau=1.3)
        size = int(assembled.rhs.size)
        matrix = sparse.csr_matrix((cp.asnumpy(assembled.data), cp.asnumpy(assembled.indices), cp.asnumpy(assembled.indptr)),
                                   shape=(size, size)).toarray()

        def recover(right_hand_side, moments):
            trace = reconstruct_trace_cupy(cp.asarray(np.linalg.solve(matrix, cp.asnumpy(right_hand_side))), r.boundary_trace, device)
            return reconstruct(trace=trace, source_rhs=moments, block_size=64, return_local_unknowns=True,
                               cached_factors=r, **common)[1]
        after_assembly = recover(assembled.rhs, r.source_rhs)
        moments = 2.*r.source_rhs-.5
        refreshed = refresh_rhs(source_rhs=moments, boundary_trace=r.boundary_trace, csr_pattern=r.csr_pattern,
                                block_size=64, cached_factors=r, **common)
        return assembled.data, assembled.rhs, after_assembly, refreshed.rhs, recover(refreshed.rhs, moments), r

    plain, stored = run('schur-lu'), run('schur-lu+responses')
    for left, right in zip(plain[:5], stored[:5]):
        cp.testing.assert_allclose(right, left, rtol=2e-10, atol=2e-10)
    extra = int(mesh.num_tri)*int(device.el_dof)*(1+3*int(device.edg_dof))*REAL_ITEMSIZE
    assert stored[5].local_factor_bytes-plain[5].local_factor_bytes == extra
    assert stored[5].local_factor_kind == 'schur-lu+responses' and stored[5].schur_lu is not None


@pytest.mark.parametrize('policy', ['schur-lu', 'schur-lu+responses'])
@pytest.mark.parametrize('order', [2, 4, 8])
def test_experimental_local_inverse_reproduces_the_stored_lu_solves(cp, monkeypatch, order, policy):
    """Not production: HYBRIDGE_POISSON_LOCAL_FLAGS=INVERSE after assembly and after a refreshed source."""
    import scipy.sparse as sparse
    from hybridge import DGMesh, DGSpace
    from hybridge.core.device import as_cupy_space
    from hybridge.hdg.condensation_device import reconstruct_trace_cupy
    from hybridge.mixed.cupy import assemble_projected_diffusion_trace_system_eliminated_raw_cupy, build_trace_reference
    from hybridge.mixed.raw_cuda import poisson
    mesh = DGMesh.from_arrays(np.array([[-1., -.9], [1.2, -.7], [-.7, 1.1], [.05, -.03]]),
                              np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64))
    space = DGSpace(mesh, order, basis_type='dub_orth')
    device = as_cupy_space(space)
    tr = build_trace_reference(device, 'legacy-lagrange')
    source = space.project_callable(lambda x, y: 1.+.2*x*y)

    def run():
        assembled = assemble_projected_diffusion_trace_system_eliminated_raw_cupy(source, space.zeros(),
            lambda x, y: 1.+x-.3*y, 1.3, space, trace_basis='legacy-lagrange', matrix_format='csr', block_size=64,
            trace_ref=tr, cache_local_factors=True, local_factor_kind=policy)
        r = assembled.raw_assembly
        common = dict(cspace=device, trace_ref=tr, d0_reference=r.d0_reference, d1_reference=r.d1_reference,
                      face_element_mass=r.face_element_mass, tau=1.3)
        size = int(assembled.rhs.size)
        matrix = sparse.csr_matrix((cp.asnumpy(assembled.data), cp.asnumpy(assembled.indices), cp.asnumpy(assembled.indptr)),
                                   shape=(size, size)).toarray()

        def recover(right_hand_side, moments):
            trace = reconstruct_trace_cupy(cp.asarray(np.linalg.solve(matrix, cp.asnumpy(right_hand_side))), r.boundary_trace, device)
            return poisson.reconstruct(trace=trace, source_rhs=moments, block_size=64, return_local_unknowns=True,
                                       cached_factors=r, **common)[1]
        after_assembly = recover(assembled.rhs, r.source_rhs)
        moments = 2.*r.source_rhs-.5
        refreshed = poisson.refresh_rhs(source_rhs=moments, boundary_trace=r.boundary_trace, csr_pattern=r.csr_pattern,
                                        block_size=64, cached_factors=r, **common)
        return assembled.rhs, after_assembly, refreshed.rhs, recover(refreshed.rhs, moments), r

    plain = run()
    monkeypatch.setattr(poisson, 'LOCAL_FLAGS', frozenset({'INVERSE'}))
    inverse = run()
    for left, right in zip(plain[:4], inverse[:4]):
        cp.testing.assert_allclose(right, left, rtol=2e-10, atol=2e-10)
    # The inverses take the place of the factors: same stored size, and a recorded residual.
    assert inverse[4].poisson_context.local_factor_bytes == plain[4].poisson_context.local_factor_bytes
    assert 0. <= inverse[4].timings['raw.local_factors.inverse_residual'] < 1e-10
    assert 'raw.local_factors.inverse_residual' not in plain[4].timings


def test_experimental_switch_names_are_checked(monkeypatch):
    from hybridge.mixed.raw_cuda import poisson
    monkeypatch.setenv('HYBRIDGE_POISSON_LOCAL_FLAGS', 'INVERSE')
    assert poisson._experimental_names('POISSON_LOCAL_FLAGS', {'INVERSE'}) == ['INVERSE']
    monkeypatch.setenv('HYBRIDGE_POISSON_LOCAL_FLAGS', 'RESPONSES,INVERSE')
    with pytest.warns(UserWarning, match='schur-lu\\+responses'):
        assert poisson._experimental_names('POISSON_LOCAL_FLAGS', {'INVERSE'}) == ['INVERSE']
    monkeypatch.setenv('HYBRIDGE_POISSON_LOCAL_FLAGS', 'INVERS')
    with pytest.raises(ValueError, match='unknown experimental switch'):
        poisson._experimental_names('POISSON_LOCAL_FLAGS', {'INVERSE'})
    # The successes of the study are production: their old switches are dropped with a warning.
    monkeypatch.setenv('HYBRIDGE_POISSON_KERNEL_FLAGS', 'WARP_SOLVE,DIRECT_FACTORS,SKIP_INTERIOR,LOAD_WIDE=4,CONTRACT_Q')
    with pytest.warns(UserWarning, match='production now'):
        names = poisson._experimental_names('POISSON_KERNEL_FLAGS', poisson._KERNEL_SWITCHES)
    assert poisson._kernel_flags(names) == {'LOAD_WIDE': 4, 'CONTRACT_Q': 1}
    monkeypatch.setenv('HYBRIDGE_POISSON_HOST_FLAGS', 'signature,resolved,nocheck,boundary')
    with pytest.warns(UserWarning, match='production now'):
        assert poisson._experimental_names('POISSON_HOST_FLAGS', {'boundary'}) == ['boundary']
    monkeypatch.setenv('HYBRIDGE_POISSON_KERNEL_FLAGS', 'INVERSE')
    with pytest.raises(ValueError, match='HYBRIDGE_POISSON_LOCAL_FLAGS=INVERSE'):
        poisson._experimental_names('POISSON_KERNEL_FLAGS', poisson._KERNEL_SWITCHES)
    assert poisson._kernel_flags(['STOP=0']) == {}


def test_production_has_no_experimental_switch_set():
    from hybridge.mixed.raw_cuda import poisson
    import os
    if any(os.environ.get('HYBRIDGE_POISSON_'+name) for name in ('LOCAL_FLAGS', 'KERNEL_FLAGS', 'KERNEL_OPTIONS', 'HOST_FLAGS')):
        pytest.skip('an experimental Poisson switch is set in this environment')
    assert not poisson.LOCAL_FLAGS and not poisson.KERNEL_FLAGS and not poisson.COMPILE_OPTIONS and not poisson.HOST_FLAGS
    assert all(poisson._experiment(phase) == () for phase in (0, 1, 2))


def test_workspace_of_the_experimental_kernel_layouts(monkeypatch):
    from hybridge.mixed.raw_cuda import poisson
    config, nel, ntr = KernelConfiguration(64, 1, False), 15, 5
    full, without_factor = shared_bytes(nel, ntr, config), shared_bytes(nel, ntr, config, factor=False)
    warp_tables = nel*REAL_ITEMSIZE + nel*4
    monkeypatch.setattr(poisson, 'KERNEL_FLAGS', {'CONTRACT_COUPLING': 1})
    assert poisson._experiment(0) == () and poisson._experiment(1) == ((('CONTRACT_COUPLING', 1),), ())
    # Assembly never takes a switch; the RHS and reconstruction kernels add the five contracted tables.
    assert _workspace_bytes(_context('schur-lu'), 0, config) == full
    assert _workspace_bytes(_context('schur-lu'), 1, config) == without_factor + warp_tables + 5*nel*REAL_ITEMSIZE
    assert _workspace_bytes(_context('none'), 2, config) == full + 5*nel*REAL_ITEMSIZE
    monkeypatch.setattr(poisson, 'KERNEL_FLAGS', {'CHOLESKY_WARP': 1})
    # The Cholesky factor is read in place and its pivot reciprocals take NEL reals; LU kernels do not change.
    assert _workspace_bytes(_context('schur-cholesky'), 1, config) == without_factor + nel*REAL_ITEMSIZE
    assert _workspace_bytes(_context('schur-cholesky'), 2, config) == without_factor
    assert _workspace_bytes(_context('schur-lu'), 1, config) == without_factor + warp_tables
    monkeypatch.setattr(poisson, 'KERNEL_FLAGS', {'NO_RESPONSE': 1})
    # Without responses, reconstruction solves again: on warps for LU, on a shared factor copy for Cholesky.
    assert _workspace_bytes(_context('schur-lu+responses'), 2, config) == without_factor + warp_tables
    assert _workspace_bytes(_context('schur-cholesky'), 2, config) == full
    monkeypatch.setattr(poisson, 'KERNEL_FLAGS', {'NO_RESPONSE': 1, 'CHOLESKY_WARP': 1})
    assert _workspace_bytes(_context('schur-cholesky'), 2, config) == without_factor + nel*REAL_ITEMSIZE


@pytest.mark.parametrize('policy,flags', [
    ('schur-lu', {'CONTRACT_Q': 1}),
    ('schur-lu', {'CONTRACT_COUPLING': 1}),
    ('schur-lu+responses', {'NO_RESPONSE': 1}),
    ('schur-lu+responses', {'NO_RESPONSE': 1, 'CONTRACT_Q': 1, 'CONTRACT_COUPLING': 1}),
    ('schur-cholesky', {'CHOLESKY_WARP': 1}),
    ('schur-cholesky', {'LOAD_WIDE': 4}),
    ('schur-cholesky', {'NO_RESPONSE': 1}),
    ('schur-cholesky', {'NO_RESPONSE': 1, 'CHOLESKY_WARP': 1, 'CONTRACT_COUPLING': 1}),
])
@pytest.mark.parametrize('order', [2, 4, 8])
def test_experimental_kernel_switches_reproduce_the_production_kernels(cp, monkeypatch, order, policy, flags):
    """Not production: HYBRIDGE_POISSON_KERNEL_FLAGS on the RHS and reconstruction of one assembled operator."""
    import scipy.sparse as sparse
    from hybridge import DGMesh, DGSpace
    from hybridge.core.device import as_cupy_space
    from hybridge.hdg.condensation_device import reconstruct_trace_cupy
    from hybridge.mixed.cupy import assemble_projected_diffusion_trace_system_eliminated_raw_cupy, build_trace_reference
    from hybridge.mixed.raw_cuda import poisson
    mesh = DGMesh.from_arrays(np.array([[-1., -.9], [1.2, -.7], [-.7, 1.1], [.05, -.03]]),
                              np.array([[0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int64))
    space = DGSpace(mesh, order, basis_type='dub_orth')
    device = as_cupy_space(space)
    tr = build_trace_reference(device, 'legacy-lagrange')
    source = space.project_callable(lambda x, y: 1.+.2*x*y)
    assembled = assemble_projected_diffusion_trace_system_eliminated_raw_cupy(source, space.zeros(),
        lambda x, y: 1.+x-.3*y, 1.3, space, trace_basis='legacy-lagrange', matrix_format='csr', block_size=64,
        trace_ref=tr, cache_local_factors=True, local_factor_kind=policy)
    r = assembled.raw_assembly
    common = dict(cspace=device, trace_ref=tr, d0_reference=r.d0_reference, d1_reference=r.d1_reference,
                  face_element_mass=r.face_element_mass, tau=1.3)
    size = int(assembled.rhs.size)
    matrix = sparse.csr_matrix((cp.asnumpy(assembled.data), cp.asnumpy(assembled.indices), cp.asnumpy(assembled.indptr)),
                               shape=(size, size)).toarray()
    moments = 2.*r.source_rhs-.5

    def run():
        refreshed = poisson.refresh_rhs(source_rhs=moments, boundary_trace=r.boundary_trace, csr_pattern=r.csr_pattern,
                                        block_size=64, cached_factors=r, **common)
        trace = reconstruct_trace_cupy(cp.asarray(np.linalg.solve(matrix, cp.asnumpy(refreshed.rhs))), r.boundary_trace, device)
        local = poisson.reconstruct(trace=trace, source_rhs=moments, block_size=64, return_local_unknowns=True,
                                    cached_factors=r, **common)[1]
        return refreshed.rhs, local

    production = run()
    monkeypatch.setattr(poisson, 'KERNEL_FLAGS', dict(flags))
    experimental = run()
    for left, right in zip(production, experimental):
        cp.testing.assert_allclose(right, left, rtol=2e-10, atol=2e-10)
    # The switched launches were selected and compiled apart from the production ones.
    resolved = r.poisson_context.resolved
    assert {key[0] for key in resolved if key[3]} == {1, 2} and {key[0] for key in resolved if not key[3]} >= {1, 2}
    # The replay diagnostics write into an argument a solve does not allocate.
    monkeypatch.setattr(poisson, 'KERNEL_FLAGS', {'PHASE_CLOCK': 1})
    with pytest.raises(ValueError, match='kernel replay'):
        run()


def test_experimental_compile_options_and_boundary_memo_reproduce_production(cp, monkeypatch):
    """Not production: HYBRIDGE_POISSON_KERNEL_OPTIONS and HYBRIDGE_POISSON_HOST_FLAGS=boundary."""
    from hybridge import DGSpace, rectangle_mesh
    from hybridge.core.device import as_cupy_space
    from hybridge.mixed.cupy import (assemble_projected_diffusion_trace_system_eliminated_raw_cupy,
                                     boundary_trace_values_cupy, build_trace_reference)
    from hybridge.mixed.raw_cuda import poisson
    space = DGSpace(rectangle_mesh(3, 2), 4, basis_type='dub_orth')
    device = as_cupy_space(space)
    tr = build_trace_reference(device, 'legacy-lagrange')
    source = space.project_callable(lambda x, y: 1.+.2*x*y)

    def assemble():
        return assemble_projected_diffusion_trace_system_eliminated_raw_cupy(source, space.zeros(),
            lambda x, y: 1.+x-.3*y, 1.3, space, trace_basis='legacy-lagrange', matrix_format='csr', block_size=64,
            trace_ref=tr, cache_local_factors=True, local_factor_kind='schur-lu')

    production = assemble()
    monkeypatch.setattr(poisson, 'COMPILE_OPTIONS', ('--maxrregcount=80',))
    limited = assemble()
    cp.testing.assert_allclose(limited.data, production.data, rtol=2e-10, atol=2e-10)
    cp.testing.assert_allclose(limited.rhs, production.rhs, rtol=2e-10, atol=2e-10)
    monkeypatch.setattr(poisson, 'COMPILE_OPTIONS', ())

    plain = boundary_trace_values_cupy(lambda x, y: 1.+x-.3*y, device, tr)
    monkeypatch.setattr(poisson, 'HOST_FLAGS', frozenset({'boundary'}))
    first = boundary_trace_values_cupy(lambda x, y: 1.+x-.3*y, device, tr)
    assert ('boundary-points', tr.kind) in device._poisson_reference_cache
    second = boundary_trace_values_cupy(lambda x, y: 2.*(1.+x-.3*y), device, tr)
    cp.testing.assert_array_equal(first, plain)
    cp.testing.assert_allclose(second, 2.*plain, rtol=1e-14, atol=0.)


def test_launch_bookkeeping_runs_on_the_host_with_stand_ins(monkeypatch):
    """Run ``_launch`` with NumPy in place of CuPy and a recording stand-in for the kernel.

    This checks, without a device, what the launch does around the kernel: it keeps one selection
    per phase, format and experimental build, reuses it on the next call, gives a build with a
    switch its own selection and tuning identity, and refuses the two replay diagnostics.
    """
    from hybridge.mixed.raw_cuda import poisson
    nel, ntr, ne = 6, 3, 4
    mesh = SimpleNamespace(num_tri=ne, aff_mats=np.zeros((ne, 4)), aff_jacs=np.ones(ne), jacs_el_fc=np.ones((ne, 3)),
                           normals=np.zeros((ne, 6)), loc2glob_edge=np.zeros((ne, 3), dtype=np.int64),
                           loc2oriented_face_coupling=np.zeros((ne, 3), dtype=np.int64))
    cspace = SimpleNamespace(mesh=mesh, el_dof=nel, edg_dof=ntr, order=2,
                             host=SimpleNamespace(mesh=mesh, reference=SimpleNamespace(basis_type='dub_orth')),
                             quad_data=SimpleNamespace(Krf_w=np.ones(6)))
    reference = {key: np.zeros(1) for key in ('dd', 'fd', 'd', 'md', 'fm', 'lifts', 'mlifts', 'edge_mass')}
    context = poisson.PoissonContext(cspace, SimpleNamespace(kind='legendre-modal'), reference, 1., 'schur-lu',
                                     np.zeros((ne, nel, nel)), np.zeros((ne, nel), dtype=np.int32), np.zeros(1), np.zeros(1))
    launches, identities = [], []

    def kernel(grid, block, arguments, shared_mem=0):
        launches.append((grid, block, shared_mem))
        arguments[-1][:] = 0                                   # the status buffer: no failure
    kernel.attributes = {'num_regs': 64}

    def select_configuration(*, identity, candidates, shared_bytes, **unused):
        identities.append(identity)
        return candidates[0], kernel, dict(reused=False, wall_seconds=0.)
    monkeypatch.setattr(poisson, 'require_cupy', lambda: np)
    monkeypatch.setattr(poisson, 'device_signature', lambda cp: {'fp32_fp64_ratio': 2})
    monkeypatch.setattr(poisson, 'select_configuration', select_configuration)
    monkeypatch.setattr(poisson, 'event_seconds', lambda cp, call: (call(), 1e-3)[1])
    monkeypatch.setattr(poisson, 'KERNEL_FLAGS', {})
    arguments = dict(phase=1, fmt='csr', source_rhs=np.zeros((ne, nel)), trace=np.zeros(9), rhs=np.zeros(3))

    first = poisson._launch(context, **arguments)
    second = poisson._launch(context, **arguments)
    assert len(identities) == 1 and len(launches) == 2 and list(context.resolved) == [(1, 'csr', 'auto', ())]
    assert first['raw.poisson.autotune.reused'] == 0. and second['raw.poisson.autotune.reused'] == 1.
    assert 'experiment' not in identities[0] and 'local_flags' not in identities[0]

    monkeypatch.setattr(poisson, 'KERNEL_FLAGS', {'CONTRACT_Q': 1})
    poisson._launch(context, **arguments)
    assert len(identities) == 2 and identities[1]['experiment'] == [[('CONTRACT_Q', 1)], []]
    assert set(context.resolved) == {(1, 'csr', 'auto', ()), (1, 'csr', 'auto', ((('CONTRACT_Q', 1),), ()))}
    # Assembly never takes a kernel switch, so its selection is the default one.
    poisson._launch(context, **dict(arguments, phase=0))
    assert (0, 'csr', 'auto', ()) in context.resolved and 'experiment' not in identities[2]

    monkeypatch.setattr(poisson, 'KERNEL_FLAGS', {'STOP': 2})
    with pytest.raises(ValueError, match='kernel replay'):
        poisson._launch(context, **arguments)
