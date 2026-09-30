"""Small host/raw-CUDA factor-cache parity checks; no AMGX or time stepping.

Set HDGFEM_CUDA_CACHE_ONLY=1 to forbid new CUDA compilation while permitting
existing cached binaries. Missing cached kernels then fail explicitly.
"""
import os
import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix
from hdgfem import DGSpace, rectangle_mesh
from hdgfem.mixed.numba import (
    build_diffusion_schur_cache_numba,
    assemble_projected_diffusion_trace_system_eliminated_numba,
    assemble_projected_diffusion_trace_rhs_eliminated_numba,
    reconstruct_projected_diffusion_local_unknowns_numba,
)
from hdgfem.linalg import expand_known_dofs


@pytest.fixture
def cp(monkeypatch):
    module = pytest.importorskip('cupy')
    if module.cuda.runtime.getDeviceCount() == 0:
        pytest.skip('No CUDA device')
    if os.environ.get('HDGFEM_CUDA_CACHE_ONLY') == '1':
        from cupy.cuda import compiler
        def forbidden(*args, **kwargs):
            """Reject compiler cache misses without launching compilation."""
            raise RuntimeError('CUDA cache miss: new compilation is forbidden')
        monkeypatch.setattr(compiler, '_compile_using_nvrtc_no_warning', forbidden)
        monkeypatch.setattr(compiler, 'compile_using_nvcc', forbidden)
    return module


def boundary(x, y):
    """Nonzero affine data exercises orientation and boundary elimination."""
    return 0.3 + x - 0.5 * y


@pytest.mark.parametrize('order', [1, 3, 6])
@pytest.mark.parametrize('basis', ['legacy-lagrange', 'legendre-modal'])
@pytest.mark.parametrize('policy', ['none', 'schur-lu', 'schur-cholesky'])
def test_numba_cached_matches_raw_cuda(cp, order, basis, policy):
    from hdgfem.core.device import as_cupy_space
    from hdgfem.mixed.cupy import (
            assemble_projected_diffusion_trace_system_eliminated_raw_cupy,
            assemble_projected_diffusion_trace_rhs_eliminated_raw_cupy,
            build_trace_reference,
            face_element_mass,
            reference_derivative_mats,
            source_moments_cupy,
            build_scalar_schur_cholesky_cache_cupy,
        )
    from hdgfem.mixed.raw_cuda.identity import (
            reconstruct_projected_diffusion_field_raw_cuda,
        )
    space = DGSpace(rectangle_mesh(2, 1, xlim=(-1., 1.), ylim=(-0.4, 0.7)), order)
    source = space.project_callable(lambda x, y: 1. + x * y)
    reaction = space.zeros()
    cache = None if policy == 'none' else build_diffusion_schur_cache_numba(
        reaction, 1.3, space, factor_kind=policy)
    kwargs = dict(trace_space=space.trace_space(basis), cached_factors=cache)
    host = assemble_projected_diffusion_trace_system_eliminated_numba(
        source, reaction, boundary, 1.3, space, **kwargs)
    raw = assemble_projected_diffusion_trace_system_eliminated_raw_cupy(
        source, reaction, boundary, 1.3, space, trace_basis=basis, matrix_format='csr',
        block_size=128, cache_local_factors=policy == 'schur-lu')
    size = host.trace_system.rhs.size
    hs = host.trace_system
    hm = coo_matrix((hs.data, (hs.rows, hs.cols)), shape=(size, size)).toarray()
    rm = csr_matrix((cp.asnumpy(raw.data), cp.asnumpy(raw.indices), cp.asnumpy(raw.indptr)),
                    shape=(size, size)).toarray()
    np.testing.assert_allclose(hm, rm, rtol=2e-9, atol=2e-10)
    np.testing.assert_allclose(hs.rhs, cp.asnumpy(raw.rhs), rtol=2e-9, atol=2e-10)
    updated = space.project_callable(lambda x, y: 0.7 - x + y)
    hrhs, _, reduction, _ = assemble_projected_diffusion_trace_rhs_eliminated_numba(
        updated, reaction, boundary, 1.3, space, **kwargs)
    rrhs = assemble_projected_diffusion_trace_rhs_eliminated_raw_cupy(
        updated, reaction, boundary, 1.3, space, cached_raw=raw.raw_assembly,
        trace_basis=basis, block_size=128)
    np.testing.assert_allclose(hrhs, cp.asnumpy(rrhs.rhs), rtol=2e-9, atol=2e-10)
    solution = np.linalg.solve(hm, hrhs)
    assert np.linalg.norm(rm @ solution - cp.asnumpy(rrhs.rhs)) / np.linalg.norm(hrhs) < 1e-9
    trace = expand_known_dofs(solution, reduction)
    host_fields = reconstruct_projected_diffusion_local_unknowns_numba(
        trace, updated, reaction, 1.3, space, **kwargs)
    cspace = as_cupy_space(space)
    trace_ref = build_trace_reference(cspace, basis)
    dx, dy = reference_derivative_mats(cspace)
    _, raw_fields, _ = reconstruct_projected_diffusion_field_raw_cuda(
        trace=cp.asarray(trace), source_rhs=source_moments_cupy(updated, cspace),
        cspace=cspace, trace_ref=trace_ref, d0_reference=dx, d1_reference=dy,
        face_element_mass=face_element_mass(trace_ref), tau=1.3, block_size=128,
        return_local_unknowns=True,
        cached_factors=raw.raw_assembly if policy == 'schur-lu' else None)
    np.testing.assert_allclose(host_fields, cp.asnumpy(raw_fields), rtol=2e-8, atol=2e-9)
    if policy == 'schur-cholesky':
        device_cache = build_scalar_schur_cholesky_cache_cupy(reaction, cspace, trace_ref, 1.3)
        hfactor = np.tril(cache.factors)
        dfactor = cp.asnumpy(device_cache.factor)
        np.testing.assert_allclose(hfactor @ hfactor.transpose(0, 2, 1),
                                   dfactor @ dfactor.transpose(0, 2, 1), rtol=2e-9, atol=2e-10)
