"""Host contracts for ADR algorithm selection, samples and retained mass data."""
from types import SimpleNamespace

import numpy as np
import pytest

from hybridge import DGSpace, rectangle_mesh
from hybridge.mixed.adr_preparation import prepare_adr_data
from hybridge.mixed.adr_numpy import local_solvers_numpy
from hybridge.mixed.coefficients import prepare_diffusion
from hybridge.mixed.raw_cuda import tensor as raw
from hybridge.mixed.raw_cuda import tensor_tuning as tuning
from hybridge.runtime.precision import REAL_DTYPE


def _numpy_cupy():
    """Use host arrays to exercise pure sample/oracle algebra without CUDA."""
    def forbidden_download(*args, **kwargs):
        """The oracle must gather in place instead of downloading indices."""
        raise AssertionError('ADR calibration downloaded an array')

    return SimpleNamespace(**{name: getattr(np, name) for name in dir(np) if not name.startswith('__')},
                           asnumpy=forbidden_download)


def _candidates(order=9, kind=6, limit=227*1024, **overrides):
    """Build the same bounded candidate roster as a normal automatic assembly."""
    nel = (order+1)*(order+2)//2
    kinds = np.array([kind])
    workspace = raw.tensor_execution_workspace(nel, kinds, (order+2)**2, 2*order+1,
                                               order=order, shared_limit=limit)
    return tuning.candidate_configurations(nel=nel, kinds=kinds, nq=(order+2)**2,
        nfq=2*order+1, order=order, block_size=128, workspace=workspace,
        shared_limit=limit, device=dict(major=9, maxThreadsPerBlock=1024), **overrides)


@pytest.mark.parametrize('limit', [48*1024, 99*1024, 227*1024])
@pytest.mark.parametrize('kind', range(7))
def test_adr_candidates_screen_actual_workspace(limit, kind):
    """Quadro/Blackwell/Hopper limits screen every layout without changing quadrature."""
    configs = _candidates(kind=kind, limit=limit)
    assert configs
    for name, config in configs.items():
        assert name == config.name
        assert config.workspace.shared_bytes <= limit
        assert 1 <= config.workspace.batch <= 8
        assert not (config.workspace.mass_inverse and config.workspace.external_mass)
        assert not config.workspace.mass_inverse or kind == 6


def test_adr_candidates_include_algorithms_layouts_batches_and_launches():
    """Historical wide/inverse choices compete with a factorized reference."""
    configs = list(_candidates().values())
    assert configs[0].threads == 128 and not configs[0].workspace.mass_inverse
    assert {config.workspace.mass_inverse for config in configs} == {False, True}
    assert {config.workspace.compact for config in configs} == {False, True}
    assert {config.workspace.external_mass for config in configs} == {False, True}
    assert len({config.workspace.batch for config in configs}) > 1
    assert {64, 128, 256, 512} <= {config.threads for config in configs}
    assert any(config.threads == 512 for config in configs[:8])


@pytest.mark.parametrize('inverse', [False, True])
def test_retained_mass_representation_constrains_every_candidate(inverse):
    """A reusable LU/inverse is never interpreted as the competing representation."""
    assert all(config.workspace.mass_inverse == inverse
               for config in _candidates(retained_inverse=inverse).values())


def test_explicit_adr_layout_and_inverse_controls():
    """Forced placements and inverse choices constrain automatic launch trials."""
    assert all(config.workspace.external_mass for config in _candidates(placement='external').values())
    assert not any(config.workspace.external_mass for config in _candidates(placement='shared').values())
    assert all(config.workspace.mass_inverse for config in _candidates(inverse_mode='on').values())
    assert not any(config.workspace.mass_inverse for config in _candidates(inverse_mode='off').values())
    assert all(config.workspace.external_mass and not config.workspace.mass_inverse
               for config in _candidates(placement='external', inverse_mode='on').values())


def test_mass_cache_tuple_compatibility_and_ambiguous_legacy_rebuild():
    """Keep old unpack/index calls and rebuild only ambiguous high-order kind-6 pairs."""
    mass, pivots = np.empty((4, 256), dtype=REAL_DTYPE), np.zeros((4, 16), dtype=np.int32)
    kwargs = dict(elements=4, nel=8, mass_stride=256, max_kind=6, order=7, signature='space')
    tagged = tuning.ADRMassFactors(mass, pivots, mass_inverse=True, signature='space')
    assert isinstance(tagged, tuple) and len(tagged) == 2
    assert tagged[0] is mass and tagged[1] is pivots
    assert tuning.reusable_mass_cache(tagged, **kwargs) is True
    assert tuning.reusable_mass_cache(tagged, **kwargs, requested_inverse=False) is None
    assert tuning.reusable_mass_cache(tuple(tagged), **kwargs) is None
    assert tuning.reusable_mass_cache(tuple(tagged), **dict(kwargs, order=6)) is False
    assert tuning.reusable_mass_cache(tagged, **dict(kwargs, signature='different')) is None


def test_mass_cache_rejects_shape_or_precision_mismatch():
    """Representation metadata cannot make incompatible buffers reusable."""
    cache = tuning.ADRMassFactors(np.empty((4, 256), dtype=np.float32),
        np.zeros((4, 16), dtype=np.int32), mass_inverse=False, signature='space')
    kwargs = dict(elements=4, nel=8, mass_stride=256, max_kind=6, order=7, signature='space')
    if np.dtype(REAL_DTYPE) != np.dtype(np.float32):
        assert tuning.reusable_mass_cache(cache, **kwargs) is None
    assert tuning.reusable_mass_cache(cache, **dict(kwargs, elements=3)) is None


def test_mass_cache_rejects_equal_size_different_geometry_and_space():
    """Same dimensions/reference data cannot authorize factors from another mesh."""
    from hybridge import DGMesh
    original = DGSpace(rectangle_mesh(2, 1), 3, basis_type='dub_orth')
    scaled_mesh = DGMesh.from_arrays(original.mesh.node_coords * 2., original.mesh.triangles)
    scaled = DGSpace(scaled_mesh, 3, basis_type='dub_orth')
    reordered_mesh = DGMesh.from_arrays(original.mesh.node_coords, original.mesh.triangles[::-1])
    reordered = DGSpace(reordered_mesh, 3, basis_type='dub_orth')
    separate_space = DGSpace(original.mesh, 3, basis_type='dub_orth')
    n, ne = original.el_dof, original.mesh.num_tri
    stride = raw.mass_factor_size(n, 6)
    signature = tuning.mass_cache_signature(original, original.trace_space('legendre-modal'),
                                             mass_stride=stride, max_kind=6)
    cache = tuning.ADRMassFactors(np.empty((ne, stride), dtype=REAL_DTYPE),
        np.zeros((ne, 2*n), dtype=np.int32), mass_inverse=False, signature=signature)
    for space in (original, scaled, reordered, separate_space):
        current = tuning.mass_cache_signature(space, space.trace_space('legendre-modal'),
                                               mass_stride=stride, max_kind=6)
        reusable = tuning.reusable_mass_cache(cache, elements=ne, nel=n, mass_stride=stride,
            max_kind=6, order=3, signature=current)
        assert (reusable is False) == (space is original)


def test_mass_cache_identity_does_not_retain_the_space():
    """Temporary calibration/assembly spaces are not kept alive by identity tokens."""
    import gc
    import weakref
    space = DGSpace(rectangle_mesh(1, 1), 2, basis_type='dub_orth')
    token = tuning._identity_token(space)
    assert tuning._identity_token(space) == token
    reference, key = weakref.ref(space), id(space)
    del space
    gc.collect()
    assert reference() is None and key not in tuning._IDENTITY_TOKENS


def test_sample_is_bounded_and_contains_rare_diffusion_classes():
    """A small rare class cannot disappear in an evenly spaced global prefix."""
    kinds = np.zeros(100_000, dtype=np.int64)
    kinds[[23, 155, 438, 781, 3601, 99893]] = np.arange(1, 7)
    indices = tuning.sample_element_indices(_numpy_cupy(), kinds, elements=kinds.size, nel=55, ntr=10)
    assert indices.size <= tuning.SAMPLE_ELEMENTS
    assert set(kinds[indices]) == set(range(7))
    assert len(np.unique(indices)) == indices.size


@pytest.mark.parametrize('diffusion', [2., (2., .2, -.1, 1.),
    (lambda x, y: 2.+.1*x, lambda x, y: .2+.02*y,
     lambda x, y: -.1+.03*x, lambda x, y: 1.+.05*y)])
@pytest.mark.parametrize('basis', ['legacy-lagrange', 'legendre-modal'])
def test_independent_sample_oracle_matches_numpy_local_inverse(diffusion, basis):
    """The sampled residual gate represents the existing full mixed equations."""
    space = DGSpace(rectangle_mesh(2, 1), 3, basis_type='dub_orth')
    trace = space.trace_space(basis)
    beta = (space*space).field((space.constant(.7), space.constant(-.2)))
    prepared = prepare_adr_data(lambda x, y: 1.+x*y, .4, beta, space,
        diffusion=diffusion, trace_space=trace)
    tensor = prepare_diffusion(diffusion, space)
    indices = np.array([0, 3])
    matrix, rhs = tuning._mixed_oracle(_numpy_cupy(), space=space, trace_space=trace,
                                      prepared=prepared, tensor=tensor, indices=indices)
    inverse = local_solvers_numpy(prepared, space, diffusion=diffusion)[indices]
    identity = np.broadcast_to(np.eye(matrix.shape[1]), matrix.shape)
    np.testing.assert_allclose(matrix @ inverse, identity, rtol=2e-11, atol=2e-11)
    assert rhs.shape == (2, 3*space.el_dof, 3*trace.edg_dof+1)
    np.testing.assert_allclose(rhs[:, :space.el_dof, -1], prepared.source_rhs[indices])


def test_sample_source_keeps_three_separate_entry_points():
    """Production ABI remains unchanged while private samples share its algebra."""
    source = raw.kernel_source(10, 4, 25, 7, 1, 'bsr', 8, 400, tuning=True,
                               cached_factors=True, mass_stride=400)
    assert source.count('void assemble_adr_tensor(') == 1
    assert source.count('void sample_adr_tensor(') == 1
    assert source.count('void sample_reconstruct_adr_tensor(') == 1
    assert '#define RAW_TUNING_SAMPLE 0' in source
    assert 'sample_elements[slot]' in source
