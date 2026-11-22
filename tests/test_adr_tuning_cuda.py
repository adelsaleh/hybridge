"""GPU qualification of measured ADR execution and mass representation reuse."""
import numpy as np
import pytest

from hybridge import DGSpace, rectangle_mesh
from hybridge.mixed.adr_preparation import prepare_adr_data
from hybridge.mixed.raw_cuda.adr_operator import (
    assemble_projected_adr_trace_operator_raw_cuda as assemble,
    reconstruct_projected_adr_local_unknowns_raw_cuda as reconstruct,
)
from hybridge.mixed.raw_cuda import tensor_tuning as tuning
from hybridge.runtime.cuda_tuning import clear_tuning_cache
from test_adr_tensor_raw_cuda import cp, dense_device, velocity
from test_adr_tensor_numba import diffusion_cases


@pytest.mark.parametrize('device_prepared', [False, True])
@pytest.mark.parametrize('diffusion_kind', ['constant', 'variable-full'])
def test_cold_adr_calibration_keeps_sample_arrays_on_device(cp, monkeypatch, tmp_path,
                                                           device_prepared, diffusion_kind):
    """A cache miss validates actual equations without downloading coefficients or indices."""
    from hybridge.mixed.coefficients_device import prepare_adr_data_cupy
    from scripts.advection_diffusion_reaction.cases.tensor_cases import raw_cuda_coefficient

    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING', 'auto')
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING_CANDIDATES', '2')
    clear_tuning_cache()
    space = DGSpace(rectangle_mesh(2, 1), 1, basis_type='dub_orth')
    trace = space.trace_space('legendre-modal')
    diffusion = 2. if diffusion_kind == 'constant' else raw_cuda_coefficient(diffusion_kind)
    prepare = prepare_adr_data_cupy if device_prepared else prepare_adr_data
    prepared = prepare(lambda x, y: 1.+x*y, .4, velocity(space), space,
                       diffusion=diffusion, trace_space=trace)
    common = dict(diffusion=diffusion, trace_space=trace, matrix_format='bsr')
    reference = assemble(prepared, .2, space, block_size=64, **common)
    trace_values = cp.sin(cp.arange(space.mesh.num_edg * trace.edg_dof, dtype=reference.assembly.rhs.dtype))
    original = cp.asnumpy

    def forbid_arrays(array, *args, **kwargs):
        """Permit scalar diagnostics while rejecting every array download."""
        if array.size > 1:
            raise AssertionError(f'ADR calibration downloaded {array.size} values')
        return original(array, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(cp, 'asnumpy', forbid_arrays)
        automatic = assemble(prepared, .2, space, **common)
        actual = reconstruct(automatic, trace_values)[0]
    record = automatic.reconstruction_data['tuning']
    assert record['source'] == 'measured' and record['sample_elements'] > 0
    assert 'rejected' not in record['candidates'][0]
    cp.testing.assert_allclose(automatic.assembly.data, reference.assembly.data, rtol=3e-10, atol=3e-10)
    cp.testing.assert_allclose(automatic.assembly.rhs, reference.assembly.rhs, rtol=3e-10, atol=3e-10)
    cp.testing.assert_allclose(actual, reconstruct(reference, trace_values)[0], rtol=3e-10, atol=3e-10)


@pytest.mark.parametrize('diffusion,kind', diffusion_cases())
@pytest.mark.parametrize('cache', ['none', 'schur-lu', 'schur-lu+mass'])
def test_measured_adr_all_diffusion_classes_and_cache_policies(cp, monkeypatch, tmp_path,
                                                             diffusion, kind, cache):
    """Sample calibration, production assembly and reconstruction agree for all classes."""
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING', 'auto')
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING_CANDIDATES', '3')
    clear_tuning_cache()
    space = DGSpace(rectangle_mesh(2, 1), 3, basis_type='dub_orth')
    trace = space.trace_space('legendre-modal')
    prepared = prepare_adr_data(lambda x, y: 1.+x*y, .4, velocity(space), space,
        diffusion=diffusion, trace_space=trace, dense_local_matrices=False)
    common = dict(diffusion=diffusion, trace_space=trace, cache_local_factors=cache, matrix_format='bsr')
    automatic = assemble(prepared, .2, space, **common)
    reference = assemble(prepared, .2, space, block_size=64, **common)
    record = automatic.reconstruction_data['tuning']
    assert record['source'] == 'measured' and 0 < record['sample_elements'] <= tuning.SAMPLE_ELEMENTS
    assert len(record['candidates']) <= 3
    assert 'rejected' not in record['candidates'][0]
    np.testing.assert_allclose(dense_device(cp, automatic.assembly), dense_device(cp, reference.assembly),
                               rtol=3e-10, atol=3e-10)
    np.testing.assert_allclose(cp.asnumpy(automatic.assembly.rhs), cp.asnumpy(reference.assembly.rhs),
                               rtol=3e-10, atol=3e-10)
    values = cp.sin(cp.arange(space.mesh.num_edg*trace.edg_dof, dtype=automatic.assembly.rhs.dtype))
    np.testing.assert_allclose(cp.asnumpy(reconstruct(automatic, values)[0]),
        cp.asnumpy(reconstruct(reference, values)[0]), rtol=3e-10, atol=3e-10)
    clear_tuning_cache()
    def forbid_samples(*args, **kwargs):
        """A persistent cache hit must not allocate or launch trial work."""
        raise AssertionError('cache hit attempted ADR calibration')
    monkeypatch.setattr(tuning, 'sample_element_indices', forbid_samples)
    hit = assemble(prepared, .2, space, **common)
    assert hit.reconstruction_data['tuning']['source'] == 'disk'
    assert hit.assembly.timings['raw.tuning.sample_elements'] == 0


@pytest.mark.parametrize('order', [7, 9])
def test_measured_adr_high_order_and_mass_representation_changes(cp, monkeypatch, tmp_path, order):
    """LU/inverse overrides safely rebuild caches, while automatic reuse preserves representation."""
    from scripts.advection_diffusion_reaction.cases.tensor_cases import raw_cuda_coefficient
    from test_adr_high_order_raw_cuda import cupy_reference

    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING', 'auto')
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING_CANDIDATES', '8')
    # Qualification can explicitly spend longer compiling all layout candidates;
    # the production first-use default remains 30 seconds/eight candidates.
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING_SECONDS', '180')
    clear_tuning_cache()
    space = DGSpace(rectangle_mesh(2, 1), order, basis_type='dub_orth')
    trace = space.trace_space('legendre-modal')
    diffusion = raw_cuda_coefficient('variable-full')
    prepared = prepare_adr_data(lambda x, y: 1.+x*y, .4, velocity(space), space,
        diffusion=diffusion, trace_space=trace, dense_local_matrices=False)
    common = dict(diffusion=diffusion, trace_space=trace, matrix_format='bsr')
    values = np.sin(np.arange(space.mesh.num_edg*trace.edg_dof)+.17)
    expected, rhs, local = [cp.asnumpy(value) for value in
        cupy_reference(cp, prepared, diffusion, space, trace, .2, values)]
    monkeypatch.setenv('HYBRIDGE_RAW_ADR_MASS_INVERSE', 'off')
    factorized = assemble(prepared, .2, space, **common)
    assert not factorized.mass_factors.mass_inverse
    monkeypatch.setenv('HYBRIDGE_RAW_ADR_MASS_INVERSE', 'on')
    inverted = assemble(prepared, .2, space, mass_factors=factorized.mass_factors, **common)
    assert inverted.mass_factors.mass_inverse
    assert inverted.assembly.timings['raw.mass_factors.rebuilt'] == 1
    for operator in (factorized, inverted):
        np.testing.assert_allclose(dense_device(cp, operator.assembly), expected, rtol=5e-10, atol=5e-10)
        np.testing.assert_allclose(cp.asnumpy(operator.assembly.rhs), rhs, rtol=5e-10, atol=5e-10)
        np.testing.assert_allclose(cp.asnumpy(reconstruct(operator, values)[0]), local, rtol=5e-10, atol=5e-10)
    monkeypatch.setenv('HYBRIDGE_RAW_ADR_MASS_INVERSE', 'auto')
    saved = [array.copy() for array in inverted.mass_factors]
    reused = assemble(prepared, .2, space, mass_factors=inverted.mass_factors, **common)
    assert reused.mass_factors.mass_inverse and reused.assembly.timings['raw.mass_factors.cached'] == 1
    for actual, original in zip(inverted.mass_factors, saved):
        cp.testing.assert_array_equal(actual, original)
    legacy = assemble(prepared, .2, space, mass_factors=tuple(inverted.mass_factors), **common)
    assert legacy.assembly.timings['raw.mass_factors.rebuilt'] == 1
    np.testing.assert_allclose(cp.asnumpy(reconstruct(legacy, values)[0]), local, rtol=5e-10, atol=5e-10)


def test_explicit_adr_block_skips_measured_selection(cp, monkeypatch):
    """An explicit launch continues to select its requested deterministic path."""
    def forbid_selector(**kwargs):
        """Fail if an explicitly selected block reaches calibration."""
        raise AssertionError('explicit block unexpectedly calibrated')
    monkeypatch.setattr(tuning, 'select_tensor_execution', forbid_selector)
    space = DGSpace(rectangle_mesh(1, 1), 2, basis_type='dub_orth')
    trace = space.trace_space('legendre-modal')
    prepared = prepare_adr_data(space.constant(1.), .4, velocity(space), space,
                                trace_space=trace, dense_local_matrices=False)
    operator = assemble(prepared, .2, space, trace_space=trace, block_size=32)
    assert operator.assembly.timings['raw.block_size'] == 32
    assert operator.reconstruction_data['tuning'] is None


def test_mixed_rare_diffusion_class_is_sampled_and_matches_production(cp, monkeypatch, tmp_path):
    """Non-prefix sample indices retain rare variable tensors among constant elements."""
    from hybridge.mixed.coefficients import prepare_diffusion
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING', 'auto')
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING_CANDIDATES', '3')
    clear_tuning_cache()
    space = DGSpace(rectangle_mesh(65, 3), 3, basis_type='dub_orth')
    trace = space.trace_space('legendre-modal')
    diffusion = (lambda x, y: 2.+np.where(x > .99, .1*x, 0.),
                 lambda x, y: np.where(x > .99, .1+.02*y, 0.),
                 lambda x, y: np.where(x > .99, -.1+.01*x, 0.),
                 lambda x, y: 2.+np.where(x > .99, .05*y, 0.))
    tensor = prepare_diffusion(diffusion, space)
    present = np.unique(tensor.kinds)
    assert set(present) == {0, 6}
    kinds = cp.asarray(tensor.kinds)
    indices = tuning.sample_element_indices(cp, kinds, elements=space.mesh.num_tri,
        nel=space.el_dof, ntr=trace.edg_dof, present_kinds=present.tolist())
    assert set(cp.asnumpy(kinds[indices])) == {0, 6}
    assert indices.size <= tuning.SAMPLE_ELEMENTS < space.mesh.num_tri
    prepared = prepare_adr_data(lambda x, y: 1.+x*y, .4, velocity(space), space,
        diffusion=diffusion, trace_space=trace, dense_local_matrices=False)
    options = dict(diffusion=diffusion, trace_space=trace, matrix_format='bsr', prepared_diffusion=tensor)
    automatic, reference = assemble(prepared, .2, space, **options), assemble(prepared, .2, space, block_size=64, **options)
    cp.testing.assert_allclose(automatic.assembly.data, reference.assembly.data, rtol=3e-10, atol=3e-10)
    cp.testing.assert_allclose(automatic.assembly.rhs, reference.assembly.rhs, rtol=3e-10, atol=3e-10)


def test_repeated_assembly_resolves_selection_once_and_keeps_results(cp, monkeypatch, tmp_path):
    """Later assemblies skip the identity and device read-backs, also after a class flip."""
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING', 'auto')
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING_CACHE_DIR', str(tmp_path))
    monkeypatch.setenv('HYBRIDGE_CUDA_TUNING_CANDIDATES', '2')
    clear_tuning_cache()
    space = DGSpace(rectangle_mesh(2, 1), 2, basis_type='dub_orth')
    trace = space.trace_space('legendre-modal')
    common = dict(diffusion=2., trace_space=trace, matrix_format='bsr')

    def problem(sign):
        """Flip the sampled sign class of the velocity and the source between assemblies."""
        return prepare_adr_data(lambda x, y: sign*(1.+x*y), .4, velocity(space, .7*sign, -.2*sign), space,
                                diffusion=2., trace_space=trace)

    first = assemble(problem(1.), .2, space, **common)
    assert first.reconstruction_data['tuning']['source'] == 'measured'

    def forbid(*args, **kwargs):
        raise AssertionError('a repeated assembly rebuilt the selection')

    with monkeypatch.context() as patch:
        patch.setattr(tuning, '_measure_tensor_execution', forbid)
        same = assemble(problem(1.), .2, space, **common)
        flipped = assemble(problem(-1.), .2, space, **common)
    for operator in (same, flipped):
        record = operator.reconstruction_data['tuning']
        assert record['reused'] and record['source'] == 'resolved'
        assert record['wall_seconds'] == 0. and record['sample_elements'] == 0
        assert record['variant'] == first.reconstruction_data['tuning']['variant']
        assert operator.assembly.timings['raw.tuning.reused'] == 1.
    reference = assemble(problem(-1.), .2, space, block_size=64, **common)
    cp.testing.assert_allclose(flipped.assembly.data, reference.assembly.data, rtol=3e-10, atol=3e-10)
    cp.testing.assert_allclose(flipped.assembly.rhs, reference.assembly.rhs, rtol=3e-10, atol=3e-10)
    # A rebuilt space is a different structural key, so it selects again.
    rebuilt = DGSpace(rectangle_mesh(2, 1), 2, basis_type='dub_orth')
    other = rebuilt.trace_space('legendre-modal')
    again = assemble(prepare_adr_data(lambda x, y: 1.+x*y, .4, velocity(rebuilt), rebuilt, diffusion=2.,
                                      trace_space=other), .2, rebuilt, diffusion=2., trace_space=other,
                     matrix_format='bsr')
    assert again.reconstruction_data['tuning']['source'] != 'resolved'
    clear_tuning_cache()
