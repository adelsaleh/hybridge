"""Tiny CPU diagnostics only: no assembly, GPU, compilation, or time integration."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hdgfem.assembly.face_dense import face_dense_matvec
from hdgfem.linalg.bsr import face_dense_to_bsr
from scripts.advection_diffusion_reaction.diagnostics import check_cached_adr_pardiso as diagnostic


@pytest.fixture
def cached_system(tmp_path):
    dense = np.array([[5., .2, -1., 0.], [-.1, 4., 0., .3],
                      [.5, 0., 6., -.4], [0., -.6, .1, 3.]])
    neighbors = np.array([[1, 0, -1], [1, 0, -1]])
    blocks = np.zeros((2, 3, 2, 2))
    for row in range(2):
        for slot in range(2):
            col = neighbors[row, slot]
            blocks[row, slot] = dense[2*row:2*row+2, 2*col:2*col+2]
    known = np.array([.2, -1., 2., .5])
    rhs = (dense @ known).reshape(2, 2)
    cache = tmp_path/'cache'
    cache.mkdir()
    for name, values in dict(blocks=blocks, neighbors=neighbors, rhs=rhs).items():
        np.save(cache/f'system_{name}.npy', values, allow_pickle=False)
    spec = tmp_path/'spec.json'
    spec.write_text(json.dumps(dict(case='tiny_nonsymmetric', cache=str(cache), result=str(tmp_path/'assembly.json'))))
    return spec, blocks, neighbors, dense, known


def test_bsr_conversion_matches_original_matvec(cached_system):
    _, blocks, neighbors, dense, known = cached_system
    matrix = face_dense_to_bsr(blocks, neighbors)
    np.testing.assert_array_equal(matrix.toarray(), dense)
    np.testing.assert_allclose(matrix @ known, face_dense_matvec(blocks, neighbors, known))
    assert matrix.has_canonical_format
    assert not np.shares_memory(matrix.data, blocks)


def test_bsr_conversion_sums_duplicate_slots(cached_system):
    _, blocks, neighbors, _, _ = cached_system
    neighbors = neighbors.copy()
    neighbors[:, 2] = neighbors[:, 0]
    blocks = blocks.copy()
    blocks[:, 2] = .5*blocks[:, 0]
    x = np.arange(4.)
    np.testing.assert_allclose(face_dense_to_bsr(blocks, neighbors) @ x,
                               face_dense_matvec(blocks, neighbors, x))


@pytest.mark.parametrize('neighbors', [np.array([[0., 1.]]), np.array([[0, 2]]), np.array([[0, -2]])])
def test_bsr_conversion_rejects_bad_topology(neighbors):
    with pytest.raises(ValueError):
        face_dense_to_bsr(np.zeros((1, 2, 2, 2)), neighbors)


def test_plan_is_readonly_and_bounds_size(cached_system, tmp_path):
    spec, *_ = cached_system
    output = tmp_path/'diagnostic'
    assert diagnostic.main(['--spec', str(spec), '--output', str(output), '--threads', '1']) == 0
    assert not output.exists()
    with pytest.raises(ValueError, match='max-dofs'):
        diagnostic.inspect_cache(spec, 3)


def test_cache_hash_mismatch_refused(cached_system, tmp_path):
    pytest.importorskip('pypardiso')
    spec, *_ = cached_system
    content = json.loads(spec.read_text())
    content['expected_operator_sha256'] = 'invalid'
    spec.write_text(json.dumps(content))
    output = tmp_path/'hash_failure'
    output.mkdir()
    args = diagnostic.parser().parse_args(['--spec', str(spec), '--output', str(output)])
    with pytest.raises(ValueError, match='recorded campaign hash'):
        diagnostic.solve_cached(args, 1, output)


def test_real_backend_reuses_existing_wrapper(cached_system, tmp_path, monkeypatch):
    pardiso = pytest.importorskip('pypardiso')
    import hdgfem.linalg.system as backend
    spec, _, _, _, known = cached_system
    output = tmp_path/'direct'
    output.mkdir()
    calls = []
    original = backend.solve_pypardiso_system

    def wrapped(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(backend, 'solve_pypardiso_system', wrapped)
    args = diagnostic.parser().parse_args(['--spec', str(spec), '--output', str(output)])
    threads = int(pardiso.ps.libmkl.MKL_Get_Max_Threads())
    assert diagnostic.solve_cached(args, threads, output) == 0
    assert len(calls) == 2
    assert all(c['matrix_type'] == 'nonsymmetric' for c in calls)
    np.testing.assert_allclose(np.load(output/'eliminated_solution.npy'), known, rtol=1e-12, atol=1e-12)
    result = json.loads((output/'result.json').read_text())
    assert result['status'] == 'passed'
    assert result['face_relative_residual'] < 1e-12
    assert result['probe_relative_solution_error'] < 1e-12
    assert 'analysis_factorization' in result['timings_ms']
    assert not (Path(json.loads(spec.read_text())['cache'])/'reference_trace.npy').exists()


def test_child_execution_sets_threads_and_saves_summary(cached_system, tmp_path):
    pytest.importorskip('pypardiso')
    spec, *_ = cached_system
    output = tmp_path/'isolated'
    assert diagnostic.main(['--spec', str(spec), '--output', str(output), '--threads', '1', '--execute']) == 0
    summary = json.loads((output/'summary.json').read_text())
    assert summary['best_measured_threads'] == 1
    assert summary['results'][0]['mkl_max_threads'] == 1
    assert summary['results'][0]['returncode'] == 0


def test_rejects_empty_scalar_row(cached_system, tmp_path):
    pytest.importorskip('pypardiso')
    spec, blocks, *_ = cached_system
    blocks[0, :, 0] = 0
    cache = Path(json.loads(spec.read_text())['cache'])
    np.save(cache/'system_blocks.npy', blocks)
    output = tmp_path/'singular'
    output.mkdir()
    args = diagnostic.parser().parse_args(['--spec', str(spec), '--output', str(output)])
    with pytest.raises(ValueError, match='empty scalar row'):
        diagnostic.solve_cached(args, 1, output)
