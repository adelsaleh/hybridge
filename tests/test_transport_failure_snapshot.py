"""Matrix/face audits and snapshot IO only; no GPU, compilation or integration."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from hdgfem.transport.diagnostics import (
    analyze_transport_snapshot,
    save_transport_failure_snapshot,
    trace_inflow_diagnostics,
)
from hdgfem.linalg.failure_snapshot import trace_matrix_diagnostics


@pytest.mark.parametrize('fmt', ['bsr', 'csr', 'coo'])
def test_matrix_audit_catches_zero_columns_with_nonzero_rows(fmt):
    dense = np.array([[1., 0., 2., 0.], [3., 0., 0., 1.],
                      [2., 0., 4., 1.], [1., 0., 1., 3.]])
    matrix = sparse.csr_matrix(dense)
    if fmt == 'bsr':
        matrix = matrix.tobsr(blocksize=(2, 2))
    elif fmt == 'coo':
        matrix = matrix.tocoo()
    arrays = {'matrix_format': fmt, 'data': matrix.data, 'rhs': np.ones(4)}
    if fmt == 'coo':
        arrays.update(rows=matrix.row, cols=matrix.col)
    else:
        arrays.update(indptr=matrix.indptr, indices=matrix.indices)
    report = trace_matrix_diagnostics(arrays)
    assert report['matrix_zero_rows'] == 0
    assert report['matrix_zero_columns'] == 1
    assert report['matrix_zero_column_samples'] == [1]
    assert report['matrix_row_l1_max'] == np.abs(dense).sum(axis=1).max()
    assert report['matrix_column_l1_max'] == np.abs(dense).sum(axis=0).max()


def face_audit(left, right, p=6):
    points, weights = np.polynomial.legendre.leggauss(len(left))
    basis = np.polynomial.legendre.legvander(points, p).T
    # Right samples arrive reversed in local orientation, just as on a mesh.
    normal = np.array([[left], [right[::-1]]])
    return trace_inflow_diagnostics(normal, np.array([[5], [5]]),
        np.array([[True], [False]]), np.array([5]), basis, weights)


def test_opposite_normal_traces_have_full_rank_inflow_coupling():
    speed = np.linspace(0.1, 0.7, 13)
    report = face_audit(speed, -speed)
    assert report['no_inflow_faces'] == 0
    assert report['numerically_rank_deficient_faces'] == 0
    assert report['worst_faces'][0]['sampling_rank'] == 7
    np.testing.assert_allclose(report['worst_faces'][0]['outward_normal_samples'][1], -speed)


def test_arbitrarily_small_double_outflow_disconnects_all_face_modes():
    speed = np.full(13, 1e-12)
    report = face_audit(speed, speed)
    assert report['double_outflow_faces'] == 1
    assert report['no_inflow_faces'] == 1
    assert report['nullity_lower_bound_from_inflow_nodes'] == 7


def test_tangent_face_is_reported_separately_from_double_outflow():
    report = face_audit(np.zeros(13), np.zeros(13))
    assert report['double_outflow_faces'] == 0
    assert report['no_inflow_faces'] == 1


def test_partial_inflow_can_leave_high_order_modes_disconnected():
    left = np.ones(13)
    left[:3] = -1
    report = face_audit(left, np.ones(13))
    assert report['no_inflow_faces'] == 0
    assert report['insufficient_inflow_node_faces'] == 1
    assert report['nullity_lower_bound_from_inflow_nodes'] == 4
    assert report['worst_faces'][0]['sampling_rank'] == 3


def test_face_rank_is_independent_of_time_step_scaling():
    left = np.linspace(-0.5, 1, 13)
    right = -left + 0.01
    a = face_audit(left, right)
    b = face_audit(left*0.0333333333, right*0.0333333333)
    assert a['worst_faces'][0]['sampling_rcond'] == pytest.approx(b['worst_faces'][0]['sampling_rcond'])


def test_snapshot_roundtrips_matrix_and_iterates_without_solver(tmp_path):
    matrix = sparse.csr_matrix(np.array([[2., 0.], [1., 0.]]))
    assembly = SimpleNamespace(matrix_format='csr', data=matrix.data,
        indptr=matrix.indptr, indices=matrix.indices, rhs=np.array([1., 2.]), raw=None)
    initial = np.array([0.1, 0.2])
    best = np.array([0.3, 0.4])
    before = matrix.data.copy()
    path = tmp_path/'failed.npz'
    report = save_transport_failure_snapshot(path, assembly, initial_guess=initial, best_solution=best)
    with np.load(path, allow_pickle=False) as saved:
        assert analyze_transport_snapshot(saved)['matrix_diagnostics'] == report['matrix_diagnostics']
        np.testing.assert_array_equal(saved['initial_guess'], initial)
        np.testing.assert_array_equal(saved['best_solution'], best)
        assert str(saved['matrix_format']) == 'csr'
    np.testing.assert_array_equal(matrix.data, before)
    assert report['system_snapshot_bytes'] == path.stat().st_size
    assert list(tmp_path.iterdir()) == [path]


def test_snapshot_survives_analysis_failure(tmp_path, monkeypatch):
    import hdgfem.transport.diagnostics as diagnostics
    def fail(_):
        raise RuntimeError('injected analysis failure')
    monkeypatch.setattr(diagnostics, 'analyze_transport_snapshot', fail)
    assembly = SimpleNamespace(matrix_format='csr', data=np.ones(1),
        indptr=np.array([0, 1]), indices=np.array([0]), rhs=np.ones(1), raw=None)
    path = tmp_path/'failed.npz'
    report = save_transport_failure_snapshot(path, assembly)
    assert path.exists()
    assert report['snapshot_analysis_error'] == 'RuntimeError: injected analysis failure'


def test_coo_audit_sums_duplicates_before_testing_for_zero_columns():
    arrays = {'matrix_format': 'coo', 'rhs': np.ones(2),
              'data': np.array([1., -1., 2., 3.]),
              'rows': np.array([0, 0, 0, 1]), 'cols': np.array([0, 0, 1, 1])}
    report = trace_matrix_diagnostics(arrays)
    assert report['matrix_zero_rows'] == 0
    assert report['matrix_zero_column_samples'] == [0]
    assert report['matrix_row_l1_max'] == 3
    assert report['matrix_column_l1_max'] == 5


def test_snapshot_detects_disconnected_mode_with_no_zero_rows_or_columns():
    points, weights = np.polynomial.legendre.leggauss(13)
    basis = np.polynomial.legendre.legvander(points, 6).T
    left = np.ones(13)
    left[[0, 3, 4, 7, 8, 9]] = -1
    right = np.ones(13)
    gamma = np.abs(left) - left
    weighted = basis.T * np.sqrt(weights * gamma)[:, None]
    # This Gram matrix couples each coordinate, but misses one combination.
    matrix = weighted.T @ weighted
    arrays = {'matrix_format': 'bsr', 'rhs': np.ones(7), 'data': matrix[None],
              'indptr': np.array([0, 1]), 'indices': np.array([0]),
              'normal_flux': np.array([[left], [right[::-1]]]),
              'edge_ids': np.array([[5], [5]]), 'orientations': np.array([[True], [False]]),
              'interior_edges': np.array([5]), 'trace_basis': basis, 'trace_weights': weights}
    report = analyze_transport_snapshot(arrays)
    assert report['matrix_diagnostics']['matrix_zero_rows'] == 0
    assert report['matrix_diagnostics']['matrix_zero_columns'] == 0
    assert report['trace_inflow_diagnostics']['nullity_lower_bound_from_inflow_nodes'] == 1
    check = report['deficient_face_matrix_checks'][0]
    assert check['column_panel_rank'] == 6
    assert check['matrix_mode_relative_residual'] < 1e-14


@pytest.mark.parametrize('snapshot_fails', [False, True])
def test_failed_stage_snapshot_preserves_original_solve_error(tmp_path, monkeypatch, snapshot_fails):
    import json
    from hdgfem.linalg.results import LinearSolveConvergenceError
    from scripts.guiding_center.runtime import runner

    error = LinearSolveConvergenceError('original solve failure')
    expected = tmp_path / 'failure_system.npz'

    def save_snapshot(path):
        assert path == expected
        if snapshot_fails:
            raise OSError('snapshot write failed')
        np.savez(path, data=np.eye(2))
        return {'system_snapshot': str(path), 'trace_inflow_diagnostics': {'no_inflow_faces': 1}}

    def fail(**kwargs):
        raise error

    error.save_transport_snapshot = save_snapshot
    monkeypatch.setattr(runner, 'transport_velocity_diagnostics', lambda _: {})
    path = tmp_path / 'failure.json'
    with pytest.raises(LinearSolveConvergenceError) as caught:
        runner._solve_transport_stage(
            SimpleNamespace(solve=fail), initial_guess=None, beta=None, step=208,
            time_value=10.4, stage='bdf2', beta_scale=1/30, failure_path=path,
        )
    assert caught.value is error
    report = json.loads(path.read_text())
    if snapshot_fails:
        assert report['snapshot_error'] == 'OSError: snapshot write failed'
    else:
        assert expected.exists()
        assert report['system_snapshot'] == str(expected)
        assert report['trace_inflow_diagnostics']['no_inflow_faces'] == 1
