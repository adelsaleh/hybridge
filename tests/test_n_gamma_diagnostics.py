"""n-Gamma study diagnostics (TODO L98 E1): weighted errors, sampled minima, records."""
import csv
from dataclasses import dataclass
import json

import numpy as np
import pytest

from hybridge import DGSpace, rectangle_mesh
from scripts.n_gamma import diagnostics as nd


@pytest.fixture(scope='module')
def space():
    return DGSpace(rectangle_mesh(3, 2, xlim=(2., 4.), ylim=(-1., 1.)), 2, basis_type='dub_orth')


@pytest.mark.parametrize('geometry,key,squared', [
    ('axisymmetric', 'n_l2R', 40.),      # int (R Z)^2 R = int_2^4 R^3 * int Z^2 = 60 * 2/3
    ('cartesian', 'n_l2', 112./9.),      # int (x y)^2 = int_2^4 x^2 * int y^2 = 56/3 * 2/3
])
def test_error_norms_follow_geometry_exactly(space, geometry, key, squared):
    field = space.project_callable(lambda R, Z: R*Z)
    zero = lambda R, Z: 0.*R
    errors = nd.error_norms({'n': field, 'Gamma': field}, {'n': zero, 'Gamma': lambda R, Z: R*Z},
                            geometry=geometry, volume_degree=6, backend='host')
    np.testing.assert_allclose(errors[key], np.sqrt(squared), rtol=1e-13)
    assert errors[key.replace('n_', 'Gamma_')] < 1e-13
    with pytest.raises(ValueError, match='geometry'):
        nd.error_norm(field, zero, geometry='slab')


def test_sampled_minimum_uses_volume_and_face_points(space):
    field = space.project_callable(lambda R, Z: (R - 3.)**2 + Z)
    trace = space.trace_space('legendre-modal')
    minimum = nd.sampled_minimum(field, space, trace)
    assert minimum <= field.values_at_ref(space.quad_data.Krf_quads).min()
    np.testing.assert_allclose(minimum, -1., atol=1e-12)  # attained on the face Z=-1 at R=3


def test_convergence_rows_report_ratios_and_orders():
    rows = nd.convergence_rows('dt', [.02, .01, .005], {'n_l2R': [4e-4, 1e-4, 2.5e-5], 'Gamma_l2R': [1e-3, None, 1e-4]})
    assert rows[0]['n_l2R_order'] is None
    np.testing.assert_allclose([rows[1]['n_l2R_ratio'], rows[2]['n_l2R_order']], [4., 2.])
    assert rows[1]['Gamma_l2R_order'] is None and rows[2]['Gamma_l2R_order'] is None
    table = nd.markdown_table(rows, ['dt', 'n_l2R', 'n_l2R_order'])
    assert table.splitlines()[0] == '| dt | n_l2R | n_l2R_order |' and '| 0.01 | 0.0001 | 2 |' in table


def test_study_recorder_writes_steps_and_summary(tmp_path):
    @dataclass
    class Floor:
        min_extrapolated_density: float
        volume_clamps: int

    @dataclass
    class Step:
        step: int
        accepted: bool
        floor: Floor

    recorder = nd.StudyRecorder(tmp_path, 'transient_baseline')
    recorder.record_step(Step(2, True, Floor(1.75, 0)), case='transient_baseline', dt=.01)
    recorder.record_step(Step(3, False, Floor(float('nan'), 4)), case='transient_baseline', dt=.01)
    rows = nd.convergence_rows('dt', [.02, .01], {'n_l2R': [4e-4, 1e-4]})
    recorder.write_summary({'case': 'transient_baseline', 'rejected_steps': 1}, {'temporal': rows})
    with (tmp_path / 'transient_baseline_steps.csv').open() as handle:
        steps = list(csv.DictReader(handle))
    assert steps[1]['floor.volume_clamps'] == '4' and steps[1]['floor.min_extrapolated_density'] == ''
    summary = json.loads((tmp_path / 'transient_baseline_summary.json').read_text())
    assert summary['tables']['temporal'][1]['n_l2R_order'] == pytest.approx(2.)
    assert '## temporal' in (tmp_path / 'transient_baseline_summary.md').read_text()
