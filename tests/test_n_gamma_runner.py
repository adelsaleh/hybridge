"""n-Gamma D-BDF2 runner (TODO L98 F1): planning, dry run and bounded Numba smoke runs.

The smoke runs take at most two steps on coarse p=1 meshes; they check the
orchestration and records, not convergence.
"""
import csv
import json

import numpy as np
import pytest

from scripts.n_gamma import run_d_bdf2 as runner


def args_for(*extra):
    return runner.build_parser().parse_args(list(extra))


def test_geometry_is_required():
    with pytest.raises(SystemExit):
        runner.parse_arguments(['--dry-run'])
    assert runner.parse_arguments(['--preset', 'mms_xy_p6_device', '--dry-run']).geometry == 'cartesian'


def test_presets_share_the_p6_xy_case_and_accept_overrides(capsys):
    host = runner.parse_arguments(['--preset', 'mms_xy_p6_numba_pardiso'])
    device = runner.parse_arguments(['--preset', 'mms_xy_p6_device', '--order', '5', '--case', 'transient_stress'])
    for args in (host, device):
        assert args.geometry == 'cartesian' and args.final_postprocess == 'both' and args.volume_degree == 17
    assert host.order == 6 and len(host.case) == 4 and host.backend == 'numba'
    assert host.pardiso_threads == 'auto' and host.numba_threads == 'all' and host.pardiso_reuse_analysis
    assert device.order == 5 and device.case == ['transient_stress'] and device.backend == 'raw-cuda'
    assert device.amgx_reuse == 'preconditioner' and device.raw_matrix_format == 'bsr'
    with pytest.raises(SystemExit):
        runner.parse_arguments(['--preset', 'missing'])
    assert runner.main(['--list-presets']) == 0
    assert 'mms_xy_p6_numba_pardiso' in capsys.readouterr().out


def test_pardiso_thread_rule():
    available = runner.available_cpus()
    assert runner.pardiso_threads_for('auto', 10_000) == min(available, 8)
    assert runner.pardiso_threads_for('auto', 40_000) == min(available, 16)
    assert runner.pardiso_threads_for('all', 1) == available
    assert runner.pardiso_threads_for('3', 1) == min(3, available)


@pytest.mark.parametrize('geometry', ['cartesian', 'axisymmetric'])
def test_plan_follows_the_study_design(geometry):
    args = args_for('--geometry', geometry)
    stationary = runner.plan_runs('stationary_stress', args)
    assert [(s.h, s.outer_vertices, s.hole_vertices) for s in stationary['spatial']] == [
        (.2, 80, 20), (.1, 160, 40), (.05, 320, 80)]
    assert stationary['temporal_check'][0].dt == pytest.approx(.0025)
    transient = runner.plan_runs('transient_baseline', args)
    assert [s.dt for s in transient['temporal']] == [.02, .01, .005, .0025]
    assert {s.h for s in transient['temporal']} == {.05}
    assert all(s.startup == 'exact' and s.steps == round((1 - s.dt)/s.dt) for s in transient['temporal'])
    check = transient['spatial_check'][0]
    assert (check.h, check.outer_vertices, check.dt) == (.025, 4, .0025)
    unchecked = runner.plan_runs('transient_baseline', args_for('--geometry', geometry, '--no-spatial-check'))
    assert list(unchecked) == ['temporal'] and unchecked['temporal'] == transient['temporal']
    startup = runner.plan_runs('transient_stress', args_for('--geometry', geometry, '--study', 'startup'))
    assert all(s.steps == 1 and s.startup == 'euler' for s in startup['startup_one_step'])
    assert [s.steps for s in startup['startup_full']] == [50, 100, 200, 400]
    with pytest.raises(ValueError, match='multiple'):
        runner.RunSpec('transient_baseline', geometry, 'x', 1, .5, 4, 0, .3, 1.).steps


def test_dry_run_prints_without_meshing(capsys, monkeypatch):
    monkeypatch.setattr(runner, 'run_single', lambda *a, **k: (_ for _ in ()).throw(AssertionError('solved')))
    assert runner.main(['--geometry', 'cartesian', '--case', 'transient_stress', '--dry-run']) == 0
    output = capsys.readouterr().out
    assert 'cartesian transient_stress temporal' in output and '320/80' in output


SMOKE = ('--backend', 'numba', '--host-solver', 'direct', '--order', '1', '--final-time', '.15',
         '--volume-degree', '4', '--error-quad-offset', '3')


@pytest.mark.parametrize('geometry', ['cartesian', 'axisymmetric'])
def test_transient_smoke_run_writes_records(tmp_path, geometry):
    pytest.importorskip('gmsh')
    output = tmp_path/'out'
    argv = ['--geometry', geometry, '--case', 'transient_baseline', '--timesteps', '.05',
            '--fixed-mesh-size', '.5', '--output-dir', str(output), '--mesh-cache-dir', str(tmp_path/'mesh'), *SMOKE]
    assert runner.main(argv) == 0
    directory = output/geometry/'transient_baseline'
    summary = json.loads((directory/'transient_baseline_main_summary.json').read_text())
    key = 'n_l2R' if geometry == 'axisymmetric' else 'n_l2'
    row = summary['tables']['temporal'][0]
    assert np.isfinite(row[key]) and row['steps'] == 2 and not row['rejected']
    assert summary['summary']['rejected_runs'] == []
    assert summary['summary']['geometry'] == geometry
    attempts = summary['summary']['spatial_contamination_attempts']
    assert 1 <= len(attempts) <= 3 and attempts[0]['check_h'] == pytest.approx(.25)
    steps = (directory/'transient_baseline_main_steps.csv').read_text().splitlines()
    assert steps[0].startswith('run,') or 'stage' in steps[0]
    assert len(steps) > 2


def test_transient_smoke_run_without_spatial_check(tmp_path):
    pytest.importorskip('gmsh')
    argv = ['--geometry', 'cartesian', '--case', 'transient_baseline', '--timesteps', '.05', '--fixed-mesh-size', '.5',
            '--no-spatial-check', '--output-dir', str(tmp_path/'out'), '--mesh-cache-dir', str(tmp_path/'mesh'), *SMOKE]
    assert runner.main(argv) == 0
    summary = json.loads((tmp_path/'out/cartesian/transient_baseline/transient_baseline_main_summary.json').read_text())
    assert summary['summary']['spatially_resolved'] is None
    assert 'spatial_contamination_attempts' not in summary['summary']
    assert [row['h'] for row in summary['tables']['temporal']] == [.5]


def test_stationary_and_startup_smoke_runs(tmp_path):
    pytest.importorskip('gmsh')
    common = ['--geometry', 'cartesian', '--output-dir', str(tmp_path/'out'),
              '--mesh-cache-dir', str(tmp_path/'mesh'), *SMOKE]
    assert runner.main(common + ['--case', 'stationary_baseline', '--mesh-sizes', '.5', '.25', '--dt', '.05']) == 0
    summary = json.loads((tmp_path/'out/cartesian/stationary_baseline/stationary_baseline_main_summary.json').read_text())
    spatial = summary['tables']['spatial']
    assert [row['h'] for row in spatial] == [.5, .25] and spatial[1]['n_l2_order'] is not None
    assert spatial[0]['actual_h'] > spatial[1]['actual_h']  # orders use the measured mesh size
    assert set(summary['summary']['temporal_contamination']) == {'n_l2', 'Gamma_l2'}
    assert runner.main(common + ['--case', 'transient_baseline', '--study', 'startup', '--timesteps', '.05',
                                 '--fixed-mesh-size', '.5']) == 0
    startup = json.loads((tmp_path/'out/cartesian/transient_baseline/transient_baseline_startup_summary.json').read_text())
    assert startup['tables']['startup_one_step'][0]['steps'] == 1
    assert startup['tables']['startup_full'][0]['steps'] == 3


def test_raw_cuda_smoke_matches_numba(tmp_path):
    """Default raw-CUDA path (face BSR + block-AMG AMGX config) reproduces the Numba errors."""
    pytest.importorskip('gmsh')
    cp = pytest.importorskip('cupy')
    pytest.importorskip('pyamgx')
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip('No CUDA device')
    base = ['--geometry', 'axisymmetric', '--case', 'transient_baseline', '--order', '2', '--timesteps', '.05',
            '--final-time', '.15', '--fixed-mesh-size', '.5', '--volume-degree', '8', '--error-quad-offset', '5',
            '--mesh-cache-dir', str(tmp_path/'mesh')]
    rows = {}
    for backend in ('numba', 'raw-cuda'):
        extra = ['--host-solver', 'direct'] if backend == 'numba' else []
        assert runner.main(base + ['--backend', backend, '--output-dir', str(tmp_path/backend), *extra]) == 0
        summary = json.loads((tmp_path/backend/'axisymmetric/transient_baseline/transient_baseline_main_summary.json')
                             .read_text())
        rows[backend] = summary['tables']['temporal'][0]
    assert not rows['raw-cuda']['rejected']
    for key in ('n_l2R', 'Gamma_l2R'):
        np.testing.assert_allclose(rows['raw-cuda'][key], rows['numba'][key], rtol=1e-6)


PRESET_SMOKE = ('--case', 'transient_baseline', '--order', '2', '--timesteps', '.05', '--final-time', '.15',
                '--fixed-mesh-size', '.5', '--volume-degree', '8', '--error-quad-offset', '5', '--plot-every', '0',
                '--verbosity', '0')


def _steps(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def test_host_preset_smoke_reuses_caches_and_postprocesses(tmp_path):
    pytest.importorskip('gmsh')
    pytest.importorskip('pypardiso')
    out = tmp_path/'host'
    assert runner.main(['--preset', 'mms_xy_p6_numba_pardiso', *PRESET_SMOKE, '--output-dir', str(out),
                        '--mesh-cache-dir', str(tmp_path/'mesh')]) == 0
    directory = out/'cartesian/transient_baseline'
    summary = json.loads((directory/'transient_baseline_main_summary.json').read_text())
    row = summary['tables']['temporal'][0]
    assert not row['rejected'] and np.isfinite(row['n_post_l2']) and np.isfinite(row['Gamma_post_l2'])
    assert summary['summary']['threads']['numba'] == runner.available_cpus()
    runs = json.loads((directory/'transient_baseline_main_runs.json').read_text())
    assert runs['temporal'][0]['threads']['pardiso'] == min(runner.available_cpus(), 8)
    steps = [r for r in _steps(directory/'transient_baseline_main_steps.csv') if r['run'] == 'time_dt0.05']
    assert steps[0]['density_solve.analysis_reused'] == 'False'  # first solve analyses the pattern
    for later in steps[1:]:
        for solve in ('density_solve', 'momentum_solve'):
            assert later[f'{solve}.analysis_reused'] == 'True' and later[f'{solve}.static_reused'] == 'True'
            assert later[f'{solve}.reconstruction_reused'] == 'True'


def test_device_preset_smoke_reuses_caches_and_postprocesses(tmp_path):
    pytest.importorskip('gmsh')
    cp = pytest.importorskip('cupy')
    pytest.importorskip('pyamgx')
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip('No CUDA device')
    out = tmp_path/'device'
    assert runner.main(['--preset', 'mms_xy_p6_device', *PRESET_SMOKE, '--output-dir', str(out),
                        '--mesh-cache-dir', str(tmp_path/'mesh')]) == 0
    directory = out/'cartesian/transient_baseline'
    summary = json.loads((directory/'transient_baseline_main_summary.json').read_text())
    row = summary['tables']['temporal'][0]
    assert not row['rejected'] and np.isfinite(row['n_post_l2']) and np.isfinite(row['Gamma_post_l2'])
    steps = [r for r in _steps(directory/'transient_baseline_main_steps.csv') if r['run'] == 'time_dt0.05']
    for later in steps[1:]:
        for solve in ('density_solve', 'momentum_solve'):
            assert later[f'{solve}.setup_reused'] == 'True' and later[f'{solve}.static_reused'] == 'True'
            assert later[f'{solve}.reconstruction_reused'] == 'True'


def test_presets_plot_every_step_with_backend_by_path():
    from scripts.n_gamma.plotting import resolve_backend
    for key, backend in (('mms_xy_p6_numba_pardiso', 'pyvista'), ('mms_xy_p6_device', 'holoviz')):
        args = runner.parse_arguments(['--preset', key])
        assert args.plot_every == 1 and resolve_backend(args.plot_backend, args.backend) == backend
    assert runner.parse_arguments(['--preset', 'mms_xy_p6_device', '--quiet']).verbosity == 0


def test_verbosity_levels(tmp_path, capsys):
    pytest.importorskip('gmsh')
    base = ['--geometry', 'cartesian', '--case', 'transient_baseline', '--timesteps', '.05', '--fixed-mesh-size', '.5',
            '--mesh-cache-dir', str(tmp_path/'mesh'), *SMOKE]
    outputs = {}
    for level in (0, 1, 2):
        assert runner.main(base + ['--output-dir', str(tmp_path/f'v{level}'), '--verbosity', str(level)]) == 0
        outputs[level] = capsys.readouterr().out
    assert outputs[0] == ''
    assert '== cartesian transient_baseline' in outputs[1] and 'temporal' in outputs[1]
    assert 'iters n/G' not in outputs[1]
    assert 'iters n/G' in outputs[2] and 'reused' in outputs[2]


def test_host_pyvista_panels_save_every_step(tmp_path):
    pytest.importorskip('gmsh')
    pytest.importorskip('pyvista')
    frames = tmp_path/'frames'
    argv = ['--geometry', 'cartesian', '--case', 'transient_baseline', '--timesteps', '.05', '--fixed-mesh-size', '.5',
            '--mesh-cache-dir', str(tmp_path/'mesh'), '--output-dir', str(tmp_path/'out'), '--plot-every', '1',
            '--plot-off-screen', '--plot-dir', str(frames), '--plot-resolution', '2', '--plot-width', '600',
            '--plot-height', '400', '--verbosity', '0', *SMOKE]
    assert runner.main(argv) == 0
    run_frames = sorted((frames/'transient_baseline'/'time_dt0.05').glob('*.png'))
    assert len(run_frames) == 3  # initial state plus two steps
    assert all(frame.stat().st_size > 0 for frame in run_frames)


def test_holoviz_panels_sample_exact_numerical_and_error_on_device(monkeypatch):
    cp = pytest.importorskip('cupy')
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip('No CUDA device')
    import hybridge.io
    from hybridge import DGSpace, rectangle_mesh
    from hybridge.io.raster import DeviceRasterSampler, RasterGeometry
    from scripts.n_gamma.cases import get_case
    from scripts.n_gamma.plotting import LABELS, NGammaPanels

    class StubViewer:
        """Real device samplers; records submitted images instead of opening a window."""
        def __init__(self, spaces, labels, **options):
            self.cp, self.device_id = cp, cp.cuda.runtime.getDevice()
            geometry = RasterGeometry.from_mesh(spaces[0].mesh, 64, 48)
            sampler = DeviceRasterSampler(spaces[0], geometry, device_id=self.device_id)
            self.samplers, self.labels, self.submitted, self.closed = [sampler]*len(spaces), labels, [], False
        def _accept_frame(self):
            return 0.
        def _enqueue(self, images, *, now, step, time_value):
            self.submitted.append((len(images), step, time_value))
        def close(self):
            self.closed = True
    monkeypatch.setattr(hybridge.io, 'HolovizScalarPanels', StubViewer)
    case = get_case('transient_baseline', geometry='cartesian')
    space = DGSpace(rectangle_mesh(3, 3, xlim=(-1., 1.), ylim=(-1., 1.)), 3, basis_type='dub_orth')
    density = space.project_callable(lambda x, y: case.density(x, y, .2))
    momentum = space.project_callable(lambda x, y: case.momentum(x, y, .2))
    panels = NGammaPanels('holoviz', space, case, title='test')
    panels.update(density, momentum, step=4, time_value=.2)
    viewer = panels._viewer
    assert viewer.labels == LABELS and viewer.submitted == [(6, 4, .2)]
    sampler = viewer.samplers[0]
    exact = sampler.sample_callable(lambda x, y: case.density(x, y, .2), device=True)
    host = sampler.sample_callable(lambda x, y: case.density(x, y, .2))
    np.testing.assert_allclose(cp.asnumpy(exact), cp.asnumpy(host), rtol=1e-13, atol=1e-13)
    error = sampler.sample(density) - exact
    np.testing.assert_allclose(cp.asnumpy(error), cp.asnumpy(sampler.sample(density)) - cp.asnumpy(host), atol=1e-13)
    assert 0. < float(cp.abs(error).max()) < .1  # projected p=3 field against the exact density
    panels.close()
    assert viewer.closed
