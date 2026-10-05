"""CPU numerical and integration checks; real GPU rendering has a static smoke script."""

from dataclasses import replace
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from hybridge import DGMesh, DGSpace, rectangle_mesh
from hybridge.io.raster import DeviceRasterSampler, RasterGeometry
from hybridge.runtime.precision import REAL_DTYPE
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.runtime import runner
import scripts.guiding_center.run_guiding_center_cases as cli
import scripts.guiding_center.runtime.plotting as plotting


def pixel_centers(geometry):
    xmin, xmax, ymin, ymax = geometry.bounds
    x = xmin + (np.arange(geometry.width) + .5) * (xmax - xmin) / geometry.width
    y = ymax - (np.arange(geometry.height) + .5) * (ymax - ymin) / geometry.height
    return np.meshgrid(x, y)


@pytest.mark.parametrize("basis", ["bernstein", "hier_C0", "dub_orth"])
@pytest.mark.parametrize("dimensions", [(43, 35), (31, 57)])
def test_raster_reproduces_physical_quadratic(basis, dimensions):
    space = DGSpace(rectangle_mesh(3, 2, xlim=(-2., 3.), ylim=(-1., 2.)), 2, basis_type=basis)
    geometry = RasterGeometry.from_mesh(space.mesh, *dimensions)
    polynomial = lambda x, y: 1. + 2. * x - 3. * y + x * y + .5 * y**2
    field = space.project_callable(polynomial)
    actual = geometry.sampling_matrix(space) @ field.coeffs.ravel()
    x, y = pixel_centers(geometry)
    valid = geometry.valid_pixels
    tolerance = 500 * np.finfo(REAL_DTYPE).eps
    np.testing.assert_allclose(actual[valid], polynomial(x, y).ravel()[valid], atol=tolerance, rtol=tolerance)
    assert np.all(actual[geometry.element_ids < 0] == 0.)


def test_raster_keeps_discontinuous_values_and_holes():
    full = rectangle_mesh(3, 3)
    # Remove both triangles of the middle cell, retaining all surrounding nodes.
    mesh = DGMesh.from_arrays(full.node_coords, np.delete(full.triangles, [8, 9], axis=0))
    space = DGSpace(mesh, 2, basis_type="bernstein")
    geometry = RasterGeometry.from_mesh(mesh, 61, 61)
    per_element = 10. + np.arange(mesh.num_tri)
    coefficients = np.broadcast_to(per_element[:, None], space.shape)
    values = geometry.sampling_matrix(space) @ coefficients.ravel()
    valid = geometry.valid_pixels
    np.testing.assert_allclose(
        values[valid], per_element[geometry.element_ids[valid]],
        rtol=10 * np.finfo(REAL_DTYPE).eps,
    )
    x, y = pixel_centers(geometry)
    hole = (np.abs(x) < .25) & (np.abs(y) < .25)
    assert hole.any()
    assert np.all(geometry.element_ids.reshape(x.shape)[hole] == -1)
    assert np.all(values.reshape(x.shape)[hole] == 0.)


def test_raster_retains_pixel_centers_for_callable_panels():
    space = DGSpace(rectangle_mesh(3, 2, xlim=(-2., 3.), ylim=(-1., 2.)), 1)
    geometry = RasterGeometry.from_mesh(space.mesh, 37, 29)
    x, y = pixel_centers(geometry)
    valid = geometry.valid_pixels
    np.testing.assert_allclose(geometry.valid_points[:, 0], x.ravel()[valid])
    np.testing.assert_allclose(geometry.valid_points[:, 1], y.ravel()[valid])


def test_device_callable_samples_and_clipped_lut_indices():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("No CUDA device")
    space = DGSpace(rectangle_mesh(2, 2), 2)
    geometry = RasterGeometry.from_mesh(space.mesh, 24, 20)
    sampler = DeviceRasterSampler(space, geometry, device_id=cp.cuda.runtime.getDevice())
    polynomial = lambda x, y: 1. + x - 2. * y + x * y
    projected = sampler.sample(space.project_callable(polynomial))
    exact = sampler.sample_callable(polynomial)
    np.testing.assert_allclose(cp.asnumpy(exact), cp.asnumpy(projected), atol=1e-12)
    assert np.all(cp.asnumpy(exact)[geometry.element_ids < 0] == 0.)
    image, _ = sampler.values_image(exact, limits=(cp.asarray(0.), cp.asarray(1.)))
    indices = cp.asnumpy(image)
    assert indices.min() >= 0. and indices.max() <= 255.


def test_sampling_map_checks_memory_before_basis_allocation():
    space = DGSpace(rectangle_mesh(1, 1), 2)
    geometry = RasterGeometry.from_mesh(space.mesh, 20, 20)
    with pytest.raises(ValueError, match="Reduce --plot-width/--plot-height"):
        geometry.sampling_matrix(space, max_bytes=1)


def test_device_sampler_rejects_cross_device_host_fallback():
    space = object()
    sampler = DeviceRasterSampler.__new__(DeviceRasterSampler)
    sampler.space, sampler.device_id = space, 0
    field = SimpleNamespace(
        space=space, device_coefficients_materialized=lambda device_id=None: device_id is None,
    )
    with pytest.raises(ValueError, match="same CUDA device"):
        sampler.sample(field)


@pytest.mark.parametrize("backend", ["pyvista", "holoviz"])
def test_factory_selects_only_requested_backend(monkeypatch, backend):
    from hybridge.io import holoviz

    def construct(*fields, **options):
        return fields, options

    def forbidden(*args, **kwargs):
        raise AssertionError("constructed the unselected backend")

    monkeypatch.setattr(plotting, "GuidingCenterPyVistaPanels", construct if backend == "pyvista" else forbidden)
    monkeypatch.setattr(holoviz, "GuidingCenterHolovizPanels", construct if backend == "holoviz" else forbidden)
    config = replace(preset_by_key("diocotron_gaussian_annulus_host_smoke"), plot_backend=backend)
    fields, options = plotting._make_plotter(
        config, "density", "potential", title="test", off_screen=True,
        screenshot_dir=None, screenshot_prefix="test", density_is_vorticity=True,
    )
    assert fields == ("density", "potential")
    assert options["density_is_vorticity"] is True
    assert options["screenshot_dir"] is None
    if backend == "holoviz":
        assert options["width"] == config.plot_width
        assert options["max_fps"] == config.plot_max_fps


def test_headless_holoviz_only_saves_when_requested(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    config = replace(preset_by_key("diocotron_gaussian_annulus_host_smoke"), plot_backend="holoviz", plot_every=1)
    assert plotting._plot_output_settings(config, "test") == (True, True, None)
    assert plotting._plot_output_settings(replace(config, screenshot_dir="frames"), "test") == (True, True, "frames")
    assert plotting._plot_output_settings(replace(config, plot_backend="pyvista"), "test")[2] is not None


def test_holoviz_cli_dry_run_does_not_start_solver(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry run started a solve")

    monkeypatch.setattr(cli, "_run_cli_case_with_terminal_log", forbidden)
    monkeypatch.setattr(sys, "argv", [
        "run_guiding_center_cases.py", "--preset", "diocotron_gaussian_annulus_host_smoke",
        "--plot", "--plot-backend", "holoviz", "--plot-width", "320", "--plot-height", "240",
        "--plot-max-fps", "5", "--dry-run",
    ])
    cli._main()
    output = capsys.readouterr().out
    assert "holoviz" in output
    assert "320" in output and "240" in output


@pytest.mark.parametrize("overrides", [
    {"plot_backend": "unknown"}, {"plot_width": 1}, {"plot_height": 0},
    {"plot_max_fps": 0}, {"plot_max_fps": float("nan")}, {"plot_every": -1},
])
def test_invalid_plot_options_fail_before_solver(overrides):
    config = replace(preset_by_key("diocotron_gaussian_annulus_host_smoke"), **overrides)
    with pytest.raises(ValueError, match="plot"):
        runner._validate_config(config)


def _viewer_queue_without_gpu():
    import threading
    from hybridge.io.holoviz import GuidingCenterHolovizPanels
    viewer = GuidingCenterHolovizPanels.__new__(GuidingCenterHolovizPanels)
    viewer._condition = threading.Condition()
    viewer._pending = viewer._inflight = viewer._last_frame = None
    viewer._closing = viewer._user_closed = viewer.off_screen = False
    viewer.frames_rendered = 0
    viewer.last_frame_latency = 0.
    return viewer


def test_minimized_frame_is_retried_until_matching_completion():
    from hybridge.io.holoviz import _Frame
    viewer = _viewer_queue_without_gpu()
    first, second = _Frame({}, None, 1, .1), _Frame({}, None, 2, .2)
    viewer._pending = first
    assert viewer._frame_for_tick() is first
    viewer._pending = second
    # No framebuffer while minimized: keep first alive and continue ticking.
    assert viewer._frame_for_tick() is first
    assert viewer._pending is second
    viewer._complete_frame(first)
    assert viewer._frame_for_tick() is second
    # A delayed duplicate acknowledgement must not release the newer frame.
    viewer._complete_frame(first)
    assert viewer._inflight is second
    assert viewer.frames_rendered == 1
    viewer._complete_frame(second)
    assert viewer.frames_rendered == 2


def test_idle_redraw_does_not_add_frames_or_prevent_shutdown():
    from hybridge.io.holoviz import _Frame
    viewer = _viewer_queue_without_gpu()
    frame = _Frame({}, None, 1, .1)
    viewer._complete_frame(frame)
    assert viewer._frame_for_tick() is frame
    viewer._complete_frame(frame)
    assert viewer.frames_rendered == 1
    viewer.off_screen = True
    assert viewer._frame_for_tick() is None
    viewer.off_screen = False
    viewer._closing = True
    assert viewer._frame_for_tick() is None


@pytest.mark.parametrize("enabled", [False, True])
def test_movie_toggle_reaches_only_selected_capture(monkeypatch, enabled):
    import hybridge.io.holoviz as holoviz
    config = replace(preset_by_key("diocotron_gaussian_m64_si_bdf2_p6_h0068_dt01_t400_fast"),
                     save_movie=enabled)
    monkeypatch.setattr(holoviz, "GuidingCenterHolovizPanels", lambda *args, **kw: kw)
    options = plotting._make_plotter(config, None, None, title="test", off_screen=True,
                                    screenshot_dir=None, screenshot_prefix="test", density_is_vorticity=True)
    assert options["movie_path"] == (config.movie_path if enabled else None)
    assert options["movie_fps"] == 20


def test_color_limits_keep_padding_until_exceeded_and_never_shrink():
    from hybridge.io.live import expanding_color_limits
    limits = expanding_color_limits(-2., 8.)
    np.testing.assert_allclose(limits, (-3., 9.))
    for bounds in [(-1., 5.), (-2.9, 8.9), (-3., 9.)]:
        np.testing.assert_allclose(expanding_color_limits(*bounds, limits=limits), limits)
    expanded = expanding_color_limits(-4., 8., limits=limits)
    assert expanded[0] < -4. and expanded[1] == limits[1]
    again = expanding_color_limits(-2., 10., limits=expanded)
    assert again[0] == expanded[0] and again[1] > 10.
    np.testing.assert_allclose(expanding_color_limits(0., 1., limits=again), again)


def test_symmetric_color_limits_expand_both_sides_only_on_exceedance():
    from hybridge.io.live import expanding_color_limits
    limits = expanding_color_limits(-2., 8., symmetric=True)
    np.testing.assert_allclose(limits, (-8.8, 8.8))
    np.testing.assert_allclose(expanding_color_limits(-8.5, 2., limits=limits, symmetric=True), limits)
    limits = expanding_color_limits(-9., 2., limits=limits, symmetric=True)
    np.testing.assert_allclose(limits, (-9.9, 9.9))
    lo, hi = expanding_color_limits(0., 0.)
    assert lo < 0 < hi


def test_sampler_keeps_same_value_same_color_until_range_is_exceeded():
    sampler = DeviceRasterSampler.__new__(DeviceRasterSampler)
    sampler.cp = np
    sampler.valid = np.array([True, True, True, False])
    sampler.geometry = SimpleNamespace(width=2, height=2)
    sampler.sample = lambda values: np.asarray(values, dtype=float)
    first, limits = sampler.image([0., 5., 10., 1000.], expand_limits=True)
    second, kept = sampler.image([2., 5., 8., -1000.], limits=limits, expand_limits=True)
    np.testing.assert_allclose(kept, limits)
    assert first.ravel()[1] == second.ravel()[1]
    _, expanded = sampler.image([-2., 5., 10., 0.], limits=kept, expand_limits=True)
    assert expanded[0] < -2. and expanded[1] == kept[1]


def test_time_labels_keep_decimal_after_accumulated_timestep_rounding():
    from hybridge.io.live import simulation_frame_label
    time_value = 0.
    for step in range(1, 31):
        time_value += .1
        if step % 10 == 0:
            label = simulation_frame_label(step=step, time_value=time_value, time_step=.1)
            assert label.startswith(f"t = {step // 10}.0 |")
    for value, expected in [(0., "0.0"), (1., "1.0"), (1.5, "1.5"),
                            (.05, "0.05"), (2.9999999999999996, "3.0"),
                            (1e6, "1.0e+06")]:
        assert simulation_frame_label(step=0, time_value=value).startswith(f"t = {expected} |")
