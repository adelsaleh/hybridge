"""CPU numerical and integration checks; real GPU rendering has a static smoke script."""

from dataclasses import replace
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from hdgfem import DGMesh, DGSpace, rectangle_mesh
from hdgfem.io.raster import DeviceRasterSampler, RasterGeometry
from hdgfem.precision import REAL_DTYPE
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
    from hdgfem.io import holoviz

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
    from hdgfem.io.holoviz import GuidingCenterHolovizPanels
    viewer = GuidingCenterHolovizPanels.__new__(GuidingCenterHolovizPanels)
    viewer._condition = threading.Condition()
    viewer._pending = viewer._inflight = viewer._last_frame = None
    viewer._closing = viewer._user_closed = viewer.off_screen = False
    viewer.frames_rendered = 0
    viewer.last_frame_latency = 0.
    return viewer


def test_minimized_frame_is_retried_until_matching_completion():
    from hdgfem.io.holoviz import _Frame
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
    from hdgfem.io.holoviz import _Frame
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
