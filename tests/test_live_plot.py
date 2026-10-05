"""Live plotting contracts using synthetic samples and an in-memory VTK stand-in."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hybridge import rectangle_mesh
from hybridge.io import PyVistaFieldPanels, scalar_color_limits
from hybridge.io import live, plot


class PolyData:
    def __init__(self, points, faces):
        self.points, self.faces = points, faces
        self.point_data = {}
        self.modified = 0

    def Modified(self):
        self.modified += 1


class Plotter:
    window_class = "vtkXOpenGLRenderWindow"

    def __init__(self, **options):
        self.options = options
        self.render_window = SimpleNamespace(GetClassName=lambda: self.window_class)
        self.meshes, self.actors, self.shows, self.screenshots, self.texts = [], [], [], [], []
        self.renders = self.updates = self.closes = self.links = self.grids = 0

    def add_mesh(self, mesh, **options):
        self.meshes.append(mesh)
        actor = SimpleNamespace(mapper=SimpleNamespace(scalar_range=options.get("clim")))
        self.actors.append(actor)
        return actor

    def subplot(self, *args):
        pass

    def add_text(self, text, **kwargs):
        actor = SimpleNamespace(text=text)
        actor.SetInput = lambda value: setattr(actor, "text", value)
        self.texts.append(actor)
        return actor

    def add_title(self, *args, **kwargs):
        pass

    def enable_parallel_projection(self):
        pass

    def view_xy(self):
        pass

    def show_grid(self, **kwargs):
        self.grids += 1

    def link_views(self):
        self.links += 1

    def show(self, **kwargs):
        self.shows.append(kwargs)

    def render(self):
        self.renders += 1

    def update(self):
        self.updates += 1

    def screenshot(self, path):
        self.screenshots.append(Path(path))

    def close(self):
        self.closes += 1


@pytest.fixture
def vtk(monkeypatch):
    module = SimpleNamespace(Plotter=Plotter, PolyData=PolyData)
    monkeypatch.setattr(live, "_require_pyvista", lambda: module)
    monkeypatch.setattr(plot, "_require_pyvista", lambda: module)
    return module


def sampled_field(space, scale=1.0, name="scalar"):
    # Each element has a distinct constant value, exercising discontinuities.
    def values_at_ref(points):
        return np.broadcast_to(scale * np.array([[-1.0], [2.0]]), (2, len(points)))
    return SimpleNamespace(space=space, name=name, values_at_ref=values_at_ref)


@pytest.mark.parametrize("window_class, off_screen, render_only", [
    ("vtkXOpenGLRenderWindow", False, False),
    ("vtkXOpenGLRenderWindow", True, True),
    ("vtkEGLRenderWindow", False, True),
    ("vtkOSOpenGLRenderWindow", False, True),
])
def test_live_updates_keep_geometry_scalars_and_fixed_range(vtk, monkeypatch, tmp_path,
                                                           window_class, off_screen, render_only):
    monkeypatch.setattr(Plotter, "window_class", window_class)
    space = SimpleNamespace(mesh=rectangle_mesh(1, 1))
    field = sampled_field(space)
    viewer = PyVistaFieldPanels(
        [("Signed", field, {"fixed_clim": True, "symmetric_clim": True, "robust_percentile": 100}),
         ("Dynamic", field, {"scalar_name": "other"})],
        resolution=3, show_mesh=True, off_screen=off_screen,
        screenshot_dir=tmp_path / "frames", screenshot_prefix="sample",
    )
    renderer = viewer.plotter
    assert len(renderer.meshes) == 4  # two fields plus coarse wireframes
    first, second = renderer.meshes[0], renderer.meshes[2]
    first_values = first.point_data["scalar"]
    second_values = second.point_data["other"]
    first_points = first.points
    assert len(first_points) == 2 * len(viewer.reference_points)
    for step in (0, 1):
        changed = sampled_field(space, scale=10)
        viewer.update([changed, changed], step=step, time_value=step * .25)
    assert renderer.meshes[0] is first and first.points is first_points
    assert first.point_data["scalar"] is first_values
    assert second.point_data["other"] is second_values
    np.testing.assert_array_equal(first_values.reshape(2, -1)[:, 0], [-10, 20])
    assert first.modified == second.modified == 2
    assert renderer.actors[0].mapper.scalar_range == (-2., 2.)
    assert renderer.actors[2].mapper.scalar_range == (-10., 20.)
    assert renderer.shows == [{"auto_close": False, "interactive_update": not render_only}]
    assert renderer.renders == int(render_only)
    assert renderer.updates == int(not render_only)
    assert renderer.links == 1 and renderer.grids == 0
    assert [p.name for p in renderer.screenshots] == [
        "sample_step00000_t0.000000.png", "sample_step00001_t0.250000.png",
    ]
    assert (tmp_path / "frames").is_dir()
    viewer.close()
    viewer.close()
    assert renderer.closes == 1
    with pytest.raises(RuntimeError, match="closed"):
        viewer.update([field, field], step=2, time_value=.5)


def test_live_rejects_different_spaces_before_updating_any_panel(vtk):
    space = SimpleNamespace(mesh=rectangle_mesh(1, 1))
    field = sampled_field(space)
    viewer = PyVistaFieldPanels([("First", field), ("Second", field)], show_mesh=False)
    try:
        with pytest.raises(ValueError, match="same length"):
            viewer.update([field], step=0, time_value=0.)
        other = sampled_field(SimpleNamespace(mesh=space.mesh))
        with pytest.raises(ValueError, match="original DGSpace"):
            viewer.update([field, other], step=0, time_value=0.)
        assert all(mesh.modified == 0 for mesh in viewer.plotter.meshes)
        assert viewer.plotter.shows == []
    finally:
        viewer.close()


def test_add_field_preserves_mesh_return_and_can_return_actor(vtk):
    space = SimpleNamespace(mesh=rectangle_mesh(1, 1))
    renderer = Plotter()
    field = sampled_field(space)
    mesh = plot.add_field_to_plotter(renderer, field, resolution=3, show_mesh=False)
    assert isinstance(mesh, PolyData)
    supplied = np.full((2, len(plot.reference_plot_points(3))), 7.)
    field.values_at_ref = lambda points: pytest.fail("resampled supplied values")
    other, actor = plot.add_field_to_plotter(
        renderer, field, resolution=3, values=supplied, show_mesh=False, return_actor=True,
    )
    np.testing.assert_array_equal(other.point_data["scalar"], supplied.ravel())
    assert actor is renderer.actors[-1]
    assert renderer.grids == 2


def test_runner_adapter_preserves_vorticity_policy(monkeypatch):
    import hybridge.io
    from scripts.guiding_center.runtime.plotting import GuidingCenterPyVistaPanels

    calls = []
    class Viewer:
        def __init__(self, panels, **options):
            calls.append((panels, options))
        def update(self, fields, **frame):
            calls.append((fields, frame))
        def close(self):
            calls.append("closed")
    monkeypatch.setattr(hybridge.io, "PyVistaFieldPanels", Viewer)
    adapter = GuidingCenterPyVistaPanels(
        "rho", "phi", resolution=4, title="case", show_mesh=False, off_screen=True,
        screenshot_dir=None, screenshot_prefix="case", include_potential=True, density_is_vorticity=True,
    )
    panels, options = calls[0]
    assert [p[0] for p in panels] == ["Density", "Potential"]
    assert panels[0][2] == dict(scalar_name="density", cmap="RdBu_r", symmetric_clim=True,
                                fixed_clim=True, robust_percentile=100.)
    assert options["window_size"] == (1500, 650)
    adapter.update("new rho", "new phi", step=3, time_value=.3)
    assert calls[1] == (["new rho", "new phi"], dict(step=3, time_value=.3))
    adapter.close()
    assert calls[-1] == "closed"


def test_symmetric_limits_handle_nonfinite_and_constant_samples():
    assert scalar_color_limits([np.nan, np.inf, -3, 2], symmetric=True, percentile=100) == (-3., 3.)
    assert scalar_color_limits([np.nan], symmetric=True) == (-1., 1.)
    assert scalar_color_limits([0., 0.], symmetric=True) == (-1.e-30, 1.e-30)
    with pytest.raises(ValueError, match="percentile"):
        scalar_color_limits([1.], percentile=np.nan)
    with pytest.raises(ValueError, match="cannot both"):
        scalar_color_limits([1.], symmetric=True, zero_min=True)
