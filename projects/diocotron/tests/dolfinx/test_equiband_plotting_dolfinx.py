"""Real off-screen PyVista rendering without changing accepted FE snapshots."""
import numpy as np
import pytest

pytest.importorskip("dolfinx", minversion="0.11.0")
pytest.importorskip("pyvista")
from dolfinx import fem

from projects.diocotron.dolfinx.equiband.cli import parser
from projects.diocotron.dolfinx.equiband.config import SolverConfig
from projects.diocotron.dolfinx.equiband.continuation import BranchController
from projects.diocotron.dolfinx.equiband.equilibrium import EquilibriumSolver
from projects.diocotron.dolfinx.equiband.plotting import EquibandPlotter
from projects.diocotron.dolfinx.equiband.radial import solve_radial


def test_saved_panels_reuse_arrays_and_preserve_working_field(tmp_path, monkeypatch):
    config = SolverConfig(mesh_size=.15, number_of_rays=16, samples_per_ray=80)
    messages = []
    report = lambda message, level=1: messages.append(message)
    solver = EquilibriumSolver(config, report=report)
    radial = solve_radial(config.band)
    guess = fem.Function(solver.V)
    guess.interpolate(lambda x: radial.values(np.hypot(x[0], x[1])))
    state = solver.solve(radial.m, initial_values=guess.x.array[:solver.owned])
    controller = BranchController(solver)
    first = controller.seed(state)
    second = controller.step(first, state.m+1e-4)
    saved = solver.phi.x.array.copy()
    saved_m = float(solver.m.value)
    args = parser().parse_args(["--config", "unused", "--output", str(tmp_path), "--m-stop", ".06",
                               "--no-plot", "--plot-off-screen", "--save-frames",
                               "--plot-window-width", "1200", "--plot-window-height", "600"])
    plotter = EquibandPlotter(solver, args, tmp_path, report)
    monkeypatch.setattr(plotter, "_wait_for_enter", lambda: pytest.fail("off-screen plots must never pause"))
    try:
        plotter.emit(first)
        assert not plotter.disabled, messages
        array = plotter.grid.GetPointData().GetArray("phi")
        actors = list(plotter.actors)
        contours = dict(plotter.contour_actors)
        camera = plotter.plotter.camera_position
        middle = plotter.plotter.renderer.actors["middle"].mapper.dataset.points.copy()
        plotter.emit(second, stage="ACCEPTED_TEST", force=True, pause=True)
        assert not plotter.disabled, messages
        assert plotter.grid.GetPointData().GetArray("phi") is array
        assert plotter.actors == actors
        assert plotter.contour_actors == contours
        assert plotter.plotter.camera_position == camera
        moved_middle = plotter.plotter.renderer.actors["middle"].mapper.dataset.points
        assert middle.shape != moved_middle.shape or not np.allclose(middle, moved_middle)
        np.testing.assert_allclose(plotter.grid.point_data["rho"], solver.window.value(plotter.grid.point_data["phi"], second.state.m))
        assert plotter.grid.n_cells == len(solver.audit_mesh.triangles)*args.plot_refinement**2
        np.testing.assert_array_equal(solver.phi.x.array, saved)
        assert float(solver.m.value) == saved_m
        assert not state.values.flags.writeable
        frames = sorted((tmp_path/"frames").glob("*.png"))
        assert len(frames) == 2
        assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in frames)
        assert any("SNES m=" in message for message in messages)
        assert any("EQUILIBRIUM_DONE" in message for message in messages)
        assert any("PLOT_UPDATE" in message and "render=" in message for message in messages)
    finally:
        plotter.close()

    # Exercise real VTK live-update/event paths without opening a desktop
    # window during CI. Only the window backend is forced off-screen here;
    # the equiband interactive controls and Enter pause still run normally.
    import pyvista as pv
    import sys
    from types import SimpleNamespace
    from projects.diocotron.dolfinx.equiband import plotting
    factory = pv.Plotter
    monkeypatch.setattr(pv, "Plotter", lambda **kwargs: factory(**{**kwargs, "off_screen": True}))
    args.plot, args.plot_off_screen, args.save_frames = True, False, False
    live = EquibandPlotter(solver, args, tmp_path, report)
    entered = []
    # MPI-forwarded terminal input is a pipe, not a TTY, but must still pause.
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False, readline=lambda: entered.append(True) or "\n"))
    monkeypatch.setattr(plotting.select, "select", lambda *_args: ([sys.stdin], [], []))
    try:
        live.emit(first, pause=True)
        live.pump(force=True)
        assert entered and not live.disabled and live._pending_error is None, messages
        args.plot_min_interval = 1e6
        count = sum(message.startswith("PLOT_UPDATE") for message in messages)
        live.emit(second)
        assert sum(message.startswith("PLOT_UPDATE") for message in messages) == count
        live.emit(second, force=True, pause=False)
        live.emit(first, pause=True)  # an explicit pause bypasses the live throttle
        assert sum(message.startswith("PLOT_UPDATE") for message in messages) == count+2
        live.plotter.close()
        live.pump(force=True)
        live.emit(second)
        assert live.disabled  # Closing the window is not a Newton/branch error.
        np.testing.assert_array_equal(solver.phi.x.array, saved)
    finally:
        live.close()


def test_plain_text_backend_restores_vtk_policy_even_on_failure():
    import vtkmodules.vtkRenderingFreeType
    from vtkmodules.vtkRenderingCore import vtkTextRenderer
    from projects.diocotron.dolfinx.equiband.plotting import _plain_text_backend
    renderer = vtkTextRenderer.GetInstance()
    previous = renderer.GetDefaultBackend()
    with pytest.raises(RuntimeError, match="render failed"):
        with _plain_text_backend():
            assert renderer.GetDefaultBackend() == vtkTextRenderer.FreeType
            raise RuntimeError("render failed")
    assert renderer.GetDefaultBackend() == previous
