"""Host-only plot sampling, logging and failure-policy checks."""
from types import SimpleNamespace

import numpy as np
import pytest

from projects.diocotron.dolfinx.equiband.geometry import SITES, VANDERMONDE_INVERSE
from projects.diocotron.dolfinx.equiband.plotting import PlotSampling, EquibandPlotter, preflight_plotting
from projects.diocotron.dolfinx.equiband.reporting import ProgressReporter


@pytest.mark.parametrize("refinement", [1, 2, 4, 8])
def test_plot_sampling_evaluates_every_cell_polynomial(refinement):
    triangles = np.array([[[0., 0.], [1., 0.], [0., 1.]], [[1., 1.], [0., 1.], [1., 0.]]])
    sample = PlotSampling.build(triangles, refinement)
    physical = (triangles[:, :1]+SITES[None, :, :1]*(triangles[:, 1:2]-triangles[:, :1])
                + SITES[None, :, 1:]*(triangles[:, 2:3]-triangles[:, :1]))
    def exact(points):
        x, y = points[..., 0], points[..., 1]
        return 1.+2*x-3*y+x*y+.4*x*x+.8*y*y
    coefficients = exact(physical) @ VANDERMONDE_INVERSE.T
    np.testing.assert_allclose(sample.values(coefficients), exact(sample.points), atol=1e-14)
    cells = sample.faces.reshape(-1, 4)
    assert len(cells) == len(triangles)*refinement**2
    assert np.all(cells[:, 0] == 3)
    assert cells[:, 1:].min() == 0 and cells[:, 1:].max() == len(sample.points)-1
    assert not np.any(sample.points[:, 2])


def test_batch_preflight_does_not_import_pyvista(monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name != "pyvista"
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    preflight_plotting(SimpleNamespace(plot=False, save_frames=False), None)


def test_plot_failure_closes_window_and_preserves_numerical_work():
    plotter = object.__new__(EquibandPlotter)
    closed, messages = [], []
    plotter.comm = SimpleNamespace(rank=0, bcast=lambda value, root: value)
    plotter.plotter = SimpleNamespace(close=lambda: closed.append(True))
    plotter.report = lambda message, level=1: messages.append(message)
    plotter._pending_error, plotter.disabled = None, False
    def fail():
        raise RuntimeError("injected render failure")
    plotter._root(fail)
    assert closed and plotter.disabled and plotter.plotter is None
    assert "PLOT_DISABLED" in messages[0] and "injected render failure" in messages[0]


def test_verbosity_is_rank_zero_only_and_does_not_disable_event_pump(capsys):
    pumped = []
    report = ProgressReporter(SimpleNamespace(rank=0), 0)
    report.pump = lambda: pumped.append(True)
    report("hidden", level=2)
    report("result", level=0)
    output = capsys.readouterr().out
    assert output.startswith("[+") and output.endswith("s] result\n")
    assert len(pumped) == 2
    report = ProgressReporter(SimpleNamespace(rank=1), 2)
    report("not root", level=0)
    assert not capsys.readouterr().out
    report = ProgressReporter(SimpleNamespace(rank=0), 2)
    report("SNES diagnostics", level=2)
    assert "SNES diagnostics" in capsys.readouterr().out
