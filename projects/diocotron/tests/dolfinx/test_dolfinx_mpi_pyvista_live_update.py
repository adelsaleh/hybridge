"""Regression tests for allocation-free live plot updates."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from mpi4py import MPI


pytest.importorskip("dolfinx")
pv = pytest.importorskip("pyvista")

SCRIPT_DIR = Path(__file__).resolve().parents[4] / "projects/diocotron/dolfinx"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from projects.diocotron.dolfinx.plotting.mpi_pyvista import (  # noqa: E402
    MPIPyVistaTorsionPlotter,
    TARGET_BAND_COLOR,
    TARGET_BAND_LINE_WIDTH,
)


def test_global_scalar_array_is_updated_in_place() -> None:
    grid = pv.PolyData(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    grid.point_data["panel_0"] = np.array([1.0, 2.0])
    before = grid.GetPointData().GetArray("panel_0")

    reused = MPIPyVistaTorsionPlotter._update_point_data_in_place(
        grid, "panel_0", np.array([3.0, 4.0])
    )

    assert reused
    assert grid.GetPointData().GetArray("panel_0") is before
    np.testing.assert_allclose(grid.point_data["panel_0"], [3.0, 4.0])


class _TextActor:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def set_text(self, position: int, text: str) -> None:
        self.calls.append((position, text))


class _Mapper:
    scalar_range = None


class _ScalarActor:
    mapper = _Mapper()


class _LivePlotter:
    def __init__(self) -> None:
        self.update_calls = 0
        self.remove_calls = 0
        self.add_text_calls = 0

    def subplot(self, *_args) -> None:
        pass

    def update(self) -> None:
        self.update_calls += 1

    def remove_actor(self, *_args, **_kwargs) -> None:
        self.remove_calls += 1

    def add_text(self, *_args, **_kwargs):
        self.add_text_calls += 1
        return _TextActor()


class _Grid:
    n_points = 2
    n_cells = 1
    point_data = {"panel_0": np.array([1.0, 2.0])}

    def Modified(self) -> None:
        pass


def test_steady_live_update_reuses_text_actor_and_renders_once() -> None:
    plotter = object.__new__(MPIPyVistaTorsionPlotter)
    live = _LivePlotter()
    text_actor = _TextActor()
    plotter._live_plotter = live
    plotter._mpi_live_actors = [_ScalarActor()]
    plotter._mpi_live_text_actors = [text_actor]
    plotter._mpi_live_key = (2, 1, 1, None, None)
    plotter.comm = MPI.COMM_SELF
    plotter.args = SimpleNamespace(plot_mesh_edges=False)

    plotter._update_mpi_live_plotter(
        _Grid(),
        ["energy step"],
        window_size=(800, 600),
        nt=1,
        ndof=2,
        contour_field_index=None,
        contour_levels=None,
        stage="ENERGY_PRIMER",
    )

    assert live.update_calls == 1
    assert live.remove_calls == 0
    assert live.add_text_calls == 0
    assert plotter._mpi_live_text_actors[0] is text_actor
    assert text_actor.calls == [(7, "energy step\nnt=1 ndof=2")]


class _BoundaryPlotter:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict]] = []

    def add_mesh(self, curve, **kwargs) -> None:
        self.calls.append((curve, kwargs))


def test_both_target_thresholds_use_thin_non_tubular_lines() -> None:
    plotter = _BoundaryPlotter()
    lower = object()
    upper = object()

    MPIPyVistaTorsionPlotter._add_band_boundaries(
        plotter,
        [(lower, TARGET_BAND_COLOR), (upper, TARGET_BAND_COLOR)],
    )

    assert [curve for curve, _ in plotter.calls] == [lower, upper]
    assert all(
        options["line_width"] == TARGET_BAND_LINE_WIDTH == 1.0
        for _, options in plotter.calls
    )
    assert all(not options["render_lines_as_tubes"] for _, options in plotter.calls)


def test_target_threshold_dash_geometry_contains_real_gaps() -> None:
    coordinates = np.linspace(0.0, 1.0, 129)
    points = np.column_stack((coordinates, np.zeros_like(coordinates), np.zeros_like(coordinates)))
    curve = pv.lines_from_points(points, close=False)

    dashed = MPIPyVistaTorsionPlotter._dashed_polyline(curve, target_dashes=8)

    assert dashed.n_verts == 0
    assert 0 < dashed.n_lines < points.shape[0] - 1
