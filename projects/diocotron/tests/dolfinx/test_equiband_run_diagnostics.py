"""Pure-NumPy regressions for consolidated run diagnostics."""
from types import SimpleNamespace
import time

import numpy as np

from projects.diocotron.dolfinx.equiband.geometry import PackedMesh
from projects.diocotron.dolfinx.equiband.run_diagnostics import (
    RunTelemetry,
    mesh_resolution_diagnostics,
    timing_log_lines,
)


class SerialComm:
    rank = 0
    size = 1

    @staticmethod
    def allgather(value):
        return [value]

    @staticmethod
    def bcast(value, root=0):
        return value


def test_realized_mesh_resolution_is_measured_not_copied_from_request():
    height = np.sqrt(3.)/2
    mesh = PackedMesh.from_triangles(np.array([[[0., 0.], [1., 0.], [.5, height]]]))
    result = mesh_resolution_diagnostics(mesh, requested_size=.25)
    assert result["physical_edge_length_max"] == 1.
    assert result["h_max_over_requested_size"] == 4.
    np.testing.assert_allclose(result["corner_minimum_angle_degrees"], 60.)
    np.testing.assert_allclose(result["corner_shape_quality_min"], 1.)
    assert result["coordinate_orientation_failures"] == 0


def test_telemetry_separates_render_and_user_wait_and_formats_rank_statistics(tmp_path):
    plotter = SimpleNamespace(performance_counters={
        "updates": 3,
        "render_seconds": .125,
        "interactive_wait_seconds": .25,
    })
    telemetry = RunTelemetry(SerialComm(), time.perf_counter()-.5)
    telemetry.record("solver", .2)
    (tmp_path/"artifact").write_bytes(b"1234")
    result = telemetry.summarize(plotter=plotter, output_directory=tmp_path)
    assert result["ranks"] == 1
    assert result["interactive_wait_seconds"]["max"] == .25
    assert result["plot_render_seconds"]["max"] == .125
    assert result["phases"]["solver"]["mean"] == .2
    assert result["output"] == {
        "snapshot_stage": "before_run_end_and_terminal_log_close",
        "files": 1,
        "bytes": 4,
    }
    lines = timing_log_lines(result)
    assert lines[0].startswith("RUN_TIMING ranks=1")
    assert "output_snapshot_files=1 output_snapshot_bytes=4" in lines[0]
    assert any("phase=solver" in line for line in lines)
