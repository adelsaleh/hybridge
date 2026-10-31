from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np

from hybridge.diagnostics.guiding_center import modal_activity
from hybridge.io.time_series import DiagnosticPanel, TimeSeries, draw_time_series_panel, numeric_time_series, plot_diagnostic_panels




def test_log_axes_mask_zeros_and_gaps_without_replacing_data(tmp_path):
    from matplotlib.figure import Figure
    figure = Figure()
    axis = figure.subplots()
    series = TimeSeries("error", np.arange(5.), np.array([1., 0., .1, -1., .01]))
    panel = DiagnosticPanel("Error", "L2", (series,), "log", "log")
    draw_time_series_panel(axis, panel)
    assert axis.get_xscale() == axis.get_yscale() == "log"
    np.testing.assert_array_equal(np.isfinite(axis.lines[0].get_ydata()), [False, False, True, False, True])
    paths = plot_diagnostic_panels([panel], tmp_path)
    assert all(file.stat().st_size > 1000 for file in paths)


def test_numeric_histories_retain_nested_stages_and_missing_values():
    rows = [
        {"time": 0, "stage": [{"residual": .1}], "flag": True, "run_configuration": {"dt": .1}},
        {"time": 1, "stage": [{"residual": None}], "flag": False, "run_configuration": {"dt": .1}},
        {"time": 2, "stage": [{"residual": .01}], "flag": True},
    ]
    series = numeric_time_series(rows, exclude=("run_configuration",))
    assert set(series) == {"stage.0.residual"}
    assert np.isnan(series["stage.0.residual"].values[1])


def test_modal_activity_tracks_changing_dominant_modes_and_missing_spectra():
    amplitudes = [[1., .02, 0], [.1, 2, 1e-5], [np.nan, 0, 0]]
    activity = modal_activity(amplitudes, [3, 9, 12])
    np.testing.assert_array_equal(activity["dominant_modes"][:2, 0], [3, 9])
    np.testing.assert_array_equal(activity["active_counts"], [2, 2, 0])
    assert np.isnan(activity["dominant_modes"][2]).all()




















def test_png_names_are_safe_and_do_not_collide(tmp_path):
    series = (TimeSeries("value", np.array([0., 1.]), np.array([1., 2.])),)
    panels = [DiagnosticPanel("Mass / error", "Value", series),
              DiagnosticPanel("Mass : error", "Value", series)]
    paths = plot_diagnostic_panels(panels, tmp_path)
    assert {path.name for path in paths} == {"diagnostics.pdf", "mass_error.png", "mass_error_2.png"}


def test_coherent_groups_paginate_without_mixing_topics():
    from hybridge.io.time_series import diagnostic_pages
    series = (TimeSeries("value", np.array([0., 1.]), np.array([1., 2.])),)
    panels = [DiagnosticPanel(f"Residual {i}", "Value", series, group="solver_residuals")
              for i in range(8)]
    panels.insert(2, DiagnosticPanel("Mass", "Value", series, group="conservation"))
    pages = diagnostic_pages(panels)
    assert [name for name, _ in pages] == [
        "solver_residuals_01.png", "solver_residuals_02.png", "conservation.png"]
    assert [len(page) for _, page in pages] == [6, 2, 1]
    assert all(len({panel.group for panel in page}) == 1 for _, page in pages)
