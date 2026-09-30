from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np

from hdgfem.diagnostics.guiding_center import modal_activity
from hdgfem.io.time_series import DiagnosticPanel, TimeSeries, draw_time_series_panel, numeric_time_series, plot_diagnostic_panels
from scripts.guiding_center.cases.guiding_center_presets import preset_by_key
from scripts.guiding_center.diagnostics.diocotron_diagnostics import DiocotronModeDiagnostics
from scripts.guiding_center.runtime.arguments import build_parser
from scripts.guiding_center.runtime.configuration import _runtime_config
from scripts.guiding_center.runtime.diagnostic_plots import diagnostic_panels, render_guiding_center_diagnostic_plots


def test_instability_is_the_equilibrium_potential_norm_on_loglog_axes():
    times = np.array([0., .2, .7, 1.8, 5.])
    norms = 2e-4*np.exp(.19*times)
    rows = [{"time": t, "diocotron_phi_eq_l2": value, "energy_relative_drift": (-1)**i*1e-7}
            for i, (t, value) in enumerate(zip(times, norms))]
    panels = diagnostic_panels(rows)
    assert panels[0].title == "Diocotron instability"
    assert panels[0].xscale == panels[0].yscale == "log"
    np.testing.assert_array_equal(panels[0].series[0].values, norms)
    assert panels[1].xscale == "linear" and panels[1].yscale == "log"
    assert next(panel for panel in panels if "drift" in panel.title).yscale == "symlog"


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


def test_modal_recorder_reports_modes_beyond_seed_harmonics():
    diagnostic = DiocotronModeDiagnostics.__new__(DiocotronModeDiagnostics)
    diagnostic.space = object()
    diagnostic.mode, diagnostic.backend, diagnostic.xp = 3, "host", np
    diagnostic.angular_points = 128
    diagnostic.radii = np.array([.2, .5])
    diagnostic.weights = np.array([.25, .25])
    diagnostic.radius = .99
    diagnostic.modes = tuple(range(1, 64))
    diagnostic.mode_ids = np.asarray(diagnostic.modes)
    diagnostic.equilibrium = np.zeros((2, 128))
    theta = 2*np.pi*np.arange(128)/128
    diagnostic.sample = lambda field: np.tile(np.cos(17*theta)+.1*np.cos(3*theta), (2, 1))
    row = diagnostic.measure(SimpleNamespace(space=diagnostic.space))
    assert row["diocotron_dominant_mode_1"] == 17
    assert row["diocotron_dominant_mode_2"] == 3
    assert row["diocotron_active_mode_count"] == 2


def test_runner_flag_and_figures_include_instability_and_all_modes(tmp_path):
    key = "diocotron_gaussian_m64_ark3_p6_h008_dt005_t70"
    args = build_parser().parse_args(["--preset", key, "--save-diagnostics"])
    config = replace(_runtime_config(preset_by_key(key), args), diagnostics_dir=str(tmp_path))
    assert config.save_diagnostics and not config.plot_diagnostics and config.diocotron_diagnostics
    times = [0., .5, 1., 2.]
    rows = [
        {"time": t, "diocotron_phi_eq_l2": 1e-4*np.exp(.2*t),
         "diocotron_phi_mode_3_l2": 1e-4*np.exp(.1*t),
         "diocotron_phi_mode_64_l2": 1e-5*np.exp(2*t)}
        for t in times
    ]
    paths = render_guiding_center_diagnostic_plots(config, rows)
    assert any(file.name == "active_modes.png" for file in paths)
    assert all(file.stat().st_size > 1000 for file in paths)
    manifest = json.loads((paths[0].parent/"manifest.json").read_text())
    assert "||phi - phi_eq||" in manifest["instability"]


def test_invariants_remain_linear_and_repeated_stages_share_a_panel():
    rows = [{"time": t, "energy_from_q_l2": 1e-6+t*1e-12} for t in (0., 1., 2.)]
    timings = [{"time": t, "poisson_physical_residual": 1e-12,
                "stage1_poisson_physical_residual": 2e-12,
                "stage2_poisson_physical_residual": 3e-12} for t in (0., 1., 2.)]
    panels = diagnostic_panels(rows, timings)
    energy = next(panel for panel in panels if panel.filename == "energy_from_q_l2")
    assert energy.xscale == energy.yscale == "linear"
    residual = next(panel for panel in panels if panel.title == "poisson physical residual")
    assert len(residual.series) == 3
    assert residual.xscale == residual.yscale == "log"


def test_display_and_save_flags_are_independent(tmp_path, monkeypatch):
    from matplotlib import pyplot as plt

    key = "diocotron_gaussian_m64_ark3_p6_h008_dt005_t70"
    rows = [{"time": t, "diocotron_phi_eq_l2": 1e-4*(1+t),
             "diocotron_phi_mode_3_l2": 1e-5*(1+t),
             "diocotron_phi_mode_64_l2": 2e-5*(1+t)} for t in (.1, 1.)]
    for show, save in ((False, False), (False, True), (True, False), (True, True)):
        plt.close("all")
        calls = []
        monkeypatch.setattr(plt, "show", lambda **kwargs: calls.append((kwargs, len(plt.get_fignums()))))
        flags = (["--plot-diagnostics"] if show else []) + (["--save-diagnostics"] if save else [])
        args = build_parser().parse_args(["--preset", key, *flags])
        directory = tmp_path/f"show{show}_save{save}"
        config = replace(_runtime_config(preset_by_key(key), args), diagnostics_dir=str(directory))
        assert config.plot_diagnostics == show and config.save_diagnostics == save
        try:
            paths = render_guiding_center_diagnostic_plots(config, rows)
            assert bool(paths) == save
            assert directory.exists() == save
            assert bool(plt.get_fignums()) == show
            assert len(calls) == int(show)
            if show:
                assert calls[0][0] == {"block": True}
                assert calls[0][1] == 3  # Undefined mass, field dynamics, and modal activity.
            if save:
                assert all(path.exists() for path in paths)
        finally:
            plt.close("all")


def test_saved_conservation_panels_keep_signed_mass_and_recorded_drifts():
    rows = [
        {"time": 0., "mass": 100., "energy_relative_drift": 0., "mass_relative_drift": 0.},
        {"time": 1., "mass": 99., "energy_relative_drift": -.02, "mass_relative_drift": -.01},
        {"time": 2., "mass": 104., "energy_relative_drift": .03, "mass_relative_drift": .04},
    ]
    panels = {panel.filename: panel for panel in diagnostic_panels(rows)}
    energy = panels["relative_electric_energy_conservation"]
    mass = panels["relative_mass_conservation"]
    np.testing.assert_allclose(energy.series[0].values, [0, .02, .03])
    np.testing.assert_allclose(mass.series[0].values, [0, -.01, .04])
    assert energy.xscale == energy.yscale == "log"
    assert mass.xscale == "linear" and mass.yscale == "symlog"
    np.testing.assert_allclose(panels["energy_relative_drift"].series[0].values, [0, -.02, .03])


def test_saved_mass_conservation_uses_initial_mass_and_labels_zero_baseline():
    rows = [{"time": 0., "energy_from_q_l2": 2., "mass": 4.},
            {"time": 1., "energy_from_q_l2": 1.5, "mass": 5.}]
    panels = {panel.filename: panel for panel in diagnostic_panels(rows)}
    np.testing.assert_allclose(panels["relative_electric_energy_conservation"].series[0].values, [0, .25])
    np.testing.assert_allclose(panels["relative_mass_conservation"].series[0].values, [0, .25])
    for row in rows:
        row["mass_relative_drift"] = None
    mass = next(p for p in diagnostic_panels(rows) if p.filename == "relative_mass_conservation")
    np.testing.assert_allclose(mass.series[0].values, [0, .25])
    zero_mass = [{"time": 0., "mass": 0.}, {"time": 1., "mass": 1e-14}]
    mass = next(p for p in diagnostic_panels(zero_mass) if p.filename == "relative_mass_conservation")
    assert np.all(np.isnan(mass.series[0].values))
    assert "undefined" in mass.title


def test_display_conservation_is_unsigned_without_changing_saved_panels_or_rows(monkeypatch):
    import copy
    import scripts.guiding_center.runtime.diagnostic_plots as plotting

    rows = [{"time": 0., "mass": 2., "energy_relative_drift": 0., "enstrophy_relative_drift": 0.},
            {"time": 1., "mass": 1., "energy_relative_drift": -.02, "enstrophy_relative_drift": -.03},
            {"time": 2., "mass": 3., "energy_relative_drift": .04, "enstrophy_relative_drift": .05}]
    original = copy.deepcopy(rows)
    rendered = {}

    def capture(panels, output_dir=None, **kwargs):
        rendered[kwargs["display"]] = {p.filename: p for p in panels}
        return []

    monkeypatch.setattr(plotting, "plot_diagnostic_panels", capture)
    monkeypatch.setattr(plotting, "show_diagnostic_figures", lambda: None)
    config = replace(preset_by_key("euler_vortex_gas_si_bdf2_p6_h0068_dt005_t50"),
                     plot_diagnostics=True, save_diagnostics=True)
    plotting.render_guiding_center_diagnostic_plots(config, rows)
    for key, expected in (
        ("relative_mass_conservation", [0., .5, .5]),
        ("relative_electric_energy_conservation", [0., .02, .04]),
        ("enstrophy_relative_drift", [0., .03, .05]),
    ):
        panel = rendered[True][key]
        np.testing.assert_allclose(panel.series[0].values, expected)
        assert "signed" not in panel.title and "|" in panel.ylabel
    np.testing.assert_allclose(rendered[False]["relative_mass_conservation"].series[0].values, [0., -.5, .5])
    np.testing.assert_allclose(rendered[False]["enstrophy_relative_drift"].series[0].values, [0., -.03, .05])
    np.testing.assert_allclose(rendered[False]["energy_relative_drift"].series[0].values, [0., -.02, .04])
    assert rows == original


def test_display_unsigned_absolute_enstrophy_preserves_undefined_mass():
    from scripts.guiding_center.runtime.diagnostic_plots import interactive_diagnostic_panels

    rows = [{"time": 0., "mass": 0., "enstrophy_drift": 0.},
            {"time": 1., "mass": 1e-14, "enstrophy_drift": -.2}]
    saved = diagnostic_panels(rows)
    overview = {p.filename: p for p in interactive_diagnostic_panels(saved)}
    mass = overview["relative_mass_conservation"]
    assert np.all(np.isnan(mass.series[0].values)) and "undefined" in mass.title
    enstrophy = overview["enstrophy_drift"]
    np.testing.assert_allclose(enstrophy.series[0].values, [0., .2])
    assert enstrophy.ylabel == r"$|Z(t)-Z(0)|$"
    np.testing.assert_allclose(next(p for p in saved if p.filename == "enstrophy_drift").series[0].values, [0., -.2])


def test_named_pngs_and_manifest_identify_their_diagnostics(tmp_path):
    config = replace(preset_by_key("diocotron_gaussian_m64_ark3_p6_h008_dt005_t70"),
                     diagnostics_dir=str(tmp_path), save_diagnostics=True)
    rows = [{"time": 0., "diocotron_phi_eq_l2": 1e-5,
             "energy_relative_drift": 0., "mass_relative_drift": 0.},
            {"time": 1., "diocotron_phi_eq_l2": 2e-5,
             "energy_relative_drift": -.001, "mass_relative_drift": 1e-12}]
    paths = render_guiding_center_diagnostic_plots(config, rows)
    names = {path.name for path in paths}
    assert {"diocotron_instability.png", "conservation.png",
            "conservation_drifts.png", "diagnostics.pdf"} <= names
    assert not any(name.startswith("diagnostics_0") for name in names)
    manifest = json.loads((paths[0].parent/"manifest.json").read_text())
    assert set(manifest["figure_descriptions"]) == {name for name in names if name.endswith(".png")}


def test_png_names_are_safe_and_do_not_collide(tmp_path):
    series = (TimeSeries("value", np.array([0., 1.]), np.array([1., 2.])),)
    panels = [DiagnosticPanel("Mass / error", "Value", series),
              DiagnosticPanel("Mass : error", "Value", series)]
    paths = plot_diagnostic_panels(panels, tmp_path)
    assert {path.name for path in paths} == {"diagnostics.pdf", "mass_error.png", "mass_error_2.png"}


def test_coherent_groups_paginate_without_mixing_topics():
    from hdgfem.io.time_series import diagnostic_pages
    series = (TimeSeries("value", np.array([0., 1.]), np.array([1., 2.])),)
    panels = [DiagnosticPanel(f"Residual {i}", "Value", series, group="solver_residuals")
              for i in range(8)]
    panels.insert(2, DiagnosticPanel("Mass", "Value", series, group="conservation"))
    pages = diagnostic_pages(panels)
    assert [name for name, _ in pages] == [
        "solver_residuals_01.png", "solver_residuals_02.png", "conservation.png"]
    assert [len(page) for _, page in pages] == [6, 2, 1]
    assert all(len({panel.group for panel in page}) == 1 for _, page in pages)


def test_tau_history_and_latex_conservation_labels_render(tmp_path):
    from scripts.guiding_center.runtime.diagnostic_plots import poisson_tau_caption
    rows = [{"time": t, "poisson_tau": tau, "mass": 4.+t*.001,
             "energy_from_q_l2": 2.-t*.01}
            for t, tau in [(0., 1000.), (1., 2000.), (2., 2000.)]]
    timings = [{"time": t, "poisson_tau": 2000.} for t in (.5, 1., 1.5, 2.)]
    caption = poisson_tau_caption(rows, timings)
    assert r"\in[1000,\,2000]" in caption
    assert poisson_tau_caption(rows, [{"time": 1.}]) == caption
    assert "=2000" in poisson_tau_caption(timings)
    panels = {panel.filename: panel for panel in diagnostic_panels(rows, timings)}
    tau = panels["poisson_tau"]
    assert tau.group == "poisson_stabilization" and tau.drawstyle == "steps-post"
    np.testing.assert_array_equal(tau.series[0].times, [0., .5, 1., 1.5, 2.])
    np.testing.assert_array_equal(tau.series[0].values, [1000., 2000., 2000., 2000., 2000.])
    assert r"\tau" in tau.ylabel
    assert r"\frac" in panels["relative_mass_conservation"].ylabel
    assert r"\mathbf{E}" in panels["energy_from_q_l2"].ylabel
    paths = plot_diagnostic_panels(list(panels.values()), tmp_path, title=caption)
    assert {"conservation.png", "poisson_stabilization.png"} <= {path.name for path in paths}
    assert all(path.stat().st_size > 1000 for path in paths)


def test_tau_caption_does_not_invent_unrecorded_history():
    from scripts.guiding_center.runtime.diagnostic_plots import poisson_tau_caption
    assert "unrecorded" in poisson_tau_caption([{"time": 0.}])
    caption = poisson_tau_caption([
        {"time": 0., "run_configuration": {"poisson_tau_initial": 123.}}])
    assert "(0)=123" in caption and "later values unrecorded" in caption


def test_combined_save_display_finishes_saving_before_opening_windows(tmp_path, monkeypatch):
    import scripts.guiding_center.runtime.diagnostic_plots as plotting

    events = []
    def panels_stub(panels, output_dir=None, **kwargs):
        events.append("display" if kwargs["display"] else "save")
        return []
    monkeypatch.setattr(plotting, "plot_diagnostic_panels", panels_stub)
    monkeypatch.setattr(plotting, "show_diagnostic_figures", lambda: events.append("show"))
    config = replace(preset_by_key("diocotron_gaussian_m64_ark3_p6_h008_dt005_t70"),
                     diagnostics_dir=str(tmp_path), save_diagnostics=True, plot_diagnostics=True)
    plotting.render_guiding_center_diagnostic_plots(
        config, [{"time": 0., "mass": 1.}, {"time": 1., "mass": 1.}])
    assert events == ["save", "display", "show"]


def test_interactive_overview_is_bounded_and_keeps_important_physics():
    from hdgfem.io.time_series import diagnostic_pages
    from scripts.guiding_center.runtime.diagnostic_plots import interactive_diagnostic_panels

    rows = [
        {"time": t, "mass": 2., "enstrophy": 3., "energy_from_q_l2": 1.,
         "energy_relative_drift": t*1e-8, "mass_relative_drift": t*1e-12,
         "diocotron_phi_eq_l2": 1e-4*(1+t), "rho_min": .1, "rho_max": 1.,
         "q_l2": 1., "poisson_tau": 1000., "poisson_solver_iterations": 12,
         "poisson_physical_rel_residual": 1e-10, "transport_solver_rel_residual": 1e-9,
         "linear_step_wall_time": .4, "transport_time": .2, "poisson_time": .1,
         **{f"internal_counter_{i}": float(i+t) for i in range(500)}}
        for t in (0., 1., 2.)
    ]
    full = diagnostic_panels(rows)
    overview = interactive_diagnostic_panels(full)
    pages = diagnostic_pages(overview)
    assert len(pages) == 3  # A separately rendered modal figure can add one.
    assert all(len(page) <= 6 for _, page in pages)
    filenames = {panel.filename for panel in overview}
    assert {"relative_electric_energy_conservation", "relative_mass_conservation",
            "diocotron_instability", "poisson_tau"} <= filenames
    instability = next(p for p in overview if p.filename == "diocotron_instability")
    assert instability.xscale == instability.yscale == "log"
    assert len(diagnostic_pages(full)) > 4
    assert not any(p.filename and p.filename.startswith("internal_counter") for p in overview)
