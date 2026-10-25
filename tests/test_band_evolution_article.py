"""Pure helpers of the E7 article script (no runs, no figures)."""
import numpy as np

from scripts.guiding_center.diagnostics.plot_band_evolution_article import longest_interval, panel_requests


def test_longest_interval_picks_the_longest_stretch():
    times = np.array([0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0])
    counts = np.array([5, 5, 7, 5, 5, 5, 8])
    assert longest_interval(times, counts == 5) == (7.5, 12.5)
    assert longest_interval(times, counts == 9) is None
    # a single saved state gives a degenerate interval; the table prints it as "t only"
    assert longest_interval(times, counts == 7) == (5.0, 5.0)
    # a stretch that reaches the last state is closed
    assert longest_interval(times, counts == 8) == (15.0, 15.0)


def test_panel_requests_parses_fields_times_and_stems():
    requests = list(panel_requests("star:seed:rho@0.05,phi@20;iter:control:rho@100:custom_stem"))
    assert requests == [("star", "seed", [("rho", 0.05), ("phi", 20.0)], "e7_star_seed"),
                        ("iter", "control", [("rho", 100.0)], "custom_stem")]
    assert list(panel_requests("")) == []
