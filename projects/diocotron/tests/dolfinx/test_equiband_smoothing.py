"""Relative smoothing defaults, physical scaling and legacy restart inputs.

These checks need no FEniCSx import. Numerical comparisons exercise the actual
window used for plots and the earlier script-side logistic implementation.
"""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from projects.diocotron.dolfinx.equiband.config import BandConfig, SolverConfig
from projects.diocotron.dolfinx.equiband.nonlinearities import Window
from projects.diocotron.dolfinx.torsion.initialization.frozen_frontier import logistic_window_and_threshold_derivatives


def test_default_ratio_and_width_replacement():
    band = BandConfig(threshold_width_delta=.003)
    assert SolverConfig().band.smoothing_mode == "relative_to_delta"
    assert band.smoothing_mode == "relative_to_delta"
    assert band.relative_epsilon == .08
    assert band.epsilon == pytest.approx(.00024)
    assert band.epsilon_over_delta == pytest.approx(.08)
    assert Window(band).peak == pytest.approx(.996146530673345)
    wider = replace(band, threshold_width_delta=.006)
    assert wider.epsilon == pytest.approx(.00048)
    assert Window(wider).peak == pytest.approx(Window(band).peak)
    sharper = replace(band, relative_epsilon=.04)
    assert sharper.epsilon == pytest.approx(.00012)
    for m in (.005, .02, .06):
        assert Window(band).value(m, m) == pytest.approx(Window(band).peak)
        assert band.epsilon == pytest.approx(.00024)


@pytest.mark.parametrize("kind", ["logistic", "mollified"])
def test_relative_window_is_the_same_absolute_window(kind):
    relative = BandConfig(.003, kind=kind, relative_epsilon=.08)
    absolute = BandConfig(.003, .00024, kind, smoothing_mode="absolute")
    q, m = np.linspace(-.003, .012, 2001), .005
    a, b = Window(relative), Window(absolute)
    for method in ("value", "derivative", "midpoint_derivative", "primitive"):
        np.testing.assert_allclose(getattr(a, method)(q, m), getattr(b, method)(q, m), atol=1e-12)
    assert a.peak == pytest.approx(b.peak)
    if kind == "mollified":
        assert a.peak == pytest.approx(1.)


def test_legacy_logistic_formula_has_not_changed():
    band = BandConfig(.003, relative_epsilon=.08)
    m = .011
    q = np.linspace(-.01, .03, 20001)
    lo, hi = band.thresholds(m)
    old, _, _, effective = logistic_window_and_threshold_derivatives(
        q, c1=lo, c2=hi, eps_mode="relative", eps_ratio=.08, eps_fixed=None)
    assert effective == pytest.approx(band.epsilon)
    np.testing.assert_allclose(Window(band).value(q, m), old, atol=3e-15)


@pytest.mark.parametrize("ratio", [0., -1., float("nan"), float("inf")])
def test_invalid_relative_ratio_is_rejected(ratio):
    with pytest.raises(ValueError, match="relative smoothing"):
        BandConfig(relative_epsilon=ratio)


def test_absolute_is_explicit_and_legacy_calls_keep_their_meaning():
    band = BandConfig(.003, .001)
    explicit = BandConfig(.003, .001, smoothing_mode="absolute")
    assert band == explicit
    assert band.relative_epsilon is None
    assert band.smoothing_mode == "absolute"
    assert Window(band).peak == pytest.approx(.6351489523872873)
    assert replace(band, threshold_width_delta=.006).epsilon == .001
    with pytest.raises(ValueError, match="positive"):
        BandConfig(smoothing_mode="absolute")
    with pytest.raises(ValueError, match="absent"):
        BandConfig(epsilon=.001, smoothing_mode="absolute", relative_epsilon=.08)
    with pytest.raises(ValueError, match="specify smoothing_mode"):
        BandConfig(epsilon=.001, relative_epsilon=.08)


@pytest.mark.parametrize("band", [BandConfig(.003), BandConfig(.003, .001)])
def test_resolved_configuration_round_trip_preserves_restart_hash(tmp_path, band):
    original = SolverConfig(band=band)
    path = tmp_path / "resolved.json"
    path.write_text(json.dumps({"schema_version": 2, **asdict(original)}))
    loaded = SolverConfig.load(path)
    assert loaded == original
    assert loaded.signature == original.signature


def test_old_epsilon_only_config_remains_absolute(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"band": {"threshold_width_delta": .003, "epsilon": .001}}))
    assert SolverConfig.load(path).band == BandConfig(.003, .001, smoothing_mode="absolute")


def test_missing_smoothing_uses_relative_default(tmp_path):
    path = tmp_path / "minimal.toml"
    path.write_text('[band]\nthreshold_width_delta = 0.003\n')
    assert SolverConfig.load(path).band == BandConfig(.003)


@pytest.mark.parametrize("epsilon", [.001, -1., float("nan"), float("inf")])
def test_conflicting_file_epsilon_is_not_silently_ignored(tmp_path, epsilon):
    path = tmp_path / "conflicting.json"
    path.write_text(json.dumps({"band": {"threshold_width_delta": .003,
        "smoothing_mode": "relative_to_delta", "relative_epsilon": .08, "epsilon": epsilon}}))
    with pytest.raises(ValueError, match="epsilon conflicts"):
        SolverConfig.load(path)
