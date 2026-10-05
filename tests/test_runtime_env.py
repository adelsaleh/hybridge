"""HYBRIDGE_* settings with the deprecated pre-rename HDGFEM_* fallback."""

import warnings

import pytest

from hybridge.runtime.env import getenv


def test_new_name_wins_and_is_silent(monkeypatch):
    monkeypatch.setenv("HYBRIDGE_HOST_THREADS", "4")
    monkeypatch.setenv("HDGFEM_HOST_THREADS", "2")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert getenv("HOST_THREADS") == "4"


def test_legacy_name_is_honored_with_a_deprecation_warning(monkeypatch):
    monkeypatch.delenv("HYBRIDGE_HOST_THREADS", raising=False)
    monkeypatch.setenv("HDGFEM_HOST_THREADS", "2")
    with pytest.warns(DeprecationWarning, match="HDGFEM_HOST_THREADS is deprecated; set HYBRIDGE_HOST_THREADS"):
        assert getenv("HOST_THREADS") == "2"


def test_default_when_neither_name_is_set(monkeypatch):
    monkeypatch.delenv("HYBRIDGE_HOST_THREADS", raising=False)
    monkeypatch.delenv("HDGFEM_HOST_THREADS", raising=False)
    assert getenv("HOST_THREADS") is None
    assert getenv("HOST_THREADS", "8") == "8"
