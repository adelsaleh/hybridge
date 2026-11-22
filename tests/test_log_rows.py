"""Timed log rows: aligned seconds, lines of other threads held back, failures closed."""
import threading

import pytest

from hybridge.linalg.amgx.host import _compact_amgx_banner
from hybridge.runtime import logging as log


@pytest.fixture(autouse=True)
def fresh_clock(monkeypatch):
    monkeypatch.setattr(log, "_log_clock_origin", None)


def test_seconds_form_one_column_for_rows_of_any_kind(capsys):
    log.timed_call("short", 1, lambda: None)
    log.timed_call("a much longer label than the first one", 1, lambda: None)
    log._timed_call("solver row", 1, lambda: None)
    log.log_row("finished row", 1.5)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    assert len({line.index(" s", log.LOG_TIME_COLUMN) for line in lines}) == 1
    assert lines[3].endswith("   1.5000 s")


def test_multiline_row_prints_the_label_first_and_the_seconds_after_the_inner_lines(capsys):
    result, elapsed = log.timed_call("work", 1, lambda: print("inner line") or 7, multiline=True)
    lines = capsys.readouterr().out.splitlines()
    assert result == 7 and elapsed >= 0.
    assert lines[0] == "work ..." and lines[1] == "inner line"
    assert lines[2].startswith("work ...") and lines[2].endswith(" s")


def test_silent_call_prints_nothing_and_returns_the_time(capsys):
    result, elapsed = log.timed_call("work", 0, lambda: 3)
    assert (result, capsys.readouterr().out) == (3, "") and elapsed >= 0.


def test_clock_column_starts_at_the_first_origin_only(capsys):
    log.start_log_clock(log.time.perf_counter() - 100.)
    log.start_log_clock(log.time.perf_counter())
    log.log_row("row", 0.)
    text = capsys.readouterr().out
    assert "t=  100." in text


def test_line_from_another_thread_waits_for_the_open_row(capsys):
    def other_thread():
        thread = threading.Thread(target=log.log_line, args=("[window] message",))
        thread.start()
        thread.join()

    log.timed_call("row", 1, other_thread)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("row ...") and lines[0].endswith(" s")
    assert lines[1] == "[window] message"


@pytest.mark.parametrize("error, note", [(KeyboardInterrupt, "interrupted"), (RuntimeError, "failed")])
def test_a_stopped_row_still_ends_its_line_and_shows_the_time(capsys, error, note):
    def stop():
        raise error()

    with pytest.raises(error):
        log.timed_call("row", 1, stop)
    assert capsys.readouterr().out.endswith(f"({note})\n")
    assert not log._row_open


def test_amgx_banner_is_one_line_without_the_plugin_notice():
    banner = _compact_amgx_banner([
        "AMGX version 2.5.0\nBuilt on today\nCompiled with CUDA Toolkit 13.1\n",
        "The AMGX_initialize_plugins API call is deprecated and can be safely removed.\n"])
    assert banner == "AMGX version 2.5.0; Built on today; Compiled with CUDA Toolkit 13.1\n"
    assert _compact_amgx_banner([]) == ""
