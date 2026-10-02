"""Viewer lifecycle checks without a CUDA device or Holoscan installation."""

from types import SimpleNamespace
from contextlib import nullcontext

import pytest

from hdgfem.io.holoviz import HolovizScalarPanels


def _viewer(*, flush_error=None, movie_error=None):
    """Use the real close path with controlled renderer and encoder resources."""
    events = []
    viewer = HolovizScalarPanels.__new__(HolovizScalarPanels)
    viewer._closed = viewer._closing = False
    viewer._pending = viewer._inflight = None

    def flush():
        events.append("flush")
        if flush_error is not None:
            raise flush_error

    def finish_movie():
        events.append("movie")
        if movie_error is not None:
            raise movie_error

    viewer.flush = flush
    viewer._future = SimpleNamespace(
        done=lambda: True,
        result=lambda timeout: events.append("renderer"),
    )
    viewer._app = SimpleNamespace(
        shutdown_async_executor=lambda *, wait: events.append(("shutdown", wait)),
    )
    viewer._movie = SimpleNamespace(close=finish_movie)
    return viewer, events


def test_context_drains_renderer_before_finalizing_movie():
    viewer, events = _viewer()
    with viewer as opened:
        assert opened is viewer
        assert events == []
    assert events == ["flush", "renderer", ("shutdown", True), "movie"]
    assert viewer._closing and viewer._closed


def test_context_closes_movie_when_body_raises():
    viewer, events = _viewer()
    error = RuntimeError("solver failed")
    with pytest.raises(RuntimeError) as raised:
        with viewer:
            raise error
    assert raised.value is error
    assert events[-1] == "movie"
    assert viewer._closed


def test_context_propagates_cleanup_error_on_clean_exit():
    error = OSError("encoder failed")
    viewer, _ = _viewer(movie_error=error)
    with pytest.raises(OSError) as raised:
        with viewer:
            pass
    assert raised.value is error
    assert viewer._closed


def test_context_preserves_body_error_and_reports_cleanup_error():
    viewer, _ = _viewer(movie_error=OSError("encoder failed"))
    error = RuntimeError("solver failed")
    with pytest.raises(RuntimeError) as raised:
        with viewer:
            raise error
    assert raised.value is error
    if hasattr(error, "add_note"):
        assert error.__notes__ == ["Holoviz cleanup also failed: OSError('encoder failed')"]


def test_context_preserves_body_error_without_exception_note_support():
    """Python 3.10 exceptions lack add_note; cleanup must still preserve them."""
    class LegacyError(RuntimeError):
        add_note = None

    viewer, _ = _viewer(movie_error=OSError("encoder failed"))
    error = LegacyError("solver failed")
    with pytest.raises(LegacyError) as raised:
        with viewer:
            raise error
    assert raised.value is error
    assert viewer._closed


def test_explicit_close_inside_context_is_idempotent():
    viewer, events = _viewer()
    with viewer:
        viewer.close()
    assert events.count("movie") == 1
    assert events.count("flush") == 1


def test_flush_error_still_shuts_down_renderer_and_movie():
    error = TimeoutError("frame did not finish")
    viewer, events = _viewer(flush_error=error)
    with pytest.raises(TimeoutError) as raised:
        with viewer:
            pass
    assert raised.value is error
    assert events == ["flush", "renderer", ("shutdown", True), "movie"]
    assert viewer._closed


def test_failed_movie_finalization_does_not_repeat_on_close():
    viewer, events = _viewer(movie_error=OSError("encoder failed"))
    with pytest.raises(OSError):
        viewer.close()
    viewer.close()
    assert events.count("movie") == 1


@pytest.mark.parametrize("limits, expected", [
    ((-14., 14.), ((-14., 14.), (-14., 14.))),
    (((-14., 14.), (-.05, .05)), ((-14., 14.), (-.05, .05))),
    ((None, (0., .1)), (None, (0., .1))),
    (None, (None, None)),
])
def test_update_fields_applies_independent_or_shared_panel_ranges(limits, expected):
    """Potential and density need separate scales without CUDA in this check."""
    calls = []
    viewer = HolovizScalarPanels.__new__(HolovizScalarPanels)
    viewer.panel_count = 2
    viewer.device_id = 0
    viewer.cp = SimpleNamespace(cuda=SimpleNamespace(Device=lambda _: nullcontext()))
    viewer._accept_frame = lambda: 0.

    def sample(field, *, limits):
        calls.append((field, limits))
        return field, limits

    viewer._samplers = [SimpleNamespace(image=sample)] * 2
    viewer._enqueue = lambda images, now, **kwargs: images
    assert viewer.update_fields(("rho", "phi"), limits=limits) == ["rho", "phi"]
    assert [call[1] for call in calls] == list(expected)


def test_update_fields_rejects_wrong_number_of_panel_ranges_before_rendering():
    """A malformed scale specification never starts device sampling."""
    viewer = HolovizScalarPanels.__new__(HolovizScalarPanels)
    viewer.panel_count = 2
    with pytest.raises(ValueError, match="one color-limit pair"):
        viewer.update_fields(("rho", "phi"), limits=((0., 1.),))
