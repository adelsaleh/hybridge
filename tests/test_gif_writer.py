"""Decode streamed real frames and verify an exact complete-file byte budget."""

import numpy as np
import pytest

from hybridge.io.movie import GifWriter


def frames():
    """Return changing images with a shared full-range grayscale palette."""
    x = np.arange(64, dtype=np.uint8)[None, :]
    y = np.arange(48, dtype=np.uint8)[:, None]
    for phase in (0, 30, 60):
        gray = (4*x + y + phase).astype(np.uint8)
        yield np.repeat(gray[:, :, None], 3, axis=2)


def test_stream_is_decodable_before_close_and_rejects_the_first_over_budget_frame(tmp_path):
    """Rejection retains every earlier frame, correct duration, and the GIF trailer."""
    Image = pytest.importorskip("PIL.Image")
    images = list(frames())
    with GifWriter(tmp_path / "reference.gif", fps=10) as reference:
        for frame in images[:2]:
            assert reference.append(frame)
    budget = reference.bytes_written
    with GifWriter(tmp_path / "bounded.gif", fps=10, max_bytes=budget) as writer:
        assert writer.append(images[0])
        with Image.open(writer.path) as gif:
            assert gif.n_frames == 1
            assert gif.size == (64, 48)
        assert writer.append(images[1])
        assert not writer.append(images[2])
        assert writer.limit_reached
        assert writer.frames_written == 2
        assert writer.bytes_written == writer.path.stat().st_size == budget
    writer.close()
    with Image.open(writer.path) as gif:
        assert gif.n_frames == 2
        assert gif.info["loop"] == 0
        for i in range(2):
            gif.seek(i)
            assert gif.info["duration"] == 100
            # Pillow's palette lookup uses a reduced RGB cache; allow its
            # maximum three-level rounding even for the grayscale fixture.
            np.testing.assert_allclose(np.asarray(gif.convert("RGB")), images[i], atol=3)
    with pytest.raises(RuntimeError, match="closed"):
        writer.append(images[0])


def test_gif_writer_validates_shape_and_opaque_frames(tmp_path):
    """Invalid inputs cannot corrupt the already published first frame."""
    pytest.importorskip("PIL.Image")
    with GifWriter(tmp_path / "shape.gif") as writer:
        assert writer.append(next(frames()))
        with pytest.raises(ValueError, match="size changed"):
            writer.append(np.zeros((10, 10, 3), dtype=np.uint8))
        with pytest.raises(ValueError, match="opaque"):
            writer.append(np.zeros((48, 64, 4), dtype=np.uint8))


@pytest.mark.parametrize("fps,durations_expected", [(24, {40, 50}), (48, {20, 30})])
def test_fractional_centisecond_cadence_preserves_the_requested_playback_speed(tmp_path, fps, durations_expected):
    """Both production frame rates retain one-second timing after GIF quantization."""
    Image = pytest.importorskip("PIL.Image")
    initial = next(frames())
    with GifWriter(tmp_path / "cadence.gif", fps=fps) as writer:
        for i in range(fps):
            assert writer.append(np.roll(initial, i, axis=1))
        assert writer.playback_ms == 1000
    with Image.open(writer.path) as gif:
        durations = []
        for i in range(gif.n_frames):
            gif.seek(i)
            durations.append(gif.info["duration"])
        assert gif.n_frames == fps
        assert sum(durations) == 1000
        assert set(durations) == durations_expected
