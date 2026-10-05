"""Encode and decode synthetic images; no rendering or time integration."""
import numpy as np
import pytest

from hybridge.io.movie import MovieWriter


def test_movie_stream_round_trip(tmp_path):
    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    path = tmp_path / "movies" / "test.mp4"
    writer = MovieWriter(path, fps=20)
    for value in (30, 120, 220):
        frame = np.full((33, 35, 4), value, dtype=np.uint8)
        frame[..., 3] = 255
        writer.append(frame)
    writer.close()
    writer.close()
    reader = ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    meta = next(reader)
    assert meta["size"] == (36, 34)
    assert meta["fps"] == 20
    assert meta["codec"] == "h264"
    frames = list(reader)
    assert len(frames) == 3
    for raw, expected in zip(frames, (30, 120, 220)):
        image = np.frombuffer(raw, dtype=np.uint8).reshape(34, 36, 3)
        assert abs(float(image[:33, :35].mean()) - expected) < 5
    with pytest.raises(RuntimeError, match="closed"):
        writer.append(frame)


def test_movie_has_monotonic_presentation_without_b_frames(tmp_path):
    import re
    import subprocess
    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    path = tmp_path / "ordered.mp4"
    writer = MovieWriter(path, fps=20)
    for i in range(60):
        frame = np.zeros((64, 64, 4), dtype=np.uint8)
        frame[..., 3] = 255
        frame[:, i:i+3, :3] = 255
        writer.append(frame)
    writer.close()
    result = subprocess.run([ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path),
                             "-vf", "showinfo", "-f", "null", "-"],
                            capture_output=True, text=True, check=True)
    rows = [line for line in result.stderr.splitlines() if "pts_time:" in line]
    assert len(rows) == 60
    assert all("type:B" not in line for line in rows)
    times = [float(re.search(r"pts_time:([\d.]+)", line)[1]) for line in rows]
    np.testing.assert_allclose(times, np.arange(60)/20, atol=1e-6)
    data = path.read_bytes()
    assert b"moof" not in data
    assert data.index(b"moov") < data.index(b"mdat")
