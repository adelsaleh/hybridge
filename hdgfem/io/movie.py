"""Streaming compressed movies and bounded GIFs from rendered images."""
from pathlib import Path

import numpy as np


class MovieWriter:
    """Encode H.264 MP4 incrementally, retaining no frame history in memory."""

    def __init__(self, path, *, fps=20., fragmented=False):
        """Validate the MP4 path and frame rate; the encoder starts on the first frame.

        ``fragmented`` writes a fragmented MP4, flushing a fragment at least every
        second of video, so it stays playable while it is written or after the
        process is killed (minus the last second); remux it
        with ``ffmpeg -c copy -movflags +faststart`` for web delivery. The default
        moves the index to the front on close, so the file is valid only once closed.
        """
        import imageio_ffmpeg

        self.path = Path(path)
        if self.path.suffix.lower() != ".mp4":
            raise ValueError("movie path must end in .mp4")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("movie FPS must be finite and positive")
        self.fps = float(fps)
        self.fragmented = bool(fragmented)
        self._encoder = None
        self._shape = None
        self._closed = False
        # Resolve the bundled executable now so missing dependencies fail early.
        imageio_ffmpeg.get_ffmpeg_exe()

    def append(self, image):
        """Encode one uint8 RGBA frame; every frame must match the first frame's size."""
        import imageio_ffmpeg

        if self._closed:
            raise RuntimeError("movie writer is closed")
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 4:
            raise ValueError("movie frames must be uint8 RGBA images")
        if self._encoder is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._shape = image.shape
            self._encoder = imageio_ffmpeg.write_frames(
                str(self.path), (image.shape[1], image.shape[0]), fps=self.fps,
                pix_fmt_in="rgba", pix_fmt_out="yuv420p", codec="libx264",
                macro_block_size=1, ffmpeg_log_level="error",
                output_params=["-crf", "23", "-preset", "veryfast", "-threads", "2",
                               "-bf", "0", "-g", str(max(1, round(self.fps * 2))),
                               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                               *(["-movflags", "+frag_keyframe+empty_moov+default_base_moof",
                                  "-frag_duration", "1000000", "-flush_packets", "1"]
                                 if self.fragmented else ["-movflags", "+faststart"])],
            )
            self._encoder.send(None)
        if image.shape != self._shape:
            raise ValueError("movie frame size changed during recording")
        self._encoder.send(np.ascontiguousarray(image))

    def close(self):
        """Finish the encoder and refuse further frames."""
        if self._encoder is not None:
            self._encoder.close()
            self._encoder = None
        self._closed = True


def concatenate_movies(segments, target):
    """Join MP4 segments from one writer without re-encoding; return the frame count.

    Segments must share size, frame rate and codec settings (``MovieWriter``
    output of one recording continued from a checkpoint). Timestamps restart
    from each segment's end, so the frame cadence stays uniform, and the joined
    frame count is checked against the sum of the segments.
    """
    import subprocess
    import tempfile

    import imageio_ffmpeg

    segments = [Path(path).resolve() for path in segments]
    target = Path(target)
    if target.suffix.lower() != ".mp4" or len(segments) < 2:
        raise ValueError("join at least two segments into an .mp4 target")
    expected = sum(imageio_ffmpeg.count_frames_and_secs(str(path))[0] for path in segments)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as listing:
        listing.writelines(f"file '{path}'\n" for path in segments)
    try:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat",
                        "-safe", "0", "-i", listing.name, "-c", "copy", "-movflags", "+faststart",
                        str(target)], check=True)
    finally:
        Path(listing.name).unlink()
    frames = imageio_ffmpeg.count_frames_and_secs(str(target))[0]
    if frames != expected:
        raise ValueError(f"joined movie has {frames} frames, segments have {expected}")
    return frames


class GifWriter:
    """Stream opaque frames with a fixed palette and an optional byte ceiling.

    The first frame defines a 256-color palette; colorbars in that frame cover
    the full display range. Frames are neither interpolated nor repeated.
    ``append`` returns false when the next real frame would exceed the budget.
    Only the current image and its encoded frame are retained in memory.
    """

    def __init__(self, path, *, fps=12., max_bytes=None):
        """Set the playback cadence and maximum complete-file size."""
        from PIL import Image

        self.path = Path(path)
        if self.path.suffix.lower() != ".gif":
            raise ValueError("GIF path must end in .gif")
        if not np.isfinite(fps) or fps <= 0 or fps > 100:
            raise ValueError("GIF FPS must be finite and lie in (0, 100]")
        if max_bytes is not None and (int(max_bytes) != max_bytes or max_bytes < 1):
            raise ValueError("max_bytes must be a positive integer")
        self.fps = float(fps)
        self.duration_ms = max(10, int(round(100 / fps)) * 10)
        if self.duration_ms > 655350:
            raise ValueError("GIF frame duration exceeds the format's 16-bit limit")
        self.max_bytes = None if max_bytes is None else int(max_bytes)
        self.frames_written = 0
        self.bytes_written = 0
        self.playback_ms = 0
        self.limit_reached = False
        self._palette = self._file = self._shape = None
        self._closed = False

    def append(self, image):
        """Append a real frame if it fits, keeping the file decodable throughout."""
        from PIL import GifImagePlugin, Image

        if self._closed:
            raise RuntimeError("GIF writer is closed")
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("GIF frames must be uint8 RGB or opaque RGBA images")
        if image.shape[2] == 4 and np.any(image[..., 3] != 255):
            raise ValueError("GIF frames must be opaque")
        shape = image.shape[:2]
        if self._shape is not None and shape != self._shape:
            raise ValueError("GIF frame size changed during recording")
        rgb = Image.fromarray(np.ascontiguousarray(image[..., :3]))
        first = self._palette is None
        if first:
            frame = rgb.quantize(colors=256, method=Image.Quantize.MEDIANCUT,
                                 dither=Image.Dither.NONE)
            header, _ = GifImagePlugin.getheader(frame, info={"loop": 0, "optimize": False})
        else:
            frame = rgb.quantize(palette=self._palette, dither=Image.Dither.NONE)
            header = []
        # GIF durations are integer centiseconds. Quantize cumulative time so
        # e.g. 24 FPS alternates 40/50 ms instead of drifting to 25 FPS.
        duration = (round((self.frames_written+1)*100/self.fps)
                    - round(self.frames_written*100/self.fps))*10
        blocks = GifImagePlugin.getdata(frame, duration=duration, disposal=1)
        added = sum(map(len, header)) + sum(map(len, blocks))
        new_size = (1 if first else self.bytes_written) + added
        if self.max_bytes is not None and new_size > self.max_bytes:
            self.limit_reached = True
            return False
        if first:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("wb")
            self._palette = frame.copy()
            self._shape = shape
        else:
            self._file.seek(-1, 2)  # Replace the previous GIF trailer.
        for block in (*header, *blocks):
            self._file.write(block)
        self._file.write(b";")
        self._file.flush()
        self.bytes_written = new_size
        self.frames_written += 1
        self.playback_ms += duration
        return True

    def close(self):
        """Close the already complete GIF; repeated close calls are harmless."""
        try:
            if self._file is not None:
                self._file.close()
        finally:
            self._closed = True

    def __enter__(self):
        """Return this streaming writer."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Finish the file without suppressing an error from the caller."""
        self.close()
        return False
