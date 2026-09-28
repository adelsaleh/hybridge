"""Streaming compressed movies from rendered uint8 RGBA frames."""
from pathlib import Path

import numpy as np


class MovieWriter:
    """Encode H.264 MP4 incrementally, retaining no frame history in memory."""

    def __init__(self, path, *, fps=20.):
        import imageio_ffmpeg

        self.path = Path(path)
        if self.path.suffix.lower() != ".mp4":
            raise ValueError("movie path must end in .mp4")
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("movie FPS must be finite and positive")
        self.fps = float(fps)
        self._encoder = None
        self._shape = None
        self._closed = False
        # Resolve the bundled executable now so missing dependencies fail early.
        imageio_ffmpeg.get_ffmpeg_exe()

    def append(self, image):
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
                               "-movflags", "+faststart"],
            )
            self._encoder.send(None)
        if image.shape != self._shape:
            raise ValueError("movie frame size changed during recording")
        self._encoder.send(np.ascontiguousarray(image))

    def close(self):
        if self._encoder is not None:
            self._encoder.close()
            self._encoder = None
        self._closed = True
