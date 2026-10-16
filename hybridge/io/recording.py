"""Record DG fields to MP4 movies through cached raster sampling.

:class:`RasterMovieRecorder` keeps one pixel-ownership map of the mesh
(:class:`~hybridge.io.raster.RasterGeometry`), samples fields on the GPU
(:class:`~hybridge.io.raster.DeviceRasterSampler`) or on the host (cached
sampling matrix), draws them in persistent Matplotlib panels
(:class:`~hybridge.io.matplotlib.MatplotlibRasterPanels`) and streams the
frames to an H.264 MP4 (:class:`~hybridge.io.movie.MovieWriter`). Only the
sampled rasters are copied to the host; nothing is interpolated across DG
faces. Case-specific panel choices (titles, color maps, limits, captions)
stay with the caller.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["RasterMovieRecorder"]


class RasterMovieRecorder:
    """Persistent raster panels streamed to an MP4.

    ``panels`` is a sequence of ``(title, options)`` with the Matplotlib panel
    options of :func:`hybridge.io.plot.plot_scalar_raster_panels_matplotlib`
    (``cmap``, ``clim``, ...). Each :meth:`frame` call takes one host raster
    per panel (see :meth:`sample`), optional per-panel color limits and a
    caption. ``device=None`` samples on the GPU when CuPy is importable.
    """

    def __init__(self, space, path, *, panels, width=1024, height=1024, size=(1600, 800), fps=10.,
                 title="", boundary_mesh=None, device=None, font_size=14, ticks=False,
                 save_first_frame=True):
        from hybridge.io.matplotlib import MatplotlibRasterPanels
        from hybridge.io.movie import MovieWriter
        from hybridge.io.raster import RasterGeometry

        self.space = space
        self.path = Path(path)
        self.geometry = RasterGeometry.from_mesh(space.mesh, int(width), int(height))
        self.shape = (self.geometry.height, self.geometry.width)
        self.mask = self.geometry.element_ids.reshape(self.shape) < 0
        if device is None:
            try:
                from hybridge.runtime.optional import require_cupy

                device = bool(require_cupy().cuda.runtime.getDeviceCount())
            except Exception:
                device = False
        self.device = bool(device)
        if self.device:
            from hybridge.io.raster import DeviceRasterSampler
            from hybridge.runtime.optional import require_cupy

            cp = require_cupy()
            self._cp = cp
            self._sampler = DeviceRasterSampler(space, self.geometry, device_id=cp.cuda.runtime.getDevice())
        else:
            self._matrix = self.geometry.sampling_matrix(space)
        self._other_samplers = {}  # id(space) -> (space, sampler): fields of other degrees on the same mesh
        blank = np.ma.array(np.zeros(self.shape), mask=self.mask)
        self.panels = MatplotlibRasterPanels(
            [(panel_title, blank, dict(options)) for panel_title, options in panels],
            self.geometry.bounds, title=title, size=tuple(int(n) for n in size),
            boundary_mesh=boundary_mesh, font_size=font_size, ticks=ticks)
        self.writer = MovieWriter(self.path, fps=fps)
        self.frames = 0
        self._save_first_frame = bool(save_first_frame)

    def _sampler_for(self, space):
        """The sampler of ``space``: the recorder's own, or one cached per other DG space on its mesh."""
        if space is self.space:
            return self._sampler if self.device else self._matrix
        entry = self._other_samplers.get(id(space))
        if entry is None or entry[0] is not space:
            space.assert_same_mesh(self.space)
            if self.device:
                from hybridge.io.raster import DeviceRasterSampler

                sampler = DeviceRasterSampler(space, self.geometry, device_id=self._cp.cuda.runtime.getDevice())
            else:
                sampler = self.geometry.sampling_matrix(space)
            entry = self._other_samplers[id(space)] = (space, sampler)
        return entry[1]

    def sample(self, field) -> np.ma.MaskedArray:
        """Masked host raster of a DG field on the recorder's mesh, of any degree (e.g. a raised-degree potential)."""
        sampler = self._sampler_for(field.space)
        if self.device:
            values = self._cp.asnumpy(sampler.sample(field))
        else:
            values = sampler @ np.asarray(field.coeffs).ravel()
        return np.ma.array(np.asarray(values, dtype=np.float64).reshape(self.shape), mask=self.mask)

    def frame(self, rasters, *, caption=None, limits=None) -> None:
        """Draw one frame; ``limits`` optionally sets per-panel color limits (None keeps them)."""
        rasters = tuple(rasters)
        if limits is not None:
            for image, clim in zip(self.panels.images, limits):
                if clim is not None:
                    image.set_clim(*clim)
        self.panels.update(rasters, caption=caption)
        image = self.panels.capture()
        self.writer.append(image)
        if self._save_first_frame and self.frames == 0:
            from PIL import Image

            Image.fromarray(image).save(self.path.with_suffix(".png"))
        self.frames += 1

    def close(self) -> None:
        """Finish the movie and release the figure."""
        self.writer.close()
        self.panels.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
