"""Persistent scalar raster panels with colorbars and CPU image capture."""

from __future__ import annotations

import numpy as np

from hdgfem.io.plot import plot_scalar_raster_panels_matplotlib


class MatplotlibRasterPanels:
    """Update existing raster artists while preserving masks and color scales.

    Panels and bounds follow ``plot_scalar_raster_panels_matplotlib``. Scalar
    arrays are host-side; callers may download a sampled device raster without
    transferring the complete DG field. Figure, axes, colorbars, and boundary
    outlines are built once. Capture returns an owning opaque RGBA image.
    """

    def __init__(self, panels, bounds, *, title="", size=(1600, 800), dpi=100,
                 boundary_mesh=None, font_size=14, background="white", ticks=True):
        """Create panels with colored coordinate boxes and white figure margins.

        ``ticks=False`` removes coordinate ticks and labels from the field
        panels, keeping their boxes; colorbars keep their ticks.
        """
        if len(size) != 2 or any(int(n) != n or n < 2 for n in size):
            raise ValueError("size must contain two positive pixel counts")
        if not np.isfinite(dpi) or dpi <= 0:
            raise ValueError("dpi must be finite and positive")
        if not np.isfinite(font_size) or font_size <= 0:
            raise ValueError("font_size must be finite and positive")
        import matplotlib as mpl

        with mpl.rc_context({"font.size": font_size, "axes.titlesize": font_size+2}):
            self.figure = plot_scalar_raster_panels_matplotlib(
                panels, bounds, suptitle=title or " ", share_clim=False, show=False,
                figsize=(size[0]/dpi, size[1]/dpi))
        color = mpl.colors.to_rgba(background)
        luminance = np.dot(color[:3], (.2126, .7152, .0722))
        foreground = "black"
        self.figure.set_facecolor("white")
        for ax in self.figure.axes:
            ax.set_facecolor(color if ax.images else "white")
            if ax.images and not ticks:
                ax.set(xticks=[], yticks=[], xlabel="", ylabel="")
            ax.tick_params(colors=foreground)
            for spine in ax.spines.values():
                spine.set_edgecolor(foreground)
        for text in self.figure.findobj(mpl.text.Text):
            text.set_color(foreground)
        self._caption = self.figure.suptitle(title or " ", fontsize=font_size+2,
                                            color=foreground)
        self.figure.set_dpi(dpi)
        self.images = tuple(image for ax in self.figure.axes for image in ax.images)
        self._shapes = tuple(image.get_array().shape for image in self.images)
        self._closed = False
        if boundary_mesh is not None:
            from matplotlib.collections import LineCollection

            segments = boundary_mesh.node_coords[
                boundary_mesh.edges[boundary_mesh.bnd_edges_inds]]
            for image in self.images:
                image.axes.add_collection(LineCollection(
                    segments, colors="#bfc5cb" if luminance < .5 else "#343a40", linewidths=.55))
        self.figure.canvas.draw()
        if hasattr(self.figure, "set_layout_engine"):
            self.figure.set_layout_engine(None)

    def update(self, values, *, caption=None):
        """Replace scalar values without rescaling or interpolating the fields."""
        if self._closed:
            raise RuntimeError("Matplotlib panels are closed")
        values = tuple(values)
        if len(values) != len(self.images):
            raise ValueError("one raster is required per panel")
        normalized = tuple(np.ma.masked_invalid(np.ma.asarray(value)) for value in values)
        if any(value.shape != shape for value, shape in zip(normalized, self._shapes)):
            raise ValueError("raster shape changed during recording")
        for image, value in zip(self.images, normalized):
            image.set_data(value)
        if caption is not None:
            self._caption.set_text(caption)

    def capture(self):
        """Draw on the CPU and return an independent uint8 RGBA image."""
        if self._closed:
            raise RuntimeError("Matplotlib panels are closed")
        self.figure.canvas.draw()
        return np.asarray(self.figure.canvas.buffer_rgba()).copy()

    def close(self):
        """Release the figure; repeated close calls are harmless."""
        if not self._closed:
            import matplotlib.pyplot as plt

            plt.close(self.figure)
            self._closed = True

    def __enter__(self):
        """Return the persistent panel figure."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Release the figure without suppressing a caller exception."""
        self.close()
        return False


__all__ = ["MatplotlibRasterPanels"]
