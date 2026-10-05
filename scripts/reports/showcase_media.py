"""Theme variants of recorded GPU showcase movies: dark recolor and transparent posters.

The recorder draws light frames (white margins, black labels, a grey fill
outside the domain). This module rebuilds that exact Matplotlib layout from a
run's metadata and the showcase mesh, then derives, without any solve:

* per-pixel coverage of the field rasters and of the wall outline;
* a dark recolor of every frame: labels, colorbars and the fill are redrawn in
  dark colors, while field pixels keep their recorded colors;
* light and dark posters on a transparent background, with a small play button.

Used by ``publish_gpu_showcase.py``; CPU only.
"""

from __future__ import annotations

from dataclasses import dataclass
import subprocess

import numpy as np

LUMA = np.array([.2126, .7152, .0722], dtype=np.float32)
BACKGROUND = {"dark": np.array([13, 17, 23], np.float32)}          # GitHub dark canvas
FOREGROUND = {"light": "#1f2328", "dark": "#e6edf3"}             # GitHub page text colors
OUTLINE = {"light": "#343a40", "dark": "#bfc5cb"}                # recorder wall colors per background
BUTTON = {"light": ((31, 35, 40, 128), (255, 255, 255, 242)),
          "dark": ((230, 237, 243, 56), (255, 255, 255, 242))}   # disc, triangle (RGBA)


def _rgb(color):
    import matplotlib as mpl

    return np.array(mpl.colors.to_rgb(color), np.float32) * 255


def tone(rgb):
    """Map luminance onto the dark theme: white -> background, black -> text."""
    lum = np.asarray(rgb, np.float32) @ LUMA
    return BACKGROUND["dark"] + (1 - lum / 255)[..., None] * (_rgb(FOREGROUND["dark"]) - BACKGROUND["dark"])


def panel_options(metadata):
    """Colormaps, limits and caption label of the recorded panels."""
    import matplotlib as mpl

    (r0, r1), (p0, p1) = metadata["color_limits"]
    if metadata["strength_mode"] == "balanced":
        return (dict(cmap="RdBu_r", clim=(r0, r1)), dict(cmap="RdBu_r", clim=(p0, p1)),
                "Two-species guiding-center plasma")
    density = mpl.colormaps["viridis"].with_extremes(under=metadata["negative_color"])
    density.colorbar_extend = "min"
    return (dict(cmap=density, clim=(r0, r1)), dict(cmap="cividis", clim=(p0, p1)),
            "Single-species guiding-center plasma")


@dataclass
class Layout:
    """The recorder's figure, rebuilt with masked fields, and its pixel layers."""
    metadata: dict
    mesh: object
    geometry: object
    alpha: np.ndarray        # field raster coverage
    beta: np.ndarray         # wall outline coverage
    interior: np.ndarray     # field-box pixels clear of the box frames
    title_rows: int          # rows holding the time caption
    dark_static: np.ndarray  # labels, colorbars and fill redrawn dark (RGB uint8)

    def panels(self, t):
        """A fresh recorder figure with masked fields and caption time ``t``."""
        from hybridge.io import MatplotlibRasterPanels

        rho, phi, label = panel_options(self.metadata)
        blank = np.ma.array(np.zeros((self.geometry.height, self.geometry.width)), mask=True)
        return MatplotlibRasterPanels(
            [(r"Charge density $\rho$", blank, rho), (r"Potential $\phi$", blank, phi)],
            self.geometry.bounds, title=f"{label} | p = 6 | t = {t:.2f}",
            size=(self.metadata["width"], self.metadata["height"]), boundary_mesh=self.mesh,
            background=self.metadata["plot_background"], font_size=self.metadata.get("font_size", 18.),
            ticks=False)


def _capture(panels):
    panels.figure.canvas.draw()
    return np.asarray(panels.figure.canvas.buffer_rgba()).astype(np.float32) / 255


def build_layout(metadata):
    """Rebuild the recorded layout and render its coverage and dark layers."""
    import matplotlib as mpl
    from hybridge.io.raster import RasterGeometry
    from scripts.reports.gpu_showcase_setup import showcase_mesh

    mesh = showcase_mesh(metadata["h"])
    size = metadata.get("raster_size", 720)
    geometry = RasterGeometry.from_mesh(mesh, size, size)
    layout = Layout(metadata, mesh, geometry, None, None, None, 0, None)
    panels = layout.panels(0.)
    fig = panels.figure
    height = metadata["height"]
    boxes = []
    for image in panels.images:
        x0, y0, x1, y1 = image.axes.get_window_extent().extents
        boxes.append((height - y1, height - y0, x0, x1))
    title = panels._caption.get_window_extent()
    layout.title_rows = int(np.ceil(height - title.y0)) + 3

    # Dark layer: same artists in dark colors, fields masked, caption blank.
    panels._caption.set_text(" ")
    fig.set_facecolor(tuple(BACKGROUND["dark"] / 255))
    for ax in fig.axes:
        ax.set_facecolor(tuple(tone(_rgb(metadata["plot_background"])) / 255) if ax.images
                         else tuple(BACKGROUND["dark"] / 255))
        ax.tick_params(colors=FOREGROUND["dark"])
        for spine in ax.spines.values():
            spine.set_edgecolor(FOREGROUND["dark"])
    for text in fig.findobj(mpl.text.Text):
        text.set_color(FOREGROUND["dark"])
    outlines = [c for image in panels.images for c in image.axes.collections]
    for c in outlines:
        c.set_color(tuple(tone(_rgb(OUTLINE["light"])) / 255))
    layout.dark_static = np.rint(_capture(panels)[..., :3] * 255).astype(np.uint8)

    # Coverage: valid raster pixels in black on white, then the outline alone.
    for image in panels.images:
        image.axes.set_facecolor("white")
        image.set_cmap(mpl.colors.ListedColormap(["black"]))
        mask = geometry.element_ids.reshape(geometry.height, geometry.width) < 0
        image.set_data(np.ma.array(np.zeros(mask.shape), mask=mask))
    for c in outlines:
        c.set_visible(False)
    layout.alpha = 1 - _capture(panels)[..., :3].mean(-1)
    for image in panels.images:
        image.set_visible(False)
    for c in outlines:
        c.set_visible(True)
        c.set_color("black")
    layout.beta = 1 - _capture(panels)[..., :3].mean(-1)
    panels.close()

    interior = np.zeros(layout.alpha.shape, bool)
    for r0, r1, c0, c1 in boxes:
        interior[int(np.ceil(r0)) + 2:int(r1) - 2, int(np.ceil(c0)) + 2:int(c1) - 2] = True
    layout.interior = interior
    return layout


def decoded_colors(layout, frame):
    """Fill and outline colors as decoded from the movie (YUV round trip)."""
    fill = _rgb(layout.metadata["plot_background"])
    empty = layout.interior & (layout.alpha == 0) & (layout.beta == 0)
    decoded_fill = np.median(frame[empty].astype(np.float32), axis=0)
    return decoded_fill, _rgb(OUTLINE["light"]) + decoded_fill - fill


class DarkRecolor:
    """Exact per-pixel recolor of light frames onto the dark layer.

    Inside the field boxes, ``new = old + beta (outline' - outline) +
    (1 - beta)(1 - alpha)(fill' - fill)``: Agg composites linearly, so pixels
    covered only by the field are unchanged; empty pixels get the dark fill.
    The time caption is tone mapped, which is exact for black text on white.
    """

    def __init__(self, layout, first_frame):
        F, O = decoded_colors(layout, first_frame)
        FD, ON = tone(_rgb(layout.metadata["plot_background"])), tone(_rgb(OUTLINE["light"]))
        i = layout.interior
        a, b = layout.alpha[i][:, None], layout.beta[i][:, None]
        self.layout = layout
        self.empty = (layout.alpha[i] == 0)[:, None]
        self.fixed = b * ON + (1 - b) * FD
        self.delta = b * (ON - O) + (1 - b) * (1 - a) * (FD - F)

    def __call__(self, frame):
        layout = self.layout
        out = layout.dark_static.astype(np.float32)
        t = layout.title_rows
        out[:t] = tone(frame[:t])
        i = layout.interior
        out[i] = np.where(self.empty, self.fixed, frame[i].astype(np.float32) + self.delta)
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def poster(layout, frame, t, theme):
    """Transparent poster: field from ``frame``, labels in the theme's text color."""
    panels = layout.panels(t)
    fig = panels.figure
    fig.patch.set_alpha(0)
    import matplotlib as mpl

    for ax in fig.axes:
        ax.set_facecolor("none")
        ax.tick_params(colors=FOREGROUND[theme])
        for spine in ax.spines.values():
            spine.set_edgecolor(FOREGROUND[theme])
            spine.set_visible(not ax.images)      # colorbar outlines stay, field boxes go
    for text in fig.findobj(mpl.text.Text):
        text.set_color(FOREGROUND[theme])
    for image in panels.images:
        image.set_visible(False)
        for c in image.axes.collections:
            c.set_color(OUTLINE[theme])
    static = _capture(panels)
    panels.close()
    F, O = decoded_colors(layout, frame)
    a = np.where(layout.interior, layout.alpha, 0)
    b = layout.beta
    w = ((1 - b) * a)[..., None]
    # Un-mix the field color from the recorded fill and outline.
    field = (frame.astype(np.float32) - b[..., None] * O - ((1 - b) * (1 - a))[..., None] * F)
    field = np.where(w > .25, field / np.maximum(w, 1e-6), frame)
    bottom = np.dstack([np.clip(field, 0, 255) / 255, a])
    ta, ba = static[..., 3:], bottom[..., 3:]
    alpha = ta + ba * (1 - ta)
    rgb = (static[..., :3] * ta + bottom[..., :3] * ba * (1 - ta)) / np.maximum(alpha, 1e-6)
    return add_play_button(np.rint(np.dstack([rgb, alpha]) * 255).astype(np.uint8), theme)


def add_play_button(rgba, theme, *, radius=42, pad=12, clearance=10, supersample=4):
    """Translucent play button in the bottom-left corner, clear of all content."""
    from PIL import Image, ImageDraw

    height, width = rgba.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    def clear(cx, cy):
        return not rgba[..., 3][(xx - cx) ** 2 + (yy - cy) ** 2 <= (radius + clearance) ** 2].any()
    cx, cy = pad + radius, height - pad - radius
    cx, cy = next((cx + s, cy - s) for s in range(0, 400, 2) if clear(cx + s, cy - s))
    disc, triangle = BUTTON[theme]
    k = supersample
    layer = Image.new("RGBA", (width * k, height * k), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    x, y, r = cx * k, cy * k, radius * k
    draw.ellipse((x - r, y - r, x + r, y + r), fill=disc)
    s = r * .42
    draw.polygon([(x - .55 * s, y - s), (x - .55 * s, y + s), (x + 1.1 * s, y)], fill=triangle)
    out = Image.alpha_composite(Image.fromarray(rgba, "RGBA"), layer.resize((width, height), Image.LANCZOS))
    return np.asarray(out)


def ffmpeg():
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def frames(path, width, height):
    """Decode an MP4 into RGB uint8 frames."""
    proc = subprocess.Popen([ffmpeg(), "-v", "error", "-i", str(path), "-f", "rawvideo",
                             "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    size = width * height * 3
    try:
        while len(chunk := proc.stdout.read(size)) == size:
            yield np.frombuffer(chunk, np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.wait()


class Encoder:
    """H.264 writer for RGB frames (CRF 18, slow preset, yuv420p, fast start)."""

    def __init__(self, path, width, height, fps):
        self.count = 0
        self.proc = subprocess.Popen(
            [ffmpeg(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-r", repr(float(fps)), "-i", "-", "-c:v", "libx264", "-crf", "18", "-preset", "slow",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def write(self, frame):
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.count += 1

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait():
            raise RuntimeError("ffmpeg failed to encode the movie")
        return self.count


def retime(source, target, speed):
    """Play a movie ``speed`` times faster without re-encoding a single frame."""
    subprocess.run([ffmpeg(), "-v", "error", "-y", "-itsscale", repr(1 / float(speed)), "-i", str(source),
                    "-c", "copy", "-movflags", "+faststart", str(target)], check=True)
