"""Live exact / numerical / error panels for the n-Gamma runner.

Six panels, one row per field ``n`` and ``Gamma``: the exact manufactured
solution at the current time, the numerical DG field, and their pointwise
difference. Two backends share this layout:

* ``pyvista`` (host path): :class:`hybridge.io.PyVistaFieldPanels` with the
  exact field evaluated pointwise through :class:`hybridge.io.AnalyticPanelField`
  and the error through :class:`hybridge.io.DifferencePanelField`;
* ``holoviz`` (device path): :class:`hybridge.io.HolovizScalarPanels`, sampling
  the device-resident DG fields and the exact solution
  (``DeviceRasterSampler.sample_callable(device=True)``) on the GPU, so no
  field is downloaded. Holoviz uses one colormap per window; each panel keeps
  its own (expanding) colour limits and the error panels are symmetric.

The exact solution is evaluated directly; nothing is projected.
"""
from __future__ import annotations

import os
from pathlib import Path

LABELS = ("n exact", "n_h", "n_h - n", "Gamma exact", "Gamma_h", "Gamma_h - Gamma")


def display_available(backend: str) -> bool:
    """Whether a window can be shown for ``backend`` (Holoviz also accepts Wayland)."""
    return bool(os.environ.get("DISPLAY")) or (backend == "holoviz" and bool(os.environ.get("WAYLAND_DISPLAY")))


def resolve_backend(backend: str, assembly_backend: str) -> str:
    """``auto`` selects Holoviz for the device path and PyVista for the host path."""
    if backend == "auto":
        return "holoviz" if assembly_backend == "raw-cuda" else "pyvista"
    if backend not in {"pyvista", "holoviz"}:
        raise ValueError("plot backend must be 'auto', 'pyvista' or 'holoviz'")
    return backend


class NGammaPanels:
    """Exact, numerical and error panels of ``n`` and ``Gamma`` updated in place."""

    def __init__(self, backend, space, case, *, title, off_screen=False, screenshot_dir=None,
                 screenshot_prefix="n_gamma", time_step=None, total_steps=None, show_mesh=False,
                 resolution=6, width=1500, height=900, max_fps=10., movie_path=None, movie_fps=20.):
        self.backend = backend
        self.space = space
        self.case = case
        self.off_screen = bool(off_screen)
        self._viewer = None
        self._options = dict(title=title, off_screen=off_screen, screenshot_dir=screenshot_dir,
                             screenshot_prefix=screenshot_prefix, time_step=time_step, total_steps=total_steps,
                             show_mesh=show_mesh, resolution=resolution, width=width, height=height,
                             max_fps=max_fps, movie_path=movie_path, movie_fps=movie_fps)
        self._limits = [None]*len(LABELS)

    def _exact(self, t):
        return (lambda a, b: self.case.density(a, b, t)), (lambda a, b: self.case.momentum(a, b, t))

    def _open_pyvista(self, density, momentum, t):
        from hybridge.io import AnalyticPanelField, DifferencePanelField, PyVistaFieldPanels
        o = self._options
        exact_n, exact_gamma = (AnalyticPanelField(self.space, f, name) for f, name in
                                zip(self._exact(t), ("n exact", "Gamma exact")))
        self._analytic = (exact_n, exact_gamma)
        fields = self._pyvista_fields(density, momentum)
        options = [{"scalar_name": LABELS[i]} for i in range(len(LABELS))]
        for index in (2, 5):
            options[index].update(cmap="RdBu_r", symmetric_clim=True)
        self._viewer = PyVistaFieldPanels(
            [(label, field, option) for label, field, option in zip(LABELS, fields, options)],
            resolution=o["resolution"], title=o["title"], shape=(2, 3), show_mesh=o["show_mesh"],
            off_screen=o["off_screen"], window_size=(o["width"], o["height"]),
            screenshot_dir=o["screenshot_dir"], screenshot_prefix=o["screenshot_prefix"],
            time_step=o["time_step"], total_steps=o["total_steps"])

    def _pyvista_fields(self, density, momentum):
        from hybridge.io import DifferencePanelField
        exact_n, exact_gamma = self._analytic
        return (exact_n, density, DifferencePanelField(density, exact_n, "n_h - n"),
                exact_gamma, momentum, DifferencePanelField(momentum, exact_gamma, "Gamma_h - Gamma"))

    def _open_holoviz(self):
        from hybridge.io import HolovizScalarPanels
        o = self._options
        # Holoviz sizes are per panel; the 2x3 grid keeps the requested total window size.
        self._viewer = HolovizScalarPanels(
            [self.space]*len(LABELS), LABELS, width=max(2, o["width"] // 3), height=max(2, o["height"] // 2),
            columns=3, title=o["title"],
            show_mesh=o["show_mesh"], off_screen=o["off_screen"], screenshot_dir=o["screenshot_dir"],
            screenshot_prefix=o["screenshot_prefix"], max_fps=o["max_fps"], time_step=o["time_step"],
            total_steps=o["total_steps"], movie_path=o["movie_path"], movie_fps=o["movie_fps"])

    def update(self, density, momentum, *, step: int, time_value: float) -> None:
        """Show the fields at ``time_value`` (the viewer is created on the first call)."""
        if self.backend == "pyvista":
            if self._viewer is None:
                self._open_pyvista(density, momentum, time_value)
            for panel, function in zip(self._analytic, self._exact(time_value)):
                panel.function = function
            self._viewer.update(self._pyvista_fields(density, momentum), step=step, time_value=time_value)
            return
        if self._viewer is None:
            self._open_holoviz()
        viewer = self._viewer
        # Ask first (as the guiding-center panels do): frames dropped by the FPS cap
        # or a busy queue then cost no sampling at all.
        now = viewer._accept_frame()
        if now is None:
            return
        images = []
        with viewer.cp.cuda.Device(viewer.device_id):
            samplers = viewer.samplers
            for row, (field, exact) in enumerate(zip((density, momentum), self._exact(time_value))):
                sampler = samplers[3*row]
                exact_values = sampler.sample_callable(exact, device=True)
                numerical = sampler.sample(field)
                for column, (values, symmetric) in enumerate(
                        ((exact_values, False), (numerical, False), (numerical - exact_values, True))):
                    index = 3*row + column
                    image, self._limits[index] = sampler.values_image(
                        values, symmetric=symmetric, limits=self._limits[index], expand_limits=True)
                    images.append(image)
        viewer._enqueue(images, now=now, step=step, time_value=time_value)

    def close(self, *, suppress_errors: bool = False) -> None:
        """Flush pending frames and release the window.

        ``suppress_errors`` is for interrupted runs: Holoscan stops its
        renderer on SIGINT, so the final flush may report a stopped renderer.
        """
        if self._viewer is not None:
            viewer, self._viewer = self._viewer, None
            try:
                viewer.close()
            except Exception:
                if not suppress_errors:
                    raise


def output_settings(backend: str, plot_every: int, *, off_screen: bool, screenshot_dir, default_dir: Path):
    """Follow the guiding-center convention: headless PyVista saves frames; Holoviz saves only on request."""
    headless = plot_every > 0 and not display_available(backend)
    directory = screenshot_dir
    if headless and directory is None and backend == "pyvista":
        directory = str(default_dir)
    return bool(off_screen or headless), directory


__all__ = ["LABELS", "NGammaPanels", "display_available", "output_settings", "resolve_backend"]
