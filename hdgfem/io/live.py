"""Reusable live scalar-field panels with lazy optional rendering dependencies."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .plot import (
    _require_pyvista,
    add_field_to_plotter,
    reference_plot_points,
    scalar_color_limits,
)


def simulation_frame_label(*, step, time_value, time_step=None, total_steps=None):
    """Describe the displayed state, independently of the preview cadence."""
    iteration = str(int(step))
    if total_steps is not None:
        iteration += f"/{int(total_steps)}"
    parts = [f"t = {float(time_value):.6g}"]
    if time_step is not None:
        parts.append(f"dt = {float(time_step):.6g}")
    parts.append(f"iteration {iteration}")
    return " | ".join(parts)


class PyVistaFieldPanels:
    """Update scalar DG panels on a fixed mesh without rebuilding VTK geometry.

    Each panel is ``(title, field)`` or ``(title, field, options)``. Options may
    set ``scalar_name``, ``cmap``, ``clim``, ``symmetric_clim``, ``fixed_clim``,
    and ``robust_percentile`` (95 by default). Explicit color limits remain
    fixed; ``fixed_clim=True`` freezes limits computed from the initial field.
    Updates require the original DGSpace for each panel. Sampling uses the
    normal DGField interface, including its host materialization for device
    fields; GPU-resident rendering is available separately through Holoviz.
    """

    def __init__(
        self, panels, *, resolution=20, title=None, shape=None, show_mesh=True,
        off_screen=False, window_size=(1600, 700), screenshot_dir=None,
        screenshot_prefix="fields", show_grid=False, time_step=None, total_steps=None,
    ):
        panels = tuple(panels)
        if not panels:
            raise ValueError("at least one panel is required")
        shape = (1, len(panels)) if shape is None else shape
        rows, columns = shape
        if rows < 1 or columns < 1 or rows * columns < len(panels):
            raise ValueError("shape does not have enough panels for all fields")
        self.reference_points = reference_plot_points(resolution)
        normalized = []
        for panel in panels:
            if len(panel) not in (2, 3):
                raise ValueError("each panel must be (title, field) or include an options dict")
            label, field = panel[:2]
            options = dict(panel[2] or {}) if len(panel) == 3 else {}
            values = field.values_at_ref(self.reference_points)
            limits = options.get("clim")
            if limits is None:
                limits = self._limits(values, options)
            normalized.append((label, field, options, values, limits))
        self.screenshot_dir = None if screenshot_dir is None else Path(screenshot_dir)
        if self.screenshot_dir is not None:
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_prefix = screenshot_prefix
        self.time_step, self.total_steps = time_step, total_steps
        self._shape = shape
        self.off_screen = bool(off_screen)
        pv = _require_pyvista()
        self.plotter = pv.Plotter(shape=shape, window_size=list(window_size), off_screen=off_screen)
        render_window = getattr(self.plotter, "render_window", None)
        window_name = (
            str(render_window.GetClassName())
            if render_window is not None and hasattr(render_window, "GetClassName") else ""
        )
        self._render_only = self.off_screen or any(
            marker in window_name for marker in ("EGL", "OSOpenGL", "Offscreen")
        )
        self._shown = self._closed = False
        self._panels = []
        self._panel_labels = [
            f"{title}\n{panel[0]}" if title else str(panel[0])
            for panel in normalized
        ]
        scalar_bar_args = {
            "vertical": False, "width": 0.55, "height": 0.08,
            "position_x": 0.225, "position_y": 0.02,
        }
        try:
            for index, (label, field, options, values, limits) in enumerate(normalized):
                scalar_name = options.get("scalar_name", field.name)
                mesh, actor = add_field_to_plotter(
                    self.plotter, field, reference_points=self.reference_points,
                    values=values, scalar_name=scalar_name, title=None,
                    subplot=(index // columns, index % columns), show_mesh=show_mesh,
                    cmap=options.get("cmap", "viridis"), clim=limits,
                    scalar_bar_args=scalar_bar_args, show_grid=show_grid,
                    title_position="upper_left", title_font_size=10, return_actor=True,
                )
                fixed = bool(options.get("fixed_clim")) or options.get("clim") is not None
                self._panels.append((field.space, mesh, actor, scalar_name, options, fixed))
            if len(panels) > 1:
                self.plotter.link_views()
        except BaseException:
            self.plotter.close()
            raise

    @staticmethod
    def _limits(values, options):
        return scalar_color_limits(
            values, percentile=options.get("robust_percentile", 95.0),
            symmetric=options.get("symmetric_clim", False),
        )

    def update(self, fields, *, step: int, time_value: float) -> None:
        """Replace panel samples in place, refresh, and optionally save a frame."""
        if self._closed:
            raise RuntimeError("cannot update closed panels")
        fields = tuple(fields)
        if len(fields) != len(self._panels):
            raise ValueError("fields must have the same length as panels")
        if any(field.space is not panel[0] for field, panel in zip(fields, self._panels)):
            raise ValueError("fixed plotting geometry requires the original DGSpace for each panel")
        for field, (_, mesh, actor, name, options, fixed) in zip(fields, self._panels):
            values = np.asarray(field.values_at_ref(self.reference_points)).reshape(-1)
            mesh.point_data[name][:] = values
            mesh.Modified()
            if not fixed:
                actor.mapper.scalar_range = self._limits(values, options)
        caption = simulation_frame_label(
            step=step, time_value=time_value,
            time_step=self.time_step, total_steps=self.total_steps,
        )
        for index in range(len(self._panels)):
            self.plotter.subplot(index // self._shape[1], index % self._shape[1])
            self.plotter.add_text(
                f"{self._panel_labels[index]}\n{caption}",
                position="upper_edge", font_size=10, shadow=False,
                name="simulation_progress", render=False,
            )
        if not self._shown:
            self.plotter.show(auto_close=False, interactive_update=not self._render_only)
            self._shown = True
        elif self._render_only:
            # EGL/OSMesa windows have no X event queue or matching interactor.
            self.plotter.render()
        else:
            self.plotter.update()
        if self.screenshot_dir is not None:
            path = self.screenshot_dir / (
                f"{self.screenshot_prefix}_step{int(step):05d}_t{float(time_value):.6f}.png"
            )
            self.plotter.screenshot(str(path))

    def close(self) -> None:
        """Release the render window and interactor; repeated closes are harmless."""
        if not self._closed:
            self.plotter.close()
            self._closed = True
