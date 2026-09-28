"""Guiding-center plotting helpers."""

from __future__ import annotations
import os
from pathlib import Path


class GuidingCenterPyVistaPanels:
    """Case labels and color policies for the shared live DG viewer."""

    def __init__(
        self, density_field, potential_field, *, resolution, title, show_mesh,
        off_screen, screenshot_dir, screenshot_prefix, include_potential=False,
        density_is_vorticity=False, time_step=None, total_steps=None,
    ):
        from hdgfem.io import PyVistaFieldPanels

        self.include_potential = bool(include_potential)
        density_options = {"scalar_name": "density"}
        if density_is_vorticity:
            density_options.update(
                scalar_name="density", cmap="RdBu_r", symmetric_clim=True,
                fixed_clim=True, robust_percentile=100.0,
            )
        panels = [("Density", density_field, density_options)]
        if self.include_potential:
            panels.append(("Potential", potential_field, {"scalar_name": "potential"}))
        base_size = (1500, 650) if self.include_potential else (820, 720)
        self.viewer = PyVistaFieldPanels(
            panels, resolution=max(2, int(resolution)), title=title, show_mesh=show_mesh,
            off_screen=off_screen, window_size=base_size,
            screenshot_dir=screenshot_dir, screenshot_prefix=screenshot_prefix,
            time_step=time_step, total_steps=total_steps,
        )

    def update(self, density_field, potential_field, *, step: int, time_value: float) -> None:
        fields = [density_field] + ([potential_field] if self.include_potential else [])
        self.viewer.update(fields, step=step, time_value=time_value)

    def close(self) -> None:
        self.viewer.close()


def _make_plotter(
        config, density_field, potential_field, *, title, off_screen,
        screenshot_dir, screenshot_prefix, density_is_vorticity: bool,
):
    """Construct only the selected optional visualization backend."""
    options = dict(
        title=title, show_mesh=config.plot_show_mesh, off_screen=off_screen,
        screenshot_dir=screenshot_dir, screenshot_prefix=screenshot_prefix,
        include_potential=config.plot_potential,
        time_step=config.dt, total_steps=config.num_steps,
        density_is_vorticity=density_is_vorticity,
    )
    if config.plot_backend == "holoviz":
        from hdgfem.io.holoviz import GuidingCenterHolovizPanels
        return GuidingCenterHolovizPanels(
            density_field, potential_field, width=config.plot_width, height=config.plot_height,
            max_fps=config.plot_max_fps, movie_path=config.movie_path if config.save_movie else None,
            movie_fps=config.movie_fps, **options,
        )
    return GuidingCenterPyVistaPanels(
        density_field, potential_field, resolution=config.plot_resolution, **options,
    )


def _plot_output_settings(config, output_stem):
    """Preserve PyVista's legacy headless saves; Holoviz saves only explicitly."""
    display = bool(os.environ.get("DISPLAY")) or (
        config.plot_backend == "holoviz" and bool(os.environ.get("WAYLAND_DISPLAY"))
    )
    headless = config.plot_every > 0 and not display
    directory = config.screenshot_dir
    if headless and directory is None and config.plot_backend == "pyvista":
        directory = str(Path(config.diagnostics_dir) / f"{output_stem}_frames")
    return headless, bool(config.plot_off_screen or headless), directory

