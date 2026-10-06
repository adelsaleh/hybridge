"""High-level scalar solution comparison plots."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from hybridge.diagnostics.errors import ScalarComparisonSamples
from hybridge.io.plot import (
    _require_pyvista,
    add_samples_to_plotter,
    contour_levels_for_order,
    plot_scalar_sample_panels_matplotlib,
    resolve_exact_plot_resolution,
    sample_callable_on_elements,
)


def _coerce_samples(samples) -> ScalarComparisonSamples:
    """Normalize a sample dataclass or legacy mapping into one sample object."""
    if isinstance(samples, ScalarComparisonSamples):
        return samples
    if not isinstance(samples, Mapping):
        raise TypeError("samples must be ScalarComparisonSamples or a compatible mapping")
    exact = samples.get("exact_values", samples.get("exact_values_for_error"))
    numerical = samples.get("numerical_values", samples.get("postprocessed_values"))
    if exact is None or numerical is None:
        raise ValueError("sample mapping requires numerical/postprocessed and exact values")
    return ScalarComparisonSamples(
        np.ascontiguousarray(samples["reference_points"], dtype=np.float64),
        np.ascontiguousarray(numerical, dtype=np.float64),
        np.ascontiguousarray(exact, dtype=np.float64),
    )


def plot_sampled_solution_comparison(
        mesh,
        exact_solution: Callable,
        samples: ScalarComparisonSamples | Mapping,
        *,
        numerical_resolution: int,
        exact_resolution: int | str | None = None,
        polynomial_order: int | None = None,
        postprocessed_samples: ScalarComparisonSamples | Mapping | None = None,
        title: str = "",
        show_mesh: bool = True,
        show: bool = True,
        off_screen: bool = False,
        backend: str = "auto",
        show_error: bool = True,
):
    """Plot numerical, exact, and error panels from backend-generated samples.

    ``show_error=False`` leaves out the absolute-error panel. ``backend`` is ``"matplotlib"`` (discontinuous per-element contours),
    ``"pyvista"``, or ``"auto"`` (Matplotlib up to 130 triangles).
    """
    if backend not in ("auto", "matplotlib", "pyvista"):
        raise ValueError("backend must be 'auto', 'matplotlib' or 'pyvista'")
    primary = _coerce_samples(samples)
    post = None if postprocessed_samples is None else _coerce_samples(postprocessed_samples)
    displayed = primary if post is None else post
    displayed_error = displayed.absolute_error
    exact_panel_resolution = resolve_exact_plot_resolution(
        exact_resolution,
        numerical_resolution=numerical_resolution,
        num_elements=mesh.num_tri,
    )
    exact_reference_points, _, exact_display_values = sample_callable_on_elements(
        mesh, exact_solution, resolution=exact_panel_resolution,
    )
    if backend == "matplotlib" or (backend == "auto" and mesh.num_tri <= 130):
        panels = [("Numerical solution", primary.reference_points, primary.numerical_values)]
        if post is not None:
            panels.append(("Postprocessed primal", post.reference_points, post.numerical_values))
        panels.append(("Exact solution", exact_reference_points, exact_display_values, {"show_mesh": False}))
        if show_error:
            panels.append((
                "Absolute error" if post is None else "Postprocessed absolute error",
                displayed.reference_points,
                displayed_error,
                {"cmap": "magma", "zero_min": True},
            ))
        order = (
            int(polynomial_order) + (1 if post is not None else 0)
            if polynomial_order is not None
            else max((int(numerical_resolution) - 3) // 2, 0)
        )
        return plot_scalar_sample_panels_matplotlib(
            mesh,
            panels,
            suptitle=title or None,
            show_mesh=show_mesh,
            cmap="jet",
            levels=contour_levels_for_order(order),
            share_clim=False,
            show=show,
        )

    finite = displayed_error.reshape(-1)
    finite = finite[np.isfinite(finite)]
    upper = float(np.percentile(finite, 95.0)) if finite.size else 1.0
    error_clim = (0.0, upper if np.isfinite(upper) and upper > 0.0 else 1.0)
    pv = _require_pyvista()
    panel_count = 2 + (post is not None) + bool(show_error)
    plotter = pv.Plotter(shape=(1, panel_count), window_size=[600 * panel_count, 650], off_screen=off_screen)
    scalar_bar_args = {
        "vertical": False,
        "width": 0.55,
        "height": 0.08,
        "position_x": 0.225,
        "position_y": 0.02,
    }
    panels = [("Numerical solution", primary.reference_points, primary.numerical_values, None, "viridis", True)]
    if post is not None:
        panels.append(("Postprocessed primal", post.reference_points, post.numerical_values, None, "viridis", True))
    panels.append(("Exact solution", exact_reference_points, exact_display_values, None, "viridis", False))
    if show_error:
        panels.append((
            "Absolute error" if post is None else "Postprocessed absolute error",
            displayed.reference_points,
            displayed_error,
            error_clim,
            "magma",
            True,
        ))
    for column, (panel_title, reference_points, values, clim, cmap, panel_mesh) in enumerate(panels):
        display_title = panel_title if column != 0 or not title else f"{panel_title}\n{title}"
        add_samples_to_plotter(
            plotter,
            mesh,
            reference_points,
            values,
            scalar_name=f"field_{column}",
            title=display_title,
            subplot=(0, column),
            show_mesh=show_mesh and panel_mesh,
            cmap=cmap,
            clim=clim,
            scalar_bar_args=scalar_bar_args,
        )
    plotter.link_views()
    if show:
        plotter.show()
    return plotter


__all__ = ["plot_sampled_solution_comparison"]
