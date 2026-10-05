"""Output and plotting helpers."""

from hybridge.io.comparison import plot_sampled_solution_comparison
from hybridge.linalg.amgx.config import (
    describe_amgx_preconditioner,
    describe_amgx_solver,
    load_amgx_config,
)
from hybridge.io.holoviz import (
    HolovizScalarPanels,
    plot_field_holoviz,
    plot_fields_holoviz,
    plot_solution_comparison_holoviz,
)
from hybridge.io.live import AnalyticPanelField, DifferencePanelField, PyVistaFieldPanels
from hybridge.io.matplotlib import MatplotlibRasterPanels
from hybridge.io.movie import GifWriter
from hybridge.runtime.logging import format_elapsed_percent, timed_call
from hybridge.io.output import pretty_print_ncol
from hybridge.io.plot import (
    contour_levels_for_order,
    plot_field,
    plot_fields,
    plot_scalar_raster_panels_matplotlib,
    plot_solution_comparison,
    refined_field_polydata,
    resolve_exact_plot_resolution,
    resolve_field_plot_resolution,
    resolve_postprocessed_plot_resolution,
    sample_field_on_elements,
    scalar_color_limits,
)

from hybridge.io.time_series import DiagnosticPanel, TimeSeries, plot_diagnostic_panels, plot_mode_history

__all__ = [
    "DiagnosticPanel",
    "TimeSeries",
    "plot_diagnostic_panels",
    "plot_mode_history",
    "HolovizScalarPanels",
    "MatplotlibRasterPanels",
    "GifWriter",
    "AnalyticPanelField",
    "DifferencePanelField",
    "PyVistaFieldPanels",
    "contour_levels_for_order",
    "describe_amgx_preconditioner",
    "describe_amgx_solver",
    "format_elapsed_percent",
    "load_amgx_config",
    "plot_field",
    "plot_fields",
    "plot_field_holoviz",
    "plot_fields_holoviz",
    "plot_scalar_raster_panels_matplotlib",
    "plot_sampled_solution_comparison",
    "plot_solution_comparison",
    "plot_solution_comparison_holoviz",
    "pretty_print_ncol",
    "refined_field_polydata",
    "resolve_exact_plot_resolution",
    "resolve_field_plot_resolution",
    "resolve_postprocessed_plot_resolution",
    "sample_field_on_elements",
    "scalar_color_limits",
    "timed_call",
]
