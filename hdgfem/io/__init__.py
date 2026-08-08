"""Output and plotting helpers."""

from .comparison import plot_sampled_solution_comparison
from .config import describe_amgx_preconditioner, describe_amgx_solver, load_amgx_config
from .output import format_elapsed_percent, pretty_print_ncol, timed_call
from .plot import (
    contour_levels_for_order,
    plot_field,
    plot_fields,
    plot_solution_comparison,
    refined_field_polydata,
    resolve_exact_plot_resolution,
    resolve_field_plot_resolution,
    resolve_postprocessed_plot_resolution,
    sample_field_on_elements,
)

__all__ = [
    "contour_levels_for_order",
    "describe_amgx_preconditioner",
    "describe_amgx_solver",
    "format_elapsed_percent",
    "load_amgx_config",
    "plot_field",
    "plot_fields",
    "plot_sampled_solution_comparison",
    "plot_solution_comparison",
    "pretty_print_ncol",
    "refined_field_polydata",
    "resolve_exact_plot_resolution",
    "resolve_field_plot_resolution",
    "resolve_postprocessed_plot_resolution",
    "sample_field_on_elements",
    "timed_call",
]
