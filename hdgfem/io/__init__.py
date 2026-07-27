"""Output and plotting helpers."""

from .output import pretty_print_ncol
from .plot import (
    plot_field,
    plot_fields,
    plot_solution_comparison,
    refined_field_polydata,
    resolve_exact_plot_resolution,
    sample_field_on_elements,
)

__all__ = [
    "plot_field",
    "plot_fields",
    "plot_solution_comparison",
    "pretty_print_ncol",
    "refined_field_polydata",
    "resolve_exact_plot_resolution",
    "sample_field_on_elements",
]
