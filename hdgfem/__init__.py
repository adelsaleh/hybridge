"""Self-contained object-oriented DG helpers.

The :mod:`hdgfem` package exposes explicit DG meshes, reference elements,
spaces, fields, reusable HDG assembly helpers, sparse solvers, and executable
advection-reaction and diffusion-reaction solver modules.
"""

from .core.mesh import (
    DGMesh,
    as_dg_mesh,
    default_mesh_cache_dir,
    gmsh_disc_mesh,
    gmsh_lshape_mesh,
    gmsh_polygon_mesh,
    gmsh_geo_mesh,
    gmsh_rectangle_mesh,
    gmsh_smooth_star_mesh,
    gmsh_star_mesh,
    gmsh_triangle_mesh,
    rectangle_mesh,
)
from .core.space import DGCoefficientLayout, DGField, DGSpace, DGTraceSpace, VectorDGField, VectorDGSpace
from .core.field_ops import (
    coefficient_field,
    field_linear_combination,
    perpendicular_vector_field,
    project_callable_to_trace,
    project_field_to_trace,
    solution_field,
    solution_trace,
    trace_linear_combination,
    vector_field_linear_combination,
)
from .core.trace_transfer import bernstein_degree_elevation_matrix, prolong_trace_coefficients
from .diagnostics import (
    ScalarComparisonSamples,
    ScalarErrorMetrics,
    ScalarHDGErrorMetrics,
    ScalarErrorReport,
    VectorComparisonSamples,
    VectorErrorMetrics,
    VectorErrorReport,
    azimuthal_mode_diagnostics,
    evaluate_scalar_error,
    evaluate_hdg_scalar_error,
    evaluate_vector_error,
    guiding_center_field_diagnostics,
    relative_drift,
    result_transfer_time,
    solver_result_metrics,
)


_SOLVER_EXPORTS = {
    "AdvectionDiffusionReactionHDGOptions",
    "AdvectionDiffusionReactionHDGSolver",
    "AdvectionDiffusionReactionResult",
    "AdvectionDiffusionReactionTimings",
    "AdvectionReactionHDGOptions",
    "AdvectionReactionHDGSolver",
    "AdvectionReactionResult",
    "AdvectionReactionTimings",
    "DiffusionReactionAssemblyResult",
    "DiffusionReactionHDGOptions",
    "DiffusionReactionHDGSolver",
    "DiffusionReactionResult",
    "DiffusionReactionTimings",
    "GlobalLengthDiffusion",
    "ScaledUpwind",
    "automatic_domain_length",
    "compute_domain_length",
    "geometric_diffusion_tau",
    "mesh_domain_measures",
    "solve_advection_reaction_hdg",
    "solve_advection_diffusion_reaction_hdg",
    "solve_diffusion_reaction_hdg",
}


def __getattr__(name: str):
    """Lazily expose solver symbols without pre-importing script modules."""
    if name in _SOLVER_EXPORTS:
        from . import solvers

        value = getattr(solvers, name)
        globals()[name] = value
        return value
    if name in {
        "LinearSolveCapacityError",
        "LinearSolveConvergenceError",
        "LinearSolveError",
        "SolveResult",
        "SolveStatus",
        "solve_global_system",
    }:
        from .linalg.system import (
            LinearSolveCapacityError,
            LinearSolveConvergenceError,
            LinearSolveError,
            SolveResult,
            SolveStatus,
            solve_global_system,
        )

        symbols = {
            "LinearSolveCapacityError": LinearSolveCapacityError,
            "LinearSolveConvergenceError": LinearSolveConvergenceError,
            "LinearSolveError": LinearSolveError,
            "SolveResult": SolveResult,
            "SolveStatus": SolveStatus,
            "solve_global_system": solve_global_system,
        }
        value = symbols[name]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AdvectionDiffusionReactionHDGOptions",
    "AdvectionDiffusionReactionHDGSolver",
    "AdvectionDiffusionReactionResult",
    "AdvectionDiffusionReactionTimings",
    "AdvectionReactionResult",
    "AdvectionReactionTimings",
    "AdvectionReactionHDGOptions",
    "AdvectionReactionHDGSolver",
    "DiffusionReactionAssemblyResult",
    "DiffusionReactionHDGOptions",
    "DiffusionReactionHDGSolver",
    "DiffusionReactionResult",
    "DiffusionReactionTimings",
    "GlobalLengthDiffusion",
    "ScaledUpwind",
    "DGCoefficientLayout",
    "DGTraceSpace",
    "DGField",
    "DGMesh",
    "DGSpace",
    "LinearSolveCapacityError",
    "LinearSolveConvergenceError",
    "LinearSolveError",
    "SolveStatus",
    "SolveResult",
    "ScalarComparisonSamples",
    "ScalarErrorMetrics",
    "ScalarHDGErrorMetrics",
    "ScalarErrorReport",
    "VectorComparisonSamples",
    "VectorErrorMetrics",
    "VectorErrorReport",
    "VectorDGField",
    "VectorDGSpace",
    "automatic_domain_length",
    "compute_domain_length",
    "geometric_diffusion_tau",
    "mesh_domain_measures",
    "as_dg_mesh",
    "azimuthal_mode_diagnostics",
    "bernstein_degree_elevation_matrix",
    "coefficient_field",
    "default_mesh_cache_dir",
    "evaluate_scalar_error",
    "evaluate_hdg_scalar_error",
    "evaluate_vector_error",
    "guiding_center_field_diagnostics",
    "field_linear_combination",
    "gmsh_disc_mesh",
    "gmsh_lshape_mesh",
    "gmsh_polygon_mesh",
    "gmsh_geo_mesh",
    "gmsh_rectangle_mesh",
    "gmsh_smooth_star_mesh",
    "gmsh_star_mesh",
    "gmsh_triangle_mesh",
    "rectangle_mesh",
    "perpendicular_vector_field",
    "project_callable_to_trace",
    "project_field_to_trace",
    "prolong_trace_coefficients",
    "relative_drift",
    "result_transfer_time",
    "solve_advection_reaction_hdg",
    "solve_advection_diffusion_reaction_hdg",
    "solve_diffusion_reaction_hdg",
    "solve_global_system",
    "solution_field",
    "solution_trace",
    "solver_result_metrics",
    "trace_linear_combination",
    "vector_field_linear_combination",
]
