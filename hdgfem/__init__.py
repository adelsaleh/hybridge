"""Self-contained object-oriented DG helpers.

The :mod:`hdgfem` package exposes explicit DG meshes, reference elements,
spaces, fields, reusable HDG assembly helpers, sparse solvers, and executable
advection-reaction and diffusion-reaction solver modules.
"""

from hdgfem.core.mesh import (
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
from hdgfem.core.space import DGCoefficientLayout, DGField, DGSpace, DGTraceSpace, VectorDGField, VectorDGSpace
from hdgfem.core.geometry import MeshDomain
from hdgfem.core.projection import project_callable
from hdgfem.core.time_integration import bdf2_transport_data, bdf3_transport_data
from hdgfem.cases.profiles import sample_gaussian_blob_field
from hdgfem.core.element_coefficients import ElementCoefficient
from hdgfem.core.pointwise import PointwiseCoefficient, PointwiseLaw, pointwise_coefficient, pointwise_law
from hdgfem.core.field_ops import (
    coefficient_field,
    field_gradient_at_ref,
    field_linear_combination,
    field_values_at_ref,
    perpendicular_vector_field,
    project_callable_to_trace,
    project_field_to_trace,
    solution_field,
    solution_trace,
    trace_linear_combination,
    vector_field_linear_combination,
)
from hdgfem.core.trace_transfer import bernstein_degree_elevation_matrix, prolong_trace_coefficients
from hdgfem.diagnostics.errors import (
    ScalarComparisonSamples,
    ScalarErrorMetrics,
    ScalarHDGErrorMetrics,
    ScalarErrorReport,
    VectorComparisonSamples,
    VectorErrorMetrics,
    VectorErrorReport,
    evaluate_scalar_error,
    evaluate_hdg_scalar_error,
    evaluate_vector_error,
)
from hdgfem.diagnostics.guiding_center import (
    azimuthal_mode_diagnostics,
    guiding_center_field_diagnostics,
)
from hdgfem.diagnostics.solver import (
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
        from hdgfem import solvers

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
        from hdgfem.linalg.results import (
                    LinearSolveCapacityError,
                    LinearSolveConvergenceError,
                    LinearSolveError,
                    SolveResult,
                    SolveStatus,
                )
        from hdgfem.linalg.system import solve_global_system

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
    "MeshDomain",
    "bdf2_transport_data",
    "bdf3_transport_data",
    "project_callable",
    "sample_gaussian_blob_field",
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
    "ElementCoefficient",
    "PointwiseCoefficient",
    "PointwiseLaw",
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
    "field_gradient_at_ref",
    "field_linear_combination",
    "field_values_at_ref",
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
    "pointwise_coefficient",
    "pointwise_law",
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
