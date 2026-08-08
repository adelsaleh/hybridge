"""Core DG mesh, basis, quadrature, space, and transfer objects."""

from .adaptivity import (
    SmoothStarGeometry,
    StructuredSizeOptions,
    gradient_weighted_indicator,
    remesh_smooth_star_from_indicator,
    structured_size_field_from_indicator,
)
from .mesh import (
    DGMesh,
    as_dg_mesh,
    default_mesh_cache_dir,
    gmsh_disc_mesh,
    gmsh_lshape_mesh,
    gmsh_rectangle_mesh,
    gmsh_smooth_star_mesh,
    gmsh_star_mesh,
    gmsh_triangle_mesh,
    rectangle_mesh,
)
from .field_ops import (
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
from .trace_transfer import bernstein_degree_elevation_matrix, prolong_trace_coefficients
from .space import DGCoefficientLayout, DGField, DGSpace, DGTraceSpace, VectorDGField, VectorDGSpace

__all__ = [
    "DGCoefficientLayout",
    "DGTraceSpace",
    "DGField",
    "DGMesh",
    "DGSpace",
    "SmoothStarGeometry",
    "StructuredSizeOptions",
    "VectorDGField",
    "VectorDGSpace",
    "as_dg_mesh",
    "bernstein_degree_elevation_matrix",
    "coefficient_field",
    "default_mesh_cache_dir",
    "gmsh_disc_mesh",
    "gmsh_lshape_mesh",
    "gmsh_rectangle_mesh",
    "gmsh_smooth_star_mesh",
    "gmsh_star_mesh",
    "gmsh_triangle_mesh",
    "gradient_weighted_indicator",
    "field_linear_combination",
    "remesh_smooth_star_from_indicator",
    "rectangle_mesh",
    "perpendicular_vector_field",
    "project_callable_to_trace",
    "project_field_to_trace",
    "prolong_trace_coefficients",
    "solution_field",
    "solution_trace",
    "structured_size_field_from_indicator",
    "trace_linear_combination",
    "vector_field_linear_combination",
]
