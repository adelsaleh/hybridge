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
from .space import DGField, DGSpace, VectorDGField, VectorDGSpace

__all__ = [
    "DGField",
    "DGMesh",
    "DGSpace",
    "SmoothStarGeometry",
    "StructuredSizeOptions",
    "VectorDGField",
    "VectorDGSpace",
    "as_dg_mesh",
    "default_mesh_cache_dir",
    "gmsh_disc_mesh",
    "gmsh_lshape_mesh",
    "gmsh_rectangle_mesh",
    "gmsh_smooth_star_mesh",
    "gmsh_star_mesh",
    "gmsh_triangle_mesh",
    "gradient_weighted_indicator",
    "remesh_smooth_star_from_indicator",
    "rectangle_mesh",
    "structured_size_field_from_indicator",
]
