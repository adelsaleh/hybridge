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
    gmsh_rectangle_mesh,
    gmsh_smooth_star_mesh,
    gmsh_star_mesh,
    gmsh_triangle_mesh,
    rectangle_mesh,
)
from .core.space import DGCoefficientLayout, DGField, DGSpace, DGTraceSpace, VectorDGField, VectorDGSpace


_SOLVER_EXPORTS = {
    "AdvectionReactionHDGOptions",
    "AdvectionReactionHDGSolver",
    "AdvectionReactionResult",
    "AdvectionReactionTimings",
    "DiffusionReactionAssemblyResult",
    "DiffusionReactionHDGOptions",
    "DiffusionReactionHDGSolver",
    "DiffusionReactionResult",
    "DiffusionReactionTimings",
    "solve_advection_reaction_hdg",
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
        "LinearSolveConvergenceError",
        "LinearSolveError",
        "SolveResult",
        "SolveStatus",
        "solve_global_system",
    }:
        from .linalg.system import (
            LinearSolveConvergenceError,
            LinearSolveError,
            SolveResult,
            SolveStatus,
            solve_global_system,
        )

        symbols = {
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
    "AdvectionReactionResult",
    "AdvectionReactionTimings",
    "AdvectionReactionHDGOptions",
    "AdvectionReactionHDGSolver",
    "DiffusionReactionAssemblyResult",
    "DiffusionReactionHDGOptions",
    "DiffusionReactionHDGSolver",
    "DiffusionReactionResult",
    "DiffusionReactionTimings",
    "DGCoefficientLayout",
    "DGTraceSpace",
    "DGField",
    "DGMesh",
    "DGSpace",
    "LinearSolveConvergenceError",
    "LinearSolveError",
    "SolveStatus",
    "SolveResult",
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
    "rectangle_mesh",
    "solve_advection_reaction_hdg",
    "solve_diffusion_reaction_hdg",
    "solve_global_system",
]
