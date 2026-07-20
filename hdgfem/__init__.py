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
from .core.space import DGField, DGSpace, VectorDGField, VectorDGSpace


def __getattr__(name: str):
    """Lazily expose solver symbols without pre-importing script modules."""
    if name in {
        "AdvectionReactionHDGOptions",
        "AdvectionReactionHDGSolver",
        "AdvectionReactionResult",
        "AdvectionReactionTimings",
        "solve_advection_reaction_hdg",
    }:
        from .solvers.adv_rea import (
            AdvectionReactionHDGOptions,
            AdvectionReactionHDGSolver,
            AdvectionReactionResult,
            AdvectionReactionTimings,
            solve_advection_reaction_hdg,
        )

        symbols = {
            "AdvectionReactionHDGOptions": AdvectionReactionHDGOptions,
            "AdvectionReactionHDGSolver": AdvectionReactionHDGSolver,
            "AdvectionReactionResult": AdvectionReactionResult,
            "AdvectionReactionTimings": AdvectionReactionTimings,
            "solve_advection_reaction_hdg": solve_advection_reaction_hdg,
        }
        return symbols[name]
    if name in {
        "DiffusionReactionHDGOptions",
        "DiffusionReactionHDGSolver",
        "DiffusionReactionResult",
        "DiffusionReactionTimings",
        "solve_diffusion_reaction_hdg",
    }:
        from .solvers.diff_rea import (
            DiffusionReactionHDGOptions,
            DiffusionReactionHDGSolver,
            DiffusionReactionResult,
            DiffusionReactionTimings,
            solve_diffusion_reaction_hdg,
        )

        symbols = {
            "DiffusionReactionHDGOptions": DiffusionReactionHDGOptions,
            "DiffusionReactionHDGSolver": DiffusionReactionHDGSolver,
            "DiffusionReactionResult": DiffusionReactionResult,
            "DiffusionReactionTimings": DiffusionReactionTimings,
            "solve_diffusion_reaction_hdg": solve_diffusion_reaction_hdg,
        }
        return symbols[name]
    if name in {"SolveResult", "solve_global_system"}:
        from .linalg.system import SolveResult, solve_global_system

        symbols = {
            "SolveResult": SolveResult,
            "solve_global_system": solve_global_system,
        }
        return symbols[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AdvectionReactionResult",
    "AdvectionReactionTimings",
    "AdvectionReactionHDGOptions",
    "AdvectionReactionHDGSolver",
    "DiffusionReactionHDGOptions",
    "DiffusionReactionHDGSolver",
    "DiffusionReactionResult",
    "DiffusionReactionTimings",
    "DGField",
    "DGMesh",
    "DGSpace",
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
