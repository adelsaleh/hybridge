"""Self-contained object-oriented DG helpers.

The :mod:`dgfem` package introduces explicit DG meshes, reference elements,
spaces, and fields. Its core mesh, quadrature, transfer, and HDG assembly
modules are local to the package so they can evolve independently from the
legacy solver modules.
"""

from .mesh import DGMesh, as_dg_mesh, gmsh_disc_mesh, gmsh_rectangle_mesh, gmsh_triangle_mesh, rectangle_mesh
from .space import DGField, DGSpace, VectorDGField, VectorDGSpace


def __getattr__(name: str):
    """Lazily expose solver symbols without pre-importing script modules."""
    if name in {"AdvectionReactionResult", "AdvectionReactionTimings", "solve_advection_reaction_hdg"}:
        from .adv_rea import AdvectionReactionResult, AdvectionReactionTimings, solve_advection_reaction_hdg

        symbols = {
            "AdvectionReactionResult": AdvectionReactionResult,
            "AdvectionReactionTimings": AdvectionReactionTimings,
            "solve_advection_reaction_hdg": solve_advection_reaction_hdg,
        }
        return symbols[name]
    if name in {"SolveResult", "solve_global_system"}:
        from .global_system import SolveResult, solve_global_system

        symbols = {
            "SolveResult": SolveResult,
            "solve_global_system": solve_global_system,
        }
        return symbols[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AdvectionReactionResult",
    "AdvectionReactionTimings",
    "DGField",
    "DGMesh",
    "DGSpace",
    "SolveResult",
    "VectorDGField",
    "VectorDGSpace",
    "as_dg_mesh",
    "gmsh_disc_mesh",
    "gmsh_rectangle_mesh",
    "gmsh_triangle_mesh",
    "rectangle_mesh",
    "solve_advection_reaction_hdg",
    "solve_global_system",
]
