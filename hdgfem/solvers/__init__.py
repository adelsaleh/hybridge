"""Executable HDG solver modules and reusable solver classes."""


def __getattr__(name: str):
    """Lazily expose solver classes without importing every CLI module."""
    if name in {
        "AdvectionReactionHDGOptions",
        "AdvectionReactionHDGSolver",
        "AdvectionReactionResult",
        "AdvectionReactionTimings",
        "solve_advection_reaction_hdg",
    }:
        from .adv_rea import (
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
        from .diff_rea import (
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AdvectionReactionHDGOptions",
    "AdvectionReactionHDGSolver",
    "AdvectionReactionResult",
    "AdvectionReactionTimings",
    "DiffusionReactionHDGOptions",
    "DiffusionReactionHDGSolver",
    "DiffusionReactionResult",
    "DiffusionReactionTimings",
    "adv_rea",
    "diff_rea",
    "solve_advection_reaction_hdg",
    "solve_diffusion_reaction_hdg",
]
