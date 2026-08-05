"""Supported HDG solver API with lazy canonical and compatibility modules.

Use the package-level objects or the descriptive ``advection_reaction`` and
``diffusion_reaction`` modules in new code. The full-name modules own the implementations; abbreviated modules are
compatibility shims only.
"""

from importlib import import_module


_ADVECTION_EXPORTS = {
    "AdvectionReactionHDGOptions",
    "AdvectionReactionHDGSolver",
    "AdvectionReactionResult",
    "AdvectionReactionTimings",
    "solve_advection_reaction_hdg",
}
_DIFFUSION_EXPORTS = {
    "DiffusionReactionAssemblyResult",
    "DiffusionReactionHDGOptions",
    "DiffusionReactionHDGSolver",
    "DiffusionReactionResult",
    "DiffusionReactionTimings",
    "solve_diffusion_reaction_hdg",
}
_CANONICAL_MODULES = {
    "advection_reaction": ".advection_reaction",
    "diffusion_reaction": ".diffusion_reaction",
}
_COMPATIBILITY_MODULES = {
    "adv_rea": ".adv_rea",
    "diff_rea": ".diff_rea",
}


def __getattr__(name: str):
    """Lazily expose the supported solver API and its module facades."""
    if name in _ADVECTION_EXPORTS:
        module = import_module(".advection_reaction", __name__)
    elif name in _DIFFUSION_EXPORTS:
        module = import_module(".diffusion_reaction", __name__)
    elif name in _CANONICAL_MODULES:
        module = import_module(_CANONICAL_MODULES[name], __name__)
        globals()[name] = module
        return module
    elif name in _COMPATIBILITY_MODULES:
        module = import_module(_COMPATIBILITY_MODULES[name], __name__)
        globals()[name] = module
        return module
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "AdvectionReactionHDGOptions",
    "AdvectionReactionHDGSolver",
    "AdvectionReactionResult",
    "AdvectionReactionTimings",
    "DiffusionReactionAssemblyResult",
    "DiffusionReactionHDGOptions",
    "DiffusionReactionHDGSolver",
    "DiffusionReactionResult",
    "DiffusionReactionTimings",
    "adv_rea",
    "advection_reaction",
    "diff_rea",
    "diffusion_reaction",
    "solve_advection_reaction_hdg",
    "solve_diffusion_reaction_hdg",
]
