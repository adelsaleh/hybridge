"""Package-level replacement for the legacy ``adv_rea_vec_msh4.py`` module.

The implementation lives in :mod:`dgfem.adv_rea`; this module keeps the old
solver name available inside the new package while avoiding duplicated solver
code.
"""

from .adv_rea import (
    AdvectionReactionResult,
    AdvectionReactionTimings,
    adv_rea_hdg_solv,
    solve_advection_reaction_hdg,
)

__all__ = [
    "AdvectionReactionResult",
    "AdvectionReactionTimings",
    "adv_rea_hdg_solv",
    "solve_advection_reaction_hdg",
]
