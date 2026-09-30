"""HDG assembly helpers and matrix-building backends."""

from hdgfem.assembly import hdg, hdg_gram, matrices_numpy
from hdgfem.core import projection

__all__ = ["hdg", "hdg_gram", "matrices_numpy", "projection"]
