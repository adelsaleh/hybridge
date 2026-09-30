"""hdgfem.hdg.reference."""

from __future__ import annotations

import numpy as np
from hdgfem.core.space import DGSpace


def _reference_derivative_matrices(space: DGSpace) -> tuple[np.ndarray, np.ndarray]:
    """Return legacy-oriented reference derivative matrices."""
    q = space.quad_data
    d0 = np.einsum("q,iq,jq->ij", q.Krf_w, q.bas_of_quads, q.dbas_of_quads[0], optimize=True)
    d1 = np.einsum("q,iq,jq->ij", q.Krf_w, q.bas_of_quads, q.dbas_of_quads[1], optimize=True)
    return np.ascontiguousarray(d0.T), np.ascontiguousarray(d1.T)
