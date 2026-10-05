"""Read-only diagnostics for an already active PyPardiso factorization.

No backend import or solver configuration is performed here. PyPardiso uses
one-based iparm numbers, unlike the C reference. Memory values below are MKL
estimates in KiB, not Python-process RSS or GPU allocations. See
docs/reference/pardiso_diagnostics.md for interpretation and limitations.
"""
from __future__ import annotations


def pardiso_factor_statistics(solver, *, matrix_nnz: int) -> dict:
    """Snapshot in-core factor memory, fill, pivots and refinement counters.

    Call after factorization, before freeing the factors. A nonpositive factor
    nonzero count is retained in raw_iparm but reported as unavailable (None).
    This can occur when reporting was disabled or an integer counter overflowed.
    """
    if matrix_nnz <= 0:
        raise ValueError("matrix_nnz must be positive")
    raw = {str(i): int(solver.get_iparm(i)) for i in (7, 14, 15, 16, 17, 18, 60)}
    symbolic, permanent, numerical = (raw[str(i)] for i in (15, 16, 17))
    in_core = raw["60"] == 0
    peak = max(symbolic, permanent + numerical) if in_core and min(symbolic, permanent, numerical) >= 0 else None
    factors = raw["18"] if raw["18"] > 0 else None
    stored = getattr(solver, "factorized_A", None)
    copy_bytes = sum(int(getattr(getattr(stored, name, None), "nbytes", 0))
                     for name in ("data", "indices", "indptr"))
    return dict(raw_iparm=raw, iparm_indexing="one_based", in_core=in_core,
                symbolic_peak_kib=symbolic, symbolic_permanent_kib=permanent,
                numerical_factors_kib=numerical, estimated_solver_peak_kib=peak,
                factor_nnz=factors, fill_ratio=factors / matrix_nnz if factors else None,
                wrapper_matrix_copy_bytes=copy_bytes,
                wrapper_uses_matrix_hash=isinstance(stored, str),
                perturbed_pivots=raw["14"], iterative_refinement_steps=raw["7"])


__all__ = ["pardiso_factor_statistics"]
