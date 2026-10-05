"""Shared workstation MPI-rank policy for torsion numerical runs."""

from __future__ import annotations

import math
from collections.abc import Mapping


MUMPS_RANKS_BY_MESH_AND_ORDER: dict[float, dict[int, int]] = {
    0.30: {2: 1, 3: 1, 4: 1, 5: 1, 6: 1},
    0.15: {2: 1, 3: 1, 4: 1, 5: 2, 6: 4},
    0.075: {2: 2, 3: 4, 4: 4, 5: 4, 6: 4},
    0.05: {2: 4, 3: 4, 4: 8, 5: 8, 6: 8},
    0.03: {2: 4, 3: 8, 4: 8, 5: 12, 6: 16},
}


def select_mpi_ranks(
    mesh_size: float | None,
    order: int,
    dofs: int | None = None,
) -> int:
    """Return the AGENTS.md recommendation for one FE discretization."""
    if mesh_size is not None:
        for known_size, row in MUMPS_RANKS_BY_MESH_AND_ORDER.items():
            if math.isclose(
                float(mesh_size), known_size, rel_tol=0.0, abs_tol=5.0e-8
            ) and int(order) in row:
                return row[int(order)]
    if dofs is None:
        raise ValueError("dofs are required when the mesh-size/order pair is absent")
    if int(dofs) < 20_000:
        return 1
    if int(dofs) < 40_000:
        return 4 if int(order) >= 5 else 2
    if int(dofs) < 120_000:
        return 4
    if int(dofs) < 250_000:
        return 8
    return 8


def smaller_rank_within_ten_percent(median_seconds_by_rank: Mapping[int, float]) -> int:
    """Choose the smallest rank whose median is within 10% of the fastest."""
    usable = {
        int(rank): float(seconds)
        for rank, seconds in median_seconds_by_rank.items()
        if int(rank) > 0 and math.isfinite(float(seconds)) and float(seconds) > 0.0
    }
    if not usable:
        raise ValueError("at least one positive finite timing is required")
    fastest = min(usable.values())
    return min(rank for rank, seconds in usable.items() if seconds <= 1.10 * fastest)

