from __future__ import annotations

from dataclasses import asdict

from scripts.advection_reaction.presets import PRESETS


def test_legacy_90k_pypardiso_ordering_presets_are_matched() -> None:
    unordered = PRESETS["test2_legacy_90k_pypardiso_none"]
    upwind = PRESETS["test2_legacy_90k_pypardiso_upwind_scc"]

    assert unordered.case == upwind.case == "test2"
    assert unordered.mesh_size == upwind.mesh_size == 0.01
    assert unordered.minimum_triangles == upwind.minimum_triangles == 90_000
    assert unordered.order == upwind.order == 6
    assert unordered.trace_basis == upwind.trace_basis == "legacy-lagrange"
    assert unordered.solver == upwind.solver == "pypardiso"
    assert unordered.preconditioner is upwind.preconditioner is None
    assert unordered.scale_system is upwind.scale_system is False
    assert unordered.trace_ordering == "none"
    assert upwind.trace_ordering == "upwind-scc"

    unordered_values = asdict(unordered)
    upwind_values = asdict(upwind)
    for key in ("description", "trace_ordering"):
        unordered_values.pop(key)
        upwind_values.pop(key)
    assert unordered_values == upwind_values
