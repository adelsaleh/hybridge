"""Sparse linear-system and graph-ordering helpers."""

from .ordering import (
    GraphOrderingDiagnostics,
    GraphOrderingResult,
    GraphOrderingTimings,
    LevelWidthDiagnostics,
    SparsePatternPlotResult,
    save_sparse_pattern_plot,
    save_upwind_reordered_matrix_patterns,
    sparse_pattern_marker_area,
    strongly_connected_component_order,
    upwind_scc_trace_ordering,
)
from .system import (
    KnownDofReduction,
    SolveResult,
    assemble_global_matrix,
    eliminate_known_dofs,
    expand_known_dofs,
    solve_global_system,
)
from .upwind_block_gs import (
    UpwindBlockGSPreconditioner,
    UpwindBlockGSStats,
    build_upwind_block_gs_preconditioner,
)

__all__ = [
    "GraphOrderingDiagnostics",
    "GraphOrderingResult",
    "GraphOrderingTimings",
    "KnownDofReduction",
    "LevelWidthDiagnostics",
    "SolveResult",
    "SparsePatternPlotResult",
    "UpwindBlockGSPreconditioner",
    "UpwindBlockGSStats",
    "assemble_global_matrix",
    "build_upwind_block_gs_preconditioner",
    "eliminate_known_dofs",
    "expand_known_dofs",
    "save_sparse_pattern_plot",
    "save_upwind_reordered_matrix_patterns",
    "solve_global_system",
    "sparse_pattern_marker_area",
    "strongly_connected_component_order",
    "upwind_scc_trace_ordering",
]
