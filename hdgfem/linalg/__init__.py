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
    LinearSolveCapacityError,
    LinearSolveConvergenceError,
    LinearSolveError,
    SolveStatus,
    SolveResult,
    assemble_global_matrix,
    clear_pypardiso_cache,
    eliminate_known_dofs,
    expand_known_dofs,
    scale_sparse_system,
    solve_global_system,
    solve_pypardiso_system,
)
from .upwind_block_gs import (
    UpwindBlockGSPreconditioner,
    UpwindBlockGSStats,
    build_upwind_block_gs_preconditioner,
)
from .upwind_block_gs_on_the_fly import (
    UpwindBlockGSOnTheFlyTimings,
    build_forward_upwind_block_gs_from_coo,
    build_forward_upwind_block_gs_from_ordered_block_coo,
)
from .upwind_block_gs_cupy import cupy_upwind_block_gs_from_host_preconditioner

__all__ = [
    "GraphOrderingDiagnostics",
    "GraphOrderingResult",
    "GraphOrderingTimings",
    "KnownDofReduction",
    "LinearSolveCapacityError",
    "LinearSolveConvergenceError",
    "LinearSolveError",
    "LevelWidthDiagnostics",
    "SolveResult",
    "SolveStatus",
    "SparsePatternPlotResult",
    "UpwindBlockGSPreconditioner",
    "UpwindBlockGSOnTheFlyTimings",
    "UpwindBlockGSStats",
    "assemble_global_matrix",
    "clear_pypardiso_cache",
    "build_forward_upwind_block_gs_from_coo",
    "build_forward_upwind_block_gs_from_ordered_block_coo",
    "build_upwind_block_gs_preconditioner",
    "cupy_upwind_block_gs_from_host_preconditioner",
    "eliminate_known_dofs",
    "expand_known_dofs",
    "scale_sparse_system",
    "save_sparse_pattern_plot",
    "save_upwind_reordered_matrix_patterns",
    "solve_global_system",
    "solve_pypardiso_system",
    "sparse_pattern_marker_area",
    "strongly_connected_component_order",
    "upwind_scc_trace_ordering",
]
