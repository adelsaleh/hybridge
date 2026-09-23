"""Linear solvers, sparse-system helpers, and graph-ordering utilities."""

from .additive_schwarz import (
    FaceAdditiveSchwarzPreconditioner,
    build_face_additive_schwarz_preconditioner,
)
from .block_jacobi import (
    FaceBlockJacobiPreconditioner,
    build_face_block_jacobi_preconditioner,
)
from .gmres import (
    GMRESResult,
    restarted_gmres,
    solve_face_dense_gmres,
)

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
    "FaceAdditiveSchwarzPreconditioner",
    "FaceBlockJacobiPreconditioner",
    "GMRESResult",
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
    "build_face_additive_schwarz_preconditioner",
    "build_face_block_jacobi_preconditioner",
    "build_upwind_block_gs_preconditioner",
    "eliminate_known_dofs",
    "expand_known_dofs",
    "restarted_gmres",
    "save_sparse_pattern_plot",
    "save_upwind_reordered_matrix_patterns",
    "solve_face_dense_gmres",
    "solve_global_system",
    "sparse_pattern_marker_area",
    "strongly_connected_component_order",
    "upwind_scc_trace_ordering",
]
