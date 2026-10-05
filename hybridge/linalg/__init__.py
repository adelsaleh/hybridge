"""Sparse linear-system and graph-ordering helpers."""

from hybridge.linalg.additive_schwarz import (
    FaceAdditiveSchwarzPreconditioner,
    build_face_additive_schwarz_preconditioner,
)
from hybridge.linalg.block_jacobi import (
    FaceBlockJacobiPreconditioner,
    build_face_block_jacobi_preconditioner,
)
from hybridge.linalg.gmres import GMRESResult, restarted_gmres, solve_face_dense_gmres

from hybridge.linalg.ordering import (
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
from hybridge.linalg.reduction import (
    KnownDofReduction,
    eliminate_known_dofs,
    expand_known_dofs,
)
from hybridge.linalg.results import (
    LinearSolveCapacityError,
    LinearSolveConvergenceError,
    LinearSolveError,
    SolveStatus,
    SolveResult,
    refine_host_linear_solution,
    scale_sparse_system,
)
from hybridge.linalg.system import assemble_global_matrix, solve_global_system
from hybridge.linalg.direct import (
    clear_pypardiso_cache,
    prepare_pypardiso_spd_matrix,
    solve_pypardiso_system,
)
from hybridge.linalg.upwind_block_gs import (
    UpwindBlockGSPreconditioner,
    UpwindBlockGSStats,
    build_upwind_block_gs_preconditioner,
)
from hybridge.linalg.upwind_block_gs_on_the_fly import (
    UpwindBlockGSOnTheFlyTimings,
    build_forward_upwind_block_gs_from_coo,
    build_forward_upwind_block_gs_from_ordered_block_coo,
)
from hybridge.linalg.gpu.upwind_block_gs import (
    cupy_upwind_block_gs_from_host_preconditioner,
)

__all__ = [
    "FaceAdditiveSchwarzPreconditioner",
    "FaceBlockJacobiPreconditioner",
    "GMRESResult",
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
    "build_face_additive_schwarz_preconditioner",
    "build_face_block_jacobi_preconditioner",
    "clear_pypardiso_cache",
    "build_forward_upwind_block_gs_from_coo",
    "build_forward_upwind_block_gs_from_ordered_block_coo",
    "build_upwind_block_gs_preconditioner",
    "cupy_upwind_block_gs_from_host_preconditioner",
    "eliminate_known_dofs",
    "expand_known_dofs",
    "prepare_pypardiso_spd_matrix",
    "refine_host_linear_solution",
    "scale_sparse_system",
    "restarted_gmres",
    "save_sparse_pattern_plot",
    "save_upwind_reordered_matrix_patterns",
    "solve_face_dense_gmres",
    "solve_global_system",
    "solve_pypardiso_system",
    "sparse_pattern_marker_area",
    "strongly_connected_component_order",
    "upwind_scc_trace_ordering",
]
