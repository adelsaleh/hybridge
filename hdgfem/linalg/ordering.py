"""Graph orderings for sparse HDG trace systems.

The routines in this module are intentionally algebraic: they operate on mesh
edge ids and directed graph edges.  They do not change mesh connectivity or HDG
assembly.  The main use case is to build an experimental trace-DOF permutation
for advection-dominated HDG systems before calling an incomplete factorization.

The implementation is serial for now.  Numba is used for the graph passes so
the code remains fast enough for benchmark experiments, but the algorithms are
kept simple and readable:

1. Build a directed edge graph from element-local inflow/outflow faces.
2. Compute strongly connected components with Kosaraju's algorithm.
3. Collapse SCCs to a DAG and topologically order the DAG.
4. Lift the ordered mesh-edge blocks to trace-DOF indices.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:  # pragma: no cover - fallback exists for environments without numba.
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


@dataclass(frozen=True)
class GraphOrderingTimings:
    """Wall-clock timings for the individual graph-ordering stages."""

    graph_pairs: float
    csr: float
    scc: float
    dag: float
    topological_order: float
    dof_permutation: float
    total: float


@dataclass(frozen=True)
class LevelWidthDiagnostics:
    """Topological level-width diagnostics for a condensation DAG."""

    num_levels: int
    max_width: int
    median_width: float
    mean_width: float
    top10_width_fraction: float
    widths: tuple[int, ...]


@dataclass(frozen=True)
class GraphOrderingDiagnostics:
    """Summary diagnostics for a trace-edge graph ordering."""

    num_nodes: int
    num_directed_edges: int
    num_components: int
    largest_component_size: int
    cyclic_components: int
    cyclic_nodes: int
    level_widths: LevelWidthDiagnostics
    timings: GraphOrderingTimings


@dataclass(frozen=True)
class GraphOrderingResult:
    """Permutation and diagnostics produced by an SCC trace ordering."""

    edge_order: np.ndarray
    dof_permutation: np.ndarray
    component_id: np.ndarray
    component_order: np.ndarray
    diagnostics: GraphOrderingDiagnostics


@dataclass(frozen=True)
class SparsePatternPlotResult:
    """Paths and sampling diagnostics for sparse-pattern plots."""

    before_path: Path
    after_path: Path
    shape: tuple[int, int]
    nnz: int
    plotted_nnz_before: int
    plotted_nnz_after: int
    marker_area: float
    max_plot_points: int


def _njit(*args, **kwargs):
    """Return ``numba.njit`` when available, otherwise a no-op decorator."""
    if njit is None:
        def decorator(function):
            """Return the decorated function unchanged when Numba is unavailable."""
            return function

        return decorator
    return njit(*args, **kwargs)


@_njit(cache=True)
def _build_upwind_edge_pairs_kernel(
        loc2glob_edge: np.ndarray,
        beta_dot_normal: np.ndarray,
        active_edge_mask: np.ndarray,
        old_to_active: np.ndarray,
        flux_tolerance: float,
):
    """Return active-node directed edges inferred from local inflow/outflow."""
    num_elements = loc2glob_edge.shape[0]
    num_face_quads = beta_dot_normal.shape[2]
    max_pairs = num_elements * 6
    sources = np.empty(max_pairs, dtype=np.int64)
    targets = np.empty(max_pairs, dtype=np.int64)
    count = 0
    face_flux = np.empty(3, dtype=np.float64)

    for element in range(num_elements):
        for face in range(3):
            total = 0.0
            for q in range(num_face_quads):
                total += beta_dot_normal[element, face, q]
            face_flux[face] = total / num_face_quads

        for source_face in range(3):
            source_edge = loc2glob_edge[element, source_face]
            if not active_edge_mask[source_edge]:
                continue
            if face_flux[source_face] >= -flux_tolerance:
                continue

            source_node = old_to_active[source_edge]
            for target_face in range(3):
                if target_face == source_face:
                    continue
                target_edge = loc2glob_edge[element, target_face]
                if not active_edge_mask[target_edge]:
                    continue
                if face_flux[target_face] <= flux_tolerance:
                    continue

                target_node = old_to_active[target_edge]
                if source_node != target_node:
                    sources[count] = source_node
                    targets[count] = target_node
                    count += 1

    return sources[:count].copy(), targets[:count].copy()


@_njit(cache=True)
def _build_csr_kernel(num_nodes: int, sources: np.ndarray, targets: np.ndarray):
    """Build CSR adjacency from directed edge arrays with duplicates allowed."""
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    for edge in range(sources.size):
        indptr[sources[edge] + 1] += 1
    for node in range(num_nodes):
        indptr[node + 1] += indptr[node]

    next_index = indptr.copy()
    indices = np.empty(targets.size, dtype=np.int64)
    for edge in range(sources.size):
        source = sources[edge]
        insert_at = next_index[source]
        indices[insert_at] = targets[edge]
        next_index[source] += 1
    return indptr, indices


@_njit(cache=True)
def _kosaraju_scc_kernel(
        num_nodes: int,
        indptr: np.ndarray,
        indices: np.ndarray,
        reverse_indptr: np.ndarray,
        reverse_indices: np.ndarray,
):
    """Compute strongly connected components with iterative Kosaraju DFS."""
    visited = np.zeros(num_nodes, dtype=np.uint8)
    finish_order = np.empty(num_nodes, dtype=np.int64)
    finish_count = 0
    stack = np.empty(num_nodes, dtype=np.int64)
    next_pos = np.empty(num_nodes, dtype=np.int64)

    for start in range(num_nodes):
        if visited[start] != 0:
            continue
        top = 0
        stack[top] = start
        next_pos[top] = indptr[start]
        visited[start] = 1

        while top >= 0:
            node = stack[top]
            pos = next_pos[top]
            end = indptr[node + 1]
            if pos < end:
                neighbor = indices[pos]
                next_pos[top] = pos + 1
                if visited[neighbor] == 0:
                    top += 1
                    stack[top] = neighbor
                    next_pos[top] = indptr[neighbor]
                    visited[neighbor] = 1
            else:
                finish_order[finish_count] = node
                finish_count += 1
                top -= 1

    component_id = np.full(num_nodes, -1, dtype=np.int64)
    component_sizes = np.zeros(num_nodes, dtype=np.int64)
    component_count = 0

    for order_index in range(num_nodes - 1, -1, -1):
        start = finish_order[order_index]
        if component_id[start] != -1:
            continue

        top = 0
        stack[top] = start
        component_id[start] = component_count
        size = 0

        while top >= 0:
            node = stack[top]
            top -= 1
            size += 1
            for pos in range(reverse_indptr[node], reverse_indptr[node + 1]):
                neighbor = reverse_indices[pos]
                if component_id[neighbor] == -1:
                    component_id[neighbor] = component_count
                    top += 1
                    stack[top] = neighbor

        component_sizes[component_count] = size
        component_count += 1

    return component_id, component_sizes[:component_count].copy()


@_njit(cache=True)
def _component_edges_kernel(
        component_id: np.ndarray,
        sources: np.ndarray,
        targets: np.ndarray,
):
    """Return directed edges between distinct SCCs."""
    comp_sources = np.empty(sources.size, dtype=np.int64)
    comp_targets = np.empty(sources.size, dtype=np.int64)
    count = 0
    for edge in range(sources.size):
        source_comp = component_id[sources[edge]]
        target_comp = component_id[targets[edge]]
        if source_comp != target_comp:
            comp_sources[count] = source_comp
            comp_targets[count] = target_comp
            count += 1
    return comp_sources[:count].copy(), comp_targets[:count].copy()


@_njit(cache=True)
def _topological_order_kernel(num_components: int, indptr: np.ndarray, indices: np.ndarray):
    """Topologically order the SCC condensation DAG."""
    indegree = np.zeros(num_components, dtype=np.int64)
    for edge in range(indices.size):
        indegree[indices[edge]] += 1

    queue = np.empty(num_components, dtype=np.int64)
    head = 0
    tail = 0
    for component in range(num_components):
        if indegree[component] == 0:
            queue[tail] = component
            tail += 1

    order = np.empty(num_components, dtype=np.int64)
    count = 0
    while head < tail:
        component = queue[head]
        head += 1
        order[count] = component
        count += 1
        for pos in range(indptr[component], indptr[component + 1]):
            target = indices[pos]
            indegree[target] -= 1
            if indegree[target] == 0:
                queue[tail] = target
                tail += 1

    return order[:count].copy()


@_njit(cache=True)
def _node_order_from_components_kernel(
        component_id: np.ndarray,
        component_order: np.ndarray,
        component_sizes: np.ndarray,
):
    """List nodes component-by-component, keeping original node order inside SCCs."""
    num_nodes = component_id.size
    num_components = component_order.size
    node_order = np.empty(num_nodes, dtype=np.int64)

    cursor = np.empty(num_components, dtype=np.int64)
    offset = 0
    for order_index in range(num_components):
        component = component_order[order_index]
        cursor[component] = offset
        offset += component_sizes[component]

    for node in range(num_nodes):
        component = component_id[node]
        insert_at = cursor[component]
        node_order[insert_at] = node
        cursor[component] += 1

    return node_order


@_njit(cache=True)
def _trace_dof_permutation_kernel(edge_order_positions: np.ndarray, edg_dof: int):
    """Lift an ordered list of edge-block positions to trace dof positions."""
    permutation = np.empty(edge_order_positions.size * edg_dof, dtype=np.int64)
    count = 0
    for edge_position in edge_order_positions:
        base = edge_position * edg_dof
        for local_dof in range(edg_dof):
            permutation[count] = base + local_dof
            count += 1
    return permutation


def sparse_pattern_marker_area(
        nnz: int,
        *,
        reference_nnz: int = 100_000,
        reference_area: float = 0.25,
        min_area: float = 0.002,
        max_area: float = 2.0,
) -> float:
    """Return a scatter marker area scaled inversely with sparse nnz.

    Matplotlib ``scatter(..., s=...)`` interprets ``s`` as marker area in
    points squared.  Scaling the area like ``1 / nnz`` keeps dense patterns
    from turning into a solid block while still making small matrices visible.
    """
    nnz = max(1, int(nnz))
    area = float(reference_area) * float(reference_nnz) / float(nnz)
    return max(float(np.clip(area, min_area, max_area)),0.1)


def _sample_sparse_coordinates(row: np.ndarray, col: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return stride-sampled sparse coordinates without allocating random indices."""
    max_points = int(max_points)
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if row.size <= max_points:
        return row, col
    step = int(np.ceil(row.size / max_points))
    return row[::step][:max_points], col[::step][:max_points]


def save_sparse_pattern_plot(
        matrix,
        output_path,
        *,
        title: str | None = None,
        max_plot_points: int = 2_000_000,
        marker_area: float | None = None,
        dpi: int = 250,
        figsize: tuple[float, float] = (8.0, 8.0),
):
    """Save a sparse matrix pattern plot and return basic plot diagnostics.

    Very large matrices are stride-sampled to ``max_plot_points`` plotted
    entries.  The marker area is still computed from the original matrix
    ``nnz``, so the visual thickness decreases as the true sparsity pattern
    becomes denser.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coo = matrix.tocoo(copy=False)
    rows, cols = _sample_sparse_coordinates(
        np.asarray(coo.row),
        np.asarray(coo.col),
        max_plot_points,
    )
    area = sparse_pattern_marker_area(coo.nnz) if marker_area is None else float(marker_area)
    sampled = rows.size != coo.nnz

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(cols, rows, s=area, c="black", marker="s", linewidths=0.0, rasterized=True)
    ax.set_xlim(-0.5, coo.shape[1] - 0.5)
    ax.set_ylim(coo.shape[0] - 0.5, -0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    if title is None:
        title = "Sparse matrix pattern"
    suffix = f"shape={coo.shape}, nnz={coo.nnz:,}, marker_area={area:.3g}"
    if sampled:
        suffix += f", plotted={rows.size:,}"
    ax.set_title(f"{title}\n{suffix}")
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return {
        "path": output_path,
        "shape": coo.shape,
        "nnz": int(coo.nnz),
        "plotted_nnz": int(rows.size),
        "marker_area": area,
        "sampled": sampled,
    }


def save_upwind_reordered_matrix_patterns(
        matrix,
        permutation: np.ndarray,
        output_dir,
        *,
        prefix: str = "trace_matrix",
        max_plot_points: int = 2_000_000,
        dpi: int = 250,
) -> SparsePatternPlotResult:
    """Save sparsity plots before and after an upwind trace permutation.

    Parameters
    ----------
    matrix
        Sparse matrix in the original trace ordering.
    permutation
        Symmetric permutation with the same convention as
        :func:`hdgfem.linalg.system.solve_global_system`, i.e.
        ``A_perm = A[permutation][:, permutation]``.
    output_dir
        Directory where ``*_before_upwind.png`` and ``*_after_upwind.png`` are
        written.
    prefix
        Filename prefix.
    max_plot_points
        Maximum number of nonzero coordinates plotted per figure.  Sampling is
        deterministic and stride-based.
    dpi
        Output image resolution.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = matrix.tocsr()
    permutation = np.asarray(permutation, dtype=np.int64)
    if permutation.shape != (matrix.shape[0],):
        raise ValueError(f"permutation must have shape ({matrix.shape[0]},); got {permutation.shape}")

    marker_area = sparse_pattern_marker_area(matrix.nnz)
    before = save_sparse_pattern_plot(
        matrix,
        output_dir / f"{prefix}_before_upwind.png",
        title="Before upwind SCC ordering",
        max_plot_points=max_plot_points,
        marker_area=marker_area,
        dpi=dpi,
    )
    after_matrix = matrix[permutation][:, permutation].tocsr()
    after = save_sparse_pattern_plot(
        after_matrix,
        output_dir / f"{prefix}_after_upwind.png",
        title="After upwind SCC ordering",
        max_plot_points=max_plot_points,
        marker_area=marker_area,
        dpi=dpi,
    )
    return SparsePatternPlotResult(
        before_path=before["path"],
        after_path=after["path"],
        shape=tuple(matrix.shape),
        nnz=int(matrix.nnz),
        plotted_nnz_before=int(before["plotted_nnz"]),
        plotted_nnz_after=int(after["plotted_nnz"]),
        marker_area=marker_area,
        max_plot_points=int(max_plot_points),
    )


def _as_active_edges(num_edges: int, active_edges: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return active edge ids, mask, and old-edge to active-node map."""
    if active_edges is None:
        edges = np.arange(num_edges, dtype=np.int64)
    else:
        edges = np.asarray(active_edges, dtype=np.int64)
        if edges.ndim != 1:
            raise ValueError("active_edges must be one-dimensional")
        if edges.size and (edges.min() < 0 or edges.max() >= num_edges):
            raise ValueError("active_edges contains edge ids outside the mesh")
    active_mask = np.zeros(num_edges, dtype=bool)
    active_mask[edges] = True
    old_to_active = np.full(num_edges, -1, dtype=np.int64)
    old_to_active[edges] = np.arange(edges.size, dtype=np.int64)
    return np.ascontiguousarray(edges), active_mask, old_to_active


def _dag_level_width_diagnostics(
        num_components: int,
        indptr: np.ndarray,
        indices: np.ndarray,
        component_order: np.ndarray,
        component_sizes: np.ndarray,
) -> LevelWidthDiagnostics:
    """Compute topological level widths for a condensation DAG."""
    if num_components == 0:
        return LevelWidthDiagnostics(
            num_levels=0,
            max_width=0,
            median_width=0.0,
            mean_width=0.0,
            top10_width_fraction=0.0,
            widths=(),
        )

    levels = np.zeros(num_components, dtype=np.int64)
    for component in component_order:
        source_level = levels[component]
        for pos in range(indptr[component], indptr[component + 1]):
            target = indices[pos]
            next_level = source_level + 1
            if levels[target] < next_level:
                levels[target] = next_level

    num_levels = int(levels.max()) + 1
    widths = np.zeros(num_levels, dtype=np.int64)
    for component in range(num_components):
        widths[levels[component]] += int(component_sizes[component])

    sorted_widths = np.sort(widths)[::-1]
    top_count = min(10, sorted_widths.size)
    total_nodes = int(np.sum(widths))
    top10_fraction = float(np.sum(sorted_widths[:top_count]) / total_nodes) if total_nodes else 0.0
    return LevelWidthDiagnostics(
        num_levels=num_levels,
        max_width=int(widths.max()) if widths.size else 0,
        median_width=float(np.median(widths)) if widths.size else 0.0,
        mean_width=float(np.mean(widths)) if widths.size else 0.0,
        top10_width_fraction=top10_fraction,
        widths=tuple(int(width) for width in widths),
    )


def strongly_connected_component_order(
        num_nodes: int,
        sources: np.ndarray,
        targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LevelWidthDiagnostics, dict[str, float]]:
    """Return node and component orders for a directed graph.

    Parameters
    ----------
    num_nodes
        Number of graph vertices.
    sources, targets
        Directed graph edge arrays.  Edge ``k`` points from
        ``sources[k]`` to ``targets[k]``.  Duplicate edges are accepted.

    Returns
    -------
    node_order
        Vertices ordered by SCC topological order.  Vertices inside one SCC
        keep their original order.
    component_id
        Component id for every vertex.
    component_order
        Topological order of SCC ids.
    component_sizes
        Number of vertices in each SCC.
    timings
        Timings for CSR, SCC, DAG, and topological phases.
    """
    sources = np.asarray(sources, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    if sources.shape != targets.shape:
        raise ValueError("sources and targets must have the same shape")
    if sources.ndim != 1:
        raise ValueError("sources and targets must be one-dimensional")
    if num_nodes < 0:
        raise ValueError("num_nodes must be nonnegative")
    if sources.size and (sources.min() < 0 or targets.min() < 0 or sources.max() >= num_nodes or targets.max() >= num_nodes):
        raise ValueError("graph edge arrays contain vertex ids outside num_nodes")

    timings: dict[str, float] = {}

    start = time.perf_counter()
    indptr, indices = _build_csr_kernel(num_nodes, sources, targets)
    reverse_indptr, reverse_indices = _build_csr_kernel(num_nodes, targets, sources)
    timings["csr"] = time.perf_counter() - start

    start = time.perf_counter()
    component_id, component_sizes = _kosaraju_scc_kernel(num_nodes, indptr, indices, reverse_indptr, reverse_indices)
    timings["scc"] = time.perf_counter() - start

    start = time.perf_counter()
    comp_sources, comp_targets = _component_edges_kernel(component_id, sources, targets)
    comp_indptr, comp_indices = _build_csr_kernel(component_sizes.size, comp_sources, comp_targets)
    timings["dag"] = time.perf_counter() - start

    start = time.perf_counter()
    component_order = _topological_order_kernel(component_sizes.size, comp_indptr, comp_indices)
    if component_order.size != component_sizes.size:
        raise RuntimeError("SCC condensation graph topological ordering failed")
    level_widths = _dag_level_width_diagnostics(
        component_sizes.size,
        comp_indptr,
        comp_indices,
        component_order,
        component_sizes,
    )
    node_order = _node_order_from_components_kernel(component_id, component_order, component_sizes)
    timings["topological_order"] = time.perf_counter() - start

    return node_order, component_id, component_order, component_sizes, level_widths, timings


def upwind_scc_trace_ordering(
        mesh,
        beta_dot_normal: np.ndarray,
        edg_dof: int,
        *,
        active_edges: np.ndarray | None = None,
        flux_tolerance: float = 0.0,
) -> GraphOrderingResult:
    r"""Build an upwind SCC ordering and trace-DOF permutation.

    Parameters
    ----------
    mesh
        :class:`hdgfem.core.mesh.DGMesh`-like object with ``loc2glob_edge`` and edge
        counts.
    beta_dot_normal
        Values of :math:`\beta_h\cdot n` on element-face quadrature points,
        with shape ``(num_elements, 3, num_face_quads)``.
    edg_dof
        Number of trace degrees of freedom per mesh edge.
    active_edges
        Optional global edge ids to order.  For boundary-eliminated systems,
        pass the non-boundary edge ids so the returned DOF permutation is in
        reduced-system coordinates.
    flux_tolerance
        Faces with mean normal flux in ``[-flux_tolerance, flux_tolerance]``
        are ignored when constructing directed inflow-to-outflow dependencies.

    Notes
    -----
    The graph node is a mesh edge block, not an individual trace dof.  Inside
    each ordered edge block, local trace dofs remain contiguous and in their
    original order.
    """
    total_start = time.perf_counter()
    beta_dot_normal = np.asarray(beta_dot_normal, dtype=np.float64)
    if beta_dot_normal.ndim != 3 or beta_dot_normal.shape[:2] != (mesh.num_tri, 3):
        raise ValueError(
            "beta_dot_normal must have shape "
            f"({mesh.num_tri}, 3, num_face_quads); got {beta_dot_normal.shape}"
        )
    if edg_dof <= 0:
        raise ValueError("edg_dof must be positive")

    active_edge_ids, active_mask, old_to_active = _as_active_edges(mesh.num_edg, active_edges)

    start = time.perf_counter()
    sources, targets = _build_upwind_edge_pairs_kernel(
        np.asarray(mesh.loc2glob_edge, dtype=np.int64),
        beta_dot_normal,
        active_mask,
        old_to_active,
        float(flux_tolerance),
    )
    graph_pairs_time = time.perf_counter() - start

    node_order, component_id, component_order, component_sizes, level_widths, timings = strongly_connected_component_order(
        active_edge_ids.size,
        sources,
        targets,
    )

    start = time.perf_counter()
    ordered_edges = np.ascontiguousarray(active_edge_ids[node_order], dtype=np.int64)
    dof_permutation = _trace_dof_permutation_kernel(node_order, int(edg_dof))
    dof_permutation_time = time.perf_counter() - start

    largest_component = int(component_sizes.max()) if component_sizes.size else 0
    cyclic_mask = component_sizes > 1
    graph_timings = GraphOrderingTimings(
        graph_pairs=graph_pairs_time,
        csr=timings["csr"],
        scc=timings["scc"],
        dag=timings["dag"],
        topological_order=timings["topological_order"],
        dof_permutation=dof_permutation_time,
        total=time.perf_counter() - total_start,
    )
    diagnostics = GraphOrderingDiagnostics(
        num_nodes=int(active_edge_ids.size),
        num_directed_edges=int(sources.size),
        num_components=int(component_sizes.size),
        largest_component_size=largest_component,
        cyclic_components=int(np.count_nonzero(cyclic_mask)),
        cyclic_nodes=int(np.sum(component_sizes[cyclic_mask])) if component_sizes.size else 0,
        level_widths=level_widths,
        timings=graph_timings,
    )
    return GraphOrderingResult(
        edge_order=ordered_edges,
        dof_permutation=np.ascontiguousarray(dof_permutation, dtype=np.int64),
        component_id=np.ascontiguousarray(component_id, dtype=np.int64),
        component_order=np.ascontiguousarray(component_order, dtype=np.int64),
        diagnostics=diagnostics,
    )


__all__ = [
    "GraphOrderingDiagnostics",
    "GraphOrderingResult",
    "GraphOrderingTimings",
    "LevelWidthDiagnostics",
    "SparsePatternPlotResult",
    "save_sparse_pattern_plot",
    "save_upwind_reordered_matrix_patterns",
    "sparse_pattern_marker_area",
    "strongly_connected_component_order",
    "upwind_scc_trace_ordering",
]
