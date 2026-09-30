"""Correctness and determinism tests for adaptive upwind graph ordering."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from hdgfem.linalg.ordering import (
    _adaptive_component_order,
    _as_active_edges,
    _build_upwind_edge_pairs_kernel,
    strongly_connected_component_order,
    upwind_scc_trace_ordering,
)


def _assert_same_partition(left: np.ndarray, right: np.ndarray) -> None:
    """Assert two component label arrays describe the same partition."""
    np.testing.assert_array_equal(
        left[:, None] == left[None, :],
        right[:, None] == right[None, :],
    )


def _assert_component_topology(
        sources: np.ndarray,
        targets: np.ndarray,
        component_id: np.ndarray,
        component_order: np.ndarray,
) -> None:
    """Assert every inter-component edge points forward in component order."""
    positions = np.empty(component_order.size, dtype=np.int64)
    positions[component_order] = np.arange(component_order.size, dtype=np.int64)
    for source, target in zip(sources, targets, strict=True):
        source_component = component_id[source]
        target_component = component_id[target]
        if source_component != target_component:
            assert positions[source_component] < positions[target_component]


@pytest.mark.parametrize(
    ("num_nodes", "sources", "targets"),
    [
        (0, [], []),
        (5, [], []),
        (6, [0, 0, 1, 2, 4], [1, 2, 3, 3, 5]),
        (4, [0, 0, 0, 1, 1, 2], [1, 1, 1, 2, 2, 3]),
        (5, [0, 1, 2, 3, 4], [0, 2, 1, 4, 3]),
        (8, [0, 1, 2, 3, 4, 4, 5, 6], [1, 2, 1, 4, 3, 5, 6, 7]),
    ],
)
def test_adaptive_scc_matches_full_reference_partition_and_topology(
        num_nodes: int,
        sources: list[int],
        targets: list[int],
) -> None:
    """Match full-graph SCC membership and preserve condensation topology."""
    sources_array = np.asarray(sources, dtype=np.int64)
    targets_array = np.asarray(targets, dtype=np.int64)
    node_order, component_id, component_order, component_sizes, levels, timings = (
        strongly_connected_component_order(num_nodes, sources_array, targets_array)
    )

    graph = sparse.csr_matrix(
        (np.ones(sources_array.size), (sources_array, targets_array)),
        shape=(num_nodes, num_nodes),
    )
    _, reference_component_id = connected_components(
        graph,
        directed=True,
        connection="strong",
    )

    assert sorted(node_order.tolist()) == list(range(num_nodes))
    assert int(component_sizes.sum()) == num_nodes
    _assert_same_partition(component_id, reference_component_id)
    _assert_component_topology(
        sources_array,
        targets_array,
        component_id,
        component_order,
    )
    assert sum(levels.widths) == num_nodes
    assert set(timings) == {
        "csr",
        "scc",
        "dag",
        "topological_order",
        "level_diagnostics",
        "node_order",
    }
    assert all(value >= 0.0 for value in timings.values())


def test_algorithm_paths_and_cyclic_residual_counts() -> None:
    """Use the DAG bypass and trim both sides of a small cyclic core."""
    dag_sources = np.array([0, 0, 1, 2], dtype=np.int64)
    dag_targets = np.array([1, 2, 3, 3], dtype=np.int64)
    dag = _adaptive_component_order(5, dag_sources, dag_targets)
    assert dag[6:] == ("acyclic-fast", 0, 0)
    assert dag[5]["scc"] == 0.0
    assert dag[5]["dag"] == 0.0

    # 0 is an acyclic prefix, 1<->2 is the residual SCC, and 3->4 is a suffix.
    cyclic_sources = np.array([0, 1, 2, 2, 3], dtype=np.int64)
    cyclic_targets = np.array([1, 2, 1, 3, 4], dtype=np.int64)
    cyclic = _adaptive_component_order(6, cyclic_sources, cyclic_targets)
    component_id = cyclic[1]
    assert cyclic[6:] == ("cyclic-residual", 4, 2)
    assert component_id[1] == component_id[2]
    assert component_id[0] != component_id[1]
    assert component_id[3] != component_id[1]
    assert component_id[4] != component_id[1]


def test_self_loop_selects_cyclic_residual_path() -> None:
    """Treat a singleton self-loop as a cycle even though its SCC size is one."""
    result = _adaptive_component_order(
        3,
        np.array([0, 0, 1], dtype=np.int64),
        np.array([0, 1, 2], dtype=np.int64),
    )
    assert result[6:] == ("cyclic-residual", 2, 1)
    np.testing.assert_array_equal(result[3], np.ones(3, dtype=np.int64))


def _cyclic_upwind_mesh_and_flux():
    """Return a mesh-like object whose flux pairs contain a cycle and tails."""
    loc2glob_edge = np.array(
        [
            [0, 1, 4],
            [1, 0, 5],
            [2, 0, 6],
            [1, 3, 7],
        ],
        dtype=np.int64,
    )
    beta_dot_normal = np.zeros((4, 3, 2), dtype=np.float64)
    beta_dot_normal[:, 0, :] = -1.0
    beta_dot_normal[:, 1, :] = 1.0
    mesh = SimpleNamespace(num_tri=4, num_edg=8, loc2glob_edge=loc2glob_edge)
    return mesh, beta_dot_normal


def test_upwind_diagnostics_active_subset_and_flux_tolerance() -> None:
    """Report adaptive diagnostics and honor reduced edges and neutral fluxes."""
    mesh, beta_dot_normal = _cyclic_upwind_mesh_and_flux()
    ordering = upwind_scc_trace_ordering(mesh, beta_dot_normal, 3)
    diagnostics = ordering.diagnostics
    assert diagnostics.algorithm_path == "cyclic-residual"
    assert diagnostics.peeled_nodes == 6
    assert diagnostics.residual_nodes == 2
    assert diagnostics.cyclic_components == 1
    assert diagnostics.cyclic_nodes == 2
    assert set(ordering.edge_order) == set(range(mesh.num_edg))

    active_edges = np.array([0, 1, 3], dtype=np.int64)
    reduced = upwind_scc_trace_ordering(
        mesh,
        beta_dot_normal,
        2,
        active_edges=active_edges,
    )
    assert set(reduced.edge_order) == set(active_edges)
    assert sorted(reduced.dof_permutation.tolist()) == list(range(6))
    assert reduced.diagnostics.algorithm_path == "cyclic-residual"
    assert reduced.diagnostics.peeled_nodes == 1
    assert reduced.diagnostics.residual_nodes == 2

    tolerant_flux = beta_dot_normal.copy()
    tolerant_flux[0, 0, :] = -0.05
    tolerant_flux[0, 1, :] = 0.05
    tolerant = upwind_scc_trace_ordering(
        mesh,
        tolerant_flux,
        1,
        flux_tolerance=0.1,
    )
    assert tolerant.diagnostics.num_directed_edges == 3


def test_parallel_pair_arrays_and_permutations_are_thread_deterministic() -> None:
    """Keep pair arrays and permutations equal with 1, 2, and 40 threads."""
    numba = pytest.importorskip("numba")
    num_elements = 11_000
    loc2glob_edge = np.arange(num_elements * 3, dtype=np.int64).reshape(num_elements, 3)
    beta_dot_normal = np.empty((num_elements, 3, 2), dtype=np.float64)
    beta_dot_normal[:, 0, :] = -1.0
    beta_dot_normal[:, 1:, :] = 1.0
    mesh = SimpleNamespace(
        num_tri=num_elements,
        num_edg=num_elements * 3,
        loc2glob_edge=loc2glob_edge,
    )
    _, active_mask, old_to_active = _as_active_edges(mesh.num_edg, None)
    expected_sources = np.repeat(np.arange(0, mesh.num_edg, 3, dtype=np.int64), 2)
    expected_targets = np.column_stack(
        (
            np.arange(1, mesh.num_edg, 3, dtype=np.int64),
            np.arange(2, mesh.num_edg, 3, dtype=np.int64),
        )
    ).ravel()

    original_threads = numba.get_num_threads()
    configured_threads = int(numba.config.NUMBA_NUM_THREADS)
    thread_counts = sorted({1, 2, min(40, configured_threads)})
    observed = []
    try:
        for thread_count in thread_counts:
            numba.set_num_threads(thread_count)
            sources, targets = _build_upwind_edge_pairs_kernel(
                loc2glob_edge,
                beta_dot_normal,
                active_mask,
                old_to_active,
                0.0,
            )
            ordering = upwind_scc_trace_ordering(mesh, beta_dot_normal, 2)
            np.testing.assert_array_equal(sources, expected_sources)
            np.testing.assert_array_equal(targets, expected_targets)
            observed.append((sources.copy(), targets.copy(), ordering.edge_order.copy()))
    finally:
        numba.set_num_threads(original_threads)

    for sources, targets, edge_order in observed[1:]:
        np.testing.assert_array_equal(sources, observed[0][0])
        np.testing.assert_array_equal(targets, observed[0][1])
        np.testing.assert_array_equal(edge_order, observed[0][2])

