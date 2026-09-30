# Upwind Graph Ordering Update Ledger

## Adaptive parallel implementation

- Parallel element-pair construction now uses deterministic count and fill
  passes. Each element owns a fixed output slice, so `prange` scheduling and
  the Numba worker count cannot change pair-array order.
- The initial graph representation is forward CSR only. Deterministic Kahn
  ordering simultaneously supplies the DAG order and detects cycles.
- Acyclic graphs use `algorithm_path="acyclic-fast"` and do not build reverse
  CSR, invoke Kosaraju, or construct a condensation graph.
- Cyclic graphs use `algorithm_path="cyclic-residual"`. Forward source peeling
  and reverse sink peeling remove acyclic prefixes, suffixes, disconnected
  nodes, and tails. Only the compact residual is passed to Kosaraju.
- Residual SCCs are recombined with peeled singleton components. The complete
  component graph is then topologically ordered, so dependencies crossing the
  trimmed/residual boundary remain valid.

## Deterministic ordering rules

- Element pairs are stored in element, source-face, target-face order.
- Kahn frontiers are emitted in ascending node order. Newly available nodes are
  sorted before the next frontier.
- Residual nodes are compacted in ascending original-node order.
- Recombined component ids are assigned by the smallest original node first.
- Nodes inside an SCC retain ascending original-node order.
- Duplicate edges remain legal and are counted consistently; singleton
  self-loops select the cyclic path and are reported as cyclic components.

These rules make pair arrays and final permutations invariant across repeated
runs and Numba thread counts.

## Parallel threshold and compilation cache

Pair counting and filling always use parallel element loops. Frontier copying
uses parallel workers only when the frontier-node plus adjacency-visit count is
at least 32,768, the conservative crossover selected on the 20-core reference
workstation. Indegree updates are serialized in deterministic buffer order and
do not require atomics.

Common DAG-path kernels and cyclic-only residual kernels use Numba's disk cache.
The reverse-graph, compaction, Kosaraju, recombination, and condensation work is
lazy: an acyclic run does not compile or load it. A second process should show
disk-cache loads for every kernel exercised by its selected path.

## Timings and complexity

The aggregate timing names `graph_pairs`, `csr`, `scc`, `dag`,
`topological_order`, `dof_permutation`, and `total` remain available. The
`topological_order` field now measures the actual successful Kahn pass rather
than also including diagnostics and node lifting. New `level_diagnostics` and
`node_order` fields report those compiled phases separately.

For a DAG, time and memory are linear in the active nodes and directed edges,
and only one CSR is built. For a cyclic graph, trimming is linear in the full
graph, while the two Kosaraju passes are linear in the residual node and edge
counts. Diagnostics expose `peeled_nodes` and `residual_nodes` so the saved work
is observable.

On the 92,552-triangle legacy `test2` preset with 40 Numba threads and
`OMP_PROC_BIND=false`, the observed warm ordering time after this change is
about 0.014--0.017 seconds. A fresh process that loads the six exercised
DAG-path kernels from disk cache takes about 0.039 seconds, and cache-debug
output confirms a data load for each kernel. A first call with a truly empty
Numba disk cache takes about 2.92 seconds and remains dominated by JIT
compilation; this does not meet the provisional 0.8-second empty-cache target
and is tracked separately from graph-execution performance.

For PyPardiso on the same preset, the median numerical solve time over three
warm adaptive-permutation runs was 0.5449 seconds, versus 0.5348 seconds for
the exact legacy permutation (ratio 1.019). This is within the five-percent
guardrail; both physical residuals were approximately `5.1e-16`.
