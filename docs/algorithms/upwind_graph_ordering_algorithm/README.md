# Upwind Graph Ordering Algorithm

This directory is the canonical home of the adaptive upwind graph ordering
used by `trace_ordering="upwind-scc"`.

- [`upwind_graph_ordering_algorithm.tex`](upwind_graph_ordering_algorithm.tex)
  derives the graph, acyclic fast path, cyclic-residual fallback, component
  ordering, diagnostics, and trace-DOF lift.
- [`UPDATE_LEDGER.md`](UPDATE_LEDGER.md) records the implementation changes and
  their performance and complexity consequences.

The generated `.aux`, `.log`, `.out`, `.toc`, and `.pdf` files that may be
present beside the source are local build artifacts. They are not canonical
sources and are not tracked.

The public option remains `upwind-scc`; the adaptive path is an implementation
detail reported through `GraphOrderingDiagnostics.algorithm_path`.
