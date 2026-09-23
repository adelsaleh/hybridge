# ADR GMRES implementation snapshot

This directory contains the alternate ADR assembler and solver implementation
used by the stationary ADR campaign and closed-loop stress runner. It was copied
from the local `hdgfem-gmres` working tree at base commit `d44acce` on
2026-09-23, including its uncommitted ADR changes. The copy is part of the
HDGFEM repository; a sibling checkout is no longer required for these runners.

The `hdgfem/` package is kept intact because its solver modules import their
own core, assembly, backend, kernel, and linear-algebra helpers. The `scripts/`
package contains the campaign workers and the documented ADR comparison and
oscillatory-study tools. `tests/test_adv_diff_rea.py` preserves the documented
small-matrix parity check.

Run alternate workers in a separate process with this directory first on
`sys.path`, as the campaign runners do. The native hp adapter selects the main
HDGFEM package explicitly. This keeps both implementations' imports isolated
and makes source snapshots and hashes reproducible from one checkout.

Historical `run_logs` and matrix caches are not included. Reports that cite
those files retain their original provenance paths.

Local compatibility changes make source paths relative to this checkout and
reuse the main package's basis-dispatcher fallback when `NUMBA_DISABLE_JIT=1`.
This permits import and small-matrix diagnostics without compilation.
