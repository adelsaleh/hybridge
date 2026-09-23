# PyPardiso factor diagnostics

`hdgfem.linalg.pardiso_diagnostics.pardiso_factor_statistics(solver, matrix_nnz=...)`
reads an already active PyPardiso solver. It neither imports the optional backend
nor changes its parameters. Call it after factorization and before releasing
the factors. It is CPU-only because PyPardiso is a host direct solver; it does
not transfer device matrices implicitly. Use the existing `face_dense_to_bsr`
conversion and `solve_pypardiso_system` wrapper for HDGFEM trace systems.

The package also exposes `prepare_pypardiso_spd_matrix(full_matrix)`: it checks
real finite square input and symmetry, then returns upper-triangular CSR for
`PyPardisoSolver(mtype=2)`. The ordinary `matrix_type="spd"` solve wrapper uses
the same preparation. Explicit factorization benchmarks may retain this
prepared storage and the Cholesky factors for repeated RHS solves. Include
symmetry checking/triangle conversion in setup timing, validate solutions
against the **original full matrix**, and report factor fill relative to the
actual upper-triangle input count. Symmetry alone does not establish positive
definiteness; native factorization must succeed as well.

For an approximately symmetric operator, `refine_host_linear_solution` can
apply bounded correction solves with caller-owned factors, using the residual
of the original full matrix. It returns a new solution and the correction
count, without changing the matrix or input solution. This is different from
MKL's internal refinement against its upper-triangle interpretation. Such a
run must be labelled **Cholesky plus original-system correction**, with all
correction solves and residual evaluations included in its timing; the caller
still verifies the final residual independently.

PyPardiso's `get_iparm` is **one-based**. The returned dictionary distinguishes:

| Field | PyPardiso iparm | Meaning |
| --- | --- | --- |
| `symbolic_peak_kib` | 15 | Peak symbolic-phase memory estimate |
| `symbolic_permanent_kib` | 16 | Symbolic data retained for subsequent phases |
| `numerical_factors_kib` | 17 | Internal numerical factor/solve storage estimate |
| `factor_nnz` | 18 | Reported factor nonzeros, or null if unavailable |
| `perturbed_pivots` | 14 | Number of perturbed pivots |
| `iterative_refinement_steps` | 7 | Refinement count from the last solve |

For in-core mode, `estimated_solver_peak_kib` is
`max(iparm(15), iparm(16) + iparm(17))`; it is null for out-of-core mode.
These definitions follow the [Intel oneMKL PARDISO reference](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2026-0/pardiso-iparm-parameter.html).
The C documentation uses indices one smaller. Values are reported in KiB,
as in Linux process diagnostics. `fill_ratio` divides the reported factor
nonzeros by the input scalar CSR nonzeros after removing explicit zeros.
Counters that are nonpositive are not interpreted as a valid factor size.

These figures are **not process RSS**. Python arrays, the face cache mapping,
CSR storage, PyPardiso's cached matrix copy/hash, temporary validation arrays,
and allocator-retained memory are additional costs. `wrapper_matrix_copy_bytes`
records the stored CSR arrays when PyPardiso keeps a copy; large matrices can
instead use a hash, recorded by `wrapper_uses_matrix_hash`.

The ADR direct-solver campaign records Linux RSS/high-water marks independently,
including snapshots immediately before and after factorization and solves.
The process high-water mark is cumulative, includes validation, and does not
reset between trials; it must not be labelled as exact LU-only allocated bytes.
The internal estimate and the observed process peak answer different questions.

See [the full ADR campaign protocol](../research/solver_studies/adr_pardiso_campaign.md).
