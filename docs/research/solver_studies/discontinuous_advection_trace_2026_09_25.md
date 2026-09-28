# Discontinuous Advection Trace Rank Diagnosis

The singular fixture is caused by a double-outflow interior face under standard
sidewise upwind. The existing `conflict-averaged-upwind` policy repairs this
fixture in NumPy, Numba, CuPy, and raw CUDA. No additional numerical kernel or
change of default policy was needed for this investigation.

## Original Fixture And Cause

`tests/test_cupy_backend.py::_discontinuous_advection_fields` now imports the
unchanged coefficient fixture from
`scripts/advection_reaction/diagnose_discontinuous_trace.py`, so the GPU tests
and reproducible diagnostic share one definition. This is a synthetic
coefficient/boundary fixture; its supplied boundary function is not a known
manufactured solution for the independently supplied source.

The original failing setup uses four triangles from `rectangle_mesh(2, 1)` on
`[-1,1] x [0,1]`, p=3, and volume quadrature parameter 8. Its velocity is
`beta_x = where(x < 0, 2, -1) + 0.2*y`, `beta_y = 0.1 + 0.05*x`, and its
positive reaction is `2 + 0.01*x*y`. The jump lies on interior edge 4 at x=0.
The two outward normal velocities there are

```text
a = 2 + 0.2*y > 0
b = 1 - 0.2*y > 0
```

Both elements therefore have outflow. Standard upwind sets
`tau=abs(beta.n)` and `gamma=tau-beta.n=0` on both incidences. The local
trace-to-element coupling and the direct trace mass contribution vanish for
every trace mode on that edge. Its p+1 global columns are exactly zero,
although their rows are nonzero. Positive reaction keeps the element-local
operators invertible but cannot restore a missing trace coupling.

## Controlled Matrix Evidence

The [recorded host diagnostics](discontinuous_advection_trace_2026_09_25.json)
contain 32 p=3 matrices: both production trace bases, NumPy and Numba, both
exterior Dirichlet treatments, and four scenarios. Every case uses the
assembly-only API. Dense SVD is limited to these tiny diagnostic matrices.
Rows are equilibrated before global SVD because the legacy exterior penalty
is O(1e20); unscaled relative-rank estimates would incorrectly discard ordinary
interior singular values. Exact zero-column detection is independent of this
equilibration.

| Scenario | Eliminated matrix rank | Penalty matrix rank | Interpretation |
|---|---|---|---|
| Original fixture, standard upwind | 8 / 12 | 32 / 36 | Four missing face modes |
| Same fixture, reaction increased to 20 | 8 / 12 | 32 / 36 | Reaction does not repair the face |
| Same fixture, conflict-averaged upwind | 12 / 12 | 36 / 36 | Existing face policy restores coupling |
| Continuous `beta_x=1+0.2*y`, original reaction | 12 / 12 | 36 / 36 | Removing the converging jump removes this failure |

These ranks agree between NumPy and Numba and between nodal and modal traces.
For the original p=3 fixture, the smallest element-local singular-value ratio
is about 0.0225. The zero reduced columns are 4--7 with elimination and 16--19
with penalty boundaries. The failure is an interior face-flux degeneracy,
not a boundary elimination error or a disagreement between assembly backends.

Some existing solve tests instead use `rectangle_mesh(1, 1)` at p=2. There the
jump cuts element interiors and is projected into different polynomials; it
is not the aligned, double-outflow face case. Both trace bases produce a full
rank reduced matrix in that smaller configuration. Its solve evidence must
not be transferred to the four-triangle fixture.

## Existing Averaging Fix, Including Vectorized Paths

Enable `advection_stabilization="conflict-averaged-upwind"` explicitly.
NumPy and CuPy share `effective_advection_normal_flux(..., xp=np|cp)`:

1. Gather the two incidences of each interior face using cached connectivity.
2. Align their quadrature samples in global edge orientation.
3. At nodes where `a>=0`, `b>=0`, and `a+b>0`, replace the outward pair by
   `sL=(a-b)/2`, `sR=-sL`, using array masks and `where`.
4. Restore each incidence's local orientation and compute both `tau` and
   `gamma` from the effective normal velocity.

Because the outward normals oppose each other, `(a-b)/2` is the normal
component of the arithmetic mean of the two vector fields. The operation
changes only conflicting numerical face fluxes; volume coefficients and
ordinary inflow/outflow pairs remain unchanged. Numba's fused kernels already
implement the same rule in `_assemble_conflict_face_trace_weights`; raw CUDA
already implements it in its face-weight construction. This task adds targeted
evidence for those existing paths.

For the original seam, averaging gives `sL=0.5+0.2*y` and
`sR=-0.5-0.2*y`, providing full inflow support on one side. This repairs the
discretization's missing interface coupling; it does not establish an
H(div) reconstruction or an energy/stability guarantee.

## Validation And Limits

Host regressions cover p=1,2,3, both trace bases, both exterior boundary
treatments, matrix/RHS parity between NumPy and Numba, local invertibility,
zero-column provenance, the reaction/continuous-velocity controls, and the
unaligned two-triangle distinction. The host suite also guards the expected
nonuniqueness of a p=3 divergence-free tangent problem: with zero reaction
the constant trace is a null vector and the assembled matrix has rank 31 / 32
even with averaging; a reaction of 1 changes the bounded matrix rank to
32 / 32. These are assembly-only checks, consistent with the PDE requiring
a separate uniqueness condition for a solve.

On 2026-09-25 the host command below passed all 33 checks in 1.40 seconds
(23 numerical/bounds regressions and 10 test-manifest checks), with Numba JIT
disabled. The diagnostic command recorded the 32 p=3 host matrices linked above.

```bash
NUMBA_DISABLE_JIT=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q \
  tests/test_discontinuous_advection_trace.py tests/test_alpha_test_matrix.py

NUMBA_DISABLE_JIT=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m \
  scripts.advection_reaction.diagnose_discontinuous_trace --degrees 3 \
  --output docs/research/solver_studies/discontinuous_advection_trace_2026_09_25.json
```

The same 33 checks also passed with JIT enabled in 0.94 seconds on 2026-09-25,
using Numba 0.67.0 and 16 Numba threads. The test process verified that JIT
remained enabled and that both `assemble_projected_trace_system_kernel` and
`assemble_projected_trace_system_eliminated_kernel` had a compiled nopython
signature after the tests. This qualifies the compiled assembly paths for
the bounded cases above. Rerun the suite with:

```bash
NUMBA_DISABLE_JIT=0 NUMBA_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m pytest -q \
  tests/test_discontinuous_advection_trace.py tests/test_alpha_test_matrix.py
```

On 2026-09-25 the 12 GPU regressions below passed in 6.34 seconds with no
skips or warnings. They cover p=2,3, both trace bases, vectorized CuPy with
elimination/penalty boundaries, and fused cooperative raw CUDA with elimination.
Each compares standard and averaged matrix/RHS values with NumPy and checks
the expected rank. CuPy runtime JIT was explicitly approved; CPU Numba JIT
remained disabled.

```bash
NUMBA_DISABLE_JIT=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q \
  tests/test_discontinuous_advection_trace_cuda.py
```

All new numerical checks forbid a global solve and produce no reconstructed
solution. Host and device regressions are included in `host-fast` and
`gpu-smoke`, respectively. PDE-solve, convergence, performance, and
time-integration qualification remain outside these matrix checks.
The broader averaging-policy qualification remains in the
[boundary/stabilization contract](../../reference/advection_boundary_stabilization.md#conflict-averaged-upwind).
