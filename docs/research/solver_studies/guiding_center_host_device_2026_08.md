# Guiding-Center Host/Device Solver Study, 2026-08-07

This report records bounded findings for the Gaussian-annulus guiding-center
case with `k=3`, `sigma=0.03`, `eps=0.05`, polynomial order 6, and
`dt=0.1`. It is historical evidence, not a claim about later module versions
or a production-preset recommendation.

The exact source, configuration, binary, and result-artifact hashes used here
are recorded in
[`guiding_center_host_device_2026_08.provenance.json`](guiding_center_host_device_2026_08.provenance.json).
The HYBRIDGE and AMGX worktrees were dirty, so the Git commits alone do not
identify the tested code. The SHA-256 entries in that manifest are therefore
part of the result identity.

## Scope And Meshes

Two related but non-interchangeable data sets are discussed:

1. The 100k PARDISO/device comparison and the strict AMGX GMRES screens used
   `mesh_size=0.008`, exactly **113,894 triangles**.
2. The time-stage SciPy ILU study used `mesh_size=0.014` and
   `minimum_triangles=30,000`. Its archived CSV/JSONL results do not retain
   the exact triangle count. Its timings must not be directly ranked against
   the 113,894-triangle runs.

The stage labels in the SciPy study are:

| Stage | Representative step/time | Interpretation |
|---|---:|---|
| early | step 10, `t=1` | predominantly linear phase |
| growth | step 200, `t=20` | developed instability growth |
| nonlinear onset | step 300, `t=30` | onset of nonlinear behavior |
| nonlinear | steps 450--500, `t=45--50` | late nonlinear phase |

## SciPy ILU And Upwind-SCC

No SciPy direct solver was allowed in this study. The tested SciPy transport
paths used BiCGSTAB with ILU, including NATURAL and COLAMD permutations and
unordered and upwind-SCC trace layouts.

| Stage | Best unordered/reused ILU | Fresh upwind-SCC/NATURAL ILU | Device reference |
|---|---:|---:|---:|
| early | 4.660 s (COLAMD medium) | 6.633 s | 0.148 s |
| nonlinear onset | 6.015 s (COLAMD medium) | 6.670 s | 0.151 s |
| nonlinear | 6.374 s (COLAMD medium) | 6.481 s | 0.151 s |

The upwind-SCC package was clearly slower early and approximately tied with
the best host path late. This is not a pure one-variable ordering comparison:
the best unordered result reused its first ILU, while the upwind path rebuilt
its ILU. The observed physics graph also collapsed to one dominant SCC, so the
ordering exposed no useful condensation-DAG depth. The evidence does not
support promoting upwind-SCC for this case, particularly not as a substitute
for preconditioner reuse.

## PARDISO Versus Device At 113,894 Triangles

The matched three-step artifacts give the following approximate steady-step
timings:

| Phase | PARDISO host | Raw-CUDA/AMGX device | Device advantage |
|---|---:|---:|---:|
| Poisson total | 6.62 s | 0.92 s | 7.2x |
| Poisson solve only | 3.68 s | 0.281 s | 13.1x |
| Transport total | 11.76 s | 0.431 s | 27.3x |
| Transport solve only | 9.17 s | 0.098 s | about 93x |

This does **not** establish whether PARDISO is as fast as SciPy+ILU. The
available SciPy stage results use the smaller `mesh_size=0.014` problem, and
the completed stage artifact contains no matched PARDISO candidate. A bounded
100k matched SciPy/PARDISO screen is still required for that conclusion.

### PARDISO Symmetry And Residual

Poisson used `pypardiso-spd`. The adapter:

- validates symmetry before the solve;
- passes only the upper triangle;
- selects PARDISO `mtype=2` for a real SPD matrix.

Transport used the nonsymmetric PARDISO path, `mtype=11`.

The 100k Poisson result had an absolute residual near `2.72e-13` and RHS norm
near `8.65e-4`, hence a relative residual near `3.14e-10`. The run was
accepted because its stopping contract used absolute tolerance `1e-12`.
A direct factorization does not promise arbitrary `||Ax-b||/||b||`: finite
precision, scaling, conditioning, and cancellation still control the observed
residual. Symmetric scaling and/or explicit post-factorization iterative
refinement should be evaluated before claiming strict relative `1e-12`.

## Strict AMGX GMRES Screen

The AMGX branch adds opt-in explicit residual verification, cycle-local
restarts, DGKS/ALWAYS reorthogonalization, and safe happy-breakdown handling.
Legacy defaults remain unchanged, and AMGX's global `Epsilon_conv` was not
modified.

| GMRES configuration | Iterations | Solve time | True relative residual |
|---|---:|---:|---:|
| restart 50, DGKS | 500 | 13.760 s | `1.158e-11` |
| restart 20, DGKS | 200 | 5.250 s | `1.169e-11` |
| restart 50, ALWAYS | 200 | 5.305 s | `1.183e-11` |

The patched solver correctly rejected the earlier estimated-residual false
convergence, but all variants stalled around `1.2e-11`. Reorthogonalization
did not remove the floor. None met the `1e-12` accuracy gate or the speed
gate, so no strict three-step preset was promoted and the planned three-step
run was skipped.

## Conclusions Bound To This Version

- Raw-CUDA/AMGX is decisively faster than PARDISO for both operators on the
  recorded 113,894-triangle three-step run.
- The historical PARDISO and production AMGX Poisson results do not satisfy a
  strict RHS-relative `1e-12` requirement, even though their absolute
  residuals met the former acceptance contract.
- Upwind-SCC did not improve the tested SciPy ILU transport path. Its early
  result was worse, and its nonlinear result was only approximately tied.
- No matched-size evidence yet shows PARDISO and SciPy+ILU to be equally fast.
- PETSc remains a separate future study and is not part of these results.

These conclusions must be requalified if any hashed module, configuration,
AMGX source file, AMGX binary, or result artifact in the provenance manifest
changes.
