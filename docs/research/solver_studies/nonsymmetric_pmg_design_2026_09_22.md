# General nonsymmetric pMG–AMG design: algebraic evidence

The recommended generalization is **right FGMRES with face-block
Petrov–Galerkin pMG and a nonsymmetric AMG endpoint**. Local block/patch
solves and transfer construction use the actual condensed operator.
No velocity, diffusion tensor, or regime selector is required.

The nonsymmetric section of the [shared hp-AMG formalism](../../algorithms/hp_amg/hp_amg.tex)
specifies the nonsymmetric generalization and its limitations. This study records design evidence,
not production support or performance qualification. No HDG assembly,
simulation, time integration, GPU initialization, build, or TeX compilation
was run for this study.

## Why this generalization

| Requirement | Design mechanism | Evidence and limit |
|---|---|---|
| Use HDG block structure | Retain face blocks; eliminate high modal groups; solve face or multi-face patches; coarsen whole remaining face blocks spatially. | The local transfer equations and diagnostic operate on mode groups within faces. No GPU speedup is established. |
| Admit nonsymmetric systems | Independent restriction/prolongation, pivoted local solves, Petrov–Galerkin coarse operators, right FGMRES. | Exact block-elimination identities require invertibility, not SPD. Small tests include singular and indefinite symmetric parts. |
| Avoid PDE-regime selection | Build from matrix entries, face grouping, degree, and optionally connectivity. | The same transfer and relaxation policy is used in all synthetic cases. No material coefficients enter the algorithm. |

This is a proposed combination of established methods. It does not prove
mesh-independent convergence of the combined algorithm on arbitrary HDG
operators. Matrix-dependent setup still responds to strong and directed
couplings; avoiding a PDE label does not mean ignoring that information.

[Block local AIR](https://arxiv.org/abs/1708.06065) provides a relevant
nonsymmetric reduction framework. Its discussion of block treatment is
particularly useful: keeping the blocks and simply applying scalar AMG to
a block-scaled matrix need not give the same robustness.
[The HDG AIR study](https://arxiv.org/abs/2010.11130) supplies direct
trace-system precedent, within its space-time advection–diffusion scope.
[Constrained AIR](https://arxiv.org/abs/2307.00229) targets the difficulty
of covering both advective and diffusive regimes at controlled complexity.
It is the preferred longer-term AMG direction for this broader objective;
plain block AIR is a reference implementation candidate. None of these
papers establishes this repository's proposed polynomial reduction.

## Reproducible no-build diagnostic

From the repository root:

```sh
python -B scripts/dev/check_nonsymmetric_pmg_algebra.py
```

The [diagnostic source](../../../scripts/dev/check_nonsymmetric_pmg_algebra.py)
uses existing NumPy and SciPy only, without importing HYBRIDGE or optional
accelerator modules. [Saved results](nonsymmetric_pmg_design_2026_09_22.json)
include dependency versions, residuals, iteration counts, and level data.

The script checks ideal restriction/interpolation, the exact Schur product,
the exact inverse decomposition, coordinate mapping back to the original
system, and the distinction between approximate `RAP` and a naively
approximated Schur complement. The ideal identity errors were below
`1e-14`. The matrix

```text
A = [[ 0, 1],
     [-1, 2]]
```

has determinant one but a zero degree-zero principal block and a singular
symmetric part. Ideal transfers give coarse matrix `[0.5]` and an exact
inverse in the diagnostic. A second example has an indefinite symmetric
part and is also inverted correctly. Conversely, the invertible matrix
`[[1, 1], [1, 0]]` has a zero high-mode block: the proposed local solve
correctly raises a singular-factor error. A production implementation would
have to change or skip that p split, rather than promise unconditional setup.

The four larger synthetic matrices have twelve faces, four modes per face,
and 48 unknowns. The hierarchy reduces block sizes `4 → 2 → 1`. Both
comparators use two face-block Jacobi high-mode sweeps before and after
coarse correction. One uses injection; the other constructs independent
local transfers on one-ring patches. Both use **an exact dense terminal
solve, not AMG**. GMRES is applied to the explicit right-preconditioned
matrix `A B`, with restart 20 and relative tolerance `1e-10`.

| Synthetic matrix | Injection iterations | Local Petrov iterations | Local Petrov true relative residual |
|---|---:|---:|---:|
| Symmetric control | 6 | 6 | 2.97e-12 |
| Directed cycle | 14 | 14 | 1.00e-11 |
| Skew-coupled modes | 14 | 5 | 2.47e-12 |
| Unequal couplings | 23 | 19 | 6.35e-11 |

All eight solves passed their original-system residual checks. These results
support the algebra and illustrate that transfer changes can matter even
with the same outer method and smoother. Four tiny synthetic matrices
cannot establish HDG convergence rates, robustness to arbitrary anisotropy,
or comparative setup/solve cost. The dense materialization in this diagnostic
is an algebraic oracle, not an implementation strategy for large systems.

The test cycles are fixed linear operators, so ordinary right GMRES suffices.
The production design permits inner GMRES smoothing or changing cycles and
therefore calls for FGMRES. That flexible implementation, the inner-Krylov
smoother, sparse block transfer setup, and AIR endpoint were not exercised.
PyAMG was not installed in the inspected environment; no dependency was
installed for this diagnostic.

## Current implementation correspondence

| Existing component | Reuse or required change |
|---|---|
| [`face_hp_multigrid.py`](../../../hybridge/linalg/multigrid/face_hp.py) | Reuse hierarchy concepts and convergence accounting. Current diagonal symmetrization, Chebyshev bounds, PCGF and SPD-oriented coarse policy cannot define the new path. |
| [`legendre_face_bsr.py`](../../../hybridge/linalg/gpu/legendre_face_bsr.py) | Reuse modal transforms, orientation conventions, and face-BSR products. New transfers have graph couplings and cannot use only principal-block extraction. |
| [`additive_schwarz.py`](../../../hybridge/linalg/additive_schwarz.py) | Reuse BSR graph lookup and patch gathering. Add the chosen high-mode grouping, local factor policy, and overlap weighting to shared helpers when implementing. |
| [`cupy_gmres.py`](../../../hybridge/linalg/gpu/gmres.py) | The current implementation is left-preconditioned GMRES. Flexible right Arnoldi must store both residual basis vectors and their actual preconditioned corrections. |
| [`face_hp_krylov.py`](../../../hybridge/linalg/multigrid/krylov.py) | Its symmetric-part adapter is a comparator, not general nonsymmetric pMG. |
| AMG endpoint | Existing AMGX availability does not establish AIR or constrained AIR support. A backend and its block/device capabilities must be qualified. |

The smallest implementation comparator would retain polynomial injection,
replace the smoother with nonsymmetric block/patch solves, and use FGMRES
plus nonsymmetric coarse AMG. It is worth measuring, but the singular
principal-block example explains why it is not the complete general design.

Before production adoption, compare both hierarchies on assembled HDG
matrices across h/p refinement, with the same algebraic policy and residual
target. Record setup plus solve time, restart storage, factor/transfer/coarse
memory, original residuals, and failed splits. Include cyclic graphs,
strongly unequal couplings, orientation changes, singular/indefinite
symmetric parts, and relevant known-nullspace treatment. If low-mode
reduction proves unsuitable, hand the current face blocks to algebraic
coarsening earlier. Neither the literature nor these diagnostics justifies
forcing every problem through a scalar degree-zero hierarchy.
