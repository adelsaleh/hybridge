# ASM–PP branch review and BSR preconditioner opportunities, September 2026

Assembling additive Schwarz once into a BSR correction is now implemented in
shared host helpers and a cache-only fully BSR AMG diagnostic. The completed
finest-level smoothing tests do **not** close the hybrid iteration gap. Wider
patches reduce some iteration counts but increase total solve time. No ASM
production preset is enabled. Native hp-BSR integration and improvements to
the coarse space remain separate, unvalidated proposals.

## September 14: completed fully BSR ASM diagnostic

The [recorded comparison](../../../artifacts/asm_bsr_convergence_20260914/README.md)
replays three captured p=6 guiding-center Poisson systems at each of 157,280
and 315,425 triangles. Candidates use the same captured matrix, RHS, guess,
physical acceptance threshold, and external PCGF implementation. ASM replaces
only finest-level smoothing; lower levels use balanced 2/2 Chebyshev smoothing.
The numerical correction and all retained full-BSR hierarchy operators remain
BSR. Patch construction uses exact cached mesh connectivity.

In addition to the branch's three-face patches, the diagnostic tests five-face
two-element patches and nine-face element-plus-neighbor patches. Their active
local matrices are exact global principal submatrices, with eliminated boundary
slots padded. Plain additive inverse contributions are assembled into C once;
its application is one BSR product. Constant-damping and fixed Chebyshev
polynomial schedules have symmetric pre/post composition around one fixed AMG
coarse correction. This is not the branch's harmonic-Ritz PP/GMRES algorithm.

| Triangles | Solver / finest smoother | Iterations, captured steps 1/2/3 | Warm solve ms |
|---:|---|---|---:|
| 157,280 | Hybrid control | 16/13/12 | 110.96 |
| 157,280 | Constant-vector full BSR control | 26/21/19 | 242.56 |
| 157,280 | Nine-face ASM, 3 pre + 3 post damped stages | 23/20/18 | 467.22 |
| 315,425 | Hybrid control | 16/13/12 | 210.21 |
| 315,425 | Constant-vector full BSR control | 27/22/19 | 454.61 |
| 315,425 | Nine-face ASM, 3 pre + 3 post damped stages | 24/20/18 | 895.74 |

Warm time is the median of captured steps 2 and 3, measuring the whole Python
PCGF call including preconditioning, vector copies, synchronization and final
true residual. It excludes setup and host physical validation, and is not a
native AMGX.solve timing. The complete report includes individual step times,
all smaller-patch screens, matched balanced-smoothing controls, and degree-two
and degree-three Chebyshev ASM screens at 157,280 triangles. None beats the
constant-vector full-BSR control in total solve time.

The nine-face C has about 5.4 times A's block count and occupies 2.51 GB / 5.04 GB
at these sizes. Recorded ASM setup (host construction and inverses plus C device
setup) is 34.50 s / 68.33 s; it is reusable across RHS vectors. Device memory
permits both cases, but the expanded inverse fill makes each smoothing product
more expensive. CPU setup is a prototype cost, not a tuned GPU setup result.

These results concern finest-level ASM with the recorded patches, damping and
polynomial schedules. They motivate measuring remaining smooth-error components
and their representation by P; they do not prove that interpolation alone causes
the gap, or rule out different ASM designs or ASM on coarse levels. The branch
review and proposals below are retained with that qualification.

## What was inspected

The requested branch is `origin/gpu_gmres_precondit`, local snapshot
`ea5ad26281f9e988194d9352399ccc4354a6633e`. No fetch or checkout was needed.
Exact source copies and hashes are in the
[branch manifest](../../../artifacts/gmres_bsr_inspiration_20260913/branch_manifest.json).
The current worktree was also inspected. Its `cupy_preconditionners.py` and
`cupy_polynomial.py` have the same syntax trees as that branch after removing
docstrings; the CUDA source strings and numerical application logic match.
[Comparison evidence](../../../artifacts/gmres_bsr_inspiration_20260913/review_evidence.json)
records the current hashes.

The branch's matrix format is fixed-width face-block storage:
`blocks[NF,S,q,q]`, `neighbors[NF,S]`, with `S=5` on ordinary triangular
interior-face rows and `q=p+1`. The device layout is
`matrix_batches[NF,q,S*q]`. This is block sparse, but it is not the compressed
`indptr/indices/data` BSR layout used by native hp-BSR. No scalar CSR conversion
is required by its operator, ASM, or PP application.

Relevant code:

- [Branch face operator](../../../artifacts/gmres_bsr_inspiration_20260913/branch_snapshot/hdgfem/backends/cupy_face_dense.py): fixed-width layout, raw and fused operator kernels.
- [Branch ASM](../../../artifacts/gmres_bsr_inspiration_20260913/branch_snapshot/hdgfem/backends/cupy_preconditionners.py): `_FUSED_ASM_KERNEL_SOURCE`, `build_face_additive_schwarz_incidence_slots`, and `CuPyFaceAdditiveSchwarzPreconditioner.apply_into`.
- [Local ASM matrices](../../../hdgfem/linalg/additive_schwarz.py): existing shared CPU formalism and inverse oracle.
- [Branch PP](../../../artifacts/gmres_bsr_inspiration_20260913/branch_snapshot/hdgfem/backends/cupy_polynomial.py): harmonic-Ritz setup, real recurrence, and reusable buffers.
- [Branch GMRES](../../../artifacts/gmres_bsr_inspiration_20260913/branch_snapshot/hdgfem/backends/cupy_gmres.py): left preconditioning, Arnoldi, and true residual recomputation.

## ASM and PP, as implemented

For element `e`, `R_e` selects its three face vectors. The local matrix `K_e`
has size `3q x 3q`: 15, 18, or 21 at p=4, 5, or 6. Its off-diagonal blocks
are the complete condensed elemental blocks; each diagonal is replaced by the
assembled global face diagonal. Thus the active part is the principal
submatrix `R_e A R_e^T` for the directly eliminated system. Eliminated local
faces use an identity pad with zero restriction and no prolongation.

The action is

\[
C r = \sum_e R_e^T K_e^{-1} R_e r.
\]

`application="fused"` uses two kernels, both with 256-thread CUDA blocks:

1. One thread owns one element-local scalar output row. It loops over all
   `3q` columns of the row-major cached inverse, gathers the global RHS through
   `element_system_faces`, and writes one local correction. There is no
   materialized restricted RHS.
2. One thread owns one global scalar output. The precomputed two-entry
   face-incidence table identifies its one or two local corrections. The
   thread sums them in a fixed order and writes the result. No atomic update
   or output zero-fill is needed.

This has fuller arithmetic lane use than the native smoother's one-warp-per-
face kernel with only `q` arithmetic lanes. However, adjacent ASM threads load
inverse rows with stride `3q` at each column; fuller lane use is not proof of
better memory throughput. The historical high-order results actually favored
the separate raw ASM path over this fused path.

PP wraps ASM as

\[
H_d = p_{d-1}(CA) C,
\qquad
p_{d-1}(B)=\sum_{j=1}^{d}\frac{1}{\theta_j}
\prod_{k<j}\left(I-\frac{B}{\theta_k}\right).
\]

One setup Arnoldi cycle estimates harmonic Ritz roots; Leja ordering keeps
complex-conjugate roots adjacent. Large vectors stay on the GPU, while the
small Hessenberg problem is handled on the CPU. Application reuses four work
vectors and fused scalar-vector update kernels. GMRES then applies `H_d A`
in Arnoldi and `H_d r` to the restart residual. The stopping decision uses the
unpreconditioned true residual at restart/termination boundaries.

This PP is a polynomial in the entire preconditioned operator. Its device
vector updates are scalar elementwise operations; the block structure is
exploited by `A` and `C` underneath them.

There is a precise application optimization: the last root does not need the
updated residual-like work vector `q`. For a real last root, only
`z += q/theta` is needed. For a conjugate last pair, the output needs `Bq`
but not `B^2q`. The current implementation computes the unused update anyway.
Removing it changes the per-application count from `d A + (d+1) C` to
`(d-1) A + d C`, with the same polynomial output. Degree 18 saves one A and one
ASM application out of 37 such applications; degree 2 saves two out of five.
These are operation counts, not measured speedups. New terminal vector kernels
and updated counters/tests would be needed in a production patch.

## Assemble ASM into BSR once

The matrix itself can be cached:

\[
C_{fg} = \sum_{e:\,f,g\in e}
             (K_e^{-1})_{\ell_e(f),\ell_e(g)}.
\]

Each pair of faces of one triangle is already coupled in the HDG block graph.
The inverse of its three-face patch therefore introduces no additional block
edges beyond that structural graph. Summing the two diagonal contributions
and retaining each off-diagonal block gives the exact same ASM operator, up
to floating-point summation order. Numerical zero blocks must not be dropped
from the original structural stencil when relying on this property.

A future setup would:

1. Obtain `element_system_faces` from existing mesh and free-face maps.
2. Gather the `3q x 3q` principal matrices from the already assembled,
   orthonormal native BSR operator. This avoids reassembling the PDE or
   importing the branch's legacy trace basis.
3. Reuse the shared batched factorization/inversion machinery in
   [cublas_batched.py](../../../hdgfem/backends/cublas_batched.py), with local
   inverse and SPD checks. Batch the setup to control its temporary memory.
4. Assemble the inverse blocks into the existing BSR pattern, using the
   existing face incidence/assembly formalism. A face-row owner can accumulate
   its one or two patch contributions without atomics.
5. Release patch inverses after construction unless retained for diagnostics.
   Use the generic cuSPARSE BSR operator for `C.apply_into(r,z)`.

The shared CPU `assemble_global_face_blocks` already performs the required
algebra. The diagnostic below reuses it. A production device implementation
belongs with the package's preconditioner/BSR helpers. Native solver setup in
`diffusion_reaction.py` currently supplies the matrix but no element-patch map;
that map must be passed through explicitly. The device operator protocol also
needs a small adapter: native `matvec(x,out=...)` versus GMRES
`matvec_into(x,out)` / preconditioner `apply_into(x,out)`.

For the measured 315,425-triangle p=6 graph:

| Storage, FP64 | Decimal MB |
|---|---:|
| Uniform cached local inverses, `NE*(3q)^2` | 1,112.819 |
| Assembled BSR C values, `nnzb*q^2` | 925.040 |
| Branch fused ASM local output workspace | 52.991 |
| Existing native block-Jacobi inverse values | 185.213 |

Thus assembly removes 16.87% of the uniform patch-inverse coefficient storage
and the local correction buffer. C can share A's block indices. C is still
much larger and more expensive to apply than a block-diagonal inverse.
The setup peak includes both input factors and output storage until released;
this table is not a measured peak-memory result.

The no-new-block-edge argument is specific to these element patches. An
arbitrary algebraic overlap patch on an AMG coarse graph can introduce fill.
Also, `CA` and higher powers generally fill beyond A's pattern: keep A and C
separate rather than assembling the complete polynomial operator.

For an SPD, directly eliminated Poisson matrix, each active patch is SPD and
`C` is SPD when patches cover all unknowns. Preserve the plain additive sum;
one-sided overlap weighting or a restricted ASM variant need not preserve
symmetry. Build C in native orthonormal coordinates. If converting an existing
matrix instead, `Ahat=S A S` requires `Chat=S^-1 C S^-1`.

## How it could improve native hp-BSR

Current Chebyshev-2 pre/post smoothing has four block-Jacobi stages per V-cycle:
one zero-start diagonal action and three full fused stages. The large p=6
profile measured 4.4166 ms per cycle for these four stages, comprising
87.297 ms full stages plus 5.452 ms zero-start work over 21 cycles.
[Recorded timing evidence](../../../artifacts/native_hp_bsr_breakdown_20260913/README.md)
puts the full stages at 56.8% of native GPU time and 43.7% of the complete
profiled Poisson call. Actual p=0 AMGX kernels are only 8.05% of native GPU time.

ASM couples all three faces of an element, whereas block Jacobi couples only
the modes of one face. The useful experiment is whether that stronger local
correction permits one fixed damped stage before and after the same p=0
correction, or reduces outer iterations enough to justify more costly stages.
Copying the four-stage schedule with C substituted for D^-1 can increase work.

For one pre-stage from zero and one post-stage, the smoothing cost is
approximately `2*t_C + t_A + vector_updates`; the A residual for coarse
restriction is additional and common to both designs. The existing generic
BSR A product measured 0.8247 ms on this case, but C has not been timed.
Select by complete native PCG time at the same assembly-basis true residual,
including iteration changes and setup amortization over the expected RHS count.

A fixed damped ASM stage uses `x <- x + omega*C*(b-A*x)`. Use a validated
spectral upper bound for CA; do not reuse the D^-1 A Chebyshev interval.
With `0 < omega < 2/lambda_max(CA)`, symmetric pre/post pairing and an SPD
coarse correction preserve the standard SPD two-level construction. Arbitrary
harmonic-Ritz PP roots do not by themselves prove positivity over the spectrum;
the branch's GMRES PP should not be copied directly into PCG without that check.

Historical evidence argues against replacing multigrid with high-degree PP.
The [primitive study](face_dense_primitives_2026_08.md), on a Quadro RTX 6000
and the legacy trace basis, measured p=6 degree-18 PP at 94.5% of recorded GPU
operation time. The best tuned degree on that 32,449-triangle case still took
1.026 s versus 0.094 s for AMGX. These are dated, different-hardware results,
not predictions for the current Blackwell/modal solver.

## A fully BSR AMG design that uses these ideas

Here, fully BSR means all retained numerical sparse A, C, P, R and Galerkin
operators use dense blocks, with no coefficient-expanded scalar CSR hierarchy.
A scalar adjacency/strength graph with one vertex per block is still useful
for graph decisions. A small cached dense terminal solve is also compatible
with that design. Relabeling the present scalar p=0 solve as 1x1 BSR would not
provide the requested additional block computation.

The local AMGX source already implements `block_graph_identity` and
`block_graph_dense` in `src/classical/classical_amg_level.cu` and
`src/classical/block_graph.cu`. Dense mode retains q-by-q transfer blocks,
exact `R=P^T`, and dense BSR Galerkin products. Its current interpolation starts
from `w_ic I_q`, performs block-Jacobi improvement on fixed D2 support, and
corrects each row to satisfy `sum_c P_ic=I_q`.

The [existing large-mesh study](classical_amg_bsr_2026_08.md) and its
[raw comparison rows](classical_amg_dense_bsr_three_way_samples_2026_08.csv)
show the limitation. At 152,909 triangles and p=6, dense pure BSR took
53 iterations / 1.231 s versus the hybrid's 22 / 0.277 s. Identity lifting was
much weaker. Support size, damping, repeated interpolation smoothing, scalar
strength variants, and Extended+i experiments did not close the gap. A new
proposal must change what the coarse space represents or lower actual cycle
cost; merely storing every level in BSR is insufficient.

A concrete next design is:

1. Keep the face-block graph and dense q-by-q transfers to reuse the existing
   AMGX implementation. Use exact block injection at coarse roots, ensuring
   full column rank of P. Begin with the existing nonaggressive support and
   enlarge it locally only when required to represent the chosen candidates.
2. Obtain a few actual smooth-error candidates Z using fixed ASM relaxation
   of random vectors, optionally seeded by projected physical traces such as
   1, x, and y. Reuse the trace projection, orientation, boundary elimination,
   and orthonormal scaling helpers. The q entries of a face are polynomial
   coefficients, not q independent physical fields. Constant coefficients in
   every modal component are therefore not a sufficient model of smooth HDG
   error. Physical traces are candidate seeds, not exact Dirichlet null modes.
3. Construct block-valued P subject to `P Z_c = Z` and root injection. Every
   retained row support needs sufficient candidate rank; test rank and add
   support or change coarse selection when it cannot satisfy the constraints.
   Apply projected energy minimization, using the assembled C as a possible
   setup preconditioner. An unprojected trial step is
   `P <- P - tau*C*(A*P)`; sparsity and candidate constraints require explicit
   projection, and a descent/line-search check is needed after projection.
   This differs from repeating the previously unsuccessful D^-1 step with an
   identity row-sum constraint. It remains a hypothesis to test.
4. Form `R=P^T` and `A_c=P^T A P` with dense blocks. Cache symbolic products,
   factors, SpMV descriptors, and workspaces. Track block nonzeros and bytes
   on every level; interpolation fill directly increases transfer and coarse
   operator cost. Never assume the scalar symbolic graph is the numerical
   coarse operator.
5. Recurse using the same block machinery. Use ASM where its reduction in
   iterations pays for its stencil, and fixed block-Jacobi/Chebyshev smoothing
   on deeper levels when cheaper. Coarse algebraic patches need their own
   incidence lists and symbolic fill; the two-element face table does not
   generalize unchanged. End at a small cached dense factorization and pair
   all pre/post smoothing symmetrically for PCG.

The general constrained interpolation principle is established in
[Olson, Schroder and Tuminaro's energy-minimization paper](https://www.unm.edu/~jbschroder/docs/OlSc2011.pdf):
minimize coarse-basis energy while preserving selected modes and a chosen
sparsity pattern. [PyAMG's reference implementation](https://pyamg.readthedocs.io/en/latest/generated/pyamg.aggregation.html#pyamg.aggregation.energy_prolongation_smoother)
accepts BSR operators and BSR tentative interpolation, with explicit fine and
coarse candidates. These references support the algorithmic direction, not
the performance of the proposed HDGFEM CUDA implementation.

There is also a concrete setup-kernel opportunity in the existing AMGX code:
`dense_bsr_galerkin_kernel` loops over each coarse output block and repeatedly
searches fine R/A/P paths, with shared-memory barriers for small dense products.
A cached symbolic map plus two staged block products `T=A*P`, `A_c=R*T` can
avoid repeated searches, at the cost of T's memory. This affects hierarchy
setup; it does not accelerate an already cached V-cycle by itself.

## Evidence produced in this review

The [CPU diagnostic](../../../artifacts/gmres_bsr_inspiration_20260913/check_algebra.py)
uses the existing face topology, global block assembler, boundary elimination,
ASM builder and polynomial reference. It ran 20 ASM cases on 12- and
24-triangle connectivity, q=2,3,5,6,7, with all faces active or Dirichlet faces
eliminated and permuted free-face numbering. Matrices range from 26 to 301
scalar unknowns. Inputs are synthetic SPD element matrices, not a PDE solve.

[Results](../../../artifacts/gmres_bsr_inspiration_20260913/algebra_checks.json):

- The assembled C has exactly A's BSR block pattern in all cases.
- Largest relative difference from the existing ASM action: `2.7053e-16`.
- Largest C symmetry defect: `7.3512e-17`; every tested C is positive definite.
- An explicitly constructed, fixed damped ASM two-level action is symmetric
  positive definite and agrees with its staged application to `3.8557e-16`.
- All 40 PP cases, including real and conjugate terminal roots and with/without
  ASM, remove exactly one effective operator application. Output differences
  from the existing recurrence are zero in these CPU cases; agreement with an
  independent complex recurrence is within `6.4456e-16`.

Reproduce from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m artifacts.gmres_bsr_inspiration_20260913.check_algebra
```

The script guards Numba compilation and invokes no CUDA kernel. No build,
compilation, GPU performance run, or time integration was performed for this
review. The small matrices prove the algebraic reuse opportunities; they do
not validate production HDG convergence, GPU kernel correctness, or speed.

Recommended implementation order is assembled BSR ASM, then a controlled
native hp smoother comparison, then candidate-constrained dense-block AMG.
The PP terminal shortcut is an independent small improvement for users of the
existing GMRES/PP solver. Production validation should retain the original A,
use the same true-residual contract and matched RHS/initial guesses at p=4–6,
and compare setup, whole-PCG wall time, per-level CUDA work, and peak memory.
